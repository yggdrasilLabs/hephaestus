"""Tests for decoder-only (text generation) exports in forge.converter."""

from __future__ import annotations

import os

import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from forge.converter import ConversionError, _decoder_only_dummy_inputs, validate_model
from tests.test_converter import _make_minimal_onnx, _write_json


def _make_decoder_only_graph(path: str, with_position_ids: bool = True) -> None:
    """Write a tiny graph with the inputs of a decoder-only export at *path*.

    Like a real causal language model, it only runs when its inputs agree
    with each other: positions are looked up in a table as long as the
    attention mask, so a position outside the mask is an error, and the
    new tokens are appended to the cache, so the mask must be as long as
    cache plus new tokens.
    """
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axes13"], ["ids_4d"]),
        helper.make_node("Mul", ["ids_4d", "ones"], ["new_key"]),
        helper.make_node(
            "Concat", ["past_key_values.0.key", "new_key"], ["present.0.key"], axis=2
        ),
        # (batch, total): one entry per cached or new token.
        helper.make_node("Gather", ["present.0.key", "zero"], ["key_h"], axis=1),
        helper.make_node("Gather", ["key_h", "zero"], ["history"], axis=2),
        # Fails unless the mask is exactly as long as the history.
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["history", "mask_f"], ["masked"]),
        helper.make_node("ReduceSum", ["masked"], ["total"], keepdims=0),
    ]
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch", "past_seq + seq"]
        ),
    ]
    logits_terms = ["total"]
    if with_position_ids:
        nodes += [
            # A table with one row per mask position, as rotary and
            # learned position embeddings are built.
            helper.make_node("Shape", ["attention_mask"], ["mask_shape"]),
            helper.make_node("Gather", ["mask_shape", "one"], ["mask_len"], axis=0),
            helper.make_node("Range", ["zero", "mask_len", "one"], ["table"]),
            helper.make_node("Gather", ["table", "position_ids"], ["looked_up"], axis=0),
            helper.make_node("Cast", ["looked_up"], ["looked_up_f"], to=TensorProto.FLOAT),
            helper.make_node("ReduceSum", ["looked_up_f"], ["position_total"], keepdims=0),
        ]
        inputs.append(
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "seq"])
        )
        logits_terms.append("position_total")
    nodes.append(helper.make_node("Sum", logits_terms, ["logits"]))
    inputs.append(
        helper.make_tensor_value_info(
            "past_key_values.0.key", TensorProto.FLOAT, ["batch", 2, "past_seq", 4]
        )
    )

    graph = helper.make_graph(
        nodes,
        "decoder_only",
        inputs,
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, []),
            helper.make_tensor_value_info(
                "present.0.key", TensorProto.FLOAT, ["batch", 2, "past_seq + seq", 4]
            ),
        ],
        initializer=[
            helper.make_tensor("axes13", TensorProto.INT64, [2], [1, 3]),
            helper.make_tensor("zero", TensorProto.INT64, [], [0]),
            helper.make_tensor("one", TensorProto.INT64, [], [1]),
            helper.make_tensor("ones", TensorProto.FLOAT, [1, 2, 1, 4], [1.0] * 8),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _write_artifacts(output_dir: str) -> None:
    _write_json(os.path.join(output_dir, "tokenizer.json"), {"type": "test"})
    _write_json(
        os.path.join(output_dir, "config.json"),
        {"architectures": ["LlamaForCausalLM"]},
    )


class TestValidateDecoderOnlyExport:
    """validate_model accepts the model.onnx of a causal language model."""

    @pytest.mark.parametrize("with_position_ids", [True, False])
    def test_decoder_only_export_is_valid(
        self, tmp_path: str, with_position_ids: bool
    ) -> None:
        output_dir = str(tmp_path)
        _make_decoder_only_graph(
            os.path.join(output_dir, "model.onnx"), with_position_ids=with_position_ids
        )
        _write_artifacts(output_dir)

        validate_model(output_dir)

    def test_all_ones_feed_is_what_the_graph_rejects(self, tmp_path: str) -> None:
        """The graph is a real check: the feed used for every other model fails on it."""
        import numpy as np

        path = os.path.join(str(tmp_path), "model.onnx")
        _make_decoder_only_graph(path)
        session = ort.InferenceSession(path)
        all_ones = {
            inp.name: np.ones(
                [1 if isinstance(d, str) or d is None else d for d in inp.shape],
                dtype=np.int64 if inp.type == "tensor(int64)" else np.float32,
            )
            for inp in session.get_inputs()
        }

        with pytest.raises(Exception):
            session.run(None, all_ones)

    def test_first_step_feed_has_an_empty_cache(self, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "model.onnx")
        _make_decoder_only_graph(path)

        feeds = _decoder_only_dummy_inputs(ort.InferenceSession(path))

        assert feeds is not None
        assert feeds["past_key_values.0.key"].shape == (1, 2, 0, 4)
        assert feeds["input_ids"].shape == (1, 1)
        assert feeds["attention_mask"].tolist() == [[1]]
        assert feeds["position_ids"].tolist() == [[0]]

    def test_other_graphs_keep_their_feed(self, tmp_path: str) -> None:
        """A graph without cache inputs is not treated as decoder-only."""
        _make_minimal_onnx(str(tmp_path))
        path = os.path.join(str(tmp_path), "model.onnx")

        assert _decoder_only_dummy_inputs(ort.InferenceSession(path)) is None

    def test_missing_tokenizer_is_still_an_error(self, tmp_path: str) -> None:
        output_dir = str(tmp_path)
        _make_decoder_only_graph(os.path.join(output_dir, "model.onnx"))
        _write_json(
            os.path.join(output_dir, "config.json"),
            {"architectures": ["LlamaForCausalLM"]},
        )

        with pytest.raises(ConversionError, match="tokenizer.json"):
            validate_model(output_dir)
