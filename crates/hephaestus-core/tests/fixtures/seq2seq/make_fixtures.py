"""Regenerate the tiny seq2seq test fixtures in this directory.

Run with any Python that has ``onnx`` and ``tokenizers`` installed, for
example Forge's environment::

    cd forge && uv run python ../crates/hephaestus-core/tests/fixtures/seq2seq/make_fixtures.py

The graphs hold no weights. Each is a few integer operations, so the
expected output of every test can be worked out by hand:

``encoder_model.onnx``
    ``last_hidden_state[b, i, 0] = float(input_ids[b, i] * attention_mask[b, i])``.

``decoder_model.onnx``
    Echoes the encoder input: given ``t`` decoder tokens, the logits at
    decoder position ``j`` are one-hot at encoder token ``j + 1`` (clamped
    to the last encoder token). Greedy decoding therefore reproduces the
    input, and stops when it reaches the input's closing ``</s>``.

``decoder_model_merged.onnx``
    The same echo, plus the inputs of an Optimum merged decoder:
    ``use_cache_branch`` and one ``past_key_values`` tensor. Feeding
    ``use_cache_branch = true`` or a non-zero cache shifts every token by
    one, so a wrong feed fails the tests.

``model.onnx``
    A fused single-pass model: ``output_ids = input_ids * attention_mask + 1``.
    Its output differs from the echo, so tests can tell which mode ran.

``tokenizer.json``
    Word-level tokenizer: ``<s>``=0, ``<pad>``=1, ``</s>``=2, ``<unk>``=3,
    then the words ``a`` .. ``l`` = 4 .. 15, wrapped as ``<s> ... </s>``.
"""

from __future__ import annotations

import pathlib

import onnx
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors

HERE = pathlib.Path(__file__).resolve().parent
VOCAB_SIZE = 16
OPSET = 14
IR_VERSION = 8


def _save(graph: onnx.GraphProto, name: str) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(HERE / name))


def make_encoder() -> None:
    nodes = [
        helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
        helper.make_node("Cast", ["masked"], ["masked_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["masked_f", "axis2"], ["last_hidden_state"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_encoder",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["batch", "seq", 1])],
        initializer=[helper.make_tensor("axis2", TensorProto.INT64, [1], [2])],
    )
    _save(graph, "encoder_model.onnx")


def _echo_nodes() -> list[onnx.NodeProto]:
    """Nodes computing ``echo[b, j] = encoder token j + 1`` for each decoder position."""
    return [
        helper.make_node("Squeeze", ["encoder_hidden_states", "axis2"], ["enc_f"]),
        helper.make_node("Cast", ["enc_f"], ["enc_i"], to=TensorProto.INT64),
        helper.make_node("Mul", ["enc_i", "encoder_attention_mask"], ["enc"]),
        helper.make_node("Shape", ["input_ids"], ["dec_shape"]),
        helper.make_node("Gather", ["dec_shape", "one"], ["dec_len"], axis=0),
        helper.make_node("Shape", ["enc"], ["enc_shape"]),
        helper.make_node("Gather", ["enc_shape", "one"], ["enc_len"], axis=0),
        helper.make_node("Add", ["dec_len", "one"], ["limit"]),
        helper.make_node("Range", ["one", "limit", "one"], ["positions"]),
        helper.make_node("Sub", ["enc_len", "one"], ["last"]),
        helper.make_node("Min", ["positions", "last"], ["clamped"]),
        helper.make_node("Gather", ["enc", "clamped"], ["echo"], axis=1),
    ]


def _echo_initializers() -> list[onnx.TensorProto]:
    return [
        helper.make_tensor("axis2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("one", TensorProto.INT64, [], [1]),
        helper.make_tensor("depth", TensorProto.INT64, [1], [VOCAB_SIZE]),
        helper.make_tensor("off_on", TensorProto.FLOAT, [2], [0.0, 1.0]),
    ]


def _decoder_inputs() -> list[onnx.ValueInfoProto]:
    # Optimum lists the encoder mask first for BART; keep that order.
    return [
        helper.make_tensor_value_info("encoder_attention_mask", TensorProto.INT64, ["batch", "enc_seq"]),
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "dec_seq"]),
        helper.make_tensor_value_info("encoder_hidden_states", TensorProto.FLOAT, ["batch", "enc_seq", 1]),
    ]


def _decoder_outputs() -> list[onnx.ValueInfoProto]:
    return [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "dec_seq", VOCAB_SIZE]),
        # Real decoders also return cache tensors; the pipeline must ignore them.
        helper.make_tensor_value_info("present.0.encoder.key", TensorProto.FLOAT, ["batch", "enc_seq", 1]),
    ]


def make_decoder() -> None:
    nodes = _echo_nodes() + [
        helper.make_node("OneHot", ["echo", "depth", "off_on"], ["logits"], axis=-1),
        helper.make_node("Identity", ["encoder_hidden_states"], ["present.0.encoder.key"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_decoder", _decoder_inputs(), _decoder_outputs(), initializer=_echo_initializers()
    )
    _save(graph, "decoder_model.onnx")


def make_merged_decoder() -> None:
    nodes = _echo_nodes() + [
        helper.make_node("Cast", ["use_cache_branch"], ["branch_shift"], to=TensorProto.INT64),
        helper.make_node("ReduceSum", ["past_key_values.0.decoder.key"], ["past_sum"], keepdims=0),
        helper.make_node("Cast", ["past_sum"], ["past_shift"], to=TensorProto.INT64),
        helper.make_node("Add", ["echo", "branch_shift"], ["shifted_once"]),
        helper.make_node("Add", ["shifted_once", "past_shift"], ["shifted"]),
        helper.make_node("OneHot", ["shifted", "depth", "off_on"], ["logits"], axis=-1),
        helper.make_node("Identity", ["encoder_hidden_states"], ["present.0.encoder.key"]),
    ]
    inputs = _decoder_inputs() + [
        helper.make_tensor_value_info("use_cache_branch", TensorProto.BOOL, [1]),
        helper.make_tensor_value_info(
            "past_key_values.0.decoder.key", TensorProto.FLOAT, ["batch", 2, "past_seq", 4]
        ),
    ]
    graph = helper.make_graph(
        nodes, "tiny_merged_decoder", inputs, _decoder_outputs(), initializer=_echo_initializers()
    )
    _save(graph, "decoder_model_merged.onnx")


def make_fused() -> None:
    nodes = [
        helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"]),
        helper.make_node("Add", ["masked", "one"], ["output_ids"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_fused",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
        ],
        [helper.make_tensor_value_info("output_ids", TensorProto.INT64, ["batch", "seq"])],
        initializer=[helper.make_tensor("one", TensorProto.INT64, [], [1])],
    )
    _save(graph, "model.onnx")


def make_tokenizer() -> None:
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for offset, word in enumerate("abcdefghijkl"):
        vocab[word] = 4 + offset
    assert len(vocab) == VOCAB_SIZE
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )
    tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>"])
    tokenizer.save(str(HERE / "tokenizer.json"))


if __name__ == "__main__":
    make_encoder()
    make_decoder()
    make_merged_decoder()
    make_fused()
    make_tokenizer()
    print("fixtures written to", HERE)
