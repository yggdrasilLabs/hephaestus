"""Regenerate the tiny text generation test fixtures in this directory.

Run with any Python that has ``onnx`` installed, for example Forge's
environment::

    cd forge && uv run python ../crates/hephaestus-core/tests/fixtures/text_generation/make_fixtures.py

The graphs hold no weights. They follow the input and output names of an
Optimum decoder-only export and compute, with a few integer operations, a
next token that depends on *every* token so far, so a test can work out
the expected text by hand and a mistake in feeding the graph changes it.

Let ``sum_t`` be the sum of the token IDs at positions ``0..=t``. The
token predicted at position ``t`` is::

    target_t = 2 (</s>)          if sum_t % 13 == 0
               sum_t % 12 + 4    otherwise (one of the words a..l)

and the logits at position ``t`` are ``8 - |v - target_t|`` for token
``v``: 8 at the target, falling off by one per step away from it.

``model.onnx``
    With a key/value cache, as ``text-generation-with-past`` exports it:
    inputs ``input_ids``, ``attention_mask``, ``position_ids`` and
    ``past_key_values.0.{key,value}`` of shape ``(batch, 2, past, 4)``;
    outputs ``logits`` and ``present.0.{key,value}``. The key tensor
    carries the token IDs seen so far and the value tensor their
    positions; ``sum_t`` is computed from the key tensor, so it is only
    right when the cache of the previous step is fed back unchanged. Two
    checks are added to ``sum_t`` and are zero for a correct feed: the
    distance of the cached positions from ``0, 1, 2, ...`` and the
    distance of the attention mask's sum from the total sequence length.

``model_no_cache.onnx``
    The same function without a cache, as a plain ``text-generation``
    export: inputs ``input_ids`` and ``attention_mask``, output ``logits``.

``tokenizer_config.json``
    A chat template made of words the tokenizer knows: a system message is
    wrapped as ``a ... b``, a user message as ``c ... d``, and the reply
    starts after ``e``.

The tokenizer is ``../seq2seq/tokenizer.json`` (words ``a`` .. ``l`` =
4 .. 15, ``</s>`` = 2); tests copy it next to the graph.
"""

from __future__ import annotations

import json
import pathlib

import onnx
from onnx import TensorProto, helper

HERE = pathlib.Path(__file__).resolve().parent
VOCAB_SIZE = 16
HEADS = 2
HEAD_DIM = 4
OPSET = 14
IR_VERSION = 8

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}a {{ message['content'] }} b "
    "{% else %}c {{ message['content'] }} d {% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}e{% endif %}"
)


def _save(graph: onnx.GraphProto, name: str) -> None:
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(HERE / name))


def _int(name: str, values: list[int]) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def _scalar(name: str, value: int) -> onnx.TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [], [value])


def _logits_nodes(total: str) -> list[onnx.NodeProto]:
    """Nodes turning the per-position sums ``total`` (int64, (b, s)) into logits."""
    return [
        helper.make_node("Mod", [total, "thirteen"], ["mod13"]),
        helper.make_node("Equal", ["mod13", "zero"], ["is_end"]),
        helper.make_node("Mod", [total, "twelve"], ["mod12"]),
        helper.make_node("Add", ["mod12", "four"], ["word"]),
        helper.make_node("Where", ["is_end", "eos", "word"], ["target"]),
        helper.make_node("Unsqueeze", ["target", "axis2"], ["target_3d"]),
        helper.make_node("Sub", ["vocab", "target_3d"], ["distance"]),
        helper.make_node("Abs", ["distance"], ["abs_distance"]),
        helper.make_node("Sub", ["eight", "abs_distance"], ["logits_i"]),
        helper.make_node("Cast", ["logits_i"], ["logits"], to=TensorProto.FLOAT),
    ]


def _logits_initializers() -> list[onnx.TensorProto]:
    return [
        _scalar("thirteen", 13),
        _scalar("twelve", 12),
        _scalar("zero", 0),
        _scalar("four", 4),
        _scalar("eos", 2),
        _scalar("eight", 8),
        _int("axis2", [2]),
        _int("vocab", list(range(VOCAB_SIZE))),
    ]


def make_cached() -> None:
    nodes = [
        # This step's cache entries: token IDs (key) and positions (value),
        # broadcast to (batch, heads, seq, head_dim).
        helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["ids_f", "axes13"], ["ids_4d"]),
        helper.make_node("Mul", ["ids_4d", "cache_ones"], ["new_key"]),
        helper.make_node("Cast", ["position_ids"], ["pos_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["pos_f", "axes13"], ["pos_4d"]),
        helper.make_node("Mul", ["pos_4d", "cache_ones"], ["new_value"]),
        helper.make_node("Concat", ["past_key_values.0.key", "new_key"], ["present.0.key"], axis=2),
        helper.make_node("Concat", ["past_key_values.0.value", "new_value"], ["present.0.value"], axis=2),
        # Every token and position seen so far: (batch, total).
        helper.make_node("Gather", ["present.0.key", "zero"], ["key_h"], axis=1),
        helper.make_node("Gather", ["key_h", "zero"], ["history_f"], axis=2),
        helper.make_node("Cast", ["history_f"], ["history"], to=TensorProto.INT64),
        helper.make_node("Gather", ["present.0.value", "zero"], ["value_h"], axis=1),
        helper.make_node("Gather", ["value_h", "zero"], ["positions_f"], axis=2),
        helper.make_node("Cast", ["positions_f"], ["positions"], to=TensorProto.INT64),
        # Running sums, kept for the positions fed in this step.
        helper.make_node("CumSum", ["history", "one"], ["running"]),
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Gather", ["ids_shape", "one"], ["seq"], axis=0),
        helper.make_node("Neg", ["seq"], ["neg_seq"]),
        helper.make_node("Unsqueeze", ["neg_seq", "axis0"], ["slice_start"]),
        helper.make_node("Slice", ["running", "slice_start", "slice_end", "axis1"], ["sums"]),
        # Check 1: cached positions are 0, 1, 2, ...
        helper.make_node("Shape", ["history"], ["history_shape"]),
        helper.make_node("Gather", ["history_shape", "one"], ["total"], axis=0),
        helper.make_node("Range", ["zero", "total", "one"], ["expected_positions"]),
        helper.make_node("Sub", ["positions", "expected_positions"], ["position_diff"]),
        helper.make_node("Abs", ["position_diff"], ["position_abs"]),
        helper.make_node("ReduceSum", ["position_abs"], ["position_error"], keepdims=0),
        # Check 2: the attention mask covers the whole sequence.
        helper.make_node("ReduceSum", ["attention_mask"], ["mask_sum"], keepdims=0),
        helper.make_node("Sub", ["mask_sum", "total"], ["mask_diff"]),
        helper.make_node("Abs", ["mask_diff"], ["mask_error"]),
        helper.make_node("Add", ["sums", "position_error"], ["checked_1"]),
        helper.make_node("Add", ["checked_1", "mask_error"], ["checked"]),
        *_logits_nodes("checked"),
    ]
    cache_shape = ["batch", HEADS, "past_sequence_length", HEAD_DIM]
    present_shape = ["batch", HEADS, "past_sequence_length + sequence_length", HEAD_DIM]
    graph = helper.make_graph(
        nodes,
        "text_generation_with_past",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence_length"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "past_sequence_length + sequence_length"]
            ),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence_length"]),
            helper.make_tensor_value_info("past_key_values.0.key", TensorProto.FLOAT, cache_shape),
            helper.make_tensor_value_info("past_key_values.0.value", TensorProto.FLOAT, cache_shape),
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence_length", VOCAB_SIZE]),
            helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, present_shape),
            helper.make_tensor_value_info("present.0.value", TensorProto.FLOAT, present_shape),
        ],
        initializer=[
            *_logits_initializers(),
            _scalar("one", 1),
            _int("axis0", [0]),
            _int("axis1", [1]),
            _int("axes13", [1, 3]),
            _int("slice_end", [2**62]),
            helper.make_tensor("cache_ones", TensorProto.FLOAT, [1, HEADS, 1, HEAD_DIM], [1.0] * (HEADS * HEAD_DIM)),
        ],
    )
    _save(graph, "model.onnx")


def make_uncached() -> None:
    nodes = [
        helper.make_node("CumSum", ["input_ids", "one"], ["sums"]),
        helper.make_node("ReduceSum", ["attention_mask"], ["mask_sum"], keepdims=0),
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Gather", ["ids_shape", "one"], ["total"], axis=0),
        helper.make_node("Sub", ["mask_sum", "total"], ["mask_diff"]),
        helper.make_node("Abs", ["mask_diff"], ["mask_error"]),
        helper.make_node("Add", ["sums", "mask_error"], ["checked"]),
        *_logits_nodes("checked"),
    ]
    graph = helper.make_graph(
        nodes,
        "text_generation",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence_length"]),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "sequence_length"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence_length", VOCAB_SIZE])],
        initializer=[*_logits_initializers(), _scalar("one", 1)],
    )
    _save(graph, "model_no_cache.onnx")


def make_tokenizer_config() -> None:
    config = {"bos_token": "<s>", "eos_token": "</s>", "chat_template": CHAT_TEMPLATE}
    (HERE / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    make_cached()
    make_uncached()
    make_tokenizer_config()
