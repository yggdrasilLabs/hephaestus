"""Regenerate the half-precision variants of ``tiny_qwen3/model.onnx``.

Run with a Python that has ``onnx`` and ``onnxruntime``, for example
Forge's environment::

    cd forge && uv run python ../crates/hephaestus-core/tests/fixtures/text_generation/make_tiny_qwen3_fp16.py

Repositories of exported language models keep several precision variants
of one graph side by side. The half-precision ones (``model_fp16.onnx``,
``model_q4f16.onnx``) compute in ``float16`` and declare ``float16`` cache
tensors; their logits are ``float16`` too, or ``float32`` behind a final
cast, depending on the tool that made them. This script converts the
float32 graph of ``make_tiny_qwen3.py`` with ONNX Runtime's own converter
and writes, next to it:

``model_fp16.onnx``
    Everything ``float16``: weights, cache inputs and outputs, logits.

``model_fp16_logits32.onnx``
    ``float16`` weights and cache, ``float32`` logits.

``model_logits16.onnx``
    ``float32`` cache inputs and outputs, ``float16`` logits (and
    ``float16`` weights in between, behind casts).

The weights are those of ``model.onnx`` rounded to ``float16``, so the
graphs are the same model at lower precision: their logits differ from
the float32 ones by about 1e-2 and their replies may differ.
"""

from __future__ import annotations

import pathlib

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "tiny_qwen3" / "model.onnx"
FLOAT16 = onnx.TensorProto.FLOAT16
FLOAT = onnx.TensorProto.FLOAT


def convert(name: str, keep: list[str], cache: int, logits: int) -> None:
    model = onnx.load(SOURCE)
    converted = convert_float_to_float16(model, keep_io_types=keep or False)
    # The converter appends the casts of kept inputs after their readers.
    nodes = list(converted.graph.node)
    casts = [node for node in nodes if node.name.startswith("graph_input_cast")]
    others = [node for node in nodes if not node.name.startswith("graph_input_cast")]
    del converted.graph.node[:]
    converted.graph.node.extend(casts + others)
    onnx.checker.check_model(converted)
    for value in list(converted.graph.input) + list(converted.graph.output):
        element = value.type.tensor_type.elem_type
        if value.name == "logits":
            assert element == logits, (name, value.name, element)
        elif value.name.startswith(("past_key_values.", "present.")):
            assert element == cache, (name, value.name, element)
    onnx.save(converted, SOURCE.with_name(name))


def main() -> None:
    model = onnx.load(SOURCE)
    cache = [
        value.name
        for value in list(model.graph.input) + list(model.graph.output)
        if value.name.startswith(("past_key_values.", "present."))
    ]
    convert("model_fp16.onnx", [], FLOAT16, FLOAT16)
    convert("model_fp16_logits32.onnx", ["logits"], FLOAT16, FLOAT)
    convert("model_logits16.onnx", cache, FLOAT, FLOAT16)


if __name__ == "__main__":
    main()
