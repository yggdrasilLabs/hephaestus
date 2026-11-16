"""Regenerate ``fp16/``: the encoder-decoder fixtures in half precision.

Run with any Python that has ``onnx`` installed, after ``make_fixtures.py``::

    cd forge && uv run python ../crates/hephaestus-core/tests/fixtures/seq2seq/make_fixtures_fp16.py

Takes ``encoder_model.onnx``, ``decoder_model.onnx`` and
``decoder_model_merged_cached.onnx`` and writes the same graphs with every
``float32`` tensor declared ``float16`` instead: inputs, outputs,
constants, cast targets, and the same inside the branches of the merged
decoder. Nothing else changes. The graphs compute in small whole numbers,
which ``float16`` holds exactly, so the half-precision export generates
exactly the text of the full-precision one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

HERE = pathlib.Path(__file__).resolve().parent
TARGET = HERE / "fp16"
GRAPHS = ("encoder_model.onnx", "decoder_model.onnx", "decoder_model_merged_cached.onnx")


def to_half(tensor: onnx.TensorProto) -> None:
    if tensor.data_type != TensorProto.FLOAT:
        return
    values = numpy_helper.to_array(tensor).astype(np.float16)
    tensor.CopyFrom(numpy_helper.from_array(values, tensor.name))


def retype(graph: onnx.GraphProto) -> None:
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT:
            value.type.tensor_type.elem_type = TensorProto.FLOAT16
    for initializer in graph.initializer:
        to_half(initializer)
    for node in graph.node:
        for attribute in node.attribute:
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                attribute.i = TensorProto.FLOAT16
            if attribute.type == onnx.AttributeProto.TENSOR:
                to_half(attribute.t)
            if attribute.type == onnx.AttributeProto.GRAPH:
                retype(attribute.g)


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    for name in GRAPHS:
        model = onnx.load(HERE / name)
        retype(model.graph)
        onnx.checker.check_model(model)
        onnx.save(model, TARGET / name)


if __name__ == "__main__":
    main()
