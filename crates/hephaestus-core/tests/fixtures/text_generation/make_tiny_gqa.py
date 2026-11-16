"""Regenerate ``tiny_gqa/``: a tiny graph shaped like an ONNX Runtime GenAI export.

Run with a Python that has ``numpy`` and ``onnx``, for example Forge's
environment::

    cd forge && uv run python ../crates/hephaestus-core/tests/fixtures/text_generation/make_tiny_gqa.py

The onnx-community exports of large language models (for example
``onnx-community/Qwen3-4B-Instruct-2507-ONNX``) are not Optimum exports.
They come from the ONNX Runtime GenAI model builder (producer
``huggingface``, opset 21 plus ``com.microsoft``), and their attention is
one fused ``com.microsoft.GroupQueryAttention`` node per layer with the
rotary embedding inside (``do_rotary = 1``, ``cos_cache`` / ``sin_cache``).
Such a graph reads ``attention_mask`` in exactly two places::

    seqlens_k             = Cast(ReduceSum(attention_mask, axis 1) - 1, int32)
    total_sequence_length = Cast(Shape(attention_mask)[1], int32)

So the mask is not a mask. Only the number of ones in a row counts: it is
the row's length. The operator takes a row's cache to be its first
``length - fed`` cache columns, numbers the fed tokens from there, appends
their keys right after, and attends to nothing else. *Where* the zeros of
a row are makes no difference: a left-padded row is read as if its padding
were real tokens.

This script builds the same structure by hand, tiny, with random weights
(seed 11): two layers, 4 heads, 2 key/value heads of size 16 (the operator wants a
multiple of 16 for its rotary embedding), 16 tokens.
Inputs and outputs carry the names and symbolic dimensions of the real
export: ``input_ids``, ``attention_mask`` and
``past_key_values.{0,1}.{key,value}`` of shape ``(batch, 2, past, 16)``,
no ``position_ids``. Per layer: RMS norm, bias-free projections, the fused
attention, a SiLU feed-forward block. (The per-head query/key norms of
Qwen3 are left out; they do not touch masks or positions.)

Written next to ``config.json``:

``model.onnx``
    float32 weights, cache and logits.

``model_fp16.onnx``
    float16 weights and cache, float32 logits behind a final cast, like
    ``model_fp16.onnx`` of the real export.

The vocabulary has 16 tokens, so the model works with the tokenizer and
the chat template of the other fixtures (``</s>`` = 2 ends a reply).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = pathlib.Path(__file__).resolve().parent
TARGET = HERE / "tiny_gqa"
SEED = 11
WEIGHT_STD = 0.35
VOCAB = 16
HIDDEN = 32
INTERMEDIATE = 64
LAYERS = 2
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
MAX_POSITIONS = 512
ROPE_THETA = 10000.0
MICROSOFT = "com.microsoft"


def build(float_type: int) -> onnx.ModelProto:
    dtype = np.float16 if float_type == TensorProto.FLOAT16 else np.float32
    rng = np.random.default_rng(SEED)
    initializers: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def weight(name: str, *shape: int) -> str:
        values = rng.normal(0.0, WEIGHT_STD, size=shape).astype(np.float32)
        initializers.append(numpy_helper.from_array(values.astype(dtype), name))
        return name

    def ones(name: str, size: int) -> str:
        initializers.append(numpy_helper.from_array(np.ones(size, dtype=dtype), name))
        return name

    def node(op: str, inputs: list[str], output: str, domain: str = "", **attributes) -> str:
        nodes.append(
            helper.make_node(op, inputs, [output], name=output, domain=domain, **attributes)
        )
        return output

    # Rotary tables, as the builder writes them.
    frequencies = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    angles = np.outer(np.arange(MAX_POSITIONS, dtype=np.float32), frequencies)
    initializers.append(numpy_helper.from_array(np.cos(angles).astype(dtype), "cos_cache"))
    initializers.append(numpy_helper.from_array(np.sin(angles).astype(dtype), "sin_cache"))
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "one_1d"))
    initializers.append(numpy_helper.from_array(np.array(1, dtype=np.int64), "one"))

    # The only readers of attention_mask.
    mask = "/model/attn_mask_reformat/attn_mask_subgraph"
    node("ReduceSum", ["attention_mask", "one_1d"], f"{mask}/ReduceSum/output_0", keepdims=1)
    node("Sub", [f"{mask}/ReduceSum/output_0", "one_1d"], f"{mask}/Sub/output_0")
    seqlens_k = node("Cast", [f"{mask}/Sub/output_0"], f"{mask}/Sub/Cast/output_0", to=TensorProto.INT32)
    node("Shape", ["attention_mask"], f"{mask}/Shape/output_0")
    node("Gather", [f"{mask}/Shape/output_0", "one"], f"{mask}/Gather/output_0", axis=0)
    total = node("Cast", [f"{mask}/Gather/output_0"], f"{mask}/Gather/Cast/output_0", to=TensorProto.INT32)

    embedding = weight("model.embed_tokens.weight", VOCAB, HIDDEN)
    hidden = node("Gather", [embedding, "input_ids"], "/model/embed_tokens/Gather/output_0")

    def norm(name: str, source: str) -> str:
        return node(
            "SimplifiedLayerNormalization",
            [source, ones(f"{name}.weight", HIDDEN)],
            f"{name}/output_0",
            axis=-1,
            epsilon=1e-6,
            stash_type=1,
        )

    def linear(name: str, source: str, inputs: int, outputs: int) -> str:
        return node("MatMul", [source, weight(f"{name}.weight", inputs, outputs)], f"{name}/output_0")

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch_size", "sequence_length"]),
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "logits", TensorProto.FLOAT, ["batch_size", "sequence_length", VOCAB]
        )
    ]
    for layer in range(LAYERS):
        prefix = f"/model/layers.{layer}"
        normed = norm(f"{prefix}/input_layernorm", hidden)
        query = linear(f"{prefix}/attn/q_proj/MatMul", normed, HIDDEN, HEADS * HEAD_DIM)
        key = linear(f"{prefix}/attn/k_proj/MatMul", normed, HIDDEN, KV_HEADS * HEAD_DIM)
        value = linear(f"{prefix}/attn/v_proj/MatMul", normed, HIDDEN, KV_HEADS * HEAD_DIM)
        attended = f"{prefix}/attn/GroupQueryAttention/output_0"
        nodes.append(
            helper.make_node(
                "GroupQueryAttention",
                [
                    query,
                    key,
                    value,
                    f"past_key_values.{layer}.key",
                    f"past_key_values.{layer}.value",
                    seqlens_k,
                    total,
                    "cos_cache",
                    "sin_cache",
                ],
                [attended, f"present.{layer}.key", f"present.{layer}.value"],
                name=f"{prefix}/attn/GroupQueryAttention",
                domain=MICROSOFT,
                num_heads=HEADS,
                kv_num_heads=KV_HEADS,
                scale=float(HEAD_DIM**-0.5),
                do_rotary=1,
                rotary_interleaved=0,
            )
        )
        projected = linear(f"{prefix}/attn/o_proj/MatMul", attended, HEADS * HEAD_DIM, HIDDEN)
        hidden = node("Add", [hidden, projected], f"{prefix}/attn/Add/output_0")
        normed = norm(f"{prefix}/post_attention_layernorm", hidden)
        gate = linear(f"{prefix}/mlp/gate_proj/MatMul", normed, HIDDEN, INTERMEDIATE)
        up = linear(f"{prefix}/mlp/up_proj/MatMul", normed, HIDDEN, INTERMEDIATE)
        sigmoid = node("Sigmoid", [gate], f"{prefix}/mlp/act_fn/Sigmoid/output_0")
        silu = node("Mul", [gate, sigmoid], f"{prefix}/mlp/act_fn/Mul/output_0")
        gated = node("Mul", [silu, up], f"{prefix}/mlp/Mul/output_0")
        down = linear(f"{prefix}/mlp/down_proj/MatMul", gated, INTERMEDIATE, HIDDEN)
        hidden = node("Add", [hidden, down], f"{prefix}/mlp/Add/output_0")
        for kind in ("key", "value"):
            inputs.append(
                helper.make_tensor_value_info(
                    f"past_key_values.{layer}.{kind}",
                    float_type,
                    ["batch_size", KV_HEADS, "past_sequence_length", HEAD_DIM],
                )
            )
            outputs.append(
                helper.make_tensor_value_info(
                    f"present.{layer}.{kind}",
                    float_type,
                    ["batch_size", KV_HEADS, "total_sequence_length", HEAD_DIM],
                )
            )

    final = norm(f"/model/layers.{LAYERS}/final_norm_layernorm", hidden)
    head = weight("lm_head.MatMul.weight", HIDDEN, VOCAB)
    if float_type == TensorProto.FLOAT16:
        node("MatMul", [final, head], "logits_internal")
        node("Cast", ["logits_internal"], "logits", to=TensorProto.FLOAT)
    else:
        node("MatMul", [final, head], "logits")

    graph = helper.make_graph(nodes, "main_graph", inputs, outputs, initializers)
    model = helper.make_model(
        graph,
        producer_name="huggingface",
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid(MICROSOFT, 1)],
    )
    model.ir_version = 10
    return model


def main() -> None:
    TARGET.mkdir(exist_ok=True)
    onnx.save(build(TensorProto.FLOAT), TARGET / "model.onnx")
    onnx.save(build(TensorProto.FLOAT16), TARGET / "model_fp16.onnx")
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "max_position_embeddings": MAX_POSITIONS,
        "model_type": "qwen3",
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "vocab_size": VOCAB,
    }
    (TARGET / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
