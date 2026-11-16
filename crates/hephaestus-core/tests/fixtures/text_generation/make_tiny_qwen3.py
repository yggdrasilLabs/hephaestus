"""Regenerate ``tiny_qwen3/``: a real decoder-only transformer, very small.

Run with a Python that has ``torch``, ``transformers`` and ``optimum``,
for example Forge's environment::

    cd forge && uv run python ../crates/hephaestus-core/tests/fixtures/text_generation/make_tiny_qwen3.py

The graphs of ``make_fixtures.py`` compute in whole numbers what a
transformer is supposed to compute. This one is the real thing, to check
the batched decode loop against real attention: a ``Qwen3ForCausalLM``
with two layers, grouped-query attention (4 heads, 2 key/value heads),
rotary positions and random weights (seed 7), exported by Optimum as
``text-generation-with-past`` and, like the onnx-community Qwen3 exports,
*without* a ``position_ids`` input: the graph numbers positions by
column. Inputs are ``input_ids``, ``attention_mask`` and
``past_key_values.{0,1}.{key,value}`` of shape ``(batch, 2, past, 8)``.

The vocabulary has 16 tokens, so the model works with the tokenizer and
the chat template of the other fixtures (``</s>`` = 2 ends a reply).
Random weights say nothing sensible, but which token is most likely
depends on every earlier token, on the positions and on the mask through
real float32 attention, so a padded row only gives the tokens of the same
request alone if padding, mask and cache are handled the way the model
expects. Logits of a padded row differ from the solo run by about 1e-6.

Only ``model.onnx`` and ``config.json`` are kept.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import warnings

import torch
from optimum.exporters.onnx import main_export
from optimum.exporters.tasks import TasksManager
from transformers import Qwen3Config, Qwen3ForCausalLM

HERE = pathlib.Path(__file__).resolve().parent
TARGET = HERE / "tiny_qwen3"
SEED = 7
WEIGHT_STD = 0.35
OPSET = 18


def main() -> None:
    warnings.filterwarnings("ignore")
    torch.manual_seed(SEED)
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=2,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config).eval()
    # The default initialisation is so narrow that every prompt gets the
    # same reply; spread the weights so the reply depends on the prompt.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() > 1:
                parameter.normal_(0.0, WEIGHT_STD)

    constructor = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model_type="qwen3",
        task="text-generation-with-past",
        library_name="transformers",
    )

    class WithoutPositionIds(type(constructor(config))):
        """Optimum's Qwen3 export, minus the ``position_ids`` input."""

        @property
        def inputs(self):
            return {name: axes for name, axes in super().inputs.items() if name != "position_ids"}

    onnx_config = WithoutPositionIds(
        config, task="text-generation", use_past=True, use_past_in_inputs=True
    )
    with tempfile.TemporaryDirectory() as work:
        source = pathlib.Path(work) / "source"
        exported = pathlib.Path(work) / "exported"
        model.save_pretrained(source)
        main_export(
            str(source),
            output=str(exported),
            task="text-generation-with-past",
            custom_onnx_configs={"model": onnx_config},
            no_post_process=True,
            do_validation=False,
            opset=OPSET,
        )
        TARGET.mkdir(exist_ok=True)
        for name in ("model.onnx", "config.json"):
            shutil.copyfile(exported / name, TARGET / name)


if __name__ == "__main__":
    main()
