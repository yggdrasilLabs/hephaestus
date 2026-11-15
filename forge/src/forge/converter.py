"""ONNX model conversion and validation logic."""

from __future__ import annotations

import importlib.metadata
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from optimum.exporters.onnx import main_export
from transformers import AutoConfig, AutoTokenizer

from forge.models import ConversionMetadata


# File names of an encoder-decoder (seq2seq) export.  ``main_export``
# writes these instead of ``model.onnx`` for models such as BART and T5.
# Hephaestus looks for the same names; keep the two in step.
ENCODER_FILE = "encoder_model.onnx"
# In Hephaestus's order of preference: it decodes without a key/value
# cache, so the plain decoder is the one it loads when both exist.
DECODER_FILES = ("decoder_model.onnx", "decoder_model_merged.onnx")

# Whisper exports use the same file names but are served by Hephaestus's
# ASR pipeline, whose files this conversion path has never produced.
WHISPER_ARCHITECTURE = "WhisperForConditionalGeneration"


class ConversionError(Exception):
    """Raised when model conversion or validation fails."""


def convert_model(model_id: str, output_dir: str) -> ConversionMetadata:
    """Convert a HuggingFace model to ONNX format.

    Downloads the model from HuggingFace, exports it to ONNX via
    ``optimum``, and ensures the tokenizer is saved alongside the
    model.  Returns metadata about the conversion.
    """
    start = time.monotonic()

    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="auto",
    )

    # Ensure tokenizer.json is present (pitfall 6: main_export may not
    # always save it in the expected format).
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.monotonic() - start

    config = AutoConfig.from_pretrained(output_dir)

    return ConversionMetadata(
        architecture=getattr(config, "model_type", "unknown"),
        original_format="pytorch",
        conversion_duration_secs=round(elapsed, 2),
        # ``optimum`` 2.x is a namespace package without ``__version__``;
        # the installed distribution's metadata has the version either way.
        optimum_version=importlib.metadata.version("optimum"),
    )


def find_encoder_decoder_export(output_dir: str) -> tuple[str, str] | None:
    """Locate an encoder-decoder export in *output_dir*.

    Returns ``(encoder_path, decoder_path)`` when ``encoder_model.onnx``
    and a decoder graph sit together in *output_dir* or in its ``onnx/``
    subdirectory, and ``None`` otherwise.  The decoder is the one
    Hephaestus will load: ``decoder_model.onnx`` if present, else
    ``decoder_model_merged.onnx``.

    Whisper models are never reported, so their conversion result is
    judged exactly as before this layout was accepted.
    """
    try:
        with open(os.path.join(output_dir, "config.json")) as f:
            architectures = json.load(f).get("architectures") or []
    except (OSError, ValueError, AttributeError):
        architectures = []
    if WHISPER_ARCHITECTURE in architectures:
        return None

    for directory in (output_dir, os.path.join(output_dir, "onnx")):
        encoder_path = os.path.join(directory, ENCODER_FILE)
        if not os.path.isfile(encoder_path):
            continue
        for decoder_file in DECODER_FILES:
            decoder_path = os.path.join(directory, decoder_file)
            if os.path.isfile(decoder_path):
                return encoder_path, decoder_path
    return None


def _validate_graph(model_path: str) -> None:
    """Run both validation stages on one graph of an encoder-decoder export.

    Stage 1 checks the graph structure by file path.  Stage 2 runs a
    dummy inference with every dynamic dimension set to 1; a merged
    decoder's boolean ``use_cache_branch`` is fed ``False``, which
    selects the branch Hephaestus runs.
    """
    name = os.path.basename(model_path)
    try:
        onnx.checker.check_model(model_path)
    except Exception as exc:
        raise ConversionError(
            f"onnx.checker.check_model failed for {name}: {exc}"
        ) from exc

    try:
        session = ort.InferenceSession(model_path)
        dummy_inputs: dict[str, np.ndarray] = {}
        for inp in session.get_inputs():
            shape = [
                1 if d is None or isinstance(d, str) else d
                for d in inp.shape
            ]
            if inp.type == "tensor(int64)":
                dummy_inputs[inp.name] = np.ones(shape, dtype=np.int64)
            elif inp.type == "tensor(bool)":
                dummy_inputs[inp.name] = np.zeros(shape, dtype=np.bool_)
            else:
                dummy_inputs[inp.name] = np.ones(shape, dtype=np.float32)
        session.run(None, dummy_inputs)
    except Exception as exc:
        raise ConversionError(
            f"dummy inference failed for {name}: {exc}"
        ) from exc


def _decoder_only_dummy_inputs(
    session: ort.InferenceSession,
) -> dict[str, np.ndarray] | None:
    """Dummy inputs for a decoder-only (text generation) ``model.onnx``.

    ``main_export`` writes a causal language model as a single
    ``model.onnx`` with ``past_key_values.*`` inputs.  Its inputs must
    agree with each other: the attention mask covers the cached tokens
    plus the new ones, and positions index into that sequence.  Feeding
    every dynamic dimension as 1 and every value as 1 breaks that (one
    cached token, one new token, a mask of one, position 1), so the
    graph is fed the first step of a generation instead: an empty cache,
    one token, a mask of one, position 0.  That is also the first call
    Hephaestus makes.

    Returns ``None`` for any graph without ``past_key_values.*`` inputs,
    and for a merged decoder (``use_cache_branch``); those are fed as
    they always were.
    """
    inputs = session.get_inputs()
    names = [inp.name for inp in inputs]
    if "use_cache_branch" in names or not any(
        name.startswith("past_key_values.") for name in names
    ):
        return None

    feeds: dict[str, np.ndarray] = {}
    for inp in inputs:
        shape = [
            1 if d is None or isinstance(d, str) else d
            for d in inp.shape
        ]
        if inp.name.startswith("past_key_values."):
            # (batch, heads, sequence, head_dim): nothing cached yet.
            if len(shape) == 4:
                shape[2] = 0
            feeds[inp.name] = np.zeros(shape, dtype=np.float32)
        elif inp.name == "position_ids":
            feeds[inp.name] = np.zeros(shape, dtype=np.int64)
        elif inp.type == "tensor(int64)":
            feeds[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            feeds[inp.name] = np.ones(shape, dtype=np.float32)
    return feeds


def validate_model(output_dir: str) -> None:
    """Two-stage ONNX model validation.

    Stage 1: ``onnx.checker.check_model`` with a *file path string*
    (avoids loading the full protobuf into memory -- pitfall 2).

    Stage 2: Dummy inference via ``onnxruntime.InferenceSession`` to
    catch runtime errors that graph-only validation misses.

    Also verifies that ``tokenizer.json`` and ``config.json`` exist
    and are parseable (D-07).

    A decoder-only (text generation) ``model.onnx`` is fed the first
    step of a generation in stage 2 (see
    :func:`_decoder_only_dummy_inputs`).

    A seq2seq export has no ``model.onnx``.  When neither ``model.onnx``
    location exists, an encoder-decoder export (see
    :func:`find_encoder_decoder_export`) is accepted instead, and both
    stages run on its encoder and on the decoder Hephaestus will load.

    Raises :class:`ConversionError` on any failure.
    """

    # --- Locate model.onnx ---
    model_path = os.path.join(output_dir, "model.onnx")
    if not os.path.isfile(model_path):
        # Some exports put it under an onnx/ subdirectory.
        alt = os.path.join(output_dir, "onnx", "model.onnx")
        if os.path.isfile(alt):
            model_path = alt
        else:
            export = find_encoder_decoder_export(output_dir)
            if export is None:
                raise ConversionError(
                    f"model.onnx not found in {output_dir}"
                )
            for graph_path in export:
                _validate_graph(graph_path)
            _validate_artifacts(output_dir)
            return

    # --- Stage 1: Graph structure validation ---
    try:
        onnx.checker.check_model(model_path)
    except Exception as exc:
        raise ConversionError(
            f"onnx.checker.check_model failed: {exc}"
        ) from exc

    # --- Stage 2: Dummy inference ---
    try:
        session = ort.InferenceSession(model_path)
        dummy_inputs = _decoder_only_dummy_inputs(session)
        if dummy_inputs is None:
            dummy_inputs = {}
            for inp in session.get_inputs():
                shape = [
                    1 if d is None or isinstance(d, str) else d
                    for d in inp.shape
                ]
                if inp.type == "tensor(int64)":
                    dummy_inputs[inp.name] = np.ones(shape, dtype=np.int64)
                else:
                    dummy_inputs[inp.name] = np.ones(shape, dtype=np.float32)
        session.run(None, dummy_inputs)
    except Exception as exc:
        raise ConversionError(
            f"dummy inference failed: {exc}"
        ) from exc

    _validate_artifacts(output_dir)


def _validate_artifacts(output_dir: str) -> None:
    """Verify ``tokenizer.json`` and ``config.json`` exist and parse (D-07)."""

    # --- Artifact presence checks (D-07) ---
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        raise ConversionError(
            f"tokenizer.json not found in {output_dir}"
        )
    try:
        with open(tokenizer_path) as f:
            json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ConversionError(
            f"tokenizer.json is not valid JSON: {exc}"
        ) from exc

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ConversionError(
            f"config.json not found in {output_dir}"
        )
    try:
        with open(config_path) as f:
            json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ConversionError(
            f"config.json is not valid JSON: {exc}"
        ) from exc
