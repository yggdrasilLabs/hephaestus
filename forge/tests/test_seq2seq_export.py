"""Tests for seq2seq (encoder-decoder) exports in forge.converter and forge.storage."""

from __future__ import annotations

import os

import onnx
import opendal
import pytest
from onnx import TensorProto, helper

from forge.converter import ConversionError, find_encoder_decoder_export, validate_model
from forge.storage import upload_to_storage
from tests.test_converter import _make_minimal_onnx, _write_json


def _make_tiny_graph(path: str, inputs: list[tuple[str, int, list]]) -> None:
    """Write a valid ONNX graph with the given inputs at *path*.

    Each input is ``(name, elem_type, shape)``.  The graph reduces every
    input to a float scalar and sums them into a single ``logits``
    output, so a dummy inference only succeeds when every input is fed
    with the right element type.
    """
    nodes = []
    scalars = []
    for index, (name, elem_type, _shape) in enumerate(inputs):
        as_float = f"f{index}"
        scalar = f"s{index}"
        nodes.append(helper.make_node("Cast", [name], [as_float], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("ReduceSum", [as_float], [scalar], keepdims=0))
        scalars.append(scalar)
    nodes.append(helper.make_node("Sum", scalars, ["logits"]))

    graph = helper.make_graph(
        nodes,
        "tiny_graph",
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)


_ENCODER_INPUTS = [
    ("input_ids", TensorProto.INT64, ["batch", "seq"]),
    ("attention_mask", TensorProto.INT64, ["batch", "seq"]),
]
_DECODER_INPUTS = [
    ("encoder_attention_mask", TensorProto.INT64, ["batch", "enc_seq"]),
    ("input_ids", TensorProto.INT64, ["batch", "dec_seq"]),
    ("encoder_hidden_states", TensorProto.FLOAT, ["batch", "enc_seq", 4]),
]
_MERGED_DECODER_INPUTS = _DECODER_INPUTS + [
    ("use_cache_branch", TensorProto.BOOL, [1]),
    ("past_key_values.0.decoder.key", TensorProto.FLOAT, ["batch", 2, "past_seq", 2]),
]


def _write_artifacts(output_dir: str, config: dict | None = None) -> None:
    _write_json(os.path.join(output_dir, "tokenizer.json"), {"type": "test"})
    _write_json(
        os.path.join(output_dir, "config.json"),
        config or {"architectures": ["BartForConditionalGeneration"]},
    )


class TestValidateEncoderDecoderExport:
    """validate_model accepts a seq2seq export that has no model.onnx."""

    def test_encoder_and_decoder_pass(self, tmp_output_dir: str) -> None:
        # Arrange -- the files main_export writes for a seq2seq model.
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(os.path.join(tmp_output_dir, "decoder_model.onnx"), _DECODER_INPUTS)
        _write_artifacts(tmp_output_dir)

        # Act / Assert -- should not raise.
        validate_model(tmp_output_dir)

    def test_export_in_onnx_subdirectory_passes(self, tmp_output_dir: str) -> None:
        onnx_dir = os.path.join(tmp_output_dir, "onnx")
        _make_tiny_graph(os.path.join(onnx_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(os.path.join(onnx_dir, "decoder_model.onnx"), _DECODER_INPUTS)
        _write_artifacts(tmp_output_dir)

        validate_model(tmp_output_dir)

    def test_merged_decoder_alone_passes(self, tmp_output_dir: str) -> None:
        # Arrange -- the boolean branch selector must be fed as a bool.
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(
            os.path.join(tmp_output_dir, "decoder_model_merged.onnx"), _MERGED_DECODER_INPUTS
        )
        _write_artifacts(tmp_output_dir)

        validate_model(tmp_output_dir)

    def test_plain_decoder_is_the_one_validated(self, tmp_output_dir: str) -> None:
        # Arrange -- Hephaestus loads decoder_model.onnx when both exist,
        # so a broken merged decoder next to it must not fail validation.
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(os.path.join(tmp_output_dir, "decoder_model.onnx"), _DECODER_INPUTS)
        with open(os.path.join(tmp_output_dir, "decoder_model_merged.onnx"), "wb") as f:
            f.write(b"not an onnx file")
        _write_artifacts(tmp_output_dir)

        validate_model(tmp_output_dir)

    def test_encoder_without_decoder_is_still_missing_model(self, tmp_output_dir: str) -> None:
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _write_artifacts(tmp_output_dir)

        with pytest.raises(ConversionError, match="model.onnx not found"):
            validate_model(tmp_output_dir)

    def test_broken_decoder_fails_validation(self, tmp_output_dir: str) -> None:
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        with open(os.path.join(tmp_output_dir, "decoder_model.onnx"), "wb") as f:
            f.write(b"not an onnx file")
        _write_artifacts(tmp_output_dir)

        with pytest.raises(ConversionError, match="decoder_model.onnx"):
            validate_model(tmp_output_dir)

    def test_missing_tokenizer_json_still_fails(self, tmp_output_dir: str) -> None:
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(os.path.join(tmp_output_dir, "decoder_model.onnx"), _DECODER_INPUTS)
        _write_json(
            os.path.join(tmp_output_dir, "config.json"),
            {"architectures": ["BartForConditionalGeneration"]},
        )

        with pytest.raises(ConversionError, match="tokenizer.json not found"):
            validate_model(tmp_output_dir)

    def test_whisper_export_is_still_missing_model(self, tmp_output_dir: str) -> None:
        # Arrange -- Whisper exports use the same file names; their
        # outcome must not change.
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(os.path.join(tmp_output_dir, "decoder_model.onnx"), _DECODER_INPUTS)
        _write_artifacts(
            tmp_output_dir, {"architectures": ["WhisperForConditionalGeneration"]}
        )

        with pytest.raises(ConversionError, match="model.onnx not found"):
            validate_model(tmp_output_dir)


class TestSingleFileExportTakesPrecedence:
    """A model.onnx is validated exactly as before, whatever sits next to it."""

    def test_encoder_decoder_files_are_ignored_when_model_onnx_exists(
        self, tmp_output_dir: str
    ) -> None:
        # Arrange -- unreadable encoder/decoder files next to a valid model.onnx.
        _make_minimal_onnx(tmp_output_dir)
        for name in ("encoder_model.onnx", "decoder_model.onnx"):
            with open(os.path.join(tmp_output_dir, name), "wb") as f:
                f.write(b"not an onnx file")
        _write_artifacts(tmp_output_dir)

        # Act / Assert -- only model.onnx is looked at.
        validate_model(tmp_output_dir)

    def test_broken_model_onnx_fails_even_with_valid_encoder_decoder(
        self, tmp_output_dir: str
    ) -> None:
        with open(os.path.join(tmp_output_dir, "model.onnx"), "wb") as f:
            f.write(b"not an onnx file")
        _make_tiny_graph(os.path.join(tmp_output_dir, "encoder_model.onnx"), _ENCODER_INPUTS)
        _make_tiny_graph(os.path.join(tmp_output_dir, "decoder_model.onnx"), _DECODER_INPUTS)
        _write_artifacts(tmp_output_dir)

        with pytest.raises(ConversionError, match="onnx.checker.check_model failed"):
            validate_model(tmp_output_dir)


class TestFindEncoderDecoderExport:
    """find_encoder_decoder_export picks the decoder Hephaestus will load."""

    def test_prefers_plain_decoder(self, tmp_output_dir: str) -> None:
        for name in ("encoder_model.onnx", "decoder_model.onnx", "decoder_model_merged.onnx"):
            with open(os.path.join(tmp_output_dir, name), "wb") as f:
                f.write(b"x")

        export = find_encoder_decoder_export(tmp_output_dir)

        assert export == (
            os.path.join(tmp_output_dir, "encoder_model.onnx"),
            os.path.join(tmp_output_dir, "decoder_model.onnx"),
        )

    def test_returns_none_without_encoder(self, tmp_output_dir: str) -> None:
        with open(os.path.join(tmp_output_dir, "decoder_model.onnx"), "wb") as f:
            f.write(b"x")

        assert find_encoder_decoder_export(tmp_output_dir) is None


class TestConvertModelMetadata:
    """convert_model reports metadata without touching the network."""

    def test_reports_installed_optimum_version(
        self, tmp_output_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange -- stub the export and the Hub lookups.
        import forge.converter as converter

        class _Tokenizer:
            def save_pretrained(self, _output_dir: str) -> None:
                return None

        class _Config:
            model_type = "bart"

        monkeypatch.setattr(converter, "main_export", lambda **_kwargs: None)
        monkeypatch.setattr(
            converter.AutoTokenizer, "from_pretrained", lambda _model_id: _Tokenizer()
        )
        monkeypatch.setattr(converter.AutoConfig, "from_pretrained", lambda _dir: _Config())

        # Act
        metadata = converter.convert_model("org/model", tmp_output_dir)

        # Assert
        assert metadata.architecture == "bart"
        assert metadata.original_format == "pytorch"
        assert metadata.optimum_version
        assert metadata.optimum_version[0].isdigit()


def test_upload_keeps_encoder_decoder_file_names(
    tmp_output_dir: str, memory_operator: opendal.Operator
) -> None:
    """A seq2seq export lands where Hephaestus's storage tier looks for it."""
    # Arrange -- the files main_export writes for a seq2seq model.
    names = [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_model_merged.onnx",
        "decoder_with_past_model.onnx",
        "tokenizer.json",
        "config.json",
        "generation_config.json",
    ]
    for name in names:
        with open(os.path.join(tmp_output_dir, name), "wb") as f:
            f.write(name.encode())
    model_id = "org/my-seq2seq"

    # Act
    paths = upload_to_storage(memory_operator, model_id, tmp_output_dir)

    # Assert -- {model_id}/{filename}, the layout Hephaestus downloads.
    assert sorted(paths) == sorted(f"{model_id}/{name}" for name in names)
    assert bytes(memory_operator.read(f"{model_id}/encoder_model.onnx")) == b"encoder_model.onnx"
