"""Tests for forge.storage -- OpenDAL-based upload logic."""

from __future__ import annotations

import os
import pathlib

import opendal
import pytest

from forge.config import ForgeSettings
from forge.storage import build_operator, storage_options, upload_to_storage


def test_upload_to_storage_writes_files(
    populated_output_dir: str, memory_operator: opendal.Operator
) -> None:
    """Uploaded paths follow the {model_id}/{filename} layout."""
    model_id = "org/my-model"
    paths = upload_to_storage(memory_operator, model_id, populated_output_dir)

    assert len(paths) == 3
    for path in paths:
        assert path.startswith(f"{model_id}/")
    assert f"{model_id}/model.onnx" in paths
    assert f"{model_id}/tokenizer.json" in paths
    assert f"{model_id}/config.json" in paths

    # Verify files are readable from the operator.
    for path in paths:
        data = memory_operator.read(path)
        assert len(data) > 0


def test_upload_paths_contain_model_id(
    populated_output_dir: str, memory_operator: opendal.Operator
) -> None:
    """All uploaded paths start with model_id and never with a leading slash."""
    model_id = "org/my-model"
    paths = upload_to_storage(memory_operator, model_id, populated_output_dir)

    assert len(paths) == 3
    for path in paths:
        assert path.startswith(f"{model_id}/")
        assert not path.startswith("/")


def test_uploaded_files_are_readable(
    populated_output_dir: str, memory_operator: opendal.Operator
) -> None:
    """Files uploaded via upload_to_storage can be read back."""
    model_id = "org/my-model"
    paths = upload_to_storage(memory_operator, model_id, populated_output_dir)

    for path in paths:
        data = memory_operator.read(path)
        assert len(data) > 0


def test_upload_includes_subdirectories(
    populated_output_dir: str, memory_operator: opendal.Operator
) -> None:
    """Files in subdirectories are included in the upload (recursive walk)."""
    subdir = os.path.join(populated_output_dir, "onnx")
    os.makedirs(subdir)
    with open(os.path.join(subdir, "model.onnx"), "wb") as f:
        f.write(b"onnx-subdir-model")

    model_id = "org/my-model"
    paths = upload_to_storage(memory_operator, model_id, populated_output_dir)

    # 3 original files + 1 file in onnx/ subdirectory.
    assert len(paths) == 4
    assert f"{model_id}/onnx/model.onnx" in paths


def _settings(
    storage_type: str,
    *,
    storage_root: str = "/data/models",
    storage_credential_path: str = "/nonexistent/key.json",
) -> ForgeSettings:
    """Build ForgeSettings for *storage_type* with every storage field populated."""
    return ForgeSettings(
        storage_type=storage_type,
        storage_bucket="test-bucket",
        storage_prefix="",
        storage_root=storage_root,
        storage_region="us-east-1",
        storage_credential_path=storage_credential_path,
        conversion_timeout_secs=60,
        log_level="debug",
    )


@pytest.mark.parametrize("storage_type", ["s3", "gcs", "fs"])
def test_build_operator_supports_storage_type(
    storage_type: str, tmp_path: pathlib.Path
) -> None:
    """An Operator builds offline for every supported storage backend."""
    # Arrange
    settings = _settings(storage_type, storage_root=str(tmp_path))

    # Act
    op = build_operator(settings)

    # Assert
    assert isinstance(op, opendal.Operator)


def test_storage_options_passes_region_only_to_s3() -> None:
    """The region option reaches the s3 backend and no other."""
    # Arrange / Act
    s3 = storage_options(_settings("s3"))
    gcs = storage_options(_settings("gcs"))
    fs = storage_options(_settings("fs"))

    # Assert
    assert s3["region"] == "us-east-1"
    assert "region" not in gcs, f"gcs options must not carry a region: {gcs}"
    assert "region" not in fs, f"fs options must not carry a region: {fs}"


def test_storage_options_passes_credential_path_only_to_gcs() -> None:
    """The credential_path option reaches the gcs backend and no other."""
    # Arrange / Act
    gcs = storage_options(_settings("gcs"))
    gcs_default = storage_options(_settings("gcs", storage_credential_path=""))
    s3 = storage_options(_settings("s3"))
    fs = storage_options(_settings("fs"))

    # Assert
    assert gcs["credential_path"] == "/nonexistent/key.json"
    assert "credential_path" not in gcs_default, (
        "empty STORAGE_CREDENTIAL_PATH must fall back to ADC / Workload Identity"
    )
    assert "credential_path" not in s3, f"s3 options must not carry a credential_path: {s3}"
    assert "credential_path" not in fs, f"fs options must not carry a credential_path: {fs}"
