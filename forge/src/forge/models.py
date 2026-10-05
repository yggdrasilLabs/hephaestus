"""Pydantic request/response models for the Forge API."""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

# Characters allowed in model_id: alphanumeric, hyphen, underscore, forward
# slash, and period.  Mirrors the Rust ``validate_model_id`` function in
# hephaestus-resolve so both sides enforce the same contract.
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-_./]+$")


class ConvertRequest(BaseModel):
    """Incoming conversion request from Hephaestus."""

    model_id: str

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        if not v:
            raise ValueError("model_id must not be empty")
        if ".." in v:
            raise ValueError("model_id must not contain '..'")
        if not _MODEL_ID_PATTERN.match(v):
            raise ValueError(
                "model_id contains invalid characters; "
                "only alphanumeric, '-', '_', '/', '.' are allowed"
            )
        return v


class ConversionMetadata(BaseModel):
    """Metadata about a completed ONNX conversion."""

    architecture: str
    original_format: str
    conversion_duration_secs: float
    optimum_version: str


class ConvertResponse(BaseModel):
    """Response returned after a successful conversion."""

    storage_paths: list[str]
    metadata: ConversionMetadata
