"""Tests for forge.api -- FastAPI endpoint contract."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from forge.config import ForgeSettings
from forge.main import create_app
from forge.models import ConversionMetadata, ConvertResponse
from forge.queue import ConversionQueue


@pytest.fixture()
def app():
    """Create the app and manually set up state (lifespan is not triggered
    by ASGITransport)."""
    application = create_app()
    application.state.settings = ForgeSettings(
        storage_type="memory",
        storage_bucket="",
        storage_prefix="models",
        storage_root="",
        storage_region="",
        conversion_timeout_secs=60,
    )
    application.state.queue = ConversionQueue()
    return application


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealthEndpoint:
    async def test_health_returns_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestConvertEndpoint:
    async def test_convert_success(self, client: AsyncClient) -> None:
        """POST /convert with valid model_id returns 200 + ConvertResponse."""
        canned = ConvertResponse(
            storage_paths=["models/org/test-model/model.onnx"],
            metadata=ConversionMetadata(
                architecture="bert",
                original_format="pytorch",
                conversion_duration_secs=12.5,
                optimum_version="2.3.0",
            ),
        )
        with patch.object(
            ConversionQueue,
            "convert",
            new_callable=AsyncMock,
            return_value=canned,
        ):
            resp = await client.post(
                "/convert", json={"model_id": "org/test-model"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["storage_paths"] == ["models/org/test-model/model.onnx"]
        assert body["metadata"]["architecture"] == "bert"

    async def test_convert_invalid_model_id_traversal(
        self, client: AsyncClient
    ) -> None:
        """model_id containing '..' is rejected with 422."""
        resp = await client.post(
            "/convert", json={"model_id": "../etc/passwd"}
        )
        assert resp.status_code == 422

    async def test_convert_invalid_model_id_empty(
        self, client: AsyncClient
    ) -> None:
        """Empty model_id is rejected with 422."""
        resp = await client.post("/convert", json={"model_id": ""})
        assert resp.status_code == 422

    async def test_convert_invalid_model_id_special_chars(
        self, client: AsyncClient
    ) -> None:
        """model_id with special characters is rejected with 422."""
        resp = await client.post(
            "/convert", json={"model_id": "org/model; rm -rf /"}
        )
        assert resp.status_code == 422

    async def test_convert_conversion_error_returns_500(
        self, client: AsyncClient
    ) -> None:
        """ConversionError during conversion returns HTTP 500."""
        from forge.converter import ConversionError

        with patch.object(
            ConversionQueue,
            "convert",
            new_callable=AsyncMock,
            side_effect=ConversionError("model.onnx not found"),
        ):
            resp = await client.post(
                "/convert", json={"model_id": "org/bad-model"}
            )

        assert resp.status_code == 500
        assert "model.onnx not found" in resp.json()["detail"]

    async def test_convert_timeout_returns_500(
        self, client: AsyncClient
    ) -> None:
        """Timeout during conversion returns HTTP 500."""
        with patch.object(
            ConversionQueue,
            "convert",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError(),
        ):
            resp = await client.post(
                "/convert", json={"model_id": "org/slow-model"}
            )

        assert resp.status_code == 500
        assert "timed out" in resp.json()["detail"]
