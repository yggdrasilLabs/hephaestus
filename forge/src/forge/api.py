"""FastAPI router for the /convert endpoint."""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException, Request

from forge.converter import ConversionError
from forge.models import ConvertRequest, ConvertResponse

logger = structlog.get_logger()

router = APIRouter()


@router.post("/convert", response_model=ConvertResponse)
async def convert(body: ConvertRequest, request: Request) -> ConvertResponse:
    """Accept a conversion request and return storage paths with metadata.

    Pydantic validates ``model_id`` automatically via the
    :class:`ConvertRequest` field validator.  The actual conversion is
    delegated to the :class:`ConversionQueue` stored on ``app.state``.
    """
    queue = request.app.state.queue
    settings = request.app.state.settings

    try:
        return await queue.convert(body.model_id, settings)
    except ConversionError as exc:
        logger.error("conversion_failed", model_id=body.model_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        logger.error("conversion_timeout", model_id=body.model_id)
        raise HTTPException(
            status_code=500,
            detail=f"conversion timed out for model '{body.model_id}'",
        ) from exc
    except Exception as exc:
        logger.error(
            "conversion_unexpected_error",
            model_id=body.model_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail=f"unexpected error during conversion: {exc}",
        ) from exc
