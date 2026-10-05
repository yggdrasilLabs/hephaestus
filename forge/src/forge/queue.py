"""Sequential conversion queue with per-model deduplication."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections import OrderedDict

import structlog

from forge.config import ForgeSettings
from forge.converter import convert_model, validate_model
from forge.models import ConvertResponse
from forge.storage import build_operator, upload_to_storage

logger = structlog.get_logger()


class ConversionQueue:
    """Process one conversion at a time, deduplicating concurrent requests.

    * ``asyncio.Semaphore(1)`` ensures only one conversion runs at a
      time (D-10).
    * A per-``model_id`` ``asyncio.Lock`` ensures that concurrent
      requests for the same model block and receive the cached result
      rather than triggering duplicate work (D-08).
    * Results are cached with LRU eviction at ``MAX_CACHED`` entries
      to prevent unbounded memory growth.
    """

    MAX_CACHED = 256

    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._locks: dict[str, asyncio.Lock] = {}
        self._results: OrderedDict[str, ConvertResponse] = OrderedDict()

    async def convert(
        self, model_id: str, settings: ForgeSettings
    ) -> ConvertResponse:
        """Convert *model_id* to ONNX, returning cached results when available."""
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        lock = self._locks[model_id]
        async with lock:
            # D-08: Return cached result if already converted.
            if model_id in self._results:
                logger.info("conversion_cache_hit", model_id=model_id)
                self._results.move_to_end(model_id)
                return self._results[model_id]

            # D-10: Only one conversion at a time.
            async with self._semaphore:
                output_dir = tempfile.mkdtemp(prefix="forge-")
                try:
                    # Let the conversion run to completion.
                    # Timeout enforcement moves to the HTTP client (Hephaestus
                    # HttpForgeClient already has FORGE_TIMEOUT_SECS).
                    # asyncio.wait_for cannot actually cancel thread pool work
                    # started by asyncio.to_thread -- the thread keeps running
                    # after cancellation, violating the D-10 one-at-a-time
                    # guarantee and risking OOM from concurrent conversions.
                    result = await self._do_convert(model_id, output_dir, settings)
                    self._results[model_id] = result
                    # Evict oldest entries to bound memory.
                    while len(self._results) > self.MAX_CACHED:
                        evicted_id, _ = self._results.popitem(last=False)
                        self._locks.pop(evicted_id, None)
                    return result
                except Exception:
                    # Clean up temp dir on any failure.
                    shutil.rmtree(output_dir, ignore_errors=True)
                    raise

    async def _do_convert(
        self,
        model_id: str,
        output_dir: str,
        settings: ForgeSettings,
    ) -> ConvertResponse:
        """Run conversion, validation, and storage upload in a thread pool."""
        logger.info("conversion_start", model_id=model_id)

        metadata = await asyncio.to_thread(convert_model, model_id, output_dir)
        logger.info("conversion_export_done", model_id=model_id)

        await asyncio.to_thread(validate_model, output_dir)
        logger.info("conversion_validated", model_id=model_id)

        op = build_operator(settings)
        storage_paths = await asyncio.to_thread(
            upload_to_storage,
            op,
            model_id,
            output_dir,
        )
        logger.info(
            "conversion_uploaded",
            model_id=model_id,
            storage_paths=storage_paths,
        )

        # Clean up temp dir after successful upload.
        shutil.rmtree(output_dir, ignore_errors=True)

        return ConvertResponse(storage_paths=storage_paths, metadata=metadata)
