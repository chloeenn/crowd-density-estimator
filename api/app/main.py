"""App factory.

    uvicorn --factory app.main:create_app --app-dir api

No module-level app: importing this module has no side effects, and tests build their own.
"""

from __future__ import annotations

import asyncio
import re
import uuid
import warnings
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.formparsers import MultiPartParser

from app.config import Settings
from app.errors import error_response, install_error_handlers
from app.inference import InferenceService
from app.logging_setup import setup_logging
from app.routers import predict, system

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
BODY_OVERHEAD_BYTES = 1024 * 1024  # multipart framing and the other form fields


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    log = setup_logging(settings.log_level)
    # The per-app pixel limit (settings.max_image_pixels) is enforced explicitly in
    # `decode_image`; we deliberately do not write Pillow's process-global MAX_IMAGE_PIXELS,
    # which would make the last app created win. Pillow's own defaults (warn above 89 MP,
    # raise above 178 MP) remain as a second line of defence. In the warn-only band our check
    # already returns a clean 413, so the warning would just be log noise.
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    # Starlette spools multipart files larger than 1 MB to a temp file on disk. Raise the
    # threshold above our upload limit so an upload is only ever held in memory. This is
    # process-wide state, so it only ever goes up: creating a second app (as tests do) can
    # never lower the guarantee for one that is already running.
    MultiPartParser.spool_max_size = max(
        MultiPartParser.spool_max_size, settings.max_upload_bytes + BODY_OVERHEAD_BYTES
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load before serving: a bad checkpoint (missing, wrong part, preprocessing mismatch)
        # makes startup fail, so the server refuses to start instead of serving wrong answers.
        app.state.service = await asyncio.to_thread(InferenceService.from_settings, settings)
        log.info("ready: models %s", sorted(app.state.service.predictors))
        yield

    app = FastAPI(title="Crowd Density API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Request id, upload-size pre-check, no-store caching, and one JSON log line."""
        supplied = request.headers.get("x-request-id", "")
        rid = supplied if REQUEST_ID_RE.match(supplied) else uuid.uuid4().hex
        request.state.request_id = rid
        start = perf_counter()
        length = request.headers.get("content-length", "")
        too_big = length.isdigit() and int(length) > settings.max_upload_bytes + BODY_OVERHEAD_BYTES
        try:
            if too_big:  # refuse before reading the body
                response = error_response(
                    413, "payload_too_large", f"File is larger than {settings.max_upload_mb} MB."
                )
            else:
                response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"request_id": rid})
            response = error_response(500, "internal_error", "Internal server error.")
        response.headers["X-Request-ID"] = rid
        response.headers["Cache-Control"] = "no-store"  # results describe a private upload
        log.info(
            "request",
            extra={
                "request_id": rid, "method": request.method, "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((perf_counter() - start) * 1000),
                "inference_ms": getattr(request.state, "inference_ms", None),
                "part": getattr(request.state, "part", None),
            },
        )  # fmt: skip
        return response

    app.add_middleware(  # added last = outermost, so CORS headers also cover error responses
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.include_router(predict.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    return app
