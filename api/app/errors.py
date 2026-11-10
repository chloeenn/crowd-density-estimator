"""One error shape for the whole API: `{"detail": <readable message>, "code": <slug>}`.

Clients never see a stack trace; unexpected failures are logged server-side and returned as a
generic 500.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ApiError(Exception):
    """An expected failure with the HTTP status and message to return."""

    def __init__(self, status_code: int, code: str, detail: str, headers: dict | None = None):
        super().__init__(detail)
        self.status_code, self.code, self.detail, self.headers = (
            status_code,
            code,
            detail,
            headers,
        )


def error_response(status: int, code: str, detail: str, headers: dict | None = None):
    return JSONResponse({"detail": detail, "code": code}, status_code=status, headers=headers)


def _validation_message(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = str(err["loc"][-1])
        if field == "file" and err["type"] == "missing":
            parts.append("Missing 'file': send the image as multipart form data.")
        else:
            parts.append(f"{field}: {err['msg']}")
    return " ".join(parts)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status_code, exc.code, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return error_response(422, "invalid_request", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, "http_error", str(exc.detail))
