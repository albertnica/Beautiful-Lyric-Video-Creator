"""FastAPI application for Beautiful Lyric Video Creator Web Studio.

Implements dependency-injected endpoints (`Depends(get_studio_service)`), strict Pydantic V2
`response_model` contracts, centralized global exception handlers, and Windows Proactor
connection-reset suppression during HTML5 video scrubbing.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
import threading
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from lyric_video_creator.config import DEFAULT_CONFIG
from lyric_video_creator.server.models import (
    ApiErrorResponse,
    ApplyPresetRequest,
    ConfigResponse,
    ConfigUpdateRequest,
    DeleteVideosRequest,
    DeleteVideosResponse,
    OpenFolderRequest,
    OpenFolderResponse,
    PlayExternalRequest,
    PlayExternalResponse,
    PresetListResponse,
    SetFolderRequest,
    SetFolderResponse,
    ShutdownResponse,
    SingleRenderRequest,
    SingleRenderResponse,
    StartRenderRequest,
    StartRenderResponse,
    StopRenderResponse,
    StudioStatusResponse,
    TrackListResponse,
)
from lyric_video_creator.server.service import StudioService, get_studio_service

# Silence harmless Windows Proactor 10054 ConnectionResetError during HTML5 video range scrubbing
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

        def _safe_call_connection_lost(self: Any, exc: Any) -> None:
            try:
                _orig_call_connection_lost(self, exc)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Manage event loop exception handlers for graceful socket disconnects on Windows."""
    loop = asyncio.get_running_loop()
    default_handler = loop.get_exception_handler()

    def proactor_exception_handler(
        current_loop: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return
        if getattr(exc, "winerror", None) == 10054:
            return
        if default_handler:
            default_handler(current_loop, context)
        else:
            current_loop.default_exception_handler(context)

    loop.set_exception_handler(proactor_exception_handler)
    yield


app = FastAPI(
    title="Beautiful Lyric Video Creator • Web Studio",
    description="Interactive Workspace, Output Video Explorer & Configuration Studio",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------- Centralized Global Exception Handlers -------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Ensure all HTTPExceptions return a consistent `ApiErrorResponse` JSON structure."""
    payload = ApiErrorResponse(
        code=exc.status_code,
        message=str(exc.detail),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format Pydantic V2 request validation failures into `ApiErrorResponse`."""
    payload = ApiErrorResponse(
        code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message="Request validation failed.",
        detail=str(exc.errors()),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=payload.model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler preventing raw stack trace leaks in API responses."""
    payload = ApiErrorResponse(
        code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected internal server error occurred.",
        detail=str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=payload.model_dump(),
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(DEFAULT_CONFIG.output_folder).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ServiceDep = Annotated[StudioService, Depends(get_studio_service)]


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Return empty 204 to suppress browser favicon 404 logs."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    """Serve the single-page Web Studio interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=404, detail="Web Studio frontend index.html not found.")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/status", response_model=StudioStatusResponse)
async def get_studio_status(service: ServiceDep) -> StudioStatusResponse:
    """Return current folder statistics, hardware backends, render progress, and output videos."""
    return await asyncio.to_thread(service.get_status)


@app.get("/api/tracks", response_model=TrackListResponse)
async def get_available_tracks(service: ServiceDep) -> TrackListResponse:
    """Return discovered audio tracks in the active music folder with companion LRC status."""
    return await asyncio.to_thread(service.list_available_tracks)


@app.get("/api/presets", response_model=PresetListResponse)
async def get_presets(service: ServiceDep) -> PresetListResponse:
    """Return standard configuration presets alongside active matching preset."""
    return await asyncio.to_thread(service.get_presets)


@app.post("/api/presets/apply", response_model=ConfigResponse)
async def apply_preset(req: ApplyPresetRequest, service: ServiceDep) -> ConfigResponse:
    """Apply a standard preset to the active rendering configuration."""
    return await asyncio.to_thread(service.apply_preset, req.preset_id)


@app.get("/api/config", response_model=ConfigResponse)
async def get_studio_config(service: ServiceDep) -> ConfigResponse:
    """Return the active rendering configuration alongside the factory default configuration."""
    return await asyncio.to_thread(service.get_config_response)


@app.put("/api/config", response_model=ConfigResponse)
async def update_studio_config(
    req: ConfigUpdateRequest,
    service: ServiceDep,
) -> ConfigResponse:
    """Validate and save updated rendering configuration parameters."""
    return await asyncio.to_thread(service.update_config, req)


@app.post("/api/config/reset", response_model=ConfigResponse)
async def reset_studio_config(service: ServiceDep) -> ConfigResponse:
    """Reset all rendering configuration parameters to their original default values."""
    return await asyncio.to_thread(service.reset_config)


@app.post("/api/open-folder", response_model=OpenFolderResponse)
async def open_project_folder(
    req: OpenFolderRequest,
    service: ServiceDep,
) -> OpenFolderResponse:
    """Open the requested project directory (`songs`, `lyrics`, `output`, `logs`) in Windows Explorer."""
    folder_path = await asyncio.to_thread(service.open_folder_in_explorer, req.target)
    return OpenFolderResponse(target=req.target, path=str(folder_path))


@app.post("/api/folders/set", response_model=SetFolderResponse)
async def set_active_folder(
    req: SetFolderRequest,
    service: ServiceDep,
) -> SetFolderResponse:
    """Update an active media directory to a custom path or restore default."""
    return await asyncio.to_thread(service.set_folder, req.target, req.path)


@app.post("/api/render/start", response_model=StartRenderResponse)
async def start_video_creation(
    service: ServiceDep,
    req: StartRenderRequest | None = None,
) -> StartRenderResponse:
    """Trigger background batch video creation across all tracks in `songs/`."""
    effective_req = req or StartRenderRequest()
    return await asyncio.to_thread(
        service.start_batch_render,
        effective_req.backend,
        effective_req.encoder,
        effective_req.fps,
    )


@app.post("/api/render/single", response_model=SingleRenderResponse)
async def start_single_video_creation(
    req: SingleRenderRequest,
    service: ServiceDep,
) -> SingleRenderResponse:
    """Trigger background video creation for a single audio track with optional LRC selection."""
    return await asyncio.to_thread(service.start_single_render, req)


@app.post("/api/render/stop", response_model=StopRenderResponse)
async def stop_video_creation(service: ServiceDep) -> StopRenderResponse:
    """Request clean cancellation of an active background batch render."""
    return await asyncio.to_thread(service.stop_batch_render)


@app.post("/api/videos/delete", response_model=DeleteVideosResponse)
async def delete_rendered_videos(
    req: DeleteVideosRequest,
    service: ServiceDep,
) -> DeleteVideosResponse:
    """Delete one or more rendered `.mp4` videos from the `output/` folder."""
    return await asyncio.to_thread(service.delete_videos, req.filenames)


@app.post("/api/videos/play-external", response_model=PlayExternalResponse)
async def play_video_in_os_player(
    req: PlayExternalRequest,
    service: ServiceDep,
) -> PlayExternalResponse:
    """Launch a rendered `.mp4` video in the user's native OS media player."""
    try:
        await asyncio.to_thread(service.open_video_external, req.filename)
        return PlayExternalResponse(filename=req.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/shutdown", response_model=ShutdownResponse)
async def shutdown_studio_server() -> ShutdownResponse:
    """Terminate the background Web Studio server process cleanly from the UI."""

    def _delayed_exit() -> None:
        os._exit(0)

    threading.Timer(0.35, _delayed_exit).start()
    return ShutdownResponse(message="Web Studio server stopped.")
