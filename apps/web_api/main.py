"""
Web API - FastAPI application entry point.

Start:
    uv run uvicorn apps.web_api.main:app --host 0.0.0.0 --port 2342 --reload
"""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from shared.database import connect_db
from routers.kiem_tra_in import router as kiem_tra_in_router
import log_broker

# -- Logging -----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
log_broker.install_log_handler(logging.INFO)

# -- App ---------------------------------------------------------------------
app = FastAPI(
    title="DK Python IMS - Web API",
    version="0.2.0",
    description="API noi bo: Kiem tra chat luong in an bang AI",
)

# -- CORS (cho phép Laravel ERP gọi SSE cross-origin) ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # hoặc chỉ định domain ERP cụ thể nếu muốn bảo mật hơn
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Static ------------------------------------------------------------------
_here = Path(__file__).parent
_static_dir = _here / "static"
_static_dir.mkdir(exist_ok=True)
_dashboard_file = _here / "templates" / "dashboard.html"

app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# -- DB ----------------------------------------------------------------------
_db = connect_db()


@app.on_event("startup")
async def startup_event():
    logger.info("=== DK Python IMS Web API khoi dong ===")
    logger.info("Dashboard: http://localhost:2342/")
    logger.info("API Docs:  http://localhost:2342/docs")
    log_broker.push("Service khoi dong thanh cong", level="OK")


# -- Include Routers ---------------------------------------------------------
app.include_router(kiem_tra_in_router)


# -- Dashboard UI (serve raw HTML, khong dung Jinja2) -----------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Giao dien dashboard."""
    html = _dashboard_file.read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# -- Health check ------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "db_connected": _db["connected"], "version": "0.2.0"}


@app.get("/db-status")
def db_status() -> dict:
    return {"db": _db}


# -- SSE Log Stream ----------------------------------------------------------
@app.get("/api/log-stream", include_in_schema=False)
async def log_stream(request: Request):
    """Server-Sent Events: push log events den dashboard."""
    queue = log_broker.subscribe()

    async def event_generator():
        yield 'data: {"level":"OK","message":"SSE stream san sang","time":"now"}\n\n'
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=60.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            log_broker.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Connection":                  "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


# -- Preview image -----------------------------------------------------------
@app.get("/api/preview-image", include_in_schema=False)
async def preview_image(path: str):
    p = Path(path)
    if not p.exists():
        return {"error": "File not found"}
    return FileResponse(str(p))
