from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.api.prediction import router as prediction_router, runtime
from app.config import FRONTEND_DIST_DIR


@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.initialize()
    yield


app = FastAPI(title="서울대공원 눈치싸움", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(prediction_router, prefix="/api")

if FRONTEND_DIST_DIR.exists():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")


@app.get("/{full_path:path}")
def frontend_app(full_path: str):
    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        return PlainTextResponse("프론트엔드 빌드 결과가 없습니다. frontend/dist를 생성해 주세요.", status_code=503)
    return FileResponse(index_file)
