from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import site

# Conda base can contain newer user-site copies of packages also present in the
# environment. Prefer the environment's own wheels, but leave user-site packages
# as a fallback for small transitive dependencies not installed in base.
_user_site = site.getusersitepackages()
_user_paths = [path for path in sys.path if path.lower().startswith(str(_user_site).lower())]
for _path in _user_paths:
    sys.path.remove(_path)
sys.path.extend(_user_paths)

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("laya-demo")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/laya-zh")).expanduser()
MODEL_NAME = os.getenv("MODEL_NAME", "laya-typed-decisions-zh")
LAYA_SOURCE = Path(__file__).resolve().parents[2] / "laya"
if str(LAYA_SOURCE) not in sys.path:
    sys.path.insert(0, str(LAYA_SOURCE))
REQUEST_LOCK = threading.Lock()
agent = None
runtime_device = "unloaded"
gpu_name: str | None = None


def _checkpoint_problems(path: Path) -> list[str]:
    required = (
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/tokenizer_config.json",
    )
    return [name for name in required if not (path / name).is_file()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent, runtime_device, gpu_name

    problems = _checkpoint_problems(MODEL_PATH)
    if problems:
        raise RuntimeError(
            f"Checkpoint is incomplete at {MODEL_PATH}; missing: {', '.join(problems)}. "
            "Mount the training output directory containing model.safetensors and rl_agent_config.json."
        )

    requested_device = os.getenv("DEVICE", "cuda").lower()
    if requested_device not in {"auto", "cuda", "cpu"}:
        raise RuntimeError("DEVICE must be one of: auto, cuda, cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE=cuda but PyTorch cannot see a CUDA GPU. Check the NVIDIA driver, "
            "NVIDIA Container Toolkit, and Compose GPU reservation."
        )
    selected_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if selected_device == "auto":
        selected_device = "cpu"

    with (MODEL_PATH / "rl_agent_config.json").open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    if not cfg.get("fine_tuned"):
        logger.warning("Checkpoint config does not mark fine_tuned=true: %s", MODEL_PATH.name)

    logger.info("Loading Laya checkpoint %s on %s", MODEL_PATH, selected_device)
    import laya

    agent = laya.load(str(MODEL_PATH), device=selected_device)
    agent.model.eval()
    runtime_device = str(agent.device)
    if agent.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(agent.device)
    logger.info("Laya ready: model=%s device=%s gpu=%s", MODEL_NAME, runtime_device, gpu_name or "n/a")
    yield
    agent = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Laya 中文场景验证 API",
    version="1.0.0",
    description="在本机加载挂载的中文 Laya checkpoint；推理请求不会发送到外部服务。",
    lifespan=lifespan,
)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str | None = Field(default=None, max_length=80)
    model: str | None = Field(default=None, max_length=120)
    state: Any
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=32)

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: Any) -> Any:
        if not isinstance(value, (str, dict, list)):
            raise ValueError("state 必须是字符串、JSON 对象或对话列表")
        if len(json.dumps(value, ensure_ascii=False)) > 30000:
            raise ValueError("state 太长；最多支持 30,000 个字符")
        return value


@app.get("/api/health")
def health() -> dict[str, Any]:
    ready = agent is not None
    return {
        "status": "ok" if ready else "loading",
        "model": MODEL_NAME,
        "checkpoint": MODEL_PATH.name,
        "device": runtime_device,
        "gpu": gpu_name,
    }


@app.post("/api/decisions")
def decisions(request: DecisionRequest) -> dict[str, Any]:
    if agent is None:
        raise HTTPException(status_code=503, detail="Laya checkpoint 尚未加载完成")

    started = time.perf_counter()
    try:
        # Keep a single in-flight forward pass to avoid GPU-memory spikes and to make
        # this one-checkpoint demo predictable under concurrent browser requests.
        with REQUEST_LOCK:
            if agent.device.type == "cuda":
                torch.cuda.synchronize(agent.device)
            prediction = agent.predict(request.state, request.questions, lang="zh")
            if agent.device.type == "cuda":
                torch.cuda.synchronize(agent.device)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.exception("CUDA ran out of memory")
        raise HTTPException(status_code=507, detail="CUDA 显存不足；请缩短 state 或减少问题数量") from exc
    except Exception as exc:
        logger.exception("Laya inference failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return {
        **prediction,
        "model": MODEL_NAME,
        "scenario_id": request.scenario_id,
        "device": runtime_device,
        "gpu": gpu_name,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
