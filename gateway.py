# gateway.py
from __future__ import annotations

import io
import os
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

# ------------------------------------------------------------------------------
# Environment / config
# ------------------------------------------------------------------------------
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-OCR")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
NGRAM_SIZE = int(os.getenv("NGRAM_SIZE", "30"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "90"))
WHITELIST_TOKEN_IDS = os.getenv("WHITELIST_TOKEN_IDS", "128821,128822")  # <td>, </td>
DOWNLOAD_DIR = os.getenv("VLLM_DOWNLOAD_DIR", os.getenv("HF_HOME", "/models"))
TP_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
DEFAULT_PROMPT = os.getenv("DEFAULT_PROMPT", "<image>\nFree OCR.")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# vLLM tuning knobs
GPU_MEM_UTIL = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5"))  # e.g. 0.3–0.6
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "4096"))  # reduce if you want smaller KV cache


def _parse_whitelist(s: str) -> set[int]:
    """Parse a comma-separated list of ints into a set."""
    try:
        return {int(x.strip()) for x in s.split(",") if x.strip()}
    except Exception:
        return set()


whitelist_token_ids = _parse_whitelist(WHITELIST_TOKEN_IDS)

# ------------------------------------------------------------------------------
# Decide where to load model from (local vs HF)
# ------------------------------------------------------------------------------

def _pick_model_source() -> str:
    """
    If download_model.py has already put a full HF snapshot in DOWNLOAD_DIR,
    prefer that local path. Otherwise fall back to MODEL_ID (HF).
    """
    # Heuristic: if there's a config.json in DOWNLOAD_DIR, assume it's a full repo
    local_config = os.path.join(DOWNLOAD_DIR, "config.json")
    if os.path.isdir(DOWNLOAD_DIR) and os.path.exists(local_config):
        print(f"[gateway] Using local model at: {DOWNLOAD_DIR}", flush=True)
        return DOWNLOAD_DIR

    # If user explicitly pointed to a local path via MODEL_PATH, honor that
    model_path_env = os.getenv("MODEL_PATH")
    if model_path_env and os.path.exists(model_path_env):
        print(f"[gateway] Using MODEL_PATH={model_path_env}", flush=True)
        return model_path_env

    # Fallback: use HF repo id
    print(f"[gateway] Using HF model id: {MODEL_ID} (download_dir={DOWNLOAD_DIR})", flush=True)
    return MODEL_ID


MODEL_SOURCE = _pick_model_source()

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(
    title="DeepSeek-OCR Gateway",
    version="1.0",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# vLLM engine (initialize once per process, using pre-downloaded model if present)
# ------------------------------------------------------------------------------
llm = LLM(
    model=MODEL_SOURCE,          # local path or HF id
    tensor_parallel_size=TP_SIZE,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    download_dir=DOWNLOAD_DIR,   # vLLM will reuse /models; no re-download needed
    logits_processors=[NGramPerReqLogitsProcessor],
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_model_len=MAX_MODEL_LEN,
    # HF token (if needed) is picked up from env by vLLM / huggingface_hub.
)

# ------------------------------------------------------------------------------
# Models / helpers
# ------------------------------------------------------------------------------
class OCRResponse(BaseModel):
    texts: List[str]


def _pil_from_upload(file: UploadFile) -> Image.Image:
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("empty file")
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image '{file.filename}': {e}",
        ) from e


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "DeepSeek-OCR gateway is running."}


@app.get("/healthz")
def healthz():
    # quick engine probe (no heavy calls)
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_source": MODEL_SOURCE,
        "tensor_parallel_size": TP_SIZE,
        "download_dir": DOWNLOAD_DIR,
        "has_hf_token": bool(HF_TOKEN),
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "max_model_len": MAX_MODEL_LEN,
    }


@app.post("/ocr", response_model=OCRResponse)
def ocr(
    files: List[UploadFile] = File(..., description="One or more image files"),
    prompt: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images = [_pil_from_upload(f) for f in files]
    user_prompt = (prompt or DEFAULT_PROMPT).strip()

    # vLLM multi-modal inputs (one per image keeps ordering)
    model_inputs = [
        {"prompt": user_prompt, "multi_modal_data": {"image": img}}
        for img in images
    ]

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_args=dict(
            ngram_size=NGRAM_SIZE,
            window_size=WINDOW_SIZE,
            whitelist_token_ids=whitelist_token_ids,
        ),
        skip_special_tokens=False,
    )

    try:
        outputs = llm.generate(model_inputs, sampling)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
        else:
            texts.append(out.outputs[0].text)

    return OCRResponse(texts=texts)


if __name__ == "__main__":
    import uvicorn

    port_env = os.getenv("PORT", "8000")
    try:
        port = int(port_env)
    except ValueError:
        port = 8000

    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
