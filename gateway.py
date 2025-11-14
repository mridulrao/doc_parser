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

# parse whitelist token ids into a set[int]
def _parse_whitelist(s: str) -> set[int]:
    try:
        return {int(x.strip()) for x in s.split(",") if x.strip()}
    except Exception:
        return set()


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
# vLLM engine (initialize once per process)
# ------------------------------------------------------------------------------
# Keep a single LLM instance per GPU process.
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=TP_SIZE,
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    download_dir=DOWNLOAD_DIR,
    # DeepSeek-OCR needs this logits processor available:
    logits_processors=[NGramPerReqLogitsProcessor],
    # If repo is gated/private, ensure HF token is in env; vLLM will pick it up.
)

whitelist_token_ids = _parse_whitelist(WHITELIST_TOKEN_IDS)


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
        raise HTTPException(status_code=400, detail=f"Invalid image '{file.filename}': {e}") from e


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    # quick engine probe (no heavy calls)
    return {
        "status": "ok",
        "model": MODEL_ID,
        "tensor_parallel_size": TP_SIZE,
        "download_dir": DOWNLOAD_DIR,
        "has_hf_token": bool(HF_TOKEN),
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
        # DeepSeek-OCR specific args (dedup/cleanup over sliding window)
        extra_args=dict(
            ngram_size=NGRAM_SIZE,
            window_size=WINDOW_SIZE,
            whitelist_token_ids=whitelist_token_ids,
        ),
        # Keep special tokens if you want raw structure; set True for plain text.
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
