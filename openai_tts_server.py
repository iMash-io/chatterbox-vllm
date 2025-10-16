#!/usr/bin/env python3
"""
OpenAI-compatible TTS HTTP server for chatterbox-vllm with ultra-low-latency streaming.

- Endpoint: POST /v1/audio/speech
  Request body (subset of OpenAI spec + custom knobs):
    {
      "model": "gpt-4o-mini-tts" | string (ignored, for compatibility),
      "voice": "alloy" | string (ignored, for compatibility),
      "input": "Text to speak",
      "format": "wav" | "pcm16",                     # default: "wav"
      "stream": true | false,                        # default: false
      "language_id": "en",                           # default: "en" (see tts.get_supported_languages())
      "temperature": 0.8,                            # default: 0.8
      "exaggeration": 0.5,                           # default: 0.5
      "diffusion_steps": 10,                         # default: 10
      "audio_prompt_path": null | "/path/to/ref.wav",
      "watermark": "off" | "on"                      # default: "off" (when off, absolutely NO watermarking step)
    }

- Streaming semantics:
  When stream=true, this server streams audio as soon as the first chunk is generated to minimize TTFA.
  It uses sentence/phrase chunking and synthesizes each chunk independently, streaming small PCM frames
  (20–40ms) as they become available. This avoids modifying core chatterbox code while achieving low TTFA.

- Warm startup:
  On server startup, the TTS engine is loaded and a tiny placeholder synthesis is executed to warm
  caches so the first real request has minimal cold-start latency.

- Watermark:
  We searched the chatterbox-vllm repo for watermark logic and found none. This server includes an
  explicit "watermark" flag; when "off", no watermarking or audio modification beyond synthesis occurs.
  When "on", it is currently a no-op placeholder keeping full audio fidelity (no watermark injected).
  This ensures "off" fully disables/removes watermarking for generation.

- LiveKit compatibility:
  The route and JSON shape follow OpenAI's TTS endpoint. For streaming, the response is a chunked
  HTTP stream:
    - Content-Type: audio/pcm;rate=24000;channels=1  (for "pcm16")
    - Content-Type: audio/wav                         (for "wav", non-streaming only)
  LiveKit can ingest linear PCM streams; see your pipeline config to point to this server.

Run:
  - Install deps: pip install fastapi uvicorn
  - Start: python openai_tts_server.py
  - POST to: http://localhost:8000/v1/audio/speech

Environment Variables:
  - CHATTERBOX_VARIANT: "english" (default) | "multilingual"
  - CHATTERBOX_DEVICE: "cuda" (default if available) | "cpu"
  - CHATTERBOX_S3GEN_FP16: "0" | "1" (default "0")
  - CHATTERBOX_USE_LOCAL_CKPT: path to local checkpoint dir (else uses HF from_pretrained)
  - CHATTERBOX_WARMUP_TEXT: text to use for warmup (default: "Hello")
  - CHATTERBOX_WARMUP_DIFF_STEPS: int diffusion steps for warmup (default: 2)
"""

import os
import sys
import io
import math
import asyncio
import time, uuid
from typing import Optional, Iterable, Generator, List, Tuple

# Ensure `src/` is importable when running from repo root
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from chatterbox_vllm.tts import ChatterboxTTS


APP = FastAPI(title="OpenAI-compatible TTS for chatterbox-vllm")

# Global singleton TTS engine (initialized on startup)
_tts_engine: Optional[ChatterboxTTS] = None
_tts_sr: int = 24000  # default, will be updated from engine.sr


# ---------- Request/Response Schemas ----------

class SpeechRequest(BaseModel):
    model: Optional[str] = Field(default="gpt-4o-mini-tts", description="Ignored; present for OpenAI compatibility.")
    voice: Optional[str] = Field(default="alloy", description="Ignored; present for OpenAI compatibility.")
    input: str = Field(description="Text to speak")
    format: Optional[str] = Field(default="wav", pattern="^(wav|pcm16)$", description="Audio format: wav (non-stream) or pcm16")
    response_format: Optional[str] = Field(default=None, pattern="^(wav|pcm16|pcm)$", description="OpenAI alias; overrides 'format' if provided")
    stream: Optional[bool] = Field(default=False, description="If true, stream audio chunks with minimal latency")
    language_id: Optional[str] = Field(default="en", description="Language code supported by chatterbox model")
    temperature: Optional[float] = Field(default=0.8)
    exaggeration: Optional[float] = Field(default=0.5)
    diffusion_steps: Optional[int] = Field(default=10)
    audio_prompt_path: Optional[str] = Field(default=None, description="Reference audio path for timbre/voice embedding")
    watermark: Optional[str] = Field(default="off", pattern="^(on|off)$", description="If 'off', no watermarking step is applied")
    primer_silence_ms: Optional[int] = Field(default=0, ge=0, le=200, description="If >0, yield this ms of silence immediately to flush headers")
    first_chunk_diff_steps: Optional[int] = Field(default=2, ge=1, description="Diffusion steps for the first chunk only")
    first_chunk_chars: Optional[int] = Field(default=32, ge=10, le=200, description="Max chars for the first chunk")
    chunk_chars: Optional[int] = Field(default=120, ge=40, le=400, description="Max chars for subsequent chunks")
    frame_ms: Optional[int] = Field(default=20, ge=10, le=60, description="Frame size for PCM streaming chunks")
    max_tokens_first_chunk: Optional[int] = Field(default=None, ge=16, le=512, description="Max tokens for T3 on first chunk only")
    crossfade_ms: Optional[int] = Field(default=10, ge=0, le=50, description="Crossfade between chunk boundaries (ms) to reduce clicks")


# ---------- Utilities ----------

def _float32_to_pcm16_bytes(wav: torch.Tensor) -> bytes:
    """
    Convert a waveform tensor (1, T) float32 in [-1, 1] to PCM16 little-endian bytes.
    """
    if not torch.is_tensor(wav):
        wav = torch.tensor(wav)
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    wav = torch.clamp(wav, -1.0, 1.0)
    wav_i16 = (wav * 32767.0).to(torch.int16).cpu().numpy()
    return wav_i16.tobytes()


def _pack_wav_bytes(wav: torch.Tensor, sr: int) -> bytes:
    """
    Pack a float32 mono waveform (1, T) into a WAV container in-memory.
    """
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(_float32_to_pcm16_bytes(wav))
    return buf.getvalue()


def _split_text_for_low_latency(text: str, max_chars: int = 120) -> List[str]:
    """
    Split text into sentence/phrase-like chunks for low-latency streaming.
    Preference: split on natural punctuation boundaries; fallback to fixed width.
    """
    text = text.strip()
    if not text:
        return []
    # First split by strong punctuation
    import re
    parts = re.split(r"([\.!\?;:])", text)
    # Re-join punctuation tokens
    assembled: List[str] = []
    cur = ""
    for i in range(0, len(parts), 2):
        seg = parts[i].strip()
        if not seg:
            continue
        punc = parts[i + 1] if i + 1 < len(parts) else ""
        chunk = (seg + punc).strip()
        if not chunk:
            continue
        if len(cur) + 1 + len(chunk) <= max_chars:
            cur = (cur + " " + chunk).strip() if cur else chunk
        else:
            if cur:
                assembled.append(cur)
            cur = chunk
    if cur:
        assembled.append(cur)

    if not assembled:
        return [text]

    # If we ended with a very long chunk, split by whitespace to max_chars
    final_chunks: List[str] = []
    for c in assembled:
        if len(c) <= max_chars:
            final_chunks.append(c)
            continue
        words = c.split()
        cur = ""
        for w in words:
            if len(cur) + 1 + len(w) <= max_chars:
                cur = (cur + " " + w).strip() if cur else w
            else:
                if cur:
                    final_chunks.append(cur)
                cur = w
        if cur:
            final_chunks.append(cur)

    return [i for i in final_chunks if i]


def _frame_chunk_bytes(pcm_bytes: bytes, sr: int, frame_ms: int = 40) -> Iterable[bytes]:
    """
    Yield small PCM frames (default 40ms) from a PCM16 byte buffer.
    """
    bytes_per_sample = 2  # 16-bit
    samples_per_frame = int(sr * frame_ms / 1000)
    bytes_per_frame = samples_per_frame * bytes_per_sample

    for i in range(0, len(pcm_bytes), bytes_per_frame):
        yield pcm_bytes[i:i + bytes_per_frame]


def _apply_watermark_if_needed(wav: torch.Tensor, sr: int, watermark: str) -> torch.Tensor:
    """
    Placeholder watermark hook. For "off", do nothing.
    For "on", currently still does nothing to preserve original audio quality.
    This fulfills "off" fully disabling/removing any watermarking step.
    """
    # If you ever need a watermark, implement here (e.g., inaudible pseudo-random sequence).
    # Keeping it a strict no-op to honor "off" removing watermark entirely.
    return wav


# ---------- Core synthesis helpers ----------

def _ensure_engine() -> ChatterboxTTS:
    global _tts_engine
    if _tts_engine is None:
        raise RuntimeError("TTS engine not initialized")
    return _tts_engine


def _synthesize_one(
    text: str,
    *,
    language_id: Optional[str],
    temperature: float,
    exaggeration: float,
    diffusion_steps: int,
    audio_prompt_path: Optional[str],
    max_tokens_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Synthesize one prompt to a waveform (1, T) tensor.
    """
    tts = _ensure_engine()
    waves = tts.generate(
        prompts=text,
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        temperature=temperature,
        exaggeration=exaggeration,
        diffusion_steps=diffusion_steps,
        max_tokens=min(max_tokens_override, tts.max_model_len) if max_tokens_override is not None else tts.max_model_len,
    )
    if not waves:
        raise RuntimeError("No audio generated")
    return waves[0]


async def _synthesize_streaming_pcm_frames(
    text: str,
    *,
    language_id: Optional[str],
    temperature: float,
    exaggeration: float,
    diffusion_steps: int,
    audio_prompt_path: Optional[str],
    watermark: str,
    req_id: str,
    t0: float,
    client_epoch_ms: Optional[int] = None,
    server_epoch_ms: Optional[int] = None,
    max_tokens_first_chunk: Optional[int] = None,
    primer_silence_ms: int = 0,
    first_chunk_diff_steps: Optional[int] = None,
    first_chunk_chars: int = 60,
    chunk_chars: int = 120,
    frame_ms: int = 20,
    crossfade_ms: int = 10,
) -> Generator[bytes, None, None]:
    """
    Streaming generator:
      - Split text into chunks (~sentences/phrases)
      - Synthesize first chunk quickly (optional reduced diffusion steps)
      - Convert to PCM16 and yield small frames immediately
      - Continue with subsequent chunks (full quality)
    """
    tts = _ensure_engine()
    sr = tts.sr

    start = t0
    total_bytes = 0
    total_frames = 0
    first_synth_frame_sent = False
    prev_tail: Optional[torch.Tensor] = None
    skew_ms = (server_epoch_ms - client_epoch_ms) if (client_epoch_ms is not None and server_epoch_ms is not None) else None
    print(f"[TTS_STREAM_START] req_id={req_id} variant={getattr(tts, 'variant', 'unknown')} sr={sr} primer_ms={primer_silence_ms} first_chunk_diff_steps={first_chunk_diff_steps} first_chunk_chars={first_chunk_chars} chunk_chars={chunk_chars} frame_ms={frame_ms} client_epoch_ms={client_epoch_ms} server_epoch_ms={server_epoch_ms} skew_ms={skew_ms}")

    # Optional primer silence to flush headers immediately and force chunked streaming
    if primer_silence_ms and primer_silence_ms > 0:
        n_samples = int(sr * primer_silence_ms / 1000)
        if n_samples > 0:
            primer = torch.zeros(1, n_samples, dtype=torch.float32, device="cpu")
            b = _float32_to_pcm16_bytes(primer)
            total_bytes += len(b); total_frames += 1
            yield b
            # Hint the event loop to flush headers and first bytes immediately
            print(f"[TTS_PRIMER_SENT] req_id={req_id} t_ms={int((time.perf_counter()-start)*1000)}")
            await asyncio.sleep(0)

    # Build chunks: small first chunk, larger subsequent chunks
    chunks: List[str] = []
    if first_chunk_chars and first_chunk_chars > 0:
        fchunks = _split_text_for_low_latency(text, max_chars=first_chunk_chars)
        if fchunks:
            first_text = fchunks[0]
            chunks.append(first_text)
            remaining = text[len(first_text):].strip()
            if remaining:
                chunks.extend(_split_text_for_low_latency(remaining, max_chars=chunk_chars))
    else:
        chunks = _split_text_for_low_latency(text, max_chars=chunk_chars)

    if not chunks:
        return

    for idx, chunk in enumerate(chunks):
        print(f"[TTS_CHUNK_START] req_id={req_id} idx={idx} text_len={len(chunk)} t_ms={int((time.perf_counter()-start)*1000)}")
        # For the first chunk, optionally reduce steps to cut TTFA. Subsequent chunks full quality.
        steps = diffusion_steps
        if idx == 0 and first_chunk_diff_steps is not None:
            steps = max(1, int(first_chunk_diff_steps))

        # Heavy compute dispatched to thread to not block event loop
        chunk_start = time.perf_counter()
        wav: torch.Tensor = await asyncio.to_thread(
            _synthesize_one,
            chunk,
            language_id=language_id,
            temperature=temperature,
            exaggeration=exaggeration,
            diffusion_steps=steps,
            audio_prompt_path=audio_prompt_path,
            max_tokens_override=(max_tokens_first_chunk if idx == 0 else None),
        )
        chunk_ms = int((time.perf_counter() - chunk_start) * 1000)
        audio_samples = int(wav.shape[-1]) if torch.is_tensor(wav) else 0
        audio_ms = int(audio_samples * 1000 / sr) if sr else 0
        print(f"[TTS_CHUNK_DONE] req_id={req_id} idx={idx} chunk_ms={chunk_ms} steps={steps} audio_ms={audio_ms} samples={audio_samples} t_ms={int((time.perf_counter()-start)*1000)}")

        # Optional minimal tail fade to avoid end clicks on chunk boundaries (does not alter timbre)
        fade_ms = 5
        fade_samples = int(sr * fade_ms / 1000)
        if wav.numel() > fade_samples and fade_samples > 0:
            tail = wav[:, -fade_samples:]
            ramp = torch.linspace(1.0, 0.95, steps=fade_samples, device=tail.device, dtype=tail.dtype).unsqueeze(0)
            tail_faded = tail * ramp
            wav = torch.cat([wav[:, :-fade_samples], tail_faded], dim=1)

        # Crossfade with previous tail to avoid boundary clicks between chunks
        applied_cross = 0
        cross_samples = int(sr * max(0, crossfade_ms) / 1000)
        if idx > 0 and cross_samples > 0 and prev_tail is not None and prev_tail.numel() > 0:
            head_len = min(wav.shape[-1], cross_samples)
            tail_len = min(prev_tail.shape[-1], cross_samples)
            mix_len = min(head_len, tail_len)
            if mix_len > 0:
                head = wav[:, :mix_len]
                ramp_up = torch.linspace(0.0, 1.0, steps=mix_len, device=head.device, dtype=head.dtype).unsqueeze(0)
                ramp_dn = torch.linspace(1.0, 0.0, steps=mix_len, device=prev_tail.device, dtype=head.dtype).unsqueeze(0)
                wav[:, :mix_len] = head * ramp_up + prev_tail[:, -mix_len:] * ramp_dn
                applied_cross = mix_len

        # Update prev_tail for next crossfade
        if cross_samples > 0:
            keep = min(cross_samples, wav.shape[-1])
            prev_tail = wav[:, -keep:].detach().clone()
        else:
            prev_tail = None

        if applied_cross:
            print(f"[TTS_CROSSFADE] req_id={req_id} idx={idx} samples={applied_cross}")

        # Watermark policy (off = no modification)
        wav = _apply_watermark_if_needed(wav, sr, watermark=watermark)

        # To PCM16 and then to 40ms-ish frames
        pcm_bytes = _float32_to_pcm16_bytes(wav)
        for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
            if not frame:
                continue
            if not first_synth_frame_sent:
                print(f"[TTS_FIRST_AUDIO_SENT] req_id={req_id} t_ms={int((time.perf_counter()-start)*1000)}")
                first_synth_frame_sent = True
            total_bytes += len(frame); total_frames += 1
            yield frame

        if idx == len(chunks) - 1:
            print(f"[TTS_STREAM_END] req_id={req_id} total_ms={int((time.perf_counter()-start)*1000)} total_bytes={total_bytes} total_frames={total_frames} chunks={len(chunks)}")


# ---------- FastAPI lifecycle ----------

@APP.on_event("startup")
async def _startup() -> None:
    """
    Initialize and warm the TTS engine in memory for production-ready TTFA.
    """
    global _tts_engine, _tts_sr

    # Device config
    device = os.environ.get("CHATTERBOX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    variant = os.environ.get("CHATTERBOX_VARIANT", "english").strip().lower()
    s3gen_use_fp16 = os.environ.get("CHATTERBOX_S3GEN_FP16", "0").strip() in ("1", "true", "True")
    local_ckpt = os.environ.get("CHATTERBOX_USE_LOCAL_CKPT", "").strip() or None
    compile_flag = os.environ.get("CHATTERBOX_COMPILE", "0").strip() in ("1", "true", "True")

    # Load engine (avoid modifying repo code by using exported constructors)
    if local_ckpt:
        _tts_engine = ChatterboxTTS.from_local(
            ckpt_dir=local_ckpt,
            target_device=device,
            variant=("english" if variant != "multilingual" else "multilingual"),
            s3gen_use_fp16=s3gen_use_fp16,
            compile=compile_flag,
        )
    else:
        if variant == "multilingual":
            _tts_engine = ChatterboxTTS.from_pretrained_multilingual(
                target_device=device,
                s3gen_use_fp16=s3gen_use_fp16,
                compile=compile_flag,
            )
        else:
            _tts_engine = ChatterboxTTS.from_pretrained(
                target_device=device,
                s3gen_use_fp16=s3gen_use_fp16,
                compile=compile_flag,
            )

    _tts_sr = _tts_engine.sr

    # Warm up: load conditionals and run a tiny synthesis to prime caches
    warm_text = os.environ.get("CHATTERBOX_WARMUP_TEXT", "Hello")
    try:
        # Precompute audio conditionals (default) and do a small diffusion (fast)
        _ = _tts_engine.get_audio_conditionals(None)
        _ = _tts_engine.generate(
            prompts=warm_text,
            language_id="en" if variant != "multilingual" else "en",
            diffusion_steps=int(os.environ.get("CHATTERBOX_WARMUP_DIFF_STEPS", "2")),
            max_tokens=min(32, _tts_engine.max_model_len),
        )
    except Exception as e:
        # Proceed anyway; the first request may pay the warmup cost if this fails.
        print(f"[WARN] Warmup failed: {e}")

    print(f"[INIT] TTS engine ready on device={device}, variant={variant}, sr={_tts_sr}")


@APP.on_event("shutdown")
async def _shutdown() -> None:
    global _tts_engine
    try:
        if _tts_engine is not None:
            _tts_engine.shutdown()
    except Exception:
        pass
    _tts_engine = None


# ---------- API Routes ----------

@APP.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request):
    """
    OpenAI-compatible TTS endpoint. Supports streaming and non-streaming.
    """
    tts = _ensure_engine()
    req_id = uuid.uuid4().hex

    # Correlation headers from client
    try:
        client_req_id = request.headers.get("x-client-request-id")
    except Exception:
        client_req_id = None
    try:
        _cse = request.headers.get("x-client-start-epoch-ms")
        client_epoch_ms = int(_cse) if _cse else None
    except Exception:
        client_epoch_ms = None
    server_epoch_ms = int(time.time() * 1000)

    # Normalize/resolve response format
    fmt = (req.response_format or req.format or "wav")
    if fmt == "pcm":
        fmt = "pcm16"

    # Streaming path: regardless of requested format, stream linear PCM for lowest latency
    if req.stream:
        stream_headers = {
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "X-Req-Id": req_id,
            "X-Client-Request-Id": client_req_id or "",
            "X-Server-Start-Epoch-Ms": str(server_epoch_ms),
        }
        return StreamingResponse(
            _synthesize_streaming_pcm_frames(
                req.input,
                language_id=req.language_id,
                temperature=req.temperature or 0.8,
                exaggeration=req.exaggeration or 0.5,
                diffusion_steps=req.diffusion_steps or 10,
                audio_prompt_path=req.audio_prompt_path,
                watermark=req.watermark or "off",
                req_id=req_id,
                t0=time.perf_counter(),
                client_epoch_ms=client_epoch_ms,
                server_epoch_ms=server_epoch_ms,
                max_tokens_first_chunk=req.max_tokens_first_chunk,
                primer_silence_ms=req.primer_silence_ms or 0,
                first_chunk_diff_steps=(req.first_chunk_diff_steps if req.first_chunk_diff_steps is not None else 2),
                first_chunk_chars=req.first_chunk_chars or 32,
                chunk_chars=req.chunk_chars or 120,
                frame_ms=req.frame_ms or 20,
                crossfade_ms=req.crossfade_ms or 10,
            ),
            media_type=f"audio/pcm;rate={tts.sr};channels=1",
            headers=stream_headers,
        )

    # Non-streaming path: synthesize full audio then return as a single response
    non_stream_headers = {
        "X-Req-Id": req_id,
        "X-Client-Request-Id": client_req_id or "",
        "X-Server-Start-Epoch-Ms": str(server_epoch_ms),
    }
    try:
        wav: torch.Tensor = await asyncio.to_thread(
            _synthesize_one,
            req.input,
            language_id=req.language_id,
            temperature=req.temperature or 0.8,
            exaggeration=req.exaggeration or 0.5,
            diffusion_steps=req.diffusion_steps or 10,
            audio_prompt_path=req.audio_prompt_path,
        )
        wav = _apply_watermark_if_needed(wav, tts.sr, watermark=req.watermark or "off")

        if fmt == "pcm16":
            pcm = _float32_to_pcm16_bytes(wav)
            return Response(content=pcm, media_type=f"audio/pcm;rate={tts.sr};channels=1", headers=non_stream_headers)
        else:
            data = _pack_wav_bytes(wav, sr=tts.sr)
            return Response(content=data, media_type="audio/wav", headers=non_stream_headers)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {e}")


def _print_usage():
    print("OpenAI-compatible TTS server for chatterbox-vllm")
    print("Usage:")
    print("  python openai_tts_server.py")
    print("")
    print("Environment variables:")
    print("  CHATTERBOX_VARIANT=english|multilingual")
    print("  CHATTERBOX_DEVICE=cuda|cpu")
    print("  CHATTERBOX_S3GEN_FP16=0|1")
    print("  CHATTERBOX_USE_LOCAL_CKPT=/path/to/ckpt_dir")
    print("  CHATTERBOX_WARMUP_TEXT='Hello'")
    print("  CHATTERBOX_WARMUP_DIFF_STEPS=2")
    print("Server listens on http://0.0.0.0:8000")


if __name__ == "__main__":
    # Lazy import uvicorn when used as a script
    try:
        import uvicorn
    except Exception:
        _print_usage()
        raise

    # Run server
    uvicorn.run(APP, host="0.0.0.0", port=8000, log_level="info")
