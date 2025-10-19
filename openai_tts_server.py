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
  - CHATTERBOX_VARIANT: "multilingual" (default) | "english"
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
import threading
import argparse
from typing import Optional, Iterable, Generator, List, Tuple

# Ensure `src/` is importable when running from repo root
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from chatterbox_vllm.tts import ChatterboxTTS


APP = FastAPI(title="OpenAI-compatible TTS for chatterbox-vllm")

# Global singleton engine (initialized on startup)
_tts_engine: Optional[ChatterboxTTS] = None

# Runtime config captured at startup and reused for loading
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
_variant: str = "multilingual"  # default: load one engine ('english' or 'multilingual')
_s3gen_use_fp16: bool = False
_local_ckpt: Optional[str] = None
_enable_compile: bool = False


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
    first_chunk_diff_steps: Optional[int] = Field(default=10, ge=1, description="Diffusion steps for the first chunk only")
    first_chunk_chars: Optional[int] = Field(default=30, ge=10, le=200, description="Max chars for the first chunk")
    chunk_chars: Optional[int] = Field(default=120, ge=40, le=400, description="Max chars for subsequent chunks")
    frame_ms: Optional[int] = Field(default=20, ge=10, le=60, description="Frame size for PCM streaming chunks")


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
    Supports CJK and Arabic punctuation for better multilingual chunking.
    """
    text = text.strip()
    if not text:
        return []
    # First split by strong punctuation (ASCII + common CJK/Arabic)
    import re
    parts = re.split(r"([\.!\?;:。！？；，、،؛؟…])", text)
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
        if cur and len(cur) + 1 + len(chunk) <= max_chars:
            cur = (cur + " " + chunk).strip()
        elif not cur and len(chunk) <= max_chars:
            cur = chunk
        else:
            if cur:
                assembled.append(cur)
            cur = chunk
            # If a single segment is already larger than max_chars (e.g., no spaces),
            # keep it as-is for now; it will be split in the next pass.
    if cur:
        assembled.append(cur)

    if not assembled:
        return [text]

    # Second pass: enforce max_chars with whitespace when available, else fixed-width slicing.
    final_chunks: List[str] = []
    for c in assembled:
        if len(c) <= max_chars:
            final_chunks.append(c)
            continue

        words = c.split()
        if len(words) <= 1:
            # No whitespace -> fixed width slicing
            for i in range(0, len(c), max_chars):
                final_chunks.append(c[i:i + max_chars])
            continue

        cur = ""
        for w in words:
            if len(w) > max_chars:
                # Extremely long token (e.g., CJK run with no spaces); flush cur and slice w
                if cur:
                    final_chunks.append(cur)
                    cur = ""
                for i in range(0, len(w), max_chars):
                    final_chunks.append(w[i:i + max_chars])
                continue
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur = f"{cur} {w}"
            else:
                final_chunks.append(cur)
                cur = w
        if cur:
            final_chunks.append(cur)

    return [i for i in final_chunks if i]


def _repair_chunk_boundaries(chunks: List[str]) -> List[str]:
    """
    Repair chunk boundaries for better prosody:
      - Ensure no chunk starts with punctuation; attach any leading punctuation to the previous chunk.
      - Trim whitespace introduced by boundary moves.
      - Drop chunks that become empty after stripping.
    This keeps punctuation attached to the preceding phrase (natural pause), which usually sounds best.
    """
    if not chunks:
        return chunks

    import re
    # Common ASCII and Unicode punctuation to consider at boundaries
    leading_punct = re.compile(r'^[\s\.,!\?\;:\-\u2014\u2013\u2026\)\]\}\u3002\uFF01\uFF1F\u3001\uFF0C\u060C\u061B\u061F]+')

    repaired: List[str] = []
    for idx, c in enumerate(chunks):
        c = c if isinstance(c, str) else str(c)
        if idx == 0:
            repaired.append(c.strip())
            continue

        # Move any leading punctuation to the previous chunk
        m = leading_punct.match(c)
        if m:
            lead = m.group(0)
            body = c[len(lead):].lstrip()
            # Attach punctuation to previous chunk if non-empty after strip
            if lead and lead.strip():
                repaired[-1] = (repaired[-1].rstrip() + lead).rstrip()
            c = body

        # Normalize and append
        c = c.strip()
        if c:
            repaired.append(c)

    # Remove any empties that may remain
    repaired = [c for c in repaired if c]
    return repaired


def _avoid_weak_endings(chunks: List[str]) -> List[str]:
    """
    Adjust chunk boundaries to avoid ending a chunk on weak function words, which
    often sounds unnatural at joins (e.g., trailing 'a', 'the', 'to').
    Strategy:
      - If a chunk (except the last) ends with a weak word, move that last word
        to the beginning of the next chunk so it binds with the following content.
      - Preserves punctuation attachment rules from _repair_chunk_boundaries.
    """
    if not chunks:
        return chunks

    WEAK_END_WORDS = {
        "a", "an", "the", "to", "of", "in", "on", "at", "and", "or", "but", "for", "nor", "so"
    }

    out: List[str] = chunks[:]
    i = 0
    while i < len(out) - 1:
        cur = out[i].strip()
        nxt = out[i + 1].strip()
        if not cur or not nxt:
            i += 1
            continue

        # Extract last word from current chunk (ignore trailing punctuation/spaces)
        import re
        # Remove trailing punctuation for word test (keep original text for reassignment)
        cur_no_trail_punct = re.sub(r"[\s\.,!\?\;:\-\u2014\u2013\u2026\)\]\}]+$", "", cur)
        words = cur_no_trail_punct.split()
        if len(words) == 0:
            i += 1
            continue

        last_word = words[-1].lower()
        if last_word in WEAK_END_WORDS:
            # Remove the last word from current chunk text safely
            # Find index of the last_word occurrence at end (case-insensitive)
            idx = cur_no_trail_punct.rfind(words[-1])
            if idx >= 0:
                new_cur = cur_no_trail_punct[:idx].rstrip()
                # Preserve any original trailing punctuation that followed the word
                trailing = cur[len(cur_no_trail_punct):]
                if new_cur:
                    out[i] = (new_cur + trailing).rstrip()
                else:
                    # If current becomes empty, attach trailing punct to next chunk
                    out[i] = ""
                    nxt = (trailing + " " + nxt).strip()

                # Move the weak last_word to the start of next chunk
                out[i + 1] = (words[-1] + " " + nxt).strip()

                # If current became empty, drop it and continue without skipping the next
                if out[i] == "":
                    del out[i]
                    # after deletion, do not increment i to re-check new boundary
                    continue

                # Re-check this boundary in case multiple weak tokens accumulate
                continue
        i += 1

    # Clean empties
    out = [c for c in out if c and c.strip()]
    return out


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

def _detect_language(text: str) -> Optional[str]:
    """
    Heuristic language detector based on Unicode ranges.
    Returns a BCP-47-ish 2-letter code matching the multilingual model's supported set.
    Only used as a fallback when language_id is not provided.
    """
    if not text:
        return None
    # Hebrew
    if any('\u0590' <= ch <= '\u05FF' for ch in text):
        return "he"
    # Arabic
    if any('\u0600' <= ch <= '\u06FF' for ch in text):
        return "ar"
    # Cyrillic (Russian)
    if any('\u0400' <= ch <= '\u04FF' for ch in text):
        return "ru"
    # Hangul (Korean)
    if any('\uAC00' <= ch <= '\uD7AF' for ch in text):
        return "ko"
    # Hiragana/Katakana (Japanese)
    if any('\u3040' <= ch <= '\u309F' for ch in text) or any('\u30A0' <= ch <= '\u30FF' for ch in text):
        return "ja"
    # CJK Unified Ideographs (Chinese)
    if any('\u4E00' <= ch <= '\u9FFF' for ch in text):
        return "zh"
    return None

def _ensure_engine() -> ChatterboxTTS:
    """
    Return the single initialized TTS engine. Raises if not initialized.
    """
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
) -> torch.Tensor:
    """
    Synthesize one prompt to a waveform (1, T) tensor.
    """
    tts = _ensure_engine()

    # Auto-detect language if caller didn't provide one (important for Hebrew and others)
    if getattr(tts, "variant", "english") == "multilingual" and (not language_id or not str(language_id).strip()):
        autodetect = _detect_language(text)
        if autodetect:
            print(f"[Server] Auto-detected language_id='{autodetect}' from input text")
            language_id = autodetect

    # For multilingual vLLM path, ensure special tokens aren't altered by tokenizer settings.
    # Do NOT ignore EOS: allow decoder to stop on speech stop token.
    extra_sampling_kwargs = {}
    if getattr(tts, "variant", "english") == "multilingual":
        extra_sampling_kwargs = {
            "spaces_between_special_tokens": False,
            "skip_special_tokens": False,
            "ignore_eos": False,
        }

    waves = tts.generate(
        prompts=text,
        audio_prompt_path=audio_prompt_path,
        language_id=language_id,
        temperature=temperature,
        exaggeration=exaggeration,
        diffusion_steps=diffusion_steps,
        max_tokens=tts.max_model_len,
        **extra_sampling_kwargs,
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
    primer_silence_ms: int = 0,
    first_chunk_diff_steps: Optional[int] = None,
    first_chunk_chars: int = 30,
    chunk_chars: int = 120,
    frame_ms: int = 20,
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


    # Optional primer silence to flush headers immediately and force chunked streaming
    if primer_silence_ms and primer_silence_ms > 0:
        n_samples = int(sr * primer_silence_ms / 1000)
        if n_samples > 0:
            primer = torch.zeros(1, n_samples, dtype=torch.float32, device="cpu")
            yield _float32_to_pcm16_bytes(primer)
            # Hint the event loop to flush headers and first bytes immediately
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

    # Repair chunk boundaries to avoid starting chunks with punctuation for better prosody
    chunks = _repair_chunk_boundaries(chunks)
    # Avoid weak function-word endings at boundaries for better continuity
    chunks = _avoid_weak_endings(chunks)

    if not chunks:
        return

    for idx, chunk in enumerate(chunks):
        # For the first chunk, optionally reduce steps to cut TTFA. Subsequent chunks full quality.
        steps = diffusion_steps
        if idx == 0 and first_chunk_diff_steps is not None:
            steps = max(1, int(first_chunk_diff_steps))

        # Heavy compute dispatched to thread to not block event loop
        wav: torch.Tensor = await asyncio.to_thread(
            _synthesize_one,
            chunk,
            language_id=language_id,
            temperature=temperature,
            exaggeration=exaggeration,
            diffusion_steps=steps,
            audio_prompt_path=audio_prompt_path,
        )

        # Optional minimal tail fade to avoid end clicks on chunk boundaries (does not alter timbre)
        fade_ms = 5
        fade_samples = int(sr * fade_ms / 1000)
        if wav.numel() > fade_samples and fade_samples > 0:
            tail = wav[:, -fade_samples:]
            ramp = torch.linspace(1.0, 0.95, steps=fade_samples, device=tail.device, dtype=tail.dtype).unsqueeze(0)
            tail_faded = tail * ramp
            wav = torch.cat([wav[:, :-fade_samples], tail_faded], dim=1)

        # Watermark policy (off = no modification)
        wav = _apply_watermark_if_needed(wav, sr, watermark=watermark)

        # To PCM16 and then to 40ms-ish frames
        pcm_bytes = _float32_to_pcm16_bytes(wav)
        for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
            if not frame:
                continue
            yield frame


# ---------- FastAPI lifecycle ----------

@APP.on_event("startup")
async def _startup() -> None:
    """
    Initialize config and load a single engine (english or multilingual).
    Default variant is 'multilingual'.
    """
    global _device, _variant, _s3gen_use_fp16, _local_ckpt, _enable_compile, _tts_engine

    # Capture runtime config
    _device = os.environ.get("CHATTERBOX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    _variant = os.environ.get("CHATTERBOX_VARIANT", "multilingual").strip().lower()
    _s3gen_use_fp16 = os.environ.get("CHATTERBOX_S3GEN_FP16", "0").strip() in ("1", "true", "True")
    _local_ckpt = os.environ.get("CHATTERBOX_USE_LOCAL_CKPT", "").strip() or None
    _enable_compile = os.environ.get("CHATTERBOX_COMPILE", "0").strip().lower() in ("1","true","yes","on")

    # Load the selected engine (single instance)
    try:
        if _local_ckpt:
            _tts_engine = ChatterboxTTS.from_local(
                ckpt_dir=_local_ckpt,
                target_device=_device,
                s3gen_use_fp16=_s3gen_use_fp16,
                compile=_enable_compile,
                variant=_variant,
            )
        else:
            if _variant == "multilingual":
                _tts_engine = ChatterboxTTS.from_pretrained_multilingual(
                    target_device=_device,
                    s3gen_use_fp16=_s3gen_use_fp16,
                    compile=_enable_compile,
                )
            else:
                _tts_engine = ChatterboxTTS.from_pretrained(
                    target_device=_device,
                    s3gen_use_fp16=_s3gen_use_fp16,
                    compile=_enable_compile,
                )
    except Exception as e:
        raise RuntimeError(f"Failed to load TTS engine for variant='{_variant}': {e}")

    # Warm up: load conditionals and run a tiny synthesis to prime caches
    warm_text = os.environ.get("CHATTERBOX_WARMUP_TEXT", "Hello")
    try:
        _ = _tts_engine.get_audio_conditionals(None)

        warm_extra_sampling_kwargs = {}
        if getattr(_tts_engine, "variant", "english") == "multilingual":
            warm_extra_sampling_kwargs = {
                "spaces_between_special_tokens": False,
                "skip_special_tokens": False,
                "ignore_eos": False,
            }

        _ = _tts_engine.generate(
            prompts=warm_text,
            language_id="en",
            diffusion_steps=int(os.environ.get("CHATTERBOX_WARMUP_DIFF_STEPS", "2")),
            max_tokens=min(32, _tts_engine.max_model_len),
            **warm_extra_sampling_kwargs,
        )
    except Exception as e:
        print(f"[WARN] Warmup failed: {e}")

    print(f"[INIT] TTS server ready on device={_device}, variant={_variant}, sr=24000, compile={_enable_compile}")


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
async def create_speech(req: SpeechRequest):
    """
    OpenAI-compatible TTS endpoint. Supports streaming and non-streaming.
    """
    tts = _ensure_engine()

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
                primer_silence_ms=req.primer_silence_ms or 0,
                first_chunk_diff_steps=(req.first_chunk_diff_steps if req.first_chunk_diff_steps is not None else 10),
                first_chunk_chars=req.first_chunk_chars or 30,
                chunk_chars=req.chunk_chars or 120,
                frame_ms=req.frame_ms or 20,
            ),
            media_type=f"audio/pcm;rate={tts.sr};channels=1",
            headers=stream_headers,
        )

    # Non-streaming path: synthesize full audio then return as a single response
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
            return Response(content=pcm, media_type=f"audio/pcm;rate={tts.sr};channels=1")
        else:
            data = _pack_wav_bytes(wav, sr=tts.sr)
            return Response(content=data, media_type="audio/wav")
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
    print("")
    print("CLI flags:")
    print("  --multi     Force multilingual variant")
    print("  --single    Force english (single-language) variant")
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

    # Parse CLI flags
    parser = argparse.ArgumentParser(description="OpenAI-compatible TTS server for chatterbox-vllm")
    parser.add_argument("--multi", action="store_true", help="Load multilingual model")
    parser.add_argument("--single", action="store_true", help="Load english (single-language) model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    # Resolve variant from flags (env var still supported)
    if args.multi and args.single:
        print("[ERROR] Cannot specify both --multi and --single")
        sys.exit(1)
    if args.multi:
        os.environ["CHATTERBOX_VARIANT"] = "multilingual"
    elif args.single:
        os.environ["CHATTERBOX_VARIANT"] = "english"

    # Run server
    uvicorn.run(APP, host=args.host, port=args.port, log_level="info")
