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
  - CHATTERBOX_STREAM_CROSSFADE_MS: crossfade length in ms for stitching between streamed chunks (default: 40)
  - CHATTERBOX_STREAM_LEADING_TRIM_MS: leading trim in ms removed from the head of each subsequent chunk (default: 60)
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
from chatterbox_vllm.mtl_tts import ChatterboxMultilingualTTS


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
    first_chunk_diff_steps: Optional[int] = Field(default=3, ge=1, description="Diffusion steps for the first chunk only")
    first_chunk_chars: Optional[int] = Field(default=60, ge=10, le=200, description="Max chars for the first chunk")
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

    # Post-pass: avoid splitting after weak function words; merge across boundaries dynamically
    cleaned = [i for i in final_chunks if i]
    try:
        NONBREAK_END_WORDS = {
            "a","an","the","to","of","in","on","for","and","or","but",
            "my","your","our","their","his","her","its"
        }
        def _ends_with_nonbreaker(s: str) -> bool:
            if not s: return False
            # strip trailing punctuation/quotes/dashes
            t = s.rstrip().rstrip("”’'\"—-) ")
            # if ends with strong punctuation, do not merge
            if t and t[-1] in ".!?;:":
                return False
            words = t.split()
            return len(words) > 0 and words[-1].lower().strip("“’'\"(") in NONBREAK_END_WORDS

        def _starts_with_word(s: str) -> bool:
            import re
            if not s: return False
            return re.match(r"^[A-Za-z]", s.strip()) is not None

        merged: List[str] = []
        i = 0
        while i < len(cleaned):
            cur_chunk = cleaned[i]
            if i < len(cleaned) - 1 and _ends_with_nonbreaker(cur_chunk) and _starts_with_word(cleaned[i+1]):
                next_chunk = cleaned[i+1]
                next_words = next_chunk.split()
                moved = 0
                # Try to attach up to 2 words if it fits max_chars
                while next_words and moved < 2:
                    candidate = (cur_chunk + " " + next_words[0]).strip()
                    if len(candidate) <= max_chars:
                        cur_chunk = candidate
                        next_words.pop(0)
                        moved += 1
                    else:
                        break
                cleaned[i+1] = " ".join(next_words).strip()
                # If we consumed all of the next chunk, skip it
                if cleaned[i+1] == "":
                    i += 1
            merged.append(cur_chunk)
            i += 1
        # Remove any emptied chunks
        cleaned = [c for c in merged if c]
    except Exception:
        # On any error, fall back to original chunks
        pass

    return cleaned


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


# ---------- Streaming smoothing helpers ----------

def _trim_head_ms(wav: torch.Tensor, ms: int, sr: int) -> torch.Tensor:
    """
    Remove the first ms of audio to eliminate vocoder onset artifacts at chunk boundaries.
    """
    if wav is None or not torch.is_tensor(wav) or wav.numel() == 0 or ms <= 0:
        return wav
    n = int(sr * ms / 1000)
    if n <= 0:
        return wav
    if wav.shape[1] <= n:
        return wav[:, 0:0]
    return wav[:, n:]


def _take_tail_ms(wav: torch.Tensor, ms: int, sr: int) -> torch.Tensor:
    """
    Return the last ms of audio (for overlap-add with the next chunk).
    """
    if wav is None or not torch.is_tensor(wav) or wav.numel() == 0 or ms <= 0:
        # Preserve shape semantics
        return wav[:, 0:0] if (wav is not None and wav.ndim == 2) else torch.zeros(1, 0)
    n = int(sr * ms / 1000)
    n = max(0, min(n, wav.shape[1]))
    return wav[:, -n:]


def _crossfade(a_tail: torch.Tensor, b_head: torch.Tensor, ms: int, sr: int) -> torch.Tensor:
    """
    Overlap-add crossfade between the tail of chunk A and the head of chunk B.
    If one side is shorter than requested ms, use the min available length.
    """
    if (a_tail is None or a_tail.numel() == 0) and (b_head is None or b_head.numel() == 0):
        return torch.zeros(1, 0)
    L_req = int(sr * ms / 1000)
    La = a_tail.shape[1] if (a_tail is not None and a_tail.ndim == 2) else 0
    Lb = b_head.shape[1] if (b_head is not None and b_head.ndim == 2) else 0
    use = max(0, min(L_req, La, Lb))
    if use <= 0:
        if a_tail is not None and a_tail.numel() > 0:
            return torch.cat([a_tail, b_head], dim=1) if (b_head is not None and b_head.numel() > 0) else a_tail
        return b_head
    a = a_tail[:, -use:]
    b = b_head[:, :use]
    ramp = torch.linspace(0.0, 1.0, steps=use, dtype=a.dtype, device=a.device).unsqueeze(0)
    return a * (1.0 - ramp) + b * ramp


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
    if isinstance(tts, ChatterboxMultilingualTTS) and (not language_id or not str(language_id).strip()):
        autodetect = _detect_language(text)
        if autodetect:
            print(f"[Server] Auto-detected language_id='{autodetect}' from input text")
            language_id = autodetect

    # For multilingual vLLM path, ensure special tokens aren't altered by tokenizer settings.
    # Do NOT ignore EOS: allow decoder to stop on speech stop token.
    extra_sampling_kwargs = {}
    if isinstance(tts, ChatterboxMultilingualTTS):
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
    first_chunk_chars: int = 60,
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

    # For multilingual engine, avoid per-chunk prefill to prevent tokenizer/block mismatch.
    # Generate full utterance once, then stream PCM frames.
    if isinstance(tts, ChatterboxMultilingualTTS):
        wav: torch.Tensor = await asyncio.to_thread(
            _synthesize_one,
            text,
            language_id=language_id,
            temperature=temperature,
            exaggeration=exaggeration,
            diffusion_steps=diffusion_steps,
            audio_prompt_path=audio_prompt_path,
        )
        wav = _apply_watermark_if_needed(wav, sr, watermark=watermark)
        pcm_bytes = _float32_to_pcm16_bytes(wav)
        for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
            if not frame:
                continue
            yield frame
        return

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

    if not chunks:
        return

    # Dynamic audio stitching knobs
    crossfade_ms = int(os.environ.get("CHATTERBOX_STREAM_CROSSFADE_MS", "40"))
    leading_trim_ms = int(os.environ.get("CHATTERBOX_STREAM_LEADING_TRIM_MS", "60"))
    hold_ms = max(0, crossfade_ms)
    hold_samples = int(sr * hold_ms / 1000)

    prev_tail: Optional[torch.Tensor] = None

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

        # Optional minimal tail fade to avoid clicks (kept conservative; crossfade handles main smoothing)
        fade_ms = 5
        fade_samples = int(sr * fade_ms / 1000)
        if wav.numel() > fade_samples and fade_samples > 0:
            tail = wav[:, -fade_samples:]
            ramp = torch.linspace(1.0, 0.95, steps=fade_samples, device=tail.device, dtype=tail.dtype).unsqueeze(0)
            wav = torch.cat([wav[:, :-fade_samples], tail * ramp], dim=1)

        # Watermark policy (off = no modification)
        wav = _apply_watermark_if_needed(wav, sr, watermark=watermark)

        if idx == 0:
            # Stream immediately except for the held tail for crossfade
            if hold_samples > 0 and wav.shape[1] > hold_samples:
                immediate = wav[:, :-hold_samples]
                prev_tail = wav[:, -hold_samples:]
                if immediate.numel() > 0:
                    pcm_bytes = _float32_to_pcm16_bytes(immediate)
                    for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                        if not frame:
                            continue
                        yield frame
            else:
                # Chunk too short to hold a tail; stream all and skip crossfade
                pcm_bytes = _float32_to_pcm16_bytes(wav)
                for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                    if not frame:
                        continue
                    yield frame
                prev_tail = None
            continue

        # Subsequent chunks: trim head to remove vocoder onset artifacts, then crossfade with previous tail
        cur = _trim_head_ms(wav, leading_trim_ms, sr)

        if prev_tail is not None and prev_tail.numel() > 0 and cur is not None and cur.numel() > 0:
            # Determine overlap length actually available
            overlap = min(hold_samples, prev_tail.shape[1], cur.shape[1])

            # If prev_tail is longer than cur head, flush the extra non-overlapped part first
            if prev_tail.shape[1] > overlap:
                leftover = prev_tail[:, :prev_tail.shape[1] - overlap]
                if leftover.numel() > 0:
                    pcm_bytes = _float32_to_pcm16_bytes(leftover)
                    for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                        if not frame:
                            continue
                        yield frame

            # Crossfade the overlapping region
            head = cur[:, :overlap]
            xfade = _crossfade(prev_tail, head, ms=crossfade_ms if overlap == hold_samples else int(1000 * overlap / sr), sr=sr)
            if xfade is not None and xfade.numel() > 0:
                pcm_bytes = _float32_to_pcm16_bytes(xfade)
                for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                    if not frame:
                        continue
                    yield frame

            # Stream the mid-body, holding back the last hold_samples for the next boundary
            start_mid = overlap
            if cur.shape[1] > start_mid + hold_samples:
                mid = cur[:, start_mid:cur.shape[1] - hold_samples]
                if mid.numel() > 0:
                    pcm_bytes = _float32_to_pcm16_bytes(mid)
                    for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                        if not frame:
                            continue
                        yield frame
                prev_tail = cur[:, -hold_samples:]
            else:
                # Not enough to hold a full tail for next time; keep whatever remains as next tail
                prev_tail = cur[:, max(start_mid, cur.shape[1] - hold_samples):]
        else:
            # No valid overlap possible. Flush previous tail (if any), then stream current normally.
            if prev_tail is not None and prev_tail.numel() > 0:
                pcm_bytes = _float32_to_pcm16_bytes(prev_tail)
                for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                    if not frame:
                        continue
                    yield frame
                prev_tail = None

            if cur is not None and cur.numel() > 0:
                if hold_samples > 0 and cur.shape[1] > hold_samples:
                    body = cur[:, :-hold_samples]
                    if body.numel() > 0:
                        pcm_bytes = _float32_to_pcm16_bytes(body)
                        for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                            if not frame:
                                continue
                            yield frame
                    prev_tail = cur[:, -hold_samples:]
                else:
                    pcm_bytes = _float32_to_pcm16_bytes(cur)
                    for frame in _frame_chunk_bytes(pcm_bytes, sr=sr, frame_ms=frame_ms):
                        if not frame:
                            continue
                        yield frame
                    prev_tail = None

    # After the final chunk, flush any held tail
    if prev_tail is not None and prev_tail.numel() > 0:
        pcm_bytes = _float32_to_pcm16_bytes(prev_tail)
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
            if _variant == "multilingual":
                _tts_engine = ChatterboxMultilingualTTS.from_local(
                    ckpt_dir=_local_ckpt,
                    target_device=_device,
                    s3gen_use_fp16=_s3gen_use_fp16,
                    compile=_enable_compile,
                )
            else:
                _tts_engine = ChatterboxTTS.from_local(
                    ckpt_dir=_local_ckpt,
                    target_device=_device,
                    variant="english",
                    s3gen_use_fp16=_s3gen_use_fp16,
                    compile=_enable_compile,
                )
        else:
            if _variant == "multilingual":
                _tts_engine = ChatterboxMultilingualTTS.from_pretrained(
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
        if isinstance(_tts_engine, ChatterboxMultilingualTTS):
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
                first_chunk_diff_steps=(req.first_chunk_diff_steps if req.first_chunk_diff_steps is not None else 3),
                first_chunk_chars=req.first_chunk_chars or 60,
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
