"""OpenAI-compatible TTS server built on top of Chatterbox."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

if __package__ in {None, ""}:
    # Allow running this module as ``python openai_tts.py`` by ensuring the
    # repository's ``src`` directory is importable before resolving package
    # imports such as ``chatterbox_vllm``.
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import torchaudio.functional as AF
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.text_utils import punc_norm
from chatterbox_vllm.models.s3tokenizer import drop_invalid_tokens
from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
from vllm import SamplingParams

LOGGER = logging.getLogger("chatterbox.openai_tts")


DEFAULT_PLACEHOLDER_PROMPT = "Chatterbox real-time service is now ready."
DEFAULT_CHUNK_DURATION_MS = 200


@dataclass
class EngineConfig:
    """Runtime configuration loaded from environment variables."""

    variant: str = os.getenv("CHATTERBOX_VARIANT", "english")
    device: str = os.getenv(
        "CHATTERBOX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    enforce_eager: bool = os.getenv("CHATTERBOX_ENFORCE_EAGER", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    compile: bool = os.getenv("CHATTERBOX_COMPILE", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    max_model_len: int = int(os.getenv("CHATTERBOX_MAX_MODEL_LEN", "1000"))
    _gpu_mem_util = os.getenv("CHATTERBOX_GPU_MEMORY_UTILIZATION")
    gpu_memory_utilization: Optional[float] = (
        float(_gpu_mem_util) if _gpu_mem_util is not None else None
    )
    max_batch_size: int = int(os.getenv("CHATTERBOX_MAX_BATCH_SIZE", "1"))
    diffusion_steps: int = int(os.getenv("CHATTERBOX_DIFFUSION_STEPS", "10"))
    warmup_prompt: str = os.getenv(
        "CHATTERBOX_WARMUP_PROMPT", DEFAULT_PLACEHOLDER_PROMPT
    )
    warmup_language: str = os.getenv("CHATTERBOX_WARMUP_LANGUAGE", "en")


class AudioSpeechRequest(BaseModel):
    """Subset of the OpenAI audio.speech schema that we support."""

    model: str = Field(description="Model identifier – retained for OpenAI parity.")
    input: str | list[str] = Field(description="Text to be synthesised.")
    voice: Optional[str] = Field(
        default=None,
        description="Placeholder for OpenAI compatibility; voices are handled externally.",
    )
    response_format: str = Field(
        default="wav",
        alias="format",
        description="Audio container to return. Supported values: wav, pcm16",
    )
    sample_rate: Optional[int] = Field(
        default=None,
        description="Optional resampling rate applied before returning audio.",
    )
    stream: bool = Field(
        default=False,
        description="When true, return an SSE stream that emits audio chunks immediately.",
    )
    watermark: str = Field(
        default="on",
        description="Set to 'off' to remove the built-in Chatterbox watermarking fade.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Optional language hint – required for multilingual variants.",
    )
    audio_prompt: Optional[str] = Field(
        default=None,
        description="Optional path to a reference audio prompt on disk.",
    )
    exaggeration: float = Field(default=0.5)
    temperature: float = Field(default=0.8)
    top_p: float = Field(default=0.8)
    repetition_penalty: float = Field(default=2.0)
    diffusion_steps: Optional[int] = Field(default=None)
    max_output_tokens: Optional[int] = Field(default=None, alias="max_output_tokens")
    chunk_duration_ms: Optional[int] = Field(
        default=DEFAULT_CHUNK_DURATION_MS,
        description="Target chunk size for streaming responses in milliseconds.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    def normalized_text(self) -> str:
        if isinstance(self.input, list):
            return " ".join(str(segment) for segment in self.input)
        return str(self.input)

    def watermark_enabled(self) -> bool:
        return str(self.watermark).lower() != "off"


class RealtimeChatterboxAdapter:
    """Adapter that mirrors :meth:`ChatterboxTTS.generate_with_conds` with tweaks."""

    def __init__(self, tts: ChatterboxTTS, default_diffusion_steps: int = 10):
        self.tts = tts
        self.default_diffusion_steps = default_diffusion_steps
        # Trim 120ms worth of audio when removing the watermark.
        self._watermark_trim_seconds = 0.12

    def synthesize(
        self,
        text: str,
        *,
        audio_prompt_path: Optional[str],
        language_id: Optional[str],
        exaggeration: float,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: Optional[int],
        diffusion_steps: Optional[int],
        apply_watermark: bool,
    ) -> torch.Tensor:
        if not text:
            raise ValueError("input text is empty")

        prompts = ["[START]" + punc_norm(text) + "[STOP]"]

        if self.tts.variant == "multilingual":
            lang = (language_id or "en").lower()
            supported = self.tts.get_supported_languages()
            if lang not in supported:
                raise ValueError(
                    f"Unsupported language '{lang}'. Supported: {', '.join(sorted(supported))}"
                )
            prompts = [f"<{lang}>{prompt}" for prompt in prompts]
        elif language_id and language_id.lower() not in {"en", "english"}:
            raise ValueError("This model only supports English – omit the language parameter.")

        s3gen_ref, cond_emb = self.tts.get_audio_conditionals(audio_prompt_path)
        cond_emb = self.tts.update_exaggeration(cond_emb, exaggeration)

        sampling_params = SamplingParams(
            temperature=temperature,
            stop_token_ids=[self.tts.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
            max_tokens=min(max_tokens or self.tts.max_model_len, self.tts.max_model_len),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        with torch.inference_mode():
            batch_results = self.tts.t3.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"conditionals": [cond_emb]},
                    }
                    for prompt in prompts
                ],
                sampling_params=sampling_params,
            )

        waveforms: list[torch.Tensor] = []
        no_trim = not apply_watermark
        diffusion_steps = diffusion_steps or self.default_diffusion_steps

        for batch_result in batch_results:
            for output in batch_result.outputs:
                speech_tokens = torch.tensor(
                    [token - SPEECH_TOKEN_OFFSET for token in output.token_ids],
                    device=self.tts.target_device,
                )
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < 6561]

                wav, _ = self.tts.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=s3gen_ref,
                    n_timesteps=diffusion_steps,
                    no_trim=no_trim,
                )

                waveform = wav.cpu()
                if not apply_watermark:
                    waveform = self._strip_watermark_residual(waveform)
                waveforms.append(waveform)

        if not waveforms:
            raise RuntimeError("No outputs returned by T3 inference")

        return waveforms[0]

    def _strip_watermark_residual(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("Waveform must have shape (channels, time)")
        num_trim = int(self.tts.sr * self._watermark_trim_seconds)
        if num_trim <= 0 or waveform.shape[1] <= num_trim:
            return waveform
        trimmed = waveform.clone()
        trimmed[:, :num_trim] = 0.0
        return trimmed


class TTSEngine:
    """Wraps blocking Chatterbox operations with asyncio-friendly helpers."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.tts: Optional[ChatterboxTTS] = None
        self.adapter: Optional[RealtimeChatterboxAdapter] = None
        self._lock = asyncio.Lock()

    @property
    def sample_rate(self) -> int:
        if not self.tts:
            raise RuntimeError("Engine has not been initialised")
        return self.tts.sr

    async def startup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_models)
        await loop.run_in_executor(None, self._warmup)
        LOGGER.info("Chatterbox TTS engine initialised")

    def _load_models(self) -> None:
        load_kwargs = dict(
            target_device=self.config.device,
            max_model_len=self.config.max_model_len,
            max_batch_size=self.config.max_batch_size,
            enforce_eager=self.config.enforce_eager,
            compile=self.config.compile,
        )

        if self.config.gpu_memory_utilization is not None:
            load_kwargs["gpu_memory_utilization"] = (
                self.config.gpu_memory_utilization
            )

        if self.config.variant.lower() == "multilingual":
            tts = ChatterboxTTS.from_pretrained_multilingual(**load_kwargs)
        else:
            tts = ChatterboxTTS.from_pretrained(**load_kwargs)

        self.tts = tts
        self.adapter = RealtimeChatterboxAdapter(tts, self.config.diffusion_steps)

    def _warmup(self) -> None:
        if not self.adapter:
            return
        try:
            self.adapter.synthesize(
                self.config.warmup_prompt,
                audio_prompt_path=None,
                language_id=self.config.warmup_language,
                exaggeration=0.5,
                temperature=0.8,
                top_p=0.8,
                repetition_penalty=2.0,
                max_tokens=self.config.max_model_len,
                diffusion_steps=self.config.diffusion_steps,
                apply_watermark=True,
            )
            LOGGER.info("Warm-up inference completed")
        except Exception as exc:  # pragma: no cover - best-effort warm-up
            LOGGER.warning("Warm-up inference failed: %s", exc)

    async def shutdown(self) -> None:
        if self.tts:
            self.tts.shutdown()
        self.tts = None
        self.adapter = None

    async def synthesize(self, request: AudioSpeechRequest) -> torch.Tensor:
        if not self.adapter:
            raise HTTPException(status_code=503, detail="TTS engine not ready")

        text = request.normalized_text()
        apply_watermark = request.watermark_enabled()

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.adapter.synthesize(
                    text,
                    audio_prompt_path=request.audio_prompt,
                    language_id=request.language,
                    exaggeration=request.exaggeration,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty,
                    max_tokens=request.max_output_tokens,
                    diffusion_steps=request.diffusion_steps,
                    apply_watermark=apply_watermark,
                ),
            )

    def to_pcm16(self, waveform: torch.Tensor, sample_rate: Optional[int]) -> tuple[bytes, int]:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) != 1:
            raise ValueError("Only mono waveforms are supported")

        sr = self.sample_rate
        if sample_rate and sample_rate != sr:
            waveform = AF.resample(waveform, sr, sample_rate)
            sr = sample_rate

        waveform = waveform.clamp(-1.0, 1.0)
        pcm = (waveform * 32767.0).round().to(torch.int16)
        return pcm.squeeze(0).numpy().tobytes(), sr

    def pcm_to_wav(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave_open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    async def stream_pcm(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        chunk_duration_ms: Optional[int],
    ) -> AsyncGenerator[str, None]:
        chunk_ms = chunk_duration_ms or DEFAULT_CHUNK_DURATION_MS
        chunk_samples = max(int(sample_rate * (chunk_ms / 1000.0)), 1)
        chunk_size = chunk_samples * 2  # 16-bit mono PCM

        yield _sse_event(
            {
                "event": "start",
                "content_type": "audio/pcm",
                "sample_rate": sample_rate,
                "chunk_duration_ms": chunk_ms,
            }
        )

        if chunk_size <= 0:
            chunk_size = max(sample_rate // 5, 1) * 2

        for offset in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[offset : offset + chunk_size]
            if not chunk:
                continue
            payload = {
                "event": "chunk",
                "audio": base64.b64encode(chunk).decode("ascii"),
            }
            yield _sse_event(payload)
            await asyncio.sleep(0)

        yield _sse_event({"event": "stop"})
        yield "data: [DONE]\n\n"


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


class wave_open:
    """Context manager wrapper around :mod:`wave` that keeps imports local."""

    def __init__(self, buffer: io.BytesIO, mode: str):
        import wave

        self._wave = wave
        self._buffer = buffer
        self._mode = mode
        self._handle = None

    def __enter__(self):
        self._handle = self._wave.open(self._buffer, self._mode)
        return self._handle

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.close()
        return False


def create_openai_tts_app(config: Optional[EngineConfig] = None) -> FastAPI:
    """Create a FastAPI app exposing an OpenAI-style TTS endpoint."""

    engine = TTSEngine(config)
    app = FastAPI(title="Chatterbox OpenAI TTS")

    @app.on_event("startup")
    async def _startup() -> None:
        await engine.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await engine.shutdown()

    @app.get("/healthz")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def audio_speech(request: AudioSpeechRequest):
        waveform = await engine.synthesize(request)
        pcm_bytes, sample_rate = engine.to_pcm16(waveform, request.sample_rate)

        if request.stream:
            generator = engine.stream_pcm(
                pcm_bytes, sample_rate, request.chunk_duration_ms
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )

        if request.response_format.lower() == "pcm16":
            media_type = "audio/pcm"
            body = pcm_bytes
        else:
            media_type = "audio/wav"
            body = engine.pcm_to_wav(pcm_bytes, sample_rate)

        headers = {
            "Content-Type": media_type,
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
        return Response(content=body, media_type=media_type, headers=headers)

    app.state.engine = engine
    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "chatterbox_vllm.server.openai_tts:create_openai_tts_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
    )
