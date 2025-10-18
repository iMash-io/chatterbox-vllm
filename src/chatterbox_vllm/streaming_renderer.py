from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import math
import os
import torch

from .models.s3gen import S3Gen
from .models.s3tokenizer import S3_TOKEN_RATE


def _tokens_to_samples(n_tokens: int, sr: int, token_rate: float = S3_TOKEN_RATE) -> int:
    return int(round(n_tokens * (sr / token_rate)))


def _cosine_ramp(n: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    # 0..1 inclusive/exclusive shape n
    if n > 0:
        t = torch.linspace(0.0, math.pi, steps=n, device=device, dtype=dtype)
        return (1.0 - torch.cos(t)) * 0.5
    return torch.zeros(0, device=device, dtype=dtype)


@dataclass
class StreamWindowCfg:
    window_tokens: int = 32          # tokens per decoding window (~1.28s @ 25 tps)
    overlap_tokens: int = 8          # token overlap between windows (~0.32s)
    first_window_tokens: int = 24    # smaller first window for quicker TTFA
    overlap_ms: Optional[int] = None # if set, overrides overlap_tokens by time


class S3StreamRenderer:
    """
    Stateless-window streaming renderer for S3Gen.

    Strategy:
    - Decode overlapping token windows with full diffusion steps for quality parity.
    - Crossfade in the audio domain over the overlap region to avoid seams.
    - Emit only the newly committed audio portion each time; keep a tail for next overlap.

    This avoids requiring incremental internal states inside S3Gen and works well with GPU overlap
    (vLLM on one stream, S3Gen/HiFiNet on another).
    """

    def __init__(
        self,
        s3gen: S3Gen,
        ref_dict: dict,
        *,
        sr: int,
        diffusion_steps: int,
        window_cfg: Optional[StreamWindowCfg] = None,
        align_hard: bool = True,
        align_safety_ms: int = 0,
        tail_trim: bool = False,
        tail_trim_db: float = -42.0,
        tail_trim_db_rel: float = -35.0,
        tail_trim_safety_ms: int = 50,
        rms_window_ms: int = 50,
        rms_hop_ms: int = 20,
    ) -> None:
        self.s3gen = s3gen
        self.ref_dict = ref_dict
        self.sr = int(sr)
        self.diffusion_steps = int(diffusion_steps)
        self.window = window_cfg or StreamWindowCfg()

        self.align_hard = align_hard
        self.align_safety = max(0, int(self.sr * align_safety_ms / 1000))

        # Tail processing
        self.tail_trim = tail_trim
        self.tail_trim_db = float(tail_trim_db)
        self.tail_trim_db_rel = float(tail_trim_db_rel)
        self.tail_trim_safety = int(self.sr * tail_trim_safety_ms / 1000)
        self.rms_window = max(1, int(self.sr * rms_window_ms / 1000))
        self.rms_hop = max(1, int(self.sr * rms_hop_ms / 1000))

        # Streaming state
        self._tokens: List[int] = []  # full token history (S3 token IDs)
        self._last_flush_end_tok: int = 0
        self._prev_tail: Optional[torch.Tensor] = None  # (1, N) float32 audio tail kept for next overlap

        # Pre-compute overlap in samples if override is provided
        if self.window.overlap_ms is not None:
            self._overlap_samples_fixed = int(self.sr * self.window.overlap_ms / 1000)
        else:
            self._overlap_samples_fixed = None

        # Optional dedicated CUDA stream for renderer to overlap with vLLM
        use_stream = os.environ.get("CHATTERBOX_RENDER_DEDICATED_STREAM", "0").lower() in ("1", "true", "yes", "on")
        try:
            dev = next(self.s3gen.parameters()).device
        except Exception:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cuda_stream = torch.cuda.Stream(device=dev) if (use_stream and str(dev).startswith("cuda")) else None

    def _decode_window(self, window_tokens: torch.Tensor) -> torch.Tensor:
        """
        Run S3Gen on the provided token window with full diffusion steps.
        Returns float32 audio tensor (1, T) on CPU.
        """
        # S3Gen expects CUDA tensor; ensure correct device
        device = next(self.s3gen.parameters()).device
        toks = window_tokens.to(device=device, dtype=torch.long)

        if self._cuda_stream is not None:
            # Run on dedicated CUDA stream
            with torch.cuda.stream(self._cuda_stream):
                wav, _ = self.s3gen.inference(
                    speech_tokens=toks,
                    ref_dict=self.ref_dict,
                    n_timesteps=self.diffusion_steps,
                )
            # Ensure current stream sees the results
            torch.cuda.current_stream(device=device).wait_stream(self._cuda_stream)
        else:
            wav, _ = self.s3gen.inference(
                speech_tokens=toks,
                ref_dict=self.ref_dict,
                n_timesteps=self.diffusion_steps,
            )  # wav: (1, T) on device dtype float32/fp16
        wav = wav.to(torch.float32).detach().cpu()

        # Length alignment using token-rate expectation for stability
        expected = _tokens_to_samples(int(toks.numel()), sr=self.sr, token_rate=S3_TOKEN_RATE)
        if self.align_hard and wav.shape[1] > 0:
            cap = min(wav.shape[1], expected + self.align_safety)
            wav = wav[:, :cap]

        return wav

    def _crossfade_blend(self, prev_tail: torch.Tensor, new_head: torch.Tensor, overlap_samples: int) -> torch.Tensor:
        """
        Crossfade two mono chunks over overlap_samples and return the fused region:
        prev_tail[-N:] * fade_out + new_head[:N] * fade_in
        """
        if overlap_samples <= 0:
            return torch.zeros(0, dtype=new_head.dtype)

        N = min(overlap_samples, prev_tail.shape[1], new_head.shape[1])
        if N <= 1:
            return torch.zeros(0, dtype=new_head.dtype)

        fade = _cosine_ramp(N, device=new_head.device, dtype=new_head.dtype).unsqueeze(0)
        fade_in = fade
        fade_out = 1.0 - fade

        fused = prev_tail[:, -N:] * fade_out + new_head[:, :N] * fade_in
        return fused

    def _pick_overlap_samples(self, overlap_tokens: int) -> int:
        if self._overlap_samples_fixed is not None:
            return self._overlap_samples_fixed
        return _tokens_to_samples(overlap_tokens, sr=self.sr, token_rate=S3_TOKEN_RATE)

    def feed(self, new_tokens: Iterable[int], *, eos: bool = False) -> Iterator[torch.Tensor]:
        """
        Feed new S3 tokens into the renderer. Yield zero or more float32 mono chunks (1, T)
        that are ready to stream immediately. Call again with eos=True to flush the tail.

        new_tokens: iterable of integer speech token IDs (already cleaned/validated)
        """
        # Append tokens
        for t in new_tokens:
            self._tokens.append(int(t))

        # Determine window constants
        W0 = max(1, int(self.window.first_window_tokens))
        W = max(1, int(self.window.window_tokens))
        O_tok = max(0, int(self.window.overlap_tokens))
        overlap_samples = self._pick_overlap_samples(O_tok)

        # Seal windows and emit chunks
        # First window end: once we have at least W0 tokens
        emitted_any = False

        def emit_from_window(window_end_tok: int) -> Optional[torch.Tensor]:
            nonlocal emitted_any
            start_tok = max(0, window_end_tok - W)
            window_tokens = self._tokens[start_tok:window_end_tok]
            if not window_tokens:
                return None

            wav = self._decode_window(torch.tensor(window_tokens, dtype=torch.long))

            # Commit policy: stream only the "new" region
            # For the first window, we keep a tail for future overlap; emit head-now
            # For subsequent, fuse prev tail with head and emit fused + remainder head-now
            if self._prev_tail is None:
                # First emission
                tail_keep = overlap_samples
                if wav.shape[1] <= tail_keep:
                    # Keep all as tail (not enough to emit yet)
                    self._prev_tail = wav
                    return None
                head = wav[:, :-tail_keep]
                self._prev_tail = wav[:, -tail_keep:]
                emitted_any = True
                return head
            else:
                # Crossfade overlap region with previous tail
                fused_overlap = self._crossfade_blend(self._prev_tail, wav, overlap_samples)
                # Remaining new non-overlap region from current window
                rest = wav[:, overlap_samples:] if wav.shape[1] > overlap_samples else torch.zeros(1, 0, dtype=wav.dtype)
                out = torch.cat([fused_overlap, rest], dim=1)
                # Update tail
                self._prev_tail = wav[:, -overlap_samples:] if wav.shape[1] >= overlap_samples else wav
                emitted_any = True
                return out

        # While we can seal another window, do it
        # Next window end is last_flush_end_tok + (W if first window else W - O)
        # We will keep pushing windows until we cannot.
        while True:
            have = len(self._tokens)
            if self._last_flush_end_tok == 0:
                # Need at least W0
                if have < W0 and not eos:
                    break
                # First window: end at min(have, W)
                window_end = min(have, W) if have >= W0 else have
                if window_end <= 0:
                    break
                chunk = emit_from_window(window_end)
                self._last_flush_end_tok = window_end
                if chunk is not None and chunk.shape[1] > 0:
                    yield chunk
                # Continue loop to see if we can seal more immediately
            else:
                step = max(1, W - self.window.overlap_tokens)
                if have - self._last_flush_end_tok < step and not eos:
                    break
                window_end = min(self._last_flush_end_tok + step, have)
                if window_end <= self._last_flush_end_tok:
                    break
                chunk = emit_from_window(window_end)
                self._last_flush_end_tok = window_end
                if chunk is not None and chunk.shape[1] > 0:
                    yield chunk
                # Continue to try more windows in same call

            # If eos and we've consumed all tokens into windows, exit loop after finalization below
            if not eos:
                # Check again in next iteration if more windows are now allowed
                continue
            else:
                # If eos, loop will try to seal remaining windows; when no step possible, break
                if len(self._tokens) - self._last_flush_end_tok <= 0:
                    break

        # If EOS: flush the remaining tail
        if eos:
            if self._prev_tail is not None and self._prev_tail.shape[1] > 0:
                wav = self._prev_tail

                # Optional RMS-based tail trim
                if self.tail_trim and wav.shape[1] >= self.rms_window:
                    x2 = (wav.unsqueeze(0) ** 2)
                    x2 = x2 if x2.ndim == 3 else x2.view(1, 1, -1)
                    kernel = torch.ones(1, 1, self.rms_window, device=wav.device, dtype=wav.dtype) / self.rms_window
                    rms = torch.sqrt(torch.nn.functional.conv1d(x2, kernel, stride=self.rms_hop)).squeeze()
                    peak = float(rms.max().item()) if rms.numel() > 0 else 0.0
                    thr = (peak * (10.0 ** (self.tail_trim_db_rel / 20.0))) if peak > 0 else 10.0 ** (self.tail_trim_db / 20.0)
                    active = torch.where(rms > thr)[0]
                    if active.numel() > 0:
                        last_active = int(active[-1].item())
                        cut = min(wav.shape[1], (last_active + 1) * self.rms_hop + self.tail_trim_safety)
                        wav = wav[:, :cut]

                yield wav

            # Reset state for safety (one-shot renderer)
            self._prev_tail = None

    @staticmethod
    def defaults_from_env() -> StreamWindowCfg:
        return StreamWindowCfg(
            window_tokens=int(os.environ.get("CHATTERBOX_STREAM_WINDOW_TOKENS", "32")),
            overlap_tokens=int(os.environ.get("CHATTERBOX_STREAM_OVERLAP_TOKENS", "8")),
            first_window_tokens=int(os.environ.get("CHATTERBOX_FIRST_WINDOW_TOKENS", "24")),
            overlap_ms=(int(os.environ["CHATTERBOX_STREAM_OVERLAP_MS"]) if os.environ.get("CHATTERBOX_STREAM_OVERLAP_MS") else None),
        )
