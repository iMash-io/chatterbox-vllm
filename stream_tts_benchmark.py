#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stream_tts_benchmark.py
Test an OpenAI-compatible TTS endpoint with streaming and detailed timings.

Features:
- Live playback via ffplay (PCM/WAV/MP3)
- Saves streamed bytes to file
- Latency metrics: TTH (headers), TTFA (first audio), totals, throughput
- Auto-wrap raw PCM (s16le) to WAV for easy playback (--also-wav)

Examples:
  python stream_tts_benchmark.py --play
  python stream_tts_benchmark.py --response-format pcm --also-wav out.wav
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import wave
from typing import Tuple, Optional

import requests
import uuid


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def parse_audio_ct(ct: Optional[str]) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Parse Content-Type like:
      audio/pcm;rate=24000;channels=1
      audio/wav
      audio/mpeg
    Returns: (kind, rate, channels)
      kind in {"pcm", "wav", "mp3", "other"}
    """
    if not ct:
        return "other", None, None
    ct = ct.lower()
    kind = "other"
    if ct.startswith("audio/pcm"):
        kind = "pcm"
    elif ct.startswith("audio/wav") or ct.startswith("audio/x-wav"):
        kind = "wav"
    elif ct.startswith("audio/mpeg") or ct.startswith("audio/mp3"):
        kind = "mp3"

    rate = None
    ch = None
    if ";" in ct:
        parts = [p.strip() for p in ct.split(";")[1:]]
        for p in parts:
            if p.startswith("rate="):
                try:
                    rate = int(p.split("=", 1)[1])
                except Exception:
                    pass
            elif p.startswith("channels="):
                try:
                    ch = int(p.split("=", 1)[1])
                except Exception:
                    pass
    return kind, rate, ch


def write_pcm_to_wav(pcm_path: Path, wav_path: Path, sample_rate: int, channels: int) -> None:
    """Wrap raw PCM s16le -> WAV."""
    data = pcm_path.read_bytes()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # s16le
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--url",
        default="https://97yesgvn9neq5y-8888.proxy.runpod.net/v1/audio/speech",
        help="TTS endpoint URL",
    )
    p.add_argument("--model", default="gpt-4o-mini-tts", help="Model name")
    p.add_argument(
        "--input",
        default="Hello from chatterbox. This is a low-latency streaming test.",
        help="Text to synthesize",
    )
    p.add_argument("--voice", default="alloy", help="Voice name/path as server expects")

    p.add_argument(
        "--response-format",
        dest="response_format",
        default="pcm",
        choices=["pcm", "wav", "mp3"],
        help='Body field "response_format"',
    )
    p.add_argument("--sample-rate", type=int, default=24000, help="Desired sample rate (Hz)")
    p.add_argument("--channels", type=int, default=1, help="Assumed channels for PCM fallback/wrap")
    p.add_argument("--language-id", dest="language_id", default="en", help='Body field "language_id"')
    p.add_argument("--watermark", default="off", choices=["off", "on"], help='Body field "watermark"')
    p.add_argument("--diffusion-steps", dest="diffusion_steps", type=int, default=10, help="Body field diffusion_steps")
    p.add_argument("--primer-silence-ms", dest="primer_silence_ms", type=int, default=0, help='Body field "primer_silence_ms" (ms of initial silence to flush headers)')
    p.add_argument("--first-chunk-diff-steps", dest="first_chunk_diff_steps", type=int, default=3, help='Body field "first_chunk_diff_steps"')
    p.add_argument("--first-chunk-chars", dest="first_chunk_chars", type=int, default=60, help='Body field "first_chunk_chars"')
    p.add_argument("--chunk-chars", dest="chunk_chars", type=int, default=120, help='Body field "chunk_chars"')
    p.add_argument("--frame-ms", dest="frame_ms", type=int, default=20, help='Body field "frame_ms" (PCM frame size)')
    p.add_argument("--max-tokens-first-chunk", dest="max_tokens_first_chunk", type=int, default=None, help="Body field 'max_tokens_first_chunk' (caps T3 tokens for first chunk)")
    p.add_argument("--crossfade-ms", dest="crossfade_ms", type=int, default=10, help="Body field 'crossfade_ms' (ms crossfade between chunks)")

    p.add_argument(
        "--outfile",
        default="out.pcm",
        help="File to save stream (use .wav for WAV, .mp3 for MP3, .pcm for raw PCM)",
    )
    p.add_argument(
        "--also-wav",
        metavar="WAV_PATH",
        default=None,
        help="If response is raw PCM, also write a WAV here (e.g., out.wav).",
    )
    p.add_argument(
        "--play",
        action="store_true",
        help="Pipe stream to ffplay for live playback (also saves to file)",
    )
    p.add_argument("--timeout", type=float, default=None, help="Request timeout (seconds)")
    p.add_argument("--show-headers", action="store_true", help="Dump all response headers")
    args = p.parse_args()

    # Build payload (mirror your cURL)
    payload = {
        "model": args.model,
        "voice": args.voice,
        "input": args.input,
        "response_format": args.response_format,  # "pcm" | "wav" | "mp3"
        "stream": True,
        "language_id": args.language_id,
        "watermark": args.watermark,
        "diffusion_steps": args.diffusion_steps,
        "sample_rate": args.sample_rate,
        # Low-TTFA tuning knobs
        "primer_silence_ms": args.primer_silence_ms,
        "first_chunk_diff_steps": args.first_chunk_diff_steps,
        "first_chunk_chars": args.first_chunk_chars,
        "chunk_chars": args.chunk_chars,
        "frame_ms": args.frame_ms,
        "max_tokens_first_chunk": args.max_tokens_first_chunk,
        "crossfade_ms": args.crossfade_ms,
    }

    client_req_id = uuid.uuid4().hex
    client_epoch_ms = int(time.time() * 1000)
    headers = {
        "Content-Type": "application/json",
        "X-Client-Request-Id": client_req_id,
        "X-Client-Start-Epoch-Ms": str(client_epoch_ms),
    }

    # Optional ffplay
    ffplay_proc = None
    if args.play:
        if shutil.which("ffplay") is None:
            print("⚠️  --play requested but ffplay not found in PATH. Continuing without live playback.")
        else:
            # For PCM we must specify raw format; for WAV/MP3 let ffplay detect.
            if args.response_format == "pcm":
                # NOTE: some ffplay builds complain about -ac; omit it and rely on mono source.
                ff_cmd = [
                    "ffplay",
                    "-autoexit",
                    "-nodisp",
                    "-f",
                    "s16le",
                    "-ar",
                    str(args.sample_rate),
                    "-i",
                    "-",
                ]
            else:
                ff_cmd = ["ffplay", "-autoexit", "-nodisp", "-i", "-"]
            ffplay_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)

    t0 = time.perf_counter()
    try:
        with requests.post(
            args.url, headers=headers, data=json.dumps(payload), stream=True, timeout=args.timeout
        ) as resp:
            t_headers = time.perf_counter()
            try:
                resp.raise_for_status()
            except requests.HTTPError:
                print(f"HTTP {resp.status_code}")
                try:
                    print(json.dumps(resp.json(), indent=2))
                except Exception:
                    print(resp.text[:2000])
                return 2

            # Parse Content-Type to refine playback/wrap params
            ct = resp.headers.get("content-type", "")
            kind, ct_rate, ct_ch = parse_audio_ct(ct)
            effective_rate = ct_rate or args.sample_rate
            effective_ch = ct_ch or args.channels

            print("🔎 Response headers:")
            if args.show_headers:
                for k, v in resp.headers.items():
                    print(f"   {k.lower()}: {v}")
            else:
                for k in [
                    "content-type",
                    "transfer-encoding",
                    "content-length",
                    "server-timing",
                    "x-request-id",
                ]:
                    if k in resp.headers:
                        print(f"   {k}: {resp.headers[k]}")
            print(f"🔎 Parsed content-type -> kind={kind} rate={effective_rate} ch={effective_ch}")
            te = (resp.headers.get("transfer-encoding", "") or "").lower()
            cl = resp.headers.get("content-length")
            if cl:
                print(f"⚠️  Streaming response included Content-Length={cl}. A proxy or server may be buffering the body.")
            if te and "chunked" in te:
                print("✅ Transfer-Encoding: chunked (streaming)")
            elif not cl:
                print("ℹ️  No Content-Length seen; proxy may still buffer the first chunk before forwarding.")
            # Correlate with server timing headers for cross-reference
            srv_req_id = resp.headers.get("x-req-id", "")
            srv_start_ms = resp.headers.get("x-server-start-epoch-ms", "")
            try:
                srv_start_ms_int = int(srv_start_ms)
            except Exception:
                srv_start_ms_int = None
            approx_skew = (srv_start_ms_int - client_epoch_ms) if srv_start_ms_int is not None else None
            print(f"↔️  Correlate IDs: client_req_id={client_req_id} server_req_id={srv_req_id}")
            print(f"🕒 Epochs: client_start_ms={client_epoch_ms} server_start_ms={srv_start_ms} approx_skew_ms={approx_skew}")

            # If server returned PCM but you asked for WAV/MP3 (or vice versa), we still just stream bytes;
            # playback/wrap uses the parsed content-type for correctness.
            t_first = None
            total_bytes = 0
            chunks = 0
            out_path = Path(args.outfile)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # If ffplay is running and server says PCM (even if you asked for WAV), just continue;
            # we already configured raw mode above. No need to restart ffplay here.

            # Stream body
            with out_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                        print(f"⏱️  TTFA (first audio bytes): {(t_first - t_headers) * 1000:.2f} ms")
                        print(
                            f"⏱️  Time to headers (connect+TLS+server accept): {(t_headers - t0) * 1000:.2f} ms"
                        )
                        print(f"⏱️  TTFA (request->first bytes): {(t_first - t0) * 1000:.2f} ms")
                        # Compute approximate server-view TTFA using epochs if available
                        try:
                            client_first_epoch_ms = int(client_epoch_ms + (t_first - t0) * 1000)
                            if 'srv_start_ms_int' in locals() and srv_start_ms_int is not None:
                                approx_ttfa_server_ms = client_first_epoch_ms - srv_start_ms_int
                                print(f"⏱️  Approx server-view TTFA (epochs): {approx_ttfa_server_ms} ms")
                                print(f"↔️  Epochs: client_first_epoch_ms={client_first_epoch_ms} server_start_epoch_ms={srv_start_ms_int} skew_ms~={(srv_start_ms_int - client_epoch_ms) if srv_start_ms_int is not None else 'n/a'}")
                        except Exception:
                            pass

                    total_bytes += len(chunk)
                    chunks += 1
                    f.write(chunk)

                    if ffplay_proc and ffplay_proc.stdin:
                        try:
                            ffplay_proc.stdin.write(chunk)
                            ffplay_proc.stdin.flush()
                        except BrokenPipeError:
                            ffplay_proc = None  # playback closed

            t_done = time.perf_counter()

            if t_first is None:
                print("❌ No audio bytes received.")
                return 3

            total_time = t_done - t0
            stream_time = t_done - t_headers
            data_time = t_done - t_first
            mb = total_bytes / (1024 * 1024)
            avg_mbps = (mb / data_time) if data_time > 0 else 0.0

            print("\n✅ Done.")
            print(f"📦 Saved: {out_path.resolve()}")
            print(f"📊 Bytes: {total_bytes} ({human_bytes(total_bytes)})  •  Chunks: {chunks}")
            print(f"⏱️  Time to headers: {(t_headers - t0) * 1000:.2f} ms")
            print(f"⏱️  TTFA (post-headers): {(t_first - t_headers) * 1000:.2f} ms")
            print(f"⏱️  TTFA (request->first bytes): {(t_first - t0) * 1000:.2f} ms")
            print(f"⏱️  Total time: {total_time * 1000:.2f} ms  •  Streaming window: {stream_time * 1000:.2f} ms")
            print(f"🚀 Avg throughput after first byte: {avg_mbps:.2f} MB/s")

            # Optional: if we got raw PCM, also write a WAV wrapper for convenience
            if (kind == "pcm" or args.response_format == "pcm") and args.also_wav:
                wav_path = Path(args.also_wav)
                sr = int(effective_rate or args.sample_rate)
                ch = int(effective_ch or args.channels)
                try:
                    write_pcm_to_wav(out_path, wav_path, sr, ch)
                    print(f"🎧 Also wrote WAV wrapper -> {wav_path.resolve()}  (sr={sr}, ch={ch})")
                except Exception as e:
                    print(f"⚠️ Failed to write WAV wrapper: {e}")

    finally:
        if ffplay_proc:
            try:
                if ffplay_proc.stdin:
                    ffplay_proc.stdin.close()
            except Exception:
                pass
            try:
                ffplay_proc.wait(timeout=5)
            except Exception:
                try:
                    ffplay_proc.kill()
                except Exception:
                    pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
