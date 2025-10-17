#!/usr/bin/env python3
import argparse
from typing import Any, Optional, Tuple, Union, List, Dict

import numpy as np
import torch

try:
    import torchaudio as ta
except Exception:
    ta = None

from chatterbox_vllm.mtl_tts import ChatterboxMultilingualTTS
from chatterbox_vllm.tts import ChatterboxTTS


class ChatterboxVLLMWrapper:
    """
    Wrapper for Chatterbox TTS with vLLM backend.
    Provides a consistent API and supports:
      - mode="multi": multilingual, via ChatterboxMultilingualTTS
      - mode="single": English-only, via ChatterboxTTS
    """

    def __init__(
        self,
        *,
        mode: str = "multi",           # "multi" or "single"
        device: str = "cuda",
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 1000,
        enforce_eager: bool = False,
        compile: bool = False,
        s3gen_use_fp16: bool = False,
        diffusion_steps: int = 10,
        use_local_ckpt: Optional[str] = None,
    ) -> None:
        assert mode in ("multi", "single"), "mode must be 'multi' or 'single'"
        self.mode = mode
        self.device = device
        self.diffusion_steps = diffusion_steps

        common_kwargs = dict(
            target_device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            compile=compile,
            s3gen_use_fp16=s3gen_use_fp16,
        )

        print(f"Loading Chatterbox vLLM model (mode={mode}) on {device}...")
        if mode == "multi":
            if use_local_ckpt:
                self.model = ChatterboxMultilingualTTS.from_local(
                    ckpt_dir=use_local_ckpt,
                    **{k: v for k, v in common_kwargs.items() if k not in ("gpu_memory_utilization", "enforce_eager")}
                )
            else:
                self.model = ChatterboxMultilingualTTS.from_pretrained(**common_kwargs)
        else:
            if use_local_ckpt:
                self.model = ChatterboxTTS.from_local(
                    ckpt_dir=use_local_ckpt,
                    target_device=device,
                    variant="english",
                    compile=compile,
                    s3gen_use_fp16=s3gen_use_fp16,
                    max_model_len=max_model_len,
                )
            else:
                self.model = ChatterboxTTS.from_pretrained(
                    target_device=device,
                    s3gen_use_fp16=s3gen_use_fp16,
                    compile=compile,
                    max_model_len=max_model_len,
                )

        self.sr = self.model.sr

        # Cache for audio conditionals
        self._cached_conditionals: Dict[str, Tuple[dict, torch.Tensor]] = {}

        print("Chatterbox vLLM model loaded successfully!")

    def get_conditioning_latents(
        self,
        audio_path: Optional[str] = None,
        exaggeration: float = 0.5
    ) -> Tuple[dict, torch.Tensor, Optional[np.ndarray]]:
        """
        Get conditioning latents for the model.

        Args:
            audio_path: Path to reference audio file (None for default voice)
            exaggeration: Emotion exaggeration level (0.0-1.0)
        """
        cache_key = audio_path or "default"
        if cache_key not in self._cached_conditionals:
            s3gen_ref, cond_emb = self.model.get_audio_conditionals(audio_path)
            self._cached_conditionals[cache_key] = (s3gen_ref, cond_emb)
        else:
            s3gen_ref, cond_emb = self._cached_conditionals[cache_key]

        # Update exaggeration if needed
        cond_emb = self.model.update_exaggeration(cond_emb, exaggeration)

        # Placeholder for compatibility (not used)
        speaker_embedding = None
        return s3gen_ref, cond_emb, speaker_embedding

    def prepare_conditionals(
        self,
        audio_path: Optional[str] = None,
        exaggeration: float = 0.5
    ) -> None:
        """Prepare and cache conditionals for inference."""
        self.get_conditioning_latents(audio_path, exaggeration)

    def inference(
        self,
        text: Union[str, List[str]],
        lang: Optional[Union[str, List[str]]] = None,
        audio_prompt_path: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 2.0,
        exaggeration: float = 0.5,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """
        Generate speech from text.

        Args:
            text: Input text or list of texts
            lang: Language code(s); used only in mode='multi'. Ignored in 'single' (assumed 'en').
            audio_prompt_path: Path to reference audio for voice cloning
        """
        # Route language_id based on mode
        language_id = lang if self.mode == "multi" else "en"

        audios = self.model.generate(
            prompts=text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            diffusion_steps=self.diffusion_steps,
            **kwargs
        )

        return self._normalize_audio_output(text, audios)

    def inference_with_conds(
        self,
        text: Union[str, List[str]],
        s3gen_ref: dict,
        cond_emb: torch.Tensor,
        lang: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.8,
        top_p: float = 0.8,
        repetition_penalty: float = 2.0,
        exaggeration: float = 0.5,
        max_tokens: int = 1000,
        **kwargs
    ) -> dict:
        """
        Generate speech with pre-computed conditionals.
        """
        language_id = lang if self.mode == "multi" else "en"

        audios = self.model.generate_with_conds(
            prompts=text,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            language_id=language_id,
            temperature=temperature,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            diffusion_steps=self.diffusion_steps,
            **kwargs
        )

        return self._normalize_audio_output(text, audios)

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported language codes and names."""
        if self.mode == "multi":
            return self.model.get_supported_languages()
        return {"en": "English"}

    def clear_cache(self) -> None:
        """Clear cached conditionals."""
        self._cached_conditionals.clear()
        torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Shutdown the model and free resources."""
        self.model.shutdown()
        self._cached_conditionals.clear()

    def _normalize_audio_output(self, text: Union[str, List[str]], audios: List[torch.Tensor]) -> dict:
        """
        Ensure audio is in [channels, samples] torch.FloatTensor(1, T).
        """
        def to_2d_tensor(wav_like) -> torch.Tensor:
            if isinstance(wav_like, torch.Tensor):
                wav = wav_like.detach().cpu()
            else:
                wav = torch.from_numpy(np.asarray(wav_like))
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            return wav

        if isinstance(text, str):
            return {"wav": to_2d_tensor(audios[0])}
        else:
            return {"wav": [to_2d_tensor(w) for w in audios]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chatterbox vLLM Inference (single/multi)")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--multi", action="store_true", help="Use multilingual model")
    mode_group.add_argument("--single", action="store_true", help="Use English-only model")

    p.add_argument("--device", default="cuda", help="Device to run the model on (cuda|cpu)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4, help="Fraction of GPU memory for vLLM")
    p.add_argument("--max-model-len", type=int, default=1000, help="Max model length (tokens)")
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs for faster startup")
    p.add_argument("--compile", action="store_true", help="Compile the model for perf")
    p.add_argument("--s3gen-fp16", action="store_true", help="Use FP16 for S3Gen")
    p.add_argument("--diffusion-steps", type=int, default=10, help="Steps for S3Gen diffusion")
    p.add_argument("--local-ckpt", type=str, default=None, help="Path to local checkpoint directory")
    p.add_argument("--save-audio", action="store_true", help="Save example outputs to disk (requires torchaudio)")
    return p.parse_args()


def main():
    args = parse_args()
    mode = "multi" if (args.multi or not args.single) else "single"

    wrapper = ChatterboxVLLMWrapper(
        mode=mode,
        device=args.device,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        compile=args.compile,
        s3gen_use_fp16=args.s3gen_fp16,
        diffusion_steps=args.diffusion_steps,
        use_local_ckpt=args.local_ckpt,
    )

    print(f"Sample rate: {wrapper.sr}")
    print(f"Supported languages: {wrapper.get_supported_languages()}")

    # Simple quick tests
    audio_prompts = [None]  # optionally add your own prompt paths

    for i, audio_prompt_path in enumerate(audio_prompts):
        print(f"\n=== Testing with audio prompt {i} (mode={mode}) ===")

        # Single text generation
        text = "You are listening to a demo of the Chatterbox TTS model running on VLLM."
        lang = "en" if mode == "single" else "en"
        result = wrapper.inference(
            text=text,
            lang=lang,
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.2,
            temperature=0.8,
        )
        wav = result["wav"]
        if args.save_audio and ta is not None:
            ta.save(f"test-single-{mode}-{i}.mp3", wav, wrapper.sr)

        # Batch generation
        if mode == "multi":
            prompts = [
                "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
                "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox.",
                "שלום, מה שלומך? זהו מודל הדיבור הרב-לשוני של Chatterbox.",
            ]
            langs = ["en", "fr", "he"]
        else:
            prompts = [
                "This is the English-only TTS model demo one.",
                "This is the English-only TTS model demo two.",
                "This is the English-only TTS model demo three.",
            ]
            langs = None  # ignored in single mode

        result = wrapper.inference(
            text=prompts,
            lang=langs,
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.2,
            temperature=0.8,
        )
        wavs = result["wav"]
        if args.save_audio and ta is not None:
            for audio_idx, wav in enumerate(wavs):
                ta.save(f"test-batch-{mode}-{i}-{audio_idx}.mp3", wav, wrapper.sr)

    # Cleanup
    wrapper.shutdown()
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
