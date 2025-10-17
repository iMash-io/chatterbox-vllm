import torch
import numpy as np
from typing import Any, Optional, Tuple, Union, List
from chatterbox_vllm.mtl_tts import ChatterboxMultilingualTTS


class ChatterboxVLLMWrapper:
    """
    Wrapper for ChatterboxMultilingualTTS with vLLM backend.
    Provides a consistent API for TTS generation with caching and conditioning.
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 1000,
        enforce_eager: bool = False,
        compile: bool = False,
        s3gen_use_fp16: bool = False,
        diffusion_steps: int = 10,
    ) -> None:
        """
        Initialize the Chatterbox model wrapper.
        
        Args:
            device: Device to run the model on (default: "cuda")
            gpu_memory_utilization: Fraction of GPU memory for vLLM (default: 0.4)
            max_model_len: Maximum model length in tokens (default: 1000)
            enforce_eager: Disable CUDA graphs for faster startup (default: False)
            compile: Compile the model for better performance (default: False)
            s3gen_use_fp16: Use FP16 for S3Gen (default: False)
            diffusion_steps: Number of diffusion steps for audio generation (default: 10)
        """
        self.device = device
        self.diffusion_steps = diffusion_steps
        
        print(f"Loading Chatterbox vLLM model on {device}...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(
            target_device=device,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            compile=compile,
            s3gen_use_fp16=s3gen_use_fp16,
        )
        
        self.sr = self.model.sr
        
        # Cache for audio conditionals
        self._cached_conditionals = {}
        
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
            
        Returns:
            Tuple of (s3gen_ref_dict, cond_emb, speaker_embedding)
        """
        # Use cached conditionals if available
        cache_key = audio_path or "default"
        if cache_key not in self._cached_conditionals:
            s3gen_ref, cond_emb = self.model.get_audio_conditionals(audio_path)
            self._cached_conditionals[cache_key] = (s3gen_ref, cond_emb)
        else:
            s3gen_ref, cond_emb = self._cached_conditionals[cache_key]
        
        # Update exaggeration if needed
        cond_emb = self.model.update_exaggeration(cond_emb, exaggeration)
        
        # Extract speaker embedding if available (for compatibility)
        speaker_embedding = None
        
        return s3gen_ref, cond_emb, speaker_embedding

    def prepare_conditionals(
        self, 
        audio_path: Optional[str] = None,
        exaggeration: float = 0.5
    ) -> None:
        """
        Prepare and cache conditionals for inference.
        
        Args:
            audio_path: Path to reference audio file
            exaggeration: Emotion exaggeration level (0.0-1.0)
        """
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
            lang: Language code(s) (e.g., 'en', 'es', 'zh')
            audio_prompt_path: Path to reference audio for voice cloning
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            exaggeration: Emotion exaggeration level (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with 'wav' key containing audio tensor(s) in shape [channels, samples]
        """
        # Generate audio
        audios = self.model.generate(
            prompts=text,
            language_id=lang,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            diffusion_steps=self.diffusion_steps,
            **kwargs
        )
        
        # Ensure audio is in correct format [channels, samples]
        if isinstance(text, str):
            # Single input - return single audio
            wav = audios[0]
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu()
                # Ensure 2D format [channels, samples]
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
            else:
                # Convert numpy to torch for consistent output
                wav = torch.from_numpy(wav)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
            return {"wav": wav}
        else:
            # Batch input - return list of audios
            wavs = []
            for audio in audios:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu()
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                else:
                    audio = torch.from_numpy(audio)
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                wavs.append(audio)
            return {"wav": wavs}

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
        
        Args:
            text: Input text or list of texts
            s3gen_ref: S3Gen reference dictionary
            cond_emb: Conditioning embeddings
            lang: Language code(s)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            exaggeration: Emotion exaggeration level
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with 'wav' key containing audio tensor(s) in shape [channels, samples]
        """
        audios = self.model.generate_with_conds(
            prompts=text,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            language_id=lang,
            temperature=temperature,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            diffusion_steps=self.diffusion_steps,
            **kwargs
        )
        
        # Ensure audio is in correct format [channels, samples]
        if isinstance(text, str):
            wav = audios[0]
            if isinstance(wav, torch.Tensor):
                wav = wav.detach().cpu()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
            else:
                wav = torch.from_numpy(wav)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
            return {"wav": wav}
        else:
            wavs = []
            for audio in audios:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu()
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                else:
                    audio = torch.from_numpy(audio)
                    if audio.dim() == 1:
                        audio = audio.unsqueeze(0)
                wavs.append(audio)
            return {"wav": wavs}

    def get_supported_languages(self) -> dict:
        """Get dictionary of supported language codes and names."""
        return self.model.get_supported_languages()

    def clear_cache(self) -> None:
        """Clear cached conditionals."""
        self._cached_conditionals.clear()
        torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Shutdown the model and free resources."""
        self.model.shutdown()
        self._cached_conditionals.clear()


# Example usage
if __name__ == "__main__":
    import torchaudio as ta
    
    # Initialize wrapper
    wrapper = ChatterboxVLLMWrapper(
        device="cuda",
        gpu_memory_utilization=0.4,
        max_model_len=1000,
        enforce_eager=True,
        diffusion_steps=10,
    )
    
    print(f"Sample rate: {wrapper.sr}")
    print(f"Supported languages: {wrapper.get_supported_languages()}")
    
    # Test with different audio prompts
    audio_prompts = [None, "docs/audio-sample-01.mp3", "docs/audio-sample-03.mp3"]
    
    for i, audio_prompt_path in enumerate(audio_prompts):
        print(f"\n=== Testing with audio prompt {i} ===")
        
        # Single text generation
        text = "You are listening to a demo of the Chatterbox TTS model running on VLLM."
        result = wrapper.inference(
            text=text,
            lang="en",
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.2,
            temperature=0.8,
        )
        wav = result["wav"]  # Already in [channels, samples] format
        ta.save(f"test-single-{i}.mp3", wav, wrapper.sr)
        
        # Batch generation
        prompts = [
            "You are listening to a demo of the Chatterbox TTS model running on VLLM.",
            "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox.",
            "你好，今天天气真不错，希望你有一个愉快的周末。",
        ]
        
        result = wrapper.inference(
            text=prompts,
            lang=["en", "fr", "zh"],
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.2,
            temperature=0.8,
        )
        
        wavs = result["wav"]
        for audio_idx, wav in enumerate(wavs):
            ta.save(f"test-batch-{i}-{audio_idx}.mp3", wav, wrapper.sr)
    
    # Cleanup
    wrapper.shutdown()
    print("\nAll tests completed!")
