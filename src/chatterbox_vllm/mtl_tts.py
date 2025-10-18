from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import time
import os

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm

REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals):
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,

                   # Original Chatterbox defaults this to False. I don't see a substantial performance difference when running with FP16.
                   s3gen_use_fp16: bool = False,
                   **kwargs) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        # Use multilingual configuration
        t3_config = T3Config.multilingual()

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_weights = load_file(ckpt_dir / "t3_mtl23ls_v2.safetensors")

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') })
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
        t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()
        
        # Heuristic: rough calculation for what percentage of GPU memory to give to vLLM.
        # Tune this until the 'Maximum concurrency for ___ tokens per request: ___x' is just over 1.
        # This rough heuristic gives 1.55GB for the model weights plus 128KB per token.
        vllm_memory_needed = (1.55*1024*1024*1024) + (max_batch_size * max_model_len * 1024 * 128)
        vllm_memory_percent = vllm_memory_needed / unused_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        base_vllm_kwargs = {
            "model": "./t3-multilingual-model",
            "task": "generate",
            "tokenizer": "MTLTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }

        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "main",
                        *args, **kwargs) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                allow_patterns=[
                    "ve.pt", 
                    "t3_mtl23ls_v2.safetensors", 
                    "s3gen.pt", 
                    "grapheme_mtl_merged_expanded_v1.json", 
                    "conds.pt", 
                    "Cangjie5_TC.json"
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        
        # Ensure the symlink in './t3-multilingual-model/model.safetensors' points to t3_mtl23ls_v2.safetensors
        t3_cfg_path = ckpt_dir / "t3_mtl23ls_v2.safetensors"
        model_dir = Path.cwd() / "t3-multilingual-model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_safetensors_path = model_dir / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)
        
        # Copy tokenizer and Cangjie file to model directory
        import shutil
        import json
        
        shutil.copy(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json", model_dir / "grapheme_mtl_merged_expanded_v1.json")
        if (ckpt_dir / "Cangjie5_TC.json").exists():
            shutil.copy(ckpt_dir / "Cangjie5_TC.json", model_dir / "Cangjie5_TC.json")
        
        # Create config.json for vLLM
        config = {
            "architectures": ["ChatterboxT3"],
            "is_multilingual": True,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attn_implementation": "sdpa",
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 2048,  # HACK: doubled from 1024 for CFG
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 16,
            "num_hidden_layers": 30,
            "num_key_value_heads": 16,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 8
        }
        
        # Write config.json
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        return cls.from_local(ckpt_dir, *args, **kwargs)

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            if self.t3_config.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)
            else:
                t3_cond_prompt_tokens = None

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = None
        if t3_cond_prompt_tokens is not None:
            cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        if exaggeration == 0.5:
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    def generate(
        self,
        prompts: Union[str, list[str]],
        language_id: Optional[Union[str, list[str]]] = None,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.4,
        max_tokens=1000, # Capped at max_model_len

        # From original Chatterbox HF generation args
        top_p=0.4,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        """
        Generate speech from text prompts.
        
        Args:
            prompts: Single text string or list of text strings
            language_id: Language code (e.g., 'en', 'es', 'zh') or list of language codes.
                        If None, auto-detection will be used. Must be from SUPPORTED_LANGUAGES.
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration level (0.0-1.0)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
        
        Returns:
            List of generated audio tensors
        """
        # Validate language_id if provided
        if language_id is not None:
            if isinstance(language_id, str):
                lang_ids = [language_id]
            else:
                lang_ids = language_id
            
            for lang_id in lang_ids:
                if lang_id and lang_id.lower() not in SUPPORTED_LANGUAGES:
                    supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
                    raise ValueError(
                        f"Unsupported language_id '{lang_id}'. "
                        f"Supported languages: {supported_langs}"
                    )

        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            language_id=language_id,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            *args, **kwargs
        )

    def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[Union[str, list[str]]] = None,
        temperature: float = 0.4,
        exaggeration: float = 0.5,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 10,

        # From original Chatterbox HF generation args
        top_p=0.4,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Handle language_id
        if language_id is None:
            language_ids = [None] * len(prompts)
        elif isinstance(language_id, str):
            language_ids = [language_id] * len(prompts)
        else:
            language_ids = language_id
            if len(language_ids) != len(prompts):
                raise ValueError(f"Number of language_ids ({len(language_ids)}) must match number of prompts ({len(prompts)})")

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and prepare text; pass language_id via tokenization kwargs to tokenizer
        request_items = []
        for p, lang_id in zip(prompts, language_ids):
            normalized = punc_norm(p)
            # Explicitly inject language tag to ensure correct language routing end-to-end.
            if lang_id:
                text = f"[{lang_id.lower()}][START]{normalized}[STOP]"
            else:
                text = f"[START]{normalized}[STOP]"
            item = {
                "prompt": text,
                "multi_modal_data": {
                    "conditionals": [cond_emb],
                },
            }
            # Keep tokenization_kwargs as a hint (safe if ignored)
            if lang_id:
                item["tokenization_kwargs"] = {"language_id": lang_id.lower()}
            request_items.append(item)

        with torch.inference_mode():
            start_time = time.time()

            # Tail handling configuration (env-tunable)
            tail_crop_k = int(os.environ.get("CHATTERBOX_TAIL_CROP_TOKENS", "2"))
            tail_trim_on = os.environ.get("CHATTERBOX_TAIL_TRIM", "1").lower() in ("1","true","yes","on")
            tail_trim_db = float(os.environ.get("CHATTERBOX_TAIL_TRIM_DB", "-42"))
            tail_trim_safety_ms = int(os.environ.get("CHATTERBOX_TAIL_TRIM_SAFETY_MS", "50"))
            tail_trim_eos_safety_ms = int(os.environ.get("CHATTERBOX_TAIL_TRIM_EOS_SAFETY_MS", "120"))
            # New: enforce a minimum tail duration after last-active to avoid early cutoffs
            tail_trim_min_ms = int(os.environ.get("CHATTERBOX_TAIL_TRIM_MIN_MS", "0"))
            tail_trim_min_eos_ms = int(os.environ.get("CHATTERBOX_TAIL_TRIM_MIN_EOS_MS", "250"))
            rms_window_ms = int(os.environ.get("CHATTERBOX_RMS_WINDOW_MS", "50"))
            rms_hop_ms = int(os.environ.get("CHATTERBOX_RMS_HOP_MS", "20"))
            # Heuristic tail-run detection (repeated filler tokens near the end). Env-tunable.
            tail_run_tokens = set(int(x) for x in os.environ.get("CHATTERBOX_TAIL_RUN_TOKENS", "7486,7487").replace(" ", "").split(",") if x)
            tail_run_min = int(os.environ.get("CHATTERBOX_TAIL_RUN_MIN", "6"))
            tail_run_window = int(os.environ.get("CHATTERBOX_TAIL_RUN_WINDOW", "50"))
            # New: when no EOS is emitted, use a larger search window to catch long gibberish runs
            tail_run_window_noeos = int(os.environ.get("CHATTERBOX_TAIL_RUN_WINDOW_NOEOS", "200"))
            # New: optional low-variance tail truncation when no EOS (disabled by default)
            tail_lowvar_window = int(os.environ.get("CHATTERBOX_TAIL_LOWVAR_WINDOW", "0"))
            tail_lowvar_min = int(os.environ.get("CHATTERBOX_TAIL_LOWVAR_MIN", "120"))
            tail_lowvar_unique_max = int(os.environ.get("CHATTERBOX_TAIL_LOWVAR_UNIQUE_MAX", "4"))
            if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                print(f"[Tail] cfg: crop_k={tail_crop_k}, trim_on={tail_trim_on}, trim_db={tail_trim_db}, "
                      f"safety_ms={tail_trim_safety_ms} (+eos={tail_trim_eos_safety_ms}), win_ms={rms_window_ms}, hop_ms={rms_hop_ms}, "
                      f"run_tokens={sorted(tail_run_tokens)}, run_min={tail_run_min}, run_window={tail_run_window}, "
                      f"run_window_noeos={tail_run_window_noeos}, lowvar_win={tail_lowvar_window}, "
                      f"lowvar_min={tail_lowvar_min}, lowvar_uniq_max={tail_lowvar_unique_max}, "
                      f"min_tail_ms={tail_trim_min_ms}, min_tail_eos_ms={tail_trim_min_eos_ms}")

            # Clamp sampling for multilingual stability when language_id is specified
            temp_use = temperature
            top_p_use = top_p
            rep_use = repetition_penalty
            top_k_use = int(os.environ.get("CHATTERBOX_TOP_K", "0"))  # 0=disabled
            if any(language_ids):
                temp_use = min(temperature, 0.5)
                top_p_use = min(top_p, 0.5)
                # Slightly stronger repetition penalty for multilingual to reduce rambling
                rep_use = max(repetition_penalty, 2.2)
                # Default to a modest top-k for stability if not provided
                if top_k_use == 0:
                    top_k_use = 40
                if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                    print(f"[T3] Clamped multilingual sampling: temperature={temp_use}, top_p={top_p_use}, "
                          f"repetition_penalty={rep_use}, top_k={top_k_use}")

            # Prepare stop token for logging
            stop_offset_id = self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET
            if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                print(f"[T3] SamplingParams -> temperature={temp_use}, top_p={top_p_use}, "
                      f"max_tokens={min(max_tokens, self.max_model_len)}, stop_token_ids={[stop_offset_id]}")

            batch_results = self.t3.generate(
                request_items,
                sampling_params=SamplingParams(
                    temperature=temp_use,

                    stop_token_ids=[stop_offset_id],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p_use,
                    top_k=top_k_use,
                    repetition_penalty=rep_use,

                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            start_time = time.time()
            results = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    if i % 5 == 0:
                        print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                    # Run gc every 10 prompts
                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    # Truncate at the first emitted stop-of-speech token if present
                    stop_offset_id = self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET
                    tok_ids = list(output.token_ids)
                    eos_found = False

                    # Debug: summarize EOS behavior
                    if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                        finish_reason = getattr(output, "finish_reason", None)
                        stop_positions = [idx for idx, t in enumerate(tok_ids) if t == stop_offset_id]
                        head = tok_ids[:10]
                        tail = tok_ids[-10:]
                        print(f"[T3][eos] finish_reason={finish_reason}, out_len={len(tok_ids)}, "
                              f"stop_found={len(stop_positions)>0}, first_stop_idx={stop_positions[0] if stop_positions else None}, "
                              f"head={head}, tail={tail}")

                    if stop_offset_id in tok_ids:
                        stop_idx = tok_ids.index(stop_offset_id)
                        eos_found = True
                        # Adaptive token-tail crop: only drop K tokens if tail shows candidate run tokens
                        tail_seg_start = max(0, stop_idx - min(tail_run_window, stop_idx))
                        tail_segment = tok_ids[tail_seg_start:stop_idx]
                        has_run_token = any(t in tail_run_tokens for t in tail_segment)
                        eff_k = int(tail_crop_k) if has_run_token else 0
                        new_end = max(0, stop_idx - max(0, eff_k))
                        if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                            removed = len(tok_ids) - new_end
                            tail_preview = tok_ids[max(0, new_end-10):new_end]
                            print(f"[T3][eos] truncating at stop_offset_id={stop_offset_id} index={stop_idx}, "
                                  f"crop_k_eff={eff_k} (has_run_token={has_run_token}), new_end={new_end}, "
                                  f"removed={removed}, tail_before_crop={tail_preview}")
                        tok_ids = tok_ids[:new_end]
                    else:
                        # If no EOS, note whether we hit max_tokens
                        if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                            hit_max = len(tok_ids) >= min(max_tokens, self.max_model_len)
                            print(f"[T3][eos] no stop token emitted; hit_max_tokens={hit_max}")

                    # Heuristic tail-run truncation: if a run of the same candidate token(s)
                    # appears near the end (e.g., 7486 repeated), cut to the start of that run.
                    if len(tok_ids) > 0 and ((tail_run_window > 0) or (tail_run_window_noeos > 0)) and tail_run_min > 0 and len(tail_run_tokens) > 0:
                        look_win = tail_run_window if eos_found else max(tail_run_window_noeos, tail_run_window)
                        look_len = min(len(tok_ids), look_win)
                        look = tok_ids[-look_len:]
                        best_start = None
                        best_len = 0
                        best_token = None
                        cur_len = 0
                        for idx, t in enumerate(look):
                            if idx == 0 or t != look[idx - 1]:
                                cur_len = 1
                            else:
                                cur_len += 1
                            if cur_len >= tail_run_min and t in tail_run_tokens:
                                start = len(tok_ids) - look_len + idx - cur_len + 1
                                best_start = start
                                best_len = cur_len
                                best_token = t
                                break  # first occurrence closest to the end is enough
                        if best_start is not None and best_start < len(tok_ids):
                            if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                                preview = tok_ids[max(0, best_start - 10):best_start]
                                print(f"[Tail][run] truncating at run_start={best_start} token={best_token} "
                                      f"run_len={best_len} window={look_len} preview_before={preview}")
                            tok_ids = tok_ids[:best_start]
                        elif (not eos_found) and tail_lowvar_window > 0 and tail_lowvar_min > 0:
                            # Fallback: low-variance tail detection when no EOS and no explicit run found.
                            lv_len = min(len(tok_ids), tail_lowvar_window)
                            window = tok_ids[-lv_len:]
                            unique = len(set(window))
                            if lv_len >= tail_lowvar_min and unique <= tail_lowvar_unique_max:
                                start = len(tok_ids) - lv_len
                                if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                                    print(f"[Tail][lowvar] truncating at start={start} len={lv_len} unique={unique} "
                                          f"(uniq_max={tail_lowvar_unique_max})")
                                tok_ids = tok_ids[:start]

                    # Map back to base speech token space
                    speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in tok_ids], device="cuda")
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]

                    # Hard cap on speech token length as a final guard (env-tunable)
                    max_speech_cap = int(os.environ.get("CHATTERBOX_MAX_SPEECH_TOKENS", "0"))
                    if max_speech_cap > 0 and len(speech_tokens) > max_speech_cap:
                        if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                            print(f"[Tail][cap] truncating speech tokens from {len(speech_tokens)} to {max_speech_cap}")
                        speech_tokens = speech_tokens[:max_speech_cap]

                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )

                    # Energy-based tail trim (RMS) to remove residual low-energy artifacts at the end
                    if tail_trim_on and wav is not None and wav.numel() > 0:
                        sr = self.sr
                        dur_before_ms = (wav.shape[1] * 1000.0) / sr
                        frame_len = max(1, int(sr * rms_window_ms / 1000))
                        hop_len = max(1, int(sr * rms_hop_ms / 1000))
                        if wav.shape[1] >= frame_len:
                            x2 = (wav.unsqueeze(0) ** 2)  # (1, 1, T) after conv1d reshape below
                            x2 = x2 if x2.ndim == 3 else x2.view(1, 1, -1)
                            kernel = torch.ones(1, 1, frame_len, device=wav.device, dtype=wav.dtype) / frame_len
                            rms = torch.sqrt(torch.nn.functional.conv1d(x2, kernel, stride=hop_len)).squeeze()
                            thr = 10.0 ** (tail_trim_db / 20.0)
                            active = torch.where(rms > thr)[0]
                            if active.numel() > 0:
                                last_active = int(active[-1].item())
                                safety_ms_eff = tail_trim_safety_ms + (tail_trim_eos_safety_ms if eos_found else 0)
                                min_tail_ms_eff = tail_trim_min_ms + (tail_trim_min_eos_ms if eos_found else 0)
                                safety = int(sr * safety_ms_eff / 1000)
                                min_tail = int(sr * min_tail_ms_eff / 1000)
                                cut_base = (last_active + 1) * hop_len + safety
                                cut_floor = (last_active + 1) * hop_len + min_tail
                                cut = min(wav.shape[1], max(cut_base, cut_floor))
                                if cut < wav.shape[1]:
                                    if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                                        dur_after_ms = (cut * 1000.0) / sr
                                        print(f"[Tail][rms] trim: win={frame_len} hop={hop_len} thr={thr:.6f} "
                                              f"last_active_frame={last_active} safety={safety} "
                                              f"(eos_found={eos_found}, safety_ms_eff={safety_ms_eff}, min_tail_ms_eff={min_tail_ms_eff}) "
                                              f"dur_before_ms={dur_before_ms:.1f} -> dur_after_ms={dur_after_ms:.1f} "
                                              f"cut_samples={wav.shape[1]-cut}")
                                    wav = wav[:, :cut]
                        else:
                            if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
                                print(f"[Tail][rms] skip: wav too short for frame_len={frame_len} (len={wav.shape[1]})")

                    results.append(wav.cpu())
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Wavform Generation time: {s3gen_gen_time:.2f}s")

            return results
        
    def shutdown(self):
        del self.t3
        torch.cuda.empty_cache()
