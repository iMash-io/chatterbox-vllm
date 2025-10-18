from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import time
import os
import re
import math

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import shutil

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm, SUPPORTED_LANGUAGES

REPO_ID = "ResembleAI/chatterbox"

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


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc, 
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: S3Gen, ve: VoiceEncoder, default_conds: Conditionals,
                 variant: str = "english"):
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
        self.variant = variant

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,
                   variant: str = "english",

                   # Original Chatterbox defaults this to False. I don't see a substantial performance difference when running with FP16.
                   s3gen_use_fp16: bool = False,
                   **kwargs) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_weights = load_file(ckpt_dir / ("t3_cfg.safetensors" if variant == "english" else "t3_23lang.safetensors"))

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
            "model": "./t3-model" if variant == "english" else "./t3-model-multilingual",
            "task": "generate",
            "tokenizer": "EnTokenizer" if variant == "english" else "MtlTokenizer",
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }

        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device).eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
            variant=variant,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                        *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        # Ensure tokenizer.json is available to the custom EnTokenizer resolver:
        # Copy the HF-downloaded tokenizer.json into ./t3-model/tokenizer.json so vLLM picks the correct vocab.
        try:
            tok_src = Path(local_path).parent / "tokenizer.json"
            tok_dst_dir = Path.cwd() / "t3-model"
            tok_dst_dir.mkdir(parents=True, exist_ok=True)
            tok_dst = tok_dst_dir / "tokenizer.json"
            if tok_src.exists():
                tok_dst.unlink(missing_ok=True)
                shutil.copy(tok_src, tok_dst)
                print(f"[TTS] Copied tokenizer.json to {tok_dst}")
            else:
                print("[TTS][WARN] tokenizer.json not found next to weights; relying on package-local fallback")
        except Exception as _e:
            print(f"[TTS][WARN] Failed to copy tokenizer.json: {_e}")

        return cls.from_local(Path(local_path).parent, variant="english", *args, **kwargs)

    @classmethod
    def from_pretrained_multilingual(cls,
                                    repo_id: str = REPO_ID,
                                    revision: str = "c819eeccdf99310da26bca3bc5ace120db93471a",
                                    *args, **kwargs) -> 'ChatterboxTTS':
        for fpath in ["ve.safetensors", "t3_23lang.safetensors", "s3gen.safetensors", "mtl_tokenizer.json", "conds.pt", "Cangjie5_TC.json"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model-multilingual/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_23lang.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model-multilingual" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="multilingual", *args, **kwargs)
    
    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return { "en": "English" }

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
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

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
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        # From original Chatterbox HF generation args
        top_p=0.8,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            language_id=language_id,
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
        language_id: Optional[str] = 'en',
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 10,

        # From original Chatterbox HF generation args
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and tokenize text
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            # Use angle brackets to avoid conflicts with other start/stop tokens.
            # This will be parsed and replaced in the tokenizer.
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        with torch.inference_mode():
            # Tail handling and debug configuration (env-tunable; mirroring multilingual path)
            tail_crop_k = int(os.environ.get("CHATTERBOX_TAIL_CROP_TOKENS", "2"))
            tail_trim_on = os.environ.get("CHATTERBOX_TAIL_TRIM", "1").lower() in ("1","true","yes","on")
            tail_trim_db = float(os.environ.get("CHATTERBOX_TAIL_TRIM_DB", "-42"))
            tail_trim_db_rel = float(os.environ.get("CHATTERBOX_TAIL_TRIM_DB_REL", "-35"))
            tail_trim_safety_ms = int(os.environ.get("CHATTERBOX_TAIL_TRIM_SAFETY_MS", "50"))
            rms_window_ms = int(os.environ.get("CHATTERBOX_RMS_WINDOW_MS", "50"))
            rms_hop_ms = int(os.environ.get("CHATTERBOX_RMS_HOP_MS", "20"))
            dbg = os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on")
            if dbg:
                print(f"[Tail] cfg: crop_k={tail_crop_k}, trim_on={tail_trim_on}, trim_db={tail_trim_db}, "
                      f"safety_ms={tail_trim_safety_ms}, win_ms={rms_window_ms}, hop_ms={rms_hop_ms}")

            # Clamp sampling for stability when a language_id is provided (English path defaults to 'en')
            temp_use = temperature
            top_p_use = top_p
            if language_id:
                temp_use = min(temperature, 0.5)
                top_p_use = min(top_p, 0.5)
                if dbg:
                    print(f"[T3] Clamped English sampling: temperature={temp_use}, top_p={top_p_use}")
            # Optional deterministic decoding for reproducibility/debugging
            if os.environ.get("CHATTERBOX_DETERMINISTIC", "0").lower() in ("1","true","yes","on"):
                temp_use = 0.0
                top_p_use = 1.0
                if dbg:
                    print(f"[T3] Deterministic mode: forcing temperature={temp_use}, top_p={top_p_use}")

            # Prepare request_items and pre-cap tokens by estimated prompt length
            request_items = []
            for p in prompts:
                request_items.append({
                    "prompt": p,
                    "multi_modal_data": { "conditionals": [cond_emb] },
                })

            try:
                tpc = float(os.environ.get("CHATTERBOX_TOKENS_PER_CHAR", "2.2"))
                tmin = int(os.environ.get("CHATTERBOX_TOKENS_MIN", "64"))
                tmax_env = int(os.environ.get("CHATTERBOX_TOKENS_MAX", "1200"))
                guard_mult = float(os.environ.get("CHATTERBOX_TOKENS_GUARD_MULT", "1.6"))
                pre_margin = int(os.environ.get("CHATTERBOX_PRE_GUARD_MARGIN", "16"))
                caps = []
                for idx, it in enumerate(request_items):
                    ptxt = it.get("prompt", "")
                    pclean = re.sub(r"\[[^\]]+\]", "", ptxt)
                    clen = sum(1 for c in pclean if not c.isspace())
                    est = int(math.ceil(clen * tpc))
                    est = max(tmin, min(tmax_env, est))
                    gcap = int(math.ceil(est * guard_mult))
                    caps.append(gcap)
                    if dbg:
                        print(f"[PreCap] idx={idx} char_len={clen} est={est} guard={gcap}")
                if caps:
                    pre_cap_tokens = max(1, min(caps) + max(0, pre_margin))
                else:
                    pre_cap_tokens = min(max_tokens, self.max_model_len)
                pre_cap_tokens = min(pre_cap_tokens, min(max_tokens, self.max_model_len))
                if dbg:
                    print(f"[PreCap] chosen_max_tokens={pre_cap_tokens} (margin={pre_margin})")
            except Exception as _e:
                pre_cap_tokens = min(max_tokens, self.max_model_len)
                if dbg:
                    print(f"[PreCap] error; falling back to max_tokens={pre_cap_tokens}: {_e}")

            # Stop token id for truncation and stopping conditions
            stop_offset_id = self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET
            if dbg:
                print(f"[T3] SamplingParams -> temperature={temp_use}, top_p={top_p_use}, "
                      f"max_tokens={pre_cap_tokens}, stop_token_ids={[stop_offset_id]}")

            start_time = time.time()
            # Dynamic control of EOS handling in vLLM:
            # - If CHATTERBOX_VLLM_USE_STOP_TOKEN is enabled, let vLLM stop on EOS.
            # - Otherwise, disable EOS stopping (ignore_eos=True) and handle early/late EOS ourselves.
            use_vllm_stop = os.environ.get("CHATTERBOX_VLLM_USE_STOP_TOKEN", "0").lower() in ("1", "true", "yes", "on")
            eos_bias_str = os.environ.get("CHATTERBOX_EOS_LOGIT_BIAS", None)
            logit_bias = None
            if eos_bias_str is not None and str(eos_bias_str).strip() != "":
                try:
                    logit_bias = {int(stop_offset_id): float(eos_bias_str)}
                except Exception:
                    logit_bias = None

            if use_vllm_stop:
                sampling_params = SamplingParams(
                    temperature=temp_use,
                    stop_token_ids=[stop_offset_id],
                    max_tokens=pre_cap_tokens,
                    top_p=top_p_use,
                    repetition_penalty=repetition_penalty,
                    logit_bias=logit_bias,
                    *args, **kwargs,
                )
            else:
                sampling_params = SamplingParams(
                    temperature=temp_use,
                    ignore_eos=True,
                    max_tokens=pre_cap_tokens,
                    top_p=top_p_use,
                    repetition_penalty=repetition_penalty,
                    logit_bias=logit_bias,
                    *args, **kwargs,
                )

            batch_results = self.t3.generate(
                request_items,
                sampling_params=sampling_params
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            # Post-process token streams -> audio with alignment and tail trims
            start_time = time.time()
            results = []
            for i, batch_result in enumerate(batch_results):
                for output in batch_result.outputs:
                    if i % 5 == 0:
                        print(f"[S3] Processing prompt {i} of {len(batch_results)}")
                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                    tok_ids = list(output.token_ids)

                    # Debug EOS behavior
                    if dbg:
                        finish_reason = getattr(output, "finish_reason", None)
                        stop_positions = [idx for idx, t in enumerate(tok_ids) if t == stop_offset_id]
                        head = tok_ids[:10]
                        tail = tok_ids[-10:]
                        print(f"[T3][eos] finish_reason={finish_reason}, out_len={len(tok_ids)}, "
                              f"stop_found={len(stop_positions)>0}, first_stop_idx={stop_positions[0] if stop_positions else None}, "
                              f"head={head}, tail={tail}")

                    # Dynamic EOS handling:
                    # - If EOS appears too early (before a fraction of estimated tokens), ignore it.
                    # - Otherwise truncate at EOS (with optional tail crop).
                    # - If no acceptable EOS, rely on guards/repetition trimming below.
                    stop_positions = [idx for idx, t in enumerate(tok_ids) if t == stop_offset_id]
                    if stop_positions:
                        try:
                            prompt_str = request_items[i].get("prompt", "")
                            prompt_clean = re.sub(r"\[[^\]]+\]", "", prompt_str)
                            char_len = sum(1 for c in prompt_clean if not c.isspace())
                            tpc = float(os.environ.get("CHATTERBOX_TOKENS_PER_CHAR", "2.2"))
                            tmin = int(os.environ.get("CHATTERBOX_TOKENS_MIN", "64"))
                            tmax = int(os.environ.get("CHATTERBOX_TOKENS_MAX", "1200"))
                            est_tokens = max(tmin, min(tmax, int(math.ceil(char_len * tpc))))
                            sup_frac = float(os.environ.get("CHATTERBOX_SUPPRESS_EOS_FRAC", "0.7"))
                            early_min = int(os.environ.get("CHATTERBOX_EARLY_EOS_MIN", "48"))
                            early_thresh = max(early_min, int(math.ceil(est_tokens * sup_frac)))
                        except Exception as _e:
                            early_thresh = 0
                            if dbg:
                                print(f"[T3][eos] early threshold compute failed, not suppressing early EOS: {_e}")

                        valid_stops = [pos for pos in stop_positions if pos >= early_thresh]
                        if valid_stops:
                            stop_idx = valid_stops[0]
                            new_end = max(0, stop_idx - max(0, int(tail_crop_k)))
                            if dbg:
                                removed = len(tok_ids) - new_end
                                tail_preview = tok_ids[max(0, new_end-10):new_end]
                                print(f"[T3][eos] truncating at stop_offset_id={stop_offset_id} index={stop_idx}, "
                                      f"crop_k={tail_crop_k}, new_end={new_end}, removed={removed}, tail_before_crop={tail_preview}, "
                                      f"early_thresh={early_thresh}")
                            tok_ids = tok_ids[:new_end]
                        else:
                            if dbg:
                                print(f"[T3][eos] early EOS suppressed (positions={stop_positions}, early_thresh={early_thresh}); not truncating here")
                    else:
                        if dbg:
                            hit_max = len(tok_ids) >= pre_cap_tokens
                            print(f"[T3][eos] no stop token emitted; hit_max_tokens={hit_max}")

                    # Repetition-based late cutoff: if trailing run of the last token is too long, truncate it
                    try:
                        max_repeat = int(os.environ.get("CHATTERBOX_REPEAT_RUN_MAX", "3"))
                        if max_repeat >= 2 and len(tok_ids) > max_repeat:
                            last_tok = tok_ids[-1]
                            run_len = 1
                            for j2 in range(len(tok_ids) - 2, -1, -1):
                                if tok_ids[j2] == last_tok:
                                    run_len += 1
                                else:
                                    break
                            if run_len >= max_repeat:
                                cut = len(tok_ids) - run_len
                                if dbg:
                                    print(f"[T3][rep] truncating repetition run: token={last_tok}, run_len={run_len}, cut={cut}")
                                tok_ids = tok_ids[:cut]
                    except Exception as _e:
                        if dbg:
                            print(f"[T3][rep] repetition trim skipped: {_e}")

                    # Map to base speech token space and drop invalids
                    speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in tok_ids], device="cuda")
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[speech_tokens < 6561]

                    # Dynamic overrun guard based on character length estimate
                    try:
                        prompt_str = request_items[i].get("prompt", "")
                        prompt_clean = re.sub(r"\[[^\]]+\]", "", prompt_str)
                        char_len = sum(1 for c in prompt_clean if not c.isspace())
                        tpc = float(os.environ.get("CHATTERBOX_TOKENS_PER_CHAR", "2.2"))
                        tmin = int(os.environ.get("CHATTERBOX_TOKENS_MIN", "64"))
                        tmax = int(os.environ.get("CHATTERBOX_TOKENS_MAX", "1200"))
                        guard_mult = float(os.environ.get("CHATTERBOX_TOKENS_GUARD_MULT", "1.6"))
                        est_tokens = int(math.ceil(char_len * tpc))
                        est_tokens = max(tmin, min(tmax, est_tokens))
                        guard_cap = int(math.ceil(est_tokens * guard_mult))
                        n_tokens_cur = int(speech_tokens.shape[0]) if speech_tokens.ndim == 1 else int(speech_tokens.shape[1])
                        if n_tokens_cur > guard_cap:
                            if dbg:
                                print(f"[Guard] char_len={char_len} est={est_tokens} guard={guard_cap} n_tokens={n_tokens_cur} -> cap to {guard_cap}")
                            speech_tokens = speech_tokens[:guard_cap]
                    except Exception as _e:
                        if dbg:
                            print(f"[Guard] skipped due to error: {_e}")

                    # Render audio
                    wav, _ = self.s3gen.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=s3gen_ref,
                        n_timesteps=diffusion_steps,
                    )

                    # Hard alignment of audio duration to token-derived expectation
                    try:
                        n_tokens = int(speech_tokens.shape[1]) if speech_tokens.ndim > 1 else int(speech_tokens.numel())
                    except Exception:
                        n_tokens = int(speech_tokens.numel())
                    expected_samples = int(round(n_tokens * (self.sr / S3_TOKEN_RATE)))
                    align_on = os.environ.get("CHATTERBOX_ALIGN_HARD", "1").lower() in ("1","true","yes","on")
                    align_safety_ms = int(os.environ.get("CHATTERBOX_ALIGN_SAFETY_MS", "0"))
                    align_safety = max(0, int(self.sr * align_safety_ms / 1000))
                    if align_on and wav is not None and wav.numel() > 0:
                        cap = min(wav.shape[1], expected_samples + align_safety)
                        if dbg:
                            print(f"[Align] tokens={n_tokens} expected_samples={expected_samples} safety={align_safety} cap={cap} wav_in={wav.shape[1]}")
                        wav = wav[:, :cap]

                    # Energy-based tail trim (RMS)
                    if tail_trim_on and wav is not None and wav.numel() > 0:
                        sr = self.sr
                        dur_before_ms = (wav.shape[1] * 1000.0) / sr
                        frame_len = max(1, int(sr * rms_window_ms / 1000))
                        hop_len = max(1, int(sr * rms_hop_ms / 1000))
                        if wav.shape[1] >= frame_len:
                            x2 = (wav.unsqueeze(0) ** 2)
                            x2 = x2 if x2.ndim == 3 else x2.view(1, 1, -1)
                            kernel = torch.ones(1, 1, frame_len, device=wav.device, dtype=wav.dtype) / frame_len
                            rms = torch.sqrt(torch.nn.functional.conv1d(x2, kernel, stride=hop_len)).squeeze()
                            peak = float(rms.max().item()) if rms.numel() > 0 else 0.0
                            if peak > 0:
                                thr = peak * (10.0 ** (tail_trim_db_rel / 20.0))
                            else:
                                thr = 10.0 ** (tail_trim_db / 20.0)
                            active = torch.where(rms > thr)[0]
                            if active.numel() > 0:
                                last_active = int(active[-1].item())
                                safety = int(sr * tail_trim_safety_ms / 1000)
                                cut = min(wav.shape[1], (last_active + 1) * hop_len + safety)
                                if cut < wav.shape[1]:
                                    if dbg:
                                        dur_after_ms = (cut * 1000.0) / sr
                                        print(f"[Tail][rms] trim: win={frame_len} hop={hop_len} thr={thr:.6f} "
                                              f"last_active_frame={last_active} safety={safety} "
                                              f"dur_before_ms={dur_before_ms:.1f} -> dur_after_ms={dur_after_ms:.1f} "
                                              f"cut_samples={wav.shape[1]-cut}")
                                    wav = wav[:, :cut]
                        else:
                            if dbg:
                                print(f"[Tail][rms] skip: wav too short for frame_len={frame_len} (len={wav.shape[1]})")

                    # Optional VAD-based end trim (disabled by default)
                    if os.environ.get("CHATTERBOX_VAD_TRIM", "0").lower() in ("1","true","yes","on") and wav is not None and wav.numel() > 0:
                        try:
                            import torchaudio.functional as AF
                            wav16 = (wav.squeeze(0).to(torch.float32).clamp(-1.0, 1.0) * 32767.0).to(torch.int16).cpu()
                            vad_out = AF.vad(wav16, sample_rate=self.sr)
                            if vad_out.numel() > 0 and vad_out.numel() < wav16.numel():
                                new_len = int(vad_out.numel())
                                if dbg:
                                    print(f"[VAD] trim: samples_before={wav16.numel()} -> samples_after={new_len}")
                                wav = wav[:, :new_len].contiguous()
                        except Exception:
                            pass

                    results.append(wav.cpu())

            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Wavform Generation time: {s3gen_gen_time:.2f}s")

            return results
        
    def shutdown(self):
        del self.t3
        torch.cuda.empty_cache()
