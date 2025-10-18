# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import os


logger = logging.getLogger(__name__)

# Candidate attention heads that tend to show strong text-speech alignment
LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    """
    Online alignment-driven EOS controller adapted for vLLM path.

    - Hooks a few Llama attention heads to extract per-step attention maps
    - Tracks approximate alignment position along the text tokens
    - Suppresses EOS early; forces EOS when long-tail/repetition is detected
    """

    def __init__(
        self,
        tfmr,
        text_tokens_slice: Tuple[int, int],
        alignment_layer_idx: int = 9,
        eos_idx: int = 0,
        text_embeds: Optional[torch.Tensor] = None,
    ):
        """
        tfmr: vLLM LlamaModel instance (self.tfmr from T3VllmModel)
        text_tokens_slice: (i, j) indices in the context where text tokens lie
        alignment_layer_idx: which Llama layer index to interrogate; we register a small set of heads
        eos_idx: absolute EOS token index in the logits vector
        """
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, max(j - i, 1))
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at: Optional[int] = None

        self.complete = False
        self.completed_at: Optional[int] = None

        # Track generated tokens for repetition detection
        self.generated_tokens: list[int] = []

        # Optional fallback alignment via cosine similarity to prefill text embeddings
        self.text_embeds_norm: Optional[torch.Tensor] = None
        if text_embeds is not None:
            try:
                # Normalize text embeddings rows for cosine similarity
                ten = text_embeds.detach().float()
                self.text_embeds_norm = F.normalize(ten, dim=-1).cpu()
            except Exception:
                self.text_embeds_norm = None

        # Fallback tail counter when using similarity-based alignment
        self.tail_frames: int = 0

        # Forward hooks to capture attentions for a few heads
        self.last_aligned_attns = []
        for k, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns += [None]
            self._add_attention_spy(tfmr, k, layer_idx, head_idx)
        if os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on"):
            try:
                print(f"[Align][init] slice={self.text_tokens_slice} eos_idx={self.eos_idx} hooks={len(LLAMA_ALIGNED_HEADS)} fallback={'on' if self.text_embeds_norm is not None else 'off'}")
            except Exception:
                pass

    def _add_attention_spy(self, tfmr, buffer_idx: int, layer_idx: int, head_idx: int):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        """

        def attention_forward_hook(module, _input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            When `output_attentions=True`, `attn_weights` has shape [B, H, Tq, Tk].
            """
            try:
                if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                    step_attention = output[1].detach().cpu()  # (B, n_heads, Tq, Tk)
                    # Take head slice
                    self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]  # (Tq, Tk)
            except Exception as _e:
                # Best-effort capture; never crash the engine
                logger.debug(f"Alignment hook failed: {str(_e)}")

        try:
            target_layer = tfmr.layers[layer_idx].self_attn
            target_layer.register_forward_hook(attention_forward_hook)
            # Ask HF config to emit attentions if available; vLLM may respect or ignore this
            if hasattr(tfmr, "config") and hasattr(tfmr.config, "output_attentions"):
                self.original_output_attentions = tfmr.config.output_attentions
                tfmr.config.output_attentions = True
        except Exception as _e:
            logger.warning(f"Failed to register attention hook for alignment: {str(_e)}")

    def step(self, logits: torch.Tensor, hidden_state: Optional[torch.Tensor] = None, next_token: Optional[int] = None) -> torch.Tensor:
        """
        Potentially modifies logits in-place:
        - Suppress EOS during early alignment (prevents early cut-offs)
        - Force EOS on long tails or alignment/token repetition (prevents rambling)

        logits: (..., V) final logits for current step (post any CFG combination and offset padding)
        next_token: last sampled token id (optional); improves repetition detection if available
        """
        dbg = (torch.jit.is_scripting() is False) and (torch.jit.is_tracing() is False) and (
            (os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1", "true", "yes", "on")) if "os" in globals() else False  # type: ignore
        )

        try:
            # Gather averaged attention for the selected heads
            aligned_avail = [a for a in self.last_aligned_attns if a is not None]
            if (os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on")):
                try:
                    print(f"[Align][step] frame={self.curr_frame_pos} avail_heads={len(aligned_avail)}")
                except Exception:
                    pass
            if not aligned_avail:
                # Fallback: estimate alignment via cosine similarity to prefill text embeddings if available
                if (self.text_embeds_norm is not None) and (hidden_state is not None):
                    try:
                        h = hidden_state
                        if isinstance(h, torch.Tensor) and h.ndim > 1:
                            h = h[-1]
                        h = F.normalize(h.detach().float(), dim=-1).cpu()
                        sims = torch.matmul(self.text_embeds_norm, h.unsqueeze(-1)).squeeze(-1)  # (S,)
                        cur_text_posn = int(torch.argmax(sims).item())
                        S = int(self.text_embeds_norm.shape[0])
                        if (os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on")):
                            try:
                                print(f"[Align][fallback] pos={cur_text_posn}/{S-1} started={self.started} complete={self.complete} tail_frames={self.tail_frames}")
                            except Exception:
                                pass
                        # Update position/complete tracking
                        discontinuity = not (-4 < cur_text_posn - self.text_position < 7)
                        if not discontinuity:
                            self.text_position = cur_text_posn
                        self.complete = self.complete or (self.text_position >= S - 3 if S > 0 else False)
                        if self.complete:
                            self.tail_frames += 1
                        # Early EOS suppression while far from end
                        if (cur_text_posn < (S - 3)) and (S > 5):
                            logits[..., self.eos_idx] = -2**15
                        # Simple long-tail cutoff if we've been complete for enough frames
                        if self.complete and self.tail_frames >= 12:
                            logits[..., :] = -(2**15)
                            logits[..., self.eos_idx] = 2**15
                    except Exception:
                        # Minimal early-EOS safety for the very first frames
                        if self.curr_frame_pos < 2:
                            logits[..., self.eos_idx] = -2**15
                else:
                    # Minimal early-EOS safety for the very first frames
                    if self.curr_frame_pos < 2:
                        logits[..., self.eos_idx] = -2**15
                self.curr_frame_pos += 1
                return logits

            aligned_attn = torch.stack(aligned_avail).mean(dim=0)  # (Tq, Tk)

            i, j = self.text_tokens_slice
            # First chunk has prefill with long context; subsequent steps often have Tq=1
            if self.curr_frame_pos == 0 and aligned_attn.shape[0] > 1:
                # slice [text] columns from the decode rows
                A_chunk = aligned_attn[-aligned_attn.shape[0]:, i:j].clone().cpu()  # (Tq, S)
            else:
                A_chunk = aligned_attn[:, i:j].clone().cpu()  # (Tq or 1, S)

            # Monotonic masking wrt progress along time (lenient)
            if A_chunk.numel() > 0:
                A_chunk[:, self.curr_frame_pos + 1 :] = 0

            # Append into running alignment
            if A_chunk.numel() > 0:
                # Ensure alignment has matching S dimension
                if self.alignment.shape[1] != A_chunk.shape[1]:
                    self.alignment = torch.zeros(0, A_chunk.shape[1])
                self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

            A = self.alignment
            T, S = A.shape if A.numel() > 0 else (0, j - i)

            # Update alignment-based position
            if A.numel() > 0:
                cur_text_posn = int(A_chunk[-1].argmax().item())
                discontinuity = not (-4 < cur_text_posn - self.text_position < 7)  # lenient
                if (os.environ.get("CHATTERBOX_DEBUG", "0").lower() in ("1","true","yes","on")):
                    try:
                        print(f"[Align][attn] pos={cur_text_posn}/{S-1} started={self.started} complete={self.complete}")
                    except Exception:
                        pass
                if not discontinuity:
                    self.text_position = cur_text_posn
            else:
                cur_text_posn = 0
                discontinuity = False

            # False start detection: avoid noisy beginnings that can trigger EOS
            false_start = (not self.started) and (
                (A[-2:, -2:].max() > 0.1 if A.shape[0] >= 2 else False) or (A[:, :4].max() < 0.5 if A.shape[1] >= 4 else True)
            )
            self.started = not false_start
            if self.started and self.started_at is None:
                self.started_at = T

            # Completion heuristic: close to end of text
            self.complete = self.complete or (self.text_position >= S - 3 if S > 0 else False)
            if self.complete and self.completed_at is None:
                self.completed_at = T

            # Track last sampled token for repetition detection (optional)
            if next_token is not None:
                try:
                    if isinstance(next_token, torch.Tensor):
                        token_id = next_token.item() if next_token.numel() == 1 else int(next_token.view(-1)[0].item())
                    else:
                        token_id = int(next_token)
                    self.generated_tokens.append(token_id)
                    if len(self.generated_tokens) > 8:
                        self.generated_tokens = self.generated_tokens[-8:]
                except Exception:
                    pass

            # Excessive token repetition detection (2x in a row)
            token_repetition = len(self.generated_tokens) >= 3 and len(set(self.generated_tokens[-2:])) == 1

            # Long-tail / alignment repetition after completion
            if self.complete and A.numel() > 0:
                long_tail = (A[self.completed_at:, -3:].sum(dim=0).max() >= 5) if self.completed_at is not None else False
                alignment_repetition = (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5) if (A.shape[1] > 5 and self.completed_at is not None) else False
            else:
                long_tail = False
                alignment_repetition = False

            # Early EOS suppression: only allow EOS when sufficiently advanced in the text
            if (cur_text_posn < (S - 3)) and (S > 5):
                logits[..., self.eos_idx] = -2**15

            # Bad endings: force EOS
            if long_tail or alignment_repetition or token_repetition:
                logger.warning(f"forcing EOS via alignment: long_tail={long_tail}, align_rep={alignment_repetition}, tok_rep={token_repetition}")
                logits[..., :] = -(2**15)
                logits[..., self.eos_idx] = 2**15

        except Exception as _e:
            # Never break decoding; fail open with no modification
            logger.debug(f"AlignmentStreamAnalyzer.step error: {str(_e)}")

        self.curr_frame_pos += 1
        return logits
