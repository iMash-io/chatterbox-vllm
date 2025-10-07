import logging
import json
import os
from typing import List, Optional, Union
from pathlib import Path
from unicodedata import category, normalize

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

# Model repository
REPO_ID = "ResembleAI/chatterbox"

# Global instances for optional dependencies
_kakasi = None
_dicta = None
_russian_stresser = None


def is_kanji(c: str) -> bool:
    """Check if character is kanji."""
    return 19968 <= ord(c) <= 40959


def is_katakana(c: str) -> bool:
    """Check if character is katakana."""
    return 12449 <= ord(c) <= 12538


def hiragana_normalize(text: str) -> str:
    """Japanese text normalization: converts kanji to hiragana; katakana remains the same."""
    global _kakasi
    
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
        
        result = _kakasi.convert(text)
        out = []
        
        for r in result:
            inp = r['orig']
            hira = r["hira"]

            # Any kanji in the phrase
            if any([is_kanji(c) for c in inp]):
                if hira and hira[0] in ["は", "へ"]:  # Safety check for empty hira
                    hira = " " + hira
                out.append(hira)

            # All katakana
            elif all([is_katakana(c) for c in inp]) if inp else False:  # Safety check for empty inp
                out.append(r['orig'])

            else:
                out.append(inp)
        
        normalized_text = "".join(out)
        
        # Decompose Japanese characters for tokenizer compatibility
        import unicodedata
        normalized_text = unicodedata.normalize('NFKD', normalized_text)
        
        return normalized_text
        
    except ImportError:
        logger.warning("pykakasi not available - Japanese text processing skipped")
        return text


def add_hebrew_diacritics(text: str) -> str:
    """Hebrew text normalization: adds diacritics to Hebrew text."""
    global _dicta
    
    try:
        if _dicta is None:
            from dicta_onnx import Dicta
            _dicta = Dicta()
        
        return _dicta.add_diacritics(text)
        
    except ImportError:
        logger.warning("dicta_onnx not available - Hebrew text processing skipped")
        return text
    except Exception as e:
        logger.warning(f"Hebrew diacritization failed: {e}")
        return text


def korean_normalize(text: str) -> str:
    """Korean text normalization: decompose syllables into Jamo for tokenization."""
    
    def decompose_hangul(char):
        """Decompose Korean syllable into Jamo components."""
        if not ('\uac00' <= char <= '\ud7af'):
            return char
        
        # Hangul decomposition formula
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        
        return initial + medial + final
    
    # Decompose syllables and normalize punctuation
    result = ''.join(decompose_hangul(char) for char in text)    
    return result.strip()


class ChineseCangjieConverter:
    """Converts Chinese characters to Cangjie codes for tokenization."""
    
    def __init__(self, model_dir=None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_cangjie_mapping(model_dir)
        self._init_segmenter()
    
    def _load_cangjie_mapping(self, model_dir=None):
        """Load Cangjie mapping from HuggingFace model repository."""        
        try:
            cangjie_file = hf_hub_download(
                repo_id=REPO_ID,
                filename="Cangjie5_TC.json",
                cache_dir=model_dir
            )
            
            with open(cangjie_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            for entry in data:
                word, code = entry.split("\t")[:2]
                self.word2cj[word] = code
                if code not in self.cj2word:
                    self.cj2word[code] = [word]
                else:
                    self.cj2word[code].append(word)
                    
        except Exception as e:
            logger.warning(f"Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        """Initialize pkuseg segmenter."""
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            logger.warning("pkuseg not available - Chinese segmentation will be skipped")
            self.segmenter = None
    
    def _cangjie_encode(self, glyph: str):
        """Encode a single Chinese glyph to Cangjie code."""
        normed_glyph = glyph
        code = self.word2cj.get(normed_glyph, None)
        if code is None:  # e.g. Japanese hiragana
            return None
        index = self.cj2word[code].index(normed_glyph)
        index = str(index) if index > 0 else ""
        return code + str(index)
    
    def __call__(self, text):
        """Convert Chinese characters in text to Cangjie tokens."""
        output = []
        if self.segmenter is not None:
            segmented_words = self.segmenter.cut(text)
            full_text = " ".join(segmented_words)
        else:
            full_text = text
        
        for t in full_text:
            if category(t) == "Lo":
                cangjie = self._cangjie_encode(t)
                if cangjie is None:
                    output.append(t)
                    continue
                code = []
                for c in cangjie:
                    code.append(f"[cj_{c}]")
                code.append("[cj_.]")
                code = "".join(code)
                output.append(code)
            else:
                output.append(t)
        return "".join(output)


def add_russian_stress(text: str) -> str:
    """Russian text normalization: adds stress marks to Russian text."""
    global _russian_stresser
    
    try:
        if _russian_stresser is None:
            from russian_text_stresser.text_stresser import RussianTextStresser
            _russian_stresser = RussianTextStresser()
        
        return _russian_stresser.stress_text(text)
        
    except ImportError:
        logger.warning("russian_text_stresser not available - Russian stress labeling skipped")
        return text
    except Exception as e:
        logger.warning(f"Russian stress labeling failed: {e}")
        return text


class MTLTokenizer(PreTrainedTokenizer):
    """
    A vLLM-compatible multilingual tokenizer that wraps the original Tokenizer implementation.
    Supports 23 languages with language-specific text processing.
    """
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: str,
        unk_token: str = UNK,
        pad_token: str = "[PAD]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs
    ):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file)
        model_dir = Path(vocab_file).parent
        self.cangjie_converter = ChineseCangjieConverter(model_dir)
        
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        self.check_vocabset_sot_eot()

    @classmethod
    def from_pretrained(cls, **kwargs):
        """
        Instantiate a tokenizer from a pretrained model or path.
        
        Args:
            pretrained_model_name_or_path: Path to the tokenizer file or model name
            **kwargs: Additional arguments to pass to the tokenizer
        """
        # Load relative to the current file path
        vocab_file = os.path.join(os.path.dirname(__file__), "mtl_tokenizer.json")
        return cls(vocab_file=vocab_file, **kwargs)

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def get_vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def preprocess_text(self, raw_text: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        """
        Text preprocessor that handles lowercase conversion and NFKD normalization.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        
        return preprocessed_text

    def apply_language_processing(self, txt: str, language_id: str = None):
        """
        Apply language-specific text processing.
        
        Args:
            txt: Input text
            language_id: Language code (e.g., 'zh', 'ja', 'he', 'ko', 'ru')
        
        Returns:
            Processed text with language-specific transformations applied
        """
        if language_id == 'zh':
            txt = self.cangjie_converter(txt)
        elif language_id == 'ja':
            txt = hiragana_normalize(txt)
        elif language_id == 'he':
            txt = add_hebrew_diacritics(txt)
        elif language_id == 'ko':
            txt = korean_normalize(txt)
        elif language_id == 'ru':
            txt = add_russian_stress(txt)
        
        # Prepend language token
        if language_id:
            txt = f"[{language_id.lower()}]{txt}"
        
        return txt

    def _tokenize(self, text: str, language_id: str = None, **kwargs) -> List[str]:
        """
        Tokenize text with optional language-specific processing.
        
        Args:
            text: Input text to tokenize
            language_id: Optional language code for language-specific processing
        
        Returns:
            List of token strings
        """
        # Apply preprocessing
        text = self.preprocess_text(text, language_id=language_id)
        
        # Apply language-specific processing
        text = self.apply_language_processing(text, language_id=language_id)
        
        # Replace spaces with SPACE token
        text = text.replace(' ', SPACE)
        
        return self.tokenizer.encode(text).tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(' ', '')
        text = text.replace(SPACE, ' ')
        text = text.replace(EOT, '')
        text = text.replace(UNK, '')
        return text

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save the tokenizer to a directory.
        
        Args:
            save_directory: Directory to save the tokenizer to
            **kwargs: Additional arguments to pass to the tokenizer
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
            
        self.tokenizer.save(os.path.join(save_directory, "mtl_tokenizer.json"))

    def text_to_tokens(self, text: str, language_id: str = None, lowercase: bool = True, nfkd_normalize: bool = True):
        """Legacy method for backward compatibility"""
        text_tokens = self.encode(
            text, 
            language_id=language_id, 
            lowercase=lowercase, 
            nfkd_normalize=nfkd_normalize
        )
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(
        self, 
        txt: str, 
        language_id: str = None, 
        lowercase: bool = True, 
        nfkd_normalize: bool = True,
        verbose: bool = False,
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True
    ):
        """
        Encode text to token IDs with language-specific processing.
        
        Args:
            txt: Input text
            language_id: Optional language code (e.g., 'en', 'zh', 'ja')
            lowercase: Whether to lowercase the text
            nfkd_normalize: Whether to apply NFKD normalization
            verbose: Whether to print verbose output
            return_tensors: If "pt", return PyTorch tensor
            add_special_tokens: Whether to add special tokens (unused, for compatibility)
        
        Returns:
            List of token IDs or PyTorch tensor if return_tensors="pt"
        """
        # Preprocess
        txt = self.preprocess_text(txt, language_id=language_id, lowercase=lowercase, nfkd_normalize=nfkd_normalize)
        
        # Apply language-specific processing
        txt = self.apply_language_processing(txt, language_id=language_id)
        
        # Replace spaces
        txt = txt.replace(' ', SPACE)
        
        # Encode
        ids = self.tokenizer.encode(txt).ids
        
        if return_tensors == "pt":
            return torch.IntTensor(ids).unsqueeze(0)
        return ids

    def decode(self, seq, skip_special_tokens: bool = False):
        """
        Decode token IDs back to text.
        
        Args:
            seq: Token IDs (list, numpy array, or torch tensor)
            skip_special_tokens: Whether to skip special tokens (partial support)
        
        Returns:
            Decoded text string
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        return txt

    @property
    def max_token_id(self) -> int:
        return max(self.tokenizer.get_vocab().values())

    @property
    def vocab_size(self) -> int:
        """Property for compatibility with vLLM"""
        return self.get_vocab_size()