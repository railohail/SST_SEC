"""FunASR Service for Chinese speech-to-text"""
import re
from pathlib import Path
from typing import Optional

from config import config

# OpenCC for simplified to traditional Chinese conversion
try:
    from opencc import OpenCC
    _opencc_converter = OpenCC('s2t')  # Simplified to Traditional
except ImportError:
    _opencc_converter = None


class FunASRService:
    """
    Chinese speech-to-text using Alibaba's FunASR.

    Models available:
    - paraformer-small: Small, fast Chinese ASR (~70MB)
    - paraformer: Standard Chinese ASR (~220MB)
    - sensevoice: Multi-language with emotion detection (~460MB)
    """

    # Available models - use full modelscope paths
    MODELS = {
        "paraformer-small": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "paraformer": "iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1",
        "sensevoice": "iic/SenseVoiceSmall",
    }

    def __init__(self, model_name: str = "sensevoice"):
        """
        Initialize FunASR service.

        Args:
            model_name: One of "paraformer-small", "paraformer", "sensevoice"
        """
        self.model_name = model_name
        self.model_id = self.MODELS.get(model_name, self.MODELS["sensevoice"])
        self.model = None
        self._loaded = False

    def load(self) -> None:
        """Load the ASR model."""
        if self._loaded:
            return

        print(f"Loading FunASR model: {self.model_name}...")

        from funasr import AutoModel

        # Use SenseVoice which is more stable
        if self.model_name == "sensevoice":
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cpu",
                disable_update=True,
            )
        else:
            self.model = AutoModel(
                model=self.model_id,
                device="cpu",
                disable_update=True,
            )

        self._loaded = True
        print(f"FunASR model loaded!")

    def _clean_text(self, text: str) -> str:
        """
        Clean the transcribed text by removing SenseVoice tags and
        converting simplified Chinese to traditional Chinese.

        Args:
            text: Raw transcribed text with potential tags

        Returns:
            Cleaned text in traditional Chinese
        """
        # Remove SenseVoice tags like <|zh|><|NEUTRAL|><|Speech|><|woitn|>
        cleaned = re.sub(r'<\|[^|>]+\|>', '', text)

        # Convert simplified to traditional Chinese if opencc is available
        if _opencc_converter:
            cleaned = _opencc_converter.convert(cleaned)

        return cleaned.strip()

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file (WAV format)

        Returns:
            Transcribed text
        """
        if not self._loaded:
            self.load()

        # Run inference
        result = self.model.generate(
            input=str(audio_path),
            batch_size_s=300,  # Process up to 300 seconds
        )

        # Extract text from result
        if result and len(result) > 0:
            # Result format varies by model
            if isinstance(result[0], dict):
                raw_text = result[0].get("text", "")
            elif hasattr(result[0], "text"):
                raw_text = result[0].text
            else:
                raw_text = str(result[0])

            # Clean tags and convert to traditional Chinese
            return self._clean_text(raw_text)

        return ""
