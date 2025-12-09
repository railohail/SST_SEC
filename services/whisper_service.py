"""Whisper Speech-to-Text Service using faster-whisper"""
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from config import config


class WhisperService:
    """
    Speech-to-text service using faster-whisper.

    Optimized for Chinese language transcription.
    """

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        language: str = None
    ):
        """
        Initialize Whisper service.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cpu, cuda, mps)
            compute_type: Compute type (int8, float16, float32)
            language: Language code (zh for Chinese)
        """
        self.model_size = model_size or config.WHISPER_MODEL
        self.device = device or config.WHISPER_DEVICE
        self.compute_type = compute_type or config.WHISPER_COMPUTE_TYPE
        self.language = language or config.WHISPER_LANGUAGE

        self.model: Optional[WhisperModel] = None
        self._loaded = False

    def load(self) -> None:
        """Load the Whisper model (lazy loading)."""
        if self._loaded:
            return

        print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        self._loaded = True
        print("Whisper model loaded!")

    def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file (WAV format)

        Returns:
            Transcribed text string
        """
        if not self._loaded:
            self.load()

        # Transcribe with Chinese language
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=5,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            )
        )

        # Concatenate all segments
        text = "".join(segment.text for segment in segments)

        return text.strip()

    def transcribe_with_timestamps(self, audio_path: Path) -> list:
        """
        Transcribe audio with word-level timestamps.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of (start, end, text) tuples
        """
        if not self._loaded:
            self.load()

        segments, _ = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True
        )

        results = []
        for segment in segments:
            for word in segment.words:
                results.append((word.start, word.end, word.word))

        return results
