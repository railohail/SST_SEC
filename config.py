"""Configuration for Speech Command App"""
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Mode
    DEBUG_MODE: bool = True  # Show status overlay, False for production

    # Hotkey (pynput format)
    HOTKEY: str = "<cmd>+<shift>+<space>"  # Toggle recording

    # STT Backend: "whisper" or "funasr"
    STT_BACKEND: str = "whisper"

    # FunASR settings (if STT_BACKEND="funasr")
    # FUNASR_MODEL: str = "sensevoice"  # "sensevoice" (recommended), "paraformer"
    FUNASR_MODEL: str = "paraformer-zh"
    # Whisper settings (if STT_BACKEND="whisper")
    WHISPER_MODEL: str = "medium"
    WHISPER_LANGUAGE: str = "zh"
    WHISPER_DEVICE: str = "cpu"  # or "cuda" or "mps"
    WHISPER_COMPUTE_TYPE: str = "int8"  # int8 for faster inference

    # Model paths
    PROJECT_ROOT: Path = Path(__file__).parent
    MODEL_WEIGHTS_DIR: Path = PROJECT_ROOT / "models" / "model_weights"
    CRF_MODEL_PATH: Path = MODEL_WEIGHTS_DIR / "model_crf_enhanced.pt"

    # Audio settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024
    AUDIO_FORMAT: str = "wav"

    # Temp file for audio (cross-platform)
    TEMP_AUDIO_PATH: Path = Path(tempfile.gettempdir()) / "speech_command_recording.wav"

    # BERT model name (same as training)
    BERT_MODEL_NAME: str = "google-bert/bert-base-multilingual-cased"
    NUM_LABELS: int = 3

    # Label mappings
    LABEL_MAP: dict = None
    ID_TO_LABEL: dict = None

    def __post_init__(self):
        self.LABEL_MAP = {
            'O': 0,
            'B-Modify': 1,
            'B-Filling': 2
        }
        self.ID_TO_LABEL = {v: k for k, v in self.LABEL_MAP.items()}


# Global config instance
config = Config()
