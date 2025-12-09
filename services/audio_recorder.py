"""Audio Recording Service using PyAudio"""
import wave
import threading
from pathlib import Path
from typing import Optional

import pyaudio

from config import config


class AudioRecorder:
    """
    Audio recorder with toggle mode support.

    Usage:
        recorder = AudioRecorder()
        recorder.start()  # Start recording
        # ... wait for user to finish speaking ...
        audio_path = recorder.stop()  # Stop and save to file
    """

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        chunk_size: int = None,
        output_path: Path = None
    ):
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate (default: from config)
            channels: Number of audio channels (default: from config)
            chunk_size: Chunk size for recording (default: from config)
            output_path: Path to save recording (default: from config)
        """
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.channels = channels or config.CHANNELS
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.output_path = output_path or config.TEMP_AUDIO_PATH

        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.frames: list = []
        self.is_recording: bool = False
        self._record_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start recording audio in a background thread."""
        if self.is_recording:
            return

        self.frames = []
        self.is_recording = True

        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        # Start recording thread
        self._record_thread = threading.Thread(target=self._record_loop)
        self._record_thread.start()

    def _record_loop(self) -> None:
        """Internal recording loop running in background thread."""
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Recording error: {e}")
                break

    def stop(self) -> Path:
        """
        Stop recording and save to file.

        Returns:
            Path to the saved audio file
        """
        if not self.is_recording:
            return self.output_path

        self.is_recording = False

        # Wait for recording thread to finish
        if self._record_thread:
            self._record_thread.join(timeout=1.0)

        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Save to WAV file
        self._save_wav()

        return self.output_path

    def _save_wav(self) -> None:
        """Save recorded frames to WAV file."""
        if not self.frames:
            return

        with wave.open(str(self.output_path), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))

    def cleanup(self) -> None:
        """Clean up PyAudio resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
