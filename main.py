#!/usr/bin/env python3
"""
Speech Command App - Main Entry Point

A hands-free speech-to-text application with correction commands.

Usage:
    python main.py [--debug] [--hotkey "<cmd>+<shift>+<space>"]

Workflow:
    1. Press hotkey to start recording
    2. Speak your text in Chinese
    3. Press hotkey again to stop and process
    4. Text is typed at cursor position
    5. To correct: speak a command like "把X改成Y"
"""
# Suppress noisy warnings from transformers about beta/gamma parameter renaming
import warnings
warnings.filterwarnings("ignore", message=".*beta.*renamed.*bias.*")
warnings.filterwarnings("ignore", message=".*gamma.*renamed.*weight.*")

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config, Config
from services.audio_recorder import AudioRecorder
from services.whisper_service import WhisperService
from services.command_processor import CommandProcessor
from services.rule_based_processor import RuleBasedProcessor
from services.gemini_processor import GeminiProcessor
from utils.hotkey_manager import HotkeyManager
from utils.keyboard_simulator import KeyboardSimulator


class SpeechCommandApp:
    """
    Main application class for speech-to-text with command correction.
    """

    def __init__(self, debug_mode: bool = None, hotkey: str = None, stt_backend: str = None, no_ml: bool = False, use_api: bool = False):
        """
        Initialize the application.

        Args:
            debug_mode: Enable debug overlay (default: from config)
            hotkey: Custom hotkey (default: from config)
            stt_backend: STT backend ("whisper" or "funasr", default: from config)
            no_ml: Use rule-based processor instead of BERT model (default: False)
            use_api: Use Gemini API for processing (default: False)
        """
        self.debug_mode = debug_mode if debug_mode is not None else config.DEBUG_MODE
        self.hotkey = hotkey or config.HOTKEY
        self.stt_backend = stt_backend or config.STT_BACKEND
        self.no_ml = no_ml
        self.use_api = use_api

        # State
        self.is_recording = False
        self.last_typed_text = ""
        self._lock = threading.Lock()

        # Services (lazy loaded)
        self._recorder = None
        self._stt = None
        self._processor = None
        self._hotkey_manager = None
        self._keyboard = None

    @property
    def recorder(self) -> AudioRecorder:
        if self._recorder is None:
            self._recorder = AudioRecorder()
        return self._recorder

    @property
    def stt(self):
        """Get the STT service (Whisper or FunASR)."""
        if self._stt is None:
            if self.stt_backend == "funasr":
                from services.funasr_service import FunASRService
                self._stt = FunASRService(model_name=config.FUNASR_MODEL)
            else:
                self._stt = WhisperService()
        return self._stt

    # Keep whisper property for backward compatibility
    @property
    def whisper(self):
        return self.stt

    @property
    def processor(self):
        """Get the command processor (API, rule-based, or ML-based)."""
        if self._processor is None:
            if self.use_api:
                self._processor = GeminiProcessor()
            elif self.no_ml:
                self._processor = RuleBasedProcessor()
            else:
                self._processor = CommandProcessor()
        return self._processor

    @property
    def keyboard(self) -> KeyboardSimulator:
        if self._keyboard is None:
            self._keyboard = KeyboardSimulator()
        return self._keyboard

    def _show_status(self, message: str) -> None:
        """Show status message (debug mode only)."""
        if self.debug_mode:
            print(f"[STATUS] {message}")

    def _hide_status(self) -> None:
        """Hide status (no-op for console mode)."""
        pass

    def on_hotkey(self) -> None:
        """Handle hotkey press - toggle recording."""
        with self._lock:
            if not self.is_recording:
                self._start_recording()
            else:
                self._stop_and_process()

    def _start_recording(self) -> None:
        """Start audio recording."""
        self.is_recording = True
        self._show_status("Recording...")
        self.recorder.start()

    def _stop_and_process(self) -> None:
        """Stop recording and process the audio."""
        self.is_recording = False
        self._show_status("Processing...")

        # Stop recording
        audio_path = self.recorder.stop()

        try:
            # Transcribe audio
            self._show_status("Transcribing...")
            text = self.whisper.transcribe(audio_path)

            if not text:
                self._show_status("No speech detected")
                time.sleep(1)
                self._hide_status()
                return

            self._show_status(f"Heard: {text}")

            # Check if this is a correction command
            if self.processor.is_command(text):
                # First, check if user has selected text to correct
                selected_text = self.keyboard.get_selected_text()
                target_text = selected_text if selected_text else self.last_typed_text
                has_selection = bool(selected_text)

                self._show_status(f"Applying correction to: '{target_text}' (selected: {has_selection})")
                result, was_command = self.processor.process(text, target_text)
                self._show_status(f"Result: '{result}', was_command: {was_command}")

                if was_command and target_text:
                    if has_selection:
                        # Replace selected text directly (typing replaces selection)
                        self.keyboard.replace_selection(result)
                    else:
                        # Delete old text and type corrected version
                        self.keyboard.replace_last_typed(target_text, result)
                    self.last_typed_text = result
                    self._show_status(f"Corrected: {result}")
                else:
                    # Couldn't apply correction, type as-is
                    self.keyboard.type_text(text)
                    self.last_typed_text = text
            else:
                # Normal dictation - just type the text
                self._show_status(f"Typing: {text}")
                try:
                    self.keyboard.type_text(text)
                    self.last_typed_text = text
                    self._show_status(f"Typed: {text}")
                except Exception as e:
                    self._show_status(f"Typing failed: {e}")
                    print(f"Typing error: {e}")

            time.sleep(0.5)
            self._hide_status()

        except Exception as e:
            self._show_status(f"Error: {e}")
            print(f"Error processing speech: {e}")
            time.sleep(2)
            self._hide_status()

    def _preload_models(self) -> None:
        """Preload models for faster first inference."""
        print("Preloading models...")
        self.whisper.load()
        self.processor.labeler.load()
        print("Models loaded!")

    def run(self) -> None:
        """Run the application."""
        print("=" * 60)
        print("Speech Command App")
        print("=" * 60)
        print(f"Debug mode: {self.debug_mode}")
        print(f"Hotkey: {self.hotkey}")
        print(f"STT backend: {self.stt_backend}")
        processor_name = 'Gemini API' if self.use_api else ('Rule-based (no ML)' if self.no_ml else 'BERT+CRF (ML)')
        print(f"Processor: {processor_name}")
        print(f"Accessibility: {'enabled' if self.keyboard.has_accessibility() else 'disabled'}")
        print()

        # Preload models
        self._preload_models()

        # Debug overlay disabled (tkinter has macOS threading issues)
        # Status shown in console instead

        # Setup hotkey listener
        self._hotkey_manager = HotkeyManager(
            hotkey=self.hotkey,
            callback=self.on_hotkey
        )
        self._hotkey_manager.start()

        print()
        print("Ready! Press the hotkey to start/stop recording.")
        print("Press Ctrl+C to exit.")
        print("=" * 60)

        # Wait for exit
        try:
            self._hotkey_manager.wait()
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._hotkey_manager:
            self._hotkey_manager.stop()
        if self._recorder:
            self._recorder.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Speech-to-text with correction commands"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with status overlay"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode"
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default=None,
        help="Custom hotkey (e.g., '<cmd>+<shift>+<space>')"
    )
    parser.add_argument(
        "--stt",
        type=str,
        choices=["whisper", "funasr"],
        default=None,
        help="STT backend: 'whisper' (default) or 'funasr' (Paraformer)"
    )
    parser.add_argument(
        "--notML",
        action="store_true",
        help="Use rule-based processor instead of BERT model (faster, no GPU needed)"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Use Gemini API for intelligent correction (requires internet)"
    )

    args = parser.parse_args()

    # Determine debug mode
    debug_mode = None
    if args.debug:
        debug_mode = True
    elif args.no_debug:
        debug_mode = False

    # Create and run app
    app = SpeechCommandApp(
        debug_mode=debug_mode,
        hotkey=args.hotkey,
        stt_backend=args.stt,
        no_ml=args.notML,
        use_api=args.api
    )

    # Handle signals
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, shutting down...")
        app.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.run()


if __name__ == "__main__":
    main()
