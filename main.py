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
from PIL import Image
import pystray
from typing import Optional, Any

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


        #icon images
        self.tray_icon: Optional[pystray.Icon] = None
        self.ready_icon_img: Optional[Image.Image] = None # 靜態（白）圖標
        self.recording_icon_img: Optional[Image.Image] = None # 錄製中（紅）圖標
        self.placeholder_icon_img: Optional[Image.Image] = None # 備用圖

        # icon blink state
        self._blink_timer = None
        self._blink_state = False

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
        """Show status message and update tray icon title."""
        print(f"[STATUS] {message}")
        # [最小化修改：新增 Pystray 狀態更新]
        if self.tray_icon:
            self.tray_icon.title = f"Speech App - {message}"

    def _hide_status(self) -> None:
        """Hide status (update tray title to 'Ready')."""
        # [最小化修改：新增 Pystray 狀態更新]
        self._show_status("Ready")

    '''    
    def on_hotkey(self) -> None:
        """Handle hotkey press - toggle recording."""
        with self._lock:
            if not self.is_recording:
                self._start_recording()
            else:
                self._stop_and_process()
    '''
    def get_toggle_text(self, icon: Any) -> str:
        """Dynamically set the menu text based on recording state."""
        if self.is_recording:
            return 'Stop Recording'
        # 保留 hotkey 提示
        return f'Start Recording ({self.hotkey})'

    def _load_icon_safe(self, filename: str) -> Optional[Image.Image]:
        icon_path = Path(__file__).parent / filename
        try:
            img = Image.open(icon_path)
            print(f"Icon loaded successfully from: {icon_path}")
            return img
        except FileNotFoundError:
            print(f"⚠️ Warning: Icon file not found at {icon_path}.")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load icon image '{filename}': {e.__class__.__name__}.")
        return None


    def setup_tray(self) -> None:
        # 1. 載入靜態（Ready）和錄製中（Recording）圖標
        self.ready_icon_img = self._load_icon_safe("speech-synthesis.png")
        self.recording_icon_img = self._load_icon_safe("speech-synthesis_red.png")
        
        # 2. 建立備用圖標 (如果載入失敗)
        if not self.ready_icon_img:
            self.placeholder_icon_img = Image.new('RGB', (16, 16), (255, 255, 255))
            self.ready_icon_img = self.placeholder_icon_img
        if not self.recording_icon_img:
            # 如果紅色圖標遺失，退回到一個實心紅色方塊
            self.recording_icon_img = Image.new('RGB', (16, 16), (255, 0, 0))
            
        # 3. 創建菜單 (保持不變)
        menu = pystray.Menu(
            pystray.MenuItem(text=self.get_toggle_text, action=self.on_hotkey),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(text='❌ Exit', action=self.cleanup_and_exit)
        )
        
        # 4. 創建 Pystray 實例，使用 Ready Icon 作為初始圖標
        self.tray_icon = pystray.Icon(
            name="SpeechCommandApp", 
            icon=self.ready_icon_img, # 使用載入的白色圖標作為初始狀態
            title="Speech App - Initializing...",
            menu=menu
        )
        self.tray_icon.title = "Speech App - Ready"

    def cleanup_and_exit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        """停止圖標並退出程式 (用於菜單退出)。"""
        print("\nExiting via System Tray Menu...")
        self.cleanup()
        if icon:
            icon.stop()
        sys.exit(0)

    #new version 
    def on_hotkey(self, icon: Optional[pystray.Icon]=None, item: Optional[pystray.MenuItem]=None) -> None:
        """Handle hotkey press - toggle recording (Fast Toggle)."""
        
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_and_process() 

    def _start_recording(self) -> None:
        """Start audio recording."""
        self.is_recording = True
        self._show_status("Recording...")
        self.recorder.start()
    '''
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
    '''
    def _stop_and_process(self) -> None:
        """Stop recording, set state, and launch core processing thread."""
        
        # 1. 立即設定狀態為非錄音，釋放給下一個 F9
        self.is_recording = False
        self._show_status("Processing...")
        
        if self._blink_timer:
            self._blink_timer.cancel()
            self._blink_timer = None
        self._set_static_icon()
        # 2. 將核心邏輯丟給新的執行緒處理
        process_thread = threading.Thread(target=self._stop_and_process_core)
        process_thread.start()

    def _stop_and_process_core(self) -> None:
        """Core logic for stopping recording and processing audio (runs in thread)."""
        
        with self._lock: 
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
        print(f"🎙️ Speech Command App starting...")
        
        # Preload models
        self._preload_models()

        # [新增 Pystray 啟動]
        self.setup_tray()
        # [結束 Pystray 啟動]

        # 啟動 Hotkey Manager
        self._hotkey_manager = HotkeyManager(hotkey=self.hotkey, callback=self.on_hotkey)
        self._hotkey_manager.start()

        print(f"Hotkey Registered: {self.hotkey}")
        # 顯示初始狀態
        self._show_status("Ready")

        # [修改主迴圈]
        # 兼容 Mac/Win：讓 Pystray 在背景執行緒中運行 (run_detached)
        self.tray_icon.run_detached()

        print("System Tray icon running. Use 'Exit' in the menu to close.")
        print("=" * 60)
        
        try:
            # Pystray 在背景執行緒中，我們手動讓主線程保持活躍
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived Ctrl+C, shutting down...")
        finally:
            self.cleanup()
            if self.tray_icon:
                self.tray_icon.stop() # 確保停止 Pystray
            sys.exit(0)

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._hotkey_manager:
            self._hotkey_manager.stop()
            
        if self._recorder:
            self._recorder.cleanup()

        if self._blink_timer:
            self._blink_timer.cancel()
            self._blink_timer = None
        
        if self.tray_icon:
            self.tray_icon.stop()
    

    def _get_recording_icon_image(self) -> Image.Image:
        """返回紅色的 Recording 圖標 (用於閃爍的 'on' 狀態)。"""
        # 使用一個 16x16 的實心紅色方塊來實現高對比度閃爍
        return Image.new('RGB', (16, 16), (255, 0, 0))

    def _set_static_icon(self) -> None:
        """將圖標設回靜態的 Ready 狀態 (使用自定義的 Ready 圖標)。"""
        if self.tray_icon and self.ready_icon_img:
            self.tray_icon.icon = self.ready_icon_img


    def _blink_icon_toggle(self) -> None:
        """定時器觸發，切換系統託盤圖標的外觀 (紅白切換)。"""
        if not self.is_recording or not self.tray_icon:
            self._set_static_icon()
            return

        if self._blink_state:
            # 狀態 A: 顯示為白色（Ready）圖標
            self.tray_icon.icon = self.ready_icon_img
        else:
            # 狀態 B: 顯示為紅色（Recording）圖標
            self.tray_icon.icon = self.recording_icon_img
            
        self._blink_state = not self._blink_state # 切換狀態
        
        # [重要] 重新啟動定時器以繼續閃爍 (0.5 秒切換一次)
        self._blink_timer = threading.Timer(0.5, self._blink_icon_toggle) 
        self._blink_timer.start()

    # main.py (在 _start_recording 函式內，約 195 行處)

    def _start_recording(self) -> None:
        """Start audio recording."""
        self.is_recording = True
        self._show_status("Recording...")
        self.recorder.start()
        
        # [新增] 啟動閃爍定時器
        self._blink_state = False 
        self._blink_icon_toggle() # 立即啟動第一次切換和定時器

    


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


