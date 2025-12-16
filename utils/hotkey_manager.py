"""
Hotkey Manager using pynput.keyboard.GlobalHotKeys (Safe Mode)
"""
import time
import sys
from typing import Callable
from pynput import keyboard

class HotkeyManager:
    """
    Manages global hotkey detection using the safe GlobalHotKeys wrapper.
    """

    def __init__(self, hotkey: str, callback: Callable[[], None]):
        """
        Initialize the hotkey manager.

        Args:
            hotkey: The hotkey combination string (e.g., '<f9>', '<ctrl>+<alt>+<space>')
            callback: Function to call when hotkey is triggered
        """
        self.hotkey_str = hotkey
        self.callback = callback
        self.listener = None
        self.is_running = False

        # [Windows 修正] pynput 在 Windows 看不懂 <cmd>，自動轉成 <ctrl>
        if sys.platform == "win32" and "<cmd>" in self.hotkey_str:
            self.hotkey_str = self.hotkey_str.replace("<cmd>", "<ctrl>")

    def start(self) -> None:
        """Start the hotkey listener safely."""
        if self.is_running:
            return
            
        self.is_running = True
        
        print(f"[HotkeyManager] Registering hotkey: {self.hotkey_str}")
        
        try:
            # 💡 核心差異：使用 GlobalHotKeys
            # 這種寫法是 pynput 內部幫你處理好判定，
            # 只有當「完全符合」F9 時，才會觸發 on_activate。
            # 其他按鍵完全不會被這裡攔截或處理。
            self.listener = keyboard.GlobalHotKeys({
                self.hotkey_str: self.on_activate
            })
            self.listener.start()
            
        except Exception as e:
            print(f"[HotkeyManager] Error starting listener: {e}")
            print(f"請檢查 config.py 的熱鍵格式是否正確 (例如 '<f9>')")
            self.is_running = False

    def on_activate(self):
        """Callback when hotkey is triggered."""
        if self.callback:
            # 這裡不需要開 Thread，因為 main.py 裡面的 callback 會處理
            self.callback()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        self.is_running = False
        if self.listener:
            try:
                self.listener.stop()
            except:
                pass
            self.listener = None

    def wait(self) -> None:
        """
        Keep the main thread alive, but allow Ctrl+C to exit.
        """
        if self.listener:
            try:
                # 💡 關鍵修正：不要用 join() 死守
                # 改用迴圈 + sleep，這樣您的 Ctrl+C 才能被 main.py 捕捉到
                while self.listener.is_alive() and self.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop()