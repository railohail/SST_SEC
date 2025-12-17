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

        if sys.platform == "win32" and "<cmd>" in self.hotkey_str:
            self.hotkey_str = self.hotkey_str.replace("<cmd>", "<ctrl>")

    def start(self) -> None:
        """Start the hotkey listener safely."""
        if self.is_running:
            return
            
        self.is_running = True
        
        print(f"[HotkeyManager] Registering hotkey: {self.hotkey_str}")
        
        try:
            self.listener = keyboard.GlobalHotKeys({
                self.hotkey_str: self.on_activate
            })
            self.listener.start()
            
        except Exception as e:
            print(f"[HotkeyManager] Error starting listener: {e}")
            print(f"Please Check if the hotkey format in config.py is correct")
            self.is_running = False

    def on_activate(self):
        """Callback when hotkey is triggered."""
        if self.callback:
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
                while self.listener.is_alive() and self.is_running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self.stop()