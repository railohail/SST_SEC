"""Global Hotkey Manager using pynput"""
from typing import Callable, Optional, Set
from threading import Thread

from pynput import keyboard
from pynput.keyboard import Key, KeyCode

from config import config


class HotkeyManager:
    """
    Manages global hotkey detection for toggle recording.

    Default hotkey: Cmd+Shift+Space (configurable)
    """

    # Map string names to pynput keys
    KEY_MAP = {
        '<cmd>': Key.cmd,
        '<shift>': Key.shift,
        '<ctrl>': Key.ctrl,
        '<alt>': Key.alt,
        '<space>': Key.space,
        '<enter>': Key.enter,
        '<tab>': Key.tab,
        '<esc>': Key.esc,
    }

    def __init__(
        self,
        hotkey: str = None,
        callback: Callable[[], None] = None
    ):
        """
        Initialize the hotkey manager.

        Args:
            hotkey: Hotkey string (e.g., "<cmd>+<shift>+<space>")
            callback: Function to call when hotkey is pressed
        """
        self.hotkey_str = hotkey or config.HOTKEY
        self.callback = callback

        # Parse hotkey string
        self.hotkey_keys = self._parse_hotkey(self.hotkey_str)

        # Track currently pressed keys
        self.current_keys: Set = set()

        # Listener
        self.listener: Optional[keyboard.Listener] = None
        self._running = False

    def _parse_hotkey(self, hotkey_str: str) -> Set:
        """Parse hotkey string into set of pynput keys."""
        keys = set()
        parts = hotkey_str.lower().split('+')

        for part in parts:
            part = part.strip()
            if part in self.KEY_MAP:
                keys.add(self.KEY_MAP[part])
            elif len(part) == 1:
                # Single character key
                keys.add(KeyCode.from_char(part))
            else:
                print(f"Warning: Unknown key '{part}' in hotkey")

        return keys

    def _on_press(self, key) -> None:
        """Handle key press event."""
        # Normalize key
        if hasattr(key, 'char') and key.char:
            self.current_keys.add(KeyCode.from_char(key.char.lower()))
        else:
            self.current_keys.add(key)

        # Check if hotkey combination is pressed
        if self.hotkey_keys.issubset(self.current_keys):
            if self.callback:
                # Call callback in separate thread to avoid blocking
                Thread(target=self.callback).start()

    def _on_release(self, key) -> None:
        """Handle key release event."""
        # Normalize key
        if hasattr(key, 'char') and key.char:
            self.current_keys.discard(KeyCode.from_char(key.char.lower()))
        else:
            self.current_keys.discard(key)

    def start(self) -> None:
        """Start listening for hotkey."""
        if self._running:
            return

        self._running = True
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        print(f"Hotkey listener started. Press {self.hotkey_str} to toggle recording.")

    def stop(self) -> None:
        """Stop listening for hotkey."""
        if not self._running:
            return

        self._running = False
        if self.listener:
            self.listener.stop()
            self.listener = None

    def wait(self) -> None:
        """Wait for listener to finish (blocks until stopped)."""
        if self.listener:
            self.listener.join()
