"""Keyboard Simulator for typing text at cursor position"""
import sys
import time
import random
from typing import Optional, Tuple

import pyperclip
from pynput.keyboard import Controller, Key

# Try to import accessibility helper
try:
    from .accessibility import AccessibilityHelper, TextFieldState
    ACCESSIBILITY_AVAILABLE = True
except ImportError:
    ACCESSIBILITY_AVAILABLE = False
    AccessibilityHelper = None
    TextFieldState = None

MODIFIER_KEY = Key.cmd if sys.platform == "darwin" else Key.ctrl


class KeyboardSimulator:
    """
    Simulates keyboard input to type text at cursor position.

    Uses clipboard + paste for reliable Chinese character support.
    Can read selected text via macOS Accessibility APIs.
    """

    def __init__(self):
        """Initialize the keyboard simulator."""
        self.keyboard = Controller()
        self._original_clipboard: Optional[str] = None
        self._accessibility: Optional[AccessibilityHelper] = None

        # Try to initialize accessibility helper
        if ACCESSIBILITY_AVAILABLE:
            try:
                self._accessibility = AccessibilityHelper()
            except Exception as e:
                print(f"Warning: Could not initialize accessibility: {e}")

    def type_text(self, text: str, use_clipboard: bool = True) -> None:
        """
        Type text at current cursor position.

        Args:
            text: Text to type
            use_clipboard: If True, use clipboard+paste (recommended for Chinese)
        """
        if not text:
            return

        if use_clipboard:
            self._type_via_clipboard(text)
        else:
            self._type_directly(text)

    def _type_via_clipboard(self, text: str) -> None:
        """Type text using clipboard and paste."""
        # Save original clipboard content
        try:
            self._original_clipboard = pyperclip.paste()
        except Exception:
            self._original_clipboard = None

        # Copy text to clipboard
        pyperclip.copy(text)

        # Small delay to ensure clipboard is updated
        time.sleep(0.05)

        # Paste (Cmd+V on Mac)
        self.keyboard.press(MODIFIER_KEY)
        self.keyboard.press('v')
        self.keyboard.release('v')
        self.keyboard.release(MODIFIER_KEY)

        # Small delay after paste
        time.sleep(0.05)

        # Optionally restore original clipboard
        # (commented out to avoid confusion - user might want to paste again)
        # if self._original_clipboard is not None:
        #     time.sleep(0.1)
        #     pyperclip.copy(self._original_clipboard)

    def _type_directly(self, text: str) -> None:
        """Type text character by character (may not work for all characters)."""
        for char in text:
            self.keyboard.type(char)
            time.sleep(0.01)  # Small delay between characters

    def delete_chars(self, count: int) -> None:
        """
        Delete characters before cursor.

        Args:
            count: Number of characters to delete
        """
        for _ in range(count):
            self.keyboard.press(Key.backspace)
            self.keyboard.release(Key.backspace)
            time.sleep(0.01)

    def select_all_and_delete(self) -> None:
        """Select all text and delete (Cmd+A, Delete)."""
        self.keyboard.press(MODIFIER_KEY)
        self.keyboard.press('a')
        self.keyboard.release('a')
        self.keyboard.release(MODIFIER_KEY)
        time.sleep(0.05)

        self.keyboard.press(Key.backspace)
        self.keyboard.release(Key.backspace)

    def replace_last_typed(self, old_text: str, new_text: str) -> None:
        """
        Replace the last typed text with new text.

        This deletes len(old_text) characters and types new_text.

        Args:
            old_text: The text that was previously typed
            new_text: The new text to type
        """
        # Delete old text
        self.delete_chars(len(old_text))

        # Type new text
        time.sleep(0.05)
        self.type_text(new_text)

    def shuffle_text_effect(self, text: str, iterations: int = 4, delay: float = 0.08) -> None:
        """
        Create a shuffle/scramble effect on text like a slot machine loading.

        Uses select + replace to avoid cursor jumping back and forth.

        Args:
            text: The text to shuffle (will be selected and replaced)
            iterations: Number of shuffle iterations (default: 4)
            delay: Delay between iterations in seconds (default: 0.08)
        """
        if not text or len(text) < 2:
            return

        text_len = len(text)
        chars = list(text)

        for i in range(iterations):
            # Shuffle the characters randomly
            shuffled = chars.copy()
            random.shuffle(shuffled)
            shuffled_text = ''.join(shuffled)

            # Select text backwards (Shift + Left Arrow for each character)
            self._select_chars_backwards(text_len)
            time.sleep(0.02)

            # Type shuffled text (replaces selection, cursor ends at end)
            self.type_text(shuffled_text)
            time.sleep(delay)

        # Select the last shuffled text so caller can replace it
        self._select_chars_backwards(text_len)

    def _select_chars_backwards(self, count: int) -> None:
        """Select characters backwards from cursor using Shift+Left Arrow."""
        self.keyboard.press(Key.shift)
        for _ in range(count):
            self.keyboard.tap(Key.left)
        self.keyboard.release(Key.shift)
        time.sleep(0.01)

    def get_selected_text(self) -> str:
        """
        Get the currently selected text from the focused field.

        Uses Accessibility API if available, otherwise returns empty string.

        Returns:
            Selected text, or empty string if none/unavailable.
        """
        if self._accessibility:
            try:
                return self._accessibility.get_selected_text()
            except Exception:
                pass
        return ""

    def get_text_field_state(self) -> Optional[TextFieldState]:
        """
        Get the full state of the focused text field.

        Returns:
            TextFieldState with text, selection, cursor info,
            or None if unavailable.
        """
        if self._accessibility:
            try:
                return self._accessibility.get_text_field_state()
            except Exception:
                pass
        return None

    def get_full_text(self) -> str:
        """
        Get all text from the focused field.

        Returns:
            Full text, or empty string if unavailable.
        """
        if self._accessibility:
            try:
                return self._accessibility.get_full_text()
            except Exception:
                pass
        return ""

    def replace_selection(self, new_text: str) -> None:
        """
        Replace the currently selected text with new text.

        If text is selected, typing will replace it automatically.

        Args:
            new_text: Text to replace selection with
        """
        # Simply type - selected text gets replaced automatically
        self.type_text(new_text)

    def has_accessibility(self) -> bool:
        """Check if accessibility features are available."""
        return self._accessibility is not None
