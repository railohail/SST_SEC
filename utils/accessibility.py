"""
Accessibility Helper (Cross-Platform)

macOS: Uses ApplicationServices API to read UI text directly.
Windows: Uses Clipboard (Ctrl+C) simulation to read selected text.
"""
import sys # [Added]
import time # [Added] For Windows delay
from dataclasses import dataclass
from typing import Optional, Tuple


try:
    if sys.platform == "darwin":
        from ApplicationServices import (
            AXUIElementCreateSystemWide,
            AXUIElementCopyAttributeValue,
            AXUIElementSetAttributeValue,
            kAXFocusedUIElementAttribute,
            kAXValueAttribute,
            kAXSelectedTextAttribute,
            kAXSelectedTextRangeAttribute,
            kAXNumberOfCharactersAttribute,
            kAXRoleAttribute,
        )
        from CoreFoundation import CFRange
        ACCESSIBILITY_AVAILABLE = True
    else:
        # Windows dependencies
        import pyperclip
        from pynput.keyboard import Controller, Key
        ACCESSIBILITY_AVAILABLE = True # Set to True to enable Windows fallback
except ImportError:
    ACCESSIBILITY_AVAILABLE = False



@dataclass
class TextFieldState:
    """State of a focused text field"""
    full_text: str = ""           # All text in the field
    selected_text: str = ""       # Currently selected text
    cursor_position: int = 0      # Cursor position (or start of selection)
    selection_length: int = 0     # Length of selection (0 if no selection)
    has_selection: bool = False   # Whether text is selected

if sys.platform == "darwin":
    class AccessibilityHelper:
        """
        Helper for reading text field state via macOS Accessibility APIs.

        Requires Accessibility permissions in System Preferences.
        """

        def __init__(self):
            if not ACCESSIBILITY_AVAILABLE:
                raise ImportError(
                    "ApplicationServices not available. "
                    "Install with: pip install pyobjc-framework-ApplicationServices"
                )
            self.system_wide = AXUIElementCreateSystemWide()

        def get_focused_element(self):
            """Get the currently focused UI element."""
            error, focused = AXUIElementCopyAttributeValue(
                self.system_wide,
                kAXFocusedUIElementAttribute,
                None
            )
            if error == 0 and focused:
                return focused
            return None

        def get_attribute(self, element, attribute: str):
            """Get an attribute value from a UI element."""
            if element is None:
                return None
            error, value = AXUIElementCopyAttributeValue(element, attribute, None)
            if error == 0:
                return value
            return None

        def get_text_field_state(self) -> Optional[TextFieldState]:
            """
            Get the current state of the focused text field.

            Returns:
                TextFieldState with text, selection, and cursor info,
                or None if no text field is focused.
            """
            focused = self.get_focused_element()
            if focused is None:
                return None

            # Check if it's a text field (has value attribute)
            role = self.get_attribute(focused, kAXRoleAttribute)

            # Get full text
            full_text = self.get_attribute(focused, kAXValueAttribute)
            if full_text is None:
                full_text = ""

            # Get selected text
            selected_text = self.get_attribute(focused, kAXSelectedTextAttribute)
            if selected_text is None:
                selected_text = ""

            # Get selection range
            selection_range = self.get_attribute(focused, kAXSelectedTextRangeAttribute)
            cursor_position = 0
            selection_length = 0

            if selection_range is not None:
                # selection_range is a CFRange (location, length)
                try:
                    cursor_position = selection_range.location
                    selection_length = selection_range.length
                except AttributeError:
                    # Try unpacking if it's a different format
                    pass

            return TextFieldState(
                full_text=str(full_text) if full_text else "",
                selected_text=str(selected_text) if selected_text else "",
                cursor_position=cursor_position,
                selection_length=selection_length,
                has_selection=selection_length > 0 or len(selected_text) > 0
            )

        def get_selected_text(self) -> str:
            """
            Get only the selected text from focused field.

            Returns:
                Selected text, or empty string if none.
            """
            state = self.get_text_field_state()
            if state and state.selected_text:
                return state.selected_text
            return ""

        def get_full_text(self) -> str:
            """
            Get all text from focused field.

            Returns:
                Full text, or empty string if unavailable.
            """
            state = self.get_text_field_state()
            if state:
                return state.full_text
            return ""

else:
    # ==========================================
    # Windows Implementation (Clipboard Fallback)
    # ==========================================
    class AccessibilityHelper:
        """
        Windows Helper using Clipboard simulation.
        Falls back to copy/paste as UIAutomation is flaky on Windows.
        """
        def __init__(self):
            self.keyboard = Controller()

        def get_selected_text(self) -> str:
            """Get selected text by simulating Ctrl+C."""
            try:
                # 1. Clear clipboard to detect failure
                pyperclip.copy("")
                
                # 2. Simulate Ctrl+C
                with self.keyboard.pressed(Key.ctrl):
                    self.keyboard.tap('c')
                
                # [UX Note] Short delay required for Windows clipboard I/O
                time.sleep(0.05) 
                
                # 3. Read clipboard
                return pyperclip.paste()
            except Exception:
                return ""

        def get_full_text(self) -> str:
            """Cannot reliably get full text on Windows without intrusion."""
            return ""

        def get_text_field_state(self) -> Optional[TextFieldState]:
            """Simulate state based on selection only."""
            text = self.get_selected_text()
            return TextFieldState(
                full_text=text, # Assume context is limited to selection
                selected_text=text,
                cursor_position=0,
                selection_length=len(text),
                has_selection=bool(text)
            )
    