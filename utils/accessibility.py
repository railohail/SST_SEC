"""macOS Accessibility API helper for reading text field state"""
from dataclasses import dataclass
from typing import Optional, Tuple

try:
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
