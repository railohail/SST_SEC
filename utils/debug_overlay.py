"""Debug Overlay for showing status messages"""
import sys
import tkinter as tk
from threading import Lock
from typing import Optional

if sys.platform == "darwin":
    FONT_FAMILY = 'SF Pro Display'
else:
    FONT_FAMILY = 'Segoe UI'

class DebugOverlay:
    """
    Transparent overlay window for showing status messages.

    Shows a small floating window at the top of the screen
    with the current status (Recording, Processing, etc.)
    """

    def __init__(self):
        """Initialize the debug overlay."""
        self.root: Optional[tk.Tk] = None
        self.label: Optional[tk.Label] = None
        self._running = False
        self._lock = Lock()
        self._pending_message: Optional[str] = None
        self._pending_hide = False

    def _setup_window(self) -> None:
        """Setup the tkinter window."""
        self.root = tk.Tk()

        # Remove window decorations
        self.root.overrideredirect(True)

        # Make window always on top
        self.root.attributes('-topmost', True)

        # Set transparency (Mac)
        self.root.attributes('-alpha', 0.85)

        # Configure window
        self.root.configure(bg='#1a1a2e')

        # Create label
        self.label = tk.Label(
            self.root,
            text="",
            fg='#ffffff',
            bg='#1a1a2e',
            font=(FONT_FAMILY, 14, 'bold'),
            padx=20,
            pady=10
        )
        self.label.pack()

        # Position at top center of screen
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        x = (screen_width - 300) // 2
        y = 50  # Distance from top

        self.root.geometry(f"+{x}+{y}")

        # Initially hidden
        self.root.withdraw()

    def show(self, message: str) -> None:
        """
        Show a status message.

        Args:
            message: Message to display
        """
        with self._lock:
            self._pending_message = message
            self._pending_hide = False

    def hide(self) -> None:
        """Hide the overlay."""
        with self._lock:
            self._pending_hide = True

    def _update(self) -> None:
        """Update the overlay (called from main thread)."""
        with self._lock:
            if self._pending_message is not None:
                self.label.config(text=self._pending_message)
                self.root.deiconify()  # Show window
                self.root.update_idletasks()

                # Resize and reposition
                self.root.geometry("")  # Auto-size
                screen_width = self.root.winfo_screenwidth()
                window_width = self.root.winfo_width()
                x = (screen_width - window_width) // 2
                self.root.geometry(f"+{x}+50")

                self._pending_message = None

            if self._pending_hide:
                self.root.withdraw()  # Hide window
                self._pending_hide = False

        # Schedule next update
        if self._running:
            self.root.after(50, self._update)

    def run(self) -> None:
        """Run the overlay (blocking - call from separate thread)."""
        self._running = True
        self._setup_window()

        # Start update loop
        self.root.after(50, self._update)

        # Run tkinter main loop
        self.root.mainloop()

    def stop(self) -> None:
        """Stop the overlay."""
        self._running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
