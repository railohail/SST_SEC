#!/usr/bin/env python3
"""
Test Mode - Type instead of speak

For testing the app without a microphone.
Type your text and corrections manually.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.command_processor import CommandProcessor


def main():
    print("=" * 60)
    print("Speech Command App - TEXT TEST MODE")
    print("=" * 60)
    print("Type text to simulate speech input.")
    print("Commands: 刪除X, 把X改成Y, 在X前面新增Y, 在X後面新增Y")
    print("Type 'quit' or 'q' to exit.")
    print("=" * 60)
    print()

    processor = CommandProcessor()
    last_typed = ""

    # Skip model loading for quick testing
    # Comment this out to test with actual model
    processor.labeler._loaded = True  # Fake loaded state for quick test

    while True:
        try:
            # Show current state
            if last_typed:
                print(f"[Current text]: {last_typed}")

            user_input = input("\n> Enter text (or command): ").strip()

            if user_input.lower() in ('quit', 'q', 'exit'):
                print("Bye!")
                break

            if not user_input:
                continue

            # Check if it's a command
            if processor.is_command(user_input):
                print(f"  → Detected as COMMAND")

                if not last_typed:
                    print("  → No previous text to correct!")
                    continue

                # Apply correction (using fallback regex mode since model not loaded)
                parsed = processor.parse_command(user_input)
                result = processor._apply_by_target(last_typed, parsed)

                print(f"  → Command type: {parsed.type.value}")
                print(f"  → Target: '{parsed.target}'")
                if parsed.replacement:
                    print(f"  → Replacement: '{parsed.replacement}'")
                print(f"  → Result: {result}")

                last_typed = result
            else:
                print(f"  → Normal text, would type: {user_input}")
                last_typed = user_input

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
