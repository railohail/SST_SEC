#!/usr/bin/env python3
"""
Test Mode WITH Model - Uses your BERT+CRF model

Slower to start (loads BERT), but uses actual model predictions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.command_processor import CommandProcessor
from services.sequence_labeler import SequenceLabeler


def main():
    print("=" * 60)
    print("Speech Command App - TEST MODE WITH MODEL")
    print("=" * 60)
    print("Loading BERT+CRF model... (this may take a minute)")
    print()

    # Initialize with real model
    labeler = SequenceLabeler()
    labeler.load()  # Actually load the model

    processor = CommandProcessor(labeler=labeler)

    print()
    print("Model loaded! Ready for testing.")
    print("Commands: 刪除X, 把X改成Y, 在X前面新增Y, 在X後面新增Y")
    print("Type 'quit' to exit.")
    print("=" * 60)
    print()

    last_typed = ""

    while True:
        try:
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

                # Use full model processing
                result, was_cmd = processor.process(user_input, last_typed)

                if was_cmd:
                    print(f"  → Model processed correction")
                    print(f"  → Result: {result}")
                    last_typed = result
                else:
                    print(f"  → Could not apply correction")

            else:
                # Check if text contains [SEP] (original + command in one)
                if '[SEP]' in user_input:
                    print(f"  → Detected [SEP] format")
                    labels, modify_pos, filling_pos = labeler.predict_with_positions(user_input)

                    # Show predictions
                    parts = user_input.split('[SEP]')
                    original = parts[0].strip()
                    command = parts[1].strip() if len(parts) > 1 else ""

                    print(f"  → Original: {original}")
                    print(f"  → Command: {command}")
                    print(f"  → B-Modify positions: {modify_pos}")
                    print(f"  → B-Filling positions: {filling_pos}")

                    # Show character labels
                    print(f"  → Labels for key positions:")
                    for i, (char, label) in enumerate(zip(user_input, labels)):
                        if label != 'O':
                            print(f"      [{i}] '{char}' = {label}")

                    last_typed = original
                else:
                    print(f"  → Normal text: {user_input}")
                    last_typed = user_input

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
