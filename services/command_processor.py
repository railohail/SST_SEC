"""Command Processor for parsing and applying correction commands"""
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .sequence_labeler import SequenceLabeler


class CommandType(Enum):
    """Types of correction commands"""
    DELETE = "delete"           # 刪除X
    REPLACE = "replace"         # 把X改成Y
    INSERT_BEFORE = "insert_before"  # 在X前面新增Y
    INSERT_AFTER = "insert_after"    # 在X後面新增Y
    NONE = "none"               # Not a command


@dataclass
class ParsedCommand:
    """Parsed correction command"""
    type: CommandType
    target: str = ""        # What to find/modify
    replacement: str = ""   # What to replace/insert with
    raw_command: str = ""   # Original command text


class CommandProcessor:
    """
    Processes correction commands for speech-to-text.

    Workflow:
    1. Detect if spoken text is a correction command
    2. Reconstruct model input: last_typed + [SEP] + command
    3. Run model to find B-Modify and B-Filling positions
    4. Apply the correction to original text
    """

    # Regex patterns for command detection
    # Multiple patterns per command type (checked in order)
    COMMAND_PATTERNS = {
        CommandType.DELETE: [
            re.compile(r'^刪除(.+)$'),      # 刪除X
            re.compile(r'^刪掉(.+)$'),      # 刪掉X
            re.compile(r'^把(.+)刪掉$'),    # 把X刪掉
            re.compile(r'^把(.+)刪除$'),    # 把X刪除
        ],
        CommandType.REPLACE: [
            re.compile(r'^把(.+)改成(.+)$'),  # 把X改成Y
            re.compile(r'^把(.+)換成(.+)$'),  # 把X換成Y
        ],
        CommandType.INSERT_BEFORE: [
            re.compile(r'^(?:請)?在(.+)前面新增(.+)$'),  # 在X前面新增Y
            re.compile(r'^(?:請)?在(.+)前面加上?(.+)$'), # 在X前面加Y
        ],
        CommandType.INSERT_AFTER: [
            re.compile(r'^(?:請)?在(.+)後面新增(.+)$'),  # 在X後面新增Y
            re.compile(r'^(?:請)?在(.+)後面加上?(.+)$'), # 在X後面加Y
        ],
    }

    @staticmethod
    def extract_replacement(text: str) -> str:
        """
        Extract the actual replacement character from descriptive text.

        Format: "X的Y" means use character Y (from reference word X)

        Smart matching for Chinese homophones:
        - If Y is found in X, use Y
        - If Y is NOT in X (Whisper misheard), use first char of X

        Examples:
            - "欣賞的欣" → "欣" (欣 is in 欣賞)
            - "欣賞的心" → "欣" (心 NOT in 欣賞, fallback to first char)
            - "氣候的氣" → "氣"
            - "心" → "心" (no 的, use as-is)

        Args:
            text: The replacement text (may contain "X的Y" pattern)

        Returns:
            The actual character(s) to use for replacement
        """
        # Pattern: "X的Y" - reference word X, target character Y
        match = re.match(r'^(.+)的(.+)$', text)
        if match:
            reference_word = match.group(1)  # e.g., "欣賞"
            stated_char = match.group(2)     # e.g., "欣" or "心"

            # Check if stated character exists in reference word
            if stated_char in reference_word:
                return stated_char

            # Whisper likely misheard - use first char of reference word
            # e.g., "欣賞的心" → 心 not in 欣賞 → use 欣
            return reference_word[0]

        return text  # No pattern, return as-is

    def __init__(self, labeler: SequenceLabeler = None):
        """
        Initialize the command processor.

        Args:
            labeler: Sequence labeler service (optional, created if not provided)
        """
        self.labeler = labeler or SequenceLabeler()

    def is_command(self, text: str) -> bool:
        """
        Check if text looks like a correction command.

        Args:
            text: Text to check

        Returns:
            True if text matches a command pattern
        """
        text = text.strip()
        for patterns in self.COMMAND_PATTERNS.values():
            for pattern in patterns:
                if pattern.match(text):
                    return True
        return False

    def parse_command(self, text: str) -> ParsedCommand:
        """
        Parse a correction command from text.

        Args:
            text: Command text

        Returns:
            ParsedCommand object
        """
        text = text.strip()

        # Try each pattern
        for cmd_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    groups = match.groups()

                    if cmd_type == CommandType.DELETE:
                        # Apply same extraction for "X的Y" pattern
                        # e.g., "割錢的錢" → "錢"
                        target = self.extract_replacement(groups[0].strip())
                        return ParsedCommand(
                            type=cmd_type,
                            target=target,
                            raw_command=text
                        )
                    elif cmd_type == CommandType.REPLACE:
                        return ParsedCommand(
                            type=cmd_type,
                            target=self.extract_replacement(groups[0]),
                            replacement=self.extract_replacement(groups[1]),
                            raw_command=text
                        )
                    elif cmd_type in (CommandType.INSERT_BEFORE, CommandType.INSERT_AFTER):
                        return ParsedCommand(
                            type=cmd_type,
                            target=self.extract_replacement(groups[0]),
                            replacement=self.extract_replacement(groups[1]),
                            raw_command=text
                        )

        return ParsedCommand(type=CommandType.NONE, raw_command=text)

    def process(
        self,
        spoken_text: str,
        last_typed_text: str
    ) -> Tuple[str, bool]:
        """
        Process spoken text and apply corrections if needed.

        Args:
            spoken_text: What the user just spoke
            last_typed_text: The text that was previously typed

        Returns:
            Tuple of (result_text, was_command)
            - result_text: The corrected text or original spoken text
            - was_command: True if a correction was applied
        """
        # Check if this is a command
        if not self.is_command(spoken_text):
            return spoken_text, False

        # Parse the command
        command = self.parse_command(spoken_text)
        if command.type == CommandType.NONE:
            return spoken_text, False

        # If no previous text, can't apply correction
        if not last_typed_text:
            return spoken_text, False

        # Reconstruct input for model (format: original [SEP] command)
        model_input = f"{last_typed_text} [SEP] {spoken_text}"

        # Get model predictions
        labels, modify_positions, filling_positions = self.labeler.predict_with_positions(model_input)

        # Apply correction based on command type and model predictions
        result = self._apply_correction(
            original_text=last_typed_text,
            command=command,
            labels=labels,
            modify_positions=modify_positions,
            filling_positions=filling_positions
        )

        return result, True

    def _apply_correction(
        self,
        original_text: str,
        command: ParsedCommand,
        labels: List[str],
        modify_positions: List[int],
        filling_positions: List[int]
    ) -> str:
        """
        Apply the correction to the original text.

        BERT-first approach:
        1. Trust the BERT model's modify_positions to determine WHERE to modify
        2. Only use extract_replacement for determining WHAT to replace with
        3. Fall back to text search only if model found no valid positions

        Args:
            original_text: Text to modify
            command: Parsed command
            labels: Labels from model
            modify_positions: Positions marked as B-Modify
            filling_positions: Positions marked as B-Filling

        Returns:
            Corrected text
        """
        orig_len = len(original_text)

        # Debug output
        print(f"  [DEBUG] original_text: '{original_text}'")
        print(f"  [DEBUG] command: type={command.type.value}, target='{command.target}', replacement='{command.replacement}'")
        print(f"  [DEBUG] modify_positions: {modify_positions}")

        # BERT-first: If model found valid positions in original text, trust them
        if modify_positions:
            for pos in modify_positions:
                if pos < orig_len:
                    # Model found a valid position - use it directly
                    actual_char = original_text[pos]
                    print(f"  [DEBUG] BERT model says modify position {pos}, char='{actual_char}'")

                    # Create a modified command with the actual character as target
                    model_based_command = ParsedCommand(
                        type=command.type,
                        target=actual_char,  # Use actual char at model position
                        replacement=command.replacement,
                        raw_command=command.raw_command
                    )
                    return self._apply_at_position(original_text, model_based_command, pos)

            print(f"  [DEBUG] Model positions {modify_positions} all out of bounds")

        # Fallback: Use command target to find position (only if model failed)
        print(f"  [DEBUG] No valid model position, falling back to text search for '{command.target}'")
        return self._apply_by_target(original_text, command)

    def _apply_at_position(
        self,
        text: str,
        command: ParsedCommand,
        position: int
    ) -> str:
        """Apply correction at a specific position."""
        target_len = len(command.target) if command.target else 1

        if command.type == CommandType.DELETE:
            # Delete target characters starting at position
            return text[:position] + text[position + target_len:]

        elif command.type == CommandType.REPLACE:
            # Replace target characters at position
            return text[:position] + command.replacement + text[position + target_len:]

        elif command.type == CommandType.INSERT_BEFORE:
            # Insert before position
            return text[:position] + command.replacement + text[position:]

        elif command.type == CommandType.INSERT_AFTER:
            # Insert after position
            return text[:position + 1] + command.replacement + text[position + 1:]

        return text

    def _apply_by_target(
        self,
        text: str,
        command: ParsedCommand
    ) -> str:
        """Apply correction by finding target string."""
        target = command.target

        if command.type == CommandType.DELETE:
            # Delete first occurrence
            pos = text.find(target)
            if pos != -1:
                return text[:pos] + text[pos + len(target):]

        elif command.type == CommandType.REPLACE:
            # Replace first occurrence
            return text.replace(target, command.replacement, 1)

        elif command.type == CommandType.INSERT_BEFORE:
            # Insert before first occurrence
            pos = text.find(target)
            if pos != -1:
                return text[:pos] + command.replacement + text[pos:]

        elif command.type == CommandType.INSERT_AFTER:
            # Insert after first occurrence
            pos = text.find(target)
            if pos != -1:
                end_pos = pos + len(target)
                return text[:end_pos] + command.replacement + text[end_pos:]

        return text
