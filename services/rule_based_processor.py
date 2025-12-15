"""Rule-Based Command Processor - No ML required

A lightweight alternative to the BERT-based CommandProcessor.
Uses pure regex and string matching to handle correction commands.

Supports:
- 把X改成Y / 把X換成Y (replace)
- 刪除X / 把X刪掉 / 把X刪除 (delete)
- 在X前面新增Y / 在X前面加Y (insert before)
- 在X後面新增Y / 在X後面加Y (insert after)

The "X的Y" pattern is supported for disambiguation:
- "把高興的興改成欣賞的欣" → replace 興 with 欣
- "把擱淺的擱刪除" → delete 擱
"""
import re
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    """Types of correction commands"""
    DELETE = "delete"
    REPLACE = "replace"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    NONE = "none"


@dataclass
class ParsedCommand:
    """Parsed correction command"""
    type: CommandType
    target: str = ""
    replacement: str = ""
    raw_command: str = ""


class RuleBasedProcessor:
    """
    Rule-based command processor for speech correction.

    No ML model required - uses pure regex and string matching.
    Designed as a drop-in replacement for CommandProcessor when --notML is used.
    """

    # Command patterns (same as CommandProcessor)
    COMMAND_PATTERNS = {
        CommandType.DELETE: [
            re.compile(r'^刪除(.+)$'),
            re.compile(r'^刪掉(.+)$'),
            re.compile(r'^把(.+)刪掉$'),
            re.compile(r'^把(.+)刪除$'),
        ],
        CommandType.REPLACE: [
            re.compile(r'^把(.+)改成(.+)$'),
            re.compile(r'^把(.+)換成(.+)$'),
        ],
        CommandType.INSERT_BEFORE: [
            re.compile(r'^(?:請)?在(.+)前面新增(.+)$'),
            re.compile(r'^(?:請)?在(.+)前面加入(.+)$'),
            re.compile(r'^(?:請)?在(.+)前面加上(.+)$'),
            re.compile(r'^(?:請)?在(.+)前面加(.+)$'),
        ],
        CommandType.INSERT_AFTER: [
            re.compile(r'^(?:請)?在(.+)後面新增(.+)$'),
            re.compile(r'^(?:請)?在(.+)後面加入(.+)$'),
            re.compile(r'^(?:請)?在(.+)後面加上(.+)$'),
            re.compile(r'^(?:請)?在(.+)後面加(.+)$'),
        ],
    }

    def __init__(self):
        """Initialize the rule-based processor (no model loading needed)."""
        # Dummy labeler attribute for compatibility with main.py's preload
        self.labeler = _DummyLabeler()

    @staticmethod
    def extract_replacement(text: str) -> str:
        """
        Extract the actual character from "X的Y" pattern.

        Examples:
            - "欣賞的欣" → "欣" (欣 is in 欣賞)
            - "欣賞的心" → "欣" (心 NOT in 欣賞, fallback to first char)
            - "氣候的氣" → "氣"
            - "心" → "心" (no 的, use as-is)
        """
        match = re.match(r'^(.+)的(.+)$', text)
        if match:
            reference_word = match.group(1)
            stated_char = match.group(2)

            # If stated char is in reference word, use it
            if stated_char in reference_word:
                return stated_char

            # Fallback: use first char of reference word (Whisper likely misheard)
            return reference_word[0]

        return text

    def is_command(self, text: str) -> bool:
        """Check if text is a correction command."""
        text = text.strip()
        for patterns in self.COMMAND_PATTERNS.values():
            for pattern in patterns:
                if pattern.match(text):
                    return True
        return False

    def parse_command(self, text: str) -> ParsedCommand:
        """Parse a correction command from text."""
        text = text.strip()

        for cmd_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    groups = match.groups()

                    if cmd_type == CommandType.DELETE:
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

    def process(self, spoken_text: str, last_typed_text: str) -> Tuple[str, bool]:
        """
        Process spoken text and apply corrections.

        This is the main entry point - same interface as CommandProcessor.

        Args:
            spoken_text: The command that was spoken
            last_typed_text: The text to apply the correction to

        Returns:
            Tuple of (result_text, was_command)
        """
        if not self.is_command(spoken_text):
            return spoken_text, False

        command = self.parse_command(spoken_text)
        if command.type == CommandType.NONE:
            return spoken_text, False

        if not last_typed_text:
            return spoken_text, False

        # Apply the correction using string matching
        result = self._apply_correction(last_typed_text, command)
        return result, True

    def _apply_correction(self, text: str, command: ParsedCommand) -> str:
        """
        Apply the correction to text using string matching.

        Tries multiple strategies:
        1. Exact match of target
        2. If target has multiple chars, try each char individually
        3. Fuzzy matching for homophones (same pinyin)
        """
        target = command.target

        print(f"  [RULE] Applying {command.type.value}: target='{target}', replacement='{command.replacement}'")
        print(f"  [RULE] Original text: '{text}'")

        # Strategy 1: Direct match
        if target in text:
            result = self._apply_at_target(text, command, target)
            print(f"  [RULE] Direct match found, result: '{result}'")
            return result

        # Strategy 2: Try each character individually (for multi-char targets)
        if len(target) > 1:
            for char in target:
                if char in text:
                    modified_command = ParsedCommand(
                        type=command.type,
                        target=char,
                        replacement=command.replacement,
                        raw_command=command.raw_command
                    )
                    result = self._apply_at_target(text, modified_command, char)
                    print(f"  [RULE] Partial match '{char}' found, result: '{result}'")
                    return result

        # Strategy 3: Try homophone matching
        result = self._try_homophone_match(text, command)
        if result != text:
            print(f"  [RULE] Homophone match found, result: '{result}'")
            return result

        print(f"  [RULE] No match found, returning original text")
        return text

    def _apply_at_target(self, text: str, command: ParsedCommand, target: str) -> str:
        """Apply the command at the first occurrence of target."""
        pos = text.find(target)
        if pos == -1:
            return text

        if command.type == CommandType.DELETE:
            return text[:pos] + text[pos + len(target):]

        elif command.type == CommandType.REPLACE:
            return text[:pos] + command.replacement + text[pos + len(target):]

        elif command.type == CommandType.INSERT_BEFORE:
            return text[:pos] + command.replacement + text[pos:]

        elif command.type == CommandType.INSERT_AFTER:
            end_pos = pos + len(target)
            return text[:end_pos] + command.replacement + text[end_pos:]

        return text

    def _try_homophone_match(self, text: str, command: ParsedCommand) -> str:
        """
        Try to find homophones of the target in the text.

        Common Chinese homophones for correction:
        - 的/得/地
        - 在/再
        - 做/作
        - 他/她/它
        - etc.
        """
        HOMOPHONES = {
            '的': ['得', '地'],
            '得': ['的', '地'],
            '地': ['的', '得'],
            '在': ['再'],
            '再': ['在'],
            '做': ['作'],
            '作': ['做'],
            '他': ['她', '它', '祂'],
            '她': ['他', '它'],
            '它': ['他', '她'],
            '那': ['哪', '拿'],
            '哪': ['那'],
            '已': ['以', '亦'],
            '以': ['已', '亦'],
            '像': ['象', '相'],
            '象': ['像', '相'],
            '相': ['像', '象'],
            '須': ['需'],
            '需': ['須'],
            '即': ['既', '及'],
            '既': ['即', '及'],
            '及': ['即', '既'],
            '坐': ['座', '做'],
            '座': ['坐'],
            '帳': ['賬', '張'],
            '賬': ['帳'],
            '歷': ['曆', '力'],
            '曆': ['歷'],
            '欣': ['新', '心', '辛', '薪'],
            '新': ['欣', '心', '辛', '薪'],
            '心': ['欣', '新', '辛', '薪'],
            '辛': ['欣', '新', '心', '薪'],
            '薪': ['欣', '新', '心', '辛'],
            '興': ['星', '腥', '惺'],
            '氣': ['器', '棄', '汽', '泣'],
            '器': ['氣', '棄', '汽', '泣'],
            '棄': ['氣', '器', '汽'],
            '汽': ['氣', '器'],
            '擱': ['歌', '哥', '鴿', '割'],
            '歌': ['擱', '哥', '鴿', '割'],
            '哥': ['擱', '歌', '鴿', '割'],
        }

        target = command.target

        # Get homophones for the target character
        if target in HOMOPHONES:
            for homophone in HOMOPHONES[target]:
                if homophone in text:
                    # Found a homophone in text - apply correction to it
                    modified_command = ParsedCommand(
                        type=command.type,
                        target=homophone,
                        replacement=command.replacement,
                        raw_command=command.raw_command
                    )
                    return self._apply_at_target(text, modified_command, homophone)

        return text


class _DummyLabeler:
    """Dummy labeler for compatibility - does nothing."""

    def load(self):
        """No-op load for compatibility with main.py preloading."""
        print("  [RULE] Rule-based mode - no ML model to load")
