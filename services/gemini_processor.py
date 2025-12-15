"""Gemini API-based Command Processor

Uses Google's Gemini 2.5 Flash model to intelligently apply text corrections.
Combines rule-based command detection with LLM-powered correction application.

Flow:
1. Detect if input is a command (using regex patterns)
2. If command, send to Gemini API with structured prompt
3. Gemini applies the correction with full language understanding
"""
import os
import re
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai not installed. Run: pip install google-genai")


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
    raw_command: str = ""


class GeminiProcessor:
    """
    Gemini API-based command processor.

    Uses LLM to understand and apply Chinese text corrections.
    Better at handling:
    - Ambiguous homophones
    - Context-dependent corrections
    - Complex multi-character operations
    """

    # Gemini API configuration
    MODEL = "gemini-2.5-flash"

    # Command detection patterns (same as rule-based)
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

    # Prompt template for Gemini (single-shot, clear instructions)
    PROMPT_TEMPLATE = """你是中文文字校正助手。用戶通過語音輸入指令來修改文字。

## 重要：理解「X的Y」格式
因為是語音輸入，用戶會用「參考詞的字」來指明要用哪個同音字。
- 「站立的站」= 用「站」這個字（站立只是參考發音）
- 「斬斷的斬」= 用「斬」這個字
- 「天氣的氣」= 用「氣」這個字
- 「器材的器」= 用「器」這個字

## 指令格式：
1. 「把A的B改成C的D」= 在原文中找到字B，替換成字D
   - A和C只是參考詞，幫助識別同音字
   - 實際操作是：找B → 換成D

2. 「把X改成Y」= 直接把X換成Y

3. 「把X刪除」= 刪除X

4. 「在X前面/後面加Y」= 插入Y

## 替換範例：
原始：新報氣流站
指令：把站立的站改成斬斷的展
分析：找「站」→ 換成「斬」
結果：新報氣流斬

原始：今天天氣很好
指令：把天氣的氣改成器材的氣
分析：找「氣」→ 換成「器」
結果：今天天器很好

原始：我很高興
指令：把高興的興改成欣賞的欣
分析：找「興」→ 換成「欣」
結果：我很高欣

原始：這首擱淺的歌
指令：把擱淺的歌刪除
分析：刪除「擱」
結果：這首淺的歌

原始：天氣好
指令：在好前面加很
結果：天氣很好

## 現在請處理：
原始文字：{original}
語音指令：{command}

只輸出修改後的文字（不要分析過程）："""

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the Gemini processor.

        Args:
            api_key: Gemini API key (default: from GEMINI_API_KEY env var)
            model: Model name (default: gemini-2.5-flash)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass api_key parameter.")
        self.model = model or self.MODEL
        self._client = None

        # Dummy labeler for compatibility with main.py preload
        self.labeler = _DummyLabeler()

    @property
    def client(self):
        """Lazy-load the Gemini client."""
        if self._client is None:
            if not GENAI_AVAILABLE:
                raise RuntimeError("google-genai not installed. Run: pip install google-genai")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_command(self, text: str) -> bool:
        """Check if text is a correction command."""
        text = text.strip()
        for patterns in self.COMMAND_PATTERNS.values():
            for pattern in patterns:
                if pattern.match(text):
                    return True
        return False

    def _get_command_type(self, text: str) -> CommandType:
        """Get the type of command."""
        text = text.strip()
        for cmd_type, patterns in self.COMMAND_PATTERNS.items():
            for pattern in patterns:
                if pattern.match(text):
                    return cmd_type
        return CommandType.NONE

    def _build_prompt(self, original_text: str, command: str) -> str:
        """
        Build the prompt for Gemini.

        Args:
            original_text: The text to modify
            command: The correction command

        Returns:
            Formatted prompt string
        """
        return self.PROMPT_TEMPLATE.format(
            original=original_text,
            command=command
        )

    def process(self, spoken_text: str, last_typed_text: str) -> Tuple[str, bool]:
        """
        Process spoken text and apply corrections using Gemini API.

        Args:
            spoken_text: The command that was spoken
            last_typed_text: The text to apply the correction to

        Returns:
            Tuple of (result_text, was_command)
        """
        # First check if it's a command
        if not self.is_command(spoken_text):
            return spoken_text, False

        if not last_typed_text:
            return spoken_text, False

        print(f"  [API] Detected command: '{spoken_text}'")
        print(f"  [API] Original text: '{last_typed_text}'")

        try:
            # Build prompt and call Gemini
            prompt = self._build_prompt(last_typed_text, spoken_text)
            print(f"  [API] Sending to Gemini...")

            # Simple single-shot call
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            result = response.text.strip()

            # Clean up result - remove any markdown or extra formatting
            result = self._clean_response(result)

            print(f"  [API] Gemini response: '{result}'")

            # Validate result - should be similar length to original (sanity check)
            if len(result) > len(last_typed_text) * 3 or len(result) == 0:
                print(f"  [API] Response seems invalid, falling back to original")
                return last_typed_text, True

            return result, True

        except Exception as e:
            print(f"  [API] Error calling Gemini: {e}")
            # Fall back to original text on error
            return last_typed_text, False

    def _clean_response(self, response: str) -> str:
        """Clean up Gemini response - remove markdown, quotes, etc."""
        # Remove markdown code blocks
        response = re.sub(r'^```.*\n?', '', response)
        response = re.sub(r'\n?```$', '', response)

        # Remove surrounding quotes
        response = response.strip('"\'""''')

        # Remove common prefixes
        prefixes = ['修改後：', '結果：', '答案：', '輸出：']
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]

        return response.strip()


class _DummyLabeler:
    """Dummy labeler for compatibility - does nothing."""

    def load(self):
        """No-op load for compatibility with main.py preloading."""
        print("  [API] Gemini API mode - no ML model to load")
