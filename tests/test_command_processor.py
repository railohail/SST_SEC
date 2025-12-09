"""Tests for Command Processor"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.command_processor import CommandProcessor, CommandType, ParsedCommand


def test_command_detection():
    """Test command detection patterns."""
    processor = CommandProcessor()

    # Should be detected as commands
    commands = [
        "刪除錯字",
        "把錯改成對",
        "在好前面新增很",
        "請在天後面新增氣",
        "把氣器改成氣候的氣",
    ]

    for cmd in commands:
        assert processor.is_command(cmd), f"'{cmd}' should be detected as command"

    # Should NOT be detected as commands
    non_commands = [
        "今天天氣很好",
        "你好",
        "這是一段普通的文字",
        "",
    ]

    for text in non_commands:
        assert not processor.is_command(text), f"'{text}' should NOT be detected as command"

    print("✅ Command detection tests passed!")


def test_command_parsing():
    """Test command parsing."""
    processor = CommandProcessor()

    # Test DELETE
    parsed = processor.parse_command("刪除錯字")
    assert parsed.type == CommandType.DELETE
    assert parsed.target == "錯字"

    # Test REPLACE
    parsed = processor.parse_command("把錯改成對")
    assert parsed.type == CommandType.REPLACE
    assert parsed.target == "錯"
    assert parsed.replacement == "對"

    # Test INSERT_BEFORE
    parsed = processor.parse_command("在好前面新增很")
    assert parsed.type == CommandType.INSERT_BEFORE
    assert parsed.target == "好"
    assert parsed.replacement == "很"

    # Test INSERT_BEFORE with 請
    parsed = processor.parse_command("請在好前面新增很")
    assert parsed.type == CommandType.INSERT_BEFORE
    assert parsed.target == "好"
    assert parsed.replacement == "很"

    # Test INSERT_AFTER
    parsed = processor.parse_command("在天後面新增氣")
    assert parsed.type == CommandType.INSERT_AFTER
    assert parsed.target == "天"
    assert parsed.replacement == "氣"

    # Test non-command
    parsed = processor.parse_command("今天天氣很好")
    assert parsed.type == CommandType.NONE

    print("✅ Command parsing tests passed!")


def test_correction_by_target():
    """Test correction application by target matching."""
    processor = CommandProcessor()

    # Test DELETE
    text = "今天天氣很好"
    cmd = ParsedCommand(type=CommandType.DELETE, target="很")
    result = processor._apply_by_target(text, cmd)
    assert result == "今天天氣好"

    # Test REPLACE
    cmd = ParsedCommand(type=CommandType.REPLACE, target="氣", replacement="器")
    result = processor._apply_by_target(text, cmd)
    assert result == "今天天器很好"

    # Test INSERT_BEFORE
    cmd = ParsedCommand(type=CommandType.INSERT_BEFORE, target="好", replacement="非常")
    result = processor._apply_by_target(text, cmd)
    assert result == "今天天氣很非常好"

    # Test INSERT_AFTER
    cmd = ParsedCommand(type=CommandType.INSERT_AFTER, target="天氣", replacement="真的")
    result = processor._apply_by_target(text, cmd)
    assert result == "今天天氣真的很好"

    print("✅ Correction by target tests passed!")


def test_correction_at_position():
    """Test correction application at specific position."""
    processor = CommandProcessor()
    text = "今天天氣很好"

    # Test DELETE at position 4 (很)
    cmd = ParsedCommand(type=CommandType.DELETE)
    result = processor._apply_at_position(text, cmd, 4)
    assert result == "今天天氣好"

    # Test REPLACE at position 3 (氣)
    cmd = ParsedCommand(type=CommandType.REPLACE, replacement="器")
    result = processor._apply_at_position(text, cmd, 3)
    assert result == "今天天器很好"

    # Test INSERT_BEFORE at position 4
    cmd = ParsedCommand(type=CommandType.INSERT_BEFORE, replacement="真的")
    result = processor._apply_at_position(text, cmd, 4)
    assert result == "今天天氣真的很好"

    # Test INSERT_AFTER at position 3
    cmd = ParsedCommand(type=CommandType.INSERT_AFTER, replacement="候")
    result = processor._apply_at_position(text, cmd, 3)
    assert result == "今天天氣候很好"

    print("✅ Correction at position tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Command Processor Tests")
    print("=" * 50)

    test_command_detection()
    test_command_parsing()
    test_correction_by_target()
    test_correction_at_position()

    print("=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
