"""Tests for TapeAssertions behavioral checks."""

from __future__ import annotations

import pytest

from agent_tape._assertions import TapeAssertions
from agent_tape._tape import Tape, TapeInteraction, TapeRequest, TapeResponse


def _tape_with_calls(*tool_names: str, stop_reason: str = "end_turn") -> Tape:
    """Build a minimal Tape with the given sequence of tool calls."""
    interactions: list[TapeInteraction] = []

    for i, name in enumerate(tool_names):
        interactions.append(
            TapeInteraction(
                id=f"call_{i}",
                request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
                response=TapeResponse(
                    status=200,
                    headers={},
                    body={
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": f"toolu_{i}",
                                "name": name,
                                "input": {"file": f"{name}.txt"},
                            }
                        ],
                        "usage": {"input_tokens": 50, "output_tokens": 20},
                    },
                ),
                duration_ms=100.0,
            )
        )

    # Final call with the given stop_reason and no tool use
    interactions.append(
        TapeInteraction(
            id=f"call_{len(tool_names)}",
            request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
            response=TapeResponse(
                status=200,
                headers={},
                body={
                    "stop_reason": stop_reason,
                    "content": [{"type": "text", "text": "All done."}],
                    "usage": {"input_tokens": 100, "output_tokens": 30},
                },
            ),
            duration_ms=80.0,
        )
    )

    return Tape(interactions=interactions)


# ── tool assertions ───────────────────────────────────────────────────────────


def test_assert_tool_called_passes() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file", "write_file"))
    ta.assert_tool_called("read_file")
    ta.assert_tool_called("write_file")


def test_assert_tool_called_fails_when_absent() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    with pytest.raises(AssertionError, match="write_file"):
        ta.assert_tool_called("write_file")


def test_assert_tool_called_before() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file", "write_file"))
    ta.assert_tool_called("read_file", before="write_file")


def test_assert_tool_called_before_fails_wrong_order() -> None:
    ta = TapeAssertions(_tape_with_calls("write_file", "read_file"))
    with pytest.raises(AssertionError, match="before"):
        ta.assert_tool_called("read_file", before="write_file")


def test_assert_tool_called_after() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file", "write_file"))
    ta.assert_tool_called("write_file", after="read_file")


def test_assert_tool_not_called() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    ta.assert_tool_not_called("delete_file")


def test_assert_tool_not_called_fails_when_present() -> None:
    ta = TapeAssertions(_tape_with_calls("delete_file"))
    with pytest.raises(AssertionError, match="delete_file"):
        ta.assert_tool_not_called("delete_file")


def test_assert_no_hallucinated_tools_passes() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file", "write_file"))
    ta.assert_no_hallucinated_tools(["read_file", "write_file", "search"])


def test_assert_no_hallucinated_tools_fails() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file", "rm_rf"))
    with pytest.raises(AssertionError, match="rm_rf"):
        ta.assert_no_hallucinated_tools(["read_file", "write_file"])


def test_assert_tool_input() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    ta.assert_tool_input("read_file", file="read_file.txt")


def test_assert_tool_input_wrong_value_fails() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    with pytest.raises(AssertionError, match="file="):
        ta.assert_tool_input("read_file", file="other.txt")


# ── step / token assertions ───────────────────────────────────────────────────


def test_assert_steps_under_passes() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file", "write_file"))
    ta.assert_steps_under(10)


def test_assert_steps_under_fails() -> None:
    ta = TapeAssertions(_tape_with_calls("a", "b", "c"))
    # 3 tool calls + 1 final = 4 interactions
    with pytest.raises(AssertionError, match="4"):
        ta.assert_steps_under(3)


def test_assert_total_tokens_under() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    # call_0: 50+20=70, call_1 (final): 100+30=130 → total 200
    ta.assert_total_tokens_under(500)


def test_assert_total_tokens_under_fails() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    with pytest.raises(AssertionError, match="200"):
        ta.assert_total_tokens_under(100)


# ── completion assertions ─────────────────────────────────────────────────────


def test_assert_task_completed() -> None:
    ta = TapeAssertions(_tape_with_calls("read_file"))
    ta.assert_task_completed()


def test_assert_task_completed_fails_on_max_tokens() -> None:
    ta = TapeAssertions(_tape_with_calls(stop_reason="max_tokens"))
    with pytest.raises(AssertionError, match="max_tokens"):
        ta.assert_task_completed()


# ── diff ──────────────────────────────────────────────────────────────────────


def test_diff_identical_tapes() -> None:
    tape = _tape_with_calls("read_file", "write_file")
    a = TapeAssertions(tape)
    b = TapeAssertions(tape)
    assert a.diff(b) == []


def test_diff_different_tool_sequence() -> None:
    a = TapeAssertions(_tape_with_calls("read_file"))
    b = TapeAssertions(_tape_with_calls("write_file"))
    diffs = a.diff(b)
    assert any("Tool call sequence" in d for d in diffs)
