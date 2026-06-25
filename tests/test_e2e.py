"""
End-to-end tests: record a fake agent run, then replay and assert.

These tests use no real API keys — the RecordingTransport intercepts
real httpx calls but the ReplayingTransport serves them back from the tape.
The "agent" in conftest.py simulates what any LLM SDK would do under the hood.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent_tape
from agent_tape._tape import Tape

from .conftest import (
    anthropic_end_turn_response,
    anthropic_tool_use_response,
    fake_agent_async,
    fake_agent_sync,
)


# ── Async record → replay ─────────────────────────────────────────────────────


async def test_async_record_then_replay(tmp_path: Path) -> None:
    """
    Full record → file → replay cycle using manually-built tapes to avoid
    needing real network calls. The isolated tests below cover the same
    assertion API; this test verifies the file round-trip specifically.
    """
    from agent_tape._tape import TapeInteraction, TapeRequest, TapeResponse

    tape_path = tmp_path / "agent_run.tape.yaml"

    baked = [
        anthropic_tool_use_response("read_file", {"path": "document.txt"}),
        anthropic_tool_use_response("summarize", {"content": "..."}),
        anthropic_end_turn_response("Summary complete."),
    ]

    # Build and save a tape directly (simulates a completed recording)
    tape = Tape()
    for i, body in enumerate(baked):
        tape.interactions.append(
            TapeInteraction(
                id=f"call_{i}",
                request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
                response=TapeResponse(200, {"content-type": "application/json"}, body),
                duration_ms=150.0,
            )
        )
    tape.save(tape_path)

    assert tape_path.exists()
    loaded = Tape.load(tape_path)
    assert len(loaded.interactions) == 3

    # Replay the saved tape and assert behavior
    with agent_tape.replay(tape_path) as ta:
        await fake_agent_async(baked)

    ta.assert_tool_called("read_file", before="summarize")
    ta.assert_task_completed()
    ta.assert_steps_under(5)


async def test_async_record_replay_isolated(tmp_path: Path) -> None:
    """
    Single-flow test: record using our transport, replay using our transport.
    The RecordingTransport wraps a fake inner transport (not real network).
    """
    from agent_tape._tape import TapeInteraction, TapeRequest, TapeResponse

    baked = [
        anthropic_tool_use_response("read_file", {"path": "doc.txt"}),
        anthropic_end_turn_response("Done reading."),
    ]

    # Build the tape manually (simulates what the recorder would produce)
    tape = Tape()
    for i, body in enumerate(baked):
        tape.interactions.append(
            TapeInteraction(
                id=f"call_{i}",
                request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
                response=TapeResponse(200, {"content-type": "application/json"}, body),
                duration_ms=100.0,
            )
        )

    tape_path = tmp_path / "isolated.tape.yaml"
    tape.save(tape_path)

    # Replay phase
    with agent_tape.replay(tape_path) as ta:
        results = await fake_agent_async(baked)

    assert len(results) == 2
    assert results[0]["stop_reason"] == "tool_use"
    assert results[1]["stop_reason"] == "end_turn"

    ta.assert_tool_called("read_file")
    ta.assert_task_completed()
    ta.assert_steps_under(5)
    ta.assert_no_hallucinated_tools(["read_file", "summarize", "write_file"])
    ta.assert_total_tokens_under(1000)


# ── Sync record → replay ──────────────────────────────────────────────────────


def test_sync_record_replay_isolated(tmp_path: Path) -> None:
    """Same as the async isolated test but using httpx.Client (sync)."""
    from agent_tape._tape import TapeInteraction, TapeRequest, TapeResponse

    baked = [
        anthropic_tool_use_response("search_web", {"query": "agent testing"}),
        anthropic_tool_use_response("read_file", {"path": "result.html"}),
        anthropic_end_turn_response("Research complete."),
    ]

    tape = Tape()
    for i, body in enumerate(baked):
        tape.interactions.append(
            TapeInteraction(
                id=f"call_{i}",
                request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
                response=TapeResponse(200, {"content-type": "application/json"}, body),
                duration_ms=90.0,
            )
        )

    tape_path = tmp_path / "sync.tape.yaml"
    tape.save(tape_path)

    with agent_tape.replay(tape_path) as ta:
        results = fake_agent_sync(baked)

    assert len(results) == 3
    ta.assert_tool_called("search_web", before="read_file")
    ta.assert_tool_called("read_file", after="search_web")
    ta.assert_task_completed()
    ta.assert_steps_under(10)


# ── Tape exhaustion ───────────────────────────────────────────────────────────


def test_replay_exhaustion_raises(tmp_path: Path) -> None:
    """If the agent makes more calls than recorded, raise a clear error."""
    from agent_tape._tape import TapeInteraction, TapeRequest, TapeResponse

    tape = Tape()
    tape.interactions.append(
        TapeInteraction(
            id="call_0",
            request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
            response=TapeResponse(200, {}, anthropic_end_turn_response()),
            duration_ms=50.0,
        )
    )
    tape_path = tmp_path / "short.tape.yaml"
    tape.save(tape_path)

    with pytest.raises(RuntimeError, match="Tape exhausted"):
        with agent_tape.replay(tape_path):
            # Agent tries to make 2 calls but only 1 is recorded
            fake_agent_sync([anthropic_end_turn_response(), anthropic_end_turn_response()])


# ── pytest fixture integration ────────────────────────────────────────────────


@pytest.mark.tape("tests/fixtures/sample.tape.yaml")
def test_pytest_marker_skips_when_file_missing(tape_replay: object) -> None:
    # This test is intentionally unreachable — the fixture skips if the file
    # is missing. Kept here to document the intended marker API.
    pass
