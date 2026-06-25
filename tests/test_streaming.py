"""Tests for streaming (SSE) record and replay support."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import agent_tape
from agent_tape._tape import (
    Tape,
    TapeInteraction,
    TapeRequest,
    TapeResponse,
    _tool_calls_from_sse,
    _sse_stop_reason,
)
from agent_tape._assertions import TapeAssertions


# ── SSE fixture helpers ───────────────────────────────────────────────────────


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _anthropic_stream_chunks(
    tool_name: str = "read_file", tool_input: dict | None = None
) -> list[str]:
    """Simulate the SSE chunks the Anthropic API sends for a tool_use stream."""
    tool_input = tool_input or {"path": "document.txt"}
    return [
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_001",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "usage": {"input_tokens": 100, "output_tokens": 0},
                },
            },
        ),
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_001",
                    "name": tool_name,
                    "input": {},
                },
            },
        ),
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_input)},
            },
        ),
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 45},
            },
        ),
        "data: [DONE]\n\n",
    ]


def _anthropic_end_turn_chunks() -> list[str]:
    return [
        _sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_002",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "usage": {"input_tokens": 200, "output_tokens": 0},
                },
            },
        ),
        _sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "All done."},
            },
        ),
        _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 12},
            },
        ),
        "data: [DONE]\n\n",
    ]


def _streaming_tape(chunks_per_call: list[list[str]]) -> Tape:
    tape = Tape()
    for i, chunks in enumerate(chunks_per_call):
        tape.interactions.append(
            TapeInteraction(
                id=f"call_{i}",
                request=TapeRequest("POST", "https://api.anthropic.com/v1/messages", {}, {}),
                response=TapeResponse(
                    200, {"content-type": "text/event-stream"}, body=None, chunks=chunks
                ),
                duration_ms=300.0,
            )
        )
    return tape


# ── Tape model tests ──────────────────────────────────────────────────────────


def test_is_streaming_flag() -> None:
    tape = _streaming_tape([_anthropic_stream_chunks()])
    assert tape.interactions[0].is_streaming
    assert not Tape().interactions  # empty tape has no streaming interactions


def test_streaming_tape_save_load(tmp_path: Path) -> None:
    chunks = _anthropic_stream_chunks("read_file")
    tape = _streaming_tape([chunks])
    path = tmp_path / "stream.tape.yaml"
    tape.save(path)

    content = path.read_text()
    assert "streaming: true" in content
    assert "read_file" in content

    loaded = Tape.load(path)
    assert loaded.interactions[0].is_streaming
    assert loaded.interactions[0].response.chunks == chunks


def test_tool_calls_from_sse() -> None:
    chunks = _anthropic_stream_chunks("search_web", {"query": "agent testing"})
    calls = _tool_calls_from_sse(chunks, "call_0")
    assert len(calls) == 1
    assert calls[0]["name"] == "search_web"
    assert calls[0]["input"]["query"] == "agent testing"


def test_sse_stop_reason_tool_use() -> None:
    chunks = _anthropic_stream_chunks()
    assert _sse_stop_reason(chunks) == "tool_use"


def test_sse_stop_reason_end_turn() -> None:
    chunks = _anthropic_end_turn_chunks()
    assert _sse_stop_reason(chunks) == "end_turn"


# ── Assertions over streaming tapes ──────────────────────────────────────────


def test_streaming_assert_tool_called() -> None:
    tape = _streaming_tape(
        [
            _anthropic_stream_chunks("read_file"),
            _anthropic_stream_chunks("write_file"),
            _anthropic_end_turn_chunks(),
        ]
    )
    ta = TapeAssertions(tape)
    ta.assert_tool_called("read_file")
    ta.assert_tool_called("write_file")
    ta.assert_tool_called("read_file", before="write_file")


def test_streaming_assert_task_completed() -> None:
    tape = _streaming_tape([_anthropic_end_turn_chunks()])
    ta = TapeAssertions(tape)
    ta.assert_task_completed()


def test_streaming_assert_task_not_completed() -> None:
    tape = _streaming_tape([_anthropic_stream_chunks()])
    ta = TapeAssertions(tape)
    with pytest.raises(AssertionError, match="tool_use"):
        ta.assert_task_completed()


def test_streaming_assert_no_hallucinated_tools() -> None:
    tape = _streaming_tape(
        [
            _anthropic_stream_chunks("read_file"),
            _anthropic_end_turn_chunks(),
        ]
    )
    ta = TapeAssertions(tape)
    ta.assert_no_hallucinated_tools(["read_file", "write_file"])
    with pytest.raises(AssertionError, match="read_file"):
        ta.assert_no_hallucinated_tools(["write_file"])


# ── Replay streaming interactions ─────────────────────────────────────────────


async def test_replay_streaming_response(tmp_path: Path) -> None:
    """Replayed streaming response should be re-playable as real SSE chunks."""
    chunks = _anthropic_stream_chunks("read_file")
    tape = _streaming_tape([chunks, _anthropic_end_turn_chunks()])
    tape_path = tmp_path / "stream.tape.yaml"
    tape.save(tape_path)

    received: list[bytes] = []

    with agent_tape.replay(tape_path):
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"content-type": "application/json"},
                json={"model": "claude-sonnet-4-6", "stream": True},
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")
                async for chunk in response.aiter_bytes():
                    received.append(chunk)

    full = b"".join(received).decode()
    assert "read_file" in full
    assert "tool_use" in full


def test_replay_streaming_sync(tmp_path: Path) -> None:
    """Sync streaming replay also works."""
    chunks = _anthropic_end_turn_chunks()
    tape = _streaming_tape([chunks])
    tape_path = tmp_path / "sync_stream.tape.yaml"
    tape.save(tape_path)

    received: list[bytes] = []

    with agent_tape.replay(tape_path):
        with httpx.Client() as client:
            with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={"content-type": "application/json"},
                json={"stream": True},
            ) as response:
                for chunk in response.iter_bytes():
                    received.append(chunk)

    full = b"".join(received).decode()
    assert "end_turn" in full


# ── Summary includes streaming count ─────────────────────────────────────────


def test_summary_includes_streaming() -> None:
    tape = _streaming_tape([_anthropic_stream_chunks(), _anthropic_end_turn_chunks()])
    s = tape.summary()
    assert "streaming" in s
