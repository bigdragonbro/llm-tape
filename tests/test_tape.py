"""Tests for Tape serialization and round-trip integrity."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_tape._tape import Tape, TapeInteraction, TapeRequest, TapeResponse


def _make_tape() -> Tape:
    return Tape(
        version="1",
        metadata={"recorded_at": "2026-06-24"},
        interactions=[
            TapeInteraction(
                id="call_0",
                request=TapeRequest(
                    method="POST",
                    url="https://api.anthropic.com/v1/messages",
                    headers={"content-type": "application/json"},
                    body={"model": "claude-sonnet-4-6", "max_tokens": 1024},
                ),
                response=TapeResponse(
                    status=200,
                    headers={"content-type": "application/json"},
                    body={
                        "id": "msg_001",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "Hello"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                ),
                duration_ms=123.4,
                timestamp=1750000000.0,
            )
        ],
    )


def test_save_and_load(tmp_path: Path) -> None:
    tape = _make_tape()
    path = tmp_path / "test.tape.yaml"
    tape.save(path)

    assert path.exists()
    loaded = Tape.load(path)

    assert loaded.version == "1"
    assert len(loaded.interactions) == 1

    i = loaded.interactions[0]
    assert i.id == "call_0"
    assert i.request.method == "POST"
    assert i.request.body["model"] == "claude-sonnet-4-6"
    assert i.response.status == 200
    assert i.response.body["stop_reason"] == "end_turn"
    assert i.duration_ms == pytest.approx(123.4, abs=0.01)


def test_yaml_is_human_readable(tmp_path: Path) -> None:
    tape = _make_tape()
    path = tmp_path / "test.tape.yaml"
    tape.save(path)

    content = path.read_text()
    # Body fields should appear as plain YAML, not as escaped JSON strings
    assert "claude-sonnet-4-6" in content
    assert "end_turn" in content
    assert "call_0" in content


def test_summary(tmp_path: Path) -> None:
    tape = _make_tape()
    s = tape.summary()
    assert "1 LLM call" in s
    assert "15 tokens" in s  # 10 + 5


def test_load_nonexistent_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Tape.load(tmp_path / "missing.tape.yaml")


def test_empty_tape(tmp_path: Path) -> None:
    tape = Tape()
    path = tmp_path / "empty.tape.yaml"
    tape.save(path)
    loaded = Tape.load(path)
    assert loaded.interactions == []
