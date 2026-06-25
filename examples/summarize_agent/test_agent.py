"""
Tests for the summarize agent — run entirely from the tape, no API key needed.

    pytest examples/summarize_agent/test_agent.py

These tests verify behavioral properties of the agent:
  - that it reads the file before writing the summary
  - that it completes the task (doesn't loop forever or hit token limits)
  - that it stays within a reasonable step and token budget
  - that it only calls tools it was given (no hallucinated tools)

If the agent's behavior changes in a way that breaks these assertions,
you'll know immediately in CI — before it reaches production.
"""
from pathlib import Path

import pytest

import agent_tape
from agent import run_agent

TAPE = Path(__file__).parent.parent.parent / "tapes" / "summarize_document.tape.yaml"
HERE = Path(__file__).parent


@pytest.fixture
def result(monkeypatch):
    """Run the agent against the tape and return (text_result, tape_assertions)."""
    # The Anthropic SDK validates an API key exists before sending requests.
    # In replay mode no real call is made, but we still need a non-empty value.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-replay-key")
    with agent_tape.replay(TAPE) as tape:
        text = run_agent("Please summarize the file notes.txt", workdir=HERE)
    return text, tape


def test_agent_reads_file_first(result):
    _, tape = result
    tape.assert_tool_called("read_file")


def test_agent_writes_summary(result):
    _, tape = result
    tape.assert_tool_called("write_summary")


def test_correct_tool_order(result):
    """Must read the file before writing the summary — not the other way around."""
    _, tape = result
    tape.assert_tool_called("read_file", before="write_summary")


def test_agent_completes_task(result):
    """Agent must finish with end_turn, not max_tokens or an error."""
    _, tape = result
    tape.assert_task_completed()


def test_agent_is_efficient(result):
    """Summarizing a short file shouldn't take more than 5 LLM calls."""
    _, tape = result
    tape.assert_steps_under(5)


def test_agent_stays_within_token_budget(result):
    _, tape = result
    tape.assert_total_tokens_under(5000)


def test_no_hallucinated_tools(result):
    """Agent must only call tools it was actually given."""
    _, tape = result
    tape.assert_no_hallucinated_tools(["read_file", "write_summary"])


def test_reads_the_right_file(result):
    """Agent should read notes.txt, not some other file."""
    _, tape = result
    tape.assert_tool_input("read_file", path="notes.txt")


def test_returns_nonempty_summary(result):
    text, _ = result
    assert text and len(text.strip()) > 20, "Expected a real summary, got empty or trivial output"
