"""Shared fixtures and fake agent responses for the test suite."""

from __future__ import annotations

from typing import Any

import httpx


# ── Fake Anthropic API responses ──────────────────────────────────────────────


def anthropic_tool_use_response(tool_name: str, tool_input: dict[str, Any]) -> dict:
    return {
        "id": "msg_fake_001",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_{tool_name}",
                "name": tool_name,
                "input": tool_input,
            }
        ],
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 120, "output_tokens": 40},
    }


def anthropic_end_turn_response(text: str = "Done.") -> dict:
    return {
        "id": "msg_fake_002",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 30},
    }


# ── Fake agent ────────────────────────────────────────────────────────────────


async def fake_agent_async(responses: list[dict]) -> list[dict]:
    """
    Minimal async "agent" that makes sequential httpx calls to the Anthropic
    messages endpoint. Each call returns the next pre-baked response.

    Used to test record/replay without real API credentials.
    """
    results = []
    for resp_body in responses:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": "test-key", "content-type": "application/json"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "do something"}],
                },
            )
            results.append(r.json())
    return results


def fake_agent_sync(responses: list[dict]) -> list[dict]:
    """Synchronous version of fake_agent_async."""
    results = []
    for resp_body in responses:
        with httpx.Client() as client:
            r = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": "test-key", "content-type": "application/json"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": "do something"}],
                },
            )
            results.append(r.json())
    return results
