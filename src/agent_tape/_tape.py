from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TapeRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: Any  # Parsed JSON dict/list, plain string, or None


@dataclass
class TapeResponse:
    status: int
    headers: dict[str, str]
    body: Any  # Non-streaming: parsed JSON or string
    chunks: list[str] | None = None  # Streaming (SSE): raw event lines


@dataclass
class TapeInteraction:
    id: str
    request: TapeRequest
    response: TapeResponse
    duration_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_streaming(self) -> bool:
        return self.response.chunks is not None


@dataclass
class Tape:
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)
    interactions: list[TapeInteraction] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "metadata": self.metadata,
            "interactions": [_to_dict(i) for i in self.interactions],
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> Tape:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            version=data.get("version", "1"),
            metadata=data.get("metadata", {}),
            interactions=[_from_dict(i) for i in data.get("interactions", [])],
        )

    def summary(self) -> str:
        tool_calls = _extract_tool_calls(self)
        total_tokens = sum(
            (i.response.body or {}).get("usage", {}).get("input_tokens", 0)
            + (i.response.body or {}).get("usage", {}).get("output_tokens", 0)
            for i in self.interactions
            if isinstance(i.response.body, dict)
        )
        streaming_count = sum(1 for i in self.interactions if i.is_streaming)
        parts = [
            f"{len(self.interactions)} LLM call(s)",
            f"{len(tool_calls)} tool call(s)",
            f"~{total_tokens} tokens",
        ]
        if streaming_count:
            parts.append(f"{streaming_count} streaming")
        return ", ".join(parts)


def _extract_tool_calls(tape: Tape) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for interaction in tape.interactions:
        # Streaming: parse tool_use blocks from SSE event lines
        if interaction.is_streaming and interaction.response.chunks:
            calls.extend(_tool_calls_from_sse(interaction.response.chunks, interaction.id))
            continue

        body = interaction.response.body
        if not isinstance(body, dict):
            continue
        # Anthropic non-streaming format
        for block in body.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(
                    {
                        "name": block["name"],
                        "input": block.get("input", {}),
                        "interaction_id": interaction.id,
                    }
                )
        # OpenAI chat completion format
        for choice in body.get("choices", []):
            msg = choice.get("message", {})
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = raw
                calls.append(
                    {
                        "name": fn.get("name"),
                        "input": args,
                        "interaction_id": interaction.id,
                    }
                )
    return calls


def _tool_calls_from_sse(chunks: list[str], interaction_id: str) -> list[dict[str, Any]]:
    """Parse Anthropic SSE stream chunks to extract tool_use blocks."""
    calls: list[dict[str, Any]] = []
    current_tool: dict[str, Any] | None = None
    input_buf: str = ""

    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool = {"name": block.get("name", ""), "input": {}}
                    input_buf = ""

            elif etype == "content_block_delta" and current_tool is not None:
                delta = event.get("delta", {})
                if delta.get("type") == "input_json_delta":
                    input_buf += delta.get("partial_json", "")

            elif etype == "content_block_stop" and current_tool is not None:
                try:
                    current_tool["input"] = json.loads(input_buf) if input_buf else {}
                except json.JSONDecodeError:
                    current_tool["input"] = input_buf
                current_tool["interaction_id"] = interaction_id
                calls.append(current_tool)
                current_tool = None
                input_buf = ""

            # Anthropic final message event carries stop_reason
            # (used by assert_stop_reason via body — we don't need to extract it here)

    return calls


def _sse_stop_reason(chunks: list[str]) -> str | None:
    """Extract stop_reason from the Anthropic message_delta SSE event."""
    for chunk in reversed(chunks):
        for line in reversed(chunk.splitlines()):
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "message_delta":
                return event.get("delta", {}).get("stop_reason")
    return None


def _to_dict(i: TapeInteraction) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "status": i.response.status,
        "headers": i.response.headers,
    }
    if i.is_streaming:
        resp["streaming"] = True
        resp["chunks"] = i.response.chunks
    else:
        resp["body"] = i.response.body

    return {
        "id": i.id,
        "timestamp": round(i.timestamp, 3),
        "duration_ms": round(i.duration_ms, 2),
        "request": {
            "method": i.request.method,
            "url": i.request.url,
            "headers": i.request.headers,
            "body": i.request.body,
        },
        "response": resp,
    }


def _from_dict(d: dict[str, Any]) -> TapeInteraction:
    req = d["request"]
    resp = d["response"]
    return TapeInteraction(
        id=d["id"],
        timestamp=d.get("timestamp", 0.0),
        duration_ms=d.get("duration_ms", 0.0),
        request=TapeRequest(
            method=req["method"],
            url=req["url"],
            headers=req.get("headers", {}),
            body=req.get("body"),
        ),
        response=TapeResponse(
            status=resp["status"],
            headers=resp.get("headers", {}),
            body=resp.get("body"),
            chunks=resp.get("chunks"),
        ),
    )
