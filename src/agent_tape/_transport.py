from __future__ import annotations

import json
import time
from typing import Any, Iterator, AsyncIterator

import httpx

from ._tape import Tape, TapeInteraction, TapeRequest, TapeResponse

_SCRUB = frozenset({"authorization", "x-api-key", "cookie", "set-cookie"})


def _scrub(headers: dict[str, str]) -> dict[str, str]:
    return {k: ("***" if k.lower() in _SCRUB else v) for k, v in headers.items()}


def _is_sse(headers: dict[str, str]) -> bool:
    return "text/event-stream" in headers.get("content-type", "")


def _parse(content: bytes, headers: dict[str, str]) -> Any:
    if not content:
        return None
    ct = headers.get("content-type", "")
    if "json" in ct or not ct:
        try:
            return json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    return content.decode("utf-8", errors="replace")


def _serialize(body: Any) -> bytes:
    if body is None:
        return b""
    if isinstance(body, (dict, list)):
        return json.dumps(body).encode()
    return str(body).encode()


# ── Sync byte streams for httpx ───────────────────────────────────────────────


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            yield chunk.encode("utf-8")


class _AsyncChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk.encode("utf-8")


# ── Recording ─────────────────────────────────────────────────────────────────


class RecordingTransport(httpx.BaseTransport):
    """Wraps a real transport and writes every interaction to a Tape."""

    def __init__(self, tape: Tape, wrapped: httpx.BaseTransport | None = None) -> None:
        self._tape = tape
        self._wrapped = wrapped or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        body_bytes = request.read()
        req_headers = _scrub(dict(request.headers))
        start = time.monotonic()
        response = self._wrapped.handle_request(request)
        duration_ms = (time.monotonic() - start) * 1000
        resp_headers = _scrub(dict(response.headers))

        if _is_sse(resp_headers):
            chunks: list[str] = []
            for raw in response.iter_bytes():
                chunks.append(raw.decode("utf-8", errors="replace"))
            self._tape.interactions.append(
                TapeInteraction(
                    id=f"call_{len(self._tape.interactions)}",
                    request=TapeRequest(
                        request.method,
                        str(request.url),
                        req_headers,
                        _parse(body_bytes, req_headers),
                    ),
                    response=TapeResponse(
                        response.status_code, resp_headers, body=None, chunks=chunks
                    ),
                    duration_ms=duration_ms,
                )
            )
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_ChunkStream(chunks),
                request=request,
            )

        resp_content = response.read()
        self._tape.interactions.append(
            TapeInteraction(
                id=f"call_{len(self._tape.interactions)}",
                request=TapeRequest(
                    request.method, str(request.url), req_headers, _parse(body_bytes, req_headers)
                ),
                response=TapeResponse(
                    response.status_code, resp_headers, body=_parse(resp_content, resp_headers)
                ),
                duration_ms=duration_ms,
            )
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=resp_content,
            request=request,
        )

    def close(self) -> None:
        self._wrapped.close()


class RecordingAsyncTransport(httpx.AsyncBaseTransport):
    """Async version of RecordingTransport."""

    def __init__(self, tape: Tape, wrapped: httpx.AsyncBaseTransport | None = None) -> None:
        self._tape = tape
        self._wrapped = wrapped or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body_bytes = await request.aread()
        req_headers = _scrub(dict(request.headers))
        start = time.monotonic()
        response = await self._wrapped.handle_async_request(request)
        duration_ms = (time.monotonic() - start) * 1000
        resp_headers = _scrub(dict(response.headers))

        if _is_sse(resp_headers):
            chunks: list[str] = []
            async for raw in response.aiter_bytes():
                chunks.append(raw.decode("utf-8", errors="replace"))
            self._tape.interactions.append(
                TapeInteraction(
                    id=f"call_{len(self._tape.interactions)}",
                    request=TapeRequest(
                        request.method,
                        str(request.url),
                        req_headers,
                        _parse(body_bytes, req_headers),
                    ),
                    response=TapeResponse(
                        response.status_code, resp_headers, body=None, chunks=chunks
                    ),
                    duration_ms=duration_ms,
                )
            )
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=_AsyncChunkStream(chunks),
                request=request,
            )

        resp_content = await response.aread()
        self._tape.interactions.append(
            TapeInteraction(
                id=f"call_{len(self._tape.interactions)}",
                request=TapeRequest(
                    request.method, str(request.url), req_headers, _parse(body_bytes, req_headers)
                ),
                response=TapeResponse(
                    response.status_code, resp_headers, body=_parse(resp_content, resp_headers)
                ),
                duration_ms=duration_ms,
            )
        )
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=resp_content,
            request=request,
        )

    async def aclose(self) -> None:
        await self._wrapped.aclose()


# ── Replaying ─────────────────────────────────────────────────────────────────


class ReplayingTransport(httpx.BaseTransport):
    """Returns pre-recorded responses from a Tape in sequence."""

    def __init__(self, tape: Tape) -> None:
        self._tape = tape
        self._index = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._index >= len(self._tape.interactions):
            raise RuntimeError(
                f"Tape exhausted after {self._index} interactions. "
                "The agent made more LLM calls than were recorded."
            )
        interaction = self._tape.interactions[self._index]
        self._index += 1

        if interaction.is_streaming:
            return httpx.Response(
                status_code=interaction.response.status,
                headers=interaction.response.headers,
                stream=_ChunkStream(interaction.response.chunks or []),
                request=request,
            )
        return httpx.Response(
            status_code=interaction.response.status,
            headers=interaction.response.headers,
            content=_serialize(interaction.response.body),
            request=request,
        )


class ReplayingAsyncTransport(httpx.AsyncBaseTransport):
    """Async version of ReplayingTransport."""

    def __init__(self, tape: Tape) -> None:
        self._tape = tape
        self._index = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._index >= len(self._tape.interactions):
            raise RuntimeError(
                f"Tape exhausted after {self._index} interactions. "
                "The agent made more LLM calls than were recorded."
            )
        interaction = self._tape.interactions[self._index]
        self._index += 1

        if interaction.is_streaming:
            return httpx.Response(
                status_code=interaction.response.status,
                headers=interaction.response.headers,
                stream=_AsyncChunkStream(interaction.response.chunks or []),
                request=request,
            )
        return httpx.Response(
            status_code=interaction.response.status,
            headers=interaction.response.headers,
            content=_serialize(interaction.response.body),
            request=request,
        )
