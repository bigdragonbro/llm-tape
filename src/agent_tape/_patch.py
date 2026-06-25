"""
Context managers that transparently patch httpx so any code using
httpx.Client or httpx.AsyncClient is intercepted — no framework-specific
hooks required.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx

from ._assertions import TapeAssertions
from ._tape import Tape
from ._transport import (
    RecordingAsyncTransport,
    RecordingTransport,
    ReplayingAsyncTransport,
    ReplayingTransport,
)


@contextmanager
def record(path: str | Path):
    """
    Record all httpx calls made within this block to a tape file.

    Usage::

        with agent_tape.record("tapes/summarize.tape.yaml"):
            result = await my_agent.run("summarize document.txt")
    """
    tape = Tape()
    sync_tr = RecordingTransport(tape)
    async_tr = RecordingAsyncTransport(tape)

    orig_client = httpx.Client.__init__
    orig_async = httpx.AsyncClient.__init__

    def _patched_client(self: httpx.Client, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("transport", sync_tr)
        orig_client(self, *args, **kwargs)

    def _patched_async(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("transport", async_tr)
        orig_async(self, *args, **kwargs)

    with (
        patch.object(httpx.Client, "__init__", _patched_client),
        patch.object(httpx.AsyncClient, "__init__", _patched_async),
    ):
        yield tape

    tape.save(path)
    print(f"[agent-tape] Saved {path} — {tape.summary()}")


@contextmanager
def replay(path: str | Path):
    """
    Replay from a tape file, returning a TapeAssertions object.

    Usage::

        with agent_tape.replay("tapes/summarize.tape.yaml") as tape:
            result = await my_agent.run("summarize document.txt")
            tape.assert_tool_called("read_file")
            tape.assert_task_completed()
    """
    tape = Tape.load(path)
    sync_tr = ReplayingTransport(tape)
    async_tr = ReplayingAsyncTransport(tape)

    orig_client = httpx.Client.__init__
    orig_async = httpx.AsyncClient.__init__

    def _patched_client(self: httpx.Client, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("transport", sync_tr)
        orig_client(self, *args, **kwargs)

    def _patched_async(self: httpx.AsyncClient, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("transport", async_tr)
        orig_async(self, *args, **kwargs)

    with (
        patch.object(httpx.Client, "__init__", _patched_client),
        patch.object(httpx.AsyncClient, "__init__", _patched_async),
    ):
        yield TapeAssertions(tape)
