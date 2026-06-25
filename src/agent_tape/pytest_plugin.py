"""
pytest plugin — registered automatically via entry_points["pytest11"].

Provides:
  - @pytest.mark.tape("path/to/tape.yaml")   → fixture replays the tape
  - tape_replay fixture                        → TapeAssertions instance
  - tape_record fixture                        → Tape instance (records to tmp file)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._assertions import TapeAssertions
from ._patch import record, replay
from ._tape import Tape


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "tape(path): replay agent interactions from a recorded tape file",
    )
    config.addinivalue_line(
        "markers",
        "record(path): record agent interactions to a tape file",
    )


@pytest.fixture
def tape_replay(request: pytest.FixtureRequest) -> TapeAssertions:
    """
    Replay interactions from a tape file and expose behavioral assertions.

    Mark your test with ``@pytest.mark.tape("tapes/my_run.tape.yaml")``.
    """
    marker = request.node.get_closest_marker("tape")
    if marker is None:
        pytest.skip("No @pytest.mark.tape marker — skipping replay fixture")
    path = Path(marker.args[0])
    if not path.exists():
        pytest.skip(f"Tape file not found: {path} — run in record mode first")
    ctx = replay(path)
    assertions = ctx.__enter__()
    yield assertions
    ctx.__exit__(None, None, None)


@pytest.fixture
def tape_record(request: pytest.FixtureRequest, tmp_path: Path) -> Tape:
    """
    Record all httpx interactions during the test.

    Mark your test with ``@pytest.mark.record("tapes/my_run.tape.yaml")``
    to choose the output path; defaults to a temp file.
    """
    marker = request.node.get_closest_marker("record")
    path = Path(marker.args[0]) if marker else tmp_path / "tape.yaml"
    ctx = record(path)
    tape = ctx.__enter__()
    yield tape
    ctx.__exit__(None, None, None)
