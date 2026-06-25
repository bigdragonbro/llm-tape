from __future__ import annotations

from ._tape import Tape, TapeInteraction, _extract_tool_calls


class TapeAssertions:
    """
    Behavioral assertions over a recorded or replayed Tape.

    Asserts structural properties of what the agent did — tool ordering,
    step counts, token budgets — not exact string output.
    """

    def __init__(self, tape: Tape) -> None:
        self._tape = tape
        self._tool_calls = _extract_tool_calls(tape)

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def interactions(self) -> list[TapeInteraction]:
        return self._tape.interactions

    @property
    def tool_calls(self) -> list[dict]:
        """All tool calls made during the run, in order."""
        return self._tool_calls

    @property
    def tool_names(self) -> list[str]:
        return [tc["name"] for tc in self._tool_calls]

    # ── Tool assertions ───────────────────────────────────────────────────────

    def assert_tool_called(
        self,
        tool_name: str,
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> None:
        """Assert that a tool was called, optionally with ordering constraints."""
        names = self.tool_names
        if tool_name not in names:
            raise AssertionError(
                f"Tool '{tool_name}' was never called.\n"
                f"Tools called (in order): {names or '(none)'}"
            )
        if before is not None:
            if before not in names:
                raise AssertionError(f"Cannot check ordering: '{before}' was never called")
            if names.index(tool_name) >= names.index(before):
                raise AssertionError(
                    f"Expected '{tool_name}' to be called before '{before}', but order was: {names}"
                )
        if after is not None:
            if after not in names:
                raise AssertionError(f"Cannot check ordering: '{after}' was never called")
            if names.index(tool_name) <= names.index(after):
                raise AssertionError(
                    f"Expected '{tool_name}' to be called after '{after}', but order was: {names}"
                )

    def assert_tool_not_called(self, tool_name: str) -> None:
        if tool_name in self.tool_names:
            raise AssertionError(f"Tool '{tool_name}' was called but should not have been")

    def assert_no_hallucinated_tools(self, allowed_tools: list[str]) -> None:
        """Assert the agent only called tools from the allowed list."""
        allowed = set(allowed_tools)
        hallucinated = [n for n in self.tool_names if n not in allowed]
        if hallucinated:
            raise AssertionError(
                f"Agent called tool(s) not in the allowed list: {hallucinated}\n"
                f"Allowed: {sorted(allowed)}"
            )

    def assert_tool_input(self, tool_name: str, **expected_fields: object) -> None:
        """Assert that a tool was called with specific input fields."""
        matches = [tc for tc in self._tool_calls if tc["name"] == tool_name]
        if not matches:
            raise AssertionError(f"Tool '{tool_name}' was never called")
        for key, value in expected_fields.items():
            for call in matches:
                inp = call.get("input") or {}
                if isinstance(inp, dict) and inp.get(key) == value:
                    return
            raise AssertionError(
                f"No call to '{tool_name}' had input {key}={value!r}.\n"
                f"Actual inputs: {[c.get('input') for c in matches]}"
            )

    # ── Step / token assertions ───────────────────────────────────────────────

    def assert_steps_under(self, max_steps: int) -> None:
        """Assert the agent completed in at most max_steps LLM calls."""
        n = len(self._tape.interactions)
        if n > max_steps:
            raise AssertionError(f"Agent took {n} LLM call(s), expected at most {max_steps}")

    def assert_total_tokens_under(self, max_tokens: int) -> None:
        """Assert the total tokens (input + output) across all calls."""
        total = 0
        for i in self._tape.interactions:
            body = i.response.body
            if isinstance(body, dict):
                usage = body.get("usage", {})
                total += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if total > max_tokens:
            raise AssertionError(f"Agent used {total} token(s), expected at most {max_tokens}")

    # ── Completion assertions ─────────────────────────────────────────────────

    def assert_stop_reason(self, reason: str) -> None:
        """Assert the final LLM call stopped for the given reason."""
        from ._tape import _sse_stop_reason

        if not self._tape.interactions:
            raise AssertionError("Tape has no interactions")
        last = self._tape.interactions[-1]

        if last.is_streaming:
            actual = _sse_stop_reason(last.response.chunks or [])
        else:
            body = last.response.body
            if not isinstance(body, dict):
                raise AssertionError("Last response body is not a JSON object")
            # Anthropic uses stop_reason; OpenAI uses choices[].finish_reason
            actual = body.get("stop_reason")
            if actual is None:
                choices = body.get("choices", [])
                actual = choices[0].get("finish_reason") if choices else None

        if actual != reason:
            raise AssertionError(f"Expected stop reason '{reason}', got '{actual}'")

    def assert_task_completed(self) -> None:
        """Assert the agent finished naturally (not by tool call or error)."""
        self.assert_stop_reason("end_turn")

    # ── Diff helpers ──────────────────────────────────────────────────────────

    def diff(self, other: TapeAssertions) -> list[str]:
        """Return a human-readable list of behavioral differences vs another run."""
        lines: list[str] = []
        a, b = self.tool_names, other.tool_names
        if a != b:
            lines.append(f"Tool call sequence changed:\n  was: {a}\n  now: {b}")
        if len(self.interactions) != len(other.interactions):
            lines.append(
                f"Step count changed: {len(self.interactions)} → {len(other.interactions)}"
            )
        return lines
