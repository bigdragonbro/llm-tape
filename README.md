# agent-tape

**Framework-agnostic record/replay test harness for LLM agents.**

Record any agent run to a portable `.tape.yaml` file. Replay it in CI — no API keys, no flakiness, no cost. Assert on behavioral properties: which tools were called, in what order, how many steps, how many tokens.

Works with any framework that uses `httpx` under the hood: Anthropic SDK, OpenAI SDK, LangChain, LangGraph, CrewAI, raw `httpx` calls.

---

## Why this exists

Testing LLM agents is a genuinely unsolved problem. The tools developers reach for — unit tests, mocks, LLM-as-judge eval frameworks — all fail in different ways:

**Unit tests break the wrong thing.** Mocking the LLM means you're testing your mock, not your agent. When the model changes its reasoning or tool-call patterns, mocked tests pass and production breaks.

**Eval frameworks test quality, not behavior.** Tools like DeepEval and MLflow are great for measuring response quality with LLM-as-judge. But they don't tell you whether your agent called the right tools in the right order, stayed within your token budget, or finished the task at all.

**Every framework solves it differently.** LangGraph has checkpointing. LangSmith has trace replay. But these only work within the LangChain ecosystem. If you switch frameworks, or use a bare `anthropic` SDK client, you start from scratch.

**The result:** most agent teams ship with no behavioral regression tests. When the model or prompt changes, they find out from users.

### What agent-tape does differently

agent-tape intercepts at the HTTP transport layer — below any framework, above the network. It records exactly what the LLM was asked and what it said, stores that as a git-committable YAML file, and replays it deterministically. Your agent code runs unchanged; it just gets pre-recorded answers instead of live API calls.

Then you write assertions about structure, not content:

```python
tape.assert_tool_called("read_file", before="write_summary")  # correct order
tape.assert_task_completed()                                   # didn't get stuck
tape.assert_steps_under(5)                                     # didn't loop
tape.assert_no_hallucinated_tools(["read_file", "write_summary"])  # stayed in scope
```

These tests run in CI with no API key, in milliseconds, and catch regressions before they reach production.

---

## Install

```bash
pip install agent-tape
```

---

## How it works

```
Your agent code
      │
      ▼
  httpx.Client / httpx.AsyncClient   ← agent-tape patches here
      │
      │  record mode: wraps real transport, saves each call to tape
      │  replay mode: returns recorded responses, no network
      ▼
  LLM API (Anthropic, OpenAI, ...)
```

Patching at the `httpx` transport layer means:
- Zero changes to your agent code
- Works with any SDK or framework that uses `httpx`
- Sensitive headers (`x-api-key`, `authorization`) are scrubbed automatically

---

## Full working example

The [`examples/summarize_agent/`](examples/summarize_agent/) directory contains a complete runnable example: a file-summarizer agent, a recording script, and a test suite.

### The agent ([`agent.py`](examples/summarize_agent/agent.py))

```python
import anthropic
from pathlib import Path

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a local text file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_summary",
        "description": "Write the final summary to a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["content", "path"],
        },
    },
]

def run_agent(task: str) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=4096,
            messages=messages, tools=TOOLS,
        )
        # collect assistant content, run tools, feed results back...
        if response.stop_reason == "end_turn":
            return response.content[0].text
        # ... (see examples/summarize_agent/agent.py for the full loop)
```

### Step 1 — Record once (requires API key)

```bash
ANTHROPIC_API_KEY=sk-... python examples/summarize_agent/record_tape.py
```

This runs the real agent, captures every LLM call, and writes [`tapes/summarize_document.tape.yaml`](tapes/summarize_document.tape.yaml). Commit it:

```bash
git add tapes/summarize_document.tape.yaml
git commit -m "chore: record summarize_agent tape"
```

### Step 2 — Test forever after (no API key)

```python
# examples/summarize_agent/test_agent.py

import agent_tape
from agent import run_agent

TAPE = "tapes/summarize_document.tape.yaml"

@pytest.fixture
def result():
    with agent_tape.replay(TAPE) as tape:
        text = run_agent("Please summarize the file notes.txt")
    return text, tape

def test_reads_file_before_writing(result):
    _, tape = result
    tape.assert_tool_called("read_file", before="write_summary")

def test_completes_task(result):
    _, tape = result
    tape.assert_task_completed()

def test_efficient(result):
    _, tape = result
    tape.assert_steps_under(5)

def test_no_hallucinated_tools(result):
    _, tape = result
    tape.assert_no_hallucinated_tools(["read_file", "write_summary"])
```

```bash
pytest examples/summarize_agent/test_agent.py
# runs in <100ms, zero API calls, works in CI
```

---

## Quick start (any agent)

```python
import agent_tape

# Record
with agent_tape.record("tapes/my_agent.tape.yaml"):
    result = my_agent.run("do the thing")

# Replay + assert
with agent_tape.replay("tapes/my_agent.tape.yaml") as tape:
    result = my_agent.run("do the thing")
    tape.assert_tool_called("search")
    tape.assert_task_completed()
    tape.assert_steps_under(10)
```

### pytest markers

```python
@pytest.mark.tape("tapes/my_agent.tape.yaml")
def test_agent_behavior(tape_replay):
    my_agent.run("do the thing")
    tape_replay.assert_tool_called("search", before="write_file")
    tape_replay.assert_task_completed()
```

### Streaming agents

SSE streaming is fully supported — chunks are captured and replayed exactly as recorded:

```python
# Record a streaming run
with agent_tape.record("tapes/streaming.tape.yaml"):
    with client.messages.stream(model="claude-sonnet-4-6", ...) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

# Replay — same streaming interface, no API call
with agent_tape.replay("tapes/streaming.tape.yaml") as tape:
    with client.messages.stream(...) as stream:
        full_text = stream.get_final_text()
    tape.assert_task_completed()
```

---

## How tapes fit into your workflow

A tape is a **snapshot of Claude's behavior** at a point in time. Once committed to git, every future test run replays that snapshot — no API key, no cost, no flakiness.

```
Developer (once, needs API key)         CI (every PR, no API key needed)
───────────────────────────────         ─────────────────────────────────
python record_tape.py                   pytest test_agent.py
        │                                       │
        ▼                                       ▼
  real Claude API call              tape replayed locally
        │                                       │
        ▼                                       ▼
tapes/my_agent.tape.yaml ──git commit──▶ assertions checked
                                          (~0.5s, $0.00)
```

### What breaks these tests?

Not random LLM variation — the tape freezes the model's responses. What breaks them is **your code changing** in a way that alters agent behavior: a refactored tool loop, a changed prompt, a new tool added. That's exactly what you want CI to catch.

### When do you re-record?

Intentionally, when you mean to change behavior:

- You updated the system prompt
- You added or renamed a tool
- You want to capture improved model behavior after a model upgrade

Run `record_tape.py` again, review the tape diff in your PR, and commit. The diff tells reviewers exactly how Claude's behavior changed — which tools it now calls, in what order, how many tokens it used. This makes behavioral changes explicit and reviewable instead of silent.

### Tapes in CI (no API key required)

The Anthropic SDK validates that an API key exists before making any call. In replay mode no real call is made, but you still need a non-empty value to pass that check. Set a dummy key in your test fixture:

```python
@pytest.fixture
def result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-replay-key")
    with agent_tape.replay("tapes/my_agent.tape.yaml") as tape:
        text = run_agent("do the thing")
    return text, tape
```

In CI, add `ANTHROPIC_API_KEY: test-replay-key` to your workflow environment — no real credentials needed.

---

## Tape format

Tapes are plain YAML — designed to be committed, reviewed in PRs, and diffed:

```yaml
version: '1'
metadata:
  recorded_at: '2026-06-24T10:00:00Z'
interactions:
  - id: call_0
    duration_ms: 412.3
    request:
      method: POST
      url: https://api.anthropic.com/v1/messages
      headers:
        x-api-key: '***'          # always scrubbed
        content-type: application/json
      body:
        model: claude-sonnet-4-6
        messages:
          - role: user
            content: Please summarize the file notes.txt
    response:
      status: 200
      body:
        stop_reason: tool_use
        content:
          - type: tool_use
            name: read_file
            input:
              path: notes.txt
        usage:
          input_tokens: 312
          output_tokens: 48
```

See [`tapes/summarize_document.tape.yaml`](tapes/summarize_document.tape.yaml) for a full 3-call example.

---

## All assertions

| Method | What it checks |
|---|---|
| `assert_tool_called(name)` | Tool was called at least once |
| `assert_tool_called(name, before=other)` | Tool was called before another |
| `assert_tool_called(name, after=other)` | Tool was called after another |
| `assert_tool_not_called(name)` | Tool was never called |
| `assert_tool_input(name, **fields)` | Tool was called with specific input values |
| `assert_no_hallucinated_tools(allowed)` | Agent only called tools from this list |
| `assert_steps_under(n)` | Agent finished in ≤ n LLM calls |
| `assert_total_tokens_under(n)` | Total tokens (input + output) ≤ n |
| `assert_task_completed()` | Final stop reason is `end_turn` |
| `assert_stop_reason(reason)` | Final stop reason matches exactly |
| `diff(other_tape)` | Returns human-readable list of behavioral differences |

---

## Limitations

- **Sequential replay only.** Tapes replay interactions in the order they were recorded. If your agent makes truly parallel LLM calls (concurrent `asyncio.gather` across multiple clients), the replay order may not match. Parallel tool execution within a single response is fine.
- **Frameworks that pre-create clients.** If a framework constructs its `httpx.Client` before the `record()`/`replay()` block is entered, that client won't be intercepted. Workaround: pass a transport explicitly at construction time, or initialize the framework inside the block.
- **Non-httpx transports.** Libraries that use `urllib3`, `aiohttp`, or `requests` directly aren't intercepted. Most major LLM SDKs use `httpx`; check your framework's dependencies if unsure.

---

## Contributing

```bash
git clone https://github.com/bigdragonbro/agent-tape
cd agent-tape
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

PRs welcome. Current roadmap:
- [ ] Parallel call support (non-sequential replay)
- [ ] Pre-built injection helpers for LangChain / CrewAI early-init
- [ ] `agent-tape diff` CLI for comparing two tape files
- [ ] VS Code extension for tape visualization

---

## License

MIT
