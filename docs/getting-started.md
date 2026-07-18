# Getting started

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Inspect the harness brain

```bash
sonec skills    # progressive expertise packs
sonec rules     # operating rules + your Cursor rules
```

## Offline multi-phase run

```bash
sonec run "Inspect the repo and summarize structure" --mock
```

You should see phases: `recon → plan → execute → verify → critique → deliver`.

## Live (Kimi K3)

```bash
export MOONSHOT_API_KEY=sk-...
sonec run "Fix X and verify with pytest" --workspace .
```

## Library

```python
import asyncio
from sonec.app import build_harness
from sonec.llm import MockProvider

async def main() -> None:
    provider = MockProvider.harness_smoke("demo")
    harness, settings, workspace, tools = build_harness(
        workspace=".", provider=provider, persist_memory=False
    )
    result = await harness.run("demo")
    print(result.final_message)

asyncio.run(main())
```

`--simple` uses the single-loop runtime; default is the full harness.
