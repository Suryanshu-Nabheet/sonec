# Contributing to SONEC

Thank you for helping build one of the strongest open-source agentic SE systems.

## Principles

Follow `target.md` and the always-on prebuilt rules:

- Production-grade only — no placeholders, no fake backends
- Smallest correct change
- Tests for every feature
- Document architecture trade-offs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check sonec tests
```

## Pull requests

1. One logical change per PR
2. Include tests
3. Update docs when behavior/architecture changes
4. Do not commit secrets or `.env`

## Skills & prebuilt rules

- Skills live in `sonec/skills/<id>/SKILL.md`
- Prebuilt rules live in `sonec/rules/prebuilt/*.mdc`
- Keep skill bodies actionable; keep rules authoritative

## Code of collaboration

Be direct, precise, and respectful. Prefer evidence over vibes.
