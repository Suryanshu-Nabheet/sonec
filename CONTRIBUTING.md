# Contributing to sonec

Thank you for contributing to sonec.

## Principles

- Ship production-ready changes — no placeholders or fake backends
- Prefer the smallest correct change
- Add tests for behavior you introduce
- Document architecture trade-offs when they change

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
3. Update docs when behavior or architecture changes
4. Do not commit secrets or `.env`

## Skills and rules

- Skills: `sonec/skills/<id>/SKILL.md`
- Prebuilt rules: `sonec/rules/prebuilt/*.mdc`
- Keep skill bodies actionable; keep rules authoritative

## Collaboration

Be direct, precise, and respectful. Prefer evidence over assumption.
