# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.x | Yes |

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/Suryanshu-Nabheet/sonec/security/advisories/new) on GitHub, or email the maintainer via the profile on GitHub.

Do **not** file public issues for undisclosed vulnerabilities.

## Security model (product)

- Workspace sandbox rejects path escapes
- Terminal blocks a set of destructive patterns; network tools off by default
- Secrets only via environment (`SONEC_API_KEY`, `OPENAI_API_KEY`, …). Local OpenAI-compatible servers typically accept any key string.
- Never commit `.env` or credentials
