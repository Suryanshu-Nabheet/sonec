# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.2.x | Yes |

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/Suryanshu-Nabheet/sonec/security/advisories/new) on GitHub, or email the maintainer via the profile on GitHub.

Do **not** file public issues for undisclosed vulnerabilities.

## Security model (product)

- Workspace sandbox rejects path escapes
- Terminal blocks a set of destructive patterns; network tools off by default
- Secrets only via environment (`MOONSHOT_API_KEY`, `SONEC_API_KEY`, …)
- Never commit `.env` or credentials
