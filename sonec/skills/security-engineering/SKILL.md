---
id: security-engineering
name: Security Engineering
description: Hostile-input mindset, OWASP-minded fixes, evidence-based auditing.
always: false
priority: 15
tags: [security, auth, owasp, vuln, xss, injection]
triggers: [security, auth, vulnerability, xss, injection, secret, permission]
---

# Security Engineering

- Assume all input is hostile.
- Validate at boundaries; least privilege; secure defaults.
- Never hardcode secrets; never log secrets.
- For audits: recon → hypothesis → prove with evidence → remediate.
- Prefer runnable checks over unverified claims.
