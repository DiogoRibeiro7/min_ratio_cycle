# Security Policy

We take the security of **min-ratio-cycle** seriously. This document explains how to report vulnerabilities and what to expect from us.

---

## Supported Versions

We provide security fixes for the latest minor release on the `main` branch. Older versions may receive fixes at the maintainer’s discretion.

| Version |      Supported      |
| ------: | :-----------------: |
|     0.x | ✅ latest patch only |

> If you depend on a specific version, please plan to upgrade promptly when a security release is announced.

---

## Reporting a Vulnerability

* **Email**: `dfr@esmad.ipp.pt`
* **Subject**: `[SECURITY] <short summary>`
* **Do not** open a public issue for security reports.

We aim to acknowledge new reports within **3 business days**.

### What to include

Please provide:

* A clear description of the issue and its impact.
* Minimal reproducible example (code or input graph) and affected versions.
* Environment details (OS, Python version, NumPy/BLAS info).
* Any known workarounds.

If the issue involves sensitive data or private repositories, indicate this in your email. Do **not** send secrets or personal data.

---

## Coordinated Disclosure

Our standard process is:

1. **Triage**: confirm and assess severity (CVSS v3.1 as guidance).
2. **Fix**: develop and test a patch, add regression tests.
3. **Release**: publish a patched version on PyPI and GitHub.
4. **Advisory**: public disclosure with credit to the reporter (unless anonymity is requested).

Target timelines (best effort):

* Critical/High: patch release within **14 days** of confirmation.
* Medium/Low: patch release within **45 days** of confirmation.

These timelines may vary for complex issues or upstream dependency problems.

---

## Scope

* Code in this repository (`min_ratio_cycle/*`, build/CI scripts, docs tooling).
* Integrity of published wheels/sdists on PyPI and GitHub releases.

**Out of scope** examples:

* Social engineering, physical attacks, or third‑party platform issues (e.g., GitHub itself).
* Vulnerabilities solely in **upstream dependencies** without an exploitable impact through our usage.
* Denial‑of‑service via extremely large inputs beyond documented limits.

If you find a dependency vulnerability that affects this project, please include links to the upstream advisory. We will coordinate with the upstream project when appropriate.

---

## Dependency & Supply‑chain Security

* We use Poetry for dependency management and encourage pinning for reproducibility.
* Static analyses (e.g., `bandit`) and pre‑commit checks help prevent common issues.
* Release artifacts are built by CI from tagged commits.

If you suspect a compromised release artifact or CI pipeline issue, contact us immediately with the tag/version details.

---

## Credits

We are happy to credit reporters in advisories and release notes. Please let us know if you prefer to remain anonymous.

---

*Last updated: 2025‑08‑28*
