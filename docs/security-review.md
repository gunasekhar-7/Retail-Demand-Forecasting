# Security review

**Date:** 2026-11-02  
**Scope:** whole repository — API, dashboard, registry, CLI, SQL, Docker, CI.  
**Threat model:** read-only public retail data; unauthenticated by default; the artifact
directory is trusted server-side state.

## Result

No high-confidence exploitable vulnerability was found. Four low or
deployment-conditional findings were identified and fixed; the remaining risks are
documented below. `pip-audit` reports no known vulnerabilities in the locked environment.

## Findings and remediation

| ID | Finding | Severity | Status |
|---|---|---|---|
| SEC-001 | API and dashboard published on all interfaces with no authentication | Low (Medium if internet-facing) | Fixed: compose binds `127.0.0.1`; optional `DEMAND_API_KEY`; per-worker rate limit; dashboard stays read-only public data |
| SEC-002 | `/health` returned the raw startup exception, including artifact paths | Low | Fixed: clients receive `model unavailable`; the full exception is logged |
| SEC-003 | GitHub Actions used floating major tags and default token permissions | Low (supply chain) | Fixed: actions pinned to commit SHAs, `permissions: contents: read`, weekly `pip-audit` |
| SEC-004 | Container lacked read-only rootfs, capability drop, no-new-privileges and limits | Low | Fixed: compose hardening plus resource/PID limits; artifacts stay `:ro` |

## Verified controls

- **Injection**: SQL is static templates; config-scoped identifiers and paths are the
  only interpolated values and `_quote()` doubles single quotes. Request bodies never
  reach SQL.
- **Deserialization**: no `pickle`, `joblib`, `eval` or `exec`. Artifacts are LightGBM
  text, Parquet and JSON; configuration uses `yaml.safe_load`.
- **Command execution**: `subprocess` uses argument lists with `shell=False`; `git` runs
  in a fixed working directory.
- **XSS**: no `unsafe_allow_html`, `st.html` or `components.html`; Streamlit escapes
  rendered values.
- **Secrets**: no credentials in source, configs or images; `.env` and local config are
  ignored; logs omit headers and bodies.
- **CORS**: no CORS middleware, so browsers receive no cross-origin grant.
- **Request surface**: `extra="forbid"`, bounded `series_id` and `horizon_days`, 64 KiB
  body limit, per-request id, allowlisted metric labels.
- **Container**: non-root `appuser`, read-only rootfs, `cap_drop: ALL`,
  `no-new-privileges`, tmpfs for runtime writes.

## Residual risks

- `DEMAND_API_KEY` is a shared secret, not per-user identity. Use OIDC and per-tenant
  authorization before serving private data.
- The rate limiter is process-local and fixed-window; a gateway must own shared limits
  and TLS termination.
- Artifacts are trusted from disk and are not signed. The read-only mount limits the
  exposure; add a checksum in `metadata.json` for tamper evidence.
- CI builds and smoke-tests the image but does not scan it (for example Trivy), and no
  DAST or external penetration test has been run.

## Verification

`tests/test_service_hardening.py` covers missing/wrong keys, open health and metrics,
request-id echo and generation, the metrics registry, readiness on a missing artifact,
generic error detail, `429` with `Retry-After`, `413` body limits and security headers.
`pip-audit` runs weekly in CI; the Docker job exercises the hardened runtime.
