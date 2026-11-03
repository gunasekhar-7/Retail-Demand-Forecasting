# 0004: Operational hardening without a gateway

## Context

The first security review found no exploitable code-level vulnerability, but the
service had no boundary controls: no authentication, no request limits, no metrics, a
single health endpoint that mixed liveness with readiness, and a `/health` response
that echoed artifact paths. Compose published both ports on every interface and CI
trusted floating action tags.

The project must stay runnable on a laptop with one command. A full identity provider,
service mesh or Redis-backed limiter would be disproportionate for a public-data demo.

## Decision

Put the controls that are cheap and local in the service, and document the ones that
belong to infrastructure:

- Optional shared-key auth via `DEMAND_API_KEY`, enforced only on `/model` and
  `/forecast`; health and metrics remain open for orchestrators.
- Per-worker fixed-window rate limiting via `DEMAND_RATE_LIMIT_PER_MINUTE`, disabled by
  default. It is a safety net, not a fleet-wide quota.
- A 64 KiB body limit before JSON parsing, request IDs, one JSON log line per request,
  and Prometheus counters/histograms with allowlisted route labels.
- Split `/health/live` and `/health/ready`; startup failures log the exception and
  return a generic detail.
- Compose binds loopback and runs read-only, non-root, capability-dropped containers
  with tmpfs and memory/PID limits.
- CI pins actions to commit SHAs, restricts token permissions, builds and smoke-tests
  the hardened image, and audits dependencies weekly.

## Consequences

The demo still starts with `demand demo` and no environment variables. Enabling auth is
one variable, and the OpenAPI document shows the `X-API-Key` requirement.

The shared key is not identity: it cannot isolate tenants or support revocation per
user. The in-process limiter is per worker, so a two-worker container allows twice the
configured rate, and a restart resets the window. Public deployment still needs a
reverse proxy for TLS, authentication and shared quotas.

Prometheus metrics are process-local; a multi-worker deployment needs one scrape target
per worker or a push/aggregation layer. Artifact signing remains out of scope; the
read-only mount is the current integrity control.
