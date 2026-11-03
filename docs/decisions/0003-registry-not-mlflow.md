# ADR 0003: File-based registry, not MLflow

Status: accepted (2026-11-02)

## Context

The project trains one model family, once per data cutoff, with a small parameter
grid. Experiment tracking, model lineage dashboards and a tracking server are not
requirements. The portfolio goal is to show an inspectable artifact contract.

## Decision

Use a directory per run with `model.txt`, `feature_spec.json`,
`serving_history.parquet`, `calendar.parquet`, `metadata.json` and a single
`latest.json` pointer. Metadata includes the git SHA, dirty flag, cutoff, horizon,
dataset scope and validation metrics.

## Consequences

- Any artifact can be inspected, diffed and rolled back with `git`-like tools; no
  service to run.
- There is no automatic parameter sweep table and no cross-run UI. If the project grows
  to multiple model families or scheduled retraining, migrate to MLflow and import the
  JSON metadata.
- `serving_history.parquet` makes the artifact self-contained: serving never re-reads
  the 59M-row panel.
