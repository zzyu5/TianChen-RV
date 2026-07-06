# experiments/ MANIFEST — thin index (org STAGE1, 2026-07-06)

> **This file was split.** The former single-body REGISTRY is now **one `MANIFEST.md` per cell**
> (campaign + role + status + durable file list). This top-level file is a thin index.
> Complete old→new path map: [`archive/MOVES.md`](archive/MOVES.md). A STAGE2 `INDEX.md` may supersede this.

## Layout (裁决二)

```
experiments/
  active/     in-production / awaiting-seal (CI-refreshed)
    visibility/            T0/T7/T2 auto-generated pack (regen: tools/visibility/)
    result-tables/         FILLED tables: T-N, T3_A, T3_B, T8
    e2e-harness/           data-side index: README + models.manifest.csv
    repack/                PARKED-P3 gemm-constructed-redeploy interim (gitignored cell)
  sealed/     immutable evidence (CI must not mutate; only allowed write = flip a MANIFEST to STALE)
    repack/                5 q4_0 repack e2e cells
    silicon/               silicon-validation-batch-1 / batch-2 / gemm
    c1-cleanliness/        quant-label-proof / opponent-facts-provenance
  archive/    old-campaign sediment (some STALE) + MOVES.md
    perf-historical/       ondevice-q5_K / -q8_0 / -q8_0-deferred / -q8_0-mbf / T3_step3
  _templates/  16 header-only zero-value T-CSV templates
  README.md    table→claim map + cell schema + Win ladder (experiments index)
  MANIFEST.md  (this thin index)
```

## Per-cell manifests

Each cell dir carries a `MANIFEST.md` listing its files + role + campaign + status
(sealed / active / archive / parked / template). `active/repack/MANIFEST.md` sits at the
campaign level because its one cell dir is gitignored.

## Data-only + harness contract (unchanged)

experiments/ stays a **data cell**: data/evidence (`*.json`/`*.csv`/`*.txt`/`*.md`/`*.objdump`/
`*.log`/`*.err`/`.gitignore`) + in-cell evidence-pointer code (`*.kernel.c`/`*.emitc.mlir`/sealed `*.o`).
All harness/protocol/driver/CI scripts live under `tools/` (`tools/e2e-harness/`, `tools/lint/`, `tools/visibility/`).

## ⚠ CI gates need STAGE2 rework (see archive/MOVES.md §8)

`tools/lint/check_manifest.py` (exact-path pin vs this single REGISTRY) and
`tools/lint/check_experiments_data_only.py` (registered-evidence-code lookup in this REGISTRY)
**assume the old flat single-REGISTRY layout** and will be RED after this reorg. They must be
reworked to consume the per-cell manifests (or the STAGE2 INDEX) before committing. Likewise
`tools/lint/check_opponent_facts_pin.sh` and `tools/visibility/*` hardcode old cell paths.

<!-- REGISTRY:BEGIN -->
<!-- SUPERSEDED by per-cell MANIFEST.md files (org STAGE1, 2026-07-06). See archive/MOVES.md.
     Kept empty so the legacy check_manifest.py regex parses without crashing; the gate is
     RED-by-design until STAGE2 rewires it to the per-cell manifests. -->
<!-- REGISTRY:END -->
