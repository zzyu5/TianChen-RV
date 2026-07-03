# Research: Four-Metric Script [COV-2] + Ledger Script [LED-1] — design

- **Query**: Design the two E6 scripts (four-coverage-metric + ledger), their inputs/outputs/location/CI-report format; check `cloc` availability; recompute the IME first point (raw vs cloc vs test split); snapshot-ID discipline.
- **Scope**: internal
- **Snapshot**: HEAD `1bbab882695e812a2d334a8d6f5c7860cc2c93b1`, 2026-07-03.
- **Authority**: [COV-2] four metrics (`科研目标总纲v2.md:130-134`); [LED-1] (`:112`); [A-5] LOC=cloc, tests single-listed (`:233`); [GOV-3] snapshot ID; E1 pattern `.trellis/scripts/check_schema_gate.py`.

---

## 1. The E1 pattern to mirror (both scripts follow it)

`.trellis/scripts/check_schema_gate.py` (16.8 KB, landed E1, commit `c8f3e945`) establishes the house style for governance tooling. Reusable idioms:
- **stdlib-only** (`argparse, hashlib, json, subprocess, sys, pathlib`) — Python is *tooling*, never the compiler stack (implementation-stack red line). Module docstring states this explicitly.
- **`REPO_ROOT = Path(__file__).resolve().parents[2]`** — file lives at `<repo>/.trellis/scripts/`.
- **Canonical JSON hash** for reproducibility: `json.dumps(obj, sort_keys=True, separators=(",",":"))` → `hashlib.sha256(...).hexdigest()` (`canonicalize`/`compute_hash`). Reuse verbatim for the snapshot ID / roster hash.
- **`git` via subprocess** helper `_git([...])` = `git -C <repo> …`.
- **`--self-test`** hermetic subcommand (no real git refs, no repo writes) with a `check(name, cond)` accumulator and PASS/FAIL print. E1 ships 16 self-tests. **Both E6 scripts must ship `--self-test`.**
- **argparse subcommands** (`report`, `gate`) with `set_defaults(func=…)`.

**Location for both E6 scripts**: `.trellis/scripts/` (same dir as `check_schema_gate.py`). Proposed names: `coverage_metrics.py` and `family_ledger.py`.

---

## 2. Four-metric script [COV-2] — `coverage_metrics.py`

### Inputs (all committed artifacts)
1. **Roster** = `schema/coverage-roster.v1.json` (the denominator; coverage-denominator.md §5).
2. **Six-state table** = a committed hand-labeled JSON, e.g. `schema/coverage-sixstate.v1.json`, mapping each roster key → `{state ∈ six-state ladder, auto_readout ∈ {structural, pending-E5}, anchor}`. Hand-labeled for MVP; E5 will later auto-fill `state` from the provenance manifest and drop `pending-E5`.

### Metrics ([COV-2] — count STRONG only for `C_construct`)
Per denominator key, take the **best six-state across its variant rows** (实验总纲 line 32). Denominator classes reported separately (A / B / C / global) because targets differ ([COV-3]).

| Metric | Numerator | Notes |
|---|---|---|
| `C_dispatch` | keys with state ≥ dispatch-wired / denom | structurally derivable NOW |
| `C_construct` | keys with state ≥ **constructed (strong)** / denom | **the burn-down headline [L-8]**; strong only; MVP reads label, `pending-E5` for auto |
| `C_construct+` | keys with state ≥ constructed-weak / denom | transition metric, **report-not-gate** |
| `C_attr` | leveled string `^CT / load / ^RT` | NOT a single ratio ([D-4]); today `^CT partial, load 0, ^RT 0` |

### Output — CI-report artifact (NOT spec; numbers carry snapshot ID [GOV-3])
- Emit a JSON/Markdown block to a CI-report path (e.g. stdout + optional `--out`), carrying `$meta`: `{repo_snapshot: <git HEAD sha>, ggml_pin: <roster $meta>, roster_sha256: <canonical hash>, sixstate_sha256, epoch, ts}`.
- Numbers go to the **CI report / 执行总纲**, never into `.trellis/spec/` (spec is zero-current-value, [GOV-1]).
- Report shape mirrors 执行总纲 §3 table: per metric = `{numerator, denominator, pct, per-class breakdown, anchor}`.

### Subcommands (E1 style)
- `report [--out PATH]` — compute + print the four metrics with the `$meta` snapshot header.
- `--self-test` — hermetic: feed a tiny synthetic roster + six-state table, assert the four numerators (e.g. 24 wired / 0 strong / 7 weak on the block-dot fixture), assert best-state-across-variants dedup, assert `C_construct` counts strong only.

### What it does NOT do (E5 boundary)
- Does not *derive* strong-vs-weak from code (that is E5's provenance manifest). It reads the label and marks `auto_readout: pending-E5`.
- Does not gate CI on `C_construct+` (transition metric).

---

## 3. Ledger script [LED-1] — `family_ledger.py`

### `cloc` availability
**`cloc` is NOT installed** (`which cloc` → empty). Therefore ship a **stdlib cloc-approximation** (Python is tooling; no dependency): count physical lines that are non-blank and not a pure comment, stripping `/* … */` block comments (regex, DOTALL) and `//`-leading line comments. This approximation is **validated against the target** (see §4): it yields 1866 for the IME source set vs the [LED-1] target ≈1865 — essentially exact. Byte-reproducible (no external tool, no version skew).

### Per-family record ([LED-1])
`{family, code_LOC (cloc-approx, EXCL tests), test_LOC (single-listed, SEPARATE), table_rows, interface_touchpoints, calendar_days}`, derived from git history + the cloc-approx:

| Field | How derived |
|---|---|
| `code_LOC` | cloc-approx over the family's **non-test source dirs** (see family→dirs manifest below), excluding `CMakeLists.txt`. |
| `test_LOC` | cloc-approx (or raw `wc -l` for `.mlir`) over the family's **test files**, single-listed separately. **⚠ attribution rule is OPEN** — see §4 escalation. |
| `table_rows` | count of the family's table entries (e.g. RVV: `monolithicBlockDotOpTable()` = 24 rows; a family's capability-fact rows). |
| `interface_touchpoints` | count of `ExtensionPlugin` virtual overrides the family implements (getCapabilities / verifyVariantLegality / buildVariantEmissionPlan / …) — grep the plugin file. |
| `calendar_days` | git span: `git log --format=%ad --date=short -- <family paths>` → last − first commit date. |

### Family → source-dirs manifest (interim, until E2b directory 归拢)
[F-3] directory 归拢 (E2b) is **NOT done**, so a family's code is spread across multiple dirs. The ledger needs a hand-maintained family→paths manifest (mirrors E1's interim `ONBOARDING_TRAILER` note). IME's 4 dirs @ HEAD (all confirmed present):
- `lib/Dialect/IME` · `lib/Plugin/IME` · `include/TianChenRV/Dialect/IME` · `include/TianChenRV/Plugin/IME`

When E2b lands `plugins/<family>/`, the manifest collapses to one prefix.

### Subcommands
- `report --family <name>` — print the 6-field record + `$meta` snapshot header.
- `--self-test` — hermetic cloc-approx on synthetic strings (blank/`//`/`/* */` stripping), calendar-day arithmetic on fixed dates, table-row counting on a fixture.

---

## 4. IME first-point recompute (raw vs cloc vs test split) @ HEAD `1bbab882`

**Code side (recomputable, verified):**

| Dir | raw `wc -l` | files (excl CMakeLists) |
|---|---|---|
| `lib/Dialect/IME` | 438 | IMEDialect.cpp |
| `lib/Plugin/IME` | 1588 | IMEExtensionPlugin.cpp (858) + IMEBackendEmissionDriver.cpp (730) |
| `include/TianChenRV/Dialect/IME` | 366 | IMEOps.td (350) + IMEDialect.h (16) |
| `include/TianChenRV/Plugin/IME` | 92 | IMEExtensionPlugin.h (65) + IMEBackendEmissionDriver.h (27) |
| **raw total** | **2484** | matches README + §8 exactly |

- **raw `wc -l` = 2484** (the README figure — this is raw, not cloc).
- **cloc-approx (blank + `//` + `/* */` stripped) = 1866** — my stdlib approximation over the same 4 dirs. This lands on the [LED-1]/§8 target **≈1865** (off by 1, i.e. essentially exact). **This is the recomputable first point for `code_LOC`.**

**Test side (OPEN — the ≈659 figure does not reproduce cleanly):**
- §8 states `test_LOC ≈ 659` should be single-listed, but gives no file set.
- A path-glob `test/**/ime-*.mlir` yields only the 6 IME-execution lits (ime-mma-*, ime-matmul-*, ime-mma-capability-absent-negative) = **~230 lines**, not 659.
- A broad `*ime*` glob pulls in unrelated runtime-scalar / dispatch-guard lits (8137 total) — clearly over-broad.
- **Root cause**: there is **no defined test→family attribution rule**. 659 came from some intermediate accounting not pinned to a file set.

### Escalation (needs a human ruling before the ledger emits `test_LOC`)
Define the family→test attribution rule explicitly, one of:
1. **Path-glob** `test/**/<family-slug>*` (narrow; gives ~230 for IME) — cheap, reproducible, but misses cross-family or differently-named lits.
2. **Explicit per-family test manifest** (a `tests: [...]` list in the ledger manifest) — precise, but hand-maintained.
The **code_LOC** first point (2484 raw / 1866 cloc) is verified and safe to ship; **test_LOC is the open sub-item** and should ship with the chosen rule + its file set pinned, not the un-sourced 659.

---

## 5. Snapshot-ID discipline ([GOV-3])
- Every metric/ledger emission carries `{repo_snapshot: git HEAD, ggml_pin, roster_sha256, sixstate_sha256, epoch, ts}` in its `$meta`.
- Cross-snapshot comparison must declare the interval explicitly.
- Numbers live in the **CI report + 执行总纲**, never in `.trellis/spec/` (spec = zero current value).
- Reuse E1's `canonicalize`/`compute_hash` for `roster_sha256` / `sixstate_sha256` so the inputs are content-addressed.

## Caveats / Not Found
- `cloc` absent → the approximation is the plan; it validates to 1866 vs target 1865 (exact enough), but a human may prefer to `apt install cloc` for the canonical [A-5] "LOC 用 cloc" wording. The approximation should document that it is a cloc-*approximation*, not cloc.
- No existing ledger script (`ledger` / git-history LOC generator grep = 0).
- `table_rows` / `interface_touchpoints` extraction for **IME** is straightforward (single plugin file); for **RVV** the "table" is `monolithicBlockDotOpTable()` (24) — the script needs a per-family table-anchor manifest, another interim hand-maintained input pending E2b.
