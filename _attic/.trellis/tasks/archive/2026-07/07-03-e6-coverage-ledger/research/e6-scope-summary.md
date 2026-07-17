# Research: E6 Scope Summary — MVP, land sequence, escalations, non-goals

- **Query**: Recommend the E6 MVP (denominator + metrics-computable-now + ledger, six-state auto-readout deferred to E5), land sequence, genuine escalations, and what E6 explicitly does NOT do.
- **Scope**: internal (synthesis of the other three research files)
- **Snapshot**: HEAD `1bbab882695e812a2d334a8d6f5c7860cc2c93b1`, 2026-07-03.
- **Sibling deliverables**: `coverage-denominator.md`, `six-state-current.md`, `metric-and-ledger-script-design.md` (same dir).

---

## 1. The load-bearing goal (why E6 exists)
E6 produces the **ruler** that lets **G1 (engine-axis `C_construct` strong burn-down)** be *scored*. Per PRD (`.trellis/tasks/07-02-full-refactor/prd.md:107,115,129`): **G1 is gated on E5/E6** — "没尺子无法给 G1 打分". So the two load-bearing outputs are: (1) the **denominator of record** (the roster), and (2) the **`C_construct` (strong) definition** — everything else is supporting. Get those two exact; the rest is enumeration + plumbing.

---

## 2. Recommended MVP (matches the PRD's own prescription — not a reach)
PRD lines 129/134 and 实验总纲 line 107 already prescribe exactly this. Ship, in one task:

1. **Denominator finalized [COV-1]** — commit `schema/coverage-roster.v1.json`:
   - ggml pin `6eab471` in `$meta` (ratify first — see §4).
   - A类 (24 vec_dot + decomposed 3 + quantize/dequantize_row + GEMM tiles + IME×{q4_0,q8_0,q4_K}) / B类 (9 forward ops) / C类 (flash-attn, bf16).
   - Denominator key = `(op, format[, shape_class])`; **decomposed family = separate keys (option a)**.
2. **Four-metric script [COV-2]** — `.trellis/scripts/coverage_metrics.py` (stdlib, mirrors E1): reads roster + a committed **hand-labeled** six-state table (`schema/coverage-sixstate.v1.json`), emits `C_dispatch` / `C_construct` (strong) / `C_construct+` (weak, report-not-gate) / `C_attr` (leveled) as a CI-report artifact with a snapshot-ID `$meta`. `C_construct` rows carry `auto_readout: pending-E5`.
3. **Ledger script [LED-1]** — `.trellis/scripts/family_ledger.py` (stdlib, cloc-approximation since `cloc` absent): per-family `{code_LOC, test_LOC, table_rows, interface_touchpoints, calendar_days}` from git history + a family→dirs manifest. **Recompute the IME first point: 2484 raw / 1866 cloc-approx (≈ target 1865) verified.**
4. Both scripts ship `--self-test` (E1 discipline).

**Numbers do NOT go in spec.** They go in the CI report / 执行总纲, each carrying a snapshot ID ([GOV-1]/[GOV-3]).

---

## 3. Land sequence
1. **Ratify the two blocking calls** (ggml pin; decomposed-key policy) — §4. Cheap, but they set the denominator.
2. **Write `schema/coverage-roster.v1.json`** (denominator of record) + `schema/coverage-sixstate.v1.json` (hand-labeled, seeded from six-state-current.md §2: 17 dispatch-wired / 7 weak / 3 strong + the 5 wired forward ops + absent entries).
3. **`coverage_metrics.py`** — reads (roster, six-state) → four metrics + `$meta`. Self-test on a synthetic fixture.
4. **`family_ledger.py`** — cloc-approx + git-history; recompute IME. Self-test on synthetic strings + fixed dates.
5. **Emit the first CI-report block** into 执行总纲/CI (not spec), pinned to HEAD.
6. (Optional, low-risk) extend `check_schema_gate.py`'s canonical-hash idiom to hash the roster + six-state table for content-addressing.

Build risk: **low** (net-new Python, like E1; touches no C++/ODS).

---

## 4. Genuine escalations (a human should rule; each moves the denominator or the headline)
1. **ggml pin** — reuse sibling `6eab471` (untagged, sibling repo not submodule) vs a tagged upstream; whether to vendor the format enum for self-containment. (coverage-denominator.md §1)
2. **Decomposed-family key policy** — endorse **(a) separate keys**; consequence: block-dot `vec_dot` `C_construct` = **0/24 today**, decomposed = **3/3 strong** in its own bucket; G1 burns the 24 down. Option (b) would let q4_0's decomposed-strong path pollute the burn-down denominator. Ratify (a). (coverage-denominator.md §4)
3. **B类 boundary: gelu/add/mul/cpy** — absent in-code. Are they **B-pending** (in denom, state=absent, pull down C_dispatch → B类 denom = 9) or **C-暂缓** (excluded → B类 denom = 5)? Directly sets the global C_dispatch target ([COV-3] M1 ≥80% / M2 ≥90%).
4. **A vs C for edge low-bit formats** — the ruling moves low-bit *into* the numerator, so all IQ/ternary/fp4 vec_dot are A. But **q1_0** (1-bit) and **nvfp4** are borderline (both in-code as own monoliths) — default A; confirm.
5. **quantize_row / dequantize_row / GEMM-tile roster width** — the largest denominator lever after the 24 vec_dot. Roster must fix the per-format enumeration (all A-formats? only activation quantizers q8_0/q8_1/q8_K?). In-code today: only `quantize_row_q8_0` + sparse dequant + 5 RVV GEMM formats + format-agnostic IME.
6. **Ledger `test_LOC` attribution rule** — the ≈659 figure does not reproduce (path-glob `ime-*.mlir` = ~230). Choose path-glob vs explicit test manifest; pin the file set. `code_LOC` (2484/1866) is safe. (metric-and-ledger-script-design.md §4)
7. **cloc-approx vs real cloc** — [A-5] says "LOC 用 cloc"; `cloc` is not installed. Approximation validates exact (1866 vs 1865); a human may prefer installing cloc for the canonical wording.

---

## 5. What E6 explicitly does NOT do (non-goals / hand-offs)
- **Does NOT block on E5.** Six-state **auto-readout** (the [K-4] state machine) + **[L-8] machine enforcement** (weak-reported-as-strong CI trap) are **E5** (provenance manifest, 实验总纲 line 107 "provenance 清单先于六态自动化"). E6's six-state input is a **committed hand-labeled table** with a `pending-E5` marker; it does not derive strong-vs-weak from code.
- **Does NOT touch C++/ODS** — pure tooling (Python), implementation-stack red line.
- **Does NOT write numbers into `.trellis/spec/`** — spec is a stable contract, zero current value ([GOV-1]); metrics live in CI report / 执行总纲 with snapshot IDs.
- **Does NOT build the directory-归拢** (E2b) — the family→dirs manifest is an interim hand-maintained input until E2b lands `plugins/<family>/`.
- **Does NOT wire CI/`.github`** (that is F-1/F-5 CI thrust E3) — E6 produces the report artifact + scripts; CI invocation is downstream.
- **Does NOT do the marginal-cost curve [LED-2]** (needs ≥3 families / X-SCALAR) or perf/burn-down curves [COV-4/5] (need hardware / attribution JSONL).

## Caveats
- All escalations in §4 are *ratifiable now* with the evidence in the sibling files; none require new investigation. Shipping (a)+`6eab471` unblocks E6 immediately; the rest (B类 width, quant/GEMM width) can be encoded as roster entries whose class a human toggles without code changes.
