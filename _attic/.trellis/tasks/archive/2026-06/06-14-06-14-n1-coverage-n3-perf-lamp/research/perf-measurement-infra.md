# Research: `ssh rvv` performance-measurement infrastructure (N3 性能灯 enabler)

- **Query**: Can TianChen-RV MEASURE performance (time/cycles) on real `ssh rvv` hardware? Does the e2e harness time or only check correctness? What does gate4 measure? Are there scalar + naive-RVV baselines? What's the gap to a reproducible cycle/time seam?
- **Scope**: internal (read-only)
- **Date**: 2026-06-14

## Verdict (one line)

**Partial.** A REAL same-target wall-time seam vs a scalar-C baseline already exists, runs on the real `ssh rvv` riscv64 board, and has produced recorded numbers — but those numbers are a **regression** (tuned RVV ~0.76–0.80× = 20–24% *slower* than the scalar-source baseline). Gaps to the spec's N3 bar: (a) no cycle counter (wall-time only), (b) **no deliberate naive-RVV baseline** (only a scalar-source comparator, which `-O2 -march=rv64gcv` may itself autovectorize), so "beat scalar AND naive RVV" is currently *unmeasurable as a controlled 3-way comparison*, and (c) no recorded win — the 灯 is honestly off.

---

## Findings

### Files Found

| File Path | Role |
|---|---|
| `scripts/rvv_generated_bundle_abi_e2e.py` (40 973 lines) | e2e ABI harness — generate RVV bundle → scp → remote `clang` → run on `ssh rvv` → **correctness PASS markers only, NO timing** |
| `scripts/rvv_generated_bundle_same_target_measure.py` (5 056 lines) | **Gate 4** — reuses the e2e bundle, builds a C timing harness, times generated-vs-scalar on `ssh rvv` with `clock_gettime`, classifies win/no-win/regression. THIS is the real perf seam. |
| `scripts/rvv_remote_probe.py` (909 lines) | Remote capability probe; reads `csrr vlenb` (VLEN, **not** a cycle timer) at line 68 |
| `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh*/` | **Recorded real-hardware runs** (git-ignored, ephemeral) — see results below |

### Q1 — Does the e2e harness measure time/cycles, or only correctness?

**Only correctness.** Repo-wide grep for `rdcycle|rdtime|rdinstret|clock_gettime|time.time|cycles|elapsed|warmup|median` over `rvv_generated_bundle_abi_e2e.py` returns **zero** timing hits — the only matches are `unique_non_monotonic...` substrings in printf text. The remote path (`run_remote_evidence`, line 30576) does:

- `clang -O2 -march=rv64gcv -mabi=lp64d -I. <harness> <object> -o <bin>` (line 30617–30625)
- run binary, then `require_contains(stdout, expectation.pass_marker)` + `require_contains(stdout, f"PASS op={kind}")` (line 30641–30650).

So the e2e harness proves numeric PASS / tolerance vs a **reference oracle** (an expected-value computation in C, e.g. `packed_i4_reference_oracle`, e2e line 22728: `acc[0] + sum_bytes(lhs.low_i4*rhs.low_i4 + lhs.high_i4*rhs.high_i4)` then f32 scale). There is **no wall-time or cycle measurement anywhere in the e2e path.**

### Q2 — What does gate4 / same_target_measure actually measure?

**Real wall-clock time — not metadata.** The timing C harness (`DEQUANT_MEASUREMENT_HARNESS_TEMPLATE`, lines 344–622) is a genuine micro-benchmark:

- `now_ns()` uses `clock_gettime(CLOCK_MONOTONIC_RAW)` (line 358–366); `TIMING_METHOD = "clock_gettime(CLOCK_MONOTONIC_RAW)"` (line 40).
- Structure: `correctness_guard` first (line 538 `CORRECTNESS_GUARD_BEFORE_TIMING`), then `MEASURE_WARMUPS` warmups (default 2, line 542), then `MEASURE_REPEATS` × `MEASURE_ITERATIONS` timed loops (defaults 5 × 8), taking the **best (min) per-iteration time** for both baseline and generated (lines 549–584).
- Speedup printed per case: `speedup = baseline_per_iter / generated_per_iter` (line 577–578), with `SUMMARY ... best_speedup=...` (line 585–592). A `volatile tcrv_measurement_sink` defeats dead-code elimination (line 356).
- `run_remote_measurement` (line 3128) scp's object+header+harness → captures `uname -m`, `clang --version`, `lscpu` → remote `clang` compile → run → `parse_measurement_stdout` → **raises if no timing records** (line 3234–3237).
- `classify_parsed_timing` (line 1866): `win iff every SUMMARY best_speedup > 1.0; regression iff every < 1.0; otherwise no-win` (line 1910–1913); `best_speedup_range = "{min:.6f}..{max:.6f}"` (line 1921).

**Two distinct layers — keep them separate:**
1. **Timing layer = REAL.** A genuine wall-time number and speedup ratio. NOT a structural/metadata measurement.
2. **Claim-gate layer = metadata.** Whether a win may be *claimed* is additionally gated by the `low_precision_resource` provider contract: `provider_contract_allows_performance_claim` (line 1933) requires `classification == win` AND `performance_selection_eligible == true` AND `dispatch_preference == PACKED_I4_PERFORMANCE_PREFERRED_DISPATCH`, cross-checked against a maturity contract (`maturity_contract_alignment`, line 1967). The script's docstring calls this `measurement-evidence-input-only; provider-owned low-precision resource facts ... remain the maturity contract` (line 64–67). The `low_precision_resource.*` metadata is the **win-claim gate**, not the speed number itself.

**So: yes, gate4 produces a real speed number** (best_speedup, wall-time ns), and separately a metadata gate decides whether that number licenses an N3 win claim.

### Recorded real-hardware result (the honest 现状)

`ssh rvv` is a **real riscv64 board** (not QEMU): `~/.ssh/config` → `Host rvv` = `192.168.8.72`, `User ubuntu`, `ProxyJump rvv-jump` (`211.87.236.75`). Captured target profile (`remote_target_profile_stdout.txt`):

```
remote_arch=riscv64
remote_uname=Linux ubuntu 6.12.23 ... riscv64 GNU/Linux
clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)
Architecture: riscv64
```

Gate4 has been run for real (artifact dirs `..._ssh` and `..._ssh_current`, not just `..._dry`). NOTE: these `artifacts/tmp/...` dirs are **git-ignored / untracked** (`git check-ignore` confirms ignored; `git ls-files` count = 0) — they are *recorded* run output on disk, reproducible by re-running gate4, not a checked-in artifact. The recorded `evidence.json` of `gate4_packed_i4_same_target_measure_ssh_current`:

```
classification              = "regression"
outcome_family              = "no-win"
best_speedup_range          = "0.761468..0.804008"
performance_win_claim_allowed = false
```

Raw per-case SUMMARYs (op `widening_product_reduce_dequantize_f32`, n ∈ {257, 4096, 65536}, all on real hw): every `best_speedup` is in **0.76–0.81** — i.e. the tuned RVV bundle is consistently ~20–24% **slower** than the scalar-source baseline binary. This matches the project's honest "通但慢" status: the measurement seam works and reports the truth; the truth is currently a regression, so the N3 灯 is off. (Caveat on what "scalar" means here: see Q3.)

### Q3 — Baselines

- **Scalar baseline: EXISTS in source, timeable — but "scalar" only at source level.** It is a real `__attribute__((noinline))` scalar C loop, e.g. `baseline_product_reduction_dequant_v1` (line 368–376): `for (i<n) sum += (int32)lhs[i]*(int32)rhs[i]; out[0]=sum*scale;`, serving a dual role `"same-target scalar C comparator and correctness oracle"` (line 2282); identities in `BASELINE_IDENTITIES`/`PACKED_I4_BASELINE_IDENTITIES` (line 41–54). **Important caveat:** the baseline TU is compiled `-O2 -march=rv64gcv` (no `-fno-vectorize`), so clang's autovectorizer may turn this int8→int32 reduction into RVV vector code. The binary that was *timed* as "scalar" may in fact be clang-autovectorized RVV. Consequence: the recorded regression is "tuned RVV slower than clang's `-O2`/march=rv64gcv compiled baseline," which may not be a true scalar loop. (Verifying would need disassembly on the board; the remote binary is cleaned up after each run, so this is unverified here.)
- **Deliberate naive-RVV baseline: DOES NOT EXIST.** Repo-wide grep for `naive.?rvv|naive.?vector|untuned` finds matches **only in spec prose** (`spec/index.md` N3 row, `guides/trunk-discipline.md`, `variant-pipeline/generation-selection-tuning.md`) — never in any harness or C template. Gate4 times exactly **one** generated vector bundle vs the scalar-source comparator; there is no second, deliberately-untuned vector variant. Nuance: because the scalar-source comparator may auto-vectorize (above), the harness may already be implicitly comparing tuned-RVV against *an* autovectorized vector loop — but there is no *controlled* tuned-vs-naive-RVV axis. Since the spec bar is "实测赢 scalar 且赢 naive RVV," that controlled half of the bar is currently **unmeasurable as a clean 3-way comparison**, not merely unmet.
- **The "generated/tuned" artifact is real RVV vector code** (so a win is in principle possible): the bundle is produced by the actual `tcrv-opt`/`tcrv-translate` pipeline (`--tcrv-opt build/bin/tcrv-opt`, `--tcrv-translate build/bin/tcrv-translate`, lines 5035–5036) and uses genuine RVV intrinsics — `__riscv_vwmacc_vv_i32m1`, `__riscv_vwredsum_vs_i16mf2_i32m1`, `__riscv_vwmul_vv_i16mf2`, `__riscv_vfcvt_f_x_v_f32m1`, `vsetvl` regions (e2e lines 321–323, 638–644, 1551–1554).

### Q4 — What a reproducible cycle/time seam + baselines would require

Most of (a)+(c)+(d) already exist for the scalar-vs-tuned, wall-time axis. The concrete remaining build:

**(a) Timing method.** Wall-time via `clock_gettime(CLOCK_MONOTONIC_RAW)` is already proven to work on this board. To add **cycles**, do NOT assume `rdcycle` works just because `csrr vlenb` does — on Linux riscv64, U-mode `rdcycle`/`rdinstret` reads are gated by `scounteren`/`mcounteren` and commonly trap (SIGILL) or read as 0; `rdtime` is more often available but is a fixed-frequency wall-clock, not a core-cycle count. Recommended order: (1) **probe** `rdcycle`/`rdtime`/`rdinstret` the same way `rvv_remote_probe.py` probes `vlenb` (inline `__asm__ volatile("csrr %0, cycle"/"rdcycle %0" : "=r"(v))`, catch SIGILL); (2) keep `clock_gettime` as the reliable fallback (already wired); (3) `perf_event_open(PERF_COUNT_HW_CPU_CYCLES)` as the portable cycle path if rdcycle traps. Note board is clang 18.1.3 — rdcycle inline-asm syntax differs between `csrr x, cycle` and the `rdcycle` pseudo. Effort: ~0.5–1 day to add+probe a cycle path; **the timing-loop scaffold (warmup/repeat/best-of, sink) is already reusable.**

**(b) Compile scalar + naive-RVV + tuned of the same kernel.** Scalar-source + tuned already compiled side-by-side. Two new pieces: (i) **pin the scalar baseline's vectorization** so "vs scalar" is honest — compile the baseline TU with `-fno-vectorize -fno-slp-vectorize` (or a no-`v` `-march`, e.g. `rv64gc`) so it is genuinely scalar; (ii) add a deliberate **naive-RVV** variant: an untuned vector lowering of the same kernel (fixed LMUL=1, no unroll, straightforward vsetvl strip-mine) emitted as a third object/identity, so the harness times all three. The clang-autovectorized loop can additionally be *enrolled as a candidate naive-RVV baseline*. This is the real new authoring work and depends on the Gearbox/lowering side (see sibling `research/gearbox-current-state.md`). Effort: dominated by producing the naive-RVV emitter + pinning the scalar TU, not the harness.

**(c) Run all on `ssh rvv` + collect.** `run_remote_measurement` already does scp → remote clang → run → parse → JSON evidence, with target-profile capture and cleanup. Extending it from 2 variants to 3+ (scalar-pinned, naive-RVV, tuned, optional autovectorized) is additive (one more object, one more timed loop, one more SUMMARY line family). Effort: small.

**(d) Results table.** `classify_parsed_timing` + `evidence.json` already record per-case `baseline_best_per_iter_ns`, `generated_best_per_iter_ns`, `best_speedup`, `best_speedup_range`, classification. To express the spec's `kernel × {scalar, naive-RVV, tuned}` table and the "beat BOTH" rule, extend the classifier to require `tuned < naive-RVV` AND `tuned < scalar` per case. Effort: small once (b) lands.

**riscv64/clang specifics to bake in:** board default `-O2 -march=rv64gcv -mabi=lp64d` (gate4 line 39); **pin the scalar baseline against autovectorization** (current biggest correctness risk for the "vs scalar" claim); sweep `-O3`/`-O1`; warmups essential (board is shared, jump-hosted → high variance); best-of-N already done; capture `lscpu` + `clang --version` per run (already captured).

### Related Specs

- `.trellis/spec/index.md` (N3 row, line 26) — "Gearbox 候选空间由 capability + resource facts 推导，且在若干 kernel 上**实测赢** scalar 与 naive RVV"
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md` (line 55) — "N3 还要求在若干 kernel 上**实测赢** scalar 且赢 naive RVV：没有胜出的 tuning 没有论文故事"
- `.trellis/spec/guides/trunk-discipline.md` (line 13) — N3 = "Gearbox 候选枚举/剪枝 + 实测胜出 scalar 与 naive RVV"
- `core-invariants.md` I8 — hardware/perf claims require real `ssh rvv` evidence (gate4's recorded riscv64 profile satisfies the *form* of I8; the *win* does not yet exist)

## Caveats / Not Found

- **No cycle counter anywhere** — repo-wide grep for `rdcycle|rdtime|rdinstret|perf_event_open` finds only `csrr vlenb` (VLEN read, not a timer) in `rvv_remote_probe.py:68`. All timing is wall-clock `clock_gettime`.
- **No deliberate naive-RVV baseline in harness/C code** — only in spec prose. The controlled "beat naive RVV" half of the N3 bar is unmeasurable today as a clean 3-way comparison.
- **The "scalar" baseline may be clang-autovectorized** (compiled `-O2 -march=rv64gcv`, no `-fno-vectorize`) — so the recorded regression is vs an `-O2`/march-rv64gcv baseline, not provably a true scalar loop. Unverified (remote binary is cleaned up; would need on-board disassembly).
- **Recorded result is a regression, not a win** (best_speedup 0.76–0.80 on the only op family with `_ssh` evidence: `widening_product_reduce_dequantize_f32`). I did not enumerate every op kind's evidence; other kernels may not have been measured on hw yet. `--dry-run` is opt-in (store_true, line 4969), so the default invocation does hit hardware; whether CI runs it with `--dry-run` was not exhaustively traced.
- The `artifacts/tmp/...` evidence is **git-ignored / untracked** (ephemeral, reproducible by re-running gate4 — not a committed artifact).
- The depth of the Gearbox candidate enumeration/pruning (whether tuning is real resource-aware vs MVP placeholder) is **out of scope here** — see sibling `research/gearbox-current-state.md`.
- I did not modify any source files; read-only.
