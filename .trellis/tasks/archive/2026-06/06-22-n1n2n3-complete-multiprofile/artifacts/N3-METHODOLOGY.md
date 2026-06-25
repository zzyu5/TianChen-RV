# N3 measurement methodology — the scientifically-correct framing (user-directed 2026-06-23)

This is the durable contract for how N3 (and the N1→N3 link) numbers are produced. It supersedes any
earlier framing that headlined scalar comparisons. Written after the user flagged the baselines as
"完全不科学".

## Rule 0 — DROP all scalar baselines from the contribution story
Vector-vs-scalar (the old "4–15×") measures "we vectorized at all" — which MLIR/autovectorization already
provide. It is NOT our contribution and is an inflated, meaningless headline. Scalar may appear ONLY as an
internal sanity check ("yes the vector path is taken"), NEVER as a reported contribution multiple.

## The three contributions, each isolated, each with its correct baseline

| Win | isolates | correct baseline | harness |
|---|---|---|---|
| **Win-A** | the compiler's **automatic tuning choice** (e.g. max-legal-LMUL width) — a pure ablation | **tune OFF**: the SAME kernel with the optimization disabled; BOTH arms compiler-emitted, same algorithm, only the tuned knob differs | single-core microbench (+ e2e where possible) |
| **Win-B** | a generated kernel that **changes the ALGORITHM** (e.g. the q4_0 repack: 16-blocks-as-lanes GEMM/GEMV) | **llama.cpp's OWN shipped RVV kernel** — the real optimized `ggml_vec_dot_q4_0_q8_0` / `ggml_gemm_*` RVV implementation in `arch/riscv/`. NOT scalar. NOT a hand-written "naive RVV". NOT the `_generic` fallback. | single-core microbench **AND** llama.cpp e2e (both required) |
| **Win-C** | an **automatic pass** that changes algorithm STRUCTURE (if/when one exists) | the pass OFF vs ON | single-core + e2e, measured separately |

Honesty test for Win-B vs Win-C: if WE authored the kernel (the repack), it is **Win-B** (kernel
contribution). It is **Win-C** only if a compiler PASS auto-derives the structural change. Do not relabel a
hand-authored kernel as an automatic-pass contribution.

## Both harnesses are required (not interchangeable)
- **single-core microbench**: isolates the kernel/selection; controls memory/threading noise; this is where
  the ablation is clean.
- **llama.cpp e2e (tok/s)**: the real model, real tensors, real graph — this is what catches integration
  bugs (it caught the RVV0.7 GEMM dst bug the microbench was blind to) and what proves the contribution
  survives at the system level. A microbench win that does not show up e2e must be disclosed as such.

## The N1 → N3 link (why profiles multiply the evidence)
- **N1 = capability determines the LEGAL CANDIDATE SET per profile** (e.g. RVV0.7 has no fractional LMUL →
  the enumerable candidate set is categorically different from RVV1.0; not a parameter, a different set).
- **N3 = Gearbox tunes within that profile-specific legal set**; the winner can differ in ALGORITHM, not
  just a parameter.
- Therefore N profiles = N DIFFERENT tuning problems, not N copies of one. The strongest, not-yet-achieved
  N1→N3 result is a case where the winning **algorithm** (not just the LMUL number) differs by profile.
  Until then, the honest claim is "legal set + winning selection differ by profile" (shown), not "winning
  algorithm differs by profile" (open).

## Current-evidence audit against this contract (2026-06-23)
- **Win-A 2.27–3.79×** (rvv, i16 contraction, tune-on vs tune-off, both compiler-emitted): baseline CORRECT
  → stands. Commit 709bb69d. The old "wide vs scalar 4–7.5×" is RETRACTED as a contribution number.
- **Win-B 2.49× / 5–6×** (rvv, repack vs "repack OFF / naive RVV"): baseline SUSPECT — must be re-measured
  against ggml's actual shipped RVV q4_0 kernel (not `_generic`, not naive). Remeasurement in progress.
- **Win-B K1 0.74× LOSS**: baseline already correct (ggml autovec'd shipped path) → honest loss, stands.
- **VLEN-strip 1.48× microbench / 1.31× e2e** (K1, 1×16 vs 2×8, both compiler-emitted): this is a Win-A-class
  selection ablation (both arms emitted, only strip width differs) → baseline correct, stands.
- **Win-C**: none claimed yet.
