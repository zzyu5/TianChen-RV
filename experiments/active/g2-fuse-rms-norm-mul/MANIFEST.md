# cell MANIFEST — g2-fuse-rms-norm-mul

- **campaign**: fuse (G2 [FUSE] rms_norm→mul epilogue; L3 memory axis)
- **status**: ACTIVE — **BOARD-MEASURED (裁决二.5, 2026-07-07, rvv/VLEN128)**: paired fused-vs-unfused
  cold N=12 => **1.308× wall speedup** (fused 165.18ms vs unfused 215.99ms; IQR 0.04–0.06%
  non-overlapping; noise floor 1.63% => ~19× floor => T-N PASS) + **measured DRAM bytes eliminated =
  256.2 MiB/iter = 100.1% of the predicted y[] round trip 8·n·rows (256.0 MiB)** via LLC load/store-miss
  deltas (cycle cross-check 50.2ms/iter ≈ wall delta 50.8ms/iter). On-hardware value equality ULP=0.
  **★ 铺量触发 (rollout-trigger) SATISFIED.** [NG-4] isolated A/B on the memory axis — NOT a ggml beat,
  NOT an e2e [PERF-1] eight-gate; rollout立项 stays the user's. Prior leg: design-review tracer
  (STOP-at-贯通, emit-level byte accounting).
- **role**: G2 [FUSE] rms_norm→mul (llama attn_norm/ffn_norm) 贯通 tracer + 设计评审包. The
  producer tcrv_rvv.elementwise_rms_norm_reduce_core carries an OPTIONAL single-block $epilogue
  region (L1 chained declaration) holding a tcrv_rvv.elementwise_mul_map consumer brick; the
  reduce-body emitter SPLICES it (L2) so the register-kept normalized vy flows straight into a
  per-lane vfmul_vv against w[] and stores z[] once — the intermediate normalized row y[] is NEVER
  stored and NEVER reloaded. BYTE ACCOUNTING (emit-level structural count, NOT a perf beat —
  [NG-4] real bandwidth 待板): the y[] round-trip = 8·n bytes/row (4·n store + 4·n reload) is fully
  eliminated (40% of the normalize+mul stage / 33% of the full kernel); per-strip the fused kernel
  emits 2 vle32 + 1 vse32 vs the two-kernel 3 vle32 + 2 vse32 (−1 load, −1 store = the y[] traffic).
  byte-exact (UNFUSED zero-regression; FUSED no-FMA per-lane identity), NG-2 compliant (kernel-level
  epilogue region, not a graph framework). STOP-at-贯通: only rms_norm→mul built; fan-out /
  [FMT-PROP] / other pairs are a 立项 decision owned by the user.
- **layout**: org STAGE2 per-cell manifest. Fusion CODE lives outside this data cell
  (include/…/RVVOps.td, lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp, RVVToEmitC.cpp
  allowlist, lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp verifier) + the tracer lit
  (test/Conversion/RVV/rvv-to-emitc-typed-elementwise-rms-norm-mul-fused-epilogue-loop-body.mlir).
  This cell holds only the design-review evidence.

## durable files

- `NOTES.md`             — design review pack: mechanism (L1 chain → L2 splice), anti-bypass, byte
  accounting, byte-exact proof, NG-2 compliance, generalizability ([FMT-PROP]/epilogue), STOP-at-贯通.
- `byte_accounting.csv`  — the intermediate-tensor byte accounting (y[] round-trip = 8·n bytes
  eliminated; stage/full-kernel scopings; emit-op census; illustrative n_embd projection).
- `emit_census.txt`      — empirical call_opaque memory-op census (FUSED 2 vle32 / 1 vse32 vs UNFUSED
  producer) + fused strip intrinsic trace + tracer lit verdicts, reproduced READ-ONLY from the
  already-built build/bin/tcrv-opt (no rebuild, no git write).
- `board_measured.md`    — ★ BOARD PAIRED MEASURE (裁决二.5): pre-registration, faithful-to-census setup,
  on-hardware ULP=0, cold N=12 wall speedup 1.308× (T-N PASS), measured DRAM bytes = 100.1% of the 8·n
  prediction (dual evidence cycles+LLC-miss), 铺量触发 verdict, [NG-4] framing discipline.
- `board_raw.txt`        — raw board output: correctness verify, N=12 paired rounds + median/IQR, the two
  perf-stat counter runs (cycles + LLC load/store-misses), byte-delta arithmetic, reproduce commands.
  Driver CODE lives at `tools/e2e-harness/board/g2_fuse_driver.c` (+ `g2_fuse_paired.sh`) — data cell holds data only.

## note — data-only cell (mechanism→board, honest scope)
This is a data/evidence cell. The tracer leg is emit-level structural byte counts; the board leg
(裁决二.5) now measures the real DRAM benefit on rvv/VLEN128 — the pending-board caveat is CLOSED for
this pair. Discipline unchanged: the 1.308× is an **isolated A/B on the memory axis** ([NG-4] internal
metric, same category as [GAP-SB] A/B), **NOT** a ggml beat and **NOT** an e2e [PERF-1] eight-gate.
Nothing here was committed / git add-ed. Fan-out to other fusion pairs and [FMT-PROP] format
negotiation are on evidence but remain OUT of scope until the user greenlights 立项.
