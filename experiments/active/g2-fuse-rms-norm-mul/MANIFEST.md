# cell MANIFEST — g2-fuse-rms-norm-mul

- **campaign**: fuse (G2 [FUSE] rms_norm→mul epilogue; L3 memory axis)
- **status**: ACTIVE (design-review tracer; 2026-07-07; STOP-at-贯通, no fan-out)
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

## note — STOP-at-贯通 (data-only cell)
This is a data/evidence cell (design-review pack). No perf number is a beat; every quantity is an
emit-level structural byte count — real DRAM bandwidth benefit is pending-board ([NG-4], [PERF-1]
八门 not entered). Nothing here was committed / git add-ed. Fan-out to other fusion pairs and
[FMT-PROP] format negotiation are explicitly OUT of scope until the user greenlights 立项.
