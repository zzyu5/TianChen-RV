# Fedora RVV0.7.1 (Sophgo SG2042 / T-Head C920) — proven targetable (2026-06-22)

## Decisive proof: the C920 runs RVV0.7.1, faults RVV1.0
- **Toolchain**: RuyiSDK `gnu-plct-xthead` 3.1.0 (XuanTie sources, PLCT build) — GCC 14.1.1, binutils
  2.42, native riscv64 host (runs on the C920), ships `riscv_vector.h`. Via proxy from the ISCAS
  RuyiSDK mirror, sha256-verified, installed `~/xuantie` (2.6G) on Fedora.
- **Hand RVV0.7.1 kernel** (i32 vadd + i16 widening dot-reduce, `__riscv_*` intrinsics, `-march=
  rv64gc_xtheadvector`): objdump shows genuine 0.7.1 `th.v*` encodings (`th.vsetvli`, `th.vadd.vv`,
  `th.vwmul.vv`, `th.vredsum.vs`); **runs green** → `dot=1258 ref=1258 PASS, exit 0`.
- **Control**: an RVV1.0 (ratified-V) binary (`vmv1r.v`, `-march=rv64gcv`) → **Illegal instruction,
  exit 132**. Clean dichotomy: C920 executes 0.7.1, faults 1.0. The exact dot arithmetic confirms a
  real run, not emulated/no-op.
- **march-spelling trap (settled)**: `-march=rv64gcv0p7` is TOOLCHAIN-dependent — stock gcc13 binutils
  assembles it to standard-V (`010072d7`, SIGILLs); XuanTie binutils assembles it to `th.*` (runs).
  Portable spelling = **`rv64gc_xtheadvector`**.

## Emitter/parser gap (bounded, ~1 eng-week to first green tcrv-opt RVV0.7 e2e)
- (A) march hardcoded `rv64gcv` at `lib/Target/RVV/RVVTargetSupportBundle.cpp:1683-1684` + probe
  candidates `scripts/rvv_remote_probe.py:440-448` → need xtheadvector march + XuanTie gcc.
- (B) ~40 `__riscv_*` 1.0 intrinsics in `lib/Plugin/RVV/EmitC/` are **empirically source-compatible**
  with XuanTie gcc (the test kernel used the emitter's vocabulary + ran) → the work is a **0.7-policy
  variant** (no ta/ma policy field; explicit EMUL re-vsetvl on widening) + a legality gate banning
  1.0-only ops (whole-register moves, differing segment/indexed forms).
- (C) **N1 GAP — the tractable entry point**: the march-parser in `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  does NOT recognize RVV0.7 as distinct — the substring tokenizer matches "gcv" inside "rv64gcv0p7" and
  **silently folds it into full RVV1.0** (SEW/LMUL/Zvl128b); `rv64gc_xtheadvector` matches no axis token
  → vector-present with ZERO divergence axes. **A real RVV0.7 capability (version property + narrowed
  0.7.1 allow-lists) must be added** so the SAME kernel gets DIFFERENT legality/selection on RVV0.7 vs
  RVV1.0. This is an N1 ISA-generation capability divergence — the deepest N1 axis (vs the VLEN axis K1
  gives). Full findings: `rvv07-targetability.log`.

## Status
- RVV0.7 hardware: **proven real + targetable** (no blocker).
- Next increment (this campaign): teach the capability model to recognize RVV0.7 (xtheadvector) as a
  distinct version → same kernel diverges on RVV0.7 vs RVV1.0 (N1 at the model level, bounded, local).
- Full RVV0.7 emission + on-C920 e2e run: ~1 eng-week (toolchain + emitter 0.7-policy variant + legality
  gate). Deferred as the follow-on; the hardware/toolchain are ready on Fedora.
