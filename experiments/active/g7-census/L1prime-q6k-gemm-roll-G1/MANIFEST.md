# MANIFEST — L1' q6_K@rvv GEMM whole-K-nest roll G1 feasibility

**Line**: G7 终编成令四·L1' 攻坚（K-quant GEMM emitter roll·q6_K@rvv deepest loss 0.037×）.
**Date**: 2026-07-14 · **HEAD**: 249145ff (branch refactor/full-refactor-m1) · **Local static account, NO board**.
**Verdict**: **NET RECOVERY (Exit A)** — whole-K-nest roll collapses vsetvli-storm (6578→21/super-block, −313×) with ZERO re-decode (decode-arith identical, 4-col tile preserved); GEMM residual = accumulator spill (reduced 2.4×, not eliminated; GEVM clean 0-spill unreachable). Predicted 0.037× → ~2.8-3×+ (sub-parity). ⇒ implant + G2 board-validation. [K-10]=structural.

## Files
| file | role |
|---|---|
| `evidence.md` | ★ full G1: disassembly attribution + roll design + static A/B account + byte-exact + NET verdict + [K-10] |
| `raw/metrics_summary.txt` | compact per-super-block dynamic table |
| `raw/gen.py` | generator (RVV sizeless types forbid arrays → explicit 32 named acc vars) |
| `raw/q6k_gemm_roll_gen.c` | hand-construct A/B (rolled vs unrolled, shared SUPERBLOCK_BODY macro, +DRIVER) |
| `raw/q6k_gemm_roll_AB.gcc15.O2.s` | A/B asm (split on `_Z15q6k_gemm_rolled..` / `_Z17q6k_gemm_unrolled..`) |
| `raw/q6k_gemm_CURRENT.emitc.c` | regenerated CURRENT emitter kernel (md5 bf477693 = batch1 census verbatim) |
| `raw/q6k_gemm_CURRENT.gcc15.O2.s` | CURRENT kernel asm (vsetvli 6182, vwmacc 2304, vs*r.v 2222, v31) |

## Provenance
- CURRENT kernel: `weft-opt <fixture> --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (read-only build-weft/bin/weft-opt @HEAD).
- Compiler: SpacemiT gcc-15.2.0 `-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on` (= batch1 census flags; deployment gcc domain).
- byte-exact = by-construction (op-stream identity) + board A/B DEFERRED (no local RVV: qemu-riscv64 absent, GNU sim signal 4).

## Hygiene
- Tracked source UNTOUCHED (RVVToEmitC* read-only; emitter mtime unchanged). No git. No board. hand-construct all under raw/.
