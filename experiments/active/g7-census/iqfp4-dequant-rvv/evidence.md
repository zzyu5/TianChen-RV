# G7 L1 Batch-3c/3d — iq/fp4 vec_dot@rvv (4) + dequant@rvv [DEQ-AXIS] (18)

**Line**: G7 终编成令三·第一段 rvv-half 收口 · iq/fp4 + dequant lane. **Board** = rvv (Linux 6.12 riscv64, VLEN128, 64c). **Compiler** = **gcc-15.2.0** (symmetric). **march** = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`. **Opponent** = `libggml-cpu.so` md5 `d1adc634…` + `libggml-base.so` md5 `1b4580c4…` (test-before==after, read-only, never modified).

## Method
- Load-gate picked **core 9** (idle 100%; co-tenant vLLM on 0,1 untouched). 224 MiB flush (>3× L3) cold before EACH region, N=12 median + relIQR, hot best-of-8.
- 3c iq/fp4 vec_dot: GEVM M=1 (primary) + M=8, K=2048 nc=512. 3d dequant: streaming K=1048576 (out=4 MiB f32 > 2 MiB L2).
- **byte-exact ZERO-MODEL**: ALL 4 vec_dot + ALL 18 dequant = **0 mismatch / 0 ULP** vs independently-compiled stock ggml oracle. No dead cell.
- **q1_0 = NOT tested** (provenance-suspect: enum 41, canon "Weft-internal"; opp symbol exists but internal-A/B risk) → flag single-列 不计数.

## Opponent classification (符号级机判 · see raw/rvv_opponent_dispatch_probe.txt)
### 3c iq/fp4 vec_dot (libggml-cpu.so, `--disassemble` dispatch @VLEN128)
- iq1_s → `_vl128.isra.0` **hand-tuned STRONG** (tree vl128/256/1024/generic). iq1_m → `_vl128.isra.0` **STRONG**. iq4_nl → `_vl128.isra.0` **STRONG**. nvfp4 → single inline impl, no vl-spec = **MODERATE** (gcc-15.2 codegen).
- ★ Corrects k1 framing (k1 iq opp = "RVV-gather generic", nvfp4 = "near-scalar weak WIN 1.06×"). On rvv opponents are stronger hand-tuned _vl128.

### 3d dequant (libggml-base.so `dequantize_row_*` = deployed `to_float` ref)
- **★ MAJOR board correction vs k1**: on rvv these are MOSTLY **gcc-15.2 AUTO-VECTORIZED** (rvv-insn>0), NOT the k1 "deployed-scalar-reference". Per-format opp class in summary_dequant_rvv.csv (e.g. q3_K 34rvv, q4_K 25rvv, iq3_s 140rvv+gather). Only **iq2_xs / iq2_s / nvfp4 stayed scalar** (0 rvv).
- OURS dequant (gcc-15.2): q2_K/q3_K/q4_K/q5_K/tq1_0/tq2_0 vectorized; iq*/mxfp4/nvfp4/iq4_nl/q6_K compiled **scalar** (vsetvl=0). ⇒ several wins are ours-scalar-vs-opp-autovec-gather (opponent pays the RVV gather/reconfig tax).

## Results

### 3c — iq/fp4 vec_dot (cold median, ours/opp, M=1 primary)
| fmt | opp class | cold M=1 | cold M=8 | verdict |
|---|---|--:|--:|---|
| iq1_s | vl128 hand-tuned STRONG | **0.969×** (o_iqr 14% noisy; best 1.14×) | 1.049× | near-PARITY (report-as-parity·noisy·not clean beat) |
| iq1_m | vl128 hand-tuned STRONG | **0.122×** | 0.121× | **LOSS** (deep·gather-trap vs strong) |
| iq4_nl | vl128 hand-tuned STRONG | **0.389×** | 0.378× | **LOSS** |
| nvfp4 | inline single-impl MOD | **0.678×** | 0.677× | **LOSS** (cf. k1 WIN 1.06×; rvv opp stronger) |

**3c tally**: 0 clean WIN · 1 near-parity (iq1_s ⚠noisy) · 3 LOSS. **matmul-axis net-new beat = 0.**

### 3d — dequant [DEQ-AXIS] (cold median · streaming · GB/s in csv)
**9 WIN / 1 PARITY / 8 LOSS** (independent sub-account · does NOT enter matmul headline):
- **WIN (9)**: q3_K 1.25×, q6_K 1.66×, iq2_xxs 3.88×, iq2_xs 2.95×, iq2_s 3.10×, iq3_xxs 1.47× (best 1.96, o_iqr 8%), iq4_nl 7.05× (best 2.06, opp cold-outlier inflated), mxfp4 5.98× (best 1.67, iqr high), tq1_0 1.13×.
- **PARITY (1)**: iq1_s 0.98×.
- **LOSS (8)**: q2_K 0.82×, q4_K 0.32× (opp autovec 3.34 GB/s), q5_K 0.61× (o_iqr 44% noisy), iq1_m 0.35×, iq3_s 0.25×, iq4_xs 0.71× (p_iqr 57% noisy), nvfp4 0.66×, tq2_0 0.49× (o_iqr 29%).
- vs k1 (13 WIN/5 LOSS): rvv has MORE losses because rvv opp `to_float` is gcc-15.2-autovectorized (stronger) where k1's was clang-scalar. Wins concentrate where opp autovec pays gather tax (iq2_*) or is scalar (iq2_xs/iq2_s).

## Board hygiene
core-pinned 9 · co-tenant 0,1 untouched · load-gate 100% · both lib md5 test-before==after (read-only) · 0 stray (pkill -x) · /tmp/g7_iqfp4_dq_rvv scratch removed · loadavg end 3.04 (co-tenant drift). No git. No source change.
