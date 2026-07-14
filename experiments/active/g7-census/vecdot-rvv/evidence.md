# G7 L1 Batch-3a/3b — vec_dot@rvv cold-start census (FLAT 5 + K-quant 5)

**Line**: G7 终编成令三·第一段 rvv-half 收口 · vec_dot lane. **Board** = rvv (openEuler, Linux 6.12 riscv64, VLEN128, 64c). **Compiler** = **gcc-15.2.0** (symmetric: ours emitted kernel + opponent `libggml-cpu.so` both gcc-15). **march** = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`. **Opponent lib** = `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-cpu.so` md5 `d1adc634c2ca04ffc389536b30d9d9c4` (== batch1-kquant-rvv established as-shipped opponent; test-before == test-after, read-only link, never modified).

## Method
- ONE binary / ONE process / core-pinned. Load-gate picked **core 8** (idle 100%; co-tenant vLLM on 0,1 untouched). gov=performance freq=2.6GHz.
- **Cold** = 224 MiB flush (>3× 64MiB L3) before EACH timed region, paired A/B, N=12 reps, median + relIQR. **Hot** = single-buffer warmup + best-of-8. Driven as GEVM output[M][nc], each cell = one vec_dot of length K=2048, nc=512.
- **M=1** GEVM (decode regime, primary verdict) + **M=8** shape point (batch regime, third-段 attribution).
- **byte-exact ZERO-MODEL gate**: ours vs independently-compiled stock ggml oracle, INT-mode bounded fill (d=1.0, dmin/min=0). **ALL 10 formats: 0 mismatch / 0 ULP** (M=1 & M=8 & nc=512). No dead cell.
- ours = weft-emitted block-dot (kernels/*.kernel.c, VLEN-invariant emit `__riscv_vsetvl_e8m1`, recompiled gcc-15.2; GEN_SEAL md5-sealed). FLAT: emits ggml's own block-dot ⇒ parity-by-adoption. K-quant: KQuant aux32 integer-core.

## Opponent classification (符号级机判 · `--disassemble=<sym>` VLEN128 runtime · see raw/rvv_opponent_dispatch_probe.txt)
**★ nm -D alone MISSES `.isra.0` local vl-specializations** — must objdump the dispatch. Corrected result:
- FLAT: q4_0/q4_1 inline light-vec (4ins/1rvv); q8_0 inline light-vec (4/3); **q5_0/q5_1 inline better-vec (17/10, 14/8 + csrr)** — genuinely vectorized native RVV.
- q2_K → dispatch `_vl128` **hand-tuned STRONG**. q3_K → `_vl128` (of vl128/256/512) **hand-tuned STRONG**. q4_K → `_vl128.isra.0` **hand-tuned STRONG** (NOT generic; nm missed it). q6_K → `_vl128.isra.0` **hand-tuned STRONG** (full vl128/256/512/1024 family). q5_K → **inline VLEN-adaptive main (36ins/23rvv/11×csrr, no clean vl-spec) = MODERATE** (qh-burdened; the one non-hand-brick K-quant).

## Results (cold median, ratio = ours/opp, >1 ours faster)

### Batch 3a — FLAT vec_dot (parity-by-adoption)
| fmt | opp class | cold M=1 | cold M=8 | verdict (M=1 primary) |
|---|---|--:|--:|---|
| q4_0 | inline light-vec | **0.920×** | 0.957× | PARITY-by-adoption ✓ tautological |
| q4_1 | inline light-vec | **1.020×** | 1.026× | PARITY-by-adoption ✓ tautological |
| q5_0 | inline better-vec | **0.265×** | 0.263× | **LOSS** ★parity-by-adoption **证伪** (qh 5-bit emit 发散·同 k1·rvv 更深) |
| q5_1 | inline better-vec | **0.272×** | 0.268× | **LOSS** ★**证伪** |
| q8_0 | inline light-vec | **1.046×** | 1.010× | PARITY-by-adoption ✓ tautological |

### Batch 3b — K-quant vec_dot (ours aux32 core vs as-shipped)
| fmt | opp class | cold M=1 | cold M=8 | verdict (M=1 primary) |
|---|---|--:|--:|---|
| q2_K | vl128 hand-tuned STRONG | **0.254×** | 0.440× | **LOSS** (weight-recon floor vs strong hand-tuned) |
| q3_K | vl128 hand-tuned STRONG | **0.272×** | 0.522× | **LOSS** |
| q4_K | vl128.isra hand-tuned STRONG | **0.167×** | 0.348× | **LOSS** (deepest) |
| q5_K | inline VLEN-adaptive MODERATE | **0.899×** | 1.200× | **LOSS/near-parity @M=1**; M=8 flips **1.20× WIN** ⚠ = opponent-relative-immaturity (q5_K opp is the only un-hand-tuned K-quant·同 k1 机制·非"赢强手调") |
| q6_K | vl128.isra hand-tuned STRONG | **0.283×** | 0.425× | **LOSS** |

## Tally (M=1 GEVM primary verdict)
- **0 WIN** · 3 PARITY-by-adoption (q4_0/q4_1/q8_0 · tautological · ours emits ggml's own block-dot) · **7 LOSS** (q5_0/q5_1 parity-by-adoption falsified; q2_K/q3_K/q4_K/q6_K weight-recon floor vs strong hand-tuned _vl128; q5_K 0.90× near-parity).
- q5_K M=8 1.20× and the only >1 M=1 ratios (q4_1 1.02×, q8_0 1.05×) are tautological-parity / opponent-immaturity — **no genuine beat**.
- **matmul-axis kernel-sym ≥parity net-new (真 beat vs 强出货·非 tautological·非 immaturity) = 0**.

## Board hygiene
core-pinned 8 · co-tenant on 0,1 untouched · load-gate 100% idle · lib md5 test-before==after (read-only) · 0 stray proc (pkill -x) · /tmp/g7_vecdot_rvv scratch removed · loadavg begin 2.09 → end 2.86 (co-tenant drift, not us). No git. No emitter/ODS/lib source change.
