# G7 L1 B-class forward-op kernel-sym census — DUAL-BOARD (k1 VLEN256 clang-18 · rvv VLEN128 gcc-15)

> **Line**: G7 终编成令三 · 第一段全量 kernel 冷启动普查 · B 类前向算子专项（softmax 用户点名必测）。
> **Boards**: k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / clang-18.1.8 (Bianbu) symmetric · **rvv / VLEN128 / gcc-15.2.0 symmetric** ([CASE-COMPILER-ASYMMETRY] shipped compiler each side).
> **Axis**: **kernel-sym micro (第二赛道)** · [NG-4] NOT e2e · NOT perf-covered · NOT a sealed Win. 永不与 perf-covered 9/83 混算 · 亦不与 matmul kernel-sym≥parity-9 混算（本组=forward-elementwise family，独立桶）.
> **Status**: **DUAL-BOARD DONE** — k1-half DONE + **rvv-half DONE** (9 ops × 8 shapes each). T-CENSUS B类双板收口. §6 = rvv-half + per-board.
> **Casefile**: `experiments/active/g7-census/bclass-forward-ops/` — kernels/, opponent_ggml.cpp, bfwd_micro_driver.c, {k1,rvv}_build.sh, {k1,rvv}_measure.sh, summary_kernel_sym_{k1,rvv}.csv, raw/{k1_run.log,k1_build_seal.txt,rvv_run.log,rvv_build_seal.txt}.

---

## 0. Harness construction (net-new — the only "construct" gap was the harness, not the kernels)

- **OURS** = weft emitted RVV strip kernels, regenerated deterministically from working-tree emitter on the
  canonical front-door fixtures:
  `weft-opt <fixture> --weft-rvv-materialize-forward-elementwise-stream-front-door --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`
  (fixtures = `test/Conversion/RVV/rvv-to-emitc-ggml-forward-elementwise-{add,mul,scale,cpy,silu,gelu,rms-norm,soft-max,rope}.mlir`).
  Compiled clang-18 -O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause (== stock ggml-cpu march).
  - harness-build patch: emitted `gelu.kernel.c` needed `#include <math.h>` added (emitter omits it; `tanhf` undeclared). No emitter/tracked-source change.
- **OPP** = ggml **as-shipped** (ggml-org/llama.cpp @ `/home/kingdom/phdworks/llama.cpp`), two provenance classes:
  - **EXPORTED stock binary** (`/data/k1build-stock/bin/libggml-cpu.so`, md5 871169a0…): `ggml_vec_silu_f32` (T @ 0x76a2c),
    `ggml_vec_soft_max_f32` (T @ 0x76e9a) — linked directly (genuine shipped kernel, zero transcription risk).
  - **Verbatim ggml source** (`opponent_ggml.cpp`, compiled same clang-18 + march): add/mul/cpy/scale/gelu/rms_norm/rope.
    These ggml helpers are `inline static` (nm -D confirms all ABSENT from .so exports) → compiling ggml's own
    source reproduces the as-shipped code path. Source line provenance cited per function in `opponent_ggml.cpp`.
- **ZERO-MODEL numeric gate**: independent fp64 oracle recomputed from the same inputs per op.
- **COLD protocol**: pool of P input+output sets, total footprint ~6 MB >> k1 L2 (512 KiB, NO L3) → every round
  streams each set cold from DRAM. P auto-sized per shape. **HOT** = set-0 reused, best-of-passes.
  Timing = CLOCK_MONOTONIC ns (k1 has NO cache-miss PMU); GB/s = bytes_moved/time. N=12 rounds, relIQR reported.
- **Load-gate**: idle-core picker over cores 0-3 (300 ms /proc/stat delta); require ≥70 % idle; run pinned core-3
  (idle 100 %, gov=performance, 1.6 GHz). Post-run stray check: no bfwd_k1 process left. k1 restored.

---

## 1. ★ k1-half four-column table (anchor n=4096; ratio = ours GB/s ÷ opp GB/s; >1 ⇒ ours faster)

| 算子 | hot | cold | 对手类 [机判·符号级] | 胜负 (cold, parity band ±5%) |
|---|:--:|:--:|---|---|
| **softmax** ★ | 0.821× | **0.822×** | **native-RVV m2**, EXPORTED `ggml_vec_soft_max_f32` (.so). ggml_v_expf poly — **bit-identical to ours** (ours-vs-opp 0 ULP) | **LOSS** (parity-by-adoption; ours −18 % scheduling; compute-bound expf) |
| add | 1.173× | **1.182×** | autovec-RVV **m2** (clang -O3 autovec of ggml scalar loop `z=x+y`) | **WIN** (our emit **m8** vs opp **m2**) |
| mul | 1.191× | **1.184×** | autovec-RVV **m2** (clang autovec of ggml scalar `z=x*y`) | **WIN** (our m8 vs opp m2) |
| scale | 0.918× | 0.935× | **native-RVV m8** (ggml explicit `__riscv_v_intrinsic` vfmul_vf — **byte-identical algo** to ours) | near-parity **LOSS** 0.93× (scheduling, identical algo) |
| cpy | 0.881× | 1.103× | autovec-RVV / memcpy (ggml scalar `y=x`) | **WIN** cold 1.10× / LOSS hot 0.88× (hot↔cold flip) |
| silu | 0.841× | 0.837× | **native-RVV m2**, EXPORTED `ggml_vec_silu_f32` (.so) — **bit-identical to ours** (0 ULP) | **LOSS** 0.84× (parity-by-adoption; −16 % scheduling; compute-bound) |
| gelu | 0.239× | 0.247× | **scalar f16-LUT** (GGML_GELU_FP16 defined @ vec.h:46 → `ggml_table_gelu_f16[t]`) | **LOSS** 0.25× (STRUCTURAL: LUT vs exact tanhf; **ours +2500× accuracy** 3.7e-7 vs 8.5e-4) |
| **rms_norm** | 1.177× | **1.335×** | native-RVV m8 scale + scalar dbl reduce, **2-pass** (memcpy x→y THEN vec_scale) | **WIN** 1.33× (ours **1-pass fusion** read-scale-write vs opp 2-pass) |
| rope | 0.981× | 0.984× | autovec-RVV apply + scalar sinf/cosf, **2-pass** cos/sin cache | **PARITY** 0.98× (bit-exact to as-shipped; compute-bound sinf/cosf) |

**cold tally (anchor)**: 4 WIN (add, mul, cpy, rms_norm) · 1 PARITY (rope) · 4 LOSS (scale, silu, softmax, gelu).

### M=8 shape sweep (cold ratio ours/opp; n=512…16384) — 便车攒 regime
| 算子 | 512 | 1024 | 2048 | 3072 | 4096 | 5120 | 8192 | 16384 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| softmax | 0.906 | 0.837 | 0.826 | 0.826 | 0.822 | 0.821 | 0.818 | 0.815 |
| add | 2.249 | 3.953 | 1.237 | 1.162 | 1.182 | 1.195 | 1.200 | 1.201 |
| mul | 2.267 | 4.029 | 1.235 | 1.183 | 1.184 | 1.187 | 1.183 | 1.193 |
| scale | 0.883 | 0.563 | 0.905 | 0.928 | 0.935 | 0.939 | 0.951 | 0.954 |
| cpy | 1.079 | 1.189 | 1.134 | 1.117 | 1.103 | 1.106 | 1.085 | 1.084 |
| silu | 0.913 | 0.825 | 0.835 | 0.836 | 0.837 | 0.835 | 0.839 | 0.840 |
| gelu | 0.256 | 0.251 | 0.248 | 0.248 | 0.247 | 0.248 | 0.247 | 0.246 |
| rms_norm | 0.976 | 1.409 | 1.328 | 1.338 | 1.335 | 1.341 | 1.424 | 1.310 |
| rope | 1.094 | 0.984 | 0.985 | 0.987 | 0.984 | 0.982 | 0.980 | 0.973 |

> Small-n add/mul/rms_norm outliers (n=512-1024) = opp m2 loop-tail / L2-resident regime noise; the ≥n2048 plateau is the stable signal. Anchor = n=4096.

---

## 2. 对手类机判证据 (符号级·objdump·非手写类目)

- EXPORTED opponents machine-probed present in stock .so: `# OPP_SYM_EXPORTED ggml_vec_silu_f32: T 0x76a2c`, `ggml_vec_soft_max_f32: T 0x76e9a`.
- Inline opponents machine-probed **ABSENT** from .so exports (⇒ correctly compiled from ggml source, not a weaker stand-in): add/mul/scale/cpy/gelu all `absent`.
- Per-symbol objdump class (opponent .o):
  - `opp_ggml_vec_add_f32`: **`vsetvli e32,m2`** + `vl2re32.v`/`vfadd.vv` → clang autovec **RVV m2**.
  - `opp_ggml_vec_scale_f32`: native-RVV (4 rvv insn, ggml explicit m8 branch).
  - `opp_ggml_rms_norm_f32`: native-RVV (9), `opp_ggml_rope_norm_f32`: native-RVV (5, apply autovec + scalar sinf/cosf).
  - `opp_ggml_vec_gelu_f32`: **0 rvv insn** → scalar f16-LUT gather.
- **OURS add/mul**: **`vsetvli e32,m8`** (LMUL=8, widest) — the add/mul win mechanism is objdump-verified: **our emitter's wide-m8 strip vs clang autovec's m2** (fewer loop iterations / better load-compute overlap on the streaming path). Connects to SEL-1 (widest-LMUL) / emitter-maturity wide-LMUL prior.
- **上游更强路径披露**: gelu opponent uses the ggml **f16 LUT** (deployed default) — NOT the tanhf path our kernel emits; this is a structural-algorithm mismatch, not a fair speed A/B (disclosed). softmax/silu opponents are the **exported native-RVV** .so kernels (strongest available), not stubs.

---

## 3. 数值 ULP 逐算子 (ZERO-MODEL·预注册档 [K-5])

| 算子 | 预注册档 | 实测 (all 8 shapes) | 判 |
|---|---|---|---|
| add | **bit-exact 0 ULP** (hard) | maxulp_ours=0, maxulp_ours_vs_opp=0 | ✅ HARD PASS |
| mul | **bit-exact 0 ULP** (hard) | 0 / 0 | ✅ HARD PASS |
| scale | **bit-exact 0 ULP** (hard) | 0 / 0 | ✅ HARD PASS |
| cpy | **bit-exact 0 ULP** (hard) | 0 / 0 | ✅ HARD PASS |
| gelu | ≤ 1.3e-6 (tanhf) | maxrel_ours 3.7e-7 (opp LUT 8.5e-4·f16 caveat, ours-vs-opp ~11000 ULP disclosed) | ✅ PASS (ours) |
| silu | ULP-bounded (expf) | maxulp_ours 2-3, relerr 2.1e-7, **ours-vs-opp 0 ULP** | ✅ PASS |
| softmax | ULP-bounded (expf, max-sub) | maxulp_ours 3, relerr 2.3e-7, **ours-vs-opp 0 ULP** (same ggml_v_expf poly) | ✅ PASS |
| rms_norm | ULP-bounded (rsqrt/Σx²) | maxulp_ours **0** (bit-exact — dbl reduce identical) | ✅ PASS |
| rope | ULP-bounded (sin/cos) | **ours-vs-opp 0 ULP @ all n** (bit-exact to as-shipped); fp64-oracle relerr grows 6e-5→7e-3 with n | ✅ PASS-by-equivalence¹ |

¹ **rope gate note**: the fp64-oracle relerr growth (and the driver's exit-code GATE=FAIL at n≥2048) is the **f32 iterative-θ accumulation** property of the ggml algorithm (θ_i via `θ *= θ_scale` in f32, drifting from the fp64 oracle's power). Ours reproduces ggml **bit-exact** (ours-vs-opp 0 ULP at every n) → kernel correctness is established by equivalence to as-shipped; the oracle divergence is inherent to (and shared with) ggml, not a weft kernel defect.

---

## 4. 净新普查发现 (B 类 near-wall memory-bound 预判 — 部分证实 / 部分证伪)

1. **预判 "多数 parity/near-wall memory-bound" = 部分证伪**. The 4 pure-elementwise ops (add/mul/cpy/scale) ARE memory-bound (cold ~2.3–5.6 GB/s, well below compute roofline), BUT the verdict is **not uniform parity**: our emitter's **wide-m8** emit beats clang's **m2** autovec on add/mul (cold ~1.18×) — a real capability-keyed kernel-efficiency edge on the streaming path, objdump-substantiated.
2. **★ rms_norm structural WIN 1.33× cold** — the standout. Ours fuses copy+scale into a **single memory pass** (read x → ×scale → write y); ggml ships **two passes** (`memcpy(y,x)` THEN `ggml_vec_scale_f32(y)`). On the memory-bound cold path the saved pass = 1.33× — a genuine one-pass-fusion win vs as-shipped (bit-exact). Net-new.
3. **Compute-bound cluster (silu/softmax/gelu/rope) is cache-independent** (hot≈cold): transcendental-dominated (~1 GB/s effective for softmax/silu). Here ours is **parity-by-adoption** on algorithm (silu/softmax emit the exact ggml_v_expf poly → ours-vs-opp 0 ULP) but loses **~16–18 % on scheduling** vs the exported .so (silu 0.84×, softmax 0.82×) — an emitter-scheduling maturity gap, not an algorithm gap.
4. **gelu 0.25× = structural, not a fair speed A/B**: as-shipped ggml uses an **f16 LUT** (GGML_GELU_FP16); ours computes exact tanhf. Ours trades 4× throughput for ~2500× accuracy (3.7e-7 vs 8.5e-4). Report as capability/accuracy trade, disclosed — **not** a kernel-quality loss.
5. **rope ≈ parity 0.98×**: compute-bound sinf/cosf; ours 1-pass vs ggml 2-pass cache nets ~parity (sin/cos cost dominates the saved pass).
6. **Axis hygiene**: this 4-WIN/1-PARITY/4-LOSS cold tally is **kernel-sym micro (第二赛道)** — it does **NOT** transplant to the census's e2e prediction "add/mul deploy 0.76–0.83×" (that is the system/e2e account, memory-bound whole-model with routing). Kernel-sym add/mul 1.18× ≠ e2e add/mul 0.83×.禁互推.

---

## 5. rvv-half PENDING 清单 (harness 复用·rvv 忙 Batch1)

- **rvv (VLEN128, gcc-15.2 对称域)**: 9 ops × 8 shapes, same harness. march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs, KCC=gcc-15 (rvv shipped compiler → [CASE-COMPILER-ASYMMETRY] domain: use gcc-15 both sides). ggml stock = `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin` (per FLAT rvv recipe). Kernels regenerate identically (emit is board-agnostic; VLEN128 vsetvl adapts at runtime). **Expected divergence from k1**: VLEN128 halves the per-vsetvl width → wide-m8 add/mul edge may shrink; scale/silu/softmax native-RVV opponents same class. rms_norm 1-pass fusion win should hold (structural, VLEN-invariant).
- To run: `scp` casefile to rvv, adapt `k1_build.sh` GGML_DIR/MARCH/KCC to rvv (gcc-15), same `k1_measure.sh` load-gate (cores per rvv topology).

---

## 6. ★ rvv-half — VLEN128 · gcc-15.2 symmetric (rvv shipped-compiler domain) · DUAL-BOARD close-out

- **Build**: KCC=`/opt/tcrv-toolchains/gcc-15.2.0/bin/riscv64-unknown-linux-gnu-g++` (GCC 15.2.0, native on-board);
  march=`rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause` (== the
  ggml-cpu gcc15 build's `compile_commands.json` march, both sides) ; mabi=lp64d ; -O3 -ffp-contract=on.
  OURS kernel source **byte-identical to k1** (all 9 md5 match k1 seal — emit is board-agnostic; VLEN128 vsetvl adapts at runtime).
  ggml stock = `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin` (md5 d1adc634…); silu/softmax **EXPORTED** (`ggml_vec_silu_f32`@0x67370, `ggml_vec_soft_max_f32`@0x67646).
  Link needed `-L/opt/tcrv-toolchains/gcc-15.2.0/lib` for gcc-15 libstdc++/libgcc_s (binutils ld path). No tracked-source / emitter change.
- **Measure**: same driver; load-gate idle-core pick over **cores 8-15** (all 100 % idle at run; pinned core-8, gov=performance, 2.6 GHz);
  co-tenant (python/vLLM on other cores) undisturbed. No real stray (`pgrep -x bfwd_rvv`=NONE post-run; the seal's STRAY=2 was a concurrent-ssh path-string false positive). N=12 rounds, relIQR≤~1.4 %.

### 6.1 rvv-half four-column table (anchor n=4096; cold parity band ±5%) + vs k1-half
| 算子 | hot | cold | 对手类 [机判·符号级·gcc-15] | 胜负 (cold) | vs k1-half (cold) |
|---|:--:|:--:|---|---|---|
| **softmax** ★ | 0.853× | **0.854×** | EXPORTED native-RVV `ggml_vec_soft_max_f32` (.so gcc-15); ggml_v_expf poly **bit-identical** (ours-vs-opp 0 ULP) | **LOSS** 0.85× (parity-by-adoption; scheduling) | k1 0.822× LOSS — **both LOSS** (≈, slightly better on rvv) |
| add | 1.163× | **0.983×** | autovec-RVV **e32,m1** (gcc-15 autovec of ggml `z=x+y`) | **PARITY** 0.98× | k1 **WIN 1.18×** → rvv PARITY — **★VLEN-dependent** |
| mul | 1.289× | **0.986×** | autovec-RVV **e32,m1** (gcc-15 autovec of ggml `z=x*y`) | **PARITY** 0.99× | k1 **WIN 1.18×** → rvv PARITY — **★VLEN-dependent** |
| scale | 1.096× | **1.020×** | native-RVV **e32,m8** (ggml explicit `vfmul_vf`, **byte-identical algo**) | **PARITY** 1.02× (marginal edge) | k1 **LOSS 0.935×** → rvv PARITY (flip up; gcc-15 sched) |
| cpy | 1.420× | **0.994×** | autovec-RVV **e8,mf4** (gcc-15 byte-copy of ggml `y=x`) | **PARITY** 0.99× | k1 **WIN 1.10×** → rvv PARITY — VLEN/sched dep |
| silu | 0.770× | **0.762×** | EXPORTED native-RVV `ggml_vec_silu_f32` (.so gcc-15) — **bit-identical** (0 ULP) | **LOSS** 0.76× (parity-by-adoption; scheduling) | k1 0.837× LOSS — **both LOSS** (worse on rvv) |
| gelu | 0.146× | **0.149×** | scalar f16-LUT (0 rvv insn; GGML_GELU_FP16) | **LOSS** 0.15× (STRUCTURAL: LUT vs exact tanhf; **ours +2300× accuracy** 3.8e-7 vs 8.7e-4) | k1 0.247× — **both LOSS** (structural, worse on rvv) |
| **rms_norm** | 1.100× | **1.152×** | native-RVV (m8 scale + e64 dbl reduce), **2-pass** | **WIN** 1.15× (ours **1-pass fusion**; bit-exact) | k1 **WIN 1.335×** — **★both WIN (VLEN-invariant)** |
| rope | 0.985× | **1.009×** | native-RVV **e32,m1** apply + scalar sinf/cosf, 2-pass | **PARITY** 1.01× (bit-exact to as-shipped) | k1 0.984× PARITY — **both PARITY** |

**rvv cold tally (anchor)**: **1 WIN (rms_norm) · 5 PARITY (add, mul, scale, cpy, rope) · 3 LOSS (softmax, silu, gelu)** · ≥parity = **6/9**.
(k1 was 4 WIN / 1 PARITY / 4 LOSS · ≥parity 5/9.)

### 6.2 M=8 cold shape sweep (rvv · ours/opp; n=512…16384)
| 算子 | 512 | 1024 | 2048 | 3072 | 4096 | 5120 | 8192 | 16384 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| softmax | 0.915 | 0.867 | 0.858 | 0.857 | 0.854 | 0.856 | 0.857 | 0.854 |
| add | 0.970 | 1.053 | 1.005 | 0.987 | 0.983 | 0.990 | 0.985 | 0.975 |
| mul | 1.001 | 1.045 | 1.000 | 0.992 | 0.986 | 0.992 | 0.971 | 0.984 |
| scale | 0.981 | 1.048 | 1.034 | 1.024 | 1.020 | 1.018 | 1.013 | 1.007 |
| cpy | 1.164 | 0.969 | 0.985 | 0.991 | 0.994 | 0.995 | 1.000 | 0.996 |
| silu | 0.885 | 0.791 | 0.774 | 0.764 | 0.762 | 0.752 | 0.753 | 0.754 |
| gelu | 0.161 | 0.156 | 0.153 | 0.153 | 0.149 | 0.149 | 0.146 | 0.144 |
| rms_norm | 1.408 | 1.259 | 1.188 | 1.160 | 1.152 | 1.141 | 1.128 | 1.124 |
| rope | 1.014 | 0.995 | 1.002 | 1.005 | 1.009 | 1.017 | 1.029 | 1.045 |

### 6.3 对手类机判 (gcc-15 objdump · 符号级 · 非手写类目)
- EXPORTED opp present in rvv stock .so: silu@0x67370, softmax@0x67646 (strongest available shipped kernels, not stubs).
- Inline opp add/mul/scale/cpy/gelu all **ABSENT** from .so exports ⇒ correctly compiled from ggml source.
- Per-symbol objdump (opp.o, binutils-2.46.1): add/mul=**e32,m1** autovec (6 rvv insn) · scale=**e32,m8** explicit (4) · cpy=**e8,mf4** (4) · gelu=**0 rvv** scalar-LUT · rms_norm=native-RVV mixed (m8 scale + e64/e8 reduce, 20 insn, 2-pass) · rope=**e32,m1** apply + scalar sinf/cosf (4).
- **★ per-board opponent divergence**: on k1 clang-18 autovec'd add/mul at **m2**; on rvv gcc-15 autovec'd them at **m1** (narrower). OURS add/mul emit **e32,m8** on both boards (source-fixed, board-agnostic). So the OURS-vs-opp *width gap* is even wider on rvv (m8-vs-m1), yet the cold ratio is **lower** (0.98× vs k1's 1.18×) → the cold WIN was never about LMUL width per se; it is a VLEN256-specific memory-pipeline effect that vanishes at VLEN128 (both saturate ~8 GB/s DRAM regardless of LMUL — memory-bound).

### 6.4 数值 ULP (ZERO-MODEL · rvv · all 8 shapes)
| 算子 | 预注册档 | rvv 实测 | 判 |
|---|---|---|---|
| add/mul/scale/cpy | bit-exact 0 ULP (hard) | maxulp_ours=0, ours-vs-opp=0 (all 8) | ✅ HARD PASS (== k1) |
| silu | ULP-bounded (expf) | maxulp_ours 3, relerr 2.4e-7, **ours-vs-opp 0** | ✅ PASS |
| softmax | ULP-bounded (expf, max-sub) | maxulp_ours 3, relerr 2.4e-7, **ours-vs-opp 0** | ✅ PASS |
| gelu | ≤1.3e-6 (tanhf) | maxrel_ours 3.8e-7 (opp LUT 8.7e-4; ours-vs-opp ~11635 ULP disclosed f16 caveat) | ✅ PASS (ours) |
| rms_norm | ULP-bounded | maxulp_ours **0** (bit-exact dbl reduce) | ✅ PASS |
| rope | ULP-bounded (sin/cos) | **ours-vs-opp 0 ULP @ all n** (bit-exact to as-shipped); fp64-oracle relerr grows→7e-3, driver GATE=FAIL n≥2048 = ggml f32 iterative-θ property (shared, not weft defect) | ✅ PASS-by-equivalence |

### 6.5 per-board analysis — is the WIN VLEN-dependent?
1. **★ The k1 add/mul/cpy cold WINs are VLEN256-only.** All three collapse to PARITY at VLEN128 (add 1.18→0.98, mul 1.18→0.99, cpy 1.10→0.99). Mechanism: pure-elementwise cold is DRAM-bandwidth-bound (~8 GB/s both sides on rvv); the wide-m8 strip's fewer-iterations advantage only converts to throughput on k1's VLEN256 memory pipeline. **Confirms §5's prediction "wide-m8 edge may shrink"** — it fully evaporates.
2. **★ rms_norm is the ONLY double-board-confirmed kernel-sym WIN** (k1 1.335× / rvv 1.152×). Structural 1-pass-fusion (read→×scale→write) vs ggml 2-pass (memcpy THEN scale); the saved memory pass is VLEN-invariant → holds on both boards (magnitude smaller on rvv). This is the census's durable forward-op finding.
3. **scale flips LOSS→PARITY** (k1 0.935× → rvv 1.020×): identical-algo (both e32,m8 native-RVV), so the delta is pure compiler scheduling — gcc-15 schedules the shared vfmul.vf strip marginally better on rvv than clang-18 did on k1.
4. **Compute-bound cluster (silu/softmax/gelu/rope) same verdict both boards**: silu/softmax parity-by-adoption LOSS (algo bit-identical, ~15–24 % scheduling gap, worse on rvv); gelu structural LUT trade; rope parity. Cache-independent (hot≈cold).
5. **Axis hygiene**: this is **kernel-sym micro (第二赛道), forward-elementwise family** — a separate bucket from the matmul kernel-sym ≥parity-9. Does NOT transplant to e2e/perf-covered. rvv-half ≥parity 6/9 are 5 PARITY + 1 WIN (near-1.0 memory-bound parities are NOT hand-brick beats; the single genuine cross-board WIN is rms_norm's structural fusion).

### 6.6 kernel-sym counting recommendation (honest)
- **Report as a distinct forward-op kernel-sym family**, dual-board, NOT merged into the matmul kernel-sym-9 (different op family, different opponent nature = as-shipped forward kernels, machine-judged, not hand-brick).
- **Both-board ≥parity intersection = 5** {add, mul, cpy, rms_norm, rope}; **both-board WIN intersection = 1** {rms_norm}.
- **rvv-half new ≥parity格 = 6/9** (add, mul, scale, cpy, rms_norm, rope); of the 4 k1 WINs, only **rms_norm** is also a WIN on rvv → the double-board-confirmed forward-op kernel-sym WIN count = **1 (rms_norm, VLEN-invariant structural fusion)**. add/mul/cpy are **PARITY-not-WIN on rvv** (their k1 WIN is VLEN256-scoped; do NOT count as dual-board wins).
