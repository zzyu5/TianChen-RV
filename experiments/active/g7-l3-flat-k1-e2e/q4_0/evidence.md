# [G7-L3 货架B] q4_0 @k1 e2e — dual-board 成色（repack-approach vs block-dot）

> Board `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / **stock clang-18** / DVFS perf-gov 1.6GHz locked).
> Model `/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (llama 1B Q4_0, 636 MiB, sha256 `da3087fb14aede55…`).
> **NO git · board reversible (private repack.cpp copy·shared source NEVER edited) · correctness-first · kernel==system (clang-18 compiler-symmetric).**
> 方法 = **repack-vs-block-dot 对照**（[WORK-ITEM-K1-KQUANT-E2E] q4_K 已验证 k1 方法·measures the repack APPROACH·非 our-emitted 部署）。

## 1. 路由确认（objdump / dispatch）
- k1 stock `ggml_repack_get_optimal_repack_type`：`GGML_TYPE_Q4_0` → riscv `case 256: { if (ne[1]%16==0) return &q4_0_16x1_q8_0; }`（**k1 stock 出货 16x1 repack**·case256 ON·tinyllama dims 2048/5632 均 %16==0 → 路由命中）。
- objdump `ggml_gemm_q4_0_16x1_q8_0`（REPACK lib）vset forms = **e32,m2 ×2 · e16,m1 ×2 · e8,mf2 ×1** = VLEN256-native full-width（healthy·非 MIRAGE half-width）。
- 对手身份 = stock 自己的 generic block-dot `ggml_vec_dot_q4_0_q8_0`（B=VECDOT·repack dispatch 关 → block-dot fallback）。non-SELF·as-shipped 参考路径。

## 2. 部署五验 + 反向控制
- ① same-tree physical .so swap：ONE 真 ELF `/data/k1build/bin/llama-bench`（wrapper 绕过·LD_LIBRARY_PATH=build-k1-flat/bin 生效）。
- ② REPACK md5=`871169a0…`（== 出货 stock·bit-for-bit）· VECDOT md5=`1f3a36a80c35993ce3d5e6050361e723`（repack return→break·私有 copy 重编 repack.cpp.o 重链）· **ON≠OFF**。
- ③ 反向控制引擎真熄灭：VECDOT 变体 dispatch 返回 nullptr → repack 不选中 → block-dot。行为证 = prefill 24.38→4.71 t/s（5.18× 时间分离·swap 真生效·非 stock-vs-stock artifact）。
- ④ objdump vl seal：repack kernel VLEN256-native e32m2（healthy）。
- ⑤ 对手 = stock block-dot（非 SELF·非 hand-brick）。
- ⑥ clang-18 双侧对称（唯一 diff = 1 dispatch return）。

## 3. correctness GREEN（greedy A==B·真模型·MIRAGE 排除）
- 4 prompt greedy·A(REPACK)==B(VECDOT)：**3/4 completion byte-identical**（"…is Paris." / Lily story / "…is 4."）；1 divergence（prompt4 "…American English. It'" vs "…It is"）= -n24 边界处 fp-summation-order near-tie between two distinct correct kernels·**both coherent fluent English·NO NaN/garbage**（≠ VLEN128-MIRAGE PPL-822057 class）。
- 同 [WORK-ITEM-K1-KQUANT-E2E] q4_K 判据（3/4 identical + 1 fp-order near-tie）→ **correctness_green = TRUE**（两变体皆正确·q4_0 整数点积 associative-exact·跨块 fp scale 顺序差致偶发 near-tie）。

## 4. 分相 e2e（phase-split paired A/B·n=12/side·1.6GHz·-t4·pin 0-3·interleaved）

| phase | A REPACK (16x1 repack) | B VECDOT (block-dot) | **REPACK/VECDOT** | relIQR (A/B) | n | freq |
|---|---|---|---|---|---|---|
| **prefill pp128** | **24.381** t/s | 4.708715 t/s | **5.1778×** | 0.37% / 0.04% | 12 | 1.6GHz |
| decode tg32 | 6.51703 t/s | 3.92877 t/s | **1.6588×** | 0.38% / 0.12% | 12 | 1.6GHz |

relIQR ≪ 噪声地板 ⇒ ratio rock-solid。**两相皆 WIN（well above parity）**。

## 5. 双账本 + 预注册出口
- **双账本**：k1 kernel-axis compiler = clang-18；system/deploy compiler = clang-18 ⇒ **kernel账 == system账（CONVERGE·同数）**。[CASE-COMPILER-ASYMMETRY] not triggered（clang-18 双侧对称）。
- **预注册出口 = ≥parity（prefill 5.1778×）→ 双板成色**：q4_0 rvv 绿（perf-covered·routing 5.9×·`q4-0-e2e-is-routing-not-kernel`）+ **k1 绿（repack-approach 5.18× prefill / 1.66× decode vs block-dot）= dual-board**。decode 亦 WIN（未回退·预注册"decode→黄-物理墙"不触发）。

## 6. ★诚实 scope（关键·禁磁量误读）
- e2e WINNER = **stock 自己 clang-编译的 q4_0 16x1 repack**（k1 as-shipped 默认路径·case256 ON），**非我方 compiler-emitted kernel**。⇒ 本测**不新增** perf-covered green（同 [WORK-ITEM-K1-KQUANT-E2E] q4_K note）。q4_0 的 perf-covered 绿仍立于 rvv 登记。
- 本测价值 = **dual-board 成色（repack approach e2e 在 k1-clang 传导 ≥parity·双板证据）** + margin 量化。our-emitted ↔ stock-repack 的 kernel-axis parity 已封（T9 `q4_0@k1-gemm-prefill 1.0022× PARITY`），传递链：our-emit ≈ stock-repack ≈ 本 5.18× e2e margin over block-dot。
- 大倍数解读：5.18× = repack vs **未优化 stock block-dot**（弱对手·L1 path-win 类·同 rvv q4_0 WinB 5.9×），**非**"比调优内核快 5×"。
- kernel+e2e / prefill+decode 永分报（[[kernel-wins-dont-transplant-to-e2e]]）。

## 7. board restored
- 共享源 `repack.cpp` md5 = `3cac40aa…`（NEVER edited·仅私有 copy 编译）· 出货 stock lib `871169a0…`（UNTOUCHED）· live-flat lib 回 REPACK(=stock 871169a0)· 0 leftover llama procs。
- 专用 build dir `/data/build-k1-flat`（seeded from k1build-stock·main build/ 未碰·测毕可删）。

## durable files
- `evidence.md`（本文）· `build_seal_raw.txt`（build+私有copy+seal 全 log·REPACK/VECDOT md5）· `measure_raw.txt`（correctness + 4×###AB n=12 JSON 全量）· `correctness_clean.txt`（清洗）· `gen_texts/`（A/B completion raw·3/4 identical 证据）。
