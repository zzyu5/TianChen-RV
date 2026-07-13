# [G7-L3 货架B] q8_0 @k1 e2e — dual-board 成色（repack-approach vs block-dot）

> Board `ssh k1` (SpacemiT X60 / VLEN256 / stock clang-18 / DVFS 1.6GHz locked).
> Model `/data/tinyllama-q8_0.gguf` (llama 1B Q8_0, 1.09 GiB).
> **NO git · board reversible (私有 repack.cpp copy·shared source NEVER edited) · correctness-first · kernel==system (clang-18 对称).**
> 方法 = repack-vs-block-dot 对照（[WORK-ITEM-K1-KQUANT-E2E] q4_K 已验证 k1 方法）。

## 1. 路由确认
- k1 stock `ggml_repack_get_optimal_repack_type`：`GGML_TYPE_Q8_0` → riscv `case 256: { if (ne[1]%16==0) return &q8_0_16x1_q8_0; }`（**k1 stock 出货 16x1 repack**·**case128 = break//TODO** → q8_0 repack VLEN256-only·与 rvv q8_0 VLEN128 破损对称）。
- objdump `ggml_gemm_q8_0_16x1_q8_0`（REPACK lib）vset = **e32,m2 ×2 · e16,m1 ×4 · e8,mf2 ×4** = VLEN256-native full-width（healthy）。
- 对手 = stock generic block-dot `ggml_vec_dot_q8_0_q8_0`。non-SELF·as-shipped。

## 2. 部署五验 + 反向控制
- ① ONE 真 ELF llama-bench·physical .so swap（LD_LIBRARY_PATH=build-k1-flat/bin 生效）。
- ② REPACK md5=`871169a0…`（== 出货 stock）· VECDOT md5=`bc195e3b577ce88d30033cd3572a7d69`（q8_0 repack return→break·私有 copy 重编重链）· ON≠OFF。
- ③ 反向控制真熄灭：VECDOT dispatch nullptr → block-dot·行为证 = prefill 10.68→4.55 t/s（2.35× 时间分离·swap 真生效）。
- ④ objdump vl seal：VLEN256-native e32m2（healthy）。
- ⑤ 对手 = stock block-dot（非 SELF）。⑥ clang-18 双侧对称（唯一 diff = 1 dispatch return）。

## 3. correctness GREEN（greedy A==B·真模型·MIRAGE 排除）
- 4 prompt greedy·A(REPACK)==B(VECDOT)：**4/4 completion byte-identical**（"…is Paris." / 乡村故事 / "…is 4." / idiom）·all coherent fluent English·NO NaN/garbage → **correctness_green = TRUE**（比 q4_0 更干净·0 divergence·q8_0 整数点积 associative-exact）。

## 4. 分相 e2e（phase-split paired A/B·n=12/side·1.6GHz·-t4·pin 0-3·interleaved）

| phase | A REPACK (16x1 repack) | B VECDOT (block-dot) | **REPACK/VECDOT** | relIQR (A/B) | n | freq |
|---|---|---|---|---|---|---|
| **prefill pp128** | **10.6846** t/s | 4.548975 t/s | **2.3488×** | 0.13% / 0.03% | 12 | 1.6GHz |
| decode tg32 | 4.48411 t/s | 3.69908 t/s | **1.2122×** | 0.23% / 0.29% | 12 | 1.6GHz |

relIQR ≪ 噪声地板 ⇒ rock-solid。**两相皆 WIN**。

## 5. 双账本 + 预注册出口
- **双账本**：clang-18 双侧 ⇒ kernel账 == system账（CONVERGE·同数）。[CASE-COMPILER-ASYMMETRY] not triggered。
- **预注册出口 = ≥parity（prefill 2.3488×）→ 双板成色**：q8_0 rvv 绿（perf-covered·correctness-carrier 4.35×/3.81× deployed·`q4-0-e2e-is-routing-not-kernel`）+ **k1 绿（repack-approach 2.35× prefill / 1.21× decode vs block-dot）= dual-board**。decode 亦 WIN（未回退·"decode→黄"不触发）。

## 6. ★诚实 scope
- e2e WINNER = stock 自己 clang-编译的 q8_0 16x1 repack（k1 as-shipped 默认·case256 ON），非我方 compiler-emitted kernel ⇒ **不新增 perf-covered green**（同 q4_K workitem note）。q8_0 perf-covered 绿仍立于 rvv 登记。
- 价值 = dual-board 成色（repack approach e2e 在 k1-clang 传导 ≥parity）+ margin 量化。our-emit↔stock-repack kernel-axis parity 已封（T9 `q8_0@k1 +4.37% item4`）·传递链 our-emit ≈ stock-repack ≈ 本 2.35× e2e margin。
- 2.35× = repack vs 未优化 stock block-dot（弱对手·L1 path-win 类）·非 tuned-kernel 对比。kernel+e2e / prefill+decode 永分报。

## 7. board restored
- 共享源 md5 = `3cac40aa…`（NEVER edited）· 出货 stock lib `871169a0…`（UNTOUCHED）· `/data/build-k1-flat` 已删 · /tmp/flat* 已清 · 0 leftover procs。（双证见 `../board_restore_proof.txt`）

## durable files
- `evidence.md` · `build_seal_raw.txt` · `measure_raw.txt`（correctness + 4×###AB n=12 JSON）· `gen_texts/`（A/B·4/4 identical）。
