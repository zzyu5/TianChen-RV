# G8 §六.3 · q5_0/q5_1@k1 e2e 可行性判定 (2026-07-15 · feasibility-only)

> 任务：判定 q5_0/q5_1@k1 VLEN256 decode 的 REDESIGN-B repack-GEVM leaf（selector-fix d109d6ed2·kernel-axis cold 1.76/2.24× byte-exact）能否 transduce 到真 e2e A/B。只做可行性，不跑完整 A/B。
> Board：k1 / SpacemiT X60 / VLEN256 / bianbu 6.6.63。只读勘察，未建 scratch，主树/build/governor 未动。

## VERDICT: **FEASIBLE**（可做 dual-build A/B · deploy-patch 路 · banner-provable）

三轴全通，无结构性 null 卡点：

### 1. 模型（无需重量化）
- `/data/g7q50/tinyllama-q5_0.gguf` sha256 `2828d768…b071d4`（767 MB, q5_0）
- `/data/g7q51/tinyllama-q5_1.gguf` sha256 `b5a90357…1bf7f9`（831 MB, q5_1）
- 两个真 q5_0/q5_1 tinyllama 均已在板，`llama-quantize` 亦在（`/data/k1build/bin/llama-quantize`）备用。**模型轴 = 满足**。

### 2. 路由（stock block-dot · deploy-patch 建净新 repack scaffold 接线）
- **Stock/baseline 路由**：`ggml_repack_get_optimal_repack_type`（`repack.cpp:4528`）有 q4_0/q4_K/q5_K/q6_K/q2_K/iq4_nl/q8_0 的 riscv `__riscv_zvfh` case，**无 Q5_0/Q5_1**。→ stock q5_0/q5_1 tensor 返回 nullptr（不 repack）→ decode 走标准 block-dot（`mul_mat_one_chunk` → `ggml_vec_dot_q5_0_q8_0`）。**不经 llamafile_sgemm 绕过**（q5_0 M=1 非 sgemm-eligible）。→ stock 下我方核**不被触及**（这正是需要 deploy-patch 的原因，非 null）。
- **接线机制 = deploy patch 建净新 repack scaffold**（`/data/g7q50/deploy_patch_q5_0_k1.py`, 353 行, G7-L3 port·q5_1 有对应 `/data/g7q51/deploy_patch_q5_1_k1.py`）：
  1. `repack.h`：新增 `block_q5_0x16{ d[16]; qs[256]; qh[64]; }`（stride 352·第 5 位 qh 无法用 `block<K,N>` 模板）。
  2. `repack.cpp`：`make_block_q5_0x16` transposed-qh interleaver（correctness-critical·UT byte-exact）+ `template<> repack<block_q5_0,1,16>` + trait 实例 `q5_0_16x1_q8_0`。
  3. **路由拦截**（patch line 249-255）：`get_optimal_repack_type` 追加 `else if (cur->type == GGML_TYPE_Q5_0){ … case 256: if(ne[1]%16==0) return &q5_0_16x1_q8_0; }` —— 与已在树的 q4_0 riscv case（`repack.cpp:4590` `case 256: return &q4_0_16x1_q8_0`）**同构**。
  4. `arch/riscv/repack.cpp`：`ggml_gemv_q5_0_16x1_q8_0`（VLEN256-gated·`#include weft_emitted_gevm_q5_0.inc`）+ `ggml_gemm_…`（`weft_emitted_gemm_q5_0.inc`）。
- **decode(M=1) 落点**：patch generic line 131 `assert(nr == 1)` = GEVM 路；repacked q5_0 tensor 的 mul_mat forward 在 nr==1 时调 `ggml_gemv_q5_0_16x1_q8_0` → **REDESIGN-B repack-GEVM leaf**。**banner 可证**：patch line 300 `fprintf(stderr, "TCRV G7-L3 EMITTED GEVM(q5_0_16x1 VLEN256 compiler-emitted vl=16) ENGAGED …")`（gate ③ engage 探针·prefill 对应 line 333 GEMM banner）。
- **deploy_method = deploy patch（净新 repack scaffold）·非 vec_dot 直换**：REDESIGN-B leaf 吃 repacked `block_q5_0x16` 权重，必经 repack dispatch，不能靠替换 `ggml_vec_dot_q5_0_q8_0`（那是 block-dot 次路）接线。

### 3. 接点就绪度 + 先例
- **树在精确 baseline**：当前 3 目标文件 md5 **完全等于** deploy patch 期望 base（`repack.cpp`=3cac40aa / `repack.h`=57851439 / `arch/riscv/repack.cpp`=c3c101fd）→ patch **clean apply**。
- **同构先例已 live**：树内 q4_0/q4_K/q8_0/q2_K/iq4_nl 净新 repack scaffold 均带 `TCRV EMITTED GEMV … ENGAGED` banner 在跑，同一 dispatch 模式。
- **8-门 e2e 先例**：G5-M2（`docs/reports/2026-07-12-perf-covered-q5_0-green-4of84.md`·rvv 板）已用同一 12-piece scaffold 跑通全 8 门真模型 decode（banner engage✓·correctness GREEN✓·对手=stock block-dot✓）。k1 patch = 该 scaffold 的 VLEN256 port。

## ★诚实预判（[CASE-MICRO-E2E]·decode memory-bound·A/B 会如实出数不粉饰）
- **decode 大概率 wash/regress，非 clean win**：G5-M2 同 scaffold（OLD GEVM·kernel micro ~1.23×）实测 **decode tg32 = 0.8165×（−18% regression）**——kernel 算力赢被 memory-wall 洗掉。REDESIGN-B kernel-axis cold 更高（1.76/2.24× @K2048N512），但 decode 仍 memory-bound，e2e decode 仍很可能 <parity。A/B 可跑、出的是真数（赢或 wash 都如实报），可行性 ≠ 预判赢。
- **prefill(GEMM) 才是 G5-M2 的传导赢面**（1.2097×·compute-amortized）；k1 patch 的 `ggml_gemm_q5_0_16x1_q8_0`（banner line 333）覆盖此轴。
- **A/B 执行细节（非可行性卡点）**：`/data/g7q50/weft_emitted_gevm_q5_0.inc` 是 OLD G7-L3 发射；要部署 REDESIGN-B leaf（commit 40c21de0 + d109d6ed2）需从当前 HEAD weft-opt 重生成 .inc（VLEN256 half_lanes=16 whole-strip 形）再跑 deploy patch。接线机制不变。

## 板卫生
- k1/X60/VLEN256；只读勘察（未 taskset pin，未跑 timed region）；loadavg 会话内 2.04→2.61（co-tenant baseline·未跑负载）。
- stock `.so` md5 记录（只读）：`/data/k1build-stock/bin/libggml-cpu.so` = **871169a0123139692177468b3c8578be**（与 k1-q5-selector-fix evidence 一致）；`/data/k1build/bin/libggml-cpu.so` = 14b6add6…。
- 未建 /tmp scratch（无需清）；未 git commit；未改 T3/T8；主树/build/governor 未动。
