# G8 §六.3 · q5_0/q5_1@k1 e2e A/B — REDESIGN-B repack-GEVM decode transduction (2026-07-15)

> **任务**：REDESIGN-B repack-GEVM leaf（selector-fix `d109d6ed2`·kernel-axis cold **1.76×(q5_0)/2.24×(q5_1)** vs stock native-vec byte-exact）的 decode 算力赢，**transduce 到真 llama e2e 吗？** dual-build-tree A/B（build-A=weft REDESIGN-B 核 / build-B=stock，仅 q5 decode 核差异）+ 部署五验 + [CASE-MICRO-E2E] 如实报。
> **Board**：k1 / SpacemiT X60 / VLEN256 / clang-18.1.8 Bianbu `-fno-integrated-as` / gov=performance 1.6GHz / pin core0-3 -t4。

## ★VERDICT: **TRANSDUCE-GREEN (decode)** — 非 washout

**q5_0 decode（tg, M=1）weft/stock = 1.97×**；**q5_1 decode = 2.07×**。**★与 [CASE-MICRO-E2E] 默认预期（decode memory-bound → 算力赢 wash out）相反**：根因 = **stock q5_0/q5_1 block-dot decode 在 X60 上是 compute-bound**（5th-bit qh 处理无好 SIMD 路 → stock tg 仅 ~3.29 t/s），故我方 VLEN256 向量化 repack-GEVM 的算力赢**真传导**。判别键 = **对手基线的 bound 类型**：q4_0（stock 有好 SIMD block-dot → decode memory-bound → G5-M2 washout 0.8165×）vs q5_0/q5_1（stock 弱 → decode compute-bound → 传导）。kernel-axis 1.76× cold → e2e decode 1.97×（不仅未 wash，反而**放大**，因 e2e 冷态 + 全模型 tensor 覆盖）。

| 格 | kernel-axis cold (K2048N512) | **e2e decode tg weft/stock** | e2e prefill pp weft/stock | 判定 |
|---|--:|--:|--:|:--|
| q5_0 | 1.760× | **1.970×** (n=6·1.967–1.974) | 2.212× (n=4·2.210–2.214) | **TRANSDUCE** |
| q5_1 | 2.241× | **2.073×** (n=6·2.046–2.091) | 2.344× | **TRANSDUCE** |

**abs t/s（core0-3·-t4·gov 1.6GHz）**：q5_0 decode 3.287→6.480 / prefill 3.835→8.481；q5_1 decode 3.091→6.406 / prefill 3.651→8.559（stock→weft）。

---

## 1. 部署五验（deployed==proven）

| # | 门 | q5_0 | q5_1 | 证据 |
|---|---|:--:|:--:|---|
| ① | banner/deployed==proven | PASS | PASS | build-A 跑 llama-bench，stderr 打 `TCRV G7-L3 EMITTED GEVM(q5_{0,1}_16x1 VLEN256 compiler-emitted vl=16) ENGAGED`（decode）+ `EMITTED GEMM(...)`（prefill）·两格均见 |
| ② | byte-exact / 输出一致 | PASS | PASS | greedy(temp0 seed42) 生成 **byte-identical** STOCK==WEFT（去控制符后 `cmp` 全等·q5_1 32-tok argmax 未因 5.6e-4 FMA 噪声翻）；kernel-axis 独立 oracle q5_0 0/512 bit-exact / q5_1 0/512 mism |
| ③ | 真路由 | PASS | PASS | GEVM banner 只在 decode 打、GEMM banner 只在 prefill 打；`nm` WEFT 有 `weft_emitc_ggml_vec_dot_q5_{0,1}_kernel`+`ggml_gemv_q5_{0,1}_16x1`，STOCK **无**；`objdump` emitted GEVM 含 `vlm.v`(mask-load)+`vwmacc.vx` = REDESIGN-B 指纹（q5_0 / q5_1 均验） |
| ④ | 反向控制 | PASS | PASS* | q5_0 CTRL build（GEVM gate `==256`→`==257`，decode leaf OFF）：GEVM banner **消失**(0)、GEMM banner 仍在(1)、decode 从 weft **6.48 塌到 1.515 t/s**（routes to generic-scalar·比 stock block-dot 3.29 还慢=证 leaf 是唯一算力源）、输出仍正确。q5_1(*)=同一 VLEN-gated engage 机制（未单独 rebuild）+ STOCK=scaffold-absent 反向控制（无 banner→block-dot） |
| ⑤ | 对手真实 | PASS | PASS | `nm` STOCK 派发真 `ggml_vec_dot_q5_0_q8_0` @0x9f83c / `q5_1_q8_1` native-RVV block-dot（非稻草人）；我方 q5 符号在 STOCK 全缺 |

## 2. dual-build-tree（仅 q5 decode 核差异）

- **build-B (STOCK)** = 当前 baseline 源（`repack.cpp`=3cac40aa/`repack.h`=57851439/`arch/riscv/repack.cpp`=c3c101fd）forced-rebuild → `libggml-cpu.so` md5 **14b6add6**（= 预存 baseline，确定性复现）。q5_0/q5_1 无 repack scaffold → decode 走 stock block-dot。
- **build-A (WEFT)** = build-B 源 + q5 deploy patch（净新 upstream repack scaffold：`block_q5_0x16` struct + `make_block_q5_0x16` interleaver + `repack<block_q5_0,1,16>` + dispatch 拦截 + `ggml_gemv_q5_0_16x1_q8_0` `#include weft_emitted_gevm_q5_0.inc`）→ md5 **df88afa3**。
  - **部署核 = REDESIGN-B leaf**（casefile `../k1-q5-selector-fix/kernels/q5_0_gevm_frontdoor_vlen256.c`·`vfmv_v_f_f32m2(0.0f,16)` half_lanes=16 whole-strip·qs@32/qh_lo@288/qh_hi@320 与 make_block 一致·`vlm.v`+`vmnand`+masked-`vsub` mask 路，**非** OLD G7-L3 bit-scatter 路）。★board 上 `/data/g7q50/weft_emitted_gevm_q5_0.inc`（G7-L3 派发）与 REDESIGN-B 二进制 diff 证实 qh 路不同（bit-scatter vs vlm-mask），本役**替换成 REDESIGN-B**再 build → objdump `vlm.v` 指纹坐实。GEMM 用 G7-L3 emitted GEMM（selector-fix 是 decode-only；prefill 仍 G7-L3 GEMM）。
  - ★**banner 措辞诚实**：banner 字串写 "G7-L3 EMITTED GEVM"（deploy-patch 遗留 label），但**编译进 .so 的核是 REDESIGN-B**（objdump `vlm.v`/`vmnand`/masked-`vsub` 坐实·非 G7-L3 bit-scatter）——label≠core，以 objdump 为准。
- A/B = 交替把 STOCK/WEFT `.so` 拷入 `/data/k1build/bin/`（llama-bench 动态链），paired interleaved（吸收 co-tenant drift）。

## 3. A/B 测量（interleaved paired · --no-warmup · -t4 pin core0-3）

### q5_0 decode（tg40, M=1·6 rounds interleaved）
| round | stock t/s | weft t/s | ratio |
|--:|--:|--:|--:|
| r1 | 3.287 | 6.471 | 1.968 |
| r2 | 3.293 | 6.479 | 1.967 |
| r3 | 3.287 | 6.478 | 1.971 |
| r4 | 3.293 | 6.482 | 1.969 |
| r5 | 3.284 | 6.483 | 1.974 |
| r6 | 3.287 | 6.484 | 1.973 |
| **median** | **3.287** | **6.480** | **1.970** |

stddev_ts per bench ~0.003–0.02（极紧）；paired range 1.967–1.974（<0.4% 抖动）。

### q5_0 prefill（pp96, M=96·4 rounds interleaved）
| round | stock t/s | weft t/s | ratio |
|--:|--:|--:|--:|
| r1 | 3.829 | 8.475 | 2.214 |
| r2 | 3.836 | 8.482 | 2.211 |
| r3 | 3.838 | 8.483 | 2.210 |
| r4 | 3.837 | 8.485 | 2.212 |
| **mean** | **3.835** | **8.481** | **2.212** |

### q5_1 decode（tg40, M=1·6 rounds interleaved）
| round | stock t/s | weft t/s | ratio |
|--:|--:|--:|--:|
| r1 | 3.086 | 6.452 | 2.091 |
| r2 | 3.103 | 6.405 | 2.064 |
| r3 | 3.091 | 6.421 | 2.078 |
| r4 | 3.084 | 6.413 | 2.080 |
| r5 | 3.089 | 6.321 | 2.046 |
| r6 | 3.092 | 6.422 | 2.077 |
| **mean** | **3.091** | **6.406** | **2.073** (min 2.046 max 2.091) |

### q5_1 prefill（pp96·4 rounds interleaved）
| round | stock t/s | weft t/s | ratio |
|--:|--:|--:|--:|
| r1 | 3.653 | 8.557 | 2.343 |
| r2 | 3.651 | 8.521 | 2.334 |
| r3 | 3.648 | 8.634 | 2.367 |
| r4 | 3.653 | 8.525 | 2.334 |
| **mean** | **3.651** | **8.559** | **2.344** |

**bonus 内嵌信号**（llama-cli greedy 打印）：q5_1 STOCK Gen 3.2 t/s / Prompt 3.6 → WEFT Gen 6.5 / Prompt 8.8（≈2.03×/2.44×，与 llama-bench 一致）。

## 4. Transduction 对账（[CASE-MICRO-E2E] aware）

- **kernel-axis cold 1.76×(q5_0)/2.24×(q5_1) → e2e decode 1.97×/2.07× = TRANSDUCE-GREEN**（未 wash；q5_0 甚至**放大** e2e>kernel）。相对 kernel-axis 排序保持（q5_1>q5_0），是真信号非噪声。
- **★为何不 wash（与 q4_0 G5-M2 decode washout 0.8165× 相反）**：stock q5_0/q5_1 decode = 真 `ggml_vec_dot_q5_{0,1}` native-RVV block-dot（nm 证），但第 5 位(qh) 逐-bit scatter 使该 kernel **compute-bound**（stock decode 仅 3.29(q5_0)/3.09(q5_1) t/s ≪ 内存带宽上限 ~13 t/s）。我方 repack-GEVM 把 qh 变 `vlm` mask-load + 全宽 VLEN256 `vwmacc` → 解除算力瓶颈 → decode 真赢。**判别键 = 对手 baseline 的 bound 类型（compute vs memory），非 format 名**：q4_0（stock 有好 SIMD block-dot → decode memory-bound → 算力赢 wash）vs q5_0/q5_1（stock 弱 5th-bit 路 → decode compute-bound → 传导）。
- **prefill 亦传导**（compute-amortized·我方 repacked GEMM vs stock block-dot）：q5_0 2.21×·q5_1 2.344×。
- **编译器对称（no [CASE-COMPILER-ASYMMETRY] artifact）**：build-A/build-B **均 clang-18.1.8**（k1 出货编译器=clang-18·与部署一致）·仅 q5 patch 差异 → 对称合法赢。
- **诚实边界**：① 这是**净新 path 赢**（stock 零 q5 repack·我方建整条 scaffold + correctness-carrier + leaf），非 q4_0 式"白嫖 routing gate"；模型 100% q5_0/q5_1 → e2e 数真实覆盖全 matmul。② "赢"的一半来自 stock q5 decode 路**本就弱**（compute-bound native-RVV）——是"补 stock 缺的向量化"而非"超越已优化对手"；对手是真出货 kernel、非稻草人，故赢合法，但成色 = beat-weak-baseline，非 beat-hand-tuned。

## 5. 板卫生

- k1/X60/VLEN256；pin core0-3 -t4；gov=performance 1.6GHz（**只读未改**）。四 `.so` md5：STOCK **14b6add6**（baseline-rebuilt·= 预存 baseline 确定性复现）· WEFT_Q50 **df88afa3** · WEFT_Q51 **43569a46** · CTRL_Q50 **29c3ddc1**。历史 stock `/data/k1build-stock`=871169a0（只读未动）。
- A/B loadavg：多为本 bench 自身（392% pin 4 核 + co-tenant≈idle，running-tasks 恒 1-2/～460）；paired interleaved 吸收 drift·ratio 抖动 <0.4%(q5_0 decode)/<1.5%(q5_1) 证 clean。
- scratch=`/tmp/g8q5e2e`（.so + logs·用完清）；deploy patch 已存 `/data/g7q5{0,1}`；**源树用完 restore 到 baseline**（3cac40aa/57851439/c3c101fd）+ 移除 arch dir 4 个 staged `.inc`；主树/build/governor 未动；未 git commit；未改 T3/T8/perf-covered（主会整合）。
- ★纠偏记录：首轮 perf 曾被一个未死的 llama-cli（`/dev/null` EOF 自旋）偷占 core0-3，已 kill；正式 A/B 在 clean board 重测。
