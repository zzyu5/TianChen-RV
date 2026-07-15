# G8 全景对账报告 · kernel-unit 口径（2026-07-15）

> **口径**：唯一计数单位 = **kernel（算子×格式）**，板是属性列（冻结整顿令一）。全部数字 recon 机算可溯源（`recon_kernel_unit.py` join `[COV-1] 83` + `T3_A(rvv)` + `T3_B(k1)`），禁手写小计。**80 个 distinct kernel**（[COV-1] 83 cell = 80 op×format + 3 个 engine 变体折叠）。
> **状态**：阶段三攻坚已冻结，本报告交付即 idle 等裁决。

---

## 执行摘要（十行内讲清全局）

1. **80 个 kernel，零悬空**：进 0.8 硬门对局 **42**、DEQ 分账 **9**、gelu 数值档例外 **1**、内部格式(q1_0) **1**、**perf 账但从未进 0.8 census 的 27**（= 本役最大发现的覆盖缺口）。
2. **硬门双板战况（首次按正确单位）**：**rvv PASS 29 / 有对局 42**；**k1 PASS 20 / 有对局 27**；**双板均有对局 27，双板全 PASS 19**。
3. **成色诚实（核心结论不变）**：真·硬碰硬赢强手调 hand-brick = **2**（q4_K / q2_K @k1，byte-verified vs 真 16x1 RVV hand-brick）；其余 PASS 多为 vs 弱对手 / cross-op block-dot / near-parity / 对手 immaturity——**计数 ≠ 强赢**。
4. **本役最强正面结果**：C1 能力键控选择 **DEPLOYED + 独立验证**（q5@k1 per-format measured-gate 修）；且该赢 **传导到真 e2e 2×**（q5@k1 decode 1.97/2.07×，独立复验），带出新 C3′ 判别键（对手 bound-type）。
5. **旧口径全废**：30/28/19+17、per-board 20/28 等「格点」数作废，一律溯 kernel 主键；9/83（perf 系统账）、84/91（M4 结构覆盖）保留但**换轴不丢结论**（见 Q8 映射表）。
6. **三件 canon 待用户裁**：q5x shape-gating 方法学、DEQ display/weighting、perf-covered k1-engine 扩展。

---

## Q1 · 83 kernel 完整去向（kernel-level·零悬空·Σ=80）

| 去向筐 | kernel 数 | 说明 |
|---|--:|---|
| **进硬门对局**（0.8 kernel-sym·≥1 板有对局） | **42** | matmul(vec_dot+gemm) + forward-op + rvv-auto-promoted dequant |
| **DEQ-分账**（照测·auto-promote-if-真向量对手） | **9** | = 3 measured-scalar-opp(`dequantize_row\|iq2_s/iq2_xs/nvfp4`·双板 DEQ-AXIS) + 6 FLAT/q1_0 dequant(`q4_0/q4_1/q5_0/q5_1/q8_0/q1_0`·[COV-1] 物理墙·**未入我 census·照测-pending**) |
| **例外-数值档挂起**（gelu） | **1** | 对手 f16-LUT·数值档不对等·不计 PASS/认输（Q6 悬案） |
| **内部格式**（q1_0） | **1** | TianChen 内部实验格式·无真 ggml 向量对手 |
| **perf 账·未入 0.8 census**（覆盖缺口） | **27** | 逐个列名见下——本役 0.8-census 范围只覆盖了 53/80 |
| **Σ** | **80** | 零悬空 |

**★27 个「perf 账·未入 0.8 census」逐个列名（本报告最大发现）**：
- **FLAT gemm 绿 5**：`gemm_tile|q4_0/q4_1/q5_0/q5_1/q8_0`——perf-covered 绿（e2e prefill 赢），但**其 prefill-GEMM 从未进 0.8 kernel-sym census**（census 只做了 FLAT 的 vec_dot/decode，没做 FLAT 的 gemm/prefill）。
- **iq/ternary vec_dot 8**：`vec_dot|iq2_s/iq2_xs/iq2_xxs/iq3_s/iq3_xxs/iq4_xs/tq1_0/tq2_0`——perf 黄-对手更强，0.8-census 未接。
- **iq/ternary/fp4 gemm 8**：`gemm_tile|iq2_s/iq2_xs/iq2_xxs/iq4_nl/iq4_xs/mxfp4/tq1_0/tq2_0`。
- **quantize_row 3**：`q8_0/q8_1/q8_K`（黄-物理墙）。
- **product_reduce 3**：`q4_0_nibble/offset_binary_n3/codebook_n3`（声明例外）。

**诚实定性**：这 27 不是"无对手"，而是**本役 0.8-census 的 scope 未覆盖它们**（census 聚焦 K-quant matmul + forward + K-quant/iq dequant）。Q5 复查其对手可得性。

**★覆盖精算（诚实·别把"有去向"当"已测"）**：**真正跑过 0.8 板测 = 47 kernel**（42 进硬门 + 3 measured-DEQ + gelu例外 + 内部）；**去向已定但未实测 = 33**（27 perf-only + 6 FLAT/q1_0 dequant）。33 全有去向（零悬空），但"有去向标记" ≠ "已 0.8 板测"。

---

## Q2 · 硬门 kernel 双板战况（首次按正确单位·三行头条）

```
rvv:  PASS 29 / 有对局 42       (含 2 marginal-provisional q5x·9 dequant-auto-promoted)
k1 :  PASS 20 / 有对局 27       (含 2 q5x deployed-verified·q2_K reinstated)
双板均有对局 27 · 双板全 PASS 19
```

**双板全 PASS 19 逐 kernel**（recon 机算）：
- forward-op 8：`add·cpy·mul·rms_norm·rope·scale·silu·softmax`（f32）
- gemm 5：`gemm_tile|q2_K·q3_K·q4_K·q5_K·q6_K`
- vec_dot 6：`vec_dot|q4_0·q4_1·q5_0·q5_1·q5_K·q8_0`

**per-board 有对局差异的根因**：rvv 42 vs k1 27 = **DEQ 大板异**——rvv 的 gcc-15.2 `to_float` 把 15/18 dequant auto-vectorize（真向量对手→升门），k1 全 scalar（0 候选，留 DEQ-分账）。故 rvv 多 15 个 dequant 对局（其中 9 PASS/6 具名-X），k1 无。

**成色分层（诚实·计数≠强赢）**：29 rvv PASS 里——3 genuine dequant vec-vs-vec（q2_K/q3_K/tq1_0）+ 6 dequant ours-scalar-vs-opp-autovec（弱）+ 2 q5x marginal-provisional + 余为 cross-op/FLAT-adoption。20 k1 PASS 里——2 真 hand-brick 硬赢（q4_K/q2_K）+ 2 q5x deployed + q3/q5/q6_K vs block-dot（k1 无 riscv repack）。

---

## Q3 · vs 主力（hand-brick）实况——真·硬碰硬对局

「主力 = 手调 hand-brick 核」。逐个列出：

| kernel | 板 | 我方 | 对手（真派发） | cold 成绩 | 胜负 | 档案 |
|---|---|---|---|--:|:--:|---|
| gemm_tile\|q4_K | k1 | vl=16 repack | `ggml_gemm_q4_K_16x1_q8_K` hand-brick | **1.187×** | **WIN**(byte-verified nbad=0) | Win-K1-VLEN·a320e227 |
| gemm_tile\|q2_K | k1 | vl=16 repack | `ggml_gemm_q2_K_16x1_q8_K` hand-brick | **1.36×** | **WIN**(timing-valid·correctness ZERO-MODEL 复核) | a099e230/aa8f59ab |
| vec_dot\|q4_K | k1 | GEVM | `ggml_vec_dot_q4_K_q8_K_vl256` hand-tuned | 0.592 | 具名-X(Exit-C·真 beat 归 GEMM 轴) | — |
| vec_dot\|q2_K/q3_K/q6_K | k1 | GEVM | `*_vl256` hand-tuned | 0.52-0.68 | 具名-X(Exit-C) | — |
| vec_dot\|q2-6_K | rvv | GEVM | `*_vl128` hand-tuned | 0.18-0.34 | 具名-X(Exit-C·M=1 weight-recon floor) | named-X-rvv-vecdot |

**当前主力战绩比**：真 hand-brick 硬碰硬对局中 **2 WIN（q4_K/q2_K@k1 GEMM）/ 全部 K-quant vec_dot 具名-X 认输**（真 beat 在 GEMM 轴，decode/M=1 是 weight-recon 结构 floor）。**q5@k1（1.19-1.31× kernel / 2× e2e）对手是 native-vec 非 hand-brick**，成色 = beat-weak-baseline（2nd tier，见 Q4）。

---

## Q4 · 攻坚台账（本役 15 个裁决登记）

**已翻正**（kernel / 杠杆 / 前后成绩）：
- `gemm_tile|q3_K@rvv` · S6-strip-outer accumulator 消 spill · 0.16→**1.40**(rolled) · byte-exact 27/27 · 独立复验 CLEAN（预判"6-bit 内禀墙"证伪）
- `gemm_tile|q6_K@rvv` · 同 S6-strip-outer · 0.15→**0.96**(unrolled) · byte-exact
- `vec_dot|q5_0/q5_1@k1` · **per-format measured-gate selector 修**（C1 novelty）· 部署 block-dot 0.38→repack **1.19/1.31× kernel** → **e2e 2×** · 独立验证 CLEAN
- `gemm_tile|q2_K@k1` · WIN 重新起复（前序误判 bug → ZERO-MODEL 证 feeding-mismatch）· 1.36×

**具名-X 认输**（kernel / X 内容·均真解剖+构造+G1/G2）：
- `vec_dot|q2/q3/q4/q6_K`（双板）· X = M=1-weight-recon-amortization-floor（nr=1 无摊销·真 beat 归 GEMM 轴）
- `vec_dot|iq1_s@rvv` · X = per-sub-block vmv.x.s extract-chain + scalar-ls（**证伪前序"标量 index"假设** + 纠正 0.27 co-tenant artifact）
- `vec_dot|iq1_m`、`vec_dot|iq4_nl@k1`（repack-decline-justified）、`vec_dot|nvfp4@rvv`（re-roll trap·[GAP-P1] 第3 format）· 各带具名 falsifiable X
- `vec_dot|q5_0/q5_1@rvv` · 次路（部署路 REDESIGN-B 已绿·此 block-dot 非部署）

**还没攻的（列队·= Q1 的 27 perf-only + DEQ k1-side）**：FLAT gemm（5·perf 绿但未 0.8-census）、iq/ternary vec_dot+gemm（16）、quantize/product（6）。

---

## Q5 · 无对手筐复查预判（只判定·不实测·标 pending-remeasure）

**修正标准 = 含标量参考即有对手**。当前 27 个 perf-only kernel + k1-side dequant，按此标准复查：

- **全部有对手**：ggml 对每个格式都有 scalar `dequantize_row_*` / `vec_dot_*_generic` 参考——故**没有真正的"无合法对手"kernel**（旧口径若把某格标"无对手"，在修正标准下作废）。
- **预计转入对局（pending-remeasure·未实测）**：
  - FLAT gemm 5（q4_0/q4_1/q5_0/q5_1/q8_0）：对手 = FLAT block-dot / 上游 repack，**预计有对局**（perf 已证 e2e 绿，kernel-sym 待补测）。
  - iq/ternary vec_dot 8：对手 = `*_vl128/256` 或 scalar generic，**预计有对局**（perf 黄-对手更强，kernel-sym pending）。
  - quantize_row 3 / product_reduce 3：对手 = scalar 参考，**预计有对局但 perf 判物理墙/声明例外**（低传导面）。
- **净预测**：0.8-census 补全后，"有对局"数从 42 可扩至 ~60+（27 中大部分转入），但**多为弱对手/物理墙成色**，硬赢面不显著扩大。**此为预判，未实测**。

---

## Q6 · 悬案清单（现状 + 待裁选项）

1. **gelu 数值档不对等案**：对手 = `ggml_table_gelu_f16_BSS_LUT`（f16 查表·牺牲精度），我方 tanhf scalar（精度高 ~2500×）。现状 = JUDGMENT-SUSPENDED（不计 PASS/认输）。**待裁**：① 同数值档重比（我方也出 f16-LUT）② 精度-速度双列并存。
2. **e2e 部署树 stale 案**：q5@k1 e2e 用 deploy-patch 建净新 repack scaffold（`.inc` 须从当前 HEAD weft-opt 重生）——本役 q5 已重生为 REDESIGN-B（label 'G7-L3'≠core，objdump 已证）。**现状**：q5 部署树已刷新；**其余 e2e 部署树（q4_0/q8_0 等 G5/G7 旧账）的 `.inc` 是否仍为旧发射未逐一核**。**待裁**：是否全量重生 e2e 部署 `.inc` 对齐当前 HEAD。
3. **iq4_xs 接线归属案**：`vec_dot|iq4_xs` 在 perf 黄-对手更强，但 0.8-census 未接线；其归属（进对局 or 声明例外）未定。**待裁**：接线补测 or 归 perf-only。

---

## Q7 · 键控化状态（每把杠杆一行·确认零手动残留）

| 杠杆 | 谓词位置 | 双板自动应用证据 | Measured 门状态 |
|---|---|---|:--|
| [ROLL] whole-K-nest | `resolveRepackMainTermRolled`(BlockQuantLinear) | byte-exact DYNAMIC·默认 unrolled | measured-gate 空置（q3_K rolled 须 decode-cost keying·follow-up） |
| wide-vmadot | `decideWideVmadotDeployment`(IME driver) | registry measured-negative | q4_K@ime 0.909× 登记 |
| [QH-MASK] native-mask | `emitNativeMaskStaticBitBias` | byte-exact·契约 doc 已订正 | by-construction |
| leaf-batching-IME | `selectIMEMacLeaf` | 默认不变 | measured-gate 空置 |
| loop-order | `selectRepackLoopOrder` | 1 board-measured seed(q4_K ColOuter) | measured |
| full-LMUL[B] | `selectRepackAccumulatorLMUL` | 18 call-site stamp·isM1≠isRVV0p7 解耦 | measured-gate 默认 mf2 |
| **★q5 per-format VLEN256-decode** | `kRepackVlen256DecodeMeasurements`(FrontDoor) | **零 format-name·board-seeded·独立复验 CLEAN** | **measured（q5=Beneficial·q4_0/iq4_nl=Negative）·真部署** |

**确认**：6+1 杠杆全 keyed·零 format-name 硬编码残留（独立核查两轮 + q5 selector 复验兑现）。measured-gate 空置项 = 板测证据待阶段三填（现冻结）。

---

## Q8 · 与历史头条对账（换模型不丢结论·映射表）

| 历史数字 | 属什么账 | 新 kernel-unit 模型下对应 | 存废 |
|---|---|---|:--|
| **9/83** | perf-covered 系统账（e2e ≥parity 已构造格） | **保留**（不同赛道·kernel-unit 不替代）·+q5@k1 e2e 绿为 k1-engine 扩展 candidate（须裁·83→85?） | **保留** |
| **84/91** | M4 结构覆盖 certified（construction-protocol） | **保留**（结构轴·非 0.8 kernel-sym·非同赛道） | **保留** |
| **旧 20/28**（per-board 格点） | 0.8 kernel-sym·**板×格混算** | **作废**→ kernel-unit：rvv PASS 29/有对局 42·k1 PASS 20/有对局 27·**双板全 PASS 19** | **superseded** |
| **旧 29/43·30/28**（board 分列） | 同上·板当计数单位 | **作废**（板是属性列） | **superseded** |
| **真硬赢 2 hand-brick** | 成色定性 | **保留不变**（q4_K/q2_K@k1） | **保留** |
| **q5@k1 e2e 2×** | 系统账 e2e transduction | 新增·独立验证·成色 beat-weak-baseline | **新** |

**结论**：换 kernel-unit 模型后，perf-covered(9/83)、certified(84/91)、成色定性(真硬赢2)**全部结论保留**；只有「板×格混算」的 0.8 计数口径（20/28 系）作废，被 kernel-unit（rvv 29/42·k1 20/27·双板全 19）取代。**没有历史结论因换模型丢失。**

---

*报告完 · 全部计数 `recon_kernel_unit.py` 机算可溯源 · 阶段三攻坚冻结中 · 待用户裁决（三件 canon + 三件悬案）。*
