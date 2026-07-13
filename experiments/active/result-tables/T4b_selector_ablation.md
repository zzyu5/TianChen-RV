# T4b — 选择器四配置消融 (selector ablation · 复用 L1 穷举 oracle · 一鱼三吃)

> **性质**：**纯 docs 综合**（货架C·G7 §L3·T4b）。**复用 L1 已产的变体穷举数据 + P2 板 M-SCAN·无板·无代码·无 schema/T8 改动·无 git**。**[NG-4] 机制 C3′ 证据·非 perf 主张·非 e2e beat·非 perf-covered 系统账**（与 perf-covered 9/83 永不混算）。
> **oracle 口径**：**L1 每格 micro 最优 = 自体 oracle（[L-4] 合规）**——每格从其自身变体穷举取 micro 最优变体，非跨格/跨板外推。
> **数据来源（复用·只读综合）**：
> - `experiments/active/g7-l1-kernelsym-fullfill/rvv-batch/evidence.md`（FLAT-5@rvv/VLEN128·nr{4,16,64} 穷举·codegen 候选集分析 §2.1）
> - `experiments/active/g7-l1-kernelsym-fullfill/k1-batch/evidence.md`（K-quant@k1/VLEN256·q4_K vl=16/vl=8·q2/q3/q6·tile-form 穷举）
> - `experiments/active/g7-l1-gevm/P2-board-verify/evidence.md` §5（GEMM plan vs GEVM plan 形状 dispatch·M-SCAN 板直证）
> - `experiments/active/t4b-selector-ablation/MANIFEST.md`（in-tree gtest (常量盲/能力派生)×(GEMM/decode) 矩阵·[SEL-2] 静默落败复现→消失）
> - memory `[[repack-winA-always-mf2]]`（RVV1.0 repack gearbox 恒发 mf2·m1 只在 isRVV0p7 分支）
> - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`（[SEL-1] 两段式选择器 / [SEL-2] 硬时序 / [D-4] reason 枚举）；`docs/canon/Weft-RV_实验总纲v1.md` §4.2（四配置定义）

---

## 0. 净结论（TL;DR·诚实定性）

1. **codegen 变体候选集 = {mf2} 单一合法（双板）→ selector 平凡最优**。同一板（rv64gcv RVV1.0 / VLEN128·k1 VLEN256）合法 LMUL/fraction 候选只有 mf2；m1 whole-LMUL 是 RVV0.7.1(`xtheadvector`) **跨-ISA** 分支，**非 same-board 候选**（板不跑 xtheadvector 码）。⇒ **遗憾度 = 0·全 4 配置·全格·双板**（单候选=无选择空间）。**变体穷举的净信息 = nr-shape 轴**（`nr∈{4,16,64}`·workload-driven·**非 selector 可选项**）。
2. **真 [SEL-2] selector 价值 = paradigm/plan 形状 dispatch**（matrix vs vector 范式；GEMM plan vs GEVM plan·M=1↔M≥2）。**常量分配置复现"矩阵静默落败"·能力先验配置令其消失**（in-tree gtest 断言·P2 板 M-SCAN 直证形状键控·非 `if(M==1)`）。
3. **selector novelty 诚实定性**：**形状 dispatch [SEL-2] = 真**（结构/机制级·两族竞标同一 GEMM 形·常量分静默误路→先验修复）；**codegen 变体选择 = trivial**（单候选·无选择空间·非贡献）。**不夸大**：选择空间小 ⇒ codegen-axis selector 价值有限；价值全在 **paradigm/plan 形状 dispatch**。
4. **双板可迁移性**：codegen 候选集 **两板皆 {mf2} 单一**（rvv VLEN128 / k1 VLEN256·VLEN-invariant·t4a 证 k1 vtype 描述符 == rvv128）→ 平凡最优两板一致；形状 dispatch 两族竞标结构两板同（P2@rvv 板证 + gtest 两 paradigm 注入）。

---

## 1. 三层 selector 决策分解（"什么能被 selector 选"·避免混算）

selector 面对的决策**不是一层**。把三层分开是本表诚实性的前提——否则会把 Layer-1 的平凡（candidate 单一）与 Layer-3 的真价值（[SEL-2] paradigm 竞标）混为一谈。

| 层 | 决策 | 候选集 | selector 现状 | 本表判 |
|---|---|---|---|---|
| **L1 codegen 变体**（LMUL/fraction） | mf2 vs m1 | **{mf2} 单一合法**（m1=RVV0.7 跨-ISA·非 same-board） | 平凡选中唯一 legal（reason=`only_feasible`） | **trivial·遗憾=0·无选择空间**·[repack-winA-always-mf2] |
| **L2 tile-form / strip-width 变体** | S6-tiled vs PLAIN·vl=16 vs vl=8 | **有真"更快变体"**（q4_K@k1 vl=16 1.197×>vl=8·q2_K S6-tiled spill 收拢） | **hand-placed / 键控 generation·非键控 VLEN 事实**（[GAP-P1] selector-key-missing-VLEN-fact） | **真价值·但未键控**（gap·非 regret）·selector novelty 的**潜在**居所、当前未兑现 |
| **L3 paradigm/plan 形状 dispatch** | matrix vs vector 范式·GEMM plan vs GEVM plan | **两族对同一 GEMM 形竞标**（M-keyed） | **已落先验层**（SEL-1-T5+T5c·M-aware）+ P2 板 M-SCAN 证形状键控 | **真 [SEL-2]·常量分静默落败·先验修复**（landed·机制级 C3′ 证据） |

> **关键分辨**：L1 的 `<parity` 格（q2_K/q3_K/q6_K@k1 = 0.60/0.34/0.30×）**不是 selector regret**——loss 来自 emitter 全展开 regfile spill（627/1058/971·[GAP-EMIT-KNEST]/weight-floor），selector 已正确选中唯一 legal mf2。**变体选择正确 ∧ 变体质量欠成熟 = 不同轴**（本表只裁前者·后者住 T8/[GAP-EMIT-KNEST]）。

---

## 2. 四配置 × 三指标 消融表（实验总纲 §4.2 · oracle = L1 每格 micro 最优）

**四配置**（§4.2）= **{常量分遗留 / 仅先验 / 仅记忆 / 先验+记忆}**。
**三指标** = **top-1 一致率**（配置 top-1 选中 == oracle 每格 micro 最优的比例）/ **平均遗憾%**（`max(0,(perf_oracle − perf_config)/perf_oracle)` 跨格均值·选中 oracle 则 0）/ **灾难逃逸率**（避开"灾难性误选"的格比例；灾难 = paradigm 错 / 选到 `<parity` 变体而 `≥parity` 可得）。

> **★配置可落地性诚实标注**（§4.2 + t4b-selector-ablation MANIFEST）：**"仅记忆 / 先验+记忆"两列 gated on [SEL-3] measurement-memory（未落地）**——今天**不可实测填**（memoized argmin 需 offline-profile 实测记录 + byte-exact gate·[SEL-3] 后才有）。今天可落 = **{常量分遗留, 仅先验}** 两列。memory 两列标 **N/A(gated)**·**不硬造**。

### 2.1 Layer-1 · codegen 变体轴（{mf2} 单一）— **退化（degenerate·四配置不可分辨）**

**覆盖格**：FLAT-5@rvv（q4_0/q4_1/q5_0/q5_1/q8_0）+ K-quant@k1（q4_K/q2_K/q3_K/q6_K）= 9 格·双板。oracle = 每格唯一 legal mf2（= 现选中）。

| 配置 | top-1 一致率 | 平均遗憾% | 灾难逃逸率 | 备注 |
|---|---:|---:|---:|---|
| **常量分遗留** | **100%** (9/9) | **0%** | **100%** | 候选集单一→reason=`only_feasible`·无从误选 |
| **仅先验** | **100%** (9/9) | **0%** | **100%** | 同上·先验层无候选可排 |
| **仅记忆**（[SEL-3] 未落） | N/A(gated) | N/A | N/A | 记忆列 gated·且单候选下记忆亦无翻盘空间 |
| **先验+记忆**（未落） | N/A(gated) | N/A | N/A | 同上 |

> **★退化结论**：候选集 |{mf2}|=1 ⇒ **四配置在 codegen 轴上不可分辨**（常量分与能力派生给出**相同** top-1、相同遗憾 0、相同逃逸）。**这是诚实的 NULL：codegen 变体选择没有 novelty 可宣称**（没有选择空间就没有"选对"的功劳）。**变体穷举的真价值不在此轴**（见 §5 nr-shape）。

### 2.2 Layer-3 · paradigm/plan 形状 dispatch 轴（[SEL-2]）— **真价值·配置可分辨**

**场景**（2）= **{GEMM prefill (M≥M*·compute-bound·roofline 要 MATRIX)·decode/fragment (M<M*·memory-bound·roofline 要 VECTOR)}**。oracle = roofline-correct paradigm（GEMM→matrix·decode→vector）。数据源 = in-tree gtest `runT4bFourConfigSelectorAblationTest` cells (a)/(b)/(c)/(d)/(c′) + P2 板 M-SCAN。

| 配置 | top-1 一致率 | 平均遗憾 | 灾难逃逸率 | 机制 |
|---|---:|---|---:|---|
| **常量分遗留**（盲 static_order·matrix 20 > vector 1·升序→vector） | **50%** (1/2) | GEMM: **matrix 范式静默落败**（cell a·misfire·量级 gated on 自有 IME GEMM 实测※）；decode: 0（碰巧对） | **50%** | 单一场景盲常量分不开 GEMM 与 decode（cell a 与 d 注入相同分 20>1→相同 verdict vector·却 a 误 d 对·[ABLATION-4]） |
| **仅先验**（能力派生·SEL-1-T5+T5c·M-aware） | **100%** (2/2) | **0** | **100%** | GEMM: matrix 0.5<1→matrix 正（cell b）；decode: parity 1.0==1.0 tie→IR order→vector 正·或 single-fragment 20→vector 正（cell c/c′·双排除·[ABLATION-2/3]） |
| **仅记忆**（[SEL-3] 未落） | N/A(gated) | N/A | N/A | 记忆列 gated；冷启动无实测记录→无排序 authority（须退回先验层） |
| **先验+记忆**（未落） | N/A(gated) | N/A | N/A | landed 后 ≥ 仅先验（命中 memoized argmin 覆盖先验·byte-exact gated）·今天不可测 |

> **※ 遗憾量级诚实标注**：常量分在 GEMM 的 misfire = **matrix(IME systolic) 范式被静默换成 vector**。该范式优势本身 **gated on 自有 IME GEMM 实测**（实验总纲 §3 轨2 = M3+ 理想·现只生成单条 leaf MAC、无自有 GEMM）→ **misfire 的遗憾"量级"当前未板量化**。[SEL-2] 证据 = **结构/机制级**（静默误路的**复现→消失**·gtest 持久断言 + P2 形状 dispatch 板证），**非**一个实测吞吐 delta。**不夸大成 measured perf 遗憾。**
>
> **★ABLATION-3（精确性·2×2 对角）**：blind→derived **只翻 GEMM verdict**（a→b: vector→matrix）·**decode verdict 不动**（d→c: vector→vector）⇒ 能力派生 + M-aware = **精确修复**·非"一律偏矩阵"。这是 [SEL-2] 价值的**判别力铁证**（变异测试：`kDerivedDecodeParityCost` 1.0→0.4 或 `kDerivedGemmMatrixCost` 0.5→20 均令断言 FAIL）。

---

## 3. 共竞标专项（令 §四.3 · [SEL-2] 直接证据·矩阵/向量对同 GEMM 竞标）

[SEL-2] 硬时序（spec）：**能力先验排序层必须先于/同于矩阵范式接管 GEMM 形 prefill；一旦向量与矩阵两族为同一 GEMM 形内核竞标而先验层缺席，矩阵范式静默落败**。本专项给两条独立直证。

### 3.1 in-tree gtest（注入半·常量分复现落败 / 先验令其消失）
`test/Transforms/VariantSelection/VariantSelectionTest.cpp :: runT4bFourConfigSelectorAblationTest`（保行为·零 lib/ 改动·仅观测断言）：

| cell | 注入 (matrix, vector) | 场景 | chosen | 正确性 | 配置 |
|---|---|---|---|---|---|
| (a) | (20, 1) | GEMM | vector | **误(misfire)** | 常量盲（== pre-SEL-1-T5 IME 常量分） |
| (b) | (0.5, 1) | GEMM | matrix | 正 | 能力派生（`ime_matmul_shape` ∧ M≥M*） |
| (c) | (1.0, 1) | decode(parity) | vector | 正（不偏矩阵） | 能力派生（M<M* → RVV base parity） |
| (d) | (20, 1) | decode | vector | 正（**碰巧**·与 a 同分不同场景） | 常量盲 |
| (c′) | (20, 1) | decode(single-fragment) | vector | 正（双排除·真-可达生产 decode 路） | 能力派生（single MAC fragment cost） |

**读数**：常量分（a）**复现**矩阵静默落败（GEMM 形选到 vector·无报错）；能力派生（b）**令其消失**（选 matrix）；且（c/d）证修法**不误伤 decode**（不把 memory-bound decode 误纠到矩阵）。**verdict DIFFER on GEMM（a→b）∧ SAME on decode（d→c）= [SEL-2] 精确修复。**

### 3.2 P2 板 M-SCAN（生产半·形状键控直证·非 `if(M==1)`）
`g7-l1-gevm/P2-board-verify/evidence.md §5`（q4_K@rvv·真部署 plugin·banner 计数）：

| M（形状） | 相 | GEVM(vector) plan banner | GEMM(matrix) plan banner | 分流事实 |
|---|---|---:|---:|---|
| **M=1**（`-p0 -n8`·decode） | decode | **4** | **0** | **纯 GEVM/vector plan·零 GEMM** |
| **M=8**（`-p8 -n0`·prefill） | prefill | 5（尾余单行） | **8** | **GEMM/matrix plan 主导** + GEVM 尾余（leftover 单行走 gevm） |

> **★[SEL-2] 生产直证**：ggml **自有** gemv-vs-gemm dispatch 在 **M=1↔M≥2 之间切换**（M* 落在 1 与 2 之间）——**selector 键控形状事实（nr/M），非硬编码 `if(M==1)`**（[K-10] 选择层=形状事实∧能力事实）。这与 gtest 的 (b)/(c) 场景键控（M≥M* matrix / M<M* vector）**同构·两板一致的键控原则**。
>
> **★诚实边界（不混算·plan-body maturity ≠ dispatch regret）**：P2 证 **形状 dispatch 本身正确**（correctness GREEN·byte-exact·3/3 prompt·banner 键控正确），**但被路由的首版 GEVM plan（colgroup-tiled TG=2）板证 decode 回归**（0.49×→**0.34×**·IPC 0.294→0.214·register-resident bank 溢出反噬·spill 534 vs 10）。**这是 plan-BODY 成熟度问题（首版结构假设 perf 未兑现·需迭代）·住 [PAT-2]/[GAP-P1]·不是 selector-config 消融的 regret**。**selector 正确按形状路由 = [SEL-2] 成立；routed plan 的 perf 质量 = 下游独立轴**。本表 §2.2 的 "遗憾" 只裁 paradigm 路由对错、不把 plan-body 回归算进 selector 账。

---

## 4. 双板对照（令 §四.4 · rvv VLEN128 / k1 VLEN256 · 可迁移性证据）

| 轴 | rvv (VLEN128·gcc-15 对称) | k1 (VLEN256·clang-18 对称) | 迁移性判 |
|---|---|---|---|
| **L1 codegen 候选集** | **{mf2} 单一**（half_lanes=8·columnsPerPass=4 auto-forced·`integer_core_lmul` absent） | **{mf2/VLA} 单一**（VLEN-invariant·t4a 证 vtype 描述符 == rvv128） | **两板同·平凡最优可迁移**（selector trivial 两板一致） |
| **L1 四配置消融** | 退化（4 配置 top-1 100%/遗憾 0/逃逸 100·§2.1） | 退化（同·含 `<parity` 格 q2/q3/q6 仍非 selector regret） | **两板同退化**·codegen selector novelty 两板皆 NULL |
| **L2 真"更快变体"** | m1 arm 对 q4 更快但 = RVV0.7 跨-ISA·hand-placed（.inc-swap）·非 auto-select（[repack-winA-always-mf2]） | **q4_K@k1 vl=16 sealed 1.197×>vl=8 0.622×**（VLEN256 满宽·vwmacc 2240→1120）·但键控 generation 非 VLEN 事实（[GAP-P1]/[GAP-Q4K-VLEN128]） | **两板皆有真变体·两板皆未键控 VLEN 事实**（[GAP-P1] selector-key-missing-VLEN-fact = 双板一致的成熟度 gap） |
| **L3 paradigm dispatch** | P2 板 M-SCAN 直证（M=1→GEVM/M=8→GEMM·形状键控） | gtest 两 paradigm 注入（M≥M* matrix / M<M* vector·同键控原则）；k1 IME = N2 PROVEN 的范式承载 | **形状键控原则两板同**（[SEL-2] 可迁移·板证半在 rvv·机制半 board-agnostic） |
| **nr-shape 穷举（净信息·非 selector）** | nr{4,16,64} 全 ≥parity·q5_0/q5_1 win↗nr（1.22→1.42） | nr{64,16,4} 全 `<parity`（q2/q3/q6·nr↓ 未翻·weight-bound） | **shape 轴两板皆 workload-driven·非 selector 可选项**（穷举证 mf2-repack 跨 nr 稳定·非 selector 功劳） |

> **迁移性净读**：**codegen selector 的"平凡最优"与"四配置退化"是两板一致的**（可迁移的 NULL）；**真 selector 价值（形状 dispatch [SEL-2]）的键控原则亦两板一致**（可迁移的正结果·板证半在 rvv/机制半 board-agnostic）；**唯一两板一致的"未兑现价值" = L2 VLEN-事实键控**（[GAP-P1]·两板皆 hand-placed/generation-keyed·selector 尚未按 VLEN 自选更快变体）。

---

## 5. 变体穷举的净信息 = nr-shape 轴（非 selector·令 §关键发现）

L1 变体穷举（`nr∈{4,16,64}`·K=2048 nc=512 固定）的**净信息不在 codegen 选择**（候选单一），而在 **shape 轴稳定性**——mf2-repack 跨全 shape 的行为：

| 格·板 | nr=4 (cold) | nr=16 (cold·锚) | nr=64 (cold) | shape scaling |
|---|---:|---:|---:|---|
| q4_0@rvv | 5.091 | 6.224 | 6.558 | cold-penalty↓ as nr↑（nr4 memory-exposed） |
| q5_0@rvv | 1.216 | 1.232 | 1.418 | **win↗nr**（better-vec 对手高 nr throughput 掉更多） |
| q8_0@rvv | 6.483 | 4.752 | 4.267 | cold>hot（对手 block-dot cold 退化更甚） |
| q2_K@k1 | 0.551 | 0.625 | 0.601 | 全 `<parity`·nr↓ 未翻（weight-bound spill） |
| q3_K@k1 | 0.340 | 0.345 | 0.344 | 全 `<parity`·HOT≈COLD（compute/weight-bound） |

> **shape 轴 ≠ selector 轴**：`nr`/M 是 **workload 形状事实**（ggml 上游 dispatch 定·非我方 codegen 变体旋钮）。穷举证的是 **mf2-repack 对 nr 的 scaling 优于/劣于对手 block-dot**（一个 kernel-质量事实），**不是** selector "选对变体"的功劳。**把 shape 稳定性算作 selector novelty = 混算·本表明禁**。

---

## 6. selector novelty 诚实定性 + 纪律边界

| 判项 | 结论 |
|---|---|
| **codegen 变体选择（LMUL/fraction·L1）** | **trivial·NULL novelty**——候选集 {mf2} 单一·四配置退化不可分辨·遗憾恒 0（含 `<parity` 格·因 loss 归 emitter 非 selector）。**禁宣称"selector 选对 mf2"为贡献**（无选择空间）。 |
| **tile-form/VLEN 变体键控（L2）** | **真价值·未兑现**——q4_K@k1 vl=16 是真更快变体（1.197×），但 hand-placed / 键控 generation·**未键控 VLEN 事实**（[GAP-P1]）。selector novelty 的**潜在**居所·当前是 gap 非 regret·**不可当已实现主张**。 |
| **paradigm/plan 形状 dispatch（L3·[SEL-2]）** | **真 novelty·机制级**——两族竞标同一 GEMM 形·常量分静默落败**复现→**先验**消失**（gtest 持久断言 + P2 板形状键控直证·非 `if(M==1)`）。这是 T4b 的**唯一**可主张 selector 价值·**且是机制/结构级 C3′ 证据（[NG-4]）·非 perf beat**。 |
| **shape 稳定性（nr 穷举）** | **kernel-质量事实·非 selector**——workload-driven·禁算 selector 账。 |

**纪律边界（本表覆盖面）**：
- **[NG-4]·非 perf·非 e2e·非 perf-covered 系统账**（与 9/83 永不混算）。本表是**决策键控轴的描述性消融**（哪些决策已由能力事实键控 / 哪些平凡 / 哪些静默误路→修复），**非**吞吐主张。
- **oracle = 自体 L1 micro 最优（[L-4]）**·永不跨格/跨板外推·永不混算 hot/cold/nr/board/编译器身份。
- **[SEL-2] 价值量级未板量化**（matrix/IME GEMM 范式优势 gated on 自有 IME GEMM 实测·§3 轨2 = M3+ 理想）→ [SEL-2] 证据是**静默误路的复现→消失**（结构/机制），**非**measured 吞吐 delta。**不夸大。**
- **plan-body 回归（P2 GEVM-CT decode 0.34×）≠ selector regret**——住 [PAT-2]/[GAP-P1]·下游独立轴·本表不算进 selector 账。

---

## 7. Provenance / 复用来源（无板·无新测·纯综合）

| 断言 | 复用来源 |
|---|---|
| codegen 候选集 {mf2} 单一（rvv） | rvv-batch/evidence.md §2.1（half_lanes=8→mf2·columnsPerPass auto-forced·`integer_core_lmul` absent·passes.td:386-394 m1=RVV0.7-only） |
| codegen 候选集 {mf2/VLA} 单一（k1·VLEN-invariant） | k1-batch/evidence.md §变体穷举（t4a 证 k1 vtype 描述符 == rvv128） |
| m1 恒非 auto-select·RVV1.0 恒发 mf2 | memory `[[repack-winA-always-mf2]]`（`RVVRepackStripWidthMaterialization.cpp` m1 只在 `if(isRVV0p7)`） |
| q4_K@k1 vl=16 1.197×>vl=8 0.622× | k1-batch/evidence.md §B（G 决胜局·N=12·对称 clang-18·byte-exact bounded-ULP） |
| nr-shape 穷举（rvv/k1） | rvv-batch §2.2 / k1-batch §hot-cold 穷举 |
| gtest (a)-(d)/(c′) 四配置矩阵 + 变异测试判别力 | t4b-selector-ablation/MANIFEST.md（`runT4bFourConfigSelectorAblationTest`·保行为·零 lib/ 改动） |
| P2 板 M-SCAN（M=1→GEVM4/GEMM0·M=8→GEMM8/GEVM5） | g7-l1-gevm/P2-board-verify/evidence.md §5 + raw/results.txt L24-25 |
| 四配置定义（常量/仅先验/仅记忆/先验+记忆）·[SEL-3] gating | 实验总纲 §4.2 + variant-pipeline/generation-selection-tuning.md（[SEL-1]/[SEL-2]/[D-4]） |
| L1 `<parity` 归 emitter 非 selector（[GAP-EMIT-KNEST]） | T-KEYING §2.1 + k1-batch §objdump SEAL（spill 627/1058/971） |

> **无 git·无板·无代码/schema/T8 改动**。本 doc 仅综合既有 casefile·oracle 自体·[L-4] 合规·[NG-4] 机制证据。
