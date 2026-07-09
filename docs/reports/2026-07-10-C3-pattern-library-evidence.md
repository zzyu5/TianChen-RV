# C3′ 能力键控优化模式库 — 证据 synthesis (论文素材，2026-07-10)

**贡献定位**: C3′ = **能力键控的优化模式库（带实测与迁移的模板）**。本文把三件已落地的资产织成一条证据链：
(1) **[XFER-1] 3-类迁移预测规律**（一条统一判据 → 三类结局，硅上 7/7 验证）；(2) **[PAT-S6] 机制化模式注册表条目**
（`schema/pattern-registry.v1.json`，模式作为一等数据对象）；(3) **SEL-1 能力键控选择器机制**
（`include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h` + 选择 pass，把编译期 per-format 硬编码升级为运行时能力键控）。

**★性质 / 措辞锁（关键）**: 纯 doc synthesis，**零 lib/schema 改、未 commit**。所有吞吐数（1.884×/1.413×/2.193×）
与 spill/maxVreg 皆引自 §二.4 `2026-07-07-paper-material-inventory.md` 已存在 cells（rvv/VLEN128 core8 measured 2026-07-08/09）。
**min-term 案已 M4 CASE CLOSED（commit `eede2ac3`/`02af804d`/`4f765790`）：底层 kernel 经决定性终审证正确，perf 数立在
现已证正确的 kernel 上 → cite 为 correct-kernel（post-M4），★不 re-narrow。** 这些是 **kernel-轴** A/B（单核、opponent=
单线程 block-dot proxy、8 [PERF-1] 门未走、[NG-4] 非 beat）；整模型 e2e 是 **projection**，不是 measured Win。

---

## 0. 一句话

> C3′ 的库不是"一堆手调 kernel"，而是**一条可迁移的判据 + 机制化的注册表条目 + 能力键控的选择器**。
> 判据用 **瓶颈形状**（结构/能力事实，从 `fold_model` 派生）当键——**换键不改条目**：一个新格式一旦映射到已有形状，
> 就免费继承它的迁移裁决（q4_0 与 iq4 同映射 `AlreadyLean` 即证）。7/7 硅预测把"这条判据可迁移"从断言升为一手实测。

---

## 1. [XFER-1] 3-类迁移预测规律（C3′ 的迁移模板核心）

**统一判据（一条）**: **S6 output-tiling 适用 ⟺ 热循环携带来自 stageable decode strips 的高寄存器压力**——
S6 stack-panel 能把一条 **COLD 已解码 strip** 外置到栈，使 **HOT 累加器 fan-out** 掉到 **≤32-vreg 悬崖**之下。
**可观测判别式（objdump）**: post-tile `maxVreg` = **v30**（跨过 ≤32 悬崖）= HOLDS；**v31**（未达悬崖）= NULL；
`spill→0` = HOLDS，`spill 上升/持平` = NULL，`tile 前已 ≤32` = 结构 no-op。

一条判据分岔出**三类结局**（键 = 瓶颈形状 `fold_model`，**非 format 名**）：

| 类 | 瓶颈形状（键） | 结构成因 | 裁决 | ships |
|---|---|---|---|---|
| **① min-fold register-cliff** | `kquant_dmin_bsums_min` | 双 d/dmin + bsums-min i32 累加器 **围绕**一个 **stageable** 单平面 nibble/scale decode；是**累加器 fan-out**（非权重装配）撑爆 32 | **HOLDS** | S6-TILED |
| **② weight-reconstruction-bound** | `kquant_single_scale_no_min` | **多平面**权重装配（q6_K ql+qh / q3_K qs+hmask），~900+ vand/vsrl/vor/vsub 重构 **non-stageable**：整条权重 strip 必须同时活着才能拼一个 int8 | **NULL** | PLAIN |
| **③ codebook-gather already-lean** | `codebook_*` / `lane_wise_vector_scale` | 码本 decode 走 `vluxei16` **内存 gather**（weight=codebook[nibble]，vwmacc=0=零寄存器权重重构）；spill **本就 ≤32**，无 stageable strip 可外置 | **STRUCTURAL NO-OP** | PLAIN |

> ②与③都 ships-PLAIN 但**机理不同**：②是**高压下没够到悬崖**（v31，杠杆施了但被权重面吃掉）；
> ③是**本就在悬崖下**（v30，无 stageable strip、无杠杆可施）。这个区分本身是迁移规律的一部分——
> "为何不 tile"有两个不同的、可判别的原因。

### 1a. 硅上 7/7 预测登记表（★现底层 kernel M4-proven-correct）

**权威源** = `schema/pattern-registry.v1.json` (`PAT-S6…XFER-1`) + `experiments/active/visibility/T7-three-curve-G3-closure.md`
（预测登记表）+ 逐格 A/B `experiments/active/result-tables/T8_winloss_gap_ledger.csv`（7 条 XFER-1 行）。
**★byte-exact / 吞吐已 RESTORED**：min-fold 判据经 M4 决定性终审证共享 `kquant_dmin_bsums_min` 对真 mat-quant dispatch
**整数-exact**（比 ggml 自身 generic 更近 int-exact），下列吞吐数**立在现已证正确的 kernel 上**。

| # | format | 瓶颈类 | 预测 | 实测（vs-opponent untiled→tiled） | 判别式（spill / maxVreg） | 一致 |
|---:|---|---|---|---|---|:--:|
| 1 | **q4_K**（anchor） | ① min-fold cliff | HOLDS→tiled | HOLDS **1.884×**（0.962→；~88% lead） | 84→**3** / v31→**v30** | ✓ |
| 2 | **q2_K** | ① min-fold cliff（共享 dmin/bsums fold） | HOLDS→tiled | HOLDS **1.413×**（0.212×LOSS→WIN **FLIP**） | 619→**7** / v31→**v30** | ✓ |
| 3 | **q5_K** | ① min-fold cliff **+qh 面**（HYBRID） | HOLDS→tiled | HOLDS **2.193×**（1.547→；already-WIN 推进） | 155→**105**（−32%）/ v31→**v30** | ✓ |
| 4 | **q6_K** | ② weight-reconstruction-bound（ql+qh） | NULL→plain | NULL（~5.4× LOSS 未救） | 913→**949 ROSE** / v31 | ✓ |
| 5 | **q3_K** | ② weight-reconstruction-bound（qs+hmask,no-min） | NULL→plain | NULL（~5.2–5.3× LOSS 未救） | 978→894（−8.6% overhead-shave）/ v31 | ✓ |
| 6 | **iq4_nl** | ③ codebook already-lean | no-op→plain | no-op（structural；front-door=退役直发器 byte-exact） | 7 / v30（**tile 前已 ≤32**，vwmacc=0） | ✓ |
| 7 | **iq4_xs** | ③ codebook already-lean | no-op→plain | no-op（structural） | 73 / v30（**tile 前已 ≤32**） | ✓ |

**= 7/7 预测-实测一致**（min-fold HOLDS ×3 / weight-bound NULL ×2 / codebook no-op ×2）。**crux（q5_K）**：一个共存的
权重面（qh 5th-bit）**降级但不击败** min-fold 杠杆（hybrid HOLDS，spill 未塌到 0 但仍达 ≤32 悬崖）——这是"键控于 min-fold、
不受正交权重面搅动"在最细分辨率上的一手证据。

### 1b. 迁移账（这是"模板"而非"个案"的证据）

- **q4_K→q2_K/q5_K 的迁移是零 framework 泛化的整体复用**：q2_K decode facts 源码明写 "REUSES the existing q4_K
  KQuantDecodeFacts fold WITHOUT any framework generalization"；q5_K = q4_K fold + qh 5th-bit plane。register-cliff 杠杆
  **整条转移**，预测→实测两次命中（validation #1/#2 PREDICTION-HIT）。
- **NULL 也是迁移裁决**：q6_K/q3_K 预测 NULL→实测 NULL 同样是 7/7 的一部分——库能**预测哪里不该优化**（并给出可判别的
  "为何不"），与预测哪里 HOLDS 同等重要，是 C3′「带迁移的模板」的负向素材（marginal-cost / maturity 边界）。
- **[NG-4]/[L-1] 范围**：kernel-轴 vs-opponent parity + spill/maxVreg objdump；**非 e2e、非 8-门封印**。整模型 e2e =
  projection（传导账：prefill matmul 占 97.3% → Amdahl `S_prefill ≈ 1.59×`，区间 1.3–1.6×，纯-q4_K 机理天花板 1.84×；
  **decode 传导 ≈ 0**，M=1 GEVM memory-bound，tiling 增益机理不存在）。projected prefill ≈ 8.24 t/s（**projection 非 measured**，
  measured Δ 集成 BLOCKED，见 `t4b` 逐环归因）。

---

## 2. [PAT-S6] 机制化模式注册表条目（模式作为一等数据对象）

`schema/pattern-registry.v1.json` 的 `PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1`（`status: mechanized`）把上面的
迁移规律**机制化为一条注册表数据行**，而非藏在 emitter 里的隐性约定。关键字段：

- **`discriminant_capability_fact`** = `vreg_count=32`：**≤32-vreg 寄存器悬崖是判别能力事实**（vlen 只定 half/numHalves 与 mf2
  链 vreg 成本）。S6 是**对已机制化 front-door 构造的 LAYER-4 output-tiling 优化**——**永不移动 C_construct**（构造轴与优化轴分离，
  core-invariants I4）。
- **`transform`** = S6 的具体几何：h-strip `hs=1` + inline-min-fold（丢掉 bsums i32 累加器族）+ 已解码权重 stack-panel +
  on-demand d/dmin widen（SSA-register 累加器）；peak live vreg **94→~29（≤32）**，同时**保住 once-per-16-weight decode 摊销**（FULL）。
- **`mechanism`** = §1 的统一判据（三类 + 可观测判别式 v30/v31）。
- **`xfer_classes`**（3 条子对象）= §1 表的机器可读版：每类带 `verdict`（HOLDS/NULL/NO-OP）、`ships`、`formats`、`requires`、
  `mechanism`——**逐类的转移条件与失效成因都是显式数据**，可被 lit/metrics 消费。
- **`metrics_hook`** = 指向 7 条 T8 XFER-1 行 + T7 登记表 + per-format cells（`l1-tile-s6-q4k` / `l1-t3-q{2,3,5,6}k` /
  `l1-m2-iq4`）；**`xfer_note`** 显式记：per-class verdict 是 **HARDWARE-MEASURED characterization（7/7）**，非 e2e、非 sealed。

> **registry-as-data 的意义**（C3′）：模式不是"读代码才知道的经验"，而是一个 status 会随证据流转的**数据对象**
> （`planned→partial→mechanized`，或 `measured-negative` 记录被证伪的优化假设作为设计空间资产）。同一 registry 里
> `MFLAT-P2c`（deferred-ordered fold）与 `WIDE-DECODE-salvage` 分别是 `measured-negative` 与 `deferred-backlog` 的范例——
> **库同时收录 HOLDS 模式、被证伪边界、与 parked salvage 路径**，这正是"带实测的模板库"该有的完整形态。

---

## 3. SEL-1 能力键控选择器机制（编译期硬编码 → 运行时能力键控）

**靶**: 8-gate ⑦ 缺口 = tiled/plain 选择今天是**编译期 per-format 硬编码**（`emitTypedRepackGemmLoopBody` 内按 `decode_model`
分派到内嵌 tiled/plain 的 body-emitter），reason≈static_order/hardcode、**非 capability-keyed**、且 **`decode_model`=format 名近亲**
（format-name 分派复发风险，T3 红线）。**SEL-1** 把它提升为运行时能力键控选择，authority 落在纯函数族
`include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h`（COST-MODEL-FREE，无 MLIR 类型，unit/lit-testable，镜像既有
`RVVFillLMULReason`/`chooseFillOptimalLMUL` prior 范式）。

### 3a. 键 = 瓶颈形状（结构/能力事实），不是 format 名 ⇒「换键不改条目」

`classifyTilingBottleneckShape(fold_model) → RVVTilingBottleneckShape`：
- `kquant_dmin_bsums_min` → **`MinFoldRegisterCliff`**（→ prior `S6Tiled`）
- `kquant_single_scale_no_min` → **`DualPlaneWeightBound`**（→ prior `Plain`）
- `codebook_flat_single_scale` / `codebook_superblock_signed6_no_min` / **`lane_wise_vector_scale`** → **`AlreadyLean`**（→ prior `Plain`）

**★迁移判据的机制证明**：flat **q4_0** 的 `lane_wise_vector_scale` 与 **iq4** 码本**同映射 `AlreadyLean`**——
**键是形状不是格式，q4_0 与 iq4 co-map**，各自免费继承 already-lean 的 Plain 裁决。这就是 C3′「换键不改条目」在代码层的
一手证据，也是**从机制上防止 format-name 分派复发**（新格式加一行 fold_model→shape 映射即接入，无需碰 emitter 分派）。

### 3b. 两段式选择（`selectRepackTilingVariant`）

- **Stage-1 合法性过滤**（`tilingVariantFeasibleSet`）：output-tile spilling 合法（栈 panel），故能供寄存器的板
  （`vlen≥128 && vreg_count>0`）**同时容 {Plain, S6Tiled}**——register cliff 是 Stage-2 prior 推理的**性能杠杆**，不是合法性门。
  退化板（无 VLEN 事实/无 vreg 预算）→ 空 feasible 集（诚实的 no-capability 行为）。
- **Stage-2 排序**：
  - **有测量**（offline-profile hit 且仍 feasible）→ memoized argmin，**`reason=Measured`**（fail-closed-revalidate：stale/now-infeasible
    的测量丢弃，落回 prior）。
  - **冷启动**（无测量）→ **[XFER-1] 能力先验**，键控于瓶颈 SHAPE，**`reason=Prior`**。
  - 空 feasible → shape 默认变体（byte-exact-preserving）+ **`reason=StaticOrder`**（**诚实**标注非 prior；不应出现在有能力的板上，
    出现=先验覆盖缺口=燃减信号）。

**reason 枚举**（`RVVTilingSelectionReason`，[D-4] 键控做在**主键**、不做脚注守卫）= `{OnlyFeasible, Measured, Prior, StaticOrder}`。

### 3c. ★q6_K/q3_K → Plain `reason=measured` 是诚实回退，IS 主张的一部分

测量库（`lookupTilingMeasurement`，seeded from REAL T8 rvv/VLEN128 board hash `3cd23a4e…`）种入 5 个 K-quant argmin 赢家：
- min-fold cliff 家族 **q4_K/q2_K/q5_K → `S6Tiled`**（S6 达 ≤32 悬崖，1.884/1.413/2.193×）。
- weight-bound no-min 家族 **q6_K/q3_K → `Plain`**：S6 是 NULL 杠杆（悬崖永不达、两平面权重重建 dominate），
  **诚实 measured 赢家 = 未 tiled body**。**这不是盲目默认，而是 [XFER-1] 迁移边界主张：测量本身在说"这里别 tile"。**
  → **q6_K/q3_K 的 `reason=measured`/`variant=Plain` NULL 回退，本身就是键控选择器的一个正确输出**，是主张的一部分而非缺席。

其它 (hash,kernel)——q4_0、iq4_nl/iq4_xs、任何未 profile 的板——MISS → nullopt → 冷启动 [XFER-1] prior（`reason=Prior`）。
**故选择器同时演示两条路径**：measured argmin（5 个 seeded K-quant）与 prior 冷启动（q4_0 + 码本对）。

### 3d. 门⑦ 关闭（MISSING→PASS）

SEL-1 **T2 曳光弹**（commit `4624740f`）把 SP4 从编译期 per-format 硬编码升级为运行时能力键控：loop-body op 上 stamp
`tcrv_rvv.tiling_selection_reason="prior"`（reason=prior 键控 `fold_model` 瓶颈 SHAPE 非 format 名，`kquant_dmin_bsums_min`
→`MinFoldRegisterCliff`→S6Tiled）+ 完整 [D-4] JSONL 归因日志（`buildTilingSelectionAttributionRecord`，canonical-JSON、复用同一
`declared_instance_hash`）。gate-7 lit `rvv-sel1-q4-k-tiling-variant-capability-keyed-gate7.mlir`、**524/524 RVV lit PASS**、
**C_construct 42 不变**（LAYER-4，选择器不移构造轴）。→ q4_K 8-gate **3/8→4/8 PASS**（门⑦ = 每 K-quant 格 reason ∈
{prior,measured,only_feasible}、**非 static_order/hardcode**）。**★仍非 sealed Win，beat 措辞 LOCKED（[NG-4]）**：
缺项 = ②VLEN256 flip lit + ⑤k1 双板（SEL-1 T4 板批同收）、④e2e 集成（独立 e2e-seal 战役）。

---

## 4. 方法学严谨背书（perf 主张的严谨性来自这里）

上面所有吞吐数**立在一个 M4-proven-correct 的 kernel 上**。这个"proven"不是随口——它由一套已入宪的方法学产出，
**背书了 perf 主张的严谨性**：

- **[CASE-MINTERM] 卷宗**（commit `b05acc94`）: M2c 曾报 q4_K/q5_K repack-GEMM 有 "VLEN128 parity-alternating min-term bug"
  （rel 5%–314%、列-parity R/correct 0.947/0.637）。M1(ZERO-MODEL) + M4(决定性终审) 推翻之——**根因 = cert-harness 的 q8
  quantizer 失配**（对照喂 row-quant `quantize_row_q8_K`，被测走 mat-quant `ggml_quantize_mat_q8_K_4x1` interleaved-q8_Kx4），
  是**语料伪影非 kernel 缺陷**。M4 决定性证据（commit `eede2ac3`）：我方 kernel vs **ggml 自身 generic GEMM**、同 mat-quant
  真模型 `dmin≠0` 张量、**整数逐位一致**（且比 generic 更近 int-exact）。→ CASE CLOSED，吞吐兑现 →1(+3 pending) **RESTORED 回 4**
  （commit `02af804d`/`4f765790`），[XFER-1] 解除、cert-lineage FULL。**★perf cite 为 correct-kernel（post-M4），不 re-narrow。**

- **证书三要件**（CI gate，commit `2363fd3c`；`tools/lint/check_cert_requirements.py` + `schema/cert-lineage.v1.json` +
  `.github/workflows/falsifier-gate.yml` 的 `cert-requirements-gate`）: 每个数值 cert 静态声明 ① **语料完备**（fold 必激活项
  非退化点火）② **输入同源**（tested/oracle quantizer 同源、dispatch-faithful）③ **oracle 独立**（∈{independent,self-compare,
  shared-impl}，空心自比不得冒充独立）。gate 在每次 PR **静态**拦截 M2c 那类声明复发（已证会抓 M2c 误诊）。

- **ZERO-MODEL 终审法**（rule spec，`docs/reports/2026-07-09-zero-model-adjudication-method.md`）: 数值正确性争议的**标准终审
  工具**——从被测物**自己的实际输入**出发、用**零复用被测读取/折叠实现**的独立参照**逐项重算**、对被测**内部 intermediates
  逐项对位**，`0-mismatch ⟺ 正确`。入宪原则：**「定罪与翻案同等严谨」**（认错和认对证据规格一致）、**「默认走 ZERO-MODEL
  对位终审两步、不再多轮 oracle 互搏」**（多轮互搏会共享同一盲区、无收敛）。

> **背书链**: perf 主张 → 立在 M4-proven kernel → M4 用 ZERO-MODEL 逐项对位裁死 → 三要件 CI gate 保证"没人再写出 M2c 那种
> 声明" → **优化模式库的每个吞吐数下面，都有一个用最强可得裁决法证正确的 kernel**。这是 C3′ 相对"一堆 microbench 数字"的
> methodology strength：**优化是可迁移的模板 + 底层正确性是可证伪、已证的**。

---

## 5. 证据 commits / cells 索引

**[XFER-1] 7/7 + PAT-S6**:
- `schema/pattern-registry.v1.json` — `PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1`（status=mechanized，3 子类数据）
- `experiments/active/visibility/T7-three-curve-G3-closure.md` — [XFER-1] 预测登记表（7/7）
- `experiments/active/result-tables/T8_winloss_gap_ledger.csv` — 7 条 XFER-1 行（tile-S6 HOLDS ×3 / tile-T3 NULL ×2 / frontdoor-tile-NOOP ×2）
- cells: `experiments/active/l1-tile-s6-q4k-repack-gemm` + `l1-t3-q{2,3,5,6}k-repack-gemm` + `l1-m2-iq4`
- q4_K S1→S6 兑现: commit `d5a28efc`(tile-S1) + `5f194cbd`(tile-S6)；设计尺子 `docs/reports/2026-07-08-G3-L1-tiling-schemes.md`
- 家族 front-door 退役（7 次 +1 C_construct ∧ −1 旁路）: `7a4250c5/0b907d0c/7ff52fc4/c3cf7301/1b367c1f/54d3741c/df8a0b76`；
  M2-后 iq4 码本对: `6c0961b6`(iq4_nl,C_construct 40→41) + `c238238a`(iq4_xs,41→42)

**SEL-1 能力键控选择器**:
- `include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h`（纯函数族：classify/feasibleSet/lookupMeasurement/select/attributionRecord）
- SEL-1 设计: `docs/reports/2026-07-09-SEL-1-design.md`；T4 板批计划: `docs/reports/2026-07-09-SEL-1-T4-plan.md`
- 门⑦ T2 曳光弹: commit `4624740f`；gate-7 lit `rvv-sel1-q4-k-tiling-variant-capability-keyed-gate7.mlir`（524/524 PASS）
- q4_K 8-gate 现状: `docs/reports/2026-07-09-q4k-8gate-status.md`（4/8 PASS）

**方法学严谨（背书 perf 主张）**:
- [CASE-MINTERM] 卷宗 + ggml 双路径备忘: commit `b05acc94`；`docs/reports/2026-07-09-ggml-q8-quant-dual-path-memo.md`
- M4 CASE CLOSED（kernel vs ggml generic 整数逐位一致）: commit `eede2ac3`；un-narrow + RESTORED: `02af804d`/`4f765790`
- min-fold 血缘 + cert 语料审计: `docs/reports/2026-07-09-minterm-fold-audit.md`（commit `4f8d815c`）
- 证书三要件 CI gate: commit `2363fd3c`；`tools/lint/check_cert_requirements.py` + `schema/cert-lineage.v1.json`
- ZERO-MODEL 终审法 spec: `docs/reports/2026-07-09-zero-model-adjudication-method.md`
- 传导账 + e2e projection: `experiments/active/kquant-family-closure/transmission_account.md`；
  T4b e2e 逐环归因: `docs/reports/2026-07-10-t4b-e2e-seal-integration-proven-kernel-variant-residual.md`（集成支柱证正确、单点 kernel-变体缺陷待修）

**板 / 措辞锁**: rvv / VLEN128（能力事实 `vlen=128`, `vreg_count=32`；declared_instance_hash `3cd23a4e…`）。
全文 kernel-轴、[NG-4] 非 beat、非 e2e、8 [PERF-1] 门未走；整模型 e2e = projection（measured BLOCKED）。**perf cite 为 correct-kernel（post-M4）。**
