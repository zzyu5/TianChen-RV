# [SEL-3] 测量记忆层 — 设计稿（案头·不施工核心）

> **性质**：纯案头设计稿。**无板·无核心代码改·无 git·无新计时**。落 schema 草案（DESIGN-DRAFT·非活 schema）+ 灌库流程 + 铁线 + 验收设计 + 排位。**禁动选择器核心代码**（`lib/`、`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h`、`VariantSelection.cpp` 不改）。
> **立项依据**：`docs/ROADMAP.md:31`〇.4 裁决（offline-profile argmin + byte-exact gate·存量 cold[T3/T9/onw2] 按 (instance-hash,kernel,variant) 灌库·冷启动查库 memoized→写回→字节门·铁线无搜索/无学习成本模型/不对标外部 tuner·验收 T4b 两列真数·排 A2/A3 后独立战役·本令先出 schema+灌库设计稿）。
> **权威链**：`docs/reports/2026-07-09-SEL-1-design.md`（②测量库 schema 蓝本）·`schema/tiling-measurements.v1.json`（已存在的 SEL-1 schedule-axis 实例·SEL-3 是其一般化）·`docs/canon/Weft-RV_科研目标总纲v2.md#[SEL-3]/[D-4]`·`.trellis/spec/variant-pipeline/generation-selection-tuning.md`（两段式选择器·权威顺序）·`.trellis/spec/architecture/core-invariants.md#I4`（实测=缓存事实非 authority）·`experiments/active/result-tables/T4b_selector_ablation.md`（验收去处）。
> **日期**：2026-07-15 · 案头执行员（度假自治·保守默认）。

---

## 0. 一句话定位

**[SEL-3] = 把已存在的离线实测（offline profile）沉淀成一张版本化、字节精确门守护的测量库，供选择器冷启动前查表命中 memoized argmin（`reason=measured`），miss 落 [SEL-1] 能力先验（`reason=prior`）。** 它是**证据轴镜像数据（core-invariants I4）**：**永不选 route/compute/dtype，永不是 `lib/` 的正确性/成本 authority**——权威顺序恒为 `硬件实测 > 本库缓存 > 静态先验`。它不是自动调优器，不学习成本模型，不与外部 tuner 对标。

**已落地的先行实例**：`schema/tiling-measurements.v1.json`（v1.1.0）已经是这套机制在**单一轴**（SEL-1 SP schedule：SP4 tiled-vs-plain + M1c loop-order）上的活 schema，含 in-memory view（`lookupTilingMeasurement`/`lookupLoopOrderMeasurement`）。**[SEL-3] 的净增量 = 把该机制一般化成跨轴的统一测量库 + 存量 cold 全量灌库 + T4b 两列真数验收**，而非从零造机制。

---

## 1. 三层选择轴 × 记忆层可兑现性（诚实分辨·避免夸大）

T4b（§1）已把选择器决策拆三层。测量记忆层能兑现的 novelty **逐层不同**，必须分开说，否则会把 Layer-1 的平凡冒充成 memoized win。

| 层 | 决策 | 候选集现状 | 记忆层今天能兑现什么 | 诚实定性 |
|---|---|---|---|---|
| **L1 codegen 变体**（LMUL/fraction） | mf2 vs m1 | **{mf2} 单一合法**（双板·m1=RVV0.7 跨-ISA 非 same-board·`[repack-winA-always-mf2]`） | **无**——单候选无翻盘空间，记忆列与常量分列给出相同 top-1/遗憾 0 | **退化 NULL**·记忆无 novelty 可宣 |
| **L2 tile-form / strip-width** | S6-tiled vs plain·vl=16 vs vl=8 | **有真"更快变体"**（q4_K@k1 vl=16 1.197×>vl=8·q2_K S6-tiled spill 收拢·q4_K@rvv col_outer 2.47×） | **真记忆内容**——这层才是 memoized argmin 有信息的地方（存量 tiling/loop-order 实测即住此） | **真价值·记忆层的实质居所** |
| **L3 paradigm/plan 形状 dispatch** | matrix vs vector·GEMM plan vs GEVM plan | 两族对同一 GEMM 形竞标（M-keyed） | **已落先验层**（[SEL-2] SEL-1-T5+T5c·M-aware）·记忆层**不越位**（形状 dispatch 是结构键控非实测 argmin） | **[SEL-2] 真·但归先验/结构轴·非记忆轴** |

> **★ 记忆层诚实边界**：**[SEL-3] 的实质记忆内容集中在 L2**（tile-form / strip-width / loop-order）。L1 退化（单候选）、L3 归结构/先验轴。所以**灌库后"记忆真能翻盘"的格数本身就少**——这是诚实的 scope，不是 bug。T4b 验收里"仅记忆/先验+记忆"两列的**非平凡差异只可能出现在 L2 覆盖格**（q4_K/q2_K/q5_K tiling · q4_K loop-order）；L1 九格的记忆列恒 = 先验列（都平凡最优·遗憾 0）。**不得把这层平凡包装成 memoized novelty。**

---

## 2. ① 测量库 schema 设计

### 2.1 主键

`primary_key = (declared_instance_hash, kernel, variant)`

- **`declared_instance_hash`** = `support::computeDeclaredInstanceHash(展开规范化能力事实集)` 的 SHA-256（lowercase hex）——**与 [D-4]① exec/schedule 归因 sink 同一 hash、同一 helper**（`DeclaredInstanceHash.h` 已存在）。跨符号改名可移植；profile≡显式能力列表两种声明哈希相等。**march/vlen/vreg 已被这枚 hash 蕴含**（它是能力事实集的哈希），故 hash 命中即"同板同能力档"命中。
- **`kernel`** = `decode_model`（tiling 叶身份，如 `q4_K`）。**注意**：选择器**从不**键控 format 名，它键控瓶颈形状（`fold_model`）；`kernel` 仅是测量行身份、非 dispatch 键。
- **`variant`** = 有界枚举成员（NG-1）。跨轴统一命名空间：`{plain, s6_tiled}`（SP4）∪ `{col_outer, row_outer}`（loop-order）∪ `{vl8, vl16}`（strip-width）∪ 未来轴的有界成员。**每轴的成员集封闭、编译期硬编码**，`variant_axis` 字段标注该 variant 属哪条轴（见 2.2）。

> **⚠ PENDING_RULING（键定义张力·只登记不执行）**：`docs/canon/Weft-RV_执行总纲v2.md:69` 的 [SEL-3] 行写"**记录键=kernel+march（非 instance-hash）**"，与 ROADMAP令〇.4 + `tiling-measurements.v1.json` 的 `(declared_instance_hash,kernel,variant)` **表述不一致**。二者**语义可调和**（instance-hash 蕴含 march/vlen/vreg，是 kernel+march 的超集精化），但**canon 条文字面冲突**属头条口径级。本设计稿**按 ROADMAP令 + 活 schema 的 `(instance-hash,kernel,variant)` 设计**（操作性指令 + 已落 schema 一致），并登记该 canon 措辞需用户裁决统一（建议：执行总纲v2:69 改注"键=declared_instance_hash〔蕴含 march/vlen/vreg〕+kernel+variant"以消歧）。**不擅改执行总纲。**

### 2.2 值形态（value shape）

每行记忆一枚 `(instance_hash, kernel, variant)` 实例的**过字节精确门的**冷测。字段（草案见 `schema/measurement-memory.design.json`）：

| 字段 | 语义 | 来源/映射 |
|---|---|---|
| `variant_axis` | 该 variant 属哪条选择轴：`sp4_tiling` \| `loop_order` \| `strip_width` \| （未来轴） | 编译期轴登记 |
| `selected` | 该行是否为其 `(instance_hash,kernel)`-组内 argmin（in-memory view 返回的赢家） | 由 `cold_median` 组内取 min 派生 |
| `cold_median` | 冷测中位（N≥10·T-N noise floor）。**语义 = 相/板/格式绑定的比值或 ns**（比值口径见 `ratio_semantics`） | T3 `rvv_cold`/`k1_cold`·T9·onw2 raw |
| `cold_iqr` | 冷测 IQR（离散度·稳定性护栏；IQR 过大 = 记忆不可信→退先验） | 存量 raw（有则填·seed 行缺则 sentinel） |
| `ratio_semantics` | `cold_median` 的分母口径 + **对手身份**：`vs_generic`（与 `_generic` 参考·选择合法输入）\| `vs_shipped_opp`（与出货对手·[CASE-COMPILER-ASYMMETRY] 系统账·**非**选择输入）\| `ab_paired`（同编译器 A/B·选择合法输入） | 逐行标·`opponent_symbol` |
| `opponent_symbol` | 对手 kernel 符号（如 `ggml_vec_dot_q4_0_q8_0`）+ 成色类（hand-brick / block-dot / parity-control） | T3 `rvv_opp_sym`/`k1_opp_sym` |
| `snapshot` | 板身份指纹：`{vlen, vreg_count, march, toolchain{ours, opp_shipped}, board_id}` | T3 board 列 + `[CASE-COMPILER-ASYMMETRY]` toolchain 维 |
| `byte_exact_gate` | `pass` — **行只在过字节精确门后写入**（fail-closed：未过门的计时永不缓存·both variant 与构造 oracle bit-identical） | 灌库门（见 §3.4） |
| `selection_valid_input` | bool — 该 `cold_median` 是否**编译器对称、可作选择输入**（`ab_paired`/`vs_generic` 同编译器 = true；`vs_shipped_opp` 跨编译器 = false·仅成色/披露） | 由 `ratio_semantics` + `snapshot.toolchain` 派生 |
| `source` | 存量出处（T8 行 / T3 行 / onw2 raw / 起源 commit） | 灌库溯源 |
| `ts` | ISO-8601 UTC；转录 seed 行 sentinel `"0"` | — |

> **★ 编译器对称性铁律入 schema（[CASE-COMPILER-ASYMMETRY]）**：`vs_shipped_opp` 口径的 `cold_median`（如 rvv seed 的 vs-gcc-shipped 1.884×）是 **clang-ours-vs-gcc-shipped artifact·`selection_valid_input=false`**——选择器**永不**键控它，它只是成色/披露注记。选择只键控 `ab_paired`（两腿同 clang·字节相同热核）与 `vs_generic`（同参考）。此约束直接从 `tiling-measurements.v1.json#case_compiler_asymmetry` 继承，一般化成每行显式布尔，防"某查询漏掉守卫字段→选择输入静默污染"。

### 2.3 决策规则（memoized argmin）

某 `(instance_hash, kernel)` 组的 memoized 赢家 = **`selection_valid_input=true` 且 `byte_exact_gate=pass` 的候选中 `cold_median` 最优者**；若并列/无决定性差异 → **更简单者胜**（plain / 无 tiling），即"边际非-cliff 削减不翻盘"（承 `tiling-measurements` decision_rule：q3_K tiled +8% 非 cliff → 记忆赢家仍 plain·"测量本身说别在这 tile"）。**结构决策门**（如 SP4 的 `register_cliff_reached`）作为**轴专属附加判据**挂在 `variant_axis=sp4_tiling` 行上（见 schema `axis_extras`），不污染通用 value。

### 2.4 与已落 `tiling-measurements.v1.json` 的关系

- `tiling-measurements.v1.json` = **[SEL-3] 在 SP schedule 轴的已落地实例**（`measurements[]` SP4 + `loop_order_measurements[]` M1c）。
- `measurement-memory.design.json`（本稿草案）= **一般化 superschema**：把 `tiling-measurements` 的 `measurements`/`loop_order_measurements` 两块统一到单张 `rows[]` + `variant_axis` 判别 + `axis_extras` 轴专属附加。
- **迁移策略（施工期·非本稿）**：`tiling-measurements.v1.json` 的 12 行按 `variant_axis∈{sp4_tiling, loop_order}` 平移进 superschema，`register_cliff_reached`/`weight_panel_larger_stream` 等落 `axis_extras`。**本稿不迁移、不改活 schema**——只出 superschema 草案供施工期落地。

---

## 3. ② 灌库流程设计

### 3.1 存量 cold 源清点（T3 / T9 / onw2）

| 源 | 文件 | 提供什么 | 映射到 value |
|---|---|---|---|
| **T3** | `experiments/active/result-tables/T3_master_rebuild.csv`（108 数据行·op/format/engine/regime/group + rvv_cold/rvv_opp_sym/k1_cold/k1_opp_sym） | 双板 cold 比值 + 对手符号 + 板身份（rvv=VLEN128/gcc-15·k1=VLEN256/clang-18） | `cold_median`←`{rvv,k1}_cold`·`opponent_symbol`←`{rvv,k1}_opp_sym`·`snapshot`←板锁 |
| **T9** | `experiments/active/result-tables/T9_kernel_sym_ledger.md`（kernel-sym ≥parity 台账·T3 派生·对手成色分层） | 对手成色类（hand-brick/block-dot/parity-control）+ 对称编译可行性标注 | `ratio_semantics`+`opponent_symbol` 成色类·`selection_valid_input` 佐证 |
| **onw2** | `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/raw/prof_onw2_full.err`·`c_onw2.out`（G6-M7 IME vmadot-tiling 板 profile raw） | IME tiling 冷测 raw（strip/tile-form 轴原始样本·可算 IQR） | `cold_iqr`（有 raw 分布）·IME 轴 `variant_axis` seed |
| **T8**（间接·已在活 schema） | `experiments/active/result-tables/T8_winloss_gap_ledger.csv` | SP4/loop-order seed（已入 `tiling-measurements.v1.json` 12 行） | 直接平移（§2.4） |

> **⚠ 灌库诚实护栏**：T3 的 cold 多为 `vs_shipped_opp` 或 `vs_generic` 口径的**单点比值**，**多数缺 IQR / N 分布**（T3 是收口台账非 raw 样本仓）。灌库时**缺 IQR → sentinel + `selection_valid_input` 保守判**；`vs_shipped_opp` 口径行一律 `selection_valid_input=false`（成色注记非选择输入）。**真正能作 memoized argmin 选择输入的存量行 = 有 `ab_paired` 同编译器 A/B 的 L2 tiling/loop-order 行（住 T8→已在活 schema）**——即 §1 的诚实边界：能兑现的记忆内容本就少。onw2 raw 可补 IQR 但需 raw 解析（施工期·非本稿；不占板不新测）。

### 3.2 灌库映射（存量→库·纯转录·无新测）

灌库 = **只读转录**存量 cold 进库行，**不触发任何新计时**（度假铁线·禁 e2e/新计时）：
1. 遍历 T3 每 `(op,format,engine)` × 板 → 生成 `(instance_hash, kernel, variant=该板部署 variant)` 行。`instance_hash` 由该格 @板 fixture 展开的能力事实集算（施工期用 `computeDeclaredInstanceHash`·转录期填 seed sentinel 或引 T8 已知 hash `3cd23a4e…`）。
2. `cold_median`←cold 列；`opponent_symbol`←opp_sym 列；`snapshot.toolchain` 按板锁（rvv=gcc-15 shipped / k1=clang-18 shipped）+ ours（rvv=clang-17 / k1=clang-18）。
3. `ratio_semantics` 按对手身份判：对手=`_generic`→`vs_generic`；对手=出货 ggml→`vs_shipped_opp`（跨编译器→`selection_valid_input=false`）；同编译器 A/B→`ab_paired`。
4. `byte_exact_gate`：**仅转录已知过门的行**（T8 seed 全 pass；T3 未过门格 → 不入库或标 `pending` 且 `selection_valid_input=false`）。

### 3.3 冷启动查库闭环（设计·不施工核心）

选择器（**核心不动**·此处只描述目标行为）冷启动前：
```
lookup(instance_hash, kernel, variant_axis):
  rows = library[(instance_hash, kernel)] filtered by variant_axis
  candidates = rows where byte_exact_gate==pass AND selection_valid_input==true
                    AND variant still feasible under current capability facts   # fail-closed-revalidate
  if candidates nonempty:
      return argmin(candidates, cold_median).variant, reason=measured
  else:
      return None  ->  fall to [SEL-1] capability prior, reason=prior
```
- **fail-closed-revalidate**（承活 schema）：缓存赢家若在当前能力事实下已不 feasible（Stage-1 合法性复检失败）→ **丢弃、退先验**；陈旧缓存永不强制不合法 variant。
- **写回闭环**：offline-profile 谐调器（on `ssh rvv`/`ssh k1`·施工期·非本稿）新测一枚 `(instance,kernel,variant)`→过字节门→写回库行（`ts` 真时戳·`selected` 重算组内 argmin）。**写回=离线批处理产物、非在线学习**（NG-1）。

### 3.4 产物过字节精确门（fail-closed）

- **门定义**：每写入行的 variant 产物必须与 `_generic` 构造 oracle **逐字节相等**（ZERO-MODEL 复算·证书三要件：语料完备/输入路径同源/oracle 独立）。未过门 → **永不缓存**（fail-closed）。
- **门在灌库/写回两处都强制**：存量转录只收已过门行；新写回先过门再落盘。
- **门=正确性 authority，库=缓存事实**：两者分离（I4）——库里 `cold_median` 只排名，从不判正确。

---

## 4. ③ 铁线明文（NG-1 / [L-4] / I4）

1. **无搜索**：variant 集 = 有界枚举（编译期硬编码封闭成员）。精化 = **实测记忆**（offline profile 批处理），**非在线学习、非 autotune 搜索、非 curve-fit 成本模型**。
2. **无学习成本模型**：库不含可学习参数；静态 cost 公式仅作 (1) Stage-1 剪枝 pruner + (2) 无记录时 fallback 默认，**非冷启动排序 authority**（排序 authority = [SEL-1] 能力先验层）。
3. **不与外部 tuner 对标 [L-4]**：oracle 恒为**自体**（每格自身变体穷举的 micro 最优），**不跨格/跨板外推、不与 TVM/AutoTVM/Triton-autotune 等外部调优器对标**。留外部 tuner 插点**仅文档化**（`external_tuner_hook`：一个可选的离线 profile 入口 schema 位·默认空·不实现）。
4. **动选择器核心走前门 + falsifier 全过**：记忆层落地时选择器核心改动须过 [SEL-2] 硬时序 falsifier + T4b 消融判别力变异测试 + 全量 byte-exact 回归 + [PERF-1] 八门。**本稿不改核心**——核心改动是后续 task（§6）。
5. **实测=缓存事实非 authority（I4）**：权威顺序 `硬件实测 > 本库 > 静态先验`；库**永不**选 route/compute/dtype，**永不**是 `lib/` 正确性/成本 authority。
6. **编译器对称才作选择输入**：`vs_shipped_opp` 跨编译器口径 `selection_valid_input=false`（成色/披露注记）；只有 `ab_paired`/`vs_generic` 同编译器口径可作 memoized argmin 输入（[CASE-COMPILER-ASYMMETRY]）。

---

## 5. ④ 验收设计（T4b 两列真数 + 共竞标遗憾归零）

### 5.1 T4b "仅记忆 / 先验+记忆" 两列填数目标

T4b（`experiments/active/result-tables/T4b_selector_ablation.md` §2.1/§2.2）现两列标 **N/A(gated on [SEL-3])**。[SEL-3] 落地后填**真数**：

| 轴 | 仅记忆 | 先验+记忆 | 预期（诚实预判·非结论） |
|---|---|---|---|
| **L1 codegen（§2.1·9 格·{mf2} 单一）** | top-1 **100%**·遗憾 **0%**·逃逸 **100%** | 同左 | **平凡=先验列**（单候选无翻盘·记忆无 novelty）·**诚实 NULL 定格**（预判·验收板测确认） |
| **L2 tile-form（q4_K/q2_K/q5_K tiling·q4_K loop-order）** | 命中 memoized argmin 的 top-1（S6-tiled 于 cliff 达成格·col_outer 于 weight-panel-larger 格） | ≥ 仅先验（记忆覆盖先验·byte-exact gated） | **记忆列的非平凡差异只在此轴**·量级 gated on 存量/写回实测（q4_K col_outer 2.47× A/B 已在 T8） |
| **L3 paradigm（§2.2·2 场景）** | N/A（记忆不越位形状 dispatch·结构键控非实测 argmin） | = 仅先验（形状 dispatch 归 [SEL-2] 先验/结构轴） | 记忆层**不认领** L3·[SEL-2] 已落先验层 |

> **验收诚实标注**：**"仅记忆/先验+记忆"两列的非平凡真数只可能出现在 L2 覆盖格**。L1 九格记忆列恒=先验列（遗憾 0·平凡）；L3 归先验轴。**不得把 L1 平凡/L3 先验冒充成 memoized novelty。** 两列填数 = **诚实定格**（L1 NULL·L2 真差·L3 不认领），非"记忆全面翻盘"。

### 5.2 共竞标专项行·记忆配置遗憾归零

T4b §3 共竞标专项（矩阵/向量对同 GEMM 竞标·[SEL-2] 直证）：记忆配置在该专项行的**遗憾须归零**——即 memoized argmin **不得**把已由 [SEL-2] 先验修好的 paradigm dispatch 又选回错范式（记忆不越位形状轴）。验收断言：**共竞标行的记忆列 verdict == 先验列 verdict**（记忆不劣化结构修复）；若记忆命中一枚过时/跨编译器行企图翻回 vector → `fail-closed-revalidate` + `selection_valid_input=false` 拦下 → 退先验 → 遗憾 0。**变异测试**：注入一枚 `selection_valid_input=false` 的伪 vector-win 记忆行，断言选择器仍出 matrix（记忆不污染 [SEL-2] 修复）。

### 5.3 验收产物形态

- T4b 两列由 N/A → 真数（L1 NULL/L2 真差/L3 不认领·逐格标口径）。
- 共竞标行记忆列 verdict==先验列 verdict（断言）+ 伪记忆行变异测试（判别力）。
- 全程 byte-exact 回归绿 + reason 归因 JSONL 每 K-quant 格 ∈ {prior, measured, only_feasible}·**非 static_order/hardcode**（[D-4] 燃减不变量）。

---

## 6. ⑤ 排位（独立战役·A2/A3 后）

- **依赖**：`docs/ROADMAP.md:31` 节点队列 = A5 收口 → e2e 传导战役 → **[SEL-3] 施工** → 论文素材冻结点。本稿 = 施工前的 schema+灌库设计稿（〇.4 令"本令先出 schema+灌库设计稿[案头不动核心]"）。
- **施工期 task 分解（供后续·本稿不实现）**：
  1. **T-SEL3-1 schema 落地**：`measurement-memory.design.json` → 活 schema `measurement-memory.v1.json`（迁 `tiling-measurements.v1.json` 12 行·`variant_axis`+`axis_extras`）+ VERSIONLOG。
  2. **T-SEL3-2 灌库转录**：存量 T3/T9/onw2 cold 只读转录进库（无新测·§3.2 映射·缺 IQR sentinel·`vs_shipped_opp` 标 `selection_valid_input=false`）。
  3. **T-SEL3-3 in-memory view 一般化**：`lookupMeasurement(instance_hash,kernel,variant_axis)`（承 `lookupTilingMeasurement` 范式·**核心改动·走前门+falsifier**）。
  4. **T-SEL3-4 写回谐调器**：offline-profile on `ssh rvv`/`ssh k1`（byte-exact 门·离线批处理·非在线）。
  5. **T-SEL3-5 T4b 验收填数**（§5·两列真数+共竞标遗憾归零+变异测试）。
- **本稿交付边界**：① schema 草案（`schema/measurement-memory.design.json`·DESIGN-DRAFT）② 本设计稿。**核心代码零改·git 零 commit·无板·无新测。**

---

## 附 touch-set 纪律 / PENDING 登记

- **触碰集**：仅本文件 + `schema/measurement-memory.design.json`（新建·DESIGN-DRAFT·非活）。**不改** `schema/tiling-measurements.v1.json`（活）、`lib/`、`include/…RVVRepackTilingSelection.h`、`VariantSelection.cpp`、任何 canon 分母/$meta/头条。与其他 agent 触碰集不相交。
- **PENDING_RULINGS 建议登记**（硬冻结·只登记不执行）：
  1. **[SEL-3-KEY] canon 键定义措辞统一**：`执行总纲v2.md:69` 的"键=kernel+march（非 instance-hash）" vs ROADMAP令+活 schema 的 `(instance-hash,kernel,variant)`。二者语义可调和（instance-hash⊇march），但 canon 字面冲突需用户裁一（建议改执行总纲v2:69 注为"declared_instance_hash〔蕴含 march/vlen/vreg〕+kernel+variant"）。本稿按操作性指令设计、不擅改执行总纲。
- **无新数字**：所有 cold/parity/spill 数引自 T3/T8/T9/`tiling-measurements.v1.json` 已存在 cell·rvv/k1 board 已测·本稿零新计时。
