# A1 · 公式 authority 冻结（HEAD 真实链）

> 机器正本：[`authority-matrix.v1.json`](./authority-matrix.v1.json)；门：
> `test/Scripts/formula-authority-matrix.test`。
>
> 本文是 **characterization**，不是目标架构，也不是“已有即合规”的验收单。
> 它回答每个决定今天实际上在哪里发生、什么已经承重、什么仍是双 authority。
> 后续 A2–A5/A8 每退役一条旧路径，必须把代码、矩阵、测试和 issue 原子更新；
> 不保留 adapter、compat alias、缺 stamp 默认或“新旧都能跑”的中间世界。

## 1. 先给结论

当前资产比早期文档强得多，但还没有形成完整的
`typed g/c/ω → formula → legal set → select → selected stamp → mechanical emit`
链：

- 五类 dequant plan **不是待创建**：`NibbleDecodePlan`、
  `CodebookGatherPlan`、`KQuantScaleMinPlan`、`GridLookupPlan`、
  `TernaryDecodePlan` 均已定义、被 provider 构造、被真实 emitter 消费，并各有
  load-bearing mutation test。
- 五类 plan 的共同缺口也很具体：emitter 用字面量 `128` 调 provider，五个
  provider 都显式忽略 `minimumVLEN`；selected plan 只存在于 emission 期间的
  C++ 栈对象，没有 emission 前的 typed stamp。
- dequant 的 g 已部分结构化，但 codebook/K-quant/grid/ternary 的 leaf 仍由
  emitter 内四段 `decode_model → enum` `StringSwitch` 再选一次；GridLookup 的
  legality 又直接查第二个 `GridDecodePlan` registry head。
- repack accumulator LMUL 已有真实 c、解析 legality、实测 winner 和 fail-closed
  emitter 消费；问题是 selector 与 reason stamp 被复制到 18 个 builder，实测表
  仍是生产 C++ 手工镜像。
- SP4/loop-order 已经有两阶段选择与 IR stamp，但 emission 还不是机械消费：
  SP4 legal set 会放入当前根本没有 emitter body 的 min-fold `plain`；缺 stamp
  会静默落到 `S6Tiled`；loop-order 对 sibling 会把已选择的 `col_outer/prior`
  改回 `row_outer`，q4_K 缺 stamp 时会重新计算 stride prior。

所以，当前的准确表述不是“公式层没有”，也不是“公式层已经完成”，而是：

> 五类机制 plan 与若干真实公式/selector 已经是可复用资产；尚缺的是统一的
> typed decision contract、真实 c 消费、单一 legality/selection authority、
> selected result 落印，以及 emitter 的纯 realization。

## 2. 状态词

| 标记 | 本文含义 |
|---|---|
| analytic | 由 g/c 的闭式关系或确定性规则推出 |
| measured | 来自离线合格实测 winner，而非 candidate/legality 的来源 |
| constant | 当前机制/ABI 的结构常量；不是自动等于坏硬编码 |
| honest-null | 该决定当前没有物理上成立的 c 或 ω 分叉，不制造假旋钮 |
| fallback | 输入不足或未命中时的 total 结果；必须与 compat/default 区分 |

## 3. 五类 dequant plan 的真实状态

### 3.1 共同链

五类共享的真实上游是：

```text
GgmlDequantizeRowOp.format
  → lookupDequantizeRowStreamFacts(format)
  → constructTypedDequantizeRowLoopBody
  → DequantizeRowDecodeCoreOp（落 g）
  → emitTypedDequantizeRowLoopBody
  → mechanism-specific FormulaProvider（临时 plan）
  → mechanism-specific emitter body
```

这条链证明“plan 真消费”已经成立；它也暴露两个未闭环点：provider 仍在
`emitTypedDequantizeRowLoopBody` 内调用，且 c 不是从 capability provider 随 selected
body 一起送达，而是统一写死为 128。

| plan | defined | g stamped | provider consumed | selected plan stamped | emitted | mutation tested | c 真消费 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NibbleDecodePlan | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗（128 + ignored） |
| CodebookGatherPlan | ✓ | ✓（部分） | ✓ | ✗ | ✓ | ✓ | ✗（128 + ignored） |
| KQuantScaleMinPlan | ✓ | ✓（primary g） | ✓ | ✗ | ✓ | ✓ | ✗（128 + ignored） |
| GridLookupPlan | ✓ | ✓（entry lanes） | ✓ | ✗ | ✓ | ✓ | honest-null，但参数仍为假 seam |
| TernaryDecodePlan | ✓ | ✓（iq1 entry lanes） | ✓ | ✗ | ✓ | ✓ | honest-null，但参数仍为假 seam |

### 3.2 NibbleDecode

- **当前实际 owner**：`nibbleDecodePlanFromFacts`。
- **g**：qk、block stride、scale/quant offset、carrier、bias、min/qh optional
  offsets；由 `DequantizeRowStreamFacts` 进入 typed core。
- **公式**：`stripLanes = qk / 2` 是真 analytic 派生；`loadLMUL = m1` 是当前
  reproduce-current constant。
- **legality/selection**：没有 typed legality 结果；emitter 只接受 m1。carrier
  在 BareInt8 与 Nibble4 两个既有 leaf 之间选择。
- **消费证据**：修改 bias 会改变 `vsub`，修改 qk 会经 `qk/2` 改变 emitted VL。
- **缺口**：selected plan 不落印；c 参数不承重。

### 3.3 CodebookGather

- **当前实际 owner**：`codebookGatherPlanFromFacts`。
- **g**：qk、stride、quant offset 来自 typed core；`CodebookScaleModel` 却由
  emitter 的 `decode_model` `StringSwitch` 产生。
- **公式资产**：`getRVVCodebookGatherAnchorLMUL(VLEN, SEW, entries)` 已存在，
  但本 provider 没有调用它；当前 `loadLMUL = m1`，legality 仅为
  `!loadLMUL.empty()`。
- **消费证据**：quant offset 与 `qk/2` strip mutation 都能改变 emitted C。
- **缺口**：这是 A3 最适合的首个真 `g+c` slice——不是再写一个公式，而是让
  已有 closed form 接管 plan/legal set，并在 emission 前落印。

### 3.4 KQuantScaleMin

- **当前实际 owner**：`kquantScaleMinPlanFromFacts`。
- **g**：primary block facts 已进 typed core；`KQuantScaleModel` 仍由 emitter
  的 `decode_model` `StringSwitch` 产生。
- **公式**：`minByteOffset = scaleBlockByteOffset + 2` 是真 analytic 派生；
  sub-scale/high-bit offsets、loadLMUL/stripLanes 仍由 scale-model 常量表选择。
- **消费证据**：scale offset mutation 同时改变 d 与派生 dmin；quant offset
  mutation 改变 load address。
- **缺口**：c 未消费，selected plan 未落印；per-leaf g 仍需 A8 分类后迁出
  emitter，不能把全部结构差异硬塞进万能 plan。

### 3.5 GridLookup

- **当前实际 owner**：`gridLookupPlanFromFacts`。
- **g**：entry lanes 已是 typed descriptor 且 load-bearing；leaf 仍由 emitter
  字符串映射产生。
- **legality**：provider 直接调用 `lookupGridDecodePlan(decodeModel)`。这让
  dequant plan head 与 block-dot registry head 在同一决定上形成双头；它不是
  “共享一份公式”的终态。
- **c/ω**：当前 narrow grid body 没有被证明存在 c 分叉，记 honest-null；禁止
  为了让公式看起来复杂而制造假 LMUL 字段。
- **消费证据**：entry lanes 8→11 改 emitted VL；缺字段 fail closed。

### 3.6 TernaryDecode

- **当前实际 owner**：`ternaryDecodePlanFromFacts`。
- **g**：leaf 由 emitter 字符串映射；iq1 的 entry lanes 已 typed/stamped，tq
  算术 leaf 无此字段。
- **legality**：plan 写 `true`；iq1 的 entry-lane presence 由 verifier/emitter
  另行 fail close。
- **c/ω**：现有 body 没有真实 c/measurement 分叉，记 honest-null。
- **消费证据**：iq1 entry-lane mutation 与 missing-field negative 均已存在。

## 4. 三个非 dequant 决定

### 4.1 Repack accumulator LMUL

```text
scale_model g
+ resolved RVV version / halfLanes / vreg_count c
+ per-shape measured row ω
  → selectRepackAccumulatorLMUL
  → {useM1, reason}
  → integer_core_lmul + reason stamp
  → 18 fail-closed emitter reads
```

这是八项里目前最接近目标链的一项：

- RVV0.7 无 fractional LMUL，m1 是 correctness-only feasible result；
- RVV1.0 下 m1 先过同一 register-pressure inequality；
- 有手工实测 row 时用 measured winner，否则回 `capability-default-mf2`；
- emitter 缺 `integer_core_lmul` 已 fail closed，不再 `value_or("mf2")`。

剩余问题不是“公式不存在”，而是 contract 没有模块化：相同 selector 调用、
typed 参数组装和 reason stamp 在 18 个 builder 重复；实测表也不是由 B1 正式
measurement control plane 生成的 qualified view。A2 应拿它做第二个 vertical
slice，A5 再替换手工 winner mirror。

### 4.2 SP4 tiling

```text
fold_model → bottleneck shape g
+ {minimum_vlen, vreg_count} c
+ lookupMeasurement(hash,kernel,SP4) ω
  → tilingVariantFeasibleSet
  → selectRepackTilingVariant
  → tiling_variant/reason/record stamp
  → emitTypedRepackGemmLoopBody
```

已经成立的部分：bounded enum、capability feasibility、measured-first/prior-second、
production 零 `static_order`、同一 declared-instance hash attribution。

但当前 legal set 与 realization 不一致：`tilingVariantFeasibleSet` 在能力充分时
无条件返回 `{Plain,S6Tiled}`；min-fold emitter 明确没有 Plain body，选到 Plain
只会在 emission 阶段失败。另一个缺口是 `tiling_variant` 不在 ODS/verifier 中，
缺 stamp 会静默实现 S6Tiled。故不能把它写成“formula→legalize→select 已闭环”。

### 4.3 Loop order

```text
{weightStride,activationStride,prefill} g
+ {minimum_vlen,vreg_count} c
+ lookupMeasurement(hash,kernel,LoopOrder) ω
  → selectRepackLoopOrder
  → loop_order/reason/record stamp
  → two emitter read sites
```

selector 本身清晰：measured hit 优先，否则用 layout stride prior。问题发生在
realization：

- q4_K 缺 stamp 时再调用 `repackColGroupOuterForLayout` 重算；
- sibling 即使 stamp 为 `col_outer/prior`，也只在 reason=`measured` 时实现
  col_outer，否则写 override comment 后实际实现 row_outer；
- 该 attr 同样不受 ODS/verifier 约束。

这不是普通 provenance 差异，而是 `selected != realized`。现有
`rvv-to-emitc-repack-gemm-q6-K-q8-K-col-outer-prior-override.mlir` 正在把这项
分裂钉成当前行为；A4 完成后，该测试必须反转为“选择什么就实现什么”，而不是
继续保留 override 兼容逻辑。

## 5. 后续施工切割面

| 真实断点 | 后续 owner | 必须删除的旧点 | 杀死旧点的验收 |
|---|---|---|---|
| dequant c=128 且五 provider ignore c | A3 / ISSUE-117 | 五个 literal-128 call 与 declared slice 的 ignored seam | VLEN/capability mutation 真翻 plan/legal set；missing/conflict fail closed |
| emitter 用 decode_model 再造 leaf/facts | A8 / ISSUE-119 | 四段 StringSwitch 与 emitter facts reconstruction | 新 typed-g leaf 不改 emitter branch；missing/forged g 转红 |
| selected dequant plan 不落印 | A4 / ISSUE-122 | emission 内 provider invocation | stale/forged stamp 在 emitter 前被拒 |
| GridLookup 再查 GridDecodePlan | A4 / ISSUE-122 | dequant slice 的 direct registry lookup | 单 row 变异同时支配 verifier/selector，无第二编辑点 |
| LMUL 18 点重复组装/盖章 | A2 / ISSUE-117 | 18 个 direct call/stamp site | 全 caller 经一个 typed decision constructor；旧 caller=0 |
| 手工 measurement mirrors | A5 / ISSUE-117 | LMUL table 与 `kSeeded[]` | generated qualified view 可复建；stale/key/illegal winner miss |
| SP4 legal candidate 不可实现 | A4 / ISSUE-125 | unconditional feasible set 或缺失的 Plain body 二选一闭合 | selector 永不返回 emitter 会拒的候选 |
| SP4 缺 stamp→S6 | A4 / ISSUE-125 | optional read/default | strip stamp 必须 fail closed |
| loop selected→emit override/recompute | A4 / ISSUE-125 | reason-measured gate、q4 recompute、optional read | prior col_outer 真实现 col_outer；缺 stamp fail closed |

## 6. A1 的测试冻结

本轮没有复制已有的大量 MLIR fixture，而是把真正承重的现有测试列为机器契约，
并新增一个 matrix gate：

- 五类 dequant 各自已有至少两个字段 mutation 或 missing-field negative；
- LMUL 有 VLEN/provider 判断、measured/default 分叉和 missing-stamp fail-close；
- SP4 有 measured/prior/shape-isolation 与生产零 static-order；
- loop-order 有 ternary 全格 stamp guard，也有当前 selected→realized override 的
  反例 characterization；
- missing capability 由 stage-B selection 的默认空 march/VLEN0 路径钉住。

`check-formula-authority-matrix.py --self-test` 在内存中分别删除一个 decision、
篡改一个 source census、断开一个 task binding；三种 mutation 都必须被门抓住。
因此该矩阵不是叙事表：后续删旧 symbol 而不更新对应 contract 会立即转红。

## 7. A1 之后的正确顺序

1. A2 先建立最小 typed decision contract，并完整迁一个 dequant slice与 LMUL
   slice；同 slice 旧入口当笔删除。
2. A3 让 codebook anchor 成为首个真实 `f(g,c)`，不强迫 grid/ternary 的
   honest-null 轴伪装成 c-driven。
3. A4 原子关闭 selected-stamp、legality 与 emitter redecision 缺口；优先处理
   ISSUE-125，因为它已经存在 `selected != realized`。
4. A5 用 B1 生成的 qualified winner view 取代手工表；measurement 只能从合法
   候选中选，不能创造 mechanism/candidate。
5. A8 再做 baked-g 全量收敛；结构常量按 [K-10] 保留，真实 g 迁入 typed owner，
   dead/wrong 路径直接退役。

每一步都按 RET-1：可以在独立 worktree 分片施工，但 declared slice 只有在全部
production caller 切换、旧入口与默认路径删除、mutation tests 转绿后才能合入。
