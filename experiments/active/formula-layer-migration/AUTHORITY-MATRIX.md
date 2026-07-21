# 公式 authority matrix（A1 基线，A3 后 HEAD 真实链）

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
- Nibble 已完成 A2 typed-decision cutover：`NibbleDecodeGeometryFacts` 是显式 g，
  c/ω 用空的具名类型表达 honest-null，旧 `nibbleDecodePlanFromFacts`、假
  `minimumVLEN` 参数与 literal-128 caller 均为 0。
- Codebook 已完成 A3 vertical cutover：canonical typed g 与 selected-provider c
  进入 `decideCodebookGather`；formula 在 bounded declared `{mf2,m1,m2}` realization
  set 内给出 legality 与最窄合法 anchor；ω honest-null。唯一 backend preparation
  materializer 在 emission 前写/核 complete stamp，direct 与 registry 路共用；
  emitter 只机械消费。旧 provider、literal-128/ignored seam 与 emitter-local
  codebook `decode_model → scale model` mapping 均为 0。
  四种 fixed ABI tuple 只有一张 shared layout row；direct/registry capability gate
  也共用 canonical set、唯一 collector 与 token predicate，无第二 parser/first-wins。
- K-quant/grid/ternary 的 leaf 仍由 emitter 内三段 `decode_model → enum`
  `StringSwitch` 再选一次；GridLookup 的 legality 又直接查第二个
  `GridDecodePlan` registry head。Nibble/KQuant/Grid/Ternary 的 selected plan 仍是
  transient，不能把 Codebook 的闭环外推成五类闭环。
- repack accumulator LMUL 也已完成 A2 cutover：`lowerOne` 每个请求只构造一次
  typed decision，18 个互斥 builder 只消费 selected result，reason/key 由一个
  helper 盖章；旧 selector、choice struct 和 18 个独立盖章点已退役。实测表仍是
  生产 C++ 手工镜像，归 A5。
- SP4/loop-order 已经有两阶段选择与 IR stamp，但 emission 还不是机械消费：
  SP4 legal set 会放入当前根本没有 emitter body 的 min-fold `plain`；缺 stamp
  会静默落到 `S6Tiled`；loop-order 对 sibling 会把已选择的 `col_outer/prior`
  改回 `row_outer`，q4_K 缺 stamp 时会重新计算 stride prior。

所以，当前的准确表述不是“公式层没有”，也不是“公式层已经完成”，而是：

> 五类机制 plan、三个机制专属 typed decision 与首个完整 c-driven dequant
> vertical slice 已是可复用资产；尚缺的是把 contract 推广到其余真实轴、关闭
> Grid/schedule 的 legality/selection 双头，并让剩余 selected result 落印、emitter
> 收口为纯 realization。

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

五类共享的 construction 上游是：

```text
GgmlDequantizeRowOp.format
  → lookupDequantizeRowStreamFacts(format)
  → constructTypedDequantizeRowLoopBody
  → DequantizeRowDecodeCoreOp（落 g）
```

此后 Codebook 走
`backend prepare → typed g/c formula → complete selected stamp → mechanical emit`；
Nibble/KQuant/Grid/Ternary 仍在 `emitTypedDequantizeRowLoopBody` 内构造 transient
plan。Nibble 已明确为 honest-null c/ω；Grid/Ternary 当前 c 也 honest-null；不能因
Codebook 有真实 capability 分叉就给其它机制制造假 c。

| plan | defined | g stamped | provider consumed | selected plan stamped | emitted | mutation tested | c 真消费 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NibbleDecodePlan | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | honest-null（typed 空轴；无假 VLEN seam） |
| CodebookGatherPlan | ✓ | ✓ canonical | ✓ | ✓ complete | ✓ mechanical | ✓ | ✓（minimum VLEN、SEW8/32、emitted LMUL chain） |
| KQuantScaleMinPlan | ✓ | ✓（primary g） | ✓ | ✗ | ✓ | ✓ | ✗（128 + ignored） |
| GridLookupPlan | ✓ | ✓（entry lanes） | ✓ | ✗ | ✓ | ✓ | honest-null，但参数仍为假 seam |
| TernaryDecodePlan | ✓ | ✓（iq1 entry lanes） | ✓ | ✗ | ✓ | ✓ | honest-null，但参数仍为假 seam |

### 3.2 NibbleDecode

- **当前实际 owner**：`decideNibbleDecode`（`RVVFormulaDecision.h`）。
- **g**：qk、block stride、scale/quant offset、carrier、bias、min/qh optional
  offsets；由 `DequantizeRowDecodeCoreOp` 投影为 `NibbleDecodeGeometryFacts`。
- **公式**：`stripLanes = qk / 2` 是真 analytic 派生；`loadLMUL = m1` 是当前
  sole-realizable structural constant。
- **c/ω**：`NibbleDecodeNoCapabilityInput` 与 `NibbleDecodeNoStaticContext` 明确
  honest-null；旧 `minimumVLEN` seam 已删除。
- **legality/selection**：非法 qk/stride/offset fail closed；合法域只有一个 analytic
  plan。emitter 只接受 m1；carrier 在 BareInt8 与 Nibble4 两个既有 leaf 间选择。
- **消费证据**：修改 bias 会改变 `vsub`，修改 qk 会经 `qk/2` 改变 emitted VL。
- **缺口**：selected plan 仍未在 emission 前落印，归 A4；这不是 c 欠工。

### 3.3 CodebookGather

- **当前实际 owner**：`decideCodebookGather`；production projection 与完整 stamp
  唯一住 `materializeRVVCodebookGatherPlans`。
- **g**：`CodebookGatherGeometryFacts`=
  `{scaleModel,qk,weightBlockStride,scaleByteOffset,quantByteOffset}`，全部来自
  construction-owned typed core；post-construction formula invocation 由 typed
  `dequant_mechanism=codebook-gather` 分类，不由 `decode_model` 再选机制。四种固定
  ABI geometry 只住 `CodebookGatherLayoutFacts` 一张 row：construction 与 formula
  共同读取，避免同一 tuple 双写。
- **c/ω**：`CodebookGatherCapabilityFacts`=
  `{minimumVLEN,SEW8/32 support,MF2/M1/M2/M4/M8 support}`，由 selected RVV provider
  的 typed properties 唯一投影；ω=`CodebookGatherNoStaticContext` honest-null。
  allow-list absent 是既有 base-V silent gate，explicit empty/坏类型/未知 token
  reject；fractional LMUL 需显式 token 或 RVV1.0 正证；可选 tail/mask policy 是
  typed 单值 enum，显式空、错类型、未知值均拒绝。
- **公式/合法域**：candidate set 是已有 realization 的 `{mf2,m1,m2}`；每个 anchor
  必须满足 `VLMAX_e8 >= 16`，且 emitter 实际直接 `vsext_vf4` 的目的 LMUL
  `{m2,m4,m8}` 可用。selector 取该声明集内最窄合法 anchor；VLEN64→m2、
  VLEN128→m1、VLEN256+fractional→mf2。该声明不覆盖未来所有理论 fractional LMUL，也没有用
  未证实 register-pressure 数字充当 correctness legality。
- **stamp/consumer**：backend preparation 写或核
  `{table,entries,strip_lanes,load_lmul,minimum_vlen,provider,selection_reason}`；partial、
  forged、stale stamp fail closed。emitter 不查 capability、不调 formula，只把完整
  stamp 机械投影为 `CodebookGatherPlan`。parent/core source provenance 必须一致，但
  `decode_model` 不再控制 Codebook emission gate 或头文件副作用。
- **证据与剩余域**：direct/registry VLEN64/128/256 decisive tests、capability/stamp
  negatives、现役 VLEN128 byte-exact goldens已覆盖。真实速度 disposition 归 B4；
  formula-independent 全家 verifier 与其它 dequant slice 不属于 A3。
- **同名资产边界**：`getRVVCodebookGatherAnchorLMUL` 仍服务
  `RVVToEmitCCodebookFp4.cpp` 的 block-dot/loop realization，并有独立 unit test；
  A3 退役的是 dequant-row 的 `codebookGatherPlanFromFacts` 与 emission-period caller，
  不是越界删除另一条已部署路径的 helper。

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
weightInterleave g
+ {hasFractionalLMUL, halfLanes, vreg_count} c
+ optional qualified {typed key, winner} ω
  → decideRepackAccumulatorLMUL
  → legal {mf2,m1} + analytic prior + selected typed result
  → integer_core_lmul/half_lanes + centralized reason/key stamp
  → 18 fail-closed emitter reads
```

这是 A2 已完成的完整 typed-decision slice：

- RVV0.7 无 fractional LMUL，mf2 非法，m1 是 correctness-only feasible result；
- RVV1.0 下 mf2/m1 都由同一 register-pressure inequality 给出显式 verdict；
- 合格实测 winner 只有仍在合法集内才命中，否则回 analytic mf2 prior；
- RVV generation 显式投影为 optional capability；Unknown 不折成 RVV1.0；
  missing/invalid capability 与空合法集 fail closed；
- `lowerOne` 只构造一次 decision，18 个互斥 builder 消费同一个 selected result；
- reason 与可选 measurement key 只由一个 helper 盖章；
- emitter 缺 `integer_core_lmul` 已 fail closed，不再 `value_or("mf2")`。

旧 `RepackAccumulatorLMULChoice`、旧 selector 及 18 个独立选择/盖章点均为 0。
剩余问题只有实测表尚未由 B1 正式 measurement control plane 生成 qualified view；
A5 负责替换手工 winner mirror，不回滚本 contract。

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
| KQuant/Grid/Ternary 仍带 literal c seam | A8 / ISSUE-117 | 三个 literal-128 call 与 ignored seam；按 decisive 或 honest-null 分类，不制造假轴 | decisive 轴有 capability mutation；honest-null 轴删除参数而非伪翻转 |
| emitter 用 decode_model 再造 leaf/facts | A8 / ISSUE-119 | KQuant/Grid/Ternary 三段 StringSwitch 与 emitter facts reconstruction | 新 typed-g leaf 不改 emitter branch；missing/forged g 转红 |
| Nibble/KQuant/Grid/Ternary selected plan 不落印 | A4 / ISSUE-122 | 对应 emission 内 provider invocation/transient plan | stale/forged stamp 在 emitter 前被拒；Codebook 作为 reference regression |
| GridLookup 再查 GridDecodePlan | A4 / ISSUE-122 | dequant slice 的 direct registry lookup | 单 row 变异同时支配 verifier/selector，无第二编辑点 |
| 手工 measurement mirrors | A5 / ISSUE-117 | LMUL table 与 `kSeeded[]` | generated qualified view 可复建；stale/key/illegal winner miss |
| SP4 legal candidate 不可实现 | A4 / ISSUE-125 | unconditional feasible set 或缺失的 Plain body 二选一闭合 | selector 永不返回 emitter 会拒的候选 |
| SP4 缺 stamp→S6 | A4 / ISSUE-125 | optional read/default | strip stamp 必须 fail closed |
| loop selected→emit override/recompute | A4 / ISSUE-125 | reason-measured gate、q4 recompute、optional read | prior col_outer 真实现 col_outer；缺 stamp fail closed |

## 6. 机器测试冻结

本轮没有复制已有的大量 MLIR fixture，而是把真正承重的现有测试列为机器契约，
并新增一个 matrix gate：

- 五类 dequant 各自已有至少两个字段 mutation 或 missing-field negative；
- A2 unit test 直接覆盖 Nibble honest-null、LMUL g/c/ω 决定性、illegal winner
  no-flip、missing capability 与 empty legal set reject；
- A3 unit/integration test 覆盖 Codebook canonical g、bounded candidate legality、
  VLEN64→m2、VLEN128→m1、VLEN256+fractional→mf2、base-V absent allow-list、direct/registry
  同路、preselection 无 selected stamp、partial/forged/capability failure；
- 四个现役 VLEN128 Codebook golden fixture 保持 byte-exact，decisive fixture 检查
  实际 `i8m2→i32m8/f32m8`、`i8m1→i32m4/f32m4` 与
  `i8mf2→i32m2/f32m2` intrinsic chain；
- LMUL 另有 VLEN/provider 判断、measured/default 分叉和 missing-stamp fail-close；
- SP4 有 measured/prior/shape-isolation 与生产零 static-order；
- loop-order 有 ternary 全格 stamp guard，也有当前 selected→realized override 的
  反例 characterization；
- missing capability 由 stage-B selection 的默认空 march/VLEN0 路径钉住。

`check-formula-authority-matrix.py --self-test` 在内存中分别删除一个 decision、
篡改一个 source census、断开一个 task binding；三种 mutation 都必须被门抓住。
因此该矩阵不是叙事表：后续删旧 symbol 而不更新对应 contract 会立即转红。

## 7. A1 之后的正确顺序

1. A2 已建立最小 typed decision contract，并完整迁入 Nibble 与 LMUL 两个 slice；
   同 slice 旧入口、旧 selector 和独立盖章点已删除。
2. A3 已让 codebook anchor 成为首个完整 `f(g,c)` selected-plan vertical slice，
   不强迫 grid/ternary 的 honest-null 轴伪装成 c-driven。
3. A4 原子关闭其余 selected-stamp、legality 与 emitter redecision 缺口；优先处理
   ISSUE-125，因为它已经存在 `selected != realized`。
4. A5 用 B1 生成的 qualified winner view 取代手工表；measurement 只能从合法
   候选中选，不能创造 mechanism/candidate。
5. A8 再做 baked-g 全量收敛；结构常量按 [K-10] 保留，真实 g 迁入 typed owner，
   dead/wrong 路径直接退役。

每一步都按 RET-1：可以在独立 worktree 分片施工，但 declared slice 只有在全部
production caller 切换、旧入口与默认路径删除、mutation tests 转绿后才能合入。
