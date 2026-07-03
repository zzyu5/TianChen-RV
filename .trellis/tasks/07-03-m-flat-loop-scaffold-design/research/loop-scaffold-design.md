# Research: M-FLAT 循环 scaffold 载重架构设计 + 对抗性可行性判决

- **Query**: 为 flat 家族翻 constructed-strong 设计 typed nb 循环 body scaffold（根因缺口 = per-block-source），并给对抗性可行性判决
- **Scope**: internal（code-anchored 设计蓝图，只研究不改代码）
- **Date**: 2026-07-03
- **base**: refactor/full-refactor-m1

---

## §0 判决（读我）

**判决 = `tractable-多会话`（无架构死墙）。** 估：**~5–7 会话** 到 q8_0 首个 byte-exact `constructed` 翻转（additive，六态 q8_0 cell constructed-weak→constructed = 真 ΔC_construct 强义进展）；**+3–5 会话** 才到 `Δ手写LOC<0`（需一起翻 emitFlatBlockDot 共享的 5 个 flat-plain 格 + 删 monolith）。逐 gap 全 code-anchored 可表达（region 携带 op、verifier 放宽、scalar-extract 桥、递归 validator 都是 MLIR 标准手段 + 有仓内先例）。

**最硬的单个子问题**：**一个携带 region 的 typed 块循环 op，其 SSA loop-carried f32 累加器要 byte-exact lower 到 `emitFlatBlockDot` 的可变 `float sumf` 形（emitc.for 无 iter_args）**——这是仓内第一个"循环进 CORE body 拓扑"（今天所有循环只活在 emitc 层，`with_vl` 是唯一非循环 region 先例），它 subsume 了 ①per-block-source 与 extract 桥（二者只在它的 region 内存在），且它的 byte-exact lowering 是最险的正确性面。**根因（要先设计的）= ①per-block-source（brick① verifier 放宽）；最险的（最易膨胀/藏墙的）= 这个循环 op + 它的 byte-exact lowering。二者是两个不同角色，别合并。**

**阻断 B（extract 桥）确认结果**：**方言里确无 typed 标量-i32-extract op**（grep 全 ODS/dialect cpp：无 `def *Extract*/Vmv*/*ToScalar*` op，`vmv_x_s` 只以 opaque call 出现于 emitFlatBlockDot:5653）。zero-opaque + byte-exact 双约束下**必须新建**此桥（见 §8）。

**为什么这次逃出前 2 次 STOP**：前 2 次的单块/合成路**翻不动任何 flat 家族 cell**（flat 需循环）。本循环路把 q8_0 的 dispatch 换到全-typed 循环 body → **直接翻 q8_0 这张 flat-family cell**（不是加第 4 条合成行）。这是 tractable 判决的结构性依据。

---

## §1 已确立的架构事实（code-anchored，设计地基）

### 1.1 anchor：弱体 = 一个 monolithic op，循环只活在它的 emitc lowering

- 翻强目标 op = `GgmlBlockDot*Q80Op` 家族（`RVVOps.td:3980+`）。它把整块 dot 当**一个** typed op，nested 在 `tcrv_rvv.with_vl` 下、吃 4 个 runtime-ABI 操作数（weight base / activation base / out / n）+ vl token，产 **一个 i32 LMUL m1 向量 token**，其 lowering 把 f32 结果写出（`RVVOps.td:3986-3991`）。
- 它的 emission 明写为 op 的 lowering：**"an outer emitc.for block loop over nb = n / QK"** + **"two scalar fp16->fp32 reads via emitc.call_opaque (the one sanctioned opaque piece)"** + 内 strip loop + `sumf = sumf + (float)sumi*...`（`RVVOps.td:3998-4006`）。
- 实体 lowering = `VariantToEmitCFunc::emitFlatBlockDot`（`RVVToEmitCBlockQuantLinear.cpp:5334`），**5 格共享**（q8_0/q4_0/q4_1/q5_0/q5_1；`wiring-cost-map.md §二`）。
- ⇒ **弱→强 = 把这个 op 的 opaque monolithic lowering 换成一个由 typed 模式库原语构造的 body（含 nb 循环）。**"one sanctioned opaque piece"（fp16 读）**也必须**变 typed（否则 [L-8] 判弱），所以 brick① per-block-source 是强义硬需，不是可选。

### 1.2 循环今天在哪：只在 emitc，不在 CORE typed IR

- `emitFlatBlockDot` 的**外 nb 块循环** = `emitc::ForOp`（`:5881` mbf==1 / `:5901` unroll main / `:5918` tail）。**内 strip 循环** = `emitc::ForOp`（`:5711`，robust 分支）。
- **方言里无任何 typed 循环 op**：grep `def *(Loop|For|Iterate)*Op` in RVVOps.td → 空。唯一 region 携带 op = `WithVLOp`（`RVVOps.td:283-334`，`region SizedRegion<1> $body`，`SingleBlock`/`NoTerminator`）、`VSetVLRegionMarkerOp`、`GearboxCrossRegionHandoffOp`（后者是 2-region product/dequant 桥，**非循环**）。
- ⇒ **CORE typed body 里没有循环拓扑先例。** 块循环要进 CORE（好让 validator 递归、好 byte-exact lower）**必须新建一个 region 携带的循环 op**——这是本设计最大的新结构件。
- **`emitc.for 无 iter_args`**（仓内已实测记录：`RVVToEmitCForwardElementwise.cpp:151/607/938`、`RVVToEmitCInternal.h:3214/3303`："emitc.for has no iter_args, so the loop-carried X is an emitc.variable lvalue + emitc.assign"）。`emitFlatBlockDot` 的 `sumf` 就是 `emitc::VariableOp` + `AssignOp`（`:5412-5417`, `:5829`），**不是** SSA loop-carried。⇒ **SSA-acc ↔ emitc-可变变量 阻抗失配**是设计核心难点（见 §3.2）。

### 1.3 pre-realized body 管线（强路的既有骨架）

单块强路（`dequantize` source front-door）的形态：
- **flat pre-realized body op**（无 region、纯 attrs + ABI 操作数）：`TypedWideningProductReduceDequantizePreRealizedBodyOp`（`RVVOps.td:1571-1626`），描述明写 "RVV plugin must realize it into explicit setvl/with_vl/load/load/widening_product/standalone_reduce/dequantize/store"（`:1588-1591`）。
- **realizer**（realization owner）：`RVVContractionSelectedBodyRealizationOwner.cpp` 用 `createRealizedWithVL`（`:332`）把 flat op 展成 `with_vl { ... }` 单块 region。
- **validator**：`validatePreRealizedRVVSelectedWideningProductReduceDequantizeBody`（`...PreRealizedValidators.cpp:937`）只校 **flat op 的 attrs**（op_kind/memory_form/accumulator_role/…，`:953-993`）+ `rejectMixedPreRealizedContractionBody<SetVLOp,WithVLOp,LoadOp,WideningProductOp,StandaloneReduceOp,StoreOp>`（`:926-932`）。
- **front-door**（auto-compiler 入口）：`RVVDequantDotSourceFrontDoor.cpp`（~1170 行 pass；`createWithVL:468`/`createStandaloneReduce:523`/`createDequantize:548`；`MaterializeRVVDequantDotSourceFrontDoorPass:1026`）。
- **注册**：`RVVConstructionProtocol.cpp` `kRetainedSelectedBodySpecializations[]`（`:357`，`.size()!=67` 门 `:933-934`）+ `RVVContractionRouteIdentity.cpp contractionRouteRegistry()`（`:33`）。

**关键**：单块强路**全程向量**（`standalone_reduce` 产 "one typed vector token whose lane 0 is the scalar output boundary" `RVVOps.td:3403` → `dequantize` 向量 fold → store lane0），**从不落标量、无 scalar extract**——这是它 zero-opaque 的原因。flat 循环路是**标量-per-block fold**（brick②③ 硬要标量 i32 / 标量 f32），**必须**落标量 → 逼出 extract 桥（§8）。

### 1.4 三砖 verifier 现状（`RVVDialectWideningOps.cpp`）

| 砖 | verifier | 硬约束 | ODS |
|---|---|---|---|
| ① `BlockFp16ScaleProductOp` | `:9731-9774` | 2 操作数**必须 `RuntimeABIValueType` 且 role∈{LHSInputBuffer}/{RHSInputBuffer}**（`:9750-9765`）= 导入 ABI 基址；产 f32 | `:9218-9265`；操作数 `AnyType`（`:9252-9260`）+ optional `lhs/rhs_scale_byte_offset`（`:9257-9258`）|
| ② `BlockComputedScaleDequantOp` | `:9776-9822` | `sumi.isInteger(32)` 标量（`:9798`）+ `computed_scale.isF32()` 标量（`:9808`）；产 f32 | `:9267-9313`；操作数 `AnyType`（`:9302-9307`）|
| ③ `CrossBlockF32AccumulateOp` | `:9824-9871` | acc/term/result 全 `isF32()` 标量（`:9853-9868`）；`strict-ascending-block-carried` | `:9315-9366`；操作数 `AnyType`（`:9355-9360`）|

**三砖操作数全 `AnyType`**（约束 100% 在 C++ verifier）⇒ **放宽 ①的 source 契约是纯 verifier 改动，零 ODS 类型改动**（可加 attr/operand）。

### 1.5 [L-8]/[K-4] 强义定义 = **body 属性**，不是删除属性

- `core-invariants.md:67-68`：`constructed-weak` = 算术主体是**被选择的手写 helper**；`constructed`（强）= **body 由模式库原语构造**。
- 判定 = provenance 清单机检（`:56`）：**清单存在（发射模式原语 ID 列表）∧ 无不透明手写 helper ⇒ 强义**；`emission-runtime-contract.md:344` = **conjunction**。
- 六态 schema `coverage-sixstate.v1.json` 现为 **HAND-LABELED**，auto_readout=pending-E5（`note:16`：`'constructed'==STRONG`，C_construct 只数 constructed+）。
- ⇒ **把 q8_0 dispatch 换到全-typed 循环 body（发原语-ID 清单、无 opaque piece）即翻 q8_0→`constructed`；不需删 emitFlatBlockDot。删 monolith = 独立的 `Δ手写LOC<0` 燃减第二轴。** 这是 §0 会话数可分离的 spec 依据。

---

## §2 产出物① — per-block-source 原语契约（**根因**，重点）

**问题**：brick① 现在只读**导入 ABI 基址**（`:9750-9765`），读不了循环里每块的 `base + ib*stride`（每块 d_x/d_y 头）。`emitFlatBlockDot` 里这靠 `blockBaseValue`（`:5451-5462`，`base + (ib+off)*stride` 用 emitc.add/mul）+ `fp16ReadAt`（`:5467-5476`，opaque `(float)*(const _Float16*)`）。

**推荐形态 = 扩 brick①（放宽 verifier + 加循环感知操作数），不新建兄弟 op。** 理由：scale_model / `f32(dx)*f32(dy)` 产品逻辑 100% 复用；类型全 `AnyType`（改动限 verifier + attr）；语义连续（同一"per-block dual-fp16 scale"表面）。

**扩后契约（建议签名，供实现，勿臆断已存在）**：
- 操作数：`lhs_scale_base`, `rhs_scale_base`（仍导入 `RuntimeABIValueType`，role LHS/RHSInputBuffer——**不放宽这点**，基址仍是导入 ABI）+ **新增 `block_index`**（循环 op 的归纳变量 SSA，index/size 类型）。
- attrs：复用 `lhs/rhs_scale_byte_offset`（块内 fp16 头偏移）+ **新增 `lhs/rhs_block_stride`（I64）**（AoS 块步长，如 q8_0=34）。
- verifier 放宽：不再要求 scale base 之外无循环派生量；改为要求 `block_index` 来自 enclosing 块循环 op 的归纳变量（结构校验：`block_index` 的 defining op / block-arg 属于 §3 的循环 op region）。
- lowering：发 `base + block_index*stride (+ byte_offset)` 的 emitc.add/mul + `(float)*(const _Float16*)` 读——**byte-exact 复刻** `blockBaseValue`+`fp16ReadAt`（`:5451-5476`），产 f32 = `f32(dx)*f32(dy)`。

**为何"扩"而非"新 op"更诚实**：新兄弟 op 会复制 scale_model/fold-order/f32-cover 全部注释与 verifier，只为把 role 校验从"导入基址"换成"导入基址 + 归纳偏移"——是同一表面的循环变体。**扩** = 承认单块 brick① 是这条循环 op 的退化（block_index 缺省/无循环）特例。

**根因闭合**：有了 `block_index` 操作数（来自 §3 循环 op），每块 d_x/d_y 可 typed 表达 ⇒ typed nb 循环 body 的 scale 端不再是 opaque。这正是 `wiring-cost-map.md §四` 点名的"要 fund 或绕开的确切设计缺口"。

---

## §3 产出物② — nb 循环 body op 结构

### 3.1 循环 op 形态：**region-based 新 op**（推荐），非 scf.for 包裹

- **推荐**：新 `TypedFlatBlockDotLoopBodyOp`（暂名），`region SizedRegion<1> $body`（仿 `WithVLOp:330`，但**这是循环** region：带归纳变量 block-arg + nb 上界操作数）。lowering imperatively 建 `emitc.for`（复刻 `:5881`），byte-exact by construction。
- **不推荐 scf.for**：RVV lowering 全树**零 scf::ForOp**（grep `lib/Plugin/RVV` `lib/Conversion/RVV` 无 scf::ForOp 使用；全是手建 emitc.for）。走 scf.for 需接 SCFToEmitC，byte-exact 控制力弱于手建 emitc.for。**记 scf.for 为备选但不推荐。**
- op 结构建议：操作数 = weight/activation/out base（导入 ABI）+ `n`/AVL + init acc（f32 0.0）；region 参数 = `block_index`（归纳变量）+ loop-carried `acc`（f32）；attrs = QK、weight/activation stride、coreLmul、multiBlockFactor、stripElision、foldModel（byte-exact 调度旋钮，同 `deriveBlockDotFacts` 读的那套 `RVVToEmitCSupport.h:388-393`）。

### 3.2 loop-carried f32 acc（**最硬**）

- CORE 语义：block_index 每步，`acc_next = brick③(acc, term)`，region 终结 yield `acc_next`；循环产最终 acc → store。这是 **SSA loop-carried**。
- emitc 目标：`float sumf=0; for(...){ ...; sumf = sumf + term; } *s = sumf;`（`:5412-5417`,`:5829`,`:5931-5945`）——**可变变量**，无 iter_args。
- ⇒ lowering 必须把 region 的 SSA loop-carried acc **映射到 emitc.variable + AssignOp**（上游 SCFToEmitC 对 scf.for iter_args 正是这么做；仓内 F3/forward-elementwise 已有"loop-carried 用 emitc.variable"先例 `RVVToEmitCForwardElementwise.cpp:938`）。**先例足 ⇒ 非墙，但是载重设计件**。

### 3.3 body primitive-ID 链（零 opaque，deliverable ② 的清单）

q8_0（PlainI8 / SumiTimesScales / 无 min / 无 qh）每块 region body 应为（对照 `emitBlockCore:5843` / `emitIntegerCore:5663` / `emitFold:5750`）：

1. **per-block-source ①′**（§2）：`block_fp16_scale_product(lhs_base, rhs_base, block_index) -> f32 scale`（复刻 `blockBaseValue+fp16ReadAt:5451-5476` + `dX*dY`）。
2. **load** ×2：`setvl` + `with_vl` + `load` i8 weight/activation（复刻 `loadI8:5521` @ elided vsetvl `:5685`）。
3. **widening_product**：`WideningProductOp` i8×i8→i16（复刻 PlainI8 `vwmul:5538-5542`）。
4. **standalone_reduce**：`StandaloneReduceOp` `signed_widening_reduce_add` i16→i32 lane0 向量（复刻 `vwredsum:5648-5652`）。
5. **scalar-extract 桥**（§8，**新原语**）：i32m1 lane0 → 标量 i32 `sumi`（复刻 opaque `vmv_x_s:5653-5655`）。
6. **brick② `block_computed_scale_dequant`**：`(sumi, scale) -> f32 term`（复刻 `(float)sumi*scaleProduct` `:5768-5774`）。
7. **brick③ `cross_block_f32_accumulate`**：`(acc, term) -> f32 acc_next`（复刻 `sumf+blockTerm:5775-5776`），yield。

**byte-exact 目标锚定**：把 emit-consistency lit 钉在**stripElision="elided"、multiBlockFactor=1、integer_core_lmul="m2" 的 q8_0 实例**（elided 核 `:5681-5700` = 一次 `vsetvl(blockLen)` + 一次 strip reduce，**无内 strip 循环**）。此形 = "一 region = 一块、无内层循环"，最干净。**robust（默认 `:392` "robust"，含内 strip emitc.for `:5711`）与 unroll（mbf>1 `:5901-5928`）留作后续 added-session 形**（robust 若复用 `standalone_reduce` 的内 strip lowering，需另证 byte-exact，非本切片）。

---

## §4 产出物③ — loop-aware **allowlist** validator

- **现状**：`rejectMixedPreRealizedContractionBody`（`RVVEmitCContractionRouteFamilyInternal.h:527-541`）走 `variant.getBody().front()`、模板 blocklist（撞已实现类型即拒）。**它不递归任何 region**（grep validators cpp 无 `.walk(`/`getRegions()`/region 递归——**零区域递归**）。它是"确保 variant 里只有那个 flat pre-realized op、没混入半实现体"。
- **反写风险**（cost-map 警告）：若把砖①②③ 塞进 blocklist 模板，会**拒**含砖的 body（方向反）。**正解 = 新 allowlist 递归 walk**：进 §3 循环 op 的 region，逐 op 断言 ∈ allowlist = {①′per-block-source, WideningProductOp, StandaloneReduceOp, scalar-extract桥, brick②, brick③, SetVLOp, WithVLOp, LoadOp, StoreOp, 循环 op 本身}；region 外仍守 rejectMixed 单块路。
- **挂哪**：contraction realization-owner 边界（`RVVContractionSelectedBodyRealizationOwner.cpp`，紧邻 `createRealizedWithVL:332` 的 validate 调用序），仿 `...PreRealizedValidators.cpp:937` 但**新增递归**。**无既有 call site 做 region 递归 ⇒ 这是新 walk（或新 post-realization pass），不是"扩既有 call site"**——诚实说清。
- **零回归**：新 validator 只吃新循环 op_kind；既有 3 单块强路仍走原 `rejectMixed`（未改动）⇒ 结构性零回归。

---

## §5 产出物④ — wire-in 替换 `emitFlatBlockDot`

- **route/protocol 注册**（仿 cost-map §二墙表，全工作量墙）：`kRetainedSelectedBodySpecializations[]` append 1 条 + bump `.size()` 门 `67→68`（`RVVConstructionProtocol.cpp:357`,`:933-934`）；`contractionRouteRegistry()` 加 1 条 ContractionRouteIdentity（`RVVContractionRouteIdentity.cpp:33`）；op-kind 枚举 + realizer + front-door（新 `RVV*BlockDotLoopSourceFrontDoor.cpp`，仿 `RVVDequantDotSourceFrontDoor.cpp` ~1170 行 additive）。
- **翻 q8_0 cell**：把 q8_0 的 `GgmlBlockDot*Q80Op` dispatch 从 `emitFlatBlockDot`（monolithic opaque）改到新循环 route → q8_0 body = §3 typed 原语链 + 原语-ID 清单 → 六态 q8_0 constructed-weak→**constructed**（§1.5 = 真 ΔC_construct 强义）。**此步 additive、不需删弱体。**
- **emit-consistency lit**：新 route lower 到与 `emitFlatBlockDot` q8_0(elided) 路**字节一致**的 emitc（emitc.for + 每块 typed 原语）。lit 锁 CORE==emission-plans（非数值，pending-hardware 不涉）。
- **`Δ手写LOC<0`（第二轴，deliverable ④ 的删除面）**：emitFlatBlockDot **5 格共享**（`:5334`；PlainI8/OffsetBinaryNibble/UnsignedNibble/FiveBitOffsetBinary + 4 fold tree `:5765-5824`）。**删不了 q8_0 单格而不动共享结构** ⇒ `Δ手写LOC<0` 要求**一起翻 q8_0+q4_0+q4_1+q5_0+q5_1** 再删整函数。这是 §0 里 +3–5 会话的来源，也是**最大 mis-scope 陷阱**（见 §7）。

---

## §6 产出物⑤ — 砖①②③ 复用/扩展/弃用（诚实）

| 砖 | 判定 | 依据 |
|---|---|---|
| **③ `CrossBlockF32AccumulateOp`** | **原样复用（drop-in）** | verifier 已是 `acc+term->f32` 标量 `strict-ascending`（`:9824-9871`），正是循环 loop-carried acc 更新。零改。|
| **② `BlockComputedScaleDequantOp`** | **原样复用，但仅 q8_0 系 fold 树 byte-exact，且被 extract 桥 gated** | verifier 已对（标量 i32 sumi + f32 scale，`:9798`/`:9808`）。喂它标量 sumi 需**新 extract 桥**（§8）；scale 来自 ①′。**只 byte-exact 于 SumiTimesScales（q8_0/q5_0-style，`(float)sumi*(dx*dy)`）**；q4_0 LeftAssoc(`:5779`)、q4_1/q5_1 ScalePlusMin(含 min term,`:5806`) fold 树**不匹配** brick② 的单积形 ⇒ 那些格需额外 fold 覆盖（非本切片）。|
| **① `BlockFp16ScaleProductOp`** | **必须扩契约（非原样复用）** | 当前只读导入基址（`:9750-9765`），**读不了 `base+ib*stride`** = 单块 route STOP 的**根因**。scale_model/`dx*dy`/f32-cover 复用；source 契约重做（§2）。|

**诚实率**：③=100% 复用；②=复用但被 §8 桥与 §2 ① gated、仅 q8_0-fold；①=核心逻辑复用、source 契约重建。**没有一砖白建**（fold 代数 + 标量域 typing 全保留），但只有 ③ 是即插，②要喂，①要改——116 行 brick 投资是**部分载重（fold 代数/标量 typing 留）+ 部分被 supersede（①导入-基址 source 契约被循环 source 替换）**。

---

## §7 产出物⑥ — 对抗性可行性判决（反向压测）

**判决：`tractable-多会话`，无架构死墙。** 逐 gap 全可表达：region 循环 op（`with_vl` 有 region 先例）、① verifier 放宽（纯 C++）、scalar-extract 桥（小新 op）、递归 allowlist（标准 walk）、SSA-acc→emitc-var（SCFToEmitC/F3 先例）。综合的诚实难因 = **这引入仓内第一个"循环进 CORE body"拓扑**，是 op+verifier+realizer+validator+lowering+front-door 的**协调建造**，非增量加法——所以是"多会话-非一砖"。

**反向压测（"为什么这也会 STOP"）——找到的软风险（皆非墙）：**
1. **mis-scope 到 q8_0-单格再撞 additive-only（最大风险，同前 2 次的病）**：若把 win 绑死在 `Δ手写LOC<0`，q8_0 单格删不动共享 monolith（§5）→ 又是纯加法。**缓解 = 分离两轴**：constructed 翻转（§1.5 body 属性，re-dispatch 即成，真 ΔC_construct）作首交付；`Δ手写LOC<0` 明标为后续 cohort-flip 砖。**此分离有 spec 依据（[L-8]/schema），不是找借口。**
2. **byte-exact 的多形爆炸**（robust 内 strip loop + mbf>1 unroll + elided，`:5681-5928`）：缓解 = lit 钉死 elided/mbf=1/m2 单实例（§3.3），其余形 deferred。
3. **robust 内 strip 复用 `standalone_reduce` 是否 byte-exact 于 emitFlatBlockDot 的 blockLen-界内 strip**：未证；避开 = 走 elided（无内 strip）。
4. **SSA-acc→emitc-var 阻抗**（§3.2）：有 SCFToEmitC/F3 先例，非墙但载重。

**未找到任何架构级不可表达**（MLIR region/walk/verifier 全支持；`with_vl` 已证 region body 可 realize+validate+lower）。⇒ 判 `tractable`，不判真墙。**若 parent 坚持单砖必须 `Δ手写LOC<0`，则该"砖"实为 5-格 cohort + monolith 删除的多会话工程，应按里程碑级排期，不按单砖。**

---

## §8 阻断 B — scalar-i32-extract 桥（确认 + 设计）

**确认**：grep 全 ODS（`def *Op : TCRVRVV_Op<"...(extract|scalar|lane|vmv_x|to_scalar)..."`）+ dialect cpp（`vmv_x_s`/`VectorToScalar`/`scalar_extract`）→ **无 typed 标量-extract op**。`emitFlatBlockDot` 的 `sumi` 由 opaque `__riscv_vmv_x_s_i32m1_i32`（`:5653-5655`）从 vwredsum 结果取出。单块强路**不需**它（全程向量 lane0，§1.3），故此前无人建。

**为何循环路必须它**：brick② 硬要**标量** i32 sumi（`:9798`）；byte-exact 目标 `(float)sumi` 是**标量** cast（`:5768`）。zero-opaque + byte-exact 双约束 ⇒ vwredsum(i32m1 lane0) → 标量 i32 的这一跳**必须 typed**。

**推荐**：新 `TypedVectorLane0ToScalarExtractOp`（暂名），操作数 = i32m1 向量 + vl，attr = kind（`lane0-i32-extract`），产标量 i32；verifier 校输入 vector/结果 scalar i32；lowering = opaque `vmv_x_s`（byte-exact 复刻 `:5653-5655`）。**这是最载重的单个新原语**：无先例 op、无"扩既有 verifier"捷径（不像 ① 是放宽）——它是整条"整数核→标量 fold"handoff 从 opaque 变 typed 的枢纽。

---

## Caveats / Not Found

- **未逐字读** front-door / realizer / route-identity 的全部 ~1170+~1000 行内部（只读结构锚点）；~1000 行 additive front-door 的具体 body-builder 未逐行核（cost-map §二已实测其量级，本设计承接）。
- **未确认** `stripElision` 缺省 "robust" 下内 strip 循环复用 `standalone_reduce` 是否 byte-exact——**推荐避开（走 elided）**，未证可复用。
- **provenance 清单机检 = pending-E5**（schema `note:7`）；当前六态 HAND-LABELED。本设计确保强 body **能**发原语-ID 链（§3.3），但机检管线本身非本任务。
- **数值 bit-exact-vs-ggml = pending-hardware**，不涉（红线）。
- **未造 n/N 计数**；进度语言 = ΔC_construct（强义）+ Δ手写LOC（红线遵守）。
- 会话数为工程量估（gap 拆解 §0/§7），非承诺；实现可能因 byte-exact 多形折叠而波动。
