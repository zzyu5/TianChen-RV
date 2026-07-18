# Research: A线 律2「点不入源」HEAD-准确 census（宽谓词穷尽）

- **Query**: 在当前 HEAD 跑准确 census（非 pin `0aee07b4`），逐处宽谓词，对 RVV 低精度 EmitC 家族逐常数
- **Scope**: internal（grep/read only·零改）
- **Date**: 2026-07-19
- **HEAD**: `f457e73371b1f714d7e4886ca8859392613e457f`（branch `refactor/full-refactor-m1`）
- **pin 参照**: `0aee07b48`（paper `03-B-DEBAKE-律2-75处违反.md`，g37/c38）
- **工作树 == HEAD**（`git status --porcelain` 对 lib/Conversion/RVV·lib/Dialect/RVV·lib/Plugin/Scalar·include 全空）

> ⚠ 本文是数据供主会话施工/回门，不是给人看的汇报。**核心订正见 §0**。

---

## §0 三条推翻主会话窄谓词头条的结论（必读）

1. **「coreLmul 焊死 = 11」是错分类**。11 个 `.cpp` 命中里只有 **5 个是真焊死+独立字面量 vtype**（=ISSUE-031(a) 假旋钮形态），另 **5 个是 `coreLmul="m2"; if(getIntegerCoreLmul()) coreLmul=*attr;` 的 live-default-then-override**（= c 轴 value_or 同类·vtype 从 coreLmul 派生·**非**假旋钮），**1 个是 verifier-pinned m1 + 派生 vtype**（CodebookFp4·结构固定）。**朴素 grep 把 default-override 误计为焊死。**
2. **「假旋钮修」这一 PRD 施工类在 HEAD 上是空的**：5 个真焊死全部**已裁 MAINTAIN**（GridCodebook 497/995 + Ternary 862 = ISSUE-031(a)「Context 无 coreLmul 字段·无 IR 宽度·强加字段=造假旋钮=[K-10] 违例」；BQL 111/404 = stage-3 [K-10] 结构常量）。**无一个可 byte-exact 直修的假旋钮·全部 gated ISSUE-031。**
3. **宽谓词发现窄头条遗漏的大面 g 轴残量**：`ForwardElementwise` dequant-row 家族（~40 处·recent B线 owned-emit·**未登记任何 issue**）+ `GridCodebook.cpp`(9) + `TernaryBinary.cpp`(8) 的 grid 几何常量（与阶段2路B 已转 KQuant 的**同一描述符类**·只是未做）。主会话「KQuant g=2」头条只覆盖了 pin-shape 残量的一角。

---

## §1 g 轴 census（宽谓词穷尽）

### §1.1 pin-shape 谓词（`int64_t NAME = <int>;` 排 `.get`/`=0`/`=1`）—— 逐处 HEAD 行号

| 文件 | HEAD 数 | pin 数 | 逐处 file:line（HEAD） |
|---|---|---|---|
| `RVVToEmitCKQuant.cpp` | **2** | 19 | 4607 `weightDOffset=80`，4608 `weightDminOffset=82` |
| `RVVToEmitCGridCodebook.cpp` | **9** | 7 | 76 `subBlockLanes=32`，504 `groupLanes=8`，1002 `groupLanes=8`，1514 `halfLanes=16`，1629 `pairHalves=4`，1986 `halfLanes=16`，2102 `pairHalves=4`，2466 `groupLanes=8`(const)，2467 `numGroups=4`(const) |
| `RVVToEmitCTernaryBinary.cpp` | **8** | 8 | 854 `groupLanes=8`，869 `halvesPerSub=2`，870 `groupsPerHalf=2`，871 `halfLanes=16`，1588 `planeLanes=32`，1589 `planesPerChunk=4`，1590 `chunkBytes=32`，2060 `dotStripLanes=32` |
| `RVVToEmitCForwardElementwise.cpp` | **3** | 1 | 2709 `qk=32`，3575 `qhOff=16`(q5_K only)，4564 `qk=256` |
| `RVVToEmitCInternal.h` | **2** | (边界3) | 3648 `stripWidth=8`，3649 `numStrips=4`（板派生·见 §1.4） |
| `RVVToEmitCBlockQuantLinear.cpp` | (2 排除) | (2 排除) | 394 `kGemmDefaultActivationCols=4`，3259 `kMainTermUnrollCodeVolumeICacheBudget=192` = **调优阈值·非 g/c 轴·pin 亦排除** |
| `RVVToEmitCCodebookFp4.cpp` | 0 | 0 | — |
| `lib/Plugin/Scalar/ScalarBackendEmissionDriver.cpp` | 2 | 2 | 239 `numPlanes=4`，240 `planeLanes=32`（=#10a·**Scalar 非 RVV 家族·仅列参照**） |
| **RVV 家族 pin-shape 合计** | **24** | 35 | （KQuant2+Grid9+Ternary8+FwdEW3+Internal2；不含 Scalar2/BQL 调优2） |

**与 pin 差**：
- **KQuant 19→2**：阶段2路B（commit `85cd5022f`）把 KQuant 的 grid/subblock 常量（numGroups/indicesPerSubBlock/groupLanes/signsPerSubBlock/groupsPerHalf/quarter/numLanes 等 17 处）换成 `coreOp.getXxx()` 描述符读（OptionalAttr `num_groups`/`group_lanes`/`indices_per_sub_block`/`groups_per_sub`/`num_lanes`·前门 stamp·fail-closed）。**仅剩 4607/4608**（=ISSUE-115·下 §2.1）。
- **GridCodebook 7→9**：行号整体位移（529→504·1052→1002·1566→1514·1706→1629·2063→1986·2181→2102）+ 净 **+2**（2466/2467 = pin 后新增 helper 里的 `const int64_t`）。**阶段2 未触碰 GridCodebook.cpp**（不在 `85cd5022f --stat`）。
- **Ternary 8→8**：不变（行号 856→854·871→869·1592→1588·2064→2060 位移）。**阶段2 未触碰。**
- **ForwardElementwise 1→3**：文件今日（07-19 04:16）大改·新增 q5_K `qhOff=16`(3575) 与 q6/IQ `qk=256`(4564)。此外**宽谓词另发现大量 switch/多变量形态**（见 §1.2）。

### §1.2 宽谓词补：多变量逗号声明 + per-format switch（pin-shape 谓词漏·尾非 `;`/switch-case）

**全在 `RVVToEmitCForwardElementwise.cpp` 的 dequant-row 发射家族**（op = `weftrvv::GgmlDequantizeRowOp`·`emitGgmlDequantizeRowExtended` 是**「monolith fallback·keyed by format 字符串」**·每格布局烘焙进发射体）。谓词 `(qk|stride|qsOff|dOff|qhOff|blockStride|quantOffset)=<int>` 排 `.get`/注释：

| file:line | 常量（enclosing 函数） |
|---|---|
| 1133 | `qk=32, blockStride=34, scaleOffset=0, quantOffset=2`（emitDequantizeRowNibble…） |
| 1372-1373 | `qk=32, blockStride=36, scaleOffset=0, sumOffset=2, quantOffset=4` |
| 1614 | `qk=256, blockStride=292, scaleOffset=0, quantOffset=4, …` |
| 2658-2670 | `emitGgmlDequantizeRow`：per-format `stride/dOff/mOff/qhOff/qsOff/sub`（iq1_s 18/iq1_m 20/… 5 分支×多常量） |
| 2975 | `qk=32, half=qk/2`（half 派生·qk 烘焙） |
| 3259, 3394 | `qk=32, stride=34, qsOff=2`（emitDequantizeRowQ8_0BodyShared / VectorBody） |
| 3574 | `qk=256, stride=isQ5?176:144, qsOff=isQ5?48:16`（Q45K·三元非 switch） |
| 3743 | `qk=256, stride=84`（Q2K VectorBody） |
| 3870 | `qk=256, stride=110`（Q3K VectorBody） |
| 4048 | `qk=256, stride=210`（Q6K VectorBody） |
| 4564 | `qk=256`（Ternary VectorBody） |
| 5331-5351 | `emitGgmlDequantizeRowExtended`：**per-format stride switch 19 case**（Q2K 84/Q3K 110/Q4K 144/Q5K 176/Q6K 210/MXFP4 qk32 str17/NVFP4 qk64 str36/TQ1 54/TQ2 66/Q10 qk128 str18/IQ4NL qk32 str18/IQ2XXS 66/IQ2XS 74/IQ2S 82/IQ3XXS 98/IQ3S 110/IQ1S 50/IQ1M 56/IQ4XS 136）——直喂指针算术 `xb = x + ib*stride` |
| 5392-5395 | `dOff=0, qsOff=2`（MXFP4 `qsOff=1`） |

**性质**：全是 ggml AoS 块字节布局（stride=块字节·qsOff/dOff=区偏移）**按 format 枚举/字符串烘焙进发射体** = 律2 g 轴命中·**宽谓词铁证**。**pin ForwardElementwise g 轴只数了 1（2709）**；HEAD 真实面 ≈ **40+ 处**（19 switch-case + ~20 多变量/单变量），**主会话窄头条「KQuant g=2」完全漏掉这一大面**。架构说明见 §2.4。

### §1.3 排除的派生式（不计 g 轴·谓词自动排/宽判据放行）

- `numSubBlocks = qk/subBlock`（KQuant 4606）、`halfBlock = qk/2`（BQL 100/386）、`half = qk/2`（FwdEW 2975）、`ib32 = 2*pair+half`、`shift = 2*shiftIdx` 等——除数字面量·值来自 `qk` 参数孔 → pin 归**边界**（qk/N 派生 33 处）。
- GridCodebook `pairLanes = 2*subBlockLanes` 等派生表达式。

### §1.4 Internal.h 板派生（边界·非纯 g 轴）

- `RVVToEmitCInternal.h:3648 stripWidth=8`、`3649 numStrips=4`、`3650 foldGroups=1`（若在则）——注释自述「derived from l8 (mf2→8/m1→16/m2→32)」。**当 l8 成为真 knob（ISSUE-113 升级）时这些应从 l8 派生**；现固定在 mf2 默认。归 c 轴板派生边界。

---

## §2 g 轴逐处裁决状态映射（cross-ref issues）

### §2.1 KQuant 4607/4608（weightDOffset=80 / weightDminOffset=82）→ **ISSUE-115（待裁·blocked）**

- enclosing = `emitTypedSuperBlockScalarScaleMinLoopBody`·coreOp = `GgmlBlockDotQ2KQ8KIntegerCoreOp`。
- 教科书并排铁证（4598-4609）：同 emit 体内 `scalesOffset=coreOp.getWeightScalesByteOffset()`、`qsOffset=coreOp.getWeightQsByteOffset()`、`q8Offset=coreOp.getActivationQuantByteOffset()`、`bsumsOffset=coreOp.getActivationBsumsByteOffset()`、`weightStride=loopBody.getWeightBlockStride()` **全参数孔**；唯 `weightDOffset=80`/`weightDminOffset=82`/`activationDOffset=0` 焊死（注释：「FIXED block_q2_K constants·the scalar fold has no separate fold brick」）。
- **裁决（ISSUE-115）**：**不能走路 B「同法加 attr」**——q2_K 整数核 op 的 ODS charter 明标 fp32 SCALAR fold（`dall=fp16(x.d@80)*y.d`）**DELIBERATELY OUT OF SCOPE**·加 `weight_d/dmin_byte_offset` attr = 把故意排除的 fold 拉进整数核 op = 语义冲突。**修法 = 专用 fold-brick op（新 op·非 attr）·待用户/supervisor 裁。** 保守默认 = 两字面量留 baked（byte-exact 不动）。公式占比封顶 35/37。
- `activationDOffset=0` 排（=0）。

### §2.2 GridCodebook.cpp 9 处 + TernaryBinary.cpp 8 处 grid 几何 → **路B 描述符类·未登记·扇出候选**

- 这 17 处（subBlockLanes/groupLanes/halfLanes/pairHalves/numGroups/halvesPerSub/groupsPerHalf/planeLanes/planesPerChunk/chunkBytes/dotStripLanes）与阶段2路B **已转 KQuant 的同一「FORMAT-DEFINED first-class fact」描述符类**（阶段2 comments 原话：「NOT a subBlock/8 derivation」·做成 OptionalAttr `group_lanes`/`num_groups`/… fail-closed）。
- **区别**：GridCodebook/Ternary 的 emit body 是**接收 Context 结构体的 helper**（`IQ3XXSGridBodyContext` 等），非直接持 op → 要读描述符需 Context 结构体从 op attr 填值（call-site）。阶段2 对 KQuant 正是这样做的。
- **状态**：**未登记任何 issue**（阶段2 `--stat` 未含这两文件）。= 真 A线残量·路B 可套用·回门候选（新 OptionalAttr per op）。**注**：ISSUE-031(a) 指 iq3 grid 几何「无 IR 宽度」是**对 c 轴 coreLmul 而言**（宽度轴不适用）；**g 轴 grid-count 是 format-defined 事实**·阶段2 已证可做描述符·两者不同轴别混。

### §2.3 ForwardElementwise pin-shape 3 处（2709/3575/4564 qk/qhOff）→ 直改候选（getQk）

- op = `GgmlDequantizeRowOp` 持 `getFormat()`(string)·`getQk()`（HEAD 用 1×）。`qk=32`/`qk=256` 烘焙**可直改读 `deqOp.getQk()`**（若该 op 在这些路径带 qk attr）。但每格函数 qk 结构固定（emitDequantizeRowQ8_0 恒 qk=32）→ 直改价值 = 去重字面量·**先验证 op 在该 path 带 qk attr**（§4）。`qhOff=16` 是 q5_K 布局偏移 → 描述符类（同 stride）。

### §2.4 ForwardElementwise dequant-row 家族 ~40 处 → **独立架构面·monolith-fallback·未登记**

- `emitGgmlDequantizeRowExtended` 注释自述 = 「**monolith fallback·keyed by the `format` string**」。format→布局（stride/qsOff/dOff/qhOff）映射**内在于该 path**·每格 switch-case 烘焙。
- 与 block-dot path（typed brick op + byte-offset attr）**是两套架构**。转描述符 = 大改（前门须为 `GgmlDequantizeRowOp` stamp 每格全布局·发射体改读）→ **非 byte-exact 直改·需独立 scoping（建议新 issue）**。架构类比 = pin #10a Scalar（format map 在发射体）。
- **这是本次宽谓词 census 相对 pin/主会话头条的最大增量·如实上报。**

---

## §3 c 轴 census（HEAD·逐处 + 分类订正）

### §3.1 `value_or("mX")` 板值自补（5·pin 23）

| file:line | 命中 | 分类 |
|---|---|---|
| `ForwardElementwise.cpp:169` | `mapOp.getStripLmul().value_or("m8")` | c 轴 value_or·strip 轴（非 integer_core）·ISSUE-113 backlog |
| `ForwardElementwise.cpp:521` | `rmsCore.getStripLmul().value_or("m8")` | 同上 |
| `KQuant.cpp:2918` | `scaledDot.getIntegerCoreLmul().value_or("mf2")` | **ISSUE-116 六 live-default 之一**（旧行 2714·front-door stamp 回门·blocked） |
| `KQuant.cpp:3694` | `b3.getIntegerCoreLmul().value_or("mf2")` | **ISSUE-116**（旧行 3485） |
| `KQuant.cpp:5971` | `coreOp.getIntegerCoreLmul().value_or("m2")` | **ISSUE-116**（旧行 5730） |

- **与 pin 差**：BQL 18 处 value_or 已清（阶段0）→ pin 23 → HEAD 5。
- **裁决（ISSUE-113/031(b)）**：value_or 默认在**真测过路径上承重**（BQL 单体 repack op / KQuant q4_K 超块砖 by 触碰集外前门 / 112 fixture）·盲改 fail-closed 破 112 fixture + 真管线。**保守默认 = 维持 value_or**·升 required 待「A2/A3 收口+fixture 冻结」触发窗（ISSUE-113·非「等稳定」）。

### §3.2 `coreLmul = "<lit>"`（12 命中·**订正分为三类**）

谓词 `coreLmul\s*=\s*"m[...]"` 命中 12（含 Internal.h）。**逐处查「后 4 行有无 `if(getIntegerCoreLmul())` override」+「vtype 是派生 `"vint8"+coreLmul` 还是独立字面量」**：

| file:line | enclosing | override? | vtype | **真类** |
|---|---|---|---|---|
| GridCodebook 497 | `emitIQ3XXSSuperBlockGridBody` | NO | 4 字面量 | **真焊死·假旋钮形态 → ISSUE-031(a) MAINTAIN**（Ctx 无 coreLmul 字段） |
| GridCodebook 995 | `emitIQ3SSuperBlockGridBody` | NO | 4 字面量 | **真焊死 → ISSUE-031(a) MAINTAIN** |
| Ternary 862 | `emitIQ1MSuperBlockGridBody` | NO | 6 字面量 | **真焊死 → MAINTAIN**（IQ1MGridBodyContext 无 coreLmul 字段·同 iq3 逻辑·[K-10]） |
| BQL 111 | `emitTypedTileGemvBody`(q4_0) | NO | 3 字面量 | **真焊死 → stage-3 [K-10] MAINTAIN**（注释明标·无 IR 宽度·ISSUE-116 二结构常量之一） |
| BQL 404 | `emitTypedTileGemmBody`(q4_0) | NO | 3 字面量 | **真焊死 → stage-3 [K-10] MAINTAIN**（ISSUE-116 之二） |
| Internal.h 3634 | 共享 helper 板阶梯 | NO | 0（用 l8/l16/l32） | 板阶梯默认·「reproduces byte-identical legacy」·ISSUE-113 相邻 |
| Ternary 1607 | `…GridLoopBodyTQ20` | **OVERRIDE@1608** | **3 派生** | **非焊死** = live-default+override·vtype 派生·**c 轴 value_or 同类**（compliant knob·默认 m2 是 VLEN-safe floor·gearbox 精化） |
| Ternary 2062 | `…GridLoopBodyTQ10` | **OVERRIDE@2063** | 1 派生 | **非焊死**·同上 |
| BQL 14232 | `emitTypedFlatBlockDotLoopBody` | **OVERRIDE@14233** | 派生(shared) | **非焊死** = ISSUE-116 六 live-default（旧行 14221·stage-3「NOT debakeable·LIVE·前门先 stamp」） |
| BQL 14315 | `emitTypedFlatBlockDotLoopBody` | **OVERRIDE@14316** | 派生(shared) | **非焊死** = ISSUE-116（旧行 14300） |
| BQL 17663 | `emitQ1_0Q8_0BlockDot` | **OVERRIDE@17664** | 派生(shared) | **非焊死** = ISSUE-116（旧行 17644） |
| CodebookFp4 109 | `…GridLoopBodyIq4xs` | NO | **3 派生**(deriveWideningChain) | 焊死 m1 但 **verifier-pinned**（「codebook gather REQUIRES m1·VLMAX≥16·verifier pins」）+ vtype 派生 → **结构固定 MAINTAIN**·非假旋钮 |

**订正小结**：主会话「coreLmul 焊死=11」= 5 真焊死(假旋钮形态·全 MAINTAIN) + 5 live-default-override(value_or 同类·ISSUE-116/113) + 1 verifier-pinned(CodebookFp4·MAINTAIN)。**真焊死+可修者 = 0。**

### §3.3 `wideLmul = "<lit>"` 焊死（5·配对）

- BQL 112/405（配 111/404·MAINTAIN）、GridCodebook 498/996（配 497/995·MAINTAIN）、Ternary 863（配 862·MAINTAIN）。**全是真焊死 coreLmul 的配对宽化字面量·随其宿主一并 MAINTAIN。**（Ternary 1610 的 `wideLmul=(coreLmul=="m2")?"m4":"m2"` 是派生·不入此列。）

### §3.4 Internal.h 板阶梯（4·B3）

- 3634 `coreLmul="mf2"`、3635 `l8="mf2"`、3636 `l16="m1"`、3637 `l32="m2"`——共享 emit helper 的「缺省板阶梯」（注释「default reproduces byte-identical legacy」）。随 ISSUE-113 c 轴 required 升级一并处理·保守默认维持。

---

## §4 待验证点（主会话施工前须核·本 census 未展开的不确定项）

1. **ForwardElementwise `GgmlDequantizeRowOp` 是否在各 dequant-row path 带 `qk`/`stride`/byte-offset attr**（决定 2709/4564 qk 能否直改读 getQk·决定 stride switch 能否路B 描述符）。谓词：读 `include/Weft/Dialect/RVV/IR/RVVOps.td` GgmlDequantizeRowOp 定义。
2. **GridCodebook/Ternary grid helper 的 call-site 是否持 op**（决定 grid-count 描述符能否像阶段2 那样填 Context）。
3. **ISSUE-116 行号已位移**（issue 记 2714/3485/5730/14221/14300/17644 = 旧 ref·HEAD = 2918/3694/5971/14232/14315/17663）——回门施工引用须用 HEAD 行号。

---

## Related Specs

- `.trellis/spec/issues/发射器与架构.md` — ISSUE-031（F-7 住址门·假旋钮判据·(a) GridCodebook iq3 结构常量订正·(b) value_or 承重）、ISSUE-113（c 轴 required 升级·收尾锁）、ISSUE-115（weightDOffset/dmin fold-brick op）、ISSUE-116（6 live-default 前门 stamp + 2 结构常量 MAINTAIN）
- `.trellis/spec/issues/性能与测量.md` — ISSUE-117（c 轴 = measured 决策·[GAP-P1] 锁·**非写 widest-legal**·别与本 census 混）
- 合规范本：`lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp`（getLmul fail-closed×21·两轴 0）、`RVVToEmitCBlockQuantLinear.cpp`（qk 派生·getIntegerCoreLmul 旋钮·halfBlock=qk/2）
- 描述符范本：`include/Weft/Dialect/RVV/IR/RVVOps.td`（OptionalAttr `groups_per_sub`/`num_groups`/`group_lanes`/`num_lanes`/`weight_d_byte_offset`/`block_stride`·fail-closed）；前门 stamp = `lib/Conversion/RVV/RVVMonolithicBlockDotSourceFrontDoor.cpp`

## Caveats / Not Found

- **ForwardElementwise dequant-row 家族 ~40 处未逐处穷举到 100%**（switch 19 + 多变量 ~20 已列代表·实际每格函数内还有 dOff/mOff/sub 等小常量）。这是 recent B线 owned-emit 大面·**建议主会话若纳入 A线范围先开独立 census/issue**（架构 = monolith-fallback-keyed-by-format·非 block-dot byte-offset-attr·转描述符是大改非直改）。
- 未编译验证任何改动（本任务 read-only）。所有 byte-exact 断言 = 静态阅读推断·施工须 regen-diff 门（§见 classification 文件）。
