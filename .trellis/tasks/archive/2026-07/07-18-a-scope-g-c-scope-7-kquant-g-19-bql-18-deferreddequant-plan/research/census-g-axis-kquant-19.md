# Research: A 线 g 轴 —— KQuant 格式字面量烘焙 19 逐处表

- **Query**: 定位 `RVVToEmitCKQuant.cpp` 里按格式字面量分支/烘焙的 19 处，逐处给目标路由（格式描述符层）+ 难度 + byte-exact 风险
- **Scope**: internal（代码普查·机算可复跑）
- **Date**: 2026-07-18

## 机算复跑（对齐 03 census · pin `0aee07b4`）

**权威谓词**（裸单字面量，带尾 `;`，排 `.get`/`=0`/`=1`）：
```
git grep -nE 'int64_t[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*[[:space:]]*=[[:space:]]*[0-9]+[[:space:]]*;' 0aee07b4 -- lib/Conversion/RVV/RVVToEmitCKQuant.cpp \
  | grep -vE '\.get' | grep -vE '=[[:space:]]*[01][[:space:]]*;'
```
- 本轮实跑 @ `0aee07b4` = **19**（与 03-B-DEBAKE §③ 逐行、逐行号相同）。
- **HEAD `c84820c4` 复跑 = 19**（scope 未漂；同一 5 文件谓词在 HEAD 与 pin 逐数复现）。

## 关键结构事实（决定路由与难度）

同一 emit 体内，**绝大多数**格式偏移已走 `coreOp.getXxx()` 参数孔（格式描述符层已在）：
`getSubBlock()` · `getQk()` · `getWeightScalesByteOffset()` · `getWeightQsByteOffset()` · `getWeightQhByteOffset()` · `getWeightGasByteOffset()` · `getActivationQuantByteOffset()` · `getActivationBsumsByteOffset()` · `getActivationDByteOffset()` · **`getWeightDByteOffset()`（14 处 ODS 定义）** · **`getWeightDminByteOffset()`（7 处 ODS 定义）** · **`getNumLanes()`（`$num_lanes` 1 处 ODS 定义，used @2695 `fold.getNumLanes()`）**。
这些 `I64Attr` 定义在 `include/Weft/Dialect/RVV/IR/RVVOps.td`（super-block core / loop-body ops），由前门 **decode-facts 描述符结构**（`kIq4XsDecodeFacts` / `KQuantDecodeFacts` / grid decode facts，见 `RVVLowerQuantContraction.cpp`）填充。

⟹ **19 处 = 尚未被抽进 getter/描述符的残量**。分三组（下表 `组` 列）：
- **G-A**：**已有 getter/派生兄弟**在同文件其它 super-block 就地被读，唯此处烘焙 —— 路由 = 换成已有 `coreOp.getXxx()`。**零新 schema 字段**。
- **G-B**：可从**已有参数孔**（`qk`/`subBlock`/`numSubBlocks`/`halfLanes`）**闭式派生**（ggml 超块几何固定）—— 路由 = 派生表达式。**零新 schema 字段**（但派生算术须 byte-exact 复核）。
- **G-C**：**grid/codebook 家族超块子结构常数，0 个已有 ODS 属性** —— 路由 = 前门 decode-facts 描述符新增字段 + 可选 op attr。**这是唯一回门点**（见 debake-staged-plan.md §4）。

## g 轴 19 逐处表

| # | file:line | 字面量 | 值 | 现烘焙形态 | 目标路由（格式描述符） | 组 | 新 schema? | byte-exact 风险 |
|---|---|---|---|---|---|---|---|---|
| 1 | RVVToEmitCKQuant.cpp:1787 | `quarter` | 8 | 裸字面量（8-elem quarters, q6_K super-block subBlock=32） | 派生 `subBlock/4`（另存 `cx.quarter`，可入描述符） | G-B | 否 | 低（32/4=8，同块唯一取值） |
| 2 | :2537 | `numLanes` | 8 | 裸字面量（canonical aux32/sums lanes, q4_K fold） | `getNumLanes()`（getter 已在，@2695 已用）或派生 `numSubBlocks/2` | G-A | 否 | 低-中（须证此 op 载 `num_lanes`；否则派生 @3387 已有先例 `numLanes=numSubBlocks/2`） |
| 3 | :2956 | `quarter` | 8 | 裸字面量（q5_K path, subBlock=32） | 派生 `subBlock/4` | G-B | 否 | 低 |
| 4 | :2957 | `numLanes` | 8 | 裸字面量（aux32/sums lanes） | `getNumLanes()` 或派生 `numSubBlocks/2` | G-A | 否 | 低-中 |
| 5 | :3791 | `weightDOffset` | 80 | **教科书并排**：邻行 scalesOffset/qsOffset/q8Offset/bsumsOffset 全走 `coreOp.getXxx()`，唯此烘焙（fp16 x.d, q3_K/q6_K subBlock=16） | **`coreOp.getWeightDByteOffset()`**（getter 已在，@2535/2937/4614/4859/5104/5332 已用） | G-A | 否 | 低（须证此 op 实例被 stamp=80；同族 no-min 超块 d 偏移 = 描述符事实） |
| 6 | :3792 | `weightDminOffset` | 82 | 同上并排烘焙（fp16 x.dmin） | **`coreOp.getWeightDminByteOffset()`**（getter 已在，@2392/2938 已用） | G-A | 否 | 低（须证 stamp=82） |
| 7 | :4202 | `groupsPerSub` | 4 | 裸字面量（4 grid groups per sub-block, l=0..3；iq-grid） | 新 grid 描述符字段 `groups_per_sub` 或派生 `subBlock/8` | G-C | **是（回门）** | 中 |
| 8 | :4415 | `groupsPerSub` | 4 | 同上（另一 grid super-block emit 体） | 同上 | G-C | **是** | 中 |
| 9 | :4620 | `numGroups` | 4 | 裸字面量（4 sign groups per sub-block） | 新字段 `num_groups` 或派生 `subBlock/8` | G-C | **是** | 中 |
| 10 | :4621 | `indicesPerSubBlock` | 8 | 裸字面量（8 grid index bytes per sub-block, 2 per group） | 新字段 `indices_per_sub_block` 或派生 `subBlock/4` | G-C | **是** | 中 |
| 11 | :4622 | `groupLanes` | 8 | 裸字面量（8 grid lanes per sign group） | 新字段 `group_lanes` 或派生 `subBlock/4` | G-C | **是** | 中 |
| 12 | :4867 | `numGroups` | 4 | 同 #9（另一 emit 体） | 同 #9 | G-C | **是** | 中 |
| 13 | :4868 | `indicesPerSubBlock` | 8 | 同 #10 | 同 #10 | G-C | **是** | 中 |
| 14 | :4869 | `signsPerSubBlock` | 4 | 裸字面量（4 explicit sign bytes per sub-block） | 新字段 `signs_per_sub_block` 或派生 `subBlock/8` | G-C | **是** | 中 |
| 15 | :4870 | `groupLanes` | 8 | 同 #11 | 同 #11 | G-C | **是** | 中 |
| 16 | :5109 | `numGroups` | 4 | 裸字面量（4 grid/sign groups per sub-block；此处邻 `coreLmul=...value_or("m2")` = KQuant c 轴，见 census-c-axis） | 同 #9 | G-C | **是** | 中 |
| 17 | :5338 | `numGroupsPerHalf` | 2 | 裸字面量（2 grid/sign groups per 16-lane half） | 新字段 `num_groups_per_half` 或派生 `halfLanes/groupLanes` (16/8) | G-C | **是** | 中 |
| 18 | :5567 | `groupsPerSub` | 4 | 同 #7 | 同 #7 | G-C | **是** | 中 |
| 19 | :5568 | `numGroupsPerHalf` | 2 | 同 #17 | 同 #17 | G-C | **是** | 中 |

**分组小计**：G-A（换已有 getter，零 schema）= **4**（#2/#4/#5/#6）· G-B（派生，零 schema）= **2**（#1/#3）· G-C（grid/codebook 描述符字段，回门）= **13**（#7–#19）。

**格式家族分层**（决定阶段切分）：
- **平 K-quant（q3_K/q4_K/q5_K/q6_K）残量 6**：#1–#6（1787/2537/2956/2957/3791/3792）。全在 G3 已退役进前门的 K-quant CONSTRUCTED emit 体内；目标全是已有 getter / 派生 → **零新 schema · 低风险**。
- **iq-grid/codebook（iq1_s/iq2_xxs/iq2_xs/iq2_s…）残量 13**：#7–#19（4202–5568）。grid 超块子结构常数，0 现存属性 → **新描述符字段（回门）· 中风险**。

## 排除的派生式（不计 19）

如 `numLanes = numSubBlocks / 2`（@3387）、`numSubBlocks = qk / subBlock`（多处）—— 已是派生表达式，谓词自动排掉。它们**恰是 G-B 目标形态的现成先例**（同文件已有干净派生写法可照抄）。

## Caveats / Not Found

- G-A 的 #5/#6（weightDOffset=80/weightDminOffset=82）风险 = **须确认承载此 emit 体的 op 实例被前门 stamp 了 80/82**；若前门当前对该 no-min 超块未 stamp d/dmin 偏移，则 G-A 退化为 G-C（先补 stamp）。**须 implement 阶段读前门 stamp 点确认**，本侦察未逐字节验证 stamp 值。
- G-B 派生正确性（quarter=subBlock/4、numLanes=numSubBlocks/2 等）**须逐 super-block byte-exact 复核**——ggml 各超块 subBlock 不一（16 vs 32），派生公式跨格是否恒等未逐格证。
- 19 处全在**前门 typed-region CONSTRUCTED emit 体**内（`emitTypedRepackGem{v,m}LoopBody` 的 K-quant / grid / codebook 分支），**非**已退役的 direct-emitter bypass（`schema/emit-bypass-whitelist.v1.json` baseline_count=0，K-quant/iq2-grid 均已 CONSTRUCTED）。⟹ A 线动的是 CONSTRUCTED body 内部的字面量来源，**不触碰** bypass 退役棘轮。
