# Research: DeferredDequant 目标形态 + 已有层清单（格式描述符 / 能力事实）

- **Query**: DeferredDequant 的 g/c 分家结构长什么样？KQuant/BQL 往它靠的目标形态？已有层现状（哪些可复用而非新建）？
- **Scope**: internal
- **Date**: 2026-07-18

## DeferredDequant —— 两轴皆干净的正例（03-B-DEBAKE §⑦ 命名）

**文件**：`lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp`（本轮 @ `0aee07b4` 复跑）

| 轴 | 干净证据（机检） |
|---|---|
| **c 轴（板值）** | `value_or("mX")` 板默认 = **0**（`git grep -c '\.value_or\("(m1\|m2\|m4\|m8\|mf2\|mf4\|mf8)"\)'` → 空） |
| **g 轴（格式字面量）** | 裸格式结构字面量 = **0**（`int64_t NAME=<int>;` 排 `.get`/0/1 后为空） |
| **板宽来源** | **读自 IR 向量类型** `getLmul()` × **21**，全程参数孔 |
| **fail-closed** | 类型不带板值就不发射：`:158 if (!accVecType \|\| accVecType.getLmul() != "m1")`、`:350 != "m8"`、`:359/:652 reduceVecType.getLmul() != "m1"`、`:640-642` acc SEW32 ∧ lmul∈{m1,m2,m4,m8} 否则拒 |

**目标形态本质 = 两条纪律**：
1. **板值（c 轴）**：**从 IR 类型 / op 参数孔读**（`getLmul()` / `getIntegerCoreLmul()`），**绝不 `.value_or("板值")` 自补**；缺则 **fail-closed**（`notifyMatchFailure`），不合成。
2. **格式常数（g 轴）**：**从 op 参数孔 / 描述符读**（`coreOp.getXxx()`），**绝不裸字面量烘焙**；描述符事实在**前门**由 decode-facts 结构 stamp。

## KQuant / BQL 往它靠的具体目标结构

### c 轴（BQL 18 + KQuant 3 + BQL 焊死 5）→ 能力事实层
`value_or("mf2")` → **fail-closed 读** `getIntegerCoreLmul()`（缺则 `notifyMatchFailure`）。
- 板宽的**能力来源已在**：配对的 `half_lanes` 由能力闭式 `deriveRepackHalfLanes(minVLEN)` 前门算出并 stamp；`integer_core_lmul` 由前门（`0b18a3da` 后）**恒显式 stamp**。⟹ emitter 侧只需**读**，与 DeferredDequant 的 `getLmul()` 读同构。
- verifier 已对 `(integer_core_lmul, half_lanes)` **I7 fail-closed 钉死**（`RVVOps.td:4580-4581`）——DeferredDequant 式 fail-closed 语义在 IR 层已具备。

### g 轴（KQuant 19）→ 格式描述符层
裸字面量 → **`coreOp.getXxx()` 参数孔**（对已有 getter：直接换）或**前门 decode-facts 描述符新字段**（对 grid/codebook 无属性者）。
- 同文件已有大量 `coreOp.getXxx()` 就地读的兄弟行（scalesOffset/qsOffset/… 全走参数孔），**目标写法有现成范本可照抄**。

## 已有层清单（骨架不推倒 = 接进这些，非新建）

### 格式描述符层（g 轴目标）—— 已在
- **前门 decode-facts 描述符结构**：`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp` 内
  - `CodebookDecodeFacts` / `kIq4XsDecodeFacts` / `kIq4NlDecodeFacts`（codebook：weightBlockStride/weightQuantByteOffset/**nSubblocks**/weightScalesLowByteOffset/weightScalesHighByteOffset/codebook…）
  - `KQuantDecodeFacts`（K-quant：block offsets，`kQ5KDecodeFacts` 等，参数化 `lowerToRepackGem{v,m}KQuant`）
  - grid decode facts（`lowerToRepackGem{v,m}Grid`，decode_model iq2_xxs/iq2_xs/iq2_s）
- **op 侧 `I64Attr` 属性**（`include/Weft/Dialect/RVV/IR/RVVOps.td`）：`sub_block` · `num_sub_blocks` · `weight_scales_byte_offset` · `weight_scales_high_byte_offset` · `weight_qs_byte_offset` · `weight_qh_byte_offset` · `weight_gas_byte_offset` · **`weight_d_byte_offset`（14 def）** · **`weight_dmin_byte_offset`（7 def）** · **`num_lanes`（1 def）** · `activation_*_byte_offset`。
- **G-C 缺口**：grid 子结构 `groups_per_sub` / `num_groups` / `indices_per_sub_block` / `signs_per_sub_block` / `num_groups_per_half` / `group_lanes` = **0 def**（回门候选新字段，见 debake-staged-plan §4）。

### 能力事实层（c 轴目标）—— 已在
- **`deriveRepackHalfLanes(vlenBits)`**（`RVVLowerQuantContraction.cpp:1222`）：能力闭式 `half_lanes = (vlen<128)?0:min(vlen/16,16)`。**[A1] 板轴 closed-form 正本**（02-差距报告 A1：同一模式零改动、`march=` 唯一变量、`half_lanes` predicted==observed 全 MATCH @ 7 march 值）。
- `half_lanes`（必填 `I64Attr`）+ `integer_core_lmul`（`OptionalAttr<StrAttr>`，前门恒 stamp）= 能力事实进 op 的载体。
- **[D-2a] / 能力谓词**：`isM1 = isRVV0p7` 已被**measured gate** 替换（`RVVLowerQuantContraction.cpp:1281` 注释「Replaces the correctness-only isM1=isRVV0p7 fork with the measured gate」`accLmulChoice.useM1`）——c 轴已从「板串分支」迁向「实测标定事实」。

## 与 A′ 上板的关系（裁决7 也放行 A′）

- **A′ 承重 = A1「同一模式零改动、两 VLEN 闭式导出到两 body、各自板上 byte-correct」**（02-差距报告 A1 · §H ★★ 最高杠杆）。A 线 c 轴收尾**直接喂 A′**：把 `integer_core_lmul` 从「缺席信号 / value_or 自补」升为「前门恒显式 stamp + emitter fail-closed 读」= 让 `(half_lanes, integer_core_lmul)` 成为**唯一随能力移动的构造锚点**，正是 A′ 要展示的「零改动双 body」的属性载体。
- **复用点**：A′ 上板（rvv VLEN128 + k1 VLEN256，同 fixture、`march=` 唯一变量）与 A 线 c 轴共用 `deriveRepackHalfLanes` 闭式 + `integer_core_lmul` stamp；A 线把该链**去烘焙**后，A′ 的「板上各自 byte-correct」腿在同一属性上兑现。**A 线（结构去烘焙·byte-exact）与 A′（上板证 byte-correct）不冲突、可先后**：A 线先净化消费侧，A′ 再上板封字节。

## Caveats

- DeferredDequant 是 **dequant（GEVM 累加+reduce）** 路径，**结构上比 BQL repack GEVM/GEMM 简单**（无 super-block fold / 无 grid gather）。「往它靠」指**两轴纪律**（fail-closed 读、零烘焙），**非**结构照搬——BQL/KQuant 的格式 fold/grid 结构保留，只把**板值来源**与**格式常数来源**换成参数孔/描述符。
- 「前门恒 stamp」的完整性（无 wired leaf 漏）本侦察未逐 leaf 枚举证（见 census-c-axis §Caveats）。
