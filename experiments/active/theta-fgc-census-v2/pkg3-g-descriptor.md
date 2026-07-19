# 包3 · g 进料口盘点(描述符) — θ=f(g,c) 三角色全景盘点 v2

**HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（`git rev-parse HEAD` 实测 == 令定 ref；工作树 == HEAD）
只读·零施工·零板时·事实+出处。是不是缺陷由上游判。

## 0 · 口径 / pin 漂移声明（读数前必看）

- 本包引用的 census 现成数据（`.trellis/tasks/archive/2026-07/07-19-aline-census-classify/research/aline-census-head.md` 等）自钉 HEAD = `f457e73371b1f714d7e4886ca8859392613e457f`，落后本 pin **19 commits**（`git rev-list --count`）。
- 描述符 td（`include/Weft/Dialect/RVV/IR/RVVOps.td`）在两 pin 间 **byte-identical**（`git diff --stat f457..d173 -- <td>` = 空）⟹ 本包所有 td file:line 在本 pin 有效。
- 发射器侧文件有漂移：`RVVToEmitCTernaryBinary.cpp` 在 f457..d173 间 **+220/-249（469 行）**（`git diff --stat`）；`GridCodebook`/`ForwardElementwise`/`KQuant`/`Internal.h` 无漂移。⟹ Ternary 的反向计数我在本 pin 机核重取（见 §4），其余照 census。
- 全文 file:line = 本 pin 工作树读数，除非显式标 `@f457`。

---

## 1 · 描述符全字段总览（RVVOps.td · 本 pin）

- `grep -c OptionalAttr <td>` = **165**（OptionalAttr 声明数）。
- 机算抽全部 attr 声明名（含 required `I64Attr:$x` / `StrAttr:$x` / `BoolAttr:$x` / `OptionalAttr<...>` / `DenseI8ArrayAttr` / `FlatSymbolRefAttr` / `WEFTRVV_PolicyAttr`）后去重 = **221 个 distinct attr 名**（`scratchpad/attrs.txt`）。这是「描述符全字段」的机算全集。

按三角色 + 元数据四分（分类为本 agent 依名义/文档判读·非上游裁定）：

| 桶 | 代表字段（非穷举） | 大致处 | 说明 |
|---|---|---|---|
| **g（格式侧结构量）** | qk / sub_block / *_block_stride / *_byte_offset / num_groups / group_lanes / half_lanes / weight_interleave / codebook / table_symbol / format / quant | 见 §2（64 个名） | 应住描述符·本包主对象 |
| **θ（输出/决定·却住在描述符）** | integer_core_lmul / strip_lmul / multi_block_factor / strip_elision / fold_structure / numerics_tier / unroll_factor / accumulator_lmul / product_lmul / result_lmul / *_sew / emit_loop_schedule / column_group_tile / half_lanes(值=strip 宽) | 见 §3.θ | **θ 焊/盖在描述符里**·旧称「c 轴烘焙」实为此类 |
| **c（板侧能力量·住在描述符）** | min_vlen / minimum_vlen / opponent_vlen_native_floor / required_capabilities / vector_register_budget / peak_live_vector_groups | 6 处名 | c 本应住能力表；描述符里的 min_vlen 由 td 明标「read-only **advisory**」（RVVOps.td:4220 区 GgmlQuantContractionOp doc） |
| **元数据/契约/关系角色（verifier 用）** | planning_contract / remediation_* / performance_admission_* / resource_cost_* / schedule_decision_* / purpose / origin / status / *_relation / *_role / *_memory_form / *_scope / from_phase / to_phase | 余数（~130+） | 多为 Typed*PreRealizedBodyOp 的结构角色 + GearboxCrossRegionHandoffOp 的规划日志 |

> **本包只对 g 桶做逐字段消费核查 + 反向 + 定点**（§2–§5）。θ/c/元数据桶给计数与代表·消费核查见其它包（θ 焊死清单 = 包1/2）。

---

## 2 · g 桶字段清单（64 个 distinct 名·`scratchpad/gfields.txt`）

### 2.1 名 / 类型 / 分组

机算谓词（byte_offset$ / _stride$ / qk / sub_block / groups / lanes / interleave / codebook / table_symbol / format / quant / layout / xor_mask / offset_bias）命中 **64 个 distinct g 字段名**。分组：

- **块字节 stride（7）**：`weight_block_stride` `activation_block_stride` `block_stride` `src_block_stride` `dst_block_stride` `lhs_block_stride` `rhs_block_stride`（全 I64Attr，多为 required；`block_stride` 在 LoadOp/一些 op 为 OptionalAttr）。
- **区偏移 byte_offset（34）**：`weight_qs_byte_offset` `weight_qh_byte_offset` `weight_d_byte_offset` `weight_dmin_byte_offset` `weight_scales_byte_offset` `weight_scales_h/l/high_byte_offset` `weight_hmask_byte_offset` `weight_gas_byte_offset` `weight_ls_byte_offset` `weight_sign(s)_byte_offset` `weight_min_byte_offset` `weight_quant_byte_offset` `weight_scale_byte_offset` `weight_offset_bias` / `activation_{d,quant,bsums,high,sum,scale}_byte_offset` / `bsums_byte_offset` `qh_byte_offset` `quant_byte_offset` `scale_byte_offset` `sum_byte_offset` / `lhs_{scale,min}_byte_offset` `rhs_{scale,sum}_byte_offset` / `src/dst_quant_byte_offset`（全 I64Attr）。
- **块/子块几何（14）**：`qk` `qk_sub` `sub_block` `num_sub_blocks` `n_subblocks` `num_groups` `groups_per_sub` `group_lanes` `half_lanes` `num_groups_per_half` `indices_per_sub_block` `signs_per_sub_block` `num_lanes` `activation_blocks_per_weight`。（`qk`/`sub_block` = required I64Attr；阶段2 新增的 `num_groups`/`groups_per_sub`/`group_lanes`/`num_lanes`/`indices_per_sub_block`/`signs_per_sub_block`/`num_groups_per_half` = OptionalAttr<I64Attr>·见 §2.3。）
- **交织宽（2）**：`weight_interleave`（I64Attr @4603/4715/4794/4901/5009/5040/10756/10912/11843）`activation_interleave`（I64Attr @5010/11844）。
- **码本/格式身份（7）**：`codebook`（**DenseI8ArrayAttr** @4048/5307/5433/7990/11546/11608·恰 16 int8 entry）`table_symbol`（StrAttr @4049）`quant`（OptionalAttr<StrAttr> @4306）`format`（StrAttr @9589）`weight_layout`（StrAttr @4310）`packing_layout`（StrAttr @404）`xor_mask`（I64Attr @4795）。
- **附注（未计入 64、边界 g-身份 StrAttr）**：`scale_model`（StrAttr·~30 op）`decode_model`（StrAttr·~10 op）`scale_role`（StrAttr @830/1618/1679/1744）`kind`——格式-解码身份串，被 f 当**族选择主键**读（见 §3）。

### 2.2 哪些格式盖了值 / 由前门哪行盖 / g 源

**成熟 g 管线（block-dot 家族）** —— g 值不焊在发射器，而是：

1. **g 源正本（per-format 常量表）** = `include/Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h:1503-1526` —— **24 张 `k<Format>Facts[]` constexpr 表**（`kQ4KFacts`/`kIQ4XSFacts`/`kIQ1SFacts`/…/`kQ10Facts`），每张是 `MonolithicBlockDotI64Attr{name,value}` 列表。例：
   - `kQ4KFacts` = `{"qk",256}{"sub_block",32}{"weight_block_stride",144}{"activation_block_stride",292}{"weight_d_byte_offset",0}{"weight_dmin_byte_offset",2}{"weight_scales_byte_offset",4}{"weight_qs_byte_offset",16}…`（RVVMonolithicBlockDotFamily.h:1503）
   - `kQ2KFacts` 内含 `{"weight_d_byte_offset",80}{"weight_dmin_byte_offset",82}`（:1512）—— 注意此二值在 block-dot 家族**是描述符化的**；同二值在 dequant 标量 fold 路径**却焊死**（§4 KQuant #5/#6·ISSUE-115·两路架构不同）。
2. **前门盖章** = `lib/Plugin/RVV/FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp`：
   - `factByName` lambda（:621-626）按名从 `entry.facts` 取值，缺则 `llvm_unreachable("…missing block-format fact")`。
   - `state.addAttribute(<name>, builder.getI64IntegerAttr(<val>))` 盖到 op：`block_stride` @323、`quant_byte_offset` @325、`lhs/rhs_block_stride` @483/485、`lhs_min/rhs_sum_byte_offset` @508/510、`qh_byte_offset`+`block_stride` @536/537、`qk`+`weight_block_stride` @678/679、`qk`+…+`activation_blocks_per_weight` @912/956、`qk_sub` @1063…（全文 g-attr stamp 字符串 `grep -c` = **395**）。
3. **发射器读** = typed getter（§3 category ii）。

⟹ block-dot 家族的 g 是「**声明一次(family header)→前门 stamp→发射器 fail-closed 读**」。24 格覆盖 = kQ4K/Q2K/Q3K/Q5K/Q6K/IQ4XS/IQ1S/IQ1M/IQ2XXS/IQ2XS/IQ2S/IQ3XXS/IQ3S/TQ10/TQ20/Q40/Q80/IQ4NL/Q41/Q50/Q51/MXFP4/NVFP4/Q10。

**dequant-stream 家族的 g（另一路·部分焊死）** —— 见 §4 反向（ForwardElementwise dequant-row·GridCodebook grid 几何·KQuant fold 偏移不走此前门）。

### 2.3 阶段2 新增字段（OptionalAttr<I64Attr>·fail-closed 范式）

commit `85cd5022f`（census §1.1 记）把 KQuant 17 处 grid/subblock **焊死**常量换成描述符读，新增 OptionalAttr<I64Attr>：`groups_per_sub`(@7149/7250/8283) `num_groups`(@7348/7476/8407) `indices_per_sub_block`(@7349/8408) `group_lanes`(@7350/8410) `num_groups_per_half`(@8151/8284) `signs_per_sub_block`(@8409) `num_lanes`(@6435/6513) `n_subblocks`(@10782/10927/11870) `weight_qh_byte_offset`(OptionalAttr @6044/10661/10768/11117/11856) `weight_offset_bias`(@10662/11118) `weight_dmin/scales_byte_offset`+`activation_bsums_byte_offset`(@10779-82) `weight_scales_high_byte_offset`(@10791) `lhs/rhs_*`(@10080-83/10138-41) `qh_byte_offset`+`block_stride`(@10193-94) `activation_quant_byte_offset`(@6237)。范式正本 = LoadOp（RVVOps.td:2645 doc 明写「block_stride/quant_byte_offset must not be set, **fail-closed**」·OptionalAttr @2677-2678）。

---

## 3 · g 逐字段消费核查（i 读入 f / ii 发射器直读 / iii 零消费）

**方法**：为 221 个 attr 生成 typed getter（`get<CamelCase>`），机扫 `lib/ tools/ include/`（排 `.inc` 生成物）命中文件数（`scratchpad/consumption.txt`）。

### 3.1 g 桶结论：64 个 g 字段 **无一落零消费**

`comm -12 gfields.txt <(零 getter 名)` = **空**。⟹ **g 桶 (iii) 零消费 = 0**：所有 g 字段都至少有一个 typed getter 读者。「搬了个寂寞」在 g 桶为空。

### 3.2 (i) 被某条 f 读（参与算 θ）——具名

两条 θ-算子 f 的 op-attr 读（`grep -oE '\.get[A-Z]\w+\(\)'` 计次）：

- **f₁ = `selectRepackAccumulatorLMUL` / `selectContractionAlgorithm` / `stampScheduleSelections`**（`RVVLowerQuantContraction.cpp`）读：
  - `getScaleModel()` **68×**（族/算法选择主键·f₁ 首参 `llvm::StringRef scaleModel` @1286）—— g-身份串。
  - `getQk()` **19×**、`getMRegime()` **4×**（regime 轴·θ 选择输入）。
  - `getOpponentVlenNativeFloorAttr()` **1×**、`getBlockDotComputeHeavy()` **1×**、`getBlockDotMemoryBound()` **1×**——「结构化对手事实」（measurement/对手 regime·非纯 g）→ 喂 `selectContractionAlgorithm` 的 θ。
  - `getActivationBlockStride()` 7× / `getQuantByteOffset()` 6× / `getWeightBlockStride()` 2× / `getActivationHighByteOffset()` 1×——**读来转 stamp 到子 op**（结构管线·非算 θ）。
- **f₂ = `MaterializeRVVRepackStripWidth`**（`RVVRepackStripWidthMaterialization.cpp`）读：
  - `getWeightInterleave()`（@140/144/156/160/173/177）→ `deriveRepackHalfLanes(vlenBits, weight_interleave)` → **stamp `half_lanes`（θ）+ RVV0.7 stamp `integer_core_lmul="m1"`（θ）**。⟹ **`weight_interleave`（g·交织宽）= category (i)**，是唯一真喂 θ-几何的纯 g 字段。

> **VLEN 不来自描述符**：f₁/f₂ 均从 `-march` 经 `deriveMinimumVLEN` 取 minVLEN（RVVLowerQuantContraction.cpp:9-12/1406/1544·RVVRepackStripWidthMaterialization.cpp:100）。td 明标 `min_vlen` attr「read-only **advisory**·NOT the source」（RVVOps.td GgmlQuantContractionOp doc + 注释 :1526/1540）。⟹ 描述符 c 字段 `min_vlen` = **零消费**（getMinVlen typed-getter 命中 0；string "min_vlen" 命中 1=doc/advisory）。

### 3.3 (ii) 被发射器直读（只做结构展开·不进 f）

g 桶剩余 ~62 字段（全部 byte_offset / stride / 子块几何 / codebook / format-身份）——发射器 typed getter 直读，emit 成指针算术 / vsetvli / gather 结构，**不参与任何 θ 决策**。例：`getWeightQsByteOffset` `getWeightScalesByteOffset` `getSubBlock` `getNumGroups` `getGroupLanes` `getHalfLanes`（读值展开）`getCodebook`（DenseI8Array 结构消费）。这是 g 的**正常归宿**（结构展开）。

### 3.4 (iii) 零消费——g 桶 = 0；描述符全集另有 34 个「零 typed-getter」名（**多非 g**·如实并列）

机扫得 **34 个 attr 无 typed-getter 命中**（`scratchpad/consumption.txt`）。**其中 0 个是 g 字段**。34 个按角色：

- **c 桶**：`min_vlen`（td 明标 advisory·上 §3.2）`required_capabilities`（typed 0·但 string-ref 27=经泛型 getAttr 消费·非真零）。
- **θ 桶**：`unroll_factor`（typed 0·string 3）`selected_variant`（typed 0·string 37=经字符串消费）`selected_path_role`（string 8）。
- **元数据/契约（GearboxCrossRegionHandoffOp @365-432 的规划日志）**：`planning_contract` `resource_cost_{model,contract,blocker,loop_body_steps}` `performance_admission_{decision,closure,reopen_requirement}` `beyond_local_repair_admission_{contract,decision,blocker,reopen_requirement}` `remediation_{plan,plan_contract,statement_strategy,vector_budget,schedule_contract,unpack_plan,product_plan,reduction_plan,vl_plan}` `schedule_decision{,_contract,_reason}` `rvv_construction_protocol` `rvv_emitc_route_mapping` `purpose` `origin?` `exec_binding` `arithmetic_kind`。
  - 这些多有 string-ref（stamp 写侧；如 planning_contract string 4 / remediation_plan 6 / schedule_decision 5 / purpose 12）——**是否只写不读需逐字段 string 读/写侧三分**（本包未穷做·超 g 范围）。**如实标注：这批是「搬了个寂寞」候选，但属元数据/θ/c 桶，不属 g 桶。**

> **g 桶净结论**：进料的 g **全部被消费**（1 个走 f = weight_interleave；余 ~62 走发射器直读；scale_model/qk/m_regime 双读=既喂 f 选择又参与 emit）。g 桶零「搬了个寂寞」。

---

## 4 · 反向：发射器仍在读 g、但描述符没字段（焊死/派生残余）

**机核（本 pin·pin-shape 谓词 `int64_t NAME=<int>;` 排 `.get`/`=0`/`=1`）**：

| 文件 | 处数(本 pin) | 逐处 file:line（值） | 登记 |
|---|---|---|---|
| `RVVToEmitCGridCodebook.cpp` | **9** | 76 `subBlockLanes=32`·504 `groupLanes=8`·1002 `groupLanes=8`·1514 `halfLanes=16`·1629 `pairHalves=4`·1986 `halfLanes=16`·2102 `pairHalves=4`·2466 `groupLanes=8`(const)·2467 `numGroups=4`(const) | **ISSUE-118**（回门待扫·扇出最高） |
| `RVVToEmitCTernaryBinary.cpp` | **7**（本 pin；census@f457=8） | 854 `groupLanes=8`·869 `halvesPerSub=2`·870 `groupsPerHalf=2`·871 `halfLanes=16`·1588 `planeLanes=32`·1589 `planesPerChunk=4`·1590 `chunkBytes=32` | **ISSUE-118** |
| `RVVToEmitCKQuant.cpp` | **2** | 4607 `weightDOffset=80`·4608 `weightDminOffset=82`（q2_K·`emitTypedSuperBlockScalarScaleMinLoopBody`） | **ISSUE-115**（=「#5/#6」·专用 fold-brick op 待裁） |
| `RVVToEmitCForwardElementwise.cpp` | pin-shape **3**（2709 `qk=32`·3575 `qhOff=16`·4564 `qk=256`）；宽谓词 **~51** | 5333-5351 per-format stride switch **19 case**（Q2K84/Q3K110/Q4K144/Q5K176/Q6K210/MXFP4 str17/NVFP4 str36/TQ1 54/TQ2 66/Q10 str18/IQ4NL str18/IQ2XXS66/IQ2XS74/IQ2S82/IQ3XXS98/IQ3S110/IQ1S50/IQ1M56/IQ4XS136）+ 多变量声明（2661-2670/3259/3394/3743/3870/4048…） | **ISSUE-119**（dequant-row ~40·monolith-fallback·须独立 scoping） |

- **Ternary 本 pin = 7 ≠ census@f457 = 8**：`dotStripLanes=32`（f457 @2060·tq1_0 dot-strip）在 f457..d173 的 469 行漂移中**已从本 pin 消失**（`grep dotStripLanes` 本 pin = 空；`git grep @f457` 命中 :2060/2061/2347/2351）。⟹ 反向计数须以本 pin=7 为准（census 数据是旧 pin）。
- **KQuant #5/#6 特例（ISSUE-115）**：`weight_d/dmin_byte_offset` **不能同法加 attr**——q2_K 整数核 op（`GgmlBlockDotQ2KQ8KIntegerCoreOp`）ODS charter 明标 fp32 scalar fold「DELIBERATELY OUT OF SCOPE」；加 attr=把故意排除的 fold 拉进整数核=语义冲突。**同二值在 block-dot 家族的 `kQ2KFacts`（Family.h:1512）却是描述符化的**——两路架构不同（此为 dequant/scalar-fold 路）。ISSUE-115 状态=待裁·公式占比封顶 35/37。
- **「coreLmul 焊死」不入 g 反向**：census §0/§3.2 + ISSUE-118 订正——主会话初判「焊死 11」经宽谓词逐处核 = 5 真焊死(θ·假旋钮形态·GridCodebook 497/995·Ternary 862·BQL 111/404)+5 live-default-override(θ value_or)+1 verifier-pinned(CodebookFp4 109)。这些 coreLmul/wideLmul 是 **θ**（不是 g·也不是 c）·全 MAINTAIN([K-10]·Context 无字段/无 IR 宽度)·**归 θ 桶（包1/2）·不在本 g 反向表**。
- **Internal.h 板派生（边界·非纯 g）**：`3648 stripWidth=8`·`3649 numStrips=4`（注释自述「derived from l8」）——c 轴板派生残余（随 ISSUE-113），不计 g。

**反向 g 残余小结（本 pin）**：pin-shape 焊死 g = GridCodebook 9 + Ternary 7 + KQuant 2 = **18 处**；另 ForwardElementwise dequant-row **~51 处宽谓词** g 布局烘焙（ISSUE-119·未穷举到 100%）。全部维持 baked（byte-exact·保守默认）·转描述符须回门/独立 scoping。

---

## 5 · 定点：各 g 类 ↔ 哪个字段 / 哪类还没字段

| g 类 | 对应描述符字段 | 状态 |
|---|---|---|
| **块宽（block width）** | `qk`（I64Attr·required·32/64/128/256）；超块另 `sub_block`（16/32）+ `num_sub_blocks`/`n_subblocks` | **有字段**（成熟·Family.h `{"qk",…}` stamp） |
| **块字节 stride** | `weight_block_stride`/`activation_block_stride`/`block_stride`（+lhs/rhs/src/dst 变体） | **有字段**；但 dequant-row 路的 stride 仍焊死（§4·ISSUE-119） |
| **交织宽（interleave）** | `weight_interleave` / `activation_interleave`（I64Attr） | **有字段**·且是唯一喂 θ(strip 宽)的纯 g（§3.2 f₂） |
| **子块几何（sub-block geometry）** | `num_groups`/`groups_per_sub`/`group_lanes`/`half_lanes`/`num_groups_per_half`/`indices_per_sub_block`/`signs_per_sub_block`/`num_lanes`（阶段2 OptionalAttr) | **block-dot 已有字段**；但 **GridCodebook/Ternary grid 几何（subBlockLanes/pairHalves/planeLanes/chunkBytes 等 16 处）还没字段**（§4·ISSUE-118·同类未做的 op） |
| **码本尺寸（codebook size）** | 小码本(FP4/IQ4NL/IQ4XS 16-entry)：`codebook`=**DenseI8ArrayAttr**（数据本身·尺寸=数组长度隐含·**无独立数字字段**）+ `table_symbol` | **无「码本尺寸」数字字段**：`grep codebook_size\|table_size\|num_entries\|grid_size` = **空**（机核·NO numeric codebook-size field） |
| **grid 码本几何（iq2/iq3 super-block grid: 256/512 entry·group-of-4/8）** | —— | **完全无字段**：grid 几何全焊在 GridCodebook.cpp（§4·ISSUE-118 反向残余）·描述符里无 grid-count/grid-size 任何字段 |
| **格式身份/解码模型** | `format`/`quant`/`scale_model`/`decode_model`/`scale_role`/`weight_layout`/`packing_layout`（StrAttr） | **有字段**（scale_model 是 f₁ 族选择主键） |

**「哪类还没字段」定点结论**：
1. **码本尺寸** —— 无专用数字字段（小码本靠 DenseI8Array 长度隐含；无 codebook_size/num_entries）。
2. **grid 码本几何**（iq2/iq3/iq1_m grid + tq plane）—— 描述符零字段·全焊 GridCodebook/Ternary（ISSUE-118 待回门·同阶段2 路B 可套但未做）。
3. **dequant-row(ForwardElementwise) 全布局**（stride/qsOff/dOff/qhOff per-format）—— 描述符零字段·monolith-fallback 焊在发射体（ISSUE-119 待独立 scoping）。
4. **q2_K scalar-fold 偏移 `weight_d/dmin_byte_offset`** —— dequant 路无字段（block-dot 路 `kQ2KFacts` 有）·ISSUE-115 语义冲突·须专用 fold-brick op。

---

## 6 · 计数汇总（纯数字·口径·无解读）

- 描述符 OptionalAttr 声明 = **165**（`grep -c OptionalAttr`）；distinct attr 名全集 = **221**（去重·含 required）。
- **g 桶 distinct 字段名 = 64**（数字/几何/偏移/stride/codebook/format-身份·`scratchpad/gfields.txt`）；另 ~4 格式-身份 StrAttr（scale_model/decode_model/scale_role/kind）未计入 64。
- **g 桶零消费(iii) = 0**（64 g 字段全有 typed-getter 读者·`comm -12` = 空）。
- 描述符全集零-typed-getter 名 = **34**（**其中 g 字段 = 0**；余为 c=min_vlen 等 / θ=unroll_factor 等 / 元数据契约·多有 string-ref 写侧·真「只写不读」需逐字段三分·未穷做）。
- **g 桶消费三分**：(i) 读入 f = 主为 `weight_interleave`（→strip 宽）+ `scale_model`/`qk`/`m_regime`/opponent 三事实（→族/LMUL 选择）；(ii) 发射器直读 = 余 ~62（byte_offset/stride/几何/codebook）；(iii) 零消费 = 0。
- **反向焊死 g（本 pin·pin-shape）= 18 处**（GridCodebook 9 + Ternary 7 + KQuant 2）；另 ForwardElementwise dequant-row 宽谓词 ~51 处（ISSUE-119·未穷举）。Ternary 本 pin=7 ≠ census@f457=8（dotStripLanes 已消失）。
- g 源正本 = `RVVMonolithicBlockDotFamily.h:1503-1526`（24 张 k*Facts 表）；前门 stamp = `RVVMonolithicBlockDotSourceFrontDoor.cpp`（g-attr stamp 字符串 grep-c=395）。

## 7 · 采不到 / 未核（如实）

- 34 个零-typed-getter attr 的「只写不读 vs 泛型 string 读」逐字段三分未穷做（超 g 范围·仅抽验 required_capabilities/selected_variant=有 string 读·planning/remediation/admission=写向）。
- ForwardElementwise dequant-row ~40/~51 未穷举到每个 dOff/mOff/sub 小常量（census §Caveat 亦标·数据源自 census 宽谓词）。
- ISSUE-118/119 提「阶段2 已建同名 attr 可复用」为类比推断（GridCodebook/Ternary 是不同 op·须各自声明·census 未逐一验证 call-site 能否填 Context）——本包照引·未独立复验。
- 未编译/未跑任何 fixture（只读令）。所有「byte-exact」「消费」断言 = 静态阅读 + getter/string 机扫推断。
