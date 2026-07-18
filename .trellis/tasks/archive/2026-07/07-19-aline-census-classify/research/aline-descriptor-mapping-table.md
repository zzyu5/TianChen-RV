# Research: A线 描述符类映射表（新增 schema 字段·供 supervisor 回门）

- **Query**: 所有「描述符」类残量出表 {常数名·现值·format·拟 attr 名·类型·前门 stamp 源·fail-closed 诊断串}
- **Scope**: internal（提案·零改·**回门材料·非施工承诺**）
- **Date**: 2026-07-19
- **ASCII key 纪律**（记忆 workflow-schema-ascii-keys）：所有 attr key = `^[a-zA-Z0-9_.-]{1,64}$` 纯 ASCII·中文只在 description 列。
- **范式**：阶段2路B 已证——`OptionalAttr<I64Attr>` per op（保 fixture 解析）+ 发射体 fail-closed 读（缺席 `notifyMatchFailure` 点名·**严禁 value_or 自补到 g 轴**）+ 前门 `RVVMonolithicBlockDotSourceFrontDoor.cpp` stamp + I7 verifier allowlist 只加 new attr + regen-diff 0。

---

## 簇 1 · GridCodebook.cpp grid 几何（9 处·套阶段2路B·未登记·**扇出最高·风险最低回门**）

enclosing helper 持 Context 结构体（`IQ2XXS/IQ3XXS/IQ3SGridBodyContext` 等）→ 描述符须经 Context 从 op attr 填（call-site）。

| HEAD file:line | 常数(现值) | format | 拟 attr key (ASCII) | attr 类型 | 前门 stamp 源 | fail-closed 诊断串(建议) |
|---|---|---|---|---|---|---|
| GridCodebook 76 | `subBlockLanes=32` | iq2/iq3 super-block | `sub_block_lanes` | `OptionalAttr<I64Attr>` | RVVMonolithicBlockDotSourceFrontDoor | `"grid body missing sub_block_lanes descriptor"` |
| GridCodebook 504 | `groupLanes=8` | iq3_xxs | `group_lanes` | `OptionalAttr<I64Attr>` | 同上（阶段2 已建同名 attr·复用） | `"iq3_xxs grid body missing group_lanes descriptor"` |
| GridCodebook 1002 | `groupLanes=8` | iq3_s | `group_lanes` | `OptionalAttr<I64Attr>` | 同上 | `"iq3_s grid body missing group_lanes descriptor"` |
| GridCodebook 1514 | `halfLanes=16` | iq3 | `half_lanes` | `OptionalAttr<I64Attr>` | 同上（td 已有 half_lanes I64Attr·核复用） | `"grid body missing half_lanes descriptor"` |
| GridCodebook 1629 | `pairHalves=4` | iq3 | `pair_halves` | `OptionalAttr<I64Attr>` | 同上 | `"grid body missing pair_halves descriptor"` |
| GridCodebook 1986 | `halfLanes=16` | iq3(other) | `half_lanes` | `OptionalAttr<I64Attr>` | 同上 | 同上 |
| GridCodebook 2102 | `pairHalves=4` | iq3(other) | `pair_halves` | `OptionalAttr<I64Attr>` | 同上 | 同上 |
| GridCodebook 2466 | `groupLanes=8`(const) | iq2/iq3 | `group_lanes` | `OptionalAttr<I64Attr>` | 同上 | 同上 |
| GridCodebook 2467 | `numGroups=4`(const) | iq2/iq3 | `num_groups` | `OptionalAttr<I64Attr>` | 同上（阶段2 已建 num_groups·复用） | `"grid body missing num_groups descriptor"` |

## 簇 2 · TernaryBinary.cpp grid/plane 几何（8 处·套阶段2路B·未登记）

| HEAD file:line | 常数(现值) | format | 拟 attr key (ASCII) | attr 类型 | 前门 stamp 源 | fail-closed 诊断串(建议) |
|---|---|---|---|---|---|---|
| Ternary 854 | `groupLanes=8` | iq1_m | `group_lanes` | `OptionalAttr<I64Attr>` | FrontDoor | `"iq1_m grid body missing group_lanes descriptor"` |
| Ternary 869 | `halvesPerSub=2` | iq1_m | `halves_per_sub` | `OptionalAttr<I64Attr>` | 同上 | `"iq1_m grid body missing halves_per_sub descriptor"` |
| Ternary 870 | `groupsPerHalf=2` | iq1_m | `groups_per_half` | `OptionalAttr<I64Attr>` | 同上 | `"iq1_m grid body missing groups_per_half descriptor"` |
| Ternary 871 | `halfLanes=16` | iq1_m | `half_lanes` | `OptionalAttr<I64Attr>` | 同上 | `"iq1_m grid body missing half_lanes descriptor"` |
| Ternary 1588 | `planeLanes=32` | tq2_0 | `plane_lanes` | `OptionalAttr<I64Attr>` | 同上 | `"tq2_0 body missing plane_lanes descriptor"` |
| Ternary 1589 | `planesPerChunk=4` | tq2_0 | `planes_per_chunk` | `OptionalAttr<I64Attr>` | 同上 | `"tq2_0 body missing planes_per_chunk descriptor"` |
| Ternary 1590 | `chunkBytes=32` | tq2_0 | `chunk_bytes` | `OptionalAttr<I64Attr>` | 同上 | `"tq2_0 body missing chunk_bytes descriptor"` |
| Ternary 2060 | `dotStripLanes=32` | tq1_0 | `dot_strip_lanes` | `OptionalAttr<I64Attr>` | 同上 | `"tq1_0 body missing dot_strip_lanes descriptor"` |

## 簇 3 · KQuant fp32-fold 偏移（2 处·**ISSUE-115 特例·不同法加 attr·须专用 fold-brick op·待裁**）

| HEAD file:line | 常数(现值) | format | 拟法 | 为何不能同法加 attr |
|---|---|---|---|---|
| KQuant 4607 | `weightDOffset=80` | q2_K (block_q2_K x.d fp16 @80) | **专用 fold-brick op**（非 attr） | `GgmlBlockDotQ2KQ8KIntegerCoreOp` ODS charter 明标 fp32 SCALAR fold DELIBERATELY OUT OF SCOPE·加 `weight_d_byte_offset` = 把故意排除的 fold 拉进整数核 op = 语义冲突（ISSUE-115） |
| KQuant 4608 | `weightDminOffset=82` | q2_K (x.dmin fp16 @82) | 同上 | 同上 |

> **回门动作**：supervisor 须先裁「是否建专用 fold-brick op」（core-op 语义级）。**在此裁决前·两字面量维持 baked**（byte-exact 不动·公式占比封顶 35/37）。`activationDOffset=0` 不入表（=0）。

## 簇 4 · ForwardElementwise dequant-row 布局（~40 处·**monolith-fallback 架构·建议独立 issue 先 scoping·非本轮直接回门**）

op = `GgmlDequantizeRowOp`·`emitGgmlDequantizeRowExtended` = keyed-by-format-string monolith fallback·format→布局映射内在于 path。**转描述符 = 前门须为每格 stamp 全布局（block_stride/quant_byte_offset/scale_byte_offset/qh_byte_offset/qk）·发射体逐格改读**——大改。代表性 attr 提案（若纳入）：

| 常数类 | 拟 attr key (ASCII) | 现值样例(逐格·见 census §1.2) |
|---|---|---|
| 块字节 stride | `block_stride` | Q2K 84/Q3K 110/Q4K 144/Q5K 176/Q6K 210/…（19 格 switch @5333-5351） |
| qs 区偏移 | `quant_byte_offset` | 2/1/4/…（per-format） |
| scale/d 偏移 | `scale_byte_offset` | 0/… |
| qh 平面偏移 | `qh_byte_offset` | q5_K 16(@3575)·iq 各异 |
| 块元素数 | `qk` | 32/64/128/256（多为 getQk() 可直改·见 classification 桶 D） |

> **注**：`block_stride`/`quant_byte_offset` 这两 key 在 td 已有先例（RVVOps.td:2665-2677 的 OptionalAttr `block_stride`+fail-closed 范式）·可复用命名。但**该架构面转描述符前须独立裁 scoping**（工作量 ≫ 簇 1/2·且是 B线 recent 代码）。

---

## 回门决策清单（供 supervisor 一屏扫）

| 簇 | 处数 | 新 attr 数(估) | 复用既有 attr? | 阻塞级 | 建议 |
|---|---|---|---|---|---|
| 1 GridCodebook grid | 9 | ~6 唯一(group_lanes/num_groups/half_lanes/pair_halves/sub_block_lanes) | **是·阶段2 已建 group_lanes/num_groups/half_lanes** | 无阻塞·同法 | **优先·风险最低回门** |
| 2 Ternary grid/plane | 8 | ~7 唯一 | 部分（group_lanes/half_lanes 复用） | 无阻塞·同法 | 次优先·与簇1 并 |
| 3 KQuant fold 偏移 | 2 | 0(需新 op) | 否·须 fold-brick op | **须用户裁 core-op 语义** | 不自决·gated ISSUE-115 |
| 4 FwdEW dequant-row | ~40 | ~5 类×多格 | block_stride/quant_byte_offset 有先例 | **须独立 scoping**(大改·B线新码) | 建议开新 issue·非本轮 |

**ASCII key 全表校验**：上列 key 全 `^[a-zA-Z0-9_.-]{1,64}$`（`sub_block_lanes`/`group_lanes`/`half_lanes`/`pair_halves`/`num_groups`/`halves_per_sub`/`groups_per_half`/`plane_lanes`/`planes_per_chunk`/`chunk_bytes`/`dot_strip_lanes`/`block_stride`/`quant_byte_offset`/`scale_byte_offset`/`qh_byte_offset`/`qk`）——**零中文 key**·中文只在 description。

## Caveats

- 簇 1/2 的「复用既有 attr」= 类比推断（阶段2 在 KQuant op 上建了同名 attr·GridCodebook/Ternary 是**不同 op**·须各自加 attr·名可复用但字段是各 op 独立声明）·施工前核 op 定义。
- 簇 4 处数 ~40 是代表性估计·未穷举到每个 dOff/mOff/sub 小常量。
- 所有 stamp 值必 = 现字面量（byte-exact）·前门缺 stamp = 硬失败点名（非 value_or 自补·阶段2 g 轴红线）。
