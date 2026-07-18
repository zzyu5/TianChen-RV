# Research: A 线去烘焙分阶段 plan + 拟新增 schema 字段（回门点）

- **Query**: 骨架不推倒·增量·每阶段 byte-exact 可验的分阶段施工 plan（从最低风险/最高扇出起）+ 反误伤守门 + 公式占比增量 + 新 schema 字段映射表
- **Scope**: internal（侦察产出·供 main 派 implement）
- **Date**: 2026-07-18
- **依据**: census-g-axis-kquant-19.md · census-c-axis-bql-18.md · deferreddequant-target-form.md

## 总纲

A 线 = 把决策从**烘焙**变成**公式**：g 轴（19）→ 格式描述符层 · c 轴（18）→ 能力事实层。**骨架不推倒 = 接进已有层**（`coreOp.getXxx()` / `getIntegerCoreLmul()` / `deriveRepackHalfLanes`），非重写。**byte-exact 是硬约束**（纯结构接层，不改任何可测行为）。

**确凿 scope（机算 · pin `0aee07b4` · HEAD 复现）= 37**：g 轴 KQuant 19 + c 轴 BQL value_or 18。相邻同轴群（KQuant value_or 3 + BQL 焊死 5，共 8）单列后续。

## 反误伤守门（每阶段必过 · 全部已有机检工件）

| 守门 | 判据 | 工件 |
|---|---|---|
| **[K-5] byte-exact** | 整数路径发出的 C **逐字节不变**（BEFORE/AFTER-EQUALITY，clean/forced rebuild） | `test/Conversion/RVV/rvv-emit-*-repack-vlen128.mlir`（q2_K…q6_K/iq*/mxfp4 每格 golden）· `tools/bench/byte-exact-baseline.sh` |
| **[K-5] VLEN 翻转矩阵** | `{128,256}×{m1,m2,m4}` lit 全绿（c 轴改后 half_lanes/lmul 链不动） | VLEN128/VLEN256 repack lit 对 |
| **[F-EMIT] 棘轮不动** | `emit-bypass-whitelist.v1.json` baseline_count 不变（A 线不退役/不新增 bypass，只改 CONSTRUCTED body 内字面量来源） | `schema/emit-bypass-whitelist.v1.json`（现 baseline_count=0）· `schema/coverage-sixstate.v1.json` · `tools/gates/check_frontdoor_provenance.py` |
| **I7 verifier fail-closed** | `(integer_core_lmul, half_lanes)` 组合验证不放松（c 轴改后仍 fail-closed） | `RVVOps.td:4580-4581` 现有 verifier |

> **注（memory 铁律）**：byte-exact gate 必须 forced/clean rebuild + BEFORE/AFTER-EQUALITY；旧的绝对指纹会 STALE，别当 pass/fail 目标——用**每格 golden 逐字节对**。**本 plan 只列守门；侦察阶段零 build。**

## 分阶段 plan（从最低风险/最高扇出起）

### 阶段 0 — c 轴 BQL 18（value_or → fail-closed 读）· **最高扇出 · 最低风险 · 零 schema · 起手**
- **触碰集**：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` **单文件**（18 处 `.value_or("mf2")` → 单一 fail-closed helper `getIntegerCoreLmul()` 缺则 `notifyMatchFailure`）。
- **为何起手**：`0b18a3da` 已让前门**恒显式 stamp** `integer_core_lmul` ⟹ 18 处 default 分支**对所有生产输入是死码** ⟹ 删除 **byte-exact by construction**。**不动 ODS（保持 optional）** = 不破测试/parser fixture。
- **扇出**：18 决策；覆盖 ternary/q4_0/q4_1/q5_0/q5_1/q8_0/codebook/K-quant 所有 repack GEVM/GEMM 体（共用这些 loop-body op）。
- **公式占比增量**：c 轴 `18/38 → 20/38`（BQL 18 转公式；焊死 5 + KQuant 3 留后）。**A 线总 `0/37 → 18/37`**。
- **守门**：byte-exact 全 repack golden · I7 verifier 不动 · 棘轮不动。

### 阶段 1 — g 轴 平 K-quant 6（换已有 getter + 派生）· **低风险 · 零 schema**
- **触碰集**：`RVVToEmitCKQuant.cpp` **单文件**（#1–#6：1787/2537/2956/2957/3791/3792）。
  - #5/#6 → `coreOp.getWeightDByteOffset()` / `getWeightDminByteOffset()`（getter 已在）——**先确认前门对该 no-min 超块 stamp 了 80/82**（若未 stamp，此 2 处降入阶段 2）。
  - #2/#4 → `getNumLanes()`（getter 已在）或派生 `numSubBlocks/2`（@3387 有先例）。
  - #1/#3 → 派生 `subBlock/4`。
- **扇出**：6 决策（q3_K/q4_K/q5_K/q6_K）。
- **公式占比增量**：g 轴 `6/37... → +6`；**A 线总 `18/37 → 24/37`**。
- **守门**：byte-exact K-quant golden（q2_K…q6_K）逐字节。

### 阶段 2 — g 轴 iq-grid/codebook 13（描述符新字段）· **中风险 · ★回门（新 schema）**
- **触碰集**：`RVVOps.td`（新 `OptionalAttr` × N）+ `RVVLowerQuantContraction.cpp`（grid/codebook decode-facts 结构新字段 + stamp）+ `RVVToEmitCKQuant.cpp`（#7–#19 换成 `coreOp.getXxx()`）。**跨 3 文件 · 跨层**（ODS + 前门 + emitter）——**唯一需先过 supervisor 的阶段**。
- **替代零-schema 路径**：#7–#19 大多可**从 `subBlock` 闭式派生**（见 §4 映射表右列）——若采纯派生则**零新 schema**、退回单文件（emitter），但派生跨格恒等须逐 super-block byte-exact 复核。**两条路 supervisor 二选一（见 §4）。**
- **扇出**：13 决策（iq1_s/iq2_xxs/iq2_xs/iq2_s…）。
- **公式占比增量**：**A 线总 `24/37 → 37/37`**（确凿 scope 全清）。
- **守门**：byte-exact iq* golden 逐字节 · 新 attr 走 optional（不破 fixture）· 棘轮不动。

### 阶段 3（可选 · 后续）— 相邻同轴群 8
- BQL `coreLmul="<lit>"` 焊死 5（ISSUE-031(a) 假旋钮：须先拆焊死 vtype，风险更高）+ KQuant value_or 3。**不并入起手**。

**阶段序总结**：0（c-18，零 schema，起手）→ 1（g 平 K-quant 6，零 schema）→ 2（g grid 13，回门）→ 3（相邻 8，可选）。**每阶段单独 byte-exact 可验、可独立 commit、可停在任一阶段边界。**

## §4 ★ 拟新增 schema 字段 + 映射表（唯一回门点 · supervisor 扫一眼再施工）

**仅阶段 2 涉及。** 两条路，supervisor 二选一：

### 路 B（描述符字段 · 显式 provenance · 对齐架构「格式描述符」目标）
在 grid/codebook loop-body / core op（`RVVOps.td`）新增 **`OptionalAttr<I64Attr>`**，由前门 grid/codebook decode-facts 结构 stamp（照 `weight_scales_high_byte_offset` 既有新增先例）：

| 拟新增 ODS 字段（`RVVOps.td`） | 类型 | 承接的 g 轴字面量（file:line 值） | 零-schema 派生替代（路 A） |
|---|---|---|---|
| `groups_per_sub` | OptionalAttr<I64Attr> | 4202=4 · 4415=4 · 5567=4 | `subBlock/8` |
| `num_groups` | OptionalAttr<I64Attr> | 4620=4 · 4867=4 · 5109=4 | `subBlock/8` |
| `indices_per_sub_block` | OptionalAttr<I64Attr> | 4621=8 · 4868=8 | `subBlock/4` |
| `signs_per_sub_block` | OptionalAttr<I64Attr> | 4869=4 | `subBlock/8` |
| `num_groups_per_half` | OptionalAttr<I64Attr> | 5338=2 · 5568=2 | `halfLanes/groupLanes` (16/8) |
| `group_lanes` | OptionalAttr<I64Attr> | 4622=8 · 4870=8 | `subBlock/4` |

- **6 个新字段 · 覆盖 13 处字面量。** 全 **optional**（不改现有 op 合法性、不破 parser/负例 fixture）。
- **路 A（零新 schema）**：全部用右列派生表达式，**不动 ODS**，触碰集退回 emitter 单文件；代价 = 派生跨 super-block 恒等须逐格 byte-exact 证（本侦察未逐格证）。
- **c 轴若要把 `integer_core_lmul` 由 optional 改 required**（更强不变量）= **另一回门项**（须先扫全 `.mlir` fixture 确认无漏 stamp）；**阶段 0 默认不做此改**（fail-closed 读已足够 byte-exact）。

> **supervisor 决策项**：(1) 阶段 2 走路 A（零 schema/派生）还是路 B（6 新 optional attr/描述符）；(2) c 轴 `integer_core_lmul` 是否升 required。**其余阶段（0/1）零新 schema，无回门。**

## Caveats

- 阶段 1 #5/#6 的 getter 复用**取决于前门 stamp**——implement 首步须读前门确认 80/82 已 stamp，否则并入阶段 2。
- 公式占比分母用**确凿 37**（19+18）；若把相邻 8 计入则分母 45。**报数须带 scope（37 确凿 / 45 含相邻）**，禁两数互替。
- 全程**零 build / 零板**（侦察纪律）；byte-exact / VLEN 矩阵 / 棘轮守门在 implement 阶段执行。
