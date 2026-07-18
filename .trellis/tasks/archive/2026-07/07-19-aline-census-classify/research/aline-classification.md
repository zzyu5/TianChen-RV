# Research: A线 律2 逐常数分类（直改 / 描述符 / 假旋钮修）+ 施工排序 + byte-exact 策略

- **Query**: 每处残量标 直改/描述符/假旋钮修·给派生式或 attr 提案·施工排序·byte-exact 验证法
- **Scope**: internal（分类·零改）
- **Date**: 2026-07-19
- 数据源 = 同目录 `aline-census-head.md`（HEAD 行号 + issue cross-ref）

> **总纲结论**：HEAD 上**几乎没有「主会话可直接 byte-exact 直改·无回门」的施工**。残量按裁决状态分四堆：**(A) 已裁 MAINTAIN·不动**（真焊死假旋钮 + 结构常量 + verifier-pinned）；**(B) 描述符类·须回门**（KQuant fold-brick / GridCodebook·Ternary grid-count / ForwardElementwise 布局）；**(C) c 轴 blocked**（value_or·ISSUE-113/116·前门 stamp）；**(D) 少量真直改候选·须先验 op attr**。

---

## §1 三桶分类总表

### 桶 D — 直改（direct-lift·无 schema·无回门）

**候选极少·且都带前置验证**：

| 处 | 现值 | 拟直改 | 确切派生式 | 前置 |
|---|---|---|---|---|
| FwdEW 2709 `qk=32` | 32 | 读 `deqOp.getQk()` | `qk = deqOp.getQk()` | **须先验** `GgmlDequantizeRowOp` 在 nibble path 带 `qk` attr（§census §4.1）；若 per-format 函数结构固定则价值仅去重 |
| FwdEW 4564 `qk=256` | 256 | 读 `deqOp.getQk()` | 同上 | 同上 |
| FwdEW 2975 `qk=32`(half=qk/2) | 32 | 读 `getQk()` | `half=getQk()/2`（half 已派生·只 qk 烘焙） | 同上 |

> **诚实**：这 3 处即便直改也只是「去重字面量·qk 本就 = getQk()」，**不是**主会话预期的高价值直改。KQuant 阶段2 已把真正的直改面（grid-count → 描述符）做完。**若 op 该 path 无 qk attr → 这 3 处退化为「描述符或 MAINTAIN」，直改桶归零。**

### 桶 B — 描述符（descriptor-attr·新增 schema·必回门）

见 §2 与 `aline-descriptor-mapping-table.md`。三簇：
1. **KQuant weightDOffset=80/weightDminOffset=82**（ISSUE-115）——**特例**：不能同法加 attr（ODS charter 冲突）·须**专用 fold-brick op**·待裁。
2. **GridCodebook.cpp(9)+TernaryBinary.cpp(8) grid 几何**——套阶段2路B（新 OptionalAttr per op·前门 stamp·fail-closed）·未登记。
3. **ForwardElementwise dequant-row 布局(~40)**——monolith-fallback 架构·大改·建议独立 scoping。

### 桶 A — MAINTAIN（已裁·[K-10]/结构本性/verifier-pinned·不动）

| 处 | 病名 | 裁决依据 |
|---|---|---|
| GridCodebook 497/995（+wideLmul 498/996） | 真焊死 coreLmul+字面量 vtype（假旋钮形态） | ISSUE-031(a)：IQ3XXS/IQ3SGridBodyContext **无 coreLmul 字段**·iq3 几何固定 8-lane grid-of-4·**无 IR 宽度可读**·强加字段=造假旋钮=[K-10] 违例 |
| Ternary 862（+wideLmul 863） | 同上 | IQ1MGridBodyContext 无 coreLmul 字段·同 iq3 逻辑 |
| BQL 111/404（+wideLmul 112/405） | 真焊死 coreLmul | stage-3 [K-10] 注释明标「structural constant NOT a knob·no IR width·debake=no-op」·commit `3f67750da` 0 debaked/8 maintained·ISSUE-116 二结构常量 |
| CodebookFp4 109 | 焊死 m1 但 vtype 派生 | verifier-pinned（gather 需 VLMAX≥16 索引 16 表项）·结构固定·非假旋钮 |
| Internal.h 3634-3637 板阶梯 | c 轴板默认 | 「default reproduces byte-identical legacy」·随 ISSUE-113 处理 |

**⚠ 假旋钮修桶（PRD 第三类）在 HEAD = 空**：所有「焊死 coreLmul + 独立字面量 vtype」= 上表 GridCodebook/Ternary/BQL·**全部无上游能力事实可 lift**（Context 无字段/无 IR 宽度）→ **lift = 造假旋钮 = [K-10] 违例**。故 PRD「lift coreLmul 到能力事实同时 vtype 从 coreLmul 派生」**没有合法施工对象**：没有能力事实可 lift。**这正是 ISSUE-031(a) 订正 + stage-3 0/8-debaked 的结论·本 census 独立复验命中同一结论。**

### 桶 C — c 轴 blocked（value_or·前门 stamp·ISSUE-113/116）

| 处 | 裁决 |
|---|---|
| KQuant 2918/3694/5971 value_or | ISSUE-116：decode/flat 前门**故意不 stamp** integer_core_lmul·fail-closed 现在 = 23 byte-exact 回归·须先前门 stamp（回门）·保守默认维持 |
| BQL 14232/14315/17663 default-override | 同 ISSUE-116（旧行 14221/14300/17644） |
| Ternary 1607/2062 default-override | live-default+override·vtype 已派生·**已合规**（compliant knob）·m2 默认 = VLEN-safe floor·gearbox 精化·**非残量·可不动** |
| FwdEW 169/521 `getStripLmul().value_or("m8")` | c 轴 value_or·strip 轴·ISSUE-113 backlog |

---

## §2 描述符提案要点（详表见 mapping-table 文件）

- **范式（阶段2路B·已证 byte-exact）**：新 `OptionalAttr<I64Attr>` per op（保 fixture 解析）+ 发射体改 `coreOp.getXxx()` **fail-closed**（缺席 `notifyMatchFailure` 点名·**严禁 `.value_or` 复制到 g 轴**·阶段2 硬条件）+ 前门 `RVVMonolithicBlockDotSourceFrontDoor.cpp` stamp + I7 verifier allowlist 只加 new attr。
- **KQuant fold-brick 特例**：weightDOffset/dmin **不加 attr**·做独立 fold-brick op（core-op 语义级·待用户裁·ISSUE-115）。
- **ForwardElementwise 特例**：monolith-fallback-keyed-by-format·转描述符 = 前门须为 `GgmlDequantizeRowOp` stamp 全格布局·大改·建议独立 issue 先 scoping。

---

## §3 施工排序建议（direct-lift 先·但 HEAD 上主要是回门）

**现实排序**（因直改桶近空·实际是回门优先级）：

1. **先验证桶 D 前置**（op attr 存在性·§census §4）→ 若成立做 3 处 qk 直改（byte-exact·无回门·最省）。
2. **GridCodebook/Ternary grid-count 描述符**（桶 B-2·套阶段2路B 成熟范式·byte-exact·回门加 ~17 OptionalAttr）——**扇出最高的真施工面**·与阶段2 同法·风险最低的回门。
3. **KQuant fold-brick op**（桶 B-1·ISSUE-115·须用户先裁 core-op 语义）——阻塞·不自决。
4. **ISSUE-116 前门 stamp**（桶 C·6 处·cross-layer 前门+emitter·须裁）——阻塞。
5. **ISSUE-113 c 轴 required 升级**（收尾锁·触发窗）——阻塞。
6. **ForwardElementwise dequant-row 布局**（桶 B-3·大改·独立 scoping）——最后·或不纳入本轮。
7. **桶 A 全部不动**（MAINTAIN·[K-10]）。

---

## §4 byte-exact + 抬反误伤策略（每类验证法）

**通用门**：lift 是纯 byte-exact 重构·**门 = BEFORE/AFTER emitc 输出 diff 为空**（regen-diff 0）。阶段2 已建此门（「production grid+q3/q4/q5/q6_K×VLEN128/256·monolithic q4_K byte-identical」）。

| 类 | byte-exact 验证法 | 抬反误伤要点 |
|---|---|---|
| 桶 D 直改（qk） | 改前后跑 dequant-row fixture regen·emitc diff 空（qk 值不变·只来源变） | 律2 机检命中 -3·发核 byte-exact 不变 |
| 桶 B-2 grid 描述符 | 同阶段2：production grid fixture + direct-input fixture 补 stamp（fail-closed working as designed）·regen-diff 0·I7 allowlist 只加 new attr | **stamp 值必 = 现字面量**（byte-exact）·前门缺 stamp → 硬失败点名（非 value_or 自补·阶段2 红线：严禁 value_or 病复制到 g 轴） |
| 桶 B-1 fold-brick | 新 op·须证 fp32 fold 语义等价 + regen-diff 0 | 语义级·超 byte-exact·待裁 |
| 桶 C 前门 stamp | ISSUE-116：前门用 **current default 值** stamp（byte-exact）→ 再 fail-closed·否则 23 回归 | **误伤铁证**：现在直接 fail-closed（不先前门 stamp）= 23 byte-exact 回归（rc 0→1）= 破真管线·**这就是「抬反误伤」的反面教材**（改了律2 机检数但破了可测行为） |
| **假旋钮修（空桶）** | N/A | **PRD 警告「不得只 relabel 不修 vtype」在此无适用对象**：无能力事实可 lift → 不 relabel·维持字面量。若强行 lift coreLmul 而 vtype 仍独立字面量 → 正是 ISSUE-031(a)「设 m2 发类型不匹配 C」的误伤·**故 MAINTAIN 是唯一 byte-exact 正确动作** |

**「抬反误伤证据」定义复述**：lift 后律2 机检命中数降·但发出的核 byte-exact 不变。**桶 B-2（grid 描述符）是唯一能真「抬反误伤证据」的施工面**（机检 -17·byte-exact 0）；桶 A 不能（无 lift）；桶 C 现在 lift 会真破 byte-exact（须前门先 stamp）。

---

## Caveats

- 桶 D 的 3 处 qk 直改**全部 gated on** `GgmlDequantizeRowOp` 是否带 qk attr——未核·若无则直改桶归零。
- 桶 B-2 的「17 处可套阶段2路B」是**类比推断**（同常量类·同 format-defined 性质）·未逐一验证 GridCodebook/Ternary helper call-site 是否持 op 可填 Context——须施工前核（§census §4.2）。
