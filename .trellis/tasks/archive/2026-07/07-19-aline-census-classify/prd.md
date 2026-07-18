# PRD — A线 律2「点不入源」重构: HEAD-准确 census + 逐常数分类 + 映射表

## 性质
**Read-only 侦察/分类·零代码改**（trellis-research·只写 research/）。产出 A线重构的**准确当前残量图 + 逐常数分类 + 映射表提案**，供主会话施工（direct-lift 直改）+ 供 supervisor 扫映射表（新增 schema 字段的回门）。

## 背景（用户 A线 directive）
「把绕层重灾区接进已有层，DeferredDequant 是目标形态，骨架不推倒。**g 轴→格式描述符、c 轴→能力事实，抬反误伤证据**。」
- 律2 = 点不入源：**具体格式值（g 轴）或具体板值（c 轴）出现在发射体/机制体代码内即违反**（而非参数孔 `cx.getXxx()` / IR 类型读值 `getLmul()`）。
- **合规目标形态**：`RVVToEmitCDeferredDequant.cpp`（两轴皆干净·板宽读自 IR 类型 `getLmul()`×21·fail-closed「类型不带板值就不发射·绝不自补」）+ `RVVToEmitCBlockQuantLinear.cpp`（g 轴已干净·块结构从 `qk` 参数孔派生·板轴用 `getIntegerCoreLmul()` 旋钮）。

## ★关键：pinned census 已 stale（HEAD 已推进）
paper 03（pin `0aee07b4`）说 g轴37/c轴38。但**主会话 grep HEAD** 显示残量大降（阶段0 已清 BQL value_or 18·阶段2 加描述符 attr）。**本任务重跑 census 在当前 HEAD**（不是 pin·不 checkout），拿**准确现残量**。主会话初步（窄谓词）：
- KQuant g轴 named-constant = **2**（`weightDOffset=80`/`weightDminOffset=82` @ RVVToEmitCKQuant.cpp:4607-8）。
- c轴 `value_or("mX")` 板值自补 = **5**（ForwardElementwise 2 + KQuant 3）。
- `coreLmul="<lit>"` 焊死裸字面量 = **11**（GridCodebook 等·B2·配假旋钮 vtype·ISSUE-031(a)）。
> ⚠ 窄谓词可能漏（多变量逗号声明、inline magic number、size_t/auto、函数实参里的字面量）。**用宽谓词穷尽**·别凭窄谓词断言残量（记忆铁律 no-head-truncation-for-absence）。

## 任务
### ① HEAD-准确穷尽 census（宽谓词·所有形态）
在**当前 HEAD**（非 pin）跑，逐文件逐处列 A线残量（RVV 低精度 EmitC 家族：KQuant/GridCodebook/TernaryBinary/ForwardElementwise/BQL 板值轴 + 相关）：
- g 轴：裸格式结构字面量（offset/stride/lanes/planes/strip/subblock/block/bytes = <int>·含多变量逗号声明·含 inline magic number·排 0/1）。
- c 轴：`value_or("mX")` 板值自补 + `coreLmul/stripLmul="<lit>"` 焊死 + Internal.h 板阶梯 + 配套字面量 vtype（`OpaqueType::get(ctx,"vintNmM_t")` 独立于 coreLmul = 假旋钮）。

### ② 逐常数分类（B/C/D·类比 c 轴可写性清单）
每处标：
- **直改（direct-lift·无 schema·无回门）**：可从**现有** `getQk()`/`getLmul()`/现有 op attr / IR 类型派生（如 `halfBlock=qk/2`）→ 主会话可直接 byte-exact 改。给出**确切派生式**。
- **描述符（descriptor-attr·新增 schema 字段·必回门）**：布局偏移非现有 attr 可派生（如 K-quant 打包偏移 80/82）→ 须新增 OptionalAttr 描述符（参照阶段2 路B 描述符 groups_per_sub 等·fail-closed 缺席硬失败）。**列进映射表**（见③）。
- **假旋钮修（false-knob-fix·抬反误伤）**：coreLmul 焊死 + 独立字面量 vtype（改 coreLmul 不改 vtype = ISSUE-031(a)「设 m2 发类型不匹配 C」）→ lift coreLmul 到能力事实的**同时** vtype 须从 coreLmul 派生（`vint8m1_t` ← f(coreLmul)）。标注派生策略。**不得只 relabel 不修 vtype**（那是误伤证据）。

### ③ 映射表提案（新增 schema 字段·供 supervisor 回门）
对所有「描述符」类：出表 `{常数名·现值·所属 format·拟新增 attr 名（ASCII·^[a-zA-Z0-9_.-]{1,64}$）·attr 类型·前门 stamp 来源·fail-closed 诊断串}`。**Chinese 只放 description 不放 key**（记忆 workflow-schema-ascii-keys）。

### ④ byte-exact + 抬反误伤策略
- lift 是**纯 byte-exact 重构**（发出的 C 逐字节不变·只是常数来源从字面量变参数孔/描述符）·门 = BEFORE/AFTER emitc 输出 diff 为空。
- 「抬反误伤证据」= lift 后律2 机检命中数降·但**发出的核 byte-exact 不变**（不改任何可测行为）。标注每处 lift 的 byte-exact 验证法。

## 交付（写 research/·结构化）
1. HEAD-准确 census 表（逐文件逐处·宽谓词·当前行号）+ 与 pin `0aee07b4` 的差（哪些已被前序工作清）。
2. 逐常数分类（直改 / 描述符 / 假旋钮修）+ 每处确切派生式或 attr 提案。
3. 映射表（描述符类·ASCII key·供回门）。
4. 施工排序建议（direct-lift 先·byte-exact·再回门后的描述符类）。
5. **不改代码·不 commit·不下施工承诺**（本任务产计划 + 映射表·施工是后续 task）。

## 硬约束
- 全程 **read-only**·grep/read only·写 research/。
- census 在**当前 HEAD**（`git grep HEAD --` 或工作树·**非** checkout pin）。
- 宽谓词穷尽·别凭窄谓词断言「只剩 N」（附谓词让主会话可复现）。

## 参照
- paper 03（pin census·口径正本）：`/home/kingdom/phdworks/papers/share/03-B-DEBAKE-律2-75处违反.md`。
- 合规范本：`lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp`（getLmul fail-closed）+ `RVVToEmitCBlockQuantLinear.cpp`（qk 派生·getIntegerCoreLmul 旋钮）。
- 阶段2 路B 描述符范例：`include/Weft/Dialect/RVV/IR/RVVOps.td`（OptionalAttr groups_per_sub 等·缺席硬失败）。
- ISSUE-031（假旋钮·判据未定）· ISSUE-117（c轴 core_lmul = measured 决策·[GAP-P1] 锁·**注：本任务是律2 点入源重构·非写 perf 公式**·两者别混：lift coreLmul 到能力事实 ≠ 写 widest-legal 公式）。
