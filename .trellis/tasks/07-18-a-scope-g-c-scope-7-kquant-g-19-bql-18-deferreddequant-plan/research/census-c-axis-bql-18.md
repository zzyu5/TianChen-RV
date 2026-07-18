# Research: A 线 c 轴 —— BQL 板值烘焙 18 逐处表

- **Query**: 定位 `RVVToEmitCBlockQuantLinear.cpp` 里按板值烘焙（value_or 板默认）的 18 处，逐处给目标路由（能力事实层）+ 难度 + byte-exact 风险
- **Scope**: internal（代码普查·机算可复跑）
- **Date**: 2026-07-18

## 机算复跑（对齐 03 census · pin `0aee07b4`）

**权威谓词**（03-B-DEBAKE §④ B1）：
```
git grep -nE '\.value_or\("(m1|m2|m4|m8|mf2|mf4|mf8)"\)' 0aee07b4 -- lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp
```
- 本轮实跑 @ `0aee07b4` = **18**（逐行号与 03 相同：2198,2297,2408,2592,2724,2947,3324,3426,3544,3712,3904,3981,4433,4946,5478,5893,7636,7686）。
- **HEAD `c84820c4` 复跑 = 18**（scope 未漂）。

## ★ 关键状态事实 —— c 轴生产侧「显式化」已在 HEAD 达成（决定风险等级）

pin 之后的 commit **`0b18a3daf`「R线 §四.3 #2 Step1: 前门 18 处显式化(silent-signal 18→0·byte-exact)」**（在 HEAD 历史内；pin `0aee07b4` 是其祖先）已把**前门生产侧**改成**总是显式 stamp** `integer_core_lmul`：

`RVVLowerQuantContraction.cpp`（HEAD :1777-1778）：
```cpp
mlir::StringAttr integerCoreLmul =
    isM1 ? builder.getStringAttr("m1") : builder.getStringAttr("mf2");
```
注释（HEAD :1738/:1759）自述：*「the width decision is spoken outright so the attr can become required」*、*「RVV1.0 stamps integer_core_lmul EXPLICITLY as "mf2" (ISSUE-033: no silent signal)」*。

⟹ **前门对所有 WIRED leaf（minVLEN≥128，`stampScheduleSelections` fallback 断言保证）总是显式 stamp `integer_core_lmul`**。因此 BQL emitter 里 18 处 `.value_or("mf2")` 的**「缺席自补」分支已对所有生产输入不可达（dead branch）**。删掉 default、改 fail-closed 读，**byte-exact by construction**（生产侧从不留空）。

> **这是 c 轴 A 线的最大去风险点**：03 census 把 `.value_or` 记为「承重」（ISSUE-033「缺席当信号」），但那是 pin 时刻状态；**HEAD 上生产侧已显式化（ISSUE-033 Step1 done）**，emitter 的 default 已是死码。A 线 c 轴 = 消费侧收尾（把死码 default 换成 DeferredDequant 式 fail-closed 读）。

## 目标形态（能力事实层 · DeferredDequant 式）

`integer_core_lmul` 是 **`OptionalAttr<StrAttr>`**（`RVVOps.td:4207/4605`）；配对的 `half_lanes` 是**必填 `I64Attr`**，**已由能力闭式** `deriveRepackHalfLanes(minVLEN)`（`RVVLowerQuantContraction.cpp:1222` · `half_lanes = min(vlen/16,16)`，128→8/256→16/<128→0）在前门算出并 stamp。verifier 对 `(integer_core_lmul, half_lanes)` 合法组合 **fail-closed 钉死（I7）**（`RVVOps.td:4580-4581`）。

对照正例 **`RVVToEmitCDeferredDequant.cpp`（两轴干净）**：板宽**读自 IR 向量类型** `getLmul()` × 21，且 **fail-closed**（`:158 if (!accVecType || accVecType.getLmul() != "m1")`、`:350 != "m8"`），**0 处 `value_or` 板默认 · 0 处裸格式字面量**。目标 = BQL 消费侧向此靠。

## c 轴 18 逐处表

18 处形态**同构**（全是 `<op>.getIntegerCoreLmul().value_or("mf2")`），按承载 op 与角色分列：

| # | file:line | 板值默认 | 承载 op / 角色 | 现烘焙形态 | 目标路由（能力事实） | 难度 | byte-exact 风险 |
|---|---|---|---|---|---|---|---|
| 1 | :2198 | `mf2` | `loopBody`（ternary repack GEVM） | 缺席自补板值 | fail-closed 读：`getIntegerCoreLmul()` 缺则 `notifyMatchFailure`（前门恒 stamp，分支死码） | 低 | 极低（死码删除） |
| 2 | :2297 | `mf2` | `loopBody`（repack GEVM 体） | 同 | 同 | 低 | 极低 |
| 3 | :2408 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 4 | :2592 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 5 | :2724 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 6 | :2947 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 7 | :3324 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 8 | :3426 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 9 | :3544 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 10 | :3712 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 11 | :3904 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 12 | :3981 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 13 | :4433 | `mf2` | `gemv`（repack GEMV） | 同 | 同 | 低 | 极低 |
| 14 | :4946 | `mf2` | `gemv` | 同 | 同 | 低 | 极低 |
| 15 | :5478 | `mf2` | `gemv` | 同 | 同 | 低 | 极低 |
| 16 | :5893 | `mf2` | `gemv` | 同 | 同 | 低 | 极低 |
| 17 | :7636 | `mf2` | `loopBody` | 同 | 同 | 低 | 极低 |
| 18 | :7686 | `mf2` | `gemm`（repack GEMM） | 同 | 同 | 低 | 极低 |

**统一路由**：18 处全部 → 单一 fail-closed helper（读 `getIntegerCoreLmul()`，缺则 `notifyMatchFailure`）。**因前门已恒 stamp（0b18a3da），去掉 default 对所有生产输入 byte-exact。**

## 相邻 c 轴群（PRD scope=18，但同轴·同去风险逻辑·可并同阶段或后续阶段）

- **KQuant value_or c 轴 3 处**（03 §④ B1 KQuant 部分）：`RVVToEmitCKQuant.cpp:2217 mf2`、`:2964 mf2`、`:5110 m2`（`coreOp.getIntegerCoreLmul().value_or(...)`）。同去风险逻辑（前门恒 stamp）→ 同 fail-closed 路由。
- **BQL `coreLmul="<lit>"` 焊死 5 处**（03 §④ B2 · ISSUE-031(a) 假旋钮）：`:107 m1`、`:397 m1`、`:14113 m2`、`:14192 m1`、`:17536 m2`。**与 value_or 不同**：这是**无 getter、无参数孔的纯焊死** + 相邻焊死 vtype（假旋钮：改 `coreLmul` 不改发出的 vtype）。路由到能力事实**难度更高**（须先证该 emit 体的板宽应从 IR/能力来，且拆假旋钮 vtype），风险更高——**建议单列后续阶段，不并入 18 的低风险起手**。

## Caveats / Not Found

- 18 处 fail-closed 化的**唯一残余风险** = **手写测试/parser fixture 里的 op 若不 stamp `integer_core_lmul`**：改 ODS 为 required 会破这些负例/parser 测试。⟹ **安全变换 = emitter 内 fail-closed 读（`notifyMatchFailure`），不动 ODS optional 性**；若要把 ODS 改 required（更强不变量），须先扫全 `.mlir` fixture —— 归**回门项**（见 debake-staged-plan.md §4，schema 变更）。
- 「前门对所有 WIRED leaf 恒 stamp」由 `RVVLowerQuantContraction.cpp:1450` 的 fallback 断言（`minVLEN < 128` 不达 stampScheduleSelections）+ 1817-1818 `if (integerCoreLmul) addAttribute` + 1777-1778 两分支均建非空 StringAttr **推得**；本侦察未逐 leaf 枚举证「无一 wired 路径漏 stamp」——implement 阶段应加一条断言/lit 覆盖坐实（DeferredDequant fail-closed 语义本身即兜底）。
