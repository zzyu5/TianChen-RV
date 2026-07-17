# Research: 「DeferredDequant 化」是什么 · 怎么把硬编决策改成它

- **Query**: RVVToEmitCDeferredDequant.cpp 的机制（宽度读自 IR·缺参 fail-closed），怎么把一处硬编改成 DeferredDequant 式
- **Scope**: 内部（lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp）
- **Date**: 2026-07-18

## 结论（一句）

`RVVToEmitCDeferredDequant.cpp`（1,070 行）是本仓「**有完整形状轴 ∧ 零决策住址**」的**唯一已存在范本**（ISSUE-033 域内存在性证明）。它的机制 = **宽度全部读自 IR 向量类型的 `getLmul()` / `getDType()`，从不 `.value_or` 补默认、从不 `StringSwitch` 按 format 名分派；遇到非法/缺失一律 `notifyMatchFailure` 直接 fail-closed**。"DeferredDequant 化" = 把一处「宽度字面量 / `.value_or` 补值」的硬编决策，改写成「从 IR 类型读宽度 + 缺失即 fail」的这个形态。

## 关键谓词（机核复验 · ISSUE-033 引用的三个 0）
- `grep -c 'value_or' lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp` = **0**
- `grep -c 'switch' ...` = **0**
- `grep -c 'StringSwitch' ...` = **0**
- 却接受 m1/m2/m4/m8 完整形状轴 —— 因为宽度读自 IR，不靠字符串属性补值。

## 机制解剖（实读代码）

### (1) 宽度读自 IR 向量类型
每个 emit 方法签名接收 `weftrvv::VectorType accVecType`（等 IR 类型），所有 LMUL/SEW/DType 从它派生：
- `srcVecType.getLmul()`、`accVecType.getLmul()`（L82–83）
- `vectorDType(accVecType)`、`vectorElementWidth(accVecType)`、`accVecType.getLmul()`（L377–379, L493–495）
- intrinsic 名由这些派生量拼装：`riscvVsetvlmaxIntrinsicName(accSEW, accLmul)`（L390–391）、`riscvWideningReductionIntrinsicName(*mnemonic, vectorDType(srcVecType), srcVecType.getLmul(), vectorDType(accVecType))`（L81–83）。
- ∴ 宽度是**从 IR 类型算出来的**，不是常量、不是 `.value_or` 补的。

### (2) 缺参 fail-closed（每个门都是 hard bail）
不是「缺了补默认」，而是「缺了/不匹配就 `notifyMatchFailure`」：
- `if (!srcVecType) return rewriter.notifyMatchFailure(reduce, "dequant reduce input not a vector");`（L67–69）
- `if (!mnemonic) return ... "unsupported dequant reduce kind";`（L72–74）
- `if (!accEmitC || !convertVectorTypeToEmitC(srcVecType)) return ... "dequant reduce type not convertible";`（L76–78）
- **形状硬门**（宽度不匹配即拒，不兜底）：`if (!accVecType || accVecType.getLmul() != "m1") return ...;`（L158, L349 类似 `!= "m8"`, L359 `!= "m1"`）—— 这就是「缺参 fail-closed 不兜底」的活写法。
- 未知 op 也 fail-closed：`else return rewriter.notifyMatchFailure(productOp, "unsupported dequant product op");`（L55–58）；`// Any unexpected op ... `（L140）。

### (3) 宽度来源 = with_vl / setvl 的 IR 类型
方法从 `weftrvv::WithVLOp scope`（L100）、`loadVL` / `sliceVL`（runtime VL = `setvl(...)`）取宽度上下文；VLMAX 用 `riscvVsetvlmaxIntrinsicName(accSEW, accLmul)` 拼装（L390），accSEW/accLmul 全来自 accVecType。∴ 宽度语义整链住在 IR 类型系统里，不在字符串属性 / 常量里。

## 「怎么把一处硬编改成 DeferredDequant 式」（范本迁移法）

以 ISSUE-033② 的 `.value_or("mf2")` 为例（BQL 18 处）：
1. **上游先显式化（前置，非发射器侧）**：前门 `RVVLowerQuantContraction.cpp` 的 `isM1 ? getStringAttr("m1") : StringAttr()`（沉默表达"不用 m1"）改成**显式盖章** core LMUL（把 "mf2" 也说出来）。缺席不再是信号。
2. **发射器侧改读 IR / required attr**：把 `getIntegerCoreLmul().value_or("mf2")` 改成：从对应 IR 向量类型 `getLmul()` 读（DeferredDequant 式），或把 attr 改 required + 缺失 `notifyMatchFailure`（fail-closed）。
3. **二点阶梯设防（ISSUE-033③）**：宽度只支持 `{mf2,m1}` 的静默算窄，改成显式 assert / fail-closed 覆盖第三档。
4. **byte-exact 门**：每处改完后**发射的 C 逐字节不变**（今日默认路径就是 mf2/m1，改的是"决策住哪"不是"发什么"）——这是 §四.2/四.3 的机检工具（byte-exact 在此是干活工具，不是主张）。

★关键顺序约束（ISSUE-033 已定）：**必须先做第 1 步（前门显式化）再做第 2 步**。若直接删发射器 `.value_or` 而上游仍沉默 → 直接 fail-closed 全表（破坏今日行为）。这决定了 ② 是「工，但有前置」。

## GridCodebook 焊死（④）的迁移
L520 / L1041 的 `coreLmul = "m1"` 字面量 → 改读 `cx.coreLmul`（同 TU 共享体 L59 已经是这样读的），或读 IR 类型。属纯工（无前门前置），但「假旋钮抓不住」这事本身是 ISSUE-031 判据丙论据（立门层面 gated）。

## 出处 / 谓词
- `RVVToEmitCDeferredDequant.cpp` L30–503（实读）；三个 0：`grep -c 'value_or|switch|StringSwitch'`
- ISSUE-033 域内存在性证明：`.trellis/spec/issues/发射器与架构.md`
