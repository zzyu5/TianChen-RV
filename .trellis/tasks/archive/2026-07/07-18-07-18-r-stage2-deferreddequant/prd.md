# PRD · R线 · §四.3 决策上收 #2：前门显式化 → BQL/KQuant DeferredDequant 化

> **权威** = 《开测篇》§四.3。地形 = 归档 `07-18-r-stage2-recon` 的 research（尤其 01/02/03）+ [ISSUE-033]②③⑤。
> **性质** = 工（有范本 `RVVToEmitCDeferredDequant.cpp`·非 iq3 结构常量那种）。**F-7 门认证 gated on ISSUE-031**（判据级·**本 task 不立 F-7·不销 PR-23**·只做 decision-density 削减这件工）。

## 一、目标（消 silent 宽度默认·宽度读自 IR·fail-closed）

现状（ISSUE-033 病名）：前门 `RVVLowerQuantContraction.cpp` 对「非 m1」用**沉默**（空 StringAttr）表达，
发射器 `.value_or("mf2")` **× 18**（BQL）+ **× 2**（KQuant）**替一个没送达的决策补值**。
目标：① 前门**把「mf2」显式说出来**（缺席不再是信号）② 发射器改**读宽度自 IR 向量类型**
（`accVecType.getLmul()`·范本 DeferredDequant）· `.value_or`/`switch`/`StringSwitch` 归零 · 缺参 **fail-closed**
（`notifyMatchFailure`·非兜底默认）。

## 二、★顺序不可倒 + 判据

1. **先前门显式化**（`RVVLowerQuantContraction.cpp` 18 处 `isM1?getStringAttr("m1"):StringAttr()` →
   `isM1?"m1":"mf2"` 显式）——**否则第 2 步 fail-closed 会破行为**（研究铁证）。
2. **再发射器 fail-closed**（BQL 18 + KQuant 2 处 `.value_or("mf2")` → 读 IR 宽度·缺参 notifyMatchFailure）。
3. **★byte-exact 判据（于发射产物 C·非 IR）**：前门显式化后**最终发射的 C 逐字节不变**（显式 "mf2" == 默认 "mf2"）·
   [[build-incremental-unreliable]] forced rebuild + 亲见·md5 BEFORE/AFTER 对受影响格。**产物变了 = 有路径宽度≠假设 → 停·登记具名**（别硬改）。
4. **lit**：前门显式化会改 IR attr → **lit CHECK 若查该 attr 须同步更新**（IR 层·合法）。发射器 fail-closed 后
   **全 lit 绿**（除 unrelated 3·stash-rerun 证）。**任何格 fail-closed 转红 = 该格前门未显式化 → 补显式化或回退具名**。

## 三、验收

1. 前门 18 处显式化 · 发射器 20 处（BQL 18+KQuant 2）`.value_or` 归零 · 该 TU `value_or`/`switch`/`StringSwitch` 计数 → 0（机检·同 DeferredDequant 范本）。
2. **发射产物 C byte-exact**（受影响格 md5 前后不变·亲见）· lit 全绿（除 unrelated 3·IR-attr CHECK 已更新）。
3. decision-density（前门空-attr 信号数 + 发射器 value_or 数）↓ 记前后对照一行。
4. **不立 F-7·不销 PR-23**（gated ISSUE-031·登记保守默认·禁自裁）· ISSUE-033① 倒挂**不碰**（判据级）。
5. **0 造数**·sealed 不动·三禁区（FlatFold/GEMM-GEVM 镜像数值逻辑/decode leaf 数值）不碰·**禁 commit·禁 add -A**·"某物不存在"禁截断命令。

## 四、触碰集 / 护栏

- `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（前门 18 处）+ `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`（18）+ `RVVToEmitCKQuant.cpp`（2）+ 前门 lit（IR-attr CHECK）。
- **共享 TU·独占本 task·无并行线**（研究已警）。**顺序不可倒**（先显式化）。
- **revert-on-breakage**：任一步 byte-exact 破 or lit 转红且非 IR-attr 预期 → 回退该步·登记具名（这是发现·非失败）。

## 五、遗留

- F-7 门 / ISSUE-033① 倒挂消解 / PR-23 销案 = gated ISSUE-031（判据级·待用户裁）。
- [SEL-3] strip-width（§四.4·ISSUE-035）= 后续（BQL 宽度轴接入测量键·记忆超 authority·delicate）。
- 本 task 若因某格前门路径复杂无法干净显式化 → 该格如实登记具名·不硬改（部分完成合法）。
