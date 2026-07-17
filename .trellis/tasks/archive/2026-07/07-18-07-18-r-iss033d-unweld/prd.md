# PRD · R线 · ISSUE-033④ GridCodebook 焊死宽度去除

> **权威** = 《开测篇》§四.3 决策上收（工·非 F-7 门本身）。地形依据 = 归档 task `07-18-r-stage2-recon` 的 `research/`。
> **F-7 门 gated on ISSUE-031**（判据级）—— 本 task 只做 ④「去焊死」这一件工·**不立 F-7·不销 PR-23**（登记 ISSUE-031 保守默认）。

## 一、目标（工·fork-independent·grid 扇出前置）

`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp` 的 `coreLmul="m1"` **焊死字面量**（research 定位 ~L520/L1041）——
ISSUE-033④「焊死宽度 + 假旋钮」。**改为宽度读自 IR 向量类型**（`RVVToEmitCDeferredDequant.cpp` 是 fail-closed 范本：
`accVecType.getLmul()` 等·缺参 `notifyMatchFailure`）。这是 grid dequant 扇出需要的**宽度读自 IR 纪律**
（trellis-check 已警 iq3_xxs body 硬编字面量不可盲拷扇出）。

## 二、★byte-exact 判据（关键·同去重）

去焊死若 **IR 宽度 == 焊死值(m1)** ⟹ **发射产物逐字节不变**（纯去假旋钮·byte-exact 门·[[build-incremental-unreliable]]）。
- **产物不变** = 成功（假旋钮拆除·行为零改·decision-density↓）。
- **产物变了** = 焊死值在掩盖真实宽度不匹配 ⟹ **不是纯去焊死·停·登记具名**（别硬改·那会改可测行为）。

## 三、方法

1. objdump/机检**改前**受影响 grid 格发射产物 md5（BEFORE·forced clean rebuild + 亲见 compile）。
2. L520/L1041 `coreLmul="m1"` 字面量 → 读自 IR 类型（fail-closed·缺参 notifyMatchFailure·范本 DeferredDequant）。
3. **改后** md5（AFTER）。**BEFORE==AFTER 才通过**·否则回退具名。
4. lit 全绿（除 unrelated 3 失·stash-rerun 证）。

## 四、验收

1. 2 处焊死去除·**受影响格发射产物 md5 前后不变**（亲见 BEFORE/AFTER）· 缺参 fail-closed（非兜底默认）。
2. lit 全绿（除 unrelated 3）· decision-density（该 TU `value_or`/字面量宽度计数）↓ 记一行。
3. **不立 F-7·不销 PR-23**（gated on ISSUE-031·登记保守默认）。
4. **0 造数**·sealed 不动·**禁 commit·禁 add -A**·"某物不存在"禁截断命令·三禁区（FlatFold/GEMM-GEVM 镜像/decode leaf 数值逻辑）不碰数值行为。

## 五、遗留

- 若 ④ 顺（byte-exact），大头 = §四.3 #2（前门 18 处显式化→BQL/KQuant 20 处 DeferredDequant 化·顺序不可倒·先显式化）= 后续 task。
- F-7 门 / ISSUE-033① 倒挂 / PR-23 销案 = gated on ISSUE-031（判据级·待用户裁）。
- ★共享 TU（GridCodebook 跨线）·本 task 独占·无并行线。
