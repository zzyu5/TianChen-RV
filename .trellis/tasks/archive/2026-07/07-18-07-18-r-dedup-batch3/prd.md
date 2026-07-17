# PRD · R线 · §四.2 发射器阶段一去重 batch3

> **权威** = 《开测篇》§四.2「发射器阶段一·去重提取（可与测量并行的案头）」。停止条件 3 组件。
> **fork-independent**：不依赖 ISSUE-105/106/107 战略裁决·纯 byte-exact 案头·安全续推。

## 一、目标

≈13k 盖章复制行按 **md5 聚类** 逐批提取共享例程。**每批提取后发射产物逐字节不变**为机检门
（byte-exact 在此是**干活工具**·非产物守恒判据·区别 K 线攻坚）。前批已提 `emitLoadByteAsInt`/`emitSizeLit`。

## 二、★三禁区（不碰）

1. **5 棵浮点折叠树**（FlatFoldModel 结构）。
2. **GEMM↔GEVM 镜像**。
3. **每格 decode leaf**（含新落的 `emitDequantizeRowIQ3XXSVectorBody`·grid-codebook bodies）。

## 三、方法

1. **md5 聚类扫描**：跨 `lib/Conversion/RVV/*.cpp` 找逐字节/近字节重复的 `rewriter.create<emitc::...>` 序列簇
   （候选大户：BlockQuantLinear/KQuant/ForwardElementwise·避开三禁区）。挑**一个清晰簇**（≥3 处重复·非禁区）提取为共享 helper。
2. **提取**：共享例程住 `RVVToEmitCSupport.{h,cpp}`（或既有 helper 宿主·同前批）·detail namespace。
3. **★byte-exact 门**（[[build-incremental-unreliable]]）：提取前后 **clean/forced rebuild** + 亲见 compile·
   对**受影响格的发射产物**做 md5 BEFORE/AFTER 比对·**逐字节不变**才算通过（发射行为零改变）。
4. 一批一提取一验证·**产物变了 = 回退**（去重不许改任何可测行为）。

## 四、验收

1. 提取 ≥1 簇（≥3 处重复→1 helper）·**受影响格发射产物 md5 前后不变**（BEFORE/AFTER 亲见）。
2. lit 全绿（除既有 unrelated 3 失·stash-rerun 证）。
3. 手写 LOC 净降（Δ手写 LOC 记一行）·三禁区未碰。
4. **0 造数**·sealed 不动·**禁 commit·禁 add -A**·"某物不存在"禁截断命令。

## 五、遗留

- 阶段二决策上收（§四.3·F-7 门）+ [SEL-3] strip-width（§四.4·ISSUE-035）= 后续 task。
- 去重是多批·本 task 一批先证方法（同单格先证纪律）。
