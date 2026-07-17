# Journal · R线 §四.3 决策上收 #2 · 前门显式化 → BQL/KQuant DeferredDequant 化

**日期**：2026-07-18 · **性质**：工（有范本）· **权威**：《开测篇》§四.3 + ISSUE-033②③⑤

## 结果一句

**第 1 步（前门显式化）＝ 完成·clean**（byte-exact + lit 绿）。**第 2 步（发射器 fail-closed）＝ 受阻·未做·保守默认维持现状**——研究前提不完整，发射器 `.value_or("mf2")` 在真实测过的路径上是承重的，其 op 由**触碰集外**的 3 个产出者留空 attr（不止已显式化的前门一处），盲改会破 112 手写 fixture + 单体/超块真管线。**部分完成合法（PRD §五）**。

## 第 1 步（完成）

- `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`：`isM1 ? builder.getStringAttr("m1") : mlir::StringAttr()` **× 18 全部** → `... : builder.getStringAttr("mf2")`（`replace_all`·缺席不再当信号）。空-attr 信号 **18 → 0**。
- 3 处描述「RVV1.0 leaves integer_core_lmul unset」的注释同步改为「stamps EXPLICITLY "mf2"」。
- **byte-exact（于发射产物 C·亲见）**：harness = 18 个 repack/super-block 测试输入 × 3 march {rv64gcv, rv64gcv_zvl256b, rv64gc_xtheadvector}，跑 `weft-opt --weft-rvv-lower-quant-contraction=march=X --weft-rvv-lower-to-emitc`，md5 全量。**baseline vs step1 vs final = 54/54 全一致**（forced relink 亲见·binary mtime 变化验证）。
- **lit**：full suite **951/954 绿**·前后不变·3 unrelated = `Scripts/rvv-generated-bundle-abi-e2e{,-explicit-,-pre-realized-}*`（基线即红·与本改无关）。
- **IR-attr CHECK**：前门 IR 现含显式 `integer_core_lmul = "mf2"`（q4_0 repack gevm 实测 4 处），但**无 lit CHECK 需更新**（无 CHECK-NOT/CHECK-SAME 在这些 repack 路径上钉住该 attr 的缺席；唯一 `CHECK-NOT integer_core_lmul` 在 `rvv-q4-k-q8-k-block-dot-source-front-door.mlir` = 另一前门·未触碰·仍绿）。

## 第 2 步（受阻·未做）— 研究前提不完整的实证

发射器 20 处 `.value_or("mf2")`（BQL 18 + KQuant 2）**全部保留不动**。原因（机核+实测·非推测）：

1. **BQL 5 处 = 单体 op（触碰集外）**：gemv/gemm 变量上的 value_or（@4433 `GgmlRepackGemvQ50Q80Op` · @4946/5478/5893 其它单体 GEVM · @7686 `GgmlRepackGemmQ41Q81Op`）= q5_0/q4_1/q8_0/q4_K **单体 repack op**。RVV1.0 下经 `lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp`（[SEL-3]·L110-111 明文「RVV1.0 leaves integer_core_lmul unset」）**故意留空**。
2. **KQuant 2 处 = 超块砖（触碰集外前门）**：`scaledDot`@2217 / `b3`@2964 = q4_K 超块砖，由 `--weft-rvv-materialize-q4-k-q8-k-block-dot-source-front-door`（≠ `RVVLowerQuantContraction`）产出。**过 `--weft-materialize-emission-plans` 后 `grep -c integer_core_lmul` = 0**（实测·attr 真缺）。
3. **112 份直喂发射器手写 fixture**：`test/Conversion/RVV/rvv-to-emitc-*repack*loop-body*` 等，输入 loop body op **缺 integer_core_lmul**（`grep -c` = 0），**依赖 mf2 缺省**。

⟹ 盲改发射器 fail-closed（notifyMatchFailure）会破这 112 fixture + 单体/超块真管线 = §四.3「产物变/lit 转红 = 有路径宽度≠假设 → 停·登记具名」的实证。修正需**显式化 3 个触碰集外产出者**（`RVVMonolithicBlockDotSourceFrontDoor` + q4-k materialize + 保住 strip-width 传导）**+ 改 112 fixture 输入** = 全超本 task 触碰集。

「先显式化再 fail-closed」的**顺序前提仍成立**，纠正点 = 「上游」不止前门一处；后续须把上述 3 产出者一并显式化才能安全 fail-closed。

## 验收对照

| 项 | 目标 | 实际 |
|---|---|---|
| 前门空-attr 信号 | 18 → 0 | **18 → 0 ✓** |
| 发射器 value_or | 20 → 0 | **20 → 20**（受阻·保守默认·部分完成合法）|
| 该 TU switch/StringSwitch | 0 | 未触碰（未做 fail-closed）|
| 发射产物 C byte-exact | 不变 | **54/54 md5 不变·亲见 ✓** |
| lit | 全绿(除 unrelated 3) | **951/954·3 unrelated·前后不变 ✓** |
| decision-density | ↓ | 前门 silent-signal **18 → 0**；发射器 value_or 20 不变（净 ↓18）|

## 铁律确认

- **0 造数**·未跑任何测量数字。
- **三禁区未碰**：FlatFold / GEMM-GEVM 镜像数值逻辑 / decode leaf 数值——全程只动 attr 显式化 + 注释，零数值逻辑。
- **sealed 9/83 未动**。
- **不立 F-7·不销 PR-23·未碰 ISSUE-033① 倒挂**（判据级·gated ISSUE-031·未自裁）。
- **禁 commit / 禁 add -A**：仅 Edit/Write，无 git 写操作。
- 「某物不存在」类断言用 `grep -c` / 精确谓词，未用 head/tail 截断。

## 触碰文件

- `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（18 显式化 + 3 注释）
- `.trellis/spec/issues/发射器与架构.md`（ISSUE-033 登记施工进展 + 订正 16→18 计数）
- 本 journal
