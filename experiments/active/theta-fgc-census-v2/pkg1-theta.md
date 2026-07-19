# 包1 · θ 总清单（输出侧逐值裸档案）

**HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（`git rev-parse HEAD` 实测 == 此 pin；下文全部 file:line 直读工作树 == HEAD）

只列事实 + 出处。是非/分类建议归上游。全文 ref 省略写作 `<pin>`。

---

## 框架换算（本包内使用）
- g = 格式侧输入（块宽 qk / 交织宽 weight_interleave / 超块几何 …）
- c = 板侧输入（VLEN / 寄存器数 / RVV 版本〔分数 LMUL 有无〕…）
- θ = 输出/决定
- 四类：(a) 由某条 f 现场算出 · (b) 人盖章进描述符/前门·fail-closed 守着 · (c) 焊死在代码里 · (d) 由测量表选出

## 现成数据锚（先于源码）
- `.trellis/spec/issues/性能与测量.md:433`：「`RVVToEmitCBlockQuantLinear.cpp` 的 c 轴 fail-closed getter = 同一模式 **×8 处**（行 2205/2310/2427/2617/2755/2984/3367/3475/3599）」— **该行文字自身称「×8」但括号内列了 9 个行号**（机核见 θ1）。
- `experiments/active/result-tables/A线柱二-c轴f可写性清单.md:9`：同一轴称「**18 哨兵（schema）**」。
- `.trellis/spec/issues/发射器与架构.md:204`（ISSUE-118 census）：「coreLmul 焊死 11 经宽谓词+逐处核 = **5 真焊死（MAINTAIN·[K-10]）+ 5 live-default-override（value_or 类·vtype 已派生）+ 1 verifier-pinned（CodebookFp4）**」。5 真焊死具名 = GridCodebook 497/995 · Ternary 862 · BQL 111/404。
- ISSUE-035 / `.trellis/spec/issues/发射器与架构.md:56`：strip-width「记忆超先验翻盘」对本轴**结构不可达**（honest-null）。

---

# A. 前门算出·发射器 fail-closed 读的 repack 链（selectRepackAccumulatorLMUL 家族）

## θ1 — `integer_core_lmul`（repack 累加器整数核 LMUL 锚：`m1` vs `mf2`）
1. **控制什么**：repack 内积把 8-bit 权重展开成整数向量时用「整条寄存器」(m1) 还是「半条」(mf2)，决定后续整个 i8→i16→i32 位宽阶梯。
2. **现状归哪类**：**(a) 由 f 现场算出**，结果盖章进 IR，发射器端再 **(b) fail-closed 读**（同一 θ 两处住址）。
3. **(a) f 住址**：`selectRepackAccumulatorLMUL`，`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1285-1304`。f 实际读的输入：
   - **c 字段**：`isRVV0p7`（RVV 版本；`:1288` `if(isRVV0p7) return {m1,...}`）；`capabilityHalfLanes`（= VLEN 派生的 e16m1 strip 宽，`:1287` 入参；`:1296` 作 `!=0` witness）；`kVectorRegisterBudget=32`（寄存器数，`:1292`，与 footprint 比）。
   - **g 字段**：`scaleModel`（格式 WHAT，`:1286` 入参）——仅作 `lookupRepackMeasuredM1Faster(scaleModel)` 的键（`:1299-1301`），**该表恒返回 `std::nullopt`**（`:1276-1279`）⟹ scaleModel 当前在此 f 内不改变结果。
   - 结构常量：`getRVVLMULRegisterFootprint("m2")+("m4")`（`:1293-1295`，字面量档名）。
   - 部署态坍缩：`isRVV0p7 ? m1 : mf2`（measured 表空 ⟹ 恒走 `:1303` `capability-default-mf2`）。
4. **(b) fail-closed 读住址（消费侧）**：`RVVToEmitCBlockQuantLinear.cpp` **18 处** `coreLmul = *X.getIntegerCoreLmul()`（机核 `grep -cE 'coreLmul = \*(loopBody|gemv|gemm)\.getIntegerCoreLmul' = 18`）：行 2211/2316/2433/2623/2761/2990/3373/3481/3605/3779/3977/4060/4518/5037/5575/5996/7745/7801，各处前有 `// Fail-closed capability-fact read (was value_or("mf2") board default)` 注释（如 `:2205`）。**性能与测量.md:433 的「×8（2205/2310/2427/2617/2755/2984/3367/3475/3599）」是这 18 中的前 9 个注释行 · 文字写「×8」与所列 9 个不一致**（原档口径矛盾，照抄）。
5. **两板现值**：VLEN128-RVV1.0 = `mf2`（部署默认；ISSUE-033 后前门显式盖 `"mf2"`，`:1778`）；VLEN256-RVV1.0（k1）= `mf2`（measured 表空，同走默认）。**RVV0.7（C920，VLEN128）另一 fork = `m1`**（`:1289`）。
6. **变更史与扇出**：ISSUE-033 §四.3 #2 Step1「前门 18 处显式化(silent-signal 18→0·byte-exact)」commit `0b18a3daf`（git log）；ISSUE-113「optional→required 退休 18 哨兵」= 待裁（用户「选1 暂不升」，`性能与测量.md:428`）。stamp 侧：`addAttribute("integer_core_lmul")` 在前门 **46 次**（`grep -c` · = 每构造点 loopState/coreState/foldState 各盖一次）；`selectRepackAccumulatorLMUL(` 调用点 **18**（`grep -c` 得 19 含定义行）。读端 `getIntegerCoreLmul` 消费计数：BQL 39 · KQuant 3 · CodebookFp4 1 · TernaryBinary 1 · `RVVDialectWideningOps.cpp`(verifier) 55（`git grep -c getIntegerCoreLmul`）。

## θ2 — `half_lanes` / `emittedHalfLanes`（repack strip 宽度）
1. **控制什么**：把 16-block 交织组切成几条 e16m1 strip、每条几车道（128→两条 8 车道，256→一条 16 车道）。
2. **现状归哪类**：**(a) 由 f 现场算出**（Schedule 层派生 + 前门 isM1 覆写）。
3. **(a) f 住址**：`deriveRepackHalfLanes`，`lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp:78-84`：`half_lanes = min(vlen/16, weightInterleave)`；`vlen<128 || weightInterleave<=0 → return 0`。读的输入：
   - **c 字段**：`vlenBits`（= `plugin::rvv::deriveMinimumVLEN(march, isaVectorHints)`，`:99-100`）。
   - **g 字段**：`weightInterleave`（`:79` 入参 = `gemv.getWeightInterleave()`，`:144`）。
   - 前门覆写：`emittedHalfLanes = isM1 ? 16 : halfLanes`（`RVVLowerQuantContraction.cpp:1767`；isM1 来自 θ1）。RVV0.7 分支另在 Schedule 强钉 `half_lanes=16`（`:106-110`）。
4. **(b)/(c)**：不适用（除 isM1 覆写与 RVV0.7 强钉，值来自 f）。
5. **两板现值**：VLEN128 = 8；VLEN256 = 16；RVV0.7(VLEN128) = 16（强钉）。（Verifier pins half_lanes∈{8,16}，`RVVRepackStripWidthMaterialization.cpp:27`；`BlockQuantLinear.cpp:5622` `half_lanes in {8,16} dividing 16`。）
6. **变更史与扇出**：ISSUE-035 strip-width 轴 = honest-null（结构不可达，`发射器与架构.md:56`，task `07-18-r-sel3-recon`）；「宽度轴测量键现为预留空 seed」`:55`。扇出：`half_lanes`/`getHalfLanes` 消费计数 BQL 66 · Internal.h 3 · `RVVDialectWideningOps.cpp` 60 · 前门 43 · Schedule 17（`git grep -c`）。

## θ3 — `accLmul`（每-strip f32/i32 累加器 LMUL：`m4` vs `m2`）
1. **控制什么**：内积累加器向量占几条寄存器。
2. **归类**：**(a)** 内联派生。
3. **(a) f 住址**：`RVVLowerQuantContraction.cpp:1772` `accLmul = isM1 ? "m4" : "m2"`（isM1 来自 θ1，无独立 g/c 输入）。类型 `:1773-1776` f32AccType/i32ResType 由 accLmul 构造。
5. **两板现值**：VLEN128/256-RVV1.0 = `m2`；RVV0.7 = `m4`。
6. **扇出**：随 θ1 stamp 链（同构造点）。零独立记录。

## θ4 — `numHalves`（disjoint strip 条数）
1. **控制什么**：一个 16-block 组拆成几条独立 strip（= 循环展开条数）。
2. **归类**：**(a)** 内联派生。
3. **(a) f 住址**：`RVVLowerQuantContraction.cpp:1768` `numHalves = kWeightInterleave / emittedHalfLanes`（kWeightInterleave=16 常量 `:160`；emittedHalfLanes = θ2）。
5. **两板现值**：VLEN128 = 16/8 = 2；VLEN256 = 16/16 = 1；RVV0.7 = 16/16 = 1。
6. **扇出**：region 内 per-strip 累加器条数。零独立记录。

## θ5 — `ContractionAlgorithm`（family 选择：`Repack` vs `BlockDot`）
1. **控制什么**：这条量化收缩走 repack 重排核还是 block-dot 逐块点积核。
2. **归类**：**(a) 由 f 现场算出**，其中一个 fact 由 **(d) 测量表**供给。
3. **(a) f 住址**：`selectContractionAlgorithm`，`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp:104-172`。读的输入：
   - **g 字段（结构对手事实，从 op 声明读，非格式名）**：`facts.ggmlVlenNativeKernelFloor`（op `opponent_vlen_native_floor`，`RVVLowerQuantContraction.cpp:1336-1337`）；`facts.blockDotComputeHeavy`（`:1338`）；`facts.blockDotMemoryBound`（`:1339`）；`mRegime`（op `m_regime` decode/prefill，`:1355-1362`）。
   - **c 字段**：`minVLEN`（`selection.cpp:106` 入参 = `deriveMinimumVLEN`，`RVVLowerQuantContraction.cpp:1547`）。
   - **d 字段**：`facts.vlen256DecodeRepackBeneficial` ← `lookupRepackVlen256Decode(op.getScaleModel())`（`:1346-1349`）。**该表非空**：`selection.cpp` 家族的 `RVVLowerQuantContraction.cpp:658-673` 5 条 seed（`kNibbleQ50` Beneficial "1.190x" / `kNibbleQ51` Beneficial "1.306x" / `kNibbleQ40` Negative "0.74x" / `kCodebookIq4Nl` Negative "0.248x"）。
4. 输出 `{algorithm, reason}`，reason 为 inert provenance 串（`selection.cpp:140-171` 枚举）。
5. **两板现值**：随 (g 事实×regime×VLEN×measured) 组合分派；部署路径由各格 op 声明决定，无单一「板取几」标量（q4_K 声明 vlen-native-floor → `block-dot-decline-q4_K-vlen-native-exists` `:156-157`；q5_0/q5_1@VLEN256 → `repack-kept-vlen256-decode-measured-beneficial` `:146-147`）。
6. **变更史与扇出**：G8 §六.3 Fact-3-MEASURED（`:1340-1349` 注释）。调用点 1（`RVVLowerQuantContraction.cpp:1547`）。

## θ6 — `RVVRepackTilingVariant`（输出分块：`Plain` vs `S6Tiled`）
1. **控制什么**：repack GEMM 输出维是否走 SP4 分块（缓解 register-cliff）。
2. **归类**：**(a) 由 f 现场算出**（可被 (d) memoized 覆盖）。
3. **(a) f 住址**：`pluginrvv::selectRepackTilingVariant(*shape, minVLEN, kRVVArchVectorRegisterCount, measurement)`，`RVVLowerQuantContraction.cpp:1431-1434`（定义在 `lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp`）。读的输入：
   - **g 字段**：`shape` = `classifyTilingBottleneckShape(foldModel)`（`:1422-1423`；foldModel = 格式 fold 几何）。
   - **c 字段**：`minVLEN`（`:1406`）；`kRVVArchVectorRegisterCount=32`（`:155`）。
   - **d 字段**：`measurement` = `lookupTilingMeasurement(declaredInstanceHash, kernel)`（`:1429-1430`）。
4. 输出盖章 `kTilingVariantAttr`/`kTilingReasonAttr`/`kTilingRecordAttr`（`:1453-1466`）。断言禁 static_order（`:1448-1451`，[SEL-1-T5]）。
5. **两板现值**：生产路径 minVLEN≥128 ∧ regcount=32 ⟹ 可行集 {plain,s6_tiled}，reason∈{measured,prior,only_feasible}（`:1436-1447`）；未见部署单值。
6. **扇出**：`getTilingVariant`/`tiling_variant`/`kTilingVariantAttr` 消费：BQL 2 · 前门 8（`git grep -c`）。

## θ7 — `RVVRepackLoopOrder`（外层组循环序：`RowOuter` vs `ColOuter`）
1. **控制什么**：prefill-GEMM 两组循环谁在外（loop interchange）。
2. **归类**：**(a) 由 f 现场算出**（仅 MEASURED col_outer 改产物，未测则 emission-neutral）。
3. **(a) f 住址**：`pluginrvv::selectRepackLoopOrder(weightStride, activationStride, isPrefillGemm, minVLEN, kRVVArchVectorRegisterCount, loopOrderMeasurement)`，`RVVLowerQuantContraction.cpp:1503-1507`。读的输入：
   - **g 字段**：`weightStride`/`activationStride`（loop op 自带 block strides，`:1497-1500`）；`isPrefillGemm`（op `m_regime`，`:1496`）。
   - **c 字段**：`minVLEN`（`:1495`）；`kRVVArchVectorRegisterCount=32`。
   - **d 字段**：`loopOrderMeasurement` = `lookupLoopOrderMeasurement(...)`（`:1501-1502`）。
4. 输出盖章 `kLoopOrderAttr`/`kLoopOrderReasonAttr`/`kLoopOrderRecordAttr`（`:1508-1519`）。注释 `:1481-1489`：11 sibling 发射体 + inline flat 都读此 stamp，但 sibling arm 只认 MEASURED col_outer；未测 layout-prior = 保留 row_outer + override record（emission-neutral）。
5. **两板现值**：q4_K 携 M1b offline seed（reason=measured, col-outer 2.47x，`:1475-1476`）；余格 cold-start layout prior（reason=prior，row_outer）。
6. **扇出**：`getLoopOrder`/`loop_order`/`kLoopOrderAttr` 消费：BQL 21 · 前门 9（`git grep -c`）。

## θ8 — repack 决策 provenance reason 串（`kAlgorithmAttr`/`kReasonAttr`/`repack_accumulator_lmul_selection_reason`）
1. **控制什么**：审计留痕（算法/理由 token）。
2. **归类**：**(a)** 由上述 f 返回，**发射器不读**（inert）。
3. **f 住址**：随 θ1/θ5（`RepackAccumulatorLMULChoice.reason` `RVVLowerQuantContraction.cpp:1265-1268` 明注「EmitC emitter never reads it, so default emit stays byte-exact」）；stamp 名 `:150`。
5. 两板现值：部署 = `capability-default-mf2`（RVV1.0）/ `correctness-rvv0p7`（RVV0.7）。
6. 扇出：emitter-inert（0 发射消费）；审计消费。

---

# B. MAINTAIN coreLmul 焊死（真焊死·5 处·ISSUE-118 归 MAINTAIN[K-10]）

> 共性（六问打包，逐值分列住址/现值）：
> 1. **控制什么**：本发射体内整数核向量宽度。
> 2. **归类 = (c) 焊死在代码里**。ISSUE-118（`发射器与架构.md:204`）+ ISSUE-033④订正（`:35`）：对应 Context **无 coreLmul 字段·无 IR 宽度可读**，强 lift = 造假旋钮 = [K-10] 违例，MAINTAIN 是 byte-exact 正确动作。
> 4. **盖章链根**：无上游盖章——值是 TU 内字面量常量，被同处 hardcoded 类型/intrinsic 名后缀共同钉死（见各处注释）。
> 5. **两板现值**：VLEN128 = VLEN256 = 同一字面量（无板轴分叉；结构常量）。
> 6. **变更史**：ISSUE-033④ 机核订正（task `07-18-r-iss033d-unweld`，`发射器与架构.md:35`）；stage-3「0/8-debaked」（`发射器与架构.md:204`）。

## θ9 — `RVVToEmitCGridCodebook.cpp:497` `coreLmul = "m1"`（iq3_xxs super-block grid body）
现值 m1 / m1。注释订正：`IQ3XXSGridBodyContext` 无 coreLmul 字段（`发射器与架构.md:35`）。

## θ10 — `RVVToEmitCGridCodebook.cpp:995` `coreLmul = "m1"`（iq3_s super-block grid body）
现值 m1 / m1。`IQ3SGridBodyContext` 无 coreLmul 字段。

## θ11 — `RVVToEmitCTernaryBinary.cpp:862` `coreLmul = "m1"`
现值 m1 / m1。下游 `:1214`/`:1337` 以 coreLmul 拼 `vle8` intrinsic 名。

## θ12 — `RVVToEmitCBlockQuantLinear.cpp:111` `coreLmul = "m1"`
现值 m1 / m1。注释 `:107-110`：「structural constant, NOT a knob」「hardcoded vint8m1_t/vint16m2_t + intrinsic 名后缀共钉同一 shape · no IR width to read · [K-10] do not fake a knob」。同处 `wideLmul="m2"`（`:112`）、`i8CoreType=vint8m1_t`（`:113`）。（ISSUE-116 `:194` 引作 BQL:107；机核字面量在 `:111`。）

## θ13 — `RVVToEmitCBlockQuantLinear.cpp:404` `coreLmul = "m1"`
现值 m1 / m1。（ISSUE-116 引作 BQL:397；机核字面量在 `:404`。）GEMM body。

---

# C. verifier-pinned（1 处）

## θ14 — `RVVToEmitCCodebookFp4.cpp:109` `coreLmul = "m1"`
1. 控制：FP4 codebook 核向量宽度。
2. **归类 = (c)**，ISSUE-118 归「verifier-pinned」子类（`发射器与架构.md:204`）。
3. 下游 `deriveWideningChain(coreLmul)`（`:110`）+ 类型名 `:112-113`。第二处同文件 `:734` 为函数入参 `coreLmul`（由 `:749` `deriveWideningChain` 消费；调用方传入）。
5. 现值 m1 / m1。
6. 该文件 coreLmul 引用计 18（`git grep -c` · 多为 intrinsic 名拼接）。

---

# D. live-default-override / value_or 软默认（vtype 已派生·非 fail-closed）

> 共性：**归类 = (c) 焊死默认值 + optional 覆写**（`if(attr) coreLmul=*attr` 或 `.value_or(默认)`）。**非 fail-closed**（缺 attr 走软默认，不报错）。各处注释均记「前门 by design UNSTAMPED · 生产 e2e 走此默认 · fail-close 会破 byte-exact」。

## θ15 — `RVVToEmitCTernaryBinary.cpp:1607` `coreLmul="m2"` + `:1609` `if(attrLmul) coreLmul=*attrLmul`
现值 m2 / m2（override 罕盖）。`:1610` `wideLmul = (m2)?m4:m2`。

## θ16 — `RVVToEmitCBlockQuantLinear.cpp:14232` `coreLmul="m2"` + `:14234` attr 覆写（binary-sign flat body）
现值 m2 / m2。注释 `:14229-14231`：「q1_0 production e2e lowers at this default」。

## θ17 — `RVVToEmitCBlockQuantLinear.cpp:14315` `coreLmul="m1"` + `:14317` attr 覆写（codebook flat body）
现值 m1 / m1。注释 `:14312-14314`：「nvfp4 production e2e lowers at this default」。

## θ18 — `RVVToEmitCKQuant.cpp:2918` `coreLmul = scaledDot.getIntegerCoreLmul().value_or("mf2")`（q4_K/q6_K scaled-dot 超块核）
现值 mf2 / mf2。`:2984` `cx.coreLmul = coreLmul` → `:365` `deriveWideningChain(cx.coreLmul)`。ISSUE-033② 记此为「真实测过路径承重」（`发射器与架构.md:40`）。

## θ19 — `RVVToEmitCKQuant.cpp:3694` `coreLmulAttr = b3.getIntegerCoreLmul().value_or("mf2")`（q4_K/q5_K min-fold 块，b3）
现值 mf2 / mf2。**此值被重载为 emit-变体选择器**（见 F 组 θ27-30）；`:3719-3720` `coreLmul = aux8Free ? "m2" : coreLmulAttr` → `:3721` `deriveWideningChain`。ISSUE-033② 承重（`发射器与架构.md:40`，前门 = `materialize-q4-k-q8-k-block-dot-source-front-door`，非触碰集前门）。

## θ20 — `RVVToEmitCKQuant.cpp:5971` `coreLmul = coreOp.getIntegerCoreLmul().value_or("m2")`（iq2_xxs pair-batched grid gearbox）
现值 m2 / m2。ISSUE-033 记 `IQ2XXSGridBodyContext` **有** coreLmul 字段 = 「真 gearbox」（对照 iq3 无字段，`发射器与架构.md:35`）。填入 `IQ2XXSGridBodyContext cx{...}`（`:6023`）。

## θ21 — `RVVToEmitCGridCodebook.cpp:59` `coreLmul = cx.coreLmul`（grid body 分派器读 context）
现值 = 上游 context 值（经 KQuant 填）。`:74` `wideLmul=(m2)?m4:m2`、`:179-180` `pairLmul`/`pairIdxLmul` 由 coreLmul 派生。**非独立决策**（值来自 θ18/θ20 类的 context 传递）。

---

# E. Internal.h 板阶梯 / 位宽链（deriveWideningChain·由 coreLmul 纯派生）

> 共性：**归类 = (a) 纯派生**——`deriveWideningChain(base)` 只读 `base`(= coreLmul 字符串)，**不读 g、不读 c**。定义 `lib/Conversion/RVV/RVVToEmitCSupport.cpp:1064-1073`。struct `WideningChain{l8,l16,l32,stripWidth,foldGroups}`（`include/Weft/Conversion/RVV/RVVToEmitCSupport.h:411-417`）。

## θ22 — `l8`/`l16`/`l32`（i8→i16→i32 三档位宽）
`Support.cpp:1066-1069`：`l8=base; l16=widenOneStep(base); l32=widenOneStep(l16)`（`widenOneStep` def `:1029`）。Internal.h 结构默认 `l8="mf2"/l16="m1"/l32="m2"`（`RVVToEmitCInternal.h:3634-3637`；注释 `:3628-3633`「default mf2→m1→m2 reproduces byte-identical legacy」）。两板现值：随 coreLmul（部署 mf2 ⟹ mf2/m1/m2）。

## θ23 — `stripWidth`（VLEN128 下 i8 strip 车道）
`Support.cpp:1070` `stripWidth = i8StripWidthAtVlen128(base)`（def `:1046`）。值表（Internal.h `:3639` 注释）：mf2→8, m1→16, m2→32。默认 `RVVToEmitCInternal.h:3648` stripWidth=8。

## θ24 — `foldGroups`
`Support.cpp:1071` `foldGroups = stripWidth/8`。默认 `RVVToEmitCInternal.h:3650` = 1（mf2 时 fold-back 不触发）。

## θ25 — `numStrips`（KQuant super-block context，32/stripWidth）
`RVVToEmitCInternal.h:3649` numStrips=4（默认 mf2）；注释 `:3638-3640` `32/stripWidth = 4/2/1`。派生自 coreLmul。

## θ26 — GridCodebook `wideLmul`/`pairLmul`/`pairIdxLmul`（pair-batched 展开档）
`RVVToEmitCGridCodebook.cpp:74/179/180`，三目 `(coreLmul=="m2")?...`。纯派生自 θ21 coreLmul。

---

# F. 宽化战役（K宽化）emit-变体门（integer_core_lmul 被重载为发射模式选择器·全 dormant）

> 共性：**归类 = (c) 焊死门 + 字符串驱动**。四门读 `coreLmulAttr`(= θ19，`RVVToEmitCKQuant.cpp:3694` value_or("mf2"))，默认 mf2 ⟹ 四门**全 false（dormant）**，产物 byte-identical。均附 `&& !hasQh`。best q4_K vec_dot@rvv = m1 0.186（`性能与测量.md:152`）。sealed 默认 byte-identical `892b6cf8`。

## θ27 — `useRegisterFusion`（`RVVToEmitCKQuant.cpp:3695` `coreLmulAttr=="fused" && !hasQh`）
现值 false / false（默认）。变更史：`性能与测量.md:148` 「register-fusion 已施工+板测=EXHAUSTED(negative)·cold 0.152 略慢·墙没动」（task `07-18-k-regfusion-q4k`）。

## θ28 — `useVwredsum`（`:3696` `=="vwredsum"`）
现值 false / false。`性能与测量.md:148` 「vwredsum.vs 已施工+板测=EXHAUSTED·cold 0.162·墙没动·sealed 892b6cf8」（task `07-18-k-issue-109-vwredsum`）。

## θ29 — `useMintermVec`（`:3704` `=="minterm-vec"`）
现值 false / false。`性能与测量.md:154` 「minterm-vec 已施工+板测=EXHAUSTED·cold 0.164·墙没动·ULP=0」（task `07-18-q4k-minterm-vec`）。

## θ30 — `useMlp`（`:3714` `=="mlp"`）
现值 false / false。`性能与测量.md:151` 「MLP lever 已施工+板测·cold 0.155 未超 best 0.186·第 5 次定格·认编译器行为脾气墙[K-4]」（task `07-18-q4k-mlp-attack`）。
（`:3715-3720` 派生：`useVwredsumBlockDot`/`useVecMinTerm`/`aux8Free`/`coreLmul = aux8Free?"m2":coreLmulAttr`。）

---

# G. flat-body 其它 optional 旋钮（value_or 软默认·非 fail-closed）

> 共性：**归类 = (c) 焊死默认 + optional**（`.value_or(默认)`），前门当前不盖 ⟹ 走默认。

## θ31 — `multiBlockFactor`（`RVVToEmitCBlockQuantLinear.cpp:14151` `getMultiBlockFactor().value_or(1)`）
控制：flat block-dot 一次跨几块（mbf 旋钮）。现值 1 / 1。注释 `:14127`「any unroll form is fail-closed (I7)」/ `:14150`「folds still require mbf==1 + elided default」。

## θ32 — `stripElision`（`:14152` `getStripElision().value_or("robust") == "elided"`）
控制：strip 循环是否省略（VLEN≥128 单strip）。现值 robust / robust（默认）。

## θ33 — `foldStructure`（`:14438` `getFoldStructure().value_or("per-block")`）
控制：fp 折叠结构（per-block vs …）。现值 per-block / per-block。

## θ34 — `numericsTier`（`:14455` `getNumericsTier().value_or("strict")`）
控制：数值档（strict vs relaxed body）。现值 strict / strict。注释 `:14447-14453`「absent = strict, fail-closed I7 · relaxed body 只在存在处 honored」。

---

# H. 审计镜像（inert·记账但不改产物路由）

## θ35 — `weft_rvv.gearbox_selected_integer_core_lmul`（`RVVDequantDotSourceFrontDoor.cpp:903`）
1. 控制：dequant-dot 变体所选 byte 锚（审计留痕）。
2. **归类**：**(b) 前门盖章·但注释明记 AUDIT-ONLY / mirror**（`:900-902`「NOT the authority · a mirror, never the route/dtype authority」）。
3. 值 = `selectedIntegerCoreLMUL`（上游 gearbox 已结构消费）。
5/6. 部署镜像；结构由 body 消费，非此 attr。

---

# 非-θ 但相邻（供上游辨识·不计入 θ 计数）
- `lib/Plugin/RVV/Schedule/RVVProbedCapabilityAxesMaterialization.cpp:124` 物化 `supported_lmul`/`supported_sew`（`:79`,`:124`）= **c 轴能力事实材料化**（不是决定）。
- `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:658-673` `lookupRepackVlen256Decode` 5-seed 表 = **(d) 测量表**本体（喂 θ5 的 fact，非独立 θ）。
- `lookupRepackMeasuredM1Faster`(`:1276-1279`, 恒 nullopt) / `lookupTilingMeasurement`(`:1430`) / `lookupLoopOrderMeasurement`(`:1502`) = θ1/θ6/θ7 的 (d) 通道；前者表空，后两者经 declaredInstanceHash 键控。

---

# 计数（纯数字·口径见括号）
- **θ 总数（本清单枚举） = 35**（θ1–θ35）。
- 按现状主类归属（每 θ 记其当前主住址一类）：
  - **(a) 由 f 现场算出 = 11**：θ1(前门算·发射器 fail-closed 读双住址·计入 a) · θ2 · θ3 · θ4 · θ5(含 d-表 fact) · θ6 · θ7 · θ8(inert) · θ22 · θ23 · θ24（θ22-24 = deriveWideningChain 纯派生；θ25/θ26 亦纯派生但归 E 组随 coreLmul，未重复计入 a — 见下「派生附注」）。
  - **(b) 盖章+fail-closed 守 = 1**：θ35（audit-only mirror；θ1 的 (b) 消费侧 18 fail-closed 读并入 θ1 的 a 记，不重复计）。
  - **(c) 焊死在代码里 = 21**：θ9–θ21（13 处 coreLmul：5 真焊死 θ9-13 + 1 verifier-pinned θ14 + 7 value_or/override θ15-21）+ θ27–θ34（4 emit门 θ27-30 + 4 flat旋钮 θ31-34 = 8）。
  - **(d) 由测量表选出 = 0**（当前无 θ 经非空测量表翻值：θ1 表恒空；θ5 读非空 vlen256 表但作 fact 融入 a；θ6/θ7 表经 hash 键控，生产 leaf 未见 measured 命中的单一部署值）。
  - **派生-未归四类 = 2**：θ25/θ26 = 纯派生（未单列入 a，避免与 θ22-24 重复计 deriveWideningChain）。
  - **合计校验**：a 11 + b 1 + c 21 + d 0 + 派生 2 = **35** ✓。
- 其它计数：BQL fail-closed `*getIntegerCoreLmul()` 读 = **18**（grep 精确）；前门 `addAttribute("integer_core_lmul")` = **46**；`selectRepackAccumulatorLMUL` 调用点 = **18**；coreLmul 逐文件计（`git grep -c coreLmul`）= BQL 318 / Internal.h 46 / KQuant 21 / CodebookFp4 18 / GridCodebook 15 / TernaryBinary 15 / Support.h 4 / WideningOps.cpp 84 / MonolithicFrontDoor 5。
