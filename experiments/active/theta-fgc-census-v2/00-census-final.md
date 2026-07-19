# 综合交付 · θ=f(g,c) 三角色全景盘点 v2 —— 六包组装

**HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（`git rev-parse HEAD` 实测 == 此 pin·全文 file:line 直读工作树 == HEAD·未 checkout）

> 角色框架：**g = 格式侧输入**（块宽/交织宽/子块几何/码本尺寸…·应住描述符）；**c = 板侧输入**（VLEN/寄存器数/分数LMUL有无…·应住能力表）；**θ = 输出/决定**（LMUL档/strip宽/家族选择/展开数…·应由 f(g,c) 算出）。
> **只列事实 + 出处·禁判断/禁分类建议/禁「因此」/禁结论。** 是不是缺陷由上游判。
> 本文件 = pkg1–pkg6 六包组装（六包各节内容照搬）；首节 = 纯计数五行；次节 = 采不到清单汇总。

---

## 首节 · 纯计数五行（从各 pkg 汇总·数字不带任何解读·口径见括号）

**1｜θ 共 35 个**（pkg1 枚举 θ1–θ35）·按现状主类：**(a) 由 f 现场算出 = 11** · **(b) 盖章+fail-closed 守 = 1** · **(c) 焊死在代码里 = 21** · **(d) 由测量表选出 = 0** · 派生-未归四类 = 2（θ25/θ26）。合计校验 11+1+21+0+2 = 35。

**2｜f 共 37 条**（pkg4 标号 F1–F37·Tier-1 主选择器 25 [F1–F25·含 F21b 无 F21] + Tier-2 能力/几何派生 12 [F26–F37]；Tier-3 θ→θ 实现派生 7 + Tier-4 焊死点/哨兵 不计入 f 总数·pkg4 口径）。**其中单侧输入 = 18**（本 agent 依 pkg4 函数签名机读归集口径）：**仅读 c = 10**（F7[shape 被 `(void)` 丢弃]·F25[签名仅吃 c·g baked]·F26·F27·F28·F29·F30·F31·F33·F35，其中 F26–F31/F33/F35 标 ★[c only]）· **仅读 g = 5**（F5·F6·F12·F19·F24）· **零输入/参数未读 = 3**（F4·F21b·F37）。

**3｜c 字段**（多口径·pkg2）：**RVVProbeCapabilityFacts struct 输入字段 = 15·零消费 = 2**（cachelineBytes·imePresent；另 `deriveIMEPresent` 零调用者）；另口径——RVV 探测产出 provider-property = **15 键·零消费 9**（vlenb:bytes·clang:version·cmake:version·compile_run 4 键·march:value·mabi:value；被 f 消费 6）；schema params namespaces = **7·全 grep=0 in code**；schema fact_record 字段 = **9·declared-no-producer 3**（subclass·provenance·trust）。

**4｜g 字段 = 64**（pkg3·g 桶 distinct 字段名·数字/几何/偏移/stride/codebook/format-身份）·**零消费 = 0**（64 g 字段全有 typed-getter 读者·`comm -12` = 空）。描述符全集另有 34 个「零-typed-getter」名·**其中 g 字段 = 0**（余为 c/θ/元数据桶·多有 string-ref 写侧）。

**5｜残余焊死 g = 18 处**（pkg3·本 pin pin-shape 谓词）：GridCodebook 9 + Ternary 7（本 pin·census@f457=8·dotStripLanes 已消失）+ KQuant 2；另 ForwardElementwise dequant-row **~51 处宽谓词**（ISSUE-119·未穷举到 100%）。

> 附计数（非五行·pkg1/pkg6）：BQL fail-closed `*getIntegerCoreLmul()` 读 = **18**（grep 精确；性能与测量.md:433 文字写「×8」与所列 9 个行号不一致·原档口径矛盾照录）；前门 `addAttribute("integer_core_lmul")` = 46；`selectRepackAccumulatorLMUL` 调用点 = 18。(b)/(c) θ 核查 22 处中——带独立字面量假旋钮（ISSUE-031(a) 型）= **5**（θ9/θ10/θ11/θ12/θ13）·非假旋钮（vtype 派生自 coreLmul）= 8·无 vtype 字面量（结构/模式/镜像）= 9。(iii) 零消费 g 死料 = 0·g 待接 = 0。

---

## 次节 · 采不到 / 未核清单（汇总各 pkg cant_collect）

**pkg1（θ）**：性能与测量.md:433 文字「×8」与括号内 9 个行号不一致（原档口径矛盾·照录未裁）。

**pkg2（c 能力表）**：
- 无单一 per-board 能力实例权威文件（JSON/MLIR）——三板现值散落三处（board册 prose + test/tool fixture + pass `-march` 字符串）；机算未见统一 board profile 台账。
- k1 的 `vreg_count` / `cacheline` / `elen` / `sew_set` 具体值：采不到（board册未列·fixture 未含·schema namespace 声明但代码零落）。
- `vreg_count` 代码侧落地：采不到（`git grep -c vreg_count` 仅 schema JSON 命中·无 struct 字段/无 property 键/无消费点）。
- scalar（无 V）板逐字段能力属性：采不到（仅「无 v 扩展」否定事实·板册.md:16 `/proc/cpuinfo` 无 `v`）。
- rvv07 板：REGISTERED-PENDING·零工作（板册.md:12）·无 fixture 值。

**pkg3（g 描述符）**：
- 34 个零-typed-getter attr 的「只写不读 vs 泛型 string 读」逐字段三分未穷做（仅抽验 required_capabilities/selected_variant=有 string 读·planning/remediation/admission=写向）。
- ForwardElementwise dequant-row ~40/~51 未穷举到每个 dOff/mOff/sub 小常量（census §Caveat 亦标·数据源自 census 宽谓词）。
- ISSUE-118/119 提「阶段2 已建同名 attr 可复用」为类比推断（GridCodebook/Ternary 是不同 op·须各自声明·census 未逐一验证 call-site 能否填 Context）——照引·未独立复验。
- 未编译/未跑任何 fixture（只读令）；所有「byte-exact」「消费」断言 = 静态阅读 + getter/string 机扫推断。

**pkg4（f 加工环节）**：
- `computeBlockDotShapeCostCore`/`getRVVBlockDotCoreLatencyDepth`/`computeRVVQ40ShapeCost` 等 cost formula 全文未逐一照抄（已记接口与 capability-blind 性质）。
- IME 侧 `kIMEWideFormatMeasurements` / IME LMUL 选择器未纳入（本包锁 RVV f·IME 为另栈）。
- 各 front-door `toGenericBlockDotCandidate`/`toGenericGemmCandidate` 适配器（纯类型转换·不产 θ 决定）未展开。

**pkg5（实证）**：
- 与「C920」绑定的具体「9/9」板测/通过数：采不到（仓内 `9/9` 命中全是 fp16 `h2f 9/9 textbook IEEE` + `bench --self-test 9/9`[门与工具.md:86] + T4b/T-CENSUS 的 9/9·均与 C920 无关）。C920 板间行为差异（RVV0.7-vs-RVV1.0 divergence·supported_lmul m1,m2,m4,m8 vs mf8..m8·fractional-LMUL 有无·fail-closed）已采到。

**pkg6（假旋钮/断链）**：
- pkg3 的 `scratchpad/gfields.txt`/`consumption.txt`（生成 g-桶零消费=0 结论的机算中间物）为会话私有·本 pin 已不在（`ls scratchpad/` 空）——未独立重跑全量枚举·照引 pkg3 机核结论（仅抽验「zero-typed-getter ≠ 零消费」推理成立：unroll_factor/selected_variant 经泛型 getAttr 活读）。
- 34 个零-typed-getter attr 的「只写不读 vs 泛型 string 读」逐字段三分未穷做（仅核 2 个 θ 桶代表 + θ35 镜像）。
- θ11 pkg1 记的下游 :1214/:1337 vle8 拼接未逐行复读（窗抽到 `riscvIntrinsicName(...,coreLmul,...)` 命中即证消费成立·未穷举全部拼接点）。
- 未编译/未跑 fixture（只读令）·所有断言 = 静态代码路径阅读 + 注释自陈引用·非运行验证。

---
---

<!-- ===== 以下六节内容照搬各 pkg（pkg1–pkg6）·各 pkg 自带 HEAD pin 与标题保留 ===== -->



<!-- ========== 包1 节 (源: pkg1-theta.md) ========== -->

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


<!-- ========== 包2 节 (源: pkg2-c-capability.md) ========== -->

# 包2 · c 进料口盘点（能力表）— θ=f(g,c) 三角色全景盘点 v2

HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`（工作树 == HEAD 此 pin）

> **角色框架**：c = 板侧输入（硅片/厂商定义的能力量：VLEN/寄存器数/分数LMUL有无…），存放地应为**能力表**。本包只列事实 + 出处，禁判断/禁分类/禁结论。
> 全文引用一律 `file:line`（工作树内容 == pin）。"不存在"类断言均附机算枚举命令。

---

## 0. 能力表有几个"表面"（枚举，非结论）

机算发现 c 进料存在**四个并行表面**，同一板事实在不同表面各有一份形态：

| 表面 | 定义处 | 形态 | 出处 |
|---|---|---|---|
| S1 schema 声明（TARGET 目标形） | `schema/capability.schema.v1.json` | JSON namespaces + fact_record 字段 | 该文件 `$meta`（15:6）自述"declares TARGET shape, NOT a snapshot of current code" |
| S2 代码侧 fact 记录 | `include/Weft/Support/CapabilityModel.h` `class CapabilityDescriptor` | symbolName/id/kind/status/availability/properties(map)/relations | CapabilityModel.h:86-146 |
| S3 探测料口结构体 | `include/Weft/Plugin/RVV/RVVCapabilityProfile.h` `struct RVVProbeCapabilityFacts` | 15 个 C++ 字段 | RVVCapabilityProfile.h:61-88 |
| S4 in-IR provider op 属性 | `weft.exec.capability` / `weft.exec.target` 上的 StringAttr 属性 | property 字符串键值对 | 由 S3 经 `buildRVVTargetCapabilitiesFromProbeFacts` 产出（RVVCapabilityProfile.cpp:486-608）+ 材料化 pass 盖章 |

schema `$meta`（capability.schema.v1.json:6）明文：`provenance/trust/subclass`、closed kind enum、namespaced-typed params、unified operand roles、serializable plugin signature **均 grep=0 in code today**，"code and schema.def are intentionally divergent"。

---

## 1. capability schema 全字段机算枚举（S1 · schema/capability.schema.v1.json）

### 1a. item_1_fact_record — 每条 capability fact 携带的字段（9 个）
出处 capability.schema.v1.json:15-85。

| # | 字段 | 类型 | 值域/默认 | 备注 | 行 |
|---|---|---|---|---|---|
| 1 | `id` | string | namespaced dotted stable id（如 `rvv.v`/`ime.vmadot`/`scalar.zbb`/`uarch.vrgather_slow`） | authoritative_key=true；bench名/日志/perf值禁成 id（I9） | 18-23 |
| 2 | `kind` | enum | → item_2（closed 4 值） | | 24-27 |
| 3 | `subclass` | string | optional | **Declared-with-no-producer in v1**（原文 31） | 28-32 |
| 4 | `status` | enum closed | {available, unavailable, disabled, missing}，default=available | missing 解释为 available；status 优先于 availability（[S-2]） | 33-44 |
| 5 | `availability` | enum closed | {available, unavailable} | | 45-53 |
| 6 | `provenance` | enum closed | {hwprobe, cpuinfo, vendor_table, manual}，default=manual | 证据线字段（I8），不参与 compute；**Declared-with-no-producer in v1**（producer 是 P2） | 54-65 |
| 7 | `trust` | enum closed | {measured, declared}，default=declared | 证据线（I8）；**Declared-with-no-producer in v1** | 66-75 |
| 8 | `params` | namespaced_struct | → item_4 | | 76-79 |
| 9 | `relations` | relations | → item_3 | | 80-83 |

### 1b. item_2_kind_enum — fact 类别轴（closed enum 4 值）
出处 capability.schema.v1.json:86-105。成员：`isa_ext` / `sub_ext` / `uarch` / `policy`。companion_field=`subclass`。映射（97-102）：isa_ext=ISA扩展 / sub_ext=子扩展(zvfh/zbb) / uarch=微架构·内存·VLEN facts / policy=toolchain·runtime-offload·thread-runtime可用性+build/permission门。

### 1c. item_3_relation_types — 关系类型表（3 关系 + 每关系语义标注）
出处 capability.schema.v1.json:106-125。

| 关系 | value_type | semantics | 行 |
|---|---|---|---|
| `provides` | list<capability_id_string> | satisfies-by-alias | 109-113 |
| `implies` | list<capability_id_string> | transitive-satisfiable（load 时物化一次闭包 [S-2]） | 114-118 |
| `conflicts` | list<capability_id_string> | fail-closed-mutual-exclusion | 119-123 |

### 1d. item_4_params_namespaces — 命名空间化 typed param 字段（7 namespace）★核查点密集区
出处 capability.schema.v1.json:126-162。

| # | namespace | type | layer | 备注 | 行 |
|---|---|---|---|---|---|
| 1 | `vlen` | int | hardware-fact | raw HW VLEN bits（Layer1）；非线程数/门/形状/AVL | 129-136 |
| 2 | `elen` | int | hardware-fact | | 137-139 |
| 3 | `sew_set` | set_of_int | hardware-fact | 支持的 SEW allow-list，typed set（非 CSV string） | 138-142 |
| 4 | `lmul_budget` | int_or_fraction | compile-time-variant-config | | 143-146 |
| 5 | **`vreg_count`** | int | hardware-fact | **★定点核查:寄存器总数——仅此处声明** | 147-150 |
| 6 | `cacheline` | int | hardware-fact | | 151-154 |
| 7 | `ime.tile` | struct | hardware-fact | IME tile 几何，namespaced under ime | 155-159 |

**机算核查（`$meta` 自述 grep=0 in code）**：`git grep -c 'vreg_count\|sew_set\|lmul_budget\|elen\|ime.tile' d173f4… -- lib include tools` → 除 schema JSON 自身外**无代码命中**（这些 namespaced-typed param 字段是 TARGET 声明，代码侧未落，属跟踪中的 conformance gap，见 `$meta` note 行 6）。

### 1e. item_5 / item_6 / not_in_shape（非 c 料口，简记）
- item_5：ExtensionPlugin quasi-ABI 签名投影（18 virtual methods，3 pure）——插件协议 shape，非板事实（163-297）。
- item_6：operand-role 词表（axis-A ABI-slot 28 成员 / axis-B body-op role 7 属性）——算子角色，非 c（298-357）。
- `not_in_shape`（358-368）明文排除：concrete fact rows、params VALUES（`sew_set={8,16,32}`、`vlen=128`）、plugin internal code、measurement library、pattern-registry entries、profiles。⟹ **schema 明确"具体板值不在 shape 内"** → 三板现值不住 schema，住 fixture/board册/pass-option（见 §3）。

---

## 2. 代码侧 c 料口全字段（S2 + S3）

### 2a. CapabilityDescriptor 字段（S2 · CapabilityModel.h:135-146）
私有成员：`symbolName`(string)、`id`(string)、`kind`(string)、`status`(string)、`availability`(enum CapabilityAvailability{Available,Unavailable})、`properties`(std::map<string,string>)、`relations`(CapabilityRelationsAttr 内含 provides/implies/conflicts)。
- kind 值域**非 closed enum**：代码里是自由 string（构造函数 CapabilityModel.cpp:211-218 直接吃 StringRef），与 schema item_2 的 closed-4 声明背离。实测产出的 kind 字面见 §2c。
- properties 是**自由 string map**（非 schema item_4 的 namespaced typed struct）；`collectCapabilityProperties`（CapabilityModel.cpp:86-99）把除 core/relation 外的所有属性 stringify 塞进 map。core 属性（isCoreCapabilityAttribute，CapabilityModel.cpp:44-52）= sym_name/id/kind/target_kind/status/capability_providers。

### 2b. RVVProbeCapabilityFacts 探测料口结构体（S3 · RVVCapabilityProfile.h:61-88）——15 字段
| # | 字段 | 类型 | 默认 | 消费见 §4 | 行 |
|---|---|---|---|---|---|
| 1 | `architecture` | std::string | | | 62 |
| 2 | `hartCount` | uint64 | 0 | | 63 |
| 3 | `vlenbBytes` | uint64 | 0 | | 64 |
| 4 | **`cachelineBytes`** | uint64 | 0 | **★零消费(见 §5)** | 65-71 |
| 5 | **`imePresent`** | bool | false | **★零消费(见 §5)** | 72-77 |
| 6 | `isaVectorHints` | std::string | | | 78 |
| 7 | `clangAvailable` | bool | false | | 79 |
| 8 | `clangVersion` | std::string | | | 80 |
| 9 | `cmakeAvailable` | bool | false | | 81 |
| 10 | `cmakeVersion` | std::string | | | 82 |
| 11 | `minimalRVVCompileRunSucceeded` | bool | false | | 83 |
| 12 | `selectedMarch` | std::string | | | 84 |
| 13 | `selectedMABI` | std::string | | | 85 |
| 14 | `sourceSHA256` | std::string | | | 86 |
| 15 | `binarySHA256` | std::string | | | 87 |
| — | `deriveIMEPresent(march,hints)` | inline bool | | **★零调用者(见 §5)**；token=`xsmtvdotii` | 194-199 |

结构体字段 4/5（cachelineBytes/imePresent）header 原文自述：cachelineBytes"NOT consumed by any selector yet"（70）、imePresent"NOT wired to any selector by this task"（76-77）。

### 2c. RVV 探测→provider op 实产出（S4 · RVVCapabilityProfile.cpp:486-608）
`buildRVVTargetCapabilitiesFromProbeFacts` 产出的 provider op 及其 property 键（原样）：

| provider symbol | id | kind | 产出 property 键 | 产出行 |
|---|---|---|---|---|
| `rvv`(getRVVPreferredCapabilitySymbol) | `rvv` | `isa-vector`(getRVVCapabilityKind) | architecture, isa_vector_hints, **supported_sew**(派生), **supported_lmul**(派生), **rvv_version**(派生) | 493-525 |
| `rvv_zvfhmin`（条件） | `rvv.zvfhmin` | `isa-vector-fp16` | —(implies rvv.zve32f) | 541-547 |
| `rvv_zvfh`（条件） | `rvv.zvfh` | `isa-vector-fp16` | —(implies rvv.zvfhmin) | 548-554 |
| `rvv_hart_count` | `rvv.hart_count` | `uarch` | `count`(=hartCount)；provides `target.hart_count` | 556-561 |
| `rvv_vlenb_bytes`（条件 vlenbBytes≠0） | `rvv.vlenb_bytes` | `uarch` | `bytes`(=vlenbBytes) | 562-568 |
| `rvv_toolchain_clang` | `rvv.toolchain.clang` | `toolchain` | `version`(=clangVersion) | 569-573 |
| `rvv_toolchain_cmake` | `rvv.toolchain.cmake` | `toolchain` | `version`(=cmakeVersion) | 574-578 |
| `rvv_probe_compile_run` | `rvv.probe.compile_run` | `toolchain` | `selected_march`, `selected_mabi`(条件), `source_sha256`(条件), `binary_sha256`(条件) | 580-592 |
| `rvv_toolchain_march` | `rvv.toolchain.march` | `toolchain` | `value`(=selectedMarch) | 594-598 |
| `rvv_toolchain_mabi`（条件） | `rvv.toolchain.mabi` | `toolchain` | `value`(=selectedMABI) | 599-605 |

**注**：cachelineBytes/imePresent 两字段在此产出函数里**从不被读取/从不产出任何 fact**（机算：`git grep -n 'cacheline\|imePresent\|\.imePresent' … -- lib/Plugin/RVV/RVVCapabilityProfile.cpp` → 命中 0，见 §5）。

### 2d. 三条派生轴（θ-形还是 c-形？—— 只列事实）
`buildRVVTargetCapabilitiesFromProbeFacts` 内把三个**派生量**盖到 `rvv` op（RVVCapabilityProfile.cpp:504-521）：
- `supported_sew` ← `deriveSupportedSEWAllowList(march,hints)`（.cpp:222-243）：zve32*→"8,16,32"；full-V/zve64/gcv/xtheadvector→"8,16,32,64"；否则""。
- `supported_lmul` ← `deriveSupportedLMULAllowList`（.cpp:260-277）：RVV0.7→"m1,m2,m4,m8"；RVV1.0→"mf8,mf4,mf2,m1,m2,m4,m8"；Unknown→""。
- `rvv_version` ← `stringifyRVVVersion(deriveRVVVersion)`（.cpp:341-375）："0.7"/"1.0"/""。
header/注释自述这些是"TARGET-CAPABILITY fact（配置目标支持什么），NOT plugin-selected config"（RVVCapabilityProfile.h:108-144 / .cpp:206-221）。**（此陈述为原文事实照录，不作三角色归类。）**

---

## 3. 三块板实例逐字段现值（原样摊开）

★ schema `not_in_shape`（capability.schema.v1.json:358-368）明文：具体板值**不在 schema 内**。机算搜索（`git grep -ln 'weft.exec.capability' … -- schema experiments`）**未发现单一"per-board 能力实例权威 JSON"**；三板现值散落在三处：(A) board册 prose、(B) test/tool fixtures、(C) pass `-march` 字符串。以下原样照录，标出处。

### 3a. board册权威（.trellis/spec/measurement/板册.md:7-16）
| 别名 | 身份 | 向量 | 测量特性 | 出处行 |
|---|---|---|---|---|
| rvv | openEuler·64核 | **VLEN128** | 有 cache-miss 硬件计数器 | 9 |
| k1 | SpacemiT X60·8核 | **VLEN256 + IME** | 无可用 PMU（仅指令/周期）·常载偏高 | 10 |
| scalar | 超锐·Fedora42·8核 | **无向量（实证）** | 载荷极低·有硬件计数器 | 11 |
| rvv07 | RVV0.7前身·128核 | 非标准 | — | REGISTERED-PENDING·零工作 | 12 |

scalar"无向量"实证依据（板册.md:16 原文）：板端 `/proc/cpuinfo` isa 串 = `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`——**无 `v` 扩展**。

### 3b. t1d 双实例 fixture 现值（tools/visibility/t1d_dual_instance.py:179-188）——rvv/k1 逐字段
```
INSTANCES = {
  "rvv": board="rvv", vlen="128", hints="rv64gcv_zvl128b",
         ime_status="missing", ime_extra="", march="rv64gcv"           # :180-182
  "k1":  board="k1",  vlen="256", hints="rv64gcv_zvl256b",
         ime_status="available",
         ime_extra=  available_harts="0-3",  ime_matmul_shape="256x256x256",
         march="rv64gcv_zvl256b"                                        # :183-188
}
```
t1d 产出的 in-IR provider 属性模板（t1d_dual_instance.py:88-100, 120-129）：rvv capability 携 `id="rvv"`, `isa_vector_hints="{hints}"`, `vlen_bits="{vlen}"`；IME capability 携 `march="rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii"`, `vlen_bits="{vlen}"`, `status="{ime_status}"`, （k1）`available_harts="0-3"`, `ime_matmul_shape="256x256x256"`。
注（t1d 头 176-179 原文）：rvv board=VLEN128 无 IME silicon（spacemit.ime status=missing→[S-2] 解释为 unavailable）；k1 board=VLEN256 带 IME。

### 3c. divergence 测试 fixture 现值（真实 profile 逐字段）
- full-V profile（test/Conversion/EmitC/…one-kernel-two-profiles.mlir:16-27 + test/Plugin/RVVCapabilityProfileDivergenceTest.cpp:163-165）：`isa_vector_hints="rv64gcv_zvl128b"`, `supported_sew="8,16,32,64"`, `supported_lmul="mf8,mf4,mf2,m1,m2,m4,m8"`, march=`rv64gcv`。
- zve32x profile（同 mlir:52-63 + Test.cpp:167-169）：`isa_vector_hints="rv64imac_zve32x_zvl128b"`, `supported_sew="8,16,32"`(无64), march=`rv64imac_zve32x`。
- rvv07 profile（Test.cpp:253-255）：`isa_vector_hints="rv64gc_xtheadvector"`, march=`rv64gc_xtheadvector`。
- makeBaseFacts 公共值（Test.cpp:56-67）：architecture="riscv64", hartCount=4, **vlenbBytes=16**, clangVersion="clang version 20.0.0", cmakeVersion="cmake version 3.28.0", selectedMABI="lp64d"。
- **scalar（无V）板无对应 RVV provider fixture**：其能力形态 = **不存在 rvv-kind provider**（board册.md:16 无 v 扩展）；vector-absent 实例住 load-time capability resolution / [F-6] independent-family 语境（CapabilityModel.h:194-211 impliedClosureAvoidsRVVNamespace）。**采不到：无 scalar 板逐字段能力属性表**（只有"无 v 扩展"这一否定事实）。

### 3d. 采不到清单（板实例）
- **无单一 per-board 能力实例权威文件**（JSON/MLIR）：现值三处分散（board册 prose + fixture + pass-option），机算未见统一 board profile 台账。
- k1 的 `vreg_count` / `cacheline` / `elen` / `sew_set` 具体值：**采不到**（board册未列，fixture 未含，schema namespace 声明但代码零落）。
- scalar 板逐字段能力属性：**采不到**（仅"无 v 扩展"否定事实）。
- rvv07 板：REGISTERED-PENDING·零工作（板册.md:12），无 fixture 值。

---

## 4. 定点核查（逐点 file:line 或"采不到"）

### 4a. 寄存器总数（vreg_count / vreg 数）
- schema 声明：`vreg_count` int hardware-fact（capability.schema.v1.json:147-150）。
- 代码侧落地：**采不到**。机算 `git grep -c 'vreg_count' d173f4… -- lib include tools` → 仅 schema JSON 命中；**无 struct 字段、无 property 键、无消费点**。
- 三板实例值：**采不到**（无任何 board 列 vreg 数）。
- ⟹ 寄存器总数 = schema-only 声明，进料/消费全空。

### 4b. 分数 LMUL 存在性
- **无独立 boolean 字段**。分数 LMUL 存在性**内嵌**在 `supported_lmul` allow-list 字符串子串里：RVV1.0 含 `mf8,mf4,mf2`（存在），RVV0.7 只有 `m1,m2,m4,m8`（不存在）。
- 派生处 `deriveSupportedLMULAllowList`（RVVCapabilityProfile.cpp:260-277）：RVV0.7 分支原文注"NO fractional LMUL exists on this generation"（263-266）；RVV1.0 分支给全格含 mf 档（267-270）。
- header 原文（RVVCapabilityProfile.h:127-142）：RVV0.7.1 XuanTie header"declares ZERO mf2/mf4/mf8 types"。
- ⟹ 分数 LMUL 存在性 = `supported_lmul` 子串，非独立字段。

### 4c. 整数倍 LMUL 合法集
- **无独立字段**。整数倍 LMUL 合法集**内嵌**在同一 `supported_lmul` 串："m1,m2,m4,m8"（两代通用整数档）。派生 RVVCapabilityProfile.cpp:260-277。
- 消费：`supported_lmul` 由 EmitC 门查（RVVToEmitC.cpp:2298-2302 provider 取 supported_lmul 排除 typed body LMUL）+ RVVEmitCRoutePlanning.cpp:244（kSupportedLMULPropertyName）。

### 4d. 向量单元有无（vector unit presence）
- **无独立 boolean "has_vector" 字段**。向量单元有无 = **rvv-kind provider 的存在/缺席**：
  - 有 V：kernel 里存在 `id="rvv"`/`kind="isa-vector"` 的 provider op（isRVVCapabilityProvider，RVVProbedCapabilityAxesMaterialization.cpp:56-66；hasAvailableRVVCapability 门 RVVExtensionPlugin.cpp:637）。
  - 无 V（scalar）：**不生成 rvv provider**（board册.md:16 无 v 扩展）→ RVV 门 supportsOperation 返 false（RVVExtensionPlugin.cpp:636-638）。
- 探测侧有向量证据校验 `hasRVVVectorHint`（RVVCapabilityProfile.cpp:140-162）+ `validateRVVProbeCapabilityFacts` 要求"ISA/vector hint must contain RVV vector evidence"（.cpp:446-447）。
- ⟹ 向量单元有无 = provider 存在性建模，非字段。

---

## 5. ★消费核查（逐字段：今天被哪些 f / 哪些代码读）

### 5a. 有消费的 c 字段（消费点清单 file:line）
| c 字段/property | 消费点（f） | file:line |
|---|---|---|
| `supported_sew` | EmitC 合法性门（排除 typed body SEW） | RVVToEmitC.cpp:2290-2294 |
| `supported_sew` | route planning 门 | RVVEmitCRoutePlanning.cpp:242, 271 |
| `supported_lmul` | EmitC 合法性门 | RVVToEmitC.cpp:2298-2302 |
| `supported_lmul` | route planning 门 | RVVEmitCRoutePlanning.cpp:244, 280 |
| `rvv_version` | EmitC 门（rvv_version=0.7 gate 掉 ta/ma agnostic body） | RVVToEmitC.cpp:2309-2321 |
| `architecture` | route planning | RVVEmitCRoutePlanning.cpp:139 |
| `isa_vector_hints` | route planning | RVVEmitCRoutePlanning.cpp:147, 151 |
| `hart_count` property `count` | HartParallelCapabilities（线程能力） | HartParallelCapabilities.cpp:45, 66, 92, 104, 178 |
| `count`/`target.hart_count` id | HartParallelCapabilities collectProvidersByID | HartParallelCapabilities.cpp:66 |
| `vlen_bits`（provider property） | **IME plugin** 合法性门 | IMEExtensionPlugin.cpp:332, 337 |
| `vlen_bits` | **IME backend emission driver**（每 fragment vreg cost） | IMEBackendEmissionDriver.cpp:426, 496, 505, 520 |
| IME `march` | IME plugin 门（token `xsmtvdotii`） | IMEExtensionPlugin.cpp:322 |
| IME `available_harts` | IME plugin 门 | IMEExtensionPlugin.cpp:515 |
| IME `ime_signedness` | IME plugin 门 | IMEExtensionPlugin.cpp:346 |
| IME `ime_matmul_shape` | IME plugin 门 | IMEExtensionPlugin.cpp:390 |
| IME `ime_weight_format` | IME plugin 门 | IMEExtensionPlugin.cpp:440 |
| IME `ime_slide` | IME plugin 门 | IMEExtensionPlugin.cpp:486 |

### 5b. 消费-但-无 producer（gate-only；探测侧不产出，属手写 fixture attr）
| property | 消费点 | producer 机算 |
|---|---|---|
| `required_tail_policy` | RVVEmitCRoutePlanning.cpp:246, 288（门） | `git grep 'required_tail_policy' … -- lib include` 仅命中该 gate 文件（RVVEmitCRoutePlanning.cpp:64,288）；探测产出函数不产出 → 无 producer |
| `required_mask_policy` | RVVEmitCRoutePlanning.cpp:248, 296（门） | 同上（RVVEmitCRoutePlanning.cpp:66,296）；无 producer |

### 5c. ★零消费字段（进了料没人用）
逐条附机算枚举（严禁 head/tail；用 `git grep -n` 全路径 + 排除 `_attic`/worktrees/build）。

| # | 零消费字段 | 机算枚举 | 结论事实 |
|---|---|---|---|
| 1 | `RVVProbeCapabilityFacts::cachelineBytes` | `git grep -n 'cachelineBytes\|cacheline' d173f4… -- lib include tools schema \| grep -v _attic` → 命中：RVVCapabilityProfile.h:65/71（声明+注释）、RVVRepackTilingSelection.h:438（注释"known GAP; once plumbed"）、schema:151（namespace 声明）。**无任何 setAttr/getProperty/selector 读点** | 结构体字段存在，探测产出函数不产出 fact，无消费者 |
| 2 | `RVVProbeCapabilityFacts::imePresent` | `git grep -n 'imePresent\|\.imePresent'` → 仅 RVVCapabilityProfile.h:73/77（声明+注释）。无 setter（除结构体默认）、无 reader | 字段存在，零产出零消费 |
| 3 | `deriveIMEPresent(...)`（inline fn） | `git grep -n 'deriveIMEPresent'` → 仅 RVVCapabilityProfile.h:73(注释)/194(定义)。**零调用者** | 函数已建，无调用点 |
| 4 | `rvv.vlenb_bytes` property `bytes`（vlenbBytes） | `git grep -n '"bytes"\|vlenb_bytes\|getRVVVLenBBytes' … -- lib include tools \| grep -v _attic` → 命中全在**产出侧**（RVVCapabilityProfile.cpp:28-30,385-391,562-568 + header 声明）。**无 getProperty("bytes") 或任何读点** | provider op 产出该 property，无消费者 |
| 5 | `rvv.toolchain.clang` property `version`（clangVersion） | `git grep -n 'toolchain.clang\|getRVVClangToolchain'` → 仅产出侧（RVVCapabilityProfile.cpp:31-34,393-399,569-573 + header）。无读点 | 证据线产出，零消费 |
| 6 | `rvv.toolchain.cmake` property `version`（cmakeVersion） | 同 5 模式（.cpp:35-38,401-407,574-578）。无读点 | 证据线产出，零消费 |
| 7 | `rvv.probe.compile_run` 属性 `selected_march`/`selected_mabi`/`source_sha256`/`binary_sha256` | `git grep -n 'probe.compile_run\|source_sha256\|binary_sha256'` → 仅产出侧（.cpp:39-42,409-415,580-592）。无读点 | 证据线产出（4 键），零消费 |
| 8 | `rvv.toolchain.march` property `value`（selectedMarch，**作为 provider property**） | `git grep -n 'toolchain.march\|getRVVSelectedMarch'` → 仅产出侧（.cpp:43-46,417-423,594-598）。无 provider-property 读点 | **注**：selectedMarch 作为 **pass -march 字符串**被大量消费（见 §6），但**作为 provider op property**（键 `value`）零消费 |
| 9 | `rvv.toolchain.mabi` property `value`（selectedMABI） | `git grep -n 'toolchain.mabi\|getRVVSelectedMABI'` → 仅产出侧（.cpp:47-50,425-431,599-605）。无读点 | 证据线产出，零消费 |

### 5d. 整条探测→capability 产出路径的 driver 挂接状态（事实）
- 自由函数 `buildRVVTargetCapabilitiesFromProbeFacts` 调用点：`git grep -n 'buildRVVTargetCapabilitiesFromProbeFacts' … -- lib include tools \| grep -v _attic` → 命中 2：定义（RVVCapabilityProfile.cpp:487）+ 插件方法内调用（RVVExtensionPlugin.cpp:665）。
- 插件方法 `RVVExtensionPlugin::buildTargetCapabilitiesFromProbeFacts` 调用点：`git grep -n 'buildTargetCapabilitiesFromProbeFacts' … -- lib include tools test` → 命中仅 声明（RVVExtensionPlugin.h:43）+ 定义（RVVExtensionPlugin.cpp:663）。**lib/tools 内无外部 driver 调用**；唯一实际驱动 = 测试直调自由函数（test/Plugin/RVVCapabilityProfileDivergenceTest.cpp:171 等）。⟹ **探测→capability-op 产出路径在本 pin 无生产驱动挂接**（事实照录，不判断）。

---

## 6. 表外读板信息的旁路（逐处列：读什么·在哪读·进没进 schema）

★这是 c 进料的**最大旁路面**：真实"板 VLEN"值在多数 f 里**不经能力表 provider op**，而经 **pass `-march` 字符串**重解析。

### 6a. march / isa-vector-hints 作为 PASS OPTION（非 in-IR fact）
- 定义：`Option<"march",…std::string,default="">` + `Option<"isaVectorHints","isa-vector-hints",…>` 在 include/Weft/Transforms/Passes.td 多处（:337-340, :398-403, :421-425, :569-573）。
- 读法：各 front-door/schedule pass 从**自己的 pass option** 取 march 字符串，喂 `deriveMinimumVLEN(march,hints)` 现场重解析出 VLEN bits——**不读 capability provider op**。
- 进 schema？**否**。这是原始 march 串，非 schema fact_record/params。t1d 自身注（t1d_dual_instance.py:457）原文："march= is a PASS OPTION, not an in-IR fact"。

### 6b. `deriveMinimumVLEN` 消费点清单（march 串 → VLEN，绕过表）
定义 RVVCapabilityProfile.cpp:289-332。以下 f 现场调用它（从 pass option march 取值）：
| 消费 f | file:line |
|---|---|
| RVVLowerQuantContraction（accLmul/contraction 选择） | RVVLowerQuantContraction.cpp:1407, 1544 |
| RVVMonolithicBlockDotSourceFrontDoor | RVVMonolithicBlockDotSourceFrontDoor.cpp:3395 |
| RVVDequantDotSourceFrontDoor | RVVDequantDotSourceFrontDoor.cpp:376 |
| RVVReductionSourceFrontDoor | RVVReductionSourceFrontDoor.cpp:345 |
| RVVPackedI4DotSourceFrontDoor | RVVPackedI4DotSourceFrontDoor.cpp:209 |
| RVVCodebookDotSourceFrontDoor | RVVCodebookDotSourceFrontDoor.cpp:210 |
| RVVScheduleDescriptorRegistry | RVVScheduleDescriptorRegistry.cpp:457 |
| RVVRepackStripWidthMaterialization | RVVRepackStripWidthMaterialization.cpp:100 |
| `deriveHasZvl128b`(内部再调 deriveMinimumVLEN) | RVVCapabilityProfile.cpp:338；消费 RVVScheduleDescriptorRegistry.cpp:458 |

### 6c. IR 类型宽度/属性读值旁路（in-IR attr 镜像 c 值）
- `minimum_vlen` / `min_vlen` op-attr：schedule descriptor 盖章（RVVScheduleDescriptorRegistry.cpp:189,257,289,322,357,388,407 `descriptor.minimumVLENAttrName="minimum_vlen"`）；verifier 读取，缺省默认 128（RVVDialectWideningOps.cpp:5226 原文"The semantic input is the `minimum_vlen` attr (the deriveMinimumVLEN capability fact); absent, it defaults to 128"）。
- RVVOps.td:4295 原文：`min_vlen` = "a deriveMinimumVLEN(march) value 的 snapshot，consumed by a later …"（op-attr 承载 c 派生值）。
- 进 schema？**否**（是 RVVOps.td 上的 op 属性，非 capability schema）。这是把 c 值（VLEN）**镜像到 op-attr** 的旁路，与能力表 provider property 并行。

### 6d. 材料化 pass = 旁路↔表的桥（把 pass-option march 盖回 provider op）
`RVVProbedCapabilityAxesMaterialization`（lib/Plugin/RVV/Schedule/RVVProbedCapabilityAxesMaterialization.cpp）：从 **pass option** march/isaVectorHints（§6a）派生 supported_sew/supported_lmul/rvv_version，**盖到 in-IR rvv provider op**（:98-126，materializeAxis），使 §6a 的 pass-option 路径与 §5a 的 provider-property 路径**收敛**。已存 hand-authored fixture attr 不覆盖（:73-82 materializeAxis 先查 hasAttr）。进 schema？盖的是 provider op property（S4 表面），非 schema 声明。
- `RVVRepackStripWidthMaterialization`（同目录）：类似，把 deriveMinimumVLEN 派生的 strip 宽盖到 op（:100-109）——注释自述"authority(deriveMinimumVLEN)是 source of truth，op attribute 是 mirror"（:18-20）。

### 6e. 探测侧 march 串解析函数群（都吃 march/hints 字符串，非表）
均在 RVVCapabilityProfile.cpp，输入 = `(selectedMarch, isaVectorHints)` 两串：
- `hasRVVVectorHint`（:140-162）：扫 zve/zvl/zvfh/gcv/xtheadvector/rv64…v 判有无向量。
- `containsIsaToken`（:124-138）：token 边界匹配（zvfh 不误命中 zvfhmin 前缀）。
- `deriveSupportedSEWAllowList`/`deriveSupportedLMULAllowList`/`deriveMinimumVLEN`/`deriveHasZvl128b`/`deriveRVVVersion`/`deriveIMEPresent`：全是 march 串解析。
- 进 schema？否——这些解析的**输入**是 march 串（pass-option 或探测 fact），**输出**才盖成 provider property。

---

## 7. 计数汇总（纯数字 + 口径 · 无解读）

- 口径A｜schema params namespaces（item_4）：**7**（vlen,elen,sew_set,lmul_budget,vreg_count,cacheline,ime.tile）；`$meta` 自述全 grep=0 in code。
- 口径B｜schema fact_record 字段（item_1）：**9**；其中 declared-no-producer = **3**（subclass, provenance, trust）。
- 口径C｜RVVProbeCapabilityFacts struct 输入字段：**15**；其中零消费 = **2**（cachelineBytes, imePresent）+ `deriveIMEPresent` 零调用者。
- 口径D｜RVV 探测产出的 provider-property 发射：**15 property 键 across 9 provider op**；被 f 消费 = **6**（architecture, isa_vector_hints, supported_sew, supported_lmul, rvv_version, hart_count:count），零消费 = **9**（vlenb:bytes, clang:version, cmake:version, compile_run 的 4 键, march:value, mabi:value）。
- 口径E｜IME provider 被消费 property：**7**（march, vlen_bits, available_harts, ime_signedness, ime_matmul_shape, ime_weight_format, ime_slide）；另 gate-only 无 producer = **2**（required_tail_policy, required_mask_policy）。
</content>
</invoke>


<!-- ========== 包3 节 (源: pkg3-g-descriptor.md) ========== -->

# 包3 · g 进料口盘点(描述符) — θ=f(g,c) 三角色全景盘点 v2

**HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（`git rev-parse HEAD` 实测 == 令定 ref；工作树 == HEAD）
只读·零施工·零板时·事实+出处。是不是缺陷由上游判。

## 0 · 口径 / pin 漂移声明（读数前必看）

- 本包引用的 census 现成数据（`.trellis/tasks/archive/2026-07/07-19-aline-census-classify/research/aline-census-head.md` 等）自钉 HEAD = `f457e73371b1f714d7e4886ca8859392613e457f`，落后本 pin **19 commits**（`git rev-list --count`）。
- 描述符 td（`include/Weft/Dialect/RVV/IR/RVVOps.td`）在两 pin 间 **byte-identical**（`git diff --stat f457..d173 -- <td>` = 空）⟹ 本包所有 td file:line 在本 pin 有效。
- 发射器侧文件有漂移：`RVVToEmitCTernaryBinary.cpp` 在 f457..d173 间 **+220/-249（469 行）**（`git diff --stat`）；`GridCodebook`/`ForwardElementwise`/`KQuant`/`Internal.h` 无漂移。⟹ Ternary 的反向计数我在本 pin 机核重取（见 §4），其余照 census。
- 全文 file:line = 本 pin 工作树读数，除非显式标 `@f457`。

---

## 1 · 描述符全字段总览（RVVOps.td · 本 pin）

- `grep -c OptionalAttr <td>` = **165**（OptionalAttr 声明数）。
- 机算抽全部 attr 声明名（含 required `I64Attr:$x` / `StrAttr:$x` / `BoolAttr:$x` / `OptionalAttr<...>` / `DenseI8ArrayAttr` / `FlatSymbolRefAttr` / `WEFTRVV_PolicyAttr`）后去重 = **221 个 distinct attr 名**（`scratchpad/attrs.txt`）。这是「描述符全字段」的机算全集。

按三角色 + 元数据四分（分类为本 agent 依名义/文档判读·非上游裁定）：

| 桶 | 代表字段（非穷举） | 大致处 | 说明 |
|---|---|---|---|
| **g（格式侧结构量）** | qk / sub_block / *_block_stride / *_byte_offset / num_groups / group_lanes / half_lanes / weight_interleave / codebook / table_symbol / format / quant | 见 §2（64 个名） | 应住描述符·本包主对象 |
| **θ（输出/决定·却住在描述符）** | integer_core_lmul / strip_lmul / multi_block_factor / strip_elision / fold_structure / numerics_tier / unroll_factor / accumulator_lmul / product_lmul / result_lmul / *_sew / emit_loop_schedule / column_group_tile / half_lanes(值=strip 宽) | 见 §3.θ | **θ 焊/盖在描述符里**·旧称「c 轴烘焙」实为此类 |
| **c（板侧能力量·住在描述符）** | min_vlen / minimum_vlen / opponent_vlen_native_floor / required_capabilities / vector_register_budget / peak_live_vector_groups | 6 处名 | c 本应住能力表；描述符里的 min_vlen 由 td 明标「read-only **advisory**」（RVVOps.td:4220 区 GgmlQuantContractionOp doc） |
| **元数据/契约/关系角色（verifier 用）** | planning_contract / remediation_* / performance_admission_* / resource_cost_* / schedule_decision_* / purpose / origin / status / *_relation / *_role / *_memory_form / *_scope / from_phase / to_phase | 余数（~130+） | 多为 Typed*PreRealizedBodyOp 的结构角色 + GearboxCrossRegionHandoffOp 的规划日志 |

> **本包只对 g 桶做逐字段消费核查 + 反向 + 定点**（§2–§5）。θ/c/元数据桶给计数与代表·消费核查见其它包（θ 焊死清单 = 包1/2）。

---

## 2 · g 桶字段清单（64 个 distinct 名·`scratchpad/gfields.txt`）

### 2.1 名 / 类型 / 分组

机算谓词（byte_offset$ / _stride$ / qk / sub_block / groups / lanes / interleave / codebook / table_symbol / format / quant / layout / xor_mask / offset_bias）命中 **64 个 distinct g 字段名**。分组：

- **块字节 stride（7）**：`weight_block_stride` `activation_block_stride` `block_stride` `src_block_stride` `dst_block_stride` `lhs_block_stride` `rhs_block_stride`（全 I64Attr，多为 required；`block_stride` 在 LoadOp/一些 op 为 OptionalAttr）。
- **区偏移 byte_offset（34）**：`weight_qs_byte_offset` `weight_qh_byte_offset` `weight_d_byte_offset` `weight_dmin_byte_offset` `weight_scales_byte_offset` `weight_scales_h/l/high_byte_offset` `weight_hmask_byte_offset` `weight_gas_byte_offset` `weight_ls_byte_offset` `weight_sign(s)_byte_offset` `weight_min_byte_offset` `weight_quant_byte_offset` `weight_scale_byte_offset` `weight_offset_bias` / `activation_{d,quant,bsums,high,sum,scale}_byte_offset` / `bsums_byte_offset` `qh_byte_offset` `quant_byte_offset` `scale_byte_offset` `sum_byte_offset` / `lhs_{scale,min}_byte_offset` `rhs_{scale,sum}_byte_offset` / `src/dst_quant_byte_offset`（全 I64Attr）。
- **块/子块几何（14）**：`qk` `qk_sub` `sub_block` `num_sub_blocks` `n_subblocks` `num_groups` `groups_per_sub` `group_lanes` `half_lanes` `num_groups_per_half` `indices_per_sub_block` `signs_per_sub_block` `num_lanes` `activation_blocks_per_weight`。（`qk`/`sub_block` = required I64Attr；阶段2 新增的 `num_groups`/`groups_per_sub`/`group_lanes`/`num_lanes`/`indices_per_sub_block`/`signs_per_sub_block`/`num_groups_per_half` = OptionalAttr<I64Attr>·见 §2.3。）
- **交织宽（2）**：`weight_interleave`（I64Attr @4603/4715/4794/4901/5009/5040/10756/10912/11843）`activation_interleave`（I64Attr @5010/11844）。
- **码本/格式身份（7）**：`codebook`（**DenseI8ArrayAttr** @4048/5307/5433/7990/11546/11608·恰 16 int8 entry）`table_symbol`（StrAttr @4049）`quant`（OptionalAttr<StrAttr> @4306）`format`（StrAttr @9589）`weight_layout`（StrAttr @4310）`packing_layout`（StrAttr @404）`xor_mask`（I64Attr @4795）。
- **附注（未计入 64、边界 g-身份 StrAttr）**：`scale_model`（StrAttr·~30 op）`decode_model`（StrAttr·~10 op）`scale_role`（StrAttr @830/1618/1679/1744）`kind`——格式-解码身份串，被 f 当**族选择主键**读（见 §3）。

### 2.2 哪些格式盖了值 / 由前门哪行盖 / g 源

**成熟 g 管线（block-dot 家族）** —— g 值不焊在发射器，而是：

1. **g 源正本（per-format 常量表）** = `include/Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h:1503-1526` —— **24 张 `k<Format>Facts[]` constexpr 表**（`kQ4KFacts`/`kIQ4XSFacts`/`kIQ1SFacts`/…/`kQ10Facts`），每张是 `MonolithicBlockDotI64Attr{name,value}` 列表。例：
   - `kQ4KFacts` = `{"qk",256}{"sub_block",32}{"weight_block_stride",144}{"activation_block_stride",292}{"weight_d_byte_offset",0}{"weight_dmin_byte_offset",2}{"weight_scales_byte_offset",4}{"weight_qs_byte_offset",16}…`（RVVMonolithicBlockDotFamily.h:1503）
   - `kQ2KFacts` 内含 `{"weight_d_byte_offset",80}{"weight_dmin_byte_offset",82}`（:1512）—— 注意此二值在 block-dot 家族**是描述符化的**；同二值在 dequant 标量 fold 路径**却焊死**（§4 KQuant #5/#6·ISSUE-115·两路架构不同）。
2. **前门盖章** = `lib/Plugin/RVV/FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp`：
   - `factByName` lambda（:621-626）按名从 `entry.facts` 取值，缺则 `llvm_unreachable("…missing block-format fact")`。
   - `state.addAttribute(<name>, builder.getI64IntegerAttr(<val>))` 盖到 op：`block_stride` @323、`quant_byte_offset` @325、`lhs/rhs_block_stride` @483/485、`lhs_min/rhs_sum_byte_offset` @508/510、`qh_byte_offset`+`block_stride` @536/537、`qk`+`weight_block_stride` @678/679、`qk`+…+`activation_blocks_per_weight` @912/956、`qk_sub` @1063…（全文 g-attr stamp 字符串 `grep -c` = **395**）。
3. **发射器读** = typed getter（§3 category ii）。

⟹ block-dot 家族的 g 是「**声明一次(family header)→前门 stamp→发射器 fail-closed 读**」。24 格覆盖 = kQ4K/Q2K/Q3K/Q5K/Q6K/IQ4XS/IQ1S/IQ1M/IQ2XXS/IQ2XS/IQ2S/IQ3XXS/IQ3S/TQ10/TQ20/Q40/Q80/IQ4NL/Q41/Q50/Q51/MXFP4/NVFP4/Q10。

**dequant-stream 家族的 g（另一路·部分焊死）** —— 见 §4 反向（ForwardElementwise dequant-row·GridCodebook grid 几何·KQuant fold 偏移不走此前门）。

### 2.3 阶段2 新增字段（OptionalAttr<I64Attr>·fail-closed 范式）

commit `85cd5022f`（census §1.1 记）把 KQuant 17 处 grid/subblock **焊死**常量换成描述符读，新增 OptionalAttr<I64Attr>：`groups_per_sub`(@7149/7250/8283) `num_groups`(@7348/7476/8407) `indices_per_sub_block`(@7349/8408) `group_lanes`(@7350/8410) `num_groups_per_half`(@8151/8284) `signs_per_sub_block`(@8409) `num_lanes`(@6435/6513) `n_subblocks`(@10782/10927/11870) `weight_qh_byte_offset`(OptionalAttr @6044/10661/10768/11117/11856) `weight_offset_bias`(@10662/11118) `weight_dmin/scales_byte_offset`+`activation_bsums_byte_offset`(@10779-82) `weight_scales_high_byte_offset`(@10791) `lhs/rhs_*`(@10080-83/10138-41) `qh_byte_offset`+`block_stride`(@10193-94) `activation_quant_byte_offset`(@6237)。范式正本 = LoadOp（RVVOps.td:2645 doc 明写「block_stride/quant_byte_offset must not be set, **fail-closed**」·OptionalAttr @2677-2678）。

---

## 3 · g 逐字段消费核查（i 读入 f / ii 发射器直读 / iii 零消费）

**方法**：为 221 个 attr 生成 typed getter（`get<CamelCase>`），机扫 `lib/ tools/ include/`（排 `.inc` 生成物）命中文件数（`scratchpad/consumption.txt`）。

### 3.1 g 桶结论：64 个 g 字段 **无一落零消费**

`comm -12 gfields.txt <(零 getter 名)` = **空**。⟹ **g 桶 (iii) 零消费 = 0**：所有 g 字段都至少有一个 typed getter 读者。「搬了个寂寞」在 g 桶为空。

### 3.2 (i) 被某条 f 读（参与算 θ）——具名

两条 θ-算子 f 的 op-attr 读（`grep -oE '\.get[A-Z]\w+\(\)'` 计次）：

- **f₁ = `selectRepackAccumulatorLMUL` / `selectContractionAlgorithm` / `stampScheduleSelections`**（`RVVLowerQuantContraction.cpp`）读：
  - `getScaleModel()` **68×**（族/算法选择主键·f₁ 首参 `llvm::StringRef scaleModel` @1286）—— g-身份串。
  - `getQk()` **19×**、`getMRegime()` **4×**（regime 轴·θ 选择输入）。
  - `getOpponentVlenNativeFloorAttr()` **1×**、`getBlockDotComputeHeavy()` **1×**、`getBlockDotMemoryBound()` **1×**——「结构化对手事实」（measurement/对手 regime·非纯 g）→ 喂 `selectContractionAlgorithm` 的 θ。
  - `getActivationBlockStride()` 7× / `getQuantByteOffset()` 6× / `getWeightBlockStride()` 2× / `getActivationHighByteOffset()` 1×——**读来转 stamp 到子 op**（结构管线·非算 θ）。
- **f₂ = `MaterializeRVVRepackStripWidth`**（`RVVRepackStripWidthMaterialization.cpp`）读：
  - `getWeightInterleave()`（@140/144/156/160/173/177）→ `deriveRepackHalfLanes(vlenBits, weight_interleave)` → **stamp `half_lanes`（θ）+ RVV0.7 stamp `integer_core_lmul="m1"`（θ）**。⟹ **`weight_interleave`（g·交织宽）= category (i)**，是唯一真喂 θ-几何的纯 g 字段。

> **VLEN 不来自描述符**：f₁/f₂ 均从 `-march` 经 `deriveMinimumVLEN` 取 minVLEN（RVVLowerQuantContraction.cpp:9-12/1406/1544·RVVRepackStripWidthMaterialization.cpp:100）。td 明标 `min_vlen` attr「read-only **advisory**·NOT the source」（RVVOps.td GgmlQuantContractionOp doc + 注释 :1526/1540）。⟹ 描述符 c 字段 `min_vlen` = **零消费**（getMinVlen typed-getter 命中 0；string "min_vlen" 命中 1=doc/advisory）。

### 3.3 (ii) 被发射器直读（只做结构展开·不进 f）

g 桶剩余 ~62 字段（全部 byte_offset / stride / 子块几何 / codebook / format-身份）——发射器 typed getter 直读，emit 成指针算术 / vsetvli / gather 结构，**不参与任何 θ 决策**。例：`getWeightQsByteOffset` `getWeightScalesByteOffset` `getSubBlock` `getNumGroups` `getGroupLanes` `getHalfLanes`（读值展开）`getCodebook`（DenseI8Array 结构消费）。这是 g 的**正常归宿**（结构展开）。

### 3.4 (iii) 零消费——g 桶 = 0；描述符全集另有 34 个「零 typed-getter」名（**多非 g**·如实并列）

机扫得 **34 个 attr 无 typed-getter 命中**（`scratchpad/consumption.txt`）。**其中 0 个是 g 字段**。34 个按角色：

- **c 桶**：`min_vlen`（td 明标 advisory·上 §3.2）`required_capabilities`（typed 0·但 string-ref 27=经泛型 getAttr 消费·非真零）。
- **θ 桶**：`unroll_factor`（typed 0·string 3）`selected_variant`（typed 0·string 37=经字符串消费）`selected_path_role`（string 8）。
- **元数据/契约（GearboxCrossRegionHandoffOp @365-432 的规划日志）**：`planning_contract` `resource_cost_{model,contract,blocker,loop_body_steps}` `performance_admission_{decision,closure,reopen_requirement}` `beyond_local_repair_admission_{contract,decision,blocker,reopen_requirement}` `remediation_{plan,plan_contract,statement_strategy,vector_budget,schedule_contract,unpack_plan,product_plan,reduction_plan,vl_plan}` `schedule_decision{,_contract,_reason}` `rvv_construction_protocol` `rvv_emitc_route_mapping` `purpose` `origin?` `exec_binding` `arithmetic_kind`。
  - 这些多有 string-ref（stamp 写侧；如 planning_contract string 4 / remediation_plan 6 / schedule_decision 5 / purpose 12）——**是否只写不读需逐字段 string 读/写侧三分**（本包未穷做·超 g 范围）。**如实标注：这批是「搬了个寂寞」候选，但属元数据/θ/c 桶，不属 g 桶。**

> **g 桶净结论**：进料的 g **全部被消费**（1 个走 f = weight_interleave；余 ~62 走发射器直读；scale_model/qk/m_regime 双读=既喂 f 选择又参与 emit）。g 桶零「搬了个寂寞」。

---

## 4 · 反向：发射器仍在读 g、但描述符没字段（焊死/派生残余）

**机核（本 pin·pin-shape 谓词 `int64_t NAME=<int>;` 排 `.get`/`=0`/`=1`）**：

| 文件 | 处数(本 pin) | 逐处 file:line（值） | 登记 |
|---|---|---|---|
| `RVVToEmitCGridCodebook.cpp` | **9** | 76 `subBlockLanes=32`·504 `groupLanes=8`·1002 `groupLanes=8`·1514 `halfLanes=16`·1629 `pairHalves=4`·1986 `halfLanes=16`·2102 `pairHalves=4`·2466 `groupLanes=8`(const)·2467 `numGroups=4`(const) | **ISSUE-118**（回门待扫·扇出最高） |
| `RVVToEmitCTernaryBinary.cpp` | **7**（本 pin；census@f457=8） | 854 `groupLanes=8`·869 `halvesPerSub=2`·870 `groupsPerHalf=2`·871 `halfLanes=16`·1588 `planeLanes=32`·1589 `planesPerChunk=4`·1590 `chunkBytes=32` | **ISSUE-118** |
| `RVVToEmitCKQuant.cpp` | **2** | 4607 `weightDOffset=80`·4608 `weightDminOffset=82`（q2_K·`emitTypedSuperBlockScalarScaleMinLoopBody`） | **ISSUE-115**（=「#5/#6」·专用 fold-brick op 待裁） |
| `RVVToEmitCForwardElementwise.cpp` | pin-shape **3**（2709 `qk=32`·3575 `qhOff=16`·4564 `qk=256`）；宽谓词 **~51** | 5333-5351 per-format stride switch **19 case**（Q2K84/Q3K110/Q4K144/Q5K176/Q6K210/MXFP4 str17/NVFP4 str36/TQ1 54/TQ2 66/Q10 str18/IQ4NL str18/IQ2XXS66/IQ2XS74/IQ2S82/IQ3XXS98/IQ3S110/IQ1S50/IQ1M56/IQ4XS136）+ 多变量声明（2661-2670/3259/3394/3743/3870/4048…） | **ISSUE-119**（dequant-row ~40·monolith-fallback·须独立 scoping） |

- **Ternary 本 pin = 7 ≠ census@f457 = 8**：`dotStripLanes=32`（f457 @2060·tq1_0 dot-strip）在 f457..d173 的 469 行漂移中**已从本 pin 消失**（`grep dotStripLanes` 本 pin = 空；`git grep @f457` 命中 :2060/2061/2347/2351）。⟹ 反向计数须以本 pin=7 为准（census 数据是旧 pin）。
- **KQuant #5/#6 特例（ISSUE-115）**：`weight_d/dmin_byte_offset` **不能同法加 attr**——q2_K 整数核 op（`GgmlBlockDotQ2KQ8KIntegerCoreOp`）ODS charter 明标 fp32 scalar fold「DELIBERATELY OUT OF SCOPE」；加 attr=把故意排除的 fold 拉进整数核=语义冲突。**同二值在 block-dot 家族的 `kQ2KFacts`（Family.h:1512）却是描述符化的**——两路架构不同（此为 dequant/scalar-fold 路）。ISSUE-115 状态=待裁·公式占比封顶 35/37。
- **「coreLmul 焊死」不入 g 反向**：census §0/§3.2 + ISSUE-118 订正——主会话初判「焊死 11」经宽谓词逐处核 = 5 真焊死(θ·假旋钮形态·GridCodebook 497/995·Ternary 862·BQL 111/404)+5 live-default-override(θ value_or)+1 verifier-pinned(CodebookFp4 109)。这些 coreLmul/wideLmul 是 **θ**（不是 g·也不是 c）·全 MAINTAIN([K-10]·Context 无字段/无 IR 宽度)·**归 θ 桶（包1/2）·不在本 g 反向表**。
- **Internal.h 板派生（边界·非纯 g）**：`3648 stripWidth=8`·`3649 numStrips=4`（注释自述「derived from l8」）——c 轴板派生残余（随 ISSUE-113），不计 g。

**反向 g 残余小结（本 pin）**：pin-shape 焊死 g = GridCodebook 9 + Ternary 7 + KQuant 2 = **18 处**；另 ForwardElementwise dequant-row **~51 处宽谓词** g 布局烘焙（ISSUE-119·未穷举到 100%）。全部维持 baked（byte-exact·保守默认）·转描述符须回门/独立 scoping。

---

## 5 · 定点：各 g 类 ↔ 哪个字段 / 哪类还没字段

| g 类 | 对应描述符字段 | 状态 |
|---|---|---|
| **块宽（block width）** | `qk`（I64Attr·required·32/64/128/256）；超块另 `sub_block`（16/32）+ `num_sub_blocks`/`n_subblocks` | **有字段**（成熟·Family.h `{"qk",…}` stamp） |
| **块字节 stride** | `weight_block_stride`/`activation_block_stride`/`block_stride`（+lhs/rhs/src/dst 变体） | **有字段**；但 dequant-row 路的 stride 仍焊死（§4·ISSUE-119） |
| **交织宽（interleave）** | `weight_interleave` / `activation_interleave`（I64Attr） | **有字段**·且是唯一喂 θ(strip 宽)的纯 g（§3.2 f₂） |
| **子块几何（sub-block geometry）** | `num_groups`/`groups_per_sub`/`group_lanes`/`half_lanes`/`num_groups_per_half`/`indices_per_sub_block`/`signs_per_sub_block`/`num_lanes`（阶段2 OptionalAttr) | **block-dot 已有字段**；但 **GridCodebook/Ternary grid 几何（subBlockLanes/pairHalves/planeLanes/chunkBytes 等 16 处）还没字段**（§4·ISSUE-118·同类未做的 op） |
| **码本尺寸（codebook size）** | 小码本(FP4/IQ4NL/IQ4XS 16-entry)：`codebook`=**DenseI8ArrayAttr**（数据本身·尺寸=数组长度隐含·**无独立数字字段**）+ `table_symbol` | **无「码本尺寸」数字字段**：`grep codebook_size\|table_size\|num_entries\|grid_size` = **空**（机核·NO numeric codebook-size field） |
| **grid 码本几何（iq2/iq3 super-block grid: 256/512 entry·group-of-4/8）** | —— | **完全无字段**：grid 几何全焊在 GridCodebook.cpp（§4·ISSUE-118 反向残余）·描述符里无 grid-count/grid-size 任何字段 |
| **格式身份/解码模型** | `format`/`quant`/`scale_model`/`decode_model`/`scale_role`/`weight_layout`/`packing_layout`（StrAttr） | **有字段**（scale_model 是 f₁ 族选择主键） |

**「哪类还没字段」定点结论**：
1. **码本尺寸** —— 无专用数字字段（小码本靠 DenseI8Array 长度隐含；无 codebook_size/num_entries）。
2. **grid 码本几何**（iq2/iq3/iq1_m grid + tq plane）—— 描述符零字段·全焊 GridCodebook/Ternary（ISSUE-118 待回门·同阶段2 路B 可套但未做）。
3. **dequant-row(ForwardElementwise) 全布局**（stride/qsOff/dOff/qhOff per-format）—— 描述符零字段·monolith-fallback 焊在发射体（ISSUE-119 待独立 scoping）。
4. **q2_K scalar-fold 偏移 `weight_d/dmin_byte_offset`** —— dequant 路无字段（block-dot 路 `kQ2KFacts` 有）·ISSUE-115 语义冲突·须专用 fold-brick op。

---

## 6 · 计数汇总（纯数字·口径·无解读）

- 描述符 OptionalAttr 声明 = **165**（`grep -c OptionalAttr`）；distinct attr 名全集 = **221**（去重·含 required）。
- **g 桶 distinct 字段名 = 64**（数字/几何/偏移/stride/codebook/format-身份·`scratchpad/gfields.txt`）；另 ~4 格式-身份 StrAttr（scale_model/decode_model/scale_role/kind）未计入 64。
- **g 桶零消费(iii) = 0**（64 g 字段全有 typed-getter 读者·`comm -12` = 空）。
- 描述符全集零-typed-getter 名 = **34**（**其中 g 字段 = 0**；余为 c=min_vlen 等 / θ=unroll_factor 等 / 元数据契约·多有 string-ref 写侧·真「只写不读」需逐字段三分·未穷做）。
- **g 桶消费三分**：(i) 读入 f = 主为 `weight_interleave`（→strip 宽）+ `scale_model`/`qk`/`m_regime`/opponent 三事实（→族/LMUL 选择）；(ii) 发射器直读 = 余 ~62（byte_offset/stride/几何/codebook）；(iii) 零消费 = 0。
- **反向焊死 g（本 pin·pin-shape）= 18 处**（GridCodebook 9 + Ternary 7 + KQuant 2）；另 ForwardElementwise dequant-row 宽谓词 ~51 处（ISSUE-119·未穷举）。Ternary 本 pin=7 ≠ census@f457=8（dotStripLanes 已消失）。
- g 源正本 = `RVVMonolithicBlockDotFamily.h:1503-1526`（24 张 k*Facts 表）；前门 stamp = `RVVMonolithicBlockDotSourceFrontDoor.cpp`（g-attr stamp 字符串 grep-c=395）。

## 7 · 采不到 / 未核（如实）

- 34 个零-typed-getter attr 的「只写不读 vs 泛型 string 读」逐字段三分未穷做（超 g 范围·仅抽验 required_capabilities/selected_variant=有 string 读·planning/remediation/admission=写向）。
- ForwardElementwise dequant-row ~40/~51 未穷举到每个 dOff/mOff/sub 小常量（census §Caveat 亦标·数据源自 census 宽谓词）。
- ISSUE-118/119 提「阶段2 已建同名 attr 可复用」为类比推断（GridCodebook/Ternary 是不同 op·须各自声明·census 未逐一验证 call-site 能否填 Context）——本包照引·未独立复验。
- 未编译/未跑任何 fixture（只读令）。所有「byte-exact」「消费」断言 = 静态阅读 + getter/string 机扫推断。


<!-- ========== 包4 节 (源: pkg4-f-functions.md) ========== -->

# 包4 · f 存量盘点(加工环节) — θ=f(g,c) 三角色全景盘点 v2

HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`(== HEAD·工作树即此 pin·未 checkout)

框架换算(本包统一用词):
- **g = 格式侧输入**(块宽/交织宽/子块几何/码本尺寸/fold_model/scale_model/stride/blockLen/sew…)
- **c = 板侧输入**(VLEN/vreg 数/LMUL 有无/RVV 版本/march/hints/reassoc_ok policy…)
- **θ = 输出/决定**(coreLmul/half_lanes/tiling variant/loop order/numerics tier/family/algorithm…)
- 本包只列事实(函数体原文 + 输入签名两栏)。**是否算缺陷由上游判**。

**口径声明(counts 用)**:本包把"f"分四层收录并计数——
- **Tier-1 主选择器**:读 g 和/或 c、输出 θ 决定的函数。
- **Tier-2 能力/几何派生**:读 c(或 g)输出中间量(VLEN/allow-list/half_lanes/VLMAX/footprint)喂给 Tier-1。
- **Tier-3 θ→θ 实现派生**:只读 θ(coreLmul 等)输出 θ(不读 g/c)——列出但**不计入** f 总数(既非"只 g"也非"只 c")。
- **Tier-4 焊死点/哨兵**:θ 被硬编码(假旋钮)或 emitter 纯 realize 读 θ 的 fail-closed 哨兵——列事实,不计入 f 总数。

---

## Tier-1 主选择器(读 g/c → θ)

### F1 · selectContractionAlgorithm — θ = Repack vs BlockDot(家族/算法选择)
`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp:104-172`

```cpp
ContractionSelection
selectContractionAlgorithm(const ContractionOpponentFacts &facts,
                           MRegime mRegime, std::int64_t minVLEN) {
  bool ggmlVlenNativeExists =
      facts.ggmlVlenNativeKernelFloor.has_value() &&
      minVLEN >= *facts.ggmlVlenNativeKernelFloor;
  bool repackRemovesRedundantWork =
      facts.blockDotComputeHeavy || facts.blockDotMemoryBound;
  bool selectRepack = !ggmlVlenNativeExists && repackRemovesRedundantWork &&
                      vlenOrPrefillFavorsRepack(minVLEN, mRegime, facts);
  if (selectRepack) {
    bool memoryCarried = !facts.blockDotComputeHeavy;
    if (mRegime == MRegime::Prefill)
      return {ContractionAlgorithm::Repack,
              memoryCarried ? "repack-kept-q8_0-memory-bound-prefill"
                            : "repack-kept-q4_0-prefill"};
    if (minVLEN >= 256)
      return {ContractionAlgorithm::Repack,
              "repack-kept-vlen256-decode-measured-beneficial"};
    return {ContractionAlgorithm::Repack,
            memoryCarried ? "repack-kept-q8_0-memory-bound-vlen128-decode"
                          : "repack-kept-q4_0-vlen128-decode"};
  }
  if (ggmlVlenNativeExists)
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-q4_K-vlen-native-exists"};
  if (!repackRemovesRedundantWork)
    return {ContractionAlgorithm::BlockDot,
            "block-dot-decline-lean-no-repack-benefit"};
  if (mRegime == MRegime::Decode && minVLEN >= 256)
    return {ContractionAlgorithm::BlockDot,
            facts.vlen256DecodeRepackBeneficial.has_value()
                ? "block-dot-decline-vlen256-decode-measured-negative"
                : "block-dot-decline-vlen256-decode-unmeasured"};
  return {ContractionAlgorithm::BlockDot,
          "block-dot-decline-no-repack-capability"};
}
```

| 输入 | g 字段(格式侧) | c 字段(板侧) | 其它 |
|---|---|---|---|
| `facts.ggmlVlenNativeKernelFloor` | 格式的 VLEN-native 对手 floor(结构 fact·op attr) | — | |
| `facts.blockDotComputeHeavy` | 格式 roofline compute-heavy(结构 fact) | — | |
| `facts.blockDotMemoryBound` | 格式 roofline memory-bound(结构 fact) | — | |
| `facts.vlen256DecodeRepackBeneficial` | 由 scaleModel 查 board 表(见 F19,g-key) | 板测结果(board fact) | |
| `mRegime` | — | — | regime(WHAT 轴·Decode/Prefill) |
| `minVLEN` | — | 板 VLEN(deriveMinimumVLEN) | |

- **θ 输出**:`ContractionAlgorithm{Repack,BlockDot}` + audit reason token。
- **消费点**:`RVVLowerQuantContraction.cpp:1546` `selection = selectContractionAlgorithm(facts,*mRegime,minVLEN)`;`selection.algorithm==Repack` 驱动 `isRepack` 分支(:1560)。
- 注:facts 由 `readOpponentFacts`(RVVLowerQuantContraction.cpp:1333)从 op 的 structured attrs 读入;头文件注释明言"selector blind to the format label"(RVVContractionPathSelection.h:50)。

### F2 · vlenOrPrefillFavorsRepack(F1 内部 fact-3 helper)
`lib/Plugin/RVV/Selection/RVVContractionPathSelection.cpp:90-100`

```cpp
bool vlenOrPrefillFavorsRepack(std::int64_t minVLEN, MRegime mRegime,
                               const ContractionOpponentFacts &facts) {
  if (mRegime == MRegime::Prefill)
    return true; // (a) prefill amortizes -- unchanged
  if (minVLEN == 128)
    return true; // (b) VLEN128 decode -- unchanged (rvv deployed cell, zero drift)
  if (minVLEN >= 256)
    return facts.vlen256DecodeRepackBeneficial.value_or(false);
  return false;
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `facts.vlen256DecodeRepackBeneficial`(见 F19) | `minVLEN` | `mRegime` |

- **θ 输出**:bool(fact-3:VLEN/regime 是否favor repack)。消费点:F1 内。

### F3 · selectRepackAccumulatorLMUL — θ = m1 vs mf2 累加器链 ★[GAP-P1] 三层
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1285-1304`

```cpp
inline RepackAccumulatorLMULChoice
selectRepackAccumulatorLMUL(llvm::StringRef scaleModel, bool isRVV0p7,
                            std::int64_t capabilityHalfLanes) {
  if (isRVV0p7)
    return {/*useM1=*/true, "correctness-rvv0p7"};
  constexpr std::int64_t kVectorRegisterBudget = 32;
  const std::int64_t m1ChainRegisterFootprint =
      pluginrvv::getRVVLMULRegisterFootprint("m2") +
      pluginrvv::getRVVLMULRegisterFootprint("m4");
  const bool m1Constructible = capabilityHalfLanes != 0 &&
                               m1ChainRegisterFootprint <= kVectorRegisterBudget;
  if (m1Constructible)
    if (std::optional<bool> measuredM1Faster =
            lookupRepackMeasuredM1Faster(scaleModel))
      return {/*useM1=*/*measuredM1Faster, "measured"};
  // [GAP-P1]: default mf2 -- never blind-widest; only a board measurement flips.
  return {/*useM1=*/false, "capability-default-mf2"};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `scaleModel`(格式 WHAT·仅传入 lookupRepackMeasuredM1Faster) | `isRVV0p7`(RVV 版本) | budget=32(常量) |
| — | `capabilityHalfLanes`(=deriveRepackHalfLanes,VLEN 派生) | footprint 常量 m2+m4 |

- **θ 输出**:`{useM1, reason∈{correctness-rvv0p7, measured, capability-default-mf2}}`。
- **消费点**:11 处调用(RVVLowerQuantContraction.cpp:1764/1956/2166/2312/2493/2639/2821/2975/3157/3295/3473/3621/3814…),stamp 为 inert audit attr `weft_rvv.repack_accumulator_lmul_selection_reason`;注释 :1266-1268 明言"EmitC emitter never reads it, so the default emit stays byte-exact"。
- **[GAP-P1] 三层原文照抄**(:1254-1262):
  > `[GAP-P1] IRON RULE -- widen-to-m1 was FALSIFIED TWICE (micro win washes at e2e / regfile spill). We do NOT blind-select the wider m1 ... The DEPLOYED default stays mf2; ONLY a per-format BOARD MEASUREMENT recording m1-faster for this (format, board) flips it (reason "measured"). The measured table is EMPTY today ... so every RVV1.0 format resolves to mf2 => BYTE-EXACT with the pre-selector emit`

### F4 · lookupRepackMeasuredM1Faster(F3 的 measured 表·EMPTY)
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1276-1279`

```cpp
inline std::optional<bool>
lookupRepackMeasuredM1Faster(llvm::StringRef /*scaleModel*/) {
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| `scaleModel`(参数名注释掉·**未读**) | — |

- **θ 输出**:恒 `std::nullopt`(measured 表今为空)。注释 :1271-1275:"STAGE THREE populates this ... EMPTY today: nullopt for every format => the mf2 default holds"。
- 事实:此函数签名带 g 参数但函数体不读任何输入(参数注释掉)。

### F5 · classifyTilingBottleneckShape — g → 瓶颈形状(SP4 选择 KEY)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:113-127`

```cpp
inline std::optional<RVVTilingBottleneckShape>
classifyTilingBottleneckShape(llvm::StringRef foldModel) {
  if (foldModel == "kquant_dmin_bsums_min")
    return RVVTilingBottleneckShape::MinFoldRegisterCliff;
  if (foldModel == "kquant_single_scale_no_min")
    return RVVTilingBottleneckShape::DualPlaneWeightBound;
  if (foldModel == "codebook_flat_single_scale" ||
      foldModel == "codebook_superblock_signed6_no_min" ||
      foldModel == "codebook_flat_e8m0_scale" ||
      foldModel == "grid_sign_single_scale_eighth" ||
      foldModel == "grid_sign_dualscale_eighth" ||
      foldModel == "lane_wise_vector_scale")
    return RVVTilingBottleneckShape::AlreadyLean;
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| `foldModel`(格式 fold 结构 WHAT) | — |

- **θ 输出**:`optional<RVVTilingBottleneckShape>{MinFoldRegisterCliff, DualPlaneWeightBound, AlreadyLean}`;nullopt=无 SP4 tiling 轴。
- **消费点**:`RVVLowerQuantContraction.cpp:1422` `shape = classifyTilingBottleneckShape(foldModel)`。**仅读 g**。

### F6 · priorTilingVariantForShape — shape(g派生) → variant 默认
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:131-141`

```cpp
inline RVVRepackTilingVariant
priorTilingVariantForShape(RVVTilingBottleneckShape shape) {
  switch (shape) {
  case RVVTilingBottleneckShape::MinFoldRegisterCliff:
    return RVVRepackTilingVariant::S6Tiled;
  case RVVTilingBottleneckShape::DualPlaneWeightBound:
  case RVVTilingBottleneckShape::AlreadyLean:
    return RVVRepackTilingVariant::Plain;
  }
  return RVVRepackTilingVariant::S6Tiled;
}
```
| g 字段 | c 字段 |
|---|---|
| `shape`(由 foldModel 派生·g) | — |

- **θ 输出**:`RVVRepackTilingVariant{S6Tiled,Plain}`(即 XFER-1 prior)。消费点:F8 Stage-2b + Stage-1 fail-safe。**仅读 g派生**。

### F7 · tilingVariantFeasibleSet — c → 可行集(Stage-1 legality)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:149-159`

```cpp
inline llvm::SmallVector<RVVRepackTilingVariant, 2>
tilingVariantFeasibleSet(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                         std::int64_t vregCount) {
  (void)shape;
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible;
  if (vlenBits < 128 || vregCount <= 0)
    return feasible; // no capability fact to select on.
  feasible.push_back(RVVRepackTilingVariant::Plain);
  feasible.push_back(RVVRepackTilingVariant::S6Tiled);
  return feasible;
}
```
| g 字段 | c 字段 |
|---|---|
| `shape`(**`(void)shape;`·显式丢弃·不读**) | `vlenBits`、`vregCount` |

- **θ 输出**:可行 variant 集(空=无能力fact)。**shape 被 `(void)` 丢弃 ⟹ 实际仅读 c**。消费点:F8。

### F8 · selectRepackTilingVariant — θ = plain vs s6_tiled(两段式)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:309-349`

```cpp
inline RVVRepackTilingChoice
selectRepackTilingVariant(RVVTilingBottleneckShape shape, std::int64_t vlenBits,
                          std::int64_t vregCount,
                          std::optional<RVVTilingMeasurementHit> measurement) {
  llvm::SmallVector<RVVRepackTilingVariant, 2> feasible =
      tilingVariantFeasibleSet(shape, vlenBits, vregCount);
  if (feasible.empty())
    return {priorTilingVariantForShape(shape),
            RVVTilingSelectionReason::StaticOrder};
  if (feasible.size() == 1)
    return {feasible.front(), RVVTilingSelectionReason::OnlyFeasible};
  if (measurement) {
    for (RVVRepackTilingVariant v : feasible)
      if (v == measurement->winner)
        return {measurement->winner, RVVTilingSelectionReason::Measured};
  }
  return {priorTilingVariantForShape(shape), RVVTilingSelectionReason::Prior};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `shape`(→prior via F6) | `vlenBits`、`vregCount`(via F7) | `measurement`(offline·via F10) |

- **θ 输出**:`{variant, reason∈{StaticOrder,OnlyFeasible,Measured,Prior}}`。
- **消费点**:`RVVLowerQuantContraction.cpp:1431` stamp `kTilingVariantAttr`/`kTilingReasonAttr`/`kTilingRecordAttr` on `TypedRepackGemmLoopBodyOp`。
- **[SEL-1-T5] 生产路径断言**(RVVLowerQuantContraction.cpp:1448-1451):`assert(choice.reason != StaticOrder ...)` — 生产 dispatch 路径零 static_order(gate `isRepack && halfLanes != 0`)。

### F9 · lookupMeasurement — offline profile 表(SP4/LoopOrder 统一 seed)
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:245-279`

```cpp
inline std::optional<RVVMeasurementHit>
lookupMeasurement(llvm::StringRef declaredInstanceHash, llvm::StringRef kernel,
                  RVVMeasurementAxis axis) {
  ...
  static constexpr llvm::StringLiteral kBoardInstanceHash =
      "3cd23a4ec9796a3ce1f863cd80c96b894267ab95b45cb0ecfeb856cc643b58c7";
  const SeededMeasurement kSeeded[] = {
      {kBoardInstanceHash, "q4_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q2_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q5_K", RVVMeasurementAxis::SP4Tiling, "s6_tiled"},
      {kBoardInstanceHash, "q6_K", RVVMeasurementAxis::SP4Tiling, "plain"},
      {kBoardInstanceHash, "q3_K", RVVMeasurementAxis::SP4Tiling, "plain"},
      {kBoardInstanceHash, "q4_K", RVVMeasurementAxis::LoopOrder, "col_outer"},
  };
  if (declaredInstanceHash.empty())
    return std::nullopt;
  for (const SeededMeasurement &row : kSeeded)
    if (row.axis == axis && row.declaredInstanceHash == declaredInstanceHash &&
        row.kernel == kernel)
      return RVVMeasurementHit{row.axis, row.winner};
  return std::nullopt;
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `kernel`(kernel 名·q4_K…) | `declaredInstanceHash`(board 展开 instance hash·c派生) | `axis` |

- **θ 输出**:offline argmin winner token(seed 6 行·SP4 5 + LoopOrder 1)。key=(hash,kernel,axis);hash 由 `computeDeclaredInstanceHash(capabilities)` 得(c派生)。
- **★byte-exact caveat 原文**(:217-228):winner 是"PRE-COMPUTED argmin ... MIRRORED here as the hardcoded seed token ... NEVER re-derives the argmin from cold_median at compile time"。
- **消费点**:经 F10/F11 wrapper 喂 F8/F13。

### F10 · lookupTilingMeasurement / F11 · lookupLoopOrderMeasurement(thin wrapper)
`RVVRepackTilingSelection.h:289-300`(SP4)/`:469-486`(LoopOrder)

```cpp
inline std::optional<RVVTilingMeasurementHit>
lookupTilingMeasurement(llvm::StringRef declaredInstanceHash,
                        llvm::StringRef kernel) {
  std::optional<RVVMeasurementHit> hit = lookupMeasurement(
      declaredInstanceHash, kernel, RVVMeasurementAxis::SP4Tiling);
  if (!hit) return std::nullopt;
  RVVRepackTilingVariant winner = hit->winner == "s6_tiled"
      ? RVVRepackTilingVariant::S6Tiled : RVVRepackTilingVariant::Plain;
  return RVVTilingMeasurementHit{winner};
}
```
- 输入同 F9(kernel[g] + hash[c派生]);θ=typed variant hit。消费点:F8(:1430)、F13(:1502)。

### F12 · repackColGroupOuterForLayout — g(stride) → col/row-outer 布局 prior
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:441-444`

```cpp
inline bool repackColGroupOuterForLayout(std::int64_t weightStride,
                                         std::int64_t activationStride) {
  return weightStride >= activationStride;
}
```
| g 字段 | c 字段 |
|---|---|
| `weightStride`、`activationStride`(repack panel DRAM stride·layout fact) | — |

- **θ 输出**:bool(col-outer iff weight panel ≥ activation panel)。**仅读 g**。注释 :438-440:此为前门 selector 与 EmitC emitter 共享的"同源同事实"单一 stride 事实源。消费点:F13。

### F13 · selectRepackLoopOrder — θ = row_outer vs col_outer
`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:498-518`

```cpp
inline RVVRepackLoopOrderChoice
selectRepackLoopOrder(std::int64_t weightStride, std::int64_t activationStride,
                      bool isPrefillGemm, std::int64_t vlenBits,
                      std::int64_t vregCount,
                      std::optional<RVVLoopOrderMeasurementHit> measurement) {
  RVVRepackLoopOrder layoutPrior =
      repackColGroupOuterForLayout(weightStride, activationStride)
          ? RVVRepackLoopOrder::ColOuter : RVVRepackLoopOrder::RowOuter;
  if (!isPrefillGemm || vlenBits < 128 || vregCount <= 0)
    return {layoutPrior, RVVTilingSelectionReason::OnlyFeasible};
  if (measurement)
    return {measurement->winner, RVVTilingSelectionReason::Measured};
  return {layoutPrior, RVVTilingSelectionReason::Prior};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `weightStride`、`activationStride`(via F12) | `vlenBits`、`vregCount` | `isPrefillGemm`(regime)、`measurement` |

- **θ 输出**:`{order∈{RowOuter,ColOuter}, reason∈{OnlyFeasible,Measured,Prior}}`。
- **消费点**:`RVVLowerQuantContraction.cpp:1503` stamp `kLoopOrderAttr`/`kLoopOrderReasonAttr`/`kLoopOrderRecordAttr`。

### F14 · chooseFillOptimalLMUL — θ = fill-optimal LMUL
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2689-2731`

```cpp
inline RVVFillLMULChoice
chooseFillOptimalLMUL(unsigned vlenBits, unsigned sew, unsigned blockLen,
                      llvm::ArrayRef<llvm::StringRef> candidates) {
  llvm::StringRef widest =
      candidates.empty() ? llvm::StringRef() : candidates.front();
  for (llvm::StringRef candidate : candidates.drop_front())
    if (isRVVLMULWider(candidate, widest))
      widest = candidate;
  if (vlenBits < 128 || sew == 0 || blockLen == 0)
    return {widest, RVVFillLMULReason::FallbackWidest};
  if (candidates.size() == 1)
    return {candidates.front(), RVVFillLMULReason::OnlyFeasible};
  llvm::StringRef best;
  std::int64_t bestNum = -1, bestDen = 1;
  for (llvm::StringRef candidate : candidates) {
    std::int64_t vlmax = getRVVStripVLMAXElements(candidate, sew, vlenBits);
    if (vlmax <= 0) continue;
    std::int64_t num = std::min<std::int64_t>(blockLen, vlmax);
    std::int64_t den = vlmax;
    bool better = best.empty() || (num * bestDen > bestNum * den);
    bool tie = !best.empty() && (num * bestDen == bestNum * den);
    if (better || (tie && isRVVLMULWider(candidate, best))) {
      best = candidate; bestNum = num; bestDen = den;
    }
  }
  if (best.empty())
    return {widest, RVVFillLMULReason::FallbackWidest};
  return {best, RVVFillLMULReason::Prior};
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `sew`(元素宽)、`blockLen`(块元素跨度) | `vlenBits` | `candidates`(constructible LMUL 集) |

- **θ 输出**:`{lmul, reason∈{Prior,OnlyFeasible,FallbackWidest}}`。
- **HONESTY CRUX 原文**(:2629-2637):"this function accesses ZERO cost model ... pure f(vlenBits, sew, blockLen, candidates)"。
- 消费点:见头文件注释,fill-optimal LMUL prior(构造期 [SEL-1] 首个能力派生 prior);此为 pure 头函数(消费点在各 fill/gearbox 落点)。

### F15 · chooseNumericsTier — θ = strict vs relaxed(数值折叠 policy tier)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2814-2821`

```cpp
inline RVVNumericsTierChoice
chooseNumericsTier(bool reassocOkPresent, bool kernelIsFpOrderSensitive) {
  if (!kernelIsFpOrderSensitive)
    return {RVVNumericsTier::Strict, RVVNumericsTierReason::StrictOnlyExact};
  if (reassocOkPresent)
    return {RVVNumericsTier::Relaxed, RVVNumericsTierReason::RelaxedByPolicy};
  return {RVVNumericsTier::Strict, RVVNumericsTierReason::StrictDefault};
}
```
| g 字段 | c 字段 |
|---|---|
| `kernelIsFpOrderSensitive`(kernel 有无可重结合 fp fold·结构 fact) | `reassocOkPresent`(`numerics.reassoc_ok` kind=policy 能力fact) |

- **θ 输出**:`{tier∈{Strict,Relaxed}, reason∈{StrictDefault,RelaxedByPolicy,StrictOnlyExact}}`。fail-closed 默认 Strict。
- 注释 :2750-2756:policy fact "today ... through the `--numerics-reassoc-ok` pass option (a policy gate, fail-closed OFF)"。

### F16 · enumerateBlockDotShapeCandidates — g(descriptor)×c(VLEN/budget) → 候选集
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2903-2987`(见附录·此处摘 legality 核)

```cpp
inline llvm::SmallVector<RVVBlockDotShapeCandidate, 18>
enumerateBlockDotShapeCandidates(const RVVBlockDotKernelDescriptor &descriptor,
                                 std::int64_t minimumVLEN,
                                 std::int64_t vectorRegisterBudget) {
  ...
  const std::int64_t rankVLEN = minimumVLEN >= 128 ? minimumVLEN : 128;
  for (llvm::StringRef coreLMUL : descriptor.coreLMULs) {
    const std::int64_t stripSEW = descriptor.stripSEW(coreLMUL);
    const llvm::StringRef stripLMUL = getRVVBlockDotStripLMUL(coreLMUL);
    const std::int64_t elisionVLMAX =
        getRVVStripVLMAXElements(stripLMUL, stripSEW, minimumVLEN);
    const std::int64_t rankVLMAX =
        getRVVStripVLMAXElements(stripLMUL, stripSEW, rankVLEN);
    const std::int64_t reductions =
        (descriptor.blockLen + rankVLMAX - 1) / rankVLMAX;
    for (std::int64_t factor : kFactors) {
      if (factor > descriptor.factorCap) continue;
      for (llvm::StringRef elision : kElisions) {
        ... candidate.cost = computeBlockDotShapeCostCore(...);
        bool elisionLegal = true;
        if (elision == "elided") elisionLegal = elisionVLMAX >= descriptor.blockLen;
        bool gatherLegal = descriptor.gatherTableEntries == 0 ||
                           elisionVLMAX >= descriptor.gatherTableEntries;
        bool budgetLegal = candidate.vectorRegisterCost <= vectorRegisterBudget;
        candidate.isLegal = elisionLegal && gatherLegal && budgetLegal;
      }
    }
  }
  return candidates;
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `descriptor.coreLMULs`(anchor 集)、`.quantFormat`、`.blockLen`、`.gatherTableEntries`、`.factorCap`、`.stripSEW`(fn) | `minimumVLEN`、`vectorRegisterBudget` | |

- **θ 输出**:`{integer_core_lmul, multi_block_factor, strip_elision}` 候选 + cost + isLegal。注释明言 capability 只从 legality prune 入口(:2886-2896)、cost formula capability-blind。
- 消费点:各 front-door `selectIntegerCoreLMUL`(F25)、`enumerateRVVQ40Q80ShapeCandidates`(F22)→ `selectGenericSchedule`(F21)。

### F17 · selectGenericMinCostCandidate — 候选集 → min-cost 合法(静态 argmin)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2319-2338`

```cpp
inline std::optional<GenericScheduleCandidate>
selectGenericMinCostCandidate(
    llvm::ArrayRef<GenericScheduleCandidate> candidates) {
  std::optional<GenericScheduleCandidate> best;
  for (const GenericScheduleCandidate &candidate : candidates) {
    if (!candidate.isLegal) continue;
    if (!best || candidate.cost < best->cost) { best = candidate; continue; }
    if (candidate.cost == best->cost &&
        candidate.tieBreakVregCost < best->tieBreakVregCost)
      best = candidate;
  }
  return best;
}
```
| 读取 | 说明 |
|---|---|
| `candidate.cost`(g/θ 派生的结构 cost)、`.isLegal`(g×c 派生)、`.tieBreakVregCost`(vreg footprint·c/θ) | 只读候选上已派生字段 |

- **θ 输出**:min-cost 合法候选(tie → 更轻 vreg footprint 胜)。**不直接读 g/c 原始字段**(读 candidate 派生量)。消费点:F21 Stage-2。

### F18 · selectGenericSchedule — 候选集(+ 记录) → 选择(measured-best/static)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:3833-3873`

```cpp
inline std::optional<GenericScheduleSelection> selectGenericSchedule(
    llvm::ArrayRef<GenericScheduleCandidate> candidates,
    const std::optional<std::string> &recordText, llvm::StringRef kernelKey,
    llvm::StringRef march, llvm::ArrayRef<llvm::StringRef> requiredKnobKeys) {
  std::int64_t legalCount = 0;
  for (const GenericScheduleCandidate &candidate : candidates)
    if (candidate.isLegal) ++legalCount;
  if (recordText) {
    std::optional<GenericTuningRecordEntry> entry = lookupGenericTuningRecord(
        *recordText, kernelKey, march, requiredKnobKeys);
    if (entry) {
      std::optional<GenericScheduleCandidate> revalidated =
          revalidateGenericTuningRecord(candidates, *entry);
      if (revalidated) { ... return selection; } // fromMeasurement=true
    }
  }
  std::optional<GenericScheduleCandidate> staticBest =
      selectGenericMinCostCandidate(candidates);
  if (!staticBest) return std::nullopt;
  ... return selection; // fromMeasurement=false
}
```
| g 字段 | c 字段 | 其它 |
|---|---|---|
| `kernelKey`(kernel 名·record 查询 key) | `march`(record 查询 key) | `candidates`、`recordText`(offline)、`requiredKnobKeys` |

- **θ 输出**:`GenericScheduleSelection{candidate, fromMeasurement, measuredNs, legalCandidateCount}`。measured-best 优先,否则 static argmin(F17),全 pruned → nullopt(fail-closed I7)。
- 消费点:`RVVScheduleDescriptorRegistry.cpp:521`(schedule 物化 runner)、各 front-door selectIntegerCoreLMUL(F25)。

### F19 · lookupRepackVlen256Decode — g(scaleModel) → board disposition(4 行数据表)
`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:659-687`

```cpp
constexpr RepackVlen256DecodeMeasurement kRepackVlen256DecodeMeasurements[] = {
    {kNibbleQ50ScaleModel, Vlen256DecodeRepackDisposition::Beneficial, "1.190x", ...},
    {kNibbleQ51ScaleModel, Vlen256DecodeRepackDisposition::Beneficial, "1.306x", ...},
    {kNibbleQ40ScaleModel, Vlen256DecodeRepackDisposition::Negative,   "0.74x",  ...},
    {kCodebookIq4NlScaleModel, Vlen256DecodeRepackDisposition::Negative, "0.248x", ...},
};
std::optional<Vlen256DecodeRepackDisposition>
lookupRepackVlen256Decode(llvm::StringRef scaleModel) {
  for (const RepackVlen256DecodeMeasurement &m : kRepackVlen256DecodeMeasurements)
    if (m.scaleModel == scaleModel) return m.disposition;
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| `scaleModel`(格式 WHAT·registry key) | (行内容=板测 board fact) |

- **θ 输出**:`optional<Disposition{Beneficial,Negative}>`;无行=nullopt(保守 decline)。**仅按 g-key 查表**。消费点:`readOpponentFacts`(:1346)→ facts.vlen256DecodeRepackBeneficial → F1。

### F20 · resolveRepackMainTermRolled — θ = rolled vs unrolled(super-block 主项)
`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:3216-3269`

```cpp
static bool resolveRepackMainTermRolled(std::optional<llvm::StringRef> stamp,
                                        llvm::StringRef coreLmul, int64_t qk,
                                        int64_t weightInterleave,
                                        int64_t activationInterleave,
                                        int64_t half) {
  if (stamp.has_value()) {
    if (*stamp == "rolled") return true;
    if (*stamp == "unrolled") return false;
  }
  int64_t numHalves = (half > 0) ? (weightInterleave / half) : 1;
  int64_t nSuperHalves = (qk > 0) ? (qk / 128) : 1;
  int64_t columnsPerPass = (coreLmul == "m1") ? 1 : activationInterleave;
  int64_t unrolledMainTermVwmacc = numHalves * nSuperHalves * 2 * 16 * columnsPerPass * 4;
  constexpr int64_t kMainTermUnrollCodeVolumeICacheBudget = 192;
  bool codeVolumeExceedsBudget =
      unrolledMainTermVwmacc > kMainTermUnrollCodeVolumeICacheBudget;
  if (codeVolumeExceedsBudget)
    if (std::optional<bool> measuredRollBeneficial =
            lookupRollMeasuredBeneficial(unrolledMainTermVwmacc, coreLmul))
      return *measuredRollBeneficial;
  return false;
}
```
| g 字段 | c 字段 | θ 输入 |
|---|---|---|
| `qk`、`weightInterleave`、`activationInterleave`(块/交织几何) | —(budget=192 常量) | `stamp`、`coreLmul`、`half`(θ) |

- **θ 输出**:bool rolled。measured 表(F21b)空 → 恒 unrolled(default,byte-exact)。消费点:super-block K-quant GEMM/GEVM dispatch 臂。

### F21b · lookupRollMeasuredBeneficial(F20 measured 表·EMPTY)
`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:3211-3214`

```cpp
static std::optional<bool>
lookupRollMeasuredBeneficial(int64_t /*unrolledMainTermVwmacc*/,
                             llvm::StringRef /*coreLmul*/) {
  return std::nullopt;
}
```
- 两参数全注释掉·**未读**;恒 nullopt。注释 :3203-3209:"EMPTY today: nullopt for every shape => the UNROLLED default holds"。

### F22 · 单旋钮候选枚举器(q1_0/tq2_0/tq1_0/iq2_xxs)— c(VLEN)+ g(块常量) → 候选
`lib/Plugin/RVV/Schedule/RVVScheduleDescriptorRegistry.cpp:48-148`(四函数同构·摘 q1_0)

```cpp
static llvm::SmallVector<GenericScheduleCandidate>
enumerateRVVQ10ShapeCandidates(std::int64_t minimumVLEN) {
  constexpr std::int64_t kQ10SubBlockLen = 32;
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (auto anchor : {llvm::StringRef("m2"), llvm::StringRef("m1")}) {
    std::int64_t stripVLMAX = getRVVStripVLMAXElements(
        getRVVBlockDotStripLMUL(anchor), getRVVBlockDotStripSEW(anchor), minimumVLEN);
    GenericScheduleCandidate candidate;
    candidate.cost = 0;
    candidate.isLegal = stripVLMAX >= kQ10SubBlockLen;
    candidate.tieBreakVregCost = (anchor == "m1") ? 1 : 2;
    candidate.knobs.push_back({"lmul", "integer_core_lmul", anchor.str(), false});
    candidates.push_back(candidate);
  }
  return candidates;
}
```
| g 字段 | c 字段 |
|---|---|
| `kQ10SubBlockLen=32`(子块几何·常量) + anchor 集 {m2,m1} | `minimumVLEN` |

- **θ 输出**:`integer_core_lmul` 单旋钮候选(m2@VLEN128 / 更轻 m1@VLEN256);cost=0,legality+tieBreak 定夺。tq2_0/tq1_0/iq2_xxs 同构(:79-148)。

### F23 · enumerateRVVQ40Q80ShapeCandidates 等(per-kernel 薄壳)
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:3006-3013`(以及 q41/q50/q51/q80/codebook 同构)

- 均调 `enumerateBlockDotShapeCandidates(getRVVQ40Q80KernelDescriptor(), minimumVLEN, budget)`;descriptor(g)在 `getRVVQ40Q80KernelDescriptor`(:2994-2998)内 baked({mf4,m1}, "nibble-offset-binary", blockLen=16)。输入 = minimumVLEN(c) + budget(c);g 全 baked 在 descriptor。

### F24 · lookupRVVScheduleDescriptor — g(kernelKey) → 调度 descriptor
`lib/Plugin/RVV/Schedule/RVVScheduleDescriptorRegistry.cpp:150-442`

- **仅读 g(`kernelKey` 字符串)** → 返回 per-kernel descriptor(含 enumerate fn、attrPrefix、budget、measured/staticReason 文案)。11 分支:q4_0/q8_0/q4_1/q5_0/q5_1/q1_0/tq2_0/tq1_0/iq2_xxs/iq4_nl/mxfp4/q4_0_q8_0_gemm;未注册 key → nullopt(fail-closed skip)。消费点:`runRVVScheduleMaterializationViaInterface`(:509)。

### F25 · selectIntegerCoreLMUL(×3:Reduction / Dequant / PackedI4 前门)— c → coreLMUL
`lib/Plugin/RVV/FrontDoor/RVVReductionSourceFrontDoor.cpp:343-379`(另二同构:RVVDequantDotSourceFrontDoor.cpp:374、RVVPackedI4DotSourceFrontDoor.cpp:207)

```cpp
std::optional<std::string>
selectIntegerCoreLMUL(llvm::StringRef march, llvm::StringRef isaVectorHints) {
  std::int64_t minimumVLEN = deriveMinimumVLEN(march, isaVectorHints);
  static constexpr llvm::StringLiteral kCoreLMULs[] = {"m1", "m2"};
  RVVBlockDotKernelDescriptor descriptor{
      /*coreLMULs=*/kCoreLMULs, /*quantFormat=*/"plain-int8",
      /*blockLen=*/kContractionBlockLen,
      /*stripSEW=*/getRVVBlockDotStripSEW,
      /*vectorRegisterCost=*/getRVVQ80ShapeVectorRegisterCost};
  descriptor.factorCap = 1;
  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> typed =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       kRVVQ80ShapeVectorRegisterBudget);
  ... toGenericBlockDotCandidate ...
  static constexpr llvm::StringRef kRequiredKnobKeys[] = {"lmul"};
  std::optional<GenericScheduleSelection> selected = selectGenericSchedule(
      candidates, /*recordText=*/std::nullopt,
      /*kernelKey=*/"widening_dot_reduce_i8", march, kRequiredKnobKeys);
  if (!selected) return std::nullopt;
  for (const NamedKnob &knob : selected->candidate.knobs)
    if (knob.recordKey == "lmul") return knob.value;
  return std::nullopt;
}
```
| g 字段 | c 字段 |
|---|---|
| descriptor 内 g 常量(anchor {m1,m2}、"plain-int8"、blockLen=kContractionBlockLen)全 **baked**,非从格式输入读 | `march`、`isaVectorHints`(签名唯一输入) |

- **θ 输出**:`optional<string>` = integer_core_lmul(m2@VLEN128 → m1@VLEN256 flip)。**函数签名仅吃 c**(g 几何 baked 在 inline descriptor)。消费点:RVVReductionSourceFrontDoor.cpp:882 / Dequant:1104 / PackedI4:758,stamp `weft_rvv.gearbox_selected_integer_core_lmul`。

---

## Tier-2 能力/几何派生(读 c 或 g → 中间量喂 Tier-1)

### F26 · deriveMinimumVLEN(march, hints) → VLEN bits ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:289-332`(全文见附录A)。扫 `zvl{N}b` token 取最大 floor;full-V/xtheadvector floor 128;embedded zve*→0。**仅读 c**。是全栈 VLEN 事实的唯一权威(F1/F3/F7/F8/F13/F14/F16/F22/F25/strip-width 全依赖)。

### F27 · deriveHasZvl128b(march, hints) → bool ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:334-339` `return deriveMinimumVLEN(...) >= 128;`。仅读 c。

### F28 · deriveRVVVersion(march, hints) → {RVV0p7,RVV1p0,Unknown} ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:353-375`。`xtheadvector`/`0p7`→0.7;`gcv`/`zve`/`_v`→1.0;否则 Unknown。仅读 c。供 F3(isRVV0p7)、strip-width、supported-LMUL allow-list。

### F29 · deriveSupportedLMULAllowList(march, hints) → LMUL 支持 allow-list θ ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:260-277`

```cpp
std::string deriveSupportedLMULAllowList(llvm::StringRef selectedMarch,
                                         llvm::StringRef isaVectorHints) {
  switch (deriveRVVVersion(selectedMarch, isaVectorHints)) {
  case RVVVersion::RVV0p7: return "m1,m2,m4,m8";            // 无 fractional LMUL
  case RVVVersion::RVV1p0: return "mf8,mf4,mf2,m1,m2,m4,m8";
  case RVVVersion::Unknown: return "";
  }
  return "";
}
```
仅读 c(经 deriveRVVVersion)。**θ 输出**:LMUL 支持集(legality gate 查询用·非 body 单选 LMUL)。消费点:`MaterializeRVVProbedCapabilityAxes`(:101)stamp `supported_lmul`。

### F30 · deriveSupportedSEWAllowList(march, hints) → SEW 支持 allow-list ★[c only]
`lib/Plugin/RVV/RVVCapabilityProfile.cpp`(签名 include/…/RVVCapabilityProfile.h:124)。仅读 c。消费点:同 F29(:99 stamp `supported_sew`)。

### F31 · deriveRepackHalfLanes(vlenBits) → half_lanes θ ★[c only](两份)
① `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1222-1226`
```cpp
std::int64_t deriveRepackHalfLanes(std::int64_t vlenBits) {
  if (vlenBits < 128) return 0;
  return std::min<std::int64_t>(vlenBits / 16, kWeightInterleave);
}
```
② `lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp:78-84`(带 weightInterleave 参数;128→8,256→16,512+→16;<128→0)。
| g 字段 | c 字段 |
|---|---|
| `kWeightInterleave`/`weightInterleave`(交织宽·此处为 clamp 上界常量/op attr) | `vlenBits` |

- **θ 输出**:half_lanes(strip 宽)∈{0,8,16}。消费点:①`RVVLowerQuantContraction.cpp:1562` gate `isRepack && halfLanes != 0` + 喂 F3 capabilityHalfLanes;②MaterializeRVVRepackStripWidth pass(见 Tier-4 strip-width 段)。

### F32 · getRVVStripVLMAXElements(coreLMUL, stripSEW, minimumVLEN) → VLMAX
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2606-2613`

```cpp
inline std::int64_t getRVVStripVLMAXElements(llvm::StringRef coreLMUL,
                                             std::int64_t stripSEW,
                                             std::int64_t minimumVLEN) {
  if (minimumVLEN <= 0 || stripSEW <= 0) return 0;
  std::pair<std::int64_t, std::int64_t> frac = getRVVLMULFraction(coreLMUL);
  return (minimumVLEN * frac.first) / (frac.second * stripSEW);
}
```
| g 字段 | c 字段 | θ 输入 |
|---|---|---|
| `stripSEW`(strip 元素宽·结构) | `minimumVLEN` | `coreLMUL`(θ) |

- **θ 输出**:strip VLMAX 元素数(legality/reduction 派生的单一真相源;verifier 亦从此重算)。消费点:F14/F16/F22。

### F33 · enumerateRVVLowPrecisionAccumulatorLMULRungs(budget, reserve) → i8链 rung 集 ★[c only]
`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2043-2074`。source rungs {mf4,mf2,m1,m2} 常量;每级两步 EMUL widen,`isLegal = acc+prod+reserve <= budget`。输入=budget/reserve(c/寄存器预算)。θ=各 rung(source/product/accumulator LMUL + legal)。

### F34 · selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs) → 最宽合法 rung
`RVVGearboxSchedule.h:2083-2095`。读 rung.isLegal/accumulatorRegisterCost(派生)。θ=最宽合法 accumulator-LMUL(A=1)。

### F35 · enumerateRVVDotReduceDeferredWideLMULRungs(budget, reserve) → i16链 rung 集 ★[c only]
`RVVGearboxSchedule.h:2131-2154`。source {mf2,m1,m2,m4};`accumulatorLMUL==productLMUL`(单 widen);legal iff acc+reserve<=budget。

### F36 · selectRVVDotReduceDeferredWideMaxLegalLMULRung(rungs) → 最宽合法 rung
`RVVGearboxSchedule.h:2160-2172`。同 F34 结构。

### F37 · makeRVVDotReduceMinimalDeferredM1Rung() → 常量 m1 rung(**无输入**)
`RVVGearboxSchedule.h:2181-2190`。硬返回 source mf2 / accumulator m1(N3 Win-C 最小合法)。**读 0 输入**(既非 g 也非 c)。

---

## Tier-3 θ→θ 实现派生(只读 θ·不读 g/c·不计入 f 总数)

- **getRVVLMULRegisterFootprint(lmul)** `RVVGearboxSchedule.h:2007-2017` — lmul(θ)→vreg 数({m1,mf2,mf4,mf8}=1,m2=2,m4=4,m8=8)。
- **getRVVNextWiderLMUL(lmul)** `:2022-2034` — lmul(θ)→宽一级 LMUL。
- **isRVVLMULWider(a,b)** `:2677-2681` — 两 LMUL(θ)有理比较。
- **getRVVBlockDotStripSEW(coreLMUL)** `:2870-2872` — coreLMUL(θ)→strip SEW(mf4→32,否则 8)。
- **getRVVBlockDotStripLMUL(coreLMUL)** `:2880-2882` — coreLMUL(θ)→strip LMUL(mf4→m1)。
- **getRVVQ40ReductionsPerHalfBlock(coreLMUL)** `:2343-2345` — coreLMUL(θ)→reduction 数。
- **deriveWideningChain(base)** `lib/Conversion/RVV/RVVToEmitCSupport.cpp:1064-1073` — base=coreLmul(θ)→{l8,l16,l32,stripWidth,foldGroups}(emitter 纯 realize 用)。

---

## Tier-4 焊死点(假旋钮 θ)+ fail-closed 哨兵

### (a) coreLmul 硬编码常量(θ 焊死·"假旋钮形")
| 落点 | 原文 | 备注 |
|---|---|---|
| `RVVToEmitCBlockQuantLinear.cpp:111-112` | `coreLmul="m1"; wideLmul="m2";` | flat GEMM-tile;注释 :107-110「[A-line stage-3: structural constant, NOT a knob] ... [K-10]: do not fake a knob -- the debake is a no-op here」 |
| `RVVToEmitCBlockQuantLinear.cpp:404` | `coreLmul="m1";` | flat carrier 第二处 |
| `RVVToEmitCGridCodebook.cpp:497-498` | `coreLmul="m1"; wideLmul="m2";` | iq2_xxs grid;硬编码 vint8m1_t |
| `RVVToEmitCCodebookFp4.cpp:109` | `coreLmul="m1";` | 注释 :107-108「codebook gather REQUIRES the m1 anchor ... the verifier pins it」 |
| `RVVToEmitCTernaryBinary.cpp:862-863` | `coreLmul="m1"; wideLmul="m2";` | 硬编码 |

### (b) coreLmul default + 读 attr(θ realize·非全焊死)
| 落点 | 原文 |
|---|---|
| `RVVToEmitCTernaryBinary.cpp:1607-1610` | `coreLmul="m2"; if(attrLmul=coreOp.getIntegerCoreLmul()) coreLmul=*attrLmul; wideLmul=(coreLmul=="m2")?"m4":"m2";` — 默认 m2·gearbox 可 refine m2→m1 |
| `RVVToEmitCGridCodebook.cpp:59` | `coreLmul = cx.coreLmul;`(从 context 读 θ) |

### (c) fail-closed 哨兵(BQL 拒无 integer_core_lmul 的 repack leaf)
机算枚举同一 error 串 `"repack integer core requires an explicit integer_core_lmul"`(`grep -c` = **18** 处·非 8):
`RVVToEmitCBlockQuantLinear.cpp` 行 2209, 2314, 2431, 2621, 2759, 2988, 3371, 3479, 3603, 3777, 3975, 4058, 4516, 5035, 5573, 5994, 7743, 7799。
- 语义:emitter 是 pure realize,读 stamped `integer_core_lmul`(θ);缺失即 fail-closed 报错(注释「front door ALWAYS stamps integer_core_lmul on every wired repack leaf」)。
- **口径注**:采集令关键文件行写「8 fail-closed 哨兵」;机算该 error 串出现 = 18。二者差异如实记(是否 8 为某子集/口径由上游判)。

### (d) l8/l16/l32 链(BQL 内联·读 θ)
`RVVToEmitCBlockQuantLinear.cpp:2991-2993`:
```cpp
llvm::StringRef l8 = coreLmul;                         // mf2->mf2; m1->m1
llvm::StringRef l16 = coreLmul == "m1" ? "m2" : "m1";  // mf2->m1;  m1->m2
llvm::StringRef l32 = coreLmul == "m1" ? "m4" : "m2";  // mf2->m2;  m1->m4
```
读 coreLmul(θ)→ widening 链(θ realize)。

---

## 选择器静态优先序段(原文照抄)

### (1) [SEL-1] SP4 两段式优先序(F8)+ static_order fail-safe
`RVVRepackTilingSelection.h:302-348`(见 F8)。序:Stage-1 legality(F7:vlenBits<128||vregCount<=0 → 空集)→ 空集则 `{prior-variant, StaticOrder}`(:329-331)→ 单元素 `OnlyFeasible`→ measured hit(feasible 内)`Measured`→ 否则 `{prior, Prior}`。
- reason 枚举 `RVVTilingSelectionReason{OnlyFeasible,Measured,Prior,StaticOrder}`(:54);头注释 :51-53「`StaticOrder`: a capability-BLIND fallback -- it must NOT appear on a capability-afforded board; its appearance signals a prior-coverage gap (the burn-down signal)」。
- **读了什么**:vlenBits(c)、vregCount(c)、shape(g→prior)、measurement(offline)。**没读**:format label(shape 由 foldModel 结构 KEY,非格式名)。

### (2) selectGenericSchedule 静态优先序(F18)
序:(1) measured-best(record 命中 + revalidate 仍合法)→ (2) static argmin `selectGenericMinCostCandidate`(cost 主键·tieBreakVregCost 破平)→ 全 pruned=nullopt(fail-closed I7)。**读**:candidates(g×c 派生)、recordText、kernelKey(g)、march(c)。

### (3) RVVScheduleDescriptorRegistry 物化 runner 序
`RVVScheduleDescriptorRegistry.cpp:444-533`:①`deriveMinimumVLEN`/`deriveHasZvl128b`(c)一次(:457-458)→ 无 clobber guard `isSchedulePinned()`(:503)→ kernelKey→`lookupRVVScheduleDescriptor`(F24·g)→ enumerate(minimumVLEN,budget)→ `selectGenericSchedule`(F18)→ stamp。`staticReason`/`measuredReason` 文案 per-kernel(:161-438)。

### (4) [SEL-1] fill-optimal LMUL 优先序(F14)
`RVVGearboxSchedule.h:2683-2731`:widest 兜底 → vlenBits<128||sew==0||blockLen==0 → `FallbackWidest`(:2702-2703)→ 单候选 `OnlyFeasible`→ util(min(blockLen,vlmax)/vlmax)最大·tie 破 widest → `Prior`。reason `RVVFillLMULReason{Prior,OnlyFeasible,FallbackWidest}`(:2655)。

---

## [GAP-P1] / measured table 相关段(原文照抄)

### selectRepackAccumulatorLMUL 三层(F3)— 已在 F3 段照抄
三层序:①`isRVV0p7` → `{useM1=true,"correctness-rvv0p7"}`(RVV0.7 无 fractional LMUL·correctness 非 perf)②`m1Constructible`(capabilityHalfLanes!=0 且 m2+m4 footprint<=32)且 `lookupRepackMeasuredM1Faster` 命中 → `{*measuredM1Faster,"measured"}`③默认 `{useM1=false,"capability-default-mf2"}`。measured 表(F4)恒 nullopt ⟹ 每 RVV1.0 格 → mf2 → byte-exact。

### [GAP-P1] IRON RULE 原文(`RVVLowerQuantContraction.cpp:1254-1262`)
```
// [GAP-P1] IRON RULE -- widen-to-m1 was FALSIFIED TWICE (micro win washes at e2e
// / regfile spill). We do NOT blind-select the wider m1 (that is exactly what the
// gate4 widest-legal selector does for a DIFFERENT kernel path; reusing it here
// verbatim would re-commit [GAP-P1]). The DEPLOYED default stays mf2; ONLY a
// per-format BOARD MEASUREMENT recording m1-faster for this (format, board) flips
// it (reason "measured"). The measured table is EMPTY today (STAGE THREE
// populates the per-format x board crossover, board-MEASURED and NEVER projected
// -- [GAP-P1]), so every RVV1.0 format resolves to mf2 => BYTE-EXACT with the
// pre-selector emit; every existing fixture stays green unchanged.
```

### STAGE THREE 表 nullopt 原文(`RVVLowerQuantContraction.cpp:1271-1279`)
```
// STAGE THREE populates this per-format (x board) board-measured m1-vs-mf2
// crossover. EMPTY today: nullopt for every format => the mf2 default holds
// (byte-exact). q4_0 is the documented m1-faster CANDIDATE and q8_0 the
// mf2-faster candidate, but NEITHER is asserted here without a board number
// ([GAP-P1]: no projection; the board key is threaded in at STAGE THREE).
inline std::optional<bool>
lookupRepackMeasuredM1Faster(llvm::StringRef /*scaleModel*/) {
  return std::nullopt;
}
```

### 三张 measured 表现状(EMPTY / SEEDED)一览
| 表 | 位置 | 现状 | key |
|---|---|---|---|
| F4 lookupRepackMeasuredM1Faster(m1-vs-mf2 累加器) | RVVLowerQuantContraction.cpp:1276 | **EMPTY**(恒 nullopt) | scaleModel(未读) |
| F9 lookupMeasurement(SP4/LoopOrder 统一) | RVVRepackTilingSelection.h:245 | **SEEDED 6 行**(SP4 5 + LoopOrder 1) | (hash[c], kernel[g], axis) |
| F19 lookupRepackVlen256Decode | RVVLowerQuantContraction.cpp:681 | **SEEDED 4 行**(q5_0/q5_1 Beneficial·q4_0/iq4_nl Negative) | scaleModel[g] |
| F21b lookupRollMeasuredBeneficial(rolled-vs-unrolled) | RVVToEmitCBlockQuantLinear.cpp:3211 | **EMPTY**(恒 nullopt) | (proxy, coreLmul)(未读) |

---

## 附录A · deriveMinimumVLEN 全文
`lib/Plugin/RVV/RVVCapabilityProfile.cpp:289-332`(见正文 F26 引用;扫 zvl{N}b token 最大 floor + full-V/xtheadvector floor 128 + embedded zve*→0·仅读 c)。

## 附录B · lowerOne 家族选择器(scale_model → 家族 facts·g-key dispatch)
`RVVLowerQuantContraction.cpp:1581-1620`:按 `op.getScaleModel()`(g)选 ternary(TQ20/TQ10)、kquant(Q4K/Q6K/Q2K/Q3K/Q5K)、codebook(Iq4Nl/Iq4Xs/Mxfp4)、grid(Iq2Xxs/Iq2Xs/Iq2S) 家族的 DecodeFacts 指针。**仅按 g(scale_model WHAT)dispatch 家族**(θ=家族选择)。此为 F-家族选择器(读 g → θ 家族);列为事实。

---

## 采不到 / 未展开(如实)
- `computeBlockDotShapeCostCore` / `getRVVBlockDotCoreLatencyDepth` / `computeRVVQ40ShapeCost` 等 cost formula 全文未逐一照抄(体量·且读 coreLMUL[θ]+reductions/factor/elision[g/θ] 输出 cost·属 Tier-1 候选 cost 的内部);已在 F16/F17 记其接口与 capability-blind 性质。
- IME 侧 `kIMEWideFormatMeasurements` / IME LMUL 选择器未纳入(本包锁 RVV f;IME 为另栈·仅在 RVV 注释中被引作 registration-as-data 类比)。
- 各 front-door 的 `toGenericBlockDotCandidate` / `toGenericGemmCandidate` 适配器(纯类型转换·不产 θ 决定)未展开。


<!-- ========== 包5 节 (源: pkg5-evidence.md) ========== -->

# 包5 · 实证包(板间行为差异现成数据) — θ=f(g,c) 三角色全景盘点 v2

HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`

> 只读·零施工·零板时·禁下结论。每条原文照抄 + file:line/run-id 出处。
> 框架换算备注(仅供对齐·非判断)：`half_lanes` / `AVL` / `vwmacc 计数` / `strip 宽` = **θ 输出**；`VLEN`/`march`/`rvv_version`/`supported_lmul`/`fractional-LMUL 有无` = **c 板侧输入**；`sub_block`/`n_sb`/`码本尺寸` = **g 格式侧输入**。以下不做「焊死/公式/缺陷」判断，只列事实。

---

## 项1 · 宽化战役逐格数（BASE/WIDE·改了哪些参数·objdump 计数）

### 1a. 逐格 cold_X（VLEN128 BASE → VLEN256 WIDE·k1）
来源：`.trellis/spec/issues/门与工具.md:393`(ISSUE-105·RESOLVED)、`:406`(扇出) + `experiments/active/result-tables/K-attack-fanout-ledger.md:13`(机制①)。

原文照抄（ISSUE-105 `门与工具.md:393`）：
> **RESOLVED（部署完成·2026-07-18·★地盘+4）** … **4 格 deployed PASS**：iq3_xxs 1.38 / iq3_s 1.21 / iq1_s 1.34 / iq1_m 1.79（master k1 具名-X→PASS … byte-exact 4-arm·objdump 真宽 AVL8→16 非 re-roll…）·**counts 不变**（certified 101/108·perf-covered 9/83·hand-brick 2·k1 标量类 headline 43→47）

原文照抄（扇出 `门与工具.md:406`）：
> **扇出（★2026-07-18 已证 3 格·a6ad CONFIRMED）**：`iq3_s@k1` 0.61→**1.20 WIN** · `iq1_s@k1` 0.59→**1.34 WIN** · `iq1_m@k1` 0.59→**1.77 WIN**（2.94× compound·spill-storm 810→42 清零）——**同 VLEN256 半宽病·逐格 byte-exact + 2-seed 板测·真宽 objdump 证·非外推**。iq2_xxs/iq2_xs/iq2_s/iq4_nl objdump-真宽但 harness 未建

原文照抄（iq3_xxs 逐指令·`门与工具.md:398-401` + ledger `:13`）：
> `iq3_xxs@k1` … → **cold 0.65 LOSS → 1.38 WIN** … objdump 真宽（AVL `vsetivli zero,8,e32,m2`→`zero,16`·vset 5397→2515·gather 1024→512·非 re-roll）·2 seed（1.3838/1.3782）

BASE→WIDE 逐格（汇总·全数字原文·k1 板）：

| 格 | BASE cold(VLEN128 half_lanes=8) | WIDE cold(VLEN256 half_lanes=16) | 出处 |
|---|---|---|---|
| iq3_xxs@k1 | 0.65 (0.6474) | 1.38 (2seed 1.3838/1.3782) | 门与工具.md:399-401·ISSUE-005 性能与测量.md:45 |
| iq3_s@k1 | 0.61 (0.6136) | 1.20 (部署后 1.21) | 门与工具.md:406·:393 |
| iq1_s@k1 | 0.59 (0.5919) | 1.34 | 门与工具.md:406·:393 |
| iq1_m@k1 | 0.59 (0.5945) | 1.77 (部署后 1.79) | 门与工具.md:406·:393 |

改了哪些参数（θ 变化·原文）：`march=rv64gcv`(half_lanes=8·VLEN128) → `march=rv64gcv_zvl256b`(half_lanes=16·VLEN256)；构造/emit-time fixture 选择（门与工具.md:393「`deriveRepackHalfLanes` 已存·仅换 emit march」）。

objdump 计数（iq3_xxs·BASE→WIDE·原文）：
- AVL：`vsetivli zero,8,e32,m2` → `zero,16`（8→16）
- vset：5397 → 2515（门与工具.md:401）／另处 5397→2515（ledger）
- gather：1024 → 512
- spill-storm（iq1 系 compound）：810 → 42（门与工具.md:406）

诊断处 objdump（机制①诊断·`性能与测量.md:347-349`·ISSUE-102）原文：
> OURS iq3_xxs leaf.o **两板逐一相同** = ins=15238/15240·**vset=5397**（配置分布 `e16,m1`×2698 + `e8,mf2`×2688 交替 = e8↔e16 宽度抖动）· 标量 sp-spill=56 · 向量整寄存器 spill(vs*r/vl*re)=77
> 对手 `vec_dot_iq3_xxs_q8_K_vl256`：**vset=41**·宽 LMUL（`e8,m2`=64 lane / `e32,m4` / `e16,m2`）+ AVL 对齐 VLEN256（`zero,16,e8,mf2`=vl16 满）

### 1b. iq2/iq4 宽化战报（成色 upgrade / 具名-X 墙）
来源：`K-attack-fanout-ledger.md:5`。原文照抄：
> **★iq2/iq4 harness 建成+板测(裁决2·af60 复核)**：iq2_xxs/iq2_xs/iq2_s **成色 upgrade**(scalar-ref 便宜档→部署 OPP-X vl256 手调·真硬赢 CROSSOP·同算子 OPP-S 缺席·PASS 计数不增非新 flip)·iq4_nl **具名-X 墙**(同算子 OPP-S hand-brick beats 2.35×·widening 解 narrow-vl 墙①但 codebook-gather 墙② ISSUE-021 立·非硬赢)

### 1c. 宽化脚本 run-id 指针（现成·未新跑）
- `experiments/active/g8-stage3-attack/P2-grid4-raw/run_widen_flip_p2.sh`（iq3/iq1 grid4 宽化 flip 驱动）
- `experiments/active/g8-stage3-attack/P1-remainder-raw/run_widen_flip_iq4nl.sh`（iq4_nl 宽化 flip）
- run 目录 `experiments/runs/20260717T193812Z-iq3_xxs-k1-a4db452a/`（objdump 前后 + verify/measure stdout + leaf_iq3_xxs_vlen256_half16.c + row.csv·门与工具.md:413 指针）

---

## 项2 · T-P 全行 + T1d 双实例（原样）

### 2a. T-P（`experiments/active/result-tables/T-P_construction_param_chain.csv`·全 16 行）
表头：`cell,leg,capability_fact,plan_param,closed_form,closed_form_site,predicted,observed,product_form,consistency,leg_status,repro_cmd`

**q3_K@rvv 链（half_lanes = θ·随 march=c 输入变）·原文照抄关键列**：
- `1_fact_to_plan_closedform` 闭式：`half_lanes = (minVLEN<128) ? 0 : min(minVLEN/16, 16)`，site `RVVLowerQuantContraction.cpp:916-921`
  - march=`<empty>` ⇒ deriveMinimumVLEN=0 ⇒ predicted 0 / observed 0 / `no repack op (block-dot stub, fail-closed)` / MATCH（行2）
  - march=`rv64gc` ⇒ minVLEN=0 ⇒ 0 / 0 / fail-closed / MATCH（行3）
  - march=`rv64gcv_zvl64b` ⇒ minVLEN=128 ⇒ 8 / 8 / `numHalves=2 disjoint strip(s)` / MATCH（行4）
  - march=`rv64gcv_zvfhmin` ⇒ minVLEN=128 ⇒ 8 / 8 / `numHalves=2` / MATCH（行5）
  - march=`rv64gcv_zvl256b_zvfhmin` ⇒ minVLEN=256 ⇒ **16 / 16 / `numHalves=1 disjoint strip(s)`** / MATCH（行6）
  - march=`rv64gcv_zvl512b` ⇒ minVLEN=512 ⇒ 16 / 16 / numHalves=1 / MATCH（行7）
  - march=`rv64gcv_zvl1024b` ⇒ minVLEN=1024 ⇒ 16 / 16 / numHalves=1 / MATCH（行8）
- `3_product_to_objdump`（site `RVVLowerQuantContraction.cpp:1423`·`numHalves = 16 / half_lanes`）：
  - march=`rv64gcv_zvfhmin`(minVLEN=128·half_lanes=8) ⇒ numHalves=2·**vwmacc.vx=2048**·ELF `q3k.rv64gcv_zvfhmin.o (115784 bytes)`（行9）
  - march=`rv64gcv_zvl256b_zvfhmin`(minVLEN=256·half_lanes=16) ⇒ numHalves=1·**vwmacc.vx=1024**·ELF `q3k.rv64gcv_zvl256b_zvfhmin.o (58808 bytes)`（行10）
  - 编译链注：`DONE-local (clang++-20 --target=riscv64 + llvm-objdump-20; NO board)`
- `3_consistency_verdict`（VLEN128 vs VLEN256·same fixture·march= only）：`half_lanes 8 vs 16`·`ratio(vwmacc.vx) == ratio(numHalves) == 2/1`·predicted 2.0000 / observed 2.0000 (2048/1024)·**CONSISTENT**（行11）

**q4_0@ime 链（rpfIn/rpfAcc/vreg_floor(njw)/deploy-decision = θ·随 vlen_bits=c 变）·site `IMEBackendEmissionDriver.cpp:446-497`·闭式**：
`rpfIn=max(1,ceil(4*8*8/vlen)); rpfAcc=max(1,ceil(4*4*32/vlen)); floor(njw)=rpfIn*(1+njw)+njw*rpfAcc<=32; rpfIn!=1 => narrow`
- vlen_bits=128 ⇒ `rpfIn=2 rpfAcc=4 floor=None narrow fallback` predicted / observed `rpfIn=2 rpfAcc=None floor=None narrow fallback [unobserved: rpfAcc]` / `emitted leaf: narrow(no _w2 leaf)` / MATCH(2 field)（行12）
- vlen_bits=256 ⇒ `rpfIn=1 rpfAcc=2 floor=7 deploy IME-VMADOT-TILE-W2` predicted==observed / `emitted leaf: mac_kloop_w2` / MATCH(5 field)（行13）
- vlen_bits=512 ⇒ `rpfIn=1 rpfAcc=1 floor=5 deploy IME-VMADOT-TILE-W2` predicted==observed / `mac_kloop_w2` / MATCH(5 field)（行14）
- `probe_table_vs_live`(vlen_bits=512·vreg_floor(njw=2))：`floor = rpfIn*(1+njw)+njw*rpfAcc = 1*3+2*1 = 5`·site `:355 (table=7) vs :483-485 (live)`·`live != table => computed, not looked up`·**table=7 live=5**·**DIVERGE-as-predicted**（行15）
- `3_product_to_objdump`(vlen_bits=256·njw=2/mac_kloop_w2 leaf)：**PENDING·`BLOCKED-toolchain`**——`upstream clang-20 rejects xsmtvdotii (…unsupported non-standard user-level extension 'xsmtvdotii'); needs SpacemiT GCC-15.2 fork on k1. Board time = P line. NOT ATTEMPTED.`（行16）

### 2b. T1d 双实例（`experiments/active/result-tables/T1d_dual_instance.csv`·原样）
表头注（行1）：`# T1d dual-instance demonstration -- ONE schema, TWO board fact instances, core diff = 0. Structural evidence (C1); NO perf claim.`
schema 行3：`schema_sha256[:16]=52db859bb046b492`·`weft_opt_sha256[:16]=135d7bee06b6090f`

**两 fact 实例（原文关键列）**：
- `inst-rvv`：board=rvv·fact_vlen_bits=**128**·hints=`rv64gcv_zvl128b`·ime_status=**missing**·`legal_variant_set_size=1`·set=`{rvv_typed_body}`·infeasible=`{ime_keyed_body}`·guard_keys=`{"rvv":"available","spacemit_ime":"unavailable"}`·selection_chosen=**rvv_typed_body**·selection_reason=**only_feasible**·**core_diff=0**·declared_instance_hash=`bff6246419427188939ec1ab2f987d81c9325249a7b8601d0c19bdcab5c2fb2f`（行7）
- `inst-k1`：board=k1·fact_vlen_bits=**256**·hints=`rv64gcv_zvl256b`·ime_status=**available**·`legal_variant_set_size=2`·set=`{ime_keyed_body,rvv_typed_body}`·infeasible=`{}`·guard_keys=`{"rvv":"available","spacemit_ime":"available"}`·selection_chosen=**ime_keyed_body**·selection_reason=**static_order**·**core_diff=0**·declared_instance_hash=`55684ba31fc409ee427c605b825f24bdf98491839304a2b45e1025e328aa475f`（行8）

**probes（SECTION 2·原文）**：
- `P1-hash-divergence`(rvv|k1)·PASS·`ONE byte-identical body, two fact instances -> 2 distinct declared_instance_hash (rvv=bff624641942... k1=55684ba31fc4...)`（行12）
- `P4-march-channel`(rvv|k1)·PASS·`march= is a PASS OPTION, not an in-IR fact: minimum_vlen 128->256 flips chosen LMUL m2->m1 (reason=prior) while declared_instance_hash stays IDENTICAL (842c4a01e778...)`（行18）
  - [框架备注·非判断]：此行 minimum_vlen(c 输入)128→256 → chosen LMUL(θ) m2→m1·reason=prior·hash 不变。

**core diff=0 机检（SECTION 3·原文）**：
- `C1-f1-zero-branch`·PASS·`[f1-zero-branch] GREEN: zero family-name-keyed branch across 69 core files (I3 holds)`（行22）
- `C2-fact-confined-diff`·PASS·`non-fact surface byte-identical; sha256(stripped)[:16]=65320bac3b27ebb4; retained body: chars=1256 lines=25 (emitted, not transcribed)`（行23）
- `C3-same-core-binary`·PASS·`sha256(build-weft/bin/weft-opt)[:16]=135d7bee06b6090f ; pipeline=--weft-select-variants`（行24）

---

## 项3 · iq3_s 板间翻转原始记录（VLEN128→256 half_lanes 8→16）

来源：`.trellis/spec/issues/性能与测量.md:162`(ISSUE-019)、`:348`(ISSUE-102)、ISSUE-005 `:45`、门与工具.md:406。

原文照抄（ISSUE-019 `性能与测量.md:162`·同一份发射产物板间翻转）：
> 活证 = iq3_s 板间翻转：同一份发射产物 rvv 1.34 赢 / k1 0.61 输。

原文照抄（ISSUE-102 `性能与测量.md:348`·同 leaf·storm 相同·verdict 相反）：
> **同一 leaf·storm 相同·verdict 相反**：cold_X **rvv iq3_s 1.34 WIN** / iq3_xxs 0.95 near-parity（VLEN128）·**k1 iq3_s 0.61 / iq3_xxs 0.65 LOSS**（VLEN256·目标 0.8）。⟹ **storm 不是 k1 gap 的判别因**

原文照抄（ISSUE-005 `性能与测量.md:45`·8 board-cell 落账数）：
> vs 部署的 vl 专化对手 rvv 0.67 / 0.57 / 0.95 / 1.34 · k1 0.59 / 0.59 / 0.65 / 0.61（8 格输 6）… iq3_s 板间翻转（rvv 赢/k1 输·同 leaf）→ K 线手调攻坚候选。
> （逐格取自 `experiments/active/g8-stage3-attack/P2-grid4-raw/*_measure.log` 的 `ratio_cold_X`·2-seed）

**翻转前后（同 leaf·c 板输入变·θ half_lanes 变）**：
- iq3_s@rvv VLEN128(half_lanes=8·VLMAX=8 满用) = **1.34 WIN**
- iq3_s@k1 VLEN128-fixture(half_lanes=8·跑在 VLEN256 硅=半宽) = **0.61 LOSS**
- iq3_s@k1 宽化 VLEN256(half_lanes=16) = **1.20 WIN**（部署后 1.21·门与工具.md:406/:393）

半宽根因（ISSUE-102 `性能与测量.md:349`·原文）：
> **真 k1 bottleneck = VLEN256 半宽欠用**：iq3 group 处理固定 **AVL=8**（8 元素 grid group），编译器选 `mf2`/`m1`（VLEN128 下 VLMAX=8 满用·VLEN256 下 VLMAX=16 → **每条向量 op vl=8/16 半宽**）

---

## 项4 · K1 家族反证 FINDING 关键行

> 说明：supervisor 具名 `:59/:106/:114-116`。仓内匹配的 K-quant decode 逐板 board-difference 文件 = `experiments/active/g8-stage3-attack/A2-batch5-kquant-decode-M1.md`（探针在 :59·per-board 判读表在 :104-118）+ 配对反证 `A2-batch4-gemm-decode-M1.md`。以下原文照抄，含 opp 身份探针与逐格数。

### 4a. n_sb 判别键反证（家族级·ISSUE-101·`性能与测量.md:334-335`）原文：
> **q4_K@k1 n_sb=8 → PASS 1.535**（batch4）vs **q5_K@k1 n_sb=8 → named-X LOSS**（batch5）——**同板·同编译器·n_sb 均=8·verdict 相反** ⟹ **sub-block 数不是判别键**。

### 4b. q4_K@k1 vs @rvv 板+对手强度分裂（`A2-batch4-gemm-decode-M1.md:21`·:91·:95·:108）原文：
> | **q4_K** | **0.066× named-X**(GCC-DEATH) | **0.361× named-X** | **1.535× PASS** | byte-exact 0/512(vs ggml) | ★**board+opp-strength 分裂**：k1 赢**弱 opp**(0.80ms)·rvv 输**强 opp**(0.22ms native-vec)·super-block fold@M=1 不摊销·rvv-gcc = **[CASE-KQUANT-GCC-CODEGEN] gcc-death**(vsetvl 1387 vs clang 57) |（:21）

> **★board+opp-strength 分裂（honest 关键）**：verdict 由 **opp 强度** + **board** 定：rvv q4_K block-dot(gcc-15 stock·0.22ms) 比 k1(clang-18 stock·0.80ms) **快 3.6×** … → k1 赢弱 opp·rvv 输强 opp。**非我方 kernel 双板异**(k1 leaf 0.53ms ≈ rvv-clang leaf 0.60ms·近同)·**是 opp 强度异**。（:95）

> | k1·clang-18(what-if) | vsetvl=57 clean | 0.522–0.526 | 0.804–0.806 | 1.53/1.53/1.54/1.54 | **1.535** | **PASS** | 赢·但 opp=k1 q4_K block-dot **偏弱**(0.80ms)·非 hand-brick·hl8 leaf VLEN256 half-util(未占满宽·真部署 hl16 更快)·front-door declines(what-if) |（:91）

### 4c. A2-batch5 opp 身份探针（`:59`）+ 逐板 decode 判读表（`:104-118`）原文：
探针（`:59`）：
> **探针（opp 身份·符号级）**：rvv gcc-15 stock `ggml_vec_dot_qX_K_q8_K` @ q2_K=0x92316 / q3_K=0x930a6 / q5_K=0x93146 / q6_K=0x935de；k1 stock @ q2_K=0xa034c / q3_K=0xa15da / q5_K=0xa1ed8 / q6_K=0xa21d2。**均 stock block-dot·native-RVV inline·非 hand-brick**。

逐板判读表（`:107-114`·rvv vs k1 同格 cold）：
> | gemm_tile | q2_K | decode | **named-X** | **0.0685**(gcc GCC-DEATH)/0.3635(clang18) | PASS | **0.9585**(clang18·what-if·near-parity) |
> | gemm_tile | q3_K | decode | **named-X** | **0.0830**(gcc)/0.2189(clang) | **named-X** | **0.4252**(clang18·what-if) |
> | gemm_tile | q5_K | decode | **named-X** | **0.1273**(gcc)/0.5040(clang) | **named-X** | **0.6806**(clang18·what-if)·[GAP-Q5K-QH-REGCLIFF] per-board |
> | gemm_tile | q6_K | decode | **named-X** | **0.0535**(gcc)/0.2595(clang) | **named-X** | **0.3771**(clang18·what-if) |

gcc_death_flag（`:114`）原文：`gcc_death_flag=**TRUE 全 4 格@rvv-gcc**（vsetvl 922–2952·[CASE-KQUANT-GCC-CODEGEN]）`
provenance（`:115`）：`A2-batch5-kquant-decode-M1-raw/logs/{rvv,k1}_run.log` + `{rvv,k1}_build_seal.txt`

---

## 项5 · C920 9/9 关键行 —— 【部分采到 + 缺什么见下】

> 采到：C920 = T-Head XuanTie xtheadvector **RVV0.7 板**（板间行为差异·ISA-generation 轴）的现成证据 = capability-model 层 lit（编译器证据·非板测）。
> 采不到：与「C920」直接绑定的「**9/9**」板测/通过数——仓内 `9/9` 命中全是 fp16 `h2f 9/9 textbook IEEE`（各 driver）与 `bench --self-test 9/9`（门与工具.md:86）+ T4b/T-CENSUS 的 9/9，**均与 C920 无关**（详见 cant_collect）。

### 5a. C920 板间行为差异（同 body → 不同合法性·c 板输入=rvv_version 驱动 θ）
来源：`test/Conversion/EmitC/rvv-capability-profile-divergence-rvv07-vs-rvv10-policy.mlir`。原文照抄：
> * Profile A (--march=rv64gcv, RVV1.0): the agnostic-policy body is ACCEPTED and lowered to __riscv_vadd_vv_i32m1.（:10-11）
> * Profile B (--march=rv64gc_xtheadvector, RVV0.7 / C920): the IDENTICAL body is REJECTED fail-closed -- RVV0.7 LACKS the ratified ta/ma policy, so the version capability gates the agnostic-policy body out (no EmitC emitted).（:12-14）

FileCheck STAMP（`:92-99`·同一 bare @rvv provider·march 派生的 c 事实差）原文：
> STAMP-RVV10: `rvv_version = "1.0"` · `supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"` · `supported_sew = "8,16,32,64"`
> STAMP-RVV07: `rvv_version = "0.7"` · `supported_lmul = "m1,m2,m4,m8"` · `supported_sew = "8,16,32,64"`

硬件依据（`:24-25`）原文：
> the C920 runs 0.7.1 `th.v*` under rv64gc_xtheadvector and SIGILLs on a 1.0 rv64gcv binary (proven on the real silicon).

### 5b. C920 = fractional-LMUL 有无（c 输入·源注释）
来源：`include/Weft/Plugin/RVV/RVVCapabilityProfile.h`。原文照抄：
> :32 `HARDWARE FACT (proven on the C920): rv64gc_xtheadvector runs 0.7.1 th.v*;`
> :133 `* RVV0.7.1 (XuanTie xtheadvector on the C920) has NO fractional LMUL at all`
配套源点：`RVVCapabilityProfile.cpp:264`(RVV0.7.1 NO fractional LMUL)·`RVVRepackStripWidthMaterialization.cpp:37,103`·`Passes.td:386`(RVV0.7 ALSO stamps integer_core_lmul)·`RVVToEmitC.cpp:208,2307`。

### 5c. C920 相关 lit 资产（现成·数量）
- C++ 测：`test/CMakeLists.txt:228` `weft-rvv-capability-profile-divergence-test` + lit `test/Plugin/rvv-capability-profile-divergence.test:1`
- EmitC divergence .mlir（4 支）：`rvv-capability-profile-divergence-f64-coverage.mlir` / `-live-probed-march.mlir` / `-one-kernel-two-profiles.mlir` / `-rvv07-vs-rvv10-policy.mlir`

---

## 项6 · 累加器 LMUL 两次证伪原始记录（[GAP-P1]·候选 f 形状/板/败在什么数）

### 6a. 候选 f 形状 + 现行 θ（源码原文·`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`）
`selectRepackAccumulatorLMUL`（`:1287-1304`）原文关键行：
> `if (isRVV0p7) return {/*useM1=*/true, "correctness-rvv0p7"};`
> `constexpr std::int64_t kVectorRegisterBudget = 32;`
> `const std::int64_t m1ChainRegisterFootprint = getRVVLMULRegisterFootprint("m2") + getRVVLMULRegisterFootprint("m4");`  // 候选 f 形状 = i16m2 product + i32m4 accumulator
> `const bool m1Constructible = capabilityHalfLanes != 0 && m1ChainRegisterFootprint <= kVectorRegisterBudget;`
> `// [GAP-P1]: default mf2 -- never blind-widest; only a board measurement flips.`
> `return {/*useM1=*/false, "capability-default-mf2"};`

候选身份 + 空表（`:1271-1279`）原文：
> `// STAGE THREE populates this per-format (x board) board-measured m1-vs-mf2 crossover. EMPTY today: nullopt for every format => the mf2 default holds (byte-exact). q4_0 is the documented m1-faster CANDIDATE and q8_0 the mf2-faster candidate, but NEITHER is asserted here without a board number ([GAP-P1]: no projection; the board key is threaded in at STAGE THREE).`
> `inline std::optional<bool> lookupRepackMeasuredM1Faster(...) { return std::nullopt; }`

reroll-禁令（`:1255-1257`）原文：
> `// We do NOT blind-select the wider m1 (that is exactly what the gate4 widest-legal selector does for a DIFFERENT kernel path; reusing it here verbatim would re-commit [GAP-P1]).`

[框架备注·非判断]：源注释显式称冷启动 mf2 为 [GAP-P1]（`ISSUE-117` `性能与测量.md:424`：「冷启动 mf2 = **[GAP-P1] IRON RULE 刻意钉**」；`:424` 「widest-legal 公式 `core_lmul=f(VLEN,块宽)` 被 [GAP-P1] 明禁（widen-to-m1 falsified 2×·byte-exact hazard）」）。

### 6b. 证伪 #1 + #2（q4_K repack GEMM full-unroll 寄存器压力「opening」·板=rvv/VLEN128）
来源：`experiments/archive/l1-kquant/l1-reroll-q4k-repack-gemm/reroll_findings.md`。
Board（文首）：`rvv / VLEN128, core 8, 2.6 GHz, governor=performance (measured 2026-07-08)`

Bottom-line 原文（两 lever 均证伪）：
> Both cheap levers on the q4_K repack GEMM's full-unroll register-pressure opening are now falsified on silicon: **byte-exact reorder (#1) = NULL (+0.65%, noise floor)**; **naive structural RE-ROLL (#2) = -11% throughput, vsetvli +43%, spill +40%, parity 0.962x -> 0.860x**.

证伪 #2 逐指令表（objdump·原文）：
> | PRE  -O2 (measured binary) | vsetvli **53** | spill 84 | reload 88 | vwmacc 2240 | textB 23486 |
> | POST -O2 (measured binary) | vsetvli **76** | spill 118 | reload 127 | vwmacc 320 | textB 10220 |
A/B（败在什么数·原文）：
> nr=64, N=12: ours_pre 4.2398 -> ours_post 3.7878 GMAC/s => **0.893 (~-10.7% REGRESSION)**
> nr=16, N=10: ours_pre 4.2460 -> ours_post 3.7853 GMAC/s => **0.892 (~-10.8%)** SHAPE-ROBUST
> PRE parity = 0.962x → POST parity = 0.860x（~10.6% FURTHER from parity）
byte-exact 门（`:0. Byte-exact identity gate`）：int/norm regime `IDENTICAL, 0 mismatch`（RE-ROLL 保 op order）。

### 6c. 相邻 m1-widen 板数（k1/VLEN256·q8_0 mbf sweep）
来源：`experiments/archive/perf-historical/ondevice-q8_0-mbf/results.csv`。原文照抄（尾注）：
> `mbf2/mbf1 = 0.975 (mbf2 is 2.5% faster). mbf2 vs autovec still ~0.77-0.91x (< 1.0). DECISION: NOT FIXED.`
> `mechanism: … q8_0 34-byte block stride (2B scale gap) blocks a 64-wide load, so mbf does NOT fill the register. Register-fill fix needs integer_core_lmul=m1 (e8m1@VLEN256=32=one block).`
CSV 数据行（board=k1·SpacemiT-X60·vlen 256）：`core_mbf1 … kernel_ns 6900.4 … kernel_vs_ref 0.75-0.88` / `core_mbf2 … 6727.0 … 0.77-0.91`。

### 6d. RVV1.0 侧 mf2 单一合法（same-board 候选集·相邻证据）
`experiments/archive/g7/g7-l1-kernelsym-fullfill/rvv-batch/evidence.md:42`原文：
> **m1 whole-LMUL core 是 RVV0.7.1(`xtheadvector`) 专属分支**（passes.td:386-394）·**非本板(rv64gcv/RVV1.0) 合法 same-board 候选**（跨-ISA·板不跑 xtheadvector 码）
`SEL3-measurement-memory-design.md:24`原文：
> mf2 vs m1 | **{mf2} 单一合法**（双板·m1=RVV0.7 跨-ISA 非 same-board·`[repack-winA-always-mf2]`）

---

## 项7 · ISSUE-120 现状原文（iq2_xs/iq2_s VLEN256 byte-broken）

来源：`.trellis/spec/issues/发射器与架构.md:216-222`(ISSUE-120)。原文照抄：
> **状态**：待修（correctness 缺口·**pre-existing 非新引入**·iq2_xs 一直 rvv-only）。
> **实质**（task `07-19-deploy-iq2xs-fused` 板测暴露）：deployed iq2_xs vec_dot serial emit（`emitIQ2XSSuperBlockGridBody`）在 **VLEN256（k1）byte-broken**——`ours_vs_int_oracle=false`·mism=512（ggml+oracle 一致·仅我方错）。**pre-existing 原 `iq2_xs.kernel.c` 同样 broken**（非部署 agent 的改动）。根因 = pair-batched `vget_i8m4_i8m1` 结构 **VLEN128-form**（LMUL8 32-lane 假设·VLEN256 崩）。iq2_s 同 pair-batched 结构·**likely 同病**。
> **影响**：iq2_xs（+likely iq2_s）vec_dot **任何 k1/VLEN256 claim 前须先修**（per-VLEN 专化 或 unbatched VLEN-universal 路）。当前 master iq2_xs@k1 若有 PASS/verdict 须复核（可能 byte-broken 未察）。
> **保守默认**：iq2_xs/iq2_s vec_dot 维持 rvv-only（k1 不 claim）·待 per-VLEN 修。
> **出处**：task `07-19-deploy-iq2xs-fused`（VLEN256 byte-exact 门 mism=512·pre-existing 复核）。

配套（ISSUE-112 §簇B `性能与测量.md:406`·同缺口登记）原文：
> **+ 新 correctness 缺口**：iq2_xs/iq2_s pair-batched serial emit **VLEN256 byte-broken**（登记 [[ISSUE-120]]·需 per-VLEN 专化或 unbatched VLEN-universal 路方可 k1 claim）

[框架备注·非判断]：ISSUE-120 = 「同一 serial emit 结构在 VLEN256(c 板输入) 下 byte-broken」的板间行为差异事实·pre-existing。

---

## 采集计数（本包 7 项）

- **采到 6 项**：项1(宽化逐格+objdump) · 项2(T-P 16 行全 + T1d 双实例全) · 项3(iq3_s 板间翻转) · 项4(K1 家族反证:n_sb 反证 + A2-batch4/5 逐板) · 项6(累加器 LMUL 两证伪) · 项7(ISSUE-120)。
- **部分采到 1 项**：项5(C920) —— C920 板间行为差异（RVV0.7-vs-RVV1.0 divergence·supported_lmul m1,m2,m4,m8 vs mf8..m8·fractional-LMUL 有无·fail-closed）**采到**；但与「C920」绑定的具体「**9/9**」数**采不到**（见 cant_collect）。


<!-- ========== 包6 节 (源: pkg6-fakeknob-broken.md) ========== -->

# 包6 · 假旋钮与断链核查 — θ=f(g,c) 三角色全景盘点 v2

**HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（`git rev-parse HEAD` 实测 == 令定 ref；全文 file:line 直读工作树 == HEAD）
**只读·零施工·零板时·事实+出处·禁结论。** 是不是缺陷由上游判。

## 0 · 口径 / 输入

- 逐条核查对象 = 包1（`pkg1-theta.md`）的 **(b)/(c) 类 θ** 与 包3（`pkg3-g-descriptor.md`）的 **(iii) 类零消费 g 字段**。
- 包1 归类：**(b) = 1**（θ35）；**(c) = 21**（θ9–θ21 十三处 + θ27–θ34 八处）。⟹ 本包 (b)/(c) θ 核查目标 = **22 处**（θ9–θ21, θ27–θ35，θ22–θ26 是 (a) 派生，不在本包）。
- 包3 §3.1/§3.4：**g 桶 (iii) 零消费 = 0**（`comm -12 gfields.txt <(零 getter 名)` = 空）；描述符全集另有 34 个「零 typed-getter」名，**其中 0 个是 g 字段**。⟹ (iii) g 死料/待接的核查集为空（见 §3）。
- 术语：**「独立字面量假旋钮」= ISSUE-031(a) 型** = θ 值旁有**独立的焊死 vtype 串**（`OpaqueType::get(ctx,"vint8m1_t")` 等），且该 θ 又在下游被拼进 intrinsic callee 名 ⟹ 改 θ 值只改 callee 名、不改 vtype 字面量 ⟹ 产物自相矛盾（vint8**m1**_t 值喂 `__riscv_..._m2` intrinsic）。**「非假旋钮」= θ 值经 `deriveWideningChain`/字符串拼接派生出全部 vtype ⟹ 改值自洽。**

---

## 1 · (b)/(c) θ 逐条核查（① 改它变产物什么 · ② 旁有无独立字面量假旋钮）

### 1.A 真焊死 coreLmul（θ9–θ13）——**全部带独立 vtype 字面量 = 假旋钮（ISSUE-031(a) 型）·5 处**

共性机核：五处的 `coreLmul` **均在下游被 `riscvIntrinsicName("vle",8,coreLmul,"i8")` 等拼进 callee 名**（`awk` 抽证见下），而**紧邻的 vtype 是独立 `OpaqueType::get(ctx,"vint8m1_t")` 字面量**（不经 coreLmul 派生）。⟹ 改 `coreLmul="m1"→"m2"` 会改 callee 后缀但不改 vtype ⟹ 产物不自洽。这正是包1「5 真焊死」/ISSUE-118「5 真焊死(假旋钮形态)」。

| θ | 字面量 file:line | ① 改它变什么（代码路径指认） | ② 旁边独立 vtype 字面量 file:line（假旋钮证据） |
|---|---|---|---|
| **θ9** | `RVVToEmitCGridCodebook.cpp:497` `coreLmul="m1"` | 下游 `i8LoadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")`（:1038 gridOf4Dot 内）+ `riscvMaskNonzeroIntrinsicName(8,coreLmul,"u8",8)` + `vmerge...` callee 后缀（`awk 862..980` 命中 3 处） | **:498 `wideLmul="m2"`(独立)** · **:499 `vint8m1_t`** · :500 `vuint8m1_t` · :501 `vbool8_t` · :502 `vint16m2_t` · :503 `vint32m1_t`（全 `OpaqueType::get` 硬串·不经 coreLmul） |
| **θ10** | `RVVToEmitCGridCodebook.cpp:995` `coreLmul="m1"` | 同 θ9（:1038 `i8LoadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")` 在 iq3_s grid body） | **:996 `wideLmul="m2"`** · **:997 `vint8m1_t`** · :998 `vuint8m1_t` · :999 `vbool8_t` · :1000 `vint16m2_t` · :1001 `vint32m1_t` |
| **θ11** | `RVVToEmitCTernaryBinary.cpp:862` `coreLmul="m1"` | 下游 `riscvIntrinsicName("vle",8,coreLmul,"i8")`（两处 q8LoadCallee，`awk 862..1400` 命中）；pkg1 记 :1214/:1337 亦拼 vle8 | **:863 `wideLmul="m2"`** · **:864 `vint8m1_t`** · :865 `vint16m2_t` · :866 `vint16m1_t` · :867 `vint32m1_t` · :875 `i8HalfType=i8CoreType`(别名) · :876 `i16HalfType=i16WideType` |
| **θ12** | `RVVToEmitCBlockQuantLinear.cpp:111` `coreLmul="m1"` | 下游 :113? → `loadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")`；另一处把 **`i8CoreType`(vtype) 与 `coreLmul`(串) 作分立参数**传 helper（`... i8CoreType, "i8", coreLmul, ...`）⟹ 两者可各自失配 | **:112 `wideLmul="m2"`** · **:113 `vint8m1_t`** · :114 `vint16m2_t` · :115 `vint32m1_t`。**注释 :107-110 自陈**「structural constant, NOT a knob · hardcoded vint8m1_t/vint16m2_t + intrinsic-name 后缀共钉同一 shape · no IR width to read · [K-10] do not fake a knob」 |
| **θ13** | `RVVToEmitCBlockQuantLinear.cpp:404` `coreLmul="m1"` | 下游 `loadCallee=riscvIntrinsicName("vle",8,coreLmul,"i8")` + helper `("i8", coreLmul, ...)`（GEMM body） | **:405 `wideLmul="m2"`** · **:406 `vint8m1_t`** · :407 `vint16m2_t` · :408 `vint32m1_t`。注释 :401-403 同 θ12 自陈「NOT a knob · debake is a no-op」 |

### 1.B verifier-pinned coreLmul（θ14）——**非独立-vtype 假旋钮**（vtype 派生自 coreLmul·只 1 处硬 acc）

- **θ14** `RVVToEmitCCodebookFp4.cpp:109` `coreLmul="m1"`。
  - **①**：coreLmul 既拼 callee（:188 `riscvIntrinsicName("vle",8,coreLmul,"i8")`、:236 `"__riscv_"+mnemonic+"_u8"+coreLmul`、:248 `"__riscv_vrgather_vv_i8"+coreLmul`）**又**派生全部 vtype：:110 `deriveWideningChain(coreLmul)`、:111 `wideLmul=chain.l16`、:112-114 `("vint8"+coreLmul+"_t")`/`("vuint8"+coreLmul+"_t")`/`("vint16"+wideLmul+"_t")`。⟹ 改 coreLmul callee 与 vtype **同步变** = **自洽**（非 ISSUE-031(a)）。
  - **② 唯一硬字面量** = `:118 i32m1Type=OpaqueType::get(ctx,"vint32m1_t")`（reduction 累加器·order-free）。注释 :107-108「codebook gather REQUIRES the m1 anchor (VLMAX>=16 to index 16 entries); **the verifier pins it**」⟹ 焊死原因 = verifier 语义约束（VLMAX 门），非 vtype 断链。
  - **第二入口** `:734` 同名 `coreLmul` 为**函数入参**（`emit...` 第二 body），:749 `deriveWideningChain(coreLmul)` + :751-752 派生 vtype ⟹ 同样自洽。（`:694 if(!blockDot.getIntegerCoreLmul())` = 该文件的 verifier/前置检查存在。）

### 1.C live-default-override / value_or 软默认（θ15–θ21）——**vtype 全派生·非假旋钮**（θ19 有「值域重载」另一类断链）

| θ | 字面量 file:line | ① 改它变什么 | ② vtype 来源（是否假旋钮） |
|---|---|---|---|
| **θ15** | `RVVToEmitCTernaryBinary.cpp:1607` `coreLmul="m2"` + :1608-1609 `if(attrLmul) coreLmul=*attrLmul` | attr 覆写则改整数核宽；:1610 `wideLmul=(m2)?m4:m2` | **派生**：:1611-1616 `("vuint8"+coreLmul)`/`("vint8"+coreLmul)`/`("vint16"+wideLmul)`。仅 :1617 `vint32m1_t` 硬(acc)。**非假旋钮** |
| **θ16** | `RVVToEmitCBlockQuantLinear.cpp:14232` `coreLmul="m2"` + :14233-14234 override | 作参传 `emitQ1_0BlockDotBodyShared(...coreLmul...)`（:14236） | 共享体（:17342+）**派生**：`boolRatio=(m2)?"4":"8"`、`("vint8"+coreLmul+"_t")`；仅 `vint16m1_t` 硬(acc)。**非假旋钮**。注释 :14228-14231「this "m2" default is LIVE · q1_0 production e2e lowers at this default · fail-closing breaks byte-exact」 |
| **θ17** | `RVVToEmitCBlockQuantLinear.cpp:14315` `coreLmul="m1"` + :14316-14317 override | 作参传 `emitNVFP4BlockDotBodyShared(...coreLmul...)`（:14319） | 共享体 :749 `deriveWideningChain(coreLmul)` + 拼 vtype；仅 `vint32m1_t` 硬(acc)。**非假旋钮**。注释 :14309-14314「codebook gather pins m1 (VLMAX>=16)·verifier-restricted to m1·this "m1" default is LIVE」 |
| **θ18** | `RVVToEmitCKQuant.cpp:2918` `coreLmul=scaledDot.getIntegerCoreLmul().value_or("mf2")` | q4_K/q6_K scaled-dot 超块核宽 | :2919 `deriveWideningChain` → :2933-2937 `("vint8"+l8)`/`("vint16"+l16)`/`("vint32"+l32)` **派生**。:2931-2932 `vuint8m2_t`/`vint8m2_t` 硬(注释 :2927-2928「Region-A unpack fixed at m2·**unused by Region C**」)。**非假旋钮**。注释 :2913-2917「default LIVE·q4_K DECODE 前门 UNSTAMPED by design」 |
| **θ19** | `RVVToEmitCKQuant.cpp:3694` `coreLmulAttr=b3.getIntegerCoreLmul().value_or("mf2")` | 见下（**值域重载**） + :3719-3720 `coreLmul=aux8Free?"m2":coreLmulAttr` → :3721 `deriveWideningChain` → :3733-3735 派生 vtype | vtype **派生**（:3733-3735）；:3731-3732/:3736-3737 `vuint8m2_t`/`vint8m2_t`/`vint32m2_t`/`vfloat32m2_t` 硬(注释「copied VERBATIM from monolith」)。**vtype 非假旋钮**；但见 §1.E 值域重载 |
| **θ20** | `RVVToEmitCKQuant.cpp:5971` `coreLmul=coreOp.getIntegerCoreLmul().value_or("m2")` | 填 `IQ2XXSGridBodyContext cx{...coreLmul...}` → 传 θ21 | 本处无 vtype 字面量；派生在 θ21。注释 :5967-5970「integer_core_lmul 是 optional Win-A gearbox·iq2_xxs DECODE 前门 UNSTAMPED by design」。对照 :5962 `if(!coreOp.getNumGroups()) return notifyMatchFailure`（num_groups **fail-closed**·无 baked default）——**同 op 内 g 字段 fail-closed vs θ 软默认并存** |
| **θ21** | `RVVToEmitCGridCodebook.cpp:59` `coreLmul=cx.coreLmul` | 分派器读 context（值来自 θ20 类填入） | **派生**：:74 `wideLmul=(coreLmul=="m2")?"m4":"m2"`、:78 `("vint8"+coreLmul+"_t")`、:179-180 pairLmul/pairIdxLmul 派生。仅 :75 `vint32m1_t` 硬(acc)。**非假旋钮**（注释 :40-42/:72-73 自陈「FLIPS with the anchor, NOT always m4·byte-exact for any legal anchor」= 真 gearbox） |

### 1.D emit-变体门（θ27–θ30）——**字符串-模式选择器·非 vtype 假旋钮·默认全 dormant**

四门读 θ19 的 `coreLmulAttr`（`RVVToEmitCKQuant.cpp:3694` value_or("mf2")）作**字符串等值**：
- **θ27** :3695 `useRegisterFusion=(coreLmulAttr=="fused") && !hasQh`
- **θ28** :3696 `useVwredsum=(=="vwredsum") && !hasQh`
- **θ29** :3704 `useMintermVec=(=="minterm-vec") && !hasQh`
- **θ30** :3714 `useMlp=(=="mlp") && !hasQh`

**①**：默认 `coreLmulAttr="mf2"` ∉ {fused,vwredsum,minterm-vec,mlp} ⟹ 四门全 false ⟹ 产物 byte-identical（包1 记「全 dormant」）。改为某 sentinel ⟹ 路由到**不同 emit 区**（:3933 `if(useRegisterFusion)` / :3946 `else if(useMlp)` / :3962 `else if(useVwredsumBlockDot)` / :4024 `useVecMinTerm`）**且** :3718 `aux8Free=true` → :3720 `coreLmul="m2"`（强制）。⟹ 改值改的是**发射区选择**，不是单一 vtype。
**②**：**无独立 vtype 字面量假旋钮**（sentinel 命中即 coreLmul 被 :3720 强制 m2，vtype 仍经 :3721 派生自洽）。断链风险属 §1.E 值域重载，非 ISSUE-031(a)。

### 1.E flat-body 软默认旋钮（θ31–θ34）——**软默认 + fail-closed 拒绝守卫·非 vtype 假旋钮**

| θ | file:line | ① 改它变什么（代码路径） | ② 有无独立字面量假旋钮 |
|---|---|---|---|
| **θ31** | `RVVToEmitCBlockQuantLinear.cpp:14151` `multiBlockFactor=getMultiBlockFactor().value_or(1)` | 外层展开数（verifier bound {1,2,4}·:14145）；注释 :14150「other folds still require mbf==1 + elided default (fail-closed)」——**仅 q4_0 left_assoc body 物化全 cross product** | **无 vtype 字面量**（循环结构整数·非 vtype）。改值在非 q4_0 路 = 走默认（无效） |
| **θ32** | `:14152` `stripElided=getStripElision().value_or("robust")=="elided"` | 内 strip 循环是否省略（VLEN≥128 单 strip） | 同上·无 vtype 字面量·结构旋钮 |
| **θ33** | `:14438` `foldStructure=getFoldStructure().value_or("per-block")` | :14439 `if(foldStructure=="deferred-ordered" && !isQ80ScheduleParam) return notifyMatchFailure`——**改为 deferred-ordered 仅 q8_0 路 honored，否则 fail-closed 拒绝**（IR-says-deferred/emit-does-per-block is a lie） | **无 vtype 字面量**；断链风险 = 拒绝守卫（改值→match failure），非产物不自洽 |
| **θ34** | `:14455` `numericsTier=getNumericsTier().value_or("strict")` | :14456 `if(numericsTier=="relaxed" && !(deferred-ordered && isQ80)) return notifyMatchFailure`——relaxed 仅 q8_0 deferred 路存在，否则 fail-closed 拒绝 | **无 vtype 字面量**；同 θ33 拒绝守卫 |

### 1.F 审计镜像（θ35 · (b) 类）——**只写不读·改值零产物影响**

- **θ35** `RVVDequantDotSourceFrontDoor.cpp:903` `rvvVariant->setAttr("weft_rvv.gearbox_selected_integer_core_lmul", getStringAttr(selectedIntegerCoreLMUL))`。
  - **①**：**改此 attr 值 → 产物零变化**。真实路由用 `selectedIntegerCoreLMUL` 局部值（:875 `loadLMUL=selectedIntegerCoreLMUL`），**不回读此 attr**。注释 :900-902 自陈「AUDIT-ONLY provenance (**NOT the authority**) · a mirror, never the route/dtype authority」。
  - **② 读端机核（全树 excl `.inc`）**：`grep -rn gearbox_selected_integer_core_lmul lib/ tools/ include/ test/` = **写侧 2 处**（DequantDot:903 + `RVVReductionSourceFrontDoor.cpp:699`）+ **test .mlir 1 处**（作已盖 attr 出现）；**`getGearboxSelectedIntegerCoreLmul` typed getter = 0 命中**；**routing 读回 = 0**。⟹ 纯写侧镜像·无 vtype 字面量。

---

## 2 · (b)/(c) θ 核查小结（纯计数·口径）

- **核查 θ 总数 = 22**（(c) 21：θ9–θ21 + θ27–θ34；(b) 1：θ35）。
- **带独立字面量假旋钮（ISSUE-031(a) 型·改值→callee 名变而 vtype 不变→不自洽）= 5**：**θ9(GridCodebook:497) · θ10(GridCodebook:995) · θ11(Ternary:862) · θ12(BQL:111) · θ13(BQL:404)**。全部有紧邻独立 `wideLmul="m2"` + `vint8m1_t`/`vint16m2_t` 硬串，且 coreLmul 同时拼进 `riscvIntrinsicName` callee（机核已抽）。= 包1「5 真焊死」/ISSUE-118「5 真焊死(假旋钮形态)」一致。
- **非-假旋钮（vtype 派生自 coreLmul·改值自洽）= 8**：θ14(verifier-pinned·仅 acc 硬) · θ15 · θ16 · θ17 · θ18 · θ19(numeric 路) · θ20 · θ21。均经 `deriveWideningChain` 或 `("vint8"+coreLmul+"_t")` 派生；残留硬 vtype 均为 order-free reduction 累加器(`vint32m1_t`)或注释标「unused/Region-A」。
- **无 vtype 字面量（结构/模式/镜像·非 ISSUE-031(a)）= 9**：θ27–θ30（字符串-模式门·默认 dormant）+ θ31–θ34（软默认+fail-closed 拒绝守卫）+ θ35（只写不读镜像）。
- **另一类断链（非 vtype·如实记）= 值域重载**：`integer_core_lmul` 同一 attr 槽同时承载 {mf2,m1,m2}=LMUL（θ19 :3721 派生 vtype）**与** {fused,vwredsum,minterm-vec,mlp}=发射模式 sentinel（θ27-30 :3695-3714 字符串门）。命中 sentinel 时 :3720 强制 coreLmul="m2"。此为**语义重载**，非独立-vtype 假旋钮。

---

## 3 · (iii) 零消费 g 字段 → 前门 stamp 死料/待接核查

**核查集 = 空。** 包3 §3.1/§3.4 机核结论：**g 桶 (iii) 零消费 = 0**（64 个 g 字段全有 typed-getter 读者·`comm -12 gfields.txt <(零 getter 名)` = 空）。⟹ **没有「盖了值却零消费」的 g 字段**，故无 g 死料、无 g 待接可核查。

- **g 死料 = 0 · g 待接 = 0**（核查集为空）。
- 包3 §2.2 佐证：block-dot 家族 g 是「family header 声明→前门 stamp→发射器 fail-closed 读」的**活链**（源正本 `RVVMonolithicBlockDotFamily.h:1503-1526` 24 张 k*Facts；前门 stamp `RVVMonolithicBlockDotSourceFrontDoor.cpp` g-attr 字符串 `grep-c`=395；发射器 typed getter 直读）。⟹ g 前门 stamp **在跑·可达·被消费**。

**相邻但非 g 桶（如实并列·不计入 (iii) g 计数）**——包3 §3.4 记描述符全集有 **34 个「零 typed-getter」名（其中 g=0）**，属 c/θ/元数据桶。本包对其中与「假旋钮/断链」主题相关的两个 θ/c 名机核了前门可达性，结论**均非死料**（经泛型 `getAttr` 消费，非 typed getter）：
- `unroll_factor`（θ 桶·typed 0）：**读端活**——`RVVToEmitCDeferredDequant.cpp:181` `scope->getAttrOfType<mlir::IntegerAttr>("unroll_factor")`（泛型 getAttr·非 typed getter）；写端活——`RVVContractionSelectedBodyRealizationOwner.cpp:709` 等。（另有独立命名空间 `weft_rvv.low_precision_resource.unroll_factor` 一整套 stamp+read 在 `RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp`·不同 attr。）⟹ **待接/死料皆非·活链(泛型读)**。
- `selected_variant`（θ 桶·typed 0）：**活**——多插件以 `constexpr StringLiteral kSelectedVariantAttrName("selected_variant")` 泛型消费（Demo/Toy/TensorExtLite/Offload/RVVPackedI4Dot/RVVCodebookDot 前门 + `ConstructionProtocol.cpp:871` / `DemoExtensionPlugin.cpp:686` 校验读）。⟹ **活链(泛型读)**。
- θ35 镜像 `weft_rvv.gearbox_selected_integer_core_lmul`（本包 §1.F）：**写侧可达（2 前门在跑）·routing 读回 = 0**——是「stamp 在跑但无路由消费者」的镜像，非 g 字段（不入 (iii) g 计数）。

---

## 4 · 完整性附注（本 pin 机核·非 θ 计数增删）

- **包1 未列入 θ 的同族 coreLmul 字面量（本 pin `grep` 命中·补记，不改包1 计数）**：
  - `RVVToEmitCBlockQuantLinear.cpp:17663` `coreLmul="m2"` + :17664 override → 传 `emitQ1_0BlockDotBodyShared`（:17667，q1_0 第二 caller·monolith 路）。**vtype 派生自共享体·非假旋钮**（同 θ16）。
  - `RVVToEmitCCodebookFp4.cpp:734` `coreLmul`（函数入参·第二 body 入口）——**vtype 派生**（:749 deriveWideningChain·同 θ14）。
- θ27-30 发射区已机核为**真分立**（:3933/:3946/:3962/:4024 分支），确认「改 sentinel = 改发射区」而非空操作。
- 全部 (b)/(c) θ 的注释均以 `[A-line stage-3]` / `[K-10]` 标注（θ12/θ13「NOT a knob·do not fake a knob」；θ16/θ17/θ18/θ19/θ20「default LIVE·前门 UNSTAMPED by design·fail-closing breaks byte-exact」），照抄不判。

---

## 5 · 采不到 / 未核（如实）

- 包3 的 `scratchpad/gfields.txt`/`consumption.txt`（生成 g-桶零消费=0 结论的机算中间物）为**会话私有·本 pin 已不在**（`ls scratchpad/` 空）。⟹ 本包对「g (iii)=0」**未独立重跑全量枚举**，照引包3 机核结论；仅**抽验佐证**其「zero-typed-getter ≠ 零消费」推理（`unroll_factor`/`selected_variant` 经泛型 getAttr 活读，见 §3）成立。
- 34 个零-typed-getter attr 的「只写不读 vs 泛型 string 读」逐字段三分**未穷做**（包3 §7 亦标·超本包 fake-knob/断链范围）；本包仅核 2 个 θ 桶代表 + θ35 镜像。
- θ11 pkg1 记的下游 :1214/:1337 vle8 拼接**未逐行复读**（本包机核 :862..1400 窗抽到 `riscvIntrinsicName(...,coreLmul,...)` 命中即证 coreLmul 下游消费成立·未穷举全部拼接点）。
- 未编译/未跑 fixture（只读令）。所有「改值→不自洽 / 派生自洽 / dormant / byte-exact」断言 = 静态代码路径阅读（vtype 字面量 vs coreLmul-拼接 callee 的对照）+ 注释自陈引用·非运行验证。
