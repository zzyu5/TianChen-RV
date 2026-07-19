# 案头判定 · 18 单侧 f 逐判 + 11 零消费键二选一（增加公式 breadth · 只读 scoping-first）

> 任务权威 = PRD `.trellis/tasks/07-20-07-20-census2-singlesided-f-zeroconsume-keys/prd.md`（行动书 §3.3 + §3.6）。
> 数据源 = census v2 `experiments/active/theta-fgc-census-v2/00-census-final.md`（pkg2/pkg4）。
> **本役纯只读**：出判定 + plan/judgment 设计；实际改码后续按判定执行（避免与 phase-2 选择器工作冲突）。未改任何源，未 git commit。

## 0. 关键前提：census pin 已漂移（必须先讲清）

- **census v2 pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`**（pkg2/pkg4 全部 file:line 以此为准）。
- **当前工作树 HEAD = `e62dba1c2ebe199983088f42d18011bab4b6a2f4`**（`git rev-parse HEAD` 实测）。
- 二者之间已落地 7 个 commit 直接改动本役对象文件（`git log d173f4c2..HEAD -- RVVCapabilityProfile.{cpp,h}`），其中 **census pkg2 所描述的多条「零消费」状态已在 HEAD 被 wire 或 delete**（详见 §B）。
- 本判定**同时报两态**：census-pin 描述态（PRD 依据）+ 当前 HEAD 真实态（诚实校正）。凡 file:line 均**已在 HEAD 机核重取**（不照抄 pin 行号）。

漂移涉及的 commit（`git log --oneline`）：
- `abb7e0304` ★r5.1-g **T1 真封口: readRVVProviderVLenBBytes 定义+wiring+判决lit** → vlenb_bytes 已接上。
- `9df771a02` W3b **8 消费者从 deriveMinimumVLEN(march) 改读 provider fact** → VLEN 轴走管道。
- `848d61f31` W3 拔管道（strip 宽消费者 C2 改读 in-IR provider op）。
- `5c820847f` W2（含 **F25 恒真** 分析 + 5 constructor 生产者边界）。

---

## A. 18 单侧 f 逐判（§3.3 · 禁批发 · 一条一理由）

判据三类：**(1) 合法单侧**（本职即解析单侧输入·别误伤）/ **(2) 假单侧须补输入**（附 judgment 设计：改被丢弃/baked 的输入→θ 必须随之变 lit·验收=输出真变，非「参数加上了」）/ **(3) 零输入 OK**（写为什么零输入对）。

census pkg4 机读口径：仅读 c=10（F7·F25·F26·F27·F28·F29·F30·F31·F33·F35）/ 仅读 g=5（F5·F6·F12·F19·F24）/ 零输入=3（F4·F21b·F37）。

### A.1 仅读 c = 10

| F# | 签名 file:line（HEAD 机核） | 判定 | 一条理由（禁批发） |
|---|---|---|---|
| **F26** `deriveMinimumVLEN(march,hints)→VLEN` | `lib/Plugin/RVV/RVVCapabilityProfile.cpp:299` | **合法单侧** | 本职 = 从 -march/hints 解析板 VLEN bits；VLEN 是纯板事实，格式 g 不该、也无法改变它。这是全栈 VLEN 事实的唯一权威（PRD 点名的 Tier-2 纯 c 派生正例）。**别误伤。** |
| **F27** `deriveHasZvl128b(march,hints)→bool` | `:348`（`= deriveMinimumVLEN(...) >= 128`） | **合法单侧** | 纯 c 谓词，直接由 F26 派生；板有无 Zvl128b 与格式无关。合法。 |
| **F28** `deriveRVVVersion(march,hints)→{0p7,1p0,Unknown}` | `:535` | **合法单侧** | RVV 版本（分数 LMUL 有无）是硅/toolchain 事实，纯 c。供 F3 `isRVV0p7`、allow-list。合法。 |
| **F29** `deriveSupportedLMULAllowList(march,hints)` | `:270`（经 `deriveRVVVersion`） | **合法单侧** | LMUL 支持集 = 板能力 allow-list（legality gate 查询用），是板事实而非格式选择。合法。 |
| **F30** `deriveSupportedSEWAllowList(march,hints)` | `:232` | **合法单侧** | 同 F29：SEW 支持集是板能力 allow-list。合法。 |
| **F31** `deriveRepackHalfLanes(vlenBits)→half_lanes` | ①`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1226`；②`lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp:78` | **合法单侧（baked-g 由 verifier 钉死）** | `half_lanes = min(vlen/16, weightInterleave)`。版本①用常量 `kWeightInterleave=16`（`RVVLowerQuantContraction.cpp:164`）；表面像 baked g，但 **verifier 硬钉 `weight_interleave==16`**（`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:3332` `if(getWeightInterleave()!=16) emitOpError`）。⟹ clamp 上界是**结构强制不变量**，非活格式旋钮；min(vlen/16,16) 事实上是 c-only。合法（**caveat**：若 verifier 日后放开 ==16 收纳其它交织宽，版本①须改吃 weightInterleave 参数；现被 verifier 守住）。 |
| **F33** `enumerateRVVLowPrecisionAccumulatorLMULRungs(budget,reserve)` | `include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2044` | **合法单侧** | 按寄存器预算（c）枚举合法 i8→i16→i32 rung；source rungs {mf4,mf2,m1,m2} 是 widening-chain **结构常量**（非格式几何）。合法单侧 c。 |
| **F35** `enumerateRVVDotReduceDeferredWideLMULRungs(budget,reserve)` | `RVVGearboxSchedule.h:2211` | **合法单侧** | 同 F33：i16 链 rung 枚举，输入=寄存器预算（c），source LMUL 是结构常量。合法。 |
| **F7** `tilingVariantFeasibleSet(shape,vlenBits,vregCount)` | `include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:150`（`(void)shape;` at `:152`） | **假单侧 → 删参（shape 非 load-bearing）** | 签名声明 `shape`（g）却 `(void)shape;` 显式丢弃，可行集只由 c 门（`vlenBits<128 \|\| vregCount<=0` → 空；否则 {Plain,S6Tiled}）定。**判定：可行性（legality）确实 shape-无关**——S6Tiled 在任何 capable 板都可发射，shape 只决定 prior（那是 F6 `priorTilingVariantForShape` 的活）。⟹ shape 参是**遗留死参**，正确动作 = **删参并注明**（非补输入）。见下 judgment 设计。 |
| **F25** `selectIntegerCoreLMUL(module,march,hints)` | `lib/Plugin/RVV/FrontDoor/RVVReductionSourceFrontDoor.cpp:346`（另二同构：`RVVDequantDotSourceFrontDoor.cpp:375`、`RVVPackedI4DotSourceFrontDoor.cpp:208`） | **假单侧（baked g·当前 invariant-guarded）** | 签名只吃 c（march/hints→VLEN），但 inline descriptor **baked g**：`blockLen=kContractionBlockLen=32`、`quantFormat="plain-int8"`、anchors {m1,m2}。baked blockLen **是承重的**（coreLMUL 由 `stripVLMAX>=blockLen` legality 翻 m1↔m2）。当前**恒真**：三前门都 gate `dimSize(0)==32`（`RVVReductionSourceFrontDoor.cpp:136`、`RVVDequantDotSourceFrontDoor.cpp:144`）拒非-32 几何 ⟹ baked g 被 gate 守住（commit `5c820847f` 的「F25 恒真」已独立证）。见下 judgment 设计与补输入 plan。 |

**小计（仅读 c=10）：合法 8（F26/F27/F28/F29/F30/F31/F33/F35）+ 假单侧 2（F7 删参 / F25 baked-g）。**

#### F7 judgment 设计（验收=输出真变，否则死参该删）
- lit：构造两条 `TypedRepackGemmLoopBodyOp`，同 `vlenBits=256, vregCount=32`，只改 `shape`（一条喂 `AlreadyLean`=codebook/grid fold_model，一条喂 `MinFoldRegisterCliff`=`kquant_dmin_bsums_min`）。
- 断言：`tilingVariantFeasibleSet` 返回的**可行集在两条上恒等**（都 = {Plain,S6Tiled}）。
- **结论条件**：若两条可行集相同（预期如此）⟹ shape 对 feasibility 零影响 ⟹ **删 shape 参**（改签名为 `tilingVariantFeasibleSet(vlenBits,vregCount)`），并把 F8 `selectRepackTilingVariant` 的 prior 分支继续用 `shape`（F6 已经在那儿读）。若反例出现某 shape 让 S6Tiled 真不可行 ⟹ 改判为「补输入」，把 shape 接进 legality。**验收非「删了参」而是「删后所有现存 fixture byte-exact 不变」**（证明 shape 本就是 dead）。

#### F25 judgment 设计 + 补输入 plan（当前 invariant-guarded，低紧迫）
- judgment 设计（证 baked g 承重）：单测直调 `getRVVEffectiveWidthInvariantLMUL(minimumVLEN=256, sew=getRVVBlockDotStripSEW("m1"), blockLen=X, {m1,m2})`，扫 `X∈{16,32,64}`，断言返回 coreLMUL **随 X 翻转**（VLMAX==blockLen 的 LMUL 变）。翻转即证 baked blockLen 是 load-bearing 输入（非 inert）。
- 补输入 plan（**仅当**将来收纳非-32 int8 收缩几何时执行）：把 `blockLen` 从三前门的 op descriptor（`type.getDimSize(0)`）读出、作参数传入 `selectIntegerCoreLMUL`，替换 `kContractionBlockLen` 常量。**验收**：新增一条 `dimSize=16` 的合法前门 op，coreLMUL 输出与 `dimSize=32` 不同（θ 真随 g 变）。
- 现状裁：当前三前门 `dimSize==32` gate 使 baked g **恒真安全**，**无需立即补**；把上面 judgment lit 挂上防回归即可（记录 baked g 的承重性 + gate 的守护性）。

### A.2 仅读 g = 5（全合法单侧）

| F# | 签名 file:line（HEAD 机核） | 判定 | 一条理由 |
|---|---|---|---|
| **F5** `classifyTilingBottleneckShape(foldModel)` | `include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:114` | **合法单侧** | 本职 = 把格式 fold 几何（`foldModel` WHAT）分类成瓶颈 shape；瓶颈形状是纯格式属性，不需板输入。合法单侧 g。 |
| **F6** `priorTilingVariantForShape(shape)` | `RVVRepackTilingSelection.h:132` | **合法单侧** | shape（g 派生）→ tiling **prior**（cold-start 结构默认）；板轴由 measurement（F8/F9）另加。prior 是结构量，单侧对。合法。 |
| **F12** `repackColGroupOuterForLayout(weightStride,activationStride)` | `RVVRepackTilingSelection.h:441` | **合法单侧** | layout-prior 子 helper（stride 比较）；c 门（vlen/vreg）在调用者 F13（`:504`）里加。本函数只做布局比较，单侧 g 对。合法。（注：注释 `:438` 想要 cacheline(c) 做更优 cost model = 与 §B 的 cachelineBytes GAP 同源，但**当前**函数合法。） |
| **F19** `lookupRepackVlen256Decode(scaleModel)` | `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:686` | **合法单侧** | 格式键（scaleModel）查 board-measurement 表；board 条件（minVLEN≥256）由调用者 F1/F2 上游 gate。VLEN256 板轴 baked 在表语义里但 caller-gated，key=格式，单侧 g 对。合法。 |
| **F24** `lookupRVVScheduleDescriptor(kernelKey)` | `lib/Plugin/RVV/Schedule/RVVScheduleDescriptorRegistry.cpp:151` | **合法单侧** | registry 按 kernel key 分派 descriptor；c（minimumVLEN/budget）在下游 enumerate 加。注册表查表本职单侧 g。合法。 |

**小计（仅读 g=5）：合法 5，无假单侧。**

### A.3 零输入 = 3（全零输入 OK）

| F# | 签名 file:line（HEAD 机核） | 判定 | 一条理由 |
|---|---|---|---|
| **F4** `lookupRepackMeasuredM1Faster(scaleModel)` | `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1351`（恒 `nullopt`） | **零输入 OK** | 有意留空的 STAGE-THREE measured 表；对所有输入恒返 nullopt ⟹ 每 RVV1.0 格→mf2 默认→byte-exact。`[GAP-P1]` 铁律**禁投影**（widen-to-m1 已被证伪两次），无板测前唯一诚实动作就是不选。scaleModel 参 = 保留的未来键。**非假旋钮**，是文档化占位。零输入正确。 |
| **F21b** `lookupRollMeasuredBeneficial(vwmacc,coreLmul)` | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:3211`（恒 `nullopt`） | **零输入 OK** | 同 F4：rolled-vs-unrolled measured 表留空，unrolled 默认 byte-exact。两参保留为未来键。零输入正确（无板测=不改产物）。 |
| **F37** `makeRVVDotReduceMinimalDeferredM1Rung()` | `include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2265`（硬返 source mf2 / acc m1） | **零输入 OK** | 返回固定常量 = 最小合法 deferred-wide rung（mf2→m1，footprint 最小、恒合法、板无关）。这是**具名结构常量**而非决策，无 g/c 可读。零输入正确。 |

**小计（零输入=3）：零输入 OK 3，无补输入候选。**

### A · 总判定（18/18）

| 类别 | 数 | 具体 |
|---|---|---|
| **合法单侧** | **13** | F5·F6·F12·F19·F24（5 仅读 g）+ F26·F27·F28·F29·F30·F31·F33·F35（8 仅读 c） |
| **假单侧（须动作）** | **2** | F7（删 shape 死参）· F25（baked blockLen 承重·当前 gate-恒真·补输入仅在收纳非-32 几何时） |
| **零输入 OK** | **3** | F4·F21b·F37（皆有意占位/结构常量） |

**无「待裁」。** 两条假单侧都给了 judgment 设计（改输入→θ 变 lit）。

---

## B. 11 零消费键二选一（§3.6 · 禁第三态「留着以后用」）

census pkg2 口径（pin d173f4c2）：9 provider-property（vlenb:bytes · clang:version · cmake:version · compile_run 4 键 · march:value · mabi:value）+ 2 probe 字段（cachelineBytes · imePresent）= 11 键；另 `deriveIMEPresent` 零调用者。
判据：**接上** iff 存在 GENUINE 待接消费者（如 T1 vlenb→VLEN 有真消费者用错源）；否则 **删除**（诚实·[K-10] 禁造假旋钮）。**禁第三态。**

**★ HEAD 校正**：census pin 之后，`abb7e0304`/`9df771a02` 等已把 census 预言的动作**执行掉一大半**——vlenb_bytes 已 wire、两 probe 字段已从 struct 删除。故下表每键标「census-pin 态 → HEAD 真实态」。

| # | 键（provider-property / probe） | 产出处（HEAD 机核） | HEAD 消费者机核 | 二选一判定 | 依据 / 考古（一句） |
|---|---|---|---|---|---|
| 1 | **vlenb_bytes**（`rvv.vlenb_bytes` / `bytes`） | `RVVCapabilityProfile.cpp:766` | **已接上** `readRVVProviderVLenBBytes()` `:389→:411` 读 `bytes` → `minimum_vlen` provider fact → `resolveRVVMinimumVLEN`（8 消费者，`9df771a02`） | **接上（HEAD 已完成）** | census 的「零消费」已被 `abb7e0304` 退休；代码注释 `:396` 原文「(Retires the zero-reader status of the vlenb producer.)」。这就是 PRD 引的 T1 干净案例（真消费者用错源→改读 hardware fact）——**已 wire，无残留动作**。 |
| 2 | **clang:version**（`rvv.toolchain.clang`） | `:772` | `git grep` 库内消费 = **0** | **删除** | toolchain 版本串 = I8 证据线（provenance），非 θ 输入；无任何 selector 读。struct `clangAvailable` 是探测期门（保留），但 in-IR **version 串属死 stamp**。 |
| 3 | **cmake:version**（`rvv.toolchain.cmake`） | `:777` | 消费 = **0** | **删除** | 同 clang:version：构建 provenance 回声，零消费。 |
| 4 | **compile_run:selected_march** | `:781` | 消费 = **0**（`selected_march` 库内唯一命中是 `RVVDialectInternal.h:271` 的**另一个** op-attr 常量 `kSelectedMarchAttrName`，非本 provider property） | **删除** | 与 march:value 冗余；march 承重值经 struct 参 + minimum_vlen 管道走，此 property 无人读。 |
| 5 | **compile_run:selected_mabi** | `:783` | 消费 = **0** | **删除** | mabi 不驱动任何决策（仅 march/hints 定 VLEN/version）。纯 provenance。 |
| 6 | **compile_run:source_sha256** | `:785` | 消费 = **0** | **删除**（I8 caveat） | 复现性 provenance（I8 证据线）。严格二选一→删；若项目维护 build-provenance 审计，仅此/binary_sha256 有**非 compute** 的存档理由，但那是证据线非「键」，不在 selector 面。 |
| 7 | **compile_run:binary_sha256** | `:787` | 消费 = **0** | **删除**（I8 caveat） | 同 source_sha256：I8 复现性证据，零 selector 消费。 |
| 8 | **march:value**（`rvv.toolchain.march`） | `:797` | 消费 = **0** | **删除** | march 承重值经 **struct 参**（`selectIntegerCoreLMUL(...,march,...)` / `deriveMinimumVLEN(march)` fallback）+ VLEN 轴经 `minimum_vlen` provider fact 走管道（`9df771a02` 拔管道拔的是 minimum_vlen，**不是** march:value）。此 in-IR march 串 = 冗余镜像。与 vlenb_bytes 不同（后者带 -march 表达不了的真实板 VLENB fact），march:value 无 probe-only 新信息，删。 |
| 9 | **mabi:value**（`rvv.toolchain.mabi`） | `:803` | 消费 = **0** | **删除** | 同 selected_mabi：mabi 零决策消费。纯 provenance 镜像。 |
| 10 | **cachelineBytes**（probe struct 字段） | **HEAD 已从 struct 删除**（`RVVProbeCapabilityFacts` 现 13 字段·无此字段） | 全仓 `grep cachelineBytes` = **0**（唯一相关命中是 `RVVRepackTilingSelection.h:438` 的 GAP 注释，非字段） | **删除（HEAD 已完成）** | 清晰 dead code：当初为「cacheline-aware repack loop-order/tiling cost model」探测（`:438` 记 GAP），该 cost model 从未建 ⟹ 无真消费者 ⟹ 已删。 |
| 11 | **imePresent**（probe struct 字段） | **HEAD 已从 struct 删除** | 全仓 `grep imePresent` = **0**（连赋值点都无） | **删除（HEAD 已完成）** | 清晰 dead code：当初为探测 IME 存在，但 IME 分派实际用 march token `xsmtvdotii` / capability op status 直判，imePresent 从未接线 ⟹ 已删。 |
| + | **deriveIMEPresent**（零调用者 inline fn） | **HEAD 已删除**（全仓 `grep deriveIMEPresent` = **0**，定义都不在了） | 0 caller | **删除（HEAD 已完成）** | 清晰 dead code：算 imePresent 的 token 检查 helper，零 caller，随 imePresent 一并已删。 |

### B · 总判定（11 键 + deriveIMEPresent）

| 判定 | 数 | 具体 | HEAD 状态 |
|---|---|---|---|
| **接上** | **1** | vlenb_bytes | **已完成**（`abb7e0304` T1 封口·真消费者=resolveRVVMinimumVLEN） |
| **删除** | **10** | clang:version · cmake:version · selected_march · selected_mabi · source_sha256 · binary_sha256 · march:value · mabi:value（8 provider provenance 回声，**HEAD 仍在产出、待删**）+ cachelineBytes · imePresent（2 probe 字段，**HEAD 已删**） | 8 待删 / 2 已删 |
| （附）deriveIMEPresent | 删除 | | HEAD 已删 |

**清晰 dead code（PRD 点名核对）**：cachelineBytes · imePresent · deriveIMEPresent —— **三者在 HEAD 均已删除**（census 的清晰-dead-code 预言已被执行）。
**零第三态**：无一条判「留着以后用」。8 个 provider provenance 回声给的是**删**（无真消费者），非「留」。

---

## C. 最高价值 3 条 actionable

1. **[删·仍待执行] 删 8 个 provider-property provenance 死 stamp**：`clang:version` / `cmake:version` / `compile_run{selected_march,selected_mabi,source_sha256,binary_sha256}` / `toolchain.march:value` / `toolchain.mabi:value`（产出 `RVVCapabilityProfile.cpp:770-805`）。**需要**：确认无审计/序列化路径依赖（`git grep`=0 已证 selector 侧 0）；`source_sha256`/`binary_sha256` 若要留 build-provenance，须显式标 I8-证据线（非「键」、不进 selector 面）——否则一并删。零风险（无消费者）。

2. **[已完成·须校正 census] vlenb_bytes 已 wire——census pkg2「零消费 9」对 HEAD 已 STALE**：`abb7e0304` 起 `readRVVProviderVLenBBytes→minimum_vlen→resolveRVVMinimumVLEN`（8 消费者从 `deriveMinimumVLEN(march)` 改读 provider fact，`9df771a02`）。这正是 PRD 引的 T1 干净接上案例，**已落地**。**需要**：把 census 的 vlenb「零消费」标注更新为「已接上」，零消费 provider-property 从 9 降到 8（其余 8 全判删）。

3. **[假单侧·防回归] F25 baked blockLen 承重性 lit + F7 shape 死参删除**：F25（`selectIntegerCoreLMUL`，3 前门）baked `blockLen=32` 是承重输入（翻 coreLMUL），当前靠三前门 `dimSize==32` gate（`RVVReductionSourceFrontDoor.cpp:136`/`RVVDequantDotSourceFrontDoor.cpp:144`）恒真守住——**需要**：挂一条 lit 断言 coreLMUL 随 blockLen∈{16,32,64} 翻转（记录承重性+gate 守护），补输入仅在收纳非-32 几何时执行。F7（`tilingVariantFeasibleSet`）的 `(void)shape;` 是死参——**需要**：一条 lit 证 shape 变而 feasible 集不变 → 删 shape 参（验收=删后 fixture byte-exact 不变）。
