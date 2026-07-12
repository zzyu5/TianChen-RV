# 模板五大件 → 目录映射（外来者 30 分钟定位图）+ 命名统一基准

> **用途**：TianChen-RV 的头条 = 「可复制 / 可扩展的能力驱动执行层软件栈**参考模板**」。
> 一个外来贡献者读完 README + 本图，应能在 **30 分钟内** 定位模板的每一大件、并知道
> 接入一个新家族要动哪些格位。本图补 [TEMPLATE-AUDIT] 认定的**头号定位债**（R1）。
> **本 doc 只描述结构与命名约定，不改任何 code / schema / 选择逻辑 / 测量数据。**
>
> 生成基准 = TEMPLATE-AUDIT 结构审计（`docs/reports/2026-07-11-TEMPLATE-AUDIT-structure.md`，
> 快照 `b3e3fef4`）；本图把审计的评分表落成**可导航的目录映射 + 代表文件清单**，并 codify
> T3p（`658e5c0f`）暴露的 **P 编号命名碰撞**，为后续 [RENAME] 提供命名统一基准。

---

## 0. 术语消歧（先读 —— 两个「五」不是同一个）

- **模板五大件（本图主体，仓库级）**：模板在**仓库目录**上的六个结构组件 ——
  ①schema · ②插件五件套 · ③前门 · ④选择器 · ⑤falsifier 组 · ⑥测量库。习惯口语称「五大件」，
  实为 **6 件**（早期未把「测量库」单列）。**本图统一按 6 件编号 ①–⑥。**
- **插件五件套（单家族级）**：②这一件**内部**的五个协议实现文件（见 §1.②）。
  一个家族接入 = 落齐这五件。**「五件套」= 一个家族的五个文件；「五大件」= 整个仓库的六个组件。**
  参考范本 = `Template/` 家族（干净、无历史包袱，五件齐全）。
  > ★**「五件套」有三个轴，别混（[GAP-P4-FIVEPIECE-COLLISION]）**：本 §1.②/§2 的
  > 五文件是**文件轴** = 交付物落在哪些文件；权威的 **[P-2] = 交付轴**（① facts+relations
  > ② legality ③ emission pattern ④ tests ⑤ ledger，"缺一不收"的验收定义，住
  > [`.trellis/spec/plugin-protocol/extension-plugin-integration.md`](../../.trellis/spec/plugin-protocol/extension-plugin-integration.md) §[P-2]）。
  > **本图的文件五件是交付轴 [P-2] 的 realization 视图，不是第二份 [P-2] 定义**（历史上本节曾标
  > 「[P-2]」致碰撞，现撤该标注）。二者不一一对应（交付③ emission 跨 2 文件；交付⑤ ledger 是
  > `docs/` 行、非 `BackendEmissionDriver`）。三轴互映表见 spec §[P-2] 的 cross-map。

---

## 1. 五大件（六组件）→ 目录映射表

| # | 大件 | 主目录 | 代表文件（30 秒可辨） | 定位度 |
|---|---|---|---|---|
| **①** | **schema**（能力 / coverage / roster / pattern / cert / retire） | `schema/`（顶层单目录，清洁独立） | `capability.schema.v1.json` · `coverage-sixstate.v1.json`（84 certified 源）· `coverage-roster.v1.json`（93 格分母）· `pattern-registry.v1.json` · `cert-lineage.v1.json` · `retired-index.generated.json`（机生） | **HIGH** |
| **②** | **插件五件套**（每家族·**真实 6 目录根**，见 §2） | `lib/Plugin/<家族>/` + `include/TianChenRV/Plugin/<家族>/` + `lib/Dialect/<家族>/IR/`（+ include 镜像）+ `lib/Target/<家族>/`（+ include 镜像） | 参考范本 `Template/`：`TemplateExtensionPlugin` · `TemplateVariantLegality` · `TemplateConstructionProtocol` · `TemplateEmitCRouteProvider` · `TemplateBackendEmissionDriver`（文件五件，= 交付 [P-2] 的 realization；真触碰集 6 根/20 文件，见 §2 [GAP-P4-TOUCHSET]） | **MEDIUM**（RVV 家族 R2/R3 已分子目录；触碰集 6 根非 5 文件） |
| **③** | **前门**（front-door 构造：抽象 contraction → in-compiler 构造 typed 区域） | 散在 `lib/Plugin/RVV/*SourceFrontDoor.cpp` + `lib/Plugin/RVV/RVVLowerQuantContraction.cpp` + `lib/Plugin/Construction/` + `lib/Dialect/RVV/IR/RVV*Construction.cpp` | `RVVMonolithicBlockDotSourceFrontDoor.cpp` · `RVVDequantDotSourceFrontDoor.cpp` · `RVVCodebookDotSourceFrontDoor.cpp` · `RVVLowerQuantContraction.cpp` · `Construction/ConstructionProtocol.cpp` | **LOW-MEDIUM**（无专属 `FrontDoor/` 目录，靠命名约定；见 R3） |
| **④** | **选择器**（能力键控 cost-model / 先验） | 二层散：`lib/Transforms/VariantSelection.cpp` + 各插件 `estimateVariantCost` + `include/.../Plugin/RVV/RVVRepackTilingSelection.h` + `schema/tiling-measurements.v1.json`（先验/测量契约） | `lib/Transforms/VariantSelection.cpp`（通用能力键控 pass）· `include/TianChenRV/Plugin/ExtensionPlugin.h`（`estimateVariantCost` 接口声明） | **MEDIUM**（无单一 `selector/` 目录；四处二层，需 R5 co-locate 文档） |
| **⑤** | **falsifier 组**（机检验收 [F-1..F-6]） | `tools/lint/` + `tools/fuzz/` + `.trellis/scripts/` + `.github/workflows/falsifier-gate.yml` + `test/` 语料 | **见 [FALSIFIER-INDEX.md](./FALSIFIER-INDEX.md)（R4：F-1..F-6 → 文件逐条映射）** | **MEDIUM-HIGH** |
| **⑥** | **测量库**（板上 A/B harness + 双账本） | `tools/e2e-harness/`（清洁独立家）+ `schema/tiling-measurements.v1.json`（double_ledger 数据契约） | `tools/e2e-harness/run_e2e.sh` · `board/*driver.c` · `aggregate_e2e.py` · `board_ab.sh` | **HIGH** |

**证据落点（与五大件正交）**：`experiments/{active,sealed,archive}/**` = 逐 cell MANIFEST + INDEX
（机检 = `tools/lint/{gen_experiments_index,check_index_consistency}.py`）；`docs/` = ROADMAP + canon
三总纲 + method（本 doc）+ reports（证据卷宗）。

---

## 2. 插件五件套（文件轴 · = 交付轴 [P-2] 的 realization · 一个家族落齐这五件）

参考范本 = `lib/Plugin/Template/`（+ `include/TianChenRV/Plugin/Template/` 头）：

| 件（文件轴） | Template 范本文件 | 承载的交付 [P-2] | 职责 |
|---|---|---|---|
| 1. 能力 / 插件入口 | `TemplateExtensionPlugin.cpp/.h` | ① facts+relations（`getCapabilities()` 实例）+ ④/⑤ hook | 声明家族能力事实 + `estimateVariantCost` override + `register…ExtensionPlugin` |
| 2. 合法性 | `TemplateVariantLegality.cpp` | ② legality 谓词 | 变体合法性（能力谓词 → 可行变体集） |
| 3. 构造协议 | `TemplateConstructionProtocol.cpp/.h` | ③ emission pattern（一半） | 抽象 op → in-compiler 构造 typed 区域（前门③的家族侧实现） |
| 4. 发射路由 | `TemplateEmitCRouteProvider.cpp/.h` | ③ emission pattern（一半） | 选中变体 → EmitC 发射路由 |
| 5. 后端发射驱动 | `TemplateBackendEmissionDriver.cpp/.h` | ③ backend 侧 | 具体 backend body 发射 |

> **交付 [P-2] 五件（权威验收轴）= ① facts+relations · ② legality · ③ emission pattern
> · ④ tests · ⑤ ledger 行**（住 spec §[P-2]）。上表文件不与交付一一对应（③跨 2-3 文件，
> ④ tests 落 `test/**`、⑤ ledger 落 `docs/method/C2_marginal_cost_ledger.md`——都不在这 5 文件内）。

### ★真实触碰集 ≠ 「5 文件」（[GAP-P4-TOUCHSET]，2026-07-12 [GOV-8] 核）

「复制 5 个 `lib/Plugin/<Fam>/` 文件」只是**名义**起点。参考范本 `Template/` 家族的**真实
可链接触碰集 = 6 个目录根 / 20 个源文件 / 6 个 `CMakeLists` + 1 个共享注册文件**——因为
插件库经 CMake **传递依赖**一个家族 `Dialect` 库和一个家族 `Target` 库（`lib/Plugin/Template/CMakeLists.txt`：
`TianChenRVTemplatePlugin LINK_LIBS … TianChenRVTemplateDialect TianChenRVTemplateTarget`）。只复制
5 个 plugin 文件跑 `cmake --build` **会在链接期撞** 缺失的 `TianChenRV<Fam>Dialect` /
`TianChenRV<Fam>Target` target：

```
lib/Plugin/<Fam>/                      5 .cpp + CMakeLists  → 4 库（Plugin/ConstructionProtocol/EmitCRouteProvider/BackendEmitter）
include/TianChenRV/Plugin/<Fam>/       4 头
lib/Dialect/<Fam>/ + IR/               <Fam>Dialect.cpp + 2 CMakeLists → 库 TianChenRV<Fam>Dialect   ← 五件套未列·MANDATORY
include/TianChenRV/Dialect/<Fam>/ + IR/  <Fam>Dialect.h + <Fam>Ops.td + 2 CMakeLists（ODS/TableGen）
lib/Target/<Fam>/                      <Fam>TargetSupportBundle.cpp + CMakeLists → 库 TianChenRV<Fam>Target  ← 五件套未列·MANDATORY
include/TianChenRV/Target/<Fam>/       <Fam>TargetSupportBundle.h
lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp   +1 共享注册（[GAP-P4-REGISTER]）              ← 五件套+[F-3] 均未列
```

**接入协议**：新家族 = 复制 `Template/` **6 根**（不止 `lib/Plugin/<Fam>/` 五件）+ 填能力事实
+ 注册表行 + 注册进 `BuiltinExtensionPlugins.cpp`（[GAP-P4-REGISTER]）+ 文档。[F-3] 触碰集 =
`lib/{Dialect,Plugin,Target}/<Fam>/` + `include/` 镜像 + 表行 + docs + 那**一行**共享注册表行
（**目录归拢是 F-3 可判定的前置**，见 R2/R3；`schema/family-manifest.v1.json` 已把这 6 根形状声明为
一家族 owned territory，故宽触碰集**非** [F-3] 违规——唯一真共享点 = 注册文件，和解见 spec §[GAP-P4-REGISTER]）。
其它家族对照：`IME/`（~3 文件，能力/构造内联进 ExtensionPlugin，复用共享 EmitC/RVV lowering）·
`Scalar/`（3，空桩待接 [X-SCALAR]）· `RVV/`（R2/R3 已按件分 `BodyRealization/FrontDoor/EmitC/Schedule/Selection/Construction` 子目录）·
`Offload/` `Toy/` `TensorExtLite/`（参考/兜底）。

---

## 3. ★命名碰撞 codify（[RENAME] 命名统一基准 · T3p `658e5c0f` 暴露）

Canon 自身携带**不一致的 P 编号**；T3p 记录碰撞而非静默选一个。**[RENAME] 会话应据此统一，
本节是唯一权威碰撞清单。**

### 3.1 「P1」有三个所指（THREE distinct referents）

| 记号 | 所指 | 出处 | 语义 |
|---|---|---|---|
| **P1-(a)** | **宽 LMUL 分组** | `docs/canon/TianChen-RV_科研目标总纲v2.md:121` | 一个 schedule/emit 优化模式（LMUL 宽度旋钮） |
| **P1-(b)** | **N-operand 构造统一**（4 前门已机制化） | `docs/canon/TianChen-RV_执行总纲v2.md:91`「P1 已机制化」 | 构造协议主张（本 = T3p 的 P1；构造轴 validated-REAL / perf-novelty RETIRED） |
| **P1-(c)** | **[GAP-P1]** = 宽 VLEN 喂饱 schedule 的 roofline gap | T8 `q4_0-gevm-k1-vlen256` 行 | 一个具名性能 GAP（misselection：selector 只键 isRVV0p7、不键 VLEN128-vs-256） |

**统一建议（供 RENAME 裁）**：三者互不相关，**不应共享「P1」字面**。建议保留 [GAP-P1] 为 GAP-域记号
（§3.3 前缀约定内自洽），把 (a) 宽 LMUL 分组、(b) N-operand 构造 各自改为**模式注册表 pattern_id**
（如 `SCHED-WIDE-LMUL` / `CONSTRUCT-N-OPERAND-ROUTE`），退役裸「P1」标签。

**✅ [RENAME] 部分执行（2026-07-12 · 本节唯一权威碰撞记录 · 以本节为准）**：
- **P1-(b) N-operand 构造**：仓库内仅有的 **in-domain（代码）** 裸「P1」footprint 已消歧（behavior-preserving·仅注释）——
  `lib/Plugin/RVV/Construction/RVVContractionRouteIdentity.cpp`（3 处：`P1 generic ABI-order`×2 + `NOT consumed by the P1`）
  + `include/TianChenRV/Plugin/RVV/RVVMonolithicBlockDotFamily.h`（1 处：`P1 N-operand descriptor refactor`）
  一律改为自描述的 **「N-operand generic ABI-order」/「N-operand descriptor refactor」**，裸「P1」token 自代码退役。
  其**建议 pattern_id = `CONSTRUCT-N-OPERAND-ROUTE`**，pending `schema/pattern-registry.v1.json` 注册（见下 ⏸）。
- **P1-(c) = [GAP-P1]**：**保留**（GAP-域自洽命名，§3.3 已裁不与 (a)/(b) 合并），无动作。
- **⏸ DEFERRED（canon 级 + schema 域·本会话触碰域外·主会话/用户裁）**：
  (1) 把 pattern-library 槽位 **P1-(a) 宽 LMUL 分组 → `SCHED-WIDE-LMUL`** 落到 canon 三总纲
  （`科研目标总纲v2.md:121` 定义行 + `:168`[C3-3]/`:170`[C3-5] 引用 + `执行总纲v2.md:91`）——牵动 **C3′ headline 模式库 [C3-1] 措辞**，属 canon 级；
  (2) 把 `CONSTRUCT-N-OPERAND-ROUTE` / `SCHED-WIDE-LMUL` **正式注册进 `schema/pattern-registry.v1.json`**（schema 域，本会话禁碰）。
  二者未落地前，代码侧裸「P1」已清，canon 的「P1」按本节 = (a)/(b) 双所指、**以本节消歧为准**。历史 dated 报告内旧「P1」引用属 append-only 存档·不回改。

### 3.2 「P4」= PAT-2 vs PAT-S6（形式槽空 vs 机制已实现）

- **P4 = 布局 / repack** 模式。`执行总纲v2.md:91` 标注**形式化的 `PAT-2` "P4" 注册对象 = 未启动**；
- 然而 repack **机制**已在 `schema/pattern-registry.v1.json` 的**另一个注册条目 `PAT-S6`** 下重度实现
  （`PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1`，status=`mechanized`，7/7 XFER-1 三类边界）。
- **碰撞**：canon 谈 P4 时可能指「未启动的 PAT-2 形式槽」或「已机制化的 PAT-S6 实现」——二者被同一个「P4」遮蔽。

**统一建议**：`pattern-registry.v1.json` 现有实际 pattern_id 命名族 = `MFLAT-1..5` / `MFLAT-P2c` /
`WIDE-DECODE-*` / `PAT-S6-*`。建议 [RENAME] **以 registry 的 pattern_id 为唯一真源**，退役 canon 散落的
「P4」「PAT-2」裸标签：把「未启动的 P4 形式槽」显式记为一个 pending pattern_id，把已实现的指向 `PAT-S6-*`。

**✅ [RENAME] 裁定（2026-07-12 · 本节唯一权威碰撞记录 · 以本节为准）**：
- **无 in-domain 代码 footprint 可安全消歧**：代码里的 **`SP4`**（`RVVLowerQuantContraction.cpp` / `RVVRepackTilingSelection.h` /
  `执行总纲v2.md:246`）= output-tiling 选择变体（register-cliff），**已= 机制化的 `PAT-S6`**、命名一致，**非** pattern-library「P4 布局/repack」槽——**不改**（`grep P4\b` 命中 `SP4` 属正则边界假阳性，勿误伤）。
  `tools/e2e-harness/g3-lode-flat-q41/MANIFEST.md` 的 `[PAT-S6]` 引用亦正确、不改。
- **⏸ DEFERRED 全项（canon 级 + schema 域·本会话触碰域外·主会话/用户裁）**：pattern-library「P4」/「PAT-2」裸标签仅存于 canon 三总纲
  （`科研目标总纲v2.md:121`定义 + `:168`/`:213`; `执行总纲v2.md:91`/`:282`/`:289`），已机制化实现的真源 = `PAT-S6-*`（in `schema/pattern-registry.v1.json`）。
  落地建议仍如上（registry pattern_id 为唯一真源、pending 槽显式记 pending id），**须编辑 canon + schema**，本会话均不碰。**以本节消歧为准**。

### 3.3 [GAP-*] 前缀约定（现状 roster + 建议规范）

现存 [GAP-*] 记号（`docs/` + `schema/` + `experiments/` 全域，去重）：

```
[GAP-1]                       通用 GAP-1 闭环规程记号（name→close→retest→cite）
[GAP-P1]                      宽 VLEN 喂饱 roofline（misselection：selector 不键 VLEN）
[GAP-SB]                      super-block per-subblock-gather 未 pair-batch
[GAP-RP]                      register-pressure whole-reg spill
[GAP-NUM]                     数值双档（relaxed 仅 q8_0 materialized）
[GAP-GRID]                    grid/codebook 解码
[GAP-FLAT-E2E]               FLAT kernel 赢未过 micro∧e2e
[GAP-FLAT-E2E-ROUTING]       FLAT/K-quant VLEN128 未路由进 forward
[GAP-IME-LEAF-PIPELINE]      IME 发射器 per-fragment leaf 未流水（4× loss）
[GAP-IME-E2E-INTEGRATION]    tcrv IME 未 wire 进 llama forward
[GAP-KQUANT-E2E-INTEGRATION] q4_K vl16 未部署 + 无 board 模型
[GAP-DEQ-KQUANT-UNPACK]      dequant super-block scale-unpack compute-bound
[GAP-FWD-M8-VSETVL]          forward 显式 m8 map 的 per-iter vsetvl
[GAP-CLANG-GATHER-TRAP]      clang vluxei gather 标量化陷阱
[GAP-XXX]                     模板占位（非真 GAP）
```

**约定（现状已自洽，建议 [RENAME] 只做归档、勿动语义）**：
- `[GAP-<AREA>-<SPECIFIC>]` = 具名性能/接线 GAP，绑 objdump/board 证据 + 三级 triage
  ∈ {missing_fact | missing_pattern | misselection | physical}，住 T8 台账。
- `[GAP-1]` 是**规程**记号（闭环四步），与具名 GAP 不同层，勿混。
- `[GAP-P1]` 虽用 P1 字面，属 GAP-域自洽命名，**保留**（§3.1 已裁不与 P1-(a)/(b) 合并）。
- `[GAP-XXX]` 是模板占位，非真 GAP，[RENAME] 时可核实模板引用后清理。

---

## 4. 外来者 30 分钟定位路径（推荐阅读序）

1. `README.md` 「**Extending the stack: add a family**」段（外部入口，直链本图 + 协议 spec + `Template/` 范本；[GAP-P4-DISCOVERY] 已补）→ 再看「Repository layout」泛目录。
2. **本图 §1** → 五大件（六组件）各指到具体目录 + 代表文件。
3. 想**接一个新家族** → §2 插件五件套（照 `Template/` **6 根**复制，非仅 5 文件；[GAP-P4-TOUCHSET]）+ 注册步骤（[GAP-P4-REGISTER]）+ [F-3] 触碰集纪律；写 up 用 [`P4-family-integration-doc-TEMPLATE.md`](./P4-family-integration-doc-TEMPLATE.md)。
4. 想懂**机检验收** → [FALSIFIER-INDEX.md](./FALSIFIER-INDEX.md)（F-1..F-6 → 脚本/lit/gtest）。
5. 想懂**证据在哪** → `experiments/INDEX.md`（机生 registry）+ 逐 cell MANIFEST。
6. 想懂**命名** → §3 P 碰撞 codify（避开三个 P1 / P4 双所指的坑）。

---

## 4.5 ★纪律：排期/里程碑判断前先跑「代码事实核查」（2026-07-11 用户裁植入）

> **规则**：任何**排期 / 里程碑 / 立项等级 / 完成度**判断落笔前，**先 `grep` 当前树的 landed 现状**
> （能力事实注册？owned 内核 commit？测试机检在位？—— 用本图 §1 的目录锚 + `git log --oneline -- <path>`），
> **再**据现状写排期。**排期报告本身会 stale**——文档-代码不同步债让「计划从零做 X」的预设常被代码事实证否。

**本轮三处 stale 教训（X-SCALAR 收口，均排期报告 vs 代码事实不符）**：
1. **XS-M1 zvfh**：排期预设「zvfh 事实未注册、需从零建」→ 代码事实：`rvv.zvfh` 早已注册（`95f1a482`，implies 链 + token-guard），[S-2] 闭包夹具已连（`CapabilityModelTest.cpp:697-736`）。XS-M0（`a820e87a`）据此**仅文档正名、未重复注册**（[NG-4]）。
2. **XS-M3 判据④**：排期预设「向量缺席→标量变体 only_feasible 真实选中」需从零 wire → 需先核 `VariantSelection` + Scalar 合法性谓词的 landed 现状再定剩余工作量。
3. **整体 70-80% landed**：任务预设「类比 IME M0-M3 从零接入五件套」→ scout 现状证否：两 owned 内核 + 曳光弹 + F-6 双断言机检**均已 landed**，X-SCALAR 实为「收口 + 正名」而非「从零建家族」。

**教训**：`grep landed 现状` 是 5 分钟的事，能把「从零大战役」reframe 成「有界收口」，避免按 stale 排期重复已完成工作。定位现实的权威源 = **代码树 + `experiments/INDEX.md` 机生 registry**，不是任何日期化排期报告。

---

## 5. 已知定位债（→ [RENAME] R1–R7）

| 债 | 现状 | 收法 | 触碰域 |
|---|---|---|---|
| 头号 = 缺五大件→目录映射 | **本 doc 已补（R1）** | done | docs（本 doc） |
| ⑤ falsifier 无 F→文件索引 | **[FALSIFIER-INDEX.md] 已补（R4）** | done | docs |
| ③ 前门无专属目录、靠命名 | **✅ R3 landed（2026-07-12·cbe21c3e）**：10 前门归 `lib/Plugin/RVV/FrontDoor/` | **done**（build+lit 验绿·behavior-preserving） | code（已执行·主会话） |
| ② RVV 插件顶层文件爆炸 | **✅ R2 landed（2026-07-12·cbe21c3e）**：33 顶层 `.cpp`→4 桶（BodyRealization13/FrontDoor10/Schedule7/Selection1）+2 root·零 #include 破坏·单 target 子路径 | **done**（计划 `docs/reports/2026-07-12-卫生档B-lib重组-move计划.md`·一次执行·build [82/82]+lit 904/907） | code（已执行·主会话） |
| ④ 选择器四处二层散 | 无 `selector/` 目录 | R5：文档 co-locate（不必物理搬） | docs |
| P 编号碰撞 | 三 P1 / P4 双所指 | **§3 已 codify；[RENAME] 部分落地（2026-07-12）**：P1-(b) 代码裸标已退役（4 处注释·behavior-preserving）；P1-(a) 宽LMUL→`SCHED-WIDE-LMUL` / P4·PAT-2→registry pattern_id = **DEFERRED**（canon 三总纲 + schema·主会话/用户裁） | docs（本 doc）+ lib 注释 |

> **✅ R2/R3 已 landed（2026-07-12·cbe21c3e·Plugin/RVV 内部 4 桶）**。剩余 [F-3] 完全可判定欠「跨 4 lib 根（Dialect/Conversion/Plugin/Target）收拢 vs 维持 MLIR 分层」的上层裁定（必要不充分·另议）+ 移 `include/` 42 头（HIGH-risk·另立批）。R5/R6/R7 见上表。
