# [P-4] Tier-2 clean-room 接入演练 — T1c 记录报告

> **性质**：[C1-4]「(M4) 外部接入实录」的 **Tier-2 正面兑现**（用户 2026-07-12 裁三.2）。
> Tier-1（`2026-07-12-X3-冷启动接入-drill-外部接入实录.md` + 重跑）只判 **docs-followability**（禁真 build）；
> **Tier-2 = 非协议作者 clean-room agent 真写代码 + 真 build 到绿 + falsifier 六门全过 + 触碰集符 [F-3]**。
> 本报告 = 一次真实的最小家族接入（`weft_demo`）逐件记录 + build/门证 + 偏离清单。
>
> **登记名义如实 = "clean-room agent 演练"**（非真第三方；先天偏置见 §偏离清单）。

---

## 0. 演练设置

- **接入者身份**：Tier-2 clean-room 适配 agent（非协议作者）。**盲纪律**：从 `README.md`「Extending the
  stack」段起，只读协议 docs（`.trellis/spec/plugin-protocol/*`、`.trellis/spec/capability-model/capability-contract.md`、
  `docs/method/{REPOSITORY-MAP-五大件,P4-family-integration-doc-TEMPLATE,FALSIFIER-INDEX}.md`）+ `lib/Plugin/Template/`
  参考范本。**禁读** RVV/IME/Scalar 插件实现源码。→ **实际 forbidden-source peek 数 = 0**（全程只读 Template 范本 + 公共头 + spec）。
- **接入目标**：最小 toy 家族 `weft_demo` = `Demo` 方言（2 op）+ `DemoExtensionPlugin`（capability profile
  `demo.extension` + legality + emission-pattern-reuse + tests + ledger row + 注册）。**capability-gate 家族**
  （[S-8]：owned body 用基础机制，家族事实充当门）。
- **build**：全新干净 build dir `build-demo/`（fresh clean，非 incremental；见项目 memory「incremental 不可靠」）。
- **禁**：git（主会话 commit）、board（host-compile 层）。

---

## 1. 交付五件 [P-2] 逐件 followability（docs 自足 / 需 peek / gap）

| # | 件 | 落地 | 判读 | 备注 |
|---|---|---|---|---|
| ① | facts + relation rows | `lib/Plugin/Demo/DemoExtensionPlugin.cpp` `getCapabilities()`（C++ `PluginCapability(demo.extension, future-extension-demo, …)`） | **docs 自足** | capability-contract 顶部 [GAP-P4-SCHEMA-DIVERGENCE] banner + landing-points 表精确：事实实例落 C++ 非 schema JSON；结构键 `id/kind/implies/conflicts/params` 被消费、`provenance/trust/subclass` aspirational。零歧义。 |
| ② | legality predicate | `lib/Plugin/Demo/DemoVariantLegality.cpp`（`verifyDemoSelectedVariantLegality`，构造协议共享权威） | **docs 自足** | interfaces-and-registry §Legality + 五件套②说清「plugin-owned、先于 route provider」；Template 范本给精确形状。 |
| ③ | emission pattern | route provider `DemoEmitCRouteProvider.cpp` + 构造协议 `DemoConstructionProtocol.cpp` + target bundle `DemoTargetSupportBundle.cpp`（**reuse 共享 EmitC materializer + ConstructionTemplate artifact adapter**）；**未注册独立 backend emission driver** | **docs 自足 · 但暴露 GAP**（见 §4 GAP-A） | [P-2]③ 明许「reuse existing pattern」声明。**发现**：要注册家族自有 EmitC backend emitter 须编辑**第二共享注册点** `lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp`——**未在 [GAP-P4-REGISTER] recipe、亦未在 `family-manifest.v1.json` `shared_allowances`**。为保 [F-3] 触碰集清洁，本演练**声明 ③ = reuse-existing、不注册独立 backend emitter**。 |
| ④ | tests | `test/Dialect/Demo/compute-skeleton.mlir`（1 正 + 5 负 verifier 用例·lit + FileCheck） | **docs 自足** | 五件套④ + FALSIFIER-INDEX + manifest `test_ranges` 指到 `test/**`；Template 范本给 copy-me 模型。lit byte-exact 达成；per-board objdump golden = 硅封层（I8·超编译-tier 演练范围，docs 明标编译≠硅封）。 |
| ⑤ | ledger row | `docs/method/C2_marginal_cost_ledger.md`（Demo drill row：core-modified=2 · family-local 2787 LOC clone-adapt · **明标非 C2 曲线点**） | **docs 自足** | landing-points 表 + 五件套⑤指到该文件。诚实判读（[NG-8]）：clone 家族 LOC = 拷贝值非设计值，不并入 IME 2484/X-SCALAR 1501 曲线。 |

**共享注册步（[GAP-P4-REGISTER]）**：`lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` +`#include` +1 `kBuiltinExtensionBundles[]`
表行 + 家族头声明 `registerDemoExtensionPlugin`。→ **docs 自足**（extension-plugin-integration §Registration 给精确 2-line recipe，
一字不差可跟随）。**唯一 core 触点**，blessed shared_allowance（family-name-as-data·for-迭代无分支）。

**触碰集（[GAP-P4-TOUCHSET]）**：REPOSITORY-MAP §2 + extension-family-plugin-template 精确预警「6 根 / 20 文件 / 6 CMakeLists +
1 共享注册」及**传递链接依赖 `Weft<Fam>Dialect`/`Weft<Fam>Target` MANDATORY**——**实测完全命中**（只复制 5 plugin 文件会链接期撞）。
→ **docs 自足**。

---

## 2. build 绿证

- **fresh clean build dir** = `build-demo/`（`cmake -S . -B build-demo -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/… -DMLIR_DIR=…`）。
- `cmake --build build-demo --target weft-opt -j64` → **258/258 · 0 error**，链接 `bin/weft-opt`（66 MB）。
- `weft-translate` + gtests `weft-plugin-registry-test` / `weft-plugin-variant-proposal-test` → **0 error**。
- **Demo 方言/插件自动进 weft-opt·零 core 编辑**：`weft-opt.cpp` 调 `registerBuiltinExtensionBundlePlugins` → `kBuiltinExtensionBundles`
  含 `demo-extension-bundle` → 插件 `registerDialects()` 插入 `weft_demo` 方言。**无需碰 `InitWeftDialects.cpp` / `weft-opt.cpp`**。
- **功能证**：`weft-opt test/Dialect/Demo/compute-skeleton.mlir --split-input-file --verify-diagnostics` → EXIT 0；
  正例 FileCheck PASS + 5 负例 diagnostics 全匹配（verifier 真跑）。
- **无既有测试破坏**：`weft-plugin-registry-test` / `weft-plugin-variant-proposal-test` 均 PASS（新增家族未破坏任何 plugin-count 断言——各 gtest 自建局部 registry）。

## 3. falsifier 六门证（real binary·`WEFT_BUILD=build-demo`）

| 门 | self-test | default（真 binary） | 判 |
|---|---|---|---|
| **[F-1]** zero-branch | GREEN | GREEN — 零 family-keyed 分支 across 69 core 文件（I3 holds；"demo" 不进 family-regex keys·未进 core scope） | **GREEN** |
| **[F-2′]** schema-def | GREEN（16/16） | `report --check` OK — `schema/capability.schema.v1.json` shape-hash 匹配 VERSIONLOG（**未触 schema.def**） | **GREEN** |
| **[F-3]** family-locality | GREEN | GREEN — 8 家族·46 source_ranges（ratchet 40→46）·Demo 6 ranges→12 文件·零 orphan/pollution/anti-widen | **GREEN** |
| **[F-4]** attribution-jsonl | GREEN | GREEN — 5/5 kernel attributed（Demo 不扰选择归因） | **GREEN** |
| **[F-5]** failclosed-fuzz | GREEN | GREEN — 20/20 fail-closed（Demo 不扰 RVV lowering） | **GREEN** |
| **[F-6]** independence | GREEN | GREEN — vector-absent 实例真选 scalar family（only_feasible）·闭包∩rvv.*=∅ | **GREEN** |

**相邻门（同 `falsifier-gate.yml` / dir-lint·含 Demo 的树全 GREEN）**：opponent-facts-pin · monolith-retire ·
frontdoor-provenance · cert-requirements · retired-index · perf-covered-recon（reconciliation_ok=True）· docs-canon —— **全 GREEN**。

## 4. 触碰集 [F-3] 符合性

**diff ⊆ Demo 家族 territory + shared_allowances**：
- 家族本地源（6 根 12 源文件 + 6 CMakeLists）：`lib/{Dialect,Plugin,Target}/Demo/` + `include/Weft/{Dialect,Plugin,Target}/Demo/`。
- shared_allowances：schema（`family-manifest.v1.json` Demo entry + ratchet）· docs（本 T1c + C2 ledger row）·
  tests（`test/Dialect/Demo/`）· build_and_tooling（5 父 CMakeLists `add_subdirectory(Demo)` / `WeftDemoPlugin` LINK）·
  **plugin_registration（唯一 core 触点 = `BuiltinExtensionPlugins.cpp` 表行·blessed）**。
- **无未授权 core dispatch/selection/lowering 编辑**。[F-3] default gate 机检 GREEN。
- **诚实注记**：CI base..head diff 模式需 git refs（禁 git），故 diff-containment 由**人工推证**（触碰集 ⊆ territory + allowances）
  + default 全树一致性机检 GREEN + self-test 双评估 GREEN，**非**diff 模式机跑。

---

## 5. 偏离清单（clean-room 先天偏置 + 演练裁剪·如实）

1. **同栖仓库 · in-house agent · 非真第三方**（Tier-3 才是真外部）。先天偏置：规则约束下的盲、非真跨组织落地。
2. **Demo = Template 参考家族的 clone-adapt**（protected-rename 复制 + 手写 CMake/manifest/test/ledger/doc + ③决策）。
   **真净新设计 ≈ 0**——本演练证的是「docs + Template 范本 → build-绿 + 六门」的**可跟随性**，家族本体是范本近复制。
   （故 ⑤ ledger 明标非 C2 曲线点。）
3. **peek 了 Template 范本源**（协议明许 copy-me）·**未 peek** RVV/IME/Scalar 插件内部（盲纪律守住·forbidden-peek=0）。
4. **[F-3] 只跑 default（全树）+ self-test**；diff 模式（base..head）禁 git 未跑，改人工推证触碰集。
5. **[F-2′] 只跑 self-test + `report --check`**（hash `capability.schema.v1.json`·未触）；`gate --base --head` onboarding-diff
   模式未跑——该模式会因编辑 `schema/family-manifest.v1.json` 命中 schema/ 前缀、要求 PR 带 `Family-Onboarding:` trailer +
   additive 分类（见 §6 GAP-C）。
6. **裁剪：不注册独立 backend emission driver**（③ 声明 reuse-existing）——为保 [F-3] 触碰集清洁而做的 scope 选择，
   根因 = §6 GAP-A（第二共享注册点未被 allowances 覆盖），**非** docs 不足。

---

## 6. ★具名 gap（Tier-2 落地真缺口 · Tier-1 docs-followability 未捕）

- **GAP-A（第二共享注册点未被覆盖·最重要）**：一个**自带 EmitC backend** 的家族除 `BuiltinExtensionPlugins.cpp`（插件 bundle）
  外，还须注册进 `lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp`（`kBuiltinBackendEmitters[]` 表 + 其 CMakeLists）。
  该表 [F-1] 已 bless 为「零分支注册表」，但 **[GAP-P4-REGISTER] recipe 只记了 BuiltinExtensionPlugins.cpp 一处**，且
  `family-manifest.v1.json` `shared_allowances.plugin_registration` **只 key 了 BuiltinExtensionPlugins.cpp**。
  → **一个需要自有 backend emitter 的家族无法在现 allowances 下保持 [F-3] 触碰集清洁**（CI diff 模式会把该编辑判 CORE-EDIT）。
  **修法建议**：把 `kBuiltinBackendEmitters[]` 行加入 `shared_allowances.plugin_registration`（镜像既有 blessing），
  并在 [GAP-P4-REGISTER] recipe 补第二注册点。（canon/schema 域·本演练不自改·主会话/用户裁。）
- **GAP-B（父 CMakeLists add_subdirectory 未入触碰集枚举·次要）**：接一个新家族须编 4 个父 CMakeLists
  （`lib/Plugin` / `lib/Dialect` / `lib/Target` / `include/Weft/Dialect` 各 +`add_subdirectory(Demo)`）+ Builtin LINK。
  [GAP-P4-TOUCHSET] 的「6 CMakeLists」是**家族本地** 6 个，**未枚举**这 4 父 wiring 编辑（虽被 build_and_tooling allowance 覆盖）。
  外来者易漏。**修法建议**：touch-set 文档补一句「+4 父 add_subdirectory + 1 Builtin LINK（build_and_tooling 允许）」。
- **GAP-C（[F-2′]/[F-3] 对 schema/ 作用域重叠·次要）**：[F-3] 视 schema/ 表行为免费 shared_allowance，但 [F-2′] operation-gate
  gate 整个 `schema/` 前缀——**[F-3] 要求的 family-manifest 编辑恰是 [F-2′] 会 flag 的**，除非 PR 带 `Family-Onboarding:` trailer +
  additive 分类。二者 by-design 自洽（trailer 是和解），但 trailer 约定只写在 `check_schema_gate.py`、**未写进 plugin-protocol spec**，
  首次接入者需自行发现。**修法建议**：extension-plugin-integration 的 Registration/五件套段补一句 onboarding-trailer 约定指针。

---

## 7. Verdict

- **build 绿 = TRUE**（fresh `build-demo/`·weft-opt+weft-translate+2 gtest·0 error）
- **falsifier 六门 = TRUE**（[F-1..F-6] 全 GREEN·real binary·相邻门全 GREEN）
- **触碰集符 [F-3] = TRUE**（default gate GREEN·diff ⊆ 家族 territory + allowances·唯一 core 触点 = blessed 注册表行）
- **五件套 followability**：①自足 ②自足 ③自足(+GAP-A) ④自足 ⑤自足；注册步自足(+GAP-A)；触碰集自足(+GAP-B)
- **偏离清单 = 6 条**；**具名 gap = 3（A 重要 / B,C 次要）**
- **★[C1-4] 正面兑现判读 = 达标**：非协议作者 clean-room agent（forbidden-peek=0）循 README→协议 docs→Template 范本，
  真写最小家族 `weft_demo` 并 **build 到绿 + falsifier 六门全过 + 触碰集符 [F-3]**——Tier-2 硬门三项全 TRUE。
  **具名残缺**：GAP-A（第二共享注册点未被 [GAP-P4-REGISTER]/allowances 覆盖）为需自有 backend 的家族的真落地缺口，
  本演练以 ③reuse-existing 规避；GAP-B/C 为文档可发现性次缺。
- **clean-room 偏置诚实注记**：同栖仓库·in-house agent（非真第三方）；Demo = Template clone-adapt（真净新设计≈0，
  证的是可跟随性非原创工程量）。Tier-3（真第三方）仍为 optional 待办。

---

## 8. 第二版 · own-emitter 类接入（Widget·B3 clean-room·2026-07-15）

> **与 §1-7 reuse-emitter(Demo) 分行·不合并表述**（判别键 = emitter-class：reuse-existing vs own-backend）。
> 详录：`experiments/active/g8-stage3-attack/B3-cleanroom-own-emitter.md`。drill 工件（Widget 玩具家族）drill 后 revert（一次性·[F-3] 默认 RED·不入库）·本节保结论 + 永久 recipe 文档修复。

- **接入类型**：**own-emitter**（自有 `WidgetBackendEmissionDriver.cpp` + 第 2 共享注册点 `BuiltinBackendEmitters.cpp`）——Demo 第一版刻意声明 ③=reuse-existing **规避** GAP-A·本轮**首次真行使 GAP-A 修好的第二注册点**。
- **接入者**：B3 冷启动无记忆 agent（非协议作者）·**forbidden-source peek = 0**（never read RVV/IME/Scalar emitter 内部）。
- **✅ 坐实项**：build 绿（weft-opt+weft-translate+6 Widget libs·0 error）· own backend emitter end-to-end functional 绿（`--weft-widget-emitc-to-cpp` 注册·发有效 C++·dialect+target FileCheck PASS·verifier 1 正+5 负）· **[F-1] 零家族分支绿**· 零回归（Template/Demo 仍过）。
- **✏ 永久 recipe 文档修复（2·保留入库）**：[GAP-B3-P4TEMPLATE-OWNBACKEND]（`P4-family-integration-doc-TEMPLATE.md` 补 own-backend 第 2 注册点 prompt + reuse-vs-own 决策）· [GAP-B3-HARNESS-BYID]（`extension-plugin-integration.md` step3 点名 `TypedBackendEmissionDriver`+`BackendEmissionRegistry.h`+Template copy-me 锚）。
- **✅ 六门机检补齐（A5 〇 补充轮·2026-07-16）**：把本轮唯一"手工推证 非机检"步（[F-3] containment）补到**机器可检**——
  **[F-1]/[F-2′]/[F-4]/[F-5]/[F-6] 五门 self-test 判别器机检 GREEN**（F-1 另 default 69-core-file 零分支）；
  **[F-3] CI diff-mode containment = 机检 GREEN**：生产评估器 `check_family_locality.evaluate_diff`（CI job `f3-family-locality` 实跑同一函数）跑 Widget 触碰集 + Widget territory + 真 manifest allowances → GREEN，**3 反向控制**（+core-file→RED CORE-EDIT·+RVV→RED CROSS-FAMILY·抽掉 backend allowance→RED·机械坐实 GAP-A allowance 价值）；
  **[F-3] default 全树 = 结构边界（如实登记·非补齐失败）**——default 门只机检**已入库的树**，Widget 是 drill-后-revert 的一次性玩具家族（不入库·避污染仓库+避 default-RED-破-CI），故 default 全树门对**已 revert 的 drill 家族结构上不可机检**（Demo 走"入库→default-GREEN"、Widget 走"revert→diff-mode 机检触碰集"·同一 [F-3] 机器·不同模式·diff-mode 恰是 [F-3] falsifier 本义）。
- **★[C1-4] v2 判读（诚实·补齐后）**：own-emitter 接入 **坐实 at build-green + own-path functional-green + [F-1]-green + forbidden-peek=0**；**六门 = falsifier 六门 self-test 机检 GREEN + [F-3] containment diff-mode 机检 GREEN**（较 Demo reuse 侧 diff-mode 手工推证**在 [F-3] containment 机检上更强**·见 §5.91）。**诚实短板（不粉饰）**：(a) clone-adapt·真净新设计 ≈0（证可跟随性·非工程量）；(b) **[F-3] default 全树 = 结构边界**（reverted-drill 家族·非可机械补·如实·非违规）；(c) build 是 `build-demo` 增量·非 fresh-clean。
- **Tier-3（真第三方·非同栖 in-house agent）仍 optional 待办**（与 v1 同注记）。
