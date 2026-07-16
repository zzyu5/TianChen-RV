# B3 clean-room 第二轮 — own-emitter 接入实录（[C1-4] v2 证据）

> **性质**：C1 头牌「合取存在性 → 可复制扩展接入协议」的 **external-followability** 证据 **第二版**。
> 第一版（T1c·`docs/reports/2026-07-12-P4-Tier-2-cleanroom-接入演练-T1c.md`）做的是 **reuse-emitter** 家族
> （`weft_demo`·声明 ③=reuse-existing·**不**注册自有 backend emitter·刻意规避 GAP-A）。
> **本轮 = own-emitter 类**：自有 `<Fam>BackendEmissionDriver.cpp` + 第 2 共享注册点 `BuiltinBackendEmitters.cpp`。
> 新家族 = `Widget`（方言 `weft_widget`·Template 参考范本 clone-adapt）。
>
> 生成日期：2026-07-15 · 接入者身份：B3 clean-room 冷启动 agent（非协议作者）· 纯案头（禁 e2e/新计时/板）· build=本地 build（非板）。
> 触碰集：recipe 文档 + 新建 Widget 玩具家族 + 本 audit；与 Line A 板测(board/T3)+其他 B agent 不相交。

---

## ① 步骤 1 — 修的 own-emitter 接入 recipe 文档缺口

**审计结论**：own-emitter 的**注册机制本体**在 spec 层已齐（GAP-A 早修）——
`extension-plugin-integration.md` §Registration **step 3**（第 2 共享注册点）+
`extension-family-plugin-template.md` §Real Touch-Set（`Weft<Fam>BackendEmitter` lib + CMakeLists LINK）+
`schema/family-manifest.v1.json` `shared_allowances.backend_emitter_registration`（机检豁免·code-verified 本轮）**三处一致覆盖**。
真缺口在**两处**（本轮修）：

| # | 缺口 | 位置 | 修法 | 状态 |
|---|---|---|---|---|
| **[GAP-B3-P4TEMPLATE-OWNBACKEND]** | 作者**照抄填空**的 [P-4] 集成文档模板 `P4-family-integration-doc-TEMPLATE.md` **不 prompt own-backend 第 2 注册点**：§1 ③ 只列三 cpp 不分 reuse/own；§2 触碰集块只列 `BuiltinExtensionPlugins.cpp`（step 2），**漏** `BuiltinBackendEmitters.cpp`（step 3）+ 其 CMakeLists LINK；"Registration done?" 只问 `register<Fam>ExtensionPlugin`。→ 冷作者拿模板当 checklist 会**漏掉 own-emitter 第 2 注册步**。 | `docs/method/P4-family-integration-doc-TEMPLATE.md` §1③ + §2 | §1③ 加 **reuse vs own-emitter 决策 prompt**（`③ mode = <reuse/own-emitter>`）；§2 触碰集块加 `BuiltinBackendEmitters.cpp` step-3 行；§2 加 3 条 own-backend 填空（backend-emitter 注册 / `weft-translate --weft-<fam>-emitc-to-cpp` 路由 / `Weft<Fam>BackendEmitter` build-link）。 | ✅ 修 |
| **[GAP-B3-HARNESS-BYID]** | recipe 命名了**注册函数** `register<Fam>BackendEmitter` 却从不命名驱动体构造的**共享 harness `TypedBackendEmissionDriver`**——冷作者知道"要注册"却不知"注册的是什么、驱动体照哪个抄"。属 [P-1] by-ID（可从 Template copy-me 反推），但 recipe 连**名字/头文件锚**都没给。 | `.trellis/spec/plugin-protocol/extension-plugin-integration.md` §Registration step 3 | step 3 加"What the driver body does"段：命名 `TypedBackendEmissionDriver` harness + 接口头 `BackendEmissionRegistry.h`（`moduleHasBackendBody` 门 + `convertModuleWithBackendEmitter`）+ Template copy-me 锚 `TemplateBackendEmissionDriver.cpp` + 指出 core seam `tryConvertModuleWithRegisteredBackend` 零家族分支 + 产 `Weft<Fam>BackendEmitter` lib + 路由 `--weft-<fam>-emitc-to-cpp`。 | ✅ 修 |

**未修（越界·仅报备主会话）**：`lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp` 顶部注释（L22-23）**stale**——
写「RVV, Toy, Template, TensorExtLite and IME are registered today」，**漏 Scalar**（实际在表 L32）**且**未注「own-backend families only」。
属**核心源注释**、非 recipe 文档 → 在「禁改核心实现」范围内，本轮不改，留主会话/用户裁（无 followability 影响·仅注释失准）。

---

## ② 步骤 2 — own-emitter clean-room 接入实录

### 盲纪律 + forbidden-peek 自证 = 0
- 只读：recipe/协议 docs（`extension-plugin-integration.md` / `extension-family-plugin-template.md` / `P4-...-TEMPLATE.md` / T1c+X3 报告 / REPOSITORY-MAP / FALSIFIER-INDEX）+ **Template copy-me 参考家族全源** + 公共头（`BackendEmissionRegistry.h` / `ConstructionTemplateArtifactAdapter.h` / `Template*Driver.h` / `TemplateExtensionPlugin.h`）+ recipe **明文点名的两个共享注册表文件**（`BuiltinExtensionPlugins.cpp` / `BuiltinBackendEmitters.cpp`——只含注册**表**，非 emitter 内部）+ CMakeLists + `family-manifest.v1.json`（声明工件）。
- **未读**任何 `lib/Plugin/{RVV,IME,Scalar}/*.cpp` 或 `lib/Conversion/RVV/*.cpp` emitter 内部实现。**forbidden-source peek = 0**（守住）。

### 五件套 [P-2] 落地（own-emitter·③=own）
| # | 件 | Widget 落地 | followability |
|---|---|---|---|
| ① | facts+relations | `lib/Plugin/Widget/WidgetExtensionPlugin.cpp` `getCapabilities()`（`widget.extension`） | docs 自足 |
| ② | legality | `lib/Plugin/Widget/WidgetVariantLegality.cpp` | docs 自足 |
| ③ | **emission pattern = OWN** | `WidgetConstructionProtocol.cpp` + `WidgetEmitCRouteProvider.cpp` + **`WidgetBackendEmissionDriver.cpp`**（自有 `TypedBackendEmissionDriver`·产 `WeftWidgetBackendEmitter` lib·`registerWidgetBackendEmitter`） | docs 自足（+本轮补 [GAP-B3-*]） |
| ④ | tests | `test/Dialect/Widget/compute-skeleton.mlir`（1 正+5 负 verifier）+ `test/Target/Widget/`（4：emitc-to-cpp / artifact-object / fail-closed / compile.test） | docs 自足 |
| ⑤ | ledger row | 见 §④ 回填清单（诚实标：clone-adapt·非 C2 曲线点） | — |

### 两个共享注册点（own-emitter 关键区别）
1. **plugin bundle**（step 2·所有家族）：`lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` +`#include` +`{"widget-extension-bundle", registerWidgetExtensionPlugin},`（表行·`plugin_registration` allowance）。
2. **★backend emitter**（step 3·**own-emitter 独有**）：`lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp` +`#include` +`::weft::plugin::widget::registerWidgetBackendEmitter,`（表行）+ 其 `CMakeLists.txt` `LINK_LIBS WeftWidgetBackendEmitter`（`backend_emitter_registration` allowance·GAP-A 修·**本轮首次被真 own-emitter 家族行使**）。

### 改的文件（触碰集）
- **新建 Widget 家族源 = 20 文件 / 6 目录根**（`lib/{Plugin,Dialect,Target}/Widget/` + `include/Weft/{Plugin,Dialect,Target}/Widget/`；Template clone + protected-rename）。
- **新建 test = 5 文件**（`test/Dialect/Widget/` 1 + `test/Target/Widget/` 4）。
- **共享文件编辑 = 8**（全在 declared shared_allowances 内）：
  - registration cpp ×2：`BuiltinExtensionPlugins.cpp`、`BuiltinBackendEmitters.cpp`；
  - CMakeLists ×6：`lib/Plugin/`、`lib/Dialect/`、`lib/Target/`、`include/Weft/Dialect/`（各 +`add_subdirectory(Widget)`）、`lib/Plugin/Builtin/`（+`WeftWidgetPlugin`）、`lib/Conversion/EmitC/Builtin/`（+`WeftWidgetBackendEmitter`）。
- **零**未授权 core dispatch/selection/lowering 编辑。

### build 绿证（本地 build·`build-demo/`）
- `cmake --build build-demo --target weft-opt` → **exit 0**，`[67/67] Linking bin/weft-opt`（67 MB·relinked·fresh ts）。所有 Widget lib 本会话新编：`libWeftWidget{Dialect,Target,Plugin,EmitCRouteProvider,ConstructionProtocol,BackendEmitter}.a`。
- `cmake --build build-demo --target weft-translate` → **exit 0**。
- **功能证**：
  - `weft-opt test/Dialect/Widget/compute-skeleton.mlir --split-input-file --verify-diagnostics` → **exit 0**（verifier 真跑：1 正例过 + 5 负例 diagnostics 全匹配）；`| FileCheck` → **PASS**。`weft_widget` 方言经**插件自动注册**（零 core dialect-init 编辑）。
  - **★own backend emitter 端到端**：`weft-translate --help` 列出 **`--weft-widget-emitc-to-cpp`**（own 路由已注册）；`weft-translate --weft-widget-emitc-to-cpp .../widget-target-artifact-object.mlir` → **exit 0**，emit 合法 C++（`int32_t v1 = weft_widget_compute_skeleton();`）；target lit `--check-prefix=HELP` + `--check-prefix=SOURCE`（含 `--implicit-check-not weft_rvv/int main`）→ **均 PASS**。
- **零回归**：`Template` / `Demo` dialect 测试仍 **PASS**（未动任何既有家族·共享编辑纯 additive）。

### falsifier 门（own-emitter 相关轴）— 〇 补充轮机检补齐（2026-07-16）
> **A5 〇 补充轮**：把本轮六门中唯一"手工推证 非机检"的一步（[F-3] containment）补到**机器可检 PASS**，
> 与 reuse-emitter(Demo) 轮同标准（Demo 轮 [F-3] = default-gate GREEN + self-test GREEN·diff-mode 亦手工推证·见 T1c §5.91）。
> 补齐用**生产评估器** `check_family_locality.evaluate_diff`（CI [F-3] 门实跑的同一函数）· **0 造数·0 manifest 污染·不重写 own-emitter**。

| 门 | 结果 | 说明 |
|---|---|---|
| **[F-1] zero-branch** | ✅ **GREEN·机检**（`check_zero_core_family_branch.py` self-test GREEN + default：0 family-keyed branch across 69 core files） | ★own-emitter 关键门：第 2 注册点触 `BuiltinBackendEmitters.cpp` 仍**零家族分支**（纯 function-pointer 表行·for 迭代）。 |
| **[F-2′] schema-def** | ✅ **GREEN·机检**（`check_schema_gate.py --self-test` 21/21） | own-emitter 未触 schema.def（`capability.schema.v1.json`/`VERSIONLOG.md`）·family-manifest 属 [F-3] 义务非 [F-2′]（必问-2 前缀收窄后·[GAP-P4-C]）。 |
| **[F-3] containment — CI diff-mode** | ✅ **GREEN·机检（补齐·本轮升级）** | **手工推证 → 机检**：生产评估器 `evaluate_diff(Widget-触碰集, families+Widget-territory, 真-manifest-allowances)` = **GREEN**（触碰集 ⊆ Widget territory(6 根) + shared_allowances{plugin_registration·backend_emitter_registration·tests·build_and_tooling}）。**3 反向控制证判别非空**：(neg-core) +`lib/Transforms/VariantSelection.cpp` → **RED CORE-EDIT**；(neg-xfam) +`lib/Dialect/RVV/...` → **RED CROSS-FAMILY-PR**；(neg-noallow) 抽掉 `backend_emitter_registration` allowance → **RED CORE-EDIT on `BuiltinBackendEmitters.cpp`**（★机械坐实 GAP-A 的价值：own-emitter 第 2 注册点**正因** drill 补的 allowance 被生产评估器消费才 contained·非白送）。命令+完整 transcript 见下方「[F-3] 机检 transcript」。 |
| **[F-3] containment — default 全树** | 🟡 **结构边界（如实登记·非补齐失败）** | default 门机检的是**已入库的树**；Widget = 一次性 drill 家族·**drill 后 revert 未入库**（[F-3] default 会因 undeclared-family-dir RED·破 CI）。要拿到 Demo 那样的 default-GREEN 必须**永久入库这个抛弃型玩具家族 + 加 phantom manifest entry**——drill 刻意不做（避污染仓库/避 default-RED-破-CI）。故 default 全树门对**已 revert 的 drill 家族结构上不可机检**（≠补齐失败）。Demo 走"入库→default-GREEN"路；Widget 走"revert→diff-mode 机检触碰集"路·**两者是同一 [F-3] 机器·不同模式**·diff-mode 恰是 [F-3] falsifier 的本义（"PR diff ⊆ 家族 territory + allowances"）。self-test GREEN（判别器健康·与 Demo 同）。 |
| **[F-4]/[F-5]/[F-6]** | ✅ **GREEN·机检**（三门 self-test 判别器全 GREEN：F-4 attribution / F-5 failclosed / F-6 independence） | 这三门管 attribution/fuzz/independence-closure·与"emitter 是 own 还是 reuse"**正交**（own-emitter 不改其行为）——本轮**机检确认**该正交性（非仅断言）·补齐 B3 原"未单跑"。 |

#### [F-3] 机检 transcript（可复现·生产评估器·无 git/无 build）
```
$ python3 - <<'PY'   # 见 §④ 后「复现命令」全文
...
=== [F-3] CI diff-mode verdict (production evaluate_diff) ===
POSITIVE (real Widget touch-set): GREEN []
NEG-CORE  (+VariantSelection.cpp):  RED  CORE-EDIT
NEG-XFAM  (+RVV territory):         RED  CROSS-FAMILY-PR
NEG-NOALLOW (drop backend allowance): RED CORE-EDIT on BuiltinBackendEmitters.cpp
PY
```
Widget 触碰集 = 14 家族源（Template-clone·6 territory 根）+ 2 共享注册表（`BuiltinExtensionPlugins.cpp`·`BuiltinBackendEmitters.cpp`）+ 6 CMakeLists + 5 test。评估器为 `tools/lint/check_family_locality.py`（CI job `f3-family-locality` 实跑同一 `evaluate_diff`·同一 `registration_allowance_files` 解析真 manifest 的 2 allowance）。

### 遇到的接入缺口 / 摩擦（真 build 暴露·docs-drill 不可见）
1. **[GAP-B3-P4TEMPLATE-OWNBACKEND] / [GAP-B3-HARNESS-BYID]**（已在 §① 修）——spec 层注册机制齐、fill-in 模板漏 prompt + harness 未命名。
2. **clone-rename 陷阱（非 docs 缺口·记为 followability 摩擦）**：贪婪 token rename 两次踩坑，均真 build 才暴露、docs-only drill 测不到：
   - `s/Template/Widget/` 误伤**共享**符号 `ConstructionTemplateArtifactAdapter`（core Target 基建·`include/Weft/Target/`）→ 首 build `fatal error: ConstructionWidgetArtifactAdapter.h 无此文件`；修=restore `ConstructionWidget→ConstructionTemplate`。
   - 保留 bare 小写 `template` 时漏了**家族身份串**（plugin id `"template-plugin"` / cap id `"template.extension"` / `evidencePrefix "weft.template"`）→ 运行期 `duplicate extension bundle plugin id 'template-plugin'`（与真 Template 撞名）；修=小写 `template→widget`（家族内无 C++ `template` 关键字·已验证·安全）。
   - **教训**：Template 范本内「family token」与「shared-infra token」/「C++ 关键字」同形（`Template`↔`ConstructionTemplate`·`template`↔身份串），clone-adapt 须 per-token 甄别，非盲 sed。此摩擦**本身是 external-followability 的真实成本**（真外部人也会撞），但可自解、非 docs 阻断。

---

## ③ [C1-4] 第二版证据结论（own-emitter 是否坐实·成色）

**判读 = own-emitter 接入 build-green + 功能-green 坐实；六门全绿 parity 未满（差一处刻意 scope-out 的 schema territory 声明）。**

- **坐实的（own-emitter 专属证据·真 build 兑现）**：
  1. 一个 forbidden-peek=0 的 clean-room agent，循 recipe docs + Template copy-me，真接入一个**自有 EmitC backend** 家族 `Widget` 到 **build 绿**（weft-opt + weft-translate + 6 Widget lib·0 error）。
  2. **第 2 共享注册点** `BuiltinBackendEmitters.cpp`（GAP-A 修的 `backend_emitter_registration` allowance）**首次被真 own-emitter 家族行使**——不再是 Demo 那样"声明 reuse 以规避"。
  3. own backend emitter **端到端功能兑现**：own 路由 `--weft-widget-emitc-to-cpp` 注册 + emit 合法 C++ + FileCheck SOURCE PASS——**不止编译、真跑通 own 路径**。
  4. **[F-1] zero-branch GREEN**：own-emitter 的第 2 注册点**不引入任何 core 家族分支**（I3 holds）——这是 own-emitter 相对 reuse-emitter 唯一新增的 core 触点，其零分支性经机检坐实。
- **成色诚实（禁"实质胜利"）**：
  - **clone-adapt·真净新设计 ≈ 0**：Widget = Template 范本近复制（protected-rename），证的是 **own-emitter recipe 的可跟随性**，非原创工程量（同 T1c Demo 偏置）。
  - **偏置仍在**：同栖仓库·in-house agent（非真第三方）；agent 的"盲"是规则约束盲、非知识真空盲。
  - **六门 parity（〇 补充轮机检补齐后·2026-07-16）**：本轮 [F-3] 原为唯一"手工推证 非机检"步，现补到机检——
    **[F-1]/[F-2′]/[F-4]/[F-5]/[F-6] 五门 self-test 判别器机检 GREEN**（F-1 另有 default 69-core-file 零分支）；
    **[F-3] CI diff-mode containment = 机检 GREEN**（生产评估器 `evaluate_diff` 跑 Widget 触碰集 + 3 反向控制判别·手工推证 → 机检·**强于** Demo diff-mode 的手工推证）；
    **[F-3] default 全树 = 结构边界（如实登记）**——default 门只机检**已入库的树**，Widget 是 drill-后-revert 的一次性玩具家族（不入库·避污染仓库+避 default-RED-破-CI），故 default 全树门**对已 revert 的 drill 家族结构上不可机检**（≠补齐失败·Demo 走"入库→default-GREEN"、Widget 走"revert→diff-mode 机检"·同一 [F-3] 机器不同模式）。
    **净**：本轮 own-emitter 证据 = "build-green + own-path 功能-green + **falsifier 六门 self-test 机检 GREEN + [F-3] containment diff-mode 机检 GREEN** + 单一 [F-3] default 全树结构边界（reverted-drill·如实）"，较 B3 原稿"[F-3] 手工推证"**升一档**、与 Demo reuse 侧"default-GREEN + self-test + diff 手工推证"标准**在 [F-3] containment 机检上更强**（Demo diff 亦手工推证·见 T1c §5.91）。
  - **build = 增量**（`build-demo` 上·非 fresh clean）：所有 Widget object + 两 binary 本会话新（重）编、未变 core object 复用（正确增量行为）；fresh clean build 可再固证（未做·保守预算）。

**净**：[C1-4] 从"仅 reuse-emitter 兑现"扩到"**own-emitter 亦兑现**（build+功能+[F-1]）"，头牌"可复制扩展接入协议"的**自有 emitter 面**得实测支撑；诚实残缺 = 六门机检 parity（可机械补）+ clone 偏置（Tier-3 真第三方仍 optional 待办）。

---

## ④ T1c 回填清单（own-emitter 行·与 reuse-emitter 分行·主会话入表·禁我改 sealed 表）

> **纪律**：own-emitter 与 reuse-emitter(Demo) **分两行·不合并表述**（emitter-class 是判别键·成色不同）。以下为建议行内容，落点/列名由主会话映射进 T1c 记录表。

**建议 own-emitter 行（新增·与 Demo 行并列）**：

| 列 | 值 |
|---|---|
| drill / 轮次 | B3 clean-room 第二轮（[C1-4] v2） |
| 家族 | `Widget`（方言 `weft_widget`） |
| **emitter-class** | **own-emitter**（自有 `WidgetBackendEmissionDriver` + 第 2 注册点 `BuiltinBackendEmitters.cpp`）〔Demo 行 = reuse-emitter〕 |
| 接入者 | in-house clean-room agent（非协议作者·非真第三方） |
| forbidden-peek | **0**（RVV/IME/Scalar emitter 内部零 peek） |
| build 绿 | **TRUE**（`build-demo`·weft-opt+weft-translate+6 Widget lib·0 error·增量非 fresh） |
| own-path 功能 | **TRUE**（`--weft-widget-emitc-to-cpp` emit 合法 C++·dialect+target lit FileCheck PASS·verifier 真跑） |
| [F-1] zero-branch | **GREEN·机检**（self-test + default 69-core-file 零分支·第 2 注册点零家族分支） |
| [F-2′] schema-def | **GREEN·机检**（self-test 21/21·own-emitter 未触 schema.def） |
| [F-3] containment | **CI diff-mode = GREEN·机检（补齐·手工推证升级为机检）**：生产评估器 `evaluate_diff(Widget 触碰集, +Widget territory, 真 manifest allowances)` GREEN + 3 反向控制（core-edit/cross-family/no-allowance 全 RED·机械坐实 GAP-A allowance 价值）。**default 全树 = 结构边界**（Widget drill-后-revert·未入库·default 门只检已入库树·对 reverted-drill 家族结构上不可机检·≠补齐失败）。self-test GREEN。 |
| F-4/5/6 | **GREEN·机检**（三门 self-test 判别器全 GREEN·机检确认 own/reuse 正交·补齐 B3 原"未单跑"） |
| 触碰集 | 20 家族源(6 根) + 5 test + 8 共享(2 registration cpp + 6 CMakeLists)·零未授权 core |
| 成色 | clone-adapt·净新设计≈0·证 own-emitter recipe followability·非原创工程量 |
| 具名 gap 修 | [GAP-B3-P4TEMPLATE-OWNBACKEND]（P4 模板补 own-backend prompt）·[GAP-B3-HARNESS-BYID]（step 3 命名 `TypedBackendEmissionDriver` harness） |
| 残缺 | **[F-3] default 全树 = 结构边界**（reverted-drill 家族·非可机械补·如实登记）；其余五门 + [F-3] containment 已机检 GREEN；fresh-build 复证；Tier-3 真第三方 |
| 证据指针 | `experiments/active/g8-stage3-attack/B3-cleanroom-own-emitter.md`（本文件·含 [F-3] 机检 transcript + 复现命令） |

**另建议主会话登记**：`BuiltinBackendEmitters.cpp` L22-23 注释 stale（漏 Scalar·未注 own-backend-only）——核心源·非 recipe·本轮未改·待裁。

---

## ⑤ [F-3] containment 机检 — 复现命令（〇 补充轮·2026-07-16·可复现·无 git/无 build/无 manifest 污染）

> 用**生产评估器**（CI job `f3-family-locality` 实跑的同一 `evaluate_diff` + `registration_allowance_files`）机检 Widget own-emitter 触碰集的 [F-3] 变更收容。
> Widget territory 与 registration allowances 均取自**真** `schema/family-manifest.v1.json`（territory 按 drill 声明·6 territory 根）·**不改 manifest**（不为抛弃型 drill 家族加 phantom entry）。

```python
python3 - <<'PY'
import sys, json
sys.path.insert(0, 'tools/lint')
from check_family_locality import evaluate_diff, registration_allowance_files, matches_range
doc = json.load(open('schema/family-manifest.v1.json'))
allow = registration_allowance_files(doc)  # 真 manifest 的 2 registration allowances
widget = {"family": "Widget", "source_ranges": [
    "lib/Dialect/Widget", "lib/Plugin/Widget", "lib/Target/Widget",
    "include/Weft/Dialect/Widget", "include/Weft/Plugin/Widget", "include/Weft/Target/Widget"]}
families = doc["families"] + [widget]
family_source = [
    "include/Weft/Dialect/Widget/IR/WidgetDialect.h", "include/Weft/Dialect/Widget/IR/WidgetOps.td",
    "include/Weft/Plugin/Widget/WidgetBackendEmissionDriver.h", "include/Weft/Plugin/Widget/WidgetConstructionProtocol.h",
    "include/Weft/Plugin/Widget/WidgetEmitCRouteProvider.h", "include/Weft/Plugin/Widget/WidgetExtensionPlugin.h",
    "include/Weft/Target/Widget/WidgetTargetSupportBundle.h", "lib/Dialect/Widget/IR/WidgetDialect.cpp",
    "lib/Plugin/Widget/WidgetBackendEmissionDriver.cpp", "lib/Plugin/Widget/WidgetConstructionProtocol.cpp",
    "lib/Plugin/Widget/WidgetEmitCRouteProvider.cpp", "lib/Plugin/Widget/WidgetExtensionPlugin.cpp",
    "lib/Plugin/Widget/WidgetVariantLegality.cpp", "lib/Target/Widget/WidgetTargetSupportBundle.cpp"]
shared_reg = ["lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp", "lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp"]
non_source = ["lib/Dialect/CMakeLists.txt","lib/Plugin/CMakeLists.txt","lib/Target/CMakeLists.txt",
    "include/Weft/Dialect/CMakeLists.txt","lib/Plugin/Builtin/CMakeLists.txt","lib/Conversion/EmitC/Builtin/CMakeLists.txt",
    "lib/Plugin/Widget/CMakeLists.txt","test/Dialect/Widget/compute-skeleton.mlir",
    "test/Target/Widget/widget-emitc-to-cpp.mlir","test/Target/Widget/widget-target-artifact-object.mlir",
    "test/Target/Widget/widget-fail-closed.mlir","test/Target/Widget/compile.test"]
ts = family_source + shared_reg + non_source
print("POSITIVE:", "GREEN" if evaluate_diff(ts, families, allow)[0] else "RED")
print("NEG-CORE:", "RED" if not evaluate_diff(ts+["lib/Transforms/VariantSelection.cpp"], families, allow)[0] else "GREEN(BUG)")
print("NEG-XFAM:", "RED" if not evaluate_diff(ts+["lib/Dialect/RVV/IR/RVVOps.td"], families, allow)[0] else "GREEN(BUG)")
print("NEG-NOALLOW:", "RED" if not evaluate_diff(ts, families, {"lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp"})[0] else "GREEN(BUG)")
PY
```
**输出（2026-07-16 实跑）**：`POSITIVE: GREEN` · `NEG-CORE: RED` · `NEG-XFAM: RED` · `NEG-NOALLOW: RED`。
四判决全符预期 → [F-3] containment 机检坐实（正例 contained·三反向控制判别非空·GAP-A allowance 价值机械可见）。
