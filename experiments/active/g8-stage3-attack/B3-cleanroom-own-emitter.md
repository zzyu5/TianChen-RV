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

### falsifier 门（own-emitter 相关轴）
| 门 | 结果 | 说明 |
|---|---|---|
| **[F-1] zero-branch** | ✅ **GREEN**（`check_zero_core_family_branch.py`：0 family-keyed branch across 69 core files） | ★own-emitter 关键门：第 2 注册点触 `BuiltinBackendEmitters.cpp` 仍**零家族分支**（纯 function-pointer 表行·for 迭代）。 |
| **[F-3] containment** | ⚠ default-mode RED（**预期·良性**） | RED 唯一因 `Widget` territory **未在 `family-manifest.v1.json` 声明**——我**刻意 scope-out 该 schema 编辑**（保触碰集与其他线不相交 + 避 [F-2'] schema-freeze trailer）。**非** containment 违规，等同任何家族 pre-manifest-entry 的 undeclared false-red（Demo T1c 加了 entry 才 green）。**own-emitter 专属 [F-3] 关切**（第 2 注册文件是否被豁免）已由 `backend_emitter_registration` allowance 覆盖（机检 `registration_allowance_files` 消费·本轮 code-verified 存在）。**手工推证**：全触碰集 ⊆ Widget territory(6 根) + shared_allowances{build_and_tooling·plugin_registration·backend_emitter_registration·tests}。 |
| F-4/F-5/F-6 | 未单跑（own-emitter delta 无关） | 这三门管 attribution/fuzz/independence-closure·与"emitter 是 own 还是 reuse"正交·T1c reuse 侧已验；own-emitter 不改其行为。 |

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
  - **六门 parity 差一步**：本轮 **未**达 T1c reuse 侧那样的"falsifier 六门全 green"——[F-3] default-mode RED（Widget territory 未声明·**刻意** scope-out schema 编辑以保触碰集不相交），F-4/F-5/F-6 未单跑。补齐 = 一条 additive `family-manifest.v1.json` Widget territory entry（机械·非 recipe 缺口）+ git-diff-mode 跑门（本轮禁 git）。**故本轮 own-emitter 证据 = "build-green + own-path 功能-green + [F-1]-green + [F-3] 手工推证"**，比"六门机检全绿"弱一档、比 docs-only drill 强多档。
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
| [F-1] zero-branch | **GREEN**（第 2 注册点零家族分支） |
| [F-3] containment | 手工推证 PASS（触碰集 ⊆ territory+allowances）；default-mode RED = Widget territory 未声明（刻意 scope-out schema·非违规）；`backend_emitter_registration` allowance 覆盖第 2 注册文件 |
| F-4/5/6 | 未单跑（own-emitter delta 正交） |
| 触碰集 | 20 家族源(6 根) + 5 test + 8 共享(2 registration cpp + 6 CMakeLists)·零未授权 core |
| 成色 | clone-adapt·净新设计≈0·证 own-emitter recipe followability·非原创工程量 |
| 具名 gap 修 | [GAP-B3-P4TEMPLATE-OWNBACKEND]（P4 模板补 own-backend prompt）·[GAP-B3-HARNESS-BYID]（step 3 命名 `TypedBackendEmissionDriver` harness） |
| 残缺 | 六门机检 parity（+1 additive manifest territory entry·git-diff-mode·可机械补）；fresh-build 复证；Tier-3 真第三方 |
| 证据指针 | `experiments/active/g8-stage3-attack/B3-cleanroom-own-emitter.md`（本文件） |

**另建议主会话登记**：`BuiltinBackendEmitters.cpp` L22-23 注释 stale（漏 Scalar·未注 own-backend-only）——核心源·非 recipe·本轮未改·待裁。
