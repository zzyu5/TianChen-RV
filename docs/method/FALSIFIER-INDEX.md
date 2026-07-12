# FALSIFIER-INDEX — [F-1..F-6] → 具体 checker / lit / gtest 映射（R4）

> **用途**：模板五大件之 ⑤falsifier 组的**软肋 = 无 F→文件映射索引**（[TEMPLATE-AUDIT] 认定）。
> 本 doc 补这张表：canon 的 [F-1..F-6] 每一个 → 它的**权威定义位置** + **具体机检工件**（脚本 / lit / gtest / CI job）
> + **诚实机制化状态**。配套 [REPOSITORY-MAP-五大件.md](./REPOSITORY-MAP-五大件.md) §1.⑤。
>
> **[F-1..F-6] 定义权威 = `docs/canon/Weft-RV_科研目标总纲v2.md:105–110`**（不在此重抄语义，只做文件映射）；
> 现状栏对齐 `docs/canon/Weft-RV_执行总纲v2.md:32–37`。**本 doc 只索引，不改 code / 不改判据。**

---

## 1. [F-1..F-6] 主映射表

| F | 一句话判据 | 定义（权威） | 具体机检工件 | CI | 机制化状态（诚实） |
|---|---|---|---|---|---|
| **[F-1]** | 零核心分支（family-branch grep=0）+ family-regex manifest + 判读规程 | `科研目标总纲v2:105` · `.trellis/spec/testing/mlir-testing-contract.md:72`（[F-1] 判读规程 + manifest 形态） | **⚠ 见 §2 命名碰撞**：零分支 grep 的 family-regex manifest **尚未落盘**（执行总纲 F-1 = 「部分」，缺 manifest/CI/判读规程）。另有一个**同名不同物**的工具 `tools/lint/check_construction_manifest_regex.py`（自标 "F-1 construction-manifest shape gate"，实为 C1 合取的**构造体 SHAPE 正则门**，非零分支门） | **✅ `f1-zero-branch-gate`**（falsifier-gate.yml·build-free） | **零分支门：✅ 进 CI**（2026-07-12·`schema/family-regex.v1.json` manifest + `tools/lint/check_zero_core_family_branch.py` 判读规程 + self-test·grep=0 across 69 core files·I3 holds）；shape 门=[C1-SHAPE]（可跑·非 CI·见 §2） |
| **[F-2′]** | 家族接入 PR diff ∩ schema.def = ∅（操作门）+ 报告门（shape-hash + 版本日志） | `科研目标总纲v2:106` · `core-invariants.md:80` | `.trellis/scripts/check_schema_gate.py`（self-test / `report --check` / `gate --base --head`）+ 红队 `.trellis/scripts/redteam_schema_gate.py` | **✅ `falsifier-gate.yml` job `schema-def-gate`** | **在位 · CI 常绿 · 带红队证伪** |
| **[F-3]** | 变更收容：家族接入 PR diff ⊆ 该家族 manifest 声明范围 + 表行 + 文档（**清单机检·非物理目录**） | `科研目标总纲v2:107` · `.trellis/spec/plugin-protocol/locality-contract.md:76` | **`tools/lint/check_family_locality.py`**（default 全仓一致性 + CI `--base/--head` diff 收容 + `--self-test` 双评估判别）·manifest = **`schema/family-manifest.v1.json`**（per-family `source_ranges`·**与 [F-EMIT] 白名单同址**·shrink-only ratchet `baseline_range_count==Σranges`）·core 权威 = `schema/family-regex.v1.json` core_scope（anti-widen 复用·不复制 canon） | **✅ `f3-family-locality`**（falsifier-gate.yml·self-test + default·build-free） | **✅ 进 CI（清单机检形态）**（2026-07-12 用户裁「**MLIR 分层为主 + family 清单机检**」：不做跨 Dialect/Conversion/Target 根大搬迁=违 MLIR 生态惯例·模板要像 MLIR 项目；收容性由 manifest 断言、非物理目录保证）。default 门 **GREEN @ HEAD**（242 family-root 源文件全归其一 manifest·零孤儿·零跨家族污染·零 range 落 core·ratchet 持）；旧「gated on 目录归拢」前提 **WITHDRAWN** |
| **[F-4]** | 归因完备（[D-4] 分级：M1 查①②，M2+ 查③）；`only_feasible`/`static_order`/`prior`/`measured` 归因出口 | `科研目标总纲v2:108` | **编译期选择归因 JSONL 出口在位**：`--weft-select-variants=attribution-jsonl=<path>` sink（`lib/Transforms/VariantSelection.cpp:1094+`）每 planned kernel 发一条 canonical-JSON `{candidates[],chosen,declared_instance_hash,kernel,keys_evaluated{},reason,ts}`；lit `attribution-jsonl.mlir`（only_feasible+static_order）+ `attribution-jsonl-instance-hash.mlir`（NoViableVariant reason=null + hash profile-等价）**双绿**；独立门 `tools/lint/check_f4_attribution_jsonl.py`（shape+enum+reason-可导出+M1 100% 覆盖+查①② only_feasible/static_order 双出口·`--self-test` 判别 GREEN·build-free SKIP lane） | **✅ `f4-attribution-jsonl`**（`falsifier-gate.yml`·self-test + build-free SKIP lane） | **✅ 编译期归因（D-4①）进 CI**（2026-07-12·JSONL 出口早在位·本轮补机检门+CI·[FALSIFIER-INDEX 前提 stale 纠偏：旧写"JSONL 出口...缺"实已在位]）；剩 D-4② 装载期解析记录（每进程 1 条·[D-2a]）+ 调度/合法性阶段归因 = **M2 增量·非 M1 门**（RVV schedule-stage sink 已在 `RVVMonolithicBlockDotSourceFrontDoor.cpp`·未纳 F-4 M1 门） |
| **[F-5]** | fail-closed 模糊：随机删/伪造事实下非法 plan 全被拒（非误编译/误发射） | `科研目标总纲v2:109` | `tools/fuzz/f5_failclosed_fuzz.sh`（变异 valid typed-region → `weft-opt --weft-rvv-lower-to-emitc`，断言 graceful-diagnostic + 非零退出 + 不崩 + 不静默过）→ 产出 `experiments/active/result-tables/T1b_failclosed_runtime.csv`；lit 语料 `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-*-fail-closed.test`（14 个） | **✅ `f5-failclosed-fuzz`**（指纹棘轮·build-free SKIP lane） | **✅ 进 CI**（2026-07-12·`check_f5_failclosed_fingerprint.py` + `schema/f5-failclosed-baseline.v1.json`）·★**现 20/20**（f96f767a 的 3 fail-OPEN=neg_qk/neg_weight_stride/neg_activ_stride 在 HEAD 已闭合·verify 拒非正 i64 facts）·门=指纹不退化棘轮·T1b sealed CSV 未改 |
| **[F-6]** | 独立家族判据：变体能力谓词 implies 闭包 ∩ {rvv.*} = ∅（脚本化）+ 存在向量缺席实例使其变体 `only_feasible` 且真实被选中 | `科研目标总纲v2:110` · `core-invariants.md:84` | lit `test/Transforms/VariantSelection/f6-independent-scalar-family-emittable.mlir` + gtest `test/Plugin/ScalarExtensionPluginTest.cpp`（F-6 双断言，989 LOC）+ **独立门 `tools/lint/check_f6_scalar_family_independence.py`**（闭包∩rvv.*=∅ 脚本化 + `only_feasible` 真实选中跑活验证·`--self-test` 判别 GREEN）；受测家族 [X-SCALAR] owned 内核已落地（tq2_0 `f96f767a` / q4_0 `2dd654d8` / 曳光弹 `5c010b2b`）· 判据④机制 landed（VariantSelection:165 只 feasible 谓词·reason=only_feasible）| **f6-independence-gate job**（`falsifier-gate.yml`·self-test + build-free 默认门 61f0c8fe/XS-M3） | **机制已闭 + 独立门在位**（判据④ landed·N2 boundary PASS 完整；仅剩 XS-M2 `scalar.zfh`/XS-M4 LED-2=家族完整性非 F-6 条件） |

**[P-3] 接入验收 = falsifier 组全绿**（`科研目标总纲v2:71`：[F-1..F-6]，独立家族含 F-6）。**✅ 2026-07-12 [F-1..F-6] 六门全进 CI = [P-3] falsifier 组全进 CI 达成**（F-3 = 最后一门·清单机检形态进 CI）。
**M1 = 证据线闭合**目标（`科研目标总纲v2:211`）= F-1 / F-2′ / F-5 / F-6 进 CI。**✅ 2026-07-12 达成**：F-1（`f1-zero-branch-gate`）+ F-5（`f5-failclosed-fuzz`）新进 CI，F-2′（`schema-def-gate`）+ F-6（`f6-independence-gate`）早在 → **[F-1..F-6] 中 F-1/F-2′/F-5/F-6 四门真进 CI = M1 证据线闭合达成**。**F-4 编译期归因（D-4①）JSONL 出口 + 门进 CI（`f4-attribution-jsonl`）** → 五门。**★F-3 变更收容（`f3-family-locality`·2026-07-12·用户裁「MLIR 分层为主 + family 清单机检」·`schema/family-manifest.v1.json` + `tools/lint/check_family_locality.py`·清单机检非物理搬迁）进 CI → [F-1..F-6] 六门全进 CI**。

---

## 2. ★[F-1] 命名碰撞 codify（供 [RENAME] · 与 REPOSITORY-MAP §3 并列）

「F-1」在仓库里指**两个不同的东西**：

| 记号 | 所指 | 工件 | 语义 |
|---|---|---|---|
| **F-1-(零分支)** | canon 的**零核心分支** falsifier | 尚缺：family-regex manifest + grep-clean CI + 真/假阳性判读规程 | grep `<family_regex>` core/ = 0；家族名不得进核心控制流 |
| **[C1-SHAPE]（原 F-1-shape 门）** | `check_construction_manifest_regex.py` 自标 **"[C1-SHAPE] construction-manifest shape gate"** | `tools/lint/check_construction_manifest_regex.py` | C1 合取的**构造体 SHAPE 正则门**：对 constructed cell 的 E5 realized-body manifest 正则出合法 typed-primitive 形态、拒不透明手写 helper（[L-8]） |

**二者不同层**（一个查「家族名不进核心分支」= I3 不变量；一个查「constructed 是可检形态」= C1 合取证据件）。
**✅ [RENAME] 已执行（2026-07-12）**：shape 门自标已从 "F-1 ..." 改为 **"[C1-SHAPE] ..."**（`check_construction_manifest_regex.py`），「F-1」标号自此**独归零分支门**，grep/阅读歧义消除。**本节是唯一权威碰撞记录**（与 REPOSITORY-MAP-五大件.md §3.1 的 P1 三所指并列）。历史 dated 报告（`2026-07-11-TEMPLATE-AUDIT-structure.md` 等）内的旧 "F-1-shape" 引用属 append-only 存档·不回改·以本节为准。

---

## 3. 相邻机检门（falsifier 家族外，同在 `falsifier-gate.yml` / `tools/lint/` 的 fail-closed 门）

这些不是 [F-1..F-6] 编号项，但同属模板 ⑤ 机检面，列此避免外来者误以为 falsifier 只有六个：

| 门 | 工件 | CI job | 判据 |
|---|---|---|---|
| **[F-EMIT] 前门 provenance**（裁决一.3） | `tools/lint/check_frontdoor_provenance.py` | `frontdoor-provenance-gate` | 手写 `emitRepackGem*` 直发射旁路必须白名单登记（`schema/emit-bypass-whitelist.v1.json`）+ shrink-only ratchet；旁路存量→0 |
| **opponent-facts pin**（裁决九.2） | `tools/lint/check_opponent_facts_pin.sh` | `opponent-facts-pin` | ggml-pin 一 bump 就 STALE-fail，逼重验 opponent-fact 行锚 |
| **monolith-retire**（裁决九.4） | `tools/lint/check_monolith_retire.py` | `monolith-retire-gate` | 每存活 vec_dot monolith 必须白名单命名+批次；无 `#if 0` verifier tomb |
| **cert 三要件**（cert-hardening） | `tools/lint/check_cert_requirements.py` | `cert-requirements-gate` | claim=full 的数值 cert 必须过语料完备 / 输入路径同源 / oracle 独立三要件（`schema/cert-lineage.v1.json`） |
| **RETIRED-INDEX**（判定书轴B③） | `tools/lint/gen_retired_index.py`（生成）+ `tools/lint/check_retired_index.py`（校验） | `retired-index-gate` | `schema/retired-index.generated.json` 机生+FRESH+四要件+覆盖完备 |

**证据卷宗 dir-lints**（`tools/lint/`，非 CI-in-falsifier-gate 但机检 experiments/ 卫生）：
`gen_experiments_index.py` / `check_index_consistency.py` / `check_experiments_layout.py` /
`check_experiments_data_only.py` / `check_manifest.py` / `check_docs_canon.py`。

---

## 4. 一句话状态板（M1 证据线闭合视角）

- **✅ 进 CI 的 [F-1..F-6]**：**F-1 + F-2′ + F-3 + F-4 + F-5 + F-6 = 全六门**（★2026-07-12·**[P-3] falsifier 组全进 CI 达成**——F-1 零分支门 `f1-zero-branch-gate` + F-5 fail-closed fuzz `f5-failclosed-fuzz` + F-4 归因门 `f4-attribution-jsonl` + **F-3 变更收容门 `f3-family-locality`**（清单机检形态·`schema/family-manifest.v1.json` per-family source_ranges·非物理目录搬迁·default 全仓 GREEN + self-test 双评估判别）新进 CI；F-2′ + F-6 早在）+ [F-EMIT] / opponent-pin / monolith-retire / cert-三要件 / RETIRED-INDEX / [C1-SHAPE]。
- **★F-5 现 20/20**（2026-07-12·非 17/20）：f96f767a 的 3 fail-OPEN（neg_qk/neg_weight_stride/neg_activ_stride）在 HEAD **已闭合**（verify 层拒非正 i64 block facts）→ 全 fail-closed。门语义 = **指纹不退化棘轮**（逐格 baseline + 语料 pin·非硬 20/20·防删格骗绿·`schema/f5-failclosed-baseline.v1.json`）。T1b sealed CSV 未改（f96f767a 快照）。
- **部分**：F-4（★编译期归因 D-4① **已进 CI**·剩 D-4② 装载期解析记录 + 调度/合法性阶段归因 = M2 增量·非 M1 门）/ F-6（闭包脚本 + `only_feasible` 真实选中 = XS-M3）。
- **✅ 曾 gated 的 F-3 已进 CI**（2026-07-12 用户裁「MLIR 分层为主 + family 清单机检」）：**旧「gated on 跨 4 lib 根目录归拢」前提 WITHDRAWN**——不做跨根大搬迁（违 MLIR 生态惯例·模板要像 MLIR 项目非自造体系），改由 `schema/family-manifest.v1.json` 逐家族声明 source_ranges + `check_family_locality.py` 机检收容（default 全仓一致性：242 family-root 源文件全归其一 manifest·零孤儿·零跨家族污染·零 range 落 core；CI base..head diff 收容；anti-widen 复用 family-regex core_scope；shrink-only ratchet）。**至此 [F-1..F-6] 无 gated 项**。
- **✅ [C1-SHAPE] 门 finding RESOLVED（2026-07-12·seal-regex bug·fixed·门 GREEN 84/84）**：曾报 RED 3/84 `gemm_tile/{q4_0,q8_0,q4_K}` "[IME-SEAL] envelope mismatch"。**根因 = regex 字段边界 bug**（非 RVV misroute·我初始假说错）：3 cell 实为真 [IME-SEAL] IME 变体（q4_0@ime/q8_0@ime/q4_K@ime·门报 op/format 不消歧 @ime 故看着像 RVV）；`IME_ENVELOPE_RE` 的 `board_seal=[^;]+` 假设 seal 无内部分号·但 2026-07-11 [GAP-IME-LEAF-PIPELINE] 线甲b re-seal 注（"…store-once; re-sealed…"）加了内部 `;` → 提前截断 → `; opaque_helper=false$` end-anchor 不匹配 → 假 mismatch。**修 `[^;]+`→`.+`**（greedy·容内部 `;`·门强度不变：0xe210312b+0-diff+ime_matmul_tile shape 仍强制·非放松）+ 加 semicolon-in-seal regression self-test（原 self-test 无分号故 GREEN 而真 cell RED）。全跑 84/84 GREEN（ime_matmul_tile=3）。**schema/cell 无需改**（re-seal 证据合法·parser 太严）。
- **✅ [C1-4] 验收要件"falsifier 组 ≥3 家族 CI 常绿"落位登记（2026-07-12 用户裁〇.1）**：canon 总纲 `docs/canon/Weft-RV_科研目标总纲v2.md:153` 的 [C1-4] 验收 = 「falsifier 组 ≥3 家族下 CI 常绿；schema.def 自 IME 起未被家族接入触及（F-2′）；六对照面成文；（M4）外部接入实录」。**"≥3 家族 CI 常绿" 分项 MET**：[F-1..F-6] 六门全进 CI（`falsifier-gate.yml` 11 jobs·2026-07-12 `460790d7`）· **F-3 family-manifest 机检覆盖 7 家族**（`schema/family-manifest.v1.json`·242 family-root 源文件全归一 manifest·零孤儿·shrink-only ratchet）≥3 家族门槛达成有余 · F-1 零分支门跨 69 core 文件 · F-5 20/20 · F-2′ schema-def-gate（IME 起 schema.def 未被家族接入触及·逐 PR 审计在案）。**登记性质 = 验收证据指针**（不编辑 canon 总纲条文·仅在本 INDEX 记 met + evidence ptr）。剩余 [C1-4] 分项：六对照面成文（部分·`docs/method/` 对照面文档）。
- **✅ [C1-4]「（M4）外部接入实录」= [P-4] 分层定标推进（2026-07-12 用户裁三·代号 X3 废止→[P-4]/T1c）**：**Tier-1 完成**（冷启动 drill `docs/reports/2026-07-12-X3-冷启动接入-drill-外部接入实录.md`：判断层可跟随 + 8 缺口具名）→ **8 缺口修复**（`98c6fafe`+`e20bd0f8`·docs 准确化·协议非改语义）→ **drill 重跑 clean-room 验证干净**（`docs/reports/2026-07-12-P4-drill-重跑-8缺口验证.md`：从 README 起 8 缺口全闭·落地层 docs 自足到 build-ready·零 peek）。**★C1 头牌措辞 = [选项 A] 原主张成立 + clean-room 验证注记（三.3 预注册命中·不收窄）**：协议判断层可跟随性 + 落地层文档充分性经 clean-room 演练验证。**2 条诚实 Limitations**：(i) drill = docs-followability 判读（禁真 build）·先天偏置仍在（同栖仓库/规则约束盲/未真 build）·判读升级仅限「文档充分性」轴；(ii) **Tier-2（= [C1-4] 正面兑现达标线）= 非协议作者 clean-room agent 真接入到 build-绿 + falsifier 六门全过 + 触碰集符 [F-3]**（理想真外部第三方·退求非作者 in-house agent·偏离清单入 T1c）·**★Tier-2 达标（2026-07-12·commit `2fec2471`）= [C1-4] 正面兑现达标**：非协议作者 clean-room agent（盲·只读协议 docs+Template·禁读实现·forbidden peek=0）真接入 `weft_demo` 家族 → **build 绿**（weft-opt 258/258·0-error·zero core edit 除注册表）· **falsifier 六门 [F-1..F-6]+相邻门全 GREEN**（14/14·F-3 ratchet 40→46·zero orphan）· **触碰集符 [F-3]**·T1c `docs/reports/2026-07-12-P4-Tier-2-cleanroom-接入演练-T1c.md`。**诚实 caveat**：(a) clean-room 偏置（同栖仓库·非真第三方·Tier-3 optional）；(b) Demo=Template clone-adapt（证 docs+Template followable 到 build 绿+六门·**非**原创 from-scratch 家族设计）；(c) **★3 新落地真缺口**（GAP-A 重要:own-EmitC-backend 家族须第2注册点 BuiltinBackendEmitters·未在 recipe/allowances[修中]·GAP-B/C 次要）——Tier-2 暴露 Tier-1 docs-drill 未捕的真 build 落地缺口。**Tier-3（真外部第三方·optional·挂钩子不立任务）** = 消除偏置 (a) 的更强证据·渠道出现时重估。
