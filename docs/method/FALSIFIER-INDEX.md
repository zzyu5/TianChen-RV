# FALSIFIER-INDEX — [F-1..F-6] → 具体 checker / lit / gtest 映射（R4）

> **用途**：模板五大件之 ⑤falsifier 组的**软肋 = 无 F→文件映射索引**（[TEMPLATE-AUDIT] 认定）。
> 本 doc 补这张表：canon 的 [F-1..F-6] 每一个 → 它的**权威定义位置** + **具体机检工件**（脚本 / lit / gtest / CI job）
> + **诚实机制化状态**。配套 [REPOSITORY-MAP-五大件.md](./REPOSITORY-MAP-五大件.md) §1.⑤。
>
> **[F-1..F-6] 定义权威 = `docs/canon/TianChen-RV_科研目标总纲v2.md:105–110`**（不在此重抄语义，只做文件映射）；
> 现状栏对齐 `docs/canon/TianChen-RV_执行总纲v2.md:32–37`。**本 doc 只索引，不改 code / 不改判据。**

---

## 1. [F-1..F-6] 主映射表

| F | 一句话判据 | 定义（权威） | 具体机检工件 | CI | 机制化状态（诚实） |
|---|---|---|---|---|---|
| **[F-1]** | 零核心分支（family-branch grep=0）+ family-regex manifest + 判读规程 | `科研目标总纲v2:105` · `.trellis/spec/testing/mlir-testing-contract.md:72`（[F-1] 判读规程 + manifest 形态） | **⚠ 见 §2 命名碰撞**：零分支 grep 的 family-regex manifest **尚未落盘**（执行总纲 F-1 = 「部分」，缺 manifest/CI/判读规程）。另有一个**同名不同物**的工具 `tools/lint/check_construction_manifest_regex.py`（自标 "F-1 construction-manifest shape gate"，实为 C1 合取的**构造体 SHAPE 正则门**，非零分支门） | 无（零分支门）/ CI 外（shape 门可跑） | **零分支门：部分**（不变量本身绿=core family 分支 grep=0，但 manifest/CI/判读规程缺）；shape 门：**在位可跑** |
| **[F-2′]** | 家族接入 PR diff ∩ schema.def = ∅（操作门）+ 报告门（shape-hash + 版本日志） | `科研目标总纲v2:106` · `core-invariants.md:80` | `.trellis/scripts/check_schema_gate.py`（self-test / `report --check` / `gate --base --head`）+ 红队 `.trellis/scripts/redteam_schema_gate.py` | **✅ `falsifier-gate.yml` job `schema-def-gate`** | **在位 · CI 常绿 · 带红队证伪** |
| **[F-3]** | 变更收容：接入 PR 仅触 `plugins/<family>/` + 表行 + docs | `科研目标总纲v2:107` · `.trellis/spec/plugin-protocol/locality-contract.md:76` | **无专属脚本**（结构原则，非机检门）；**前提 = 家族代码目录归拢**（当前单家族横跨 `lib/{Dialect,Plugin,Conversion,Target}/<fam>` → 不可判定） | 无 | **缺失 · 有界工作项**（gated on R2/R3 目录归拢；归拢后才可判定） |
| **[F-4]** | 归因完备（[D-4] 分级：M1 查①②，M2+ 查③）；`only_feasible`/`static_order`/`prior`/`measured` 归因出口 | `科研目标总纲v2:108` | lit `test/Transforms/VariantSelection/attribution-jsonl.mlir`（归因 JSONL 出口 lit）；in-IR 归因属性在编译期选择阶段（D-4① 原料） | 无 | **部分**（编译期选择阶段富属性在位；JSONL 出口 + 调度/合法性阶段归因缺；不进 CI） |
| **[F-5]** | fail-closed 模糊：随机删/伪造事实下非法 plan 全被拒（非误编译/误发射） | `科研目标总纲v2:109` | `tools/fuzz/f5_failclosed_fuzz.sh`（变异 valid typed-region → `tcrv-opt --tcrv-rvv-lower-to-emitc`，断言 graceful-diagnostic + 非零退出 + 不崩 + 不静默过）→ 产出 `experiments/active/result-tables/T1b_failclosed_runtime.csv`；lit 语料 `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-*-fail-closed.test`（14 个） | 未进 CI（fuzz 可重跑，确定性指纹） | **在位可跑**（2026-07-07 发现 17/20 fail-closed，3 fail-OPEN = neg_qk/neg_weight_stride/neg_activ_stride，已记 T1b 不藏；未进 CI） |
| **[F-6]** | 独立家族判据：变体能力谓词 implies 闭包 ∩ {rvv.*} = ∅（脚本化）+ 存在向量缺席实例使其变体 `only_feasible` 且真实被选中 | `科研目标总纲v2:110` · `core-invariants.md:84` | lit `test/Transforms/VariantSelection/f6-independent-scalar-family-emittable.mlir` + gtest `test/Plugin/ScalarExtensionPluginTest.cpp`（F-6 双断言，989 LOC）+ **独立门 `tools/lint/check_f6_scalar_family_independence.py`**（闭包∩rvv.*=∅ 脚本化 + `only_feasible` 真实选中跑活验证·`--self-test` 判别 GREEN）；受测家族 [X-SCALAR] owned 内核已落地（tq2_0 `f96f767a` / q4_0 `2dd654d8` / 曳光弹 `5c010b2b`）· 判据④机制 landed（VariantSelection:165 只 feasible 谓词·reason=only_feasible）| **f6-independence-gate job**（`falsifier-gate.yml`·self-test + build-free 默认门 61f0c8fe/XS-M3） | **机制已闭 + 独立门在位**（判据④ landed·N2 boundary PASS 完整；仅剩 XS-M2 `scalar.zfh`/XS-M4 LED-2=家族完整性非 F-6 条件） |

**[P-3] 接入验收 = falsifier 组全绿**（`科研目标总纲v2:71`：[F-1..F-6]，独立家族含 F-6）。
**M1 = 证据线闭合**目标（`科研目标总纲v2:211`）= F-1 / F-2′ / F-5 / F-6 进 CI。**当前只有 F-2′ 真进 CI**；
F-5 可跑未进 CI；F-1 零分支门 manifest 缺；F-3 gated on 目录归拢；**F-6 已进 CI**（`f6-independence-gate` job·61f0c8fe/XS-M3·判据④机制 landed + 独立门·N2 boundary PASS 完整）。**当前 F-2′ + F-6 真进 [F-1..F-6] CI**。

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

- **进 CI 的 falsifier**：**F-2′ + F-6**（[F-1..F-6] 真进 CI 的两个）+ [F-EMIT] / opponent-pin / monolith-retire / cert-三要件 / RETIRED-INDEX。
- **可跑未进 CI**：F-5（fuzz，17/20，3 fail-OPEN 已记 T1b）。
- **部分**：[C1-SHAPE] shape 门（可跑·非 CI）/ F-4（编译期富属性在位、JSONL 出口缺）/ F-6（家族 owned 内核 + lit + 989-LOC gtest 在位；闭包脚本 + `only_feasible` 真实选中缺）。
- **缺失 / gated**：F-1 零分支 manifest（缺 manifest+CI+判读规程）· **F-3（★R2/R3 已 landed 2026-07-12·cbe21c3e·Plugin/RVV 内部 4 桶可判定；剩跨 4 lib 根收拢裁定=必要不充分）** · F-6 闭包脚本（家族 owned 内核已落地；剩闭包脚本 + `only_feasible` 真实选中 = XS-M3）。
- **★[C1-SHAPE] 门 finding（2026-07-12·非 reorg 引入·pre-existing·非 CI 非阻塞）**：`check_construction_manifest_regex.py` 全跑报 RED 3/84 = `gemm_tile/{q4_0,q8_0,q4_K}` "[IME-SEAL] envelope mismatch (not the IME matmul-tile seal form)"。判 pre-existing（reorg behavior-preserving·lit 同 904/907·tcrv-opt 输出不变→门验不变）。疑似 G4 IME-SEAL 期门 quirk（RVV gemm_tile 被拿 IME form 检）或真 manifest 议题·**待 later triage**（非本会话 code 引入）。
