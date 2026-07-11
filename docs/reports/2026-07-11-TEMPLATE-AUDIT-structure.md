# 模板结构审计一页表 + 清理清单 + [RENAME] 合并建议 — [TEMPLATE-AUDIT] 线 A

> 审计快照 HEAD = `b3e3fef4` · 生成 2026-07-11 · 触碰集 = docs 域(只读全仓 + 新建本报告)。
> **定位义务**:外来者按仓库结构 + README,能否 30 分钟内定位**模板五大件**(schema · 插件五件套 · 前门 · 选择器 · falsifier 组 · +测量库)?
> **只审计、不执行**:清理**只列清单不删**(删是审计交付后的独立机械步);本报告不改 code/schema/lit/ROADMAP/tools。

---

## 任务 A.1 — 顶层结构图 + 五大件定位度评分

### 顶层骨架(模板五大件在目录上的落点)

```text
/home/kingdom/phdworks/TianchenRV/
├── schema/                         ①模板 schema(能力/coverage/roster) ★清洁独立
│   ├── capability.schema.v1.json           能力 schema(声明 estimateVariantCost)
│   ├── coverage-sixstate.v1.json           六态(84 certified 源)
│   ├── coverage-roster.v1.json             roster(93 格分母)
│   ├── tiling-measurements.v1.json         ④选择器先验/测量库(double_ledger 字段)
│   ├── retired-index.generated.json        RETIRED-INDEX(去记忆化)
│   ├── pattern-registry.v1.json / cert-lineage.v1.json / *-whitelist.v1.json
│   └── (消费脚本却散在 .trellis/scripts + tools/visibility + tools/lint)
├── include/TianChenRV/
│   ├── Dialect/{RVV,IME,Scalar,Offload,Exec,Template,Toy,TensorExtLite}/IR/*.td   ②ODS(前件)
│   ├── Plugin/{RVV,IME,...}/*.h            ②插件接口头(estimateVariantCost 声明在 ExtensionPlugin.h)
│   └── Transforms/{Passes.td, VariantSelection.h}   ③前门 pass 定义 / ④通用选择器
├── lib/
│   ├── Dialect/{RVV,IME,...}/IR/           ②方言实现 + 部分③构造 op
│   ├── Plugin/                             ②插件五件套主体
│   │   ├── ExtensionPlugin.cpp             ④estimateVariantCost 基座 + 排名
│   │   ├── Construction/                   ③构造协议(通用)
│   │   ├── RVV/ (~36 文件)                 ②RVV 插件★爆炸展开 + ③*SourceFrontDoor.cpp + ④选择
│   │   ├── IME/ (3 文件)                   ②IME 插件(能力/构造内联在 ExtensionPlugin.cpp)
│   │   ├── Scalar/ (3) · Template/ (5·参考家族) · Offload · Toy · TensorExtLite
│   ├── Transforms/VariantSelection.cpp     ④通用能力键控选择 pass
│   └── Target/RVV/                         目标导出
├── tools/
│   ├── lint/  ★falsifier/卫生闸清洁家     ⑤check_construction_manifest_regex(F-1)/check_cert_requirements/
│   │          check_frontdoor_provenance/check_monolith_retire/gen+check_retired_index/check_index_consistency
│   ├── fuzz/f5_failclosed_fuzz.sh          ⑤F-5
│   ├── e2e-harness/  ★测量库清洁家(⑥)    board_ab.sh / board/*driver.c / run_e2e.sh / aggregate_e2e.py
│   └── visibility/ · bench/ · ci/
├── .github/workflows/falsifier-gate.yml    ⑤falsifier CI(F-2′/opponent-pin/monolith/frontdoor/cert/retired-index)
├── test/  ⑤falsifier MLIR 语料(散在 Conversion/RVV + Transforms/VariantSelection + Scripts/*fail-closed*)
├── experiments/  测量证据(active/sealed/archive/_templates + T 表)
├── docs/  ROADMAP + canon 三总纲 + method + reports(证据卷宗)
└── .trellis/  spec(稳定契约)+ scripts(coverage_metrics/family_ledger/schema_gate)+ tasks
```

### 五大件定位度评分(30 分钟外来者可定位度)

| 大件 | 主目录 | 代表文件 | 一眼可辨 | 评级 |
|---|---|---|---|---|
| **① schema** | `schema/`(顶层单目录) | `capability.schema.v1.json` · `coverage-sixstate.v1.json` · `coverage-roster.v1.json` | 顶层 `schema/` 一目了然;**唯一软肋 = 消费脚本三处散**(.trellis/scripts · tools/visibility · tools/lint) | **HIGH** |
| **⑥ 测量库** | `tools/e2e-harness/` | `T3_step3/board_ab.sh` · `board/kquant_gemm_paired_driver.c` · `run_e2e.sh` | 清洁独立家;双账本数据契约在 `schema/tiling-measurements.v1.json`(合理) | **HIGH** |
| **⑤ falsifier 组** | `tools/lint/` + `.github/workflows/falsifier-gate.yml` | `check_construction_manifest_regex.py`(F-1) · `f5_failclosed_fuzz.sh`(F-5) · `check_schema_gate.py`(F-2′) | 闸脚本清洁集中;**软肋 = 无 F-1..F-6→文件映射索引**,且测试语料散在 `test/` 三处 | **MEDIUM-HIGH** |
| **② 插件五件套** | `lib/Plugin/<家族>/` + `include/.../Plugin/<家族>/` + `lib/Dialect/<家族>/IR/` | 参考家族最清:`Template/{Ops.td, VariantLegality, EmitCRouteProvider, ExtensionPlugin, ConstructionProtocol}` | 家族目录命名清晰,但**"五件套"跨 3 根**(Dialect/X · Plugin/X · lib/Plugin/X)且**文件级不统一**:Template=5 清 / Scalar=3 / IME=~4 内联 / **RVV=~36 爆炸**;统一在**角色**而非文件 | **MEDIUM** |
| **④ 选择器(cost-model/先验)** | 二层散:`lib/Transforms/VariantSelection.cpp` + 各插件 `estimateVariantCost` + `RVVRepackTilingSelection.h` + `schema/tiling-measurements.v1.json` | `VariantSelection.cpp` · `include/.../Plugin/RVV/RVVRepackTilingSelection.h` · `ExtensionPlugin.h` | 无单一"selector/"目录;通用 pass + per-plugin cost + RVV-local tiling prior + JSON 先验四处;需读文档才知全貌 | **MEDIUM(偏低)** |
| **③ 前门(front-door 构造)** | 三层散:`lib/Plugin/RVV/RVV*SourceFrontDoor.cpp` + `RVVLowerQuantContraction.cpp` + `lib/Plugin/Construction/` + `lib/Dialect/RVV/IR/RVV*Construction.cpp` | `RVVMonolithicBlockDotSourceFrontDoor.cpp` · `RVVLowerQuantContraction.cpp` · `ConstructionProtocol.cpp` | **无单一 `FrontDoor/` 目录**;靠命名约定(`*SourceFrontDoor.cpp`/`*StreamFrontDoor.cpp`)识别,外来者最难 | **LOW-MEDIUM** |

### 定位度总判(诚实)

**30 分钟可定位 = 部分成立**。**清洁可辨 = ①schema · ⑥测量库**(顶层单目录,README「Repository layout」直接指到);**需约定知识/散挂 = ②插件五件套统一性(RVV 爆炸)· ③前门(无专属目录、靠命名)· ④选择器(四处二层)**。

**对"参考模板"定位的关键结构缺口 = 无"五大件 → 目录"顶层映射文档**。README 的「Repository layout」只列 include/lib/tools/test 泛目录,不告诉外来者"schema/插件/前门/选择器/falsifier/测量库分别在哪"。对一个头条 = 可复制性/可扩展性的**参考模板**,这张映射表的缺失是**头号定位债**(见 [RENAME] 合并建议 R1)。

---

## 任务 A.3 — 模块边界核查(5 新特性是否落对格位)

| 新特性 | 实际落点 | 应属格位 | 判 |
|---|---|---|---|
| **先验层 estimateVariantCost** | 接口 `include/.../Plugin/ExtensionPlugin.h` 声明 + 基座 `lib/Plugin/ExtensionPlugin.cpp` + 各插件 override(RVV/IME/Scalar/...)+ 消费 `lib/Transforms/VariantSelection.cpp` | ④选择器 | **✅ 在位**(接口清洁;impl per-plugin 是设计使然,非散挂) |
| **双账本 harness** | `schema/tiling-measurements.v1.json`(double_ledger)+ `tools/e2e-harness/T3_step3/board_ab.sh` + `board/*driver.c`;in-compiler 消费 `RVVRepackTilingSelection.h` | ⑥测量库 | **✅ 在位**(harness 清洁;唯"账本纪律"文档散在 schema + docs/reports casefile = 小瑕) |
| **certified checker check_construction_manifest** | `tools/lint/check_construction_manifest_regex.py`(读 committed HEAD 的 sixstate;消费 `e5_strong_readout.py` manifest) | ⑤falsifier F-1 | **✅ 在位**(清洁单文件) |
| **RETIRED-INDEX** | 索引 `schema/retired-index.generated.json` + 生成 `tools/lint/gen_retired_index.py`(.td prose + monolith-retire + emit-bypass whitelist 三合并)+ 校验 `tools/lint/check_retired_index.py` + CI `falsifier-gate.yml` | ①schema + ⑤falsifier CI | **✅ 在位**(按角色三处清洁分置:索引→schema、gen+check→tools/lint、CI→workflow) |
| **IME tile ops(vmadot / gemm_tile@ime)** | ODS `include/.../Dialect/IME/IR/IMEOps.td`(MMAOp/vmadot · Q40/Q80/Q4K MatMulTileOp)+ verifier `lib/Dialect/IME/IR/IMEDialect.cpp` + dispatch/emitter `lib/Plugin/IME/{IMEExtensionPlugin,IMEBackendEmissionDriver}.cpp` | ②插件五件套(IME 家族) | **✅ 在位**(Dialect/IME + Plugin/IME 清洁独立家) |

**边界核查结论:5/5 新特性全部在位,零散挂**。这是结构轴的正面发现——新增特性都被吸收进模板既有格位,没有"顺手挂在仓库根"的孤儿。唯二小瑕(非散挂,记账即可):(a) schema 消费脚本三处散;(b) 双账本纪律文档跨 schema+docs。均属"文档归拢"而非"错位",可随 [RENAME] 一并收。

---

## 任务 A.2 — 冗余清点 → 清理清单(只列不删 · 逐项复原指针 · 须过 RETIRED-INDEX 闸)

> **复原指针纪律(ROADMAP「退役与记忆两闸」)**:一切 retirement = git 历史可查 + retired-ledger 登记(格名/依据/替代路径/复原指针);删 sealed 证据 = 不可逆动作 = **必问用户**。下表凡删除类均须先过 **RETIRED-INDEX 闸**。当前 git ref = `b3e3fef4`(全项均可 `git show b3e3fef4:<path>` 复原)。

| # | 路径 | 为何废/问题 | 建议 | 复原指针 |
|---|---|---|---|---|
| C1 | `experiments/INDEX.md` | STALE vs 生成器(check_index_consistency #1) | **regen**(`tools/lint/gen_experiments_index.py`)· 机械、非删 | git `b3e3fef4` |
| C2 | `experiments/active/rvv-e2e-m1/token_tile_selection.md` · `experiments/active/vlen-adapt/vl16_static_account.md` | orphan durable:无 owning cell MANIFEST(hygiene #2)。**注:vl16_static_account.md 被 SEALED-WIN-REGISTRY 引为证据 → 严禁删**,须登记 | **注册进 MANIFEST**(非删) | git `b3e3fef4` |
| C3 | `experiments/active/cert-status/repack-probes/*.mlir`(28 个:14 格 × gemm/gevm) | 未注册 + "code leaked into data cell"(hygiene + layout 双报) | **git mv 到 tools/** 或注册进 cert-status MANIFEST(证据指针) | git `b3e3fef4` |
| C4 | `experiments/active/result-tables/{T-PERF1b_q4k_e2e_prefill_regression, T-VALIDITY-STAGE1_k1_symmetric_remeasure, T-VALIDITY-STAGE1_rvv_symmetric_remeasure, T-VALIDITY_compiler_symmetry_ledger}.md`(4 个) | 未注册进 result-tables MANIFEST(hygiene #4)。**均为 [CASE-COMPILER-ASYMMETRY] 承重证据 → 注册非删** | **注册进 MANIFEST** | git `b3e3fef4` |
| C5 | `experiments/archive/perf-historical/`(ondevice-q8_0* · ondevice-q5_K · T3_step3;**16 个 tracked .o 二进制** + pre-swap/撤回数) | STALE 历史 perf 数(换板前不可比)+ 二进制入库膨胀 | **保留归档**(已 `MOVES.md` 登记),但确认覆盖于 RETIRED-INDEX/MOVES ledger;**.o 二进制** 评估移出版本控制 | git `b3e3fef4` + `MOVES.md` |
| C6 | `experiments/active/result-tables/T3_A_*.csv` 内的 stale/INVALID/撤回行(1.884×/1.413×/2.193× 等 compiler-asymmetry) | 已就地 inline 隔离(标 INVALID/WITHDRAWN) | **原地保留隔离**(= [CASE-COMPILER-ASYMMETRY] 证据);仅确认隔离标记完整,**不删** | git `b3e3fef4` |
| C7 | `artifacts/{grill-consensus-20260515, grill-rvv-maturity-ladder-2026051{8,9}, tianchenrv_rvv_gearbox_autotuning_pass_v3, trellis_spec_audit_prompt}.md` | 旧 scratch/prompt(2026-05,pre-refactor)散在 artifacts 根 | **归档或移出**(过 RETIRED-INDEX 闸;非证据) | git `b3e3fef4` |
| C8 | `.touch-set/{line-C-iq2s, line-D-f5, line-xscalar-f6, line-xscalar}.txt`(6 tracked) | per-line 触碰集 scratch,对应线已落地后 stale | **归档**(保留 `_example.txt` + `README.md`) | git `b3e3fef4` |
| C9 | `docs/reports/2026-07-06-TCRV_IR三层与编译器身份_parallel-writer-leftover.md` | 并行写手遗留(文件名自陈 leftover) | **内容并入正规 doc 或移除** | git `b3e3fef4` |
| C10 | `.worktrees/cache/`(14 缓存 worktree) | gitignored(未跟踪)· 磁盘杂物 | **`git worktree prune` / rm**(可再生,无需复原指针) | N/A(gitignored) |
| C11 | `scripts/`(rvv_fair_three_way_measure / rvv_accumulator_sweep_measure / rvv_generated_bundle_* / tensorextlite_runtime_abi_e2e / codex_serial_supervisor{.py,_prompt.md}) | 与 `tools/e2e-harness` 测量库职能重叠;`codex_serial_supervisor*` 疑编排 scratch | **评估合并**测量脚本入 tools/ 测量库家;codex_serial_supervisor 查 staleness | git `b3e3fef4` |

**清理清单量级**:check_index_consistency 报 **35 issue**(C1×1 + C2×2 + C3×28 + C4×4);另 C5–C11 为审计新增的结构冗余(scratch/二进制/重叠脚本)。**全部只列清单,未删任何文件**。

---

## 任务 A.4 — [RENAME] 合并建议(与改名会话一次做 · 避免两遍搬家)

> 触发 = ROADMAP 排队 #6「[TEMPLATE-AUDIT + RENAME] 合并会话」;下列均"文件移动/命名层"动作,与改名同批做省一次搬家。凡触碰 code/schema 归主会话/构造线,**与构造互斥**(ROADMAP 并行拓扑)。

| R# | 建议 | 归并理由 | 触碰域 |
|---|---|---|---|
| **R1** ★头号 | **新增"五大件 → 目录"顶层映射**(README 章节或 `docs/TEMPLATE-MAP.md`):schema/插件五件套/前门/选择器/falsifier/测量库各指到具体目录 + 参考家族(Template)= 五件套标准范本 | 补最大定位债;"参考模板"的可复制性头条需要这张图;改名会同时动命名层,一次落 | docs(可先行) |
| **R2** | **RVV 插件按件分子目录**:`lib/Plugin/RVV/` ~36 文件按格位分组(`FrontDoor/`·`BodyRealization/`·`Schedule/`·`Selection/`),对齐 Template 参考家族的五件套 | 改名 = 文件移动,与分子目录同为 mv;两遍搬家浪费 | code(与构造互斥·主会话) |
| **R3** | **前门统一命名/目录约定**:给 `*SourceFrontDoor.cpp`/`*StreamFrontDoor.cpp` 一个共享 `FrontDoor/` 归属,消除"靠命名识别" | ③前门是最低定位度件;与 R2 同批 mv | code(主会话) |
| **R4** | **falsifier F-1..F-6 → 文件映射索引**(`tools/lint/README.md` 或 falsifier-gate.yml 顶注):列明哪个脚本/测试是哪个 F | ⑤软肋轻补;文档级,可随改名落 | docs/tools(轻) |
| **R5** | **选择器四件co-locate 文档**:VariantSelection + estimateVariantCost + RVVRepackTilingSelection + tiling-measurements.v1.json 在一处"selection"标题下互指 | ④二层散;文档归拢,不必物理搬 | docs |
| **R6** | **MANIFEST 注册批**:C2 orphan(2)+ C3 cert-probes(28)+ C4 result-tables(4)一次注册/归位 | 都动 experiments/ layout,与改名同批省一次 | experiments(数据域·可先行) |
| **R7** | **scratch 归档批**:C7 artifacts scratch + C8 .touch-set + C9 parallel-writer-leftover + C10 worktree prune 一次清 | housekeeping 攒批;过 RETIRED-INDEX 闸 | docs/artifacts/根(可先行) |

**先行/互斥标注**:R1/R4/R5/R6/R7 = docs/data 域,可**与构造并行先行**;R2/R3 = 动 lib/ code,**与构造互斥**,须排 G4 M1 贯通后(ROADMAP 并行拓扑:[RENAME] 排 M1 贯通后)。

---

## 产出确认

- 结构审计一页表(顶层结构图 + 五大件定位度评分 + 模块边界核查)见任务 A.1 / A.3。
- 清理清单(C1–C11 · 逐项复原指针 git `b3e3fef4` · 只列不删)见 A.2;[RENAME] 合并建议(R1–R7)见 A.4。
- **定位度总判**:①schema · ⑥测量库 = HIGH(清洁独立);⑤falsifier = MEDIUM-HIGH;②插件套 · ④选择器 = MEDIUM;③前门 = LOW-MEDIUM。**头号债 = 缺"五大件→目录"顶层映射(R1)**。
- **模块边界**:5/5 新特性全在位、零散挂。
- HEAD = `b3e3fef4` 未变;未改任何 code/schema/lit/ROADMAP/tools(本报告纯新增 docs)。
