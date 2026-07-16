# Weft-RV 执行总纲 v2（代码锚定核查稿 · 内部）

> **文档性质。** 这是《科研目标总纲 v2.0》的**工程孪生**——总纲 [GOV-1] 里"现状与数字住 CI 报告 + 内部核查稿(file:line 锚点)"的那一份。**与送审的思想层总纲相反,本稿必须带 file:line 锚点**,含现值、六态、覆盖率、越界清单等一切时效内容。它按总纲 §9 协议产出:先确认基线 [B-1..B-8],再逐条款核查,附六态表 / 四覆盖率 / [S-5] 底账 / 三级归因 / [X-SCALAR] 落点 / [PERF-1] 八门 / LED 复算 / 越界清单 / 三张汇总表。
> **快照纪律([G-4]/[GOV-3])。** 本轮核查钉 **HEAD `7185a62b`**(2026-07-02;仓库实时在动,一切数字仅对此快照有效)。核查方法:6 簇并行 code-anchored 审计 + 多轮内联 grep/read 交叉验证。
> **口径([L-8]/[A-5]/[A-6]):** C_construct 只计**强义**(模式库原语构造),弱义单列 C_construct+;不得把弱义报为强义;数字仅 in-code 相对值(分母未钉,见 §3)。

---

## §0 基线确认 [B-1..B-8] @ HEAD 7185a62b(§9 [A-2] 第一步)

**7 条 CONFIRM,1 条(B-4)UPDATE。无失效冻结项。**

| 基线 | 裁决 | 锚点 / 证据 |
|---|---|---|
| **B-1** N-operand 产品-归约 byte-exact + offset-binary N=3 + LUT N=3;掩码/比较排除 | **CONFIRM** | e2e:`test/Target/RVV/{non-deferred-wide-product-reduce-dequantize-f32,packed-i4-offset-binary-dot-product-reduce,codebook-gather-dot-product-reduce}-front-door-export-e2e.mlir`;掩码族排除=文档边界非代码强制(`Route 7 masked_widening_dot_reduce` 标 "route-DATA ONLY",`lib/Plugin/RVV/Construction/RVVContractionRouteIdentity.cpp:319`) |
| **B-2** 块量化点积统一表驱动 + dispatch-wired | **CONFIRM(强化)** | `monolithicBlockDotOpTable()` 恰 24 项(15 SuperBlock+9 Flat),`include/Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h:1444-1657`;单一表驱动 pass `lib/Plugin/RVV/RVVMonolithicBlockDotSourceFrontDoor.cpp:685`(24 per-op front-door 已坍缩为 1,净 −17,587 LOC) |
| **B-3** 双发射路径共存 | **CONFIRM** | (a) 单体→平面发射器 `emitFlatBlockDot`(`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:5334`);(b) 分解 N-operand 构造路(`lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp` 等 4 个构造 front door) |
| **B-4** 平面 body 参数化完成;码本参数化未启动 | **⚠ UPDATE** | 前半成立。**后半部分失效:flat 16-entry 码本(iq4_nl+mxfp4)已并入 `emitFlatBlockDot`**(`hasCodebook`/`CodebookGatherNibble` 作描述符字段,`RVVToEmitCBlockQuantLinear.cpp:5306-5327`;`RVVToEmitCCodebookFp4.cpp:80,706`;commit e85d8bcd)。**改述为:"大网格/超块码本(2048-entry iq1/iq2/iq3、K-quant)参数化未启动;flat 16-entry 码本已折入平面发射器(弱义)"** |
| **B-5** 编译期选择归因富;调度/合法性阶段缺 | **CONFIRM** | in-IR 选择属性 `preference_score/rank/policy/explanation/tie_break`(`lib/Transforms/VariantSelection.cpp:326-358`;`include/Weft/Dialect/Exec/IR/ExecOps.td:261-266`);无 JSONL;调度阶段(为何选此 LMUL)、合法性阶段(为何拒)未落持久归因 |
| **B-6** 冷启动=每插件常量;两族不竞标 | **CONFIRM** | RVV `setScore(1.0)`(`lib/Plugin/RVV/RVVExtensionPlugin.cpp:645`);IME `setScore(20.0)`(`lib/Plugin/IME/IMEExtensionPlugin.cpp:581`);Scalar `1000.0`;排序纯按 explicitPreference→score→fallback→index→symbol(`lib/Plugin/ExtensionPlugin.cpp:1383-1403`),无能力先验;IME 仅对 mma/matmul 提议、RVV 无 typed IR 即 decline(`RVVExtensionPlugin.cpp:601-604`)→ 不 co-propose |
| **B-7** implies 一跳;闭包修复字节中性 | **CONFIRM(强化)** | `satisfiesID = id\|provides\|implies` 一跳直扫(`lib/Support/CapabilityModel.cpp:235-238`);**生产代码根本不声明 implies**(RVV 事实构造器 `/*implies=*/{}`,`lib/Plugin/RVV/RVVCapabilityProfile.cpp:57-58`);唯一非空 implies 在单测(1 跳,`test/Support/CapabilityModelTest.cpp:454`)→ 闭包修复平凡字节中性 |
| **B-8** 资源感知成本模型 + 真硬件探测未启动 | **CONFIRM(精化)** | `hwprobe`/`__riscv_hwprobe`/`instance-hash`/`shape-hash` 全 grep=0(lib/include/tools)。**易读作矛盾处**:`resource_cost_model`(`RVVGearboxSchedules.cpp:2096`)、`measured_ns`(`RVVScheduleMaterialization.h:211`)**存在但是静态 in-IR 串 / 属性,不是按 instance-hash 键控的测量库**;所谓"MEASUREMENT-backed autotuner"实为 tune-once→cache→read(`RVVGemmScheduleMaterialization.cpp:3-19`),非 live 库 |

---

## §1 逐条款核查(§9 [A-4] 顺序;每条 `现状(锚点) · 等级 · 动作标签 · 关联`)

### Falsifier 组 [F-*]

| 条款 | 现状(锚点) | 等级 | 标签 | 关联 |
|---|---|---|---|---|
> **★现状同步(2026-07-13·自决现状-维护·权威源 = `docs/method/FALSIFIER-INDEX.md` + `.github/workflows/falsifier-gate.yml`)**：**[F-1..F-6] 六门全进 CI(2026-07-12) = [P-3] falsifier 组达成**。本表旧「grep=0/缺失/无 CI/不可判定」快照 comprehensively stale，下同步为 landed；细节不在此复制(见 FALSIFIER-INDEX)。

| **F-1** 零分支 + regex-manifest + CI | ✅ **进 CI** `f1-zero-branch-gate`(`schema/family-regex.v1.json` manifest + `tools/lint/check_zero_core_family_branch.py` 判读规程·核心 family 分支 grep=0 across 69 core files·I3 holds) | 满足 | — | C1 |
| **F-2′** schema.def 操作门 | ✅ **进 CI** `schema-def-gate`(在位·常绿·`check_schema_gate.py` + 红队 `redteam_schema_gate.py`) | 满足 | — | C1 |
| **F-3** 变更收容 | ✅ **进 CI** `f3-family-locality`(用户裁「MLIR 分层为主 + family 清单机检·非物理目录搬迁」·`schema/family-manifest.v1.json` + `tools/lint/check_family_locality.py`·GREEN @ HEAD·242 family-root 归一 manifest·shrink-only ratchet 持·旧「gated on 目录归拢」WITHDRAWN) | 满足 | — | C1 |
| **F-4** 归因完备 | ✅ **进 CI** `f4-attribution-jsonl`(编译期归因 D-4① JSONL 出口在位 `lib/Transforms/VariantSelection.cpp:1094+` + 门·lit 双绿) | 满足(M1) | D-4② 装载期 = M2 增量 | C1 |
| **F-5** fail-closed 模糊 | ✅ **进 CI** `f5-failclosed-fuzz`(`check_f5_failclosed_fingerprint.py` + `schema/f5-failclosed-baseline.v1.json`·现 20/20·指纹不退化棘轮) | 满足 | — | C1 |
| **F-6** 独立性判据 | ✅ **进 CI** `f6-independence-gate`(机制已闭·独立门 `tools/lint/check_f6_scalar_family_independence.py` 在位·判据④ landed·闭包∩rvv.*=∅ 脚本化 + `only_feasible` 真实选中·[X-SCALAR] owned 内核已落地) | 满足 | XS-M2 `scalar.zfh`/XS-M4 = 家族完整性非 F-6 条件 | C1 |

### 能力 Schema [S-*]

| 条款 | 现状(锚点) | 等级 | 标签 | 关联 |
|---|---|---|---|---|
| **S-1** 结构化事实 + provenance/trust | `CapabilityDescriptor{id,kind,status,properties,relations}`(`include/Weft/Support/CapabilityModel.h:69-78`);`kind`=开放 StrAttr(`ExecOps.td:112`),`status`=闭合 I32Enum(`:42-58`);`params`=隐式 `map<string,string>` CSV(`CapabilityModel.cpp:82-95`);**provenance/trust grep=0(完全缺席)** | 部分 | 有界工作项 | C1 |
| **S-2** 关系语义 | conflicts 双向 fail-closed(`CapabilityModel.cpp:375-417`→`CheckCapabilityRequires.cpp:162-172`)、未知=假 已成立;**implies 传递闭包已建**(worklist BFS 到不动点 `computeImpliedClosure`+`satisfiesIDTransitively`,路由 5 个 set 级 satisfies 调用点,`95f1a482`;`CapabilityModelTest` 传递性+cycle-safety+zvfh 实例测) | 满足 | 已满足 | C1 |
| **S-3** 探针只写事实 | 探针层不存在(`hwprobe` grep=0);"探到X走Y"隐藏分支 grep=0(规则未违,机制未启动);build-time facts 适配器自述 "probes no hardware"(`include/…/RVVCapabilityProfile.h:61-75`) | 缺失 | 有界工作项 | C1 |
| **S-4** uarch 事实 / P6 | 粗粒度 `hart_count`(kind=uarch)真驱动选择(`RVVCapabilityProfile.cpp:490-494`→`HartParallelCapabilities.cpp:43-98`,走表);**每核 quirk 表(vrgather_slow…)grep=0**,P6 键控未建 | 部分 | 增量新建(quirk 表) | C3′ |
| **S-5** schema.def 声明工件 | 六项契约全散落 TableGen/C++/map,无哈希对象、无版本日志(全 grep=0)。六项反写见 §4 | 缺失 | 增量新建 | C1 |
| **S-6** 两级门 | 无 schema.def 可 diff、无排序 JSON→SHA256、无 RFC 日志、无 CI;依赖 S-5 先落盘 | 缺失 | 有界工作项 | C1 |
| **S-7** 成本住测量库 | (a) schema 本体不含成本(`CapabilityModel.h:69-78` 无 cost 成员)✓;(b) 按 instance-hash 键控测量库缺失(`resource_cost_model` 是静态 in-IR 串) | 部分 | 有界工作项 | C3′ |
| **S-8** 家族语义(能力门家族) | "家族=能力声明+所有权"机制在(各插件 own 能力 ID);[X-SCALAR] owned 内核**已落地**(Scalar 现 own tq2_0 vec_dot `f96f767a` + q4_0 dequant `2dd654d8`);能力门子事实(zbb/zvbb/`scalar.zfh`)仍 grep=0 未建(XS-M2) | 部分 | 有界工作项 | C2 |

### 插件 · 调度 · 归因 · 选择器 [P-*]/[D-*]/[SEL-*]

| 条款 | 现状(锚点) | 等级 | 标签 | 关联 |
|---|---|---|---|---|
| **P-1** 接口冻结 | `ExtensionPlugin` 稳定虚 C++ ABI(`include/Weft/Plugin/ExtensionPlugin.h:653-693`);可序列化签名入 schema.def⑤ 不存在;"改动需 RFC" 无 CI 强制 | 部分 | 有界工作项 | C1 |
| **P-2** 接入五件套 | 前四件有载体(getCapabilities/verifyVariantLegality/buildVariantEmissionPlan + lit);**第⑤件 ledger 条目无机制**(`ledger` grep=0) | 部分 | 增量新建 | C1 |
| **P-3** 验收=falsifier 全绿 | 无脚本、无 CI、无 manifest → 无法执行 | 缺失 | 增量新建 | C1 |
| **P-4** 外部可接入 M4 | 无 onboarding/plugin-guide/CONTRIBUTING(find=0);无实录 | 缺失 | 有界工作项 | C1 |
| **D-1** 编译期门自足 fail-closed | unavailable/conflicting 自足拒 + signalPassFailure(`CheckCapabilityRequires.cpp:47-63/153-172`);**但对未知符号 `continue` 静默(`:150-151`)**,真正拒未知落在选择期(`VariantSelection.cpp:184-188`)与提案校验(`ExtensionPlugin.cpp:1428-1433`)——三处 unknown 语义不一致(符号 vs ID),F-5 单跑该门会漏 | 部分 | 重构现有 | C1 |
| **D-2a** 装载期最小解析 | 编译期物化单个 `dispatch_available` ABI 参、调用方一次给(`DispatchRuntimeGuard.cpp:210-223`),热路径零逐次(NG-3 遵守);**declared-instance-hash grep=0、无解析记录落盘** | 部分 | 重构现有 | C1 |
| **D-2b** 完整运行期链 | 运行期 hwprobe grep=0;现有是 build-time 工具链探针,非运行期;无 instance-hash 键控 | 缺失 | 增量新建 | C1 |
| **D-3** instance-hash 调度表 | instance-hash grep=0;dispatch 表按 runtime-guard 布尔键控(`DispatchRuntimeGuard.cpp:225-248`);工具链 `sourceSHA256/binarySHA256` 是探针二进制哈希非能力实例哈希 | 缺失 | 增量新建 | C1 |
| **D-4** 三级归因 | ①富 in-IR 属性(`VariantSelection.cpp:326-358`)但非 JSONL、reason 值='variant-selected'/'fallback-coverage-missing'(`:36-40`)非 {only_feasible,static_order,prior,measured}、缺 candidates[]/keys_evaluated/declared_instance_hash/ts;②装载期缺;③运行期缺 | 部分 | 重构现有 | C1 |
| **SEL-1** 两段式先验层 | 合法性过滤(`analyzeRequirementLegality`)+ score argmin 有;**能力先验层(GEMM∧ime.present→矩阵范式)缺失**,现由每插件常量分代理 | 部分 | 增量新建 | C3′ |
| **SEL-2** 先于/同于 P7 硬时序 | **潜伏错选(定量):** P7 令两族对同一 GEMM co-propose 时,升序 RVV(1.0)<IME(20.0)→ 向量变体被选、矩阵范式静默落败无 error(`VariantSelection.cpp:649-650`)。P7 未落地→风险未触发但已定量可判(详见 §5) | 缺失 | 有界工作项 | C3′ |
| **SEL-3** 测量回填 + 外部 tuner 插点 | enumerate→dump→load→measured-best\|结构成本 fallback,无搜索(NG-1 合规);**记录键=kernel+march(非 instance-hash)**,且是 RVV 本地 LMUL 调优、非 exec 层跨族 memoized | 部分 | 重构现有 | C3′ |
| **SEL-3 键订正注**(〇.5 裁·2026-07-16 · PR-5 RESOLVED) | 上行旧措辞「记录键=kernel+march」**订正为对齐活 schema** `tiling-measurements.v1.json` / `measurement-memory.v1.json`:**记录键 = `declared_instance_hash`〔蕴含 march/vlen/vreg〕+ kernel + variant** 的复合主键(令〇.4 权威·instance-hash 展开自规范化能力事实集 [D-2a])。canon 条文本体不改措辞·仅加此订正注(硬冻结) | — | — | C3′ |

### 发射与构造 [K-*] / [L-8]

| 条款 | 现状(锚点) | 等级 | 标签 | 关联 |
|---|---|---|---|---|
| **K-0** 双路承认与收敛 | 双路真实(平面 `emitFlatBlockDot` `RVVToEmitCBlockQuantLinear.cpp:5334` + 分解路);终态"泛型层唯一权威"未达,平面路仍是块点积生产权威 | 部分 | 有界工作项 | C3′ |
| **K-1** Emission Plan 中心工件 | `VariantEmissionPlan` 字段丰富(`ExtensionPlugin.h:440-491`)、可 lit;**"可序列化"缺**(serialize/toJSON grep=0),非声明式输入配方 | 部分 | 重构现有 | 工程 |
| **K-2** Body 模式库 | 无模式库注册表(`PatternRegistry/patternLibrary` grep=0);模式原语 ad-hoc:分解路由 typed 原语构造(constructed),平面路 decode 是被描述符选择的手写 helper;无 (SEW,LMUL,VLEN) 统一参数化 | 部分 | 增量新建 | C3′ |
| **K-2b** 宽钳位/双区 body | 单区钳位 body 已建+lit(`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:1088-1154`)——但这是回归中**输 autovec 0.5× 的单scope losing 形**;**制胜的 widen-clamp-narrow 双区 body 未建** | 缺失 | 有界工作项 | C3′ |
| **K-3** N-operand 基线(带边界) | 三形状 e2e 字节精确、VLEN128/256 lit-locked;掩码族排除入边界 | 满足 | 已满足 | C1 |
| **K-3b** 泛型层唯一权威 + 掩码重发射 | 掩码族无经泛型层重发射路径;平面块点积权威未被泛型层接管 | 缺失 | 有界工作项 | C3′ |
| **K-4** 六态阶梯(可自动读出) | 无状态机:six-state/dispatch-wired 状态名核心代码 grep=0(仅在 spec 语境);六态是人工标签、无自动读出器;燃减清单无自动生成 | 缺失 | 增量新建 | 工程 |
| **K-5** 正确性门 | 整数字节精确达成(24 op e2e lit diff green);VLEN{128,256}×{m1,m2,m4} 部分(分解路证 VLEN128 m2/m4+VLEN256 m1/m2);objdump golden 仅 X60 2 内核;**ULP 上界声明缺;无 CI(`.github` grep=0)** | 部分 | 有界工作项 | 工程 |
| **K-6** 缺口关闭环 [GAP-1] | 环未机制化(triage/gap_id grep=0);首实例(钳位回归)已诊断为"缺 wide-clamp 能力",**但闭环第2-4步(命名→关闭→复测→引用 gap-ID)未走** | 缺失 | 有界工作项 | C3′ |
| **L-8** 弱/强构造纪律 | 判据清晰、文档遵守(descriptor-selected=弱 vs typed 原语=强);**无机制化护栏**(无 C_construct 计量器阻止弱充强),为 prose 纪律 | 部分 | 有界工作项 | C3′ |

> **[G3] 成熟 compiler 目标态 + 旁路清零(canon 一.2)**：成熟判据 = 强构造 `C_construct` ≥70%(向 90% 推)+ 旗舰格吞吐兑现 + **构造全走 front-door** + 发射权威唯一([K-3b])+ **旁路存量(直连发射器格数)→0**。直连发射器(`kBlockDotKernels` 里的 `emitRepackGem{v,m}<fmt>`,绕过 `typed_repack_gem{v,m}_loop_body` front-door)= **transitional scaffolding**:如实标 `dispatch-wired`(喂 `C_dispatch`,永不 `C_construct`),每格带 `retirement_batch` 逐格退役计划(仿 q4_0 已证 `typed_repack` 先例)。**`C_dispatch` 降为附注**,头条 = `C_construct`(强义)+ 旁路存量。机检护栏(L-8 的"机制化"缺口首次落地):`schema/emit-bypass-whitelist.v1.json`(shrink-only ratchet,`baseline_count==条目数`)+ `schema/coverage-sixstate.v1.json` 的 provenance 字段 + `tools/lint/check_frontdoor_provenance.py`([F-EMIT],CI job `frontdoor-provenance-gate`),对未白名单旁路 / 旁路充 constructed / 白名单陈旧 / ratchet 破坏一律 fail-closed。当前旁路存量 = 17 repack gemm_tile 格(仅 q4_0 decode/prefill 走 front-door constructed);终态 = 白名单清空。

### 模式注册表 · 成本账 · 覆盖率 [PAT-*]/[LED-*]/[COV-*]

| 条款 | 现状(锚点) | 等级 | 标签 | 关联 |
|---|---|---|---|---|
| **PAT-1** 注册表为数据 | 收缩族已数据化(`ContractionRouteIdentity` 7 路由 `RVVContractionRouteIdentity.cpp:33-377`);块点积接线层数据化(`monolithicBlockDotOpTable` 24 条目);**但无 `{pattern_id,requires,transform,mechanism,metrics_hook,status}` 统一 schema**(`metrics_hook` grep=0) | 部分 | 重构现有 | C3′ |
| **PAT-2** 初始 P1-P8 条目 | 无带 status 枚举的注册表对象;P1 已机制化(4 构造 front door)、P2 dispatch-wired、P2b 缺、P3/P4 未启动、P5 zvfh 被拒、P8 zvbb grep=0 | 缺失 | 增量新建 | C3′ |
| **PAT-3** 双板迁移判据进 CI | 无 `.github`;两板经 export-e2e lit 各跑,但无 "diff 注册表=0" 门、无 registry-diff 测试(grep=0) | 缺失 | 有界工作项 | 工程 |
| **LED-1** 每家族记录脚本 | 首点可复算:IME 4 目录 raw `wc-l`=2484(与 README 吻合);**但 [LED-1] 口径是 cloc≈1865(未装 cloc 近似);test_LOC≈659 未单列;无 git-历史生成脚本**(grep=0) | 部分 | 重构现有 | C2 |
| **LED-2** 边际成本曲线 | 仅 IME 一个真家族有数(需 ≥3);依赖 [X-SCALAR] 落地;无出图脚本 | 缺失 | 有界工作项 | C2 |
| **LED-3** 对照锚 | 均为 prose,无提交工件(Cambrian 定性锚在文字;ggml churn 量化砖 grep=0) | 缺失 | 有界工作项 | C2 |
| **COV-1** 覆盖分母 | 无 ggml commit 钉死、无提交的格式 roster;in-code 宇宙草稿=24 vec_dot(见 §3) | 缺失 | 增量新建 | 工程 |
| **COV-2** 四指标脚本 | 无自动脚本;六态机未自动读出;现值可手算(§3)未产品化 | 部分 | 增量新建 | C3′ |
| **COV-3** 目标值对表 | 志向值无自动测量对表;C_construct 强义≈0/24 远未及 M2≥40% | 缺失 | 有界工作项 | C3′ |
| **COV-4** 烧减曲线 | `handwritten_LOC(t)`/`C_construct(t)` 双曲线未进 CI;减量仅散在 commit 文字 | 缺失 | 有界工作项 | C3′ |
| **COV-5** 性能回归架 | 仅零散 ssh Win 记录(product-reduce-dequantize lamp kernel);无 instance-hash、无储库、无分相、无 CI | 缺失 | 有界工作项 | 工程 |

### 扩展 · 性能门 · 治理 · 非目标 [X-*]/[PERF-1]/[GOV-*]/[NG-*]

| 条款 | 现状(锚点) | 等级 | 标签 | 关联 |
|---|---|---|---|---|
| **X-ZVFH**(第一优先) | 事实**已注册**:`rvv.zvfh`(kind=isa-vector-fp16)从 probed ISA 证据 mint 一等事实 + implies 链 `rvv.zvfh⇒rvv.zvfhmin⇒rvv.zve32f`(`RVVCapabilityProfile.cpp:527-554`,token-boundary guard 防伪造 I5,`95f1a482`);[S-2] 传递闭包实体夹具已连(`CapabilityModelTest.cpp:697-736`)。注:`verify.mlir:58-59` 的裸 `@zvfh` 是 Exec verifier 对**未声明局部符号**的通用负控,非 rvv.zvfh 注册缺席证据(误读已更正)。**f16 路径 board 实测已落**(P5 functional·`abf57268`/`0219d816`·标量 `fcvt.s.h`+向量 `vfadd.vv` f16m1 双板正确);**剩:zvfh packager fact-keying**(`RVVTargetSupportBundle.cpp:2227` 硬编码 deployment-assumed·deliberate/parked·低紧急) | 部分 | 有界工作项 | C3′ |
| **X-SCALAR**(第二优先) | owned 内核**已落地**:主选三值 2-bit `tq2_0_q8_k_vec_dot`(byte-exact golden,`f96f767a`)+ 保底 `dequantize_row_q4_0`(`2dd654d8`) · 曳光弹 `compute_skeleton`(`5c010b2b`) · F-6 双断言机检 989-LOC gtest+lit;判据④ 真实选中(XS-M3)**已 landed**(`61f0c8fe`·N2 boundary PASS 完整·独立 F-6 门进 CI `bcd3c6bd`)·ledger(XS-M4)**已落**(1501 raw·LED-2 `c46678e1`);剩 `scalar.zfh` 事实(XS-M2·**DEFER**·标量族无 object-packager 可键控·裸注册 vestigial)未闭。数学素材在向量路亦存(见 §6) | 部分 | 有界工作项 | C1 |
| **X-ZVBB** | zvbb grep=0;无事实行/P8 条目/micro | 缺失 | 有界工作项 | C3′ |
| **X-AME** | 仅 Passes.td 注释 3 处;conflicts 含 AME-vs-IME 但 inert;条件项/论文不押注,缺失符合预期 | 缺失 | 有界工作项 | C1 |
| **uarch 表/P6** | 粗粒度事实在(X60 per-hart);每核 quirk 表(vrgather_slow)grep=0,"走表不走 if" 未起步 | 部分 | 增量新建 | C3′ |
| **PERF-1** 八门 | 0/8 整套齐全:②满足、①⑥部分、③④⑤⑦缺失、⑧越界(详见 §7) | 缺失 | 有界工作项 | C3′ |
| **C1-RW** 六对照面 | 4/6 成文(A Cambrian / B ACT / E FMV·IFUNC·hwprobe / F heteroMx,`literature/04-related-work-positioning.md`);**C CONVOLVE、D CUTEv2 缺(grep=0)、B 下 TAIDL 缺** | 部分 | 有界工作项 | C1 |
| **GOV-1** 单一权威 | 成立(总纲v2 规范源、本稿现状、无冲突副本) | 满足 | 已满足 | 工程 |
| **GOV-2** 任务口径统一 | 张力仍在(母任务标待审核、子任务多完成);无子项自动聚合脚本(grep=0) | 部分 | 有界工作项 | 工程 |
| **GOV-3** 快照纪律 | 成立(本轮钉 7185a62b) | 满足 | 已满足 | 工程 |
| **NG-1/2/3/5/6** | 未越界(无搜索调优/无图级框架/无 per-dispatch/无 Zk*/无 upstream 合入) | 满足 | 已满足 | 工程 |
| **NG-4** beat 措辞门 | **越界**:代码注释含未反汇编钉死、[PERF-1] 前的 beat 措辞——"beats ggml ~13%"(`RVVQ40ScheduleMaterialization.cpp:17`、`RVVGearboxSchedule.h:2209/2413`、`Passes.td:473`) | 越界 | 有界工作项 | 工程 |

---

## §2 六态阶梯全分母表 @ HEAD 7185a62b([K-4])

**分母 = 块点积 24 op(`RVVMonolithicBlockDotFamily.h:1444-1657`)+ 分解 N-operand 3 形状。** 每内核取最高态。

| 态 | 内核 | 计数 | 锚点/理由 |
|---|---|---|---|
| absent / emittable | — | 0 | 24 op 全至少 dispatch-wired |
| **dispatch-wired**(body 手写 ggml `_generic` 单体) | q2_K q3_K q4_K q5_K q6_K, iq4_xs iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s, tq1_0 tq2_0(15 超块)**+ nvfp4, q1_0**(2 flat 未进白名单) | **17** | 超块走 `SuperBlock` 表;nvfp4 自有单体(`RVVToEmitCCodebookFp4.cpp:761+`);q1_0/nvfp4 不在 `deriveFlatBlockDotDescriptor` 白名单 |
| **constructed-weak**(描述符选 decode/fold,算术=手写 helper 经 VerbatimOp) | q8_0 q4_0 q4_1 q5_0 q5_1(flat-plain)**+ iq4_nl, mxfp4**(flat-codebook,HEAD 新增) | **7** | `deriveFlatBlockDotDescriptor` 7-kind → `emitFlatBlockDot`(`RVVToEmitCBlockQuantLinear.cpp:5279-5334`);iq4_nl/mxfp4 用 `CodebookGatherNibble`+`hasCodebook` |
| **constructed**(typed rvv 原语,强义) | 分解 N-operand 3 形状:q4_0-nibble / offset-binary N=3 / codebook N=3 LUT | **3 shapes** | typed `WideningProductOp/StandaloneReduceOp/DequantizeOp` 组合(`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:1075-1085`) |
| **covered**(强义 + K-5 全绿 + CI) | 0(3 形状是 export-lit 级;product-reduce-dequant 在 X60 objdump-sealed;但无 CI、ULP 缺、objdump 未全板) | **0 严格** | K-5 全项未达(CI 缺) |

> q4_0 同时以 constructed-weak(平面路,生产权威)与 constructed(分解路,强义演示)存在——正是 [K-0] 待收敛的双路。

**单体 body 燃减清单(仍手写、待强义构造;自动生成器不存在):**
1. 超块/K-quant(15):q2_K q3_K q4_K q5_K q6_K iq4_xs iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s tq1_0 tq2_0(2048-entry 大网格 `vluxei16` verbatim)。
2. flat 剩余(2):nvfp4(`RVVToEmitCCodebookFp4.cpp:761-1081`)、q1_0。
3. flat 已半燃减但仍弱义(7):q4_0/q8_0/q4_1/q5_0/q5_1/iq4_nl/mxfp4 的 decode/fold 仍是 `emitFlatBlockDot` 内被描述符选择的手写 C helper。

---

## §3 四覆盖率现值(in-code 相对值;分母未钉,报告不 gate)

**分母 caveat:** [COV-1] 未定稿入仓(无 ggml commit 钉死、无 roster,grep=0)。用 in-code 宇宙=24 vec_dot 作分母草稿;ggml 全量格式更多,故一切 % 为 in-code 相对值。

| 指标 | 分子(六态口径) | 现值(/24 in-code) | 锚点 |
|---|---|---|---|
| **C_dispatch** | ≥dispatch-wired = 24 | **24/24 = 100%(in-code)** | `RVVMonolithicBlockDotFamily.h:1444`;`RVVMonolithicBlockDotSourceFrontDoor.cpp:685` |
| **C_construct(强义)** | =0(24 body 皆手写 `_generic`,非模式库原语) | **0/24 = 0%** | README §Track B "bodies still hand-written" |
| **C_construct+(弱义)** | constructed-weak = 7 | **7/24 ≈ 29%** | `RVVToEmitCBlockQuantLinear.cpp:5279-5334` |
| **C_attr** | ^CT 部分(仅选择阶段 in-IR);装载期/^RT=0 | **^CT 部分,^load=0,^RT=0** | 选择属性族(B-5);无 JSONL/instance-hash |

**强义 constructed 的真实所在地(独立 bucket,不并入 /24):** 分解产品-归约 3 形状(q4_0 nibble / offset-binary N=3 / codebook N=3 LUT),两板 VLEN128/256 export-lit 锁定。**这是 C_construct 燃减主指标的分子锚。**

**离目标:** M1 要 C_dispatch(A)=100%+全局≥80%(分母未钉不能认证全局)、C_attr^CT=100%(现仅选择阶段);M2 要 C_construct≥40% 强义(现 0/24,分解 3 形状另计)——**M2 主战场 = 把 constructed-weak(7)与 dispatch-wired(17)推向强义 constructed**。

---

## §4 [S-5] schema.def 六项内容反写(起草 v1 的底账)

**当前无一项被序列化(schema.def/shape-hash grep=0)。** 反写=把散落契约提取成"该入 shape 的形态":

| 项 | 当前形态(锚点) | 反写目标 |
|---|---|---|
| ① 事实字段+类型(含 provenance/trust) | `CapabilityDescriptor{id,kind,status,availability,properties,relations}`(`CapabilityModel.h:69-78`);`id`=自由点分串;`status`=闭合 I32Enum(`ExecOps.td:42-58`);**provenance/trust grep=0** | 新增 `provenance∈{hwprobe,cpuinfo,vendor_table,manual}`、`trust∈{measured,declared}` |
| ② kind 闭合枚举 | 开放 StrAttr(`ExecOps.td:112`),verifier 只查非空单行;取值散见 "uarch"(`RVVCapabilityProfile.cpp:492`)等 | 闭合 `{isa_ext,sub_ext,uarch,policy}` |
| ③ 关系类型表 | 扁平三 `OptionalArrayRefParameter<StringAttr>` provides/implies/conflicts(`ExecOps.td:73-76`),verifier 仅 hygiene,**无类型元数据** | 每关系型加语义标注(implies=传递可满足、conflicts=fail-closed 互斥) |
| ④ params 命名空间 | 隐式 `map<string,string>`(`CapabilityModel.h:74`),值 stringify(如 `supported_sew="8,16,32"`)**无命名空间/类型** | 命名空间化结构字段 `vlen/elen/sew_set/lmul_budget/vreg_count/cacheline/ime.tile(...)` |
| ⑤ 插件接口签名 | 纯虚 C++ vtable(`ExtensionPlugin.h:653-695`),**非声明式**、无可序列化投影 | 可序列化 ABI 签名(P-1 冻结,改动走 RFC) |
| ⑥ 操作数角色词表 | **混合**:N-operand 路已结构化(`RVVEmitCRouteFamilyDerivation.cpp:689,699`);块点积角色仍是表字符串(`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:819+`) | 统一操作数角色词表 |

**不入 shape(确认):** 具体事实行、params 取值、插件内部代码、测量库(`measured_ns`/`resource_cost_model` in-IR 串)、模式注册表条目。

---

## §5 三级归因现状 + SEL-2 时序风险

**三级归因:**
- **① 编译期选择(最成熟但形态不合规):** `addPreferenceMetadata` 物化 `origin/preference_score/rank/policy/explanation/tie_break/fallback_role`(`VariantSelection.cpp:326-358`)+ selected-marker `reason/selection_kind`(`:908-919`)+ RVV 路 `selection_reason/candidate_count/legal_candidate_count/selected_cost/measured_ns`。**差距:非 JSONL(grep=0);reason 值='variant-selected'/'fallback-coverage-missing' 非 {only_feasible,static_order,prior,measured};缺 candidates[]/keys_evaluated/declared_instance_hash/ts。** → D-4① = 由此重构出口(重构现有,原料齐)。
- **② 装载期解析记录:缺**(instance-hash/resolution-record grep=0)。
- **③ 运行期归因:缺**(无 hwprobe/instance-hash 调度表)。
- **阶段覆盖:** 选择阶段富(非 JSONL);调度阶段(为何此 LMUL)缺;合法性阶段仅 emitError(`CheckCapabilityRequires.cpp:120-129`)不落记录。

**SEL-2 潜伏错选(定量,P7 触发):** 今天两族不 co-propose(B-6),故常量分排序无害。**P7"矩阵范式接管 GEMM prefill"落地瞬间**——两族对同一 GEMM co-propose——`rankKernelVariantsByCost` 升序:`RVV 1.0 < IME 20.0 ⇒ 向量变体被选、矩阵范式静默落败无 error`(`ExtensionPlugin.cpp:1383-1403`+`VariantSelection.cpp:649-650`)。**这是 P7 落地瞬间被激活的潜伏错选**——[SEL-2] 硬时序(先验层必须先于/同于 P7)的根因;建议 M3 "P7 落地前 SEL-1 常青" 作门禁。

---

## §6 [X-SCALAR] owned 内核落点(数学可行性 + [F-6] 独立性走查)

**当前落点态(已落地):** 两候选 owned 内核**均已落地为净新增发射器**(非重构现有):主选三值 2-bit `vec_dot`(`f96f767a`)+ 保底 q4_0 `dequantize_row`(`2dd654d8`),前接曳光弹 `compute_skeleton`(`5c010b2b`,parse→verify→lower→byte-exact 纯标量 C 已贯通)。`ScalarExtensionPlugin` own `scalar.fallback`;F-6 双断言机检(989-LOC gtest + lit)已落。

**数学素材已在向量路存在(可移植为标量路):**
- 三值/2-bit 点积走 **base-3 trit 拆包 + 整数 MAC,非 XOR-popcount**(`RVVToEmitCTernaryBinary.cpp:1981` "(A) the BASE-3 trit unpack",`:1985` `xi-1` 三值映射);**全仓 `popcount` grep=0** → [J-3] "低比特不走 popcount" 代码级 CONFIRM。

| 候选 | 可行性 | 依据 | 判据① |
|---|---|---|---|
| **主选 = 三值 2-bit `vec_dot` 标量路** | **已落地** | 2-bit 位域抽取=base-I;`xi-1`=base-I 减法;整数点积=标量累加;标量路 byte-exact golden 已落(`f96f767a`;向量路等价体 `RVVScheduleDescriptorRegistry.cpp:266-283`) | A 类热内核(最强) |
| **保底 = q4_0 `dequantize_row` 标量路** | **已落地** | 逐元素 nibble 抽取,无归约,字节精确近乎 by-construction(`2dd654d8`) | 辅助算子(较弱) |

**[F-6] 独立性走查:** 每原语(2-bit 抽取 / `xi-1` / 整数 MAC)非向量 → 闭包 ∩ {rvv.*}=∅ 成立;判据④(向量缺席实例真实选中)**未验,落地必做**。

**⚠ per-block fp16 scale 折叠——三路,须显式声明(主选 + 保底都命中):**
- **zfh(标量半精)**:声明为 `scalar.zfh` 事实 → 闭包 ∩ {rvv.*}=∅,**独立性保持**;
- **zvfh(向量半精)**:会把 `rvv.*` 拉进闭包 → **破坏 [F-6],禁止**;
- **软件 fp16→fp32(纯 base-I)**:不需新能力,**最干净的独立性叙事**。
- 现仓 `scalar.zfh`(标量半精,独立性保持)**未作独立标量事实注册** → 仍是 [X-SCALAR] 的**前置声明工作项(XS-M2)**。(注:向量族 `rvv.zvfh` 已注册 `95f1a482`,但那会把 rvv.* 拉进闭包、故**不用于标量路**——见上「禁止」条;二者是不同 fact。)

---

## §7 [PERF-1] 八门逐门现状(击败唯一放行通道;0/8 整套齐全)

| 门 | 判据 | 现状(锚点) | 等级 |
|---|---|---|---|
| ① | 字节精确 | 整数路齐(README §28);**浮点路 ULP 上界声明缺**(ULP 仅注释 `RVVOps.td:7504`) | 部分 |
| ② | VLEN 翻转 lit(128/256) | 具备(`test/Target/RVV/*-export-e2e.mlir` 跑 128/256×{m1,m2,m4}) | **满足** |
| ③ | 双板各一次 objdump 验封 | **缺(单板)**:objdump 仅单板(`scripts/rvv_fair_three_way_measure.py:502`,ssh 'rvv') | 缺失 |
| ④ | micro **且** e2e(llama-bench 分相) | micro 有;**e2e llama-bench 仅归档无 harness**;prefill/decode 分相未接线 | 缺失 |
| ⑤ | 双板都验证 | **缺**:全 scripts 单一 ssh 'rvv'(`rvv_remote_probe.py:28`) | 缺失 |
| ⑥ | 实验纪律(固频/钉核/中位数+方差/同 commit ggml) | 部分:有 warmup/repeat/median;**无 taskset 钉核、无 governor 固频**;脚本自述 board 共享高方差 | 部分 |
| ⑦ | 机制合成归因(selector 日志证明获胜变体由能力键选出) | **缺(双重)**:(a) 无 JSONL;(b) 选择是 capability-blind 结构成本+烘焙测量,先验层不存在→即便有日志获胜变体也非能力键选出 | 缺失 |
| ⑧ | 措辞门(主张精确到相) | **越界(=NG-4)**:注释含未封 "beats ggml ~13%"(`RVVQ40ScheduleMaterialization.cpp:17`) | 越界 |

**结论:门②满足、①⑥部分,③④⑤⑦缺失、⑧越界。门⑦是机制性缺口(先验层未落地),非仅日志缺。**

**[P2c 证伪后新规 —— 2026-07-05;perf 测量前置门 + perf-thesis 再瞄准]:**
1. **preflight + 工具链政策(硬前置,凌驾门③⑥)**:任何双板 A/B perf 测量【之前】,`tools/e2e-harness/T3_step3/board_ab.sh` 的 fail-closed preflight **四门必须全绿**——① march 完整性(板 hwprobe/cpuinfo 实测扩展 ⊄ 编译 march 则 FAIL,尤其 zfh/zvfhmin)② 双侧 objdump libcall 扫描(任一侧含 `__extendhfsf2` 类软浮点 fp16 libcall 则 FAIL)③ 同 clang 版本 + 同旗标 ④ 环境指纹-T格坐标匹配。双板统一 **clang-20** + march **含板全能力**(见实验总纲v1 §1 第 9/10 条)。**动因**:P2c deferred "1.69× vs factory" 已被公平复测证伪 = **100% fp16-libcall 混淆**(对手 march 漏 zfh → scalar fp16 走 libcall → 残废对手);step-3 双板数(1.004×/1.019×)因双方均 libcall 压比值向 1 = **全 stale**(见 GAP-1/P2c、P2c-A、P2c-B)。
   > **★[CASE-COMPILER-ASYMMETRY] 落宪修订(2026-07-10)**:**preflight ③「same-compiler」原只接到 format-micro 线、漏接 repack-tiling/kernel-axis/gate4 的 vs-opponent 计时腿** → clang-ours-vs-gcc-shipped 不对称数(q4_K-S6 1.884× 等)畅通进 T8/8-gate,对称重测全蒸发(0.272/0.386/0.775/0.120×)。**修复(入宪、fail-closed)**:preflight ③ 提为**所有 vs-opponent 计时腿的前置**(含 gate4/tiling/kernel-axis);**且测前先查对手编译器身份**——对手 = board shipped `.so` 时,查其出货编译器(rvv=gcc-15 / k1=clang-18),要么两侧同工具链重建(如 batch2c gcc/gcc)、要么 REFUSE(如 GAP-SB)、要么明确归**系统账**(见下「双账本」块)。**奠基假设失效标记**:本条「双板统一 clang-20 + 厂商编译器(K1/Bianbu clang-18)仅记 footnote、不做第二套测量」的旧前提 **失效日期 2026-07-10 · 卷宗 [CASE-COMPILER-ASYMMETRY]** —— 被双账本取代:系统账中**出货工具链是一等测量维度**(k1 出货即 clang-18 → k1 as-shipped 测量本就对称,非 footnote;rvv 出货即 gcc-15 → 我方 clang 对拼 = 不对称,须归系统账并列最强基线)。
2. **perf-thesis 再瞄准(beat 的住址)**:**指令级高尔夫**(单核 vec_dot 逐指令抠 ns)= **已证薄边际**(P2c 双板 parity),**永久降级为 sanity 层活动**;beat 的真**住址 = 编译器够不着的层**——**repack/权重布局**、**同一二进制跨板装载期调度**、**e2e 分相**(prefill/decode)。[PERF-1] 八门的 beat 语境只对这些结构层开放;指令级 parity 是物理确认、不进 beat。

3. **[指令级 emit golf → sanity 层,宪法默认 —— P3 第三次 null 后钉死]:** 指令级 emit 微调经**三次 null**(P2c deferred-fold `physical` null / 2a vsetvli `machine-neutral` null / P3 register-resident `static≠runtime` regression)后,**默认降级为 sanity 层活动、不追**。**例外唯一放行 = item4 级【双证据】**:(a) 改动是真【C-结构杠杆】——语句顺序/落点这类 **clang 会尊重**的 C 结构选择,**不是** clang 会自己重推的低层旋钮(vsetvli 插入 / 寄存器分配,见①machine-neutral);**且** (b) **objdump 双板确认机器码真移位、非被 clang 抹平**(item4 fcvt.s.h 早发→后发是真移;2a 删 vsetvli 前后 byte-IDENTICAL 是被抹平)。缺任一证据 → 不改(无收益改动不留树上 = 正面纪律,F1/F2/P3-narrow 均已 revert)。**收官定性**(指令级 emit 轴):P2c null / 2a null / **item4 赢**(C-结构杠杆,k1/VLEN256 +4.37%,board-conditional)/ 2c q5_K board-split(deferred-reduce 微架构条件性 + 我方核比 ggml 出厂 riscv 核更忠于 ggml scalar 参考)/ P3 regression。四特性化 + 忠实性档案见 `docs/method/perf-characterizations-layer4.md`;两板条件赢台账见 `experiments/active/result-tables/T8_winloss_gap_ledger.csv`(q8_0-item4 / q5_K-2c,分板 `T3_A`/`T3_B`)。**★所有指令级赢 = KERNEL-micro-only**(phase=micro,e2e 大概率 wash,见 memory `kernel-wins-dont-transplant`),**不进 beat 语境**;beat 仍只对结构层(repack/装载期调度/e2e 分相)开放。

   > **★[CASE-COMPILER-ASYMMETRY] 性能宪章修订(2026-07-10 落宪)**:旧措辞「**指令微质量非赛场(同一 clang)**」**扩写为双账本口径**——
   > - **kernel 账内**:指令/codegen 微质量**非赛场**,因为**编译器对称强制**(两侧同一编译器);同一 clang 下 emit golf 三次 null 的结论不变。
   > - **系统账内**:**后端选择 = 系统身份**——我方 L3 后端 = **clang 为设计事实**(我们出货 clang 编的 `.o`)。系统账里「我方 clang-codegen vs 对手 gcc-codegen」**不是** kernel 微质量赛场,是**两套产品栈对拼**,须走系统账规则(各方自有完整栈 + 工具链披露 + 最强基线列)。
   > - **残留观察**:「**clang-17/18 RVV 后端于此类 mixed-SEW 全展开 kernel 上显著优于 gcc-15**」(rvv 承重 4 格对称 gcc/gcc 蒸发、clang-域逐格复现 = 直接证据;gcc 820/742 vs clang 71/3 spill)**限定 pattern-specific**,归**生态碎片化叙事素材**(RISC-V 工具链后端成熟度不齐),**非**我方 beat ggml 算法。**永不立项适配 gcc**(我方出货形态 = clang,gcc-codegen 差是 gcc 的问题、不是我们的目标)。

4. **★[CASE-COMPILER-ASYMMETRY] 双账本入宪(2026-07-10,用户全批;perf 主张的两把尺,每条 perf 主张必须标注所属账本):**
   - **kernel 账(源码质量主张)= 编译器对称强制。** 主张形态 = 「我方 kernel 源码 vs 对手 kernel 源码,**同一编译器同旗标编译**」。执法 = §7①preflight ③same-compiler,**扩到全部 perf harness 的 vs-opponent 计时腿(含 gate4 / repack-tiling / kernel-axis)**,fail-closed。对手 = board shipped `.so`(异编译器)时,要么两侧同工具链重建、要么 REFUSE、要么改判系统账。kernel 账内**指令/codegen 微质量非赛场**(对称下 emit golf 三次 null)。
   - **系统账(产品对拼)= 各方自有完整栈 + 工具链披露 + 最强基线列。** 主张形态 = 「我方产品(clang-codegen `.o`)vs 对手产品(as-shipped,其自有工具链)」。**必须披露双方工具链**(rvv 出货=gcc-15 / k1 出货=clang-18);**clang 重编 ggml 可行处必列为最强基线列**(否则拿对手 gcc-codegen 的弱当我方赢 = 系统账造假)。系统账里**后端选择 = 系统身份**(我方 L3=clang 为设计事实)。
   - **判别键 = 板出货编译器**:rvv=gcc-15、k1=clang-18。ours 恒 clang → k1 as-shipped 对拼**本就对称**(kernel 账合法 micro beat,如 q4_K-k1 3.106×/q5_K-k1 1.916×,NON-e2e);rvv as-shipped 对拼**不对称**(须归系统账并列 clang 重编最强基线;kernel 账下 rvv 承重 4 格对称重测蒸发 = 撤回)。**iq4_xs batch2b→2c(1.4556→0.7178)= 最早先例。**
   - **落地台账**:`experiments/active/result-tables/T-VALIDITY_compiler_symmetry_ledger.md`(Stage-2 banner + 四分类 + 6 承重格结局) · casefile `docs/reports/2026-07-10-CASE-COMPILER-ASYMMETRY-casefile.md` · T8 逐格 `★CASE-COMPILER-ASYMMETRY` 注 · q4_K 8-gate `docs/reports/2026-07-09-q4k-8gate-status.md`(Stage-2 reconcile)。
   - **★clang-18 统一 canon 注(〇.1 裁·2026-07-16 · PR-9 RESOLVED-BY-RULING)**:**rvv 板生态默认出货 gcc(=gcc-15),本文 campaign 部署账统一采 clang-18 保双板对称**(k1 出货即 clang-18);此 clang-18-symmetric 主表对称域是 perf 主表主数口径。**rvv 之 gcc 差异强制以 footnote 呈现**(rvv archive 8/8 LOSS·clang-micro≠deployment·gcc-death 见 [CASE-COMPILER-ASYMMETRY]),成色标 clang-micro 防误读。此注与上「双板统一 clang-20」奠基旧前提的失效标记(见 §7①落宪修订)一致——部署账口径以出货工具链为一等维度,campaign 主数取 clang-18 对称;domain-tagged 双列策略登记待裁。

5. **★部署 SOP 入宪(2026-07-10;出货正门 + 测量库维度):**
   - **出货形态 = clang `.o` 对象导出(正门)。** 我方 kernel 经 `--weft-rvv-lower-to-emitc | mlir-translate` → C → **clang 编 `.o`**,以对象形式链接/部署。这是我方 L3 后端的设计事实。
   - **禁 `.inc` 注入对手(gcc)构建链。** 把我方 `.inc`/源塞进 board 的 gcc 构建链再计时 = 制造 clang-ours-vs-gcc-shipped 不对称(=本案根因之一);属**禁行**。对手侧要计时,用其 as-shipped 二进制(系统账)或两侧同工具链重建(kernel 账)。
   - **选择器测量库键增加部署工具链维度。** `schema/tiling-measurements.v1.json` 的 `board_identity` 增 `toolchain`(ours-compiler / opp-shipped-compiler)维度;`vs_ggml_blockdot_ratio` 显式标注 opponent-compiler(rvv seed 行的 1.884/1.413/2.193 = clang-ours-vs-gcc-shipped artifact,对称重测 0.272/0.386/0.775)。**注**:selector 的**选择正确性不受影响**——SP4 tiled-vs-plain **键控于 register-cliff SHAPE + internal-A/B(均 ours-clang,compiler-symmetric)**,**从不**键控 `vs_ggml_blockdot_ratio`(那只是 seed 行的成色注记)。

---

## §8 LED 首点复算(IME family-local)

| 目录 | raw `wc-l` | 说明 |
|---|---|---|
| `lib/Dialect/IME` | 438 | IMEDialect.cpp |
| `lib/Plugin/IME` | 1588 | IMEBackendEmissionDriver.cpp(730)+IMEExtensionPlugin.cpp(858) |
| `include/…/Dialect/IME` | 366 | IMEOps.td(350)+IMEDialect.h(16) |
| `include/…/Plugin/IME` | 92 | IMEExtensionPlugin.h(65)+IMEBackendEmissionDriver.h(27) |
| **合计** | **2484** | 排除 5 个 CMakeLists |

**结论:2484 精确复算为 4 目录 raw `wc-l`(与 README 吻合)。但 [LED-1] 口径是 cloc(不含注释/空行/测试)≈1865;test_LOC≈659 应单列。README 的 2484 是 raw-wc 非 cloc,复算须双列。**

**★锚 drift 已查(先查后改·2026-07-11 用户裁·禁静默替换 2484):上表 = C2 founding-slice(@git `54465ee7`/2026-07-03,历史 slice 口径)。`family_ledger.py --family IME` 现报 raw `5153` = 同 7 文件从 founding-slice 长起(delta +2669)= C3′ 家族内扩展(更多 MMA op 变体+发射路),【非首接入成本、非计数口径差】(同工具同 raw-wc-l)。C2 边际曲线锚永远钉 founding-slice `2484`;live `5153` 是当前家族体积(informational)。ledger 脚本已建(`.trellis/scripts/family_ledger.py` + `tools/visibility/recompute_ledger_anchor.sh` 双数口径,`experiments/active/visibility/T2-ledger-anchor.md`)。**

---

## §9 越界清单(§9 [A-4] NG 扫描)

- **[NG-4] beat 措辞越界(立即整改):** 代码注释含未反汇编钉死、[PERF-1] 前的 "beats ggml ~13%"——`lib/Plugin/RVV/RVVQ40ScheduleMaterialization.cpp:17`、`include/Weft/Plugin/RVV/RVVGearboxSchedule.h:2209/2413`、`include/Weft/Transforms/Passes.td:473`。违 [L-1](注释同守)/[L-7](板级未封)。**动作:改为登记式 Win-B 措辞或删除,过 [PERF-1] 前不得留 beat 断言。**
- 其余 [NG-1/2/3/5/6] 未越界。
- **[GOV-2] 治理张力(非越界但需消除):** 母任务标待审核、子任务多完成;无自动聚合脚本。

---

## §10 三张汇总表(§9 [A-3])

### 按贡献

| 贡献 | 满足 | 部分 | 缺失/越界 | 关键缺口 |
|---|---|---|---|---|
| **C1** | K-3;S-2;(GOV) | S-1/D-1/D-2a/D-4/P-1/X-SCALAR/C1-RW | F-2′/F-6/P-3/P-4/D-2b/D-3/S-5/S-6/S-3 | schema.def 工件化 + 独立家族 owned 内核(已落地,剩 F-6 判据④) + 目录归拢 + 外部接入 |
| **C2** | — | LED-1/S-8 | LED-2/LED-3 | 自动 ledger + 第二家族(依赖 X-SCALAR)成曲线 |
| **C3′** | R-6(结构翻转) | K-0/K-2/L-8/SEL-1/SEL-3/PAT-1/S-4/S-7/COV-2/X-ZVFH | K-2b/K-3b/K-6/SEL-2/PAT-2/PAT-3/COV-3/COV-4/COV-5/X-ZVBB/PERF-1 | body 模式库(C_construct 主战场)+ 归因 JSONL + 先验层 + 覆盖率 CI |
| **工程** | GOV-1/GOV-3/NG-1235 6 | K-1/K-5/GOV-2 | F-3/K-4/F-1/COV-1/NG-4(越界) | 目录归拢 + 六态状态机 + CI + 越界清理 |

### 按等级

- **满足/已满足(~9):** K-3、S-2、R-6、GOV-1、GOV-3、NG-1/2/3/5/6、PERF-1 门②。
- **部分(~20):** F-1/F-4/F-5、S-1/S-4/S-7/S-8、P-1/P-2、D-1/D-2a/D-4、SEL-1/SEL-3、K-0/K-1/K-2/K-5/L-8、PAT-1、LED-1、COV-2、X-SCALAR/X-ZVFH/uarch、C1-RW、GOV-2。
- **缺失(~21):** F-2′/F-3/F-6、S-3/S-5/S-6、P-3/P-4、D-2b/D-3、SEL-2、K-2b/K-3b/K-4/K-6、PAT-2/PAT-3、LED-2/LED-3、COV-1/COV-3/COV-4/COV-5、X-ZVBB/X-AME/X-ZVQDOT、PERF-1(整体)。
- **越界(1):** NG-4(beat 措辞)。

### TOP-10 优先动作(证据线优先,便宜且解锁最多 → 引擎线主战场 → 硬件)

| # | 动作 | 关联条款 | 里程碑 | 标签 |
|---|---|---|---|---|
| 1 | **清除代码注释里的 beat 措辞**(越界,零成本立即) | NG-4/L-1/L-7 | 即刻 | 有界工作项 |
| 2 | **起草 schema.def v1**(§4 六项反写已备)+ 操作门 F-2′ | S-5/S-6/F-2′ | M1 | 增量新建 |
| 3 | **家族代码目录归拢**(重构窗口红利,现在便宜) | F-3 | M1 | 有界工作项 |
| 4 | **编译期归因→JSONL 出口**(reason 改三值枚举 + 补 candidates/keys/declared_instance_hash/ts)+ **装载期解析记录** | D-4①/D-2a/F-4 | M1 | 重构现有 |
| 5 | **编译期门未知即拒自足**(把 `continue` 改拒绝,统一 symbol/ID 语义) | D-1 | M1 | 重构现有 |
| 6 | **覆盖率脚本 + 分母定稿 + 六态状态机可自动读出** | COV-1/COV-2/K-4 | M1 | 增量新建 |
| 7 | **Falsifier 组进 CI**(curated regex + 判读规程 + fuzz + 独立性判据) | F-1/F-5/F-6 | M1 | 增量新建 |
| 8 | **SEL-1 能力先验层**(硬绑:先于/同于 P7) | SEL-1/SEL-2 | M2/M3 | 增量新建 |
| 9 | **body 模式库 Fork-B 续**(大网格码本参数化 + 把 weak/dispatch-wired 推向强义 constructed = C_construct 主战场) | K-2/K-0/K-3b | M2 | 增量新建 |
| 10 | **[K-2b] 宽钳位双区 body**(闭合钳位回归 [K-6/GAP-1])+ **[X-ZVFH] 接入**(含 [S-2] 闭包修复) | K-2b/K-6/X-ZVFH/S-2 | M2 | 有界工作项 |

---

## §11 与 spec/PRD 修订的衔接(下一步)

本稿是修订 `.trellis/spec/` 与母 PRD 的事实底座。修订要点(据本核查):
- **spec Novelty 表**:按总纲v2 的 C1/C2/C3′ 终态 + [L-6]/[L-8] 措辞宪法 + [K-4] 六态定义更新;把"能被调度路由 ≠ 主体由机制构造"写成硬不变量(现 spec 未区分弱/强构造)。
- **core-invariants**:补 [F-2′]/[F-6]/[NG-3 per-dispatch 永禁]/[SEL-2 先验层先于 P7] 为不变量;把 [NG-4] beat 措辞门写成注释级红线。
- **PRD(母重构任务)**:①三指标改挂 C_dispatch/C_construct强义/C_attr 三覆盖率(而非笼统 coverage);② pillar 增补证据线(schema.def/归因 JSONL/目录归拢/CI)为 M1 交付,与引擎线并列;③消除 [GOV-2] 母/子任务状态张力;④把本稿 §10 TOP-10 作为子任务分解草案的输入。
- **快照纪律**:spec 只写稳定契约(零现值);一切数字(六态/覆盖率/LED/Win)留本稿 + CI,携快照 ID。

---

## §12 三支柱协同 + 实验总纲接口(v1 补)

**三支柱定稿:** 科研目标总纲 v2(**证什么**·目标态·送审)· 本稿执行总纲 v2(**现在到哪**·代码锚定核查·钉快照)· 实验总纲 v1(**怎么证**·实验宪法 + 表集 T0–T8·空表规格)。三者 + dossier(研究档)共享条款编号与快照纪律。

**本稿 ↔ 实验总纲的接口:**
- 本稿 §2 六态表 / §3 四覆盖率现值 = 实验总纲 **T0 / T7** 在当前快照的**填充**(实验总纲定义空表结构,本稿填当前快照,论文引用填好的格)。
- 本稿 §10 TOP-10 证据线 = 实验总纲 §6 的**前置工具清单**(T-N 噪声地板 / 对手解析探针 / provenance 清单 / 归因 JSONL / schema.def / ledger 脚本 / coverage 脚本)——**一次建设、两处收益**:解锁实验 + 关闭 C1/C_attr/C2 的机检缺口。
- **[L-8]/[K-4] 的执法工具落地 = provenance 清单**(实验总纲 §1.6):每个机制构造 body 发射时写出模式原语 ID 列表 → 六态脚本判强/弱义、人不可辩解。本稿 §1 里 [L-8]"现为 prose 纪律"、[K-4]"六态无自动读出器"两条缺口,**由 provenance 清单闭合**——这是从"审计标签"升级到"机器可判"的具体机制。
- **IME(本稿现状 = e2e MEASURED-NULL / 不可隔离)与实验总纲 §3 一致**:IME 表钉死轨1(结构 + 硅上正确)+ T5d(方法学对照);轨2(自有机制构造 GEMM + 净范式消融)= M3+ 理想、gated,**不是轻活**(≥ block-dot body 构造),回退不改任何已发表主张。
- **★落表纪律(G8 用户裁·硬·2026-07-14):此后一切测量 / 解析结果只回填既有模板(T3 / T8 / T9 及四分账行)·禁在新位置另建账本 / 报告体系。** 新指标 = 既有表加列(如实验总纲 §2.1 的 T3 8 新列),不是新表;casefile / 反汇编存档可为过程工件,但**结论数只住 T3/T8/T9**。**T3 主表体例权威 = 实验总纲 §2.1**(clang-18 对称域 schema:四分账 `{matmul kernel-sym / forward-op / DEQ-AXIS / batch-regime}` + cold + 对手成色 + 对手符号/探针指针 + 编译器口径 + 键控状态列;**0.8 硬门分母 = matmul kernel-sym + forward-op**·DEQ-AXIS 照测不进头条分母·唯对手成色机判升门·batch-regime 封存)。本稿 §7 的双账本 / [CASE-COMPILER-ASYMMETRY] = T3 `compiler_axis` 列的背靠证据层(`T-VALIDITY*` 派生链·validity 轴·非 coverage 轴·零新账本)。

**⚠ 快照漂移(治理提示 [GOV-3]):** 本稿 §0–§11 的核查钉 `7185a62b`;仓库其后已前进(现 `6e2e4e56`),新增 **Fork C 超块构造判定 = CONSTRUCTIBLE 但 LOW-ROI(已 deprioritize,非边界)** 等。这**不改结论方向**(六态/覆盖率的定性判断不变),但下一轮核查须**重钉快照并重跑 §9 [A-2] 基线确认** —— 尤其复核 [B-4](码本/超块参数化)与六态表(super-block 强义构造的 constructible-but-low-ROI 定性)。一切数字随快照演进,携 ID。

> **文档族(三支柱定稿):** 规范源 = `Weft-RV_科研目标总纲v2.md`(思想层·可送审);工程核查稿 = 本稿(代码锚定·钉快照);实验设计 = `Weft-RV_实验总纲v1.md`(空表规格);研究档 = dossier(`README.md`/`notes/`/`literature/`/`data-sources/`)。AI 草案《实验方案与结果表集 v1》作实验总纲的详细表-schema 附录(在迭代文档夹)。四者共享条款编号与快照纪律,为完整重构做万全准备。
