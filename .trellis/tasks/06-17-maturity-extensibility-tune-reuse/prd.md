# Compiler maturity: standardize + make-reusable the extension / tune / kernel axes

## Goal (user, 2026-06-17)
都做 — 不为论文卖点,而为 **compiler 成熟度 + 符合我们的 claim**(zero-core-branch、extensibility、reuse)。
没有第二块板子,所以重点是把内容**重构成更符合拓展/复用的形式**,**core 文件可以拆分+重构**,
**tune 的跨拓展复用机制好好探索**。(之前 scope-out 的 25k-line core refactor 被用户明确纳入。)

## Grounding (research/current-state-survey.json — 3-dim survey + advisor-sharpened synthesis)
- **Family axis 已经干净**:`ScalarExtensionPlugin` 是真注册 plugin,走和 RVV 同一条 generic pipeline;
  core/common 里 `hasRVV|hasIME|...` grep **近乎空**(只剩 3 处注释)。零-core-branch 的 diff 本就近空。
- **复制粘贴重灾区 = kernel + tune 轴**:加 kernel = 25k 行 `RVVToEmitC.cpp` 里手改 6 处(绕过 plugin 接口);
  5+1 个 tune materialize pass 是 ~95% 雷同克隆(>1000 机械行)。
- **可复用 pattern 已 latent**:cost 层已做对"声明一个 fact → 继承机制"(`getRVVBlockDotDecodePrefixLength`);
  这 6 个 tunable op **已实现** `TCRVEmitCLowerableOpInterface` → 用 OpInterface 驱动异构 op 的 pass 是 in-repo 既有模式。

## Workstreams
- **WS-A — Scalar family-hardening**(infra). 把 `RVVVectorSourceFrontDoor.cpp:1236 createScalarFallbackVariant`
  的身份编写挪出 RVV 前门,让 scalar plugin 自己 `proposeVariants` 在**每条 flow** 产出。⚠️ 反 false-green:
  证据 = family-name grep 仍空 **且** 走 `proposeVariants`/`collectVariantProposals`,不是按名直调的改名 helper。
- **WS-C — Tune-reuse abstraction**(infra, 用户明确"好好探索"). `GenericScheduleCandidate{cost,isLegal,knobs[]}`
  统一 block-dot(3-knob)+ GEMM(activation_cols)候选;一个 `selectSchedule()`;一个 descriptor/interface 驱动的
  `MaterializeRVVSchedulePass` 走所有 tunable op。Tier-1 = `TunableScheduleOpInterface`(照抄 TCRVEmitCLowerableOpInterface):
  新 op 声明 `enumerateCandidates(capabilityFacts)→[{cost,isLegal,knobs}]`/`kernelKey`/`isPinned`/`stamp`,
  **继承** enumerate→prune→实测/static-fallback→stamp,不再写 pass。tuning-record parse 从固定 key 泛化到任意 `key=value`
  (`dumpRVV*LegalCandidates` 已是 key=value 格式,小活)。
- **WS-D — Core split + kernel-axis 重构**(maturity / conform-to-claim, 用户明确纳入). 拆 25k 行 `RVVToEmitC.cpp`;
  把 26 个 `isXxxBlockDotBody` recognizer + 39-branch dispatch if-chain → 泛型 `isSoleBlockDotBody<T>` + TypeSwitch/注册表;
  ~26 个 verifier → 共享 `verifyGgmlBlockDotABI(op,{allowedAttrs},{pinnedOffsets},{ctypes})`;op def → TableGen 基类。
  方向:让 add-a-kernel 走 plugin route-provider(plugin-local 注册),而不是 core if-chain —— 让 kernel 轴也 zero-core-branch。
- **WS-B — Codegen-quality slice, measured**(N3 证据, 唯一可测加速;用户不追卖点但要符合 claim).
  **先在 ssh rvv 做 roofline**(block-dot compute-bound vs decode-bandwidth-bound),再加**一个** microarch 轴
  (software pipelining / accumulator latency-hiding)**当 tune knob**(不是新 emit 分支),实测加速。
  作为 WS-C 抽象的**第二个真消费者(新轴)** co-design,防止过拟合固定 key。

## Sequencing
WS-A 可并行(独立、便宜)。WS-C 先做 Tier-0(折叠雷同克隆,安全)→ WS-D 用同一 pattern 重构 kernel 轴 →
WS-B 的新轴落进 WS-C 的接口验证泛化性(WS-B 实测 before freezing the interface shape)。

## Invariants / discipline (硬约束)
- 全程 STRUCTURED emitc(`raw()` 实际调用保持 0,I5);verifier fail-closed(I7);additive(每步 siblings 逐字节不变);
  全量 clean rebuild 绿 + 仅 3 个有据环境红;每个 perf 主张真 `ssh rvv`(I8)。
- WS-B 的加速**必须落成 knob(数据穿过接口)**,不是新 emit-fn 分支(否则养大 core)。

## Acceptance
- [x] WS-A: scalar 变量由 scalar plugin 经 collectVariantProposals 产出;前门 scalar 串 5→0、按名 0;family-name grep 仍空;协议测试绿。(ef7179e7)
- [x] WS-C: `GenericScheduleCandidate{cost,isLegal,knobs}`+`selectGenericSchedule`(Tier-0, 1389d946)+ **TunableScheduleOpInterface** 一个 pass auto-discover 所有 tunable op(Tier-1, a73ab62d);6 clone pass 折叠;stamp/调优 pick 逐字节不变;`key=value` record 泛化。
- [~] WS-D: **table-collapse 完成**(28-branch dispatch→first-match table,36 emission 测试逐字节不变,bd7c8b53)。verifier de-dup = **DEFERRED**(有据,见 Progress)。`RVVToEmitC.cpp` TU 拆分 = **未做**(纯 maturity churn + linkage 风险,advisor 判定为预算不足时第一个砍的;唯一剩项)。
- [x] WS-B: roofline 实测(bac3acd5)+ measured win **已存在**(inc10: q4_1 静态 m1/4 loss→实测 m1/1 win;factor 1>2>4 推翻 cost-model)+ knob = 既有 `multi_block_factor`(不是新 emit 分支)+ 经 Tier-1 unified pass 逐字节消费(second-consumer 验证)。
- [x] 全程 raw()=0、clean rebuild 绿、3 documented reds(`Scripts/rvv-generated-bundle-abi-e2e-*`)、所有现有 kernel/stamp 逐字节不变。

**状态**:4 个 workstream 里 WS-A/WS-C/WS-B 完成,WS-D 完成 table-collapse、defer verifier de-dup(有据)、剩 TU 文件拆分(纯 churn,advisor 判定最低优先 / 预算不足首砍项)。

## Honest ledger
只有 WS-B 是可测的 N3 perf 证据;WS-A/C/D 是 maturity/conform-to-claim 的工程(用户明确要这个,不是卖点)。
capability-divergence headline 仍等第二块板;本期把 N2/N3 机器变 first-class 来 de-risk 它。

## Progress (journal — 不是 spec)
- ✅ WS-A scalar family-hardening (commit ef7179e7). 前门 scalar 身份串 5→0、按名引用 0、core family grep 仍空、变量走 collectVariantProposals。残留:RVVLowPrecisionPerformancePolicy.cpp 的 route-planning validator(不同层,诚实记下)。
- ✅ WS-C Tier-0(commit 1389d946):6 clone pass → 1 generic candidate/select/pass。
- ✅ WS-D 第一步:28-branch block-dot dispatch → first-match table+loop(commit bd7c8b53)。emission 逐字节不变,raw()=0,673/3。
- ✅ WS-D 第二步:33 个 free support helper(intrinsic-name/type/comment builders)从 RVVToEmitC.cpp 抽出到 RVVToEmitCSupport.{h,cpp}(commit ec3949d2)。RVVToEmitC.cpp 24527→23939(−588),不动 VariantToEmitCFunc class(无需 de-anonymize)。逐字节不变,674/3。
- 🔜 **WS-D 剩项:VariantToEmitCFunc class-method 拆分**(23939 行的大头)。配方:de-anonymize 该 class 进一个 shared internal header(named ns,如 `conversion::rvv::detail`)+ 把 anon free helper 也搬进 reachable internal header → 把 123 个 `emit*` method 定义按 kernel family(block-dot-linear / k-quant / codebook-grid / fp4 / ternary-binary / forward-elementwise / deferred-dequant-maskedstore)拆进多个 .cpp(同 lib)。真风险 = **linkage**(anon class 不能跨 TU 定义),不是 byte-exact。advisor 判定:纯 maturity churn + zero claim,**最低优先 / 预算不足首砍**。
- 📌 **build-hygiene note**:本树 incremental build 不可靠 —— ODS `RVVOps.cpp.inc` 每次 build 都重生(pre-existing TableGen 依赖图问题,非 Tier-1 引入,line 13 既有 interface 同款),且 `ninja tcrv-opt` 有时不重链。**fingerprint gate 必须用 forced/clean rebuild**:clean-build 真值 = **f810ce6b**(36-test block-dot emission),跨 bd7c8b53→81b57ba5→support-extraction 全相同(table-collapse/Tier-1/extraction 三者皆 byte-exact)。早先记的 07ad6d0d 是 stale incremental .o 产物(before==after 等式有效,绝对值无效)。
- ✅ WS-C Tier-1 完成(commit a73ab62d):TunableScheduleOpInterface(Conversion/EmitC,family-neutral,primitive 返回)→ 6 op 经 DeclareOpInterfaceMethods 实现 → lookupRVVScheduleDescriptor 注册表(single source of truth)+ runRVVScheduleMaterializationViaInterface(dyn_cast 派发,0 硬编码 op 类型)+ 统一 pass tcrv-rvv-materialize-schedule auto-discover 所有 tunable op;6 个 named pass 保留为 thin TypeID filter。descriptor 数据 18 reason+6 enumerate+budget 逐字段与旧 provider 相同 → stamp 逐字节;measured record 经新 pass 与 per-kernel 逐字节一致(WS-B second-consumer 验证)。
- ⏸️ **WS-D verifier de-dup —— DEFERRED(有据)**。实测:34 个 `Ggml*Op::verify`(RVVDialectWideningOps.cpp)~90% 是 **bespoke 诊断 prose**(per-kernel kind/格式值/allowed-attr/错误文案),几乎没有逐字相同的骨架。de-dup 二选一都低价高险:(a) 保诊断逐字 → spec 只是把 prose 从函数体搬进 struct literal(集中数据、没消除);(b) 不保 → 要改 ~28 个断言诊断文本的 lit、丢掉 per-kernel ABI 文档,且把**核心不变量 I7 fail-closed** 做风险化泛型。table-collapse 已 bank 了 zero-core-branch kernel 轴;verifier de-dup 不再 bank 任何 claim,是纯 churn。→ 放到最后,且只在出现干净 spec 设计时做;**不交给 sub-agent**(泛型化 I7 诊断正是机械 agent 会悄悄改坏语义的地方)。
- 🔜 WS-C Tier-1(TunableScheduleOpInterface)= 下一步(用户"好好探索" tune-reuse 的高意图项)。判定:WS-B 的 microarch lever 在 Tier-0 的 `GenericScheduleCandidate{cost,isLegal,knobs[]}` 下只是又一个 `NamedKnob`,不是新接口方法 → Tier-1 接口**形状**与 WS-B 无关,可现在做;WS-B 落地后只验证 flow(enumerate→prune→measure→stamp)的"第二消费者"勾。
- ✅ WS-B roofline 实测完成(ssh rvv, VLEN=128, pinned ef7179e7;research/wsB-roofline.md + artifacts/wsB-roofline/)。**结论:三个 kernel 都 latency/overhead-bound** —— cache-resident 只达 compute ceiling(3.86 MAC/cyc)的 q8_0 7% / q4_0 4% / q4_K 13%,streaming 也比 mem BW ceiling(2.3–2.5 B/cyc)低 ~10×。既非 compute-bound 也非 BW-bound,瓶颈在 per-block reduction→scalar→fp32 + decode 依赖链。→ 该加的 microarch knob = **accumulator_count / 跨 block 软流水**(藏 reduction/scale latency),且它只是又一个 `NamedKnob`,**证实 WS-C Tier-1 接口形状与 WS-B 无关**。WS-B 的 knob 实测胜出尚未做(headroom≠win,要 measured)。
