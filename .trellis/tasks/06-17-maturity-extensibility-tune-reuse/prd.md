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
- [ ] WS-A: scalar 变量由 scalar plugin 在每条 flow 产出;family-name grep 仍空;协议测试绿。
- [ ] WS-C: 一个泛型 candidate+select+pass(或 TunableScheduleOpInterface);6 个 clone pass 折叠;每个 kernel 的 stamp 与调优 pick 逐字节不变;`key=value` record 泛化。
- [ ] WS-D: `RVVToEmitC.cpp` 拆成模块;recognizer/dispatch/verifier de-dup;每个 kernel 的 emit 逐字节不变(additive 重构)。
- [ ] WS-B: roofline 实测 + 一个 microarch knob 落进 tune 接口 + ssh-rvv 实测加速(诚实 win/parity)。
- [ ] 全程 raw()=0、clean rebuild 绿、3 documented reds、所有现有 kernel 逐字节不变。

## Honest ledger
只有 WS-B 是可测的 N3 perf 证据;WS-A/C/D 是 maturity/conform-to-claim 的工程(用户明确要这个,不是卖点)。
capability-divergence headline 仍等第二块板;本期把 N2/N3 机器变 first-class 来 de-risk 它。

## Progress (journal — 不是 spec)
- ✅ WS-A scalar family-hardening (commit ef7179e7). 前门 scalar 身份串 5→0、按名引用 0、core family grep 仍空、变量走 collectVariantProposals。残留:RVVLowPrecisionPerformancePolicy.cpp 的 route-planning validator(不同层,诚实记下)。
- ✅ WS-C Tier-0(commit 1389d946):6 clone pass → 1 generic candidate/select/pass。
- ✅ WS-D 第一步:28-branch block-dot dispatch → first-match table+loop(commit bd7c8b53)。36 个 emission 测试逐字节不变(fingerprint 07ad6d0d…),raw()=0,673/3。
- ⏸️ **WS-D verifier de-dup —— DEFERRED(有据)**。实测:34 个 `Ggml*Op::verify`(RVVDialectWideningOps.cpp)~90% 是 **bespoke 诊断 prose**(per-kernel kind/格式值/allowed-attr/错误文案),几乎没有逐字相同的骨架。de-dup 二选一都低价高险:(a) 保诊断逐字 → spec 只是把 prose 从函数体搬进 struct literal(集中数据、没消除);(b) 不保 → 要改 ~28 个断言诊断文本的 lit、丢掉 per-kernel ABI 文档,且把**核心不变量 I7 fail-closed** 做风险化泛型。table-collapse 已 bank 了 zero-core-branch kernel 轴;verifier de-dup 不再 bank 任何 claim,是纯 churn。→ 放到最后,且只在出现干净 spec 设计时做;**不交给 sub-agent**(泛型化 I7 诊断正是机械 agent 会悄悄改坏语义的地方)。
- 🔜 WS-C Tier-1(TunableScheduleOpInterface)= 下一步(用户"好好探索" tune-reuse 的高意图项)。判定:WS-B 的 microarch lever 在 Tier-0 的 `GenericScheduleCandidate{cost,isLegal,knobs[]}` 下只是又一个 `NamedKnob`,不是新接口方法 → Tier-1 接口**形状**与 WS-B 无关,可现在做;WS-B 落地后只验证 flow(enumerate→prune→measure→stamp)的"第二消费者"勾。
- ✅ WS-B roofline 实测完成(ssh rvv, VLEN=128, pinned ef7179e7;research/wsB-roofline.md + artifacts/wsB-roofline/)。**结论:三个 kernel 都 latency/overhead-bound** —— cache-resident 只达 compute ceiling(3.86 MAC/cyc)的 q8_0 7% / q4_0 4% / q4_K 13%,streaming 也比 mem BW ceiling(2.3–2.5 B/cyc)低 ~10×。既非 compute-bound 也非 BW-bound,瓶颈在 per-block reduction→scalar→fp32 + decode 依赖链。→ 该加的 microarch knob = **accumulator_count / 跨 block 软流水**(藏 reduction/scale latency),且它只是又一个 `NamedKnob`,**证实 WS-C Tier-1 接口形状与 WS-B 无关**。WS-B 的 knob 实测胜出尚未做(headroom≠win,要 measured)。
