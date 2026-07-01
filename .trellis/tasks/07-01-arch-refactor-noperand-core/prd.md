# TianChen-RV 成熟化重构：让 mature compiler 本身成为科研贡献

> **状态:待用户审核(REVIEW-GATE)。只定义 parent + 分解草案;不 spin 子任务、不动代码,直到批准。**
>
> **Supersedes [[06-26-compiler-maturity-retest]]**(增量-砖组织,撞 q4_0 multi-validator 墙 → 组织原则换成"做成一个成熟 compiler")。旧 series 关停见 `closure-proposal.md`(独立审核项)。
>
> **frame(不可动摇)**:**mature compiler 本身就是科研贡献。** 不是"编译器成熟"与"论文创新"两条轴——是**一件事**:一个能力驱动、覆盖广、跨 family、可实测的成熟 RISC-V compiler,它的成熟度 = 它的 novelty。我们**瞄准强论文**(需要这次重构),不写靠 reframe rung 的保守版。rigor(byte-exact / 证据状态 / 不 over-claim)是**审美不是保守**,保留。

---

## 三个终极指标(本 PRD 按此组织;pillar 是达成它们的执行计划,在后)

用户定的三个交付指标:**① kernel 覆盖度 + 成熟 compiler ② 量化的、强的论文创新点(每个点理清:是什么、要怎样)③ 最终性能实验。** 下面三节 = 这三个指标的 endpoint 描述;第四节 = 达成它们的 pillar 执行计划。

---

## 指标 ① —— 成熟 compiler 是什么(endpoint,coverage)

**endpoint:一个 generic、能力驱动的构造机制,把我们做过的所有 kernel body 都自动生成出来——不是 per-kernel 手写 monolith,而是一条成熟机制覆盖全 zoo。** "我们是一个编译器,不是一条编译器路径。"

具体"成熟"= 下面每一类 **body-shape 类别**都经 generic construction(Track-B 泛化)自动构造、byte-exact、能力驱动选形状、production-export 端到端可达:

> **⚠ coverage 用 body-shape 类别 bound(不是逐个 26+ op,那是无限工程)。同类别内具体 kernel = 参数化,非新代码。** 下面是**建议 taxonomy,待用户确认/调整**(审核项 R1):

| # | body-shape 类别 | 代表 kernel | 现状 |
|---|---|---|---|
| C1 | 平凡 contraction / dequant(widening product→reduce[→dequant]) | dequant dot | ✅ **production-e2e DONE**(VLEN128 m2/m4 + VLEN256 m1/m2) |
| C2 | nibble-decode contraction(packed-i4 nibble,2-operand) | q4_0/q4_1/q5_0/q5_1 nibble | 🟡 G1 整数核 done;full-kernel 未 |
| C3 | **offset-binary contraction(packed-i4,3-input weight+qlo+qhi)** | q4_0_q8_0 | 🔴 **BLOCKED(multi-validator 墙)——P1 根债** |
| C4 | codebook-gather contraction(broadcast+gather) | iq4_nl/mxfp4/iq2_xxs | 🟡 G2 整数核 done(真 VLEN flip);full 未 |
| C5 | K-quant super-block(scale/min bit-dance + super-block loop + fp fold) | q4_K/q5_K/q6_K | 🟡 6-brick 见证 byte-exact,**未接 production** |
| C6 | IQ-gather(vluxei16 signs64) | iq1_s/iq1_m/iq3/iq2 | 🟡 iq1_s done;iq2 卡 signs64 op-attr |
| C7 | full-kernel GEMM/GEVM + repack(整数核之上的 multi-block / 内存布局层) | q4_0/q8_0/q4_K GEMM/repack-GEMM(26 BlockDot/7031 LOC) | 🔴 全手写 monolith,12/26 上 tunable |
| C8 | forward 算子(silu/softmax/norm 等 elementwise+reduction) | silu/softmax | ⚪ 手写;low-pri coverage |

**指标①成功 = C1–C7 每类都经 generic construction 自动构造 + byte-exact + 能力驱动选形状 + production-export 可达**(C8 low-pri)。这是"支撑之前所有 kernel"的具体含义。

---

## 指标 ② —— 量化的强论文创新点,逐点耦合到重构(科研化)

> 这是"把 journal 的科研分析**应用到**重构"。每点:**(a) 现状诚实(引 [[06-26-research-realign-maturity-roadmap]] journal + dossier `notes/07`§0 Correction,不重推) (b) "强"是什么(note-07 的 ceiling-raiser) (c) 重构交付什么把它做到 (d) status = in-progress via P_x。** ⚠ rigor:"强 = ceiling-raiser" 是 **target 非 achievement**;status 写 in-progress,不 smuggle premature claim。

### N1 —— capability substrate
- **现状**:substrate、已 demote(queryable object 是 prior-art:DLTI/IREE-HAL/TVM/SubtargetFeature+TTI/FODA);novelty 只在跨 family conjunction;三个真洞 = probe "probes no hardware"、conflicts inert、implies mechanism-thin。可迁移原则是 **mechanism**(同一 relation-bearing schema 统一 compile-time generation + runtime guard、跨 paradigm 边界不改复用)**非 discovery**("扩展会 layer" 是 `SubtargetFeature.Implies` 已建模的)。
- **"强"**:N1 **active on silicon**——真 probe 驱动一个真决定;且 relation-set 驱动**一条成熟广覆盖机制**的 generation(不是 5 砖),conjunction 在**全 zoo + 跨 family** fire。
- **重构交付**:P4(真 probe ingestion + relation 真 fire)+ P1/P2(N-operand generation + full-zoo 构造让能力事实驱动**每个** kernel 怎么造 = "active measured driver" 上规模)。
- **status**:in-progress via **P1/P2/P4**。

### N2 —— 零-core-branch 跨 family admission(keystone)
- **现状**:结构 **PROVEN**(grep-clean、K1 16/16 = 一个 4×4 tile 的 16 int32 字);但 IME implies RVV(矩阵 paradigm 挂 vector 核、非独立 family);payload correctness-only(无 benchmarked GEMM)。
- **"强"**:那条**成熟机制**正是第二 family branch-free 接入的东西——广 RVV 覆盖 + IME 走同一路 强化"一条 common 路服务所有";再加 **benchmarked IME GEMM payload**(非仅正确)。
- **重构交付**:P2(IME 接入的成熟机制)+ IME GEMM payload(原在 deferred perf,现耦合进③)。**唯一 feasibility-gated 项 = 独立第三 family 硅片,在重构 scope 外**(硅片可能不存在;提一次,不作 frame)。
- **status**:结构 PROVEN;payload in-progress via **P2 + ③**。

### N3 / Track-B —— 能力驱动构造 + tune(指标①的引擎)
- **现状**:Track-B 真在整数核(4 前门/2 flip)、q4_K decomposable-未接、全 zoo 手写;N3 selector capability-**盲**(memoization + 静态 argmin)、corollary;dequant 现 production-e2e(export-lit tier,非硅封)。N3 独立 tuning **不 licensed**(弱于 TopHub/Roller/Welder)——残值 = capability-keyed 统一选择,realization 杠杆在 Track-B。
- **"强"**:generic 机制生成**全 zoo**(= Track-B-full-kernel ceiling,指标①);resource-aware cost model 驱动 selection;**mechanism-synthesized beat**(综合一个 ggml 没手写的 within-kernel 形状、e2e 更快)。
- **重构交付**:P1(N-operand enabler)+ P2(full-zoo 构造 = coverage)+ P3(resource-aware cost + selection)+ beat 在 ③ 证。
- **status**:in-progress via **P1/P2/P3 + ③**。

### 性能 beat
- **现状**:vs ggml 自己 kernel = **parity-now**(q4_0 ~0.94×/q8_0 ~1.0×/q4_K 1.26× micro+manual-stamp);无 clean e2e beat;赢 naive 2.27–3.79× = 内部 sanity 绝非基线。
- **"强"**:一个 mechanism-synthesized within-kernel 形状,e2e 快过 ggml 自己 kernel,两板。
- **重构交付**:P3(resource-aware selection 综合形状)+ ③ 证。
- **status**:**target 非 achieved**,in-progress via **P3 + ③**。

### admission boundary(次要,非 pillar)
- **现状**:设计原则非 mechanized gate(falsifier fires)。**"强"**:可 registration-time mechanize(**永不 per-dispatch**——破 N2 falsifier);note-07 = claim-scoping choice、optional、non-existential。**重构交付**:optional 低优先(P4-adjacent),非 pillar。**status**:设计原则(保留);mechanize = optional future。

---

## 指标 ③ —— 最终性能实验(finale)

**在指标①覆盖成熟、②机制落地之后**跑(重构会 invalidate 一堆 perf 格,先测是浪费)。新板(rvv=211.87.236.28/openEuler)、**一致 7B 模型**、correctness-before-timing、同板 baseline+artifact。

- **coverage sweep**:C1–C7 每类 ≥1 代表 kernel,自动构造 body 的 correctness(vs scalar oracle **且** vs ggml)+ micro。
- **beat 条件**:mechanism-synthesized 形状 vs ggml 自己 kernel,micro **且** e2e,两板;**kernel-micro 与 e2e 分报**(compute-bound 胜不传导 memory-bound decode,[[kernel-wins-dont-transplant-to-e2e]])。
- **跨 family**:benchmarked IME GEMM(tag 我们-emitted 非 ggml-spacemit;Win-B2=parity 天花板)。
- **纪律**:parity 与 beat 分格、每格证据状态标、naive/scalar 绝不当贡献倍数、重构后触及格 STALE 必重测。

---

## Pillar 执行计划(达成①②③的 how-to;character 是执行注记非组织原则)

> P1 先,非商量(根债 + 具体墙 + 解锁已知阻塞)。bounded exit = **coverage 里程碑**(非一个 body)。

| pillar | character | bounded exit(成功 = 这个具体覆盖状态) | 服务指标 |
|---|---|---|---|
| **P1 N-operand route 统一** | **真架构核·第一** | product-head arity 抽成**一个** N-operand route-identity,三套 mirror-validator(route-family / construction-protocol / emit-role)从它派生;**C3 offset-binary + C4 codebook** 经与 C1 同一 production route 端到端导出;全 2-operand 路 byte-exact 无回归 | ① C3/C4;② N1/N3 |
| **P2 generic 构造全覆盖** | 半重构→coverage 引擎 | C2/C5/C6/C7 逐类经 generic construction 自动构造(整数核 + **full-kernel** GEMM/GEVM),byte-exact vs monolith;"哪类由 G 覆盖 vs fallback" 落成机制;C8 low-pri | ① 全 zoo;② Track-B/N2 payload |
| **P3 resource-aware cost + selection** | gap-fill→选择成熟 | cost model 读真 resource fact(VLEN/ELEN/vreg/mask/tail)非 1-bit;在跨类 kernel 上能力驱动选形状 + 综合出 beat 候选形状 | ② N3/beat;③ |
| **P4 capability live path** | gap-fill·perf-free | 真硅片 probe → capability ingestion(替 march fact)两板;≥1 relation(conflict/implies)真 probe 下 change 一个决定 | ② N1 active-on-silicon |

**排序**:P1 → P2(受益 N-operand 统一)→ P3/P4 并行可后。③ 在①②足够成熟后收尾。

---

## Scale 诚实(拒 small-scope,非拒 slow-scope)

**这是 multi-quarter 到 multi-year 的工程,不是 multi-month。** 全 body-shape 类别经 generic construction + full-kernel GEMM + N1-on-silicon + resource cost + 一个实测 beat = 大工程。**诚实说规模不是 hedging 野心**——野心不带 scale 诚实,就会"以为快到了"再次卡半途。分季度里程碑:先 P1(解锁 C3/C4)→ P2 逐类推 coverage → P3/P4 并行 → ③ 收尾。

## Scope 纪律 / 保住 working

- **非推倒重写**:保 dequant production-e2e / IME N2 / 5 board-sealed 砖 / 429 RVV lit——任一回归 = 停。
- **byte-exact / no-regression gate** 每改必过(forced clean rebuild + BEFORE==AFTER,[[build-incremental-unreliable]])。
- 实现走 sub-agent(trellis-implement → trellis-check),不在主会话写代码;spec/doc/task-status 主会话。每 pillar 落地后同步 doc/spec 现状口径。

## Out of Scope

- **推倒重写**(毁 working)。**独立第三 family 硅片**(feasibility-gated,可能不存在;提一次、非 frame、重构 scope 外)。frontend linalg/tosa;discrete-card offload;JIT/runtime tuning;RA/指令调度/软件流水(非我们 EmitC 边界)。
- **注(非否认)**:论文的 reframe rung 作为不需要重构的 fallback **客观存在**,但我们**不瞄它、不按它组织**——PRD 瞄强论文(需要重构)。若强论文 slip deadline,reframe 是未言明的安全网,不是 frame。

## 子任务分解草案(future children,待批才 spin)

| pillar | 草拟子任务 | 依赖 |
|---|---|---|
| **P1** | 1a N-operand route-identity 抽象设计;1b route-family 接入(参考旧 arc 弃掉的 additive 层);1c construction-protocol N-load 化(`RVVConstructionProtocol.cpp`);1d 探接第 3 validator(emit/role);1e C3 e2e 封;1f C4 e2e 封 | 链 |
| **P2** | 2a full-kernel 构造机制设计(挑一类如 C5 q4_K,6-brick 已在);2b 逐类 C2/C5/C6/C7 推 coverage;2c 归属规则落机制 | P1 后 |
| **P3** | 3a resource-fact 入 cost model;3b 跨类 selection flip + beat 候选 | 可后 |
| **P4** | 4a 真硅片 probe ingestion(吸收 `substrate-probe-hart`);4b relation 真 fire | 可后·perf-free |

## Decision(ADR-lite)—— 待用户审核

**待批**:(A) 三指标结构 + coverage-by-body-shape-taxonomy(R1:C1–C8 分类对不对?)+ pillar 执行计划 + scale 诚实;(B) `closure-proposal.md` 旧 series 关停(独立决定)。

## Technical Notes
关联 memory:[[paper-realign-framing]] [[backend-maturity-triton-reframe]] [[n1-substrate-emission-not-maturity]] [[kernel-wins-dont-transplant-to-e2e]] [[build-incremental-unreliable]] [[k1-ime-n2-hardware-candidate]] [[option2-path-selection-real-pass]] [[emitter-maturity-vluxei16-widelmul]]。P1 根因 recon 在 [[06-26-compiler-maturity-retest]] 的 `research/production-export-wide-body-gap-FINDING.md` DEEPEST。科研分析全文在 [[06-26-research-realign-maturity-roadmap]] journal + dossier `papers/TianchenRV/`。
