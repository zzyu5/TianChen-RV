# compiler 架构重构：N-operand route 统一(核) + 成熟度 consolidation

> **状态:待用户审核(REVIEW-GATE)。本 PRD 只定义 parent task + 子任务分解草案;不 spin 任何子任务、不动代码,直到用户批准。**
>
> **Supersedes [[06-26-compiler-maturity-retest]]**:那个 task(2026-06-26,34 子任务/5 phase/9 child)按 **"增量砖 + 表填"** 组织。它交付了真东西(emitter L0-L3、Track-B G1/G2、dequant production-e2e、5 砖 board-seal),但**它的组织原则本身撞了墙**——q4_0 production-export 的 multi-validator 阻塞(见 [[06-26-compiler-maturity-retest]] 下 `production-export-wide-body-gap-FINDING.md` DEEPEST 段)证明:再多"砖"也过不去,因为 2-operand 假设**复制**在多个平行 mirror-validator 里。**结论:组织原则要从"增量砖"换成"在根上做架构 consolidation"。本 task 是继任者。** 旧 series 的关停方案见配套 `closure-proposal.md`(独立审核项)。

---

## ⚠ 硬护栏(不可违,写在最前——防两类已知失败模式)

1. **paper-defensibility 与 compiler-maturity 是两条可分离的轴**(见 [[paper-realign-framing]])。**论文今天在 reframe rung 就能出**,不需要本重构任何一块落地。**本重构是工程 + 可能的 ceiling-raiser,绝不是 paper gate。** 任何子任务 DoD 不得写"论文需要这个"。
2. **ceiling-raise 是可能的副产品,永不是任一 pillar 的成功判据。** 每个 pillar 的**成功 = 它的成熟度产出**(见各 pillar 的 bounded exit)。"P3 映射到 mechanism-synthesized beat" 这类话是**方向性副益**,**写死为 non-criterion**——否则三个月后它变成"P3 必须出 beat",而按 [[n1-substrate-emission-not-maturity]] cost-model-blind 根本不决定 selection、P3 大概率不出 beat。**若某句话不能同时守住这条线,删掉那句话。**
3. **"完整重构" ≠ 推倒重写。** 是**targeted 架构 consolidation**——保住所有已工作功能(dequant production-e2e、IME N2、5 board-sealed 砖、429 RVV lit)。每个改动过 **byte-exact / no-regression gate**(forced clean rebuild + BEFORE==AFTER,见 [[build-incremental-unreliable]])。**推倒重写明确 out-of-scope**(会毁掉 working functionality + 违反纪律)。

---

## Goal

把 TianChen-RV 的**累积 per-shape special-casing**(2-operand product-head 假设复制在 route-family identity / construction-protocol conformance / 可能 emit-role 三套平行 mirror-validator 里)在**根上 consolidate 成一个 N-operand 抽象一次满足**,并借同一次重构收掉相邻的成熟度债。产出 = 一个**可持续泛化**的 compiler(新 body 形状不再需要跨多个 validator 各 patch)。

## 根诊断(为什么是架构问题,不是又一块砖)

q4_0 offset-binary(3-input product head: weight + qlo + qhi)撞墙,不是因为"少写了一块",而是因为 **2-operand(lhs×rhs)假设被硬编码进多个独立子系统**:
- route-family identity/ABI(已在 06-26 arc 里 additive 建过 N-operand 层,验证 429/429,但因为撞第 2 墙而**弃**);
- construction-protocol conformance(`RVVConstructionProtocol.cpp:2142` `appendWideningProductReduceAddRoleSteps` 发 2-load/12-step spec、`:6367` `verifyRVVSelectedBodySelectedRoleSequence`);
- 作者估计还有第 3 个(emit/role-execution 层)。

**这是架构债的教科书形态**:同一个不变式(product-head arity)散落在多个 mirror,每加一种 body 形状就要在每个 mirror 各改一遍。**refactor = 把 arity 抽成一个 N-operand route-identity,三套 validator 都从它派生。**

---

## 四个 pillar(character 诚实标注 + bounded exit + 副益非判据)

> **character 分级(advisor 核):P1 是真架构重构;P2 勉强算重构;P3/P4 是 gap-fill/feature work,不是 consolidation。四者同挂一个 parent 只为连贯管理,character 不同——写清楚,防"完整重构"膨胀成"我想要的一切 compiler"。**

### P1 —— N-operand contraction-route 统一 【真·架构核 · 排序第一 · 非商量】
- **character**:真重构。撞了具体墙、根因精确(arity 跨 mirror 复制)、进出干净。
- **bounded exit(成功 = 这个具体状态,非"变好")**:q4_0 **offset-binary** body + codebook body 经**与 dequant body 同一条 production route**(`--tcrv-materialize-emission-plans → --tcrv-rvv-lower-to-emitc`)端到端导出;**全部 2-operand 路 byte-exact 无回归**(dequant e2e + 429 RVV lit 不变);**N-operand 假设住在一个抽象里**(三套 mirror-validator 从它派生,不再各自硬编码 2-operand)。
- **可能副益(非判据)**:N-operand generality 是 note-07 的一条 ceiling-raiser。**但 P1 成功只看上面 bounded exit,不看论文。**
- **为什么排第一(非商量)**:它是根债、是具体墙、且解锁**已知被阻塞**的工作(q4_0/codebook production-export)。其余 pillar 不 block 在它上,但它是唯一"必须先做"的。

### P2 —— Track-B full-kernel 自动构造 【勉强算重构 · P1 后】
- **character**:半重构。26 BlockDot / 7031 LOC 手写 zoo 是"每 kernel 各特判"的债,一个自动构造抽象能 unify;但比 P1 更接近"扩覆盖"。
- **bounded exit**:至少**一个完整 kernel(非仅整数核)经 Track-B generic 机制端到端 emit**(note-07 的 Track-B ideal),byte-exact vs 现手写 monolith;并把"哪些 quant 由 G 覆盖 vs fallback 手写"的归属规则(见旧 PRD"Body 发射归属规则")落成机制而非约定。**不要求覆盖全 zoo**(那是无限工程)——要求证明机制能出一个 full kernel。
- **可能副益(非判据)**:Track-B full kernel 是 ceiling-raiser。**P2 成功只看"一个 full kernel 机制 emit + byte-exact",不看论文、不看 beat。**

### P3 —— resource-aware cost model 【gap-fill,非 consolidation · 可后 · 可 defer】
- **character**:feature work(把已知 thin 的东西加厚),不是架构 consolidation。诚实说:cold-start argmin 现 capability-blind(**by design 的已知缺口**,[[n1-substrate-emission-not-maturity]]),live 路是 offline memoization。
- **bounded exit**:cost model 读**真 resource fact**(VLEN/ELEN/vreg/mask/tail 等)而非 1-bit 结构量,且在**至少一个 kernel 上其排序 change 一个真 selection**(否则 = 无 observable 变化,按 costmodel-verdict 关成 investigated 非 build)。
- **⚠ 副益陷阱(advisor 点名)**:**绝不**把"P3 → mechanism-synthesized beat"写成判据。cost-blind 不决定 production selection(live 路是 memoization),P3 大概率**不**出 beat。P3 成功 = cost model 读真 fact 且 flip 一个 selection,**句号**。

### P4 —— capability substrate live path 【gap-fill,非 consolidation · 可后 · perf-free】
- **character**:feature work。N1 的三个"真洞":probe "probes no hardware"(喂 march/synthetic fact)、conflicts live set inert、implies mechanism-thin。
- **bounded exit**:真硅片 probe → capability ingestion(替代 march-derived fact)在两板跑通;**至少一条 relation(conflict 或 implies)在真 probe 下 change 一个 legality/selection 决定**。perf-free。
- **可能副益(非判据)**:N1-active-on-silicon 是 ceiling-raiser。**P4 成功只看"真 probe ingestion + 一条 relation 真 fire",不看论文。**

---

## 排序(写死)

1. **P1 先,非商量**(根债 + 具体墙 + 解锁已知阻塞)。
2. P2 在 P1 后(full-kernel 机制受益于 N-operand route 统一)。
3. P3 / P4 可并行、可 defer(gap-fill,不 block 任何东西;perf-free 的 P4 尤其低风险)。

## Scope 纪律

- **保住 working**:dequant production-e2e(VLEN128 m2/m4 + VLEN256 m1/m2)、IME N2(K1 16/16)、5 board-sealed 砖、429 RVV lit——**任一回归 = 停**。
- **byte-exact / no-regression gate** 每改必过(forced clean rebuild + BEFORE==AFTER;绝不增量;不用 stale 绝对指纹,[[build-incremental-unreliable]])。
- **实现走 sub-agent**(trellis-implement → trellis-check),**不在主会话写代码**;spec/doc/task-status 编辑在主会话。
- 每个 pillar 落地后,**其触及的 doc/spec 现状口径同步更新**(spec = 稳定契约非状态)。

## Out of Scope(本重构不做)

- **推倒重写**(毁 working + 违纪)。
- **全面新 kernel 性能自测**——是**独立的、deferred 的收尾步**(板刚换 = 211.87.236.28/openEuler;一致 7B 模型;correctness-before-timing;kernel-micro 与 e2e 分报;naive/scalar 绝不当贡献倍数)。见 [[06-26-research-realign-maturity-roadmap]] journal §5。**本重构完成后、按需另起**,不混进本 task。
- **论文**(可分离轴,今天就能出)。
- 全 quant zoo full-kernel 覆盖(P2 只证机制能出一个 full kernel;全覆盖是无限工程)。
- frontend linalg/tosa;discrete-card offload;JIT/runtime tuning;RA/指令调度/软件流水(clang/gcc 的,非我们 EmitC 边界)。

## 子任务分解草案(future children —— 待审核后才 spin)

> **不 spin,待批。** 每个子任务独立 PRD + byte-exact/no-regression DoD + trellis-implement/check。

| pillar | 草拟子任务 | 依赖 | 风险 |
|---|---|---|---|
| **P1** | (1a) N-operand route-identity 抽象设计(一个 arity 抽象,三 mirror 派生);(1b) route-family 层接入(复用旧 arc 弃掉的 additive 层为参考);(1c) construction-protocol conformance 接入(`RVVConstructionProtocol.cpp` role-step N-load 化);(1d) 探明并接第 3 个 validator(emit/role);(1e) q4_0 offset-binary e2e 封;(1f) codebook e2e 封 | 1a→1b→1c→1d→1e→1f(链) | **HIGH(跨多子系统结构改;但有精确 recon)** |
| **P2** | (2a) full-kernel 自动构造机制设计(挑一个 kernel,如 q4_K 全体—6 brick 见证已在);(2b) 机制 emit + byte-exact vs monolith;(2c) 归属规则落成机制 | P1 后;2a→2b→2c | HIGH(结构) |
| **P3** | (3a) resource-fact 入 cost model 设计;(3b) 一个 kernel 上 flip 一个 selection 的实证 | 可后 | 中(可能 investigated-only) |
| **P4** | (4a) 真硅片 probe ingestion(吸收旧 `substrate-probe-hart` 意图);(4b) 一条 relation 真 fire | 可后;perf-free | 低 |

## Decision(ADR-lite)—— 待用户审核

**Context**:compiler-maturity-retest 的增量-砖组织撞 multi-validator 墙;用户要求关旧 series + 起一个符合科研路径 + 编译器成熟度的完整重构 task,审核后再 spin 子任务。
**Proposed Decision**:如上四 pillar,P1 为核 + 排序第一;P2 半重构;P3/P4 gap-fill;两轴分离 + ceiling-raise 非判据;非推倒重写;perf 自测独立 deferred。
**待用户批**:(a) 这个 parent 结构/范围;(b) 配套 `closure-proposal.md` 的旧 series 关停(独立决定)。

## Technical Notes

- 关联 memory:[[paper-realign-framing]] [[backend-maturity-triton-reframe]] [[n1-substrate-emission-not-maturity]] [[kernel-wins-dont-transplant-to-e2e]] [[build-incremental-unreliable]] [[k1-ime-n2-hardware-candidate]] [[option2-path-selection-real-pass]]。
- P1 根因 recon(每-validator file:line)在 [[06-26-compiler-maturity-retest]] 下 `research/production-export-wide-body-gap-FINDING.md` DEEPEST 段。
- P2 的 6-brick 见证在 `06-26-track-b-generic-lowering`。P4 吸收 `06-26-substrate-probe-hart`。
