# research-story re-align + honest maturity/refactor roadmap

> 父任务上下文：`06-26-compiler-maturity-retest`(compiler 成熟化，实现侧)。本任务是**文档/spec/科研叙事**侧,不是代码实现。纪律护栏见父 PRD。走 trellis 流程:journal 先理清思路 → 再动 README/dossier/spec。

## Goal

把「最近推进 + 更新的理解」对齐进三个文档面,并理清科研定义(谁贡献了什么),留一份诚实的成熟度/重构 roadmap。**不改代码**(除非 spec 编辑)。

三个文档面:
1. **项目 README** `README.md`(2026-06-12,**严重 stale**——N1 当 blockbuster 卖、说"IME does not exist yet / only RVV is a real family"、说"RVV speedups at or below scalar";全是 pre-calibration 误导框架)。**主要重写目标**。
2. **论文 dossier** `/home/kingdom/phdworks/papers/TianchenRV/`(ARS pipeline 精校过,但 **frozen @ HEAD `15d65544`**;需 fold 进 freeze 后 delta + note-07 dependent-layering 头条原则)。**外科式增补,非重写**。
3. **/spec** `.trellis/spec/`(index 已带 N1-substrate-demote + N3-ggml-discipline;深层文件 + dequant production-e2e 里程碑 + multi-validator 边界需对齐)。

## 关键裁定(journal 里展开,证据在 workflow ground-truth)

- **N1 = 重表达(complement),不是删除。** 精校过的「substrate,由 N2 证,非独立卖点」cell **保留**;头条**增补** note-07 §0 的 **dependent ISA-extension layering + cost-of-generality** 原则(把 IME-implies-RVV 从弱点转成中心 *finding*,答"artifact, not idea"攻击)。是 thesis-headline 增补,不是 contribution-structure 改动。
- **N3 ↔ Track B = 呈现选择,不偷换。** N3-独立-tuning 不 licensed(弱于 TopHub/Roller/Welder)。两选项:(i) N3 留 N1+N2 corollary-cell + 显式指针「capability-driven-shape *realization* 杠杆在 Track-B」(4 贡献保留,低风险);(ii) 塌成三贡献(把 selection fold 进 Track-B)。dequant production-e2e capability-flip = "N3 价值发生在 Track-B 构造里" 的实证。journal 呈现两者 + 推荐。
- **贡献归属三桶**:OURS(能力 substrate + 跨 family 零-core-branch 复用 + Track-B 能力驱动构造 + dequant e2e 能力翻转)/ ADOPTED-FROM-GGML(采纳 ggml 自己的 shape/指令 = parity 非 win)/ PRIOR-ART(EmitC→RVV 路、capability-object 想法、FMV/IFUNC/hwprobe、IME MAC 数值)。诚实 delta = **the conjunction**。

## Freeze 后 delta(要 fold 进 dossier + README)

- **(A) Track-B dequant body 现 production-reachable e2e**(VLEN128 m2/m4 + VLEN256 m1/m2,strip 随 VLEN 结构翻转;`ed4a8424`→`43932670`→`59b08f9a`)。**升级** dossier hedge「Track-B witnesses NOT wired to production」→ 一个 body 已 wired + 真能力翻转。
- **(B) q4_0 multi-validator finding**(`fe21f92c`/`10de9c26`/`1ec2d258`):production-export 推到第 2 个 body = N-operand 跨多个 2-operand-baked mirror-validator 的 deliberate 重设计。**双用**:paper cost-of-generality 证据 **+** 用户「compiler 需重构」直觉的具体支撑。

## ⚠ 硬护栏(advisor 定,不可违)

**paper-defensibility 与 compiler-maturity 是两条可分离的轴。** 论文**今天**就能在 reframe rung 出(无需新证据、无需重构);重构是工程收益,**不是 paper gate**。journal 绝不把 paper 进度写成被重构 gate 住——否则「真 compiler」变成跨 deadline 的 scope creep。

## Out of scope(本任务不做)

- **全面新 kernel 性能自测**(用户说「最终」):板刚换(新 rvv=211.87.236.28/openEuler),perf 是**收尾步**,单独一致模型(7B 非 8B)。本任务只**规划**它,不跑。
- **q4_0 N-operand 真重构**(multi-validator 闭环):deliberate fresh-context 设计弧,记进 roadmap,本任务不实现。
- 任何 perf 格改动 / bank 新数字。

## DoD

- [x] journal 落地(`journal.md`,commit `3987af12`):研究定义裁定(N1 重表达成 mechanism-framing / N3 = option(i) corollary+指针 / 归属三桶)+ freeze 后 delta + note-07 对齐现实评估(发现 note-07:61 事实错)+ 诚实成熟度/重构 roadmap(与 paper-defensibility 分开)。
- [x] 项目 README 重写(commit `9897a9a4`):N2 PROVEN(删'IME does not exist yet')/ N1 重表达 / N3 corollary+Track-B / parity-not-below-scalar(naive 2-4×=内部 sanity)/ dequant production-e2e(export-lit tier)/ Track-B 机制。
- [x] 论文 dossier 外科 fold-in(papers repo commit `26fa87f`,8 文件):freeze 后 delta 两项 + note-07:61 事实修正 + dependent-layering 重铸成 mechanism + N3 option(i) + E14/E4/E20/E21/E24 + 00 post-freeze addendum + 05/06/01 reconciliation。
- [x] /spec 对齐(commit `9897a9a4`,4 文件):capability-contract N1-demote echo、design-boundaries paper-language demote + (B) N-operand 已知边界、ime-plugin 'zero family-name branch'、rvv-plugin N1/N2/N3 非 coequal。core-invariants/gen-sel-tuning/emitc-route 不动。
- [x] 全程无 unmeasured-as-success;每 perf 主张证据状态正确;归属清晰。**2 个 adversarial verifier PASS**:README/spec 面 10/10 guardrail PASS(1 MINOR:naive 范围 across-3-chips 误att→修 `8c04cf03`);dossier 面 HARD guardrail 1-8 全 CLEAN + git-verified frozen 完整(5 MINOR = count/pointer 传播残留→修 papers `28464c0`)。tier/name/over-claim/N1-demote/N3-subordinate/paper-vs-maturity 护栏全守。
- **Deferred(非本 DoD)**:全面新 kernel 性能自测(用户"最终";板刚换)只**规划**未跑——见 journal §5。q4_0 N-operand 真重构见 roadmap §4.1。

## Deps / Risk

无代码 dep。风险 = over/under-claim(护栏:每主张对 ground-truth cell;paper vs 重构分离)。
