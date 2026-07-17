# 旧 trellis task series 关停方案(独立审核项)

> **待用户批准;未执行。** 这是与 parent PRD **分开**的第二个审核决定(advisor:给用户 cleaner control)。批准前不 archive/finish 任何 task。

## 为什么关停

`compiler-maturity-retest` 系列(2026-06-26,34 子任务/5 phase/9 child)按 **"增量砖 + 表填"** 组织。它**交付了真东西**(下面标 ✅ done-history),但其组织原则撞了 multi-validator 墙——继任者 [[07-01-arch-refactor-noperand-core]] 改用"根上架构 consolidation"。旧 series 的**活的意图**要么被继任者吸收、要么归入 deferred perf 自测,不再作为独立 active task 悬着。

## 逐 task 处置(建议)

| task | 现状 | 建议处置 | 去向 |
|---|---|---|---|
| [[06-26-emitter-l0-l3-infra]] | completed | **保留 done-history**(不动) | 已完成的 byte-exact 整理 |
| [[06-26-phase0-doc-spec-closure]] | completed | **保留 done-history**(不动) | 已完成的 doc/spec 闭环 |
| [[07-01-research-realign-maturity-roadmap]] | planning(DoD 实已全绿) | **标 completed** | 本会话刚交付(journal+README+spec+dossier,2 verifier PASS) |
| [[06-26-compiler-maturity-retest]] | in_progress(parent) | **archive —— 被继任者 supersede** | 砖/发现是 done-history;perf child→deferred perf;架构 child→refactor pillar |
| [[06-26-gemm-op-builds-tooling]] | in_progress | **archive —— 意图拆分吸收** | `trackB-production-export`→**P1**(正是 q4_0/codebook production-export);`build-q80/q50-gemm`→**deferred perf** |
| [[06-26-track-b-generic-lowering]] | in_progress | **archive —— 意图入 P2** | G1/G2 + 6-brick 见证 = done-history;full-kernel 泛化→**P2** |
| [[06-26-substrate-probe-hart]] | planning | **archive —— 意图入 P4** | 真硅片 probe + hart gate→**P4**(perf-free) |
| [[06-26-n2-ime-gemm]] | planning | **archive —— 意图入指标②/③** | IME GEMM payload = N2 payload(指标②)+ 跨 family benchmark(指标③ finale) |
| [[06-26-row2-beat-levers]] | planning | **archive —— 意图入指标②/③** | cm4/cm5/cm6 beat = mechanism-synthesized 形状(指标② beat)+ 指标③ 证 |
| [[06-26-winA-parity-bricks]] | in_progress | **archive —— 意图入指标③** | wa1/wa2 已做;wa 砖 = 指标③ coverage sweep 的 parity 格 |
| [[06-26-table-retest-fill]] | planning | **archive —— 意图入指标③** | 表重测/填 = 指标③ finale 的诚实表(重构后整批重测) |

## perf 去哪(不再是独立 deferred task)

新 PRD 把性能实验做成**指标③——重构的 finale**(在①覆盖成熟、②机制落地之后跑)。所以上面 4 个 perf child 的意图**归入指标②/③**,不是"独立 deferred perf task"。它们 archive;perf 工作在重构的 ②(payload/beat)+ ③(finale)里发生。新板(211.87.236.28/openEuler)、一致 7B、correctness-before-timing、kernel-micro 与 e2e 分报、naive/scalar 绝不当贡献倍数;重构会 invalidate 一堆 perf 格,所以**重构后整批重测**(先测是浪费)。

## ⚠ 执行后的 stale-hook 提示(给下个 session)

当前 workflow-state hook 指 `compiler-maturity-retest → trellis-implement → trellis-check → trellis-update-spec → finish`。**该 task 一旦 archive,这条 hook 建议即 stale**——下个 session 不该再按它重开那个 loop。批准执行后,current-task 指针应转向 [[07-01-arch-refactor-noperand-core]](或留空由用户 `task.py start`)。

## 批准后我会执行的动作(命令级,现在不跑)

1. `task.py` 标 `research-realign-maturity-roadmap` = completed。
2. `task.py archive` 上表 7 个(compiler-maturity-retest 及其 archive-建议 children)。**注**:先核 archive 是否级联 child;若不级联,逐个 archive。archive 保留内容在 `.trellis/tasks/archive/`,非删除。
3. current-task 指针 → `arch-refactor-noperand-core`(待用户决定是否立即 `start`)。
4. **不** spin 任何 refactor 子任务(那是 parent 批准后的下一步)。

**两个独立审核决定**:(A) parent PRD [[07-01-arch-refactor-noperand-core]] 的结构/范围;(B) 本关停方案。可分别批。
