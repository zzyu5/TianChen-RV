# N1 substrate：真硅片 probe + hart 并行 gate（perf-free）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。补 §9.4 两个 doc-没提的真洞。[[n1-substrate-emission-not-maturity]]

## Goal
关掉 N1 的两个 substrate 洞：capability 目前只从 march/synthetic fact 触达（非真硅片 probe）；板子 64/128-core 但全是单 kernel/单线程。

## In-scope（2 原子项）
- **sc10 真硅片 probe→capability ingestion**：`buildRVVTargetCapabilitiesFromProbeFacts` 已 wired 但喂的是 march/toolchain-derived fact（pass 自陈 "probes no hardware"）→ **N1 divergence 目前只能从合成 fact 触达 = 真洞**。做真硅片探针喂进 capability。**两板**，perf-free。
- **sc13 hart 并行 gate 正确性**：`tcrv.exec.hart_parallel`→OpenMP/pthread 的 gate 正确性（`--tcrv-check-hart-parallel-capabilities`）。**硬 dep sc10**（拿真 hart 数）。

## DoD
- sc10：probe 出的 capability fact 与 march-derived 对照、divergence 真从硅片触达；不伪造 SEW/LMUL/tail/mask 等编译期 config（probe 边界，I6）。
- sc13：gate 正确性（perf-free，gate-correctness 非 perf 主张）。
- 证据状态标；两板 `ssh` 证据（I8）。

## Deps / Risk
sc10→sc13 硬 dep。perf-free，低 over-claim 风险。**must-NOT**：把 probe fact 翻成 route/typed-body authority（I6）。
