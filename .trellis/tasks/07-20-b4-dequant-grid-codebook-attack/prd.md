# B4：Dequant grid/codebook 性能边界攻坚

## Goal

选择少量代表性 grid/codebook dequant cell，区分可由 A3 公式/计划表达的结构杠杆与无法静态读取的 gather/compiler 脾气墙，并在 official bench 下推进或诚实封边。重点是 mechanism 判别与真实消费，不是扫全格式表。

## Pre-registration

- op：`dequantize_row`
- representative formats：启动时从 registry/master 选一个 grid（优先 iq3_xxs/同墙代表）和一个 codebook（优先 iq4_nl 或已注册 FP4）；最多再加一个对照格式。
- board：rvv；k1 只有 B2/现有 harness 已证明 VLEN256 correctness 后加入。
- engine：rvv；regime 按 canonical master row，不猜空值。
- opponent：板上实际部署 ggml dequant symbol；另保留 generated prior/selected variant，不用 scalar 假扮强对手。
- correctness：byte-exact/规定 ULP、ZERO-MODEL、anti-hollow、CORE==PROD。
- timing：official cold/run-id；近门独立复测。

## Scope

- 通过 ours/opponent objdump 确定 HW gather、scalar-load、table residency、load LMUL/strip 的差分。
- A3 提供新 c-driven plan 时做旧/新 plan paired test；formula 只限定合法候选，不预写 winner。
- 仅对读得到且可键控的公式墙实施结构杠杆。
- gather/codegen 等脾气墙走三步证伪后登记 boundary，禁止 inline asm/手排调度绕过。
- 保留现有 de-lottery 与已测通过资产，不无理由重测全部格式。

## Primary Touch Set

- selected grid/codebook dequant emitter/plan
- `tools/bench/cells/dequantize_row.sh`（与 B2 不同时修改）
- focused tests
- official run lineage/master controlled update

## Dependencies

- 依赖 B1；harness coverage 依赖 B2。
- A3 新 plan paired path 依赖 A3；不依赖它的现状瓶颈分析可先行。
- 与 B3/B5 可并行，真板串行。

## Acceptance Criteria

- [ ] 格式选择有 registry/master 依据，最多 2–3 个代表 cell。
- [ ] 每格有 opponent identity、objdump 差分和墙型。
- [ ] 公式杠杆只消费 typed g/c，未消费字段不进 plan。
- [ ] 正确性和 official cold/run-id 完整。
- [ ] 正向改进、no-flip、gather/compiler wall 均入账。
- [ ] 成功 lever 进入 canonical plan/selector；失败 lever 留证据但不留下 benchmark-only route、format 特例或 dormant production branch。
- [ ] 单格结论不无证据外推整个 grid/codebook family。
- [ ] micro 结果不自动外推 e2e。

## Verification

- focused build/lit + byte-exact/ULP;
- official bench verify/measure;
- paired objdump/counter analysis;
- near-gate independent rerun;
- recon/metrics consistency.

## Out of Scope

- nvfp4 vec_dot ISSUE-100 的独立 compiler-call attack、全 dequant sweep、inline asm、在线 profile。

## Issue Mapping

- ISSUE-107、ISSUE-112；与 A3/ISSUE-117 交叉。ISSUE-100 明确留给后续独立性能任务，不在本任务偷带。
