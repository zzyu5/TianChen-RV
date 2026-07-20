# B3：K-quant vec_dot 强对手结构攻坚

## Goal

承接 ISSUE-109，在 q4_K/q6_K 及合理 fanout 上对真实手调部署对手做一次结构级攻坚。已证伪的 register-fusion/vwredsum/min-term/单 super-block 旋钮不复试；当前候选是跨 super-block MLP 或经新反汇编确认的同等级结构杠杆。

## Pre-registration

- op：`vec_dot`
- primary formats：q4_K、q6_K；q2/q3/q5_K 仅在 B2 oracle 完成且 fanout 机制同源后加入。
- board：rvv；k1 仅在 B2 VLEN256 correctness/对手身份闭环后加入。
- engine：rvv；regime：decode/M=1 对应现有主表键，启动时以 master canonical token 为准。
- opponent：板上实际派发的 ggml `_vl128`/`_vl256` 手调 K-quant vec_dot，双方 clang-18/同会话配对。
- correctness：整数 byte-exact + ZERO-MODEL + 3/4-arm anti-hollow + CORE==PROD。
- timing：cold、预注册重复/seed/cache flush；正式写三目的地。
- success：先以现行 0.8 gate 判地盘，`>1` 才可写 beat；near-parity 不夸大。

## Scope

- 重新确认当前瓶颈和对手结构，验证 memory-stall/call/MLP 假说。
- 一次只实现一个结构杠杆，保留 baseline/attack/negative isolation。
- 若杠杆 no-flip，记录 objdump/perf disposition；不连续叠旋钮追数。
- 有效改进必须部署到真实 emitter path，再更新 master；standalone proven 不等于 deployed。
- 若机制可扇出，先列 same-wall evidence，再逐格式验证，禁止未经验证外推。

## Primary Touch Set

- K-quant vec_dot emitter/body realization
- `tools/bench/cells/vec_dot.sh`（仅必要调用，不与 B2 并行改同文件）
- focused correctness/emit tests
- `experiments/runs/<run-id>/`、runs.log、master controlled update

## Dependencies

- 依赖 B1 正式数据 control 和 B2 q4/q6 harness 合格。
- 与 B4/B5 可并行，但真板按单 owner 调度。
- A 线若改同一 emitter/stamping，需以 commit boundary 串行并做 B6 paired regression。

## Acceptance Criteria

- [ ] 瓶颈以 deployed ours/opponent objdump 和可用 counters 证成。
- [ ] 杠杆有独立 baseline/attack/negative isolation，正确性全绿。
- [ ] 正式 cold 数据走 bench/run-id/三目的地。
- [ ] 有效候选进入真实 emitter 后再称 deployed；否则只称 proven candidate。
- [ ] no-flip/墙同样落 disposition，不重复已证伪 lever。
- [ ] 对手档、compiler symmetry、板/相/格式和账本齐全。
- [ ] e2e 是否传导交给 B6，不在本任务外推。

## Verification

- focused build/lit + byte-exact;
- official bench verify/measure;
- objdump/counter comparison;
- recon/metrics after controlled update;
- independent rerun for near-gate results.

## Out of Scope

- inline asm 绕 clang、弱对手替换、全 K-quant 一次扇出、e2e 总结、在线 tuning。

## Issue Mapping

- ISSUE-109、ISSUE-112；ISSUE-114 correctness 前置由 B2 处理。
