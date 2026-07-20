# B6：E2E 传导与重构前后 paired regression

## Goal

对 B3/B4/B5 中至少一个闭环 candidate family 做 kernel→deployed path→e2e 的传导实验，并维护 A 线重构前后性能无回退证据。micro improvement、e2e wash、系统收益都作为结果，禁止只保留好看的方向。

## Entry Conditions

- B1 control plane 已稳定。
- B3/B4/B5 至少一个 slice 已 deployed、正确性和 official micro measurement 完整。
- A2/A4/A6 若涉及该 slice，已有明确 before/after commits 与相同输入路径。

## Pre-registration

- workload：从现有真实 ggml/llama.cpp 接入中选，不新增 workload domain。
- 至少一个 compute-dominant 场景；若已有 memory-bound 代表场景则并列一个，用于传导判别。
- boards/threads/model/input/build/toolchain/frequency 固定。
- paths：stock deployed ggml、Weft analytic prior、Weft qualified winner；strong opponent 只有能公平嵌入同一调用边界才进 e2e，否则只留 micro 对照。
- metrics：correctness、既定 e2e 指标、关键 kernel time share、route/variant trace、初始化/选择开销。

## Scope

- 复用已有 e2e harness 和 run lineage，不另建报告体系。
- 对 A 线重构做 same-board/same-session paired regression，区分结构等价与性能变化。
- 用 Amdahl/实际 time share 解释 micro→e2e；不从倍率直觉外推。
- 若 improvement 被 memory/dispatch/packing 洗掉，登记 wash 与判别键。
- 汇总 deployed ggml、strong opponent、e2e 三类证据的互补关系。

## Primary Touch Set

- existing e2e harness/scripts and route trace
- official e2e raw lineage/ledger destination
- task analysis artifacts;最终数字仍由 master/正式表和 run-id 引用
- 不改 compiler 公式逻辑，发现问题回 A/B 对应 owner

## Dependencies

- 依赖 B1 和至少一个 B3/B4/B5 完成 slice。
- paired regression 依赖对应 A 线 commit。
- 环境冻结/历史盘点可并行；真板由单 owner 串行。

## Acceptance Criteria

- [ ] stock/prior/winner 使用同模型、输入、线程、链和板状态。
- [ ] route trace 证明执行预期 variant，无 silent fallback。
- [ ] 旧 production symbol/caller、平行 runtime dispatch、compat route 为 0；缺 stamp/route 的负例 fail-closed，并有防复活回归。
- [ ] e2e 数字可追到关键 kernel 占比和 micro run-id。
- [ ] initialization/selection overhead 独立报告。
- [ ] A 线 before/after correctness 与性能配对完整，无隐藏退化。
- [ ] e2e gain、parity、wash 都有正式 disposition。
- [ ] 不把 deployed-path 正确性夸写为全面击败 strong opponent。

## Verification

- e2e correctness and route identity;
- same-session paired timing and noise qualification;
- micro/e2e transduction accounting;
- official lineage/master/evidence consistency;
- independent confirmation for headline/near-gate result.

## Out of Scope

- 新模型家族、全工作负载 sweep、把 strong opponent 强行嵌入不公平系统路径、图级 compiler 重构。

## Issue Mapping

- ISSUE-030 与现有 e2e/transduction ledger；任何新 wash/route mismatch 登记新 issue，不在 PRD 隐藏。
