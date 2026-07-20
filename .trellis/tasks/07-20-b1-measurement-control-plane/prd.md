# B1：Measurement control plane 与资格收口

## Goal

在不跑新板数据的前提下，把正式性能数据的控制面收成单一、可复算链：明确 master 写 ownership、regime 行键、reader 默认路径、qualification/freshness/T-N 与已有资产矩阵，确保后续 B2–B6 的正式数字不会被 recon 覆盖、错行或误升格。

## Existing Assets

- runner、四元行键、三目的地、五类 cell harness 已建。
- T3 board CSV 当前 disposition 零未分类；perf-covered 报告三源一致。
- ISSUE-067/073/093 已关闭，不得重新建设 runner/master/runs。

## Scope

- 生成当前资产矩阵：每个可引用 cell 的 row key、board、opponent、correctness、run-id、freshness、qualification 与 evidence lane。
- 对 ISSUE-098 给出并落地单一 master ownership：recon 生成与 bench controlled update 不能相互静默覆盖。
- 关闭 ISSUE-097 regime 覆盖风险和 ISSUE-108 reader 默认旧路径。
- 定义/实现 T-N qualification 的生成与引用位置；区分 measured、T-N-qualified、selection-valid。
- 只处理影响 official master/runs 的 ISSUE-069 findings；不在本任务清完全部 732/1372 目录债。
- 为 A5 提供稳定 source-of-truth、freshness 与 winner qualification 契约。

## Primary Touch Set

- `tools/bench/bench`
- `.trellis/scripts/recon_master_rebuild.py`
- `.trellis/scripts/recon_t3_disposition.py`
- `experiments/master/` schema/manifest（不手改数字）
- measurement-related tests/gates
- task artifacts for matrix;不另建第二正式账本

## Dependencies

- B 线首任务；A1 可并行。
- B2 可先做不写 master 的 harness 代码，正式测量依赖本任务。
- A5 的 winner view 接线依赖本任务输出。

## Acceptance Criteria

- [ ] 一个明确 writer policy 保证 recon 重建不覆盖合法 bench 结果，或 bench 不直接写生成物；有冲突负控。
- [ ] 四元行键在 runner、master、reader/recon 一致；regime 不再以空值通配覆盖多行。
- [ ] 所有 reader 默认指 `experiments/master/`。
- [ ] measured / T-N-qualified / selection-valid 三种状态可机算区分。
- [ ] 资产矩阵不把 deployed、strong-opponent、e2e 混成一列。
- [ ] `perf_covered_metrics.py report`、disposition recon 与 runner self-test 继续通过。
- [ ] 没有新正式数字、没有人工改写历史 raw run。

## Verification

- `tools/bench/bench --self-test`;
- recon/disposition/metrics self-tests and reports;
- master ownership conflict regression;
- row-key uniqueness and reader-path checks;
- diff/check and layout gate targeted delta.

## Out of Scope

- 真板测量、全目录 findings 清零、公式 selector 实现、性能攻坚。

## Issue Mapping

- ISSUE-069（仅 official paths 子集）、ISSUE-097、ISSUE-098、ISSUE-108；与 A5 联动但不实现编译期 selector。
