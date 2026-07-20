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
- B1 不拥有 candidate、legality、prior 或 compute selection；这些由 A2/A5 消费本任务的 qualified view 后决定。

## Primary Touch Set

- `tools/bench/bench`
- `.trellis/scripts/recon_master_rebuild.py`
- `.trellis/scripts/recon_t3_disposition.py`
- `tools/bench/measurement_keys.py` / `tools/bench/tn_qualify.py`
- `experiments/master/` schema/manifest（不手改数字）
- measurement-related tests/gates
- task artifacts for matrix;不另建第二正式账本

## Dependencies

- B 线首任务；A1 可并行。
- B2 可先做不写 master 的 harness 代码，正式测量依赖本任务。
- A5 的 winner view 接线依赖本任务输出。

## Acceptance Criteria

- [x] 一个明确 writer policy 保证 recon 重建不覆盖合法 bench 结果，或 bench 不直接写生成物；有冲突负控。
- [x] 四元行键在 runner、master、reader/recon 一致；regime 不再以空值通配覆盖多行。
- [x] 所有 reader 默认指 `experiments/master/`。
- [x] 被替代的 master writer、旧 reader、含混 regime 通配路径和 promotion alias 为 0；历史 raw/run 只作不可变证据。
- [x] measured / T-N-qualified / selection-valid 三种状态可机算区分。
- [x] T-N 资格不是文本 token：N=10 正控与 N=9、CI 含零、source hash 漂移负控常驻。
- [x] 资产矩阵不把 deployed、strong-opponent、e2e 混成一列。
- [x] `perf_covered_metrics.py report`、disposition recon 与 runner self-test 继续通过。
- [x] 没有新正式数字、没有人工改写历史 raw run。

## Verification

- `tools/bench/bench --self-test`;
- recon/disposition/metrics self-tests and reports;
- master ownership conflict regression;
- row-key uniqueness and reader-path checks;
- `python3 tools/bench/tn_qualify.py --self-test`;
- diff/check and layout gate targeted delta.

## Out of Scope

- 真板测量、全目录 findings 清零、公式 selector 实现、性能攻坚。

## Issue Mapping

- ISSUE-069（仅 official paths 子集）、ISSUE-097、ISSUE-098、ISSUE-108；与 A5 联动但不实现编译期 selector。

## Completion Record（2026-07-20）

- canonical chain 已闭合：exact roster key → immutable official run → structured T-N qualification → harmonizer 派生资格 → recon 独立重验并原子发布 master。
- 守恒：master 仍为 108 行；`perf-covered=9/83`、`C_construct=101/108`；逐格性能值与 verdict 未借本任务改判。当前 `master-qualified=0` 只说明历史资产尚未统一接入新 run/T-N 链，不等于没有既有性能证据。
- 结构化 T-N 正控 N=10；N=9、无效应、CI 含零、source hash/freshness 漂移均 fail closed。control-plane hermetic bridge 证明 qualified official run 可进入 recon，stale 负控被拒。
- recon 连续两次产物字节稳定：master SHA-256 `20dc9c594311ae12cab5d9890ad4e240d143ed9b80bee7324c39203ab7b5b90a`；rowclue SHA-256 `4bb6b9d87884f5dbe8bb8b0ae0bbc0b632137bae945595c184875f712480e830`。
- layout gate 仍为基线既有 732 findings（本任务前后相同；B1 仅负责 official-path 子集）。E5 real report 未运行：主目录与 worktree 均无 `build/bin/weft-opt`；hermetic E5 self-test 已通过，未伪称 real report 通过。
