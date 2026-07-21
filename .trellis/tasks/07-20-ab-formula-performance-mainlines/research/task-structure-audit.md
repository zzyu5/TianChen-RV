# Task 结构审计

## Verified Tool Facts

- `task.py` 支持 create、validate、start/finish、set-scope、add-subtask、archive。
- `create --parent <dir>` 会同时创建 child 并更新 parent link。
- 当前创建双主线前没有其他 active task；历史 2026-07 archive 有 101 个任务，故新任务必须引用旧资产而非重新开同名战役。
- `task.py` 没有原生 depends-on 字段；DAG 应写在父/子 PRD 的 Dependencies 中，不私造 task.json schema。

## Recommended Tree

~~~text
ab-formula-performance-mainlines
├── a-formula-consumption-refactor
│   ├── a1-formula-authority-freeze
│   ├── a2-typed-decision-contract
│   ├── a3-dequant-c-driven-plans
│   ├── a4-single-authority-stamping
│   ├── a5-qualified-winner-view
│   └── a6-ime-decision-slice
└── b-real-performance-progress
    ├── b1-measurement-control-plane
    ├── b2-bench-cell-coverage
    ├── b3-kquant-exhausted-strategy-retirement
    ├── b4-dequant-grid-codebook-attack
    ├── b5-gemm-deployed-path
    └── b6-e2e-transduction-regression
~~~

## Each Child PRD Must Include

- baseline assets and what is already done;
- concrete objective and out-of-scope;
- primary touch set and authority boundary;
- dependencies/blocks and parallelism;
- executable acceptance commands or measurement contract;
- mapped ISSUE numbers;
- rollback/no-regression requirement;
- for experiments: board, engine, regime, opponent, correctness, lineage and destination.

## Avoided Duplicates

- runner/master/runs are not new tasks;
- clean-room integration is not reopened;
- five dequant plans are not recreated;
- scalar experiments are not declared absent;
- ISSUE-123 evidence relabeling does not masquerade as formula code refactor;
- deployed ggml/strong opponent/e2e remain separate evidence lanes.
