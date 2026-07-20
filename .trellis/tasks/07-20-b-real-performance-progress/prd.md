# B 线：真实性能推进与证据闭环

## Goal

沿正式 measurement pipeline 持续推进真实性能：先补齐 cell/harness、数据资格和 master ownership，再选择少量高价值结构杠杆攻坚，同时维护 deployed ggml、代表性强对手和 e2e 三条互补证据。B 线不等 A 线全部完成才开工，但 A 线改变 decision/selection 后必须由 B 线做配对回归和收益验证。

## Current Assets

- official runner、四元行键、`experiments/master/`、`experiments/runs/<run-id>/`、`experiments/runs.log` 已存在。
- `gemm_tile`、`dequantize_row`、`vec_dot`、`product_reduce`、`scalar_vec_dot` 五类 cell harness 已存在。
- master 的两张板源表当前分别有 119/86 行，disposition recon 为零未分类行。
- `perf_covered_metrics.py report` 当前可复算 `9/83`，且三源一致；这是当前快照，不是最终目标。
- 已有 deployed ggml 路径、手调/strong opponent、Scalar/no-V、IME/vendor、e2e 与多项负边界资产。
- 已有 tq1_0、iq2_xxs、VLEN256 grid 等部署改进，不得重新列为未完成。

## Actual Gaps

- 部分 harness 只覆盖子集：例如 K-quant vec_dot 主要覆盖 q4_K/q6_K，q2/q3/q5 oracle 和 k1 路仍有空洞。
- q4_K min-term 标准 harness 存在空心臂欠账（ISSUE-114）。
- master 同时被 recon 整表生成和 bench 外科更新，ownership/重建覆盖风险未关闭（ISSUE-098）。
- regime、默认 reader 路径、T-N qualification 与已有 ad-hoc measured 数据仍需正式化。
- 具名-X/近门中仍有结构 lever：K-quant vec_dot memory-stall/MLP、grid/codebook gather、nvfp4 compiler-call wall 等需分开判断。
- kernel 改进到 deployed/e2e 的传导需持续配对验证，不能只报 micro。

## Requirements

- 每个正式实验预注册 op/format/engine/regime、board、对手、正确性门、重复数与判定。
- 所有正式新数字走 `tools/bench/bench`，原始工件和 run-id 落三目的地。
- deployed ggml、代表性强对手、e2e 分账报告；不把任一条替代另外两条。
- 先正确性、对手身份和 compiler symmetry，再计时。
- 攻坚以反汇编/瓶颈和可读结构杠杆为起点；compiler/HW 脾气墙诚实登记。
- A 线重构前后保留同输入、同对手、同板的 paired regression。

## Acceptance Criteria

- [ ] runner/cell coverage 缺口按 op×family×board 明确并逐任务关闭。
- [ ] master ownership、reader 默认路径、regime key 与 T-N qualification 形成单一可复算流程。
- [ ] 至少一个代表性 strong-opponent hot-kernel campaign 完整走正确性→cold→objdump→入账。
- [ ] 至少一个 deployed ggml 路径有重构前后无回退或改进证据。
- [ ] 有意义的 kernel improvement 至少选一个进入 e2e 传导实验；wash 也登记。
- [ ] 输、赢、no-flip、VOID 与 wall 均有 run-id 和 disposition，不静默删除。
- [ ] 性能汇总由脚本现算，不把 task PRD 中的快照数字当长期 canon。

## Child Modules

1. [B1 measurement control plane](../07-20-b1-measurement-control-plane/prd.md)：master ownership、regime、reader、qualification/T-N。
2. [B2 bench cell coverage](../07-20-b2-bench-cell-coverage/prd.md)：K-quant/k1/scalar/product-reduce coverage 与空心臂修复。
3. [B3 K-quant vec_dot strong opponent](../07-20-b3-kquant-vecdot-strong-opponent/prd.md)：ISSUE-109 剩余结构杠杆。
4. [B4 dequant grid/codebook attack](../07-20-b4-dequant-grid-codebook-attack/prd.md)：代表性 grid/codebook 公式墙与脾气墙。
5. [B5 GEMM deployed path](../07-20-b5-gemm-deployed-path/prd.md)：shape、selection、deployed ggml 与强对手分账。
6. [B6 e2e transduction](../07-20-b6-e2e-transduction-regression/prd.md)：micro→deployed→e2e 与 A 线前后 paired regression。

每项分开，不建立一个无限性能 campaign；ISSUE-100 nvfp4 vec_dot 等未纳入项明确留作后续增量 task。

## Dependencies and Parallelism

- runner/cell qualification 与独立 hot-kernel attack 可以分文件并行。
- 新正式数字入 master 前，master ownership 与 row qualification 必须明确。
- A 线的 structural refactor 与 B 线不相交的现状测量可并行。
- A 线改变 selected decision/emission 后，paired regression 串行跟进。
- 真板任务按 board 窗串行打包；分析、harness、代码施工可并行。

## Out of Scope

- 追求所有 83 格一次翻绿。
- 用弱对手替代部署/代表性强对手。
- 只为论文表格补齐而扩大无价值 workload。
- 在线 tuner、runtime sparse/MoE。
- 在本父任务 PRD 固定未来所有具体数字。

## Technical Notes

- Measurement authority: `.trellis/spec/measurement/index.md`
- Master: `experiments/master/T3_master_rebuild.csv`
- Runner/cells: `tools/bench/bench`、`tools/bench/cells/`
- Metrics: `.trellis/scripts/perf_covered_metrics.py report`
- Relevant issues: ISSUE-098、ISSUE-099、ISSUE-100、ISSUE-107、ISSUE-108、ISSUE-109、ISSUE-112、ISSUE-114。
