# B 线只读审计

## Verified Current State

- official runner 与三目的地已建，self-test 11/11。
- 现有 cell harness：dequantize_row、vec_dot、gemm_tile、product_reduce、scalar_vec_dot。
- T3 两张板源表 disposition recon 为 119/119、86/86 全分类；`perf_covered_metrics.py report` 当前快照为 9/83 且三源一致。
- deployed ggml、代表性 strong opponent 与 e2e 都已有资产，三者证明不同性质，必须保留并分账。
- 当前主要债不是“没有性能”，而是部分 cell/board/format coverage、T-N/qualification、master ownership，以及少量具名结构杠杆。

## Recommended Modules

1. B1：measurement control plane——资产矩阵、master ownership、regime/reader 默认、qualification/T-N。
2. B2：bench cell coverage 与 correctness hardening——q2/q3/q5 K-quant vec_dot、k1、q4_K min-term、scalar/product_reduce 路由。
3. B3：K-quant vec_dot 强对手攻坚——承接 ISSUE-109，先做跨 super-block MLP/真实瓶颈，不复试已证伪杠杆。
4. B4：dequant grid/codebook 性能边界——承接 ISSUE-107/100，区分公式墙与 compiler/gather 脾气墙。
5. B5：GEMM/shape/deployed ggml 路——候选、合法性、实际派发与 strong opponent 分列。
6. B6：e2e 传导与 A 线重构前后 paired regression——正向收益和 wash 同权。

## DAG

~~~text
B1 → B2 ─┬→ B3 ─┐
         ├→ B4 ─┼→ B6
         └→ B5 ─┘
~~~

B3/B4/B5 的分析和代码可并行；同一真板由单 owner 串行打包。B6 等待至少一个候选族闭环和 A 线对应切片落地。

## Guardrails

- 已有合格数字不无理由重跑。
- 新正式数字必须走 runner + run-id + 三目的地。
- strong opponent 不替代 deployed ggml，micro 不替代 e2e。
- scalar 证据保持 enablement/NON-Win 边界。
- 输、no-flip、VOID、wall 和 e2e wash 全部保留。
