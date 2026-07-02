# experiments/ —— 结果表集(空模板,按快照由 CI 填)

> 结构与判定规则以 `docs/TianChen-RV_实验总纲v1.md`(实验总纲)为准;详细表-schema 附录见 `papers/TianchenRV/备份-迭代文档/TianChen-RV_实验方案与结果表集v1.md`。**本目录只放空模板;一切数字随快照进 CI,零现值。**

## 表 → 主张 映射(读 T3 前必读 T-N 与对手探针)

| 表 | 证的主张 | 性质 |
|---|---|---|
| `T-N_noise_floor.csv` | 效应判定地基(无地板无判定资格) | 板相关·测量 |
| `T0_kernel_census_sixstate.csv` | 成熟编译器(六态普查) | 板无关·仓库态 |
| `T1_C1_structural_conjunction.csv` | **C1** 零分支 + schema 复用(二值) | 板无关·结构 |
| `T1b_failclosed_runtime.csv` | C1 运行期半边(fail-closed) | 结构 |
| `T1d_dual_instance_same_schema.csv` | **C1 一图流**(同 schema 双实例,论文 Table 1 候选) | 板无关·结构 |
| `T1c_external_reproduction.csv` | C1 可复制协议(M4) | 结构 |
| `T2_C2_ledger_marginal_cost.csv` | **C2** 边际成本曲线 | 板无关·源码 |
| `T3_A_board_A_rvv1.0_vlen128.csv` | **C3′ + 成熟**(板 A 测量) | 板相关·测量 |
| `T3_B_board_B_rvv1.0_vlen256.csv` | **C3′ + 成熟**(板 B 测量) | 板相关·测量 |
| `T3p_pattern_ablation.csv` | C3′ 模式逐条消融 | 板相关·测量 |
| `T3m_migration_criterion.csv` | C3′ 迁移性(双板 diff=0) | 板无关·结构 |
| `T4a_attribution_samples.csv` | C3′/N3 可归因(选择日志) | 结构+测量 |
| `T4b_selector_ablation.csv` | N3 选择价值(gated on 先验层) | 测量 |
| `T5a..d_ime_*.csv` | **C1/N2 佐证 + P7**(IME 卷宗) | 见实验总纲 §3 |
| `T6_e2e_phase_split.csv` | beat(唯一舞台,过 PERF-1 八门) | 板相关·测量 |
| `T7_coverage_burndown.csv` | 成熟编译器(四指标 + 烧减,论文 Fig.1) | 板无关·仓库态 |
| `T8_winloss_gap_ledger.csv` | C3′ 负例叙事 + 缺口关闭环 | 登记 |

## 格 schema(进 CI 的最小单元,两型)
- **测量格** = `{值, 单位, 状态∈{measured|stale|board-pending|open|n_a}, 环境指纹, 快照, 工件指针, 对手指针, 配对会话}`。
- **结构证明格** = `{判定 pass/fail, CI 日志, 快照}`。
- **纪律:** `open`(原 presumed)永不可被正文引用;指纹任一分量变 → 同指纹格自动 `stale`;跨会话不比;强/弱义构造由 provenance 清单机检([L-8]);vs-framework 无对手探针工件即 INVALID。

## Win 登记阶梯（[L-7] 定案 · 阶梯与对手类一一对应）
- **Win-A** = 能力形状旋钮(narrow↔wide)vs naive —— **sanity**。
- **Win-S** = 两轴 kernel 赢 vs 真标量 + 指令级核验朴素向量 —— **sanity 类**。*alias:即历史一度误标 "Win-B" 的两轴 datapoint;改名归 sanity,登记簿(T8)留 alias 注记、不改史。*
- **Win-B** = 算法/布局 vs **框架自家 kernel** —— **kernel 级贡献**;**B1/B2 下标 = 该板出厂路径探针结果**(B1=出厂 block-dot / B2=出厂 repack);**在出厂走 repack 的板上打赢其 block-dot ≠ Win-B**,只是 `algorithm-matched` 诊断增量,**不入 T8 登记簿**。
- **Win-C** = 相级 e2e 赢(过 [PERF-1] 八门后 prefill/decode 分相 vs 出厂 ggml)—— **相级贡献**。
- **A/S/B 绝不入 beat 语境([NG-4]);任何落败/对比结论反汇编钉死前不得书面引用。**

## 板
- **板 A** = RVV1.0 VLEN128(无 repack → block-dot 是唯一 ggml 对手)。
- **板 B** = RVV1.0 VLEN256 + 整数矩阵扩展(部分格式 repack→GEMM 绕过 block-dot,对手身份随格式变)。
- **板 C**(RVV0.7.1)只进 `T1b` 的 conflicts 见证一行,永不进性能表。

## 状态(现状见 `docs/TianChen-RV_执行总纲v2.md`)
C1 结构证据今天最强、板无关;C2 首点可复算;C3′/成熟/beat 全部 gated on 待建工具(T-N/对手探针/provenance 清单/归因 JSONL/schema.def/ledger/coverage 脚本)+ 硬件重测。这批工具即证据线(执行总纲 §10 TOP-10),一次建设两处收益。
