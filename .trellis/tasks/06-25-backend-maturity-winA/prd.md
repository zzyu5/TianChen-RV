# backend-maturity: 丰富 Win-A(能力驱动 lowering)+ 填满自查表

## Goal
让**后端主线真正成熟化**:把 Win-A 从"从 2-3 个 LMUL 档里挑一个 + capability-blind cost model"提升到**真正能力/资源驱动的算子 lowering**(这才是后端优化层的体现)。并行把**自查表(`KERNEL-优化自查表.md`)的空格/问题项尽量测满**。N1/N2/N3、Win-A、Win-B 框架**不变**。

## 关键定位(本会话掰清的,不可回退)
- 我们对标 **Triton 的后端**(拿给定算子→自动生成该硬件最优码),**不是 TVM**(自动搜调度),也**不是选算法**。
- **后端贡献** = 能力驱动的 lowering:同一算子,按能力 lower 成不同硬件码(VLEN→LMUL、有IME→IME、自动向量化/选指令/流水)。**Win-A 就是这层的旋钮**——要丰富的是这个。
- **前端贡献(另算一栏,有价值但不是后端 novelty)** = repack / 改算法结构 / option-2 选算法。**不再塞进编译器当 N3 卖**;option-2 越界(伸进 ggml 加载布局)已 demote。
- 边界规矩:后端 lower"给定布局的给定算子";选/改布局是前端/框架。

## Requirements
### 主线 A:丰富 Win-A 的水平(后端成熟化)
- A1 现状(maturity 调查已确认):4 个旋钮(integer_core_lmul / multi_block_factor / strip_elision / acc-LMUL),默认走 **capability-blind 的 static cost-model argmin**(只看 1 个 Zvl128b bit);offline 实测闭环存在但非默认;K-quant block-dot **不在** schedule-descriptor flow(Win-A 靠手 attr,没被 gearbox 自动选)。
- A2 目标(候选,设计阶段定):(a) **cost model 真能力/资源感知**(喂 VLEN/ELEN/vreg-budget/elem-width,不再 1 bit);(b) **更多 kernel 接入 gearbox 自动选**(K-quant block-dot 等,从手 attr → 自动按能力 stamp);(c) 让"measured > static"成默认或真闭环;(d) 增/激活 lowering 轴。
- A3 每个改动要 byte-exact 门 + 两板(rvv VLEN128 / k1 VLEN256)+ trellis-check。

### 主线 B:填满自查表
- 空格/问题项实测:IQ/FP4/ternary 的 k1 micro、q8_0 k1 选择 e2e、q4_1 repack prefill 旋钮、q4_0 repack k1 GEMM、q5_0 repack k1、q4_K repack k1 oracle、forward-pass micro、q1_0 micro。
- 填完后更新表 + 前端/后端表述。

## 纪律(硬约束)
- **板子谨慎并行**:rvv 易掉线/出 bug;**同一块板 bench 串行**(干净数 + 不撞 build);k1/rvv 可各一个 campaign;**绝不一次几十线程压板子**。
- 主 session 不改 lib/ 代码——派 trellis-implement + trellis-check;host build 撞 tcrv-opt 要串行。
- 硬件/性能主张要真 ssh 证据 + byte-exact;over-claim 立即纠。
- 杀共享 k1/rvv 进程需用户授权。

## Out of Scope
- option-2 C3 / repack 选算法(已 demote 为前端,不在本任务后端范围)。
- 自动调度搜索(TVM 路)。

## 当前状态
- 自查表已在仓库根 `KERNEL-优化自查表.md`(含成熟度定位段)。
- maturity 调查 3 份在旧任务 archive(`archive/2026-06/06-22.../artifacts/maturity-*.md`)。
