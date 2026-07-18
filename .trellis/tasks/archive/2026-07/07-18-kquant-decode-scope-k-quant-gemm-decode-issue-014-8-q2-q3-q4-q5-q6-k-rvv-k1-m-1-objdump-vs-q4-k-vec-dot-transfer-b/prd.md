# PRD · K-quant gemm-decode 真成本中心诊断（ISSUE-014·B 线判据侦察）

> **权威** = ISSUE-014（K-quant decode 成本中心留白·fold@M=1 经四腿证非物理地板·真成本中心待诊断）+ 《行动书 r3》B 线（**判据=公式墙还是脾气墙**·**先 objdump 对手**）。
> **性质** = 只读侦察（**objdump 对手 + 真成本中心定位 + q4_K vec_dot 弧线是否 transfer**·persist research/·零代码改·零板攻·objdump 随便读·不测速度省板窗）。

## 一、背景
K-quant gemm-decode 8 格（`gemm_tile` q2_K/q3_K/q4_K/q5_K/q6_K @rvv 及 k1·M=1 regime）现 = 具名-X·"对手结构优势具名·成本中心留白·真成本中心待诊断"（ISSUE-014·非物理墙·[§三.15]）。q4_K **vec_dot** 弧线刚证真墙 = memory-stall floor（m2 register-pressure·m1-宽度 lever）——**decode（gemm M=1）是否同墙？**

## 二、要回答的（B 线判据）
1. **objdump 对手**（decode 格的部署对手·native-vec block-dot / repack）：它的成本中心结构（向量化了什么·访存模式·M=1 怎么摊销/不摊销）。
2. **我方 decode 真成本中心**（M=1·fold 非物理地板已证·那真成本中心是什么）：objdump 我方 decode leaf + 与 q4_K vec_dot 弧线对比（同 memory-stall floor？还是 fold@M=1 无行摊销的不同墙？）。
3. **★q4_K vec_dot 弧线是否 transfer**：vec_dot 的四次定格（aux8→serial→scalar-min-term→memory-stall→m2 register-pressure·lever=m1-宽度多流）是否适用 decode？decode M=1 单行·无行摊销·成本中心可能不同。
4. **★判据裁定（逐格或分组）：公式墙〔可键控·必攻·施工入口〕 vs 脾气墙〔micro·走三步登记边界〕**——带证据。注意成色：便宜档格（vs generic·禁称硬赢）先分离·真强手调对手格才是攻坚对象。

## 三、产出（research/·非代码·非 spec）
1. **逐格对手成本中心**（objdump·8 格·便宜档 vs 真手调分离）。
2. **我方 decode 真成本中心**（objdump·与 q4_K vec_dot 弧线对比·同墙/异墙）。
3. **★判据裁定**（公式墙 → 施工入口 + 是否复用 vec_dot 的 m1-宽度/STACK lever；脾气墙 → 三步走完的边界登记建议）。
4. **成色分离**（8 格哪些便宜档〔禁称硬赢·非攻坚对象〕·哪些真强手调〔攻坚对象〕）。

## 四、纪律
- **只读·零代码改·零板攻**（objdump 随便读·**不测速度**·省板窗）·persist research/。
- **B 线判据是产物**（公式墙必攻/脾气墙走三步·非赢没赢）· 便宜档分离（成色）· 禁外推（逐格或有据分组·q4_K vec_dot 弧线 transfer 须证非假设）。
