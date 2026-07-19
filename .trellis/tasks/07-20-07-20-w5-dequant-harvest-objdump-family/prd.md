# PRD — W5 B线 dequant 真向量发射器收割 + objdump 分拣族级杠杆全族翻

## 权威 = r5.1 goal
B线「**vs 部署路径 ≥0.8 即赢·不问强弱**」+「dequant 真向量发射器收割 36 格 + objdump 分拣派侦察找族级杠杆全族翻」。修缺口不撤·遇挫不退·**不停在记边界**。

## 做（vs 部署 ≥0.8 入账·objdump 前置分拣·族级杠杆全族翻）
1. **dequant 真向量发射器收割**：master 现 dequant 行(24 行·grid 族已翻 iq3_xxs 1.39/iq3_s 1.71/iq2_xs 3.88 等)。**逐格 vs 部署路径板测**（`ssh rvv`/`ssh k1`·对手 = deployed `dequantize_row_<fmt>`·编译器对称）·byte-exact 前置·**≥0.8 即赢入账**（不问强弱）。收割未测/pending 格·目标 36 格覆盖。
2. **objdump 分拣前置**（省板时·B1 分拣标配）：出表附「对手结构 vs 我方结构」diff（HW-gather/vsext/strip/unroll）·板窗只验证不探测。
3. **族级杠杆**：objdump 找出族共性杠杆（如 grid 族 gather-free 装配·nibble 族宽化·standalone dequant autovec 抽签）→ 一个杠杆**全族翻**（律3 全族受益·非逐格凿）。<0.8 记具名-X + objdump 族级墙型（compiler-behavior vs 微架构·别沉底）。
4. 更 master（走 recon·非手改）。

## 门（≥0.8·族级·成色诚实）
- byte-exact 先于计时(dequant mism=0·CORE==PROD)· **vs 部署 ≥0.8 即赢入账**(不问强弱·便宜档标 opp-immaturity·禁称硬赢)· 编译器对称 · 2-seed cold · objdump 分拣前置(对手结构 diff)· 族级杠杆(非逐格)· 0 造数 · 🔴无 inline-asm/钉死调度绕 clang(脾气墙诚实记·跑不过打第一块能赢的)· 未 git commit(板数据 experiments·master recon)。
- 🔴 禁碰 A线(W1/W2/W3)·iq2 **vec_dot** 体(W4)。**W5 只碰 dequant 真向量发射体**(ForwardElementwise dequant / GridCodebook 的 emitDequantizeRow*·非 vec_dot)·worktree 隔离·串行集成。板: `ssh rvv`/`ssh k1`。

## 汇报（首节: dequant vs 部署 ≥0.8 入账 x 格 / 36）
逐格 vs 部署 ratio + ≥0.8 入账几格 + <0.8 具名-X + objdump 族级杠杆(找到几条·全族翻几族) + 墙型(compiler-behavior/微架构)。禁新建分析文档·结论进 final message。
