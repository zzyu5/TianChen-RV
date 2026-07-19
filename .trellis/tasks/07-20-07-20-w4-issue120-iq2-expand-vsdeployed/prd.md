# PRD — W4 B线扩 iq2 面: ISSUE-120 已修→iq2_xs/iq2_s vec_dot vs 部署≥0.8 收割

## 权威 = r5.1 goal
B线「**vs 部署路径 ≥0.8 即赢·不问强弱**」+「修 ISSUE-120 扩 iq2 面」。ISSUE-120（iq2_xs/iq2_s vec_dot VLEN256 byte-broken）**已修**（`dcbace2c9`·W2 独立复验双板 byte-exact·k1 mism 512→0）——**correctness 阻塞已除·现收割 iq2 面**。

## 做（vs 部署路径板测·≥0.8 入账·遇挫不退·不停在记边界）
1. **iq2_xs/iq2_s vec_dot @k1 vs 部署对手**：现 byte-exact 双板（vslidedown 修·48 op·CORE==PROD md5 iq2_xs b4439d95/iq2_s 8f792e89）。板测 `ssh k1` vs **部署路径对手**（ggml deployed iq2_xs/iq2_s vec_dot·编译器对称 clang-18）·2-seed cold·**vs 部署 ≥0.8 即赢入账**（哲学锁①·不问强弱·成色注记只为论文）。
2. **扩 iq2 面**：iq2_xs/iq2_s/iq2_xxs（+ 相邻 grid vec_dot 格）逐格 vs 部署·byte-exact 前置·≥0.8 入账·<0.8 记具名-X（objdump 对手结构·不停在记边界·找机制杠杆）。
3. **rvv 侧**同格 vs 部署（VLEN128）·补齐双板。
4. 更 master（走 recon·非手改）。

## 门（≥0.8·成色诚实）
- byte-exact 先于计时（vec_dot mism=0·CORE==PROD md5 已在）· **vs 部署路径 ≥0.8 即赢入账**（不问强弱·便宜档标 opp-immaturity 涨地盘不涨成色·禁称硬赢）· 编译器对称 clang-18 · 2-seed cold · 0 造数(0 样本不编数) · 无 inline-asm · 未 git commit（板数据/FINDING 走 experiments·master 走 recon）。
- 🔴 禁碰 A线(W1/W2/W3)。**iq2 vec_dot 发射体在 GridCodebook·与 W5(dequant 收割)可能重叠 GridCodebook**·W4 只碰 vec_dot 体(emitIQ2XS/IQ2SSuperBlockGridBody)·不碰 dequant 体·worktree 隔离·串行集成。板: `ssh k1`/`ssh rvv`。

## 汇报（首节: iq2 面 vs 部署 ≥0.8 入账 x 格）
iq2_xs/iq2_s/iq2_xxs 各 vs 部署 ratio(k1+rvv) + ≥0.8 入账几格 + <0.8 具名-X 几格(objdump 对手结构) + byte-exact 确认。禁新建分析文档·结论进 final message。
