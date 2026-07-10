# Sealed Win 登记册（canon · 登记=用户裁决权、不可逆；措辞逐字锁定）

> 登记纪律：sealed Win 登记是 canon/不可逆动作（覆盖 Win 登记 = 决策卡必问）。每条 Win 措辞逐字锁定（八门措辞门口径），随附已知限制为登记档**不可分割部分**。加固/升级/维持措辞按 follow-up 结果由用户裁。

---

## Win #1 — **Win-K1-VLEN**（首个 sealed Win · 登记 2026-07-10 用户裁定）

**措辞（逐字入册，八门措辞门口径）**：
> 在 K1 板（VLEN256）、q4_K、e2e prefill：由 VLEN 能力事实驱动的满宽（vl=16）机制构造 kernel，对该板真实出货 hand-brick（clang 对称）交付 **1.085×**（中位数，95% CI [1.0828, 1.0884]，10 轮配对），byte-exact，部署五验在案。

**随附已知限制（登记时；★加固后状态见下）**：
1. ~~⑤双板门开放~~——**[FU-2] RESOLVED（2026-07-10）**：rvv col-outer vs block-dot **1.336× CI[1.329,1.344]**，block-dot 编译器不敏感（1.002×）→ **编译器对称干净 kernel-account 赢 → ⑤升双板**。
2. ~~ABI shim 在途~~——**[FU-3] RESOLVED**：fixture 路径原生符号 ABI 直配正门、**零 shim**，win 纯态成立。
3. ~~未叠加 col-outer~~——**[FU-1] RESOLVED**：col-outer 在 k1 无显著 margin（正交、rvv-specific），vl=16 满宽即承载全部 1.085×。

**★加固升格（RATIFIED 2026-07-10 · 用户定稿逐字入册 · 八门措辞门口径 · 加固报告 `docs/reports/2026-07-10-Win-K1-VLEN-加固报告.md`）**：
> **Win-K1-VLEN（升双板 · 能力驱动执行方法之双板验证）**：同一模板（能力事实→瓶颈识别→杠杆选择）在两类硬件上分别交付**编译器对称的 kernel-account e2e prefill 赢**——**K1/VLEN256**：lane-width 杠杆（vlen 事实→vl=16），vs **该板真实出货 hand-brick**，**1.085×（CI [1.083,1.088]）**；**RVV/VLEN128**：schedule 杠杆（内存层级事实→col-outer），vs **clang-symmetric block-dot**（构建对称对手；该板 as-shipped 为 gcc 构建，block-dot 已证编译器不敏感 1.002×），**1.336×（CI [1.329,1.344]）**。两板均 byte-exact、部署身份五验、置信区间不跨 parity。定性 = **方法跨板泛化（非同一 kernel 双板复制）**；正交性证据：col-outer 于 K1 无收益（1.002×，跨 1.0）——杠杆各有辖区，由瓶颈形状决定。
> **成色**：此 Win = **C1（模板协议）与 C3′（产出质量）的共同性能实证**；论文 headline 与 q4_0 5.9×（系统账路由赢）**并列陈述、账本分明**。ABI 纯 drop-in（fixture 路径零 shim）。**3 caveat 全 RESOLVED**。

**证据指针**：`docs/reports/2026-07-10-vlen-adapt-m1-k1-sealed-win-candidate.md`（1.085× CI + 五验 + byte-exact 8/8 + genuine + k1 复原）· `docs/reports/2026-07-10-k1-seal-e2e-transduction.md`（K1-SEAL 0.750× 基线 + hand-brick 对手身份）· `experiments/active/vlen-adapt/vl16_static_account.md`（M0 GREEN 判定）· 板证据 `k1:/tmp/tcrv_k1_vlen_adapt/`。

**成色/意义**：修法 = **纯 capability-input**（无新手写核 / 无格式名分派 / 无搜索）→ 本 Win 同时是 **C1/C3′ 核心主张的性能实证**（能力事实驱动构造）。论文 headline 候选**并列** q4_0 5.9×（系统账路由赢）——**成色不同、分开陈述**（Win-K1-VLEN = kernel-account 满宽机制构造 clean 对称赢；5.9× = 系统账 routing 白嫖）。

**八门状态**（登记时）：①byte-exact ✓ ②VLEN-flip ✓ ③双板 objdump ✓ ④micro∧e2e ✓ ⑥纪律 ✓ ⑦selector ✓ ⑧措辞 ✓ ｜ **⑤双板 e2e-win 开放**（k1 clean / rvv 待 [FU-2] clang-symmetric）。

**加固队列**（[FU-1/2/3] 完成即出"Win-K1-VLEN 加固报告"，措辞按结果升级或维持）：
- [FU-1] vl16 × col-outer 叠加测（k1，正交性 + margin 上探）
- [FU-2] rvv clang-symmetric 对手构建 + 补测（定⑤双板 kernel-account；正则升双板、负则维持单板）
- [FU-3] ABI 纯 drop-in 对齐（消 shim）
