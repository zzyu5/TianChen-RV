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

---

## Win #2 — **q4_K@k1 e2e prefill 1.101×**（首个 e2e beat-hand-brick · 旗舰入册 2026-07-16 · PR-14 RESOLVED）

**措辞（逐字入册，八门措辞门口径）**：
> 在 K1 板（VLEN256·出货 clang-18）、q4_K、**e2e prefill**：我方 HEAD-live emitc GEMM（byte-verified drop-in·nbad=0·md5 `9e057adb`·deployed==proven·greedy token-identical）对该板**真出货手调 GEMM** `ggml_gemm_q4_K_16x1_q8_K`（compute-bound·objdump 767 insn / 80 vwmacc·M=128 权重复用）交付 **1.101×**。倍数中等（Amdahl 稀释 kernel 1.187× → e2e 1.101×），但对手 = **最强对手类（真手调 hand-brick）**、传导为**真 e2e 传导**（非 wash·非 routing 白嫖·净新发射体 body-swap）= **首个 e2e beat-hand-tuned**。

**机制归因（★净新增·正判据）· [DISCRIMINATOR-OPPONENT-BOUND-TYPE]**：
> kernel 算力赢**传导 e2e IFF 对手 compute-bound·washes IFF 对手 memory-bound**。判别键 = **对手 bound-type**（由 相(M)×板×出货编译器 对该 kernel 计算路 autovec 质量共同决定·**非格式名·非 kernel 倍数大小**）。**正交** compiler 轴：同格式 q4_K @k1(clang)=1.101× 传导 vs @rvv(gcc)=0.334× gcc-death → 两轴分立归因。三点隔离 CONFIRMED：① 同板同编译器同核族同相·唯一差=格式（q4_0@k1 decode 0.857× WASH vs q5@k1 decode 1.97× 传导）② 同格式跨板（q5@rvv wash vs q5@k1 传导）③ 相轴（q4_K@k1 prefill 传导 vs decode isolation 0.995×）。

**成色 / 边界（登记档不可分割部分）**：
1. **perf-covered 硬冻结不动 = 9/83**：q4_K@k1 早已 perf-covered 绿（kernel 账 · Win-K1-VLEN 血脉）；本条 = **成色升级**（kernel 账 → e2e-transduced beat-hand-brick）·**非新格**（any-board 规则·收口令〇.1 · 头条口径不增）。
2. **倍数中等由 Amdahl 稀释**：kernel 1.187× → e2e 1.101×（q4_K GEMM 占 prefill 相内时间比例有限）·非"大倍数"·成色立于「最强对手类 × 真传导」非倍数。
3. **计数 ≠ 强赢**：本役真硬赢强手调锚点 = 2 verified hand-brick（q4_K/q2_K@k1 GEMM prefill·byte-verified）+ q5@k1 C1 部署 + 本条 q4_K@k1 e2e（首个 e2e beat-hand-brick）。

**证据指针**：`docs/reports/2026-07-16-G8-paper-evidence-freeze.md`（§净新增①·三点隔离 + 1.101× 卷）· `experiments/active/g8-stage3-attack/`（HEAD-live emitc drop-in·md5 `9e057adb`·nbad=0·deployed==proven）· 血脉 Win #1 Win-K1-VLEN（同 q4_K@k1·kernel 满宽机制）。

**八门状态**：①byte-exact ✓（nbad=0·md5 verified）②VLEN-flip ✓ ③objdump 对手身份 ✓（vendor 767 insn/80 vwmacc）④micro∧e2e ✓（1.187 kernel ∧ 1.101 e2e 真传导）⑤ e2e-win ✓（k1·greedy token-identical）⑥纪律 ✓（deployed==proven·body-swap）⑦selector ✓ ⑧措辞 ✓。
