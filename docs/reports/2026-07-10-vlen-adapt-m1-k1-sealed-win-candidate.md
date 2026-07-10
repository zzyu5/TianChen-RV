# [VLEN-ADAPT] M1 — k1 vl=16 首个 sealed Win 候选：VLEN-adaptivity 缺口 CLOSED（2026-07-10）

**VERDICT: A_vl16/hand-brick = 1.0850× · 95%CI [1.0828, 1.0884]（CI 不跨 1.0）→ 缺口关闭 → K1-SEAL 重开八门。** 报数 + 预注册判读，**不自宣 sealed**（裁决权主会话+用户，[NG-4]）。

**板** `ssh k1`（SpacemiT X60, VLEN256, clang-18 出货）。修法 = M0 判定的**纯 front-door capability-input**（emit 喂 VLEN256 fact → `deriveMinimumVLEN→256→half_lanes=16→numHalves=1→vl=16`），发**纯 S6 vl=16**（M1b loop-interchange 已回退隔离单因子）。**未写新宽核**。

## 核心：半宽→满宽把 0.750× 拉到 1.085×
K1-SEAL vl=8 = 0.750× vs hand-brick（半 VLEN256 宽）→ **vl=16 满宽 = 1.085×**（**+45% 相对，反超真出货 hand-brick 8.5%**）。

## 四比值（10 轮配对，pp128 prefill，core4-7 -t4，中位/IQR/自举95%CI）
| 比值 | median | 95%CI | 判读 |
|---|---|---|---|
| **A_vl16/Chand（vs hand-brick vl=16 = 真出货）** | **1.0850×** | **[1.0828,1.0884]** | **≥parity（CI 全 >1）** |
| A_vl16/Bstock（vs factory） | 1.0876× | [1.077,1.093] | ≥parity |
| A_vl16/Bq4kOFF（vs block-dot） | 2.9477× | [2.945,2.961] | ≥parity |
| decode tg32 A/Cbrick·A/Bstock | 1.0017×·0.9977× | 跨 1.0 | parity（预声明兑现） |

原始中位 A_vl16 15.656 / Cbrick 14.453 / Bstock 14.454 / Boff 5.309；基线逐一复现 K1-SEAL。自检 CV 0.45%、四变体 IQR<0.7%、**min-ratio A/Chand=1.0806 最差轮仍 >parity**、无测中劣化。

## ★关键：k1 = 编译器对称 = 干净 kernel-account（无编译器 caveat）
**k1 出货 ggml = clang-18** → A_vl16（我方 clang-18）vs hand-brick（clang-18）= **compiler-symmetric** → 这是**干净 kernel-account 赢，无 [CASE-COMPILER-ASYMMETRY] 系统账 caveat**（区别于 rvv M1b 的 clang-vs-gcc 绝对数）。且 vs 真 as-shipped（hand-brick，非 block-dot strawman）→ 打的是**真对手**。

## 五验（全 PASS，llvm-objdump-18）
① 符号 ② 我方 S6 核 vwmacc=**1120**（=vl8 2240 的一半、numHalves 2→1）· spill=2 ③ **vl=16 确认 `vsetivli zero,0x10,e32,m2`（0x10=16、vl=8 计数 0）** ④ banner `VLEN256 compiler-emitted S6 ENGAGED` ⑤ clang-18 zfh-native。

## byte-exact（8/8 配置，0-mismatch vs golden oracle）
int+norm × nr{16,64} × nc{256,512} × 3 seeds 全逐位同（int 0x5617ca5a…、norm 0x0d5363c8…）；vl=16 与 vl=8 同算术、仅宽 lane。**同时证 ABI-shim 映射正确**（错则 garbage）。genuine：no-preload 控制 FAIL（undefined symbol）→ 证跑的就是我 preload 的 clang 对象。k1 复原 stock 871169a0 PRE==POST。

## 八门状态（k1 vl=16 e2e sealed Win 候选，主会话初评待终审）
| 门 | 状态 |
|---|---|
| ① 字节精确 | ✅ 8/8 byte-exact vs golden oracle |
| ② VLEN-flip lit | ✅ vl=8@128 / vl=16@256（gate2 lit + 本次 vl=16 实证） |
| ③ 双板 objdump | ✅ k1 vl=16 sealed（vwmacc 1120/vsetivli 0x10）+ rvv vl=8 S6 sealed |
| ④ micro ∧ e2e | ✅ **e2e 1.085× ≥parity（★曾失败的 e2e 门现 MET）** + micro S6 |
| ⑤ 双板都验证 | ⚠ k1 e2e 干净赢（对称-clang）；rvv e2e = M1b 翻正（schedule 2.47×，但绝对含系统账 caveat）→ 双板 e2e-win 口径待终审定 |
| ⑥ 实验纪律 | ✅ 自检 SOP/N=10/配对/IQR<0.7% |
| ⑦ selector 能力键 | ✅ vl←VLEN capability fact（numHalves 派生非常量） |
| ⑧ 措辞门 | ✅ 双账本（k1 对称-clang = kernel-account clean）+ NG-4 |

## 残留 caveat（供终审）
1. **ABI-shim 部署**：现发射器 ABI `(nr,bs,n,s,vx,vy,nc)` ≠ 部署 A-PATCHED ggml（90d454da 期）期望的 board-ABI；根因 = ABI 导出序 5f194cbd 后（SEL-1 期）变过、**非 M1b**；包零逻辑转发 shim（byte-exact 背书映射、每 mul_mat 一次转发可忽略）。**若终审要求纯 LD_PRELOAD drop-in → 需 ABI 对齐**（重建 A-PATCHED 到现 ABI 或建 90d454da 期发射器）。
2. **纯 S6 vl=16**：M1b loop-interchange 是正交第二轴、**未叠加**（可后续叠测，理论上 k1 也可再压 backend-idle）。

## 净口径
**首个 sealed Win 候选兑现**：k1 vl=16 S6 kernel，byte-exact 正确、**1.085× 反超真出货 hand-brick、编译器对称（干净 kernel-account）、CI 不跨 1.0**。VLEN-adaptivity 缺口 = 纯 capability-input 修复（M0 预判 GREEN 兑现）。**sealed Win 登记 = 主会话+用户裁决权**（必问，[NG-4] 不自宣）。
