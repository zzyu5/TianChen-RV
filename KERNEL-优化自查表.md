# TianChen-RV 自查表 v4(2026-06-25)

> 表格只放数字,没做就空着。所有解释在最后单独成段。

## 图例(看表前先读这个)

- **数字 = 我们的 kernel 跟 ggml 官方 kernel 的速度比**:`>1` 我们更快(赢),`<1` 我们更慢(输)。例:`1.32×` = 我们快 32%;`0.58×` = 我们只有 ggml 的 0.58 倍速(慢)。
- **Win-A 列的数字 = 旋钮开 vs 关**(同一个 kernel,把向量寄存器从窄调到宽,快多少)。例:`1.30×` = 调宽后快 30%。
- **空格 = 这个板子/这一项没做**(如实留空,不是不适用)。
- **N/A = 本质不适用**(这个 kernel 没有这个轴可做)。
- **Win-A·e2e 的数字 = 旋钮开 vs 关传导到 llama 整体多少**(不是 vs ggml)。
- 板子:**rvv** = SG2044 VLEN128;**k1** = X60 VLEN256;**ime** = X60 的矩阵单元。

> **★内存墙——诚实版(2026-06-25 配对实测,纠我自己一次过度声明)**:我先前说「decode 算力 win 被内存墙washes成 1.0」过于绝对,但随后说「你完全对、内存墙不是绝对的」**又过头了**。配对实测(k1,8 对)真相更细:q4_K 的 m1 vs mf2 旋钮在 **decode 传导 1.221×、prefill 1.258×**(极稳,range 0.003)——**但这是「把 q4_K repack 关掉、强制走 per-row vec_dot」的 regime**;关掉 repack 后 decode 也变成算力受限(所以传导是预期的,不构成对内存墙的反驳)。**而出厂默认 VLEN256 会把 q4_K repack 成 GEMM、完全绕过这个 kernel → 这个旋钮对出厂 decode 的 e2e 影响 = 0**。诚实判定:**既没证实也没推翻内存墙**;真正的内存受限判官(8B 模型 @VLEN128 on rvv)还**没跑成**(rvv 板子掉线 blocked)。你的直觉对**部分** kernel 成立(q4_0 repack 改了内存搬运 → e2e 真 2.6×),但 q4_K 这种 block-dot 旋钮在出厂路径上被 repack 绕过了。

---

## 表 1:Block-dot 算子(我们 emit 的算法和 ggml 一样,所以只有「调旋钮(Win-A)」和「跟 ggml 官方 PK」)

| 算子 | Win-A·rvv | Win-A·k1 | Win-A·e2e | vs-ggml·rvv | vs-ggml·k1 | vs-ggml·e2e | ime |
|---|---|---|---|---|---|---|---|
| q4_K | 1.26× | 1.30× | 1.22×decode/1.26×pp(注9) |  | 0.72× |  | N/A |
| q6_K |  | 1.15× |  |  | 0.52× |  | N/A |
| q3_K |  | 1.15× |  |  | 0.55× |  | N/A |
| q5_K |  | 1.30× |  |  | 1.00× |  | N/A |
| q2_K | N/A | N/A |  |  | 1.02× |  | N/A |
| q8_0 | 1.08×(注12) | **1.17×(注12)** |  |  |  |  | N/A |
| q4_1 | 1.03×(注1) | 1.11×(注1) |  |  |  |  | N/A |
| q4_0 | 1.01×(注1) |  |  |  |  |  | N/A |
| q5_0 | N/A | N/A |  |  |  |  | N/A |
| q5_1 | N/A | N/A |  |  |  |  | N/A |
| iq1_s | N/A | N/A |  | 0.43× | 0.36×(注10) |  | N/A |
| iq1_m | N/A | N/A |  | 0.17× | 0.24×(注10) |  | N/A |
| iq3_xxs | N/A | N/A |  | 0.12× | 0.20×(注10) |  | N/A |
| iq3_s | N/A | N/A |  | 0.20× | 0.21×(注10) |  | N/A |
| iq2_xxs | N/A | N/A |  | 0.52× | 0.66×(注10) |  | N/A |
| iq2_xs | N/A | N/A |  | 0.29× | 0.52×(注10) |  | N/A |
| iq2_s | N/A | N/A |  | 0.53× | 0.66×(注10) |  | N/A |
| iq4_nl | N/A | 1.27×(注14) |  | **1.32×** | 0.84→**1.08×**(注14) |  | N/A |
| iq4_xs | N/A | N/A |  | **1.28×** | 1.04×TIE(注11) |  | N/A |
| mxfp4 | N/A | 1.28×(注14) |  | **1.21×** | 0.80→**1.01×**(注14) |  | N/A |
| nvfp4 | N/A | N/A |  | N/A(注2) |  |  | N/A |
| tq2_0 | N/A | N/A |  | 0.60× | 0.36×(注11) |  | N/A |
| tq1_0 | N/A | N/A |  | 0.44× | 0.36×(注11) |  | N/A |
| q1_0 | N/A | N/A |  | 0.033×(注13) |  |  | N/A |

---

## 表 2:Repack 算子(我们自创了一个跟 ggml 不同的算法,所以有真正的 Win-B)

> Win-A = 调 strip/宽度旋钮;Win-B = 我们的新算法 vs ggml 官方算法。

| 算子 | Win-A·rvv | Win-A·k1 | Win-B·rvv | Win-B·k1 | Win-B·e2e·rvv | Win-B·e2e·k1 |
|---|---|---|---|---|---|---|
| q4_0 repack(decode) |  | 1.48× | 1.22× |  | **2.60×** | 0.74× |
| q4_0 repack(prefill) | 1.30× |  | 6.36× |  | **5.68×** |  |
| q4_1 repack(decode) |  |  | 正确性✓ |  | 上游卡(注3) | 上游卡 |
| q4_1 repack(prefill) | 1.24×(注4) |  | 正确性✓ |  | 上游卡 | 上游卡 |
| q8_0 repack(decode) |  | 1.95× | 0.65× |  | 推定平 | 未测(注5) |
| q4_K repack(decode) |  |  | 0.55× |  | 推定平 |  |
| q4_K repack(prefill) |  |  | 0.74× |  | 推定输 |  |
| q5_0 repack(decode) |  |  | 0.77× |  |  |  |

---

## 表 3:前向算子(softmax 这一系列)—— 都能生成,但**一个性能都没测**

| 算子 | 能否生成 | Win-A | Win-B | micro | e2e |
|---|---|---|---|---|---|
| softmax | ✓ | N/A | N/A | 空 | 空 |
| silu | ✓ | N/A | N/A | 空 | 空 |
| rms_norm | ✓ | N/A | N/A | 空 | 空 |
| rope(rope_norm) | ✓ | N/A | N/A | 空 | 空 |
| scale | ✓ | N/A | N/A | 空 | 空 |
| quantize_row_q8_0 | ✓ | N/A | N/A | 空 | 空 |

---

## 表 4:IME 矩阵算子(ime,N2 第二 family)

> 我们生成的是 IME 的**单条 MAC 指令**(leaf),不是完整 GEMM。

| 算子 | 能否生成 | k1 硅片正确性 | perf·micro | perf·e2e |
|---|---|---|---|---|
| vmadot(有符号×有符号) | ✓ | 对 16/16 |  |  |
| vmadotu(无×无) | ✓ | 对 16/16 |  |  |
| vmadotsu(有×无) | ✓ | 对 16/16 |  |  |
| vmadotus(无×有) | ✓ | 对 16/16 |  |  |
| mma_slide(滑窗) | ✓ | 对 16/16 |  |  |
| tiled-matmul | ✓ | 对 |  |  |
| IME GEMM 整体性能 | — | — | **5.66×~12.9×**(注6) | 1.0×(注7) |

---

## 表 5:option-2「编译器自己选算法」(N3 旗舰,目标 q4_0)

| 阶段 | 板子 | 结果 |
|---|---|---|
| 编译器选 + 声明 + 产出 layout(A/B/C1/C1b) | host | 做完 |
| 自选路径 emit 出的 kernel 字节相同(M1) | rvv | 做完 |
| 自选 kernel 真跑 llama(M2) | rvv | **t1 7.0×**(注8) |
| 真 pipeline 自动化(C3) | — | **没做** |

---

# 解释(每条对应上表,白话,不说黑话)

**关于 Win-A 是不是「tune」**:Win-A 就是齿轮箱按板子的向量宽度,从 2-3 个合法的「向量寄存器宽度档位」(窄 mf2 / 中 m1 / 宽 m2)里**挑最快的一档**。所以它确实是「2-3 选 1」的小选择,不是花哨的大 sweep。N3 在 LMUL 这条上的「tune」本质就是这个 + 用实测挑赢家。我之前包装得太玄,纠正。

**表 1 各列什么意思**:
- 我们 emit 这些 block-dot 用的算法**和 ggml 一模一样**(逐 block 算点积),所以这里只有两件事可比:① 调旋钮(Win-A);② 我们生成的代码 vs ggml 官方代码谁快(vs-ggml)。
- **q4_K/q6_K/q3_K 的 Win-A**:把寄存器调宽(m1)比窄(mf2)快 1.15~1.30 倍——这是「同一个 kernel 调旋钮」的纯效果。
- **q4_K/q6_K/q3_K 的 vs-ggml 都 < 1(输)**:0.72/0.52/0.55——意思是即使调到最快的旋钮,我们还是比 ggml 的官方 K-quant kernel 慢(它用了更聪明的手写拆分)。q5_K 打平(1.00),q2_K 略赢(1.02)。
- **iq 系列 vs-ggml 都 < 1(还是输,但比以前好很多)**:这些是稀疏量化。**我们以前用「一个一个标量地查表」(慢),现在改成和 ggml 一样「一次性整块查表」**,所以从「慢 5~22 倍」收到「慢 2~8 倍」(iq2_xxs 最好,只慢 1.91 倍)。诚实讲:这是**追平 ggml 的成熟度提升,还没反超**。
- **iq4_nl / iq4_xs / mxfp4 > 1(真赢)**:这几个的查表表很小(≤16 项),我们用一种更省的查表方式,在 VLEN128 上反而比 ggml 快 21~32%。
- **q8_0/q4_1/q4_0 的 Win-A 是「N1 翻转」(注1)**:它们没有 LMUL 旋钮可调,但「在 rvv 上选 A 档、在 k1 上选 B 档,赢家跨板子换了」——这正是 N1(异构按能力选)。数字很小(1.01~1.11×),因为本来差别就小。
- **e2e 列**:除了 repack,block-dot 的 e2e 我**都没真测**,以前写成「推定 NULL」是不对的,正在用 wf25 实测 q4_K(decode + prefill 都测),用数据说话。

**表 2 repack**:
- **只有 q4_0 是真赢(2.60× decode / 5.68× prefill)**——因为它把权重在内存里重新排成「连续 16 块铺一排」,**改了内存搬运方式**,所以真传导到 llama。
- q8_0/q4_K/q5_0 的新算法 vs-ggml 都 < 1(0.65/0.55/0.77,输)——它们是「算得多」型的 quant,ggml 官方算法更强,我们的 repack 反而慢。**这正好告诉编译器:这些该选回 ggml 的算法**(option-2 干的就是这个)。
- **q4_0 在 k1 上 Win-B e2e = 0.74×(反而输)**:X60 的自动向量化比我们的 repack 路强,已如实记下。

**表 3 前向算子**:softmax/silu/rms_norm/rope/scale/quantize 这一系列**全能生成、结果正确(和 ggml 逐位一样)**,但**一个性能都没测、也没接进 llama**。它们是「覆盖广」,不是「优化证据」——因为它们和 ggml 同算法(没有不同算法可 PK),也没有旋钮可调。要把它们变成性能证据,得先给一个可调的轴,现在没有。

**表 4 IME**:
- 我们生成的是 IME 的**单条乘加指令**(vmadot 等 6 种),在 K1 硅片上跑出来**逐位正确(16/16)**。**我们没有生成完整的 IME GEMM kernel**——完整 GEMM 是 ggml-spacemit 写的。
- (注6)那个 **5.66~12.9× 是 ggml-spacemit 的 GEMM(它内部用了我们能生成的 vmadot 那种指令)vs ggml 普通 RVV**,不是「我们的 GEMM 赢」。
- (注7)IME 的 **e2e ≈ 1.0(没赢)**:开 IME 的 llama 整体快 1.65×,但我们做了对照——在「矩阵单元根本帮不上忙」的纯解码(M=1)上也快 1.47×,说明那 1.65× 是**换了一整套 kernel**带来的,不是矩阵单元本身。所以矩阵单元的 e2e 净贡献 ≈ 0。

**表 5 option-2**:(注8)t1 7.0× 是「编译器自己选出来的 kernel」真跑 llama 解码,精确复现了历史数字。诚实讲这是「字节级证明它选对了 kernel」,不是一个新性能数;**把这套自动接进真 pipeline(C3)还没做**。

**注解**:
- 注1:q8_0/q4_1/q4_0 这几格是「N1 跨板子选不同档」,不是 LMUL tune,数字小。
- 注2:ggml 在 riscv 上根本没写 nvfp4 的向量 kernel(只有标量),没有合法对手 = N/A。
- 注3:q4_1 repack 想跑 llama,需要 ggml 提供一个配套的激活打包器,ggml 没有 → 接不进去(不是我们没做)。
- 注4:**q4_1 repack 的 prefill(GEMM)旋钮 = 1.24×(WIDE/NARROW),byte-exact(2026-06-25,rvv VLEN128)**——WIDE(m1)比 NARROW(mf2)快,同一个 `repack_gemm_q4_1_q8_1` op、改 strip-width 旋钮(默认 mf2 / xtheadvector-stamp m1)、读同样的字节、WIDE↔NARROW norm 0.000e+00。LMUL 两档 source-grep 证明 disjoint(SG2044 objdump 解不了 RVV vtype)。报 min(min 是 best-of-reps 正确估计:噪声只会加时间),square 最低 1.24× / mlp 1.31×;更高的 1.5× 是 NARROW 在板子负载累积下 min 抬高的上界噪声,非更大真赢。**这是手动 stamp 的 m1 vs 默认 mf2 旋钮 ON/OFF,不是 RVV1.0 auto-tune 赢**。机理未隔离(GEMM 的 WIDE 反而 vle8 更多,与 GEVM 相反,所以不是 strip-count 减半——只报实测比值)。e2e 仍上游卡(无 q8_1x4 量化器)。GEMM 正确性 vs scalar oracle 早已验过(archive GAP 2,norm ≤7.6e-6)。详见 `06-25-backend-maturity-winA/artifacts/rvv-fill-q41-q10-FINDING.md`。
- 注5:q8_0 在 k1 上选宽档比窄档快 1.95×,但这个选择有没有传导到 llama e2e 还没测(最该补的)。
- 注6/7/8:见表 4、表 5 上文。
- 注9:**k1 配对实测(8 对,极稳)**——q4_K 的 m1 vs mf2 旋钮:prefill **1.258×**、decode **1.221×**(range 0.003)。仅在「关掉 q4_K repack、走 per-row vec_dot」regime 下成立;出厂默认会 repack 绕过这个 kernel → 对出厂 e2e 影响=0。**既没证实也没推翻内存墙**(关 repack 后 decode 本就算力受限);真·内存受限判官(8B@VLEN128 rvv)还没跑成(rvv 掉线)。详见 `qk-winA-e2e-FINDING.md` 的 FINAL 段。
- 注10:**IQ k1 micro(2026-06-25,12 格 vs-ggml·k1 全填)**——对手 = ggml 出厂在 k1 实际派发的 `_vl256` kernel(不是 rvv 列用的 `_vl128`;k1 `vlenb=32→VLEN256` 已 1 行二进制 probe)。**全 7 个 IQ 都 byte-exact(0.000e+00)、全 LOSS,比值 0.20–0.66**(ggml 1.5–5.1× 快)。注意:k1 比 rvv 列「好很多」(iq3_xxs 0.12→0.20)**是 emitter 从「标量查表」成熟到「真 vluxei16 整块查」**带来的,**不是 VLEN256 效应**——rvv 列那些数是旧 0-vluxei emit,两列不可直接比。残留 1.5–5× = gather LMUL/打包旋钮,非 primitive 缺失。详见 `k1-micro-fill-FINDING.md`。
- 注11:**FP4/ternary k1 micro(同上,vs `_vl256`)**——rvv 上的 FP4「赢」**不传导到 k1**:iq4_nl 1.32→0.84、mxfp4 1.21→0.80、iq4_xs 1.28→**1.04 TIE**(稳定 ×3,唯一非 LOSS,已按纪律复核);tq2_0 0.60→0.36、tq1_0 0.44→0.36。原因:VLEN256 上 ggml `_vl256` 用的正是我们在 VLEN128 上靠它赢的 mf2 split-16 gather 形状 + 2-blocks/iter 打包,而我们的 emit 仍是 VLEN128 形(`vsetvl_e8m1(16)`=VLEN256 下半个 m1 寄存器,sub-VLMAX)。rvv FP4 赢是 VLEN128 形状产物、非可移植优势。全部 byte-exact(iq4_xs/tq1_0 整数 bit-exact + 小 fp-reassoc)。命名 gearbox 目标 = VLEN-aware FP4/ternary lowering(VLEN256 出 mf2/e8m1-at-32 而非硬钉 vsetvl(16))。
- 注12:**q8_0 = Win-A 砖#1「能力驱动 lowering」成熟化(2026-06-25,commit a874a56b,两板封印)**——以前 q8_0 的档是 capability-blind cost model 写死选 m2。现在 **gearbox 按真 VLEN fact 选**:VLEN128→m2-elided / VLEN256→m1-elided,**256 阈值从 strip-VLMAX 算术涌现**(非硬编码 `if VLEN>=256`)。**两板 byte-exact 各 2400/2400**(k1 m1-elided 是全新 emit,load-bearing,无 fold-back 回归)。micro:k1 m1 比 m2 **+17.2%**、rvv m2 比 m1 +7.9%(对齐历史)。**关键:这不只是「挑快的」——在 VLEN128 强行 m1-elided 会 11/12 算错(半个 block 没加),所以这个选择是 gearbox `VLMAX≥blockLen` 剪枝守住的【正确性边界】**。这是"同一算子按能力 lower 成不同、且各自正确的码"的首个非-NULL 实证。详见 `q8-vlen-flip-BOARD-FINDING.md`。
- 注13:**q1_0 block-dot vs-ggml·rvv = 0.033×(LOSS,慢约 30×,byte-exact,2026-06-25)**——**不是 N/A**:q1_0 是真 ggml type(`GGML_TYPE_Q1_0`=41,二元 {−1,+1} 符号解码),ggml 在 riscv 上有真向量 kernel `ggml_vec_dot_q1_0_q8_0_vl128`(VLEN128 派发),所以有合法对手。我们 emit `q1_0_q8_0_block_dot` vs 它,8 seed × 6 size **全 byte-exact(max_rel 0.000e+00)**。LOSS 已 root-cause:**ggml `_vl128` 一次 vwredsum 整 32 lane(每 sub-block 1 次 reduce),我们 emit 按 8-lane 分组(每 super-block 32 次 vwredsum + 32 次 vmerge,8× 更多、更窄、全展开)**——和 IQ block-dot 同一个"窄 strip、没调 LMUL/shape"成熟度坑,q1_0 的 8-lane 组是最坏情形。**命名 emitter 成熟目标 = wide-LMUL block-as-32-lane emit**(不是正确性/算法选择问题)。byte-exact 旁注:ggml `_vl128` 在 i8 域取负(vneg(−128)=−128 溢出),我们先 widen 到 i16 再取负(全 int8 范围正确)——只在 q8 quant byte==−128 时分歧,而真 q8_0 量化域是 [−127,127](−128 永不出现),harness 按真域填 → byte-exact;−128 边界我们反而**更**正确(成熟旁注,非门失败)。详见 `06-25-backend-maturity-winA/artifacts/rvv-fill-q41-q10-FINDING.md`。

- 注14:**iq4_nl/mxfp4 = Win-A 砖#2「FP4 VLEN-aware 补宽」k1 板封印(2026-06-25,commit e65edf76)**——正是注11 那个成熟度坑的修复。以前默认 emit 是 VLEN128 形(VLEN256 只用半寄存器)→ k1 上 0.84/0.80 **LOSS**。现在 **gearbox 默认按真 VLEN fact 自动选**:VLEN256→**mf2 宽形**(满 VLMAX=16)、VLEN128→m1 不变。k1 板:objdump 确认 e8mf2、**byte-exact 0.000e+00**(mf2 gather 在 VLMAX=16 正确 index 全 16 codebook 项,注11 标的风险点已验)、**mf2 比旧 m1 快 ~1.27×**(Win-A·k1 列)、**vs-ggml 从 0.84/0.80 LOSS 拉回 1.08/1.01 parity**。诚实:天花板是 parity(采用 ggml 自己的 mf2 形状),非 beat-ggml;iq4_xs 因 fold-back live 留作独立一砖。详见 `fp4-mf2-k1-BOARD-FINDING.md`。

---

# 没做清单(自查漏洞,如实列)

1. **block-dot e2e:只测了 q4_K(注9,~1.22×传导)**,其余 block-dot 的 e2e 还没测。
2. ~~**IQ / FP4 / ternary 在 k1 上的 micro 全没测**(只在 rvv 测了)。~~ **DONE 2026-06-25**:12 格 vs-ggml·k1 全填(注10/注11,`k1-micro-fill-FINDING.md`)——全 byte-exact;0 win / 1 TIE(iq4_xs 1.04)/ 11 LOSS。对手是 ggml 出厂派发的 `_vl256`(不是 rvv 列的 `_vl128`)。rvv 的 FP4「赢」是 VLEN128 形状产物、不传导到 k1。
3. **q4_0 repack 在 k1、q5_0 repack 在 k1、q4_K repack 在 k1 都没做;q4_K repack k1 的正确性 oracle 还没跑**。
4. **6 个前向算子(softmax 等)0 性能测试**。
5. ~~q1_0 block-dot 没测~~ **DONE 2026-06-25**:q1_0 vs-ggml·rvv = **0.033×(byte-exact,LOSS ~30×,注13)**——非 N/A(ggml 有 `_vl128`);q5_0/q5_1 block-dot 的旋钮仍是 m1-only(没东西可调)。
6. **能补的真 gap**:q8_0 在 k1 的选择 e2e(头号)、~~q4_1 repack prefill 旋钮(最易)~~ **DONE 2026-06-25 = 1.24×(注4)**、option-2 的真 pipeline 自动化 C3(深活)。

---

# 成熟度定位(对标 Triton 后端,不是 TVM)

我们对标的是 **Triton 的「偏后端」部分**(拿 tile 级 IR,自动做 layout/向量化/指令选择/软件流水 → 硬件码),**不是 TVM**(自动搜索调度空间)。按这个尺子诚实定位:

| 维度 | Triton 后端会做 | 我们现在 | 判定 |
|---|---|---|---|
| **派发**(选哪个 family/target) | 按 target 选后端 | capability-fact 选 RVV/IME,**零 core 分支、非字符串匹配** | ✅ 成熟、且是 novelty(N1/N2) |
| **算法选择**(repack vs block-dot) | Triton 不做(用户写) | repack/option-2 = **前端贡献**(改算法/布局),有价值但**不是后端 novelty**;伸进 ggml 加载布局已越界,demote | ⬜ 前端栏,不计入后端成熟度 |
| **宽度/LMUL 选择** | 自动 layout assignment | Gearbox 按**真 VLEN fact** 选 LMUL——**砖#1 q8_0 已封印**:VLEN128→m2/VLEN256→m1,阈值从 VLMAX 算术涌现,两板 byte-exact,且是正确性边界(注12) | 🟢 砖#1 done;待推广更多 kernel + 真资源 fact 剪枝 |
| **向量化 + 指令选择** | 从 tile IR **自动**生成向量指令 | **手写**:emitter 里逐条敲 `__riscv_*`(~100 个),每个 quant 一个 emit 方法 | ❌ **手写,非自动**——「像玩具」的根源 |
| **软件流水 / ILP** | **自动**流水 pass | **手写**:emitter 里手摆 2-strip 交错;无任何自动流水 pass | ❌ 手写 |
| **通用 lowering** | 一个通用 pass 处理所有 kernel | **部分**:有通用 `emitWideningDotReduce`/`emitDequantize` 原语,但每个 quant 的解包是逐 kernel 手写(24 个) | 🟡 半通用 |

**一句话**:我们现在是「**capability 驱动的选择/派发层 + 手写 kernel 体**」。**选择/派发那层是真后端、是 novelty**(派发非字符串、还能选算法);但**代码生成那层(向量化/流水)是手写的,没有从 tile IR 自动 lower**——这就是为什么「去了字符串匹配,却还是手写 kernel 库」。

**通往成熟 Triton 后端的关键一步**(不是重写整个项目):把**一个** kernel 类(比如 dot-reduce)做成「从 tile/vector-dialect 级表示**自动**向量化 + 自动选 LMUL + 自动流水」的通用 lowering,证明后端能**自动生成**而不只是手写。这是把「玩具感」变「成熟 compiler」最直接、可论证的一步。**MLIR 的 vector/linalg dialect 我们现在一个都没用**(直接手写 EmitC)——这正是要补的入口。
