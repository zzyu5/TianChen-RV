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
| q8_0 | 1.08×(注12) | **1.17×(注12)** |  | **1.19×(注19)** |  |  | N/A |
| q4_1 | 1.03×(注1) | 1.11×(注1) |  | 0.26×(注19) |  |  | N/A |
| q4_0 | 1.01×(注1) |  |  | 0.21×(注19) |  |  | N/A |
| q5_0 | N/A | N/A |  | 0.26×(注19) |  |  | N/A |
| q5_1 | N/A | N/A |  | 0.29×(注19) |  |  | N/A |
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
| q1_0 | N/A | N/A |  | 1.38×(注13) | 0.98×TIE(注13) |  | N/A |

---

## 表 2:Repack 算子(我们自创了一个跟 ggml 不同的算法,所以有真正的 Win-B)

> Win-A = 调 strip/宽度旋钮;Win-B = 我们的新算法 vs ggml 官方算法。

| 算子 | Win-A·rvv | Win-A·k1 | Win-B·rvv | Win-B·k1 | Win-B·e2e·rvv | Win-B·e2e·k1 |
|---|---|---|---|---|---|---|
| q4_0 repack(decode) |  | 1.48× | 1.22× |  | **2.60×** | 0.74× |
| q4_0 repack(prefill) | 1.30× |  | 6.36× | 0.997×(注16) | **5.68×** |  |
| q4_1 repack(decode) |  |  | 正确性✓ |  | 上游卡(注3) | 上游卡 |
| q4_1 repack(prefill) | 1.24×(注4) |  | 正确性✓ |  | 上游卡 | 上游卡 |
| q8_0 repack(decode) |  | 1.95× | 0.65× |  | 推定平 | 平=vs-ggml-stock 1.00×(注5) |
| q4_K repack(decode) |  |  | 0.55× | 正确✓(注17) | 推定平 |  |
| q4_K repack(prefill) |  |  | 0.74× |  | 推定输 |  |
| q5_0 repack(decode) |  |  | 0.77× | 0.821×(注18) |  |  |

---

## 表 3:前向算子(softmax 这一系列)—— micro 已填(rvv VLEN128,注15)

| 算子 | 能否生成 | Win-A | Win-B | micro | e2e |
|---|---|---|---|---|---|
| softmax | ✓ | N/A | N/A | 0.854×(注15) | 空 |
| silu | ✓ | N/A | N/A | ≈0.84×(注15) | 空 |
| rms_norm | ✓ | N/A | N/A | 1.001×TIE(注15) | 空 |
| rope(rope_norm) | ✓ | N/A | N/A | 0.964×TIE(注15) | 空 |
| scale | ✓ | N/A | N/A | 1.006×TIE(注15) | 空 |
| quantize_row_q8_0 | ✓ | N/A | N/A | 0.986×TIE(注15) | 空 |

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

**表 3 前向算子**:softmax/silu/rms_norm/rope/scale/quantize 这一系列**全能生成、结果正确(和 ggml 逐位一样)**。**micro 已测(rvv VLEN128,注15)**:4 个 TIE(scale 1.006×、rms_norm 1.001×、rope 0.964×、quantize 0.986×)+ 2 个 LOSS(silu 0.840×、softmax 0.854×),**全 byte-exact、0 win**。这正是预期——它们和 ggml **同算法**(scale/quantize 还是同 m8 路),所以本就该打平;**2 个 LOSS 已 root-cause**(silu/softmax 走的共享 exp 多项式,ggml 有 `vcpop` 短路跳过慢路径 fixup、我们 emit 无条件走慢路径 → 慢 15-16%,byte-exact,是缺数据相关分支消除的 emitter 成熟度坑,不是算法/旋钮差异)。rms_norm 是诚实意外:ggml 没 RVV 路(标量源)、本以为向量化会赢,实测 TIE——因为它**归约受限**,主导成本是串行 f64 `sum+=x²` 折叠(两边都标量、byte-exact 要求),向量化归一化只是小尾巴。它们是「覆盖广 + 同算法 parity」,不是「优化证据」;也没有旋钮可调(Win-A N/A)、没接进 llama(e2e 仍空)。

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
- 注5:**q8_0 repack e2e 已实测(2026-06-26,k1 VLEN256,taskset 0-3,tinyllama-1.1b Q8_0;用户授权重建 ggml-cpu + 跑 llama-bench/cli;不改我们 lib/、用现有 tcrv-opt)**——routing **有利**:q8_0 decode 在 k1 走我们的 repack GEVM(`ggml_gemv_q8_0_16x1_q8_0`,**不像 q4_K 被 GEMM 绕过**),是出厂 decode 路径。3 种 .so prebuild + md5-assert 热插不重建、tight-interleave、engagement print 每档确认、coherence("Paris")PASS(ABI 7→5 arg shim 验过)。**两个结果**:① **WIDE/NARROW 旋钮传导**:decode tg **1.42×**(t4 配对 3 轮:WIDE 4.453 / NARROW 3.135,sd≤0.018;t1 配对 1.47×→t4 1.42×=多线程带宽只轻微吃掉旋钮),prefill pp **1.00×**(WIDE 10.70/NARROW 10.70,flat-by-construction:prefill 走 ggml 自己的 GEMM、我们 decode-only GEVM 旋钮碰不到,是证伪对照而非失败传导)。**1.42× ≪ 1.95× micro**——内存受限+Amdahl(decode 含 attention/norm/kv/sampling)重度衰减,**1.95× 是内部旋钮 micro、不是 e2e;1.42× 才是**。② **vs ggml-stock = PARITY**:从 pristine 重建 ggml 出厂 kernel 做基线,our-WIDE **== ggml-stock**(t4 配对 round-1 4.4199 vs 4.4203=**0.99990×**;t4/t1 综合 0.99–1.01×)——我们 gearbox 的 WIDE 选择**性能等同 ggml 专家手写的 `q8_0_16x1`**,同 q4_0 0.997× TIE,**收敛到专家选择、不是赢 ggml**(与 micro 表里 q8_0 repack Win-B 0.65× LOSS 正交:那是 vs block-dot;这里 GEVM 就是出厂路径、等于 ggml 自己的 GEVM)。**诚实定位:1.42× 证明"选对 lane-width 的能力驱动 lowering 买到 42% decode",战略上是 parity 非 beat-ggml。**板子跑后已 forced-rebuild 复原 pristine(注意:第一次 restore 是 stale-object no-op 出 WIDE 指纹,touch+强制重建后 live .so=stock 14b6add6、0 个 TCRV 串=真干净——印证"增量构建不可靠"memory)。详见 `q8-repack-e2e-FINDING.md` 的 ## FINAL e2e 数。
- 注6/7/8:见表 4、表 5 上文。
- 注9:**k1 配对实测(8 对,极稳)**——q4_K 的 m1 vs mf2 旋钮:prefill **1.258×**、decode **1.221×**(range 0.003)。仅在「关掉 q4_K repack、走 per-row vec_dot」regime 下成立;出厂默认会 repack 绕过这个 kernel → 对出厂 e2e 影响=0。**既没证实也没推翻内存墙**(关 repack 后 decode 本就算力受限);真·内存受限判官(8B@VLEN128 rvv)还没跑成(rvv 掉线)。详见 `qk-winA-e2e-FINDING.md` 的 FINAL 段。
- 注10:**IQ k1 micro(2026-06-25,12 格 vs-ggml·k1 全填)**——对手 = ggml 出厂在 k1 实际派发的 `_vl256` kernel(不是 rvv 列用的 `_vl128`;k1 `vlenb=32→VLEN256` 已 1 行二进制 probe)。**全 7 个 IQ 都 byte-exact(0.000e+00)、全 LOSS,比值 0.20–0.66**(ggml 1.5–5.1× 快)。注意:k1 比 rvv 列「好很多」(iq3_xxs 0.12→0.20)**是 emitter 从「标量查表」成熟到「真 vluxei16 整块查」**带来的,**不是 VLEN256 效应**——rvv 列那些数是旧 0-vluxei emit,两列不可直接比。残留 1.5–5× = gather LMUL/打包旋钮,非 primitive 缺失。详见 `k1-micro-fill-FINDING.md`。
- 注11:**FP4/ternary k1 micro(同上,vs `_vl256`)**——rvv 上的 FP4「赢」**不传导到 k1**:iq4_nl 1.32→0.84、mxfp4 1.21→0.80、iq4_xs 1.28→**1.04 TIE**(稳定 ×3,唯一非 LOSS,已按纪律复核);tq2_0 0.60→0.36、tq1_0 0.44→0.36。原因:VLEN256 上 ggml `_vl256` 用的正是我们在 VLEN128 上靠它赢的 mf2 split-16 gather 形状 + 2-blocks/iter 打包,而我们的 emit 仍是 VLEN128 形(`vsetvl_e8m1(16)`=VLEN256 下半个 m1 寄存器,sub-VLMAX)。rvv FP4 赢是 VLEN128 形状产物、非可移植优势。全部 byte-exact(iq4_xs/tq1_0 整数 bit-exact + 小 fp-reassoc)。命名 gearbox 目标 = VLEN-aware FP4/ternary lowering(VLEN256 出 mf2/e8m1-at-32 而非硬钉 vsetvl(16))。
- 注12:**q8_0 = Win-A 砖#1「能力驱动 lowering」成熟化(2026-06-25,commit a874a56b,两板封印)**——以前 q8_0 的档是 capability-blind cost model 写死选 m2。现在 **gearbox 按真 VLEN fact 选**:VLEN128→m2-elided / VLEN256→m1-elided,**256 阈值从 strip-VLMAX 算术涌现**(非硬编码 `if VLEN>=256`)。**两板 byte-exact 各 2400/2400**(k1 m1-elided 是全新 emit,load-bearing,无 fold-back 回归)。micro:k1 m1 比 m2 **+17.2%**、rvv m2 比 m1 +7.9%(对齐历史)。**关键:这不只是「挑快的」——在 VLEN128 强行 m1-elided 会 11/12 算错(半个 block 没加),所以这个选择是 gearbox `VLMAX≥blockLen` 剪枝守住的【正确性边界】**。这是"同一算子按能力 lower 成不同、且各自正确的码"的首个非-NULL 实证。详见 `q8-vlen-flip-BOARD-FINDING.md`。
- 注13:**q1_0 block-dot vs-ggml·rvv = 1.38× WIN(byte-exact,rvv VLEN128,2026-06-26)——从 0.033× LOSS(慢约 30×)修成 WIN**,wide-LMUL emit 成熟度落地的标杆。q1_0 是真 ggml type(`GGML_TYPE_Q1_0`=41,二元 {−1,+1} 符号解码),ggml 在 riscv 上有真向量 kernel `ggml_vec_dot_q1_0_q8_0_vl128`(VLEN128 派发),合法对手。**修法 = 把 emit 改成 ggml 出厂 `_vl128` 的 lane 结构**:旧 emit 按 8-lane 分组(每 sub-block 4 次窄 vwredsum + kmask/vmv_v_x/vand/vmsne/vwcvt/i16-merge 一大串),新 emit 一个 **32-lane sub-block body**——`vlm_v_b{ratio}` 把 4 个打包 bit-byte **直接当 i8 符号 mask 读**(packed bit 即 mask,省掉整个 kmask 解码)、`vle8` 取 32 q8、**i8 域** vneg/vmerge、**每 sub-block 1 次** `vwredsum_vs_i8{anchor}_i16m1`。anchor 随 VLEN 移动(32-lane sub-block 跨 m1 在 128/256 的 VLMAX 边界,同 q8_0):**VLEN128→m2(vlm_v_b4)/ VLEN256→m1(vlm_v_b8)**,由 gearbox 自动 stamp(`getRVVStripVLMAXElements` 单一真值源,非硬编码)。8 seed × 6 size **byte-exact max_rel 0.000e+00**。byte-exact 旁注(本次有意收敛):新 emit 跟 ggml 一样在 **i8 域** vneg,所以 −128 边界现在跟 ggml **完全一致**(旧 emit widen-to-i16 在 −128 时"更正确";真 q8_0 量化域 [−127,127],−128 永不出现,门不受影响)——这是为 byte-exact-by-construction 故意放弃的 superset robustness,记录在案。**这是 within-block lowering(纯丰富 Win-A 后端旋钮),非跨块拼/repack/选算法。** **VLEN256 m1 形 k1 板验 DONE(2026-06-26)**:k1(SpacemiT X60,VLEN256,vlenb=32→e8m1 VLMAX=32 板上探针确认)上 **byte-exact 0.000e+00 vs 独立 scalar oracle**(结构无关、不与 ggml 向量参考共享 bit→lane 解码,真守 silent-wrong)**且 0.000e+00 vs ggml `_vl256`**(8 seed × 6 size,强制 ±127 边界 lane)。`llvm-objdump-20` 板上二进制确认 **每 sub-block `vsetvli e8,m1`(AVL=32→vl=32 满 32-lane)× 4,vlm.v/vle8.v/vrsub.vi/vmerge.vvm/vwredsum.vs 各 4 次,0× `e8,m2`**(`e16,m2` 仅是 i16 reduce-seed `vmv.s.x` + 4 次标量 `vmv.x.s` 提取,LMUL don't-care)=gearbox m2→m1 翻转**硬件证实**。**micro k1 = 0.98× TIE**(稳定 ×3):VLEN256 上 ggml `_vl256`(quants.c:484)**就是**我们 emit 的 m1 形(指令逐条相同),所以 parity 是天花板、不是 beat——rvv 的 1.38× WIN 是 vs ggml `_vl128`(m2 劣形)的 VLEN128 产物,不可移植。详见 `06-25-backend-maturity-winA/artifacts/q10-wide-lmul-BOARD-FINDING.md`。

- 注14:**iq4_nl/mxfp4 = Win-A 砖#2「FP4 VLEN-aware 补宽」k1 板封印(2026-06-25,commit e65edf76)**——正是注11 那个成熟度坑的修复。以前默认 emit 是 VLEN128 形(VLEN256 只用半寄存器)→ k1 上 0.84/0.80 **LOSS**。现在 **gearbox 默认按真 VLEN fact 自动选**:VLEN256→**mf2 宽形**(满 VLMAX=16)、VLEN128→m1 不变。k1 板:objdump 确认 e8mf2、**byte-exact 0.000e+00**(mf2 gather 在 VLMAX=16 正确 index 全 16 codebook 项,注11 标的风险点已验)、**mf2 比旧 m1 快 ~1.27×**(Win-A·k1 列)、**vs-ggml 从 0.84/0.80 LOSS 拉回 1.08/1.01 parity**。诚实:天花板是 parity(采用 ggml 自己的 mf2 形状),非 beat-ggml;iq4_xs 因 fold-back live 留作独立一砖。详见 `fp4-mf2-k1-BOARD-FINDING.md`。

- 注15:**表3 前向算子 micro(2026-06-25,rvv VLEN128,现有 tcrv-opt 无 rebuild、不改 lib/)**——6 个 f32 前向算子(同算法重写 ggml)的 OUR-emit vs ggml 官方同算子,**全 6 个 byte-exact gate PASS 后才报 perf**,min-of-reps。fresh-emit 与归档 inc15–inc20(已验正确)**逐字节相同**(grep-diff),故旧正确性证据适用。结果:**scale 1.006× / rms_norm 1.001× / rope 0.964× / quantize_q8_0 0.986× = 4 TIE**;**silu 0.840× / softmax 0.854× = 2 LOSS**;**0 win**。① 4 个 TIE 是预期的同算法 parity(scale/quantize 同 m8 路 → 必平;rope 标量 libm 主导 → 平)。② **2 个 LOSS 已 root-cause 且非噪声/非算法差异**:silu+softmax 共用向量 exp 多项式 `ggml_v_expf_m2`,ggml 有 `if(!vcpop_m(c)) return 快路径` 的**数据相关短路**(良性输入无 lane 溢出时跳过昂贵的慢路径 overflow-fixup `vmerge` 链),而**我们 emit 的 kernel 0 个 `vcpop`**(grep 确认)、**无条件**走慢路径 → 慢 15-16%,byte-exact,是**缺数据相关分支消除的 emitter 成熟度坑**(命名目标 = 给共享 `ggml_v_expf_m2` emit `vcpop`-gated 快路径短路)。③ **rms_norm 是诚实意外**:ggml 无 RVV 路(标量源)、本以为向量化(m8)会赢,实测 1.001× TIE——已验 clang `-O3 -march=rv64gcv` **未** auto-vec ggml 标量参考(asm grep:ggml_ref 区 0 向量指令),但我们仍不快,因为 rms_norm **归约受限**:主导成本是串行 f64 `sum+=x²` 折叠(两边都标量、byte-exact 要求),向量化只在小尾巴上 → 向量化 rms_norm **不是 win**(纠"我们向量化了 ggml 留标量的"诱惑)。Win-A/Win-B 全 N/A(无 LMUL 旋钮、同算法);e2e 仍空(本任务只测 micro)。详见 `forward-pass-micro-FINDING.md`。

- 注16:**q4_0 repack prefill GEMM 在 k1 = 0.997× TIE(parity,2026-06-25,k1 VLEN256,现有 tcrv-opt 无 rebuild、不改 lib/)**——对手 = ggml 出厂自己的真向量 `ggml_gemm_q4_0_16x1_q8_0`(**同一 x16 repack 布局**,ggml 自己的 VLEN-向量化 GEMM,从 k1 已 build 的 `libggml-cpu.so` 链接,**非标量**)。我们的 h16 emit(VLEN256-native 单 mf2 strip,16 lane)**byte-exact gate 先过**:① vs ggml `_generic` 正确性 norm **9.06e-6** PASS;② perf micro SANITY ours==ggml-real **max_rel 0.000e+00**。3 次稳定 **0.996–0.997×(TIE)**。诚实:**我们 emit 收敛到 ggml 自己手写的 16x1 GEMM**(输出逐字节同、延迟约等)——这是"parity = 采用 ggml 自己同布局的指令"的成熟 emit,**非 perf win**。与历史 rvv q4_0 GEVM repack e2e 2.6× 是不同轴(那是 decode 改内存搬运;这是 compute-bound prefill GEMM vs ggml 匹配的 kernel)。详见 `k1-repack-fill-FINDING.md`。

- 注17:**q4_K repack GEVM 在 k1 正确性 oracle = PASS,WORST_NORM 7.07e-7(2026-06-25,k1 VLEN256,现有 tcrv-opt 无 rebuild)**——填没做 #3 的「q4_K repack k1 正确性 oracle」。emit 在 **VLEN256-native `half_lanes=16` 单 mf2 strip**(verifier 接受 h∈{8,16};h16 emit 是 rvv h8 两 strip 路的恰好一半 vle8/vwmacc = 一个 16-lane strip)。**注:h16 是手动 stamp 输入 op 的 `half_lanes` 属性(默认 emit 是 h8,无 VLEN→half_lanes auto-select)——VLEN-aware 自动选 half_lanes 是另一个 Win-A gap、本处未做**,只是手 emit native strip 并测正确性。INDEPENDENT 标量 q4_K dequant-matmul oracle(从 ORIGINAL pre-repack `block_q4_K` 算,喂 kernel 同样 blocks 经 `make_block_q4_Kx16` 重排——不同布局同值,repack-inversion bug 不能两边一致)。8 shape 全 PASS(bar 1e-4),两 negative control 硬触发:NOMIN 396,124× margin、PERM 2,080,930× margin。norm ≡ rvv 列(7.07e-7)因整数 dot 精确、仅 fp16 scale-fold 舍入 = VLEN-不变。**注:k1 q4_K MICRO(vs ggml)未做**——任务只要 oracle;且 VLEN256 上 ggml baseline 翻成它自己的 `ggml_gemv_q4_K_16x1_q8_K` repack(VLEN≥256 才能正确跑;VLEN128 上返回 nullptr 退回 block-dot),是比 rvv q4_K winB(0.55×/0.74× vs block-dot)更强、不同的对手,留作 follow-up,不静默并入。详见 `k1-repack-fill-FINDING.md`。

- 注18:**q5_0 repack GEVM 在 k1 micro = 0.821× LOSS(best anchor,2026-06-25,k1 VLEN256,现有 tcrv-opt 无 rebuild)**——对手 = ggml 真向量 q5_0 **block-dot**(`ggml_vec_dot_q5_0_q8_0` 的 unified `__riscv_v` body,`vlenb==32` 走 VLEN256 `vslideup`+`vlmul_ext` 分支;ggml 不出 q5_0 repack → per-column block-dot 是方法论正确 baseline)逐列跑 16 次。**byte-exact gate 先过**:两 anchor(mf2/m1)norm **0.000e+00** @ NC∈{16,32,64,336},SANITY ours==ggml-per-col max_rel 0。比值:**mf2(h16 native 16-lane 单 strip)0.821×**、**m1(h16 over-provision 32-lane)0.562×**——best anchor mf2 **0.821× LOSS**(ggml ~1.22× 快)。setup **handicap ggml / 利我们**(ggml 每组重读激活 16×、我们读 1×)、对比对**我们慷慨**,仍输 → LOSS 反而**更稳健**(去掉 ggml 的 handicap → ggml 更快 → 我们输更多),钉在 COMPUTE 轴非内存假象。**h16 同样是手动 stamp(默认 h8,无 auto-select)**。原因 = q5_0 compute-bound:ggml 一个宽 masked `vsub` 折 qh,我们须逐 nibble lane-wise 展开 transposed qh mask(vid/vsrl/vand/vsll/vncvt)= 记录的 N3 模式「gather-heavy 赢、compute-bound 输 ggml wide LMUL」。**k1 0.821× 比 rvv 0.769× 略好**——native 16-lane strip 恰配 VLEN256,rvv VLEN128 拆两 8-lane strip。详见 `k1-repack-fill-FINDING.md`。

- 注19:**标准 quant block-dot vs-ggml·rvv micro 全填(2026-06-25,rvv VLEN128,现有 tcrv-opt 无 rebuild、不改 lib/)**——填 q8_0/q4_1/q4_0/q5_0/q5_1 这 5 格(原全空)。对手 = ggml 出厂真向量 `ggml_vec_dot_*_q8_*`(verbatim 自 quants.c,q5_0/q5_1 走 `vlenb==16` VLEN128 分支),`taskset -c 0` min-of-reps,**5/5 byte-exact gate 先过(max_rel 0.000e+00,8 seed × 6 size)再报 perf**。结果:**q8_0 = 1.19× WIN**(唯一赢)、**q4_0 0.21× / q4_1 0.26× / q5_0 0.26× / q5_1 0.29× = 4 LOSS**(慢 3.4–4.8×)。**单一机理(grep .cpp 确认):是我们 emitter 选的 LMUL/strip 形状,非算法**——q8_0 block-dot **无 nibble 解包**,我们 emit 取 **m2 满 32-lane 单 strip**(`vle8m2`/`vwmul_i16m4`/一次 vwredsum),VLEN128 下 e8m2 VLMAX=32 → 内 strip 循环**恰跑一次**,和 ggml **同形**却仍稳定更快 **~1.17–1.20× WIN**——**两个判别器排除了廉价解释**:① 把 ggml 的 `vmv_v_x_i32m1(0)` 提出循环(loop-invariant)→ 仍 1.167–1.192×(**不是 zero-mv 提升**);② 反转计时顺序(先测 ours)→ 仍 1.200–1.201×(**不是 ordering/cache 假象**)。所以是真·同形 edge,但**机理未隔离**(大概率是我们 emit 的 load 调度/显式临时变量 vs ggml 嵌套表达式),**报 WIN 但不声明机理**;而 4 个 nibble-decode quant 我们都 emit 成 **mf4(¼-LMUL,8-lane)sub-VLMAX strip + 每 strip vsetvl 开销**,ggml 用 m1(16-lane)半块 → 我们 ~2× 更多更窄 strip → LOSS。这是**记录在案的 N3「窄 strip、没调 LMUL/shape」emitter 成熟度坑**(同 q1_0 0.033× / IQ block-dot 一家),**非正确性/选算法问题**;命名 emitter 目标 = **nibble-decode quant 的 wide-LMUL(mf4→m1)block-dot emit**。诚实:这 4 个标准 quant 的算法家在 **repack**(q4_0 repack 是真 2.6× e2e WIN,表2),block-dot 是构件,此处 LOSS 不与 repack WIN 矛盾(不同轴);q8_0 block-dot 本身就是 lean kernel(无 nibble)所以是真宽-strip parity+edge。详见 `06-25-backend-maturity-winA/artifacts/rvv-fill-standard-quant-FINDING.md`。
- 注20:**Track B = 成熟类-Triton 后端核心,第一块两板封印(2026-06-26,commit 85c7f1ab/0ed17da4)**——编译器从一个 **generic `vector.multi_reduction`**(int8→int32 dot-reduce,K=32)**自动构造** RVV body(新前端 `RVVReductionSourceFrontDoor`,LMUL 从 gearbox 能力 fact `deriveMinimumVLEN`+schedule authority 推、**非硬开关**),**不动手写 emitter**。能力翻转:VLEN128→e8m2/vwmul_i16m4、VLEN256→e8m1/vwmul_i16m2。**两板 byte-exact 封印**(rvv 16/16+8/8、k1 16/16+8/8 vs scalar oracle,全 i8 范围 ±127/−128 + 累加循环精确)。trellis-check PASS(verifier 放宽 fail-closed、收紧了 1 处 dead-permissive SEW16、2 砖无回归)。**这是"后端按能力自动生成 lowering(非 per-kernel 手写)"的首个非-NULL+硅片证明**;诚实:窄 novelty=能力 fact 驱动选择融进 vector→tcrv→EmitC(vector→RVV 上游已有,我们不声明"造了 Triton 后端"),只 1 个 bounded contraction,推广全 quant zoo 是后续。一个 tooling gap:production export 路对新 m2 body 默认走 block-quant mf4、需后续接。详见 `trackB-auto-lowering-DESIGN.md` + `trackB-board-FINDING.md`。

---

# 没做清单(自查漏洞,如实列)

1. **block-dot e2e:只测了 q4_K(注9,~1.22×传导)**,其余 block-dot 的 e2e 还没测。
2. ~~**IQ / FP4 / ternary 在 k1 上的 micro 全没测**(只在 rvv 测了)。~~ **DONE 2026-06-25**:12 格 vs-ggml·k1 全填(注10/注11,`k1-micro-fill-FINDING.md`)——全 byte-exact;0 win / 1 TIE(iq4_xs 1.04)/ 11 LOSS。对手是 ggml 出厂派发的 `_vl256`(不是 rvv 列的 `_vl128`)。rvv 的 FP4「赢」是 VLEN128 形状产物、不传导到 k1。
3. ~~**q4_0 repack 在 k1、q5_0 repack 在 k1、q4_K repack 在 k1 都没做;q4_K repack k1 的正确性 oracle 还没跑**。~~ **DONE 2026-06-25(k1 VLEN256,现有 tcrv-opt 无 rebuild、不改 lib/,`k1-repack-fill-FINDING.md`)**:① q4_K repack k1 正确性 oracle = **PASS,WORST_NORM 7.07e-7**(VLEN256-native h16 单 strip,2 negative control 硬触发,注17);② q5_0 repack k1 micro = **0.821× LOSS**(best anchor mf2;m1 0.562×;两 anchor byte-exact norm 0,注18);③ q4_0 repack prefill GEMM k1 = **正确✓(norm 9.06e-6)+ 0.997× TIE** vs ggml 真 16x1 GEMM(注16)。**残留**:k1 q4_K MICRO(vs ggml 自己 VLEN256 repack,更强对手)+ q5_0/q4_0 e2e 未做(只测 micro/oracle 轴)。
4. ~~**6 个前向算子(softmax 等)0 性能测试**。~~ **DONE 2026-06-25 micro(rvv VLEN128,注15)**:6/6 byte-exact;4 TIE(scale/rms_norm/rope/quantize)+ 2 LOSS(silu 0.840×/softmax 0.854×,root-cause = 缺 ggml exp 多项式的 `vcpop` 快路径短路)。e2e 仍空(本任务只测 micro)。
5. ~~q1_0 block-dot 没测~~ **DONE 2026-06-25** → **wide-LMUL emit 修成 WIN 2026-06-26**:q1_0 vs-ggml·rvv 从 **0.033× LOSS 修到 1.38× WIN(byte-exact,注13)**——把 emit 换成 ggml `_vl128` 的 32-lane lane 结构(`vlm_v_b{ratio}` 直读符号 mask + i8 域 vneg/vmerge + 每 sub-block 1 次 vwredsum),接 gearbox 自动选 anchor(VLEN128→m2/VLEN256→m1)。这是"窄 strip、没调 LMUL/shape"成熟度坑的**第一个落地修复 + 自动 stamp**。q5_0/q5_1 block-dot 的旋钮仍是 m1-only(没东西可调)。
5b. ~~**标准 quant(q8_0/q4_1/q4_0/q5_0/q5_1)的 block-dot vs-ggml·rvv 全空**。~~ **DONE 2026-06-25(rvv VLEN128,注19)**:5/5 byte-exact;**q8_0 1.19× WIN**(满 m2 32-lane 单 strip = ggml 同形)+ q4_0 0.21×/q4_1 0.26×/q5_0 0.26×/q5_1 0.29× **4 LOSS**(mf4 ¼-LMUL 窄 strip vs ggml m1,命名 emitter 目标 = nibble-decode quant 的 wide-LMUL block-dot emit)。这些是构件 micro,nibble quant 的算法家在 repack(表2)。
6. **能补的真 gap**:~~q8_0 在 k1 的选择 e2e(头号)~~ **DONE 2026-06-26 = decode 1.42×传导 / pp 1.00× flat / vs-ggml-stock parity 0.99–1.01×(注5)**、~~q4_1 repack prefill 旋钮(最易)~~ **DONE 2026-06-25 = 1.24×(注4)**、option-2 的真 pipeline 自动化 C3(深活)。

---

# 成熟度定位(对标 Triton 后端,不是 TVM)

我们对标的是 **Triton 的「偏后端」部分**(拿 tile 级 IR,自动做 layout/向量化/指令选择/软件流水 → 硬件码),**不是 TVM**(自动搜索调度空间)。按这个尺子诚实定位:

| 维度 | Triton 后端会做 | 我们现在 | 判定 |
|---|---|---|---|
| **派发**(选哪个 family/target) | 按 target 选后端 | capability-fact 选 RVV/IME,**零 core 分支、非字符串匹配** | ✅ 成熟、且是 novelty(N1/N2) |
| **算法选择**(repack vs block-dot) | Triton 不做(用户写) | repack/option-2 = **前端贡献**(改算法/布局),有价值但**不是后端 novelty**;伸进 ggml 加载布局已越界,demote | ⬜ 前端栏,不计入后端成熟度 |
| **宽度/LMUL 选择** | 自动 layout assignment | Gearbox 按**真 VLEN fact** 选 LMUL,阈值从 VLMAX 算术涌现、两板 byte-exact。**砖#1 q8_0**(VLEN128→m2/VLEN256→m1,正确性边界,注12)+ **砖#2 FP4 codebook**(VLEN256 自动选 mf2 宽形,把 0.80 LOSS 拉回 parity,注14)**已封印**;gearbox 已泛化到 q8_0/q4_0/q5_0/codebook 等 | 🟢 2 砖 done,真后端在做;待推广更多 kernel + cost model 真资源 fact 剪枝(现仍 1-bit) |
| **从高层 op 自动生成 body** | 从 tile/vector IR **自动**构造,非手写 | **Track B 已开第一个口**:generic `vector.multi_reduction` 经新前端 `RVVReductionSourceFrontDoor` **自动构造** dot-reduce body(LMUL 从 gearbox 能力 fact 推),VLEN128→e8m2 / VLEN256→e8m1,**两板 byte-exact 封印**(注20)。是第一个"自动生成非手写"的块 | 🟢 第一块 board-sealed(K=32 int8 dot-reduce);其余 quant 仍手写 emitter |
| **指令级 emit(打印机)** | — | RVVToEmitC 是 dumb 1:1 op→intrinsic 打印机(智能在 op 结构里,不是 gap);~100 个 `__riscv_*` | ✅ 不是瓶颈(被 Track B 复用不动) |
| **软件流水 / ILP** | **自动**流水 pass | 现"流水"=emitter/body 构造里摆 N 条 strip 链(Track B 可泛化);无独立 scf 流水 pass | 🟡 body-构造级,非 scf transform |
| **通用 lowering** | 一个通用 pass 处理所有 kernel | 通用原语 `emitWideningDotReduce`/`emitDequantize` + **Track B 的 generic-op 前端**(先 dot-reduce);各 quant 解包仍逐 kernel | 🟡 半通用,Track B 在扩 |

**一句话(2026-06-25 更新)**:**后端的"能力驱动 lowering 选择"这一层,从"capability-blind、写死一档"升级成了"按真 VLEN fact 自动选、两板 byte-exact、阈值从 VLMAX 算术涌现"——2 块砖封印(q8_0 正确性边界 + FP4 codebook LOSS→parity),且 gearbox 泛化到多个 kernel。这是真后端在做能力驱动 lowering、是 novelty,不再是"只会选不会 lower"。** 但要诚实分清还没做的:**指令级代码生成(向量化/软件流水)仍是手写的**(emitter 逐条敲 `__riscv_*`,无自动向量化/流水 pass),所以现在是「**能力驱动的 lowering-选择 + 手写指令体**」。前端那栏(repack/算法选择)价值另算、不混进后端 novelty。

**还差什么(诚实,2026-06-26 更新)**:(1) cost model 的 scalar 公式仍 capability-blind(VLEN 已进枚举/合法性、砖#1/2 用的就是;但 ELEN/vreg/mask/tail 等真资源 fact 仍没进——Track A 调查发现这条目前不翻转任何选择=近 NULL,只剩 RVV0.7 防御性 gap);(2) **"从高层 op 自动生成 body"这条——原来说"一个都没做",现在 Track B 已board-sealed 第一个块**(generic `vector.multi_reduction`→能力驱动 RVV body,两板 byte-exact),证明这条路通;但只 1 个 bounded contraction(K=32 int8 dot-reduce),**推广到带 dequant/codebook 的全 quant zoo 是后续大工程**。**进展定性:从"能力驱动选择=真后端(2 砖)"再进一步到"能力驱动自动生成 body=真后端(Track B 第一块封印)"——"玩具感"正在被一块块换成可两板封印的成熟度证据。**
