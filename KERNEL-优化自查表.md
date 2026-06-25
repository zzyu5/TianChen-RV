# TianChen-RV 自查表 v4(2026-06-25)

> 表格只放数字,没做就空着。所有解释在最后单独成段。

## 图例(看表前先读这个)

- **数字 = 我们的 kernel 跟 ggml 官方 kernel 的速度比**:`>1` 我们更快(赢),`<1` 我们更慢(输)。例:`1.32×` = 我们快 32%;`0.58×` = 我们只有 ggml 的 0.58 倍速(慢)。
- **Win-A 列的数字 = 旋钮开 vs 关**(同一个 kernel,把向量寄存器从窄调到宽,快多少)。例:`1.30×` = 调宽后快 30%。
- **空格 = 这个板子/这一项没做**(如实留空,不是不适用)。
- **N/A = 本质不适用**(这个 kernel 没有这个轴可做)。
- **Win-A·e2e 的数字 = 旋钮开 vs 关传导到 llama 整体多少**(不是 vs ggml)。
- 板子:**rvv** = SG2044 VLEN128;**k1** = X60 VLEN256;**ime** = X60 的矩阵单元。

> **★内存墙纠正(2026-06-25 实测)**:我之前断言「算力侧优化在 decode 被内存墙washes成 1.0」是**错的**。wf25 在 k1 实测 q4_K 的 Win-A(m1 vs mf2,纯算力旋钮):**decode 也传导了 ~1.22×、prefill ~1.26×**(初步,配对版确认中)。说明 **decode 没把内存带宽占满,算力变快真的让解码变快**——和你「改 kernel 变快就真变快」的记忆一致。所以下表 Win-A·e2e 不再写「推定 NULL」;没测的就空着。

---

## 表 1:Block-dot 算子(我们 emit 的算法和 ggml 一样,所以只有「调旋钮(Win-A)」和「跟 ggml 官方 PK」)

| 算子 | Win-A·rvv | Win-A·k1 | Win-A·e2e | vs-ggml·rvv | vs-ggml·k1 | vs-ggml·e2e | ime |
|---|---|---|---|---|---|---|---|
| q4_K | 1.26× | 1.30× | **1.22~1.26×实测**(注9) |  | 0.72× |  | N/A |
| q6_K |  | 1.15× |  |  | 0.52× |  | N/A |
| q3_K |  | 1.15× |  |  | 0.55× |  | N/A |
| q5_K |  | 1.30× |  |  | 1.00× |  | N/A |
| q2_K | N/A | N/A |  |  | 1.02× |  | N/A |
| q8_0 | 1.07×(注1) | 1.07×(注1) |  |  |  |  | N/A |
| q4_1 | 1.03×(注1) | 1.11×(注1) |  |  |  |  | N/A |
| q4_0 | 1.01×(注1) |  |  |  |  |  | N/A |
| q5_0 | N/A | N/A |  |  |  |  | N/A |
| q5_1 | N/A | N/A |  |  |  |  | N/A |
| iq1_s | N/A | N/A |  | 0.43× |  |  | N/A |
| iq1_m | N/A | N/A |  | 0.17× |  |  | N/A |
| iq3_xxs | N/A | N/A |  | 0.12× |  |  | N/A |
| iq3_s | N/A | N/A |  | 0.20× |  |  | N/A |
| iq2_xxs | N/A | N/A |  | 0.52× |  |  | N/A |
| iq2_xs | N/A | N/A |  | 0.29× |  |  | N/A |
| iq2_s | N/A | N/A |  | 0.53× |  |  | N/A |
| iq4_nl | N/A | N/A |  | **1.32×** |  |  | N/A |
| iq4_xs | N/A | N/A |  | **1.28×** |  |  | N/A |
| mxfp4 | N/A | N/A |  | **1.21×** |  |  | N/A |
| nvfp4 | N/A | N/A |  | N/A(注2) |  |  | N/A |
| tq2_0 | N/A | N/A |  | 0.60× |  |  | N/A |
| tq1_0 | N/A | N/A |  | 0.44× |  |  | N/A |
| q1_0 | N/A | N/A |  |  |  |  | N/A |

---

## 表 2:Repack 算子(我们自创了一个跟 ggml 不同的算法,所以有真正的 Win-B)

> Win-A = 调 strip/宽度旋钮;Win-B = 我们的新算法 vs ggml 官方算法。

| 算子 | Win-A·rvv | Win-A·k1 | Win-B·rvv | Win-B·k1 | Win-B·e2e·rvv | Win-B·e2e·k1 |
|---|---|---|---|---|---|---|
| q4_0 repack(decode) |  | 1.48× | 1.22× |  | **2.60×** | 0.74× |
| q4_0 repack(prefill) | 1.30× |  | 6.36× |  | **5.68×** |  |
| q4_1 repack(decode) |  |  | 正确性✓ |  | 上游卡(注3) | 上游卡 |
| q4_1 repack(prefill) | 空(注4) |  | 正确性✓ |  | 上游卡 | 上游卡 |
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
- 注4:q4_1 repack 的 prefill 旋钮没测(最容易补的一格)。
- 注5:q8_0 在 k1 上选宽档比窄档快 1.95×,但这个选择有没有传导到 llama e2e 还没测(最该补的)。
- 注6/7/8:见表 4、表 5 上文。
- 注9:**k1 实测**——q4_K 的 m1 vs mf2 旋钮传导到 llama:prefill ~1.26×、**decode ~1.22×**(初步,配对版确认中)。是「repack 关掉走 per-row vec_dot」regime 下隔离的旋钮效果,**不是说 vec_dot 打败了 repack GEMM**。意义:推翻了「decode 内存墙washes一切算力 win」。

---

# 没做清单(自查漏洞,如实列)

1. **block-dot e2e:只测了 q4_K(注9,~1.22×传导)**,其余 block-dot 的 e2e 还没测。
2. **IQ / FP4 / ternary 在 k1 上的 micro 全没测**(只在 rvv 测了)。
3. **q4_0 repack 在 k1、q5_0 repack 在 k1、q4_K repack 在 k1 都没做;q4_K repack k1 的正确性 oracle 还没跑**。
4. **6 个前向算子(softmax 等)0 性能测试**。
5. **q1_0 block-dot 没测;q5_0/q5_1 block-dot 的旋钮是 m1-only(没东西可调)**。
6. **能补的真 gap**:q8_0 在 k1 的选择 e2e(头号)、q4_1 repack prefill 旋钮(最易)、option-2 的真 pipeline 自动化 C3(深活)。

---

# 成熟度定位(对标 Triton 后端,不是 TVM)

我们对标的是 **Triton 的「偏后端」部分**(拿 tile 级 IR,自动做 layout/向量化/指令选择/软件流水 → 硬件码),**不是 TVM**(自动搜索调度空间)。按这个尺子诚实定位:

| 维度 | Triton 后端会做 | 我们现在 | 判定 |
|---|---|---|---|
| **派发**(选哪个 family/target) | 按 target 选后端 | capability-fact 选 RVV/IME,**零 core 分支、非字符串匹配** | ✅ 成熟、且是 novelty(N1/N2) |
| **算法选择**(repack vs block-dot) | Triton 不做(用户写) | option-2 编译器按能力选,实测对 | ✅ 比 Triton 还多一层(N3) |
| **宽度/LMUL 选择** | 自动 layout assignment | Gearbox 按 VLEN 自动选 LMUL/strip(2-3 选 1 + 可选实测) | 🟡 有,但只 1 个 capability bit,弱 |
| **向量化 + 指令选择** | 从 tile IR **自动**生成向量指令 | **手写**:emitter 里逐条敲 `__riscv_*`(~100 个),每个 quant 一个 emit 方法 | ❌ **手写,非自动**——「像玩具」的根源 |
| **软件流水 / ILP** | **自动**流水 pass | **手写**:emitter 里手摆 2-strip 交错;无任何自动流水 pass | ❌ 手写 |
| **通用 lowering** | 一个通用 pass 处理所有 kernel | **部分**:有通用 `emitWideningDotReduce`/`emitDequantize` 原语,但每个 quant 的解包是逐 kernel 手写(24 个) | 🟡 半通用 |

**一句话**:我们现在是「**capability 驱动的选择/派发层 + 手写 kernel 体**」。**选择/派发那层是真后端、是 novelty**(派发非字符串、还能选算法);但**代码生成那层(向量化/流水)是手写的,没有从 tile IR 自动 lower**——这就是为什么「去了字符串匹配,却还是手写 kernel 库」。

**通往成熟 Triton 后端的关键一步**(不是重写整个项目):把**一个** kernel 类(比如 dot-reduce)做成「从 tile/vector-dialect 级表示**自动**向量化 + 自动选 LMUL + 自动流水」的通用 lowering,证明后端能**自动生成**而不只是手写。这是把「玩具感」变「成熟 compiler」最直接、可论证的一步。**MLIR 的 vector/linalg dialect 我们现在一个都没用**(直接手写 EmitC)——这正是要补的入口。
