# TianChen-RV 阶段性总结 — 逐 kernel 优化证据表(2026-06-25)

> 目的:以 kernel 为单位理清「做了什么 / 没做什么 / 哪里可能有问题」。
> 数据源:`OPTIMIZATION-EVIDENCE-MATRIX.md` + `WIN-ABC-COMPLETENESS-AUDIT.md` + 各 FINDING。本表是中文汇总,数字以 matrix 为准。

## 0. 名词约定

- **类型**:`RVV` = 向量扩展(我们的主栈);`RVM` = 矩阵扩展(= SpacemiT **IME**,K1 上的非 RVV family,N2 的第二 family)。
- **板子**:`rvv` = Sophgo SG2044(RVV1.0,**VLEN128**,clang18);`k1` = SpacemiT X60(RVV1.0,**VLEN256** + **IME1**,只用 hart 0–3)。**RVV0.7 已弃**(按你的要求只做 RVV1.0 128/256 + K1 IME)。
- **N 轴**:N1 = 扩展异构作为一等 capability(VLEN→LMUL family 选择、能力派生);N2 = 零-core-branch 加新 family(IME);N3 = capability/resource-aware 的 tune 且实测胜出。
- **Win-A** = 同一算法的旋钮 on/off(LMUL 宽度 / strip 数),baseline = 同 kernel 旋钮关。**Win-B** = 改算法的新 kernel,baseline = **ggml 自己出厂的 RVV kernel**(不是 scalar)。**Win-C** = 改算法结构的 pass on/off。每个都要 **kernel-micro + llama-e2e** 两类。
- **状态记号**:✅实测 / ⚪reasoned-NULL(不测也能判,memory wall)/ ⛔BLOCKED(上游 ggml 缺口)/ ⬜GAP(可测但没测)/ N/A(本质不适用)。

---

## 1. Repack 类(RVV,block-as-lane,Win-B 是核心轴 + 多数也有 Win-A)

| Kernel | 类型/板子 | N | Win-A·micro | Win-A·e2e | Win-B·micro(vs ggml 真 RVV) | Win-B·e2e | Win-C |
|---|---|---|---|---|---|---|---|
| **q4_0 repack GEVM**(decode) | RVV / rvv+k1 | N3+N1 | ✅ VLEN-strip 1×16 vs 2×8 **1.48×**@k1(LMUL m1-vs-mf2 那条是 RVV0.7 hand-place,已弃) | ✅ VLEN-strip **1.31×**@k1(SEAL);LMUL ⚪flat | ✅ **1.22×**@rvv,norm=0 | ✅ **~2.6×**@rvv(memory-locality);**k1 0.74× LOSS**(已披露) | NONE |
| **q4_0 repack GEMM**(prefill) | RVV / rvv | N3 | ✅ WIDE/NARROW **1.27–1.38×** | ✅ **1.10× t16**(pp256,clean) | ✅ **6.36×** | ✅ **5.68×**(matmul-bound,micro≈e2e) | NONE |
| **q4_1 repack GEVM** | RVV / rvv | N3 | ✅ ~1.80×(RVV0.7-form,已弃口径) | ⛔继承 BLOCK | ✅ oracle PASS(仅正确性) | ⛔ 无 q8_1x4 quantizer / 路由(上游) | NONE |
| **q4_1 repack GEMM** | RVV / rvv | N3 | ⬜ GAP-4b(GEVM 测了,GEMM 复用同 harness,**最易补**) | ⛔继承 BLOCK | ✅ oracle PASS(正确性) | ⛔ 同上游 | NONE |
| **q8_0 repack GEVM** | RVV / rvv+k1 | N3 | ✅ strip-tune@k1 **1.95×**(选 hl=16,真 RVV1.0 auto-tune) | ⬜ GAP(.inc-swap 未做)= **audit gap#1/#3** | ✅ **LOSS ~1.3–1.7×**@rvv(q8_0 是 lean block-dot) | ⚪reasoned-NULL@128 | N/A |
| **q4_K repack GEVM**(主力 quant) | RVV / rvv | N3 | ⬜ GAP(strip-tune 未接线) | — | ✅ **LOSS ~1.5–2.1×**@128 vs ggml 手调 `_vl128`(正确性 norm 7e-7) | ⚪reasoned-NULL(脚印 byte 相同+memory-bound) | N/A |
| **q4_K repack GEMM**(prefill) | RVV / rvv | N3 | — | — | ✅ **LOSS 0.59–0.89×**(摊销缩小但不过 1.0) | ⚪reasoned-LOSS | N/A |
| **q5_0 repack GEVM**(本次新增) | RVV / rvv | N3 | (mf2/m1) | — | ✅ **LOSS 0.769×**(正确:byte-exact + 对抗 i16 边界 PASS) | ⬜未测 | N/A |

**Repack 一句话**:**q4_0 是唯一真 Win-B 胜(2.6× e2e)**——因为它把 memory traffic 改了(连续 16-block-as-lane 布局)。q8_0/q4_K/q5_0 都是**正确扩展但 compute-bound LOSS**(ggml 的宽 LMUL / 手调 _vl128 更强)——这就是 §8b 规律「heavy/gather 赢、compute-bound 输」,也是 N3 path-selection 该 DECLINE 的依据。

---

## 2. Block-dot 类(RVV)

### 2a. N1-selection block-dots(Win-A 列实为 N1 的 VLEN→LMUL family 翻转,不是 N3 tune)

| Kernel | 板子 | N | Win-A·micro(N1 翻转) | Win-A·e2e | Win-B |
|---|---|---|---|---|---|
| **q8_0** block-dot(load-bearing N1) | rvv↔k1 | N1 | ✅ m2@128 vs m1@256 **±~7% 翻转**(static cost-model 在 k1 选错 family,慢 ~6.5%) | ⬜ **GAP-6**(无 e2e 印证选中者真跑赢) | micro-only |
| **q4_1** block-dot | rvv↔k1 | N1 | ✅ elision 翻转 rvv+2.9% / **k1+11.2%** | ⬜ GAP-6 | ✅ 2.47×(micro) |
| **q4_0** block-dot | rvv↔k1 | N1 | ✅ within-m1 ~0.8% | ⬜ GAP-6(e2e 经 repack 那行到达) | (算法家在 repack) |
| **q5_0 / q5_1** block-dot | rvv↔k1 | N1 | ⚪NULL 翻转(m1-only,tie) | GAP | micro-only |

### 2b. K-quant block-dots — **Win-A 已做(本次重点)**

| Kernel | 板子 | N | Win-A·micro(LMUL tune,vs 自己 mf2) | Win-A·e2e | Win-B·micro(vs ggml `_vl256`) |
|---|---|---|---|---|---|
| **q4_K** | rvv+k1 | N3 | ✅ **m1 把 loss 1.80→1.38×**(both-board byte-exact) | ⚪NULL(compute-side,memory wall) | ✅ LOSS 1.72×(mf2 臂) |
| **q6_K** | rvv+k1 | N3 | ✅ **m1 2.19→1.91×**(ceiling m1) | ⚪NULL | ✅ LOSS 2.26× |
| **q3_K** | rvv+k1 | N3 | ✅ **m1 2.10→1.83×**(check 建了 VLEN256 判别对照) | ⚪NULL | ✅ LOSS 2.13× |
| **q5_K** | k1 | N3 | ✅ 由 q4_K 共享 helper 自动覆盖 | ⚪NULL | ✅ TIE 0.998× |
| **q2_K** | k1 | N3 | N/A(本身是 gather-WIN) | — | ✅ **WIN 1.016×** |

**K-quant 一句话**:compute-bound 的 q4_K/q6_K/q3_K 的 **Win-A LMUL tune 全做了**,m1 都把对 ggml `_vl256` 的 loss **缩小 ~13%(narrows-not-closes)**——残差是 ggml 的 nibble-split **算法形状**,不是 LMUL,正好是 N3-Gearbox 的动机。e2e 都是 ⚪reasoned-NULL(改 compute 不改 memory)。

### 2c. IQ block-dots — **vluxei16 maturity(全 7 个完成)**

| Kernel | 板子 | N | 性质 | micro(vs ggml 真 RVV `vluxei16`) |
|---|---|---|---|---|
| **iq1_s** | rvv | maturity | 标量 gather → 硬件 vluxei16 | ✅ gap **7.4→2.3×** |
| **iq1_m** | rvv | maturity | per-half gather | ✅ gap 9.3→5.8× |
| **iq3_xxs / iq3_s** | rvv | maturity | per-sign-group | ✅ 22.35→8.14× / 5.88→5.04× |
| **iq2_xxs** | rvv | maturity | signs64 emit-const(解 blocker) | ✅ **6.81→1.91×**(最接近 parity) |
| **iq2_xs / iq2_s** | rvv | maturity | 同 signs64 路 | ✅ 8.79→3.39× / 5.60→1.88× |
| **iq4_xs** | rvv | N3(FP4-codebook) | 本就 vrgather | ✅ **1.28× WIN** |

**IQ 一句话**:全 7 个 IQ block-dot 现在都发 **硬件 vluxei16**(和 ggml 同指令),从「标量 gather 输 5–22×」收到「接近 parity」。诚实定性:这是 **maturity(收 gap),不是 beat-ggml**;残差是窄 gather-LMUL + 标量 index 组装,不是 gather 本身。e2e ⚪decode-NULL(compute-side)。

### 2d. FP4-codebook block-dots — **N3 真 micro-WIN(覆盖完成)**

| Kernel | 板子 | N | Win-B·micro(vs ggml 真 `_vl128`) |
|---|---|---|---|
| **iq4_nl** | rvv | N3 | ✅ **1.32× WIN** |
| **mxfp4** | rvv | N3 | ✅ **1.21× WIN**(bit-exact,含 E8M0 denormal 新分支) |
| **iq4_xs** | rvv | N3 | ✅ **1.28× WIN**(跨 super-block 几何泛化) |
| **nvfp4** | rvv | N/A | bit-exact 但 ggml 无 RVV nvfp4 kernel → vs-scalar 0.57× = N/A |

**FP4 一句话**:tiny ≤16-entry codebook 的 **寄存器 vrgather** 在 VLEN128 把 ggml 的 32/64-lane gather 比下去——**4 个真 micro-WIN**(iq4_nl/mxfp4/iq4_xs + q2_K gather)。e2e ⚪decode-NULL。

### 2e. Ternary / forward-pass

| Kernel | 板子 | N | 状态 |
|---|---|---|---|
| tq2_0 / tq1_0 block-dot | rvv | — | 正确,micro LOSS(compute-bound),e2e ⬜GAP |
| silu / softmax / rope / rms_norm | rvv | — | hard-pinned,Win-A/B N/A,e2e ⬜未接线 |

---

## 3. Win-A 旗舰 + Win-C(RVV)

| Kernel | 板子 | N | 实测 |
|---|---|---|---|
| **i16 dot-reduce contraction**(Win-A 旗舰) | rvv+k1 | N3 | ✅ **2.27–3.79×**(rvv)/ 1.8–3.6×(k1),vs 同算法窄-LMUL 关旋钮。e2e 类比即 q4_0 repack-LMUL 那行 |
| **Win-C**(`reduction_structure` pass) | rvv | — | ⚠ **STRUCTURAL NULL**:pass on/off 3.0–3.3×,但 register-kept 分解证明纯结构 ≈1.00×,3× 全是 per-iter `out[0]` 内存往返(emitter artifact)。**结构-Win-C 未证明**;pass 保留为 Win-A LMUL sweep 的结构 enabler |

---

## 4. IME = RVM(N2 第二 family,k1 X60)

| 项 | 板子 | N | 实测 |
|---|---|---|---|
| **N2 零-core-branch 加 family** | k1 | **N2** | ✅ 已证 + 本次出**可复现 reviewer 证明**:founding commit **0 行 lib/Transforms**(core)、~1356 行 IME-local、3 行 wiring;ops 2/3/4 各 **0-core**;core 内 **0** family token、**0** `contains(matmul)&&contains(spacemit)` 反模式;选择靠 `spacemit.ime` **fact** 派生 |
| **6 个 IME op**(vmadot / tiled-matmul / vmadotu / vmadotsu / **mma_slide** / **mma_us**) | k1 | N2 | ✅ 每个 **0-core rapid-add** + K1 bit-exact。**mma_slide**=vmadot1 滑窗 MAC(**全新 kernel 类型**,conv/A-reuse);**mma_us**=补全符号族 {ss,uu,su,us}(check 抓到一个真 fail-closed bug 修了) |
| **IME perf — kernel micro** | k1 | N3 | ✅ **5.66×(M=1)/ ~12.9×(prefill GEMM)** vs **ggml 真 RVV**(编译器恒定,smt.vmadot 32× engaged)。**旧 5.51× 已撤**(那是对 forbidden 手卷 baseline,且混了 GCC-vs-clang 把它吹到 14.3×) |
| **IME perf — e2e** | k1 | N3 | ⚪/✅ **干净 SPACEMIT-toggle 已做**:pp **1.65×**,但 **decode falsifier tg16 = 1.47× 没塌到 1.0** → objdump 证明 ON 换了整个 **227-symbol kernel 家族**,不是 IME unit。**结论:IME-unit e2e 无法干净隔离出胜(NULL)**;能 defend 的 e2e 是 SPACEMIT 后端家族 ~1.65×,**明确不是矩阵单元** |

**IME 一句话**:**N2(结构,零-core-branch 加 family)= 强、已证、有可复现证明**;**IME 的 perf(N3)= kernel-micro 真胜、e2e unit-NULL(诚实)**——micro 胜不传导到 memory-bound 的 tinyllama decode/prefill,这是全 campaign 的 memory-wall 中心发现。

---

## 5. option-2 = N3 旗舰:「编译器自己选算法」(跨 kernel,不是单 kernel)

这是把 §8b 的 repack-vs-block-dot **选择**做成**真编译器能力**的尝试,目标 quant = q4_0(VLEN128):

| 阶段 | 板子 | 状态 |
|---|---|---|
| **A** 抽象 contraction op + identity pass | host | ✅ byte-exact,零行为变化 |
| **B** capability-aware **选择**(在 lib/Plugin/RVV) | host | ✅ **「编译器选算法」已真**:同一 binary 因 VLEN fact 翻转选 repack/block-dot,branch-free,无字符串匹配 |
| **C1/C1b** 声明 x16 contract + 隔离 packer(plain→x16 memcmp==0 / 5.2M block) | host | ✅ lit-only 完成 |
| **M1** emit-identity | rvv | ✅ 自选路径 emit 出和直接 repack **字节相同**的 mf2 kernel(SHA 一致)→ 关上 auto-selection gap |
| **M2** e2e SEAL | rvv | ✅ 自选 kernel 真跑赢:**t1 7.0×(精确复现历史 7.05×)/ t16 ~2.6–3.3×**,机制点名 = **stored-x16**(load 时打包一次)。诚实:这是 **SEAL(字节预定)不是独立新数**;**C3 集成自动化仍 OPEN** |

**option-2 一句话**:**编译器选 + emit + 产出 + 真跑赢已闭环(q4_0)**;只剩 **C3 producer**(在真 pipeline 里自动 author 抽象 op,把 hand-applied patch 自动化)是多 session 的深活。

---

## 6. 我的分析 —— 做了什么 / 没做什么 / 哪里可能有问题

### 做了(可 defend 的)
1. **N2(IME 结构)** 最强:零-core-branch + 6 op + 可复现证明 + K1 bit-exact。
2. **N3 真胜的几处**:q4_0 repack **2.6× e2e**(改 memory)、4 个 **FP4-codebook micro-WIN**、K-quant **Win-A LMUL tune**(缩小 loss)、IME **kernel-micro 5.66×/12.9×**、**option-2 编译器选算法 + M2 e2e SEAL**。
3. **覆盖广**:全 quant-family block-dot 正确;repack q4_0/q4_1/q8_0/q4_K/q5_0;IQ vluxei16 全 7;6 IME op。
4. **诚实纪律**:全程抓了 ~15 个 over-claim;trellis-check 挡下真 bug(VLEN256 fold 误编、IME fail-closed、q5_0 文档漂移)。审计:31 cell 实测 / 24 reasoned-NULL / 24 N/A / 7 真 gap / 4 上游 blocked。

### 没做(真 gap,可补)
- **gap#1 q8_0 VLEN-flip selection e2e**(k1):strip-tune 选中者是否扛过 memory wall(类比 q4_0 VLEN-strip 已证 1.31×)——**最高价值的真 gap**,但是大 campaign。
- **gap#2 q4_1 repack GEMM Win-A micro**:复用同 harness,**最易补**(注意:m1-vs-mf2 是 RVV0.7-form 口径,按弃 RVV0.7 应只测 strip 轴)。
- **option-2 C3 producer**:真 pipeline 自动化(多 session)。
- q5_1/q6_K repack 等:大概率又是 compute-bound 的 on-pattern LOSS(扩展但非胜)。

### 哪里**可能有问题** / 要小心的口径
1. **IME e2e 不能说成 IME-unit 胜**:1.65× prefill 是 **kernel-family swap**(227 符号),不是矩阵单元——必须按 decode-falsifier 口径报(这点已在 finding/matrix 里锁死,别回退)。
2. **IME micro 5.66× 的分解**:= 2.45×(IME-MAC)× 2.35×(ggml 出厂 kernel 偏弱);**测的是 ggml-spacemit 的 gemm_kernel_i8i4**,不是 tcrv 自己 emit 的完整 IME GEMM(tcrv 只 emit IME **leaf** op)——这点要明说,别含糊成「我们的 IME GEMM 胜」。
3. **option-2 M2 是 SEAL 不是独立新数**:t16 那 ~3.3× 有一半是本 session OFF baseline 偏慢;锚点用 **t1 7.0× 复现 + 字节相同**,别 headline 3.3×。
4. **repack 的 m1「WIDE」数字**(q4_0/q4_1 那些 2.1×/1.80×)是 **RVV0.7-form 口径**,不是 RVV1.0 auto-tune;按你「弃 RVV0.7」,这些不该再当 RVV1.0 的 Win-A 胜。
5. **K-quant / IQ / FP4 的 Win-A·e2e 是 reasoned-NULL 不是 gap**:compute-side 改 compute 不改 memory,decode 被 memory wall 盖住——这是**正当判定**,不要为了「填满 e2e」去硬测(审计已把它和真 gap 分开)。
6. **Win-C 没有结构 novelty**:那个 3× 是内存往返 artifact,纯结构 ≈1.00×;别 claim 结构 Win-C。
7. **q5_0 repack 没有 producer op**:`.td` 是唯一 qh-layout 契约(check 修了 5 处 INVERTED 文档漂移);将来要做 producer 必须按 NON-inverted 打包。

### 一句话总判
**N1/N2 结构主张 + N3 的「编译器选算法(option-2)」和若干真 micro/e2e 胜都立住了,证据广且诚实;剩下的是「补几个可测 gap」+「option-2 C3 自动化(深活)」,以及守住上面这些口径不回退。**
