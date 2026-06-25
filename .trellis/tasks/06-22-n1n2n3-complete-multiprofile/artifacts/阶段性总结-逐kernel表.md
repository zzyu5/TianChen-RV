# TianChen-RV 阶段性总结 — 逐 (kernel × 板子) 优化证据表(2026-06-25 重写)

> 规则:**同一个 kernel,rvv 的结果和 k1 的结果各占一行**。每个格子直接写「结论 + 一句原因」。

---

## 0. 先把术语讲清楚(NULL / BLOCK / GAP 到底是什么、为什么)

| 记号 | 中文 | 含义 + **为什么** |
|---|---|---|
| **✅** | 实测 | 真在板子上跑了,数字就是结果(胜/平/负都算实测)。 |
| **⚪ reasoned-NULL** | 推定无 e2e 提升 | **不用真跑 llama 也能判定 e2e ≈ 持平**。**为什么**:这类优化只改「**算力**」(把 compute 做快,比如加宽 LMUL、换 gather 指令),**不改「内存搬运量」**;而 llama 的 **decode**(每生成 1 token,矩阵宽 M=1)是**内存带宽瓶颈**——权重每步都要从 RAM 流过,算力再快也被「内存墙」盖住。所以 compute-side 的优化在 decode e2e 上 ≈ 1.0 = NULL。这是本 campaign 的中心结论「**kernel micro 胜 ≠ e2e 胜**」。 |
| **⛔ BLOCKED** | 上游 ggml 卡住 | **实验我们想做,但缺一个上游 ggml 的组件,不是我们 compiler 的锅**。例:q4_1 repack 要在 llama 里跑 e2e,需要 ggml 提供 `block_q8_1x4` 激活量化器 + q4_1-repack 的路由;ggml 根本没有(它的 `block_q4_1x16` 是另一套 ABI)→ 我们没法把它接进 e2e。= 不是没做,是没法做。 |
| **⬜ GAP** | 能测但还没测 | 这个实验**可以做、会出一个真数字**,只是这个 session 还没跑(是真·待办)。 |
| **N/A** | 本质不适用 | 这个轴对这个 kernel 不成立。例:emit-only 的 block-dot **没有 LMUL 旋钮** → 没 Win-A 可 toggle;forward-pass kernel 是 ggml 同算法的 **bit-identical 重写** → 没有「不同算法」可比 = Win-B N/A。 |

**轴定义**:Win-A=同算法旋钮 on/off(LMUL/strip);Win-B=改算法的新 kernel,**baseline 必须是 ggml 出厂的真 RVV kernel**(不是 scalar);Win-C=改结构的 pass on/off。
**板子**:`rvv`=Sophgo SG2044(RVV1.0,**VLEN128**);`k1`=SpacemiT X60(RVV1.0,**VLEN256** + **IME**)。RVV0.7 已弃。
**Win-C 全表统一**:唯一一个 Win-C(`reduction_structure` pass)被证明是 **结构-NULL**(pass on/off 有 3×,但纯结构 ≈1.00×,3× 全是 per-iter 内存往返 artifact)→ 所有 kernel 的 Win-C 列都是「无结构 novelty」,下表不再单列。

---

## 1. q4_0 repack(唯一真 Win-B e2e 胜的 kernel —— 因为它改了内存布局)

| Kernel | 板子 | N | Win-A·micro | Win-A·e2e | Win-B·micro | Win-B·e2e |
|---|---|---|---|---|---|---|
| q4_0 repack GEVM(decode) | **rvv** | N3 | ✅ 1.48× 那条在 k1;rvv 上 LMUL 那条是 RVV0.7 旧式(已弃口径) | ⚪ NULL(LMUL 只改算力,decode 内存墙盖住) | ✅ **1.22×** vs ggml 真 vec_dot | ✅ **~2.6×**(它把权重布局改成连续 16-block,**改了内存搬运** → 才传导到 e2e) |
| q4_0 repack GEVM(decode) | **k1** | N1+N3 | ✅ VLEN-strip 1×16 vs 2×8 **1.48×**(VLEN256 才放得下 16 lane) | ✅ **1.31×**(SEAL,选中者真跑赢,扛过内存墙) | — | ❌ **0.74× LOSS**(X60 的 autovec 比 repack 路强,已诚实披露) |
| q4_0 repack GEMM(prefill) | **rvv** | N3 | ✅ WIDE/NARROW 1.27–1.38× | ✅ 1.10×(t16,clean) | ✅ **6.36×** | ✅ **5.68×**(prefill 是算力瓶颈 M≫1,micro≈e2e) |

---

## 2. 其它 repack(都正确,但 Win-B 是 compute-bound 的诚实 LOSS)

| Kernel | 板子 | N | Win-A·micro | Win-B·micro(vs ggml 真 RVV) | Win-B·e2e |
|---|---|---|---|---|---|
| q4_1 repack GEVM | **rvv** | N3 | ✅ 1.80×(RVV0.7 旧式口径) | ✅ 仅正确性 oracle PASS | ⛔ **BLOCKED**:ggml 无 q8_1x4 量化器/路由(上游缺件,接不进 e2e) |
| q4_1 repack GEMM | **rvv** | N3 | ⬜ **GAP**:GEVM 测了,GEMM 复用同 harness,**最易补的一格** | ✅ 仅正确性 | ⛔ BLOCKED(同上游) |
| q8_0 repack GEVM | **rvv** | N3 | (VLEN128 只能 hl=8) | ✅ **LOSS ~1.3–1.7×**(q8_0 block-dot 很 lean,repack 没东西可省) | ⚪ NULL(脚印字节相同+memory-bound) |
| q8_0 repack GEVM | **k1** | N3 | ✅ strip-tune **1.95×**(VLEN256 选 hl=16,**这是真 RVV1.0 auto-tune**) | (k1 vs ggml 自己的 repack,tie-likely) | ⬜ **GAP**:strip 选中者 e2e 没测 = **audit 头号 gap** |
| q4_K repack GEVM(主力 quant) | **rvv** | N3 | ⬜ GAP(strip-tune 没接线) | ✅ **LOSS ~1.5–2.1×** vs ggml 手调 `_vl128`(正确:norm 7e-7) | ⚪ NULL(脚印 2304B 字节相同+decode 内存墙;对比 q4_0 是因为 q4_0 改了内存、q4_K 没改) |
| q4_K repack GEMM(prefill) | **rvv** | N3 | — | ✅ **LOSS 0.59–0.89×**(摊销缩小但不过 1.0) | ⚪ reasoned-LOSS(prefill 算力瓶颈+micro 输→e2e 也输) |
| q5_0 repack GEVM(本次新增) | **rvv** | N3 | (mf2/m1) | ✅ **LOSS 0.769×**(正确:byte-exact+对抗 i16 边界 PASS;q5_0 compute-bound) | ⬜ 未测 |

**这堆一句话**:q8_0/q4_K/q5_0 都是「**正确扩展但 compute-bound → 对 ggml 输**」,符合规律「gather/heavy 赢、compute-bound 输」;输的格子正好是 N3 path-selection 该 **DECLINE(选回 ggml block-dot)** 的依据,不是 bug。

---

## 3. Block-dot:N1 family 翻转(Win-A 列其实是 N1 的 VLEN→LMUL 选择)

> 这几条是「同 kernel,rvv 和 k1 选不同 LMUL family」——**赢家跨板子翻转**就是 N1 主张本身。

| Kernel | 板子 | N | Win-A·micro(N1 翻转) | Win-A·e2e |
|---|---|---|---|---|
| q8_0 block-dot(load-bearing N1) | **rvv** | N1 | ✅ 选 m2,比 m1 快 **+7.0%** | ⬜ **GAP-6**(没 e2e 印证选中者真跑赢) |
| q8_0 block-dot | **k1** | N1 | ✅ 选 m1,比 m2 快 **+6.9%**(**赢家从 m2 翻到 m1** = N1) | ⬜ GAP-6 |
| q4_1 block-dot | **rvv** | N1 | ✅ elision 轴 +2.9% | ⬜ GAP-6 |
| q4_1 block-dot | **k1** | N1 | ✅ elision 轴 **+11.2%** | ⬜ GAP-6 |
| q4_0 / q5_0 / q5_1 block-dot | rvv↔k1 | N1 | q4_0 ~0.8% 翻转;q5_0/q5_1 ⚪NULL(m1-only,tie) | GAP |

---

## 4. Block-dot:K-quant —— Win-A LMUL tune 本次全做了(rvv + k1 都验)

> Win-A 是「同一 q4_K block-dot,LMUL 旋钮 m1 vs 默认 mf2」。两块板子都做了 byte-exact 验证(check 还专门建了 VLEN256 判别对照,证明 gate 抓得住 fold-bug)。

| Kernel | 板子 | N | Win-A·micro(m1 vs mf2) | Win-A·e2e | Win-B·micro(vs ggml `_vl256`) |
|---|---|---|---|---|---|
| q4_K block-dot | **rvv** | N3 | ✅ m1 更快(VLEN128 也是 m1 最佳) | ⚪ NULL(改算力不改内存) | — |
| q4_K block-dot | **k1** | N3 | ✅ **m1 把对 ggml 的 loss 1.80→1.38×**(缩小但不过 1.0) | ⚪ NULL | ✅ **LOSS 1.72×**(mf2 臂) |
| q6_K block-dot | **k1** | N3 | ✅ **m1 2.19→1.91×**(ceiling=m1) | ⚪ NULL | ✅ LOSS 2.26× |
| q3_K block-dot | **k1** | N3 | ✅ **m1 2.10→1.83×** | ⚪ NULL | ✅ LOSS 2.13× |
| q5_K block-dot | **k1** | N3 | ✅ q4_K 共享 helper 自动覆盖 | ⚪ NULL | ✅ TIE 0.998× |
| q2_K block-dot | **k1** | N3 | N/A(本身 gather,无 LMUL loss 可缩) | — | ✅ **WIN 1.016×** |

**K-quant 一句话**:compute-bound 的 q4_K/q6_K/q3_K 的 **Win-A 旋钮全做了**,m1 都把对 ggml 手调 `_vl256` 的 loss **缩小 ~13%(narrows-not-closes)**;残差是 ggml 的 nibble-split **算法形状**(不是 LMUL)= N3-Gearbox 的动机。**e2e 全是 ⚪NULL**(原因见术语:LMUL 改算力、decode 内存墙)。

---

## 5. Block-dot:IQ —— vluxei16 maturity(rvv,全 7 个完成)

> 这不是「Win」轴,是 **maturity / 正确性 + 收 gap**:把标量 gather 换成和 ggml 一样的硬件 `vluxei16`。

| Kernel | 板子 | N | micro(vs ggml 真 RVV vluxei16) | 说明 |
|---|---|---|---|---|
| iq1_s | **rvv** | maturity | ✅ gap **7.4→2.3×** | 标量→硬件 gather,byte-exact |
| iq1_m | **rvv** | maturity | ✅ 9.3→5.8× | per-half gather |
| iq3_xxs / iq3_s | **rvv** | maturity | ✅ 22.35→8.14× / 5.88→5.04× | per-sign-group |
| iq2_xxs / iq2_xs / iq2_s | **rvv** | maturity | ✅ 6.81→**1.91×** / 8.79→3.39× / 5.60→1.88× | 解了 signs64 blocker(emit-const,无 ODS) |
| iq4_xs | **rvv** | N3 | ✅ **1.28× WIN** | 它本就是 vrgather(归 FP4 codebook,见 §6) |

**IQ 一句话**:7 个 IQ block-dot 现在都发硬件 `vluxei16`(和 ggml 同指令)。诚实定性:**这是 maturity(收 gap),不是 beat-ggml**——残差是窄 gather-LMUL + 标量 index 组装,不是 gather 本身。e2e 是 ⚪NULL(compute-side)。

---

## 6. Block-dot:FP4-codebook —— N3 真 micro-WIN(rvv,覆盖完成)

| Kernel | 板子 | N | Win-B·micro(vs ggml 真 `_vl128`) |
|---|---|---|---|
| iq4_nl | **rvv** | N3 | ✅ **1.32× WIN** |
| mxfp4 | **rvv** | N3 | ✅ **1.21× WIN** |
| iq4_xs | **rvv** | N3 | ✅ **1.28× WIN** |
| nvfp4 | **rvv** | N/A | bit-exact,但 **ggml 在 riscv 上根本没有 nvfp4 的 RVV kernel**(只有 scalar)→ 没有合法 Win-B baseline = N/A(不是 loss) |

**FP4 一句话**:tiny ≤16-entry codebook 用**寄存器 vrgather**,在 VLEN128 把 ggml 的 32/64-lane gather 比下去 → **4 个真 micro-WIN**(iq4_nl/mxfp4/iq4_xs + q2_K)。e2e ⚪NULL(compute-side)。

---

## 7. IME = RVM(k1 X60,N2 第二 family)

| 项 | 板子 | N | 结论 |
|---|---|---|---|
| N2 零-core-branch 加 family | **k1** | **N2** | ✅ 已证 + 可复现:加整个 IME family **0 行 core(lib/Transforms)**、~1356 行 plugin、3 行 wiring;后续每个 op 都 0-core;core 内 0 family 字符串、0 反模式;靠 `spacemit.ime` **fact** 派生选择 |
| 6 个 IME op | **k1** | N2 | ✅ 每个 0-core + K1 bit-exact。含 **mma_slide**(滑窗 MAC,全新 kernel 类型)+ **mma_us**(补全符号族) |
| IME kernel-micro | **k1** | N3 | ✅ **5.66×(M=1)/ ~12.9×(prefill GEMM)** vs **ggml 真 RVV**(编译器恒定)。**旧 5.51× 已撤**(对 forbidden 手卷 baseline + 混了编译器) |
| IME **e2e** | **k1** | N3 | ⚪ **IME-unit e2e = NULL**:干净 SPACEMIT-toggle 出 pp 1.65×,但 **decode 对照 1.47× 没塌到 1.0** → 证明那是换了整个 **227-symbol kernel 家族**,不是矩阵单元。**为什么 NULL**:同 decode 内存墙 + IME 增益无法和 family-swap 干净分离。能 defend 的只是「SPACEMIT 后端家族 ~1.65×」,**不是 IME unit** |

**IME 一句话**:**N2(结构,零-core-branch)= 强、已证**;**IME perf(N3)= kernel-micro 真胜,但 e2e unit-NULL(诚实)**。注意:micro 测的是 **ggml-spacemit 的 `gemm_kernel_i8i4`**——tcrv 只 emit IME **leaf** op,不是我们 emit 完整 IME GEMM(这点别含糊)。

---

## 8. option-2 = N3 旗舰「编译器自己选算法」(跨 kernel,目标 q4_0)

| 阶段 | 板子 | 结论 |
|---|---|---|
| A/B/C1/C1b(抽象 op + 选择 + 声明 contract + 隔离 packer) | host | ✅ 「编译器选算法」已真:同一 binary 因 VLEN fact 翻转选 repack/block-dot,无字符串匹配;packer plain→x16 memcmp==0(520 万 block) |
| **M1** emit-identity | **rvv** | ✅ 自选路径 emit 出和直接 repack **字节相同**的 mf2 kernel |
| **M2** e2e SEAL | **rvv** | ✅ 自选 kernel 真跑赢:**t1 7.0×(精确复现历史 7.05×)**/ t16 ~2.6–3.3×。机制=stored-x16。**诚实:这是 SEAL(字节预定)不是独立新数**;**C3 真 pipeline 自动化仍 OPEN**(多 session) |

---

## 9. 我的分析:做了 / 没做 / 哪里可能有问题

**真站得住的胜**:q4_0 repack **2.6× e2e**(唯一改内存的)、**4 个 FP4 micro-WIN**、option-2 **M2 e2e SEAL(t1 7.0×)**、IME **kernel-micro 5.66×/12.9×**、K-quant **Win-A 缩小 loss**、K1 VLEN-strip **1.31×**。

**真 gap(能测没测,可补)**:
- `q8_0 VLEN-flip selection e2e`(k1):strip 选中者扛不扛内存墙——**最高价值**,但大 campaign。
- `q4_1 repack GEMM Win-A micro`(rvv):复用同 harness,**最易补**。
- option-2 **C3 producer**:真 pipeline 自动化(深活,多 session)。

**可能有问题 / 要守住别回退的口径(7 条)**:
1. **IME e2e 不能说成 IME-unit 胜**——1.65× 是 kernel-family swap,不是矩阵单元(decode-falsifier 已锁死)。
2. **IME micro 测的是 ggml-spacemit 的 GEMM**,不是 tcrv 自己 emit 的完整 IME GEMM(tcrv 只 emit leaf op),别含糊。
3. **M2 是 SEAL 不是独立新数**;锚点用 t1 7.0× 复现 + 字节相同,别 headline t16 的 3.3×(有一半是本次 OFF baseline 偏慢)。
4. **repack 的 m1「WIDE」数字(2.1×/1.80×)是已弃的 RVV0.7-form**,不是 RVV1.0 auto-tune。
5. **K-quant/IQ/FP4 的 Win-A·e2e 是 reasoned-NULL 不是 gap**——别为「填满 e2e」去硬测(memory wall,正当判定)。
6. **Win-C 没有结构 novelty**(3× 是内存往返 artifact)。
7. **q5_0 repack 没 producer op**——`.td` 是唯一 qh-layout 契约,将来 producer 必须 NON-inverted 打包。

**一句话总判**:N1/N2 结构 + N3 的「编译器选算法(option-2)」+ 若干真 micro/e2e 胜都立住,证据广且诚实;剩下是补几个可测 gap + option-2 C3 自动化(深活),以及守住上面这些口径。
