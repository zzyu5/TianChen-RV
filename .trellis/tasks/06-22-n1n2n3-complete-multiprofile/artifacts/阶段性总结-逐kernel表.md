# TianChen-RV 自查表 — 逐 (kernel × 板子) 优化证据(2026-06-25 v3,自查不自嗨)

> 原子单位 = kernel。分三部分:**Part 1 只有 Win-A(同原版算法,只 pass on/off)** / **Part 2 又有 Win-A 又有 Win-B(我们自创算法的 kernel)** / **Part 3 Win-C**。
> 同一 kernel,**rvv 和 k1 各占一行**;没做就写「未做」,不省略。

## 0. 术语(每个都讲清「为什么」)

| 记号 | 含义 + 为什么 |
|---|---|
| **✅实测** | 真跑了,数字是结果(胜/平/负都算)。 |
| **❌LOSS** | 实测,比 ggml 慢(诚实负)。 |
| **⚪未测·推定NULL** | 没真跑 e2e,**只是推定**。推定理由:decode(M=1)每 token 把全部权重从 RAM 读一遍,若内存带宽是瓶颈,则只改算力(LMUL/gather 指令)的优化时间被读卡住 ≈ 1.0。**注意:这是假设不是结论**,本次正在用 `wf25` 实测验证(见 §尾)。 |
| **⬜未测·GAP** | 能测、会出真数字,只是还没跑。 |
| **⛔BLOCKED** | 想测但缺**上游 ggml 组件**(如 q4_1 repack 要 ggml 的 `block_q8_1x4` 量化器+路由,ggml 没有),接不进 e2e。不是我们没做。 |
| **N/A** | 本质不适用(emit-only block-dot 无 LMUL 旋钮 → 无 Win-A 可 toggle;或 ggml 在 riscv 上根本没这个 kernel → 无合法 baseline)。 |

板子:`rvv`=SG2044 RVV1.0 **VLEN128**;`k1`=X60 RVV1.0 **VLEN256**+IME。RVV0.7 已弃。

---

# Part 1:只有 Win-A 的 kernel(我们 emit 的是 ggml 的**同一个算法**,优化 = LMUL/strip 旋钮 on/off)

> 列:**正确性**(我们 emit 对不对)/ **Win-A 旋钮·micro**(LMUL on/off,只有 K-quant 做了)/ **Win-A·e2e** / **同算法 emit·micro**(我们 emit vs ggml 出厂同算法 kernel,= 代码生成质量,不是 Win-B)。

## 1a. K-quant block-dot(有 Win-A LMUL 旋钮)

| Kernel | 板子 | 正确性 | Win-A 旋钮·micro(m1 vs mf2) | Win-A·e2e | 同算法 emit·micro(vs ggml `_vl256`) |
|---|---|---|---|---|---|
| q4_K | rvv | ✅byte-exact | ✅ m1 最佳(VLEN128) | ⚪未测·推定(wf25 实测中) | (rvv 上未单测 vs ggml) |
| q4_K | k1 | ✅byte-exact | ✅ **m1 把 loss 1.80→1.38×** | ⚪未测·推定(wf25 实测中) | ❌LOSS 1.72×(mf2 臂) |
| q6_K | k1 | ✅byte-exact | ✅ **m1 2.19→1.91×** | ⚪未测·推定 | ❌LOSS 2.26× |
| q3_K | k1 | ✅byte-exact | ✅ **m1 2.10→1.83×** | ⚪未测·推定 | ❌LOSS 2.13× |
| q5_K | k1 | ✅byte-exact | ✅ q4_K 共享 helper 自动覆盖 | ⚪未测·推定 | ✅TIE 0.998× |
| q2_K | k1 | ✅byte-exact | N/A(gather 型,无 LMUL loss 可缩) | — | ✅**WIN 1.016×** |

## 1b. 标准 quant block-dot(Win-A = N1 的 VLEN→LMUL family 选择,跨板子翻转)

| Kernel | 板子 | 正确性 | Win-A 旋钮·micro(N1 翻转) | Win-A·e2e | 同算法 emit·micro |
|---|---|---|---|---|---|
| q8_0(load-bearing N1) | rvv | ✅ | ✅ 选 m2,比 m1 快 +7.0% | ⬜GAP-6 | (micro-only) |
| q8_0 | k1 | ✅ | ✅ 选 m1,快 +6.9%(**赢家从 m2 翻 m1**=N1) | ⬜GAP-6 | (micro-only) |
| q4_1 | rvv | ✅ | ✅ elision +2.9% | ⬜GAP-6 | ✅2.47×(其算法家在 repack) |
| q4_1 | k1 | ✅ | ✅ elision **+11.2%** | ⬜GAP-6 | — |
| q4_0 | rvv | ✅ | ✅ within-m1 ~0.8% | ⬜GAP-6(经 repack 行到达 e2e) | (算法家在 repack §2) |
| q4_0 | k1 | ✅ | ✅(同上) | ⬜GAP-6 | — |
| q5_0 | rvv/k1 | ✅ | ⚪NULL 翻转(m1-only,tie ≤1.3%) | ⬜GAP | (micro-only) |
| q5_1 | rvv/k1 | ✅ | ⚪NULL 翻转(tie ≤0.05%) | ⬜GAP | (micro-only) |

## 1c. IQ block-dot(无 Win-A 旋钮;优化 = 标量 gather → 硬件 vluxei16,= 同算法的 emit 成熟度)

> 这不是 Win-A 旋钮(不能 toggle),是一次性把 gather 指令换成和 ggml 一样的硬件 `vluxei16`。列在 Part 1 因为**算法和 ggml 相同**。

| Kernel | 板子 | 正确性 | Win-A 旋钮 | Win-A·e2e | 同算法 emit·micro(vs ggml 真 vluxei16) |
|---|---|---|---|---|---|
| iq1_s | rvv | ✅ | N/A(emit-only) | ⚪推定NULL | ✅ gap **7.4→2.3×**(收 gap,非 beat-ggml) |
| iq1_m | rvv | ✅ | N/A | ⚪推定NULL | ✅ 9.3→5.8× |
| iq3_xxs | rvv | ✅ | N/A | ⚪推定NULL | ✅ 22.35→8.14× |
| iq3_s | rvv | ✅ | N/A | ⚪推定NULL | ✅ 5.88→5.04× |
| iq2_xxs | rvv | ✅ | N/A | ⚪推定NULL | ✅ 6.81→**1.91×**(最接近 parity) |
| iq2_xs | rvv | ✅ | N/A | ⚪推定NULL | ✅ 8.79→3.39× |
| iq2_s | rvv | ✅ | N/A | ⚪推定NULL | ✅ 5.60→1.88× |
| **iq1_s/m, iq2_*, iq3_* 在 k1** | k1 | — | — | — | **未做**(vluxei16 micro 只在 rvv 测了,k1 没测) |

## 1d. FP4-codebook block-dot(无旋钮;优化 = tiny codebook 寄存器 vrgather,同算法 emit 更优)

| Kernel | 板子 | 正确性 | Win-A 旋钮 | Win-A·e2e | 同算法 emit·micro(vs ggml 真 `_vl128`) |
|---|---|---|---|---|---|
| iq4_nl | rvv | ✅ | N/A | ⚪推定NULL | ✅ **1.32× WIN** |
| mxfp4 | rvv | ✅ | N/A | ⚪推定NULL | ✅ **1.21× WIN** |
| iq4_xs | rvv | ✅ | N/A | ⚪推定NULL | ✅ **1.28× WIN** |
| nvfp4 | rvv | ✅ | N/A | — | N/A(ggml 在 riscv 无 nvfp4 RVV kernel,只有 scalar → 无合法 baseline) |
| q1_0 | rvv | ✅(能 emit) | N/A | — | **未做**(q1_0 非主线 quant,没测 micro) |
| iq4_nl/xs, mxfp4 在 k1 | k1 | — | — | — | **未做**(只在 rvv 测) |

## 1e. Ternary block-dot

| Kernel | 板子 | 正确性 | Win-A | 同算法 emit·micro(vs ggml) |
|---|---|---|---|---|
| tq2_0 | rvv | ✅ | N/A | ❌LOSS 0.60×(compute-bound) |
| tq1_0 | rvv | ✅ | N/A | ❌LOSS 0.44× |
| tq2_0/tq1_0 在 k1 | k1 | — | — | **未做** |

## 1f. Win-A 旗舰 kernel(纯 Win-A,不是 block-dot)

| Kernel | 板子 | Win-A·micro(旋钮 on/off) | Win-A·e2e |
|---|---|---|---|
| **i16 dot-reduce contraction**(Win-A 旗舰) | rvv | ✅ **2.27–3.79×**(vs 同算法窄-LMUL,byte-exact) | N/A(非独立 llama 热路径;e2e 类比 = q4_0 repack-LMUL 那行) |
| i16 dot-reduce | k1 | ✅ **1.8–3.6×** | N/A |

## 1g. 前向算子(forward-pass operators)—— 一系列 ggml 算子,我们都能 emit,但**优化/性能基本没做**

> ⚠ 自查纠正:这一类我第一次漏了(grep 太窄)。代码里确实有一整套前向算子(`Ggml*F32Op` + `emitGgml*`),**都能生成、bit-identical 重写 ggml,但都没做 Win-A/B/C、也没测性能**。这是真实的「覆盖广但没优化」的现状——如实列出。

| Kernel | 算子 op / emit 方法 | 板子 | 正确性 | Win-A | Win-B | micro/e2e 性能 |
|---|---|---|---|---|---|---|
| **softmax** | `GgmlVecSoftMaxF32Op` / `emitGgmlVecSoftMaxF32`(+ `emitGgmlVExpfM2`) | rvv/k1 | ✅ 能 emit,bit-identical 重写 ggml | N/A(hard-pinned f32m2,无旋钮) | N/A(同算法重写,无不同实现可比) | **未做**(没测 micro,没接 e2e) |
| **silu** | `GgmlVecSiluF32Op` / `emitGgmlVecSiluF32` | rvv/k1 | ✅ | N/A | N/A | **未做** |
| **rms_norm** | `GgmlRmsNormF32Op` / `emitGgmlRmsNormF32` | rvv/k1 | ✅ | N/A | N/A | **未做** |
| **rope_norm**(RoPE) | `GgmlRopeNormF32Op` / `emitGgmlRopeNormF32` | rvv/k1 | ✅ | N/A | N/A | **未做** |
| **scale** | `GgmlVecScaleF32Op` / `emitGgmlVecScaleF32` | rvv/k1 | ✅ | N/A | N/A | **未做** |
| **quantize_row_q8_0**(激活量化器) | `GgmlQuantizeRowQ80Op` / `emitGgmlQuantizeRowQ80` | rvv/k1 | ✅ | N/A | N/A | **未做** |

**前向算子总结**:**6 个前向算子全能 emit + 正确(bit-identical 重写 ggml),但 0 优化、0 性能测试**。原因:它们是 ggml 同算法的忠实重写(没有「我们的不同算法」可比 = Win-B N/A),也是 hard-pinned f32m2(没有 LMUL 旋钮 = Win-A N/A)。**诚实定性:这是「kernel 覆盖」不是「优化证据」**——它们证明 compiler 能产出这些算子,但不是 N1/N2/N3 的性能主张。要把它们变成优化证据,需要先给一个可 toggle 的轴(比如 strip/LMUL),目前没有。

## 1h. 底层原语层(primitive ops,kernel 的搭建积木,不是独立算子)

代码里还有一大套 RVV 指令级原语(不是用户意义上的「算子」,是上面那些 kernel 的搭建块,**单独不测性能**):`Load/Store/Binary/Compare/Select/Reduce/MAcc/Move/Splat` + 加宽类 `WideningDotReduce/WideningMAcc/WideningProduct/WideningConvert` + `Strided*/Masked*/Segment2*/IndexedLoad-Store/MaskedIndexed*` + i32 标量类 `I32Add/Mul/Sub/CmpEq/Select/Load/Store` + 一大批 `Typed*PreRealizedBody`(类型化的预实现体)。其中 **`WideningDotReduce` 就是 §1f 那个 Win-A 旗舰(2.3–3.8×)的核心原语**;其余是积木,N/A 单独性能。

---

# Part 2:又有 Win-A 又有 Win-B 的 kernel(我们**自创算法** = block-as-lane repack / IME 矩阵)

> 列:**Win-A 旋钮·micro+e2e**(strip/LMUL on/off)/ **Win-B·micro+e2e**(我们的新算法 vs ggml 出厂同功能 kernel)。

## 2a. Repack(block-as-lane,16 列铺 lane,lane-wise vwmacc,省掉 per-block vredsum)

| Kernel | 板子 | 正确性 | Win-A·micro | Win-A·e2e | Win-B·micro(vs ggml 真 RVV) | Win-B·e2e |
|---|---|---|---|---|---|---|
| **q4_0 repack GEVM**(decode) | rvv | ✅ | LMUL 旧式(已弃口径) | ⚪推定NULL | ✅ **1.22×** | ✅ **~2.6×**(改了内存布局才传导) |
| q4_0 repack GEVM | k1 | ✅ | ✅ VLEN-strip **1.48×** | ✅ **1.31×**(SEAL,扛过内存墙) | (vs ggml repack,tie) | ❌**0.74× LOSS**(X60 autovec 更强,已披露) |
| **q4_0 repack GEMM**(prefill) | rvv | ✅ | ✅ 1.27–1.38× | ✅ 1.10×(t16,clean) | ✅ **6.36×** | ✅ **5.68×**(prefill 算力瓶颈,传导) |
| q4_0 repack GEMM | k1 | ✅ | **未做** | **未做** | **未做** | **未做** |
| q4_1 repack GEVM | rvv | ✅ | ✅ 1.80×(旧式口径) | ⛔BLOCKED | ✅ 仅 oracle 正确性 | ⛔BLOCKED(ggml 无 q8_1x4 量化器) |
| q4_1 repack GEVM | k1 | ✅ | **未做** | ⛔BLOCKED | **未做** | ⛔BLOCKED |
| q4_1 repack GEMM | rvv | ✅ | ⬜**GAP-4b(最易补)** | ⛔BLOCKED | ✅ 仅正确性 | ⛔BLOCKED |
| q8_0 repack GEVM | rvv | ✅ | (VLEN128 hl=8 only) | ⚪推定NULL | ❌**LOSS 1.3–1.7×**(q8_0 lean) | ⚪推定NULL |
| q8_0 repack GEVM | k1 | ✅ | ✅ strip **1.95×**(真 RVV1.0 auto-tune) | ⬜**GAP(头号)** | (vs ggml repack,tie) | ⬜**GAP**(strip 选中者 e2e 没测) |
| **q4_K repack GEVM**(主力) | rvv | ✅(norm 7e-7) | ⬜GAP(strip 没接线) | ⚪推定NULL | ❌**LOSS 1.5–2.1×**(vs 手调 `_vl128`) | ⚪推定NULL(脚印字节相同) |
| q4_K repack GEVM | k1 | ⬜未测(oracle pending) | **未做** | **未做** | ⬜GAP(vs ggml 自己 repack,tie-likely) | **未做** |
| q4_K repack GEMM | rvv | ✅(norm 7e-7) | **未做** | — | ❌LOSS 0.59–0.89× | ⚪推定LOSS |
| **q5_0 repack GEVM**(本次新增) | rvv | ✅(byte-exact+对抗 i16 边界) | (mf2/m1) | **未做** | ❌**LOSS 0.769×**(compute-bound) | **未做** |
| q5_0 repack GEVM | k1 | **未做** | **未做** | **未做** | **未做** | **未做** |

**Repack 总结**:**只有 q4_0 真 Win-B e2e 胜(2.6×/5.68×)**——因为它改了内存搬运。q8_0/q4_K/q5_0 是「正确扩展但 compute-bound → micro LOSS」,e2e 该 DECLINE(选回 ggml)。q4_1 整条 e2e 被上游 ggml 卡死(BLOCKED)。

## 2b. IME = RVM(我们 emit 的是 IME **leaf MAC 指令**,不是完整 GEMM——讲清楚)

> **我们生成了什么**:compiler 把高层 op 降到 IME 的 leaf 指令(`smt.vmadot` 等),emit 成 EmitC。**6 个 leaf op**:vmadot/vmadotu/vmadotsu/vmadotus(4 种符号组合)+ mma_slide(滑窗)+ tiled-matmul。
> **跑了什么**:在 K1 X60 上这些 leaf 被 SpacemiT 工具链编成真 `smt.*` 指令,跑出 **bit-exact 16/16**(vs scalar oracle)。
> **perf 测的是谁**:micro 测的是 **ggml-spacemit 的 `gemm_kernel_i8i4`**(它内部用 vmadot)vs ggml 真 RVV——**不是我们 emit 的完整 IME GEMM**(我们只 emit leaf,没 emit 完整 GEMM kernel)。这点必须说清,别含糊成「我们的 IME GEMM 胜」。

| 项 | 板子 | 正确性 | Win-B·micro | Win-B·e2e |
|---|---|---|---|---|
| IME MAC(6 个 leaf op) | k1 | ✅ **bit-exact 16/16** + 4-way 符号判别 | — | — |
| IME GEMM perf(ggml-spacemit kernel,用我们能 emit 的 vmadot) | k1 | — | ✅ **5.66×(M=1)/ 12.9×(prefill)** vs ggml 真 RVV(编译器恒定) | ⚪**IME-unit e2e = NULL**:pp 1.65× 但 decode 对照 1.47× 没塌→证明是换了整个 227-symbol kernel 家族,不是矩阵单元 |

**IME 总结**:**N2 结构(零-core-branch 加 family)= 强、已证**(加整个 family **0 行 core**);**IME 的 6 个 leaf op K1 bit-exact**;**perf 是 kernel-micro 胜、e2e unit-NULL**。注意 micro 的对象是 ggml-spacemit 的 GEMM,不是我们的完整 GEMM。

## 2c. option-2「编译器自己选算法」(跨 kernel,目标 q4_0,N3 旗舰)

| 阶段 | 板子 | 结论 |
|---|---|---|
| A/B/C1/C1b | host | ✅ 编译器靠 VLEN fact 选 repack/block-dot(无字符串匹配);packer plain→x16 memcmp==0(520 万 block) |
| M1 emit-identity | rvv | ✅ 自选路径 emit 出和直接 repack **字节相同**的 kernel |
| M2 e2e SEAL | rvv | ✅ 自选 kernel 真跑赢 **t1 7.0×**(复现历史 7.05×)/ t16 ~2.6–3.3×。诚实:SEAL 非独立新数;C3 真自动化仍 OPEN |

---

# Part 3:Win-C(改算法结构的 pass on/off)—— 讲清楚:**没有结构 novelty**

- **唯一的 Win-C 候选**:`reduction_structure` pass(在 i16 dot-reduce 上 toggle)。
- **现象**:pass on/off 有 **3.0–3.3×** 差。
- **但拆开看**:用 register-kept 分解做对照(把 per-iter 的 `out[0]` 留在寄存器),纯结构差 ≈ **1.00×**。
- **结论**:那 3× **全是 per-iter `out[0]` 内存往返**(emitter 的实现 artifact),**不是结构带来的**。所以 **结构-Win-C = NULL,未证明**。pass 保留(正确、正交)只作为 Win-A LMUL sweep 的结构 enabler,**不 claim 成 Win-C novelty**。
- **其它 kernel**:没有别的 pass 改算法结构 → 全表 Win-C = N/A / NULL。

---

# 自查结论:哪里漏了 / 哪里没做 / 哪里要小心

**真站得住的胜**:q4_0 repack 2.6×/5.68× e2e、4 个 FP4 micro-WIN、option-2 M2 SEAL t1 7.0×、IME kernel-micro 5.66×/12.9×、K-quant Win-A micro 缩 loss、K1 VLEN-strip 1.31×、i16 dot-reduce 2.3–3.8×。

**明确「未做」的(自查漏洞)**:
- **K-quant / IQ / FP4 / repack 的 Win-A·e2e 全是「未测·推定 NULL」**——本次 `wf25` 正在实测 q4_K 的(decode+prefill,k1+rvv)来验证推定对不对。
- **IQ/FP4/ternary block-dot 在 k1 全没测 micro**(只 rvv)。
- **q4_0 repack GEMM 在 k1 没做;q5_0 repack 在 k1 没做;q4_K repack 在 k1 oracle pending**。
- **6 个前向算子(softmax / silu / rms_norm / rope_norm / scale / quantize_row_q8_0)全能 emit + 正确,但 0 优化、0 性能测试**(§1g)——覆盖广但不是优化证据。q1_0 block-dot 没测 micro。
- **gap(可补,会出真数字)**:q8_0 VLEN-flip selection e2e(头号)、q4_1 repack GEMM Win-A micro(最易)、option-2 C3 producer(深活)。
- **BLOCKED(上游卡,非我们)**:q4_1 repack 全条 e2e(ggml 无 q8_1x4 量化器/路由)。

**要小心别回退的口径**:
1. IME e2e 不是 IME-unit 胜(1.65× 是 kernel-family swap)。
2. IME micro 测的是 ggml-spacemit 的 GEMM,我们只 emit leaf,不是完整 GEMM。
3. M2 是 SEAL(字节预定),锚 t1 7.0×,别 headline t16 3.3×。
4. repack m1「WIDE」是已弃 RVV0.7-form,不算 RVV1.0 auto-tune。
5. **「reasoned-NULL」是假设不是结论**——正在用 wf25 实测纠偏。
6. Win-C 无结构 novelty。
7. IQ vluxei16 是 maturity(收 gap),不是 beat-ggml。
