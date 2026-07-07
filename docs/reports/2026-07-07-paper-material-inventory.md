# 论文素材现状清单 — 2026-07-07（裁决二 板批落表后）

**目的**：把当前**已落表**的每条论文可用主张，映射到一个具体的证据表行指针，并标注证据成色。
纪律：[NG-4] 措辞守（internal A/B ≠ beat；beat 需过 [PERF-1] 八门、vs 同-ISA 框架内核、分相报告）；
scalar/naive 只作 internal sanity，**绝不**作贡献倍数。三贡献框架见 `.trellis/spec/index.md`
（C1 合取存在性 / C2 边际成本 / C3′ 能力键控模式库）。

**证据成色图例**
- `board-proven` — 真硅实测、T-N 显著、协议达标（preflight/freq-lock/median+IQR/cold>3×L3/restore）。
- `kernel-only` — 真硅实测但 KERNEL-micro；e2e 传导缺席（compute/latency-bound 多半 washes，见 memory: kernel-wins-dont-transplant）。
- `pending` — 机理投影 / 未跑板 / 八门未闭（②③④ 类）。
- `structural-block` — 阻塞是 emitter 成熟度（construction-queue），code-verifiable，非测量、未花板。

主表位置：`experiments/active/result-tables/`（T3_A=rvv/VLEN128，T3_B=k1/VLEN256，T8=win/loss/gap ledger，
T-N=噪声地板，T-PERF1=八门台账）。每条主张的 `T8:<entry_id>` 指向 T8 行。

---

## L1 — 路径选择赢（option-2 能力键控 repack-vs-blockdot 路由；前端 novelty → C3′ 的路径分支 / C1 栈内实体）

| # | 主张 | headline | 表行指针 | 成色 | [NG-4] 边界 |
|---|---|---|---|---|---|
| L1-1 | q4_0 repack GEMM prefill 路由赢（VLEN128） | **e2e prefill 5.9195×** [5.91,5.934] IQR0.25%（>200× T-N 地板）vs stock ggml block-dot | `T8:q4_0-repack-gemm-rvv-vlen128-prefill-5.92x`（win_type C，**5/8 门 PASS**）；`T3_A` 镜像行；sealed cell `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/`；八门台账 `T-PERF1_q4_0_vlen128_prefill_8gate.md` | **board-proven（e2e，VLEN128-conditional）** | **未封普适 beat**：②VLEN256-flip-lit ③k1/VLEN256-objdump-seal ④micro↔e2e-Amdahl 未闭；门①=FMA-fold **bounded-ULP**（非 byte-exact）；beat 措辞 LOCKED |
| L1-2 | 同栈 decode 伴随赢（VLEN128） | **e2e decode tg32 1.9102×** [1.904,1.9196] 带宽饱和（ceiling-bounded） | 同上 `T8:...-5.92x` 行内伴随；`T-N` q4_0-decode-tg32 行 | **board-proven（e2e，VLEN128）** | 机制=packed-weight 连续流式→memory locality，非更快 codegen |
| L1-3 | VLEN-flip 诚实反面（k1/VLEN256） | prefill **PARITY 1.0022×**（compute-bound）; decode **0.854× LOSS**（bandwidth-bound，未吃满 78% vs 对手 91%） | `T8:q4_0-gemm-k1-vlen256-prefill-parity` + `T8:q4_0-gevm-k1-vlen256-decode-P1-roofline-GAP`；`T3_B` lineA 行 | **board-proven（e2e，k1/VLEN256）** | 赢是 **VLEN128-conditional**；VLEN256 是 VLEN-flip MIRROR；诚实登记为 real GAP（misselection），非 confound |
| L1-4 | q4_K repack GEMM prefill 路由**候选**（VLEN128；opponent-absence 已证） | **结构 path-win 候选 CONFIRMED**；吞吐 **PARITY 0.94–0.97×**（ours 略慢 vs ggml `_vl128`-tuned block-dot） | `T8:q4_K-repack-gemm-rvv-vlen128-prefill-L1-candidate`（win_type C；**opponent-absence PROVEN 2-ways**：源码 `repack.cpp:4616 case128=TODO no-op` + LIVE vlenb=128 探针 NULLPTR；8 门未走）；`T3_A` 镜像行；cell `experiments/active/kquant-l1-q4k-q5k-repack-prefill/`（opponent_probe.md + dispatch_probe_raw.txt + prefill_paired.csv + kernels_objdump_seal.objdump） | **candidate（board-probed opponent-absence + kernel-prefill 吞吐；八门未走）** | **禁 beat**：opponent 无 working repack@128 已证，但八门未走、单核、opponent=单线程 block-dot loop（kernel-axis proxy，非 threaded mul_mat）、吞吐-only 未复验数值；结构 path-win **≠ 吞吐 win**（q4_K 是 parity，ggml block-dot 已 VLEN128-调优）；⚠ memory 纠正：旧「ggml q4_K repack@256 hardcode/opponent@128 事实错」refine 为「`ggml_gemm_q4_K_16x1_q8_K` 存在但 VLEN256-gated」，「无 WORKING repack@128」仍成立 |
| L1-5 | q5_K repack GEMM prefill 路由**候选**（VLEN128；opponent 全 VLEN 无 repack） | **结构 path-win 候选 CONFIRMED**；ours **~1.50–1.62× faster** vs opponent **UNTUNED** generic block-dot | `T8:q5_K-repack-gemm-rvv-vlen128-prefill-L1-candidate`（win_type C；opponent q5_K **零** RISC-V repack：`repack.cpp:4644` 仅 neon 子分支；8 门未走）；`T3_A` 镜像行；同 cell | **candidate（board-probed opponent-absence + kernel-prefill 吞吐；八门未走）** | **禁 beat（即便 ours 更快）**：opponent=UNTUNED generic block-dot（最软基线，无 `_vl128` 变体）；八门未走、单核、单线程 proxy、吞吐-only 未复验；~1.5× 是**争夺格候选数**非封印 win |
| L1-6 | q6_K repack GEMM+GEVM prefill 路由**候选**（VLEN128；opponent-absence 已证 + **板上数值已证**） | **结构 path-win 候选 CONFIRMED + SILICON byte-exact-integer 数值已证**；吞吐 **LOSS 0.18–0.19×（ours ~5.5× SLOWER）** vs ggml mature block-dot | `T8:q6_K-repack-gemm-rvv-vlen128-prefill-L1-candidate`（win_type C；opponent q6_K **零** RISC-V repack@任何 VLEN：get_tensor_traits 无 `ggml_cpu_has_riscv_v()` 分支、NEON-only；LIVE vlenb=128 探针 NULLPTR；8 门未走）；`T3_A` 镜像行；cell `experiments/active/kquant-l1-q6q2q3-repack/`（numeric_silicon.txt/.csv + opponent_probe.md + dispatch_probe_raw.txt + prefill_paired.txt/.csv + kernels_objdump_seal.txt） | **candidate（board-numeric-verified：SILICON byte-exact-integer INT-mismatch=0 全 8 shape GEVM+GEMM×2 seed + bounded-norm 6.83e-07；opponent-absence board-probed；**prefill LOSS**；八门未走）** | **禁 beat（moot：LOSS 不可能是 beat）**：opponent 无 repack@任何 VLEN 已证 + 板上数值 byte-exact-int 已证，但吞吐 ~5.5× 慢（first-construction **un-pipelined** emit：g6.o 21 vsetvl ~0.2–0.3 MAC/cycle ~20× off peak vs ggml MATURE hand-tuned block-dot）；八门未走、单核、单线程 proxy；结构 opening 存在但当前构造**未转成吞吐** = marginal-cost/maturity 证据，perf 轴 pending |
| L1-7 | q2_K repack GEMM+GEVM prefill 路由**候选**（VLEN128；opponent case128=TODO no-op + **板上数值已证**） | **结构 path-win 候选 CONFIRMED + SILICON byte-exact-integer 数值已证**（含 dual d/dmin + bsums-min fold）；吞吐 **LOSS 0.20–0.21×（ours ~5× SLOWER）** vs ggml mature `_vl128` block-dot | `T8:q2_K-repack-gemm-rvv-vlen128-prefill-L1-candidate`（win_type C；opponent 有 riscv 分支但 `case 128:{break;}//TODO` → nullptr@128（256-only）= 同 q4_K 先例；LIVE 探针 NULLPTR；8 门未走）；`T3_A` 镜像行；同 cell | **candidate（board-numeric-verified：SILICON byte-exact-integer INT-mismatch=0 全 8 shape 含 dual d/dmin+bsums-min fold + bounded-norm 5.77e-07；opponent-absence board-probed；**prefill LOSS**；八门未走）** | **禁 beat（moot：LOSS）**：opponent 无 WORKING repack@128 已证（case128=TODO）+ 板上数值 byte-exact-int 已证（含双 d/dmin 折叠），但吞吐 ~5× 慢（g2.o **77 vsetvl** dual d/dmin min-fold un-pipelined vs ggml MATURE `ggml_vec_dot_q2_K_q8_K_vl128`）；八门未走、单核、单线程 proxy；结构 opening 未转吞吐 = marginal-cost 证据，perf 轴 pending |
| L1-8 | q3_K repack GEMM+GEVM prefill 路由**候选**（VLEN128；opponent **最强 absence** + **板上数值已证**） | **结构 path-win 候选 CONFIRMED（家族最强 absence）+ SILICON byte-exact-integer 数值已证**；吞吐 **LOSS 0.176×（ours ~5.7× SLOWER）** vs ggml mature `_vl128` block-dot | `T8:q3_K-repack-gemm-rvv-vlen128-prefill-L1-candidate`（win_type C；opponent q3_K **完全不在 selector**（`grep GGML_TYPE_Q3_K\|q3_K_16x1\|repack_q3_K repack.cpp = 0`）+ ggml 里**任何地方都没有 q3_K repack kernel** = 家族最强 absence；LIVE 探针 NULLPTR；8 门未走）；`T3_A` 镜像行；同 cell | **candidate（board-numeric-verified：SILICON byte-exact-integer INT-mismatch=0 全 8 shape + bounded-norm 4.36e-07（一个 worst_ulp=917132 outlier=near-zero-crossing cancellation 非缺陷，INT pass EXACT）；opponent-absence board-probed（最强）；**prefill LOSS**；八门未走）** | **禁 beat（moot：LOSS）**：opponent 零 q3_K repack（任何 VLEN、任何 ggml 位置）+ 板上数值 byte-exact-int 已证，但吞吐 ~5.7× 慢（g3.o 13 vsetvl un-pipelined vs ggml MATURE `ggml_vec_dot_q3_K_q8_K_vl128`）；八门未走、单核、单线程 proxy；结构 opening（最强）未转吞吐 = marginal-cost 证据，perf 轴 pending |

> L1 一句话：路径选择（option-2）在 VLEN128 上兑现 e2e 5.9× prefill / 1.91× decode（board-proven，但 5/8 门、
> beat 措辞锁）；在 VLEN256 上 prefill 打平、decode 反成 GAP —— 赢有明确 VLEN 边界，反面同表登记。
> **新增 q4_K/q5_K（board-harvest2 增补二）**：两者 opponent 都无 working repack@VLEN128（q4_K `case128=TODO`、
> q5_K 全 VLEN 无 riscv 分支，源码+LIVE 探针双证）→ 结构 path-win **候选** CONFIRMED（同 q4_0 先例）；但吞吐
> q4_K=PARITY、q5_K=~1.5×-vs-UNTUNED，**八门未走 → 禁 beat，仅 candidate 数**（结构 path-win ≠ 吞吐 win）。
> **家族收口 q6_K/q2_K/q3_K（board-probe，kernels @924dc31f「K-quant 家族完整」）**：**K-quant repack 家族 L1 探针完整 = 5 格
> oracle-GREEN（q4_0/q4_K/q5_K/q6_K/q2_K/q3_K 的构造正确性）+ 3 格 board-numeric-confirmed（q6_K/q2_K/q3_K 首次把导出的
> `.kernel.c` 上板 clang-17 编译+跑 rvv 硅、对独立逐块参考 = SILICON byte-exact-integer INT-mismatch=0 全 8 shape GEVM+GEMM×2 seed
> + bounded-norm <7e-7，把构造 oracle MODEL 升级为硬件运行）**。opponent-absence 三格全证（源码 f3e1828 + LIVE vlenb=128 探针）：
> q2_K case128=TODO（同 q4_K）、q6_K 无 riscv 分支（同 q5_K）、**q3_K 完全不在 selector 且 ggml 里没有任何 q3_K repack = 家族最强
> absence**。但吞吐**三格全 LOSS**：q6_K 0.18–0.19×(~5.5×慢)、q2_K 0.20–0.21×(~5×)、q3_K 0.176×(~5.7×)——诚实非-cache 病因=
> first-construction correctness-first **un-pipelined** emit（q2_K GEMM 77 vsetvl、q6_K 21、q3_K 13；~0.2–0.3 MAC/cycle ~20× off
> peak）vs opponent MATURE 手调 `ggml_vec_dot_q{2,3}_K_q8_K_vl128`。**[NG-4] moot（LOSS 不可能是 beat）；八门未走 → 仅 candidate**：
> 结构 opening + 板上数值双证已立，但当前构造**未把结构 opening 转成吞吐** = marginal-cost / compiler-maturity 证据，perf 轴 pending。

---

## L2 — 调度赢 / 选择键结论（kernel 内 LMUL·指令调度·数值档；→ C3′ 键控模式）

| # | 主张 | headline | 表行指针 | 成色 | [NG-4] 边界 |
|---|---|---|---|---|---|
| L2-1 | iq2_xxs pair-batching **翻盘**（GAP-1 闭环范例） | vs-generic **0.735× LOSS → 2.570× WIN**；内部 A/B(PRE/POST) **3.52×**；ULP0 | `T8:format-micro-iq2_xxs-gapsb-pair-batching-flip`（win-**CANDIDATE**，4/8 门）；cell `gapsb_derisk_iq2xxs/` | **kernel-only（win-候选）** | 唯一 gate-clean 对手=scalar generic；vs-SIMD-dispatch(gcc-15) 2.53× 更快但 **REFUSE**（clang20-vs-gcc15 compiler-confound）；不外推家族 |
| L2-2 | **GAP-P1 键控修 = 证伪**（选择错误例，反向 misselection） | 「widen-to-m1」在**可达 prefill GEMM** 上 **2.72× REGRESSION**；q8_0 GEVM 亦慢 11% | `T8:q4_0-gemm-k1-vlen256-GAP-P1-fix-confirm-REGRESSION`；cell `p1-gemm-vlen256-fix-confirm/` | **board-proven（kernel-micro；负结果）** | 宽 m1 核 **存在但严格更慢**（f32m4 寄存器压力）→ 非缺模式；正确 VLEN256 变体是 mf2 **wide-strip**（half_lanes16）非 whole-LMUL；诚实负结果 |
| L2-3 | q8_0 **do-not-widen VINDICATED**（选择键确认） | 3-way：wide-hl16 mf2 **1.72** > m1 whole-LMUL **1.55** > narrow-hl8 **1.27** GB/s；wide/m1 **1.111×** | `T8:q8_0-gevm-k1-vlen256-mf2-vs-m1-do-not-widen-VINDICATED` | **board-proven（kernel-micro 选择键）** | 确认 ledger「q8_0 mf2 快于 m1」；off VLEN256 production path（fact-3，decode 降级 block-dot） |
| L2-4 | item4 fcvt-reschedule（emitter-sched 税闭合） | k1/VLEN256 our-m2 4096→3617ns **+4.37%** vs factory；「+11% fill」溶解为 fcvt-tax | `T8:q8_0-item4-fcvt-reschedule`（7/8 门，④ micro-only） | **board-proven（kernel-micro，board-conditional）** | C-structure lever（clang 尊重的 objdump 真移）；instruction-golf sanity-EXCEPTION |
| L2-5 | 双档（数值 strict/relaxed）税账 | q8_0 有 relaxed；**iq4_xs/tq1_0 无 relaxed body → 税不可测** | `T8:format-micro-iq4_xs-tq1_0-GAP-NUM-relaxed-not-materialized`（q8_0 侧见 `T8:q8_0-deferred-ordered-fold-P2c` / `T8:q5_K-2c-deferred-reduce-vs-aux8-roundtrip`） | **q8_0 档 board-proven；iq4_xs/tq1_0 档 structural-block** | relaxed 仅 q8_0 materialized（`RVVToEmitCBlockQuantLinear.cpp:7182-7188` fail-close 其余）；两格 ours 已是 ULP0 FAITHFUL 侧；税账目前 q8_0-only，其余格 = 立项（先建 relaxed body） |

> L2 一句话：调度轴既有翻盘赢（iq2_xxs，kernel-only 候选）又有 board-proven 的**选择键结论**——
> 一个键控修被证伪（P1 反向 misselection）、一个键控决定被证成（q8_0 不广化）；两者合起来即 C3′
> 「机制选变体、能力谓词键控」的正反双证。数值双档税账目前只在 q8_0 可算（其余 structural-block）。

---

## L3 — 字节赢（内存流量；→ C3′ memory-axis 变换）

| # | 主张 | headline | 表行指针 | 成色 | [NG-4] 边界 |
|---|---|---|---|---|---|
| L3-1 | **G2 rms_norm→mul 融合首个 L3 字节实测** | wall **1.308×**（T-N PASS，~19× 地板，IQR 不重叠）；**实测 DRAM 消除 256.22 MiB = 预测 y[] 往返 8·n·rows 的 100.1%**；ndiff=0/ULP0 | `T8:g2-fuse-rms-norm-mul-rvv-vlen128-memory-axis`；`T3_A` fused-epilogue 行；cell `g2-fuse-rms-norm-mul/board_measured.md` | **board-proven（[NG-4] memory-axis A/B，kernel-micro pair）** | 是 **isolated A/B**（fused vs 我方自己 unfused 两趟），**非 ggml beat、非 e2e 八门**；whole-model e2e（65 norms/token≈2.03 MiB/token）是 projection（`pending`） |
| L3-2 | **[FMT-PROP] rms_norm→mul→quantize(q8_0) 融合** — 第二条 L3 字节腿（把下游激活量化 pass 折进 epilogue） | wall **1.139×**（T-N PASS，~30× 地板，IQR 0.082/0.032% 不重叠）；**实测 DRAM 消除 255.8 MiB = 预测 f32 激活往返 2·n·4·rows 的 99.9%**（双计数器：store-miss 127.97=z[]store 100.0% + load-miss 127.80=z[]reload 99.8%）；cycles Δ 30.01≈wall Δ 30.32 ms/iter（99.0%）；ndiff=0/4.46M（byte-exact fused≡unfused） | `T8:fmtprop-rms-norm-mul-quantize-q8_0-rvv-vlen128-memory-axis`；`T3_A` fused-quantize-epilogue 行；cell `fmtprop-rms-norm-mul-quantize/board_measured.md` | **board-proven（[NG-4] memory-axis A/B，kernel-micro pair）；铺量-phase1 board-proven COMPLETE** | 是 **isolated A/B**（fused vs 我方自己 unfused 两趟：fused-mul + 独立 `quantize_row_q8_0`），**非 ggml beat、非 e2e 八门**；**比 g2-fuse 1.308× 小且诚实**（FMT-PROP unfused 基线多做整趟 quantize → 同 256 MiB 消除占总流量比例更小；载重主张=消除**量级**复现非 wall 比值）；whole-model e2e 是 projection（`pending`） |

> L3 一句话：融合的机理（省掉激活往返：G2 是 y[]=8·n·rows，FMT-PROP 是 f32 激活 z[]=2·n·4·rows≈256MiB）
> **精确传导到硅**（两腿都 cycles Δ≈wall Δ、双计数器字节归因到 99–100%），是 board-proven 的内存轴数字；框架守住
> [NG-4]（内部 A/B、非 beat）。这是 L1/L2 的 compute/latency 轴之外，编译器变换在 **memory 轴** 上真赢 wall 的
> 病例族——两条腿（rms→mul、rms→mul→quantize）都复现了「消除一趟激活往返≈256MiB」的载重量级；铺量 phase1 board-proven 完成。

---

## 缺口闭环两案例（C3′ 方法学：LOSS → triage → 模式/修 → retest；C2 边际成本素材）

| 案例 | 类型 | 闭环 | 表行指针 | 成色 | 论文用法 |
|---|---|---|---|---|---|
| **[GAP-SB] iq2_xxs**（正例） | missing_pattern **闭合** | LOSS 0.735× → triage(per-subblock-gather-not-batched) → 模式 pair-batching → **retest WIN 2.570×** | `T8:format-micro-iq2_xxs-gapsb-pair-batching-flip` | **kernel-only** | 「发现缺模式→构造→复测翻盘」的**首个整环**范例 |
| **[GAP-P1]**（选择错误例） | misselection，**修被证伪** | LOSS 诊断 exists-but-not-selected → 修(widen-to-m1) **FALSIFIED 2.72× 退化** → 诊断精化（正确变体=mf2 wide-strip 非 m1） | `T8:q4_0-gevm-k1-vlen256-decode-P1-roofline-GAP` + `T8:q4_0-gemm-k1-vlen256-GAP-P1-fix-confirm-REGRESSION` | **board-proven（kernel-micro）** | 方法学的**诚实性**病例：提出的修可以是错的，ledger 如实登记并精化诊断 |
| **[GAP-RP] tq2_0**（部分闭合，补充例） | missing_pattern 部分闭合 | 4 spills→0；A/B ~1.92× + vs-generic 3.9× WIN；但 vs-SIMD-factory 仍 LOSS ~0.45× | `T8:format-micro-tq2_0-gaprp-spill-fix-rvv-vlen128` | **kernel-only** | spill 在关键路径→~2× 传导；对手仍是手调 SIMD → 缺口砍半非清零；sibling tq1_0 同类未修 |

---

## 家族机理**双向**验证（C3′ 迁移 + 边界）

| 方向 | 主张 | 表行指针 | 成色 |
|---|---|---|---|
| 机制**泛化**（正） | [GAP-SB] pair-batching 机制落在**全部 4 格**（objdump gather-quartering + iq3 AVL=2-storm 消除 + A/B 全 >1，gearbox byte-exact） | `T8:format-micro-iq2iq3-gapsb-pair-batching-family` | **kernel-only** |
| 赢**不泛化**（负边界） | **无家族 LOSS→WIN flip**：仅 iq2_xxs 越过 generic 线；iq3 对仍 compute-axis LOSS。**放大倍数跟着 bottleneck shape 走**（gather-shell-bound 3.52× / grid-decode-bound 仅 1.09–1.29×） | 同上；`T3_A` 的 de-risk(iq2_xxs) 与 roll-out(iq2_xs/iq2_s/iq3_xxs/iq3_s) **分列不合并** | **kernel-only** |
| **别一刀切**（per-format 键必要性） | whole-LMUL/m1 方向对 q4_0 GEMM（2.72× 退化）与 q8_0 GEVM（慢 11%）**双双回归** → 证明 per-format 键控是必要的、非一刀切 | `T8:...-GAP-P1-fix-confirm-REGRESSION` + `T8:q8_0-...-do-not-widen-VINDICATED` | **board-proven（kernel-micro）** |

> 家族一句话：机制**证明可迁移**（全 4 格 objdump+A/B），但**赢不外推**（唯一 flip 不放大为家族主张）——
> 放大倍数由 bottleneck shape 决定，是 C3′「换键不改条目」迁移判据 + [NG-4] 边界的一手叙事材料。

---

## 三贡献映射速查

- **C1（合取存在性）**：L1 的 5.9× 是成熟编译器栈内的实体证据之一；结构合取证据在 `T1_C1_structural_conjunction.csv`（本清单不复述）。
- **C2（边际成本规律）**：缺口闭环 / construction-queue 项（GAP-RP tq2_0 未清、GAP-NUM relaxed 未建、tq1_0 spill 未修、family roll-out）逐格喂 `T2_C2_ledger_marginal_cost.csv` 的边际成本点。
- **C3′（能力键控模式库）**：L1/L2/L3 + 两缺口案例 + 家族双向 = 主体。每模式以能力谓词表达、机制选出、带实测（board-proven 优先）与迁移判据（VLEN-flip / per-format 键）；vs 框架内核分相报告，beat 须过八门（当前 L1 仅 5/8）。

## 待闭 / pending 清单（诚实缺口）

- L1 八门 ②③④（VLEN256-flip-lit / k1-objdump-seal / micro↔e2e-Amdahl）— beat 措辞锁到闭合。
- L1-4/L1-5 q4_K/q5_K：opponent-absence 已 board-probed 双证，但**八门整体未走**（kernel-prefill 单核 proxy、吞吐-only 未复验数值）— `candidate`，beat 措辞锁；q4_K 吞吐=PARITY、q5_K=~1.5×-vs-UNTUNED（争夺格候选数非 win）。
- L1-6/L1-7/L1-8 q6_K/q2_K/q3_K：opponent-absence board-probed（q3_K 家族最强 absence）**且板上数值已证**（SILICON byte-exact-integer + bounded-norm，比 q4_K/q5_K 多一层硬件数值确认）— `candidate（board-numeric-verified）`；但吞吐**三格全 LOSS ~5–6×**（first-construction un-pipelined emit vs ggml mature `_vl128` block-dot）→ **[NG-4] moot（LOSS 不可能是 beat）**，八门未走；结构 opening 未转吞吐 = marginal-cost/maturity 证据，perf 轴 pending（construction-queue：pipeline/schedule 化 emit 才可能翻）。
- L3 G2 whole-model e2e（65 norms/token 投影）未跑板 — `pending`。
- L3-2 FMT-PROP whole-model e2e（激活量化 epilogue，逐 token 投影）未跑板 — `pending`（isolated A/B 已 board-proven，铺量 phase1 完成）。
- L2-5 iq4_xs/tq1_0 relaxed body 未建 — `structural-block`（双档税账仅 q8_0 可算）。
- 缺口余项：tq2_0 vs-SIMD-factory 仍 LOSS；tq1_0 同 spill 类未修 — construction-queue。
- 所有 kernel-only 项：e2e 传导多半 washes（compute/latency-bound），永远 kernel 与 e2e 分开报。
