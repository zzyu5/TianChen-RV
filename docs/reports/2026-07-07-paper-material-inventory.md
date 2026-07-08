# 论文素材现状清单 — 2026-07-07（裁决二 板批落表后；2026-07-08 G3 四.1 — G2/FMT-PROP whole-model e2e 落表更新；2026-07-08 G3 四(archival) — L3 从性能支柱重定位为机制展示/方法学素材 + FMT-PROP prefill 相传导账 owed 下板批；**2026-07-08/09 G3 K-quant 家族收口 — 五超块全构造退役 + S6 tiling 铺开 + [XFER-1] 4/4 + 传导账,见 §二.4**）

**目的**：把当前**已落表**的每条论文可用主张，映射到一个具体的证据表行指针，并标注证据成色。
纪律：[NG-4] 措辞守（internal A/B ≠ beat；beat 需过 [PERF-1] 八门、vs 同-ISA 框架内核、分相报告）；
scalar/naive 只作 internal sanity，**绝不**作贡献倍数。三贡献框架见 `.trellis/spec/index.md`
（C1 合取存在性 / C2 边际成本 / C3′ 能力键控模式库）。

**证据成色图例**
- `board-proven` — 真硅实测、T-N 显著、协议达标（preflight/freq-lock/median+IQR/cold>3×L3/restore）。
- `kernel-only` — 真硅实测但 KERNEL-micro；e2e 传导缺席（compute/latency-bound 多半 washes，见 memory: kernel-wins-dont-transplant）。
- `pending` — 机理投影 / 未跑板 / 八门未闭（②③④ 类）。
- `structural-block` — 阻塞是 emitter 成熟度（construction-queue），code-verifiable，非测量、未花板。
- `e2e-diluted-Amdahl` — 真硅 e2e 实测但相分裂 Δ **亚噪声（NOT-SIGNIFICANT）**；isolated kernel 赢经 Amdahl 稀释到噪声地板下（e2e 效应 ≈ 0），如实登记为诚实 null，**非 beat、非失败**（= kernel-wins-dont-transplant 的量化病例）。

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
| L1-9 | q4_K repack GEMM 输出**瓦片化**把结构 opening **转成 kernel-轴吞吐**（VLEN128；S1→S6 两步，board-measured 2026-07-08） | **kernel-轴 A/B 兑现**：S1 h-strip tile spill 84→23（−73%）/ A/B **+52.4%** / vs-opponent parity **0.962×→1.473× FLIP**（loss→~47% lead）；S6 min-fold+stack-panel spill→0（hot-loop live 23→0，maxVreg v30 ≤32）/ A/B **+27.9%** / parity **→1.884×**（~88% lead）；golden→S6 ~1.96× 复合，全程 byte-exact（vwmacc 2240 不变 + PRE/POST identity gate 0 mismatch） | `T8:q4_K-repack-gemm-tile-S1-hstrip-rvv-vlen128-kernel-axis`（d5a28efc）+ `T8:…-tile-S6-minfold-stackpanel-…`（5f194cbd）；设计尺子 `docs/reports/2026-07-08-G3-L1-tiling-schemes.md`（静态 `peak(mr,hs)` 账 + 6 方案 ≤32 证）；cells `experiments/active/l1-tile-s1-q4k-repack-gemm/` + `…-s6-…/`；构造 oracle `039133ea` | **board-proven（kernel-轴 A/B，单核 vs REAL-dispatched `_vl128` block-dot；八门未走）** | **[NG-4] 非 beat / [L-1] kernel-级、板-格式-bound**：八门未走、单核、opponent=单线程 block-dot proxy（kernel-axis proxy 非 threaded mul_mat）、correctness=构造 oracle + in-cell byte-exact identity gate；**非 e2e**（e2e **预估 +7%级 per Amdahl，待 K-quant 铺族后整模型验证**）；把 L1-4 的 q4_K 从吞吐-PARITY 提升为 kernel-轴 LEAD 的**成熟前沿兑现**（对应下方 line 134 张力闭合） |

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
> **★吞吐兑现格数 1→2→4（G3 裁决〇 board 2026-07-08 → G3 K-quant 家族收口 board 2026-07-08/09，见 §二.4）**：q4_K repack GEMM 输出**瓦片化**（S1→S6，L1-9）**在 q4_K 上闭合了**上面「结构 opening 未转吞吐」的成熟张力——S1 spill 84→23 / parity 0.962→1.473，S6 spill→0 / parity→**1.884×**（kernel-轴 A/B，byte-exact）。→ 兑现吞吐的格从 **1（q4_0 e2e 5.9×）→ 2（+q4_K）**；**家族铺开后再 →4**（+q2_K S6 **1.413×** FLIP LOSS→WIN、+q5_K S6 **2.193×**——两格 S6 tiling **HOLDS**）；q6_K/q3_K tiling **NULL**（weight-bound，~5.3–5.4× LOSS 未救）不加兑现。第 2–4 格均 **kernel-轴（非 e2e）**、八门未走、[NG-4] 非 beat；整模型 e2e 仅 **projection**（§二.4 传导账 prefill Amdahl 上限 ≈1.59×，decode NULL，measured Δ 集成 BLOCKED）。**family rollout = 完成**（K-quant 5/5 全构造退役）。

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

## L3 — 字节轴**机制展示 / 方法学素材**（内存流量变换；kernel-级机制已证、whole-model 被 Amdahl 稀释 → **不作性能支柱**；→ C3′ memory-axis 变换 + micro↛e2e 传导会计正面教材）

> **★定位（2026-07-08 G3 四.1 落表后调整）**：L3 融合从【性能支柱】重定位为【机制展示 + 方法学素材】。
> sealed 的 isolated 隔离数字（G2 1.308× / FMT-PROP 1.139× / 256 MiB DRAM 消除）**不动**、仍 board-proven；
> 但叙事**禁止暗示 whole-model e2e 收益**。**★锁定标准表述**：*「kernel 级字节轴机制已证,whole-model 中被
> Amdahl（norm 占 decode 0.05%）与 cache 驻留（8KiB 中间张量）稀释,e2e 不显著——传导会计精确预测了这一点」*。
> 用途 = C3′ memory-axis 变换机制的存在性证据 + **micro↛e2e 传导会计的档案级正面教材**（G2 e2e T8 行是「机制真、
> 但 Amdahl 切片 < 噪声地板」的诚实病例,**非失败、非缺陷**）——不是击败、不是 L1/L2 那样兑现 wall 的性能腿。

| # | 主张 | headline | 表行指针 | 成色 | [NG-4] 边界 |
|---|---|---|---|---|---|
| L3-1 | **G2 rms_norm→mul 融合首个 L3 字节实测** | wall **1.308×**（T-N PASS，~19× 地板，IQR 不重叠）；**实测 DRAM 消除 256.22 MiB = 预测 y[] 往返 8·n·rows 的 100.1%**；ndiff=0/ULP0 | `T8:g2-fuse-rms-norm-mul-rvv-vlen128-memory-axis`；`T3_A` fused-epilogue 行；cell `g2-fuse-rms-norm-mul/board_measured.md` | **board-proven（[NG-4] memory-axis A/B，kernel-micro pair）** | 是 **isolated A/B**（fused vs 我方自己 unfused 两趟），**非 ggml beat、非 e2e 八门**；whole-model e2e（G3 四.1，board-measured 2026-07-08）= **NOT-SIGNIFICANT**：tinyllama-q4_0 prefill −0.11%/decode −0.27% ns（+ llama-2-7b canonical-65-norm prefill +0.06%/decode +0.12%）均亚噪声、greedy 逐字节等价（value-preserving）→ isolated 1.308× **不传导**（成色 `e2e-diluted-Amdahl`：fused-norm 仅占 decode 0.05%、q4_0 matmul 90.2%、ceiling +0.05%<地板；norm 中间体 cache-resident 8KiB/norm decode ~1MiB prefill<L2/L3，isolated >L3(128MiB) micro 的 DRAM 往返在整模型尺度不存在）；见 `T8:g2-fuse-rms-norm-mul-WHOLE-MODEL-e2e-rvv-vlen128-NOT-SIGNIFICANT` + `T3_A` e2e 行 + cell `g2-e2e-wholemodel/`；无 e2e beat |
| L3-2 | **[FMT-PROP] rms_norm→mul→quantize(q8_0) 融合** — 第二条 L3 字节腿（把下游激活量化 pass 折进 epilogue） | wall **1.139×**（T-N PASS，~30× 地板，IQR 0.082/0.032% 不重叠）；**实测 DRAM 消除 255.8 MiB = 预测 f32 激活往返 2·n·4·rows 的 99.9%**（双计数器：store-miss 127.97=z[]store 100.0% + load-miss 127.80=z[]reload 99.8%）；cycles Δ 30.01≈wall Δ 30.32 ms/iter（99.0%）；ndiff=0/4.46M（byte-exact fused≡unfused） | `T8:fmtprop-rms-norm-mul-quantize-q8_0-rvv-vlen128-memory-axis`；`T3_A` fused-quantize-epilogue 行；cell `fmtprop-rms-norm-mul-quantize/board_measured.md` | **board-proven（[NG-4] memory-axis A/B，kernel-micro pair）；铺量-phase1 board-proven COMPLETE**（★prefill 相 Amdahl **分解** owed = 下板批；prefill e2e **结果**已测 NOT-SIGNIFICANT，见 pending 清单） | 是 **isolated A/B**（fused vs 我方自己 unfused 两趟：fused-mul + 独立 `quantize_row_q8_0`），**非 ggml beat、非 e2e 八门**；**比 g2-fuse 1.308× 小且诚实**（FMT-PROP unfused 基线多做整趟 quantize → 同 256 MiB 消除占总流量比例更小；载重主张=消除**量级**复现非 wall 比值）；whole-model e2e（G3 四.1）= **board-Amdahl-derived**（同一 decode profile：`quantize_row_q8_0`=0.14% of decode → 单独 e2e ceiling +0.14%、与 norm 合并 0.19%，**均亚噪声** → 成色 `e2e-diluted-Amdahl`，e2e 效应 ≈0 与 G2 同族）；见 `T8:g2-fuse-...-WHOLE-MODEL-e2e-...-NOT-SIGNIFICANT` 的 Amdahl 分解 + cell `g2-e2e-wholemodel/`；无 e2e beat |

> L3 一句话：融合的机理（省掉激活往返：G2 是 y[]=8·n·rows，FMT-PROP 是 f32 激活 z[]=2·n·4·rows≈256MiB）
> **精确传导到硅**（两腿都 cycles Δ≈wall Δ、双计数器字节归因到 99–100%），是 board-proven 的内存轴数字；框架守住
> [NG-4]（内部 A/B、非 beat）。这是 L1/L2 的 compute/latency 轴之外，编译器变换在 **memory 轴** 上于 **isolated
> kernel** 真赢 wall 的**机制展示**病例族——两条腿（rms→mul、rms→mul→quantize）都复现了「消除一趟激活往返≈256MiB」的
> 载重量级；铺量 phase1 board-proven 完成。**★但这是机制/方法学素材、不是性能支柱**：whole-model **不传导**（见下 ★），
> 叙事禁暗示 e2e 收益。**锁定标准表述**：*「kernel 级字节轴机制已证,whole-model 中被 Amdahl（norm 占 decode 0.05%）与
> cache 驻留（8KiB 中间张量）稀释,e2e 不显著——传导会计精确预测了这一点」*。
>
> **★whole-model e2e（G3 四.1，board-measured 2026-07-08，NOT-SIGNIFICANT）**：两腿的 isolated 内存赢**不传导**到整模型——
> tinyllama-q4_0 prefill −0.11%/decode −0.27% ns（+ llama-2-7b canonical-65-norm prefill +0.06%/decode +0.12%）**全亚噪声**、
> greedy 逐字节等价（value-preserving，on_sha=off_sha=c0c1274bd2aa8b01）。e2e A/B = ggml **自己**的 fusion ON/OFF 开关
> （`GGML_CPU_DISABLE_FUSION`；upstream f3e1828 **默认已融合** → 无 unfused-stock 可打；toggle 测的是我方编译器 emit 的**同一变换**、
> preflight(0) 同二进制对称）。**Amdahl 结构性解释**：fused-norm 仅占 decode task-clock **0.05%**、q4_0 matmul **90.2%**、
> norm-fusion ceiling **+0.05%**（+FMT-PROP quantize 0.14% → 合并 0.19%）**均在噪声地板下** → e2e null 是**结构性**非测量缺失；
> 病因=整模型 norm 中间体 **cache-resident**（8KiB/norm decode、~1MiB prefill，均<L2/L3），isolated >L3(128MiB) micro 强迫的
> DRAM 往返在整模型 norm 尺度**不存在**。成色 `e2e-diluted-Amdahl`；**无 e2e beat、无 Win-B/Win-C**，铺量 e2e 推广**不在证据上**
> （轴是 q4_0 matmul 非 norm）——这正是 memory `kernel-wins-dont-transplant-to-memory-bound-e2e` 被量化到 0.05% 切片的诚实病例。

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
- **★L1 成熟前沿 = 精确诊断出的开放张力（两曳光弹 board-证伪，2026-07-08）**：q4_K repack GEMM 上两个 cheap maturity 杠杆【全证伪于硅】——① byte-exact schedule 重排 = NULL（+0.65% 噪声地板、cell `l1-pipeline-q4k-repack-gemm`）；② 结构 RE-ROLL（回卷全展开超块→runtime `emitc.for`）= **−11% 退化**（oracle byte-exact GREEN 但 vsetvli 53→76/spill 84→118 反升、parity 0.962→0.860、cell `l1-reroll-q4k-repack-gemm`）。★病灶精确：emitter 全展开超块→~80 累加器>32 寄存器→普遍 spill（2188 次整寄存器访存=~20× off-peak）；但**寄存器压力 opening 在 16-累加器 `[col][half]` fan-out**——回卷 ii/k 不缩它（且 emitc.for 无 iter-args→累加器落内存每迭代 spill、SEW 交替 e8↔e16→vsetvli 非 loop-invariant 回卷反乘 toggle），唯一缩它的 column 维【牺牲 once-per-16-weight decode amortization=GEMM 相对 GEVM 全部优势】。→ **吞吐赢是 amortization-vs-register-pressure 真张力、非 cheap maturity 修**；转吞吐需更深 restructure（平衡 tiling），是精确定界的 **future-work 前沿**（诚实 negative = C2 边际成本 / C3′ 迁移边界素材，非失败）。 → **【已闭·G3 裁决〇 2026-07-08，L1-9】** 那个「更深 restructure（平衡 tiling）」已落地并 board-兑现（【预测→选型→板上】对账，"先诊断后动手" 系列第三例，与 `[GAP-SB]`/`[GAP-RP]` 并列）：设计尺子 `docs/reports/2026-07-08-G3-L1-tiling-schemes.md` 先立**静态 `peak(mr,hs)` 寄存器账**（`hs`=免费轴/`mr`=付费轴）+ 6 方案 ≤32 证 → 荐 **S1→S6 两步** → 板上兑现 **S1 h-strip tile**（peak live-set halves-not-sums、spill 84→23、A/B +52.4%、parity 0.962→1.473 **FLIP**）→ **S6 inline-min-fold+stack-panel**（≤32-vreg cliff、hot-loop spill→0、maxVreg v30、A/B +27.9%、parity→**1.884×**），全程 byte-exact（vwmacc 2240 不变 + identity gate 0 mismatch）。★附带一个**预测-纠正**点：build 期静态账（driver-export basis）曾把 S1 判为「spill≈FLAT 结构 NULL」，campaign-canonical fixture + 硅纠正之（两 h-strip pass 是**顺序**的 → peak **halves 非 sums**）。future-work 前沿由此**从 open 张力升级为 board-proven kernel-轴兑现**（`T8:…-tile-S1/S6…`；[NG-4] 非 beat、[L-1] kernel-级、**非 e2e**、八门未走；e2e 仅预估 +7%级待铺族后整模型验证）。
- ~~L3 G2 whole-model e2e 未跑板 — `pending`~~ → **已跑板已闭（G3 四.1，2026-07-08）**：**NOT-SIGNIFICANT**（tinyllama-q4_0 prefill −0.11%/decode −0.27% ns 均亚噪声；llama-2-7b canonical-65-norm 同 null；value-preserving 逐字节等价）；成色 `e2e-diluted-Amdahl`（Amdahl ceiling +0.05%<地板、norm cache-resident、isolated 1.308× 不传导）；行 `T8:g2-fuse-rms-norm-mul-WHOLE-MODEL-e2e-rvv-vlen128-NOT-SIGNIFICANT` + `T3_A` e2e 行 + cell `g2-e2e-wholemodel/`。isolated 内存轴赢仍 board-proven，e2e 推广不在证据上（轴=matmul 非 norm）。
- ~~L3-2 FMT-PROP whole-model e2e 未跑板 — `pending`~~ → **已闭（G3 四.1，board-Amdahl-derived）**：同一 decode profile 的 `quantize_row_q8_0`=0.14% of decode → 单独 e2e ceiling +0.14%、与 norm 合并 0.19%，均亚噪声；成色 `e2e-diluted-Amdahl`（e2e 效应 ≈0，与 G2 同族）；见上同 cell/T8 行的 Amdahl 分解。isolated A/B 仍 board-proven、铺量 phase1 完成。
- **[FMT-PROP] prefill 相传导账 — `owed`（下板批）**：现有 Amdahl 分解**只有 decode task-clock profile**（`experiments/active/g2-e2e-wholemodel/amdahl_perf_tinyllama.txt`：`quantize_row_q8_0`=0.14% of **decode**）。**激活量化在 prefill 相的占比 ≠ decode 0.14%**——prefill（pp128, M=128 GEMM）权重复用度高（weights streamed once × M=128 列）→ matmul 更 compute-bound、其 task-clock 占比只会**高于** decode 的 90.2%,而 `quantize_row_q8_0` 工作量 ∝ 激活量、matmul ∝ 激活×N → prefill 的 quantize 相内占比**期望 ≤ decode 的 0.14%**,故合并 norm+quantize 的 prefill Amdahl ceiling **≤ ~0.19%（保守上界,projection 非实测分解）**,与**已实测**的 prefill e2e Δ（tinyllama −0.11% / llama-7b +0.06%,均亚噪声）自洽。**owed = prefill 相 task-clock profile**（传导会计四列的「相内时间占比%」列）以把 projection 升级为实测分解；board-derivable 非 code-blocked → **标下板批**。（注：prefill e2e **结果**已测=NOT-SIGNIFICANT,欠的只是相内**分解**列。）
- L2-5 iq4_xs/tq1_0 relaxed body 未建 — `structural-block`（双档税账仅 q8_0 可算）。
- 缺口余项：tq2_0 vs-SIMD-factory 仍 LOSS；tq1_0 同 spill 类未修 — construction-queue。
- 所有 kernel-only 项：e2e 传导多半 washes（compute/latency-bound），永远 kernel 与 e2e 分开报。

---

## 二.4 — G3 K-quant 家族收口包（board 2026-07-08/09，rvv/VLEN128）

> 五超块 K-quant repack GEMM 家族全构造退役 + S6 输出-tiling 铺开 + [XFER-1] 迁移验证 + 整模型传导账。
> **纪律**：kernel-轴数是**兑现-判定指针**（[NG-4] 非 beat / [L-1] kernel-级、板-格式-bound）；整模型 e2e = **projection**（集成 BLOCKED）。

### 收口三数（schema-authoritative @ working tree）

| 指标 | G3 起→终 | 现值 | 权威源 |
|---|---|---|---|
| **燃减 C_construct(强义)** | 33 → **40** | **40/93 = 43.0%**（过 M2≥40% 门，向 ≥70%→90% 推） | `coverage_metrics.py report` / `schema/coverage-sixstate.v1.json` |
| **吞吐兑现(格数)** | 1 → **4** | q4_0 e2e 5.9× + q4_K/q2_K/q5_K S6 kernel-轴 HOLDS | `T3_A` §二.1 块 + `l1-tile-s6-q4k`/`l1-t3-q{2,5}k` cells |
| **旁路存量(直连发射器)** | 17 → **10** | ternary 2 + K-quant 5 全退役；残 10（flat4 + iq2×3 + iq4×2 + mxfp4） | `schema/emit-bypass-whitelist.v1.json` `baseline_count=10` |

**T7 三曲线**（`experiments/active/visibility/T7-three-curve-G3-closure.md`）：三序在 G3 上**机械耦合**——每 front-door 构造 = **+1 C_construct ∧ −1 旁路**（7 次一一对应，7a4250c5/0b907d0c/7ff52fc4/c3cf7301/1b367c1f/54d3741c/df8a0b76），其中 3 个 K-quant S6 HOLDS 额外 **+1 兑现**。（`T7-burndown.md` 生成器渲染到 C=39，差 1 = 末构造 `df8a0b76` subject 漏 `C_construct 39→40` token;非未提交,schema 权威=40。）

### 二.1 K-quant 五格汇总（S6 register-cliff transfer；瓶颈形状分类 → tiling 判定）

| fmt | 六态 | 瓶颈形状分类 | tiling 判定 | 对手位（VLEN128 dispatch） | spill 前→后（peak vreg） | vs-opponent(untiled→tiled) | [XFER-1] 预测→实测 |
|---|---|---|---|---|---|---|---|
| **q4_K** | constructed-strong | **min-fold register-cliff** | **HOLDS**（anchor） | case128=TODO no-op→nullptr@128;`_vl128` 手调 | 84→**3**（v31→**v30**） | 0.962→**1.884×**（~88% lead） | anchor/origin（HOLDS） |
| **q2_K** | constructed-strong | **min-fold register-cliff**（共享 dmin/bsums-min fold） | **HOLDS** | case128=TODO→nullptr@128;`_vl128` 成熟 | 619→**7**（v31→**v30**;text 减半） | 0.212→**1.413×**（FLIP LOSS→WIN） | 预测 HOLDS → **HOLDS ✓** |
| **q5_K** | constructed-strong | **min-fold-cliff + qh 残留**（HYBRID） | **HOLDS**（hybrid） | 全 VLEN 无 riscv repack→nullptr;UNTUNED generic | 155→**105**（−32%;v31→**v30**;reload −62%） | 1.547→**2.193×**（already-WIN 推进） | 预测 HOLDS → **HOLDS ✓**（crux：min-fold 存活 qh 面） |
| **q6_K** | constructed-strong | **dual-plane weight-bound**（ql+qh） | **NULL** | 无 riscv 分支 NEON-only→nullptr@每 VLEN;成熟 | 913→**949**（rose;v31→**v31**） | 0.184→0.181（~5.4× LOSS 未救） | 预测 NULL → **NULL ✓** |
| **q3_K** | constructed-strong | **dual-plane weight-bound**（qs+hmask,no-min） | **NULL** | 不在 selector,ggml 零 q3_K repack（**最强 absence**）→nullptr;成熟 `_vl128` | 978→**894**（−8.6% 仅 overhead-shave;v31→**v31**） | 0.176→0.191（~5.3× LOSS 未救） | 预测 NULL → **NULL ✓** |

- **判别式**：≤32-vreg 寄存器悬崖（**v30=HOLDS / v31=NULL**）严格追踪瓶颈形状——S6 stack-panel lever 键控于 **min-fold**（把 idle i32 MIN 累加器 + decode strips 外置到栈,byte-exact）：HOLDS ⟺ 峰值压力=decode/min strips（q4_K/q2_K/q5_K 共享 `kquant_dmin_bsums_min` fold）;NULL ⟺ 峰值压力=dual-plane 权重重构（q6_K/q3_K 共享 `kquant_single_scale_no_min` fold,panel 不触及）。全程 **byte-exact**（IDENTITY cmp 0 + silicon INT_mismatch=0 + vwmacc multiset 不变）。
- **[XFER-1] register-cliff 迁移 = 4/4 命中硅**：从 q4_K S6 anchor 出发的 **4 个 sibling 预测**（q2_K/q5_K 预测 HOLDS、q6_K/q3_K 预测 NULL）**全部实测命中**。这是 C3′「换键不改条目」迁移判据在最细分辨率上的一手素材：register-cliff lever 键控于 min fold,一个共存的权重面（q5_K qh）**降级但不击败**它（hybrid HOLDS,spill 未塌到 0）。

### e2e 结论（不显著 → projection + 归因 + 八门状态）

- **measured 整模型 Δ = N/A**（集成 BLOCKED）：构造 q4_K repack GEMM 活在编译器 emitter（`emitRepackKQuantGemmBodyQ4K`,working-tree 未 commit）,未接入 ggml Q4_K `mul_mat` dispatch;且 VLEN128 K-quant repack trait=`nullptr`(**无可翻 env-toggle**,不同于 G2 fusion ON/OFF)。热插需离线权重 repack + 新 dispatch + q8_K 激活量化胶水(历史 e2e-seal 难点)。
- **传导账（档案 projection,`kquant-family-closure/transmission_account.md`）**：实测 prefill matmul 占比 **97.3%**（q4_K vec_dot 79.08% + q6_K 18.18%,perf task-clock 798K samples pin8-15 -t8）→ Amdahl（q4_K@1.884× + q6_K@1.0×）**S_prefill ≈ 1.59×**（区间 1.3×–1.6×,机理天花板 1.84× 纯-q4_K）;**decode 传导效率 ≈ 0**（M=1 GEVM memory-bound,tiling 增益机理不存在,Δ≈1.00× NULL）。整模型 = decode-dominated → 净生成 tok/s ≈ NULL,**价值严格住 prefill 相**。stock 基线 pp128 **5.184 t/s**（cv 0.02%）/ tg32 **2.087 t/s**（cv 0.13%）;若集成满额 projected prefill ≈ 8.24 t/s（**projection 非 measured**）。→ 入 `T3_A` §二.2 e2e-transmission-projection 行（成色 `pending`/`e2e-diluted-by-blocker`）。

### q4_K 八门（§4.4 权威；`kquant-family-closure/T-PERF1_q4_K_vlen128_prefill_8gate.md`）

**3/8 PASS · 2 partial · 3 missing → 非 sealed Win,beat 措辞 LOCKED（[NG-4]）**：**PASS** ①字节精确（bounded-ULP 非 ULP0）· ⑥实验纪律 · ⑧措辞门;**PARTIAL** ③双板 objdump（rvv128 ✓ / k1-256 ✗）· ④micro∧e2e（micro 1.884× ✓ / e2e projection,measured BLOCKED）;**MISSING** ②VLEN256 flip lit · ⑤双板都验证（k1 未测 q4_K）· ⑦selector 归因（op-identity 选中,reason≈static_order,非 capability-keyed）。下板批：q4_K VLEN256 lit + k1 objdump/复测 + e2e 集成或 measured A/B + tiled 变体能力键归因。

### 落地指针（本收口无 lib/ 改动;deliverable 未 commit,留用户提交）

- 传导账 + 八门 + phase-split：`experiments/active/kquant-family-closure/`（9 文件）+ harness `tools/e2e-harness/board/kquant_transmission_amdahl.sh`。
- 五格汇总 + e2e projection 入 `T3_A`（§二.1 / §二.2 块,28-列 schema）;五格 primary 证据 = `l1-tile-s6-q4k` + `l1-t3-q{5,2,6,3}k` cells。
- 三曲线：`experiments/active/visibility/T7-three-curve-G3-closure.md`（registered;INDEX regen;dir-lint GREEN）。
