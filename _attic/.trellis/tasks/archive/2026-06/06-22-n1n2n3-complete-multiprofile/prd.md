# N1/N2/N3 completion: Win-A in llama + clean ablation + 3 real profiles + N1 breadth

## Goal (user, 2026-06-22)
推进 spec 的 N1/N2/N3 **成立**。新解锁:**两台真实异构 RISC-V 机器**(+ 现有 rvv)= 3 个真实 profile。
五条线:
1. **N1 kernel breadth** — kernel 还偏少,多拓展(服务于「更多 profile 分化 / 实测胜出」,非堆 op 数)。
2. **N1 多-profile capability 分化(真硅片,③)** — 同 kernel 在 3 个真实 capability 向量上得到不同 legality/selection/dispatch。
3. **N2 第二 family(④)** — **修正(用户 2026-06-22 身份边界,见 `spec/architecture/design-boundaries.md` Family 准入边界 + memory [[n2-family-entry-boundary]])**:
   N2 第二 family 判据 = 能力可表达为 RISC-V capability 事实 + 零-core-branch 消费。**Fedora RVV0.7 / K1 VLEN256 都还是 RVV 家族 → 它们是 N1 *profile*,不是 N2 *family***。
   真正的 N2 第二 family 仍是 IME-class(挂 RV 核的非-RVV 扩展);无 IME 硬件 → N2 仍 hardware-blocked,**绝不拿独立离散卡(GPU/TPU/910B)顶替**(那会稀释 novelty)。用户的 offload 加速要拿到编程/寄存器模型后按判据重判(挂 RV 核经扩展暴露=正当;独立离散卡=拒)。
4. **N3 Win-A 进 llama(PRIZE)** — 把 max-legal-LMUL 接进 q4_0 block-dots,让 compiler-automatic tune 出现在真实 llama 推理。
5. **N3 Win-A microbench ablation 补完** — 现 2–4× vs naive 不是干净的「全-compiler、只变 LMUL 宽度」ablation
   (adversarial workflow 揭示:selector 是在 deferred-wide vs per-iter-reduce 两个**算法**间选,OFF 基线 sub-scalar;
   competent-naive 是手写)。要让 tcrv-opt **也 emit competent narrow-deferred**(同 deferred-accumulate 算法、窄 LMUL),
   使 LMUL-宽度 ablation 全部来自编译器。详见 archive/2026-06/06-22-n1n2n3-establish-rvv-mature/artifacts/winA-ablation-remeasure/WIN-A-FINDING.md。

## 硬件清单(2026-06-22 已连通 + 免密 + 免密 sudo)
| host | IP / 接入 | profile | 算力 | 编译器 | 用途 |
|---|---|---|---|---|---|
| `rvv`(现有) | 192.168.8.72 via rvv-jump | **VLEN=128, RVV1.0** | 64 核 | clang18 | 主基准 + Win-A/B 已封 |
| `k1` | 10.27.241.99 via rvv-jump | **VLEN=256, RVV1.0**, Spacemit X60 | **弱**:8 核/7GB | gcc13.2 | N1 VLEN 分化(128→256 翻转 strip 宽);仅 0.5–3B 模型 |
| `fedora-rvv07` | 192.168.102.104 via rvv-jump | **RVV0.7**(rv64gcv0p7), Fedora38 | **强**:128 核/250GB,可 -t64 | gcc13.2.1(march 支持 rv64gcv 与 rv64gcv0p7) | N1 ISA-代 分化 / N2 第二-family 候选 |
- 接入:都经 `rvv-jump`(211.87.236.75)ProxyJump;ssh config 已加 `fedora-rvv07`/`k1`;key 已 copy;NOPASSWD sudo 已设。
- 代理(HF 下载):`211.87.236.51:7890`;HF token:`/home/kingdom/ops/hf.md`。
- **关键 capability delta**:VLEN 128(rvv)vs 256(k1)→ e16m1 在 256 上是 16 lane(无需 two-halves split);RVV0.7(fedora)vs 1.0 → 指令编码/intrinsic 不同(RVV1.0 二进制在 0.7 硅上会 trap)。

## Gating 结果(2026-06-22)+ advisor 重构
- **K1 = 干净可用 RVV1.0 @ VLEN=256**(`RVV1.0 RAN ok VLEN=256`,gcc13)。→ 最便宜的 N1 真硅 win + 关键 cross-profile 轴。
- **Fedora = toolchain 阻塞**:`riscv_vector.h` 缺失(header/工具链问题,**不是硬件 trap**;trap-or-run 还测不了)。→ 先搁置,需装工具链。
- **PRIZE 结构确认 hardcoded**:repack/block-dot emitter 写死「VLEN≥128 / vsetvl(16) / two 8-lane halves」,**非 gearbox 选**。
- **advisor 重构(关键)**:
  - block 量化有 per-block scale → 每 32-elem block 必须先 reduce 再 scale-fold,**不能像 i16 contraction 那样跨 block defer**。
    那个 per-block reduce 就是墙。**repack(Win B,已 emit)就是 block-regime 的 deferred-wide 等价物**——16-blocks-as-lanes 化解了墙。
    ∴「把 max-LMUL 移植进 block-dots」会**重新撞同一堵墙**(1.8× 慢是结构性,非 tuning miss)。
  - **可辩护的「Win-A 进 llama」= 把 repack 的 strip 宽/LMUL 变成 resource-aware gearbox 决策**:VLEN=128→two 8-halves,
    VLEN=256→one 16-lane strip。→(a)compiler-automatic **selection** 进了 llama 真热路径;(b)K1 VLEN=256 是天然 ablation 轴。
  - **诚实 caveat(从一开始挂)**:**单芯片 block-dot 大 LMUL ablation 很可能不存在**(更宽=tiling 更多 output 列=above-layer 几何,
    被 ablation-table 排除为非-pass-row)。∴「Win-A 进 llama」最稳的形态 = **cross-profile strip 适配(128↔256)+ 现有 measured>static 1.0–1.6×**,
    **不是**新的单芯片 2–4× llama 数。别许诺结构给不了的东西。
  - **⑤ clean ablation 校准**:让 narrow-deferred 变 compiler-emitted 只是**确认 2–4×(全-compiler provenance),不产生新数**——
    2–4×(wide_vs_narrow,mf2→m2 同 deferred-accumulate 算法)本就是 LMUL-宽度贡献。Bank「clean ablation 确认 2–4×、现端到端全编译器」即走。

## 建议顺序(advisor 校准)
- 先 **⑤(Win-A clean ablation,本地+rvv)**:补 compiler narrow-deferred → 干净 LMUL-only ablation,把 N3 头条做实(不需新硬件,先把已质疑的补好)。
- 再 **④-PRIZE(Win-A 进 llama)**:max-legal-LMUL 接进 q4_0 block-dots,rvv 上 ON/OFF e2e ablation。
- 再 **②(N1 多-profile)**:K1(VLEN256)最快出 N1 分化(strip 宽翻转);Fedora RVV0.7 是更深的 ISA-代 分化/N2。
- 贯穿 **①(N1 breadth)**:沿分化需要拓 kernel,别堆数。

## 编译纪律(承接,用户多次强调)
- 先搞清状况再编;合理 `-j`(fedora 可 -t64;k1 弱→ -j4~8;rvv -j16);用 remote `timeout` 防 zombie;背靠背同-load 比值抗负载。
- 板子专用目录,绝不碰别人的树;structured emitc raw()=0;fail-closed verifier;real ssh 证据;诚实 ledger(区分 Win-A 编译器-tune vs Win-B kernel-swap)。

## CAMPAIGN OUTCOME (2026-06-23) — canonical record: `artifacts/N1N2N3-LEDGER.md`
- **N1**: capability→selection divergence **measured on 3/3 real profiles** (rvv m2@128, K1 m1@256, C920 m2@128/RVV0.7 — tracks VLEN, not ISA-gen). **Kernel coverage increased**: repack matrix path q4_0 → q4_0+q4_1 (`c7069111`). Open: winning-*algorithm*-by-profile still unproven; q4_1 GEMM + RVV0.7 block-dot emission deferred.
- **N2**: **PROVEN** (`2eeabff9`) — IME second family rides the common pipeline, **zero core-branch**, capability-fact dispatch (no string match), K1 bit-exact; **+breadth** 2nd op `mma_u` (`c7069111`). Foundation empirically validated on real X60 (`8bed5961`). Honest: one justified asm leaf (no IME intrinsic), IME perf unstarted.
- **N3**: methodology fixed (scalar dropped, durable contract → `spec/validation/experiment-reference.md`). **Win-A in llama (two e2e legs)**: VLEN-strip 1.31× decode (K1) + LMUL-width 1.70× prefill (rvv, engagement bug = harness toggle, fixed `2b1547b0`); + 2.27–3.79× contraction microbench. **Win-B** vs ggml's REAL RVV kernel: prefill ~6×, decode regime-dependent (`22c844a2`). Honest: K1 0.74× loss disclosed; Fedora coherent-llama e2e seal NOT achieved.
- **trellis**: trellis-check PASS on all invariants incl. I3/I5 (`bacebf2d`); methodology promoted to spec.

## PERF PHASE OUTCOME (2026-06-23, user-directed: kernel+e2e for every kernel) — canonical: `N1N2N3-LEDGER.md §7`
- **N2 IME perf** (`f8a42258`/`051ebff1`): kernel **5.51× vs RVV** (real K1, int8 256³, bit-exact); e2e **0.86–0.98× NULL** (lib-swap method; my env-toggle draft was a qemu-only no-op = IME-vs-IME, caught + corrected).
- **N1 q4_1 coverage+perf** (`f1753051`): q4_1 repack **GEMM added** (GEVM+GEMM pair complete); q4_1 GEVM **2.47× vs ggml's real RVV q4_1 kernel** (q4_0 control 2.38× → shared memory-locality mechanism). q4_1 e2e blocked (upstream q8_1x4 quantizer gap).
- **Fedora bug RESOLVED — not our compiler** (`6c885916`): BOTH RVV0.7 kernels (GEMM+GEVM) proven bit-exact in the EXACT llama strided/chunked/offset regimes; the "strided-store defect" theory refuted; e2e garbage is a ggml-side routing confound (decode GEVM 0 ENGAGED), NOT our kernel math.
- **CAPSTONE matrix** (`1b4f582a`, ledger §7) + central finding ([[kernel-wins-dont-transplant-to-e2e]]): compute-bound kernel wins (1.22–6.36×) do NOT transplant to memory-bound decode e2e; e2e wins live in memory-layout (Win-B prefill ~6×) / selection (VLEN-strip 1.31×). All nulls/losses/blocks disclosed.
- **trellis-check** PASS on q4_1 GEMM + IME matmul (I3/I5/I7/structural); 3 pre-existing unrelated masked-strided-dot failures noted (out of scope, last touched `1bf69314`).

## Acceptance (evolving) — status 2026-06-22
- [x] **⑤ tcrv-opt emit competent narrow-deferred → 干净全-compiler LMUL ablation** `[3d2a2b3f]`:budget 32→m8/12→m4/9→m1,
  同 deferred-accumulate 算法、只变 LMUL、两臂都编译器 emit;rvv 实测 wide÷narrow **2.27–3.79×**,两臂数值 EXACT vs scalar oracle `[709bb69d]`。
- [x] **④-PRIZE: Win-A 进 llama hot path** `[87c540ec]`:advisor 重构后落地为 **VLEN-aware repack strip**(非 max-LMUL 进 block-dot,那会撞 per-block-reduce 墙)。
  K1 microbench:compiler 自动选 1×16@VLEN256,数值 PASS,**1.48×** vs 2×8 `[08a11764]`。
  **2026-06-23 K1 重启后补完 e2e**:`llama-bench -t1` 真模型 tinyllama-q4_0,两 compiler-emitted 臂 back-to-back,
  **1×16 = 2.12 t/s vs 2×8 = 1.62 t/s = 1.31× e2e WIN**,两臂 ENGAGED marker 都 fire → strip-SELECTION 在真 llama decode、真 VLEN256 硅上胜出,印证 microbench。
  (与 repack-vs-generic 的 X60 loss 是不同问题,别混。)artifact `k1-vlen256/e2e-SEAL-and-caveat.md`。
- [x] **② N1 多-profile 真硅分化** `[4d36bdb3, d6022f58]`:q8_0 winner m2@128→m1@256(同-session,~7%,categorical LMUL-family reversal);
  q4_1 elision flip(K1 +11.2%);q4_0 factor flip(marginal);q5_0/q5_1 null(m1-only,结构性)。static argmin 在**每个 kernel×chip 都错**→ measured>static 普遍必需。
- [~] **④ / Fedora RVV0.7 — N1 ISA-generation 轴推进 (def25aba, a71aa201)**:
  Fedora 确认真 RVV0.7(SG2042/C920);**真硅证明可 target**:XuanTie 工具链(RuyiSDK gnu-plct-xthead 3.1.0)已装 Fedora,
  手写 `rv64gc_xtheadvector` kernel 出 `th.v*` 0.7.1 编码、**实跑绿**(dot PASS),RVV1.0 binary SIGILL —— 干净二分。
  **capability 模型已加 RVV0.7 版本轴**:`RVVVersion`(1p0/0p7)成 queryable fact,同 kernel 在 RVV0.7 vs RVV1.0 **legality 分化**
  (ta/ma agnostic policy 是 1.0-only → 0.7 上 fail-closed 拒;rv64gcv 输出 byte-identical)。这是 N1 最深轴(ISA 代),补 K1 的 VLEN 轴。
  **RVV0.7 emission 已落地**(`ec548291` GEMM one-col-per-pass + GEVM):3 kernel class 都出 fraction-free `th.v*` 0.7.1、**isolated bit-exact**(gemm_verify norm=0 / GEVM bit-exact / dot PASS)。
  **诚实修正(2026-06-23, 第3次 over-optimism 纠偏)**:**e2e coherent-llama seal = 未达成**。disciplined fresh run(`llama-completion`,
  quantizer objdump 确认全 scalar、emitted GEMM ENGAGED)仍输出 `####` 垃圾;之前 `6a864212` 的「scalar化 quantizer→coherent」**被反驳**;
  早先 v3 的 "Paris" 不可复现。残留 e2e 缺陷未隔离(候选:ggml 手写 q4_0 **weight repack** 在 xtheadvector 下误编、或 decode GEVM 路由——都非我们 compiler)。
  Fedora 可辩护主张 = isolated-kernel + capability-divergence 轴,**不是** coherent e2e。见 `fedora-rvv07/FINDING.md` CORRECTION。
- [~] **③ N2 第二 family — IME 现在 ACTIONABLE,foundation 验证中 (2026-06-23, `k1-ime-n2-feasibility.md` CORRECTION + `research/spacemit-ime.md`)**:
  K1 Spacemit X60 = **IME1**(Integer Matrix Extension,非-RVV 矩阵扩展挂 RV 核)= spec 点名的 N2 非-RVV family。**纠正早先「toolchain-blocked」**(只对 stock Bianbu 成立):
  march = `xsmtvdotii`(非 `rv64gcv_ime`)+ SpacemiT GCC15.2 fork + 公开 spec(`riscv-ime-extension-spec`/`docs-ai`)+ llama.cpp 已 merge 的 `spacemit/` 后端 oracle(PR#15288)——**四要素齐**。
  **当前推进(active front)**:N2/IME **foundation 经验验证**——本地装 SpacemiT 交叉工具链 → cross-compile 最小 `vmadot` → **真 K1 上跑、确认不 SIGILL 且 IME==scalar**(验证 A60≡X60 / xsmtvdotii≡Xsmti8i32mm 推断,满足 I8)。pass 后下一增量 = trellis-implement 做 `xsmtvdotii` capability-fact + `vmadot` 零-core-branch emit。**N2 = effort-bounded,不再 blocked**。
- [x] **① N1 breadth**:沿 VLEN 分化拓到 q8_0/q4_1/q4_0/q5 family-coverage map(见上)。
- [x] **全程**:structured raw()=0 / fail-closed verifier / real ssh 证据 / Win-A(编译器-tune)vs Win-B(kernel-swap)分清 / clean rebuild 绿 / 诚实 ledger(advisor 抓过 verifier 错机器、我抓 q4_1 overturn)。
- **Open 下一阶段(用户 steer)**:(2) full llama e2e on K1;(3) Fedora RVV0.7 codegen(XuanTie 工具链);(4) consolidate for paper;扩 VLEN strip 到更多 quant repack。
