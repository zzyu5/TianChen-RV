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

## Acceptance (evolving)
- [ ] ⑤ tcrv-opt emit competent narrow-deferred → 干净「全-compiler、只变 LMUL」ablation,rvv 实测 N(数值 PASS)。
- [ ] ④-PRIZE: max-legal-LMUL 进 q4_0 block-dots,rvv 上 ON/OFF e2e(llama)ablation 实测(诚实,哪怕 modest)。
- [ ] ② N1 多-profile:同 kernel 在 rvv(128)/k1(256)[/fedora RVV0.7] 上 legality/selection/dispatch 实测分化 + 真硅 run。
- [ ] ④ N2:RVV0.7 codegen 可行性(rv64gcv0p7)+ 能否 zero-core-branch 当第二 family。
- [ ] ① N1 breadth:沿分化/胜出拓 kernel。
- [ ] 全程 structured raw()=0 / fail-closed / real ssh 证据 / 诚实区分 Win-A vs Win-B / clean rebuild 绿。
