# G7 P0 — GEVM Plan 字节审计 (decode/GEVM regime · 设计输入)

> board `ssh rvv` openEuler VLEN128 64c · cores **8-15 disjoint-pin** · co-tenant vLLM/kernspan on 其它核 (未重启)
> NO git · board reversible · **live .so 测后 byte-exact restore = 05a62e6a stock-pristine (CONFIRMED)**
> 目的: 裁 [GAP-REPACK-GEVM] 布局假设 — **H1 (字节≈同而慢 = GEVM 结构不成熟)** vs **H2 (字节显著多 = 布局字节税)**
> 依据 canon: [K-10] 实证① (GEMM 兼职 GEVM = 结构级) · [PAT-2] P9 GEVM regime-plan · [GAP-REPACK-GEVM] 归因修正

## 0. 口径与方法 (validated)

- **kernel-axis DRAM 流量口径 = `cache-misses × 64B line`**，**校准合格**: 已知 4 GiB stream → cache-misses 67,143,848 ×64 = 4.0007 GiB (**+0.05%**). `LLC-load-misses` 在本板被 aliased 到 `cache-references`(膨胀 2.8×)→ 弃用。(raw/calib/)
- **per-token 归一 = 两点差分** `(cache_miss(N=64) − cache_miss(N=16)) / 48`，**抵消模型装载固定成本**。协议 `llama-bench -m M -p 0 -n N -r 1 --no-warmup -t 8`，taskset 8-15。N=64 双测差 <0.01% (稳)。
- **部署我方 kernel** = swap sealed ON .so (engage banner 逐格实证 "TCRV EMITTED GEVM ENGAGED")；correctness 沿用既有 byte-exact seal ([裁: P0 是审计不重验])。**stock** = live 出货 gcc-15 as-shipped (q4_K/q5_0/q5_1/q4_1 上游 riscv repack 缺席或 VLEN256-only-broken → 真路径 = generic block-dot)。
- **理论下限** = decode M=1 权重零复用 → 全 transformer+output 权重读一遍/token = 量化权重字节 (format-spec 值 ≈ 板实测 stock roofline；stock 每权重读~一次即触底 roofline)。
- **稳定性核**: `cache-misses` 是**进程私有计数器**，对**其它核负载鲁棒**(模型 5-6GB >> LLC，无论页缓存都从 DRAM 流) → **byte/token 是本审计的硬信号**；wall-time t/s 受负载(测中 loadavg 5-10) 压制，仅作旁证 (与 g5/systemacct sealed 比对)。

## 1. 逐格三值 (per-token · decode M=1 GEVM regime)

| 格·板 | 我方实测流量 | stock 同口径 | 理论下限 | **我方/stock** | 我方/理论 | decode t/s (实测/sealed) | 判读 |
|---|---|---|---|---|---|---|---|
| **q4_K@rvv** (anchor·0.50×) | **4.391 GiB** | 4.409 GiB | 4.41 GiB (spec 4.18) | **0.996** | 0.996 | 1.04 / 0.50× LOSS | **H1** |
| **q5_0@rvv** (0.82×) | **4.939 GiB** | 4.961 GiB | 4.96 GiB (spec 4.79) | **0.995** | 0.995 | 1.24 / 0.82× LOSS | **H1** |
| **q5_1@rvv** (0.78×) | **5.353 GiB** | 5.373 GiB | 5.37 GiB (spec 5.23) | **0.996** | 0.996 | 1.13 / 0.78× LOSS | **H1** |
| **q4_1@rvv** (对照·decode WIN) | **4.542 GiB** | 4.551 GiB | 4.55 GiB (spec 4.36) | **0.998** | 0.998 | 1.61 / 1.67× **WIN** | **H1** |
| q5_K@k1 (0.73×) | **PENDING** (k1 被 L3 占·延后) | — | ~5.4 GiB (spec 5.5b/w) | — | — | 0.73× LOSS | **H1-implied**¹ |

¹ q5_K@k1 直接字节审计延后 (k1 释放后)。但 [GAP-REPACK-GEVM] canon 反证已在案: **stock repack-GEVM@k1 decode 1.284× 胜原生 vec_dot** (同 interleaved-repack 布局·stock kernel·b12afd57) → k1 上布局非墙 → q5_K@k1 loss 结构性 (H1)，与 rvv 同。

## 2. 结构轴伴随证据 (为何字节同却快慢不同 = 结构不成熟的指纹)

| 格 | 我方 insn/tok | stock insn/tok | insn 我/stock | 我方 IPC | stock IPC | 我方 cyc/tok | stock cyc/tok | cyc 我/stock |
|---|---|---|---|---|---|---|---|---|
| q4_K | 5.79 G | 5.15 G | **1.12** | **0.29** | 0.52 | 19.87 G | 9.88 G | **2.01×** |
| q5_0 | 11.61 G | 7.18 G | **1.62** | 0.70 | 0.61 | 16.50 G | 11.86 G | 1.39× |
| q5_1 | 11.65 G | 8.14 G | **1.43** | 0.65 | 0.60 | 17.82 G | 13.61 G | 1.31× |
| q4_1 (WIN) | 5.90 G | 7.68 G | **0.77** | 0.48 | 0.39 | 12.26 G | 19.64 G | **0.62×** |

- **q4_K**: 同字节，2× cycles，**IPC 崩 0.52→0.29** = M=1 无算力可藏 memory 延迟、GEMM-tile 累加器/重建开销全 L1-resident (+17% L1-loads·off-DRAM) → **停顿型结构损**。
- **q5_0/q5_1**: 同字节，慢在 **+43~62% 重建指令** (qh 第5位组装 + repack unpack 在 M=1 不摊销) → **计算型结构损** (IPC 反而略高，但多干活)。
- **q4_1 (对照·WIN)**: 同字节，**快在更少指令 + 更高 IPC** (stock q4_1 block-dot 本身病态 IPC 0.39) → **赢也来自结构、非字节**。

## 3. 逐格 H1/H2 判读 + 聚合结论

- **逐格**: q4_K/q5_0/q5_1/q4_1 **全部 H1**。**H2 (布局字节税) 逐格 REJECTED** — 无一格 我方 >> stock 或 >> 理论×1.5；实测字节比全在 **0.995–0.998×** (我方甚至**略少** stock，全部 ≤ 1.0 ≈ roofline)。
- **聚合 (4/4 measured rvv → H1; q5_K@k1 H1-implied)**:
  1. **布局非墙 · 布局契约无需重议**: resident repack 布局字节已触底 roofline (permutation 非 duplication，footprint 不增)。[GAP-REPACK-GEVM] 的"布局之罪"假设 **证伪** (与 stock-repack-GEVM@k1 1.284× 反证同向)。
  2. **decode-loss 主因 = GEVM 结构不成熟 ([K-10] 实证① GEMM 兼职 GEVM 坐实)**: 同 DRAM 字节流下，win↔loss 的判别键**全在结构轴** (insn / IPC / cycles)，**从不在字节轴**。
  3. **★[PAT-2] P9 目标标签需精化**: registry 现写"权重零复用→**字节最少化**"。P0 实测: **字节已最少 (roofline)**，不是可动的杠杆。GEVM plan 的真实优化目标 = **把 roofline 带宽转成吞吐** = 在 M=1 抬 IPC / 削 per-element 重建指令 / 藏权重条带装载延迟 (即 [PAT-2] TRANSFORM 已写的 流式K归约×列组·累加器常驻·预取进结构)。**"字节最少化"应改述为"M=1 带宽→吞吐的结构效率"**。

## 4. GEVM plan 设计输入建议

- **plan 立项方向 = 攻结构 (非攻布局)**: H2 全格否决 → 不做双布局、不重议布局契约；沿用 resident repack 布局，新建独立 GEVM Emission Plan (独立迭代结构 [K-10] 结构级)，目标函数 = **M=1 结构效率** (IPC/latency-hiding/reconstruction 摊销)，非字节数。
- **P1 首格 = q4_K@rvv (审计浪费最大者)**: 同 4.4 GiB/token 字节下只兑现 stock 一半带宽 (0.50× · IPC 0.29 · +2× cycles) = roofline-利用率-gap 最大；且 = 超块最难结构 + sealed 正确性 + 现成部署。次序建议 q4_K → q5_0/q5_1 (重建指令摊销类·同 qh 家族) → 对照 q4_1 已是结构赢样板 (可作 plan"健康 decode"回归锚)。
- **成功判据信号 (设计期锚)**: plan 生效 = 同字节下 IPC/cycles 向 stock roofline 收敛 (q4_K 目标 IPC 0.29→≥0.52·cyc 2.01×→≤1.0×)，**字节保持 ≈ roofline 不回升** (若字节升 = 引入了 H2 布局税，需回退)。

## 5. 污染状态 + restore

- **disjoint-pin 净测**: 全程 taskset 8-15；co-tenant (vLLM qwen3 / kernspan monitor) 在其它核，未重启。测中 loadavg 5-10 → **仅压 wall-time t/s (旁证)，不动 cache-misses 进程私有字节计数 (硬信号)**。byte/token 两点差 N64 双测 <0.01% 稳。
- **restore = TRUE**: live .so md5 = **05a62e6a** (== sealed stock-pristine·systemacct cert-1)；残留 tcrv 符号 = 2 个 **q4_0 baseline** (shipped WinB·非 stray·非我方 q4_K/q5/q4_1)；无 q4_K/q5_0/q5_1/q4_1 emitted 残留；无遗留进程。**NO git · 无 source patch · 无 rebuild** (只 swap sealed ON .so)。
