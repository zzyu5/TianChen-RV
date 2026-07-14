# G7 P0 补格 — q5_K@k1 decode(M=1 GEVM) 审计 (第 5 格 · zero-gap 收口)

> board `ssh k1` SpacemiT X60 / **VLEN256** (vlenb=32) / 8-core / Bianbu **clang-18** / gov=performance **1.6GHz** (DVFS locked)
> cores **0-3 disjoint-pin** (承 g5-M2 k1 methodology) · co-tenant = bluetoothd(~92% 单核)/avahi 在其它核 · loadavg 测前 2.5 测后 7.6
> NO git · board reversible · **swap-only 无 rebuild** · 测后 byte-exact restore = 14b6add6 OFF-pristine (CONFIRMED·md5+nm 双证)
> 目的: 补完 P0 第 5 格 (q5_K@k1·decode 0.729× LOSS·VLEN256) — 承 `../evidence.md` 方法, 裁 H1(结构不成熟) vs H2(布局字节税)

## 0. 口径 — ★DRAM 字节口径在 k1 退化 (X60 PMU 硬件限制, 非权限)

P0 rvv 口径 = `cache-misses × 64B` (openEuler PMU 校准 +0.05%). **k1/X60 PMU 无此事件**:
自建 perf_event_open 探针 (raw/pmu_probe.c·result.txt) 实测——

| 事件 | X60 结果 | 可用 |
|---|---|---|
| `HW_CACHE_MISSES` | val=0 (256MB stream 后仍 0) | ✗ 未映射 |
| `HW_CACHE_REFERENCES` | val=0 | ✗ |
| `PERF_TYPE_HW_CACHE LL_READ_MISS` | val=0 | ✗ |
| `L1D_READ_MISS` | val=78 (stub·应百万级) | ✗ |
| **`HW_INSTRUCTIONS`** | val=2.72e8 (合理) | **✓ minstret 固定计数器** |
| **`HW_CPU_CYCLES`** | val=6.25e8 (合理) | **✓ mcycle 固定计数器** |

ISA 含 `sscofpmf`+`zihpm` (有 HPM 计数器) 但 SpacemiT 未把 cache-miss 映射进 generic perf event
(`/sys/bus/event_source/devices/cpu/events/` 空·无 DDR/uncore PMU). paranoid=2·**非权限问题, 是事件不存在**.
→ **字节流量 ①② 无法直接测**; 退化路径 = 【解析布局 footprint 恒等 (板无关) + 实测结构轴 (instructions/IPC/cycles·工作计数器)】.
> 此退化【更强】而非更弱: H2 的核心问题"我方布局是否比 stock 读更多字节"是**布局 footprint 属性**, 可解析判定;
> 而 P0 rvv 已【实测】同一 `block_q5_Kx16` 结构 = 0.995–0.998× (permutation·非 duplication) → 板无关地转移为佐证.

- **部署** = swap sealed ON .so (a408563e·engage banner "TCRV G5-M2 EMITTED GEVM ENGAGED" 逐 ON run 4 hits 实证) vs OFF (14b6add6·stock block-dot·**clang-18 对称**·同树同编译器·唯一差 = q5_K dispatch). 承 g5-M2 6cbd9c19 seal.
- **部署核** = t4a export **gemm ba30ba54 / gevm c445b89e** (= k1 M2 部署核·vwmacc gemm=2240/gevm=560 匹配). correctness = GREEN@bounded-ULP (引 g5-M2·**P0 是审计不重验**).
- **两点差分 per-token** = `(cnt(N=64) − cnt(N=16)) / 48` (抵消模型装载/graph 固定成本)·协议 `llama-bench -m M -p 0 -n N -t 4 -r 1 --no-warmup`·taskset 0-3·3 reps 取 median.
- **结构轴稳定性**: instructions = 进程私有 retired 计数, **对 co-tenant 负载完全鲁棒** (硬信号·3 reps 内 <3%); cycles/IPC 对跨核竞争敏感 (旁证·与 g5 sealed t/s 交叉验).

## 1. 三值 (per-token · decode M=1 GEVM regime)

| 量 | 值 | 来源 |
|---|---|---|
| ③ **理论下限** | **702.14 MiB/token (0.6857 GiB)** | 解析 gguf: 总 tensor 745.11 MiB − token_embd 42.97 MiB(仅 gather 不流式)·model=q5_K 599.2MiB(80%)+q6_K 145.6MiB+f32 0.4MiB·parser 对齐 llama.cpp 报 745.11 MiB✓ |
| ① **我方(ON)流量** | **= 702 MiB (= floor·布局恒等推定)** | X60 无字节计数器→不可直测; block_q5_Kx16 = permutation→每字节读一遍→= 理论 floor |
| ② **stock(OFF)流量** | **= 702 MiB (= floor)** | 同上·布局恒等 |
| **我方/stock (H2 键)** | **1.000 (解析 footprint 恒等)** | 见 §2·permutation 非 duplication |
| 我方/理论 | 1.00 (= roofline) | permutation→触底 |

**跨板佐证 (rvv P0 实测同结构)**: q5_0/q5_1/q4_K/q4_1 @rvv 我方/stock = **0.995–0.998×** (≤roofline·permutation). `block_q5_Kx16` 在 k1 与 rvv 是**同一数据结构** → rvv 字节实测板无关转移.

## 2. ★H2 (布局字节税) — 解析 footprint 恒等, 板无关 REJECTED

`block_q5_K` (stock·QK_K=256): d(2)+dmin(2)+scales(12)+qh(32)+qs(128) = **176 B/超块** (repo `static_assert(sizeof(block_q5_K)==176)`).
`block_q5_Kx16` (我方 repack·g5 seal·stride 2816): d[16]@0 + dmin[16]@32 + scales[192]@64 + qh[512]@256 + qs[2048]@768
= 32+32+192+512+2048 = **2816 B / 16 超块 = 176 B/超块**.

→ **2816 = 16 × 176 · 逐区 permutation (interleave)·零 duplication · footprint 逐字节恒等 = 5.5 bit/weight 不变.**
GEVM(nr=1) 每权重参与唯一输出行, 每字节读一遍·两侧均顺序流经权重张量一次·cache-line 全用 → **我方与 stock 读【相同】DRAM 字节 = 理论 floor. H2 逐格 REJECTED (板无关·比 rvv 实测更硬).**

> ★修正 g5-M2 evidence.md L101 措辞: 该处假设"block_q5_Kx16 interleaved (stride 2816, +qh plane) costs **more bandwidth** than
> plain block_q5_K for nr=1 → roofline loss" = [GAP-REPACK-GEVM] "布局之罪" 假设. **本审计证伪**: footprint 恒等,
> decode loss【不是】带宽/字节税, 是结构指令税 (§3). g5 decode-loss 归因由"布局带宽"改判为"GEVM 结构不成熟".

## 3. 结构轴 (k1 实测·两点·3 reps median) — H1 指纹

| 变体 | instr/tok | cyc/tok | IPC | decode t/s |
|---|---|---|---|---|
| **OFF** (stock block-dot·clang-18) | **0.704 G** | 2.624 G | 0.268 | ~2.44 |
| **ON** (我方 repack GEVM·clang-18) | **1.400 G** | 3.599 G | 0.389 | ~1.78 |
| **ON/OFF** | **1.99×** | **1.372×** | 1.45× (更高) | **0.729×** |

- **交叉验证 (内部一致)**: cyc/tok 比 1.372× = 1/0.729 = decode t/s 比·**独立复现 g5-M2 sealed decode 0.729× LOSS** (本审计 median t/s 1.78/2.44 = 0.730×). 结构计数与 sealed 吞吐自洽.
- **判别键**: 同 DRAM footprint (§2 恒等) 下, 我方 decode 每 token 执行 **~2× 指令** = qh 第5位平面重建 + 超块 scale 6-bit 解包 + interleaved-repack gather, 在 **M=1 (nr=1·权重零复用)** 无从摊销 → +99% 指令 → 1.37× cycles → 0.73× 吞吐.
- **子型 = 计算型结构损** (P0 q5_0/q5_1 同型: "同字节·+43~62% 重建指令·IPC 反而略高但多干活"). q5_K@k1 更极端 (+99%), 与 q5_K 超块重建成本 (三层: sub-scale/qh/qs) 一致. IPC 我方 0.389 > stock 0.268 (核质量不差·但活多一倍→净慢).

## 4. H1/H2 判读 + 跨板对照

- **本格 = H1 (结构不成熟·计算型)**. **H2 (布局字节税) REJECTED**: 我方/stock 字节 = 1.000 (footprint 恒等·非 >>stock·非 >>理论×1.5).
- **★P0 5/5 格全 H1** (q4_K/q5_0/q5_1/q4_1 @rvv 实测 + **q5_K@k1 本格**): 从 "H1-implied(k1 stock-repack-GEVM 1.284× 反证)" **升格 H1-measured (结构轴直证 +99% 指令)**.
- **跨板对照 (k1/VLEN256 vs rvv/VLEN128)**:
  - q5_K@k1 **prefill WIN 1.641×** (compute-bound·GEMM 摊销重建) vs **decode LOSS 0.729×** (memory-bound GEVM·M=1 重建不摊销) = **同一 VLA 核·GEMM 兼职 GEVM 的结构损** ([K-10] 实证①), 与 rvv 回退格【同机制】.
  - 字节轴两板均触底 roofline (permutation·板无关); 缺口两板均在**结构轴** (instructions), **从不在字节轴**.
  - → **GEVM plan 在 k1/VLEN256 同样适用** (攻结构·非攻布局): decode 慢 = M=1 重建指令税, VLEN256 更宽亦不救 (nr=1 无并行行可摊销). **[PAT-2] P9 目标"字节最少化"→"M=1 带宽→吞吐结构效率"精化在 k1 亦成立** = P4 跨板铺开输入.

## 5. 污染状态 + restore (双证)

- **restore = TRUE**: LIVE (`/data/build-k1-q5k/bin/libggml-cpu.so.0.15.1`) 测后 md5 = **14b6add6** (== g5-M2 sealed OFF-pristine); RUNPATH 副本 `/home/bianbu/tcrv-k1-llama/build/bin/` 亦 = 14b6add6. **nm 双证**: live q5_K tcrv 符号 = **0** (我方核下线·ON 时为 2). (残留 3 个 ENGAGED 字符串 = tcrv 树内 q4_K/q2_K 等【他格】banner 常量·非 q5_K·ON 时为 5·差 2 = q5_K gemm+gevm). 无遗留进程.
- **swap-only**: 仅 `cp` sealed ON/OFF 进 LIVE (LD_LIBRARY_PATH=build-k1-q5k/bin 优先)·**无 source patch·无 rebuild·NO git**. sealed 库来自 `/tmp/g5_q5k/libggml-cpu.so.{ON,OFF}` (g5-M2 build-seal 产物).
- **disjoint-pin 净测**: 全程 taskset 0-3·co-tenant 在其它核; instructions = 进程私有 (co-tenant 无关·3 reps <3%) = 本审计硬信号; cycles/IPC 旁证 (与 g5 sealed t/s 0.729× 交叉验一致).
