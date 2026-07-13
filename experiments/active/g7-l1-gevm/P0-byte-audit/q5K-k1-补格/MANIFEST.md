# MANIFEST — g7-l1-gevm / P0-byte-audit / q5K-k1-补格

- campaign: G7 主战役 (L1 GEVM plan 构造) · **P0 字节审计第 5 格补完** (q5_K@k1·VLEN256).
- role: 补 P0 唯一 deferred 格 (k1 曾被 L3 占) → P0 达 **5/5 格·zero-gap** 设计输入 + GEVM plan 跨板 (k1/VLEN256) 适用性输入.
- board: `ssh k1` SpacemiT X60 VLEN256 8-core · clang-18 · gov=performance 1.6GHz · cores 0-3 disjoint-pin · NO git · swap-only 无 rebuild.
- account: decode M=1 GEVM regime · ours(部署 sealed emitted repack-GEVM a408563e) vs OFF(clang-18 对称 block-dot 14b6add6) vs 理论下限.

## ★口径 (关键差异 vs P0 rvv)
- **DRAM 字节口径在 k1 退化**: X60 PMU 无 cache-miss 事件 (自建 perf_event_open 探针实证 HW_CACHE_MISSES/REFERENCES/LL 全 =0·非权限·raw/pmu_probe_result.txt). 仅 minstret/mcycle 固定计数器可用.
- 退化路径 = **① 解析 footprint 恒等 (板无关·H2 判定)** + **② 实测结构轴 instructions/cycles/IPC 两点差分 (H1 判定)**. 两者合力给出 H1/H2 裁决 (较 P0 rvv 字节实测口径不同但结论同向).
- 两点差分 per-token = (cnt(N=64)−cnt(N=16))/48 · `-p 0 -n N -t 4 -r 1 --no-warmup` · 3 reps median.

## canon 依据 (读死)
- [K-10] 结构级/参数级判据 · 实证① GEMM 兼职 GEVM = 结构级.
- [PAT-2] P9 GEVM regime-plan · [GAP-REPACK-GEVM] 归因修正 (布局降为待审计假设·本格解析证伪其"布局之罪").
- 起点: `../evidence.md` (P0 4 格 H1 方法+结论) · `experiments/active/g5-wiring/M2-q5_K-k1-e2e/evidence.md` (sealed 核+部署+0.729× decode) · `experiments/active/kquant-k1-vlen256-kernel-axis-t4a` (t4a export 核 ba30ba54/c445b89e).

## provenance (sealed ON/OFF · 板上 /tmp/g5_q5k · swap-only)
- ON  = /tmp/g5_q5k/libggml-cpu.so.ON  (a408563e·我方 emitted VLA repack GEMM+GEVM·nm q5_K tcrv=2·banner) — 部署核 gemm ba30ba54 / gevm c445b89e.
- OFF = /tmp/g5_q5k/libggml-cpu.so.OFF (14b6add6·clang-18 对称 block-dot·nm q5_K tcrv=0) = restore target.
- LIVE = /data/build-k1-q5k/bin/libggml-cpu.so.0.15.1 (LD_LIBRARY_PATH 优先) · BENCH = /data/k1build/bin/llama-bench (真 ELF).
- model = /data/tinyllama-1.1b-Q5_K_M.gguf (745.11 MiB tensor·q5_K 599MiB+q6_K 146MiB·1.10B).

## durable files
- evidence.md — 三值 + H2 解析恒等 REJECTED + 结构轴 H1 (+99% 指令) + 跨板对照 + restore 双证.
- raw/g7q5k_main.log — 全量 12 DATA 行 (OFF/ON × N16/N64 × 3 reps·instr/cycles/ts/engage/freq) + restore md5.
- raw/run_{OFF,ON}_N{16,64}_r{1,2,3}.txt — 逐 run 原始 (structcnt 计数 + llama-bench json + engage banner).
- raw/pmu_probe_result.txt + pmu_probe.c — X60 PMU cache-miss 不可用实证 (口径退化根据).
- raw/structcnt.c — 结构轴计数 harness (perf_event_open inherit·instructions+cycles).
- raw/g7q5k_measure.sh — 测量脚本 (interleaved swap·两点·restore). raw/gguf_bytes.py — 理论下限解析.

## 结论 (一句)
**q5_K@k1 decode = H1 (结构不成熟·计算型)·H2 布局字节税解析 REJECTED** (block_q5_Kx16 2816B = 16×block_q5_K 176B·permutation 非 duplication·footprint 恒等 = 理论 floor 702 MiB/token). 判别键在结构轴: 同字节下我方 **1.99× 指令/token** (qh 平面+超块重建在 M=1 不摊销) → 1.37× cycles → **0.729× decode** (独立复现 g5 sealed). **P0 达 5/5 全 H1** (q5_K@k1 从 H1-implied 升 H1-measured); 跨板对照坐实 **GEVM plan 攻结构·k1/VLEN256 同样适用** (prefill WIN↔decode LOSS = 同 VLA 核 GEMM 兼职 GEVM 结构损·两板同机制). restore=14b6add6 OFF-pristine (md5+nm 双证).

## reversibility
- 板改动仅 swap LIVE .so (测中 ON·测后 restore OFF-pristine). source 全程 pristine·无 rebuild·NO git.
- restore 证: live md5=14b6add6 (== g5-M2 sealed OFF)·live q5_K tcrv 符号=0·RUNPATH 副本亦 14b6add6·无遗留进程.
