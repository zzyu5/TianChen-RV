# MANIFEST — g7-l1-gevm / P2-board-verify

- campaign: G7 主战役 (L1 GEVM plan) · **P2 板 IPC 验证** = P1 colgroup-tiled GEVM plan 的 perf 兑现验证.
- role: 裁 P1 GEVM plan(806a7cc1) 在 q4_K@rvv decode M=1 是否修复 P0 的结构损(IPC 0.29→≥0.52 收敛).
- board: `ssh rvv` openEuler VLEN128 64c · cores 8-15 disjoint-pin · co-tenant vLLM(cores 0-7)未重启.
- account: kernel/system 双账 · ours(部署 emitted colgroup-tiled GEVM) vs OLD(per-column GEVM 915e6733) vs STOCK(gcc-15 block-dot 05a62e6a) · 三 .so 交织净测 · 披露编译器身份(NEW/OLD=clang-18·STOCK=gcc-15).
- 口径: perf stat -e instructions,cycles,cache-misses 两点差分(N16,64)/48 · DRAM=cache-misses×64(+0.05%校准) · taskset 8-15.
- NO git · swap-only · 测后 byte-exact restore live=05a62e6a · md5 双证.

## ★裁决 (一句)
**成功信号 NEGATIVE** — colgroup-tiled GEVM plan(TG=2) 在 M=1 令 IPC **反向下降 0.294→0.214**·cyc/tok +45%(vs OLD)·decode 0.49×→**0.34×**回归·字节仍 roofline(0.998×·H2 REJECTED)=**纯结构回归**·根因 = **register-resident bank 溢出反噬**(反汇编 scalar spill 534 vs 10·53×·[GAP-P1] register-pressure trap 同型)。**P1 构造(byte-exact·mechanized·[K-10] novelty)仍成立**·P2 = 首版结构假设 perf 未兑现·需迭代(反 register-pressure 方向).

## canon 依据 (读死)
- P0 evidence (H1 结构损·字节触底 roofline·M=1 无算力藏延迟) · P1 evidence §6 recipe (colgroup-tiled plan·byte-exact 三证).
- [K-10] 结构级 plan · [PAT-2] P9 GEVM regime-plan · [GAP-REPACK-GEVM] (布局非墙) · [GAP-P1] re-roll/register-pressure trap · 性能宪章规则1(反汇编认瓶颈)/规则2(直觉投影不可信).

## provenance (板·swap-only 无 rebuild 残留)
- NEW: /tmp/g7p2/libggml-cpu.so.q4kON.gevmct.clangrepack (md5 8f48c116) · mixed build(gcc base + clang-18 repack TU·relink g++) · P1 emitter 806a7cc1 → weft-opt+mlir-translate → tcrv_emitted_gevm_ct_q4_K.inc(md5 75af07f9) → deploy_patch_q4k_gevm_ct.py.
- OLD: /tmp/dkgv/libggml-cpu.so.q4kON.clangrepack (915e6733·per-column GEVM·P0 seal).
- STOCK: /tmp/dkgv/libggml-cpu.so.q4kOFF.gcc (05a62e6a·gcc-15 as-shipped block-dot).
- engage: "TCRV G7-P2 EMITTED GEVM-CT(q4_K_16x1 VLEN128 colgroup-tiled TG=2) ENGAGED"(decode) · "TCRV G5-M2 EMITTED GEMM"(prefill).
- model: /home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf (8.03B).

## durable files
- evidence.md — correctness GREEN + perf 三值表(STOCK/OLD/NEW) + 成功信号判读 NEGATIVE + 反汇编归因 + M-SCAN selector 分流 + M* + 污染判(insn-based) + schema label 建议 + restore.
- raw/pertoken_twopoint.txt — 三变体 per-token 两点差分总表 + 裁决.
- raw/results.txt — 3-pass × 3-variant × N{16,64} 原始 perf 计数 + M-SCAN banner + MEASURE-DONE(live restored).
- raw/new_gevmct.dis / raw/old_gevm.dis — NEW vs OLD kernel objdump (spill 534 vs 10 归因证据).
- raw/seal_gevmct.dis — NEW kernel seal disasm (vsetivli16=0·spill=12·vl=8).

## reversibility
- 板改动仅 swap LIVE build-gcc15-rv64gcv/bin/libggml-cpu.so.0.15.1(测中 ON·测后 restore stock). source 全程 pristine(mixed build 后 pristine gcc rebuild 复原·GEN=deb61a29·ARCH=99131cf7·CML=4426f543)·NO git.
- restore 证: live md5=05a62e6a82857cdf247ba1933c695849(== stock-pristine byte-exact)·live q4_K/gevm_ct emitted 符号=0·ARCHDIR 无 stray .inc·无遗留进程.
