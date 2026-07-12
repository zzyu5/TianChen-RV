# MANIFEST — decisive-kquant-gcc-vs-vlen

- campaign: 裁二.2 决定性实验 — rvv q4_K e2e LOSS 归因隔离（gcc codegen vs VLEN128 重建摊销）
- role: 隔离 gcc 因素。同板(rvv/VLEN128)·同格(q4_K)·同分母(gcc block-dot)·仅换 emitted repack kernel 编译器
  (gcc-15 → clang-18)·重跑 e2e A/B。gcc-death 冠名权前置。
- board: `ssh rvv` openEuler 24.03 VLEN128 64c · cores 8-15 perf-gov 2.6GHz · load ~2.2 (<10 gate)
- account: internal-A/B 机制探针 ([NG-4]·不计 perf-covered) · 对手 stock generic ggml_vec_dot_q4_K_q8_K block-dot
- NO git (主会话 commit) · board reversible (source byte-exact restore + pristine rebuild)

## provenance (读死)
- rvv 0.42×/0.334× 源 = experiments/active/g5-wiring/M2-q4_K/evidence.md (e2e) + result-tables/T-PERF1b_q4k_e2e_prefill_regression.md
- 742-spill 源 = docs/reports/2026-07-10-{q4k-micro-vs-e2e-regression-mechanism,CASE-COMPILER-ASYMMETRY-casefile}.md
  (sibling md5 90d454da) — 本实验首次 objdump EXACT 部署 kernel 6cbd9c19/e909a9bd 闭 provenance gap
- emitted kernel = M2 tcrv_emitted_gemm_q4_K.inc(6cbd9c19) + tcrv_emitted_gevm_q4_K.inc(e909a9bd)
- deploy = M2 tools/e2e-harness/board/g5-m2-q4k/deploy_patch_q4k_emitted.py (case128 flip + emitted intercept)

## board build/toolchain
- gcc-15.2.0 @ /opt/tcrv-toolchains/gcc-15.2.0 (shipped baseline compiler for rvv)
- clang-18.1.8 @ /opt/tcrv-toolchains/llvm-18.1.8 (TCRV LLVM = "our clang"; has __riscv_vcreate_v_*;
  link via -B/usr/lib/gcc/riscv64-openEuler-linux/12 + -isystem libstdc++/12; -c only for repack TUs)
- march (gcc): rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
- march (clang, repack TUs): rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause (drop experimental; kernel semantics unchanged)
- model: /home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf (== T-PERF1b apples-to-apples)

## durable files
- evidence.md — verdict + spill table + e2e A/B
- MANIFEST.md — this
- .gitignore — ignore *.so/*.o/*.inc/raw big binaries (regenerable via scripts)
- objdump_spill_compare.sh — standalone 3-compiler spill objdump of EXACT M2 kernels
- clang_build_seal.sh — (superseded, full-clang-tree attempt; kept for record; blocked by quants.c inline-asm)
- mixed_build.sh — SURGICAL compiler-isolation build (gcc tree, only 2 repack TUs clang-recompiled + relink)
- measure_ab.sh — parametrized e2e A/B (NUMER repack vs DENOM block-dot, phase-split)
- correctness_ab.sh — greedy A==B correctness gate + engage banner + NaN sanity
- parse_ab.py — llama-bench JSON phase-split aggregator
- raw/ — objdump_spill_compare.txt + (gitignored) json/dis/libs

## reversibility
- 板改动: 仅 source patch (repack.cpp GEN+ARCH) 期间 + scratch build-clang17-dkgv(removed) + /tmp/dkgv scratch
- 恢复: GEN=deb61a29 ARCH=99131cf7 CML=4426f543 byte-exact + gcc pristine rebuild (live lib=05a62e6a) + 0 stray .inc
