# G5-M2 q6_K 曳光弹 — L-接线② 净新 riscv scaffold · K-quant no-min · CASEFILE

> **campaign**: G5 接线战役 · **M2 L-接线② = q6_K**（测 net-new scaffold 路径是否延伸到 K-quant 超块）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 nproc=64 · A-tree restored baseline（deb61a29/57851439/99131cf7·live .so 05a62e6a 0 syms）
> **HEAD (TianChen-RV)** = 6f9099e5（禁 git·零 tracked lib 源改动·scaffold 全在 board deploy patch）
> **结论**: correctness **GREEN**（emitted vl=8 q6_K repack = silicon UT byte-exact + greedy A==B 5/5 byte-identical）· **perf prefill 重回退 ON/OFF < 0.07× → yellow-kernel-axis · perf-covered 维持 6/84**（q6_K weight-reconstruction-bound·emitted vl=8 repack ~15× 慢于 stock hand-tuned block-dot）。
> **登记性质**: C1 K-quant 超块净新 scaffold e2e 结构延伸（correctness）· **非 perf-covered 绿格**（perf 不延伸到 K-quant）。
> **上游 framing**: q6_K riscv `1,16` 路径【净新】（上游仅 aarch64 block_q6_Kx8·NEON-gated·rvv 恒 nullptr）；q8_K 激活【全 present 上游】→ 无 activation sub-scaffold。

## durable files

- `evidence.md` — recon + emit + net-new scaffold + silicon UT + build/seal + correctness + perf 分相 + A-tree restore 全数据
- `correctness_GREEN_raw.txt` — greedy A==B 5/5 byte-identical · 43 engage banners · no NaN/Inf · CORRECTNESS_GATE GREEN
- `L7_reverse-control_disasm.md` — 二.1 L-7 反汇编钉死（fallback-exclusion 三链 + 反向控制 + 0.07× 全展开-vectorized-repack 构成 + 部署五验 + 归因双层·判决=首发不成熟→provisional 转正 黄-对手更强 final）
- `L7_raw_wrapper-nm-objdump.txt` — L-7 raw（nm ON/OFF + wrapper objdump[beq VLEN128·generic not-taken unreachable] + emitted 指令统计 + hot-body window）
- `.gitignore` — gitignore 两大 emitted .inc（3MB·regenerable）

> emitted `.inc` = **gitignored**（`tcrv_emitted_gemm_q6_K.inc` md5 9c49195c·1.8MB / `tcrv_emitted_gevm_q6_K.inc` md5 768a3892·1.1MB·regenerable via evidence.md §二 emit recipe·避 repo bloat）。
> board harness 脚本住 `tools/e2e-harness/board/g5-m2-q6_K/`（deploy_patch_q6_K_emitted.py · ut_q6_K_build_run.sh · g5_m2_q6_K_build_seal.sh · g5_m2_q6_K_correctness.sh · g5_m2_q6_K_phase_split.sh · analyze_phase_split.py · kquant_repack_verify_q6K.c）。
