# G5-M2 q4_1 曳光弹 — L-接线② 净新 scaffold（复用 q5_1 q8_1-activation 家族素材）· CASEFILE

> **campaign**: G5 接线战役 · **M2 L-接线② q4_1**（复用 q5_1 净新 scaffold recipe·**零改动复用 q8_1 sub-scaffold** = q8_1-activation repack 家族第 2 成员）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree f3e1828（**测后 restore 验 clean**·live .so 05a62e6a OFF-pristine·3 源 baseline md5）
> **HEAD (TianChen-RV)** = daf6db23 · 禁 git
> **★结论 = GREEN-6/84**: **q4_1 净新 scaffold BUILT + correctness GREEN + perf GREEN（prefill-axis）→ perf-covered 5/84→6/84**。q4_1 = q5_1 减 5th-bit qh（更简单）= q4_0 nibble decode + min fold + **q8_1 激活**（复用 q5_1 净新 block_q8_1x4 + `ggml_quantize_mat_t<1,Q8_1>`）= **q8_1-activation repack 家族第 2 成员**。UT 双 GREEN（interleaver 1.97e-06 / GEMM 6.36e-06）·correctness 5/5 byte-identical A==B·PPL 22.05 coherent·objdump vl=8 sealed·**perf prefill 3.6844× WIN + decode 1.6668× WIN（atypical·诚实 caveat：weak stock baseline）**·八门全过 prefill 轴·双账本 kernel==system·测后 board restore 验 clean。

## 一、emit（host·LLVM20.1.8）
- fixtures 已存（无需新写）：`test/Conversion/RVV/rvv-emit-quant-contraction-q4-1-repack-gemm-prefill-vlen128.mlir`（GEMM）+ `rvv-emit-identity-quant-contraction-q4-1-repack-vlen128.mlir`（GEVM）。
- `tcrv-opt --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`
- 产物：`tcrv_emitted_gemm_q4_1.inc`（md5 31912b13）· `tcrv_emitted_gevm_q4_1.inc`（md5 d645aa42）。plain nibble（无 qh vor）+ min fold + stride 320（无 qh）/ act 144。

## 二、净新 scaffold（deploy_patch_q4_1_emitted.py·3 文件 tracked·可逆·无 arch-fallback.h）
复用 q5_1 recipe + q4_1 简化 delta：① weight block 320B（d@0/m@32/nibbles@64·**无 qh**）；② **零改动复用 q5_1 净新 q8_1 基建**（block_q8_1x4 144B + inline `ggml_quantize_mat_t<1,GGML_TYPE_Q8_1>`）；③ make_block_q4_1x16（d@0/m@32/qs@64·**无 qh 转置**）+ generics unsigned nibble+min（无 qh）+ dispatch case Q4_1。

## 三、验证链
- **UT（board·独立 oracle·MIRAGE de-risk）**：interleaver/GEVM GREEN fails=0/16 max_rel 1.97e-06；GEMM/prefill GREEN fails=0/64 max_rel 6.36e-06。
- **build+seal（gcc-15.2.0 对称）**：OFF 05a62e6a 0syms / ON a3774cdf 2syms·banner gevm=1 gemm=1·objdump vsetivli imm=8=1 imm16/64=0（VLEN128-safe）·源 restore byte-exact。
- **correctness GREEN（hard gate）**：DeepSeek-8B-Q4_1（自 Q5_0 requantize·决策卡④·sha256 e10c9d99）·5/5 prompt byte-identical A(emit)==B(stock)·44 banner fires·no NaN/Inf·PPL(ON)=22.0523 coherent。
- **perf 分相**：prefill 3.6844× WIN / decode 1.6668× WIN（n=20·relIQR≪地板·⟨§七 / phase_split_raw.txt⟩·诚实 caveat：stock q4_1 无 repack path=弱 baseline）。

## durable files
- `evidence.md`（recon + emit + 净新 scaffold + q8_1 复用 + perf §七 + 诚实 caveat）
- `tcrv_emitted_gemm_q4_1.inc`（md5 31912b13·25KB·GEMM prefill emitted vl=8）
- `tcrv_emitted_gevm_q4_1.inc`（md5 d645aa42·14KB·GEVM decode emitted vl=8）
- `seal_raw.txt`（build OFF/ON + patch + objdump vl=8 seal log）
- `objdump_gemm_q4_1_seal.txt`
- `objdump_gevm_q4_1_seal.txt`
- `correctness_GREEN_raw.txt`（5/5 byte-identical·44 banner·PPL 22.05）
- `phase_split_raw.txt`（n=20·llama-bench json·freq 2.6GHz）
- `transmission_accounting.csv`（perf 双账本·八门键·诚实 caveat）
- `.gitignore`（board-src/ + *.log ignore）

> board harness 住 `tools/e2e-harness/board/g5-m2-q4_1/`。两 .inc 小（39KB total）保留 tracked（无 board-src/·无需 gitignore）。
