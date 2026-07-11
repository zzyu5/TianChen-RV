# G5-M2 q5_1 曳光弹 — L-接线② 净新 scaffold（复用 q5_0 recipe + q8_1 min 家族）· CASEFILE

> **campaign**: G5 接线战役 · **M2 L-接线② q5_1**（复用 q5_0 净新 scaffold recipe·+ 净新 q8_1 activation mat-quantizer = q8_1-activation repack 家族首格）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree f3e1828（**测后 restore 验 clean**·live .so 05a62e6a OFF-pristine·3 源 baseline md5）
> **HEAD (TianChen-RV)** = d92a706d · 禁 git
> **★结论 = GREEN-5/84**: **q5_1 净新 scaffold BUILT + correctness GREEN + perf GREEN（prefill-axis）→ perf-covered 4/84→5/84**。q5_1 = q5_0 qh gather（unsigned/无 -16）+ q4_1 min fold + **q8_1 激活**（净新 block_q8_1x4 + `ggml_quantize_mat_t<1,Q8_1>` mat-quantizer，上游只有 Q8_0/Q8_K）= **q8_1-activation repack 家族首格**。UT 双 GREEN（interleaver 4.99e-07 / GEMM 3.35e-06）·correctness 5/5 byte-identical A==B·PPL 18.84 coherent·objdump vl=8 sealed·**perf prefill 1.0900× WIN（n=20·relIQR≪地板）/ decode 0.7836× regression 披露**·八门全过 prefill 轴·双账本 kernel==system·测后 board restore 验 clean。

## 一、emit（host·LLVM20.1.8）
- fixtures 已存（无需新写）：`test/Conversion/RVV/rvv-emit-quant-contraction-q5-1-repack-gemm-prefill-vlen128.mlir`（GEMM）+ `rvv-emit-identity-quant-contraction-q5-1-repack-vlen128.mlir`（GEVM）。
- `tcrv-opt --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`
- 产物：`tcrv_emitted_gemm_q5_1.inc`（md5 0f52ad46）· `tcrv_emitted_gevm_q5_1.inc`（md5 84b69d60）。unsigned（vsub=0）+ min fold（vfadd present）+ stride 384/qh 320/act 144。

## 二、净新 scaffold（deploy_patch_q5_1_emitted.py·3 文件 tracked·可逆·无 arch-fallback.h）
复用 q5_0 12-piece recipe + q5_1 三 delta：① weight block 加 m@32；② **净新 q8_1 activation 基建**（block_q8_1x4 144B + inline `ggml_quantize_mat_t<1,GGML_TYPE_Q8_1>` byte-match stock quantize_row_q8_1）；③ make_block_q5_1x16（d@0/m@32/qs@64/transposed-qh@320）+ generics unsigned+min + dispatch case Q5_1。

## 三、验证链
- **UT（board·独立 oracle·MIRAGE de-risk）**：interleaver/GEVM GREEN fails=0/16 max_rel 4.99e-07；GEMM/prefill GREEN fails=0/64 max_rel 3.35e-06。
- **build+seal（gcc-15.2.0 对称）**：OFF 05a62e6a 0syms / ON 38377d3a 2syms·banner gevm=1 gemm=1·objdump vsetivli imm=8=1 imm16/64=0（VLEN128-safe）·源 restore byte-exact。
- **correctness GREEN（hard gate）**：DeepSeek-8B-Q5_1（自 Q5_0 requantize·决策卡④）·5/5 prompt byte-identical A(emit)==B(stock)·45 banner fires·no NaN/Inf·PPL(ON)=18.8388 coherent。
- **perf 分相**：⟨§八 / phase_split_raw.txt⟩。

## durable files
- `evidence.md`（recon + emit recipe + 净新 scaffold + q8_1 delta + 全锚点）
- `correctness_GREEN_raw.txt`（5/5 byte-identical·PPL·banner）
- `seal_raw.txt`（build+patch+seal 全 log·objdump vl=8）
- `objdump_gemm_q5_1_seal.txt`
- `objdump_gevm_q5_1_seal.txt`
- `phase_split_raw.txt`
- `transmission_accounting.csv`（perf·双账本·八门键）
- `tcrv_emitted_gemm_q5_1.inc`（md5 0f52ad46·30KB）
- `tcrv_emitted_gevm_q5_1.inc`（md5 84b69d60·21KB）
- `.gitignore`

> board harness 住 `tools/e2e-harness/board/g5-m2-q5_1/`（deploy_patch·build_seal·correctness·phase_split·ut ×2·analyze_phase_split.py）。两 .inc 小（51KB）保留 tracked。
