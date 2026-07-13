# MANIFEST — rvv-iqtq-cold (G7 L1 货架A·rvv §1.2 IQ/TQ vec_dot 族 + iq4_nl gemm 补 cold)

- **campaign**: G7 L1 kernel-sym 全量·rvv §1.2 收口（承 rvv-batch §1.1 FLAT-5·同形状 hot/cold·gcc-15 对称）。第二赛道 kernel-sym·NOT e2e·NOT perf-covered·不入 ≥parity 计数。
- **board**: rvv / openEuler / VLEN128 / core8 (disjoint-pin 8-15) / gcc-15.2.0 出货对称 / 2026-07-14 load~2.3.
- **status**: BOARD-MEASURED. 9/9 cell cold <parity（verdict 不翻）·cold≈hot·两 headline（tq2_0 hot 1.15× cache-only 惊喜 §3·iq4_nl gcc-symmetric 0.23× 校正 l1-m2 clang 0.837× §4）。**主树/build/.so/governor 未改·.so md5 d1adc634 双证·无 git add·无 stray proc。**

## durable files
- `evidence.md` — 主写：净结论表(9 cell hot/cold/对手机判/fp/regime)·cold 协议·cold≈hot 逐格·tq2_0 惊喜·iq4_nl 校正·污染纪律·restore·T9 §1.2 cold 建议(留主会话)。
- `raw/vecdot_hotcold_sweep.log` — board raw 8 vec_dot hot/cold N=12 sweep（RESULT lines + load pre/end）。
- `raw/iq4nl_gemm_matrix.txt` — iq4_nl gemm nr{64,16}×mode{hot,cold}×seed{C0FFEE,BEEF01} + compiler-gap(spill 7 clang / 41 gcc)。
- `raw/confirm_remeasure.txt` — tq2_0 hot 3-seed N=20 confirm + high-relIQR cold N=16 复测。
- `raw/exported_kernel_md5.txt` — 9 exported .kernel.c provenance (md5)。

## 复现 recipe（board scratch `/tmp/g7_iqtq_cold` ephemeral·确定性可再生）
1. **vec_dot ours 导出（host）**: `git show 0ca224f7:test/Target/RVV/<dashfmt>-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir | .worktrees/cache/0ca224f7.../tcrv-opt - --tcrv-rvv-materialize-<dashfmt>-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp > <fmt>.kernel.c`（8 fmt）。
2. **iq4_nl ours 导出（host）**: `build-weft/bin/weft-opt test/Conversion/RVV/rvv-to-emitc-repack-gemm-iq4-nl-q8-0.mlir --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`（weft_emitc symbol）。
3. **board build**: vec_dot ours = g++-15.2.0 -O3 -march=rv64gcv_zfh_zvfh_zba_zbb_zbs；factory.o = gcc-15.2.0 -O3 编 `$R/ggml/src/ggml-cpu/{arch/riscv/quants.c,quants.c}` (ld -r)·`$R=/home/ubuntu/llama.cpp-upstream-native`(建出货 .so 同源)；link driver(format_micro_driver.c) + ours + factory + -lggml-cpu -lggml-base -lggml -lgomp -lm。iq4_nl ours = g++-15 -O3·link iq4nl_gemm_hc_driver.c + ours + .so(ggml_vec_dot_iq4_nl_q8_0)。
4. **run**: vec_dot `taskset -c 8 ./iqtq_bench <fmt> <ours|factory> 4096 12 <pool: 0=hot|256=cold> 0 <seed>`；iq4_nl `taskset -c 8 ./iq4nl_bench iq4_nl 2048 <nr> 512 12 <seed> <hot|cold>`。ratio=factory/ours(vec_dot) / ours_gmacs/opp_gmacs(iq4_nl)。
5. **对手机判**: `nm factory.o` → 8× ggml_vec_dot_<fmt>_q8_K DEFINED-T；`nm -D libggml-cpu.so` → ggml_vec_dot_iq4_nl_q8_0 @0x937da T。

## [NG-4] discipline
L1 kernel-axis micro·**NOT a beat**·9/9 LOSS（cold verdict 不翻）·不入 kernel-sym ≥parity 计数·禁写 e2e 语境。cold=hot robustness 补强。Nothing committed·主树/build/.so 未改。
