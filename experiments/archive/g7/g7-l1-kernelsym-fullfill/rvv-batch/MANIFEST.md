# cell MANIFEST — g7-l1-kernelsym-fullfill/rvv-batch（货架A 全量 rvv-半·FLAT-5 hot/cold + 变体穷举）

- **campaign**: G7 重编令 §L1 货架A 全量 — rvv-半 kernel-sym 格补 cold 行 + 变体穷举（T9 §1.1/§1.2 派生·[VERIFY-LADDER] G2 层·同形状 cold micro）。
- **status**: ACTIVE — **BOARD-MEASURED 2026-07-13**·rvv（localhost.localdomain / openEuler / VLEN128 vlenb=16 / L1d64K L2 2M **L3 64M** / 64c / **core 8** disjoint-pin / gcc-15.2.0 出货对称域）。**no git**（主树 + build/ 未改·board scratch 纯 micro·ggml .so md5 双证 UNCHANGED）。
- **role**: 第二赛道 kernel-sym（kernel-axis micro·同编译器/flags/march 对称·NOT e2e·NOT perf-covered·与系统账 9/83 永不混算）。§1.1 FLAT-5 逐格 cold 双态（同形状 nr=16 锚）+ nr-shape 变体穷举 {4,16,64}（T4b selector 自体 oracle·[L-4]）+ ZERO-MODEL 数值门 + 对手符号机判 + gcc/clang 双账本。**§1.2 pending（第二 batch）**。

## board identity fingerprint
- host=localhost.localdomain / openEuler / VLEN128 (vlenb=16) / rv64gcv+zfh+zvfh / 64c / **core 8** / L1d=64K L2=2MiB **L3=64MiB**（cold pool ws 82.6–110.1MiB > 64MiB LLC）。
- co-tenant: vLLM Worker + kernspan artifact loop（在别核·测前测后 core 8-15 仅系统 daemon·未触碰/未重启·disjoint）。
- toolchains（/opt/tcrv-toolchains via env.sh）: **kernel-sym symmetric = gcc-15.2.0**（rvv shipped=gcc-15 → kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered）；deploy = clang-18.1.8（双账本）；link/harness always gcc-15。
- opponent = 板自有 `libggml-cpu.so` md5 `d1adc634c2ca04ffc389536b30d9d9c4`（gcc-15 rv64gcv·= FLAT-5/batch1 同一 .so·测前后 md5 UNCHANGED = restore 双证）。opponent 符号 = **机判** nm -D public T `ggml_vec_dot_qX_qY`（NOT `_generic`·NOT hand-brick）。

## method（touch-set 全在 scratch·主树只读）
1. **Export**（host·**build/ READ-ONLY**）：`build-weft/bin/weft-opt`（current HEAD·weft 方言·2026-07-13 build）从 current-HEAD test/Conversion/RVV/rvv-emit-quant-contraction-{q4-0,q4-1,q5-0,q5-1,q8-0}-repack-gemm-prefill-vlen128.mlir export：
   `weft-opt <mlir> --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`（5/5 ZERO error·symbol prefix `weft_emitc_`）。exported .kernel.c md5 见 `raw/exported_kernel_md5.txt`。
2. **cold driver**（`flat_gemm_cold_driver.c`·= tools/e2e-harness/board/flat_gemm_paired_driver.c 扩 P-tile POOL cold 协议）：P=140 独立 weight tile·ws >64MiB LLC·每 round 扫全池 OURS 再 OPP·median-of-12。activation 单份（decode 暖激活/冷权重）。报 achieved weight-GB/s（roofline）。
3. **ZERO-MODEL 数值门**（correctness before timing·tile 0）：独立 fp64 scalar decoder vs ggml block-dot vs 我方 repack；relerr_ours≈relerr_opp·nbad=0·GATE=PASS 全 15+5。
4. **变体穷举**：codegen 候选集 {mf2} 单一合法（VLEN128 rv64gcv·m1=RVV0.7-only 非 same-board 候选·passes.td:386/[repack-winA-always-mf2]）；实测轴 = nr-shape {4,16,64}（同批 → T4b selector 自体 oracle）。
5. **双账本**：gcc-15 -O2（主·kernel-sym·nr{4,16,64}×5）/ clang-18 -O3（deploy·nr16 锚×5）。
6. **对手符号机判 + objdump 对称双证**（raw/objdump_symmetry.txt）。

## HEADLINE（gcc-15 主账·N=12·同形状锚 nr=16·median ratio ours/opp·VLEN128 K=2048 nc=512）
| fmt | HOT | COLD | cold regime | 对手(机判·折中态) | verdict |
|---|---:|---:|---|---|---|
| q4_0 | 6.780× | 6.224× | compute-bound(0.27GB/s) | vec_dot_q4_0_q8_0·单实现折中·light-vec 弱 | ≥parity |
| q4_1 | 6.747× | 6.105× | compute-bound(0.30) | vec_dot_q4_1_q8_1·单实现折中·light-vec 弱 | ≥parity |
| q5_0 | 1.228× | 1.232× | compute-bound(0.17·cold≈hot) | vec_dot_q5_0_q8_0·单实现折中·better-vec 较强 | ≥parity |
| q5_1 | 1.499× | 1.412× | compute-bound(0.18) | vec_dot_q5_1_q8_1·单实现折中·better-vec 较强 | ≥parity |
| q8_0 | 4.193× | 4.752× | memory-leaning(cold>hot) | vec_dot_q8_0_q8_0·**破损**·light-vec 弱 | ≥parity |

**15/15 cell（5fmt×nr{4,16,64}）hot∧cold 全 ≥parity·GATE=PASS·correctness ZERO-MODEL bit-exact·对手=factory block-dot 机判（非 hand-brick·非 SELF）。clang deploy 双账本 10/10 cold ≥parity（verdict 稳健）。**

## judgment
- **kernel-sym 第二赛道 cold 补强**：FLAT-5@rvv §1.1 hot ≥parity（f8da2f5c 立账）→ 本 batch **同形状 cold 双态确认全 ≥parity**（cold 最严 penalty q5_1 nr4 1.124× 仍 ≥parity）·两账本稳健。
- **★rvv-特有 cold-penalty 结构**（k1 无 L3 看不到）：cold-penalty 键控 memory-exposure·nr↓→penalty↑（q4_0/q4_1/q5_1 在 nr4 penalty 最大·nr64→cold≈hot）；q8_0 反常 cold>hot（对手 block-dot 权重足迹大·cold 退化 > ours）。
- **T4b oracle**：codegen 候选集 {mf2} 单一合法·selector 平凡最优·变体穷举净信息 = nr-shape 轴（mf2-repack 跨 nr 全 ≥parity·q5_0/q5_1 win↗nr）。
- **纪律**：仅 kernel-axis 覆盖面·NOT e2e·系统账仍 [GAP-FLAT-E2E] 黄格·禁混算 perf-covered 9/83。计数不因 cold 补强变（cold=hot robustness·非新格）。
- **§1.2 pending**：q5_K/q2_K/q3_K/q6_K/iq4_nl gemm + iq/tq vec_dot 需独立 driver·第二 batch·regime 已从 T8 确认 compute/weight-bound LOSS（cold 预期不翻）。

## durable files
- `evidence.md` — 主发现·per-cell 表·变体穷举·折中状态·cold-penalty 机制·双账本·§1.2 pending·T9 建议。
- 探针 flat_gemm_cold_driver.c / run_rvv_cold.sh → 抽出至 tools/e2e-harness/board/g7-l1-kernelsym-fullfill/（G8 §一 大扫除·2026-07-14·可复用探针不埋 archive）。
- `export/gemm_{q4_0,q4_1,q5_0,q5_1,q8_0}.kernel.c` — 5 exported repack GEMM 内核（weft-opt current HEAD·md5 in raw/）。
- `raw/gcc_sweep.log` — gcc-15 主账 board stdout（15 HOT + 15 COLD + header + 探针）。
- `raw/clang_anchor_nr16.txt` — clang-18 deploy 双账本 nr16 锚。
- `raw/summary_gcc.csv` — 15 cell 解析表。
- `raw/objdump_symmetry.txt` — OURS vs OPP rvv-insn 对称双证 + 对手符号 addr。
- `raw/exported_kernel_md5.txt` — 5 kernel provenance。

## data-only cell
证据+recipe only。exported .kernel.c 确定性可再生（method §1·weft-opt current HEAD）。board scratch `/tmp/g7_l1_cold`（driver+kernel+.o+bin+log·ephemeral·未删留复现·主树/build/.so/governor 未改·md5 双证）。**Nothing git-added.**
