# cell MANIFEST — kernel-sym-flat5-rvv-micro（第二赛道 kernel-sym 首轮立账）

- **campaign**: 令三·第二赛道 kernel-sym 台账立账 — FLAT 5 gemm@rvv/VLEN128 逐格独立对称 kernel-axis MICRO A/B（补齐 T9 §1.3 「≥parity 待板批对称 micro 补测」5 格）。
- **status**: ACTIVE — **BOARD-MEASURED 2026-07-13**, rvv（localhost.localdomain / openEuler / VLEN128 / rv64gcv+zfh+zvfh / 64c / core 14 perf-gov 2.6GHz）。**no git**（main tree + build/ untouched·board deployment tree 未改·纯 micro）。
- **role**: 第二赛道 kernel-sym（kernel-axis micro·同编译器/flags/march 对称·NOT e2e·NOT perf-covered·与系统账 9/83 永不混算）逐格 N=12 对称 A/B + ZERO-MODEL 数值门 + 对手身份 objdump 探针 + 三账本（gcc-O3 主 / gcc-O2 稳健 / clang-O3 deploy）。

## board identity fingerprint
- host=localhost.localdomain / openEuler / VLEN128 / rv64gcv+zfh+zvfh / 64c / **core 14** perf-gov 2.6GHz
- co-tenant: vLLM Worker_TP pinned **cores 0,1**（disjoint·测前测后 `taskset` 双证仍在 0,1·未触碰/未重启）
- toolchains（board /opt/tcrv-toolchains via env.sh）: **kernel-sym symmetric = gcc-15.2.0**（`riscv64-unknown-linux-gnu-gcc (GCC) 15.2.0`·= 板出货 ggml 编译器·rvv shipped=gcc-15 → kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered）; deploy = clang-18.1.8; link/harness always gcc-15（neutral）。
- opponent = 板自有 `libggml-cpu.so` md5 `d1adc634c2ca04ffc389536b30d9d9c4`（gcc-15 rv64gcv build·= flat-covering-batch1 同一 .so·测前测后 md5 UNCHANGED = restore 双证）。opponent symbol = arch-dispatched `ggml_vec_dot_qX_qY`（NOT `_generic`）。

## method（touch-set 全在 scratch·main tree 只读）
1. **Export**（host·**build/ READ-ONLY**）：host `build/bin/tcrv-opt`（stale·知 `tcrv` 方言·build 2026-07-12·rename fc72fe53 前）不识当前 HEAD 的 `weft.exec.kernel` 测试文件 → 从 **pre-rename commit `49057e34`(=fc72fe53~1)** 取 5 个 `rvv-emit-quant-contraction-{q4-0,q4-1,q5-0,q5-1,q8-0}-repack-gemm-prefill-vlen128.mlir` export：
   `tcrv-opt <mlir> --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`（5/5 ZERO error）。
   **★semantic-identity 证**：pre-rename mlir vs current-HEAD(weft) mlir diff = **仅 CHECK 行的 emit 符号前缀 `tcrv_emitc_` vs `weft_emitc_`**（rename=机械 byte-exact 零漂移·commit fc72fe53）→ kernel body 字节相同·measured kernel 语义==current HEAD（唯一差异是 extern 符号名字符串·零 codegen/perf 影响·driver+kernel 内部一致用 `tcrv_emitc_`）。exported .kernel.c md5 见 `raw/exported_kernel_md5.txt`。
2. **ZERO-MODEL 数值门**（correctness before timing·in-driver·每 rep 独立 seed）：随机 f16-scaled PLAIN 块 → 独立 fp64 scalar decoder（零复用我方 intermediate）vs ggml block-dot vs 我方 repack kernel；`relerr_ours≈relerr_opp`·`nbad=0`·GATE=PASS 全 60 reps。
3. **Timing**：driver best-of-7（warmup + iters=25），外层 **REPS=12**（每 rep 独立 seed·打印 1 FLATGEMM 行）·`taskset -c 14`。K=2048 nr=16 nc=512（macs=nr·nc·K）。median + relIQR（linear-interp Q1/Q3）逐格算。
4. **三账本**：gcc-15 -O3（主·kernel-sym）/ gcc-15 -O2（稳健·复现 batch1）/ clang-18 -O3（deploy 双账本）。harness = `flat_gemm_paired.sh`（改：9th arg=kernel opt level·default core 14）+ `flat_gemm_paired_driver.c`（原样·来自 tools/e2e-harness/board/）。
5. **objdump 对称双证**：OURS(gcc-15 O3 .o) vs OPP(gcc-15 .so symbol) rvv-insn 计数（raw/objdump_symmetry.txt）。

## HEADLINE（kernel-sym 主账 gcc-15 -O3·N=12·median ratio ours/opp·VLEN128 K=2048 nr=16 nc=512）
| fmt | ratio_med | relIQR | ours_gmacs | opp_gmacs | opp rvv-insn | verdict |
|---|---:|---:|---:|---:|---:|---|
| q4_0 | 6.707× | 0.51% | 8.263 | 1.233 | 10 (light·弱) | ≥parity |
| q4_1 | 6.829× | 0.84% | 8.630 | 1.264 | 8 (light·弱) | ≥parity |
| q5_0 | 1.221× | 5.35%(opp-driven) | 3.848 | 3.151 | 26 (better·较强) | ≥parity |
| q5_1 | 1.407× | 4.06%(opp-driven) | 3.993 | 2.844 | 24 (better·较强) | ≥parity |
| q8_0 | 4.077× | 0.52% | 5.095 | 1.249 | 7 (light·弱) | ≥parity |

**5/5 ≥parity·correctness bit-exact·对手=factory block-dot as-shipped（非 hand-brick·非 SELF）。kernel-sym ≥parity 计数建议 4→9（入账留主会话）。**

## judgment
- **kernel-sym 第二赛道**：5/5 逐格独立 ≥parity·N=12·gcc-15 对称·correctness-gated·opponent-identity objdump-probed·三账本(O3/O2/clang)+batch1 全一致（verdict 不翻转）。
- **成色**：对手皆 factory block-dot（q4_0/q4_1/q8_0 light-vec 弱·q5_0/q5_1 better-vec 较强）·**均非 hand-brick 强对手**（唯一 hand-brick ≥parity = q4_K@k1）。
- **纪律**：本 5 格 ≥parity **仅 kernel-axis 覆盖面**·NOT e2e·系统账仍 [GAP-FLAT-E2E] 黄格·**禁混算 perf-covered 9/83·禁写「加速 N kernel」于 e2e 语境**。
- **污染**：我方 kernel 侧 relIQR <1% 全 5 格·q5 ratio 抖动隔离为对手侧方差·disjoint-pin clean·vLLM 未触碰·.so md5 双证 restore。

## durable files
- `evidence.md` — 主发现·per-cell 表·三账本·计数建议·污染/restore 纪律。
- `raw/raw_gcc_O3.txt` — kernel-sym 主账 board stdout（60 FLATGEMM + header/vsetvl/gate）。
- `raw/raw_gcc_O2.txt` — gcc-15 -O2 稳健复现。
- `raw/raw_clang_O3.txt` — clang-18 -O3 deploy 双账本。
- `raw/objdump_symmetry.txt` — OURS(gcc-15) vs OPP(gcc-15 .so) rvv-insn 对称双证。
- `raw/exported_kernel_md5.txt` — 5 exported .kernel.c md5（provenance = pre-rename 49057e34·语义==current HEAD）。
- `raw/analyze.py` `raw/analyze2.py` — median/relIQR + ours/opp/ratio relIQR 分离脚本（regenerable）。

## data-only cell
证据+recipe only。exported .kernel.c 确定性可再生（method §1）。board scratch `/tmp/flat_export` 测后已删。deployment tree/.so 未改（md5 双证）。**Nothing git-added.**
