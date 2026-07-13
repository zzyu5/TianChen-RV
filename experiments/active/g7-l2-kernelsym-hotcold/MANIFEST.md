# cell MANIFEST — g7-l2-kernelsym-hotcold

- **campaign**: G7 L2 货架A · **task**: kernel-sym hot/cold 双态首批（k1 存量 4 格补 cold 行）
- **status**: ACTIVE · board-measured 2026-07-13 · board=k1 (SpacemiT X60 / VLEN256 / core3 / clang-18.1.8) · **NOT committed** (user commits)
- **role**: 为 T9 kernel-sym §1.1 的 4 个 k1 存量格（q4_K / q5_K / q4_0-gemm / q8_0-gevm）补 **cold micro A/B**（cache-cold·WS>>LLC 512KiB pool）·逐格 hot/cold 对照 + e2e-decode 预测器验证（立柱二·货架A hot/cold 双态）。
- **[NG-4] DISCIPLINE**: 第二赛道 **kernel-axis** cold datapoint·**NOT** e2e beat·**NOT** sealed 8-gate Win·**与 perf-covered 系统账（9/83）永不混算**·**禁**表述"加速 N kernel"于 e2e 语境。opponent = board factory `libggml-cpu.so` block-dot（gcc-15 as-shipped·ours=clang-18 对称域）。

## board identity（cross-machine INCOMPARABLE）
- `k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / core3 (L2 512KiB shared 0-3·NO L3) / Bianbu clang 18.1.8 / gov=performance 1.6GHz`
- march = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -O2`·libcall-free（zfh/zvfh native fp16）已验。
- opponent lib = `/data/k1build-stock/bin/libggml-cpu.so`（factory gcc-15·只读链接·md5 未变）。

## 核心结论（详见 evidence.md）
- **hot ≈ cold（全 4 格）**：q4_K 3.19→3.18 · q5_K 1.92→1.92 · q4_0 3.51→3.50 · q8_0 1.48→1.48（cache-cold 不改 kernel-axis 比值）= hot/cold NULL 区分（k1 L2<tile + compute-bound）。
- **cold ≥parity（全 4 格·全 shape）**：min = q5_K nr4 1.79×。
- **cold 作 e2e-decode 预测器 = 条件有效（3/4 命中·q5_K 证伪）**：q8_0 GEVM 机制性命中（同 decode kernel·memory-leaning 1.7GB/s → e2e decode 1.21× ✓）；**q5_K 决定性 MISMATCH**（micro compute-bound 1.79× WIN ↔ e2e memory-bound GEVM 0.729× LOSS）——K-quant repack-GEMM micro 从不触及 memory wall（0.01–0.5 GB/s）·不能当 decode 预测器。

## durable files
- `evidence.md` — 全结果 + cold 协议口径 + roofline 分类 + e2e 预测器立柱二裁决 + T9 cold 列建议 + 污染/restore。
- `MANIFEST.md` — 本文件。
- `summary_hotcold.csv` — 7 行 per-shape 汇总（fmt/opponent/regime/hot/cold/GB-s/predicts）。
- `kquant_gemm_cold_driver.c` — q4_K/q5_K cold-pool paired driver（POOL>>LLC·median-of-N·派生自 kquant_gemm_paired_driver.c）。
- `gevm_q8_cold_driver.c` — q8_0 GEVM hot+cold paired driver（vs vec_dot_q8_0_q8_0 block-dot）。
- `gemm_q4_0_cold_driver.c` — q4_0 GEMM hot+cold paired driver（vs vec_dot_q4_0_q8_0 block-dot）。
- `run_k1_kquant_cold.sh` — 板 runner（q4_K/q5_K·hot best-of-5 + cold nr64/nr4·load-gate·core3·pin）。
- `run_k1_q80_q40_cold.sh` — 板 runner（q8_0 GEVM + q4_0 GEMM·hot/cold）。
- `raw/q4k_q5k_hotcold.log` — q4_K/q5_K 板全 log（preflight/load-gate/md5/hot×3/cold nr64/cold nr4）。
- `raw/q80_q40_hotcold.log` — q8_0/q4_0 板全 log（hot/cold·achieved GB/s）。

## board scratch / restore
- board scratch: `k1:/tmp/g7_l2_hotcold/`（ephemeral）· **主树 + build/ + stock ggml .so + governor 全 UNTOUCHED** = LIVE 零改动·**无需 restore**（纯 micro·符合任务口径）。
