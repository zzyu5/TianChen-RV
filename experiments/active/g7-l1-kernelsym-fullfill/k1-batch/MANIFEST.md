# MANIFEST — g7-l1-kernelsym-fullfill / k1-batch

- **campaign**: G7 §L1 货架A 全量 micro（kernel-sym 表填满 hot/cold 双态）· **k1-半**（VLEN256·clang-18 出货对称域·pin core3）
- **status**: ACTIVE · board-measured 2026-07-14 · board=k1 (SpacemiT X60) · **NOT committed**（用户提交）
- **role**: 二事 —
  1. **★B G 决胜局**：q4_K@k1 **sealed vl=16 核**（`s6_q4K_vl16_sealed.c` md5 e437fd3b·vwmacc 1120）vs 真 16x1 hand-brick `ggml_gemm_q4_K_16x1_q8_K` **kernel-axis micro 直测**（e2e 早证 1.085×·kernel-axis 从未测）。裁定 sealed 是**双轴** or **e2e-only** hand-brick win。
  2. **A K-quant@k1 kernel-sym**：q2_K/q3_K/q6_K @k1/VLEN256 hot/cold 同形状 GEMM micro（补 T9 §1.1 现有 q4_K/q5_K/q4_0/q8_0@k1 之外的）·变体穷举（nr-shape）·对手符号机判。
- **[NG-4] DISCIPLINE**: kernel-axis micro datapoint·**NOT** e2e beat·**NOT** sealed 8-gate Win·**与 perf-covered 9/83 永不混算**。

## 结论速览
- **★G 决胜局裁决**：**双轴 verified hand-brick win**。sealed vl=16 核 vs 真 16x1 = **~1.197× WIN**（kernel-axis micro·N=12·HOT∧COLD∧O2∧O3∧3seed·全 12 > parity）+ e2e 1.085× byte-exact → 两轴皆赢真 hand-brick。**首个双轴 verified hand-brick win**。投影 ~1.2× 兑现。
- **A q2/q3/q6@k1**：见 evidence.md §A（板测回填）。

## board identity（cross-machine INCOMPARABLE）
`k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / 8-core / Bianbu clang 18.1.8 / glibc / gov=performance 1.6GHz / core3`
- ours march = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`（canonical -O2）· 对手 exact march = `rv64gcv_zfh_zvfh_zicbop_zihintpause`（-O3）· 两 recipe 皆测（G 决胜局）。
- opponent lib = `/data/k1build-stock/bin/libggml-cpu.so`（upstream ggml·factory-as-shipped·clang++-18 -O3 compile_commands.json 证）→ ours(clang-18) vs stock(clang-18) = **对称·无 [CASE-COMPILER-ASYMMETRY]**。
- confound-clean: march zfh/zvfh → fp16 native fcvt.s.h·**0 `__extendhfsf2` libcall**（ours .o libcall-free 板验）。

## 文件
- `evidence.md` — 主证据（★B G 决胜局裁决 + A q2/q3/q6 hot/cold + 穷举 + 对手符号机判 + 折中态）
- `raw/gdecisive_vl16_vs_16x1.log` — G 决胜局全 raw（preflight/seal/correctness/HOT/COLD·12 rounds×3seed×2recipe）
- `raw/kq236_hotcold.log` — A q2/q3/q6 hot/cold 全 raw（回填）
- `kquant_gemm_hotcold_q236_driver.c` — A 通用 hot/cold driver（q2/q3/q6·pool-cold·block-dot 对手·基于 g7-l2 kquant_gemm_cold_driver.c + q6q2q3 strides）
- `run_k1_kq236_hotcold.sh` — A 板 runner
- G 决胜局复用 `experiments/active/g7-l2-kernelsym-hotcold/q4K-handbrick-resolve/q4k_handbrick_driver.c`（换 ours .o 为 sealed vl=16 核）+ `run_k1_q4k_vl16_hb.sh`（job scratch）

## provenance（REUSED cached emit-C·NO local tcrv-opt regen）
- sealed vl=16: `s6_q4K_vl16_sealed.c` md5 `e437fd3b`（== `/tmp/vlen_adapt_m1/`·sealed Win-K1-VLEN 核·vwmacc 1120·板 md5 双证）
- q2_K GEMM: `repack_gemm_q2_K_q8_K.kernel.c` md5 `6c9322f6`（kq_export_q6q2q3·PLAIN campaign export·symbol `tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_...`）
- q3_K GEMM: md5 `f4cfb641`（PLAIN·S6-NULL）· q6_K GEMM: md5 `0f14791e`（PLAIN·S6-NULL）
- 板 scratch: `k1:/tmp/q4k_vl16_hb/` + `k1:/tmp/kq236_k1/`（ephemeral）；主树 + local build **UNTOUCHED**；governor left as-found（perf/1.6GHz）；stock lib read-only。**零 restore**（纯 micro）。
