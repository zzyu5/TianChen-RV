# 线A · A1 — rvv 部署账统一 clang-18 全量重测（真-对称域·对手 ggml clang-18 重编）

> **裁决基础**：用户直接裁定（2026-07-16·本会话内）——「**用 clang 就用 clang-18，全部都用**」。即我方 kernel 与 ggml 对手**全家同链 clang-18 重编**（真-对称域·非 batch6/§2.3 的「ours-clang vs opp-gcc-stock」footnote 域）。
> **纪律**：cold 唯一·N≥24 中位+2-seed·load-gate 单核 idle≥70% 单实例·byte-exact ZERO-MODEL 正确门·0 造数·禁 git（主会审后 commit）·stock .so 只读。
> **测于**：2026-07-16 · rvv（VLEN128·core8-15 load-gated·clang-18.1.8 `/opt/tcrv-toolchains`）。

---

## 0. 关键前置：clang-18 全量 ggml 对手重编（NEW·本役地基）

板上此前**无** clang-18 全量 ggml 构建（仅 clang-17 `build-openeuler-clang17`·gcc-15 `build-gcc15-rv64gcv`）。本役新建：

- **构建**：`/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv`·git pin `f3e18281`（**与 gcc-stock 同 commit**·apples-to-apples）。
- **工具链**：clang-18.1.8（`/opt/tcrv-toolchains/llvm-18.1.8`）·同 march `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_...`·`--gcc-toolchain=gcc-15.2.0`。
- **两 blocker 解**（复现 ROADMAP 一.三 B1/B2）：① `-L$GTOOL/lib -Wl,-rpath` 解 `libgcc_s` 缺失（lld 不搜 gcc lib 路径）② `-fno-integrated-as` 解 `quants.c` 内联 vsetvli（clang 集成汇编器拒收·走 GNU as）。
- **产物**：`libggml-cpu.so` md5 `e85fceda…` · `libggml-base.so` md5 `d9c07980…`（`.comment` = clang 18.1.8·GCC 15.2.0 = -fno-integrated-as 的 GNU as+libgcc·正常）·build exit=0·0 error。
- **对手 codegen 探针（clang-18 vs 部署 gcc-15）**：`dequantize_row_*` 多数被 clang-18 **也自动向量化**（q4_0 rvv=232·q8_0 rvv=268·仅 q4_K rvv=0 scalar）→ DEQ 仍是**公平 vec-vs-vec**（非 k1-Bianbu-scalar 那种「编胖对手」假象）。**但 clang-18 autovec 明显弱于 gcc autovec**（见 §2 GB/s）。

---

## 1. iq/tq/fp4 GEMM rvv 7 格 — 真-对称 clang-18（ours-clang-18 vs opp-clang-18-generic）

harness = A2-batch6 driver·同 fixture（K=2048 nr=16 nc=512 reps=25 2-seed）·GATE ZERO-MODEL nbad=0/8192·GGML 指向 clang-18 build。

| 格 | 旧值·gcc-deploy-MAIN（stale·具名-X gcc-death） | 旧值·clang-footnote（opp=gcc-stock·§2.3） | **新·真-对称 clang-18（opp=clang-18）s1/s2** | 判定 | 归因 |
|---|---|---|---|---|---|
| iq4_xs | 0.50（named-X） | 3.85/3.95 | **3.15 / 3.11** | PASS | 便宜档·opp clang-vec 29ms |
| iq2_xxs | 0.51（named-X） | 3.67/3.69 | **2.32 / 2.34** | PASS | 便宜档 |
| iq2_xs | 0.38（named-X·输 gcc 纯标量） | 2.58/2.57 | **12.32 / 12.29** | PASS | ★opp clang 编巨胖 160ms（gcc-stock 只 33ms）→ 大倍数=opp-clang-bloat 假象 |
| iq2_s | 0.37（named-X·输 gcc 纯标量） | 2.55/2.56 | **11.49 / 11.46** | PASS | ★同上·opp clang 150ms |
| mxfp4 | 0.75（named-X） | 2.63/2.63 | **3.65 / 3.66** | PASS | 便宜档 |
| tq1_0 | 0.35（named-X） | 3.80/3.96 | **7.90 / 7.93** | PASS | 便宜档·opp clang 24ms |
| tq2_0 | 0.45（named-X） | 4.84/4.89 | **9.91 / 10.10** | PASS | 便宜档·ours vsetvl=8 最简 |

**tally：7/7 PASS**（真-对称 clang-18·全 byte-exact·2-seed <2% 稳）。stock 库只读（cpu md5 `d1adc634` 未参与·本役用 clang-18 build）。STRAY=0·core8 idle=100%。

**★成色诚实（铁线·禁称硬赢）**：
- 对手 = ggml **自带 generic 标量参考**（`_generic`·便宜档·非 as-shipped 手调 block-dot）→ **本就不入 matmul 0.8 headline 分母**（scalar-ref·[NG-4]）。
- iq2_xs/iq2_s 的 11-12× **半是 opp-clang-bloat 假象**（clang-18 把 generic 编成 150-160ms·gcc-stock 只 33ms）·非我方核硬赢。
- **单一诚实数 ≈ 2.3-3.7×**（iq4_xs/iq2_xxs/mxfp4·opp clang-vec 紧凑时的真 repack-over-scalar 优势·便宜档）。
- **0 verified hand-brick**。这是 **[CASE-COMPILER-ASYMMETRY] 家族**：同一格三域 {gcc-death 0.35-0.75 / opp-gcc-stock 2.5-4.8 / opp-clang-18 2.3-12.3} 排开 = ratio 由**双侧编译器 codegen**主导。

---

## 2. dequant FLAT-5 rvv 5 格 — 真-对称 clang-18（ours-clang-18 vs opp-clang-18 `dequantize_row_*`）

harness = A2-batch3 dequant_flat5 driver·streaming K=1048576（out 4MiB>L2·224MiB flush）·N=24 median+2-seed·GATE byte-exact ZERO-MODEL 全 0mism/0ULP。

| 格 | 旧值·gcc-deploy-MAIN（stale） | **新·真-对称 clang-18 s1/s2** | ours GB/s | opp GB/s | 判定 | 归因（诚实） |
|---|---|---|---|---|---|---|
| q4_0 | **具名-X 0.70**（opp gcc-autovec 5.5 GB/s 强） | **2.51 / 2.90** | 4.03/4.64 | 1.60 | **PASS**（翻转） | ★opp clang-vec 只 1.60 GB/s（gcc 5.5）→ 翻转=opp-clang-under-vec 伪影·非部署赢 |
| q4_1 | PASS(WIN) 1.18-1.39 | **2.59 / 2.63** | 3.70/3.74 | 1.43 | PASS | 同上·opp clang 弱 autovec |
| q5_0 | **具名-X 0.62** | **1.007 / 1.007** | 0.60 | 0.60 | **PASS-marginal**（翻转） | ★真 near-parity·双方 memory-bound 0.60 GB/s（qh 5-bit scatter·DRAM 墙·编译器中性） |
| q5_1 | **具名-X 0.70-0.74** | **1.02 / 1.02** | 0.61 | 0.60 | **PASS-marginal**（翻转） | 同 q5_0·真 parity |
| q8_0 | PASS-marg 0.84 | **2.26 / 2.26** | 3.54 | 1.57 | PASS | opp clang-vec 弱（1.57 vs gcc 更强） |

**tally：5/5 PASS**（真-对称 clang-18·全 byte-exact·load-gate OK）。base.so md5 `d9c07980` before==after。STRAY=0。

**★成色诚实（铁线）**：
- **q4_0/q4_1/q8_0 的 PASS（含 3 个 gcc 域 具名-X→PASS 翻转）不是我方核变强**——是 **clang-18 把对手 `dequantize_row_*` 编得比部署 gcc 慢**（opp 1.6 GB/s clang vs 5.5 GB/s gcc·~3.4× under-vec）。**部署现实 = gcc-15 对手（5.5 GB/s·ours 输）**·clang-18 对称 PASS ≠ 部署赢。有趣：clang 甚至把 ours q4_0/q4_1/q8_0 dequant 编成 **scalar**（vsetvl=0）·仍赢过 clang 的胖 opp-vec。
- **q5_0/q5_1 的 near-parity（1.007/1.02）是唯一编译器中性的真结果**——DRAM-bound·双方 0.60 GB/s·genuine parity（从 gcc 域 0.62/0.70 X 变 parity 也主要因 gcc-opp 在 q5x 上 autovec 强、clang-opp 弱化了）。

---

## 3. 未纳入本役 clang-18 重测的 rvv 部署账（诚实登记·非本役 scope 或已对称）

- **matmul kernel-sym（0.8 分母·9+格）**：**已 clang-18 对称**（§五 opponent re-parse·对手 TU quants.c/ggml-quants.c/vec.cpp/ops.cpp 全 clang-18 重编·pin `e36a602b`）。非本役新测（已committed）。
- **K-quant GEMM S6 q3_K/q6_K**：§六.3 attack#1 已 G2 cold clang-18（q6_K 0.96 unrolled / q3_K 1.40 rolled）·ROADMAP 标「gcc-death 部署未测·strip-outer 消 spill 编译器无关预计亦改善·待复测」= 可后续同域重测项（本役未覆盖·如实登记）。
- **DEQ K-quant/iq 15 真向量候选**（reparse §③·q2_K…tq2_0）：本役只重测 FLAT-5（有独立可跑 harness）·K/iq 候选用 G7 census gcc 域数据·**未 clang-18 重测**（如实登记·后续项·harness = census dequant-rvv）。

---

## 4. 部署树 clang-18 一致性

- **我方 L3 = clang（设计事实·系统身份·执行总纲v2 §7）**→ 部署核 by construction clang-compiled。
- **A5 已验 deployed==proven bit-identical**（`A5-e2e-deploy-sync.md`）：git 源控不变量（emitter `d109d6ed2..HEAD` 零漂移）+ HEAD 重生 emitc byte-identical vs 板部署核（q5_0/q5_1 GEVM md5 相等·REDESIGN-B 指纹一致）。
- **本役未另做 rvv e2e .so 全量 clang-18 重建**（A5 覆盖 emitter 输出层·编译器身份=clang by design）·如实登记 scope。

---

## 5. 板卫生 + gcc 归档指针

- **stock 库只读全程**：gcc-15 `libggml-cpu.so` md5 `d1adc634`·`libggml-base.so` `1b4580c4`·before==after 未动。
- **新增板产物（evidence·非 stock）**：`build-clang18-rv64gcv`（clang-18 对手·保留作 clang-18 域基线）·`/tmp/g8_a2b6_rvv_sym_*`·`/tmp/g8_a2b3_dq_c18sym`（scratch·可清）。
- **gcc 存量归档**：gcc-deploy-MAIN 数字（iq/tq gemm 7/7 具名-X gcc-death·DEQ q4_0/q5_0/q5_1 具名-X）**保留不删**·归 [CASE-COMPILER-ASYMMETRY] 素材·指针 = `A2-batch6-iqtq-gemm-scalar.md §2.2`·`A2-batch3-deq-quant-preduce.md §2.1`·`deq-axis-reparse/evidence.md`·`T-VALIDITY_compiler_symmetry_ledger.md`。
- **禁 git**（主会审后提交）。

---

## 6. ★政策张力（登记 PENDING_RULINGS·见 PR-17）

用户裁「clang-18 全部都用」→ 真-对称 clang-18 域下 iq/tq gemm 7/7 PASS·DEQ 5/5 PASS。但 committed canon（PR-9/11/12·[CASE-COMPILER-ASYMMETRY] 双账本）以 **gcc-deploy = MAIN、clang = footnote** 为 DEQ/scalar-ref 域政策·理由 = Amdahl 同域律（部署对手是 gcc·clang-micro 喂部署 = garbage-in）。本役数据**实证了该张力**：翻转的 PASS 多半是 opp-clang-codegen 弱化伪影（DEQ opp 1.6 vs 5.5 GB/s·iq2_xs opp 160 vs 33ms）·非部署赢。→ 登记 PR-17：真-对称 clang-18 域「主表升格」vs 保留 gcc-deploy footnote 的诚实成色标注·须用户裁头条口径。保守默认 = 双域并记（不删 gcc·不擅改 headline 分母）。
