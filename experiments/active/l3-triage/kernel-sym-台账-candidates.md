# L3 案头 — kernel-sym 台账立账 candidate 识别（任务 B · 三.2）

> **性质**：纯案头 read-only·**禁改 schema/T8/ROADMAP**·产 candidate 清单给主会话。
> **台账定义**：kernel 账 = **第二赛道**（与 perf-covered 系统账**永不混算**）。T3 派生·第二常驻计数 = **「kernel-sym ≥parity 格数」**。
> **candidate 判据**：{我方 emitted kernel 在 ∧ 对手 kernel 真实存在(非 absent) ∧ 对称编译可行(同编译器/flags/march)}。
> **对手类隔离**：**非 as-shipped 或 internal-A/B(对手 SELF) 的行 → 单列·不入计数**。
> **测量协议（板批缝隙补·主会话执行）**：micro A/B·同编译器/flags/march·N≥10·T-N·对手类必标。

---

## 0. 净结论（candidate 计数）

| 桶 | 格数 | 说明 |
|---|---:|---|
| **kernel-sym candidate（入计数池）** | **15 格-板点** | 有 real opponent kernel + 对称编译可行 |
| 其中 **≥parity（已测·计入第二常驻计数）** | **4** | q4_K@k1(3.106×)·q5_K@k1(1.916×)·q4_0@k1-gemm(prefill parity 1.0022×)·q8_0@k1(item4 +4.37%) |
| 其中 **<parity（对称重测 LOSS·candidate 但不计 ≥parity）** | **9** | q5_K@rvv(0.120×)·q2_K@rvv(0.386×)·q3_K@rvv(~0.18×)·q6_K@rvv(~0.18×)·iq4_nl@rvv(0.217×)·iq2/iq3/iq4_xs/tq batch2c 8 格(0.16–0.78×) |
| 其中 **≥parity 待补测（板批缝隙·主会话）** | **~5** | FLAT 5 gemm@rvv gcc-symmetric（ROADMAP 称 "≥parity 全幸存"·但逐格对称 kernel-axis micro 数部分缺·须板批补） |
| **对手类单列·不入计数** | **3+N** | IME 3 格（对手 SELF·internal-A/B）+ 一切 CASE-COMPILER-ASYMMETRY 行（clang-ours-vs-gcc-shipped·已撤回） |
| **opponent-absent·非 candidate** | **~9** | gemm iq/tq 7 格（gemm-轴对手 absent）+ q1_0/nvfp4（疑无 ggml 对位·待主会核） |

**★第二常驻计数 「kernel-sym ≥parity 格数」当前硬值 = 4**（q4_K@k1 / q5_K@k1 / q4_0@k1-gemm-prefill / q8_0@k1）·**FLAT 5 gemm@rvv 待板批对称 micro 补测后可能 +5**。

---

## 1. kernel-sym candidate 清单（逐格·四字段）

字段：{emitted kernel 指针 · 对手 kernel 身份 · 对称编译可行否 · 已有 micro A/B 数据否(引 T8)}

### 1.1 ≥parity 已测（计入第二常驻计数 · 4）

| 格·板 | emitted kernel 指针 | 对手 kernel 身份 | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| **q4_K @k1/VLEN256** | 前门 lowerToRepackGemm KQuant（typed_repack·S6-tiled·spill→0 v30≤32 cliff）·experiments/active/kquant-k1-vlen256-kernel-axis-t4a | **真出货 hand-brick**（k1 stock RVV q4_K 16x1 repack·case256 fires·NOT block-dot·**唯一强对手**） | ✓ **clang-18 双侧对称**（k1 stock=clang-18） | ✓ **3.106× ≥parity**（T8 row74·ALL 12 rounds>1·NOT-e2e·kernel-axis） |
| **q5_K @k1/VLEN256** | 前门 KQuant（qh 5th-bit leaf·min-fold·S6 HOLDS·XFER-1 #2） | factory block-dot（generic·k1 ships zero q5_K repack·通用路径对手·弱） | ✓ clang-18 对称 | ✓ **1.916× ≥parity**（T8 row75·ALL 12 rounds>1·NOT-e2e） |
| **q4_0 @k1/VLEN256 (gemm prefill)** | repack GEMM（mf2 fractional·columnsPerPass 4） | e2e-factory block-dot（VLEN-flip prefill control） | ✓ freq-locked paired 同源 | ✓ **prefill 1.0022× PARITY**（T8 row7·CI[1.0014,1.0030]·compute-bound·parity=result） |
| **q8_0 @k1/VLEN256** | repack GEVM wide-hl16 mf2 one-strip·item4 fcvt-reschedule | factory-block-dot | ✓ clang-18 对称·preflight 4/4 dual-board | ✓ **+4.37%（item4·T8 row4）** / mf2-wide vs m1 +11%（T8 row20·do-not-widen VINDICATED） |

### 1.2 <parity 已测（candidate·对称重测 LOSS·不计 ≥parity · 9）

| 格·板 | emitted kernel 指针 | 对手 kernel 身份 | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| q5_K @rvv/VLEN128 | 前门 KQuant repack GEMM | factory generic block-dot（untuned·real·as-shipped） | ✓ **gcc-15.2 双侧对称** | **0.120× LOSS**（~12.9× reversal·T8 row61·[CASE-COMPILER-ASYMMETRY] Stage1 蒸发·symmetric-gcc） |
| q2_K @rvv/VLEN128 | 前门 KQuant repack（dual d/dmin+bsums-min fold·S6-tiled） | factory hand-tuned _vl128 block-dot（real） | ✓ gcc-15 对称 | **0.386× LOSS**（T8 分类报告§4·对称-gcc·row63 throughput 0.20×） |
| q3_K @rvv/VLEN128 | 前门 KQuant（3-bit subtractive-hmask·PLAIN/UNTILED·S6 NULL） | factory hand-tuned _vl128 block-dot（real·但零 repack anywhere） | ✓ gcc-15 对称 | **~0.176–0.20× LOSS**（T8 row64/row70·weight-bound·S6 NULL） |
| q6_K @rvv/VLEN128 | 前门 KQuant（6-bit dual-plane·PLAIN·S6 NULL） | factory mature block-dot（real） | ✓ gcc-15 对称 | **~0.18–0.19× LOSS**（T8 row62/row68·2995 vsetivli 无调度·weight-recon floor） |
| iq4_nl @rvv/VLEN128 (gemm) | 前门 Codebook repack GEMM（kvalues_iq4nl·vl=8） | factory block-dot（real·routing 对位·cross-op） | ✓ **rv64gcv shipped=gcc-15.2 对称**（kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered） | **prefill 0.217× / decode 0.505× LOSS**（T8 iq4_nl·八门全过·clean window·g5-wiring M2-iq4_nl） |
| iq3_xxs @rvv (vec_dot) | 我方 block-dot emit | factory SIMD-dispatch block-dot（real·gcc-15.2） | ✓ **batch2c gcc-15.2 -O3 双侧对称** | **0.158× LOSS**（WORST·T8 row12·fraclmul-2elem scalarization+vsetvli storm） |
| iq3_s @rvv (vec_dot) | 同上 | factory block-dot | ✓ batch2c 对称 | **0.280× LOSS**（T8 row8） |
| iq2_xs @rvv (vec_dot) | 同上 | factory block-dot | ✓ batch2c 对称 | **0.329× LOSS**（T8 row10）〔GAP-SB POST 1.509× vs-generic·但 vs-SIMD 仍 LOSS〕 |
| { iq2_s / iq2_xxs / iq4_xs / tq1_0 / tq2_0 } @rvv (vec_dot) | 我方 block-dot emit | factory SIMD-dispatch block-dot（real·gcc-15.2） | ✓ batch2c 对称 | iq2_s 0.477× / iq2_xxs 0.780×(MILDEST) / iq4_xs 0.718× / tq1_0 0.550× / tq2_0 0.231×（T8 row9/11/13/15/14）·tq2_0 spill-fix→仍 0.45× vs-SIMD（row21） |

### 1.3 ≥parity 待板批对称 micro 补测（FLAT 5 gemm@rvv · 主会话缝隙）

| 格·板 | emitted kernel 指针 | 对手 kernel 身份 | 对称编译 | micro A/B 状态 |
|---|---|---|---|---|
| q4_0 / q4_1 / q5_0 / q5_1 / q8_0 @rvv/VLEN128 (gemm) | 净新 repack scaffold（q5_0 make_block_q5_0x16 / q5_1 block_q8_1x4 / q4_1 q8_1 家族 / q8_0 .inc=M1b） | factory block-dot（as-shipped·q4_0 上游 repack VLEN128-gated / q4_1/q5_0/q5_1 无 stock repack→generic / q8_0 上游 VLEN128 破损） | ✓ **gcc-15 出货对称**（rvv shipped=gcc-15·kernel-axis==system-axis） | **ROADMAP 称 "gcc-symmetric ≥parity 全幸存"（[GAP-FLAT-E2E]）·但逐格独立对称 kernel-axis micro 数部分缺**（q8_0 有 item4/fill·q4_0 有 prefill parity·q4_1/q5_0/q5_1 待补）→ **板批缝隙补 N≥10 对称 micro** |

---

## 2. 对手类单列（不入计数 · 判据 = 非 as-shipped 或 internal-A/B）

| 格 | 对手类 | 为何不入 kernel-sym 计数 | 指针 |
|---|---|---|---|
| q4_0 @ime | **对手 SELF**（internal-A/B·我方非-IME 参考） | 无独立 factory IME kernel 对位·compute-account 2.09× = vs SELF·非 kernel-vs-kernel | T8 row206/218/220 |
| q8_0 @ime | 对手 SELF | 同上 | 同上 |
| q4_K @ime | 对手 SELF | 同上·G6-A 桥 exit-b（0.5529× stock·仍<parity·厂商 VEN/OFF 1.997× 是 vendor 非 our-kernel 对位） | T8 row220 |
| q5_K@rvv "1.5×" / q2_K@rvv "1.413×" / q4_K@rvv "1.884×" 等 kernel-axis "赢" | **CASE-COMPILER-ASYMMETRY**（clang-ours -O2 vs gcc-shipped generic） | 非对称编译·已撤回（对称-gcc 重测蒸发 0.12–0.39×）·kernel-axis account WITHDRAWN | T8 row61；memory perf-constitution |

> ★纪律：IME 三格若未来建独立 vendor-IME-kernel-vs-our-IME-kernel 对称对位，可从"对手 SELF 单列"升为 candidate；现状 = SELF·单列不计数。

---

## 3. opponent-absent（非 candidate · gemm-轴无对手 kernel）

| 格 | 缺口 | 说明 |
|---|---|---|
| gemm/iq2_xxs·iq2_xs·iq2_s·iq4_xs·mxfp4·tq1_0·tq2_0（7） | **gemm-轴对手 absent**（opponent 零 iq/tq repack GEMM·只有 block-dot vec_dot·跨-op） | 对位退化为 our-gemm vs opponent-block-dot（cross-op·如 iq4_nl 已做）·**非 same-op kernel-sym**·主会话若采 cross-op 口径可纳入（但须明标 cross-op·区别 same-op kernel-sym） |
| vec_dot/q1_0 | 疑 no-fair-opponent（Weft-internal binary·非标准 ggml） | 待主会核 ggml 是否有 q1_0 vec_dot |
| vec_dot/nvfp4 | 疑 no-fair-opponent（fp4·ggml 疑无 nvfp4） | 待主会核 |

---

## 4. 台账立账建议（给主会话）

1. **第二常驻计数「kernel-sym ≥parity 格数」= 4**（q4_K@k1 · q5_K@k1 · q4_0@k1-gemm-prefill-parity · q8_0@k1）·**FLAT 5 gemm@rvv 补测后潜在 +5 → 上限 9**。
2. **成色分层**：q4_K@k1 = 唯一 **强对手（hand-brick）≥parity**（其余 ≥parity 对手皆 block-dot/parity-control 弱对手或 memory-bound）→台账须标对手成色（hand-brick vs block-dot vs parity-control）。
3. **板×编译器锁**：k1=clang-18 对称域 / rvv=gcc-15 对称域（board shipped-compiler·[CASE-COMPILER-ASYMMETRY] 判别键）·跨板不可比·逐点标 board identity。
4. **永不混算**：kernel-sym 台账 ≠ perf-covered 系统账（7/83）·**禁表述为系统收益·禁写"加速了 N 个 kernel"于 e2e 语境**（措辞纪律·令六 lint）。kernel-sym 是内核轴覆盖面（T3 行·C3′ 证据）·非 e2e 拉绿杠杆。
5. **板批缝隙补测队列（主会话·非本案头执行）**：FLAT 5 gemm@rvv 逐格对称 kernel-axis micro（N≥10·T-N·gcc-15 双侧·对手=as-shipped block-dot）→ 补齐 ≥parity 待测 5 格。

**需主会核**：① cross-op 口径（gemm-iq/tq vs block-dot）是否纳入 kernel-sym 台账（须明标 cross-op）；② q1_0/nvfp4 ggml 对手存在性；③ IME"对手 SELF"是否未来升 candidate（需独立 vendor-kernel 对位）。
