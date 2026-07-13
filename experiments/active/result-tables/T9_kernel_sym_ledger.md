# T9 — kernel-sym 台账（第二赛道·T3 派生）

> **建成**：2026-07-13 L3-apply（casefile `experiments/active/l3-triage/kernel-sym-台账-candidates.md` 派生）。
> **来源基**：T3_A/T3_B（board micro A/B）+ T8_winloss_gap_ledger.csv（对称重测账）。

---

## 0. 台账定义头

- **性质**：**第二赛道** kernel 内核轴覆盖面台账（C3′ 证据·T3 派生）。**与 perf-covered 系统账（`schema/perf-covered-category.v1.json` · 7/83）永不混算**。
- **第二常驻计数**：**「kernel-sym ≥parity 格数」**（当前硬值见 §1.1）。
- **candidate 判据**：`{我方 emitted kernel 在 ∧ 对手 kernel 真实存在(非 absent) ∧ 对称编译可行(同编译器/flags/march)}`。三者缺一 → 非 candidate（列 §3 opponent-absent 或 §2 对手类单列）。
- **测量协议**：micro A/B·同编译器/flags/march·N≥10·T-N（noise floor）·对手类必标。
- **板×编译器锁**：k1=clang-18 对称域 / rvv=gcc-15 对称域（board shipped-compiler·[CASE-COMPILER-ASYMMETRY] 判别键）·**跨板不可比**·逐点标 board identity。

### ★铁律（入台账头·令六 lint）

1. **kernel 账不得表述为系统收益**。**禁写「加速了 N 个 kernel」于 e2e 语境**。kernel-sym = 内核轴覆盖面（T3 行·C3′ 证据），**非** e2e 拉绿杠杆。
2. **对手类 SELF / internal-A/B / [CASE-COMPILER-ASYMMETRY] 行单列·不入计数**（§2）。判据 = 非 as-shipped 对手 或 对手=我方参考实现。
3. **gemm iq/tq 7 格 + q1_0 / nvfp4 = opponent-absent·非 candidate**（§3）。gemm-轴无 same-op 对手 kernel；cross-op 口径（our-gemm vs opponent-block-dot）须明标 cross-op·区别 same-op kernel-sym（待主会核是否纳入）。
4. **成色分层**：对手 = 手调 hand-brick（如 q4_K@k1·case256 真出货 repack）= 最强对手成色；对手 = 通用 block-dot / parity-control / memory-bound = 弱对手成色。台账逐行标对手成色。

---

## 1. kernel-sym candidate 清单（逐格·四字段）

字段：`{emitted kernel 指针 · 对手 kernel 身份 · 对称编译可行否 · 已有 micro A/B 数据(引 T8)}`

### 1.1 ≥parity 已测 — 计入第二常驻计数 ★ **kernel-sym ≥parity 格数 = 4**

| 格·板 | emitted kernel 指针 | 对手 kernel 身份（成色） | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| **q4_K @k1/VLEN256** | 前门 lowerToRepackGemm KQuant（typed_repack·S6-tiled·spill→0·v30≤32 cliff）·`experiments/active/kquant-k1-vlen256-kernel-axis-t4a` | **真出货 hand-brick**（k1 stock RVV q4_K 16x1 repack·case256 fires·NOT block-dot·**唯一强对手·成色最硬**） | ✓ clang-18 双侧对称（k1 stock=clang-18） | ✓ **3.106× ≥parity**（T8·ALL 12 rounds>1·NOT-e2e·kernel-axis） |
| **q5_K @k1/VLEN256** | 前门 KQuant（qh 5th-bit leaf·min-fold·S6 HOLDS·XFER-1 #2） | factory block-dot（generic·k1 ships zero q5_K repack·**通用路径·弱对手**） | ✓ clang-18 对称 | ✓ **1.916× ≥parity**（T8·ALL 12 rounds>1·NOT-e2e） |
| **q4_0 @k1/VLEN256 (gemm prefill)** | repack GEMM（mf2 fractional·columnsPerPass 4） | e2e-factory block-dot（VLEN-flip prefill control·**parity-control**） | ✓ freq-locked paired 同源 | ✓ **prefill 1.0022× PARITY**（T8 row `q4_0-gevm-k1-vlen256...` 伴随行·CI[1.0014,1.0030]·compute-bound·parity=result） |
| **q8_0 @k1/VLEN256** | repack GEVM wide-hl16 mf2 one-strip·item4 fcvt-reschedule | factory-block-dot（**弱对手·memory-bound**） | ✓ clang-18 对称·preflight 4/4 dual-board | ✓ **+4.37%（item4）** / mf2-wide vs m1 +11%（T8 `q8_0-gevm-k1-vlen256-mf2-vs-m1...` do-not-widen VINDICATED） |

> 成色分层：**q4_K@k1 = 唯一 强对手（hand-brick）≥parity**；其余 3 格对手皆 block-dot / parity-control / memory-bound（弱对手）。

### 1.2 <parity 已测（candidate·对称重测 LOSS·**不计** ≥parity · 9）

| 格·板 | emitted kernel 指针 | 对手 kernel 身份 | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| q5_K @rvv/VLEN128 | 前门 KQuant repack GEMM | factory generic block-dot（untuned·real·as-shipped） | ✓ gcc-15.2 双侧对称 | **0.120× LOSS**（~12.9× reversal·[CASE-COMPILER-ASYMMETRY] Stage1 蒸发·symmetric-gcc） |
| q2_K @rvv/VLEN128 | 前门 KQuant repack（dual d/dmin+bsums-min fold·S6-tiled） | factory hand-tuned _vl128 block-dot（real） | ✓ gcc-15 对称 | **0.386× LOSS**（分类报告§4·对称-gcc·throughput 0.20×） |
| q3_K @rvv/VLEN128 | 前门 KQuant（3-bit subtractive-hmask·PLAIN/UNTILED·S6 NULL） | factory hand-tuned _vl128 block-dot（real·但零 repack anywhere） | ✓ gcc-15 对称 | **~0.176–0.20× LOSS**（weight-bound·S6 NULL） |
| q6_K @rvv/VLEN128 | 前门 KQuant（6-bit dual-plane·PLAIN·S6 NULL） | factory mature block-dot（real） | ✓ gcc-15 对称 | **~0.18–0.19× LOSS**（2995 vsetivli 无调度·weight-recon floor） |
| iq4_nl @rvv/VLEN128 (gemm) | 前门 Codebook repack GEMM（kvalues_iq4nl·vl=8） | factory block-dot（real·routing 对位·cross-op） | ✓ rv64gcv shipped=gcc-15.2 对称（kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered） | **prefill 0.217× / decode 0.505× LOSS**（八门全过·clean window·g5-wiring M2-iq4_nl） |
| iq3_xxs @rvv (vec_dot) | 我方 block-dot emit | factory SIMD-dispatch block-dot（real·gcc-15.2） | ✓ batch2c gcc-15.2 -O3 双侧对称 | **0.158× LOSS**（WORST·fraclmul-2elem scalarization+vsetvli storm） |
| iq3_s @rvv (vec_dot) | 我方 block-dot emit | factory block-dot | ✓ batch2c 对称 | **0.280× LOSS** |
| iq2_xs @rvv (vec_dot) | 我方 block-dot emit | factory block-dot | ✓ batch2c 对称 | **0.329× LOSS**〔GAP-SB POST 1.509× vs-generic·但 vs-SIMD 仍 LOSS〕 |
| { iq2_s / iq2_xxs / iq4_xs / tq1_0 / tq2_0 } @rvv (vec_dot) | 我方 block-dot emit | factory SIMD-dispatch block-dot（real·gcc-15.2） | ✓ batch2c 对称 | iq2_s 0.477× / iq2_xxs 0.780×(MILDEST) / iq4_xs 0.718× / tq1_0 0.550× / tq2_0 0.231×·tq2_0 spill-fix→仍 0.45× vs-SIMD |

> 注：§1.2 末行是 5 格合并展示（vec_dot 同族·gcc-15.2 batch2c 对称），逐格数据在 T8。计入 candidate 池但 **不计** ≥parity 常驻计数。

### 1.3 ≥parity 待板批对称 micro 补测（FLAT 5 gemm@rvv · 主会话缝隙·潜在 +5）

| 格·板 | emitted kernel 指针 | 对手 kernel 身份 | 对称编译 | micro A/B 状态 |
|---|---|---|---|---|
| q4_0 / q4_1 / q5_0 / q5_1 / q8_0 @rvv/VLEN128 (gemm) | 净新 repack scaffold（q5_0 make_block_q5_0x16 / q5_1 block_q8_1x4 / q4_1 q8_1 家族 / q8_0 .inc=M1b） | factory block-dot（as-shipped·q4_0 上游 repack VLEN128-gated / q4_1·q5_0·q5_1 无 stock repack→generic / q8_0 上游 VLEN128 破损） | ✓ gcc-15 出货对称（rvv shipped=gcc-15·kernel-axis==system-axis） | ROADMAP 称「gcc-symmetric ≥parity 全幸存」（[GAP-FLAT-E2E]）·但逐格独立对称 kernel-axis micro 数**部分缺**（q8_0 有 item4/fill·q4_0 有 prefill parity·q4_1/q5_0/q5_1 待补）→**板批缝隙补 N≥10 对称 micro** |

---

## 2. 对手类单列（**不入计数** · 判据 = 非 as-shipped 或 internal-A/B）

| 格 | 对手类 | 为何不入 kernel-sym 计数 | 指针 |
|---|---|---|---|
| q4_0 @ime | **对手 SELF**（internal-A/B·我方非-IME 参考） | 无独立 factory IME kernel 对位·compute-account 2.09× = vs SELF·非 kernel-vs-kernel | T8 IME 行 |
| q8_0 @ime | 对手 SELF | 同上 | 同上 |
| q4_K @ime | 对手 SELF | 同上·G6-A 桥 exit-b（0.5529× stock·仍<parity·厂商 VEN/OFF 1.997× 是 vendor 非 our-kernel 对位） | T8 IME 行 |
| q5_K@rvv "1.5×" / q2_K@rvv "1.413×" / q4_K@rvv "1.884×" 等 kernel-axis "赢" | **[CASE-COMPILER-ASYMMETRY]**（clang-ours -O2 vs gcc-shipped generic） | 非对称编译·已撤回（对称-gcc 重测蒸发 0.12–0.39×）·kernel-axis account WITHDRAWN | T8 row `...symmetric...`；memory perf-constitution |

> ★纪律：IME 三格若未来建独立 vendor-IME-kernel-vs-our-IME-kernel 对称对位，可从「对手 SELF 单列」升为 candidate；现状 = SELF·单列不计数。

---

## 3. opponent-absent（**非 candidate** · gemm-轴无 same-op 对手 kernel）

| 格 | 缺口 | 说明 |
|---|---|---|
| gemm/iq2_xxs · iq2_xs · iq2_s · iq4_xs · mxfp4 · tq1_0 · tq2_0（7） | **gemm-轴对手 absent**（opponent 零 iq/tq repack GEMM·只有 block-dot vec_dot·跨-op） | 对位退化为 our-gemm vs opponent-block-dot（cross-op·如 iq4_nl 已做）·**非 same-op kernel-sym**·主会话若采 cross-op 口径可纳入（但须明标 cross-op·区别 same-op） |
| vec_dot/q1_0 | no-fair-opponent（Weft-internal binary·非标准 ggml·[主会话4裁定④]） | ggml 无 q1_0 vec_dot 对位 |
| vec_dot/nvfp4 | no-fair-opponent（fp4·ggml 疑无 nvfp4·[主会话4裁定④]） | ggml 无 nvfp4 对位 |

---

## 4. 计数汇总 + 待办

| 桶 | 格数 | 说明 |
|---|---:|---|
| **★第二常驻计数「kernel-sym ≥parity 格数」** | **4** | q4_K@k1 · q5_K@k1 · q4_0@k1-gemm-prefill-parity · q8_0@k1 |
| ≥parity 待板批补测（FLAT 5 gemm@rvv） | ~5 | 补测后潜在 +5 → 上限 9 |
| <parity candidate（对称 LOSS·不计 ≥parity） | 9 | §1.2 |
| 对手类单列（SELF/internal-A/B/CASE-COMPILER-ASYMMETRY） | 3 + N | §2·不入计数 |
| opponent-absent（非 candidate） | ~9 | gemm iq/tq 7 + q1_0/nvfp4·§3 |

**板批缝隙补测队列（主会话·非本台账执行）**：FLAT 5 gemm@rvv 逐格对称 kernel-axis micro（N≥10·T-N·gcc-15 双侧·对手=as-shipped block-dot）→ 补齐 §1.3 待测 5 格。

**需主会核**：① cross-op 口径（gemm-iq/tq vs block-dot）是否纳入 kernel-sym 台账（须明标 cross-op）；② IME「对手 SELF」是否未来升 candidate（需独立 vendor-kernel 对位）。
