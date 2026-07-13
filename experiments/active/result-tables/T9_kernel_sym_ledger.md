# T9 — kernel-sym 台账（第二赛道·T3 派生）

> **建成**：2026-07-13 L3-apply（casefile `experiments/active/l3-triage/kernel-sym-台账-candidates.md` 派生）。
> **来源基**：T3_A/T3_B（board micro A/B）+ T8_winloss_gap_ledger.csv（对称重测账）。

---

## 0. 台账定义头

- **性质**：**第二赛道** kernel 内核轴覆盖面台账（C3′ 证据·T3 派生）。**与 perf-covered 系统账（`schema/perf-covered-category.v1.json` · **9/83**·recon 权威）永不混算**。★**2026-07-13 警示**：kernel-sym ≥parity 现 = **9**·perf-covered 现 = **9/83**——**两个 9 是巧合·完全不同赛道**（kernel-sym=kernel-axis 对称 micro 覆盖面 / perf-covered=系统账 e2e ≥parity 格）·**报告时必分开标·禁合并禁互推**。
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

### 1.1 ≥parity 已测 — 计入第二常驻计数 ★ **kernel-sym ≥parity 格数 = 9**（★2026-07-13 kernel-sym 首轮 FLAT 5 @rvv 补测·commit `f8da2f5c`·4→9 上限达成）

| 格·板 | emitted kernel 指针 | 对手 kernel 身份（成色） | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| **q4_K @k1/VLEN256** | 前门 lowerToRepackGemm KQuant（typed_repack·S6-tiled·spill→0·v30≤32 cliff）·`experiments/active/kquant-k1-vlen256-kernel-axis-t4a` | **真出货 hand-brick**（k1 stock RVV q4_K 16x1 repack·case256 fires·NOT block-dot·**唯一强对手·成色最硬**） | ✓ clang-18 双侧对称（k1 stock=clang-18） | ✓ **3.106× ≥parity**（T8·ALL 12 rounds>1·NOT-e2e·kernel-axis） |
| **q5_K @k1/VLEN256** | 前门 KQuant（qh 5th-bit leaf·min-fold·S6 HOLDS·XFER-1 #2） | factory block-dot（generic·k1 ships zero q5_K repack·**通用路径·弱对手**） | ✓ clang-18 对称 | ✓ **1.916× ≥parity**（T8·ALL 12 rounds>1·NOT-e2e） |
| **q4_0 @k1/VLEN256 (gemm prefill)** | repack GEMM（mf2 fractional·columnsPerPass 4） | e2e-factory block-dot（VLEN-flip prefill control·**parity-control**） | ✓ freq-locked paired 同源 | ✓ **prefill 1.0022× PARITY**（T8 row `q4_0-gevm-k1-vlen256...` 伴随行·CI[1.0014,1.0030]·compute-bound·parity=result） |
| **q8_0 @k1/VLEN256** | repack GEVM wide-hl16 mf2 one-strip·item4 fcvt-reschedule | factory-block-dot（**弱对手·memory-bound**） | ✓ clang-18 对称·preflight 4/4 dual-board | ✓ **+4.37%（item4）** / mf2-wide vs m1 +11%（T8 `q8_0-gevm-k1-vlen256-mf2-vs-m1...` do-not-widen VINDICATED） |
| **q4_0 @rvv/VLEN128 (gemm)** | repack GEMM（XOR-0x88 signed-nibble） | factory block-dot as-shipped（**light-vec 弱·10 rvv-insn**） | ✓ **gcc-15.2 双侧对称**（rvv 出货=gcc-15·kernel-axis==system-axis·objdump✓） | ✓ **6.707× ≥parity**（N=12·relIQR 0.51%·kernel-sym-flat5-rvv-micro·f8da2f5c） |
| **q4_1 @rvv/VLEN128 (gemm)** | repack GEMM（q8_1 家族·d+m） | factory block-dot as-shipped（**light-vec 弱·8 rvv-insn**） | ✓ gcc-15.2 双侧对称 | ✓ **6.829× ≥parity**（N=12·relIQR 0.84%） |
| **q5_0 @rvv/VLEN128 (gemm)** | repack GEMM（make_block_q5_0x16·transposed-qh） | factory block-dot as-shipped（**better-vec 较强·26 rvv-insn**） | ✓ gcc-15.2 双侧对称 | ✓ **1.221× ≥parity**（N=12·min 1.206×·opp-side 方差 relIQR 5.35%·我方 kernel 侧 <1%） |
| **q5_1 @rvv/VLEN128 (gemm)** | repack GEMM（block_q8_1x4·transposed-qh） | factory block-dot as-shipped（**better-vec 较强·24 rvv-insn**） | ✓ gcc-15.2 双侧对称 | ✓ **1.407× ≥parity**（N=12·min 1.350×·opp-side 方差 relIQR 4.06%·我方侧 <1%） |
| **q8_0 @rvv/VLEN128 (gevm)** | repack GEVM（.inc=M1b·i8@32） | factory block-dot as-shipped（**light-vec 弱·7 rvv-insn**） | ✓ gcc-15.2 双侧对称（clang-O3 deploy 3.47× 亦 ≥parity·三账本稳健） | ✓ **4.077× ≥parity**（N=12·relIQR 0.52%） |

> ★成色分层（9 格·令六 lint·成色须标）：**⚠2026-07-13 修正见 §6：q4_K@k1 "hand-brick 对手" 标签【存疑】——实测对手=block-dot·真 repack `ggml_gemm_q4_K_16x1_q8_K` 未接·"唯一 hand-brick ≥parity·成色最硬"主张【未验证·须另测】·修正成色分布=0 verified hand-brick + 2 better-vec + 7 block-dot/light**。以下为旧（存疑）表述：**q4_K@k1 = 唯一 强对手（hand-brick·case256 真出货 repack）≥parity·成色最硬**；其余 8 格对手皆 **factory block-dot / parity-control / memory-bound（非 hand-brick·弱—中对手）**。其中 FLAT @rvv 5 格再分：q4_0/q4_1/q8_0 = light-vec 弱对手（big win 4-6.8×·成色低）· **q5_0/q5_1 = better-vec 较强对手（modest win 1.22-1.41×·成色相对硬·打败 better-vectorized block-dot）**。**计数 9 是覆盖面（第二赛道·C3′ 证据）·非成色等价**——9 格里只有 q4_K@k1 是赢 hand-brick。correctness 全 ZERO-MODEL PASS。

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

### 1.3 ≥parity 待板批对称 micro 补测 — ✅ **DONE（2026-07-13·f8da2f5c·5 格全 ≥parity 已移入 §1.1·计数 4→9 达成）**

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
| **★第二常驻计数「kernel-sym ≥parity 格数」** | **9** | q4_K@k1 · q5_K@k1 · q4_0@k1-gemm-prefill · q8_0@k1 · ★FLAT@rvv: q4_0 · q4_1 · q5_0 · q5_1 · q8_0（f8da2f5c）· **★成色分布（L0.3③·计数旁标）= 1 hand-brick（q4_K@k1·成色最硬）+ 2 better-vec block-dot（q5_0/q5_1@rvv·成色中）+ 6 block-dot/light（q5_K@k1/q4_0@k1/q8_0@k1/q4_0/q4_1/q8_0@rvv·弱对手·成色低）** |
| ~~≥parity 待板批补测（FLAT 5 gemm@rvv）~~ | ✅ DONE | 2026-07-13 f8da2f5c·5 格全 ≥parity·4→9 上限达成 |
| <parity candidate（对称 LOSS·不计 ≥parity） | 9 | §1.2·+ q2_K@rvv 系统账 fresh 0.857×(14f4631a·同向) |
| 对手类单列（SELF/internal-A/B/CASE-COMPILER-ASYMMETRY） | 3 + N | §2·不入计数（含 IME 三格 SELF·但注：IME q4_0/q8_0@ime 已在 perf-covered 转绿=不同赛道·此处 SELF-account 仍单列） |
| opponent-absent（非 candidate） | ~9 | gemm iq/tq 7 + q1_0/nvfp4·§3 |

**★成色警示（令六 lint·计数 9 ≠ 成色 9）**：9 格覆盖面里 **仅 q4_K@k1 赢 hand-brick（强对手·成色最硬）**；FLAT@rvv 5 格中 q5_0/q5_1 赢 better-vec block-dot（成色相对硬）·q4_0/q4_1/q8_0 赢 light-vec block-dot（弱对手·成色低）。**第二赛道计数=kernel-axis 对称覆盖面（C3′ 证据）·非 e2e·非 perf-covered·禁互推**。系统账 FLAT 仍 [GAP-FLAT-E2E] 黄（micro∧e2e + selector-routing 两门未过·T6 whole-model 后续）。

**需主会核**：① cross-op 口径（gemm-iq/tq vs block-dot）是否纳入 kernel-sym 台账（须明标 cross-op）；② IME「对手 SELF」是否未来升 candidate（需独立 vendor-kernel 对位）。

---

## 5. {rvv × k1} 双板矩阵（货架B·e2e 板覆盖·G7 L3·★≠ 第二赛道 kernel-sym 计数·≠ perf-covered 系统账·三账互不推）

> **性质**：**货架B（e2e 双板矩阵）**·记每格在 rvv/k1 两板的 e2e 板覆盖态。**★禁互推**：此列 = e2e 板覆盖成色（哪些板测过 e2e ≥parity）·**非** kernel-sym ≥parity 计数（§1.1·kernel-axis 对称 micro）·**非** perf-covered 绿（系统账·须我方-constructed kernel）。
> **成色须标**：e2e ≥parity 的 winner 身份（our-emitted kernel / stock-repack / stock-block-dot）逐格标明。

| 格 | rvv e2e | k1 e2e | winner 身份 | dual-board 态 |
|---|---|---|---|---|
| **q4_0** | perf-covered 绿（系统账 routing-win·我方翻 gate 路由上游 repack） | ✅ **≥parity**（prefill 5.18×/decode 1.66× both-WIN·G7 L3 `156ece53`） | k1 = **stock 16x1 repack**（as-shipped·非我方 emitted） | **dual-board 成色**（repack approach 双板 e2e 传导 ≥parity·⚠ k1 winner=stock repack·不增 perf-covered） |
| **q8_0** | perf-covered 绿（系统账 correctness-carrier·我方 vl=8 承载） | ✅ **≥parity**（prefill 2.35×/decode 1.21× both-WIN·4/4 byte-id） | k1 = **stock 16x1 repack** | **dual-board 成色**（同上·⚠ k1 winner=stock repack） |
| **q4_1** | perf-covered 绿（净新 scaffold） | ⏳ **pending**（k1 stock 纯 block-dot·无 repack 对位·需部署我方 net-new VLEN256 repack·结构缺口非测量失败） | — | rvv-only green·k1 dual-board pending net-new deploy |
| **q5_0** | perf-covered 绿（净新 scaffold） | ⏳ pending（同 q4_1） | — | 同上 |
| **q5_1** | perf-covered 绿（净新 scaffold） | ⏳ pending（同 q4_1） | — | 同上 |

**★货架B 首批结论（诚实）**：q4_0/q8_0 e2e 在 **双板均 ≥parity**（repack approach 传导·dual-board 成色证据）·但 **k1 半的 winner = stock 自己的 repack（非我方 emitted kernel）** → **不新增 perf-covered green·维持 9/83**。our-emit↔stock-repack kernel-axis parity 已封（§1.1 q4_0@k1 1.0022×/q8_0@k1 +4.37%）·传递链 our-emit≈stock-repack≈本 e2e margin。q4_1/q5_0/q5_1 k1 半需 net-new deploy（VLEN256 kernel 重 emit + scaffold 移植 + 模型 provisioning·结构缺口跟进项·rvv 绿不受影响）。IME 格豁免双板（rvv 无矩阵单元·令四）。

---

## 6. hot/cold 双态首批（货架A·k1 4 存量格·G7 L2·`be524605`）+ ★口径校正

> cold 协议：P=8 weight-tile POOL footprint >> LLC·每 round 扫全池·每 tile DRAM cold-read·median N=12。★board fact：k1 L2=512KiB < q4_K tile 576KiB → "hot" 本非真 warm。

| 格@k1 | hot | cold(nr64/nr4) | cold≥parity | regime(GB/s) | e2e decode | cold→decode 预测? |
|---|---:|---:|:---:|---|---:|:---:|
| q4_K | 3.19× | 3.18×/2.83× | ✅ | compute-bound(0.03–0.5) | 1.284× WIN | ✅ 弱(运气一致) |
| q5_K | 1.92× | 1.92×/1.79× | ✅ | spill-bound(0.01–0.19) | **0.729× LOSS** | ❌ **MISMATCH** |
| q4_0 | 3.51× | 3.50×/3.44× | ✅ | compute-bound(0.03–0.5) | 1.66× WIN | ✅ 弱 |
| q8_0(GEVM) | 1.48× | 1.48× | ✅ | **memory-leaning(1.73)** | 1.21× WIN | ✅ **机制/干净** |

**★发现①：hot≈cold = NULL 区分**（cache-cold 不改 kernel-axis 比值·因 compute(dequant/spill)-bound + L2<tile·缓存驻留仅二阶）。
**★发现② cold-predictor 铁律（q5_K 决定性证伪）**：`cold ≥parity ⟹ 预测 e2e decode ≥parity` **仅在 micro=decode-GEVM ∧ 触 memory-wall 时机制成立**（q8_0 GEVM 实证·1.73GB/s memory-leaning→命中 1.21×）。**cold-GEMM micro【禁】当 decode 预测器**（q5_K micro compute-bound repack-GEMM 1.79× WIN 无法预测 e2e memory-bound GEVM decode 0.729× LOSS·双重错配 GEMM≠GEVM ∧ compute≠memory·预测器正在最该预测的 memory 墙失效）。q4_K/q4_0 方向命中=运气一致（GEMM-micro↔e2e·非机制捕获）。**⇒ 货架A cold 行作 e2e decode 预测器须补 per-format decode-GEVM cold micro（现仅 q8_0 有·K-quant 待 [GAP-REPACK-GEVM] GEVM plan 落地·连 G7 L1 P1）**。

### ★★口径校正（需主会周期复核·honesty·影响 §1.1 成色）
1. **q4_K@k1 "hand-brick" 标签【存疑】**：sealed 3.106× 与本测 3.19× 实测对手都是 `ggml_vec_dot_q4_K_q8_K` **block-dot**（非 hand-brick）。板上确有更强 stock repack `ggml_gemm_q4_K_16x1_q8_K` 但**未接**（未作对手）→ **§1.1 "唯一 hand-brick ≥parity·成色最硬" 主张【未验证】·须另测 vs 真 repack 才能立**。**修正成色分布**：kernel-sym 9 = **0 verified hand-brick**（q4_K@k1 待确认·实测 block-dot） + 2 better-vec + 7 block-dot/light（含 q4_K@k1 实测 block-dot 对手）。
2. **q4_0/q8_0@k1 sealed 对手口径**：sealed q4_0 1.0022×=VLEN-flip **self-control**·q8_0 +4.37%=item4 **internal 变体**·**皆非 vs block-dot**。本 casefile 首次给 **ours-vs-factory-block-dot 干净对**（q4_0 3.5×/q8_0 1.48×·both ≥parity·light-vec block-dot 对手）。§1.1 那两行的旧数是 self/internal 口径·本行是 vs-block-dot 口径·**两口径并存不混**。
