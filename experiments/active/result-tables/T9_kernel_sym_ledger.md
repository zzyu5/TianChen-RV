# T9 — kernel-sym 台账（第二赛道·T3 派生）

> **建成**：2026-07-13 L3-apply（casefile `experiments/active/l3-triage/kernel-sym-台账-candidates.md` 派生）。
> **来源基**：T3_A/T3_B（board micro A/B）+ T8_winloss_gap_ledger.csv（对称重测账）。

---

## 0. 台账定义头

- **性质**：**第二赛道** kernel 内核轴覆盖面台账（C3′ 证据·T3 派生）。**与 perf-covered 系统账（`schema/perf-covered-category.v1.json` · **9/83**·recon 权威）永不混算**。★**2026-07-14 警示**：kernel-sym ≥parity 现 = **12**（FLAT@k1 +3）·perf-covered 现 = **9/83**——**完全不同赛道**（kernel-sym=kernel-axis 对称 micro 覆盖面 / perf-covered=系统账 e2e ≥parity 格）·**报告时必分开标·禁合并禁互推**（此前"两个 9 巧合"已解除·kernel-sym 12≠perf 9）。
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

### 1.1 ≥parity 已测 — 计入第二常驻计数 ★ **kernel-sym ≥parity 格数 = 12（matmul 家族·★2026-07-14 收窄订正：B类 forward 不并入此计数·见 §1.4 独立桶）**（★2026-07-13 FLAT 5 @rvv `f8da2f5c` 4→9 · ★2026-07-14 FLAT@k1 kernel-axis q4_1/q5_0/q5_1 `abb26043` 9→12·FLAT 家族双板收口 rvv5+k13）
> **★★收窄订正（`27fdc898`·2026-07-14·axis hygiene 禁并入）**：此前误把 B类前向算子 k1-half 4 WIN 并入 matmul kernel-sym（12→16）·**撤回**——① forward-elementwise=**独立 op 家族桶**（对手=as-shipped 前向核非 hand-brick·**禁并入 matmul kernel-sym-12**）② B类 rvv-half（`27fdc898`）证 **k1 的 add/mul/cpy WIN=VLEN256-only·VLEN128 部署板全塌回 PARITY**（纯 elementwise cold=DRAM 带宽绑定·wide-m8 优势 VLEN256-特定蒸发）→ **B类双板确认 WIN 仅 1（rms_norm·VLEN-invariant 1-pass fusion·成色=结构融合非硬碰硬手调）**。matmul kernel-sym 回 **12**·B类见 §1.4。

| 格·板 | emitted kernel 指针 | 对手 kernel 身份（成色） | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| **q4_K @k1/VLEN256** | 前门 lowerToRepackGemm KQuant（typed_repack·S6-tiled·spill→0·v30≤32 cliff）·`experiments/active/kquant-k1-vlen256-kernel-axis-t4a` | ⚠**改判 = `ggml_vec_dot_q4_K_q8_K` block-dot（弱—中对手·非 hand-brick·b29c269c 三重证）**。★真 hand-brick `ggml_gemm_q4_K_16x1_q8_K`（k1 真出货默认·case256·VLEN256 满宽 vl=16 专调）**另测 ours 输 0.622×**（hand-brick 快 1.61×·对称 clang-18·§6/q4K-handbrick-resolve）→ **"赢 hand-brick" 证伪·成色=vs-block-dot** | ✓ clang-18 双侧对称 | ✓ **3.106× ≥parity（vs block-dot）**（T8·kernel-axis）·**vs 真 hand-brick 0.622× LOSS**（另测·[GAP-KQUANT-VLEN256-UNROLL-VS-ROLLED] 全展开 25KB≠perf） |
| **q5_K @k1/VLEN256** | 前门 KQuant（qh 5th-bit leaf·min-fold·S6 HOLDS·XFER-1 #2） | factory block-dot（generic·k1 ships zero q5_K repack·**通用路径·弱对手**） | ✓ clang-18 对称 | ✓ **1.916× ≥parity**（T8·ALL 12 rounds>1·NOT-e2e） |
| **q4_0 @k1/VLEN256 (gemm prefill)** | repack GEMM（mf2 fractional·columnsPerPass 4） | e2e-factory block-dot（VLEN-flip prefill control·**parity-control**） | ✓ freq-locked paired 同源 | ✓ **prefill 1.0022× PARITY**（T8 row `q4_0-gevm-k1-vlen256...` 伴随行·CI[1.0014,1.0030]·compute-bound·parity=result） |
| **q8_0 @k1/VLEN256** | repack GEVM wide-hl16 mf2 one-strip·item4 fcvt-reschedule | factory-block-dot（**弱对手·memory-bound**） | ✓ clang-18 对称·preflight 4/4 dual-board | ✓ **+4.37%（item4）** / mf2-wide vs m1 +11%（T8 `q8_0-gevm-k1-vlen256-mf2-vs-m1...` do-not-widen VINDICATED） |
| **q4_1 @k1/VLEN256 (gemm)** | vl=16 emitted repack `weft_emitted_gemm_q4_1.inc`（board md5 `736a716c`） | 机判 `ggml_vec_dot_q4_1_q8_1`（0x9f774·block-dot·**单实现折中·k1 无 stock repack·light-vec 弱**） | ✓ clang-18 双侧对称（stock=clang-18·0 gcc refs） | ✓ **7.188× hot / 7.109× cold ≥parity**（N=12·abb26043·kernel-axis·nbad=0·objdump libcall-free vwmacc=8 无 spill·**禁互推 §5 e2e**） |
| **q5_0 @k1/VLEN256 (gemm)** | vl=16 emitted repack `weft_emitted_gemm_q5_0.inc`（board md5 `7b1ac4e6`） | 机判 `ggml_vec_dot_q5_0_q8_0`（0x9f81c·block-dot·**单实现折中·k1 无 stock repack**） | ✓ clang-18 双侧对称 | ✓ **2.329× hot / 2.326× cold ≥parity**（N=12·abb26043·kernel-axis·nbad=0·cold≈hot NULL compute-bound·**禁互推 §5 e2e**） |
| **q5_1 @k1/VLEN256 (gemm)** | vl=16 emitted repack `weft_emitted_gemm_q5_1.inc`（board md5 `03866f50`） | 机判 `ggml_vec_dot_q5_1_q8_1`（0x9f942·block-dot·**单实现折中·k1 无 stock repack**） | ✓ clang-18 双侧对称 | ✓ **2.574× hot / 2.571× cold ≥parity**（N=12·abb26043·kernel-axis·nbad=0·**禁互推 §5 e2e**） |
| **q4_0 @rvv/VLEN128 (gemm)** | repack GEMM（XOR-0x88 signed-nibble） | factory block-dot as-shipped（**light-vec 弱·10 rvv-insn**） | ✓ **gcc-15.2 双侧对称**（rvv 出货=gcc-15·kernel-axis==system-axis·objdump✓） | ✓ **6.707× ≥parity**（N=12·relIQR 0.51%·kernel-sym-flat5-rvv-micro·f8da2f5c） |
| **q4_1 @rvv/VLEN128 (gemm)** | repack GEMM（q8_1 家族·d+m） | factory block-dot as-shipped（**light-vec 弱·8 rvv-insn**） | ✓ gcc-15.2 双侧对称 | ✓ **6.829× ≥parity**（N=12·relIQR 0.84%） |
| **q5_0 @rvv/VLEN128 (gemm)** | repack GEMM（make_block_q5_0x16·transposed-qh） | factory block-dot as-shipped（**better-vec 较强·26 rvv-insn**） | ✓ gcc-15.2 双侧对称 | ✓ **1.221× ≥parity**（N=12·min 1.206×·opp-side 方差 relIQR 5.35%·我方 kernel 侧 <1%） |
| **q5_1 @rvv/VLEN128 (gemm)** | repack GEMM（block_q8_1x4·transposed-qh） | factory block-dot as-shipped（**better-vec 较强·24 rvv-insn**） | ✓ gcc-15.2 双侧对称 | ✓ **1.407× ≥parity**（N=12·min 1.350×·opp-side 方差 relIQR 4.06%·我方侧 <1%） |
| **q8_0 @rvv/VLEN128 (gevm)** | repack GEVM（.inc=M1b·i8@32） | factory block-dot as-shipped（**light-vec 弱·7 rvv-insn**） | ✓ gcc-15.2 双侧对称（clang-O3 deploy 3.47× 亦 ≥parity·三账本稳健） | ✓ **4.077× ≥parity**（N=12·relIQR 0.52%） |

> ★成色分层（12 格·令六 lint·成色须标）：**★★2026-07-14 更新（+FLAT@k1 3 格 abb26043·对手全 block-dot 单实现折中·0 新 hand-brick）：kernel-sym 12 = 【0 verified hand-brick】+ 2 better-vec（q5_0/q5_1@rvv）+ 10 block-dot/light（含 q4_K@k1 实测对手=block-dot·+q4_1/q5_0/q5_1@k1 单实现折中弱—中对手）**。q4_K@k1 **kernel-sym（vl=8 核 s6_q4K.c md5 90d454da）**"赢 hand-brick·成色最硬" 主张**已证伪**（ours vl=8 vs 真 hand-brick 16x1 repack = 0.622× LOSS）——**★但此仅限 vl=8 kernel-sym 核**（27658b8a·五 复核·双核分立）。**★★分轴诚实结论（收窄"两轴 0"过度外推）**：**kernel-sym（micro 轴·9 ≥parity）= 0 verified hand-brick win**（正确·vl=8 核 vs block-dot·输真 16x1）；**★perf-covered+sealed（e2e 轴 + kernel 轴）= 1 verified 【双轴】hand-brick win = Win-K1-VLEN**（q4_K@k1 **sealed vl=16 核** s6_q4K_vl16_sealed.c md5 e437fd3b·**★G 决胜局 RESOLVED（7d57a571·12 测全>parity·kernel-axis median 1.197× + e2e 1.085×/1.0876× byte-exact）= 首个双轴 verified hand-brick win**·区别 vl=8 kernel-sym 核[双核分立立]）。→ 成色质变"成排赢手调"仍未达（**仅 1 格·勉强**）·但**"两轴 0"错·应"kernel-sym 计数-9(vl=8 核) 0 · Win-K1-VLEN sealed(vl=16 核) 双轴 1"**。★**判别键 = 核形态（S6-tiled/vl=16 满宽 fit 预算 vs vl=8/full-unroll spill）·非 board/VLEN**（register-budget-fit 律·vl=16 spill 2-3=vl=8 的一半）。计数 9 不变（覆盖面/C3′ 证据·非成色等价）。〔旧存疑表述已撤〕；其余 8 格对手皆 **factory block-dot / parity-control / memory-bound（非 hand-brick·弱—中对手）**。其中 FLAT @rvv 5 格再分：q4_0/q4_1/q8_0 = light-vec 弱对手（big win 4-6.8×·成色低）· **q5_0/q5_1 = better-vec 较强对手（modest win 1.22-1.41×·成色相对硬·打败 better-vectorized block-dot）**。**计数 9 是覆盖面（第二赛道·C3′ 证据）·非成色等价**——9 格里只有 q4_K@k1 是赢 hand-brick。correctness 全 ZERO-MODEL PASS。

### 1.2 <parity 已测（candidate·对称重测 LOSS·**不计** ≥parity · 12·+q2_K/q3_K/q6_K@k1）

> **★cold 双态 robustness 收口（2026-07-14·a8ed3a09·rvv §1.2 IQ/TQ 9/9 全测·同形状 hot/cold·gcc-15.2 对称·N≥12·`29d48dd7`）**：8 vec_dot + iq4_nl gemm **全 <parity·cold≈hot·verdict 不翻**（compute-latency/gather-bound·ours cold GB/s 0.18–1.5 ≪ DRAM 墙·cache 态二阶·同 FLAT-5/k1-L2 型）→ §1.2 LOSS 判定 cold 稳健。**★headline①（不入 ≥parity 计数）**：**tq2_0 hot 1.15× ≥parity 惊喜**（L1-resident·ours vsetvl=3 近零 vsetvli storm BEATS factory·rock-solid N=20 relIQR~1%）→ **cold 0.21× 蒸发**（ours-penalty 8.6× vs factory 1.6×·cache-resident-only micro-win 不存活 DRAM streaming·thesis-coherent·**部署域 LOSS**）。**★headline②**：iq4_nl 部署域校正见下行（0.837× 存档数 = [CASE-COMPILER-ASYMMETRY]·gcc-sym 0.23×）。cold 逐格数据在 `experiments/active/g7-l1-kernelsym-fullfill/rvv-iqtq-cold/evidence.md §6`。

| 格·板 | emitted kernel 指针 | 对手 kernel 身份 | 对称编译 | 已有 micro A/B（T8） |
|---|---|---|---|---|
| q5_K @rvv/VLEN128 | 前门 KQuant repack GEMM | factory generic block-dot（untuned·real·as-shipped·qh-burden 弱 1.36 GMAC/s） | ✓ gcc-15.2 双侧对称 | **★0.763× LOSS**（G7 census `249145ff` fresh symmetric-gcc-15.2 on-board·**修正 T9 旧记 0.120×**·旧数 provenance 存疑 clang-asymmetric·gcc-ours 1.04 vs clang 2.19 GMAC/s·**C-dominant closest-to-flip**：若 gcc codegen 匹配 clang→~1.6× WIN vs 弱对手·B=qh 5th-bit [GAP-Q5K-VLEN128-QH-REGCLIFF] NOT-FIXABLE） |
| q2_K @rvv/VLEN128 | 前门 KQuant repack（dual d/dmin+bsums-min fold·S6-tiled） | factory hand-tuned _vl128 block-dot（real） | ✓ gcc-15.2 对称 | **0.390× LOSS**（G7 census 249145ff·MATCHES 旧 0.386×·gcc vsetvli 1433·A=_vl128 hand-tuned·B=dual-fold·C=gcc codegen） |
| q3_K @rvv/VLEN128 | 前门 KQuant（3-bit subtractive-hmask·PLAIN/UNTILED·S6 NULL） | factory hand-tuned _vl128 block-dot（real·零 repack anywhere） | ✓ gcc-15.2 对称 | **★0.049× LOSS**（G7 census 249145ff fresh symmetric-gcc·**修正 T9 旧 ~0.176×**[clang-asymmetric]·**deep**·**★C-dominant=gcc-15.2 vsetvli-storm 5115** on full-unroll body·clang-17 3.4× faster·roll GEMM 攻坚靶） |
| q6_K @rvv/VLEN128 | 前门 KQuant（6-bit dual-plane·PLAIN·S6 NULL） | factory mature block-dot（real·NEON-8x8-only repack disclosed unused） | ✓ gcc-15.2 对称 | **★0.037× LOSS**（G7 census 249145ff fresh symmetric-gcc·**修正 T9 旧 ~0.18×**[clang-asymmetric]·**deepest**·**★C-dominant=gcc-15.2 vsetvli-storm 6278**·vwmacc~2200 equal MAC·loss=overhead 非 MAC·roll GEMM 攻坚靶 L1prime-q6k in-flight） |
> **★★whole-family [CASE-COMPILER-ASYMMETRY] 修正（G7 census 249145ff·2026-07-14）**：T9 旧记 K-quant@rvv 数（标"gcc 对称"实多为 clang-asymmetric·casefile clang-ours 1.5× 偏快）↔ **正确 symmetric-gcc-15.2 on-board census** 背离：q5_K 0.120×→0.763×(旧数 provenance 存疑)·q3_K 0.176×→0.049×(deeper)·q6_K 0.18×→0.037×(deeper)·q2_K 0.386×≈0.390×(matches)。**★根因分层=loss 是 gcc-15.2 vsetvli-storm+spill 于 full-unroll GEMM body(vwmacc~2200 equal MAC·非 MAC 质量)·q3/q6 C-dominant(vsetvli 5115/6278→20-25× loss)**·攻坚靶=roll GEMM emitter(whole-K-nest 谱系·G1 in-flight `ad287c7c`·核查 re-decode killer)。byte-exact 全 mismatch=0(correct-but-slow 非 miscompiled)。
| iq4_nl @rvv/VLEN128 (gemm) | 前门 Codebook repack GEMM（kvalues_iq4nl·vl=8） | factory block-dot（real·routing 对位·cross-op） | ✓ rv64gcv shipped=gcc-15.2 对称（kernel-axis==system-axis·[CASE-COMPILER-ASYMMETRY] not triggered） | **prefill 0.217× / decode 0.505× LOSS**（八门全过·clean window·g5-wiring M2-iq4_nl） |
| iq3_xxs @rvv (vec_dot) | 我方 block-dot emit | factory SIMD-dispatch block-dot（real·gcc-15.2） | ✓ batch2c gcc-15.2 -O3 双侧对称 | **0.158× LOSS**（WORST·fraclmul-2elem scalarization+vsetvli storm） |
| iq3_s @rvv (vec_dot) | 我方 block-dot emit | factory block-dot | ✓ batch2c 对称 | **0.280× LOSS** |
| iq2_xs @rvv (vec_dot) | 我方 block-dot emit | factory block-dot | ✓ batch2c 对称 | **0.329× LOSS**〔GAP-SB POST 1.509× vs-generic·但 vs-SIMD 仍 LOSS〕 |
| { iq2_s / iq2_xxs / iq4_xs / tq1_0 / tq2_0 } @rvv (vec_dot) | 我方 block-dot emit | factory SIMD-dispatch block-dot（real·gcc-15.2） | ✓ batch2c 对称 | iq2_s 0.477× / iq2_xxs 0.780×(MILDEST) / iq4_xs 0.718× / tq1_0 0.550× / tq2_0 0.231×·tq2_0 spill-fix→仍 0.45× vs-SIMD |

| q2_K @k1/VLEN256 | 前门 KQuant PLAIN full-unroll | factory block-dot ggml_vec_dot_q2_K_q8_K(机判) | ✓ clang-18 对称 | **0.60× LOSS**(hot≈cold·spill 627·weight-bound·VLEN256 不救 full-unroll·7d57a571) |
| q3_K @k1/VLEN256 | 前门 KQuant PLAIN full-unroll | factory block-dot ggml_vec_dot_q3_K_q8_K(机判) | ✓ clang-18 对称 | **0.34× LOSS**(spill 1058·weight-bound) |
| q6_K @k1/VLEN256 | 前门 KQuant PLAIN full-unroll | factory block-dot ggml_vec_dot_q6_K_q8_K(机判) | ✓ clang-18 对称 | **0.30× LOSS**(spill 971·weight-bound·nr64 3-seed) |

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
| vec_dot/nvfp4 | ~~no-fair-opponent（fp4·ggml 疑无 nvfp4·[主会话4裁定④]）~~ **SUPERSEDED** | **★对手身份存疑·superseded（G8 阶段三 §五 opponent-reparse 2026-07-14）**：`ggml_vec_dot_nvfp4_q8_0` **存在**（generic·ggml-cpu/quants.c·双板 tot~215/rvv23/mac2·native-vec-generic 弱·无 riscv hand-tuned 专化）→ **≠ absent**·in-0.8-denom（弱对手）。仅 riscv 手调专化缺·非对手 absent。casefile=`experiments/active/g8-stage3-opponent-reparse/` |

> **★G8 阶段三 §五 opponent-reparse 交叉引用（2026-07-14）**：全表对手 clang-18 对称域重解析·casefile `experiments/active/g8-stage3-opponent-reparse/`（evidence.md + 双板 objdump_metrics）。本 T9 rvv opponent 机判**逐格 confirmed**（同符号同成色·无稻草人）；两处收窄/纠正：① nvfp4 vec_dot "absent"→generic-弱（上行 superseded）；② iq4_nl vec_dot `_vlNNN` 旧"STRONG"→native-vec-中（thin wrapper·rvv18/k1 24）。**★k1 K-quant GEMM 对手大纠正**：k1 stock .so 带 **real same-op repack hand-brick**（q2_K/q4_K/q5_K/q6_K·8x8/8x4/16x1·仅 q3_K 无）→ k1 K-quant GEMM **非** cross-op（≠ rvv 全 cross-op）·per-board 对手成色独立·禁跨板沿用。q4_K@k1 双核清算：kernel-sym vl=8 核 races block-dot(vl256 rvv54)·sealed Win-K1-VLEN vl=16 核 races 真 16x1 hand-brick（decode `ggml_gemv_q4_K_16x1` rvv209 / prefill `ggml_gemm_q4_K_16x1` rvv224）。

---

## 4. 计数汇总 + 待办

| 桶 | 格数 | 说明 |
|---|---:|---|
| **★第二常驻计数「matmul kernel-sym ≥parity 格数」** | **12** | q4_K@k1 · q5_K@k1 · q4_0@k1-gemm-prefill · q8_0@k1 · ★FLAT@rvv: q4_0 · q4_1 · q5_0 · q5_1 · q8_0（f8da2f5c）· ★FLAT@k1: q4_1 · q5_0 · q5_1（abb26043）· **★成色分布 = 0 verified hand-brick + 2 better-vec block-dot（q5_0/q5_1@rvv）+ 10 block-dot/light（弱—中对手）**·〔B类前向不并入·见下独立桶〕 |
| **★★forward-op kernel-sym（独立桶·2026-07-14·`27fdc898`·禁并入 matmul-12·禁混算 perf-covered）** | **双板 WIN 交集 = 1** | **rms_norm**（k1 1.335×/rvv 1.152×·VLEN-invariant 1-pass fusion vs ggml 2-pass·唯一双板确认 WIN·成色=结构融合非硬碰硬）｜双板 ≥parity 交集 = 5{add,mul,cpy,rms_norm,rope}（add/mul/cpy 的 k1 WIN=VLEN256-only·rvv PARITY 不得计双板赢）｜dual-LOSS 3{softmax −15~18% sched·silu·~~gelu LUT-vs-tanhf 结构~~→**G.0.3 同档重比后 gelu 不再 dual-LOSS**：f16-LUT 同档 byte-exact 0 ULP → **rvv WIN 1.116×**(便宜档表查·gcc-15 调度·非硬赢)/**k1 PARITY 0.957×**([GAP-GELU-K1-VLEN256-CLANG-SCHED])·旧 LUT-vs-tanhf 系跨精度档非公平·见 `g7-census/gelu-f16lut-rematch/`}｜byte-exact/ULP 全硬门过·净新 territory |
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
| **q4_1** | perf-covered 绿（净新 scaffold） | ✅ **≥parity·★真 our-kernel**（prefill 4.9955×/decode 1.9205× WIN·VLEN256 vl=16 net-new emitted repack·d811b323） | **我方 net-new VLEN256 kernel**（k1 无 stock repack·winner=我方·**≠ q4_0/q8_0 的 stock-repack 白嫖**·成色更强） | **★real our-kernel dual-board**（rvv 绿 + k1 e2e ≥parity·winner=我方 kernel）·⚠caveat: 5× vs 未优化 block-dot(弱对手·L1 path-win)·k1 含 dormant vendor repack.cpp q4_1[NOT compiled·启用则对手更强]·perf-covered 9/83 不变(已绿·dual-board 成色) |
| **q5_0** | perf-covered 绿（净新 scaffold） | ✅ **≥parity·★真 our-kernel**（prefill 2.2096×/decode 0.7171×·VLEN256 vl=16·c96a613c） | 我方 net-new VLEN256 kernel（k1 无 stock repack·winner=我方） | **★real our-kernel dual-board**·caveat 5× 级弱对手 block-dot·9/83 不变 |
| **q5_1** | perf-covered 绿（净新 scaffold） | ✅ **≥parity·★真 our-kernel**（prefill 2.4525×/decode 0.8103×·net-new q8_1 mat-quant·c96a613c） | 我方 net-new VLEN256 kernel | **★real our-kernel dual-board**·A==B 3/4+1 benign near-tie(coherent)·9/83 不变 |

**★货架B 首批结论（诚实）**：q4_0/q8_0 e2e 在 **双板均 ≥parity**（repack approach 传导·dual-board 成色证据）·但 **k1 半的 winner = stock 自己的 repack（非我方 emitted kernel）** → **不新增 perf-covered green·维持 9/83**。our-emit↔stock-repack kernel-axis parity 已封（§1.1 q4_0@k1 1.0022×/q8_0@k1 +4.37%）·传递链 our-emit≈stock-repack≈本 e2e margin。★★FLAT@k1 五格 dual-board 矩阵 **CLOSED**（c96a613c）：q4_0/q8_0@k1 = stock-repack winner（白嫖·不增 perf-covered）· **q4_1/q5_0/q5_1@k1 = 真 our-kernel dual-board**（winner=我方 VLEN256 vl=16 kernel·k1 无 stock repack·prefill 4.99×/2.21×/2.45× WIN·成色强于 stock-repack 二格·decode 预着色 LOSS·caveat 弱对手 block-dot L1 path-win）·perf-covered 9/83 不变（dual-board 成色·禁互推）。IME 格豁免双板（rvv 无矩阵单元·令四）。

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

---

## 6.2 货架A 全量 rvv-半 §1.1 cold 双态 + nr 穷举（G7 §L1·`668f1d45`）

> 15/15 cell（FLAT 5 × nr{4,16,64}）hot∧cold 全 ≥parity·GATE=PASS·gcc-15 对称·对手全 = factory block-dot（机判 nm public symbol·非 hand-brick·印证分轴修正 = kernel-sym 0 verified hand-brick）。

| 格@rvv | 对手符号(机判) | 折中态 | HOT | COLD(nr16) | cold regime |
|---|---|---|---:|---:|---|
| q4_0 | ggml_vec_dot_q4_0_q8_0 | 单实现折中 | 6.780× | 6.224× | compute-bound |
| q4_1 | ggml_vec_dot_q4_1_q8_1 | 单实现折中 | 6.747× | 6.105× | compute-bound |
| q5_0 | ggml_vec_dot_q5_0_q8_0 | 单实现折中 | 1.228× | 1.232× | compute-bound |
| q5_1 | ggml_vec_dot_q5_1_q8_1 | 单实现折中 | 1.499× | 1.412× | compute-bound |
| q8_0 | ggml_vec_dot_q8_0_q8_0 | **破损** | 4.193× | 4.752× | memory-leaning |

**★发现**：① cold-penalty 键控 memory-exposure（nr↓→penalty↑·低 weight-reuse 赢家更 memory-exposed·最严 q5_1 nr4 1.124× 仍 ≥parity）② q8_0 反常 cold>hot（对手 block-dot 权重足迹 34B/blk 最大·cold 退化>ours）③ q5_0/q5_1 win↗nr。**★T4b oracle**：VLEN128 repack codegen 候选集 = **{mf2} 单一合法**（half_lanes=8→mf2·columnsPerPass=4 auto-forced·m1 是 RVV0.7-only 非 same-board 候选·[repack-winA-always-mf2]）→ **selector 平凡最优**·变体穷举净信息 = nr-shape 轴（mf2-repack 跨 nr 全 ≥parity·[L-4] 合规）。双账本稳健（clang nr16 10/10 cold ≥parity）。

**进度分数（货架A 全量·§六.1 分数化）**：**rvv-半 §1.1 = 5/5 filled**（cold 双态+nr 穷举+双账本）· **rvv-半 §1.2 = IQ/TQ 9/9 filled**（`a8ed3a09` 2026-07-14·8 vec_dot + iq4_nl gemm cold 双态·全 <parity·cold≈hot·verdict 不翻·tq2_0 hot-only 惊喜蒸发·iq4_nl 部署域校正）· **余 4 = q5_K/q2_K/q3_K/q6_K@rvv gemm（K-quant·另 batch·regime T8 确认 LOSS）**· **★k1-半 §1.1 = 7/7 filled**（q4_0/q8_0@k1 存量 + ★FLAT@k1 kernel-axis q4_1/q5_0/q5_1 cold 双态 `abb26043` 2026-07-14·全 ≥parity·nr4/16/64 穷举·nbad=0）· **k1-半 §1.2 = q2_K/q3_K/q6_K@k1 已测 <parity**。★**kernel-sym ≥parity 计数不因 cold 变**（cold=hot robustness·非新格·NOT e2e·禁混算 perf-covered 9/83）。
