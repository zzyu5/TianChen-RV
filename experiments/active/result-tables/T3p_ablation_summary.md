# T3p — 消融归拢总表 + 立柱对账 + micro↛e2e 卷宗（货架C·G7 令 L4·北极星第④步）

> **建成**：2026-07-13（G7 货架C·消融归拢+性能立柱提炼）。**性质**：纯 docs 综合——从 G7 战役已产 findings（T8 最近 ~15 行 + G7 casefile P0/P1/P2 + M7 + L2 三格 + hand-brick 另测）归拢**消融/负结果**为统一体例一表，对账 [T-PILLARS] 两柱预注册，归拢 micro↛e2e 病例卷宗，提炼诚实**方法学立柱**。
> **禁**：本表不改代码/schema/T8/ROADMAP/T-PILLARS（[T-PILLARS] 更新建议见 §2 末·改留主会话）。**无板**：不新测，只综合已 sealed 证据。
> **★三账永不混算**（令六铁律）：`perf-covered = 9/83`（系统账 e2e ≥parity·recon 权威）· `kernel-sym ≥parity = 9`（内核轴对称 micro 覆盖面·**与 9/83 数字巧合·完全不同赛道**）· `dual-board`（rvv×k1 e2e 板覆盖·q4_0/q8_0 双板但 k1 winner=stock repack·不增 perf-covered）。本表引数时逐处标账本，**禁互推**。
> **体例（[K-6] 缺口关闭环·正负例边界齐备）**：每行 `{消融维度 · 变体 on/off|paired · 结果(账×板×CI) · 归因([K-6] 四选一+反汇编) · design-space 边界(正例↔负例)}`。负结果不粉饰。

---

## 1. 消融总表（G7 七维·统一体例）

| # | 消融维度 | 变体 (on/off \| paired) | 结果（账·板·口径） | 归因（[K-6] 四选一 · 反汇编） | design-space 边界（正例 ↔ 负例） | 证据指针 · snapshot |
|---|---|---|---|---|---|---|
| A1 | **M7 wide-vmadot tile 宽度**（IME array-util·[PAT-1]·参数级 [K-10]） | paired W1 → **W2** → W4（`selectVmadotTilePattern` by 32-vreg budget·ONE vle8 A 喂 NJW 独立 vmadot 链） | **W2 正**：compute 1.740→1.095s（1.589×）· e2e onw2/onjout **1.1385× clean**（98% 传导·填平 M6 10% 缺口）。**W4 measured-negative**：epilogue-spill 反噬。 | **missing_pattern→CLOSED**（array-util under-utilization=单累加器串行+A 重载·wide-tiling 填补）·byte-exact 三重证 | **正**：W2 = 各板 32-vreg budget 内最优·capability-keyed 选择正确（选 W2 非 W4）。**负边界**：W4 越 32-vreg 预算→epilogue 溢出→反噬（宽度旋钮单调只到 register-budget cap·同 A4 GEVM TG=2 的参数级镜像） | `g6-a-ime-perf-bridge/M7-vmadot-tiling/evidence.md §0/§6.1` · 468fcef7(发射器)/0ed58a5a(绿翻) |
| A2 | **IME array-util 杠杆的 format 键控**（M7 铺三格） | paired q4_0 / q8_0 / q4_K @ime（同 format-invariant vmadot leaf·compute-isolation ~1.56–1.59× 三格一致） | **SPLIT**：q8_0 int8-direct 轻 epilogue → e2e onw2/off **2.233× 真 beat stock**（无 IQR 重叠·**无内存税**）→ 绿 8→9/83；q4_0 nibble → **1.1385× tie-stock**（≥parity·内存税 ×1.78）→ 绿 8/83；q4_K super-block two-level fold（scale+min）重 epilogue → e2e **0.909× <parity** → 维持黄。 | **format-keyed applicability boundary**：compute-axis 三格一致 monotone·但 **e2e payoff = format-keyed on epilogue weight**（非 universal）·q4_K missing_pattern（需更深 epilogue 向量化发射器） | **正**：int8-direct / 轻 epilogue → array-util 传导（q8_0）。**负边界**：super-block / 重 two-level fold → array-util 增量被 epilogue 稀释到 e2e 1.024×（q4_K）→ **[PAT-1] format-keyed 适用边界·C3′ 负结果同价值** | `g6-a-ime-perf-bridge/M7-vmadot-tiling-q80-q4k/evidence.md` · 9acb6f3b/6af32c5a(q8_0 绿)/6af32c5a |
| A3 | **loop-schedule re-roll**（emit 形态·[GAP-EMIT-UNROLL]/[GAP-P1]） | on/off unrolled(全展开 2304 vwmacc/27KB) vs **rolled**(compact 16 vwmacc·`emit_loop_schedule` ODS 键) | **NOT-FLIP（回归）**：FORM seal 赢（vwmacc 2304→384·text −53.6%·byte-exact A/B 43008/43008）**但 perf 反向**——k1 e2e prefill rolled **0.406× vs unrolled 0.875×（慢 2.15×）**·rvv gcc-15 kernel-axis rolled 0.106× vs unrolled 0.390×。 | **misselection→board-falsified**（[GAP-P1] **re-roll trap 实证**）：rolled runtime loop 存 i16 partials 到 stack panel（324 vle16+373 vse16·spill 6→53·~1.8 mem-op/vwmacc）→ compute-bound prefill round-trip 主导·**瞄准瓶颈(code-volume/I-cache)非真瓶颈**（性能宪章规则2） | **正（form）**：rolled 赢 code-volume（I-cache 杠杆·gcc-15 下 −82.9% text 愈合爆炸）。**负边界（perf）**：code-volume 赢 ≠ perf 赢·真杠杆 = compact(16 vwmacc)∧register-resident(spill 4) **同时**·loop-schedule knob 拧不出（那是**结构级** KNEST 重构·非 schedule 参数·[K-10] 实证②） | `g6-b-emit-unroll/P2-q2k-k1-e2e/evidence.md` + `rvv-gcc15-kernel-axis/` · 6379bda2/80d18ad5/79b2ffe8 |
| A4 | **GEVM plan colgroup-tiled bank 加宽 TG**（结构级 [K-10]·[PAT-2] P9） | paired NEW colgroup-tiled **TG=2**（2 列组 state·register-resident bank）vs OLD GEMM-M=1 per-column vs STOCK | **NEGATIVE board-falsified**：decode M=1·IPC **NEW 0.214 < OLD 0.294 < STOCK 0.517**·e2e **0.336×**（比 OLD 0.49× **更慢=结构回归**）·字节 0.998× stock（H2 布局税仍 REJECTED·纯结构回归）。correctness GREEN A==B 3/3。 | **misselection→board-falsified**（[GAP-P1] register-pressure trap 同型·**结构级 register-budget 越界**）：scalar stack spill **534 vs OLD 10 = 53×**·TG=2 把 2 列组 state（4 f32m2 bank+2×权重条带+共享激活）叠 q4_K 超块重叶→爆 32-vreg→per-iter spill 回栈·M=1 无算力藏延迟→IPC 崩 | **正**：P1 构造 byte-exact mechanized·**plan 库 extensibility DEMONSTRATED**（[K-10] 结构级 novelty 真·独立 provenance/证书）。**负边界**：**结构对齐必要非充分·还须 register-budget-fit**——TG=2 bank 加宽在 M=1 memory-latency-bound 反噬·redesign=TG=1(软件流水+深 prefetch·减 live set) 或攻削 M=1 重建指令（bank 假设已证伪） | `g7-l1-gevm/P2-board-verify/evidence.md` + `raw/{new_gevmct.dis,old_gevm.dis}` · 67d49316（P1=806a7cc1） |
| A5 | **编译器身份 swap（gcc-death 隔离）** | paired 部署核 gcc-15(as-shipped) vs clang(重编·对称)·跨 q4_K/q2_K/q5_K 三格 | **gcc-death 必要非充分律（rvv 系统账 三格定律）**：q4_K clang **1.344× WIN**（spill 742→4·翻正反超）· q2_K clang **0.857× LOSS**（spill 2937→6 治愈**但未反超**）· q5_K clang **0.8688× LOSS**（spill 4124→155·未反超）。gcc full-unroll 病理：q2_K 27KB / q5_K spill 4124 e2e >10min 测不动。 | **misselection**（q4_K:治愈后反超）/ **missing_pattern**（q2_K/q5_K:治愈后仍慢·emitter-quality gap）：gcc-death 是 WIN 的**必要非充分**·消除后仍须 emitter-quality 达标（[CASE-COMPILER-ASYMMETRY]·[CASE-KQUANT-GCC-CODEGEN]） | **正**：gcc-death 消除 → q4_K 翻正反超 block-dot（emitter-quality 足）。**负边界**：q2_K/q5_K clang 治愈 gcc-death **但我方全展开核在 clang 部署域仍慢 stock ~14%**（跨双板一致 rvv 0.857×≈k1 0.873×·**非 gcc artifact·真 kernel-quality LOSS**）→ CLOSES rvv/VLEN128 fix-hypothesis | `l2-kquant-rvv-systemacct/{q2_K,q5_K}/evidence.md` · 14f4631a(q2_K)/e30c0055(q5_K) |
| A6 | **q4_K@k1 全展开 vs 真 hand-brick**（[GAP-KQUANT-VLEN256-UNROLL-VS-ROLLED]） | paired ours(全展开 25KB·VLEN-invariant) vs **真 hand-brick** `ggml_gemm_q4_K_16x1_q8_K`(compact rolled+vl=16·VLEN256 专调·clang-18 对称·N=12·3 seed·HOT+COLD) | **FALSIFIED**：ours/真-16x1 = **0.622× LOSS**（hand-brick 快 1.61×）。GMAC/s：block-dot 1.146 < ours 3.72 < 16x1 hand-brick 5.98。sealed 3.106× 的对手 = **block-dot 非 hand-brick**（三重证）。 | **missing_pattern**（[GAP-KQUANT-VLEN256-UNROLL-VS-ROLLED]）：ours 全展开体量赢 ≠ perf 赢·输 hand-brick 紧凑 rolled+满宽 VLEN256 专调（连 [GAP-EMIT-KNEST]/wide-LMUL 领域·非物理墙） | **正**：ours 3.106× vs block-dot（弱—中对手·kernel-sym 计入）。**负边界**：**vs 手调 hand-brick 输 0.622×**·hand-brick 可测（测了输了·非不可测）→ **kernel-sym 9 = 0 verified hand-brick**·两赛道成色质变均未达 | `g7-l2-kernelsym-hotcold/q4K-handbrick-resolve/evidence.md` + `summary_q4k_handbrick.csv` · b29c269c |
| A7 | **cache 温度（hot ≈ cold）** | paired hot vs cold(nr64/nr4·P=8 weight-tile POOL >> LLC·每 tile DRAM cold-read·N=12·k1 4 存量格) | **hot≈cold = NULL 区分**：q4_K 3.19↔3.18/2.83 · q5_K 1.92↔1.92/1.79 · q4_0 3.51↔3.50/3.44 · q8_0 1.48↔1.48。cache-cold **不改** kernel-axis 比值。 | **physical/机制**：compute(dequant/spill)-bound + L2(512KiB)<tile(576KiB)→缓存驻留仅二阶（"hot" 本非真 warm） | **正（cold-predictor 铁律）**：`cold ≥parity ⟹ 预测 e2e decode ≥parity` **仅在 micro=decode-GEVM ∧ 触 memory-wall**（q8_0 GEVM 1.73GB/s→命中 1.21×）。**负边界**：**cold-GEMM micro 禁当 decode 预测器**（q5_K cold-GEMM 1.79× WIN → e2e decode **0.729× LOSS**·双重错配 GEMM≠GEVM ∧ compute≠memory·决定性证伪）·q4_K/q4_0 方向命中=运气一致非机制捕获 | `T9_kernel_sym_ledger.md §6`（`be524605`） |

### 1.1 补充负例（[K-6] 归档·前 G7 / 跨战役·完整性）

| # | 消融维度 | 变体 | 结果 | 归因 | 边界 | 指针 |
|---|---|---|---|---|---|---|
| B1 | q6_K/q3_K S6 min-fold tiling（PAT-S6 XFER-1 class-2） | paired S6-tiled vs plain-untiled | **structural-NULL**：q6_K spill 913→949 **ROSE**·v31 无 cliff·~0.18× LOSS（2995 vsetivli weight-recon floor） | missing_pattern（weight-reconstruction-bound·S6 NULL·非 register-cliff 可闭） | **正**：q4_K/q2_K/q5_K min-fold class-1 HOLDS（spill→0/3/7·v30≤32 cliff）。**负边界**：no-min/weight-bound（q6_K/q3_K）class-2 NULL·即使循环形态键控闭合仍受权重重建 floor 封顶 | `T8` PAT-S6 XFER-1 7 行 |
| B2 | reduction deferred-vs-per-iter fold（winC） | on/off pass + 3-arm decomp | pass toggles 3.0–3.3×·**但 register-kept decomp = ≈1.00×（PURE-STRUCTURE NULL）**·3× 全在 OFF 臂 out[0] MEMORY round-trip | missing_pattern→structural-NULL（EMITTER ARTIFACT·非 reduction-structure-latency 优势） | **正**：pass 保留作 Win-A LMUL sweep enabler。**负边界**：禁 claim 结构 Win-C（over-optimism 纠偏典范） | `T3p_pattern_ablation.csv`（39eb9c48/e93a3b43） |
| B3 | P2 deferred-ordered-fold（q8_0·P2c） | on/off deferred vfredosum vs per-block round-trip + vs factory | deferred 23% 慢 factory·7.5% 慢 per-block·round-trip-elimination HYPOTHESIS **FALSIFIED**（vsetvli churn 19 vs 5 > round-trip saved） | physical（vsetvli churn 主导·dual-board 256/256 ULP0） | **正**：mechanized 构造 green-lit·numerics ULP0。**负边界**：优化 hypothesis measured-false = layer-4 design-space asset·非失败 brick | `experiments/ondevice-q8_0-deferred/fair/` |

**总表覆盖的消融维度（7 主 + 3 补 = 10）**：A1 IME tile 宽度参数级 · A2 IME array-util format 键控 · A3 re-roll trap · A4 GEVM TG=2 结构级 register-pressure · A5 gcc-death 必要非充分 · A6 hand-brick unroll-vs-rolled · A7 cold hot≈cold NULL / cold-predictor · B1 S6 weight-bound NULL · B2 winC structural-NULL · B3 P2c deferred-fold physical。

**★统一律（三级 register-budget-fit·横贯 A1/A4/A5/A6）**：M=1 memory-latency-bound decode 无算力藏延迟·**任一"必要条件"（结构对齐 A4 / 参数键控 A1 / 非病理编译器 A5 / 紧凑发射形态 A6）都是必要非充分·绑定的第二条件 = register-budget-fit**（spill 往返串行化流水→IPC 崩·GEVM TG=2 spill 53× / re-roll spill 6→53 / W4 epilogue-spill / hand-brick 需 register-resident）。

---

## 2. ★立柱对账（[T-PILLARS] 预注册 vs G7 measured）

> 逐条标 `{预测 · 实测 · 命中/证伪/边界}`。**不符行如实列出**（负例=边界·C3′ 素材·非粉饰）。对账结果 = 两柱**均命中但均需精化**（结构对齐/参数键控都是必要非充分·绑 register-budget-fit）。

### 2.1 柱一 · 结构对齐（plan 库 / 能力入模式库）

**预注册预测**：对手有折中/破损 → 我方结构对齐 → **倍数赢/正确性赢**；对手无折中（同拓扑专调）→ 我方 **parity~小赢**（同结构·拼参数级精调）。

| 对账点 | 预测 | 实测 | 判读 |
|---|---|---|---|
| **q4_K@rvv 系统账** | 折中格（gcc-death 破损 block-dot）→ 倍数赢 | **1.344× WIN**（clang 消 spill 742→4 反超） | **命中**（折中格倍数赢·conditional on gcc-death 消除∧emitter-quality 足） |
| **q2_K@rvv 系统账** | 折中格 → 倍数赢 | **0.857× LOSS**（clang 治 gcc-death 但全展开核仍慢 stock 14%） | **★边界**：折中格未必赢——我方**也缺结构对齐 plan**（KNEST 超块流式嵌套缺席）→ 折中不足以保证赢 |
| **★GEVM plan P2（结构对齐尝试）** | 结构对齐（独立 GEVM Emission Plan 接管 GEMM 兼职 GEVM 结构损）→ decode 转赢 | **0.336× e2e·IPC 反向 0.214<0.517·board-falsified** | **★边界（非全证伪）**——见下方专段 |
| **q4_K@k1 vs 真 hand-brick** | 无折中格（hand-brick compact rolled vl=16 VLEN256 专调）→ parity~小赢 | **0.622× LOSS**（输 hand-brick 1.61×） | **★边界**：parity 前提=我方**HAVE 结构匹配 plan**·此处我方=全展开 VLEN-invariant ≠ hand-brick 结构→缺 plan→无折中格也 LOSS |

**★GEVM P2 board-falsified 对柱一 = 证伪还是边界？→ 边界（柱一精化·非全证伪）**：
- **不是全证伪**，因为：① P1 构造 byte-exact mechanized，**plan 库 extensibility（柱一本体主张）DEMONSTRATED**——独立 GEVM Emission Plan 可造、独立 provenance/证书/roster、[K-10] 结构级 novelty 真；② P2 证伪的是**首版结构假设**（colgroup-tiled TG=2 bank 加宽），非柱一 thesis；③ redesign 路径（TG=1 软件流水 / 削 M=1 重建指令）在案。
- **是边界（精化）**：GEVM plan 是结构对齐尝试，但 M=1 register-pressure 反噬（TG=2 把 2 列组 state 叠 q4_K 超块重叶→爆 32-vreg→spill 53×→M=1 无算力藏延迟→IPC 崩）。**⟹ 柱一精化：结构对齐【必要非充分】·还须 register-budget-fit**（结构 plan 须适配目标板/regime 的寄存器预算）。这与 A5 gcc-death 必要非充分（编译器级）、A1 W4 over-budget（参数级）**同底约束的结构级镜像**。

**★KNEST 延迟检验点（q2_K/q6_K）状态 = OPEN / PENDING（prior 下调）**：
- 预注册：q2_K/q6_K 预挂"柱一缺口（KNEST plan 缺席）"·KNEST plan **若落地**该批格应转赢=柱一可证伪延迟检验。
- **现状 = 未检验**（KNEST plan **未落地**·[GAP-EMIT-KNEST] 正名为结构级 plan 缺席·与 GEVM plan 并列挂号·GEVM 完工后 [X-0] 评估）。
- **prior 下调**（两条间接负证据）：① **GEVM P2 先例**——plan 库首个结构 plan（GEVM）首版即 board-falsified（结构对齐必要非充分）→ 降低"KNEST plan 单独落地即翻正 q2_K/q6_K"的先验；② **G6-B P3 exit-b**——真杠杆 register-resident **IS expressible**（非框架不可能）但 nr knob throughput exit-b（best rolled 0.63× < shipped unrolled 1.00×）·且 q6_K/q3_K 另有 weight-reconstruction floor 双阻（B1）。**⟹ 延迟检验预期部分证伪风险高·但形式上仍 OPEN**（须 KNEST plan 真落地才裁）。

### 2.2 柱二 · 实例精调（能力键控 / 参数级迁移）

**预注册预测**：① 同格跨板换键不改条目（[PAT-3] CI 常绿）且各板取各自最优；② 键控杠杆消融对 on/off 单调正向（re-roll trap 除外=结构级误当参数级·非柱二反例）。

| 对账点 | 预测 | 实测 | 判读 |
|---|---|---|---|
| **M7 wide-vmadot W1→W2**（A1） | 键控杠杆 on 优于 off·单调正向 | W2 compute 1.589×·e2e 1.1385×·selector by 32-vreg 选 W2 | **命中**（capability-keyed 单调正向·各板取最优） |
| **M7 W4 越界**（A1） | （单调正向假设） | W4 measured-negative（epilogue-spill） | **★边界**：参数键控单调**只到 register-budget cap**·W4 越 32-vreg 预算反噬·selector 选 W2 不选 W4=键控正确 |
| **q5_K@k1 WIN vs q5_K@rvv LOSS**（[GAP-Q5K-VLEN128-QH-REGCLIFF]） | 各板取各自最优·跨板换键 | 同核 k1(VLEN256) **1.641× WIN** / rvv(VLEN128) **0.87× LOSS**·qh 5th-bit plane VLEN128 register-cliff（spill 155 vs q4_K 4） | **★边界**：emitter-quality-beat = **per-board（VLEN-dependent register-cliff）非 per-format 单值**·VLEN128 需 register-aware tile=柱二参数键控边界（同 plan·register 压力） |
| **IME array-util format 键控**（A2） | 键控杠杆单调正向 | compute-axis 三格 ~1.56–1.59× 一致·但 e2e 传导 format-keyed（q8_0 2.233× / q4_K 0.909×） | **★边界**：参数杠杆 compute-axis monotone·**e2e 传导 format-conditional**（on epilogue weight·非 universal） |
| **re-roll trap**（A3） | （预注册明示：非柱二反例） | rolled NOT-FLIP·结构级 KNEST 误当 schedule knob | **命中（分类正确）**：re-roll trap = 结构级误当参数级·[K-10] 实证②·验证柱二/柱一 taxonomy·正确排除 |
| **跨板换键不改条目** | [PAT-3] CI 常绿 | q4_0/q8_0@k1 sealed 参数级（VLEN-flip/item4）CI 常绿 | **命中·带边界**：q4_1/q5_0/q5_1 k1 半 pending net-new deploy=**结构缺口**（非换键·货架B）·换键律仅覆盖已键控参数 |

**柱二对账小结**：**命中·均带 register-budget / format-condition 边界**。参数键控在 register-budget cap 与 format-epilogue-weight 两处出现单调性边界（W4 越界 / q4_K 稀释 / q5_K per-board cliff），但 selector 的 capability-keyed 选择本身正确（选 W2·选各板最宽合法宽度）。

### 2.3 ★[T-PILLARS] 更新建议（改留主会话·本表不改 T-PILLARS）

1. **柱一精化条款**（加于 T-PILLARS §柱一预测后）：*"结构对齐是 M=1 decode 转赢的**必要非充分**条件·还须 **register-budget-fit**（结构 plan 须适配目标板/regime 寄存器预算·否则 spill 反噬 IPC）。证据=GEVM plan P2 board-falsified（TG=2 spill 53×·67d49316）。此与 gcc-death 必要非充分（编译器级·A5）、W4 over-budget（参数级·A1）同底约束的结构级镜像。"*
2. **柱一"无折中→parity"精化**：*"parity 前提=我方 HAVE 结构匹配 plan；我方缺匹配结构（如 q4_K@k1 全展开 vs hand-brick compact rolled）时·无折中格也 LOSS（0.622×·b29c269c）。"*
3. **KNEST 延迟检验点状态更新**：由"预挂"→"OPEN·prior 下调"（GEVM P2 先例 + G6-B P3 exit-b 两条间接负证据·延迟检验预期部分证伪风险高·仍待 KNEST plan 真落地才裁）。
4. **柱二边界补注**：W4 over-budget（A1）、q5_K per-board register-cliff（[GAP-Q5K-VLEN128-QH-REGCLIFF]）、IME format-keyed e2e 传导（A2）三处参数键控单调性边界入柱二"负例=边界"清单。

---

## 3. micro↛e2e 病例卷宗（指针归拢·[[kernel-wins-dont-transplant-to-e2e]] 谱系）

> **卷宗主论点**：**compute-bound / byte-correct / form-compact 的 micro 胜，一致地 NOT 传导到 memory-latency-bound M=1 decode e2e**；**传导与否取决于 kernel 是否主导 e2e 时间**（桥慢主导→传导 / kernel 小占比→wash）或**是否改变 memory 行为**（repack 布局/locality）。**永远 kernel-micro 与 llama e2e 分开报**。

| 病例 | micro 胜 | e2e 实际 | 传导判别 | 指针 |
|---|---|---|---|---|
| **① IME compute↛decode**（历史 canon） | IME matmul **5.51× kernel**（256³ int8·真 K1） | decode **0.86–0.98×**（GEVM M=1·矩阵单元帮不上） | wash（kernel 小占比·memory-bound decode） | [[kernel-wins-dont-transplant-to-e2e]] |
| **② Win-A LMUL 调**（历史 canon） | decode **2.1× kernel** | decode **FLAT**（prefill 1.70× holds·compute-bound） | wash（decode memory-bound）/ prefill 传导 | 同上 |
| **③ cold-predictor 铁律**（G7 T9 §6） | q5_K cold-GEMM micro **1.79× WIN** | e2e decode **0.729× LOSS** | **决定性证伪**：cold-GEMM≠decode 预测·双重错配 GEMM≠GEVM ∧ compute≠memory·仅 decode-GEVM∧memory-wall 机制成立（q8_0 GEVM 1.73GB/s→1.21× 命中） | `T9 §6`（be524605·本表 A7） |
| **④ IME 桥主导→干净传导**（G6-A M2·传导律精化） | 桥慢 stock 17.8×·多线程 kernel 3.886× | e2e **3.886× 干净传导** | **传导（对照 wash）**：桥慢主导 e2e→并行直接兑现·**传导与否取决于 kernel 是否主导 e2e** | `g6-a-ime-perf-bridge/M2-multithread/`（8156e017） |
| **⑤ GEVM micro-byte↛decode-IPC**（G7 P2·最新病例） | P1 byte-exact 结构 novelty·同 DRAM 字节 0.998× stock | e2e decode **0.336×**·IPC 反向 0.214<0.517 | **证伪**：micro-byte-correctness 不预测 decode-IPC·M=1 register-pressure spill 53× 崩 IPC（字节触底 roofline·杀伤全在结构效率） | `g7-l1-gevm/P2-board-verify/`（67d49316·本表 A4） |

**传导的 e2e 胜（改 memory 行为·反面参照）**：Win-B repack **prefill ~6×**（连续 16-blocks-as-lanes 布局流权重）· q4_0 GEVM **1.22× compute → 2.6× e2e decode**（纯 memory-locality 随模型规模 GROWS·最干净支持）· VLEN-strip **1.31× e2e decode**。

**卷宗律（durable·G7 精化）**：micro↛e2e **不是绝对**——G6-A M2 证 **kernel 主导 e2e 时干净传导**（桥慢 17.8×）；反之 kernel 小占比（IME leaf ~2%）或 M=1 decode（无算力藏延迟）则 wash。判别键 = **kernel 占 e2e 时间比 × regime(prefill compute / decode memory-wall)**。

---

## 4. ★性能立柱提炼（北极星第④步·诚实方法学立柱）

> **G7 诚实综合**：**结构 novelty 造得出**（GEVM plan 库 extensibility·[K-10] 结构级·byte-exact 三证·[PAT-1] IME array-util·KNEST 正名）· **但 M=1 memory-latency-bound decode perf 兑现难**（register-pressure 反噬 / 无算力藏延迟 / 权重重建 floor / 紧凑∧驻留发射形态产不出）· **两轴无赢手调**（perf-covered 9/83 + kernel-sym 9 均无一格赢 hand-brick/vendor IME）。**perf 兑现难时·方法学是 G7 的真价值**。

### 4.1 方法学立柱（三支柱·可复用·C3′ 模板产出质量·非性能数字）

**立柱 = 「能力键控优化模式库 + 缺口闭环 + 诚实边界」**——这是 C3′（模板产出质量）作为**方法学**的陈述，即使 perf 兑现难仍可辩护：

1. **能力键控优化模式库（capability-keyed pattern library）**：[PAT-1] wide-vmadot-tiling（IME array-util·capability-keyed by 32-vreg budget）· [PAT-2]/P9 GEVM Emission Plan（[K-10] 结构级·独立 plan）· PAT-S6 output-tiling（register-cliff 键控）· [K-10] 结构级/参数级判据。模式由**机制选出**（归因日志·非 workload-name 路由）·**换键不改条目**迁移（[PAT-3] CI 常绿）。**plan 库 extensibility 是 DEMONSTRATED**（GEVM P1 首格 byte-exact·可造独立 plan）——即便某 plan 首版 perf board-falsified，extensibility 本体（能造、能证、能挂号）成立。

2. **缺口闭环（[K-6] gap-closure loop·[GAP-1]）**：每个 perf 落败/意外持平 triage 为四选一 `{缺能力事实 | 缺模式 | 选择错误 | 物理(带宽墙)}`·反汇编证据（[L-7]）·缺模式/缺事实者命名进 schema/[PAT-1] 注册表→关闭→同核同板复测→归因日志引 gap-ID。**本 T3p 表的负结果档案本身 = C3′ 叙事素材**（负例→可归因·如 GEVM P2→register-budget-fit 精化 / re-roll trap→[GAP-P1] / hand-brick→[GAP-KQUANT-VLEN256-UNROLL-VS-ROLLED]）。

3. **诚实边界（正例负例边界齐备·[K-6] 体例）**：每个模式带其 design-space 边界——W2 正/W4 越界（A1）· int8-direct 传导/super-block 稀释（A2）· 结构对齐必要非充分+register-budget-fit（A4/柱一）· gcc-death 必要非充分（A5）· cold-predictor 仅 decode-GEVM∧memory-wall（A7）· micro↛e2e 除非 kernel 主导 e2e（§3）。**边界可预测·迁移可预测 = 模板产出质量的实测背书**。

### 4.2 诚实边界声明（不粉饰·四条钉死）

- **两轴成色质变均未达**：perf-covered 9/83（系统账 e2e ≥parity）+ kernel-sym 9（内核轴对称·**0 verified hand-brick**）·**无一格硬碰硬赢手调对手**（hand-brick/vendor IME 全赢我方·q4_K@k1 vs hand-brick 0.622× / IME 我方 0.505× vendor）。G6/G7 "成色质变=赢手调对手成排" **两轴皆未达**。
- **计数 vs 成色分开报**：M7 铺 IME +2 格（→9/83）但成色质变没到（q8_0 真 beat 的 vendor q8_0 路弱 0.984× 非强手调·q4_0 仅 tie）。**计数动了·碾压强手调没到**。
- **结构造得出 ≠ 性能立得住**：GEVM plan/KNEST 结构 novelty 真·但 M=1 decode 兑现受 register-budget / weight-recon floor / 无算力藏延迟三重物理约束封顶。
- **micro↛e2e**：kernel/byte/form micro 胜不传导 memory-bound M=1 decode·除非 kernel 主导 e2e 或改 memory 行为（§3 卷宗）。

### 4.3 durable 律（G7 落卷·可复用）

1. **三级 register-budget-fit**：结构对齐（A4）/ 参数键控（A1）/ 非病理编译器（A5）/ 紧凑发射形态（A6）都必要非充分·绑 register-budget-fit（M=1 无算力藏延迟·spill 崩 IPC）。
2. **传导律**：micro→e2e 传导取决于 kernel 是否主导 e2e（桥慢主导→传导 / kernel 小占比→wash）。
3. **系统账 WIN 双条件**：gcc-death 消除 ∧ emitter-quality-beat（q4_K rvv WIN / q2_K/q5_K rvv LOSS 同构印证·A5）。
4. **array-util e2e payoff = format-keyed on epilogue weight**（int8-direct 传导 / super-block heavy-fold 稀释·A2）。
5. **cold-GEMM micro 禁当 decode 预测器**（仅 decode-GEVM∧memory-wall·A7）。

---

## 关联

- **消融**：本表 §1（G7 七维 + 3 补）· `T3p_pattern_ablation.csv`（G7 前 P1/P2/P4/P7/winC 模式消融·in-tree 结构轴）· `T4b_selector_ablation.csv`（选择器四配置消融·需板·pending 不做）。
- **立柱**：`T-PILLARS_perf-pillars-preregistration.md`（预注册·§2 对账·更新建议 §2.3 改留主会话）· [K-10]（结构级/参数级判据）· [K-6]（缺口闭环）。
- **账本**：`perf-covered-category.v1.json`（9/83 系统账·recon 权威）· `T9_kernel_sym_ledger.md`（kernel-sym 9·§6 cold）· §5 dual-board 矩阵。**三账互不推**。
- **memory**：[[measurement-offensive-perf-covered]]（perf-covered 头条）· [[g6-perf-campaign-ime-bridge-emit-lever]]（G6 findings）· [[kernel-wins-dont-transplant-to-e2e]]（micro↛e2e）· [[q4-0-e2e-is-routing-not-kernel]]（gcc-death）· [[zero-model-adjudication-cert-hardening]]（correctness 硬门）。
