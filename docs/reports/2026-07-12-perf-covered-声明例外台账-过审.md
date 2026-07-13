# perf-covered 声明例外台账（过审版）— 27 格逐格五字段 + anti-gate 自证 + 物理墙 roofline 指针

> 生成 2026-07-12 · 触碰集 = docs/tooling 域（只读 roster/coverage 脚本 + 三 casefile + 三具名 GAP 报告，写本报告 + `schema/perf-covered-category.v1.json` 标签 + `.trellis/scripts/perf_covered_metrics.py` recon）。**未跑板·未 git·未改 ODS/lib/·未改 ROADMAP/结构 roster/coverage_metrics.py。**
> **性质**：用户 2026-07-12 裁决**三（声明例外台账过审）**的落地。把 perf-covered 分类中的 **27 格声明例外**从"分类册一行注记"firm 成**逐格五字段台账**（不做理由 · Amdahl 上限数字/占比 · 占比来源 · 重估条件 · 重估钩子），机检强制字段完整（`perf_covered_metrics.py` reconciliation·缺字段 = CI 红），并给 **anti-gate 自证**（声明例外判定与黄格分母无关）。
> **口径**：`perf-covered = 绿 / denom(83·fold q4_0-gemm regime·any-board)`。声明例外 **在分母内**（是 certified-but-off-hot-path 格·非域外·非未构造）。数据源 = `2026-07-12-perf-covered-零未定义格分类.md` §1.3/1.5/1.6 + 铺面③ casefile `experiments/active/covering-batch3-stream-rvv/` + `2026-07-12-三具名GAP修复评估.md`。
>
> **★★2026-07-13 更新（G7 L0.2 anti-gate 抽查追认 + 例外池扩至 49）**：
> 1. **★anti-gate 抽查 5/5 PASS·例外池整体追认**（recon `random.seed(20260713)` 抽 decode/GEVM 3 + IQ/TQ 2 = {vec_dot/q3_K, vec_dot/q4_K, rope/f32, gemm_tile/iq2_s@rvv, dequantize_row/iq4_xs}·逐格三项 {Amdahl profile 实/结构声明·roster 热流量 有·对手身份 属实} 全绿·无翻案）。**核心结论：分母无灌水**（49 例外全在分母内·移出反而抬高 headline 9/83→9/34·例外只压低不抬高·无灌水动机）。
> 2. **★本文档仅覆盖 27 格·晚增 22 格（vec_dot 15 + gemm_tile iq/tq 7·2026-07-12→07-13 黄-未接线全迁入声明例外·headline 不变·裁决驱动）无人读台账行 → 以 `schema/perf-covered-category.v1.json`（机检强制五字段·全 49 完整）为权威·22 格新增见 `2026-07-12-perf-covered-零未定义格分类.md` 分类报告**。人读全 49 台账补版待排。
> 3. **★测量债周期复核点（附注B）**：`gemm_tile/iq2_s@rvv` 的"opponent-absent"与已引 iq4_nl 0.217× 实测损失锚存在张力（严格读可 morphology-declare 为"对手更强"）。项目纪律辩护 = **"对手更强"判据须真测量**（未测格不外推损失·暂泊例外挂 reestimate_hook）·此为一致纪律非低估。但 **"0 未接线"是强主张·须周期性复核测量债兑现**（板批填充时优先测这批 morphology-declared 例外格·验证是否该转"对手更强"）。

---

## 0. 台账口径（读死）

- **声明例外定义**（六类冻结集之一）：分母内、certified（byte-exact 构造）、但**无公平对手** OR **Amdahl e2e 上限 < 噪声地板**（off-hot-path 结构事实）→ 不以 perf 名义立项、只以机制/方法学名义存在。
- **noise floor**：~1%（`T-N_noise_floor.csv`·自检 IQR 0.14–0.19%·门 = hist-floor×1.5）。所有声明例外 Amdahl 上限 << 此。
- **同域纪律（[CASE-COMPILER-ASYMMETRY]）**：Amdahl 传导估算的 kernel 因子输入必须与目标部署【同域】（部署 = clang-deploy·非 gcc-micro）。iq4_nl 用 clang-deploy 0.94× 喂 Amdahl（非 gcc 4.83×），否则 garbage-in。
- **27 = 19 dequant（super-block/codebook/fp4/ternary/binary）+ 3 product_reduce + 5 逐元素/norm（add/mul/rms_norm/softmax/rope）**。

---

## 1. 声明例外 27 格逐格台账（五字段·机检字段完整）

> 每格五字段 = `perf_covered_metrics.py` 的 `declared_exception` block 必填项 {reason / amdahl_ceiling / ceiling_source / reestimate_condition / reestimate_hook}；缺任一 → reconciliation FAIL（CI 红）。下表即该 block 的人读投影。

### 1.1 dequantize_row super-block（5 格·Amdahl <0.15%）

| # | 格 | 不做理由 | Amdahl 上限 | 占比来源 | 重估条件 | 重估钩子 |
|---|---|---|---|---|---|---|
| 1 | dequantize_row/q4_K | super-block dequant·off 热 matmul 路（q4_K matmul 走 block-dot/repack GEMM·非 row-dequant）·可修黄格但修完 e2e-inert（诚实反例·LAW 第5例） | **<0.15%** | [GAP-DEQ-KQUANT-UNPACK] 三具名GAP §2·**q4_K 实测锚** kernel 0.636×/deploy 0.879×·mem-bound scale-unpack 3.0 GB/s·<1% × (1−1/1.2) | dequant 进 matmul 热路（get_rows fraction 上升过噪声地板 ~1%） | T6 e2e phase-split 实测 dequant fraction·或 K-quant 流式族 byte-exact 成熟度整批推进复用向量化-解码 helper |
| 2 | dequantize_row/q2_K | 同 super-block·morphology-declared | <0.15% | [GAP-DEQ-KQUANT-UNPACK] 族级·q4_K 实测锚 morphology-declared·dispatch-definitional off-hot-path | 同上 | 同上 |
| 3 | dequantize_row/q3_K | 同 super-block·morphology-declared | <0.15% | [GAP-DEQ-KQUANT-UNPACK] 族级·morphology-declared·off-hot-path | 同上 | 同上 |
| 4 | dequantize_row/q5_K | 同 super-block（q4_K sibling·共享向量化-解码族级杠杆） | <0.15% | [GAP-DEQ-KQUANT-UNPACK] 族级·morphology-declared | 同上 | 同上 |
| 5 | dequantize_row/q6_K | 同 super-block·morphology-declared | <0.15% | [GAP-DEQ-KQUANT-UNPACK] 族级·morphology-declared·off-hot-path | 同上 | 同上 |

### 1.2 dequantize_row codebook/fp4（11 格·Amdahl <0.4%）

| # | 格 | 不做理由 | Amdahl 上限 | 占比来源 | 重估条件 | 重估钩子 |
|---|---|---|---|---|---|---|
| 6 | dequantize_row/iq4_nl | codebook 族·off 热 matmul 路·**真赢潜伏在分离热 vec_dot iq 族**（非本冷叶·gather-trap 冷叶 e2e-null·LAW 第6例·新第5根因类） | **<0.4%** | [GAP-CLANG-GATHER-TRAP] 三具名GAP §1·**iq4_nl 实测锚** deploy 0.94×/kernel 4.83× latent·★同域纪律用 clang-deploy 喂 Amdahl·<0.5% × (1−1/5)≈0.4% | gather-batching 落地且外推到热 vec_dot iq 族·或 iq 模型成主流部署 | uarch.gather_slow scalar-pin 杠杆板测·或 vec_dot iq 族 gather-batching 立项（新立项须问·队列外） |
| 7 | dequantize_row/iq1_s | codebook/grid 族·off 热路·morphology-declared | <0.4% | [GAP-CLANG-GATHER-TRAP] 族级·iq4_nl 实测锚 morphology-declared | 同 #6 | 同 #6 |
| 8 | dequantize_row/iq1_m | 同 codebook·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 9 | dequantize_row/iq2_xxs | 同 codebook·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 10 | dequantize_row/iq2_xs | 同 codebook·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 11 | dequantize_row/iq2_s | 同 codebook·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 12 | dequantize_row/iq3_xxs | 同 codebook·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 13 | dequantize_row/iq3_s | 同 codebook·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 14 | dequantize_row/iq4_xs | 同 codebook（iq4_nl sibling）·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 15 | dequantize_row/mxfp4 | fp4 族·codebook gather-trap 共享·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |
| 16 | dequantize_row/nvfp4 | fp4 族（同 mxfp4）·morphology-declared | <0.4% | 同上 | 同 #6 | 同 #6 |

### 1.3 dequantize_row ternary/binary（3 格·Amdahl <0.4%）

| # | 格 | 不做理由 | Amdahl 上限 | 占比来源 | 重估条件 | 重估钩子 |
|---|---|---|---|---|---|---|
| 17 | dequantize_row/tq1_0 | ternary 族·off 热路·morphology-declared·ternary 少数部署 | <0.4% | codebook/grid 律 morphology-declared·dispatch-definitional off-hot-path（分类报告 §1.3） | ternary 成主流部署且 dequant 进热路 | ternary dequant/vec_dot 板测 |
| 18 | dequantize_row/tq2_0 | 同 ternary·morphology-declared | <0.4% | 同上 | 同上 | 同上 |
| 19 | dequantize_row/q1_0 | binary 族·off 热路·morphology-declared | <0.4% | morphology-declared·off-hot-path·**q1_0 dequant 实跑 byte-exact 460800 元素 0-mismatch** | binary 成主流部署且 dequant 进热路 | q1_0 dequant/vec_dot 板测 |

### 1.4 product_reduce（3 格·无框架对手）

| # | 格 | 不做理由 | Amdahl 上限 | 占比来源 | 重估条件 | 重估钩子 |
|---|---|---|---|---|---|---|
| 20 | product_reduce/q4_0_nibble | 内部子原语（喂 dequant/vec_dot 复合）·**无框架-kernel 公平对手**（ggml 无 standalone product_reduce·vs-scalar 10.8× 是 sanity 层非 framework）·perf 由复合格捕获 | **N/A**（无独立 e2e·无公平探针·结构无对手） | [L-6] vs-naive ≠ vs-framework·分类报告 §1.5 | ggml 建 standalone product_reduce 参考层（框架对手出现） | 复合格（dequant/vec_dot）测量捕获·或框架对手出现则独立测 |
| 21 | product_reduce/offset_binary_n3 | 同上·内部原语·无框架对手 | N/A | 同上 | 同上 | 同上 |
| 22 | product_reduce/codebook_n3 | 同上·内部原语·无框架对手 | N/A | 同上 | 同上 | 同上 |

### 1.5 逐元素/norm（5 格·Amdahl <0.6% 或 结构判 <噪声）

| # | 格 | 不做理由 | Amdahl 上限 | 占比来源 | 重估条件 | 重估钩子 |
|---|---|---|---|---|---|---|
| 23 | add/f32 | 逐元素 residual·off 热路·near-wall memory-bound·未接 e2e·机制未证（无 objdump·违宪章规则1·提交前阻塞·第二 micro↛e2e 教材） | **<0.6%** | [GAP-FWD-M8-VSETVL] 三具名GAP §3·deploy 0.83×/kernel 0.81× near-wall 7.3 GB/s·0.01–0.03 × (1−1/1.25) | objdump-first 认真瓶颈（LMUL vs vsetvl vs memory-scheduling）+ 接 e2e | add/mul objdump 两 .o + e2e wiring（提交前 objdump-first gate） |
| 24 | mul/f32 | 逐元素 gating·同 add·双账本收敛 LOSS·无 compiler-artifact | <0.6% | [GAP-FWD-M8-VSETVL] 三具名GAP §3·deploy 0.76×/kernel 0.76× near-wall 6.4 GB/s | 同 #23 | 同 #23 |
| 25 | rms_norm/f32 | norm off 热路·reduction·铺面③ 未覆盖·结构判 | **<<0.05%** | G2 fusion e2e null 实证·norm 占 decode 0.05% × 融合改善 → +0.05% < 噪声地板 | norm fraction 上升过噪声地板 | T6 e2e phase-split norm fraction 实测 |
| 26 | softmax/f32 | off 热路·cheap reduction·铺面③ 声明 separate shape out-of-map class·未测·结构判 | <噪声（结构判·精确 fraction 未测） | dispatch-definitional off-hot-path·铺面③ morphology 外提（分类报告 §1.6） | softmax fraction 上升过噪声地板（长 context attention 主导） | T6 e2e phase-split softmax fraction 实测 |
| 27 | rope/f32 | off 热路·cheap rotate·同 softmax·未测·结构判 | <噪声（结构判·精确 fraction 未测） | dispatch-definitional off-hot-path（分类报告 §1.6） | rope fraction 上升过噪声地板 | T6 e2e phase-split rope fraction 实测 |

**逐格五字段完整性 = 机检**：`python3 .trellis/scripts/perf_covered_metrics.py report` → `reconciliation.incomplete_field_cells == []`（27 格 declared_exception block 每格五字段齐·缺 = CI 红）。

---

## 2. anti-gate 自证（声明例外判定与黄格分母无关 — 用户裁三）

**主张**：声明例外的判定依据 = **Amdahl<噪声（off-hot-path 结构事实）** / 无公平对手，**与"想让 headline 好看"无关**。

**机检形式（`perf_covered_metrics.py` 的 `anti_gate` block·三条同时成立）**：

1. **分子 numerator-independent**：headline 分子 = **绿格数**。声明例外 cell 移入/移出**只改分母、不改分子**。机检：`green_with_exception (=6) == green_without_exception (=6)`——把全部 27 个声明例外 cell 从分母移除，绿分子恒 = 6。
2. **例外全 certified-but-off-hot-path**：27 个声明例外 cell **全部 ∈ certified 集**（byte-exact 构造·非未构造/未接线）。机检：`all_exceptions_certified == True`。它们是"造得出但 off-hot-path"，不是"造不出所以塞进例外"。
3. **判定钩子非分母**：每格 `reestimate_condition` = "fraction 上升过噪声地板"（结构事实），`ceiling_source` = 具名 GAP/实测锚/dispatch-definitional，**无一条以"分母大小/headline 分数"为判据**。

**⇒ 声明例外不能被用来粉饰 headline**：例外增减只动分母（分母越大 headline 分数越小·移出例外反而【抬高】headline），且例外都是 certified-off-hot-path 而非"未构造伪装成例外"。这与"逐出 IME 3 格才到 90.9% 但明文拒绝逐出"的 M4 anti-gate 同构（`2026-07-11-M4-三分类终态全表.md` §判定书 anti-gate）。

**机检输出**（HEAD 快照·`report` 粘贴）：
```
anti_gate_ok: True | green_with_exc= 6  green_without_exc= 6  exc_cells= 27
note: 声明例外 cell 判定依据 = Amdahl<噪声（off-hot-path 结构事实）·与 headline 分子无关:
      分子只数绿·例外增减仅动分母·且例外全 certified-but-off-hot-path（非未构造）
```

---

## 3. 黄-物理墙 12 格逐格 roofline 证据指针（用户裁三·逐格挂 roofline）

> 三要求：逐格挂 roofline 证据指针 + 标注【实测 or morphology-declared】。铺面③ casefile 4 格实测锚 + 8 格 morphology-declared 传播（casefile 明示 same-family declared-covered）。

| # | 格 | roofline 证据指针 | 实测/morphology | 比值（deploy / kernel） |
|---|---|---|---|---|
| 1 | dequantize_row/q8_0 | covering-batch3/raw_board_results.txt·mem-bound scalar **3.7 GB/s** roofline·byte-exact maxrel0 | **实测** | 0.98× / 0.86× |
| 2 | dequantize_row/q4_0 | flat 族·q8_0 实测锚 3.7 GB/s roofline·同 flat/scalar 律·parity 类推（分类报告 §1.3） | morphology-declared | — |
| 3 | dequantize_row/q4_1 | 同 flat 族·q8_0 实测锚·morphology-declared | morphology-declared | — |
| 4 | dequantize_row/q5_0 | 同 flat 族·q8_0 实测锚·morphology-declared | morphology-declared | — |
| 5 | dequantize_row/q5_1 | 同 flat 族·q8_0 实测锚·morphology-declared | morphology-declared | — |
| 6 | quantize_row/q8_0 | covering-batch3·mem-bound **2.8 GB/s**·byte-exact diff0·激活量化 | **实测** | 0.99× / 1.00× |
| 7 | quantize_row/q8_K | covering-batch3·mem-bound **2.5 GB/s**·byte-exact | **实测** | 1.08× / 0.98× |
| 8 | quantize_row/q8_1 | q8-family scale+int8 store·at-wall parity 预期·correctness-essential（wired 进绿 q5_1/q4_1 激活路·分类报告 §1.4） | morphology-declared | — |
| 9 | gelu/f32 | covering-batch3·compute-bound tanhf **0.11 GB/s**·maxrel 1.3e-6·LUT-deploy caveat 记录 | **实测** | 0.99× / 1.00× |
| 10 | scale/f32 | trivial memory-bound map·at-wall·parity 类推（分类报告 §1.6） | morphology-declared | — |
| 11 | cpy/f32 | trivial map·at-wall·parity 类推（分类报告 §1.6） | morphology-declared | — |
| 12 | silu/f32 | gelu scalar-transcendental 类·at-wall·parity 类推（分类报告 §1.6） | morphology-declared | — |

**物理墙 = 满分非 GAP**：at-wall parity（对手也贴同一 roofline）·byte-exact·非 loss·非 emitter-maturity soft-gap（区别于声明例外的 super-block/codebook named-gap，那些是可修黄格但 Amdahl<噪声）。机检：`perf_covered_metrics.py` 的 physical_wall block 每格 {roofline_pointer / evidence} 齐（缺 = CI 红）。

---

## 4. check_docs_canon 自检 + 交叉引用

- 本报告 = `docs/reports/2026-07-12-perf-covered-声明例外台账-过审.md`·带 `YYYY-MM-DD-` 前缀（reports/ append-only 合规·非 canon/ 越界）→ PASS 预期。
- **未改**：结构 roster `coverage-roster.v1.json` / `coverage_metrics.py`（M4 84/91 不动）· ROADMAP · lib/ · ODS · board。**未 git·未跑板。**
- **机检基座**：`python3 .trellis/scripts/perf_covered_metrics.py report`（roster_sha256 `2704968065bd…`·labels_sha256 `f6a0dee53ddd…`·reconciliation_ok=True·anti_gate_ok=True）。
- **交叉引用**：`2026-07-12-perf-covered-零未定义格分类.md`（supersede 源·§1.3/1.5/1.6）·`2026-07-12-perf-covered-roster-unification-recon.md`（fold 决策 + 目标数 6/83）·`2026-07-12-三具名GAP修复评估.md`（3 GAP Amdahl<噪声）·`experiments/active/covering-batch3-stream-rvv/`（铺面③ 4 实测锚 + morphology-declared）·`2026-07-11-M4-三分类终态全表.md`（M4 anti-gate 同构）·`schema/perf-covered-category.v1.json`（标签工件）·`.trellis/scripts/perf_covered_metrics.py`（recon 脚本）·`docs/method/LAW-FIRST-EMISSION.md`（第5/6例）·memory `measurement-offensive-perf-covered` / `kernel-wins-dont-transplant-to-e2e`。
