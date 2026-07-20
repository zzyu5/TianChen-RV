# 性能诚实盘点 · 成色分层 Scorecard（只读产出）

> 任务：`.trellis/tasks/07-20-07-20-perf-honest-scorecard/prd.md`（权威）。服务用户令「性能也很重要·都要搞清楚」。
> 铁律：0 造数（数字全来自主表机读 + 度量脚本机算·禁手抄）· 便宜档禁称硬赢 · measured-非-T-N 明标 · 未改任何源 · 未 commit。
> 快照：`repo d3c7bb88`（脚本 `$meta.repo_snapshot`）· 主表 `experiments/master/T3_master_rebuild.csv`（108 数据行·row2 起）。

---

## 〇、权威机算数（脚本产出·非手抄）

| 指标 | 值 | 来源 |
|---|---|---|
| **perf-covered** | **9 / 83 = 10.84%**（any-board·fold_regime=True 唯一发布口径） | `python3 .trellis/scripts/perf_covered_metrics.py report` → `headline.perf_covered` |
| perf-covered 分类和 | 绿 9 / 黄-传导稀释 1 / 黄-物理墙 12 / 黄-对手更强 12 / 声明例外 49 = 83 | 同上 `classification`（reconciliation_ok=True·三源一致 9==9==9） |
| C_dispatch | 102 / 108 = 94.44% | `coverage_metrics.py report` → `metrics.global` |
| C_construct | 101 / 108 = 93.52% | 同上 |
| M4 三分类 | certified 101 · blocked_on_IME 0 · 声明例外 7 · 域外 2（roster 110） | 同上 `m4_classification`（reconciliation_ok=True） |

**关键区分**：`perf-covered 9/83` 是**系统账/e2e 的「公平协议≥parity 或赢」占比**（fold regime）；C_construct 101/108 是**结构造得出**占比。二者都**不是**「硬碰硬赢强手调对手」的成色数——那个数见下 §一（=**1 格**）。**结构造得出 ≠ 性能立得住**（成色 canon 定调锁）。

---

## 一、分层分布（逐板-相 cell·机算·成色 canon 判据）

判据正本：`成色与措辞.md`[L-1/L-6/L-12] · `对手与档位.md`（三档：手调 / 通用向量 / 标量类；只有 **factory-dispatched 手调 STRONG** 计 beat；CROSSOP=跨算子不计同算子赢；near-parity 0.8–1.0=tie 非 beat）· `测量判据.md`§二.5（**无 T-N = 无判定资格**）。

统计单位 = 主表每个「(op,format,engine,regime) × 板」且**带比值**的 cell（跳过 N/A-hw / 域外-q1_0 / pending-fold = 19 格）。优先级判层规则见文末〔附·判层规则〕。

| 层 | 定义 | cell 数 |
|---|---|---:|
| **A 真硬赢** | beat **STRONG 手调**（factory hand-brick）· 非 CROSSOP · byte-exact · 编译器对称 · ≥parity | **3**（=2 格式：q4_K@k1 decode+prefill · q2_K@k1 prefill） |
| **B 便宜档** | 赢**弱**对手（标量类/scalar-ref · 通用 block-dot path-win · 弱 vendor）· opp-immaturity · **禁称硬赢** | **58** |
| **C 部署变更** | de-lottery（baseline<0.8→owned PASS·vs codegen 抽签·非 vs 对手）· measured-gate LMUL flip | **18** |
| **D 跨算子/near-parity/弱势/败** | CROSSOP · tie(0.8–1.0) · loss(<0.8·具名-X) · 对强手调 vec_dot 全败 · repack-DEFERRED · forward parity | **115** |
| **E measured-非-T-N** | r5.1 grid dequant flip·走 ad-hoc run_*.sh·bench 通道零行·无 T-N | **3**（iq2_xs 3.88 · iq3_xxs 1.39 · iq3_s 1.71 @rvv） |
| pending/NA | 无比值·N/A-hw·域外·pending-fold | 19 |

**横切旗标（不并入上表·防重复计数·但同等重要）：**
- **measured-非-T-N 横切 = 46 格**（全 dequant 24×2 中带比值者 + r5.1 gemm fixtures）。CLAUDE.md:35 现行法：**dequant T-N 命中 0**；grid/repack 真数走 ad-hoc `run_*.sh`·`runs.log` 07-19/20 零行 = **全 dequant 与 r5.1 批 = measured 非 T-N-qualified·论文引用前须走 bench 通道复测**。E 层只是其中「以新 perf 倍数为 headline」的 3 格；**C 层 17 格 de-lottery dequant 同样 measured-非-T-N**（成色注 = 便宜档·qualification 注 = 非-T-N）。
- **de-lottery 横切 = 17 格**（C 层主体）。

---

## 二、A 层 · 真硬赢逐格（唯一可当论文 headline 的层）

项目自锁：主表 row73/74 明文「**真硬赢锁 2 = q4_K/q2_K**」（q5_K 8x8 前提证伪·非 verified hand-brick·F-2 纠）。机算与此一致。

| # | cell | ratio | 对手身份（STRONG） | 门/成色 |
|---|---|---|---|---|
| A1 | **gemm_tile/q4_K@k1 decode** | **1.535** | `ggml_gemm_q4_K_16x1_q8_K`(REAL 16x1 hand-brick byte-drop-in) | **Win-K1-VLEN**·byte-exact |
| A2 | **gemm_tile/q4_K@k1 prefill** | **1.187** | 同上 REAL hand-brick STRONG | 同上 |
| A3 | gemm_tile/q2_K@k1 prefill | 1.364 | `ggml_gemm_q2_K_8x8_q8_K`(REAL 16x1/8x8 hand-brick STRONG) | prefill-only |

**q4_K@k1 = 唯一双轴（kernel + e2e）verified hard win**：kernel prefill 1.187 / decode 1.535 + **e2e prefill 1.085×**（perf-covered 绿 row·`kernel账净绿-sealed-Win·Win-K1-VLEN RATIFIED`·八门 ①②③④⑥⑦⑧ + ⑤ FU-2 RESOLVED·clang-18 对称·k1/VLEN256）。**这是整个盘点里成色最强、且唯一同时过 kernel+e2e+多门的格。**

**诚实缺口（A 层也不许美化）**：
- **q2_K@k1 (A3)**：仅 **prefill 赢**；**decode 0.9585 = near-parity 非 win**（主表 row67 明文「★near-parity 非 win·唯一非-loss·成色低不称赢」）→ decode 归 D。故 q2_K 是**单相**硬赢·e2e 立点弱于 q4_K。
- **bench 通道**：A1–A3 属早期 sealed/C1-deploy campaign（`a6fdf1a3`/batch4）·**非** r5.1 ad-hoc 批·q4_K@k1 已过门⑥（[PERF-2] 噪声自检 + 中位数+CI）= T-N 式纪律在案。但按 CLAUDE.md:35 最新现行法「真数须经 bench 通道」·`runs.log` 尚无对应行 → **最后一步 = 走一次正式 bench 通道复封**（不降级为 E·但未 100% 闭合最新门）。

---

## 三、9 个 perf-covered 绿格 → 分层映射（关键诚实发现）

perf-covered「9/83 绿」= **系统账/e2e** 口径。逐格拆成色后：**9 格里只有 1 格（q4_K@k1）是 A 层真硬赢**，其余 8 格是 path-win / 便宜档 / tie / 成色 DEFERRED。

| perf-covered 绿格 | e2e 比值 | 对手身份 | 成色层 |
|---|---|---|---|
| q4_0 gemm@rvv | prefill 5.92× / decode 1.91× | block-dot（stock repack VLEN128 **gate-off**·routing 白嫖·correctness 白送） | **B**（routing path-win·非 beat repack） |
| q4_1 gemm@rvv | prefill 3.68× / decode 1.67× | 未优化 generic block-dot（stock q4_1 无 riscv repack·异常慢~2×） | **B**（path-win·vs 未优化 generic） |
| q8_0 gemm@rvv | prefill 4.35× / decode 3.81× | FLAT block-dot（**上游 VLEN128 破损**·correctness-carrier） | **B**（大倍数=上游破损·非 beat 手调） |
| q5_0 gemm@rvv | prefill 1.21× WIN / decode 0.82× loss | FLAT block-dot（通用向量） | **B**（弱赢通用·decode 亏） |
| q5_1 gemm@rvv | prefill 1.09× WIN / decode 0.78× loss | FLAT block-dot | **B**（near-parity 弱赢·decode 亏） |
| q8_0@ime k1 | prefill 2.233× stock（真 beat·无 IQR 重叠） | stock 非-IME block-dot；赢 vendor 2.269× | **B**（★所赢 vendor q8_0 IME **路弱** VEN/OFF=0.984× stock·非强手调） |
| q5_K gemm@k1 | prefill 1.641× WIN / decode 0.729× loss | our-emitted `ggml_gemm_q5_K_8x4`(repack) | **D/deferred**（成色升级 **DEFERRED**·对手非 verified hand-brick·F-2 纠） |
| q4_0@ime k1 | prefill 1.0088× stock | stock 非-IME RVV | **D**（★**登记 PARITY(TIE)·非 win**·IQR 重叠·CI 含 1.0；vs vendor IME 仅 0.505× LOSS） |
| **q4_K gemm@k1** | **prefill 1.085×** | **`ggml_gemm_q4_K_16x1`(REAL hand-brick STRONG)** | **A**（唯一） |

> **头条订正**：不可把「perf-covered 9/83」讲成「9 个硬赢」。它是**公平协议下≥parity/赢的构造格占比**（含 path-win/correctness-carrier/tie/deferred）。**硬碰硬赢强手调 = 1 格**（q4_K@k1，另加 q2_K@k1 prefill 单相 = 2 格式）。

---

## 四、B / C / D / E 层要点（各带 file/ratio/对手依据）

**B 便宜档（58）· 全标 opp-immaturity·禁外推 win：**
- **dequant k1 侧 + 未 de-lottery 标注侧**：对手 = clang-18 per-format autovec（标量类）。e.g. q5_1@k1 4.421 · q5_K@k1 5.127 · iq4_nl@rvv 5.51 · mxfp4@rvv 6.42。主表每格明文「**opp-immaturity·§三.12·非 perf 硬赢·勿外推**」。
- **gemm vs ggml scalar-ref(fallback) prefill 大倍数**：iq2_xs@rvv **12.32** · iq2_s@rvv 11.49 · tq2_0@rvv 9.91 · tq1_0@rvv 7.9 · mxfp4 3.65 …→ 主表明文「大倍数=opp-clang 编巨胖伪影(150–160ms)·便宜档禁称硬赢(A1§1)」。**对手是坏/未优化对手·禁用其时间算倍率**（对手与档位 §一.5）。
- gelu@rvv 1.116×（f16-LUT 同档重比·便宜档表查·非硬赢）。

**C 部署变更（18）· 非 vs-对手 perf claim：**
- **de-lottery 17 格**（dequant@rvv owned emit）：baseline 是 clang autovec **codegen 抽签**（<0.8 或不稳）→ 造 owned body 使其不吃彩票（`[L-8]` construction·gather=0·byte-exact GREEN）。**倍数（如 q5_0 5.96 / q5_1 6.44）是 vs 抽签基线·非 vs 强对手** → 结构/部署贡献，成色注仍 = 便宜档 + measured-非-T-N。
- **measured-gate C1 flip 2 格**：q5_0@k1 decode 1.205 · q5_1@k1 decode 1.317（`PASS-DEPLOYED·C1·batch4 免测确认`·repack-GEVM 部署·成色 = beat-weak-baseline·非 beat-hand-tuned）。

**D 跨算子/near-parity/败（115·最大桶）·关键子群：**
- **★对强手调 vec_dot 全败**（kernel 账·SpacemiT vl128/vl256 STRONG·具名-X）：q4_K@rvv **0.189** · q6_K@rvv 0.18 · q3_K@rvv 0.258 · q2_K@rvv 0.342 · iq1_m@rvv 0.152 · iq3_xxs@rvv 0.156 · iq3_s@rvv 0.213 …（两板 21 格败）。**这是最诚实的现状：M=1 vec_dot 对厂商 vl 特化手调核大面积落败**（对手结构优势具名·非物理墙）。
- **CROSSOP（我方 repack-GEMM vs 对手 per-column block-dot vec_dot·非同算子硬赢）**：全 gemm K-quant/grid 的 vl128/vl256 对手行 + iq4_nl gemm。即便 ratio>1（如 q3_K@k1 prefill 2.92·iq2_xxs@k1 gemm 1.458）·**跨算子·禁计同算子 beat**。
- **repack-DEFERRED**：q5_K@k1 prefill 4.448 · q6_K@k1 prefill 1.769（对手 repack 但**非 verified hand-brick**·成色升级 DEFERRED·F-2 纠）→ 不入 A。
- **near-parity/tie**：vec_dot iq2_xxs@rvv 0.84 · tq1_0 0.82(双板) · tq2_0 0.975/0.879 · q4_0/q8_0 vec_dot ~0.94–0.99；forward: softmax 1.007 · silu 0.997 · scale 0.998 · rope 1.013 · rms_norm 1.24 · add/mul/cpy 0.96–1.21（同源 intrinsic/autovec parity）· quantize_row q8_0/q8_1/q8_K 1.0（vs 同一手写 RVV intrinsic·parity）。
- **IME kernel-axis 败/tie**：q4_K@ime 0.049（输 vendor IME 20×·C3′[PAT-1] 边界最锋利）· q4_0@ime 0.196（输 5.1×）· q8_0@ime 0.937（near-parity·对手=RVV-repack 回退真核·**非 vendor-IME·禁写赢了 IME**）。
- **注**：kernel-axis vec_dot **PASS>1 vs 手调** 4 格（iq2_s@rvv **1.92** · iq4_xs@rvv 1.31 · iq4_xs@k1 1.196 · iq4_nl@rvv 1.15）落 D——对手是 **LIGHT 手调**（codebook-gather LIGHT）或 **board-split**（iq2_s@rvv 1.92 但 k1 0.562 LOSS）·无 e2e·未过 [PERF-1] 双板+e2e → **不入 A**，但成色高于便宜档（见 §五候选）。

**E measured-非-T-N（3·以新 perf 倍数为 headline 的 r5.1 grid flip）：**
- iq2_xs@rvv dequant **3.88**（r5.1-d·真地盘 FLIP·gather-free·byte-exact）· iq3_xxs@rvv **1.39**（r5.1-W4·grid gather 墙证伪=codegen-STRUCTURE 可翻）· iq3_s@rvv **1.71**（r5.1-c·median 7-seed·审计订正 max-of-N→median）。
- **三格皆真 byte-exact 且倍数真·但**：主表 iq3_s row49 明文「数走 ad-hoc `run_*.sh` 非 bench 合法通道(`runs.log` 07-19/20 零行)·无 T-N 噪声地板(测量判据.md:32·dequant T-N 命中 0)= **按项目自己法尚无判定资格·pending bench-channel + T-N 正式确认·成色 = measured 非 T-N-qualified**」。**论文引用前须走 bench 通道复测。**

---

## 五、下一个真赢候选（最可能达 A 层·各需什么）

| 候选 | 现状（file/ratio/对手） | 达 A 层需要什么 |
|---|---|---|
| **① q2_K@k1 decode**（最近） | decode **0.9585** near-parity（row67）· prefill 已 1.364 赢 REAL `ggml_gemm_q2_K_8x8` hand-brick STRONG | 补齐 ~4% decode 缺口（16-sub-block vsetvl 81 storm 诊断）→ 成**第 2 个双轴 A**（与 q4_K 并列）· + e2e + **走 bench 通道过 T-N**。**最短路径**（对手已是 verified STRONG·只差 decode 相 + 正式门）。 |
| **② q5_K / q6_K@k1 prefill**（验对手） | prefill 4.448 / 1.769 赢 repack·但对手**非 verified hand-brick·成色 DEFERRED**（F-2 纠） | **验证对手确为真 hand-brick**（q5_K 8x8 前提已证伪→须钉死该板真最强 repack 符号 + objdump full-unroll 证据）·若坐实 STRONG → 直接升 A·**低成本**（不改我方核·只补对手身份取证 + bench/T-N）。 |
| **③ B1 余项 q8_0 / q4_0 / q4_1 gemm@rvv**（造强对手·扇出最大） | 现 = path-win vs block-dot（VLEN128 gate-off / 无 riscv repack）· e2e 4.35× / 5.92× / 3.68× 但**对手弱**（B 层） | B1 已证 **ggml x16 是 VLEN256-only 非法对手**（VLEN128 上跑不了）→ 须先**识别并构造真强 VLEN128 对手**（`4x8` / SpacemiT VLEN128 repack / 正确 gate 的 block-dot）·再编译器对称 byte-exact 击败之·走 bench+T-N。**扇出最大但前置重**（须先造出强对手才有硬赢资格）。 |

**不建议当候选（诚实劝退）**：q4_K@rvv / q6_K@rvv 等 vec_dot **对 SpacemiT vl128 STRONG 深度落败 0.18–0.19**——非 near-A·是「对手结构优势具名」的真败；iq2_s@rvv 1.92 kernel-axis 单板赢但 k1 半 0.562 败 + 无 e2e·需先补 board-split 与 e2e 才谈 A。

---

## 〔附·判层规则（可复现·优先级从高到低）〕
1. **A** ⟸ gemm_tile ∧ fmt∈{q4_K,q2_K} ∧ board=k1 ∧ 对手含 REAL/hand-brick ∧ 非 CROSSOP ∧ ratio≥1.0（项目自锁「真硬赢锁 2」）。
2. **D** ⟸ CROSSOP（跨算子·恒不计同算子赢）。
3. **D** ⟸ ratio<1.0（near-parity 0.8–1.0 tie 或 loss<0.8·含具名-X）。
4. **D** ⟸ forward/quantize parity 族（同源 intrinsic/autovec）。
5. **D** ⟸ gemm repack-DEFERRED（q5_K/q6_K@k1·对手非 verified hand-brick）。
6. **C** ⟸ de-lottery 标注（dequant owned-emit）∨ gemm C1 measured-gate deploy。
7. **E** ⟸ r5.1 标注（grid flip·以新倍数为 headline）。
8. **B** ⟸ 余下 ratio≥1.0 赢弱对手（scalar-ref / block-dot path / 弱 vendor / dequant autovec / gelu-LUT / routing）。
> 横切旗标（measured-非-T-N、de-lottery）独立于主层·并行标注·防重复计数。规则与数字均可由 `experiments/master/T3_master_rebuild.csv` + 上列脚本机器重放。
