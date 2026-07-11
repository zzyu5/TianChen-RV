# perf-covered 基线精算 + 测量欠账表精填 — [TEMPLATE-AUDIT] 线 B

> 审计快照 HEAD = `b3e3fef4` · 生成 2026-07-11 · 触碰集 = docs 域(只读全仓 + 新建本报告)。
> **口径**：`perf-covered = 已构造格中经公平协议(八门 + 双账本 + 对手探针)测得 ≥parity/赢 的格数 / 已构造格数(84 certified)`。
> **严判**：孤立 micro / 撤回数 / 未过八门 / vs-sanity(scalar/naive) / IME 零吞吐 / FLAT 零对位 **一律不计入分子**。
> 数字住本报告 + CI,不写进 spec。本报告不改任何 code/schema/ROADMAP。

---

## 0. 一句话结论

**perf-covered = 1 / 84 = 1.19%(严判)**。唯一入分子格 = `gemm_tile/q4_K`(经 **Win-K1-VLEN** RATIFIED sealed Win:k1/VLEN256 e2e prefill 1.085× vs 真出货 hand-brick,byte-exact,八门 ①②③④⑥⑦⑧ ✓ + ⑤ FU-2 RESOLVED)。

**放宽一格(仅供对照,不作头条)= 2 / 84 = 2.38%**:并入 `gemm_tile/q4_0`(e2e prefill 5.92× 系统账 routing 赢,**但仅 5/8 门**,②③④ pending,beat 措辞 LOCKED [NG-4])——按"未过八门不计"**严判剔除**,仅列为最近边界格。

诚实定性(与 ROADMAP 头条纠偏一致):**结构轴 certified 92.31% ∧ 测量轴 perf-covered ≈1.2%**。造得出 84 格 ≠ 立得住性能;当前只有 1 格穿透完整公平协议达 ≥win-vs-shipped。这是**诚实起点**,不是失败——头条纠偏预注册"预期很低"。

---

## 1. 分母 = 84 certified 格(`coverage_metrics.py report` · sixstate a7366d09)

93 roster = **84 constructed(=certified)** + 7 声明例外 + 2 域外。分母 91(去 2 域外);certified 84;C_construct 84/91 = 92.31%。
perf-covered 分母 = **84 certified 格**,逐 op 分布:

| op | 数 | 格(format) |
|---|---:|---|
| vec_dot | 23 | q4_0 q4_1 q5_0 q5_1 q8_0 q1_0 q2_K q3_K q4_K q5_K q6_K iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs nvfp4 tq1_0 tq2_0 |
| dequantize_row | 24 | (同上 24 格,含 mxfp4) |
| gemm_tile | 22 | q4_0 q4_1 q5_0 q5_1 q8_0 q2_K q3_K q4_K q5_K q6_K iq2_xxs iq2_xs iq2_s iq4_nl iq4_xs mxfp4 tq1_0 tq2_0 **+ q4_0@ime q8_0@ime q4_K@ime**(3 IME 格) |
| quantize_row | 3 | q8_0 q8_1 q8_K |
| product_reduce | 3 | q4_0_nibble offset_binary_n3 codebook_n3 |
| 逐元素/norm | 9 | rms_norm/softmax/rope/silu/scale/gelu/add/mul/cpy × f32 |
| **合计** | **84** | |

---

## 2. 分子逐格核查(严判 · 具名 + 数据来源 + 八门状态)

### 2.1 入分子(qualifying · perf-covered = 1)

| # | 格 | 数据 | 对手探针 | 双账本 | 八门 | 来源 | 判 |
|---|---|---|---|---|---|---|---|
| 1 | **gemm_tile/q4_K** | k1/VLEN256 e2e prefill **1.085×**(CI[1.083,1.088],N=10 paired,byte-exact,部署五验) | 真出货 hand-brick(k1 as-shipped ggml = **clang-18**,对称-clang) | **kernel 账合法**(k1 出货即 clang,对称核心成立) | **①②③④⑥⑦⑧ ✓** + **⑤ FU-2 RESOLVED**(升双板) | `SEALED-WIN-REGISTRY.md` Win#1 + `2026-07-10-Win-K1-VLEN-加固报告.md` + `2026-07-10-vlen-adapt-...candidate.md` | **✅ 入分子** |

**caveat(诚实并列 · 登记档不可分割)**:Win-K1-VLEN 定性 = **"方法跨板泛化(非同一 kernel 双板复制)"**。k1 半 = 1.085× vs 真出货 hand-brick(clean,入分子的依据);rvv 半 = 1.336× vs **clang-symmetric block-dot**(构造对手,**非板出货路径**——出货是 hand-brick,我方 emitted vl=8 vs 真出货实为 **0.75× LOSS**,归 [VLEN-ADAPT] emitter 宽度缺口)。故本格 perf-covered 立足点 = **k1 半**;rvv 半是方法泛化证据,不是 vs-shipped 赢。

### 2.2 最近边界格(未过八门 · 严判剔除 · 放宽则 +1)

| 格 | 数据 | 八门 | 为何剔除 |
|---|---|---|---|
| **gemm_tile/q4_0** | rvv/VLEN128 e2e prefill **5.92×** [5.91,5.93] + decode 1.91×;k1/VLEN256 e2e prefill 1.0022× PARITY(VLEN-flip 镜像) | **5/8**(①bounded-ULP 非 byte-exact · ②MISSING RVV1.0 VLEN256 flip lit · ③PARTIAL 1/2 板 objdump · ④PARTIAL 缺 Amdahl 传导 + 多 prompt 长度 · ⑤PASS · ⑥PASS · ⑦PASS · ⑧PASS) | **系统账 routing 白嫖**(上游 f3e1828 自带 repack 全链、VLEN128 gate OFF、我方一行翻开 + 字节等价构造);win_type C,beat 措辞 **LOCKED [NG-4]**;②③④ pending → 严判"未过八门不计"。来源:`T-PERF1_q4_0_vlen128_prefill_8gate.md` |

### 2.3 逐类剔除清单(严判 · 均**不入分子**)

| 类 | 格/数 | 数据 | 剔除依据 |
|---|---|---|---|
| **撤回(kernel 账)** | gemm_tile/{q4_K,q2_K,q5_K} S6 kernel-轴 | 1.884× / 1.413× / 2.193× | [CASE-COMPILER-ASYMMETRY] CLOSED:clang-ours-vs-gcc-shipped artifact;对称 gcc/gcc 重测蒸发 0.272/0.386/0.775×(e2e 对称 0.334×)。系统账仅存 codegen 观察,非 beat。撤回。 |
| **micro/kernel-轴 NON-e2e(对称-clang 幸存但非 e2e、非八门)** | q4_K/q5_K @k1 kernel-轴 | 3.106× / 1.916×(SURVIVED 对称-clang) | 幸存但 = kernel-micro,**非 e2e、非八门**;对手 = block-dot(q4_K 是 non-shipping strawman)。q4_K@k1 的 **e2e** 已由 §2.1 sealed Win 单独覆盖;此 micro 数不另计。来源 `T-VALIDITY-STAGE1_k1_symmetric_remeasure.md` |
| **对称-fair 但 LOSS(≥parity 不成立)** | vec_dot iq3_s/iq2_s/iq2_xs/iq2_xxs/iq3_xxs/iq4_xs/tq1_0/tq2_0 | 0.16–0.78×(ours 1.3–6.3× 慢) | batch2c 对称-gcc measured-symmetric-fair,但**全 LOSS**;iq4_xs/tq1_0 还 OUTPUT-DIVERGENT。非 ≥parity。 |
| **kernel-prefill CANDIDATE(未封八门,多为 LOSS)** | gemm_tile q4_K/q5_K/q6_K/q2_K/q3_K repack-gemm | q4_K 0.94–0.97× parity · q5_K 1.50–1.62×(vs UNTUNED)· q6_K/q2_K/q3_K 0.18–0.20× LOSS | CANDIDATE-not-sealed,八门 NOT walked,beat FORBIDDEN;q5_K 赢的是 untuned softest baseline。 |
| **vs-sanity(scalar/naive · Win-A/S)** | rvv dequantize(product_reduce)micro | vs scalar 10.8× + vs naive-RVV 3.3× | 对手 = scalar/naive-RVV(sanity 层),**非 framework kernel**;KERNEL-only 非 e2e,非八门。头条列它是 sanity,不入 perf-covered。 |
| **micro-only parity(非 e2e、非 clang-20、非八门)** | vec_dot/gemm q8_0 typed-flat | VLEN256 1.04× WIN / VLEN128 parity(stale) | micro-only,board+clang-bound(clang-18/17 非 clang-20),phase=micro;非 e2e、非八门。 |
| **孤立 memory-轴 A/B(无 ggml 对手探针)** | rms_norm_mul fused / rms_norm_mul_quantize fused | isolated A/B win | 无 ggml factory 对手(no opponent probe);[NG-4] NOT-ggml-beat NOT-e2e-8gate。 |
| **e2e null** | rms_norm_mul whole-model e2e fusion | NOT-SIGNIFICANT | 预注册诚实 not-significant 分支;无 e2e speedup。 |
| **IME 3 格零吞吐** | gemm_tile q4_0/q8_0/q4_K @ime | (无) | 结构 certified(silicon-sealed int32 0-diff)但**零 perf 数**;T5b 未跑。任务明示不计。 |
| **FLAT 4 格零对位** | gemm_tile q4_1/q5_0/q5_1 + q8_0 | (无对位) | 已构造但零头对头对位;覆盖式铺面①待填。任务明示不计。 |
| **流式格未测** | dequantize_row 24 / quantize_row 3 / vec_dot 大批 | (无对位) | 无公平对位测量;预期多 parity(带宽受限),覆盖式铺面③待抽样。 |
| **e2e 正确性(非 perf)** | gemm_tile/q4_K e2e-integration | PPL 12.008≈12.05 correct | correct-proven / perf-pending;正确性不是 perf-covered。 |

---

## 3. perf-covered = 1/84 = 1.19%(严判) · 支撑面小结

- **e2e ≥win 且穿透完整公平协议** = **1 格**(q4_K@k1 via sealed Win)。
- **e2e ≥win 但八门未齐** = q4_0(5/8,系统账);放宽计入 → 2/84 = 2.38%。
- **其余 82 格** = 撤回 / micro-only / vs-sanity / LOSS / 零吞吐 / 零对位 / 未测 → **零 qualifying**。
- 与 ROADMAP 头条"已知 sealed perf 点少数"对账:头条列 3 点(q4_0 5.9× 系统账 · Win-K1-VLEN q4_K · rvv dequantize micro 10.8×)——严判后**仅 Win-K1-VLEN q4_K 清完整八门**;q4_0 = 5/8 系统账;rvv dequantize = vs-sanity micro,不入。

---

## 4. 测量欠账表精填(对账实验总纲 T 表 · 供主会话据此更新 ROADMAP 欠账表)

> 行计口径:CSV 数据行 = 去注释、去表头。模板(`experiments/_templates/`)= 0 数据行。

| T 表 | 现状(实测行数) | 应有(量级) | 欠账精填 |
|---|---|---|---|
| **T3**(格×杠杆对位·板 A/B) | 活跃 **T3_A 41 行 / T3_B 7 行**,但绝大多数 stale / INVALID / 撤回 / n_a;真 measured-symmetric-fair = iq/tq batch2c **8 行(全 LOSS)**;**qualifying = q4_K sealed(1)+ q4_0 e2e(边界)** | perf-相关子集 ≈ 24 vec_dot + 22 gemm_tile + 流式抽样 | **FLAT 4 格(q4_1/q5_0/q5_1 + q8_0 gemm_tile)零对位**;**K-quant 全族 5(q2/q3/q4/q5/q6_K)kernel 账撤回(compiler-asymmetry)→ 对称口径重填 owed**(修复不重测=白修);**iq/tq 8 格 measured 但全 LOSS(登记≠covered)**;dequant 24 / quantize 3 流式格未测 |
| **T3p**(模式逐条消融) | **0**(仅模板) | P1/P2/P2b/P4/P7 × 适用格 on/off | **整表空**;每条模式的机理声明未获数据判决;gated on 模式注册表(收缩族已数据化、块量化 body 待数据化)+ T-N |
| **T4b**(选择器四配置消融) | **0**(仅模板;`t4b-m*/` 目录是 minterm/dispatch tracer,非消融表) | 4 配置(常量/仅先验/仅记忆/先验+记忆)× oracle 遗憾% | **整表空**;先验层 SEL-1 M0 已落但消融未跑;**"矩阵静默落败"复现专项 owed**([SEL-2] 证据) |
| **T5b**(IME 2×2 范式×布局×M 扫描) | **0**(仅模板) | {RVV@vec-opt, RVV@mat-opt, IME@mat-opt} × M{1..512} | **整表空**;**G4-M3 在飞填**;IME 3 格零吞吐由此产出;M=1≈parity 预注册 |
| **T5c**(M* 写回先验闭环) | **0**(仅模板) | 交叉点 M* per format → P7 先验 | **整表空**;G4-M3 产出 |
| **T5d**(厂商路径方法学对照) | **0**(仅模板) | 厂商 IME 家族级 e2e + M=1 对照 | **整表空**;G4-M3 附带;仅作方法学对照、不进贡献 |
| **T6**(整模型 e2e 分相双板) | **1 行**(`T6_kquant_transmission.csv` = 1 条 projection·非实测 e2e 行;projection BLOCKED 8-gate 3/8) | per-format × 板 × {prefill,decode} 实测 e2e | 仅 1 条 projection(blocked);**col-outer+vl16 修复后双板重跑补行 owed**;q4_0 e2e(rvv 5.92×/1.91× + k1 parity)散在 sealed/T3、未落 T6 分相行 |
| **IME 3 格吞吐** | **0** | gemm_tile q4_0/q8_0/q4_K @ime 吞吐 | 结构 certified(silicon-sealed)但**零 perf 数**;T5b 产出 |

**附带欠账(ROADMAP 8 行外,补记)**:
- **T4a**(归因样本)= 文件**absent**(模板引用但无活跃文件);归因 JSONL 出口未建(reason 枚举 {only_feasible,static_order,prior,measured} 未落 JSONL),D-4 三级归因部分在(in-IR 属性)。
- **T8**(Win/Loss + GAP 台账)= **40 行**(已填充良好;负例/candidate 叙事账,非 perf-covered 分子源;含 q4_0 5.92×、q4_K sealed 相关行、撤回行、NULL/XFER-1 行)。
- **T-N**(噪声地板)= 1 表存在(`T-N_noise_floor.csv`);perf 判定地基,非 covered 格。

---

## 5. 产出确认

- perf-covered 精算 = **1/84 = 1.19%**(严判)/ 放宽 2/84 = 2.38%(含 q4_0 系统账 5/8)。分子逐格具名见 §2.1–2.2;逐类剔除见 §2.3。
- 测量欠账表精填见 §4(逐 T 表现状行数 / 应有 / 欠账)。
- HEAD = `b3e3fef4` 未变;未改任何 code/schema/ROADMAP(本报告纯新增 docs)。
