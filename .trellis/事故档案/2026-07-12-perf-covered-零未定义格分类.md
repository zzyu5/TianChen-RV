# perf-covered 零未定义格逐格分类 — approximate 黄格计数 firm 成 per-cell 判读

> 审计快照 HEAD = `c4365b31`（未变）· 生成 2026-07-12 · 触碰集 = docs 域（只读全仓 + 机检 roster/coverage 脚本 + 唯一写本报告）。
> **性质**：用户 2026-07-12 裁决（一·第四问·零未定义格终态）的落地。把 ROADMAP `docs/ROADMAP.md:36` 的 **approximate 黄格分类计数**（绿 6 / 声明例外 ~10 / 物理墙 ~4 / 传导稀释 ~3 / 对手更强 ~9 / 未接线 ~52）**firm 成 84 格逐格精确判读·Σ=84·零未定义格自证**。
> **口径**（承 `2026-07-11-perf-covered-baseline.md` + 6 份绿登记·不改口径）：`perf-covered = 已构造格中经公平协议（八门+双账本+对手探针）测得 ≥parity/赢 的格数 / 已构造格数(84 certified)`。
> **read-only**：未跑板、未 git、未改 code/schema/ROADMAP。分母 roster 由 `coverage_metrics.py report`（roster_sha256 `27049680…`）机检落定，非人工假设。

---

## 0. 一句话结论 + 计数汇总

**零未定义格达成（分类轴）：84 格逐格皆有硬判读，Σ=84，无「待定义」桶。** 但**判读成色分三档**（诚实）：**已测锚定 27 格**（6 绿 e2e + 8 铺面③流式实测 + 8 iq/tq vec_dot batch2c + 4 K-quant gemm 对称重测 + IME 3 硅证 + q4_K/e2e）、**形态外推 34 格**（铺面③ morphology-declared 覆盖 + K-quant/IQ 同族类推）、**结构-Amdahl 判 23 格**（dispatch-definitional off-hot-path）。

| 类别 | 计数 | 主体 |
|---|---:|---|
| **绿**（e2e ≥parity·公平协议·成色注记） | **7**（=6 格式·q4_0-gemm 拆 decode+prefill 两 regime 双绿） | 全 gemm_tile：q4_0(×2 regime)·q4_1·q5_0·q5_1·q8_0·q4_K |
| **黄-物理墙**（roofline 证·parity-at-wall·满分非 GAP） | **12** | flat dequant 5 + quant 3 + 逐元素 gelu/scale/cpy/silu 4 |
| **黄-传导稀释**（compute-account 赢·decode wash·带 Amdahl 账） | **3** | IME gemm：q4_0@ime·q8_0@ime·q4_K@ime |
| **黄-对手更强**（公平测得对手 ≥ 我·具名 GAP + 可修否） | **12** | K-quant gemm 4（q2/q3/q5/q6_K 对称 LOSS）+ iq/tq vec_dot 8 |
| **黄-未接线**（constructed 未 wire·无公平 e2e 测·接线队列位标注） | **23** | vec_dot 15（FLAT+K-quant+iq1/iq4_nl/nvfp4 block-dot）+ gemm iq/tq/mxfp4 8 |
| **声明例外**（[X-0]+Amdahl 上限 < 噪声地板·或无公平对手·负控） | **27** | dequant 19（super-block/codebook/ternary/fp4/binary）+ product_reduce 3 + 逐元素 add/mul/rms_norm/softmax/rope 5 |
| **Σ** | **84** | ✓ 零未定义格 |

**验算**：7 + 12 + 3 + 12 + 23 + 27 = **84** ✓。逐 op 对账见 §2。

**头条对账（诚实·未改数）**：绿 = **6/84 = 7.14%**（头条口径·按 `format` 计·q4_0-gemm 折叠一次）= **7 roster-cell**（按 84 roster 计·q4_0-gemm 拆 decode+prefill 双绿）。两数指向同一事实，仅计数单位不同（format vs roster-cell）；本报告表体按 **roster-cell（84 格）** 计以对齐 Σ=84 与机检 roster，头条 6/84 仍以 format 计——**明标以防"定义与登记册各说各话"**。

---

## 1. 84 格逐格分类表（机检 roster·`coverage-roster.v1.json`·roster_sha256 `27049680…`）

> 分母外提示（不计入 84·仅列以防漂移）：**M4 声明例外 7**（`vec_dot/mxfp4` ZERO-MODEL 负控 + `gemm_tile/{q1_0,iq1_s,iq1_m,iq3_xxs,iq3_s,nvfp4}/rvv`·非 certified·在 91 M4 分母、**不在 84 perf-covered 分母**）+ **域外 2**（`bf16/all`·`flash_attn/tile`）。**ROADMAP approximate 把"M4 7"混入 perf-covered 声明例外 = 分母串台**，本报告纠正（§2）。

### 1.1 gemm_tile（22 格 = 绿 7 + 对手更强 4 + 未接线 8 + 传导稀释 3）

| # | 格（engine/regime） | 类别 | 依据 / 证据指针 / 账 |
|---|---|---|---|
| 1 | q4_0 / rvv / **prefill** | **绿** | 系统账 routing-win e2e prefill **5.92×**·5/8 门·成色注记「上游本有可用 repack 路径·我方翻 gate 白嫖·correctness 白送」·[NG-4] 措辞 LOCKED。`baseline §2.2`·memory `q4-0-e2e-is-routing-not-kernel` |
| 2 | q4_0 / rvv / **decode** | **绿** | 同格 decode e2e **1.91×**·routing 系统账（同 §1 注记）。T6 分相双板 |
| 3 | q4_1 / rvv | **绿** | 净新-scaffold e2e prefill **3.68×** + decode 1.67×·八门全过·**成色 caveat**：stock q4_1 弱（无 stock riscv repack·generic block-dot·异常慢 ~2×）→ 3.68× = **path-win**（vs 未优化 generic·非比调优核快）·both-phase-win 反常留 scrutiny·verdict 立于 prefill≥parity。`q4_1-green-6of84` |
| 4 | q5_0 / rvv | **绿** | 净新-scaffold prefill-axis green·e2e prefill **1.21× WIN**（1.23× micro 确传导·非 wash）+ decode 0.82× regression（如实披露·memory-bound GEVM）·prefill-win/decode-loss split·C1 template extensibility 实证。`q5_0-green-4of84` |
| 5 | q5_1 / rvv | **绿** | 净新-scaffold prefill-axis green·e2e prefill **1.09× WIN** + decode 0.78× reg（披露）·q8_1-activation repack 家族首成员（净新 block_q8_1x4 + quantize_mat_t<Q8_1>·C1 extensibility 扩展）。`q5_1-green-5of84` |
| 6 | q8_0 / rvv | **绿** | ★full-stack correctness-carrier **无星号**绿格·e2e prefill **4.35× / decode 3.81×**（CI 排除 parity）·deployed=proven（40ac20ca·selector 能力键控 `block_dot_memory_bound` 自然路由·.inc 字节等于 M1b·部署==证过）·成色最强（能力事实→发射正确 vl=8→部署→真路由→正确性修复上游 VLEN128 破损→e2e 兑现·无外援）。`q8_0-green-3of84` |
| 7 | q4_K / rvv | **绿** | kernel-account sealed Win（Win-K1-VLEN RATIFIED·k1/VLEN256 e2e prefill **1.085×** vs 真出货 hand-brick·byte-exact·full 八门 ①②③④⑥⑦⑧ + ⑤ FU-2 RESOLVED）。**成色注记（诚实并列）**：绿由 **k1/VLEN256 半**成立；**rvv/VLEN128 半 = e2e 0.334× LOSS**（correctness-carrier perf<parity·[GAP-Q4K-VLEN128] + gcc-codegen·见 §4）——同一 roster cell 的 rvv-half 是 yellow，绿立足点 = k1-half。`baseline §2.1`·`2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md` |
| 8 | q2_K / rvv | **黄-对手更强** | kernel-axis **对称-gcc LOSS 0.386×**（micro 1.413× = clang-ours-vs-gcc-shipped artifact·[CASE-COMPILER-ASYMMETRY] CLOSED·撤回）·具名 GAP = **gcc-15 RVV codegen（mixed-SEW 全展开 regfile-spill·S6 first-emission·LAW 例1）**·**可修否 = 低**（S6 tiling 只对 clang 有效·gcc 出货反效果；须 gcc-codegen-aware 再发射或 k1/VLEN256 部署）·在 prefill 热路（Amdahl 高·非声明例外）。`l1-t3-q2k`·`baseline §2.3 撤回行` |
| 9 | q3_K / rvv | **黄-对手更强** | kernel-axis **对称 LOSS ~0.18–0.20×**（dual-plane weight-bound·S6 tiling NULL·未过 cliff v31）·具名 GAP 同 q2_K（gcc-codegen + weight-materialization floor）·**可修 = 低**（结构 weight-bound + gcc-codegen 双阻）·热路 Amdahl 高。`l1-t3-q3k` |
| 10 | q5_K / rvv | **黄-对手更强** | kernel-axis **对称 LOSS 0.775×**（族内最浅·S6 hybrid-holds v31→v30·+41.7% vs UNTUNED 但对称 gcc/gcc 仍 <1）·具名 GAP 同族·**可修 = 中-低**（tiling lever 过 cliff·但 qh-plane 105-spill 残留 + gcc-codegen）·**最接近 parity 的 K-quant·wiring 待判**（§4）·热路 Amdahl 高。`l1-t3-q5k` |
| 11 | q6_K / rvv | **黄-对手更强** | kernel-axis **对称 LOSS ~0.18–0.20×**（dual-plane weight-bound·S6 NULL·v31）·具名 GAP 同 q3_K·**可修 = 低**·热路 Amdahl 高。`l1-t3-q6k` |
| 12 | iq2_xxs / rvv | **黄-未接线** | constructed repack-GEMM·**无 gemm-轴公平对位测量**（batch2c 是 vec_dot 非 gemm）·sibling vec_dot 0.78× LOSS → wiring 预判 likely 对手更强·接线队列 = **IQ（K-quant 后）** |
| 13 | iq2_xs / rvv | **黄-未接线** | 同上·接线队列 IQ |
| 14 | iq2_s / rvv | **黄-未接线** | 同上·接线队列 IQ |
| 15 | iq4_nl / rvv | **黄-未接线** | constructed·gemm PLAIN/UNTILED **结构-NULL**（≤32-vreg 悬崖·无 tiling 杠杆·parity-ish 预期）·无 gemm 对位·接线队列 IQ |
| 16 | iq4_xs / rvv | **黄-未接线** | 同 iq4_nl 结构-NULL·sibling vec_dot 0.72× LOSS·接线队列 IQ |
| 17 | mxfp4 / rvv | **黄-未接线** | constructed·gemm PLAIN/UNTILED 结构-NULL·tiny-codebook 无杠杆·无 gemm 对位·接线队列 IQ |
| 18 | tq1_0 / rvv | **黄-未接线** | constructed·无 gemm 对位·sibling vec_dot LOSS+OUTPUT-DIVERGENT·接线队列 IQ（低优先·ternary 少数部署） |
| 19 | tq2_0 / rvv | **黄-未接线** | 同 tq1_0·接线队列 IQ |
| 20 | q4_0 @ ime | **黄-传导稀释** | 结构 certified（silicon-sealed int32 0-diff·vmadot 0xe210312b）·[GAP-IME-LEAF-PIPELINE] 闭环后 kernel-axis compute-account **~2.09×**（对手 SELF）·**decode wash**·M=1 不传导·**Amdahl 账**：厂商 IME 天花板参照证即便完全接线·内存墙 1B decode 无 cleanly-isolated e2e win → compute-account ~2× 不传导 → decode e2e 上限 < 噪声·接线队列 = **IME（IQ 后·跨范式名义非 perf）**。memory `m4-closure`·`measurement-offensive` |
| 21 | q8_0 @ ime | **黄-传导稀释** | 同上·q8_0 flat IME tile |
| 22 | q4_K @ ime | **黄-传导稀释** | 同上·q4_K super-block IME（P7 `ime∧shape` format-agnostic MMAOp 统一接管三格） |

### 1.2 vec_dot（23 格 = 对手更强 8 + 未接线 15）

| 格 | 类别 | 依据 / 账 |
|---|---|---|
| iq2_xxs | **黄-对手更强** | batch2c 对称-gcc measured-fair **0.78× LOSS**·decode matmul 热路·具名 GAP = **per-subblock gather/redsum 未 batch**（`uarch.gather_slow` 邻接·GAP triage ①）·**可修 = 争议**（gather-batching 较大 emit 重构·iq 少数部署·「真赢潜伏」低 confidence·新立项须问） |
| iq2_xs | **黄-对手更强** | batch2c 对称 LOSS·同 GAP·可修争议 |
| iq2_s | **黄-对手更强** | batch2c 对称 LOSS·同 GAP·可修争议 |
| iq3_xxs | **黄-对手更强** | batch2c 对称 **0.16× LOSS**（ours 6.3× 慢）·同 GAP·可修争议 |
| iq3_s | **黄-对手更强** | batch2c 对称 **0.28× LOSS**·同 GAP·可修争议 |
| iq4_xs | **黄-对手更强** | batch2c 对称 **0.72× LOSS**·OUTPUT-DIVERGENT 曾记·同 GAP·可修争议 |
| tq1_0 | **黄-对手更强** | batch2c 对称 LOSS·OUTPUT-DIVERGENT·同 GAP·ternary 少数部署 |
| tq2_0 | **黄-对手更强** | batch2c 对称 LOSS·同 GAP·ternary 少数部署 |
| q4_0 | **黄-未接线** | block-dot vec_dot（decode matmul 备选路）·**parity-by-adoption 预期**（采 ggml 自身指令）·**无公平 e2e/八门测量**·绿路走 repack GEVM（gemm cell #2）非本 cell·接线队列 = FLAT-decode（低优先·绿已由 gemm 覆盖） |
| q4_1 | **黄-未接线** | 同 q4_0·parity-by-adoption·未测 |
| q5_0 | **黄-未接线** | 同上 |
| q5_1 | **黄-未接线** | 同上 |
| q8_0 | **黄-未接线** | block-dot·micro-only VLEN256 1.04×/VLEN128 parity（非 e2e·非八门·board+clang-bound·stale）·未过公平协议 → 不计绿·接线队列 FLAT-decode |
| q1_0 | **黄-未接线** | block-dot·binary·未测·接线队列 IQ/尾 |
| q2_K | **黄-未接线** | K-quant block-dot（decode matmul）·未直接测（K-quant 测量在 gemm/repack 轴）·接线队列 K-quant-decode |
| q3_K | **黄-未接线** | 同 q2_K·接线队列 K-quant-decode |
| q4_K | **黄-未接线** | 同上·decode block-dot·未直接测·接线队列 K-quant-decode |
| q5_K | **黄-未接线** | 同上 |
| q6_K | **黄-未接线** | 同上 |
| iq1_s | **黄-未接线** | 1.5-bit 2048-grid block-dot·未在 batch2c·grid-decode-bound 疑似 LOSS 族（未测）·接线队列 IQ |
| iq1_m | **黄-未接线** | iq1_s sibling·未测·接线队列 IQ |
| iq4_nl | **黄-未接线** | codebook block-dot·未在 batch2c（vec_dot 轴·区别于 dequant 轴的 gather-trap 冷叶）·接线队列 IQ |
| nvfp4 | **黄-未接线** | fp4 codebook block-dot·未测·接线队列 IQ |

### 1.3 dequantize_row（24 格 = 物理墙 5 + 声明例外 19）

> 全 op **off 热 matmul 路**（服务 get_rows/embedding/conversion·非 matmul）→ Amdahl e2e 上限 **< 噪声地板**（dispatch-definitional·<~0.5% decode/prefill）。铺面③（`covering-batch3`）逐形态抽样 + 同族 morphology-declared 覆盖（casefile 明示 "untested same-family formats declared-covered"）。

| 格 | 类别 | 依据 / 账 |
|---|---|---|
| q8_0 | **黄-物理墙** | ★铺面③**实测** parity-physical（deploy 0.98× / kernel 0.86×·mem-bound scalar **3.7 GB/s** roofline·byte-exact maxrel0·满分非 GAP 非 loss） |
| q4_0 | **黄-物理墙** | flat 族·morphology-declared（q8_0 实测锚·同 flat/scalar 律·parity 类推·roofline-bound） |
| q4_1 | **黄-物理墙** | flat 族·morphology-declared |
| q5_0 | **黄-物理墙** | flat 族·morphology-declared |
| q5_1 | **黄-物理墙** | flat 族·morphology-declared |
| q2_K | **声明例外** | super-block 族（q4_K 实测**named-gap** [GAP-DEQ-KQUANT-UNPACK]·kernel 0.64× LOSS·scalar 逐元素仪式·emitter-maturity soft-gap 非物理墙）·**Amdahl e2e 上限 <0.15% < 噪声**（off 热路）→ [X-0] Amdahl 关·**修完 e2e-inert（诚实反例）** → 声明例外（C3′ 机制名义·LAW 第5例）·morphology-declared |
| q3_K | **声明例外** | 同 super-block·morphology-declared·Amdahl<噪声 |
| q4_K | **声明例外** | super-block **实测锚** [GAP-DEQ-KQUANT-UNPACK]·kernel 0.64× / deploy 0.88×·mem-bound scale-unpack 3.0 GB/s·可修黄格但 Amdahl<噪声 → 声明例外（`三具名GAP修复评估 ②`） |
| q5_K | **声明例外** | super-block（q4_K sibling·共享向量化-解码族级杠杆）·morphology-declared·Amdahl<噪声 |
| q6_K | **声明例外** | 同 super-block·morphology-declared·Amdahl<噪声 |
| iq1_s | **声明例外** | codebook/grid 族（iq4_nl 实测 split-ledger·gather-trap·MANIFEST 明示"all codebook fmts share"）·Amdahl<噪声·morphology-declared |
| iq1_m | **声明例外** | 同 codebook·morphology-declared |
| iq2_xxs | **声明例外** | 同 codebook·morphology-declared |
| iq2_xs | **声明例外** | 同 codebook·morphology-declared |
| iq2_s | **声明例外** | 同 codebook·morphology-declared |
| iq3_xxs | **声明例外** | 同 codebook·morphology-declared |
| iq3_s | **声明例外** | 同 codebook·morphology-declared |
| iq4_nl | **声明例外** | codebook **实测锚**·split-ledger [GAP-CLANG-GATHER-TRAP]（deploy 0.94× parity / kernel 4.83× latent·compute-bound gather 0.55 GB/s）·**被测冷叶 Amdahl<0.4% < 噪声** → 声明例外（真赢潜伏在**分离热 vec_dot iq 族**·非本冷叶·`三具名GAP修复评估 ①`·LAW 第6例·新第5根因类） |
| iq4_xs | **声明例外** | codebook·morphology-declared·Amdahl<噪声 |
| mxfp4 | **声明例外** | fp4 族·codebook gather-trap 共享·morphology-declared·Amdahl<噪声 |
| nvfp4 | **声明例外** | fp4 族·同 mxfp4·morphology-declared·Amdahl<噪声 |
| tq1_0 | **声明例外** | ternary 族·morphology-declared·Amdahl<噪声 |
| tq2_0 | **声明例外** | ternary 族·morphology-declared·Amdahl<噪声 |
| q1_0 | **声明例外** | binary 族·morphology-declared·Amdahl<噪声 |

### 1.4 quantize_row（3 格 = 物理墙 3）

| 格 | 类别 | 依据 / 账 |
|---|---|---|
| q8_0 | **黄-物理墙** | ★铺面③**实测** parity-physical（deploy 0.99× / kernel 1.00×·mem-bound 2.8 GB/s·byte-exact diff0·激活量化·Amdahl ~0.33% profiled 小·but 实测 at-wall → 物理墙满分） |
| q8_K | **黄-物理墙** | ★铺面③**实测** parity-physical（1.08× / 0.98×·mem-bound 2.5 GB/s·byte-exact） |
| q8_1 | **黄-物理墙** | morphology-declared（同 q8-family scale+int8 store）·**correctness-essential**（净新 sub-scaffold·wired 进绿 q5_1/q4_1 激活路）·at-wall parity 预期 |

### 1.5 product_reduce（3 格 = 声明例外 3）

| 格 | 类别 | 依据 / 账 |
|---|---|---|
| q4_0_nibble | **声明例外** | 内部子原语（喂 dequant/vec_dot 复合）·**无框架-kernel 公平对手**（ggml 无 standalone product_reduce·vs-scalar 10.8× 是 sanity 层非 framework）→ 结构无对手·perf 由复合格捕获·声明例外（无独立 e2e·无公平探针） |
| offset_binary_n3 | **声明例外** | 同上·内部原语·无框架对手 |
| codebook_n3 | **声明例外** | 同上·内部原语·无框架对手 |

### 1.6 逐元素 / norm（9 格 = 物理墙 4 + 声明例外 5）

| 格 | 类别 | 依据 / 账 |
|---|---|---|
| gelu | **黄-物理墙** | ★铺面③**实测** parity-physical（0.99× / 1.00×·compute-bound tanhf 0.11 GB/s·maxrel 1.3e-6·LUT-deploy caveat 记录·at-wall 满分） |
| scale | **黄-物理墙** | morphology-declared（trivial memory-bound map·parity 类推·at-wall） |
| cpy | **黄-物理墙** | morphology-declared（trivial map·parity 类推） |
| silu | **黄-物理墙** | morphology-declared（同 gelu 的 scalar-transcendental 类·parity 类推） |
| add | **声明例外** | ★铺面③**实测** named-gap [GAP-FWD-M8-VSETVL]（deploy 0.83× / kernel 0.81×·near-wall 7.3 GB/s·**autovec > 显式 m8 intrinsic ~20%**·memory-bound）·机制**未证**（无 objdump·违宪章规则1·提交前阻塞）·**Amdahl e2e 上限 <0.6% < 噪声** + **未接 e2e** → 声明例外（机制名义·第二 micro↛e2e 教材） |
| mul | **声明例外** | ★铺面③**实测** named-gap [GAP-FWD-M8-VSETVL]（0.76× / 0.76×·near-wall 6.4 GB/s·同 add·双账本收敛 LOSS·无 compiler-artifact）·Amdahl<噪声 + 未接 → 声明例外 |
| rms_norm | **声明例外** | off 热路·**norm 占 decode 0.05%**（G2 fusion e2e null 实证）·Amdahl e2e 上限 <<噪声（G2 融合 +0.05% < 噪声档案）→ 声明例外（reduction·铺面③ 未覆盖·结构判） |
| softmax | **声明例外** | off 热路·cheap reduction·结构 Amdahl<噪声（铺面③ 声明 "separate shape, out of map class"·未测·结构判） |
| rope | **声明例外** | off 热路·cheap rotate·结构 Amdahl<噪声（同 softmax·未测·结构判） |

---

## 2. 分类计数汇总 + vs ROADMAP approximate 对账

### 2.1 逐 op × 类别矩阵（Σ=84·零未定义自证）

| op | 分母 | 绿 | 物理墙 | 传导稀释 | 对手更强 | 未接线 | 声明例外 | 行 Σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemm_tile | 22 | 7 | 0 | 3 | 4 | 8 | 0 | 22 |
| vec_dot | 23 | 0 | 0 | 0 | 8 | 15 | 0 | 23 |
| dequantize_row | 24 | 0 | 5 | 0 | 0 | 0 | 19 | 24 |
| quantize_row | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 3 |
| product_reduce | 3 | 0 | 0 | 0 | 0 | 0 | 3 | 3 |
| 逐元素/norm | 9 | 0 | 4 | 0 | 0 | 0 | 5 | 9 |
| **列 Σ** | **84** | **7** | **12** | **3** | **12** | **23** | **27** | **84** |

**零未定义格自证**：每格恰属一类·列 Σ = 84 = 分母·无「待定义」。机检 roster 84 certified（`coverage_metrics.py` blocked_on_IME=0·全 IME certified·roster_sha256 `27049680…`）。

### 2.2 vs ROADMAP `:36` approximate 对账（firm 化 delta·逐类解释）

| 类别 | ROADMAP approximate | 本报告 firm | Δ + 原因（firm 化动作） |
|---|---|---:|---|
| 绿 | 6 | **7**（=6 格式） | +1 = q4_0-gemm 拆 decode+prefill 两 regime-cell（roster 计法·非新绿·头条 6/84 按 format 不变） |
| 声明例外 | ~10（3 GAP + 7 M4） | **27** | +17·**两处纠正**：① **M4 7 剔除**（非 84 分母·分母串台·真正在 84 内的声明例外 = 0 个 M4）；② **[X-0]+Amdahl 落地**：off-hot-path 流式/逐元素/原语（super-block/codebook/ternary/fp4/binary dequant 19 + add/mul/norm/softmax/rope 5 + product_reduce 3）从"未接线"firm 成声明例外（Amdahl<噪声·dispatch-definitional） |
| 黄-物理墙 | ~4 | **12** | +8 = 铺面③ 4 实测锚（q8_0-deq/q8_0-quant/q8_K-quant/gelu）+ morphology-declared 传播（flat dequant 4 + q8_1-quant + scale/cpy/silu）·**casefile 明示同族 declared-covered** |
| 黄-传导稀释 | ~3 | **3** | **0·MATCH**（IME gemm 三格） |
| 黄-对手更强 | ~9 | **12** | +3·**成分调整**：iq/tq vec_dot 8 保留 + **K-quant gemm 4 新增**（q2/q3/q5/q6_K 对称重测 LOSS·从"未接线"firm 出）·q4_K correctness-carrier 归入绿的 rvv-half 注记（不单列） |
| 黄-未接线 | ~52 | **23** | −29·**大幅收缩**：流式/逐元素 39 firm 成物理墙/声明例外 + K-quant gemm 4 firm 成对手更强·剩 vec_dot 15（block-dot 未测）+ gemm iq/tq/mxfp4 8 = 真-未接线 |

**结论**：ROADMAP approximate 的 "未接线 ~52" 是**保守未跑判读的兜底**；firm 化后**减半到 23**——多数原 "未接线" 实为 ① 已被铺面③ 形态覆盖判 物理墙（12）② off-hot-path Amdahl<噪声 判 声明例外（27）③ K-quant 对称重测判 对手更强（4）。真正"待接线判读"= **23 格**（vec_dot block-dot 15 + gemm iq/tq/mxfp4 8·无公平 e2e/gemm 对位）。

---

## 3. 诚实：判读成色三档 + 待接线-pending 标注（防 stale·防硬判）

**用户裁"别为凑零未定义硬判"落地——每格判读标注证据成色**：

- **A. 已测锚定（27 格·硬）**：6 绿 e2e（登记册）· 铺面③流式实测 8（q8_0-deq/quant·q8_K-quant·gelu·q4_K-deq·iq4_nl-deq·add·mul）· iq/tq vec_dot 8（batch2c 对称重测）· K-quant gemm 4（对称-gcc 重测·[CASE-COMPILER-ASYMMETRY]）· IME 3（硅证 int32 0-diff + T5b 2.09×）· q4_K rvv-e2e 0.334×（`q4k-regression`）。**这些判读跑了事实核查、非假设。**
- **B. 形态外推（34 格·中·casefile-sanctioned）**：铺面③ morphology-declared 覆盖（flat/super-block/codebook/ternary/fp4/binary dequant 的未测同族 + q8_1-quant + scale/cpy/silu）+ K-quant gemm q3/q6_K（同 weight-bound 律）。**依据 = casefile 明示 "untested same-family declared-covered (same律)"·非我方臆造**·但**逐格板测 pending**（标注在各格）。
- **C. 结构-Amdahl 判（23 格·中·dispatch-definitional）**：off-hot-path 流式/逐元素的 Amdahl<噪声 = **dispatch 定义性**（dequantize_row 定义上服务 get_rows/embedding·非 matmul；norm 0.05%）·非逐格 phase 分解实测（T6 表仅 projection）→ **定性 <噪声 高 confidence·精确 fraction% 未测**（诚实）。

**待接线-pending 硬标（23 格·黄-未接线·非硬分类）**：vec_dot 15 + gemm iq/tq/mxfp4 8 = **无公平 e2e/gemm-对位测量**·随 K-quant/IQ/IME 接线线逐格出判读转分类（可能转 绿/对手更强/物理墙）。**未虚构判读**——这 23 格明标"待接线判读"。

**stale 前提警惕（本 session 教训·已核查）**：① 机检 roster = 84 certified（**blocked_on_IME=0**·IME 三格已 silicon-certified·非 baseline 时的 blocked 状态）——用现值非旧值；② q4_0-gemm roster 拆 decode+prefill 两 cell（头条 6/84 按 format·表体按 roster-cell·已明标防串台）；③ M4 7 声明例外**不在** 84 perf-covered 分母（ROADMAP approximate 混入·已纠正）；④ K-quant kernel-axis "赢"（1.4–2.2×）**全为 compiler-asymmetry/vs-untuned artifact·对称重测蒸发**（0.18–0.78×）——用对称账非 micro 账。

---

## 4. [X-0] 三问 + Amdahl 上限预估（队首待接线格·供接线线开工前判「做/不做」）

> [X-0] 三问（canon 科研目标总纲v2:180）：①服务哪条贡献？②有无负载价值或曲线价值？③成本进 ledger 且预算被接受？
> 接线队列（ROADMAP:31/132）：**K-quant 净新接线（续）→ IQ 系 → IME bridge**。

### 队首①：K-quant gemm（q2_K/q3_K/q5_K/q6_K·当前 黄-对手更强）

- **[X-0]①贡献**：C1（extensibility·净新 K-quant repack GEMM scaffold）+ C3′（perf 证词）。✓
- **[X-0]②负载**：**HIGH**——K-quant（q4_K_M）是**最主流部署量化**·负载价值大。✓
- **[X-0]③成本**：中（净新 scaffold 或 wire 已存 repack·q5_K 素材最厚 / q3_K 最难）。
- **Amdahl 上限**：K-quant gemm 在 **prefill matmul 热路（97%）·Amdahl fraction 高·NOT <噪声 → 非声明例外**。**但对称-gcc kernel-axis = LOSS**（q5_K 0.775× 最浅 / q2_K 0.386× / q3_K·q6_K 0.18–0.20×；q4_K 部署先例 e2e **0.334×**）→ **wiring 预判 = e2e 回归（对手更强·非绿）**·根因 = gcc-15 RVV codegen 在 mixed-SEW 全展开核上远弱 clang（820 vsetvli / 742 spill vs 71/3·[CASE-COMPILER-ASYMMETRY] GCC-specific 病理）。
- **预判（供开工前判）**：**做·但非拉绿**——wiring 产出 **黄-对手更强带账**（具名 [GAP-KQUANT-GCC-CODEGEN]·可修否逐格判）·**非 perf-covered 绿**。**绿路径 = k1/VLEN256 部署**（q4_K sealed Win 先例·q5_K hybrid-holds 是次优候选·wiring 待判）**或 gcc-codegen-aware 再发射**。**不是声明例外**（Amdahl 高·热路真载重）→ 值得接线出判读（黄格带账 = 交付）·仅**管理预期：rvv/VLEN128-gcc 出货路不转绿**。

### 队首②：IQ 系（iq gemm/vec_dot·当前 对手更强 + 未接线）

- **[X-0]①贡献**：C3′（coverage + gather-batching 机制）。✓（弱）
- **[X-0]②负载**：**LOW**——iq 系**少数部署**（几无主流模型用 iq 量化）·负载价值低。⚠
- **[X-0]③成本**：**高**——gather-batching（per-subblock gather/redsum over super-block）是较大 emit 重构。
- **Amdahl 上限**：iq vec_dot 在 iq-模型 decode matmul 热路（模型内 Amdahl 高·但 iq 模型少 → 全域部署贡献低）·batch2c 对称 **0.16–0.78× LOSS** → wiring 预判 **对手更强（非绿）**·iq gemm 多结构-NULL parity-ish。
- **预判**：**低优先·非声明例外**（有公平对手 + 模型内热路·非 Amdahl<噪声）·但 [X-0]②负载低 × ③成本高 × LOSS 预判 → **cost-benefit LOW**。**「真赢潜伏」在 gather-batching**（GAP triage ① 唯一 e2e-可达邻接·但争 parity-recovery 非头条倍数·iq 少数部署·低 confidence·**新立项须问**·ROADMAP 队列外）。产出 = 对手更强/物理墙带账·非绿。

### 队首③：IME bridge（q4_0/q8_0/q4_K @ime·当前 黄-传导稀释）

- **[X-0]①贡献**：C1（跨范式完整性·format-agnostic MMAOp）·**名义非 perf**（用户裁 2026-07-12·⑤ IME bridge 批准 = 跨范式完整性名义）。
- **Amdahl 上限**：kernel-axis compute-account ~2.09×·**但 decode wash·M=1 不传导**·厂商 IME 天花板证内存墙 1B decode 无 cleanly-isolated e2e win → **decode e2e 上限 < 噪声**（micro↛e2e 铁律）。
- **预判**：**做·跨范式名义**（C1 桥·非 perf 绿）·prefill 可能小传导 / decode 不传导 → 维持**黄-传导稀释带账**（Amdahl 账已立）·**不预设转绿**（用户裁·L-接线③ 不预设）。排 K-quant/IQ 后。

**队首 [X-0]+Amdahl 净结论**：**队列内无 Amdahl<噪声 声明例外候选**——K-quant/IQ/IME 皆热路（模型内 Amdahl 高）·声明例外全在 off-hot-path 流式/逐元素/原语（已判·非 wiring 队列目标）。队首三线**预判无一「易转绿」**：K-quant = gcc-codegen 阻（对手更强·除非 k1/VLEN256）· IQ = 低负载 + LOSS（对手更强·真赢潜伏低 confidence）· IME = micro↛e2e（传导稀释）。**三线皆产黄格带账（交付）·非拉绿**——与 T6 判定"perf-covered 上升唯一杠杆 = 接线战役·且受接线-gap + micro↛e2e 双封顶"一致。

---

## 5. check_docs_canon 自检 + 产出确认

- **check_docs_canon**：本报告 = `docs/reports/2026-07-12-perf-covered-零未定义格分类.md`·带 `YYYY-MM-DD-` 日期前缀（append-only 命名合规·非 canon/ 越界·非 ledger allowlist 需求）→ **PASS 预期**（reports/ append-only 门只验日期前缀·§下机检确认）。
- **HEAD** = `c4365b31` 未变·未跑板·未 git·未改 code/schema/ROADMAP/spec·纯新增本 docs。
- **口径不改**：承 baseline + 6 绿登记·perf-covered 定义未动·头条 6/84（format）= 7 roster-cell（明标单位）。
- **交叉引用**：`2026-07-11-perf-covered-baseline.md`（分母精算）· `2026-07-12-perf-covered-{q8_0,q5_0,q5_1,q4_1}-green-*of84.md`（绿登记）· `2026-07-12-三具名GAP修复评估.md`（3 GAP Amdahl<噪声）· `2026-07-11-M4-三分类终态全表.md`（M4 7 声明例外 + 2 域外·分母外）· `experiments/active/covering-batch3-stream-rvv/`（铺面③ 8 实测 + morphology-declared）· `2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md`（q4_K rvv 0.334× + gcc-codegen GAP）· `docs/method/LAW-FIRST-EMISSION.md`（六例同律）· memory `measurement-offensive-perf-covered` / `m4-closure-three-classification` / `kernel-wins-dont-transplant-to-e2e`。
</content>
</invoke>
