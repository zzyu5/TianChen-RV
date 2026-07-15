# 论文素材指针索引 (paper evidence index) — 三贡献 → 表行/doc 指针 · 初版 2026-07-10

**用途**：这是**写作期开启判据用的索引**（每主张 → 具体证据行/文件指针），**不是论文本体、不是新主张**。
所有数字与定性住在被指向的表行/doc/cert 里；本索引只做**导航 + 成色 + 诚实状态**，[NG-4] 措辞锁沿用。

**三贡献框架**（权威 = `.trellis/spec/index.md` §Novelty 表 + N1/N2/N3↔C bridge）：
- **C1** = 合取机制存在性 → 可复制协议（capability schema 跨计算范式 × 跨独立 **extension family** 复用 + 外部贡献者接入协议）。
- **C2** = 泛化代价 → 边际成本规律（独立 **extension family**（RVV/IME/scalar/zvfh）逐家族接入的边际成本；锚 IME≈2484，曲线待 [X-SCALAR]）。
- **C3′** = 能力键控优化模式库 → 带实测与迁移的模板（含 front-door 构造协议 + 前门化边际成本谱系）。
- （+ 工程面「成熟编译器」= 覆盖率/正确性门，进 CI 不作 slide 卖点。）

---

## 模板叙事主视图（2026-07-10 定位升级 · 挂载顺序：可扩展性主证 → 性能证词 → 元证据）

> **定位 = 基于 MLIR 的能力驱动可扩展执行层软件栈之参考模板；RISC-V 量化推理为首个高性能实例**（权威 `docs/canon/TianChen-RV_定位-v2.md`）。
> **本视图不产新证据、不改任何数字/成色/指针**——只把下方 C1/C2/C3′ 分节的**同一批指针**按"可扩展性主证在前、性能证词在后"重新陈列，供写作期以模板叙事主线取用。原分节口径全部保留。

**① 主证 = 可扩展性（模板本体 C1 / 模板经济学 C2）**
- C1 模板协议本体：合取四事实逐格机器可检（→ C1 段「合取四事实」行 + `T1_C1_structural_conjunction.csv`）；falsifier 组 [F-1..F-6] + schema.def 逐 PR 审计（→ C1 段「六态 ladder + [F-EMIT]」行）。
- front-door 五族复用 + 前门化接入边际成本（→ C3′ 段「front-door 构造协议」+「前门化边际成本谱系」行；`frontdoor_framework_ledger.csv`）。
- [XFER-1] 3-类迁移预测律 **7/7** 命中（→ C3′ 段 [XFER-1] 行；`T7-three-curve-G3-closure.md`）。
- C2 IME 首点边际成本锚（→ C2 段「首点可复算」行）；**诚实：曲线 1/≥3 缺失**（→ C2 段「曲线现状=缺失」行，honest-missing 不省略）。

**② 证词 = 性能（模板产出质量 C3′；均沿用原成色/账本/八门锁）**
- q4_0 routing 5.9×（L1 路径赢，routing 白嫖非 kernel 质量赢）（→ 头条数段）。
- Win-K1-VLEN 1.085×（一条 vlen 事实换满宽；对称-clang kernel 账）（→ C3′ 段「q4_K 吞吐兑现」行 + `SEALED-WIN-REGISTRY.md`）。
- RVV 翻正闭环 [CASE-MICRO-E2E]（→ 在飞卷宗；ROADMAP）。

**③ 元证据 = 诚实清算史（测量宪法可信）**
- 撤回编译器不对称数 [CASE-COMPILER-ASYMMETRY]（→ 跨切面·方法学段）；双账本方法学（kernel 账/系统账，判别键=板出货编译器）（→ 同段「双账本」行）；certified 双列口径（→ C1 段「覆盖率现值」+ ROADMAP certified/labeled 双列）。

---

**★诚实现状口径（全文锁）**：
- **e2e = correct-proven + perf-pending(board-gated)**：full-construct q4_K 集成正确性**已兑现**
  （PPL 12.008≈stock 12.05、OUR compiler-emitted VLEN128 GEVM ENGAGED、greedy 相干）；**e2e-PERF 未闭**，缺口 = 门④
  **board-availability-gated**（共享板 SPEC+vllm 争用），**非能力缺口**。见 `T8:q4_K-e2e-integration-CORRECT-full-construct-rvv-vlen128`。
- **头条 = repack routing 5.9×（q4_0）**，且**已 canon 修正为 routing 白嫖非 kernel 质量赢**（上游 `f3e1828` 自带 q4_0 repack
  全链路、VLEN256 已路由、仅 VLEN128 gate OFF；我方一行翻开 + 字节等价构造）。链 memory `[[q4-0-e2e-is-routing-not-kernel]]`；
  **禁以 5.9× 外推 K-quant**（K-quant VLEN128 无可翻 gate）。
- **min-term 案 = CASE CLOSED（M4 定案 commit `4f765790`）**：吞吐/byte-exact 全 RESTORED；perf 数立在**现已证正确的 kernel** 上。
- **★kernel-轴 K-quant 数按编译器对称性重读（[CASE-COMPILER-ASYMMETRY] CASE CLOSED 2026-07-10）**：
  - **rvv/VLEN128 vs-opponent 数（1.884×/2.193×/1.413×/1.50-1.62×）= 撤回**（kernel 账）：= clang-ours-vs-gcc-shipped **编译器不对称 artifact**；对称 gcc/gcc 重测蒸发（0.272/0.775/0.386/0.120×），e2e 对称 = 0.334/0.764× LOSS。**系统账保留**为 clang-域 codegen 观察（LLVM17≫gcc15 pattern-specific，生态碎片化素材，非 beat ggml）。
  - **幸存 kernel-轴 = k1/VLEN256（对称-clang，k1 出货 ggml=clang-18）**：q4_K 3.106×、q5_K 1.916×（Stage1 三口径复现）；**kernel-轴 micro，NON-e2e**。★**K1-SEAL post-close caveat（2026-07-10，casefile §10 / `2026-07-10-k1-seal-e2e-transduction.md`）**：此 3.106× 是 **vs block-dot（非 k1 出货路径——stock 与 tcrv 都 repack hand-brick vl=16）= non-shipping strawman**；vs 真 as-shipped hand-brick，我方 emitted vl=8 kernel **LOSES 0.75×**（S6 半 VLEN256 宽 → emitter VLEN-adaptivity gap，归 [VLEN-ADAPT]）；**deployed g₄ vs block-dot = 2.80×（micro 90% 保留）= kernel 本质健全**。仍 [NG-4] 非 sealed beat；两板皆无 e2e Win vs as-shipped。
  - 判别键 = **板出货编译器**（rvv=gcc / k1=clang）；台账 `T-VALIDITY_compiler_symmetry_ledger.md`、casefile `2026-07-10-CASE-COMPILER-ASYMMETRY-casefile.md`。

---

## C1 — 合取存在性 → 可复制协议

| 主张（成立所需证据见 spec 表） | 主证据指针 | 成色 |
|---|---|---|
| C1「可复制协议」= capability schema 跨独立 extension family 复用 + **外部贡献者按文档接入新 family**（[C1-1] 末句）—— front-door decode 构造协议 **≠ 此处**，已正名 C3′/覆盖（见 C3′ 段） | 科研目标总纲v2 [C1-1] L145 / [C1-3] L147；N2 接入已证 memory `[[k1-ime-n2-hardware-candidate]]`（commit 2eeabff9） | proven（N2 接入 DONE）+ structural |
| 合取四事实逐格机器可检（IR-op ∧ verifier 可拒 ∧ pass 可 lower ∧ provenance 可溯） | `experiments/active/result-tables/T1_C1_structural_conjunction.csv`（逐格 fact1–4 + realized_body_manifest + conjunction_holds） | machine-checked（[F-1..F-6] falsifier + [L-8] 六态执法 CI） |
| 跨计算范式（向量 SIMD → 整矩阵 MAC）× 跨独立家族（向量缺席标量家族）同 schema 复用 | N1/N2 bridge `.trellis/spec/index.md`；N2 已证 memory `[[k1-ime-n2-hardware-candidate]]`（IME plugin 同 common pipeline、零-core-branch、K1 bit-exact） | proven（N2 结构主张 DONE，commit 2eeabff9） |
| 六态 ladder 自动读出 + [F-EMIT] 旁路护栏（constructed=STRONG 才计 C_construct） | `schema/coverage-sixstate.v1.json`（`$meta.g3_bypass_provenance` + states）；`.trellis/scripts/e5_strong_readout.py`；`tools/lint/check_frontdoor_provenance.py` | machine-checked（fail-closed） |
| 覆盖率现值 | **C_construct 66/93 = 71.0%**（过 M3 70% 门，commit `0e39f60a` 60→66；schema snapshot 停 42 未 regen，见 T7 对账） | machine-anchored |

## C2 — 泛化代价 → 边际成本规律（独立 extension family 接入成本）

> **对象锁（2026-07-10 正名）**：C2 的"家族"= 独立 **extension family**（RVV/IME/scalar/zvfh，[F-6] 闭包 ∩ rvv.*=∅），**不是** decode 格式。decode-format 前门化 LOC 摊销 = **C3′ 构造经济学**（已下移 C3′ 段）。数值全留、仅正名归属。

| 主张 | 主证据指针 | 成色 |
|---|---|---|
| [C2-1] 逐 extension family 接入代价 = **两 regime 成本分解**（〇.2 定案 2026-07-15·废"统一 <300"）：轨一 integrated 子扩展 <100（zvfh≈70 pin @`95f1a482`）/ 轨二 independent 家族如实报 pin-verified 实测（IME 2484 pin @`54465ee7` → X-SCALAR 1501 pin @`2dd654d8`·**成本高即论点本身、非缺陷**） | 科研目标总纲v2 [C2-1]+[C2-1′] 订正注 / [LED-1..3]；执行总纲v2 [LED-1] L93 | canon（权威口径） |
| C2 首点可复算锚 = IME 家族 raw wc-l **2484** | `experiments/active/visibility/T2-ledger-anchor.md`；`experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv` **seq 0**（axis=C2-extension-family） | machine-anchored（recompute PASS 2484==2484） |
| **曲线现状 = 2/≥3 点（未成完整曲线）** | 轨二 independent：IME 2484 + X-SCALAR 1501（家族#3·已 landed·pin @`2dd654d8`）两点·仍缺第 3 独立家族点；轨一 integrated 子扩展 zvfh≈70（pin @`95f1a482`·<100 坐实）粒度点已落 | **honest-missing**（[LED-2] 缺失级；诚实标·2/≥3 不得省略） |

## C3′ — 能力键控优化模式库（带实测与迁移的模板）

| 主张 | 主证据指针 | 成色 |
|---|---|---|
| **front-door 构造协议**（abstract op→typed region 可复制构造；4 decode 格式家族 × 3 op 谱同构）— **原 C1-C2-construction Part A，2026-07-10 从 C1 正名 C3′/覆盖** | `docs/reports/2026-07-09-C1-C2-construction-evidence.md` Part A（§A.1 配方 / §A.2 家族表 / §A.3 可复制性 / §A.4 commit 锚） | structural（`git show --numstat` 可重算，不跑硬件） |
| **前门化边际成本谱系**（framework-re-pay 付一次、尾格系统性归零 + family-specific 残差 ∝ 结构距离）— **原 Part B，从 C2 正名 C3′** | `docs/reports/2026-07-09-C1-C2-construction-evidence.md` Part B（§B.2 LOC 表 / §B.3 三印证 / §B.4 摊销曲线）；`experiments/active/frontdoor-framework/frontdoor_framework_ledger.csv`（seq 0–31） | structural/LOC（numstat-可锚，与 perf 脱钩） |
| 逐 flip 构造 LOC 台账（paid-once / family-specific / net_LOC 三列） | `experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv` **seq 1–13**（axis=C3'-construction）；`experiments/active/frontdoor-framework/frontdoor_framework_ledger.csv`（seq 0–31） | machine-anchored（recompute_ref HEAD 锚） |
| 原语内部 monolith→constructed 构造边际成本（成本住结构距离非表面复杂度） | `docs/method/C2_marginal_cost_ledger.md`（q3_K 表面最难却最便宜 +16；q2_K 不起眼却最贵 +388） | structural/LOC |
| NULL 也是构造/迁移边界数据点（q6_K/q3_K tiling NULL = weight-bound 迁移边界） | `T8:q6_K-repack-gemm-tile-T3-stackpanel-NULL-…` + `T8:q3_K-repack-gemm-tile-T3-NULL-XFER1-…`；T7 [XFER-1] 表 | kernel-axis（objdump spill/maxVreg） |
| **主体综合**（[XFER-1] 迁移律 + [PAT-S6] 注册表 + SEL-1 选择器一条链） | `docs/reports/2026-07-10-C3-pattern-library-evidence.md`（§1 3-类迁移 / §2 PAT-S6 / §3 SEL-1 / §4 方法学背书 / §5 commit 索引） | 见分行 |
| [XFER-1] 3-类迁移预测律（一条判据 → HOLDS/NULL/no-op，硅上 **7/7** 命中） | `experiments/active/visibility/T7-three-curve-G3-closure.md` [XFER-1] 登记表；`T8` 7 条 XFER-1 行（tile-S6 HOLDS×3 / tile-T3 NULL×2 / frontdoor-tile-NOOP×2） | **kernel-axis-measured（7/7；非 e2e、非 sealed）** |
| 模式作为一等数据对象（status 随证据流转） | `schema/pattern-registry.v1.json`（`PAT-S6-repack-gemm-output-tiling-register-cliff-XFER-1` status=mechanized；`MFLAT-P2c`=measured-negative；`WIDE-DECODE-salvage`=deferred-backlog） | machine-readable |
| SEL-1 能力键控选择器（编译期 per-format 硬编码 → 运行时能力键控；键=瓶颈 SHAPE 非 format 名） | `include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h`（纯函数族）；门⑦ lit `test/Conversion/RVV/rvv-sel1-q4-k-tiling-variant-capability-keyed-gate7.mlir`（524/524 PASS）；commit `4624740f` | machine-checked（lit）；kernel-轴 |
| 「换键不改条目」机制证明（q4_0 `lane_wise_vector_scale` 与 iq4 码本 co-map `AlreadyLean`） | C3 doc §3a；`RVVRepackTilingSelection.h` `classifyTilingBottleneckShape` | structural |
| L1 路径赢 / L2 调度赢 / L3 字节轴（正反双证 + 缺口闭环两案例 + 家族双向） | `docs/reports/2026-07-07-paper-material-inventory.md`（L1/L2/L3 全表 + [GAP-SB]/[GAP-P1]/[GAP-RP] + 家族机理双向） | 逐行标 board-proven/kernel-only/pending/structural-block/e2e-diluted-Amdahl |
| q4_K 吞吐兑现（S1→S6 把结构 opening 转 kernel-轴吞吐） | `T8:q4_K-repack-gemm-tile-S1-hstrip-…` + `…-tile-S6-minfold-stackpanel-…`；设计尺子 `docs/reports/2026-07-08-G3-L1-tiling-schemes.md` | **★重读（CASE-COMPILER-ASYMMETRY）**：rvv 1.884× kernel-轴数**撤回**（编译器不对称 artifact）；**internal-A/B（+52.4%/+27.9% register-cliff，均 ours-clang）+ objdump 封 = 幸存**（构造/成熟度轴，compiler-symmetric）；vs-opponent 吞吐兑现仅 **k1/VLEN256 对称-clang 3.106×**（micro，非 e2e、非 beat） |

---

## 跨切面 — 方法学严谨（背书 perf 主张 / 正确性可证伪）

| 资产 | 指针 | 状态 |
|---|---|---|
| [CASE-MINTERM] 卷宗（误诊→ZERO-MODEL 翻案→M4 终审全链） | `T8` `[CASE-MINTERM]` banner + `docs/reports/2026-07-09-minterm-fold-audit.md`（顶 CASE-CLOSED 指针）；commit `4f765790` | **CASE CLOSED** |
| ZERO-MODEL 终审法（rule spec） | `docs/reports/2026-07-09-zero-model-adjudication-method.md`（顶 CASE-CLOSED 指针）；memory `[[zero-model-adjudication-cert-hardening]]` | 入宪；首判例 CLOSED |
| 证书三要件 CI gate（语料完备 ∧ 输入同源 ∧ oracle 独立） | `schema/cert-lineage.v1.json`（7 cert：1 failure + 2 full[M4] + 4 partial[owed/hollow 补标]）；`tools/lint/check_cert_requirements.py`（**GREEN**）；`.github/workflows/falsifier-gate.yml` `cert-requirements-gate` | **GREEN**；仍 owed = q2_K direct dmin≠0 + repack-GEVM mat-quant 头对头 |
| ggml q8 双路径备忘（mat-quant vs row-quant） | `docs/reports/2026-07-09-ggml-q8-quant-dual-path-memo.md` | canon |
| **[CASE-COMPILER-ASYMMETRY] 卷宗（同源反汇编→对称重测→k1 反转→双账本）** | `docs/reports/2026-07-10-CASE-COMPILER-ASYMMETRY-casefile.md`（§9 终审）+ `experiments/active/result-tables/T-VALIDITY_compiler_symmetry_ledger.md` + Stage-1 `T-VALIDITY-STAGE1_{rvv,k1}_symmetric_remeasure.md` | **CASE CLOSED**（与 [CASE-MINTERM] 并列） |
| **★双账本方法学（perf 主张的两把尺 = 论文素材）** | kernel 账（编译器对称强制，源码质量）/ 系统账（各方自有栈+工具链披露+最强基线列，产品对拼）；判别键=板出货编译器；`docs/canon/TianChen-RV_执行总纲v2.md` §7 第 4/5 条 | **入宪**；方法学贡献（诚实分账、避免把工具链后端成熟度差误报成算法赢） |

## 跨切面 — e2e 状态（correct-proven + perf-pending）

| 项 | 指针 | 状态 |
|---|---|---|
| full-construct q4_K 集成正确性 | `T8:q4_K-e2e-integration-CORRECT-full-construct-rvv-vlen128`；harness `tools/e2e-harness/board/t4b_*`；`docs/reports/2026-07-10-t4b-e2e-seal-integration-proven-kernel-variant-residual.md`（部署变体残差已由 kernel-swap RESOLVED） | **correct-proven**（PPL 12.008≈12.05、OUR emitted GEVM engaged、coherent）；**perf-pending** |
| q4_K [PERF-1] 八门 | `docs/reports/2026-07-09-q4k-8gate-status.md`（★Stage-2 reconcile）；`T8` `[8-GATE-STATUS]` banner | **★Stage-2 revised**：characterization 门（①②③⑥⑦⑧ = byte-exact/VLEN-flip-lit/codegen-objdump-封/纪律/selector/措辞）**稳**；perf 门塌（**④ e2e = LOSS 0.334/0.764×**、**⑤ = 板A死板B活 reframe**）；非 sealed win；kernel-轴 perf rvv 撤回 / k1 幸存 micro |
| void gate④ = 测量纪律正面案例 | `T8` `[CASE-GATE4-VOID]`（workflow w5y3n36v8：loadavg 8-11 板测中止、0 样本、拒造数、A-tree 复原） | 正面教材（部署变体≠证过变体） |
| 传导账 / e2e projection | `experiments/active/kquant-family-closure/transmission_account.md`；`docs/reports/2026-07-10-t4b-e2e-seal-integration-proven-kernel-variant-residual.md` | **★Stage-2 更正**：~~prefill Amdahl ≈1.59×~~ = garbage-in（喂了 clang-micro 因子 1.884 而非部署 gcc 真因子 0.334）；**measured e2e = LOSS 0.334/0.764×**（对称口径，`T-PERF1b`），非 pending。decode NULL。projection 的 kernel 因子须在部署目标同一编译器下测得。 |

## 头条数（disclosure 绑定）

| 头条 | 指针 | disclosure 锁 |
|---|---|---|
| q4_0 repack GEMM e2e prefill **5.9195×**（+ decode 1.91×） | `T8:q4_0-repack-gemm-rvv-vlen128-prefill-5.92x`（win_type C，**5/8 门**）；sealed `experiments/sealed/repack/rvv-vlen128-q4_0-gemm-constructed-sealed/`；八门台账 `T-PERF1_q4_0_vlen128_prefill_8gate.md` | **routing 白嫖非 kernel 质量赢**（上游自带 repack、VLEN128 gate OFF、我方一行翻开+字节等价构造）；`docs/reports/2026-07-09-q4_0-5.9x-routing-attribution-correction.md`；memory `[[q4-0-e2e-is-routing-not-kernel]]`；禁外推 K-quant；beat 措辞 LOCKED（②③④ 未闭） |

---

## 燃减/矿脉当前状态（覆盖-构造轴，与正确性/吞吐无关）

- **C_construct 66/93 = 71.0%**（过 M3 70% 门）；42→66 的 +24 全在 dequant/quantize 前门（线B），非 repack gemm_tile 轴。
- **★吞吐兑现重记（[CASE-COMPILER-ASYMMETRY] CASE CLOSED 2026-07-10）**：~~=4（q4_0 e2e 5.9× + q4_K/q5_K/q2_K S6 kernel-轴）~~。**幸存 perf 叙事** = ① **q4_0 e2e 5.9×**（routing 白嫖，byte-identical kernel 共模，对称-gcc，稳）+ ② **k1 kernel-轴**（q4_K 3.106×/q5_K 1.916×，对称-clang as-shipped micro，NON-e2e）+ ③ **clang-域 rvv**（1.884× 等，带 caveat = 系统账 codegen 观察 LLVM17≫gcc15，非 beat）+ ④ **format-micro parity**（batch2c 对称-gcc）。**撤回** = rvv vs-gcc-shipped kernel wins（q4_K/q2_K/q5_K S6 kernel-轴 = 对称 gcc/gcc 重测蒸发）。
- **矿脉队列 = 8 待战役 gemm_tile**（flat4 `q4_1/q5_0/q5_1/q8_0` + iq2×3 + mxfp4；**FLAT 启动、`q4_1` 曳光弹在飞**）+ 9 absent；SEL-1 T4 后逐格战役、禁批量。
- 权威三曲线 cell = `experiments/active/visibility/T7-three-curve-G3-closure.md`（顶 M4 RESTORED banner + 追认 66/93 记分板；本索引不复述曲线，只指针）。

---

*本文件为纯 doc 索引（论文写作期导航），未 commit；touch-set 仅本文件；lib/schema 零改。指向的每个表行/doc/cert 各自带 [NG-4]/成色锁。*
