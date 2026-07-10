# IME gating 报告 —— 上桌版（决策级 · 供 IME 战役立项裁决）

**类型**: 纯文档 · 决策级综合（立项就绪度评估 + M4-linkage 框架）。★零 code/schema/experiments 改、未 git。仅新增本文。
**日期**: 2026-07-11。**板**: 本文不上板；所有 IME perf 数字均引自既有 memory + archived artifacts（provenance 见 §7）。
**用途**: 供用户对「IME 战役是否立项 / 立项范围」做裁决。本文是**决策级摘要 + 新增 M4-linkage 论证**，不重复前两份的深潜细节。

## 沿革（本文在前两份基础上升级，不复制、不另起炉灶）
- `docs/reports/2026-07-09-IME-gating-review.md` —— [SEL-2] 选择器**深潜 + 曳光弹计划 + 开工时点**（读"怎么关门"看它）。
- `docs/reports/2026-07-10-IME-gating-report.md` —— gating **现状台账**（consolidate + 三选项定性；读逐项证据看它）。
- **本文（上桌版）** = 在 07-10 台账基础上升级到**决策级**：① 收敛为立项裁决所需的最小结构；② **新增 M4-linkage 框架**（用户明令，前两份未含）；③ 给出**立项就绪度结论 + 范围边界**。前两份作为证据卷宗保留，不废。

---

## 0. 一句话上桌（供裁）

**IME N2 结构 = CLOSED（已证、不 re-litigate）；IME 剩余全在 N3（perf 半边）= 键控缺 + perf/board-pending。IME 立项的落地物一次兑现两处：(a) C3′ 能力键控选择器的第三条轴 + exec 级先验层（论文机制 novelty）；(b) M4 分母里 3 格 blocked-on-IME（gemm_tile/{q4_0,q8_0,q4_K}）转正为 certified，清零 M4 分母的 IME 依赖。故 IME 战役与 M4 终局收口是【两事合流】，不是两条独立线。** 硬件/工具链就绪度已满（`ssh k1` 通、march/toolchain 已知、silicon bit-exact 已证）；触发条件 = exec 级选择器立项裁决（排队 #5，已 flag 待裁）。**建议立项范围 = N3 perf 半边 + 3 格解锁，不含范围外扩张；且【不得以 perf 名义立项】（[NG-4] 措辞锁），以贡献链条 + 工程成熟度名义立项。**

---

## 1. N2 结构主张状态 = CLOSED（C1「模板协议本体跨范式」的硬证）

IME 作为 N2 家族 #2 = **零-core-branch plugin 泛化用第二 family 证明**，直接支撑 C1 合取存在性（N1∧N2）——它证明「模板协议本体」不止在 RVV 一个范式内成立，而是**跨到矩阵范式（非向量、非独立离散卡）仍成立**。四件结构判据**已证据、2026-07-11 当前码复核在位**：

| 结构判据 | 证据（当前码 / commit） | 状态 |
|---|---|---|
| **走同一 common pipeline** | op 挂 common EmitC 接口经 `TCRVEmitCLowerableRoute` 材化（`IMEExtensionPlugin.cpp` buildVariantEmissionPlan） | ✓ |
| **capability-fact dispatch（非字符串）** | `hasAvailableIMECapability` = `lookupProviderByID("spacemit.ime").isAvailable()`（`IMEExtensionPlugin.cpp:341`）；boundary 路由是 stamped FACT 的纯 read-back，不 re-classify 族名 | ✓ |
| **零-core-branch** | grep-clean：`lib/` 减 IME dir 0 个 IME 族名分支；4 zero-core op 加建 = "wiring 付一次"实证 | ✓ |
| **K1 bit-exact（硅片）** | 真 X60 `taskset -c 0-3`（harts 0-3 带 `_ime`、hart 4 SIGILL = per-hart 异构事实），16/16 == scalar；march `xsmtvdotii` + SpacemiT GCC15.2 fork | ✓ |

- **能力派生 fail-closed 强**：`deriveIMEMatmulCapability`（`IMEExtensionPlugin.cpp:184-335`）从 march `xsmtvdotii`+VLEN 派生 4×4×8 MAC fragment；march token 缺 / VLEN≠256 / signedness 越界 / shape 非 MAC 整数倍 → **全拒**。VLEN=256 = 唯一 real-K1-validated shape（VLEN128 直接拒）= N1 能力钩子在 IME 上的体现。
- **已建原语 = 6 boundary op**（`IMEOps.td`，signedness/shape/slide 三轴均能力派生 FACT）：`mma`(vmadot,ss)·`mma_u`(vmadotu,uu)·`mma_su`(vmadotsu,su，**量化主用**)·`mma_us`(vmadotus,us)·`mma_slide`(vmadot1/2/3,conv/A-reuse)·`matmul`(tiled whole-matrix)。
- **指针**：commit `2eeabff9`（N2 founding，"IME second family added as a plugin — zero-core-branch, capability-fact dispatch, K1 bit-exact"）· `2eeabff9` 后续 op `70bb845a`/`fb3ca137`/`f2446b75`（各 0-core 加建）· memory `k1-ime-n2-hardware-candidate` · reviewer-facing `artifacts/n2-ime/N2-RAPID-ADD-PROOF.md`（commit `329421af`）· spec `.trellis/spec/index.md` N2 bridge。

> **纪律锁（勿回炉）**：N2 结构已证 + reviewer-clear；**勿再 re-litigate「能否加家族」、勿重做 showcase**。调 vendor lib / ggml IME 路径**不是 N2-novel**。**这一节在立项裁决里是"已完成资产"，不是待办。**

---

## 2. N3（perf 半边）gap 精确表 —— 立项要解决的确切缺口

N3 = capability-aware tuning 的兑现（→ C3′）。IME 的 N2 结构关掉后，剩余 **全部在 N3**，分三条精确缺口。**这张表 = 立项 scope 的正表（要做的）；反表见 §4 边界（不做的）。**

| # | 缺口 | 当前确切状态（码/证据复核 2026-07-11） | 谁在修 / 归属 | 立项后目标 |
|---|---|---|---|---|
| **G-1** | **exec 级跨范式选择器 = 能力盲常量分** | `ExtensionPluginRegistry::rankKernelVariantsByCost`（`ExtensionPlugin.cpp:1383-1391`，`std::stable_sort` 升序，小者胜）：**RVV `setScore(1.0)`**（`RVVExtensionPlugin.cpp:692`）< **IME `setScore(20.0)`**（`IMEExtensionPlugin.cpp:581`）< Scalar 1000.0，**均能力盲常量**。**今天无害**（向量族/矩阵族不为同一 GEMM co-bid，[B-6]）；**P7 矩阵范式接管 GEMM prefill 落地瞬间 → 升序令矩阵范式静默落败、无 error**（[SEL-2] 硬时序根因）。 | **selector 域并行线**（[SEL-1-T5] Selector-D）——**只指指针、不抢它的活**：SEL-1-T5 的 RVV-族内 tiling 选择器部分已落（生产 dispatch 零 `static_order`）；**exec 级跨范式先验（Selector-D）与本 IME gating 是同一 gap、同一待裁项**（ROADMAP:47/55，"= IME gating 关门 + P7 enabler"）。 | 落 [SEL-1] 冷启动能力先验层于 exec 层：`GEMM 形 ∧ ime.present → 矩阵范式变体`；燃减信号 = P7-GEMM 内核 `static_order→0`（翻 `prior`）。 |
| **G-2** | **IME GEMM 的 e2e 传导未证** | kernel-micro 已测（M=1 **5.66×** / GEMM M≥4 **11.7–12.9×**，compiler-held-constant vs ggml 真 shipped RVV），但 ① 用的是 **ggml-spacemit kernel、非 tcrv 自有 IME GEMM**（自有 = gated X2、未建）；② **e2e decode = 0.86–0.98× NULL**（memory-wall，M=1 GEVM 矩阵单元帮不上）；③ 可辩护 e2e = SPACEMIT backend **家族 swap** ~1.65× prefill（227-symbol，**明确不是矩阵单元**，IME-unit 增量 +0.18 不可隔离）。 | memory `kernel-wins-dont-transplant-to-e2e`（campaign-central finding）；archived `IME-PERF-REBASELINE-FINDING.md` / `IME-E2E-SPACEMIT-TOGGLE-FINDING.md`。 | compute-bound kernel 胜**不自动传导** memory-bound decode；e2e 赢须改内存行为或 prefill/matmul-bound。**立项不承诺 e2e beat**。 |
| **G-3** | **cost model 仍 1-bit / 指令级 emit 仍全手写** | 选择器成本仍 1-bit（常量分，见 G-1）；IME LEAF op 仍全手写 emit（无自动向量化 / 流水调度）。 | memory `backend-maturity-triton-reframe`（后端成熟度残留 gap）。 | 后端成熟度的长期轴，非本次立项必交付；诚实标注为残留。 |

**三条 perf 诚实（必须随任何 IME perf 措辞同行）**：
1. 历史 12.9× 用 ggml-spacemit `ime1::gemm_kernel_i8i4`，tcrv 只 emit 它由之构成的 IME LEAF op（vmadot/mma_u/mma_su/mma_slide）；**自有完整 IME GEMM = spec 明示的 gated X2，未建**。
2. compute-bound kernel 胜**不传导** memory-bound decode e2e。
3. 无 IME kernel-轴 perf cell 进 live ledger（`experiments/` 下只有 `_templates/T5{a,b,d}_ime_*.csv` 模板骨架），数字住 memory + archive。

---

## 3. ★M4-linkage 框架（本次新增 · 用户明令）—— IME 战役 ⟺ M4 终局收口【两事合流】

### 3.1 事实基座（schema 复核 2026-07-11）
M4 分母 = 93 格，当前 **certified 80/93 = 86.02%**（ROADMAP 当前坐标 · byte-exact-by-retirement 路径已穷尽 · 旁路 0 / dispatch-wired 0 / RED 0）。剩余 13 absent 格里，**恰有 3 格是 blocked-on-IME**：

| roster 格 | engine | state | anchor（schema 原文） |
|---|---|---|---|
| `gemm_tile / q4_0` | ime | **absent** | "IME MMAOp is format-agnostic; no format-keyed tile (aspirational)" |
| `gemm_tile / q8_0` | ime | **absent** | 同上 |
| `gemm_tile / q4_K` | ime | **absent** | 同上 |

- **指针**：`schema/coverage-sixstate.v1.json:669-688`（3 格 state=absent + aspirational anchor）· `schema/coverage-roster.v1.json:17`（"gemm_tile IME = the ratified aspirational triple {q4_0,q8_0,q4_K}. IME exposes a format-AGNOSTIC MMAOp/MMAUOp execution op, not a format-keyed tile → these 3 keys are absent (roster goal, not an in-code op set)."）· ROADMAP:33-42（M4 门去向 fork，"IME-GEMM aspirational（3）… = N3/IME 独立战役"）。

### 3.2 合流论证（核心，供裁）
M4 终局的 13 absent residual 分三类，**退役路径互斥**：
- **net-new quant kernel（7）** —— 无 in-code emitter → 需 board/ggml golden 执行（新工作种类 + board 依赖，见 ROADMAP fork(B)）。
- **out-of-scope（2）** —— flash_attn（非量化 kernel）· bf16（float 类型 · [G-2] 排除）—— **永久声明例外**。
- **IME-aspirational（3）** —— gemm_tile/{q4_0,q8_0,q4_K} —— **退役路径【只有一条】= IME 战役产出 format-keyed IME tile / 自有 IME GEMM 路径**，令 3 格 absent → constructed → certified。

**⇒ 合流命题**：IME 立项的落地物**同时**是两件事：
1. **N3 / C3′ 侧**：能力键控选择器长出**跨范式第三条轴**（`GEMM 形 ∧ ime.present → 矩阵范式`，G-1 的先验层）+ 承载它的 IME GEMM 路径 = 论文机制 novelty（P7 模式库条目 + [SEL-2] 时序契约的兑现载体）。
2. **M4 分母侧**：这 3 格 blocked-on-IME **就地转正 certified** → **M4 分母内「IME 依赖项」清零**。

这两件**不是两次投入**：同一次 IME 立项落地即两处兑现。**故 IME 战役 = M4 终局收口里「IME-aspirational 那一类 residual」的收口动作，不是 M4 之外的独立战役。** 这是「两事合流」的确切含义。

### 3.3 与 M4 fork 三选项（ROADMAP:39-42）的关系 —— IME 是 IME-aspirational 分支的唯一杠杆
| M4 fork 选项 | 对 IME-aspirational 3 格的处置 | IME 立项如何介入 |
|---|---|---|
| **(A) M4 实质达成** | 3 格归类 aspirational residual（声明例外） | IME 立项把 3 格从**声明例外 → certified**，是唯一"就地转正"杠杆 |
| **(B) 冲字面 90%** | 需 +4，全落 net-new/aspirational | IME 立项贡献 +3（3 格转正），是冲 90% 里**唯一非-board-harness 的 +格来源** |
| **(C) 重定分母** | 把 IME-aspirational 排出分母 | IME 立项后 3 格重获资格进分母且 certified —— **不必永久排除** |

**无论 M4 走哪个 fork，IME 立项都是把这 3 格从 residual【就地转正】的唯一路径。** 不立 IME，这 3 格只能永远停在 blocked-on-IME。

### 3.4 诚实边界（不夸大合流）
- **M4 达到"真正 100%（每格 ∈ {certified, blocked-on-IME, 声明例外}）"不以 IME 立项为前置**：blocked-on-IME 本身已是 roster **ratified 的合法终局类别**（"aspirational"）；M4 可在 IME 未立项时经 fork(A) 声明实质收口，把这 3 格记为 blocked-on-IME 例外。
- **IME 立项的准确价值** = 把这 3 格从"合法**例外**"**升格为 certified**，并让"blocked-on-IME"这个标签**有真实、已资源化的转正路径背书**（否则它只是"暂缓"的委婉说法）。即：**IME 立项使 M4 分母的 IME-那一类 residual 从"声明例外"变"清零"**——这才是合流的精确兑现，不是"M4 卡在 IME 上"。

---

## 4. 立项就绪度评估（GO / NO-GO 级）

### 4.1 就绪度清单
| 就绪项 | 状态 | 证据指针 |
|---|---|---|
| **硬件** | ✅ 就绪 | `ssh k1`（SpacemiT X60 / VLEN256 / harts 0-3 带 IME、hart 4 SIGILL）一直可用；memory `hardware-test-access` / `k1-ime-n2` |
| **march / toolchain** | ✅ 已知 | `-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d` + SpacemiT GCC15.2 fork（v1.2.4）；memory `k1-ime-n2` |
| **oracle** | ✅ 在手 | llama.cpp MERGED SpacemiT X60 IME backend（PR #15288，`ggml/src/ggml-cpu/spacemit/`）= N2 oracle |
| **N2 结构前置** | ✅ CLOSED | §1（commit `2eeabff9` + 6 op + K1 bit-exact） |
| **触发条件（exec 选择器立项）** | ⏳ **待裁** | exec 级跨范式选择器能力先验（Selector-D）= 与 IME gating **同一待裁项**，ROADMAP 排队 #5「⚠ 已 flag 待裁新立项」（ROADMAP:47/55）；[SEL-2] spec 明示先验层验证**与自有 IME GEMM(X2) 及 e2e-seal 双解耦**（可 in-tree forced-stub 先关门，见 07-09 review §3.1） |
| **资源冲突** | ⚠ 注意 | 先验层触碰集（`VariantSelection.cpp`/`ExtensionPlugin.cpp` cost 路 + lit）与 e2e-seal(T4b) 的 `lib/` dispatch 触碰集**基本不相交**，但 exec 选择器 cost 路是共享基建 → 按 memory `parallel-lines-need-disjoint-files` 应**串行 or 明确不相交守卫**，勿盲信 worktree 隔离 |

### 4.2 触发判定
**唯一未决 = exec 级选择器立项裁决**（排队 #5，已 flag）。硬件/工具链/oracle/N2 前置全绿。**[SEL-2] 关门动作（forced-stub 先验层验证）不必等 e2e-seal / 不必等自有 IME GEMM(X2)**——它 in-tree、免上板、是最小最高-C3′-价值-每单位-成本动作（07-09 review §3.1）。

---

## 5. 建议立项范围 + 边界（正表 / 反表）

### 5.1 建议 IN（正表）
1. **[SEL-2] exec 级先验层关门**（G-1）：forced-stub 复现"矩阵静默落败 → 先验层修复"（in-IR reason + 归因 JSONL 双证，reason 翻 `static_order→prior`），落 `GEMM ∧ ime.present → 矩阵范式`。**这是关门动作、C3′ 价值最高。**
2. **3 格 blocked-on-IME 转正**（M4-linkage）：IME 战役产出 format-keyed IME tile / 自有 IME GEMM 路径，令 gemm_tile/{q4_0,q8_0,q4_K} absent→certified，清零 M4 分母 IME 依赖。
3. **（可选、低优先）IME perf K1 kernel-轴 re-characterize**（曳光弹 B）：延后、并入后续 K1 板会话，不与 e2e-seal 战役并跑；增益已 banked、不传导 e2e → 低边际价值。

### 5.2 建议 OUT（反表 · 防范围外扩张）
- ✗ **不以 perf 名义立项**（[NG-4] 措辞锁）：IME 现状无一条走完八门、两板对称对位的 perf-sealed-Win；kernel-micro 非 sealed 口径、e2e 是 NULL/家族-swap。立项名义 = **贡献链条（C1 跨范式 + C2 family#2 经济学 + C3′ 第三轴）+ 工程成熟度**，**不以"论文需要"为由**（论文降温纪律）。
- ✗ **不承诺 e2e beat**（G-2）：compute-bound kernel 胜不传导 memory-bound decode。
- ✗ **不 re-litigate N2 结构**（§1 已 CLOSED）；调 vendor lib 不是 N2-novel。
- ✗ **不抢 selector 域并行线（SEL-1-T5 / Selector-D）的实现活**——本报告只指指针。
- ✗ **不把 3 格转正误锁在自有 IME GEMM(X2) 之后**：spec [SEL-2] 明示先验层验证与 X2 双解耦。

---

## 6. 诚实标注 —— C2 经济学：IME = 1/≥3 曲线，缺 2 点

**IME 首点 ≠ 完整 C2。** C2「模板经济学 = 泛化代价 → 边际成本规律」需要**曲线**（≥3 点才成"规律"），IME 只是**第一个异质家族接入点**：
- **已有 = 1 点**：IME family #2 接入（0-core 快接、6 op、wiring-paid-once）= 曲线第 2 个数据点（RVV 是第 1 个 baseline 点）。
- **缺 = 补曲线的 2 条路**：**[X-SCALAR]（家族 #3）** + **zvfh（子扩展）**——第三/第四个接入点才让"边际成本递减"从单点变**规律**。
- **指针**：ROADMAP:68「C2 IME 首点（诚实：1/≥3 曲线缺失）」· ROADMAP:73（X-SCALAR 家族#3 / zvfh 子扩展 = C2 叙事刚需，但**开工按既定触发条件、本备注不解锁新战役**）。
- **纪律**：**别把 IME 单点夸大为完整 C2**；IME 立项对 C2 的贡献 = "把曲线从 1 点推到 2 点 + 证明第二个异质家族接入多顺"，不是"C2 达成"。

---

## 7. 附 provenance / [NG-4] / 诚实纪律

- 本文 = 纯 consolidate + 决策级综合 + M4-linkage 新增；**无新 board 数字**。所有 5.66×/11.7–12.9×/0.86–0.98×/1.65×/+0.18 引自 memory `k1-ime-n2-hardware-candidate` + `kernel-wins-dont-transplant-to-e2e` + archived `n2-ime/IME-PERF-REBASELINE-FINDING.md` / `IME-E2E-SPACEMIT-TOGGLE-FINDING.md`。
- 码/schema 复核为 2026-07-11 当前树：`IMEExtensionPlugin.cpp`(:341 dispatch / :581 score20 / :184-335 derive) · `RVVExtensionPlugin.cpp:692`(score1.0) · `ExtensionPlugin.cpp:1383-1391`(selector 升序) · `schema/coverage-sixstate.v1.json:669-688`(3 IME 格 absent + aspirational) · `schema/coverage-roster.v1.json:17`(ratified aspirational triple) · ROADMAP:33-42(M4 fork) / :47,55(Selector-D 待裁 = IME gating 同一项)。
- **诚实核心（随 IME 主张同行）**：① N2 结构已证、勿回炉；② 原语级键控有、exec 级选择器**能力盲**（缺先验层，P7 前必补）；③ perf = kernel-micro（vendor kernel、不传导、不在 ledger），自有 IME GEMM=gated X2 未建，e2e NULL/家族-swap；④ IME 首要价值 = **C1 跨范式 + C2 family#2 经济学 + C3′ 选择器第三轴 + M4 分母 IME 依赖清零**，**不是 perf 倍数**；⑤ C2 = 1/≥3 曲线，别夸大单点。
- **无 `lib/`/`schema/`/`experiments/` 改、无 git。** 唯一新增 = 本文。前两份 IME 报告作为证据卷宗保留。
