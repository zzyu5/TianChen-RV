# IME gating 报告 — N2 结构现状 + N3 gating 机制盘点 + 诚实定性

**类型**: 纯文档 consolidate + gating 现状定性。★零 code/schema/experiments 改、未 git（`docs/reports/` only）。
**日期**: 2026-07-10。**板**: 本文不上板；所有 IME perf 数字均引自既有 memory + archived artifacts（provenance 见附）。
**触发**: 队列项，前置齐 —— [CASE-COMPILER-ASYMMETRY] 案结 ∧ FLAT 收口=q8_0 已落。
**关系**: 本文是 gating **现状台账**（consolidate + 定性）；姊妹件 `docs/reports/2026-07-09-IME-gating-review.md` 是 [SEL-2] 选择器**深潜 + 曳光弹计划 + 开工时点**。两者不重复：读现状看本文，读怎么关门看 07-09。
**读入并已对当前码复核**: `IMEExtensionPlugin.cpp`(:341/:581/:184-335) · `RVVExtensionPlugin.cpp`(:680) · `ExtensionPlugin.cpp`(:1362-1391) · spec `variant-pipeline/generation-selection-tuning.md`([SEL-1]:43/[SEL-2]:49-51/reason枚举:65) · `extension-plugins/ime-plugin.md` · index.md N2 bridge(:47/:53/C2表) · memory `k1-ime-n2`/`kernel-wins-dont-transplant-to-e2e`/`backend-maturity-triton-reframe` · archived `n2-ime/IME-PERF-REBASELINE-FINDING.md`/`IME-E2E-SPACEMIT-TOGGLE-FINDING.md`。

---

## 0. 一句话定性

- **IME N2（结构）= CLOSED**（已证、6 op 已建、K1 bit-exact、零-core-branch、capability-fact dispatch）。不再 re-litigate。
- **IME N3（gating）= 主要是 (b) 键控层缺，兼 (c) 真-perf 未建/board-pending**：
  - **原语级** dispatch 已键控于能力事实（`lookupProviderByID("spacemit.ime").isAvailable()`，非字符串）——**有**。
  - **家族级 / exec 选择器**（向量范式 vs 矩阵范式为同一 GEMM 竞标）**未接能力先验层**，走的是**能力盲常量分**（RVV `1.0` < IME `20.0` 升序）——**这是 gating 的真缺口**。今天无害（两族不 co-bid），P7 落地瞬间变**静默错选**（[SEL-2] 硬时序）。
  - **perf**：kernel-micro 已测（M=1 5.66× / GEMM 11.7–12.9×），但 ① 用的是 **ggml-spacemit kernel、非 tcrv 自有 IME GEMM**（自有 = gated X2，未建）；② e2e **NULL/不可隔离**（decode 0.86–0.98×；prefill 1.65× 是 227-symbol backend 家族 swap、非矩阵单元）；③ **不在 live 结果 ledger**（`experiments/` 下只有 T5 模板骨架，实数住 memory+archive）。
- **一句 C2 定位**：IME = **family #2 integrated**，是 **C2 边际成本经济学叙事的刚需锚**（0-core 快接、6 op、wiring-paid-once = 曲线第 2 点）；它的价值**首先是"接入多顺"这条经济学，不是 perf 倍数**。

---

## 1. IME N2 现状 —— 结构主张 CLOSED（consolidate + 指针）

N2 = 零-core-branch plugin 泛化，用**第二 family** 证明（→ C1 合取存在性）。IME 是 spec Novelty 表点名的 N2 非-RVV 家族（[n2-family-entry-boundary]：integrated 家族#2，复用向量寄存器堆、非独立离散卡）。以下四件**已证据、当前码复核在位**：

| 结构判据 | 证据（当前码/commit） | 状态 |
|---|---|---|
| **走同一 common pipeline** | op 挂 common EmitC 接口经 `TCRVEmitCLowerableRoute` 材化（`IMEExtensionPlugin.cpp` buildVariantEmissionPlan:618-668） | ✓ |
| **capability-fact dispatch（非字符串）** | `hasAvailableIMECapability` = `lookupProviderByID(kIMECapabilityID).isAvailable()`（:341-342，`kIMECapabilityID="spacemit.ime"`）；boundary 路由是 stamped FACT 的纯 read-back（`singleFragmentBoundaryOpForVariant:352`，不 re-classify 族名） | ✓ |
| **零-core-branch** | grep-clean：`lib/` 减 IME dir 0 个 IME 族名分支（spec `ime-plugin.md` 记 proven；4 zero-core op 加建 = "wiring 付一次"实证，`k1-ime-n2` memory） | ✓ |
| **K1 bit-exact（硅片 I8）** | 真 X60 `taskset -c 0-3`（harts 0-3 带 `_ime`、hart 4 SIGILL = per-hart 异构事实），16/16 == scalar；march `xsmtvdotii` + SpacemiT GCC15.2 fork | ✓ |

**能力派生 fail-closed 强**：`deriveIMEMatmulCapability`（:184-335）从 march `xsmtvdotii`+VLEN 派生 4×4×8 MAC fragment；march token 缺 / VLEN≠256 / signedness 越界 / shape 非 MAC 整数倍 → 全拒。**VLEN=256 是唯一 real-K1-validated shape**（VLEN128 直接拒）= N1 能力钩子在 IME 上的体现。

**已建原语 = 6 boundary op**（`IMEOps.td`；signedness/shape/slide 三轴均能力派生 FACT）：`mma`(vmadot,ss) · `mma_u`(vmadotu,uu) · `mma_su`(vmadotsu,su，**量化主用**) · `mma_us`(vmadotus,us) · `mma_slide`(vmadot1/2/3,conv/A-reuse) · `matmul`(tiled whole-matrix，signed/unsigned)。

**指针**：commit `2eeabff9`（N2 founding）· memory `k1-ime-n2-hardware-candidate` · spec `.trellis/spec/index.md` N2 bridge(:47 "C1 = N1∧N2 合取"、:53 "N2 第二 family 证明") · `extension-plugins/ime-plugin.md`。

> **纪律锁（勿回炉）**：N2 结构已证 + reviewer-clear（`n2-ime/N2-RAPID-ADD-PROOF.md`）；**勿再 re-litigate "能否加家族"、勿重做 showcase**。调 vendor lib / ggml IME 路径**不是 N2-novel**。

---

## 2. IME N3 gating 机制盘点 —— 键控在哪 / 缺在哪 / perf 测没测

N3 = capability-aware tuning 的兑现（→ C3′）。IME gating 的核心问题：**编译器怎么能力键控选 IME vs RVV？** 分两层，结论相反：

### 2.1 原语级 dispatch —— 键控 **有**（能力事实主键）
- IME 变体的合法性/材化/发射全 gate 在 `spacemit.ime` capability fact（`isAvailable()`），非字符串族名比对（:341/:375/:541/:640/:694）。
- boundary op（ss/uu/su/us/slide/matmul）由 stamped `ime.signedness`/`ime.slide`/`ime_matmul_shape` FACT 的 data-flow 决定，不 re-classify。
- ⇒ **"给定要用 IME 时，选哪个 IME 原语"这层是纯能力事实驱动**，满足 [I3]/[F-6]。

### 2.2 家族级 / exec 选择器 —— 键控 **缺**（能力盲常量分，这是真缺口）
`ExtensionPluginRegistry::rankKernelVariantsByCost`（`ExtensionPlugin.cpp:1362-1391`，`std::stable_sort`）：comparator 先比 explicit-preference（RVV/IME 均 true = 平），再 `getScore()` **升序**（小者胜）。已核实分值：

| plugin | score | 来源 | 性质 |
|---|---|---|---|
| RVV | **1.0** | `RVVExtensionPlugin.cpp:680` | 能力盲常量 |
| IME | **20.0** | `IMEExtensionPlugin.cpp:581`（`estimateVariantCost`，`explicitPreference=true`） | 能力盲常量 |
| Scalar | 1000.0 | 末位兜底 | 能力盲常量 |

- **今天无害**：向量族与矩阵族**不为同一内核 co-bid**（[B-6]）→ 常量分排序对结果无影响；exec 层现只发 reason=`only_feasible`/`static_order`（能力盲冷启动排序），**从不发 `prior`**（严格保留给能力派生先验层、[SEL-1] 落地前绝不发，spec reason 枚举:65）。
- **P7 落地瞬间被激活的错选**：矩阵范式接管 GEMM prefill（总纲 [PAT-2] P7 = `ime ∧ shape`）令两族对同一 GEMM co-propose → 升序 `RVV 1.0 < IME 20.0` ⇒ **向量变体被选、矩阵范式静默落败、无 error**。这正是 [SEL-2] 硬时序（能力先验层必须**先于/同于** P7）的根因（spec:49-51）。
- **缺的具体件** = spec [SEL-1] 冷启动能力先验层在 **exec 层** 的落地：`GEMM 形 ∧ ime.present → 矩阵范式变体`（spec:43）。SEL-1 T3（commit `d2a14781`）已落的是 **RVV 族内 TILING** 选择器（tiled/plain，键=`fold_model`），**不是** exec 层跨范式选择器 —— 两层同构、复用同一两段式 + 同一 reason 枚举，但 IME 入选是把该机制**上移一层**。**燃减信号**：先验层落地 = P7-GEMM 内核 `static_order→0`（翻 `prior`）。

### 2.3 perf 现状 —— kernel 已测、真-perf 未建、e2e NULL、不在 ledger

| 维度 | 数字 | 口径 / caveat | 在 ledger? |
|---|---|---|---|
| **kernel micro（M=1 GEVM）** | **5.66×**（5.66/5.68/5.66） | vs ggml **真 shipped RVV** `vec_dot_q4_0_q8_0`（**compiler held constant**=clang-18 literal object；GCC 混编曾虚高到 14.3×，advisor-caught）；NMSE-gated；分解=2.45× IME-MAC × 2.35× ggml-shipped-weakness | ✗（memory+archive） |
| **kernel micro（GEMM M≥4）** | **11.7–12.9×** | compute-bound prefill-like；M-growth = IME GEMM-blocking（cross-row B reuse）ggml RVV lacks | ✗ |
| **e2e decode（M=1 GEVM）** | **0.86–0.98× NULL** | memory-wall；矩阵单元在 M=1 帮不上（[[kernel-wins-dont-transplant-to-e2e]]） | ✗ |
| **e2e prefill（SPACEMIT toggle）** | **1.65×**（1.62–1.66） | ★**是 227-symbol backend kernel-FAMILY swap、NOT 矩阵单元**；IME-unit 增量 +0.18 real 但**不可 cleanly 隔离**（within-prefill 比在 M≈32 峰后衰减 = FLOP-intensity win 的反面） | ✗ |

**三条 perf 诚实（必须随任何 IME perf 措辞同行）**：
1. **历史 12.9× 用 ggml-spacemit `ime1::gemm_kernel_i8i4`**，tcrv 只 emit 它由之构成的 IME LEAF op（vmadot/mma_u/mma_su/mma_slide），**无完整 tcrv IME GEMM**。**自有完整 IME GEMM = spec 明示的 gated X2**（未建）。
2. **compute-bound kernel 胜不传导 memory-bound decode e2e**（campaign-central finding）。
3. **无 IME kernel-轴 perf cell 进 live ledger**：`experiments/` 下只有 `_templates/T5{a,b,d}_ime_*.csv` **模板骨架**，`active/result-tables/` 无 IME 实测行。数字住 memory + `n2-ime/IME-PERF-REBASELINE-FINDING.md` / `IME-E2E-SPACEMIT-TOGGLE-FINDING.md`。

---

## 3. gating 现状定性（诚实，对照三选项）

任务三选项：(a) 已键控+已测=闭合 / (b) 键控在缺 capability 事实 / (c) perf 未测=board pending。**IME 是 (b) 为主 + (c) 为辅的复合，非 (a)**：

- **不是 (a) 闭合**：exec 层跨范式选择器**未**能力键控（走能力盲常量分）；且**真-perf（自有 IME GEMM + 对称 e2e 赢）未建/未证**。
- **(b) 键控层缺 —— 这是主缺口，但要精确**：capability **事实本身不缺**（`spacemit.ime` 已 first-class、原语级已消费它）；缺的是**把该事实喂进 exec 级选择器的能力先验层**（[SEL-1] 冷启动排序上移到 exec 层）。即：**事实在、家族级选择器还盲**。今天 latent-harmless，P7 落地即 hard-block（静默错选）。**关门动作 = forced-stub 复现"静默落败→先验层修复"**（in-tree、免上板、与 X2 及 e2e-seal 双解耦，spec:51；执行细节见 07-09 review §3.1）。
- **(c) perf board-pending —— 对真-perf 成立**：能构成 N3 perf-赢的东西（**自有 IME GEMM** + **与 as-shipped 对手对称对位的 e2e**）**尚未建/尚未证**；已有的 5.66×/12.9× 是 kernel-micro（用 vendor kernel、不传导 e2e、不在 ledger），重测只是 re-characterize、非新发现。

> **对照双板 sealed Win 经验的诚实**：IME 若要成 **perf 赢**，须与 **as-shipped 对手对称对位**（编译器 held-constant、对手 = ggml 真 shipped kernel、八门走完、kernel+e2e 分开报）。当前 IME **没有一条走完八门的 perf 赢**——kernel-micro 有数但非 sealed-Win 口径，e2e 是 NULL/家族-swap。**故 IME 现状 = 结构 CLOSED + gating 键控缺 + perf board/build-pending，不得以 perf 名义立项**（[NG-4] 措辞锁）。

---

## 4. 与本会话 perf 收敛的关系 —— IME 是"第三个杠杆"

本会话 perf 收敛在两个能力键控杠杆上转：**lane-width（LMUL，Win-A）** + **schedule（tiling，SEL-1 T3）**，都在**向量范式内**。IME 加的是**第三个正交杠杆 = systolic MAC 单元**（矩阵范式），是能力键控选择器该长出的第三条轴：

- **加 IME 轴的路径** = §2.2 的 exec 级能力先验层（`GEMM 形 ∧ ime.present → 矩阵范式`）。这与 lane-width/schedule 是**同一 [SEL-1] 两段式机制**，只是把键从"LMUL/tiling 形状"扩到"范式（向量 vs 矩阵）"。键仍是**瓶颈形状**（compute-bound GEMM-shape × `ime.present`，非 format/族名）→ 满足 C3′「换键不改条目」迁移判据。
- **但杠杆的 payoff 面窄且必须双口径报**：IME 5.51×/12.9× 是 **compute-bound prefill/GEMM micro**；**decode e2e = 0.86× washout**（[[kernel-wins-dont-transplant-to-e2e]]，须 e2e/对称口径重读）。即 IME 这根杠杆**只在 compute-bound prefill 有 kernel-payoff**，decode 一律 memory-wall-flat。**永远 kernel+e2e 分开报**，别把 12.9× 当 e2e。
- **定位**：IME 轴的**首要贡献是让能力键控选择器长出跨-范式的第三条轴（C3′ 模式库 P7 条目 + [SEL-2] 时序契约）**，不是拿 12.9× 刷 perf。P7（矩阵范式接管）是 IME 键控的模式库条目、IME 是它的承载 family。

---

## 5. 若需 k1 IME perf 实测 —— 板批要点（对称口径 SOP）

**优先级低**（增益已 banked、不传导 e2e、真 novelty 是自有 IME GEMM=X2 而非重测）。**若上板**，须按双板 sealed-Win 对称口径：

1. **对手身份 = as-shipped**：RVV 臂 = ggml **literal shipped** `ggml_vec_dot_q4_0_q8_0`（clang-18 object **linked**，非 recompile）；IME 臂 = ggml-spacemit `ime1::gemm_kernel_i8i4`（tcrv emit LEAF、disclose）。**禁 scalar/hand-rolled/naive 当倍数分母**（那是内部 sanity）。
2. **编译器 held-constant**（load-bearing）：两臂同 compiler-family（clang-18 shipped object）。**GCC-recompile RVV 臂会把比虚高到 14.3×**（已 caught）——绝不重犯。
3. **对称对位**：同 op（`C=dequant(Q4_0)·F32→F32`）、同 buffer、repack+A-quant symmetric 且**排除出计时区**、NMSE gate（<1e-3）**在计时前**过。
4. **engagement 硬证**：`objdump`/`--mattr` 确认 `vmadot`（`e2..382b`）真 present（历史 32×）；RVV 臂 0 vmadot。
5. **self-check SOP**：`taskset -c 0-3`（IME-bearing harts）· `vlenb==32` 断言 · idle-cert（`top %id` pre/post + loadavg D-state floor 说明）· best-of-N（N≥3 cold）· 板 identity 标注（K1/X60/VLEN256/IME1）· **kernel-轴 ONLY、[NG-4] beat 措辞锁（八门未走）**。
6. **e2e 若测**：用 **SPACEMIT=ON/OFF 单-toolchain toggle**（只翻 `-DGGML_CPU_RISCV64_SPACEMIT`、fresh OFF build），且**预声明**：decode control（tg16 M=1）**不会塌到 1.0**（227-symbol 家族 swap 达 M=1）、prefill 1.65× **不是矩阵单元**、IME-unit 增量 +0.18 **不可隔离**。别把家族-swap 当 IME-unit e2e 赢。
7. **入账**：若产数，进 `experiments/active/result-tables/`（现只有模板），绑对手身份 + 编译器对称 + 板 identity + 八门状态（[NG-4]/双账本）。

---

## 6. 与双板 sealed Win 方法的关系

双板 sealed Win（q8_0/FP4/dequant front-door 等）的方法学 = **两 VLEN emit 字节不同（非-NULL、避 [[winc-structural-null]]）+ 两板 byte-exact + 采用能力事实翻转 strip**。IME 对照：

- **IME 天然是"两板"的对偶但方向不同**：sealed Win 的两板 = 同族(RVV) × 两 VLEN（128/256）；IME 的双板经验点 = **RVV 族(rvv 机) vs IME 族(k1 机)** 的**跨-范式**对位。IME 的"翻转"= `ime.present` 决定向量→矩阵范式（比 LMUL 翻转更粗粒度）。
- **共享纪律**：① perf 主张绑 **相×板×格式×对手身份×八门状态**；② 对手贴墙 parity=满分（采用 ggml 自己指令=parity 非 beat）；③ 指令微质量非赛场。IME 现状**尚未产出一条走完八门、两板对称对位的 perf-sealed-Win**（kernel-micro 非 sealed 口径、e2e 是 NULL）——这正是 §3 定性为"perf build/board-pending"的原因。
- **方法迁移**：先验层验证（forced-stub）复用双板 sealed 的**归因严谨**（in-IR reason + JSONL 双证、reason 翻 `static_order→prior`），只是**载体是选择器科学、不是 board 数字**——这是关 [SEL-2] 门的最小、最高 C3′-价值动作，不需上板。

---

## 附 provenance / [NG-4] / 诚实纪律

- 本文 = 纯 consolidate + 定性；**无新 board 数字**。所有 5.66×/11.7–12.9×/0.86–0.98×/1.65×/+0.18 引自 memory `k1-ime-n2` + `kernel-wins-dont-transplant-to-e2e` + archived `n2-ime/IME-PERF-REBASELINE-FINDING.md` / `IME-E2E-SPACEMIT-TOGGLE-FINDING.md`。
- 码复核为 2026-07-10 当前树：`IMEExtensionPlugin.cpp`(:341 dispatch/:581 score20/:184 derive) · `RVVExtensionPlugin.cpp:680`(score1) · `ExtensionPlugin.cpp:1362-1391`(selector) · spec `generation-selection-tuning.md`([SEL-1]:43/[SEL-2]:49-51/reason:65)。
- **诚实核心（随 IME 主张同行）**：① N2 结构已证、勿回炉；② 原语级键控有、exec 级选择器**能力盲**（缺先验层，P7 前必补）；③ perf = kernel-micro（vendor kernel、不传导、不在 ledger），**自有 IME GEMM=gated X2 未建**，e2e NULL/家族-swap；④ IME 首要价值 = **C2 family#2 经济学 + C3′ 选择器第三轴**，不是 perf 倍数。
- **无 `lib/`/`schema/`/`experiments/` 改、无 git**。唯一新增 = 本文。
