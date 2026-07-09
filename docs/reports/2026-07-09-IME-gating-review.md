# [G3 裁决四 / SEL-2 前置] IME gating 复核报告 — 纯文档评估

**类型**: 纯 read-only 复核 + 家族先验方案 + 曳光弹计划 + 开工时点建议。★零 code/schema/experiments 改、未 commit、未动 `lib/`。仅新增本文。
**日期**: 2026-07-09。**板**: 评估针对 `ssh k1`(SpacemiT X60 / VLEN256 / harts 0-3) — 本文不上板。
**gating 前置已达成**: [SEL-1] T3（commit `d2a14781`，全格能力键控 tiling 选择）+ T4a（commit `5b339407`，k1 双板 kernel-轴 q4_K 3.106×/q5_K 1.916×）。两件均已落地 → 本报告产出（开工时点与 T4b 进度统筹后报用户裁）。
**读入**: `IMEExtensionPlugin.cpp` · `IMEDialect.cpp`/`IMEOps.td` · `VariantSelection.cpp`/`ExtensionPlugin.cpp`（exec 选择器排序）· spec `generation-selection-tuning.md`([SEL-1]/[SEL-2]) · `core-invariants.md`([SEL-2]/[F-6]) · 执行/科研总纲 v2([B-6]/[SEL-2]/[PAT-2]P7) · memory `k1-ime-n2` / `kernel-wins-dont-transplant-to-e2e` / `n2-family-entry-boundary` · SEL-1 T1 设计 + T4 计划 + T8 ledger。

---

## 0. 一句话结论

IME 作为 N2 **结构** 已证、6 op 已建、K1 bit-exact；[SEL-2] gating 的真正缺口**不在 IME 原语**，而在 **exec 级变体选择器缺能力先验层** —— 今天两族不为同一内核竞标（[B-6]），常量分排序无害；一旦 P7（矩阵范式接管 GEMM prefill）落地，升序 `RVV(1.0) < IME(20.0)` 令矩阵范式**静默落败无 error**。**关闭 [SEL-2] 门的最小动作 = `forced` stub 矩阵候选复现"静默落败→先验层修复"（in-tree、便宜、与自有 IME GEMM=gated X2 及 T4b e2e 双解耦）**；IME perf 曳光弹（K1 kernel-轴单 GEMM）是**独立第二件**、增益已历史 banked（5.66×/12.9×）、且**诚实预声明 kernel↛memory-bound e2e**。建议：先验层验证先行（可并行/独立立项），IME perf 复测与自有 IME GEMM 延后并入后续 K1 板会话，不与 T4b 的 lib/dispatch 战役并跑。

---

## 1. IME 原语现状（编译器里的真实状态）

### 1.1 结构（N2 已证，不再 re-litigate）
- **plugin 走同一 common pipeline**（commit `2eeabff9`）：op 挂 common EmitC 接口，capability-fact dispatch = `lookupProviderByID("spacemit.ime").isAvailable()`（`IMEExtensionPlugin.cpp:337-343`，**非**字符串族名比对），零-core-branch（core/common pass 无 `if IME`）。这是 [I3]/[F-6] 独立性硬判据满足项。
- **能力派生**：`deriveIMEMatmulCapability`（`:184-335`）从 march `xsmtvdotii` + VLEN 派生 MAC fragment。**fail-closed 强**：march token 缺、VLEN≠256、signedness 越界、shape 非 MAC 整数倍 → 全拒。**VLEN=256 是唯一 real-K1-validated shape**（`:237-247`，VLEN128 直接拒），即 N1 能力钩子在 IME 上的体现。
- **硅片**：真 X60 `taskset -c 0-3`（harts 0-3 才带 `_ime`，hart 4 SIGILL = per-hart 异构事实），bit-exact 16/16 vs scalar。march `xsmtvdotii` + SpacemiT GCC15.2 fork。

### 1.2 已建原语（6 个 boundary op，`IMEOps.td`）

| op | mnemonic | 类型 | 状态 |
|---|---|---|---|
| `tcrv.ime.mma` | `vmadot` | signed int8→int32 4×4×8 MAC fragment（**dot ii**） | 建+K1 bit-exact |
| `tcrv.ime.mma_u` | `vmadotu` | unsigned MAC | 建+bit-exact |
| `tcrv.ime.mma_su` | `vmadotsu` | mixed-sign（signed A × unsigned B，**量化主用**） | 建+bit-exact |
| `tcrv.ime.mma_us` | `vmadotus` | reversed mixed-sign（完成 signedness 家族） | 建+bit-exact |
| `tcrv.ime.mma_slide` | `vmadot1/2/3` | sliding-window（conv/A-reuse，IME1 第二子扩展） | 建+K1 4-way discriminator |
| `tcrv.ime.matmul` | `vmadot`/`vmadotu` | **tiled whole-matrix** MxNxK（signed/unsigned only） | 建 |

- **signedness/shape/slide 三轴均是能力派生 FACT**（`ime_signedness`/`ime_matmul_shape`/`ime_slide` property），boundary op 路由是纯 data-flow of the fact（`singleFragmentBoundaryOpForVariant:352-369` 读回 stamped attr、不 re-classify 族名）。

### 1.3 关键缺口（诚实）
- **无 tcrv-owned 完整 IME repack-GEMM**。`tcrv.ime.matmul` 是 tiled boundary op，但它**发射 FOUNDATION-validated `vmadot` LEAF kernel 经 common EmitC route**——历史 12.9× prefill GEMM 数字用的是 **ggml-spacemit `ime1::gemm_kernel_i8i4`**（tcrv 只 emit 它由之构成的 IME LEAF op vmadot/mma_u/mma_su/mma_slide，**非** 一个完整 tcrv IME GEMM）。**完整自有 IME GEMM = spec 明示的 gated X2**（`generation-selection-tuning.md:51`）。调 vendor lib / ggml IME 路径**不是 N2-novel**（memory `k1-ime-n2`）。
- **IME 未接入任何能力先验选择层**。`estimateVariantCost`（`:574-592`）发常量 `score=20.0` + `explicitPreference=true`，但这只是 exec 选择器的**能力盲常量分**（[B-6] 每插件常量），不是能力派生先验。
- **无 IME kernel-轴 perf cell**（`experiments/` 下只有 `_templates/T5{a,b,d}_ime_*.csv` 模板骨架，无实测 cell）。历史 IME perf 数（5.66×/12.9×/e2e 0.86×/1.65×）住 memory + `artifacts/n2-ime/IME-PERF-REBASELINE-FINDING.md`，非 T8 ledger 行。

---

## 2. 家族级先验方案（IME 怎么进选择器 SEL-1/SEL-2）

### 2.1 ★两个选择器要分清（否则错锁）
- **SEL-1 T3 已落地的是 TILING 选择器**：RVV block-dot GEMM emitter 内部的 tiled(S6)/plain 变体选择，键 = `fold_model` 瓶颈形状，reason ∈ {prior,measured,only_feasible}（commit `d2a14781`，门⑦ PASS）。**这是 RVV 族内、向量范式内的选择。**
- **IME gating 活在 exec 级变体选择器**：`ExtensionPluginRegistry::rankKernelVariantsByCost`（`ExtensionPlugin.cpp:1383` `std::stable_sort`）——这里 **RVV 向量范式候选 vs IME 矩阵范式候选**为同一 GEMM 内核竞标。**SEL-2 / IME 先验就锁在这一层。**
- **两层同构、复用同一 [SEL-1] 两段式 + 同一 reason 枚举**，但作用在不同粒度。T3 证明的是 tiling 层；IME 入选是把**同一先验层机制**上移到 exec 层。

### 2.2 ★定量潜伏错选（[SEL-2] 根因，已可判）
排序 comparator（`ExtensionPlugin.cpp:1383-1391`）：先 explicit-preference（RVV/IME 均 true = 平），再 `getScore()` **升序**（小者胜）。
- 已核实分值：**RVV `score=1.0`**（`RVVExtensionPlugin.cpp:680`）· **IME `score=20.0`**（`IMEExtensionPlugin.cpp:581`）· Scalar `1000.0`（末位兜底）。
- **今天无害**：两族不 co-propose（[B-6]，向量/矩阵现不为同一内核竞标）→ 常量分排序无害。
- **P7 落地瞬间被激活的错选**：矩阵范式接管 GEMM prefill（[PAT-2] P7 = `ime ∧ shape`）令两族对同一 GEMM co-propose → 升序 `RVV 1.0 < IME 20.0` ⇒ **向量变体被选、矩阵范式静默落败、无 error**。这正是 [SEL-2] 硬时序（先验层必须**先于/同于** P7）的根因（`执行总纲v2 §5` 已定量记此为"潜伏错选"）。

### 2.3 能力键 + 冷启动先验（IME 作为一个能力键控变体）
- **能力键** = `ime.present`（= `spacemit.ime` capability fact 在场且 available）。**这是 RISC-V capability 事实**（march `xsmtvdotii` 派生），满足 [n2-family-entry-boundary]（integrated 家族#2，复用向量寄存器堆，非独立离散卡）。
- **[SEL-1] 冷启动先验（spec `:43`）**：`GEMM 形 ∧ ime.present → 矩阵范式变体`（[L-3]：向量核上的**矩阵范式**，非"矩阵家族"）；否则寄存器预算内最宽 LMUL；否则默认档。**独立于成本函数、不污染成本纯度**（成本住测量库、按 instance-hash 键控）。
- **reason 主键**（[D-4]①）：今天 exec 层只发 `only_feasible`/`static_order`（`VariantSelection.cpp:1077-1091` 注释明示 `prior` 严格保留、[SEL-1] 落地前绝不发）。IME 先验落地 = P7-GEMM 内核的 `static_order` 归零→翻 `prior`。**`static_order` 出现数是燃减/诊断信号**。

### 2.4 与 [XFER-1] / bottleneck-shape 类的关系
- **[XFER-1]** = tiling 选择器的冷启动 3-类能力先验（min-fold register-cliff→S6Tiled / dual-plane weight-bound→Plain / already-lean→Plain，键=`fold_model` 瓶颈形状；SEL-1 T1 设计 §④）。
- **IME 入选 = 又一个 bottleneck-shape 类**：**compute-bound prefill GEMM 形 → 矩阵范式**（IME）；**memory-bound decode（M=1 GEVM）→ 非矩阵**（矩阵单元帮不上、见 §3 诚实教训）。键仍是**瓶颈形状（compute-bound GEMM-shape × ime.present），非 format/族名** → 满足 C3′「换键不改条目」迁移判据，且从机制上防族名分派复发。
- **[PAT-2] P7** 是 IME 键控的模式库条目（矩阵范式接管）；IME 是它的承载 family（canon `科研目标总纲v2:181` 已记 SpacemiT IME = 家族#2 integrated、keystone、C3′-P7）。

---

## 3. 曳光弹计划（两件独立，勿混）

### 3.1 ★曳光弹 A —— [SEL-2] 先验层验证（选择器科学，**这才是关门动作**）
- **不阻塞在自有 IME GEMM 上**（spec `:51` 明示）：复现"矩阵静默落败→先验层修复"只需**一个能与向量变体对同一内核竞标的矩阵候选**。
- **载具** = `forced` stub 矩阵候选（`selection_mode = forced`，**仅供消融科学、不入产品主张**）与 RVV 向量变体对同一 GEMM co-bid。
- **两步**：① 复现——stub 矩阵候选 + 向量候选竞标同一 GEMM，观察升序常量分令矩阵 `static_order` 静默落败（in-IR reason + 归因 JSONL 双证）；② 修复——落 [SEL-1] 冷启动能力先验层于 exec 选择器（`GEMM ∧ ime.present → 矩阵范式`），reason 翻 `static_order→prior`、矩阵候选被选。
- **成本/触碰面**：in-tree（lit + `VariantSelection.cpp`/`ExtensionPlugin.cpp` cost 路），**无需上板、无需 X2、无需 T4b e2e glue**。这是 SEL-2 gating 的最小、最高 C3′-价值-每-单位-成本动作。

### 3.2 曳光弹 B —— IME perf 最小验证（K1 板、kernel-轴、任务所问）
- **靶**：单 GEMM `tcrv.ime.matmul`(IME) vs RVV block-dot/repack GEMM，K1/VLEN256，kernel-轴 micro。
- **历史 banked**（`IME-PERF-REBASELINE-FINDING.md`，compiler-held-constant vs ggml 真 shipped RVV `vec_dot_q4_0_q8_0`）：**M=1 5.66× / prefill GEMM M≥4 ~11.7-12.9×**（分解 = 2.45× IME-MAC × 2.35× ggml-shipped-weakness）。★DISCLOSED：kernel = ggml-spacemit `ime1::gemm_kernel_i8i4`（tcrv emit LEAF op、非完整 tcrv IME GEMM）。
- **★诚实预声明（load-bearing，必须写进任何 IME perf 措辞）**：compute-bound kernel 胜**不传导** memory-bound decode e2e —— IME e2e = **0.86-0.98× NULL**（memory wall，M=1 GEVM 矩阵单元帮不上，memory `kernel-wins-dont-transplant-to-e2e`）。可辩护 e2e = SPACEMIT **backend FAMILY** ~1.65× Q4_0 prefill（227-symbol kernel 家族 swap），**明确不是矩阵单元**（`IME-E2E-SPACEMIT-TOGGLE-FINDING.md`：IME-unit e2e win 不可 cleanly 隔离，within-prefill 比在 M≈32 峰后衰减 = FLOP-intensity win 的反面）。
- **载具**：复用 `kquant_gemm_paired.sh` harness 模式（合成 buffer、kernel-轴），BOARD=k1，单 GEMM shape，N=12 cold + `llvm-objdump --mattr` 封。**kernel-轴 ONLY、[NG-4] beat 措辞锁**（八门未走）。
- **优先级**：**低于曳光弹 A**。理由：(i) perf 增益已历史 banked，重测是 re-characterize 非新发现；(ii) 不传导 e2e；(iii) 完整自有 IME GEMM 才是真 novelty 且是 gated X2。曳光弹 B 只 characterize kernel-轴、不解锁 e2e/beat。

---

## 4. 与 T4b 统筹 + 开工时点建议

### 4.1 现状约束
- **task.json 明写**："SEL-1 T4 验收 = SEL-2/IME 前置"。但 spec `:51` 明示 **[SEL-2] 先验层验证与自有 IME GEMM(X2) 及 T4b e2e 双解耦**——"接 G3 pillar 的 agent 勿把先验层验证误锁在 X2 之后"。故 gating 的**关门动作（曳光弹 A）不必等 T4b**。
- **T4a 已完成**（commit `5b339407`，q4_K 6/8 kernel-轴双板）。**T4b = e2e-seal 集成 = NO-GO 单会话、多会话独立战役**（SEL-1 T4 计划 §5）：从零补 G1 离线 repacker + G2 q8_K×4 激活交织 + G3 mul_mat dispatch patch，触碰 `lib/`(dispatch/激活交织) + 板上维护 A-tree patch，且 build 与线 B(dequant batch)共享。

### 4.2 资源冲突分析（memory `parallel-lines-need-disjoint-files`）
- **曳光弹 A（先验层）触碰集** = `VariantSelection.cpp`/`ExtensionPlugin.cpp` cost-路 + 新 exec-层先验层 + lit。
- **T4b 触碰集** = `lib/` dispatch/激活交织 + 板 A-tree patch + build。
- **两者文件集基本不相交**，但 exec 选择器 cost 路是**共享基建**（多线天然跨线共享）→ 按 memory 应**串行 or 明确不相交守卫**，不盲信 worktree 隔离。曳光弹 A 本身轻（选择器 wiring 非 body construction），一次会话可收。

### 4.3 ★开工时点建议（供用户裁）
1. **曳光弹 A（[SEL-2] forced-stub 先验层验证）= 建议先行 / 独立立项、可与 T4a 后续并行**。理由：in-tree、便宜、**这是真正关 [SEL-2] 门的动作**、C3′-价值最高（先验层是 pillar C 的核心 novelty）、与 X2 及 T4b 双解耦。**不需等 T4b。**
2. **曳光弹 B（IME perf K1 kernel-轴复测）= 延后、并入后续 K1 板会话**（像 T4a 那样一次板会话），**不与 T4b 的 lib/dispatch 战役并跑**（build + 板资源冲突）。增益已 banked、不传导 e2e → 低边际价值。
3. **自有 IME GEMM（X2）= 延后**，非 SEL-2 前置、大工程、真 novelty 但排在先验层与 e2e-seal 之后。
4. **P7 硬时序守卫**：建议把"P7 落地前 [SEL-1] 先验层常绿"设为门禁（`科研目标总纲v2 M3` 已有此项）——P7 与先验层的相对时序是 [SEL-2] 硬契约，不可 P7 先落而先验层缺席。
5. **整体 IME 开工时点**：按任务指令，**开工与否 + 具体时点在 T4b 进度明朗后报用户裁**；本报告不自决开工。

---

## 附 touch-set / [NG-4] / 诚实纪律
- 本文 = 纯 read-only 复核 + 计划；**无新 board 数字**（所有 5.66×/12.9×/0.86×/1.65×/3.106×/1.884× 引自 memory `k1-ime-n2`/`kernel-wins-dont-transplant-to-e2e`、`artifacts/n2-ime/*FINDING.md`、T8 ledger、SEL-1 T4 计划已存在 cell）。
- **诚实核心（必须随 IME 主张同行）**：① IME kernel 胜**不传导** memory-bound e2e（campaign-central finding）；② 历史 12.9× 用 ggml-spacemit kernel、**非完整 tcrv IME GEMM**；③ 调 vendor lib 不是 N2-novel；④ N2 结构已证 + reviewer-clear，勿 re-litigate "能否加家族"。
- **无 `lib/`/`schema/`/`experiments/` 改、无 git stash/rm/mv/add/commit。** 唯一新增 = 本文 `docs/reports/2026-07-09-IME-gating-review.md`（未 commit，留用户提交）。开工时点 = 用户裁决点。
</content>
</invoke>
