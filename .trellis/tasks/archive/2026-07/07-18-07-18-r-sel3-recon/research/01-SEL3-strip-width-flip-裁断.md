# Research: §四.4 [SEL-3] strip-width「首次真实改判」是否存在真实 memory-over-authority flip

- **Query**: strip-width 轴是否有真实「记忆 verdict 超 authority」翻盘 or 诚实 null
- **Scope**: 内部（lib + include + spec + evidence + ledger + master 表）
- **Date**: 2026-07-18

## ★★裁断（一句）

**诚实 NULL —— 不存在真实的 memory-over-authority flip，strip-width 结构上也无法在固定 capability 下承载一个。** §四.4「[SEL-3] 首次真实改判（记忆 verdict 超 authority）」**honestly 不可满足**。这是一个 valid finding（非失败），evidence §2.3 与 T4b §0.5/§2.1b 已多处明文「authority 层非翻盘·禁夸记忆翻盘」。建议 §四.4 该项登记为 **honest-null**，禁硬造 flip。

---

## 1. 测量记忆层机制现状（selection_valid / write-back / authority 层）

**代码**：`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h`（PURE·cost-model-free·unit/lit-testable）。
**schema**：`schema/measurement-memory.v1.json`（v1.0.0·活）+ `schema/measurement-memory.design.json`（草案）。
**写回**：`.trellis/scripts/sel3_writeback_harmonizer.py`（离线批处理·非在线学习）。
**验收**：`experiments/active/result-tables/T4b_selector_ablation.md` §2.1b。
**设计稿**：`experiments/active/g8-stage3-attack/SEL3-measurement-memory-design.md`。

三段式选择器（`selectRepackTilingVariant` / `selectRepackLoopOrder`·`.h:309`/`:498`）：
1. **Stage-1 合法性过滤** `tilingVariantFeasibleSet`（`.h:149`）：`vlen<128||vreg<=0` → 空集 → `static_order` fail-safe（生产路径 INERT·前门 gate 挡掉·`.h:319-331`）。
2. **Stage-2a memoized argmin**（`.h:340`）：`lookupMeasurement(hash,kernel,axis)` 命中且仍 feasible → `reason=measured`。**fail-closed-revalidate**：陈旧/不再 feasible 缓存丢弃退先验。
3. **Stage-2b 能力先验**（`.h:348`）：keyed on bottleneck SHAPE / stride fact → `reason=prior`。

**复合主键（canon §二.6）**：`(declared_instance_hash〔含 march/vlen/vreg〕, kernel, variant, op, engine, regime)`。`selection_valid_input` = **配对变体严格语义**（仅 `ab_paired`/`vs_generic` 同编译器 true·`vs_shipped_opp` 跨编译器 false）。

**★记忆库真实 seed 内容（`lookupMeasurement` kSeeded[]·`.h:261-271`）= 6 行·全在 rvv/VLEN128 hash `3cd23a4e…`**：
- SP4Tiling 5 行：q4_K/q2_K/q5_K → s6_tiled·q6_K/q3_K → plain。
- LoopOrder 1 行：q4_K → col_outer（M1b 板 A/B 2.47×·编译器对称）。
- **StripWidth 0 行**（见 §2）。

memory `[[sel3-and-goal-queue-complete]]` 已锁：记忆层完整·selection_valid 恒 12（schema 12 行 selection_valid=true）·write-back fail-closed vindicated·**authority 非翻盘**。

## 2. width 测量键 seed 态（「预留空 seed」具体是什么）

`RVVMeasurementAxis::StripWidth`（`.h:183`）= 枚举成员已在（键位在）。但：
- **`.h:179-180` 明文**：`StripWidth = a RESERVED extension slot (vl8/vl16) with NO live seed today (schema's strip_width axis is a future T-SEL3-4 writeback)`。
- **kSeeded[] 无任何 StripWidth 行**（`.h:261-271` 只有 5 SP4 + 1 loop_order）。
- **schema `variant_axis_registry.strip_width`**（`measurement-memory.v1.json:84-94`）：variants `{vl8,vl16}`·axis_extras `{vlen_gate}`·**无 rows[]**·seed 注：`"q4_K@k1 vl16 1.197x>vl8 (Win-K1-VLEN); NOT transcribed here (sealed vl16 kernel data lives in perf-covered/sealed, not tiling-measurements; a future writeback seeds it)."`

⟹ 「预留空 seed」= **枚举键位 + schema 轴注册在·但 rows[] 空 · lookupMeasurement 对该轴恒 miss → 恒退先验**。接线 = 未来 T-SEL3-4 写回一枚过字节门的 strip-width A/B 行。

**独立宽度导出机制**（≠ 记忆层·capability-prior 侧）：`lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp` `deriveRepackHalfLanes(vlen,interleave)=min(vlen/16,interleave)`（128→8·256→16）。这是**能力先验直接导出**，不查记忆库。

## 3. ★flip 候选存在性（遍历有 strip-width 轴的格）

**判据**：真 flip = 固定 capability（同板同 VLEN）下，capability-prior 选宽度 A，但**同板板测记忆 verdict 说 B 更快**，selector 用记忆覆盖 prior 选 B（prior 与 measurement 真矛盾）。

**遍历结果 —— 无一格满足**：

| 格 | capability-prior 选 | 板测最快 | 矛盾? | 说明 |
|---|---|---|---|---|
| q4_K@k1 (VLEN256) | vl16（256/16=16） | vl16（1.197× decode·e2e 1.085×·Win-K1-VLEN） | **无** | prior=measurement（VLEN256→16·全宽赢）。master 表 `T3_B_..._vlen256.csv:35`「vl8-was-0.622x-flip」的 "flip" = 把旧 hand-placed vl8 部署换成 capability-正确的 vl16 = **能力先验方向**·非记忆超先验 |
| q4_0/q4_1/q8_0 repack @rvv (VLEN128) | vl8（128/16=8） | vl8（16-way interleave 下唯一满宽） | **无** | 单一 capability-legal 宽度 |
| q5_K@rvv (VLEN128) | vl8 | — | **无** | `q5k-rvv-regcliff-fix-G1/evidence.md`：VECTOR spill 在 VLEN128 与 VLEN256 **完全相同**·WIN/LOSS 判别 = numHalves tile-doubling + qh 指令量·**register-pressure 减压是错的杠杆**⟹ 无「窄宽因 regfile 压力反快」的 flip 机制 |
| iq3_xxs@k1 K宽化 (VLEN128→256) | half 8→16 | 8→16（0.65 LOSS→1.38 WIN） | **无·任务已排除** | capability-prior-DRIVEN（换 march→换 half_lanes）·prior 与 measurement 一致·跨板发散非固定 capability flip |

**结构性论证（为何 strip-width 无法承载 flip）**：strip width 由 VLEN **确定性导出**（`min(vlen/16,16)`），16-way interleave 安全不变量下**每板只有一个 capability-legal 宽度**（VLEN128→vl8·VLEN256→vl16）。vl8 与 vl16 只在**跨板**（不同 VLEN）区分——那是 capability 发散（iq3_xxs@k1 K宽化演示的东西），**不是**固定 capability 下的 A/B。在 VLEN256 跑 vl8 只是人为欠用上半 lane（严格更差），测 vl16>vl8 只**确认**先验（用满你的宽度）。故 strip-width 在固定 capability 下**候选集实为 singleton**（近似 L1 codegen_lmul 退化·非真 A/B）。

**跨全轴复核（不止 strip-width）**：SP4-tiling / loop-order **结构上能**在同板承载两 feasible 变体（能出 flip），但其 6 格 seeded verdict **全与 shape/stride 先验一致**（T4b §2.1b「★★L2 诚实命门」·`priorTilingVariantForShape` 本身已良设计·冷启动即对）。**全库无一格记忆翻正错先验。**

**全仓穷举确认**：`grep -rin "memory-over-authority|记忆翻盘|首次改判|首个翻盘"` 全仓**唯一命中** = evidence §2.3 的诚实边界「authority 层非翻盘」。无任何登记的 flip 候选。

## 4. 诚实结论

**诚实 null 成立**：所有测过/seeded 的格，记忆 verdict 与 capability-prior **一致**（prior 对·无覆盖）。strip-width 轴 seed 空且结构上无法在固定 capability 承载 flip。⟹ §四.4「[SEL-3] 首次真实改判」= **honestly 不可满足**。

- **满足的**（已达）：strip-width **能力键控发散**首次真实活证 = iq3_xxs@k1 K宽化（half 8→16·objdump 双产物·归因日志·`experiments/runs/20260717T193812Z-iq3_xxs-k1-a4db452a/`）—— 但这是 authority 一致方向，**非** memory-over-authority。
- **不满足的**：memory verdict 覆盖 capability prior 的翻盘 = 无（且结构上 strip-width 不可产）。

## 5. §四.4 处置建议（null·非硬造 flip）

1. **登记为 honest-null**：§四.4「[SEL-3] 首次真实改判」条 → 明标「诚实 null·记忆 verdict 全格平先验·strip-width 结构上不承载固定-capability flip」。承 evidence §2.3 + T4b §2.1b 既有诚实定格·**不新增 flip 主张**。
2. **ISSUE-035 状态订正建议**（禁自裁·登记供用户裁）：其「接入 = 测量记忆层第一个『记忆 verdict 超 authority』的翻盘实例」的**目标形态本身对 strip-width 不可达**（width 是 capability-forced singleton）。接线 strip-width seed 只能得到「记忆确认先验」（authority 一致），得不到「记忆超先验」。若要「首次真实改判」，须换轴到**结构上双-feasible 且存在 prior-错-measurement-对 的格**——当前全库无此格。
3. **若坚持接线 strip-width seed**（T-SEL3-4 写回）：产物 = q4_K@k1 vl16 行入库·`reason=measured`·但 verdict==prior（vl16）⟹ 仍是 authority-一致·**不得**措辞为「翻盘/改判」。
4. **铁律遵守**：false-flip 是严重违规。本勘察结论 = null·**如实报·禁硬造**。

## 诚实边界 / 未找到

- 未 ssh 板复跑（任务允许只读勘察·但既有 objdump/board 证据已足以裁 null·无需新测）。
- 「记忆 verdict 超 authority」的**其他非-strip-width 候选**：全库 grep 无登记 flip 候选；SP4/loop-order 6 格 verdict 全平先验（已核）。**无遗漏的 flip 候选**。
