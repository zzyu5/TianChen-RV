# Research: strip-width 导出接轴（§四.4 / [SEL-3] 首次真实改判）现状

- **Query**: strip-width 导出接轴现状 · K宽化 iq3_xxs@k1 是否已满足 §四.4 / [SEL-3] 首次真实改判
- **Scope**: 内部（lib + spec + ledger）
- **Date**: 2026-07-18

## 结论（一句）

strip-width 导出**机制已在代码**（`RVVRepackStripWidthMaterialization.cpp` · `deriveRepackHalfLanes`），且 K宽化 iq3_xxs@k1 已产出「同算子两份能力配置·选择不同·objdump 双产物+归因」的活证。**但**《开测篇》§四.4 的两件要求 —— (a) **strip-width 导出接轴** 与 (b) **[SEL-3] 测量记忆层首次真实改判** —— **只满足了 (a) 的能力键控侧，(b) 未满足**：K宽化的翻盘是**能力先验（VLEN 事实）驱动**，不是**测量记忆 verdict 超越能力先验**的翻盘；且 iq3_xxs 是 GridCodebook grid-gather leaf，**不是** `deriveRepackHalfLanes` 覆盖的 q4_0/q4_1/q8_0 repack 算子（见下「关键订正」）。

## strip-width 导出接轴：机制已在代码

`lib/Plugin/RVV/Schedule/RVVRepackStripWidthMaterialization.cpp`：
- `deriveRepackHalfLanes(vlenBits, weightInterleave)`：`half_lanes = min(vlen/16, weightInterleave)`；**128→8 · 256→16**。从**保证最小 VLEN**（`deriveMinimumVLEN(march, isaVectorHints)`）导出，经 plugin-local authority（I1/I4/I5/I7 核心不变量），**不探硬件、不读 toolchain 事实**。
- 覆盖算子（module.walk dyn_cast）：`GgmlRepackGemvQ41Q81Op` · `GgmlRepackGemmQ41Q81Op` · `GgmlRepackGemvQ80Q80Op`（q4_1 GEMV/GEMM + q8_0 GEMV）。
- **q4_0 GEMM/GEVM 单体 op 已 RETIRED**：其 strip-width 现在**构造时**（repack 前门 `RVVLowerQuantContraction.cpp lowerToRepackGemm/Gemv`）直接烘进 typed region（region 结构 baked to half_lanes·不能 post-hoc 重 stamp）。
- 第二分歧轴 = ISA generation（RVV0.7 无 fractional LMUL → pin whole-LMUL core `m1` + half_lanes=16）。

## strip-width 键控接入的 spec 状态（ISSUE-035）

`.trellis/spec/issues/发射器与架构.md` **ISSUE-035**（状态：**已就绪**·是工·C3 节点）：
> 宽度轴测量键现为**预留空 seed**（键位已在、值未接线）；接入 = 测量记忆层**第一个「记忆 verdict 超 authority」的翻盘实例**，同时闭合宽度轴既有键控缺口。

⟹ §四.4 的 [SEL-3] 首次真实改判的**目标形态** = 记忆 verdict 超能力 authority。这在 ISSUE-035 明列为「已就绪·未施工」。

## [SEL-3] 测量记忆层的诚实边界（evidence §2.3）

`.trellis/spec/evidence/三贡献证据地图.md §2.3`：
> **★诚实边界（必随行）**：authority 层**非翻盘** —— 记忆 verdict 与能力先验**一致**。价值 = 测量 argmin **AUTHORITY** + fail-closed-revalidate + byte-exact provenance。**禁夸「记忆翻盘」。**

记录键 = 复合主键 `(declared_instance_hash〔含 march/vlen/vreg〕, kernel, variant, op, engine, regime)`（`canon/覆盖状态机与选择归因.md §二.6 [SEL-3]`）。`selection_valid_input` = **配对变体严格语义**（仅配对 A/B true·单点 vs-opponent false）。

## K宽化 iq3_xxs@k1 活证（ISSUE-105 / ledger 机制①）

- 同叶 `march=rv64gcv`（VLEN128·half_lanes=8）vs `march=rv64gcv_zvl256b`（VLEN256·half_lanes=16）→ cold **0.65 LOSS → 1.38 WIN**。
- objdump 双产物真宽：AVL `vsetivli zero,8,e32,m2` → `zero,16` · vset 5397→2515 · gather 1024→512（**非 re-roll**）。
- G1 byte-exact PASS（4-arm anti-hollow 真隔离）· 2-seed（1.3838/1.3782）。
- **★关键订正（ledger 机制① 原文）**：iq3_xxs@k1 的真叶 = **`RVVToEmitCBlockQuantLinear.cpp:24897`·非 GridCodebook.cpp**；宽化器 `deriveRepackHalfLanes` **已存·未改源码**（换 march 即换 half_lanes）。

## §四.4 满足度评估

| §四.4 要件 | K宽化 iq3_xxs@k1 是否满足 | 说明 |
|---|---|---|
| strip-width 导出接轴 | **机制满足**（`deriveRepackHalfLanes` 已导出·march→half_lanes 键控真发散） | 但 ISSUE-035 记宽度**测量键**仍「预留空 seed·值未接线」 |
| 同算子两份能力配置 | **满足** | 同叶 · VLEN128 vs VLEN256 两份 march 配置 |
| 选择不同 | **满足** | half_lanes 8 vs 16 · objdump 双产物证 |
| objdump 双产物 | **满足** | vset 5397→2515·gather 1024→512·AVL zero,8→zero,16 |
| 归因日志 | **满足**（task journal + `experiments/runs/20260717T193812Z-iq3_xxs-k1-a4db452a/`） | objdump 前后 + verify/measure stdout + row.csv |
| **[SEL-3] 首次真实改判（记忆 verdict 超 authority）** | **未满足** | K宽化翻盘 = **能力先验（VLEN 事实）驱动**·非记忆 verdict 超越先验；evidence §2.3 明文「authority 层非翻盘·记忆 verdict 与能力先验一致」 |

## 结论：K宽化**部分**满足 §四.4，**不**满足 [SEL-3] 首次真实改判

- **满足**：strip-width 导出接轴的**能力键控侧**全部要件（两配置·选择不同·objdump 双产物·归因日志）——iq3_xxs@k1 是能力键控 strip-width 首次真实**发散**的活证。
- **不满足**：[SEL-3]「记忆 verdict 超 authority」的翻盘 —— 那要求**测量记忆库的 argmin 覆盖能力先验的默认**，K宽化是 VLEN 事实（能力先验）直接导出 half_lanes，属 authority 一致方向，非「记忆翻先验」。ISSUE-035「记忆 verdict 超 authority 的翻盘实例」**仍未施工**（宽度测量键仍空 seed）。
- **部署边界**（ISSUE-105·待裁）：1.38 是 **PROVEN·NOT DEPLOYED**；标准 k1 分支 march 无 zvl256b ⟹ deployed 叶仍 VLEN128=0.65·master 维持具名-X 0.6474。

## 对 R 线 dequant 的接轴意义

dequant 真向量发射器若首攻 iq3_xxs，其 emit 会经 GridCodebook grid-gather 机体（`coreLmul` anchor m2/m1）——与 strip-width 的 half_lanes 是**不同参数轴**（GridCodebook 用 coreLmul + EMUL·repack 用 half_lanes）。§四.4「strip-width 导出接轴」在 R 线的落点 = 把宽度轴接进 dequant leaf 的能力键控（现 GridCodebook 4/5 格式宽度焊死为字面量·ISSUE-031/033 假旋钮问题）。**dequant 轴的 strip-width 接入 ≠ 复用 K宽化的 iq3_xxs@k1 gemm 证据**（不同 op·不同参数轴），需独立落地。

## 诚实边界 / 未找到

- **未找到** 宽度测量键「接线」的代码（ISSUE-035 记其为「预留空 seed」）·未在 `RVVRepackTilingSelection.h` 逐行确认 seed 位置（本轮只读 evidence §2.3 的工件指针）。
- [SEL-3] 首次真实改判是否有**其他候选**（非 strip-width）本轮未查。
