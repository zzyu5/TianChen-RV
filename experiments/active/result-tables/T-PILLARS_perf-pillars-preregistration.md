# T-PILLARS — 性能立柱假设预注册（货架C·体例级·G7 补充令二）

> **建成**：2026-07-13（G7 补充令二·用户裁）。**性质**：预测先行、事后对账。两柱假设**正式预注册**，供 kernel-sym 全量表（货架A）+ e2e 矩阵（货架B）验证。表满后按本预注册对账命中率，**不符行如实列出（负例 = 边界·C3′ 素材）**。
> **禁**：事后改预测（预注册锁定）；对账时挑数据。

---

## 柱一 · 结构对齐（plan 库 / 能力入模式库）

**机制**：我方 kernel 的性能取决于是否有**结构对齐的 Emission Plan**（[K-1]·[K-10] 结构级）匹配该 op×regime 的迭代拓扑/布局契约/优化目标。

**预注册预测**：
- **对手实现无折中的格（结构同拓扑·对手为该 regime 专门实现）** → 我方 kernel-sym **parity ~ 小赢**（同结构·拼参数级精调）。
- **对手有折中的格（对手单实现硬扛多 regime / 多 VLEN / 上游破损）** → 我方 **倍数赢 或 正确性赢**（我方结构对齐击败对手的折中）。

**判别键（L2 全量填充时每行必标）**：`对手折中状态 ∈ {无折中 | 单实现折中 | 破损}`。

## 柱二 · 实例精调（能力键控 / 参数级迁移）

**机制**：固定结构内，性能由**参数键控**（[SEL-1]·VLEN/LMUL/tile/展开·[K-10] 参数级）落地度决定。

**预注册预测**：
- **同格跨板换键不改条目**（[PAT-3] CI 常绿）**且各板取各自最优**（k1/VLEN256 与 rvv/VLEN128 各选最宽合法 LMUL/tile）。
- **键控杠杆消融对（on/off）单调正向**（每个已键控杠杆 on 优于 off·无反噬；反例 = re-roll trap，那是结构级误当参数级，不算柱二反例）。

---

## 对账规则（表满后·货架A/B 填充完成时执行）

1. kernel-sym 全量表（货架A·84 格 × hot/cold）每行标 `{对手折中状态}` + `{结构对齐? 参数键控?}`。
2. 按柱一预测对账：无折中格是否 parity~小赢？折中格是否倍数赢/正确性赢？**命中率逐桶算**。
3. 按柱二预测对账：跨板换键条目是否不变？各板是否各自最优？键控消融对是否单调正向？
4. **不符行如实列出**（负例 = 边界·记入货架C 归因体例·C3′ 素材·非粉饰）。

## ★可证伪延迟检验点（预挂·KNEST plan）

**q2_K / q6_K 预挂「柱一缺口（KNEST plan 缺席）」预测**：
- 现状 = q2_K/q6_K 属"对手有折中但我方**也缺结构对齐 plan**"（[GAP-EMIT-KNEST]·K-quant 超块流式嵌套 plan 缺席·结构级欠账）→ 故 kernel-sym LOSS（q2_K@rvv 0.386×对称/q6_K ~0.18×）。
- **延迟检验**：KNEST plan（结构级·超块流式嵌套·单宽 resident accumulator）**若未来落地**，该批格**应转赢**（柱一预测：我方补上结构对齐后击败对手折中）。
- **若 KNEST plan 落地后 q2_K/q6_K 仍不转赢** → 柱一预测**证伪**（结构对齐非充分·另有因）= 边界发现。
- **★状态更新（2026-07-13·L4 消融归拢·`93f9b984`）= OPEN·prior 下调**：KNEST plan 未落地；但 **GEVM P2 board-falsified（67d49316·结构对齐 plan 落地了却 register-pressure 反噬）+ G6-B P3 exit-b** 两条间接负证据 → **延迟检验预期"部分证伪"风险高**（结构对齐落地 ≠ 自动转赢·见下 register-budget-fit）。仍待 KNEST 真落地才裁。

## ★G7 实测精化（2026-07-13·L4 消融归拢·两柱均命中带边界·`93f9b984`/`67d49316`/`b29c269c`）

**★★register-budget-fit 统一律（durable·三级镜像·G7 核心方法学发现）**：「必要非充分」在三层同底——
- **编译器级**：gcc-death 消除【必要非充分】·还须 emitter-quality-beat（q4_K WIN / q2_K·q5_K LOSS·[[q4-0-e2e-is-routing-not-kernel]]）。
- **参数级**：加宽 tile/LMUL【必要非充分】·还须 fit 32-vreg 预算（M7 W4 越预算反噬 / q5_K per-board register-cliff·[GAP-Q5K-VLEN128-QH-REGCLIFF]）。
- **结构级**：结构对齐 plan【必要非充分】·还须 register-budget-fit（**GEVM P2 board-falsified**·67d49316·TG=2 bank 加宽 M=1 spill 53×→IPC 崩）。
⟹ **柱一精化**：`结构对齐 → 赢` 改判为 `结构对齐【必要非充分】∧ register-budget-fit → 赢`。GEVM P2 **非柱一全证伪**——P1 byte-exact mechanized 证明 **plan 库 extensibility（柱一本体·C1）DEMONSTRATED**·P2 证伪的是"首版结构假设（TG=2 bank 加宽提 IPC）"·柱一 thesis（结构对齐是赢的必要条件）仍立·仅补 register-budget-fit 充分条件。

**柱一「无折中→parity」精化**（`b29c269c`）：前提 = **我方 HAVE 匹配结构 plan**。反例：q4_K@k1 vs 真 hand-brick 0.622× LOSS（对手无折中[VLEN256 满宽专调]·我方缺匹配结构[全展开 25KB VLEN-invariant]→LOSS）。⟹ `对手无折中 ∧ 我方有匹配结构 → parity~小赢`（缺任一则可 LOSS）。

**柱二边界补注**：M7 W2 单调正向命中；边界三处 =（① W4 越 32-vreg 预算反噬 ② q5_K per-board register-cliff ③ IME format-keyed e2e 传导[q8_0 传导/q4_K 稀释]）。re-roll trap 正确排除（结构级误当参数级·[K-10] 实证②·非柱二反例）。

## 关联

- [K-10]（core-invariants·结构级/参数级判据）· [PAT-2] P9（GEVM plan·柱一第①实例修法）· [GAP-EMIT-KNEST]（KNEST plan·柱一延迟检验）· [GAP-REPACK-GEVM]（GEVM plan·柱一 decode 修法）。
- 货架A = kernel-sym 全量表（`T9_kernel_sym_ledger.md`·柱二主验证场）· 货架B = e2e 矩阵（柱一 e2e 传导验证）· 货架C = 本表 + T3p 消融 + T4b。
