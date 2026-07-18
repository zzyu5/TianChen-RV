# Research: ★判据裁定 — q4_K vec_dot@rvv memory-stall floor = 公式墙 vs 脾气墙（B 线核心）

- **Query**: lever (d) memory-scheduling 是公式墙〔可键控·必攻〕还是脾气墙〔硬件/clang 微架构脾气·走三步登记边界〕？
- **Scope**: internal（判据裁定·带 objdump + perf 证据·非赢没赢）
- **Date**: 2026-07-18

## 判据（canon 正本·`.trellis/spec/canon/覆盖状态机与选择归因.md` [K-4] 攻坚 backlog 三态）

- **公式墙** = 「可键控 / 闭式导出的**决策空间未算尽**」 → **必须攻**。
- **脾气墙** = 「**clang codegen / 硬件微架构脾气·非我方决策空间**」 → 走三步方登记边界、默认接着找。
- **边界登记合法性**：「**仅当杠杆清单真空 ∧ 走过三步**（objdump 对手 → 定杠杆 → 板测证伪留档）」。

---

## ★裁定 = 公式墙（FORMULA WALL·必攻·不得现在登记边界）

**主因**：memory-scheduling 轴（load 宽度 + 寄存器驻留多流 load）是**闭式可键控**的决策空间、且**从未被三 lever 触及（UNTRIED）**、且**对手在同一硬件证其可达**。现在登记边界会**违反 [K-4] 三步规则**（杠杆清单非空·板测证伪未做）。

### 证据链（四条·全 objdump/perf 硬证）

**证据 1 — 对手同硬件证可达（排除「硬件带宽不可逾越」）**
- 对手 ratio 0.186 = **同板同数据快 5.4×**。ours 与对手流同一 590KB 权重·若我方已在带宽饱和·对手不可能 5.4× 快 ⟹ **我方非带宽-bound·是延迟-bound·有 headroom**。
- IPC 0.08 / 96.2% backend-idle = 核**在等内存·非算力饱和**。gap = stall = memory-latency 暴露·**非 compute**。

**证据 2 — gap 全在 load 宽度 + MLP（两条可键控·闭式）**
- 对手隐藏延迟靠三件（`01-opponent-mlp-mechanism.md`）：宽 e8m1 load·权重寄存器驻留·**~16 条独立宽 load 提前发射（intra-super-block MLP）**。
- 其中「权重寄存器驻留」「向量化 min-term/归约」= **已试**（register-fusion/vwredsum/minterm-vec·cold inert）。
- **未试的两件 = load 宽度（e8mf2 8B → e8m1/m2 16–32B）+ load 流并行度（多流提前发射）** ——两者都是**纯代码生成决策**（LMUL 宽度 / 循环展开度 / load 发射顺序）= canon 定义的「可键控 / 闭式导出决策空间」= **公式墙**。

**证据 3 — 三 lever 未触及此轴 ⟹ 杠杆清单非空**
- 三 lever 全动**指令流轴**（消 scratch / 消 serial 链 / 向量化 scalar）·**没有一个改 load 宽度或 load 流数**。
- 「指令流轴 EXHAUSTED」为真·但「memory-scheduling 轴 EXHAUSTED」**为假**——[K-4] 三步的「定杠杆（memory-scheduling）→ 板测证伪」**从未做** ⟹ **不满足边界登记前提**。

**证据 4 — register-fusion 的 33× cache-miss↓ 而 cold inert = 根因在 load-scheduling·非已试三源**
- 见 `02-our-serialize-rootcause.md` §二：那 2.2B miss 是 aux8 的 L1/L2 流量·非 DRAM 关键路径。真墙 = 原始权重 DRAM 延迟被**低 MLP 串行暴露** ⟹ 正是未试的 load-scheduling 轴。

### 结论
> **q4_K vec_dot@rvv 的 memory-stall floor = 公式墙**（memory-scheduling 轴：load 宽度 + 寄存器驻留多流 load·闭式可键控·对手同硬件证可达·三 lever 从未触及）。**per [K-4] + B 线：必攻·给施工入口**（见 `04-construction-entry-if-formula.md`）。**现在登记边界 = 违反三步规则**（杠杆清单非空）。

---

## ★必带的诚实 caveat — 内含一个真实的**脾气墙-风险子墙**（决定攻坚的验收门与 5 次定格触发点）

裁定是「公式墙·必攻」，但攻坚**必然撞到并须检验**一个 clang-codegen 脾气墙风险：

- **对手用 inline asm 手排 load 调度**（绕过 clang 调度器强制 MLP）。**我方 emit = C intrinsic·调度权在 clang**。「独立宽 load 提前发射并保持在飞（MLP）」在 C-intrinsic 世界是**委托给 clang 的调度属性**。
- **已有反证**：register-fusion 把结构改干净（消 scratch·cache-miss 33×↓）后 clang **未自发** hoist 出 MLP·cold inert ⟹ clang-18 RVV 调度器**不会仅因 C 更干净就产出 MLP**。这与 [ISSUE-100] nvfp4 脾气墙（`jalr` call-clobber = clang reg-alloc 行为·非我方决策空间·[K-4] 脾气墙）、[ISSUE-107] grid dequant（beat 调度良好 clang-18 需手写发射器欠缺的调度·compiler-maturity gap）**同族**。

**⟹ 精确定性（两层）**：
1. **顶层 = 公式墙**：决策空间（load 宽度 + 寄存器驻留多流循环重构）可键控·闭式·未算尽 → **必攻**（本 scope 的 B 线裁定）。
2. **子墙 = 脾气墙-风险**：攻坚构造后板测·若 objdump 证「我方 emit 的宽多流 load 被 clang **重新串行化**（load 未 hoist·MLP 未达·cold 仍 inert）」→ **残余 = clang-调度脾气墙**（[K-4]·[ISSUE-100]/[ISSUE-107] 族）→ **此时才**满足三步（objdump 对手✓ → 定杠杆 load-scheduling✓ → 板测证伪✓）→ **ISSUE-109 第 5 次定格·登记 memory-scheduling 脾气墙边界**（见 `05-boundary-registration-if-temper.md`）。

**这不是骑墙**：canon 要求「公式墙必攻」，而攻坚的**验收门**恰是「clang 是否保住 MLP 调度」。攻之前它是未算尽的公式墙（禁登记边界）；攻之后若 clang 证再串行化·它才降为脾气墙（合法登记）。B 线要的正是这个「先攻·攻后据 clang 行为定 5 次定格」的裁定。
