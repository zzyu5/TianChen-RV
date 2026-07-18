# Research: 边界登记建议（脾气墙 fallback）— 仅当攻坚后 clang 证再串行化时才用

- **Query**: 若脾气墙·三步走完的边界登记建议（[K-4] 边界·ISSUE-109 五次定格·honest-null / 具名-X 带脾气墙标注）
- **Scope**: internal（fallback 登记模板·**非现在登记**）
- **Date**: 2026-07-18

## ★前置条件（严·未满足前禁登记）

本 scope 的裁定 = **公式墙·必攻**（`03-verdict`）。**现在登记边界 = 违反 [K-4] 三步规则**（memory-scheduling 杠杆清单非空·板测证伪未做）。本文件只是**攻坚后**的 fallback 模板——**仅当** `04-construction-entry` 的构造板测后满足**全部**下列条件才启用：

1. objdump 对手✓（本 scope 已做·`01-opponent-mlp-mechanism.md`）。
2. 定杠杆 = memory-scheduling（宽 load + 寄存器驻留多流 MLP）✓（本 scope 已定）。
3. **板测证伪**：owned 宽多流 emit 构造 + 板测·且 objdump 证 **clang 把宽多流 load 重新串行化**（load 未 hoist·outstanding load 未增·MLP 未达·cold 仍 <0.8 且 inert）——**此步未做·是启用本文件的唯一闸**。

## 若三步走完（clang 证再串行化）→ ISSUE-109 第 5 次定格建议

### 登记形态 = 具名-X + 脾气墙标注（**非** honest-null·**非**架构不可达）

- **verdict**：`vec_dot q4_K@rvv` 维持**具名-X best m1 0.186**·墙升级为 **memory-scheduling clang-调度脾气墙**（[K-4] 边界·[ISSUE-100]/[ISSUE-107] compiler-maturity 族）。
- **禁措辞**：禁「架构不可达」（对手同硬件达到·格式 rvv 无结构障碍·[ISSUE-100] premature 订正的教训）。禁「honest-null」（那是**结构不可达**如 [ISSUE-035] strip-width singleton·此处**结构可达·卡在 clang 调度**·性质不同）。
- **准病名建议**：`[MECH-WEIGHT-RECONSTRUCTION-BOUND]` floor 的**真墙 = 低 MLP·DRAM-latency 暴露**·而**唯一未试 lever（memory-scheduling）撞 clang-调度脾气墙**（emit 宽多流·clang 再串行化）——**gated on 发射器/编译器指令调度成熟度**（要 beat 对手 inline-asm 手排调度·需我方发射器欠缺的调度控制·**项目级 compiler-maturity 缺口·非 q4_K 专属**）。

### 五次定格的诚实叙事（承四次）

| 定格 | lever | 板测 | 墙诊断 |
|---|---|---|---|
| 1 | register-fusion | cold 0.152 inert·cache-miss 33×↓ | 曾诊 latency/dependency |
| 2 | vwredsum.vs | cold 0.162 inert | 真墙订正 = scalar-heavy |
| 3 | minterm-vec | cold 0.164 inert | 真墙订正 = 整核指令流 |
| 4（现） | 三 lever 综判 | 三核 cold ~1.32M ns 恒 | **memory-stall floor·指令流轴 EXHAUSTED·剩 memory-scheduling(d)** |
| **5（fallback）** | **memory-scheduling 宽多流** | **若 cold inert + objdump 证 clang 再串行化** | **真墙 = 低 MLP·未试 lever 撞 clang-调度脾气墙·compiler-maturity gated** |

### 保守默认（登记后）
- master census `vec_dot q4_K@rvv` 维持具名-X 0.186·不翻 PASS。
- 具名-X 保留（**非软认输**：全环走完·objdump 对手 + owned 构造 + 板测 + clang 再串行化证据齐·补充令二红线守住）。
- **剩余候选（清单**仍**可非空）**：若攻坚证 clang 是唯一障碍·则「emit inline-asm 手排 load 调度（如对手）」是理论残余 lever·但**与我方 C-intrinsic emit 范式冲突**（同 [ISSUE-107] 标量-load lever = maturity-gated·战略裁是否值得投）→ 归 ISSUE-109 / compiler-maturity·**战略裁·agent 不自决**。
- 扇出：q6_K/q2_K/q3_K/q5_K vec_dot 同 floor·likely 同脾气墙·**mechanism-level 具名·禁逐格外推计数**（承机制③ ledger「2 板测数据点·structural-inference」纪律）。

## 与现有 issue 的挂靠

- **ISSUE-109**：本裁定（公式墙·必攻）+ 5 次定格 fallback → 更新 ISSUE-109 状态（现「待施工·剩 lever (d)」→ 攻坚工单挂 [K-4]「攻坚中」态）。
- **ISSUE-112 census**：`vec_dot q4_K@rvv` 现为 live-lever（vwredsum 曾记未试）——本 scope 把 lever 精化为 memory-scheduling·仍**清单非空**·移不出软认输面须待攻坚（非本 scope）。
