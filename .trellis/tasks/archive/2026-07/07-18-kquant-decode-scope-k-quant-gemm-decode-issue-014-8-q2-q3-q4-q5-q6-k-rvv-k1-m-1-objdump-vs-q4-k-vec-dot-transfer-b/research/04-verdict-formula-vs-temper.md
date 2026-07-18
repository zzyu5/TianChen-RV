# Research: ★判据裁定 — decode 8 格 = 公式墙 vs 脾气墙 (B 线核心·分组·带证据)

- **Query**: PRD §二.4 — 逐格/分组裁定：公式墙〔可键控·必攻·施工入口〕 vs 脾气墙〔micro·走三步登记边界〕
- **Scope**: internal (裁定·带 objdump 证据·非赢没赢)
- **Date**: 2026-07-18

## 判据 (canon 正本·`.trellis/spec/canon/覆盖状态机与选择归因.md` [K-4] 三态·:17-21)

- **公式墙** = 「可键控 / 闭式导出的决策空间未算尽」→ **必须攻**。
- **脾气墙** = 「clang codegen / 硬件微架构脾气·非我方决策空间」→ 走三步方登记边界、默认接着找。
- **边界登记合法性** = 「仅当杠杆清单真空 ∧ 走过三步 (objdump 对手 → 定杠杆 → 板测证伪留档)」。

---

## ★净裁定：8 格全部含一个未算尽的公式墙 ⟹ 现在登记边界 = 违 [K-4] 三步

每个 rvv 格是**两墙叠加**；k1 格是**一墙 + VLEN256 co-factor**。逐层裁：

### 墙-1 (仅 rvv 5 格·deploy overlay) = gcc-death = 脾气墙 (已三步·可登记·但非核心)

- **性质**：rvv 出货 gcc-15.2·把 decode leaf 编成 vsetvl **922/2225/2952/2461** (q2/q3/q5/q6) + q4_K·regfile spill/scalarize；clang **81/14/57/22** = **10–160× 更多 vsetvl**。[CASE-KQUANT-GCC-CODEGEN]·K-ledger 机制②。
- **裁定 = 脾气墙**：这是 clang/gcc codegen 脾气·非我方算法决策空间。**三步已走完** (objdump 对手✓ · 杠杆 = gcc→clang✓ · 板测 gcc-vs-clang vsetvl 差实证✓) ⟹ **可合法登记为脾气墙边界** (机制②·[CASE-KQUANT-GCC-CODEGEN])。
- **但非攻坚核心**：gcc→clang 恢复 ~5.5× (q4_K 0.066→0.361·q5_K 0.128→0.504) 但**仍 <0.8** ⟹ 修脾气墙不翻正。A2-b5 §4「修 gcc-death 不翻正·不立项」结论**成立** (但其理由「关键路径=fold 摊销」已被 ISSUE-014 证 fold 非物理·真理由 = 墙-2)。

### 墙-2 (全 8 格·真成本中心·ISSUE-014 留白) = 公式墙 (候选·必攻·未算尽)

- **性质** (objdump 结构·`02` §二)：编译器对称 (ours-clang vs opp) 下仍输 2–5×·残留成本中心 = **memory-scheduling 轴** = 窄 e8mf2 (8B) load + clang 低 MLP over weight-DRAM-stream (M=1 无行摊·intensity=1)·对手宽 e8m1 (16B) + 手排多流 MLP。
- **可键控闭式决策空间 (未算尽)**：
  - (i) **load 宽度** e8mf2 → e8m1/m2 (LMUL 参数·纯 codegen 决策)。
  - (ii) **多流 load 提前发射** (循环展开度 / load-stream 解耦·镜像对手 16 独立宽 load)。
  - (iii) **VLEN256 宽化** hl8 → hl16 (仅 k1·[ISSUE-019] 机制①·iq3 已证 4 格翻正)。
  - 三者皆 [K-4] 定义的「可键控 / 闭式导出决策空间」= **公式墙**。
- **裁定 = 公式墙·必攻**：这三个杠杆**从未在 decode leaf 上试过**·[K-4] 三步的「板测证伪」未做 ⟹ **杠杆清单非空** ⟹ **现在登记边界违 [K-4]**。
- **★确定性 caveat (决定攻坚第一步)**：decode leaf **无 perf-stat** (无 IPC/backend-idle)·「memory-scheduling 是主成本中心」是 **objdump 结构推断非实测** (`03` transfer)。⟹ **公式墙攻坚第一步 = perf-stat decode leaf 证 memory-bound** (填 ISSUE-014 留白·本 objdump task 做不到)。若 perf-stat 证反是 unroll/frontend-bound → 杠杆改为「降展开/icache」·仍公式墙 (可键控) 但换轴。

---

## 逐格/分组裁定表 (成色 + 墙性质)

| 格 | 对手成色 | 墙-1 (deploy) | 墙-2 (真成本中心) | 裁定 |
|---|---|---|---|---|
| q2_K@rvv | 真手调 vl128 STRONG | gcc-death 脾气墙 | memory-sched 公式墙 (16-sub-block 重) | **公式墙·必攻** |
| q3_K@rvv | 真手调 vl128 STRONG | gcc-death 脾气墙 | memory-sched 公式墙 (hmask/16-sub) | **公式墙·必攻** |
| q4_K@rvv | 真手调 vl128 STRONG | gcc-death 脾气墙 | memory-sched 公式墙 (8-sub·2 板测据点最多) | **公式墙·必攻** |
| q5_K@rvv | **中/弱 native-vec (no vl-spec)·成色降披露** | gcc-death 脾气墙 (vsetvl 2952 最重) | memory-sched 公式墙 + qh-reg-cliff | **公式墙·必攻·成色标弱 opp** |
| q6_K@rvv | 真手调 vl128 STRONG | gcc-death 脾气墙 | memory-sched 公式墙 (6-bit/16-sub 最重) | **公式墙·必攻** |
| q3_K@k1 | 真手调 vl256 STRONG (无 repack·身份干净) | 无 (clang) | 公式墙 = VLEN256 宽化① + memory-sched | **公式墙·必攻** |
| q5_K@k1 | **对手身份存疑** (`01` §四) | 无 | 公式墙 (同上) + qh-reg-cliff | **公式墙·必攻·先核对手身份** |
| q6_K@k1 | **对手身份存疑** (`01` §四) | 无 | 公式墙 (同上·16-sub) | **公式墙·必攻·先核对手身份** |

## 关键成色分离 (禁称硬赢·非攻坚对象)

- **真强手调对手 (真攻坚价值)** = q2/q3/q4/q6_K@rvv (vl128 手调) + q3_K@k1 (vl256 手调·身份干净)。
- **中/弱对手 (成色降披露·翻了也非硬赢)** = q5_K@rvv+k1 (native-vec no-vl-spec·对手自己也慢)。
- **对手身份待核** = q5_K/q6_K@k1 (A2-b5 用 block-dot·真部署可能是 gemv repack·须部署身份探针·ISSUE-004)。
- **不在 8 内 (已出队)** = q2_K@k1 0.9585 near-parity + q4_K@k1 1.535 what-if (弱 opp·非 win)。

## ⟹ 定 8 格继续攻 (公式墙)·非登记边界

**8 格全部 = 公式墙 (墙-2 memory-scheduling 未算尽)·必攻**·**禁现在登记 honest-null/架构不可达** (违 [K-4] 三步·杠杆清单非空)。rvv 5 格叠加的 gcc-death **脾气墙**可独立登记 (机制②·三步已走)·但那是 deploy overlay 不是攻坚核心。施工入口见 `05`。

## 出处
- [K-4]：`覆盖状态机与选择归因.md:17-21`
- gcc-death：`A2-batch5-kquant-decode-M1.md:30,111`·K-ledger 机制② (`K-attack-fanout-ledger.md:14`)
- 真成本中心结构：`02-our-decode-cost-center.md`·transfer：`03-transfer-analysis.md`
- ISSUE-014 留白：`性能与测量.md:114-120`
