# PRD · K线 · q6_K vec_dot 攻坚（走环 · weight-reconstruction floor 泛化 2nd 数据点）

> **权威** = 《开测篇》§二.3-4（手调档全数进环冲 0.8·完整攻坚环→PASS 或经环具名）。
> **性质** = 手调档攻坚（走环 2nd K-quant vec_dot 数据点）。用 `vec_dot.sh` harness（q6_K oracle 已建·B）。承 q4_K 环范式（归档 `07-18-k-kquant-vecdot-attack`·ISSUE-109）。

## 一、目标与诚实预期

q6_K vec_dot @rvv baseline（B 测）= **具名-X 0.202**（vs 手调 `_vl128` STRONG）。走**完整攻坚环**冲 0.8。
- **诚实预期**（q4_K 环已证·ISSUE-109）：q6_K 同 weight-reconstruction floor（scratch store→load roundtrip + serial min-term·stall-bound）·旋钮 `integer_core_lmul` 预期同 EXHAUSTED ⟹ **likely 经环具名 + 同三档墙**。
- **本 task 价值** = **2nd 板测数据点**（确认 [MECH-WEIGHT-RECONSTRUCTION-BOUND] floor 泛化·强化 mechanism-level 具名上报·非外推）。若意外翻正则真收口（罕见）。

## 二、方法（完整攻坚环·同 q4_K·禁跳环）

1. **解剖**：objdump q6_K vec_dot ours leaf vs 手调 `_vl128`（数 vset/spill/scratch roundtrip/serial MAC·锁成本中心·对比 q4_K 的 aux8[256] 墙）。q6_K = 6-bit quant + 8-bit scale·−32 offset·2-bit qh 拼接。
2. **构造**：机制③ `integer_core_lmul` 旋钮扫（mf2/m1/m2·byte-exact fold-back·objdump vset 真降）+ 机制② clang-18 世界。
3. **前门 byte-exact**（[K-5]·B 的 q6_K oracle 已建）。
4. **板测 cold**（`vec_dot.sh rvv verify/measure q6_K`·2-seed vs baseline 0.202）+ **perf stat**（IPC/backend-idle/cache-miss·验 stall-bound 假设·同 q4_K IPC 0.08）。

## 三、验收

1. **完整攻坚环走完**（objdump 解剖 + 旋钮扫 byte-exact + 板测 cold + perf stat·非跳环）。
2. **翻正**：cold≥0.8 → PASS（罕见·真赢手调=硬赢）。
3. **翻不正**（预期）：cold<0.8 → **经环具名 + 三档墙**（objdump + perf·确认同 q4_K weight-reconstruction floor·ISSUE-109·mechanism-level 2nd 实例）。台账机制③行加 q6_K 数据点。
4. **成色如实**（手调档 STRONG 对手）·**re-roll trap 铁律**（每步反汇编证 vset/spill 真降·未降=无效=停具名）。
5. **0 造数**·byte-exact 硬门·sealed 不动·master 不直写·**禁 commit·禁 add -A**·"某物不存在"禁截断。

## 四、触碰集 / 遗留

- 触碰：`RVVToEmitCKQuant.cpp`（q6_K vec_dot·独占此 TU·与并行线 block-quant dequant emit `RVVToEmitCForwardElementwise.cpp` 不相交·勿碰 GridCodebook.cpp）+ 前门 lit + `vec_dot.sh`（已建）+ bench 跑。用 rvv 板。clean rebuild+亲见。
- **遗留**：q6_K 环后·q2_K/q3_K/q5_K vec_dot（oracle 待补）—— 若 q4_K+q6_K 两点均同 floor·主会话可**mechanism-level 具名上报** q2/q3/q5 群（ISSUE-109·weight-reconstruction floor 结构泛化·同 grid dequant 家族范式）而非逐格补 oracle 走环（待主会话裁）。@k1 gated ISSUE-105。
