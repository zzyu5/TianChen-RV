# PRD · K线 · 手调档 K-quant vec_dot 攻坚首格 q4_K（完整攻坚环）

> **权威** = 《开测篇》§二.3-4（手调档全数进环冲 0.8·完整攻坚环→PASS 或经环具名上报）。
> **性质** = 手调档攻坚（best-effort 冲 0.8）。用 B 建的 `vec_dot.sh` harness。台账机制②③。

## 一、目标与诚实预期

q4_K vec_dot @rvv baseline（B 测）= **具名-X 0.166**（vs 手调 `_vl128` STRONG·weight-reconstruction floor）。
走**完整攻坚环**（解剖→构造→前门→板测→cold）冲 0.8：
- 机制②「gcc→clang codegen 杠杆」（台账：gcc→clang 恢复 q4_K 0.066→0.361·~5.5×·但 clang 残差仍 <0.8）。
- 机制③「[MECH-WEIGHT-RECONSTRUCTION-BOUND] super-block 位重建原语 (SEW,LMUL,VLEN) 参数化」（真成本中心）。
- **诚实预期**（台账）：叠加机制②③仍可能 <0.8（weight-reconstruction floor vs 手调 vl128·q4_K repack 即使 prefill 也 0.94×parity·不泛化 rvv）。⟹ **可能经环具名+墙**（同 nvfp4 范式·带 objdump 逐指令墙证据）。

## 二、方法（完整攻坚环·禁跳环）

1. **解剖**：objdump q4_K vec_dot ours leaf vs 手调 `_vl128`（数 vset/spill/weight-reconstruction 指令·锁定成本中心）。
2. **构造**：机制③ weight-reconstruction 原语参数化（super-block 6-bit scale utmp 解包 + min 项·(SEW,LMUL,VLEN) 参数化）·+ 机制② clang codegen（已 clang-18 世界）。
3. **前门**：byte-exact vs oracle（[K-5] ZERO-MODEL·B 的 oracle 已建）。
4. **板测**：`vec_dot.sh rvv verify/measure q4_K`·cold 2-seed vs baseline 0.166。
5. **cold**：≥0.8 → PASS（手调档真收口·罕见）·<0.8 → **经环具名 + 逐指令墙**（三档墙分类：可攻坚旋钮/我方内禀墙/对手结构优势）。

## 三、验收

1. **完整攻坚环走完**（objdump 解剖 + 构造 + 前门 byte-exact + 板测 cold·非跳环）。
2. **翻正格**：cold≥0.8·recon-dict 入账·手调档 PASS（成色：真赢手调 = 硬赢·可称）。
3. **翻不正格**：cold<0.8·**经环具名上报**（objdump 逐指令墙 + 三档分类·同 ISSUE-100 nvfp4 范式·带证据）·**非跳环认输·非硬凑**。
4. **成色如实**：对手=手调 `_vl128` STRONG（非便宜档）·真赢=硬赢·输=具名墙。
5. **0 造数**·byte-exact 硬门·sealed 不动·master 不直写·**禁 commit·禁 add -A**·"某物不存在"禁截断。

## 四、触碰集 / 铁律

- 触碰：`RVVToEmitCKQuant.cpp`（q4_K vec_dot weight-reconstruction·独占）+ 前门 lit + `cells/vec_dot.sh`（已建·B）+ bench 跑。
- **独占 K-quant emit TU**（与并行线 block-quant dequant = `RVVToEmitCForwardElementwise.cpp` 不相交·勿碰 GridCodebook.cpp/dequant emit）。
- **re-roll trap 铁律**（三次证伪）：每步先反汇编·证 spill/weight-reconstruction 指令真降·未降=无效=停该步具名·禁硬凑。
- 用 rvv 板。[[build-incremental-unreliable]] clean rebuild+亲见。

## 五、遗留

- q4_K 证环后扇 q6_K/q2_K/q3_K/q5_K vec_dot（各须补 oracle·B 标 q2/q3/q5 oracle 待补）。
- @k1 部署 gated ISSUE-105。若 q4_K 经环具名 = 手调档 weight-reconstruction floor 具名墙（诚实·台账机制③负结果）。
