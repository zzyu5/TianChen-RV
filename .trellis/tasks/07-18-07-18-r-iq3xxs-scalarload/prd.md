# PRD · R线 · 裁决3 iq3_xxs 候选② owned 标量-load 查表 dequant emit（完成 grid dequant 攻坚环）

> **权威** = 用户裁决 3（2026-07-18）+ 补充令二 §五.4（ISSUE-107 候选②·从未试）。
> **性质** = 发射器攻坚（owned 标量-load 结构·完成 grid dequant 攻坚环）。**判别键 = 谁决定·非标量还是向量。**

## 一、目标与关键澄清

iq3_xxs@rvv dequant owned **真向量(HW-gather)** = 0.36 gather 天花板（ISSUE-107）。但**候选②从未试**：
**对手那个形状** —— objdump 对手 `dequantize_row_iq3_xxs` `HW_GATHER=0`·用 **37 条标量 load + unit-stride 向量算术**·结构性快 2.8×。

**★澄清（裁决 3）**：改造理由**不是"必须用向量"**。是 —— certified 101/108 里 24 行 dequant·若向量内容是宿主 clang 送的·那 24 行不算「由我们机制构造」·101/108 虚（ISSUE-002）。**判别键 = "谁决定"·候选②（我们发标量-load 查表结构）满足 de-lottery [L-8]·且不撞 gather 墙·且它是标量的。**

**没试②就对 grid 族报「具名-X / 架构不可达」= 攻坚环没走完**（[缺口与认输 §2]：造了等价能力仍差在具名的 X 方可具名-X）。

## 二、方法

1. **objdump 对手** `dequantize_row_iq3_xxs`：确认 HW_GATHER=0·标量 grid-index load + unit-stride 向量算术的结构（37 标量 load）。
2. **author owned 标量-load 查表 dequant emit**（iq3_xxs·**复刻对手结构**：标量 load 查 grid 码本 + unit-stride 向量算术·**非 HW vluxei gather**）。这是 owned emit（我们发射·确定性·de-lottery [L-8]）·非宿主抽签。
3. **前门 byte-exact**（[K-5] ZERO-MODEL·oracle 已建·dequant decode-only）。
4. **板测 cold**（`dequantize_row.sh rvv verify/measure iq3_xxs` 的候选②变体·2-seed）。

## 三、★预期天花板 ~parity —— 这是预测不是实测（先测不预告）

[§五.15]：一切选择键值 per-format 板测定·直觉投影已被独立证伪三次。**先测·不预告。**
- verdict：cold≥0.8 → **grid dequant 真收口 via owned 标量-load**（de-lottery·[L-8]·不撞 gather 墙·关 ISSUE-002 该格敞口·解 ISSUE-107 fork）。
- cold<0.8 → **具名 + 真墙（环走完·候选②已试）**·带 objdump 逐指令墙。

## 四、验收

1. **owned 标量-load emit**（objdump 证标量 load 结构·非 HW gather·非宿主 autovec·OWNED 确定性）。
2. **byte-exact GREEN**（[K-5]·harness 实证）。q8_0 范式 · 但注意 grid 码本查表（非 block）。
3. **cold 2-seed**·verdict（先测·≥0.8 收口 / <0.8 真墙）。**报数带 ULP 界（补充令二 §二·若浮点尾巴有税）。**
4. **成色**：对手=部署标量-autovec·de-lottery·[L-8] construction·opp-immaturity（§三.12·非硬赢）。
5. **0 造数**·byte-exact 硬门·sealed 不动·master 不直写（recon-dict 留 main）·**禁 commit·禁 add -A**·objdump 全量。

## 五、触碰集 / 遗留

- 触碰：iq3_xxs dequant emit（**新增标量-load body 或改道·独占 dequant emit 路径**·与并行线 FMA-probe/D-2a-research 不相交·勿碰 KQuant/GridCodebook 的 vec_dot/gemm 路径）+ 前门 test + `dequantize_row.sh` + bench 跑。用 rvv 板。
- **遗留**：候选②证后·grid 族（iq3_s/iq2*）同结构可扩（若②work）。三选一（ISSUE-107）由本 task 结果定：②work → grid 走候选② de-lottery；②不 work → 具名真墙。
