# [G3 主线A / T1] q4_K repack GEMM 输出-tiling 方案评审包

**类型**: 纯设计侦察 / 一页评审包。★零 code 改、未 commit、未动 `lib/`。
**日期**: 2026-07-08。**板**: rvv / VLEN128（能力事实：`vlen=128`, `vreg_count=32`）。
**读入**: `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:5367` `emitRepackGemmQ4KQ8K` +
`experiments/active/l1-pipeline-q4k-repack-gemm`（曳光弹 #1，byte-exact reorder）+
`experiments/active/l1-reroll-q4k-repack-gemm`（曳光弹 #2，naive re-roll）。
**选型是用户裁决点** —— 本包给排名 + 一个推荐 + 依据，不自选落地。

---

## 0. 板证钉死的问题（两便宜偏方已证伪）

`emitRepackGemmQ4KQ8K` 把整超块（`nSuperHalves`2 × `subPerSuper/2`2 pair × k2 × ii16 ×
`activationInterleave`4 col × `numHalves`2 half）全展开进**一个 basic block**。持久累加器
fan-out = **[col c]×[half h] = 4×2 = 8** 个向量 lane，每 lane 养 3 个累加器族
（`sumfVar` f32m2 / `sumiVar` i32m2 / `bsumsVar` i32m2），全部**同时活跃** →
持久累加器 = 8×(2+2+2) = **48 vreg**，加解码带宽 → 峰值活跃 **~92 vreg ≫ 32** →
普遍 spill → 0.2 MAC/cycle。

两个便宜杠杆已在硅上证伪（不是外推）:
- **曳光弹 #1（byte-exact reorder）**: vsetvli 52→52、spill 未动、A/B **+0.65%**（噪声地板）、
  vs-opponent parity 0.962x→0.970x（仍 <1.0）。→ 字节等价重排搬不动峰值 liveness（结构量）。
- **曳光弹 #2（naive re-roll，k/ii→emitc.for）**: vsetvli **53→76 (+43%)**、spill **84→118 (+40%)**、
  A/B **−11%**、parity 0.962x→**0.860x**（更深的 loss）。两个失败机理钉死了真解法约束:
  1. **vsetvli 非 loop-invariant** —— 子块体真在 SEW 间交替（e8/mf2 nibble decode ↔ e16/m1 i16 accumulate），
     rolling 只会**倍增** vtype toggle，不会 hoist。
  2. **`emitc.for` 无 iter-args** —— loop-carried 累加器落 C locals（内存）每迭代 spill/reload；
     全展开的 PRE 反而把这些链留在 SSA 寄存器里。

→ 真解法**只能**是经典 GEMM 输出瓦片化: 缩 fan-out 到装得下 32 vreg，且尽量保住
once-per-16-weight 解码摊销。约束二（iter-args 缺）意味着落地必须让**累加器留在寄存器**
（straight-line per-tile），把要外溢的东西换成**冷的**已解码权重，而非**热的**累加器。

---

## 1. 两条 tiling 轴 + 静态账公式（先立尺子）

输出瓦片 = **mr 行（激活列 M）× nr 列（权重列 N）**。本 kernel 的向量 lane 本身**就是 N 的一部分**:
一个 f32m2 累加器 = `half`=8 个权重列 × 1 个激活列。所以:
- **mr** = 瓦片内激活列数（现 `columnsPerPass`=4；此旋钮**已存在**，现由 `coreLmul` 键控 mf2→4/m1→1）。
- **hs** = 瓦片内 h-strip 数（现 `numHalves`=2 @VLEN128；每 strip = 8 权重列）。nr = hs × 8。

**@VLEN128 mf2 链的每值物理 vreg**（`half`=8 lane 贯穿）:

| 值 | 类型 | LMUL | 物理 vreg |
|---|---|---|---:|
| `sumfVar`（主 f32 累加器） | f32m2 | m2 | **2** |
| `sumiVar`（scale 主项 i32） | i32m2 | m2 | **2** |
| `bsumsVar`（min 项 i32） | i32m2 | m2 | **2** |
| `sLo`/`sHi`（i16 partial） | i16m1 | m1 | **1** |
| `scaleVal`/`minVal`（6-bit 解出的 i16 带） | i16m1 | m1 | **1** |
| `nLo`/`nHi`（nibble decode） | i8mf2 | mf2 | **1** |
| `dF32`/`dminF32`（f16 scale widen 后） | f32m2 | m2 | **2** |

**峰值活跃账公式**（最热点 = k-chunk 内 ii 循环，sumf 携带 + sumi/bsums 携带 + scale/min 活 +
sLo/sHi 活 + nibble 瞬态 + d/dmin 持有 + ~2 地址/cvt scratch）:

```
peak(mr, hs) = 8·mr·hs   [持久累加器 6·mr·hs + sLo/sHi 2·mr·hs]
             + 14·hs     [scale/min 8·hs + nibble 2·hs + d/dmin 4·hs]
             + 2         [scratch]
```

**关键洞（决定全局）**: `14·hs` 是 per-strip 项。**hs 乘整个账**（含解码带），**mr 只乘 8·mr·hs 那半**。
→ **先切 h-strip（hs→1）是最省寄存器的杠杆**；而且 h-strip 之间权重**不相交**（`loadU8Strip` 的 `h*half`
偏移、scale/min 的 `h*half` loByte/hiByte 各读各的字节）→ **切 hs 零摊销损失**。切 mr 才动摊销（见 §3）。

---

## 2. 枚举 6 方案 + 逐项 vreg 账（★证 ≤32 否）

记号: **S0** = 现状基线。逐项账在最热点 ii 循环处结算。

### S0 — 基线（现状，已证伪）: mr=4, hs=2, nr=16, 3 族全展开
| 项 | vreg each | count | 小计 |
|---|---:|---:|---:|
| sumf/sumi/bsums | 2 | 4·2·3=24 | 48 |
| sLo/sHi | 1 | 4·2·2=16 | 16 |
| scaleVal/minVal | 1 | 4·2·2=16 | 16 |
| nLo/nHi | 1 | 2·2=4 | 4 |
| dF32/dminF32 | 2 | 2·2=4 | 8 |
| scratch | — | — | 2 |
| **峰值** | | | **94** |

**≤32? ✗（94，~3×预算）**。spill 灾难（-O2 measured 84 spill / 88 reload；-O3 143/158）。0.2 MAC/cyc。

### S1 — 仅 h-strip tile: mr=4, hs=1, nr=8（外加一层 h 循环包住整个 contraction）
`peak = 8·4·1 + 14·1 + 2 = 48`。**≤32? ✗（48）**，但 **94→48 一刀砍掉一半**。
摊销 **FULL**（strip 不相交，解码不重算）。**最低风险**（纯循环重排，可做成 byte-exact-able），
是 S5/S6 的底座。spill: 大幅减但非零（累加器 24 装得下，解码带把它顶过 32）。

### S2 — 仅 column-pair: mr=2, hs=2, nr=16
`peak = 8·2·2 + 14·2 + 2 = 62`。**≤32? ✗（62）**。摊销 **减半（decode×2）**。
→ **被 S1 严格支配**（寄存器更差 62>48 且摊销更差）。**弃**。

### S3 — h-strip + column-pair: mr=2, hs=1, nr=8
`peak = 8·2·1 + 14·1 + 2 = 32`。**≤32? ⚠ 恰好 32（临界）**。含 v0 mask + 真实地址/cvt scratch
（实测 scratch 常 3–5 而非 2）→ 实际 ~34，**边缘、留 1–3 残余 spill**。摊销 **减半（decode×2）**。
干净但让出一半 GEMM 优势。

### S4 — h-strip + 单列: mr=1, hs=1, nr=8
`peak = 8·1·1 + 14·1 + 2 = 24`。**≤32? ✓（24，宽裕）**。但摊销 **decode×4 = 退化成 4 个独立 GEVM**，
**GEMM 优势全数交回**。寄存器 headroom 被浪费。**弃（除非只想验证 spill→0 的下界）**。

### S5 — h-strip + 累加器重构（min 项 inline-fold，去掉 bsums 族）: mr=4, hs=1, nr=8
去 bsums 族: 持久累加器 3→2 族。`peak = 6·mr·hs + 14·hs + 2 = 6·4 + 14 + 2 = 40`。
**≤32? ✗（40）** —— **单去一族不够**（只省 4 col×2 = 8 vreg，48→40）。诚实结论: **mr=4 单靠去族达不到 ≤32**。
须叠加 §S6 的解码外置。摊销 **FULL**。代价: min 修正改成 per-sub-block 的 `vfnmsac`（+~32 f32 op/block，非解码）。

### S6 — ★h-strip + inline-min-fold + 已解码权重 stack-panel: mr=4, hs=1, nr=8
把「外溢冷的、留住热的」做实: 每 block 把一条 strip 的 nibble/scale/min **解码一次写进小栈数组**
（`int8_t nib[...]` local），列循环从栈读；d/dmin 改成**用时才 load**（本就 per-block loop-invariant，
load 点下移，仍只 load 一次）。再叠 S5 的 inline-min-fold 去 bsums 族。
| 项 | vreg each | count | 小计 |
|---|---:|---:|---:|
| sumf/sumi（2 族） | 2 | 4·1·2=8 | 16 |
| sLo/sHi | 1 | 4·1·2=8 | 8 |
| 解码 reload scratch（栈→reg，1 strip 当前项） | 1 | ~3 | 3 |
| cvt / 地址 scratch | — | ~2 | 2 |
| **峰值** | | | **~29** |
（d/dmin 用时 load，不常驻。）**≤32? ✓（~29，唯一同时 spill→0 且摊销 FULL 的方案）**。
代价: 显式栈流量（已解码权重，**L1 冷、访问少**）+ inline-fold 的 +f32 op。
—— 这正是经典 GEMM「pack panel into contiguous buffer」搬到已解码权重上；把现在**热累加器**的被动 spill
换成**冷解码权重**的主动 L1 staging。

---

## 3. 三轴权衡表 （fan-out mr×nr）×（累加器驻留）×（摊销率）

| 方案 | mr×nr (M×N) | 峰值 vreg | ≤32? | 解码 reuse | decode×base | spill delta（半定量） | 落地风险 |
|---|---|---:|:--:|:--:|:--:|---|---|
| **S0** 基线 | 4×16 | 94 | ✗ | 4× | 1.0 | 灾难（84/88 @-O2；2188 类往返） | —（现状） |
| **S1** hs=1 | 4×8 | 48 | ✗ | 4× | 1.0 | **~半**（累加器24装下，解码带顶过32） | **低**（纯循环重排） |
| S2 col-pair | 2×16 | 62 | ✗ | 2× | 2.0 | 重（被 S1 支配） | 低 |
| **S3** hs=1+col-pair | 2×8 | 32(临界) | ⚠ | 2× | 2.0 | **~0（1–3 残余）** | **低**（复用现有 columnsPerPass=2） |
| S4 hs=1+单列 | 1×8 | 24 | ✓ | 1× | 4.0 | **0**（但退化 GEVM） | 低 |
| S5 hs=1+去族 | 4×8 | 40 | ✗ | 4× | 1.0 | 减但非零（单去族不足） | 中 |
| **S6** ★hs=1+inline+stack-panel | 4×8 | ~29 | ✓ | 4× | 1.0 | **→0，且摊销 FULL** | 高（栈 panel + inline-fold） |

**(b) 摊销损失账（decode/output-element）**: 重解码 = per-16-weight nibble decode + 6-bit scale/min unpack，
在 mr 个激活列间复用。每 super-block 每 16-权重组: nibble decode ~128 次、scale/min unpack ~16 次，
**共享给 mr 列**。→ nibble decode/输出元素 = 128/(16·mr): **mr=4 → 2.0**、**mr=2 → 4.0（×2）**、
**mr=1 → 8.0（×4，= GEVM 无摊销）**。切 hs **不改** reuse（strip 不相交）。
一句话: **hs 是免费轴，mr 是付费轴**。板证已钉死「只 roll column 才缩 fan-out，但那正是 forfeit GEMM 对 GEVM 的
唯一优势」——本表把「forfeit 多少」量化成了 decode×base 列。

**(c) 预期 spill delta**: 峰值 ≤32 ⟺ spill→0（S4/S6/临界S3）；峰值 40–48（S1/S5）→ spill 显著减但残留；
峰值 62–94（S0/S2）→ 重 spill。硅上已验: 峰值是**结构量**，byte-exact 重排（#1）与 naive re-roll（#2）都搬不动它，
必须真正缩 fan-out（本表左二列）才动得了 spill。

---

## 4. 排名 + 推荐（★选型是用户裁决点，不自选）

**Pareto 前沿**（其余被支配）:
- **想要 FULL 摊销 + ≤32**: 唯 **S6**。诚实约束: **mr=4 装进 32 只能靠降 per-列累加器 vreg**
  （inline-fold 去一个 i32 族）**并**把解码外置到栈 —— 单独任一都不够（S5=40、S1=48）。
- **接受半摊销、要最简干净装下**: **S3**（mr=2,hs=1，临界 32，复用现有 `columnsPerPass` 旋钮）。
- **要最低风险的第一增量**: **S1**（纯 h 循环外提，摊销 FULL，94→48，是 S5/S6 的底座）。
- **S2 弃**（被 S1 支配）、**S4 弃**（退化 GEVM，只作 spill→0 下界参照）。

**推荐（供裁决）**: **S1 → S6 的两步序**。
- **依据**: 距 C3′（能力键控优化模式库 + 实测 Win-B）最短的**可测**台阶。
  **S1 先行**——风险最低、单步收益最大（峰值一刀砍半 94→48）、摊销零损、且**必然是 S6 的前置底座**
  （S6 的 hs=1 外层就是 S1）。**S6 收官**——是**唯一**同时达成 spill→0 **且**保住 once-per-16-weight
  全摊销（GEMM 对 GEVM 的唯一优势）的方案，即真正有望 vs-opponent 过 1.0x 的那个。
- **S3 作为低风险 fallback**: 若 S6 的栈-panel + inline-fold 重构代价过高，S3 是干净 ≤32 的退路，
  但让出一半摊销（decode×2），vs-opponent 未必过线——**这个取舍留给用户**。
- ★**不自选落地**: S1/S6/S3 之间是工程投入 vs 摊销保全的裁决，属用户裁决点。

---

## 5. 增量 framing —— 瓦片参数由能力事实键控，住 L2 schedule 属性（说明怎么住，不实现）

瓦片旋钮**不硬编码**，由能力事实计算后**声明式住在 op 的 L2 schedule 属性**里:
- **能力事实输入**: `vlen=128` 定 `half`=8 / `numHalves`=2 / mf2 链（i32 累加器 = m2 = 2 vreg）；
  `vreg_count=32` 是硬预算。（VLEN256: `half`=16、`numHalves`=1，h 轴自动消失、hs 强制=1；但 mr=4 仍
  8·4=32 顶预算 → 同样需 S5/S6 重构 —— 证明选择器**真·能力参数化**，非常量。）
- **属性形态**（拟）: `tcrv.l2_schedule = {tile_mr, tile_hstrips, decode_staging, min_fold}`，挂在
  `GgmlRepackGemmQ4KQ8KOp`（或 `WithVLOp` scope）上。**缺省 = 现状**（mr=4,hs=2,unroll）→ 向后兼容。
- **谁填**: 一个能力键控选择 pass，读 (`vlen`,`vreg_count`,…) 事实，用 §1 的 `peak(mr,hs)` 公式选**最大的**
  仍 `≤ vreg_count` 的瓦片（这是 [SEL-1] 最宽-合规 的直接实例: 在能力允许下选最富摊销的 tile）。
- **谁读**: `emitRepackGemmQ4KQ8K` 读属性参数化循环嵌套。**天然贴合现有结构**:
  - `columnsPerPass`（`for (cLo...)` 列-pass 循环）**已存在**，现由 `coreLmul` 二值键控 → 提升为
    能力键控的 `tile_mr`（mr 就是这个旋钮的推广）。
  - `numHalves` 的 h 维**已是循环** → 把它从「同 basic block 内并列 2 strip」提升为「外层 tile 循环、
    hs=1 一次一 strip」，即 S1 的循环外提。
  - `decode_staging`（`"inline"`|`"stack"`）、`min_fold`（`"deferred"`|`"inline"`）为 S6 的两个模式旗。

—— 即瓦片化不是新机制，是把**已被能力键控的 `columnsPerPass`** 从「coreLmul 二值」推广成
「(vlen,vreg_count) 键控的连续 tile 选择」，再补一层 h-tile 外提 + 两个模式旗。这落在 C3′ 的
「能力键控优化模式库」正中，且 novelty = **选择/键控**（NOVELTY 非新数字），落地/实测归后续 task。

---

## 附: [NG-4] 纪律

本包纯设计/分析，**无 board 数字、无 beat 主张**。所有 spill/vsetvli/parity 数引自已存在的两个证伪
cell（#1 pipeline / #2 reroll，rvv/VLEN128 core8 measured 2026-07-08）。vreg 账为静态 LMUL 推算
（mf2 链 @VLEN128），非 objdump 实测 —— 落地后须以 forced/clean rebuild + objdump vsetvli/spill
BEFORE/AFTER 与 board A/B 复核（[PERF-1] 八门另计）。选型是用户裁决点。
