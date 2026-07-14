# [VLEN-ADAPT] M0 静态账 — VLEN256-native vl=16 emit 的寄存器账 + 半宽根因定位 + M1 可行性判定

**Scope**: 纯静态（objdump 类比 / emitter 源 / plugin 源 / lit gate / RVV 寄存器语义账）。**不动刀 / 不改代码 / 不上板 / 不 build tcrv-opt / 不 git。** 供 [VLEN-ADAPT] M1 曳光弹选型用。

**依据源**：
- K1-SEAL 终审 `docs/reports/2026-07-10-k1-seal-e2e-transduction.md`（emitted vl=8 在 VLEN256 输 hand-brick vl=16 0.750×）。
- M1a 选型 `experiments/archive/rvv-e2e/rvv-e2e-m1/token_tile_selection.md`（peak_hot(d)=6d+7 vs 32-vreg，S6 实测 C=7）。
- S6 tiling `experiments/active/l1-tile-s6-q4k-repack-gemm/tile_s6_findings.md`（VLEN128 numHalves=2、vl=8、spill 23→0、maxVreg v30）。
- Emitter `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:6747` `emitRepackKQuantGemmBodyQ4K`（只读账）。
- Strip-width plugin `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp`（`deriveRepackHalfLanes` / `deriveMinimumVLEN`）。
- Widening 语义 `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:30-43`（16-way interleave byte-identity 不变量）。
- ★决定性 lit gate `test/Conversion/RVV/rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir`（同 kernel、仅 -march 变、V128 vl=8 vs V256 vl=16，已测已锁）。

---

## TL;DR（三问三答）

1. **vl=16 寄存器账（越崖?）** → **不越崖。peak_hot(d=4) = 6d+7 = 31 vreg (v30)，与 VLEN128 逐字相同。** vl 8→16 是 **VLEN 填满 lane**，**不是** LMUL 变宽。SEW/LMUL 完全不变（i8mf2→i16m1→i32m2→f32m2），f32m2=2 vreg / i32m2=2 vreg / i16m1=1 vreg 的**寄存器数由 LMUL 决定、与 VLEN 无关**。numHalves 2→1（strip 数减半，且 strip 是结构 tile、不叠加 peak）。**S6 的 ≤32-vreg 崖在 vl=16 下 HOLDS，无需 e16m2/更大 LMUL、无需回 S1、无需新档。**
2. **半宽根因（emitter 没出满宽 vs S6 假设 vl=8）** → **都不是。** 发射器 `deriveRepackHalfLanes(256,16)=min(256/16,16)=16`（**没钳 vl=8**），已能出 native vl=16/numHalves=1，且 lit gate 已测、byte-exact。S6 也**没**硬编码 vl=8（`half` 是入参、`vl8=sizeLit(half)`）。真根因 = **部署/配置失配**：sealed 源 md5 90d454da 是在 rvv/VLEN128 上**以 VLEN128 config 发射**（half=8，numHalves=2），K1-SEAL **原样搬到 k1/VLEN256（无 re-emit）**——half=8 是编译期常量烤进 vsetvli AVL，上 VLEN256 只填 8/16 lane = 半宽。且更深一层：emit 期 `-march` 经 `deriveMinimumVLEN` 只取 **ISA 保证的最小 VLEN**，plain `rv64gcv`（含 k1 build march `rv64gcv_zfh_zvfh_...`，**无 zvl256b**）→ 派生 128 非 256。要出满宽须 emit 期 march 带 **`zvl256b`**（或 VLEN256 capability fact）。
3. **token-tile 交互** → **两轴正交**。vl=half（VLEN 填 lane）与 token-tile 深度 d=columnsPerPass（LMUL-**数**扇出累加器）**互不干涉**：peak_hot=6d+7 **与 vl 无关**。M1a 选的 loop-interchange（col-outer/row-outer，d=4 保留）**register-neutral 且在最外两层**，与 vl=16 **完全兼容并存**（peak 仍 31）。vl=16 **不**给 token-deepening 腾寄存器（d=8 在 VLEN128/256 一样越崖 55>32）。两轴同库、键控在**不同 fact**（vl←VLEN capability；loop-order/d←cache fact）。

**M1 可行性判定：可行（GREEN）。** vl=16 native ≤32 vreg、发射器路径已存在+已 lit-测+byte-exact。M1 = 纯 front-door capability-input 改动（emit 期喂 VLEN256 fact），非 kernel 重写、非回退。

---

## 账 1 — vl=16 寄存器 / 累加器账（逐 accumulator）

### 1.1 关键 RVV 语义（消歧「lane 宽 vs vreg 数」）

RVV 里 **LMUL 决定一个 vector register group 占几个物理 vreg**（m1=1, m2=2, m4=4；mf2=1），**与 VLEN 无关**；**VLEN 只决定每个 vreg 装几个 element**（lane 数 = VLEN·LMUL/SEW）。所以：

| 类型 | vreg 数（LMUL 定，VLEN 无关） | lane 数 @VLEN128 | lane 数 @VLEN256 |
|---|---|---|---|
| f32m2 (sumf) | **2** | 8 | 16 |
| i32m2 (sumi) | **2** | 8 | 16 |
| i16m1 (sLo/sHi) | **1** | 8 | 16 |
| i8mf2 (nibble) | 1 | 8 | 16 |

⇒ **vl 8→16 = 同一批 vreg、每个 vreg 从半空到填满**，vreg **数**一个不变。

### 1.2 emitter 在 VLEN256 用的是哪档 LMUL？—— 仍 mf2，不升 LMUL

`emitRepackKQuantGemmBodyQ4K` 的 LMUL 锚点 `coreLmul`：RVV1.0 默认 **"mf2"**（`l8=mf2, l16=m1, l32=m2`），只有 RVV0.7.1（xtheadvector）才强制 "m1" 整-LMUL 链。K1 = SpacemiT X60 = **RVV1.0**（非 0.7.1）→ **coreLmul 保持 "mf2"**。源证：
- `RVVToEmitCBlockQuantLinear.cpp:6828` 注释「amortizing path on rvv (VLEN128 mf2) **and K1 (VLEN256 mf2)** is columnsPerPass==4」。
- `RVVToEmitCBlockQuantLinear.cpp:6819` `numHalves = weightInterleave/half = 2 @128, 1 @256`。
- ★lit gate `rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir` V256 CHECK：`__riscv_vfmv_v_f_f32m2(…, %VL16)` / `__riscv_vmv_v_x_i32m2(…, %VL16)` —— **f32m2 / i32m2 后缀与 V128 逐字相同，只有 vl 字面 8→16**。

⇒ VLEN256 native 就是 **e32m2 满 16 lane**（现 VLEN128 也 e32m2 但半空）——**同 LMUL、只是 lane 用满**（正是 task 问的第一读法）。**不需要** e16m2/更大 LMUL。

### 1.3 逐 accumulator 的 vreg 占用（peak hot-loop，per h-strip）

峰值活集在**单个 h-strip 内**（numHalves 是结构 tile：每个 h 有独立 block loop，peak 只含一个 strip 的累加器——`RVVToEmitCBlockQuantLinear.cpp:7082-7101` + `:7102` 的 `for h`）。d = columnsPerPass = activationInterleave = 4：

| accumulator | 类型 | 每个 vreg | 份数 (d=4) | 小计 vreg |
|---|---|---|---|---|
| sumfVar[c]（f32 主累加，`:7107`）| f32m2 | 2 | 4 | 8 |
| sumiVar[c]（i32 scale 主项，`:7255`）| i32m2 | 2 | 4 | 8 |
| sLo[c]+sHi[c]（i16 partial，`:7343`）| i16m1 | 1 | 2×4 | 8 |
| **6d 小计** | | | | **24** |
| 解码/标量常量/循环 overhead（C，S6 实测标定）| | | | **7** |
| **peak_hot** | | | | **31 (v30)** |

- `bsums` MIN 累加器 + decode strips + d/dmin widen 已被 S6 **stack-panel / on-demand** 挪出热集（`:7160-7237` scalePanel/minPanel/bsumsPanel = ArrayType 栈数组，`:7392-7396` d/dmin 延迟到 fold）→ **0 vreg on peak**（S6 findings §1 硬件确认：hot-loop live-value spill 23→0）。

**这 31 = 6d+7 与 VLEN 无关**（vreg 数 LMUL 定）。所以 **VLEN256 native vl=16 的 peak_hot 仍 = 31 (v30)，与 S6 在 VLEN128 实测逐字相同。**

### 1.4 numHalves 2→1 对寄存器是「更好」不是「更坏」

VLEN128：numHalves=2，两个 h-strip **顺序**执行（各自 block loop），peak 只含一个 strip → 不叠加。VLEN256：numHalves=1，单 strip。**peak per-strip 不变，总 strip 数减半。** 若有任何方向的变化，是**减负**（少一遍 strip 的 setup），不是增负。

### 1.5 栈用量（唯一随 half 增长的量，非 vreg）

scale/min panel = `subPerSuper·half` int16 = 4·8=**32**（V128）→ 4·16=**64**（V256，lit gate `!emitc.array<64x int16_t>` 已 CHECK）；bsums panel = `columnsPerPass·half` int32 类似翻倍。这些**在栈上、非 vreg**，vle/vse 各一条（vl=16，指令数不变）。**栈翻倍、vreg 不动、不触崖。**

### 账 1 结论
**vl=16 native 不越崖。peak_hot = 31 vreg (v30) ≤ 32，与 VLEN128 逐字相同。SEW/LMUL 不变（mf2 链、e32m2 满 16 lane）。无需 e16m2/更大 LMUL、无需回 S1、无需新档。**

---

## 账 2 — lane-width(vl=16) × token-tile(d) 交互账

### 2.1 两轴正交（数学）

- **lane-width 轴**：vl = half（8/16），由 **VLEN capability** 键控，填 lane，**不改 vreg 数**。
- **token-tile 轴**：d = columnsPerPass（现 4），由 **cache/schedule** 决定，**LMUL-数扇出累加器**（每列 6 vreg），peak_hot = **6d+7**。

`peak_hot = 6d + 7` 式中**没有 VLEN/vl 项** —— 两轴在寄存器预算上**解耦**。vl=16 既不省也不费 token-tile 的寄存器。

### 2.2 vl=16 与 M1a 已选 loop-interchange 并存 —— 兼容

M1a 选型 = **交换最外两层 for y⇄for x（col-group 外 / row-group 中），d=4 保留**（`token_tile_selection.md` §4-5）。该改动：
- 在**最外两层**循环 header（`RVVToEmitCBlockQuantLinear.cpp:7024/7044` rowLoop/colLoop 对调 + bGroup 提升），**不碰**内层累加器 LMUL/lane 结构；
- **register-neutral**（peak 仍 31，d=4 不变）。

⇒ **vl=16（内层 lane 填满）+ loop-interchange（外层流量解）完全正交并存**，peak 仍 31。两者是 K1-SEAL 点名的**两个独立 emitter-maturity gap**（(a) rvv H-B tile-schedule=loop-interchange，board-specific；(b) VLEN-adaptivity=vl=16，board-无关），可**独立落、独立键控、同库共存**。

### 2.3 vl=16 不给 token-deepening 腾寄存器

若 M1b 试图**同时**累加器深瓦片（d=8）：peak = 6·8+7 = **55 > 32 → 越崖 spill≈23**，**在 VLEN128 与 VLEN256 一样**（vl 不改预算）。即 d≥8 深瓦片在两 VLEN 都被 M1a 已证伪（`token_tile_selection.md` §1-3：d=8 wash、d=16 net-neg），**vl=16 不改变这个结论**。token 收益走 M1a 已裁的 **loop-interchange（d=4 安全）**，不走扇出深瓦片。

### 2.4 两轴能否同库键控？—— 能

同一 schedule library，两个正交键：
- `half`（→ vl）键控 = **VLEN capability fact**（`deriveRepackHalfLanes(vlen,16)`，已是现设计）。
- `loop-order`（col-outer vs row-outer）/ `d` 键控 = **cache fact**（col-group 权重 ≤ L1d ⇒ col-outer；类比现有 `numHalves = weightInterleave/half` 的 VLEN-keyed tile 计数）。

无冲突：一个填 lane、一个排循环，触碰的 emitter 面不相交（内层 vsetvli 常量 vs 外层 for header）。

### 账 2 结论
**vl=16 与 token-tile 正交。M1a 的 loop-interchange(d=4) 与 vl=16 兼容并存（peak 仍 31），两轴同库、键控在不同 fact。vl=16 不放宽 token-deepening 预算（d=8 仍越崖）。**

---

## 账 3 — 「半宽核」根因定位

**问**：emitter 没按 VLEN256 出满宽，还是 S6 staging 假设了 vl=8？**答**：**两者都不是**——是**部署配置失配** + **capability-input 缺 zvl256b**。

### 3.1 排除「emitter 没出满宽」

`RVVRepackStripWidthMaterialization.cpp:72-83` `deriveRepackHalfLanes(vlenBits, weightInterleave)`：
```
if (vlenBits < 128 || weightInterleave <= 0) return 0;
lanes = vlenBits / 16;              // e16m1 lane count
return std::min(lanes, weightInterleave);   // 128→8, 256→16, 512+→16
```
VLEN256 → `min(256/16, 16) = 16`，**没有钳到 8**。发射器**已能**出 half_lanes=16 / numHalves=1 / vl=16 native。lit gate `rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir` V256 已 CHECK：`literal "16"` + f32m2/i32m2 @vl16 + `array<64x int16_t>` 面板。**已实现、已测、byte-exact**（gate 注释 line 19：VLEN256 一 16-lane strip 与 VLEN128 两 8-lane half 读**byte-identical** repacked data）。⇒ **emitter 不是根因。**

### 3.2 排除「S6 假设 vl=8」

`emitRepackKQuantGemmBodyQ4K` 全程 `half` 是**入参**（`:6757` 形参 `int64_t half`），`vl8 = sizeLit(half)`（`:6850`），numHalves = weightInterleave/half（`:6819`）。S6 的 stack-panel / min-fold / decode-panel 全部按 `half`/`numHalves` **参数化**（panel 尺寸 `subPerSuper*half`、strip 偏移 `h*half`）。**S6 没有硬编码 vl=8。** ⇒ **S6 staging 不是根因。**

### 3.3 真根因（两层）

**层 1 — 部署（直接因）**：K1-SEAL **复用**已 emit 的 sealed 源 md5 90d454da（报告 line 3：「无 re-emit / 无 tcrv-opt」），而该源是在 **rvv / VLEN128 板**上 emit 的（S6 findings header：「Board: rvv / VLEN128」）→ half=8、numHalves=2 **烤成编译期常量**。搬到 k1/VLEN256 后，vsetvli 的 AVL 请求 8（`vsetvli t,8,e32,m2` → vl=min(8,VLMAX=16)=8）→ **填 8/16 lane = 半宽 → 0.75× hand-brick vl=16**。这是 K1-SEAL 为**隔离 kernel** 故意选的「不 re-emit」，副产物就是半宽。

**层 2 — capability-input（更深因，决定「re-emit 也不够」）**：strip-width pass 的 vlen 来自 `deriveMinimumVLEN(march, isaVectorHints)`（`:99-100`），它取 **ISA 保证的最小 VLEN**（`RVVCapabilityProfile.cpp:289-333`）：只认 `zvl{N}b` 显式 token 取最大 floor，否则 full-V 兜底 128。**plain `rv64gcv`、以及 k1 clang build march `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`（K1-SEAL line 6）都不含 `zvl256b` → 派生 128 非 256**。即便在 k1 上**用该 march re-emit，仍出 vl=8**——因为 march 只保证 ratified Zvl128b，编译器**正确地**不擅自假设 VLEN256。要出 native vl=16，**emit 期 march 必须带 `zvl256b`**（或注入 VLEN256 capability fact / vlenb=32）。

### 账 3 结论
**半宽根 = 部署了 VLEN128-emit 的 sealed 源到 VLEN256 板（无 re-emit）+ emit 期 march 缺 `zvl256b`（deriveMinimumVLEN 只给最小 128）。发射器与 S6 都已 VLEN-参数化、已 lit-测能出满宽 vl=16。这是 config/front-door 问题，非 kernel 能力缺陷。**

---

## M1 可行性判定 —— GREEN（可行，不回 S1，不需新档）

| 判据 | 结论 | 依据 |
|---|---|---|
| vl=16 native ≤32 vreg? | **YES（31, v30）** | 账 1；LMUL 不变、vreg 数 VLEN-无关 |
| 需回 S1 / 新档? | **NO** | 账 1.3-1.4；同 mf2 链、peak 逐字同 VLEN128 |
| 发射器已能出 vl=16? | **YES，已 lit-测** | `deriveRepackHalfLanes(256)=16`；flip-gate2 V256 |
| byte-exact 路径存在? | **YES** | 16-way interleave 不变量（WideningOps:30-34）；gate byte-identity |
| 与 token-tile/loop-interchange 兼容? | **YES（正交）** | 账 2 |
| 改动性质 | **front-door capability-input（非 kernel 重写）** | 账 3.3 层 2 |

**判定：M1 曳光弹可行。** 核心不是「写新宽核」（已存在），而是「**在 emit 期把 VLEN256 capability 事实喂进 strip-width pass**」，让 `deriveMinimumVLEN→256 → half_lanes=16 → numHalves=1 → vl=16 native`，再 re-seal k1 部署源。

---

## M1 曳光弹实现要点

1. **Front-door lane-width 轴（capability-keyed on VLEN，禁硬编码）**：
   - 不动 emitter body（已参数化）。改的是 **emit-time capability 输入**：给 k1-targeted 的 q4_K GEMM lowering 喂 `-march=…_zvl256b`（或等价 VLEN256/vlenb=32 capability fact 进 `RVVRepackStripWidthMaterialization` 的 `march`/`isaVectorHints` pass option）。
   - 键控依据 = **VLEN capability fact**（与现有 `deriveRepackHalfLanes` / flip-gate2 同机制），**不写死 half=16 常量**。
2. **Re-emit + re-seal（禁手改 .c）**：以 VLEN256 march 走 lower→translate→emit 全链重出 q4_K GEMM sealed 源（新 md5，half=16/numHalves=1），**禁**在部署 .c 上手 swap（front-door 纪律）。
3. **byte-exact gate（复用 S6 harness 型）**：
   - vl=16 kernel 对 **k1 板 scalar oracle** byte-exact（经 16-way interleave 不变量传递；**注意**：与 vl=8 kernel **不**逐字相同——不同板不同核，各自 vs 自己板的 oracle）。
   - raw fp32 `cmp`：int regime（unit scale = 精确整核）+ norm regime，nr64 & nr16，0-mismatch。
   - objdump 验：vsetvli AVL=16、累加器 f32m2/i32m2（**非** e16m2/m4）、maxVreg ≤ v30、vwmacc multiset 与 vl=8 版对应（每 lane 独立列点积、K 累加序不变）。
4. **k1 板 A/B（M1 上板，board-specific 兑现）**：
   - 三方配对：**emitted vl=16**（新）vs **hand-brick vl=16**（K1-SEAL Cbrick 14.47 t/s）vs **emitted vl=8**（K1-SEAL A 10.84 t/s，STALE 半宽基线）。
   - 静态预测：填满 16 lane 应把 A/Cbrick 从 **0.750×** 拉向 **≥parity**（半宽浪费一半向量宽的直接回收）。
   - **诚实 caveat**：账只证「vl=16 可发射、≤32-vreg、byte-exact」= **可行性**；是否真达 hand-brick parity 是**板问题**（hand-brick 可能另有 schedule 优势），须 M1 A/B 实测，不由静态账下结论。
5. **与 loop-interchange 的次序（正交，可分可合）**：vl=16（board-无关）与 loop-interchange（rvv-board-specific）是两个独立 gap，可**先 vl=16 单独 A/B**（隔离 lane-width 因子），再叠 loop-interchange；两轴同库、键控不同 fact，触碰面不相交。

---

## 附：关键源坐标（复现用）

- 半宽派生：`lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp:72-83`（deriveRepackHalfLanes）、`:96-119`（pass 取 march→deriveMinimumVLEN→gate<128 skip）。
- 最小-VLEN 派生：`lib/Plugin/RVV/RVVCapabilityProfile.cpp:289-333`（deriveMinimumVLEN；zvl token / full-V floor 128 / embedded 0）。
- Emitter 参数化：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:6747`（body 入口，`half` 形参）、`:6819`（numHalves=weightInterleave/half）、`:6828`（K1 VLEN256 mf2）、`:6850`（vl8=sizeLit(half)）、`:7102`（for h numHalves）、`:7107/7255/7343`（sumf/sumi/sLo-sHi 累加器）、`:7160-7237`（S6 stack panels）。
- 16-way interleave byte-identity 不变量：`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:30-43`。
- ★决定性 lit gate：`test/Conversion/RVV/rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir`（V128 vl8 / V256 vl16、同 kernel 仅 -march 变、f32m2/i32m2 LMUL 逐字同、byte-identical repacked data）。
