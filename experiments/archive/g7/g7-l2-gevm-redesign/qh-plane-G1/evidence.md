# G7 L2 — qh-plane GEVM decode 削重建指令 重设计 · G1 静态账 (零板时·[VERIFY-LADDER] G1 层·禁跳级)

> **retarget**：L2 q4_K 重建-count thesis G1 打回（`../G1-static/evidence.md`：q4_K M=1 损=**停顿型**·zvbb 舰队缺席+0-yield·仅 4.7% amenable → 无重建 headroom）。真靶 = **qh-plane 格**（q5_0/q5_1/q5_K·+43~62% 重建 headroom）。本任务 = **首格 q5_0 @rvv decode M=1**。
> 依据 canon：[VERIFY-LADDER] G1 静态账门（精化①部署编译器测 spill·精化②regime-keyed）· [K-10] 结构级独立 plan · [X-ZVBB] 纯 base-RVV · [PAT-2] P9 GEVM regime-plan · 性能宪章规则1-2 · [CASE-COMPILER-ASYMMETRY]。
> **本任务止于 G1(静态账)+设计 · 禁 git · 禁上板 · 禁改 schema label（本文件仅产建议+工作树 hand-construct 证据）。**
>
> **★裁决(TL;DR)：G1 = 双子门皆过（literal）→ q5_0@rvv 削重建线 获 G2 资格。** register-budget-fit **PASS**（部署 gcc-15 spill-free·16 vreg<32·优于 OLD 18）· instruction-count **PASS(literal 1.047× ≤1.10×)**·**★带 vsetvli-attribution caveat（去 vset 纯工作读 1.76×·载重·由 G2 IPC 裁决）**。重建 headroom **真实兑现**：total v-insn 1.81×→1.047×（−34%）·重建指令 −43%（**区别 q4_K 的 0-yield**·retarget thesis 坐实）。

---

## 0. 方法与工具链 (本地静态账·零板时·可复现)

- **★部署编译器同族本地代理（G1 精化① 合规升级）**：本轮用 **SpacemiT riscv64 `gcc-15.2.0`**（`spacemit-toolchain-.../bin/riscv64-unknown-linux-gnu-g++`·`-march=rv64gcv_zvfh -mabi=lp64d -O3`）——**与 rvv 板 shipped gcc-15 同大版本族**。q4_K 前案只能本地 clang-20（异编译器·spill 膨胀 50×·只可相对排序）；**本轮 spill/live-set 直接以部署编译器族测**（精化① 满足度更强）。clang-20 备作向量计数交叉核（casefile §0 已证向量轴编译器鲁棒）。
- **发射管道**：`build-weft/bin/weft-opt <mlir> --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`（沿用 EXPORT_RECIPE）。fixture = `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir`（auto-select → mf2/half_lanes=8 two-strip repack-GEVM·**M=1 decode 正靶**）。
- **对手（stock 同路径）**：`ggml_vec_dot_q5_0_q8_0`（`llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:328`·block-dot·q5_0 无 compiled riscv repack → 出货即 block-dot·`raw/stock_q5_0_gevm.c` 自包含抽出·同 gcc-15 编）。
- 归一化 = **每 output-block（一 output row × 一 32-elem block）向量指令数**（消去 repack 16-row 摊销 vs stock 1-row 的粒度错配·[casefile §4.1] unrolled-vs-rolled 恒-fail 读法回避）。我方 nib-loop body(=一 step) × 16 steps 产 16 outputs·1 block → per-output-block = nib-loop-body v-insn。stock gcc ×2 展开 → 42/2=21。

---

## 1. ★设计 (削 M=1 重建指令路径 · 令 §一)

### 1.1 根因（为何 q5_0 重建重·真 headroom 所在）
stock q5_0 用**极优**技巧：qh 32-bit 平面 `vlm_v_b4` 一次装成**原生 bool mask** → `vmnand` 反相 → **单条 masked `vsub_vx_i8m2_mu`** 把 5th-bit **与** −16 bias **一并**施加（整 block 重建 ~3 op·native 布局）。**我方 repack-GEVM 的 transposed-qh 布局**（qh_lo/qh_hi u16×16·bit r=row r 第 k 位）被 OLD 发射器用 **per-lane 展开**重建：每 decode 单元 `vmv splat + vid + vsrl_vv + vand&1 + vsll<<4 + vncvt`(→{0,16}) + `vor` + `vsub16` ≈ 8 op·×4(2 strip×lo/hi)×16 step。**这就是 +62% 重建**——布局逼出的 per-lane 展开税。

### 1.2 削重建杠杆（board-available·纯 base-RVV·非 zvbb）
**关键洞察：transposed-qh 的 u16-per-step 布局【可直接】装成 per-lane bool mask**——step i 的 u16 中 bit r = row r 的第 5 位；strip h（8 rows）= 该 u16 的 byte h（`vlm_v_b16` 从 `qh_base + i*2 + h` 装 8 bit·lane l = bit l = row(l+8h)·**与 OLD expandQhBit 的 (vid+8h) 选位逐位相等**）。∴ 可把 stock 的 native-mask masked-vsub 技巧**移植进 transposed 布局**：
- **REDESIGN-B（终选·stock masked-vsub 复刻）**：每 decode 单元 = nibble-extract(`vand`/`vsrl`,1) + `vlm_v_b16`(1) + `vmnand`(反相,1) + `vsub_vx_i8mf2_mu`(mask, nib, nib, 16)(1) = **4 op**（bias+5th-bit 融进一条 masked-vsub）。撤销 OLD 的 splat/vid/vsrl_vv/vand/vsll/vncvt/vor 全链。
- （REDESIGN-A 变体 = add-trick `vsub16 + masked-vadd`·省反相但 gcc 多 3× `vmv1r.v` 寄存器搬运 → 25 v-insn·B 优·见 §2。）
- **TG=1 不变**（撤 bank 加宽·P2 已证反噬·register-budget-fit 前提·非收益源）。

### 1.3 byte-exact 由构造（[K-10] 结构级 provenance·核累加序不动）
`weight=(nibble|(bit<<4))−16 ≡ bit?nibble:(nibble−16)` = masked-vsub(inverted mask)·**代数恒等**。仅改**重建指令编排**（per-lane 展开 → native bool-mask）·per-block leaf 的 nibble 抽取 / vwmacc 累加序 / scale-fold **逐字不动**。**实证核验**：`raw/byteexact_model.c`（纯-C 模型·穷举 nibble∈[0,15]×bit∈{0,1}）→ **q5_0 0/32 · q5_1 0/32 mismatch = BYTE-EXACT**。选位等价见 §1.2（vlm byte-h ≡ vid+8h）。

---

## 2. ★G1 静态账（部署 gcc-15·per output-block=32-elem·`raw/*.gcc15.O3.s`）

| 变体 | v-insn/output-block | 比 stock | 重建 v-insn | spill(store-to-sp) | distinct vreg | vsetvli/block |
|---|---|---|---|---|---|---|
| **STOCK** (block-dot) | **21** | 1.00× | ~6 | 0 | — | ~8.5 (vl-toggle) |
| **OLD** (per-lane expand) | 38 | **1.81×** | ~28 | 0 | 18 | 2 |
| **★REDESIGN-B** (masked-vsub) | **22** | **1.047×** | ~16 (**−43%** vs OLD) | **0** | **16** | 0 |
| REDESIGN-A (add-trick) | 25 | 1.190× | ~16 | 0 | 16 | 0 |

REDESIGN-B nib-loop(.L4) histogram：`4 vwmacc.vx`(productive) · `4 vmnot.m`+`4 vlm.v`+`4 vadd.vi`(=masked vsub−16 施 5th-bit+bias)+`2 vsrl.vi`+`2 vand.vi`(重建 16) · `2 vle8.v`(memory)。

### 2.1 判读
- **register-budget-fit 子门 = ✅ PASS**：部署 gcc-15 上 OLD/NEW-B **均 spill-free**（0 store-to-sp·无 stack frame）·NEW-B **16 distinct vreg < 32 预算**·且**优于 OLD(18)**（重建撤 u16 中间量·更省寄存器）。P2 TG=2 的 534-spill trap 无复现（TG=1 保持）。
- **instruction-count 子门 = ✅ PASS (literal 1.047× ≤ 1.10×)**。
  - **★★vsetvli-attribution caveat（载重·flag for G2）**：literal total-instruction 读法含 vsetvli。stock 因 vl-toggle(qk/2↔qk) 付 **~8.5 vsetvli/output-block**·我方 constant-vl(=8) **0 vsetvli in-loop**。**去-vsetvli 纯工作读**：NEW-B 22 vs stock **12.5** = **1.76×**（未 parity·但仍从 OLD 的 2.88× 显著改善）。∴ 1.047× PASS **有约一半裕度来自 constant-vl 设计避开 stock vset 税·非纯重建削减**。literal 门（实际执行指令数）过·**由 G2 IPC/cycles 裁决 vset 归因**（vsetvli 真实耗周期·G2 是正确判别键）。

### 2.2 与 q4_K 打回对比（retarget thesis 坐实）
| | q4_K (打回) | **q5_0 (本案·过)** |
|---|---|---|
| 损型 | 停顿型(IPC 崩·非指令爆) | **计算型(+62% 重建)** |
| board-available 重建杠杆 | **无**(zvbb 缺席+0-yield·4.7% amenable) | **★有**(transposed-qh→native bool-mask·−43% 重建) |
| G1 instruction-count | NOT MET 1.12–1.14× | **PASS(literal) 1.047×**（去-vset 1.76×) |
| total v-insn 改善 | ~0 | **1.81×→1.047× (−34%)** |

→ **"M=1 削重建指令" thesis 在 qh-plane【真兑现】**（区别 q4_K 的结构/停顿型 0-yield）·retarget 判断正确。

---

## 3. [X-ZVBB] 复用确认
REDESIGN-B 全用 **base-RVV V1.0** mask op：`vlm_v_b16` / `vmnand_mm_b16` / `vsub_vx_i8mf2_mu` / `vand`/`vsrl`——**无 zvbb**（舰队缺席不阻·`../G1-static` 已证 rvv∧k1 均无向量 zvbb）。stock 自身也用 `vlm_v_b4`+`vmnand`+`vsub_mu` = 同族 base-RVV·**移植合法**。

## 4. [PAT-2] P9 / [K-10] 建议（改留主会话·仅建议）
- **P9 GEVM regime-plan**：qh-plane 格新增**"native-mask 重建"结构条目**——transposed-qh u16-per-step 装 bool mask + masked-vsub（融 5th-bit+bias）·区别 flat 格的 nibble-only 重建。status：**mechanism 设计成立·G1 双子门 literal PASS·byte-exact 由构造(model 0/32)·性能兑现 pending G2**。
- **[K-10]**：重建-orchestration 改（per-lane expand → native bool-mask）= **参数级/发射编排**（不改迭代拓扑/布局契约/优化目标）·**非新 plan**·在既有 GEVM plan 内（[SEL-1] 键控·结构对齐不变）。register-budget-fit 保持（16<32·反向确认 TG=1 不翻案）。
- **vsetvli 归因**：literal-vs-pure-work 双读法制度化（同 casefile §4.1 精神）·qh-plane 格 G1 pass 须**双读并陈**·G2 IPC 定论。

## 5. 下一步 (令 §四)
**G1 双子门 literal PASS → q5_0@rvv 削重建线 【获 G2 资格】**（[VERIFY-LADDER]·**禁跳级·G2=cold GEVM M=1 micro on k1/rvv·IPC/cycles·N≥10·同会话 A/B·门=≥parity**——G2 正是解 vsetvli-attribution caveat 的判别键）。
- **发射器实装（G2-prep 工程单元·本 G1 任务范围外·deferred）**：`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` GEVM leaf（`emitRepackQ4LaneWiseIntegerCore`·**1653-1717**·**与 sealed GEMM leaf 909-966 隔离**）——`expandQhBit`/`assemble5` 换 `vlm_v_b16`+`vmnand`+`vsub_mu`（q5_1 arm offsetBias==0 用 masked-vadd·无反相）·更新 `rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir` QH lit。q4_K 前案先例 = G1 verdict 不需 .cpp surgery（`../G1-static` "工作树未改发射器"）·故本任务 hand-construct（`raw/q5_0_gevm_REDESIGN-B.c` gcc-15 objdump'd）为 G1 静态账正当代理。
- **corollary（同 leaf 自然铺开）**：q5_1(+43% 重建·offsetBias==0 更简·省反相)·q5_K(super-block 6-bit scale·headroom 更大·需先移 K-quant scale 重建)。

**禁上板 · 禁 git · 无 schema label 改动 · 工作树未改发射器（hand-construct 在 `$CLAUDE_JOB_DIR/tmp` + `raw/`）。**

---
### 复现
```
WEFT=build-weft/bin/weft-opt; F=test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir
$WEFT $F --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp  # OLD emitc
GXX=spacemit-toolchain-.../bin/riscv64-unknown-linux-gnu-g++
$GXX -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/q5_0_gevm_{OLD.emitc,REDESIGN-B,stock_q5_0_gevm}.c   # 计数 nib-loop v-insn
gcc -O2 raw/byteexact_model.c -o /tmp/be && /tmp/be   # 0/32 byte-exact
```
