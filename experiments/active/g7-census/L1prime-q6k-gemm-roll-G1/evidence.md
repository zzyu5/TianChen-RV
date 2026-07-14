# L1' q6_K@rvv GEMM whole-K-nest roll — 攻坚 G1 可行性（纯本地静态账·无板）

> **线定位**：G7 终编成令四·L1' 攻坚（K-quant GEMM emitter roll·q6_K@rvv 代表·deepest loss 0.037×）。
> **前提**：batch1 census（`249145ff`）已定 GAP — q6_K@rvv GEMM prefill 0.037×（deepest·vsetvli 6278）·loss 根因=**gcc-15.2 vsetvli-storm+spill 于巨型 full-unroll body·非 MAC 质量**。
> **杠杆**：whole-K-nest roll 谱系（GEVM 已 rolled `34fded0d`/`2bbee909`）。**核心问题=whole-K-nest resident 结构能否 collapse vsetvli-storm ∧ 规避 re-decode killer（G6-B narrow-tile GEMM roll 0.41-0.63×）= NET recovery？**
> **约束**：read-only 分析 + hand-construct 独立 .c·勿改 emitter·禁上板·byte-exact 代数硬门·[K-10] 判据。
> **工具链**：SpacemiT gcc-15.2.0（`/home/kingdom/spacemit-ime/.../riscv64-unknown-linux-gnu-g++`·`-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on` = batch1 census flags·部署 gcc 域）。

---

## ★裁决 TL;DR

**NET RECOVERY（静态账坐实·Exit A）**。whole-K-nest roll of q6_K GEMM（**保 4-column tile·非 narrow-tile**）：

1. **collapse vsetvli-storm**：per-super-block DYNAMIC vsetvli **6578 → 21（−313×）**·hot p-loop 内 **0 vsetvli**（gcc hoist·整个 p-loop body 均 e8,mf2 单 vtype——decode e8 ops ∧ widening vwmacc.vx 源 SEW=e8 同 vtype）。
2. **★re-decode killer 【不适用】**：decode-arithmetic op count **两形态恒等（vand 768 / vsrl 640 / vsll 512 / vor 512 / vsub 512·合 2944·rolled==unrolled）** → **权重每元素解码恰一次·跨 4 activation columns 共享**（emitter 本就 amortize·roll 保持）。G6-B 的 re-decode 罚源自 **narrow-tile roll**（窄 output-tile → 每权重多 pass）·**whole-K-nest roll 不窄 tile → 无 re-decode**。
3. **GEMM 专属残留=accumulator SPILL（register cliff·非 re-decode）**：spill(store+reload) **4444 → 1869/super-block（−2.4×）**·但**未清零**（hot p-loop 每迭 10 store+10 reload）。元凶=4-column tile 的 **32 i16 partials + 8 i32 sumi + decode 工作集 > 32 vreg**（maxVreg v31·两形态皆 spill）。**GEVM（M=1·~8 partials）的 clean 0-spill hot loop 对 GEMM 4-column tile 不可达**。
4. **NET**：byte-exact 恒等 useful work（vwmacc 2304 + decode 2944 两形态同）·rolled **总 dynamic v-insn 21756 → 7822（−2.8×）**·由**消 vsetvli-storm（313×）+ 消冗余 weight-load（vle8 3004→416·7.2×）+ 减 spill（2.4×）** 驱动。**NET 明确正**——vsetvli 省 ≫ spill 残留·**re-decode 根本未发生**。
5. **预估**：0.037× → 保守 **~2.8-3×**（instruction-count 线性·→ ~0.10×）·乐观至更高（vsetvli-stall 超线性·gcc vtype-flush stall）。**sub-parity**（density floor：vl=8 dual-strip element-wise vwmacc + 6-bit dual-plane 重解码 vs stock wide block-dot 未变）。**非 claimed WIN·是自伤 [CASE-COMPILER-ASYMMETRY] codegen 崩塌的 recovery**。
6. **出口**：**实装 + G2 board-validation 资格**（Exit A）。板定 exact ratio + prefill e2e 传导（compute-bound prefill 传导概率 > GEVM decode）。
7. **[K-10]**：GEMM roll = **结构级**（独立 GEMM Emission Plan·非 GEVM emit_loop_schedule key 的参数翻转）——register calculus 根本不同（32-partial spill cliff vs GEVM 8-partial clean fit）·印证 memory `[K-10] GEMM↔GEVM 结构级`。

---

## 1. 步骤1 — q6_K@rvv GEMM 现状反汇编（read-only）

**核 provenance**：`weft-opt test/Conversion/RVV/rvv-to-emitc-repack-gemm-q6-K-q8-K.mlir --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`（md5 `bf477693` = batch1 census MANIFEST 逐字·同核）。发射真源 = `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:10638 emitRepackKQuantGemmBodyQ6K`（read-only）。

### 1.1 loop 结构（VLEN128·half=8·numHalves=2·activationInterleave=4·columnsPerPass=4）
```
rowLoop      for y in nr/4          RUNTIME emitc.for   (OUTER·4-col row-group)
 colLoop     for x in nc/16         RUNTIME emitc.for
  [cLo pass  columnsPerPass=4       C++ compile-time·单 pass]
   sumfVar[c][h]  4col×2half f32m2  ← persist across blockLoop
   blockLoop  for l in nb           RUNTIME emitc.for    (K super-block)
    ┌─ super-block BODY  = FULLY UNROLLED ─────────────────┐
    │ j(2)×sh(2)×k(2)×p(8) = 64 element-positions          │  ← storm 来源
    │   sumiVar[c][h]  4col×2half i32m2                     │
    │   scaleVal[h][q] 2half×4quad i16m1  (shared k,cols)   │
    │   acc[c][h][q]   4col×2half×4quad = 32 i16m1 partials │  ← register cliff
    │   p-loop: decode wq[h][q] (SHARED across 4 cols) →    │
    │           c×q×h vwmacc16 into acc                     │
    │   scale fold: sumiVar += Σq scaleVal·acc  (vwmacc.vv) │
    └──────────────────────────────────────────────────────┘
    end-block fold: sumfVar += cvt(sumiVar)·(dF32·aD)
   store: vse32 sumfVar[c][h]
```

### 1.2 objdump 现状（gcc-15.2 -O2·本地复现 batch1 census）
| metric | 本线复现 | batch1 census evidence.md | 判 |
|---|--:|--:|---|
| vsetvli | **6182** | 6278 | ✓ MATCH（板-vs-本地 gcc-15.2 微差）|
| vwmacc（2048 vx + 256 vv）| **2304** | 2304 | ✓ 逐字 |
| vs*r.v（spill store）| 2222 | 2234 | ✓ |
| vle8 | 2788 | — | |
| maxVreg | **v31** | v31 | ✓ 满寄存器压力 |
| total v-insn | 20906 | — | |

**★vsetvli 6182 来源（SEW toggle 分解）**：`3071 e8,mf2` + `120 e16,m1` + `1 e32,m2`。**e8,mf2 storm（3071）= weight-decode（u8 vand/vsrl/vsll/vor/vsub）在巨型全展开体内被 gcc-15.2 与 e16 vwmacc 逐 op 交错·无法 batch**（register pressure 逼 decode↔macc 交织）→ 每 decode op 附近重设 vtype。**weight-decode 位置**：p-loop 内 per-(j,sh,k,p,h) 解码 wq·**SHARED 跨 4 activation columns**（emitter `for c` 复用 `wq[h][q]`·非重解码）——output-tile 共享【已存在】·[re-decode 已 amortize]。

---

## 2. 步骤2 — whole-K-nest GEMM roll 设计（规避 re-decode）

**hand-construct**（`raw/q6k_gemm_roll_gen.c`·gen 于 `raw/gen.py`）：ONE source·两函数 `q6k_gemm_rolled`/`q6k_gemm_unrolled`·**共享 SUPERBLOCK_BODY 宏**（同 decode+vwmacc op 序·asc j,sh,k,p,c,q,h）·唯一差 = 元素 nest `#pragma GCC unroll 1`（rolled·runtime）vs `64`（unrolled·全展开=现行部署）。RVV sizeless type 禁数组 → c/q/h compile-time 展开成 32 具名 acc 变量（**此禁数组本身即 register-pressure 信号**）。忠实 `emitRepackKQuantGemmBodyQ6K`（offsets ql=1312 qh=288 scales=32 aq=16·stride 3360/1168·VLEN128 half=8）。

### 2.1 roll 如何 collapse vsetvli ∧ 规避 re-decode
- **roll 元素 nest（j,sh,k,p）成 runtime loop**：gcc 把 hot p-loop body（均 e8,mf2 单 vtype·decode e8 ∧ vwmacc.vx 源 SEW=e8）的 vsetvli **hoist 出 loop → hot loop 内 0 vsetvli**。scale-fold（e16 vwmacc.vv）留 COLD 区（per-jshk·不与 hot 交织）→ 消 e8↔e16 storm。
- **★规避 re-decode（关键）**：decode wq[h][q] 在 p-loop 内、**c-loop 之外** → 每元素解码一次·**驻寄存器共享跨 4 columns**（保 emitter amortize）。**非 narrow-tile**（tile 仍 4-col）→ **无 cross-column re-decode**。decode-arith count 两形态恒等（§3.2）实证零 re-decode。

### 2.2 ★register-budget-fit（P2 教训·GEMM 决定性差异）
| 配置 | resident 向量态 | gcc-15.2 spill | 判 |
|---|---|---|---|
| **GEVM（M=1·34fded0d）** | ~8 i16 partials + sumi/bsum/sumf | **0 hot spill·28 vreg** | ✅ CLEAN FIT |
| **GEMM whole-nest roll（本设计·4-col tile）** | **32 i16 partials + 8 i32 sumi + decode 工作集** | **hot 10 store+10 reload/迭·v31** | ⚠ **SPILL·register cliff** |
| 假想 narrow-tile roll（G6-B） | 窄 tile → 每权重多解码 | — | ❌ re-decode 0.41-0.63× |

**⇒ GEMM 4-column tile 的 32-partial resident set 结构性 > 32 vreg → 必 spill**。这是 GEVM（8-partial clean fit）与 GEMM 的**判别键**·**非 re-decode·是 accumulator spill**。roll 把 spill **减 2.4×**（4444→1869）但**不清零**（GEVM 的 clean 0-spill 对 GEMM 不可达）。

---

## 3. 步骤3 — G1 静态账（gcc-15.2 -O2·VLEN128·hand-construct A/B）

### 3.1 ★per-super-block DYNAMIC 账（rolled hot×8-trip + cold vs unrolled 直线）
| metric | **UNROLLED**（现行部署）| **ROLLED**（whole-nest）| Δ | 判 |
|---|--:|--:|---|---|
| **vsetvli** | **6578** | **21** | **−313×** | ★storm 塌（hot loop 内 0）|
| vwmacc（useful·2048vx+256vv）| **2304** | **2304** | **1.0×（恒等）** | byte-exact useful work |
| decode_arith（vand+vsrl+vsll+vor+vsub）| **2944** | **2944** | **1.0×（恒等）** | ★**零 re-decode 证据** |
| vle8（weight load）| 3004 | **416** | −7.2× | 消冗余 reload |
| spill_store（vs*r.v）| 2216 | 926 | −2.4× | 残留 register cliff |
| spill_reload（vl*re*.v）| 2228 | 943 | −2.4× | 〃 |
| **total v-insn** | **21756** | **7822** | **−2.8×** | NET |
| maxVreg | v31 | v31 | 同 | 皆满压 |

（unrolled 直线体 static==dynamic/super-block；rolled hot p-loop body ×8-trip + cold×1；8 个 jshk hot loops 各 ~200 行·gcc unroll 了 j/sh/k 成 8 个 p-loop。）

### 3.2 byte-exact 硬门（代数模型·mismatch=0）
- **DYNAMIC op-stream 恒等**：useful work（2304 vwmacc + 2944 decode-arith）+ vsext 32 + vfmacc 8 **两形态逐 op 恒等**·roll = 纯 loop-structure 变·不改算术序 → byte-exact by construction（同 GEVM 34fded0d/2bbee909 "identical vwmacc accumulation order; only residence changes"）。
- **★decode-arith 恒等（2944 both）= 零 re-decode 的直接证据**：若 re-decode 发生·rolled 的 vand/vsrl/vor/vsub 会↑（每 column 重解码）→ 实测恒等 → 权重每元素解码一次·共享跨 4 columns。
- **DRIVER**（`raw/q6k_gemm_roll_gen.c -DDRIVER` A/B memcmp）已 build（`q6k_ab`）·**数值 A/B = DEFERRED**（本地无 RVV：qemu-riscv64 缺·GNU sim signal 4）→ 依 construction-identity + G6-B/GEVM 板 A/B 先例（同 emitc.for-vs-unroll transform·真硅 0-mismatch）。**诚实标注**：byte-exact = by-construction·非本线新板跑。

### 3.3 ★NET 判读（vsetvli 省 vs re-decode/spill 涨）
- **vsetvli collapse 省**：6557 vsetvli/super-block（313×·storm 主因）。
- **re-decode 涨**：**0**（decode-arith 恒等·whole-nest 不窄 tile）。
- **spill 净变**：−2575/super-block（4444→1869·spill **减** 不增）。
- **⇒ NET = total v-insn −2.8×（21756→7822）·全部方向为省**。**vsetvli collapse 主导·re-decode 零·spill 减**·**NET 明确正（recovery）**。

---

## 4. 步骤4 — 攻坚 G1 裁决

### 4.1 三门
| 门 | 判据 | 结果 |
|---|---|---|
| **(a) collapse vsetvli-storm** | 6578→21（−313×）·hot 0 | ✅ YES |
| **(b) 规避 re-decode killer** | decode-arith 恒等 2944·weight 跨 4-col 共享·非 narrow-tile | ✅ YES（re-decode 未发生）|
| **(c) register-budget** | 32-partial > 32 vreg → **spill 但减 2.4×**（非增）·GEVM clean-fit 不可达 | ⚠ FIT-with-residual-spill |
| **(d) byte-exact** | useful+decode op-stream 恒等·板 A/B deferred | ✅ by-construction |
| **⇒ NET recovery？** | vsetvli 省 ≫·re-decode 0·spill 减 | **✅ NET RECOVERY** |

### 4.2 裁决 = **NET RECOVERY → 实装 + G2 board-validation 资格（Exit A）**
- **预估 ratio**：0.037× → 保守 ~2.8-3×（instruction-count 线性·~0.10×）·至乐观更高（vsetvli vtype-flush stall 超线性）。**sub-parity**（density floor 未动：vl=8 dual-strip element-wise vwmacc + 6-bit dual-plane 重·vs stock wide block-dot）。
- **诚实定性**：**非 perf-covered WIN·是 [CASE-COMPILER-ASYMMETRY] 自伤 codegen 崩塌的 recovery**（我方 un-pipelined full-unroll + gcc-15.2 = 崩塌·roll 愈合）。到 parity 需另攻 density（vl-widen / repack-format·[远期]）。
- **perf 传导**：GEVM roll = ZERO perf green（[CASE-MICRO-E2E] decode memory-bound）·**但本靶=GEMM prefill·compute-bound**（census q6 hot≈cold 全 nr 平坦=compute-bound）→ kernel-axis recovery **传导 prefill e2e 概率 > GEVM decode**·**待 G2 板定**（不预先 claim）。

### 4.3 [K-10] 判定 = **结构级**（独立 GEMM Emission Plan）
- emit_loop_schedule attr【已存在】（GEVM 建·C2 marginal-cost 低）·但 GEMM rolled leaf 的 **register-cliff 管理（accept spill / col-tile sub-block）与 GEVM（clean 8-partial fit）根本不同** → GEMM roll **非 GEVM key 的参数翻转·是自有 emission 结构**·须独立 plan。印证 memory `[K-10] GEMM↔GEVM 结构级·禁结构级当旋钮`。

---

## 5. 复现
```
OPT=build-weft/bin/weft-opt; TR=/usr/bin/mlir-translate-20
GPP=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-g++
FLAGS="-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on"
# CURRENT 核（read-only·md5 bf477693 = census 逐字）
$OPT test/Conversion/RVV/rvv-to-emitc-repack-gemm-q6-K-q8-K.mlir --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp > raw/q6k_gemm_CURRENT.emitc.c
$GPP $FLAGS -S raw/q6k_gemm_CURRENT.emitc.c   # vsetvli 6182 / vwmacc 2304 / vs*r.v 2222 / v31
# A/B（rolled vs unrolled·同源·byte-exact）
python3 raw/gen.py && $GPP $FLAGS -S raw/q6k_gemm_roll_gen.c -o raw/q6k_gemm_roll_AB.gcc15.O2.s
#   per-func split on _Z15q6k_gemm_rolled.. / _Z17q6k_gemm_unrolled..（C++ mangled）
# 数值 A/B（DEFERRED·本地无 RVV）：$GPP $FLAGS -DDRIVER -static raw/q6k_gemm_roll_gen.c -o q6k_ab（板可跑）
```
**禁 git · 禁上板 · 禁改 tracked 源码（RVVToEmitC* 只读·hand-construct 全 raw/）· byte-exact by-construction + 板 deferred。**
