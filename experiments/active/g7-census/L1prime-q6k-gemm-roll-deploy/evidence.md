# L1' q6_K@rvv GEMM whole-K-nest roll — 发射器实装 + G2 board-validate（实测坐实）

> **线定位**：G7 终编成令四·L1' 攻坚·**实装+G2**（G1 NET RECOVERY 已裁 → Exit A 实装 → board 兑现）。
> **前提**：G1（casefile `../L1prime-q6k-gemm-roll-G1/evidence.md`·commit `85228b96`）静态账裁 NET RECOVERY·预测 0.037×→~0.10×（保守 2.8-3×，乐观更高·vsetvli-stall 超线性）。
> **本线**：把 whole-K-nest roll **实装进部署发射器**（capability-keyed·byte-exact 硬门·绝对隔离），并上 board 兑现 recovery + **on-silicon byte-exact**。
> **诚实定性**：**sub-parity kernel-axis recovery（非 WIN·非 perf-covered green·density floor 不变）**——愈合自伤的 [CASE-COMPILER-ASYMMETRY] gcc-15.2 codegen 崩塌，非硬碰硬赢。

---

## ★裁决 TL;DR

**实装 GREEN·G2 兑现且超预期**。q6_K GEMM whole-K-nest roll 部署发射器落地：

1. **发射器实装**（capability-keyed·[K-10] 结构级）：`emitRepackKQuantGemmBodyQ6K` 加 `bool rolledMainTerm` 分支——dominant per-8-position p-loop materialize 成 runtime `emitc.for`·32 i16 partials（4col×2half×4quad）驻 SSA-register VariableOps（seed 上/load-accumulate-store 内）·weight 解码留 p-loop 内、column loop 外 → **每权重解码一次·跨 4-col 共享（零 re-decode·非窄-tile）**。default（无 stamp）= frozen unrolled = shipped 零漂移。
2. **byte-exact 硬门全过**：
   - ① Conversion/RVV lit **248/248**（prev 247 + 新 rolled fixture）·零回归。
   - ② **on-silicon byte-exact**：4 shape（nr4/8/16 × K2048、nr8 × K4096）rolled vs unrolled 全 **memcmp=0·ndiff=0·maxabsdiff=0**（真硅 bit-identical·比 G1 by-construction 强）。rolled ≡ unrolled ≡ 已 sealed oracle-verified unrolled → rolled correct-by-transitivity。
   - ③ **default 零漂移**：q6_K GEMM 无-stamp emit md5 `f2518117` before==after。
   - ④ **绝对隔离**：全 240-fixture emit-corpus fingerprint before/after **仅 1 处变**（新 rolled fixture）·其余 239（q3_K/q2_K/q4_K/q5_K GEMM·全 GEVM·sealed q4 lane-wise·flat·ternary·native-mask·REDESIGN-B）**byte-identical**。git diff = 仅 2 tracked 源（emitter+header·全 q6_K-局部）+ 1 新 test。
3. **objdump（gcc-15.2 on-board·部署对称）**：
   | metric | UNROLLED(.o) | ROLLED(.o) | Δ |
   |---|--:|--:|--:|
   | vsetvli | **6278** | **21** | **−299×**（storm 塌·hot p-loop 内 0）|
   | vwmacc | 2304 | 512 | −4.5×（main-term 8× roll·scale-fold vv 不变）|
   | spill store | 2234 | 289 | −7.7× |
   | spill reload | 2281 | 324 | −7.0× |
   | maxVreg | v31 | v31 | 同（皆满压·32-partial cliff 在）|
   | obj size | 299632B | 22768B | −13.2× |

   ★vsetvli 6278 逐字复现 batch1 census（`249145ff`）· roll → 21 兑现 G1 预测（6578→21）。
4. **★G2 cold GEMM prefill 兑现（超预期·core 8·pin 8-15 band·gcc-15.2 对称·224MiB flush·reps=12·median）**：
   | K,nr,nc | recovery(roll/unroll) | roll/opp | unroll/opp |
   |---|--:|--:|--:|
   | 2048,4,512  | 6.67× | 0.258× | 0.0387× |
   | 2048,8,512  | 6.78× | 0.265× | 0.0391× |
   | 2048,16,512 | 6.87× | 0.280× | 0.0407× |
   | 4096,8,256  | 6.62× | 0.260× | 0.0393× |

   - **unroll/opp ~0.039×** 逐字复现 census q6_K@rvv 0.037×（deepest loss·基线校准 ✓）。
   - **recovery ~6.7×**（med；best 6.7-6.9×）——**远超 G1 保守 2.8-3×·落在乐观 super-linear（vsetvli vtype-flush stall 消除）档**。
   - **roll/opp ~0.26×**：0.037× → **0.26×**·kernel-axis 大 recovery（~6.7×）·**仍 sub-parity（0.26× < 1.0）**——density floor 未动（vl=8 dual-strip element-wise vwmacc + 6-bit dual-plane 重解码 vs stock wide block-dot）。
5. **e2e prefill 冒烟 = 不跑（判据未满足）**：kernel-axis rolled 仍 0.26×（比 stock 慢 3.8×）→ **无 e2e 优势可传导**（compute-bound prefill 传导 odds 讨论 moot·roll 愈合自伤但未达 parity → 无 win 可上 WinB）。诚实：这是 recovery 非 win。

---

## 出口 = 实装 GREEN·kernel-axis recovery 兑现（非 perf-covered green）

- **成色**：**非 WIN·非 perf-covered green·非 e2e beat·非 sealed 8-gate**。是 **emitter-maturity 大 recovery**（[GAP-EMIT-VSETVL-TAX] for GEMM·自伤 gcc-15.2 崩塌愈合 6.7×）+ **on-silicon byte-exact 坐实**。
- **到 parity 仍需**攻 density（vl-widen / repack-format wide block-dot·[远期]）——roll 不动 density floor。
- **[K-10] 结构级坐实**：GEMM roll = 独立结构（4-col tile 32-partial spill cliff·v31 满压·roll 减 spill 7.7× 但未清零；GEVM 8-partial clean-fit 不可达）·非 GEVM emit_loop_schedule 参数翻转 → 独立 GEMM plan。
- **scope**：q6_K GEMM 完整（capability-guard·q3_K 同 no-min GEMM dispatch 但走 early-return·**未 wire·frozen unrolled**·同类 deferred）。

## 复现
```
# 实装（tracked·未 commit·禁 git per 任务）
#   lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp  (emitRepackKQuantGemmBodyQ6K + dispatch)
#   lib/Conversion/RVV/RVVToEmitCInternal.h            (signature)
#   test/Conversion/RVV/rvv-to-emitc-repack-gemm-q6-K-q8-K-rolled.mlir  (新 lit)
# forced rebuild:
rm -f build-weft/bin/weft-opt build-weft/lib/Conversion/RVV/CMakeFiles/obj.WeftConversionRVV.dir/RVVToEmitCBlockQuantLinear.cpp.o
ninja -C build-weft weft-opt
# lit: (cd build-weft/test && lit -sv Conversion/RVV)  -> 248/248
# board G2:
bash board/run_q6k_roll_ab.sh    # env.sh gcc-15.2·core 8·byte-exact + 3-way cold prefill
```
raw board 输出见 `board/` (driver/kernels/run script)·objdump + timing 逐行于本文件 §4。
```
Q6KROLL VLEN=128 K=2048 nr=16 nc=512 ... BYTEEXACT memcmp=0 | recovery_roll/unroll_med=6.8669 roll/opp_med=0.2797 unroll/opp_med=0.0407
# Q6_unrolled(.o): vsetvli=6278 spill=2234 reload=2281 vwmacc=2304 v31 299632B
# Q6_rolled(.o):   vsetvli=21   spill=289  reload=324  vwmacc=512  v31 22768B
```
**禁 git commit·byte-exact 硬门全过·default 零漂移·240-fixture 隔离·board disjoint-pin(core8)+load-gate+cleaned no-stray+co-tenant 未扰（loadavg 2.08→2.90 全程我单核）。**
