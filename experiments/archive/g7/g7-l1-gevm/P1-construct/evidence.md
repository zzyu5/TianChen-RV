# G7 L1 P1 — 独立 GEVM Emission Plan 构造 (q4_K@rvv · 首格 · byte-exact 机制里程碑)

> 结构级 · 编译器层 · 本地构造 (emitter/ODS/lit/oracle · 不占板)。P2 板 IPC 验证是后续步。
> 依据 canon: [K-10] 结构级/参数级判据 · [PAT-2] P9 GEVM regime-plan · P0 字节审计 (攻结构不攻布局) · 证书三要件。
> **禁 git commit/add** — 构造留工作树，主会话验证后提交。

## 0. 交付物 (四件齐 · 全本地可验)

| 件 | 路径 | 状态 |
|---|---|---|
| **ODS op** | `include/Weft/Dialect/RVV/IR/RVVOps.td` · `TypedRepackGemvColgroupTiledLoopBodyOp` (+ yield ParentOneOf 拓宽) | ✅ build clean · round-trip verified |
| **verifier** | `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` · `TypedRepackGemvColgroupTiledLoopBodyOp::verify()` | ✅ bounded surface fail-closed |
| **发射器** | `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` · dispatch `emitTypedRepackGemvColgroupTiledLoopBody` + body `emitRepackKQuantGemvColgroupTiledBodyQ4K` | ✅ 全 lower 到 emitc · 0 leftover op/cast |
| **predicate + 注册** | `lib/Conversion/RVV/RVVToEmitC.cpp` · `isTypedRepackGemvColgroupTiledLoopBody` + dispatch 表行 | ✅ |
| **header 声明** | `lib/Conversion/RVV/RVVToEmitCInternal.h` · 三声明 | ✅ |
| **lit golden** | `test/Conversion/RVV/rvv-to-emitc-repack-gevm-colgroup-tiled-q4-K-q8-K.mlir` | ✅ CHECK / NOWALL / ROUNDTRIP 三前缀全过 |
| **host ZERO-MODEL oracle** | `test/Target/RVV/q4-K-q8-K-repack-gevm-colgroup-tiled-oracle.c` | ✅ `ORACLE PASS` (816 列 mismatch=0) |

## 1. 构造决策 (攻结构 · [K-10] 独立 plan)

**P0 已定输入**: 字节触底 roofline (H2 布局税证伪)·真实优化目标 = M=1 带宽→吞吐结构效率 (抬 IPC·削重建·藏装载延迟)。故 **攻结构不攻布局** (沿用 resident repack 布局·不做双布局)。

**现状盘点** (构造前研究):
- 已存在 q4_K GEVM plan (`typed_repack_gemv_loop_body` + `repack_gemv_kquant_core` + `emitRepackKQuantGemvBodyQ4K`),结构 = **列组-OUTER / block-INNER**：activation block base `a+l*292` 在 block 内层算,每个列组都重算一遍 activation → 无跨列组共享 (P0 IPC 0.29 停顿型结构损)。
- 已存在 q4_K GEMM plan (`typed_repack_gemm_loop_body`),M×N tile,M=1 退化即 [K-10] 实证① "GEMM 兼职 GEVM"。

**新 plan 的结构级差异** ([K-10] 三问 · 相对现有 GEVM 与 GEMM 都不同):
1. **迭代空间拓扑变**：列组循环被 `column_group_tile` (TG) **切片**；contraction-block 循环上提为 tile 内的**共享中层**。
2. **数据消费契约变**：ONE q8_K activation block base + delta d_y **每 block 只算一次**,被 tile 内 TG 个列组**共享** (= [PAT-2] "交织组多列共享输入向量加载")。
3. **优化目标 = M=1 结构效率**：TG×numHalves 个 per-strip f32 累加器 = **register-resident bank**,横跨 block 流常驻 (修 P0 的 GEMM-tile 累加器 L1-round-trip 停顿);weight strip **预取节奏进结构** (下一 block 提前一步预取)。

∴ 按 [K-10] 硬禁 (结构级差异禁当 plan 内旋钮),实现为 **独立 typed op · 独立 L2 region · 独立 provenance · 独立 lit/oracle**,而非在现有 GEVM plan 上加 attr 分支。

## 2. byte-exact 三证 (硬门 · int 累加序/dequant 逐字不动 · 只改迭代编排)

**by-construction 论证**：GEVM 输出列是**独立归约**。新 plan 只重排/切片**独立输出列**的访问顺序;对任一固定列,per-block 归约序、8 sub-block/super-half 拆分、split-32 整数点积、bsums-min 校正、dual d/dmin fp16 fold **全部不动**。重排独立列归约不可能改变单列值 ⇒ byte-exact。发射器复用**同一** per-block leaf (`emitBlockFold` = 现有 GEVM body 逐字),仅改 envelope。

**host ZERO-MODEL oracle** (`test/Target/RVV/q4-K-q8-K-repack-gevm-colgroup-tiled-oracle.c`·`cc -O2 -std=c11`·板无关):

```
grid: N in {16,32,64} x nb in {1,2,3} x TG in {1,2,4}   (816 列/项)
(1) tiled-envelope == per-column(GEMM-M=1) : mismatch=0  [BIT-exact]   ← 证 A
(2) tiled-envelope == independent zero-model: mismatch=0  [BIT-exact]   ← 证 B
control naive-associativity  worst_rel = 1.857e-07  (>0: fold 序 load-bearing)
control NOMIN (drop min fold) best_rel  = 2.473e-02  (>>0: min fold 真实非空心)
ORACLE PASS
```

- **证 A (envelope 不变性)**：TG∈{1,2,4} colgroup-tiled envelope 逐 bit 等于 per-column (= GEMM-plan-M=1 算术) envelope。
- **证 B (ZERO-MODEL 独立重算)**：从 raw packed 字节独立物化 logical 数组 (canonical `get_scale_min_k4` + raw nibble·不同 code path·零复用 kernel 中间量) 后按同 fold 序重算 → 逐 bit 等于 tiled。满足证书三要件 (语料完备:正偏置激活令 dmin/min fold 两项都被激发;输入路径同源;oracle 独立)。
- **anti-hollow 控制**：naive 逐元素 dequant-dot (不同 float 结合序) 有 ~1e-7 差异 (证 fold 序 load-bearing);NOMIN (去 min 项) 差 2.47e-2 (证 min fold 真实,非空心表示)。

**范围诚实标注**：oracle 在**逻辑 q4_K super-block 层**机检算术 + envelope 不变性 (native block_q4_K decode 出的 logical sc/m/nibble = repack lane-wise 位解包产出的同一逻辑值)。repacked block_q4_Kx16 **字节布局** + lane-wise emit 的正确性由 **lit golden** 机检 (发射的 C 复用同一 vle8/vand/vsrl/vzext/vwmacc/vfnmsac 叶),两者分工不重叠。

## 3. provenance 差异 (与 GEMM plan 迭代结构条目不同 · [K-10] 机检锚点)

发射的 emitc 中每个 op 都带 provenance 戳 `source_op=weft_rvv.typed_repack_gemv_colgroup_tiled_loop_body` (6367 条注释),归因到**新独立 plan**,与 GEMM plan (`typed_repack_gemm_loop_body`) 迥异。结构级 step 条目 (机检锚点·GEMM plan 不发这些):

| provenance 条目 | 计数 | 含义 (新 plan 独有) |
|---|---|---|
| `colgroup_tile_count` | 1 | 列组 TILE 数 = ncGroups / TG (第三个 div·GEMM/per-column GEVM 都无) |
| `colgroup_tile_base` | 2 (=TG) | tile 内第 cg 列组索引 xt*TG+cg |
| `register_resident_bank_seed` | 2 | 常驻累加器 bank 播种 (4 个 vfmv_v_f = TG×numHalves) |
| `shared_act_block_base` | **1** | activation block base 每 block 只算一次 (per-column plan 是每列组一次) |
| `shared_act_scale_scalar` | **1** | 共享 delta d_y (1 个 `*(const float *)`) |
| `weight_strip_prefetch` | 2 | 预取节奏进结构 (2 个 `__builtin_prefetch`·GEMM/GEVM 都无) |
| `output_addr` | 4 | TG×numHalves = 4 个 lane-wise vse32 store |

对比现有 per-column GEVM plan 的 provenance (`weight_group_base` 在 col-outer·activation base 在 block 内层每列组重算·无 tile_count/bank_seed/prefetch/shared_act 条目) → 迭代结构条目**可机检地不同**。`NOWALL-NOT: redsum` 确认切片未重新引入横向归约墙。

## 4. 构建 / lit 状态

- **weft-opt 构建 clean** (LLVM-20 system·`ninja -C build-weft bin/weft-opt`·增量 ~18s·ODS regen ~50s)。
- **新 lit 三前缀全过** (CHECK 全 lower / NOWALL 无 redsum / ROUNDTRIP parse+verify+print)。
- **零回归**：`lit build-weft/test/Conversion/RVV` = **242/242 PASS** (含现有 q4_K/q5_K/q2_K GEVM + GEMM goldens 不受扰)。
- **全库 lit** = 907/910 PASS。**3 个 FAIL 是预存且无关**：`Scripts/rvv-generated-bundle-abi-e2e-*` (standalone python `scripts/rvv_generated_bundle_abi_e2e.py` 自测在 "product-reduction dequant" 断言处崩溃·**未被我触碰**·不 invoke weft-opt·零引用本 op)。我方触碰集 = 5 源文件 + 2 新测试文件,`scripts/` 未动。

## 5. 触碰文件清单 (工作树·未 commit)

```
 M include/Weft/Dialect/RVV/IR/RVVOps.td                      (+ 新 op def · yield ParentOneOf 拓宽)
 M lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp               (+ 新 op verifier)
 M lib/Conversion/RVV/RVVToEmitCInternal.h                    (+ 3 方法声明)
 M lib/Conversion/RVV/RVVToEmitC.cpp                          (+ predicate + dispatch 表行)
 M lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp          (+ dispatch + body emitter)
?? test/Conversion/RVV/rvv-to-emitc-repack-gevm-colgroup-tiled-q4-K-q8-K.mlir   (lit golden)
?? test/Target/RVV/q4-K-q8-K-repack-gevm-colgroup-tiled-oracle.c                (host ZERO-MODEL oracle)
```

## 6. P2 板验证准备 (后续步 · recipe)

本增量止于 byte-exact 机制里程碑。P2 板 IPC 验证 recipe (P0 成功判据锚):
- **部署**：新 plan emit 的 q4_K@rvv decode kernel 构造 emitc→C→gcc-15 .o,swap 进 sealed .so (engage banner "TCRV EMITTED GEVM-CT ENGAGED")·correctness 沿用既有 byte-exact seal + 本 oracle。
- **口径**：`ssh rvv` openEuler VLEN128·taskset 8-15 disjoint-pin·`cache-misses×64B` DRAM 流量 (进程私有·负载鲁棒)·两点差分 per-token。
- **成功信号** (P0 设计期锚)：同字节下 IPC 0.29→≥0.52·cyc 2.01×→≤1.0× 向 stock roofline 收敛,**字节保持 ≈roofline 不回升** (回升 = 引入 H2 布局税须回退)。
- **M 扫描 → M\* 标定**：M∈{1,8} 扫,M\* 实测写回 schema (禁 `if(M==1)` 硬编码·[K-10] 选择层 = 形状事实∧能力事实)。
- **TG 键控**：column_group_tile 是能力键 (32-vreg budget 约束 register-resident bank 宽度)·per-format 板测定 (性能宪章规则2·禁直觉投影)。

## 7. 一句话结论

G7 核心 novelty 首格交付：q4_K@rvv **独立 GEVM Emission Plan** (列组切片·block 流·共享激活·register-resident bank·预取进结构) 已构造为**独立 ODS op + verifier + 发射器 + lit + host ZERO-MODEL oracle**,byte-exact 由构造 (三证 mismatch=0),provenance 迭代结构条目与 GEMM plan 可机检地不同 ([K-10] 结构级),weft-opt 构建 clean·Conversion/RVV lit 242/242 零回归。补齐 plan 库北极星执行图第①步 (GEVM plan 从"待构造"→"已 mechanized·本地证毕·待 P2 板 IPC")。
