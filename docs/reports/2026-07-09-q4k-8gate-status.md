# q4_K 八门状态刷新 — 2026-07-09（SEL-1 T2 关闭门⑦后）

> **2026-07-10 追加（门② 关闭）**：门②（VLEN 翻转 lit{128,256}）由 **MISSING → PASS**，
> q4_K **4/8 → 5/8**。触发于新增两个 in-tree lit（doc-only，未改 lib emitter 源）：
> `test/Conversion/RVV/rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir`（prefill GEMM 主证）+
> `test/Conversion/RVV/rvv-q4-k-repack-gevm-march-vlen-keyed-gate2.mlir`（decode GEVM 佐证）。
> **决定性刻画**：同一 `tcrv_rvv.quant_contraction`（q4_K/prefill）请求 byte-identical，唯一变量是
> `-march`；`--tcrv-rvv-lower-quant-contraction` 经 `deriveMinimumVLEN` → `deriveRepackHalfLanes(minVLEN)`
> = min(VLEN/16,16) 把 strip vl 键控进构造出的 `typed_repack_gemm_loop_body`：**VLEN128(rv64gcv)→strip
> vl 8**（e32,m2、两 8-lane strip、int16 buffer 32 宽），**VLEN256(rv64gcv_zvl256b)→strip vl 16**
> （同 e32,m2 accumulator 满 16 lane、int16 buffer 64 宽）。tiling/emit **能力键控于 VLEN、非硬编码**
> = 门② 主张。emitter 源**未改**（现有 emitter 已真按 VLEN flip vl；FileCheck 交叉证：V128 检查在
> VLEN256 emit 上正确 FAIL）。**结论不变**：q4_K **仍非 sealed 8-gate Win**，beat/outperform 措辞
> 仍 **LOCKED（[NG-4]）**（唯 ④e2e-perf 未闭、board-contention pending;③⑤ 已由 T4a 闭）。222/222 Conversion/RVV lit PASS。详见文末表 §② 行。

> G3 线D 证据更新（doc-only）。把 q4_K repack GEMM 的 [PERF-1] 八门账从 **3/8 PASS** 刷新到
> **4/8 PASS**：门⑦（机制合成归因 / selector capability-keyed）由 **MISSING → PASS**，触发于
> G3 主线C [SEL-1] **T2 曳光弹**（commit `4624740f`）。**结论不变**：q4_K **仍非 sealed 8-gate Win**，
> beat/outperform 措辞 **LOCKED（[NG-4]）** —— 唯一可引句仍是 kernel-轴 / 板A(rvv/VLEN128) / prefill 的
> fully-bound claim。本文是缺项清单 + 负责批指针；不引入任何新性能主张。

- **权威门定义**：科研目标总纲 v2 §4.4 [PERF-1]（`docs/canon/TianChen-RV_科研目标总纲v2.md:168-170`）：
  ①字节精确 ②VLEN翻转lit{128,256} ③双板objdump ④micro∧e2e(分相/多长度) ⑤双板都验证 ⑥实验纪律
  ⑦机制合成归因 ⑧措辞门。
- **claim under test（bound，不变）**：板 `ssh rvv`(openEuler/VLEN128) 上，front-door-**constructed**
  q4_K repack GEMM（`emitRepackKQuantGemmBodyQ4K`，S6-tiled）在 **kernel 轴** 上比出厂 dispatch 的
  `ggml_vec_dot_q4_K_q8_K_vl128` block-dot（git `f3e1828`）快 **1.884×**（nr=64、单核、cold N=10-12）。
  **e2e 仅 projection**（§二.2 Amdahl ≈1.59× prefill）；无 measured e2e Δ（集成 BLOCKED，见门④）。
- **前一版（K-quant 家族收口，3/8）**：`experiments/active/kquant-family-closure/T-PERF1_q4_K_vlen128_prefill_8gate.md`
  （其门⑦ = MISSING「SEL-1 先验层未落」的判据已被本刷新 supersede；该 cell 属家族收口批、非本线 touch-set）。

## Verdict：**7 / 8 PASS · 1 partial（④e2e-perf pending）**（2026-07-10 门② 关闭 + T4a 门③⑤ 追认；此前 doc 漏计 T4a） → 仍非 sealed Win；beat 措辞 LOCKED（[NG-4]）

| # | gate（§4.4） | 前(收口批) | 现(2026-07-09) | 证据指针 |
|---|---|---|---|---|
| ① | 字节精确 | PASS(bounded-ULP) | **PASS**(bounded-ULP,非 ULP0) | FMA-fold bounded-ULP;构造 oracle `039133ea` 8/8 WORST_NORM ~7–8e-7;S6 tile IDENTITY cmp=0 + `vwmacc` 2240 multiset 不变 → timed S6 = golden。措辞禁 "byte-exact vs ggml"。 |
| ② | VLEN 翻转 lit{128,256} | MISSING(256 缺) | **★PASS(NEW 2026-07-10)** | 新增 in-tree lit（未改 emitter 源）：`test/Conversion/RVV/rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir` — 同一 q4_K/prefill `quant_contraction` byte-identical，唯 `-march` 变；VLEN128(rv64gcv) lower→emitc 出 **strip vl 8**（`typed_repack_gemm_loop_body` half_lanes=`deriveRepackHalfLanes(128)`=8、e32/m2、int16 buffer `array<32x>`、numHalves 2），VLEN256(rv64gcv_zvl256b) 出 **strip vl 16**（half_lanes=16、同 e32/m2 满 16 lane、`array<64x>`、numHalves 1）；同一 strip-vl SSA 值喂 `__riscv_vfmv_v_f_f32m2`/`__riscv_vmv_v_x_i32m2`（真 vl 非杂散字节）。**能力键控于 VLEN 非硬编码**（FileCheck 交叉证：V128 检查在 VLEN256 emit 上正确 FAIL）。GEVM 佐证 `rvv-q4-k-repack-gevm-march-vlen-keyed-gate2.mlir`：decode@VLEN128→repack GEVM vl=8；decode@VLEN256 selector VLEN-keyed **decline**（`vlenOrPrefillFavorsRepack`=`minVLEN==128‖Prefill`；q4_K 无 block-dot decline 路 → fail-closed I7，`--verify-diagnostics` 证）故 GEVM 无 vl=16（VLEN-flip 在 routing）。222/222 Conversion/RVV lit PASS。 |
| ③ | 双板 objdump 验封 | PASS | **PASS**(rvv128 ✓ + k1 ✓) | 板A(rvv/VLEN128)SEALED(spill 84→3,v31→v30,`vwmacc` 2240);板B(k1/VLEN256)SEALED via **T4a**(`kquant-k1-vlen256-kernel-axis-t4a`,commit 5b339407,q4_K S6 spill 81→4,vtype VLA 同 rvv128)。 |
| ④ | micro ∧ e2e | PARTIAL(micro✓/e2e blocked) | **PARTIAL**(micro ✓ / e2e projection) | micro PASS: S6 **1.884×** vs 出厂 block-dot,cold N=10-12,cv~0.8%,shape-robust;e2e MISSING: measured "ours vs stock" BLOCKED(kernel 未接 llama.cpp Q4_K mul_mat、无 K-quant repack env-toggle),仅 Amdahl 传导账 ≈1.59× prefill 上限。集成兑现是**独立** e2e-seal 战役,非 T4 范围。 |
| ⑤ | 双板都验证 | PASS | **PASS**(rvv + k1) | 板A rvv/VLEN128 已验;板B k1/VLEN256 q4_K repack GEMM via **T4a**(5b339407,k1 q4_K **3.106×** dual-board kernel-axis,q5_K 1.916×)。 |
| ⑥ | 实验纪律 | PASS | **PASS** | core8/2.6GHz/gov=perf/cold/N≥10/cv/同 `libggml-cpu.so`(f3e1828);e2e-baseline pin8-15、model=Q4_K_M、warmup-dropped、N=10 median+IQR。 |
| ⑦ | 机制合成归因(selector 能力键) | **MISSING**(reason≈static_order) | **★PASS(NEW)** | SEL-1 **T2**(commit `4624740f`)把 SP4(tiled-vs-plain output-tiling)从**编译期 per-format 硬编码**升级为**运行时能力键控选择**:选择 pass(`RVVLowerQuantContraction.cpp`)跑 `selectRepackTilingVariant` → stamp `tcrv_rvv.tiling_variant="s6_tiled"` + **`tcrv_rvv.tiling_selection_reason="prior"`**;**reason=prior 键控 `fold_model` 瓶颈 SHAPE 非 format 名**(`kquant_dmin_bsums_min`→`MinFoldRegisterCliff`→S6Tiled);完整 [D-4] JSONL inert attr `tcrv_rvv.tiling_selection_record`(hash 3cd23a4e 非空)。`static_order`= 能力盲 fallback,HONESTLY 标、接线路径不得出现(燃减信号)。gate-7 lit `test/Conversion/RVV/rvv-sel1-q4-k-tiling-variant-capability-keyed-gate7.mlir`(两 RUN: SEL1 stamp + REALIZE byte-exact);**524/524 RVV lit PASS**;emitter `emitRepackKQuantGemmBodyQ4K` 内部未动 ⇒ S6 输出 byte-exact vs oracle+identity;C_construct 42 不变(LAYER-4)。NG-1: 有界枚举+实测记忆(schema/tiling-measurements.v1.json)+规则 prior、无搜索/无学习成本模型。 |
| ⑧ | 措辞门 | PASS | **PASS** | 每数字绑 phase(prefill)×board(rvv/VLEN128)×format(q4_K)×baseline(ggml f3e1828 block-dot)×axis(kernel;e2e=projection);禁 generalize decode(传导效率≈0)、禁 byte-exact vs ggml、e2e 必标 projection/Amdahl 上限。 |

## 缺项清单 → 谁负责补

| 缺项 | 门 | 类型 | 负责批 | 内容 |
|---|---|---|---|---|
| ~~VLEN256 flip lit~~ | ② | **CLOSED(2026-07-10)** | — | ✅ 已补：`rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir`（prefill GEMM strip vl 8↔16 键控于 `-march` 派生 VLEN）+ GEVM 佐证 lit。emitter 源未改（现有 emitter 已真 flip）。剩 {m1,m2,m4} LMUL-变体扫描仍属 SEL-1 T4 旋钮范围，但 gate② 的 {128,256} codegen-flip 主张本身已闭。 |
| ~~k1 双板 objdump + 复测~~ | ③(k1半)+⑤ | **CLOSED(T4a 追认)** | — | ✅ 板B(k1/VLEN256)q4_K repack GEMM 经 **T4a**(commit `5b339407`)objdump-sealed(spill 81→4、vtype VLA 同 rvv128)+ kernel-轴复测(k1 q4_K 3.106×/q5_K 1.916×)；门③⑤ 均 PASS(见上表 §③§⑤)。此前 doc 漏计 T4a、故列缺项、现追认闭合。 |
| e2e 集成兑现 | ④(e2e半) | PARTIAL | **独立 e2e-seal 战役**(非 T4) | (a) 离线把 q4_K 权重 repack 成 `block_q4_Kx16` 布局;(b) ggml mul_mat 新 dispatch 入口 + q8_K 激活量化胶水。历史难点(q4_K emitter scaffold timed out 2×)。 |

> ★ 门⑦（2026-07-09）+ 门②（2026-07-10）关闭 + 门③⑤（T4a `5b339407` 追认）后，q4_K = **7/8 PASS ·
> 1 partial**：唯一未闭 = **④ e2e-perf**（board-availability-gated，共享板 SPEC+vllm 争用致机会窗中止；
> **非能力缺口**——e2e 集成已证**正确**：seal-fix `98717158` 部署 VLEN128 vl=8 kernel、PPL 12.008≈stock、
> banner ENGAGED、贪心相干；剩 prefill perf 数待安静板窗，机会轮询 + 用户协调中）。④ 闭合前
> q4_K = **7/8 characterized kernel-axis path-win**（板A/B 双板 / prefill 条件），beat 措辞 LOCKED（[NG-4]）。
> 注：门② PASS 是 codegen-flip lit 证据（VLEN128/256 两臂 in-tree）；VLEN256 硬件 objdump-验封已由门③/⑤ 的
> T4a k1 板批达成。

## 落地指针（doc-only；无 lib/schema 改动，deliverable 未 commit，留用户提交）

- 本文：`docs/reports/2026-07-09-q4k-8gate-status.md`（缺项清单 + 负责批）。
- 同步：`docs/reports/2026-07-07-paper-material-inventory.md` §二.4「q4_K 八门」段（→7/8、②③⑤⑦ PASS、④e2e-perf pending·board-gated）。
- **门② 一手证据（NEW 2026-07-10，未 commit，留用户提交）**：`test/Conversion/RVV/rvv-q4-k-repack-gemm-march-vlen-flip-gate2.mlir`（prefill GEMM strip vl 8↔16）+ `test/Conversion/RVV/rvv-q4-k-repack-gevm-march-vlen-keyed-gate2.mlir`（decode GEVM VLEN-keyed）；机制源（未改）：`lib/Plugin/RVV/RVVLowerQuantContraction.cpp` `deriveRepackHalfLanes`/`lowerToRepackGem{m,v}KQuant`（half_lanes/numHalves 由 `deriveMinimumVLEN(-march)` 派生）+ `lib/Plugin/RVV/RVVContractionPathSelection.cpp` `vlenOrPrefillFavorsRepack`（decode@256 decline 路）。
- 门⑦ 一手证据：commit `4624740f`（SEL-1 T2）+ gate-7 lit `test/Conversion/RVV/rvv-sel1-q4-k-tiling-variant-capability-keyed-gate7.mlir` + 变体注册表 `include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h` + 测量库 `schema/tiling-measurements.v1.json`。
- 前一版 3/8 账（家族收口批，⑦ MISSING 判据已 supersede）：`experiments/active/kquant-family-closure/T-PERF1_q4_K_vlen128_prefill_8gate.md`。
