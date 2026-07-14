# T-CENSUS — 全量 kernel 冷启动普查队列（G7 终编成令三·第一段·案头供弹）

> **建成**：2026-07-14 · L2 例外池账本级复审 + L1 普查队列构建线（纯 read-only 账本分析·无板·无源码改动）。
> **性质**：**第二赛道 kernel-sym 冷启动普查弹药表**（与 perf-covered 系统账 9/83 **永不混算**）。本表 = 案头先行产物，给主会话直接喂 L1 普查执行。
> **来源基**：`schema/perf-covered-category.v1.json`（83 格·recon 权威）+ `T9_kernel_sym_ledger.md`（kernel-sym ≥parity=12）+ 逐 emitter/opponent 符号级机判（见 §0.3）。

---

## 0. 头（体例·账本锁·对手探针）

### 0.1 一格两账并行（终编成令二·体例四）
- **e2e 分类不动**：声明例外的 e2e 状态（category=声明例外·amdahl_ceiling）**保留原样**，本表不改 schema。
- **新增 kernel-sym 状态列**：与热路径占比**无关**（Amdahl 是 e2e 门·非 micro 门）。kernel-sym 资格 = 三条件全满足：
  1. **我方 emitted kernel 在**（真 emitted 核·非仅占位）；
  2. **对手原生 kernel 在**（该板 as-shipped 真出货 kernel·符号级探针可解析）；
  3. **对称编译可行**（我方与对手可同编译器/flags/march 编·kernel 账有效）。

### 0.2 板×编译器锁（[CASE-COMPILER-ASYMMETRY] 判别键）
- **k1 = clang-18 对称域** / **rvv = gcc-15.2 对称域**（board shipped-compiler）。跨板不可比·逐点标 board identity。

### 0.3 对手探针机判（符号级·q4_K 稻草人永久前车）
- **对手 baseline = 该板 as-shipped 真出货 kernel·符号级机判**（禁手写类目/挑弱对手/internal-A/B）。
- **★对手树实证（2026-07-14 本线新查·纠正前案头假设）**：部署 opponent 树 = `/home/kingdom/phdworks/llama.cpp`（remote **ggml-org/llama.cpp**·genuine upstream）。逐符号 grep 实证 as-shipped 对手**存在性**：
  - `ggml-cpu/quants.c` 有 `ggml_vec_dot_{q2_K,q3_K,q4_K,q5_K,q6_K,iq1_s,iq1_m,iq4_nl,iq4_xs,mxfp4,nvfp4,q1_0,tq1_0,tq2_0}_q8*_generic` — **全部 vec_dot 对手存在**。
  - `ggml-quants.c` 有 `dequantize_row_{全 19 格含 mxfp4/nvfp4/q1_0}` — **全部 dequant 对手存在**。
  - **★★纠正**：前案头（`kernel-sym-台账-candidates.md` §3·`未接线-22格-分诊.md` §5.3）判 **q1_0/nvfp4 = no-fair-opponent（opponent-absent）** ← 基于标准 upstream ggml。**本树实测：`GGML_TYPE_MXFP4=39 / NVFP4=40 / Q1_0=41` 均在类型枚举内 + `_generic` 参考实现存在**。→ nvfp4/mxfp4 = genuine upstream 对手（有效）；**q1_0 = 枚举 41·canon-memory 标 "Weft-internal binary·非标准 ggml"·provenance 存疑**（`_generic` 参考可能 fork/Weft 添加=internal-A/B 风险）→ **flag 主会核·leans 单列不计数**。
- **上游更强未启用路径披露**：K-quant@rvv 对手 = factory block-dot（generic·非 hand-brick）；**q4_K@k1 唯一真 hand-brick**（`ggml_gemm_q4_K_16x1_q8_K`·case256）。iq/tq gemm-轴对手 = **absent**（零 iq/tq repack GEMM·仅 block-dot vec_dot·跨-op）。

---

## 一、49 声明例外逐格 kernel-sym 资格判

> 判据 = §0.1 三条件全满足。逐格 {✅入队 | ❌不入队+缺条件}。emit 指针 = 逐 emitter 文件机判（见列）。

### 一.A GEMM iq/tq @rvv（7 格）— ❌ 缺条件② same-op opponent-absent（cross-op 可选）

| 格 | ①我方 emit | ②对手 | ③对称 | 判 |
|---|---|---|---|---|
| gemm_tile/iq2_xxs@rvv | ✓ 前门 grid repack GEMM | **✗ same-op absent**（零 iq repack GEMM·仅 block-dot vec_dot） | ✓ gcc-15.2 | ❌ 不入队（缺②same-op）·cross-op 可选（our-gemm vs opp-block-dot·须明标·如 iq4_nl 先例） |
| gemm_tile/iq2_xs@rvv | ✓ | ✗ same-op absent | ✓ | ❌（同上） |
| gemm_tile/iq2_s@rvv | ✓ | ✗ same-op absent | ✓ | ❌（同上） |
| gemm_tile/iq4_xs@rvv | ✓ 前门 codebook repack GEMM | ✗ same-op absent | ✓ | ❌（同上·结构-NULL S6 no-op） |
| gemm_tile/mxfp4@rvv | ✓ PLAIN/UNTILED | ✗ same-op absent | ✓ | ❌（同上·tiny-codebook 无杠杆） |
| gemm_tile/tq1_0@rvv | ✓ ternary repack GEMM | ✗ same-op absent | ✓ | ❌（同上） |
| gemm_tile/tq2_0@rvv | ✓ ternary repack GEMM | ✗ same-op absent | ✓ | ❌（同上） |

**小结**：7 格全 ❌（缺条件②·same-op opponent-absent）。**cross-op 口径**（our-gemm vs ggml-block-dot·iq4_nl 先例 0.217×/0.505× LOSS）主会话若采可入队（须明标 cross-op·区别 same-op kernel-sym）→ 列入 §三 Batch-3 as cross-op-flagged。

### 一.B FLAT vec_dot（5 格）— ✅ 入队·成色=parity-by-adoption

| 格 | ①我方 emit | ②对手 | ③对称 | 判 |
|---|---|---|---|---|
| vec_dot/q4_0 | ✓ emitQ4_0Q8_0BlockDot | ✓ ggml_vec_dot_q4_0_q8_0（light-vec） | ✓ | ✅ 入队（parity-by-adoption·发射 ggml 自身 block-dot→构造性 parity·**成色 tautological·非独立 beat 空间**） |
| vec_dot/q4_1 | ✓ | ✓ q4_1_q8_1（light-vec） | ✓ | ✅（同上） |
| vec_dot/q5_0 | ✓ | ✓ q5_0_q8_0（better-vec） | ✓ | ✅（同上） |
| vec_dot/q5_1 | ✓ | ✓ q5_1_q8_1（better-vec） | ✓ | ✅（同上） |
| vec_dot/q8_0 | ✓ | ✓ q8_0_q8_0（light-vec·上游 VLEN128 破损） | ✓ | ✅（同上·stale micro parity VLEN256 1.04×） |

**小结**：5 格全 ✅ 入队。**成色 = parity-by-adoption**（我方发射 ggml 自身 block-dot 指令·A/B ≈1.0× by construction）。**note**：同 format 的 **repack-GEMM** 已在 T9 §1.1 kernel-sym ≥parity（q4_0/q4_1/q5_0/q5_1/q8_0@rvv·真 beat 1.2–6.8×）——本 vec_dot 格 = 独立 block-dot 路的 parity 确认·**不产 beat·低值·补全 4 列即可**。

### 一.C K-quant vec_dot（5 格）— ✅ 入队·ours-block-dot vs ggml-block-dot

| 格 | ①我方 emit | ②对手 | ③对称 | 判 |
|---|---|---|---|---|
| vec_dot/q2_K | ✓ KQuant aux32 integer-core | ✓ ggml_vec_dot_q2_K_q8_K_generic | ✓ | ✅ 入队（block-dot·弱—中对手·数值档 ZERO-MODEL INT bit-exact） |
| vec_dot/q3_K | ✓ Q3_K aux32 core | ✓ q3_K_q8_K_generic | ✓ | ✅（同上·weight-recon floor 强先验 LOSS） |
| vec_dot/q4_K | ✓ Q4_K aux32 core | ✓ q4_K_q8_K_generic | ✓ | ✅（同上·decode 贡献另在 gemm@k1 记账·此为独立 vec_dot 路） |
| vec_dot/q5_K | ✓ Q5_K core | ✓ q5_K_q8_K_generic | ✓ | ✅（同上） |
| vec_dot/q6_K | ✓ Q6_K aux32 core | ✓ q6_K_q8_K_generic | ✓ | ✅（同上·2995 vsetivli weight-recon floor 强先验 LOSS） |

**小结**：5 格全 ✅ 入队。对手 = ggml generic block-dot（弱—中）。**预判**：weight-reconstruction floor 强先验 → 多数 <parity（同 gemm@rvv 0.18–0.39×）。数值档 = INT bit-exact（ZERO-MODEL）。

### 一.D IQ/binary/fp4 vec_dot（5 格）— ✅×4 入队 + q1_0 flag

| 格 | ①我方 emit | ②对手 | ③对称 | 判 |
|---|---|---|---|---|
| vec_dot/iq1_s | ✓ iq1_s_q8_k_grid_core（前门） | ✓ ggml_vec_dot_iq1_s_q8_K_generic | ✓ | ✅ 入队（grid gather·gather-trap 强先验 LOSS·数值档 INT bit-exact） |
| vec_dot/iq1_m | ✓ iq1_m_q8_k_grid_core（前门·C2 复用 iq1_s） | ✓ iq1_m_q8_K_generic | ✓ | ✅（同上） |
| vec_dot/iq4_nl | ✓ iq4_nl codebook core | ✓ iq4_nl_q8_0_generic | ✓ | ✅（codebook gather·**vec_dot 路未测**·区别已测 iq4_nl gemm 0.217×） |
| vec_dot/nvfp4 | ✓ emitNVFP4BlockDotBodyShared | ✓ **nvfp4_q8_0_generic（存在·纠正前案头"absent"）** | ✓ | ✅ 入队（★对手实证存在·upstream 枚举 40·fp4 codebook·罕见格） |
| vec_dot/q1_0 | ✓ q1_0 FLAT block_q8_0 core | ⚠ q1_0_q8_0_generic 存在但 **provenance 存疑**（枚举 41·canon "Weft-internal"·internal-A/B 风险） | ✓ | ⚠ **flag 主会核**（对手符号存在→leans ✅·但 internal-A/B 风险→leans 单列）·暂入 §三 Batch-3 provenance-pending |

**小结**：iq1_s/iq1_m/iq4_nl/nvfp4 = ✅ 入队（4 格）；q1_0 = ⚠ flag。

### 一.E dequantize_row（19 格）— ✅×17 入队（streaming·数值档 ULP·低值）+ q1_0 flag

> 条件①注记：dequant OP 格的 emitted 核 = dequant 本身（`emitDequantizeRow{KQuant/IQGrid/CodebookGrid/Nibble/Q*}BodyShared` 全在）。**★条件① "非仅 dequant/占位" 边界**：dequant OP 格本体 = 真 emitted 流式核（非占位）→过①；但**这是 dequant-轴非 matmul-轴**·主会话可选择将纯-dequant 格归入**独立 dequant-轴子账**·不进 matmul ≥parity headline 计数（体例决定·本表标 [DEQ-AXIS]）。

| 格族 | 我方 emit | 对手 as-shipped | 判 |
|---|---|---|---|
| dequant/{q2_K,q3_K,q4_K,q5_K,q6_K}（5） | ✓ emitDequantizeRowKQuant | ✓ dequantize_row_q*_K | ✅ 入队 [DEQ-AXIS]（mem-bound·数值档 f32 ULP≈0·parity-expected @roofline） |
| dequant/{iq1_s,iq1_m,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs}（9） | ✓ emitDequantizeRowIQGrid/Codebook | ✓ dequantize_row_iq* | ✅ 入队 [DEQ-AXIS]（gather-trap·数值档 ULP≈0·parity/LOSS-expected） |
| dequant/mxfp4（1） | ✓ emitDequantizeRowCodebookGrid | ✓ dequantize_row_mxfp4（upstream 39） | ✅ 入队 [DEQ-AXIS] |
| dequant/nvfp4（1） | ✓ | ✓ **dequantize_row_nvfp4（存在·纠正 "absent"·upstream 40）** | ✅ 入队 [DEQ-AXIS] |
| dequant/{tq1_0,tq2_0}（2） | ✓ ternary dequant | ✓ dequantize_row_tq* | ✅ 入队 [DEQ-AXIS] |
| dequant/q1_0（1） | ✓ | ⚠ dequantize_row_q1_0 存在·provenance 存疑（枚举 41） | ⚠ flag（同 vec_dot/q1_0） |

**小结**：**18 格 ✅ 入队 [DEQ-AXIS]**（5 K-quant + 9 iq + mxfp4 + nvfp4 + 2 tq·streaming·数值档 ULP·parity-expected·低值·可批量流式补测）；q1_0 = ⚠ flag。

### 一.F product_reduce（3 格）— ❌ 缺条件② 无框架对手（internal sub-primitive）

| 格 | ①我方 emit | ②对手 | ③对称 | 判 |
|---|---|---|---|---|
| product_reduce/q4_0_nibble | ✓ 内部子原语 | **✗ 无框架 kernel 对手**（ggml 无 standalone product_reduce·vs-scalar 10.8× 是 sanity 层非 framework） | — | ❌ 不入队（缺②·结构无框架对手·perf 由复合格 dequant/vec_dot 捕获） |
| product_reduce/offset_binary_n3 | ✓ | ✗ 无框架对手 | — | ❌（同上） |
| product_reduce/codebook_n3 | ✓ | ✗ 无框架对手 | — | ❌（同上） |

### 一.G B-类前向算子（5 声明例外）→ 见 §二 专项（softmax/rms_norm/rope/add/mul）

---

## 二、★B 类前向算子专项（9 算子逐个·softmax 必列）

> 用户点名·softmax 必入。逐个查 ① 我方 emit（grep lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp）② 对手 as-shipped ③ 对称。**全 f32·浮点算子标数值档 ULP 界**。**e2e 分类保留**（gelu/silu/scale/cpy=黄-物理墙·add/mul/rms_norm/softmax/rope=声明例外）·**新增 kernel-sym 列**。

| 算子 | ①我方 emit（函数指针） | ②对手 as-shipped | ③对称 | kernel-sym 判 | 数值档 ULP |
|---|---|---|---|---|---|
| **softmax** | ✓ `emitElementwiseSoftMaxReduceStrip`（+ vexpf·reduce-model·ln 953+） | ✓ ggml_compute_forward_soft_max（native reduce+expf） | ✓ f32 | **✅ 入队（softmax 必测·我方在）** | ULP-bounded（expf·max-sub 数值稳定） |
| **rms_norm** | ✓ `emitElementwiseRmsNormReduceStrip`（Σx² reduce-core） | ✓ ggml_compute_forward_rms_norm | ✓ | ✅ 入队 | ULP-bounded（rsqrt/Σx² 归约序） |
| **rope** | ✓ `emitElementwiseRopeRotateStrip`（rotate-core·rope_norm） | ✓ ggml_compute_forward_rope | ✓ | ✅ 入队 | ULP-bounded（sin/cos） |
| **silu** | ✓ `emitElementwiseSiluMapStrip`（+ vexpf·two-buffer map） | ✓ ggml_vec_silu_f32 | ✓ | ✅ 入队 | ULP-bounded（sigmoid/expf） |
| **gelu** | ✓ `emitElementwiseGeluMapStrip` / `emitForwardGeluScalarLoop` | ✓ ggml_vec_gelu_f32（LUT-deploy caveat） | ✓ | ✅ 入队 | maxrel 1.3e-6（tanhf·JSON 物理墙实测锚） |
| **add** | ✓ `emitElementwiseBinaryMapStrip`（binary map z=x+y） | ✓ ggml_compute_forward_add / ggml_vec_add_f32 | ✓ | ✅ 入队 | **bit-exact**（IEEE f32 加·0 ULP） |
| **mul** | ✓ `emitElementwiseBinaryMapStrip`（binary map z=x*y） | ✓ ggml_compute_forward_mul / ggml_vec_mul_f32 | ✓ | ✅ 入队 | **bit-exact**（IEEE f32 乘·0 ULP） |
| **scale** | ✓ scale map（`emitForwardVecMapStrip`·y*=v in-place） | ✓ ggml_vec_scale_f32 | ✓ | ✅ 入队 | **bit-exact**（0 ULP） |
| **cpy** | ✓ `emitElementwiseCopyMapStrip`（y=x copy map） | ✓ ggml_compute_forward_dup / cpy | ✓ | ✅ 入队 | **bit-exact**（0 ULP·纯拷贝） |

**★softmax 状态**：**我方 emit 在**（`emitElementwiseSoftMaxReduceStrip`·`isTypedElementwiseSoftMaxReduceLoopBody` 识别·`ElementwiseSoftMaxReduceCoreOp` brick）→ **✅ 入队**（非"缺则列首个构造"）。
**B 类小结**：**9/9 全 ✅ 入队**（全 emitted·全 as-shipped 对手存在·全 f32 对称）。**从未入测**（ROADMAP 明标 B 类前向算子从未入测）= 净新普查territory。**预判对手类 = 流式原生 near-wall memory-bound**（add/mul 实测 deploy 0.76–0.83×·near-wall 6.4–7.3 GB/s）→ 多数 parity/near-wall·softmax/rope 冷路轻算·成色低但**账面净新**。

---

## 三、L1 普查队列（按族分批·双板 rvv/k1·四列 {hot,cold,对手类,胜负} 待填）

> 每格预填 {格@板·族·我方 emitted 指针·预判对手类[符号级机判]·浮点算子数值档 ULP}。四列待普查填。测量协议 = micro A/B·同编译器/flags/march·N≥10·T-N·冷态·同形状类（GEVM 测 M=1 / GEMM 测 prefill）·顺手加 M=8 shape 点（第三段攒 regime）。
> **组成**：(甲) 现有未测格（T9 缺列的 kernel-sym 格）+ (乙) L2 新入列格（§一/§二 判入队）。

### Batch 1 — K-quant@rvv 补齐（★首批·可立即喂 L1）

> 现有 hot 已测 <parity（T9 §1.2）·**cold 列缺**（§6.2 明标 "余 4 = q5_K/q2_K/q3_K/q6_K@rvv gemm 另 batch"）。opponent 已知（factory block-dot·gcc-15.2 对称）·emit 在 → 零阻塞。

| 格@板 | 族 | 我方 emit 指针 | 预判对手类（符号机判） | hot | cold | 对手类 | 胜负 |
|---|---|---|---|:--:|:--:|:--:|:--:|
| q5_K@rvv gemm | K-quant | 前门 KQuant repack GEMM | factory generic block-dot（弱） | 0.120×〔T8〕 | ☐ | ☐ | ☐ |
| q2_K@rvv gemm | K-quant | 前门 KQuant repack（dual d/dmin·S6-tiled） | factory hand-tuned _vl128 block-dot | 0.386× | ☐ | ☐ | ☐ |
| q3_K@rvv gemm | K-quant | 前门 KQuant（3-bit hmask·PLAIN·S6 NULL） | factory hand-tuned _vl128 block-dot | ~0.18× | ☐ | ☐ | ☐ |
| q6_K@rvv gemm | K-quant | 前门 KQuant（6-bit dual-plane·PLAIN·S6 NULL） | factory mature block-dot | ~0.18× | ☐ | ☐ | ☐ |
| q4_K@rvv gemm | K-quant | 前门 KQuant S6-tiled（Win-K1-VLEN 的 rvv-half） | factory block-dot（[GAP-Q4K-VLEN128] e2e 0.334×） | ☐ | ☐ | ☐ | ☐ |

**首批建议（item ⑥）**：上表 5 格（q5_K/q2_K/q3_K/q6_K/q4_K@rvv gemm）cold 双态 + nr{4,16,64} 穷举 + M=8 shape 点。全 emit 在·对手 = as-shipped block-dot（机判·gcc-15.2 对称）→ **可立即喂 L1**。预判全 <parity（weight-recon floor·T8 confirmed LOSS）·四出口预备 = 对手结构优势具名（weight-bound）。

### Batch 2 — B 类前向算子（净新·从未入测·双板）

> §二 判 9/9 入队。全 f32 emitted·对手 native。ROADMAP 明标从未入测。
> **★ k1-half DONE（2026-07-14·L1 B类前向线）**：9 ops × 8 shapes 测毕·casefile `experiments/active/g7-census/bclass-forward-ops/`（evidence.md + summary_kernel_sym_k1.csv + raw/k1_run.log）。下表 hot/cold = **k1 anchor n=4096 ratio ours/opp**（>1 ours 快）·对手类=符号级机判·**rvv-half PENDING**（rvv 忙 Batch1·harness 复用）。

| 格@板 | 族 | 我方 emit 指针 | 对手类 [机判] | 数值档 [实测] | hot(k1) | cold(k1) | 胜负(cold) | rvv |
|---|---|---|---|---|:--:|:--:|:--:|:--:|
| softmax@k1 ★ | B-fwd | emitElementwiseSoftMaxReduceStrip | native-RVV m2·EXPORTED .so·bit-identical(0ULP) | ULP 3·0ULP-vs-opp | 0.821 | 0.822 | LOSS(parity-by-adoption −18%) | ☐pending |
| rms_norm@k1 | B-fwd | emitElementwiseRmsNormReduceStrip | native-RVV m8 scale·**2-pass**(memcpy+scale) | **0 ULP** bit-exact | 1.177 | **1.335** | **WIN**(1-pass fusion) | ☐pending |
| rope@k1 | B-fwd | emitElementwiseRopeRotateStrip | autovec-RVV+scalar sin/cos·2-pass cache | 0ULP-vs-opp(f32-θ drift) | 0.981 | 0.984 | PARITY 0.98 | ☐pending |
| silu@k1 | B-fwd | emitElementwiseSiluMapStrip | native-RVV m2·EXPORTED .so·bit-identical | ULP 2·0ULP-vs-opp | 0.841 | 0.837 | LOSS(parity-by-adoption −16%) | ☐pending |
| gelu@k1 | B-fwd | emitElementwiseGeluMapStrip | **scalar f16-LUT**(GGML_GELU_FP16) | maxrel 3.7e-7 ≤1.3e-6 | 0.239 | 0.247 | LOSS(STRUCTURAL LUT·ours +2500×acc) | ☐pending |
| add@k1 | B-fwd | emitElementwiseBinaryMapStrip | autovec-RVV **m2**(clang) | **0 ULP** bit-exact | 1.173 | **1.182** | **WIN**(our m8 vs opp m2) | ☐pending |
| mul@k1 | B-fwd | emitElementwiseBinaryMapStrip | autovec-RVV **m2**(clang) | **0 ULP** bit-exact | 1.191 | **1.184** | **WIN**(our m8 vs opp m2) | ☐pending |
| scale@k1 | B-fwd | emitForwardVecMapStrip(scale) | native-RVV m8·**byte-identical algo** | **0 ULP** bit-exact | 0.918 | 0.935 | LOSS 0.93(scheduling) | ☐pending |
| cpy@k1 | B-fwd | emitElementwiseCopyMapStrip | autovec-RVV/memcpy | **0 ULP** bit-exact | 0.881 | 1.103 | WIN cold/LOSS hot(flip) | ☐pending |

**k1 anchor cold tally**：4 WIN（add·mul·cpy·rms_norm）· 1 PARITY（rope）· 4 LOSS（scale·silu·softmax·gelu）。
**★预判校准**：census「多数 parity/near-wall」= **部分证伪**。① 4 纯 elementwise（add/mul/cpy/scale）确 memory-bound（cold 2.3–5.6 GB/s），但**非齐一 parity**——我方 **wide-m8** emit 击败 clang **m2** autovec（add/mul cold ~1.18×·objdump 实证 `vsetvli e32,m8` vs `e32,m2`）。② **rms_norm 1.33× cold = 结构 WIN**（我方 1-pass read-scale-write vs ggml 2-pass memcpy+scale）·净新亮点。③ compute-bound 簇（silu/softmax/gelu/rope）cache-invariant·silu/softmax parity-by-adoption 但输 16–18% 调度·gelu 0.25× 是 LUT-vs-tanhf 结构差(非公平速度 A/B·我方胜精度)。④ **禁互推**：kernel-sym add/mul 1.18× ≠ census e2e 预判 add/mul deploy 0.76–0.83×（system 账·不同赛道）。

### Batch 3 — 码本/低比特余格（vec_dot + dequant + gemm cross-op）

**3a · FLAT vec_dot（§一.B·5 格·parity-by-adoption）** — 对手=block-dot·成色 tautological·补 4 列即可：
> **★ k1-half DONE（2026-07-14·L1 vec_dot k1-lane）**：cold_med ratio ours/opp (>1 ours 快)·对手类=objdump 逐符号机判·byte-exact 全格 0 mismatch/0 ULP·casefile `experiments/active/g7-census/vecdot-k1/`·**rvv-half PENDING**。

| 格@板 | 我方 emit | 对手类 [机判·符号级] | hot(k1) | cold M=1 | cold M=8 | 胜负(cold) | rvv |
|---|---|---|:--:|:--:|:--:|:--:|:--:|
| q4_0@k1 | block-dot emit | native-RVV inline (41ins/16rvv) | 0.983 | **0.983** | 0.983 | PARITY-by-adoption 0.98×（✓ tautological）| ☐pending |
| q4_1@k1 | block-dot emit | native-RVV inline (45/14) | 1.007 | **1.008** | 1.009 | PARITY-by-adoption 1.01×（✓）| ☐pending |
| q5_0@k1 | block-dot emit | native-RVV inline (80/32·csrr) | 0.385 | **0.383** | 0.386 | **LOSS 0.38×** ★parity-by-adoption **证伪**（qh 5-bit 发射发散·非 ggml 本体）| ☐pending |
| q5_1@k1 | block-dot emit | native-RVV inline (90/30·csrr) | 0.402 | **0.400** | 0.403 | **LOSS 0.40×** ★**证伪** | ☐pending |
| q8_0@k1 | block-dot emit | native-RVV inline (34/10) | 0.984 | **0.979** | 0.983 | PARITY-by-adoption 0.98×（✓）| ☐pending |

**3b · K-quant vec_dot（§一.C·5 格）** — ours-block-dot vs ggml-as-shipped·数值档 INT byte-exact（0 mismatch/0 ULP 全格）：
> **★ 对手纠正**：预判「ggml generic block-dot(弱)」**证伪**——as-shipped k1 VLEN256 dispatched = **全 hand-tuned native-RVV**（q2_K/q3_K/q5_K 单体 inline heavy RVV；q4_K/q6_K runtime dispatcher→`_vl256` 手调 specialization）=**强对手**。

| 格@板 | 我方 emit | 对手类 [机判·符号级] | hot(k1) | cold M=1 | cold M=8 | 胜负(cold) | rvv |
|---|---|---|:--:|:--:|:--:|:--:|:--:|
| q2_K@k1 | KQuant aux32 core | hand-tuned native-RVV inline (376/243) | 0.678 | **0.685** | 0.678 | **LOSS 0.68×**（weight-recon floor vs heavy 手调）| ☐pending |
| q3_K@k1 | KQuant aux32 core | hand-tuned native-RVV inline (239/116) | 0.538 | **0.538** | 0.538 | **LOSS 0.54×** | ☐pending |
| q4_K@k1 | KQuant aux32 core | runtime dispatch→`_vl256` 手调 (main 9ins csrr) | 0.610 | **0.617** | 0.606 | **LOSS 0.62×** | ☐pending |
| q5_K@k1 | KQuant aux32 core | native-RVV inline (214/97·**NO vl-spec**) | 1.081 | **1.085** | 1.081 | **WIN 1.09×** ⚠机制=对手 q5_K 无 vl256 specialization(唯一未手调格)·非"赢手调"·L1' 审 | ☐pending |
| q6_K@k1 | KQuant aux32 core | runtime dispatch→`_vl256` 手调 (main 28ins csrr) | 0.547 | **0.554** | 0.553 | **LOSS 0.55×** | ☐pending |

**★ vec_dot@k1 cold tally**：**1 WIN**（q5_K·⚠对手 immaturity 机制）· **3 PARITY**（q4_0/q4_1/q8_0·parity-by-adoption tautological）· **6 LOSS**（q2_K/q3_K/q4_K/q6_K weight-recon floor vs 强 native 对手；q5_0/q5_1 parity-by-adoption 证伪）。
**★ 输格三出口候选（K-quant vec_dot·供 L1'）**：Exit A=对手结构优势具名 `[GAP-KQUANT-VECDOT-VS-NATIVE-RVV]`；Exit B=能力键控 VLEN256/wide-LMUL K-quant block-dot 发射（SEL-1）；Exit C=**轴转移**（K-quant 真 beat 在 repack-GEMM 轴 T9§1.1/Win-K1-VLEN·vec_dot 天然 weight-recon-bound·**推荐主处置=vec_dot 轴维持 LOSS·format-beat 归 GEMM 轴**）。
**★ kernel-sym 计数建议（主会裁）**：非-tautological ≥parity = **q5_K 1.085×**（1 格·vs 真 native-RVV·须披露对手 immaturity 机制）；FLAT parity-by-adoption q4_0/q4_1/q8_0（3 格·tautological·report-as-parity 不计独立 beat）；q5_0/q5_1 证伪剔除；q2_K/q3_K/q4_K/q6_K LOSS。**禁互推 perf-covered/certified**。

**3c · IQ/fp4 vec_dot（§一.D·4 格 ✅ + q1_0 flag）** — gather-trap·数值档 INT bit-exact：
| 格 | 我方 emit | 对手类 | 预判 |
|---|---|---|---|
| vec_dot/iq1_s@rvv | iq1_s grid core | iq1_s_q8_K_generic | <parity（gather-trap·sibling 0.16–0.78×） |
| vec_dot/iq1_m@rvv | iq1_m grid core | iq1_m_q8_K_generic | <parity |
| vec_dot/iq4_nl@rvv | iq4_nl codebook core | iq4_nl_q8_0_generic | <parity（vec_dot 路未测·区别 gemm 0.217×） |
| vec_dot/nvfp4@rvv | emitNVFP4BlockDotBodyShared | nvfp4_q8_0_generic（★存在） | <parity（罕见格·fp4 codebook） |
| vec_dot/q1_0@rvv | q1_0 FLAT core | ⚠ q1_0_q8_0_generic（provenance 存疑） | flag 主会核·leans 单列 internal-A/B |

**3d · dequant [DEQ-AXIS]（§一.E·17 格 ✅ + q1_0 flag）** — streaming·数值档 ULP≈0·parity-expected·低值批量：
| 格族 | 对手类 | 预判 |
|---|---|---|
| dequant/{5 K-quant}@rvv/@k1 | dequantize_row_q*_K（mem-bound） | parity @roofline（同物理墙 dequant 体例） |
| dequant/{9 iq}+mxfp4+nvfp4+{2 tq}@rvv | dequantize_row_*（mem-bound/gather） | parity/LOSS @roofline |
| dequant/q1_0 | ⚠ provenance 存疑 | flag |

**3e · GEMM iq/tq cross-op（§一.A·7 格·主会话采 cross-op 才入·须明标）**：
| 格 | 口径 | 对手类 | 预判 |
|---|---|---|---|
| gemm_tile/{iq2_xxs,iq2_xs,iq2_s,iq4_xs,mxfp4,tq1_0,tq2_0}@rvv | **cross-op**（our-gemm vs opp-block-dot） | block-dot（gather-bound） | <parity（iq4_nl anchor 0.217× 同族·须明标 cross-op·非 same-op kernel-sym） |

---

## 四、待发射构造清单（估成本）

> §一/§二 全部 ✅ 入队格 **emit 均在**（逐 emitter 机判确认）→ **无"待发射"缺口**。以下为**新构造需求 = 0**（本轮普查纯测量·无构造）。

| 项 | 状态 | 成本 |
|---|---|---|
| B 类 9 算子 emit | **全在**（softmax/rms_norm/rope/silu/gelu/add/mul/scale/cpy）| 0（无待发射·softmax 必测已满足） |
| FLAT/K-quant/iq/fp4 vec_dot emit | **全在** | 0 |
| dequant 19 格 emit | **全在**（KQuant/IQGrid/Codebook/Nibble/Q* dequant） | 0 |
| GEMM iq/tq emit | **全在**（cross-op 口径） | 0 |

**唯一"构造"性缺口 = 测量 harness 而非 kernel**：B 类 9 算子 × 2 板对称 micro A/B harness（f32·N≥10·冷态）未建 → **低成本 harness 补建**（流式算子·非 kernel 构造）。gemm iq/tq cross-op 口径 harness = iq4_nl 先例已存·复用。

---

## 五、计数汇总

| 桶 | 格数 | 明细 |
|---|---:|---|
| **49 例外 → ✅ clean 入队** | **37** | FLAT vec_dot 5 + K-quant vec_dot 5 + iq/fp4 vec_dot 4（iq1_s/iq1_m/iq4_nl/nvfp4）+ dequant 18[DEQ-AXIS] + B 类 5（add/mul/rms_norm/softmax/rope） |
| **49 例外 → ⚠ flag（provenance 存疑）** | **2** | vec_dot/q1_0 + dequant/q1_0（对手符号存在·枚举 41·canon "Weft-internal"·internal-A/B 风险·主会核） |
| **49 例外 → ❌ 不入队** | **10** | GEMM iq/tq 7（缺②same-op·cross-op 可选）+ product_reduce 3（缺②无框架对手） |
| **待发射** | **0** | 全 emit 在（唯 B 类 harness 低成本补建） |
| **L1 普查队列总格-板点** | **~63 起** | Batch1 K-quant@rvv 5 + Batch2 B 类 18（9×2 板）+ Batch3a FLAT vec_dot 10 + 3b K-quant vec_dot 10 + 3c iq/fp4 vec_dot 5 + 3d dequant [DEQ-AXIS] 18 + 3e gemm cross-op 7（可选） |

> ★37 ✅ + 2 flag + 10 ❌ = 49 ✓。**headline 保守**：37 clean ✅ + 2 provenance-flag + 7 cross-op-optional + 3 hard-❌。

---

## 附：体例合规自检
- e2e 分类**未改**（本表 read-only·schema 不动）·新增 kernel-sym 列一格两账并行 ✓
- 对手类**机判符号级**（§0.3 逐符号 grep 实证·非手写类目）✓
- 上游更强未启用路径披露（q4_K@k1 hand-brick / iq/tq gemm opponent-absent / q1_0 provenance）✓
- 待发射**诚实标**（=0·全 emit 在·非假装）✓
- **禁互推**：kernel-sym（本表）≠ perf-covered 9/83 ≠ certified ✓
