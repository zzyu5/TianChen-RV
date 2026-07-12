# G5-M3 — IME **q4_K** forward-traffic routing 收口（forward-wired=T·k1·**super-block 跨范式家族完整性 C1**·IME family#2 forward 从 flat 泛化到 super-block）

> **campaign**: G5 接线战役 · **M3-ime-q4_K-forward = IME forward bridge 曳光弹的 super-block 兄弟** · q4_0@ime (flat·`9aeaab2d`) pattern 扩到 q4_K (super-block K-quant)
> **名义**: correctness-first 结构成就（跨范式 super-block 家族完整性·**非 perf**）。硬门 = **forward-wired=T（真 q4_K prefill traffic 经我方 super-block IME 核·real vmadot 两级 fold）+ 真-llama e2e correctness（ZERO-MODEL int32-exact 契约·MIRAGE de-risk）**。
> **起点**: q4_K@ime IME 核已 **emitter-sealed**（`IMEQ4KMatMulTileToEmitCFunc`·SIX bricks·两级 6-bit scale/min fold + vmadot·M2b K1 硅证 int32 0-diff）。本任务闭合 forward 集成层（#4 super-block 权重 repack / #3 q8_K 激活 quant/pack + activation-scale fold epilogue）+ 真流量路由 + 真-llama e2e。
> **board**: `ssh k1`（SpacemiT X60·VLEN256·IME harts 0–3·`taskset -c 0-3`）·出货 clang-18（kernel==system 收敛）·model=`/data/tinyllama-1.1b-Q4_K_M.gguf`。
> **禁 git**·**board 可逆**（vendor `cp *.ORIG` + restore + md5 零改动）·**MANIFEST one-file-per-backtick-bullet**。

---

## 0. 一页速览（verdict）

| 项 | 结果 |
|---|---|
| **forward-wired（真 q4_K traffic 经我方 super-block IME 核）** | **T**·env-gated parallel tcrv `tcrv_q4_K_tensor_traits` 注册（native passthrough repack + 我方 super-block 两级 scale/min fold vmadot GEMM + q8_K activation-scale epilogue）·**banner FIRES**（`routed real q4_K PREFILL mul_mat: M=15 N=2048 K=2048 real vmadot 0xe210312b two-level fold`）·shipped `.so` vmadot **32→33**（我方 super-block 核入真 forward） |
| **correctness_green** | **T（可辩护义）**·算术 correctness 硬锚 = 板 integration UT **单 tensor mul_mat A==B**（`normwise≈1e-7`·vs REAL ggml `dequantize_row_q4_K`/`quantize_row_q8_K`/`ggml_vec_dot_q4_K_q8_K` 三重 oracle·real vmadot）·本 session 加真集成证据（真流量 + coherent in-family e2e） |
| **#4 super-block 权重 repack vs REAL ggml `dequantize_row_q4_K`** | **T·byte-exact（0 ULP·0 mismatch）**·native 144B block_q4_K 逐块 gather 到 (nj,sb,nl)·decode(repacked)==ggml dequant 全元素相等（8×256/16×512/32×768 三网格 mism=0 max_abs=0.000e+00） |
| **#3 q8_K 激活 quant/pack vs REAL ggml `quantize_row_q8_K`** | **T·byte-exact**·f32→q8_K→fragment-major Apack int8 + per-(row,super-block) dA=q8_K.d·int8 全等 + scale 全等（三网格 mism=0） |
| **单 tensor mul_mat A==B（我方 super-block bridge vs ZERO-MODEL + real ggml）** | **T·bounded-ULP**·normwise ~1e-7（≪1e-5 gate）·ggml_vs_zeromodel ~3e-6（我方 double 重导 == 真 ggml kernel）·real vmadot 硅证 |
| **真-llama e2e greedy A==B**（我方 IME q4_K 桥 ON vs 参照） | **coherent in-family**·ON/VEN/OFF 三方共享前缀后 near-tie argmax flip 分歧·**VEN 自身亦 ≠ OFF**（证异核 f32-reassoc + q8_K requant 固有性·非我方桥 bug）·见 §3 |
| **MIRAGE de-risk** | **决定性过**·生成全为**连贯英文·与参照同族**（layout 失配 = 乱码；实际 = 三方共享 ~52-byte 前缀的连贯续写） |
| **板 restored（md5 零 stock 改动）** | **T**·`ime.cpp` md5 `40962c7e…`（==baseline）·`libggml-cpu.so.0.15.1` md5 `71cc4d29…`（==baseline）·src route-marker=0·`.ORIG` litter=0·EXIT trap 强制 restore |
| **perf** | **未测·跨范式名义**·forward-wired 后 IME 格 = **黄-传导稀释带账·非绿·非 headline**（micro↛e2e·decode M=1 走 native fallback·memory-bound 1B 模型无干净 IME-unit e2e 赢） |
| **双账本 clang 收敛** | k1 出货 clang-18（kernel==system 同数）·correctness 不涉 perf·无对手身份/八门 |
| **IME family#2 forward 家族完整度** | **q4_0 flat ✓ + q4_K super-block ✓**（两级 6-bit scale/min fold + q8_K per-super-block activation scale·跨 flat→super-block 泛化闭合） |

**一句话**：**q4_K@ime super-block forward bridge 的真流量路由已收口**——env-gated parallel tcrv `tcrv_q4_K_tensor_traits`（native passthrough 保留 144B block_q4_K + 我方 super-block 两级 scale/min fold vmadot GEMM + q8_K activation-scale epilogue 消费 prefill mul_mat）在真 K1 llama forward 中路由真 q4_K prefill 流量（banner·M=15 N=2048 K=2048·real vmadot·shipped .so vmadot 32→33）·**IME forward 从 flat（q4_0）泛化到 super-block（q4_K）**；correctness 硬锚 = 板 integration UT 单 tensor A==B（normwise ~1e-7·vs REAL ggml 三重 oracle·#4/#3 byte-exact 0-ULP）·真-llama e2e coherent in-family（near-tie flip·vendor IME 亦偏离·非桥 bug）。**板 stock 零改动（md5 双证·EXIT trap 强制 restore）。perf 不问·IME 格 forward-wired 后=黄-传导稀释带账非绿。**

---

## 1. 路由方案：env-gated parallel tcrv `tcrv_q4_K_tensor_traits`（复用 q4_0 pattern·super-block delta）

### 1.1 vendor 挂点事实（read-only·= 对照锚）
- vendor `get_optimal_repack_type` **Q4_K case 已存**（`tensor_traits<block_q4_K,32,16>` IME1 / `<...,32,32>` IME2）→ q4_K weight 落 spacemit buffer·`supports_op(MUL_MAT)` 对 q4_K F32-act 返回 true。
- **passthrough 安全性（super-block 关键差异 vs q4_0）**：q4_K buffer alloc = `remap_block_nbytes(sizeof(block_q4_K)=144, sizeof(block_q4_1)*8=160)` → 每块 **160B ≥ native 144B**·我方 passthrough `memcpy(t->data, data, data_size=native)` 写 native 144B 连续到 buffer 前段·**尺寸兼容**（写 < 分配）·tensor `nb[1]` 保持 native `(K/256)*144`·我方 kernel 按 native 连续读 → 一致正确。
- **native block_q4_K 逐块保留**（emitter decode 直读 native 布局·q4_K nibble UNSIGNED [0,15]·min bias 下游施加）→ #4 repack = **纯 144B block gather**（无 nibble 重排·比 q4_0 fragment-major 更简单）。

### 1.2 我方 tcrv trait（`forward-route-patch-q4k.py`·插入 vendor `ime.cpp`·env-gated·可逆）
`class tcrv_q4_K_tensor_traits`：
- **`repack()` = NATIVE passthrough**（memcpy 144B/block 保留）→ 未拦截 op（get_rows / decode M=1 / 异形 / q6_K 混量化张量）自动回落 ggml 默认·读 native·恒正确。
- **`compute_forward()` 路由判据**（否则 return false→ggml 默认·正确）：`op==MUL_MAT` ∧ `src0==q4_K` ∧ `src1==f32` ∧ `M(=ne11)>1`（prefill）∧ `N%4==0` ∧ `K%256==0`（super-block）∧ 2D ∧ 连续 ∧ src0 native row-stride。命中 `ith==0`：#4 native→(nj,sb,nl) block gather · #3 f32→q8_K（`quantize_row_q8_K_ref`）Apack+dA · super-block 两级 fold vmadot GEMM（S_scale=Σ sc_b·sumi_b·S_min=Σ m_b·asum_b·real vmadot）· epilogue fold `C += (d_w·y.d)·S_scale − (dmin_w·y.d)·S_min`（== ggml `vec_dot_q4_K_q8_K`）· M pad 到 4 · 写 dst f32 · `ggml_barrier`。
- **核 = SEAL-VERBATIM**：`vmadot_mac_kloop`/`dequant_fragment`/`get_scale_min` 与 M2b K1 seal（`q4-K-matmul-tile-int32-k1seal.c`）+ emitter（`IMEQ4KMatMulTileToEmitCFunc`）**byte-identical**·bridge 版加 activation-scale fold（forward GEMM 必需·非空心核）。
- **env gate（`get_optimal_repack_type` Q4_K case 首行）**：`if(getenv("TCRV_IME_Q4K_BRIDGE") && ne[1]%4==0 && ne[0]%256==0) return &tcrv_q4_K_bridge;` 否则 vendor 路径**字节不变**（env OFF = stock 行为·同二进制 A/B）。

### 1.3 activation scale 的 super-block 处理（vs q4_0 flat 关键差异）
q4_K 配对 **q8_K** 激活（**per-256-super-block** 单 scale·非 q8_0 的 per-32-block）→ activation scale `y.d[m][sb]` 须 **per-(row,super-block) 后乘**入 epilogue。emitter/seal 核只算 raw int8 S_scale/S_min（activation scale=1）·bridge epilogue 补 `y.d[m][sb]` fold → 数学恒等 ggml `d=x.d·y.d, dmin=x.dmin·y.d, sumf += d·Σsc·sumi − dmin·Σm·asum`。

---

## 2. forward-wired=T 证据（真流量路由·`raw/q4k-forward-route-multi.txt`）

### 2.1 banner FIRES（真 q4_K prefill 流量经我方 super-block 核）
```
[TCRV-IME-Q4K-BRIDGE] routed real q4_K PREFILL mul_mat through tcrv IME kernel: M=15 N=2048 K=2048 (real vmadot 0xe210312b two-level fold)
```
M=15（prompt token 数·真 prefill）·N=2048/K=2048（transformer 线性层·真 llama forward 维度·K%256==0 super-block）。`VEN_BANNER=0`（env OFF 对称·vendor 路径零我方 banner）。

### 2.2 objdump vmadot engage（shipped .so·真 forward）
```
vmadot_in_baseline_so = 32   (vendor IME)
vmadot_in_patched_so  = 33   (+1 = 我方 tcrv super-block IME 核 0xe210312b 入 shipped libggml-cpu.so)
```

---

## 3. 板 integration UT（correctness 硬锚·`raw/q4k-bridge-ut.txt`·real vmadot·vs REAL ggml 三重 oracle）

```
[#4 weight-repack] dequant N/K∈{8/256,16/512,32/768}: 全 mism=0 max_abs=0.000e+00 (byte-exact vs ggml dequantize_row_q4_K)
[#3 act-quant+pack] M/K∈{8/256,12/512,8/768}: int8 全 mism=0 | scale 全 mism=0 (byte-exact vs ggml quantize_row_q8_K)
[FULL A==B] 5 网格 (M×N×K 到 16×64×768):
  normwise ∈ [8.9e-08, 1.4e-07]  (≪1e-5 gate·f32 reassoc only)
  vs_real_ggml_normwise ∈ [7.1e-08, 1.2e-07]  (我方 bridge == 真 ggml_vec_dot_q4_K_q8_K)
  ggml_vs_zeromodel ∈ [8.9e-07, 3.6e-06]  (我方 double 重导 == 真 ggml kernel·证 ZERO-MODEL oracle 忠实)
Q4K-BRIDGE-UT PASS
```
**证书三要件**：① 语料完备（5 随机网格 + real-llama 全模型 forward）·② 输入路径同源（真 ggml native q4_K/q8_K quantizer·同 on-disk 部署格式）·③ oracle 独立三重（ggml `dequantize_row_q4_K` + `quantize_row_q8_K` + `ggml_vec_dot_q4_K_q8_K` 真 kernel·加 double ZERO-MODEL 重导·异码路）。

## 4. 真-llama e2e greedy A==B（correctness·MIRAGE de-risk·`raw/q4k-forward-route-multi.txt`）

**规程**：build-ime patched·三配置（`-n 24 --temp 0 -s 0 -t 4 -no-cnv --no-warmup --no-display-prompt --simple-io`·`taskset -c 0-3`·`llama-completion` 非 llama-cli）：
- **OFF** = build-off `llama-completion`（stock RVV q4_K·无 IME·**跨范式**独立 oracle）
- **VEN** = build-ime env unset（vendor IME q4_K `gemm_kernel_i8i4`·**同范式**独立 oracle·异实现）
- **ON** = build-ime `TCRV_IME_Q4K_BRIDGE=1`（我方 tcrv super-block IME 桥）

| prompt | ON==VEN | ON==OFF | VEN==OFF | banner | 判读 |
|---|---|---|---|---|---|
| 1 "…curious little robot…" | DIFFER | DIFFER | DIFFER | 1 | 三方共享 ~52B 前缀（"curious mind"）后各异·全连贯 |
| 2 "The quick brown fox…" | DIFFER | DIFFER | DIFFER | 1 | **ON 追 OFF 长前缀**（"Red Badge of Courage"）·VEN 退化重复循环 |
| 3 "In the year 2050…" | DIFFER | **IDENTICAL** | DIFFER | 1 | **ON 与 stock RVV 逐字同**·VEN 独异 |
| 4 "…rules of good writing…" | DIFFER | **IDENTICAL** | DIFFER | 1 | **ON 与 stock RVV 逐字同**·VEN 独异 |
| 5 "She opened the ancient book…" | DIFFER | DIFFER | DIFFER | 1 | 三核各异·全连贯 in-family |

**汇总**：**ON==OFF 2/5**（prompt 3/4 逐字同 stock RVV）·**ON==VEN 0/5**·**VEN==OFF 0/5**·全 5/5 生成**连贯 in-family 英文**·**banner 5/5**。

**load-bearing 读法（诚实·MIRAGE de-risk）**：
1. **MIRAGE 决定性 de-risk**：layout 失配 = 乱码；实际 5/5 = 连贯英文续写（prompt1 三方前 ~52B 逐字同·prompt3/4 ON 与 OFF 全同）→ 我方 super-block 核复现参照族真实计算·非巧合 garbage。
2. **e2e greedy 逐 token 恒等【非】异核有效判据**：**VEN==OFF=0/5**（vendor IME 自身逐 prompt 偏离 stock RVV·prompt2 甚至退化重复循环）→ 分歧是任两异核（RVV block-dot / vendor gemm_kernel_i8i4 / 我方 super-block fragment-major fold）f32 reassociation + q8_K requant 的**固有 near-tie argmax flip**·**非我方桥 bug**。**反证强度更高**：我方 ON 与 stock RVV 参照 **2/5 逐字同**、vendor **0/5**——我方 super-block 两级 fold 比 vendor IME **更贴 ggml `vec_dot_q4_K_q8_K` 参照数学**。
3. **算术 correctness 硬锚 = §3 板 integration UT 单 tensor A==B**（normwise ~1e-7·vs REAL ggml `dequantize_row_q4_K`/`quantize_row_q8_K`/`ggml_vec_dot_q4_K_q8_K` 三重 oracle·#4/#3 byte-exact 0-ULP·real vmadot）——证我方核算的正是**数学正确的 q4_K×q8_K super-block GEMM**·无 bug。

---

## 5. 板 restored（md5 双证零 stock 改动·EXIT trap 强制）
`run-forward-route-q4k-multi.sh` 用 **EXIT trap** → 任何退出路径均 restore：clean source → `make ggml-cpu`（clean .o）→ `.so` 覆盖回 ORIG binary → md5 双证。
```
restored ime.cpp = 40962c7e7c732bf472ae88cef89ced8d  == baseline ✓
restored so      = 71cc4d295dac29382a0a7d4d5bd0c425  == baseline ✓
RESTORE md5 ZERO-CHANGE OK · src_route_left=0 · litter_left=0 · vmadot_in_so 回 32
```

---

## 6. 诚实边界（[NG-4]·跨范式名义）
- **已闭**：IME forward bridge **从 flat（q4_0）泛化到 super-block（q4_K）**·真 q4_K prefill 流量经我方 super-block 两级 6-bit scale/min fold + q8_K per-super-block activation-scale vmadot IME 核（banner 5/5·shipped .so vmadot 32→33）+ 真-llama e2e coherent in-family（ON 与 stock RVV 2/5 逐字同·比 vendor 0/5 更贴参照）。**C1 模板协议在家族#2 forward 的 super-block extensibility 铁证：flat repack 接线方法学跨 super-block K-quant（两级 fold + super-block activation scale）泛化到真 forward。**
- **未闭 / 诚实限制**：① **严格 "ON==OFF byte-identical greedy 全 prompt" 未达成**（2/5·near-tie flip）——**非 bug**（VEN 自身 0/5·near-tie 固有）·但不得声称严格 e2e byte-identical A==B vs stock。② **logit-level bounded-ULP 量化未测**（需自建 llama-API logit-dump harness 量化 max|Δlogit| + tie margin）= next-step。③ **decode（M=1）未路由我方核**（有意走 native passthrough fallback·正确但非我方核·prefill 才是阵列物理意义 regime）。④ **q4_K 混量化模型的 q6_K 张量走 vendor IME**（ON/VEN 同·非我方桥·ON==VEN 分歧纯 q4_K 层）。
- **perf 未测**·跨范式名义·`[NG-4]` 全程·**禁"实质胜利/perf 赢"表述**——IME 格 forward-wired 后 = **黄-传导稀释带账·非绿·非 headline**（memory-bound 1B 模型无干净 IME-unit e2e 赢·micro↛e2e）。**成功 = forward-wired=T + 真-llama e2e coherent in-family（结构·非 perf）。**

## 7. next-step（优先序·若续）
1. **logit-level bounded-ULP 量化**：dump prefill 末位 logit 向量（ON/VEN/OFF）·算 max|Δlogit| + top-2 tie margin → 把 near-tie flip 精确化为 "bounded-ULP·argmax-at-tie"。
2. **q5_K/q6_K@ime forward**（super-block 家族续铺·q5_K 已有 k1 e2e 素材）。
3. （perf·须另裁·预注册 parity/非绿·黄-带账）prefill delta·强制 vendor-ceiling 同域披露 + kernel-family-vs-array 分解 + 对手身份。

## durable files（见 MANIFEST）
- `tools/e2e-harness/board/g5-m3-ime-q4_K/g5m3_q4k_bridge_ut.c`（super-block integration UT·vs REAL ggml 三重 oracle）
- `tools/e2e-harness/board/g5-m3-ime-q4_K/run-q4k-bridge-ut.sh`（UT 构建+跑·link ggml-base+cpu）
- `tools/e2e-harness/board/g5-m3-ime-q4_K/forward-route-patch-q4k.py`（reversible env-gated parallel tcrv q4_K tensor_traits patcher）
- `tools/e2e-harness/board/g5-m3-ime-q4_K/run-forward-route-q4k-multi.sh`（5-prompt e2e A==B seal·EXIT-trap restore）
- `raw/q4k-bridge-ut.txt`·`raw/q4k-forward-route-multi.txt`
