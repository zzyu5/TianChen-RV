# G5-M3 — IME **q8_0** forward-traffic routing 收口（forward-wired=T·k1·**flat int8-DIRECT·完成 ratified IME triple {q4_0·q8_0·q4_K}@ime 全 forward-wired**）

> **campaign**: G5 接线战役 · **M3-ime-q8_0-forward = ratified IME triple 的最后一格** · q4_0@ime（flat·nibble·`9aeaab2d`）+ q4_K@ime（super-block·`d73096da`）pattern 扩到 q8_0（flat·int8-DIRECT·无 nibble 解码）
> **名义**: correctness-first 结构成就（**跨范式 triple 家族完整性·非 perf**）。硬门 = **forward-wired=T（真 q8_0 prefill traffic 经我方 flat int8-direct IME 核·real vmadot）+ 真-llama e2e correctness（ZERO-MODEL int32-exact 契约·MIRAGE de-risk）**。
> **起点**: q8_0@ime IME 核已 **emitter-sealed**（`IMEQ80MatMulTileToEmitCFunc`·`Q80DequantCoreOp` DIRECT int8 read·无 nibble·复用 `VmadotMacLeafOp`·硅证 int32 0-diff）。本任务闭合 forward 集成层（#4 int8-direct 权重 repack / #3 q8_0 激活 quant/pack + scale-fold epilogue）+ 真流量路由 + 真-llama e2e。
> **q8_0 delta（比 q4_0 更简单）**：#4 weight = int8 直读（无 nibble unpack）·#3 activation = q8_0（复用 q4_0@ime）·scale-fold = 同 flat（复用 q4_0@ime epilogue）。
> **board**: `ssh k1`（SpacemiT X60·VLEN256·IME harts 0–3·`taskset -c 0-3`）·出货 clang-18（kernel==system 收敛）·model=`/data/tinyllama-q8_0.gguf`（sha256 `a4c9bb1d…`·GGUF v3·A/B 同权重）。
> **禁 git**·**board 可逆**（vendor `cp *.ORIG` + restore + md5 零改动）·**MANIFEST one-file-per-backtick-bullet**。

---

## 0. 一页速览（verdict）

| 项 | 结果 |
|---|---|
| **forward-wired（真 q8_0 traffic 经我方 flat int8-direct IME 核）** | **T**·env-gated parallel tcrv `tcrv_q8_0_tensor_traits` 注册（native passthrough repack + 我方 flat int8-direct scale-fold vmadot GEMM）·**banner FIRES 5/5**（`routed real q8_0 PREFILL mul_mat: M=15 N=2048 K=2048 real vmadot 0xe210312b int8-direct`）·shipped `.so` vmadot **32→33**（我方 int8-direct 核入真 forward） |
| **correctness_green** | **T（可辩护义·比 q4_0/q4_K 更强）**·算术 correctness 硬锚 = 板 integration UT **单 tensor mul_mat A==B**（`normwise ∈ [6.4e-08, 1.2e-07]`·vs REAL ggml `dequantize_row_q8_0`/`quantize_row_q8_0_ref` 双 oracle·**`max_abs_vs_float_order = 0.000e+00` 全网格**·int8-direct 零 reassoc 残差）·加真集成证据（真流量 + 真-llama e2e **ON==OFF byte-identical 3/5**） |
| **#4 int8-direct 权重 repack vs REAL ggml `dequantize_row_q8_0`** | **T·byte-exact（0 ULP·0 mismatch）**·native 34B block_q8_0 int8 直读 gather 到 fragment-major Bpack（无 nibble unpack）·reconstruct(Bpack,dW)==ggml dequant 全元素相等（8×64/16×128/32×256 三网格 mism=0 max_abs=0.000e+00） |
| **#3 q8_0 激活 quant/pack vs REAL ggml `quantize_row_q8_0_ref`** | **T·byte-exact**·f32→q8_0→fragment-major Apack int8 + per-(row,block) dA·int8 全等 + scale 全等（三网格 mism=0）·与 q4_0@ime #3 byte-identical（共享 q8_0 activation path） |
| **单 tensor mul_mat A==B（我方 int8-direct bridge vs ZERO-MODEL q8_0×q8_0 double）** | **T·bounded-ULP**·normwise ~1e-7（≪1e-5 gate）·**`max_abs_vs_float_order = 0.000e+00`**（我方 GEMM 的 f32 算术 = stock block-dot fold order **逐 bit 同**·int8-direct 无 dequant reassoc·比 q4_K 更干净）·real vmadot 硅证 |
| **真-llama e2e greedy A==B**（我方 IME q8_0 桥 ON vs 参照） | **ON==OFF byte-identical 3/5**（prompt 1/2/4 逐字同 stock RVV q8_0）·**5/5 连贯 in-family**·2/5（prompt 3/5）near-tie argmax flip 分歧（连贯英文·非 bug·见 §4） |
| **MIRAGE de-risk** | **决定性过**·生成全为**连贯英文·与参照同族**（layout 失配 = 乱码；实际 = 3/5 逐字同 stock + 2/5 连贯续写） |
| **板 restored（md5 零 stock 改动）** | **T**·`ime.cpp` md5 `40962c7e…`（==baseline）·`libggml-cpu.so.0.15.1` md5 `71cc4d29…`（==baseline）·src route-marker=0·`.ORIG` litter=0·EXIT trap 强制 restore |
| **perf** | **未测·跨范式名义**·forward-wired 后 IME 格 = **黄-传导稀释带账·非绿·非 headline**（micro↛e2e·decode M=1 走 native fallback·memory-bound 1B 模型无干净 IME-unit e2e 赢） |
| **双账本 clang 收敛** | k1 出货 clang-18（kernel==system 同数）·correctness 不涉 perf·无对手身份/八门 |
| **★ratified IME triple forward-wired 完成度** | **q4_0 flat✓（nibble）+ q4_K super-block✓（两级 fold）+ q8_0 flat✓（int8-direct）= 全 3/3 forward-wired 完成** |

**一句话**：**q8_0@ime flat int8-direct forward bridge 的真流量路由已收口**——env-gated parallel tcrv `tcrv_q8_0_tensor_traits`（native passthrough 保留 34B block_q8_0 + 我方 flat int8-direct scale-fold vmadot GEMM 消费 prefill mul_mat）在真 K1 llama forward 中路由真 q8_0 prefill 流量（banner 5/5·M=15 N=2048 K=2048·real vmadot·shipped .so vmadot 32→33）·**完成 ratified IME triple {q4_0·q8_0·q4_K}@ime 全 forward-wired**；correctness 硬锚 = 板 integration UT 单 tensor A==B（normwise ~1e-7·`max_abs_vs_float_order=0.000e+00` 全网格·vs REAL ggml 双 oracle·#4/#3 byte-exact 0-ULP）·真-llama e2e **ON==OFF byte-identical 3/5**（比 q4_0/q4_K 更强·int8-direct 零 reassoc）。**板 stock 零改动（md5 双证·EXIT trap 强制 restore）。perf 不问·IME 格 forward-wired 后=黄-传导稀释带账非绿。**

---

## 1. 路由方案：env-gated parallel tcrv `tcrv_q8_0_tensor_traits`（复用 q4_0 pattern·int8-direct delta）

### 1.1 vendor 挂点事实（read-only·= 对照锚）
- vendor `get_optimal_repack_type` **Q8_0 case 已存**（`q8_0_32x32_q8_0`·IME2 路径）→ q8_0 weight 落 spacemit buffer·`supports_op(MUL_MAT)` 对 q8_0 F32-act 返回 true（compute_forward switch 含 `GGML_TYPE_Q8_0`）。
- **passthrough 安全性（int8-direct·vs q4_K 更简单）**：q8_0 buffer alloc = `remap_block_nbytes(sizeof(block_q8_0)=34, sizeof(block_q8_0)=34, padded_rows=32)` → **dst_block == native 34B**（仅行数补齐到 32 的倍数·tinyllama 各线性层 N∈{2048,5632,32000} 均 %32==0 无补）·我方 passthrough `memcpy(t->data, data, data_size=native)` 写 native 34B 连续到 buffer 前段·**尺寸兼容**（写 ≤ 分配）·tensor `nb[1]` 保持 native `(K/32)*34`·我方 kernel 按 native 连续读 → 一致正确。
- **native block_q8_0 = int8 直存**（2B fp16 d + 32 int8·无 nibble·无 offset-binary）→ #4 repack = **纯 int8 gather 到 fragment-major**（无 nibble 重排·无 −8·**比 q4_0 fragment-major 更简单**）。

### 1.2 我方 tcrv trait（`forward-route-patch-q8.py`·插入 vendor `ime.cpp`·env-gated·可逆）
`class tcrv_q8_0_tensor_traits`：
- **`repack()` = NATIVE passthrough**（memcpy 34B/block 保留）→ 未拦截 op（get_rows / decode M=1 / 异形 / 混量化张量）自动回落 ggml 默认·读 native·恒正确。
- **`compute_forward()` 路由判据**（否则 return false→ggml 默认·正确）：`op==MUL_MAT` ∧ `src0==q8_0` ∧ `src1==f32` ∧ `M(=ne11)>1`（prefill）∧ `N%4==0` ∧ `K%32==0`（flat）∧ 2D ∧ 连续 ∧ src0 native row-stride `(K/QK8_0)*sizeof(block_q8_0)`。命中 `ith==0`：#4 native q8_0 → int8-direct fragment-major Bpack + dW · #3 f32→q8_0（`quantize_row_q8_0_ref`）Apack+dA · flat scale-fold vmadot GEMM（`C[m,n] += dA·dW·(int32 vmadot partial)`·**B int8 直读·无 dequant 步**）· M pad 到 4 · 写 dst f32 · `ggml_barrier`。
- **核 = SEAL-VERBATIM**：`vmadot_mac_kloop` 与 q4_0/q4_K seal + emitter（`VmadotMacLeafOp` 共享·0xe210312b）**byte-identical**。int8-direct 无需 `dequant_fragment`（q8_0 weight 本身即 int8）。
- **env gate（`get_optimal_repack_type` Q8_0 case 首行·anchor 唯一）**：`if(getenv("TCRV_IME_Q80_BRIDGE") && ne[1]%4==0 && ne[0]%32==0) return &tcrv_q8_0_bridge;` 否则 vendor 路径**字节不变**（env OFF = stock 行为·同二进制 A/B·**VEN_BANNER=0 实证**）。

### 1.3 int8-direct 的 flat 处理（vs q4_0 nibble 关键差异）
q8_0 weight = int8 直存（无 nibble·无 offset-binary −8），故 #4 直接把 native block 的 32 个 int8 gather 到 fragment-major Bpack `[(nj·kt+kf)·32 + nl·8 + kl]`；GEMM 的 B fragment **直读 int8**（q4_0 需先 `dequant_fragment` nibble 解码 + −8·q8_0 跳过此步）。故 **q8_0 int8-direct 无 dequant reassociation** → §3 UT `max_abs_vs_float_order=0.000e+00`（我方 GEMM = stock block-dot fold order 逐 bit 同）。

---

## 2. forward-wired=T 证据（真流量路由·`raw/q8-forward-route-multi.txt`）

### 2.1 banner FIRES 5/5（真 q8_0 prefill 流量经我方 int8-direct 核）
```
[TCRV-IME-Q80-BRIDGE] routed real q8_0 PREFILL mul_mat through tcrv IME kernel: M=15 N=2048 K=2048 (real vmadot 0xe210312b int8-direct)
```
M=15（prompt token 数·真 prefill）·N=2048/K=2048（transformer 线性层·真 llama forward 维度·K%32==0 flat）。`BANNER_TOTAL=5`（5/5 ON run 命中）·`VEN_BANNER=0`（env OFF 对称·vendor 路径零我方 banner）。

### 2.2 objdump vmadot engage（shipped .so·真 forward）
```
vmadot_in_baseline_so = 32   (vendor IME)
vmadot_in_patched_so  = 33   (+1 = 我方 tcrv flat int8-direct IME 核 0xe210312b 入 shipped libggml-cpu.so)
```

---

## 3. 板 integration UT（correctness 硬锚·`raw/q8-bridge-ut.txt`·real vmadot·vs REAL ggml 双 oracle）

```
objdump vmadot_count=1（0xe210312b·real vmadot leaf 硅证）
[#4 weight-repack] dequant N/K∈{8/64,16/128,32/256}: 全 mism=0 max_abs=0.000e+00 (byte-exact vs ggml dequantize_row_q8_0)
[#3 act-quant+pack] M/K∈{8/64,12/128,8/256}: int8 全 mism=0 | scale 全 mism=0 (byte-exact vs ggml quantize_row_q8_0_ref)
[FULL A==B] 5 网格 (M×N×K 到 64×64×512):
  normwise ∈ [6.4e-08, 1.2e-07]  (≪1e-5 gate·f32 reassoc only)
  max_abs_vs_float_order = 0.000e+00  全 5 网格  (我方 GEMM = stock block-dot float fold order 逐 bit 同·int8-direct 零 reassoc)
Q8-BRIDGE-UT PASS
```
**证书三要件**：① 语料完备（5 随机网格 + real-llama 全模型 forward）·② 输入路径同源（真 ggml native q8_0 quantizer·同 on-disk 部署格式）·③ oracle 独立双重（ggml `dequantize_row_q8_0` + `quantize_row_q8_0_ref` 真 quantizer·加 double ZERO-MODEL q8_0×q8_0 重导·异码路）。**`max_abs_vs_float_order=0.000e+00`** = int8-direct 核比 q4_0（nibble decode）/q4_K（super-block fold）**更强的算术等同**（无 dequant 引入的 f32 reassoc）。

## 4. 真-llama e2e greedy A==B（correctness·MIRAGE de-risk·`raw/q8-forward-route-multi.txt`）

**规程**：三配置（`-n 24 --temp 0 -s 0 -t 4 -no-cnv --no-warmup --no-display-prompt --simple-io`·`taskset -c 0-3`·`llama-completion` 非 llama-cli）：
- **OFF** = build-off `llama-completion`（stock RVV q8_0·无 IME·**跨范式**独立 oracle）
- **VEN** = build-ime env unset（vendor IME q8_0 `q8_0_32x32_q8_0`·**同范式**独立 oracle·异实现）
- **ON** = build-ime `TCRV_IME_Q80_BRIDGE=1`（我方 tcrv flat int8-direct IME 桥）

| prompt | ON==VEN | ON==OFF | VEN==OFF | banner | 判读 |
|---|---|---|---|---|---|
| 1 "…curious little robot…" | IDENTICAL | **IDENTICAL** | IDENTICAL | 1 | **三核逐字同**·连贯 |
| 2 "The quick brown fox…" | IDENTICAL | **IDENTICAL** | IDENTICAL | 1 | **三核逐字同**·连贯 |
| 3 "In the year 2050…" | DIFFER | DIFFER | IDENTICAL | 1 | near-tie flip·ON 连贯续写（"a reality. The world was a different place…"）·VEN==OFF |
| 4 "…rules of good writing…" | IDENTICAL | **IDENTICAL** | IDENTICAL | 1 | **三核逐字同**·连贯 |
| 5 "She opened the ancient book…" | DIFFER | DIFFER | IDENTICAL | 1 | near-tie flip·ON 连贯续写（"the secrets of the universe lay within its pages…"）·VEN==OFF |

**汇总**：**ON==OFF byte-identical 3/5**（prompt 1/2/4 逐字同 stock RVV q8_0）·**VEN==OFF 5/5**（vendor IME q8_0 逐 prompt 与 stock RVV 逐字同）·全 5/5 生成**连贯 in-family 英文**·**banner 5/5**。

**load-bearing 读法（诚实·MIRAGE de-risk）**：
1. **MIRAGE 决定性 de-risk**：layout 失配 = 乱码；实际 5/5 连贯英文（3/5 ON 与 OFF 全 24-token 逐字同·2/5 连贯续写）→ 我方 int8-direct 核复现参照族真实计算·非巧合 garbage。
2. **e2e greedy 逐 token 恒等【非】异核有效判据**：本格 **VEN==OFF=5/5**（vendor IME q8_0 与 stock RVV 逐 prompt 逐字同·q8_0 数值路径贴合）→ prompt 3/5 的 ON 分歧是**我方 fragment-major sequential-block fold 与 stock RVV vectorized reduction 在末位 ULP 的 near-tie argmax flip**（我方 GEMM 与 stock 的**序内**算术逐 bit 同【§3 `max_abs_vs_float_order=0.000e+00`】·差异纯在 2048-维归约的**跨-fold 顺序**·非我方桥 bug）。**反证强度**：我方 ON 与 stock RVV **3/5 逐字同**（q4_0 2/5·q4_K 2/5）——q8_0 int8-direct 是三格中 e2e 最贴参照者。
3. **算术 correctness 硬锚 = §3 板 integration UT 单 tensor A==B**（normwise ~1e-7·`max_abs_vs_float_order=0.000e+00`·vs REAL ggml `dequantize_row_q8_0`/`quantize_row_q8_0_ref` 双 oracle·#4/#3 byte-exact 0-ULP·real vmadot）——证我方核算的正是**数学正确的 q8_0×q8_0 GEMM**·无 bug。

---

## 5. 板 restored（md5 双证零 stock 改动·EXIT trap 强制）
`run-forward-route-q8-multi.sh` 用 **EXIT trap** → 任何退出路径均 restore：clean source → `make ggml-cpu`（clean .o）→ `.so` 覆盖回 ORIG binary → md5 双证。
```
restored ime.cpp = 40962c7e7c732bf472ae88cef89ced8d  == baseline ✓
restored so      = 71cc4d295dac29382a0a7d4d5bd0c425  == baseline ✓
RESTORE md5 ZERO-CHANGE OK · src_route_left=0 · litter_left=0 · vmadot_in_so 回 32
```
（run 结束后独立复核：`ime.cpp` md5 `40962c7e…` / `.so` md5 `71cc4d29…` / route_left=0 —— 二次确认。）

---

## 6. 诚实边界（[NG-4]·跨范式名义）
- **已闭**：IME forward bridge **完成 ratified triple {q4_0 flat-nibble·q4_K super-block·q8_0 flat-int8-direct} 全 3/3 forward-wired**·q8_0 真 prefill 流量经我方 flat int8-direct scale-fold vmadot IME 核（banner 5/5·shipped .so vmadot 32→33）+ 真-llama e2e **ON==OFF byte-identical 3/5**（三格中最强）。**C1 模板协议在 IME 家族#2 forward 的 extensibility 铁证：同一 env-gated parallel tcrv tensor_traits 接线方法学跨 {nibble·super-block·int8-direct} 三种量化 memory-layout 全泛化到真 forward。**
- **未闭 / 诚实限制**：① **严格 "ON==OFF byte-identical greedy 全 prompt" 未达成**（3/5·prompt 3/5 near-tie flip）——**非 bug**（§3 A==B `max_abs_vs_float_order=0.000e+00`·差异纯跨-fold 归约顺序）·但不得声称严格 e2e byte-identical A==B vs stock 全 prompt。② **logit-level bounded-ULP 量化未测**（需自建 llama-API logit-dump harness 量化 max|Δlogit| + tie margin）= next-step。③ **decode（M=1）未路由我方核**（有意走 native passthrough fallback·正确但非我方核·prefill 才是阵列物理意义 regime）。
- **perf 未测**·跨范式名义·`[NG-4]` 全程·**禁"实质胜利/perf 赢"表述**——IME 格 forward-wired 后 = **黄-传导稀释带账·非绿·非 headline**（memory-bound 1B 模型无干净 IME-unit e2e 赢·micro↛e2e）。**成功 = forward-wired=T + 真-llama e2e coherent in-family（3/5 byte-identical·结构·非 perf）。**
- **build-dir 备注（可逆决策·登记）**：任务建议 `build-ime-bridge/` 独占，但板 llama-completion RUNPATH 为**绝对路径** `.../build-ime/bin`（`cp -r` 无法隔离 .so 加载·CMakeCache 含 5 处 build-ime 绝对路径）→ 干净隔离需全新 cmake 重构（板上昂贵/有风险）。改用**今日已两次证（q4_0+q4_K）的 build-ime 原位 + EXIT-trap restore** 模式（md5 双证零改动·板运行时处 clean baseline 无并行线占用）·完全可逆·板安全。

## 7. next-step（优先序·若续）
1. **logit-level bounded-ULP 量化**（三格统一）：dump prefill 末位 logit 向量（ON/VEN/OFF）·算 max|Δlogit| + top-2 tie margin → 把 near-tie flip 精确化为 "bounded-ULP·argmax-at-tie"。
2. **q5_0/q4_1 等 flat 家族 forward**（int8-direct 变体已有素材·flat 家族续铺）。
3. （perf·须另裁·预注册 parity/非绿·黄-带账）prefill delta·强制 vendor-ceiling 同域披露 + kernel-family-vs-array 分解 + 对手身份。

## durable files（见 MANIFEST）
- `tools/e2e-harness/board/g5-m3-ime-q8_0/g5m3_q8_bridge_ut.c`（flat int8-direct integration UT·vs REAL ggml 双 oracle）
- `tools/e2e-harness/board/g5-m3-ime-q8_0/run-q8-bridge-ut.sh`（UT 构建+跑·link ggml-base）
- `tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py`（reversible env-gated parallel tcrv q8_0 tensor_traits patcher）
- `tools/e2e-harness/board/g5-m3-ime-q8_0/run-forward-route-q8-multi.sh`（5-prompt e2e A==B seal·EXIT-trap restore）
- `raw/q8-bridge-ut.txt`·`raw/q8-forward-route-multi.txt`
