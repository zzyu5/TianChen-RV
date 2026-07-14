# G5-M3 session 2 — IME q4_0 forward bridge：集成层 4 缺口收尾（k1 · 跨范式完整性名义 C1·N2 family#2）

> **campaign**: G5 接线战役 · **M3-ime-q4_0-bridge = IME forward bridge 曳光弹** · **session 2/≥3**
> **名义**: correctness-first 结构成就（跨范式完整性·**非 perf**）。硬门 = **correctness A==B（ZERO-MODEL int32-exact 契约·部署变体≠证过变体）**。
> **起点**: session-1（commit `13c1c74a`·evidence.md）闭 bridge **#1 scale-fold epilogue** + **#2 runtime shape**（host+K1 硅 correctness GREEN·emitter 实发+lit）·`forward-wired=F`。
> **本 session 目标 = 集成层 4 缺口**（#4 权重 repack 桥 / #3 激活 quant 桥 / #5 forward hook / ④ e2e A==B）。
> **board**: `ssh k1`（SpacemiT X60·`use_ime1:1`·VLEN256·IME harts 0–3·`taskset -c 0-3`）·出货 clang-18（system-axis）/ kernel-axis seal = stock gcc-13 `-march=rv64gcv_xsmtvdotii1p0`。
> **禁 git**·**board 可逆**（vendor `cp *.ORIG` + restore + md5 零改动）·**MANIFEST one-file-per-backtick-bullet**。

---

## 0. 一页速览（verdict）

| 项 | 结果 |
|---|---|
| **#4 权重 repack 桥**（native q4_0 → fragment-major Bnib+dW） | **DONE**·板 UT vs **真 ggml native dequant** byte/value-exact（0 mismatch·随机 3 shape + 真模型 tensor）·MIRAGE 防线过 |
| **#3 激活 quant+pack 桥**（f32 → q8_0 → Apack+dA） | **DONE**·板 UT vs **真 ggml `quantize_row_q8_0_ref`** int8+scale byte-exact（0 mismatch·3 shape） |
| **④ 单 tensor mul_mat A==B**（bridge vs stock q4_0×q8_0） | **DONE（standalone·真硅）**·`max_abs_vs_float_order=0.000e+00`（**float 折叠序 bit-exact** vs stock 块-dot）·normwise ~1e-7 vs double 真值（纯 f32 reassociation）·**随机 5 shape + 真 `token_embd.weight` 全过**·real `vmadot 0xe210312b` |
| **#5 forward hook**（vendor `forward_mul_mat` patch） | **PARTIAL（reachability-proven·非 traffic-routing）**·可逆 probe patch 证 q4_0 PREFILL hook 点在**真 llama-bench forward** fires（banner `gemm_m=64 n=2048 k=2048`·env-gated ON/OFF 对称）·**full 路由 deferred**（w_data=vendor-repacked·须 vendor-layout 适配·next-session） |
| **forward-wired** | **F**（诚实）·hook 点 reachability 已证于真 forward·但我方 fragment-major kernel **未** 路由真流量（vendor repacked-layout 适配 = 独立工作量·next-session） |
| **e2e A==B（单 tensor）** | **T**（standalone·bit-exact float order·real vmadot）·真-llama e2e logits sanity 随 #5 full 路由 deferred |
| **correctness_green** | **T**（#4/#3 byte-exact vs 真 ggml native·full A==B bit-exact float order + ~1e-7 normwise·真硅 vmadot） |
| **perf** | **未测**·跨范式名义·micro↛e2e（vendor 全接线 IME GEMM 同域 toggle 已是 ceiling·recon §4） |
| **双账本 clang 收敛** | k1 出货 clang-18（kernel==system 同数）·本 session correctness 不涉 perf 主张·无对手身份/八门 |
| **板 restored** | **T**·vendor `ime.cpp` md5 `40962c7e…`（==ORIG）· `libggml-cpu.so.0.15.1` md5 `71cc4d29…`（==ORIG）·src/so probe 残留=0·`.ORIG` litter=0·vmadot=32 intact |

**一句话**：**q4_0@ime forward bridge 的两个集成层 correctness-critical 数据路桥（#4 权重 repack + #3 激活 quant/pack）已在 K1 板对【真 ggml native 格式 + 真 ggml 量化器】byte/value-exact 收口（MIRAGE 防线过）·且整条桥的【单 tensor mul_mat A==B】在真硅上对 stock q4_0×q8_0 块-dot 达 float 折叠序 bit-exact（`max_abs_vs_float_order=0`·normwise ~1e-7 vs double 真值=纯 f32 reassociation）·随机 5 shape + 真 `token_embd.weight` 全过**；这把 session-1 的 self-built canonical 升级为**真 ggml 部署格式**的 correctness 契约。**#5 forward hook = reachability-proven（可逆 probe 证 q4_0 PREFILL hook 点在真 llama forward fires）·非 traffic-routing**（`forward-wired=F`·vendor repacked-layout 适配是 next-session）。**correctness 是唯一硬门·已 GREEN·perf 不问。板 stock 零改动（md5 双证）。**

---

## 1. #4 权重 repack 桥（native q4_0 → fragment-major Bnib+dW）

**桥语义**（`g5m3_bridge_ut.c: tcrv_bridge4_repack_q4_0`）：ggml **native** q4_0 权重（row-major·N 行×nb 个 18B block·`{fp16 d; qs[16]}`·`qs[j]` low=w[b·32+j] high=w[b·32+j+16]）→ tcrv fragment-major `Bnib`（col-tile 内 kt 个 18B fragment·nibble idx `(n%4)*8+(k%8)`）+ 抽 per-(col,block) `dW = ggml_fp16_to_fp32(block.d)`。

**MIRAGE 防线（build 前 de-risk·板 UT vs 独立 oracle byte-match）**：
- native 由**真 ggml `quantize_row_q4_0_ref`** 产（非自建）·参考 dequant 由**真 ggml `dequantize_row_q4_0`** 产。
- 判据 = 我方 `Bnib+dW` 解码（decode nibble-8 × dW）== ggml native dequant，**每元素 bit-exact**（同 nibble·同 fp16 d → 同 f32 积·0 ULP）。

```
[bridge #4 weight repack: native q4_0 -> fragment-major Bnib+dW]
  #4 weight-repack N=8  K=64 : dequant 512/512   exact vs ggml native (mism=0 max_abs=0.000e+00)
  #4 weight-repack N=16 K=128: dequant 2048/2048 exact vs ggml native (mism=0 max_abs=0.000e+00)
  #4 weight-repack N=32 K=256: dequant 8192/8192 exact vs ggml native (mism=0 max_abs=0.000e+00)
```
+ 真模型 tensor（§4）再证一次（`token_embd.weight` 的真 native 字节）。

---

## 2. #3 激活 quant+pack 桥（f32 → q8_0 → fragment-major Apack+dA）

**桥语义**（`tcrv_bridge3_quant_pack_q8_0`）：ggml f32 激活 → **真 ggml `quantize_row_q8_0_ref`** 产 native q8_0（`{fp16 d; qs[32]}`·d=amax/127·q=roundf(x/d)）→ tcrv fragment-major `Apack[mi*4*K+kf*32+ml*8+kl]` + 抽 per-(row,block) `dA = ggml_fp16_to_fp32(block.d)`。

**MIRAGE 防线**：我方 `Apack`/`dA` reconstruct == native q8_0 的 int8/scale，**byte-exact**（判 repack 索引无 layout 失配）：

```
[bridge #3 activation quant+pack: f32 -> q8_0 -> fragment-major Apack+dA]
  #3 act-quant+pack M=8  K=64 : int8 512/512   exact (mism=0) | scale 16/16 exact (mism=0)
  #3 act-quant+pack M=12 K=128: int8 1536/1536 exact (mism=0) | scale 48/48 exact (mism=0)
  #3 act-quant+pack M=8  K=256: int8 2048/2048 exact (mism=0) | scale 64/64 exact (mism=0)
```
> 注：因 #3 直接调 ggml `quantize_row_q8_0_ref`，激活量化本身即 ggml-exact（by construction）；UT 证的是 **fragment-major repack 索引**无失配。session-1 已核实 `quantize_row_q8_0_ref` 与 canonical 逐字一致。

---

## 3. ④ 单 tensor mul_mat A==B（唯一硬门·真硅）

**A** = 我方整条桥（native q4_0 →#4 Bnib+dW· f32 →#3 Apack+dA · scale-fold GEMM `tcrv_ime_q4_0_vmadot_matmul_f32` 用 **EMITTER-VERBATIM `vmadot` asm 叶子** on K1）。
**B** = stock q4_0×q8_0 **块-dot**（== `ggml_vec_dot_q4_0_q8_0` 的 stock RVV 算的东西），从**同一批 ggml native 字节**独立重算（异码路径），两种精度：
- `ref_d` = **double** 累加（阶不敏感的真值）。
- `ref_f` = **float** 累加·与 kernel A **同 per-block 折叠序**（判 A 的算术是否与 stock 逐比特一致）。

**判据（诚实·normwise）**：gate = **normwise 相对误差 `max_abs/‖C‖`**（GEMM 精度标准度量·免疫 per-cell 抵消）；per-cell 相对误差 report 但**不 gate**（|C[m,n]|≈0 抵消处无意义）。

```
[FULL single-tensor mul_mat A==B: our bridge (real vmadot) vs stock q4_0xq8_0 block-dot]
  M=8  N=8  K=64 : normwise=5.827e-08 (max_abs=2.449e-07 |C|max=4.203)  max_abs_vs_float_order=0.000e+00 per_cell_rel=1.253e-06 OK
  M=4  N=8  K=96 : normwise=3.912e-08 (max_abs=2.943e-07 |C|max=7.524)  max_abs_vs_float_order=0.000e+00 per_cell_rel=2.289e-06 OK
  M=12 N=16 K=128: normwise=6.161e-08 (max_abs=6.915e-07 |C|max=11.224) max_abs_vs_float_order=0.000e+00 per_cell_rel=8.558e-07 OK
  M=8  N=32 K=256: normwise=7.981e-08 (max_abs=1.214e-06 |C|max=15.210) max_abs_vs_float_order=0.000e+00 per_cell_rel=1.133e-04 OK
  M=64 N=64 K=512: normwise=1.121e-07 (max_abs=3.597e-06 |C|max=32.102) max_abs_vs_float_order=0.000e+00 per_cell_rel=1.804e-03 OK
BRIDGE-UT PASS
```

**load-bearing 读法**：**`max_abs_vs_float_order = 0.000e+00`（所有 shape）= 决定性证据** —— 我方 kernel 的 f32 输出与 stock 块-dot 在**同 float 折叠序下逐比特相同**（不止 bounded-ULP·是 bit-exact）。residual vs double 真值 = normwise ~1e-7·随 K 稳定增长 = **纯 f32 reassociation**（非桥 bug）。K=256/512 的 `per_cell_rel` 涨到 1.8e-3 是**抵消-cell 度量伪影**（`max_abs_vs_float_order=0` 已证算术恒等·`max_abs` 仅 ~1e-6）·**非 correctness 退化**（对比 session-1 gate 曾用 per-cell rel<1e-4·仅因彼 seed 无抵消-cell 而侥幸过·本 session 换 normwise 是修正度量·非放宽门）。

int32 core 的真硅 bit-exact 由 session-1 seal + 本 UT 的 `vmadot` 叶子承载（objdump `vmadot_count=1`·`0xe210312b`）。

---

## 4. 真模型 tensor A==B seal（correctness 升级·非随机）

把 ④ 从随机矩阵升级到**真部署权重字节**：`g5m3_realtensor_ab.c` 用 gguf API 打开 `tinyllama-q4_0.gguf`·抽真 q4_0 2D 权重 `token_embd.weight`（K=2048·N=32000·真 on-disk native 字节·= 真 llama forward 在 vendor repack 前消费的字节）·取 N=64×K=512 sub-tile·经我方整桥 vs stock 块-dot：

```
G5-M3 REAL-MODEL-TENSOR A==B: tensor 'token_embd.weight' [K=2048 N=32000] sub-tile N=64 K=512 M=32 (real vmadot)
  A==B  normwise=1.139e-07 (max_abs=6.083e-08 |C|max=0.534) max_abs_vs_float_order=0.000e+00 per_cell_rel=1.326e-04 OK
REALTENSOR PASS
```
`max_abs_vs_float_order=0` 于真模型字节 → correctness 契约对**真部署数据**成立（非仅合成）。

---

## 5. #5 forward hook（reversible reachability probe·**非 traffic-routing**）

### 5.1 架构裁定：为何 hook 是 reachability-probe 而非 full 路由
vendor `forward_mul_mat`（`ime.cpp:234`）体内 `w_data = src0->data` **已是 vendor-repacked** 权重（`get_optimal_repack_type` 挂的 `tensor_traits<block_q4_0,32,16>` 在 weight load 时把 native→16×32/32×32 interleave·`ime.cpp:1235`）·且激活走 vendor `quantize_a_row_i8`→TCM buffer→`gemm_kernel_i8i4` 多线程 tiling（`ggml_barrier`）。**我方桥用的是 fragment-major Bnib/Apack·与 vendor repacked-layout 不同构** → 在 `forward_mul_mat` 内 drop-in 我方 kernel **须先把 kernel 改成消费 vendor repacked-layout**（独立工作量·recon §2.1 早已定级）。故本 session 的 hook = **reachability probe**（证 hook 点在真 forward reachable）·**非** 把真流量路由过我方 kernel。**`forward-wired=F` 诚实**。

### 5.2 可逆 probe（`forward-probe-patch.py` + `run-forward-probe.sh`）
在 `forward_mul_mat` 的 `gemm_n=ne01` 后插入 **env-gated（`TCRV_IME_BRIDGE_PROBE`·默认 OFF）· q4_0-only（`is_same_v<BLOC_TYPE,block_q4_0>`）· prefill-only（`gemm_m>1`）· one-shot（`ith==0`）** banner。默认 OFF = 零行为改动（ON/OFF 对称）。

**规程**：`cp ime.cpp ime.cpp.ORIG` + `cp libggml-cpu.so.0.15.1 .ORIG` → patch → `make -C build-ime ggml-cpu`（~70s·1 TU+link·warnings-only）→ 跑 llama-bench prefill → **restore（clean source → rebuild clean .o → .so 覆盖回 ORIG binary）** → md5 双证零改动 → 删 `.ORIG` litter。

**结果**（`raw/session2-forward-probe.txt`·板 `use_ime1:1`）：
```
[TCRV-IME-BRIDGE-PROBE] forward_mul_mat q4_0 PREFILL hook reached: gemm_m=64 gemm_n=2048 gemm_k=2048   # env ON
banner_when_off = 0                                                                                     # env OFF（对称）
vmadot_in_rebuilt_so = 32   # vendor IME 完好
```
→ **q4_0 PREFILL hook 点在真 llama-bench forward fires**（真 q4_0 流量·prefill 64 token·hidden 2048）·env-gated 对称·llama-bench clean（pp64 47.06 t/s·无 crash）。

### 5.3 板 restored（md5 双证零 stock 改动）
```
ime.cpp                 md5 40962c7e7c732bf472ae88cef89ced8d  == baseline ✓
libggml-cpu.so.0.15.1   md5 71cc4d295dac29382a0a7d4d5bd0c425  == baseline ✓
src_probe_left=0  so_probe_left=0  .ORIG_litter=0  vmadot_in_so=32
```
restore 用「clean source → make（clean .o）→ .so 覆盖回 ORIG binary」三步 → 未来 `make` 不会从 stale .o 重引入 probe。**stock 零改动**。

---

## 6. 双账本 + 工具链身份

| 轴 | 工具链身份 | 用途 |
|---|---|---|
| kernel-axis（bridge seal 复现） | 板 `gcc-13`（Bianbu 13.2·`-march=rv64gcv_xsmtvdotii1p0`·real vmadot） | #4/#3 UT + full A==B seal（int32-exact + scale-fold） |
| oracle（真 ggml 格式/量化器） | `libggml-base.so`（build-off·RVV stock）符号 `quantize_row_q4_0_ref`/`quantize_row_q8_0_ref`/`dequantize_row_q4_0`/`ggml_fp16_to_fp32`/gguf API | ZERO-MODEL 参考 = 真 ggml native 格式（非自建 canonical） |
| system-axis（板出货·#5 probe rebuild） | **clang-18**（build-ime·SPACEMIT=ON） | forward hook reachability probe（reversible·restored） |

本 session correctness **不涉 perf**·无对手身份·无八门·无账本 perf 主张（纯结构 correctness）。若 next-session #5 full 路由后测 prefill perf → 强制绑 vendor-ceiling 同域披露 + kernel-family-vs-array 分解（recon §4·预注册 expecting parity/非绿·micro↛e2e）。

---

## 7. 诚实边界（partial·多-session·[NG-4]）

- **已闭**：`[GAP-IME-E2E-INTEGRATION]` 的**集成层数据路 correctness**（#4 权重 repack + #3 激活 quant/pack·对真 ggml native byte-exact）+ **整桥单 tensor mul_mat A==B**（bit-exact float order·真硅·真模型字节）。
- **未闭**：**真 forward traffic-routing**（#5 full·我方 fragment-major kernel 须适配 vendor repacked-layout 才能在 `forward_mul_mat` 内取代 `gemm_kernel_i8i4`）+ **真-llama e2e logits sanity**（随 full 路由）。本 session #5 = hook 点 reachability probe（reversible·已证 fires）·**非** 路由。
- **perf 未测**·跨范式名义·`[NG-4]` 全程·**禁"实质胜利/forward 完成"表述**——这是集成层 correctness 收口 + hook reachability·**非** full forward 闭环·**非** perf 赢。

---

## 8. next-session-step（优先序）

1. **#5 full traffic-routing**：把 tcrv IME q4_0 kernel 适配为消费 **vendor repacked-layout**（`tensor_traits<block_q4_0,32,16>` 的 16×32 interleave）+ 激活取 vendor `quant_a_buffer` 布局 → 在 `forward_mul_mat` prefill 分支真替换 `ime1::gemm_kernel_i8i4`（A-tree `cp *.ORIG` 可逆·clang-18 对称）。**或** 建平行 tcrv `tensor_traits`（自有 repack = 我方 fragment-major·绕开 vendor repack）——后者与本 session 桥直接对接（Bnib/Apack 已 byte-exact）。
2. **真-llama e2e A==B + logits sanity**：full 路由后·单 tensor（或全模型 greedy）vs stock RVV（build-off）bounded-ULP + objdump `vmadot` engage + banner FIRES + restore md5 零改动。
3. （perf·须另裁·预注册 parity/非绿）prefill delta·**强制 vendor-ceiling 同域披露 + kernel-family-vs-array 分解 + 对手身份**（recon §4）。

---

## durable files（本 session 新增·见 MANIFEST append）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/g5m3_bridge_ut.c`（#4/#3 UT + full A==B·真 ggml native oracle）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/g5m3_realtensor_ab.c`（真模型 tensor A==B）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-bridge-ut.sh` · `run-forward-probe.sh` · `forward-probe-patch.py`（reversible）
- `raw/session2-bridge-ut.txt` · `raw/session2-realtensor-ab.txt` · `raw/session2-forward-probe.txt`
