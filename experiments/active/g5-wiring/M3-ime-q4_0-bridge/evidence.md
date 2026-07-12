# G5-M3 — IME e2e forward bridge 曳光弹（q4_0@ime · Line-B · k1）

> **campaign**: G5 接线战役 · **M3-ime-q4_0-bridge = IME forward bridge 曳光弹**（用户裁：批准·**跨范式完整性名义 C1·N2 family#2 forward**·非 perf）
> **名义**: correctness-first 结构成就。成功判据 = **correctness GREEN（scale-fold vs 独立 oracle + 真硅 seal）**，闭 `[GAP-IME-E2E-INTEGRATION]` 的第一层。**不是 perf 数字**（micro↛e2e 铁律·vendor 全接线 IME GEMM 同域 toggle 已证无干净 e2e 赢）。
> **board**: `ssh k1`（SpacemiT X60·VLEN256·IME on harts 0–3·`taskset -c 0-3`）·出货 clang-18 → **kernel-axis ≠ system-axis**（双账本）。
> **session**: 曳光弹 ≥3 session 的 **session 1**。本 session = scale-fold epilogue（bridge #1）+ 单/多 shape 泛化（bridge #2）+ 对 oracle correctness（host + K1 硅 seal）+ emitter 实发。**partial·多-session**（bridge #3/#4/#5 forward-hook 见 next-step）。

---

## 0. 一页速览（verdict）

| 项 | 结果 |
|---|---|
| **bridge #1 scale-fold epilogue** | **DONE**（emitter 实发 `tcrv_ime_q4_0_vmadot_matmul_f32`·逐 32-block `d_a*d_w*int32partial` f32 折叠）·host + K1 硅 correctness GREEN |
| **bridge #2 shape driver（runtime 泛化）** | **DONE（单 tensor 级）**·runtime M/N/K·四 shape 板测（8×8×64 / 4×8×96 / 12×16×128 / 8×32×256）全过 |
| **bridge #3 激活 quant + Apack pack** | **PARTIAL**（golden 内自建 canonical q8_0 量化 + fragment-major pack·验证正确）·真 ggml `quantize_a_row` 桥 = next-session |
| **bridge #4 权重 repack（native→fragment-major）** | **PARTIAL**（golden 内自建 repack + int32-core 验证 packing 正确）·真 ggml native q4_0 → fragment-major 板侧桥 = next-session |
| **forward hook（#5 patch `forward_mul_mat`）** | **NOT STARTED**（next-session·A-tree 可逆 patch + 真 e2e A==B vs stock RVV） |
| **correctness_green** | **T**（int32 core bit-exact on real vmadot·scale-fold max_rel ~1e-5 < 1e-4 tol·host==board 同数） |
| **forward-wired** | **F**（未接 `forward_mul_mat`·standalone kernel + oracle/seal only·诚实：结构桥的 correctness 层已闭·集成层未闭） |
| **perf** | **未测**（本 session 纯 correctness·跨范式名义·perf 预注册 parity/非绿·见 recon §4 vendor-ceiling 同域披露） |
| **板 restored** | **T**（未改板 vendor 源·仅 `/tmp/g5m3` 编译 scratch·板 `ggml/` 零触碰·read-only 对照锚） |

**一句话**：**q4_0@ime forward bridge 的两处 correctness-critical 层（#1 scale-fold epilogue + #2 runtime shape）已在 host 与真 K1 硅上 correctness GREEN（int32 core bit-exact via real `vmadot` 0xe210312b·scale-fold 对 canonical q4_0×q8_0 ZERO-MODEL reference max_rel ~1e-5）·且 emitter 实发（新 `..._f32` 导出 wrapper·lit golden）**；这是 `[GAP-IME-E2E-INTEGRATION]` **第一层（kernel→f32 forward-ready 语义）的结构闭环**·**非** 全 forward 闭环（#3/#4 真 ggml 桥 + #5 hook 是 next-session）。**perf 未测·跨范式完整性名义·micro↛e2e 全程。**

---

## 1. 做了什么（bridge #1 + #2）

### 1.1 核心不对称回顾（recon §2.1）
tcrv IME emitted kernel 原状 = **int32 累加器微 tile 叶子**（seal `q4-0-matmul-tile-int32-k1seal.c`·M/N/K baked·无 per-block scale）。真 ggml q4_0 mul_mat forward 需 **逐 32-block d_a×d_w f32 折叠**（ggml q4_0×q8_0 dot = `Σ_block d_a·d_w·Σ_{k∈block} qa[k]·qw[k]`）。这一折叠**当前 emitter 不 emit**（原注释 `IMEBackendEmissionDriver.cpp` 明示 scale fold 是下游 epilogue）。

### 1.2 scale-fold kernel 设计（复用 sealed core·仅加 f32 fold）
新 helper `tcrv_ime_q4_0_vmadot_matmul_f32`：
- **复用 sealed 件 verbatim**：`tcrv_ime_q4_0_dequant_fragment`（offset-binary nibble 解码）+ `tcrv_ime_vmadot_mac_kloop`（register-resident batched vmadot·M1b 真硅 0-diff）。
- **仅新增算术** = 每 4×4 输出 tile · 每 32-K block（= 4 fragments）：MAC → int32 partial（sealed int32-EXACT）→ `Cf[m][n] += dA[m*nb+b]·dW[n*nb+b]·partial`。
- **layout**：weight nibbles = fragment-major 18-byte 块（**与 int32 seal 同布局**·2-byte d 槽仅 provenance）；per-(column,block) fp16 `d_w` 走并行 `dW[n*nb+b]`；per-(row,block) `d_a` 走 `dA[m*nb+b]`。
- **runtime shape**（bridge #2）：M/N/K = runtime 参数（固定微 tile 泛化到真 shape）。

### 1.3 emitter 实发（lib/ 扩展·非仅 harness）
`lib/Plugin/IME/IMEBackendEmissionDriver.cpp`：
- 新 `q40ScaleFoldMatmulHelperBody(macKloopHelperName)` 生成上述 f32 kernel（byte-identical 于 host/board harness）。
- `IMEQ40MatMulTileToEmitCFunc` 追加 **第二个 `extern "C"` 导出 wrapper** `tcrv_emitc_<kernel>_<variant>_f32(const int8_t* Apack, const float* dA, const uint8_t* Bnib, const float* dW, float* Cf)`（M/N/K baked·forward hook 调用点）。**int32 wrapper 保持不变**（seal object 不动·existing lit/seal/oracle 全绿保持）。
- 与既有 **q4_K emitter 已 emit f32 fold**（`q4KMatmulHelperBody` 写 `Cf`）体例一致——**非 canon 变更**·q4_0 对齐 q4_K。

---

## 2. correctness 证据（唯一硬门）

### 2.1 host oracle（`test/Target/IME/q4-0-matmul-tile-scalefold-oracle.c`·gcc host）
ZERO-MODEL：reference 用 canonical ggml q4_0 权重量化 + q8_0 激活量化 of **原始 f32**·plain triple-loop + per-block fold（不碰 packed bytes·异码路径）。
- Check(1) int32 core：kernel per-block int32 partial == 独立 partial（from packed bytes·**int32-EXACT**）→ 验 decode + MAC + **repack/pack 正确性**。
- Check(2) scale fold：kernel f32 == canonical reference（bounded-ULP·仅 f32 reassociation）。

```
G5-M3 q4_0@ime scale-fold epilogue oracle (bridge #1 fold + #2 runtime shape)
  shape M=8  N=8  K=64  nb=2: int32-core 128/128  exact (mism=0) | scalefold max_abs=2.338e-07 max_rel=2.427e-06 (|C|max=5.685)
  shape M=4  N=8  K=96  nb=3: int32-core 96/96    exact (mism=0) | scalefold max_abs=4.438e-07 max_rel=2.667e-07 (|C|max=8.929)
  shape M=12 N=16 K=128 nb=4: int32-core 768/768  exact (mism=0) | scalefold max_abs=1.044e-06 max_rel=5.287e-06 (|C|max=12.054)
  shape M=8  N=32 K=256 nb=8: int32-core 2048/2048 exact (mism=0) | scalefold max_abs=2.295e-06 max_rel=1.083e-05 (|C|max=24.929)
ORACLE PASS: ... exit=0
```

### 2.2 K1 硅 seal（`test/Target/IME/q4-0-matmul-tile-scalefold-k1seal.c`·真 `vmadot`）
= oracle 逐字·唯一差异 = scalar-substitute MAC → **EMITTER-VERBATIM `vmadot` asm 叶子**（macKloopHelperBody·signed vmadot）。build `gcc-13 -march=rv64gcv_xsmtvdotii1p0`·objdump 证 `vmadot v2,v0,v1 = 0xe210312b`·`taskset -c 0-3`。

```
objdump: 1103e: e210312b  vmadot v2,v0,v1   (+ 111d6 同)
G5-M3 q4_0@ime scale-fold epilogue K1 SEAL (real vmadot; bridge #1 fold + #2 runtime shape)
  M=8  N=8  K=64  nb=2: int32-core 128/128  exact | scalefold max_rel=2.427e-06
  M=4  N=8  K=96  nb=3: int32-core 96/96    exact | scalefold max_rel=2.667e-07
  M=12 N=16 K=128 nb=4: int32-core 768/768  exact | scalefold max_rel=5.287e-06
  M=8  N=32 K=256 nb=8: int32-core 2048/2048 exact | scalefold max_rel=1.083e-05
SEAL PASS: ... on K1 (real-vmadot int32 core exact, fold within f32 tol) ... exit=0
```

**host == board 逐数一致**（int32 core 真硅 bit-exact·scale-fold max_rel 同）→ ZERO-MODEL int32-exact 契约在真 `vmadot` 上成立·scale-fold f32 语义正确。

### 2.3 emitter lit golden（`test/Conversion/EmitC/ime-q4-0-matmul-tile-materialization.mlir`）
扩展 EMITC 检查：int32 kernel+wrapper → 新 `scale_fold_epilogue=tcrv_ime_q4_0_vmadot_matmul_f32` verbatim + `Cf[...] += dA[...]*dW[...]*(float)frag[...]` → 第二 wrapper `..._slice_f32` + `call_opaque "tcrv_ime_q4_0_vmadot_matmul_f32"`。**REGION + EMITC 全绿**（clean-rebuild `build-ime-bridge/bin/tcrv-opt`·系统 LLVM-20·`FileCheck`）·**q8_0/q4_K/mma/mma_u/mma_su/mma_us/mma_slide/matmul 回归全绿**（共享 emitter 文件行为不变）。

**emitted-golden == harness 契约**：tcrv-opt 实发的 `tcrv_ime_q4_0_vmadot_matmul_f32` kernel body 与 host/board harness 的同名函数 **token-identical**（仅注释/换行差·`raw/emitted-f32-helper.c` vs `raw/harness-f32-helper.c`）→ K1 硅 seal 验证的正是 compiler 实发的字节。

---

## 3. 双账本 + 工具链身份（[CASE-COMPILER-ASYMMETRY] 分账）

| 轴 | 工具链身份 | 用途 |
|---|---|---|
| **kernel-axis（tcrv emit 复现）** | host `cc`（gcc 11.4·oracle）/ 板 `gcc-13`（Bianbu 13.2·`-march=rv64gcv_xsmtvdotii1p0`·seal 真硅） | correctness seal（int32 + scale-fold） |
| **system-axis（板出货）** | **clang-18**（板 `build-ime` SPACEMIT=ON·`-fno-integrated-as`） | 若 forward hook 部署（next-session）→ 出货 = clang .o 正门·须双账本披露 |
| **host emit build** | 系统 LLVM-20 + `c++`（build-ime-bridge·独占·clean·**绝不动 `build/`**） | emitter 实发验证（tcrv-opt lit） |

**本 session correctness 不涉 perf·无对手身份·无八门·无账本 perf 主张**（纯结构 correctness）。**若 next-session 测 prefill perf → 强制绑 vendor-ceiling 同域披露 + kernel-family-vs-array 分解 + 对手身份 + 编译器对称性**（recon §4）。

---

## 4. 诚实边界（partial·多-session·[NG-4]）

- **已闭**：`[GAP-IME-E2E-INTEGRATION]` 的 **kernel→f32-forward-ready 语义层**（scale-fold + runtime shape·真硅 correctness）+ emitter 实发。
- **未闭**：**真 forward 闭环**（#3 真 ggml 激活 quant 桥 / #4 native q4_0 → fragment-major 权重 repack 板侧桥 / #5 `forward_mul_mat` hook + e2e A==B vs stock RVV）。本 session golden 内的 #3/#4 是**自建 canonical 版**（验证 scale-fold 算术正确）·**非** 真 ggml 数据路径接入。
- **perf 未测**·预注册 parity/非绿（micro↛e2e·vendor toggle 已是 ceiling proxy）。**禁"实质胜利/目标达成"表述**——这是结构桥 correctness 第一层·**非** e2e forward 完成·**非** perf 赢。

---

## 5. next-session-step（优先序）

1. **#4 权重 repack 板侧桥**：真 ggml native q4_0 weight（row-major 32-K/列块）→ tcrv fragment-major Bnib + dW 抽取·板 UT vs 独立 oracle byte-match（build 前 de-risk·MIRAGE trap 防线）。
2. **#3 激活 quant 桥**：ggml f32 激活 → q8_0-style per-block d_a + int8 → tcrv fragment-major Apack + dA（对标 vendor `quantize_a_row_i8` 语义·但走 tcrv pack）。
3. **#5 forward hook**：patch `spacemit/ime.cpp` `forward_mul_mat` prefill `gemm_m≥M*` 分支（或平行 tcrv `tensor_traits`）·`#include` emitted `..._f32.inc`·**A-tree `cp *.ORIG` 可逆**·对称 ON/OFF build（clang-18）。
4. **e2e correctness 门**：单 tensor mul_mat 经 IME 桥 vs stock RVV-fallback **A==B（bounded-ULP·f32 路）** + logits sanity + objdump `vmadot` engage + banner FIRES + **测后 restore + md5 证零 stock 改动**。
5. （可选·perf 名义须另裁）prefill delta·**强制 vendor-ceiling 同域披露 + 分解**。

---

## durable files
- `evidence.md`（本文）
- `raw/` — host + board 原始输出
- `MANIFEST.md` · `.gitignore`
