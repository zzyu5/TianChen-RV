# G5-M2 q5_1 曳光弹 — evidence (L-接线② 净新 scaffold · 复用 q5_0 recipe + q8_1 min 家族)

> workflow (2026-07-12) · board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree HEAD f3e1828 ·
> **HEAD (TianChen-RV)** = d92a706d · 禁 git · **A-tree 测后 restore 验 clean**（源回 baseline md5 · live .so 回 05a62e6a OFF-pristine）。
> **结论 = GREEN-5/84**：**q5_1 净新 scaffold BUILT + correctness GREEN + perf GREEN prefill 1.0900×（§七）→ perf-covered 4/84→5/84** → 复用 q5_0 recipe 成功、**并净新建 q8_1 activation mat-quantizer**（上游只有 Q8_0/Q8_K）= q8_1-activation repack 家族首格。q5_1 = q5_0 的 qh gather（unsigned/无 -16）+ q4_1 的 min fold + q8_1 激活。decode 0.7836× regression 如实披露。

## 一、recon：q5_1 = 真零 present（同 q5_0）+ q8_1 mat-quant 缺口（★超 q5_0 的净新）

板 A-tree read-only（baseline md5 GEN=`deb61a29` / HDR=`57851439` / ARCH=`99131cf7` = 干净 WinB-q4_0-ON）：
- **GEN+ARCH q5_1 riscv repack grep = 0**（同 q5_0：block<K,N> 无法表达第 5 位 qh + q5_1 的 min）→ 必须净新建。
- **★额外缺口（q5_1 独有，超 q5_0）**：q5_1 激活 param type = **GGML_TYPE_Q8_1**（非 q5_0 的 Q8_0）。板上：
  - `block_q8_1x4` = **0**（不存在；`block<8,4>` 模板只有 d[N]+qs[]、无 q8_1 的 running-sum `s` 字段）。
  - `ggml_quantize_mat_t<1, GGML_TYPE_Q8_1>` = **0**（上游只有 `<*,Q8_0>` / `<*,Q8_K>`；GEMM 路径 `ggml_quantize_mat_t<INTER_SIZE,PARAM_TYPE>` 对 Q8_1 会 **undefined-reference → link fail**）。
  - GEVM 路径 `from_float = ggml_get_type_traits_cpu(Q8_1)->from_float = quantize_row_q8_1` **已注册**（core ggml 类型），无需新建。
  - 无 q4_1 repack（q4_1 同族 q8_1、上游亦无 repack）→ q5_1 是 **q8_1-activation repack 家族首格**。

## 二、kernel emit（host·vl=8 VLEN128-safe·zero 16/64·min fold·unsigned）

host `./build/bin/tcrv-opt`（LLVM20.1.8）+ `/usr/lib/llvm-20/bin/mlir-translate`，源 fixture 已存（无需新写）：
- **GEMM (prefill)** 源 `test/Conversion/RVV/rvv-emit-quant-contraction-q5-1-repack-gemm-prefill-vlen128.mlir`
  → `tcrv_emitted_gemm_q5_1.inc`（30170B·md5 **0f52ad46**）·func `tcrv_emitc_ggml_gemm_q5_1_q8_1_kernel_ggml_gemm_q5_1_q8_1`
  ·ABI `(size_t nr, size_t bs, size_t n, float* s, size_t nc, const uint8_t* vx, const uint8_t* vy)`
- **GEVM (decode)** 源 `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-1-repack-vlen128.mlir`
  → `tcrv_emitted_gevm_q5_1.inc`（21077B·md5 **84b69d60**）·func `tcrv_emitc_ggml_vec_dot_q5_1_q8_1_kernel_ggml_vec_dot_q5_1_q8_1`
  ·ABI `(size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc)`
- **★源级/板级 seal**：两 .inc `, 16)` `, 64)` 计数 **0**、AVL 常量列 = 8；weight stride **384**（d[16]@0·m[16]@32·nibbles@64·transposed-qh@320）、act stride **144**（block_q8_1x4：d[4]@0·s[4]@8·qs[128]@16）/ q8_1 single 36（GEVM）。
  - **UNSIGNED 5-bit**：`vsub_vx_i8` 计数 **0**（q5_0 为 4/8）= 无 offset-binary -16 偏置。
  - **min fold**：`vfadd_vv_f32m2` present（GEMM 24·GEVM 12）= `isum·d_w·d_a + m_w·s_a`。
  - 板 objdump（build_seal §3）：两 emitted 符号 `vsetivli imm=8=1 · imm=16=0 · imm=64=0`；SEW,LMUL 全 mf2/m1/m2（zero f32m4/whole-register/redsum）。

## 三、净新 scaffold（复用 q5_0 recipe + q5_1 三 delta·3 文件 tracked·可逆）

deploy `tools/e2e-harness/board/g5-m2-q5_1/deploy_patch_q5_1_emitted.py`（baseline-gated·同 q5_0 触碰集 GEN+HDR+ARCH 3 文件，**无 arch-fallback.h 改动**）。相对 q5_0 recipe 的 delta：
1. **weight block 加 m 字段**：`struct block_q5_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; uint8_t qh[64]; }`（384B·m@32·qh@320）。
2. **★净新 q8_1 activation 基建**（q5_1 独有）：
   - `struct block_q8_1x4 { ggml_half d[4]; ggml_half s[4]; int8_t qs[128]; }`（144B·亦不可用 block<K,N>，需 s 字段）。
   - `template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_1>`（inline body·避 arch-fallback.h 编辑）—— **byte-match STOCK `quantize_row_q8_1_ref`**：`d=amax/((1<<7)-1)`·`roundf`·`s = sum·d`（原始 float d，非 fp16 roundtrip）·interleave=1 → `qs[e*4+m]`·per-row sum。
3. **★correctness-critical = make_block_q5_1x16 interleaver**：d/m byte-offset memcpy（block_q5_1 用 C++ AGGR union `.data.data.d`，故 raw-offset 读 d@0/m@2）·qs interleave=1 无 xor·qh transpose（qh_lo[k]@0 / qh_hi[k]@32，bit c = 5th bit of elem k / k+16 of col c）—— 与 q5_0 qh 转置同构、仅 struct 把 qh 放 @320。
- 其余机械件（repack fn/template、gemv/gemm template、scalar generics【unsigned+min】、trait 注册、dispatch case128/256=ON、ARCH VLEN128 intercept + banner + emitted call）逐件镜像 q5_0。

**★净新方法学产出（C1 extensibility 强化）**：q5_0 证 "template 建 upstream repack scaffold"；**q5_1 更进一步 = template 建 upstream scaffold + 净新一整条 q8_1 activation 量化基建**（block_q8_1x4 + mat-quantizer），打通 q8_1-activation repack 家族（q4_1/q5_1）首格。

## 四、UT 独立 oracle（★MIRAGE de-risk·board VLEN128·build 前先验）

`ut_q5_1_interleaver.cpp`（GEVM）+ `ut_q5_1_gemm.cpp`（GEMM）—— make_block_q5_1x16 + emitted kernel vs 独立标量 q5_1 oracle（unsigned decode + q4_1 min + q8_1 s）：
- **interleaver/GEVM GREEN**：fails=0/16·max_rel **4.99e-07**（bit-exact）。
- **GEMM/prefill GREEN**：fails=0/64·max_rel **3.35e-06**（block_q8_1x4 布局 + min 正确）。
→ interleaver m@32 + qh@320 transpose + min fold + q8_1x4 activation packing 全 byte-exact 于独立 oracle，MIRAGE trap 关。

## 五、build + seal（gcc-15.2.0 对称·OFF/ON·objdump vl=8）

`g5_m2_q5_1_build_seal.sh`（seal_raw.txt）：
- OFF build md5=`05a62e6a`·q5_1_tcrv_sym=**0**（pristine，与全 baseline 同）。
- ON build md5=`38377d3a`·q5_1_tcrv_sym=**2**（gemm + vec_dot 在）·banner gevm=1/gemm=1·ON≠OFF。
- objdump vl-seal：两 emitted 符号 `vsetivli imm=8=1 · imm=16=0 · imm=64=0`（VLEN128-safe·部署==证过）。
- **restore**：3 源回 baseline byte-exact（deb61a29/57851439/99131cf7）·pristine rebuild·live .so=`05a62e6a` 0 syms。

## 六、correctness GREEN（★hard gate·A==B byte-identical·真模型）

`g5_m2_q5_1_correctness.sh`（correctness_GREEN_raw.txt）· model=DeepSeek-R1-Distill-Llama-8B-Q5_1.gguf（sha256 5ffdc7c5…·自 Q5_0 requantize Q5_1 `--allow-requantize`·决策卡④·任意合法 q5_1 gguf 皆可，A/B 读同权重）：
- ⟨见 correctness_GREEN_raw.txt / §八 verdict⟩ prompt byte-identical A(emit)==B(stock) coherent·emitted-kernel banner fires·no NaN/Inf·PPL 有限 coherent。

## 七、perf 分相（★clean·2026-07-12·board rvv openEuler VLEN128 gcc-15.2.0·correctness GREEN 后）

> 同树物理 .so swap（ours=q5ON `38377d3a` 2 syms / stock=q5OFF `05a62e6a` 0 syms）·ONE llama-bench·gcc-15.2.0 双侧对称·唯一 A/B diff = q5_1 净新 dispatch gate + emitted kernels + q8_1 mat-quant。
> model=DeepSeek-R1-Distill-Llama-8B-Q5_1.gguf（sha256 5ffdc7c5…）·taskset 8-15 · 8 threads · DVFS performance 锁 2.6GHz（4 ###AB freq_khz 全 2600000）· **T-N: PASSES=2×REPS=10 = n=20/side/phase** · engage-probe = 5 fires（部署-during-e2e 实证·p16n8）。

**结果（analyze_phase_split.py·中位·relIQR 卫生·n=20）**：

| phase | ours(q5ON) t/s | stock(q5OFF) t/s | ratio | ours relIQR | stock relIQR | n | 判 |
|---|---|---|---|---|---|---|---|
| **prefill pp128** | **3.462485** | 3.176470 | **1.0900×** | 0.118% | 0.474% | 20 | **≥parity · WIN +9.0%** |
| decode tg32 | 1.228230 | 1.567385 | 0.7836× | 0.419% | 3.018% | 20 | regression −21.6%（memory-bound GEVM wash） |

per-pass 岩石稳（ours pp pass1 3.4644/pass2 3.4597·tg 1.2304/1.2250；stock pp pass1 3.1841）·全 relIQR 远低地板×1.5·freq 全程 2.6GHz。

**★verdict = GREEN-5/84**（perf-covered 4/84→5/84·FLAT 净新 scaffold 第二格 perf 绿·q8_1-activation repack 家族首格）。判据 = 预注册规则「prefill ≥parity 过 T-N+八门 → 绿」，prefill 1.090× 清晰 ≥parity。★kernel-axis micro 1.41× 【部分传导】到 e2e prefill 1.090×（比 q5_0 的 1.21× 略小、仍清晰 win；min fold + q8_1 mat-quant 开销吃掉部分 margin，但净 WIN）。

**八门状态**（全过·prefill 轴）：① 同树同 binary 物理 .so swap ✓ ② 部署验 nm ON=2/OFF=0 syms ✓ ③ banner engage 5 fires 真模型 p16n8 ✓ ④ objdump vl=8 seal（never 16/64）✓ ⑤ 对手身份 = stock q5_1 block-dot 同树（非 SELF）✓ ⑥ 编译器对称 gcc-15.2.0 双侧（board shipped rv64gcv=gcc-15 → kernel==system）✓ ⑦ correctness GREEN 前置（5/5 byte-identical A==B·PPL 18.84）✓ ⑧ T-N 卫生 n=20·relIQR≪地板·DVFS 锁 2.6GHz·freq 全稳 ✓。

**双账本**：board rv64gcv 出货工具链 == gcc-15 ⇒ **kernel-axis == system-axis**（两账本数值同 1.0900×/0.7836×，均有效；对照 [CASE-COMPILER-ASYMMETRY] k1=clang-18 才需分账）。opponent identity = stock ggml q5_1 block-dot（同树·仅 scaffold diff）。

**诚实 caveat**：decode −21.6% 是 memory-bound GEVM 回退（repack 布局利 prefill GEMM locality、损 decode 带宽），与 q5_0（−18%）/q4_0 WinB 同型 prefill-win/decode-wash split。kernel+e2e / prefill+decode 永分报（[[kernel-wins-dont-transplant-to-e2e]]）。perf-covered 绿基于 prefill ≥parity 公平协议格。

## 八、verdict

**★GREEN-5/84**：correctness GREEN（5/5 byte-identical A==B·PPL 18.84·UT 双 GREEN·objdump vl=8） ∧ prefill 1.0900× ≥parity（T-N n=20·八门全过·gcc-symmetric）→ **perf-covered 4/84 → 5/84**。decode 0.7836× regression 如实披露（FLAT prefill-win/decode-wash·同 q5_0 先例）。
**方法学**：q5_1 = 复用 q5_0 净新 scaffold recipe + 净新 q8_1 activation mat-quantizer（block_q8_1x4 + `ggml_quantize_mat_t<1,Q8_1>`，上游只有 Q8_0/Q8_K）= **打通 q8_1-activation repack 家族（q4_1/q5_1）首格** · C1 template extensibility 再实证。
**board**：测后 restore 验 clean（源 baseline byte-exact·live .so 05a62e6a 0 syms·无 .inc stray·DVFS performance 2.6GHz）。

## durable files
- `evidence.md`（本文）
- `tcrv_emitted_gemm_q5_1.inc`（md5 0f52ad46·30KB）· `tcrv_emitted_gevm_q5_1.inc`（md5 84b69d60·21KB）
- `seal_raw.txt`（build+patch+seal 全 log）
- `objdump_gemm_q5_1_seal.txt` · `objdump_gevm_q5_1_seal.txt`（vl=8 seal）
- `correctness_GREEN_raw.txt` · `phase_split_raw.txt` · `transmission_accounting.csv`（后续落盘）
> 两 .inc 小（51KB 合计）保留 tracked；emit recipe 见 §二可再生。board harness 住 `tools/e2e-harness/board/g5-m2-q5_1/`。
