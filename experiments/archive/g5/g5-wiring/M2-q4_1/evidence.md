# G5-M2 q4_1 曳光弹 — evidence (L-接线② 净新 scaffold · 复用 q5_1 q8_1-activation 家族素材)

> workflow (2026-07-12) · board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree HEAD f3e1828 ·
> **HEAD (TianChen-RV)** = daf6db23 · 禁 git · **A-tree 测后 restore 验 clean**（源回 baseline md5 · live .so 回 05a62e6a OFF-pristine）。
> **结论 = GREEN-6/84**：**q4_1 净新 scaffold BUILT + correctness GREEN + perf GREEN prefill 3.6844×（§七）→ perf-covered 5/84→6/84** → 复用 q5_1 净新 scaffold recipe + **直接复用 q8_1 sub-scaffold**（block_q8_1x4 + `ggml_quantize_mat_t<1,Q8_1>`）= **q8_1-activation repack 家族第 2 成员**。q4_1 = q5_1 减去 5th-bit qh（更简单）= q4_0 nibble decode + min fold + q8_1 激活。**decode 亦 WIN 1.6668×（atypical·§七诚实 caveat）**。

## 一、recon：q4_1 = 真零 present（同 q5_0/q5_1·absent 需净新）

板 A-tree read-only（baseline md5 GEN=`deb61a29` / HDR=`57851439` / ARCH=`99131cf7` = 干净 WinB-q4_0-ON）：
- **GEN+HDR+ARCH q4_1 riscv repack grep = 0**（同 q5_0/q5_1：block<K,N> 无法表达 q4_1 的 min）→ 必须净新建（**非** q4_0/q8_0 的翻-gate 模式）。
- q4_1 激活 param type = **GGML_TYPE_Q8_1**（同 q5_1 家族）：板上 `block_q8_1x4` = 0、`ggml_quantize_mat_t<1,Q8_1>` = 0（上游只 Q8_0/Q8_K）→ 需 q8_1 sub-scaffold（**直接复用 q5_1 已建的**）。GEVM 路径 `from_float=quantize_row_q8_1` 已注册。
- **q4_1 比 q5_1 简单**：无第 5 位 qh → weight block 无 qh 字段（stride 320 vs q5_1 384）、interleaver 无 qh 转置、decode 纯 nibble（无 vor 5th-bit）。

## 二、kernel emit（host·LLVM20.1.8·vl=8 VLEN128-safe·zero 16/64）

host `./build/bin/tcrv-opt` + `/usr/lib/llvm-20/bin/mlir-translate`，fixtures 已存（无需新写）：
- **GEMM (prefill)** 源 `test/Conversion/RVV/rvv-emit-quant-contraction-q4-1-repack-gemm-prefill-vlen128.mlir`
  → `tcrv_emitted_gemm_q4_1.inc`（25520B·md5 **31912b13**）·func `tcrv_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1`
  ·ABI `(size_t nr, size_t bs, size_t n, float* s, size_t nc, const uint8_t* vx, const uint8_t* vy)`
- **GEVM (decode)** 源 `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q4-1-repack-vlen128.mlir`
  → `tcrv_emitted_gevm_q4_1.inc`（13610B·md5 **d645aa42**）·func `tcrv_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1`
  ·ABI `(size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc)`
- **★源级 seal**：两 .inc `, 16)` `, 64)` 计数 **0**·setvl 为 dynamic `vsetvl_e32m1`·plain nibble decode（`& 15` / `>> 4`·无 qh vor）。weight stride **320**（d[16]@0·m[16]@32·nibbles@64·无 qh）·act stride **144**（block_q8_1x4：d[4]@0·s[4]@8·qs[128]@16）/ q8_1 single 36（GEVM·s@2）。
- **板 objdump seal**（build_seal §3）：两 emitted 符号 `vsetivli imm=8=1 · imm=16=0 · imm=64=0`（VLEN128-safe·部署==证过）·SEW,LMUL 全 e8mf2/e16m1/e32m2（zero whole-register/f32m4/redsum）。

## 三、净新 scaffold（deploy_patch_q4_1_emitted.py·3 文件 tracked·可逆·无 arch-fallback.h）

复用 q5_1 净新 scaffold recipe + q4_1 简化 delta：
1. **weight block（无 qh）**：`struct block_q4_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; }`（320B·m@32·nibbles@64·**无 qh[64]**·比 q5_1 少 64B）。
2. **★q8_1 activation 基建（直接复用 q5_1 净新）**：`struct block_q8_1x4 { ggml_half d[4]; ggml_half s[4]; int8_t qs[128]; }`（144B）+ inline `ggml_quantize_mat_t<1,GGML_TYPE_Q8_1>`（byte-match stock `quantize_row_q8_1_ref`）。**这是 q5_1 打通的家族基建、q4_1 零改动复用** = 家族续铺的直接证据。
3. **★correctness-critical = make_block_q4_1x16 interleaver**：d/m byte-offset memcpy（block_q4_1 AGGR union·raw d@0/m@2）·qs interleave=1 无 xor（unsigned nibble）·**无 qh 转置**（比 q5_1 更简单）。
- 其余机械件（repack fn/template、gemv/gemm template、scalar generics【unsigned nibble + min·无 qh】、trait 注册、dispatch case128/256=ON、ARCH VLEN128 intercept + banner + emitted call）逐件镜像 q5_1。

**★方法学产出（C1 extensibility 续铺）**：q5_1 建 q8_1-activation 家族首格 + 净新 q8_1 量化基建；**q4_1 = 该家族第 2 成员、零改动复用 q8_1 sub-scaffold**（仅换 weight interleaver、去 qh）→ 实证「一次建家族基建、后续成员边际成本递减」（C2 模板经济学证词）。

## 四、UT 独立 oracle（★MIRAGE de-risk·board VLEN128·build 前先验）

`ut_q4_1_interleaver.cpp`（GEVM）+ `ut_q4_1_gemm.cpp`（GEMM）—— make_block_q4_1x16 + emitted kernel vs 独立标量 q4_1 oracle（unsigned nibble decode + min + q8_1 s）：
- **interleaver/GEVM GREEN**：fails=0/16·max_rel **1.974e-06**（bit-exact）。
- **GEMM/prefill GREEN**：fails=0/64·max_rel **6.364e-06**（block_q8_1x4 布局 + min 正确）。
→ interleaver（m@32·无 qh）+ min fold + q8_1x4 activation packing 全 byte-exact 于独立 oracle，MIRAGE trap 关。

## 五、build + seal（gcc-15.2.0 对称·OFF/ON·objdump vl=8）

`g5_m2_q4_1_build_seal.sh`（seal_raw.txt）：
- OFF build md5=`05a62e6a`·q4_1_tcrv_sym=**0**（pristine，与全 baseline 同）。
- ON build md5=`a3774cdf`·q4_1_tcrv_sym=**2**（gemm + vec_dot 在）·banner gevm=1/gemm=1·ON≠OFF。
- objdump vl-seal：两 emitted 符号 `vsetivli imm=8=1 · imm=16=0 · imm=64=0`（VLEN128-safe·部署==证过）。
- **restore**：3 源回 baseline byte-exact（deb61a29/57851439/99131cf7）·pristine rebuild·live .so=`05a62e6a` 0 syms。

## 六、correctness GREEN（★hard gate·A==B byte-identical·真模型）

`g5_m2_q4_1_correctness.sh`（correctness_GREEN_raw.txt）· model=DeepSeek-R1-Distill-Llama-8B-Q4_1.gguf（sha256 e10c9d9936fa1901·自 Q5_0 requantize Q4_1 `--allow-requantize`·决策卡④·A/B 读同权重）：
- **5/5 prompt byte-identical A(emit)==B(stock)**·emitted-kernel banner **44 fires**·no NaN/Inf·**PPL(ON)=22.0523** 有限 coherent（<1000）→ CORRECTNESS_GATE: GREEN。

## 七、perf 分相（★clean·2026-07-12·board rvv openEuler VLEN128 gcc-15.2.0·correctness GREEN 后）

> 同树物理 .so swap（ours=q4ON `a3774cdf` 2 syms / stock=q4OFF `05a62e6a` 0 syms）·ONE llama-bench·gcc-15.2.0 双侧对称·唯一 A/B diff = q4_1 净新 dispatch gate + emitted kernels + q8_1 mat-quant。
> model=DeepSeek-8B-Q4_1.gguf·taskset 8-15 · 8 threads · DVFS performance 锁 2.6GHz（4 ###AB freq_khz 全 2600000）· **T-N: PASSES=2×REPS=10 = n=20/side/phase** · engage-probe = 12 fires（部署-during-e2e 实证·p16n8）。

**结果（analyze_phase_split.py·中位·relIQR 卫生·n=20）**：

| phase | ours(q4ON) t/s | stock(q4OFF) t/s | ratio | ours relIQR | stock relIQR | n | 判 |
|---|---|---|---|---|---|---|---|
| **prefill pp128** | **5.561785** | 1.509540 | **3.6844×** | 0.61% | 0.11% | 20 | **≥parity · WIN +268%** |
| decode tg32 | 1.761415 | 1.056795 | **1.6668×** | 0.89% | 0.04% | 20 | **WIN +66.8%（atypical both-phase）** |

全 relIQR ≪ 地板×1.5·freq 全程 2.6GHz·n=20/side/phase。

**★verdict = GREEN-6/84**（perf-covered 5/84→6/84·FLAT 净新 scaffold 第三格 perf 绿·q8_1-activation 家族第 2 成员）。判据 = 预注册规则「prefill ≥parity 过 T-N+八门 → 绿」，prefill 3.68× 压倒性 ≥parity。

**八门状态**（全过·prefill 轴）：① 同树同 binary 物理 .so swap ✓ ② 部署验 nm ON=2/OFF=0 syms ✓ ③ banner engage 12 fires 真模型 p16n8 ✓ ④ objdump vl=8 seal（never 16/64）✓ ⑤ 对手身份 = stock q4_1 block-dot 同树（非 SELF·无 repack）✓ ⑥ 编译器对称 gcc-15.2.0 双侧（board shipped rv64gcv=gcc-15 → kernel==system）✓ ⑦ correctness GREEN 前置（5/5 byte-identical A==B·PPL 22.05）✓ ⑧ T-N 卫生 n=20·relIQR≪地板·DVFS 锁 2.6GHz·freq 全稳 ✓。

**双账本**：board rv64gcv 出货工具链 == gcc-15 ⇒ **kernel-axis == system-axis**（两账本数值同 3.6844×/1.6668×，均有效；对照 [CASE-COMPILER-ASYMMETRY] k1=clang-18 才需分账）。opponent identity = stock ggml q4_1 block-dot（同树·仅 scaffold diff·无 repack path）。

**★诚实 caveat（大倍数解读 + 反常 both-phase win）**：
- **stock q4_1 baseline 弱**：q4_1 无 stock riscv repack path（同 q5_0/q5_1）→ stock=generic block-dot；stock q4_1（1.51 pp / 1.06 tg）明显慢于 stock q5_0（3.37 pp）/ q5_1（3.18 pp / 1.57 tg）。3.68× = 我方净新 repack GEMM vs **未优化的 stock q4_1 block-dot**（一个 L1 path win，同 q4_0 WinB 5.9× 类），**非**「q4_1 比一个调优内核快 3.68×」。
- **decode 亦 WIN（1.67×）= atypical**：与 q5_0/q5_1/q4_0 的 prefill-win/decode-wash 定式相反。主因 = stock q4_1 decode baseline 弱（1.06 t/s）；同 emitted-GEVM 手法在 q5_1 上是 decode regression（0.78×）。q4_1 更小的无-qh 320B repacked 布局（vs q5_1 384B）亦利 decode 带宽。**flag for scrutiny**；verdict 立于 prefill≥parity（不依赖 decode）。
- kernel+e2e / prefill+decode 永分报（[[kernel-wins-dont-transplant-to-e2e]]）。

## 八、verdict

**★GREEN-6/84**：correctness GREEN（5/5 byte-identical A==B·PPL 22.05·UT 双 GREEN·objdump vl=8） ∧ prefill 3.6844× ≥parity（T-N n=20·八门全过·gcc-symmetric）→ **perf-covered 5/84 → 6/84**。decode 1.6668× 亦 WIN（atypical·§七 caveat 诚实披露·weak stock baseline 驱动）。
**方法学**：q4_1 = 复用 q5_1 净新 scaffold recipe + **零改动复用 q8_1 sub-scaffold**（block_q8_1x4 + `ggml_quantize_mat_t<1,Q8_1>`）·仅去 qh、换 weight interleaver = **q8_1-activation repack 家族第 2 成员** · C1 template extensibility + C2 边际成本递减再实证。
**board**：测后 restore 验 clean（源 baseline byte-exact·live .so 05a62e6a 0 syms·无 q4_1 .inc stray·DVFS performance 2.6GHz）。

## durable files
- `evidence.md`（本文）
- `tcrv_emitted_gemm_q4_1.inc`（md5 31912b13·25KB）
- `tcrv_emitted_gevm_q4_1.inc`（md5 d645aa42·14KB）
- `seal_raw.txt`（build+patch+seal 全 log）
- `objdump_gemm_q4_1_seal.txt`（vl=8 seal）
- `objdump_gevm_q4_1_seal.txt`（vl=8 seal）
- `correctness_GREEN_raw.txt`（5/5 byte-identical·PPL·banner）
- `phase_split_raw.txt`（4×###END·n=20·llama-bench -o json 全量）
- `transmission_accounting.csv`（双账本·八门键·诚实 caveat）
- `.gitignore`
> 两 .inc 小（39KB 合计）保留 tracked；emit recipe 见 §二可再生。board harness 住 `tools/e2e-harness/board/g5-m2-q4_1/`。
