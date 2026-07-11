# G5-M2 q5_0 曳光弹 — evidence (L-接线② 净新 scaffold · phase-1 recon+emit+design)

> workflow (2026-07-12) · board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree HEAD f3e1828 ·
> **HEAD (TianChen-RV)** = a7cacf68 · 禁 git · **A-tree 零改动**（本 phase 只 read-only recon + host emit；scaffold 建设 deferred）。
> **结论 (phase-1)**：**upstream_absent 确证**（q5_0 riscv repack 分支【真零 present】，与 q4_K 全-present 相反）· **kernel EMITTED**（GEMM+GEVM，vl=8 全 AVL·zero 16/64）· **scaffold = 12 件净新，deferred 到下一里程碑**（含 1 件 correctness-critical：make_block_q5_0x16 interleaver）。**远超 q4_K deploy（q4_K = 翻 1 行 gate；q5_0 = 建整条链路）**。

## 一、recon：q5_0 = 真零 present（★与 q4_K 决定性区别）

板 A-tree read-only（baseline md5：GEN=`deb61a29` / ARCH=`99131cf7` = 干净 WinB-q4_0-ON，与 q4_K/q8_0 同基线）：
- **GEN `repack.cpp` q5_0 grep count = 0**；**ARCH `arch/riscv/repack.cpp` q5_0 grep count = 0**。
- 对比 q4_K：trait+case256+arch-body 全 present、仅 case128 gate OFF（翻 1 行）。**q5_0 无任何 riscv repack 素材** → 必须净新建。
- 根因：ggml repack 用 `block<K,N>` 模板（`block_q8_0x16 = block<8,16>`：仅 `d[N]+qs[]`）。**q5_0 的 5-bit（独立 qh 高位）无法套 `block<K,N>`** → 无上游 x16 支持。q5_K 有 super-block x8 支持但那是 K-quant 家族、非 flat q5_0。

## 二、kernel emit（host·vl=8 VLEN128-safe·zero 16/64）

host `./build/bin/tcrv-opt`（LLVM20.1.8）+ `/usr/lib/llvm-20/bin/mlir-translate`，与 q8_0/q4_K 同 recipe：
- **GEMM (prefill)** 源 `test/Conversion/RVV/rvv-emit-quant-contraction-q5-0-repack-gemm-prefill-vlen128.mlir`：
  `tcrv-opt <fix> --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`
  → `tcrv_emitted_gemm_q5_0.inc`（27237B·md5 **f03c6566**）·func `tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0`
  ·ABI `(size_t nr, size_t bs, size_t n, float* s, size_t nc, const uint8_t* vx, const uint8_t* vy)`【与 q8_0 GEMM 同】
- **GEVM (decode)** 源 `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q5-0-repack-vlen128.mlir`（同 pipeline）
  → `tcrv_emitted_gevm_q5_0.inc`（20180B·md5 **f38920496...**·`f3892049...`）·func `tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0`
  ·ABI `(size_t n, float* s, size_t nc, const uint8_t* vx, size_t bx, const uint8_t* vy, size_t by, int32_t nrc)`【与 q8_0 GEVM 同：调用 bx/by/nrc 传 0】
- **★broken-upstream 排雷（防第 3 个 MIRAGE）**：两 .inc 全 AVL 常量 = 8，`, 16)` 与 `, 64)` 计数【均 0】；setvl 为 dynamic `vsetvl_e32m1(n)`。LMUL 形全 mf2/m1/m2（zero f32m4/whole-register/redsum wall）。weight stride 352（block_q5_0x16=16d+256nib+64qh）、qh offset 288、act stride 136(GEMM)/34(GEVM) 均在。q5_0 5th-bit decode assembly（vor_vv_u8mf2 / vsrl_vv_u16m1 / vreinterpret_u8→i8 / vsub bias-16）全 present。
- **★源级 vl=8 confirmed；板 objdump vl-seal（vsetivli imm=8 only）留 deploy phase 出证。**

## 三、★净新 scaffold recipe（L-接线② 方法学产出·供 q5_1 直接复用·12 件）

trait `tensor_traits<block_q5_0, 1, 16, GGML_TYPE_Q8_0>` 全链需净新（activations = q8_0 → from_float/quantize_mat 复用 q8_0，零新激活量化器）。逐件（含板 anchor 行）：

**A. header `ggml/src/ggml-cpu/repack.h`（净新）**
1. `struct block_q5_0x16 { ggml_half d[16]; uint8_t qh[64]; uint8_t qs[256]; };`（=352B）+ `static_assert(sizeof==352)`。★**不可用 `block<K,N>` 模板**（那模板无 qh）。
2. `#if defined __riscv_zvfh` 内加声明：`ggml_gemv_q5_0_16x1_q8_0` + `ggml_gemm_q5_0_16x1_q8_0`（+ `_generic` 变体）。

**B. GEN `ggml/src/ggml-cpu/repack.cpp`（净新）**
3. **★correctness-critical**：`make_block_q5_0x16(block_q5_0* in, unsigned blck_size_interleave)` interleaver — 必须【byte-exact】产出 emitted kernel 解码的布局（16 d @0；nibbles @32 interleave=1；**transposed qh @288**，位序须与 fixture 的 `vsrl_vv/vand/vsll` per-strip 选位一致）。镜像 `make_block_q8_0x16`(GEN:3512)/`make_block_q4_0x16`(GEN:2811) 但须加 qh 转置逻辑。**此件是唯一高风险/需板上 A==B 验证的净新。**
4. `repack_q5_0_to_q5_0_16_bl(t, interleave=1, data, data_size)` — 镜像 `repack_q8_0_to_q8_0_16_bl`(GEN:3535)，循环调 #3。
5. `template <> int ggml::cpu::repack::repack<block_q5_0, 1, 16>(t, data, data_size)` → `return repack_q5_0_to_q5_0_16_bl(t, 1, data, data_size);`（镜像 GEN:3951 q8_0 / 3939 q4_0）。
6. `template <> void gemv<block_q5_0, 1, 16, GGML_TYPE_Q8_0>(...)` → `ggml_gemv_q5_0_16x1_q8_0(...)`（镜像 GEN:4047-48）; 同理 `gemm<block_q5_0, 1, 16, GGML_TYPE_Q8_0>` → `ggml_gemm_q5_0_16x1_q8_0`。
7. `ggml_gemv_q5_0_16x1_q8_0_generic` + `ggml_gemm_q5_0_16x1_q8_0_generic` 标量 fallback body（镜像 q8_0 generic GEN:1525/2580，须解 q5_0 5-bit；VLEN128 部署由 ARCH override 拦截，但符号须存在可链）。
8. dispatch case：`ggml_repack_get_optimal_repack_type` 内加 `case GGML_TYPE_Q5_0:` 块，riscv 分支镜像 q8_0（GEN:~4710）：`case 128:{ if(ne[1]%16==0) return &q5_0_16x1_q8_0; }` + `case 256` 同。**★net-new 直接建 case128=ON（非 //TODO），我方 emitted vl=8 承载。**
9. trait 注册：`static const ggml::cpu::repack::tensor_traits<block_q5_0, 1, 16, GGML_TYPE_Q8_0> q5_0_16x1_q8_0;`（镜像 GEN:4569 q8_0）。

**C. ARCH `ggml/src/ggml-cpu/arch/riscv/repack.cpp`（净新·我方 emitted 拦截）**
10. `ggml_gemv_q5_0_16x1_q8_0` body：`UNUSED(blocklen);` 后插 VLEN128 分支 → banner + `tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0((size_t)n,s,(size_t)nc,(const uint8_t*)vx,(size_t)0,(const uint8_t*)vy,(size_t)0,(int32_t)0)` + `return;`（**与 q8_0 M1b ARCH 体逐字同构**，只换 q8_0→q5_0）。
11. `ggml_gemm_q5_0_16x1_q8_0` body：VLEN128 分支 → banner + `tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0((size_t)nr,bs,(size_t)n,s,(size_t)nc,(const uint8_t*)vx,(const uint8_t*)vy)` + `return;`。
12. `#include "tcrv_emitted_gemm_q5_0.inc"` + `#include "tcrv_emitted_gevm_q5_0.inc"`（在 q4_0 .inc 并列处）。

**部署（scaffold 建成后）**：复用 `tools/e2e-harness/board/g5-m2-q4k/` 全套（deploy_patch adapt 名字·build_seal·correctness·phase_split），3 挂点模板不变。

## 四、工作量诚实评估

- **机械件（1,2,4,5,6,7,8,9,10,11,12）**：~半日细致镜像（q8_0/q4_0 有逐件对照 anchor，低风险）。
- **★#3 make_block_q5_0x16（correctness-critical）**：transposed-qh 布局须与 emitted kernel byte-match；错则编译过但 e2e garbage（= MIRAGE 陷阱）。需板上 build + greedy A==B / PPL 验证闭环，若 qh 位序细节有偏差可能 1-2 会话 debug。
- **判定**：**远超 q4_K deploy**；按停机/降级条款【scaffold 建设列为下一里程碑】，不本 phase 硬啃（防未验证 compiling-scaffold 冒充完成 = 反 MIRAGE 纪律）。**recon+emit+design 本 phase 已交付。**

## 五、durable files
- `evidence.md`（本文）
- `tcrv_emitted_gemm_q5_0.inc`（md5 f03c6566·27KB）
- `tcrv_emitted_gevm_q5_0.inc`（md5 f3892049·20KB）
> 两 .inc 小（47KB 合计）故保留 tracked（区别 q4_K 的 2.5MB gitignored）；emit recipe 见 §二可再生。

## 六、下一里程碑（scaffold 建设 phase-2）
1. 建 12 件 scaffold（A-tree 可逆·备份 GEN/ARCH/repack.h md5·restore byte-exact）。
2. **先 make_block_q5_0x16 单元对照**（构一小 q5_0 张量·repack→对比 emitted kernel 期望布局 byte-exact）再全模型。
3. 板 build OFF/ON（gcc-15.2.0 对称）+ objdump vl-seal（imm=8 only）+ greedy A==B / PPL correctness-first。
4. correctness GREEN 后才 perf 分相（FLAT class·kernel-axis 1.23× gcc-symmetric·**别预设绿**·1.23× 小 margin 可能 washes）。

## 七、phase-2 perf 分相（★clean 重测·2026-07-12·board rvv openEuler VLEN128 gcc-15.2.0）

> workflow glitch 后的 phase-2 perf 补测。测量进程经 nohup 存活跨 agent glitch → **单次连续未中断 clean run**（非拼接）。同树物理 .so swap（ours=q5ON `8d136dd9` 2 syms / stock=q5OFF `05a62e6a` 0 syms）·ONE llama-bench·gcc-15.2.0 双侧对称·唯一 A/B diff = q5_0 净新 dispatch gate + emitted kernels。
> model=DeepSeek-R1-Distill-Llama-8B-Q5_0.gguf（sha256 4b11ec8d…）·taskset 8-15 · 8 threads · DVFS performance 锁 2.6GHz · **T-N: PASSES=2×REPS=10 = n=20/side/phase** · engage-probe banner = 11 fires（部署-during-e2e 实证）。

**结果（analyze_phase_split.py·中位·relIQR 卫生）**：

| phase | ours(q5ON) t/s | stock(q5OFF) t/s | ratio | ours relIQR | stock relIQR | n | 判 |
|---|---|---|---|---|---|---|---|
| **prefill pp128** | **4.08104** | 3.37359 | **1.2097×** | 0.044% | 0.19% | 20 | **≥parity · WIN +21%** |
| decode tg32 | 1.470735 | 1.80124 | 0.8165× | 0.92% | 1.9% | 20 | regression −18%（memory-bound GEVM wash） |

per-pass 无漂移（prefill ours 4.0817/4.0802·stock 3.3771/3.3704 = 岩石稳；decode 略噪 memory-bound 预期内）·全 relIQR 远低地板×1.5·freq 全程 2600000·co-tenant clean（692% = 自身 8-thread bench，VLLM idle 0.2%）。

**★verdict = GREEN-4/84**（perf-covered 3/84→4/84·FLAT 净新 scaffold 首个 perf 绿）。判据 = 预注册规则「prefill ≥parity 过 T-N+八门 → 绿」，prefill 1.21× 清晰 ≥parity。**★1.23× kernel-axis micro margin 【确传导】到 e2e prefill 1.21×（非 wash）**——预测「很可能 yellow」被证伪，prefill 侧确赢（GEMM compute+repack locality）。

**八门状态**（全过·prefill 轴）：① 同树同 binary 物理 .so swap ✓ ② 部署验 nm ON=2/OFF=0 syms ✓ ③ banner engage 11 fires 真模型 ✓ ④ objdump vl=8 seal（never 16/64，build_seal §3）✓ ⑤ 对手身份 = stock q5_0 block-dot 同树（非 SELF）✓ ⑥ 编译器对称 gcc-15.2.0 双侧（board shipped rv64gcv=gcc-15 → kernel==system，无 clang 不对称）✓ ⑦ correctness GREEN 前置（byte-identical A==B·PPL 17.88，correctness_GREEN_raw.txt）✓ ⑧ T-N 卫生 n=20·relIQR≪地板·DVFS 锁·无 co-tenant ✓。

**双账本**：board rv64gcv 出货工具链 == gcc-15 ⇒ **kernel-axis == system-axis**（两账本数值同 1.2097×/0.8165×，均有效；对照 [CASE-COMPILER-ASYMMETRY] k1=clang-18 才需分账）。opponent identity = stock ggml q5_0 block-dot（同树·仅 scaffold diff·objdump 探针对手身份）。

**诚实 caveat**：decode −18% 是 memory-bound GEVM 回退（repack 布局利 prefill GEMM locality、损 decode 带宽），与 q4_0 WinB 同型 prefill-win/decode-wash split。kernel+e2e / prefill+decode 永分报（[[kernel-wins-dont-transplant-to-e2e]]）。perf-covered 绿基于 prefill ≥parity 公平协议格。

**durable perf files**：`phase_split_raw.txt`（4×###END·n=20·llama-bench -o json 全量）·`transmission_accounting.csv`（双账本·per-pass·八门键）·本 §七。

**A-tree restore（测后·board 铁律）**：phase_split.sh 尾 `swap OFF` 自恢复 → live=05a62e6a·both trees（build-gcc15-rv64gcv + build_verify）live .so 05a62e6a 0 syms·source 3-file baseline md5（deb61a29/57851439/99131cf7）·无 q5_0 .inc stray·git 无 q5_0 stray（`tcrv_emitted_repack_gemm/gemv.inc` = 2026-07-06 WinB-q4_0 baseline infra 非本次 stray）·DVFS performance 2.6GHz。**板 clean 验讫。**
