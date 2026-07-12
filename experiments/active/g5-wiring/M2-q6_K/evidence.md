# G5-M2 q6_K 曳光弹 — evidence（L-接线② 净新 riscv scaffold · K-quant no-min · correctness GREEN · perf yellow）

> workflow (2026-07-12) · board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree HEAD 保持 · nproc=64 ·
> **HEAD (TianChen-RV)** = 6f9099e5 · 禁 git · **A-tree 测后 restore 验 clean**（源回 baseline md5 · live .so 回 05a62e6a OFF-pristine）。
> **结论 = yellow-kernel-axis**：**q6_K 净新 riscv scaffold BUILT + correctness GREEN**（我方 emitted vl=8 q6_K repack kernel 承载正确·silicon UT byte-exact + greedy A==B 5/5）· **但 perf 重回退（prefill ON/OFF < 0.07×）→ perf-covered 维持 6/84**。**net-new scaffold 路径【结构】延伸到 K-quant 超块（C1）· perf【不】延伸**（q6_K weight-reconstruction-bound·emitted vl=8 repack ~15× 慢于 stock hand-tuned block-dot）。

## 一、recon：q6_K = riscv 1,16 路径【净新】（上游仅 aarch64 x8·NEON-gated）

板 A-tree read-only（baseline md5 GEN=`deb61a29` / HDR=`57851439` / ARCH=`99131cf7` = 干净 WinB-q4_0-ON）：
- **上游 q6_K 部分 present、但 riscv 缺**（★纠偏「零 riscv repack」的简单假设）：
  - HDR `block_q6_Kx8`（aarch64 8-wide interleave）present；**`block_q6_Kx16` = 0**。
  - GEN `make_block_q6_Kx8` present；**`make_block_q6_Kx16` = 0**。
  - GEN `gemv/gemm<block_q6_K,4,8/8,8,Q8_K>` + `q6_K_8x4_q8_K`/`q6_K_8x8_q8_K` traits present（aarch64 NEON i8mm/dotprod）；**`<block_q6_K,1,16,Q8_K>` = 0**（我方目标 trait）。
  - ARCH riscv q6_K refs = **0**。
  - dispatch：q6_K 仅 `ggml_cpu_has_neon()&&(matmul_int8||dotprod)` 路由 8x8/8x4 → **rvv 上 get_tensor_traits 恒 nullptr → stock block-dot**。无 riscv 分支。
  - → **riscv `1,16` 路径（block_q6_Kx16 stride 3360）真净新**（L-接线②）。
- **★q8_K 激活【全 present 上游】**（与 q4_K 共享）：`block_q8_Kx4`(stride 1168)、`ggml_quantize_mat_t<1/4/8, GGML_TYPE_Q8_K>`、`from_float=quantize_row_q8_K` 全在 → **NO activation sub-scaffold**（区别 q5_1 的 q8_1 净新 · 更简单）。
- **工作量诚实评估 = tractable 单 session**：activation 复用（省 sub-scaffold）+ make_block_q6_Kx16 与 decode 全由板 verifier 规定（见 §四）+ ABI == q4_K current-HEAD → scaffold 镜像 q5_1 deploy patch（去 q8_1 + 换 q6_K weight/decode）。

## 二、kernel emit（host·vl=8 VLEN128-safe·zero 16/64）

host `./build/bin/tcrv-opt`（LLVM20.1.8）+ `/usr/lib/llvm-20/bin/mlir-translate`，fixtures 已存：
- **GEMM (prefill)** 源 `test/Conversion/RVV/rvv-emit-quant-contraction-q6-K-repack-gemm-prefill-vlen128.mlir`
  → `tcrv_emitted_gemm_q6_K.inc`（1872450B·md5 **9c49195c**）·func `tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K`
  ·ABI `(size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`（== q4_K current-HEAD 序）
- **GEVM (decode)** 源 `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q6-K-repack-vlen128.mlir`
  → `tcrv_emitted_gevm_q6_K.inc`（1138112B·md5 **768a3892**）·func `tcrv_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K`
  ·ABI `(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`（== q4_K GEVM）
- **★源级 seal**：两 .inc `, 16)`=0 `, 64)`=0，全 AVL 常量=8；`vsetvl_e32m1` dynamic。weight stride **3360**（block_q6_Kx16：d[16]@0·scales[256]i8@32·qh[1024]@288·ql[2048]@1312）。
- **板 objdump seal**（build §3）：GEVM `vsetivli imm=8=1175·imm=16=0·imm=64=0`；GEMM `imm=8=2995·imm=16=0·imm=64=0`；SEW,LMUL 全 e8,mf2 / e16,m1 / e32,m2（VLEN128-safe·部署==证过·零死宽度）。**★huge unrolled（6065 e8mf2 ops in GEMM）= perf 病灶伏笔**。

## 三、净新 scaffold（deploy_patch_q6_K_emitted.py·3 文件 tracked·可逆·无 arch-fallback.h·无 q8_K mat-quant 净新）

复用 q5_1 净新 scaffold recipe + q6_K delta（activation 复用上游 q8_K）：
1. **weight block（no-min·6-bit·2 planes）**：`struct block_q6_Kx16 { ggml_half d[16]; int8_t scales[256]; uint8_t qh[1024]; uint8_t ql[2048]; }`（3360B·NO dmin·16 signed int8 scales·qh(2-bit)@288 + ql(4-bit)@1312）。
2. **★q8_K activation = 零净新**（复用上游 `block_q8_Kx4` + `ggml_quantize_mat_t<*,Q8_K>` + `quantize_row_q8_K`·q4_K 共享）→ 比 q5_1/q4_1 省一整条 sub-scaffold。
3. **★correctness-critical = make_block_q6_Kx16 interleaver**：straight 16-way（d[c]@0·scales[s*16+c]@32·qh[i*16+c]@288·ql[i*16+c]@1312·无 bit-transpose·无 scale-min dance·block_q6_K 无 union）。Proven bit-exact vs 独立 oracle（§四）。
- 其余机械件（repack fn/template `repack<block_q6_K,1,16>`、gemv/gemm template `<block_q6_K,1,16,Q8_K>`、scalar generics【no-min 6-bit decode·从 ref_isum_block】、trait 注册 `q6_K_16x1_q8_K`、dispatch **插入既有 q6_K else-if 内的 riscv 分支** case128/256=ON、ARCH VLEN128 intercept + banner + emitted call）逐件镜像 q5_1/q4_K。

## 四、UT 独立 oracle（★MIRAGE de-risk·board VLEN128 silicon·build 前先验·GREEN）

复用板 canonical verifier `tools/e2e-harness/board/kquant_repack_verify_q6K.c`（其 `pack_w` == make_block_q6_Kx16 byte-identical·`ref_isum_block` = ggml canonical q6_K no-min decode）+ current-HEAD GEMM ABI shim（verifier 内嵌 924dc31f 旧序 `(n,s,vx,vy,nr,nc,bs)` → 当前 `(nr,bs,n,s,vx,vy,nc)`）。g++-15.2.0 编译 emitted .inc + verifier·rvv silicon 执行（seed 20260712）：
- **INT（unit d·magnitude-bounded·byte-exact-integer）**：8 shapes（GEVM nc={16,32,256,160} + GEMM nr={4,8,16,4}）**int-mismatch=0 全 BYTE-EXACT-INT**。
- **NORM（adversarial fp16 d·f64 ref·bounded-ULP）**：worst_ulp=307·worst_norm **4.564e-07**。
- **VERDICT = SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM**（`INT_mismatch_total=0`）→ interleaver + emitted vl=8 kernel silicon-正确·MIRAGE trap 关。

## 五、build + seal（gcc-15.2.0 对称·OFF/ON·objdump vl=8·restore byte-exact）

`g5_m2_q6_K_build_seal.sh`：
- OFF build md5=`05a62e6a`·q6_K_tcrv_sym=**0**（pristine·全 baseline 同）。
- ON build md5=`cff0585c`·q6_K_tcrv_sym=**2**（gemm + gemv 在）·banner gevm=1/gemm=1·ON≠OFF。patch 全件 confirm（struct/make/repack_tmpl/gemv_tmpl/gemm_tmpl/trait/dispatch_riscv/generics=2/inc×2/bodies/calls）。
- objdump vl-seal：两 emitted 符号 `vsetivli imm=8 only·0x imm=16·0x imm=64`（VLEN128-safe·部署==证过）。
- **restore**：3 源回 baseline byte-exact（deb61a29/57851439/99131cf7）·pristine rebuild·live .so=`05a62e6a` 0 syms（ZERO net change）。

## 六、correctness GREEN（★hard gate·greedy A==B byte-identical·真模型·silicon UT byte-exact）

`g5_m2_q6_K_correctness.sh`· model=DeepSeek-R1-Distill-Llama-8B-Q6_K.gguf（sha256 84a11990…·自 Q5_1 requantize Q6_K `--allow-requantize`·决策卡④·A/B 读同权重）：
- **5/5 prompt BYTE-IDENTICAL** A(ON emitted q6_K repack)==B(OFF stock q6_K block-dot)·emitted-kernel banner **43 fires**（真 forward engage 探针）·no NaN/Inf。
- PPL：kernel 在 forward engage（16 GEMM banners·n=4096 prefill），但板 llama-perplexity 未在预算内收敛出可解析 estimate（tooling-slowness·非 correctness·脚本判 non-fatal ppl_ok=1）。
- **★correctness 基座 = greedy A==B 5/5 byte-identical（GREEN）+ silicon UT INT byte-exact（GREEN）**（SOP §4「真正确性基座 = greedy 生成文本字节相同」）→ **CORRECTNESS_GATE: GREEN**。broken_upstream 不适用（q6_K 无 riscv body·我方 emitted 是唯一 riscv kernel）。

## 七、perf 分相（correctness 过后·gcc-15 对称·DVFS 锁 2.6GHz·yellow）

> 同树物理 .so swap（ours=q6ON `cff0585c` 2 syms / stock=q6OFF `05a62e6a` 0 syms）·gcc-15.2.0 双侧对称·唯一 A/B diff = q6_K 净新 dispatch + emitted kernels。engage 探针 = 9 fires（部署-during-e2e）。
> **★emitted vl=8 q6_K repack kernel 病态慢**：full phase_split ON 侧单次 llama-bench(pp128+tg32 -r10) 跑 >11 min（vs OFF 秒级）·light(tg8 r8) ON 侧亦 >11 min → 降到 prefill-only 直测。

| phase | ours(q6ON) | stock(q6OFF) | ratio | 判 |
|---|---|---|---|---|
| **prefill pp128**（-t8·DVFS 锁 2.6GHz） | **< 0.28 t/s**（单 rep 128 tok >460s compute 未完成·DNF） | **4.14 t/s**（±0.00·r3） | **< 0.07×** | **重回退 · ≪parity** |
| decode tg | 病态慢（ON tg 侧 llama-bench 11+ min） | ~stock block-dot | ≪1× | 重回退（weight-reconstruction-bound GEVM） |

- OFF pp128 = 4.14 t/s（-t8）/ 4.08 t/s（-t64）= 内存受限·线程饱和·stock hand-tuned q6_K vec_dot。
- ON pp128 < 0.28 t/s（emitted vl=8 repack GEMM·2995 vsetivli + 6065 e8mf2 ops 巨大展开·无调度）→ **~15× 慢于 stock**。三独立观测一致（full/light phase_split ON 侧 11+min · pp128 -t8 单 rep DNF）·correctness GREEN 证非误算（kernel 正确·纯慢）。
- **★verdict = yellow-kernel-axis**：prefill 清晰 ≪parity → **perf-covered 维持 6/84**（fair-protocol ≥parity 不满足·禁以 perf 名义入台账）。

**双账本**：board rv64gcv 出货 == gcc-15 ⇒ kernel-axis == system-axis（数值同·均有效）。opponent identity = stock ggml q6_K block-dot（同树·仅 scaffold diff·无 riscv repack path）。
**诚实**：q6_K prefill <0.07× 比 q4_K（0.42×）更差——6-bit ql+qh 双平面 decode 更重·emitted 巨大 vl=8 kernel 无调度。与 memory `q4-0-e2e-is-routing-not-kernel`（K-quant perf 立不住·weight-reconstruction-bound）一致·q6_K 是又一诚实反例。

## 八、verdict

**★yellow-kernel-axis · perf-covered 维持 6/84**。
- **correctness GREEN**（greedy A==B 5/5 byte-identical · silicon UT INT byte-exact 8 shapes 0-mismatch · 43 engage banners · objdump vl=8 · PPL non-fatal）→ **C1 template 延伸到 K-quant no-min 超块净新 scaffold e2e 结构成就**（比 FLAT 更强 C1 claim：template 从零建 riscv `1,16` q6_K repack 链路·复用上游 q8_K 激活·全模型 route+repack+dispatch 自建集成经 ggml 对照证正确）。
- **prefill perf 重回退**（ON <0.28 vs OFF 4.14 t/s·<0.07×）→ **net-new scaffold 路径【perf 不】延伸到 K-quant**（weight-reconstruction-bound·emitted vl=8 repack ~15× 慢于 stock hand-tuned block-dot）。
- **k_quant_path_extends（结构）= true · k_quant_path_extends（perf）= false**。
**board**：测后 restore 验 clean（源 baseline byte-exact deb61a29/57851439/99131cf7·live .so 05a62e6a 0 syms·无 q6_K .inc stray）。

## durable files
- `evidence.md`（本文）· `correctness_GREEN_raw.txt`（5/5 A==B·43 banners·GREEN）
- `tcrv_emitted_gemm_q6_K.inc`（md5 9c49195c·1.8MB·**gitignored**）· `tcrv_emitted_gevm_q6_K.inc`（md5 768a3892·1.1MB·**gitignored**）
- board harness `tools/e2e-harness/board/g5-m2-q6_K/`（deploy_patch + ut + build_seal + correctness + phase_split + analyze + verifier copy）
> 两 .inc 合计 3MB > 阈值 → gitignored（regenerable via §二 emit recipe·md5 已记）。
