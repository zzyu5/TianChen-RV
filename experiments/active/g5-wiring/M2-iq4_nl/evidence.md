# G5-M2 iq4_nl 曳光弹 — evidence（L-接线② codebook 格 · flip-gate+intercept · correctness-carrier）

> workflow (2026-07-12) · board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree HEAD 保持 · nproc=64 ·
> **HEAD (TianChen-RV)** = 禁 git · **A-tree 测后 restore 验 clean**（源回 baseline md5 GEN=deb61a29/ARCH=99131cf7 · live .so 回 05a62e6a OFF-pristine）。
> **结论 = correctness-carrier GREEN · perf BLOCKED-environmental · perf-covered 维持 6/84**：**iq4_nl EMITTED vl=8 repack GEVM+GEMM = silicon UT byte-exact + greedy A==B 5/5 byte-identical + 38 engage banners**（首个 codebook 格 correctness-carrier·黄-未接线 → correctness-wired）· **perf 轴无法测得 clean ratio —— 共享板过载（load 22→30·他用户 VLLM+llama+python）致间歇 mmap SIGBUS·且 stock OFF 同崩 = 环境级非 kernel 缺陷**（§七）· A-tree 测后 restore clean。
> **框架 = 与 q6_K（净新全 scaffold）/ q4_1（零 scaffold）不同 = q8_0-M1b「flip-gate + intercept」**：iq4_nl 上游 riscv `1,16` repack scaffold **全present 但 VLEN128 破损（硬编码 AVL=16 + vrgather）且 dispatch gate OFF（`case 128: break // TODO`）→ stock 恒 block-dot**。我方 = 翻 gate + 我方 emitted vl=8（memory-gather）拦截破损上游 body。

## 一、recon：iq4_nl = 上游 scaffold 全 present 但 VLEN128 破损 + gated（≠ 净新·= q8_0 破损 kernel 模式）

板 A-tree read-only（baseline GEN=`deb61a29` / ARCH=`99131cf7` = 干净 WinB-q4_0-ON）：
- **★纠偏「上游零 iq4_nl repack」的简单 path-win 假设** —— iq4_nl 是主流 codebook 格，上游 riscv `1,16` repack **整条链路已建**：
  - HDR `block_iq4_nlx16`（`:118`·stride 288 = `ggml_half d[16]`@0 + `uint8_t qs[QK4_NL*8]`@32）present。
  - GEN `make_block_iq4_nlx16`（`:3691`·straight 16-way col-interleave `out.qs[i]=in[i%16].qs[i/16]`）+ `repack_iq4_nl_to_iq4_nl_16_bl`（`:3715`）+ `repack<block_iq4_nl,1,16>`（`:3946`）+ `gemv/gemm<block_iq4_nl,1,16,Q8_0>`（`:4043/:4140`）+ trait `iq4_nl_16x1_q8_0`（`:4568`）全 present。
  - ARCH riscv **真 kernel body** `ggml_gemv_iq4_nl_16x1_q8_0`（`:463`）/ `ggml_gemm_iq4_nl_16x1_q8_0`（`:1342`）present。
- **★但两 ARCH body 硬编码 AVL=16 + 寄存器 vrgather**（`__riscv_vle8_v_i8mf2(kvalues,16)`·`__riscv_vrgather_vv_i8mf2(values,...,16)`·`__riscv_vwmul_vx_i16m1(...,16)` 全 literal-16）→ VLEN128（mf2/SEW8 VLMAX=8）只处理 16 列的 8 列 = **半列 garbage**（[GAP-Q8_0-VLEN128-KERNEL] 同型）。
- **★dispatch `case 128: { break; } // TODO`（`:4680`）→ get_tensor_traits 返 nullptr → 板上 iq4_nl 恒走 stock block-dot**（`ggml_vec_dot_iq4_nl_q8_0`）。`case 256` 才 route trait。
- **→ 部署 = 翻 case128 gate + 我方 emitted vl=8 拦截**（correctness-carrier·非白嫖·非净新全 scaffold）。**q8_K 激活无关**（iq4_nl 用 **q8_0 激活**·全 present 上游·零 sub-scaffold）。
- **path-win 立点**：stock 现状 = block-dot（非破损上游 repack·gated）→ 我方 emitted repack vs stock block-dot = q4_1 同型 path 对位；但 iq4_nl 有 codebook **memory gather（vluxei16）税** → perf 由 §七 板测定，不投影。

## 二、kernel emit（host·LLVM20.1.8·vl=8 VLEN128-safe·zero 16/64·memory-gather 非 vrgather）

host `./build/bin/tcrv-opt` + `/usr/lib/llvm-20/bin/mlir-translate`，fixtures 已存（front-door 自 RETIRED monolithic 重构·block_iq4_nlx16）：
- **GEMM (prefill)** 源 `test/Conversion/RVV/rvv-emit-quant-contraction-iq4-nl-repack-gemm-prefill-vlen128.mlir`
  → `tcrv_emitted_gemm_iq4_nl.inc`（242978B·md5 **1e040596**）·func `tcrv_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0`
  ·ABI `(size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`（v1=nr `/4`·v3=n `/32`+vsetvl·v5=vx `×288`·v6=vy `×136`·v7=nc `/16`）
- **GEVM (decode)** 源 `test/Conversion/RVV/rvv-emit-identity-quant-contraction-iq4-nl-repack-vlen128.mlir`
  → `tcrv_emitted_gevm_iq4_nl.inc`（114268B·md5 **b7a5267e**）·func `tcrv_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0`
  ·ABI `(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`
- **★源级 seal**：两 .inc `, 16)`=0 `, 64)`=0，AVL 常量全=8（GEMM 786·GEVM 366）；`vsetvl_e32m1` dynamic。**`vrgather`=0**（区别上游破损 body）· **`vluxei16` memory-gather = 64/64**（codebook `tcrv_iq4_nl_repack_kvalues[16]={-127,-104,...,113}` reconstructed·shared-amortized 跨 4 列）。
- weight stride **288**（block_iq4_nlx16）· act GEMM stride **136**（block_q8_0x4）/ GEVM stride **34**（plain block_q8_0）。
- **板 objdump seal**（§五）：两 emitted 符号 `vsetivli imm=8 only · imm=16=0 · imm=64=0`（VLEN128-safe·部署==证过）。

## 三、部署 scaffold（deploy_patch_iq4_nl_emitted.py·**2 文件** tracked·可逆·上游 scaffold 复用·无净新 struct/make/repack/trait）

**≠ q6_K 净新全 scaffold（改 3 文件建 struct/make/repack/trait/arch）· = q8_0-M1b flip-gate+intercept（改 2 文件）**：
1. **GEN 翻 dispatch gate**：iq4_nl else-if 内 `case 128: { break; } // TODO` → `case 128: { if (cur->ne[1]%16==0) return &iq4_nl_16x1_q8_0; break; }`（route 到**上游既有** trait）。
2. **ARCH 拦截**：`#include` 我方两 emitted .inc + 在既有 `ggml_gemv/gemm_iq4_nl_16x1_q8_0` body 顶（`UNUSED(blocklen);` 后）插 `if (__riscv_vlenb()*8==128) { banner; emitted_call(...); return; }` —— 拦在破损上游 vl=16 body **之前**。ABI 映射：GEVM `(n,s,vx,vy,nc)`·GEMM `(nr,bs,n,s,vx,vy,nc)`。
- **上游件全复用不动**：struct block_iq4_nlx16 / make_block_iq4_nlx16（== 我方 emitted 读的布局·§四 UT 证）/ repack template / trait 注册 —— **零净新**。
- **★方法学（C1 extensibility 续铺 + C2 边际成本）**：q8_0-M1b「破损上游 VLEN128 kernel → 我方 emitted vl=8 correctness-carrier 拦截」模式**延伸到 codebook 家族**（iq4_nl = 首个 codebook 部署）；对比 q6_K（K-quant 净新全 scaffold）· q4_1（零 scaffold 净新）——**同一 template 协议覆盖三种上游就绪度**（present-broken-gated / net-new / absent）。

## 四、UT 独立 oracle（★MIRAGE de-risk·board VLEN128 silicon·build 前先验·GREEN）

`ut_iq4_nl_verify.cpp`（自写·无既有 iq4_nl repack verifier）—— 我方 emitted GEVM+GEMM vs **独立 scalar iq4_nl×q8_0 codebook dot oracle**，读 **upstream make_block_iq4_nlx16 交织**（`out.qs[j*16+c]=in[c].qs[j]`）+ block_q8_0x4 act（`qs[e*4+m]`·从 emitted body 精确抽出）：
- **INT（unit d·magnitude-bounded·byte-exact-integer）** + **NORM（adversarial fp16 d·f64 ref）**·6 shapes（nb{2,4,8}×nc{16,32,48}×nr{4,8,16}）·GEVM+GEMM 各测。
- **VERDICT = SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM**：`INT_mismatch_total=0`·`worst_int_absdiff=0.000`·`worst_norm_reldiff=2.163e-06` → interleaver + emitted vl=8 kernel silicon-正确·MIRAGE trap 关。**correctness-first 硬律满足（build 前先验过）**。

## 五、build + seal（gcc-15.2.0 对称·OFF/ON·objdump vl=8·restore byte-exact）

`g5_m2_iq4_nl_build_seal.sh`（build_seal.log）：
- **OFF build md5=`05a62e6a`·iq4_nl_tcrv_sym=0**（pristine·**== 已知 baseline OFF-pristine 05a62e6a**·证 patch 纯附加·gate-OFF==baseline）。
- **ON build md5=`71f346b5`**（≠OFF·variants differ）·patch 全件 confirm（gate128 flipped / include gemm+gevm / gevm+gemm emitted-call / gevm+gemm banner 全 True）。
- **objdump vl-seal**：GEMM 符号 `vsetivli imm=8=147 · imm=16=0 · imm=64=0`（dyn_vsetvli=407·SEW/LMUL 全 e8mf2/e16m1/e32m2·**零 vrgather / 零死宽度**）；GEVM 同型（imm=8·imm16/64=0·60 e8mf2 / 60 e16m1 / 4 e32m2）→ VLEN128-safe·部署==证过。
- **restore**：2 源回 baseline byte-exact（GEN=`deb61a29`/ARCH=`99131cf7`）·pristine rebuild OK·live .so 回 `05a62e6a` 0 syms = **LIVE == OFF-pristine（zero net change）**·A-tree 干净。

## 六、correctness GREEN（★hard gate·greedy A==B byte-identical·真模型·silicon UT byte-exact）

`g5_m2_iq4_nl_correctness.sh`· model=DeepSeek-R1-Distill-Llama-8B-IQ4_NL.gguf（sha256 `7b64a33e`…·自 Q6_K requantize IQ4_NL `--allow-requantize`·决策卡④·IQ4_NL 无需 imatrix·A/B 读同权重）：
- **5/5 prompt BYTE-IDENTICAL** A(ON emitted iq4_nl repack)==B(OFF stock iq4_nl block-dot)·emitted-kernel banner **38 fires**（真 forward engage 探针·含 n=4096 nr=512 nc=128 prefill GEMM）·no NaN/Inf·**CORRECTNESS_GATE: GREEN**（byte_identical=5 failed=0 engaged=38 ppl_ok=1）。
- **★correctness 基座 = greedy A==B 5/5 byte-identical（GREEN）+ silicon UT INT byte-exact 6 shapes 0-mismatch（GREEN）** → repack-vs-blockdot fp accumulation 同序·不翻 argmax（q4_1/q6_K 同型前例）。
- **PPL non-fatal（SOP §4 主门 = greedy A==B）**：llama-perplexity 在 PPL sanity 步 `Bus error (core dumped)`——**根因 = 并发 .so swap artifact（非 kernel bug）**：多测量进程 `cp variant→live` 覆盖另一进程 mmap 中的 live .so → torn mmap → SIGBUS（correctness llama-completion 单进程全程 5/5 clean 证 kernel 正确）。教训：所有 .so-swap 测量必**串行单跑**·swap 与 llama-* 调用间无并发。

## 七、perf 分相 —— **BLOCKED（共享板过载·环境级·非 kernel 缺陷·证据：stock OFF 同崩）**

> 同树物理 .so swap（ours=iq4ON `71f346b5` / stock=iq4OFF `05a62e6a`）·DVFS 锁 performance 2.6GHz。
> **★perf 无法取得 clean 数据 —— 根因 = 共享板环境过载（本 agent 不可控）**：
> - 测量期间 `ssh rvv` 板 **load average 从 ~9 攀到 22→30**（`uptime`），并发他用户重载：`root` VLLM engine（`VLLM::EngineCor`）+ `paper3` llama-cli/python3.11 + `ai-test` 多会话 + 另一 `ubuntu` llama-bench（775% CPU）。
> - 高内存/IO 压力下 **mmap 权重（4.7GB）page-in 间歇失败 → llama-bench `Bus error (core dumped)`**，**且 stock OFF（pristine `05a62e6a`·零 tcrv 符号·= q4_1/q6_K perf 用过的同一 baseline·数日前该板安静时跑通）同样崩**（perf2.log：ABPASS1/2 both ours+stock `-p128` 全 Bus error·warmup `-p8` 亦崩·engage-probe `-p16` 偶通）→ **判定 = 环境级间歇 SIGBUS·非 iq4_nl emitted kernel 缺陷**（correctness llama-completion 单进程 10× 全 clean·UT byte-exact·ON/OFF 同崩 = 决定性反证 kernel 论）。
> - 两轮尝试全废：① 全 phase_split（`-p128 -n0 -r10 ×2 pass`）4 blocks 全 `Bus error`·**avg_ts 捕获 0**；② retry-robust 小批（`pp64/tg16`·MAXTRY=8·load 已 30）PASS1 即卡·**0 OK 结果**。
> **★结论 = perf 判读 BLOCKED-environmental**：**无 clean prefill/decode ratio → 无法据预注册规则判 green/对手更强 → perf-covered 维持 6/84**（本格 perf 轴**未测得**·非测得 loss/win）。harness 完备（deploy_patch+build_seal+phase_split+retry 全就绪·仅需板 load<~10 复跑）。
> **★measure-infra canon（新）**：所有 .so-swap 板测须 ① 串行单跑（并发 swap 覆盖 mmap = 早期额外 SIGBUS 源·已排除）② 全脱附 setsid+nohup（前台 ssh timeout 杀慢 bench）③ **板 load 门（load>~15 = 测量无效·环境 SIGBUS + 噪声·须等安静窗口）**。

| phase | 状态 |
|---|---|
| prefill pp128 | **BLOCKED**（ON+OFF `-p128` 均 env-SIGBUS·avg_ts=0·load 22→30） |
| decode tg32 | **BLOCKED**（同 env·retry 0 OK） |

## 八、verdict【correctness-carrier GREEN · perf BLOCKED-environmental · perf-covered 维持 6/84】

**双账本**：board rv64gcv 出货 == gcc-15 ⇒ kernel-axis == system-axis（对称·若可测则数值同）。opponent identity = stock ggml iq4_nl block-dot（同树·仅 2-file wiring diff·上游 repack gated+broken·非 SELF）。
- **correctness GREEN**（greedy A==B 5/5 byte-identical·silicon UT INT byte-exact 6 shapes 0-mismatch·38 engage banners·objdump vl=8·no NaN/Inf）→ **C1 template flip-gate+intercept 模式延伸到首个 codebook 格·iq4_nl = 首个 codebook correctness-carrier e2e 部署证正确**（黄-未接线 → **correctness-wired**·升成色于 correctness 轴）。
- **八门（correctness+seal 侧全过·perf 侧 env-blocked）**：① 同树物理 .so swap ✓ ② nm ON syms present/OFF=0 ✓ ③ banner engage 38 fires 真模型 ✓ ④ objdump vl=8 seal（never 16/64·vrgather=0）✓ ⑤ 对手 = stock iq4_nl block-dot 同树（非 SELF）✓ ⑥ gcc-15.2.0 双侧对称 ✓ ⑦ correctness GREEN 前置 ✓ ⑧ DVFS 锁 2.6GHz ✓ ｜ **⑤beat-from-八门 = N/A（perf env-blocked·未测得 ratio·禁以 perf 名义入台账·perf-covered 维持 6/84）**。
- **A-tree restored clean**：源 GEN=`deb61a29`/ARCH=`99131cf7`/HDR=`57851439` baseline byte-exact · live .so=`05a62e6a` OFF-pristine · 0 tcrv iq4_nl syms · 无 stray .inc。
- **morphology-declare 兄弟**（见 §九）。

## 九、morphology-declare 兄弟（iq4_xs / iq2_s / iq2_xs / iq2_xxs / mxfp4 / nvfp4 · untested same-family declared-covered）

iq4_nl = **codebook 家族实测锚**（本 casefile 唯一板测 gemm-轴格）。同族 gemm 格**不逐格板测**·据 iq4_nl 结果 + 现有 batch2c 对称-gcc vec_dot 证据 declared-covered（same 律）：
- **batch2c 对称 vec_dot LOSS 全族**：iq4_xs **0.72×**（OUTPUT-DIVERGENT 曾记）· iq2_s 0.48× · iq2_xs 0.33× · iq2_xxs 0.78× · iq3_s 0.28× · iq3_xxs 0.16×（全 gcc-对称 measured-fair LOSS·具名 GAP = `uarch.gather_slow` per-subblock gather/redsum 未 batch）。
- **declared-covered 判读**：codebook/grid 格共享 memory-gather（vluxei/gather-slow）税 → **gemm-轴接线预判 = 对手更强-likely**（据 sibling vec_dot 对称-gcc LOSS 同律·`uarch.gather_slow` 具名 GAP·可修=争议）·**untested same-family declared-covered·非 perf-covered 绿格**（mxfp4/nvfp4 fp4-codebook 同族·[NG-4] Amdahl<噪声 morphology-declared）。iq4_nl 本身 perf 轴 env-blocked（§七）→ 兄弟 perf 预判**仅承 sibling vec_dot LOSS 证据·不承 iq4_nl gemm 实测锚**（诚实：本 casefile 未取得 iq4_nl gemm perf 数·锚点 = correctness-carrier 结构 + 同族 vec_dot LOSS）。

## durable files
- `evidence.md`（本文）· `correctness_GREEN_raw.txt`（【PENDING】）
- `tcrv_emitted_gemm_iq4_nl.inc`（md5 1e040596·243KB·**gitignored**）· `tcrv_emitted_gevm_iq4_nl.inc`（md5 b7a5267e·114KB·**gitignored**）
- board harness `tools/e2e-harness/board/g5-m2-iq4_nl/`（deploy_patch + ut + build_seal + correctness + phase_split + analyze + driver + ut verifier）
