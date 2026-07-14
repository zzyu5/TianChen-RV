# G8 §六.3 P1-followup — k1 K-quant GEMM vs REAL native-layout opponent (真攻坚) + q2_K correctness

> **性质**：decree 真攻每格。前序 P1(a320e227/996fc1260) 把 q5_K/q6_K 8x8 A/B 判为 **DEFERRED**
> （手工构 8x8 → 0.09-0.13 GMAC/s，误诊为 cache-thrash 伪影）。本役用 **ggml 自家 exported repack**
> 无损构造原生布局，独立 scalar dequant 参考对拍验证喂对，再冷态 A/B。同时验 ours q2_K correctness。
> **[NG-4] kernel-axis micro**·非 e2e·stock .so 只读(md5 871169a0)·core pin·224MiB 冷 flush paired。
> 原始：`raw/measure_stdout.txt` · `raw/run_8x8.log` · `raw/dispatch_and_opponent_nature.txt` · `raw/build_seal.txt`。

---

## 0. Headline（两条结论，都是**证伪/翻案**，非顺风加分）

| 项 | 前序 P1 结论 | 本役真攻结论 | 依据 |
|---|---|---|---|
| **q5_K/q6_K 8x8** | "RVV-specialized hand-brick·A/B DEFERRED(布局无法构造)" | ★**premise 证伪**：k1 上 q5_K/q6_K **根本不 repack**（dispatch 无 riscv 分支→nullptr→block-dot）。8x8 GEMM 是 **ARM/x86-only 死代码**，在 k1 上是 generic fallback（gather 重·vwmacc=0·真 0.09 GMAC/s，**非伪影**）。**真派发对手 = block-dot**（ours 4.45×/1.77× PASS·census）。 | dispatch 反汇编 + 源码 §2/§3 |
| **q2_K correctness** | "1.36× timing-valid WIN(byte-layout 异 nbad=8114)" | ★**WIN 撤销**：喂 canonical ggml block_q2_Kx16 + q8_Kx4(4x1)（q4_K 正控 + ggml-16x1 双双 nbad=0 验证此布局正确），**ours q2_K = max_rel 2.0·nbad≈全错** → ours q2_K 核**算错**（真 bug，非布局伪影）。 | §4 正控法 |

---

## 1. 施工（正确构造原生布局的关键 = 用 ggml 自家 repack，不手搓）

**stock .so 导出了 ggml repack 自由函数本体**（前序手搓失败的根因是没用它）：
- `ggml::cpu::repack::repack<block_q5_K,8,8>(ggml_tensor*,const void*,size_t)` @0x39212（q6_K@0x39684，q2_K<1,16>@0x3cc76，q4_K<1,16>@0x3b250）——直接 link 调用，喂 plain q5_K 块产 `block_q5_Kx8` 原生交织。
- 激活：**8x8 路 INTER_SIZE=8 → `ggml_quantize_mat_q8_K_4x8`**；**16x1 路 INTER_SIZE=1 → `ggml_quantize_mat_q8_K_4x1`**（★前序若混用会全错——见 §4 harness bug 自查）。产 `block_q8_Kx4`。
- GEMM 对手 = stock 出货符号 `ggml_gemm_q5_K_8x8_q8_K` / `q6_K_8x8` / `q2_K_16x1` / `q4_K_16x1`（真派发候选）。
- ours = weft vl=16 repack-GEMM（commit 38abf20eb 导出核，unrolled=deploy default）。
- **独立参考**：本 harness 自带 `dequantize_row_q{2,4,5,6}_K` 重写（ggml quant 定义 re-derive·含 get_scale_min_k4）→ dequant(权重)·dequant(激活 d·qs) float dot = ZERO-MODEL 独立 oracle，且被 `ggml_vec_dot` 二次校验（max_rel ~1e-6，参考自证正确）。

配方：`harness_8x8.cpp` + `run_8x8.sh`（clang-18.1.8 Bianbu·march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b -O2·core pin·N=12 224MiB 冷 flush paired·median）。

---

## 2. q5_K/q6_K：8x8 对手**喂对了**（nbad=0）但它**不是 k1 派发对手**

**喂对确证**（非伪影）：ggml 自家 repack 产的原生 8x8 buffer 喂 `ggml_gemm_q5_K_8x8_q8_K`，输出 vs 独立参考：

| 格 | nr | XCHK vs 独立 ref | vecdot vs ref | opp 8x8 GMAC/s | ours GMAC/s | ratio(8x8) |
|---|--:|---|---|--:|--:|--:|
| q5_K | 16 | max_rel 8.8e-3 **nbad=0/8192** | 1.5e-5 | 0.0906 | 2.622 | 28.95× |
| q5_K | 64 | max_rel 6.5e-3 **nbad=0/32768** | 1.9e-6 | 0.0906 | 2.637 | 29.11× |
| q6_K | 16 | max_rel 5.1e-4 **nbad=0/8192** | 8.6e-7 | 0.1308 | 2.220 | 16.97× |
| q6_K | 64 | max_rel 1.9e-3 **nbad=0/32768** | 1.1e-6 | 0.1309 | 2.269 | 17.34× |

IQR opp <0.2% / ours <0.7% 冷态稳定。⟹ **0.09-0.13 GMAC/s 是 8x8 GEMM 的真吞吐**（输出 bit-correct），前序"cache-thrash 伪影"诊断**证伪**。standalone probe8_8way 早测的 0.0907 与此 0.0906 一致——那个数一直是真的，只是当时以为喂错了。

**但 8x8 不是 k1 上 q5_K/q6_K 的真派发对手**（这是本役核心发现）：
- `ggml_gemm_q5_K_8x8_q8_K` **只在 `arch/arm/repack.cpp` + `arch/x86` 定义**，riscv 无特化 → k1 上 link 到 **generic fallback**。反汇编 insn mix = `vluxei64`×15（indexed gather）+ `vmacc.vx`×14 + **vwmacc=0**（无 widening 整数 MAC）= clang 自动向量化 generic 参考，**非手调 RVV 砖**（对比 q2_K_16x1 vwmacc.vx×16 = 真手调）。
- **dispatcher 判决**（`ggml_repack_get_optimal_repack_type` repack.cpp:4644-4665）：q5_K/q6_K 分支**只有 NEON 条件**（`has_neon()&&matmul_int8`→8x8 / `has_neon()&&dotprod`→8x4），**无 `has_riscv_v()` 分支**。k1(riscv·无 NEON) 全部 fall-through → 函数尾 `return nullptr`（4724）→ **q5_K/q6_K 在 k1 上不 repack** → mul_mat 走标准 **block-dot** 路。
- 对比：q4_K/q2_K 分支**有** `has_riscv_v()` case 256 → 选 16x1（真派发·真手调砖）。

⟹ **q5_K/q6_K 在 k1 的真派发 prefill 对手 = block-dot**（`ggml_vec_dot_q{5,6}_K_q8_K`），非 8x8。8x8 是 registry 里 declared-but-never-selected 的 ARM-only 死代码。

**deployment-口径判定**（对手成色 = block-dot·真派发）：
| 格 | ours GMAC/s | block-dot GMAC/s (census) | ratio | 判定 |
|---|--:|--:|--:|:--:|
| q5_K | 2.62 | 0.588 | **4.45×** | PASS(vs 真派发 block-dot) |
| q6_K | 2.22 | 1.25 | **1.77×** | PASS(vs 真派发 block-dot) |

**PENDING 2 格 resolve**：不是"打赢 8x8 hand-brick"（该砖 riscv 不存在/不派发），而是**证伪"8x8=真派发对手"前提**，回退到真派发 block-dot → ours PASS。29×/17× vs 8x8 仅作**死代码上界旁注**（若未来 k1 加 q5/q6 riscv repack 分支，现 generic 8x8 会是灾难，ours 碾压）——**不作 headline**（对手非部署）。

**具名残余**：`[GAP-K1-Q5Q6-NO-RISCV-REPACK-DISPATCH]`（可检验）= repack.cpp:4644-4665 q5_K/q6_K 无 riscv_v 分支；`ggml_gemm_q5_K_8x8_q8_K` 无 riscv 定义。二者独立可复核。

---

## 3. 对手真实性总账（反稻草人）

| 符号 | k1 上性质 | 派发? | insn 证据 |
|---|---|:--:|---|
| ggml_gemm_q5_K_8x8_q8_K | **generic fallback**(arm/x86-only 特化) | ✘(nullptr) | vluxei64×15·vmacc.vx×14·vwmacc=0 |
| ggml_gemm_q6_K_8x8_q8_K | 同上 generic | ✘(nullptr) | vwmacc=0 |
| ggml_gemm_q2_K_16x1_q8_K | **真 RVV 手调砖** | ✔(riscv_v case256) | vwmacc.vx×16 |
| ggml_gemm_q4_K_16x1_q8_K | **真 RVV 手调砖** | ✔(riscv_v case256) | vwmacc 80(前序) |
| ggml_vec_dot_q{5,6}_K_q8_K | block-dot 标准路 | ✔(q5/q6 fallback) | — |

---

## 4. q2_K correctness（正控法·ours 核算错·WIN 撤销）

**q4_K 正控**（验证整条 16x1 harness + 独立参考）：喂 canonical repack<q4_K,1,16>+q8_Kx4(4x1)：
- ggml_gemm_q4_K_16x1 vs 独立 ref：**nbad=0/8192**（max_rel 2.2e-3）✔
- **ours q4_K vs 独立 ref：nbad=0/8192**（max_rel 5.2e-4）✔ — ours q4_K on-silicon byte-correct（比 census ours-vs-ggml 更强：独立 ZERO-MODEL oracle）。ours 1.17× beat ggml 16x1（复现 census 1.187×）。

⟹ 16x1 喂法 + 参考 + ours 16x1 机制**全部验证通过**。

**q2_K 判定**（同一 harness·同族布局·同激活）：
| 对拍 | nr16 | nr64 |
|---|---|---|
| ggml_gemm_q2_K_16x1 vs ref | nbad=0/8192 ✔ | nbad=1/32768(一个近零元) ✔ |
| **ours q2_K vs ref** | **max_rel 2.0·nbad=7984/8192** ✘ | **max_rel 2.0·nbad=32001/32768** ✘ |

★**前序 harness bug 自查**：第一版误用 `q8_K_4x8` 喂 16x1（应 `4x1`），令 ggml-16x1 也 nbad~全→暴露后改 4x1，ggml-16x1 立即 nbad=0/1，ours 仍全错 → 证 ours 错**非**激活布局问题。

**结论**：ours q2_K 在 **ggml-16x1 与 ours-q4_K 双双 nbad=0 的同一 canonical 布局**上产生 max_rel=2.0 全错输出（非转置·transp 已择优）→ **ours q2_K 核算错（真 bug）**。
⟹ **q2_K "1.36× timing-valid WIN" 撤销**——那是错核的计时，correctness-first 下非 win。（q2_K DOES repack 16x1 on k1，对手是真手调砖，但 ours 先要算对才能谈赢。）

**残余 caveat（诚实）**：本役喂 ours 的是 ggml-canonical block_q2_Kx16。若 ours-q2_K 独一无二地期望**非** canonical 布局（与 ours-q4_K 相反），则属喂法失配非核 bug；但 (a) ours-q4_K 同族同 harness 吃 canonical 布局 nbad=0，(b) census 明述 q2_K/q4_K 16x1 皆"drop-in same byte layout as ours" → 证据强烈指向 ours-q2_K 核 bug。修法/根因诊断 = 后续 task。

---

## 5. 板卫生 · 触碰文件 · 禁 git

- **board 卫生**：core pin（本役选到 core2 idle=100%·gov=performance 1.6GHz）·co-tenant 存在(loadavg ~2.8)·paired within-proc ratio 抵消·IQR 低证·k1 无 cache PMU 用 wall/median·scratch `/tmp/g8k1_8x8/`(ephemeral·用完可清)·stock .so 只读(md5 871169a0 未变)·ggml 源树 `/home/bianbu/tcrv-k1-llama`(commit 20039c9)**只读**(nm/objdump/sed·未改未 rebuild)。
- **触碰（本机·未 commit）**：本 casefile `kquant-gemm-k1-8x8/`（harness_8x8.cpp·run_8x8.sh·evidence.md·summary CSV·raw/）。**未 git add/commit**·**未改 T3_B/T8**（主会审整合）。
- **主树**：仅**读** commit 38abf20eb 导出核（未改 lib/）。
