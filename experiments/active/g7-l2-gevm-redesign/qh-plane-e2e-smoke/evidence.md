# G7 L2 — qh-plane GEVM REDESIGN-B · q5_0@rvv e2e decode 传导冒烟 · [CASE-MICRO-E2E] 层

> **前置**：G2（kernel-axis）完成 — q5_0 GEVM leaf 换 REDESIGN-B（vlm+masked-vsub 4-op 削重建·HEAD 40c21de0）·cold decode-GEVM M=1 micro NEW-vs-stock 2.2–2.5× / NEW-vs-OLD 1.44–1.67×（casefile `../qh-plane-G2/evidence.md`）。
> **本线**：kernel-axis ≠ e2e 系统账（禁互推）。q5_0@rvv 是 perf-covered 绿格但**仅 prefill 轴**（G5-M2·prefill 1.21× WIN / **decode 0.82× regression**）。**本测：REDESIGN-B 是否让 q5_0@rvv e2e decode 从 0.82× 翻到 ≥parity。**
> **★裁决(TL;DR)**：见 §5。

---

## 0. 环境 · 板编译器身份核实（★G2 纠偏之再核实）
- 板：`ssh rvv`·openEuler·`localhost.localdomain`·kernel 6.12.66 riscv64·VLEN128·64c·isa `rv64imafdcv...zvfh` (full)。
- **板 SYSTEM 默认编译器**：`gcc (GCC) 12.3.1 (openEuler 12.3.1-30.oe2403)` — **`riscv_vector.h` 缺失**（`fatal error: riscv_vector.h: No such file or directory`）·**无 RVV intrinsics 能力**。`clang version 17.0.6` — **`riscv_vector.h` 可用**（编译通过）。
- **e2e 部署工具链（本 llama 部署路径真实使用）**：`/opt/tcrv-toolchains/gcc-15.2.0`（经 `/opt/tcrv-toolchains/env.sh` 置于 PATH 首位）— `gcc (GCC) 15.2.0`·**`riscv_vector.h` 可用**。G5-M2 harness `source env.sh`→ ggml/llama 全树用此 **gcc-15.2.0** 编译。
- **★编译器身份三分裁决（诚实分层）**：
  1. **板出货默认（out-of-box）** = gcc-12.3.1（无 rvv intrinsics）+ clang-17（有）。若无 tcrv 工具链，ggml RVV 只能走 **clang-17**（gcc-12.3.1 编不了 intrinsics）→ 此即 G2 casefile 与本任务 brief 所指「板 RVV 部署编译器 = clang-17」的语义（**[CASE-COMPILER-ASYMMETRY] 判别键：板 out-of-box RVV baseline = clang-17**）。
  2. **本 e2e 实际部署** = **gcc-15.2.0**（installed toolchain·非板出货默认·`/opt/tcrv-toolchains`）。**双方（NEW/OLD/OFF）全用同一 gcc-15.2.0** → **编译器对称**·无 [CASE-COMPILER-ASYMMETRY] 不对称污染·kernel==system 账本一致。
  3. **CLAUDE.md「rvv=gcc-15」** 在「板出货默认」义上【不成立】（出货 gcc-12.3.1）；在「本 e2e installed toolchain」义上【成立且对称】（gcc-15.2.0）。两义不同，分记。
- load-gate：测前 load ~2.10–2.14/64c（~3%）·governor performance·freq 2.6GHz·`taskset -c 8-15`·co-tenant vLLM 在 cores 0,1 idle 0.0%（未重启·本就 idle）。

## 1. 三方 .so 构建（同树·gcc-15.2.0 对称·物理 .so swap）
同一 A-tree `/home/ubuntu/tcrv-llamacpp`·`build-gcc15-rv64gcv`·唯一 A/B/C diff = q5_0 GEVM leaf 发射器：
| 变体 | scaffold | GEVM .inc | .so md5 | q5_syms | 语义 |
|---|---|---|---|---|---|
| **q5OFF (stock)** | 无 | — | `05a62e6a` | 0 | pristine ggml q5_0 block-dot |
| **q5OLD** | 12-piece | OLD per-lane-expand (`f3892049`) | `8d136dd9` | 2 | G5-M2 原部署（复现 bit-exact） |
| **q5NEW (REDESIGN-B)** | 12-piece | REDESIGN-B (`3b00eb43`·G2 emit·weft→tcrv 符号改名) | `8791f420` | 2 | vlm+masked-vsub 削重建 |
- **★复现验证**：OFF md5=`05a62e6a` 与 OLD md5=`8d136dd9` **逐字节复现 G5-M2 evidence**（q5OFF=05a62e6a / q5ON=8d136dd9）→ harness 忠实复现·OLD = G5-M2 原部署本尊。
- GEMM .inc（`f03c6566`·sealed leaf）OLD/NEW **完全相同**（REDESIGN-B 只动 GEVM leaf）。
- 部署编译器 = gcc-15.2.0（`env.sh`）·restore 后 live=`05a62e6a` 0 syms（源 3-file byte-exact 恢复·pristine rebuild）。

## 2. ★decode 路由验证（deployed==proven·objdump 符号机判）
部署 .so 内 GEVM 符号 `tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_...` objdump 反汇编（gcc-15.2.0）：

| | 总 v-insn | qh-decode 签名 | native-mask (v0.t) | vsetivli(8/16/64) |
|---|---|---|---|---|
| **q5NEW (REDESIGN-B)** | **88** | `vlm.v`×4 · `vmnot.m`×4 · `vadd.vi`×4(masked) · **0× vor.vv/vnsrl.wi/vsrl.vv/vsll.vi/vid.v** | `v0.t`×4 | 1/0/0 ✓ |
| **q5OLD (per-lane-expand)** | **105** | `vsrl.vv`×4 · `vsll.vi`×4 · `vor.vv`×4 · `vnsrl.wi`×4 · `vid.v`×1 · **0× vlm.v/v0.t** | 0 | 1/0/0 ✓ |

- **★路由裁决**：部署 NEW .so 的 GEVM 符号**唯一携带 REDESIGN-B native-mask 签名**（vlm.v 直装 mask bits + vmnot.m 反相 + v0.t masked-vadd 融 5th-bit+bias·gcc-15 把 `vmnand`→canonical `vmnot.m`、把 `vsub_mu`→masked `vadd.vi`）·OLD 唯一携带 per-lane 展开链 → **deployed==proven（符号层）**·88-vs-105 v-insn（−16%）坐实削重建落到部署二进制。
- **decode 真路由到我方 GEVM**（非 stock block-dot）证据 = ARCH scaffold `ggml_gemv_q5_0_16x1_q8_0` 拦截 VLEN128 → 打 banner `TCRV G5-M2 EMITTED GEVM(q5_0_16x1 VLEN128) ENGAGED` → 调我方 emitted GEVM·dispatch case128 gate（`ne[1]%16==0`→`q5_0_16x1_q8_0` trait）。**banner 在真 decode run 触发计数见 §3**（part A engage）。vsetivli imm=8 全 seal（never 16/64·VLEN128-safe·部署==证过宽度）。

## 3. e2e 正确性（coherence 硬门）
**COHERENCE_GATE: GREEN**。5 prompt（DeepSeek-R1-Distill-Llama-8B-Q5_0）· NEW==OLD 5/5 IDENTICAL · NEW==OFF(stock) 5/5 IDENTICAL · FAIL=0 · NEW NaN/Inf=0 · coherent 输出。**★ENGAGE banner（运行时路由证据）= NEW 49 / OLD 47 / OFF 0**（我方 emitted GEVM 在真 decode run 被调 49 次·stock=0）→ **deployed==proven（运行时·非仅符号层）**。

## 4. e2e decode + prefill 系统账（REDESIGN-B vs stock vs OLD·rvv VLEN128·deploy_cc gcc-15.2·8-thread disjoint-pin 8-15·PP=128 TG=32 REPS=6 PASSES=2）
| 变体 | prefill tok/s（P1/P2·med） | decode tok/s（P1/P2·med） |
|---|---|---|
| **NEW（REDESIGN-B）** | 4.0798/4.0824 · **4.081** | 1.5965/1.5926 · **1.594** |
| **OLD（per-lane expand）** | 4.0715/4.0779 · **4.075** | 1.4360/1.4378 · **1.437** |
| **stock（block-dot）** | 3.3692/3.3787 · **3.374** | 1.7388/1.6922 · **1.715** |

**比值（系统账·e2e）**：
- **Prefill NEW/stock = 1.210×**（q5_0@rvv 已知 prefill 绿·**REDESIGN-B decode-only 不动 GEMM**）· **NEW/OLD = 1.001×**（prefill 恒等·坐实 REDESIGN-B 仅改 GEVM decode leaf·隔离实证）。
- **Decode NEW/stock = 0.929×**（<parity·**NO green flip**）· **Decode NEW/OLD = 1.109×**（REDESIGN-B 削重建 e2e decode 净改善 ~11%）· OLD/stock = 0.838×（OLD 即已知 q5_0 decode ~0.82-0.84× regression）。

## 5. ★门结论
**q5_0@rvv e2e：prefill 绿 INTACT（1.21×）· decode NO green flip（0.929×<parity）· REDESIGN-B 削重建【部分传导】（NEW/OLD 1.11×）**。
- **① perf-covered 不动 = 9/83**（q5_0@rvv 维持 prefill-only 绿·decode 仍 regression·从 OLD 0.838× 改善到 NEW 0.929× 但未达 parity）。
- **② [CASE-MICRO-E2E] 边界实证**：REDESIGN-B kernel-axis decode-GEVM micro 1.44-1.67×（vs OLD·G2）→ **e2e decode 稀释到 1.11×（vs OLD）**——削重建净贡献真传导但被 memory-bound e2e decode 稀释（q5_0 GEVM 仅占 decode 一部分·KV/attention/其余层 memory 主导）。**传导真实但不足以翻越 stock block-dot decode 基线**（repack-GEVM decode 在 memory-bound e2e 天然慢于 block-dot·削重建缩小但未消除此差）。
- **③ 诚实定性**：REDESIGN-B = 真 kernel-axis win（byte-exact·结构 redesign board-validated·retarget thesis 兑现）+ e2e 部分传导（narrows decode regression）·**但非 e2e 系统账 win·非 both-phase 绿**。structural novelty 立·kernel-axis perf 立·e2e decode 传导受 memory-wall 限（C3′ 边界素材·非失败）。
- **④ deployed==proven**：符号层（NEW 88 v-insn REDESIGN-B 签名 vs OLD 105·−16%）+ 运行时（ENGAGE banner 49 hits）双证·decode 真调我方 GEVM·非空头传导。

## 6. 板 restore 双证 + 无 stray
- **restore**：`== LIVE after restore md5=05a62e6a82857cdf247ba1933c695849 (expect==q5OFF 05a62e6a) ==` → 板 .so 复原到 stock baseline（前后 md5 一致）· deploy_cc=gcc (GCC) 15.2.0 · model_sha256=4b11ec8d06e24600 未改。
- **无 stray**：`pgrep llama|bench` = 0（测后·主会话 backstop 核实·exit=0）· load_post 10.26/10.03/9.23（8-thread bench + co-tenant·disjoint-pin·非我核污染）· scratch /tmp/g5_q5 保留供审（ephemeral）。
- **★板编译器纠偏（G2 flag RESOLVED）**：本 e2e ENV FINGERPRINT `deploy_cc=gcc (GCC) 15.2.0` → **RVV 部署 ggml = gcc-15.2.0**·canon [CASE-COMPILER-ASYMMETRY] rvv=gcc-15 **CONFIRMED 非 STALE**。G2 micro 报的 clang-17 = 板默认 `gcc`=12.3.1(无 riscv_vector.h) 的 partial view·部署实走 env gcc-15.2（G2 kernel-axis 数 symmetric 有效但 clang-17 域·e2e 数 gcc-15.2 域=部署真相）。

---
_finalize 2026-07-14·主会话 backstop（agent ac42e2f0 stall·bench DONE 后主会话收口）·raw/measure_3variant.out(609 lines)+objdump_gevm_NEW.txt。_
