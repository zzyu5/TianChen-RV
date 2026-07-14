# G8 §六.3 攻坚 P1 — k1(VLEN256) K-quant GEMM 5-家族 cold A/B 普查 (2026-07-14/15)

> **性质**：decree P1 measure-first。用 commit 38abf20eb S6-strip-outer emitter 导出 k1 VLEN256
> (half_lanes=16, vl=16 满宽) 的 ours-repack K-quant GEMM 核，对 k1 出货真同-op hand-brick 做冷态 A/B。
> **第二赛道 kernel-axis** micro（NG-4）·非 e2e beat·非 sealed 8-gate Win·与 perf-covered 系统账永不混算。
> **板**：k1 / SpacemiT-X60 / VLEN256 / clang-18.1.8 Bianbu 对称域 / core0 pin / gov=performance 1.6GHz。
> **口径**：K=2048 nc=512 nr∈{16,64}·N=12 冷态 224MiB-flush paired·median·ratio=ours_gmacs/opp_gmacs。
> **禁 git**（主会审后 commit）·主树/build/stock-.so(只读)/governor 全未动。

---

## 0. Headline 结果表 (cold median·unrolled=deploy default·K=2048 nc=512)

| 格 | 真派发对手 (成色) | opp 有效性 | cold nr16 | cold nr64 | 0.8门 | 备注 |
|---|---|---|--:|--:|:--:|---|
| **q4_K** | `ggml_gemm_q4_K_16x1_q8_K` RVV hand-brick (real@0xabe58) | **drop-in byte-verified** (nbad=0) | **1.187×** | **1.161×** | **PASS/WIN** | ★**锚点复现**(Win-K1-VLEN≈1.197×) |
| **q2_K** | `ggml_gemm_q2_K_16x1_q8_K` RVV hand-brick (real@0xacd96) | timing-valid·byte-layout 异(nbad>0) | **1.364×** | **1.353×** | **PASS/WIN** | 同-stride timing 公平·非 drop-in |
| **q3_K** | `ggml_vec_dot_q3_K_q8_K` block-dot | 无 hand-brick(cross-op)·correctness-separate | 2.920× | 3.030× | PASS/WIN | q3_K 唯一无 repack·弱对手 |
| **q5_K** | block-dot(floor)·8x8 hand-brick DEFERRED | correctness-separate | 4.448× | 4.456× | PASS/WIN | hand-brick 轴 measure-only 待深攻 |
| **q6_K** | block-dot(floor)·8x8 hand-brick DEFERRED | correctness-separate | 1.769× | 1.734× | PASS/WIN | hand-brick 轴 measure-only 待深攻 |

**IQR**：全 <2.2%（ours）/ <1.9%（opp），冷态稳定。原始 = `raw/run_full.log` + `raw/run_blockdot.log`。

**判读**：
- **q4_K = 唯一 byte-verified 真-hand-brick WIN**：ours vl=16 6.99 GMAC/s vs 真 16x1 hand-brick 5.89 = **1.187× cold**，correctness cross-check **nbad=0/8192**（ours == ggml 自家 16x1 bounded-ULP·max_rel 4.2e-4）。**锚点 Win-K1-VLEN ≈1.197× 复现**（smoke K=512 亦得 1.1996×）。这翻转了 archive/g7 q4K-handbrick-resolve 的 **0.622× LOSS**——那是 **vl=8 核**；本役 **vl=16 满宽核**在同 16x1 hand-brick 上翻正。判别键 = **核 VLEN 利用度**（vl=8→vl=16），与 memory「vl=8 kernel-sym≠vl=16 sealed 双核分立」一致。
- **q2_K**：ours 4.92 vs 真 16x1 hand-brick 3.60 = **1.364×**。同-stride(1344)·同 activation(q8_Kx4 1168)·work-volume 内容无关 → **timing 公平**；但 ggml q2_K 16x1 的 group 内字节排布与我方 repack 不同（correctness nbad=8114 → **非 byte-exact drop-in**）。作 **timing-valid WIN** 报（有 caveat）。block-dot floor 另测 4.65×。
- **q3_K**：dispatcher 完全无 `ggml_gemm_q3_K`（objdump=1-insn stub）→ **q3_K 永不 repack** → block-dot 是**真** prefill fallback（cross-op）。ours 2.92× WIN vs block-dot（**弱对手·成色须标**）。
- **q5_K/q6_K**：8x8/8x4 hand-brick 存在且 **RVV-specialized**（q5_K_8x8 rvv=646 / q6_K_8x8 rvv=201），但**布局无法无损构造** → A/B **DEFERRED**（详 §3）。报 block-dot floor（q5 4.45× / q6 1.77×·correctness-separate 公平）。

---

## 1. 对手真实性 (反稻草人·machine-probe + dispatcher 反汇编)

**stock lib** = `/data/k1build-stock/bin/libggml-cpu.so` md5 `871169a0123139692177468b3c8578be`（repack.cpp 亦 clang++-18 -O3 编 → 对称域·无 [CASE-COMPILER-ASYMMETRY]）。

**每格真派发符号（nm + objdump RVV 计数）**：
| 格 | 对手符号 | 地址 | insns | rvv | vwmacc | 成色 |
|---|---|---|--:|--:|--:|---|
| q4_K | ggml_gemm_q4_K_16x1_q8_K | 0xabe58 (real) | 767 | 238 | 80 | **RVV-SPECIALIZED hand-brick** |
| q2_K | ggml_gemm_q2_K_16x1_q8_K | 0xacd96 (real) | 494 | 196 | 16 | **RVV-SPECIALIZED hand-brick** |
| q5_K | ggml_gemm_q5_K_8x8_q8_K | 0x2fa90 | 2426 | 646 | 15 | RVV-specialized（无 16x1 候选） |
| q6_K | ggml_gemm_q6_K_8x8_q8_K | 0x321fa | 526 | 201 | 3 | RVV-specialized（无 16x1 候选） |
| q3_K | (无 ggml_gemm_q3_K) | — | 1 | 0 | 0 | **NO repack → block-dot cross-op** |

**dispatcher `ggml_repack_get_optimal_repack_type` 反汇编**（168 lines·1 csrr=VLEN-keyed）：static tensor_traits 候选 = {q4_K_16x1, q2_K_16x1, q4_K_8x4/8x8, q5_K_8x4/8x8, q6_K_8x4/8x8, q2_K_8x8, q4_0_*, q8_0_*}。⇒ **q4_K/q2_K 有 16x1 RVV 候选**（VLEN256 命中）·**q5/q6 仅 8x4/8x8**·**q3_K 缺席**。与 archive/g7 evidence（q4_K case256→16x1）+ T3_B PENDING 行标注一致。

---

## 2. 施工 (导出配方·byte-exact 构造)

**核导出**（每格 unrolled + rolled 两 schedule·仅翻 half_lanes 8→16 保原名以匹配 driver 符号）：
```
weft-opt <fixture>_vlen256_<sched>.mlir --weft-rvv-lower-to-emitc \
  | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp  > repack_gemm_<fmt>_q8_K_vlen256_<sched>.kernel.c
```
- fixtures = `fixtures/<fmt>_vlen256_{unrolled,rolled}.mlir`（自 test/Conversion/RVV base fixture sed `half_lanes=8→16`；rolled 加 `emit_loop_schedule="rolled"`）。
- 5 格 × 2 sched = 10 核·全 emit 正确符号 `weft_emitc_ggml_repack_gemm_<fmt>_q8_K_kernel_ggml_repack_gemm_<fmt>_q8_K`·literal "16"（f32m2 满宽 16 lane = VLEN256 确认）。
- **ours objdump**（clang-18 -O2 .o·`raw/build_seal.txt`）：全 **cleanfp**（zfh/zvfh native fp16·0 softfp）。unrolled vwmacc 1088–1152（= vl=16 单 16-lane strip·rvv casefile vl=8 双 8-lane strip 的一半）·q4_K spill 2 / q2_K spill 3（低）·q3_K/q5_K spill 48/45（高·super-block 重）。

**driver** = `kquant_gemm_k1_census_driver.c`：ours `(n,s,vx,vy,nr,nc,bs)` · 16x1 hand-brick `(n,s,bs,vx,vy,nr,nc)` drop-in 同 Wr/Ar · block-dot 逐(r,c) · correctness cross-check（16x1 模式）+ cold paired A/B。

---

## 3. q5_K/q6_K 8x8 hand-brick 为何 DEFERRED（认输体例·非预判·实测证据）

- **不是预判「结构墙」**——8x8/8x4 是 **RVV-specialized**（rvv=646/201·§1），机制上是真竞争者。
- **实测构造尝试**（`raw/probe8_8way.c`）：喂 (a) 我方 block_qX_Kx16 (16-way stride 2816/3360) 与 (b) 正确 8-way stride (1408/1680) + q8_Kx8 activation + 全 buffer 有限 fp16 填充，**两者皆得 0.09–0.13 GMAC/s**（q5 740ms / q6 513ms per GEMM）。
- **具名 X（可检验）**：0.09 GMAC/s = **cache-thrash 伪影**，非真吞吐。根因 = ggml x8 kernel 的**原生 weight 8-way interleave 内部排布 + q8_Kx8 activation 交织格式 + fp16 scale 精确位置**未被无损重构 → 每 load 落错址 → 内存 stall 主导。真吞吐需用 ggml **自家 repack 函数**产 buffer（e2e-ish·超本 micro census scope）。
- **对比 q2_K 16x1 为何成立**：16x1 与我方**同 stride(1344)**·同 activation(q8_Kx4)·scale 在头 64B（我方 fill 覆盖）→ 3.60 GMAC/s 稳定可信（IQR 1.02/0.22%·跨 nr 一致）。q5/q6 8x8 三者皆不满足。
- **结论**：q5/q6 hand-brick 轴 = **measure-only 待深攻**（残余 = ggml-native 8x8 layout 构造）。本役报 block-dot correctness-separate floor（q5 4.45× / q6 1.77×）。

---

## 4. G1 byte-exact (k1 silicon·on-board correctness)

- **q4_K**：ours vl=16 vs ggml 自家 16x1 hand-brick 在**同 Wr/Ar**上 **nbad=0/8192**（nr16）+ **0/32768**（nr64）·max_rel 4.2e-4/5.9e-4（IEEE fp reassoc noise）。**独立 oracle = ggml 出货核**·整数核任何 byte 失配会放大远超 4e-4 → **整数核 byte-exact 确证**（on-silicon）。
- q3_K：ours 与 block-dot 布局不同（无 drop-in cross-check）·但 emitter byte-exact 构造 + 既有 per-format repack verifier（lit 绿）承载。
- q2_K/q5_K/q6_K：correctness-separate（既有 verifier）·本役未跑独立 scalar INT ref（q4_K vs-ggml-16x1 = 本 census 的 on-silicon correctness 锚）。

---

## 5. schedule 轴 (unrolled=deploy default·best-schedule per-format 异)

deploy 缺省 = **unrolled**（base fixture 无 emit_loop_schedule stamp → measured-gate 空 → unrolled）。ours_gmacs（opp-无关·`raw/run_full.log`）：
| 格 | unrolled ours | rolled ours | best |
|---|--:|--:|:--:|
| q2_K | 4.92–5.01 | 1.32–1.33 | **unrolled**（rolled 灾难 -73%） |
| q3_K | 3.42–3.52 | 3.92–4.03 | rolled(+15%) |
| q4_K | 6.92–6.99 | 6.10 | **unrolled**(+13%) |
| q5_K | 2.60–2.63 | **3.26–3.28** | rolled(+25%) |
| q6_K | 2.20–2.23 | 2.15–2.16 | unrolled(~tie) |

⇒ 部署缺省 unrolled 下 §0 表成立；q5_K rolled stamp 可再+25%（vs blockdot 4.45→5.55×）。**自动选择需把 [ROLL] gate 键细化到 decode_model**（同 rvv casefile 结论·结构级 follow-up·非本役 scope）。

---

## 6. 板卫生 · 触碰文件 · 禁 git

- **board 卫生**：core0 pin（idle 100%·gov=performance 1.6GHz）· co-tenant 存在（loadavg 2.0–3.1·core2 busy·paired within-proc ratio 抵消共同 contention·IQR 低证）· k1 无 cache-miss PMU → 未用 cache-misses×64B（仅 wall/median）· scratch `/tmp/g8k1_gemm_census/`（ephemeral）· stock .so 只读（nm/objdump/link）。
- **provenance**：k1 树无 .git → so-hash md5 `871169a0…`（stock）· ours 核 clang-18.1.8 Bianbu · march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b。
- **触碰（本机·未 commit）**：本 casefile 目录（driver + kernels/ + fixtures/ + raw/ + summary CSV + 本 evidence）。**回填 T3_B 5 个 PENDING GEMM 行**（gemm|q{2,3,4,5,6}_K·coldm1/ratio/cold_ratio/status·见 §7）。**无新建账本·无 git add/commit**。
- **主树**：仅**读** commit 38abf20eb emitter 导出核（未改 lib/）。

## 7. T3_B 回填值（gemm|qX_K|repack-gemm 行·PENDING→填数）
| 行 | coldm1 / ratio_vs_opp / cold_ratio | opponent_grade | status | hardgate |
|---|---|---|---|---|
| q4_K | 1.187(nr16)/1.161(nr64) vs 16x1 hand-brick·nbad=0 drop-in·锚点1.197x复现 | hand-tuned | PASS | in-denom;PASS@0.8 |
| q2_K | 1.364/1.353 vs 16x1 hand-brick(timing-valid·byte-layout异) · 4.65 vs block-dot | hand-tuned(timing-only) | PASS | in-denom;PASS@0.8 |
| q3_K | 2.920/3.030 vs block-dot(cross-op·无repack) | block-dot(cross-op) | PASS | in-denom;PASS@0.8 |
| q5_K | 4.448/4.456 vs block-dot floor·8x8-handbrick-DEFERRED | block-dot(floor) | PASS(floor);handbrick-deferred | in-denom;PASS@0.8-vs-blockdot |
| q6_K | 1.769/1.734 vs block-dot floor·8x8-handbrick-DEFERRED | block-dot(floor) | PASS(floor);handbrick-deferred | in-denom;PASS@0.8-vs-blockdot |
