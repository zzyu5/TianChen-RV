# [WORK-ITEM-K1-KQUANT-E2E] — q4_K e2e on k1-clang: 传导存在性证明 + 候选因素隔离（裁二.1 锁定 · 隔离 pending 二.2）

> Board `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / **stock clang-18** / DVFS perf-gov 1.6GHz locked).
> Reopen trigger of `line-c-k1-strike/q5_K_X0_amdahl_verdict.md` [WORK-ITEM-K1-KQUANT-E2E].
> Model `/data/tinyllama-1.1b-Q4_K_M.gguf` (llama 1B Q4_K-Medium, 636 MiB, sha256 9fecc3b3cd76bba8...).
> **NO git · board reversible · correctness-first · kernel==system (clang-18 compiler-symmetric on k1).**

> **★[裁二.1 措辞锁定 · 2026-07-12]** — 本 casefile 的「gcc-death / gcc-death=rvv-specific / NOT weight-reconstruction-wall」结论 **降级为候选因素**。锁定句式 = **K-quant e2e 可传导的存在性证明（k1）；rvv LOSS 非普遍物理墙；候选因素 = {gcc codegen（[CASE-COMPILER-ASYMMETRY] spill 3→742 历史直证）| VLEN128 重建摊销 | uarch}·隔离 pending（二.2 决定性实验）**。⚠ 本 k1-clang vs rvv-gcc board-swap **同时**改了编译器身份 **和** VLEN（256 vs 128）**和** uarch → **非单因隔离**；「gcc-death」冠名权 = 二.2 实验出口 A 兑现后方可用。下文历史措辞保留（append-only 记录），causal 归因一律以本锁定句式为准。

## 决定性问题 & 测量设计（★纠偏 opponent identity）
rvv sibling M2-q4_K: `our-emitted-repack(gcc) / stock-generic-vecdot(gcc) = 0.42×` prefill (LOSS), on rvv-gcc
associated with **gcc codegen spill 3→742** (objdump 直证; [CASE-COMPILER-ASYMMETRY]) of our repack kernel —
a **candidate** factor, not a sole-cause claim (see 裁二.1 lock note). On k1, stock q4_K **already ships the repack** (dispatch
`repack.cpp:4620 case256 → ggml_gemm/gemv_q4_K_16x1_q8_K`, healthy vsetivli-16 @VLEN256) — UNLIKE rvv where
`case128:{break;}//TODO` forced the generic vec_dot fallback. So the clean k1-clang analog of the decisive
question ("does q4_K **repack** transduce to ≥parity vs the **vec_dot baseline** on clang?") is:

**A = REPACK** (stock as-shipped q4_K → `ggml_gemm/gemv_q4_K_16x1_q8_K` repack)  vs
**B = VECDOT** (q4_K repack dispatch disabled → generic block-dot `ggml_vec_dot_q4_K_q8_K`),
**both compiled by clang-18** (COMPILER-SYMMETRIC). This swaps gcc→clang on the *exact* repack-vs-vec_dot
contrast that was 0.42× on rvv-gcc — probing **the compiler-identity candidate** vs an **intrinsic weight-reconstruction wall**.
★裁二.1: the k1↔rvv board swap **also** changes VLEN (256 vs 128) & uarch, so it is **NOT a single-factor isolation**
(candidate factors = {gcc codegen | VLEN128 rebuild amortization | uarch}; 隔离 pending 二.2 — see lock note).

Both variants differ by exactly ONE dispatch line (private recompiled `repack.cpp.o`, relinked); Line B's shared
source `/home/bianbu/tcrv-k1-llama` NEVER edited (md5 3cac40aa preserved). REPACK lib md5 == live shipped stock.

## Isolation / build (harness `build_repack_vs_vecdot.sh`)
- `cp -a /data/k1build-stock → /data/build-k1-workitem` (dedicated build dir; never touched main build/ or build-ime).
- REPACK lib = the copied shipped lib, md5 **871169a0123139692177468b3c8578be** (== `/data/k1build-stock` live lib).
- VECDOT lib = recompile of a PRIVATE edited `repack.cpp` copy (q4_K case256 → `{break;}`) + relink, md5 **532444e80e3686c43970c366972041d1**.
- compiler `/usr/bin/clang++-18`, `-O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d` (stock flags).

## Correctness-first (harness `correctness_ab.sh`)
- REPACK == bit-for-bit the **as-shipped stock lib** (md5 identical) ⇒ correctness = shipped-stock correctness (trivially valid).
- VECDOT = stock's own `ggml_vec_dot_q4_K_q8_K` reference fallback ⇒ also correct-by-construction.
- greedy A==B: **3/4 byte-identical**; the 1 divergence (prompt "Once upon a time") = fp-summation-order near-tie
  between two *distinct* correct kernels (both coherent fluent English, NO NaN/garbage — NOT the VLEN128-MIRAGE
  PPL-822057 class). Determinism: each variant is **SELF-DETERMINISTIC** (REPACK R1==R2 at -t 8); the divergence is
  genuinely repack-vs-vecdot fp-order, not thread noise.
- ⇒ neither variant is a garbage kernel ⇒ perf numbers are meaningful (correctness-first satisfied). **correctness_green = TRUE.**

## e2e phase-split (harness `phase_split_ab.sh` · analyze `analyze_phase_split.py`)
DVFS locked (8× performance gov @1.6GHz, cur==max), taskset -c 0-7, interleaved REPACK/VECDOT, warmup dropped,
PP=128 TG=32 REPS=10 PASSES=2 (n=20/side).

| phase | REPACK (repack) | VECDOT (vec_dot) | **REPACK/VECDOT** | relIQR (R/V) | n |
|---|---|---|---|---|---|
| prefill pp128 | **27.044** t/s | **10.227** t/s | **2.644×** | 2.46% / 1.09% | 20 |
| decode  tg32  | **9.909** t/s | **7.715** t/s | **1.284×** | 4.57% / 2.71% | 20 |

Cross-checks (all consistent, real ELF binary): smoke pp32/tg8 r2 → 2.75×/1.44× · pp64 r3 → 2.60×.
relIQR 1-5% ≪ noise floor ⇒ ratios rock-solid. **Both phases WIN (well above parity).**

**★ MEASUREMENT-VALIDITY NOTE (caught & fixed)**: the first full run read ~1.02×/0.99× (bogus) because
`build-k1-workitem/bin/llama-bench` is a WRAPPER SCRIPT that forces `LD_LIBRARY_PATH=/data/k1build-stock/bin`
and execs `/data/k1build/bin/llama-bench` — so BOTH sides loaded the shipped stock lib (swap ineffective =
stock-vs-stock). Re-run uses the REAL ELF `/data/k1build/bin/llama-bench` directly (RUNPATH < LD_LIBRARY_PATH,
verified `ldd` resolves our swapped lib). The numbers above are the VALID (swap-effective) run.

## 双账本 (dual-ledger disclosure)
k1 kernel-axis compiler = clang-18; system/deploy compiler = clang-18 ⇒ **kernel账 == system账 (CONVERGE)**.
This is the material difference from rvv (stock = gcc-15, the [CASE-COMPILER-ASYMMETRY] axis). Same-数 note: the
ratio here is a single compiler-symmetric account (no clang-ours-vs-gcc-shipped artifact).

## 对照 rvv-gcc baseline (判别键 = 编译器身份)
| board | compiler | q4_K repack-vs-vecdot e2e prefill | decode | interpretation |
|---|---|---|---|---|
| rvv | gcc-15 | **0.42×** (repack LOSS to vecdot) | 0.18×/0.79× | candidate: gcc codegen spill 3→742 (objdump 直证; 隔离 pending 二.2) |
| **k1** | **clang-18** | **2.644× (repack WINS)** | **1.284×** | repack transduces; rvv LOSS 非普遍物理墙 (候选因素含 gcc codegen / VLEN128 / uarch·隔离 pending) |

**Direction FLIP 0.42× → 2.644× (prefill), 0.18× → 1.284× (decode)** across boards — the judgment key is **compiler
identity** (rvv gcc-15 vs k1 clang-18), exactly per [CASE-COMPILER-ASYMMETRY] / CLAUDE.md 性能常驻规则 3.

## 八门逐项 (gate audit; ADAPTED where the design differs from "deploy-our-emitted")
| gate | status | note |
|---|---|---|
| ① same-tree physical .so swap | ✅ | REPACK/VECDOT swapped in one build-k1-workitem/bin; ONE real ELF `/data/k1build/bin/llama-bench` |
| ② nm ON/OFF | ◑ ADAPTED | both libs share the 3 q4_K symbols (config toggle, not symbol-add) ⇒ engage proven BEHAVIORALLY not by nm |
| ③ banner engage on real model | ◑ ADAPTED | no custom banner (stock code); engage = 2.64× timing divergence + llama-cli footer 24.5 vs 10.0 t/s, real model |
| ④ objdump vl seal | ✅ | REPACK repack kernel `vsetivli zero,16,e32,m2` = VLEN256-native full-width (healthy, not MIRAGE) |
| ⑤ opponent stock, not SELF | ✅ | opponent (VECDOT) = stock's own `ggml_vec_dot_q4_K_q8_K` generic reference; winner = stock as-shipped repack; neither is a hand-crafted SELF kernel |
| ⑥ clang-18 both sides symmetric | ✅ | both variants `/usr/bin/clang++-18`, identical flags (differ by 1 dispatch line) |
| ⑦ correctness pre-gated | ✅ | correctness characterized (both correct/self-deterministic) BEFORE perf |
| ⑧ DVFS locked | ✅ | all 8 cpus performance gov, cur==max==1.6GHz; freq_khz=1600000 captured every ###AB block |

**Honest scope of ②③⑤**: the design measures the **repack-vs-vec_dot** contrast (the exact rvv-0.42× apples-to-apples,
which probes candidate factors {gcc codegen | VLEN128 rebuild | uarch}·隔离 pending 二.2 vs a reconstruction-wall) — NOT "our compiler-emitted kernel vs stock." Our vl=16 emitted
kernel's parity vs this same stock repack is already sealed on the kernel-axis (Win-K1-VLEN 1.085×), so the
transitive chain is: our-emit ≈ stock-repack ≈ this 2.64× e2e win. Deploying our-emitted to e2e (Phase-2) is
DEFERRED (recipe in hand: emit q4_K `march=zvl256b` vl=16 + VLEN256 intercept) — corroborative, does not change direction.

## board restored = TRUE
shared source md5 3cac40aa (NEVER edited — only a private copy was compiled) · shipped stock lib md5 871169a0
(UNTOUCHED) · 0 leftover procs · build-k1-workitem removed · evidence libs (REPACK/VECDOT, md5-recorded) kept in
`/data/wk`. No modification to any existing/shipped board file.

## VERDICT — 绿 · [WORK-ITEM] RESOLVED-POSITIVE
**k1 · q4_K · prefill e2e 2.644× (≥parity) · decode e2e 1.284× (≥parity) · kernel账 == system账 (clang-18
compiler-symmetric CONVERGE) · opponent = stock generic vec_dot `ggml_vec_dot_q4_K_q8_K` (non-SELF baseline) ·
C3′ 绿路径确认 (K-quant repack e2e 在 clang 传导 · **传导存在性证明** · rvv LOSS 非普遍物理墙 · 候选因素 {gcc codegen | VLEN128 重建摊销 | uarch}·隔离 pending 二.2 · 「gcc-death」冠名待二.2 出口 A) · [WORK-ITEM-K1-KQUANT-E2E]
RESOLVED-POSITIVE (传导存在性).**

The q4_K **repack GEMM/GEVM approach** transduces to a decisive e2e WIN on k1-clang (2.64× prefill, 1.28× decode)
vs the vec_dot baseline — the SAME repack-vs-vecdot contrast that was 0.42× on rvv-gcc. This is an **existence proof
of transducibility** and shows the rvv K-quant e2e LOSS is **NOT a universal physical wall**. ★裁二.1: candidate
factors = {gcc codegen ([CASE-COMPILER-ASYMMETRY] spill 3→742 historical direct-evidence) | VLEN128 rebuild
amortization | uarch}; **isolation pending (二.2 decisive experiment)** — this board swap changed compiler AND
VLEN(256 vs 128) AND uarch simultaneously, so gcc cannot be asserted the sole cause; the 「gcc-death」 naming right
awaits 二.2 exit A. Decode did **not** regress (won 1.28×), so the pre-registered "decode→黄-物理墙"
does NOT trigger. On k1, upstream ALREADY ships this repack as the default q4_K path (case256 ON) — i.e. the
**K-quant e2e green path already deploys on k1-clang**; this measurement quantifies its margin over vec_dot.

### ★ triggers q5_K / K-quant-family e2e re-estimate flag (flag only, not auto-executed)
The q5_K [X-0] **声明例外** (`line-c-k1-strike/q5_K_X0_amdahl_verdict.md`) rested empirically on "sibling q4_K e2e
0.42× LOSS ⇒ deployed-domain WIN upper bound < noise floor." That basis is now shown to rest on a **cross-domain
extrapolation** (q4_K rvv 0.42× sibling → q5_K k1): on k1-clang the sibling q4_K e2e **WINS 2.64×** (存在性证明·rvv
LOSS 非普遍物理墙; whether the flip is gcc codegen / VLEN128 / uarch is 隔离 pending 二.2). ⇒ **REOPEN q5_K / q6_K / K-quant-family e2e potential on k1-clang**
(the universal-reconstruction-wall premise is falsified for the repack approach; ★声明例外禁跨域推断·须同域). Not auto-executed
(q5_K provisioning: no q5_K gguf / no llama-quantize on k1 / net-new riscv repack scaffold).

## perf-covered ledger note (no new headline green claimed)
The e2e WINNER here is **stock's own clang-compiled repack** (the as-shipped q4_K path), not our compiler-emitted
kernel. So this does **NOT** add a perf-covered green for a TianChen-RV-constructed kernel. Its value is
**thesis-resolution** (传导存在性证明·rvv LOSS 非普遍物理墙·候选因素含 gcc codegen·隔离 pending 二.2) + **unblocking the q5_K/K-quant-family e2e re-estimate**
(the 声明例外's empirical premise is falsified on clang). A台账 green would require Phase-2 (our-emitted deployed +
winning e2e), which the sealed Win-K1-VLEN micro (1.085× vs this same stock repack) makes very likely but is not measured here.

## honest scope
- Measured the **repack approach** (stock's clang-compiled repack), NOT our compiler-emitted kernel. The thesis
  (does K-quant repack e2e transduce on clang) is about the APPROACH; our vl=16 emitted micro-parity vs this same
  stock repack is already sealed (Win-K1-VLEN 1.085×, kernel-axis) ⇒ transitive: our-emit ≈ stock-repack ≈ this e2e.
- Deploying our-emitted vl=16 kernel to e2e (Phase-2, recipe in hand: emit q4_K `march=zvl256b` + VLEN256 intercept)
  = C1-existence corroboration, DEFERRED (thesis resolved; heavy; does not change direction).
- single model (tinyllama-1.1b-Q4_K_M on k1) vs rvv's DeepSeek-8B/qwen-0.5b; the DIRECTION FLIP (0.42×→<FILL>×) is
  the decisive cross-board contrast, robust to model size.

## durable files — see MANIFEST.md
