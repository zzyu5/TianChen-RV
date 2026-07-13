# [l1-m2-iq4 / M2-tile] iq4_nl + iq4_xs repack GEMM — does the K-quant register-cliff tiling lever transfer to the CODEBOOK family?

**Board:** rvv / VLEN128 (openEuler), core 8, 2.6 GHz, governor=performance (measured 2026-07-09).
**Question (the tiling-lever transfer to the codebook family):** the q4_K S6 register-cliff step
(cell `l1-tile-s6-q4k-repack-gemm`) converted a full-unroll opening into throughput by staging idle
decode/min strips to STACK PANELS, dropping spill 84→23→3 and peak vreg v31→v30. M2 asks whether the
same lever has any purchase on the two CODEBOOK-decode formats — `iq4_nl` (flat block-32, single
fp16 scale, no min) and `iq4_xs` (super-block QK_K=256, 6-bit signed sub-block scale), whose 4-bit
nibble is an INDEX into a 16-entry non-linear int8 codebook decoded by a **memory `vluxei16` GATHER**
(not a register-resident bit reconstruction).

**Answer: NO — codebook-bound NULL, but by the OPPOSITE mechanism to q6_K.** The untiled codebook
GEMM is ALREADY register-light and ALREADY at the ≤32-vreg cliff (iq4_nl spill=7 / **v30**;
iq4_xs spill=73 / **v30** — never v31). There is simply no register pressure for the S6-style
stack-panel lever to relieve; throughput is gated by the memory codebook gather (`vluxei16`, ×64
nl / ×512 xs), which output-register tiling does not touch. The pre-registered gate
(spill→~0 ∧ MAC/cycle↑ ∧ wall passes T-N) cannot be met because its PRECONDITION — high untiled
spill / v31 saturation that tiling collapses — is absent.

## Provenance (no rebuild, no commit)
- **UNTILED** = HEAD **81a8050b** front-door export of `rvv-to-emitc-repack-gemm-iq4-nl-q8-0.mlir` /
  `-iq4-xs-q8-K.mlir`, lowered through **read-only `build/bin/tcrv-opt`** (`--tcrv-rvv-lower-to-emitc`)
  + `mlir-translate-20 --mlir-to-cpp` (ZERO opt/translate errors). iq4_nl GEMM = 2684 lines C;
  iq4_xs GEMM = 20464 lines C. Codebook decode = `__riscv_vluxei16_v_i8mf2(tcrv_iq4_*_repack_kvalues,…)`.
- **TILED** = **not built.** The iq4 family is still a dispatch-wired DIRECT-emitter (monolithic,
  pre-front-door-retirement); no tiling emitter delta exists and none was compiled into `tcrv-opt`
  (main tree + `build/` untouched, per "不碰主树 build"). The untiled objdump below forecloses the
  register-cliff lever, so an untiled→tiled A/B was not fabricated.
- Compile: `clang-17.0.6 -O2/-O3 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -x c++`.
- vs-opponent link: board's own `libggml-cpu.so` (SpacemiT GCC15.2 build), real dispatched
  `ggml_vec_dot_iq4_nl_q8_0` / `ggml_vec_dot_iq4_xs_q8_K` block-dot (iq4 has no repack trait @VLEN128
  → ggml falls back to exactly this per-(row,col) block-dot).

## 1. objdump seal (UNTILED) — the "is it weight-bound v31, or already at v30 cliff?" gate
| form | vsetvli | spill | reload | maxVreg | vwmacc | vlux | textB |
|---|---:|---:|---:|---:|---:|---:|---:|
| iq4_nl -O2 | 656  | **7**  | 8   | **v30** | 0 | 64  | 6954  |
| iq4_nl -O3 | 656  | 7      | 7   | v30     | 0 | 64  | 7002  |
| iq4_xs -O2 | 5162 | **73** | 149 | **v30** | 0 | 512 | 55202 |
| iq4_xs -O3 | 5162 | 72     | 147 | v30     | 0 | 512 | 55198 |

- **iq4_nl spill=7, v30** — already at the register cliff; spill is comparable to q4_K's *winning*
  S6 state (spill=3), an order of magnitude below q4_K golden (84) and two below q6_K-NULL (913).
- **iq4_xs spill=73, v30** — heavier super-block full-unroll, but still under q4_K golden's 84 and
  crucially still **v30, not v31** (a spare register survives). No saturation.
- **maxVreg v30 (both, both -O)** — the ≤32-vreg cliff is ALREADY REACHED in the untiled form. This
  is neither q6_K's "v31 weight-bound NULL" nor a "tiling drives v31→v30 HOLDS": there is no
  untiled→tiled transition to win because the untiled form starts past the cliff.
- **vwmacc=0** — the product path is not widening-MAC; the codebook value is gathered from memory
  (`vluxei16`) then multiplied, so weight "decode" lives in memory-indexed loads, NOT in a stack of
  live vector registers. This is exactly why register pressure stays low → **codebook = memory-gather
  bound, not register bound.**

**Why the lever can't transfer (the boundary):** q4_K's peak pressure WAS the idle min accumulator +
6-bit decode strips → staging them to panels collapsed spill and hit v30. q6_K's peak pressure was a
915-spill dual-plane 6-bit weight reconstruction that saturated v31 → the panel couldn't reach the
cliff. The codebook formats sit at the third pole: the decode is a `vluxei16` **memory gather**, so
there are no idle decode registers to stage and the untiled body is already at v30/low-spill. The
lever's mechanism (register-pressure relief) has zero purchase either way → NULL, codebook-bound.

## 2. A/B (untiled→tiled) — NOT MEASURED
No tiling emitter constructed (see Provenance). The untiled objdump (§1) already forecloses the
S6-style register-cliff lever, so no tiled binary was fabricated. Recorded honestly as "lever has no
precondition on this family," not as a measured ratio.

## 3. vs-opponent (repack GEMM vs ggml block-dot) — modest→moderate LOSS, our side gather-flat
Clean anchor nr=64 (K=2048, nc=512, cold paired, N=12; noisefloor 40 ns ≪ 27–36 ms region):
- **iq4_nl:** OURS 2.455 / OPP 2.934 → **0.837×** (seed C0FFEE); 2.458 / 2.907 → **0.846×** (seed BEEF01).
  nr=16 (N=10): OURS 2.440 / OPP 3.413 → **0.715×**.
  > **★校正（2026-07-14·a8ed3a09·[CASE-COMPILER-ASYMMETRY]）**：此 0.837×/0.846×/0.715× = **clang-ours -O2 vs gcc-shipped generic** 非对称编译 artifact（机判铁证：同 kernel.c·clang spill=7 / gcc spill=41 → gcc-ours 慢 3.7×）。**部署域（gcc-15.2 双侧对称）cold = 0.228×(nr64)/0.205×(nr16)**，匹配 g5 e2e prefill 0.217×（同域）。→ iq4_nl 部署域 verdict = **~0.22-0.23× LOSS**（本 0.837× 数**不可作部署主张**·仅存档为 asymmetry 病例）。
- **iq4_xs:** OURS 1.874 / OPP 2.874 → **0.652×** (nr=64); OURS 1.775 / OPP 2.870 → **0.618×** (nr=16).

Our GMAC/s is **flat across shapes** (iq4_nl 2.44–2.46; iq4_xs 1.72–1.87), the signature of a
throughput bound by the codebook gather (independent of nr). The opponent's GMAC/s rises at the
smaller shape (iq4_nl 2.93→3.41), so the parity ratio drops at nr=16 entirely on the opponent side.
Far better parity than q6_K (0.18×) but still a genuine LOSS; iq4_xs's heavier gather (×512, 73
spills) loses more than iq4_nl's (×64, 7 spills). Tiling for register relief cannot move a
gather-bound number.

## Pre-registered judgment → codebook family M2-tile does NOT hold (NULL, codebook-bound)
- spill: iq4_nl **7 / v30**, iq4_xs **73 / v30** — already at cliff, no v31 pressure to collapse ✗
- MAC/cycle: no lever precondition → no untiled→tiled gain available ✗
- wall: our throughput is codebook-gather-flat (2.44 / 1.87 GMAC/s), a 0.60–0.85× vs-opponent LOSS
  that register tiling does not touch ✗

This maps the K-quant register-cliff lever's transfer boundary (C3′ pattern-library): it wins where
peak pressure IS the decode/min strips (q4_K), is NULL-by-saturation where peak pressure is dual-plane
weight reconstruction (q6_K), and is NULL-by-already-at-cliff where decode is a memory codebook gather
(iq4_nl/iq4_xs). **Codebook-bound NULL confirmed on silicon** — via the "already at v30 / gather-bound"
mechanism, NOT the hypothesised "unexpected v30 register-cliff HOLDS."

## Loadavg / contention honesty
Board loadavg 2.0–2.5 throughout (a shared 64-core box; core-8 taskset pin, 2.6 GHz gov=performance).
vs-opponent is PAIRED (ours then opp back-to-back each cold round) so shared contention is common-mode
and cancels in the ratio; ratios reproduced across two seeds + two shapes.

## [NG-4] discipline
L1 kernel datapoint, **NOT a beat**. Eight [PERF-1] gates unwalked; single core; opponent =
single-thread block-dot proxy (not threaded `mul_mat`); our correctness = the construction oracle for
the iq4 direct-emitters (byte-exact @ silicon-validation-batch-1) — the vs-opponent A/B here times
data-independent steady state with exact strides, not a numerical check. Data-only cell; nothing
committed; main tree + `build/` untouched (UNTILED via read-only `tcrv-opt`; no TILED built).
