# T-X six-column evidence — X-SCALAR real no-V silicon (超锐/scalar board) · S1 duel (LIVE EVIDENCE)

> **Nature**: **real-silicon (a) physical vector-absent profile** — the 超锐 (scalar) board is genuine
> no-V hardware (isa `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`, no `v`/`zve*`,
> clang-18 installed). This **retires the B4 rehearsal's central premise** ("hand has no physical
> no-V silicon" → rv64gc narrow-exempt path (b)). Column 1 is now **(a) physical no-V silicon**;
> the narrow-exempt tag is removed; the wording iron-line "validated on silicon" holds **literally**.
> **Wording iron-line (applied verbatim)**: "targeting rv64gc (vector-absent profile), validated on silicon".
> **Domain**: [X-SCALAR] enablement (correctness / zero-vector machine-check / independent witness / S1 duel).
> **[L-6]**: scalar is NEVER a contribution baseline — the S1 ratio is diagnostic, **NON-Win**, does NOT
> enter the system ledger / perf-covered / denominator. Column 5 never escalates to Win.
> **Definition authority (layer belonging)**: **[ISSUE-103] · 待裁** — B4 said §T-X lives in
> `.trellis/spec/testing/…`, but the six-layer refactor has NO testing layer. Conservative default:
> this file is the LIVE evidence in `experiments/` (data cell); the spec-side definition pointer is
> pending (measurement? evidence?). **No spec layer/file created by this task.**
> **Harness**: `tools/bench/cells/scalar_vec_dot.sh` (ISSUE-090 contract: no repo-side writes, all stdout).
>   Full bench-driven invocation of this cell is gated on **[ISSUE-104]** (runner scalar block=ISSUE-061,
>   op→harness name resolution, gemm_tile-specific parser, no scalar master column / [L-6] should have none).
> **Assets** (data cell, read-only): `scalar_s1_driver.cpp` (S1 driver linking the REAL deployed ggml
>   symbol — upgrade over the A3 rehearsal's inline reimpl) · `weft_scalar_tq2_0_kernel.cpp` (owned
>   emitter output) · `fp16util.cpp` (pure-integer soft-fp16 oracle primitive).

| # | Column | Filled evidence | Status |
|---|--------|-----------------|--------|
| 1 | **Target identity** | **超锐 (scalar) board**: Fedora42, `uname` 6.19.9 riscv64, 8-core. **verified isa (fingerprint) = `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`** — **no `v`, no `zve*` = (a) PHYSICAL vector-absent silicon** (not "V-board-run-as-noV"). Compiler: **clang 18.1.8** (`/usr/bin/clang-18`, riscv64-redhat-linux-gnu, board shipped). narrow-exempt tag **REMOVED** (B4 PR-1 procurement now RESOLVED-BY-FACT). | **(a) physical · measured** |
| 2 | **Build** | `clang-18 -march=rv64gc -mabi=lp64d -O2`. Kernel = **owned `weft_scalar` emitter output** (`weft_emitc_tq2_0_kernel_scalar_fallback_first_slice`, op `weft_scalar.tq2_0_q8_k_vec_dot`), NOT a fallback stub. Two fold regimes recorded: **`-ffp-contract=on`** (symmetric with the deployed ggml build — matches the rvv/k1 bench harness convention) **AND `-ffp-contract=off`** (compiler-neutral ZERO-MODEL). Single-world clang-18 (both OURS and OPP the board-shipped chain). Build GREEN. | `measured` |
| 3 | **Zero-vector machine-check** (static, board-independent) | `# ZEROVEC` on owned kernel `.o`: **scalar_ins=191, vector_mnemonic=0, vset=0, cpop_clmul=0, riscv_intrinsic=0, weft_rvv_sym=0, fp16_libcall=0**. The fp16 scale read compiles to inline scalar rv64gc (no `__extendhfsf2` libcall on this chain). Deployed ggml lib whole-file: **v_mnemonic_lines=0** (confirms no-V build ⟹ dispatch thunk == generic = true deployed scalar path, 板册 §3.7). | **PASS** (structural) |
| 4 | **byte-correct** | On scalar silicon, **ZERO-MODEL [K-5]**: OURS byte-exact vs a **libcall-free integer oracle** (pure-integer `weft_h2f`, int64 contraction, recomputed from actual input bytes — independent of OURS internals) **AND** byte-exact vs the **real deployed ggml `ggml_vec_dot_tq2_0_q8_K`**. Under symmetric `contract=on`: **ours == ggml == int-oracle ALL true** (e.g. seed 0xD00D `0x46a18a33`; seed 0x1357 `0xc50a8972`). Under `contract=off`: ours == int-oracle **always** byte-exact; ours-vs-ggml byte-exact when the deployed lib's FMA float-fold rounding coincides (seed 0xD00D all agree `0x46a18a31`), else differs by ≤ a few ULP (seed 0x1357: ours/int `0xc50a896c` vs ggml `0xc50a8966` = 6 ULP) — residual = the deployed lib's **FMA fold contraction** (IEEE-legal, [K-5] float-path ULP clause), **not a kernel defect**. fp16 primitive `weft_h2f` validated **9/9** vs textbook IEEE-754 half. **Anti-hollow** (three-way cross-check bites): INJECT=1 (corrupt OURS out) → byte-exact false, isolates OURS (BITES-OK); INJECT=2 (corrupt oracle) → byte-exact false, isolates oracle (BITES-OK). | **byte-exact GREEN · measured** |
| 5 | **Per-competitor produced-kernel count** | **codegen competitors: heteroMx = 0, xDSL-RVV = 0, 10x-IREE = 0** (documented scope covers no vector-absent rv64gc ggml-ternary produced kernel). **codegen 竞品产出为零.** The sole existing alternative = the board's **hand-written scalar reference `ggml_vec_dot_tq2_0_q8_K` (== `_generic` on no-V), which IS runnable and WAS run** as the S1 opponent (真部署对手·对手法 §3.4). **S1 cold duel** (N=25, 2-seed, flush 32 MiB, core idle 100%, symmetric contract=on): seed 0x1357 `ratio_cold_X`=**1.0650** (ours 844905 / ggml 899805 ns, IQR 1.5/2.0%); seed 0xACE2 `ratio_cold_X`=**1.0583** (ours 846105 / ggml 895406 ns, IQR 1.7/2.3%). Adversary-class词表: **标量类** (scalar-ref sanity). **Explicitly NON-Win** (enablement, [L-6]) — ratio recorded, **NOT** in system ledger / perf-covered / denominator. | `measured` (diagnostic · NON-Win) |
| 6 | **Reference-path disclosure** | **Deployed == proven variant**: the single `weft_scalar.tq2_0_q8_k_vec_dot` typed body, lowered via the shared backend-emission registry (no `--weft-materialize-emission-plans`; scalar plugin fail-closes as Unsupported). The measured/timed binary and the byte-exact-proven binary are the same `s1on` (contract=on) build. **Enablement, no performance claim.** OPP disclosure: the S1 opponent is the board's deployed `ggml_vec_dot_tq2_0_q8_K`, which on the no-V build collapses to `_generic` (真部署标量派发). **[F-6] relation**: this physical-silicon witness **strengthens, does NOT replace** the committed synthetic capability instance (the closure∩rvv.*=∅ gate needs no board). | `measured` (physical · enablement) |

## Channel roll-up

- **Real silicon rv64gc (超锐/scalar)**: build GREEN ✓ · zero-vector objdump ✓ (0 across all scans, kernel.o + deployed lib) · byte-correct ✓ (ZERO-MODEL: ours == independent int-oracle == real deployed ggml, byte-exact under symmetric fold) · anti-hollow ✓ (three-way isolates the faulted impl) · S1 duel = **1 pairing** (OURS vs same-board same-clang18 REAL deployed ggml scalar reference), `ratio_cold_X` = 1.058–1.065 over 2 seeds → **ours marginally faster, NON-Win / enablement** ([L-6]).
- **Competitor**: heteroMx 0 / xDSL-RVV 0 / 10x-IREE 0 produced kernels (documented scope); ggml scalar reference runnable and run. **codegen 竞品产出为零** (iron-line honored; NOT "竞品跑不了").
- **Toolchain note (honest)**: ggml's `GGML_CPU_FP16_TO_FP32` on the rv64gc (no-zfh) generic path uses a lookup table populated by `ggml_cpu_init()`; without that call x.d folds to 0 (silent all-zero output). The driver calls it before any ggml vec_dot. This is a downstream ggml-init requirement, surfaced and handled, not a kernel issue.

## Honest limitations

1. **S1 is enablement, NON-Win** ([L-6]): the ~1.06× is ours-vs-deployed-scalar-generic parity-plus; scalar is never a contribution baseline. No beat claim rides on it; it does not enter perf-covered/denominator.
2. **Symmetric fold required for the duel**: ours-vs-ggml is bit-identical only under matched `-ffp-contract` (the deployed ggml built with FMA fold). A fair cold duel uses the symmetric `contract=on` build (rvv/k1 harness convention). The compiler-neutral ZERO-MODEL (`contract=off`, ours vs independent int-oracle) is the authoritative correctness gate and is byte-exact regardless.
3. **Competitor channel** used documented-scope, not literal install-and-invoke of IREE/xDSL toolchains (out of scope).
4. **bench-runner integration is gated** ([ISSUE-104]): the harness is contract-compliant and directly runnable, but the official `bench` scalar path is blocked (ISSUE-061), the runner's op→harness resolution and parser are gemm_tile-specific, and there is (correctly, per [L-6]) no scalar master column — so S1 data lands here in `experiments/`, not the main table.
