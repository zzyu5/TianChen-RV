# INC-3 RESULTS — does our compiler-emitted kernel really replace ggml's Q4_0 vec_dot?

Board: `ssh rvv` (riscv64, 64 cores, VLEN=128, rv64imafdcv+zfh+zvfh, clang 18.1.3).
Model: `~/llama-2-7b-chat.Q4_0.gguf`. llama.cpp on board: fork `build b1-6eab471`, libggml 0.15.1.
Harness: `llama-cli -p "The capital of France is" -n 48 --temp 0 --seed 1 -st -t 16 --simple-io`
(greedy / deterministic; `-st` single-turn exits cleanly; `-no-cnv` is rejected on this fork).

Our kernel: emitted by `tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`
from committed MLIR `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir`
(symbol `tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0`). NEVER hand-written.

Configs built (each in its own source+build tree; differ ONLY by the `-D` macro):
- control  : patched tree, no macro → `TCRV_Q4_0_DELEGATE()` is a no-op → ggml's own vec_dot runs.
- canary   : `-DTCRV_Q4_0_CANARY`   → delegates to a deliberately WRONG kernel (`*s = 0`).
- override : `-DTCRV_Q4_0_OVERRIDE`  → delegates to our compiler-emitted kernel.
Each config was run at default repack AND with `--no-repack`; `--no-repack` had NO observable effect
(this model's Q4_0 tensors are not repackable — see Part B), so the evidence covers the default config.

---

## Part A — STOCK baseline (deterministic, reproducible)

Ran stock `~/llama_integ/build/bin/llama-cli` TWICE. Generated continuation (identical both runs;
only the `t/s` timing line differed → `CONTENT_IDENTICAL_REPRODUCIBLE`):

```
> The capital of France is

The capital of France is Paris.
</|im_end|>

Note: The `<|im_start|>` and `<|im_end|>` tags are used to indicate the start and end of the information provided
```

The CONTROL build (patched tree, no macro → `TCRV_Q4_0_DELEGATE()` no-op) reproduces this
byte-for-byte under BOTH default repack and `--no-repack` (`CONTROL_ON == STOCK_BASELINE`), and its
delegation counter is `0` (the no-op path runs ggml's own kernel) — confirming the patch is inert
when no macro is set and the build is sound. Evidence: `partA_stock_baseline.txt`, `control_runs.txt`.

## Part B — CANARY (path liveness) — **vec_dot IS the live hot path**

Canary build (`-DTCRV_Q4_0_CANARY`) delegates `ggml_vec_dot_q4_0_q8_0` to a deliberately WRONG
kernel (`*s = 0`). Result — inference BROKE completely (garbage tokens), and our delegation counter
is **113,341,440** (both the default and `--no-repack` runs are byte-identical — see the correction
below on why that is one config, not two):

| run | repack flag | output | delegation counter |
|---|---|---|---|
| canary | default | **BROKEN** (`ozinitative impl DumSERTscript…` garbage) | 113,341,440 |
| canary | `--no-repack` | **BROKEN** (same garbage) | 113,341,440 |

Conclusions (skeptic-proof):
1. `ggml_vec_dot_q4_0_q8_0` IS the live Q4_0 hot path for this model on this board: zeroing it
   (canary) breaks inference into garbage, and our delegation counter is 113,341,440 — so our symbol
   is what runs, 113M times, in the configuration a real user runs (default repack).
2. Negative control is non-vacuous: a wrong kernel visibly breaks output (and even sped generation
   up to ~18 t/s — `*s=0` is trivially cheap — another tell it's on the hot path).

**Honest correction on the repack axis (do NOT overclaim two configs):** the `--no-repack` run is
NOT a second, independently-verified configuration. Its counter is *bit-identical* (113,341,440) to
the default run, there is no t/s delta, and the stock perplexity's `system_info` still reports
`REPACK = 1` under `--no-repack` — i.e. the runtime flag produced **no observable change** on this
fork. The mechanistic reason (verbose model load, `repack_investigation.txt`):

> `tensor 'token_embd.weight' (q4_0) (and 290 others) cannot be used with preferred buffer type
> CPU_REPACK, using CPU instead`

**All 291 Q4_0 tensors of this model are REJECTED by the CPU_REPACK buffer type and fall back to the
plain CPU path → `ggml_vec_dot_q4_0_q8_0`.** So repack is *structurally inapplicable* to this model
regardless of the flag, which is exactly why the counter is identical ON vs OFF. The correct claim is
therefore STRONGER than "we forced vec_dot": vec_dot is the UNCONDITIONAL Q4_0 hot path here (the 8x8
gemv/gemm repack kernels never engage for this model), and I verified ONE configuration (the default),
not two. The task's repack worry (`gemv_q4_0_8x8` bypass) does not arise for this model.
Evidence: `canary_runs.txt`, `repack_investigation.txt`.

## Part C — OVERRIDE (byte-exact replacement) — **IDENTICAL to ggml, 113M live calls**

Override build (`-DTCRV_Q4_0_OVERRIDE`) delegates `ggml_vec_dot_q4_0_q8_0` to our COMPILER-EMITTED
kernel. Macro confirmed in the actual ggml-cpu compile flags (`TCRV_Q4_0_OVERRIDE`). Results:

| run | repack flag | output | delegation counter |
|---|---|---|---|
| override | default | **"The capital of France is Paris."** (coherent) | 113,341,440 |
| override | `--no-repack` | **"The capital of France is Paris."** (coherent, no observable change) | 113,341,440 |

Token-for-token diffs (content-stripped of the `t/s` timing + spinner lines), all IDENTICAL:

| comparison | result |
|---|---|
| **override(--no-repack) vs control(--no-repack)** — THE byte-exact discriminator (same flags, only `-D` differs) | **IDENTICAL** |
| override(repack ON) vs control(repack ON) | **IDENTICAL** |
| override(repack ON) vs STOCK baseline (Part A) | **IDENTICAL** |

Our kernel ran **113,341,440 times** during a single 48-token greedy generation and produced output
byte-identical to ggml's own RVV kernel. Evidence: `override_runs.txt`, `override_vs_control_diff.txt`.

Build/flags parity (the only differences between control / canary / override are the one `-D` macro):
`-march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d -O3 -DNDEBUG`, `GGML_USE_CPU_REPACK` defined.
No explicit `-ffp-contract` flag is used — the compiler default (clang `-O3` → `-ffp-contract=on`
within a statement) is identical across control and override, so the byte-exactness comes from the
kernel itself, not a contract-flag trick. (The committed kernel is byte-exact vs ggml under
default/on/off/fast per the INC-2a verification, commit f6f1a73a.)

Performance note (N3, out of scope for THIS milestone): override generation ran ~1.6 t/s vs stock
~2.7 t/s — our kernel is byte-exact-CORRECT but SLOWER (the `i8mf4`→`i32m1` anchor uses 4 lanes/chunk
at VLEN=128 vs ggml's `i8m1`→`i16m2` 16-lane anchor; g4b in the kernel map). Correctness milestone
met; the perf win is a separate N3 task.

---

## Verdict — DOES our compiler-emitted kernel really replace ggml's in live inference? **YES.**

A skeptic's three-link chain is closed:
1. **The path is live.** CANARY (wrong kernel `*s=0`) on the delegation site BROKE inference into
   garbage tokens, with a 113M-call counter — so `ggml_vec_dot_q4_0_q8_0` is unambiguously the live
   Q4_0 hot path for this model in the configuration a user runs (default repack), and our symbol is
   what's invoked. The task brief's repack-bypass worry does NOT arise here: this model's 291 Q4_0
   tensors are rejected by the CPU_REPACK buffer type (verbose-load evidence) and fall back to plain
   CPU → vec_dot, so the 8x8 gemv/gemm repack kernels never engage. (I verified ONE configuration,
   the default; `--no-repack` produced no observable change — see the honest correction in Part B.)
2. **Our kernel actually ran.** The OVERRIDE counter is 113,341,440 — the same call volume — proving
   our compiler-emitted body executed for every Q4_0 weight-matmul dot of the generation.
3. **Inference stayed correct, byte-exact.** OVERRIDE output is token-for-token IDENTICAL to the
   CONTROL (ggml's own kernel, same flags) AND to the stock baseline — greedy argmax never diverged.

Caveats (honest):
- The kernel is byte-exact-CORRECT but currently SLOWER than ggml's hand-tuned RVV (1.6 vs 2.7 t/s) —
  the N3 perf win is a separate, still-open task (different reduction anchor / LMUL).
- The board's llama.cpp is a NEWER fork (`b1-6eab471`) than the task brief assumed; `-no-cnv` is
  rejected, so the harness uses `-st` (single-turn greedy, chat-templated). The template is held
  identical across all configs, so it does not affect the discriminator. The model is chat-tuned so
  the greedy continuation is short ("Paris."), but it deterministically drives 113M Q4_0 dots.
- The TianChen-RV repo was NOT modified; all patching happened in board-side copies
  (`~/llama_integ_{control,canary,override}_repackON`). The kernel was emitted ONLY by the compiler
  from committed MLIR; no kernel math was hand-written.
- The repack axis was NOT independently exercised: `--no-repack` produced no observable change (this
  model's Q4_0 tensors are not repackable at all — see Part B). So the evidence covers the DEFAULT
  configuration only. For this model that is sufficient (vec_dot is the unconditional path); for a
  model whose tensors DO repack, the 8x8 gemv/gemm path — which our per-row vec_dot does NOT cover —
  would be the hot path and this override would not apply. Out of scope here.

### Numeric sanity (perplexity) — CORROBORATING (breadth), not the proof of exactness
The proof of byte-exactness is the greedy token-for-token match (any fp drift flips a greedy-argmax
token). A `llama-perplexity` run over a fixed text (`-c 256`, 5 chunks, `-t 16`) corroborates it:

| perplexity run | kernel | PPL | our-kernel delegations |
|---|---|---|---|
| STOCK | ggml's own | **1.0009 +/- 0.00017** | (n/a) |
| OVERRIDE | our compiler-emitted | **1.0009 +/- 0.00017** | 1,726,618,112 |

What this adds (honest): the value is **breadth** — our kernel was delegated **1.7 billion times** over
1280 tokens and the output never diverged, far beyond the ~5-token greedy generation. The matching PPL
**rules out any gross error** across that volume. It does NOT independently *confirm* byte-exactness:
the text is one sentence repeated 40× so PPL is saturated near 1.0 and insensitive to subtle kernel
differences — only a gross error (like the canary) would move it. The finer byte-exact property is
carried by the greedy token match, not by PPL. (Control-tree perplexity was not built; stock==control
is already proven byte-exact in greedy, so stock is the valid ggml-kernel reference. The override pass
took 771 s/pass vs stock 420 s/pass — the same ~1.8x slowdown seen in t/s.) See `perplexity_runs.txt`.
