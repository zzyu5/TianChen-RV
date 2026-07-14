# NOTES — G3 四.1 whole-model e2e fusion measure

## What this cell is
The board e2e test the 裁决 asked for after the isolated G2 fusion micro (`../g2-fuse-rms-norm-mul/`)
proved the rms_norm→mul fusion on the memory axis (1.308× wall, 256 MiB DRAM eliminated, bit-exact).
Question: does that fusion move a **real llama forward pass** (every-token attn_norm/ffn_norm)?

## The pivotal discovery (drove the whole design)
While planning the "patch ggml to call our fused kernel" integration, code-read of the board's ggml
(f3e1828) showed **upstream ggml already fuses RMS_NORM+MUL**:
- `ggml-cpu.c:3130 ggml_cpu_try_fuse_ops` → matches `{GGML_OP_RMS_NORM, GGML_OP_MUL}` via
  `ggml_can_fuse`, calls single-pass `ggml_compute_forward_rms_norm_mul_fused` (`ops.cpp:3757`).
- ON by default; the ONLY gate is `getenv("GGML_CPU_DISABLE_FUSION")` (ggml-cpu.c:3978).
- Verified in the compiled artifact: `libggml-cpu.so` carries both the env string and the fused symbol.

So the task's premise — "fused-build vs stock (unfused) build" — is **structurally void**: there is no
unfused stock. This is the honest "ggml-routing blocker", except it's not a wall, it's a **reframe**:
the correct symmetric e2e A/B is **ggml's fusion ON vs OFF** on one binary (env toggle). That measures
exactly the transformation our compiler emits — the isolated cell proved our emitted kernel is bit-exact
to the two-pass, hence arithmetically identical to ggml's fused kernel, i.e. a drop-in equivalent. We did
NOT hot-swap our kernel into ggml (that would only show [NG-2] parity with ggml's already-present fusion,
at large integration risk — the historical e2e-seal difficulty). The env-toggle is strictly more
informative and toolchain-symmetric by construction (same .so; only one getenv branch differs).

## Protocol actually run (preflight-0 anti-confound)
- **Symmetry**: A/B = same `llama-bench` + same `libggml-cpu.so` + same toolchain. Preflight asserts
  fusion-knob + fused-symbol present in the lib (fail-closed). No compiler asymmetry possible.
- **Cold**: one dropped warmup per arm (pages in mmap, settles DVFS), then measured.
- **Paired-interleaved**: on,off,on,off across PASSES=2 (between-pass = T-N floor).
- **N≥10**: REPS=5 × PASSES=2 = 10 samples/arm/phase (llama-bench `-r 5 -o json` gives per-rep samples).
- **Phase-split**: llama-bench native pp (prefill) + tg (decode) from one call.
- **DVFS guard**: taskset cores 8-15, gov=performance, freq printed per call (all 2.6 GHz).
- **T-N**: aggregator (`aggregate_fuse_e2e.py`) flags SIGNIFICANT only if |Δ|>2×floor AND |Δ|>2% AND
  IQR bands non-overlapping.

## Results (see board_measured.md for tables)
- Numeric: fusion ON vs OFF greedy completion **byte-identical** (sha match). Value-preserving e2e.
- Phase-split (tinyllama-q4_0): prefill Δ = −0.11% (ns), decode Δ = −0.27% (ns). Both sub-noise, IQR
  overlapping → **fusion e2e effect ≈ 0**.
- Amdahl (task-clock decode profile): `rms_norm_mul_fused` = **0.05%** of decode; `q4_0_q8_0` matmul =
  **90.2%**. Amdahl ceiling of the fusion = +0.05% < noise floor → e2e null is **structural**.

## Why the isolated 1.31× doesn't transplant (mechanism)
The isolated micro used 128 MiB tensors (>64 MiB L3) so the intermediate y[] genuinely round-tripped to
DRAM; the fusion removed 256 MiB of real DRAM traffic → 1.31× on the memory-bandwidth wall. At
whole-model norm sizes the intermediate is **cache-resident** (8 KiB/norm decode, ~1 MiB prefill) — no
DRAM round trip exists to remove. The 1.31× lived on DRAM traffic that is absent e2e. Textbook
"kernel-micro win doesn't transduce to memory-bound e2e", here quantified to the 0.05% Amdahl slice.

## Gotchas captured (for future e2e cells)
- **llama-cli is now a chat frontend** in this ggml: it ignores `-no-cnv`, and under nohup (stdin=EOF)
  loops forever printing empty `> ` prompts → a runaway that filled /tmp with 200+ MB. Use
  **`llama-completion`** for non-interactive greedy completion. (Killed via explicit PID — note
  `pgrep -f llama-cli` self-matches the shell running it; use `[l]lama-cli`.)
- **perf_event_paranoid=2** blocks PMU sampling (`-e cycles` → "no samples"). Use software
  **`-e task-clock`** for unprivileged function time-share attribution.

## Restore
Nothing to restore: main tree + build/ UNTOUCHED, nothing committed, no git stash. Board scratch under
`/tmp/g2e2e` (scripts + logs) only; models/binaries pre-existing and unmodified.
