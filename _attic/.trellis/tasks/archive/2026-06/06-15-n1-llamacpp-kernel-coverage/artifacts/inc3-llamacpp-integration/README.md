# INC-3 — really replace llama.cpp's Q4_0 vec_dot kernel in live llama-2-7b inference

Goal: prove our COMPILER-EMITTED kernel (`tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_…`,
emitted by `tcrv-opt` from the committed MLIR `rvv-to-emitc-q4-0-q8-0-block-dot.mlir`)
really stands in for ggml's `ggml_vec_dot_q4_0_q8_0` during live Q4_0 inference on `ssh rvv`.

## Assets (this dir)
- `tcrv_q4_kernel.cpp`  — the compiler-emitted kernel (NEVER hand-edited; regenerate via the
  `tcrv-opt … | mlir-translate-20 --mlir-to-cpp` pipeline in the parent task brief).
- `tcrv_q4_shim.cpp`    — global call counter + the deliberately-wrong CANARY kernel (`*s=0`)
  + an at-exit reporter (`[TCRV_Q4_INTEG] … delegated … N times`).
- `tcrv_q4_integ.h`     — the `TCRV_Q4_0_DELEGATE()` macro: no-op stock; canary under
  `-DTCRV_Q4_0_CANARY`; our kernel under `-DTCRV_Q4_0_OVERRIDE`.
- `apply_patch.sh`      — installs the 3 files + patches `arch/riscv/quants.c` (one include +
  one `TCRV_Q4_0_DELEGATE();` at the top of `ggml_vec_dot_q4_0_q8_0`) + `CMakeLists.txt`
  (adds the 2 TUs to the riscv source list). Idempotent.
- `build_config.sh`     — builds one config (stock|control|canary|override) × REPACK(ON|OFF)
  into its OWN source+build tree (no stale flags). Records the exact ggml-cpu compile flags.
- `run_infer.sh`        — runs the fixed greedy prompt against a given binary.

## Environment finding (deviation from the task brief)
The board's `~/llama_integ` is a NEWER llama.cpp fork (`build b1-6eab471`, libggml 0.15.1)
where `-no-cnv`/`--no-conversation` is REJECTED at runtime ("please use llama-completion
instead") and only `llama-cli` + `llama-perplexity` are built. The deterministic single-shot
path used here is `--single-turn (-st) --temp 0 --seed 1` (greedy, exits after one turn).
This is a chat-tuned model, so `-st` wraps the prompt in the llama2 chat template; that is
held IDENTICAL across all configs, so it does not affect the discriminator (canary-breaks /
override==control). This fork exposes a runtime `--repack/--no-repack`, but it had NO observable
effect for this model: all 291 Q4_0 tensors are rejected by the CPU_REPACK buffer type and fall back
to plain CPU → `ggml_vec_dot_q4_0_q8_0`, so vec_dot is the UNCONDITIONAL Q4_0 hot path here and the
experiment verifies ONE configuration (the default). See RESULTS.md Part B correction +
repack_investigation.txt.

See RESULTS.md for the A/B/C evidence and the honest verdict.
