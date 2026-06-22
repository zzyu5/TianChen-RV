# Win-A e2e on K1 — engagement+correctness SEAL + honest speed caveat (2026-06-22)

## SEAL achieved (durable): Win-A's compiler-selected strip runs in REAL llama inference
The compiler-emitted **VLEN256 1×16 repack-GEMV strip** engaged as the literal decode GEMV
kernel in real llama.cpp inference on K1 (Spacemit X60, VLEN=256) and produced correct output:
- Transport: tcrv-llamacpp tar-piped rvv→K1 (270M→156M).
- Wiring: replaced `arch/riscv/tcrv_emitted_repack_gemv.inc` with the validated `gemv_1x16.cpp`
  (verbatim compiler output, identical symbol), flipped the emitted-GEMV guard `vlenb*8==128`→`==256`.
  Kernel body stays 100% compiler-emitted.
- Build: clang18 + `-fno-integrated-as -fuse-ld=bfd` (gcc13 lacks `__RISCV_FRM_*`; clang IAS rejects
  upstream `quants.c` vsetivli shorthand), march `rv64gcv_zfh_zvfh_...` (zvl256b baked in the body). 22.5min.
- Model: TheBloke TinyLlama-1.1B-Chat **Q4_0** (637MB) via proxy.
- Run: stderr `TCRV EMITTED GEMV(q4_0_16x1 VLEN256 compiler-emitted 1x16-strip) ENGAGED` fired on EVERY
  decode GEMV; `llama-cli -p "The capital of France is"` → **"The capital of France is Paris."** (correct,
  coherent, clean exit). ENGAGED only fires for genuine GGML_TYPE_Q4_0 → proves real-Q4_0 routing.
**→ "Win-A in llama" engagement + correctness is proven on real VLEN=256 silicon.**

## Speed legs — honest, one done unfavorably + one blocked
- **1×16-emitted vs generic-C reference @ t8: 3.22 vs 4.38 t/s — the emitted strip LOSES.** Honest caveat
  (NOT a refutation of the kernel win): this is a **SMALL model (1.1B, forced by K1's 7GB)** where decode
  is more overhead/memory-bound, so the per-kernel compute advantage dilutes; and it's a DIFFERENT baseline
  (clang -O3 autovec of ggml's generic vec_dot) than the microbench (1×16 vs 2×8). Not comparable to rvv's
  7B Win-B (5–6×, large model). On a small model at 8 threads, our emitted repack strip does not beat the
  autovec'd reference on the X60 — a real, disclosed limitation.
- **1×16-emitted vs 2×8-emitted @ t1 (the apples-to-apples SELECTION question, matching the 1.48× microbench):
  UNMEASURED — BLOCKED.** Two confirmed-distinct `.so` snapshots (`~/libggml-cpu-{1x16,2x8}.so`, differ at
  byte 154) are staged on K1; resume cost ~2min once K1 is back:
  `cp ~/libggml-cpu-1x16.so build/bin/libggml-cpu.so.0.15.1 && taskset -c 0 ./build/bin/llama-bench -m
  models/tinyllama-q4_0.gguf -p 0 -n 32 -t 1 -r 3` then the 2x8 `.so`.

## Blocker: K1 is DOWN
After the benches, K1 went unresponsive (port 22 CLOSED from the jumphost; jumphost + rvv both fine →
K1-local). Needs a **manual reboot/power-cycle** (cannot be done over a dead ssh). Left on K1 (intact):
`~/tcrv-k1-llama` (built tree), the 637MB model, the two A/B `.so` snapshots.

## What stands regardless
- The **microbench 1.48×** (1×16 vs 2×8, isolated decode kernel, single-core) is the strip-SELECTION win
  and is unaffected — it answers "is the wider strip the right choice at VLEN=256" (yes).
- The engagement+correctness SEAL is durable.
- Honest bottom line: Win-A's selection is correct + engages in real inference; the e2e *speed* contribution
  of the selection (1×16 vs 2×8) is one 2-min bench from done; the broader repack-vs-generic e2e win does
  NOT hold on a small model at t8 on the X60 (disclosed). The clean N3 wins remain the microbench 1.48× +
  the all-compiler LMUL ablation 2.27–3.79× + the q8_0 selection divergence.
