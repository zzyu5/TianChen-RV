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
  MEASURED 2026-06-23 — 1×16 WINS 1.31×.** K1 rebooted (back up); ran both staged `.so` snapshots
  back-to-back, `taskset -c 0 llama-bench -m models/tinyllama-q4_0.gguf -p 0 -n 32 -t 1 -r 3`:
  | arm (both compiler-emitted repack, t1, VLEN=256) | tg32 t/s | ENGAGED marker |
  |---|---|---|
  | **1×16 strip** (gearbox's VLEN256 selection) | **2.12 ± 0.00** | fired |
  | 2×8 strip (the VLEN128 shape) | 1.62 ± 0.00 | fired |
  → **1×16 / 2×8 = 1.31× e2e in real llama decode.** Both arms are verbatim compiler output (same kernel,
  only strip width differs), both ENGAGED markers fired → this is the pure strip-SELECTION win, IN llama,
  on real VLEN=256 silicon. It corroborates the isolated-kernel **1.48× microbench** (e2e slightly lower,
  expected: more memory/overhead at the model level dilutes the per-kernel compute win). **This is distinct
  from the repack-vs-generic X60 loss below** — that compares repack-path vs autovec'd-generic (a different
  question); this compares two strip widths within the repack path. The compiler-automatic VLEN→strip
  selection is the right call at 256, and now demonstrably so end-to-end. Log: `k1_1x16.out` / `k1_2x8.out`.

## Blocker: K1 is DOWN
After the benches, K1 went unresponsive (port 22 CLOSED from the jumphost; jumphost + rvv both fine →
K1-local). Needs a **manual reboot/power-cycle** (cannot be done over a dead ssh). Left on K1 (intact):
`~/tcrv-k1-llama` (built tree), the 637MB model, the two A/B `.so` snapshots.

## Mechanism RESOLVED (2026-06-22): X60-microarch-specific, NOT small-model
Tested the small-model hypothesis on rvv (UP) instead of asserting it — same TinyLlama-1.1B Q4_0,
same t8, repack ON vs OFF:
| board (same 1.1B, t8) | repack ON | repack OFF | ratio |
|---|---|---|---|
| rvv / SG2044 | 22.25 t/s (ENGAGED) | 8.93 t/s (generic) | **2.49× WIN** |
| K1 / X60 | 3.22 (1×16 strip) | 4.38 (ref) | **0.74× LOSS** |
Holding model size + thread count fixed and changing ONLY the board flips a 0.74× loss into a
2.49× win → **small-model dilution is REJECTED; the K1 regression is X60-microarch-specific**
(strong clang-18 autovec of the generic q4_0 path on X60; our repack strip wasn't tuned to beat it).
On SG2044 the same repack engine wins 2.49× even at 1.1B (and 5–6× at 7B — the win shrinks with size
but does not invert). Honest caveat: the two boards' OFF baselines aren't byte-identical (rvv OFF =
no-repack; K1 OFF = strip-guard-off on the still-repacked layout) — but the symmetric load-bearing
fact (rvv repack-vs-its-own-no-repack wins on the small model) is what rejects board-independent dilution.
Log: `rvv-smallmodel-repack-toggle.log`. rvv restored to repack-ON (md5-verified).

## The real N3 implication (the honest takeaway)
This is itself a **capability/resource-aware-tune signal**: "always use the repack path" is the WRONG
static choice on X60 (where it regresses); a measured>static decision at the PATH level (repack-vs-generic,
not just strip-width) would pick generic on X60 and repack on SG2044. Same pattern as the q8_0 finding
(static wrong on every chip → measurement needed). So the X60 loss doesn't weaken N3 — it points to the
next N3 layer: the tune should select the *vehicle* (repack vs autovec'd-generic) per real microarch, not
just the strip width within the repack. Our gearbox currently tunes within-repack; per-microarch
path-selection (driven by real measurement like this) is the honest next step.

## What stands regardless
- The **microbench 1.48×** (1×16 vs 2×8, isolated decode kernel, single-core) is the strip-SELECTION win
  and is unaffected — it answers "is the wider strip the right choice at VLEN=256" (yes).
- The engagement+correctness SEAL is durable.
- Honest bottom line: Win-A's selection is correct + engages in real inference; the e2e *speed* contribution
  of the selection (1×16 vs 2×8) is one 2-min bench from done; the broader repack-vs-generic e2e win does
  NOT hold on a small model at t8 on the X60 (disclosed). The clean N3 wins remain the microbench 1.48× +
  the all-compiler LMUL ablation 2.27–3.79× + the q8_0 selection divergence.
