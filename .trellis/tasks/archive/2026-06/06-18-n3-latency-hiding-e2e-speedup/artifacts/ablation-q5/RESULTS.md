# q5_0 / q5_1 block-dot autotuner ablation — the GENERALITY column (06-18)

Measurement-only extension of the INC-10 measured-block-dot campaign
(`06-15…/artifacts/inc10-measurement-tuner/`) to the **q5_0 / q5_1** kernels, which
were previously byte-exact-validated (inc21/inc22) but **never perf-measured**. The
q5_0/q5_1 schedule passes (`tcrv-rvv-materialize-q5-0-schedule` /
`…-q5-1-schedule`) already exist with `dump-candidates` + `march`; this task only
drives them through the same FILTER-then-RANK on-board tuner. **No repo source was
modified** — all changes live in this artifacts dir.

## Method (identical to inc10, gentle knobs)

For each (kernel, capability-march, n):
1. `--dump-candidates` → the legal shape set (lmul × factor × elision) — the single
   source of truth for the capability+budget prune (the driver never re-implements
   the Zvl128b rule).
2. Emit each candidate's C: `tcrv-opt <stamp shape attrs>
   --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`, renamed to a
   distinct per-candidate symbol.
3. **FILTER**: gate each candidate byte-exact vs **ggml's REAL q5 RVV kernel** (a
   SEPARATE translation unit, `kern_ggml`, so it is `-ffp-contract=fast`-contracted
   exactly as the standalone kernel that ships) over 9 n × many trials. Drift even
   ~1 ULP ⇒ INELIGIBLE, dropped BEFORE timing. The ggml-real recipes are
   transcribed verbatim from the inc21/inc22 byte-exact validators (which transcribe
   `llama.cpp ggml-cpu/arch/riscv/quants.c` q5_0 :328-379, q5_1 :382-433):
   `kernels/kern_ggml_q5_0.cpp`, `kernels/kern_ggml_q5_1.cpp`.
4. **RANK**: time the eligible survivors best-of-N min, taskset-pinned, vs the ggml
   baseline (oracle context only — I9: ggml is NOT a speedup baseline).

Driver: `tune_block_dot_q5.py` (a port of inc10's `tune_block_dot.py`); base MLIR
templates `q5_0_base.mlir`, `q5_1_base.mlir` (mirror the inc21/inc22 op inputs with
`KERNELNAME/VARIANTNAME/SHAPE` placeholders).

## Board / clock / provenance

- Host `ubuntu` (ssh alias `rvv`), `riscv64`, kernel 6.12.23, VLEN=128.
- ISA: `rv64imafdcv … zfh zfhmin zvfh zvfhmin zve32x …` (zfh native — deployment-
  faithful for ggml's scalar fp16→fp32 `fcvt.s.h`).
- Compile: `clang++-18` (Ubuntu clang 18.1.3) `-O3 -march=rv64gcv_zfh_zvfh
  -ffp-contract=fast`, `taskset -c 3`.
- Reps: best-of-**80** (rv64gcv) / best-of-**60** (zve32x proxy), inner iters
  1200–1500, gate 50–60 trials × 9 n. GENTLE: low load throughout (loadavg
  0.18→0.6, peak from my own ssh/compile; board never stalled, all runs returned
  promptly). Fresh remote dir `/home/ubuntu/tcrv-q5-ablation/`; the forbidden
  `~/llama_integ*` / `~/workspace/workspace3/llama.cpp` / `*_repackON/_canary/
  _override` dirs were NEVER touched.

## OFF baseline (what the pass emits with NO tuning record)

Confirmed locally: `tcrv-opt q5_{0,1}_input.mlir --tcrv-rvv-materialize-q5-{0,1}-
schedule=march=rv64gcv` (no dump, no record) stamps **m1/4/elided** for BOTH q5_0
and q5_1 (`selection_reason = "min-cost legal Q5_x shape …"`, cost=1005 = the static
cost-model argmin). So **ON÷OFF == measured÷static**: OFF = static-argmin
m1/4/elided; ON = the measured-fastest legal shape. The OFF baseline (m1/4/elided)
is byte-EXACT in every rv64gcv run (it survives the gate; the only q5_1 drift drops
are factor-4-**robust**, which the static model never selects) — so the denominator
is valid.

## Results — the ablation table (absolute ns, ggml-free)

| kernel | march | n | OFF (static m1/4/elided) | ON (measured) | **ON÷OFF** | factor pick | ggml-ctx (ON) | drift-dropped |
|--------|-------|---|--------------------------|----------------|------------|-------------|----------------|---------------|
| **q5_0** | rv64gcv | 4096 | m1/4/elided 3687.5 ns | **m1/1/robust** 2401.8 ns | **1.54×** | **factor=1** | 0.747× | none (9/9 exact) |
| **q5_0** | rv64gcv | 11008 | m1/4/elided 9118.4 ns | **m1/1/robust** 5927.9 ns | **1.54×** | **factor=1** | 0.749× | none (9/9 exact) |
| **q5_1** | rv64gcv | 4096 | m1/4/elided 3937.9 ns | **m1/1/robust** 2428.1 ns | **1.62×** | **factor=1** | 0.719× | m1/4/robust, mf4/4/robust |
| **q5_1** | rv64gcv | 11008 | m1/4/elided 10754.7 ns | **m1/1/robust** 6472.9 ns | **1.66×** | **factor=1** | 0.768× | m1/4/robust, mf4/4/robust |
| q5_0 | rv64gc_zve32x (PROXY) | 4096 | m1/1/robust 3249.4 ns† | m1/1/robust 3249.4 ns | 1.00× | factor=1 | 0.752× | none (6/6 exact) |
| q5_1 | rv64gc_zve32x (PROXY) | 4096 | m1/1/robust 3980.7 ns† | m1/1/robust 3980.7 ns | 1.00× | factor=1 | 0.718× | m1/4/robust, mf4/4/robust |

† zve32x has NO elided candidates (strip elision is capability-pruned with Zvl128b
OFF), so the static argmin among the 6 robust-only shapes already equals the
measured winner → ON÷OFF is trivially 1.00×, no measured>static gain at this tier.
**zve32x ns is a VLEN=128 PROXY** (no sub-128 silicon) — the legal SET is real, the
relative ns is timed on the VLEN=128 board. The headline lives on rv64gcv.

q5_0 @ n=4096 was run TWICE (run1 = `runs/q5_0_n4096_run1_record.txt`, run2 =
`runs_rerun/q5_0_n4096_run2_record.txt`): identical ladder ORDER, winner ns within
~4% (2506.2 / 2401.8) — the picks are stable, the 1.5× gap is unambiguous (not
noise). Table cites run2.

## The headline — a DOUBLE inversion that BROADENS the q4 centerpiece

The full measured ladder (e.g. `runs_rerun/q5_0_n4096_run2_record.txt:8-16`) is
cleanly bimodal and **OVERTURNS the static cost model on BOTH knobs**:

1. **elision inverts (NEW axis).** The whole q4 family prefers **elided**; q5
   prefers **robust** — every robust shape (m1/1/2/4) beats every elided shape by
   ~1.5×. The static model assigns elided the LOWEST cost (1005) and so picks
   m1/4/elided — which MEASURES SLOWEST among the legal m1 shapes. This is the
   measured>static thesis landing harder than q4_1 (1.5–1.7× vs q4_1's 1.20×).
2. **factor inverts (the inc10 axis).** Within robust, **factor=1 wins** (m1/1 <
   m1/2 < m1/4), matching q4_1's factor=1 preference — and opposite q4_0's factor=4.

So the inc10 "unroll inversion" centerpiece (q4_0 wants factor=4, q4_1 wants
factor=1) generalizes from **one knob to two**: across q4_0/q4_1/q5_0/q5_1 **no
single static shape — not factor, not elision — serves all four**. That is exactly
why per-kernel MEASURED selection (not a global constant or a cost-model argmin) is
necessary. Generality column: extended.

### Why fewer-cost-but-slower (mechanism, local, no board)

The static cost model picks **m1/4/elided** because it scores it the cheapest
(cost=1005) — but the board measures it ~1.5× SLOWER than m1/1/robust. To check
whether line-count explains this we emitted all four q5_0 corner shapes locally and
isolated each knob (one varies at a time):

| shape | emitted lines | `vsetvl` |
|-------|---------------|----------|
| m1/1/elided | 116 | 4 |
| m1/1/robust | 122 | 6 |
| m1/4/elided | 516 | 14 |
| m1/4/robust | 540 | 22 |

Two honest findings: (a) the line count is driven by **factor** (the 4× unroll:
116/122 → 516/540), NOT by elision — at fixed factor, elided is actually slightly
*fewer* lines than robust. So a "longer elided chain" story would be WRONG; we do
not have a line-count mechanism for the elision penalty. (b) The pure
cost-vs-latency point still holds: the cost model's lowest-cost shape (m1/4/elided)
measures slowest among the legal m1 shapes — it counts instructions, the board
measures latency/ILP, and they disagree. We did NOT further isolate WHY robust beats
elided by ~1.5× on q5 (the strip-elided fixed-VL path vs the robust setvl-loop body
differ in how the board schedules the qh `vlm`/`vmnand`/`vsub`-masked reconstruction,
but that is a hypothesis, not measured here). The load-bearing claim — measured
overturns static on the elision knob — stands on the ON÷OFF timing, not on this
line-count probe.

## Honesty notes (no inflation)

- **Best q5 pick is ~0.72–0.77× vs ggml** (a real LOSS vs ggml's hand-written
  kernel). Reported plainly as oracle context (I9), NOT a baseline. This matches the
  project's honest "RVV 通但慢" status. The **ablation is ON÷OFF same-compiler** and
  is wholly unaffected by the ggml gap — the pass IS the source of the 1.5–1.7×, ggml
  is not in that comparison.
- **q5_1 drift drops (2/9):** m1/4/robust and mf4/4/robust are DRIFT-ineligible —
  the factor-4-robust two-term fold (`+ xm·ys`, q5_1 shares q4_1's scale+min fold)
  contracts differently than ggml under `-ffp-contract=fast`, the SAME inc10 q4_1
  behavior. The gate drops them BEFORE ranking and ranks among the 7 survivors; the
  winner (m1/1/robust) and the OFF baseline (m1/4/elided) are both byte-exact. q5_0
  (single-term fold) has 0 drops (9/9 exact). Stated, not hidden.
- **zve32x = proxy, 1.00× ablation** — no measured>static gain at that tier (elided
  pruned away); included only for set-completeness, clearly labelled.

## Files

- `tune_block_dot_q5.py` — the driver. `q5_0_base.mlir` / `q5_1_base.mlir` — op
  templates. `kernels/kern_ggml_q5_{0,1}.cpp` — the ggml-real correctness oracles.
- `runs/` + `runs_rerun/` — per-(kernel,march,n) `tuning record` + the raw board
  stdout (`*_board_stdout.txt`, with the `GATE`/`gate_done`/`RESULT` lines) +
  `harness.cpp` / `candidates.cpp` / `ggml_ref.cpp` per run.

## Reproduce
```
python3 tune_block_dot_q5.py --kernel q5_0 --kernel q5_1 --march rv64gcv \
  --tcrv-opt build/bin/tcrv-opt --mlir-translate /usr/bin/mlir-translate-20 \
  --out runs --record runs/tuning_record_q5.txt --n 4096 \
  --reps 80 --host rvv --cpu 3 --remote-root /home/ubuntu/tcrv-q5-ablation
```
