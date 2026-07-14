# cell MANIFEST — g2-e2e-wholemodel

- **campaign**: G2 fusion (rms_norm→mul) — G3 四.1 whole-model e2e
- **status**: ACTIVE — NOT-SIGNIFICANT e2e (pre-registered branch; isolated 1.31× Amdahl-diluted below noise)
- **role**: board e2e phase-split A/B of the rms_norm→mul fusion on a real llama forward pass (rvv/VLEN128)

## headline
- Upstream ggml (f3e1828) **already fuses** RMS_NORM+MUL by default
  (`ggml_compute_forward_rms_norm_mul_fused`); the honest e2e A/B is its **ON/OFF env toggle**
  (`GGML_CPU_DISABLE_FUSION`) on ONE binary — toolchain-symmetric by construction — NOT
  fused-vs-unfused-stock (there is no unfused stock). Our emitted fused kernel is bit-exact to
  ggml's, so it is a drop-in equivalent and the toggle answers the e2e question for both.
- Phase-split (tinyllama-q4_0, N=10/arm): **prefill −0.11% (ns), decode −0.27% (ns)** — sub-noise.
- Numeric: fusion ON vs OFF greedy completion **byte-identical** (sha `c0c1274bd2aa8b01`).
- Amdahl (task-clock decode profile): fused-norm = **0.05%** of decode, q4_0 matmul = **90.2%** →
  fusion Amdahl ceiling +0.05% sits *below* the noise floor ⇒ e2e null is structural, not a miss.
- Verdict: NOT-SIGNIFICANT e2e; isolated per-pair silicon win still owned by the sibling cell.
  No Win-B/Win-C. Rollout of the fusion e2e is NOT on evidence (the axis is the q4_0 matmul).

## board
- `ssh rvv` openEuler VLEN128 / 64c / gov=performance @2.6 GHz / cores 8-15 (taskset) / 8 threads.
- toolchain: upstream llama.cpp/ggml `f3e1828`, gcc15.2 full-march (L1-SEAL override). preflight(0)
  symmetry TOTAL (same `llama-bench` + same `libggml-cpu.so`; only one `getenv` branch differs).
- main tree + build/ UNTOUCHED; nothing committed; no git stash. board scratch = `/tmp/g2e2e` only.

## harness (lives in tools/, not this data cell)
- `tools/e2e-harness/board/g2_fuse_e2e_ab.sh` — paired phase-split fusion ON/OFF A/B + preflight(0).
- `tools/e2e-harness/board/g2_fuse_e2e_numeric_amdahl.sh` — greedy determinism + task-clock Amdahl.
- `tools/e2e-harness/board/aggregate_fuse_e2e.py` — pools ###FAB samples → median/IQR/speedup/T-N verdict.

## durable files
- `board_measured.md` — primary writeup: pre-reg, reframe, setup, numeric, phase-split table, Amdahl, verdict.
- `NOTES.md` — methodology, the ggml-already-fuses discovery, protocol, mechanism, gotchas, restore.
- `board_raw_tinyllama.txt` — raw g2_fuse_e2e_ab.sh output (preflight + fingerprint + ###FAB json blocks).
- `numeric_amdahl_tinyllama.txt` — raw greedy ON/OFF determinism + perf phase (llama-completion).
- `amdahl_perf_tinyllama.txt` — task-clock decode profile (self% per symbol).
- `board_raw_llama7b.txt` — canonical 65-norm confirmation (llama-2-7b-chat.Q4_0), decode-focused ON/OFF.
