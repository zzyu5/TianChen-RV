# cell MANIFEST — roofline

- **campaign**: roofline (physical-reference calibration)
- **status**: ACTIVE (measured dual-board; 2026-07-07-lineA-batch1)
- **role**: the two PHYSICAL ceilings every decode/prefill parity judgment is measured against
  (增补一①): sustained READ bandwidth (membw_probe.c, single + 4-core aggregate on the e2e core set)
  and peak int8 vector compute (int8_peak_probe.c, register-resident vwmacc GMACs/GOPS). Dual-board:
  rvv (VLEN128, 2.6GHz) read_agg4c=5.24 GB/s / int8_peak=83.1 GOPS ; k1 (VLEN256, 1.6GHz)
  read_agg4c=4.26 GB/s / int8_peak=51.0 GOPS. Ridge ~16 (rvv) / ~12 (k1) ops/byte => q4_0 GEVM decode
  (~1-2 op/byte) is BANDWIDTH-BOUND on both boards. rvv ceiling cross-checks the sealed gemm cell's 5.23 GB/s.
- **layout**: org STAGE1. Probes live under tools/e2e-harness/board/ (membw_probe.c, int8_peak_probe.c,
  roofline_probe.sh); this cell holds the measured ceilings + raw board output.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `roofline.csv`      — the two ceilings per board + ridge + provenance + self-check caveat
- `roofline_raw.txt`  — verbatim board probe output (rvv + k1) with per-cycle-rate confirmation

## note — self-check caveat
The int8_peak objdump self-check reported `vwmacc_in_binary=0` on BOTH boards: their stock binutils objdump
does not decode the vwmacc mnemonic (a disassembly artifact, not loop-folding). The measured issue rate is the
exact single-issue physical peak (rvv 1.00 vwmacc/cycle @2.6GHz; k1 0.50 vwmacc/cycle @1.6GHz for the VLEN256
m1->m2 widening), only reachable if the loop truly ran vector MACs — so the ceiling is physically confirmed.

> Board sessions ended with the /tmp probe dirs removed; no persistent board state was modified (probes ran in /tmp).
