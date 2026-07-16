# oracle-repack — ZERO-MODEL byte-exact repack oracles (12 formats + iq2_grids.h + iq1s_grid.h). Source campaign: .trellis/tasks/increment1-q6k-repack (G3 线B dequant/repack construction). Extracted G8 §一 cleanup.

Each oracle compares a REFERENCE that decodes the ORIGINAL (pre-repack) ggml block
against an EMITTER-MODEL that reads the REPACKED strip, on a pure-integer certificate.
Both are x86-native (`g++ -O2 -std=c++17`); there is no `qemu-riscv64` on this host, so
an oracle gates the repack layout + arithmetic MODEL, not the emitted RVV C itself.

`oracle_repack_iq1_s.cpp` (C4a-2) additionally:
* checks TWO integer certificates (`sumi` and `sumi1`), since iq1_s has two accumulators
  and gating only the first would leave the whole delta mechanism unverified;
* MEASURES and prints corpus completeness rather than claiming it (all 2048 11-bit grid
  indices, both qh bit15 delta polarities, all 8 ls steps) and fails if any axis is short;
* fails if any negative control comes back NOT EXERCISED (a dead control is a hole);
* pins its modelled byte layout against the offsets the front door ships.
