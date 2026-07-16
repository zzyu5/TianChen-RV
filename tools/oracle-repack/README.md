# oracle-repack — ZERO-MODEL byte-exact repack oracles (13 formats + iq2_grids.h + iq1s_grid.h). Source campaign: .trellis/tasks/increment1-q6k-repack (G3 线B dequant/repack construction). Extracted G8 §一 cleanup.

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

`oracle_repack_iq1_m.cpp` (C4a-3) follows the iq1_s pattern and adds three things:
* a THIRD integer certificate, `scale_u16`, checked **exhaustively** rather than sampled.
  block_iq1_m has no inline `d`: its fp16 is four nibbles scattered across the scales
  words (`(sc[0]>>12) | ((sc[1]>>8)&0x00f0) | ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000)`),
  assembled once at repack time. That assembly's entire input domain is those four
  nibbles = 16^4 = 65536 combinations, so the oracle walks **all** of them (with the low
  12 bits of each word randomized, to prove the assembly ignores the ls fields it must
  not read). Reference uses ggml's fused expression; the model uses a per-nibble loop —
  two derivations, not one shared helper.
* PER-SLOT delta coverage. iq1_m's four deltas per sub-block are INDEPENDENT bits, so
  coverage is measured for each of the 4 group slots separately; a corpus that only ever
  flipped slot 0 would leave slots 1-3 unverified *and* would silently kill the control
  below. ls1 and ls2 are likewise swept and reported separately (dual ls).
* the `BSUMS16` negative control, which is the one that earns the
  `GridFoldArith::DeltaGridGroupSum` axis value. It makes the reference pretend the two
  8-groups inside one 16-group SHARE a delta — i.e. exactly what a per-16 block_q8_K
  bsums entry *could* express. It spikes 100%, which MEASURES the claim the whole leaf
  rests on: per-16 bsums genuinely cannot express iq1_m's delta term, so iq1_m cannot
  ride the iq1_s bsums leaf. Had it not spiked, the new axis value would have been
  unjustified and iq1_m should have been a parameter of the iq1_s leaf instead.
