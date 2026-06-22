# N1 silicon probes — K1 block-dot divergence + Fedora RVV0.7 (2026-06-22)

Workflow `wqnlpikb1` (2 probes + adversarial verify + synthesis). **Lead correction inside:
the Fedora adversarial verifier made a target-confusion error — do NOT trust its refutation.**

## The three real RISC-V vector profiles (now established)
| host | chip | RVV gen | VLEN | toolchain | role |
|---|---|---|---|---|---|
| `ssh rvv` | Sophgo **SG2044** | **1.0** | 128 | clang18 | main board (N3 Win-A/B sealed) |
| `ssh k1` | Spacemit X60 | **1.0** | **256** | gcc13 + clang18 | N1 VLEN-divergence profile |
| `ssh fedora-rvv07` | Sophgo **SG2042**/Mango, T-Head **C920** | **0.7.1** | 128(arch) | gcc13(no rvv hdr)/clang16 | RVV0.7 ISA-gen target (expensive) |

## K1 — REAL N1 selection divergence (verdict: holds, severity caveat)
Same legal block-dot candidate set, on-board measured>static tuner re-ranks by VLEN. **Honest
count = 1 decisive + 1 marginal + 1 null** (the probe JSON overcounted q4_0 as co-equal):
- **q8_0 — DECISIVE (load-bearing).** rvv VLEN=128 picks **m2** (m2/1/elided 851.3ns; whole m1
  block ≥1274); K1 VLEN=256 picks the whole **m1** block (7138–7301) over m2 (best 7629.9),
  ~6.4%. **Categorical LMUL-family reversal, not a near-tie.** Mechanism: ~256-bit *effective
  register-group width* is the sweet spot — reached by m2@VLEN128 and m1@VLEN256 → winning LMUL
  tracks constant effective width as VLEN doubles. This is a clean N1 capability(VLEN)→selection
  divergence on the q8_0 hot-path kernel, measured on two real chips.
- **q4_0 — marginal** (K1 m1/2/elided vs rvv m1/4/elided, 0.84% near-tie; reproduced, not noise,
  but NOT co-equal). q4_1 — null (m1/1/elided both).
- **Caveat CLOSED (2026-06-22 same-session paired re-measure)**: ran q8_0 tuner on rvv(SG2044,
  live vlenb=128) + K1(live vlenb=256) back-to-back, same harness, clang18, identical n=4096 /
  iters=2000 / reps=200 / taskset / -ffp-contract=fast, identical 12-candidate enumeration (12/12
  byte-exact both). **Reversal directly observed**: rvv winner **m2/4/elided** (5818ns; m2 beats
  best-m1 +7.01%) vs K1 winner **m1/2/robust** (7138ns; m1 beats best-m2 +6.88%). ~7% both ways,
  far outside noise; only the VLEN token (rv64gcv vs rv64gcv_zvl256b) differed. The fresh rvv
  winner is a different m2 *sub-shape* than the archived m2/1/elided — same **m2 family**, so the
  family-reversal claim is robust; exact sub-shape varies across board states. Log:
  `k1-vlen256/q8_0-paired-rvv128-k1256.log`.
- **N3 corollary (strong)**: the STATIC cost model picks `m2/2/elided` for BOTH chips — family-correct
  on rvv (within 0.1% of measured) but **wrong-family on K1 (6.5% slower than the measured m1 winner)**.
  Static can't see the VLEN-driven family flip; measurement fixes it. So this single result is both
  N1 (capability→selection divergence) AND N3 (measured>static where static is blind to VLEN).

## Fedora — genuine RVV0.7 (probe holds; verifier REFUTATION IS INVALID)
- **Probe (ran on `fedora-rvv07`)**: `clang -march=rv64gcv` RVV1.0 binary **SIGILLs** on the
  hardware (objdump-confirmed real vle32/vadd.vv/vse32 in main → trap is from RVV, not scalar);
  /proc/cpuinfo mvendorid **0x5b7 (T-Head/XuanTie)**, dmesg **"Sophgo Mango" = SG2042**, C920
  cores = RVV **0.7.1**. `rv64gcv0p7` unsupported by clang16 (LLVM dropped draft-0.7 V); the
  T-Head/XuanTie toolchain didn't resolve via the proxy. **Classification: RVV0.7 silicon —
  NOT a cheap 3rd RVV1.0 profile; expensive (needs RVV0.7 codegen + T-Head toolchain).**
- **Verifier ERROR (caught by lead)**: the adversarial verifier "refuted" this by re-running on
  **`ssh rvv`** — a DIFFERENT machine (SG2044/Ubuntu24.04/RVV1.0/clang18), where RVV1.0 of course
  RUNS (vlenb=16). It confused "the actual hardware" with the canonical `ssh rvv` board. Its
  refutation describes SG2044, not `fedora-rvv07`, so it does **not** rebut the probe. The probe's
  on-`fedora-rvv07` SIGILL + T-Head vendor id + SG2042 dmesg stand. (Useful side-fact it surfaced:
  the main `ssh rvv` board is **SG2044**, not SG2042.)
- **Caveat**: Fedora VLEN=128 is architectural/inferred (the only program reaching `vlenb` SIGILLs);
  a measured VLEN needs an RVV0.7 toolchain run.

## Next (N1)
1. **Solidify q8_0 divergence**: same-session paired rvv(128)+K1(256) re-measure → directly-observed
   reversal (removes the archived-baseline caveat). Highest-value N1 strengthening, no new hardware.
2. **Fedora RVV0.7**: real but expensive — needs the XuanTie/T-Head toolchain (rv64gcv0p7 codegen).
   Defer until toolchain sorted; it's an ISA-generation divergence (legit RISC-V, Case A), a deeper
   N1/N2 target than the VLEN profiles.
