# cell MANIFEST — p1-k1-vlen256-decode-roofline

- **campaign**: repack (P1 decision, roofline-resolved)
- **status**: ACTIVE (measured; 2026-07-07-lineA-batch1; HEAD anchor d1a26e4a)
- **role**: P1 决议 for the parked k1/VLEN256 q4_0 GEVM decode reversal. Fresh paired N=10 (llama-bench
  tg32 REPS=5 x PASSES=2, freq 1.6GHz 0.00% span) REPRODUCES the sealed 0.854x reversal, and the fresh k1
  roofline (read ceiling 4.260 GB/s, ridge 12 ops/byte => decode bandwidth-bound) resolves it: stock reaches
  ~91% of the DRAM read wall, ours only ~78% => **GAP (真差), confound EXCLUDED, parity-at-floor EXCLUDED**.
  The same run's prefill (GEMM pp128) = 1.0022x PARITY (compute-bound, ~parity per VLEN-flip, as expected).
  This is the VLEN-flip mirror of the sealed rvv/VLEN128 result (ours saturates ~100% -> decode WIN 1.91x).
- **lineage**: A=/data/k1build is the deployed constructed q4_0 REPACK GEVM from the sealed k1-flip cell (same
  binaries; this batch re-measured perf paired, did NOT re-export/redeploy from pinned HEAD). B=/data/k1build-stock.

## durable files

- `bandwidth_analysis.txt` — the P1 roofline gate: achievement math + three-tier judgment + caveats
- `evidence.json`         — aggregate_e2e.py paired medians/CI/IQR (prefill PARITY + decode DIFFERENCE)
- `aggregate.txt`         — human aggregate print (DVFS span 0.00%)
- `phase_split_raw.txt`   — verbatim board llama-bench json A/B interleaved (n=10, env fingerprint)

## note — 本批变更 (this-batch snapshot, k1 board session end)
- roofline k1: read_single 3.602 / read_agg4c 4.260 GB/s ; int8_peak 25.51 GMACs = 51.0 GOPS ; ridge 12.0 op/byte.
- P1 decode  : ours 5.67 / stock 6.64 t/s = 0.854x DIFFERENCE => roofline GAP (stock 91% wall / ours 78%).
- GEMM prefill: ours 24.69 / stock 24.63 t/s = 1.0022x PARITY (compute-bound; ~parity per VLEN-flip).
- board restored: /tmp run dir removed, no llama-bench left running, governor untouched (was already performance/locked).
