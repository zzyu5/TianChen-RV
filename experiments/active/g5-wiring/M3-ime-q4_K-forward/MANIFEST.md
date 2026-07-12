# MANIFEST — G5-M3 IME q4_K forward-wiring

Durable artifacts (one file per bullet). Board-side big binaries are gitignored; md5 recorded.

## casefile
- `evidence.md`
- `MANIFEST.md`
- `.gitignore`
- `raw/q4k-bridge-ut.txt`
- `raw/q4k-forward-route-multi.txt`

## board harness (durable · reversible)
- `tools/e2e-harness/board/g5-m3-ime-q4_K/g5m3_q4k_bridge_ut.c`
- `tools/e2e-harness/board/g5-m3-ime-q4_K/run-q4k-bridge-ut.sh`
- `tools/e2e-harness/board/g5-m3-ime-q4_K/forward-route-patch-q4k.py`
- `tools/e2e-harness/board/g5-m3-ime-q4_K/run-forward-route-q4k-multi.sh`

## board binaries (gitignored · md5 for provenance)
- baseline `ggml/src/ggml-cpu/spacemit/ime.cpp` md5 `40962c7e7c732bf472ae88cef89ced8d` (restored == baseline)
- baseline shipped `libggml-cpu.so.0.15.1` md5 `71cc4d295dac29382a0a7d4d5bd0c425` (restored == baseline)
- model `/data/tinyllama-1.1b-Q4_K_M.gguf` (board-resident · provisioned prior WORK-ITEM · reused zero-provisioning)
