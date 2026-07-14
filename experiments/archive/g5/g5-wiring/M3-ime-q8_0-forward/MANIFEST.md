# MANIFEST — G5-M3 IME q8_0 forward-wiring (完成 ratified IME triple 全 forward-wired)

Durable artifacts (one file per bullet). Board-side big binaries are gitignored; md5/sha256 recorded.

## casefile
- `evidence.md`
- `MANIFEST.md`
- `.gitignore`
- `raw/q8-bridge-ut.txt`
- `raw/q8-forward-route-multi.txt`

## board harness (durable · reversible)
- `tools/e2e-harness/board/g5-m3-ime-q8_0/g5m3_q8_bridge_ut.c`
- `tools/e2e-harness/board/g5-m3-ime-q8_0/run-q8-bridge-ut.sh`
- `tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py`
- `tools/e2e-harness/board/g5-m3-ime-q8_0/run-forward-route-q8-multi.sh`

## board binaries (gitignored · md5/sha256 for provenance)
- baseline `ggml/src/ggml-cpu/spacemit/ime.cpp` md5 `40962c7e7c732bf472ae88cef89ced8d` (restored == baseline)
- baseline shipped `libggml-cpu.so.0.15.1` md5 `71cc4d295dac29382a0a7d4d5bd0c425` (restored == baseline)
- model `/data/tinyllama-q8_0.gguf` sha256 `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59` (board-resident q8_0 GGUF v3 · reused zero-provisioning · A/B same weights)
