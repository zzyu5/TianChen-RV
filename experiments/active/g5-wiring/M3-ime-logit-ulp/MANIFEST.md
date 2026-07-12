# MANIFEST — G5-M3 IME triple logit-level bounded-ULP / near-tie 量化

Durable artifacts (one file per bullet). Board-side big logit .bin dumps are gitignored; md5 recorded.

## casefile
- `evidence.md`
- `MANIFEST.md`
- `.gitignore`
- `raw/report_q4_0.txt`
- `raw/report_q8_0.txt`
- `raw/report_q4_K.txt`
- `raw/q4_0-seqs.txt`
- `raw/q8_0-seqs.txt`
- `raw/q4_K-seqs.txt`

## board harness (durable · reversible)
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/logit_dump.cpp`
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/compare_logits.py`
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/run_board.sh`
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/run-logit-ulp-bg.sh`
- `tools/e2e-harness/board/g5-m3-ime-logit-ulp/run-logit-ulp.sh`

## reused forward-route env-gate patches (bridge · not re-authored here)
- `tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py`
- `tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py`
- `tools/e2e-harness/board/g5-m3-ime-q4_K/forward-route-patch-q4k.py`

## board binaries (gitignored · md5 for provenance · restored == baseline all 3 formats)
- baseline `ggml/src/ggml-cpu/spacemit/ime.cpp` md5 `40962c7e7c732bf472ae88cef89ced8d`
- baseline shipped `libggml-cpu.so.0.15.1` md5 `71cc4d295dac29382a0a7d4d5bd0c425`
- model `/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (board-resident · reused)
- model `/data/tinyllama-q8_0.gguf` sha256 `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59` (board-resident · reused)
- model `/data/tinyllama-1.1b-Q4_K_M.gguf` (board-resident · reused)

## board scratch (transient · not committed · cleaned after harvest)
- `/tmp/g5m3ulp/<fmt>/logits_{off,ven,on}_<i>.bin` (5MB each · teacher-forced per-position logits · gitignored/removed)
- `/tmp/g5m3ulp/<fmt>/vocab.tsv` (token-id→piece · gitignored/removed)
