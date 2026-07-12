# MANIFEST — G5-M3 IME triple e2e ON/OFF paired perf (transmission-dilution finalized)

Durable artifacts (one file per bullet). Board-side bench `.err`/`route_build.log` scratch gitignored; md5 recorded.

## casefile
- `evidence.md`
- `MANIFEST.md`
- `.gitignore`
- `raw/run_q4_0.log`
- `raw/run_q8_0.log`
- `raw/run_q4_K.log`
- `raw/agg_q4_0.txt`
- `raw/agg_q8_0.txt`
- `raw/agg_q4_K.txt`

## board harness (durable · reversible · nohup+poll)
- `tools/e2e-harness/board/g5-m3-ime-e2e-pairing/run_pair_board.sh`
- `tools/e2e-harness/board/g5-m3-ime-e2e-pairing/run-pair-bg.sh`
- `tools/e2e-harness/board/g5-m3-ime-e2e-pairing/agg_pair.py`

## reused forward-route env-gate patches (bridge · not re-authored here)
- `tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py`
- `tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py`
- `tools/e2e-harness/board/g5-m3-ime-q4_K/forward-route-patch-q4k.py`

## compute-account (amdahl anchor · not re-measured here · referenced)
- `experiments/active/g5-wiring/M0-接线机制解剖.md` (tcrv IME ~2.09x compute-account · vendor toggle prefill 1.49-1.61x / decode floor ~1.47x)
- `experiments/active/g5-wiring/M3-ime-bridge-recon/deploy-readiness.md` (sealed `[GAP-IME-LEAF-PIPELINE]` register-resident batched vmadot leaf = 2.09x enabler)

## board binaries (gitignored · md5 for provenance · restored == baseline all 3 formats)
- baseline `ggml/src/ggml-cpu/spacemit/ime.cpp` md5 `40962c7e7c732bf472ae88cef89ced8d`
- baseline shipped `libggml-cpu.so.0.15.1` md5 `71cc4d295dac29382a0a7d4d5bd0c425`
- model `/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf` (board-resident · reused)
- model `/data/tinyllama-q8_0.gguf` (board-resident · reused)
- model `/data/tinyllama-1.1b-Q4_K_M.gguf` (board-resident · reused)

## board scratch (transient · not committed · cleaned after harvest)
- `/tmp/g5m3pair/<fmt>/bench_{on,ven,off}_p{1,2}.err` (llama-bench stderr + banner · gitignored/removed)
- `/tmp/g5m3pair/<fmt>/route_build.log` (patch rebuild log · gitignored/removed)
