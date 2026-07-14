# cell MANIFEST — g5-wiring / M2-q2_K-k1-e2e

- `evidence.md`
- `raw/build_seal.log`
- `raw/objdump_OUR_gemm_q2_K.txt`
- `raw/objdump_OUR_gevm_q2_K.txt`
- `raw/objdump_STOCK_gemm_q2_K.txt`
- `raw/objdump_STOCK_gevm_q2_K.txt`
- `raw/silicon_verify_q2K.txt`
- `raw/ppl_ON.log`
- `raw/ppl_OFF.log`
- `raw/phase_split.log`
- `raw/phase_analysis.txt`

## harness (reproduce entry)
- `tools/e2e-harness/board/g5-q2_K-k1/deploy_patch_q2_K_emitted.py`
- `tools/e2e-harness/board/g5-q2_K-k1/g5_q2_K_build_seal.sh`
- `tools/e2e-harness/board/g5-q2_K-k1/g5_q2_K_correctness.sh`
- `tools/e2e-harness/board/g5-q2_K-k1/g5_q2_K_phase_split.sh`
- `tools/e2e-harness/board/g5-q2_K-k1/analyze_phase_split.py`
- `tools/e2e-harness/board/kquant_repack_verify_q2K.c` (silicon oracle, symbol tcrv→weft at run)

## gitignored (board artifacts / large — md5/sha recorded in evidence.md)
- emitted kernels: gemm md5 d9dee831 / gevm md5 ee76a4d3 (host+board /tmp/g5_q2k/weft_emitted_*.inc)
- libggml-cpu.so.OFF md5 14b6add6 · libggml-cpu.so.ON md5 5b036155 (board /tmp/g5_q2k, ephemeral)
- model tinyllama-1.1b-Q2_K_M.gguf sha256 7ef48ade (board /data, 411 MiB, k1-local requantized from q8_0)

## board scratch (ephemeral, untracked): k1:/tmp/g5_q2k/ ; dedicated build k1:/data/build-k1-q2k/
## main tree + /data/k1build-stock + /data/build-k1-q5k UNTOUCHED (restore verified byte-exact)

## ★ opponent identity (裁二.3 X-0 纠偏)
q2_K@k1 对手 = STOCK 手调 RVV q2_K 16x1 repack (case256 fires), NOT block-dot (与 q5_K 相反).
= q4_K Win-K1-VLEN 同型对局 (our-emit vs 真出货 hand-brick).

## ★ VERDICT
q2_K@k1 e2e = **黄-对手更强 (维持)**: our-emit vs stock hand-brick prefill **0.873×** / decode **0.875×** (both <parity LOSS·clang-18 对称·correctness GREEN@bounded-ULP). 我方全展开 27KB emitted 核输给 stock 紧凑 rolled loop. perf-covered 维持 **7/83** (未 flip 绿). q4_K Win-K1-VLEN 的 CONTRAST (同对手·q4_K 胜 1.085× / q2_K 负 0.873×).
