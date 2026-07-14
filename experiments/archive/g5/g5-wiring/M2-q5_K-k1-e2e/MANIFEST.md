# cell MANIFEST — g5-wiring / M2-q5_K-k1-e2e

- `evidence.md`
- `raw/build_seal.log`
- `raw/objdump_gemm_q5_K.txt`
- `raw/objdump_gevm_q5_K.txt`
- `raw/ppl_ON.err`
- `raw/ppl_OFF.err`
- `raw/correctness_ON_1.txt`
- `raw/correctness_OFF_1.txt`
- `raw/correctness_ON_2.txt`
- `raw/correctness_OFF_2.txt`
- `raw/correctness_ON_3.txt`
- `raw/correctness_OFF_3.txt`
- `raw/correctness_ON_4.txt`
- `raw/correctness_OFF_4.txt`
- `raw/phase_split.log`
- `raw/phase_analysis.txt`

## harness (reproduce entry)
- `tools/e2e-harness/board/g5-m2-q5_K-k1/deploy_patch_q5_K_emitted.py`
- `tools/e2e-harness/board/g5-m2-q5_K-k1/g5_m2_q5_K_build_seal.sh`
- `tools/e2e-harness/board/g5-m2-q5_K-k1/g5_m2_q5_K_correctness.sh`
- `tools/e2e-harness/board/g5-m2-q5_K-k1/g5_m2_q5_K_phase_split.sh`
- `tools/e2e-harness/board/g5-m2-q5_K-k1/analyze_phase_split.py`

## gitignored (board artifacts / large — md5 recorded in evidence.md)
- emitted kernels: gemm md5 ba30ba54 (host /tmp/kquant_export/gemm_q5_K_q8_K.kernel.c), gevm md5 c445b89e
- libggml-cpu.so.OFF md5 14b6add6 · libggml-cpu.so.ON md5 a408563e (board /tmp/g5_q5k, ephemeral)
- model tinyllama-1.1b-Q5_K_M.gguf sha256 6002d505 (board /data, 783 MB, k1-local requantized from q8_0)

## board scratch (ephemeral, untracked): k1:/tmp/g5_q5k/ ; dedicated build k1:/data/build-k1-q5k/
## main tree + /data/k1build-stock UNTOUCHED (restore verified byte-exact)
