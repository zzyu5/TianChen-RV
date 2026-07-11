#!/usr/bin/env bash
# T6 batch-k1 — whole-model phased e2e "IME reference" runner (k1 board).
#
# ============================== HONEST SCOPE ==============================
# This harness measures the VENDOR ggml-spacemit IME backend (build-ime =
# GGML_CPU_RISCV64_SPACEMIT=ON) vs the same-toolchain RVV fallback
# (build-off = OFF), on tinyllama-1B-Q4_0.
#
# It is NOT a measurement of the tcrv IME GEMM kernel. As of HEAD d078fba3
# the tcrv IME matmul_tile / vmadot_mac_kloop kernels are STANDALONE certified
# constructs (host oracle + k1 silicon int32 0-diff, ~2.09x compute-account,
# g4-m1b-reseal-batched) and are NOT wired into any llama.cpp forward mul_mat
# path.  [GAP-IME-E2E-INTEGRATION].
#
# This runner exists to answer the *best-case transduction* question: if a
# fully-wired IME GEMM existed, would the ~2x compute-account transduce to
# whole-model e2e?  The vendor backend (whose OWN ime1::gemm_kernel_i8i4 IS
# wired into forward) is the strongest available proxy for that ceiling.
# Filename distinct from the rvv batch (t6-rvv-*).
# =========================================================================
set -u
MODEL="${MODEL:-$HOME/tcrv-k1-llama/models/tinyllama-q4_0.gguf}"
IME_BENCH="$HOME/tcrv-k1-llama/build-ime/bin/llama-bench"   # SPACEMIT=ON, vmadot live (32)
OFF_BENCH="$HOME/tcrv-k1-llama/build-off/bin/llama-bench"   # SPACEMIT=OFF, RVV fallback (0 vmadot)
PP="${PP:-256,512}"     # prefill (M=seq) — IME matrix's regime
TG="${TG:-64}"          # decode (M=1) — GEVM, IME cannot help; kernel-family control
REPS="${REPS:-3}"
THREADS="${THREADS:-4}"

echo "### board self-check"
grep -o ime /proc/cpuinfo | head -1
echo "vmadot in ime-lib: $(objdump -d "$HOME/tcrv-k1-llama/build-ime/bin/libggml-cpu.so.0" 2>/dev/null | grep -c vmadot)"
echo "vmadot in off-lib: $(objdump -d "$HOME/tcrv-k1-llama/build-off/bin/libggml-cpu.so.0" 2>/dev/null | grep -c vmadot)"

echo "### BUILD-IME (vendor SPACEMIT IME ON)"
taskset -c 0-3 "$IME_BENCH" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -t "$THREADS" 2>/dev/null
echo "### BUILD-OFF (RVV fallback, no vmadot)"
taskset -c 0-3 "$OFF_BENCH" -m "$MODEL" -p "$PP" -n "$TG" -r "$REPS" -t "$THREADS" 2>/dev/null
