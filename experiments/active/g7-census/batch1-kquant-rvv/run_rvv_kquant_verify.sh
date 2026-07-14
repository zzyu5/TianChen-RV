#!/usr/bin/env bash
# run_rvv_kquant_verify.sh — G7 L1 Batch-1 byte-exact correctness gate (q2/q3/q4/q6 @rvv).
# Fresh silicon byte-exact of the CENSUS-timed regenerated kernels vs STOCK ggml block-dot
# (independent reference decoded from the ORIGINAL block; INT byte-exact + bounded-NORM).
# q4 = GEMM-only verifier; q2/q3/q6 = GEMM+GEVM verifiers (need the gemv kernel too).
# gcc-15.2 -O2 (symmetric); link board libggml-cpu.so. Main tree + build/ UNTOUCHED; no git.
# q5_K byte-exact cited from existing seals (34fded0d GEVM byte-exact / k1 t4a byte-id /
# rvv construction oracle 8e-07 / qh-recon HARD GATE) — same fixture+tool+flags as q2/q3/q4/q6.
set -uo pipefail
CORE="${CORE:-8}"; MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML_BIN="/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin"
RDIR="/tmp/g7_census_kquant_verify"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[verify] core=$CORE march=$MARCH"
ssh rvv "mkdir -p $RDIR"
for f in q2 q3 q4 q6; do scp -q "$HERE/verify/verify_${f}K.c" rvv:$RDIR/; done
for f in q2 q3 q4 q5 q6; do scp -q "$HERE/kernels/repack_gemm_${f}_K_q8_K.kernel.c" rvv:$RDIR/; done
for f in q2 q3 q6; do scp -q "$HERE/kernels/repack_gemv_${f}_K_q8_K.kernel.c" rvv:$RDIR/; done

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  GXX=\$(command -v g++); echo '[cc] '\$(\$GXX --version|head -1)
  echo '# ggml_so_md5='\$(md5sum $GGML_BIN/libggml-cpu.so|cut -d' ' -f1)' loadavg='\$(cat /proc/loadavg)
  for f in q2 q3 q4 q5 q6; do md5sum repack_gemm_\${f}_K_q8_K.kernel.c; done

  compile_kern(){ \$GXX -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ \$1 -c -o \$2 2>k.err || { echo KERN_FAIL \$1; sed -n '1,6p' k.err; return 9; }; }

  for f in q2 q3 q4 q6; do
    echo \"=== [verify \${f}_K] ===\"
    compile_kern repack_gemm_\${f}_K_q8_K.kernel.c gm_\${f}.o || exit 3
    KOBJS=gm_\${f}.o
    if [ \$f != q4 ]; then compile_kern repack_gemv_\${f}_K_q8_K.kernel.c gv_\${f}.o || exit 3; KOBJS=\"\$KOBJS gv_\${f}.o\"; fi
    \$GXX -O2 -march=$MARCH -mabi=lp64d -x c++ verify_\${f}K.c -c -o v_\${f}.o 2>v.err || { echo VERIFY_CC_FAIL \$f; sed -n '1,8p' v.err; exit 4; }
    \$GXX v_\${f}.o \$KOBJS -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vbin_\${f} 2>vld.err || { echo VERIFY_LINK_FAIL \$f; sed -n '1,10p' vld.err; exit 5; }
    LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./vbin_\${f} 20260714 2>&1 | grep -E 'int.mismatch|int_mismatch|BYTE-EXACT|INT-BUG|VERDICT|SUMMARY|MISMATCH|worst' | head -40
  done
  echo '# loadavg_end='\$(cat /proc/loadavg)
  echo '=== VERIFY DONE ==='
"
