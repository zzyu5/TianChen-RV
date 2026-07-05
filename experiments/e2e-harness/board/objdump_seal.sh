#!/usr/bin/env bash
# objdump_seal.sh -- MECHANISM seal + symmetry check, run ON the board.
#
# (1) MECHANISM: disassemble A's compiler-emitted repack GEMM/GEMV (tiled/packed
#     weight, contiguous vector streaming = memory locality) vs B's ggml
#     block-dot vec_dot_q4_0_q8_0 (per-block dequant+dot). Emits a per-symbol
#     RVV instruction histogram as objective evidence of the two mechanisms.
# (2) SYMMETRY: scan BOTH trees' ggml-cpu objects for the fp16 softfloat libcall
#     confound (__extendhfsf2-class). Must be ABSENT on both (third pass at the
#     P2c anti-confound, now under full-capability march).
#
# Env: A_BUILD B_BUILD | OUTDIR(/tmp)
set -u
A_BUILD="${A_BUILD:?}"; B_BUILD="${B_BUILD:?}"; OUTDIR="${OUTDIR:-/tmp/tcrv_l1seal}"
OD=$(command -v objdump || command -v llvm-objdump)
NM=$(command -v nm || echo "")
mkdir -p "$OUTDIR"

find_cpuobj() {  # prefer the standalone repack/quants .o (has our symbols), else the .so
  local d="$1"
  find "$d" -name 'repack.cpp.o' -path '*ggml-cpu*arch/riscv*' 2>/dev/null | head -1
}
find_so() { find "$1" -name 'libggml-cpu.so' 2>/dev/null | head -1; }

LIBCALL_RE='__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee|__extendhfxf2'

hist() {  # $1=disasm-file -> RVV instruction histogram (loads/stores/macc/vset)
  echo "   vsetvli/vsetivli : $(grep -cE '\bvset[i]?vli\b' "$1")"
  echo "   unit-stride load vle*  : $(grep -cE '\bvle[0-9]+\.v\b' "$1")"
  echo "   strided load  vlse*    : $(grep -cE '\bvlse[0-9]+\.v\b' "$1")"
  echo "   indexed load  vlux/vlox: $(grep -cE '\bvl[ou]xei[0-9]+\.v\b' "$1")"
  echo "   whole-reg load vl*re*  : $(grep -cE '\bvl[0-9]+re[0-9]+\.v\b' "$1")"
  echo "   store   vse*/vsse*     : $(grep -cE '\bvs(e|se)[0-9]+\.v\b' "$1")"
  echo "   int mul-acc vmacc/vmadd: $(grep -cE '\bvn?m(acc|add)[a-z.]*\b' "$1")"
  echo "   widening vw* (mul/add) : $(grep -cE '\bvw[a-z0-9.]+\b' "$1")"
  echo "   fp mul-acc vf*macc     : $(grep -cE '\bvf[a-z]*macc[a-z.]*\b' "$1")"
  echo "   total insns            : $(grep -cE '^\s+[0-9a-f]+:' "$1")"
}

dis_sym() {  # $1=objfile $2=regex $3=out  -> disassemble matching symbols
  local obj="$1" re="$2" out="$3"
  $OD -d -C "$obj" 2>/dev/null | awk -v re="$re" '
    /^[0-9a-f]+ <.*>:/ { insym = ($0 ~ re) }
    insym { print }
  ' > "$out"
  wc -l < "$out"
}

echo "############ MECHANISM SEAL ############"
AOBJ=$(find_cpuobj "$A_BUILD"); ASO=$(find_so "$A_BUILD")
BOBJ=$(find_cpuobj "$B_BUILD"); BSO=$(find_so "$B_BUILD")
echo "A repack.o=$AOBJ"; echo "A so=$ASO"
echo "B repack.o=$BOBJ"; echo "B so=$BSO"

echo ""
echo "== A-side symbols matching tcrv/repack/gemm/gemv =="
[ -n "$NM" ] && $NM -C "$AOBJ" 2>/dev/null | grep -iE 'tcrv|repack.*(gemm|gemv)|gemv_q4_0|gemm_q4_0' | grep -iE ' [Tt] ' | head -20

echo ""
echo "== A: repack GEMM/GEMV disasm histogram (OUR mechanism: packed weight, streamed) =="
n=$(dis_sym "$AOBJ" 'tcrv.*(gemm|gemv)|repack.*(gemm|gemv)|gemm_q4_0|gemv_q4_0' "$OUTDIR/A_repack_disasm.txt")
echo "   (disasm lines captured: $n -> $OUTDIR/A_repack_disasm.txt)"
[ "$n" -gt 0 ] && hist "$OUTDIR/A_repack_disasm.txt"

echo ""
echo "== B: vec_dot_q4_0_q8_0 block-dot disasm histogram (STOCK mechanism: per-block dot) =="
n=$(dis_sym "$BOBJ" 'vec_dot_q4_0_q8_0|ggml_vec_dot_q4_0' "$OUTDIR/B_blockdot_disasm.txt")
echo "   (disasm lines captured: $n -> $OUTDIR/B_blockdot_disasm.txt)"
[ "$n" -gt 0 ] && hist "$OUTDIR/B_blockdot_disasm.txt"

echo ""
echo "== A also carries vec_dot_q4_0_q8_0? (block-dot present but bypassed at VLEN128) =="
dis_sym "$AOBJ" 'vec_dot_q4_0_q8_0|ggml_vec_dot_q4_0' "$OUTDIR/A_blockdot_disasm.txt" >/dev/null
[ -s "$OUTDIR/A_blockdot_disasm.txt" ] && hist "$OUTDIR/A_blockdot_disasm.txt" || echo "   (none)"

echo ""
echo "############ SYMMETRY: fp16 softfloat libcall scan (both trees) ############"
scan() {
  local d="$1" hit=0 n=0 o
  for o in $(find "$d" -name '*.o' -path '*ggml-cpu*' 2>/dev/null); do
    n=$((n+1))
    if $OD -d "$o" 2>/dev/null | grep -Eq "$LIBCALL_RE"; then echo "   LIBCALL in $(basename "$o")"; hit=1; fi
  done
  echo "scanned=$n libcall_hit=$hit"
}
echo "-- A tree --"; scan "$A_BUILD"
echo "-- B tree --"; scan "$B_BUILD"
echo "############ objdump_seal DONE ############"
