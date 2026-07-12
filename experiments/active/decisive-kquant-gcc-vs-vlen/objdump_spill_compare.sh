#!/usr/bin/env bash
# [decisive-kquant-gcc-vs-vlen] Standalone objdump spill/vsetvli comparison of the
# EXACT M2-q4_K emitted kernels (GEMM md5 6cbd9c19, GEVM md5 e909a9bd) under
# gcc-15.2.0 vs clang-17.0.6 vs clang-18.1.8. -c only (no link) => clang-18 OK.
# Closes provenance gap: prior 742-spill was on sibling 90d454da, never on 6cbd9c19.
set -u
D=/tmp/dkgv
GCC=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
CL17=/usr/bin/clang++-17
CL18=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
MARCH_GCC="-march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause -mabi=lp64d"
MARCH_CL="-march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbop_zihintpause -mabi=lp64d"
OBJD=/opt/tcrv-toolchains/binutils-2.46.1/bin/objdump
[ -x "$OBJD" ] || OBJD=objdump

count() {  # $1=objfile $2=symbol
  local f=$1 s=$2
  $OBJD -d --disassemble="$s" "$f" 2>/dev/null > "$f.$s.dis" || $OBJD -d "$f" > "$f.$s.dis" 2>/dev/null
  local insns vset spill_whole spill_scalar
  insns=$(grep -cE '^\s+[0-9a-f]+:' "$f.$s.dis")
  vset=$(grep -cE 'vset[i]*vli' "$f.$s.dis")
  # whole-register vector spills/reloads (vs1r/vs2r/vs4r/vs8r store, vl1re/vl2re/vl4re/vl8re load)
  spill_whole=$(grep -cE 'vs[1248]r\.v|vl[1248]re(8|16|32|64)\.v' "$f.$s.dis")
  echo "  insns=$insns vsetvli=$vset vec_whole_reg_spill=$spill_whole"
}

for label in gcc15 clang17 clang18; do
  case $label in
    gcc15)   CXX=$GCC;  MARCH="$MARCH_GCC";  OPT="-O3" ;;
    clang17) CXX=$CL17; MARCH="$MARCH_CL";   OPT="-O3" ;;
    clang18) CXX=$CL18; MARCH="$MARCH_CL";   OPT="-O3" ;;
  esac
  echo "=========================================================="
  echo "[$label] $CXX $OPT $MARCH"
  for kind in gemm gevm; do
    src=$D/tcrv_emitted_${kind}_q4_K.inc
    obj=$D/${kind}_${label}.o
    cp "$src" "$D/${kind}_${label}.cpp"
    if $CXX $MARCH $OPT -c "$D/${kind}_${label}.cpp" -o "$obj" 2>"$D/${kind}_${label}.clog"; then
      sym=tcrv_emitc_ggml_repack_${kind}_q4_K_q8_K_kernel_ggml_repack_${kind}_q4_K_q8_K
      objsz=$(stat -c%s "$obj")
      echo " [$kind] objsz=${objsz}B md5(o)=$(md5sum "$obj"|cut -c1-12)"
      count "$obj" "$sym"
    else
      echo " [$kind] COMPILE FAIL:"; head -5 "$D/${kind}_${label}.clog"
    fi
  done
done
echo "=========================================================="
echo "DONE"
