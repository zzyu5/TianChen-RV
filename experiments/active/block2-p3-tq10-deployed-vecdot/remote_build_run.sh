#!/usr/bin/env bash
# remote_build_run.sh -- runs ON the board (rvv or k1). B线 block2 P3: DEPLOYED-EMIT
# tq1_0 vec_dot direct M=1 board measure (P2 iq2 paradigm). Compiler-SYMMETRIC: the
# DEPLOYED-EMIT leaf (weft-opt --materialize-tq1-0-...-source-front-door --lower-to-emitc
# | mlir-translate --mlir-to-cpp; CORE==PROD diff=0) is compiled clang-18 -x c++, the ggml
# factory is ALSO clang-18. Two factory modes:
#   FACTORY_MODE=lib    : link the prebuilt clang-18 symmetric libggml-cpu.so ($GGML) --
#                         the EXACT deployed opponent (rvv: build-clang18-rv64gcv).
#   FACTORY_MODE=source : compile arch/riscv/quants.c + quants.c from $R with clang-18.
# Env in: CC, MARCH, RDIR, CORES, FACTORY_MODE, [GGML], [R], [ENVSH]
# ONLY difference vs the P1 standalone harness: the leaf is DEPLOYED EMIT (extern "C"
# C++), so it is compiled `-x c++` and the link pulls -lstdc++. Driver + oracle + gate +
# cold identical to P1 (same symbol weft_emitc_ggml_vec_dot_tq1_0_q8_K_kernel_rvv_...).
set -u
: "${CC:?}"; : "${MARCH:?}"; : "${RDIR:?}"; : "${CORES:?}"; : "${FACTORY_MODE:?}"
KERNEL_SRC="${KERNEL_SRC:-kernels/tq1_0_deployed_emit.cpp}"
[ -n "${ENVSH:-}" ] && { source "$ENVSH" 2>/dev/null && CC="$(command -v clang)"; }
cd "$RDIR"; SEAL="$RDIR/seal.txt"; : > "$SEAL"
say(){ echo "$@" | tee -a "$SEAL"; }
say "# BOARD=$(uname -srm) CC=$($CC --version|head -1) march=$MARCH factory_mode=$FACTORY_MODE kernel=$KERNEL_SRC (DEPLOYED-EMIT)"

# -x mode: .cpp deployed-emit -> c++ ; .c standalone -> c
case "$KERNEL_SRC" in *.cpp|*.cc|*.cxx) KX="c++"; STDCXX="-lstdc++";; *) KX="c"; STDCXX="";; esac

FACT_OBJ=""; FACT_LIBS=""
if [ "$FACTORY_MODE" = "source" ]; then
  : "${R:?}"; INC="-I$R/ggml/src -I$R/ggml/src/ggml-cpu -I$R/ggml/include"
  say "# ggml_root=$R commit=$(cd $R && git log -1 --format='%h %ci' 2>/dev/null || echo n/a)"
  say "=== [B] factory from source (clang-18 symmetric) ==="
  $CC -O3 -march=$MARCH -mabi=lp64d -fno-integrated-as -ffunction-sections -fdata-sections -fno-stack-protector $INC \
    -c $R/ggml/src/ggml-cpu/arch/riscv/quants.c -o fo_arch.o 2>a.err || { say "ARCH_FAIL"; sed -n 1,12p a.err|tee -a $SEAL; exit 1; }
  $CC -O3 -march=$MARCH -mabi=lp64d -ffunction-sections -fdata-sections -fno-stack-protector $INC \
    -c $R/ggml/src/ggml-cpu/quants.c -o fo_gen.o 2>g.err || { say "GEN_FAIL"; sed -n 1,12p g.err|tee -a $SEAL; exit 1; }
  ld -r fo_arch.o fo_gen.o -o factory.o || { say "LD_R_FAIL"; exit 1; }
  OPPSYM=$(nm factory.o 2>/dev/null | awk '$3=="ggml_vec_dot_tq1_0_q8_K"{print $2}' | head -1)
  say "# factory.o=$(wc -c<factory.o)B ggml_vec_dot_tq1_0_q8_K=[$OPPSYM] .comment=$(readelf -p .comment fo_arch.o 2>/dev/null|grep -ioE 'clang version [0-9.]+'|head -1)"
  printf '#include <stdlib.h>\n__attribute__((weak)) void ggml_abort(const char*f,int l,const char*fmt,...){(void)f;(void)l;(void)fmt;abort();}\n' > leafstub.c
  $CC -O2 -march=$MARCH -mabi=lp64d -c leafstub.c -o leafstub.o 2>/dev/null
  FACT_OBJ="factory.o leafstub.o"
else
  : "${GGML:?}"
  say "# opponent lib = $GGML/libggml-cpu.so md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)"
  say "# lib .comment = $(readelf -p .comment $GGML/libggml-cpu.so 2>/dev/null|grep -ioE 'clang version [0-9.]+'|sort -u|head -1)"
  say "# lib exports ggml_vec_dot_tq1_0_q8_K = $(nm -D $GGML/libggml-cpu.so 2>/dev/null|awk '$3=="ggml_vec_dot_tq1_0_q8_K"{print $2}'|head -1) (dispatch calls local _vl* specialization)"
  FACT_LIBS="-L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml"
  export LD_LIBRARY_PATH="$GGML:${LD_LIBRARY_PATH:-}"
fi

say "=== [A] ours DEPLOYED-EMIT leaf ($KERNEL_SRC, clang-18 -x $KX symmetric) + self-probe ==="
$CC -O3 -march=$MARCH -mabi=lp64d -x $KX "$KERNEL_SRC" -c -o k.o 2>ck.err || { say "KERN_FAIL"; sed -n 1,12p ck.err|tee -a $SEAL; exit 2; }
FP=$(objdump -d k.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo SOFTFP || echo cleanfp)
VS=$(objdump -d k.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
WMACC=$(objdump -d k.o 2>/dev/null | grep -cwE 'vwmacc.vv|vwmacc.vx')
WREDSUM=$(objdump -d k.o 2>/dev/null | grep -cwE 'vwredsum.vs')
WMULU=$(objdump -d k.o 2>/dev/null | grep -cwE 'vwmulu.vx|vwmulu.vv')
VMACC=$(objdump -d k.o 2>/dev/null | grep -cwE 'vmacc.vv')
VMUL=$(objdump -d k.o 2>/dev/null | grep -cwE 'vmul.vv|vmul.vx')
TOT=$(objdump -d k.o 2>/dev/null | grep -cE '^\s+[0-9a-f]+:')
VEC=$(objdump -d k.o 2>/dev/null | grep -cE '\bv[a-z][a-z]')
say "# OURS leaf: fp16=$FP vsetvli=$VS vwmacc=$WMACC vmacc=$VMACC vmul=$VMUL vwredsum=$WREDSUM vwmulu=$WMULU insn~$TOT vec~$VEC size=$(wc -c<k.o)B md5=$(md5sum "$KERNEL_SRC"|cut -d' ' -f1)"

say "=== [C] driver + link (-x c driver, C++ leaf, $STDCXX) ==="
$CC -O2 -march=$MARCH -mabi=lp64d -x c tq10_vecdot_driver.c -c -o driver.o 2>d.err || { say "DRV_FAIL"; sed -n 1,12p d.err|tee -a $SEAL; exit 3; }
LINK_OK=0
for drv in "$CC -O2 -march=$MARCH -mabi=lp64d -no-pie -Wl,--gc-sections" "gcc -O2 -no-pie -Wl,--gc-sections"; do
  if $drv driver.o k.o $FACT_OBJ $FACT_LIBS $STDCXX -lm -o bench 2>ld.err; then say "# LINK_OK via [$drv] bench=$(wc -c<bench)B"; LINK_OK=1; break; fi
done
[ $LINK_OK -eq 1 ] || { say "LINK_FAIL"; sed -n 1,20p ld.err|tee -a $SEAL; exit 4; }
DISP=$(objdump -d bench 2>/dev/null | grep -oE 'ggml_vec_dot_tq1_0_q8_K_(vl128|vl256|generic)' | sort -u | tr '\n' ',')
say "# opponent specialization visible in bench = [${DISP:-in-shared-lib}]"

say "=== [V] BYTE-EXACT VERIFY (ZERO-MODEL + 3-arm anti-hollow, 2 seeds) ==="
taskset -c $(echo $CORES|awk '{print $NF}') ./bench verify 8192 0xBEEF01 | tee -a $SEAL
taskset -c $(echo $CORES|awk '{print $NF}') ./bench verify 8192 0xC0FFEE | tee -a $SEAL

say "=== [S] COLD SWEEP (5-seed, load-gated, compiler-symmetric) ==="
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in $CORES; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=-1; BESTIDLE=-1
for c in $CORES; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); say "# core$c idle_pct=$pct";
  if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
if [ $BESTIDLE -lt 70 ]; then say "# LOAD_GATE_FAIL best=$BESTC idle=$BESTIDLE ABORT"; exit 9; fi
CORE=$BESTC
say "# LOAD_GATE_OK core=$CORE idle=${BESTIDLE}% freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)"
run(){ taskset -c $CORE ./bench "$@" 2>/dev/null; }
gf(){ echo "$1" | sed -nE "s/.*$2=([-0-9.e]+).*/\1/p"; }
gfp(){ echo "$1" | sed -nE 's/.*fingerprint=(0x[0-9a-f]+).*/\1/p'; }
N=4096; ROUNDS=20; POOL=256
say "# fmt | seed | ours_ns | opp_ns | RATIO_opp/ours | ours_iqr% | opp_iqr% | ours_GBs | opp_GBs | fp"
for seed in 0xBEEF01 0xC0FFEE 0x3 0x4 0x5; do
  run time ours    $N $ROUNDS $POOL 0 $seed >/dev/null
  LO=$(run time ours    $N $ROUNDS $POOL 0 $seed)
  LF=$(run time factory $N $ROUNDS $POOL 0 $seed)
  mo=$(gf "$LO" ns_per_block_median); mf=$(gf "$LF" ns_per_block_median)
  io=$(gf "$LO" ns_per_block_iqr);    iff=$(gf "$LF" ns_per_block_iqr)
  go=$(gf "$LO" achieved_GBs);        gff=$(gf "$LF" achieved_GBs)
  fpo=$(gfp "$LO"); fpf=$(gfp "$LF")
  r=$(awk -v a="$mf" -v b="$mo" 'BEGIN{if(b>0)printf "%.4f",a/b; else print "NA"}')
  oip=$(awk -v i="$io" -v m="$mo" 'BEGIN{if(m>0)printf "%.2f",100*i/m; else print "NA"}')
  fip=$(awk -v i="$iff" -v m="$mf" 'BEGIN{if(m>0)printf "%.2f",100*i/m; else print "NA"}')
  fpm=$([ "$fpo" = "$fpf" ] && echo MATCH || echo DIVERGE)
  say "RESULT tq1_0 | $seed | $mo | $mf | ${r}x | $oip | $fip | $go | $gff | $fpm"
done
say "=== ALL_DONE ==="
