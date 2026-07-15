#!/usr/bin/env bash
set -u
R=/home/bianbu/tcrv-k1-llama
M=rv64gcv_zfh_zvfh_zicbop_zihintpause
CC=clang-18
cd /tmp/a2_iqtq
SEAL=build_seal.txt; : > $SEAL
say(){ echo "$@" | tee -a $SEAL; }
INC="-I$R/ggml/src -I$R/ggml/src/ggml-cpu -I$R/ggml/include"
say "# BUILD iq/tq @k1 CC=$($CC --version|head -1) march=$M"
say "# ggml_src=$R commit=$(cd $R && git log -1 --format='%h %ci' 2>/dev/null)"
FMTS="iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs iq4_xs tq2_0 tq1_0"

say "=== [B] factory (arch/riscv fno-integrated-as + generic), ld -r ==="
$CC -O3 -march=$M -mabi=lp64d -fno-integrated-as -ffunction-sections -fdata-sections -fno-stack-protector $INC \
  -c $R/ggml/src/ggml-cpu/arch/riscv/quants.c -o fo_arch.o 2>a.err || { say "ARCH_FAIL"; sed -n 1,8p a.err|tee -a $SEAL; exit 1; }
say "# arch/riscv/quants.o = $(wc -c<fo_arch.o)B"
$CC -O3 -march=$M -mabi=lp64d -ffunction-sections -fdata-sections -fno-stack-protector $INC \
  -c $R/ggml/src/ggml-cpu/quants.c -o fo_gen.o 2>g.err || { say "GEN_FAIL"; sed -n 1,8p g.err|tee -a $SEAL; exit 1; }
say "# quants.o = $(wc -c<fo_gen.o)B"
ld -r fo_arch.o fo_gen.o -o factory.o || { say "LD_R_FAIL"; exit 1; }
say "# factory.o = $(wc -c<factory.o)B  .comment=$(readelf -p .comment fo_arch.o 2>/dev/null|grep -ioE 'clang version [0-9.]+'|head -1)"
say "=== [B.probe] opponent identity ==="
for f in $FMTS; do s=ggml_vec_dot_${f}_q8_K; say "# OPP_$f: $s = $(nm factory.o 2>/dev/null|awk -v x=$s '$3==x{print $2}'|head -1)"; done

say "=== [A] ours kernels clang-18 -O3 ==="
OBJS=""
for f in $FMTS; do
  $CC -O3 -march=$M -mabi=lp64d -x c++ kernels/$f.kernel.c -c -o k_$f.o 2>ck_$f.err || { say "KERN_FAIL $f"; sed -n 1,6p ck_$f.err|tee -a $SEAL; exit 2; }
  FP=$(objdump -d k_$f.o 2>/dev/null|grep -Eqc '__truncsfhf2|__extendhfsf2' && echo SOFTFP||echo cleanfp)
  VS=$(objdump -d k_$f.o 2>/dev/null|grep -cE 'vsetvli|vsetivli')
  say "# OURS_$f: $FP vsetvli=$VS $(wc -c<k_$f.o)B md5=$(md5sum kernels/$f.kernel.c|cut -d' ' -f1)"
  OBJS="$OBJS k_$f.o"
done

say "=== [C] driver + link ==="
$CC -O2 -march=$M -mabi=lp64d -x c format_micro_driver.c -c -o driver.o 2>d.err || { say DRVFAIL; sed -n 1,8p d.err|tee -a $SEAL; exit 3; }
printf '#include <stdlib.h>\n__attribute__((weak)) void ggml_abort(const char*f,int l,const char*fmt,...){(void)f;(void)l;(void)fmt;abort();}\n' > leafstub.c
$CC -O2 -march=$M -mabi=lp64d -c leafstub.c -o leafstub.o 2>/dev/null
LINK_OK=0
for drv in "$CC -O2 -march=$M -mabi=lp64d -no-pie -Wl,--gc-sections" "gcc -O2 -no-pie -Wl,--gc-sections"; do
  if $drv driver.o $OBJS factory.o leafstub.o -lm -o bench 2>ld.err; then say "# LINK_OK via [$drv] bench=$(wc -c<bench)B"; LINK_OK=1; break; fi
done
[ $LINK_OK -eq 1 ] || { say "LINK_FAIL"; sed -n 1,20p ld.err|tee -a $SEAL; exit 4; }
say "=== BUILD_DONE_K1 ==="
