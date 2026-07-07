#!/usr/bin/env bash
# gevm_q8_ab.sh -- [GAP-P1] q8_0 per-format board batch: 3-way A/B of the
# symmetric q8_0 REPACK GEVM strip-width core at VLEN256:
#   narrow = half_lanes 8 (two-strip, RVV1.0 authored default)
#   wide   = half_lanes 16 (one-strip, VLEN256 strip-width materializer)
#   m1     = integer_core_lmul m1 whole-LMUL (RVV0.7-form, compiled for RVV1.0)
# SAME clang / SAME flags / SAME driver; only the exported kernel .c differs.
# args: $1=RDIR $2=NARROW.c $3=WIDE.c $4=M1.c $5=CORE $6=EXP_VLEN $7=K $8=nc $9=iters $10=seed $11=rounds
set -u
RDIR="$1"; KN="$2"; KW="$3"; KM="$4"; CORE="${5:-4}"; EXP_VLEN="${6:-256}"
K="${7:-2048}"; NC="${8:-2048}"; ITERS="${9:-60}"; SEED="${10:-0xC0FFEE}"; ROUNDS="${11:-12}"
cd "$RDIR"
C=$(command -v clang-18 || command -v clang-20 || command -v clang-19 || command -v clang-17 || command -v clang)
MARCH="${TCRV_MARCH:-rv64gcv_zfh_zvfhmin_zvfh_zba_zbb_zbs}"
FLAGS="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=on"
OD=$(command -v llvm-objdump-18 || command -v llvm-objdump || command -v objdump)
echo "== ENV =="; echo "clang=$C $($C --version|head -1)"; echo "march=$MARCH"; echo "core=$CORE governor=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null)"
echo "shape K=$K nc=$NC iters=$ITERS seed=$SEED rounds=$ROUNDS"
freq(){ cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }
coldcache(){ sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true; }
echo "== COMPILE =="
$C $FLAGS -x c++ "$KN" -c -o kn.o 2>ccn.err && echo "narrow OK" || { echo FAIL_narrow; cat ccn.err; exit 20; }
$C $FLAGS -x c++ "$KW" -c -o kw.o 2>ccw.err && echo "wide OK"   || { echo FAIL_wide;  cat ccw.err; exit 21; }
$C $FLAGS -x c++ "$KM" -c -o km.o 2>ccm.err && echo "m1 OK"     || { echo FAIL_m1;    cat ccm.err; exit 22; }
$C $FLAGS -x c gevm_q8_timing_driver.c -c -o drv.o 2>ccd.err && echo "driver OK" || { echo FAIL_driver; cat ccd.err; exit 23; }
$C $FLAGS drv.o kn.o -lm -o gevm_narrow && $C $FLAGS drv.o kw.o -lm -o gevm_wide && $C $FLAGS drv.o km.o -lm -o gevm_m1 && echo LINK_OK || { echo LINK_FAIL; exit 24; }
# preflight: march-complete + libcall-free + VLEN
pf_fail(){ echo "PREFLIGHT_FAIL($1): $2"; exit 30; }
BOARD_ISA=$(grep -m1 -i isa /proc/cpuinfo|tr 'A-Z' 'a-z')
for ext in zfh zvfhmin zvfh zba zbb zbs; do echo "$BOARD_ISA"|grep -Eq "(_|^| )$ext(_| |\$)" && ! echo "$MARCH"|grep -q "$ext" && pf_fail march "board has $ext, march omits"; done
for o in kn.o kw.o km.o; do $OD -d "$o" 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee' && pf_fail libcall "$o soft-fp16"; done
AV=$(taskset -c $CORE ./gevm_narrow $K 16 1 1 2>/dev/null | sed -n 's/.*VLEN=\([0-9]*\).*/\1/p'|head -1)
[ "$AV" = "$EXP_VLEN" ] || pf_fail fp-cell "board VLEN=$AV != $EXP_VLEN"
echo "== PREFLIGHT OK (march-complete, libcall-free, VLEN=$AV) =="
echo "== OBJDUMP SEAL vle8 widths =="
echo "narrow:"; $OD -d kn.o 2>/dev/null | grep -iE 'vle8|vsetvli' | grep -iE 'mf2|m1,' | head -4
echo "wide:";   $OD -d kw.o 2>/dev/null | grep -iE 'vle8|vsetvli' | grep -iE 'mf2|m1,' | head -4
echo "m1:";     $OD -d km.o 2>/dev/null | grep -iE 'vle8|vsetvli' | grep -iE 'mf2|m1,' | head -4
echo "== VALUE-IDENTITY (all three must match) =="
cn=$(taskset -c $CORE ./gevm_narrow $K $NC 2 $SEED 2>/dev/null|sed -n 's/.*cksum=\([0-9a-f]*\).*/\1/p')
cw=$(taskset -c $CORE ./gevm_wide   $K $NC 2 $SEED 2>/dev/null|sed -n 's/.*cksum=\([0-9a-f]*\).*/\1/p')
cm=$(taskset -c $CORE ./gevm_m1     $K $NC 2 $SEED 2>/dev/null|sed -n 's/.*cksum=\([0-9a-f]*\).*/\1/p')
echo "cksum narrow=$cn wide=$cw m1=$cm  $([ "$cn" = "$cw" ] && [ "$cw" = "$cm" ] && echo ALL_IDENTICAL || echo DIVERGENT)"
gm(){ taskset -c $CORE "./$1" $K $NC $ITERS $SEED 2>/dev/null | sed -n 's/.*gbps=\([0-9.]*\).*/\1/p'; }
ns(){ taskset -c $CORE "./$1" $K $NC $ITERS $SEED 2>/dev/null | sed -n 's/.*kernel_ns=\([0-9.]*\).*/\1/p'; }
taskset -c $CORE ./gevm_narrow $K $NC $ITERS $SEED >/dev/null 2>&1
for PASS in 1 2; do
  echo "== ABPASS $PASS =="
  for r in $(seq 1 $ROUNDS); do
    coldcache
    fa=$(freq); nn=$(ns gevm_narrow); gn=$(gm gevm_narrow)
    fb=$(freq); nw=$(ns gevm_wide);   gw=$(gm gevm_wide)
    fc=$(freq); nm=$(ns gevm_m1);     gmm=$(gm gevm_m1)
    echo "AB pass=$PASS round=$r side=narrow_hl8  kernel_ns=$nn gbps=$gn freq_khz=$fa"
    echo "AB pass=$PASS round=$r side=wide_hl16   kernel_ns=$nw gbps=$gw freq_khz=$fb"
    echo "AB pass=$PASS round=$r side=m1_wholeLMUL kernel_ns=$nm gbps=$gmm freq_khz=$fc"
  done
done
echo "== DONE =="
