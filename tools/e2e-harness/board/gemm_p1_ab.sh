#!/usr/bin/env bash
# gemm_p1_ab.sh -- [GAP-P1] board batch: paired A/B of the q4_0 PREFILL REPACK GEMM
# under the mf2(pre-fix, underfed) vs m1(post-fix, wide) core selection at VLEN256.
# Runs ON the board. SAME clang / SAME flags / SAME driver for both sides; only the
# exported kernel .c (mf2 vs m1) differs. Interleaved, core-pinned, freq-logged,
# cold-cache dropped per round, 2 passes (pass2 => T-N between-pass floor).
#
# args: $1=RDIR  $2=PRE_KERNEL_C(mf2)  $3=POST_KERNEL_C(m1)  $4=CORE
#       $5=EXP_VLEN  $6=K  $7=nr  $8=nc  $9=iters  $10=seed  $11=rounds
set -u
RDIR="$1"; PREK="$2"; POSTK="$3"; CORE="${4:-4}"; EXP_VLEN="${5:-256}"
K="${6:-2048}"; NR="${7:-128}"; NC="${8:-256}"; ITERS="${9:-40}"; SEED="${10:-0xBEEF01}"; ROUNDS="${11:-12}"
cd "$RDIR"
C=$(command -v clang-18 || command -v clang-20 || command -v clang-19 || command -v clang-17 || command -v clang)
MARCH="${TCRV_MARCH:-rv64gcv_zfh_zvfhmin_zvfh_zba_zbb_zbs}"
FLAGS="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=on"
OD=$(command -v llvm-objdump-18 || command -v llvm-objdump || command -v objdump)

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "clang=$C  $($C --version | head -1)"
echo "march=$MARCH  flags=$FLAGS"
echo "nproc=$(nproc)  pinned_core=$CORE"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/cpuinfo_max_freq 2>/dev/null)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "shape: K=$K nr=$NR nc=$NC iters=$ITERS seed=$SEED rounds=$ROUNDS"

freq() { cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }
coldcache() { sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true; }

# ---- compile: SAME clang/flags both sides ----
echo "== COMPILE =="
$C $FLAGS -x c++ "$PREK"  -c -o k_pre.o  2>cc_pre.err  && echo "k_pre.o(mf2) OK"  || { echo "COMPILE_FAIL pre";  cat cc_pre.err;  exit 20; }
$C $FLAGS -x c++ "$POSTK" -c -o k_post.o 2>cc_post.err && echo "k_post.o(m1) OK" || { echo "COMPILE_FAIL post"; cat cc_post.err; exit 21; }
$C $FLAGS -x c   gemm_timing_driver.c -c -o driver.o 2>cc_drv.err && echo "driver.o OK" || { echo "COMPILE_FAIL driver"; cat cc_drv.err; exit 22; }
$C $FLAGS driver.o k_pre.o  -lm -o gemm_pre  2>ld_pre.err  && echo "gemm_pre OK"  || { echo "LINK_FAIL pre";  cat ld_pre.err;  exit 23; }
$C $FLAGS driver.o k_post.o -lm -o gemm_post 2>ld_post.err && echo "gemm_post OK" || { echo "LINK_FAIL post"; cat ld_post.err; exit 24; }

# ==== FAIL-CLOSED PREFLIGHT (4-gate; before any measurement) ====
pf_fail(){ echo "PREFLIGHT_FAIL($1): $2"; exit 30; }
echo "== PREFLIGHT (fail-closed 4-gate) =="
BOARD_ISA=$(grep -m1 -i isa /proc/cpuinfo 2>/dev/null | tr 'A-Z' 'a-z')
[ -n "$BOARD_ISA" ] || pf_fail march "cannot read /proc/cpuinfo isa"
for ext in zfh zvfhmin zvfh zba zbb zbs; do
  if echo "$BOARD_ISA" | grep -Eq "(_|^| )$ext(_| |\$)" && ! echo "$MARCH" | grep -q "$ext"; then
    pf_fail march "board has '$ext' but march '$MARCH' omits it"
  fi
done
echo "PREFLIGHT(1) march-complete: OK"
for obj in k_pre.o k_post.o; do
  if $OD -d "$obj" 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee|__extendhfxf2'; then
    pf_fail libcall "$obj has soft-float fp16 libcall (crippled build)"
  fi
done
echo "PREFLIGHT(2) libcall-free: OK (both sides hardware fp16)"
[ -n "$C" ] || pf_fail compiler "no clang"
CVER=$($C --version | head -1); echo "$CVER" | grep -qi clang || pf_fail compiler "not clang"
echo "PREFLIGHT(3) same-compiler: OK ($CVER)"
ACT_VLEN=$(taskset -c $CORE ./gemm_pre $K 4 16 1 1 2>/dev/null | sed -n 's/.*VLEN=\([0-9]*\).*/\1/p' | head -1)
[ -n "$ACT_VLEN" ] || pf_fail fp-cell "cannot read board VLEN"
[ "$ACT_VLEN" = "$EXP_VLEN" ] || pf_fail fp-cell "board VLEN=$ACT_VLEN != target $EXP_VLEN"
echo "PREFLIGHT(4) fingerprint<->T-cell: OK (board VLEN=$ACT_VLEN == target $EXP_VLEN)"
echo "== PREFLIGHT PASS (4/4) =="

# ---- objdump seal: confirm the mf2 vs m1 core in the emitted machine code ----
echo "== OBJDUMP SEAL pre(mf2) vsetvli+vle8 =="
$OD -d --no-show-raw-insn k_pre.o  2>/dev/null | grep -iE 'vsetvli|vle8|vfmacc|vfwmul' | head -12
echo "== OBJDUMP SEAL post(m1) vsetvli+vle8 =="
$OD -d --no-show-raw-insn k_post.o 2>/dev/null | grep -iE 'vsetvli|vle8|vfmacc|vfwmul' | head -12

# ---- value-identity: pre & post must produce the SAME checksum (byte/value-exact) ----
echo "== VALUE-IDENTITY =="
CK_PRE=$(taskset -c $CORE ./gemm_pre  $K $NR $NC 2 $SEED 2>/dev/null | sed -n 's/.*cksum=\([0-9a-f]*\).*/\1/p')
CK_POST=$(taskset -c $CORE ./gemm_post $K $NR $NC 2 $SEED 2>/dev/null | sed -n 's/.*cksum=\([0-9a-f]*\).*/\1/p')
echo "cksum pre(mf2)=$CK_PRE  post(m1)=$CK_POST  $([ "$CK_PRE" = "$CK_POST" ] && echo IDENTICAL || echo DIVERGENT)"

# ---- paired A/B, 2 passes ----
run_ns(){ taskset -c $CORE "./$1" $K $NR $NC $ITERS $SEED 2>/dev/null | sed -n 's/.*kernel_ns=\([0-9.]*\).*/\1/p'; }
run_gm(){ taskset -c $CORE "./$1" $K $NR $NC $ITERS $SEED 2>/dev/null | sed -n 's/.*gmacs=\([0-9.]*\).*/\1/p'; }
# warmup drop
taskset -c $CORE ./gemm_pre  $K $NR $NC $ITERS $SEED >/dev/null 2>&1
taskset -c $CORE ./gemm_post $K $NR $NC $ITERS $SEED >/dev/null 2>&1
for PASS in 1 2; do
  echo "== ABPASS $PASS =="
  for r in $(seq 1 $ROUNDS); do
    coldcache
    fo=$(freq); no=$(run_ns gemm_pre);  go=$(run_gm gemm_pre)
    ff=$(freq); nf=$(run_ns gemm_post); gf=$(run_gm gemm_post)
    echo "AB pass=$PASS round=$r side=pre_mf2  kernel_ns=$no gmacs=$go freq_khz=$fo"
    echo "AB pass=$PASS round=$r side=post_m1  kernel_ns=$nf gmacs=$gf freq_khz=$ff"
  done
done
echo "== DONE =="
