#!/usr/bin/env bash
# [CASE-COMPILER-ASYMMETRY Stage-1 rvv] Phase A+B: compile + objdump-seal + link + identity
# Symmetric remeasure of the 3 load-bearing asymmetric rvv/VLEN128 cells (replicates 0b methodology):
#   F1 q2_K-S6   (T8 r69/32b, historical clang-ours/gcc-opp = 1.413x) kernel=tiled_q2K_gemm.c md5 9c5bac28
#   F2 q5_K-T3S6 (T8 r71/34b, historical = 2.193x)                    kernel=tiled_q5K_gemm.c md5 c209226b
#   F3 q5_K-L1   (T8 r61/25,  historical = 1.50-1.62x, untiled)       kernel=untiled_q5K_gemm.c md5 ba30ba54
# Compile OUR kernel with BOTH gcc-15.2.0 (deploy compiler => SYMMETRIC vs gcc opponent) and clang-17
# (=> CLANG-domain reference reproducing the historical asymmetric number). Opponent block-dot =
# gcc-15 upstream libggml-cpu.so (read-only). march=rv64gcv_zvfh matched both sides (Stage-0a: gcc
# march-invariant; clang rejects full deploy march). NO A-tree mutation, NO git.
set -uo pipefail
RDIR=/tmp/case_stage1_rvv
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
GCC=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
CLANG=/usr/bin/clang-17
MARCH=rv64gcv_zvfh
mkdir -p "$RDIR"; cd "$RDIR"

# --- gather kernel sources (cached exports) + drivers (scp'd separately) ---
cp -f /tmp/kquant_tile_q2k_t3_ab/tiled_q2K_gemm.c   ./q2K_tiled.c
cp -f /tmp/kquant_tile_q5k_t3_ab/tiled_q5K_gemm.c   ./q5K_tiled.c
cp -f /tmp/kquant_tile_q5k_t3_ab/untiled_q5K_gemm.c ./q5K_untiled.c
echo "[src md5]"; md5sum q2K_tiled.c q5K_tiled.c q5K_untiled.c
echo "[env] gcc=$($GCC --version|head -1)  clang=$($CLANG --version|head -1)"
echo "[preflight] loadavg=$(cat /proc/loadavg) gov=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_governor 2>/dev/null||echo NA) freq=$(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)"
ls "$GGML/libggml-cpu.so" >/dev/null || { echo "MISSING upstream libggml-cpu.so"; exit 40; }

# ---------- Phase A: compile ours x {gcc,clang} ----------
cc_one(){ # $1=cc $2=tag $3=src
  local t0=$SECONDS
  "$1" -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$3" -c -o "$2.o" 2>"$2.cc.err" \
    && echo "  [$2] OK $((SECONDS-t0))s $(stat -c%s $2.o)B" || { echo "  [$2] FAIL"; tail -6 "$2.cc.err"; }
}
echo "=== [A] compile ours (gcc-15 SYMMETRIC + clang-17 REFERENCE), -O2 ==="
cc_one "$GCC"   q2K_gcc   q2K_tiled.c
cc_one "$CLANG" q2K_clang q2K_tiled.c
cc_one "$GCC"   q5T3_gcc  q5K_tiled.c
cc_one "$CLANG" q5T3_clang q5K_tiled.c
cc_one "$GCC"   q5L1_gcc  q5K_untiled.c
cc_one "$CLANG" q5L1_clang q5K_untiled.c

# ---------- objdump seal ----------
seal(){ local DIS; DIS=$(objdump -d "$1" 2>/dev/null)
  printf '  %-12s vsetvli=%-5s spill=%-5s reload=%-5s vwmacc=%-5s maxVreg=v%-3s textB=%s\n' "$2" \
    "$(printf '%s' "$DIS"|grep -cE 'vsetvli|vsetivli')" \
    "$(printf '%s' "$DIS"|grep -cE 'vs[1248]r\.v')" \
    "$(printf '%s' "$DIS"|grep -cE 'vl[1248]r\.v')" \
    "$(printf '%s' "$DIS"|grep -cE 'vwmacc')" \
    "$(printf '%s' "$DIS"|grep -oE '[ ,(]v[0-9]+'|grep -oE '[0-9]+'|sort -n|tail -1)" \
    "$(objdump -h "$1" 2>/dev/null|awk '/\.text/{print strtonum("0x"$3)}')"
}
echo "=== [A-seal] objdump gcc-vs-clang (0a-style: vsetvli/spill storm) ==="
echo "-- F1 q2_K-S6 --"; seal q2K_gcc.o 'q2K gcc';  seal q2K_clang.o 'q2K clang'
echo "-- F2 q5_K-T3S6 --"; seal q5T3_gcc.o 'q5T3 gcc'; seal q5T3_clang.o 'q5T3 clang'
echo "-- F3 q5_K-L1 --"; seal q5L1_gcc.o 'q5L1 gcc'; seal q5L1_clang.o 'q5L1 clang'

# ---------- Phase B: link timing binaries (ours .o + gcc opponent block-dot) ----------
echo "=== [B] link timing binaries (driver + ours.o + gcc-blockdot .so) ==="
ln_one(){ # $1=out $2=driver.o $3=kernel.o
  "$CLANG" "$2" "$3" -o "$1" -L"$GGML" -Wl,-rpath,"$GGML" -lggml-cpu -lggml-base -lggml -lm 2>"$1.ln.err" \
    && echo "  linked $1" || { echo "  link $1 FAIL"; tail -5 "$1.ln.err"; }
}
$CLANG -O2 -march=$MARCH -mabi=lp64d -x c q2k_driver.c -c -o drv_q2k.o 2>/dev/null
$CLANG -O2 -march=$MARCH -mabi=lp64d -x c q5k_driver.c -c -o drv_q5k.o 2>/dev/null
ln_one bin_q2K_gcc   drv_q2k.o q2K_gcc.o
ln_one bin_q2K_clang drv_q2k.o q2K_clang.o
ln_one bin_q5T3_gcc   drv_q5k.o q5T3_gcc.o
ln_one bin_q5T3_clang drv_q5k.o q5T3_clang.o
ln_one bin_q5L1_gcc   drv_q5k.o q5L1_gcc.o
ln_one bin_q5L1_clang drv_q5k.o q5L1_clang.o

# ---------- identity: gcc-build output == clang-build output? (explicit RVV intrinsics => fixed op multiset) ----------
echo "=== [B-identity] gcc-build vs clang-build byte-exact output cmp (K=2048 nr=64 nc=512) ==="
idcmp(){ # $1=fmt-env $2=gccbin $3=clangbin
  local D=$RDIR; local ev=$1
  env ${ev}=$D/o_gcc.bin  taskset -c 8 ./$2 ${ev%%_DUMP}_x 2>/dev/null >/dev/null || true
}
# q2_K
Q2K_DUMP=$RDIR/oq2_gcc.bin   taskset -c 8 ./bin_q2K_gcc   q2_K 2048 64 512 10 0xC0FFEE >/dev/null 2>&1
Q2K_DUMP=$RDIR/oq2_clang.bin taskset -c 8 ./bin_q2K_clang q2_K 2048 64 512 10 0xC0FFEE >/dev/null 2>&1
if cmp -s oq2_gcc.bin oq2_clang.bin; then echo "  F1 q2_K  IDENTITY gcc==clang BYTE-EXACT"; else echo "  F1 q2_K  DIFFER: $(cmp oq2_gcc.bin oq2_clang.bin 2>&1|head -1)"; fi
# q5_K T3
Q5K_DUMP=$RDIR/oq5t3_gcc.bin   taskset -c 8 ./bin_q5T3_gcc   q5_K 2048 64 512 10 0xC0FFEE >/dev/null 2>&1
Q5K_DUMP=$RDIR/oq5t3_clang.bin taskset -c 8 ./bin_q5T3_clang q5_K 2048 64 512 10 0xC0FFEE >/dev/null 2>&1
if cmp -s oq5t3_gcc.bin oq5t3_clang.bin; then echo "  F2 q5T3  IDENTITY gcc==clang BYTE-EXACT"; else echo "  F2 q5T3  DIFFER: $(cmp oq5t3_gcc.bin oq5t3_clang.bin 2>&1|head -1)"; fi
# q5_K L1
Q5K_DUMP=$RDIR/oq5l1_gcc.bin   taskset -c 8 ./bin_q5L1_gcc   q5_K 2048 64 512 10 0xC0FFEE >/dev/null 2>&1
Q5K_DUMP=$RDIR/oq5l1_clang.bin taskset -c 8 ./bin_q5L1_clang q5_K 2048 64 512 10 0xC0FFEE >/dev/null 2>&1
if cmp -s oq5l1_gcc.bin oq5l1_clang.bin; then echo "  F3 q5L1  IDENTITY gcc==clang BYTE-EXACT"; else echo "  F3 q5L1  DIFFER: $(cmp oq5l1_gcc.bin oq5l1_clang.bin 2>&1|head -1)"; fi
echo "=== [Stage-1 compile/seal/link DONE] ==="
