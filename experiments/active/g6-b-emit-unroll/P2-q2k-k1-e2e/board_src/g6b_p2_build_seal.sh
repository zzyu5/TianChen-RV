#!/usr/bin/env bash
# [G6-B Phase-2 q2_K @ k1] Build THREE libggml-cpu.so variants from the SAME k1 tcrv tree
# (clang-18 SYMMETRIC; the ONLY A/B/C diff = the q2_K GEMM main-term emission schedule):
#   OFF         = STOCK hand-tuned RVV q2_K 16x1 repack (case256, real opponent / hand-brick)
#   ON_UNROLLED = OUR emitted VLA S6-tiled GEMM, FULL-STATIC-unroll (Phase-1 default; == g5 baseline .873x)
#   ON_ROLLED   = OUR emitted VLA S6-tiled GEMM, rolled-loop main term (emit_loop_schedule="rolled")
# GEVM is identical in both ON variants (rolled only touches the GEMM main term). make_block (straight)
# + arch body-replacement are identical across ON_UNROLLED/ON_ROLLED -> ON variants differ ONLY in the
# gemm .inc content. Dedicated build /data/build-k1-q2k-rolled (seeded from k1build-stock; g5's
# /data/build-k1-q2k + /tmp/g5_q2k UNTOUCHED). Restores 3 source files byte-exact at the end.
set -uo pipefail
T=/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu
GEN=$T/repack.cpp; HDR=$T/repack.h; ARCH=$T/arch/riscv/repack.cpp; ARCHDIR=$T/arch/riscv
STOCK=/data/k1build-stock
BUILD=/data/build-k1-q2k-rolled
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SRCDB=/data/k1build/compile_commands.json
SCR=/tmp/g6b-p2-q2k
BASE_GEN=3cac40aa55aece1f69e3d08d4e7e9ae2
BASE_HDR=57851439e7c6f5e35aca7986e148e42b
BASE_ARCH=c3c101fdcc07cf803c4b70551c94343a
GEMM_SYM=weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K
GEVM_SYM=weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K
mkdir -p "$SCR"

echo "[g6b-p2] board=k1 SpacemiT-X60 VLEN256 clang-18 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify (3 files) + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); H0=$(md5sum "$HDR"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN =$G0 (base $BASE_GEN)"; echo "  HDR =$H0 (base $BASE_HDR)"; echo "  ARCH=$A0 (base $BASE_ARCH)"
if [ "$G0" != "$BASE_GEN" ] || [ "$H0" != "$BASE_HDR" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$HDR" "$SCR/repack.h.ORIG"; cp "$ARCH" "$SCR/arch_repack.cpp.ORIG"
echo "  .inc md5s: gemm_U=$(md5sum "$SCR/weft_emitted_gemm_q2_K.UNROLLED.inc"|awk '{print $1}') gemm_R=$(md5sum "$SCR/weft_emitted_gemm_q2_K.ROLLED.inc"|awk '{print $1}') gevm=$(md5sum "$SCR/weft_emitted_gevm_q2_K.inc"|awk '{print $1}')"

echo "=== extract exact compile commands from configured tree ==="
python3 - "$SRCDB" "$BUILD" "$SCR" <<'PY'
import json,sys
db,build,scr=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(db))
def cmd(suffix):
    hits=[e['command'] for e in d if e['file'].endswith(suffix)]
    assert len(hits)==1, f"{suffix}: {len(hits)} hits"
    return hits[0]
cg=cmd('ggml-cpu/repack.cpp'); ca=cmd('arch/riscv/repack.cpp')
open(scr+'/cmd_gen.sh','w').write(f"cd {build}/ggml/src\n"+cg+"\n")
open(scr+'/cmd_arch.sh','w').write(f"cd {build}/ggml/src\n"+ca+"\n")
print("  cmd_gen  ->",scr+"/cmd_gen.sh")
print("  cmd_arch ->",scr+"/cmd_arch.sh")
PY

echo "=== seed dedicated build dir (cp -a stock) ==="
rm -rf "$BUILD"; cp -a "$STOCK" "$BUILD"

recompile(){   # recompile 2 .o from CURRENT source + relink ggml-cpu
  bash "$SCR/cmd_gen.sh"  >/tmp/g6bp2_cc_gen.log  2>&1 || { echo "  cc gen FAIL";  tail -30 /tmp/g6bp2_cc_gen.log;  return 1; }
  bash "$SCR/cmd_arch.sh" >/tmp/g6bp2_cc_arch.log 2>&1 || { echo "  cc arch FAIL"; tail -40 /tmp/g6bp2_cc_arch.log; return 1; }
  ( cd "$BUILD/ggml/src" && bash CMakeFiles/ggml-cpu.dir/link.txt ) >/tmp/g6bp2_link.log 2>&1 || { echo "  link FAIL"; tail -20 /tmp/g6bp2_link.log; return 1; }
  return 0
}
recompile_arch_only(){   # recompile ONLY arch .o (gemm .inc swapped) + relink
  bash "$SCR/cmd_arch.sh" >/tmp/g6bp2_cc_arch.log 2>&1 || { echo "  cc arch FAIL"; tail -40 /tmp/g6bp2_cc_arch.log; return 1; }
  ( cd "$BUILD/ggml/src" && bash CMakeFiles/ggml-cpu.dir/link.txt ) >/tmp/g6bp2_link.log 2>&1 || { echo "  link FAIL"; tail -20 /tmp/g6bp2_link.log; return 1; }
  return 0
}

echo "=== [1] build OFF variant (baseline source = STOCK hand-tuned q2_K repack) ==="
recompile || { echo "  OFF build FAIL"; exit 20; }
cp -a "$LIVE" "$SCR/libggml-cpu.so.OFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.OFF"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.OFF" 2>/dev/null | grep -cE "weft_emitc_ggml_repack_gem[vm]_q2_K" || true)
echo "  OFF .so md5=$OFFMD5  q2_K_weft_emit_sym(expect 0)=$OFFSYM"

echo "=== [2] apply deploy patch + build ON_UNROLLED (gemm .inc = UNROLLED) ==="
cp -f "$SCR/weft_emitted_gevm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"
cp -f "$SCR/weft_emitted_gemm_q2_K.UNROLLED.inc" "$ARCHDIR/weft_emitted_gemm_q2_K.inc"
if ! python3 "$SCR/deploy_patch_q2_K_emitted.py"; then
  echo "  PATCH FAILED -- restoring"; cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
  rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"; exit 11
fi
if ! recompile; then
  echo "  ON_UNROLLED build FAIL -- restoring"; cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
  rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"; exit 21
fi
cp -a "$LIVE" "$SCR/libggml-cpu.so.ON_UNROLLED"
UNMD5=$(md5sum "$SCR/libggml-cpu.so.ON_UNROLLED"|awk '{print $1}')
echo "  ON_UNROLLED .so md5=$UNMD5"

echo "=== [3] swap gemm .inc -> ROLLED + recompile arch only + build ON_ROLLED ==="
cp -f "$SCR/weft_emitted_gemm_q2_K.ROLLED.inc" "$ARCHDIR/weft_emitted_gemm_q2_K.inc"
if ! recompile_arch_only; then
  echo "  ON_ROLLED build FAIL -- restoring"; cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
  rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"; exit 22
fi
cp -a "$LIVE" "$SCR/libggml-cpu.so.ON_ROLLED"
RLMD5=$(md5sum "$SCR/libggml-cpu.so.ON_ROLLED"|awk '{print $1}')
echo "  ON_ROLLED .so md5=$RLMD5"

echo "=== [4] SEAL: nm symbols + banners + objdump (all 3) ==="
for v in ON_UNROLLED ON_ROLLED; do
  echo "  -- $v --"
  nm -C "$SCR/libggml-cpu.so.$v" 2>/dev/null | grep -E "weft_emitc_ggml_repack_gem[vm]_q2_K" | sed 's/^/      /'
  GEVMB=$(strings "$SCR/libggml-cpu.so.$v" | grep -c "WEFT G5-q2K EMITTED GEVM" || true)
  GEMMB=$(strings "$SCR/libggml-cpu.so.$v" | grep -c "WEFT G5-q2K EMITTED GEMM" || true)
  echo "      banner gevm=$GEVMB gemm=$GEMMB"
  objdump -d --disassemble="$GEMM_SYM" "$SCR/libggml-cpu.so.$v" 2>/dev/null > "$SCR/objdump_${v}_gemm.txt"
  NVSET=$(grep -cE 'vset[i]*vli' "$SCR/objdump_${v}_gemm.txt" || true)
  VWMACC=$(grep -cE 'vwmacc' "$SCR/objdump_${v}_gemm.txt" || true)
  SPILL=$(grep -cE 'vs[0-9]+r\.v' "$SCR/objdump_${v}_gemm.txt" || true)
  echo "      OUR GEMM: vsetvli=$NVSET vwmacc=$VWMACC spill=$SPILL"
done
echo "  [$([ "$UNMD5" != "$OFFMD5" ] && echo ok) ON_UNROLLED!=OFF] [$([ "$RLMD5" != "$OFFMD5" ] && echo ok) ON_ROLLED!=OFF] [$([ "$RLMD5" != "$UNMD5" ] && echo ok) ROLLED!=UNROLLED]"
# opponent (stock) seal
for sym in ggml_gemm_q2_K_16x1_q8_K ggml_gemv_q2_K_16x1_q8_K; do
  objdump -d --disassemble="$sym" "$SCR/libggml-cpu.so.OFF" 2>/dev/null > "$SCR/objdump_STOCK_$sym.txt"
  NVSET=$(grep -cE 'vset[i]*vli' "$SCR/objdump_STOCK_$sym.txt" || true)
  VWMACC=$(grep -cE 'vwmacc' "$SCR/objdump_STOCK_$sym.txt" || true)
  SPILL=$(grep -cE 'vs[0-9]+r\.v' "$SCR/objdump_STOCK_$sym.txt" || true)
  echo "    STOCK $sym: vsetvli=$NVSET vwmacc=$VWMACC spill=$SPILL"
done

echo "=== [5] restore 3 source files + rm .inc + rebuild OFF-pristine live ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); HR=$(md5sum "$HDR"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
echo "  restored md5: GEN=$GR HDR=$HR ARCH=$AR"
if [ "$GR" = "$BASE_GEN" ] && [ "$HR" = "$BASE_HDR" ] && [ "$AR" = "$BASE_ARCH" ]; then echo "  SOURCE RESTORED byte-exact"; else echo "  *** restore mismatch"; exit 30; fi
cp -f "$SCR/libggml-cpu.so.OFF" "$LIVE"
echo "=== [g6b-p2 build+seal DONE] ==="
md5sum "$SCR/libggml-cpu.so.OFF" "$SCR/libggml-cpu.so.ON_UNROLLED" "$SCR/libggml-cpu.so.ON_ROLLED"
