#!/usr/bin/env bash
# [G5 q2_K @ k1] Build OFF (baseline: STOCK hand-tuned RVV q2_K 16x1 repack) + ON (our-emitted
# VLA S6-tiled GEMM + plain GEVM replacing the stock bodies) libggml-cpu variants from the SAME
# k1 tcrv tree (clang-18 SYMMETRIC; only the q2_K make_block + the two q2_K arch kernel bodies
# differ). Dedicated build dir /data/build-k1-q2k (seeded from k1build-stock; main build/ +
# build-k1-q5k untouched). Recompiles ONLY the two changed .o (repack.cpp.o + arch/riscv/
# repack.cpp.o), relinks ggml-cpu. Seal: nm weft_emitc q2_K symbol absent(OFF)/present(ON),
# banners, objdump. Then RESTORE 2 source files + rm .inc + rebuild OFF-pristine.
#
# NOTE: q2_K@k1 opponent = STOCK hand-tuned RVV repack (case256 fires), NOT block-dot -> this is
# the q4_K Win-K1-VLEN contest (our-emit vs 真出货 hand-brick), not the q5_K net-new-vs-block-dot.
set -uo pipefail
T=/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu
GEN=$T/repack.cpp; HDR=$T/repack.h; ARCH=$T/arch/riscv/repack.cpp; ARCHDIR=$T/arch/riscv
STOCK=/data/k1build-stock
BUILD=/data/build-k1-q2k
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SRCDB=/data/k1build/compile_commands.json
SCR=/tmp/g5_q2k
BASE_GEN=3cac40aa55aece1f69e3d08d4e7e9ae2
BASE_HDR=57851439e7c6f5e35aca7986e148e42b
BASE_ARCH=c3c101fdcc07cf803c4b70551c94343a
GEMM_SYM=weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K
GEVM_SYM=weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K
mkdir -p "$SCR"

echo "[g5-q2_K] board=k1 SpacemiT-X60 VLEN256 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify (3 files) + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); H0=$(md5sum "$HDR"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN =$G0 (base $BASE_GEN)"; echo "  HDR =$H0 (base $BASE_HDR)"; echo "  ARCH=$A0 (base $BASE_ARCH)"
if [ "$G0" != "$BASE_GEN" ] || [ "$H0" != "$BASE_HDR" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$HDR" "$SCR/repack.h.ORIG"; cp "$ARCH" "$SCR/arch_repack.cpp.ORIG"
echo "  .inc md5s: gemm=$(md5sum "$SCR/weft_emitted_gemm_q2_K.inc"|awk '{print $1}') gevm=$(md5sum "$SCR/weft_emitted_gevm_q2_K.inc"|awk '{print $1}')"

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
  bash "$SCR/cmd_gen.sh"  >/tmp/g5_q2k_cc_gen.log  2>&1 || { echo "  cc gen FAIL";  tail -30 /tmp/g5_q2k_cc_gen.log;  return 1; }
  bash "$SCR/cmd_arch.sh" >/tmp/g5_q2k_cc_arch.log 2>&1 || { echo "  cc arch FAIL"; tail -40 /tmp/g5_q2k_cc_arch.log; return 1; }
  ( cd "$BUILD/ggml/src" && bash CMakeFiles/ggml-cpu.dir/link.txt ) >/tmp/g5_q2k_link.log 2>&1 || { echo "  link FAIL"; tail -20 /tmp/g5_q2k_link.log; return 1; }
  return 0
}

echo "=== [1] build OFF variant (baseline source = STOCK hand-tuned q2_K repack) ==="
recompile || { echo "  OFF build FAIL"; exit 20; }
cp -a "$LIVE" "$SCR/libggml-cpu.so.OFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.OFF"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.OFF" 2>/dev/null | grep -cE "weft_emitc_ggml_repack_gem[vm]_q2_K" || true)
echo "  OFF .so md5=$OFFMD5  q2_K_weft_emit_sym(expect 0)=$OFFSYM"

echo "=== [2] copy emitted .inc + apply patch + build ON ==="
cp -f "$SCR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gemm_q2_K.inc"
cp -f "$SCR/weft_emitted_gevm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"
if ! python3 "$SCR/deploy_patch_q2_K_emitted.py"; then
  echo "  PATCH FAILED -- restoring source"; cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
  rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"; exit 11
fi
if ! recompile; then
  echo "  ON build FAIL -- restoring source"; cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
  rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"; exit 21
fi
cp -a "$LIVE" "$SCR/libggml-cpu.so.ON"
ONMD5=$(md5sum "$SCR/libggml-cpu.so.ON"|awk '{print $1}')
echo "  ON .so md5=$ONMD5"

echo "=== [3] SEAL: nm symbols + banners + objdump ==="
nm -C "$SCR/libggml-cpu.so.ON" 2>/dev/null | grep -E "weft_emitc_ggml_repack_gem[vm]_q2_K" | sed 's/^/    /'
ONGEVM=$(strings "$SCR/libggml-cpu.so.ON" | grep -c "WEFT G5-q2K EMITTED GEVM" || true)
ONGEMM=$(strings "$SCR/libggml-cpu.so.ON" | grep -c "WEFT G5-q2K EMITTED GEMM" || true)
echo "  banner gevm(expect>=1)=$ONGEVM  gemm(expect>=1)=$ONGEMM"
[ "$ONMD5" != "$OFFMD5" ] && echo "  ON != OFF (variants differ, good)" || echo "  *** ON == OFF -- patch had no effect!"
for sym in "$GEVM_SYM" "$GEMM_SYM"; do
  objdump -d --disassemble="$sym" "$SCR/libggml-cpu.so.ON" 2>/dev/null > "$SCR/objdump_$sym.txt"
  NVSET=$(grep -cE 'vset[i]*vli' "$SCR/objdump_$sym.txt" || true)
  VForms=$(grep -oE 'vset[i]*vli[^#]*' "$SCR/objdump_$sym.txt" | grep -oE 'e(8|16|32),(mf2|mf4|m1|m2|m4)' | sort | uniq -c | tr '\n' ' ')
  VWMACC=$(grep -cE 'vwmacc' "$SCR/objdump_$sym.txt" || true)
  SPILL=$(grep -cE 'vs[0-9]+r\.v' "$SCR/objdump_$sym.txt" || true)
  echo "    $sym: vsetvli=$NVSET vwmacc=$VWMACC spill=$SPILL  SEW,LMUL: $VForms"
done
# also seal the STOCK bodies from OFF (opponent identity)
for sym in ggml_gemm_q2_K_16x1_q8_K ggml_gemv_q2_K_16x1_q8_K; do
  objdump -d --disassemble="$sym" "$SCR/libggml-cpu.so.OFF" 2>/dev/null > "$SCR/objdump_STOCK_$sym.txt"
  NVSET=$(grep -cE 'vset[i]*vli' "$SCR/objdump_STOCK_$sym.txt" || true)
  VWMACC=$(grep -cE 'vwmacc' "$SCR/objdump_STOCK_$sym.txt" || true)
  SPILL=$(grep -cE 'vs[0-9]+r\.v' "$SCR/objdump_STOCK_$sym.txt" || true)
  echo "    STOCK $sym: vsetvli=$NVSET vwmacc=$VWMACC spill=$SPILL"
done

echo "=== [4] restore 2 source files + rm .inc + rebuild OFF-pristine ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
rm -f "$ARCHDIR/weft_emitted_gemm_q2_K.inc" "$ARCHDIR/weft_emitted_gevm_q2_K.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); HR=$(md5sum "$HDR"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
echo "  restored md5: GEN=$GR HDR=$HR ARCH=$AR"
if [ "$GR" = "$BASE_GEN" ] && [ "$HR" = "$BASE_HDR" ] && [ "$AR" = "$BASE_ARCH" ]; then echo "  SOURCE RESTORED byte-exact"; else echo "  *** restore mismatch"; exit 30; fi
cp -f "$SCR/libggml-cpu.so.OFF" "$LIVE"
echo "=== [g5-q2_K build+seal DONE] ==="
md5sum "$SCR/libggml-cpu.so.OFF" "$SCR/libggml-cpu.so.ON"
