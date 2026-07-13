#!/usr/bin/env bash
# [G7-L3 q5_x@k1 net-new] Build OFF(block-dot, pristine) + ON(our repack) libggml-cpu
# from the SAME k1 tree (clang-18 SYMMETRIC; only the q5_x net-new scaffold differs).
# Shared tree patched in place then RESTORED (.ORIG + md5 double-proof). Dedicated
# build dir seeded from stock. Usage: q5_build_seal.sh q5_0 | q5_1
set -uo pipefail
FMT="${1:?usage: q5_build_seal.sh q5_0|q5_1}"
case "$FMT" in
  q5_0) N=50 ;;
  q5_1) N=51 ;;
  *) echo "unknown FMT $FMT"; exit 2 ;;
esac
T=/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu
GEN=$T/repack.cpp; HDR=$T/repack.h; ARCH=$T/arch/riscv/repack.cpp
STOCK=/data/k1build-stock
BUILD=/data/build-k1-$FMT
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SRCDB=/data/k1build/compile_commands.json
BASE=/data/g7q$N
SCR=$BASE/scratch
GEMM_INC=weft_emitted_gemm_$FMT.inc
GEVM_INC=weft_emitted_gevm_$FMT.inc
PATCH=$BASE/deploy_patch_${FMT}_k1.py
BASE_GEN=3cac40aa55aece1f69e3d08d4e7e9ae2
BASE_HDR=57851439e7c6f5e35aca7986e148e42b
BASE_ARCH=c3c101fdcc07cf803c4b70551c94343a
mkdir -p "$SCR"

echo "[$FMT-k1] board=k1 VLEN256 load=$(cat /proc/loadavg)"
echo "=== [0] verify shared source at baseline ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); H0=$(md5sum "$HDR"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN=$G0 HDR=$H0 ARCH=$A0"
[ "$G0" = "$BASE_GEN" ] && [ "$H0" = "$BASE_HDR" ] && [ "$A0" = "$BASE_ARCH" ] || { echo "  *** NOT baseline -- ABORT"; exit 10; }

echo "=== [1] snapshot .ORIG (scratch, not tree) ==="
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$HDR" "$SCR/repack.h.ORIG"; cp "$ARCH" "$SCR/arch_repack.cpp.ORIG"

echo "=== [2] seed dedicated build dir (cp -a stock) ==="
rm -rf "$BUILD"; cp -a "$STOCK" "$BUILD"

python3 - "$SRCDB" "$BUILD" "$SCR" <<'PY'
import json,sys
db,build,scr=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(db))
def cmd_for(suffix):
    h=[e for e in d if e['file'].endswith(suffix)]
    assert len(h)==1, f"{suffix}: {len(h)} hits"
    return h[0]['command']
open(scr+'/cc_gen.sh','w').write(f"cd {build}/ggml/src\n"+cmd_for('ggml-cpu/repack.cpp')+"\n")
open(scr+'/cc_arch.sh','w').write(f"cd {build}/ggml/src\n"+cmd_for('arch/riscv/repack.cpp')+"\n")
print("  cc_gen.sh + cc_arch.sh written")
PY

build_relink(){ # $1 = label
  local lab="$1"
  bash "$SCR/cc_gen.sh"  >"$SCR/cc_gen_$lab.log"  2>&1 || { echo "  cc_gen $lab FAIL"; tail -25 "$SCR/cc_gen_$lab.log"; exit 20; }
  bash "$SCR/cc_arch.sh" >"$SCR/cc_arch_$lab.log" 2>&1 || { echo "  cc_arch $lab FAIL"; tail -25 "$SCR/cc_arch_$lab.log"; exit 21; }
  ( cd "$BUILD/ggml/src" && bash CMakeFiles/ggml-cpu.dir/link.txt ) >"$SCR/link_$lab.log" 2>&1 || { echo "  link $lab FAIL"; tail -20 "$SCR/link_$lab.log"; exit 22; }
  cp -a "$LIVE" "$SCR/libggml-cpu.so.$lab"
  echo "  $lab md5=$(md5sum "$SCR/libggml-cpu.so.$lab"|awk '{print $1}')"
}

echo "=== [3] OFF build (pristine source -> $FMT block-dot) ==="
build_relink OFF

echo "=== [4] apply $FMT net-new patch + copy vl=16 incs ==="
cp "$BASE/$GEMM_INC" "$T/arch/riscv/"
cp "$BASE/$GEVM_INC" "$T/arch/riscv/"
python3 "$PATCH" || { echo "  PATCH FAIL"; exit 30; }

echo "=== [5] ON build (patched -> our vl=16 repack) ==="
build_relink ON
[ "$(md5sum "$SCR/libggml-cpu.so.ON"|awk '{print $1}')" != "$(md5sum "$SCR/libggml-cpu.so.OFF"|awk '{print $1}')" ] \
  && echo "  ON != OFF (variants differ, good)" || { echo "  *** ON == OFF (patch had no effect!)"; exit 31; }

echo "=== [6] SEAL: $FMT symbols + objdump vl=16 (ON) ==="
echo "  ON weft_emitc $FMT syms: $(nm -C "$SCR/libggml-cpu.so.ON"|grep -c "weft_emitc_ggml_.*$FMT" || true)"
echo "  OFF weft_emitc $FMT syms: $(nm -C "$SCR/libggml-cpu.so.OFF"|grep -c "weft_emitc_ggml_.*$FMT" || true) (expect 0)"
echo "  OFF stock $FMT repack syms: $(nm -C "$SCR/libggml-cpu.so.OFF"|grep -iE "${FMT}.*(gemm|gemv)|(gemm|gemv).*${FMT}"|grep -civ generic || true)"
for SYM in $(nm -C "$SCR/libggml-cpu.so.ON"|grep -oE "weft_emitc_ggml_[a-z_0-9]*$FMT[a-z_0-9]*"|sort -u); do
  objdump -d --disassemble="$SYM" "$SCR/libggml-cpu.so.ON" 2>/dev/null > "$SCR/objdump_$SYM.txt"
  echo "  $SYM vset forms:"
  grep -oE 'vset[i]*vli[^#]*' "$SCR/objdump_$SYM.txt" | grep -oE 'e(8|16|32),(mf2|mf4|m1|m2|m4|m8)' | sort | uniq -c | sed 's/^/    /'
done

echo "=== [7] RESTORE shared source + verify baseline md5 ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_repack.cpp.ORIG" "$ARCH"
rm -f "$T/arch/riscv/$GEMM_INC" "$T/arch/riscv/$GEVM_INC"
G1=$(md5sum "$GEN"|awk '{print $1}'); H1=$(md5sum "$HDR"|awk '{print $1}'); A1=$(md5sum "$ARCH"|awk '{print $1}')
echo "  post-restore GEN=$G1 HDR=$H1 ARCH=$A1"
[ "$G1" = "$BASE_GEN" ] && [ "$H1" = "$BASE_HDR" ] && [ "$A1" = "$BASE_ARCH" ] || { echo "  *** RESTORE MISMATCH"; exit 40; }
echo "  RESTORE OK (all 3 == baseline; stray $FMT incs: $(ls "$T/arch/riscv/"weft_emitted_*$FMT* 2>/dev/null|wc -l))"

echo "=== [8] leave live lib at OFF (block-dot pristine) ==="
cp -f "$SCR/libggml-cpu.so.OFF" "$LIVE"
echo "=== [$FMT-k1 build DONE] ==="
md5sum "$SCR/libggml-cpu.so.OFF" "$SCR/libggml-cpu.so.ON"
