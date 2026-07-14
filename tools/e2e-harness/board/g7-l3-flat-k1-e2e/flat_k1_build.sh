#!/usr/bin/env bash
# [G7-L3 FLAT@k1 e2e] Build REPACK (stock as-shipped FLAT repack 16x1 @VLEN256) +
# VECDOT (repack dispatch disabled -> generic block-dot) libggml-cpu variants from the
# SAME k1 tcrv tree (clang-18 SYMMETRIC; only the ONE dispatch return differs).
# Private repack.cpp copy compiled separately -> shared source tree NEVER edited.
# Dedicated build dir /data/build-k1-flat (seeded from k1build-stock; main build/ untouched).
# Usage: FORMAT=q4_0|q8_0 flat_k1_build.sh
set -uo pipefail
FMT="${FORMAT:?set FORMAT=q4_0 or q8_0}"
T=/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu
GEN=$T/repack.cpp
STOCK=/data/k1build-stock
BUILD=/data/build-k1-flat
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SRCDB=/data/k1build/compile_commands.json
SCR=/tmp/flat_$FMT
BASE_GEN=3cac40aa55aece1f69e3d08d4e7e9ae2
STOCK_LIB_MD5=871169a0123139692177468b3c8578be
case "$FMT" in
  q4_0) DIS='return &q4_0_16x1_q8_0;'; REPACK_SYM=ggml_gemm_q4_0_16x1_q8_0 ;;
  q8_0) DIS='return &q8_0_16x1_q8_0;'; REPACK_SYM=ggml_gemm_q8_0_16x1_q8_0 ;;
  *) echo "unknown FORMAT $FMT"; exit 2 ;;
esac
mkdir -p "$SCR"

echo "[flat-k1 $FMT] board=k1 VLEN256 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify shared source (NEVER edited) ==="
G0=$(md5sum "$GEN"|awk '{print $1}')
echo "  GEN=$G0 (base $BASE_GEN)"
[ "$G0" = "$BASE_GEN" ] || { echo "  *** GEN not baseline -- ABORT"; exit 10; }

echo "=== [1] seed dedicated build dir (cp -a stock) ==="
rm -rf "$BUILD"; cp -a "$STOCK" "$BUILD"
# REPACK lib = the seeded live lib == shipped stock
cp -a "$LIVE" "$SCR/libggml-cpu.so.REPACK"
RMD5=$(md5sum "$SCR/libggml-cpu.so.REPACK"|awk '{print $1}')
echo "  REPACK md5=$RMD5 (stock $STOCK_LIB_MD5) $([ "$RMD5" = "$STOCK_LIB_MD5" ] && echo MATCH || echo '*** MISMATCH')"

echo "=== [2] private repack.cpp copy w/ $FMT repack DISABLED (-> block-dot) ==="
cp "$GEN" "$SCR/repack_vecdot.cpp"
# disable ONLY this format's repack return; unique substring
perl -0pi -e "s/\Q$DIS\E/break;/g" "$SCR/repack_vecdot.cpp"
NCH=$(grep -c "break;" "$SCR/repack_vecdot.cpp")
grep -q "$DIS" "$SCR/repack_vecdot.cpp" && { echo "  *** disable FAILED (return still present)"; exit 11; }
echo "  disabled '$DIS' -> break; (verified absent in private copy)"

echo "=== [3] extract exact compile cmd for repack.cpp, recompile private copy -> build tree .o ==="
python3 - "$SRCDB" "$BUILD" "$SCR" <<'PY'
import json,sys
db,build,scr=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(db))
hits=[e for e in d if e['file'].endswith('ggml-cpu/repack.cpp')]
assert len(hits)==1, f"repack.cpp: {len(hits)} hits"
e=hits[0]; cmd=e['command']
# swap the input file token to our private copy; keep -o pointing at build-tree object
import re
priv=scr+'/repack_vecdot.cpp'
# original input is the shared source absolute path at end of command
cmd2=cmd.replace(e['file'], priv)
assert priv in cmd2, "input swap failed"
open(scr+'/cc_vecdot.sh','w').write(f"cd {build}/ggml/src\n"+cmd2+"\n")
print("  cc_vecdot.sh written; input ->", priv)
PY
bash "$SCR/cc_vecdot.sh" >"$SCR/cc_vecdot.log" 2>&1 || { echo "  cc FAIL"; tail -30 "$SCR/cc_vecdot.log"; exit 20; }
( cd "$BUILD/ggml/src" && bash CMakeFiles/ggml-cpu.dir/link.txt ) >"$SCR/link.log" 2>&1 || { echo "  link FAIL"; tail -20 "$SCR/link.log"; exit 21; }
cp -a "$LIVE" "$SCR/libggml-cpu.so.VECDOT"
VMD5=$(md5sum "$SCR/libggml-cpu.so.VECDOT"|awk '{print $1}')
echo "  VECDOT md5=$VMD5"
[ "$VMD5" != "$RMD5" ] && echo "  VECDOT != REPACK (variants differ, good)" || { echo "  *** VECDOT == REPACK (edit had no effect!)"; exit 22; }

echo "=== [4] SEAL: repack symbol still present in both (config toggle, behavioral engage) ==="
echo "  REPACK $REPACK_SYM count: $(nm -C "$SCR/libggml-cpu.so.REPACK"|grep -c "$REPACK_SYM" || true)"
echo "  VECDOT $REPACK_SYM count: $(nm -C "$SCR/libggml-cpu.so.VECDOT"|grep -c "$REPACK_SYM" || true) (present but dispatch-off)"
objdump -d --disassemble="$REPACK_SYM" "$SCR/libggml-cpu.so.REPACK" 2>/dev/null > "$SCR/objdump_${REPACK_SYM}.txt"
echo "  $REPACK_SYM vset forms (REPACK, VLEN256-native seal):"
grep -oE 'vset[i]*vli[^#]*' "$SCR/objdump_${REPACK_SYM}.txt" | grep -oE 'e(8|16|32),(mf2|mf4|m1|m2|m4|m8)' | sort | uniq -c | sed 's/^/    /'

echo "=== [5] leave live lib at REPACK (pristine stock) ==="
cp -f "$SCR/libggml-cpu.so.REPACK" "$LIVE"
echo "=== [flat-k1 $FMT build DONE] ==="
md5sum "$SCR/libggml-cpu.so.REPACK" "$SCR/libggml-cpu.so.VECDOT"
