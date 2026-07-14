#!/usr/bin/env bash
# [G7-L2 e2e-smoke] Build 3 .so variants of ggml-cpu from the SAME tcrv A-tree,
# gcc-15.2.0 symmetric, ONLY the q5_0 GEVM leaf emitter differs:
#   q5OFF  = pristine (NO q5_0 scaffold)  -> stock ggml q5_0 block-dot
#   q5OLD  = 12-piece scaffold + OLD gevm .inc (per-lane-expand: vor/vncvt/vsrl/vsll)
#   q5NEW  = 12-piece scaffold + REDESIGN-B gevm .inc (vlm_v_b16 + vmnand + vsub_mu)
# GEMM .inc identical for OLD/NEW (sealed leaf untouched). Objdump seal proves the
# deployed GEVM symbol carries the REDESIGN-B signature (deployed==proven routing).
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
HDR=$ATREE/ggml/src/ggml-cpu/repack.h
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
ARCHDIR=$ATREE/ggml/src/ggml-cpu/arch/riscv
SCR=/tmp/g5_q5
LIVE=$BIN/libggml-cpu.so.0.15.1
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_HDR=57851439e7c6f5e35aca7986e148e42b
BASE_ARCH=99131cf791e30348b588423b2388e0b8
GEVM_SYM=tcrv_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0
GEMM_SYM=tcrv_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}
CC_VER=$(gcc --version | head -1)
echo "[3variant] board=rvv load=$(cat /proc/loadavg) deploy_cc=$CC_VER"

echo "=== [0] baseline verify + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); H0=$(md5sum "$HDR"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN=$G0 HDR=$H0 ARCH=$A0"
[ "$G0" = "$BASE_GEN" ] && [ "$H0" = "$BASE_HDR" ] && [ "$A0" = "$BASE_ARCH" ] || { echo "*** NOT baseline -- ABORT"; exit 10; }
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$HDR" "$SCR/repack.h.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  .inc md5s: gemm=$(md5sum "$SCR/tcrv_emitted_gemm_q5_0.inc"|awk '{print $1}')"
echo "             gevm_OLD=$(md5sum "$SCR/tcrv_emitted_gevm_q5_0_OLD.inc"|awk '{print $1}')"
echo "             gevm_NEW=$(md5sum "$SCR/tcrv_emitted_gevm_q5_0_NEW.inc"|awk '{print $1}')"

build(){ touch "$GEN" "$HDR" "$ARCH"; cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >"$SCR/build_$1.log" 2>&1; }

echo "=== [1] build OFF (pristine, no scaffold) ==="
if build off; then echo "  OFF build OK"; else echo "  OFF build FAIL"; tail -40 "$SCR/build_off.log"; exit 20; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q5OFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.q5OFF"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.q5OFF" 2>/dev/null|grep -cE "tcrv_emitc_ggml_(vec_dot|gemm)_q5_0" || true)
echo "  OFF md5=$OFFMD5 q5_sym(expect 0)=$OFFSYM"

echo "=== [2] apply 12-piece scaffold + OLD gevm .inc -> build OLD ==="
cp -f "$SCR/tcrv_emitted_gemm_q5_0.inc" "$ARCHDIR/tcrv_emitted_gemm_q5_0.inc"
cp -f "$SCR/tcrv_emitted_gevm_q5_0_OLD.inc" "$ARCHDIR/tcrv_emitted_gevm_q5_0.inc"
python3 "$SCR/deploy_patch_q5_0_emitted.py" || { echo "PATCH FAILED"; exit 11; }
if build old; then echo "  OLD build OK"; else echo "  OLD build FAIL"; tail -80 "$SCR/build_old.log"; exit 21; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q5OLD"
OLDMD5=$(md5sum "$SCR/libggml-cpu.so.q5OLD"|awk '{print $1}')
OLDSYM=$(nm -C "$SCR/libggml-cpu.so.q5OLD" 2>/dev/null|grep -cE "tcrv_emitc_ggml_(vec_dot|gemm)_q5_0" || true)
echo "  OLD md5=$OLDMD5 q5_sym(expect 2)=$OLDSYM"
objdump -d --disassemble="$GEVM_SYM" "$SCR/libggml-cpu.so.q5OLD" 2>/dev/null > "$SCR/objdump_gevm_OLD.txt"
echo "  OLD GEVM objdump: total-insn=$(grep -cE '^\s+[0-9a-f]+:' "$SCR/objdump_gevm_OLD.txt")"

echo "=== [3] swap in REDESIGN-B (NEW) gevm .inc -> build NEW ==="
cp -f "$SCR/tcrv_emitted_gevm_q5_0_NEW.inc" "$ARCHDIR/tcrv_emitted_gevm_q5_0.inc"
if build new; then echo "  NEW build OK"; else echo "  NEW build FAIL"; tail -80 "$SCR/build_new.log"; exit 22; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q5NEW"
NEWMD5=$(md5sum "$SCR/libggml-cpu.so.q5NEW"|awk '{print $1}')
NEWSYM=$(nm -C "$SCR/libggml-cpu.so.q5NEW" 2>/dev/null|grep -cE "tcrv_emitc_ggml_(vec_dot|gemm)_q5_0" || true)
echo "  NEW md5=$NEWMD5 q5_sym(expect 2)=$NEWSYM"
[ "$NEWMD5" != "$OLDMD5" ] && echo "  NEW != OLD (variants differ, good)" || echo "  *** NEW == OLD -- gevm .inc swap had no effect!"
objdump -d --disassemble="$GEVM_SYM" "$SCR/libggml-cpu.so.q5NEW" 2>/dev/null > "$SCR/objdump_gevm_NEW.txt"
echo "  NEW GEVM objdump: total-insn=$(grep -cE '^\s+[0-9a-f]+:' "$SCR/objdump_gevm_NEW.txt")"

echo "=== [4] ROUTING SEAL: vsetivli widths + REDESIGN-B vs OLD signature in deployed GEVM ==="
for tag in OLD NEW; do
  f="$SCR/objdump_gevm_$tag.txt"
  N8=$(grep -cE 'vsetivli[^,]*,[[:space:]]*8,' "$f" || true)
  N16=$(grep -cE 'vsetivli[^,]*,[[:space:]]*16,' "$f" || true)
  N64=$(grep -cE 'vsetivli[^,]*,[[:space:]]*64,' "$f" || true)
  VLM=$(grep -cE '\bvlm\.v\b' "$f" || true)
  VMNAND=$(grep -cE 'vmnand' "$f" || true)
  VOR=$(grep -cE '\bvor\.vv\b' "$f" || true)
  VNCVT=$(grep -cE 'vncvt' "$f" || true)
  VSUB=$(grep -cE '\bvsub\b|vsub\.vx' "$f" || true)
  echo "  [$tag] vsetivli(8)=$N8 (16)=$N16[MUST 0] (64)=$N64[MUST 0] | vlm.v=$VLM vmnand=$VMNAND vsub=$VSUB | vor.vv=$VOR vncvt=$VNCVT"
done
echo "  (NEW=REDESIGN-B expects vlm.v>0 & vmnand>0 & vor.vv==0/vncvt==0; OLD expects vor.vv>0 & vncvt>0 & vlm.v(qh-decode)~ )"

echo "=== [5] restore source + pristine rebuild ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
rm -f "$ARCHDIR/tcrv_emitted_gemm_q5_0.inc" "$ARCHDIR/tcrv_emitted_gevm_q5_0.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); HR=$(md5sum "$HDR"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
[ "$GR" = "$BASE_GEN" ] && [ "$HR" = "$BASE_HDR" ] && [ "$AR" = "$BASE_ARCH" ] && echo "  SOURCE RESTORED byte-exact" || { echo "*** restore mismatch GEN=$GR HDR=$HR ARCH=$AR"; exit 30; }
if build restore; then echo "  pristine rebuild OK"; else echo "  restore build FAIL"; exit 31; fi
LIVEMD5=$(md5sum "$LIVE"|awk '{print $1}'); LIVESYM=$(nm -C "$LIVE" 2>/dev/null|grep -cE "tcrv_emitc_ggml_(vec_dot|gemm)_q5_0" || true)
echo "  live md5=$LIVEMD5 q5_sym(expect 0)=$LIVESYM"
echo "=== [3variant DONE] ==="
md5sum "$SCR/libggml-cpu.so.q5OFF" "$SCR/libggml-cpu.so.q5OLD" "$SCR/libggml-cpu.so.q5NEW"
