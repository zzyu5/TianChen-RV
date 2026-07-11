#!/usr/bin/env bash
# [G5 M1] Engage proof (banner FIRES with ON / ABSENT with OFF) + e2e correctness
# gate (greedy-token A==B + logits sanity) via same-tree physical .so swap.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
SCR=/tmp/g5_q8
MODEL=$ATREE/models/tinyllama-q8_0.gguf
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"; NTOK="${NTOK:-24}"
PIN="taskset -c $CORES"
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$SCR/libggml-cpu.so.q8$1" "$LIVE"; }

echo "=== [ENGAGE] ON: expect GEVM + GEMM banners FIRE ==="
swap ON
$PIN "$BIN/llama-bench" -m "$MODEL" -p 32 -n 8 -r 1 -t "$THREADS" -o json >/dev/null 2>/tmp/g5_engage_on.err || true
echo "  GEVM banner (ON, expect>=1): $(grep -c 'TCRV G5-M1 GEVM(q8_0_16x1 VLEN128' /tmp/g5_engage_on.err || true)"
echo "  GEMM banner (ON, expect>=1): $(grep -c 'TCRV G5-M1 GEMM(q8_0_16x1 VLEN128' /tmp/g5_engage_on.err || true)"
grep 'TCRV G5-M1' /tmp/g5_engage_on.err | head -3 | sed 's/^/    /'
echo "=== [ENGAGE] OFF: expect banners ABSENT (block-dot) ==="
swap OFF
$PIN "$BIN/llama-bench" -m "$MODEL" -p 32 -n 8 -r 1 -t "$THREADS" -o json >/dev/null 2>/tmp/g5_engage_off.err || true
echo "  GEVM+GEMM banner (OFF, expect 0): $(grep -c 'TCRV G5-M1' /tmp/g5_engage_off.err || true)"

echo ""
echo "=== [CORRECTNESS] greedy-token A(ON)==B(OFF) + logits sanity ==="
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:"
normalize() {
  tr -cd '[:print:]\n' \
    | sed -E -e '/Loading model/d' -e '/\[ Prompt:/d' -e '/Exiting/d' \
        -e '/chat history/d' -e '/text file/d' -e '/globbing/d' -e 's#[|/\\-]{3,}##g' \
    | grep -aE '[A-Za-z0-9]' | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}
: > /tmp/g5_cg_stderr.txt
gen(){ printf '%s' "$PROMPT" | timeout "${GEN_TIMEOUT:-120}" $PIN "$BIN/llama-cli" \
    -m "$MODEL" -st --temp 0 --top-k 1 --seed 1 -t "$THREADS" -n "$NTOK" \
    2>>/tmp/g5_cg_stderr.txt | normalize; }
PASS=0; FAIL=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  swap ON;  RA=$(gen)
  swap OFF; RB=$(gen)
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A(ON) : $(printf '%s' "$RA" | tr '\n' ' ' | cut -c1-140)"
  echo "   B(OFF): $(printf '%s' "$RB" | tr '\n' ' ' | cut -c1-140)"
  if [ -z "$RA" ] || [ -z "$RB" ]; then echo "   VERDICT: FAIL (empty reply)"; FAIL=$((FAIL+1)); continue; fi
  if [ "$RA" = "$RB" ]; then echo "   VERDICT: GREEDY-TOKEN-CONSISTENT (A==B)"; PASS=$((PASS+1))
  else echo "   VERDICT: MISMATCH (A != B)"; FAIL=$((FAIL+1)); diff <(printf '%s' "$RA") <(printf '%s' "$RB")|head -6|sed 's/^/     /'; fi
done <<< "$PROMPTS"
echo "== LOGITS SANITY =="
if grep -aiE 'nan|-nan|[^a-z]inf[^a-z]|inf$' /tmp/g5_cg_stderr.txt >/dev/null 2>&1; then
  echo "   FAIL: NaN/Inf in generation stderr"; FAIL=$((FAIL+1)); else echo "   OK: no NaN/Inf (finite logits)"; fi
echo "== CORRECTNESS SUMMARY: consistent=$PASS failed=$FAIL =="
[ "$FAIL" = 0 ] && echo "CORRECTNESS_GATE: GREEN" || echo "CORRECTNESS_GATE: RED"
swap OFF  # leave live at pristine OFF
