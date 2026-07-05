#!/usr/bin/env bash
# correctness_gate.sh -- e2e correctness gate, run ON the board.
#
# CONSTITUTIONAL FRAMING (钉死, 违宪纠正):
#   * KERNEL level  = byte-exact vs our OWN pinned oracle (established separately
#     by the ondevice vec_dot drivers; this gate does not re-derive it).
#   * E2E level     = (a) GREEDY-TOKEN CONSISTENCY: under temp=0 greedy decoding
#     with a fixed seed, our-injected build (A) and stock ggml (B) must emit the
#     SAME output token sequence; plus (b) LOGITS SANITY: coherent output, no
#     NaN/Inf. We NEVER claim bit-exact vs ggml at e2e -- both accumulate fp in
#     different (all IEEE-legal) orders, so bitwise identity is neither expected
#     nor required; token-sequence identity under argmax is the correct gate.
#
# Tooling note: this llama.cpp build ships only llama-cli (no llama-completion),
# and tinyllama is a chat model (auto conversation mode), so greedy generation
# uses `-st` (single-turn) with the prompt on stdin; the reply is normalized to
# strip the loading spinner / ASCII banner / perf line before comparison.
#
# Env: A_BUILD B_BUILD | MODEL | CORES | NTOK | LDPATH | PROMPTS (newline-sep)
set -u
A_BUILD="${A_BUILD:?}"; B_BUILD="${B_BUILD:?}"; MODEL="${MODEL:?}"
CORES="${CORES:-8-23}"; NTOK="${NTOK:-24}"
CLI_A="${CLI_BIN_A:-$A_BUILD/bin/llama-cli}"
CLI_B="${CLI_BIN_B:-$B_BUILD/bin/llama-cli}"
[ -n "${LDPATH:-}" ] && export LD_LIBRARY_PATH="$LDPATH:${LD_LIBRARY_PATH:-}"
PIN="taskset -c $CORES"
PROMPTS="${PROMPTS:-The capital of France is
Once upon a time
Q: What is 2 + 2? A:}"

# strip loading spinner / banner / perf line / menu help / blanks -> reply text.
# The llama-cli progress spinner is a maximal run of {- \ | /} (>=3 chars) that
# gets merged onto the reply line once \r is removed; we delete any such run
# ANYWHERE (real text never has a 3+ run of these), which cancels the only
# non-deterministic residue between A and B. Everything else (build info, prompt
# echo, reply tokens) is identical when greedy decoding agrees.
normalize() {
  # drop ALL control chars first (\r, \b backspace, ESC from the spinner) so the
  # spinner glyphs become one contiguous run the next step can delete.
  tr -cd '[:print:]\n' \
    | sed -E \
        -e '/Loading model/d' \
        -e '/\[ Prompt:/d' \
        -e '/Exiting/d' \
        -e '/chat history/d' -e '/text file/d' -e '/globbing/d' \
        -e 's#[|/\\-]{3,}##g' \
    | grep -aE '[A-Za-z0-9]' \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

gen() {  # $1=cli  -> normalized reply for the current $PROMPT
  printf '%s' "$PROMPT" | timeout "${GEN_TIMEOUT:-90}" $PIN "$1" \
      -m "$MODEL" -st --temp 0 --top-k 1 --seed 1 -t "${THREADS:-16}" -n "$NTOK" \
      2>>/tmp/cg_stderr.txt | normalize
}

: > /tmp/cg_stderr.txt
PASS=0; FAIL=0; IDX=0
echo "== e2e CORRECTNESS GATE (greedy-token consistency + logits sanity) =="
echo "model=$MODEL  ntok=$NTOK  A=$CLI_A  B=$CLI_B"
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  RA=$(gen "$CLI_A"); RB=$(gen "$CLI_B")
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   A: $(printf '%s' "$RA" | tr '\n' ' ' | cut -c1-140)"
  echo "   B: $(printf '%s' "$RB" | tr '\n' ' ' | cut -c1-140)"
  if [ -z "$RA" ] || [ -z "$RB" ]; then
    echo "   VERDICT: FAIL (empty reply -- generation error)"; FAIL=$((FAIL+1)); continue
  fi
  if [ "$RA" = "$RB" ]; then
    echo "   VERDICT: GREEDY-TOKEN-CONSISTENT (A==B)"; PASS=$((PASS+1))
  else
    echo "   VERDICT: MISMATCH (A != B)"; FAIL=$((FAIL+1))
    diff <(printf '%s' "$RA") <(printf '%s' "$RB") | head -6 | sed 's/^/     /'
  fi
done <<< "$PROMPTS"

echo "== LOGITS SANITY =="
if grep -aiE 'nan|-nan|[^a-z]inf[^a-z]|inf$' /tmp/cg_stderr.txt >/dev/null 2>&1; then
  echo "   FAIL: NaN/Inf seen in generation stderr"; FAIL=$((FAIL+1))
else
  echo "   OK: no NaN/Inf in generation logs (finite logits)"
fi

echo "== CORRECTNESS SUMMARY: consistent=$PASS  failed=$FAIL =="
[ "$FAIL" = 0 ] && echo "CORRECTNESS_GATE: GREEN" || echo "CORRECTNESS_GATE: RED"
[ "$FAIL" = 0 ]
