#!/usr/bin/env bash
# [G3 T4b seal-fix] Full prefill + decode bench with kernel-isolation.
# Three .so, ALL run from the tcrv bin (same llama-bench binary) so the ONLY diff is the .so:
#   A        = .A-q4kON     (tcrv tree, q4_K gate ON  -> OUR emitted vl=8 repack GEMM/GEVM)
#   Bq4kOFF  = .B-q4kOFF    (tcrv tree, q4_K gate OFF -> ggml block-dot)  [kernel-isolated opponent]
#   (Bstock upstream handled separately by t4b_bench.sh)
# Prefill pp {128,256,512} and decode tg 32, reps=$REPS. JSON per variant.
set -u
. /opt/tcrv-toolchains/env.sh 2>/dev/null
REPS="${1:-10}"
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
TCRV=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
export LD_LIBRARY_PATH="$TCRV:${LD_LIBRARY_PATH:-}"
run() {
  local SO="$1" TAG="$2" PP="$3" TG="$4"
  cp -f "$TCRV/libggml-cpu.so.0.15.1.$SO" "$TCRV/libggml-cpu.so.0.15.1"
  local OUT=/tmp/tf_${TAG}.json ERR=/tmp/tf_${TAG}.err
  taskset -c 8-15 "$TCRV/llama-bench" -m "$MODEL" -p "$PP" -n "$TG" -t 8 -r "$REPS" -o json > "$OUT" 2> "$ERR"
  echo "== $TAG (so=$SO) exit=$? =="
  python3 - "$OUT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for e in d:
    lbl=("pp" if e.get("n_prompt",0)>0 and e.get("n_gen",0)==0 else "tg")
    n=e.get("n_prompt") or e.get("n_gen")
    print(f"  {lbl}{n}: avg_ts={e['avg_ts']:.3f} +/- {e.get('stddev_ts',0):.3f} tok/s  (samples={e.get('samples_ns') and len(e['samples_ns'])})")
PY
}
echo "### A = OUR vl=8 kernel"
run A-q4kON A "128,256,512" 32
echo "### Bq4kOFF = same tcrv tree, gate OFF (block-dot) -- KERNEL-ISOLATED"
run B-q4kOFF Bq4kOFF "128,256,512" 32
echo "### DONE"
