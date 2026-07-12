#!/usr/bin/env bash
# [G5-M2 iq4_nl] Autonomous board driver: wait for IQ4_NL quantize -> build+seal (2-file
# wiring, OFF/ON variants, objdump vl=8 seal, restore byte-exact) -> correctness HARD GATE
# (greedy A==B 5/5 + banner engage; STOP before perf if not GREEN) -> perf phase-split ->
# analyze. Captures each stage to its own log under /tmp/g5_iq4nl.
set -u
SCR=/tmp/g5_iq4nl
cd "$SCR"

echo "########## [0] wait for IQ4_NL quantize ##########"
while pgrep -f llama-quantize >/dev/null 2>&1; do sleep 15; done
M=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-IQ4_NL.gguf
if [ ! -s "$M" ]; then echo "MODEL MISSING -> ABORT"; exit 2; fi
echo "model=$M size=$(ls -la "$M"|awk '{print $5}') sha256=$(sha256sum "$M"|cut -c1-16)"
tail -3 /tmp/g5_iq4nl_quant.log

echo "########## [1] build + seal ##########"
bash "$SCR/g5_m2_iq4_nl_build_seal.sh" > "$SCR/build_seal.log" 2>&1
BS=$?
tail -40 "$SCR/build_seal.log"
if [ $BS -ne 0 ]; then echo ">>> BUILD_SEAL FAILED (rc=$BS) -> ABORT"; exit 10; fi
grep -q "SOURCE RESTORED byte-exact" "$SCR/build_seal.log" || { echo ">>> RESTORE NOT VERIFIED -> ABORT"; exit 11; }
grep -q "ON build OK" "$SCR/build_seal.log" || { echo ">>> ON BUILD MISSING -> ABORT"; exit 12; }

echo "########## [2] correctness HARD GATE ##########"
bash "$SCR/g5_m2_iq4_nl_correctness.sh" > "$SCR/correctness.log" 2>&1
cat "$SCR/correctness.log"
if ! grep -q "CORRECTNESS_GATE: GREEN" "$SCR/correctness.log"; then
  echo ">>> CORRECTNESS NOT GREEN -> STOP (no perf, per MIRAGE discipline)"; exit 20; fi
echo ">>> CORRECTNESS GREEN -> proceed to perf"

echo "########## [3] perf phase-split ##########"
bash "$SCR/g5_m2_iq4_nl_phase_split.sh" > "$SCR/phase_split.log" 2>&1
echo "-- phase_split tail --"; tail -20 "$SCR/phase_split.log"

echo "########## [4] analyze ##########"
python3 "$SCR/analyze_phase_split.py" "$SCR/phase_split.log" > "$SCR/analyze.log" 2>&1
cat "$SCR/analyze.log"

echo "########## DONE ##########"
echo "logs: build_seal.log correctness.log phase_split.log analyze.log"
