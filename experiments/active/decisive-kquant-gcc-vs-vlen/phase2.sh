#!/usr/bin/env bash
# [decisive-kquant-gcc-vs-vlen] phase-2 SEQUENTIAL: gcc-account anchor A/B, then
# correctness gate. One job, no overlap (single LIVE swapper at a time).
set -u
cd /tmp/dkgv
echo "###### PHASE2 start $(date -u) ######"
echo "=== gcc account: q4kON.gcc / q4kOFF.gcc (in-session anchor) ==="
NUMER=q4kON.gcc DENOM=q4kOFF.gcc TAG=8b_gcc PP=128 TG=16 REPS=2 PASSES=1 bash measure_ab.sh
echo "=== gcc account parse ==="
python3 parse_ab.py /tmp/dkgv/ab_8b_gcc
echo
echo "=== correctness: q4kON.clangrepack repack vs q4kOFF.gcc block-dot ==="
NUMER=q4kON.clangrepack DENOM=q4kOFF.gcc NTOK=24 bash correctness_ab.sh
echo "###### PHASE2 done $(date -u) ######"
