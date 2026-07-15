#!/usr/bin/env bash
# sweep.sh -- measurement sweep using pre-built ./bench. cold(pool=256)+hot(pool=0), 2 cold seeds.
set -u
RDIR="${RDIR:?}"; CORES="${CORES:?}"; ROUNDS="${ROUNDS:-20}"; N=4096
cd "$RDIR"; LOG="$RDIR/sweep.log"; : > "$LOG"
FMTS="iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs iq4_xs tq2_0 tq1_0"
say(){ echo "$@" | tee -a "$LOG"; }
say "# loadavg_begin=$(cat /proc/loadavg)"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in $CORES; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=-1; BESTIDLE=-1
for c in $CORES; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); say "# core$c idle_pct=$pct";
  if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
if [ $BESTIDLE -lt 70 ]; then say "# LOAD_GATE_FAIL best=$BESTC idle=$BESTIDLE ABORT"; exit 9; fi
CORE=$BESTC
say "# LOAD_GATE_OK core=$CORE idle=${BESTIDLE}% freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)"
run(){ taskset -c $CORE ./bench "$@" 2>/dev/null; }
gf(){ echo "$1" | sed -nE "s/.*$2=([-0-9.e]+).*/\1/p"; }
gfp(){ echo "$1" | sed -nE 's/.*fingerprint=(0x[0-9a-f]+).*/\1/p'; }
say "# fmt | mode | seed | ours_ns | fact_ns | RATIO_f/o | ours_iqr% | fact_iqr% | ours_GBs | fact_GBs | fp"
for f in $FMTS; do
  for spec in "hot 0 0xBEEF01" "cold 256 0xBEEF01" "cold 256 0xC0FFEE"; do
    set -- $spec; mode=$1; pool=$2; seed=$3
    run $f ours $N $ROUNDS $pool 0 $seed >/dev/null
    LO=$(run $f ours    $N $ROUNDS $pool 0 $seed)
    LF=$(run $f factory $N $ROUNDS $pool 0 $seed)
    mo=$(gf "$LO" ns_per_block_median); mf=$(gf "$LF" ns_per_block_median)
    io=$(gf "$LO" ns_per_block_iqr);    iff=$(gf "$LF" ns_per_block_iqr)
    go=$(gf "$LO" achieved_GBs);        gff=$(gf "$LF" achieved_GBs)
    fpo=$(gfp "$LO"); fpf=$(gfp "$LF")
    r=$(awk -v a="$mf" -v b="$mo" 'BEGIN{if(b>0)printf "%.4f",a/b; else print "NA"}')
    oip=$(awk -v i="$io" -v m="$mo" 'BEGIN{if(m>0)printf "%.2f",100*i/m; else print "NA"}')
    fip=$(awk -v i="$iff" -v m="$mf" 'BEGIN{if(m>0)printf "%.2f",100*i/m; else print "NA"}')
    fpm=$([ "$fpo" = "$fpf" ] && echo MATCH || echo DIVERGE)
    say "RESULT $f | $mode | $seed | $mo | $mf | ${r}x | $oip | $fip | $go | $gff | $fpm"
  done
done
say "# loadavg_end=$(cat /proc/loadavg)"
say "=== SWEEP_DONE ==="
