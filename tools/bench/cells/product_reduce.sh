#!/usr/bin/env bash
# tools/bench/cells/product_reduce.sh <board> <mode> <fmt> <regime> — product_reduce 每格对拍/计时 harness。
#
# op = product_reduce（N3 gearbox 归约子原语·ours-vector 整数核 -> int32 归约·NO ggml 框架对手）。
# DUT = 我方板端编译的 emitted 向量整数核（nibble/offbin/codebook）。
# OPP/REF = 计算【同一】整数算术的 constructed scalar-ref（标量类档·codegen-lottery·对手法 §〇/§3.4）。
#   ★product_reduce 是 INTERNAL sub-primitive：ggml 无 standalone product_reduce 框架核
#     （T-CENSUS §一.F）⟹ 无部署 thunk 可探；公平参照 = scalar-ref。CHEAP TIER：大倍数是预期·
#     【非硬赢】（[NG-4] kernel-axis SANITY·非 e2e·非 perf-covered·便宜档禁称硬赢）。
#
# 权威 = .trellis/spec/measurement/哲学与目的地.md §3.2.4（住 tools/、写 experiments/）
#        + 《测试与收尾总令-开测篇》§〇.2（ISSUE-090 harness 契约）
#        + 对手法 §3.4（标量类档）+ 正确性门 [K-5]（整数 byte-exact ZERO-MODEL）。
#        补 K-actionable-queue item1 / ISSUE-099 gcc 车道清欠（3 格·clang-18 单世界对称重测）。
#
# ★契约（硬·同 dequantize_row.sh / scalar_vec_dot.sh / gemm_tile.sh）：
#   - 本 harness 由 ../bench 按声明接口调用：
#     `product_reduce.sh <board> <mode> <fmt> <regime>`。
#   - **harness 自身禁写任何【仓库侧】持久文件** —— 板端跑完把结果全部打到 stdout；
#     bench 解析 stdout，一切仓库侧持久写入经 runner 的 fail-closed 写入闸落三目的地。
#   - 板端 /tmp/$RDIR 下的 seal/log = 板端临时（可接受）；仓库侧【不 scp 回、不落任何文件】。
#   - 数据格只提供 driver/oracle；三个 DUT kernel 均由当前 clean HEAD 现场构造。
#   - ★单世界对称：我方向量核与 scalar-ref 对手【同一 clang-18·同板】编译（清 gcc 车道欠账）。
#
#   board: rvv | k1（两板皆 clang-18 出货链·§3.1 板册）
#   fmt  : q4_0_nibble | offset_binary_n3 | codebook_n3
#   regime: micro-fixed（固定 nb/hit 的内部子原语 microbenchmark）
#   mode : verify  = build + ZEROVEC objdump 探针 + ZERO-MODEL byte-exact + 2-arm 反空心 (NO TIMING)
#          sanity  = 预测量噪声自检 3 轮
#          measure = cold N=25 2-seed flush(>LLC)
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"; FMT="${3:-q4_0_nibble}"; REGIME="${4:-}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/../../.." && pwd)"
ASSETS="${PRODUCT_REDUCE_ASSET_ROOT:-$ROOT/experiments/active/k-product-reduce}"
EXPORTER="$ROOT/tools/bench/export_current_artifact.py"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/weft-current-product-reduce.XXXXXX")" || {
  echo "# HARNESS-VOID cannot create current-artifact staging directory"; exit 3;
}
trap 'rm -rf -- "$STAGE"' EXIT
mkdir -p "$STAGE/kernels"

NB_MEASURE=262144; HIT=6           # 262144 独立 16-lane 块 -> int32 归约（cold 流式 > LLC when flushed）
NB_VERIFY=262144                   # 大语料：全 256 nibble 组合 + qh 位选全覆盖
REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
RDIR=/tmp/bench_cells_product_reduce_${BOARD}

echo "# HARNESS product_reduce board=$BOARD mode=$MODE fmt=$FMT regime=$REGIME support_assets=$ASSETS"

# fmt 白名单（不认即 HARNESS-VOID exit 2·ssh 之前·零板可证）
case "$FMT" in
  q4_0_nibble|offset_binary_n3|codebook_n3) : ;;
  *) echo "# HARNESS-VOID bad fmt $FMT (product_reduce 族仅 q4_0_nibble|offset_binary_n3|codebook_n3)"; exit 2 ;;
esac
case "$MODE" in
  verify|sanity|measure) : ;;
  *) echo "# HARNESS-VOID bad mode $MODE (仅 verify|sanity|measure)"; exit 2 ;;
esac
if [ "$REGIME" != micro-fixed ]; then
  echo "# HARNESS-VOID unsupported regime ${REGIME:-<missing>} (product_reduce 当前只有 micro-fixed workload)"; exit 2
fi

if [ "$BOARD" = rvv ]; then
  CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
  GT=/opt/tcrv-toolchains/gcc-15.2.0
  MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
  CFLAGS="-fno-integrated-as -ffp-contract=on"
  LDEXTRA="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
  ENVSRC="source /opt/tcrv-toolchains/env.sh;"
  CORES="${BENCH_CORES:-8 9 10 11 12 13 14 15}"   # 0,1 = co-tenant vLLM, NEVER touched
elif [ "$BOARD" = k1 ]; then
  CC=clang-18                                       # Bianbu clang-18 = k1 出货链
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  CFLAGS="-ffp-contract=on"
  LDEXTRA=""
  ENVSRC=""
  CORES="${BENCH_CORES:-0 1 2 3 4 5 6 7}"
else
  echo "# HARNESS-VOID unsupported board $BOARD (product_reduce 族当前 rvv|k1)"; exit 2
fi

DRV="$ASSETS/preduce_driver.c"
KN="$STAGE/kernels/nibble.pr.c"
KO="$STAGE/kernels/offbin.pr.c"
KC="$STAGE/kernels/codebook.pr.c"
[ -f "$DRV" ] || { echo "# HARNESS-VOID missing support driver $DRV"; exit 3; }
[ -x "$EXPORTER" ] || { echo "# HARNESS-VOID missing current-artifact exporter $EXPORTER"; exit 3; }
for pair in q4_0_nibble:"$KN" offset_binary_n3:"$KO" codebook_n3:"$KC"; do
  export_fmt="${pair%%:*}"; export_path="${pair#*:}"
  python3 "$EXPORTER" product_reduce "$export_fmt" --board "$BOARD" \
    --output "$export_path" --require-clean \
    || { echo "# HARNESS-VOID current compiler failed to export product_reduce/$export_fmt"; exit 3; }
  [ -f "$export_path" ] || { echo "# HARNESS-VOID current compiler produced no kernel $export_path"; exit 3; }
done
echo "# CURRENT_ARTIFACT_SET three_kernels=PASS head=$(git -C "$ROOT" rev-parse HEAD)"

ssh "$BOARD" "mkdir -p $RDIR/kernels" || { echo "# HARNESS-VOID ssh mkdir failed"; exit 3; }
scp -q "$DRV" "$BOARD:$RDIR/preduce_driver.c"        || { echo "# HARNESS-VOID scp driver"; exit 3; }
scp -q "$KN"  "$BOARD:$RDIR/kernels/nibble.pr.c"     || { echo "# HARNESS-VOID scp nibble"; exit 3; }
scp -q "$KO"  "$BOARD:$RDIR/kernels/offbin.pr.c"     || { echo "# HARNESS-VOID scp offbin"; exit 3; }
scp -q "$KC"  "$BOARD:$RDIR/kernels/codebook.pr.c"   || { echo "# HARNESS-VOID scp codebook"; exit 3; }

ssh "$BOARD" "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL

  echo '# BUILD product_reduce fmt=$FMT board=$BOARD CC='\$($CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  echo '# board='\$(uname -srm)' isa='\$(awk -F': ' '/^isa/{print \$2; exit}' /proc/cpuinfo) | tee -a \$SEAL

  # ---- compile OURS vector kernels (single-world clang-18) + zero-vector objdump self-probe ----
  for pair in nibble:q4_0_nibble offbin:offset_binary_n3 codebook:codebook_n3; do
    f=\${pair%%:*}
    $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ kernels/\${f}.pr.c -c -o kp_\${f}.o 2>ccp_\${f}.err \
      || { echo VOID-BUILD kernel \$f; head -8 ccp_\${f}.err; exit 3; }
  done
  # ★zero-vector objdump 机检：真向量核 vec_core(vwmul|vwmacc|vwredsum|vrgather|vand|vsrl) 远 > vsetvl
  probe_kernel(){ local f=\$1 nm=\$2
    objdump -d kp_\${f}.o > k_\${f}_dis.txt 2>/dev/null
    local VS VR NV SZ
    VS=\$(grep -cE 'vsetvli|vsetivli' k_\${f}_dis.txt)
    VR=\$(grep -cE 'vwmul|vwmacc|vwredsum|vrgather|vand|vsrl|vwadd' k_\${f}_dis.txt)
    NV=\$(awk -F'\t' '\$3 ~ /^v/ && \$3 !~ /^vset/ {c++} END{print c+0}' k_\${f}_dis.txt)
    SZ=\$(wc -c<kp_\${f}.o)
    echo \"# OURS_\${nm} kernel=\${f} vsetvl=\$VS vec_core=\$VR non_vset_vector=\$NV size=\${SZ}B (真向量: vec_core>>vsetvl)\" | tee -a \$SEAL
  }
  probe_kernel nibble   q4_0_nibble
  probe_kernel offbin   offset_binary_n3
  probe_kernel codebook codebook_n3

  # ---- compile driver (scalar-ref oracle inside·noinline) + link ALL 3 kernels (driver refs all externs) ----
  $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c preduce_driver.c -c -o drv_p.o 2>ccdrv.err \
    || { echo VOID-BUILD driver; head -12 ccdrv.err; exit 4; }
  $CC drv_p.o kp_nibble.o kp_offbin.o kp_codebook.o $LDEXTRA -lstdc++ -lm -o preduce 2>ldp.err \
    || { echo VOID-BUILD link; head -20 ldp.err; exit 5; }
  echo '# linked OK -> ./preduce md5='\$(md5sum preduce|cut -d' ' -f1) | tee -a \$SEAL

  # ---- OPP identity probe：product_reduce 无 ggml standalone 框架核 -> 参照 = scalar-ref（标量类档） ----
  #   scalar-ref 是 codegen-lottery（clang-18 可自向量化）——如实报其形态（透明·不要求纯标量）。
  objdump -d drv_p.o > drv_dis.txt 2>/dev/null
  for s in sc_nibble sc_offbin sc_codebook; do
    SV=\$(awk -F'\t' -v S=\$s 'index(\$0,\"<\"S\">:\"){f=1;next} f&&/^[0-9a-f]+ </{f=0} f&&\$3~/^v/&&\$3!~/^vset/{c++} END{print c+0}' drv_dis.txt)
    echo \"# OPP-IDENTITY product_reduce ref=scalar-ref(\$s) caliber=标量类 ggml_standalone=ABSENT(internal sub-primitive·T-CENSUS §一.F) scalarref_nonvset_vector=\$SV(codegen-lottery)\" | tee -a \$SEAL
  done

  # ---- hygiene / single-instance ----
  pkill -x preduce 2>/dev/null; sleep 0.3
  echo '# PRE_STRAY='\$(pgrep -x -c preduce || echo 0) | tee -a \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG

if [ \"$MODE\" = verify ]; then
  CORE=\$(echo $CORES | awk '{print \$1}')
  echo \"# core=\$CORE (verify-only: correctness, no timing => no load-gate)\" | tee -a \$LOG
  runv(){ taskset -c \$CORE ./preduce \"\$@\" 2>&1 | tee -a \$LOG; }
  echo '=== [A] CLEAN INJECT=0 (nb=$NB_VERIFY, verify-only) — expect ZERO-MODEL byte-exact-vs-scalar-ref OK ===' | tee -a \$LOG
  runv $FMT $NB_VERIFY 1 10 $SV 1 0
  echo '=== [B] ANTI-HOLLOW #1 DUT-fault (INJECT=1, ours[mid]+=1) — expect BITES-OK (mismatch>0) ===' | tee -a \$LOG
  runv $FMT $NB_VERIFY 1 10 $SV 1 1
  echo '=== [C] ANTI-HOLLOW #2 ORACLE-fault (INJECT=2, scalar-ref const poisoned) — expect BITES-OK (mismatch>0) ===' | tee -a \$LOG
  runv $FMT $NB_VERIFY 1 10 $SV 1 2
elif [ \"$MODE\" = sanity ]; then
  CORE=\$(echo $CORES | awk '{print \$1}')
  echo '=== 3x re-measure noise self-check (seed=$SV nb=$NB_MEASURE N=$REPS) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; taskset -c \$CORE ./preduce $FMT $NB_MEASURE $HIT $REPS $SV 0 0 | tee -a \$LOG; done
else
  # measure：load gate best-idle core（阈 70·§3.5 静板预飞）
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5
  BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null)\" | tee -a \$LOG
  echo '=== S1 COLD seed=$S1 (nb=$NB_MEASURE N=$REPS) ===' | tee -a \$LOG
  taskset -c \$CORE ./preduce $FMT $NB_MEASURE $HIT $REPS $S1 0 0 | tee -a \$LOG
  echo '=== S2 COLD seed=$S2 ===' | tee -a \$LOG
  taskset -c \$CORE ./preduce $FMT $NB_MEASURE $HIT $REPS $S2 0 0 | tee -a \$LOG
fi
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# STRAY='\$(pgrep -x -c preduce || echo 0) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "# HARNESS_RC=$RC"
# ★仓库侧【不落任何文件】：无 scp 回、无写盘。bench 解析上面的 stdout。
exit $RC
