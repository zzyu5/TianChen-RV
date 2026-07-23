#!/usr/bin/env bash
# tools/bench/cells/scalar_vec_dot.sh <board> <mode> <fmt> <regime> — scalar 真 no-V 硅 vec_dot
# 每格对拍/计时 harness（S 线·超锐/scalar 板·rv64gc·clang-18）。
#
# 权威 = .trellis/spec/measurement/哲学与目的地.md §3.2.4（住 tools/、写 experiments/）
#        + 《测试与收尾总令-开测篇》§〇.2（ISSUE-090 harness 契约）
#        + 对手法 §3.4 板别提醒（无 V 板 _generic = 真部署对手）+ 板册 §3.7。
#
# ★契约（硬·同 gemm_tile.sh）：
#   - 本 harness 由 ../bench 按声明接口调用：
#     `scalar_vec_dot.sh <board> <mode> <fmt> <regime>`。
#   - **harness 自身禁写任何【仓库侧】持久文件** —— 板端跑完把结果全部打到 stdout；
#     bench 解析 stdout，一切仓库侧持久写入经 runner 的 fail-closed 写入闸落三目的地。
#   - 板端 /tmp/$RDIR 下的 build/log = 板端临时（可接受）；仓库侧【不 scp 回、不落任何文件】。
#
# ★域（钉死）：[L-6] scalar 永不作贡献基线 —— 计时是 enablement 诊断·NON-Win·不进系统账/
#   perf-covered。倍数如实记，域标 enablement-NONWIN。
#
# A3 数据格只提供 scalar_s1_driver.cpp 与 fp16util.cpp；DUT kernel 由当前 clean HEAD
# 现场构造，暂存后上板。
#
#   board: scalar（仅此板 —— 标量对局唯一合法场地·板册 §3.7）
#   fmt  : tq2_0
#   regime: micro-fixed（固定 nb/flush 的 enablement microbenchmark）
#   mode : verify  = 零向量 objdump 机检 + ZERO-MODEL byte-exact（对称 & 交叉）+ 反空心 fault（NO TIMING）
#          sanity  = 3 轮预测量噪声自检
#          measure = cold N=25 2-seed flush（symmetric-with-deployed-ggml 构建）
set -uo pipefail
BOARD="${1:-scalar}"; MODE="${2:-verify}"; FMT="${3:-tq2_0}"; REGIME="${4:-}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/../../.." && pwd)"
ASSETS="${SCALAR_VECDOT_ASSET_ROOT:-$ROOT/experiments/active/g8-stage3-attack/A3-xscalar-rv64gc}"
EXPORTER="$ROOT/tools/bench/export_current_artifact.py"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/weft-current-scalar-vec-dot.XXXXXX")" || {
  echo "# HARNESS-VOID cannot create current-artifact staging directory"; exit 3;
}
trap 'rm -rf -- "$STAGE"' EXIT

REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
NB_MEASURE=2048          # n = 524288 elements
NB_VERIFY=4096           # larger corpus for correctness
FLUSH_MB=32              # > scalar board LLC (§3.7: no L3; DRAM cold)
RDIR=/tmp/bench_cells_scalar_vec_dot_${FMT}

echo "# HARNESS scalar_vec_dot board=$BOARD mode=$MODE fmt=$FMT regime=$REGIME support_assets=$ASSETS"

if [ "$BOARD" != scalar ]; then
  echo "# HARNESS-VOID board=$BOARD unsupported (scalar_vec_dot 仅 scalar 板 —— 板册 §3.7 标量对局唯一合法场地)"; exit 2
fi
if [ "$FMT" != tq2_0 ]; then
  echo "# HARNESS-VOID fmt=$FMT unsupported (scalar_vec_dot 族当前仅 tq2_0)"; exit 2
fi
case "$MODE" in
  verify|sanity|measure) : ;;
  *) echo "# HARNESS-VOID bad mode $MODE (仅 verify|sanity|measure)"; exit 2 ;;
esac
if [ "$REGIME" != micro-fixed ]; then
  echo "# HARNESS-VOID unsupported regime ${REGIME:-<missing>} (scalar_vec_dot 当前只有 micro-fixed workload)"; exit 2
fi

GGML='~/llama.cpp-scalar/build-clang18-rv64gc/bin'   # GGML_RVV=OFF·GGML_NATIVE=OFF（板册 §3.1）
CC=clang-18                                           # /usr/bin/clang-18 · 18.1.8 · 板出货链
MARCH=rv64gc
CORES="${BENCH_CORES:-0 1 2 3 4 5 6 7}"

DRV="$ASSETS/scalar_s1_driver.cpp"
KERN="$STAGE/weft_scalar_tq2_0_kernel.cpp"
FP16="$ASSETS/fp16util.cpp"
for f in "$DRV" "$FP16"; do
  [ -f "$f" ] || { echo "# HARNESS-VOID missing asset $f"; exit 3; }
done
[ -x "$EXPORTER" ] || { echo "# HARNESS-VOID missing current-artifact exporter $EXPORTER"; exit 3; }
python3 "$EXPORTER" vec_dot tq2_0 --board scalar --output "$KERN" --require-clean \
  || { echo "# HARNESS-VOID current compiler failed to export vec_dot/tq2_0"; exit 3; }
[ -f "$KERN" ] || { echo "# HARNESS-VOID current compiler produced no kernel $KERN"; exit 3; }

ssh "$BOARD" "mkdir -p $RDIR" || { echo "# HARNESS-VOID ssh mkdir failed"; exit 3; }
scp -q "$DRV" "$KERN" "$FP16" "$BOARD:$RDIR/" || { echo "# HARNESS-VOID scp assets"; exit 3; }

ssh "$BOARD" "set -uo pipefail; cd $RDIR
  GGML=$GGML; CC=$CC
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL

  echo '# BUILD scalar_vec_dot fmt=$FMT board=$BOARD CC='\$(\$CC --version|head -1)' march=$MARCH' | tee -a \$SEAL
  LIB=\$(find \$GGML -name libggml-cpu.so.0 | head -1)
  echo '# cpu_lib='\$LIB' md5_before='\$(md5sum \$LIB|cut -d' ' -f1) | tee -a \$SEAL
  echo '# isa='\$(awk -F': ' '/^isa/{print \$2; exit}' /proc/cpuinfo) | tee -a \$SEAL

  # ---- fp16util (shared) ----
  \$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -c fp16util.cpp -o fp16util.o 2>cc_fp.err \
    || { echo VOID-BUILD fp16util; head -15 cc_fp.err; exit 3; }

  # ---- symmetric-with-deployed-ggml build (contract=on = ggml 出货折叠口径·同 rvv/k1 harness) ----
  \$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -c scalar_s1_driver.cpp -o drvON.o 2>cc_on.err \
    || { echo VOID-BUILD driverON; head -20 cc_on.err; exit 4; }
  \$CC drvON.o fp16util.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o s1on 2>ldon.err \
    || { echo VOID-BUILD linkON; head -20 ldon.err; exit 5; }
  echo '# linked s1on md5='\$(md5sum s1on|cut -d' ' -f1)' (contract=on·symmetric)' | tee -a \$SEAL

if [ \"$MODE\" = verify ]; then
  # ---- compiler-neutral cross build (contract=off·ZERO-MODEL 复现口径) ----
  \$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=off -c scalar_s1_driver.cpp -o drvOFF.o 2>cc_off.err \
    || { echo VOID-BUILD driverOFF; head -20 cc_off.err; exit 4; }
  \$CC drvOFF.o fp16util.o -L\$GGML -Wl,-rpath,\$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o s1off 2>ldoff.err \
    || { echo VOID-BUILD linkOFF; head -20 ldoff.err; exit 5; }

  # ---- [T-X col3] zero-vector objdump machine-check on OUR kernel .o (static·board-independent) ----
  \$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -include cstddef -c weft_scalar_tq2_0_kernel.cpp -o kernel.o 2>cc_k.err \
    || { echo VOID-BUILD kernel.o; head -15 cc_k.err; exit 4; }
  objdump -d kernel.o > kern_dis.txt 2>/dev/null
  ZV=\$(awk -F'\t' '\$3 ~ /^[a-z]/ {ins++; if(\$3 ~ /^v/) v++; if(\$3 ~ /^vset/) vs++; if(\$3 ~ /cpop|clmul/) pc++} END{printf \"scalar_ins=%d vector_mnemonic=%d vset=%d cpop_clmul=%d\", ins+0,v+0,vs+0,pc+0}' kern_dis.txt)
  RI=\$(grep -c '__riscv_' kern_dis.txt); WR=\$(grep -c 'weft_rvv' kern_dis.txt); FL=\$(grep -cE '__extendhfsf2|__truncsfhf2' kern_dis.txt)
  echo \"# ZEROVEC kernel.o [\$ZV] riscv_intrinsic=\$RI weft_rvv_sym=\$WR fp16_libcall=\$FL\" | tee -a \$LOG
  # also the deployed ggml lib (whole-file) must be v-free (板册 §3.7 实证)
  LIBV=\$(objdump -d \$LIB 2>/dev/null | awk -F'\t' '\$3 ~ /^v/ {c++} END{print c+0}')
  echo \"# ZEROVEC deployed_ggml_lib v_mnemonic_lines=\$LIBV (0 = no-V build·真部署标量派发)\" | tee -a \$LOG

  # ---- [T-X col4/5] ZERO-MODEL byte-exact (symmetric contract=on) + cross-neutral (contract=off) ----
  echo '=== [A] byte-exact symmetric (contract=on·vs deployed ggml) ===' | tee -a \$LOG
  LD_LIBRARY_PATH=\$GGML ./s1on $NB_VERIFY 0 $SV 0 0 | tee -a \$LOG
  echo '=== [Boff] byte-exact compiler-neutral (contract=off·ZERO-MODEL vs independent int-oracle) ===' | tee -a \$LOG
  LD_LIBRARY_PATH=\$GGML ./s1off $NB_VERIFY 0 $SV 0 0 | tee -a \$LOG
  echo '=== [C] ANTI-HOLLOW #1 corrupt OURS output (INJECT=1) — expect byte-exact BITES-OK ===' | tee -a \$LOG
  LD_LIBRARY_PATH=\$GGML ./s1on 256 0 $SV 1 0 | tee -a \$LOG
  echo '=== [D] ANTI-HOLLOW #2 corrupt oracle constant (INJECT=2) — expect byte-exact BITES-OK ===' | tee -a \$LOG
  LD_LIBRARY_PATH=\$GGML ./s1on 256 0 $SV 2 0 | tee -a \$LOG

elif [ \"$MODE\" = sanity ]; then
  echo '=== 3x re-measure noise self-check (seed=$SV N=$REPS flush=${FLUSH_MB}MiB) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG
    LD_LIBRARY_PATH=\$GGML taskset -c 2 ./s1on $NB_MEASURE $REPS $SV 0 $FLUSH_MB | tee -a \$LOG; done

else   # measure
  # light load gate (§3.7: 载荷极低·but honest): pick best-idle core from CORES
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0; for c in $CORES; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.5; BESTC=-1; BESTIDLE=-1
  for c in $CORES; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# VOID-LOAD best core\$BESTC idle=\${BESTIDLE}%\" | tee -a \$LOG; exit 9; fi
  echo \"# LOAD_GATE_OK core=\$BESTC idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${BESTC}/cpufreq/scaling_governor 2>/dev/null)\" | tee -a \$LOG
  echo '=== S1 COLD seed=$S1 (N=$REPS flush=${FLUSH_MB}MiB·contract=on symmetric) ===' | tee -a \$LOG
  LD_LIBRARY_PATH=\$GGML taskset -c \$BESTC ./s1on $NB_MEASURE $REPS $S1 0 $FLUSH_MB | tee -a \$LOG
  echo '=== S1 COLD seed=$S2 ===' | tee -a \$LOG
  LD_LIBRARY_PATH=\$GGML taskset -c \$BESTC ./s1on $NB_MEASURE $REPS $S2 0 $FLUSH_MB | tee -a \$LOG
fi
  echo '# cpu_md5_after='\$(md5sum \$LIB|cut -d' ' -f1) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "# HARNESS_RC=$RC"
# ★仓库侧【不落任何文件】：无 scp 回、无写盘。bench 解析上面的 stdout。
exit $RC
