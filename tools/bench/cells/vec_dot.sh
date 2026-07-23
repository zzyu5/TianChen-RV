#!/usr/bin/env bash
# tools/bench/cells/vec_dot.sh <board> <mode> <fmt> <regime> — K-quant vec_dot 每格对拍/计时 harness。
#
# op = vec_dot (block_qX_K · block_q8_K -> f32 scalar · HAS fp reduction ⇒ fp-contract 对称 =
#      同 scalar_vec_dot.sh S1)。DUT = 我方板端编译的 emitted block-dot leaf；
# OPP/REF = ggml 部署导出的 ggml_vec_dot_<fmt>_q8_K。若该格式有板宽 `_vl128/_vl256`
#      专化则记录专化；若实际部署仅有 `_generic`/exported thunk（q5_K 当前即如此），则记录
#      真实可达路径及其档位。真实路径与强对手成色分列，绝不因专化缺席伪造身份或回退别的函数。
#
# 权威 = .trellis/spec/measurement/哲学与目的地.md §3.2.4（住 tools/、写 experiments/）
#        + 《测试与收尾总令-开测篇》§〇.2（ISSUE-090 harness 契约）
#        + 对手法 §3.4（部署事实 + 逐格档位）+ 正确性门.md [K-5]（整数 byte-exact·ZERO-MODEL）。
#        补 K-attack-fanout-ledger item3/4：K-quant vec_dot 手调族攻坚缺 harness。
#
# ★runner 接线（B2·ISSUE-104 facet 2/3）：
#   - canonical runner 通过显式 CELL_ROUTES 精确匹配 op/format/board/engine；本文件只承接
#     q2_K–q6_K × rvv/k1 × engine=rvv，不再按 cells/<op>.sh 同名猜路。
#   - scalar vec_dot 精确路由到 scalar_vec_dot.sh；其 route/parser 仅 dormant，roster 与
#     ISSUE-061/104 仍使真跑在 SSH 前失败。本文件保留 board=scalar guard 只是纵深防御，
#     不是第二个 dispatcher，也不授予 scalar coverage。
#
# ★契约（硬·同 gemm_tile.sh / dequantize_row.sh / scalar_vec_dot.sh）：
#   - 本 harness 由 ../bench 按声明接口调用：
#     `vec_dot.sh <board> <mode> <fmt> <regime>`。
#   - **harness 自身禁写任何【仓库侧】持久文件** —— 板端跑完把结果全部打到 stdout；
#     bench 解析 stdout，一切仓库侧持久写入经 runner 的 fail-closed 写入闸落三目的地。
#   - 板端 /tmp/$RDIR 下的 build/log = 板端临时（可接受）；仓库侧【不 scp 回、不落任何文件】。
#   - 数据格只提供 driver/oracle；五个 DUT leaf 均由当前 clean HEAD 现场构造，暂存后上板。
#
#   board: rvv | k1（ISSUE-105 已解决；k1 走本板 clang-18、真实 VLEN256 与 `_vl256` 探针）
#   fmt  : q2_K | q3_K | q4_K | q5_K | q6_K（五个 raw-byte oracle 分立）
#   regime: micro-fixed（固定 K/M/nc 的 kernel microbenchmark）
#   mode : verify  = build + ZEROVEC objdump 探针 + OPP 部署派发探针(档位) + ZERO-MODEL 3-way
#                    byte-exact + 3 个结果故障臂 + q8-bsums 输入契约负控；q2/q4/q5 另有
#                    min-term-active 反事实臂（全部 NO TIMING）
#          sanity  = 预测量噪声自检 3 轮
#          measure = cold N=25 2-seed flush（fp-contract=on symmetric·同 scalar S1）
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"; FMT="${3:-q4_K}"; REGIME="${4:-}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/../../.." && pwd)"
ASSETS="${VEC_DOT_ASSET_ROOT:-$ROOT/experiments/active/k-vecdot-harness}"
EXPORTER="$ROOT/tools/bench/export_current_artifact.py"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/weft-current-vec-dot.XXXXXX")" || {
  echo "# HARNESS-VOID cannot create current-artifact staging directory"; exit 3;
}
trap 'rm -rf -- "$STAGE"' EXIT

K=2048; NC_MEASURE=512; NC_VERIFY=64; REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
RDIR=/tmp/bench_cells_vec_dot_${BOARD}_${FMT}

echo "# HARNESS vec_dot board=$BOARD mode=$MODE fmt=$FMT regime=$REGIME support_assets=$ASSETS"

case "$FMT" in
  q2_K|q3_K|q4_K|q5_K|q6_K) : ;;
  *) echo "# HARNESS-VOID bad fmt $FMT (vec_dot K-quant 仅 q2_K|q3_K|q4_K|q5_K|q6_K)"; exit 2 ;;
esac
case "$MODE" in
  verify|sanity|measure) : ;;
  *) echo "# HARNESS-VOID bad mode $MODE (仅 verify|sanity|measure)"; exit 2 ;;
esac
if [ "$REGIME" != micro-fixed ]; then
  echo "# HARNESS-VOID unsupported regime ${REGIME:-<missing>} (vec_dot 当前只有 micro-fixed workload)"; exit 2
fi
OURSYM="weft_emitc_ggml_vec_dot_${FMT}_q8_K_kernel_rvv_${FMT}_q8_K_block_dot"; OPPSYM="ggml_vec_dot_${FMT}_q8_K"

if [ "$BOARD" = rvv ]; then
  GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
  CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang
  GT=/opt/tcrv-toolchains/gcc-15.2.0
  MARCH=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zba_zbb_zbc_zbs_zicbom_zicboz_zawrs_zicond_zfa_zihintntl_zihintpause
  CFLAGS="-fno-integrated-as -ffp-contract=on"
  LDEXTRA="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"
  ENVSRC="source /opt/tcrv-toolchains/env.sh;"
  CORES="${BENCH_CORES:-8 9 10 11 12 13 14 15}"   # 0,1 = co-tenant vLLM, NEVER touched
  FLUSH_MB=224                                     # > rvv L3
  VLTAG=vl128
elif [ "$BOARD" = scalar ]; then
  echo "# HARNESS-VOID board=scalar (scalar vec_dot 走 cells/scalar_vec_dot.sh·[L-6] enablement·本 harness 是 rvv/k1 K-quant·不冒充 scalar 覆盖·ISSUE-104)"; exit 2
elif [ "$BOARD" = k1 ]; then
  GGML=/data/k1build-stock/bin
  CC=clang-18
  MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs
  CFLAGS="-ffp-contract=on"
  LDEXTRA=""
  ENVSRC=""
  CORES="${BENCH_CORES:-0 1 2 3 4 5 6 7}"
  FLUSH_MB=32
  VLTAG=vl256
else
  echo "# HARNESS-VOID unsupported board $BOARD (vec_dot 族仅 rvv|k1)"; exit 2
fi

DRV="$ASSETS/kquant_vecdot_driver.c"
FMTS=(q2_K q3_K q4_K q5_K q6_K)
LEAVES=()
[ -f "$DRV" ] || { echo "# HARNESS-VOID missing support driver $DRV"; exit 3; }
[ -x "$EXPORTER" ] || { echo "# HARNESS-VOID missing current-artifact exporter $EXPORTER"; exit 3; }
for qfmt in "${FMTS[@]}"; do
  leaf="$STAGE/$qfmt.kernel.c"
  python3 "$EXPORTER" vec_dot "$qfmt" --board "$BOARD" --output "$leaf" --require-clean \
    || { echo "# HARNESS-VOID current compiler failed to export vec_dot/$qfmt"; exit 3; }
  [ -f "$leaf" ] || { echo "# HARNESS-VOID current compiler produced no leaf $leaf"; exit 3; }
  LEAVES+=("$leaf")
done
echo "# CURRENT_ARTIFACT_SET five_leaves=PASS head=$(git -C "$ROOT" rev-parse HEAD)"

ssh "$BOARD" "mkdir -p $RDIR" || { echo "# HARNESS-VOID ssh mkdir failed"; exit 3; }
scp -q "$DRV" "${LEAVES[@]}" "$BOARD:$RDIR/" || { echo "# HARNESS-VOID scp driver/leaves"; exit 3; }

ssh "$BOARD" "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL

  echo '# BUILD vec_dot fmt=$FMT board=$BOARD CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB (clang-18 symmetric·fp-contract=on)' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# opponent_sha256='\$(sha256sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_set_md5='\$(md5sum q2_K.kernel.c q3_K.kernel.c q4_K.kernel.c q5_K.kernel.c q6_K.kernel.c | sha256sum | cut -d' ' -f1)' drv_md5='\$(md5sum kquant_vecdot_driver.c|cut -d' ' -f1) | tee -a \$SEAL

  # ---- FAULT leaf: first accumulator 0.0f -> 1.0f in the tested fmt ----
  # q2_K uses a scalar seed; the other four use vfmv. First-match mutation reaches
  # either form without a per-format compatibility path.
  sed '0,/0\.0f/s//1.0f/' ${FMT}.kernel.c > ${FMT}_FAULT.c
  D=\$(cmp -l ${FMT}.kernel.c ${FMT}_FAULT.c 2>/dev/null | wc -l)
  echo '# fault_leaf differing_bytes='\$D' (must be exactly 1)' | tee -a \$SEAL
  if [ \"\$D\" != 1 ]; then echo '# VOID-EXPORT: fault leaf is not a single-byte delta'; exit 6; fi

  # ---- all five owned leaves build and co-link; absence of any one is fatal ----
  for F in q2_K q3_K q4_K q5_K q6_K; do
    $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ \${F}.kernel.c -c -o k_\${F}.o 2>cc_\${F}.err \
      || { echo VOID-BUILD leaf \$F; sed -n '1,15p' cc_\${F}.err; exit 3; }
  done
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ ${FMT}_FAULT.c -c -o k_${FMT}_F.o 2>cc_leafF.err \
    || { echo VOID-BUILD leafF; sed -n '1,15p' cc_leafF.err; exit 3; }

  # ---- driver build (pure C) ----
  $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c kquant_vecdot_driver.c -c -o drv.o 2>cc_drv.err || { echo VOID-BUILD drv; sed -n '1,25p' cc_drv.err; exit 4; }
  # Link the exact five owned objects against the board's deployed ggml library.
  $CC drv.o k_q2_K.o k_q3_K.o k_q4_K.o k_q5_K.o k_q6_K.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vd_$FMT 2>ld0.err \
    || { echo VOID-BUILD link0; sed -n '1,20p' ld0.err; exit 5; }
  # Five canonical owned symbols must be present exactly once in the clean binary.  This is
  # a link-closure gate; the five per-format board invocations below prove execution.
  NSYM=\$(nm --defined-only vd_$FMT 2>/dev/null | grep -cE 'weft_emitc_ggml_vec_dot_q[2-6]_K_q8_K_kernel_rvv_q[2-6]_K_q8_K_block_dot\$')
  echo '# OWNED_FIVE_SYMBOLS count='\$NSYM' expected=5' | tee -a \$SEAL
  if [ \"\$NSYM\" != 5 ]; then echo '# VOID-BUILD: five-leaf symbol closure failed'; exit 5; fi
  mv k_${FMT}.o k_${FMT}_clean.o
  mv k_${FMT}_F.o k_${FMT}.o
  $CC drv.o k_q2_K.o k_q3_K.o k_q4_K.o k_q5_K.o k_q6_K.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vd_${FMT}_F 2>ldF.err \
    || { echo VOID-BUILD linkF; sed -n '1,20p' ldF.err; exit 5; }
  echo '# linked OK md5='\$(md5sum vd_$FMT|cut -d' ' -f1)' FAULTbin md5='\$(md5sum vd_${FMT}_F|cut -d' ' -f1) | tee -a \$SEAL

  # ---- ZEROVEC objdump machine-check on OUR leaf.o (owned emitter axis) ----
  objdump -d k_${FMT}_clean.o > leaf_dis.txt 2>/dev/null
  ZV=\$(awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) v++; if(\$3 ~ /^vset/) vs++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) gat++; if(\$3 ~ /^(vfmul|vfcvt|vfmacc|vfmadd|vfredosum|vfredusum)/) vf++ } END{printf \"scalar_ins=%d vector_mnemonic=%d vset=%d gather=%d vfp=%d\", ins+0,v+0,vs+0,gat+0,vf+0}' leaf_dis.txt)
  NONVSET=\$(awk -F'\t' '\$3 ~ /^v/ && \$3 !~ /^vset/ {c++} END{print c+0}' leaf_dis.txt)
  OWNED=\$(grep -oE '__riscv_v[a-z0-9_]+' ${FMT}.kernel.c | grep -v vsetvl | sort -u | wc -l)
  OWNEDCALL=\$(grep -cE '__riscv_v[a-z0-9_]+' ${FMT}.kernel.c)
  echo \"# OURS_leaf [\$ZV] non_vset_vector=\$NONVSET OWNED_src_vec_intrinsics=\$OWNED distinct/\$OWNEDCALL calls(vsetvl 除外) sym=$OURSYM\" | tee -a \$LOG

  # ---- OPP dispatch probe：实际部署路径与对手档分列；specialization absence 不等于 opponent absence ----
  objdump -t $GGML/libggml-cpu.so 2>/dev/null | grep -oE '$OPPSYM(_vl128|_vl256|_vl512|_generic)?(\.isra\.[0-9]+)?' | sort -u > opp_family.txt
  FAM=\$(tr '\n' ',' < opp_family.txt)
  VLSYM=\$(grep -E '$OPPSYM(_$VLTAG)(\.isra\.[0-9]+)?\$' opp_family.txt | sed -n '1p')
  GENSYM=\$(grep -E '$OPPSYM(_generic)(\.isra\.[0-9]+)?\$' opp_family.txt | sed -n '1p')
  THUNKSYM=\$(grep -E '$OPPSYM(\.isra\.[0-9]+)?\$' opp_family.txt | sed -n '1p')
  if [ -n \"\$VLSYM\" ]; then
    RUNSYM=\$VLSYM; PATHKIND=$VLTAG; PROOF=board-vlen-specialization
  elif [ -n \"\$THUNKSYM\" ]; then
    objdump -d --disassemble=\"\$THUNKSYM\" $GGML/libggml-cpu.so > opp_thunk_dis.txt 2>/dev/null
    if [ -n \"\$GENSYM\" ] && grep -q \"\$GENSYM\" opp_thunk_dis.txt; then
      RUNSYM=\$GENSYM; PATHKIND=generic; PROOF=exported-thunk-ref-generic
    else
      RUNSYM=\$THUNKSYM; PATHKIND=exported; PROOF=exported-symbol-body
    fi
  else
    echo \"# OPP $OPPSYM ABSENT family=[\$FAM] (fail closed: exported opponent missing)\" | tee -a \$SEAL
    exit 7
  fi
  objdump -d --disassemble=\"\$RUNSYM\" $GGML/libggml-cpu.so > opp_dis.txt 2>/dev/null
  OZ=\$(awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/ && \$3 !~ /^vset/) v++ } END{printf \"ins=%d non_vset_vector=%d\", ins+0,v+0}' opp_dis.txt)
  NRVV=\$(awk -F'\t' '\$3 ~ /^v/ && \$3 !~ /^vset/ {c++} END{print c+0}' opp_dis.txt)
  if [ \"\$NRVV\" -eq 0 ]; then CAL=标量类
  elif [ \"\$PATHKIND\" = "$VLTAG" ]; then CAL=手调
  else CAL=通用向量
  fi
  echo \"# OPP $OPPSYM PRESENT runtime_path=\$RUNSYM path_kind=\$PATHKIND dispatch_proof=\$PROOF [\$OZ] caliber=\$CAL family=[\$FAM]\" | tee -a \$SEAL

  # ---- hygiene / single-instance ----
  stray(){ { pgrep -x vd_$FMT; pgrep -x vd_${FMT}_F; } 2>/dev/null | wc -l; }
  for b in vd_$FMT vd_${FMT}_F; do pkill -x \$b 2>/dev/null; done; sleep 0.3
  echo '# PRE_STRAY='\$(stray) | tee -a \$LOG
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG

if [ \"$MODE\" = verify ]; then
  CORE=\$(echo $CORES | awk '{print \$1}')
  echo \"# core=\$CORE (verify-only: correctness, no timing => no load-gate)\" | tee -a \$LOG
else
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
fi
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./\$1 \"\${@:2}\" 2>&1 | tee -a \$LOG; }

if [ \"$MODE\" = verify ]; then
  echo '=== [A] CLEAN (nc=$NC_VERIFY reps=0 verify-only) — expect 3-way byte-exact ALL=true + fp16 golden PASS ===' | tee -a \$LOG
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 0 0 0
  echo '=== [B] ANTI-HOLLOW #1 DUT-out fault (inject=1) — expect BITES-OK (ours != oracle & != ggml) ===' | tee -a \$LOG
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 1 0 0
  echo '=== [C] ANTI-HOLLOW #2 ORACLE-const fault (inject=2) — expect BITES-OK (oracle != ours & != ggml) ===' | tee -a \$LOG
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 2 0 0
  echo '=== [D] ANTI-HOLLOW #3 LEAF seed fault (emitted RISC-V) — expect ours_vs_* RED (byte-exact bites the emit) ===' | tee -a \$LOG
  run vd_${FMT}_F $FMT $K 1 $NC_VERIFY 0 $SV 0 0 0
  case $FMT in
    q2_K|q4_K|q5_K)
      echo '=== [E] MIN-TERM-ACTIVE dmin=1 + nonuniform mins/q8 group sums — expect byte-exact + nonzero witness ===' | tee -a \$LOG
      run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 0 0 1 ;;
  esac
  echo '=== [F] INPUT-CONTRACT fault q8 bsums != qs — expect validator BITES-OK before evaluation ===' | tee -a \$LOG
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 3 0 0
elif [ \"$MODE\" = sanity ]; then
  echo '=== 3x re-measure noise self-check (seed=$SV nc=$NC_MEASURE N=$REPS) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; run vd_$FMT $FMT $K 1 $NC_MEASURE $REPS $SV 0 $FLUSH_MB 0; done
else
  echo '=== S1 COLD seed=$S1 (nc=$NC_MEASURE N=$REPS flush=${FLUSH_MB}MiB) ===' | tee -a \$LOG; run vd_$FMT $FMT $K 1 $NC_MEASURE $REPS $S1 0 $FLUSH_MB 0
  echo '=== S2 COLD seed=$S2 ===' | tee -a \$LOG; run vd_$FMT $FMT $K 1 $NC_MEASURE $REPS $S2 0 $FLUSH_MB 0
fi
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  echo '# STRAY='\$(stray) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
RC=$?
echo "# HARNESS_RC=$RC"
# ★仓库侧【不落任何文件】：无 scp 回、无写盘。bench 解析上面的 stdout。
exit $RC
