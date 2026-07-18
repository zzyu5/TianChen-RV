#!/usr/bin/env bash
# tools/bench/cells/vec_dot.sh <board> <mode> <fmt> — K-quant vec_dot 每格对拍/计时 harness。
#
# op = vec_dot (block_qX_K · block_q8_K -> f32 scalar · HAS fp reduction ⇒ fp-contract 对称 =
#      同 scalar_vec_dot.sh S1)。DUT = 我方板端编译的 emitted block-dot leaf；
# OPP/REF = ggml 部署派发的 ggml_vec_dot_<fmt>_q8_K —— @VLEN128 落 **手调 `_vl128` 专化**
#      (hand-tuned STRONG·手调档·非便宜 _generic 档·对手法 §3.4 部署事实·成色如实由 OPP 探针机判)。
#
# 权威 = .trellis/spec/measurement/哲学与目的地.md §3.2.4（住 tools/、写 experiments/）
#        + 《测试与收尾总令-开测篇》§〇.2（ISSUE-090 harness 契约）
#        + 对手法 §3.4（部署事实 + 逐格档位）+ 正确性门.md [K-5]（整数 byte-exact·ZERO-MODEL）。
#        补 K-attack-fanout-ledger item3/4：K-quant vec_dot 手调族攻坚缺 harness。
#
# ★命名与 runner 接线（ISSUE-104 facet 2）：
#   - runner cell_harness(op) 按 cells/<op>.sh 解析·主表 K-quant vec_dot 行 op=vec_dot ⟹ 本文件
#     命名 `vec_dot.sh` **直接接线**（`bench vec_dot q4_K --board rvv` → 本 harness）。
#   - scalar 板 vec_dot 另有 **scalar_vec_dot.sh**（scalar-scoped·[L-6] enablement）——**文件名不冲突**；
#     但 runner 按 op-only 解析 ⟹ `--board scalar` 亦会路由到本文件。本 harness board=scalar 即
#     HARNESS-VOID 并指向 scalar_vec_dot.sh（不冒充 scalar 覆盖）。**harness 粒度 op-全板 vs op×板
#     未钦定 = 判据级·登记 ISSUE-104·本 harness 不擅定**（保守默认：board-guard 分流）。
#
# ★契约（硬·同 gemm_tile.sh / dequantize_row.sh / scalar_vec_dot.sh）：
#   - 本 harness 由 ../bench 按声明接口调用：`vec_dot.sh <board> <mode> <fmt>`。
#   - **harness 自身禁写任何【仓库侧】持久文件** —— 板端跑完把结果全部打到 stdout；
#     bench 解析 stdout，一切仓库侧持久写入经 runner 的 fail-closed 写入闸落三目的地。
#   - 板端 /tmp/$RDIR 下的 build/log = 板端临时（可接受）；仓库侧【不 scp 回、不落任何文件】。
#   - 源资产（driver / oracle / leaf）住数据格，本 harness 只【读】（ASSET_ROOT / KERNEL_ROOT 覆写）。
#
#   board: rvv          （@k1 gated on ISSUE-105 VLEN256 半宽 fixture — 本格 @rvv 先证·k1=VOID）
#   fmt  : q4_K | q6_K  （baseline 首攻 2 格·oracle 已建；q2_K/q3_K/q5_K = 后续 oracle 待补）
#   mode : verify  = build + ZEROVEC objdump 探针 + OPP 部署派发探针(档位) + ZERO-MODEL 3-way
#                    byte-exact + 4-arm 反空心 (NO TIMING)
#          sanity  = 预测量噪声自检 3 轮
#          measure = cold N=25 2-seed flush（fp-contract=on symmetric·同 scalar S1）
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"; FMT="${3:-q4_K}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/../../.." && pwd)"
ASSETS="${VEC_DOT_ASSET_ROOT:-$ROOT/experiments/active/k-vecdot-harness}"
KERNELS="${VEC_DOT_KERNEL_ROOT:-$ROOT/experiments/active/g7-census/vecdot-rvv/kernels}"

K=2048; NC_MEASURE=512; NC_VERIFY=64; REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
RDIR=/tmp/bench_cells_vec_dot_${BOARD}_${FMT}

echo "# HARNESS vec_dot board=$BOARD mode=$MODE fmt=$FMT assets=$ASSETS kernels=$KERNELS"

# driver serves both q4_K/q6_K (ZERO-MODEL oracle built for both) ⟹ both leaves must link;
# the tested $FMT is the one the driver drives + the one the fault-arm perturbs. OTHER = the co-linked clean leaf.
case "$FMT" in
  q4_K) OTHER=q6_K ;;
  q6_K) OTHER=q4_K ;;
  q2_K|q3_K|q5_K) echo "# HARNESS-VOID fmt=$FMT: leaf 存在但 driver ZERO-MODEL oracle 待补（q2_K/q3_K/q5_K = 后续 task·K-attack-fanout item4）"; exit 2 ;;
  *) echo "# HARNESS-VOID bad fmt $FMT (vec_dot K-quant 首攻仅 q4_K/q6_K)"; exit 2 ;;
esac
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
elif [ "$BOARD" = scalar ]; then
  echo "# HARNESS-VOID board=scalar (scalar vec_dot 走 cells/scalar_vec_dot.sh·[L-6] enablement·本 harness 是 rvv/k1 K-quant·不冒充 scalar 覆盖·ISSUE-104)"; exit 2
elif [ "$BOARD" = k1 ]; then
  echo "# HARNESS-VOID board=k1 (vec_dot @k1 gated on ISSUE-105 VLEN256 半宽 fixture — 本格 @rvv 先证隔离该变量)"; exit 2
else
  echo "# HARNESS-VOID unsupported board $BOARD (vec_dot 族当前仅 rvv)"; exit 2
fi

DRV="$ASSETS/kquant_vecdot_driver.c"
LEAF="$KERNELS/$FMT.kernel.c"
LEAFO="$KERNELS/$OTHER.kernel.c"
for f in "$DRV" "$LEAF" "$LEAFO"; do
  [ -f "$f" ] || { echo "# HARNESS-VOID missing asset $f"; exit 3; }
done

ssh "$BOARD" "mkdir -p $RDIR" || { echo "# HARNESS-VOID ssh mkdir failed"; exit 3; }
scp -q "$DRV"   "$BOARD:$RDIR/kquant_vecdot_driver.c" || { echo "# HARNESS-VOID scp driver"; exit 3; }
scp -q "$LEAF"  "$BOARD:$RDIR/leaf_${FMT}.c"          || { echo "# HARNESS-VOID scp leaf"; exit 3; }
scp -q "$LEAFO" "$BOARD:$RDIR/leaf_${OTHER}.c"        || { echo "# HARNESS-VOID scp leaf-other"; exit 3; }

ssh "$BOARD" "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL

  echo '# BUILD vec_dot fmt=$FMT board=$BOARD CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB (clang-18 symmetric·fp-contract=on)' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_md5='\$(md5sum leaf_${FMT}.c|cut -d' ' -f1)' other_md5='\$(md5sum leaf_${OTHER}.c|cut -d' ' -f1)' drv_md5='\$(md5sum kquant_vecdot_driver.c|cut -d' ' -f1) | tee -a \$SEAL

  # ---- FAULT leaf: flip accumulator seed 0.0f -> 1.0f in the TESTED fmt (bites the EMITTED RISC-V numerics) ----
  sed 's/vfmv_v_f_f32m2(0.0f/vfmv_v_f_f32m2(1.0f/' leaf_${FMT}.c > leaf_${FMT}_FAULT.c
  D=\$(cmp -l leaf_${FMT}.c leaf_${FMT}_FAULT.c 2>/dev/null | wc -l)
  echo '# fault_leaf differing_bytes='\$D' (must be exactly 1)' | tee -a \$SEAL
  if [ \"\$D\" != 1 ]; then echo '# VOID-EXPORT: fault leaf is not a single-byte delta'; exit 6; fi

  # ---- ours leaf builds (both fmt clean + tested-fmt fault) : extern-C C++ ----
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_${FMT}.c        -c -o k_${FMT}.o    2>cc_leaf.err  || { echo VOID-BUILD leaf;  head -15 cc_leaf.err;  exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_${OTHER}.c      -c -o k_${OTHER}.o  2>cc_leafo.err || { echo VOID-BUILD leaf-other; head -15 cc_leafo.err; exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_${FMT}_FAULT.c  -c -o k_${FMT}_F.o  2>cc_leafF.err || { echo VOID-BUILD leafF; head -15 cc_leafF.err; exit 3; }

  # ---- driver build (pure C) ----
  $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c kquant_vecdot_driver.c -c -o drv.o 2>cc_drv.err || { echo VOID-BUILD drv; head -25 cc_drv.err; exit 4; }
  # NB: link flags INLINED (k1 zsh word-split hazard — kept for parity). REF ggml_vec_dot_${FMT}_q8_K in libggml-cpu.
  $CC drv.o k_${FMT}.o   k_${OTHER}.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vd_$FMT      2>ld0.err || { echo VOID-BUILD link0; head -20 ld0.err; exit 5; }
  $CC drv.o k_${FMT}_F.o k_${OTHER}.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vd_${FMT}_F  2>ldF.err || { echo VOID-BUILD linkF; head -20 ldF.err; exit 5; }
  echo '# linked OK md5='\$(md5sum vd_$FMT|cut -d' ' -f1)' FAULTbin md5='\$(md5sum vd_${FMT}_F|cut -d' ' -f1) | tee -a \$SEAL

  # ---- ZEROVEC objdump machine-check on OUR leaf.o (owned emitter axis) ----
  objdump -d k_${FMT}.o > leaf_dis.txt 2>/dev/null
  ZV=\$(awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) v++; if(\$3 ~ /^vset/) vs++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) gat++; if(\$3 ~ /^(vfmul|vfcvt|vfmacc|vfmadd|vfredosum|vfredusum)/) vf++ } END{printf \"scalar_ins=%d vector_mnemonic=%d vset=%d gather=%d vfp=%d\", ins+0,v+0,vs+0,gat+0,vf+0}' leaf_dis.txt)
  NONVSET=\$(awk -F'\t' '\$3 ~ /^v/ && \$3 !~ /^vset/ {c++} END{print c+0}' leaf_dis.txt)
  OWNED=\$(grep -oE '__riscv_v[a-z0-9_]+' leaf_${FMT}.c | grep -v vsetvl | sort -u | wc -l)
  OWNEDCALL=\$(grep -cE '__riscv_v[a-z0-9_]+' leaf_${FMT}.c)
  echo \"# OURS_leaf [\$ZV] non_vset_vector=\$NONVSET OWNED_src_vec_intrinsics=\$OWNED distinct/\$OWNEDCALL calls(vsetvl 除外) sym=$OURSYM\" | tee -a \$LOG

  # ---- OPP dispatch probe (家族一次身份探针 §3.4)：list specializations + disasm VLEN128 path + 档位 ----
  objdump -t $GGML/libggml-cpu.so 2>/dev/null | grep -oE '$OPPSYM(_vl128|_vl256|_vl512|_generic)?(\.isra\.[0-9]+)?' | sort -u > opp_family.txt
  FAM=\$(tr '\n' ',' < opp_family.txt)
  VL128SYM=\$(grep -E '$OPPSYM(_vl128)(\.isra\.[0-9]+)?\$' opp_family.txt | head -1)
  if [ -n \"\$VL128SYM\" ]; then
    objdump -d --disassemble=\"\$VL128SYM\" $GGML/libggml-cpu.so > opp_dis.txt 2>/dev/null
    OZ=\$(awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/ && \$3 !~ /^vset/) v++ } END{printf \"ins=%d non_vset_vector=%d\", ins+0,v+0}' opp_dis.txt)
    NRVV=\$(awk -F'\t' '\$3 ~ /^v/ && \$3 !~ /^vset/ {c++} END{print c+0}' opp_dis.txt)
    CAL=\$([ \"\$NRVV\" -gt 0 ] && echo 手调 || echo 标量类)
    echo \"# OPP $OPPSYM PRESENT vl128_runtime_path=\$VL128SYM [\$OZ] caliber=\$CAL family=[\$FAM]\" | tee -a \$SEAL
  else
    echo \"# OPP $OPPSYM family=[\$FAM] vl128_specialization=ABSENT (档位探针 gated·须查运行时派发)\" | tee -a \$SEAL
  fi

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
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 0 0
  echo '=== [B] ANTI-HOLLOW #1 DUT-out fault (inject=1) — expect BITES-OK (ours != oracle & != ggml) ===' | tee -a \$LOG
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 1 0
  echo '=== [C] ANTI-HOLLOW #2 ORACLE-const fault (inject=2) — expect BITES-OK (oracle != ours & != ggml) ===' | tee -a \$LOG
  run vd_$FMT $FMT $K 1 $NC_VERIFY 0 $SV 2 0
  echo '=== [D] ANTI-HOLLOW #3 LEAF seed fault (emitted RISC-V) — expect ours_vs_* RED (byte-exact bites the emit) ===' | tee -a \$LOG
  run vd_${FMT}_F $FMT $K 1 $NC_VERIFY 0 $SV 0 0
elif [ \"$MODE\" = sanity ]; then
  echo '=== 3x re-measure noise self-check (seed=$SV nc=$NC_MEASURE N=$REPS) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; run vd_$FMT $FMT $K 1 $NC_MEASURE $REPS $SV 0 $FLUSH_MB; done
else
  echo '=== S1 COLD seed=$S1 (nc=$NC_MEASURE N=$REPS flush=${FLUSH_MB}MiB) ===' | tee -a \$LOG; run vd_$FMT $FMT $K 1 $NC_MEASURE $REPS $S1 0 $FLUSH_MB
  echo '=== S2 COLD seed=$S2 ===' | tee -a \$LOG; run vd_$FMT $FMT $K 1 $NC_MEASURE $REPS $S2 0 $FLUSH_MB
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
