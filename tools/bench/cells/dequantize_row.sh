#!/usr/bin/env bash
# tools/bench/cells/dequantize_row.sh <board> <mode> <fmt> — dequantize_row 每格对拍/计时 harness。
#
# op = dequantize_row (block_qX -> f32 row · void-return · NO reduction · NO opponent vec_dot)。
# DUT = 我方板端编译的 emitted leaf；OPP/REF = ggml 部署的 dequantize_row_<fmt>（scalar-source
# host-autovec = codegen-lottery 对手·标量类档·对手法 §〇.1）。
#
# 权威 = .trellis/spec/measurement/哲学与目的地.md §3.2.4（住 tools/、写 experiments/）
#        + 《测试与收尾总令-开测篇》§〇.2（ISSUE-090 harness 契约）
#        + 对手法 §〇.1（标量类档）。补 ISSUE-099 dequant harness 缺口。
#
# ★契约（硬·同 gemm_tile.sh / scalar_vec_dot.sh）：
#   - 本 harness 由 ../bench 按声明接口调用：`dequantize_row.sh <board> <mode> <fmt>`
#     （runner cell_harness(op) 按 cells/<op>.sh 解析·op=dequantize_row）。
#   - **harness 自身禁写任何【仓库侧】持久文件** —— 板端跑完把结果全部打到 stdout；
#     bench 解析 stdout，一切仓库侧持久写入经 runner 的 fail-closed 写入闸落三目的地。
#   - 板端 /tmp/$RDIR 下的 seal/log = 板端临时（可接受）；仓库侧【不 scp 回、不落任何文件】。
#   - 源资产（driver / leaf / tables）住数据格，本 harness 只【读】（ASSET_ROOT 覆写）。
#
#   board: rvv          （@k1 dequant 部署 gated on ISSUE-105 半宽 — 本格 @rvv 先证·k1=VOID）
#   fmt  : iq3_xxs      （grid-codebook super-block·candidate ② OWNED SCALAR-load 查表 +
#                        unit-stride vector 算术·HW_GATHER=0·复刻对手形状·裁决3/ISSUE-107）
#          iq3_xxs_grid （★A/B 参照：同格的 grid HW-gather 变体·frozen 0.36 gather 天花板·
#                        vluxei16/vluxei8 indexed gather·裁决前的 sealed 记录）
#          q8_0         （NON-GRID flat block·owned real-vector·NO gather·首个标量门 owned 收口探路）
#          q4_0 q5_0    （flat nibble SAFE set·单 mul·owned real-vector·byte-exact by construction）
#          q4_1 q5_1    （flat nibble FMA set·x*d+m·owned real-vector·fused vfmacc·byte-exact 须证 contraction 一致）
#          q4_K q5_K    （K-quant super-block·per super-sub owned real-vector·fused vfmsac d1*v-m1·byte-exact 须证 contraction 一致）
#          q2_K         （K-quant 2-bit super-block·owned real-vector·fused vfmsac dl*q-ml·byte-exact 须证 contraction 一致）
#          q3_K q6_K    （K-quant super-block·owned real-vector·single mul（无 min·byte-exact by construction））
#          mxfp4 nvfp4  （FP4 e2m1 16-entry 码本·owned real-vector·vrgather_vv_i8m1 REGISTER 码本 gather（非 vluxei 内存 gather·无 HW-gather 墙）·E8M0 块标度 / 4×UE4M3 子标度·single mul·byte-exact by construction）
#          iq4_nl iq4_xs（16-entry 非线性码本·owned real-vector·vrgather_vv_i8m1 REGISTER 码本 gather·flat fp16 标度 / super-block signed-6 子标度·single mul·byte-exact by construction）
#          tq2_0 tq1_0  （ternary super-block QK_K=256·owned real-vector·PURE ARITHMETIC 解包（无码本·无 gather）·tq2_0 = 2-bit vsrl/vand·tq1_0 = base-3 vmul_vx(pow3)+((q*3)>>8)·single mul·byte-exact by construction）
#   mode : verify  = build + ZEROVEC objdump 探针 + ZERO-MODEL byte-exact + 3-arm 反空心 (NO TIMING)
#          sanity  = 预测量噪声自检 3 轮
#          measure = cold N=25 2-seed flush
set -uo pipefail
BOARD="${1:-rvv}"; MODE="${2:-verify}"; FMT="${3:-iq3_xxs}"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/../../.." && pwd)"
ASSETS="${DEQUANT_ROW_ASSET_ROOT:-$ROOT/experiments/active/r-dequant}"

REPS=25; S1=0x1357; S2=0xACE2; SV=0xD00D
RDIR=/tmp/bench_cells_dequantize_row_${BOARD}_${FMT}

echo "# HARNESS dequantize_row board=$BOARD mode=$MODE fmt=$FMT assets=$ASSETS"

# Per-format: emitted leaf · standalone driver · optional codebook table · the single-byte
# AoS FAULT anchor (arm D) · the deployed-opponent symbol · the cold working-set nb.
# NB_MEASURE is chosen so k (f32 out = nb*QK) ~= 131072 across formats (> LLC when flushed
# cold); NB_VERIFY sweeps the format's full decode corpus (grid idx / sign sel / quant val).
#   iq3_xxs : grid-codebook super-block (QK=256) · vluxei gather · GEN_SEAL BATCH-WIDE-GATHER
#   q8_0    : NON-GRID flat block (QK=32) · vle8+vsext_vf4+vfcvt+vfmul_vf+vse32 · NO gather ·
#             fault flips the quant byte offset `+ 2` -> `+ 1` (in-bounds, value-changing)
#   q4_0/q5_0 : flat nibble SAFE set (QK=32) · vle8+vand/vsrl+vzext+[q5 5th-bit spread]+vsub+
#             vfcvt+vfmul_vf+vse32 · NO gather · single mul (byte-exact by construction) ·
#             fault flips the nibble byte offset (+2->+1 / +6->+5, in-bounds, value-changing)
#   q4_1/q5_1 : flat nibble FMA set (QK=32) · ...+vfmv_v_f(m)+vfmacc_vf(d)+vse32 · fused
#             mul-add (byte-exact needs contraction consistency: oracle=std::fma, opponent
#             autovec=vfmadd) · fault flips the nibble byte offset (+4->+3 / +8->+7)
case "$FMT" in
  iq3_xxs) LEAFC="kernels/iq3_xxs_dequant.c"; DRVC="dequant_row_driver.cpp"
           TBLC="tables/iq3xxs_tables.h";     GSED='s/0x04040404U/0x04040405U/'
           OPPSYM="dequantize_row_iq3_xxs";   NB_MEASURE=512;  NB_VERIFY=4096 ;;
  iq3_xxs_grid) LEAFC="kernels/iq3_xxs_grid_dequant.c"; DRVC="dequant_row_driver.cpp"
           TBLC="tables/iq3xxs_tables.h";     GSED='s/0x04040404U/0x04040405U/'
           OPPSYM="dequantize_row_iq3_xxs";   NB_MEASURE=512;  NB_VERIFY=4096 ;;
  q8_0)    LEAFC="kernels/q8_0_dequant.c";    DRVC="q8_0_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/+ 2;/+ 1;/'
           OPPSYM="dequantize_row_q8_0";      NB_MEASURE=4096; NB_VERIFY=4096 ;;
  q4_0)    LEAFC="kernels/q4_0_dequant.c";    DRVC="q4_0_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/+ 2;/+ 1;/'
           OPPSYM="dequantize_row_q4_0";      NB_MEASURE=4096; NB_VERIFY=4096 ;;
  q5_0)    LEAFC="kernels/q5_0_dequant.c";    DRVC="q5_0_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/+ 6;/+ 5;/'
           OPPSYM="dequantize_row_q5_0";      NB_MEASURE=4096; NB_VERIFY=4096 ;;
  q4_1)    LEAFC="kernels/q4_1_dequant.c";    DRVC="q4_1_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/+ 4;/+ 3;/'
           OPPSYM="dequantize_row_q4_1";      NB_MEASURE=4096; NB_VERIFY=4096 ;;
  q5_1)    LEAFC="kernels/q5_1_dequant.c";    DRVC="q5_1_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/+ 8;/+ 7;/'
           OPPSYM="dequantize_row_q5_1";      NB_MEASURE=4096; NB_VERIFY=4096 ;;
  q4_K)    LEAFC="kernels/q4_K_dequant.c";    DRVC="q4_K_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 16;/v8 + 17;/'
           OPPSYM="dequantize_row_q4_K";      NB_MEASURE=512;  NB_VERIFY=1024 ;;
  q5_K)    LEAFC="kernels/q5_K_dequant.c";    DRVC="q5_K_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 48;/v8 + 49;/'
           OPPSYM="dequantize_row_q5_K";      NB_MEASURE=512;  NB_VERIFY=1024 ;;
  q2_K)    LEAFC="kernels/q2_K_dequant.c";    DRVC="q2_K_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 80;/v8 + 81;/'
           OPPSYM="dequantize_row_q2_K";      NB_MEASURE=512;  NB_VERIFY=1024 ;;
  q3_K)    LEAFC="kernels/q3_K_dequant.c";    DRVC="q3_K_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 108;/v8 + 107;/'
           OPPSYM="dequantize_row_q3_K";      NB_MEASURE=512;  NB_VERIFY=1024 ;;
  q6_K)    LEAFC="kernels/q6_K_dequant.c";    DRVC="q6_K_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 208;/v8 + 207;/'
           OPPSYM="dequantize_row_q6_K";      NB_MEASURE=512;  NB_VERIFY=1024 ;;
  mxfp4)   LEAFC="kernels/mxfp4_dequant.c";   DRVC="mxfp4_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/8, 12, 0/8, 13, 0/'
           OPPSYM="dequantize_row_mxfp4";     NB_MEASURE=4096; NB_VERIFY=4096 ;;
  iq4_nl)  LEAFC="kernels/iq4_nl_dequant.c";  DRVC="iq4_nl_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/89, 113}/89, 114}/'
           OPPSYM="dequantize_row_iq4_nl";    NB_MEASURE=4096; NB_VERIFY=4096 ;;
  nvfp4)   LEAFC="kernels/nvfp4_dequant.c";   DRVC="nvfp4_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/8, 12, 0/8, 13, 0/'
           OPPSYM="dequantize_row_nvfp4";     NB_MEASURE=2048; NB_VERIFY=2048 ;;
  iq4_xs)  LEAFC="kernels/iq4_xs_dequant.c";  DRVC="iq4_xs_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/89, 113}/89, 114}/'
           OPPSYM="dequantize_row_iq4_xs";    NB_MEASURE=512;  NB_VERIFY=1024 ;;
  tq2_0)   LEAFC="kernels/tq2_0_dequant.c";   DRVC="tq2_0_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 64;/v8 + 63;/'
           OPPSYM="dequantize_row_tq2_0";     NB_MEASURE=512;  NB_VERIFY=1024 ;;
  tq1_0)   LEAFC="kernels/tq1_0_dequant.c";   DRVC="tq1_0_dequant_row_driver.cpp"
           TBLC="";                           GSED='s/v8 + 52;/v8 + 51;/'
           OPPSYM="dequantize_row_tq1_0";     NB_MEASURE=512;  NB_VERIFY=1024 ;;
  *) echo "# HARNESS-VOID bad fmt $FMT (dequantize_row 族: iq3_xxs·iq3_xxs_grid·q8_0·q4_0·q5_0·q4_1·q5_1·q4_K·q5_K·q2_K·q3_K·q6_K·mxfp4·iq4_nl·nvfp4·iq4_xs·tq2_0·tq1_0)"; exit 2 ;;
esac

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
elif [ "$BOARD" = k1 ]; then
  echo "# HARNESS-VOID board=k1 (dequant @k1 部署 gated on ISSUE-105 VLEN256 半宽 — 本格 @rvv 先证隔离该变量)"; exit 2
else
  echo "# HARNESS-VOID unsupported board $BOARD (dequantize_row 族当前仅 rvv)"; exit 2
fi

DRV="$ASSETS/$DRVC"
LEAF="$ASSETS/$LEAFC"
REQ=("$DRV" "$LEAF")
[ -n "$TBLC" ] && { TBL="$ASSETS/$TBLC"; REQ+=("$TBL"); }
for f in "${REQ[@]}"; do
  [ -f "$f" ] || { echo "# HARNESS-VOID missing asset $f"; exit 3; }
done

ssh "$BOARD" "mkdir -p $RDIR/tables" || { echo "# HARNESS-VOID ssh mkdir failed"; exit 3; }
# The board build always compiles $RDIR/dequant_row_driver.cpp (per-format driver source,
# fixed board name) + $RDIR/leaf_dequant.c; the codebook table (grid formats only) rides in
# its relative path so the driver's `#include "tables/..."` resolves.
scp -q "$DRV"  "$BOARD:$RDIR/dequant_row_driver.cpp" || { echo "# HARNESS-VOID scp driver"; exit 3; }
scp -q "$LEAF" "$BOARD:$RDIR/leaf_dequant.c"         || { echo "# HARNESS-VOID scp leaf"; exit 3; }
[ -n "$TBLC" ] && { scp -q "$TBL" "$BOARD:$RDIR/$TBLC" || { echo "# HARNESS-VOID scp table"; exit 3; }; }

ssh "$BOARD" "set -uo pipefail; $ENVSRC cd $RDIR
  SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run_${MODE}.log; : > \$LOG; : > \$SEAL

  echo '# BUILD dequantize_row fmt=$FMT board=$BOARD CC='\$($CC --version|head -1)' march=$MARCH flush=${FLUSH_MB}MiB' | tee -a \$SEAL
  echo '# cpu_md5_before='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$SEAL
  echo '# leaf_md5='\$(md5sum leaf_dequant.c|cut -d' ' -f1)' drv_md5='\$(md5sum dequant_row_driver.cpp|cut -d' ' -f1) | tee -a \$SEAL

  # ---- FAULT leaf: flip ONE source byte of the emitted decode (single-byte delta) ----
  #   grid formats flip a codebook table byte; non-grid q8_0 flips the quant byte offset.
  sed '$GSED' leaf_dequant.c > leaf_dequant_FAULT.c
  D=\$(cmp -l leaf_dequant.c leaf_dequant_FAULT.c 2>/dev/null | wc -l)
  echo '# fault_leaf differing_bytes='\$D' (must be exactly 1)' | tee -a \$SEAL
  if [ \"\$D\" != 1 ]; then echo '# VOID-EXPORT: fault leaf is not a single-byte delta'; exit 6; fi

  # ---- leaf builds (clean + fault) ----
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_dequant.c -c -o leaf.o 2>cc_leaf.err \
    || { echo VOID-BUILD leaf; head -15 cc_leaf.err; exit 3; }
  $CC -O3 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -x c++ leaf_dequant_FAULT.c -c -o leafF.o 2>cc_leafF.err \
    || { echo VOID-BUILD leafF; head -15 cc_leafF.err; exit 3; }

  # ---- driver builds: INJECT 0/1/2 (arm C uses leafF.o) ----
  for I in 0 1 2; do
    $CC -O2 -march=$MARCH -mabi=lp64d $CFLAGS $LDEXTRA -DFLUSH_MB=$FLUSH_MB -DINJECT=\$I \
        dequant_row_driver.cpp -c -o drv\$I.o 2>cc_drv\$I.err \
      || { echo VOID-BUILD drv\$I; head -25 cc_drv\$I.err; exit 4; }
  done
  # NB: link flags INLINED (k1 zsh word-split hazard — kept for parity). REF dequantize_row_iq3_xxs lives in libggml-base.
  $CC drv0.o leaf.o  $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o d_$FMT      2>ld0.err || { echo VOID-BUILD link0; head -20 ld0.err; exit 5; }
  $CC drv1.o leaf.o  $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o d_${FMT}_i1 2>ld1.err || { echo VOID-BUILD link1; head -20 ld1.err; exit 5; }
  $CC drv2.o leaf.o  $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o d_${FMT}_i2 2>ld2.err || { echo VOID-BUILD link2; head -20 ld2.err; exit 5; }
  $CC drv0.o leafF.o $LDEXTRA -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o d_${FMT}_i3 2>ld3.err || { echo VOID-BUILD link3; head -20 ld3.err; exit 5; }
  echo '# linked OK md5='\$(md5sum d_$FMT|cut -d' ' -f1)' FAULTbin md5='\$(md5sum d_${FMT}_i3|cut -d' ' -f1) | tee -a \$SEAL

  # ---- ZEROVEC objdump machine-check on OUR leaf.o (the dequant-axis 头等证据·ISSUE-001 反面) ----
  objdump -d leaf.o > leaf_dis.txt 2>/dev/null
  ZV=\$(awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/) v++; if(\$3 ~ /^vset/) vs++; if(\$3 ~ /^(vlux|vloxei|vrgather)/) gat++; if(\$3 ~ /^(vfmul|vfcvt|vfmacc|vfmadd)/) vf++ } END{printf \"scalar_ins=%d vector_mnemonic=%d vset=%d gather=%d vfp=%d\", ins+0,v+0,vs+0,gat+0,vf+0}' leaf_dis.txt)
  RI=\$(grep -c '__riscv_' leaf_dis.txt); FL=\$(grep -cE '__extendhfsf2|__truncsfhf2' leaf_dis.txt)
  # 真向量 = vector_mnemonic 远大于 vset(不能全是死 vsetvl)·gather 或 vfp > 0
  NONVSET=\$(awk -F'\t' '\$3 ~ /^v/ && \$3 !~ /^vset/ {c++} END{print c+0}' leaf_dis.txt)
  # ★真判据（PR-31·ISSUE-001 反面）= OWNED 源级 __riscv_ 向量 intrinsic 数（在 leaf .c 源里数）。
  #   objdump non_vset_vector 会把宿主 autovec(codegen 抽签)也算进来 —— 那不是我方发射器的控制权。
  #   scalar 基线：源级 owned __riscv_v = 2（皆 vsetvl 死值·= ISSUE-001 铁证）；真向量 emit ⟹ owned ≫ 2。
  OWNED=\$(grep -oE '__riscv_v[a-z0-9_]+' leaf_dequant.c | grep -v vsetvl | sort -u | wc -l)
  OWNEDCALL=\$(grep -cE '__riscv_v[a-z0-9_]+' leaf_dequant.c)
  echo \"# ZEROVEC leaf.o [\$ZV] non_vset_vector=\$NONVSET(autovec+owned·抽签污染) OWNED_src_vec_intrinsics=\$OWNED distinct/\$OWNEDCALL calls(vsetvl 除外·真判据) fp16_libcall=\$FL\" | tee -a \$LOG

  # ---- probe the deployed opponent (ggml $OPPSYM·标量类档 autovec) ----
  objdump --disassemble=$OPPSYM $GGML/libggml-base.so > opp_dis.txt 2>/dev/null
  OZ=\$(awk -F'\t' '\$3 ~ /^[a-z]/ { ins++; if(\$3 ~ /^v/ && \$3 !~ /^vset/) v++ } END{printf \"ins=%d non_vset_vector=%d\", ins+0,v+0}' opp_dis.txt)
  echo \"# OPP $OPPSYM [\$OZ] (标量类档·host autovec = codegen-lottery)\" | tee -a \$SEAL

  # ---- hygiene / single-instance ----
  stray(){ { pgrep -x d_$FMT; pgrep -x d_${FMT}_i1; pgrep -x d_${FMT}_i2; pgrep -x d_${FMT}_i3; } 2>/dev/null | wc -l; }
  for b in d_$FMT d_${FMT}_i1 d_${FMT}_i2 d_${FMT}_i3; do pkill -x \$b 2>/dev/null; done; sleep 0.3
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
  echo '=== [A] CLEAN INJECT=0 (nb=$NB_VERIFY, verify-only) — expect ours-vs-oracle PASS + ggml-vs-oracle PASS + CORPUS COMPLETE ===' | tee -a \$LOG
  run d_$FMT $NB_VERIFY 0 $SV
  echo '=== [B] ANTI-HOLLOW #1 ORACLE-FAULT (INJECT=1, oracle rotates one decoded value) — expect ours-vs-oracle RED ===' | tee -a \$LOG
  run d_${FMT}_i1 $NB_VERIFY 0 $SV
  echo '=== [C] ANTI-HOLLOW #2 DUT-FAULT (INJECT=2, ours[mid]+=1.0f) — expect ours-vs-oracle RED ONLY ===' | tee -a \$LOG
  run d_${FMT}_i2 $NB_VERIFY 0 $SV
  echo '=== [D] ANTI-HOLLOW #3 LEAF single-byte source fault — expect ours-vs-oracle RED ONLY (bites the EMITTED RISC-V) ===' | tee -a \$LOG
  run d_${FMT}_i3 $NB_VERIFY 0 $SV
elif [ \"$MODE\" = sanity ]; then
  echo '=== 3x re-measure noise self-check (seed=$SV nb=$NB_MEASURE N=$REPS) ===' | tee -a \$LOG
  for r in 1 2 3; do echo \"--- round \$r ---\" | tee -a \$LOG; run d_$FMT $NB_MEASURE $REPS $SV $FLUSH_MB; done
else
  echo '=== S1 COLD seed=$S1 (nb=$NB_MEASURE N=$REPS flush=${FLUSH_MB}MiB) ===' | tee -a \$LOG; run d_$FMT $NB_MEASURE $REPS $S1 $FLUSH_MB
  echo '=== S2 COLD seed=$S2 ===' | tee -a \$LOG; run d_$FMT $NB_MEASURE $REPS $S2 $FLUSH_MB
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
