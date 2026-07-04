#!/usr/bin/env bash
# board_ab.sh -- constitutional paired A/B for q8_0 vec_dot micro-fixed.
# Runs ON the board. Compiles our kernel + faithful ggml factory with the SAME
# board clang / same flags (fp16 path cancels), pins to one core, interleaves
# our/factory, prints raw per-invocation kernel-ns. Two full passes => T-N floor.
#
# args: $1=RDIR  $2=OUR_KERNEL_CPP_basename  $3=CORE  $4=LABEL(m1|m2)  $5=DO_WINA(1|0)
#       $6=EXP_VLEN(target T-cell VLEN, e.g. 128|256 — REQUIRED by preflight gate 4)
set -u
RDIR="$1"; OURK="$2"; CORE="${3:-4}"; LABEL="${4:-mX}"; DO_WINA="${5:-0}"; EXP_VLEN="${6:-}"
cd "$RDIR"
# toolchain policy (实验总纲v1 §1 第9条): perf 测量双板统一最新稳定 clang (当前 clang-20).
C=$(command -v clang-20 || command -v clang-19 || command -v clang-18 || command -v clang-17 || command -v clang)
# march 政策 (§1 第9条): 必须含板全能力 (zfh/zvfhmin/zb*), Phase 1 由 hwprobe/cpuinfo 生成
# 后经 TCRV_MARCH 注入; preflight gate1 fail-closed 校验其完整性. 默认仅 rv64gcv_zfh (最小).
MARCH="${TCRV_MARCH:-rv64gcv_zfh}"
FLAGS="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=on --rtlib=compiler-rt"
# driver perf knobs: n=4096 (128 blocks/dot), 256 verify trials, 60000 timed iters best-of-7
DN=4096; DT=256; DI=60000
ROUNDS=12

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "clang=$C  $($C --version | head -1)"
echo "march=$MARCH  flags=$FLAGS"
echo "nproc=$(nproc)  pinned_core=$CORE"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/cpuinfo_max_freq 2>/dev/null)"
echo "libc=$(ls -l /lib/libc.so* 2>/dev/null | head -1; getconf GNU_LIBC_VERSION 2>/dev/null)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"

freq() { cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

# ---- compile: SAME clang/flags for our kernel AND factory ----
echo "== COMPILE =="
$C $FLAGS -x c++ "$OURK"           -c -o k_ours.o    2>cc_ours.err   && echo "k_ours.o OK"    || { echo "COMPILE_FAIL ours"; cat cc_ours.err; exit 20; }
$C $FLAGS -x c++ kernel_factory.c  -c -o k_factory.o 2>cc_factory.err&& echo "k_factory.o OK" || { echo "COMPILE_FAIL factory"; cat cc_factory.err; exit 21; }
$C $FLAGS -x c q8_0_verify_driver.c -c -o driver.o   2>cc_driver.err && echo "driver.o OK"    || { echo "COMPILE_FAIL driver"; cat cc_driver.err; exit 22; }
$C $FLAGS driver.o k_ours.o    -lm -o driver_ours    2>ld_ours.err   && echo "driver_ours OK"    || { echo "LINK_FAIL ours"; cat ld_ours.err; exit 23; }
$C $FLAGS driver.o k_factory.o -lm -o driver_factory 2>ld_factory.err&& echo "driver_factory OK" || { echo "LINK_FAIL factory"; cat ld_factory.err; exit 24; }
if [ "$DO_WINA" = "1" ]; then
  $C $FLAGS -x c++ kernel_m2.cpp -c -o k_m2.o 2>cc_m2.err && echo "k_m2.o OK" || { echo "COMPILE_FAIL m2"; cat cc_m2.err; }
  $C $FLAGS driver.o k_m2.o -lm -o driver_m2 2>ld_m2.err && echo "driver_m2 OK" || echo "LINK_FAIL m2"
fi

# ==== FAIL-CLOSED PREFLIGHT (P2c 反混淆前置门; 在任何 A/B 测量之前; 实验总纲v1 §1 第9/10条) ====
# 每次跑重教规则=对手构建保真 + 工具链一致. 任一门 FAIL 即 exit 30, 不产任何测量数.
OD=$(command -v llvm-objdump-20 || command -v llvm-objdump-19 || command -v llvm-objdump-18 || command -v llvm-objdump-17 || command -v llvm-objdump || command -v objdump)
pf_fail() { echo "PREFLIGHT_FAIL($1): $2"; exit 30; }
echo "== PREFLIGHT (fail-closed 4-gate) =="
# (1) march 完整性: 板实测扩展 (hwprobe/cpuinfo isa) 中的关键扩展必须全在编译 march 里, 否则报缺哪个.
BOARD_ISA=$(grep -m1 -i isa /proc/cpuinfo 2>/dev/null | tr 'A-Z' 'a-z')
[ -n "$BOARD_ISA" ] || pf_fail march "cannot read /proc/cpuinfo isa string"
for ext in zfh zvfhmin zvfh zba zbb zbs; do
  if echo "$BOARD_ISA" | grep -Eq "(_|^| )$ext(_| |\$)" && ! echo "$MARCH" | grep -q "$ext"; then
    pf_fail march "board has '$ext' but compile march '$MARCH' omits it (board isa: $BOARD_ISA)"
  fi
done
echo "PREFLIGHT(1) march-complete: OK ($MARCH covers board critical exts)"
# (2) 双侧 objdump libcall 扫描: 任一侧核热路径含软浮点 fp16 libcall = 这次 P2c 混淆的指纹 -> FAIL.
for obj in k_ours.o k_factory.o; do
  if $OD -d "$obj" 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee|__extendhfxf2'; then
    pf_fail libcall "$obj 含软浮点 fp16 libcall (crippled build; march 需含 zfh/zvfhmin)"
  fi
done
echo "PREFLIGHT(2) libcall-free: OK (ours & factory 均无 __extendhfsf2 类软浮点 libcall)"
# (3) 同编译器断言: 两侧同一 clang + 同旗标 (本脚本单一 \$C/\$FLAGS 保证; 断言其已解析且为 clang).
[ -n "$C" ] || pf_fail compiler "no clang resolved (policy: clang-20)"
CVER=$($C --version | head -1)
echo "$CVER" | grep -qi clang || pf_fail compiler "resolved compiler is not clang: $C ($CVER)"
echo "PREFLIGHT(3) same-compiler: OK (both sides = $CVER; flags='$FLAGS')"
# (4) 指纹-格匹配: 板实测 VLEN 必须 == 目标 T-格坐标 EXP_VLEN, 否则指纹与格不一致 -> FAIL.
[ -n "$EXP_VLEN" ] || pf_fail fp-cell "EXP_VLEN(\$6) 未提供; 无法校验指纹-格匹配 (传目标 T-格 VLEN, 如 128|256)"
ACT_VLEN=$(taskset -c $CORE ./driver_ours $MARCH 64 1 1 2>/dev/null | sed -n 's/.*VLEN(bits)=\([0-9]*\).*/\1/p' | head -1)
[ -n "$ACT_VLEN" ] || pf_fail fp-cell "无法从 driver_ours 读出板实测 VLEN"
[ "$ACT_VLEN" = "$EXP_VLEN" ] || pf_fail fp-cell "板实测 VLEN=$ACT_VLEN != 目标 T-格 EXP_VLEN=$EXP_VLEN"
echo "PREFLIGHT(4) fingerprint<->T-cell: OK (board VLEN=$ACT_VLEN == target $EXP_VLEN)"
echo "== PREFLIGHT PASS (4/4 gates green) =="

# ---- objdump seal (instruction封) ----
echo "== OBJDUMP SEAL ours ($LABEL) =="
$OD -d --no-show-raw-insn k_ours.o 2>/dev/null | sed -n '/tcrv_emitc/,/ret/p' | grep -E ':\s' | head -60
echo "== OBJDUMP SEAL factory =="
$OD -d --no-show-raw-insn k_factory.o 2>/dev/null | sed -n '/tcrv_emitc/,/ret/p' | grep -E ':\s' | head -60

# ---- correctness (bit-exact vs pinned oracle for ours; premul+fmaf for factory) ----
echo "== CORRECTNESS ours =="
taskset -c $CORE ./driver_ours $MARCH $DN $DT 2000 2>&1 | grep -E 'exact-match|VERDICT|property:|mutation caught|VLEN\(bits\)'
echo "== CORRECTNESS factory =="
taskset -c $CORE ./driver_factory $MARCH $DN $DT 2000 2>&1 | grep -E 'exact-match|VERDICT|property:|mutation caught'

# ---- paired A/B, 2 passes (pass2 => T-N between-run floor) ----
run_ns() { taskset -c $CORE "./$1" $MARCH $DN $DT $DI 2>/dev/null | awk '/^kernel/{print $3}'; }
# warmup (dropped)
taskset -c $CORE ./driver_ours $MARCH $DN $DT $DI >/dev/null 2>&1
taskset -c $CORE ./driver_factory $MARCH $DN $DT $DI >/dev/null 2>&1

for PASS in 1 2; do
  echo "== ABPASS $PASS =="
  for r in $(seq 1 $ROUNDS); do
    fo=$(freq); no=$(run_ns driver_ours)
    ff=$(freq); nf=$(run_ns driver_factory)
    echo "AB pass=$PASS round=$r side=ours    kernel_ns=$no freq_khz=$fo"
    echo "AB pass=$PASS round=$r side=factory kernel_ns=$nf freq_khz=$ff"
  done
done

# ---- winA (m1 vs m2, pure LMUL, both our pipeline) ----
if [ "$DO_WINA" = "1" ] && [ -x ./driver_m2 ]; then
  echo "== WINA (ours=$LABEL vs m2) =="
  taskset -c $CORE ./driver_m2 $MARCH $DN $DT $DI >/dev/null 2>&1
  for r in $(seq 1 $ROUNDS); do
    f1=$(freq); n1=$(run_ns driver_ours)
    f2=$(freq); n2=$(run_ns driver_m2)
    echo "WINA round=$r side=$LABEL kernel_ns=$n1 freq_khz=$f1"
    echo "WINA round=$r side=m2    kernel_ns=$n2 freq_khz=$f2"
  done
fi
echo "== DONE =="
