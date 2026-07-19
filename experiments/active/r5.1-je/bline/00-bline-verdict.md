# B线 · 高性能收割 —— 首波交付（r5.1-je）

**worktree base pin = `1d207aab58178c1fbcbe8d09ac189532913b182a`**（`git rev-parse HEAD` == 此 · 核对通过）
census-v2 pin `d173f4c2` = 本 base 的祖先（相隔 2 commit·diff-stat 仅 journal/_attic/census/nvfp4-实验·**零源码改动** ⟹ census 事实在本 pin 全成立）。

**verdict = 收据（输格三步·非 GREEN）**：本波**无 NEW 干净 ≥3 扇出收割面**（byte-exact + cold 翻 PASS）。最强 ≥3 扇出面 = **K-quant decode load-width 杠杆**（8 格）= 已板证的**结构墙**（M=1 arithmetic-intensity-1·微架构非编译器行为）·给三步收据如下。其余 ≥3 扇出面均：**已收割**（grid VLEN256-widening 4 格 / tq1_0+iq2_xxs vec_dot）· **已墙**（grid dequant HW-gather / IME asymmetry）· **被阻**（iq2_xs/iq2_s VLEN256 byte-broken ISSUE-120 / [GAP-P1] 默认 flip 必问）。

---

## 一 · 判决实验自证（emit 级单变量翻转·本 pin 新跑·无板时）

复现 census pkg5 §项2a T-P 链（q3_K@rvv repack-GEMM prefill）——**能力文件单变量翻转 = 换 `-march`**：

```
FIX=test/Conversion/RVV/rvv-emit-quant-contraction-q3-K-repack-gemm-prefill-vlen128.mlir
# θ 提取（承重 half_lanes / numHalves）：
build/weft/bin/weft-opt "$FIX" --weft-rvv-lower-quant-contraction=march=<MARCH> \
  | grep -oE 'half_lanes = [0-9]+|integer_core_lmul = "[a-z0-9]+"'
# 指令级扇出（vwmacc.vx / obj 字节）：
build/weft/bin/weft-opt "$FIX" --weft-rvv-lower-quant-contraction=march=<MARCH> --weft-rvv-lower-to-emitc \
  | mlir-translate-20 --mlir-to-cpp \
  | clang++-20 --target=riscv64-unknown-linux-gnu -march=<MARCH> -ffreestanding -O2 -c -x c++ - -o o.o \
  && llvm-objdump-20 -d --mattr=+v,+zvfhmin o.o | grep -c 'vwmacc\.vx'
```

| c 输入（march） | half_lanes | integer_core_lmul | vwmacc.vx | obj_bytes |
|---|---|---|---|---|
| `rv64gcv_zvfhmin`（VLEN128） | **8** | "mf2" | **2048** | **115784** |
| `rv64gcv_zvl256b_zvfhmin`（VLEN256） | **16** | "mf2" | **1024** | **58808** |

- **翻转 pinned**：march-A `rv64gcv_zvfhmin` / θ-A `half_lanes=8,vwmacc=2048` → march-B `rv64gcv_zvl256b_zvfhmin` / θ-B `half_lanes=16,vwmacc=1024`。ratio(vwmacc)=2.0000==ratio(numHalves)=2/1 **CONSISTENT**（逐字节复中 census pkg5 的 115784/58808）。
- **两杠杆分离（关键）**：march 翻转只动 `half_lanes`（θ2·strip 宽度）；`integer_core_lmul` 两侧恒 **"mf2"**（θ1·[GAP-P1] 默认钉死·march 不改它）。⟹ **VLEN-widening 杠杆 ≠ load-width 杠杆**，是两条独立的承重 θ。

---

## 二 · 收据（输格三步）：K-quant decode load-width 杠杆（≥3 扇出·真墙）

**扇出面 = 8 格 K-quant decode**（5 rvv: q2/q3/q4/q5/q6_K + 3 k1: q3/q5/q6_K）。杠杆 = `integer_core_lmul` mf2→m1（θ18/θ19·decode leaf `typed_repack_gemv_loop_body`）。

1. **objdump 对手**（部署 block-dot `ggml_vec_dot_qX_K_q8_K` / repack-GEVM）= **VLEN 专化 full-unroll·register-resident**（0 scratch store·per-sub-block `vwredsum.vs` 归约·vectorized min-term/scale）。我方 decode leaf = 通用 aux32 核·窄 load（576×vle8 mf2 8B）。
2. **定杠杆** = load-width mf2→m1（vle8 576→288·8B→16B·decode 无 roundtrip ⟹ knob 直击 memory-scheduling·byte-exact 免费·无新 schema）。
3. **板测证伪**（task `07-19-kquant-decode-attack` abbe / `07-18-kquant-decode-scope` a13·2026-07-19）：杠杆 = **真 micro win**（q4_K 1.32× / q6_K 1.82× cold vs 自身基线·byte-exact ULP=0·clang 保调度 spill 8→6·**NOT 脾气墙**）——**但 vs 部署 block-dot 仍 0.47 / 0.49 < 0.8 = 具名-X**。
   - **墙型 = 微架构/结构（非编译器行为）**：perf-stat MEMORY-BOUND confirmed（IPC 0.695 L2→0.297 DRAM 单调塌·LLC-miss bytes/iter == weight bytes = **arithmetic-intensity-1**）。残余 ~2% = **M=1 无行摊**（repack-GEVM interleave-transpose·both stream same weight·load-width 修不了）。
   - **族级泛化**（ISSUE-014 承重四腿）：纯算术地板 `256/(256+fold_ops)` 对 q2_K=0.889 / q3_K/q4_K/q5_K/q6_K=0.941 **全 > 0.8** ⟹ fold@M=1 单独产不出任何具名-X ⟹ 8 格的 <0.8 全由**对手 codegen 结构优势**（VLEN 专化 full-unroll）驱动·非我方 fold。2 板测数据点（q4_K/q6_K vec_dot floor 0.186/0.214·同 stall-bound·q6_K 无 min-term 仍同 floor）已授权 **mechanism-level 具名上报 q2/q3/q5_K 群**（无须逐格补 oracle 走环）。
   - **flip 阻断**：mf2→m1 默认翻转 = [GAP-P1] canon-adjacent（必问/灰区·`RVVLowerQuantContraction.cpp:1255-1257,1271-1279` IRON RULE·abbe 守 conservative mf2 未 flip）·**agent 不自决**。

---

## 三 · 其余 ≥3 扇出面为何不够（具名）

- **VLEN256-widening 杠杆（half_lanes 8→16·§一 自证）**：**已收割 4 grid 格**（iq3_xxs 1.38 / iq3_s 1.21 / iq1_s 1.34 / iq1_m 1.79 @k1·deployed PASS·ISSUE-105 RESOLVED）。余扇出：iq2_xxs@k1 = 成色 upgrade **非新 PASS flip**（裁决2 af60）；**iq2_xs/iq2_s@k1 = byte-broken VLEN256**（ISSUE-120·mism=512·pre-existing·须 per-VLEN 修方可 claim）；**iq4_nl@k1 = codebook-gather 墙**（ISSUE-021·同算子 hand-brick beats 2.35×）。⟹ widening 收割面已耗尽。
- **grid dequant owned 真向量（HW-gather）**：honest-null 真墙（ISSUE-107·三变体全 <0.8：naive 0.181/标量-load 0.33/HW-gather 0.36·对手 HW_GATHER=0 结构 2.8× 优势·裁决3 证伪标量-load lever）。grid 族（iq3_s/iq2*）通用天花板。
- **IME kernel-sym**（q4_0/q4_K@ime）：[CASE-COMPILER-ASYMMETRY] 极端案（对手 = 663-insn 手写 asm 单体 tile）·**e2e 已 cover**（q4_0 tie 1.0088× / q4_K 黄 0.909×·两赛道禁互推）·🔴 禁 inline-asm 绕过。
- **归约 fusion**（tq1_0 aux8-elim / iq2_xs lean-decode）：**format-specific·非通用 ≥3 杠杆**（ISSUE-100 line 407 板上 3-way 隔离证伪「通用 serial-reduce floor lever」）。tq1_0 vec_dot 已 deployed（双板 PASS 0.82·手调 9→10/10→11）；iq2_xs 维持具名-X（floor 0.617·lean-decode lever VLEN128-form·独立 scope + ISSUE-120）。

---

## 四 · 会话卫生
- **改了哪些源文件 = NONE**（纯 emit 级只读跑 weft-opt·未 author 任何 leaf·未碰 capability/front-door 共享文件）。
- **碰的共享文件 = 无**（供主会话合并：无）。
- 工件仅此 md（emit cpp 2MB/1MB 可由 §一 一行命令复生·未留）。**未 git commit / 未 git add**。
- **新 θ / 新焊死 = 0**（未加代码 ⟹ 律2 三谓词 N/A）。
