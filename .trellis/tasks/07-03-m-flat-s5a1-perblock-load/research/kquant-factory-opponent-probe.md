# Research: K-quant factory 对手探针 (2c 预飞 · 只读)

- **Query**: ggml q2_K/q3_K/q4_K/q5_K/q6_K vec_dot 在 RISC-V 上的真实派发路径与手调程度；识别弱带 + 排序；对账我方 constructed；框定 vs-framework 对手身份
- **Scope**: mixed (llama.cpp ggml 只读 + TianChenRV 我方 emit 只读)
- **Date**: 2026-07-05
- **Board 前提**: `ssh rvv` = openEuler / VLEN128 / 64c(见 memory `hardware-test-access`)。VLEN256 分支存在但当前板测不到，凡涉 VLEN256 的弱点均标注"仅 VLEN256 板显现"。

---

## Findings

### Files Found

| File Path | Description |
|---|---|
| `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` | RISC-V 手调 vec_dot 核（6596 行）。K-quant 派发+函数体全在此 |
| `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/quants.c` | 通用 `_generic` scalar fallback（含 aux8 标量 decode） |
| `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch-fallback.h` | `_generic`→public 符号重映射；riscv 块(203–241) |
| `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` | traits 表：`GGML_TYPE_Q5_K.vec_dot = ggml_vec_dot_q5_K_q8_K`(311–313) |
| `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/CMakeLists.txt` | riscv64 build 逻辑(438–513)：加入 arch/riscv/quants.c + march 拼装 |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCKQuant.cpp` | 我方 super-block K-quant emit(4401 行) |
| `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Plugin/RVV/RVVMonolithicBlockDotFamily.h` | SuperBlock vs Flat 族分裂 + loop selector 枚举 |

---

### 1. Factory 真实派发路径（逐格）

派发是**运行时 VLEN 分派**：public `ggml_vec_dot_qX_K_q8_K` 内 `switch (__riscv_vlenb()*8)` 选 vlNNN 变体；`#if defined __riscv_xtheadvector` 走 xtheadvector 变体；两者都无才落 `_generic`。板上 march = `rv64gcv...`(CMakeLists 464–467 `MARCH_STR="rv64gc"+"v"`)，故 `__riscv_v` 恒定义 → **手调 RVV 体一定跑，_generic 不跑**。

| 格 | 派发入口 | 变体覆盖 | 板上真跑(VLEN128) | 实现风格 |
|---|---|---|---|---|
| q2_K | 944–958 | xthead / vl128 / vl256 | vl128 | **手写 inline ASM**(vl128 @719,748) |
| q3_K | 1607–1630 | xthead / vl128 / **vl256 / vl512 / vl1024** | vl128 | **手写 inline ASM**(vl128 @1134,1170) |
| q4_K | 2064–2079 | xthead / vl128 / vl256 | vl128 | **手写 inline ASM**(vl128 @1808，软流水交织 load) |
| **q5_K** | **2081–2197（单体，无分派）** | **无 vlNNN、无 xthead** | 单体 | **plain intrinsics**，fixed vl=32，无 ASM |
| q6_K | 2720–2743 | xthead / vl128 / **vl256 / vl512 / vl1024** | vl128 | **手写 inline ASM**(vl128 @2324) |

**结论**:q2/q3/q4/q6_K 每格都有【手写 inline 汇编 vl128 核 + xtheadvector 变体】；**q5_K 是唯一没有任何 ASM、没有 VLEN 分派、没有 xthead 变体的 K-quant** —— 只有一个 plain-intrinsics 单体(2081)。所有格在板上真跑的都是手调 RVV(非 _generic)。

`_generic`(quants.c) 只在非向量 RISC-V build 触发；结构 = `aux8[QK_K]` 标量 decode + `aux16/aux32` 标量 MAC(q5_K_generic @720–798，q6_K_generic @800–853)。

### 2. 手调深度（有手调格）

- **VLEN-dispatch**:q3_K/q6_K 最全(vl128/256/512/1024)；q2_K/q4_K 只 vl128 + vl256(default 覆盖 256+);**q5_K 零分派**。
- **LMUL 随 VLEN 调**:vlNNN 变体本质=按 VLEN 选 LMUL 填满寄存器。q4_K vl256(1975)用 vl=32/m1(@VLEN256 恰满)，vl128(1770)用 ASM 手排。**q5_K fixed vl=32 + fixed LMUL(m2/m4/m8，@2145/2166/2169)**——@VLEN128 恰满，**@VLEN256 只填一半 = 次优(仅 VLEN256 板显现)**。
- **decode 向量化**:factory 手调核**全部寄存器驻留、无 aux8 中转**。q5_K(2143–2177)、q4_K vl256、q6_K vl256(2456–2481)都在寄存器里 vand/vsrl/vor/vadd_mu 解 nibble/qh，不落 scratch 数组。
- **reduction round-trip / stall**:factory vl256/vl128 intrinsics 体**在内 j-loop 里 vredsum**(串行跨-lane 归约/迭代):q5_K 每 j 两次 vredsum(2172–2175)、q4_K vl256 每 j 两次(2040,2048)、q6_K vl256 每 j 四次(2500–2503)。这是 factory intrinsics 体的共性 stall 点(手写 ASM 体 vl128 则手排规避)。

### 3. ★弱带识别 + 超越概率排序（高→低）

1. **q5_K —— THE 弱带（首打）**。唯一 plain-intrinsics 单体：无 inline ASM(其余四格 vl128 全 ASM)、无 VLEN 分派、无 xthead、fixed vl=32/fixed LMUL(VLEN-blind)、内 j-loop vredsum。@VLEN128 也是"plain intrinsics vs 别人手排 ASM"的异类；@VLEN256 再叠加 fixed-vl 欠填。
2. **q6_K** —— 有 ASM vl128 + register-resident vl256/512/1024，但 decode 最重(6-bit ql+qh 重组 2452–2481)、vl256 每 j 四次 vredsum。手调但归约密。
3. **q2_K** —— ASM vl128 + vl256 only(无 vl512/1024)，VLEN 覆盖略逊 q3/q6。
4. **q4_K** —— ASM vl128 **软流水交织 load(1808 起手排)** = 单核最优；vl256 才回落 plain intrinsics。**factory 最硬**(2c 三目标里最难超越)。
   - q3_K 归入"手调强"梯队(ASM vl128 + 四档 VLEN)，非 2c 直接目标但同属难超越。

**2c 建议先打顺序**:**q5_K → q6_K → q4_K**。(q4_K 虽是 flat 族首翻焦点，但其 vl128 手排 ASM 是 factory 最强单核，留最后。)

### 4. 对账我方 constructed（结构优势 or parity）

我方 super-block emit(RVVToEmitCKQuant.cpp)对 q2/q3/q4/q5/q6_K 均存在，e2e emit-golden 测试在 `test/Target/RVV/qX-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir`。registry(RVVMonolithicBlockDotFamily.h 1422–1444):q4_K/q5_K 共用 `SuperBlockTwoLevelScaleMin` 双累加体(按 weight stride 消歧)、q6_K = `SuperBlockScalesTimesSumi` 单累加、q2_K = `SuperBlockScalarScaleMin`。

我方结构 vs factory：

| 维度 | 我方 constructed | factory 手调 RVV | 判定 |
|---|---|---|---|
| decode 落点 | **aux8[256] scratch**(decode→存 aux8→重读；注释 826 明言"mirroring _generic's aux8") | **寄存器驻留，无 aux8** | 我方**劣势**(多一次 aux8 内存 round-trip) |
| vl 适配 | `__riscv_vsetvl_e8<l8>` **VLEN-adaptive** | q5_K fixed vl=32(VLEN-blind);其余 vlNNN 手选 | 我方**优势**(尤其对 q5_K @VLEN256) |
| LMUL | `integer_core_lmul` 旋钮(SEL-1 widest-anchor-first，默认 mf2→m1→m2) | 每变体 fixed LMUL | 我方**可调**(Win-A 旋钮) |
| reduction | **deferred**:vwmul i16/vwmacc i32 累加进持久 i32 累加器 + 末尾寄存器内 regroup(vslidedown+vadd) 一次(emitQ4_KScaledDotIntoAux32 1175–1187) | intrinsics 体**内 j-loop vredsum**(每 sub-block 归约) | 我方**结构优势**(少 stall) |

**净判**:我方结构=【generic aux8 数据流的向量化版】+【VLEN-adaptive vl / LMUL 旋钮 / deferred reduction】。相对 factory 的**寄存器驻留手调核**，我方 aux8 round-trip 是逆风;真正的超越杠杆=(a) deferred reduction 少 stall、(b) VLEN-adaptive + LMUL 选择,而**非**更精简的数据流。对 q5_K 尤其：factory VLEN-blind + vredsum-in-loop 恰是我方两个杠杆的正面，但我方多背 aux8 round-trip → **净超越需实测,非结构自证**。

### 5. 框定 vs-framework 对手身份（板上真实符号）

- traits 表(ggml-cpu.c 311–313)调 **public `ggml_vec_dot_q5_K_q8_K`**；arch/riscv/quants.c 定义了该 public 符号(2081) → 链接进 libggml-cpu 的是 riscv 手调体，非 `_generic`。arch-fallback.h riscv 块(203–241)**不含** q5_K/q4_K/q6_K 的 `_generic`→public 重映射 = 确认 riscv 覆盖了这些格(只对 nvfp4 等未实现格回落)。
- **确认方式(探针后、上板前先做)**:
  1. `nm -C libggml-cpu.so | grep vec_dot_q5_K` + `objdump -d --start-address=<symaddr>` 反汇编，确认体是 intrinsics(vand/vsrl/vwmul/vredsum) 而非 _generic 的标量 mul，且**无 vlNNN 分支**(q5_K 单体→无 `__riscv_vlenb` 分派)。
  2. 确认 build march 含 `v`(rv64gcv…) 使 `__riscv_v` 生效(CMakeLists 464–467)。
  3. q4_K/q6_K 需额外 `__riscv_vlenb()*8` 确认板走哪个 vlNNN(VLEN128 → vl128 ASM 体)。
- **对手身份定格**:q5_K 对手 = `ggml_vec_dot_q5_K_q8_K`@arch/riscv/quants.c:2081(plain-intrinsics 单体，VLEN128 上无分支)。q4_K 对手 = vl128 手排 ASM 体(@1808);q6_K 对手 = vl128 ASM 体(@2324)。

---

## Caveats / Not Found

- **Board 断层**:当前 `ssh rvv` = VLEN128。q5_K"fixed vl=32 欠填"只在 VLEN256 板显现;VLEN128 上 q5_K 弱点=【plain intrinsics + vredsum-in-loop vs 对手手排 ASM】,非 VLEN 欠填。两条都指向 q5_K 为弱带,但机理随板不同——报数须标 board identity(跨机不可比,见 memory)。
- **我方超越非结构自证**:aux8 round-trip 是我方共性逆风;deferred-reduction 杠杆须警惕 memory `winc-structural-null`(deferred-vs-per-iter 一旦 register-kept 实测 ≈1.00×,可能 wash)。q5_K 净超越=实测问题,本探针只定"softest 对手"。
- **未上板/未反汇编**:本探针纯静态只读。符号反汇编确认(§5)与 microbench 是探针后步骤。
- **spacemit/IME 路**:`GGML_CPU_RISCV64_SPACEMIT`(CMakeLists 444–461)是 K1 IME 独立路,非通用 RVV 板对手,本探针不涉。
- **未逐读**:q2_K/q3_K 全体、q4_K vl128 ASM 全 200 行、我方 q5_K emit 全体(确认与 q4_K 共 loop selector 即止)。如需 q4_K ASM 逐指令软流水分析可加做。
