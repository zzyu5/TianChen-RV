# TianChen-RV RVV Gearbox Autotuning Pass：面向 RV 变体的档位化向量执行自调优方案 v3

## 0. 结论

这个方向应该做成 **MLIR pass pipeline**，而不是一个抽象“realization 概念”。

更准确地说：

```text
TianChen-RV RVV Gearbox Autotuning Pass
= 一组 TianChen-RV/MLIR pass
= 把 pre-realized / tunable tcrv_rvv MLIR
  pass 成多个可执行候选 tcrv_rvv MLIR variant
= 通过静态资源模型 + 实机/profile feedback 选择最优档位
= 最终保留一个或多个 selected realized tcrv_rvv variants
= 再由 RVV route provider faithful lowering 到 intrinsic C / backend route
```

它的核心不是“做一个 LMUL hint”，也不是“照搬某个专利”，而是：

> **把 RVV 的 SEW/LMUL/VL/unroll/tile/accumulator/pipeline/register-pressure/remat/prefetch 变成 TianChen-RV 自己可表达、可枚举、可验证、可实测、可缓存的 MLIR schedule space。**

这就是 RISC-V/RVV 版的 Triton-style autotuning：Triton 面向 GPU 调 BLOCK_SIZE、num_warps、num_stages；TianChen-RV 面向 RVV/RV 变体调 `e32,m1`、`e32,m2`、`e32,m4`、`e32,m8`、tile、unroll、vsetvl region、accumulator layout、pipeline、prefetch、remat 策略。

---

## 1. 为什么它必须是 pass

MLIR 的基本工作方式就是 pass pipeline：一个 pass 读入 IR，做分析、重写、lowering 或 canonicalization，然后输出新的 IR。TianChen-RV 本来就在 MLIR 技术栈里，所以这个优化应该是：

```text
MLIR in  ->  MLIR out
```

更具体：

```text
pre-tuned tcrv_rvv body
  -> tcrv-rvv-build-gear-space
  -> tcrv-rvv-prune-gear-space
  -> tcrv-rvv-materialize-gear-candidates
  -> tcrv-rvv-profile-gears / static-select-gears
  -> tcrv-rvv-realize-selected-gear
  -> realized tcrv_rvv body
```

所以，“realization”不是和 pass 对立的东西。**realization 是某个 pass 的工作内容**：把还没有最终决定的 schedule/config/tuning space，落成具体、可 lower 的 `tcrv_rvv` MLIR structure。

---

## 2. 创新点重写：不是 LMUL pass，而是 RVV Gearbox Autotuning

### 2.1 创新点一：把 RVV 的“档位”作为 compiler-level search space

RVV 的关键特点是运行时可设置 `vtype/vl`：

```text
SEW: e8/e16/e32/e64
LMUL: m1/m2/m4/m8
VL: runtime chosen by vsetvl/vsetvli
policy: tail/mask agnostic or undisturbed
```

对同一个 kernel，例如 elementwise、reduction、GEMM micro-kernel、conv micro-kernel，TianChen-RV 可以生成不同档位：

```text
e32,m1 + unroll=4 + acc=4
e32,m2 + unroll=2 + acc=4
e32,m4 + unroll=1 + acc=2
e32,m8 + unroll=1 + acc=1
```

这些不是 metadata，而是不同的 MLIR candidate body。每个 candidate 会真的生成不同的 RVV intrinsic type、不同的 `vsetvl`、不同的 loop body、不同的 live range。

这就是“真实切档位”。

### 2.2 创新点二：资源约束的 autotuning，而不是盲目暴力搜索

普通 autotune 很容易爆炸：LMUL × unroll × tile × pipeline × prefetch × accumulator × dtype × shape。

TianChen-RV 的优势是它有 typed `tcrv_rvv` body，因此可以先做静态资源模型，把明显不可能高效的候选剪掉：

```text
available_vector_groups(lmul)
peak_live_vector_groups(candidate)
EMUL / widening pressure
mask/v0 usage
accumulator count
load/store live range
vsetvl region count
expected spill risk
```

剪枝规则示例：

```text
if peak_live_groups > available_groups:
    reject candidate

if widening op causes EMUL > 8:
    reject candidate

if vsetvl switching too frequent:
    penalize candidate

if e32,m8 gives only 4 logical groups and body needs 7 groups:
    reject or reduce unroll/accumulator
```

这比“全试一遍”更像 compiler，也更像论文创新。

### 2.3 创新点三：用 emitted assembly feedback 解决“我们不直接控制物理寄存器”的问题

如果 TianChen-RV 当前最后走 RVV intrinsic C，那么最终物理寄存器分配确实由 clang/gcc 的 RISC-V 后端完成。这不是问题，反而可以变成系统优势。

TianChen-RV 可以做：

```text
for each candidate gear:
  emit intrinsic C
  compile to asm/object
  inspect asm:
    count vsetvli/vsetivli
    count vector stack spill/reload
    count scalar spill/reload
    stack frame size
    loop body instruction count
  optionally run benchmark
  cache best gear
```

也就是说，我们不假装知道下游 RA 的全部细节，而是把下游 compiler 当作真实 evaluator。这个 feedback 比单纯静态模型更强。

最终选择可以基于：

```text
score = measured_latency
      + penalty(vector_spill_count)
      + penalty(vsetvli_count)
      + penalty(code_size)
      + penalty(stack_frame)
```

这就是 **compiler-in-the-loop RVV autotuning**。

### 2.4 创新点四：TianChen-RV 的抽象让 tuning 不再是 C intrinsic 宏地狱

手写 intrinsic C 很难做这个，因为 RVV C intrinsic 的类型本身携带 LMUL：

```c
vfloat32m1_t
vfloat32m2_t
vfloat32m4_t
vfloat32m8_t
```

从 m1 切到 m4，不是改一个常量，而是整个数据流类型、load、store、compute intrinsic 都变了。手写代码通常会变成模板爆炸或宏爆炸。

TianChen-RV 的 typed `tcrv_rvv` 可以把这些变成 IR 层的候选生成：

```text
body says:
  elem = f32
  candidate lmul = {m1,m2,m4,m8}
  op = fma/add/load/store

provider derives:
  vfloat32m1_t / vfloat32m2_t / ...
  __riscv_vle32_v_f32m1 / __riscv_vle32_v_f32m2 / ...
  __riscv_vfmacc_vv_f32m1 / __riscv_vfmacc_vv_f32m2 / ...
```

这就是“我们作为 kernel compiler 后端”的价值：不是写一份 intrinsic C，而是生成一族候选 intrinsic C / backend routes。

---

## 3. 和 TianChen-RV 机制怎么真正结合

TianChen-RV 的当前 RVV-first path 应该保持：

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ or equivalent backend representation
```

Gearbox Autotuning Pass 插在 selected `tcrv_rvv` body 到 route provider 之间：

```text
selected pre-tuned tcrv_rvv body
  -> Gearbox Autotuning Pass Pipeline
  -> selected realized tcrv_rvv gear body
  -> RVV route provider
  -> EmitC / intrinsic C
```

关键边界：

```text
tcrv.exec:
  只负责 envelope、参数角色、variant、dispatch，不决定 LMUL。

tcrv_rvv:
  表达 typed vector dataflow、VL/SEW/LMUL constraints、load/store、fma/reduction、mask/tail。

Gearbox pass:
  枚举/剪枝/实测/选择档位，把选择写回真实 tcrv_rvv body。

RVV route provider:
  只读已经 selected/realized 的 body，推导 intrinsic，不再发明 schedule。
```

---

## 4. Pass pipeline 设计

### 4.1 Pass 0：`tcrv-rvv-identify-tunable-regions`

输入：已有 selected `tcrv_rvv` body。

任务：识别可以 tune 的 region：

```text
elementwise stripe loop
broadcast + elementwise
reduction
contraction-like accumulation
GEMM micro-kernel
conv sliding-window micro-kernel
memory movement / dtype conversion
```

输出：给 region 打上 tune boundary，但不决定档位。

```mlir
tcrv_rvv.tunable_region @kernel_main {
  // vector-level body
}
```

### 4.2 Pass 1：`tcrv-rvv-build-gear-space`

为 region 生成候选空间：

```text
SEW: derived from dtype, e.g. e32
LMUL: m1/m2/m4/m8
unroll: 1/2/4/8
accumulator count: 1/2/4/8
prefetch distance: 0/1/2/4
pipeline stages: 1/2/3
remat/sink policy: none/splat/const/index/cheap-chain
vsetvl policy: per-loop/per-block/fused-region
```

候选可以先表示为 `tcrv_tune.space`：

```mlir
tcrv_tune.space @saxpy_gears key = ["target", "dtype", "N_bucket"] {
  tcrv_tune.variant @e32_m1_u4 attributes {sew = 32, lmul = "m1", unroll = 4}
  tcrv_tune.variant @e32_m2_u2 attributes {sew = 32, lmul = "m2", unroll = 2}
  tcrv_tune.variant @e32_m4_u1 attributes {sew = 32, lmul = "m4", unroll = 1}
  tcrv_tune.variant @e32_m8_u1 attributes {sew = 32, lmul = "m8", unroll = 1}
}
```

### 4.3 Pass 2：`tcrv-rvv-prune-gear-space`

用 legality + resource model 剪枝。

核心估计：

```text
available_groups = floor((NumVRegs - ReservedVRegs) / LMUL)
peak_live_groups = max live vector group demand over region
```

注意 widening/narrowing 的 EMUL：

```text
normal op:    EMUL = LMUL
widening op:  EMUL may be 2 * LMUL
narrowing op: source/dest EMUL differs
```

如果 `LMUL=m8` 且 widening 需要 `EMUL=16`，直接 illegal。

### 4.4 Pass 3：`tcrv-rvv-materialize-gear-candidates`

把候选变成真实 body。

```text
candidate config
  -> clone region
  -> choose concrete setvl {sew,lmul,policy}
  -> choose unroll/accumulator layout
  -> place vsetvl region
  -> materialize prefetch/pipeline
  -> sink cheap splat/const/index values
  -> output candidate tcrv_rvv body
```

生成例子：

```mlir
tcrv.exec.variant @saxpy_e32_m4_u2 attributes {origin = "rvv-plugin", gear = "e32_m4_u2"} {
  tcrv_rvv.stripe_loop %i from 0 to %n step dynamic {
    %vl = tcrv_rvv.setvl %remaining {sew = 32, lmul = "m4", tail = "agnostic", mask = "agnostic"}
    // unroll/pipeline/load/compute/store structure here
  }
}
```

### 4.5 Pass 4：`tcrv-rvv-profile-gears`

两种模式：

#### AOT build-time tuning

```text
compile candidates
run microbench on target / ssh-rvv
collect latency and asm metrics
cache best config
```

#### JIT/runtime tuning

```text
first time shape appears:
  benchmark candidate subset
  choose best
  cache by key
next time:
  direct dispatch to cached best
```

缓存 key：

```text
target_id
microarch_id
VLEN/ELEN
supported extensions
kernel signature
shape bucket
input dtype / layout
```

### 4.6 Pass 5：`tcrv-rvv-select-gear`

把 tuning space 替换成 selected variant：

```mlir
tcrv.exec.dispatch @saxpy {
  // optional multi-version dispatch
  tcrv.exec.case @e32_m4_u2 if shape_bucket == "large"
  tcrv.exec.case @e32_m1_u4 if shape_bucket == "small"
}
```

或者 AOT 只保留一个 best body：

```mlir
tcrv.exec.variant @rvv_selected attributes {gear = "e32_m4_u2"} {
  // realized tcrv_rvv body
}
```

---

## 5. 成本模型：先能用，再变聪明

第一版不要上来就做复杂 ML 模型。先做：

```text
static_score =
    compute_ops / estimated_vector_throughput
  + memory_bytes / bandwidth
  + vsetvl_count * vsetvl_cost
  + pressure_penalty
  + spill_risk_penalty
  + code_size_penalty
```

其中：

```text
pressure_penalty = max(0, peak_live_groups - available_groups) * big_number
```

如果有 emitted asm：

```text
asm_score =
    vector_spill_count * huge_penalty
  + scalar_spill_count * medium_penalty
  + vsetvli_count * cost
  + loop_instruction_count
  + stack_frame_size * cost
```

如果有 runtime：

```text
final_score = measured_cycles_or_ns
```

优先级：

```text
runtime measurement > emitted asm metrics > static cost model
```

---

## 6. 第一批 kernel 选择

不要一开始做全模型。先做最能体现 LMUL trade-off 的 kernel：

```text
1. vector add / saxpy / relu
   低寄存器压力，预计大 LMUL 有优势。

2. reduction sum / max
   有 accumulator 和 horizontal reduction，LMUL/unroll 有明显 trade-off。

3. matmul micro-kernel
   accumulator 多，寄存器压力高，m8 可能翻车，m1/m2/m4 需要 tuning。

4. conv1d/conv2d micro-kernel
   sliding window + weight reuse + accumulator，适合展示 pressure-aware tuning。

5. fused op: matmul + bias + relu / conv + relu
   展示跨阶段统一档位、避免中间 materialization。
```

---

## 7. 为什么这比两个专利强

两个专利可以当灵感，但不该作为技术底座。

### 专利方向 A 的局限

“算子 LMUL 数据库 + 计算图协商”有意义，但太高层，容易变成表格驱动，无法真正控制低层 schedule。

TianChen-RV Gearbox 的改进：

```text
不是只给 Conv/ReLU 打 LMUL 注解，
而是生成真实的 tcrv_rvv candidate bodies，
并通过 static model + emitted asm + runtime feedback 选择。
```

### 专利方向 B 的局限

“寄存器压力下 remat vs spill”有意义，但如果当前不拥有物理 register allocator，直接承诺替代 spill 不稳。

TianChen-RV Gearbox 的改进：

```text
不假装自己控制下游 RA，
而是通过候选 body + asm feedback 观测真实 spill，
把 “m8 是否真的翻车” 变成可测量事件，
再回选 m4/m2 或改 unroll/remat policy。
```

这更工程真实，也更像高性能系统论文。

---

## 8. 和 Triton 的类比

Triton 的 autotune 思想是：给一个 kernel 多个 config，用 key 触发不同 shape/参数下重新评估，然后选择最快配置。

TianChen-RV 的类比：

| Triton / GPU | TianChen-RV / RVV |
|---|---|
| BLOCK_SIZE | tile size / stripe size |
| num_warps | LMUL / unroll / accumulator count |
| num_stages | software pipeline stages |
| shared memory layout | memory form / alignment / prefetch |
| tensor core mapping | RVV op form / custom RV variant intrinsic |
| autotune key | target + VLEN + shape bucket + dtype + layout |
| generated PTX/SASS profile | emitted RVV asm + runtime profile |

论文表达可以是：

> **Triton made GPU kernel schedules tunable at the language/compiler level; TianChen-RV Gearbox makes RVV vector configuration and schedule tunable at the MLIR selected-body level.**

---

## 9. 真实可行的开发路线

### Phase 1：只做 static candidate generation + static pruning

目标：能看到同一 kernel 生成 `e32,m1/m2/m4/m8` 四个 body。

验收：

```text
mlir-opt --tcrv-rvv-build-gear-space
mlir-opt --tcrv-rvv-prune-gear-space
mlir-opt --tcrv-rvv-realize-gear=e32,m4
```

### Phase 2：接 EmitC/intrinsic C，编出不同版本

目标：每个 gear 都能生成不同 intrinsic C。

验收：

```text
e32,m1 -> vfloat32m1_t + __riscv_*_f32m1
e32,m2 -> vfloat32m2_t + __riscv_*_f32m2
e32,m4 -> vfloat32m4_t + __riscv_*_f32m4
e32,m8 -> vfloat32m8_t + __riscv_*_f32m8
```

### Phase 3：asm feedback

目标：统计真实后端结果。

```text
vsetvli count
vector spill/reload count
stack frame size
loop instruction count
```

### Phase 4：runtime autotune

目标：在 ssh-rvv / real board 上跑候选并缓存 winner。

输出：

```json
{
  "target": "tianchen-rvv-vlen256",
  "kernel": "saxpy_f32",
  "shape_bucket": "N>=4096",
  "best_gear": "e32_m4_u2_pf1",
  "latency_ns": 12345,
  "asm": {
    "vsetvli": 1,
    "vector_spill": 0
  }
}
```

### Phase 5：跨 RV 变体推广

只换 target capability / hardware evaluator：

```text
标准 RVV
TianChen RVV-like variant
reduced vector register variant
custom dot/conv extension
future IME/TensorExtLite
```

每个 plugin 定义自己的 gear space：

```text
RVV:      SEW/LMUL/VL/unroll/pipeline
IME:      tile/fragment/accumulator/scratchpad
Offload:  chunk/DMA/staging/call-wait overlap
Scalar:   unroll/GPR pressure/remat
```

---

## 10. 最终论文题目建议

中文：

```text
TianChen-RV Gearbox：面向 RISC-V 向量变体的 MLIR 级档位化内核自调优编译框架
```

英文：

```text
TianChen-RV Gearbox: MLIR-Level Gear-Based Autotuning for RISC-V Vector Kernel Compilation
```

更硬一点：

```text
Compiler-in-the-Loop RVV Gear Autotuning: SEW/LMUL/VL-Aware Kernel Generation for RISC-V Variants
```

---

## 11. 一句话定位

> **TianChen-RV Gearbox 不是一个 LMUL hint pass，而是一个 MLIR pass pipeline：它把 RVV 的 SEW/LMUL/VL/unroll/tile/pipeline/register-pressure 变成可搜索的 kernel execution space，通过静态资源模型、下游汇编反馈和实机 profile 找到最佳档位，并把选择实化为 typed `tcrv_rvv` body。**
