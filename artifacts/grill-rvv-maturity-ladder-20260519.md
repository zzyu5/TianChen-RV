根据 2026-05-19 上传的 grill 文档和补充对话，这份讨论的真正主线应该改写成：

> **TianChen-RV 要把 RVV 做成一个 route-supported、typed、plugin-owned 的低层执行 surface；dtype、参数、性能调度都必须进入并落在 selected `tcrv_rvv` body 里，而不是藏在 `i32m1` route builder、`tcrv.exec` metadata、EmitC、route id、artifact 名字或测试名里。**

我上一版讲得太抽象，确实漏掉了三个关键细节：**dtype 怎么进入、参数怎么消费、性能层怎么实化**。这三块不是附属问题，而是整个 grill 最重要的工程契约。

---

# 1. 我们最终在讨论什么

不是：

```text
i32m1 add/sub/mul 能不能继续跑
```

也不是：

```text
RVV intrinsic C 怎么直接生成
```

更不是：

```text
Linalg frontend 现在怎么接进来
```

真正讨论的是：

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++
  -> runtime / hardware evidence when claimed
```

也就是说，**权威对象是 selected variant 里的 typed `tcrv_rvv` body**。`tcrv.exec` 负责 envelope 和参数角色；`tcrv_rvv` 负责 RVV vector-level 计算结构；RVV plugin/provider 负责 legality、realization、intrinsic route；EmitC/common 只负责 faithful materialization。文档明确把 `tcrv_rvv` 定义成 RVV-owned vector-level execution IR，抽象层级接近 MLIR Vector：高于一串 intrinsic call，低于 Linalg/tensor/kernel semantics。它不是 high-level matmul/softmax/reduction-kernel IR，也不是 one-op-per-intrinsic wrapper。

---

# 2. `tc rvv` / `tcrv_rvv` 应该是什么样子

最终状态应该是：

```text
selected tcrv.exec RVV variant
  contains:
    typed low-level tcrv_rvv body

tcrv_rvv body expresses:
  vector typed dataflow
  vector load/store
  memory access form
  arithmetic / compare / select / FMA
  reduction primitive / accumulator
  mask / tail / policy
  dynamic VL / AVL relation
  SEW / LMUL / vtype constraints
  runtime ABI value use
  low-level vector control
  realization hints only if consumed by RVV realization
```

它的范围由 **Linalg-like structured kernels** 校准，但不是现在就做 Linalg frontend。Linalg/Vector 在这里的作用是告诉我们低层 RVV surface 最终要覆盖哪些数学和数据移动类别：elementwise、broadcast、reduction、contraction-like accumulation、memory movement、dtype conversion、runtime shape/control 等。文档把这个定义成 RVV maturity 的 coverage target，并明确 representation 必须是 typed low-level `tcrv_rvv` execution bodies，而不是 high-level kernel ops、one-intrinsic wrappers、metadata mega-ops 或 `i32m1` clone batches。

所以 `tc rvv` 的正确心智模型是：

```text
tcrv_rvv 不是：
  tcrv_rvv.matmul
  tcrv_rvv.softmax
  tcrv_rvv.reduction_kernel
  __riscv_vadd_vv_i32m1 wrapper op
  route_id_as_compute

tcrv_rvv 是：
  typed vector value
  vector load/store
  binary / unary / ternary / select / fma
  reduction primitive
  accumulator
  mask / tail
  setvl / vtype
  stripe loop / low-level vector control
  memory movement / layout / conversion
```

对 matmul-like 或 reduction-like work，pre-realized `tcrv_rvv` body 里已经应该是 vector loads、broadcast/movement、FMA/update、accumulator、reduction primitive、mask/tail、store 这些 vector-level 结构。RVV realization 只负责选择 RVV 执行结构，不能从名字、route、artifact 或 C 里“发现” matmul/reduction 语义。

---

# 3. dtype 到底怎么进入

这里最终结论应该非常明确：

> **dtype 不能从 C ABI string、参数名、route id、artifact 名字、测试文件名里推断出来。dtype 必须进入 explicit typed `tcrv_rvv` body。**

完整流动应该是：

```text
source semantics / hand-authored TianChen-RV MLIR
  decides element dtype:
    i32 / f32 / f16 / bf16 / etc.

        ↓

selected tcrv.exec.variant { origin = "rvv-plugin" }

        ↓

tcrv_rvv typed body contains dtype authority:
  !tcrv_rvv.vector<elem = i32, lmul = m1>
  or equivalent typed value/config form

        ↓

tcrv_rvv config/control carries or constrains:
  elem type
  SEW
  LMUL
  tail policy
  mask policy
  VL / AVL
  memory form
  operation kind
  operand form

        ↓

RVV provider derives concrete route:
  op kind + elem dtype + SEW + LMUL + policy + operand form + memory form
    -> RVV intrinsic spelling
    -> C RVV vector type
    -> headers
    -> ABI mapping
```

对你举的例子，Stage 1 后希望变成这种权威结构：

```text
!tcrv_rvv.vector<elem = i32, lmul = m1>
%a = tcrv_rvv.load %lhs[%i], %vl
%b = tcrv_rvv.load %rhs[%i], %vl
%c = tcrv_rvv.binary {kind = add} %a, %b, %vl
tcrv_rvv.store %out[%i], %c, %vl
```

然后 route provider 从结构推导：

```text
kind        = add
elem        = i32
SEW         = 32
LMUL        = m1
operand     = vv
policy      = tail/mask policy
memory form = unit-stride load/store

=> __riscv_vadd_vv_i32m1
=> vint32m1_t
=> riscv_vector.h
```

这就是和旧路径的根本区别。旧路径是：

```text
!tcrv_rvv.i32m1
tcrv_rvv.i32_load
tcrv_rvv.i32_add / i32_sub / i32_mul
tcrv_rvv.i32_store
RVVI32M1ArithmeticSlice
exact __riscv_*_i32m1 spelling
```

文档说得很清楚：旧 `i32m1` path 证明过 executable route 可以存在，但 dtype、LMUL、operation family、body shape、route id、intrinsic spelling 被过早耦合，所以它不是成熟架构，Stage 1 要把它改成 typed RVV route structure；如果旧 route 不能干净适配，就 fail-closed 或删除，而不是 compatibility glue。

---

# 4. 参数怎么传、怎么消费

这个也不是一句“exec 管参数”就够了。正确分工是三层：

```text
tcrv.exec envelope
  declares ABI/runtime roles

selected tcrv_rvv body
  explicitly imports those ABI/runtime values

RVV provider
  consumes typed body + ABI mapping + capability
  and lowers to route
```

更具体：

```text
tcrv.exec envelope
  uses:
    mem_window
    runtime_param
    ABI envelope

  to bind:
    lhs / rhs / out / n
    buffer / scalar
    input / output
    runtime count
    access / ownership
    C ABI spelling / provenance

        ↓

selected tcrv.exec.variant { origin = "rvv-plugin" }

        ↓

tcrv_rvv body
  imports those values explicitly:
    %lhs = tcrv_rvv.runtime_abi_value @lhs
    %rhs = tcrv_rvv.runtime_abi_value @rhs
    %out = tcrv_rvv.runtime_abi_value @out
    %n   = tcrv_rvv.runtime_abi_value @n

  then uses them in real RVV dataflow:
    %vl = tcrv_rvv.setvl %remaining {sew, lmul, policy}
    %a  = tcrv_rvv.load %lhs[%i], %vl
    %b  = tcrv_rvv.load %rhs[%i], %vl
    %c  = tcrv_rvv.binary {kind = add} %a, %b, %vl
    tcrv_rvv.store %out[%i], %c, %vl

        ↓

RVV provider
  validates the body and derives route
```

`tcrv.exec` 可以绑定参数角色，但不能根据参数名叫 `lhs` / `rhs` / `out` 就推断这是 add；也不能根据 C type string 推断 dtype 语义；也不能根据 route id 或 artifact metadata 推断 compute。文档明确说 `tcrv.exec envelope` 绑定 ABI/runtime roles，但不从 parameter names、C type strings、route ids、artifact metadata 推断 compute；`tcrv_rvv body` 通过 explicit binding ops 导入这些 ABI/runtime values，并在 typed dataflow/control/config ops 中使用它们。

所以参数消费的最终判断标准是：

```text
参数角色在 exec envelope 里声明。
参数值在 tcrv_rvv body 里显式 bind/import。
计算语义在 tcrv_rvv typed ops 里表达。
route provider 只根据 typed body + capability + ABI mapping lower。
```

如果某个路径是：

```text
看到参数名 lhs/rhs/out
  -> 猜 add
  -> 猜 i32
  -> 猜 intrinsic
```

那就是错误架构。

---

# 5. Stage 应该怎么分层

## Stage 1：route authority reset

Stage 1 不是“清理一点代码”，也不是“先保住 i32m1 demo”。它的目标是：

```text
replace / fail-close i32m1-as-architecture-authority
```

Stage 1 的最终状态不是：

```text
i32m1 add/sub/mul 又能跑了，所以 Stage 1 完了
```

而是：

```text
没有 active compiler path 再把 i32m1 arithmetic 当成 RVV route authority。
```

更具体地说，Stage 1 要把这个旧权威：

```text
!tcrv_rvv.i32m1
tcrv_rvv.i32_load
tcrv_rvv.i32_add / i32_sub / i32_mul
tcrv_rvv.i32_store
RVVI32M1ArithmeticSlice
exact __riscv_*_i32m1 route
```

替换成更规范的 typed route authority：

```text
typed RVV value/config/body:
  !tcrv_rvv.vector<elem = i32, lmul = m1>
  or equivalent typed value/config representation

generic vector-level ops:
  tcrv_rvv.load
  tcrv_rvv.binary {kind = add}
  tcrv_rvv.store

route provider derives intrinsic from:
  op kind
  element type
  SEW
  LMUL
  policy
  operand form
  memory form
```

也就是：

```text
旧：
  i32m1 route knows it is i32m1 add.

新：
  body says:
    elem = i32
    lmul = m1
    kind = add
    operand form = vv
    policy = ...
  provider derives:
    __riscv_vadd_vv_i32m1
```

文档对 Stage 1 的约束很强：允许 fail-close/delete 旧 i32m1-specific route，允许在现有代码中原地改，只要是真的替换 route authority；不允许用 compatibility wrapper 保存旧架构；不允许把 Stage 1 拖成 high-level frontend、reduction/matmul、测试清理、dashboard、artifact index 或 report 工作。

---

## Stage 2：route-supported RVV expansion

Stage 2 是在 Stage 1 修正后的 typed surface 上扩展能力，不是继续 clone `i32m1`：

```text
Stage 2:
  route-supported RVV coverage expansion
  on corrected typed low-level tcrv_rvv execution surface
```

它要扩展的不是“小 batch”：

```text
i32 done
f32 done
reduction done
```

而是一个 coverage matrix：

```text
dtype / SEW / LMUL / policy
VL / AVL / setvl
mask / tail
memory access forms
arithmetic / compare / select
FMA / update
reduction / accumulator
movement / layout
broadcast / scalar-vector interaction
conversion / dtype
runtime shape/control
structured-kernel data movement needs
```

Stage 2 的成熟判断不是“某几个 intrinsic 打印出来了”，而是：

```text
route-supported units can be represented in typed tcrv_rvv body
RVV plugin can validate/legalize them
RVV provider can derive route or fail closed
EmitC/common can faithfully materialize the route
```

文档把 Stage 2 定成 corrected typed low-level surface 上的 RVV coverage expansion，并明确不要把 dependency order 变成 “i32 done / f32 done” 这种 false completion batch。

---

## Stage 3：extension-family generalization / second-family integration

Stage 3 不是“让 `tcrv.exec` generic”。`tcrv.exec` 从 Stage 1/2 就必须是 generic 的。Stage 3 的真正目标是：

```text
prove one real non-RVV family can use the same common TCRV path
without leaking family-specific branches into core/common
```

最终形态是：

```text
source semantics
  + target capability/profile
  + ABI/runtime envelope
        |
        v

RVV plugin constructs RVV candidate:
  tcrv.exec.variant { origin = "rvv-plugin", requires = [...] } {
    tcrv_rvv body:
      setvl
      load lhs
      load rhs
      add
      store out
  }

Scalar plugin constructs scalar candidate:
  tcrv.exec.variant { origin = "scalar-plugin", requires = [...] } {
    scalar body / scalar boundary:
      scalar loop
  }

Offload plugin constructs offload candidate:
  tcrv.exec.variant { origin = "offload-plugin", requires = [...] } {
    offload body/boundary:
      bind / runtime call / wait
  }

Core pipeline:
  verify capability / legality
  estimate cost / preference
  select variant or build dispatch
  lower selected variant through origin plugin
```

文档明确 Stage 3 maturity proof 是：一个真实非 RVV extension family 使用 `tcrv.exec` 做 envelope/ABI roles，自己拥有 typed body 或 selected boundary，自己拥有 capability/legality/cost/route construction，并通过 `TCRVEmitCLowerableRoute` 和 common EmitC/export machinery lowering，同时避免 family-name leakage into core/common。

---

# 6. exec 的通用性到底怎么理解

`exec` 不是 RVV-specific，也不是未来才 generic。它一直只承载：

```text
kernel
target/capability scope
mem_window
runtime_param
variant
requires
dispatch
fallback
diagnostics
```

真正 plugin-specific 的东西在 selected variant body 或 selected plugin boundary 里：

```text
RVV:
  typed tcrv_rvv dataflow/config/body

IME / TensorExtLite:
  fragment/MMA-style body or boundary

Offload:
  bind/call/wait style runtime-offload body or boundary

Scalar:
  scalar fallback body or boundary
```

文档把这个分工说得很清楚：core materializes variant envelope，比如 `origin`、`requires`、condition、guard、policy、plugin attrs；plugin-owned construction/materialization 才提供 typed extension body 或 selected boundary。core passes 通过 registry-mediated calls 去 verify legality、estimate cost、materialize selected boundary、build emission plan、build EmitC route，不能在 core/common 里 branch on family names 去 synthesize compute。

所以 `exec` 的最终规则是：

```text
tcrv.exec 不拥有 compute。
tcrv.exec 不拥有 dtype authority。
tcrv.exec 不拥有 RVV schedule。
tcrv.exec 只组织 execution envelope、参数角色、capability、variant、dispatch/fallback。
plugin-owned body 才拥有 family-specific compute/config。
```

---

# 7. 性能层到底在哪里

这一块确实是上次最没讲清楚的。最终结论应该是：

> **性能层 = RVV plugin-owned selected-body realization。**
>
> 输入是 selected pre-realized `tcrv_rvv` body + capability + runtime SSA/ABI values + hints/config；输出是 realized `tcrv_rvv` body。不是 EmitC，不是 route string，不是 artifact，不是 dashboard，也不是状态机。

文档明确说当前 concrete RVV path 还太 direct：`RVVEmitCRouteProvider.cpp` 只是 route-building helper，不是 mature performance layer；当前 route builder 直接构造 bounded `i32m1` route，包括 `riscv_vector.h`、type mapping、`__riscv_vsetvl_e32m1`、vector load、arithmetic、store，这只是 route-shape evidence，不是成熟性能架构。正确 pipeline 需要在 selected lowering boundary 之后、emission planning 之前加 selected-body realization slot。

正确顺序是：

```text
materialize plugin variants
  -> verify plugin variant legality
  -> select variants
  -> materialize dispatch runtime guards
  -> check capability requires
  -> materialize selected lowering boundaries
  -> realize selected extension bodies
  -> materialize emission plans
  -> check execution plan coherence
  -> materialize EmitC lowerable routes
```

这个 realization slot 是 common orchestration，但不是 common RVV pass：

```text
common pass:
  find selected variant
  read variant.origin
  registry.lookup(origin)
  call plugin.realizeSelectedVariantBody(...)
  propagate diagnostics
  fail closed

RVV plugin:
  choose / legalize SEW, LMUL, policy
  place dynamic VL / vsetvl
  realize mask/tail behavior
  choose memory forms
  choose register-pressure-safe unroll
  choose pipeline / prefetch structure
  choose accumulator / reduction layout
  rewrite body into realized tcrv_rvv structure
```

补充对话里也明确：这个东西可以复用，但复用的是 **common plugin-dispatch pass slot**，不是 RVV 性能逻辑。common 不知道 `VL/SEW/LMUL`、`vsetvl placement`、mask/tail、prefetch、pipeline、unroll、intrinsic spelling；这些都留在 RVV plugin。

---

# 8. 性能 config 怎么落地

这里最关键的一句是：

```text
config 不是最终产物。
config 必须被 consume 成真实 tcrv_rvv structure。
```

例如：

```text
hint/config:
  preferred_lmul = m2
  unroll = 2
  prefetch_distance = 2
  max_live_vector_groups = 24
```

不能长期停留成：

```text
body attrs:
  {unroll = 2, lmul = m2, prefetch = 2}
```

也不能让 route builder 或 EmitC generator 看到这些 attr 后自己脑补 schedule。

正确落点是：

```text
realized tcrv_rvv body:
  tcrv_rvv.setvl {sew = 32, lmul = m2, tail = ..., mask = ...}
  explicit dynamic stripe loop
  explicit load0/load1/compute0/store0 schedule
  explicit prefetch op or explicit unsupported/no-op decision
  explicit accumulator layout
  explicit mask/tail handling
  explicit unroll/pipeline structure
```

文档明确说：config source 是 RVV plugin-local tuning；hints 可以来自 hand-written TianChen-RV MLIR、future frontend、profile/autotune data、user policy、capability facts，但 none of those hints is route authority。RVV plugin 必须 check and consume。`tcrv.exec`、common passes、EmitC route builders、artifacts、route ids、test names 都不能 invent performance config。RVV realization 是一次性线性转换：`selected pre-realized tcrv_rvv body -> RVV plugin-local realization -> realized tcrv_rvv body`，不是 repeated optimization loop，也不是 state machine/progress/status model。

所以最终性能 contract 是：

```text
before:
  selected pre-realized tcrv_rvv body

input:
  target capability
  runtime SSA / ABI values
  body facts
  optional hints / policy / profile

RVV plugin-local realization:
  legalize / choose:
    SEW
    LMUL
    tail/mask policy
    dynamic VL placement
    memory form
    unroll
    pipeline
    prefetch
    accumulator/reduction layout
    register-pressure-safe structure

after:
  realized tcrv_rvv selected body

then:
  route builder reads structure
  builds TCRVEmitCLowerableRoute
  common materializer lowers to EmitC
  EmitC exports intrinsic C/C++
```

---

# 9. body 必须能表达流水

你的补充里这点非常重要：如果 `tcrv_rvv body` 不能表达 pipeline/unroll/prefetch，那所谓性能层就是空的。

成熟 body 应该能表达：

```text
dynamic setvl loop
SEW / LMUL / policy
stripe / unroll
prefetch
software pipeline
live vector register pressure
reduction / accumulator layout
mask / tail
```

补充文档直接给了 mature body 的例子：

```text
tcrv.exec.variant @rvv_selected {
  %lhs = tcrv_rvv.runtime_abi_value ...
  %rhs = tcrv_rvv.runtime_abi_value ...
  %out = tcrv_rvv.runtime_abi_value ...
  %n   = ...

  tcrv_rvv.stripe_loop %i from 0 to %n {
    %remaining = ...
    %vl = tcrv_rvv.setvl %remaining {
      sew = 32,
      lmul = "m2",
      tail = "agnostic",
      mask = "agnostic"
    }

    tcrv_rvv.prefetch %lhs[%i + prefetch_distance]
    tcrv_rvv.prefetch %rhs[%i + prefetch_distance]

    %a0 = tcrv_rvv.load %lhs[%i], %vl
    %b0 = tcrv_rvv.load %rhs[%i], %vl
    %c0 = tcrv_rvv.add %a0, %b0, %vl
    tcrv_rvv.store %out[%i], %c0, %vl
  }
}
```

如果做软件流水，body 可以表达 explicit scheduled stripes，或者先用 structured schedule/config 表达意图，再由 realization 变成 explicit realized ops。补充对话里已经把两层权责定清楚了：structured schedule/config 表达意图与约束；realized `tcrv_rvv` ops 表达最终真实执行结构；EmitC route 只把 realized ops 翻译成 intrinsic C。

所以判断成熟性能层的标准不是：

```text
能生成某段 intrinsic C
```

而是：

```text
tcrv_rvv body 本身已经显式表达或结构化约束：
  dynamic VL
  LMUL / register pressure
  unroll
  pipeline
  prefetch
  accumulator/reduction layout
  mask/tail
```

补充文档也明确说，现在的 `tcrv_rvv` body 还太窄：`SetVLOp` 只覆盖 bounded SEW/LMUL/policy，`WithVLOp` 只是 VL scope，i32 load/add/store 只是 first slice，还不能表达 dynamic stripe loop、prefetch、pipeline stages、unroll、register pressure budget、multiple accumulators、mask/tail dataflow、general memory forms、broader dtype/SEW/LMUL families。

---

# 10. route builder 和 realization 的关系

这个边界必须写死，否则又会回到旧问题。

```text
realizeSelectedVariantBody:
  modifies / validates selected extension body
  makes it target-realized
  for RVV:
    expands stripes
    inserts prefetch
    realizes accumulators
    places vsetvl
    legalizes SEW/LMUL/policy
    shapes unroll/pipeline

buildVariantEmitCLowerableRoute:
  reads already-realized body
  builds TCRVEmitCLowerableRoute
  does not invent schedule
  does not invent missing compute
```

短期如果没有 realization hook，route builder 可以临时承担一部分 realization，但文档必须标成 transitional implementation，不是 final architecture。否则 route provider 会继续黑箱化，`tcrv_rvv` body 仍然没有承载真实结构。补充文档明确说：`realizeSelectedVariantBody` 改写/确认 selected extension body 已经 target-realized；`buildVariantEmitCLowerableRoute` 读取已经 realized 的 body 构造 route，不应该再发明 schedule。

---

# 11. 最终清晰版：一张总图

我建议把最终状态压成下面这张图：

```text
High-level semantics
  future only; not current Stage 1/2 authority
  Linalg/Vector only calibrate coverage target
        |
        v

tcrv.exec envelope
  owns:
    kernel
    capability scope
    selected variants
    dispatch / fallback
    diagnostics
    mem_window / runtime_param
    ABI/runtime parameter roles

  does not own:
    compute semantics
    dtype authority
    RVV schedule
    intrinsic spelling

        |
        v

selected tcrv.exec.variant { origin = "rvv-plugin" }
  owns selected RVV path under generic exec envelope

        |
        v

typed tcrv_rvv body
  owns RVV vector-level execution:
    explicit ABI value binding
    dtype / typed vector values
    SEW / LMUL / policy constraints
    VL / AVL / setvl
    load / store / memory form
    arithmetic / compare / select / FMA
    reduction / accumulator
    mask / tail
    movement / conversion
    runtime value use
    low-level vector control

        |
        v

RVV plugin-local realization
  one linear transformation:
    pre-realized body
      -> realized body

  consumes:
    target capability
    runtime SSA / ABI values
    selected body facts
    hints / policy / profile

  decides:
    legal SEW / LMUL / policy
    vsetvl placement
    dynamic stripe loop
    memory form
    mask / tail materialization
    register-pressure-safe unroll
    prefetch / software pipeline
    accumulator / reduction layout

  must not:
    change compute semantics
    change dtype semantics
    change parameter roles
    change variant origin
    turn runtime n/AVL into fake constants
    emit direct C

        |
        v

realized tcrv_rvv selected body
  operative RVV execution structure

        |
        v

RVV provider / route builder
  consumes realized body
  validates supported structures
  derives:
    C/RVV vector type
    intrinsic spelling
    headers
    ABI mapping
    TCRVEmitCLowerableRoute payload

  unsupported:
    fail closed

        |
        v

common EmitC materializer
  target-neutral:
    include
    emitc.func
    emitc.for / loop/control shell
    emitc.call_opaque for __riscv_* calls
    artifact/export packaging

  must not:
    choose RVV intrinsic
    invent RVV schedule
    infer compute

        |
        v

RVV intrinsic C/C++
  riscv_vector.h surface

        |
        v

compile / run / runtime evidence
  only required when executable/runtime correctness is claimed
```

文档在 full exec-envelope to intrinsic-C flow 里也就是这么定的：`tcrv.exec` 声明 kernel、capability、variant、dispatch/fallback 和 parameter roles；selected RVV variant 的 body 是 vector-level typed `tcrv_rvv` IR；selected-body realization 调 origin plugin，把 capability、body facts、runtime values、hints 消费成 realized body；emission planning 和 EmitC route materialization 结构化消费 realized body，不得合成缺失 RVV compute 或 invent performance scheduling；RVV route builder 再把 realized body 转成 `TCRVEmitCLowerableRoute`；common route materializer 降成 EmitC；最后 EmitC export 生成 `riscv_vector.h` C/C++。

---

# 12. 最终结论，用更工程化的话说

这份 grill 文档最终应该落成下面这段话：

```text
TianChen-RV RVV-first is not an i32m1 add/sub/mul demo,
not a Linalg frontend phase,
not an intrinsic-wrapper dialect,
and not an EmitC scheduling trick.

The goal is to build a route-supported low-level RVV execution surface inside
selected tcrv.exec RVV variants.

tcrv.exec binds execution envelope and parameter roles.
Typed tcrv_rvv body owns RVV vector-level compute/config/dataflow.
Dtype enters through typed tcrv_rvv values/config, not through ABI strings or
route names.
Runtime parameters are imported by explicit ABI-value binding ops and consumed
by tcrv_rvv load/store/control/dataflow ops.
RVV plugin owns legality, selected-body realization, route support, intrinsic
mapping, type mapping, and fail-closed diagnostics.
Common EmitC/export only performs neutral materialization and packaging.

Stage 1:
  reset route authority.
  Replace or fail-close i32m1-as-architecture-authority.
  Build the corrected typed RVV route surface where route derives from
  op kind + dtype + SEW + LMUL + policy + operand/memory form.

Stage 2:
  expand route-supported RVV coverage on that corrected surface.
  Coverage is calibrated by Linalg-like structured computation classes:
  elementwise, broadcast, reduction, contraction-like accumulation,
  movement/layout, dtype conversion, runtime shape/control.
  Implement RVV plugin-local selected-body realization for performance:
  dynamic VL, vsetvl placement, LMUL/register pressure, unroll, prefetch,
  software pipeline, accumulator/reduction layout, mask/tail.

Stage 3:
  after RVV maturity, prove a real non-RVV family can reuse the same common
  TCRV path.
  Exec remains generic throughout; plugin-specific semantics live in selected
  variant bodies or selected plugin boundaries.

Performance layer:
  performance config is not a state, route, artifact, or dashboard.
  It is consumed by one RVV plugin-local realization step into real tcrv_rvv
  structure.
  EmitC and intrinsic C only faithfully print/lower the realized structure.
```

这才是我认为“更有价值、更清晰的最终状态”：**不是再说 RVV-first 要成熟，而是明确每一种信息的权威位置：参数在 exec envelope 里声明、在 body 里 bind；dtype 在 typed body 里成为 route authority；性能 config 在 RVV plugin realization 里被消费成真实结构；route builder 和 EmitC 只 faithful lower，不再发明语义或调度。**
