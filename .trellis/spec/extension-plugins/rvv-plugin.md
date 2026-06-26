# RVV Plugin

RVV 是 TianChen-RV 当前的主真实硬件 family，也是第一个走通 typed-body → plugin-owned route → 公共 EmitC → `ssh rvv` 证据的完整 family。它是 N1/N2/N3 的第一个证据点，但 **RVV-first ≠ RVV-only**：同一套 plugin 协议要能让 IME / offload 复用（N2）。

durable 路径（authority chain，全项目唯一权威版本）：

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv vector-level body
  -> RVV plugin-owned legality / selected-body realization(Gearbox) / route provider
  -> TCRVEmitCLowerableRoute
  -> 公共 EmitC materializer
  -> RVV intrinsic C/C++
  -> target artifact
  -> ssh rvv evidence（当 runtime/correctness/performance 被主张时）
```

> 本文件只描述 RVV 插件的**稳定边界**。具体覆盖到第几个算子、哪个候选在调、measurement 跑成什么样——是状态，进 task/journal，不进这里。

## Authority Placement

谁拥有什么。这是 RVV 线最容易被 AI 搞错的地方（从 ABI 名/route id/artifact 名反推 compute）。底层规则见 [core-invariants](../architecture/core-invariants.md) I2/I3/I5；下面是 RVV 的具体落点。

**`tcrv.exec`** 只拥有 execution envelope：kernel、capability scope、selected variants、requires、dispatch/fallback、diagnostics、`mem_window`/`runtime_param` 的 ABI 角色声明。它**不**拥有 RVV compute 语义、dtype、schedule、intrinsic 拼写或 selected route authority。

**selected `tcrv_rvv` body** 拥有 typed vector-level 执行结构：typed vector 值与 element dtype、SEW/LMUL/vtype policy、VL/AVL/setvl、load/store 与 memory form、arithmetic/compare/select/FMA、reduction/accumulator、mask/tail、movement/layout/conversion、runtime ABI 值消费。dtype/config/operation 事实**只**经 typed 值、config、body 结构进入——不从 route id、C ABI 串、参数名、artifact 名、test 名、`__riscv_*_i32m1` 拼写、旧 `!tcrv_rvv.i32m1` helper 名推断（I5）。

期望的 typed 形态：

```mlir
!tcrv_rvv.vector<elem = i32, lmul = m1>
%vl = tcrv_rvv.setvl %remaining {sew = 32, lmul = m1, policy = ...}
%a  = tcrv_rvv.load %lhs[%i], %vl
%c  = tcrv_rvv.binary {kind = add} %a, %b, %vl
tcrv_rvv.store %out[%i], %c, %vl
```

**RVV 插件** 拥有：RVV body/config/control/dataflow 的 legality；hints/config/profile 影响生成代码时的 selected-body realization；route 支持与 route provider 输出；intrinsic 映射；C/RVV 向量类型映射；ABI 映射；fail-closed 诊断。Common/core 只调插件 interface + 校验通用结构，**不**按 RVV 语义分支、不选 intrinsic、不推 dtype、不建 schedule、不合成 body（I3）。

**公共 EmitC/export** 只做中性 materialization 与打包，materialize provider 建好的 `TCRVEmitCLowerableRoute`。它**不**发明 RVV compute、dtype、SEW/LMUL、schedule、intrinsic 选择或 ABI 角色语义。

## Parameter Flow

`mem_window` / `runtime_param` 声明 ABI/runtime 角色（lhs/rhs/out/n、buffer/scalar、runtime count、C ABI 拼写/provenance）。selected `tcrv_rvv` body 必须显式 bind/import 这些值，再经 typed control/dataflow 消费：

```mlir
%lhs = tcrv_rvv.runtime_abi_value @lhs
%n   = tcrv_rvv.runtime_abi_value @n
```

然后经 `setvl`、load/store、compute、mask、reduction、movement 使用。`tcrv.exec` 不从 ABI 角色名 / C 类型串 / artifact metadata 推断 add/mul/reduce/dtype。

## Route Provider —— 一个通用契约

RVV 路线历史上为每个算子族重复抄了 ~40 份同结构契约。它们其实是**同一条通用契约**反复套用。这里写一次：

RVV route provider 从 typed body 事实派生 route，事实包括：operation kind、element type、SEW、LMUL、tail/mask policy、operand form、memory form、runtime ABI binding、target capability。由这些派生 intrinsic/backend 拼写、C 类型、header、operand binding。

通用规则（对所有算子族一致，不需要每族再写一遍）：

- route 事实**结构性来自 typed body**或被 realize 进 body；metadata 镜像可有，但不是 authority（I4/I5）。
- 缺合法 typed body / 缺必要 capability / 只有 metadata → **fail closed** 给 bounded 诊断，绝不从 metadata 合成 C body（I7）。
- provider 输出先于公共 EmitC；公共 EmitC 不补 RVV 语义。
- runtime/correctness/performance 主张 → 真 `ssh rvv` 证据（I8）。

新增一个算子族 = 让它满足上面这条通用契约 + 进下面的覆盖表，**不是**再抄一份 7 段模板。

## Selected-Body Realization（Gearbox）

性能来自 RVV plugin-local 的 selected-body realization——即 Gearbox。它把 selected pre-realized body 变成 realized（调优过的）body，可 materialize：dynamic VL/setvl 放置、合法 SEW/LMUL/policy、memory form、mask/tail、register-pressure-safe unroll、prefetch/software-pipeline、accumulator/reduction layout。

它的 durable 契约（候选枚举/剪枝、resource model、autotuning 模式、跨 family 复用、"怎么判断是否真 resource-aware"）统一在 [variant-pipeline / generation-selection-tuning](../variant-pipeline/generation-selection-tuning.md)，RVV 不重述。RVV 这里只承诺：影响生成代码的选择在 route 构造前 realize 进 `tcrv_rvv` 结构，不外溢到公共 EmitC 或 artifact metadata。

## 计算类契约面（coverage surface，按结构化计算类）

RVV provider 按 route 契约要支持的算子族，按 [variant-pipeline](../variant-pipeline/index.md) 的结构化计算类组织。这是**广度清单**，不是契约阶梯——每族都走上面那条通用 provider 契约（当前覆盖到哪些族是状态，进 task/journal）：

| 计算类 | 算子族 |
|---|---|
| elementwise / arithmetic | binary 算术、compare、select、math |
| broadcast | scalar-broadcast 算术、splat-store |
| reduction / accumulation | standalone reduction、MAcc（plain + scalar-broadcast）|
| contraction-like | widening dot/product-reduce、widening MAcc |
| 低精度 contraction | i8/u8 widening product → widening reduction → dequant(i32→f32)[+clamp/select]、packed-i4 |
| movement / layout | unit-stride load/store、strided、indexed gather/scatter、segment2 deinterleave/interleave（plain + computed-mask）|
| dtype conversion | widening / narrowing conversion |
| mask / tail | computed-mask 各 memory/compute 形态、tail policy |
| runtime shape / control | `setvl` / `with_vl`、runtime AVL/VL、runtime-scalar 形态 |

低精度 contraction（i8/u8/packed-i4 widening-product-reduce-dequant）是 N3 的 RVV **性能轴**证据点。它必须按 typed body / provider 事实推进，不被 q8/q4/llama 之类工作负载名当 route authority（I9）。

## Legacy / fail-closed

历史上的 `i32m1`-as-route-authority、`RVVI32M1*` route spec、descriptor / microkernel / direct-C 导出、source-front-door 合成 route——全部是**删除目标或 fail-closed 债**，不是过渡架构（I7）。任何这些路径在当前树里只能 fail closed，不能作为 positive route 保留。新可执行工作一律走 typed body → provider → 公共 EmitC。

## 给 agent 的判断点（不是 gate）

- 在写"第 N 个 route 契约"之前：它真满足上面的通用契约吗？若是，进覆盖表即可，别抄模板。
- 在做"又一个 artifact/ABI evidence closeout"之前：读 [trunk-discipline](../guides/trunk-discipline.md)——大概率该转主干了。
- 性能主张落地前：Gearbox 真的枚举/剪枝候选了吗？真的对**框架出厂同-ISA kernel**胜出/打平了吗（**不是**赢 scalar）？没有就如实说，别用 metadata 包装成"成熟"。
