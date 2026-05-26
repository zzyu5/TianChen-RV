# 25 个 RVV Slice 任务

这个清单给出 25 个边界清晰的 RVV slice。每个 slice 的目标是形成一个可以 review 的 PR：不是只改测试，也不是直接堆 intrinsic wrapper，而是补齐一块 typed `tcrv_rvv` surface / RVV provider route / EmitC output / 测试证据。

每个 slice 都必须保持下面的结构：

```text
typed tcrv_rvv body/config
  + explicit runtime ABI values
  + target capability facts
  -> RVV plugin-owned legality and route derivation
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> RVV intrinsic C/C++
```

不要通过 legacy `tcrv_rvv.i32_*` helper、`RVVI32M1*` table、route-id 语义、source-front-door marker 或 common EmitC RVV 分支来实现。

## 通用开展方式

每个任务先找最接近的现有模式：

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
test/Dialect/RVV/
test/Conversion/EmitC/
test/Target/RVV/
```

通用要求：

1. 只在必要时扩展 generic typed RVV body surface。
2. 只有 provider 确实需要新 operation/memory class 时，才新增或扩展 `RVVSelectedBodyOperationKind` / `RVVSelectedBodyMemoryForm`。
3. 添加 verifier negative case，覆盖 unsupported dtype、LMUL、policy、operand shape、mask source、ABI role 或 memory form。
4. 添加 FileCheck，证明生成的 C/C++ 使用了预期 RVV type、header、setvl、load/store 和 intrinsic family。
5. 如果声明 runtime correctness，附本地 QEMU 命令和输出。

## 1. Compress Store

特性：支持 mask-controlled dense packing，即 `vcompress` 风格的压缩写出。

基础要求：

- 添加 typed body form：`compress(mask, value, vl)` 或等价 selected-body slice。
- 从 element type、LMUL、policy、mask type 推导 compress intrinsic。
- 定义输出内存规则：true lane 紧密写出，false lane 不写。
- 添加 mask 来源非法的 negative test。

进阶要求：

- 暴露 compressed lane count 的 runtime result boundary。
- 用稀疏 mask 和 sentinel output buffer 做 QEMU proof。

## 2. Slide Down

特性：支持 `vslidedown` 类 lane movement。

基础要求：

- 添加 direction 为 `down` 的 generic slide op 或 pre-realized selected-body slice。
- 支持 runtime 或 immediate offset，但必须在 typed body 中显式表达。
- 推导正确的 slide-down intrinsic。

进阶要求：

- 同时支持 vector-immediate 和 vector-scalar offset。
- QEMU 覆盖 offset 为 0、1、超过单个 VL chunk 的情况。

## 3. Slide Up

特性：支持 `vslideup` 类 lane movement。

基础要求：

- 与 slide-down 对称，添加 direction 为 `up` 的 body/config。
- 验证 lane fill、destination overlap、policy 行为。
- 生成正确的 `vslideup` intrinsic。

进阶要求：

- 添加 `slide1up` 标量插入特例。
- 用 sentinel 检查 inactive/tail lane 行为。

## 4. Register Gather

特性：支持 register-indexed gather，即 `vrgather`。

基础要求：

- 添加 typed index-vector operand form，区别于 memory indexed load。
- 从 index element type 和 data element type 推导 C type 与 gather intrinsic。
- 验证 data/index LMUL 或 index element type 不匹配的错误。

进阶要求：

- 支持 immediate gather 和 scalar-index gather。
- 添加 duplicate index、out-of-range index 测试。

## 5. Mask Logical Operations

特性：支持 mask `and/or/xor/not` 等逻辑操作。

基础要求：

- 添加 generic mask op 或扩展现有 mask-composition。
- route 到 RVV mask logical intrinsics。
- 结果 mask 可以驱动 select、masked store、masked reduction 或 compress。

进阶要求：

- 添加两个 compare 结果组合的 selected-body 示例。
- 添加不同 element/LMUL mask 混用的 negative test。

## 6. Mask Population And First-Set Queries

特性：支持 mask scalar query，例如 population count 和 first-set。

基础要求：

- 添加消费 typed mask、产生 scalar result 的 selected-body operation。
- route 到 `vcpop` 或 `vfirst` intrinsic family。
- 定义 scalar result 的 runtime ABI boundary。

进阶要求：

- 添加 before-first / including-first mask 生成。
- QEMU 覆盖 empty mask、首 lane set、末 lane set。

## 7. Unsigned Compare And Select

特性：补齐 unsigned integer predicate。

基础要求：

- 添加 `ltu`、`leu`、`gtu`、`geu` 等 unsigned predicate kind。
- 从 typed element kind 推导 unsigned compare intrinsic。
- 复用现有 compare/select/mask body 结构。

进阶要求：

- 添加 signed 和 unsigned 结果故意不同的 case。

## 8. Signed Predicate Completion

特性：补齐 signed predicate 的缺口。

基础要求：

- 支持 `ne`、`sgt`、`sge` 以及等价 canonical form。
- route planning 可以 canonicalize predicate，但不能改变 typed body contract。
- 添加 unknown predicate string 的 negative test。

进阶要求：

- 在 compare-select 和 computed-mask store/reduction family 之间共享 predicate canonicalization。

## 9. Integer Min/Max

特性：支持 vector min/max。

基础要求：

- 添加 `binary {kind = min/max/minu/maxu}` 或等价 selected body。
- route 到 signed/unsigned min/max intrinsic。
- 至少覆盖 i32 和一个更小整数宽度。

进阶要求：

- 添加 masked min/max 变体和 inactive lane 检查。

## 10. Bitwise Logical Arithmetic

特性：支持整数 bitwise `and/or/xor`。

基础要求：

- 扩展 generic `binary {kind}`。
- 对支持的整数 dtype 生成对应 RVV bitwise intrinsic。
- 拒绝 floating element type。

进阶要求：

- 添加 vector-scalar 和 immediate 形式。

## 11. Shift Operations

特性：支持 `sll`、`srl`、`sra`。

基础要求：

- 添加 shift operation kind。
- shift amount 是 vector、scalar 还是 immediate 必须由 typed body 显式表达。
- 验证 shift-width 和 dtype 组合。

进阶要求：

- 添加 masked shift 和边界 shift amount QEMU proof。

## 12. Vector-Scalar VX Arithmetic

特性：添加通用 vector-scalar operand form。

基础要求：

- scalar 必须作为 runtime ABI value 或 MLIR scalar operand 显式进入 body。
- provider 从 VX operand form 推导 intrinsic。
- 覆盖 add/sub/mul/min/max/bitwise 中至少一个 family。

进阶要求：

- 复用到 compare 和 shift scalar operand。

## 13. Immediate VI Arithmetic

特性：添加 RVV 支持的 immediate operand form。

基础要求：

- 只对 RVV 有合法 VI form 的操作添加 immediate attribute。
- 验证 immediate range 和 signedness。
- route 到 VI intrinsic，不通过 route id 表达。

进阶要求：

- immediate 超范围时给出明确诊断，提示应使用 scalar form。

## 14. Reverse Subtract And Negation

特性：支持 operand-order-sensitive arithmetic。

基础要求：

- 添加 explicit operation kind 或 operand-order attribute。
- route planning 必须保留 lhs/rhs 意义。
- 添加会暴露 operand swap 错误的测试。

进阶要求：

- 覆盖 vector-scalar reverse-subtract immediate form。

## 15. Widening Add/Subtract

特性：添加 widening add/sub。

基础要求：

- 至少支持一种关系，例如 i16 -> i32 或 i32 -> i64。
- 验证 source/destination element width、SEW relation、LMUL relation。
- route 到 RVV widening add/sub intrinsic。

进阶要求：

- 添加 mixed signed/unsigned widening form。

## 16. Widening Multiply

特性：添加不归约到 accumulator 的 widening multiply。

基础要求：

- 与现有 widening dot/reduce、widening MAcc 区分开。
- 验证 destination width 是 source width 的两倍。
- 显式表达 signed、unsigned 或 mixed signedness。

进阶要求：

- 添加 masked widening multiply。

## 17. Narrowing Conversion

特性：支持 narrowing conversion，例如 truncate 或 narrowing shift。

基础要求：

- 添加 source/destination typed conversion relation。
- route 到 narrowing conversion / narrowing shift intrinsic。
- 如有 rounding/saturation/policy，必须显式建模并验证。

进阶要求：

- 添加 saturating/rounding 变体。

## 18. F32 Elementwise Arithmetic

特性：支持 f32 add/sub/mul/div。

基础要求：

- 扩展 f32 vector type legality 和 C type mapping。
- 从 typed body/config 推导 floating elementwise intrinsic。
- 拒绝 integer-only operation kind。

进阶要求：

- QEMU 覆盖 NaN、Inf、signed zero。

## 19. F32 FMA

特性：支持 floating fused multiply-add。

基础要求：

- 添加 typed ternary arithmetic body 或 pre-realized FMA slice。
- 推导 accumulator/result C vector type 和 FMA intrinsic。
- 验证 operand/result element type 一致。

进阶要求：

- 添加 fused negative multiply-add 或 multiply-subtract。

## 20. Floating Compare And Select

特性：支持 f32 compare mask 和 select。

基础要求：

- 添加 floating predicate kind，必要时区分 ordered/unordered。
- 从 f32 typed body 推导 mask type、compare intrinsic 和 select。
- 拒绝 integer-only predicate。

进阶要求：

- QEMU 覆盖 NaN predicate 行为。

## 21. Segment3 Load/Store

特性：把 segment memory movement 从 segment2 扩到 segment3。

基础要求：

- 添加 segment count 3 route metadata 和 tuple C type mapping。
- 三个 field 的 runtime ABI role 必须显式表达。
- route 到 segment3 load/store/extract/insert intrinsic。

进阶要求：

- 在 unmasked segment3 稳定后添加 computed-mask segment3 update。

## 22. Segment4 Load/Store

特性：把 segment memory movement 扩到 segment4。

基础要求：

- 类似 segment3，但 segment count 为 4。
- field role 不能藏在 artifact name 里。
- 验证缺失 field role 和重复 field role。

进阶要求：

- QEMU 做四字段 interleave/deinterleave case。

## 23. Whole-Register Load/Store

特性：支持 whole-register memory operation。

基础要求：

- 把 whole-register load/store 与普通 unit-stride vector load/store 区分。
- route 到 toolchain 支持的 whole-register intrinsic。
- 不支持的 dtype/LMUL 给出 fail-closed diagnostic。

进阶要求：

- 添加 multi-register group count。

## 24. Fault-Only-First Load

特性：支持 fault-only-first load，用于 scan-like memory case。

基础要求：

- typed memory op 或 selected-body slice 需要表达 data vector 和 effective VL/update semantics。
- 定义 updated VL 或 loaded count 如何进入 selected body/runtime boundary。
- route 到 fault-only-first intrinsic，拒绝 unsupported form。

进阶要求：

- 在本地 QEMU 可稳定表达时，添加 early stop proof。

## 25. Indexed EEW Variants

特性：把 indexed gather/scatter 扩展到更多 index EEW。

基础要求：

- index EEW 必须在 typed index vector/config 中显式表达。
- 从 data type、data LMUL、index EEW、memory form 推导 indexed intrinsic。
- 添加 unsupported index/data LMUL 组合的 negative test。

进阶要求：

- 添加 ordered / unordered indexed memory policy。

## 优先级建议

优先做能补齐一个 operation class 或 memory class 的 slice。单纯把已有 i32/m1 fixture 换个名字不算高价值贡献。保留 i32 add/sub/mul 例子只有一种合理方式：它必须只是 generic typed RVV surface 的普通实例，route 由 typed facts 推导。
