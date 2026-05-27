# 25 个 RVV Slice 任务

这个清单面向真实 PR 贡献，不是练习题合集。每个 slice 都应该补齐一块低层 typed `tcrv_rvv` surface、RVV plugin provider route、EmitC/RVV C++ 输出和测试证据。

本清单刻意避开当前主线 loop 正在推进的 `segment2` provider route-family / route-entry owner 工作。不要把学生任务写成 `segment2` 泛化、source-front-door、Toy/Template/TensorExtLite 或 common EmitC RVV 分支。

## 当前已有能力和避让区

当前仓库已经有一批 Stage2 RVV 面，学生不要重复实现：

- generic typed vector value/config：`!tcrv_rvv.vector<elem, lmul>`、`!tcrv_rvv.mask<elem, lmul>`、`!tcrv_rvv.vl`；
- 基础 `setvl`、`with_vl`、unit-stride load/store、generic binary add/sub/mul；
- compare/select、computed-mask select、masked binary、masked load/store；
- scalar broadcast / runtime scalar compare-select 的若干 bounded slice；
- reduction / standalone reduction / MAcc / widening MAcc / widening dot-reduction；
- indexed load/store、strided memory、segment2 interleave/deinterleave 和 computed-mask segment2 load/store/update；
- i64 add、widening conversion i16->i32 / i32->i64 的已有路线。

当前避让区：

- 不做 `segment2` route-family planning owner 或 route-entry registry；
- 不做 `segment3/segment4`，直到主线 loop 把 segment2 provider-family 边界稳定下来；
- 不碰 source-front-door / source-artifact positive path；
- 不改 common EmitC 去识别 RVV 语义；
- 不用 legacy `tcrv_rvv.i32_*`、`!tcrv_rvv.i32m*`、`RVVI32M1*`、`rvv-i32m1` route id。

## 每个 Slice 的统一交付形状

每个 PR 至少走下面这条链：

```text
typed tcrv_rvv body/config
  + explicit runtime ABI values
  + target capability facts
  -> RVV plugin-owned legality and selected-body realization when needed
  -> RVV provider route derivation
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> RVV intrinsic C/C++
  -> FileCheck and optional QEMU proof
```

常见改动路径：

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

模块化原则：

1. 先判断能否复用 generic op，例如 `binary`、`compare`、`select`、`load`、`store`、`masked_*`。能复用就不要新增大 op。
2. 只有新语义无法由现有 op 表达时，才新增 typed RVV op 或 compact pre-realized selected-body op。
3. provider 负责从 elem type、SEW、LMUL、policy、operand form、memory form 推导 RVV C type、mask type、header 和 intrinsic。
4. verifier negative case 必须覆盖至少一个“会误导 route authority”的错误，例如 dtype/LMUL 不匹配、ABI role 缺失、mask 来源不合法、immediate 超界、unsupported policy。
5. runtime proof 可选；如果 PR 声明运行正确性，必须附 QEMU 或等价 RVV 环境命令和输出。

## 难度与分配审查

这 25 个 slice 都是真实 compiler contribution，不是只补 FileCheck 的练习。为了让 25 个人工作量相对平衡，分配时按下面规则处理：

- A 类：适合作为大多数学生的两周任务。基础要求完成后，再选一个进阶或 QEMU proof 做区分。
- B 类：需要更多 provider / ABI / verifier 设计，适合能力较强或两人 review 更密的学生。
- C 类：真实可做，但工具链、语义或主线交叉风险更高。只建议分给能先写清边界的学生。

| # | Slice | 建议难度 | 平衡说明 |
|---|---|---|---|
| 1 | Compress Store | B | 语义清楚但需要 mask、dense write、可选 lane-count ABI。 |
| 2 | Slide Down | A | 典型 movement slice，offset verifier 和 QEMU 输入容易区分。 |
| 3 | Slide Up | A | 可复用 slide-down surface，但要处理 fill/overlap/policy。 |
| 4 | Register Gather | B | 和 memory indexed load 区分明确，index vector verifier 有工作量。 |
| 5 | Mask Logical Completion | B | 需要接入现有 mask composition，避免重复 mask_and。 |
| 6 | Mask Population And First-Set Queries | B | 有 scalar result ABI，适合训练 boundary 设计。 |
| 7 | Unsigned Compare And Select | A | signed/unsigned 差异容易构造测试，边界清楚。 |
| 8 | Signed Predicate Completion | A | 重点是 predicate canonicalization 和 operand-swap 测试。 |
| 9 | Integer Min/Max | A | generic binary 扩展，和 xor 示例同类但有 signedness。 |
| 10 | Bitwise Logical Arithmetic | A | xor 已完成作参考；and/or 足够小但需要完整链路。 |
| 11 | Shift Operations | B | VV/VX/VI amount form 需要建模，进阶空间足。 |
| 12 | Vector-Scalar VX Arithmetic | B | 触及 operand form 和 runtime scalar binding，不能退化成 broadcast helper。 |
| 13 | Immediate VI Arithmetic | B | immediate range/signedness verifier 是主要工作量。 |
| 14 | Reverse Subtract And Negation | A | 小而有效，测试能暴露 operand-order bug。 |
| 15 | Widening Add/Subtract | B | 要避开已有 widening conversion/MAcc/dot-reduce，边界需写清。 |
| 16 | Widening Multiply | B | 和 widening dot/reduction 区分明显，signedness relation 有工作量。 |
| 17 | Narrowing Conversion | C | rounding/saturation/policy 设计较难，适合强学生。 |
| 18 | F32 Elementwise Arithmetic | C | dtype/C type/floating intrinsic 链路较大，基础要求应限制在 add/mul。 |
| 19 | F32 FMA | C | ternary/FMA surface 和 intrinsic family 复杂，适合作为挑战题。 |
| 20 | Floating Compare And Select | C | NaN/ordered policy 容易出歧义，需严格限定。 |
| 21 | Saturating Add/Subtract | B | fixed-point 语义明确，QEMU 输入能很好地区分正确性。 |
| 22 | Multiply-High Family | B | 高位乘法测试有区分度，signedness relation 明确。 |
| 23 | Whole-Register Load/Store | C | toolchain intrinsic 支持和 group count 风险较高，先做 FileCheck。 |
| 24 | Fault-Only-First Load | C | runtime 行为可能受 QEMU/内存异常语义影响，QEMU proof 只作为进阶。 |
| 25 | Indexed EEW Variants | B | 扩展已有 indexed surface，重点是 EEW/LMUL legality。 |

整体上，A 类约 8 个、B 类约 11 个、C 类约 6 个。真实分配时可以让基础较弱的学生做 A 类并要求 QEMU proof，让强学生做 B/C 类但严格限制边界。不要把 C 类扩成 runtime 框架或前端工程。

## 测试用例收集格式

每个学生 PR 都应该能被教师按同一种方式收集测试证据。最低要求：

```text
1. MLIR compiler test
   - dialect positive/negative
   - EmitC materialization FileCheck
   - target RVV C++ FileCheck

2. generated kernel evidence
   - 生成命令
   - generated C++ 中的 exported kernel signature
   - generated C++ 中的关键 RVV intrinsic / vector type

3. C/C++ harness input-output evidence
   - harness.cpp
   - 输入数组/矩阵/标量/mask/index 的生成方式
   - scalar oracle 或 expected output
   - QEMU 命令和实际输出，如果声明 runtime correctness
```

推荐格式见 `docs/testcase-submission-format.md`。不要把 generated C++ 当作手写源码长期维护；默认应从 MLIR fixture 再生成。PR 中可以记录关键片段和运行输出。

## 任务清单

### 1. Compress Store

特性：支持 mask-controlled dense packing，即 `vcompress` 风格的压缩写出。

基础要求：

- 添加 `compress(mask, value, vl)` typed body form，或等价 pre-realized selected-body slice。
- 从 element type、LMUL、policy、mask type 推导 `vcompress` intrinsic。
- 输出语义必须是 true lane 紧密写出，false lane 不写。
- negative test 覆盖 mask 非 compare/mask op 产生、mask/vector 类型不匹配。

主要改动：

- `RVVOps.td`：新增 compress op 或 selected-body op。
- `RVVDialect.cpp`：验证 mask/value/vl 关系。
- `RVVEmitCRoutePlanning.cpp`：新增 compress operation kind 和 route payload。
- `test/Target/RVV/`：FileCheck `__riscv_vcompress...`。

进阶要求：暴露 compressed lane count runtime result，并用 sentinel output buffer 做 QEMU proof。

### 2. Slide Down

特性：支持 `vslidedown` lane movement。

基础要求：

- 添加 generic slide op，`direction = "down"`。
- offset 必须是 immediate 或 runtime scalar operand，不能藏在 route id 中。
- route 到 `vslidedown` intrinsic。
- negative test 覆盖 offset operand form 不合法和 unsupported dtype/LMUL。

主要改动：

- `RVVOps.td`：新增 `tcrv_rvv.slide` 或 selected-body slice。
- `RVVDialect.cpp`：验证 direction/offset/source/result type。
- `RVVEmitCRoutePlanning.cpp`：新增 movement route family。
- `test/Dialect/RVV/`、`test/Target/RVV/`：覆盖 immediate 和 scalar offset。

进阶要求：QEMU 覆盖 offset 为 0、1、超过当前 VL 的 case。

### 3. Slide Up

特性：支持 `vslideup` lane movement。

基础要求：

- 与 slide-down 共享同一个 generic slide op，`direction = "up"`。
- 明确 lane fill、destination overlap、tail/mask policy。
- route 到 `vslideup` intrinsic。

主要改动：

- 优先复用任务 2 的 `slide` op 和 provider helper。
- 新增 `direction = "up"` 的 verifier 和 route case。
- `test/Target/RVV/` 检查 `__riscv_vslideup...`。

进阶要求：添加 `slide1up` 标量插入特例。

### 4. Register Gather

特性：支持 register-indexed gather，即 `vrgather`，区别于 memory indexed load。

基础要求：

- 添加 register gather op，显式接受 data vector、index vector、VL。
- index vector 类型和 data vector 类型都必须是 typed body 的一部分。
- route 到 `vrgather` intrinsic。
- negative test 覆盖 data/index LMUL 或 index element width 不匹配。

主要改动：

- `RVVOps.td`：新增 `tcrv_rvv.register_gather`。
- `RVVDialect.cpp`：验证 index vector relation。
- `RVVEmitCRoutePlanning.cpp`：新增 register movement/gather route。
- `test/Dialect/RVV/`：区别 `index_load/indexed_load` 与 `register_gather`。

进阶要求：支持 immediate gather 或 scalar-index gather。

### 5. Mask Logical Completion

特性：补齐 mask `or/xor/not`。当前已有 bounded `mask_and` 路线，不能重做。

基础要求：

- 扩展现有 mask-composition surface，支持 `or`、`xor`、`not`。
- 结果 mask 必须能驱动 select、masked store、masked reduction 或 compress。
- route 到 RVV mask logical intrinsic。
- negative test 覆盖不同 element/LMUL mask 混用。

主要改动：

- `RVVOps.td`：扩展现有 mask op 的 kind，而不是新增 `mask_or_i32` 之类 helper。
- `RVVDialect.cpp`：验证 mask kind 和 mask type 一致性。
- `RVVSelectedBodyRealization.cpp`：如果 pre-realized body 需要组合 mask，在 realization 中落到显式 mask op。
- `RVVEmitCRoutePlanning.cpp`：添加 mask logical provider route。

进阶要求：添加两个 compare 结果组合后驱动 select 的 target FileCheck。

### 6. Mask Population And First-Set Queries

特性：支持 mask scalar query，例如 population count 和 first-set。

基础要求：

- 添加消费 typed mask、产生 scalar result 的 selected-body operation。
- route 到 `vcpop` 或 `vfirst` intrinsic family。
- 定义 scalar result 的 runtime ABI boundary。
- negative test 覆盖结果 ABI role 缺失和 mask 类型不匹配。

主要改动：

- `RVVOps.td`：新增 mask query op 或 compact selected body。
- `RVVDialect.cpp`：验证 result boundary。
- `RVVEmitCRoutePlanning.cpp`：新增 mask-query route family。
- `test/Target/RVV/`：FileCheck `__riscv_vcpop...` / `__riscv_vfirst...`。

进阶要求：QEMU 覆盖 empty mask、首 lane set、末 lane set。

### 7. Unsigned Compare And Select

特性：补齐 unsigned integer predicate。

基础要求：

- 支持 `ltu`、`leu`、`gtu`、`geu` predicate kind。
- 从 typed element kind 推导 unsigned compare intrinsic。
- 复用现有 compare/select/mask body 结构，不新建 unsigned wrapper dialect。
- 添加 signed 和 unsigned 结果故意不同的 FileCheck/QEMU case。

主要改动：

- `RVVDialect.cpp`：扩展 predicate verifier。
- `RVVEmitCRoutePlanning.cpp`：predicate canonicalization 映射到 unsigned intrinsic。
- `test/Dialect/RVV/`：unknown predicate 和 signed/unsigned 混用 negative。

进阶要求：让 unsigned predicate 同时覆盖 computed-mask store 或 reduction 的一个 selected-body fixture。

### 8. Signed Predicate Completion

特性：补齐 signed predicate 缺口，例如 `ne`、`sgt`、`sge`。

基础要求：

- route planning 可以 canonicalize predicate，例如把 `sgt` 改写为等价 operand order，但不能改变 typed body contract。
- unknown predicate 必须 fail closed。
- 添加能暴露 operand swap 错误的测试。

主要改动：

- `RVVEmitCRoutePlanning.cpp`：集中维护 predicate canonicalization helper。
- `RVVDialect.cpp`：验证 predicate kind。
- `test/Target/RVV/`：compare-select 和 computed-mask select 各至少一个 case。

进阶要求：和任务 7 共享 predicate helper，避免重复 if-chain。

### 9. Integer Min/Max

特性：支持 vector min/max。

基础要求：

- 扩展 `binary {kind = min/max/minu/maxu}` 或等价 selected body。
- signed/unsigned min/max 必须从 typed element 和 operation kind 推导。
- 至少覆盖 i32 和一个更小整数宽度。
- negative test 覆盖 float dtype 使用 integer min/max。

主要改动：

- `RVVOps.td` / `RVVDialect.cpp`：扩展 binary kind。
- `RVVEmitCRoutePlanning.cpp`：新增 min/max intrinsic mapping。
- `test/Conversion/EmitC/`：materialization positive/negative。

进阶要求：添加 masked min/max 变体和 inactive lane 检查。

### 10. Bitwise Logical Arithmetic

特性：补齐整数 bitwise `and/or`。本分支已经把普通 vector-vector `xor` 作为参考 slice 完成，学生不要重复实现 `xor`，应阅读 `docs/add-rvv-xor-slice-workflow.md` 后按同样路径补齐剩余 bitwise slice。

基础要求：

- 扩展 generic `binary {kind}`。
- 对整数 dtype 生成 RVV bitwise intrinsic。
- 拒绝 floating element type。
- `xor` 已经是可参考的正向例子；新 PR 必须新增自己的正向 typed body、provider mapping、target FileCheck，并保留 unsupported dtype negative。

主要改动：

- `RVVDialect.cpp`：binary kind legality。
- `RVVEmitCRoutePlanning.cpp`：bitwise intrinsic mapping。
- `test/Dialect/RVV/`：补充正向 typed body case，并保留 unsupported dtype negative。

进阶要求：添加 VX 或 VI 形式中的一个。

### 11. Shift Operations

特性：支持 `sll`、`srl`、`sra`。

基础要求：

- 添加 shift operation kind。
- shift amount 是 vector、scalar 还是 immediate 必须由 typed body 显式表达。
- 验证 shift-width 和 dtype 组合。
- route 到 RVV shift intrinsic。

主要改动：

- `RVVOps.td`：可复用 binary，也可新增 shift op，如果 operand form 需要更清楚。
- `RVVDialect.cpp`：验证 amount type/range。
- `RVVEmitCRoutePlanning.cpp`：区分 VV/VX/VI shift route。

进阶要求：masked shift 和边界 shift amount QEMU proof。

### 12. Vector-Scalar VX Arithmetic

特性：添加通用 VX operand form，不是 scalar broadcast helper。

基础要求：

- scalar 必须作为 runtime ABI value 或 MLIR scalar operand 显式进入 body。
- provider 从 VX operand form 推导 intrinsic，例如 `vadd_vx`、`vmin_vx`。
- 至少覆盖一个 arithmetic family。
- negative test 覆盖 scalar role 缺失和错误 scalar dtype。

主要改动：

- `RVVOps.td`：为 binary/compare/shift 引入 operand form attribute 或专用 scalar operand form。
- `RVVSelectedBodyRealization.cpp`：pre-realized body 需要把 scalar 显式导入 selected body。
- `RVVEmitCRoutePlanning.cpp`：添加 VX route mapping。

进阶要求：复用到 compare 或 shift scalar operand。

### 13. Immediate VI Arithmetic

特性：添加 RVV 支持的 immediate operand form。

基础要求：

- 只对 RVV 有合法 VI form 的操作添加 immediate attribute。
- 验证 immediate range 和 signedness。
- route 到 VI intrinsic，不通过 route id 表达。
- 超范围时诊断应提示使用 VX scalar form。

主要改动：

- `RVVOps.td`：添加 immediate attr。
- `RVVDialect.cpp`：range verifier。
- `RVVEmitCRoutePlanning.cpp`：VI intrinsic mapping。
- `test/Target/RVV/`：FileCheck `..._vi_...` intrinsic。

进阶要求：和任务 11 的 immediate shift 共享 range helper。

### 14. Reverse Subtract And Negation

特性：支持 operand-order-sensitive arithmetic。

基础要求：

- 添加 explicit operation kind 或 operand-order attribute。
- route planning 必须保留 lhs/rhs 意义。
- 添加会暴露 operand swap 错误的测试。

主要改动：

- `RVVDialect.cpp`：验证 operand-order attr。
- `RVVEmitCRoutePlanning.cpp`：route 到 `vrsub` 或 negation intrinsic/sequence。
- `test/Target/RVV/`：用非对称输入检查 generated intrinsic。

进阶要求：覆盖 vector-scalar reverse-subtract immediate form。

### 15. Widening Add/Subtract

特性：添加 widening add/sub。不要和已有 widening conversion、widening MAcc、widening dot-reduce 混在一起。

基础要求：

- 至少支持一种关系，例如 i16 -> i32 或 i32 -> i64。
- 验证 source/destination element width、SEW relation、LMUL relation。
- route 到 RVV widening add/sub intrinsic。

主要改动：

- `RVVOps.td`：新增 widening binary op 或扩展现有 widening relation。
- `RVVDialect.cpp`：验证 source/result relation。
- `RVVEmitCRoutePlanning.cpp`：新增 widening add/sub route，与 widening conversion/MAcc route 分开。

进阶要求：添加 mixed signed/unsigned widening form。

### 16. Widening Multiply

特性：添加不归约到 accumulator 的 widening multiply。不要实现 dot、reduce 或 MAcc。

基础要求：

- 验证 destination width 是 source width 的两倍。
- 显式表达 signed、unsigned 或 mixed signedness。
- route 到 widening multiply intrinsic。

主要改动：

- `RVVOps.td`：widening multiply op/relation。
- `RVVDialect.cpp`：signedness 和 result type verifier。
- `RVVEmitCRoutePlanning.cpp`：与 widening dot-product route 分开。

进阶要求：添加 masked widening multiply。

### 17. Narrowing Conversion

特性：支持 narrowing conversion，例如 truncate 或 narrowing shift。

基础要求：

- 添加 source/destination typed conversion relation。
- route 到 narrowing conversion / narrowing shift intrinsic。
- rounding/saturation/policy 必须显式建模并验证。

主要改动：

- `RVVOps.td`：新增 `narrowing_convert` 或扩展 conversion op。
- `RVVDialect.cpp`：验证 dest width 是 source width 的一半或指定 relation。
- `RVVEmitCRoutePlanning.cpp`：narrowing intrinsic mapping。

进阶要求：添加 saturating/rounding 变体。

### 18. F32 Elementwise Arithmetic

特性：支持 f32 add/sub/mul/div。

基础要求：

- 扩展 f32 vector type legality 和 C type mapping。
- 从 typed body/config 推导 floating elementwise intrinsic。
- 拒绝 integer-only operation kind。
- 至少覆盖 add 和 mul，div 可作为进阶。

主要改动：

- `RVVConfigContract.cpp`：f32 SEW/LMUL legality。
- `RVVEmitCRoutePlanning.cpp`：floating C type 和 intrinsic mapping。
- `test/Target/RVV/`：FileCheck `vfloat32...` type 和 `vfadd/vfmul`。

进阶要求：QEMU 覆盖 NaN、Inf、signed zero。

### 19. F32 FMA

特性：支持 floating fused multiply-add。

基础要求：

- 添加 typed ternary arithmetic body 或 pre-realized FMA slice。
- 推导 accumulator/result C vector type 和 FMA intrinsic。
- 验证 operand/result element type 一致。

主要改动：

- `RVVOps.td`：新增 ternary/FMA op。
- `RVVDialect.cpp`：验证三输入和 result type。
- `RVVEmitCRoutePlanning.cpp`：route 到 `vfmacc/vfmadd` 等明确 family。

进阶要求：添加 fused negative multiply-add 或 multiply-subtract。

### 20. Floating Compare And Select

特性：支持 f32 compare mask 和 select。

基础要求：

- 添加 floating predicate kind，必要时区分 ordered/unordered。
- 从 f32 typed body 推导 mask type、compare intrinsic 和 select。
- 拒绝 integer-only predicate。

主要改动：

- `RVVDialect.cpp`：predicate kind 按 integer/floating 分类。
- `RVVEmitCRoutePlanning.cpp`：floating compare intrinsic mapping。
- `test/Dialect/RVV/`：NaN predicate policy 的 verifier 或 target case。

进阶要求：QEMU 覆盖 NaN predicate 行为。

### 21. Saturating Add/Subtract

特性：支持 fixed-point saturating add/sub，例如 `vsadd`、`vsaddu`、`vssub`、`vssubu`。

基础要求：

- 添加 `saturating_add/sub` operation kind，区分 signed/unsigned。
- typed body 必须表达 source/result element type 和 saturation semantics。
- route 到 saturating intrinsic。
- negative test 覆盖 floating dtype 和 unsupported width。

主要改动：

- `RVVOps.td`：扩展 binary kind 或新增 saturating op。
- `RVVDialect.cpp`：验证 signedness 和 integer-only legality。
- `RVVEmitCRoutePlanning.cpp`：saturating intrinsic mapping。

进阶要求：QEMU 覆盖溢出、下溢、不饱和三类输入。

### 22. Multiply-High Family

特性：支持 `vmulh`、`vmulhu`、`vmulhsu`，即乘法高位结果。

基础要求：

- 添加 multiply-high operation kind。
- 明确 signed/signed、unsigned/unsigned、signed/unsigned 三种关系。
- result type 与 source type 同宽，不要误写成 widening multiply。
- route 到 multiply-high intrinsic。

主要改动：

- `RVVOps.td`：扩展 binary kind。
- `RVVDialect.cpp`：验证 signedness relation。
- `RVVEmitCRoutePlanning.cpp`：mul-high intrinsic mapping。

进阶要求：QEMU 用大数输入证明取高位而不是低位乘法。

### 23. Whole-Register Load/Store

特性：支持 whole-register memory operation。

基础要求：

- 区分 whole-register load/store 与普通 unit-stride load/store。
- route 到 toolchain 支持的 whole-register intrinsic。
- unsupported dtype/LMUL/group count 必须 fail closed。

主要改动：

- `RVVOps.td`：新增 whole-register memory op 或 memory form。
- `RVVDialect.cpp`：验证 group count 和 type。
- `RVVEmitCRoutePlanning.cpp`：whole-register memory route。

进阶要求：添加 multi-register group count。

### 24. Fault-Only-First Load

特性：支持 fault-only-first load，用于 scan-like memory case。

基础要求：

- typed memory op 或 selected-body slice 需要表达 data vector 和 effective VL/update semantics。
- 定义 updated VL 或 loaded count 如何进入 selected body/runtime boundary。
- route 到 fault-only-first intrinsic，拒绝 unsupported form。

主要改动：

- `RVVOps.td`：新增 fault-only-first load op。
- `RVVDialect.cpp`：验证 updated VL/result boundary。
- `RVVEmitCRoutePlanning.cpp`：fault-only-first route payload。

进阶要求：在本地 QEMU 可稳定表达时添加 early-stop proof。

### 25. Indexed EEW Variants

特性：把 indexed gather/scatter 扩展到更多 index EEW。

基础要求：

- index EEW 必须在 typed index vector/config 中显式表达。
- 从 data type、data LMUL、index EEW、memory form 推导 indexed intrinsic。
- 添加 unsupported index/data LMUL 组合的 negative test。
- 不要重做已有 indexed load/store；只扩展 index EEW coverage。

主要改动：

- `RVVOps.td`：扩展 index vector/config 表达。
- `RVVDialect.cpp`：验证 index EEW/data LMUL relation。
- `RVVEmitCRoutePlanning.cpp`：indexed intrinsic suffix/type mapping。

进阶要求：添加 ordered / unordered indexed memory policy。

## 优先级建议

第一批更适合学生并且不撞主线 loop：

```text
1 compress
2 slide-down
3 slide-up
4 register-gather
6 mask-query
10 bitwise
11 shift
13 immediate
14 reverse-subtract
17 narrowing
21 saturating
22 multiply-high
23 whole-register
24 fault-only-first
25 indexed-EEW
```

需要和教师确认边界后再分配：

```text
5 mask logical completion
7/8 predicate completion
12 VX arithmetic
15/16 widening add/sub/mul
18/19/20 floating family
```

暂不分配：

```text
segment3 / segment4 / segmentN generic route-family
```
