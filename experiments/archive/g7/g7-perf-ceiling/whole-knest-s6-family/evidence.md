# G7 whole-K-nest S6 roll — K-quant GEVM **FAMILY** extension (q2/q3/q4/q6_K · emitter-implanted · kernel-axis C1)

> **线定位（诚实前提）**：把 q5_K 已部署（commit `34fded0d`·casefile `../whole-knest-s6-deploy/`）的 whole-K-nest S6 rolled envelope **从单格泛化到 K-quant GEVM 家族其余 4 格**（`emitRepackKQuantGemvBodyQ2K/Q3K/Q4K/Q6K`）·capability-keyed（复用 q5_K 的 `emit_loop_schedule` OptionalAttr + `resolveRepackMainTermRolled` resolver）·byte-exact 逐格硬门。**= C1 extensibility 家族证 + C2 marginal-cost**（同一 rolled capability 低成本复用铺开·非单格）。**零 perf green**（roll = icache/code-volume/vsetvli 杠杆·非指令密度杠杆·M=1 broadcast 密度限·[CASE-MICRO-E2E] decode memory-wall；commit/ledger 明记零 perf）。
> **纯本地 forced build·无需上板**（byte-exact + objdump 即足·kernel-axis C1 artifact）。禁 git·禁触碰 sealed GEMM leaf / q5_0-q5_1 REDESIGN-B / q5_K rolled(刚部署) / GEMM 兄弟 / q4_K colgroup-tiled / 各格互扰（capability-guard 逐格隔离）。

---

## ★裁决 TL;DR

1. **whole-K-nest rolled envelope 泛化到 q2_K·q3_K·q4_K·q6_K GEVM（全 4 格完成·capability-keyed·byte-exact）**。加上已部署 q5_K = **K-quant GEVM 全 5 格家族**（q2/q3/q4/q5/q6_K）均带 rolled 变体。每格把 dominant per-element inner 循环（q4:ii=16 · q2:mm=16 · q3:l=16 · q6:p=8）roll 成 runtime `emitc.for`·per-strip(×lane/quadrant) i16 partials 作 **RESIDENT SSA-register VariableOps**（seed 于环上·环内 load-accumulate-store·与 sumf/sumi/bsums 同机制）。per-element 解码（各格 decode：q4 nibble · q2 2-bit peel · q3 qs|hmask 3-bit subtractive native-mask · q6 ql|qh 6-bit two-plane -32 bias）环内即消即用·从不 materialize 过栈。
2. **hot-path tile-roundtrip CLOSED 全 4 格**：每格 hot element-loop 副本（q4/q6 各 8 · q2/q3 各 4·= j×pair×k / sh×grp 展开）**全部 0 `vs*r.v` ∧ 0 `vsetvli`**（objdump 逐环证·`raw/metrics_summary.txt §B`）。[GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] hot 路 + [GAP-EMIT-VSETVL-TAX] hot 路 **全 4 格塌缩**。
3. **整体账**（gcc-15.2.0 -O3·VLEN128 numHalves=2·default→rolled）：v-insn q2 −86%/q3 −95%/q4 −89%/q6 −90%·vsetvli q2 −87%/q3 −99%/q4 −90%/q6 −98%·vs\*r.v q2 −90%/q3 −97%/q4 −96%/q6 −98%（详表 §A）。
4. **residual vs\*r.v（如实标·非夸大）**：**全在 COLD block-loop-top**（accumulator seed/flush + scale-unpack）·**非 hot 解码 nest**：q2=12 · q3=2 · q4=6 · q6=0（q6 全 0！）vs\*r.v。= numHalves=2（VLEN128 dual-strip）resident-bank setup 税·**与 q5_K 的 9 residual 同类·预期**·核心目标（hot-path tile-RT 消）**全 4 格达成**。
5. **byte-exact 硬门全过（逐格）**：① Conversion/RVV lit **247/247 PASS**（was 243·+4 新 rolled fixtures·零回归）② 默认（unrolled）路 emitc **byte-IDENTICAL** before/after（全 4 格·shipped 默认零漂移）③ DYNAMIC op-stream 恒等（rolled runtime 环执行的 vwmacc16 总数 = default full-unroll 静态 1024·全格·§C3）④ 板 numeric A/B = DEFERRED（kernel-axis 无板·construction-identity + q5_K deploy 先例 host oracle 0/32）。
6. **零回归 + 隔离实证**：git diff 仅 `.cpp`+`.h`（+4 新 fixtures）·diff hunk context **仅** 4 目标 GEVM leaf（Q2K/Q3K/Q4K/Q6K）+ 共享 dispatch `emitTypedRepackGemvLoopBody`·**q5_K rolled+default emitc byte-IDENTICAL 到 committed deploy 输出（34fded0d）**（dispatch refactor 零漂移）·GEMM 兄弟 / sealed GEMM leaf / q5_0-q5_1 REDESIGN-B / q4_K colgroup-tiled **零 diff**·`RVVOps.td` 无改（`emit_loop_schedule` attr 已由 q5_K deploy 建·无需 ODS/verifier 改动）。

---

## 0. 实装（改了什么·capability-keyed·增量·强隔离）

**2 文件·additive/byte-exact-neutral**（+4 新 test fixtures）：

| 文件 | 改动 | 性质 |
|---|---|---|
| `lib/Conversion/RVV/RVVToEmitCInternal.h` | 4 GEVM leaf 签名（Q2K/Q3K/Q4K/Q6K）各 +`bool rolledMainTerm` | 单参 |
| `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | (a) MIN-fold dispatch：`rolledMainTerm` resolve **上提为家族共享**（q5_K/q4_K/q2_K 共用一个 resolve）·传 q4_K/q2_K；(b) no-min dispatch：新增家族共享 resolve·传 q6_K/q3_K；(c) 4 leaf 各内 `if(rolledMainTerm){rolled element-loop}else{原 unrolled 逐字}` | leaf-local + dispatch |

**capability-key 机制**（复用 q5_K deploy 的既有件）：`resolveRepackMainTermRolled(getEmitLoopSchedule(), ...)` —— 显式 stamp `"rolled"`/`"unrolled"` 优先·ABSENT ⇒ frozen sentinel = unrolled（shipped 零改）。**仅带显式 `emit_loop_schedule = "rolled"` stamp 才走 rolled**。

**C2 marginal-cost（低）**：copy-then-adapt q5_K 模式·复用①`emit_loop_schedule` OptionalAttr（q5_K deploy 已建·本线零 ODS 改）②`resolveRepackMainTermRolled` resolver（dispatch 每 fold-family 一次·共享）③resident-bank VariableOp load-acc-store 模式。每格增量 = 一个局部 rolled/unrolled 分支（含保留的 unrolled else 逐字）。总 +406/−115 行覆盖 4 leaf + dispatch。**无新机制·无新 op·无新 verifier**——纯 capability-keyed 复用。

---

## 1. objdump 部署验证（hot 路 tile-RT 全消·residual 如实标）

见 `raw/metrics_summary.txt §A/§B`。核心：**每格 hot element-loop 副本全 0 vs\*r.v ∧ 0 vsetvli**（Python 逐环扫描·`raw/{q2,q3,q4,q6}k_gevm_rolled.s`）。residual = cold block-loop-top bank seeds/flush（numHalves=2 dual-strip·非 hot tile）。

## 2. byte-exact 硬门（§5 / metrics §C·全 4 格全过）

## 3. scope

- **q2_K·q3_K·q4_K·q6_K GEVM rolled = 全 4 格完整**（element-loop 全 roll·capability-keyed·byte-exact·hot-path tile-RT 消）。**零 deferred**（无格因结构过复杂放弃）。
- 加已部署 q5_K = **K-quant GEVM 全 5 格家族 rolled**。
- **residual 0 化 = DEFERRED / 非本线**（numHalves=2 达字面 0 需 roll h-strip 或 whole j/pair/k nest·风险高·收益递减·hot 路已净·同 q5_K 判定）。

## 4. 诚实定性（明记零 perf）

**kernel-axis C1 emitter-maturity 家族泛化**·closes hot-path [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] + [GAP-EMIT-VSETVL-TAX]（hot 路·全 4 格）。**零 perf green**：roll = icache/code-volume/vsetvli 杠杆（v-insn −86~95%·vsetvli −87~99%·hot tile-RT 消）·**非指令密度杠杆**·**非 perf mover**（M=1 broadcast 密度结构性超 stock·roll 不改密度·[CASE-MICRO-E2E] decode memory-wall）。价值 = **C1 extensibility 家族证**（whole-K-nest roll capability 非单格·跨 5 格 K-quant GEVM 泛化）+ **C2 marginal-cost**（capability-keyed 低成本复用·无新机制）。**非 perf 数字**。

**诚实标注**：hot-path tile-RT 消达成·但 numHalves=2 cold residual vs\*r.v（q2:12/q3:2/q4:6/q6:0）说明 dual-strip resident-bank 未达字面全 0（预期·非 hot 税·同 q5_K）。**不夸大成"全 0"**。板 numeric A/B DEFERRED（kernel-axis·construction-identity + q5_K host-oracle 先例）。

## 5. 复现
```
WEFT=build-weft/bin/weft-opt ; TR=/usr/lib/llvm-20/bin/mlir-translate
TC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin
GPP=$TC/riscv64-unknown-linux-gnu-g++
for f in q2 q3 q4 q6; do
  $WEFT test/Conversion/RVV/rvv-to-emitc-repack-gemv-${f}-K-q8-K.mlir        --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp > raw/${f}k_gevm_default_AFTER.emitc.c
  $WEFT test/Conversion/RVV/rvv-to-emitc-repack-gemv-${f}-K-q8-K-rolled.mlir --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp > raw/${f}k_gevm_rolled.emitc.c
  diff raw/${f}k_gevm_BASELINE_default.emitc.c raw/${f}k_gevm_default_AFTER.emitc.c   # default 0-drift
  $GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -S -xc++ raw/${f}k_gevm_{default_AFTER,rolled}.emitc.c
done
# hot element-loop bodies 全 0 vs*r.v / vsetvli (raw/metrics_summary.txt §B)
(cd build-weft/test && lit Conversion/RVV)   # 247/247
```
**禁 git · 禁上板 · sealed/REDESIGN-B/q5_K-rolled/GEMM-兄弟/colgroup 零改（git diff 证）· byte-exact 逐格硬门全过。**
