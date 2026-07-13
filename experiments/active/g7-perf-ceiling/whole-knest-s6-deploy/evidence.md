# G7 whole-K-nest S6 roll — q5_K GEVM ROLLED envelope **DEPLOYMENT** (emitter-implanted · kernel-axis C1)

> **线定位（诚实前提）**：把 G1 已证【结构可行∧byte-exact∧预算合规】的 whole-K-nest S6 roll **从 hand-construct 实装进真发射器**（`emitRepackKQuantGemvBodyQ5K`），capability-keyed·仅 q5_K·byte-exact 硬门。**= kernel-axis C1 emitter-maturity artifact**（closes hot-path [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]）·**零 perf green**（rolled ~1.78× 指令密度 > 1.10× 子门 + [CASE-MICRO-E2E] decode memory-wall；roll 是 icache/code-volume/vsetvli 杠杆·非指令密度杠杆）。
> 依据 canon：G1 casefile `../whole-knest-s6-feasibility-G1/`（commit dad1e1c2）· [VERIFY-LADDER] · [K-10] 结构级 · [CASE-MICRO-E2E] · [GAP-P1] · [CASE-COMPILER-ASYMMETRY]（部署 gcc-15.2·rvv 板 VLEN128）。
> **纯本地 forced build·无需上板**（byte-exact + objdump 即足·kernel-axis C1 artifact）。禁 git·禁触碰 sealed/REDESIGN-B/q3_K-native-mask/其他 K-quant leaves。

---

## ★裁决 TL;DR

1. **q5_K GEVM whole-K-nest rolled envelope 已实装**（capability-keyed·**仅 q5_K**·byte-exact）。发射器把 dominant per-16-element inner **ii-loop roll 成 runtime `emitc.for`**·per-strip i16 partials `sLoVar/sHiVar` 作 **RESIDENT SSA-register VariableOps**（单一 wide accumulator·seed 于环上·环内 load-accumulate-store·与 sumf/sumi/bsums 同 resident-across-`emitc.for` 机制）。per-element 解码 tile（nibble+qh）**环内即消即用·从不 materialize 过栈**。
2. **hot-path tile-roundtrip CLOSED**：8 个 runtime ii-loop（.L4–.L11·j×pair×k 展开副本）**全部 0 `vs*r.v` ∧ 0 `vsetvli`**（objdump 逐环证）。示例 .L4 body = 4×vle8 + 4×vwmacc.vx + 解码 bit-dance·accumulator v3/v6/v7/v8 驻寄存器跨迭代·backward branch `bne .L4`。**[GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] hot 路全塌** + **[GAP-EMIT-VSETVL-TAX] hot 路全塌**。
3. **整体账**（部署 gcc-15.2 -O3·VLEN128 numHalves=2）：v-insn **8751→609（−93.0%·14.4×）**·vsetvli **2943→80（−97.3%·36.8×）**·vs\*r.v **209→9（−95.7%）**·vle8 1378→57·asm 25121→1049（−95.8%）。
4. **★9 residual vs2r.v（如实标·非夸大）**：**全在 COLD 区**（prologue + block-loop-top accumulator ZERO-SEED + block-boundary flush）·**非 hot 解码 nest**。= numHalves=2（VLEN128 dual-strip）的 **RESIDENT accumulator bank**（sumi/bsums/sumf 6×m2=12 vreg + 8 scale strips）在 cold block 边界被 gcc 溢出几个。**G1 casefile 的 0 是 numHalves=1 single-strip**（bank 减半）·numHalves=2 bank 翻倍→9 cold seeds 残留。**这是 resident-bank setup 税·非 tile-roundtrip 税·预期**。核心目标（hot-path tile-RT 消）**已达成**。
5. **byte-exact 硬门全过**：① 默认（unrolled）路 emitc **byte-IDENTICAL** before/after（diff==0·shipped 默认零漂移·rolled 是纯新增变体）② rolled-vs-unrolled **DYNAMIC op-stream 恒等**（vwmacc16 512==512·vwmacc_vv 64==64·全 fold 恒等·仅 ii-loop materialize 成 runtime for）③ q5_K host decode oracle **0/32**（stock==current==KNEST）④ numeric 板 A/B = DEFERRED（kernel-axis 无板·依 construction-identity + G6-B P3 板先例）。
6. **零回归 + 隔离实证**：Conversion/RVV lit **243/243 PASS**（was 242·+1 新 rolled fixture）·其他格 lit 全未改全过·git diff 仅 4 文件·`.cpp` hunk 限 fwd-decl + q5_K dispatch 分支 + q5_K leaf·**其他 leaf（q2/q3/q4/q6_K GEVM · sealed GEMM · q5_0/q5_1 REDESIGN-B · q5_K/q2_K GEMM）零 diff 行**。

---

## 0. 实装（改了什么·capability-keyed·增量·强隔离）

**4 文件·additive/byte-exact-neutral**：

| 文件 | 改动 | 性质 |
|---|---|---|
| `include/Weft/Dialect/RVV/IR/RVVOps.td` | `TypedRepackGemvLoopBodyOp` 增 `OptionalAttr<StrAttr>:$emit_loop_schedule`（镜像 GEMM op 既有 knob）| additive·ABSENT⇒default·existing fixtures 不携带⇒零漂移 |
| `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | GEVM op verifier 增 `emit_loop_schedule` spelling 校验（bounded {"unrolled","rolled"}·fail-closed I7）| additive 校验 |
| `lib/Conversion/RVV/RVVToEmitCInternal.h` | `emitRepackKQuantGemvBodyQ5K` 签名 +`bool rolledMainTerm` | 单 leaf 签名 |
| `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | (a) `resolveRepackMainTermRolled` 前向声明（复用 GEMM 既有 resolver）(b) q5_K GEVM dispatch 分支内 resolve schedule + 传参（**在 `if(decodeModel=="q5_K")` 内·q2/q4/q6/q3 dispatch 零改**）(c) `emitRepackKQuantGemvBodyQ5K` leaf 内 `if(rolledMainTerm){rolled ii-loop}else{原 unrolled 逐字}` | leaf-local |

**capability-key 机制**：schedule 经 `resolveRepackMainTermRolled(getEmitLoopSchedule(), ...)` 解析 —— 显式 stamp `"rolled"`/`"unrolled"` 优先·ABSENT ⇒ capability-derived 默认（frozen sentinel = unrolled·shipped 零改）。**仅 q5_K GEVM leaf 收 `rolledMainTerm` 且带 rolled 分支**；其他 K-quant GEVM leaf 不收此参·完全忽略 schedule。∴ **只有 q5_K 能走 rolled·且仅当显式 stamp**。

**隔离铁律遵守**：`git diff` 证 sealed GEMM leaf(`emitRepackGemmQ4LaneWiseIntegerCore`) · q5_0/q5_1 REDESIGN-B(`emitRepackQ4LaneWiseIntegerCore`) · q3_K native-mask(`emitRepackKQuantGemvBodyQ3K`) · q2/q4/q6_K GEVM leaves · q5_K/q2_K GEMM bodies **零 diff 行**。

---

## 1. objdump 部署验证（hot 路 tile-RT 全消·9 cold residual 如实标）

见 `raw/metrics_summary.txt` §A。核心：
- **8 hot ii-loop（.L4–.L11）全 0 vs\*r.v ∧ 0 vsetvli**（Python 逐环扫描证·`raw/q5k_gevm_rolled.gcc15.O3.s`）。
- 整体 vs\*r.v 209→9·vsetvli 2943→80·v-insn 8751→609。
- 9 residual = cold accumulator-bank seeds/flushes（numHalves=2 dual-strip·§A2）·**非 hot 解码 tile**。

hot .L4 body（真 objdump·精简）：
```
.L4:
  vle8.v v2,0(s7); vle8.v v5,0(s4); vle8.v v1,0(s4-504)   # weight nibble + qh(2 strips)
  lb s5,0(a4); lb s4,32(a4)                               # activation i8
  vsrl/vand/vsll/vor ...                                  # q5_K 4-bit nibble + qh 5th-bit 解码
  vwmacc.vx v8,s5,v5;  vwmacc.vx v6,s4,v11                # strip h=0 sLo/sHi dot
  vwmacc.vx v7,s5,v2;  vwmacc.vx v3,s4,v1                # strip h=1 sLo/sHi dot
  bne t3,s7,.L4                                           # runtime ii-loop (ROLLED)
# NO vs*r.v · NO vsetvli · accumulators v3/v6/v7/v8 resident
```

---

## 2. byte-exact 硬门（见 §5·全过）

见 `raw/metrics_summary.txt` §B：① 默认路 byte-identical ② dynamic op-stream 恒等 ③ oracle 0/32 ④ 板 A/B deferred（依 construction + G6-B 先例）。

---

## 3. scope

- **q5_K GEVM rolled = 完整**（ii-loop 全 roll·capability-keyed·byte-exact·hot-path tile-RT 消）。
- **家族其他格 = DEFERRED**（q2/q3/q4/q6_K GEVM 仍 full-unroll·未收 `rolledMainTerm` 参）。本线明记只做 q5_K（代表格·素材最厚·风险隔离到单格）。家族推广 = 后续 line（每格独立 leaf·同 copy-then-adapt·须各自 byte-exact 硬门）。
- **residual 0 化 = DEFERRED / 非本线**：要 numHalves=2 也达 0 需 roll h-strip 或 whole j/pair/k nest（casefile full-nest 路·撞 j==0/j==1 scale-unpack 条件分支·风险高·收益递减——hot 路已净）。本线判 ii-roll = 正确高价值 byte-exact 隔离交付。

---

## 4. 诚实定性（明记零 perf）

**kernel-axis C1 emitter-maturity artifact**·closes **hot-path** [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]（8 hot ii-loop 全 0 vs\*r.v/vsetvli）+ 塌缩 [GAP-EMIT-VSETVL-TAX]（hot 路）。**零 perf green**：
- rolled DYNAMIC 指令密度 per-output ~232 vs stock 130 = **~1.78× > 1.10× 子门**（M=1 element-wise broadcast 密度结构性超 stock·roll 不改此密度）。
- [CASE-MICRO-E2E] decode memory-wall（GEVM M=1 削重建 e2e 受限）。
- roll = **icache / code-volume / vsetvli 杠杆**（v-insn −93%·asm −96%·vsetvli −97%·hot tile-RT 消）·**非指令密度杠杆**·**非 perf mover**。价值 = C1 plan 库 emitter 成熟度深度（家族最深可行靶的首格实装）·**非 perf 数字**。

**诚实标注**：核心目标（hot-path tile-RT 消）达成·但 9 cold residual vs2r.v 说明 numHalves=2 resident-bank 未达 casefile numHalves=1 的字面 0（dual-strip bank setup 税·预期·非 hot 税）。**不夸大成"全 0"**。

---

## 5. 复现
```
WEFT=build-weft/bin/weft-opt ; TR=/usr/lib/llvm-20/bin/mlir-translate  # (mlir-translate-20)
TC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin
GPP=$TC/riscv64-unknown-linux-gnu-g++
# 默认(unrolled) vs rolled emitc
$WEFT test/Conversion/RVV/rvv-to-emitc-repack-gemv-q5-K-q8-K.mlir        --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp > raw/q5k_gevm_BEFORE.emitc.c
$WEFT test/Conversion/RVV/rvv-to-emitc-repack-gemv-q5-K-q8-K-rolled.mlir --weft-rvv-lower-to-emitc | $TR --mlir-to-cpp > raw/q5k_gevm_rolled.emitc.c
$GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -S -xc++ raw/q5k_gevm_{BEFORE,rolled}.emitc.c   # -> .s
#   hot-loop 扫描：8 ii-loop(.L4-.L11) body 内 vs*r.v/vsetvli == 0
# byte-exact oracle
../whole-knest-s6-feasibility-G1/raw/be_q5k_host       # 0/32
# lit (243/243)
(cd build-weft/test && lit Conversion/RVV)
```
**禁 git · 禁上板 · 其他格/sealed/REDESIGN-B/native-mask leaf 零改（git diff 证）· byte-exact 硬门全过。**
