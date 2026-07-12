# Experiment Reference

## Status

This spec is a validation reference. It must not decide system structure.

Weft-RV MLIR is first a capability-driven RISC-V execution layer. Experiments test whether that design holds.

## Hardware Conditions

具体硬件环境（RVV main、IME、RISC-V Sophgo/offload）是当前事实，会变；权威定义在 [../capability-model/profiles.md](../capability-model/profiles.md)，不在本验证参考里重复。下面只保留与证据解释相关的、durable 的部分。

Repeatable bounded hardware/toolchain evidence should be captured with
`scripts/rvv_remote_probe.py`. Its artifacts are written below
`artifacts/tmp/rvv_probe/<run-id>/` and include sanitized command logs plus a
JSON summary of uname/kernel, architecture, hart count, clang/cmake
availability, bounded RISC-V/vector CPU hints, non-interactive sudo
capability, and the minimal hand-written RVV intrinsic compile/run result.
The JSON artifact may also include a sanitized `capability_facts` section for
the compiler-facing profile boundary. Those facts are input to the plugin-local
C++ RVV capability profile, which validates them and populates
`TargetCapabilitySet`; they are not themselves compiler internals or proof that
Weft-RV emitted executable RVV code.

This probe is a prerequisite evidence source for future RVV compiler claims,
but it is not itself a Weft-RV compiler correctness, runtime, supported
emission, or performance artifact.

## Research Questions

### Q1: Can Weft-RV generate valid code on real RVV hardware?

Objects:

```text
matmul
batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

These objects calibrate RVV coverage and future frontend proof. They do
not make current high-level Linalg/frontend lowering the source authority.
Current RVV codegen claims must still flow through selected `weft.exec`
variants, typed `weft_rvv` bodies, RVV plugin legality/realization, provider
routes, and common EmitC.

Comparisons:

```text
MLIR linalg/vector default lowering
LLVM auto-vectorization
scalar/OpenMP baseline
hand-written RVV kernels if available
existing AI-Benchmark RVV kernels if usable
```

MLIR Linalg/Vector default lowering is comparison/reference only unless a
future frontend task explicitly selects it.

Metrics:

```text
correctness
single-thread performance
multi-thread performance
compile success rate
performance over default lowering
variant-local tuning benefit
```

### Q2: Does capability model participate in pass decisions?

Profiles:

```text
RVV only
RVV + offload runtime
RVV + IME
fallback-only profile
```

Expected behavior:

```text
RVV only -> RVV variant + fallback
RVV + offload -> RVV variant + offload variant + dispatch + fallback
RVV + IME -> RVV variant + IME variant + dispatch + fallback
fallback-only -> fallback
```

Metrics:

```text
generated variants match capability
illegal variants are rejected by verifier
dispatch conditions are correct
capability changes alter pass decisions
diagnostics are clear
```

### Q3: Is extension plugin integration local?

Reference process:

```text
system has mature RVV plugin
add offload plugin
add IME plugin
measure core pass changes and plugin boundary
```

Metrics:

```text
core pass modified LOC
plugin LOC
new capabilities
new ops/types
new variant generators
supported high-level op count
extension-specific branches in core pass
reuse of weft.exec.variant / dispatch / verifier orchestration
```

### Q4: Can runtime-offload capability join the same execution layer?

Q4 validates runtime-offload capability join; offload routes fail closed until a
real producer exists (见 core-invariants I7). It must not introduce
source-front-door or offload artifact authority (these routes fail closed)。

Objects:

```text
large matmul
conv
transformer block or MLP block if runtime supports
```

Comparisons:

```text
RVV CPU variant
Sophgo offload variant
RVV + offload dispatch
fallback
```

Metrics:

```text
offload threshold
end-to-end latency
host-device transfer overhead
runtime launch overhead
shape-size effect on selection
fallback correctness
```

Hard rule:

```text
This validates runtime-offload capability, not custom RISC-V ISA.
```

### Q5: Can plugin-local matrix-extension integration be shown (IME second family)?

Q5 covers IME plugin-local matrix-extension integration; IME runtime/performance
claims need real IME hardware/toolchain evidence (I8).

Objects:

```text
matmul
batched matmul
attention qk/av block
MLP dense block
int8/fp16/bf16 dot-like kernels
```

Comparisons:

```text
RVV variant
IME variant
fallback
hand-written or vendor IME kernel if available
```

Metrics:

```text
core pass modified LOC
IME plugin LOC
IME variant generation coverage
IME legality verifier effectiveness
IME emission success
performance over RVV for suitable kernels
```

## Ablation References

Capability model ablation:

```text
with capability-driven variant generation
without capability-driven variant generation, using fixed RVV path
```

Plugin locality ablation:

```text
plugin-based integration
hand-coded extension branches in core pass
```

Variant selection ablation:

```text
static RVV selection
capability-aware variant selection
dispatch with offload threshold
```

## 证据格与状态枚举（引 testing 层）

一切进 CI 的证据落成测量格 / 结构证明格，状态 ∈ `{measured|stale|board-pending|open|n_a}`；这两型格与三条铁律（`open` 永不可正文引用、指纹变即 `stale`、跨会话不比）的**权威声明在** [../testing/mlir-testing-contract.md](../testing/mlir-testing-contract.md) 的"格 schema 二分"。本层按引用对齐，不重定义。效应判定的 T-N 噪声地板资格前置、对手解析探针与对手类词表同样在 testing 层声明。

## 六态 provenance 清单机检（[L-8] 执法）

六态阶梯（`absent → emittable → dispatch-wired → constructed-weak → constructed → covered`）定义在 core 科研总纲 [K-4]，此处只写**机检口径**——把 [L-8] 从 prose 纪律变成脚本可判：

- 每个机制构造的 body 在发射时写出其**模式原语 ID 列表**（provenance 清单）。
- **强义 `constructed`** = 清单存在 ∧ 无不透明手写 helper（算术主体是模式库原语）。
- **弱义 `constructed-weak`** = 描述符选择的手写片段（算术主体是被选择的手写 helper）。
- 六态从此脚本可判、不可辩解；弱义充强义 = 违宪（[L-8]）。清单是被发射的可测工件，其断言按 testing 层 `HARNESS`-source vs 运行期的来源区分执行。
- 硅封状态（每板 objdump golden）单列跟踪，不并入六态。现值住 CI（[GOV-1]），不写进本 spec。

## 四覆盖率口径（[COV-2]；C_construct 只计强义）

- **C_dispatch** = ≥dispatch-wired 的 kernel 数 / 分母。
- **C_construct** = ≥constructed（强义）/ 分母 —— 燃减主指标与论文口径（[L-8]）。
- **C_construct+** = ≥constructed-weak / 分母 —— 过渡指标，**报告不 gate**。
- **C_attr** = 按 [D-4] 三级分级（编译期 ^CT / 装载期 / 运行期 ^RT），**不报笼统比例**。
- 分母 = 负载域 kernel 清单（[COV-1]），钉 ggml commit 后定稿。四指标现值、烧减曲线、六态计数一律住 CI/执行总纲，spec 只钉口径。

## 双层行键（覆盖率口径 vs 性能口径）

- **分母键（覆盖率口径）** = `(算子, 格式[, 形状类])`，每对只计一次；覆盖态 = 其全部变体行的最佳六态（防多路径重复计数虚增）。
- **测量行键（性能口径）** = `(算子, 格式, 路径, arity, 形状类)` + 变体 ID + 选中方式 ∈ `{mechanism-selected, forced}`；**产品级主张只引 mechanism-selected 行，forced 行只用于消融科学**。
- **形状类必入键**：`{decode-GEMV(M=1), prefill-GEMM(M 扫描), micro-fixed}` —— micro↛e2e 与 IME 交叉点实验的载体。

## 传导会计（micro→e2e Amdahl 四列）

- 核级赢**不得**直接写成 e2e 赢；须给 Amdahl 四列：`相内时间占比% → Amdahl 预测相级Δ → 实测相级Δ → 传导效率`。无传导列的核级赢不得声称传导到 e2e。
- micro 赢不在 e2e 出现须如实披露；regime（compute-bound / memory-bandwidth-bound / per-block reduction-latency-bound）须声明，不得藏在更大的数字后。
- **带宽受限内核以 parity 为零假设**（parity 是物理确认，不是失败）。
- **memory-轴 epilogue 融合（消一趟中间张量往返）= 机制展示 / 方法学素材,不作性能支柱**：其 isolated A/B（融合 vs 自己两趟）字节消除即便 board-proven,也**不得**暗示 whole-model e2e 收益。当融合的中间张量在整模型尺度 **cache-resident**（< L2/L3）时,isolated micro（>L3 强迫 DRAM 往返）的字节赢**不传导**,whole-model e2e 被 **Amdahl（该算子相内占比切片）+ cache 驻留**双双稀释到噪声地板下 = **结构性 null**（非测量缺失、非失败）。此类 e2e 须作 **micro↛e2e 传导会计的正面档案教材**登记,成色 = `e2e-diluted-Amdahl`。**锁定标准表述形态**：*「kernel 级字节轴机制已证;whole-model 中被 Amdahl（该算子占相内 X%）与 cache 驻留（中间张量 < L2/L3）稀释,e2e 不显著——传导会计精确预测了这一点」*。**绝不**据此声称 Win-B/Win-C 或 e2e beat。

> 表 schema 与落点（T0/T3/T6/T7 等）详情引 [docs 实验总纲 §2](../../../docs/Weft-RV_实验总纲v1.md) + `experiments/`；本层只写口径与门，零现值。

## N3 Performance-Claim Discipline (baselines — durable contract)

N3 ("capability/resource-aware tune that measurably wins") claims must obey a fixed baseline discipline so a
number names a real contribution, not an artifact of a weak comparand. The comparand classes referenced below
(**factory-dispatched** = the only beat baseline; **algorithm-matched** = diagnostic; **naive-RVV** /
**scalar-oracle** = sanity) are the adversary-class taxonomy declared in the testing layer
([../testing/mlir-testing-contract.md](../testing/mlir-testing-contract.md) 对手解析探针); a vs-framework cell
without an adversary-probe artifact is INVALID:

- **Scalar is NEVER a contribution baseline.** vector-vs-scalar measures "we vectorized at all" — which
  MLIR/autovectorization already provide. It may appear only as an internal sanity check, never as a reported
  multiple. (The old "wide ÷ scalar 4–15×" framings are retracted.)
- **Registered Win ladder (定案 taxonomy; A/S = sanity, B = kernel-level contribution, C = phase-level; opponent-class 1:1 per [L-7]):**
  - **Win-A** = the compiler-automatic *tune* paired ablation (narrow↔wide LMUL/strip, same kernel/board/session). Baseline =
    the SAME kernel with the tuned knob OFF (both arms compiler-emitted; only the knob differs). Opponent = naive. **sanity — never a contribution.**
  - **Win-S** = a capability-tuned kernel beating BOTH true scalar AND an instruction-level-verified naive vector (two-axis, kernel-level).
    Opponent = {scalar-oracle, naive-RVV}. **sanity class.** *(alias: this is the two-axis datapoint historically mislabeled "Win-B";
    renamed to the sanity class — the ledger keeps an **alias note** rather than silently rewriting history.)*
  - **Win-B** = a generated kernel for an algorithm/layout, measured vs the **framework's OWN dispatched kernel** (contribution axis).
    Opponent = factory-dispatched. **The B1/B2 subscript is FIXED by the opponent-resolution probe (§ testing) = that board's out-of-box path**, never chosen by hand:
    - **Win-B1 = beating the framework's block-dot, WHERE the probe says the out-of-box path IS block-dot** (e.g. a VLEN where the framework
      ships no repack — ggml's `case 128: break` → no q4_0 repack at VLEN128). A GAP-FILL *coverage* result against the real shipping
      baseline — but any "acceleration/beat" wording stays gated on [PERF-1] 八门 / [NG-4] (registered Wins never enter beat 语境, [L-7]);
      it is a frontend (added-algorithm) result, not a backend-codegen win.
    - **Win-B2 = beating the framework's OWN repack, WHERE the probe says the out-of-box path IS repack.** The honest success criterion is
      PARITY (matching the expert's hand-written kernel).
    - **Beating the framework's block-dot on a board whose out-of-box path is repack is NOT any Win-B** — it is an `algorithm-matched`
      *diagnostic* increment (wrong opponent), and **it is NOT entered in the Win ledger**.
    NEVER scalar / naive / `_generic`. *(Changing the algorithm/layout is a frontend/library contribution, NOT a backend novelty; weight-storage
    repack is the offline-prepack class — Marlin/AWQ/CUTLASS analog — that even Triton leaves outside the compiler.
    See the frontend-vs-backend discriminator in [system-positioning](../architecture/system-positioning.md).
    Win-B is honest-measurement discipline, not a backend claim.)*
  - **Win-C** = a **phase-level end-to-end win** (llama-bench, prefill/decode 分相, vs the SAME-version out-of-box ggml),
    reported ONLY **after the [PERF-1] 八门 are all green**. Opponent = e2e factory ggml. **This is the only phase-level
    contribution; a kernel-level micro win is never relabeled as a phase win.**
- **Attribution rigor (applies to every structural claim, NOT a separate Win):** a pass/structure ON-OFF number is NOT automatically a
  *structural* win. If the ON arm changes BOTH the structure AND an incidental emission property (e.g. it avoids a memory round-trip
  the OFF arm's emitter happens to incur), the ON/OFF delta conflates the two. To attribute the win to the STRUCTURE you MUST
  decompose against a *competently-emitted baseline of the SAME structure* (e.g. a register-kept-accumulator per-iteration reduction).
  If that same-structure competent baseline TIES the ON arm, the structural contribution is **NULL** — report only the ON/OFF number
  with its real but non-structural mechanism. *(A real instance: a deferred-vs-per-iteration reduction pass showed a large ON/OFF gap,
  but a register-kept per-iter control tied the deferred arm — the gap was a per-iter `out[0]` memory round-trip, not
  reduction-structure latency; the structural novelty was NOT demonstrated.)*
- **Both harnesses are required and not interchangeable:** an isolated single-core microbench (clean
  ablation) AND a real end-to-end run (catches integration/memory effects). A microbench win that does not
  appear e2e must be disclosed as such; regime-dependence (compute-bound vs memory-bandwidth-bound vs
  **per-block reduction-latency-bound** — the block-quant decode regime is the latter, not bandwidth, and a
  repack that dissolves the per-block reduction wall is what transplants) must be stated, not hidden behind
  the larger number.
- **Every reported cell carries an evidence-status tag** — the canonical five-state enum
  `{measured | stale | board-pending | open | n_a}` declared in the testing lattice
  ([../testing/mlir-testing-contract.md](../testing/mlir-testing-contract.md) 格 schema 二分; `open` renames the
  old `presumed`, `stale` is added). An unmeasured cell defaults to NO claim — never to the success state. "presumed parity / presumed null" is not a
  result, it is an open measurement; banking it as success is the recurring over-optimism failure mode. After any
  refactor, prior numbers are STALE until re-measured on the named profile — a parity/win is a target re-measured
  per build, not a banked metric.
- **A repack / block-as-lane kernel's perf claim must name the VLEN regime AND compare vs the framework's
  ACTUAL same-VLEN baseline** (its hand-tuned VLEN-native kernel if one ships, else its generic fallback) —
  never a different-VLEN kernel. Whether the repack wins is set by **competitor strength × compute-density**,
  NOT by the repack's lane-shape (the 2×8 mf2 form IS the correct VLEN128 tiling; the 2-strip split is
  beneficial ILP — the q8_0 NARROW>WIDE ISO datum). The repack wins only when the same-VLEN fallback is a
  HEAVY kernel it out-streams; against a LEAN fallback or a hand-tuned VLEN-native kernel it LOSES, and the
  tune should DECLINE it (select the fallback = match the framework's own algorithm) — declining = matching
  the framework = loss-avoidance, NOT a backend N3 contribution. Choosing repack-vs-fallback is an
  algorithm/layout choice (frontend/library), NOT capability-driven lowering of a fixed op+layout (the
  backend N3 face); see [system-positioning](../architecture/system-positioning.md) N3 boundary. It is NOT a
  kernel bug and NOT a Win-B speedup. *(A real instance: the same repack at one fixed correct lane-shape WON
  vs a HEAVY same-VLEN fallback but LOST vs a LEAN one and vs a hand-tuned VLEN-native kernel — competitor
  strength, not lane-shape, set the outcome.)*
- Performance claims still require real `ssh`-hardware evidence on a named profile (I8); engagement of the
  emitted kernel must be proven (e.g. an ENGAGED marker / objdump of the FINAL staged binary), not assumed.
- **A new-hardware-unit / build-swap claim needs a can't-possibly-help control.** When a number compares two
  separately-built artifacts (e.g. an IME-enabled lib vs a non-IME lib), it confounds the new unit with ANY
  global toolchain/codegen difference between the builds. Include a CONTROL regime where the new unit
  *physically cannot* contribute (e.g. M=1 memory-bandwidth-bound decode for a matrix/MAC array); if the "win"
  appears THERE too, it is a global codegen artifact, not the unit. A clean unit-isolation requires the SAME
  toolchain with only the unit toggled (e.g. `SPACEMIT=ON` vs `OFF` on one compiler), NOT two independently
  built libs. *(A real instance: a reconstructed IME-enabled lib showed a "prefill" speedup but a similar
  speedup in M=1 decode where the matrix unit cannot help → the gain was global codegen, not the unit; the
  unit-isolated e2e win stayed NULL.)*

## Forbidden Interpretations

Do not claim:

```text
Sophgo offload is RISC-V custom ISA extension.
Ordinary tile-size tuning is the main theory.
A vector-vs-scalar speedup is an N3 tune contribution.
A microbench win is an end-to-end win without an e2e measurement.
A pass-ON/OFF speedup is a structural-transform contribution without a same-structure competently-emitted control.
A build-swap speedup (e.g. IME-lib vs non-IME-lib) isolates the new hardware unit without a can't-possibly-help control regime.
AME is current verified primary hardware.
Any future extension never needs core changes.
Weft-RV is a new high-level tensor IR.
Structured kernel validation objects are current source-route authority.
Offload or IME dispatch is required before RVV typed-route maturity.
Source-front-door generated artifacts prove RVV maturity.
```

Use:

```text
Sophgo offload is runtime-offload capability.
Tuning is a system ability inside capability-aware variant selection.
Current mainline is RVV; later IME validates new extension plugin integration.
Extensions that map to existing interfaces support plugin-local integration.
Weft-RV is a RISC-V execution layer after high-level MLIR.
```
