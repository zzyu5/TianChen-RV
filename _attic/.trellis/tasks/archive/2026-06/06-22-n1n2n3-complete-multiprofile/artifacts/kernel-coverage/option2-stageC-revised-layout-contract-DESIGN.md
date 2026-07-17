# Option-2 STAGE C — REVISED: weight LAYOUT as compiler INPUT / declared OUTPUT CONTRACT (DESIGN, 2026-06-24)

READ-ONLY design. NO `lib/` edits. **This SUPERSEDES the framing of
`option2-stageC-weightpacking-DESIGN.md`** ("the compiler must drive model-load weight-packing /
hard per-tensor limit / ship partial-C, defer the rest"). It KEEPS that doc's file:line facts
(where ggml packs, the repack op's x16 needs, the GEMM/prefill defect) and B's selection/audit
(`option2-stageB-selection-DESIGN.md`, committed 06eb0ff8). It REPLACES the architecture frame.

The user supplied the better industrial frame: **an operator-compiler takes weight LAYOUT as an
INPUT and DECLARES it as an OUTPUT CONTRACT. It does NOT own model-load weight-packing.** This is
the right design — and it makes the COMPILER side clean. The honest correction the rest of this
doc enforces: **the per-tensor limit does NOT dissolve under this frame; it RELOCATES from the
compiler to the SYSTEM, and therefore lives inside every e2e number.**

---

## 0. THE THREE-LINE FRAME (carry it EXACTLY — getting this honest is the whole point)

1. **Layout-as-input / declared-contract (ADOPT).** The compiler EMITS the selected kernel AND
   declares its weight-layout requirement as an output contract (an op attr + a route-descriptor
   field: *"this operator instance requires weights in `layout=x16`"* or *"=plain"*). It compiles
   whatever layout it is handed; a JIT harness invokes it per-call with the layout context. **From
   the COMPILER's standpoint there is NO per-tensor limit.** Clean, general, lit-testable.

2. **The limit RELOCATES, it does NOT dissolve.** At any instant a tensor's bytes are in ONE
   physical layout. Making layout an *input* moves the DECISION to the runtime; it does not make
   the bytes be two things at once. So the limit LEAVES the compiler but STAYS in the SYSTEM — and
   therefore in any e2e NUMBER (e2e is measured where bytes have one layout). The old doc's "no
   clean in-compiler fix / hard per-tensor limit" is REPLACED by: *compiler clean, system pays.*

3. **Every e2e number names its mechanism + cost.** For a tensor that wants one layout in decode
   and another in prefill, the SYSTEM does exactly ONE of:
   - **(SYS-a) DUAL-STORE both layouts** = 2x memory for that tensor.
   - **(SYS-b) REPACK PER-CALL (JIT)** = compute cost per call. The memory-wall finding predicts
     this does **NOT amortize for decode** (M=1, per-token, bandwidth-bound) → JIT-repack is a
     **PREFILL** win (M≫1), **NOT** a decode one. *Promising a decode JIT-repack win without
     measuring amortization is the session's transplant error — DISALLOWED.*
   - **(SYS-c) PICK ONE layout** = the current per-(tensor,regime) result; the non-served regime
     runs its non-native path.

   The compiler can emit kernels for ALL THREE. The SYSTEM picks ONE and pays.

> **LETTER-COLLISION WARNING (do not conflate).** The OLD doc's (a)/(b)/(c) named the
> *materialization mechanism* (compiler-emitted packer / in-IR transform / build-load coop). THIS
> doc's (a)/(b)/(c) — written `SYS-a/SYS-b/SYS-c` to keep them distinct — name the *system payment*
> (dual-store / JIT-per-call / pick-one). They are different axes. The old (a) "compiler-emitted
> packer" survives in this doc only as **C1's isolated packer** — a *capability proof* that the
> compiler CAN materialize x16, NOT the e2e data path.

---

## 1. STORE vs PER-CALL — the distinction the whole honesty hinges on (READ FIRST)

The plain→x16 transform (`make_block_q4_0x16`, the `^0x88` bake + 16-way interleave) can be paid in
two physically different places, and conflating them is exactly the transplant error:

- **STORED-x16 (paid ONCE, at load).** The tensor's bytes are converted to `block_q4_0x16` at
  model-load and stay x16 for the tensor's whole lifetime. Every later compute call reads
  pre-x16 bytes. The transform cost is amortized over the entire run.
- **JIT-REPACK (paid PER CALL).** The tensor stays in some base layout; each `mul_mat` call
  repacks the slice it needs just-in-time, then runs the x16 kernel.

The C920-VLEN128 decode **1.22→2.6x** win (recorded in the old doc + MEMORY
kernel-wins-dont-transplant as "q4_0 GEVM 1.22→2.6x") is a **STORED-x16** win: layout cost paid
ONCE at load, decode reads pre-x16 bytes, and the win is that the **x16 KERNEL is faster**
(lane-wise `vwmacc`, the per-block `vredsum` wall is gone). **It is NOT per-call repack.**

What the memory-wall finding says "does not amortize for decode" is the **per-call LAYOUT
TRANSFORM (JIT-REPACK)** — *not* the x16 kernel. So there is NO contradiction between the old
doc's "decode keep is RECORDED REAL" and this frame's "JIT does not help decode": once
STORED ≠ PER-CALL, both are true.

**Honesty rule for the e2e table (§4):** any cell that reports an x16 decode win MUST name
`stored-x16` (SYS-c pick-one-x16 or SYS-a dual-store). Writing "decode-repack-keep 1.22→2.6x"
WITHOUT naming the layout-store reads as the DISALLOWED JIT-decode claim.

---

## 2. WHAT C KEEPS FROM THE OLD DOC (file:line facts — unchanged)

These are mechanism FACTS, independent of the framing; carried verbatim:

- **ggml OWNS the plain→x16 transform at load** — `make_block_q4_0x16` (the `^0x88` bake),
  reached via `repack_q4_0_to_q4_0_16_bl` (`llama.cpp/ggml/src/ggml-cpu/repack.cpp:3358/3939`),
  triggered at load by `init_tensor` (:4727) / `set_tensor`→`repack()` (:4739). The compiler-emitted
  kernel CONSUMES already-packed bytes; it does not pack
  (`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:2734+`, :2799 "*repacked nibbles already
  carry the ^0x88 bias*"; emitted signature `…(size_t n, float* s, const uint8_t* vx, …)` — `vx`
  is pre-interleaved, NO packing prologue arg).
- **The buffer-type SELECTION gate** `get_optimal_repack_type` (`repack.cpp:4528–4595`): **VLEN256
  → `case 256` returns `q4_0_16x1` (ggml AUTO-repacks)**; **VLEN128 → `case 128: break // TODO` =
  NO upstream repack**, and force-defining `__riscv_zvfh` ALONE is insufficient (the `break` sits
  inside the `#if`; `case 128` still nullptrs → `supports_op` (:4779) kills VLEN128 repack). The
  fedora `ENGAGED@128` markers (`llama-e2e-rvv07.log:97`) prove a `case 128 → return q4_0_16x1`
  edit existed; **that one-line ggml patch is a REQUIRED C4 edit, not a `-D` flag.**
- **The repack op's x16 contract** (`RVVOps.td:4339` `GgmlRepackGemvQ40Q80Op`; verifier
  `RVVDialectWideningOps.cpp:1718`): `weight_block_stride=288`, `weight_interleave=16`,
  `weight_quant_byte_offset=32`, `activation_block_stride=34`, `activation_quant_byte_offset=2`,
  `half_lanes∈{8,16}` (8@VLEN128 / 16@VLEN256, `deriveRepackHalfLanes`,
  `RVVRepackStripWidthMaterialization.cpp:78`), optional `integer_core_lmul` mf2(RVV1.0)/m1(RVV0.7).
- **The prefill-GEMM numerical defect** (`fedora-rvv07/FINDING.md:212–228`): GEVM/decode RVV0.7 is
  bit-exact + coherent; **GEMM/prefill is numerically broken** (chain-shift `m1→i16m2→i32m4→f32m4`
  emission OR the interleaved `ggml_quantize_mat_q8_0_4x8` quantizer). **Carried as a flag:** the
  prefill-keep cell is NOT measured-coherent at C920 yet.
- **No producer in `lib/` today** — Stage A found no `rewriter.create<…ContractionOp>`; every
  concrete contraction op is hand-authored in fixtures. The real producer is a NEW build-side
  component (C3).

What is REPLACED: the old doc's §2 "(c) compiler-DRIVEN / harness-EXECUTED, not materialized" and
its §5/§8 "no clean in-compiler fix / hard per-tensor limit / ship partial-C." Under the new frame
the compiler is clean (it declares a contract); the limit relocates to the system layer (§3.2).

---

## 3. THE CONTRACT MECHANISM (compiler side — CLEAN, no per-tensor limit, lit-testable)

### 3.1 What the compiler declares
The compiler emits, for each lowered operator instance, a **weight-layout contract**: the layout
its emitted kernel REQUIRES of the weight bytes it will be handed. Two carriers, both already in
the codebase's idiom:

**(A) In-IR op attribute (primary, B-extending).** B already stamps a `tcrv_rvv.*` audit triple on
the lowered op (`contraction_algorithm` / `path_selection_reason` / `path_materialization`,
allow-listed at `RVVDialectWideningOps.cpp:910` `isAllowedBlockDotAttr`). C ADDS one field:

```
tcrv_rvv.weight_layout_contract = "x16" | "plain"     (StrAttr; the OUTPUT CONTRACT)
```

`"x16"` on a `repack_gemv` op, `"plain"` on a `block_dot` op. This is a pure ASSERTION about the
bytes the kernel will read — `weight_layout_contract="x16"` ⇔ the op's `weight_block_stride=288`
contract; `"plain"` ⇔ stride-18. The IR cannot tell a plain base from an x16 base (both are
`const uint8_t *`); the attr is the compiler's DECLARED requirement that some layer must honor.

**(B) Route-descriptor metadata field (secondary, harness-readable).** The emit already produces a
`tcrv_rvv.*` route-metadata stream as `{key,value}` push_backs
(`RVVEmitCRouteMetadata.cpp:108+`, e.g. `…route_family_plan`, `gearbox.producer_scope`,
`gearbox.consumer_scope` at :208–215). C appends, alongside them, ONE field:

```
tcrv_rvv.weight_layout_contract = "x16" | "plain"
```

so the build/load harness (§4, which reads the route descriptor, not the IR) sees the contract
without re-parsing MLIR. This is the **C2 deliverable** and the JIT layout-context channel.

**No per-tensor limit on the compiler side.** The compiler emits whatever the selection demands and
declares the matching contract. If a JIT runtime hands it plain bytes it emits/uses the `plain`
kernel; x16 bytes → the `x16` kernel. There is no point at which the COMPILER must choose one
layout for a tensor's whole life — that choice is not its to make. Clean, general, lit-testable.

### 3.2 Where the limit went (the relocation, stated explicitly)
The compiler is now clean BECAUSE the decision left it. A tensor's bytes are still ONE layout at any
instant. So whoever OWNS the bytes — the load-time producer / the JIT runtime / the harness (§4) —
now carries the limit: for a tensor whose decode-regime and prefill-regime want DIFFERENT layouts,
THAT layer pays SYS-a (dual-store 2x mem) or SYS-b (JIT per-call compute) or SYS-c (pick-one). The
limit is conserved; the frame just relocated it to the layer that can actually pay it.

### 3.3 The in-IR bridge (the partial-C C1: repack-selected op → REAL `GgmlRepackGemvQ40Q80Op`)
B left the Repack-selected cell as an audited block-dot stub (`path_materialization="deferred-stage-c"`).
C1 replaces that branch in `RVVLowerQuantContractionPass` with a real
`create<GgmlRepackGemvQ40Q80Op>` carrying the x16 contract — the in-IR half, which is TRIVIAL
because the materialization lives OUTSIDE the IR:

```
case Repack:
  half = deriveRepackHalfLanes(minVLEN);          // 8@128, 16@256 (StripWidthMaterialization.cpp:78)
  create<GgmlRepackGemvQ40Q80Op>(
      weight_base,                                 // SAME SSA pointer (IR can't tell plain from x16)
      activation_base, output, element_count,
      /*nc=*/column_count,                         // the nc stage A carries (why A always carries nc)
      vl)
    with attrs: kind="ggml_repack_gemv_q4_0_q8_0", scale_model="dual-fp16-per-block-d_x.d_y",
                qk=32, weight_block_stride=288, activation_block_stride=34,
                weight_quant_byte_offset=32, activation_quant_byte_offset=2,
                weight_interleave=16, half_lanes=half,
                tcrv_rvv.weight_layout_contract="x16",          // ← the DECLARED CONTRACT (3.1A)
                (optional) integer_core_lmul="mf2"|"m1";
  erase abstract op;
```

The verifier (`RVVDialectWideningOps.cpp:1718`) pins 288/34/32/2/16, `half_lanes∈{8,16}` — the
`create` satisfies it fail-closed. **The op's `weight_block_stride=288` is an ASSERTION the §4
harness must make true; the bridge COMMITS to the x16 contract and TRUSTS the producer.** Lit-only,
no hardware. (The decline branch is unchanged: `block_dot` op + `weight_layout_contract="plain"`,
byte-identical `f810ce6b`.)

---

## 4. THE RUNTIME / HARNESS HONORING (system side — WHERE THE LIMIT LIVES, and pays)

The compiler declared a contract; SOMETHING must read it and provide bytes in that layout. This is
the layer that picks {SYS-a / SYS-b / SYS-c} and pays. Two honoring modes:

### 4.1 Load-time honoring (the llama.cpp e2e demo — STORED layout)
A **producer + load-time pack**:
- **Producer (C3):** a NEW build-side tool (the analogue of the existing
  `cmake_inject.py`/`transform_repack.py`/`arch_repack.cpp` family — build-side, NOT `lib/`)
  enumerates the model's q4_0 `MUL_MAT` weight tensors, authors the abstract `GgmlQuantContractionOp`
  `.mlir`, drives `tcrv-opt --tcrv-rvv-lower-quant-contraction --tcrv-rvv-lower-to-emitc`, and
  collects the emitted `.inc` + the §3.1B route-descriptor (carrying `weight_layout_contract`).
- **Load-time pack (C4):** the harness READS each kernel's `weight_layout_contract` and makes
  ggml's load-time buffer-type decision honor it: for `="x16"` patch `get_optimal_repack_type`'s
  `case 128: break` → `return &q4_0_16x1_q8_0` (the required one-line ggml edit, §2) + per-file
  vector march + `__riscv_zvfh`, so `make_block_q4_0x16` runs ONCE at load → tensor STORED x16; for
  `="plain"` SUPPRESS the `case 256` auto-repack so the tensor stays plain stride-18 for ggml's
  native `vec_dot_q4_0_q8_0`. Then inject the named emitted kernel into the dispatch.
- **This layer picks and pays.** If decode wants plain and prefill wants x16 on ONE tensor, THIS
  layer chooses SYS-c (one stored layout, the non-served regime runs non-native) or SYS-a (two
  stored copies, 2x mem for that tensor). It cannot make the bytes be both.

### 4.2 JIT-style honoring (layout as per-call INPUT — the clean-compiler demonstration)
An isolated JIT harness where layout is a per-call context: the runtime, per `mul_mat` call, inspects
the call's layout context (the §3.1B field as a runtime input), and EITHER dispatches the matching
pre-compiled kernel OR (SYS-b) repacks the slice just-in-time with C1's isolated packer (which can
double as the per-call repacker) then runs the x16 kernel.
- **What it PROVES:** the compiler is clean — per-call layout selection works, the compiler emitted
  a kernel + contract for each layout with no per-tensor limit baked in. The "layout as input"
  thesis is demonstrable in isolation.
- **What it does NOT prove:** it does **NOT** make decode JIT-repack amortize. SYS-b per-call repack
  pays off only for prefill (M≫1); for decode (M=1, bandwidth-bound) the per-call transform is pure
  added cost — *physics, not a compiler capability.* The JIT harness shows compiler-cleanliness;
  it does not repeal the memory wall.

---

## 5. THE E2E VALIDATION TABLE — EACH CELL NAMES ITS MECHANISM + PREDICTED COST

Grounded in **which layout each regime WANTS** (the mechanism falls out of the want + conflict):

| Cell | regime wants | conflict? | mechanism (named) | cost | predicted result |
|---|---|---|---|---|---|
| **q4_0 @ K1-VLEN256, decode** | plain (x16 kernel LOSES 0.74x here) | YES (prefill wants x16; **x16 is the K1 LOAD DEFAULT** — `case 256` auto-repacks) | **SYS-c pick-one / `plain`** — *actively SUPPRESS* `case 256` auto-repack so plain stride-18 survives | regression-removal is FREE (skipping `make_block_q4_0x16` costs no mem/compute); but the **layout choice is NOT free** — this pick-one forfeits K1 *prefill*'s x16 (→ the conflict row) | decode-decline **0.74x → TIE** (regression REMOVAL, not a new win) `[ssh k1]` |
| **q4_0 @ C920-VLEN128, decode** | x16 (x16 kernel WINS; **plain is the C920 LOAD DEFAULT** — `case 128: break`) | architecturally none (both regimes want x16) — but see cost note | **stored-x16: SYS-c pick-one-x16** (one stored x16 layout) | load-time pack paid ONCE; **caveat:** the currently-*coherent* C920 prefill path is scalar `_generic` which needs PLAIN (x16 GEMM numerically broken, §2) — so pick-one-x16 cleanly serves the DECODE win + the *broken* x16 prefill; "serves both" fully holds only after the GEMM-defect fix | decode-repack-keep **1.22 → 2.6x** — *tensor STORED x16 at load*, win = x16 kernel `[ssh rvv]` |
| **q4_0 prefill (M≫1)** | x16 | — | **the legit JIT/stored win**: either stored-x16 (free if already x16) OR **SYS-b JIT-repack that AMORTIZES over M≫1** | per-call transform / M → ~0 | prefill-keep — **MUST MEASURE amortization**; AND carry the §2 GEMM-defect flag (not measured-coherent at C920 yet) |
| **q4_0 @ K1-VLEN256, decode+prefill BOTH-WIN on ONE tensor, FREE** | decode wants plain, prefill wants x16 | **YES** | — | — | **DISALLOWED.** One tensor = one layout at an instant. Both-win-free is IMPOSSIBLE without SYS-a (dual-store 2x mem) or SYS-b (JIT-repack-cost, prefill-only). Name the cost or don't claim the cell. |

**The C920 vs K1 asymmetry, stated plainly (note the load defaults are OPPOSITE):** at
**C920-VLEN128** the load default is PLAIN (`case 128: break`); decode AND prefill architecturally
both want x16, so SYS-c pick-one-x16 (one stored layout, packed once at load) serves the decode win
— this is why the C920 decode keep is clean (caveat: the *currently coherent* prefill path is
scalar `_generic` over plain, so full both-serve waits on the GEMM-defect fix). At **K1-VLEN256**
the load default is X16 (`case 256` auto-repacks); decode wants plain but prefill wants x16 →
CONFLICT → the system must pick-one (suppress repack → decode ties but K1 prefill loses x16),
dual-store (2x mem), or JIT-repack-for-prefill (the legit M≫1 amortizing win). The disallowed cell
is the K1 conflict claimed as free.

**Report discipline (MEMORY kernel-wins-dont-transplant / winc-structural-null):** micro and e2e
are SEPARATE claims; never transplant. The decode keep is a STORED-x16 e2e claim, not a per-call one.

---

## 6. ORDERED BUILD STEPS FOR FULL C (user authorized FULL C — C1–C5; NO partial-C deferral)

| Step | What | Lit / Hardware | Effort | Honest risk |
|---|---|---|---|---|
| **C1** | **in-IR bridge + isolated bit-exact packer** (the partial-C, no hardware). (a) Replace B's deferred block-dot stub in the `Repack` branch with the real `create<GgmlRepackGemvQ40Q80Op>` (§3.3) carrying `weight_layout_contract="x16"`. (b) Emit a `make_block_q4_0x16`-equivalent EmitC PACKER op+verifier+emitter on the consumer-emitter template; prove BIT-EXACT vs ggml's `make_block_q4_0x16` using `…/emit-repack-gemv/verify_emitted_gemv.cpp:63` (inlined reference). This is the OLD doc's mechanism-(a), now a *capability proof* the compiler CAN materialize x16 (and the §4.2 per-call repacker), NOT the e2e path. | **LIT-ONLY** (lit + isolated microbench oracle, no hardware) | ~1 session | LOW. C1(a) trivial (a `create`); C1(b) is the consumer-emitter template applied to the simpler producer. Forced/clean rebuild (MEMORY build-incremental); decline cells stay `f810ce6b`. |
| **C2** | **contract-declaration mechanism.** Add `tcrv_rvv.weight_layout_contract` to (A) the op-attr allow-list (`isAllowedBlockDotAttr` + repack op) and (B) the route-metadata stream (`RVVEmitCRouteMetadata.cpp` push_backs). Lit-CHECK: `x16` on repack cells, `plain` on block-dot cells; emitted C byte-identical (the field is emitter-inert provenance, like B's audit triple). | **LIT-ONLY** | ~½ session | LOW. Must re-prove emitter-inertness via the fingerprint gate (B3 precedent). |
| **C3** | **the real producer.** NEW build-side tool: enumerate the model's q4_0 mul_mat tensors, author the abstract op in the real pipeline, drive `tcrv-opt`, collect `.inc` + route descriptor (with the contract field). | **HARDWARE/build-integration** (gguf + ggml build) | ~1–1.5 sessions | MED. Cost is model-enumeration + ggml-build integration, NOT MLIR. M-regime is per-CALL not per-tensor — the producer authors per-(tensor) and the contract per-(kernel); §3.2 conflict handling is C4's. |
| **C4** | **harness honoring (§4.1+§4.2).** Reads the contract; provides the layout. Includes the **1-line ggml `case 128 → return &q4_0_16x1_q8_0`** patch (KEEP@128) + the `case 256` suppress (DECLINE@K1); injects the named kernel; and the §4.2 JIT harness (per-call layout input). **This layer is where the limit lives — it picks {SYS-a/SYS-b/SYS-c} per (tensor, regime-conflict) and pays.** | **HARDWARE/multi-session** | ~1 session | MED-HIGH. The load-time boundary bites here: a conflicting tensor forces a pick-and-pay choice. |
| **C5** | **2-profile e2e, each number's mechanism named (§5).** `ssh k1`: K1 decode DECLINE (0.74x→tie, SYS-c/plain, free). `ssh rvv`: C920 decode KEEP (1.22→2.6x, stored-x16/SYS-c-x16). Prefill: MEASURE amortization (SYS-b or stored), carry the GEMM-defect flag. Report micro and e2e SEPARATELY. | **HARDWARE/multi-session** | ~1 session | MED. Prefill-keep gated on the §2 GEMM-emission fix; do NOT claim it measured-coherent. Do NOT claim the K1 conflict cell free. |

**The 2 ggml patches + their `#if` guards.** Both are build-harness (C4), NOT `lib/`:
1. **`get_optimal_repack_type` `case 128: break` → `return &q4_0_16x1_q8_0`** — guarded by the
   `#if defined __riscv_zvfh` block it already sits in (`repack.cpp:4589–4597`); requires the
   per-file `-D__riscv_zvfh` + vector march (`cmake_inject.py`). Enables STORED-x16 at VLEN128.
2. **The compute-kernel SWAP** into `ggml_gemv_q4_0_16x1_q8_0` (`if (vlenb*8==128)
   tcrv_emitc_…(); return;`, the existing `transform_repack.py:48` rewrite) — guarded by the same
   per-file march so the emitted RVV0.7/RVV1.0 kernel is what the build compiles for that file.
   (DECLINE@K1 needs NO patch beyond suppressing `case 256` so the native path runs.)

**Lit-vs-hardware split:** **C1–C2 are LIT-ONLY** (the clean compiler side — contract + bridge +
isolated packer, all in `tcrv-opt` + the isolated bit-exact oracle, no hardware). **C3–C5 are
HARDWARE / multi-session** (the system side — producer integration, the load coupling where the
limit lives, the 2-profile e2e).

---

## 7. CRITICAL FILES (file:line)
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:4339` `GgmlRepackGemvQ40Q80Op` — C1 bridge target
  (the x16 contract: 288/16/32, `half_lanes`, `integer_core_lmul`).
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:1718` repack-gemv verifier (the x16 pins the bridge
  `create` satisfies); `:910` `isAllowedBlockDotAttr` (C2 widens for `weight_layout_contract`).
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:2734+/:2799` — repack-GEMV emitter CONSUMES
  pre-packed x16, does NOT pack (proves the compiler is a consumer; materialization is §4's).
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteMetadata.cpp:108+` (`:208–215` the gearbox scope fields) — the
  route-metadata stream C2(B) appends `weight_layout_contract` to.
- `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp:78` `deriveRepackHalfLanes` — C1 wires.
- `llama.cpp/ggml/src/ggml-cpu/repack.cpp:3358/3939` `make_block_q4_0x16` (the load-time x16
  transform ggml owns); `:4727/4739` load-time trigger; `:4528/4592/4593` `get_optimal_repack_type`
  (the C4 `case 128`/`case 256` patches; `:4779` `supports_op` gate).
- `…/emit-repack-gemv/verify_emitted_gemv.cpp:63` — inlined reference `make_block_q4_0x16` = the
  C1(b) isolated bit-exact oracle.
- `…/fedora-rvv07/FINDING.md:212–228` — GEVM/decode bit-exact + coherent; **GEMM/prefill broken**
  (the prefill-keep flag). `llama-e2e-rvv07.log:97` — `ENGAGED@128` (proves the `case 128` patch
  existed; force-define alone insufficient).
- `…/fedora-rvv07/rvv07-perfile-build/{cmake_inject.py,transform_repack.py,arch_repack.cpp}` — the
  C4 build harness (per-file march + kernel swap); the C3 producer is the same artifact class.

---

## 8. THE DISALLOWED-CLAIM GUARD (one paragraph, load-bearing)

Do NOT claim, for ONE tensor copy, that decode and prefill BOTH win for FREE. At any instant the
tensor's bytes are ONE layout. When decode and prefill want DIFFERENT layouts (the K1-VLEN256 case:
decode wants plain, prefill wants x16), serving both requires SYS-a (dual-store, 2x memory for that
tensor) or SYS-b (JIT per-call repack — which amortizes ONLY for prefill M≫1, never for decode M=1
by the memory wall). When they want the SAME layout (the C920-VLEN128 case: both want x16), SYS-c
pick-one-x16 serves both — and THAT is why the C920 decode keep is clean. Every x16 decode win is a
STORED-x16 claim (cost paid once at load), never a per-call JIT-decode claim. The compiler is clean
(declares a contract, no per-tensor limit); the SYSTEM picks one of {dual-store / JIT / pick-one}
and pays — name the mechanism in every e2e cell or do not report the cell.
