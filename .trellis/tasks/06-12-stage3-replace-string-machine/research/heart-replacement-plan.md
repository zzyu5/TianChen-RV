# Stage 3 换心 master plan — replace the RVV route-plan string machine with real emitc DialectConversion

Author: Fable (senior MLIR architect investment). Date: 2026-06-12.
Task: `.trellis/tasks/06-12-stage3-replace-string-machine/`. Parent ADR: 06-12-mlir-audit-refactor.

This plan is the architecture decision for replacing the live ~82k-line RVV codegen
string machine with structured emitc IR, strangler-fig, one route family at a time,
each validated with real `ssh rvv` hardware evidence (I8).

---

## 1. Architecture decision — **HYBRID** (B-frame spine + A-discipline bodies)

**Decision: HYBRID.** Stand up a *real* MLIR `DialectConversion` (the thing the audit says
is missing — `RewritePatternSet` + `OpConversionPattern` + `TypeConverter` + `ConversionTarget`)
over the **already-existing** typed generic `tcrv_rvv` dataflow ops, and have each pattern's
`matchAndRewrite` construct emitc ops **directly on typed SSA Values** — no intermediate
string plan, no C-expression re-parser. Intrinsic calls stay as `emitc.call_opaque` with a
string *callee* (idiomatic emitc) but **typed Value operands**, where the callee comes from a
single pure SEW/LMUL/dtype name-mangling helper called *with operand types*, not stitched into
a serialized plan.

This is explicitly **hybrid**, not pure A and not pure B, for three reasons that are all true
at once:

- **B is the spine** (frame). The load-bearing thesis driver — "prefer REAL MLIR mechanisms"
  and the audit's核心 red flag "ZERO ConversionPattern" — is only retired by introducing the
  real conversion framework keyed on the typed ops. Pure A would leave the project at zero
  ConversionPatterns forever (a tidier hand `walk()`+`OpBuilder`), failing the thesis.
- **A is the discipline** (bodies). A's good idea — never build a string plan; construct
  emitc fine-grained — is *how* each pattern body works. The string round-trip
  (`TCRVEmitCLowerableRoute` string fields + the ~470-line re-parser) is deleted, not
  reimplemented inside a pattern.
- **The migration is literally hybrid in the tree.** Strangler-fig: a converted family runs
  through the new conversion pass while every other family keeps the legacy string machine
  unchanged. The codebase carries both paths, family by family, until the last family lands.

I am NOT introducing a brand-new typed "emitc-intrinsic op" ODS layer (a naive reading of pure
B). The dataflow source ops already exist (`tcrv_rvv.binary/compare/select/reduce/macc/load/
store/...`, each already tagged `TCRVEmitCLowerableOpInterface`). So the work is "**add the
lowering as patterns, reuse the op layer**," and the bodies emit emitc directly. That dual
nature is exactly why the honest label is hybrid.

### What gets BUILT
- One `TypeConverter` mapping the `!tcrv_rvv.*` types to emitc C types:
  `!tcrv_rvv.vl` → `emitc.opaque<"size_t">`; `!tcrv_rvv.<dtype><lmul>` (e.g. `i32m1`) →
  `emitc.opaque<"vint32m1_t">`; `!tcrv_rvv.runtime_abi_value` → the pointer/scalar C type from
  the op's `c_type` attribute. This is the piece that replaces `getEmitCTypeForCType`'s string
  matching — derived from typed facts, not re-parsed text.
- One `ConversionTarget` + `RewritePatternSet`, and a family `populate*Patterns(TypeConverter&,
  RewritePatternSet&)` entry (mirrors upstream `populateArithToEmitCPatterns`), so a second
  family (IME, N2) registers its own patterns without the core branching on family.
- `OpConversionPattern`s for the beachhead ops: `setvl`, `with_vl` (→ `emitc.for` + induction +
  remaining-AVL `emitc.sub`), generic `load`, generic `binary`, generic `store`. Pointer/index
  arithmetic (`ptr + i`, `n - i`) is built as `emitc.add`/`emitc.sub` on typed Values inside the
  pattern.
- A pure name-mangler `riscvIntrinsicName(mnemonic, sew, lmul, dtype)` (the ONE legitimate
  survivor of the Twine logic) consumed inside patterns from operand *types*.
- A new pass `--tcrv-rvv-lower-to-emitc` (or a guarded mode of the existing
  `--tcrv-materialize-emitc-lowerable-routes`) that runs the conversion for converted families
  and *delegates to the legacy string path for everything else*.

### What gets DELETED (per converted family, once green on hardware)
- That family's `*PlanOwners.cpp` string-plan construction (e.g.
  `RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`, 736 lines).
- The family's `Twine("__riscv_...")+sew+lmul` callee assembly and raw `StringRef "+"/"-"`
  operand-expression concatenation in `RVVEmitCRoutePlanning.cpp`.
- Once NO family emits string arithmetic: the entire operand re-parser in
  `TCRVEmitCLowerableMaterializer.cpp` (`materializeOperandExpression`, `parseSimpleBinary…`,
  `parseLeftAssociativeBinaryChain`, `parseSimpleProduct…`, `parseFloorMultiple…`,
  `parseScaledPointer…`, `parseCStyleCastScaledPointer…`, `parseSimpleSubscript…`,
  `getEmitCTypeForCType`) — ~470 fragile lines.
- End-state: the `std::string` executable-fact fields of `TCRVEmitCLowerableRoute`
  (`callee/expression/cType/declarationInitializer/inductionVarName`) and the bulk of
  `RVVEmitCRoutePlanning.cpp` (40,919 lines) retire.

---

## 2. Rationale against the thesis

- **Fixes the 0-ConversionPattern red flag (real MLIR mechanism).** Confirmed empirically:
  `grep RewritePatternSet|ConversionPattern|OpRewritePattern|applyPartialConversion lib/ include/`
  = 0 hits; transforms are hand `walk()`+`OpBuilder`. The hybrid introduces the project's first
  genuine `OpConversionPattern`/`TypeConverter`/`ConversionTarget`. Pure A does not do this and
  therefore does not satisfy the thesis; that is the decisive reason A is rejected.
- **Kills the string explosion.** The combinatorial fan-out today is intrinsic-name strings
  built by `Twine("__riscv_…")+sew+lmul` across the planner and operand C-expressions built by
  `StringRef "+"/"-"` concatenation, then re-parsed back into the very structure the planner
  threw away. The hybrid collapses fan-out into one pure name-mangling function over *types* and
  builds arithmetic as `emitc.add/sub/mul` on Values — the round-trip and its re-parser are
  deleted, not relocated.
- **Strangler-fig-able one family at a time.** The conversion pass is keyed per-op; a converted
  family legalizes through patterns while unconverted families keep the legacy string path. The
  artifact/export entry points (`buildSelectedEmitCArtifactRoute` →
  `materializeTCRVEmitCLowerableRoute` → `translateToCpp`) are unchanged for unconverted
  families, so the chain always produces a working artifact. No big-bang.
- **Keeps the `emitc.call_opaque` idiom but typed operands.** `emitc.call_opaque` legitimately
  takes a *string callee* (an opaque intrinsic identifier is genuinely a name) with *typed Value
  operands* — that is exactly idiomatic emitc and exactly what the materializer already does for
  callees today. The change is to stop serializing operands as strings: feed the pattern's own
  typed Values directly. So the keep/cut line is principled: callee string = legitimate name;
  operand strings = lossy round-trip to delete.
- **N1 (capability IR).** Lowering now stands on real typed dataflow + real types, not string
  attributes — capability/config authority flows as IR facts into a `TypeConverter`, the
  prerequisite for capability-driven codegen being "real IR."
- **N2 (zero-core-branch plugin generality).** A shared family-agnostic `TypeConverter` +
  `ConversionTarget` legalization frame with per-family `populate*Patterns` is precisely how a
  second family (IME) contributes lowering without the core knowing family specifics — mirrors
  upstream `populateArithToEmitCPatterns`/`SCFToEmitC`. Under pure A each family would ship its
  own bespoke string planner + re-parser: the opposite of generality.
- **N3 + I8 (tune + real hardware).** Each family is proven on real `ssh rvv` (compile + run +
  numeric correctness) before its string plan is deleted, so the换心 never trades honesty for
  tidiness; Gearbox tune decisions then sit on real IR.

---

## 3. Beachhead family — plain unit-stride elementwise `add` (i32, m1)

The single simplest, most self-contained, best-backstopped family.

- **Smallest plan / smallest owner.** `RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp` is
  736 lines, the smallest `*PlanOwners.cpp` (vs `RVVEmitCRoutePlanning.cpp` 40,919 and
  `RVVEmitCContractionRouteFamilyPlanOwners.cpp` 11,745). Plain `add` emits exactly 6 statements:
  pre-loop full-chunk `setvl`, then loop body = remaining-AVL `setvl`, `vle32` lhs, `vle32` rhs,
  `vadd`, `vse32`.
- **Only 4 intrinsics, all simple Twine concat:** `__riscv_vsetvl_e32m1`,
  `__riscv_vle32_v_i32m1`, `__riscv_vadd_vv_i32m1`, `__riscv_vse32_v_i32m1`.
- **Lowest re-parser surface.** Operands are only `ptr + i` and `n - i` (the single simplest
  re-parser shape). No scaled-pointer, floor-multiple, or subscript machinery — so converting
  this family lets the hardest parser branches stay untouched until later families.
- **Best test backstop.** Typed-body input fixture `test/Conversion/EmitC/
  rvv-first-slice-materialization.mlir`; golden emitted-C handoff
  `test/Target/RVV/emitc-to-cpp-handoff.mlir:74-83`; artifact fixtures
  `test/Target/RVV/explicit-selected-body-artifact-{add,sub,mul}.mlir`; and the ssh-rvv e2e
  driver (`add/sub/mul` are `DEFAULT_OP_KINDS`) with dry-run fixture
  `test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`.
- **Trivial follow-on.** `sub`/`mul` differ only by the `vadd`→`vsub`/`vmul` mnemonic
  (`RVVEmitCRoutePlanning.cpp:3557-3573`) — same patterns, different `kind` attr.

Explicitly DEFER (more complex rungs): scalar-broadcast / masked elementwise; strided-add
(+byte-stride scaled-pointer math); base/computed-mask memory; reduction; MAcc; segment2;
contraction/widening-dot (the 11,745-line owner).

---

## 4. Per-family conversion recipe (the repeatable strangler-fig loop)

Apply this ordered loop to ONE family at a time (beachhead = `add`):

1. **Pin the baseline.** Capture the family's current golden emitted-C and the artifact bundle
   from the legacy string path (lit + a dry-run e2e). These are the equivalence oracle for the
   swap. Confirm `build/bin/tcrv-opt`/`tcrv-translate` build green and lit baseline is recorded.
2. **TypeConverter (once, reused after).** Add/extend the `!tcrv_rvv.*` → emitc C-type
   conversions this family needs (`vl`→`size_t`, `i32m1`→`vint32m1_t`, `runtime_abi_value`→its
   `c_type`). For the beachhead this is the full set; later families only extend (mask, tuple,
   widened types).
3. **Patterns.** Write `OpConversionPattern`s for the family's ops, bodies constructing emitc
   directly on typed adaptor Values: `setvl`→`emitc.call_opaque "__riscv_vsetvl_e…"`;
   `with_vl`→`emitc.for` with induction var + remaining AVL via `emitc.sub`; `load`/`store`→
   `emitc.call_opaque "__riscv_vle…/vse…"` with pointer `emitc.add(base, i)`; `binary`→
   `emitc.call_opaque "__riscv_vadd/vsub/vmul…"`. Callee string comes from the pure
   `riscvIntrinsicName(...)` mangler over types/`kind`. No string plan; no operand expressions.
4. **Pass wiring behind a flag.** Run these patterns in a new
   `--tcrv-rvv-lower-to-emitc` pass (or a guarded mode), gated so ONLY this family's ops are
   legalized by patterns; all other ops fall through to the legacy string path unchanged. Build
   stays green; other families unaffected.
5. **Structural lit test (assert IR, not strings).** Add a `test/Conversion/EmitC/` fixture that
   `FileCheck`s the *structured* emitc (`emitc.for`, `emitc.call_opaque` with `__riscv_vadd…`,
   `emitc.add`/`emitc.sub` for addressing) — pin structure, not re-parsed text. Also assert the
   exported C still matches the golden handoff (`emitc-to-cpp-handoff.mlir`) — byte-equivalence
   to the legacy path is the swap's contract.
6. **ssh-rvv hardware lamp (I8).** Run the e2e bundle-ABI harness for this family WITHOUT
   `--dry-run` (Section 5). Require remote `uname -m = riscv64`, clang compile/link success, and
   numeric `PASS op=<kind>` + `tcrv_rvv_generated_bundle_abi_<kind>_ok`. Record commands +
   `evidence.json` (`status=success`, `ssh_evidence=true`) + remote stdout.
7. **Flip default + delete the string plan.** Make the converted path the default for this
   family; migrate any remaining consumers/fixtures off the legacy strings (dead-mirror-removal
   guide: move consumers first, then delete); delete the family's `*PlanOwners.cpp` plan
   construction and its `Twine`/`StringRef` assembly in `RVVEmitCRoutePlanning.cpp`. When NO
   family still emits string arithmetic, delete the corresponding `materializeOperandExpression`
   parser branches.
8. **Re-green.** build green + lit honest-green (≤3 environmental reds), then move to the next
   family (`sub`/`mul`, then up the complexity ladder).

---

## 5. E2E ssh-rvv validation recipe (the beachhead hardware lamp)

Prereqs (all confirmed present): `build/bin/tcrv-opt`, `build/bin/tcrv-translate`; an
RVV-capable local clang on PATH (`/usr/lib/llvm-20/bin/clang` qualifies — verified it compiles
`-target riscv64 -march=rv64gcv -mabi=lp64d -c` to an object); reachable `ssh rvv` (ProxyJump,
`DEFAULT_SSH_TARGET="rvv"`). The local bundle export shells out to clang
(`RVVTargetSupportBundle.cpp:1690-1707`), so a local RVV clang is needed even for the smoke step.

**Step 1 — smoke (no hardware, proves local generation):**
```
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --dry-run --artifact-root /tmp/bh.artifacts --run-id add-bh --overwrite \
  --op-kind add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```
Expect stdout `dry_run_success` and a bundle (`.o`/`.h`/`.c`) under the artifact dir. Internally:
`tcrv-opt --tcrv-materialize-emission-plans` → `tcrv-translate --tcrv-rvv-emitc-to-cpp` →
`tcrv-translate --tcrv-export-target-artifact-bundle` (local clang → riscv64 `.o`).

**Step 2 — the real lamp (drop `--dry-run`, add `--ssh-target rvv`):**
```
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/bh.artifacts --run-id add-bh-hw --overwrite --ssh-target rvv \
  --op-kind add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```
This scp's object+header+harness to `ssh rvv`, runs
`clang -O2 -march=rv64gcv -mabi=lp64d -I. <harness.c> <object.o> -o <bin>`, executes the binary,
and asserts `tcrv_rvv_generated_bundle_abi_add_ok` + `PASS op=add`. Success →
`evidence.json` with `status=success`, `ssh_evidence=true`, plus `remote_compile_stdout.txt`
(showing `uname -m=riscv64` + clang version) and `remote_run_stdout.txt`. That triple
(compile + run + correct numeric result on real RISC-V) IS the I8 lamp.

Critically: the conversion must keep feeding the SAME `--tcrv-rvv-emitc-to-cpp` /
`--tcrv-export-target-artifact-bundle` exports, so this existing harness validates the rewritten
path with ZERO harness changes. Use ONE family (add), not the full 138-fixture sweep, for the
first lamp.

---

## 6. First concrete step (smallest buildable edit, build+lit no-worse-than-baseline)

Land the **TypeConverter + ConversionTarget + a no-op-registered pass shell behind a new flag**,
parallel to the string path, converting NOTHING yet:

1. Add `lib/Conversion/RVV/RVVToEmitC.cpp` (+ header) defining
   `populateRVVToEmitCTypeConversions(TypeConverter&)` for `!tcrv_rvv.vl`/`i32m1`/
   `runtime_abi_value`, and `populateRVVElementwiseToEmitCPatterns(TypeConverter&,
   RewritePatternSet&)` (initially empty).
2. Add pass `--tcrv-rvv-lower-to-emitc` (Passes.td + a `RVVToEmitCPass`) that builds the
   `TypeConverter`, marks emitc legal + `tcrv_rvv` ops legal-for-now (so it is a structural
   no-op), and runs `applyPartialConversion`. Register it; CMake the new lib.
3. A single positive lit smoke test that runs the new pass on the beachhead body and checks it
   round-trips unchanged (proves the conversion harness is wired and green).

This compiles, registers the project's FIRST `ConversionPattern`/`TypeConverter`/
`ConversionTarget` (red flag begins to fall), and leaves the live string path and all lit
baselines untouched. The very next edit adds the first real pattern (`tcrv_rvv.binary` add →
`emitc.call_opaque "__riscv_vadd_vv_i32m1"`) and grows from there.

---

## 7. Risks

- **Typed C-type derivation.** Generic ops use `AnyType` operands (`BinaryOp` lhs/rhs/vl/result
  are `AnyType`); concrete C types (`vint32m1_t`, `int32_t*`) must come from `!tcrv_rvv` type
  encodings + the op's `c_type`/config attrs. The `TypeConverter` must reproduce exactly what
  the planner derived (`RoutePlanning.cpp:3177+`). Mitigation: reuse that derivation logic;
  pin byte-equivalence against the golden C before deleting anything.
- **Byte-identical C regression.** `translateToCpp` formatting (spacing, temp-var naming, loop
  shape) must match the golden handoff or downstream lit/e2e fixtures break. Mitigation: diff
  exported C against `emitc-to-cpp-handoff.mlir` early; treat any delta as a conversion bug, not
  a fixture rewrite, until intentionally re-baselined.
- **Two materialization entry points.** There is `--tcrv-materialize-emitc-lowerable-routes`
  (first-slice/generic-stage2 tests) AND `--tcrv-materialize-emission-plans` (the e2e bundle
  path). The conversion must slot so BOTH the lit IR path and the bundle/export path go through
  it for converted families. Mitigation: wire the new pass into
  `ExecutionPlanningPipeline`/`TargetArtifactExport` for converted op-kinds only.
- **Legacy vs generic op duality.** The body still carries legacy specialized ops
  (`tcrv_rvv.i32_load`/`i32_add`, fail-closed) alongside generic ops (`tcrv_rvv.load`/`binary`).
  Convert against the GENERIC ops (the live materialization source) — confirm the selected body
  is generic before patterns run.
- **Partial-conversion legality traps.** Mixed converted/unconverted families in one module can
  trip `ConversionTarget` legality if `with_vl` scope ops aren't handled coherently. Mitigation:
  gate the target by op-kind set; keep `with_vl`/`setvl` conversion atomic with the family.
- **Hardware access flakiness.** `ssh rvv` via ProxyJump can be down; an I8 claim must not be
  asserted from a dry-run. Mitigation: gate "delete string plan" on a real `ssh_evidence=true`
  evidence.json, never on dry-run.
- **Scope creep.** The planner is 40,919 lines; the temptation is to convert too much at once.
  Mitigation: strangler-fig discipline — one family, one lamp, one deletion, re-green, repeat.

## 8. Open questions

- Does the `TypeConverter` need per-op config context (policy/SEW/LMUL beyond the type encoding)
  to pick the C type, or do the `!tcrv_rvv.<dtype><lmul>` types fully determine it? (Determines
  whether the converter is pure-type or needs an analysis side-channel.)
- Should `--tcrv-rvv-lower-to-emitc` be a new pass or a guarded mode of the existing
  `--tcrv-materialize-emitc-lowerable-routes`? (Affects how the e2e bundle path opts converted
  families in.)
- Will byte-identical C be achievable from `translateToCpp`, or should some fixtures be
  intentionally re-baselined (and how to keep that honest vs hiding a regression)?
- Where exactly does the `for`-loop / remaining-AVL / pre-loop-chunk structure get owned —
  inside the `with_vl` pattern, or a dedicated control pattern reused across families? (Affects
  reuse for reduction/macc later.)
- For N2 generality: is `populate*Patterns` the right registration seam now, or wait for an
  upstream `ConvertToEmitCPatternInterface`-style umbrella (llvm-21) before committing the
  plugin-facing API?

---

## 9. Evidence table (verified file:line)

| Claim | Evidence (file:line) |
|---|---|
| ZERO ConversionPattern/RewritePatternSet in lib/+include/ | `grep RewritePatternSet\|ConversionPattern\|OpRewritePattern\|applyPartialConversion lib/ include/` = 0 hits (verified) |
| String executable-fact model (callee/expression/cType all std::string) | `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h:42-57` |
| String round-trip bounded by route struct (more string fields) | `…/TCRVEmitCLowerableInterface.h:59-159` (declarationInitializer:63, inductionVarName:74) |
| Intrinsic callees by Twine concat (vsetvl/vle/vse/vadd…) | `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp:3225,3233,3301,3557-3573` |
| Operand expressions by raw StringRef "+"/"-" concat (`ptr + i`, `n - i`) | `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp:546-562`; `lib/Dialect/RVV/IR/RVVConfigContract.cpp:1945-1948` |
| Re-parser rebuilds emitc from strings (parse* fns) | `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp:256-456` |
| materializeOperandExpression core string→SSA bridge | `…/TCRVEmitCLowerableMaterializer.cpp:627-1011` |
| cType strings re-parsed into emitc types | `…/TCRVEmitCLowerableMaterializer.cpp:124-133` |
| Typed generic source ops already exist, tagged lowerable interface | `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3215 Binary, :3276 Compare, :3333 Select, :3360 Reduce, :3474 MAcc, :2643 Load, :3838 Store, :3744 Dequantize` |
| BinaryOp args AnyType lhs/rhs/vl + StrAttr kind, AnyType result | `RVVOps.td:3231-3242` |
| Op interface today is provenance-only (no lower method) | `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td:20-39` |
| llvm-20 emitc conversion infra available + TypeConverter signature | `/usr/lib/llvm-20/include/mlir/Conversion/{ArithToEmitC,FuncToEmitC,SCFToEmitC,MemRefToEmitC,MathToEmitC}`; `ArithToEmitC.h:16 populateArithToEmitCPatterns(TypeConverter&,…)` |
| Live route path build→materialize→translateToCpp | `lib/Target/TargetArtifactExport.cpp:1210,2035-2056,2086-2092` |
| Materialize pass + flag exists (conversion slot) | `include/TianChenRV/Transforms/Passes.td:304-323`; `lib/Transforms/EmitCLowerableMaterialization.cpp:255-307` |
| Beachhead typed-body input fixture (generic ops) | `test/Conversion/EmitC/rvv-first-slice-materialization.mlir:3-36` |
| Beachhead golden emitted-C (4 intrinsics) | `test/Target/RVV/emitc-to-cpp-handoff.mlir:74-83` |
| Smallest PlanOwners = ElementwiseArithmetic (736); planner 40,919; total 87,975 | `wc -l lib/Plugin/RVV/EmitC/*.cpp` (verified) |
| ssh-rvv remote evidence (scp/compile/run/assert) | `scripts/rvv_generated_bundle_abi_e2e.py:30399-30473` |
| Local bundle generation runs unconditionally (dry-run + real) | `scripts/rvv_generated_bundle_abi_e2e.py:30242-30312` |
| Local bundle export shells to clang -target riscv64 | `lib/Target/RVV/RVVTargetSupportBundle.cpp:1690-1707` |
| lit feature gate tianchenrv-local-rvv-object-clang | `test/lit.cfg.py:22-61,148-149` |
| --ssh-target default "rvv"; dry-run vs real branch | `scripts/rvv_generated_bundle_abi_e2e.py:44 (DEFAULT_SSH_TARGET), 37326-37343` |
