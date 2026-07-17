# IME mma_us (vmadotus) — silicon bit-exact gate on real K1 (Spacemit X60) — FINDING

**Date:** 2026-06-25
**Hardware under test:** `ssh k1` = Spacemit(R) X60, harts 0-3 (IME1), via `taskset -c 0-3`.
**Invariant in force:** I8 — only a real `ssh` run on the X60 counts as runtime evidence. **No QEMU.**
**Op under test:** the 6th IME op `tcrv.ime.mma_us` (vmadotus), int8 UNSIGNED A × int8 SIGNED B → int32.
This COMPLETES the IME signedness family {ss=vmadot, uu=vmadotu, su=vmadotsu, **us=vmadotus**}.

---

## VERDICT (TL;DR)

> **PASS — operand convention CONFIRMED, 16/16 bit-exact.**
> On real X60 silicon the compiler-emitted `vmadotus` kernel produced a result
> **bit-exactly equal to the reversed-mixed reference (A uint8 UNSIGNED × B int8 SIGNED, int32 accumulate)**
> and **different from all three other interpretations** (the su flip, pure-signed, pure-unsigned).
> ⇒ the assumed **unsigned-A(v0) × signed-B(v1)** operand binding in `macHelperBody` is **correct on silicon** —
> NO operand swap needed. No SIGILL (exit 0, both markers); VLEN=256 / 4×4×8 MAC pinned (vlenb=vl=32).
> The IME us result is bit-exactly the `reverse` matrix the committed mma_su oracle already recorded
> (first row `361 -8926 9122 -3020`), an independent cross-check against that prior finding.

This is the **N2 STRUCTURAL rapid-add** (0-core breadth) that closes the signedness square. It is breadth,
not a new algorithm/tune: **Win-A/Win-B/Win-C = N/A** for this leaf variant, and **NO perf claim** is made
(it is a MAC-signedness variant; the IME perf story is the kernel-micro re-baseline + e2e-unit-null already
recorded in IME-PERF-REBASELINE-FINDING.md / [[kernel-wins-dont-transplant-to-e2e]]).

---

## 0. The change — 0 CORE, plugin-only (the N2 abstraction in action)

Cloned the COMMITTED `tcrv.ime.mma_su` (vmadotsu, `70bb845a`) → `tcrv.ime.mma_us` (vmadotus).
The variant signedness FACT flows as DATA (`ime_signedness = "unsigned_signed"` → `vmadotus`); NO new
capability id, NO family string-match in core.

```
$ git diff --numstat -- lib/Transforms lib/Conversion/EmitC/Builtin
(empty)                  # 0 CORE lines — the 0-core gate

$ git diff --numstat
56  0   include/TianChenRV/Dialect/IME/IR/IMEOps.td          # ODS MMAUSOp (clone MMASUOp)
19  0   lib/Dialect/IME/IR/IMEDialect.cpp                    # MMAUSOp::verify + kExpectedMixedSignUSIMEOp("vmadotus")
27  3   lib/Plugin/IME/IMEBackendEmissionDriver.cpp          # vmadotusHelperBody + IMEMACToEmitCFunc<MMAUSOp> + illegal/pattern rows
48 18   lib/Plugin/IME/IMEExtensionPlugin.cpp                # unsigned_signed derive arm + routing + matmul-shape fail-closed
```

**Plugin LOC:** `lib/Plugin/IME` = +75 / −21. **Dialect surface** (ODS op + verifier) = +75 / −0.
Total ≈ 150 LOC added, ALL in the IME dialect/plugin (the conversion-target `addIllegalOp` + `patterns.add`
rows live in `lib/Plugin/IME/IMEBackendEmissionDriver.cpp`, NOT in `lib/Conversion`/`lib/Transforms`).
**Zero core branches.** This is the same rapid-add cost shape as mma_u / mma_su — the cheapness IS the N2 thesis.

**One advisor-caught correctness item (invisible to the requested lit set):** `deriveIMEMatmulCapability`'s
`ime_matmul_shape` guard fail-closes mixed-sign via `signedness == kSignednessMixedSign`; that constant is
ONLY `"signed_unsigned"`, so the new `"unsigned_signed"` would have fallen THROUGH into the tiled vmadot/vmadotu
path (dying later with a confusing "ime_op must be 'vmadot'"). Fixed: the guard now covers BOTH mixed forms
(`== kSignednessMixedSign || == kSignednessMixedSignUS`). The slide guard (`!= kSignednessSigned`) already
rejects `unsigned_signed` correctly — left untouched.

---

## 1. objdump encoding — `smt.vmadotus = e210112b` ENGAGED

The compiler-emitted static binary, cross-built with SpacemiT GCC 15.2 fork
(`-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`), disassembles to a REAL opcode:

```
$ riscv64-unknown-linux-gnu-objdump -d ime_mma_us_gate | grep vmadot
   10594:	e210112b   smt.vmadotus	v2,v0,v1
```

**Encoding = `e210112b`** (objdump word column), mnemonic `smt.vmadotus v2,v0,v1` — exactly the value the
IME-KERNEL-EXPANSION-DESIGN.md §0 table names as "the one missing plain signedness sibling". The full
signedness square is now objdump-confirmed on this toolchain:

| mnemonic   | objdump encoding | selector nibble [13:12] | bit13=A-signed, bit12=B-signed | signedness            |
|------------|------------------|-------------------------|--------------------------------|-----------------------|
| vmadot     | `e210312b`       | `3` = `11`              | A signed, B signed             | signed × signed       |
| vmadotu    | `e210012b`       | `0` = `00`              | A unsigned, B unsigned         | unsigned × unsigned   |
| vmadotsu   | `e210212b`       | `2` = `10`              | A signed, B unsigned           | signed × unsigned     |
| **vmadotus** | **`e210112b`** | **`1` = `01`**          | **A unsigned, B signed**       | **unsigned × signed (this op)** |

**Encoding-level corroboration (independent of the silicon run):** for vmadotus the selector nibble is
`1` = binary `01` ⇒ bit13=0 (A unsigned) / bit12=1 (B signed). So the encoding ITSELF says unsigned-A ×
signed-B — which the X60 run then confirmed numerically. vmadotsu (`...212b`) and vmadotus (`...112b`) are
genuinely distinct opcodes, so the silicon could in principle have implemented either binding — that is why
the 4-way discriminator below is load-bearing.

---

## 2. The 4-way discriminator (the load-bearing test)

The single-fragment 4×4×8 kernel was driven on the COMPILER-EMITTED helper `tcrv_ime_vmadotus_mma_4x4x8(...)`
— extracted verbatim from the materialization pipeline
(`tcrv-opt test/Conversion/EmitC/ime-mma-us-materialization.mlir --tcrv-materialize-plugin-variants
--tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes`
→ `mlir-translate --mlir-to-cpp`), NOT a hand copy — so this is an N2 claim about OUR emitter, not an isolated
asm probe. Helper binds A→v0, B→v1, C→v2/v3 and executes `vmadotus v2, v0, v1` (the FOUNDATION load/store
structure, mnemonic the only divergence vs the su sibling).

**HIGH-BIT discriminating data** = the EXACT A/B byte payload from the committed mma_su oracle (asymmetric
high-bit placement between A and B, already proven to make all four interpretations pairwise distinct). Four
scalar int32 references over the SAME payload; the gate ABORTS (exit 4) unless all four are pairwise distinct
— they are. On X60 (`taskset -c 0-3`):

```
MATCH: us=1 su=0 signed=0 unsigned=0
```

| reference (over identical bytes)               | IME == ref? |
|------------------------------------------------|-------------|
| (1) us / reverse-mixed: A UNSIGNED × B SIGNED  | **YES** ✓   |
| (2) su / mixed-sign:    A SIGNED   × B UNSIGNED | no          |
| (3) pure-signed:        both SIGNED             | no          |
| (4) pure-unsigned:      both UNSIGNED           | no          |

The IME result matrix equals (1) exactly in all 16 int32 elements, first row `361 -8926 9122 -3020`
(us-ref) vs su `1129 18210 2210 -15308`, signed `105 -14302 1954 17460`, unsigned `1385 23586 9378 29748`
— all four refs visibly different, IME picks (1).

**Verdict mapping:** `IME == us(1) AND ≠ su(2) AND ≠ signed(3) AND ≠ unsigned(4)`
⇒ **PASS, unsigned-A(v0) × signed-B(v1) CONFIRMED. No operand swap. Bit-exact 16/16.**

(Had it matched su(2), the verdict would have been "SU-FLIP — A treated signed, B unsigned — wrong op/binding".
It did not. The us result is bit-exactly the `reverse` matrix the committed su finding already recorded,
cross-validating both findings.)

---

## 3. Hardening — the gate is not vacuous

- **SIGILL mechanically detectable:** `MARK_BEFORE_VMADOTUS` / `MARK_AFTER_VMADOTUS` both printed, exit 0
  (not 132). The `vmadotus` executed natively.
- **Negative control (proves it's the real IME, not a fallback):** `taskset -c 4 ./ime_mma_us_gate` →
  **exit 132 (SIGILL)**. Hart 4 lacks the `_ime` token (per FOUNDATION.md), so the instruction is illegal
  there — confirming harts 0-3 genuinely *execute* the IME opcode rather than emulating it.
- **MAC-unit pin:** vlenb=32, vl(e8,m1)=32 asserted (VLEN=256, 4×4×8 int32-accum), else abort (exit 3).
- **Discriminator pin:** all four refs pairwise distinct, asserted at runtime (else abort exit 4).
- **Determinism:** 3/3 independent runs on harts 0-3 → exit 0 (PASS), `us=1 su=0 signed=0 unsigned=0`, identical.

---

## 4. lit — FileCheck PASS (75/75 in Dialect/IME + Conversion/EmitC, no regression)

- `test/Dialect/IME/mma-us.mlir` — positive round-trip (`tcrv_ime.mma_us`, `ime_op="vmadotus"`, 4×4×8) +
  a ROTATED negative (`ime_op="vmadotsu"` rejected: "ime_op must be 'vmadotus'…", the verifier message is
  parameterized by the expected mnemonic) + 3 more fail-closed negatives (pure-signed mnemonic, wrong elem
  width, unknown attr).
- `test/Conversion/EmitC/ime-mma-us-materialization.mlir` — the full `unsigned_signed` materialization:
  derive FACT → propose `ime_vmadotus_mma_slice` → materialize real `tcrv_ime.mma_us` → lower to the
  `vmadotus` asm leaf through the common EmitC route. The `--implicit-check-not` set asserts NO family leak,
  and specifically that **none of the OTHER three signedness siblings leaks** (vmadot, vmadotu, AND the
  vmadotsu sibling `tcrv_ime_vmadotsu_mma_4x4x8` / `ime_vmadotsu_mma_slice`).

---

## 5. What this seals

- The 6th IME op `tcrv.ime.mma_us` now has the **same bit-exact-16/16-on-X60 seal** as mma (vmadot,
  FOUNDATION.md) and mma_su (vmadotsu). The signedness family is COMPLETE end to end:
  emit (0-core-branch, capability-fact-driven) → cross-build → objdump-real-opcode (`e210112b`) → silicon
  bit-exact, plus SIGILL negative control + 3/3 determinism.
- This is ANOTHER live 0-core rapid-add — the N2 abstraction demonstrated again: a new signedness sibling
  slots in at ~150 plugin/dialect LOC with ZERO core edits and ZERO core family-branches.
- **NO perf claim** (breadth, not algorithm/tune). Win-A/B/C = N/A for this leaf variant.

## Reproduce
Local scratch (NOT in git): `/home/kingdom/spacemit-ime/mma-us/` holds `ime_mma_us_gate.c`
(`#include "emitted_helper.c"` — the pipeline-emitted leaf), `emitted_helper.c`, `emitted_full.c`
(the raw `mlir-translate --mlir-to-cpp` output), `pipeline.out`, and the static binary.
Toolchain root: `/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4`. On K1 the
binary is at `~/n2-ime-probe/ime_mma_us_gate`.
