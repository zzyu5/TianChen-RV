# IME mma_su (vmadotsu) — silicon bit-exact gate on real K1 (Spacemit X60) — FINDING

**Date:** 2026-06-24
**Hardware under test:** `ssh k1` = Spacemit(R) X60, harts 0-3 (IME1), via `taskset -c 0-3`.
**Invariant in force:** I8 — only a real `ssh` run on the X60 counts as runtime evidence. **No QEMU.**
**Op under test:** the 4th IME op `tcrv.ime.mma_su` (vmadotsu), int8 SIGNED A × uint8 UNSIGNED B → int32, committed on main `70bb845a`.

---

## VERDICT (TL;DR)

> **PASS — operand convention CONFIRMED, 16/16 bit-exact.**
> On real X60 silicon the compiler-emitted `vmadotsu` kernel produced a result
> **bit-exactly equal to the mixed-sign reference (A int8 SIGNED × B uint8 UNSIGNED, int32 accumulate)**
> and **different from all three other interpretations** (reverse-mixed, pure-signed, pure-unsigned).
> ⇒ the assumed **signed-A(v0) × unsigned-B(v1)** operand binding in `macHelperBody` is **correct on silicon** —
> NO operand swap needed. No SIGILL (exit 0, both markers); VLEN=256 / 4×4×8 MAC pinned (vlenb=vl=32).
> This completes the live N2 rapid-add worked example: the 4th op now has the same "bit-exact 16/16 on X60"
> seal as the 1st op (vmadot, FOUNDATION.md task 3), AND resolves the one honest caveat (signedness asymmetry).

---

## 1. objdump encoding — CORRECTED (the commit's spec-assertion was wrong)

The compiler-emitted static binary, cross-built with SpacemiT GCC 15.2 fork
(`-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`), disassembles to a REAL opcode:

```
$ riscv64-unknown-linux-gnu-objdump -d ime_mma_su_gate | grep vmadot
   10594:	e210212b   smt.vmadotsu	v2,v0,v1
```

**Encoding = `e210212b`** (objdump word column), mnemonic `smt.vmadotsu v2,v0,v1`. 32-bit instruction.

> **CORRECTION:** commit `70bb845a` (and IMEBackendEmissionDriver.cpp:44, N2-CLARITY-DESIGN.md) spec-asserted
> `0xe210232b`. **The real assembled encoding is `e210212b`, not `0xe210232b`.** The driver's source comment
> and the design doc should be corrected. (This is a doc/comment fix only — it does NOT affect the emitted
> kernel, which is generated from the MNEMONIC string `"vmadotsu"`, not a hardcoded encoding.)

The full sibling family, all confirmed from objdump on this toolchain. The four share the word
`e210_12b`; only the 5th hex nibble (bits [13:12], the signedness selector) varies. (Encodings are the
objdump-printed 32-bit word value, i.e. the encoding-table form — note FOUNDATION.md task 2 records vmadot
as `0x2b3110e2`, the byte-reversed "little-endian word" reading of the SAME instruction; both forms agree.)

| mnemonic   | objdump encoding | selector nibble [13:12] | bit13=A-signed, bit12=B-signed | signedness            |
|------------|------------------|-------------------------|--------------------------------|-----------------------|
| vmadot     | `e210312b`       | `3` = `11`              | A signed, B signed             | signed × signed       |
| vmadotu    | `e210012b`       | `0` = `00`              | A unsigned, B unsigned         | unsigned × unsigned   |
| **vmadotsu** | **`e210212b`** | **`2` = `10`**          | **A signed, B unsigned**       | **signed × unsigned (this op)** |
| vmadotus   | `e210112b`       | `1` = `01`              | A unsigned, B signed           | unsigned × signed     |

**Encoding-level corroboration (independent of the silicon run):** for vmadotsu the selector nibble is
`2` = binary `10` ⇒ bit13=1 (A signed) / bit12=0 (B unsigned). So the encoding ITSELF says signed-A ×
unsigned-B — which the X60 run then confirmed numerically. vmadotsu and vmadotus are genuinely distinct
opcodes (`...212b` vs `...112b`), so the silicon could in principle have implemented either binding — that
is exactly why the 4-way discriminator below is load-bearing.

---

## 2. The 4-way discriminator (the load-bearing test)

The single-fragment 4×4×8 kernel was driven on the COMPILER-EMITTED helper
`tcrv_ime_vmadotsu_mma_4x4x8(...)` — extracted verbatim from the materialization pipeline
(`tcrv-opt test/Conversion/EmitC/ime-mma-su-materialization.mlir --tcrv-materialize-plugin-variants
--tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes`),
NOT a hand copy — so this is an N2 claim about OUR emitter, not an isolated asm probe. Helper binds
A→v0, B→v1, C→v2/v3 and executes `vmadotsu v2, v0, v1` (the FOUNDATION load/store structure, mnemonic
the only divergence).

**HIGH-BIT discriminating data** (bytes ≥ 0x80 where signed vs unsigned interpretation differs), with
**ASYMMETRIC high-bit placement between A and B** (different rows/cols) so that mixed ≠ reverse. Four scalar
int32 references computed from the SAME byte payload; the gate ABORTS (exit 4) unless all four are pairwise
distinct — they are. On X60 (`taskset -c 0-3`):

```
MATCH: mixed=1 reverse=0 signed=0 unsigned=0
```

| reference (over identical bytes)              | IME == ref? |
|-----------------------------------------------|-------------|
| (1) mixed-sign:  A SIGNED   × B UNSIGNED       | **YES** ✓   |
| (2) reverse-mixed: A UNSIGNED × B SIGNED       | no          |
| (3) pure-signed: both SIGNED                    | no          |
| (4) pure-unsigned: both UNSIGNED                | no          |

The IME result matrix equals (1) exactly in all 16 int32 elements, e.g. first row
`1129  18210  2210  -15308` (mixed-ref) vs reverse `361 -8926 9122 -3020`, signed `105 -14302 1954 17460`,
unsigned `1385 23586 9378 29748` — all four refs visibly different, IME picks (1).

**Verdict mapping:** `IME == mixed(1) AND ≠ reverse(2) AND ≠ signed(3) AND ≠ unsigned(4)`
⇒ **PASS, signed-A(v0) × unsigned-B(v1) CONFIRMED. No operand swap. Bit-exact 16/16.**

(Had it matched reverse(2), the verdict would have been "operand swap needed — A treated unsigned, B signed —
swap A/B binding in macHelperBody": a 1-line driver fix. It did not.)

---

## 3. Hardening — the gate is not vacuous

- **SIGILL mechanically detectable:** `MARK_BEFORE_VMADOTSU` / `MARK_AFTER_VMADOTSU` both printed, exit 0
  (not 132). The `vmadotsu` executed natively.
- **Negative control (proves it's the real IME, not a fallback):** `taskset -c 4 ./ime_mma_su_gate` →
  **exit 132 (SIGILL)**. Hart 4 lacks the `_ime` token (per FOUNDATION.md), so the instruction is illegal
  there — confirming harts 0-3 genuinely *execute* the IME opcode rather than emulating it.
- **MAC-unit pin:** vlenb=32, vl(e8,m1)=32 asserted (VLEN=256, 4×4×8 int32-accum), else abort (exit 3).
- **Discriminator pin:** all four refs pairwise distinct, asserted at runtime (else abort exit 4).
- **Determinism:** 3/3 independent runs on harts 0-3 → exit 0 (PASS), identical.

---

## 4. Run transcript (real X60)

```
$ taskset -c 0-3 ./ime_mma_su_gate
OK: all four scalar refs pairwise distinct (discriminator is load-bearing)
vlenb=32  vl(e8,m1)=32  (expect 32/32 for VLEN=256)
MARK_BEFORE_VMADOTSU
MARK_AFTER_VMADOTSU
... (4 ref matrices + IME matrix) ...
MATCH: mixed=1 reverse=0 signed=0 unsigned=0
VERDICT: PASS  (IME == mixed-sign ref, != all others) -> signed-A(v0) x unsigned-B(v1) convention CONFIRMED (16/16)
EXIT_CODE=0
```

---

## 5. What this seals

- The 4th IME op `tcrv.ime.mma_su` now has the **same bit-exact-16/16-on-X60 seal** as the 1st op
  (vmadot, FOUNDATION.md). The live N2 rapid-add worked example is complete end to end:
  emit (0-core-branch, capability-fact-driven) → cross-build → objdump-real-opcode → silicon bit-exact.
- The honest caveat from commit `70bb845a` ("vmadotsu signedness-ASYMMETRIC, operand convention UNVERIFIED
  on silicon") is **resolved: the convention is correct, no swap.**
- One doc/comment correction is owed: the spec-asserted encoding `0xe210232b` should be `e210212b`
  (IMEBackendEmissionDriver.cpp:44 comment + N2-CLARITY-DESIGN.md). Kernel emission is unaffected
  (driven by mnemonic, not hardcoded encoding).

## Reproduce
Local scratch (NOT in git): `/home/kingdom/spacemit-ime/mma-su/` holds `ime_mma_su_gate.c`
(`#include "emitted_helper.c"` — the pipeline-emitted leaf), `emitted_helper.c`, and the static binary.
Toolchain root: `/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4`. On K1 the
binary is at `~/n2-ime-probe/ime_mma_su_gate`.
