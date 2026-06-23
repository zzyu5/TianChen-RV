# N2 / IME plugin BREADTH — a SECOND IME capability/op (unsigned `vmadotu`)

**Date:** 2026-06-23
**Builds on:** `PLUGIN-SLICE.md` (the first op `tcrv.ime.mma` = signed `vmadot`, committed `2eeabff9`)
and `FOUNDATION.md` (real-K1 bit-exact `vmadot`). `research/spacemit-ime.md` (the IME1 envelope).
**Goal:** deepen the N2 proof — show the IME plugin GENERALIZES (not a one-op special case) by adding
a SECOND, genuinely-different IME capability/op that gates on the SAME capability-fact dispatch.

---

## VERDICT (TL;DR)

> **A second IME op — `tcrv.ime.mma_u` (UNSIGNED int8 MAC, `vmadotu`) — is added, capability-fact
> gated (zero-core-branch), fail-closed, and VALIDATED bit-exact on real K1 (Spacemit X60).**
> It is a genuinely different emitted instruction (`vmadotu`, encoding `0xe210012b`, vs the signed
> `vmadot`'s `0xe210312b`) with distinct unsigned numeric semantics, NOT a renamed op. On real X60,
> the COMPILER-EMITTED `vmadotu` kernel is **bit-exact (16/16) vs an UNSIGNED scalar reference AND
> differs in all 16/16 from the SIGNED reference** — proving it genuinely drives `vmadotu`, not
> `vmadot`. Build CLEAN; lit 688/691 (the 3 failures are pre-existing RVV Script tests, confirmed
> failing identically on the main known-good build).

---

## Which second capability — and WHY it is a REAL second capability, not cosmetic

Per the task's three candidates, the most tractable REAL second capability is the **signedness
variant** `vmadotu` (unsigned × unsigned → int32). The IME1 spec (`research/spacemit-ime.md`,
`instruction-list.adoc`) lists the signedness family `vmadot / vmadotu / vmadotsu / vmadotus`. I
verified empirically (NOT from docs) that the SpacemiT GCC15 fork assembles all four under the SAME
`xsmtvdotii` march token, with **distinct encodings**:

| mnemonic | semantics              | encoding   |
|----------|------------------------|------------|
| `vmadot`   | signed × signed (ss)   | `0xe210312b` |
| **`vmadotu`**  | **unsigned × unsigned (uu)** | **`0xe210022b`** (asm) / `0xe210012b` (in the emitted fn) |
| `vmadotsu` | signed × unsigned      | `0xe210232b` |
| `vmadotus` | unsigned × signed      | `0xe210142b` |

So `vmadotu` is (a) a genuinely different MACHINE instruction, and (b) a genuinely different NUMERIC
operation — on bytes with the high bit set (≥ 0x80), unsigned and signed MACs differ. It is NOT a
renamed op. It rides the SAME `spacemit.ime` capability fact (the `xsmtvdotii` token assembles both
signedness forms), so it is plugin breadth, not a second capability id.

---

## The new op + variant (file:line)

- **`include/TianChenRV/Dialect/IME/IR/IMEOps.td:81`** — `def MMAUOp : TCRVIME_Op<"mma_u", …>` — the
  second IME execution boundary op `tcrv.ime.mma_u` (unsigned uint8→int32 `vmadotu`), same attribute
  schema / fragment-shape envelope as `tcrv.ime.mma`.
- **`lib/Dialect/IME/IR/IMEDialect.cpp:184`** — `verifyIMEMACBoundary(op, expectedIMEOp, emitErr)` —
  the shared fail-closed (I7) verifier; `MMAOp::verify()` (`:280`) pins `vmadot`, `MMAUOp::verify()`
  (`:292`) pins `vmadotu`. The ONLY structural axis that varies is the admitted mnemonic — so the
  signed/unsigned divergence is exactly the instruction, everything else (int8→int32, the
  VLEN-derived MAC shape, selected-path binding, no-benchmark-claim) is identical and shared.
- **`lib/Plugin/IME/IMEExtensionPlugin.cpp:117`** — `deriveIMEMatmulCapability` now derives a
  **signedness FACT** from the capability (`ime_signedness` property; `:141`): absent ⇒ `signed`
  (back-compat with the first slice), `unsigned` ⇒ `vmadotu`, anything else fails closed (`:155`).
  The unsigned variant name `ime_vmadotu_mma_slice` (`:31`) is proposed when the fact is unsigned
  (`:225`), stamping the derived `ime.signedness` plugin attribute on the variant.
- **`lib/Plugin/IME/IMEBackendEmissionDriver.cpp:107`** — `vmadotuHelperBody()` emits the `vmadotu`
  asm leaf; the EmitC pattern is templated `IMEMACToEmitCFunc<OpT>` and registered for BOTH
  `MMAOp` and `MMAUOp` (`:250`), so both ops lower through the SAME structured wrapper, differing
  ONLY in the one justified asm leaf.

---

## How it stays ZERO-CORE-BRANCH (the capability fact it gates on)

The second op gates on the **SAME** first-class `spacemit.ime` capability FACT as the first op —
`request.getCapabilities().lookupProviderByID("spacemit.ime")` + `isAvailable()`
(`IMEExtensionPlugin.cpp:183, :202`), exactly the registry/interface shape the core uses for every
family. Which of the two boundary ops (`mma` / `mma_u`) materializes is decided by a
**capability-DERIVED fact** (the `ime_signedness` request derived in `deriveIMEMatmulCapability`),
flowed through the materialized variant's `ime.signedness` plugin attribute and read back by
`variantRequestsUnsignedIME` (`:191`) at boundary/emission time. There is **NO `if family==…`** and
**NO string family-match**: a non-IME capability still fails `lookupProviderByID("spacemit.ime")` and
the IME plugin declines; a second capability id is NOT introduced (it is the same `xsmtvdotii`
envelope). The signedness is a fact of the queried capability, not a name.

Verified: no `template`/`toy`/`rvv`/`ime`/`mma_u`/`vmadotu` token appears in any core dispatch loop
in `lib/Transforms/*` or `lib/Conversion/EmitC/*`; the unsigned path rides the generic
`--tcrv-materialize-emitc-lowerable-routes` backend-registry iteration (the lit test below asserts
no other family dialect leaks, and that the SIGNED `tcrv_ime_vmadot_mma_4x4x8` helper does NOT
appear on the unsigned path).

### raw()-leaf honesty (unchanged I5 caveat)
As for the first op: there is NO IME intrinsic header, so the single instruction leaf is ONE
justified `vmadotu` asm verbatim inside a `static inline` helper, reached by a structured
`emitc.call_opaque` on the A/B/C block args. All dataflow is structured emitc. We do NOT claim
`raw()==0` for IME; surfaced, not buried.

---

## Build + lit status

- **Build CLEAN.** Fresh CMake configure in the isolated worktree (`-DLLVM_DIR=/usr/lib/llvm-20/…`,
  Ninja); `ninja -j16` of the default target + `tcrv-opt` + `tcrv-translate` is GREEN. The IME
  dialect tablegen now emits both `tcrv_ime.mma` and `tcrv_ime.mma_u`; plugin + backend emitter
  compile and link.
- **lit: 688 / 691 PASS.** The 3 failures
  (`Scripts/rvv-…-strided-input-widening-dot-reduce-add-dry-run.test` ×2, `…-e2e-self-test.test`)
  are PRE-EXISTING RVV Script tests (RVV fake-bundle metadata, no IME/vmadot token) — confirmed
  failing IDENTICALLY on the main known-good build, outside this change scope (IME files + tests).
- **New/updated tests, all green:**
  - `test/Dialect/IME/mma-u.mlir` — positive round-trip of the `tcrv_ime.mma_u` envelope + 3
    fail-closed negatives (verifier rejects a wrong mnemonic `vmadotsu`, wrong `elem_in_bits`, and a
    generic tile attr).
  - `test/Conversion/EmitC/ime-mma-u-materialization.mlir` — capability FACT `spacemit.ime` with
    `ime_signedness="unsigned"` → generic pipeline → `tcrv_ime.mma_u` (derived 4x4x8) → EmitC
    `vmadotu` kernel; `--implicit-check-not` asserts NO other family dialect AND NO signed
    `vmadot` helper leaks into the unsigned path.
  - `test/Dialect/IME/mma.mlir` — updated the wrong-op negative to the refactored verifier message
    (now mnemonic-parameterized) and to a different signedness mnemonic (`vmadotsu`).

---

## K1 bit-exact verdict — PASS (I8, real X60)

The COMPILER-EMITTED kernel (from the pipeline above through `mlir-translate --mlir-to-cpp`, NOT
hand-written) was cross-compiled with the SpacemiT GCC15.2 fork
(`-march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii -mabi=lp64d`, static). objdump confirms a real
`smt.vmadotu v2,v0,v1` (encoding `e210012b`) inside the emitted
`tcrv_emitc_ime_mma_u_kernel_ime_vmadotu_mma_slice` — distinct from the signed `vmadot`'s `e210312b`.

The driver uses **discriminating data with high-bit-set bytes** (≥ 0x80) so the unsigned and signed
references DIFFER, computes BOTH references, and requires IME == UNSIGNED ref AND IME != SIGNED ref
(so the gate cannot pass on the signed instruction — the discriminator is load-bearing).

Run on real `ssh k1` Spacemit X60, pinned `taskset -c 0-3` (IME harts; hart 4 has no `_ime`):
```
vlenb=32  vl(e8,m1)=32   (VLEN=256 / 4x4x8 MAC unit asserted — gate not vacuous)
MARK_BEFORE_EMITTED_KERNEL ... MARK_AFTER_EMITTED_KERNEL   (no SIGILL: both markers, exit 0)
UNSIGNED scalar reference C: 927 150094 66256 228908 / 736 181243 57584 182366 / ...
SIGNED   scalar reference C:  -97   1870 -11824   300 / -288   1275  8176   2142 / ...
EMITTED IME vmadotu result C: 927 150094 66256 228908 / 736 181243 57584 182366 / ...
RESULT: PASS  (IME == UNSIGNED ref all 16/16; DIFFERS from signed ref in 16/16
              => genuinely vmadotu, not vmadot)
EXIT_CODE=0
```

So the second op our IME plugin emits — from the SAME `spacemit.ime` capability fact, through the
common zero-core-branch pipeline, selected by the derived signedness fact — runs natively on real
X60 and is bit-exact vs an UNSIGNED scalar oracle, AND provably distinct from the signed MAC.

Scratch (NOT in git): `/home/kingdom/spacemit-ime/n2-ime-breadth/` (`ime_u_kernel.c` =
compiler-emitted, `ime_u_driver.c`, `ime_u_gate`); on K1 left at `~/n2-ime-probe/ime_u_gate`.

---

## N2 plugin-breadth verdict — REACHED

The IME plugin now carries TWO genuinely-different execution ops (`vmadot` signed, `vmadotu`
unsigned), both gated on the SAME `spacemit.ime` capability fact via `lookupProviderByID` (no string
match, no second capability id), both fail-closed verified, both lowered through the common EmitC
route with one justified asm leaf each. The choice between them is a pure data flow of a
capability-DERIVED signedness fact. This demonstrates the plugin GENERALIZES — it is not a one-op
special case — deepening the N2 second-family proof, validated end-to-end on real silicon.

### Honest scope notes
- This is BREADTH within IME1 (a second signedness form of the SAME int8→int32 MAC), not a new dtype
  or a new fragment shape. The other candidate ops (`vmadot1` accumulate-chaining, a wider MAC tile,
  the `_slide` form) remain available as further breadth; `vmadotu` was chosen as the smallest REAL
  second capability that is a genuinely different emitted instruction with a load-bearing
  discriminating test. It was sufficient for one pass and did not need scoping down.
- No target-artifact-export route was added (same scope boundary as the first slice / PLUGIN-SLICE.md):
  the EmitC emission + K1 bit-exact are the deliverable; target-export is a separate adapter and is
  not a zero-core-branch gap (dispatch + emission are fully capability-driven and branch-free).
