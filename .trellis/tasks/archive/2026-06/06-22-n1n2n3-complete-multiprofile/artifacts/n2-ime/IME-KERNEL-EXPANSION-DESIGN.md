# IME KERNEL EXPANSION — design doc (read-only; no lib/ edits)

**Date:** 2026-06-25
**Scope:** Design (only) a meaningful NEW IME kernel that demonstrates the N2 abstraction
(capability-fact dispatch, zero-core-branch) — "增加更多 ime kernel". No `lib/` edits.
**Hardware target:** `ssh k1` = SpacemiT X60, RVV1.0 VLEN=256, IME1, `taskset -c 0-3`
(harts 0-3 carry `_ime`; hart 4 does not). march `rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii`,
SpacemiT GCC15.2 fork at `/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/`.
**Existing 4 IME ops:** `tcrv.ime.mma`(vmadot) / `mma_u`(vmadotu) / `mma_su`(vmadotsu) +
`tcrv.ime.matmul` (tiled). All four are signedness variants of the SAME single 4×4×8 int8→int32
MAC fragment (funct7 `111000`). That axis (signedness) is largely worked; the next REAL breadth
is a **different funct7 / different kernel shape**, not a 5th signedness.

---

## 0. What the SpacemiT assembler ACTUALLY accepts (empirical, this box)

Probed `riscv64-unknown-linux-gnu-as -march=…xsmtvdotii` directly + objdump. The full
`vmadot{,1,2,3}` × `{ss,uu,su,us}` 16-mnemonic matrix ASSEMBLES, all distinct encodings:

| family | ss (signed)  | uu (unsigned) | su (mixed) | us (mixed) | funct7 |
|--------|-------------|---------------|------------|------------|--------|
| `vmadot`  | `e210312b` ✓DONE | `e210012b` ✓DONE | `e210212b` ✓DONE | `e210112b` (us, missing) | `111000` |
| **`vmadot1`** | **`e610312b`** | `e610012b` | `e610212b` | `e610112b` | `111001` |
| `vmadot2` | `e610712b` | `e610412b` | `e610612b` | `e610512b` | `111001` |
| `vmadot3` | `e610b12b` | `e610812b` | `e610a12b` | `e610912b` | `111001` |

`vmadotu2`/`vmadotsu3`/etc. (suffix-after-signedness) REJECT — the index is a prefix
(`vmadot1u`, not `vmadotu1`). bit26 (`e6` vs `e2`) is the funct7 `111000`→`111001` flip.

**Why `vmadot1` is the right pick (not just "accepts"):** `FindSMTIME.cmake`
(`ggml/src/ggml-cpu/cmake/FindSMTIME.cmake`) compile-probes `vmadot1 v2,v0,v1` on its OWN symbol
(`SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOTN`, line 17) — separate from the plain `vmadot` that
*determines* IME1 (line 11→23) and from the `i4`/`vpack`/`vnspack` triple that determines IME2-only
(lines 12,18,19→27). So `vmadot1` is NOT an IME2-gated opcode; it sits in the plain-`xsmtvdotii`
slide family. The stronger K1/X60-support signal is the research doc: `Xsmti8i32mm_slide` is listed
explicitly as one of **K1's two supported IME1 sub-extensions** (`Xsmti8i32mm`, `Xsmti8i32mm_slide`).
`vmadot2`/`vmadot3` assemble (`e610712b`/`e610b12b`) but are NOT individually probed and have no
independent K1-support signal → follow-ons, silicon-gated. (HONEST: K1-support is documentary, not
yet silicon-run — §5 step 1 is the silicon gate.)

---

## 1. SEMANTICS — pinned from the spec source, NOT inferred

Authoritative: `/home/kingdom/spacemit-ime/instruction-func.adoc` (riscv-ime-extension-spec
`src/instruction-func.adoc`) + `instruction-list.adoc`. **Correction of an earlier wrong guess:**
`vmadot1/2/3` are NOT accumulator-bank-index / wider-tile variants. The `vmadot-x` data-type table
lists them under categories **`slide-1` / `slide-2` / `slide-3`** — they are the
**SLIDING-WINDOW dot-product MAC** (`Xsmti8i32mm_slide`). Integer-slide pseudocode (verbatim, the oracle):

```
Copies = (sqrt(VLEN/64)==floor(...) ? 1 : 2)        // VLEN=256 ⇒ Copies=1
for (i=0;i<M;i++) for (j=0;j<N;j++) for (k=0;k<K;k++)
    C[i*N+j] += int32( A[ slide*K + i*K + k ] * B[ j*K + k ] );
```

The difference from plain `vmadot`: **A is read from an EVEN VS1:VS1+1 register PAIR** holding a
`2*M × K` = **8×8** int8 A-block; the `slide` index (0/1/2/3) shifts the A read-window DOWN by
`slide` rows. So one loaded 8×8 A-pair feeds FOUR overlapping 4×8 matmuls:
`vmadot`→A rows 0..3, `vmadot1`→rows 1..4, `vmadot2`→rows 2..5, `vmadot3`→rows 3..6.
B (VS2) and C (even VD pair, 4×4 int32) are identical to the non-slide MAC.

**This is the 1-D sliding-window / strided-A-reuse convolution primitive** — a genuinely new
kernel SHAPE (different funct7, A-pair input, slide field), not a renamed MAC.

---

## 2. FIRST candidate (+ follow-ons)

> **FIRST: `tcrv.ime.mma_slide` — the IME1 sliding-window int8→int32 MAC (`vmadot1`, slide=1).**
> Lead concrete instruction `vmadot1` (signed ss). HARDWARE-VERIFIABLE (assembles `e610312b`,
> IME1-probe-listed, objdump-confirmable, scalar-oracle runnable on X60). MEANINGFUL (the
> `Xsmti8i32mm_slide` conv/strided-reuse building block — the *second* of K1's two IME1
> sub-extensions; the inference path's depthwise/1-D-conv and overlapped-window GEMV uses it).
> DEMONSTRATES N2 (selected by a capability-DERIVED fact `ime_slide`, zero-core-branch, fail-closed).
>
> **Follow-on A:** the slide-index knob `slide ∈ {1,2,3}` (`vmadot1/2/3`) — the Win-A tune axis
> (window stride), all under the SAME `spacemit.ime` capability, silicon-gated for slide 2/3.
> **Follow-on B (airtight, zero-risk breadth):** `tcrv.ime.mma_us` — the ONE missing plain
> signedness sibling `vmadotus` (`e210112b`, unsigned-A × signed-B). Same proven 4×4×8 envelope as
> the existing 3; encodes cleanly; bit-exact oracle is trivial. Use this if the slide family's
> silicon semantics come back awkward — it closes the signedness square with no shape risk.

**Honest framing (advisor catch):** assembler-accept ≠ silicon-execute ≠ known-semantics. The slide
A-pair binding (which even-pair half is row 0, exact slide direction) is spec-pinned but
silicon-UNVERIFIED — exactly the state `vmadotsu` was in before its X60 run. So write `vmadot1` as
**lead candidate, semantics-probe-FIRST** (§5 step 1 is a discriminating X60 run), not as an
already-sealed kernel.

---

## 3. The op design (clone `MMASUOp` — the most recent rapid-add)

### 3a. ODS schema — `include/.../IME/IR/IMEOps.td`
Clone `MMASUOp` (single-fragment surface). Identical attribute schema PLUS one slide fact:

```tablegen
def MMASlideOp : TCRVIME_Op<"mma_slide",
    [DeclareOpInterfaceMethods<TCRVEmitCLowerableOpInterface,
      ["getTCRVEmitCLowerableSourceOpName","getTCRVEmitCLowerableSourceRole"]>]> {
  let summary = "IME int8->int32 sliding-window MAC (vmadot1/2/3) boundary";
  // ... description: Xsmti8i32mm_slide; A from even VS1:VS1+1 8x8 pair; slide shifts
  //     A window by slide rows; C += A_window . B^T; funct7 111001 (not 111000).
  let arguments = (ins
    StrAttr:$source_kernel, FlatSymbolRefAttr:$selected_variant,
    StrAttr:$origin, StrAttr:$role, StrAttr:$status,
    ArrayAttr:$required_capabilities,
    StrAttr:$ime_op,                 // "vmadot1" | "vmadot2" | "vmadot3"
    I64Attr:$elem_in_bits,           // 8   (fail-closed)
    I64Attr:$accum_bits,             // 32  (fail-closed)
    I64Attr:$mac_m, I64Attr:$mac_n, I64Attr:$mac_k,   // 4,4,8 (VLEN=256)
    I64Attr:$slide,                  // 1|2|3 — the capability-derived window-stride FACT
    StrAttr:$available_harts,        // "0-3"
    OptionalAttr<StrAttr>:$ime_reason);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}
```
The ONLY schema delta vs `MMASUOp` is the `$slide` integer fact (1/2/3). Register the op in the
illegal-op set + the templated `IMEMACToEmitCFunc` registration list of the emission driver.

### 3b. Capability-fact gating — `lib/Plugin/IME/IMEExtensionPlugin.cpp`
Same first-class `spacemit.ime` capability FACT (`lookupProviderByID("spacemit.ime")` +
`isAvailable()`) — NO new capability id, NO family string-match. In `deriveIMEMatmulCapability`,
read a new property `ime_slide` (the existing pattern for `ime_signedness`/`ime_matmul_shape`):
- absent / `"0"` ⇒ non-slide (back-compat: existing `vmadot` path),
- `"1"|"2"|"3"` ⇒ derive `imeOp = "vmadot{1,2,3}"`, stamp an `ime.slide` plugin-owned variant attr,
- anything else ⇒ **fail closed** (only slide 1/2/3 modeled at VLEN=256; matches the existing
  fail-closed style for out-of-envelope signedness). A non-IME capability still fails the lookup and
  the plugin declines — the slide choice is a FACT of the queried capability, not a name.

### 3c. Emission — `lib/Plugin/IME/IMEBackendEmissionDriver.cpp`
The existing `macHelperBody(helperName, mnemonic)` is ALREADY parameterized over mnemonic. Add a
slide helper variant that (i) loads A as an even PAIR (`vle8 v0,(A)` + `vle8 v1,(A+32)` → v0:v1 is
the 8×8 A-block; B→v2, C→v4:v5 even pair to respect "VS1 even, VD even"), (ii) emits the one
justified asm leaf `vmadot{1,2,3} v4, v0, v2`. Same single-asm-leaf-in-`static inline`-helper
discipline (no IME intrinsic header exists; surfaced, not buried — `raw()==0` NOT claimed). The
signed/slide divergence is exactly the mnemonic string + the A-pair load; everything structural is
shared.

### 3d. Verifier (boundary check) — `lib/Dialect/IME/IR/IMEDialect.cpp`
Reuse the shared `verifyIMEMACBoundary(op, expectedIMEOp, …)`. `MMASlideOp::verify()` pins
`ime_op ∈ {vmadot1,vmadot2,vmadot3}` (extend the single-mnemonic check to the 3-member slide set or
parameterize per `slide`), int8→int32 envelope, positive 4×4×8 fragment, selected-path binding,
no-benchmark-claim — all already enforced. ADD: `slide ∈ {1,2,3}` (fail-closed; reject 0 here since
slide=0 is the existing non-slide op, and reject ≥4 — out of the documented slide-1..3 family).

### 3e. Rapid-add cost (plugin LOC, 0 core)
| piece | file | approx LOC | core? |
|-------|------|-----------|-------|
| ODS op (clone MMASUOp + `$slide`) | `IMEOps.td` | ~50 | no (plugin dialect) |
| capability derive (`ime_slide` fact) | `IMEExtensionPlugin.cpp` | ~25 | no (plugin) |
| variant attr + boundary route | `IMEExtensionPlugin.cpp` | ~15 | no (plugin) |
| slide emitter helper + registration | `IMEBackendEmissionDriver.cpp` | ~30 | no (plugin) |
| verifier slide-set + slide∈{1,2,3} | `IMEDialect.cpp` | ~15 | no (plugin) |
| lit: round-trip + 3 fail-closed neg + materialization | `test/.../ime-mma-slide*.mlir` | ~80 | no |
| **total** | | **~135 plugin LOC, 0 core LOC** | **0 core branches** |

Matches the established rapid-add cost (mma_u / mma_su were the same shape). The N2 claim is exactly
this: a genuinely-new-shape IME kernel slots in at ~135 plugin LOC with ZERO core edits and ZERO
core family-branches.

---

## 4. Win-A / Win-B / Win-C mapping FOR THIS kernel

- **Win-B (algorithm-change vs ggml's shipped RVV kernel — the MANDATED baseline, NOT scalar):**
  the candidate is `Xsmti8i32mm_slide`. A 1-D sliding-window int8 matmul on RVV (no IME) must
  re-load / re-stride the overlapping A-window per output position (or im2col-expand A). The IME
  slide kernel computes 4 overlapping windows from ONE loaded 8×8 A-pair in 4 instructions with NO
  A re-load — a real algorithm change (structural A-reuse the RVV path cannot express). Report BOTH:
  **micro on K1** (slide-conv tile vs ggml/RVV strided-dot baseline) AND **e2e** — but see §6: the
  e2e regime caveat is load-bearing. This is the dimension the slide kernel primarily fills.
- **Win-A (an IME tune knob on/off):** **YES — `slide ∈ {1,2,3}`** is a genuine IME tune knob (the
  window stride / dilation), AND the A-pair-reuse factor (how many slides share one loaded A-pair: 2
  vs 4) is a second knob. On/off = slide-reuse kernel vs naive per-window reload. This is the
  cleanest Win-A the IME family offers beyond the existing signedness axis. (Honest: a microbench
  knob; whether it surfaces e2e is §6.)
- **Win-C (a pass that changes structure):** **NONE expected.** Like the mma_su rapid-add, this is a
  boundary-op + emitter; it adds no new structural pass. Per `[[winc-structural-null]]`, do NOT
  manufacture a Win-C. The A-pair-reuse is in the emitted kernel body, not a separable pass.

So this kernel fills **Win-B (primary) + Win-A (slide/reuse knob)**; Win-C = honest NULL.

---

## 5. K1 verify plan (emit → cross-build → objdump → X60 oracle)

1. **SEMANTICS-DISCRIMINATING X60 run FIRST (advisor-mandated, mirrors the mma_su 4-way test).**
   Before claiming the kernel: drive `vmadot` AND `vmadot1` on the SAME loaded 8×8 A-pair with
   HIGH-BIT discriminating data; compute the 4 candidate oracles (slide-down-1, slide-up-1,
   no-slide, pair-other-half) from the §1 pseudocode; require IME==slide-1-oracle AND ≠ all others.
   This pins the A-pair half / slide direction on silicon (the one spec-pinned-but-unverified fact).
2. **Emit:** run the materialization pipeline on a `tcrv.ime.mma_slide` test (`--tcrv-materialize-
   plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries
   --tcrv-materialize-emitc-lowerable-routes`) → `mlir-translate --mlir-to-cpp` → the emitted helper
   (proves it's OUR emitter, not a hand probe).
3. **Cross-build:** SpacemiT GCC15.2 `-O2 -static -march=rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii
   -mabi=lp64d`.
4. **objdump:** confirm a real `smt.vmadot1 v4,v0,v2` (`e610312b`) inside
   `tcrv_emitc_<kernel>_<variant>` — distinct from `vmadot`'s `e210312b`.
5. **Run on X60 `taskset -c 0-3`:** vs the §1 scalar slide oracle, require bit-exact 16/16.
   Pin the MAC unit (vlenb==32, vl(e8,m1)==32) and assert the 4 oracles pairwise-distinct
   (load-bearing discriminator). Negative control: `taskset -c 4` ⇒ SIGILL (exit 132) proves native
   IME execution (hart 4 lacks `_ime`), not emulation. 3/3 determinism.

---

## 6. HONEST perf caveat (the regime that matters)

Per `[[kernel-wins-dont-transplant-to-e2e]]`: prior IME microbench wins do NOT transplant to
memory-bound DECODE e2e (IME 5.51× micro → **0.86× decode** e2e). And `IME-PREFILL-PROBE.md` showed
even the prefill "wins" (1.37–1.95×) were a reconstructed-build CODEGEN artifact (decay with M,
+1.25× even in the can't-help M=1 decode control), NOT an IME-matmul-intensity win — so there is
currently **no idle-box-confirmed IME-attributable e2e win** on tinyllama-Q4_0.

**Therefore frame the slide kernel's perf honestly:**
- The IME-suitable regime is **compute-bound PREFILL / GEMM / batched-conv windows**, NOT M=1
  decode. The slide primitive's reuse advantage (one A-pair → 4 windows) is an arithmetic-intensity
  play, which is exactly the regime that *can* help — but only if measured against ggml's RVV
  slide/conv path in a prefill-shaped or batched-window microbench, AND only if a build-confound-free
  A/B is constructed (the prefill probe's lesson: same toolchain both arms, or the codegen delta
  swamps the IME delta).
- **Defensible claim ceiling = kernel-level micro win vs the RVV slide baseline + bit-exact silicon
  correctness.** Do NOT pre-announce an e2e win; report kernel and e2e SEPARATELY, and expect the
  decode-e2e transplant to be NULL (memory-bound), consistent with every prior IME finding.

---

## 7. Verdict (one line)

Add `tcrv.ime.mma_slide` (IME1 `vmadot1`, the `Xsmti8i32mm_slide` sliding-window MAC) as the first
new kernel: hardware-verifiable (assembles `e610312b`, IME1-probe-listed, objdump+oracle on X60),
meaningful (K1's *second* IME1 sub-extension; the conv/strided-A-reuse GEMM building block),
N2-demonstrating (capability-derived `ime_slide` fact, zero-core-branch, fail-closed), ~135 plugin
LOC / 0 core. It fills Win-B (algorithm-change A-reuse vs RVV) + Win-A (slide/reuse knob); Win-C =
honest NULL. Follow-ons: the slide-index knob `vmadot2/3` (silicon-gated) and the airtight missing
signedness sibling `vmadotus` (`tcrv.ime.mma_us`). Perf: micro-win regime is compute-bound
prefill/conv; decode-e2e transplant expected NULL (memory wall) — report kernel and e2e separately.
