# N2 clarity — SHARPENED (the crystal-clear rapid-family-add proof)

*Read-only design doc (2026-06-25). Supersedes the framing of `N2-RAPID-ADD-SHOWCASE.md` where the two
disagree. Every number here is re-derivable with the exact commands shown; no lib/ edits are proposed.*

The user says the N2 IME experiment "感觉不是很清晰" — it is not clear how we PROVE a rapid family-add, WHAT
we reused, or how we leverage the N2 abstraction. A standalone showcase already exists and the complaint
**persisted past it**, so the answer is NOT "here is another showcase." Below: (§1) what specifically reads as
muddy and why the prior pass did not close it, (§2) the reuse-quantified rapid-add proof with the missing
counterfactual, (§3) the dispatch-by-FACT demo scoped correctly to CORE, (§4) the one-sentence abstraction
thesis, (§5) the two undeniable artifacts a reviewer checks in 30 seconds.

---

## §1 — DIAGNOSE the clarity gap (why the existing showcase did not land)

The proof content is right; the *presentation* leaks legibility in three specific places:

1. **It leads with a big ADD number that reads as "expensive."** The headline is "1918 isolated plugin lines."
   A reader's first impression of a *rapid* add should be **what we did NOT have to write**, not how many lines
   the family is. The reuse is *asserted* ("6 interfaces, modified=0") but never given its **counterfactual**:
   *which core passes ran IME end-to-end while contributing zero IME lines.* That counterfactual is the actual
   answer to "发挥n2抽象 / 不像成熟compiler" and it is missing.

2. **The headline LOC contradicts the commit a reviewer will run.** The showcase says **1918**; the founding
   commit `2eeabff9` is **+1644 −0**; the live tree today is **2056**. Three different numbers with no
   provenance ⇒ a skeptic who runs `git show` concludes the doc is loose. The fix is not a single number — it
   is a **decomposition with provenance** (founding-op vs cumulative; local vs test vs doc vs wiring).

3. **The "not string-match" claim is stated tree-wide, but the plugin DOES compare strings.** `IMEBackendEmissionDriver.cpp:383` is literally `matmul.getImeOp() == "vmadotu"`; mnemonic string constants
   live throughout the plugin and dialect. A skeptic greps `lib/Plugin/IME` and finds them, and the claim looks
   false. The real, defensible claim is narrower and stronger: **zero family-identity branch in any CORE pass**;
   the plugin reading its OWN derived `ime_op`/signedness fact is *capability consumption*, not core
   string-matching. The doc must scope this to CORE precisely.

**Net:** N2 needs no new structural work (it is PROVEN — commit `2eeabff9`, K1 bit-exact). It needs (a) the
reuse counterfactual made concrete, (b) the LOC numbers reconciled with provenance, (c) the anti-string-match
claim scoped to core. This doc supplies all three.

---

## §2 — The reuse-quantified rapid-add proof (with the missing counterfactual)

### 2a. The LOC table, reconciled with provenance (dissolves the 1644/1918/2056 contradiction)

Founding commit `2eeabff9` ("IME second family added as a plugin") — `git show --numstat 2eeabff9`:

| Bucket | LOC | What it is |
|---|---:|---|
| **IME-local** (src + headers + CMake under `*/IME/*`) | **1267** | the new family — dialect, plugin, driver |
| Tests (3 lit files) | 143 | round-trip + materialization + capability-absent-negative |
| Doc (`PLUGIN-SLICE.md`) | 225 | the slice writeup (not code) |
| **Wiring** (registration rows + `add_subdirectory`) | **9** | 4 sites, all additive — see 2b |
| **CORE modified** (`lib/Transforms/*`, capability model, EmitC core) | **0** | nothing in any selection/materialization pass |
| Deletions, anywhere | **0** | purely additive |
| | **= +1644 −0** total in the commit | |

The **1918** in `N2-RAPID-ADD-SHOWCASE.md` is the *cumulative* IME footprint quoted at the time the showcase
was written; the **live tree today is 2056** (ops 2/3/4 grew it):
`git ls-files lib/Plugin/IME lib/Dialect/IME include/TianChenRV/Dialect/IME include/TianChenRV/Plugin/IME | xargs wc -l` ⇒ **2056 total**. Present the founding op (1267 local / 9 wiring / **0 core**) as the rapid-add datum;
the cumulative number is incidental and should never be the headline.

**The repeatability that proves the wiring was "paid once"** — ops 2/3/4 each added an op with **0 core / 0
wiring** (`git show --numstat <c>`, filtered to core/wiring paths ⇒ 0 touches):

| Op | Commit(s) | Core LOC | Wiring LOC |
|---|---|---:|---:|
| 1st `vmadot` (mma) | `2eeabff9` (founding) | 0 | 9 (paid here, once) |
| 2nd `vmadotu` (mma_u) | `c7069111` | **0** | **0** |
| 3rd tiled `matmul` | `0b8c6168` | **0** | **0** |
| 4th `vmadotsu` (mma_su) | `70bb845a` + `a75adc27` | **0** | **0** |

### 2b. The wiring is 4 single rows in registration tables (the "mature-compiler registry" shape)

All 9 wiring lines are **registration-table rows + link deps**, sitting in the *same array as RVV/Toy/Template*:

- `lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp` (+2): one `#include` + one row
  `::tianchenrv::plugin::ime::registerIMEBackendEmitter,` appended **directly after**
  `registerRVVBackendEmitter` / `registerToyBackendEmitter` in `kBuiltinBackendEmitters[]`.
- `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` (+2): one `#include` + one bundle row
  `{"ime-extension-bundle", registerIMEExtensionPlugin},`.
- 3× `add_subdirectory` (`include/.../Dialect`, `lib/Dialect`, `lib/Plugin` CMakeLists) + 2 `LINK_LIBS` rows.

There is **no** `case IME:` and no `if (family == ...)` — the core iterates the registry array. This is exactly
LLVM's `TargetMachine` / `RegisterTargetMachine` registry shape.

### 2c. The COUNTERFACTUAL — the core passes that ran IME for free (the missing piece)

This is the answer to "发挥n2抽象." Each pass below is a generic `lib/Transforms/*` pass that orchestrated the
IME variant **end-to-end** while containing **0 IME lines** (proven by the §3 grep). The IME plugin supplied
only vtable bodies; the *control flow* is the core's:

| Core pass (`lib/Transforms/`) | What it did for IME | How it called IME (generic, no family name) |
|---|---|---|
| `VariantMaterialization.cpp` | asked every plugin to propose IME variants | `registry.collectVariantProposals(request, …)` **:596** |
| `VariantSelection.cpp` | picked among RVV/IME/fallback by cost | calls plugin `estimateVariantCost` via registry |
| `PluginVariantLegality.cpp` | fail-closed legality (I7) | calls plugin `verifyVariantLegality` via registry |
| `VariantDispatchSynthesis.cpp` | synthesized the capability-gated dispatch | reads the variant's required-capability attr (data) |
| `LoweringBoundary.cpp` | materialized the `tcrv.ime.mma` boundary op | calls plugin `materializeSelectedLoweringBoundary` |
| `EmitCLowerableMaterialization.cpp` | lowered the boundary op to EmitC C++ | `tryConvertModuleWithRegisteredBackend(module)` **:268** |

**The reused-interface anchors (RVV declares them; IME reuses them UNMODIFIED, modified = 0):**

| Reused common infra | RVV anchor | IME anchor (same signature) |
|---|---|---|
| `ExtensionPlugin` vtable (8 hooks) | `RVVExtensionPlugin.cpp:456` (`collectVariantProposals`) | `IMEExtensionPlugin.cpp:412` (`collectVariantProposals`) |
| `registerXXXBackendEmitter(BackendEmissionRegistry&)` | `RVVBackendEmissionDriver.h:18` | `IMEBackendEmissionDriver.h:21` / `.cpp:531` |
| `TCRVEmitCLowerableOpInterface` (keys the EmitC route) | declared in RVV ODS | `IMEOps.td:5,36` (`DeclareOpInterfaceMethods`) |
| `CapabilityDescriptor` / `lookupProviderByID` | RVV queries its own id | `IMEExtensionPlugin.cpp:262-263` |
| variant/dispatch/legality/boundary orchestration | the `lib/Transforms/*` passes above | IME writes **none** — fills vtable hooks |
| EmitC → C++ route (`mlir-translate --mlir-to-cpp`) | stock | IME adds **no** exporter |

**One-line counterfactual:** to add IME we wrote 1267 isolated lines + 9 registration rows; the **six** core
passes in 2c (variant materialization, selection, legality, dispatch synthesis, boundary, EmitC lowering)
**ran the IME family without a single IME-specific line** — that "ran for free" is the rapid-add.

---

## §3 — DISPATCH-NOT-STRING-MATCH (scoped to CORE; the negative test)

**The fact, flowing as DATA.** The discriminator the core uses is the *presence of a derived capability
provider keyed on the family's OWN id*, never a family name the core spells:

```
// IMEExtensionPlugin.cpp:262-263 (the plugin queries ITSELF; core names no family)
const support::CapabilityDescriptor *capability =
    request.getCapabilities().lookupProviderByID(kIMECapabilityID);   // "spacemit.ime"
return capability && capability->isAvailable();
```

- `"spacemit.ime"` is defined **once** (`IMEExtensionPlugin.cpp:25`).
- It exists only because **`deriveIMEMatmulCapability` (:147)** derived it from *validated ISA evidence*: the
  march token **`xsmtvdotii`** (`:98`) ⇒ `vmadot` assembles; VLEN=256 / SEW=8 ⇒ the 4×4×8 int32 MAC fragment
  **computed from capability, not hard-coded** (change VLEN ⇒ a different fragment is derived — an N1 hook
  living inside N2).
- The op's signedness travels as a **derived `ime_op` fact in the variant attribute**, read back at boundary
  time. The plugin's `matmul.getImeOp() == "vmadotu"` (`driver:383`) reads **its own derived data**, not a
  core string about a family.

**The immature alternative — named, and PROVEN ABSENT in CORE (the precise claim):**

```
if (op.getName().contains("matmul") && target.march.contains("spacemit")) emitVmadot();  // does NOT exist in any core pass
```

```
$ grep -rinE '\bime\b|spacemit|vmadot|xsmtvdotii' lib/Transforms/        ⇒ 0 hits
$ grep -rinE '\bvmadot\b|xsmtvdotii|spacemit\.ime' lib/ include/ \
      --include=*.cpp --include=*.h --include=*.td | grep -viE '/IME/'    ⇒ 0 hits
```

**Scope statement (so the claim is not falsifiable):** zero family-identity branch in any **core** pass
(grep-proven, 0 hits outside `*/IME/*`). Mnemonic strings exist **inside the plugin** because that is where the
family's own derived fact is consumed — capability consumption, not core string-matching. "No string compare
anywhere" would be false; "no family branch in core" is true and is the load-bearing claim.

**The complementary-arm negative test** — `test/Conversion/EmitC/ime-mma-capability-absent-negative.mlir`: an
RVV-but-not-IME target does NOT satisfy `lookupProviderByID("spacemit.ime")`, IME **declines** (`supportsOperation`
returns false), the op routes elsewhere, and **no** `vmadot`/`spacemit` token is materialized. Dispatch is
fact-gated, not name-hardcoded — the fact-arm and the absent-arm are both exercised.

---

## §4 — How this leverages the N2 abstraction (the one-sentence thesis)

> **A hardware family is admissible to this compiler ⟺ its capability can be expressed as a RISC-V capability
> FACT *and* consumed with zero core-family branches** — admission is a derived-capability descriptor + a
> plugin vtable, never a core edit.

IME instantiates it exactly: the fact is `spacemit.ime`, **derived** from the `xsmtvdotii` ISA envelope (not a
vendor name string); the zero-core-branch consumption is the §2c registry dispatch (grep-0 in §3). Because the
admission criterion is "is it a RISC-V capability fact," a discrete accelerator (GPU/TPU/910B) is *out* — it
cannot present its capability as a RISC-V ISA fact — while IME, an X60 ISA extension, is *in*. The thesis both
admits IME and draws the boundary, which is what makes it a *first-class* abstraction rather than a special case.

**N2 = STRUCTURE only.** The Q5 metric "performance over RVV for suitable kernels" is **explicitly out of
scope here** — IME e2e perf is an N3 question and is currently NULL/regime-dependent (micro 5.51× → decode
0.86×; see memory). N2's proven claim is the rapid, zero-core-branch, fact-dispatched, K1-bit-exact STRUCTURAL
add. Do not let a perf number leak into the N2 proof.

### Q3 / Q5 metric crosswalk (speak the spec's language; perf rows marked N3)

| Spec metric (experiment-reference Q3/Q5) | N2 value | Anchor |
|---|---|---|
| core pass modified LOC | **0** | `git show --numstat 2eeabff9`; grep §3 |
| plugin LOC | 1267 founding / 2056 current | §2a |
| new capabilities | 1 (`spacemit.ime`, derived) | `IMEExtensionPlugin.cpp:25,147` |
| new ops/types | 4 ops (mma / mma_u / mma_su / matmul) | `IMEOps.td:35,81,131,185` |
| extension-specific branches in core pass | **0** | grep §3 |
| reuse of variant/dispatch/verifier orchestration | full reuse, 0 modified | §2c table |
| IME variant generation coverage | 4 ops proposed via `collectVariantProposals` | `:412` |
| IME legality verifier effectiveness | fail-closed (I7), admits only the validated envelope | `IMEDialect.cpp:198` `verifyIMEMACBoundary` |
| IME emission success | EmitC → objdump real `smt.vmadot*`, K1 bit-exact | IME-MMA-SU-K1-FINDING.md |
| performance over RVV | **N3 — out of N2 scope** (currently NULL/regime-dependent) | memory: kernel-wins-dont-transplant |

---

## §5 — What makes it UNDENIABLE (two reproducible artifacts; both read-only)

A reviewer should not need to trust prose. Two commands settle it in ~30 seconds each; **no 5th op is needed**
— the 4th op is already the live worked example (0 core / 0 wiring + silicon 16/16).

**Artifact A — the rapid-add LOC table, mechanically (proves "core ≈ 0, wiring paid once"):**
```
git show --numstat 2eeabff9                          # founding op: +1644 −0, 0 in lib/Transforms
for c in c7069111 0b8c6168 70bb845a a75adc27; do     # ops 2/3/4: each 0 core / 0 wiring
  git show --numstat $c | grep -E 'lib/Transforms|EmitC/Builtin|Plugin/Builtin' ; done
git ls-files lib/Plugin/IME lib/Dialect/IME include/TianChenRV/Dialect/IME \
             include/TianChenRV/Plugin/IME | xargs wc -l   # current footprint = 2056
```
Result: founding 1267 local / 9 wiring / 0 core; ops 2–4 each 0/0 ⇒ the wiring was paid exactly once.
Repeatability *is* the rapid-add proof (assertion → demonstration).

**Artifact B — dispatch-by-fact, mechanically (proves "no core string-match"):**
```
grep -rinE '\bime\b|spacemit|vmadot|xsmtvdotii' lib/Transforms/                       # ⇒ 0
grep -rinE '\bvmadot\b|xsmtvdotii|spacemit\.ime' lib/ include/ \
     --include='*.cpp' --include='*.h' --include='*.td' | grep -viE '/IME/'           # ⇒ 0
tcrv-opt test/Conversion/EmitC/ime-mma-capability-absent-negative.mlir ...            # IME declines, 0 vmadot token
```
Result: 0 family tokens anywhere in core; the negative lit shows the fact-gate declining when `spacemit.ime` is
absent. Dispatch is a derived FACT flowing as data, not a name the core spells.

These two artifacts + the existing K1 silicon seal (IME-MMA-SU-K1-FINDING.md: `vmadotsu` 16/16 bit-exact on
real X60, 4-way signedness discriminator) close the clarity gap: the rapid-add is *counted*, the dispatch is
*grep-falsifiable*, and the emitted kernel *runs on silicon*.
