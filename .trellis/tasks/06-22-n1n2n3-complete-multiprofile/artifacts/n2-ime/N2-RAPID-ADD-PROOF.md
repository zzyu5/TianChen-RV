# N2 RAPID-ADD PROOF — the reviewer-facing answer to "prove the rapid family-add + what was reused" (2026-06-25)

This supersedes the *presentation* of `N2-RAPID-ADD-SHOWCASE.md` (the structure was right; it led with the wrong
number). Every figure here is reproduced by a one-line command (shown), not asserted.

## The claim, in one sentence
**Adding a whole new RISC-V ISA family (SpacemiT IME — a matrix/MAC extension, a fundamentally different compute
shape from RVV) required ZERO core-pass lines.** The compiler routed IME end-to-end through the SAME six core
passes that serve RVV, selected by a capability FACT, with no IME knowledge anywhere in `lib/Transforms/`.

## Proof 1 — the counterfactual (what we did NOT write)
Founding IME commit `2eeabff9` (`git show --numstat 2eeabff9`):

| bucket | lines | note |
|---|---|---|
| **core passes** (`lib/Transforms/`) | **0** | the load-bearing number — the core never learned about IME |
| IME-local (`*/IME/*`) | ~1356 | the plugin: dialect, ops, verifier, capability, emission driver |
| tests + docs | ~285 | |
| **wiring** | **3** | `BuiltinBackendEmitters.cpp +2` + a `CMakeLists.txt +1` — registration ROWS, not logic |
| **total / deletions** | 1644 / **0** | a purely additive plugin; nothing in core was edited or removed |

And the wiring was paid ONCE — subsequent IME ops added at **0 core, 0 wiring**:

| op | commit | `lib/Transforms` Δ | total |
|---|---|---|---|
| op2 `mma_u` / vmadotu | `c7069111` | **0** | 3041 (mostly tests/coverage) |
| op3 tiled `ime.matmul` | `0b8c6168` | **0** | 598 |
| op4 `mma_su` / vmadotsu | `70bb845a` | **0** | 338 |

Reproduce: `git show --numstat 2eeabff9 c7069111 0b8c6168 70bb845a | awk '$3 ~ /lib\/Transforms\//{print}'` → empty.

## Proof 2 — what was REUSED (the N2 abstraction doing the work)
The IME plugin authored **0 lines** in these six core passes, yet each ran IME end-to-end (file:line are the core
entry points, all RVV-shared):
`VariantMaterialization` (`lib/Transforms/...:596` `registry.collectVariantProposals`) · `Selection` ·
`PluginVariantLegality` · `DispatchSynthesis` · `LoweringBoundary` · `EmitCLowerableMaterialization`
(`...:268`). IME provides only a provider that answers the registry's `collectVariantProposals` (IME plugin `:412`,
the exact shape RVV uses at `:456`) — the core orchestration (variant generation, capability-legality, dispatch
synthesis, EmitC lowering) is **inherited verbatim**. *That* is the reuse: not a copied kernel, but the entire
common pipeline consuming a second family with no new control flow.

## Proof 3 — selection is a capability FACT, NOT a family string-match (the "mature compiler" bar)
The honest scoping the old showcase missed: the plugin **does** read its own mnemonic (legitimate — it emits the
instruction). The claim is precisely **zero family branch in CORE**:

- `grep -rniE "spacemit|vmadot|xsmtvdotii" lib/Transforms/` → **0** — the core has no IME-family token at all.
- `grep -rnE "contains\(.*matmul.*\).*contains\(.*spacemit" lib/ | grep -v /IME/` → **0** — the forbidden
  `if(name.contains("matmul") && march.contains("spacemit"))` anti-pattern exists nowhere.
- The ONLY mnemonic comparison is `IMEBackendEmissionDriver.cpp:383 getImeOp()=="vmadotu"` — **inside the plugin,
  reading the plugin's OWN derived `ime_op` fact** (capability consumption, not a core dispatch branch).
- Selection is `lookupProviderByID("spacemit.ime")->isAvailable()` — the id defined ONCE (`IMEExtensionPlugin.cpp:25`),
  DERIVED by `deriveIMEMatmulCapability:147` from the march ISA envelope (`xsmtvdotii` + VLEN/SEW ⇒ a 4×4×8 MAC
  capability fact). The negative lit `ime-mma-capability-absent-negative.mlir` exercises the declining arm (no IME
  fact ⇒ IME variant not proposed).

## Proof 4 — it runs on real silicon (not just emits)
K1 (SpacemiT X60, IME1, `taskset -c 0-3`): the 4th op `vmadotsu` cross-built with the SpacemiT GCC15.2 fork,
objdump-confirmed `smt.vmadotsu` (`e210212b`), **bit-exact 16/16** vs a mixed-sign scalar oracle with a 4-way
discriminator (IME == mixed-sign ref AND ≠ pure-signed AND ≠ pure-unsigned), hart-4 SIGILL negative control. The
4th op IS the live rapid-add example — no 5th op is needed to make the point.

## The thesis (and its boundary)
**A family is admissible into this execution layer ⟺ its capability is expressible as a RISC-V capability FACT
AND is consumed with zero core-branch.** IME instantiates it (fact `spacemit.ime` derived from the ISA envelope,
consumed via registry dispatch); the same criterion *excludes* a discrete GPU/TPU (no RISC-V capability fact). This
is the N2 contribution and it is **STRUCTURE only** — rapid-add + reuse + zero-core-branch + bit-exact silicon.

**Honest boundary:** IME *performance over RVV* is N3, NOT N2, and is currently **NULL/unproven** — the prior 5.51×
micro was measured against a forbidden hand-rolled `vwmacc+vredsum` baseline (not ggml's real RVV) and washed to
**0.86× at decode e2e** (memory wall). The honest IME perf work (re-baseline vs ggml's shipped RVV; the
SPACEMIT-ON/OFF-one-toolchain + can't-help-decode control; the prefill-only regime) is the open N3 task — see
`IME-PERF-METHODOLOGY-DESIGN.md`. N2's structural claim does not depend on it.
