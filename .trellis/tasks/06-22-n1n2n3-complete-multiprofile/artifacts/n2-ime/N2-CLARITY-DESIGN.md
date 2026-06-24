# N2/IME clarity — design + verdict (2026-06-24)

**VERDICT: N2 needs PRESENTATION, not new structural work.** The rapid-add is already strongly proven
(core-LOC≈0 three ways; the complementary-arm negative test; TWO prior incremental ops — vmadotu `c7069111`,
tiled matmul `0b8c6168` — added with ZERO core/wiring lines = "wiring paid once" demonstrated, not asserted).
The user's thrice-flagged "不像成熟compiler / 别字符串匹配 / 发挥n2抽象" is a LEGIBILITY/optics complaint: the
proof is buried in `OPTIMIZATION-EVIDENCE-MATRIX.md §3`. 

## Part 1 (PRIMARY) — `N2-RAPID-ADD-SHOWCASE.md`, a 5-min standalone artifact
§0 thesis (0 core lines; core never learns what "IME" is). §1 the abstraction CONTRACT (the plugin vtable
hooks the generic core calls: registerDialects / supportsOperation / collectVariantProposals /
verifyVariantLegality / estimateVariantCost / buildVariantEmissionPlan / materializeSelectedLoweringBoundary
+ a BackendEmissionRegistry driver — the mature-compiler shape, cf LLVM TargetMachine registry; core iterates
the registry, never branches on family identity). §2 the 6 reused interfaces (file:line anchors, modified=0).
§3 capability-FACT dispatch NOT string-match: `lookupProviderByID("spacemit.ime")->isAvailable()`, the id
defined ONCE (IMEExtensionPlugin.cpp:25), derived by `deriveIMEMatmulCapability` (:137) from `xsmtvdotii`
march evidence ⇒ 4×4×8 MAC; the anti-pattern `if(name.contains("matmul")&&march.contains("spacemit"))` proven
absent (grep 0 family tokens in lib/Transforms/*); the negative test ime-mma-capability-absent-negative.mlir.
§4 LOC quantification table. §5 one-line story.

## Part 2 (OPTIONAL, live worked example) — 4th op `tcrv.ime.mma_su` (vmadotsu, int8 signed × uint8 unsigned → int32)
Tractable: ~75-90 plugin LOC, 0 core, 0 wiring, HIGH K1 confidence (SpacemiT GCC15.2 already accepts all 4
mnemonics under xsmtvdotii). Canonical quantized mixed-sign case (signed activations × unsigned weights) —
meaningful, not cosmetic. Distinct encoding `0xe210232b`. Edits (all sites verified):
- `IMEOps.td`: clone `MMAUOp` → `MMASUOp` (~50 lines, same attr schema).
- `IMEDialect.cpp`: `kExpectedMixedSignIMEOp("vmadotsu")` + `MMASUOp::verify()` calling the
  already-mnemonic-generic `verifyIMEMACBoundary` (~8 lines).
- `IMEBackendEmissionDriver.cpp`: `macHelperBody(name,"vmadotsu")` + `IMEMACToEmitCFunc<MMASUOp>` specialization
  + add to patterns/addIllegalOp lists (~15 lines — helper already (name,mnemonic)-parameterized).
- `IMEExtensionPlugin.cpp`: extend signedness derivation {signed,unsigned} → +"signed_unsigned" ⇒ vmadotsu
  (~12 lines in deriveIMEMatmulCapability + a variant-name constant). Rides the SAME spacemit.ime fact.
- **Rotate the negative control:** vmadotsu is CURRENTLY the negative control in mma.mlir:62 → swap to a
  still-rejected sibling `vmadotus`. Narrative beat: the test architecture pre-figured the signedness family.
- Lits: positive mma-su.mlir round-trip + rotated negative + ime-mma-su-materialization (no other family leak).
**Defuse the string-swap objection (MANDATORY in the showcase):** selected by a derived signedness FACT
(`ime_signedness ⇒ vmadotsu`) flowed as DATA through the variant attribute, read at boundary time — NO
`if(name=="vmadotsu")` in the core; the core never sees the mnemonic. The cheapness IS the thesis.
**K1 verify plan:** emit → SpacemiT GCC15.2 cross-build (`-march=…xsmtvdotii`) → objdump `smt.vmadotsu`
(e210232b) → on X60 taskset -c 0-3, high-bit data, mixed-sign scalar oracle; require IME == mixed-sign ref
AND ≠ pure-signed AND ≠ pure-unsigned (the discriminator).
**Do NOT pursue int4/int16/fp32 live** (assembler/fragment/silicon unverified in read-only) — future breadth.
