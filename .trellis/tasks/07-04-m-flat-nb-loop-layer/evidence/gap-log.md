# M-FLAT nb-loop-layer — GAP evidence log (机制章素材)

The tracer-bullet target IR (`../target-ir/q8_0-nb-loop-target.mlir`) driven
parse → verify → lower → byte-exact. Each wall = one GAP-1 entry: 命名 → 拆法 →
复测 → 归因. This is the load-domain main shape (vec_dot 24/92 keys; GEMM tile
recurs the same shape) whose LOOP layer the single-block-all-vector dialect never
grew.

---

## GAP-1 / W2 — 逐块寻址 load (per-block addressing)  [TORN 2026-07-04]

**命名.** The typed integer-core load must address a per-block strip at
`base + block_index*block_stride (+ quant_byte_offset)`. The dialect's
`tcrv_rvv.load` only addressed a single ABI-base buffer (`$buffer, $vl`) — the
per-block form the nb loop needs was never built. Surfaced at the FIRST parse of
the target IR (`error: expected ','` on `load %vx block %ib`).

**拆法 (minimal, mirrors brick① step-3, no preventive generalization).**
- ODS (`RVVOps.td` `def LoadOp`): + `Optional<Index>:$block_index`,
  `OptionalAttr<I64Attr>:$block_stride`, `OptionalAttr<I64Attr>:$quant_byte_offset`;
  assembly `(block $block_index^ : type)?` after the regular operands. Absent
  block_index = byte-identical single-block form.
- `isAllowedLoadAttr` (`RVVDialect.cpp`): allow exactly `block_stride` +
  `quant_byte_offset` (all other load attrs stay disallowed).
- `LoadOp::verify` (`RVVDialectMemoryOps.cpp`): operand count 2|3 by block_index;
  block_index present ⟺ block_stride present, nonzero (fail-closed); absent form
  must carry neither (fail-closed).
- i8m2 recognition (W1-adjacent, same turn): the single-block widening-source
  fast-paths require the whole load→product→reduce chain to be DIRECT children of
  the `with_vl`; in the loop body their parent is the loop op, so they are skipped
  and the load falls to the generic-route type check (rejects SEW8). Minimal fix:
  when `block_index` present, accept exactly `isGenericRVVVectorSignedI8M2` (the
  q8_0 core's load type) — loop-body allowlist (step 4) + byte-exact lowering (W4)
  guard chain integrity.

**复测.** target IR advances parse→verify CLEAN (exit 0); the load's 3 operands +
block_index induction var + attrs accepted (`load(%buffer,%vl,%ib)<{block_stride,
quant_byte_offset}>`). New per-block-load unit lit: TODO (deferred into W4 — the
byte-exact load emit is produced by the walker, not yet wired). Zero regression:
all 352 Conversion/RVV + Target/RVV green (existing single-block load routes,
3 strong routes, step1-4, nesting-relaxed 352 unchanged).

**归因.** The dialect was built for single-block, all-vector strong routes whose
loads address one ABI base and whose whole dataflow chain are direct children of a
single `with_vl`. The nb loop's per-block addressing + a loop op sitting between
the chain and its vl scope are structurally new — a missing capability, not a bug.

**⚠ still helper-driven.** After W2 the target IR LOWERS (exit 0) but the emit is
produced by `emitFlatBlockCore` firing off the loop op — it computes its OWN
`block_base_x` + `vle8_v_i8m2` and ERASES the typed chain (incl. the new per-block
load). W4 (lowering-walks-chain) is the next + load-bearing wall: it decides
chain-driven (genuine) vs helper-driven (cosmetic). The per-block load capability
is built but not yet wired into the emit.

---

## GAP-1 / W4 — lowering 走链 (chain-driven / op-by-op)  [TORN 2026-07-04]

**命名.** After W2 the loop body LOWERED but the emit was `emitFlatBlockCore(descriptor)`
gated on the chain — attribute-derived, region ops only *checked*. A gate and a
lowering fail delete-mutation IDENTICALLY (advisor + verify agent), and "byte-exact
vs the monolith emit" is CIRCULAR (matched because it *was* the monolith emitter).
Neither proved construction.

**拆法 (op-by-op, operand-driven).** Rewrote the brick2 branch of
`emitTypedFlatBlockDotLoopBody` (`RVVToEmitCBlockQuantLinear.cpp` ~:6154-6477) to
hand-emit each region op FROM ITS OPERANDS — no `emitFlatBlockCore` call:
per-block LoadOp -> addr `base+ib*block_stride+quant_byte_offset` from its own
operands + vle8; WideningProductOp -> `vwmul(valueMap[lhs],valueMap[rhs])`;
StandaloneReduceOp -> vwredsum(input); extract -> vmv_x_s -> sumi; brick2/3 ->
fused fold (sumi from extract, d_x/d_y from brick1). block_base memoized on
(buffer,block_index) so brick1+load share one emission.

**复测 (operand-flow, not delete-mutation, not byte-exact-vs-monolith).**
M1 (isolating): swap widening_product operands -> emitted `vwmul(%27,%32)`->`(%32,%27)`
(main agent re-confirmed). M4 (kills attr-derivation): attr-only stride 34->68 with
load operand at 34 -> EXIT 1 fail-closed (a shell would emit 68). M3 (load-only)
also fail-closed. M2: stride/quant flow into address literals, per-load independent.
Zero regression: 352 green.

**归因.** No path WALKED a typed chain in a loop-body region sourcing emit from the
region SSA. That walk is the loop layer that was never built — W2's missing
capability one level up (emission, not typing).

**⚠ two disclosed residues (structural constants, do NOT undermine the arithmetic
chain):** (1) reduce SEED hardcodes literal-0; `%zero_seed` operand not read —
semantically forced (fresh per-block 0), a MODELING wart (op needs a
runtime_abi_value seed, flat loop's seed is literal 0); pre-flip cleanup candidate.
(2) inner block-cap vl = `qk` literal (scheduling constant). byte-exact-vs-monolith
DIVERGES only on setvl structure + provenance; per-block ARITHMETIC core is
byte-identical. Numerical bit-exact vs scalar oracle = pending-hardware.

**Stage: 靶 IR parse ✓ verify ✓ lower ✓ (op-by-op genuine).** Remaining: 拓真实 nb
(multi_block_factor) → q8_0 首翻 constructed (+ resolve seed residue) → 五格 cohort
→ 删 monolith.
