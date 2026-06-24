# Option-2 STAGE C1b — the ISOLATED bit-exact PACKER (DESIGN, 2026-06-24)

READ-ONLY design. NO `lib/` edits. Scopes the **materialization-side half of C1**: proving the
compiler can DRIVE the plain→x16 weight materialization (the OLD doc's **mechanism-(a)**: a
compiler-emitted pack kernel), byte-identical to ggml's `make_block_q4_0x16`.

This is the SIBLING of C1's in-IR bridge (the bridge half — `Repack`-branch →
`create<GgmlRepackGemvQ40Q80Op>` carrying `weight_layout_contract="x16"` — is being built now;
`option2-stageC-revised-layout-contract-DESIGN.md` §3.3). C1b is the OTHER half of C1: the
COMPILER-EMITTED PACKER that produces the x16 bytes the bridge's op declares it consumes.

## 0. HONEST SCOPE (carry it EXACTLY — state it in every report)

- **This is mechanism-(a):** a COMPILER-EMITTED packer C function (the OLD doc's "compiler-driven
  materialization"). It is the **capability proof that the compiler CAN materialize x16**.
- **It is e2e-REDUNDANT.** ggml ALREADY packs at model-load (`repack_q4_0_to_q4_0_16_bl` → `make_block_q4_0x16`,
  via `init_tensor`/`set_tensor`→`repack()`). The compiler-emitted packer is NOT on any e2e data
  path; swapping it in changes ZERO e2e numbers. The §4.2 JIT/per-call repacker (which COULD reuse
  this same emitted transform) is the only place it could ever run live — and per the memory wall it
  amortizes ONLY for prefill (M≫1), NEVER for decode (M=1).
- **The value is ISOLATED, not e2e:** it completes the **layout-as-input/declared-contract** story at
  the isolated level — the compiler doesn't merely CONSUME a pre-packed layout (every existing
  `Ggml*RepackGemv*` emitter does that), it can also PRODUCE the layout it declares. "The compiler
  drives the materialization" becomes a demonstrated capability, not a claim. Report it as
  **bit-exact isolated proof, NOT a kernel/e2e win** (MEMORY kernel-wins-dont-transplant discipline).

---

## 1. THE PACK TRANSFORM SPEC (the EXACT bytes the emitted packer must reproduce)

Source of truth: `make_block_q4_0x16(in, blck_size_interleave)` at
`llama.cpp/ggml/src/ggml-cpu/repack.cpp:2811`, called with **`interleave_block = 1`** from
`repack_q4_0_to_q4_0_16_bl(t, 1, …)` (`repack.cpp:3939`). So the LIVE path is the
`blck_size_interleave == 1` branch (the `else` `GGML_ASSERT(false)` is dead for q4_0x16).

**Input layout — `block_q4_0`, stride 18** (`ggml-common.h:184`, `QK4_0=32`):
```
struct block_q4_0 { ggml_half d;          // fp16 scale       @ +0   (2 bytes)
                    uint8_t   qs[16]; };   // 32 nibbles       @ +2  (16 bytes)  => 18 bytes
```
The packer is handed **16 consecutive source rows** (one block from each of 16 columns / interleaved
rows), gathered by `repack_q4_0_to_q4_0_16_bl`'s inner loop `dst_tmp[i] = src[x + i*nblocks]` into
`block_q4_0 in[16]`. (The row-gather is the CALLER's; the packer transform itself is the per-16-block
`make_block_q4_0x16`. The isolated harness reproduces both — see §3.)

**Output layout — `block_q4_0x16` = `block<4,16>`, stride 288**
(`repack.h:38`/`:24`, `static_assert` at `repack.h:32`):
```
struct block_q4_0x16 { ggml_half d[16];     // 16 fp16 scales  @ +0    (32 bytes)
                       int8_t    qs[256]; }; // (QK4_0*16*4)/8  @ +32   (256 bytes) => 288 bytes
```

**The exact transform (verbatim from the `interleave_block==1` branch):**
1. **Scales (copy, no transform):** `out.d[i] = in[i].d` for `i in [0,16)`. 16 fp16 verbatim.
2. **Quants (16-way interleave + `^0x88` bake):** `end = QK4_0*8/1 = 256`; `xor_mask = 0x88`; for
   `i in [0,256)`:
   ```
   src_id     = i % 16          // which of the 16 source blocks
   src_offset = i / 16          // which nibble-byte (0..15) within that block
   out.qs[i]  = in[src_id].qs[src_offset] ^ 0x88
   ```
   i.e. `out.qs[i] = in[i%16].qs[i/16] ^ 0x88`. The interleave is **block-major-within-byte**:
   consecutive output bytes walk the 16 blocks at a fixed source offset, then advance the offset.
   The `^0x88` flips BOTH nibbles' sign bit (offset-binary → the bias the consumer GEMV already
   expects; `RVVToEmitCBlockQuantLinear.cpp:2800` "*repacked nibbles already carry the ^0x88 offset-binary bias*").

**These three facts (stride-18 in / stride-288 out / `out.qs[i]=in[i%16].qs[i/16]^0x88` + scales-verbatim)
are the entire contract the emitted packer must reproduce byte-for-byte.** No fp arithmetic, no
quantization, no vector reduction — it is a pure byte gather + XOR. (This is why C1b is SMALL.)

---

## 2. THE EMIT VEHICLE — a NEW emitter method (modeled on the consumer-emitter template, but simpler)

**Finding: there is NO existing pack/transform emitter to reuse.** `grep` over
`lib/Conversion/RVV/` shows every `0x88`/`interleave`/`materializ` hit is a CONSUMER comment in
`RVVToEmitCBlockQuantLinear.cpp` (`:2200/:2280/:2800` etc.) — emitters that READ pre-x16 bytes. Every
`Ggml*` op in `RVVOps.td` (BlockDot / RepackGemv / RepackGemm, lines 3873–6620) is a CONSUMER; none
emits the pack transform. **The pack emitter is genuinely NEW** (but small — see §5).

**Vehicle: same op-driven emit pipeline as the GEMV emitters, with a NEW scalar-transform op.** The
existing pipeline (proven by the archived `emit-repack-gemv/` artifact) is:

```
input.mlir (tcrv.exec.kernel w/ ONE dialect op)
   │  tcrv-opt --tcrv-rvv-lower-to-emitc        (RVVLowerToEmitCPass, RVVToEmitC.cpp:5351)
   ▼
emitted.emitc.mlir (emitc dialect)
   │  mlir-translate --mlir-to-cpp
   ▼
emitted-packer.cpp  (standalone C function)
   │  thin adapter + host g++   (NO hardware)
   ▼
verify  (memcmp vs inlined make_block_q4_0x16)
```

**The new op (model on `GgmlRepackGemvQ40Q80Op`, `RVVOps.td:4339`, but STRIP the vector machinery):**
```
def GgmlPackQ40ToX16Op : TCRVRVV_Op<"pack_q4_0_to_q4_0x16", [...]> {
  // operands: %src (const uint8_t*), %dst (uint8_t*), %nblocks (index)   — NO vl, NO setvl, NO LMUL
  // attrs:   weight_block_stride=288, src_block_stride=18, weight_interleave=16,
  //          weight_quant_byte_offset=32, src_quant_byte_offset=2, qk=32, xor_mask=0x88
}
```
Emitted signature target:
```c
void tcrv_pack_q4_0_to_q4_0x16(const block_q4_0 *src, block_q4_0x16 *dst, int nblocks);
```

**The emit method (model on `VariantToEmitCFunc::emitRepackGemvQ4_0Q8_0`, dispatched at
`RVVToEmitC.cpp:347` via the `{isRepackGemvQ4_0Q8_0Body → emitRepackGemvQ4_0Q8_0}` table).** Add a
parallel pair:
```
{ &VariantToEmitCFunc::isPackQ4_0ToX16Body,  &VariantToEmitCFunc::emitPackQ4_0ToX16 }   // RVVToEmitC.cpp:347 table
bool isPackQ4_0ToX16Body(WithVLOp)  → isa<GgmlPackQ40ToX16Op>(op)    (mirror :1089/:1092)
void emitPackQ4_0ToX16(...)         → the scalar gather + XOR body
```
The body is **emitc scalar ops only** (`emitc::VerbatimOp` for the loop skeleton / `^0x88` /
`memcpy` of the 16 scales, or `emitc::ForOp`+`AddOp`/`MulOp`/`CallOpaqueOp`), NO `setvl`/`with_vl`/
LMUL/`vwmacc`/`vredsum`. It is a STRICT SUBSET of the GEMV emitter — no widening, no scale-fold, no
accumulator. The verifier (model on `RVVDialectWideningOps.cpp:1718`) just pins
288/18/16/32/2 — fail-closed, trivial.

**Cheaper alternative (note, not the recommendation):** a pure `emitc::VerbatimOp`-only emit (one
verbatim block with the literal scalar loop) would skip the structured-emitc lowering. The op-driven
version is preferred because it keeps the layout facts (stride/offset/interleave/xor) as **op attrs**
— the same provenance the rest of the dialect carries — and is the honest "compiler drives it" form
(the transform is DERIVED from attrs, not a frozen string). Pick op-driven unless effort forces the
verbatim shortcut.

---

## 3. BIT-EXACT VERIFICATION PLAN — HOST, no hardware

**Host-side, no `ssh rvv`.** The pack transform is a pure byte gather + XOR (no vector compute, no
RVV intrinsics in the emitted packer). So the emitted C compiles + runs on the dev host with plain
`g++` — exactly like the byte-fingerprint gates. **Hardware is NOT needed and would add nothing**
(it's a memcmp on bytes, identical on any ISA).

**The oracle = ggml's own `make_block_q4_0x16`, inlined.** The canonical harness
`…/archive/2026-06/06-22-n1n2n3-establish-rvv-mature/artifacts/emit-repack-gemv/verify_emitted_gemv.cpp`
ALREADY contains the verbatim reference: `make_block()` (`:63`, the inlined
`interleave_block==1` `^0x88` path), `make_random_q4_0_row()` (`:108`), and the struct defs
(`block_q4_0`/`block_q4_0x16`, `:54/:56`). **Copy-then-adapt** that file into a new
`verify_emitted_packer.cpp`:

1. **Random-weight oracle.** For each trial: `make_random_q4_0_row` → `NC` rows × `nb` blocks of
   random plain q4_0 (random fp16 scale + random nibbles). Seed fixed (`std::mt19937(20260622)`),
   sizes `n in {64,256,4096,11008,14336}` × groups `NC in {16,32,64,336}` (the harness's existing
   stress matrix), so cross-group pointer math (`dst += group*288`) is exercised.
2. **Two packers on the SAME bytes.**
   - REFERENCE: the inlined `make_block()` (group-gather `tmp[i]=w[(g*16+i)*nb+x]` then
     `make_block(tmp)`), exactly the C1 harness's `vx[g*nb+x]` construction (`:140–148`).
   - EMITTED: `tcrv_pack_q4_0_to_q4_0x16(src, dst, nblocks)` (via the thin adapter, modeled on
     `emitted_adapter_gemv.cpp`, `#include`-ing the emitted `.cpp`). Feed it the SAME group-gathered
     `block_q4_0[16*nb]` and read back `block_q4_0x16[nb]`.
3. **The memcmp gate (the verdict).** `memcmp(emitted_x16, ref_x16, ngrp*nb*sizeof(block_q4_0x16)) == 0`.
   **EXACT byte equality — not a norm/tolerance** (this is a byte transform; a single differing byte
   FAILS). Report PASS/FAIL per (n, NC) and a single VERDICT, mirroring the GEMV harness's print
   format. (Contrast: the GEMV verify uses a `norm<1e-4` fp gate; the PACKER verify is `memcmp==0`,
   stricter, because no arithmetic intervenes.)

**Two scrutiny points the memcmp catches for free:** (a) the `^0x88` is applied to ALL 256 quant
bytes and to NOTHING else; (b) the 16 fp16 scales are copied VERBATIM (no XOR, no reorder). A wrong
interleave index, a missed/extra XOR, or a scale-byte corruption all differ → FAIL.

---

## 4. PRECISE ORDERED BUILD STEPS (each compiles; final = byte-exact memcmp gate)

All steps are **LIT/host-only** (no hardware). Use a **forced/clean rebuild** of `tcrv-opt` —
incremental builds are unreliable in this tree (MEMORY build-incremental-unreliable: ODS `.inc`
regen + `tcrv-opt` relink flakiness).

| # | Step | Compiles? | Gate |
|---|---|---|---|
| **S1** | Add `def GgmlPackQ40ToX16Op` to `RVVOps.td:4339`-vicinity (model on `GgmlRepackGemvQ40Q80Op`; strip vl/LMUL; attrs 288/18/16/32/2/qk32/xor). | ODS regen + `tcrv-opt` clean build | op parses in a hand-written `input-pack.mlir` (`tcrv-opt` round-trips it) |
| **S2** | Add the trivial verifier (model `RVVDialectWideningOps.cpp:1718`): pin 288/18/16/32/2, fail-closed. | clean build | a malformed-stride `.mlir` fails verification; the good one passes |
| **S3** | Add `isPackQ4_0ToX16Body` (mirror `RVVToEmitC.cpp:1089`) + `emitPackQ4_0ToX16` (scalar gather+XOR body) + register the pair in the `:347` dispatch table. | clean build | — |
| **S4** | Author `input-pack-q4_0.mlir` (a `tcrv.exec.kernel` w/ ONE `tcrv_rvv.pack_q4_0_to_q4_0x16` op; model `input-repack-gemv.mlir`). Run `tcrv-opt --tcrv-rvv-lower-to-emitc`. | `tcrv-opt` run | emits valid `emitted.emitc.mlir`; `lower.err` empty |
| **S5** | `mlir-translate --mlir-to-cpp emitted.emitc.mlir > emitted-packer.cpp`. | translate run | `tcrv_pack_q4_0_to_q4_0x16(...)` symbol present; `translate.err` empty |
| **S6** | Copy-adapt `verify_emitted_gemv.cpp` → `verify_emitted_packer.cpp`: keep `make_block`/`make_random_q4_0_row`/structs, REPLACE the gemv-vs-gemv_ref loop with packer-vs-`make_block` + **`memcmp`** (§3). Thin adapter `#include`s `emitted-packer.cpp`. `g++ -O2 verify_emitted_packer.cpp -o vpack`. | host g++ | compiles clean |
| **S7** | `./vpack` over the stress matrix. | run | **VERDICT = PASS ⟺ `memcmp==0` on every (n,NC)** — the byte-exact gate |

**New artifact dir:** `…/06-22-n1n2n3-complete-multiprofile/artifacts/kernel-coverage/pack-q4_0-x16/`
holding `input-pack-q4_0.mlir`, `emitted.emitc.mlir`, `emitted-packer.cpp`,
`verify_emitted_packer.cpp`, `pack-verify.log`. (Decline/other cells stay byte-identical `f810ce6b`;
this adds a NEW op + emit path, so re-run the global emitter-inertness fingerprint gate to confirm
the addition didn't perturb existing emitters.)

---

## 5. EFFORT, RISK, BLOCKERS

- **Effort: ~½–1 session.** Smaller than a consumer-kernel emitter. The TRANSFORM is trivial (byte
  gather + XOR, no fp, no vector). The cost is the standard new-op overhead: ODS def + verifier +
  dispatch-table wiring + one emit method + clean rebuild — the same scaffold the q4_K stage-1a /
  q8_0 expansions paid (MEMORY kernel-expansion-q8-q4k), but with a SIMPLER body. The harness is
  copy-then-adapt of an EXISTING, proven file — minimal new code.
- **Risk: LOW.** No hardware, no numerical tolerance (exact memcmp = unambiguous pass/fail). The
  transform spec is fully pinned from primary source (§1). The only real failure modes are mechanical:
  (a) wrong interleave index `i%16` vs `i/16` — caught by memcmp; (b) XOR applied to scales or missed
  on quants — caught by memcmp; (c) struct stride/offset mismatch — caught at compile (struct
  static_asserts) or by memcmp.
- **Blockers / watch-items:**
  1. **NEW op, not a reuse** — there is no pack emitter to copy a body from; the emit method is
     written fresh (but it's ~20 lines of scalar emitc, the smallest emit body in the dialect).
  2. **Clean rebuild mandatory** (MEMORY build-incremental-unreliable) — incremental `.inc`/relink
     can silently serve a stale `tcrv-opt`; force a clean build before S4–S7.
  3. **Emitter-inertness re-gate** — adding an op + dispatch entry must NOT change any existing
     emitter's bytes; re-run the byte-fingerprint gate (block-dot `f810ce6b` clean truth) after S3.
  4. **`interleave_block==1` is the ONLY live path** — do NOT emit the `else GGML_ASSERT(false)`
     branch or any multi-byte interleave; q4_0x16 is always `blck_size_interleave==1`.

---

## 6. CRITICAL FILES (file:line)

- `llama.cpp/ggml/src/ggml-cpu/repack.cpp:2811` `make_block_q4_0x16` (the transform; the live
  `interleave_block==1` `^0x88` branch); `:3358/:3939` `repack_q4_0_to_q4_0_16_bl(t, 1, …)` (the
  16-row gather wrapper + the `interleave_block=1` arg).
- `llama.cpp/ggml/src/ggml-cpu/repack.h:24/:38` `block<4,16>` template + `block_q4_0x16` alias
  (`d[16]`@0..32, `qs[256]`@32..288, stride 288); `ggml-common.h:184` `block_q4_0` (stride 18).
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:4339` `GgmlRepackGemvQ40Q80Op` — the op to MODEL the
  new `GgmlPackQ40ToX16Op` on (strip vl/LMUL).
- `lib/Conversion/RVV/RVVToEmitC.cpp:347` the emit dispatch table (`{isBody, emitMethod}` pairs);
  `:1089/:1092` `isRepackGemvQ4_0Q8_0Body` (mirror for `isPackQ4_0ToX16Body`); `:5351`
  `RVVLowerToEmitCPass` (`--tcrv-rvv-lower-to-emitc`).
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:2800` "*repacked nibbles already carry the ^0x88
  bias*" (proves consumers READ pre-x16; none packs) — the gap C1b fills.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:1718` repack-gemv verifier — model for the trivial
  pack verifier.
- `…/archive/2026-06/06-22-n1n2n3-establish-rvv-mature/artifacts/emit-repack-gemv/verify_emitted_gemv.cpp:63`
  inlined `make_block` reference oracle (`:108` random-row gen, `:54/:56` structs); `emitted_adapter_gemv.cpp`
  the thin ABI adapter; `input-repack-gemv.mlir` the op-driven input form — all copy-then-adapt sources.

---

## 7. ONE-LINE FRAMING TO CARRY

C1b proves the compiler can **PRODUCE** the x16 layout it **DECLARES** (mechanism-(a)), bit-exact to
ggml — an ISOLATED capability proof, host-only, memcmp-exact. It is **e2e-REDUNDANT** (ggml packs at
load; this changes zero e2e numbers) and its only live home is the §4.2 prefill-amortizing JIT
repacker. Report it as a bit-exact isolated proof, NEVER as a kernel or e2e win.
