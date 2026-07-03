# Research: Coverage Denominator [COV-1] — pinned ggml + A/B/C roster + in-code reconciliation

- **Query**: Finalize the [COV-1] coverage denominator for E6 — pin a ggml commit, enumerate the A/B/C roster, reconcile against the in-code op universe, propose the roster artifact file.
- **Scope**: internal (repo) + one external anchor (ggml checkout)
- **Snapshot**: repo HEAD `1bbab882695e812a2d334a8d6f5c7860cc2c93b1` (branch `refactor/full-refactor-m1`), 2026-07-03. Prior audit anchors were pinned `7185a62b`/`6e2e4e56`; all counts below re-verified at current HEAD.
- **Authority**: `docs/TianChen-RV_科研目标总纲v2.md` [COV-1] (lines 126–129); `docs/TianChen-RV_执行总纲v2.md` §2/§3 (lines 119–153); `docs/TianChen-RV_实验总纲v1.md` line 32 (denominator key).

---

## 1. The pinned ggml commit (the roster is defined against this)

**There is NO submodule and NO vendored ggml source tree in this repo.** Greps:
- `git submodule status` → empty (no submodules).
- Only vendored `ggml-common.h` copy is under an **archived** task artifact (`.trellis/tasks/archive/2026-06/…/blockdot-bench/ggml-common.h`), not a live pin.
- Every live "ggml commit" mention in `docs/` is the *absence* note itself ([COV-1]/§3 say "无 ggml commit 钉死").

**A sibling working checkout already exists and is already the routing authority:**
- Path: `/home/kingdom/phdworks/llama.cpp`
- HEAD: **`6eab47181cbd3532c88a105682b81b4729ab809b`** (short `6eab471`), branch `master`, subject `wasm : fix fallback symbol collision (#24639)`. **No git tag** (`git describe --tags` → none).
- This exact commit is already cited as "the ggml checkout the routing is defined against" in an archived audit: `.trellis/tasks/archive/2026-07/06-26-table-retest-fill/research/routing-audit.md:6` and `:124`.

### Recommendation (human-ratifiable)
Pin the roster to **ggml `6eab471`** by recording it in the roster artifact's `$meta` block. Reasons: it is already the de-facto routing/baseline checkout, and reusing it keeps roster ↔ baseline consistent (COV-5 measures "对同 commit 上游 ggml").

**Escalation for a human** (do not silently decide):
1. `6eab471` is a **sibling repo, not a submodule, and carries no tag**. The pin is a raw SHA against a checkout that lives outside this repo. A human should ratify (a) reusing `6eab471` vs pinning a fresher/tagged upstream, and (b) whether to vendor a `ggml-common.h` + the format enum at that SHA into the repo so the roster is self-contained (实验总纲 line 28 epoch discipline: "分母清单 = 版本化工件(钉 ggml 版本)").
2. **Epoch discipline** (实验总纲 line 28): upstream adding a new quant format ⇒ open a new *epoch*, annotate a curve breakpoint; every coverage export carries `{repo 快照, ggml 版本, 纪元}`. The roster `$meta` should carry an `epoch` field from day one.

---

## 2. The A/B/C roster (concrete format list)

Denominator key (实验总纲 line 32): **`(算子, 格式[, 形状类])`**, each pair counted **once**; a key's covered-state = the **best six-state across all its variant rows** (prevents multi-path double counting). Denominator = the **full workload roster including not-yet-built kernels** (absent kernels stay in the denominator and pull coverage down — that is what makes `C_dispatch(A)=100%` a real target).

### A类 — 关键路径 (关键: low-bit family is IN the numerator per ruling)

| Roster line | Concrete keys | Count |
|---|---|---|
| Block-quant `vec_dot` — flat plain | q4_0, q4_1, q5_0, q5_1, q8_0, q1_0 | 6 |
| Block-quant `vec_dot` — K-quant super-block | q2_K, q3_K, q4_K, q5_K, q6_K | 5 |
| Block-quant `vec_dot` — codebook / low-bit LUT | iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs | 9 |
| Block-quant `vec_dot` — fp4 | mxfp4, nvfp4 | 2 |
| Block-quant `vec_dot` — ternary | tq1_0, tq2_0 | 2 |
| **vec_dot subtotal** | | **24** |
| `quantize_row` (per A-format) | roster = each A-format's activation quantizer (q8_0, q8_1, q8_K at minimum) | roster-defined |
| `dequantize_row` (per A-format) | roster = each A-format's row dequantizer | roster-defined |
| Decomposed product-reduction family | q4_0-nibble, offset-binary (N=3), codebook (N=3 LUT) | 3 |
| Prefill GEMM tile — RVV (all formats) | roster = RVV GEMM/GEMV per format | roster-defined |
| Prefill GEMM tile — IME | q4_0, q8_0, q4_K (起步) | 3 |

### B类 — 前向算子 (roster fixed by [COV-1] line 128)
rms_norm, softmax, rope, silu, gelu, add, mul, scale, cpy — **9 keys**.

### C类 — 显式暂缓 (防漂移, [COV-1] line 129)
flash-attn tile; bf16 全套 (Zvfbfmin/bf16 per roadmap X-line); plus any B类 op left unbuilt if a human rules it 暂缓 rather than pending (see §3 gaps).

---

## 3. In-code vs roster reconciliation (the gap list) @ HEAD `1bbab882`

**In-code op universe** = 24 block-dot ops (`include/TianChenRV/Plugin/RVV/RVVMonolithicBlockDotFamily.h:1444–1660`, `monolithicBlockDotOpTable()`, 15 `SuperBlock` + 9 `Flat`, re-counted at HEAD) + 3 decomposed shapes + a set of forward-op and GEMM/quant emitters.

### A类 reconciliation

| Roster item | In-code @ HEAD | Gap |
|---|---|---|
| 24 `vec_dot` formats | **All 24 present** (table verified line-by-line) | ✅ none |
| Decomposed product-reduction (3 shapes) | Present (strong/typed; front doors `RVVDequantDotSourceFrontDoor.cpp`, `RVVCodebookDotSourceFrontDoor.cpp`, `RVVReductionSourceFrontDoor.cpp`, `Construction/RVVContractionRouteIdentity.cpp`) | ✅ none |
| `quantize_row` | **q8_0 only** (`GgmlQuantizeRowQ80Op` / `quantize_row_q8_0`) | ⚠ only 1 format in-code; rest of roster = **absent** |
| `dequantize_row` | **sparse** (`dequantize_row_q5_0` referenced; product-reduce-dequantize decomposed demo) | ⚠ most A-formats **absent** |
| RVV prefill GEMM tile "all formats" | repack GEMM/GEMV emitters for **q4_0, q4_1, q4_K, q5_0, q8_0** (`emitRepackGemm{Q4_0Q8_0,Q4_1Q8_1,Q4KQ8K}`, `emitRepackGemv{Q4_0Q8_0,Q4_1Q8_1,Q4KQ8K,Q5_0Q8_0,Q8_0Q8_0}`) | ⚠ 5 formats, **not "all formats"**; K-quant/IQ/fp4/ternary GEMM = absent |
| IME GEMM ×{q4_0, q8_0, q4_K} | IME dialect exposes **`MMAOp`/`MMAUOp`** = *format-agnostic execution ops* ("an IME EXECUTION op, not a high-level matmul/tile op", `include/TianChenRV/Dialect/IME/IR/IMEOps.td:25,35,81`). No format-keyed IME GEMM tile in code. | ⚠ **aspirational** — IME is format-agnostic; the "×{q4_0,q8_0,q4_K}" tiling is a roster goal not an in-code op set |

### B类 reconciliation (evidence-based, all 9 grepped)

| Roster op | In-code dialect op | Six-state (see six-state-current.md) |
|---|---|---|
| rms_norm | `GgmlRmsNormF32Op` | ≥dispatch-wired (emitter `RVVToEmitCForwardElementwise.cpp` + lit `test/Conversion/RVV/rvv-to-emitc-ggml-rms-norm-f32.mlir`) |
| softmax | `GgmlVecSoftMaxF32Op` | ≥dispatch-wired (emitter + lower) |
| rope | `GgmlRopeNormF32Op` | ≥dispatch-wired (emitter + lit `…ggml-rope-norm-f32.mlir`) |
| silu | `GgmlVecSiluF32Op` | ≥dispatch-wired (emitter + lit `…ggml-vec-silu-f32.mlir`) |
| scale | `GgmlVecScaleF32Op` | ≥dispatch-wired (emitter + lit `…ggml-vec-scale-f32.mlir`) |
| gelu | **absent** (0 files) | absent |
| add | **absent** (no dedicated `Ggml…Op`; word "add" is incidental in 61 files) | absent |
| mul | **absent** (no dedicated op) | absent |
| cpy | **absent** (0 files) | absent |

**B类 gap**: 5 of 9 present and dispatch-wired (rms_norm, softmax, rope, silu, scale); **gelu, add, mul, cpy absent**.

### C类 reconciliation
- flash-attn tile: **absent** (`flash_attn`/`flashattn` grep = 0 in `lib/`/`include/`) — confirmed 暂缓.
- bf16: no dedicated bf16 forward/vec_dot ops (roadmap X-line Zvfbfmin/bf16, v2) — confirmed 暂缓.

---

## 4. The load-bearing denominator-key call (endorse (a); ratify by a human)

The roster lists **"分解产品-归约家族" as a distinct A类 line item**, separate from the block-dot `vec_dot` formats. That structural fact is the tie-breaker:

- **(a) DEFAULT-OF-RECORD — separate keys.** The 3 decomposed shapes are their **own** denominator keys (op = product-reduce route; distinct `算子` from `vec_dot`). Consequence:
  - The q4_0 *format* appears under **two different roster keys**: the `vec_dot` q4_0 key (state = constructed-weak) **and** the decomposed product-reduce q4_0-nibble key (state = constructed/strong). This is *not* one key whose best-state upgrades — they are different ops, so 实验总纲 line 32's "best six-state across variant rows" does **not** merge them. (This is the exact resolution of the 执行总纲 §2 "q4_0 双路" note.)
  - **Numeric consequence**: block-dot `vec_dot` `C_construct` (strong) = **0/24 today** (all 24 weak/dispatch-wired); decomposed family = **3/3 strong in its own bucket**. G1 (engine-axis burn-down) burns the 24 `vec_dot` keys toward strong. The 24-key block-dot bucket stays the clean burn-down denominator.
- **(b) rejected — merge/upgrade.** If the decomposed strong shape upgraded the same-format `vec_dot` key, q4_0 (and every dual-path format) would silently flip to strong and **pollute the G1 burn-down denominator**. This is why the call matters and why a human should ratify.

**Ship (a).** Flag (b) as the rejected alternative in the roster's doc note so no future editor merges the buckets.

---

## 5. Proposed roster artifact file (format + location)

**Location**: reuse E1's governance home — **`schema/coverage-roster.v1.json`** (sits next to `schema/capability.schema.v1.json` + `schema/VERSIONLOG.md`; already the committed-artifact + canonical-hash home E1 established). Alternative discussed: a new top-level `coverage/` dir — rejected for MVP because it fragments the governance surface E1 just centralized. (No `coverage/` dir exists today.)

**Format**: canonical JSON (so it can reuse E1's `canonicalize` / SHA256 idiom in `check_schema_gate.py`). Sketch:

```json
{
  "$meta": {
    "roster_version": "v1",
    "ggml_pin": "6eab47181cbd3532c88a105682b81b4729ab809b",
    "ggml_ref_note": "sibling checkout /home/kingdom/phdworks/llama.cpp @ master, no tag",
    "epoch": 1,
    "denominator_key": "(op, format[, shape_class])",
    "decomposed_family_policy": "SEPARATE-KEYS (option a); vec_dot and product-reduce are distinct ops"
  },
  "kernels": [
    {"op": "vec_dot", "format": "q4_0", "class": "A", "bucket": "flat-plain"},
    {"op": "vec_dot", "format": "q4_K", "class": "A", "bucket": "superblock"},
    {"op": "product_reduce", "format": "q4_0_nibble", "class": "A", "bucket": "decomposed"},
    {"op": "quantize_row", "format": "q8_0", "class": "A"},
    {"op": "gemm_tile", "format": "q4_0", "engine": "rvv", "class": "A"},
    {"op": "gemm_tile", "format": "q4_0", "engine": "ime", "class": "A"},
    {"op": "rms_norm", "format": "f32", "class": "B"},
    {"op": "gelu", "format": "f32", "class": "B"},
    {"op": "flash_attn", "format": "tile", "class": "C", "reason": "explicit defer"}
  ]
}
```

Six-state per key lives in a **separate** hand-labeled table (see six-state-current.md), not in the roster — the roster is the stable denominator; the six-state is the time-varying numerator input.

## Caveats / judgment calls for a human (each changes the denominator)

1. **ggml pin** — `6eab471` (sibling, untagged) vs a tagged upstream; whether to vendor the format enum. (§1)
2. **Decomposed-family key policy** — endorse (a) separate keys; ratify. (§4)
3. **Which low-bit formats are truly A vs C** — the ruling moves low-bit *into* the numerator, so all IQ/ternary/fp4 `vec_dot` are A. But **q1_0** (1-bit, in-code, its own monolith) and **nvfp4** are edge cases: are they A (in-numerator) or C (暂缓)? Currently both are in the 24-op in-code set → default A.
4. **gelu/add/mul/cpy** — absent in-code B类 ops: are they **B-pending** (in denominator, state=absent, pull down C_dispatch) or **C-暂缓** (excluded)? This directly sets the B类 denominator size (9 vs 5) and thus the global C_dispatch target.
5. **quantize_row/dequantize_row & GEMM-tile roster width** — the roster must decide the *per-format* enumeration (all A-formats? only the activation quantizers q8_0/q8_1/q8_K?). This is the largest denominator lever after the vec_dot 24.
