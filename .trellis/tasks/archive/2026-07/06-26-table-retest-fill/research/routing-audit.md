# Research: ggml kernel-dispatch routing audit (baseline authority for table re-test)

- **Query**: For each quant in `doc/KERNEL-优化自查表.md` tables 1–2, which ggml kernel does ggml ACTUALLY dispatch on rvv (SG2044, VLEN128) and k1 (X60, VLEN256), for decode (GEVM, M=1) and prefill (GEMM)? block-dot vs repack? where does ggml ship NO kernel?
- **Scope**: external (ggml/llama.cpp source `/home/kingdom/phdworks/llama.cpp`)
- **Date**: 2026-06-26
- **ggml checkout**: `/home/kingdom/phdworks/llama.cpp` @ `6eab471` (branch `master`)

> **This file is the baseline authority.** A wrong baseline silently invalidates a perf comparison. Every dispatch claim below cites ggml `file:line`. Boards: **rvv** = VLEN128 (`__riscv_vlenb()==16`, `vlenb*8==128`); **k1** = VLEN256 (`__riscv_vlenb()==32`, `vlenb*8==256`).

File shorthands (all under `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/`):
- `disp` = `repack.cpp` — generic dispatch: `ggml_repack_get_optimal_repack_type()` + `forward_mul_mat*` (the gemv/gemm chooser).
- `bd`   = `arch/riscv/quants.c` — RISC-V block-dot kernels (`ggml_vec_dot_*`).
- `rp`   = `arch/riscv/repack.cpp` — RISC-V repack kernels (`ggml_gemv_*_16x1` / `ggml_gemm_*_16x1`).

---

## 0. Preconditions (stated once, apply to every cell below)

**P1 — Repack engages only when a repack *type* is selected.** The CPU-repack buffer's `supports_op` requires `ggml_repack_get_optimal_repack_type(op->src[0]) != nullptr` (`disp:4779`, `disp:4795`). If that returns `nullptr`, the weight is NOT placed in the repack buffer and the matmul falls to the standard per-row **block-dot** path (`ggml_vec_dot_*`).

**P2 — RISC-V repack is `zvfh`-gated and 16-row-gated.** All RISC-V repack instances are inside `#if defined __riscv_zvfh` (`disp:4565`; impls `rp:342/1369/...`). Each riscv `case 256` also requires `cur->ne[1] % 16 == 0`. For real llama weights (rows = multiple of 256) the 16-gate is satisfied; the doc's empirical k1 results (notes 5/16/17) prove `zvfh` is enabled in the k1 build. On rvv the gate is moot — see P3.

**P3 — rvv (VLEN128) ships NO repack for ANY quant.** Every RISC-V `case 128` is `{ break; }` (TODO), falling through to `return nullptr`: q4_0 `disp:4592`, q4_K `disp:4619`, q2_K `disp:4636`, iq4_nl `disp:4680`, q8_0 `disp:4713`. ⇒ **On rvv, block-dot is the only ggml baseline for the entire quant list.** (Confirms the doc's q4_0-VLEN128 claim — and generalizes it to all 5 repack-capable types.)

**P4 — decode vs prefill split inside a repacked tensor.** `forward_mul_mat_one_chunk` chooses by row count: `if (nrows > 3) gemm(...)` else/remainder `gemv(...)` (`disp:4241`, `disp:4246`). So **decode M=1 → `ggml_gemv_*_16x1`; prefill batch ≥ 4 → `ggml_gemm_*_16x1`** (batch 2–3 also → gemv). For non-repacked types both decode and prefill loop the SAME `ggml_vec_dot_*` per output row (no separate prefill kernel).

**P5 — three block-dot symbol-identity classes** (matters for harness grep — see §3):
- **Unified body** (one symbol, runtime `vsetvl`, identical on both boards): q4_0 `bd:222`, q4_1 `bd:277`, q8_0 `bd:435`, q5_K `bd:2081`.
- **One symbol, internal `if (vlenb==16)` branch** (board-divergent code, same symbol): q5_0 `bd:356`, q5_1 `bd:410`.
- **Explicit `_vl128`/`_vl256` symbols** chosen by `switch (__riscv_vlenb()*8)`: all K-quant (except q5_K), all IQ, mxfp4, tq1_0, tq2_0, q1_0.

---

## 1. Master routing table — `quant × {rvv-decode, rvv-prefill, k1-decode, k1-prefill}`

Cell = **dispatched ggml symbol** · **kind** · gate `file:line`. "block-dot" = `ggml_vec_dot_*` per-row. "repack-gemv/gemm" = `ggml_gemv/gemm_*_16x1`.

### 1a. Repack-eligible types (q4_0, q4_K, q2_K, iq4_nl, q8_0) — DIVERGE between boards

On rvv all four cells are block-dot (P3). On k1 the matmul is **repacked → block-dot is BYPASSED in the default/e2e path**.

| quant | rvv-decode | rvv-prefill | k1-decode | k1-prefill |
|---|---|---|---|---|
| **q4_0** | `vec_dot_q4_0_q8_0` block-dot (unified, `bd:222`; no repack `disp:4592`) | same `vec_dot_q4_0_q8_0` block-dot | **`gemv_q4_0_16x1_q8_0`** repack-gemv (`disp:4593`→`rp:206`, P4) | **`gemm_q4_0_16x1_q8_0`** repack-gemm (`disp:4593`→`rp:897`) |
| **q4_K** | `vec_dot_q4_K_q8_K_vl128` block-dot (`disp:4619` break→nullptr; `bd:2070`) | same `..._vl128` block-dot | **`gemv_q4_K_16x1_q8_K`** repack-gemv (`disp:4620`→`rp:260`) | **`gemm_q4_K_16x1_q8_K`** repack-gemm (`disp:4620`→`rp:983`) |
| **q2_K** | `vec_dot_q2_K_q8_K_vl128` block-dot (`disp:4636` break; `bd:950`) | same `..._vl128` block-dot | **`gemv_q2_K_16x1_q8_K`** repack-gemv (`disp:4637`→`rp:497`) | **`gemm_q2_K_16x1_q8_K`** repack-gemm (`disp:4637`→`rp:1406`) |
| **iq4_nl** | `vec_dot_iq4_nl_q8_0_vl128` block-dot (`disp:4680` break; `bd:5596`) | same `..._vl128` block-dot | **`gemv_iq4_nl_16x1_q8_0`** repack-gemv (`disp:4681`→`rp:392`) | **`gemm_iq4_nl_16x1_q8_0`** repack-gemm (`disp:4681`→`rp:1252`) |
| **q8_0** | `vec_dot_q8_0_q8_0` block-dot (unified, `bd:435`; no repack `disp:4713`) | same `vec_dot_q8_0_q8_0` block-dot | **`gemv_q8_0_16x1_q8_0`** repack-gemv (`disp:4714`→`rp:447`) | **`gemm_q8_0_16x1_q8_0`** repack-gemm (`disp:4714`→`rp:1336`) |

> **k1 dual-baseline** (see §2): for these 5, the e2e/default ggml baseline is the **repack** symbol; the micro/algorithm-parity baseline (our block-dot vs ggml's block-dot) is the `_vl256` / unified block-dot symbol, which the harness can still reach but which the e2e path bypasses.

### 1b. Non-repack types — SAME class both boards (block-dot only on rvv AND k1)

These are absent from `ggml_repack_get_optimal_repack_type` (or their `cur->type` branch has no riscv path) ⇒ no repack on EITHER board ⇒ block-dot for decode and prefill alike. decode-kernel == prefill-kernel within a board.

| quant | rvv (decode = prefill) | k1 (decode = prefill) | repack present? |
|---|---|---|---|
| **q6_K** | `vec_dot_q6_K_q8_K_vl128` block-dot (`bd:2726`) | `vec_dot_q6_K_q8_K_vl256` block-dot (`bd:2729`) | none (no riscv branch `disp:4655`) |
| **q3_K** | `vec_dot_q3_K_q8_K_vl128` block-dot (`bd:1613`) | `vec_dot_q3_K_q8_K_vl256` block-dot (`bd:1616`) | none |
| **q5_K** | `vec_dot_q5_K_q8_K` block-dot (UNIFIED, `bd:2081`) | **same symbol** `vec_dot_q5_K_q8_K` block-dot | none (no riscv branch `disp:4644`) |
| **q4_1** | `vec_dot_q4_1_q8_1` block-dot (UNIFIED, `bd:277`) | **same symbol** `vec_dot_q4_1_q8_1` block-dot | **none — absent from dispatch entirely** |
| **q5_0** | `vec_dot_q5_0_q8_0` block-dot (1 symbol, `vlenb==16` path, `bd:356`) | **same symbol**, `else` VLEN256 path (`vslideup`+`vlmul_ext`, `bd:358-361`) | none |
| **q5_1** | `vec_dot_q5_1_q8_1` block-dot (1 symbol, `vlenb==16` path, `bd:410`) | **same symbol**, `else` VLEN256 path (`bd:412-415`) | none |
| **iq1_s** | `vec_dot_iq1_s_q8_K_vl128` block-dot (`bd:3140`) | `vec_dot_iq1_s_q8_K_vl256` block-dot (`bd:3143`) | none |
| **iq1_m** | `vec_dot_iq1_m_q8_K_vl128` block-dot (`bd:3723`) | `vec_dot_iq1_m_q8_K_vl256` block-dot (`bd:3726`) | none |
| **iq3_xxs** | `vec_dot_iq3_xxs_q8_K_vl128` block-dot (`bd:5461`) | `vec_dot_iq3_xxs_q8_K_vl256` block-dot (`bd:5464`) | none |
| **iq3_s** | `vec_dot_iq3_s_q8_K_vl128` block-dot (`bd:5084`) | `vec_dot_iq3_s_q8_K_vl256` block-dot (`bd:5087`) | none |
| **iq2_xxs** | `vec_dot_iq2_xxs_q8_K_vl128` block-dot (`bd:4785`) | `vec_dot_iq2_xxs_q8_K_vl256` block-dot (`bd:4788`) | none |
| **iq2_xs** | `vec_dot_iq2_xs_q8_K_vl128` block-dot (`bd:4508`) | `vec_dot_iq2_xs_q8_K_vl256` block-dot (`bd:4511`) | none |
| **iq2_s** | `vec_dot_iq2_s_q8_K_vl128` block-dot (`bd:4217`) | `vec_dot_iq2_s_q8_K_vl256` block-dot (`bd:4220`) | none |
| **iq4_xs** | `vec_dot_iq4_xs_q8_K_vl128` block-dot (`bd:5954`) | `vec_dot_iq4_xs_q8_K_vl256` block-dot (`bd:5957`) | none (only iq4_**nl** has riscv repack) |
| **mxfp4** | `vec_dot_mxfp4_q8_0_vl128` block-dot (`bd:6587`) | `vec_dot_mxfp4_q8_0_vl256` block-dot (`bd:6590`) | none on riscv (dispatch `cur->type==MXFP4` has only avx2/neon, `disp:4688-4698`; generic mxfp4 repack instances at `disp:4554-4555` are NOT reachable on riscv) |
| **tq2_0** | `vec_dot_tq2_0_q8_K_vl128` block-dot (`bd:6458`) | `vec_dot_tq2_0_q8_K_vl256` block-dot (`bd:6461`) | none |
| **tq1_0** | `vec_dot_tq1_0_q8_K_vl128` block-dot (`bd:6285`) | `vec_dot_tq1_0_q8_K_vl256` block-dot (`bd:6288`) | none |
| **q1_0** | `vec_dot_q1_0_q8_0_vl128` block-dot (`bd:572`) | `vec_dot_q1_0_q8_0_vl256` block-dot (`bd:570`) | none |

Out-of-list completeness (in doc Table 1 but not in the task quant list):
- **nvfp4** — only `ggml_vec_dot_nvfp4_q8_0_generic` exists (generic `quants.c:279`); **NO riscv vector kernel on either board** ⇒ N/A-by-absence-of-vector-baseline (matches doc note 2).

---

## 2. k1 dual-baseline rule (the load-bearing distinction)

For the **5 repack-eligible types on k1** (q4_0, q4_K, q2_K, iq4_nl, q8_0), TWO defensible "ggml baselines" exist and a re-test MUST pick the right one:

| comparison intent | correct ggml baseline on k1 | symbol |
|---|---|---|
| **e2e / default-path** (what llama actually runs) | **repack** | `gemv_*_16x1` (decode) / `gemm_*_16x1` (prefill) |
| **micro / same-algorithm parity** (our block-dot vs ggml block-dot) | **block-dot** `_vl256` (or unified) | `vec_dot_*_q8_*` |

⇒ The doc's existing k1 MICRO numbers that compare against `_vl256` block-dot are baseline-correct *as micro* (e.g. note 14 iq4_nl@k1 1.08× vs `_vl256`; note 10 IQ series vs `_vl256`; doc q2_K@k1 1.02×). They are **NOT** the e2e baseline — in the default k1 path ggml repacks these and the block-dot is bypassed. **Any k1 *e2e* re-test of q4_0 / q4_K / q2_K / iq4_nl / q8_0 must baseline against the repack gemv/gemm, not block-dot.** (Doc note 17 already flags exactly this gap for q4_K: "VLEN256 上 ggml baseline 翻成它自己的 `ggml_gemv_q4_K_16x1_q8_K` repack ... 留作 follow-up"; the same applies to q2_K and iq4_nl, which the doc has NOT yet flagged.)

---

## 3. Harness symbol-grep guidance (so re-test greps the right symbol)

- **Unified-body quants** — grep the bare symbol, there is NO `_vl128`/`_vl256`: `ggml_vec_dot_q4_0_q8_0`, `ggml_vec_dot_q4_1_q8_1`, `ggml_vec_dot_q8_0_q8_0`, `ggml_vec_dot_q5_K_q8_K`. Same machine code path selected by runtime `vsetvl`; board difference is AVL only.
- **q5_0 / q5_1** — single symbol with an internal `if (vlenb==16)` fork (`bd:356`/`bd:410`). Grepping `_vl128`/`_vl256` finds nothing; the board split is inside one function.
- **Everyone else** (K-quant except q5_K, all IQ, mxfp4, tq1_0/tq2_0, q1_0) — grep `_vl128` for rvv, `_vl256` for k1.
- **Repack gemv/gemm** live in `arch/riscv/repack.cpp` (NOT quants.c): `ggml_gemv_*_16x1_q8_*` (decode) and `ggml_gemm_*_16x1_q8_*` (prefill), only for {q4_0, q4_K, q2_K, iq4_nl, q8_0}, only on k1.

---

## 4. Table-2 (repack ops) baseline authority

The doc Table 2 rows are OUR repack kernels. The correct ggml competitor per board:

| our repack op | rvv baseline | k1 baseline |
|---|---|---|
| **q4_0 repack** (decode/prefill) | ggml has **no** q4_0 repack on rvv (P3) → baseline = **block-dot** `vec_dot_q4_0_q8_0`. *(This is why doc's rvv q4_0 repack "2.60×/5.68×" is vs block-dot, not vs a ggml repack.)* | ggml **has** q4_0 repack → baseline = **`gemv/gemm_q4_0_16x1_q8_0`** → comparison is parity-vs-ggml-own-repack (doc note 16: prefill 0.997× TIE; note 5 cites q4_0 decode 0.997×). |
| **q4_1 repack** (decode/prefill) | ggml has **no** q4_1 repack anywhere (absent from dispatch) → baseline = **block-dot** `vec_dot_q4_1_q8_1` both boards. The doc's "上游卡" (note 3) is a *consequence* of this absence — no companion q8_1 activation packer exists because ggml never repacks q4_1. | same — **block-dot** `vec_dot_q4_1_q8_1`; no ggml repack competitor. |
| **q8_0 repack** (decode) | ggml has **no** q8_0 repack on rvv (P3) → baseline = **block-dot** `vec_dot_q8_0_q8_0`. | ggml **has** q8_0 repack → baseline = **`gemv_q8_0_16x1_q8_0`** → parity-vs-ggml-own-repack (doc note 5: 0.99–1.01× TIE; q8_0 decode stays repack-gemv, NOT bypassed). |
| **q4_K repack** (decode/prefill) | ggml has **no** q4_K repack on rvv (P3) → baseline = **block-dot** `vec_dot_q4_K_q8_K_vl128` (doc Win-B 0.55×/0.74× is vs this block-dot). | ggml **has** q4_K repack → stronger baseline = **`gemv/gemm_q4_K_16x1_q8_K`** (the doc's note-17 undone "harder opponent"). |
| **q5_0 repack** (decode) | ggml has **no** q5_0 repack anywhere → baseline = **block-dot** `vec_dot_q5_0_q8_0` (doc note 18 0.821× is vs block-dot, methodologically correct). | same — **block-dot** `vec_dot_q5_0_q8_0` (VLEN256 internal path); no ggml repack competitor. |

---

## 5. Caveats / Not found

- **zvfh build assumption**: §1a k1 repack cells assume the k1 `libggml-cpu.so` is built with `__riscv_zvfh` (gate `disp:4565`). The doc's notes 5/16/17 empirically confirm repack engages on k1, so this holds for their build. If a future k1 build drops zvfh, ALL 5 repack types fall back to block-dot `_vl256` on k1 too — re-confirm with `objdump`/probe before trusting the repack baseline.
- **xtheadvector**: q2_K/q3_K/q4_K/q6_K dispatchers have a `#if defined __riscv_xtheadvector` first branch (`bd:945` etc.) routing to a separate `*_xtheadvector` kernel. This is the K1/Fedora RVV0.7 profile, NOT the VLEN256 `__riscv_v` k1 path used here. The rvv/k1 columns above assume the `__riscv_v` (RVV1.0) build; if a board is compiled `xtheadvector`, the K-quant baseline symbol changes — out of scope for the two RVV1.0 boards but flagged.
- **gemv tail on prefill**: even in the gemm (prefill) path, the row remainder `nrows % 4` is finished with gemv (`disp:4246`). For a large prefill the kernel is overwhelmingly gemm; for batch 2–3 it is ALL gemv. Not a baseline error, but a measurement nuance.
- **Not a routing question (excluded)**: Table 3 forward ops (softmax/silu/…) are not matmul dispatch; Table 4 IME and Table 5 option-2 are out of the rvv/k1 routing scope of this audit.
- All line numbers are against ggml checkout `6eab471`; re-confirm if the sibling `/home/kingdom/phdworks/llama.cpp` is updated.
