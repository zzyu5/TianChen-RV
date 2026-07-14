# G3 四.1 — G2 fusion chain WHOLE-MODEL e2e board measure (rvv/VLEN128, 2026-07-08)

> The isolated cell (`../g2-fuse-rms-norm-mul/board_measured.md`) proved the rms_norm→mul
> fusion's **memory-axis** win on silicon: 1.308× wall, 256.2 MiB DRAM round-trip eliminated
> (100.1% of the 8·n·rows prediction), bit-exact. That was a **kernel-micro** on tensors sized
> >L3 to force the y[] round trip to DRAM. This cell is the **whole-model e2e** test the 裁决
> asked for: does that fusion move the needle on a real llama forward pass (every-token
> attn_norm/ffn_norm)? Raw: `board_raw_tinyllama.txt`, `numeric_amdahl_tinyllama.txt`,
> `amdahl_perf_tinyllama.txt`.

## 0. Pre-registration (stated before the numbers)
- **Registration rule**: a statistically-significant e2e speedup of fusion over no-fusion, per the
  T-N rule (Δ > 2× noise floor AND > 2% economically AND non-overlapping IQR), **phase-split**
  (prefill pp / decode tg), gets registered with wording bound to {phase, board, fusion scope}.
- **Not-significant** → reported honestly as the isolated win + **e2e Amdahl dilution**, with the
  Amdahl decomposition (fused-norm's share of token time). Archival accounting value, no Win claimed.
- **Blocked** → report the exact ggml-routing blocker.
- **Protocol**: preflight(0) toolchain symmetry / cold(warmup-dropped) / paired-interleaved /
  N≥10 median+IQR / T-N floor via passes / restore.

## 1. KEY FINDING that reshapes the A/B (the honest "blocker", = a reframe)
**Upstream ggml (this build, f3e1828) ALREADY fuses RMS_NORM+MUL by default.**
`ggml_cpu_try_fuse_ops` (ggml-cpu.c:3130) matches `{GGML_OP_RMS_NORM, GGML_OP_MUL}` via
`ggml_can_fuse` and calls the single-pass `ggml_compute_forward_rms_norm_mul_fused` (ops.cpp:3757) —
on by default, gated ONLY by the env knob `GGML_CPU_DISABLE_FUSION`. Both the stock
(`llama.cpp-upstream-native`) and our repack tree are at f3e1828 and both `libggml-cpu.so` carry the
knob + the fused symbol.

Consequence: **there is no "unfused stock" build to beat.** The task's framing "fused-build vs stock
(unfused) build" is structurally void — stock is already fused in ggml's own hand-written CPU path.
The honest, symmetric e2e A/B is therefore **fusion ON vs OFF on the SAME binary** (`GGML_CPU_DISABLE_FUSION`
toggle). This measures the *exact transformation our compiler emits*: the isolated cell proved our
emitted fused kernel is **bit-exact** to the store-then-reload two-pass (ndiff=0/max_ulp=0), i.e. it is
arithmetically identical to what ggml's fused kernel produces. Our emitted kernel is thus a **drop-in
equivalent** of ggml's fused kernel — the e2e question "does this fusion help the whole model?" has the
same answer whether the fuser is ours or ggml's, and toggling ggml's own fuser answers it without the
historical e2e-seal integration risk. (We did NOT hot-swap our kernel into ggml: that would only
demonstrate parity with ggml's already-present fusion — [NG-2] parity, not a win — at large integration
cost. The measurement below is strictly more informative.)

## 2. Setup — preflight(0) symmetry is TOTAL by construction
- A = fusion ON (default). B = fusion OFF (`GGML_CPU_DISABLE_FUSION=1`). **Same `llama-bench`, same
  `libggml-cpu.so`, same toolchain** (upstream f3e1828, gcc15.2 full-march L1-SEAL override
  `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_...`). The ONLY difference between the two arms
  is one `getenv()` branch inside the graph executor. No compiler-asymmetry defect is possible.
- Preflight asserts the fusion knob + fused symbol are present in the shared lib (fail-closed).
- Model: `tinyllama-q4_0.gguf` (llama 1B, 22 layers → **45 norms/token**: 22 attn_norm + 22 ffn_norm +
  1 final). Canonical 65-norm confirmation on `llama-2-7b-chat.Q4_0` in §6.
- board_fp = rvv-openEuler-VLEN128 / 64c / gov=performance @2.6GHz / cores 8-15 (taskset) / 8 threads /
  cold = one dropped warmup then measured. main tree + build/ UNTOUCHED, nothing committed, no git stash.
- Paired interleaved (on,off,on,off) over PASSES=2 × REPS=5 = **N=10 samples/arm/phase**.

## 3. Numeric (value-preservation) gate ★
Greedy (`--temp 0`, `--seed 1`) completion via `llama-completion`, fusion ON vs OFF, same prompt:
**byte-for-byte IDENTICAL** (on_sha=off_sha=`c0c1274bd2aa8b01`, 273 B each).
→ ggml's rms_norm→mul fusion is value-preserving e2e — same tokens, same bytes. (Consistent with the
isolated cell's on-hardware ndiff=0/max_ulp=0.) No numerical tax.

## 4. Phase-split e2e A/B (cold, N=10/arm, paired-interleaved) ★
tinyllama-q4_0, pp128 (prefill) + tg32 (decode), tok/s (higher = faster):

| phase | arm | N | median t/s | IQR t/s | IQR% | min | max |
|---|---|--:|--:|--:|--:|--:|--:|
| prefill pp128 | fusion ON  | 10 | 10.4364 | 0.0106 | 0.102% | 10.4295 | 10.4450 |
| prefill pp128 | fusion OFF | 10 | 10.4474 | 0.0263 | 0.252% | 10.4303 | 10.4614 |
| decode  tg32  | fusion ON  | 10 |  9.2829 | 0.0509 | 0.548% |  9.0479 |  9.3338 |
| decode  tg32  | fusion OFF | 10 |  9.3082 | 0.1738 | 1.867% |  9.0043 |  9.3360 |

- **prefill**: speedup(on/off) = 0.9989 (**−0.11%**); floor 0.25%; IQR bands overlap → **NOT SIGNIFICANT**.
- **decode**:  speedup(on/off) = 0.9973 (**−0.27%**); floor 1.87%; IQR bands overlap → **NOT SIGNIFICANT**.
- Both deltas are sub-noise and slightly *negative* (fusion marginally slower, but within IQR = a
  statistical zero). **The isolated 1.31× memory-axis win does NOT transplant to whole-model e2e.**

## 5. WHY — Amdahl decomposition (the accounting) ★
Decode task-clock profile (whole model, fusion ON, 160 tok, cores 8-15; `amdahl_perf_tinyllama.txt`,
software `task-clock` sampling since `perf_event_paranoid=2` blocks PMU):

| symbol | self% of decode | role |
|---|--:|---|
| `ggml_vec_dot_q4_0_q8_0`            | **90.20%** | q4_0 weight matmul (DRAM-bandwidth bound) |
| `ggml_vec_dot_q6_K_q8_K_vl128`      | 3.31% | output/embedding (q6_K) |
| `gomp_team_barrier_wait_end`        | 3.15% | OpenMP barrier |
| `ggml_compute_forward_mul_mat`      | 0.85% | matmul dispatch |
| `…flash_attn_ext_f16…`              | 0.40% | attention |
| `quantize_row_q8_0`                 | 0.14% | **activation quantization (FMT-PROP target)** |
| **`ggml_compute_forward_rms_norm_mul_fused`** | **0.05%** | **the fused norm — the thing this cell is about** |

- The fused rms_norm→mul is **0.05% of decode time**. Its **Amdahl ceiling** — the e2e speedup if the
  fusion made the norm *infinitely fast* — is 1/(1−0.0005) = **1.0005× (+0.05%)**. That ceiling is
  **below the measurement noise floor** (0.25–1.87%), so an e2e null is *structurally guaranteed*, not a
  measurement failure. The measured −0.11%/−0.27% are noise around a true effect of ≈0.
- The fusion doesn't even collect that 0.05%: e2e the intermediate y[] is **cache-resident** — per norm
  it is 4·n_embd = 8 KiB (decode, one token) or ~1 MiB (prefill pp128), both < L2/L3 — so it never makes
  the DRAM round trip that the isolated micro (128 MiB tensors, >L3) deliberately forced. The 1.31× lived
  entirely on real DRAM traffic that simply does not exist at whole-model norm sizes.
- Even the **combined** norm-fusion + activation-quantize (FMT-PROP) e2e ceiling is 0.05%+0.14% ≈ **0.19%**
  — still sub-noise. The whole model is **q4_0-matmul-bound (90%)**; elementwise epilogues are not the axis.

## 6. Canonical 65-norm confirmation (llama-2-7b-chat.Q4_0) ★
7B = 32 layers → **65 norms/token**, exactly the task's "~65 norms/token". Decode-focused ON/OFF
(pp32+tg32, N=6/arm = REPS=3×PASSES=2 — reduced N because 7B decode ≈1.37 t/s makes N=10 ~40 min;
this is a corroborating confirmation, the N=10 primary is tinyllama §4). `board_raw_llama7b.txt`:

| phase | arm | N | median t/s | IQR% | speedup(on/off) |
|---|---|--:|--:|--:|--:|
| prefill pp32 | ON | 6 | 1.5549 | 0.028% | — |
| prefill pp32 | OFF | 6 | 1.5539 | 0.113% | **1.0006 (+0.06%)** ns |
| decode tg32 | ON | 6 | 1.3702 | 0.162% | — |
| decode tg32 | OFF | 6 | 1.3686 | 0.099% | **1.0012 (+0.12%)** ns |

- Both sub-noise, IQR bands overlap → **NOT SIGNIFICANT**. The null holds on the canonical 65-norm
  model (deltas here flip marginally positive, +0.06%/+0.12%, but are statistical zeros). As predicted,
  the larger model's even higher matmul:norm ratio dilutes the fusion further — the e2e effect is ≈0
  a fortiori. (7B decode is ~6.8× slower than tinyllama at 1.37 vs 9.3 t/s — the q4_0 weight stream is
  6× larger, norm work grows only with n_embd, so norm's token-time share shrinks vs tinyllama's 0.05%.)

## 7. Verdict — NOT-SIGNIFICANT e2e (pre-registered "not-significant" branch)
- **Integration**: succeeded, but via the honest path — stock ggml already carries the fusion, so the
  e2e A/B is its ON/OFF toggle (toolchain-symmetric by construction), not a fused-vs-unfused-stock beat.
- **prefill Δ = −0.11% (ns), decode Δ = −0.27% (ns)** — both sub-noise; fusion e2e effect ≈ 0.
- **logit/greedy check**: byte-identical ON vs OFF (value-preserving).
- **Amdahl**: fused-norm = 0.05% of decode; q4_0 matmul = 90.2%; ceiling +0.05% < noise floor. Isolated
  1.31× is a memory-axis kernel fact on >L3 tensors; at whole-model norm sizes the intermediate is
  cache-resident so the benefit vanishes. This is the "kernel wins don't transplant to memory-bound e2e"
  pattern, quantified.
- **Framing** ([NG-4]/[NG-2]): the isolated per-pair silicon win stands (that cell owns it); this cell
  owns the honest e2e accounting — the fusion is real and byte-exact but Amdahl-diluted below noise on a
  q4_0-matmul-bound model. No Win-B/Win-C, no e2e speedup claimed. 立项 to roll fusion out further e2e is
  **not** on evidence (the axis is the matmul, not the norm).
