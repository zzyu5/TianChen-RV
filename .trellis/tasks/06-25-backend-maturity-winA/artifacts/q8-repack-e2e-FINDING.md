# q8_0 repack GEVM — WIDE-vs-NARROW e2e transduction on k1 (头号 gap / 注5) — FINDING

**Date:** 2026-06-25
**Board:** `ssh k1` (Spacemit X60, RISC-V, **VLEN256**, 8 cores; ISA `…zvfh ime`),
`taskset -c 0-3`, serial, k1-only (rvv untouched). Model: tinyllama-1.1b-chat-v1.0
**Q8_0** (1.17 GB, downloaded host → scp `/data/tinyllama-q8_0.gguf`).
**Tooling discipline:** OUR kernels emitted with the existing host `build/bin/tcrv-opt`
(built 06-25 21:52) — **NO rebuild of our compiler, NO change to TianChenRV `lib/`.**
The `.inc`-swap into ggml + the ggml-cpu rebuild are the prescribed e2e method (same as
the q4_K wf25 finding, which rebuilds "ggml-cpu only").

## STATUS — routing + scope RESOLVED (the load-bearing findings); the tg/pp A/B is
**PENDING a permission gate** (auto-mode classifier blocked *running* the rebuilt
llama-cli). Everything up to and including the WIDE build (BUILD_RC=0) is done; the board
is left staged (pristine backed up). See BLOCKER below.

---

## HEADLINE (what is solidly established without the perf A/B)

1. **ROUTING = FAVORABLE (the q4_K trap does NOT bite here).** At VLEN256 ggml routes
   q8_0 decode through its OWN x16 repack: `repack.cpp:4714`
   `ggml_repack_get_optimal_repack_type`, `GGML_TYPE_Q8_0`, `case 256: if (ne[1]%16==0)
   return &q8_0_16x1_q8_0;` → q8_0 weight tensors are repacked to `block_q8_0x16` at LOAD
   and decode dispatches to `ggml_gemv_q8_0_16x1_q8_0` (arch/riscv `repack.cpp:518`). This
   IS the slot our q8_0 repack GEVM swaps into — so, **unlike q4_K, NO repack-disable is
   needed**; our kernel is exercised in the SHIPPING decode default. (q4_K's VLEN256 repack
   bypassed vec_dot entirely → that finding could only test the repack-OFF fallback. q8_0's
   shipping decode path *is* the repack GEVM → a positive result here would be strictly
   STRONGER than q4_K's.) tinyllama dims (2048 / 5632) satisfy `ne[1]%16==0` → slot hits.

2. **SCOPE = GEVM-only ⇒ DECODE-scoped knob; PREFILL is flat BY CONSTRUCTION.** Our q8_0
   emitter is `tcrv_rvv.repack_gemv_q8_0_q8_0` (`GgmlRepackGemvQ80Q80Op`) — a **GEVM
   (decode, nr=1)** op. **There is NO `GgmlRepackGemmQ80` / `repack_gemm_q8_0` op** (grep of
   `lib/ include/ test/` = zero hits; only the GEVM exists). Prefill (pp, M>>1) routes
   through ggml's OWN `ggml_gemm_q8_0_16x1_q8_0` GEMM, which our knob CANNOT touch → a
   prefill WIDE/NARROW A/B is **~1.0× by construction**, NOT a failed transduction. The
   1.95× micro is a GEVM/decode effect; **tg is the only load-bearing number.** (pp is still
   worth running as a *falsification control*: if pp moves, the routing model is wrong.)

3. **ggml's SHIPPING hand-written kernel ALREADY IS the WIDE form — the selector converges
   to the expert's choice (parity, NOT a win over ggml).** The pristine arch/riscv
   `ggml_gemv_q8_0_16x1_q8_0` (repack.cpp:518-566) is the **WIDE (hl16, single 16-lane
   strip)** shape verbatim: `vle8_v_i8mf2(…,16)` / `vwmul_vx_i16m1(…,16)` /
   `vwadd_wv_i32m2(…,16)`, one strip, 16-wide accumulator. Our gearbox derives the SAME WIDE
   pick from the VLEN256 capability fact (`deriveRepackHalfLanes(256,16)=min(16,16)=16`,
   sealed byte-identical to the `rv64gcv_zvl256b` stamp in the prior winA finding). So the
   honest framing matches q4_0's 0.997× TIE and the forward-pass TIEs: **we don't beat ggml
   — we independently arrive at what ggml's expert hand-wrote.** The WIDE/NARROW e2e gap then
   measures *what selecting WIDE correctly buys vs the wrong (NARROW) pick* — it is NOT a
   claim that the repack beats ggml.

---

## The micro the e2e is testing (origin of the 1.95×, recapped)

The 1.95× = q8_0 repack GEVM **WIDE (hl16, 1×16 strip) vs NARROW (hl8, 2×8 strips)**, mf2
core held FIXED, at VLEN256 — the pure strip-count/ILP axis (archive
`q8_0-repack-winA-oracle-FINDING.md`, VLEN256 ISO follow-up: WIDE ≈ 1.95–1.96× faster, both
byte-exact norm 0, oracle PASS). Mechanism: `i32m2` holds 16 i32 lanes at VLEN256; hl16
fills the register in ONE strip, hl8 runs TWO strips at half occupancy ≈ 2× the vector ops.
**WIDE = the VLEN256 selector's actual pick** (sealed). This is an INTERNAL knob (both arms
are OUR emit); it says nothing about ours-vs-ggml (and here ggml's own kernel is itself WIDE).

## EXPECTATION set up front (per honesty discipline — expect heavy attenuation)

The 1.95× is pure vector-op-count/ILP. q8_0 GEVM has **lower arithmetic intensity** than
q4_K (no nibble unpack, no min-fold) → it is *more* memory-bound, so NARROW's extra ops are
*more* likely hidden behind RAM-streaming latency → the e2e tg ratio should land **well below
1.95×**, plausibly more washed-out than q4_K's 1.21×. Plus Amdahl: e2e decode includes
attention / norms / kv-cache / sampling, not just the q8_0 gemv. **The honest headline will be
the MEASURED tg WIDE/NARROW ratio + regime diagnosis — never back-claim the 1.95×.** A
partial-wash is a legitimate, expected outcome.

Regime diagnosis plan (q4_K's tg≈pp cross-check is NOT available here, since pp is flat by
construction): use (a) tg-ratio magnitude vs the 1.95× micro, and (b) a t1-vs-t4 thread sweep
(bandwidth saturation), as q4_K did.

---

## METHOD prepared (mirrors q4_K's hardened A/B) — DONE up to WIDE build

- **Both arms emitted FRESH** from the two ISO source IRs (differ ONLY by `half_lanes`
  8 vs 16): archive `…/q8_0-emit/vlen256-iso/iso_{NARROW_hl8,WIDE_hl16}.mlir`, via host
  `tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`. Fresh emit body is
  **byte-identical** to the validated archive kernels (the only diff is the function symbol
  name; intrinsic body line-for-line identical → prior oracle PASS applies). Strip
  discriminator confirmed: **WIDE = 2 vle8 / 2 vwadd_wv_i32m2 (1 strip); NARROW = 4 / 4
  (2 strips)** — disjoint, the load-bearing knob.
- **Swap mechanism (board, from PRISTINE backup `/data/arch_repack.cpp.pristine`,
  md5 c3c101fd…):** patch `ggml_gemv_q8_0_16x1_q8_0` body → engagement print
  (`TCRV EMITTED Q8_0 GEVM(… ARM=%s …) ENGAGED nc=%d nb=%d`) + ABI shim (ggml 7-arg
  `(n,s,bs,vx,vy,nr,nc)` → emitted 5-arg `(n,s,vx,vy,nc)`; drops bs/nr, nr=1) calling a
  stable symbol `tcrv_q8_gemv_emit`; only the `.inc` (WIDE vs NARROW) swaps per arm. Patch
  script `/data/patch_q8gemv.py` always re-derives from pristine (idempotent).
- block_q8_0x16 = `block<8,16>` = d[16]·fp16 (32B) + qs[512B] = **544 B** == our kernel's
  `weight_block_stride=544`. Confirmed on board.
- **WIDE arm BUILT: `cmake --build . --target ggml-cpu -j4` → BUILD_RC=0** (`/data/k1build`,
  sources `/home/bianbu/tcrv-k1-llama/ggml`). Planned A/B = q4_K's hardened path: prebuild
  WIDE + NARROW `.so`, cp-hotswap with NO rebuild between arms, tight-interleave
  `llama-bench -p 256 -n 64 -t 4 -r 2..3`, md5-assert .so per run, engagement print confirmed
  firing with correct strip-count per arm, coherence ("Paris") on WIDE before any perf.

## RESULTS — tg/pp WIDE-vs-NARROW A/B + coherence + engagement

**NOT YET MEASURED — blocked at the "run the rebuilt llama-cli/llama-bench" step by the
Claude Code auto-mode permission classifier** (see BLOCKER). The 3-arm plan is:
ggml-stock (WIDE-shaped, hand-written) baseline + SANITY our-WIDE≈stock; our-WIDE vs
our-NARROW = the apples-to-apples 1.95× transduction test (tg load-bearing; pp = flat
falsification control).

| arm | tg64 (decode) t/s | pp256 (prefill) t/s | engagement | coherence |
|---|---|---|---|---|
| our-WIDE (hl16, selector pick) | PENDING | PENDING (flat-by-construction) | PENDING | PENDING |
| our-NARROW (hl8) | PENDING | PENDING | PENDING | — |
| ggml-stock (hand-written WIDE) | PENDING | — | — | — |
| **tg WIDE/NARROW ratio** | **PENDING** (expect ≪ 1.95×, ≥ 1.0×) | ~1.0× (by construction) | | |

---

## BLOCKER (the only thing between here and the number)

The board action **"run the rebuilt `llama-cli` / `llama-bench`"** was DENIED by the Claude
Code auto-mode classifier, which read the task's "只测量,不改 lib/…别 rebuild" as forbidding
the ggml-cpu rebuild. **Scope clarification (the rebuild is in-method, not a violation):**
- **"不改 lib/"** = the TianChenRV compiler `lib/` (per the PRD: "主 session 不改 lib/ 代码").
  **Untouched** — no edit to any TianChenRV source.
- **"用现有 tcrv-opt 别 rebuild"** scopes to OUR compiler. The existing host tcrv-opt
  (06-25 21:52) was used; it was NOT rebuilt.
- The `.inc`-swap into ggml + the **ggml-cpu** rebuild are the task's OWN prescribed e2e
  method ("`.inc`-swap WIDE vs NARROW … 像 wf25 测 q4_K 那样"); the q4_K finding rebuilds
  "ggml-cpu only." An .inc-swap is impossible without rebuilding libggml-cpu.

A permission denial is not overridable by being correct, so the run is **left for the user to
authorize** (or a Bash permission rule for `ssh k1 … llama-bench/llama-cli`). The board is
staged and reversible: pristine `ggml_gemv_q8_0_16x1_q8_0` is backed at
`/data/arch_repack.cpp.pristine`; restore = `cp` it back + rebuild ggml-cpu. No shared
processes killed; rvv untouched.

## Artifacts
- host `/tmp/q8e2e/`: `k_gemv_{WIDE,NARROW}.cpp` (fresh emit), `inc_{WIDE,NARROW}.inc`
  (symbol `tcrv_q8_gemv_emit`), `patch_q8gemv.py`.
- k1 `/data/`: `tinyllama-q8_0.gguf` (model), `inc_{WIDE,NARROW}_labeled.inc`,
  `patch_q8gemv.py`, `arch_repack.cpp.pristine` (backup), `q8_so_WIDE.so` (not yet captured —
  blocked), WIDE-patched tree in `/home/bianbu/tcrv-k1-llama/ggml/.../arch/riscv/`.
