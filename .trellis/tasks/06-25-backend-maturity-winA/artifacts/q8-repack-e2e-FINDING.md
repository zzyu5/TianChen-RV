# q8_0 repack GEVM — WIDE-vs-NARROW e2e transduction on k1 (头号 gap / 注5) — FINDING

**Date:** 2026-06-25 (routing/scope/staging) → **2026-06-26 (FINAL e2e measured)**
**Board:** `ssh k1` (Spacemit X60, RISC-V, **VLEN256**, 8 cores; ISA `…zvfh ime`),
`taskset -c 0-3`, serial, k1-only (rvv untouched). Model: tinyllama-1.1b-chat-v1.0
**Q8_0** (1.17 GB, downloaded host → scp `/data/tinyllama-q8_0.gguf`).
**Tooling discipline:** OUR kernels emitted with the existing host `build/bin/tcrv-opt`
(built 06-25 21:52) — **NO rebuild of our compiler, NO change to TianChenRV `lib/`.**
The `.inc`-swap into ggml + the ggml-cpu rebuild are the prescribed e2e method (same as
the q4_K wf25 finding, which rebuilds "ggml-cpu only").

## STATUS — DONE (2026-06-26). User authorized ggml-cpu rebuild + llama-bench/llama-cli
run. The tg/pp WIDE-vs-NARROW A/B is **MEASURED** (3 paired t4 rounds + 1 t1 pair), plus a
**ggml-stock baseline** (3rd rebuild from pristine) confirming our-WIDE ≈ ggml-stock = exact
parity. Coherence ("Paris") PASS on WIDE (ABI shim certified). Board restored to pristine
after measurement (reversible). See **## FINAL e2e 数** below for the load-bearing numbers.

---

## HEADLINE (routing/scope facts — established pre-A/B; the measured A/B is in ## FINAL e2e 数)

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

## FINAL e2e 数 (2026-06-26, k1 VLEN256, taskset -c 0-3, tinyllama-1.1b Q8_0)

Method: prebuild WIDE (md5 `208c2cf8`) + NARROW (`1a65d011`) + ggml-stock (`14b6add6`, from
pristine) `.so`; cp-hotswap with **md5-assert per run** (live==intended), NO rebuild between
arms; tight-interleaved `llama-bench -r 3`; engagement print confirmed per arm (WIDE/NARROW
trip `TCRV EMITTED … ARM=%s`, stock trips nothing = `ourprint=0`). Coherence on WIDE = PASS
("Yes, the capital of France is Paris.", engagement fired 4×, ABI 7→5-arg shim certified).
(Disclosure: NARROW was NOT separately Paris-checked — its correctness rests on the
byte-exact WinA oracle [NARROW↔WIDE norm 0] + the SHARED ABI shim, which WIDE's Paris
certifies; NARROW emits the same intrinsics with 2 strips instead of 1, same shim. Sound
chain, but stated explicitly so the doc doesn't imply both arms got a fresh generation.)

**tg (decode) — load-bearing — t4 (4 threads), 3 paired rounds (min/mean shown):**

| arm | tg64 t/s (3 rounds) | mean |
|---|---|---|
| our-WIDE (hl16, selector pick) | 4.459 / 4.445 / 4.455 | **4.453** |
| our-NARROW (hl8) | 3.135 / 3.132 / 3.138 | **3.135** |
| ggml-stock (hand-written x16) | 4.420 / 4.433 (2 rounds) | **4.426** |

- **tg WIDE/NARROW = 4.453 / 3.135 = 1.42×** (decode; range tiny, sd ≤0.018) — the knob
  TRANSDUCES to decode (~42% faster decode picking WIDE vs the wrong NARROW).
- **our-WIDE ≈ ggml-stock = 4.453 / 4.426 = 1.006× (PARITY, sealed empirically).** Round-1
  matched pair: stock 4.4203 vs our-WIDE 4.4199 = 0.99990×. Our gearbox's WIDE pick is
  performance-identical to ggml's expert hand-written `q8_0_16x1` kernel — **we converge on
  the expert's choice, we don't beat ggml.** This matches the source-read (pristine kernel is
  1× `vle8_v_i8mf2` / `vwmul_vx_i16m1` / `vwadd_wv_i32m2`, all `,16)` = single 16-lane strip =
  WIDE) and the q4_0 0.997× TIE.

**pp (prefill) — flat falsification control — t4:** WIDE 10.696 / 10.703 / 10.710; NARROW
10.704 / 10.703 / 10.695 → **pp WIDE/NARROW = 1.000×** (flat, range 0.015). Confirms the
routing model: prefill (GEMM, M>1) routes through ggml's OWN `ggml_gemm_q8_0_16x1_q8_0`,
which our decode-only GEVM knob CANNOT touch — flat BY CONSTRUCTION, not a failed
transduction. (If pp had moved, the routing model would be wrong; it didn't → confirmed.)

**Regime diagnosis — tg t1 (1 thread) vs t4:**

| | WIDE tg t/s | NARROW tg t/s | WIDE/NARROW | stock tg | stock/WIDE |
|---|---|---|---|---|---|
| t1 (1 thread) | 1.406 | 0.954 | **1.47×** | 1.416 | 0.99× (parity) |
| t4 (4 threads) | 4.453 | 3.135 | **1.42×** | 4.426 | 1.01× (parity) |

t1 ratio (1.47×) ≳ t4 ratio (1.42×): at 4 threads bandwidth saturation **mildly** attenuates
the compute knob, but the WIDE strip retains most of its advantage even multi-threaded — q8_0
GEVM still has compute/ILP-bound headroom WIDE exploits (vs q4_K, where the repack knob was
fully bypassed at the factory VLEN256 path). **Both ratios ≪ 1.95× micro** — heavy Amdahl
attenuation (e2e decode = q8_0 gemv + attention + norms + kv-cache + sampling), exactly as
predicted up-front. The micro 1.95× is NOT the e2e number; **1.42× is.**

### Honest headline (per discipline)
The load-bearing e2e fact: **selecting WIDE correctly (the VLEN256 gearbox pick) buys ~1.42×
decode vs the wrong NARROW pick** — the 1.95× internal-knob micro DOES transduce to decode,
attenuated to 1.42× by memory-boundedness + Amdahl. **Strategically this is PARITY, not a win
over ggml**: ggml's shipping `q8_0_16x1` is itself WIDE, and our-WIDE measured == ggml-stock
(0.999–1.006×). So the number certifies that our capability-driven selector arrives at the
expert's kernel (and that picking the wrong lane-width would cost 1.42× of decode), NOT that
our repack beats ggml. (q8_0 repack Win-B vs-ggml block-dot was 0.65× LOSS in the micro table
— orthogonal; here the GEVM IS the shipping path and equals ggml's own GEVM.)

## Board restored
After measurement, pristine `repack.cpp` (md5 `c3c101fd`) restored to tree + ggml-cpu rebuilt.
**Stale-object catch (confirms the "incremental builds unreliable" memory):** the first
restore-rebuild was a no-op — cmake saw repack.cpp unchanged and reused a stale `.o`, leaving
the live `.so` at the WIDE fingerprint `208c2cf8`. A `touch repack.cpp` + forced rebuild then
produced the true stock binary (`14b6add6`, **0** `TCRV EMITTED` strings = genuinely
unpatched), and the leftover `tcrv_emitted_repack_gemv_q8_0.inc` (our artifact, unreferenced
by pristine) was removed. Final live `.so` = `14b6add6` stock. No shared processes killed
(only my own llama-bench/driver); rvv untouched; /data not filled (109G free).

## Artifacts
- host `/tmp/q8e2e/` + `/tmp/q8_ab_driver.sh`, `/tmp/q8_stock_ab.sh`, `/tmp/q8_build_stock.sh`.
- k1 `/data/`: `tinyllama-q8_0.gguf`; prebuilt `q8_so_{WIDE,NARROW,STOCK}.so`
  (md5 `208c2cf8`/`1a65d011`/`14b6add6`); `inc_{WIDE,NARROW}_labeled.inc`; `patch_q8gemv.py`;
  `arch_repack.cpp.pristine` (backup); results `/tmp/q8_ab_results.txt`,
  `/tmp/q8_stock_results.txt`, `/tmp/q8_ab_raw.csv`; coherence `/tmp/q8_coh_wide.out`.
