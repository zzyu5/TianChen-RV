# G4-M1b — 线甲b: k1 IME batched re-seal + T5b batched re-measurement

> Board casefile for **线甲b (board·k1)**: after the 线甲a emitter fix
> (`2f1af5c8`, `[GAP-IME-LEAF-PIPELINE]` — per-fragment leaf → register-resident
> **batched** leaf `tcrv_ime_vmadot_mac_kloop`), this re-seals the three IME GEMM
> tiles on real K1 silicon in their **batched** form and re-runs the T5b
> paradigm-lever sweep to confirm the batched leaf = Cell1b ~2× win.
> **[NG-4] methodology framing only** — correctness silicon seal + range-vs-range
> ablation; opponent = **SELF** (our own RVV-vector construction); eight-gate NOT
> invoked; NO "IME X-times" e2e headline; NO vs-ggml beat.

## Provenance / discipline
- **HEAD unchanged** at `f78ad2da` throughout (no git in this line; main session commits).
- Board dir isolated (`~/tcrv-m1b-reseal`), zero stock-tree modification, cleaned after seal.
- Four sources copied from in-tree, **md5 byte-identical** (board == in-tree):
  - `q4-0-matmul-tile-int32-k1seal.c` = `ea8ef66ced3f6bf892f42c3f70ed454c`
  - `q8-0-matmul-tile-int32-k1seal.c` = `e7c4d0c57771db33be61333ed03b36a5`
  - `q4-K-matmul-tile-int32-k1seal.c` = `7bb64d7287fae016e1ad4168446c65d7`
  - `q4-0-paradigm-lever-t5b-k1.c`    = `c0565b6a55091de3a454064d2b6febf2` (unchanged since M3; already carried Cell1b)
- **As-emitted (pre-fix) M3 data is NOT touched** — lives in
  `experiments/active/g4-m3-ime-paradigm-t5b/` as the修复前对照.

## Board fingerprint
| field | value |
|---|---|
| board | k1 (SpacemiT X60), VLEN=**256**bit (t5b probe), 8 harts, harts pinned 0–3 via `taskset -c 0-3` |
| toolchain | gcc (Bianbu 13.2.0-23ubuntu4bb3) **13.2.0**, `-O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d` |
| assembler | GNU binutils/objdump **2.42** (assembles/decodes `vmadot` once the march token unlocks the opcode) |
| kernel | Linux bianbu 6.6.63 riscv64 |
| loadavg | pre ≈2.2 (background avahi daemon; harts 0–3 free), post ≈2.2 stable |
| ledger | KERNEL (compiler-symmetric: ONE binary, ONE compiler for both ranges) + SYSTEM (absolute ns) |
| opponent-identity | **SELF** (our RVV-vector Cell3b/Cell3c) — range-vs-range ablation, NOT vs-ggml |
| eight-gate | **NOT invoked** |

## step-0 board self-check — PASS
`ssh k1` reachable; `gcc -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d` compiles (MARCH_OK);
`objdump` 2.42 present; `taskset` present; nproc=8.

## (1) objdump golden — vmadot 0xe210312b at every leaf (encoding UNCHANGED)
The batched leaf still disassembles to the **identical** encoding — only the
*scheduling* changed (per-fragment ceremony → single `vsetvli` + K/8 in-register
accumulate + store-once), not the instruction:
```
q4-0seal  10838:  e210312b   vmadot v2,v0,v1
q8-0seal  10860:  e210312b   vmadot v2,v0,v1
q4-Kseal  10ece:  e210312b   vmadot v2,v0,v1
t5b       1160e:  e210312b   vmadot v2,v0,v1   (Cell1 leaf)
t5b       11746:  e210312b   vmadot v2,v0,v1   (Cell1b batched leaf)
```
See `seal_run_raw.txt`. **No encoding change** ⇒ safe to refresh seal.

## (2) k1 silicon int32 0-diff — SEAL PASS ×3 (batched leaf, no epsilon)
Register-resident batched `vmadot` == ZERO-MODEL, verified in-register (accumulate
in v2/v3 across the K/8 fragment loop):
| format | decode | int32 core (real vmadot, batched) | epilogue | max\|·\| |
|---|---|---|---|---|
| q4_0 | 512/512 nibbles bit-exact | 64/64 tiles bit-exact | — | max\|C\|=7946 |
| q8_0 | 512/512 int8 quants bit-exact | 64/64 tiles bit-exact | — | max\|C\|=102040 |
| q4_K | 2048/2048 nibbles + 128/128 6-bit sc/m bit-exact | 128/128 tiles (S_scale **AND** S_min) bit-exact | 64/64 float epilogue byte-exact | max\|S_scale\|=1403271 |
All three exit 0 + `SEAL PASS`. Raw in `seal_run_raw.txt`.

## (3) T5b batched re-measurement — Cell1b ~2× confirmed; as-emitted 4× loss eliminated
Full-sweep raw: `t5b_batched_run1_full.txt`; cross-run key rows: `t5b_batched_run2_run3_keyrows.txt`.
- ZERO-MODEL correctness gate: **all 6 cells × all 10 M = int32-EXACT (0-diff)** before any timing.
- self-check(M=64, Cell3, 3×) spreads: **4.73% / 3.77% / 2.13%** — all `OK(<=5%)`.
- Hygiene SOP: paired alternation (all cells interleaved per rep), N=15 median, adaptive inner-repeat ≥300 µs, taskset pin.

**Ratio = ns / Cell1b (register-resident batched IME = fair matrix baseline); >1 ⇒ slower than batched-IME.**
Cell3b = the fair vector opponent (vector-optimal layout, MR=4 4-row register blocking).

| M | Cell1 (as-emitted leaf) | Cell1b (batched)=1 | **Cell3b (fair vector best)** | Cell3c (i8 traffic) |
|--:|--:|--:|--:|--:|
| 1   | 8.63 | 1.00 | **1.32** | 1.57 |
| 2   | 8.72 | 1.00 | **2.66** | 3.14 |
| 4   | 8.71 | 1.00 | **1.92** | 2.35 |
| 8   | 8.65 | 1.00 | **1.92** | 2.34 |
| 16  | 8.59 | 1.00 | **1.92** | 2.34 |
| 32  | 8.54 | 1.00 | **1.92** | 2.35 |
| 64  | 8.66 | 1.00 | **1.96** | 2.40 |
| 128 | 8.60 | 1.00 | **1.98** | 2.46 |
| 256 | 8.57 | 1.00 | **2.07** | 2.51 |
| 512 | 8.43 | 1.00 | **2.09** | 2.52 |

Cross-run reproducibility (Cell3b/Cell1b, 3 runs): M=1 {1.32,1.35,1.32}; M=4 {1.92,1.96,1.92};
M=64 {1.96,1.97,1.96}; M=256 {2.07,2.12,2.07}; M=512 {2.09,2.13,2.08} — reproduce to ≈1–2%.
Matches the M3 prediction (M3 Cell1b was already the batched form; M3: M=1≈1.33, M=4≈1.93, M=512≈2.1).

### Readings
1. **Emitter GAP closed (mechanism headline).** As-emitted Cell1 (per-fragment leaf) = **8.5× Cell1b ≈ 4.4× Cell3b** — a ~4× LOSS to the fair vector path, pure per-fragment ceremony overhead (e8↔e32 `vsetvli` toggle + zeroing + store + scalar acc[]), NOT `vmadot` throughput. The 线甲a emitter fix moves production emission to the batched leaf (Cell1b), which flips IME from a 4× loss to a **~2× win**. `[GAP-IME-LEAF-PIPELINE]` is now **closed in production emission and re-sealed on silicon**. Same律 as K-quant repack un-pipelined→tiling (LAW-FIRST-EMISSION §L1).
2. **Crossover M\* (compute-isolated kernel account).** With the fair register-resident baseline the **matrix range wins at ALL M ∈ {1..512}** (M\* ≤ 1 in the compute-isolated regime), advantage **grows with M**: ≈1.33× at M=1 (GEVM, matrix's weakest regime) → ≈1.9× by M=4 → ≈2.1× at M=512 (prefill), **saturating by M≥4**.
3. **micro↛e2e caveat (critical).** This microbench pre-decodes weights + is L2-resident ⇒ **compute-bound by design**; the M=1 compute-micro win (1.33×) **does NOT** transplant to the memory-bound decode/GEVM regime (weight streaming + dequant dominate). Decode M=1 must be recorded as **roofline parity**, not an IME decode win ([[kernel-wins-dont-transplant-to-e2e]]: IME 5.51× kernel → 0.86× decode e2e).
4. **Paradigm advantage is real compute, not a traffic artifact.** Cell3c (vector path with IME's *exact* int8 byte traffic) is SLOWER than Cell3b (2.3–2.5× vs 1.9–2.1×) — reducing weight bytes did not help ⇒ both paradigms compute/issue-bound; the ~2× is a genuine matrix-MAC-throughput win (`vmadot` 128 MAC/instr vs `vwmacc` 32 MAC/instr).

## perf-covered honest classification (per 决策权限卡 / M4 口径)
IME 3 格 (q4_0/q8_0/q4_K @ime) batched → compute-account ~2× win, **BUT**:
opponent = **SELF**, eight-gate **not invoked**, M=1 compute-micro **does not transduce to decode e2e**.
⇒ **内核轴对称候选 · 黄格** (analogous to FLAT `[GAP-FLAT-E2E]`), **NOT counted as perf-covered green**.
GAP status downgraded: was "emitter 未流水" → now "**kernel-axis-only, awaiting e2e / vs-opponent**".
Correctness structural seal (m4_class=certified) is unchanged and independent of this perf classification.

## schema board_seal pointer refresh (done)
`schema/coverage-sixstate.v1.json` states[79]/[80]/[81] (`gemm_tile/{q4_0,q8_0,q4_K}@ime`):
the `[IME-SEAL]` `board_seal=` evidence pointer refreshed to the **batched** codegen
(register-resident `tcrv_ime_vmadot_mac_kloop`, encoding 0xe210312b unchanged/scheduling-only,
re-sealed 线甲b 2026-07-11). `state=constructed` and `m4_class=certified` **UNCHANGED** for all three
(correctness invariant; structural axis untouched).

## T5c trigger — SATISFIED (writeback left to 线甲c / main session, NOT done here)
Pre-registered condition (ROADMAP T5c row): **`[GAP-IME-LEAF-PIPELINE]` closed ∧ post-fix T5b re-measure yields a real M\***. Both met:
- GAP closed (线甲a emitter fix, silicon-re-sealed here).
- Real M\*: matrix range wins ∀M∈{1..512} compute-isolated (M\*≤1), saturating M≥4 at ~2×.
⇒ **T5c can auto-land**: write M\* to P7 prior by instance-hash (M≥M\* prefer matrix in the compute-bound prefill regime; **decode M=1 recorded as roofline parity** — do NOT key selection on the 1.33× compute-micro) + T4a attribution sample verifies "M>M\* auto-selects matrix". Selector prior writeback is **in-tree另一域** — left to 线甲c / main session; this board line does **not** touch the selector.
