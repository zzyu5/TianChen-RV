# q3_K native-mask KNEST GEVM decode-leaf — DEPLOYED (emitter implantation)

> Resolves **[GAP-EMIT-KNEST-Q3K-HMASK]** (casefile `../kquant-family-knest-G1/evidence.md` §5.1, commit `05f96739`).
> Predecessor design + byte-exact PROVEN there (native ≡ OLD ≡ stock, be_q3k 0/8); template = q5_0/q5_1 REDESIGN-B (`40c21de0`).
> **Honest framing (unchanged): kernel-axis C1 artifact — the reusable native-mask element is now DEPLOYED (designed→deployed) across q5_0/q5_1/q3_K, byte-exact. NOT a perf-covered move** (K-quant M=1 GEVM instruction-count structural limit already proven — q3_K 4.77× stock; [CASE-MICRO-E2E]). Value = compositional extensibility realized in the emitter.
> **No board · no git · byte-exact hard gate + objdump = sufficient (kernel-axis).**

---

## TL;DR
1. **Deployed**: `emitRepackKQuantGemvBodyQ3K` hmask recon swapped OLD per-lane expand chain (`vsrl(hm,p) | vand 0x01 | vsll 2 | vor | vsub 4`) → **native-mask KNEST** (`vand(1<<p) + vmseq==0` then `vadd_vx_i8_mu(mask0, base, base, -4)` — high-bit select + −4 bias FUSED into ONE masked op). Byte-exact by construction.
2. **Isolation PROVEN**: only 3 files touched — the q3_K **GEVM** leaf + its 2 lit tests. Both `.cpp` diff hunks are inside `emitRepackKQuantGemvBodyQ3K` (16840, 16857). Sealed GEMM leaf (`emitRepackKQuantGemmBodyQ3K` 17208+), q5_0/q5_1 REDESIGN-B leaf, and other K-quant leaves (q2/q4/q5/q6_K) UNTOUCHED — proven by full Conversion/RVV suite green (242/242, no other test perturbed).
3. **byte-exact hard gate = PASS**: be_q3k algebraic anchor 0/8 (native ≡ OLD ≡ stock); source-level emitc = 1024 `vmseq` + 1024 `vadd..._mu`, ZERO `vsll/vor/vsub`; lit 242/242 (prev baseline 242, no regression).
4. **objdump (deployed asm, gcc-15.2 -O3, VLEN128) verified**: native-mask signature deployed + v-insn reduction real.

## byte-exact hard gate
| gate | result |
|---|---|
| be_q3k (base2∈[0,3]×hbit∈{0,1}=8 cases) | **0/8** — NATIVE ≡ OLD ≡ stock scalar (`PASS`) |
| source emitc intrinsic histogram | **1024 `__riscv_vmseq_vx_u8mf2_b16` + 1024 `__riscv_vadd_vx_i8mf2_mu`**; `vsll/vor/vsub` = **0** (OLD chain retired) |
| Conversion/RVV lit | **242/242 PASS** (baseline 242 — zero regression) |
| q3_K GEVM lit (`rvv-to-emitc-repack-gemv-q3-K-q8-K` + `rvv-emit-identity-...-q3-K-repack-vlen128`) | PASS — CHECK updated to native-mask order + `RETIRED-NOT` proves vsll/vor/vsub gone |

## objdump / deployed-asm verification (per super-block-col-group · gcc-15.2 -O3 · VLEN128)
| metric | OLD (cur) | NATIVE | Δ |
|---|---|---|---|
| total v-insn | 8846 | **7125** | **−19.5%** |
| vset* (vsetvli tax) | 2675 | 2183 | −18.4% |
| vs*r.v / vl*r.v (tile-materialization stack traffic) | 174 | 153 | −12.1% |
| vsll.vi (OLD hbit lift) | 512 | **0** | RETIRED |
| vor.vv (OLD hbit merge) | 512 | **0** | RETIRED |
| vmseq.vi (native mask create) | 0 | **512** | native |
| vadd.vi masked (`,-4,v0.t`) | 0 | **512** | native fused bias |
| vadd.vi plain (OLD unmasked −4; gcc folded vsub→vadd) | 512 | 0 | fused into masked |
| vand.vi/vx | 1024 | 1024 | unchanged (qs&0x03 + hm&(1<<p)) |

Deployed signature (asm): `vmseq.vi v0,vN,0` → `vadd.vi vM,vM,-4,v0.t` (512× masked). OLD had **0** masked vadd (select was separate vsll+vor). This is the q5_0/q5_1 native-mask element, byte-exact, now on q3_K.

**Note on magnitude**: rolled apples-to-apples (casefile §3.3) showed −43.7% on an isolated single strip; the deployed FULL-UNROLL form shows −19.5% total (diluted by the fixed loop/vsetvli overhead native-mask does not touch), directionally consistent. The native-mask also RELIEVES register pressure enough to cut stack traffic 174→153, but does NOT eliminate it (still per-element-broadcast tile round-trip) — that needs the separate rolled-strip redesign (deferred; [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP]).

## Scope
- **native-mask leaf swap = COMPLETE + byte-exact + deployed** (primary deliverable).
- **rolled register-budget redesign = NOT done (deferred, honest)**: deployed form still has 153 vs*r.v stack traffic (down from 174). register-budget-fit remains deployed-FAIL / rolled-PASS-achievable per casefile §3.2 — orthogonal structural change, not a correctness item.
- **perf-gate still打回**: instruction-count 4.77× stock structural limit unchanged (native-mask trims −19.5% but does not approach ≤1.10×) — expected, [CASE-MICRO-E2E]. This is emitter-maturity / compositional-extensibility, not a perf-covered move.

## Files changed (no git)
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` — `emitRepackKQuantGemvBodyQ3K` only: removed `vsll/vor/vsub` OLD helpers, added native-mask helpers (`q3BoolType`, `vmseqZero`, `addBiasMasked`), rewrote `assembleWeight`.
- `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q3-K-q8-K.mlir` — native-mask CHECK order + `RETIRED-NOT` prefix.
- `test/Conversion/RVV/rvv-emit-identity-quant-contraction-q3-K-repack-vlen128.mlir` — native-mask CHECK order.

## Reproduce
```
GCC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc
GPP=${GCC%gcc}g++ ; WEFT=build-weft/bin/weft-opt
F=test/Conversion/RVV/rvv-emit-identity-quant-contraction-q3-K-repack-vlen128.mlir
$WEFT $F --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp > raw/q3k_gevm_NATIVE.emitc.c
$GPP -march=rv64gcv_zvfh -mabi=lp64d -O3 -S raw/q3k_gevm_NATIVE.emitc.c -o raw/q3k_gevm_NATIVE.gcc15.O3.s
../kquant-family-knest-G1/raw/be_q3k   # byte-exact 0/8 anchor
# lit: (cd build-weft/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -s Conversion/RVV)  -> 242/242
```
