# [GAP-SB] iq2_xxs pair-batching — real machine-code objdump (PRE vs POST)

de-risk question: emit-level "gather 16->8" was an EMIT-level proxy. Does the real
RISC-V machine code actually drop the gather/vsetvli count? Counted here on the board
(llvm-objdump-17) over the exported .o, NOT the emit-plan.

## provenance
- PRE  = ours @ 0c64477c (before pair-batching)  export: tcrv-opt(pinned) | tcrv-translate(pinned) --tcrv-export-target-artifact
- POST = ours @ 169f0cc0 (the [GAP-SB] pair-batching commit)
- input mlir: test/Target/RVV/iq2-xxs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir (git-show at each sha; PRE/POST differ only in FileCheck `// CORE:` comment lines — codegen is emitter-driven from the source_front_door attr, compiled into the binary)
- PRE/POST binaries built via tools/bench/byte-exact-baseline.sh cached (detached worktrees; main tree + build/ untouched; no git stash)
- neither intervening commit (e37e64ef GAP-NUM q8_0, 1076d75e heal-fix) touches iq2_xxs => PRE->POST isolates the pair-batching
- object size: PRE 6640B -> POST 6520B
- sha256: PRE f86b5f4d... POST f9e87d23...

## per-mnemonic count (whole exported object == single kernel + its .L0 body)
(register-operand tokens vNN excluded; these are true instruction mnemonics)

| mnemonic         | PRE | POST | delta | role |
|------------------|-----|------|-------|------|
| **vluxei16.v (GATHER)** | **16** | **8** | **-8 (HALVED)** | grid + derived-sign indexed gather — the [GAP-SB] target |
| vle16.v (index load)    | 16  | 8    | -8 (HALVED) | u16 byte-offset index load feeding the gather |
| vmul.vv (sign fold)     | 8   | 4    | -4 (HALVED) | +-1 sign folded onto grid |
| vzext.vf2               | 8   | 4    | -4 (HALVED) | decode widen |
| vsll.vi                 | 8   | 4    | -4 (HALVED) | decode shift |
| vsetvli                 | 35  | 26   | -9  | config churn |
| vsetivli                | 21  | 14   | -7  | config churn |
| **ALL vset* (config churn)** | **56** | **40** | **-16 (-29%)** | per-sub-block config -> per-pair |
| vwmul.vv                | 8   | 8    | 0   | GEARBOX DISCRIMINANT — per-sub-block widening product (unchanged by design) |
| vwredsum.vs             | 8   | 8    | 0   | per-sub-block reduction (unchanged by design) |
| vmv.x.s                 | 8   | 8    | 0   | scalar extract of each sub-block partial |
| vse16.v                 | 8   | 8    | 0   | store |
| vle8.v (q8 load)        | 16  | 12   | -4  | q8 activation load |

### NEW instructions introduced by pair-batching (the cost side)
| mnemonic        | PRE | POST | note |
|-----------------|-----|------|------|
| vslide1down.vx  | 0   | 12   | +12  pair assembly / lane shuffle |
| vslideup.vi     | 0   | 4    | +4 |
| vmv.v.x         | 0   | 4    | +4 |
| vand.vx         | 0   | 4    | +4 |
| vmv1r.v         | 0   | 3    | +3  (this is the `vget_v_i8m4_i8m2` lowering — a whole-register move, not a literal vget) |
| vlenb           | 0   | 4    | +4 |
| vl1r.v / vs1r.v | 0   | 2/2  | +4  register-group spill/reload |

### NET
- total instructions (incl. scalar): **PRE 459 -> POST 436  (-23, -5.0%)**
- The targeted gather/index/config instructions ARE genuinely halved on real machine code
  (proxy CONFIRMED). But the m4-pair -> per-sub-block-m2 re-split introduces slide/register
  shuffle overhead (+12 vslide1down + spills), so the NET machine-code reduction is only ~5%,
  NOT ~50%. => whether it is FASTER on silicon is not decidable from the count; that is exactly
  what the A/B micro measures.
- vwmul/vwredsum 8->8 unchanged confirms the integer core (the fold order / value semantics) is
  byte-exact-preserved, consistent with the STRICT byte-exact claim.
