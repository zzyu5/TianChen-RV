# K-quant Win-A knob — precisely-scoped DEFERRED gap (2026-06-24)

**Goal:** make the K-quant block-dots (q4_K/q6_K/...) Win-A-ablatable (gearbox-selectable LMUL), like the
5 existing block-dots (q4_0/q8_0/q4_1/q5_0/q5_1).

**Finding (why it's not quick wiring):** the K-quant emitter `lib/Conversion/RVV/RVVToEmitCKQuant.cpp` is
**NOT LMUL-parametric** — it hardcodes `m2` throughout (`__riscv_*_u8m2`/`_i8m2`/`_i32m2`, `vsetvl_e8m2`,
`vint8m2_t`, ... at dozens of sites). The 5 wired block-dots work because their emitter
(`RVVToEmitCBlockQuantLinear.cpp`) is parametric (reads `integer_core_lmul`, builds the type strings from
it). The gearbox wiring (TunableScheduleOpInterface on the op + a `RVVScheduleDescriptorRegistry` case +
`enumerateRVVQ4KQ8KShapeCandidates`) is ~30 lines and trivial — but it has NO effect until the emitter is
parametrized, because WIDE and NARROW would emit the identical hardcoded-m2 kernel.

**Precise scope to do it (deferred — a real emitter refactor, ~0.5–1 day, 3 K-quant emitters share the
pattern):** parametrize `RVVToEmitCKQuant.cpp` to read the stamped `integer_core_lmul` and build the
`u8<L>/i8<L>/i16<2L>/i32<2L>` type+intrinsic strings from it (mirror `RVVToEmitCBlockQuantLinear.cpp:63-77`),
THEN add the 30-line gearbox wiring. The candidate LMUL set is the K-quant-legal {m1, m2} (super-block of 256
= 8×e8m2 strips or 16×e8m1; both legal at VLEN128/256).

**Why this matters (the N3 story it would complete):** the just-measured K-quant **Win-B micro is a LOSS**
(q4_K ggml 1.72×, q6_K 2.26×, q3_K 2.13× vs ggml's hand-tuned `_vl256`) **precisely because our emit is fixed
at one LMUL** while ggml's `_vl256` tunes LMUL/shape per kernel. Parametrizing + adding the Win-A knob would
let the Gearbox pick the better LMUL → directly tests whether the tune closes the hand-tuning gap. This is the
cleanest "Win-A narrows a real Win-B loss" demonstration available. DEFERRED, not abandoned — precisely scoped.
