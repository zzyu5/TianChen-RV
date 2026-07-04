# M-FLAT nb-loop-layer — GAP evidence log (机制章素材)

The tracer-bullet target IR (`../target-ir/q8_0-nb-loop-target.mlir`) driven
parse → verify → lower → byte-exact. Each wall = one GAP-1 entry: 命名 → 拆法 →
复测 → 归因. This is the load-domain main shape (vec_dot 24/92 keys; GEMM tile
recurs the same shape) whose LOOP layer the single-block-all-vector dialect never
grew.

---

## GAP-1 / W2 — 逐块寻址 load (per-block addressing)  [TORN 2026-07-04]

**命名.** The typed integer-core load must address a per-block strip at
`base + block_index*block_stride (+ quant_byte_offset)`. The dialect's
`tcrv_rvv.load` only addressed a single ABI-base buffer (`$buffer, $vl`) — the
per-block form the nb loop needs was never built. Surfaced at the FIRST parse of
the target IR (`error: expected ','` on `load %vx block %ib`).

**拆法 (minimal, mirrors brick① step-3, no preventive generalization).**
- ODS (`RVVOps.td` `def LoadOp`): + `Optional<Index>:$block_index`,
  `OptionalAttr<I64Attr>:$block_stride`, `OptionalAttr<I64Attr>:$quant_byte_offset`;
  assembly `(block $block_index^ : type)?` after the regular operands. Absent
  block_index = byte-identical single-block form.
- `isAllowedLoadAttr` (`RVVDialect.cpp`): allow exactly `block_stride` +
  `quant_byte_offset` (all other load attrs stay disallowed).
- `LoadOp::verify` (`RVVDialectMemoryOps.cpp`): operand count 2|3 by block_index;
  block_index present ⟺ block_stride present, nonzero (fail-closed); absent form
  must carry neither (fail-closed).
- i8m2 recognition (W1-adjacent, same turn): the single-block widening-source
  fast-paths require the whole load→product→reduce chain to be DIRECT children of
  the `with_vl`; in the loop body their parent is the loop op, so they are skipped
  and the load falls to the generic-route type check (rejects SEW8). Minimal fix:
  when `block_index` present, accept exactly `isGenericRVVVectorSignedI8M2` (the
  q8_0 core's load type) — loop-body allowlist (step 4) + byte-exact lowering (W4)
  guard chain integrity.

**复测.** target IR advances parse→verify CLEAN (exit 0); the load's 3 operands +
block_index induction var + attrs accepted (`load(%buffer,%vl,%ib)<{block_stride,
quant_byte_offset}>`). New per-block-load unit lit: TODO (deferred into W4 — the
byte-exact load emit is produced by the walker, not yet wired). Zero regression:
all 352 Conversion/RVV + Target/RVV green (existing single-block load routes,
3 strong routes, step1-4, nesting-relaxed 352 unchanged).

**归因.** The dialect was built for single-block, all-vector strong routes whose
loads address one ABI base and whose whole dataflow chain are direct children of a
single `with_vl`. The nb loop's per-block addressing + a loop op sitting between
the chain and its vl scope are structurally new — a missing capability, not a bug.

**⚠ still helper-driven.** After W2 the target IR LOWERS (exit 0) but the emit is
produced by `emitFlatBlockCore` firing off the loop op — it computes its OWN
`block_base_x` + `vle8_v_i8m2` and ERASES the typed chain (incl. the new per-block
load). W4 (lowering-walks-chain) is the next + load-bearing wall: it decides
chain-driven (genuine) vs helper-driven (cosmetic). The per-block load capability
is built but not yet wired into the emit.

---

## GAP-1 / W4 — lowering 走链 (chain-driven / op-by-op)  [TORN 2026-07-04]

**命名.** After W2 the loop body LOWERED but the emit was `emitFlatBlockCore(descriptor)`
gated on the chain — attribute-derived, region ops only *checked*. A gate and a
lowering fail delete-mutation IDENTICALLY (advisor + verify agent), and "byte-exact
vs the monolith emit" is CIRCULAR (matched because it *was* the monolith emitter).
Neither proved construction.

**拆法 (op-by-op, operand-driven).** Rewrote the brick2 branch of
`emitTypedFlatBlockDotLoopBody` (`RVVToEmitCBlockQuantLinear.cpp` ~:6154-6477) to
hand-emit each region op FROM ITS OPERANDS — no `emitFlatBlockCore` call:
per-block LoadOp -> addr `base+ib*block_stride+quant_byte_offset` from its own
operands + vle8; WideningProductOp -> `vwmul(valueMap[lhs],valueMap[rhs])`;
StandaloneReduceOp -> vwredsum(input); extract -> vmv_x_s -> sumi; brick2/3 ->
fused fold (sumi from extract, d_x/d_y from brick1). block_base memoized on
(buffer,block_index) so brick1+load share one emission.

**复测 (operand-flow, not delete-mutation, not byte-exact-vs-monolith).**
M1 (isolating): swap widening_product operands -> emitted `vwmul(%27,%32)`->`(%32,%27)`
(main agent re-confirmed). M4 (kills attr-derivation): attr-only stride 34->68 with
load operand at 34 -> EXIT 1 fail-closed (a shell would emit 68). M3 (load-only)
also fail-closed. M2: stride/quant flow into address literals, per-load independent.
Zero regression: 352 green.

**归因.** No path WALKED a typed chain in a loop-body region sourcing emit from the
region SSA. That walk is the loop layer that was never built — W2's missing
capability one level up (emission, not typing).

**⚠ two disclosed residues (structural constants, do NOT undermine the arithmetic
chain):** (1) reduce SEED hardcodes literal-0; `%zero_seed` operand not read —
semantically forced (fresh per-block 0), a MODELING wart (op needs a
runtime_abi_value seed, flat loop's seed is literal 0); pre-flip cleanup candidate.
(2) inner block-cap vl = `qk` literal (scheduling constant). byte-exact-vs-monolith
DIVERGES only on setvl structure + provenance; per-block ARITHMETIC core is
byte-identical. Numerical bit-exact vs scalar oracle = pending-hardware.

**Stage: 靶 IR parse ✓ verify ✓ lower ✓ (op-by-op genuine).** Remaining: 拓真实 nb
(multi_block_factor) → q8_0 首翻 constructed (+ resolve seed residue) → 五格 cohort
→ 删 monolith.

---

## GAP-1 / P2c — q8_0 typed-flat per-block kernel = PARITY vs ggml factory (非 Win-B) [OPEN → P2c 关闭]

**命名.** schedule-参数化相 step 3 宪法双板重测(snapshot fc1dd132,k1 VLEN256 board_fp
0ccffb5f9130967a / rvv VLEN128 7040412c20700942,固频钉核 T-N IQR<0.02%):我方 SEL-1-选中
per-block 核 vs ggml **出厂派发** q8_0 核(faithful non-packed TU + Zfh 硬件 fcvt,两侧同板同 clang
同旗标——两个侦察 confound[strawman packed opponent + fp16 libcall]已 objdump 证伪并移除,旧
"1.21×/1.43× win" 是 artifact 非真赢)= **两板 PARITY**:VLEN256 我方 m1 = **1.004×** factory、
VLEN128 我方 m2 = **1.019×** factory(过 2×地板仅因 min-统计地板~0.01%,经济上 parity)。

**Triage([K-6] 四选一)= 缺模式(missing pattern),非选择错误/非缺能力事实/非带宽墙.**
反汇编证据(objdump-verified,fold_isolation_k1.txt):
- **能力机制真有效**:winA 纯 LMUL m1=3689ns vs m2=4096ns = **+11.0% 寄存器填满增益**(m1 VLMAX_e8m1@256=32=1 块满填;m2 半填 32/64)——SEL-1 先验选中 m1 是真机制赢。
- **但被 correctness 约束成本抵消**:factory 对我方 shipped m2 的 +10.6% edge = **+8.4%(我方 step-1a 钉死 no-FMA 折叠成本:m2 no-FMA 4096 → m2-FMA 3777)** + 2.0%(vsetvli/sched),精确 1.084×1.020=1.106。
- **净**:register-fill(+11.0%)≈ factory 的 fold+sched edge(+10.6%)→ m1 ≈ factory。
- roofline_class = **latency-bound**(cache-resident,per-block 串行 reduce→vmv.x.s→标量 fmaf 链,非 DRAM 带宽墙)。

**根因 = 家族级 per-block 串行折叠瓶颈(VLEN-无关,block-quant 全家族共享)+ 钉死 no-FMA 折叠
的 +1 fp-op/block 延迟成本**。m1-with-FMA 能打赢 factory 但破 pinned-oracle bit-exactness——
故 **no-FMA 成本是显式数值契约(layer-3)的【实测代价】**,不是缺陷。

**关闭动作 = P2c(step 5):deferred-ordered 折叠**(相A 批量整数+打包、相B vfcvt→两次 vfmul→
vfredosum.vs 有序种子归约,bit-exact vs **同一** §1 钉死 oracle)——把 per-block vector→scalar→
fmaf 跨域往返消掉、strip k 有序归约与 strip k+1 整数相重叠,目标从 sum(int,fold) 逼近
max(int,fold)=地板,**在 bit-exact 前提下**回收 +8.4% 折叠成本 + 兑现 +11.0% fill 增益 → 真 Win-B。
关闭后同板复测、attribution 引用本 gap。

**证据素材**:layer-1(机制选中+可归因 winA +11.0%)已证;layer-3(数值契约实测成本 +8.4%)已量;
layer-4(家族串行折叠瓶颈命名 + fold 成本反汇编分解 + 同 .o 双板 parity 展品)已立;layer-2(P2c
insight×coverage)= step 5。填 T3_A/T3_B(micro_vs_factory=parity,status=measured)+ T-N 地板;
T8 台账首条 loss/parity 实例(→ P2c 关闭)。**诚实定位:parity 对 latency-bound kernel 是物理确认
非失败;真 Win-B gated on P2c。无 beat 措辞(过 [PERF-1] 八门前)。**
