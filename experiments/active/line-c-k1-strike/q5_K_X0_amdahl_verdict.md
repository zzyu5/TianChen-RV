# q5_K e2e [X-0] gate + Amdahl 传导预估 → VERDICT: 声明例外 (declared exception)

> **Line C · 段2 deliverable**. Pre-flight gate for the "k1 唯一新绿精准点" (q5_K e2e). Per canon 科研目标总纲
> [X-0] 三问 + Amdahl 上限. **correctness-first**: no e2e attempted (gate not passed on deployed-domain terms).
> **honest, NOT a failure narrative** (硬纪律 5): q5_K e2e = weight-reconstruction-bound micro↛e2e archival +
> bounded work-item. Board read-only this session; restore trivially clean.

## [X-0] 三问
**Q1 — 在热路径?** ✅ YES. q5_K quantizes all large weight tensors (attn qkv/o + ffn gate/up/down); the q5_K
`mul_mat` dominates prefill compute (~0.85 of prefill wall-time for a dense LLM, matmul-bound). On the hot path.

**Q2 — deployed-domain 同域 micro 证据?** ⚠️ MIXED — this is the decisive question.
- **kernel-axis micro (k1, t4a)** = `S6-repack-GEMM vs factory block-dot` = **1.916×** (K=2048 nr=64 nc=512,
  single-core prefill-GEMM proxy). **★NOW compiler-symmetric-VALID**: k1 stock ggml = clang-18 (CMakeCache proof,
  `opponent_map_k1.md` P1), our deploy = clang-18 ⇒ this is NOT the rvv clang-ours-vs-gcc artifact (the t4a
  "asymmetry" assumption is corrected). Opponent = a real VECTORIZED block-dot (`ggml_vec_dot_q5_K_q8_K` 13 rvv/15
  vsetvli, `opponent_map_k1.md` P5).
- **BUT the micro is a single-core prefill-GEMM PROXY, NOT the threaded `mul_mat` e2e input-path.** Canon rule
  (CLAUDE.md 性能常驻规则 4 + [CASE-COMPILER-ASYMMETRY]): the Amdahl kernel-factor input must be **同输入路径** (same
  deployed input path), not just same compiler.
- **★deployed-domain same-input-path K-quant e2e factor is EMPIRICALLY ESTABLISHED by the sibling q4_K** — same
  super-block class, same S6 repack scaffold, same correctness-carrier deploy pattern (`g5-wiring/M2-q4_K`):
  | model | prefill e2e | decode e2e |
  |---|---|---|
  | DeepSeek-8B-Q4_K_M | **0.4236×** | 0.1815× |
  | qwen2.5-0.5b-q4_k_m | **0.8400×** | 0.7916× |
  **全相 < parity ⇒ q4_K verdict = yellow, perf-covered Δ=0.** The kernel-axis→e2e gap for K-quant repack is
  empirically **catastrophic**: 1.884× micro (rvv-clang) → 0.42× e2e (rvv-gcc). The proxy over-predicts by >4×.

**Q3 — Amdahl 上限 vs 噪声地板?** (noise floor: rvv q4_0 e2e floor 0.25–2.5%, T-N_noise_floor.csv)
- **(a) LITERAL task formula** (proxy micro 1.92×, f=0.85): upper bound = 1/((1−0.85)+0.85/1.92) = **~1.68×** ≫ floor
  ⇒ pass-**on-paper**.
- **(b) DEPLOYED-DOMAIN formula** (correct per canon: same-input-path factor = sibling q4_K e2e ~0.42× threaded):
  q5_K is the SAME K-quant super-block class (even MORE weight-reconstruction: 5th-bit qh plane vs q4_K's 4-bit) ⇒
  deployed-domain e2e projection ≈ **0.4×–0.9× (≤ parity, likely LOSS)** ⇒ **WIN contribution = 0 < 噪声地板**.

## VERDICT — 声明例外 (declared exception)
The task's literal Amdahl (proxy 1.92×) passes on paper, but **the canon rule requires the deployed-domain
same-input-path factor**, which the sibling q4_K proves is ≤parity (LOSS) for K-quant repack e2e. Feeding the
single-core kernel-axis proxy into Amdahl is exactly the "garbage-in" the rule warns against (the q4_K sibling:
1.884× micro → 0.42× e2e = the proxy is not deployed-domain-valid). ⇒ **deployed-domain e2e WIN upper bound <
noise floor ⇒ 声明例外.** q5_K e2e is NOT a defensible new-green point; it is the **weight-reconstruction-bound**
yellow/LOSS class, same as q4_K. **NOT executed** (correctness-first; no deployed-domain path to a WIN).

**Registered as**: (1) **micro↛e2e archival** high-quality sample (kernel-axis 1.916× compiler-symmetric-VALID on
k1, but K-quant repack does not transduce to threaded e2e — the [[kernel-wins-dont-transplant-to-e2e]] law, K-quant
instance); (2) **声明例外** in the perf-covered denominator (declared-exception, zero undefined). **perf-covered
unchanged (6/84)** — no k1 new green (口径: FLAT-复刻/K-quant-yellow do not add headline).

## ★honest uncertainty / bounded work-item (do NOT overclaim the exception)
The sibling q4_K e2e LOSS was measured on **rvv-GCC** (deploy compiler gcc-15, root-caused to gcc 742-spill-death,
T-PERF1b). **On k1 the deploy compiler is clang-18** (which produced the GOOD micro codegen). So there is a
**GENUINE, UNMEASURED** possibility that k1-clang K-quant e2e avoids the gcc-death and transduces closer to the
kernel-axis. This does NOT rescue a q5_K WIN claim this session (unmeasured; heavy provisioning), but it IS the
sharp bounded work-item:
- **[WORK-ITEM-K1-KQUANT-E2E]**: measure a K-quant e2e on **k1-clang** to resolve **gcc-death vs
  intrinsic-reconstruction-overhead**. Cheapest probe = **q4_K** (model `/data/tinyllama-1.1b-Q4_K_M.gguf` EXISTS;
  wiring recipe exists `g5-wiring/M2-q4_K`). If q4_K prefill ≥ parity on k1-clang ⇒ q5_K becomes worth full
  provisioning (re-open [X-0]); if q4_K < parity on k1-clang too ⇒ 声明例外 is confirmed with real k1 evidence
  (intrinsic reconstruction overhead, not compiler).
- **q5_K-specific provisioning blockers** (why not done in-session): no q5_K gguf on k1, **no llama-quantize on k1**,
  no base f16 model (must requantize off-board + scp + sha256); **net-new e2e wiring** (q5_K has ZERO riscv repack
  branch → net-new trait/route/arch-skeleton per G5-M2 "q5_0/q5_1 建法"); needs a k1 ggml build tree distinct from
  Line B's `/home/bianbu/tcrv-k1-llama`. HEAVY — deferred, not abandoned.

## 段3 — decode 预着色 = 黄-物理墙 (roofline)
q5_K decode (GEVM, M=1) is memory-bandwidth-bound (weight streaming) + weight-reconstruction-bound. Sibling q4_K
decode e2e = 0.18×/0.79× (rvv). Expected k1 outcome = **parity / roofline-bound ⇒ 黄-物理墙 (roofline 证)**, NOT a
failure narrative (per task段3 + [[kernel-wins-dont-transplant-to-e2e]]: memory-bound decode does not transplant a
compute-account win). Not measured this session (gated behind the 声明例外).

## pre-registered 双出口 status (task措辞模板)
- **绿** (k1·q5_K·prefill·kernel账净绿·≥parity·双账本对称): **NOT reached** — deployed-domain gate not passed.
- **黄** (micro↛e2e 卷宗高质量样本): **the operative registration** — kernel账 micro 1.916× compiler-symmetric-VALID
  in案; e2e transduction ≤parity by sibling (q4_K 0.42×); Amdahl/objdump 指针 = T-PERF1b (gcc-spill) + M2-q4_K (e2e).
- **声明例外**: the STOP decision (deployed-domain WIN upper bound < noise floor), with [WORK-ITEM-K1-KQUANT-E2E]
  as the reopen trigger.

## durable files (this cell)
- `q5_K_X0_amdahl_verdict.md` — this gate + Amdahl (dual computation) + 声明例外 verdict + decode pre-color + work-item.
