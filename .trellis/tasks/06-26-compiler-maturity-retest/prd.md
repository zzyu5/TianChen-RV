# compiler 成熟化 + 重测填完 kernel 表

> 状态：brainstorm — 全量分解已成（8-agent understand+decompose+red-team），用户已 converge 为 **FULL scope（全做）**。子任务已 scaffold。
> **本对话边界**：这次对话只建 PRD/task 结构，**不写实现代码**；实现由后续 session 走 trellis-implement/trellis-check 推进（impl 代码不在 main session 直接改）。
> 源：`doc/项目现状与方向-诚实版.md`(v3) + `doc/KERNEL-优化自查表.md`(v4)。

## Goal

把 TianChen-RV 从现状——**能力驱动"选择"已成熟（5 砖 board-sealed）、但指令体逐 kernel 手写 + cost-model 1-bit + N2 完整 IME GEMM 空 + 自查表大量格子未实测**——推进到**可辩护的成熟度**，并**最终重测+填完 `doc/KERNEL-优化自查表.md`**（每格带证据状态标，重构后旧数字一律重测）。

## Success Definition（诚实分级——避免 sign-off 层 over-claim）

成熟度有两腿，**不可混淆**：
- **腿(1) L4-MECHANISM（=parity/regression-removal）**：按真 VLEN fact 自动 lower 成不同正确形状，天花板 = 追平 ggml 自己的 `_vlN` 形状。`wa1–wa5` + memoization 属此腿。**这不是 beat headline。**
- **腿(2) row-② BEAT（=真成熟 headline）**：**综合一个 ggml 没手写的 within-kernel 形状**（wide-LMUL / VLEN-tuned strip / multi-accumulator）。**唯一来源 = `cm4`/`cm5`**（"copy-then-adapt、multi-day、timed-out-2×"emitter 类，坐 Win-C-NULL + decode-washout 两颗雷上）。**`wa1` 单独不交付 beat headline。**

> ⚠ 红队硬话：若把 "MVP recommended" 读成"拿下了成熟/beat headline"，就是 over-optimism 在 sign-off 层复发。MVP 只交付腿(1)。

## 方向（doc §8 已拍板，不重新 litigate）

N3 = 行①追平专家(parity) + 行②系统性 beat(within-kernel 形状综合，**不是** repack/option-2)。主线 = L4 形状综合 + cost-model 真资源 fact + IME-GEMM；L0–L3(byte-exact) 作前置基建并行铺。repack/算法选择 = 前端贡献，唯一保留 = capability-driven DECLINE。

## 纪律护栏（每个非-byte-exact 子任务 DoD 必须 enforce）

- perf 报数前先过 **byte-exact gate**（vs 独立 scalar oracle **且** vs ggml 出厂 kernel）；byte-exact 用 **forced clean rebuild + glob BEFORE==AFTER**（ODS .inc 每次重生、tcrv-opt 有时不重链，**绝不增量**；不用 stale 绝对指纹 [[build-incremental-unreliable]]）。hw/perf 主张要真板证据（rvv=SG2044/VLEN128、k1=X60/VLEN256 harts 0-3、ime=X60，I8）。
- 口径：Win-A(自消融 ON/OFF) + Win-B1(vs block-dot, gap-fill=前端) + Win-B2(vs ggml 自己 repack, 成功=parity)，各 ×{micro,e2e}；**每格证据状态标** `{measured|presumed|board-pending|N/A-by-construction}`；未测默认 **NO claim**。
- **micro win ≠ e2e win**；decode = per-block reduction 延迟受限（非带宽）；micro/e2e 分开报。**parity 非 banked**——重构后该格 STALE、必重测。
- must-NOT：repack/option-2 升后端 novelty / 发明"第三类" [[backend-maturity-triton-reframe]]；bank "推定 parity" 当成功；声明 2–5× tune（真值 1.0–1.43×）；把 ggml-spacemit 的 GEMM 当我们 IME win；称 N1 flip 是 transform pass；把 L0–L3 byte-exact 清理当成熟。

---

## 子任务全量分解（34，按 phase；DAG 见下）

> 计数已复核（2026-06-26）：bare-`__riscv_` literals **764**、CallOpaqueOp **607**（9 emitter 文件 in `lib/Conversion/RVV/`）；GEMM 注册表确认 **有** Gemm/RepackGemm `{Q40,Q41,Q4K}`、**无** Q80Q80 / Q50Q80（q8_0/q5_0 prefill 须先 build op）。TunableScheduleOpInterface adopters 实现时复核（≥6，已扩到 q1_0/tq2_0/tq1_0/iq2_xxs）。

### Phase 0 — no board / no perf-cell invalidation（可立即起）
| id | title | 性质 |
|---|---|---|
| `doc-bank-null-audit` (CM-1) | 审计 doc/表里"推定"格子，标回 NO-claim | 诚实闭环 |
| `ime-prefill-corrigendum` (N2-C) | IME e2e 只测过 decode-heavy，补 prefill-regime 口径（保留 M=1 decode 对照） | 诚实闭环 |
| `spec-status-update` (§8⑥) | §9.1–9.4 构想纳入 spec 现状口径（SC-list 须覆盖全部行+§9.1 三 qualifier+§9.2 三轴；填 SC-02/04/07 编号空洞） | spec |
| `tf-routing-audit` | **硬前置**：确认每个 e2e 重测里 ggml 在该板实际派发哪个 kernel（k1 派 `_vl256`、VLEN128 q4_0 无 repack） | 前置 |
| `emitVCall-canary` (L0a) | `emitVCall` helper + KQuant 3 lambda 转用它（最硬 canary） | L0 第一砖 |

### Phase 1 — byte-exact infra（no board，并行，不动任何 perf 格）
| id | title | 备注 |
|---|---|---|
| `emitVCall-consolidation` (L0b/c) | 补 ~10–12 widening mangler + 全量 sweep（764 bare literal/607 CallOpaque 趋零） | XL，glob BEFORE==AFTER 全 RVV lit |
| `derive-widening-chain` (L1) | 单一真值源，**修潜伏 m1/m2 bug seam**（KQuant:686 vs :1597） | **需 m2 discriminating 单测**（{mf2,m1} byte-exact 不够） |
| `blockdot-facts-handle` (L2) | emitter 侧 fact 读句柄（legality 权威仍在上游） | 去重便利，非 blocker |
| `per-family-emitter-tables` (L3) | forward/dequant/masked-store 12 个 `if(is*Body)` → per-family 表（BlockDot 表已成，35 entry） | 顺序 load-bearing |
| `trackB-production-export` (WA-6) | production 路 3 面 cascade（RouteFamilyCommon:310/PlanOwners:509-626/Validation:478）接 byte-anchor body | **仅 D5=production 时** |
| `rvv07-fractional-prune` (CM-3) | RVV0.7 无 fractional-LMUL 防御性剪枝 | low-pri / defer-eligible |

### Phase 2 — selection-quality + L4 bricks（board，**非** byte-exact，invalidate 触及格）
| id | title | 天花板/风险 |
|---|---|---|
| `memoization-default` (CM-2) | live 路明确"实测记录优先、否则静态 argmin"（保留诚实 caveat：是 default 非 active search） | 机制澄清 |
| `wa1-nibble-widelmul-q4` (WA-1) | q4_0/q4_1 nibble-decode wide-LMUL mf4→m1 block-dot（**MVP 砖**） | **ceiling=parity/regression-removal，非 beat** |
| `wa2-nibble-widelmul-q5` (WA-2) | q5_0/q5_1 同 | 可能仍 LOSS |
| `wa3-iq-gather-lmul` (WA-3) | iq1_s/iq1_m/iq3 gather-LMUL（**先解 signs64 op-attr**，没做#8） | dep: op-attr 扩展 |
| `wa4-repack-halflanes` (WA-4) | repack `half_lanes` VLEN-aware auto-select（现手 stamp） | **MECHANISM-only，无 beat/e2e/第三类** |
| `cm4-vlen256-widecover` (CM-4) | **VLEN256 wide-cover——真 row-② 候选** | **HIGH 风险（腿2）** |
| `wa5-iq4xs-foldback` (WA-5) | FP4 iq4_xs fold-back 砖 | parity |
| `cm5-kquant-multilmul` (CM-5) | **K-quant 多-LMUL EMIT——真 row-② 杠杆**（≥2 个不同指令的 LMUL 形，宽形必须实测更快否则=NO claim） | **HIGH/multi-day（腿2）** |
| `cm6-acc-lmul` (CM-6) | multi-accumulator LMUL | defer-eligible，**先 pre-board 可行性证明** |
| `c3-silu-softmax-vcpop` (C3) | 给共享 exp 多项式 emit `vcpop`-gated 快路径（silu/softmax 0.84×→parity） | approaches-parity |

### Phase 3 — N2 IME + substrate
| id | title | 备注 |
|---|---|---|
| `n2-a1-int8-gemm-micro` (N2-A1) | 从 tcrv emit 真 IME int8 GEMM micro（**tag 我们-emitted 非 ggml-spacemit**；Win-B2=parity 天花板） | 无 dep，可并入 Phase 2 |
| `n2-a3-quant-decode-ime-gemm` (N2-A3) | quant-decode IME GEMM（**multi-week**；per-arm NMSE gate；parity+disclosed-null 天花板，受 K1 7GB 内存墙界定） | **硬 dep a1，仅 D4=d** |
| `sc10-silicon-probe` (SC-10) | 真硅片 probe→capability ingestion（现喂 march-derived fact，**N1 真洞**） | perf-free，两板 |
| `sc13-hart-gate` (SC-13) | hart/core 并行 gate 正确性 | dep sc10 |

### Phase 4 — 表重测+填（deliverable closure；按 per-cell L4-touch 门控）
| id | title | gate |
|---|---|---|
| `tf-micro-fills` (A7+A10+A6) | 残留 micro 格（**带 per-cell L4-touch 映射**） | 触及格须在对应砖后 |
| `tf-prefill-reseal` (A4) | q4_0 GEMM prefill 5.68× compiler-emitted 复测盖章 | after memoization-default |
| `tf-presumed-e2e` (A2) | q8_0/q4_K GEVM e2e·rvv、q4_K GEMM e2e·rvv（**预期 LOSS 非 parity**） | board |
| `tf-blockdot-e2e-column` (A3) | ~24 block-dot e2e（只测过 q4_K，**其余非 default-NULL**） | **board-budget-gated（重）** |
| `tf-forward-e2e` (A8) | 6 前向算子 e2e | after c3 |
| `tf-trackB-seal` (A9) | Track-B 板封印（iq4_nl/iq2_xxs VLEN256 k1） | gated by D5 |
| `build-q80-gemm` (B1) | 造 Q80Q80 GEMM op（**现无 → q8_0 prefill 无 op 可测**） | **仅 D4≥b**，否则 N/A-by-absence |
| `build-q50-gemm` (B2) | 造 Q50Q80 GEMM op（q5_0 compute-bound→建了多半也 LOSS） | **仅 D4=c** |
| `tf-memwall-8b` (A1) | **8B@VLEN128 on rvv = 内存墙真判官**（最干净 runnable 重测） | **board-budget-gated（重）** |

## 依赖 DAG（关键边 + per-cell L4-touch 门控 R1）

- Phase 0 全部可立即起；`tf-routing-audit` 是所有 e2e/presumed/memwall/prefill-reseal 的**硬前置**。
- Phase 1 byte-exact，**不阻塞** WA 砖（m1 是 in-tree 路；L1 只在某砖 stamp m2 时才相关；L2 只省重复）。
- Phase 2 砖并行；`cm5`/`cm6` **复用 `cm4` 的 wide-LMUL emitter 技术**（soft dep，先做 cm4）；`cm6` 须先 pre-board 可行性证明。
- Phase 3：`n2-a1`→`n2-a3`(硬 dep)；`sc10`→`sc13`(硬 dep)。
- **Phase 4 per-cell L4-touch 门控（必须在对应砖落地后、重测，不准沿用旧数字）**：
  q4_0/q4_1 ← `wa1`；q5_0/q5_1 ← `wa2`；iq1/iq3 gather ← `wa3`；q4_K-repack k1 micro ← `wa4`；选定 K-quant 格 ← `cm5`；VLEN256 wide-cover 格 ← `cm4`；silu/softmax ← `c3`；q4_0 GEMM prefill reseal ← `memoization-default`；q8_0/q5_0 prefill ← `build-q80`/`build-q50`（否则 N/A-by-absence）。**其余无 L4-touch 的格 Phase 2/3 即可并行填。**

## 执行排序——第一增量（**非** scope 上限；scope=全做）

实现起步时的第一增量：`doc-bank-null-audit` + `ime-prefill-corrigendum` + `spec-status-update` + `tf-routing-audit` + `emitVCall-canary` + `wa1`(rvv+k1 MICRO)。这是**排序建议**，不缩小范围——全 34 项都在 scope 内。
**⚠ 第一增量交付腿(1) MECHANISM；row-②-beat headline 由 `cm4`/`cm5` 在后续增量交付（已授权，HIGH 风险，landing=TARGET 非 banked）。**

## Out of Scope（本轮）/ 已确认 done（不重做）

- 实际推进/写实现代码（本轮只建 task）。
- option-2 C3 真 pipeline 自动化（深活，doc §9.3 keep-frontend）；frontend linalg/tosa；discrete-card offload；JIT/runtime tuning。
- 已 done 不重做：N1=substrate/三轴 + evidence-status 轴 + Win-B1/B2 + "VLEN flip≠pass" 的 **spec 重写已落地**（commit bcde9b89/2361d4d8/5102eaa6，红队已核实在 `.trellis/spec/`，非仅 commit message）。

## Decision（ADR-lite）— RESOLVED：FULL SCOPE（用户 2026-06-26 converge）

**Context**：用户要"真实做一个完成成熟化的 task"；本对话只建 PRD/task 结构、不写实现代码。
**Decision = 全做**：全 5 phase / 34 子任务 IN。
- **D-beat = YES**：`cm4`/`cm5`（row-②-beat 杠杆）**授权**——这是拿"成熟 compiler / 反超 ggml"headline 的唯一一类。
- **build + IME = 全**：`build-q80-gemm` + `build-q50-gemm`（解锁 q8_0/q5_0 prefill 格；现确无 Q80Q80/Q50Q80 GEMM op，code-verified）+ `n2-a3` multi-week quant-decode IME GEMM 全 IN。
- **board/e2e = 重构改变过的全测（micro+e2e）**：按 per-cell L4-touch map，每个被砖触及的格子 micro **且** e2e 重测；最重的判官 `tf-memwall-8b`(8B@VLEN128) + `tf-blockdot-e2e-column`(~24) IN。
- **D5 = production-export IN**：`trackB-production-export` 做，使 Track-B e2e 经 production 路可达、seal 耐久。
**Consequences（诚实）**：大型多-session 工程；HIGH-risk 杠杆（`cm4`/`cm5`/`n2-a3`）坐 Win-C-NULL/decode-washout/infra-timeout 风险上——**landing = 诚实证据（parity 或 disclosed-null 或 board-pending），beat headline 是 TARGET 非 banked outcome**；任何"宽形必须实测更快否则=NO claim"。每次重构后触及格 STALE、必重测。

## 子任务映射（已 scaffold；8 workstream children）

| child task | 覆盖原子项 | phase | board | 风险 |
|---|---|---|---|---|
| [[06-26-phase0-doc-spec-closure]] | doc-bank-null-audit, ime-prefill-corrigendum, spec-status-update, tf-routing-audit | 0 | no | 低 |
| [[06-26-emitter-l0-l3-infra]] | L0a canary, L0b/c emitVCall, L1 widening-chain(修m2bug), L2 facts-handle, L3 per-family表, CM-3 rvv07-prune | 1 | no(byte-exact) | 低（≠成熟） |
| [[06-26-winA-parity-bricks]] | wa1-5, memoization-default(CM-2), c3-silu/softmax-vcpop | 2 | board | 中（ceiling=parity） |
| [[06-26-row2-beat-levers]] | cm4, cm5, cm6 | 2 | board | **HIGH（腿2 headline）** |
| [[06-26-n2-ime-gemm]] | n2-a1, n2-a3(multi-week) | 3 | k1/ime | **HIGH（decode-washout）** |
| [[06-26-substrate-probe-hart]] | sc10 真硅片 probe, sc13 hart-gate | 3 | both | 低（perf-free） |
| [[06-26-gemm-op-builds-tooling]] | build-q80/q50-gemm（Phase4 board/perf）, trackB-production-export（Phase1 byte-exact tooling） | 1+4 | no(export)/board(builds) | 中（解锁格；builds 非 byte-exact） |
| [[06-26-table-retest-fill]] | tf-micro-fills, tf-prefill-reseal, tf-presumed-e2e, tf-blockdot-e2e-column, tf-forward-e2e, tf-trackB-seal, tf-memwall-8b | 4 | board | 中-重（deliverable closure） |

> 执行顺序：Phase0 + emitter-infra 立即起 → WA 砖 / cm 杠杆 / n2 / substrate 并行（按 DAG）→ build/tooling 解锁 → **table-retest-fill 收口**（per-cell L4-touch 门控）。

## Technical Notes

- 关联 memory：[[backend-maturity-triton-reframe]] [[n1-substrate-emission-not-maturity]] [[kernel-wins-dont-transplant-to-e2e]] [[winc-structural-null]] [[option2-path-selection-real-pass]] [[repack-winA-always-mf2]] [[emitter-maturity-vluxei16-widelmul]] [[k1-ime-n2-hardware-candidate]] [[build-incremental-unreliable]]。
- 全量 work-item code-ground（file:line）见 understand-workflow 产出 `tasks/w9vcapvwk.output`（emitter L0-L4 / cost-model / IME / table-fill / Win-A / scaffold 六片 + 红队核验）。
- scaffold 时复核：bare-literal 764、interface-adopter ~12、SC 编号空洞。
