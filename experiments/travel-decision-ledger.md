# 出行期自主裁决 ledger（2026-07-05 起，用户离线）

常备令：三线并行（L1 T6 harness / L2 harness-free 预备 / L3 coverage IQ）；文件集互不相交才并行、交集串行；遇 fork 按规则自决不 ping；绝不自决=开第四线/放宽正确性门/canon-schema.def 级/删活消费者（park 入返程包）；STOP 仅 3 种（结构死墙/双板全不可达无替代/falsifier 红因不明）→ 切他线永不闲置。返程交付=一页 checkpoint + 问题包。

自决规则序：①触红线?→选不触;②在关键路径?→选在的;③可逆?→直接做;不可逆两案皆合规→选证据多+记此 ledger;仍平局→park+推他线+入问题包。

---

## Decisions

### D1 — 三线并行拓扑（触碰集 diff）
**判定**：L1(touch=`experiments/` harness + 只读 export) 与 L2 prep(touch=`RVVContractionPathSelection.cpp`+`RVVCapabilityProfile.h`) **交集空 → 并行**。L2③ q4_0 repack construction + L3 IQ construction **都触 ODS/front-door/schema six-state/e5_strong_readout + 可能 ContractionPathSelection → 交集非空 → 串行**（L2 prep 完 → L2③ → L3；L1 全程并行，disjoint）。
**规则**：③可逆(diff 预计)+关键路径。已 dispatch L1(a8aa18fc)+L2(ae2581d6)。

### D2 — 构造线串行（ninja-lock 现实，非仅源文件不相交）
**判定**：源文件集不相交是并行【必要非充分】条件——子代理共享主树 `build/`，并发 `ninja` 撞 build-lock（ninja 单进程锁，第二个报 busy）+ ODS `.cpp.inc` 每次重生（见 [[build-incremental-unreliable]]）。故【本地 build 型构造线并发上限=1】。安全拓扑：**L1(harness，主要跑硬件、本地 build 前置一次) 全程并行 + 构造线(L2prep→L2③ repack→L3 IQ)串行**（一次一个本地 build）。L2③/L3 共享 ODS/six-state/e5 本就要串（D1），叠加 ninja-lock 只强化。
**规则**：①红线无②关键路径（避免 build 竞态毁构造）③可逆。此刻不派 L3——L2③/L3 正当串行等 L2 prep 完，非闲置违规（STOP-永不闲置 针对【线撞墙】，非【正当串行等待】）。

### D3 — 对手图核正 + 目标 ggml 版本（自决，非 park）
**事件**：L2 fact-fix 发现【侦察的"事实错"本身反了】。侦察说"K-quant 无 riscv vec_dot、q4_K@128=scalar_fallback";我亲读 `arch/riscv/quants.c`(6596 行,2026-06-15)证：q2_K/q3_K/q4_K/q6_K 各有 VLEN-特化 inline-asm vec_dot，q4_0/q8_0 intrinsic，仅 q5_K plain-intrinsics 无分派。**roster 无一格 scalar——全手调对手。** selector 原 `ggmlHandTunedVLENNativeExists(Q4_K,128)==true` 其实对。已修至地面真值(commit ee9be42a)+ 更正 memory [[repack-campaign-terrain]]。
**决策(自决,规则③可逆+关键路径)**：目标 ggml = **当前树**(板上跑的、诚实基线),非更旧 generic-K-quant 版。理由:[L-7] BASELINE=真 ggml factory;板跑当前树;beat 更旧 generic 是弱/不诚实 claim。**不 park**(可读文件解、非用户独有决策)。**战略含义**:VLEN128 Win-B=repack-vs-手调-block-dot(更强 claim、非弱带易赢);q4_0 STALE 5.68× 若重测 holds=vs-手调 强赢。战略不变、opponent 刻画更准。
**FYI 入返程**:对手全手调=Win 门槛更高、但赢的 paper claim 更强。

## 关键发现（findings，入返程 checkpoint）

### F1 — L1 T6 harness 建成 + q4_0@128 bring-up 基线（重大正面，status=bring-up 非 sealed）
harness(`experiments/e2e-harness/`)端到端跑通:preflight 4/4、prefill/decode 分相 llama-bench 原生、paired A/B 同会话、greedy-token 正确性门 GREEN(未 claim bit-exact vs ggml)。**注入机制=板上双构建树 A(tcrv-llamacpp 编译器发射 repack)vs B(upstream 出厂 block-dot),同 gcc-15.2.0/rv64gcv-O3/REPACK=ON、唯一差补丁=干净对称。** q4_0@128(rvv/VLEN128,tinyllama,固频 2.6GHz):**prefill pp128(GEMM) 4.99× DIFFERENCE [4.98,4.99] floor 0.02%**、decode tg32(GEVM) 1.55× DIFFERENCE。STALE 5.68× 重测 clean≈5×，**验证 repack 战役核心命题(VLEN128 prefill ~5× e2e vs ggml block-dot)**。
**advisory(未 sealed)**:①march=rv64gcv 漏板 zfh/zvfh/zb*——gate-2 objdump 证两侧对称无 fp16-libcall(非残废对手、P2c 混淆缺席),但 sealed 需全能力 march 重建;②decode 内存受限+共享 64c 板压力敏感(轻载 2.0×→重载 1.55×)、封须多快照。**Win 措辞未出(八门未过、bring-up 态)。** 消费者:①repack e2e 就绪已产数 ②gate④ 半接(需 Amdahl 传导会计)③T6 cell 结构就绪(bring-up 态不污染 sealed 模板)。

### F2 — L2③ q4_0 repack GEVM 构造 = 诚实 scaffold M1（非 flip，C_construct 停 13）
q4_0 repack→constructed 是**真多里程碑**(repack GEVM 结构 ≠ 两个已有 loop op)。M1 落:新 typed op `tcrv_rvv.typed_repack_gemv_loop_body`(+yield)= **第三个 distinct typed-loop shape**（per-strip **lane-wise f32 VECTOR loop-carried accumulator**，无 cross-lane vredsum、整 16-lane strip 直写 vse32；flat 是 scalar-f32、super-block 是 horizontally-reduced）。byte-exact=**skeleton-scope**(loop-nest + 累加器骨架 vs monolith one-strip 形，结构经 lit CHECK 验；region CORE 是 stub=flat step-1 同法)。**C_construct 停 13、六态停 dispatch-wired、e5 未动=诚实未 flip(未 fake)。** +610 LOC 全正、零回归 170/170、forced clean rebuild、无 ninja-lock 撞(L1 seal 用板 gcc、不撞本地 build/)。
**剩余到 flip**:M2=lane-wise 整数 CORE 进 region(operand-driven + block_index 反绕闸)；M3=fold 进 region + 泛化 VLEN128 numHalves==2 多累加器 + RVV0.7 f32m4 + full-body byte-exact vs monolith + 退役 emitRepackGemvQ4_0Q8_0 + 真 front-door 构造 → 才 flip +1。

### D4 — L2③ 后接 M2（续 repack 构造，非切 L3）
**判定**:standing-order 优先级 L2(repack、喂 L1)> L3(rolling coverage);M2 续 M1 的 typed_repack op(同文件、承接结构)。故 L2③ 后接 **M2**(parallel L1 seal——local 构造 vs 板 seal 不相交、无 ninja 撞)、L3 IQ 仍串后。规则②关键路径(repack 是主线)。

## 返程问题包（park 项，累积）
（暂空）
