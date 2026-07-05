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

### F3 — L2③-M2 整数 CORE 进 region（诚实增量，C_construct 停 13）
新 operand-driven 砖 `repack_lane_wise_q4_x_i8_dot`(建模 super-block Q4KScaledDotOp);monolith 整数 CORE 抽共享 leaf `emitRepackQ4LaneWiseIntegerCore`、monolith+region 都调(byte-identical,monolith lit 444/444)。反绕闸三门 fail-closed + operand-driven 正向(verify 自跑 mutation:offset 32→99/2→77 传进 emit 地址字面量)。零回归 444/444、forced clean rebuild(clean 322 文件)。六态停 dispatch-wired、C_construct 13。+607/−238。scaffold_note "M1+M2 LANDED"。**M3(flip)剩**:fold 进 region + 泛化 VLEN128 numHalves==2 多累加器 + RVV0.7 f32m4 + full-body byte-exact + 退役 monolith + **repack front-door 构造(非 test-authored=constructed 关键 bar)** + flip → C_construct 13→14 + E5 强路由。

### F4 — L1 seal:q4_0@128 prefill SEALED 5.08×（★机制=routing+locality 非 codegen）
全能力 march(rv64gcv_zfh_zvfh_zb*... = /proc/cpuinfo 全 codegen 扩,readelf 对象级双树 byte-identical)、preflight 4/4 **gate-1 advisory CLOSED**、fp16 libcall 0/0 双树(混淆缺席第三证)、DVFS 0.00% 锁频、correctness greedy-token GREEN。**prefill pp128 5.077× [CI 5.039,5.111] DIFFERENCE**(verify 独立重跑 5.15× 复现;stock 未随全 ISA 提速=真赢非残废对手;★live 重测击败 static≠runtime 顾虑)。**decode range 1.50–1.56×**(3 压力快照,honest range 非单点)。
**★★机制修正(verify 权威,self-report 误标已纠)**:赢=**capability-keyed dispatch/ROUTING + memory locality,非 codegen 优越**。tiled `ggml_gemm_q4_0_8x8` prefill 核**双树 byte-identical(upstream 码)**→非我方-vs-ggml codegen 胜;赢是【我方编译器 ROUTE q4_0 到 repack@VLEN128,stock 留 block-dot(repack VLEN-gated/TODO)】+ repack 内存局部性。**decode GEVM(q4_0_16x1)是我方编译器发射**;prefill tiled GEMM 是 ggml 的、被我方 path-selection 路由。**诚实 claim="我方 path-selection engages repack 得 5×",非"更快 repack 核"=option-2 path-selection novelty([[option2-path-selection-real-pass]])。** 归因基=ENGAGED 运行时诊断+5×gap+static objdump;**perf PMU 0 样本(RISC-V PMU paranoid=2)→ sampled 热点 profile 是剩余升级。**
**sealed=单板 rvv/VLEN128 prefill;非完整 Win-B(措辞未出"Win")。剩 8 门**:(a)k1/VLEN256 dual-board(q4_0 应 parity=VLEN-flip 故事)(b)micro↔e2e Amdahl 传导(gate④)(c)prefill M-shape sweep + q8_0 T6 首批(d)PMU sampled 归因升级。
**板状态(返程 FYI)**:A/B 双树重建到全能力 march(seal artifact);板 ggml CMakeLists MARCH_STR override,备份 .bak-l1seal(一 cp 可还原)。

### F5 — L1-phase-2 dual-board flip：k1/VLEN256 q4_0 → prefill PARITY 确认 + decode SURPRISE（不 commit）
**板/工具链（非结构墙）**：k1 可达 = SpacemiT X60 **VLEN256** 8c@1.6GHz governor=performance、glibc2.39、clang-18.1.8；march `rv64gcv_zfh_zvfh_zicbop_zihintpause`（板 ISA 多 zba/zbb/zbs/zvfhmin/ime → gate-1 ADVISORY，**A/B 同 march 对称**、gate-2 libcall-clean 双树）。q4_0 模型已在板（`/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf` sha `da3087fb14aede55`=rvv 同模型）。**A 树已存**（`/data/k1build` patched，VLEN256 decode 已发射 emitted GEVM）；**B 树自建**（`/data/k1build-stock`，surgical out-of-place relink：仅 patch-off `arch/riscv/repack.cpp` 的 GEMV `==256` 分支重编 repack.o + relink，其余 .o 复用 A、~2min、**未碰 A 树**、源已还原）。B 执行体 = LD_LIBRARY_PATH wrapper（RUNPATH 是 abs、patchelf 缺）。
**preflight 4/4**（gate1 advisory / gate2 libcall-free 双树 / gate3 same-clang-18+flags / gate4 **board VLEN=256==target**）。DVFS span 0.00%。
**结果（paired 2pass×5rep，median+IQR+CI+T-N floor）**：
- **prefill pp128(GEMM)：ours 24.77 / stock 24.67 = 1.0043× [1.0018,1.0056]，|Δ|0.43% < 2×floor0.58% → PARITY**（zero-hypothesis）。
- **decode tg32(GEVM)：ours 5.68 / stock 6.62 = 0.8574× [0.8492,0.8653]，|Δ|16.6%≫floor → DIFFERENCE（我方 emitted GEVM 慢 ~14%）**。
**机制（objdump）**：A GEMV 有 1 call `tcrv_emitc_ggml_repack_gemv`（emitted 接合）、B GEMV 0（跑 ggml native 16x1 repack fallback）；**GEMM 两树反汇编 287 行仅差 1 条（stderr GOT 偏移，banner 用）→ prefill compute byte-identical，双侧都 = ggml native repack GEMM**。correctness greedy-token A==B 3/3 GREEN、无 NaN/Inf（未 claim bit-exact vs ggml）。
**★flip 判读**：**prefill flip 确认**——VLEN128 SEALED 5.08× **不在 VLEN256 复现**（ggml 自己 repack q4_0 GEMM、GEMM 码 byte-identical）→ 赢键控在 **VLEN128 capability gap（ggml repack VLEN-gated）非 universal**。**decode 是 REVERSAL 非 parity**：VLEN128 decode 1.5× 赢是 repack-vs-block-dot；VLEN256 变 **emitted-repack-vs-ggml-native-repack、我方 emitted（construction-line scaffold）反慢 14%**。两相都印证核心命题（赢=routing-into-gap 非更强核）。**decode 慢 = surprising**：最可能 = emitted VLEN256 GEVM scaffold 欠优化 vs ggml 手调 native（且 vs 早期手写 repack；memory 历史 parity 0.997× 是 pre-emitted-kernel）→ **需 adversarial verify**（park）。
**board identity 钉清**：k1 SpacemiT VLEN256，**cross-board 不可比 rvv/VLEN128**。产物 `experiments/e2e-harness/results/k1-vlen256-q4_0-flip/{preflight,phase_split_raw,correctness,aggregate,evidence.json}`。harness 两处泛化（experiments/ 内）：preflight gate-1 march 兜到 compile_commands.json、gate-4 VLEN 通用取数 + decode 探针（触 GEVM banner）——**additive、不破 rvv/128**。**未 commit（用户提交）**；未碰 lib//ODS/schema/test。

### F6 — L2③-M3 fold 进 region（诚实增量非 flip;q4_0 repack=7 里程碑链）
M3 落"fold 进 region"(numHalves==1 臂 full-body byte-exact,消 M2 stub):新砖 `repack_dual_fp16_scale_fold`(vle16 scale/vfwmul/vfcvt/vfmacc 抽共享 leaf `emitRepackDualFp16ScaleFold`、monolith+region 都调=by-construction byte-identical)+ 反绕闸/数据流绑门。**未 flip**:C_construct 停 13(canonical coverage_metrics.py 证)、schema/e5 零 diff。零回归 forced clean rebuild 269 targets、789/792。**q4_0 repack→constructed = 7 里程碑链**:M1(loop shape)M2(整数 CORE)M3(fold)已落;**剩 Phase B**(泛化全臂:整数砖 $result→Variadic、numHalves accs、去 2 emit-gate[:2166 numHalves≠1 / :2177 coreLmul≠mf2]、full-body byte-exact 全臂 + 加 empirical region-vs-monolith byte-diff 测[verify flagged:M3 byte-exact 是 by-construction 非实测、且 contingent on offset 0])→ **C**(front-door RVVLowerQuantContraction.cpp:246 改产 region)→ **D**(退役 GgmlRepackGemvQ40Q80Op+emitRepackGemvQ4_0Q8_0)→ **E**(flip C_construct 13→14 + e5 q4_0 repack 强路由 token+PATHS + provenance)。决策:续推(L2 优先、链已投 3 workflow、设计已 map);Phase B → C-D-E 两 workflow 到 flip。

## 返程问题包（park 项，累积）
- **P1（F5 park）**：k1/VLEN256 q4_0 **decode emitted-GEVM 慢 ggml-native-repack 14%**（0.857×，稳定 paired、非测量混淆）——adversarial verify：①是否 emitted VLEN256 GEVM scaffold 欠优化（vs 早期手写 repack + vs ggml native）②B 的 native 16x1 fallback provenance（upstream ggml vs tcrv 自写 reference C；板 git 空、未证）——prefill parity 与此无关（GEMM byte-identical、铁）。
- **P2（q4_K e2e 扩展决策）**：repack 战役 #2 格 q4_K 可行(无结构墙)但**是赌**——**q4_K repack MICRO 是 LOSS 0.55×/0.74× vs block-dot**(selector 正为此 decline),e2e 赢是【memory-locality 在 e2e 放大压过 micro loss】的**假说非证据**(q4_0 micro 是 1.22× 赢起点、q4_K 从 loss 起点),**可能落 loss 非 Win-B**。setup 实(A 树 q4_K repack 未 wire:emit .inc[emitter 已建]+填 stub+Patch-A@disp:4619+q8_K 激活 ABI[block_q8_Kx4]+rebuild+A/B on DeepSeek-8B-Q4_K_M[rvv 已有];k1 parity leg 需传 q4_K 模型)。素材:rvv 有模型、in-compiler op+emitter+lit 已建、q4_0 harness 模板全复用。**决策=pursue(赌一把、赢=强 2nd 数据点[harder 对手+micro-loss 仍 e2e 赢=强 thesis];输=诚实 bound campaign)还是 skip/defer(押 q4_0 flip+L3 IQ)?** research: q4k-e2e-repack-feasibility.md。未自决=不在常备令显式清单 + 结果真未知 + 非关键路径 → 用户 steer。
