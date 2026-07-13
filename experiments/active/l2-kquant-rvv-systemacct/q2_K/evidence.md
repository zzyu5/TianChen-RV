# [L2-KQUANT-RVV-SYSTEMACCT / q2_K] — 三账并列: 系统账(clang deploy) · 对称-gcc账(算法本体) · k1 kernel账

> board `ssh rvv` openEuler VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · disjoint-pin vs vLLM(0,1)
> G6 令·二 (最便宜绿候选池·★新绿候选·非升级) · A-tree `/home/ubuntu/tcrv-llamacpp` · **NO git · board reversible**
> 承 q4_K 系统账模板 (commit b2531fe1). q2_K cell 现 **黄-对手更强** (schema IDX 18); 翻绿【完全依赖系统账】.
> **绿/黄裁量收归主会话** (裁决张力精化): q2_K 与 q4_K 结构同构, 但 q2_K 无既绿 (k1+对称 gcc 双 LOSS)·风险更高.

## ★★ 三账并列 (裁决张力·framing 铁律·必读) — 系统账 = 被撤回 micro 1.413× 的换轴复述
[CASE-COMPILER-ASYMMETRY]: 本格 **系统账** (我方 clang .o vs stock gcc-15 .o) 与 schema IDX 18 里被撤回的
**micro 1.413×** (曾作 kernel-axis 赢·"clang-ours-vs-gcc-shipped artifact"·已 CLOSED) 是【**同一物理比较**】,
只是换轴报 (系统账 vs kernel账). **framing 铁律**: 若系统账 ≥parity, 只能表述为
「**部署账赢 · gcc-death 机制 · 非 kernel/算法赢**」, 且【**必须同披露对称-gcc LOSS + k1 kernel LOSS 0.873×**】,
否则=重新兜售已撤回的 artifact. 三账全列 (下表), 缺一即违 framing 铁律.

| 账 | 我方 repack 编译器 | 对手 | 轴 | prefill | decode |
|---|---|---|---|---|---|
| ① **系统账** (headline) | **clang-18** (我方出货 [L-10]) | stock gcc-15 block-dot | deployment vs deployment | **0.857× LOSS** | 0.516× LOSS |
| ② **对称-gcc账** (算法本体·诚实底数) | **gcc-15** (对称) | stock gcc-15 block-dot | kernel==system (对称) | **0.497× LOSS** | 0.150× LOSS |
| ③ **k1 kernel账** (clang 对称·引用·不重测) | clang-18 (k1 出货) | k1 真出货 hand-brick (NOT block-dot) | kernel==system (对称) | 0.873× LOSS | 0.875× LOSS |

**★核心结论: 三账全 LOSS·q2_K@rvv 系统账【也】LOSS (0.857× prefill·非 ≥parity) → 【不翻绿】·维持黄-对手更强.**
与 q4_K 的决定性区别: q4_K clang 账 **治好 gcc-death 且反超 block-dot** (0.334×→1.344× WIN); q2_K clang 账
**治好大半 gcc-death 但仍输 block-dot** (0.497×→0.857×·spill 2937→6 治愈但不足反超). 我方 q2_K emitted 全展开
27KB S6-tiled 核在 clang 部署域【仍慢】stock block-dot ~14% —— 与 k1 kernel-axis (our-emit 输 hand-brick 0.873×)
【同量级同方向】·**跨双板 (rvv/VLEN128 clang-deploy + k1/VLEN256 clang-sym) 一致的真 kernel-quality LOSS·非 gcc artifact**.

## [1] 路由确认 (route confirmation) — 与 q4_K 结构同构
| 端 | 变体 | md5 | q2_K 路由 | 编译器 |
|---|---|---|---|---|
| OFF (stock as-shipped) | q4kOFF.gcc | 05a62e6a | generic `ggml_vec_dot_q2_K_q8_K` **block-dot** (GEN case128 `{break;}//TODO` → 不激活 repack; 运行时实证 block-dot 符号 ×4) | gcc-15.2.0 |
| ON-a (系统账 NUMER) | q2kON.clangrepack | 0b01c53a | 我方 emitted **VLA repack** GEMM(S6-tiled)+GEVM (gate128 flip + intercept·spill 6/22·vsetivli16=0 VLEN128-safe) | clang-18.1.8 |
| ON-b (对称账 NUMER) | q2kON.gcc | b27bd686 | 同 emitted repack·**gcc-15 编** (spill 2937/218 = gcc-death) | gcc-15.2.0 |
- emitted kernel = M2-q2_K-k1-e2e export (weft_emitted_gemm_q2_K.inc md5 d9dee831 VLA S6-tiled + gevm ee76a4d3)·VLA→VLEN128-correct.
- q2_K@VLEN128 stock 与 q4_K 完全同构: case128 `{break;}` → block-dot (无 compiler-symmetric "repack vs repack" stock 对照·stock RVV q2_K 硬调核只 case256 fires).

## [2] 部署五验 + 反向控制 (raw/deploy_5verify_reverse_control.txt) = GREEN
| 验 | OFF(stock-gcc15) | ON(ours-clang18) | 反向控制(OFF-again) |
|---|---|---|---|
| v1 md5 | 05a62e6a PASS | 0b01c53a PASS | 05a62e6a PASS |
| v2 nm weft_sym | 0 PASS | 2 PASS | 0 PASS |
| v3 strings banner | 0 PASS | 2 PASS | 0 PASS |
| v4 objdump | 无 emitted 符号; block-dot `ggml_vec_dot_q2_K_q8_K` ×4 | spill GEMM=6 GEVM=22 · vsetivli16=0/0(VLEN128-safe) | 无 emitted 符号 |
| v5 runtime engage | — | 12 hits "ENGAGED n=4096 nr=16 nc=128" | 0 (kernel 真下线) |
反向控制双向干净: swap OFF→ON 引擎点亮(12)、ON→OFF 引擎熄灭(0).

## [3] correctness GREEN (硬门·前置)
- **rvv/VLEN128 greedy A==B byte-identical** (本会话 fresh·raw/correctness_gate.txt): A(ours clang emitted VLA repack,
  straight make_block) == B(stock gcc block-dot) **byte-identical 两 prompt** · engage=13(>0) · nan=0 · pass=2 fail=0
  → **CORRECTNESS_GATE: GREEN** (timing-independent byte-compare·contention-immune·VLA 核 VLEN128 byte-exact).
- **ZERO-MODEL bit-exact-integer** (M2-q2_K-k1-e2e · kquant_repack_verify_q2K · silicon INT 独立 oracle): 同 emitted export
  (d9dee831) INT_mismatch=0 全 8 shapes (GEVM nr=1 + GEMM nr∈{4,8,16}·双 seed)·NORM worst_ulp=3402 (dual fp16 d/dmin 路).
- **MIRAGE 排除**: stock block-dot (`ggml_vec_dot_q2_K_q8_K`) = VLEN128 canonical 正确路径; 我方 straight make_block_q2_Kx16
  匹配 emitted 核 decode leaf (两自洽对给 byte-identical 输出)·非空心 oracle.

## [4] 分相 e2e — 三账 (DeepSeek-8B-Q2_K · taskset -c 8-15 -t 8 · pp128/tg16 · 交织 · DVFS 2.6GHz 锁 · 4 passes/side · REPS=2 · raw/measure_3way_parse.txt)
| 相 | stock(gcc block-dot) t/s | ours-clang(repack) t/s | ours-gcc(repack) t/s | 系统账 clang/stock | 对称账 gcc/stock |
|---|---|---|---|---|---|
| **prefill pp128** | **5.030** (relIQR 0.10%) | **4.310** (relIQR 0.33%) | **2.502** (relIQR 0.17%) | **0.857× LOSS** | **0.497× LOSS** |
| decode tg16 | 3.285 (relIQR 2.27%) | 1.694 (relIQR 0.57%) | 0.492 (relIQR 0.14%) | 0.516× LOSS | 0.150× LOSS |

- **relIQR 干净 (disjoint-pin 净测·非 env-blocked)**: prefill 全项 relIQR 0.10–0.33% (headline 相极稳)·decode 0.14–2.27%
  (stock decode 2.27% 系单低样·不改 LOSS 定性). 4 passes 全程 load 11.7–12.1 = **我方 8-thread bench 自身负荷** (cores 8-15)·
  freq 2.6GHz 全程锁. 主会话巡检见 load-12·经核 = 我核 bench (vLLM 在 cores 0,1·disjoint)·relIQR 0.1–0.3% 即证无越核污染
  (若 vLLM 突入我核·relIQR 必炸). **测 4 clean passes 后按收敛指令即停** (未续到 8·因结论 LOSS 稳健·n=4 relIQR 已足判).
- **gcc-death 归因在档**: EXACT 部署 kernel objdump spill **q2kON.gcc GEMM=2937 GEVM=218** vs **q2kON.clangrepack GEMM=6 GEVM=22**
  = ~490×(GEMM)/~10×(GEVM) 整-向量-寄存器 spill 差 ([CASE-COMPILER-ASYMMETRY])·比 q4_K(742/4)更极端.
- **对手身份**: OFF = stock as-shipped **gcc-15.2.0** block-dot (板出货编译器) · ON-a = ours **clang-18.1.8** emitted repack
  (我方出货形态 [L-10]) · ON-b = ours **gcc-15** emitted repack (对称账).
- **★板污染处置**: 邻居 vLLM qwen3-32b TP2 (Worker_TP pin cores **0,1**·98%CPU steady·`taskset -pc` 实证 affinity 0/1)·
  测量核 cores 8-15 仅 idle kworker (`ps psr` 实证·无 non-mine 进程). start-load-gate (5.0) 系 disjoint vLLM 过计→抬至 8.0·
  非我核污染. disjoint-pin + relIQR 是真守卫 (承 q4_K 同处置·不重启他人 job).

## VERDICT (预注册出口 = 意外分支「q2_K rvv 也 LOSS」命中)
**系统账 q2_K@rvv (我方 clang-18 emitted repack 正门部署 vs stock as-shipped gcc-15 block-dot):**
- **Prefill = 0.857× LOSS (< parity)** → **预注册意外出口命中**: 「若 prefill <parity (意外·q2_K rvv 也 LOSS)→ 如实报
  (黄·带账·系统账 LOSS 归因)」. **q2_K【不翻绿】· 维持黄-对手更强.**
- **Decode = 0.516× LOSS** (照预着色黄·VLEN128-recon + kernel-quality).
- **三账全 LOSS (framing 铁律遵守·无 win 可售·天然免疫重售撤回 artifact)**:
  ① 系统账 clang-deploy 0.857×/0.516× LOSS · ② 对称-gcc 0.497×/0.150× LOSS · ③ k1 kernel 0.873×/0.875× LOSS.
- **gcc-death 真但不足**: 对称 gcc 0.497× → 系统 clang 0.857× (spill 2937→6 治愈·prefill +0.72×改善) 但**未达 parity** ——
  区别 q4_K (clang 治好 gcc-death 后反超 block-dot·1.344× WIN). q2_K emitted 核在 clang 部署域【仍慢 stock block-dot】~14%.
- **跨双板一致的真 kernel-quality LOSS**: rvv/VLEN128 clang-deploy 0.857× ≈ k1/VLEN256 clang-sym 0.873× (both our-emit vs
  竞品·both clang·~0.86×). 我方 q2_K 全展开 27KB S6-tiled 核【本身】比 stock 慢·非编译器 artifact·[GAP-KQUANT-GCC-CODEGEN]
  的 codegen 部分被 clang 消除后剩余仍 LOSS = emitter-quality gap (紧凑 rolled loop 对标未达·新杠杆未具名).
- **correctness GREEN** (fresh rvv/VLEN128 greedy A==B byte-identical + ZERO-MODEL k1 INT bit-exact + MIRAGE 排除) ——
  干净 perf LOSS·非 broken kernel.
- **系统账铁律遵守**: 此 prefill 0.857× 是**系统账** LOSS·非 kernel 质量单独声明·三账全披露·如实标.
- **板 restore = TRUE** (live 05a62e6a OFF-pristine·source GEN deb61a29/ARCH 99131cf7 byte-exact·0 q2_K stray .inc·0 straggler)·**NO git**.

### schema label 建议 (改留主会话·我只报建议)
- cell IDX 18 `{op:gemm_tile, format:q2_K, engine:rvv}` **维持黄-对手更强·perf-covered 计数不变** (本格 LOSS·非绿候选兑现).
- **建议更新 symmetric_account_pointer / 加 rvv/VLEN128 系统账数据点**: 「rvv/VLEN128 **系统账** (ours clang-18 emitted repack
  正门部署 vs stock gcc-15 block-dot·DeepSeek-8B-Q2_K e2e·4 passes relIQR<0.5% prefill): **prefill 0.857× / decode 0.516× LOSS**;
  对称-gcc 账 (ours gcc-15 repack vs stock gcc block-dot·同 harness·fresh e2e) **prefill 0.497× / decode 0.150× LOSS** (复现/更新
  旧 l1-t3-q2k 0.386× 底数·同向 LOSS·数值差因 harness/model 不同·本次系 fresh DeepSeek-8B-Q2_K systems-harness e2e);
  gcc-death spill 2937→6 治愈但不足反超·q2_K emitted 核 clang 部署域仍慢 block-dot ~14%·与 k1 kernel-axis 0.873× 同量级·
  **跨双板一致 kernel-quality LOSS·非 gcc artifact·CLOSES rvv/VLEN128 fix-hypothesis (同 k1 已 closed)**」.
- **★q4_K vs q2_K 对照 (强档案素材)**: 同构对局 (both ours-clang-repack vs stock-gcc-block-dot·rvv/VLEN128 系统账)·
  **相反结果** (q4_K 1.344× WIN / q2_K 0.857× LOSS) = clang 消 gcc-death 后·kernel-form 决定成败 (q4_K emitted 核反超·
  q2_K emitted 核不足). 印证 [CASE-COMPILER-ASYMMETRY]: gcc-death 是【必要非充分】·消除后仍须 emitter-quality 达标才 WIN.
- **framing 铁律天然满足**: 本格无 win → 不存在"部署账赢误当算法赢"风险·三账全 LOSS 一致披露·不涉重售撤回的 micro 1.413× artifact.
