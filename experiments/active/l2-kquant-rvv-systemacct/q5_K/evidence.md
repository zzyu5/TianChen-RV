# [L2-KQUANT-RVV-SYSTEMACCT / q5_K] — 三账并列: 系统账(clang deploy) · 对称-gcc账(算法本体) · k1 kernel账

> board `ssh rvv` openEuler VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · disjoint-pin vs vLLM(0,1)
> G6 令·二 (完成 L2 三格 + gcc-death 必要非充分律【第三数据点·tiebreaker】) · A-tree `/home/ubuntu/tcrv-llamacpp` · **NO git · board reversible**
> 承 q4_K(b2531fe1·WIN)/q2_K(14f4631a·LOSS) 系统账模板. q5_K cell 现 **绿** (schema cell[20]·via k1/VLEN256 半).
> **裁量收归主会话**: q5_K 已绿 (k1 our-kernel 1.641× prefill)·故 rvv 系统账 WIN=dual-board 升级 / LOSS=律第三例.

## ★★ 三账并列 (裁决张力·framing 铁律·必读) — 系统账 = 换轴复述, 不是新 kernel 赢
[CASE-COMPILER-ASYMMETRY]: 本格 **系统账** (我方 clang .o vs stock gcc-15 .o) 若 ≥parity 只能表述为
「**部署账·gcc-death 机制·非 kernel/算法赢**」, 且【**必须同披露对称-gcc LOSS + k1 状态**】. 三账全列:

| 账 | 我方 repack 编译器 | 对手 | 轴 | prefill | decode |
|---|---|---|---|---|---|
| ① **系统账** (headline) | **clang-18** (我方出货 [L-10]) | stock gcc-15 block-dot | deployment vs deployment | **0.8688× LOSS** (n4·relIQR 0.28%) | 0.6522× LOSS |
| ② **对称-gcc账** (算法本体·诚实底数) | **gcc-15** (对称) | stock gcc-15 block-dot | kernel==system (对称) | **~0.120× LOSS** (schema/T9·fresh repro env-blocked·见 [4b]) | (~12.9× reversal·gcc-death 极端) |
| ③ **k1 kernel账** (clang 对称·已绿·引用·不重测·k1 禁) | clang-18 (k1 出货) | stock q5_K block-dot (k1 ships ZERO q5_K repack·NOT hand-brick) | kernel==system (对称) | **1.641× WIN** | 0.729× roofline-loss |

**★核心结论 (预注册 <parity 出口命中): q5_K@rvv 系统账 = 0.8688× prefill LOSS (n=4·relIQR 0.28%) → gcc-death 必要非充分律【第三数据点】·q5_K 属 q2_K-class.**
- q4_K@rvv = 1.344× WIN (clang 治好 gcc-death spill 742→4 后【反超】block-dot·emitter-quality 达标).
- q2_K@rvv = 0.857× LOSS (clang 治好 spill 2937→6 但 emitter-quality 不足·未反超).
- **q5_K@rvv = 0.8688× LOSS** (clang 把 gcc spill 4124→155 大幅治愈但【仍未反超】block-dot·与 q2_K 0.857×【同量级同方向】).
- **★q5_K tiebreaker 解**: q5_K@k1 **WIN** 1.641× (VLEN256) 而 q5_K@rvv **LOSS** 0.87× (VLEN128) —— **同一 VLA 核·相反结果**.
  判别 = VLEN128 register-cliff: clang q5_K GEMM spill=155 (对比 q4_K clang spill=4)·**qh 5th-bit plane 残留 spill floor**·
  VLEN128 S6-tiled 核做 TWO 8-lane strips (VLEN256 只 1 strip). 故律【每板成立】: k1 核 fits→emitter-quality-beat→WIN;
  rvv 核 register-cliff→emitter-quality 不足→LOSS. **q5_K@rvv 归 q2_K-class (律第三例·emitter-quality 不足)**.

## [1] 路由确认 (route confirmation) — 与 q2_K/q4_K 结构同构
| 端 | 变体 | md5 | q5_K 路由 | 编译器 |
|---|---|---|---|---|
| OFF (stock as-shipped) | q4kOFF.gcc | 05a62e6a | generic `ggml_vec_dot_q5_K_q8_K` **block-dot** (q5_K dispatch NEON-only `q5_K_8x8/8x4` under has_neon → nullptr on riscv → block-dot; verify: block-dot 符号 present=2·emitted sym=0) | gcc-15.2.0 |
| ON-a (系统账 NUMER) | q5kON.clangrepack | 53b2d3b0 | 我方 emitted **VLA S6-tiled repack** GEMM+GEVM (net-new riscv 1,16 scaffold·case128 flip·spill 155/97·vsetivli16=0 VLEN128-safe) | clang-18.1.8 |
| ON-b (对称账 NUMER) | q5kON.gcc | a381efbf | 同 emitted repack·**gcc-15 编** (spill 4124/454 = gcc-death 极端) | gcc-15.2.0 |
- emitted kernel = kernel-axis t4a export (VLA·__riscv_vsetvl_e32m1 avl=8 f32m2·VLEN≥128-valid): gemm ba30ba54 (=tcrv_emitted_gemm_q5_K.inc·S6-tiled) + gevm c445b89e. 同 k1 M2 部署核 (vwmacc gemm=2240/gevm=560 匹配 k1 seal).
- q5_K rvv scaffold NET-NEW (rvv A-tree ships q2_K/q4_K 1,16 repack 但无 q5_K·deploy_patch_q5_K_rvv.py 加全 scaffold·block_q5_Kx16 stride2816 qh@256·make_block_q5_Kx16 straight interleave == q4_K body + qh plane).

## [2] 部署五验 + 反向控制 (raw/deploy_5verify_reverse_control.txt) = GREEN
| 验 | OFF(stock-gcc15) | ON(ours-clang18) | 反向控制(OFF-again) |
|---|---|---|---|
| v1 md5 | 05a62e6a PASS | 53b2d3b0 PASS | 05a62e6a PASS |
| v2 nm tcrv_sym | 0 PASS | 2 PASS | 0 PASS |
| v3 strings banner | 0 PASS | 2 PASS | 0 PASS |
| v4 objdump | 无 emitted 符号; block-dot `ggml_vec_dot_q5_K_q8_K` present=2 | spill GEMM=155 GEVM=97 · vsetivli16=0/0(VLEN128-safe) | 无 emitted 符号 |
| v5 runtime engage | — | 4 hits "ENGAGED n=4096 nr=16 nc=128 **vlen=128**" | 0 (kernel 真下线) |
反向控制双向干净: swap OFF→ON 引擎点亮(tcrv_sym 0→2·banner 0→2·engage 4)、ON→OFF 引擎熄灭(全 0). engage banner vlen=128 (VLA 核 VLEN128 正确).

## [3] correctness GREEN@bounded-ULP (硬门·前置)
- **ZERO-MODEL bit-exact-INTEGER** (PROVEN·独立 oracle): kernel-axis t4a (k1/VLEN256 ULP0 int / 8e-7 norm vs 独立 scalar q5_K
  dequant-matmul oracle) + M4-q5K full cert (cert-lineage·REAL dmin!=0·同 ggml_quantize_mat_q8_K 喂两侧·INTEGER MAIN/MIN
  0-mismatch·我方 repack-GEMM CLOSER to int-exact fold than ggml generic). 我方 emitted VLA 核 integer fold VLEN-无关 (同 avl=8 归约序) → carries to VLEN128.
- **fresh rvv/VLEN128 greedy A vs B** (raw/correctness_gate.txt): MIX = byte-identical (p2 "Once upon a time…") + bounded-ULP
  **coherent divergence** (p1: A 进 reasoning trace·B 直答 Paris·both coherent·单 early argmax flip cascade). q5_K our-repack fp-fold
  ≠ block-dot fold → argmax-敏感 DeepSeek-R1 部分 prompt cascade. **与 q5_K@k1 M2 同标准 (3/4 byte-id + 1/4 coherent-flip)**·非 strict
  byte-identity (区别 q4_K/q2_K 的 shared min-fold byte-identical). engage vlen=128·nan=0.
- **MIRAGE 排除**: both variants coherent reasoning output (layout/ABI bug → garbage / ppl 数百千 / NaN·cf vl16 PPL 822057). q5_K
  real dmin≠0 + qh 5th-bit plane 双 exercised (block_q5_Kx16 stride2816·qh@256). → **CORRECTNESS_GATE: GREEN@bounded-ULP.**
- (fresh rvv ppl 数值 anchor 尝试但 ppl batch nr=512 使 S6 核 VLEN128 pathologically slow>8min·env·abort+restore; k1 M2 ppl
  ON 17.9683±3.809 vs OFF 17.9095±3.785 Δ0.33% within±3.8 = argmax-immune 数值 anchor·同 VLA 核/同 fold·stands.)

## [4] 分相 e2e — 三账 (DeepSeek-8B-Q5_K_M 5.7GB · taskset -c 8-15 -t 8 · pp128/tg16 · DVFS 2.6GHz 锁 · raw/measure_lean_parse.txt)
[PHASE-A systems clang-vs-stock 4 passes + PHASE-B symmetric gcc-vs-stock 2 passes·-r1·lean 因 5.7GB 模型每 bench 慢~7min]

| 相 | stock(gcc block-dot) t/s | ours-clang(repack) t/s | 系统账 clang/stock | ours-gcc(repack) t/s | 对称账 gcc/stock |
|---|---|---|---|---|---|
| **prefill pp128** | **1.7307** (relIQR 0.20%·n4) | **1.5035** (relIQR 0.28%·n4) | **0.8688× LOSS** | [见 4b·env-blocked] | ~0.120× (schema/T9) |
| decode tg16 | 0.7865 (relIQR 0.15%·n4) | 0.5130 (relIQR 0.92%·n4) | 0.6522× LOSS | [见 4b] | [schema/T9] |

## [4b] 对称-gcc 账 (Phase B) — fresh e2e repro ENV-BLOCKED by gcc-spill4124 pathological runtime · 引 schema/T9 0.120×
- fresh pp64 gcc-vs-stock repro 尝试 (q5k_sym_gcc.sh): **q5kON.gcc (spill 4124) e2e bench 未完成** —— gcc pass 1 pp64
  bench 运行 >10min 仍无输出 (gcc-death spill 4124 → 运行速率 ~0.06× stock·pp64 单 bench 估 >22min)·遂 abort + restore.
  **此 runtime pathology 本身 = gcc-death 二次佐证** (spill 4124·bench 慢到测不动). 板已 restore (不占资源).
- **disclosed 对称-gcc 底数 = schema/T9 q5_K@rvv 对称-gcc 0.120× LOSS** (~12.9× reversal). 与 q4_K/q2_K 一致方向 (q4_K 0.334×·
  q2_K 0.497×·q5_K 0.120× = q5_K gcc-death 最极端). **gcc-death 机制在本会话【直证】= EXACT 部署 kernel objdump spill
  q5kON.gcc GEMM=4124/GEVM=454 vs q5kON.clangrepack GEMM=155/GEVM=97** (build_seal.txt·~27× GEMM spill 差·clang flip
  把 gcc 0.120× 提到 0.8688× = 治愈大半 gcc-death 但未反超·[CASE-COMPILER-ASYMMETRY]).

- **系统账 4-pass 收敛** (Phase A·clang vs stock·pp128/tg16·-r1·interleaved): **prefill 0.8688× / decode 0.6522× LOSS**·
  clang prefill 1.5013–1.5061·stock 1.7268–1.7312·relIQR prefill 0.28%/0.20%·decode 0.92%/0.15%·**最差 pass 仍 0.868× LOSS**
  (稳健·n=4). 4 passes 全程 load 11.5–12.1 = 我方 8-thread bench 自负荷 (cores 8-15)·freq 2.6GHz 全程锁·vllm%cpu 0.0/0.0
  (Worker cores 0,1 disjoint·净测·非 env-blocked·relIQR<1% 即证无越核污染).
- **对称-gcc 账 (SYM·gcc spill4124 e2e·pp64 因 pp128 gcc-bench ~25min env-pathological)**: 见 [4b] + VERDICT·复现 schema/T9 0.120×.
- **gcc-death 归因在档**: EXACT 部署 kernel objdump spill **q5kON.gcc GEMM=4124 GEVM=454** vs **q5kON.clangrepack GEMM=155 GEVM=97**
  = ~27×(GEMM) spill 差 ([CASE-COMPILER-ASYMMETRY])·比 q2_K(2937/6)的 gcc 更极端·但 clang q5_K spill 155 ≫ q2_K/q4_K clang(6/4)
  = **qh 5th-bit plane register-cliff 残留·clang 亦未清零** (区别 q4_K min-fold clang 治到 4).
- **对手身份**: OFF = stock as-shipped gcc-15.2.0 block-dot (板出货编译器) · ON-a = ours clang-18.1.8 emitted repack (出货形态 [L-10]) · ON-b = ours gcc-15 emitted repack (对称账).
- **★污染处置**: vLLM qwen3-32b TP2 (Worker_TP pin cores 0,1·disjoint) 与测量核 8-15 disjoint·pass load 载中我核 bench 自负荷·
  freq 2.6GHz 全程锁·relIQR 拒污染 (非 env-blocked). [注: 5.7GB Q5_K_M 每 bench ~7min·both clang(1.50) 与 stock(1.73) 皆慢·
  非 ON 核独慢·memory-heavy 大模型板特性].

## VERDICT (预注册出口 = <parity 出口命中·q5_K 属 q2_K-class·律第三例)
**系统账 q5_K@rvv (我方 clang-18 emitted repack 正门部署 vs stock as-shipped gcc-15 block-dot):**
- **Prefill = 0.8688× LOSS (< parity·n=4·relIQR 0.28%/0.20%·最差 pass 0.868×)** → **预注册 <parity 出口命中**: 「prefill <parity →
  gcc-death 必要非充分【第三例】·q5_K 属 q2_K-class·emitter-quality 不足·如实黄带账」. **q5_K cell 维持绿 (via k1 半)·rvv/VLEN128 半
  = 系统账 LOSS·NOT dual-board 升级.**
- **Decode = 0.6522× LOSS** (照预着色黄·VLEN128-recon + kernel-quality·block_q5_Kx16 stride2816+qh 多带宽 for nr=1).
- **三账披露 (framing 铁律遵守)**: ① 系统账 clang-deploy **0.8688×/0.6522× LOSS** · ② 对称-gcc **~0.120× LOSS** (schema/T9·fresh repro
  env-blocked·gcc-death 极端·见 [4b]) · ③ k1 kernel **1.641× WIN / 0.729×** (引用·已绿·kernel账·NOT 系统账).
- **★律第三数据点 (tiebreaker 解)**: gcc-death 必要非充分 = 消 gcc-death (spill 4124→155) 后仍须 emitter-quality-beat 才 WIN.
  q5_K@rvv clang 大幅治愈 spill 但**未反超** block-dot (0.87×·同 q2_K 0.857× 量级) —— gcc-death 必要非充分【第三确认】.
  **★关键新识**: q5_K@k1 WIN(1.641×·VLEN256) vs q5_K@rvv LOSS(0.87×·VLEN128) = 同 VLA 核相反结果·判别 = qh 5th-bit plane
  在 VLEN128 的 register-cliff (clang spill 155 vs q4_K 4·2-strip)·[GAP-Q5K-VLEN128-QH-REGCLIFF]. 律【每板成立】.
- **correctness GREEN@bounded-ULP** (ZERO-MODEL INTEGER bit-exact 独立 cert + fresh greedy byte-id/coherent + MIRAGE 排除) —— 干净 perf LOSS·非 broken kernel.
- **系统账铁律遵守**: 此 prefill 0.8688× 是**系统账** LOSS·非 kernel 质量单独声明·三账全披露·如实标.
- **板 restore = TRUE** (md5 三证·见 [5])·**NO git**.

## [5] board restore + md5 三证 = GREEN
- **cert-1 (live .so OFF-pristine)**: md5=05a62e6a8285 · tcrv_q5k_sym=0  PASS (测后每步 restore·final live OFF-pristine)
- **cert-2 (source 3-file byte-exact baseline)**: GEN=deb61a29 · HDR=57851439 · ARCH=99131cf7  PASS (build script restore + pristine rebuild 闭环)
- **cert-3 (无 q5_K stray .inc)**: tcrv_emitted_*q5_K*.inc = 0 (build trap rm·ARCHDIR 干净)  PASS · 0 campaign 遗留进程
- 板改动仅测中 swap LIVE .so (ON/OFF·测后 restore OFF-pristine)·source 3 文件全程 pristine·全程 NO git.

### schema label 建议 (改留主会话·我只报建议)
- cell[20] `{op:gemm_tile, format:q5_K, engine:rvv}` **维持绿 (via k1/VLEN256 半·any-board 判绿)·perf-covered 计数不变** (本 rvv-half LOSS·非新绿/非升级).
- **建议 quality_note 补 rvv/VLEN128 系统账数据点**: 「rvv/VLEN128 **系统账** (ours clang-18 emitted q5_K repack 正门部署 vs
  stock gcc-15 block-dot·DeepSeek-8B-Q5_K_M e2e·n=4·relIQR<1%): **prefill 0.8688× / decode 0.6522× LOSS**; 对称-gcc 账 (ours gcc-15
  repack) ~0.120× (schema/T9·fresh repro env-blocked by gcc-spill4124 bench); gcc-death spill 4124→155 治愈但不足反超·与 q2_K 0.857×
  同量级·**gcc-death 必要非充分律
  第三数据点**·q5_K 属 q2_K-class (emitter-quality 不足). ★q5_K@k1 WIN(1.641×) vs q5_K@rvv LOSS(0.87×) 同核相反=qh 5th-bit plane
  VLEN128 register-cliff [GAP-Q5K-VLEN128-QH-REGCLIFF]·clang spill 155 vs q4_K 4」.
- **★q4_K/q2_K/q5_K rvv 三格对照 (强档案素材·完成 L2)**: 同构对局 (ours-clang-repack vs stock-gcc-block-dot·rvv/VLEN128 系统账)·
  q4_K 1.344× WIN / q2_K 0.857× LOSS / **q5_K 0.8688× LOSS** = gcc-death 必要非充分律三点定. 消 gcc-death 后·kernel-form (VLEN128
  register 压力·qh plane) 决定成败.
- **framing 铁律天然满足**: 本格 系统账 LOSS → 无 win 可售·不涉重售撤回 artifact·三账全披露 (k1 WIN 引用明标 kernel账-已绿·非系统账).
