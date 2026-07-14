# [L2-KQUANT-RVV-SYSTEMACCT / q4_K] — 系统账翻正: 我方 clang q4_K repack 正门部署 vs stock gcc-15

> board `ssh rvv` openEuler VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · load-gated
> G6 令·二 (最便宜绿格池首格) · A-tree `/home/ubuntu/tcrv-llamacpp` · **NO git · board reversible**
> **账本 = 系统账 (deployment vs deployment)** · ON = 我方 clang-18 emitted q4_K repack .o 正门部署
> (== 我方出货形态 [L-10] 设计事实) · OFF = stock 板出货 gcc-15 as-shipped block-dot
> 披露双方编译器身份 (我方 clang-18 / 板 shipped gcc-15) · 区别撤回的 kernel-axis clang-vs-gcc artifact

## 系统账 vs 撤回的 kernel-axis artifact (记账正当性)
[CASE-COMPILER-ASYMMETRY]: kernel 账 vs-opponent 数【仅编译器对称时有效】。故 decisive 实验里
"ours-clang-repack vs ours-gcc-repack"(同 block-dot 分母, 隔离 gcc 因素) 记为 internal-A/B 机制探针
([NG-4]·不计 perf-covered)。**本格 = 系统账**: 两边各用其真实出货编译器 —— 我方出货形态本就是
clang .o 正门链接([L-10]), 板出货形态是 gcc-15 as-shipped。这是**部署现实**(deployment vs deployment),
非 kernel 质量声明, 合法计 perf-covered。★注: q4_K 在 VLEN128 **不存在** compiler-symmetric 的
kernel-axis 备选(stock RVV repack 硬编码 AVL=16 = VLEN256-only, VLEN128 输出垃圾 PPL 822057), 故
"repack vs repack 同编译器" 物理上不可行 —— 系统账是唯一忠于真 baseline 的记法。
**系统账铁律(裁·七): kernel 账不得表述为系统收益; 此数是系统账, 如实标。[NG-4] beat 只从八门放行。**

## [1] 路由确认 (route confirmation)
| 端 | 变体 | md5 | q4_K 路由 | 编译器 |
|---|---|---|---|---|
| OFF (stock as-shipped) | q4kOFF.gcc | 05a62e6a | generic `ggml_vec_dot_q4_K_q8_K` **block-dot** (stock RVV repack VLEN256-only, @VLEN128 不激活) | gcc-15.2.0 |
| ON (ours front-door) | q4kON.clangrepack | 915e6733 | 我方 emitted **repack** GEMM(prefill)+GEVM(decode) VLEN128 vl=8 (case128 flip + intercept) | clang-18.1.8 |
- emitted kernel = g5-wiring M2 `tcrv_emitted_gemm_q4_K.inc`(6cbd9c19)+`gevm`(e909a9bd) · VLEN128-safe vl=8。
- 复用 decisive-kquant-gcc-vs-vlen /tmp/dkgv 内 md5-sealed .so 变体 (无 rebuild)。

## [2] 部署五验 + 反向控制 (raw/deploy_5verify_reverse_control.txt) = GREEN
| 验 | OFF(stock-gcc15) | ON(ours-clang18) | 反向控制(OFF-again) |
|---|---|---|---|
| v1 md5 | 05a62e6a PASS | 915e6733 PASS | 05a62e6a PASS |
| v2 nm tcrv_sym | 0 PASS | 2 PASS | 0 PASS |
| v3 strings banner | 0 PASS | 1 PASS | 0 PASS |
| v4 objdump | 无 emitted 符号; block-dot `ggml_vec_dot_q4_K_q8_K` 在(4) | spill GEMM=4 GEVM=8 · vsetivli16=0/0(VLEN128-safe) | 无 emitted 符号 |
| v5 runtime engage | — | 9 hits "ENGAGED n=4096 nr=16 nc=128" | 0 (kernel 真下线) |
反向控制双向干净: swap OFF→ON 引擎点亮、ON→OFF 引擎熄灭。

## [3] correctness GREEN (硬门·前置)
- **ZERO-MODEL bit-exact-integer** (t4b-m4-decisive · 同 emitter): REAL dmin!=0 Q4_K_M tensor
  (blk.0.attn_k.weight K=4096 N=1024, MIN-TERM ACTIVE 16384/16384), 同 ggml mat-quant q8 喂两侧 →
  **INTEGER BIT-EXACT MAIN mismatches=0 / MIN mismatches=0**。
- **bounded-ULP**: OURS vs int-exact oracle max_abs 7.6e-6 / **max_rel 2.86e-6**; ours vs ggml_generic
  max_rel 6.31e-5 (残差纯 FP reassociation) —— **OURS 比 ggml 自己的 generic 更贴近 int-exact fold**。
- **MIRAGE 排除**: stock RVV `ggml_gemm_q4_K_16x1_q8_K` = 破损 VLEN256-only (max_abs 29.3, 4081/4096 bad),
  故 VLEN128 正确实现 = block-dot(_generic); 板 canonical verifier 用真 make_block 字节路径, 非空心 oracle。
- **e2e greedy A==B / coherent** (deployed 6cbd9c19 kernel · 本会话 fresh · raw/correctness_gate.txt):
  A(ours clang repack)==B(stock gcc block-dot) **byte-identical** 两 prompt · engage=17(>0) · nan=0 · pass=2 fail=0
  → **CORRECTNESS_GATE: GREEN**。(计时无关 byte-compare·不受板污染影响)。

## [4] 分相 e2e — 系统账 (DeepSeek-8B-Q4_K_M · taskset -c 8-15 -t 8 · pp128/tg16 · 交织 · DVFS 2.6GHz 锁 · 12 passes/side · REPS=2 · raw/parse_perpass_relIQR.txt)
| 相 | ON ours(clang repack) t/s | OFF stock(gcc block-dot) t/s | 比 ours/stock | relIQR | 判读 |
|---|---|---|---|---|---|
| **prefill pp128** | **6.970** (min 6.820 max 7.026) | **5.183** (min 5.172 max 5.206) | **1.344×** (per-pass 1.317–1.358) | 0.74%/0.24% | **WIN ≥parity (出口 A)** |
| decode tg16 | 1.046 (min 1.035 max 1.050) | 2.091 (min 2.037 max 2.108) | **0.499×** (per-pass 0.491–0.516) | 0.97%/0.60% | LOSS (黄·VLEN128-recon) |

- 12 passes 全项 relIQR<1% · **最差 prefill pass = 1.317× 仍 WIN** (稳健于噪声与污染)。
- **★板污染处置 (coordinator 巡检 + 我方主动核)**: 测中 rvv 板邻居 vLLM qwen3-32b 突发服务 (loadavg 5→11.7,
  passes 10-12 窗)。**但我方测量核 pin cores 8-15 与污染源 disjoint**: vLLM Worker_TP0/TP1 pin cores **0,1**
  (CHIP_MAP=0,1) · `ps -o psr` 实证 cores 8-15 **仅** 我方 llama-bench (无 non-mine 进程)。突发仅经共享内存
  总线致 ours_pf passes 11-12 微降 ~1.5% (6.820/6.877 vs 6.970 median·<relIQR floor·stock 恒 5.17-5.21)。
  **无样本越污染门 (IQR×1.5)·无需作废重测·非 env-blocked = disjoint-pin 净测。** DVFS 2.6GHz 全程锁 (无降频)。
- **对手身份**: OFF = stock as-shipped **gcc-15.2.0** block-dot (板出货编译器) · ON = ours **clang-18.1.8** emitted repack
  (我方出货形态 [L-10])。**gcc-death 归因在档**: EXACT 部署 kernel objdump spill gcc **742**(GEMM)/**217**(GEVM)
  vs clang **4**/**8** = ~185×/27× regfile-spill 差 ([CASE-COMPILER-ASYMMETRY])。gcc→clang flip: prefill 0.334×→1.344×。
- **无 kernel-axis 备选**: q4_K@VLEN128 stock RVV repack 破损 (AVL=16·VLEN256-only)·不存在 compiler-symmetric
  "repack vs repack" 对照 → 系统账是唯一忠于真 baseline 的记法 (非撤回的 kernel-axis clang-vs-gcc artifact)。

## [5] board restore + md5 双证 = GREEN
- **cert-1 (live .so OFF-pristine)**: md5=05a62e6a8285 (expect 05a62e6a) · tcrv_sym=0 · banner=0  PASS
- **cert-2 (source byte-exact baseline)**: GEN=deb61a29dd07 · ARCH=99131cf791e3  PASS (source 全程未 patch·无 rebuild)
- **cert-3 (无 q4_K-specific stray .inc)**: tcrv_emitted_*q4_K*.inc = 0  PASS (decisive 已清·本会话未生成)
- ★澄清: ARCHDIR 内 `tcrv_emitted_repack_gemm.inc`(md5 4e21a79c)+`_gemv.inc`(md5 9575add6·均 2026-07-06) =
  **baseline 文件·非 stray** (pristine ARCH repack.cpp L22-23 #include 之·属出货 baseline)·**不得删**·其存在即 pristine 态。
- 0 campaign 遗留进程。板改动仅测中 swap LIVE .so (ON/OFF·测后 restore OFF-pristine)·全程 NO git。

## VERDICT (预注册出口·裁·二) — 出口 A 满足
**系统账 q4_K@rvv (我方 clang-18 emitted repack 正门部署 vs stock as-shipped gcc-15 block-dot):**
- **Prefill = 1.344× WIN ≥parity** (12-pass median·relIQR 0.74%·最差 pass 1.317× 仍 WIN·disjoint-pin 净测·非 env-blocked)
  → **出口 A 命中: rvv/VLEN128 半 category 黄→绿 (系统账)**。成色注记: **系统账 (deployment vs deployment)** ·
  gcc-death 归因在档 (742→4 spill) · 对手=stock as-shipped gcc-15。
- **Decode = 0.499× LOSS** → 照预着色**黄** (VLEN128 权重重建摊销·memory-bound GEVM·prefill-win/decode-loss split 定式·同 q5_K)。
- **correctness GREEN** (fresh greedy A==B byte-identical + ZERO-MODEL INTEGER BIT-EXACT + bounded-ULP + MIRAGE 排除)。
- **系统账铁律遵守**: 此 prefill WIN 是**系统账** (我方 clang .o vs stock gcc-15 部署)·**非** kernel 质量声明·如实标账本。
  [NG-4]: beat 只从八门放行 (本格 e2e single-board rvv·②⑤ single-board caveat)。
- **板 restore = TRUE** (live 05a62e6a OFF-pristine·source baseline·md5 双证)·**NO git**。

### schema label 建议 (改留主会话·我只报建议)
- cell `{op:gemm_tile, format:q4_K, engine:rvv}` **现已 category=绿** (绿立足 **k1/VLEN256** 半·Win-K1-VLEN sealed kernel 账)。
- **本结果 = 升级该 cell 的 rvv/VLEN128 半** (原 quality_note "同 cell rvv-half 是 yellow·0.334× LOSS [GAP-Q4K-VLEN128]+gcc-codegen")
  → **rvv/VLEN128 半新增系统账绿** (prefill 1.344× WIN·系统账·gcc-death 在档)。
- **★诚实提醒 (非我裁)**: 该 cell 已计绿 (via k1 any-board)·故本 rvv-half 翻正**未必**改变 perf-covered 分子计数 7→8
  (可能是同 cell 从 single-board-green 升 dual-board-green·消 [GAP-Q4K-VLEN128] 黄 caveat)。**7/83→8/83 的计数
  裁量属主会话 reconcile**·我不自行改 schema/T8/count。建议 quality_note 补: "rvv/VLEN128 半 = 系统账绿
  (ours clang-18 vs stock gcc-15·e2e prefill 1.344× WIN·decode 0.499× 黄 VLEN128-recon·gcc-death 归因在档·disjoint-pin
  净测·single-board rvv)"。
- decode 半维持黄 (VLEN128-recon·未摊销)。
