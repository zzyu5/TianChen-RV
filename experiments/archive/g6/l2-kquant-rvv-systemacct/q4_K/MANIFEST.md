# MANIFEST — l2-kquant-rvv-systemacct / q4_K

- campaign: G6 令·二 — K-quant/RVV **系统账翻正线** (最便宜绿格池首格). q4_K@rvv 先行格.
- role: 把 decisive-kquant-gcc-vs-vlen 的 ours(clang-repack) vs stock(gcc-block-dot) 对比,
  从 "internal-A/B 机制探针记账 ([NG-4]·不计 perf-covered)" **重记为合法系统账**
  (deployment-vs-deployment): ON = 我方 clang-18 emitted q4_K repack .o 正门部署 (== 我方出货
  形态 [L-10] 设计事实) · OFF = stock 板出货 gcc-15 as-shipped block-dot. 更严纪律重打
  (≥12 samples/side·load-gate·relIQR·分相). 区别撤回的 kernel-axis clang-vs-gcc artifact
  ([CASE-COMPILER-ASYMMETRY]: kernel 账 vs-opponent 仅编译器对称时有效) —— 系统账两边各用其
  真实出货编译器 (我方 clang / 板 gcc-15), 是部署现实非 kernel 质量声明.
- board: `ssh rvv` openEuler 24.03 VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · load-gate<6
- account: **系统账 (deployment vs deployment)** · 我方 clang-18 .o vs stock gcc-15 · 披露双方编译器身份.
  系统账铁律 (裁·七): kernel 账不得表述为系统收益 — 此格是系统账, 如实标. [NG-4] beat 只从八门放行.
- shared-board 警示: 邻居 vLLM server (qwen3·别杀) + kernspan monitor loop. load-gate + relIQR 拒污染.
- NO git (主会话 commit/add) · board reversible (只 swap LIVE .so; source 全程 pristine·测后 restore md5 双证)

## provenance (读死)
- .so 变体源 = experiments/active/decisive-kquant-gcc-vs-vlen (裁二.2 决定性实验) · /tmp/dkgv 内 md5-sealed:
  - OFF q4kOFF.gcc          md5 05a62e6a (860648B) = stock as-shipped gcc-15 block-dot [DENOM]
  - ON  q4kON.clangrepack   md5 915e6733 (913016B) = 我方 clang-18 emitted q4_K repack [NUMER]
- emitted kernel = g5-wiring M2 tcrv_emitted_gemm_q4_K.inc(6cbd9c19) + gevm(e909a9bd) · VLEN128-safe vl=8
- deploy = M2 deploy_patch_q4k_emitted.py (case128 flip + emitted intercept)
- correctness ZERO-MODEL 源 = t4b-m4-decisive (INTEGER BIT-EXACT MAIN 0/MIN 0 on real dmin!=0 Q4_K_M
  tensor · bounded-ULP max_rel 2.86e-6 · MIRAGE 排除: stock RVV repack = 破损 VLEN256-only)
- gcc-death 归因 = decisive evidence.md ([CASE-COMPILER-ASYMMETRY] 742/217 spill vs clang 4/8)
- e2e 锚 (gcc 账 LOSS) = result-tables/T-PERF1b + g5-wiring/M2-q4_K (prefill 0.334×/decode 0.18×)

## durable files
- evidence.md — verdict + route confirm + 5-verify + correctness + phased e2e (systems account)
- MANIFEST.md — this
- raw/deploy_5verify_reverse_control.txt — deploy 5-verify + reverse control seal
- raw/provenance_objdump_spill_compare.txt — 3-compiler spill objdump of EXACT deployed kernels
- raw/provenance_decisive_correctness_gate.txt — decisive greedy A==B seal (same ON/OFF pair)
- raw/correctness_gate.txt — fresh in-session greedy A==B seal (this campaign)
- raw/passlog.txt + parse.txt — per-pass load/freq + phased e2e aggregate (relIQR)
- raw/*.json — llama-bench phase-split JSON per pass (ours_p*/stock_p*)

## reversibility
- 板改动: 仅 swap LIVE libggml-cpu.so.0.15.1 (测中 ON/OFF·测后 restore to OFF-pristine 05a62e6a).
  source .cpp 全程 pristine (GEN deb61a29 / ARCH 99131cf7 未动). 无 rebuild·无 source patch.
- ★.inc 澄清 (初判纠正): ARCHDIR 的 tcrv_emitted_repack_gemm.inc(4e21a79c)+_gemv.inc(9575add6·均 2026-07-06)
  = **baseline 文件** (pristine ARCH repack.cpp L22-23 #include·属出货 baseline)·**非 stray·不得删**. 真正的
  q4_K-specific stray (tcrv_emitted_*q4_K*.inc) = 0 (decisive 已清·本会话未生成). 我方 restore 脚本硬编码删名
  (..._q4_K.inc) 正确未匹配 baseline 文件·零误删.
- restore md5 双证: cert-1 live .so == 05a62e6a (OFF-pristine·tcrv_sym=0·banner=0) + cert-2 source GEN/ARCH
  byte-exact baseline + cert-3 0 q4_K-specific stray .inc + 0 campaign 遗留进程. 全 PASS.
- 测量污染处置: vLLM qwen3 突发 (cores 0,1·passes 10-12) 与测量核 (cores 8-15) disjoint·ps psr 实证·
  relIQR<1%·无越门样本·disjoint-pin 净测 (非 env-blocked).
