# MANIFEST — l2-kquant-rvv-systemacct / q2_K

- campaign: G6 令·二 — K-quant/RVV **系统账翻正线** (最便宜绿候选池). q2_K@rvv (★新绿候选·非升级).
  承 q4_K 成功模板 (commit b2531fe1). q2_K cell 现为 **黄-对手更强** (schema IDX 18); 翻绿【完全依赖
  系统账】(k1 kernel + 对称 gcc 双 LOSS)·比 q4_K (已绿·升级) 风险更高 → 绿/黄裁量收归主会话.
- role: 三账并列忠实记录 (裁决张力精化·主会话查 schema 后加的硬约束):
  ① **系统账** (headline·deployment vs deployment): ON = 我方 clang-18 emitted q2_K repack .o 正门部署
     (== 我方出货形态 [L-10]) · OFF = stock 板出货 gcc-15 as-shipped block-dot.
  ② **对称-gcc 账** (算法本体·诚实底数): 我方 q2_K repack 也用 gcc-15 编 (q2kON.gcc) vs stock gcc-15 block-dot.
     复现/更新 schema IDX 18 的 0.386× LOSS —— "我方算法本身几斤几两" 的对称编译器底数.
  ③ **k1 kernel-axis** (clang 对称·已知): M2-q2_K-k1-e2e prefill 0.873× / decode 0.875× LOSS (对手=k1 真出货
     hand-brick·NOT block-dot). 引用, 不重测 (k1 禁·M7 在跑).
- **framing 铁律 (裁决张力·[CASE-COMPILER-ASYMMETRY])**: 系统账与被撤回的 micro 1.413× 是【同一物理比较】,
  只换轴报. 若系统账 ≥parity, 只能表述为 "**部署账赢·gcc-death 机制·非 kernel/算法赢**", 且【必须同披露
  对称-gcc LOSS + k1 kernel LOSS 0.873×】—— 否则=重新兜售已撤回的 artifact.
- board: `ssh rvv` openEuler VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · disjoint-pin vs vLLM(cores 0,1)
- NO git (主会话 commit/add) · board reversible (只 swap LIVE .so; source 全程 pristine·测后 restore md5 双证)

## provenance (读死)
- .so 变体 (built by q2k_mixed_build.sh · gcc-15 tree · only 2 repack TUs recompiled per variant · /tmp/dkgv):
  - OFF  q4kOFF.gcc         md5 05a62e6a (860648B) = stock as-shipped gcc-15 block-dot [DENOM·三账共用]
    (q2_K@VLEN128 case128 {break;}//TODO -> block-dot·与 q4_K 同构·pristine 全 K-quant OFF baseline)
  - ON-a q2kON.clangrepack  md5 0b01c53a = 我方 clang-18 emitted q2_K repack [系统账 NUMER]
  - ON-b q2kON.gcc          md5 b27bd686 = 我方 gcc-15 emitted q2_K repack [对称-gcc 账 NUMER]
- emitted kernel = M2-q2_K-k1-e2e export: weft_emitted_gemm_q2_K.inc (md5 d9dee831·VLA S6-tiled)
  + weft_emitted_gevm_q2_K.inc (md5 ee76a4d3·plain VLA). VLA (__riscv_vsetvl_ runtime AVL) -> correct at VLEN128.
- deploy = deploy_patch_q2_K_rvv.py (gate128 flip + STRAIGHT make_block_q2_Kx16 + gemm/gevm intercept·mirrors q4_K)
- gcc-death seal = build_seal.txt: EXACT deployed kernel objdump spill
  q2kON.gcc GEMM=2937 GEVM=218  vs  q2kON.clangrepack GEMM=6 GEVM=22  (~490× GEMM spill·[CASE-COMPILER-ASYMMETRY])
- correctness ZERO-MODEL 源 = M2-q2_K-k1-e2e kquant_repack_verify_q2K (INT_mismatch=0 all 8 shapes·k1/VLEN256)
  + 本会话 rvv/VLEN128 greedy A==B byte-identical (correctness_gate.txt).

## durable files
- evidence.md — verdict (三账) + route + 5-verify + correctness + phased e2e (systems + symmetric-gcc)
- MANIFEST.md — this
- raw/build_seal.txt — q2k_mixed_build seal (3 variants·gcc-death spill: gcc 2937/218 vs clang 6/22)
- raw/deploy_5verify_reverse_control.txt — deploy 5-verify + reverse control (engage double-clean)
- raw/correctness_gate.txt — fresh in-session greedy A==B byte-identical (VLEN128)
- raw/measure_3way_parse.txt — 3-way phased e2e parse + passlog (per-pass load·relIQR·4 clean passes)
- raw/json/ — llama-bench phase-split JSON (clang/stock/gcc p1-4)

## VERDICT (one-line) — 三账全 LOSS·不翻绿
系统账 prefill 0.857× / decode 0.516× LOSS · 对称-gcc 0.497×/0.150× LOSS · k1 kernel 0.873×/0.875× LOSS (引用).
q2_K@rvv 维持 **黄-对手更强**·perf-covered 计数不变. gcc-death 真 (spill 2937→6) 但不足反超 block-dot (区别 q4_K WIN).
correctness GREEN (greedy A==B byte-identical VLEN128 + k1 ZERO-MODEL INT bit-exact). 板 restore TRUE (md5 双证).
- (board scripts live in /tmp/dkgv: q2k_mixed_build.sh, deploy_patch_q2_K_rvv.py, q2k_verify.sh,
   q2k_correctness.sh, q2k_measure3.sh, q2k_parse3.py; and tools/e2e-harness/board/g5-q2_K-k1 for the k1 export)

## reversibility
- 板改动: 仅 swap LIVE libggml-cpu.so.0.15.1 (测中·测后 restore to OFF-pristine 05a62e6a).
  source .cpp 全程 pristine (GEN deb61a29 / ARCH 99131cf7). q2k_mixed_build 内 patch->build->restore
  byte-exact + pristine gcc rebuild 已闭环 (每变体建完即 restore). 无 source patch 残留·无 rebuild 需求.
- model provisioning (决策卡④): DeepSeek-R1-Distill-Llama-8B-Q2_K.gguf on-board requantized from Q6_K
  (--allow-requantize·llama-quantize Q2_K·3.16 BPW·3.03GB). 公开可复现·登记.
- restore md5 双证: cert-1 live .so == 05a62e6a (OFF-pristine·weft_sym=0) + cert-2 source GEN/ARCH byte-exact
  + cert-3 0 q2_K-specific stray .inc (ARCHDIR) + 0 campaign 遗留进程.
- 测量污染处置: vLLM qwen3-32b TP2 (cores 0,1·98%CPU steady) 与测量核 (cores 8-15·仅 idle kworker) disjoint·
  taskset psr 实证·relIQR 拒污染 (非 env-blocked·start-load-gate 系 disjoint vLLM 过计·非我核).
