# MANIFEST — l2-kquant-rvv-systemacct / q5_K

- campaign: G6 令·二 — K-quant/RVV **系统账翻正线** (完成 L2 三格). q5_K@rvv (★已绿 via k1·系统账
  = dual-board 升级 or gcc-death 必要非充分律【第三数据点·tiebreaker】). 承 q4_K(b2531fe1·WIN)/
  q2_K(14f4631a·LOSS) 成功模板.
- ★关键定位: 律 = 系统账 WIN ⟺ gcc-death ∧ emitter-quality-beat (两者缺一不可).
  q4_K@rvv = 1.344× WIN (gcc-death 治愈 742→4 后【反超】block-dot).
  q2_K@rvv = 0.857× LOSS (gcc-death 治愈 2937→6 更极端但 emitter-quality 不足).
  q5_K = tiebreaker: q5_K@k1 已绿 (our-kernel 1.641× prefill vs stock block-dot); rvv 系统账 WIN=
  dual-board 升级(q5_K 属 q4_K-class) / LOSS=律第三例(q5_K 属 q2_K-class).
- role: 三账并列忠实记录:
  ① **系统账** (headline·deployment vs deployment): ON = 我方 clang-18 emitted q5_K repack .o 正门
     (== 出货形态 [L-10]) · OFF = stock 板出货 gcc-15 as-shipped block-dot.
  ② **对称-gcc 账** (算法本体·诚实底数): 我方 q5_K repack 也 gcc-15 编 (q5kON.gcc) vs stock gcc-15
     block-dot. 复现/更新 schema/T9 q5_K@rvv 对称-gcc **0.120× LOSS** (~12.9× reversal·gcc-death 极端).
  ③ **k1 kernel-axis** (clang 对称·已绿·不重测·k1 禁): q5_K@k1 e2e prefill 1.641× WIN / decode
     0.729× roofline-loss (kernel-axis micro 1.916×)·对手 = stock q5_K block-dot (NOT hand-brick·
     k1 ships ZERO q5_K repack). 引用 (schema cell[20] 绿·ledger 2026-07-12-perf-covered-q5_K-k1-green).
- **framing 铁律 ([CASE-COMPILER-ASYMMETRY])**: 系统账与被撤回 micro 是同一物理比较换轴报. 若系统账
  ≥parity 只能表述 "部署账赢·gcc-death 机制·非 kernel/算法赢" + 必须同披露对称-gcc LOSS + k1 状态.
- board: `ssh rvv` openEuler VLEN128 64c · cores 8-15 perf-gov 2.6GHz (DVFS locked) · disjoint-pin vs
  vLLM(cores 0,1·CHIP_MAP=0,1·98%CPU steady)
- NO git · board reversible (只 swap LIVE .so; source 3 文件 GEN/HDR/ARCH 全程 pristine·restore md5 三证)

## provenance (读死)
- .so 变体 (built by q5k_mixed_build.sh · gcc-15 tree · recompile 2 repack TUs per variant · /tmp/dkgv):
  - OFF  q4kOFF.gcc         md5 05a62e6a (共用 q2_K/q4_K 的 pristine block-dot baseline·q5_K@VLEN128
    dispatch NEON-only -> nullptr on riscv -> stock ggml_vec_dot_q5_K_q8_K block-dot) [DENOM]
  - ON-a q5kON.clangrepack  md5 53b2d3b0 = 我方 clang-18 emitted q5_K repack [系统账 NUMER]
  - ON-b q5kON.gcc          md5 a381efbf = 我方 gcc-15 emitted q5_K repack [对称-gcc 账 NUMER]
- 我方 q5_K rvv scaffold = NET-NEW (rvv A-tree ships q2_K/q4_K 1,16 riscv repack 但 q5_K dispatch NEON-only
  -> 我方 deploy_patch_q5_K_rvv.py 加全 scaffold: struct block_q5_Kx16(stride2816·d@0 dmin@32 scales@64
  qh@256 qs@768) + make_block_q5_Kx16(straight interleave·== make_block_q4_Kx16 body + qh 5th-bit plane)
  + repack/gemv/gemm templates + q5_K_16x1_q8_K trait + riscv dispatch case128=ON + arch bodies).
  Adapted from k1 deploy_patch_q5_K_emitted.py (case256->case128).
- emitted kernel = kernel-axis t4a export (VLA·__riscv_vsetvl_e32m1 avl=8 f32m2·VLEN>=128-valid):
  gemm_q5_K_q8_K.kernel.c md5 **ba30ba54** (=tcrv_emitted_gemm_q5_K.inc·S6-tiled·kernel-axis k1 1.916×)
  + k_gemv_q5K.cpp md5 **c445b89e** (=tcrv_emitted_gevm_q5_K.inc). 符号 tcrv_emitc_ggml_repack_gem{m,v}_
  q5_K_q8_K_kernel_... (7-arg gemm n,s,vx,vy,nr,nc,bs / 5-arg gevm n,s,vx,vy,nc).
- gcc-death seal = build_seal.txt: EXACT deployed kernel objdump spill
  q5kON.gcc GEMM=4124 GEVM=454  vs  q5kON.clangrepack GEMM=155 GEVM=97  (~27× GEMM spill·比 q2_K 2937 更极端·
  clang q5_K GEMM spill 155 > q2_K/q4_K clang (6/4) = qh 5th-bit plane 残留 spill floor·[CASE-COMPILER-ASYMMETRY]).
- correctness ZERO-MODEL 源 = kernel-axis t4a (k1 VLEN256 ULP0 int / 8e-7 norm·独立 scalar q5_K oracle) +
  M4-q5K full cert (cert-lineage·REAL dmin!=0·INTEGER MAIN/MIN 0-mismatch) + 本会话 rvv/VLEN128 greedy
  (byte-id p2 + bounded-ULP coherent p1·MIRAGE 排除·engage vlen=128).
- deploy = deploy_patch_q5_K_rvv.py (3 files·baseline GEN deb61a29/HDR 57851439/ARCH 99131cf7·case128 flip).

## durable files
- evidence.md — verdict (三账) + route + 5-verify + correctness + phased e2e (systems + symmetric-gcc + k1 ref)
- MANIFEST.md — this
- raw/build_seal.txt — q5k_mixed_build seal (3 variants·gcc-death spill gcc 4124/454 vs clang 155/97·restore 三证)
- raw/deploy_5verify_reverse_control.txt — deploy 5-verify + reverse control (engage double-clean·vlen=128)
- raw/correctness_gate.txt — fresh in-session greedy A vs B (VLEN128·byte-id + bounded-ULP coherent + MIRAGE)
- raw/measure_lean_parse.txt — 2-phase e2e parse (systems clang/stock + symmetric gcc/stock·per-pass load·relIQR)
- raw/json/ — llama-bench phase-split JSON (clang/stock/gcc)

## reversibility
- 板改动: 仅 swap LIVE libggml-cpu.so.0.15.1 (测中·测后 restore to OFF-pristine 05a62e6a).
  source 3 文件全程 pristine (GEN deb61a29 / HDR 57851439 / ARCH 99131cf7). q5k_mixed_build 内 patch->
  build->restore byte-exact + pristine gcc rebuild 已闭环. 无 source patch 残留·无 rebuild 需求.
- model provisioning (决策卡④): DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf on-board requantized from Q6_K
  (--allow-requantize·llama-quantize Q5_K_M·5.7GB). 公开可复现·登记.
- restore md5 三证: cert-1 live .so == 05a62e6a (OFF-pristine·tcrv_q5k_sym=0) + cert-2 source
  GEN/HDR/ARCH byte-exact + cert-3 0 tcrv_emitted_*q5_K*.inc stray (ARCHDIR·build trap rm) + 0 遗留进程.
- 污染处置: vLLM qwen3-32b TP2 (Worker_TP pin cores 0,1·disjoint) 与测量核 8-15 disjoint·relIQR 拒污染.
  (board scripts live in /tmp/dkgv: q5k_mixed_build.sh, deploy_patch_q5_K_rvv.py, q5k_verify.sh,
   q5k_correctness.sh, q5k_measure_lean.sh, q5k_parse_lean.py; emitted kernels /tmp/case_0b0d + /tmp/q5k_verify)
