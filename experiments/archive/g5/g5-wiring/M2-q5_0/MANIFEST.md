# G5-M2 q5_0 曳光弹 — L-接线② 净新 scaffold · CASEFILE

> **campaign**: G5 接线战役 · **M2 L-接线② q5_0**（首个真净新上游 scaffold·验证 template 建 upstream scaffold 能力 = C1 extensibility）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree f3e1828（restore 中·workflow glitch 后 board-restore agent 收尾）
> **workflow**: `wtpdeaoes`（phase-1 recon）+ `w7cwrau8d`（phase-2 build·**末尾 StructuredOutput glitch FAILED·但工作已完成**）· **HEAD (TianChen-RV)** = a7cacf68
> **★结论**: **净新 12-piece scaffold BUILT + correctness GREEN + perf GREEN（prefill-axis）→ perf-covered 3/84→4/84**。scaffold（make_block_q5_0x16 interleaver 成功）+ correctness（5/5 byte-identical·PPL 17.88·objdump vl=8）+ **perf 重测（a68526c0）prefill 1.2097× WIN（≥parity·1.23× micro 确传导·非 wash）/ decode 0.8165× regression（memory-bound GEVM·披露）**·八门全过 prefill 轴·双账本 kernel==system（gcc-15 出货对称）。**= L-接线② 净新方法学验证成功 · C1 template extensibility 实证 + perf 传导**（template 从零建 upstream repack scaffold·prefill e2e ≥parity）。★caveat：**prefill-axis green**（非 q8_0 式双相赢·decode 回退 −18% 披露）。

## 一、phase-1 recon（wtpdeaoes·02bfa2c4）
- **upstream_absent 确证**（q5_0 GEN+ARCH grep=0·block<K,N> 模板无法表达第 5 位 qh·与 q4_K 全-present 决定性相反）·kernel EMITTED（vl=8·VLEN128-safe）·12-piece 净新 scaffold recipe 全设计（evidence.md §三）。

## 二、phase-2 build+correctness（w7cwrau8d·从 workflow 残留恢复·board-src + seal + correctness 落盘）
- **12-piece 净新 scaffold BUILT**（seal_raw.txt：HDR struct/decls·GEN make_block_q5_0x16/repack_tmpl/gemv_tmpl/gemm_tmpl/trait/dispatch/generics·ARCH gemv/gemm body+call 全 True）——**make_block_q5_0x16 transposed-qh interleaver 一次成功**（correctness-critical MIRAGE trap 过关）。
- **★correctness GREEN**（correctness_GREEN_raw.txt）：DeepSeek-8B-Q5_0 · 5/5 prompt byte-identical A(emit)==B(stock) coherent · 45 emitted-kernel banner fires · PPL(ON)=17.88 finite/coherent · no NaN/Inf · **CORRECTNESS_GATE GREEN**。
- **objdump vl-seal**（objdump_gemm/gevm_q5_0_seal.txt）：q5_0 emitted 符号 `vsetivli imm=8=1·imm=16=0·imm=64=0` = VLEN128-safe·部署==证过。nm 两符号在·ON≠OFF。
- **perf UNMEASURED**：workflow 末尾 StructuredOutput glitch failed → phase-2 measure agent 未跑 → **无 perf 分相数**。perf-covered **维持 3/84**（不以未测冒绿·q5_0=correctness-carrier·perf 待 clean 重测·预期 FLAT 1.23× 小 margin 可能 wash）。

## 三、L-接线② 方法学产出（★C1 extensibility 实证·独立于 perf）
- **net-new scaffold 建法验证成功**：template 从零建 upstream repack scaffold（12-piece·GEN trait+dispatch+repack fns+generic fallback·ARCH gemv/gemm skeleton·repack.h struct block_q5_0x16）→ correctness-carrier GREEN。**q4_K 是复用（上游 present）·q5_0 是真净新**——证 L-接线② 的"template 能建 upstream scaffold"主张（C1）。
- **q5_1 复用**：deploy_patch_q5_0_emitted.py + build_seal/correctness/phase_split harness（`tools/e2e-harness/board/g5-m2-q5_0/`）· recipe 见 evidence.md §三 + board-src patched 源（swap q5_0→q5_1·GGML_TYPE_Q8_1 激活·block_q5_1x16 加 m 字段）。

## durable files
- `evidence.md`（recon + emit recipe + 12-piece 净新 scaffold recipe·全锚点）
- `correctness_GREEN_raw.txt`（phase-2 correctness 证据·5/5 byte-identical·PPL·banner）
- `phase_split_raw.txt`（perf 重测·prefill 1.21×/decode 0.82×·llama-bench json·n=20）
- `transmission_accounting.csv`（双账本·per-pass·八门键）
- `seal_raw.txt`（build+patch+seal 全 log·12-piece 建成证据）
- `objdump_gemm_q5_0_seal.txt`（GEMM vl=8 objdump seal）
- `objdump_gevm_q5_0_seal.txt`（GEVM vl=8 objdump seal）
- `tcrv_emitted_gemm_q5_0.inc`（host-emitted GEMM·md5 f03c6566·27KB）
- `tcrv_emitted_gevm_q5_0.inc`（host-emitted GEVM·md5 f3892049·20KB）
- `.gitignore`（gitignore board-src patched 源·regenerable）

> `board-src/`（patched ggml 源·GEN 196KB/ARCH 103KB/repack.h·320KB total）= **gitignored**（regenerable via `tools/e2e-harness/board/g5-m2-q5_0/deploy_patch_q5_0_emitted.py` 套 baseline·避 bloat）。board harness 住 `tools/e2e-harness/board/g5-m2-q5_0/`。
