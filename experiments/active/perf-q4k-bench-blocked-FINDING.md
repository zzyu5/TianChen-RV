# FINDING — q4_K@k1 headline 无法走 bench 通道合格化（bench cell 覆盖=grid4-only·3 层阻塞）

**task**：`07-20-perf-q4k-bench-tn-qualify`（性能诚实盘点续·把唯一 headline 做成 T-N-qualified）
**verdict**：**NOT ADJUDICABLE**——bench 通道对 q4_K 确定性阻塞·**0 样本达到计时阶段**·[PERF-2]「0 样本永不造数」未造数·未走 ad-hoc 兜底（PRD 🔴 禁）。
**headline 现状**：master rows 71/72 q4_K@k1 decode **1.535** / prefill **1.187**（vs `ggml_gemm_q4_K_16x1_q8_K`·k1_cold）**未动·仍 measured-非-T-N**。B4「runs.log 零行(qualified)」债对 q4_K **维持 OPEN**。
**runs.log**：落 **VOID 行**（`20260719T183819Z-q4_K-k1-fe7eb556·VOID-我方错(verify harness rc=2)`）= 诚实的端到端通道结果（通道被行使·卡在 harness）·非 ad-hoc 数。master 未写（VOID 不写 master）。

## 3 层阻塞（各带 command / file:line / error）

**BLOCKER 1（主·硬停·ISSUE-090 harness落点）**：`tools/bench/cells/gemm_tile.sh:41-50` 的 `case "$FMT"` **只列 grid4 IQ 家**（iq1_s|iq1_m|iq3_xxs|iq3_s|iq2_xxs|iq2_xs|iq2_s）·q4_K 落 `*)` → `HARNESS-VOID bad fmt q4_K`·`exit 2`。
- 根因：「gemm_tile」cell 实为 **grid4-IQ 家 harness**·硬绑 `experiments/active/g8-stage3-attack/P2-grid4-raw/`（driver `grid4_gemm_prefill_p2.cpp`·leaves `kernels_grid4{,_vlen256}/`·tables `iq*`）。**该树无 q4_K leaf/driver/table**。q4_K 是 super-block（不同 driver ABI·repack GEMM·stride 2304·x16-interleaved）——grid4 driver 托不了。真 q4_K 资产在**另一 ad-hoc 树** `A2-batch4-gemm-decode-M1-raw/`（`kquant_gevm_m1_driver.cpp`·`leaves/q4_K_gevm_hl8.c`·`run_flat_gevm_k1.sh`）·**从未接进 bench cell**。
- 端到端确认：`python3 tools/bench/bench gemm_tile q4_K --board k1 --engine rvv --regime decode` 过 roster+step1 preflight+step2 lineage·step3 VOID（`bench:481-489`）·artifact `experiments/runs/20260719T183819Z-q4_K-k1-fe7eb556/verify.stdout.txt`。

**BLOCKER 2（ISSUE-090 对手契约）**：`tools/bench/bench:531-537`（step3_crosscheck）硬编码 `thunk = ggml_vec_dot_{fmt}_q8_K`·k1 要 `ggml_vec_dot_q4_K_q8_K_vl256`。但 q4_K headline 对手是 **`ggml_gemm_q4_K_16x1_q8_K`（GEMM hand-brick·非 vec_dot）**。⟹ 即便有 harness·runner 验③ 会验错对手·把 gemm-vs-gemm 赢**误记成 cross-op vec_dot**。runner 对手模型只 fits grid4 IQ 家（其对手确是 vec_dot cross-op）·不 fits q4_K gemm-vs-gemm-hand-brick。

**BLOCKER 3（runner step4 只 prefill）**：`tools/bench/bench:553-611`（step4_cold）唯一计时正则 `_GEMM_PREFILL_RE`·**无 decode/GEVM 路**。`--regime decode`（1.535 headline·M=1 GEVM 测量）无提取 → decode harness 会 `VOID-脏板(解析不出 GEMM_PREFILL)`·或错把 prefill ratio 写进 decode 行（`update_master_cell:643-683` regime 错配）。

**ISSUE-091 不是阻塞**：四元键 `(gemm_tile,q4_K,rvv,decode)` 无歧义定位 row 71·roster+lineage 全过。阻塞纯 090（缺 q4_K harness 契约）+ runner 的 decode/对手 gap。

## 前置（据证报·非重跑 ad-hoc）
- **byte-exact**：batch4 seal `A2-batch4-gemm-decode-M1-raw/logs/q4_K_raw.txt:4` = `mism=0/512 maxrel≤6.76e-07`（**2-codegen 一致门·非** PRD 要的全 ZERO-MODEL 3-arm/4-arm·fresh 3-arm re-seal 同样卡 Blocker 1）。
- **compiler-symmetric**：live lineage probe = `Bianbu clang 18.1.8`·双方 k1-shipped clang-18。
- **[L-11] 同域**：board SpacemiT X60·VLEN256·8c·deployed lib md5 **871169a0** == batch4 seal k1 stock md5（域未变）。

## Unblock（下一 task·本役未做）
给 gemm cell 加 q4_K 臂三件：① super-block driver+leaf+oracle 配方（改编 `kquant_gevm_m1_driver.cpp`/`q4_K_gevm_hl8.c` 进 `tools/bench/cells/`）② step4 加 decode/GEVM 计时 emitter+regex ③ 对手符号参数化（验③ 对 `ggml_gemm_q4_K_16x1_q8_K`(prefill) + M=1 block-dot(decode)·替硬编码 vec_dot）。三件齐才能跑 T-N 地板 + bootstrap CI。

## 意义
**这解释了 r5.1 为何走 ad-hoc**：bench 通道（ISSUE-090「RESOLVED」）实只为 **grid4 IQ 家**接通·super-block/repack/K-quant 家**无 cell 臂**。⟹ CLAUDE.md:35「现有 grid/repack 数须走 bench 复测」对 super-block/repack **本身即被 Blocker 1 阻**（bench 复测这条路对这些家尚不存在）。**bench=唯一合法测量动作** canon 成立·但**操作覆盖 = grid4-only**·扩到全格是真实基建欠账。
