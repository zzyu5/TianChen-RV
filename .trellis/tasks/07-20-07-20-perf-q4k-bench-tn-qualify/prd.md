# PRD — 性能真赢合格化: q4_K@k1 headline 走 bench 通道 + 过 T-N（关闭 measured-非-T-N 债·柱一证明书）

## 缘起（性能诚实盘点结论·用户令"性能搞清楚"续）
盘点（`experiments/active/perf-honest-scorecard.md`）定论：**真硬赢仅 2 格**·其中 **q4_K@k1（decode 1.535 + prefill 1.187·唯一双轴·vs 真强手调 `ggml_gemm_q4_K_16x1_q8_K`）是唯一 headline-eligible**。但**它仍 measured-非-T-N**——未走 bench 合法通道（runs.log 零行）·newest law（CLAUDE.md:35）要求 bench 重 seal。**对论文柱一（证明书），一个 T-N-qualified headline > 一个未合格的第二个赢。** 本役 = 把唯一 headline 做成可信。

## 做（走合法通道·过 T-N·禁 ad-hoc 兜底）
1. **走 bench 通道**：用 `tools/bench/bench`（cell `tools/bench/cells/gemm_tile.sh`）测 q4_K@k1·四元行键 `(op=gemm_tile, format=q4_K, engine=<定位>, regime∈{decode,prefill})`·对手 = `ggml_gemm_q4_K_16x1_q8_K`（盘点已确认真强手调·编译器对称 clang-18）。**目的地写 `experiments/runs.log` + `master/`**（关闭 B4「runs.log 零行」债）。
2. **过 T-N 噪声地板**（测量判据.md:32·判定资格地基）：
   - 每板 × gemm_tile 类跑 **rerun-rerun 得运行间 IQR%**（= 噪声地板）。
   - q4_K 的 decode/prefill ratio 的 Δ（vs parity 或 vs 对手）须 **|Δ| > 2× 噪声地板 且 bootstrap 95%CI 不含 0**。
   - **[PERF-2]**：测前 3 次 rerun-rerun sanity·IQR ≤ 历史地板 ×1.5 才开测·测中 IQR 突劣 ≥3× 作废重跑。**0 样本永不造数**。
3. **byte-exact 前置**（不可绕）：q4_K@k1 emit 仍 byte-exact（CORE==PROD·3-arm·此前已 seal·复验 md5 未漂）。
4. **产出**：q4_K@k1 decode + prefill 的 **T-N-qualified 数**（ratio + 噪声地板 + bootstrap CI）+ 环境指纹（board/编译器/deployed variant/输入路径·[L-11] 同域）。

## ★若 bench 通道阻塞（诚实诊断·不 ad-hoc 绕）
r5.1 grid/repack 走了 ad-hoc `run_*.sh` 而非 bench——**说明 bench 通道对真格有摩擦**。若 q4_K@k1 走不通 bench：
- **精确诊断阻塞**：是 harness 落点缺（ISSUE-090·gemm_tile.sh 对 q4_K 的被调契约）？命令签名不定位主表行（ISSUE-091·engine 推不出）？行键歧义 fail-closed？board 侧缺 fixture？——**逐条定位·file:line/命令/报错**。
- **🔴 禁 ad-hoc 兜底**：不许"走不通就 run_*.sh 出个数"（那是 r5.1 已犯的·perpetuate measured-非-T-N 债）。**宁可交诊断·不可交 ad-hoc 数**。
- 诊断本身 = 解锁产物（告诉我们 bench 通道差哪一步·下一 task 补）。

## 门
- **T-N-qualified**（|Δ|>2×地板 且 bootstrap CI 不含 0·或诚实报"Δ 在噪声内=不可判"）· 走 bench 通道（runs.log 有行·非 ad-hoc）· byte-exact 复验（md5 未漂）· 编译器对称（clang-18 双方）· [L-11] 同域环境指纹 · **0 造数**（0 样本宁可无数）· 未 git commit（board 数据+FINDING 写 experiments/·master 更新走 recon）。
- 🔴 禁碰在飞 agent 文件（phase-1=dequant descriptor / MIG-5=RVVGearboxSchedule.h）——本役纯 board 测量+experiments·天然不碰源。

## 板 & 汇报
- `ssh k1`（VLEN256·主）。**correctness 前置在 rvv 也可复**。
- 汇报（自然·首节 T-N-qualified 结果）：q4_K@k1 decode/prefill 的 ratio + 噪声地板 IQR% + bootstrap 95%CI + 是否过门（可判/不可判）+ runs.log 是否落行 + 若阻塞则精确诊断。final message 放关键数 + 判定资格结论。
