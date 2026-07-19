# PRD — 性能诚实盘点（只读·"性能搞清楚"·成色分层 + 下一真赢路径）

## 缘起（用户令：增加公式 与 性能 都很重要·都要搞清楚）
"性能搞清楚" = **不糊·分层诚实**：哪些是**真硬赢**（硬碰硬赢强手调对手·编译器对称·bench+T-N-qualified）· 哪些是**便宜档**（vs 弱/scalar-ref 对手·big-multiple 弱赢·禁称硬赢）· 哪些是**部署变更**（LMUL flip·非 vs-对手 perf）· 哪些是 **measured-非-T-N**（ad-hoc 无 bench 通道无噪声地板）。**0 造数·0 夸大**·[成色与措辞] 铁律。

## 做（纯只读·零改码·出一份诚实 scorecard）
1. **读权威源**：
   - 主表 `experiments/master/T3_master_rebuild.csv`（+ `T3_A_board_A_rvv1.0_vlen128.csv` / `T3_B_board_B_rvv1.0_vlen256.csv`）——逐格 ratio + engine + regime + 对手身份 + 成色注。
   - 度量脚本跑一遍取权威数：`python3 .trellis/scripts/perf_covered_metrics.py` + `coverage_metrics.py`（perf-covered 分子/分母·别手抄）。
   - 成色 canon（判据正本·据此分层）：`.trellis/spec/canon/成色与措辞.md` / `对手与档位.md`（对手五档）/ `测量判据.md`（T-N 噪声地板·判定资格）。
2. **产出 scorecard**（分层·每格标 file/ratio/对手身份/门状态）：
   - **A. 真硬赢**：硬碰硬赢**强手调**对手（非 scalar-ref/非弱 vendor）· 编译器对称 · byte-exact · **bench+T-N-qualified**。逐格列（预期极少·如 Win-K1-VLEN q4_K vl16 双轴）。**这是唯一能当论文 headline 的层**。
   - **B. 便宜档赢**：vs scalar-ref / 弱 vendor / opp-immaturity（big-multiple 但对手弱）· 标 `opp-immaturity`·**禁称硬赢**（涨地盘不涨成色）。
   - **C. 部署变更（非 perf claim）**：measured-gate LMUL flip（q4_0/q4_1→m1 等·vs mf2 自身·非 vs-对手）· de-lottery（baseline<0.8→PASS·vs codegen 抽签）。标"部署/结构·非 vs-对手硬赢"。
   - **D. cross-op / near-parity / 弱势**：贴平或跨算子比较·不算赢。
   - **E. measured-非-T-N**：r5.1 grid/repack 这批·走 ad-hoc run_*.sh·无 bench 无 T-N（CLAUDE.md:35 现行法）= **无判定资格·论文引用前须 bench 复测**。逐格标。
3. **点名下一个真赢**：给 1-3 个**最可能达到 A 层**的候选格 + 它需要什么（识别真强手调 VLEN128/256 对手 + 走 bench 通道 + 过 T-N）。含 B1 余项（q8/q4_0/q4_1 vs 真-VLEN128-对手·B1 证 ggml x16 是 VLEN256-only 非法对手·真对手=4x8/SpacemiT/block-dot 待识别）。

## 门
- **0 造数**（只读主表+脚本机算·禁手抄 perf-covered 数）· 每格判层带 file:line/ratio/对手身份依据 · **成色铁律**（便宜档禁称硬赢·measured-非-T-N 明标）· 禁"已尽力/资产"式美化 · 未改任何源 · 未 git commit。
- 🔴 禁碰任何在飞 agent 文件（架构 Workflow 只读 / MIG-5 选择器）——本任务纯只读·天然不冲突。

## 交付
写 `experiments/active/perf-honest-scorecard.md`（分层表 + 下一真赢候选）。final message 给：A/B/C/D/E 各层格数 + 真硬赢逐格清单 + 下一真赢候选 1-3 + 各需什么。禁把 B/C/E 层数灌进 A 层。
