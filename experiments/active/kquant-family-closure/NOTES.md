# NOTES — G3 裁决二.2/二.3 K-quant 家族收口：整模型传导账 + q4_K 八门

## What this cell is
族收口大动作（K-quant 5/5 front-door 构造 + kernel-轴 tiling 后）的**整模型兑现率**账：
1. **二.2 传导账**（`transmission_account.md`）：实测 Q4_K_M prefill 相 matmul 占比 × 构造 kernel 增益
   → e2e prefill 增益 Amdahl 上限区间。
2. **整模型 llama-bench 分相**（`phase_split.json` + 下 §2）：stock 基线 pp/tg 分相 + measured e2e Δ 的
   **集成 blocker**。
3. **二.3 q4_K 八门**（`T-PERF1_q4_K_vlen128_prefill_8gate.md`）：逐门 green/missing。

板：`ssh rvv`（openEuler VLEN128 64c，gov=performance 2.6GHz）。ggml `f3e1828`。
模型：`DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf`（sha256 `f8eba201522ab44b`）。

## 0. Pre-registration（先于数字）
- **登记规则**：measured e2e prefill 显著加速（T-N：Δ>2×floor 且 >2% 且 IQR 不重叠）分相绑相 → 登记首个
  "构造 kernel 群→整模型 prefill 增益" 数字。
- **不显著/无 measured** → 如实报 kernel-axis 胜 + **e2e Amdahl 稀释账** + 归因（哪环稀释）。档案价值，不 claim Win。
- **集成卡** → 报确切 blocker + 传导账 projection 作档案估计。
- **协议**：preflight(0) 工具链对称 / cold(warmup-dropped) / paired / N≥10 median+IQR / T-N floor via passes / restore。

## 1. 二.2 传导账（headline）——见 `transmission_account.md`
- prefill 实测 self%：**q4_K vec_dot 79.1%**（78.35 vl128 + 0.73 generic）+ **q6_K 18.2%** → **matmul ≈ 97.3%**。
- Q4_K_M 是混合量化（多数 q4_K，`output.weight`/部分张量 q6_K；**无 q5_K**）。
- Amdahl（q4_K@**1.884×** + q6_K@1.0× + 余项不变）：**S_prefill ≈ 1.59×**。
- **区间 1.3×（保守传导折损）– 1.6×（满额）**，机理天花板 **1.84×**（纯 q4_K 模型）。
- **decode**：95% 仍在 vec_dot，但 **M=1 GEVM memory-bound** → 1.884×（M>1 tiling 增益）**不存在** →
  **传导效率≈0，projected decode Δ≈1.00×（NULL）**。整模型生成 decode-dominated → 净生成 tok/s 增益≈NULL；
  价值严格住 **prefill/prompt-processing 相**。

## 2. 整模型 llama-bench 分相 + measured e2e Δ = **BLOCKED**（确切 blocker）
measured "ours(构造 repack GEMM) vs stock" e2e Δ **无法取得**，机理（非放弃）：
1. **构造 kernel 未接入**：q4_K repack GEMM 活在编译器 emitter（`emitRepackKQuantGemmBodyQ4K`,
   `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`，working-tree 未 commit），**未热插进 llama.cpp/ggml 的
   Q4_K `mul_mat` dispatch**。
2. **无 ggml K-quant repack 开关**：不同于 G2（ggml 自带 fusion ON/OFF env-toggle 可同二进制 A/B），
   ggml 在 VLEN128 下 **K-quant repack trait = `nullptr`**（`kquant-l1-q4k-q5k-repack-prefill/dispatch_probe_raw.txt`）
   → **没有可翻的开关**，出厂路径**就是** block-dot。
3. **热插所需胶水超出本 session touch-set**：需 (a) 离线把权重 repack 成 `block_q4_Kx16` 布局，(b) ggml `mul_mat`
   新 dispatch 入口 + q8_K 激活量化胶水——历史 e2e-seal 难点（memory：q4_K emitter scaffold timed out 2×）；
   即便板上 patch 临时副本也要整套 repack+dispatch glue，非单会话可封。
→ 依裁决："集成卡 → 报确切 blocker + 传导账 projection 作档案估计"。**projection = §1 的 ≈1.59× prefill 上限**。

**本 session 仍取得的 measured**：stock Q4_K_M 分相**基线** pp/tg 吞吐 + T-N floor（`phase_split.json`），
把 projection 锚到真 e2e 吞吐（Amdahl 分母 = 实测 profile，非估）。measured-vs-projected 对账的 **measured 列
= N/A（blocked）**，传导效率列 = projection-only（prefill 上限 1.59× / decode NULL）。

## 3. logits / 数值门
- kernel-axis **byte-exact 已封**：S6 IDENTITY cmp 0 mismatch（tiled==golden）+ 构造 oracle WORST_NORM ~7-8e-7
  （bounded-ULP，非 ULP0——浮点 FMA-fold）。
- e2e logits bit-exact/ULP：**contingent on 集成**（无 "ours" 可比）。stock 路径 greedy 决定性沿用 G2 已证的
  harness 决定性（同二进制同 seed 逐 token 一致）。

## 4. 八门逐门（见 `T-PERF1_q4_K_vlen128_prefill_8gate.md`）
**3/8 PASS · 2 partial · 3 missing** → q4_K **不是** sealed Win，beat 措辞 LOCKED ([NG-4])。
- PASS：①字节精确(bounded-ULP) ⑥实验纪律 ⑧措辞门。
- PARTIAL：③双板objdump(128✓/256✗) ④micro∧e2e(micro✓/e2e=projection BLOCKED)。
- MISSING：②VLEN256 flip lit ⑤双板(k1未测) ⑦selector 归因(非能力键选中)。
- 下板批：q4_K VLEN256 lit + k1 objdump/复测 + e2e 集成 or measured A/B + tiled 变体能力键归因。

## 5. Restore
主树 `lib/`+`build/` **未动**（read-only；无重建、无 `git stash`）。板 scratch 全在 `/tmp/kqclose/`
（perf .data + phase_split.json + 脚本）；模型/二进制预存未改。本 cell 只新增 `experiments/active/kquant-family-closure/`
+ 板 harness `tools/e2e-harness/board/kquant_transmission_amdahl.sh`。未 commit（用户提交）。
