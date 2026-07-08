# cell MANIFEST — kquant-family-closure

- **campaign**: repack / [KQUANT-L1] (G3 裁决二.2/二.3 — K-quant 家族收口：整模型**传导账** + q4_K **八门**。
  K-quant 5 超块全 front-door 构造 + kernel-轴 tiling 之后（q4_K 1.884× / q5_K 2.193× / q2_K 1.413×
  HOLDS；q6_K/q3_K construction-only 1.0× weight-bound NULL），本 cell 回答**整模型 e2e 兑现率**：
  构造 kernel 群的 kernel-轴增益能传导多少到整模型 prefill？以及 q4_K 逐 [PERF-1] 门 green/missing。)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-09, `ssh rvv` openEuler VLEN128, cores 8-15, 2.6 GHz,
  gov=performance, ggml `f3e1828`)**. ★ 头条：
  (1) **二.2 传导账**：实测 Q4_K_M（DeepSeek-R1-Distill-Llama-8B, sha `f8eba201`）prefill self% =
  **q4_K vec_dot 79.1% + q6_K 18.2% → matmul 97.3%**（强 matmul-bound）；Amdahl（q4_K@1.884×, q6_K@1.0×,
  余项不变）→ **S_prefill ≈ 1.59×**（区间 1.3×–1.6×，机理天花板 1.84× 纯 q4_K）。**decode**：95% 仍在
  vec_dot 但 M=1 GEVM memory-bound → 1.884×（M>1 tiling 增益）不存在 → **传导效率≈0，decode Δ≈1.00× NULL**。
  (2) **整模型 llama-bench 分相**：measured "ours vs stock" e2e Δ = **BLOCKED**（构造 kernel 未接入 ggml
  mul_mat + ggml 无 K-quant repack 开关，repack trait=nullptr，无可翻开关；热插 glue 超本 session
  touch-set）→ 依裁决报确切 blocker + 传导账 projection 作档案估计；stock 分相**基线** pp/tg 已实测锚定。
  (3) **二.3 q4_K 八门**：**3/8 PASS**（①字节精确 bounded-ULP · ⑥实验纪律 · ⑧措辞门）· **2 partial**
  （③objdump 128✓/256✗ · ④micro✓/e2e=projection BLOCKED）· **3 missing**（②VLEN256 flip lit · ⑤k1 未测 ·
  ⑦selector 非能力键选中）→ **不是 sealed Win，beat 措辞 LOCKED ([NG-4])**。
- **role**: G3 裁决二 K-quant 家族收口的整模型兑现账。**传导账 = 档案 projection**（1.59× prefill 上限 /
  decode NULL），**非 measured e2e beat**；八门逐门给下板批 3 缺项清单。板 harness =
  `tools/e2e-harness/board/kquant_transmission_amdahl.sh`（perf task-clock profile + stock 分相基线）。
  本 cell 只放数据/证据，**不改** result-tables 的 T8/inventory（synth 数据在 cell 内）。主树 `lib/`+`build/`
  read-only（无重建、无 git stash）；板 scratch = `/tmp/kqclose/`。

## durable files
- `NOTES.md` — 裁决二.2/二.3 叙事（pre-registration / 传导账 / 集成 blocker / 数值门 / 八门 / restore）。
- `transmission_account.md` — 二.2 传导账（prefill/decode self% + Amdahl 区间 + decode NULL 机理）。
- `T-PERF1_q4_K_vlen128_prefill_8gate.md` — 二.3 q4_K 逐八门 green/missing（镜像 result-tables 的 q4_0 ledger 格式）。
- `T6_kquant_transmission.csv` — synth T6-schema 数据行（8 门列 + prefill/decode projection；非 canonical 表）。
- `prefill_profile.txt` — 板 perf task-clock prefill self% 原始报告（798K samples）。
- `decode_profile.txt` — 板 perf task-clock decode self% 原始报告（189K samples）。
- `phase_split.json` — stock Q4_K_M 分相基线 llama-bench JSON（prefill pp128 + decode tg32, N=10, median+IQR）。
- `phase_split_env.txt` — 分相基线 env fingerprint（board/pin/threads/model_sha/protocol note）。
