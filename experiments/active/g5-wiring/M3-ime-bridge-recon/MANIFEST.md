# G5-M3-ime-bridge-recon — IME e2e forward bridge 侦察 · CASEFILE

> **campaign**: G5 接线战役 · **M3-ime-bridge-recon = L-接线③ IME e2e 桥部署就绪度侦察**（用户 ③ 优先级 · `[GAP-IME-E2E-INTEGRATION]` 评估）
> **status**: ACTIVE · recon/scout 交付（纯 docs）
> **role**: 纯 scout/docs — 本仓 read-only 静态分析 + k1 板 **read-only 核**（禁执行集成 · 禁 build/deploy · 禁 git · 板不改）。零 lib/schema/ROADMAP 触碰（只读引）。
> **HEAD (TianChen-RV)**: `61596dcc` 未变。
> **产出**: `deploy-readiness.md` —— IME 3 格（q4_0/q8_0/q4_K@ime）emitted 状态盘点 + k1 vendor mul_mat 派发链活核 + 集成方案（tcrv MAC 微 tile 叶子 ↔ vendor 完整 GEMM 的 5 缺口）+ 工作量级/分批（多 session·4 correctness-critical 桥）+ perf 诚实预期（Amdahl 同域·vendor-ceiling·micro↛e2e·likely 无 e2e green）+ 建议（perf 名义不值当下做·曳光弹首格 q4_0@ime）。
> **一句话结论**: IME 桥 = 最重 L-接线③（多 session·无翻 gate 白嫖·换板 k1）；N2 结构价值真实（C1 跨范式 extensibility 闭 `[GAP-IME-E2E-INTEGRATION]`）但 perf payoff 诚实近零（vendor 全接线 IME GEMM 同域 toggle 已证无干净 IME-unit e2e 赢·tcrv 2.09× compute-account 不传导）；建议 perf 名义不做·若为 C1 结构完整性做则曳光弹首格 = q4_0@ime（零 provisioning·最简 scale fold·seal-proven tile·correctness-first·预注册非绿）。
> **provenance 锁**: 板 vendor 集成机制（`spacemit/ime.cpp` `forward_mul_mat:234` / `get_optimal_repack_type:1257` / `gemm_kernel_i8i4`）+ tcrv 0-hit 全部来自 **k1 板 `/home/bianbu/tcrv-k1-llama/ggml`（read-only 活核·2026-07-12）**；emitter ABI（`IMEBackendEmissionDriver.cpp:1116/1237/1358`）+ Apack/Bq4 布局（`test/Target/IME/*.c` seal harness）来自**本仓 read-only**。perf 数（vendor toggle 1.61×/1.46×）引自 `t6-k1-ime-q4k-e2e`（不重测）。

## durable files
- `deploy-readiness.md`
