# G5-M1b 部署 emitted q8_0(R1 HIT·3/84)

parent: `07-11-G5-wiring`（G5 接线战役）· status: **completed**（completedAt 2026-07-12）。

## 结果 — R1 HIT（decisive positive）
- commit **55022402**：部署我方 emitted q8_0 kernel（**correctness-carrier**·拦截破损上游 [GAP-Q8_0-VLEN128-KERNEL]）。
- **correctness-first 重测**：host emit .inc → deploy 挂点③（#include emitted .inc via deploy_patch）→ board byte-exact → perf 分相。
- **prefill 4.35× / decode 3.812×**（correctness-carrier·决定性正例）——与 M1 的 8.69× MIRAGE(VOID) 相反：M1 翻 gate 只 route 到破损上游 body = e2e garbage（correctness-gate-catches-mirage）；M1b 部署 emitted kernel 拦截破损上游，是**真路径**正例。
- **perf-covered 2 → 3/84**：q8_0 追认为 full-stack 无星号绿格（用户 2026-07-12 裁·登记 `docs/reports/2026-07-12-perf-covered-q8_0-green-3of84.md`）。

## 意义
M1b 是 G5 接线战役 L①链路层的决定性正例：证明「部署 emitted kernel 作 correctness-carrier」的接线机制成立，为 M2（L② scaffold / L③ bridge 铺线）提供 deploy 机制先证。decode 3.812× 同时暴露张力 A（selector lean-decline 被证伪 → `07-12-G5-tensionA-selector`）。
