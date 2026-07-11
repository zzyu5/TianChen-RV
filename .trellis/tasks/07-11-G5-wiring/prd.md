# G5 接线战役 — emitted tcrv kernel 接入 ggml 真实 forward

接线 = 补丁/链接层集成(NG-2 不动·不做图框架)·接线 ≠ 自动转绿(micro↛e2e 铁律仍管辖)·板 A-tree 可逆+restore。perf-covered 当前 2/84·唯一拉绿杠杆 = 本战役。

## 里程碑表

| M | 内容 | 状态 | provenance |
|---|---|---|---|
| M0 | 接线机制解剖(两物理挂点:dispatch gate repack.cpp:4592/4713 + kernel arch/riscv q8_0 gemv:518/gemm:1426;挂点③=#include emitted .inc via deploy_patch) | ✅ done | commit ab054260·casefile experiments/active/g5-wiring/M0-接线机制解剖.md |
| M1 | 曳光弹 q8_0(翻 dispatch gate routing) | ✅ done·**R4 correctness RED** | commit f8b8dabb·[GAP-Q8_0-VLEN128-KERNEL](上游 q8_0 VLEN128 repack kernel 数值破损·ggml repack.cpp:230/240/285/339 硬编码 AVL=16→VLEN128 钳 vl=8→garbage)·翻 gate 只 route 到破损上游 body=e2e garbage·8.69× MIRAGE(VOID)=correctness-gate-catches-mirage 正面教材 |
| M1b | 部署我方 emitted q8_0 kernel(correctness-carrier·拦截破损上游) | ✅ **done·R1 HIT** | commit 55022402·prefill 4.35×/decode 3.812× correctness-carrier 决定性正例·task 07-12-G5-M1b-q8_0-deploy(completed) |
| M2 | 铺线(部署 emitted·各格)·**用户 2026-07-12 裁·三级接线分层** | 🔵 in-flight(L② 首格 q4_K) | 见下「2026-07-12 进展」 |
| M3 | 收口(接线机制文档化) | ⚪ pending·gated on M2 | task 07-12-G5-M3-wiring-doc(planning) |

## 2026-07-12 进展

- **M1b ✅ R1 HIT**（commit **55022402**）：部署 emitted q8_0 kernel = correctness-carrier·拦截破损上游 [GAP-Q8_0-VLEN128-KERNEL]，**prefill 4.35× / decode 3.812×** 决定性正例（与 M1 的 8.69× MIRAGE(VOID) 相反=真路径）。task `07-12-G5-M1b-q8_0-deploy`（completed）。
- **q8_0 追认 perf-covered 3/84**（用户 2026-07-12 裁）：full-stack 无星号绿格·登记 `docs/reports/2026-07-12-perf-covered-q8_0-green-3of84.md`。
- **M2 三级接线分层**（用户 2026-07-12 裁）：
  - **L①链路层**（翻 dispatch gate routing）：q8_0 / q4_0 已穷尽。
  - **L②新建上游 scaffold**（真工程）：q5_0 / q5_1 / q4_K；**首格 = q4_K 曳光弹**（task `07-12-G5-M2-qk-tracer`·in_progress·workflow `wyhnspmt6`；成功后 q5x 复用 = 占位 task `07-12-G5-M2-q5x-tracers`·planning·gated·→6/84）。
  - **L③跨框架 forward bridge**：IQ + IME e2e（后续）。
- **张力 A（selector-fix 在飞）**：M1b decode 3.812× 证伪 selector lean-decline → 能力键控修强化 q8_0 绿格（task `07-12-G5-tensionA-selector`·in_progress·workflow `wrt3za9le`；撞 canon 则停+surface）。
- **三具名 GAP 修复评估**（本地·并行）：[GAP-CLANG-GATHER-TRAP]/[GAP-DEQ-KQUANT-UNPACK]/[GAP-FWD-M8-VSETVL] 根因+可修性+Amdahl+排序（task `07-12-G5-gap-assessment`·in_progress·workflow `wez16hgku`）。
