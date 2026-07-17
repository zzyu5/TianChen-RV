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
- **三具名 GAP 修复评估**（本地·并行）：[GAP-CLANG-GATHER-TRAP]/[GAP-DEQ-KQUANT-UNPACK]/[GAP-FWD-M8-VSETVL] 根因+可修性+Amdahl+排序（task `07-12-G5-gap-assessment`·**completed 2026-07-12**·workflow `wez16hgku`）。

## 2026-07-12 收口追认 + 四问裁决

用户 2026-07-12 裁（四问定性 + 三线并行授权 + deferred 批准）。逐字要点：

- **FLAT 家族全绿 6/84 追认**：L② 新建上游 scaffold 建法可复制已证（q8_0→q4_K→q5_0/q5_1），perf-covered 追认 **6/84**（task `07-12-G5-M2-q5x-tracers`·completed·q5_0/q5_1/q4_1 曳光弹 perf green）。
- **perf-covered 终态 = 零未定义格**：84 格每格必落声明制（certified / 带账 yellow / 域外例外），禁未定义（deferred task `07-12-perf-zero-undefined-cell-classify`·planning·随接线线逐格判读）。
- **q8_0 升 T8**：q8_0 MIRAGE 案（M1 8.69× VOID vs M1b 4.35×/3.812× 真路径）升格入 T8 = correctness-gate-catches-mirage + upstream-VLEN-碎片化（论文双用·commit afbb49e2）。
- **接线复利 C2**：net-new scaffold 一次建、逐格复用（q4_K→q5x）= 模板经济学 C2 边际成本递减证词。
- **F-3 = family-manifest**：F-3 收窄定义为「清单 ⊆ 检」（每家族 manifest + check_family_locality.py + CI），非跨根搬迁（task `07-12-F-3-family-manifest`·in_progress·线②）。
- **IME bridge 批准（跨范式名义）**：IME e2e bridge 立项批准，名义 = 跨范式完整性、**非 perf**（deferred task `07-12-IME-forward-bridge`·planning·gated on K-quant 收口·k1 板·≥3 session）。
- **RENAME+schema 批准（独占会话）**：pattern 编号/registry + 方言前缀 + 项目改名 批准；变更表准备（task `07-12-RENAME-schema-changetable`·in_progress·线③）→ 机械替换独占会话（deferred task `07-12-RENAME-schema-exclusive-session`·planning·gated on K-quant 收口 + 变更表）。
- **三线并行（K-quant / F-3 / RENAME）**（用户裁五）：
  1. **K-quant** — q6_K 净新接线曳光弹（测 net-new scaffold 是否延伸 K-quant）·task `07-12-G5-K-quant-q6_K-tracer`·wf/agent `ac896a2e`。
  2. **F-3** — family-manifest 机检·task `07-12-F-3-family-manifest`·agent `a3a1173b`。
  3. **RENAME** — 变更表准备（独占会话铺路）·task `07-12-RENAME-schema-changetable`·agent `a921c8f4`。

## 2026-07-12（续）· 裁决全消费执行成果 + 后续裁决落地

**用户 2026-07-12 第2/3 裁决（分类纠偏 + q6_K 定案加固 + K1 精准打击立项）全面消费落地**（详 `docs/ROADMAP.md` 头 + `docs/reports/2026-07-12-perf-covered-*.md` + T8 ledger + memory）：

**裁决消费清单（committed）**：
- **一 单位统一 recon**（`2c53e581`）：`perf_covered_metrics.py` 机算 perf-covered = **6/83**（fold q4_0-gemm regime·any-board·三处同源·anti-gate·CI job `perf-covered-recon`·禁手填）·M4 84/91 不动·备选 7/84 flagged 待用户复核。
- **一.2 IME 重分类**（黄-传导稀释→黄-未接线·bridge 未建）· **一.3 六类冻结**· **三 声明例外台账过审**（27 格五字段 + anti-gate + 物理墙 12 roofline）。
- **〇.1 [C1-4] 落位** · **〇.2 q6_K T8+T3p-X** · **二.1 q6_K L-7 反汇编钉死→转正**（`8f80df11`·first-emission 非部署病·fallback 三链排除·黄-对手更强 final）· **二.2 L-11 同域标注**。
- **四 K1 精准打击**（`17c193eb`）：opponent_map（q8_0-k1 硬对拼格 + 碎片化2nd证据）· **k1 编译器对称订正**（k1 stock=clang-18→kernel-axis VALID·[CASE-COMPILER-ASYMMETRY]-k1-refinement）· q5_K [X-0] 声明例外。

**★里程碑：perf-covered 6/83 → 7/83**：
- **[WORK-ITEM-K1-KQUANT-E2E] RESOLVED-POSITIVE**（`b12afd57`）：q4_K repack e2e 在 k1-clang 传导 2.644×·**rvv K-quant e2e LOSS=gcc-742-spill-death 非 weight-recon wall**·C3′ 绿路径确认·q6_K 收口 framing 精化 weight-recon→compiler-codegen。
- **★q5_K@k1 NEW GREEN**（`47e29b35`）：首个 our-kernel K-quant e2e 传导（prefill 1.641×·净新 dispatch·裁四.2 k1 唯一新绿点兑现·第2 K-quant transduction with our kernel）→ perf-covered **7/83**。
- iq4_nl correctness-carrier GREEN→firm yellow-对手更强（`d0bf1ff8`·codebook 家族首·[GAP-VLEN128] 碎片化第3族）。

**IME bridge（裁三·跨范式名义·family#2 forward）**：session-1（`13c1c74a`·scale-fold+shape driver）· session-2（`3b6ff278`·数据路径 correctness on REAL ggml + 单tensor A==B bit-identical + hook reachability）· **session-3 在飞**（forward traffic routing → forward-wired·family#2 forward 完成硬门）。

**RENAME+schema**：变更表已备（`64a60ae5`）·独占会话待用户命名（B 方言前缀 / C 项目名）·排 K-quant 收口后（已收）。
