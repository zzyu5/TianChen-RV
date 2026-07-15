# PENDING_RULINGS · 延后裁决登记册（度假自治制 · 常驻工件）

> **制度**（G8 全量 0.8 攻坚总令·一.2 延后裁决制·2026-07-15 生效）：度假期间必问级事项**不再阻塞等裁**——登记于此（编号+问题+选项+agent 建议+采用的保守默认），随后**按保守默认继续推进**。用户回归一次清账。
> **保守默认铁线**：不扩分母 · 不改 canon/$meta 措辞 · 不增头条口径 · 不动队列排序 · 争议格标 pending 而非强判。
> **硬冻结（保守默认也不许碰·只登记不执行）**：canon 分母/roster $meta 变更 · 队列排序变更 · 新头条口径 · 板采购支出。
> **状态**：OPEN=待用户裁 · RESOLVED-BY-RULING=已被后续裁决解决 · APPLIED-DEFAULT=已按保守默认推进待追认。

---

## 登记表

| # | 问题 | 选项 | agent 建议 | 采用的保守默认 | 状态 |
|---|---|---|---|---|---|
| PR-1 | B4 X-SCALAR 真板采购（向量缺席板 rv64gc 无 V）：手头无合规板时是否采购？交期/预算？ | ① 采购 ② 用手头合规板窄豁免 ③ 暂缓仅 spec 落 T-X 定义 | 先盘板源+落 T-X 表定义（案头）·采购决策待裁 | **③**：仅 spec 落 T-X 六列定义 + 盘板源回报·**采购不执行**（硬冻结·板支出）。**2026-07-15 案头回填**：盘点完成（`experiments/active/g8-stage3-attack/B4-xscalar-boardsource-and-TX.md`）——手头【无】物理 no-V 真硅（rvv/k1 均带 V）；**窄豁免 (b)「带 V 板 run-as-noV」路径可辩护·五件验收计划已 design（不施工）**；商用 no-V 板（U74/U54 世代 HiFive Unmatched/VisionFive 1 等）可购但属板支出硬冻结→**采购不执行**。T-X 六列已落 `testing/mlir-testing-contract.md`。灰区：(b) 是否算"合规板"待用户裁；升级触发（①采购真硅→五件重跑去窄豁免标签）挂本 PR。 | OPEN |
| PR-2 | C2 canon [C2-1] 原"<300"轨一口径：〇.2 已定案双 regime（轨一<100/轨二如实实测），[C2-1] 旧条文措辞需 canon 级订正 | ① 改 [C2-1] 条文 ② 保留旧条文加订正注 | 〇.2 已裁口径·条文订正属 canon 措辞（硬冻结） | 文档清除"<300"残留（〇.2 授权执行）·但 **canon 条文 [C2-1] 本体只加订正注不改措辞**（硬冻结） | APPLIED-DEFAULT |
| PR-3 | regime-split decode cells 继承 constructed：per-cell emit-golden 细验（q5x 已验·余 15 格未逐验）是否要求逐格降级未验者？ | ① 逐格 emit-golden 验·未过降 constructed-weak ② 维持继承（扫描证双 lowering 存在即 constructed） | 结构判据已证 lowering 存在=constructed 合法·emit-golden 是更高门·细验属 provenance-hardening 后续 | **②**：维持继承（不降分母·不改 certified 101/108）·细验排 provenance-hardening 后续战役 | OPEN |

| PR-4 | 〇.3 能力措辞：`core-invariants.md:29-31 [I7]` 条文本体（fail-closed）——现系统是编译期 VariantSelection 执行·[I7] 措辞是否加"编译期"限定？ | ① 改 [I7] 条文加"编译期 by VariantSelection" ② 保留条文加旁注 | [I7] 是核心不变量条文（硬冻结·spec 稳定契约）·实现事实已在 index.md/README 层拆分标注 | **②**：[I7] 条文本体零改·仅在实现层文档（已改）标注编译期事实（硬冻结·canon 条文） | OPEN |
| PR-5 | [SEL-3] canon 键定义张力：`执行总纲v2.md:69` [SEL-3] 行写"记录键=kernel+march"·与令〇.4 + 活 schema `tiling-measurements.v1.json` 的 `(declared_instance_hash,kernel,variant)` 不一致 | ① 改 canon 键定义对齐 instance-hash ② 保留加订正注 | 令〇.4 已定 instance-hash 键·canon 条文订正属 canon 措辞（硬冻结） | **②**：设计稿用 instance-hash（令〇.4 权威）·canon [SEL-3] 行只加订正注不改条文（硬冻结） | OPEN |
| PR-6 | B2 §〇.5 物理墙收严定义：是否加 canon 注"内存墙/物理墙 ⟺ 双方同贴同一 roofline 实测证据·单侧 at-wall/对手换数值档=不许标墙"？ | ① 加 canon 收严定义注 ② 维持现 §〇.5 三档 | 收严定义使 gelu/q6_K@rvv/quantize 等 over-claim 归位·但改 §〇.5 = canon 改定义（硬冻结） | **②**：物理墙收严=**canon 改定义硬冻结·只登记**·争议格（gelu/q6_K@rvv/quantize/preduce）标 pending 非强判物理墙 | OPEN |
| PR-7 | B2 sealed 主表输局桶归一：`2026-07-15-G8-主表重铸与对手档位.md` 3 处词表外标签（q5_K opp-immaturity·q5x M=1 floor·iq4_nl@k1 codebook-bound）→ 归 §〇.5 三档 | ① 改主表输局桶标签 ② 维持（sealed 主表·头条口径） | 归一到三档更规范·但改 sealed 主表输局桶 = 头条口径（硬冻结） | **②**：sealed 主表输局桶=**头条口径硬冻结·只登记**·下次 recon 重出时归一 | OPEN |
| PR-8 | B1 falsifier 真二进制门 self-hosted runner 供应（label `weft-build`·持已构建 weft 树）：`falsifier-binary-gate.yml` 已就绪但 runner 就位前 DORMANT | ① 供应 self-hosted runner ② 轻量 CI 从零 build LLVM(破 stdlib-only·canon 工具链) ③ 维持板端手动跑 | 选①(最小)·②破设计原则 | **③**：脚本+yaml 就绪·**runner 供应=基建硬冻结·只登记**·板端手动 `bash run_falsifier_binaries.sh` 可跑（已本地验收真绿/反向红） | OPEN |
| PR-9 | A2 rvv 3 格（iq2_s/iq4_xs/tq2_0）clang-18-micro PASS vs rvv-deployed gcc-15 LOSS：`rvv_0p8_disp` 记 clang-micro(PASS·对称kernel-axis) 还是 gcc-deploy(LOSS·[CASE-COMPILER-ASYMMETRY])？ | ① clang-18-main(campaign 对称域·加 gcc footnote) ② gcc-deploy-main ③ domain-tagged 双列 | campaign 已定 clang-18-symmetric 主表对称域·rvv deploy=gcc footnote(既有 compiler_axis schema) | **①**：clang-18-main(per campaign 主表对称域)·**强制 gcc-deploy footnote(rvv archive 8/8 LOSS·clang-micro≠deployment)**·成色标 clang-micro 防误读·domain 双列策略登记待裁 | OPEN |

| PR-10 | quantize 3(q8_0/q8_1/q8_K)成色+ledger: 对手=arch/riscv手写intrinsic(通用向量)·6/6 PASS=**parity-by-roofline**(对手贴墙满分·非独立beat)·ledger_account 归属(quant-stream子账?) | ① 记通用向量PASS+parity成色注 ② 独立quant-stream子账 | 单分母下已在通用向量档·parity=满分(非beat)·ledger归属=display | **①**：通用向量PASS·**强制parity-by-roofline成色注(非beat·禁外推win)**·quant-stream子账=display登记 | OPEN |
| PR-11 | DEQ/quantize 域=per-lane deploy-matched(rvv gcc-15.2/k1 clang-18)≠matmul clang-18双板对称: dequant rvv opp=gcc-autovec TRUE-VEC·agent建议'升门vec-vs-vec'·但§〇.1(autovec→标量类)+收口令DEQ resolved说标量类 | ① 维持§〇.1标量类(DEQ per-lane compiler_axis注) ② rvv autovec升通用向量 | §〇.1 autovec归标量类·收口令DEQ display已resolved(两板统一标量仗)·per-lane是compiler_axis note | **①**：dequant=标量类(§〇.1权威·收口令resolved)·per-lane deploy记compiler_axis·rvv gcc-autovec不升通用向量 | OPEN |
---

## 已被后续裁决解决（存档）

- ~~falsifier CI 真二进制 toolchain~~ → **RESOLVED**（〇.1 裁定 = 板端 lane·CI 派发真板·废 SKIP-绿·反向测试）。
- ~~capability-model:26 "load-time transitive closure" 措辞~~ → **RESOLVED**（〇.3 裁定 = 两层拆分·现系统"编译期 VariantSelection fail-closed"·load-time 降展望条目·全量替换）。
- ~~perf-covered k1-扩张 / q5@k1 吸纳~~ → **RESOLVED**（收口令〇.1 终裁驳回·any-board 只升成色·维持 9/83）。
