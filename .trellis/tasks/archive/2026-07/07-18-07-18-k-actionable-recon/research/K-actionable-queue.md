# Research: K线 concrete actionable 工作队列勘察（《开测篇》§二 stop-cond 1）

- **Query**: map concrete actionable K线 work per stop-condition 1；区分 actionable-work / user-gated / 架构不可达；纠正过早「exhausted」宣告
- **Scope**: internal（master T3 + issues + harness cells + runs.log + fanout ledger）
- **Date**: 2026-07-18
- **权威源**: `experiments/master/T3_master_rebuild.csv`（108 行 op 格 × 2 板）· `.trellis/spec/issues/` · `tools/bench/cells/` · `experiments/active/result-tables/K-attack-fanout-ledger.md` · `experiments/runs.log`

---

## 〇、总体三分（核心区分）

| 类 | 定义 | 本轮判定 |
|---|---|---|
| **actionable-work** | 我方能做：建 harness + bench 重测/攻坚·非 gated | **大量存在**（下列 6 项每项都有）—— 过早「exhausted」是错的 |
| **user-gated** | 待用户裁：ISSUE-105 / 107 / 031 / 035 / 099 设计分叉 / 099 scope | 有边界但**不覆盖全部剩余工** |
| **架构不可达（经环证实·可具名上报）** | 完整攻坚环走完·带逐指令墙证据 | 目前仅 **1 格铁证**（nvfp4@rvv vec_dot·ISSUE-100）+ grid dequant 天花板族（ISSUE-107·非不可达但 gated） |

**★纠错点**：多数剩余非 PASS 格的真前置 = **harness 未建**（可提案即建·ISSUE-096 已 RESOLVED 的 bench 通道），**不是** user-gated。把「harness 缺口」误当「gated」= 本勘察要纠的错。

---

## 一、gcc 28 格重测（§二.2 清欠先行 · ISSUE-099）

**真 gcc 污染 = 8 格**（`audit_gcc_lane.py` 机算·DATA=gcc ∧ 未 CLANG_WORLD ∧ 在分母），非 28：

| 子族 | 格（5+3=8） | harness 状态 | actionable? |
|---|---|---|---|
| **FLAT gemm prefill** | q4_0 / q4_1 / q5_0 / q5_1 / q8_0 @prefill | `gemm_tile.sh` **只认 4 IQ 格**（iq1_s\|iq1_m\|iq3_xxs\|iq3_s）→ FLAT 全落 `HARNESS-VOID exit 2`；叶子资产亦缺（`kernels_grid4/` 只有 iq1/iq3 四叶） | **半-gated**：现值来自【部署 repack 管线】(e2e 绿·对手=部署 block-dot)，非 P2 standalone。补 P2 叶子复现的是【另一件事】(另对手/oracle)。「FLAT 重测走 e2e/repack 还是 standalone 微核」= **真设计分叉·待裁**（ISSUE-099 候选去向②） |
| **product_reduce** | codebook_n3 / offset_binary_n3 / q4_0_nibble | `bench product_reduce ... → CellRecipeMissing`（`cells/product_reduce.sh` **不存在**） | **actionable-work**：N3 gearbox 归约子原语·**可直接建 harness**（ISSUE-099 明列「可直接建」）。建成即解锁 3 格重测（offset_binary_n3@k1 现为具名-X·其余 PASS） |

**7 格已 CLANG_WORLD·不需再测**：iq2_s/iq2_xs/iq2_xxs/iq4_xs/mxfp4/tq1_0/tq2_0（gemm_tile 格·已 clang-18 override）。
**★恒等式门已死**（表5 空心性测试：即便输入 100% 自认 gcc·门仍读 0·master 逐字节不变）——只认真测。

**actionable 结论 item1**：建 `product_reduce.sh` = **确定 actionable**（1 族 harness·解锁 3 格）；FLAT 5 格 = **待设计分叉裁**（gated·非我方能自决对手政策）。

---

## 二、dequant 重测全部完成（§四.1 · 18 格式真向量发射器 · ISSUE-001/107）

**dequantize_row 共 24 格**（23 在分母 + q1_0 域外）。§四.1 目标 = 每格发**真向量 kernel**（复用 C4a 参数化·PR-31 销案）终结 codegen 抽签。

- **已落真向量 emit = 1 格**：`iq3_xxs@rvv`（首攻·owned de-lottery·byte-exact GREEN·[L-8] 满足·`runs.log` 20260718T000000Z）→ cold 0.360 **具名-X**（HW-gather 天花板·ISSUE-107·**非架构不可达**·有标量-load lever gated 调度成熟度）。
- **其余 22 格现值 = autovec-lottery PASS/具名-X**（CLANG_WORLD 单世界 clang·**非真向量 emit**·全部标 interim·ISSUE-001 未解）。§二.5：dequant 轴数字全标 interim·不进头条。

**harness**：`dequantize_row.sh` **已建**（rvv·首攻 iq3_xxs·含 zero-vector objdump 探针 + ZERO-MODEL byte-exact + 四臂反空心）。但：
- 格白名单**只认 iq3_xxs**（`*)→HARNESS-VOID exit 2`）→ 扇出其余格须**加白名单格 + 造该格真向量 leaf 资产**。
- **@k1 硬 VOID**（`board=k1 → exit 2`·gated on ISSUE-105 VLEN256 半宽）。

**分族（真 actionable vs 天花板）**：
| 族 | 格 | 判定 |
|---|---|---|
| **grid-codebook（gather 天花板·ISSUE-107）** | iq3_xxs✓ / iq3_s / iq2_xxs / iq2_xs / iq2_s | 真向量(HW-gather) decode 撞同一 ~0.36 天花板·**owned 真向量到不了 PASS**。三选一措辞（真向量 具名-X / 标量-load owned ~parity / 保 autovec PASS）= **user 战略裁**（ISSUE-107 待裁）。iq1_s/iq1_m 亦 grid 但 vluxei16 大 grid 是 win（[emitter-maturity-vluxei16]） |
| **non-grid（可能 PASS·真 actionable）** | q4_0/q4_1/q5_0/q5_1/q8_0/q2_K/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs/mxfp4/nvfp4/tq1_0/tq2_0 | **actionable-work**：author-owned 真向量 emit + 板测·**我方能做**（gated 仅在 emit 侧先落地 + 每格 leaf 资产 + harness 白名单）。真前置 = R 线真向量发射器工程 |

**actionable 结论 item2**：`dequantize_row.sh` harness 骨架**能复用扇出**（加白名单 + 造 leaf）；真 actionable = **~16 non-grid 格需 author-owned 真向量 emit + 板测**（emit 侧工·可分批）；grid 族 5 格 = 天花板·**gated on ISSUE-107 战略裁**（禁为 PASS 强推·白费板时）。

---

## 三、手调档全数进环冲 0.8（§二.4 机制列队 · K-attack-fanout-ledger）

**手调档非 PASS 格**（vec_dot K-quant + iq 系 + gemm_tile decode-M1）。**扇出台账 5 机制**（`K-attack-fanout-ledger.md`·源 = attack-dissect 解剖·trellis-check 核）：

| 机制 | 命中格 | 状态 | actionable? |
|---|---|---|---|
| **① VLEN256 宽化 leaf**（原「满展开」已诊断证伪·re-scope·ISSUE-102 待裁仅标签） | 8 格 iq1_s/iq1_m/iq3_xxs/iq3_s @rvv,k1 | **首个扇出成功 = iq3_xxs@k1 proven WIN 1.38**（板测·byte-exact·trellis-check 复核）·**但 NOT deployed**（deployed 叶仍 VLEN128=0.65·gated ISSUE-105） | **actionable-work（施工）** + **deploy gated（ISSUE-105）**。iq3_s@k1 / iq1 系**同病·likely 同 lever·未证**·逐格须 byte-exact+板测（禁外推计数） |
| **② [CASE-KQUANT-GCC-CODEGEN]**（gcc→clang·仅排序不翻正） | q4_K/q5_K@rvv decode + rvv-gcc K-quant decode 面 | 候选未施工 | actionable 但**单独不够翻 PASS**（clang 残差 0.361/0.496 仍 <0.8·须叠 ③） |
| **③ [MECH-WEIGHT-RECONSTRUCTION-BOUND]**（super-block 位重建·(SEW,LMUL,VLEN) 参数化） | q4_K/q5_K@rvv decode M=1 · vec_dot q4_K@rvv | 候选未施工 | actionable·**翻 0.8 未证**（rvv VLEN128 q4_K repack 即使 prefill 只 0.94 parity·不泛化·须丙板测） |
| **④ [GAP-DEQ-ZERO-VECTOR-EMISSION]**（真向量 dequant emit·= 上 §二 R 线） | 9 格 dequant | cold 未接线 | 同 item2 |
| **⑤ nvfp4 FP4 codebook-gather**（vrgather 16-entry·mxfp4 路已存） | dequant nvfp4@k1（0.743 scalar loss） | 候选未施工 | actionable·**弱赢**（目标 ≥0.8·opp=scalar-ref 便宜档·非硬赢） |

**harness**：**vec_dot K-quant / vec_dot-iq 无专用 harness**。gemm_tile.sh 只 4 IQ 格·scalar_vec_dot.sh 只 scalar+tq2_0。⟹ vec_dot 手调格攻坚缺 harness（须建 `vec_dot.sh` 族·注意 ISSUE-104 命名：跨板须区分 scalar-scoped）。

**actionable 结论 item3**：机制①施工（iq3_s@k1/iq1 系宽化）= actionable-work（但 deploy gated ISSUE-105）；②③⑤ 施工 = actionable-work（须叠加 + 板测·多数须建 vec_dot harness）；**「攻坚候选翻 0.8 多须板测(丙)证实 = gated on bench harness 基建」**。

---

## 四、K 线格全数完整攻坚环（§二.3 · 逐格解剖→构造→前门→板测→cold）

**具名-X 统计**：rvv 21 格（含 7 decode-M1）· k1 22 格（含 4 decode-M1 + 2 IME）。

**走过完整环的**：
- `iq3_xxs@k1`（机制①·proven WIN·byte-exact·objdump 真宽·2 seed·trellis-check 复核）——**证架构可达·gated on 部署 ISSUE-105**。
- `iq3_xxs@rvv` dequant（owned 真向量·byte-exact·objdump 双证）——**证 HW-gather 天花板·ISSUE-107**。
- `nvfp4@rvv` vec_dot（构造 re-roll→G1 byte-exact GREEN→G2 板测 NULL 1.00→反汇编 spill 前后不变）——**架构不可达铁证·ISSUE-100·置顶终审 census**。

**经环证实【架构不可达·可具名上报】**：
- **ISSUE-100**：`vec_dot·nvfp4@rvv`（[GAP-NVFP4-VLEN128-SUBBLOCK-GRANULARITY-SPILL]·44/371 指令 0 贡献·三态 e8m1/e16m2/e32m4 强制 spill·对手 ggml 架构覆盖空洞非选弱对手）。
- **ISSUE-107 天花板（非不可达·gated）**：grid dequant 族 owned 真向量 ~0.36 天花板。
- **iq4_nl@k1 decode**（master line 87·完整环走完·机制级墙 = codebook-gather-bound·64×vluxei16 vs 对手 32×vrgather·判别键=codebook 尺寸·**可修=emitter 成熟度缺口**·非不可达）。

**未走环的**：其余 ~18 具名-X 格（q2_K/q3_K/q4_K/q6_K/q5_K vec_dot@双板·iq 系·gemm_tile decode-M1 群）。多数**已有 P2 解剖墙证据**（对手 VLEN 专化 full-unroll 150–314 ins vs 我方 generic aux32 13k–21k ins）→ 归机制①②③扇出，**未走完 前门→板测→cold**。

**actionable 结论 item4**：**未走环格的前置 = harness**（vec_dot 族 / FLAT gemm 族 harness 缺）。走环 ⟹ 建 harness 是必经。仅 nvfp4@rvv 达「架构不可达」终态。

---

## 五、VOID 清偿（§二.2）

- **runs.log 存量 VOID = 1 行**：`20260717T172907Z-iq3_xxs-rvv-67bc1c5d·VOID-脏板`（静板预飞不过·transient）。**已被下一行 supersede**：`20260717T173052Z-iq3_xxs-rvv-65332b85·near-parity·cold_X=0.9542·master-updated`（同格立即重跑成功）。
- **master 主表 VOID 格 = 0**（disp 计数无 VOID 值·全为 PASS/具名-X/pending/域外/N-A-hw）。
- **bench runner VOID 出口 = 四值**（`tools/bench/bench:217-222`）：VOID-脏板（静板预飞）/ VOID-我方错 / VOID-对手废 / VOID-派发不符（后三 = 对拍三验）。task 文「三出口」应指对拍三验的后三类。

**actionable 结论 item5**：**VOID 存量实为 0-1 条 transient·已自清**（重跑即偿）。**无待清 VOID 积压**——新 runner 体制下 VOID 是即时重跑事件·非欠账。此项**基本已偿**。

---

## 六、pending ≤ 1（§七终审门）

**现 pending = rvv 7 / k1 5**（远超 ≤1）：

| 板 | 格 | pending 原因 | 可测(走环)? / 待裁? |
|---|---|---|---|
| rvv | vec_dot\|mxfp4 | pending-fold | **可测**（fold=decode/prefill 聚合待落·harness 待补） |
| rvv | dequantize_row\|iq2_xs | pending(照测未定verdict) | **可测**（全表唯一该值·照测过·verdict 待定·走 dequant harness） |
| rvv | gemm_tile\|q4_1@decode | pending-fold | **可测**·gated on gemm harness（FLAT·ISSUE-099 分叉） |
| rvv | gemm_tile\|q8_0@decode | pending-fold | 同上 |
| rvv | gemm_tile\|iq4_nl@decode | pending-fold | **可测**·iq4_nl 有 harness? (gemm_tile 只 4 IQ·iq4_nl 不在)→缺 harness |
| rvv | gemm_tile\|iq4_nl@prefill | pending-fold | 同上 |
| rvv | gemm_tile\|nvfp4 | pending-真(待补标量仗) | **可测**·待补标量仗(scalar_vec_dot 族·nvfp4 harness 缺) |
| k1 | vec_dot\|mxfp4 | pending-fold | 可测·harness 待补 |
| k1 | gemm_tile\|q4_1@decode | pending-fold | 可测·gated FLAT harness |
| k1 | gemm_tile\|q8_0@decode | pending-fold | 可测·gated FLAT harness |
| k1 | gemm_tile\|nvfp4 | pending-真(待补标量仗) | 可测·待补标量仗 |
| k1 | gemm_tile\|q8_0@ime | pending(IME 结构·该板无合法对手·q8_0 落 RVV-repack 回退·缺 vendor-IME 对手·探针证据) | **真待裁/结构**（q8_0 不进 vendor IME dispatch·结构无 IME 对手·探针已证·非 harness 可解） |

**actionable 结论 item6**：12 pending 中 **11 可测**（gated on harness：dequant 扇出 / gemm FLAT / iq4_nl-gemm / nvfp4 标量仗）；**1 真结构**（q8_0@ime·无合法对手）。pending→≤1 的路径 = **建 harness 逐格照测落 verdict**·非待裁。

---

## 七、★harness 建设量估计（解锁多格的杠杆）

| harness 族 | 现状 | 建设量 | 解锁 |
|---|---|---|---|
| `product_reduce.sh` | **缺** | 中（N3 gearbox 归约·copy gemm_tile 骨架） | 3 格（codebook_n3/offset_binary_n3/q4_0_nibble·gcc 清欠 item1） |
| `dequantize_row.sh` **扇出** | **已建·只认 iq3_xxs** | 低-中（加白名单格 + 每格造真向量 leaf·decode-only oracle 比 gemm 简单） | ~16 non-grid dequant 格（item2）+ grid 族(天花板·gated 107) |
| `vec_dot.sh`（K-quant + iq 系） | **缺**（gemm_tile 只 4 IQ·scalar_vec_dot 只 scalar/tq2_0） | 中-高（ISSUE-104 命名须区分跨板/scalar-scoped·CROSSOP oracle） | K-quant vec_dot 手调族（q2_K/q3_K/q4_K/q6_K/iq 系·item3/4 攻坚环前门→板测） |
| FLAT-gemm 叶子（gemm_tile 扩格） | **缺叶 + 设计分叉** | 中·**但对手政策待裁**（部署 repack vs standalone·ISSUE-099②） | 5 FLAT gcc 格重测(item1) + q4_1/q8_0 decode pending(item6) |
| `dequantize_row.sh` **@k1 分支** | **VOID·gated ISSUE-105** | 低·但 gated | k1 dequant 全轴（gated 105 半宽） |
| `forward` / `quantize` harness | 未见·但相关格全 PASS | — | 已 PASS·非优先 |

**契约**（`cells/README.md` + ISSUE-090·两现役 harness 逐字共有）：`<op>.sh <board> <mode> <fmt>`；harness **禁写任何仓库侧持久文件**（全 stdout·bench 解析后经 runner fail-closed 落三目的地 `master`/`runs/<run-id>`/`runs.log`）；板端 `/tmp` 临时可；mode ∈ {verify（build+probe+byte-exact+反空心·NO TIMING）/ sanity / measure（cold N 2-seed flush）}；格白名单不认即 `HARNESS-VOID exit 2`（ssh 之前·零板可证）。范本 = `gemm_tile.sh`（骨架）+ `scalar_vec_dot.sh`（zero-vector objdump 探针）+ `dequantize_row.sh`（ZERO-MODEL decode oracle）。

---

## 八、三分总账 + 最高价值先做

### actionable-work 队列（我方能做·非 gated）
1. **建 `product_reduce.sh`** → 解锁 3 gcc 清欠格（item1·ISSUE-099 明列可直接建）。**批次：1**。
2. **`dequantize_row.sh` 扇出 non-grid** → ~16 格 author-owned 真向量 emit + 板测（item2·§四.1 R 线主体）。**批次：多（emit 侧分批·每格 leaf）**。
3. **机制①宽化施工**（iq3_s@k1/iq1 系）→ 已有 iq3_xxs@k1 proven 范本·逐格 byte-exact+板测（item3·deploy gated 105 但施工不 gated）。**批次：~2-3**。
4. **建 `vec_dot.sh` 族** → K-quant/iq 手调格走前门→板测→cold（item3/4）。**批次：1 harness + 逐格攻坚**。
5. **pending 逐格照测落 verdict**（11 可测·gated on 上述 harness）→ pending→≤1（item6）。

### 真 user-gated（登记+保守默认·禁停等）
- **ISSUE-105**：k1 部署 re-baseline（per-board VLEN256 fixture）—— 机制①的 proven 赢无部署通道。
- **ISSUE-107**：grid dequant 三选一措辞（真向量具名-X / 标量-load ~parity / 保 autovec PASS）—— 战略裁。
- **ISSUE-099②**：FLAT gemm clang-18 重测走 e2e/repack 还是 standalone 微核 —— 对手政策设计分叉（5 FLAT 格 + q4_1/q8_0 decode pending 卡此）。
- **ISSUE-031**：F-7 决策住址门判据（甲/乙/丙）—— R 线阶段二立门待裁。
- **ISSUE-035**：strip-width 目标形态 retarget（honest-null·结构不可达·[SEL-3]）。

### 架构不可达（经环证实·可具名上报·置顶终审 census）
- **ISSUE-100**：`vec_dot·nvfp4@rvv`（完整环·逐指令墙·三档墙分类·**唯一铁证**）。
- **ISSUE-107 天花板**（grid dequant owned 真向量 ~0.36·非严格不可达·有标量-load lever gated 调度成熟度）。
- **iq4_nl@k1 decode** 墙（codebook-gather-bound·可修=emitter 成熟度·非不可达）。

### 最高价值先做（勘察建议·raw）
1. **`product_reduce.sh`**（低风险·纯 actionable·解锁 3 格清欠·无待裁·1 批次）。
2. **`dequantize_row.sh` non-grid 扇出**（§四.1 R 线主体·论文立足点·骨架已建·扇出成本低）。
3. **机制① iq3_s@k1/iq1 系宽化**（有 proven 范本·手调档硬赢候选·但须先或并推 ISSUE-105 部署裁以免 proven-not-deployed 堆积）。
4. **`vec_dot.sh` 族 harness**（解锁最大攻坚面 K-quant 手调群·但建设量最高·ISSUE-104 命名先厘清）。

---

## 九、诚实边界 / 未找到

- **未逐格重跑板测**：本勘察纯读 master/issues/harness/ledger·未 ssh 板复算 cold（0 造数）。cold 数字引 master 现值（recon 生成态·sha 守恒）。
- **「28 格」原始清单未逐条列全**：ISSUE-099 权威定真 gcc=8·7 已覆盖；余 13 达 28 之数疑为 dequant DATA=gcc 行（audit 表2「★标签矛盾」组·折入 item2 dequant interim 轴）·本勘察未强行对齐「28」这个历史计数（PR-47 scope 收到 8 = 队列口径·ISSUE-099 候选去向③·用户独占）。
- **pending-fold 语义**：= verdict 待 decode/prefill 聚合(fold)·非缺测·多数 harness 补齐即可落 verdict（流水线与 schema §里列为合法 disp 值）。
- **harness 建设量为定性估计**（低/中/高）·非工时·依赖 emit 侧 leaf 资产是否先落地。
</content>
</invoke>
