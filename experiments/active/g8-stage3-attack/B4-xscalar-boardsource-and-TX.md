# B4 X-SCALAR 真板 enablement · 板源盘点 + T-X 表定义 + 窄豁免五件计划

> **性质**：纯案头 · DEFINITION-ONLY + DESIGN-ONLY · 不施工 · 不占板 · 无性能主张 · 采购=硬冻结→PENDING（PR-1）。
> **日期**：2026-07-15 · **agent**：G8 证据线案头执行员（独立域）。
> **触碰集**：本文件 + `.trellis/spec/testing/mlir-testing-contract.md`（加 §T-X 定义）+ `docs/PENDING_RULINGS.md`（PR-1 更新）。与 Line A 板测 **不相交**（本文件不 ssh 任何板、不产计时、不动 T3/recon/主表）。
> **裁决依据**：PR-1 保守默认 ③（仅 spec 落 T-X + 盘板源回报，采购不执行）。

---

## 一. 板源盘点（向量缺席真板 rv64gc-no-V）

### 1.1 结论头条

**手头【无】物理向量缺席（rv64gc 无 V / 无 zve）真硅。** 现有两块真板均**带向量**：

| 板 | 身份 | 向量能力 | 是否 rv64gc-no-V 真硅 |
|---|---|---|---|
| `ssh rvv` | openEuler-24.03 · riscv64 · VLEN128 · 64c | `rv64imafdcv…zve32*/zve64*/zvfh/zvfhmin`（**有向量 V**·标量 zbb 有·向量 zvbb 无） | **否**（有 V） |
| `ssh k1` | SpacemiT-X60 · Bianbu-2.2.1 · VLEN256 · 8c | RVV1.0 **有 V** + IME（`xsmtvdotii`） | **否**（有 V + IME） |

依据：memory `hardware-test-access.md` + 归档实证 `experiments/archive/g7/g7-l2-gevm-redesign/G1-static/evidence.md:61`（rvv 板 march 明列 `…dcv…zve…`）+ `experiments/archive/perf-historical/ondevice-q8_0/results_summary.csv`（两板均记 `rv64gc … pure-scalar-noV` **是 baseline 构建，不是板身份**）。**盘点纯案头完成，未 ssh 占板**（Line A 正在板测·触碰集回避）。

### 1.2 关键辨析：no-V **构建** ≠ no-V **真硅**

历史 q8_0 实验在 rvv/k1 上跑过 `-march=rv64gc` 的 `pure-scalar-noV` 基线（`run_rv64gc.txt`：`VLEN(bits)=0 … march=rv64gc`）。这是**把带 V 的板当 rv64gc 目标构建 + 运行**——二进制**真的零向量指令**（objdump 可静态证），板**以标量执行**它。它**不**证明"芯片物理无向量单元"，只证明"该二进制在此板上零向量指令 + byte-correct"。

对 **X-SCALAR enablement**（正确性 / 零向量机检 / 独立家族见证，**无性能主张**），这个区别的载重很轻：

- **能力模型侧**：[F-6] 的"向量缺席实例"本就是一个 **committed 合成 capability instance**（lit 目标 profile·闭包 ∩ rvv.* = ∅），由 `check_f6_scalar_family_independence.py` 经 weft-opt/weft-translate 机检——**不需要任何板**。
- **真硅侧欠缺的只是"物理无 V 芯片上跑过"这一条 cosmetic 证词**，它**加强**而非**替换** committed 合成实例；正确性与 enablement 事实不依赖它。

### 1.3 采购可得性（信息回报·不采购·硬冻结）

商用 rv64gc-no-V（真无 V）单板**存在且可购**（信息层，供用户裁 PR-1；本 agent **不下单**）：典型无 V 类 = SiFive HiFive Unmatched（U740）/ StarFive VisionFive 1（U74）/ Kendryte K210 等 U74/U54 世代 rv64gc 无 V 核。交期/价格随渠道波动，**属板支出=硬冻结**，不在本案头权限内执行。**采购决策 → PR-1 PENDING（见 §四）。**

---

## 二. T-X 表定义（六列 · DEFINITION-ONLY · 落 spec）

**T-X = X-SCALAR 独立家族真板 enablement 证据表**（与 T-N 噪声地板并列的"证据表定义"，非活测量）。落 `.trellis/spec/testing/mlir-testing-contract.md` 新 §。六列钉死如下（spec 中为权威声明，本文件为设计稿）：

| # | 列名 | 定义（钉死口径） | 格型 |
|---|---|---|---|
| 1 | **目标身份** | 真硅 + capability-facts，须按 `capability-model/profiles.md` 命名 profile；X-SCALAR 目标须为**向量缺席实例**：或 (a) 物理 no-V 真硅，或 (b) 带 V 板**强制 `-march=rv64gc` no-V 构建+运行**的**窄豁免**（后者**必须显式标 `narrow-exempt: V-board-run-as-noV`**）。指纹全分量：SoC/CPU/kernel/libc/编译器+旗标/march（须无 `v`/`zve*`）。 | 测量格·环境指纹 |
| 2 | **构建** | 精确工具链 + `-march=rv64gc -mabi=lp64d`（march **无向量扩展**）+ `-ffp-contract=off`；编译器身份+旗标入案；内核须经 **Weft-RV emitter 产出的 owned `weft_scalar` 内核**（非 fallback 兜底桩·见 `extension-plugins/scalar-fallback-plugin.md` §边界）。 | 结构证明格 |
| 3 | **零向量指令机检** | 对产出 `.o`/`.elf` **静态 objdump**：须**零 RVV opcode**（无 `v*`/`vset*`/`vle*`/`vse*` 等）+ 无 `__riscv_` 向量 intrinsic + 无 `weft_rvv` 符号 + 无 XOR-popcount 码本。脚本化（`check_f6…py` emitted-C 独立分类器的姊妹）。**板无关（静态）** = PASS/FAIL。 | 结构证明格 |
| 4 | **byte-correct** | 在目标上运行，**byte-exact** vs 独立 host oracle（ZERO-MODEL 从实际输入重算·证书三要件：语料完备/输入路径同源/oracle 独立）；整数路径 = byte-exact 门 [K-5]。 | 测量格 |
| 5 | **逐竞品产出数** | 按**对手类词表**（factory-dispatched / algorithm-matched / naive-RVV / scalar-oracle）逐类记产出数 + **对手探针工件指针**。**enablement 域 = 无 beat 主张**：scalar 永不作贡献基线（[L-6]），向量缺席目标无 factory 向量路径 → 本表数**仅 diagnostic/sanity·显式非 Win**。 | 测量格·对手指针 |
| 6 | **参考路径披露** | 披露**实际走的路径**（deployed variant = proven variant·证书三要件）；披露本表 = **enablement（无性能主张）** + 若 (b) 须标**窄豁免**；披露 [F-6] 关系（真硅证词**加强**而非**替换** committed 合成实例）。 | 结构证明格·快照 |

**状态枚举沿用 §格 schema 二分** `{measured|stale|board-pending|open|n_a}`。**铁律**：列 3（静态机检）板无关随时可产；列 4/5 无真板运行 = `board-pending`（不得进正文性能主张，[L-5]）；列 5 永不升 Win（enablement 域·词表 scalar-oracle 只作 sanity）。

---

## 三. 窄豁免五件验收计划（DESIGN-ONLY · 手头 (b) 类合规目标）

**窄豁免成立性**：手头虽无物理 no-V 真硅，但 rvv/k1 可 `-march=rv64gc` no-V 构建+运行（§1.2）。对 **enablement 性质**（无性能主张·非 k1/rvv 的性能基线用途·纯正确性+零向量机检）**窄豁免可辩护**：二进制真零向量指令（objdump 证）+ 板以标量执行 + byte-correct。**下为 design·不施工**（施工须用户裁 PR-1 或授权）。

| 件 | 产出 | 填哪列 | 板占用 | 可逆性 |
|---|---|---|---|---|
| **①** | owned `weft_scalar` 内核经 emitter 产出 → `-march=rv64gc` no-V 构建 + **静态 objdump 零向量机检工件** | 列 1(b)/2/3 | **零**（本地构建+objdump） | 全可逆（不改代码路径·只构建） |
| **②** | 目标上 byte-exact 运行 vs ZERO-MODEL host oracle（整数 byte-exact [K-5]） | 列 4 | **极小**（一次 scalar 运行·非性能·可与 Line A 错峰·须 config 会话 A/B 纪律） | 可逆（只读运行） |
| **③** | 逐竞品产出数 + 对手探针工件（scalar-oracle sanity·**显式非 Win**·enablement 域） | 列 5 | 同 ② | 可逆 |
| **④** | [F-6] 真硅证词回接：attribution record `reason=only_feasible` 在向量缺席 capability 实例上真实选中 + 声明 committed 合成实例仍权威 | 列 6 + FALSIFIER-INDEX [F-6] 注 | 零（机检） | 可逆（加注·不改门逻辑） |
| **⑤** | 参考路径披露 + C2 ledger 挂账（clone/owned 标注·[NG-8] 诚实判读）+ 窄豁免标签 + PR-1 采购 PENDING 引用 | 列 6 + `docs/method/C2_marginal_cost_ledger.md` | 零 | 可逆（文档） |

**验收门**：五件齐 ∧ 列 3 PASS ∧ 列 4 byte-exact ∧ 列 5 全标非 Win ∧ 列 6 披露窄豁免+F-6 关系。**任一涉真板运行（②③）在无授权前维持 `board-pending`**，不进正文。

**升级触发**：若用户裁 PR-1=①采购物理 no-V 真硅 → 五件在真硅重跑，列 1 去窄豁免标签，成 (a) 类完整证词。

---

## 四. PR-1 更新（采购=硬冻结→PENDING）

盘点结论回填 PR-1（`docs/PENDING_RULINGS.md`）：**手头无物理 no-V 真硅**（rvv/k1 均带 V）；**窄豁免 (b) 路径可辩护且五件计划已 design**；**商用 no-V 板可购但属板支出硬冻结**——采购**不执行**，维持 OPEN 待用户裁。保守默认 ③ 不变（仅 spec 落 T-X + 盘板源回报）。

> **〔W1 载体批 · 2026-07-19 事实更正注 · 本行历史 premise 已过期·原字节保留供考古〕**：本行 premise「**手头无物理 no-V 真硅**」= **FALSE**。`scalar`（超锐）板 = **(a) 类物理 no-V 真硅**（isa `rv64imafdch…`无 v/zve · DT `rv64imafdcbh` · AT_HWCAP V_bit21=0 · V-opcode SIGILL 硬件陷阱 · clang-18 · Fedora42）——四路机检 + byte-exact run-id 钉死于 `experiments/active/r5.1-w1-carrier/`（run-id `carrier-w1-scalar-20260719T092244Z`）。**PR-1 = REOPENED-BY-FACT-CORRECTION**（见 `…/r5.1-w1-carrier/PR-1-reopen-by-fact-correction.md`）。★**板=到货非采购·采购否决那条硬冻结不被本更正推翻**；canon 词表/铁线4 措辞订正=自动解锁进待裁·下轮裁。

**灰区标记（供用户裁）**：窄豁免 (b)「带 V 板 run-as-noV」是否算"手头合规板"，是解释性判断。本 agent 选**可逆路径**：出 design（不施工）+ 显式 `narrow-exempt` 标签 + 升级触发挂 PR-1。用户回归可批量追认或改判为①采购 (a) 类真硅。
