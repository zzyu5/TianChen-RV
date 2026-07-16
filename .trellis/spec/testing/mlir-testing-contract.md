# MLIR Testing Contract

测试验证真实编译器行为：dialect syntax、verification、pass 行为、plugin interface、route-provider materialization、公共 EmitC lowering、target artifact 打包，以及被主张时的 runtime 证据。

测试**不是** dashboard、readiness 状态、artifact ledger，也不是替代架构 authority（见 [core-invariants](../architecture/core-invariants.md) I4）。

## 测试类型与用途

**lit / FileCheck** —— dialect syntax/parsing；verifier 成功/失败；pass 重写行为；fail-closed 诊断；selected-body realization 在 IR 中可见；route-provider handoff 经发出的 IR/mirror 可见；公共 EmitC materialization。

**C++ tests** —— plugin registry API；route provider API；selected-body realization 结果对象；capability helper API；非文本编译器工具。

**Runtime / Hardware Evidence** —— runtime/correctness/performance 主张需要**真实执行证据**。RVV 即真 `ssh rvv` 输出（I8）。本地 compile-only、静态 MLIR check、Python smoke 都**不是** RVV runtime 证据。

## Hardware Evidence —— 一个通用契约

历史上为每个算子族抄了 ~15 份 "Generated-Bundle Evidence" 契约（per-op：reduction / MAcc / widening / clamp-select / mask-tail / packed-i4 …）。它们是**同一条证据契约**。写一次：

当某条 route 主张"真的在硬件上跑/正确/快"时，证据必须包含：

- 被测 artifact 身份（生成它的 selected variant / route）；
- 相同命名的 `ssh rvv` target/profile；
- **先正确性、后计时**：与可信 oracle（scalar 参考或外部基线）对比的正确性结果；
- 原始 stdout 或 evidence JSON，且其中**真的含**正确性与（若主张性能）计时字段。

测试断言要分清来源：`HARNESS` 前缀读的是生成的 C harness 源（断言函数调用、表达式、prototype、`printf` 格式串等源级事实）；`pattern=1 ok ...` 这类**运行期**行只能对真远端输出或确实含 stdout 的 evidence JSON 断言。

## Performance-Comparison 证据（更高门槛）

性能主张比"artifact ABI 能跑"要求更多。对比 scalar 或外部基线（如 llama.cpp）时，证据必须含：基线实现身份与版本/路径；生成 artifact 身份；两侧同一命名 `ssh rvv` target/profile；compile flags、输入尺寸、数据初始化、warmup/重复策略、计时方法；计时前的正确性检查；同时含正确性与计时的原始输出或 JSON。

> artifact ABI closeout、generated-bundle dry-run、本地 compile-only、单个 success marker —— **都不是**性能证据，不得描述成 llama.cpp parity 或"成熟"。这正是 [trunk-discipline](../guides/trunk-discipline.md) 点名的"无限证据 closeout"反模式。

## 格 schema 二分（证据格的稳定契约）

一切进 CI 的证据都落成两型**格**之一；本节是这两型格与其状态枚举的**权威声明**，别处（validation 层、experiments/ 模板、docs/实验总纲）按引用对齐，不重定义。

- **测量格** = `{值, 单位, 状态, 环境指纹, 快照, 工件指针, 对手指针, 配对会话}`。
- **结构证明格** = `{判定 ∈ {pass, fail}, CI 日志, 快照}`。

**状态枚举（闭合）** = `{measured | stale | board-pending | open | n_a}`：

- `measured` —— 真硬件 `ssh` 证据在案（I8），且指纹全分量与当前一致。
- `stale` —— 曾 measured，但环境指纹任一分量（板/SoC、CPU、内核、libc、双方编译器与旗标、ggml 版本与构建开关、线程/亲和、governor、内存）已变，或发生 refactor 后未重测。
- `board-pending` —— 待硬件重测（board-pending 期间旧性能格不得进任何文本，[L-5]）。
- `open` —— 未测量（原 `presumed`；含"presumed parity / presumed null"）。
- `n_a` —— 按构造不适用。

**三条铁律（机检，不靠自觉）：**

1. **`open` 永不可被正文引用。** 未测量默认 = 无主张，绝不默认成功态；把 `open` 当结果登记是复发性过度乐观失败模式。
2. **指纹变即 `stale`。** 指纹任一分量变 → 同指纹的所有格自动降 `stale`；parity/win 是"每次构建重测的目标"，不是可长期结转的银行数字。
3. **跨会话不比。** 只有同一配对会话（我方/对手同会话 A/B 交替）内的两值可作差；跨会话的两个 `measured` 不得相减成主张。

## T-N 噪声地板（效应判定地基）

任何"效应"（Δ、加速、回归）主张的资格前置是该板×该基准类的 **T-N 噪声地板**（重测-重测的运行间 IQR%）。判据：**`|Δ| > 2×噪声地板` 且 bootstrap 95%CI 不含 0**；**无 T-N = 无判定资格**。T-N 表结构与产出口径见 docs/实验总纲 §1.3 + `experiments/T-N_noise_floor.csv`；spec 只钉门槛。

## T-X X-SCALAR 真板 enablement 证据表（独立家族 [F-6] 真硅证词 · DEFINITION-ONLY）

**T-X** 定义 [X-SCALAR] 向量缺席独立家族在真硅上的 **enablement 证据表**（正确性 / 零向量机检 / 独立见证；**enablement 域·无性能主张**）。它与 T-N 并列为"证据表定义"，本节为**权威定义**，别处按引用对齐、不重定义。**六列钉死**：

1. **目标身份** —— 真硅 + capability-facts，按 [../capability-model/profiles.md](../capability-model/profiles.md) 命名 profile。X-SCALAR 目标须为**向量缺席实例**：(a) 物理 no-V 真硅，或 (b) 带 V 板**强制 `-march=rv64gc` no-V 构建+运行**的**窄豁免**（(b) **必须显式标 `narrow-exempt: V-board-run-as-noV`**）。指纹全分量含 march（须无 `v`/`zve*`）。
2. **构建** —— 精确工具链 + `-march=rv64gc -mabi=lp64d`（march 无向量扩展）+ `-ffp-contract=off`；双方编译器身份+旗标入案；内核须为 **Weft-RV emitter 产出的 owned `weft_scalar` 内核**，非 fallback 兜底桩（边界见 [../extension-plugins/scalar-fallback-plugin.md](../extension-plugins/scalar-fallback-plugin.md)）。
3. **零向量指令机检** —— 对产出 `.o`/`.elf` **静态 objdump**：零 RVV opcode（无 `v*`/`vset*`/`vle*`/`vse*` 等）+ 无 `__riscv_` 向量 intrinsic + 无 `weft_rvv` 符号 + 无 XOR-popcount 码本。脚本化（`check_f6_scalar_family_independence.py` emitted-C 独立分类器姊妹）。**板无关（静态）** = PASS/FAIL 结构证明格。
4. **byte-correct** —— 目标上运行，**byte-exact** vs 独立 host oracle（ZERO-MODEL 从实际输入重算·证书三要件：语料完备/输入路径同源/oracle 独立）；整数路径 = byte-exact 门 [K-5]。
5. **逐竞品产出数** —— 按对手类词表（factory-dispatched / algorithm-matched / naive-RVV / scalar-oracle）逐类记数 + 对手探针工件指针。**enablement 域 = 无 beat 主张**：scalar 永不作贡献基线（[L-6]），向量缺席目标无 factory 向量路径 → 本表数**仅 diagnostic/sanity·显式非 Win**。
6. **参考路径披露** —— 披露实际走的路径（deployed variant = proven variant·证书三要件）；披露 enablement（无性能主张）+ 若 (b) 标窄豁免；披露 [F-6] 关系（真硅证词**加强**而非**替换** committed 合成实例）。

**状态枚举沿用上节 §格 schema 二分** `{measured|stale|board-pending|open|n_a}`。铁律：列 3 静态机检板无关随时可产；列 4/5 无真板运行 = `board-pending`（不进正文，[L-5]）；列 5 永不升 Win（enablement 域）。T-X 是**定义**，不是活测量入口；活证据落 experiments/ 并按本 § 引用对齐。

> **活证据指针（引用对齐·非定义变更）**：A3 rv64gc 双通道彩排（narrow-exempt `V-board-run-as-noV`）六列填表见 `experiments/active/g8-stage3-attack/A3-xscalar-rv64gc/TX-six-column-evidence.md`（+ `evidence/`）。摘要：列 3 zero-vector 机检 PASS（owned kernel `.o` 0 向量 opcode）；列 4 byte-exact vs 两独立 oracle（`0x45511772`）；列 5 竞品产出数 heteroMx/xDSL-RVV/10x-IREE = 0（enablement·NON-Win）。彩排级·narrow-exempt·无性能主张。

## 对手解析探针（vs-framework 证据的有效性门）

对比框架（ggml/llama.cpp）的性能证据，必须在钉死的框架版本 + 该板默认构建下，探测每个 `(算子, 格式, 形状类)` **实际派发的 kernel**，产**对手探针工件**并写入测量格的对手指针。**vs-framework 测量格无探针工件 = CI 判 INVALID**（routing 随板/格式/版本变，手填必腐烂）。

**对手类（闭合枚举，本节权威声明；validation 层的三 Win 基线纪律按此引用，不重列）：**

- **factory-dispatched** —— 框架在该 `(板, 格式, 形状类)` 上**实际派发**的出厂 kernel；**唯一贡献/beat 基线**，只有它计 beat。
- **algorithm-matched** —— 同算法/同布局对照，仅作诊断（"布局/算法是否有帮助"）。
- **naive-RVV** —— 朴素向量对照，仅 sanity。
- **scalar-oracle** —— 标量对照，仅 sanity；**scalar 永不作贡献基线**（vector-vs-scalar 只测"我们向量化了"，MLIR/autovectorization 已提供，[L-6] vs-naive≠vs-framework）。

三层基线纪律与 **Win 登记阶梯（[L-7] 定案：Win-A / Win-S = sanity（vs naive / scalar+naive）；Win-B = 贡献轴 vs factory-dispatched，**B1/B2 下标由对手探针定 = 该板出厂路径**，出厂走 repack 的板上打赢其 block-dot ≠ Win-B 而是 algorithm-matched 诊断、不入登记簿；Win-C = 相级 e2e 过 [PERF-1] 八门）** 的完整措辞见 [../validation/experiment-reference.md](../validation/experiment-reference.md)，本节只提供其引用的对手类词表（factory-dispatched = 唯一 beat 基线 / algorithm-matched / naive-RVV / scalar-oracle）。

## [F-1] 零分支 falsifier —— manifest + 判读规程（C1/N2 头牌证据门）

零核心分支不变量（I3）由 CI 门机检。可执行契约:

- **family-regex manifest** —— 版本化表文件,每家族登记其名字/助记符正则(如 `rvv`、`ime`/`vmadot`、`scalar.zbb`)。falsifier 对**核心目录**(`lib/Dialect`、`lib/Support`、`lib/Transforms` 等 plugin 外的核心)grep 这些正则;白名单只含**表/数据文件**(如 monolithic op table、schedule descriptor registry)。manifest / 白名单变更走 **RFC 标记的 commit**。
- **判读规程(真分支 vs 同名假阳性,命中即需裁决):** 一处命中判 **RED(真分支违规)** 当且仅当它是**核心里按家族名做的控制流/派发决策**(if/switch/StringSwitch/starts_with 键控 family)。以下为**声明式假阳性类**(manifest 逐条 allow,不判 RED):
  1. **数据/表文件里的字符串字面量**(family 名作表数据,非分支);
  2. **身份比较**(如 `candidate.origin == selectedRoute.originPlugin`,按 origin 相等而非按 family 名分支);
  3. **op 属性名/类型助记符里含 family 子串**(如 `target_kind` / `region_kind`(E2a 引入)、类型助记符 `weft_rvv.*` / `!rvv...`)—— 是命名不是决策。
- **allow-list 精确到 (文件 glob, 正则)**;新假阳性须**显式登记**方可豁免,不得放宽全局正则。命中不在 allow-list = RED。
- 详细每家族正则表 + 白名单文件清单是 **E3 pillar 交付物**(本节钉判读规程与 manifest 形态,具体条目住 manifest 工件 + docs/科研目标总纲 [F-1])。

## 正确性门：byte-exact 先于计时 + 浮点 ULP 上界（[K-5]）

计时永远在正确性之后（见上"先正确性、后计时"）。正确性门的形态（[K-5]）：

- **整数路径 byte-exact**：与结构无关 oracle 逐字节相等。
- **浮点路径 ULP 上界**：不追字节精确，改**声明 ULP 上界**并断言不越界（fp16 次正规/极端指数等极端输入进性质测试）。
- **oracle 结构无关（实验宪法 §1.8）**：标量 oracle **不与**任何向量参考共享 bit→lane 解码；配性质测试（全零/交替符号/满幅/fp16 次正规）+ 变异测试；oracle 入库有版本。
- VLEN 翻转 lit 矩阵 ≥ `{128,256}×{m1,m2,m4}`，每板 objdump golden（指令形态 + LMUL 断言）—— 结构证明格，进 CI。

## [PERF-1] 击败八门（beat 的唯一放行通道）

任何 beat/outperform 主张在 [PERF-1] 八门全绿前**不存在**（[NG-4]/[L-1]）；八门权威定义在 docs 科研总纲 [PERF-1]，此处只映射**测试证据可承担的子集**，不重抄：byte-exact（上）；VLEN 翻转 lit `{128,256}×{m1,m2,m4}`；双板各一次 objdump 验封；micro **且** e2e（llama-bench，prefill/decode 分相）；双板都验证；机制合成归因（selector 日志证明获胜变体由能力键选出）。其余门（实验纪律、措辞门）住实验/写作层。beat 措辞前须过全部八门。

## 给 agent 的判断点（不是 gate）

- 写"第 N 个 generated-bundle evidence 测试"前：它验证的是一条**新改动的** route 吗？还是在已验证过的 family 上重复刷证据（枝节）？
- 主张性能前：有没有"先正确、后计时、且赢 baseline"的真 `ssh rvv` JSON？没有就别说快。
- 测一条 MLIR 行为时，优先 lit/FileCheck；只有文本测不了的语义才上 C++。
