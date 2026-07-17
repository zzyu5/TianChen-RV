# 裁决九 — G1 卫生自查证据表(收口宣布前置件)

> 日期 2026-07-07 · 分支 `refactor/full-refactor-m1` · HEAD `d719eccb`(vec_dot 声称全格 constructed,C_construct 33)。
>
> 本表是 **G1 收口宣布的前置件**:六项卫生自查各给 {问题 / 治愈判据(引原裁决)/ 活证据指针 / 现场复验结果 / HEALED|UNHEALED}。
> 全程只读复验(demo 注入均已 `git checkout` / `rm` 清回基线,`git status` 无痕)。
> 每项 UNHEALED 均已建具名 trellis 修复任务(见文末队列表),留待并行线协调后串行执行。
>
> 判定语义:**HEALED** = 治愈判据的机制**存在 + 活哨兵钉死 + 现场复验通过**;
> **UNHEALED** = 拦截逻辑可能已有效,但判据存在**硬缺口**(无判定命令 / 未接 CI / 活树已 RED / 有未命名尾巴)使其**不自执行**或**残留≠0**。

## 证据表(6 行)

| # | 问题 | 治愈判据(引原裁决) | 活证据指针 | 现场复验结果 | 判定 |
|---|---|---|---|---|---|
| 九.1 | quant 格式是否回退成路由分派键(G1 复发排查) | 抽象 `tcrv_rvv.quant_contraction` 删掉 `quant` attr 仍正确路由;算法/lowering 由结构化能力事实(三事实 AND)驱动,不读格式名;新格/前向算子靠 op 身份进入 | 删除式法官 lit `test/Conversion/RVV/rvv-lower-quant-contraction-facts-drive-routing.mlir`(故意不写 quant attr,`CHECK-NOT` + 断言仍构造逐字节相同 repack 区);fail-closed 哨兵 `isWiredDequantizeRowFormat`(`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:9013`) | 法官单测 JUDGE: PASS;lit `Conversion/RVV`+`Dialect/RVV` **228/228**、`Target`+`Plugin`+`Transforms` **327/327** 绿;全 lib/ 新格(tq2_0/tq1_0/q1_0/nvfp4)+前向算子(scale/silu/rms_norm/softmax/rope)格式名 `==` 分派 grep=**0**,全走 distinct typed op | **HEALED** |
| 九.2 | 对手事实防腐(pin/provenance 静默腐烂) | (a) TableGen 对手事实带探针来源注记 (b) 对手库 pin bump→STALE 校验 CI 钩子 | STALE 脚本 `tools/lint/check_opponent_facts_pin.sh`(`--self-test`);provenance `experiments/sealed/c1-cleanliness/opponent-facts-provenance/opponent-facts.pin.json`(逐条 fact→ggml 行锚);pin 真源 `schema/ggml-pin.lock.json`;`.td` attrs `RVVOps.td:4302/4303`;唯一 workflow `.github/workflows/falsifier-gate.yml` | 脚本三重实证:LIVE 默认路径 `MATCHES` exit0 / `--self-test` GREEN / 真实 pin bump 模拟 `STALE(RED)` exit1 **拦截成功**、还原后复归 clean。但 (b) **未接 CI**(全仓 grep `check_opponent_facts_pin`=0,只是人得手跑的 lint);(a) 无显式探针来源注记(绑定单向) | **UNHEALED** |
| 九.3 | ABI 推导纯度(M 行数 nb / 输出步长是否吃格式键) | 宽度/步长推导只吃操作数类型+几何属性,不吃量化格式键;变异测试锁定"改操作数类型→推导值跟随" | 纯类型 mangler `riscvIntrinsicName`(`lib/Conversion/RVV/RVVToEmitCSupport.cpp:41-47` 自证);`nb=n/QK`(`RVVToEmitCBlockQuantLinear.cpp:148`)+ output-store(`:318-338`)均无 `kind`;变异测试 `test/Conversion/RVV/rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir`(sed 改 SEW/LMUL → `M1:` 前缀逐 token 断言推导跟随) | 变异测试 lit **6/6 PASS**(二进制 14:14 晚于源 14:13,非 STALE);mangler 文件量化格式键命中仅 3 处全在注释、零代码分支;nb 与 output-store 无 `kind` → 推导路径无格式键成分(GREEN) | **HEALED** |
| 九.4 | 退役口径("vec_dot 残留=0"是否机检、尾巴是否有名有批次) | 有文档化"残留=0"判定命令 + 公开白名单(枚举允许存活的旧 monolith + 原因 + 批次);尾巴允许但必须有名有批次 | 判定命令 **无 canonical**(本次自建 R1/R2/R3);白名单文件 **无**(散点注释:mxfp4 keep 在 `RVVToEmitC.cpp:439-440`、q1_0 批次在 `.trellis/tasks/07-07-retire-q1_0-monolith/`、nvfp4 `#if 0` 在 `RVVDialectWideningOps.cpp:3572`);最近似机检物 `coverage-sixstate.v1.json`+`e5_strong_readout.py` 只 gate 翻转不 gate 删除 | 自建残留命令跑出 **残留≠0**:8 全-monolith vec_dot op-def 仍在 ODS + 9 verify() 在 dialect + 3 live monolith emitter dispatch + 1 个 nvfp4 `#if 0` 死块;有 **未命名尾巴**(nvfp4 `#if 0`、K-quant×5 op-def)违反"有名有批次" | **UNHEALED** |
| 九.5 | 三条目录 lint(experiments 白名单+sealed 只读 / canon 白名单+reports append-only / INDEX 双向一致)是否 fail-closed 且 CI 常跑 | 三 lint 各 fail-closed、覆盖三域,且在 CI 常跑 | ① `tools/lint/check_experiments_layout.py` ② `tools/lint/check_docs_canon.py` ③ `tools/lint/check_index_consistency.py`(+`gen_experiments_index.py --check`);唯一 workflow `.github/workflows/falsifier-gate.yml` | self-test 三条全 GREEN、三注入(sealed写/非法文件/canon非总纲)全拦 fail-closed。但 **CI 不常跑**(三 lint 无任何 CI/hook/Makefile 调用,只挂"(CI)"注释,grep=0);**活树已 RED**:① exit1(12 个未注册代码工件泄漏进 `gap_triage_batch2c/` 数据格)、③ exit1(同格 32 处未注册漂移) | **UNHEALED** |
| 九.6 | 杂项:finish-task 积压 / 并行基建三根因 / 板恢复欠账 | finish 无积压(current-task 是活指针非 stuck、零 done-未归档);并行基建三根因落地(HEAD 分支 worktree / 独立 build CMakeCache / pinned 基线 detached-worktree 绝不 git stash);板恢复 F23 类无欠账 | 基建脚本 `tools/bench/{_build-common.sh,byte-exact-baseline.sh,configure-line-build.sh,new-line-worktree.sh}`+协议 `docs/method/parallel-build-and-baseline.md`+`.touch-set/`;账本 `docs/reports/travel-decision-ledger.md` F23→F25 闭 | finish 零积压(current=07-07-line-C-coverage 活指针);四脚本 `bash -n` SYNTAX-OK、两入口 `--help` OK、`byte-exact-baseline.sh HEAD tcrv-opt --dry-run` 干净 exit0 解析 sha=d719eccb `main tree NOT touched`,`.worktrees/cache/` 6 个 pinned base 实用;板恢复 F25 已闭 F23 pending | **HEALED** |

## 汇总

- **HEALED 3 项**:九.1(quant 纯标签)、九.3(ABI 推导纯度)、九.6(杂项:finish/并行基建/板恢复)。
- **UNHEALED 3 项**:九.2(对手事实防腐)、九.4(退役口径)、九.5(三条目录 lint)。
- **共同失效模式**:三项 UNHEALED 的拦截逻辑本身**都已证有效**(脚本/法官/lint 都能拦),硬缺口高度同构——**守卫写好了却没接电**:关键机制未接进唯一 workflow `falsifier-gate.yml`(它只跑 schema.def F-2' 门),于是 pin bump / 残留 / 目录腐烂的 PR 会 CI 全绿通过。九.4/九.5 更进一步:因无 CI 常跑,腐烂已沉积到活树(残留≠0、`gap_triage_batch2c/` 现行 RED)。

### 退役尾巴清单 + 批次(源自九.4;★=有名有批次,❌=未命名)

| 尾巴 | 锚点 | 类别 | 批次状态 |
|---|---|---|---|
| q1_0 monolith | `GgmlBlockDotQ10Q80Op`(ODS:5436)+verify(:3739)+dispatch(:442)+3 测 | 纯 dead-wiring(cell 已 constructed,前门不再造) | ★ **有名**=`07-07-retire-q1_0-monolith`,但 status=planning **未执行**(implement/check.jsonl 仍模板占位) |
| nvfp4 `#if 0` 死块 | `GgmlBlockDotNVFP4Q80Op::verify` 包在 `#if 0`(`RVVDialectWideningOps.cpp:3572-3737`,~165 行) | 编译外死码待删 | ❌ **无名无批次** |
| K-quant 5 op-def 尾巴 | `GgmlBlockDot{Q4K,Q6K,Q5K,Q2K,Q3K}Q8KOp`(ODS:6614/5839/6741/6836/7449)+各 verify() | 死 op-def,仅经 `getOperationName()` opName 字符串被前门 gate 引用(isa/dyn_cast=0) | ❌ **无名无批次**(retire 任务只裁 q1_0) |
| mxfp4 monolith | `GgmlBlockDotMXFP4Q80Op`(ODS:5193)+verify(:3350)+real dyn_cast(:675) | **蓄意保留**(FP4-class 负控,喂 e5 判别器 NOT-strong) | ⚠ 仅注释说明(`RVVToEmitC:439-440`),**未进公开白名单** |
| q4_0 monolith | `GgmlBlockDotQ40Q80Op`(ODS:4118)+verify(:1431)+dispatch(:359) | **live 第二路径**(抽象 N-operand contraction 的 block-dot 臂,`RVVLowerQuantContraction.cpp:654` 仍 create) | ⚠ 设计内存活,但**无处记为退役例外** |

> 补:`e5_strong_readout.py` 注释断言 Q5_K/Q6_K/Q3_K 等"retired same action as the flip",复验证伪其**完整性**——opaque emitter helper 确退,但 ODS op-def + verify() + opName-string family entry **未退**(正是 K-quant×5 残留)。六态过(翻转 constructed)⊥ 残留=0(删除),二者正交。

### UNHEALED → 具名修复任务(已建,进队列)

| 源裁决 | 具名 trellis 任务 | 优先级 | 核心修复动作 |
|---|---|---|---|
| 九.2 | `.trellis/tasks/07-07-wire-opponent-facts-pin-ci/` | P1 | ①`check_opponent_facts_pin.sh`(默认路径+`--self-test`)接进 `falsifier-gate.yml`(升 lint→真 CI 钩子)②`RVVOps.td:4302/4303` 加探针来源注记回指 provenance+STALE 门 ③(次)清 `experiments/MANIFEST.md:45`+`archive/MOVES.md:252` 已证伪 path-break 警告 |
| 九.4 | `.trellis/tasks/07-07-retire-residual-gate-whitelist/` | P1 | ①固化 R1/R2/R3 机检残留命令(未白名单 op-def/emitter/`#if 0`==0)接 CI ②建 `schema/monolith-retire-whitelist.v1.json` 逐条 reason+批次(mxfp4=永久负控;q4_0=contraction 臂;q1_0=已有 task;K-quant×5+nvfp4=新批次)③补建 K-quant×5+nvfp4 缺失批次 ④执行停摆的 q1_0 批次 |
| 九.5 | `.trellis/tasks/07-07-dir-hygiene-lints-ci-clean-red/` | P1 | ①三 lint(+`gen_experiments_index.py --check`)加 CI job(`falsifier-gate.yml` 或新 `dir-hygiene.yml`,on PR+push,fail-closed,带 `--self-test`)②清红:`gap_triage_batch2c/` harness/代码 `git mv`→`tools/`、真证据注册进 owning cell MANIFEST,使 ①③ 回 GREEN |

> 三条修复任务均为 `07-02-full-refactor` 子任务、status=planning、P1(=G1 收口宣布前置件)。执行会触碰 `lib/`/`include/`/`experiments/`/`tools/`/`.github/` 共享文件,须与并行线协调后串行(动 ODS/emitter/experiments 前 re-read)。

## 收口判据

**G1 收口宣布的前置件 = 上述 3 项 UNHEALED 全部转 HEALED**(各自具名任务落地后重跑本表复验)。当前 3/6 HEALED,3/6 UNHEALED——**尚未满足收口前置件**。
