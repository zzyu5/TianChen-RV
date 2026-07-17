# PRD · bench 每格对拍 harness 建设（cells/ · K线板测公共前置）

> **权威** = 《测试与收尾总令-开测篇》§二.1（bench 唯一通道·五步流水）+ §〇.2（harness 契约）+ ISSUE-096。
> **性质**：**建测量基建**（把 bench 从「能干跑」变「能真测」），**非测量本身** —— 不产入账头条数。

## 一、目标

bench 真跑（`execute=True`）时 step3/4/5 现在 `raise CellRecipeMissing`（每格对拍 harness 未建）。
建 `tools/bench/cells/` 的对拍 harness + 接进 bench 五步，使 `bench <gemm格> --board rvv` **真跑一格**产真 cold + 入行。
**gemm_tile 族先行**（P2/PR-47/dequant/S1 后续复用）。

## 二、可复用资产（已验证跑通 8 格）

`experiments/active/g8-stage3-attack/P2-grid4-raw/run_grid4_p2.sh` —— 完整五步板端逻辑：
- **verify** = build + probe + ABI-GATE + T1 board byte-exact + T2 对拍三验（ours/OPP vs oracle mism=0/8192）+ 4-arm anti-hollow（fault-leaf differing_bytes=1）
- **sanity** = 噪声自检 3 轮
- **measure** = 2-seed cold（K=2048 nr=16 nc=512 N=25）
- **单世界 clang-18 双板配方**：rvv → `/opt/tcrv-toolchains/llvm-18.1.8/bin/clang` + env.sh binutils；k1 → `clang-18`（Bianbu）；scalar → `/usr/bin/clang-18`（Fedora）
- **静板/世系**：`cpu_md5_before` · `PRE_STRAY` · `BUILD … CC=clang version 18.1.8`

## 三、bench 侧接口（约定已在·只差实现）

| bench 函数 | 现状 | 要做 |
|---|---|---|
| `step3_crosscheck(fmt,board,cfg,execute)` | execute 分支 `raise CellRecipeMissing` | 定位 `cells/` harness → ssh 调 verify → 解析 ABI-GATE/T2 mism/向量指令数/对手档/对手身份 → 返回（任一失败落对应 VOID） |
| `step4_cold(fmt,board,cfg,execute)` | 同 raise | 调 harness measure → 解析 `ratio_cold_X` → 返回 cold |
| `step5_row(run_id,execute)` | raise（注释 gated on ISSUE-073） | **073 已 RESOLVED** → 写主表行经 `guarded_open(DEST_MASTER)` + `append_runs_log`；更新那条过时 gate 注释 |
| `guarded_open` / `append_runs_log` / `DEST_MASTER`/`DEST_RUNSLOG` | 已就位 | 直接用（三目的地落法已在） |

## 四、★契约（§〇.2·harness 禁自写文件）

- harness 住 `tools/bench/cells/`，由 `bench` **按声明接口调用**。
- **harness 自身禁写任何持久文件** —— 板端跑完把结果**打到 stdout**，bench **解析 stdout**，一切持久写入**经 runner 的写入闸落三目的地**。
- （run_grid4 原版 harness 会往板上 `$RDIR` 写 seal/log —— 那是**板端临时**，可接受；关键是**仓库侧持久写入全经 runner**，harness 不在 `tools/` 下落任何仓库文件。）

## 五、验收标准

1. **真跑一格**：`bench gemm_tile <fmt> --board rvv --engine rvv --regime prefill`（非 dry-run）→ 走完五步 → 产**真 cold ratio**（不是 DRY-RUN 占位）。
2. **三目的地写入**：`runs.log` +1 真行（含 run-id·cold·判定·对手身份+档·世系）· `runs/<run-id>/` 有原档（计时原档/反汇编/探针）· 主表行更新。
3. **对拍三验真跑**：ABI-GATE PASS · T2 ours/OPP vs oracle mism=0 · 对手 = 部署派发（objdump dispatch）—— 任一失败落对应 VOID（我方错/对手废/派发不符），**不计时**。
4. **无计划外文件**：`find experiments/ -newer` = 仅 runs.log + runs/<run-id>/（写入闸 destination guard 生效）。
5. **self-test 仍绿**：`bench --self-test`（干跑路径不回归）。
6. **★与既有 seal 交叉验证**：拿一个 P2 已测格（如 iq3_xxs@rvv）真跑，cold ratio 应与前役 seal 的 `ratio_cold_X`（0.95）**同域**（板噪声内）—— 证明新通道产的数与旧配方一致（不是新造）。
7. **触碰集**：`tools/bench/`（cells/ 新建 + bench step3/4/5 实现）。不碰 spec 算法/主表数据/recon。

## 六、遗留 / 交接

- 先 gemm_tile 族；dequant/vec_dot/forward 族 harness 后续 task（同框架）。
- **交接 trellis-check**：核 harness 契约（禁自写仓库文件·写入全经 runner）· 真跑产的 cold 与既有 seal 同域（新旧通道一致性）· destination guard 负控（写闸外路径中止）· VOID 三出口真能落。
- 建成后 ISSUE-096 → RESOLVED；PR-47/S1/dequant 板测全走真 bench。
