# A6 — falsifier real-binary runner 通电 + 六门真跑 (PR-8)

线A · A6：falsifier runner 通电 + 反向测试 + self-hosted runner 架设。
0 造数 · 真跑真留证 · git 树干净（断开步骤测毕已还原）。

## 1. 真二进制门真跑 (F-4 / F-5 / F-6, `--require-binaries`, 无 SKIP-绿)

- 门脚本：`tools/lint/run_falsifier_binaries.sh`（对 F-4/F-5/F-6 强制 `--require-binaries`；
  空/坏 `WEFT_BUILD` 直接 exit 2；任一门非零 => 整门 RED）。
- 真二进制：`build-weft/bin/weft-opt` + `build-weft/bin/weft-translate`（真 weft-opt，非 SKIP）。
- 六门口径澄清：F-1/F-2'/F-3 是纯源码/schema 门（无编译器二进制，住 `falsifier-gate.yml` self-test），
  **真二进制门 = F-4/F-5/F-6**（本脚本三门），这是 `--require-binaries` 消除 SKIP-绿的实体门。

| 门 | 检查 | exit | 判 |
|----|------|------|----|
| F-4 attribution-JSONL | 5/5 kernel 归因，shape+reason enum+[D-4] M1 only_feasible/static_order 覆盖 | 0 | GREEN |
| F-5 fail-closed fuzz | 20/20 fail-closed，corpus 钉基线不退化 | 0 | GREEN |
| F-6 scalar-family independence | vector-absent 实例真选 scalar family(only_feasible)，closure∩rvv.*=∅，emit 纯 scalar | 0 | GREEN |
| **整门** | — | **0** | **GREEN** |

证据：`green-baseline.log`（exit 0）、`green-after-restore.log`（exit 0）。

### 板端说明（host-agnostic 澄清 · 硬冻结登记）
falsifier 门是**编译期治理门**（确定性 MLIR 编译 + JSONL/attribution 产物解析，**不触板硬件**）——
x86 dev host 与 riscv64 板跑结果同信号。`build-weft/bin/weft-opt` = **x86-64** 真二进制；
两块板（`rvv`=riscv64 / `k1`=riscv64）**无 weft 构建**，board-native weft-opt 需在板上构建项目-pin 的
LLVM/MLIR = **硬冻结（工具链/基建，已于 `falsifier-binary-gate.yml` 注释登记）**，未擅自装 LLVM。
故真二进制门以**真 weft-opt 二进制**在 dev host 达成（PR-8 决议③"板端手动跑"接受的 manual lane）。

## 2. 反向测试（PR-8 核心：断开 => 门必红 => 恢复）

| 反向测试 | 断开动作 | 期望 | 实测 exit | 结果 |
|----------|----------|------|-----------|------|
| R-a 坏构建 | `WEFT_BUILD` 指向无 `bin/weft-opt` 的空目录 | 门 RED（**非 SKIP-绿**） | **2** | RED ✓ 契约成立 |
| R-b 内容回归 | F-4 corpus 删 static_order kernel（weft-opt 真发射回归 JSONL） | 分类器 RED（[D-4] M1 覆盖缺 static_order） | **1** | RED ✓ falsifier 真抓 |

- R-a 证据：`reverse-Ra-broken-build.log`（临时空目录，零树内改动）。
- R-b 证据：`reverse-Rb-corpus-regression.log`；断后已 `git checkout` 还原，
  与原文件 **byte-identical**（RESTORE_VERIFIED），随后 `green-after-restore.log` exit 0 复绿。
- 断前绿(exit0) → 断后红(R-a exit2 / R-b exit1) → 恢复复绿(exit0) 三点对照成立。

## 3. self-hosted runner 架设

`falsifier-binary-gate.yml` 消费 label `[self-hosted, weft-build]` 的 runner。板端探测：
- 无 `~/actions-runner`、无 `gh` CLI（本地亦无 `gh`）。
- **Gap-1 注册 token**：注册 self-hosted runner 需 repo-admin registration token（仓库 Settings→Actions→Runners）。
  无采购、无仓库设置权限 => **runner 注册 token gap**（如实登记，未假装通电）。
- **Gap-2 riscv64 runner 二进制**：GitHub Actions runner **无官方 riscv64 发布**；板=riscv64 无法跑官方 x64 runner。
- **Gap-3 板上 WEFT_BUILD**：即便 runner 就位，workflow 需 runner-local 已构建 weft 树；板上构建 = §1 同一硬冻结。

结论：**板端 manual lane 真跑已达（真二进制·反向红·复绿）**；**GitHub-connected runner 待 token(+riscv64 runner+板构建) 三 gap**，如实登记，禁假装通电。

## 触碰集
- 本 evidence 目录（新增，纯数据）。
- `docs/PENDING_RULINGS.md`（PR-8 状态行）。
- 反向测试临时改动 `test/Transforms/VariantSelection/attribution-jsonl.mlir` 已 `git checkout` 还原（byte-exact）。
- **不改** canon / 选择器 / schema。**不 git commit**（主会话 §四.3 复验后提交）。
