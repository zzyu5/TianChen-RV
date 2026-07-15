# B1 Falsifier CI 板端 lane — 裁决 〇.1 执行验收

> 纯案头执行（独立域·零板时·零新计时）。2026-07-15。
> 裁决 〇.1：F-4/F-5/F-6 真二进制检查由 CI 派发真板执行·任何通道缺真二进制=FAIL·**废除 SKIP-绿**·验收含反向测试（人为断开构建步骤·CI 必红）。
> 前置诊断：`B1c-falsifier-ci-audit.md`（选项 A 推荐 + 选项 C papercut）。

---

## 落地项（案头可直改部分·全部完成）

### ① 新脚本 `tools/lint/run_falsifier_binaries.sh`（板端真二进制 lane）
- `set -euo pipefail`；`WEFT_BUILD` 空/未设 → `exit 2`（setup RED·**无 build-free fallthrough**）；
  非目录 / 无 `bin/weft-opt` 可执行 → `exit 2`。
- 依次跑 F-4 / F-5 / F-6 `--require-binaries`（聚合三门·任一非零 → 整体 `exit 1`·RED）。
- **无 SKIP 分支**（对比轻量 CI 的 build-free SKIP-绿）。

### ② locate_opt papercut 修复（选项 C）
- `check_f4_attribution_jsonl.py` / `check_f5_failclosed_fingerprint.py` `locate_opt` 候选补
  `<REPO>/build-weft/bin/weft-opt`。
- `check_f6_scalar_family_independence.py` `locate_binary`（按 name）补 `<REPO>/build-weft/bin`
  → 同时覆盖 `weft-opt` 与 `weft-translate`。
- 效果：改名 Weft 后规范构建目录 `build-weft/` 即被真检自动发现（不必手设 `WEFT_BUILD`）。

### ③ 新 workflow `.github/workflows/falsifier-binary-gate.yml`
- `on: schedule`（nightly `17 3 * * *`）+ `workflow_dispatch`。
- `runs-on: [self-hosted, weft-build]`（占位·PENDING runner）。
- step：`export WEFT_BUILD=${WEFT_BUILD:-$GITHUB_WORKSPACE/build-weft}` → `bash tools/lint/run_falsifier_binaries.sh`。
- YAML 语法校验通过（`yaml.safe_load` OK）。

### ④ `docs/method/FALSIFIER-INDEX.md` 诚实化
- 新增「维度诚实（B1 裁决 〇.1）」注（[P-3] 行下）：区分 **self-test 判别门（轻量 CI·常绿）** vs
  **真二进制门（板端 lane·nightly·跑于有 build 之处·尚未绑无人值守 CI）**。
- §「进 CI 的 [F-1..F-6]」新增「F-4/F-5/F-6 两层门·诚实拆分」bullet（层① self-test = ✅ 进 CI；
  层② 真二进制 = ⚠ SKIP lane·PENDING runner）。
- **废止「六门真检已在 CI 绿」夸大表述**（明文标注：进 CI 的是 self-test + 源码门·真二进制维度未绑无人值守 CI）。

### ⑤ 反向测试（证断开构建步骤 → CI 必红）
见下「验收证据」TEST 2/3/4：WEFT_BUILD 空 / 坏路径 / 有目录无 weft-opt → 全 `exit≠0`。

---

## 验收证据（本机·2026-07-15·`build-weft/bin/{weft-opt,weft-translate}` 今日构建）

### 正向：有 build 时三门真绿（`exit 0`）
```
$ export WEFT_BUILD=$PWD/build-weft
$ bash tools/lint/run_falsifier_binaries.sh
[F-4-attribution] GREEN: 5/5 kernels attributed (shape + reason enum + [D-4] M1 coverage intact)
[F-5-failclosed]  GREEN: fail-closed coverage did not regress (20/20 fail-closed)
[F-6-independence] GREEN: vector-absent instance selects scalar (only_feasible), closure ∩ rvv.*=∅
[falsifier-binaries] GREEN: F-4 / F-5 / F-6 real-binary gates all pass.        EXIT=0
```

### 反向：断开构建 → RED（`exit≠0`·废除 SKIP-绿）
```
TEST 2  WEFT_BUILD 空/未设                       → RED · EXIT=2
TEST 3  WEFT_BUILD=/nonexistent/broken/tree      → RED · EXIT=2
TEST 4  WEFT_BUILD 有目录但 bin/weft-opt 缺席     → RED · EXIT=2
```

### papercut：无 WEFT_BUILD·靠 build-weft/bin 自动发现（`--require-binaries`）
```
$ unset WEFT_BUILD; python3 tools/lint/check_f4_attribution_jsonl.py --require-binaries   → GREEN·EXIT=0
$ unset WEFT_BUILD; python3 tools/lint/check_f6_scalar_family_independence.py --require-binaries → GREEN·EXIT=0
```

### 无回归：三门 self-test 仍绿
```
check_f4_attribution_jsonl --self-test  OK
check_f5_failclosed_fingerprint --self-test  OK
check_f6_scalar_family_independence --self-test  OK
```

---

## 硬冻结 · PENDING 登记（未擅自执行）
- **self-hosted runner（label `weft-build`）供应 = 基建/工具链** → 登记 **PENDING**（需用户/基建）。
  runner 就位前 `falsifier-binary-gate.yml` DORMANT（schedule/dispatch 触发但无匹配 runner）。
- **在轻量 CI 里装 LLVM/MLIR 从零 build（选项 B）= canon 级工具链变更** → 未采纳·未改
  `falsifier-gate.yml`（保持 stdlib-only 设计）。
- 本执行**只**落案头可改部分（脚本 + checker 候选路径 + 新独立 workflow + 文档诚实化）·零板时·
  触碰集与 Line A 板测不相交。

## 触碰集
- 新增：`tools/lint/run_falsifier_binaries.sh`、`.github/workflows/falsifier-binary-gate.yml`、本文件。
- 修改：`tools/lint/check_f4_attribution_jsonl.py`、`check_f5_failclosed_fingerprint.py`、
  `check_f6_scalar_family_independence.py`（locate 候选补 build-weft）、`docs/method/FALSIFIER-INDEX.md`（诚实化）。

## 一句话结论
裁决 〇.1 案头部分**全部落地**：真二进制 lane 脚本（无 SKIP·空/坏 build 必 RED）+ 夜跑 workflow（runner PENDING）+
papercut 修 + INDEX 诚实化；本机正/反向测试俱证（有 build 三门真绿·断开构建三态全 RED）。
唯一缺口 = self-hosted runner 供应（硬冻结·PENDING·待用户）。
