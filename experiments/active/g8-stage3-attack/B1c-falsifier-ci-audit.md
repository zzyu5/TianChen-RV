# B1.③ Falsifier CI 绑定审计 — F-4 / F-5 / F-6 真二进制检查

> 案头审计（纯案头·独立板域·零板时·零新计时）。2026-07-15。
> 指控：F-4/F-5/F-6 falsifier 是真二进制检查（反空心证书·六门），核实其是否绑进 CI。
> 目标：六门真检在 CI 绿（build step 或 `--require-binaries`）。

---

## 裁决：部分成立

**成立的部分**：F-4/F-5/F-6 三门**都已作为 job 绑进 CI**（`.github/workflows/falsifier-gate.yml`：
`f4-attribution-jsonl` / `f5-failclosed-fuzz` / `f6-independence-gate`），且每门的**hermetic self-test（分类器判别）
在 CI 中是真 fail-closed 强制门、当前全绿**。

**不成立 / 被夸大的部分**：这三门的**真二进制检查（=实际跑 `weft-opt`/`weft-translate` 打真编译产物）
在 CI 中【不执行】——它 SKIP（exit 0）**。原因两条，缺一即降级：
1. `falsifier-gate.yml` **没有任何 build step**（grep `cmake|make|ninja|weft-opt` 命中 0 个构建动作；
   workflow 显式设计为「Stdlib-Python only；sole external action = actions/checkout」）；
2. 三门的 default 命令行**都没有传 `--require-binaries`**（`falsifier-gate.yml:215 / 249 / 320`）。

结果：CI 里跑到 binary 那一步时 `weft-opt` 不存在 → 走 build-free SKIP 分支 → exit 0 →**假绿**。
真正的「反空心证书」二进制检查（对真编译器输出下判决）**从未在 CI 触发**。

`FALSIFIER-INDEX.md:23-24 / 62` 的「六门全进 CI = [P-3] falsifier 组全进 CI 达成」在 **job 层 / self-test 层准确**，
但对 F-4/F-5/F-6 的**二进制维度是夸大**：进 CI 的是 self-test，不是真二进制检查。

---

## 证据（file:line + 命令可溯源）

### 1. CI 只绑 self-test，binary 走 SKIP（无 build step、无 `--require-binaries`）
- `.github/workflows/falsifier-gate.yml:214-215`（F-6 default 门：`python3 ...check_f6_scalar_family_independence.py`，无 flag）
- `.github/workflows/falsifier-gate.yml:248-249`（F-5：`...check_f5_failclosed_fingerprint.py`，无 flag）
- `.github/workflows/falsifier-gate.yml:319-320`（F-4：`...check_f4_attribution_jsonl.py`，无 flag）
- `grep -n 'cmake|make|ninja|weft-opt build' falsifier-gate.yml` → **0 个构建动作**（只有 checkout + `python3`）
- `grep -rn 'require-binaries' --include=*.sh --include=*.yml --include=Makefile .` → **仓库内零处真调用**
  （只有 `falsifier-gate.yml` 注释 + checker 自身 help 文本；无任何 CI/Makefile/pre-commit/board 脚本真传该 flag）

### 2. checker 的 SKIP 分支（binary 缺席 → exit 0）
- `tools/lint/check_f4_attribution_jsonl.py:225-233`（`opt` 未找到且非 `--require-binaries` → 打印 SKIP、`return 0`）
- `tools/lint/check_f5_failclosed_fingerprint.py:161-169`（同构）
- `tools/lint/check_f6_scalar_family_independence.py:177-186`（同构，需 `weft-opt` + `weft-translate`）

### 3. 本地复现（证「假绿」与「flag 生效」）
```
$ python3 tools/lint/check_f4_attribution_jsonl.py            # default 无 flag
[f4-attribution] weft-opt not built ... -- SKIP (build-free lane; pass --require-binaries to gate)   exit=0
$ python3 tools/lint/check_f4_attribution_jsonl.py --require-binaries   # binary 缺席时
[f4-attribution] weft-opt not built ... -- RED (--require-binaries)                                   exit=2
```
CI 跑的是第一条（exit 0 假绿）。

### 4. self-test 部分真绿（这部分指控成立、机制健康）
```
check_f4_attribution_jsonl --self-test  exit=0  GREEN: validator discriminates all cases
check_f5_failclosed_fingerprint --self-test exit=0  GREEN: classifier discriminates all cases
check_f6_scalar_family_independence --self-test exit=0  GREEN: classifier discriminates all cases
```

### 5. ★真二进制检查逻辑健康——只差 CI 绑定（关键）
本机存在今日构建的 `build-weft/bin/weft-opt` + `weft-translate`。指向它跑真二进制门，**三门全真绿**：
```
$ export WEFT_BUILD=$PWD/build-weft
$ python3 tools/lint/check_f4_attribution_jsonl.py --require-binaries
[f4-attribution] GREEN: 5/5 kernels attributed (shape + reason enum + [D-4] M1 coverage intact)   exit=0
$ python3 tools/lint/check_f5_failclosed_fingerprint.py --require-binaries
[f5-fingerprint] GREEN: fail-closed coverage did not regress (20/20 fail-closed)                   exit=0
$ python3 tools/lint/check_f6_scalar_family_independence.py --require-binaries
[f6-independence] GREEN: vector-absent instance selects scalar (only_feasible), closure∩rvv.*=∅   exit=0
```
→ 差距是**纯 CI-binding**（check 逻辑对、工件对、真编译产物对）；不是坏门。

### 6. 附带真 papercut：checker 不探测当前规范 build 目录
`locate_opt` 只探 `$WEFT_BUILD/bin` 与 `<REPO>/build/bin`
（`check_f4_...py:163-171`；f5/f6 同构）。但改名 Weft 后本地规范构建目录是 **`build-weft/`**。
开发者即便构建到 `build-weft/` 也会**静默 SKIP**（除非手设 `WEFT_BUILD`），削弱「有 build 就跑真检」的落地性。

---

## 施工计划（供主会话落地；均非纯案头可直改，故出计划不擅动）

### 选项 A（推荐·最低成本·尊重 CI stdlib-only 设计）——复用既有构建的 binary 门
不在轻量 CI 里从零 build LLVM/MLIR/weft（那是 workflow 刻意规避的、且属工具链决定=必问）。改为把**真二进制门绑到已有构建的执行面**：
1. 新增脚本 `tools/lint/run_falsifier_binaries.sh`：`set -e`，`: "${WEFT_BUILD:?}"`，依次跑
   F-4/F-5/F-6 `--require-binaries`（任一非零即 RED）。
2. 绑定处（择一或叠加）：
   - **board/self-hosted lane 或 nightly**：新 workflow `falsifier-binary-gate.yml`，`on: schedule`（夜跑）
     或 `runs-on: [self-hosted, weft-build]`，step 先 `export WEFT_BUILD=<已构建树>`（或增量 build）再跑该脚本。
     ——绑真检到无人值守自动化（需 board/self-hosted runner 协调，非本审计员板时范围）。
   - **本地 check loop / dev-gate**：把该脚本挂进 trellis-check 的 pre-commit/pre-push（存在 `$WEFT_BUILD` 时才跑、
     否则 advisory SKIP），使真检成为提交前默认门。

### 选项 B（重·触工具链设计=必问）——CI 全构建 job
在 `falsifier-gate.yml` 增 `binary-falsifier` job：装 LLVM/MLIR（apt 或 prebuilt cache）→ configure+build
`weft-opt`+`weft-translate` → 跑三门 `--require-binaries`。代价：CI 构建分钟数大增、破坏「stdlib-only」设计原则
（决策卡：工具链/CI 常驻形态变更 = canon 级·必问用户）。仅在用户接受重 CI 时采纳。

### 选项 C（低风险·可即改·建议连同 A 落地）
1. `locate_opt` 候选列表补 `<REPO>/build-weft/bin/weft-opt`（f4/f5/f6 三 checker 同址；f6 另补
   `weft-translate`）——修 §6 papercut，让「构建到规范目录即被真检发现」。
2. `FALSIFIER-INDEX.md` 诚实化：把 F-4/F-5/F-6 的「进 CI」措辞拆成
   **「self-test 强制门 ✅ 进 CI」** vs **「真二进制检查：跑于有 build 之处·尚未绑无人值守 CI（走 SKIP lane）」**，
   避免「六门真检在 CI」被读成二进制维度已闭。

---

## 一句话结论
三门已进 CI 但**只有 self-test 是真门**；真二进制检查在 CI 里 SKIP（无 build step + 无 `--require-binaries`），
且仓库内**无任何自动化真传该 flag**——真检只在人手本地跑时才触发（本机今日跑真绿）。
→ 「六门真检在 CI 绿」**未达成**；差距是纯 CI-binding，按选项 A（+C）落地即可闭合，选项 B 触工具链需用户裁。
