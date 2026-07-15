# B1b — C2 成本台账 commit-pin 审计（案头 · 纯分析）

> 任务：核实 T2 成本台账四数 **2484 / 1501 / 907 / 757** 的 provenance commit + 口径，
> 为每笔钉 pin，使 ledger 支持 `--at-commit` 重跑印出台账数。
> 触碰集 = `.trellis/scripts/family_ledger.py`（加 `--at-commit`）+ 本审计文件（新建）。
> **禁 board / 禁计时 / 禁新账本**；只读全仓 + git 史 + 改既有 ledger 脚本。

---

## 〇、一句话结论

**四数全部指控成立且 byte-exact 可复现**，各钉到确定 pin commit；口径澄清如下表。
原两个 recompute 脚本**都不支持 `--at-commit`**（一个把 2484 硬编码为常量、只印 live；另一个只读 live 工作树），
故按任务「若脚本存在则加」给 `family_ledger.py` **新增 `--at-commit` git-blob 复算模式**，
四数在各自 pin 上重跑**逐一印出同数**（见 §四验收）。

---

## 一、四数溯源表（值 · pin · 口径 · 可复现命令）

| 数 | 含义 | pin commit | 口径 | 复现命令（在 pin 上重跑印出同数） |
|---|---|---|---|---|
| **2484** | IME 家族 founding-slice 总量（extension-family onboarding 成本 anchor） | **`54465ee7`**（2026-07-03，pin [LED-1] 首点的 commit） | raw wc-l，7 文件求和，排除 CMakeLists.txt，**historical slice 口径 · FROZEN** | `python3 .trellis/scripts/family_ledger.py report --family IME --at-commit 54465ee7` → `raw_wc_l=2484` |
| **1501** | X-SCALAR(Scalar) 家族 founding 总量（第二 onboarding 点） | **`2dd654d8`**（2026-07-07，家族收口末 commit；landed 链 `5c010b2b→f96f767a→2dd654d8`） | raw wc-l，9 文件求和，排除 CMakeLists.txt | `python3 .trellis/scripts/family_ledger.py report --family Scalar --at-commit 2dd654d8` → `raw_wc_l=1501` |
| **757** | IME **emitter 住址** raw（= `IMEBackendEmissionDriver.cpp` 730 + `.h` 27） | **`54465ee7`** | raw wc-l，`*BackendEmissionDriver.{cpp,h}` 求和；F-6「自有 emitter」成本住址 | 同 IME 行 → 输出字段 `emitter_raw_wc_l=757` |
| **907** | X-SCALAR **emitter 住址** raw（= `ScalarBackendEmissionDriver.cpp` 878 + `.h` 29） | **`2dd654d8`** | raw wc-l，`*BackendEmissionDriver.{cpp,h}` 求和 | 同 Scalar 行 → 输出字段 `emitter_raw_wc_l=907` |

**cloc-approx 次口径**（同命令一并印出）：IME `cloc_approx=1866`、Scalar `cloc_approx=1148` —— 与
`docs/reports/2026-07-12-C2-LED-2-边际曲线-三家族点.md` §二/§三表一致。

---

## 二、逐数核实证据（file:line / 命令）

### 2484 — 成立（byte-exact @54465ee7）
```
git ls-tree -r --name-only 54465ee7 -- lib/Dialect/IME lib/Plugin/IME \
  include/TianChenRV/Dialect/IME include/TianChenRV/Plugin/IME | grep -v CMakeLists.txt \
  | while read f; do git show "54465ee7:$f" | wc -l; done | awk '{s+=$1}END{print s}'
# => 2484
```
逐文件：858 IMEExtensionPlugin.cpp · 730 IMEBackendEmissionDriver.cpp · 438 IMEDialect.cpp ·
350 IMEOps.td · 65 IMEExtensionPlugin.h · 27 IMEBackendEmissionDriver.h · 16 IMEDialect.h = **2484**。
- 台账登记：`experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv` seq 0
  （`raw_wc_l=2484 (absolute anchor; not a delta)`）+ seq 17（RVV substrate 行明标「substrate-origin≠2484；2484=IME 首接入」）。
- 脚本常量：`tools/visibility/recompute_ledger_anchor.sh:32` `FOUNDING_SLICE_ANCHOR=2484` / `:33` `FOUNDING_SLICE_REF=54465ee7`。

### 1501 — 成立（byte-exact @2dd654d8，且 live 亦 = 1501 无漂移）
```
git ls-tree -r --name-only 2dd654d8 -- lib/Dialect/Scalar lib/Plugin/Scalar \
  include/TianChenRV/Dialect/Scalar include/TianChenRV/Plugin/Scalar | grep -v CMakeLists.txt \
  | while read f; do git show "2dd654d8:$f" | wc -l; done | awk '{s+=$1}END{print s}'
# => 1501
```
逐文件：878 ScalarBackendEmissionDriver.cpp · 294 ScalarExtensionPlugin.cpp · 150 ScalarOps.td ·
56 ScalarExtensionPlugin.h · 37 ScalarEmitCRouteProvider.h · 29 ScalarBackendEmissionDriver.h ·
23 ScalarEmitCRouteProvider.cpp · 18 ScalarDialect.cpp · 16 ScalarDialect.h = **1501**。
- 台账登记：`T2_C2_ledger_marginal_cost.csv` seq 14（`raw_wc_l=1501 / cloc_approx=1148`）。
- live `family_ledger.py report --family Scalar` 现亦印 `raw_wc_l=1501`（家族接入后未再长 → live==founding，巧合但真）。

### 757 / 907 — 成立（emitter 住址；LED-2 报告 §三手算，本审计机核复算一致）
- 757 = `54465ee7:lib/Plugin/IME/IMEBackendEmissionDriver.cpp`(730) + `include/.../IMEBackendEmissionDriver.h`(27)。
- 907 = `2dd654d8:lib/Plugin/Scalar/ScalarBackendEmissionDriver.cpp`(878) + `include/.../ScalarBackendEmissionDriver.h`(29)。
- 报告出处：`docs/reports/2026-07-12-C2-LED-2-边际曲线-三家族点.md:90`（emitter 行 `757/511 | 907/669`）+ `:96`（878 单文件最大头）。
- 原本这两数**仅在报告里手算、无脚本复算入口** → 本审计新增 `emitter_raw_wc_l` 字段闭合。

---

## 三、发现的缺口 + 已施工修复

**指控（隐含）：ledger 支持 `--at-commit` 重跑 → 部分成立（缺口真实）。**

修前状态：
1. `recompute_ledger_anchor.sh` 把 2484 **硬编码为常量**（`:32`），运行时只印 live IME 大小（现 5799，随 C3′ 家族扩展漂移），**不从 `54465ee7` 复算 2484**。2484 的真复算命令只散落在 LED-2 报告附录。
2. `family_ledger.py report` 经 `_iter_code_files`（`:97`）**只读 live 工作树**，无 `--at-commit`；对 Scalar 恰好 live==1501，但对 IME live≫2484，无法在脚本内复现任一 founding-slice pin。
3. 757/907 emitter 住址**无任何脚本复算入口**。

已施工（`.trellis/scripts/family_ledger.py`，additive、不改 live 路径）：
- 新增 `_git_blob()`（不 strip，保 raw wc-l 精确）+ `gather_code_loc_at_commit(ref,dirs)`：纯 `git ls-tree`/`git show` 复算，**checkout-free**（脏树可跑），并处理 2026-07-12 改名（`include/Weft`↔`include/TianChenRV` 双前缀探测去重）。
- `report` 增 `--at-commit REF`：印 founding-slice `code_LOC`（raw/cloc/files + `emitter_raw_wc_l`），`$meta` 记 ref 全 hash + manifest sha256。
- 保持 live 路径与 self-test 不变（回归见 §四）。

**未改**（硬纪律）：T2 CSV 数值 pin / recompute_ledger_anchor.sh / 报告 / spec / T8 / memory / board。
（`recompute_ledger_anchor.sh` 运行会重生 `T2-ledger-anchor.md` auto-gen 文件，我核实后已 `git checkout` 还原，工作树仅留 `family_ledger.py` 一处改动。）

---

## 四、验收（任何人在 pin 上重跑印出同数）

```
$ python3 .trellis/scripts/family_ledger.py report --family IME    --at-commit 54465ee7
  raw= 2484 cloc= 1866 files= 7 emitter= 757        # ✓ 2484 / 757
$ python3 .trellis/scripts/family_ledger.py report --family Scalar --at-commit 2dd654d8
  raw= 1501 cloc= 1148 files= 9 emitter= 907        # ✓ 1501 / 907

$ python3 .trellis/scripts/family_ledger.py --self-test            # ALL PASS (7/7)  回归绿
$ python3 .trellis/scripts/family_ledger.py report --family Scalar # live raw=1501    未受影响
```

四数逐一在其 pin 上机核复现，验收通过。

---

## 五、遗留 / 建议（供主会话，非本审计施工）

1. **口径待裁（canon 必问，非本审计范围）**：canon `[C2-1]` 的 `<300` 目标 vs 实测 1501（4-5×）
   —— LED-2 已给「两 regime」reframe（独立家族付真 emitter 成本 / 集成子扩展近零），
   但 `<300` 单点目标是否正式撤销/改写仍 pending 用户裁（见 `docs/reports/2026-07-13-...真欠地图.md:95`）。本审计不动 canon。
2. 可选加固：把 `--at-commit` 复现纳入 CI（对 IME@54465ee7=2484 / Scalar@2dd654d8=1501 做断言），
   使 pin 复现成为 gate 而非仅手跑。属可逆增补，留主会话决定是否值得。
