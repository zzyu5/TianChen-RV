# [RENAME] TianChen-RV / TCRV → **Weft** — MOVES map & rename record

> 2026-07-12 · 裁四 独占 [RENAME] 会话 · 机械改名 · byte-exact 零漂移 · 主会话验后 commit
> 复演脚本：`tools/rename_to_weft.py`（`--plan` / `--content` / `--moves` / `--all` / `--collapse-ns`）

## 1. Token map (case-sensitive · 互不重叠)

| old | new | 用途 |
|---|---|---|
| `tianchenrv::tcrv` | `weft` | **C++ namespace 对折**（见 §4）— project::dialect-group 两级折成单级 |
| `TianChen-RV` | `Weft-RV` | RISC-V 实例文案（hyphenated prose） |
| `TianChenRV` | `Weft` | CamelCase 代码标识符 / 项目名 / CMake target |
| `TIANCHENRV` | `WEFT` | ALL-CAPS CMake option/var |
| `tianchenrv` | `weft` | C++ 根 namespace |
| `TCRV` | `WEFT` | acronym: .td def 名 / C++ 类 / 宏 |
| `tcrv` | `weft` | 方言前缀（tcrv_rvv/tcrv_ime/…、tcrv-opt、tcrvrvv、`tcrv.` IR mnemonic） |

**故意不改 `TianchenRV`（大写 T·小写 c）** = 仓库 checkout 目录 basename（`/home/kingdom/phdworks/TianchenRV`），只出现在绝对路径 + 历史散文里；改它会打断真实路径。六个 token 均非其子串 → 自动保留。

8 方言字符串名：`tcrv`(核心 Exec)→`weft`，`tcrv_rvv/ime/scalar/offload/template/tensorext_lite/toy`→`weft_*`。

## 2. Scope（改）vs 具名例外（留）

**改（功能 + CI + 活契约面·158 处 path move · 1367 处 content）**：
`include/ lib/ schema/ cmake/ scripts/ test/ .github/ CMakeLists.txt`、
`tools/{tcrv-opt→weft-opt, tcrv-translate→weft-translate, lint, fuzz, ci, visibility, hooks, CMakeLists.txt}`、
`.trellis/scripts/ .trellis/spec/`、`docs/canon/ docs/method/`、`README.md CLAUDE.md AGENTS.md RTK.md`。

**具名例外（历史 / append-only / checkout-path·留原样）**：
`.trellis/backup/ .trellis/tasks/ .trellis/workspace/ .trellis/.template-hashes.json`、
`docs/reports/`（本报告在此）、`artifacts/`、`experiments/`、`tools/e2e-harness/ tools/bench/`、
以及全仓 `TianchenRV`（checkout-path token）与 `/home/kingdom/phdworks/TianchenRV` 绝对路径。
理由：历史记录里的旧名 = 当时事实，改写 = 破坏审计链，且不影响 byte-exact / CI / build。

## 3. 结构性 path moves（file/dir 基名变化，非纯目录平移）

- `cmake/TianChenRVToolchain.cmake` → `cmake/WeftToolchain.cmake`
- `docs/canon/TianChen-RV_定位-v2.md` → `docs/canon/Weft-RV_定位-v2.md`
- `docs/canon/TianChen-RV_实验总纲v1.md` → `docs/canon/Weft-RV_实验总纲v1.md`
- `docs/canon/TianChen-RV_执行总纲v2.md` → `docs/canon/Weft-RV_执行总纲v2.md`
- `docs/canon/TianChen-RV_科研目标总纲v2.md` → `docs/canon/Weft-RV_科研目标总纲v2.md`
- `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h` → `include/Weft/Conversion/EmitC/WEFTEmitCLowerableInterface.h`
- `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h` → `include/Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.h`
- `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td` → `include/Weft/Conversion/EmitC/WEFTEmitCLowerableOpInterface.td`
- `include/TianChenRV/InitTianChenRVDialects.h` → `include/Weft/InitWeftDialects.h`
- `lib/Conversion/EmitC/TCRVEmitCLowerableOpInterface.cpp` → `lib/Conversion/EmitC/WEFTEmitCLowerableOpInterface.cpp`
- `lib/InitTianChenRVDialects.cpp` → `lib/InitWeftDialects.cpp`
- `tools/tcrv-opt/tcrv-opt.cpp` → `tools/weft-opt/weft-opt.cpp`
- `tools/tcrv-translate/tcrv-translate.cpp` → `tools/weft-translate/weft-translate.cpp`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md` → `.trellis/spec/core-dialect/weft-exec-contract.md`

外加 **144** 处纯目录平移 `include/TianChenRV/**` → `include/Weft/**`（基名不变，随目录移动）。全 158 条 git 均识别为 rename（R/RM）。

## 4. ★唯一判断点：C++ namespace 对折（`tianchenrv::tcrv` → 单 `weft`）

原 C++ 命名空间 = `tianchenrv::tcrv::<dialect>`（项目::方言组::方言）。若 `tianchenrv`→`weft` 与 `tcrv`→`weft` 各自独立替换 → 产生 `weft::weft::<dialect>`，内层 `weft` 遮蔽项目根 → `weft::plugin`/`weft::support` 等跨组引用查找失败（编译 error: `'plugin' in namespace 'weft::weft' does not name a type`）。

**裁定**：把限定对 `tianchenrv::tcrv` 整体折成单 `weft`，方言落到 `weft::rvv`/`weft::exec`…（与 `weft::plugin`/`weft::support` 平级，符合 MLIR 惯例如 `mlir::rvv`）。
- 代码用 C++17 nested-namespace 语法（`namespace weft::weft::rvv {`）→ 纯 token 替换 `weft::weft`→`weft` 即完成对折，零 brace 手术。
- 仅 2 个 block-form TU（`lib/Dialect/RVV/IR/RVVDialect.cpp`、`RVVDialectInternal.h`）用 `namespace weft {`×2 → 脚本 `collapse_block_namespace` 去重 opener/closer。
- **不改逻辑 / 不改 emit 字节**（namespace 是内部结构，非发射输出）。已 byte-exact 交叉验证（§6）。

## 5. [F-2′] schema.def 触碰 = extension-not-modification（记 minor）

`schema/capability.schema.v1.json` 含 4 处 `include/TianChenRV/…` 路径引用 + 1 处 `tcrv.exec.target` 散文 → 改名后 normalized-SHA256 漂移。按裁四 [F-2′] 预案：SHAPE 未变（keys/enum/relations 全同），仅标识符字符串 VALUES 改 → **minor bump v1.0.0→v1.1.0 · additive**，`schema/VERSIONLOG.md` 加一行（新 hash `bff75104…`），report gate 重新 seal 且**不改写已封的 v1.0.0 hash**。`report --check` = GREEN。非 canon 级、非家族接入违规。

## 6. 验证

- **clean rebuild**：fresh `build-weft/`（LLVM-20/Ninja/gcc）→ `weft-opt`+`weft-translate`+20 gtest 全 0-error 链成。weft-opt 注册 8 方言 `weft/weft_ime/weft_rvv/weft_scalar/weft_offload/weft_template/weft_tensorext_lite/weft_toy`、pass 名 `--weft-*`、IR 发射 `weft.exec.*`/`#weft_rvv.policy` 一致。
- **full lit（check-weft）**：**904/907 (99.67%)**。3 failed = `test/Scripts/rvv-generated-bundle-abi-e2e-*`（self-test + 2 dry-run）**在 pristine HEAD 用旧 tcrv-opt 二进制复现同一失败**（"missing pattern" / self-test AssertionError）→ **pre-existing·非改名回归**。改名新增失败 = 0。
- **falsifier 六门 [F-1..F-6]** + 相邻门（schema-def / redteam / monolith-retire / frontdoor-provenance / cert-requirements / retired-index / opponent-facts-pin / perf-covered）+ 全 `--self-test`：**全 GREEN**（含 F-1 零分支、F-3 family-locality 对改名后源树的真一致性 grep）。`schema/retired-index.generated.json` 已 `gen_retired_index.py` 重生 → fresh。
- **byte-exact golden**：weft-opt 发射 vs 旧 tcrv-opt 发射经 token-map 折算 → **6/6 byte-identical**（ime-mma 1736B / ime-mma-su 1759B / ime-matmul 3293B 实体匹配）。改名不改 emit 字节，只名字变。
- **dir-hygiene CI**（experiments/ 布局）：3 default 门 RED，但**在 pristine HEAD 同样 RED**（我零改 experiments/）→ pre-existing·域外。

## 7. 残留

- 功能面 `tcrv/TCRV/TianChenRV/tianchenrv/TIANCHENRV` 真残留 = **0**；唯一命中 = `schema/VERSIONLOG.md` 那行**描述改名本身**的 changelog（自证条目，故意保留旧名做审计）。
- 例外面（历史/checkout-path）：`.trellis/backup`(2734)、`.trellis/tasks`(1228)、`experiments`/`artifacts`/`tools/e2e-harness,bench` 等 ~4628 文件带 tcrv/TCRV + 807 文件带 `TianchenRV` checkout-path token — 全属 §2 具名例外，故意不改。
