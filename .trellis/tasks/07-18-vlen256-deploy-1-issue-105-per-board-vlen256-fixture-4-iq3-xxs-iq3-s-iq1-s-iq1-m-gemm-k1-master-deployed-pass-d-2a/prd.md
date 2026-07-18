# PRD · VLEN256 per-board fixture 部署（裁决1·ISSUE-105 放行·4 格 → deployed PASS）

> **权威** = 用户裁决1（2026-07-18）：「ISSUE-105 部署：放行 per-board VLEN256 fixture 落地 4 格真硬赢（iq3_xxs 1.38 / iq3_s 1.219 / iq1_s 1.340 / iq1_m 1.762）。**check 过即入 master 转 deployed PASS**。同时 [D-2a] 运行期分发继续推进作长期形态——两条不互斥·先 fixture 落地拿分·[D-2a] 成熟后迁移。」
> **性质** = **部署**（4 格 proven flip → deployed PASS·a6ad trellis-check 已 CONFIRMED）。**赢=地盘**（byte-exact + 性能达标 = 地盘+4）。

## 一、背景：flip 已证·gap = GEN_SEAL 对 k1 发 VLEN128 fixture

- ISSUE-105 根因：`gemm_tile.sh` 消费预发射 fixture leaf `$ASSETS/kernels_grid4/${FMT}_gemm.c`·**GEN_SEAL 对所有板用 `march=rv64gcv`（VLEN128·half_lanes=8）**·含 k1（真 VLEN256）= 半宽欠用。
- **4 格 proven flip（a6ad trellis-check 独立 k1 CONFIRMED·byte-identical GEN_SEAL baseline·objdump 真宽 AVL 8→16 非 re-roll·apples-to-apples oppX 不变·byte-exact 4-arm·真硬赢 vs 手调 vl256）**：iq3_xxs 0.65→1.38 · iq3_s 0.61→1.20 · iq1_s 0.59→1.34 · iq1_m 0.59→1.77。
- 宽化机制**已存在**（`deriveRepackHalfLanes(minVLEN)`·VLEN256→16）·仅 GEN_SEAL/harness 对 k1 的 emit march 需从 `rv64gcv`→`rv64gcv_zvl256b`。

## 二、你要做：per-board VLEN256 fixture 路由

1. **定位 GEN_SEAL / harness 的 fixture emit march 选择**（`tools/bench/` + fixture 生成路径·`kernels_grid4/${FMT}_gemm.c` 怎么发射）。
2. **加 per-board 路由**：**k1（VLEN256 硬件）→ emit `march=rv64gcv_zvl256b`（half_lanes=16）**·rvv（VLEN128）维持 `rv64gcv`。**这是构造期 / 发射期的 per-board fixture 选择·不是 per-dispatch 运行期检查**（[NG-3] per-dispatch 强制检查**永禁**·不碰）。**[D-2a] 运行期分发是长期形态·本 task 是 fixture 落地拿分·两条不互斥·勿动 [D-2a] 已落地的 A+B**。
3. **k1 板验 4 格**（`ssh k1`·`gemm_tile.sh` 走 deployed（现 = VLEN256）fixture）：byte-exact G1（4-arm·mism=0）+ cold 2-seed ≥0.8（复现 1.38/1.20/1.34/1.77 = deployed PASS）+ objdump 真宽（deployed 叶现 AVL=16）。
4. **rvv 无回归**（rvv 维持 VLEN128 fixture·byte-exact 不变·sealed 不误伤）。

## 三、验收

1. **per-board fixture 路由落地**（k1→zvl256b·rvv→rv64gcv·objdump 证 k1 deployed 叶现 AVL=16 真宽）·构造期选择（非 per-dispatch）。
2. **4 格 k1 byte-exact GREEN + cold 2-seed ≥0.8**（deployed 现实现在 = PASS·非 proven-not-deployed）。
3. **rvv + 其他 k1 格无回归**（byte-exact·lit 绿·sealed 不误伤——**列出 sealed/master 计数变动**：4 格从 具名-X → deployed PASS 的账目影响，报 main 定 recon-dict 入账）。
4. **成色**：对手 = vl256 **手调档**（真硬赢·CROSSOP 系统账）·OPP-G generic 便宜档禁称硬赢。
5. **0 造数**·byte-exact 硬门·objdump 全量·**禁 commit·禁 add -A**（提交 + recon-dict/master 入账由 main 做·判据级）·证据 `experiments/runs/` 或 scratchpad。

## 四、★main 入账边界（判据级·你只实现+板验+报数）
- **master / recon-dict 入账（4 格 具名-X → deployed PASS）= main 做**（裁决1 授权·但 recon-dict 编辑 + 分母/sealed 计数影响属判据级）。你报：4 格 deployed cold + byte-exact + sealed/master 账目变动清单。
- **check**：本 task = 部署（headline flip）·施工后 main 派 trellis-check 独立复核 deployed fixture（byte-exact + cold PASS + 无回归 + sealed 计数正确）再 commit。

## 五、触碰集 / 遗留
- 触碰：GEN_SEAL / harness fixture emit march per-board 路由（`tools/bench/`）+ k1 板验。**与 iq2/iq4 harness task（`07-18-iq2-iq4-widen-harness`）触碰 bench/harness 码重叠 → 本 task 先行·iq2/iq4 后置**（避冲突）。用 k1 板。
- **遗留**：iq2/iq4 flip（harness 建成后·同 per-board fixture 路由自动覆盖）→ 部署面再扩 4 格。[D-2a] 运行期分发迁移（长期·裁决1）。
