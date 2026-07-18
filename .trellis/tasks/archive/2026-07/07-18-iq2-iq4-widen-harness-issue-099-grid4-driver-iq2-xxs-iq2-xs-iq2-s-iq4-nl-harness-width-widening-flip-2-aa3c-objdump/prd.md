# PRD · iq2/iq4 宽化 harness + 板测扩 flip（裁决2·最高优先·ISSUE-099）

> **权威** = 用户裁决2（2026-07-18）：「iq2/iq4 宽化 harness（ISSUE-099）：直接建、直接板测扩 flip。**最高优先**。」+ aa3c（task `07-18-k1-vlen256-widen-fanout`）：iq2_xxs/iq2_xs/iq2_s/iq4_nl **objdump-真宽已证·未板测**（grid4 driver 仅支持 {iq1_s,iq1_m,iq3_xxs,iq3_s}·tables+oracle 未建=ISSUE-099）。
> **性质** = harness 基建（build grid4 driver 扩展 + tables + ZERO-MODEL oracle）+ 板测扩 width-widening flip。**赢=地盘**（byte-exact + cold≥0.8 = 地盘+1/格）。

## 一、背景：aa3c 已证 objdump-真宽·卡在 harness

width-widening 机制**已存在**（`deriveRepackHalfLanes(minVLEN)`·VLEN256→16·换 emit march `rv64gcv`→`rv64gcv_zvl256b`·**无源改**）。aa3c 4/4 harness 格全翻（iq3_xxs 1.38/iq3_s 1.22/iq1_s 1.34/iq1_m 1.76·真硬赢 vs 手调 vl256）。**iq2_xxs/iq2_xs/iq2_s/iq4_nl objdump-真宽确认但 harness 未建**：

| 格 | vset B→W | gather B→W | ins B→W | AVL | widened leaf md5 |
|---|---|---|---|---|---|
| iq2_xxs | 5461→2515 | 1024→512 | 14945→7284 | 8→16 | `eb94d9af` |
| iq2_xs | 5477→2531 | 1024→512 | 16037→7324 | 8→16 | `8f906c28` |
| iq2_s | 5477→2531 | 1024→512 | 16037→7324 | 8→16 | `a36e6cf4` |
| iq4_nl | 656→328 | 64→32 | 1784→955 | 8→16 | `bf135f40` |

## 二、你要做

1. **定位 grid4 driver + tables + oracle**（aa3c note：grid4 driver 仅认 {iq1_s,iq1_m,iq3_xxs,iq3_s}·其余 `#error`）。扩展支持 **iq2_xxs / iq2_xs / iq2_s / iq4_nl**：
   - grid4 driver 的 format dispatch（去掉 `#error`·加 4 格）。
   - **每格 ZERO-MODEL oracle**（[K-5]·从实际输入零复用重算·对 ggml 真核 byte-exact）·grid/codebook 查表（iq2=grid·iq4_nl=16-entry codebook）。
   - 对应 `tools/bench/cells/` harness（契约合规：禁写仓库文件·全 stdout·经 runner·bench 唯一通道·签名带四元行键）。
2. **逐格板测**（k1·`ssh k1` 免密）：emit baseline(`rv64gcv`) + widened(`rv64gcv_zvl256b`)·byte-exact G1（4-arm anti-hollow·mism=0）·objdump 前后（复核 aa3c 的真宽 metrics·AVL 8→16·vset/gather 减半·**非 re-roll**）·k1 cold 2-seed（flush+idle 100%+gov performance）。
3. **verdict 逐格**：cold≥0.8 → **proven flip**（byte-exact + 性能达标 = 地盘+1）；cold<0.8 → **具名墙**（objdump 逐指令 + 残余 co-factor）。

## 三、★预期 —— 预测不是实测（先测不预告·[§五.15]）

aa3c 4 格锚全翻·但 **iq2/iq4 每格独立·不外推**（iq2 是不同 grid 结构·iq4_nl 叠 ISSUE-021 tiny-codebook gather）。**先测·不预告。**
- iq4_nl：宽化是**叠加 co-factor**（非主 lever）·主结构 = ISSUE-021 codebook-gather（16-entry vluxei vs 对手 vrgather·selector 已 decline codebook·**禁性能名义**）——宽化后是否翻取决于 codebook 是否成为新瓶颈·如实分离两 co-factor。

## 四、验收

1. **harness 建成**（grid4 driver 4 格 + tables + ZERO-MODEL oracle + cells/ harness·契约合规·`--self-test` 若有则绿）。
2. **逐格 byte-exact GREEN**（G1·4-arm·mism=0·对 ggml 真核）。
3. **逐格 objdump 真宽**（复核 aa3c metrics·AVL 8→16·非 re-roll·[GAP-P1] 守门）。
4. **逐格 k1 cold 2-seed + verdict**·**runs.log 每格一行**（proven flip / 墙·同 iq3_xxs 锚格式）。
5. **deployed 路径（裁决1）**：本轮板测 flip 格·**check 过即可入 master 转 deployed PASS**（per-board VLEN256 fixture·裁决1 放行）——但**master 入账 + deploy 由 main 做**（你只板测 + 报数 + 建 harness）。sealed 不动。
6. **成色**：对手 = vl256 **手调档**（真硬赢·CROSSOP 系统账）；CROSSOP 便宜档 cold_G（vs generic）**禁称硬赢**·标 opp-immaturity。iq4_nl codebook 便宜档同律。
7. **0 造数**·byte-exact 硬门·objdump 全量·**禁 commit·禁 add -A**（harness 代码你写·但提交由 main）·master/sealed 不动·证据 `experiments/runs/<run-id>/` 或 scratchpad。

## 五、触碰集 / 遗留

- 触碰：grid4 driver（format dispatch 扩 4 格）+ tables（iq2_xxs/iq2_xs/iq2_s grid + iq4_nl codebook）+ ZERO-MODEL oracle（每格）+ `tools/bench/cells/` harness + bench 跑（k1）。**harness 代码 = repo 写**（本 task 的交付物·非 0-repo-write 类）·但**禁 commit**（main 提交）。用 **k1 板**。
- **遗留**：flip 格 → 裁决1 deploy 候选（master 转 deployed PASS·per-board fixture）；未 flip 格 → 具名墙 + co-factor 诊断。harness 建成后 = ISSUE-099 该族关闭·iq2/iq4 width-widening 板测面补齐（proven-not-deployed → 裁决1 deployed）。
