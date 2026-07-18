# Research: ★ISSUE-105 kernel-账正解 —— [D-2a] 怎么把 k1 半宽群变 deployed

- **Query**: [D-2a] 如何解 ISSUE-105（一份二进制运行期分发=与 ggml 同形态⟹记 kernel 账非系统账）；现只一板一份；落地后 k1 半宽群怎么变 deployed
- **Scope**: internal（ISSUE-105 + GEN_SEAL fixture 路径 + 构造期宽度派生 + 账目 canon）
- **Date**: 2026-07-18

## 一、ISSUE-105 现状（问题正本）

`.trellis/spec/issues/门与工具.md:389-413`（状态 **待裁·判据级**；index 行 `issues/index.md:300`）：

- **实质**（`:392-395`）：bench `gemm_tile.sh` 消费**预发射 fixture leaf** `$ASSETS/kernels_grid4/${FMT}_gemm.c`（harness **不发射·只跑**）。GEN_SEAL 现对**所有板**用 `march=rv64gcv`（VLEN128·half_lanes=8）发射该 fixture，**含 k1（真 VLEN256 硬件）** ⟹ k1 上跑 VLEN128 叶 = **半宽欠用**（每条向量 op 只用 8/16 lane）。
- **proven 证据**（`:396-399`，task `07-18-k-mech1-vlen-leaf`·k1 板测）：`iq3_xxs@k1` 用同叶 `march=rv64gcv_zvl256b` 重发射（half_lanes=16）→ cold 0.65 LOSS → **1.38 WIN**；objdump 真宽（`vsetivli zero,8,e32,m2`→`zero,16`、vset 5397→2515、gather 1024→512、非 re-roll）；G1 byte-exact PASS；2 seed（1.3838/1.3782）。
- **deployed-vs-proven**（`:400-403`）：标准 k1 分支 march = `rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`（**无 zvl256b**）⟹ **deployed k1 fixture leaf 仍 VLEN128 = 0.65 LOSS**；master 主表 `(gemm_tile,iq3_xxs,k1)` 维持 具名-X cold=0.6474（proven 赢归攻坚台账、不进 master）。
- **候选裁法**（`:409-411`，供裁不代裁）：① per-board fixture 变体（k1 走 `rv64gcv_zvl256b` half_lanes=16 + harness 板路由选叶）；② 维持现状（板无关 VLEN128）；③ 折中（仅已 proven 格加 k1-VLEN256 变体）。

## 二、根因机制解剖（宽度键**已在编译/构造期能力派生**）

关键事实：**宽化机制已存在**，缺口不在选择器，在 **fixture 发射时 march 被钉死**。

- `lib/Plugin/RVV/RVVCapabilityProfile.cpp:289` `deriveMinimumVLEN(march,hints)`：`rv64gcv`→128、`rv64gcv_zvl256b`→256（`:299`）。
- `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1222` `deriveRepackHalfLanes(vlenBits)`：`min(vlen/16,16)` ⟹ 128→8、256→16、<128→0（`:1223-1225`）。
- task journal 实证（`.trellis/tasks/archive/2026-07/07-18-07-18-k-mech1-vlen-leaf/task.json:24`）：「宽化机制**已存在** = `deriveRepackHalfLanes(minVLEN)` 256→16……缺口 = harness 一直只发射+测 VLEN128 fixture leaf（march=rv64gcv→half_lanes=8）跑在 VLEN256 硬件(k1)=半宽。」无源码改动即用 `rv64gcv_zvl256b` 重发射得 1.38 WIN。

⟹ 「一板一份」的根源 = **GEN_SEAL 在编译期把 march 钉死为 `rv64gcv`，一次性预发射一个宽度的 leaf**；部署物是那个预发射 `.c`。宽度是编译期常量，不是运行期可选。

## 三、[D-2a] 如何把它变 kernel 账（PRD §二.4 的"正解"逻辑链）

### 3.1 系统账 vs kernel 账 的判据（canon 法源）

- 账目分野：`.trellis/spec/canon/账目与分母.md`（kernel/system 账）；性能主张绑 `账本(kernel/system)`（CLAUDE.md §硬件与测量 3）。
- ggml 出货形态 = **一份二进制 + 运行期分发**（FMV/IFUNC/hwprobe，单家族·仅运行期·同范式）——见相关工作对照面 E：`.trellis/spec/canon/暂定-科研主张.md:78` 「E 运行期分发现状（FMV/IFUNC/hwprobe：单家族、仅运行期、同范式）」。

### 3.2 正解逻辑链

1. **今日一板一份 = 系统账优/劣势混入**：per-board 预发射 fixture 让「谁跑哪个宽度」由**构建期人手选 march** 决定。这既是 k1 半宽欠用的病根（发窄），也意味着任何"发对宽度"的赢都夹带**"我们替这块板手挑了 fixture"的系统账**，不是纯 kernel 质量对比。
2. **[D-2a] = 一份二进制、装载期解析 → 运行期分发到最宽合法宽度**：装载期读能力事实（k1→VLEN256）→ 展开 → declared-instance-hash → 解析记录 → 该进程分发到 half_lanes=16 的变体。**同一二进制在 rvv(128) 上分发到 half_lanes=8、在 k1(256) 上分发到 16**。
3. **⟹ 与 ggml 同形态**：一份二进制运行期按 hwprobe 分发 —— 这正是 ggml/FMV 的形态。两边都"一份二进制自己在目标板上选宽度"，对比落到**同宽度下 kernel 谁更快** = **kernel 账**，而非"谁的构建脚本替板挑了 fixture"的系统账。
4. **k1 半宽群变 deployed 的机制**：[D-2a] 落地后，k1 上跑的**就是**装载期解析出的 half_lanes=16 变体，无需 per-board 手改 GEN_SEAL march、无需热改流水线 re-baseline —— proven 的 1.38（及扇出候选 iq3_s@k1 0.61、iq1 系）从"proven 但 deployed 仍 0.65"转为"deployed = 装载期解析即宽"。宽度选择从**编译期 march 钉死**上移为**装载期能力解析**。

### 3.3 与现有骨架的接点

- declared-instance-hash 已是**板身份代理**：不同板→不同展开事实集→不同 hash（`RVVRepackTilingSelection.h:256`「A different board => a different hash => a MISS」）。[D-2a] 装载期解析产出的正是"本进程 = 哪个 hash/哪套事实"，天然键控到对的宽度变体与对的测量记忆行。
- guard param 编译期戳（`DispatchRuntimeGuard.cpp:324-338`）是装载期解析的落点雏形。

## 四、落地后 k1 半宽群的账目转换（须逐格 byte-exact，禁外推）

- ISSUE-105 明列扇出（`门与工具.md:404-405`）：`iq3_s@k1`（0.61）+ iq1 系（双板皆输）**同 VLEN256 半宽病**、likely 同 lever；但「须逐格 byte-exact + 板测·**禁按此外推计数**」（连 `kernel-wins-dont-transplant-to-e2e`）。
- ⟹ [D-2a] 提供的是**部署通道**（deployed 通道打开）；每格是否翻 PASS 仍须逐格 byte-exact + k1 板测，proven≠deployed（连 `zero-model cert-hardening`「部署变体≠证过变体」，`门与工具.md:400`）。

## 五、保守默认（[D-2a] 落地前·现行·勿破）

`.trellis/tasks/.../prd.md:29` + ISSUE-105 §保守默认（`门与工具.md:406-408`）：
- k1 维持 deployed 现实：master `(gemm_tile,iq3_xxs,k1)` 具名-X cold=0.6474；
- **不擅改 GEN_SEAL 发 k1-VLEN256**（re-baseline 所有 k1 gemm_tile 格 = 热改流水线·判据级·须先裁 ISSUE-105）；
- proven 1.38 记攻坚台账，不进 master、不扩分母、不开头条；sealed 9/83 不动，sha256 `dc876ef6` 守恒。

## 六、一句话结论

k1 半宽欠用的根因是 **GEN_SEAL 编译期把 march 钉死为 `rv64gcv`、一板一份预发射窄 leaf**，而宽度键 `deriveRepackHalfLanes(deriveMinimumVLEN(march))` **本已能力派生**。[D-2a]「一份二进制 + 装载期解析 → 运行期分发到最宽合法宽度」使部署形态**与 ggml 的 hwprobe 运行期分发同形**，从而把对比拉回 **kernel 账**，并为 k1 半宽群打开 deployed 通道（每格仍须逐格 byte-exact+板测，proven≠deployed，禁外推计数）。ISSUE-105 的裁定本身是**判据级**（re-baseline 所有 k1 gemm_tile 格），须先裁。
