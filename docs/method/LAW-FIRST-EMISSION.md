# [T8·LAW-FIRST-EMISSION] 元规律 —— 首次发射不成熟律（2026-07-11 用户裁定立卷）

> **归属**：T8 win/loss/gap 台账（`experiments/active/result-tables/T8_winloss_gap_ledger.csv`）的元规律层。
> **性质**：C3′（模板产出质量）核心实证素材 + perf-covered 按批变绿的机制依据。

## 律文
> **首次发射（first emission）功能正确、但性能不成熟；不成熟的根因是逐片段/逐块的发射仪式开销（非算法、非物理墙、非指令微质量），可【具名】为一个成熟度 GAP；关闭 GAP 的成熟杠杆可【能力键控】、可【族级复制】。**

推论：
1. **结构 certified ≠ perf-covered**：一个格前门构造 + byte-exact 认证（结构轴），其首次发射几乎必然是性能不成熟态；perf-covered 只在**成熟杠杆落地后**变绿。
2. **GAP 可具名 = 进度可度量**：每个不成熟态出一个具名 `[GAP-XXX]`（逐格/族级），关闭即一批格变绿——这是测量总攻"欠账表随批递减"的机制。
3. **杠杆能力键控 + 族级复制**：成熟杠杆按板/格能力事实选择（非逐格手调），一次落地惠及全族（如一个 leaf-pipeline 改法惠及 IME 三格）。

## 四例同律连档（第四例 2026-07-11 入档）
| # | 格/族 | 首次发射不成熟态 | 逐片段仪式开销（根因） | 成熟杠杆（具名·能力键控） | 证据 |
|---|---|---|---|---|---|
| 1 | **K-quant S6**（repack） | 未流水首构造 5× 慢 | 全展开 regfile spill | S6 tiling（mbf×lmul·SEW/LMUL 键） | 反汇编认瓶颈（性能宪章规则 1） |
| 2 | **RVV col-outer**（token-tile schedule） | 热 micro 1.884× → 冷 e2e 0.764× 蒸发 | 4-token 浅瓦片致权重冷流重读 32× | col-outer loop-interchange（内存层级事实键） | [CASE-MICRO-E2E]·LLC-miss↓5.2×·翻正 2.47× |
| 3 | **K1 vl16**（lane-width） | 部署 half=8 半宽 0.75× | 半宽烤成常量·缺 zvl256b | vl=16 满宽（VLEN 事实键·纯 capability-input） | Win-K1-VLEN RATIFIED 1.085× |
| 4 | **IME leaf-pipeline**（★2026-07-11 · G4-M3） | as-emitted 叶子全 M 输公平向量 ~4× | 每 vmadot 外包 e8↔e32 vsetvli 切换+清零+存回+标量 acc[] | 寄存器驻留批处理叶子（单 vsetvli·只存一次·[GAP-IME-LEAF-PIPELINE]） | T5b 板测 a9d7a9c6·Cell1b ~2× 胜·Cell3c 证真 matrix-MAC（128 vs 32 MAC/指令） |

## 与项目主张的关系
- **直接实证"能力键控须延伸到发射器成熟度/内存层级"**（性能收敛主论点）：四例都不是物理墙/算法败/编译器 artifact，是发射器成熟度题——与项目核心主张同路。
- **反面清单守卫**：指令微质量非赛场；对手贴墙 parity=满分；micro↛e2e 铁律（例 2/4 都带此 caveat）。
- **T8 交叉引用**：四例的 win/loss/gap 明细行在 `T8_winloss_gap_ledger.csv`；本律文是其上层归纳。

## 用法
- 测量总攻中每遇首次发射 LOSS，先问"是否 first-emission 不成熟律的又一例"——是则**具名 GAP + 黄格出口**（非声明例外、非物理 parity），关闭杠杆后重测变绿。
- perf-covered 汇报时，first-emission-immature 格标"黄格·GAP 待修"，**不计绿格**（诚实），但**记为可变绿的具名机会**（区别于物理 parity-at-floor 与声明例外）。
