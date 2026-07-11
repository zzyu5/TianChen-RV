# [T8·LAW-FIRST-EMISSION] 元规律 —— 首次发射不成熟律（2026-07-11 用户裁定立卷）

> **归属**：T8 win/loss/gap 台账（`experiments/active/result-tables/T8_winloss_gap_ledger.csv`）的元规律层。
> **性质**：C3′（模板产出质量）核心实证素材 + perf-covered 按批变绿的机制依据。

## 律文
> **首次发射（first emission）功能正确、但性能不成熟；不成熟的根因是逐片段/逐块的发射仪式开销（非算法、非物理墙、非指令微质量），可【具名】为一个成熟度 GAP；关闭 GAP 的成熟杠杆可【能力键控】、可【族级复制】。**

推论：
1. **结构 certified ≠ perf-covered**：一个格前门构造 + byte-exact 认证（结构轴），其首次发射几乎必然是性能不成熟态；perf-covered 只在**成熟杠杆落地后**变绿。
2. **GAP 可具名 = 进度可度量**：每个不成熟态出一个具名 `[GAP-XXX]`（逐格/族级），关闭即一批格变绿——这是测量总攻"欠账表随批递减"的机制。
3. **杠杆能力键控 + 族级复制**：成熟杠杆按板/格能力事实选择（非逐格手调），一次落地惠及全族（如一个 leaf-pipeline 改法惠及 IME 三格）。

## 六例同律连档（第四例 2026-07-11·第五-六例 2026-07-12 入档·含诚实反例 + 新根因类）
| # | 格/族 | 首次发射不成熟态 | 逐片段仪式开销（根因） | 成熟杠杆（具名·能力键控） | 证据 |
|---|---|---|---|---|---|
| 1 | **K-quant S6**（repack） | 未流水首构造 5× 慢 | 全展开 regfile spill | S6 tiling（mbf×lmul·SEW/LMUL 键） | 反汇编认瓶颈（性能宪章规则 1） |
| 2 | **RVV col-outer**（token-tile schedule） | 热 micro 1.884× → 冷 e2e 0.764× 蒸发 | 4-token 浅瓦片致权重冷流重读 32× | col-outer loop-interchange（内存层级事实键） | [CASE-MICRO-E2E]·LLC-miss↓5.2×·翻正 2.47× |
| 3 | **K1 vl16**（lane-width） | 部署 half=8 半宽 0.75× | 半宽烤成常量·缺 zvl256b | vl=16 满宽（VLEN 事实键·纯 capability-input） | Win-K1-VLEN RATIFIED 1.085× |
| 4 | **IME leaf-pipeline**（★2026-07-11 · G4-M3） | as-emitted 叶子全 M 输公平向量 ~4× | 每 vmadot 外包 e8↔e32 vsetvli 切换+清零+存回+标量 acc[] | 寄存器驻留批处理叶子（单 vsetvli·只存一次·[GAP-IME-LEAF-PIPELINE]） | T5b 板测 a9d7a9c6·Cell1b ~2× 胜·Cell3c 证真 matrix-MAC（128 vs 32 MAC/指令） |
| 5 | **[GAP-DEQ-KQUANT-UNPACK]**（★2026-07-12 · 流式 dequant q4_K/q5_K·**诚实反例扩律**） | scalar 逐元素仪式首发·kernel 账 0.636×（对称更狠） | 每 elem loadIntAt/nibble-peel/i2f/scalar fMul·零 RVV intrinsic（RVVToEmitCForwardElementwise.cpp:4055） | 向量化解码叶（SEW8→SEW32 widen·LMUL 键·复用 e8m2 idiom·族级 q2/q3/q5/q6_K） | ★**边界反例**：修完 **e2e-inert**（off 热 matmul 路·<噪声）→ 区分"可修黄格带真潜伏赢（例1/3/4）"vs"fix-is-e2e-inert"·报告 `2026-07-12-三具名GAP修复评估.md` |
| 6 | **[GAP-CLANG-GATHER-TRAP]**（★2026-07-12 · iq4_nl 码本 gather·**新第5根因类**） | 发射器不拥有 lowering·部署编译器 autovec 选中 uarch-hostile gather（clang→vluxei·SG2044 慢~5×·gcc scalar 快） | 非"逐片段仪式"（前4例）而是 **deploy-autovec-roulette + 可路由 uarch quirk** | scalar-pin（键 `uarch.gather_slow`·gcc/clang 双双走表·robust emitter-owned·族级码本叶） | ★真赢潜伏在**分离的热 vec_dot iq 族 gap**（非本冷叶）·实例化 canon 具名-但-未建 `uarch.vrgather_slow` P6·撤回-S6 的**诚实正镜像**（对称 4.83× 不蒸发·clang-deploy 0.94× 不利） |

## 与项目主张的关系
- **直接实证"能力键控须延伸到发射器成熟度/内存层级"**（性能收敛主论点）：四例都不是物理墙/算法败/编译器 artifact，是发射器成熟度题——与项目核心主张同路。
- **反面清单守卫**：指令微质量非赛场；对手贴墙 parity=满分；micro↛e2e 铁律（例 2/4 都带此 caveat）。
- **T8 交叉引用**：四例的 win/loss/gap 明细行在 `T8_winloss_gap_ledger.csv`；本律文是其上层归纳。

## 用法
- 测量总攻中每遇首次发射 LOSS，先问"是否 first-emission 不成熟律的又一例"——是则**具名 GAP + 黄格出口**（非声明例外、非物理 parity），关闭杠杆后重测变绿。
- perf-covered 汇报时，first-emission-immature 格标"黄格·GAP 待修"，**不计绿格**（诚实），但**记为可变绿的具名机会**（区别于物理 parity-at-floor 与声明例外）。
