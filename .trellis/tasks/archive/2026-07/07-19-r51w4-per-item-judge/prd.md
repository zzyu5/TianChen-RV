# PRD — W4 逐条判（零板时·禁打包·逐值判定同格计分）

## 哲学锁 + 记分牌修正（用户裁）
**每条逐值判定 = 消灭一个未知 = 与判决实验同格计分。**「登记不是进展」只针对「把待裁登记当主产品」·**不贬低逐条判定**（逐条判定是 A 线砌砖）。🔴 **禁用「N 判决实验全绿」打包顶替逐条——每条单独计分。** 零板时（全静态/机检）。

## 四子项（逐条·禁批发豁免）

### ① 37 条 f 输入签名表（18 单侧逐条判）
census pkg4 有 37 条 f。逐条出**输入签名表**·每条归一类：**{双侧(读 g 且 c) / 合法单侧+理由 / 已补输入 / 待裁}**。
- **18 单侧 f 逐条判**：Tier-2 纯 c 派生（如 `deriveMinimumVLEN` 本职解析 c）标**「合法单侧+理由」别误伤**；**F7**（`(void)` 丢弃 shape）· **F25**（g 全 baked）逐条判。
- **判决实验（每条·伪造不了）**：改被 void/baked 的输入·**θ 必须随之变**（加参数但输出不动 = 摆设现形·记「摆设」）。对 F7/F25 各构造一个「改输入看 θ 动没动」的 pinned 实验。
- 交 **37 条全表** + 18 单侧的逐条判定 + 摆设名单（若有）。

### ② 7 处软默认逐处正式登记
census/ISSUE 的 7 处软默认（value_or/default-override）·**逐处**：
- **θ18 在承重路径（ISSUE-002 同族活口）优先核**。
- 辩护词**逐处写成正式登记**（为何该软默认合法·或活口）；**写不成 → 排 fail-closed 改造**（列出改造点）。
- 🔴 **禁批发豁免·一处一条理由**。

### ③ 9 零消费键二选一（+ 核对已删 3 键考古）
census pkg2 的 9 个 provider 属性零消费（vlenb/clang/cmake/compile_run 4键/march:value/mabi）·**逐个二选一**：
- **接上**（交消费 f + 单变量证据·证明接上后有 f 读它·某变量动它动）· 或 **删除**（+ 考古一句：为何进来的·谁盖的）。
- 🔴 **禁第三态「留着以后用」。**
- **核对已删 3 键**（cachelineBytes/imePresent/deriveIMEPresent·W-1 波 JE4 删的）的考古：确认删对了（无 dangling·grep=0）。

### ④ ISSUE-119 scoping（先出不修）
ForwardElementwise dequant-row ~40/~51 g 轴（de-lottery harvest 引入）→ 出**三栏 scoping**：{改动面 / 架构选项 / byte-exact 风险 + 判决实验}。**先出不修**（本 task 只 scope）。

### ⑤ census 重跑钉新 pin
W3（拔管道·g/c 收敛）+ 本 task（逐条判）之后·**重跑 census**（θ=f(g,c) 三角色·`experiments/active/theta-fgc-census-v2/` 范式）· 钉**新 pin**·让律2「腿二」从「抓到 75 处」变「代码正在通过的检查」。**⚠ 依赖 W3 落地**——若 W3 未完·本子项以当前 HEAD 重跑并标「W3 前基线」。立场文诚实伤 #1 回传新数（37/11 两口径拆开·见交付七）。

## 门
- **零板时**（全静态 grep/机检/weft-opt emit·无 board·无 perf）。
- **判决实验自证**（每个逐值判定带 pinned 实验或机算证据·连提取代码）。
- **0 造数**·「某物不存在」禁 head/窗口截断（用 grep -c/awk 到边界）。
- canon 措辞变更提案入 ISSUES（不自改）。硬冻结四类只登记不执行。

## 交付（首节三项 + 逐条计数）
1. 单侧 f **x/18** · 软默认 **x/7** · 零消费键 **x/9**（逐条·禁打包数）· 37 条 f 全表 · ISSUE-119 三栏 · census 新 pin。
2. 待裁的（判不了的·具名 + 为何待裁）。
3. 本轮问题三分法 {新欠债/本来有被照亮/真阻塞}。

## 账面纪律
- 不 git commit·git add 只纳源/experiments/issues·勿纳 build/worktree·worktree base（`d55f9ab4e`）。
- 返回结构化：三计数（18/7/9 逐条明细）+ 37 f 表 + 摆设名单 + ISSUE-119 三栏 + census 新 pin + 问题三分法。

## 参照
- census：`experiments/active/theta-fgc-census-v2/`（pkg1 θ / pkg2 c / pkg4 f·18 单侧/9 零消费/7 软默认 逐处 file:line 已在档）。
- θ18/ISSUE-002：memory · `.trellis/spec/issues/`。ISSUE-119：`发射器与架构.md`。
