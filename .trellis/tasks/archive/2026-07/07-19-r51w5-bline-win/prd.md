# PRD — W5 B 线真赢（门 0.8·vs 部署≥0.8 就是赢·修缺口不撤）

## 哲学锁（用户裁·覆盖一切成色兜底）
**① vs 部署路径比值 ≥0.8 = 赢 = 入账·不问对手强弱**（autovec/病理核/坏路径都算）。成色注记（opp-immaturity 等）**只为论文写作保留·不作为「算不算赢」判据**。② 负结果非贡献。③ **修缺口不撤·遇挫不退·输了修了再打·不停在记边界。** 衡量你的**不是**手调档 8/24·9/29 那分母·**是「vs 部署路径 ≥0.8 的格数」**——目标是这个数往上·门 0/6 往上。

## 按优先序（修缺口→扩真赢面）

### ① dequant 真向量发射器（最大收割·PR-31 已立项·复用 C4a 参数化）
- **病**：18 格式 × ~36 板格·现在向量内建**恒 2 条全是「设置向量长度」·真运算=0**（codegen 抽签·host autovec）。
- **做**：做**真发射器**把 36 板格从「codegen 抽签」变**我方控制**（owned 真向量 emit·复用 C4a 参数化·扩 dequant de-lottery 已证模式到剩余格）。
- **入账门**：vs autovec 部署 **≥0.8 即入账**（不问 autovec 强弱·门 0.8）。byte-exact 先于计时（mism=0·3-arm）·板测 compiler-symmetric 2-seed·cold≥0.8 入账。
- **★account 纪律**：master 入账前 `git stash` recon 核 baseline（收割 agent 屡高报真地盘·逐格 diff·真地盘=仅 baseline<0.8→≥0.8 的格）。

### ② objdump 系统分拣（真仗侦察·族级杠杆·一杆多格）
- 每格出「**对手结构 vs 我方结构**」diff·具名剩余缺哪条机制 → **族级杠杆清单**（一杆多格）。
- 上一波只做 K-quant decode 8 格·**铺开**（全 vec_dot/gemm 输格·找**大扇出杠杆**·修缺口→全族翻）。
- 出杠杆清单：{杠杆 / 扇几格 / objdump 证据}·扇出≥3 的排前。

### ③ q4_K 最后几点（分 regime·别沉底）
- **decode = 带宽墙**（微架构·MEMORY-BOUND M=1）· **prefill = compiler-behavior（可翻盘·别沉底）**——措辞**分开**·🔴 别把 prefill 可翻盘判据埋进「微架构墙」。
- q4_K prefill 若 compiler-behavior 可翻盘 → 发射压低一层（编译器层动作·合法）攻。🔴 **严禁内联汇编/钉死调度绕 clang**（存得数+违律1+砸论文地基·发射压低≠手排）。

### iq2 扩面（W2 修完 ISSUE-120 后）
W2 修完 iq2_xs/iq2_s VLEN256 → 把 K-quant/宽化杠杆往 iq2 族延·vs 部署≥0.8 入账。**依赖 W2**（若 W2 未完·本项标 blocked-on-W2）。

## 三闸（B 线不欠 A 线债·每提交自查）
① **三 grep 体检**：裸格式字面量 / 板值+value_or / march 解析·三计数**相对基线只减不增**（新代码出生即干净）。
② **新 θ 出生即四类归档**（归 c 当场写为何不能是 a/b）。
③ **B 线缺字段走字段需求单给 A 线**（禁就地加字段或焊常量·出「字段需求单」给 W3/A 线）。

## 门（byte-exact 先于计时·硬）
- **byte-exact 先于任何计时**（ZERO-MODEL·mism=0·3-arm anti-hollow·CORPUS）。门未过=不测 perf。
- 板测 compiler-symmetric（我方 clang·对手 clang18-symmetric 路）·2-seed（稳则5）。**vs 部署路径**（as-shipped·非 generic）。
- headline flip（<0.8→≥0.8）主会话独立 check 再入账·正负对称。

## 交付（首节三项 + B 线数）
1. **vs 部署≥0.8 的格数 Δ**（收割几格·各 pinned byte-exact+cold）· 三 grep 体检 Δ（只减）· 新 θ 归类 · 字段需求单流转数。
2. 未赢的输格·三步收据（objdump 对手→定杠杆→板测证伪·墙型**分微架构 vs 编译器行为**）· 族级杠杆清单。
3. 论文侧：vs-部署-≥0.8 格数（门 0/6 的分子）· 大扇出杠杆。

## 账面纪律
- 不 git commit·git add 只纳源+kernel+harness·勿纳 build/worktree·worktree base（`d55f9ab4e`）。
- **0 造数**·数走 bench·对手按部署事实定。sealed 核 md5 变须报。
- 返回结构化：每格 verdict（vs 部署 cold·≥0.8?）·byte-exact mism·三 grep Δ·新 θ 归类·字段需求单·族级杠杆清单·三步收据（输格）。

## 参照
- dequant de-lottery 已证模式：commit `ababfada9`/`d24bf2038`/`0b53efdc9`（K-quant/tiny-codebook/ternary owned emit）· CLANG_WORLD dict `recon_master_rebuild.py`。
- PR-31/C4a：`docs/PENDING_RULINGS.md` · 真向量 leaf `RVVToEmitCGridCodebook.cpp`/`RVVToEmitCForwardElementwise.cpp`。
- objdump 范式：`archive/2026-07/07-19-block2-objdump-recon-clusterA/research/clusterA-wall-type-map.md`。
- q4_K：ISSUE-109（vec_dot 脾气墙·decode）· ISSUE-014（K-quant decode 承重）· B 线首波 `experiments/active/r5.1-je/bline/`。
