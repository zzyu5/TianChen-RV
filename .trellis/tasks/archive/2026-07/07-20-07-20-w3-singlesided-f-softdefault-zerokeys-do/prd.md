# PRD — W3 A线逐条判并做: 18单侧f补输入/删 + 软默认7登记 + 零消费键9删（★做·非只判·禁打包）

## 权威 = r5.1 goal
A线「18 单侧 f / 软默认 7 / 零消费键 9 **逐条判·禁打包**」。census2 已分类（`experiments/active/census2-singlesided-f-zeroconsume-judgment.md`）·**本役 = 把判定落成真改动 + judgment lit**（判定不是终点·做才是）。逐值判定与判决实验同格计分·禁登记灌水。

## 做（逐条·每条独立 byte-exact + judgment lit·禁批发）
### A. 18 单侧 f 中的 2 真 fake（census2 已判·现做）
1. **F7 `(void)shape` dead param** → 删参·+ lit 证「feasible set 随 shape 变时不变」（判决=删参后 5 格 emit byte-exact·shape 真无关）。
2. **F25 `selectIntegerCoreLMUL` baked blockLen=32**（load-bearing·flips coreLMUL·现被 3 前门 `dimSize==32` gate invariant-guarded）→ **加 regression lit**（记录 baked-g 承重 + gate guard·= 「blockLen 变→coreLMUL 变」的 judgment·证它真承重非摆设）·**非 32 几何入前不 thread**（census2 判「现恒真」）。
- 13 legit 单侧 **不动**（标 legit·别误伤·census2 已具名 F5/F6/F12/F19/F24/F26-F35）。3 零输入 OK 不动。

### B. 软默认 7 处（θ15-21·KQuant/Ternary/flat value_or）逐处登记
逐处写成正式登记（为什么此处软默认是设计非欠账）·**θ18「真实测过路径承重」优先核**（ISSUE-002 同族活口）。写不成的排 fail-closed 改造候选。**不许批发豁免·一处一理由。**（本役 = 登记·实际 fail-closed 改造 gated·不做。）

### C. 零消费键 9 删（census2 判 delete·现做）
删 **8 个 provider-property provenance dead-stamp**（`RVVCapabilityProfile.cpp:770-805`·clang/cmake version·compile_run selected_march/mabi/source_sha256/binary_sha256·march:value/mabi:value·`git grep`=0 消费者）+ 「为什么当初探测」一句考古注。⚠ `source_sha256`/`binary_sha256` 带 **I8-provenance caveat**（证据线·可能须留·逐个核 grep 消费者·真 0 才删·带 caveat 的留+注明）。（cachelineBytes/imePresent/deriveIMEPresent 已删@HEAD·不重做。）

## 门（禁打包·禁灌水）
- 每条独立: F7 删参 byte-exact + lit / F25 regression lit PASS(blockLen→coreLMUL judgment) / 8 键删后 `git grep`=0 且 build+全套绿 / 软默认 7 逐处登记(一处一理由) · **禁批发**（117 教训）· byte-exact(F7/删键不改任何 emit) · 全套 lit 无 NEW 失败(基线 3 pre-existing) · 无 inline-asm · 未 git commit。
- 🔴 禁碰: 拔管道 deriveMinimumVLEN(W1)·repack width 选择器(W2)·B线(W4/W5)。**RVVCapabilityProfile 与 W1 可能重叠·只删 provenance dead-stamp 段(:770-805)·不碰 resolve/materialize/deriveMinimumVLEN**·worktree 隔离·串行集成。

## 汇报（首节: 单侧f x/18 + 零消费键 x/9 + 软默认 x/7）
F7 删+lit / F25 lit 结果 + 8 键删(哪些真删·哪些带 caveat 留) + 软默认 7 逐处登记 + byte-exact + 全套失败数(应仍3)。禁新建分析文档·结论进 final message。
