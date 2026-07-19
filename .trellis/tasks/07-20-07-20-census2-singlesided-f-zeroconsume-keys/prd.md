# PRD — census2 案头判定: 18 单侧 f 逐判 + 11 零消费键二选一（增加公式 breadth·只读 scoping-first·禁批发）

## 权威 & 缘起（用户「增加公式据 census2·都裁决同意」）
行动书 §3.3（18 单侧 f 逐条判·禁打包·117 教训迁移）+ §3.6（c 侧零消费键二选一·禁第三态）。census v2 正本 = `experiments/active/theta-fgc-census-v2/00-census-final.md`（pkg2/pkg4）。**这是「增加公式」的 breadth 判定**：判每条 f「该不该吃两侧」、每个键「该不该留」。**本役 = scoping-first**（出判定表 + 每条 actionable 的 plan/judgment 设计）·实际编辑（补输入/删）后续按判定执行——**避免与将来 phase-2 碰选择器文件冲突**。

## 做（纯只读枚举 + 判定·零码改·出判定表）
### A. 18 单侧 f 逐判（§3.3·census pkg4：仅读 c=10 / 仅读 g=5 / 零输入=3）
逐条（F# + 签名 file:line + 现状）分三类·**禁批发**：
1. **合法单侧**（Tier-2 纯 c 派生如 `deriveMinimumVLEN`/`deriveRVVVersion`——本职解析 c·单侧对）：标「合法单侧+理由」·**别误伤**。
2. **假单侧**（须补输入）：`F7 shape 参数 `(void)` 丢弃`——查遗留 vs 有意·legality 该看 shape 就补/真不需要就删参注明；`F25 签名只吃 c·g baked`——判 baked 的 g 常量对现格式集是否恒真·将来新格式会破就把 g 提为参数。**每条假单侧给 judgment 设计**（改被 void/baked 的输入→θ 必须随之变 lit·验收非「参数加上了」而是「输出真随之变」）。
3. **零输入 3 条**（F4 空测量表 / F21b 空 / F37 最小合法常量）：逐条写「为什么零输入对」·写不出→补输入候选。
**交付 = 37 条 f 输入签名表**（或至少 18 单侧）·每条标 {双侧 / 合法单侧+理由 / 假单侧→补输入 plan+judgment 设计 / 待裁}。

### B. 11 零消费键二选一（§3.6·census pkg2：9 provider property + 2 probe cachelineBytes/imePresent + deriveIMEPresent 零调用者）
每键二选一（**禁第三态「留着以后用」**=欠账换名字）：
- **接上**：交出「消费它的 f + 该 f 输出确实依赖它的单变量证据」的 plan（如 T1 vlenb→VLEN 那样有真消费者）。
- **删除**：探测代码一并删 + 「为什么当初探测了」一句考古。
**判据**：只有存在 GENUINE 待接消费者才「接上」（否则=造假旋钮[K-10]·T1 是干净案例=VLEN 有真消费者用错源）；无真消费者→**删**（诚实）。
**交付 = 11 键判定表**·每键标 {接上 plan / 删除+考古}·标哪些是清晰可删的 dead code（cachelineBytes/imePresent/deriveIMEPresent 若零消费者）。

## 门
- **0 造数**（只读 census + 机核签名·禁手抄）· 每条判定带 file:line/签名依据 · **禁批发**（一条一理由·117 教训）· 假单侧判定必附 judgment 设计（改输入→θ 变）· 零消费键禁第三态 · 未改任何源 · 未 git commit。
- 🔴 纯只读·不动码（补输入/删的实际编辑后续按判定执行·本役只出 plan）。禁碰在飞 agent 文件（当前无·天然安全）。

## 交付
写 `experiments/active/census2-singlesided-f-zeroconsume-judgment.md`（A 表 18 单侧 f + B 表 11 键）。final message 给：单侧 f 判定 x/18（合法几/假单侧几/待裁几）+ 零消费键 x/11（接上几/删除几·哪些清晰 dead code）+ 最高价值的 3 条 actionable（补输入 or 删）。
