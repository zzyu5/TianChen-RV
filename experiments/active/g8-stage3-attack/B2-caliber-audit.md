# B2 口径五修 + 欠账 · 案头审计报告（2026-07-15 · 纯案头轮）

> **审计员定位**：Weft-RV 论文证据线案头审计（独立板域·零板时·grep/读源/objdump 放行·禁新计时/e2e）。
> **纪律**：指控成立就认、不成立就驳、部分成立说清；每项 file:line 或命令可溯源；清晰可改直接改、拿不准出计划。
> **触碰集**（本轮已改文件·与其他 agent 不相交）：
> - `experiments/active/result-tables/T8_winloss_gap_ledger.csv`（row 184 note·1 行·byte 级 CRLF 保持·item②）
> - `experiments/active/g8-stage3-attack/old83-to-28-mapping.md`（新建·item⑥）
> - `experiments/active/g8-stage3-attack/B2-caliber-audit.md`（本文件）
> - **未改**：任何 dated 报告（sealed）、canon/`$meta`、spec、schema、recon、board。

---

## 逐项裁决

### ① T3 输局桶（§〇.5 三档）全表覆盖 + 「物理墙」收严复审 —— **部分成立**

**§〇.5 三档定义**（`docs/reports/2026-07-15-G8-主表重铸与对手档位.md:81`）= {指令数内禀 | 内存墙 roofline | 对手结构优势具名}。

**(1a) 全表覆盖核查**：losing/marginal 行大多有桶，但**有出词表外标签 + 非规范措辞**：

| 行 | 现标（file:line） | 问题 | 建议规范桶 |
|---|---|---|---|
| q5_K vec_dot | `opp-immaturity（display 待裁）`（主表:95） | **出词表**（第四类，且方向反：对手弱≠对手结构优势）。rvv 0.843<parity 之因在**我方 emit 近-parity**，不是对手强 | **指令数内禀**（our-emit near-parity）+ 保留"对手=唯一未手调 K-quant"披露 |
| q5_0/q5_1 vec_dot | `内禀 M=1 floor`（主表:89-90） | 非规范措辞（口语化） | 归 **指令数内禀**（M=1 element-broadcast floor·[CASE-MICRO-E2E]） |
| iq4_nl vec_dot(k1) | `codebook-bound(k1 具名-X)`（主表:99） | 出词表口语 | 归 **对手结构优势具名**（hand-vlNNN codebook-gather chip-tuned） |
| iq/tq gemm+vec_dot pending | 无桶（pending） | **可接受**（未实测不强标桶） | 待补测后落桶 |

结论：**三档词表未做到全表唯一归一**——3 处 losing 行用了词表外/口语标签（q5_K/q5x/iq4_nl@k1）。pending 行无桶属正当。

**(1b)「物理墙」收严复审**（收严定义 = 双方【同贴 roofline】才许标 内存墙/物理墙）：存量「物理墙」标签有 over-claim：

| 行 | 现标（file:line） | 收严裁决 | 需改为 |
|---|---|---|---|
| **gelu** | `黄-物理墙（compute-bound tanhf at-wall 0.99×）`（v2:153） | **✗ 不许**：只证我方 at-wall；对手 = f16-LUT **不同数值档**（apples-to-oranges·已判 JUDGMENT-SUSPENDED）→ 双方非同贴同一 roofline | **挂起/数值档不对等**（非物理墙）·同档重比前不得标物理墙 |
| **q6_K@rvv** e2e | `physical-bound(weight-reconstruction floor)` + `missing_pattern(regfile spill)`（T8:182） | **✗ 部分**：我方 0.07× 主因 = first-emission 全展开 **regfile spill**（emit 未熟·NOT at roofline）→ physical-bound 被 emit 缺陷抢跑 | 主桶 **指令数内禀**（spill）·physical-bound 仅次级未证·须先治 spill 再谈墙 |
| **quantize 3 / product_reduce 3** | `预计…perf 判物理墙`（kernel-unit:106-107） | **✗ 不许**：**预判·未实测**（报告自陈"此为预判"） | 标 **pending·未实测**·禁未测先扣物理墙 |
| q5_K@k1 decode | `黄-物理墙(roofline)`（T8:191） | **✓ 可留**：repack-GEVM decode vs stock block-dot decode 双方 memory-bound·同贴带宽墙（收严定义满足）·但建议补"双方同贴 roofline"一句坐实 | 保留·加同贴 roofline 明证 |
| DEQ [COV-1] 6 格 | `物理墙·未入 census·照测-pending`（kernel-unit:26） | **✓ 语义已带 pending**·非硬判 | 保留（已标 pending） |

**处置**：以上为 **dated sealed 报告**（2026-07-15/T8 ledger 判定列），案头审计**不擅改 sealed 判定**（避免与主会话/recon 冲突）→ **列出需改的行**（上二表即交付）。主会话落地时：三档词表收严归一（3 行）+「物理墙」降级（gelu/q6_K/quantize 预判 共 4~5 处）。**建议**：把收严定义写进 §〇.5 canon 注（"内存墙/物理墙 ⟺ 双方同贴同一 roofline 之实测证据；单侧 at-wall 或对手换数值档 = 不许标墙"）——此为 canon 级注，**必问**。

---

### ② VLEN 乱码实验注记补正 —— **成立·已改**

**证据**：`T8_winloss_gap_ledger.csv:184`（q8_0-k1 碎片化证据行）原注记述"裸翻 gate garbage@128 / 我方 emitted vl=8 coherent"，但**未澄清出货真实回退行为**——读者易误读为"我方 beat garbage"。

**核实事实**（源 `experiments/archive/line-c-k1-strike/opponent_map_k1.md` P4 + G5-M1b）：garbage 仅『裸翻强制』（手动去 VLEN gate 强跑 hardcoded-AVL=16 body）产物；shipped 出货有 VLEN guard→VLEN128 走**回退慢标量/generic block-dot**（正确但慢），**不跑 garbage**。行本已标 `n_a(碎片化证据·非 win/loss)`（未升格为 win）——底子诚实。

**已改**（byte 级·保 CRLF·1 行）：注记补入 →
- "garbage 仅『裸翻强制』产物·shipped 有 VLEN guard→回退慢标量/generic block-dot(非跑 garbage)·公平对拼对手 = 出货回退慢标量【非】garbage(不得升格为『beat garbage』win)"
- 定位改 → "手写快路径 VLEN 不可移植:对(回退慢标量·数值正确)与快(仅恰好一个 VLEN 满宽)不可兼得"

`git diff --stat` = 1 insertion/1 deletion（-w 同），edited row 12/12 字段完整。

---

### ③ XFER-1 登记表拆正/负预测两列 + 负预测命中单列计数 —— **成立·出计划（不擅改归档 csv）**

**证据**：XFER-1 登记 = `experiments/archive/line-c-k1-strike/T3p-X_xfer1_prediction_k1.csv`（17 行×3 lever 列 S6/col_outer/vl16·值 ∈ {HOLDS,NULL,NO-OP,LOSS}）。主张 `[XFER-1] 7/7 迁移命中` 见 `docs/reports/2026-07-13-full-refactor-母program-pillar-对账-真欠地图.md:60,96`。

**指控成立**：`7/7 迁移命中` **混算了正预测（HOLDS 兑现）与负预测（NULL/NO-OP/LOSS 兑现）**。csv 里 lever 结果分布：正-transfer 预测（HOLDS）仅 q4_K/q5_K/q2_K 少数格；大多数是 NULL/NO-OP/LOSS（预测"杠杆无效/不产赢"）。把"正确预测了失败/无效"也算作"迁移命中"并以 `7/7` 呈现，**读者会误读为 7 个正向迁移赢**。真相 = 曳光弹（预注册）里**负预测占多数**，命中负预测 ≠ 迁移成功。（对照 T8:182 q6_K 行明写 `XFER-1-hit(净新 scaffold perf 不延伸 K-quant·曳光弹预注册命中)` = 负预测命中。）

**出计划**（归档 csv + dated 报告·不擅改）：
1. XFER-1 登记表加两列拆分：`pred_polarity`（正=预测 transfer-HOLDS / 负=预测 NULL/NO-OP/LOSS）+ `hit`（预测=实测? True/False）。
2. 头条改双数：**正预测命中 X/N₊**（真迁移赢）· **负预测命中 Y/N₋**（正确预判无效·单列计数·不与正预测相加）。
3. 报告 `:60/:96` 的 `[XFER-1] 7/7` 改为拆分呈现（禁再以单一 `7/7` 暗示 7 个正向迁移赢）。
4. 复算脚本存在则加 assert：`正命中 + 负命中` 不得合并作"迁移命中"头条。

---

### ④ schema 文档如实分层（shipped=provides/implies/conflicts·其余标设计中/未落地）—— **大体不成立（已分层·仅可加显式标记）**

**证据**：`schema/capability.schema.v1.json`。
- `$meta.note`（:6）**已明文分层**：`provenance/trust/subclass、closed kind enum、namespaced params、unified operand roles、serializable plugin signature 均 grep=0 in code today；divergence 是 tracked conformance gap`。
- 逐字段已带落地标注：`provenance`(:49)/`trust`(:56)/`subclass`(:29) 均标 `Declared-with-no-producer in v1`。
- **relations 三型**（provides/implies/conflicts·:82-96）**未列入 grep=0**，且代码实证 **shipped**：`grep -rEn 'provides|implies|conflicts' lib/ include/` = **57 命中**（`lib/Plugin/*ExtensionPlugin.cpp`/`RVVCapabilityProfile.cpp`/`RVVTargetSupportBundle.cpp` 等·PluginCapability 关系解析真在跑）。与 memory「SubtargetFeature.Implies 已建模 layering」一致。

**裁决**：指控（"schema 未如实分层"）**大体不成立**——分层已在 `$meta.note` + 逐字段注做到：**shipped = provides/implies/conflicts（不在 grep=0 名单·代码 57 命中）；其余六项 grep=0 = 设计中/未落地**。**唯一残留**：relations 块本身**未带显式 `"shipped": true` 正标记**（只靠"不在 grep=0 名单"反推）。

**出计划（可选加固·canon 级·必问）**：在 `item_3_relation_types` 加一句 `"landed": true, "note": "provides/implies/conflicts 已在 lib/Plugin 落地(57 refs)·区别于本 schema 其余 declared-with-no-producer 字段"`。因触碰 `$meta`-邻近的 in-shape 声明 = **canon 级·必问**，本轮不擅改。

---

### ⑤ C2 口径清理（统一"两 regime 成本分解"·清除 law/律 与 ≥3 点曲线声明）—— **部分成立（≥3 曲线已无·"律"与 canon 纠缠→出计划）**

**证据**：`docs/method/C2_marginal_cost_ledger.md` + `docs/reports/2026-07-12-C2-LED-2-边际曲线-三家族点.md` + `docs/reports/2026-07-13-C2-zvfh-轨一粒度点.md`。

**已满足部分（指控不成立）**：
- **≥3 点曲线声明** —— **无一处 claim**。所有 C2 doc 都**显式反声明**："两点≠单调递减曲线"（ledger:107）、"仍 <3·缺≥1点"（ledger:6,107·zvfh:57·LED-2:51）。**这部分已诚实到位**。
- **两 regime 成本分解** —— **已存在**：`LED-2:71-72`「C2 曲线正确刻画不是随家族序号单调递减，而是**两 regime**：独立家族付真 emitter 成本 / 集成子扩展近零」；`zvfh:12`「轨一≠轨二·两条不同曲线」。框架已在。

**残留部分（指控成立）**：
1. **"律/law" 措辞**仍在（把 2 点观察称"律"）：ledger:98`独立家族的边际成本律`、ledger:72/12`成本律主张②`、ledger:125`双轨律`。C2 只有 **2 个 onboarding 点**，"律"言过其实（对照 C3′ 构造谱有 8+ 点，其"规律"更站得住·**item⑤ 只针对 C2·不动 C3′ 的 ledger:56/78 规律**）。
2. **"曲线/边际曲线"命名**：LED-2 报告标题/§2.1「曲线表」称 2 点为"曲线"（虽正文反声明·命名仍误导）。

**出计划（不擅改·canon 纠缠）**：
- 把 C2 的"边际成本**律**"降为"边际成本**两 regime 分解/刻画**"（去"律"字），"边际曲线"降为"**两 regime 成本点**"。
- **⚠ 必问红线**：`双轨律` 已被 `[C2-1]（2026-07-11 用户裁）` **codify 为 canon**（ledger:124-125 明记"用户裁"）。降"律"= 改既有 canon 条文措辞 = **canon 级·必问**，案头不擅改。→ 本项统一改由主会话在用户裁下批量执行（把 [C2-1]"双轨律"→"[C2-1] 双轨成本分解"，同批扫 ledger:72/98/12 + LED-2 标题/§2.1）。

---

### ⑥【欠账】旧「83→28 逐格排除名单」→ 新 rowclue 工件 + 映射说明 —— **成立·已交付**

**已交付**：`experiments/active/g8-stage3-attack/old83-to-28-mapping.md`（一页·数据零改）。
- 拆清旧数（83=perf-covered e2e 分母 / 28=gemm roster / 46=matmul kernel / 20-28·29-43·30-28·kernel-sym12 = board-pt 作废）。
- 逐类映射旧"排除格"→新单分母字典落点（`T3_master_rowclue.txt` 档+状态）：**唯一仍排除 = N/A-hw 3 + q1_0 1**；旧"排除名单"其余全部**收进 rvv85/k188 分母挂 pending**（单分母制取代逐格排除的核心）。
- 旧头条→新头条对应表（perf-covered 9/83 + certified 84/91 独立赛道保留·board-pt 数作废）。
- 校验：旧排除名单每格新字典唯一落点·无悬空。

---

## 汇总

| 项 | 裁决 | 处置 |
|---|---|---|
| ① 输局桶覆盖 + 物理墙收严 | 部分成立 | 列出需改行（3 词表外标签 + 4~5 处物理墙 over-claim）·§〇.5 收严 canon 注=必问 |
| ② VLEN 乱码注记 | 成立 | **已改** T8:184（byte 级·1 行·保 CRLF） |
| ③ XFER-1 拆正/负预测 | 成立 | 出计划（拆两列+负预测单列计数·改 `7/7` 头条·归档 csv 不擅改） |
| ④ schema 分层 | 大体不成立 | 已分层（$meta grep=0 + 逐字段注·relations shipped 57 refs）·可选加显式 landed 标记=必问 |
| ⑤ C2 律/曲线清理 | 部分成立 | ≥3 曲线已无声明·两 regime 已在；"律"字与 [C2-1] canon 纠缠→出计划·必问批量改 |
| ⑥ 83→28 映射 | 成立 | **已交付** old83-to-28-mapping.md |

**共识底线**：真正的直接改仅 ②（1 行·低风险·增诚实）；①③④⑤ 或涉 dated sealed 报告、或涉 canon/`$meta`/[C2-1] 用户裁条文 → 按权限卡"canon 级=必问"出计划，不擅改。⑥ 为新建工件（欠账·无 canon 触碰）。

*证据源：主表报告 `docs/reports/2026-07-15-G8-主表重铸与对手档位.md` · v2 `docs/reports/2026-07-15-G8-全景报告-v2.md` · kernel-unit `docs/reports/2026-07-15-G8-全景对账报告-kernel-unit.md` · `experiments/active/result-tables/{T3_master_rowclue.txt,T3_master_rebuild.csv,T8_winloss_gap_ledger.csv}` · `experiments/archive/line-c-k1-strike/T3p-X_xfer1_prediction_k1.csv` · `schema/capability.schema.v1.json` · `docs/method/C2_marginal_cost_ledger.md` · grep `lib/ include/` relations=57 refs。*
