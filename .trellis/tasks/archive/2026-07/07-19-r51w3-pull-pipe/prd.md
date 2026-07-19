# PRD — W3 真拔管道（A 线核心债·接线不重写·真重构非演示）

## 哲学锁 + 纠错（用户裁·本 task 的存在理由）
上一波 JE1「翻 -march 看 strip 跟着翻」**恰恰证明旁路还活**（strip 随 march 翻 = 消费点现场重解析 march·不是管道通）。🔴 **禁再用那个假绿形态当验收。** 本 task 真拔线·不标绿。

## 病（census 实测·必先复核为真）
- 承重决策（VLEN→strip 宽 / LMUL）的输入**散落 ~10 个 pass**·各自现场 `deriveMinimumVLEN(-march 串)` 重解析（je1-pipe/README.md B2 列 C1-C10）。
- 探测→provider op **生产驱动零挂接**（`RVVProbedCapabilityAxesMaterialization` 唯一调用者是测试）。
- schema **7 个 typed 参数命名空间代码零落**（census pkg2 口径A）。
- 四表面刻意分歧。**类型化能力层是空转招牌·承重的水从 march 旁路流。**

## 做（壳/桥·6 条已通消费路全留·不重写算法）
1. **挂生产驱动**：给探测→provider op（`RVVProbedCapabilityAxesMaterialization`）挂**真生产驱动**（不再只测试调用）·使 provider op 在生产 pipeline 里被物化一次·携带 capability 事实（vlen/version/lmul-allow 等）为 **in-IR 类型化属性**。
2. **~10 消费点改读 provider op**：C1-C10（`Schedule/RVVRepackStripWidthMaterialization` / `Schedule/RVVScheduleDescriptorRegistry` / `FrontDoor/RVVLowerQuantContraction` / 5 个 SourceFrontDoor / `FrontDoor/RVVMonolithicBlockDotSourceFrontDoor` 等）从 `deriveMinimumVLEN(march)` **改成读 in-IR provider op 的类型化 VLEN 属性**。**6 条已通消费路的行为全留**（byte-exact·只换输入来源）。

## ★判决实验（收紧·伪造不了·本 task 唯一验收）
**能力实例文件写 vlen=256·march 仍写 zvl128b（故意打架）**·编译·看 strip / LMUL 跟谁走：
- **跟能力文件（256·strip=16）走 = 管道真通 = GREEN。**
- **跟 march（128·strip=8）走 = 旁路还活 = 未完成（不许标绿）。**
- 构造：注入 provider op 携带 vlen=256·同时 march=rv64gcv_zvl256b 设为... **不·反过来**：provider vlen=256·march 参数 zvl128b（128）·若 strip=16 跟 provider·则消费点真读了 provider（管道通）；若 strip=8 跟 march·则还在 re-parse march（旁路）。记 pinned（provider-vlen / march-token / strip-θ）。

## 三条硬门（全绿才算成·缺一未完成）
① **能力文件 ⊥ march 判决实验**：承重 θ 跟能力文件（上述·strip 跟 provider vlen 非 march）。
② **`git grep deriveMinimumVLEN` 生产路径收敛到探测层一处**：**「funnel 过同一函数」≠「只解析一次」**——10 个消费点**真收敛**（不再各自调 deriveMinimumVLEN·改读 provider op）。🔴 **禁判「funnel 过权威·不动」放走**（那是上一波的错）。记 before/after：生产路径 `deriveMinimumVLEN` 调用点数应从 ~10 降到探测层内极少数（1 生产者）。
③ **t1d 注记删得掉**：`t1d` 里「march= is a PASS OPTION, not an in-IR fact」的注记——拔线后 march 真成 in-IR fact（provider op）·该注记删得掉（找到并删·或报为何删不掉=未完成）。

## 同批（W3 内·禁打包顶替·各自判决实验）
- **选择器闭式化**：动 **[GAP-P1] 那个** static_order/measured 兜底的 **family / 累加器 LMUL 选择器**（`selectRepackAccumulatorLMUL`·θ_family=f(VLEN)·**m2@128≡m1@256**·K1 反证已钉必要性）——**不是上一波演示的 reduction 路 `selectIntegerCoreLMUL`**。落**能力翻转→改判记录带 pin**。判决实验 = 翻能力文件 VLEN·family/宽度**必须翻**（当前 [GAP-P1] 钉死 mf2·闭式化 = 让它随 VLEN 算·mf2@128 / m1@256·带 K1 反证的必要性论证）。⚠ [GAP-P1] 是 canon·若闭式化触碰 canon 措辞·提案入 ISSUES 待裁（agent 不自改 canon）·但**闭式化本身（读 vreg_count/VLEN 算 widest-legal）是施工**·先做能做的。
- **vreg_count 升格能力事实**：θ1 选择器 + budget 参数改读 `vreg_count`（现 census：schema 声明但代码零落·`kVectorRegisterBudget=32` 硬编码）→ 升格为读能力事实 vreg_count。**+ 分数 LMUL 存在性升独立布尔字段**。这是**累加器最后一试的前置**。

## 门（byte-exact 先于任何）
- **6 条已通消费路 byte-exact**（拔线只换输入来源·发出的核逐字节不变·除非判决实验的 conflicting 输入·那正是要它变）。regen-diff 对正常输入=0。
- build + lit 绿（953/956 基线）。三闸①三 grep 只减不增。新 θ（若加 vreg_count 读）出生即四类归档 + 登记。
- ★**新增 schema 字段**（vreg_count 升格 / 分数 LMUL 布尔）= 出映射表·**但本令已授权拔管道**（用户裁「vreg_count 升格能力事实」）·施工；若涉 canon [GAP-P1] 措辞变更·提案入 ISSUES。

## 交付（首节三项）
1. 拔管道完成度：判决实验（能力⊥march·strip 跟能力）GREEN/未过 + 三硬门各状态（判决/收敛 grep before-after/t1d 注记）+ 选择器闭式化改判记录 pin + vreg_count 升格。
2. 未拔尽的消费点（若 ~10 中某几个卡）· 具名 + 卡在什么。
3. 论文侧：类型化能力层从「空转招牌」变「承重真读」· 律2 腿二数据。

## 账面纪律
- 不 git commit·git add 只纳源·勿纳 build/worktree·worktree base（`d55f9ab4e`）。改共享文件（capability/front-door/schedule/selector）多·**报清单供主会话合并**（本 task 是最大改动面·与 W2/W5 emitter 尽量隔离·碰同文件报冲突）。
- **0 造数·判决实验自证·独立复核跑判决实验不看 diff**。机算断言连提取代码。
- 返回结构化：判决实验 pinned（provider-vlen/march/strip-θ）·收敛 grep before/after（deriveMinimumVLEN 生产调用点数）·t1d 注记 diff·选择器闭式化 pin·vreg_count 升格 diff·6 消费路 byte-exact 证·改的文件清单。

## 参照
- 消费点 C1-C10：`experiments/active/r5.1-je/je1-pipe/README.md` B2 表。
- 探测层：`lib/Plugin/RVV/Schedule/RVVProbedCapabilityAxesMaterialization.cpp` · `lib/Plugin/RVV/RVVCapabilityProfile.cpp`（derive* 权威族·hasRVVVectorHint 已导出）。
- [GAP-P1] 选择器：`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp:1255-1304 selectRepackAccumulatorLMUL`。
- vreg budget：`RVVLowerQuantContraction.cpp:1292 kVectorRegisterBudget=32` · schema `capability.schema.v1.json`。
- census：`experiments/active/theta-fgc-census-v2/`（pkg2 c 进料口·pkg4 f 存量）。
