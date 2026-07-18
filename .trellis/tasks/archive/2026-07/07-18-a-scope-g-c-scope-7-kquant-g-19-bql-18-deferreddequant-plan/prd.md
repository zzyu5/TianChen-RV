# PRD · A 线去烘焙 scope 侦察（裁决7·g/c 分家·产出分阶段 plan）

> **权威** = 用户裁决7（2026-07-18·现行法·直接干）+ 《项目行动书 r3》A 线：「把绕层重灾区（**KQuant g 轴 19 / BQL 板轴 18**）接进已有层·**DeferredDequant 是目标形态**·骨架不推倒。g 轴→格式描述符·c 轴→能力事实·抬反误伤证据。」
> **性质** = **只读侦察 + 产出 plan**（研究·不改代码·persist 到 task research/）。产出 = A 线分阶段施工 plan 供 main 派 implement。

## 一、要回答的（scope 侦察）
1. **KQuant g 轴 19 处**：定位 KQuant 发射器里 **按格式（g 轴·q2_K/q3_K/q4_K/q5_K/q6_K…）分支/字面量** 的 19 处（对齐 `papers/share/03-B-DEBAKE-律2-75处违反.md` 的 g 轴 census N_g=37 里属 KQuant 的部分 + `02-差距报告-工程接口.md` 的 g→mᵢ(g) 接口）。逐处：是格式字面量烘焙？能否路由到**格式描述符**（已有层）？
2. **BQL 板轴 18 处**：定位 `RVVToEmitCBlockQuantLinear` 里 **按板/能力（c 轴·VLEN/march/isRVV0p7…）分支/板值烘焙** 的 18 处（对齐 03 的 c 轴 census N_c=38 里属 BQL 的部分）。逐处：能否路由到**能力事实**（`deriveRepackHalfLanes`/能力谓词等已有层）？
3. **DeferredDequant 目标形态**：读现有 DeferredDequant（`papers/share` 称「两轴干净」的正例）——它的 g/c 分家结构长什么样？KQuant/BQL 往它靠的**目标形态**具体是什么？
4. **已有层清单**：格式描述符层 / 能力事实层现状（哪些已存·哪些 KQuant/BQL 可复用而非新建）。**骨架不推倒**=接进已有层·非重写。
5. **抬反误伤证据**：g/c 分家后如何机检**无误伤**（byte-exact 不变 + [K-5] + registry-diff）？哪些是「反误伤」守门证据。

## 二、上岗（只读·persist research/）
- `papers/share/02-差距报告-工程接口.md`（g→mᵢ(g) / c→能力事实 接口件·A1 板轴 closed-form）· `03-B-DEBAKE-律2-75处违反.md`（g 轴 N_g=37 + c 轴 N_c=38 宽口径 census·钉 `0aee07b4`·`git grep` 可复跑）· `04-事实基座` DeferredDequant 正例指针。
- `lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（g 轴 19）· `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`（板轴 18）· DeferredDequant 相关（grep DeferredDequant / 格式描述符 / 能力事实 层）。
- `.trellis/spec/architecture/`（发射与降级·core-invariants·已有层契约）。

## 三、产出（写 task research/·非 spec·非代码）
1. **g 轴 19 逐处表**：{位置 · 现烘焙形态(格式字面量) · 目标(格式描述符路由) · 难度 · byte-exact 风险}。
2. **c 轴 18 逐处表**：{位置 · 现烘焙形态(板值) · 目标(能力事实路由) · 难度 · byte-exact 风险}。
3. **DeferredDequant 目标形态描述**（KQuant/BQL 往它靠的具体结构）。
4. **分阶段 plan**（骨架不推倒·增量·每阶段 byte-exact 可验·从最低风险/最高扇出起）：阶段序 + 每阶段触碰集 + 反误伤守门 + 预期公式占比增量（decisions 从烘焙→公式的格数）。
5. **★与 A′ 上板的关系**（裁决7 也放行 A′ 上板）· 与已有 [D-2a] / `deriveRepackHalfLanes` 能力层的复用点。

## 四、纪律
- **只读侦察**·零代码改·persist research/。**新增 schema 字段须先出映射表给 supervisor 扫一眼**（裁决唯一回门点）——若 plan 涉及新 schema 字段·**单列一节「拟新增 schema 字段 + 映射表」**待 main/supervisor 过目·不擅施工。
- census 数字**机算可复跑**（`git grep` @ `0aee07b4`·对齐 03 的 75 处/N_g 37/N_c 38·禁手数）。
- 骨架不推倒·抬反误伤证据（byte-exact 是硬约束·A 线是「接层」非「重写」）。
