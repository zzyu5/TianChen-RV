# A 线柱二 · c 轴 f 可写性清单（workflow wf_f2cd1666-243·2026-07-19）

> **柱二真度量** = 决策 formula-expressible（公式占比）·非 fail-closed 哨兵安装。workflow 3 分类 agent 成功·综合由 main 做（综合 agent schema 中文 key 报 400 失败）。

## ★核心裁定：公式多已存在·非"分子 0 待写"

| 决策 | 现状结构 | verdict | 扇出 | 证据 |
|---|---|---|---|---|
| **c 轴 core_lmul**（18 哨兵源）| 前门单函数 `selectRepackAccumulatorLMUL`(RVVLowerQuantContraction.cpp:1285-1304) 三层: ①correctness fork isRVV0p7=**闭式 f(march)** ②measured gate(SEL-3·今空 nullopt) ③冷启动 mf2=**[GAP-P1] IRON RULE 刻意钉**(非[SEL-1]widest-legal)。部署态坍缩 = `core_lmul=isRVV0p7?m1:mf2`=闭式 f(march) | **MIXED**（已闭式·widest-legal 禁）| 18（schema）| 决策源 1285-1304·stamp 无条件 1818·18 哨兵 grep 精确 |
| **宽化收益选择器** | half_lanes=`deriveRepackHalfLanes=min(VLEN/16,16)` 已闭式已消费·benefit-prediction 无 selection consumer（strip-width per-board **singleton**·无 A/B·ISSUE-035 honest-null）| **D** | 0 | RVVRepackStripWidthMaterialization.cpp:78·ISSUE-035 |
| **家族 θ** | ①家族轴(矩阵vs向量)=闭式 cost 层(crossoverM=macM=f(VLEN)·SEL-1-T5/T5c 已消常量盲) ②LMUL-fill=闭式 f(vlenBits,sew,blockLen)·**发 prior** ③SP4-tiling/loop-order=闭式·发 prior/measured·生产路零 static_order ④m1-vs-mf2=[GAP-P1] measured | **MIXED**（多已闭式·发 prior）| 0（归因）| IMEExtensionPlugin.cpp:373-383·chooseFillOptimalLMUL·[SEL-1]canon |

## ★柱二真 gap（都不是"写新闭式公式"）
1. **(a) c 轴 ISSUE-113**：integer_core_lmul optional→required（schema 退休 18 哨兵）·byte-exact（前门恒 stamp 已证 totality·stamp 值逐位不变）·**触发条件 A2/A3 接层收口现已满足**（阶段0-3 落地·fixture 更新·fail-closed 硬失败 a860 已证）。
2. **(b) 归因 relabel**：家族轴/LMUL-fill/SP4 已闭式但 EXEC reason 部分仍 static_order（能力盲）·relabel→prior 使能力派生 provenance 可见·**碰 F4 门 V3 CONSISTENCY（reason 从 feasibleCount 反推）+ V5 M1-ENUM 覆盖**（账本级·kernel byte-exact·只改归因 JSONL + 门）。

## 🔴 [GAP-P1] 锁（不自决·禁写）
用户假设 `core_lmul=f(VLEN,块宽)=widest-legal` 被 **[GAP-P1] IRON RULE 明禁**（widen-to-m1 falsified 2×: micro win washes / regfile spill·deployed=mf2 until board-measured）。写 = 破 byte-exact + 违 canon。**c 轴 perf 公式被 canon 刻意锁死·是"measured 决策"非"公式-writable 决策"。**

## 已进货能力事实（一项不缺）
march/isRVV0p7·minVLEN·halfLanes(deriveRepackHalfLanes)·块宽 kWeightInterleave=16·寄存器预算 kVectorRegisterBudget=32 + getRVVLMULRegisterFootprint。c 轴哨兵之源所需一项不缺——但"缺"的是 SEL-3 measured 板测数（恒 nullopt·刻意非能力事实·补数=STAGE THREE 板测非本柱补公式）。

---

## ★代码层定论（2026-07-19·grep 精确·非摘要）——c 轴哨兵处「没有公式可写」

`RVVToEmitCBlockQuantLinear.cpp` 的 c 轴 fail-closed getter 是**同一模式 ×8 处**（行 2205/2310/2427/2617/2755/2984/3367/3475/3599）：

```cpp
// Fail-closed capability-fact read (was value_or("mf2") board default): the
// front door ALWAYS stamps integer_core_lmul on every wired repack leaf.
if (!loopBody.getIntegerCoreLmul())
    return rewriter.notifyMatchFailure(loopBody, "repack integer core requires an explicit integer_core_lmul ...");
llvm::StringRef coreLmul = *loopBody.getIntegerCoreLmul();
```

**全是消费者·零计算**：它们不 compute 公式·只 read 前门 stamp 的上游闭式决策（`selectRepackAccumulatorLMUL` RVVLowerQuantContraction.cpp:1285-1304）。**「换成闭式公式」在此处无对象**——公式已存在于上游。

### 退休三路（全非「写 inline 公式」）
| 路 | 机制 | 状态 | 阻塞 |
|---|---|---|---|
| ① schema-required | integer_core_lmul `OptionalAttr→required`·前门恒 stamp（a860 fail-closed + stamp totality 已证 byte-exact-safe）→ 哨兵 unreachable 可删 | **用户「c轴required=选1 暂不升」** | 需用户反转 选1（schema 接口变更·回门） |
| ② STAGE-THREE 板测 | 板测 repack accumulator LMUL 配对 A/B（m1 vs mf2 vs m2）→ 填 SEL-3 measured gate ②→ reason=measured 替 cold-start ③ | SEL-3 恒 nullopt | 测量通道 ISSUE-090/091 待裁 + [GAP-P1] 律 |
| ③ widest-legal inline | `core_lmul=widest_legal(VLEN,块宽)` | 🔴 **[GAP-P1] IRON RULE 明禁**（widen-to-m1 falsified 2×·micro win washes/regfile spill） | canon 锁·写=破 byte-exact+违律 |

### 判据归属：c 轴 = **measured-decision 墙·非公式墙**
[GAP-P1] 明言「c 轴 perf 公式被 canon 刻意锁死·是 measured 决策非公式-writable 决策」。依 `公式墙 vs 脾气墙` 判据：c 轴依赖的量 = **板测性能数**（非可读闭式能力量·非 clang 不可读量）→ 属第三类「measured 决策」·移动它靠**板测填 SEL-3**（STAGE THREE·测量通道裁决前置）·**不靠写公式**。

> **★为凑「退休一个哨兵」写 inline 公式 = 存得数 + 破 byte-exact + 违 [GAP-P1]**——严禁。柱二 c 轴曲线「卡在 0」= canon 锁 + 用户 选1暂缓 + 测量通道待裁 的**合法叠加**·非 agent 不作为。移动它 = 用户裁 ①（反转选1）或 ②（放行测量通道跑 SEL-3 A/B）——**入 ISSUE-117 待裁·不自决**。
