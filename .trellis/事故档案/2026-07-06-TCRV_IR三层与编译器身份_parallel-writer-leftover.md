# TCRV 自我认知：IR 三层 × 编译器身份 × NG-1 精确含义（docs/ 常驻）

> **用途**：统一全员（含 agent）对"我们是什么"的表述；纠正一次措辞事故（"没有人写 body 的 MLIR"≠"没有 MLIR"）。思想层，可送审；与《科研目标总纲 v2》《实验总纲 v1》共用条款号，冲突以总纲为准。

---

## 1. 三层 IR——全部是 tcrv MLIR，区别只在"谁产生它"

| 层 | 是什么 | 谁产生 | Triton 对应 |
|---|---|---|---|
| **L1 声明层（输入）** | `tcrv.quant_contraction` 等高层 op / 带 `exec.target`/`exec.region` 标记的源函数 + 能力事实实例 | **人写**（格式接入者；框架侧） | Triton IR（用户 DSL 降下来的那层） |
| **L2 typed region 层（决策物化层）** | loop-body op（region）+ bricks（integer CORE / fold / decode / grid-gather）+ schedule 属性（lmul-anchor/mbf/fold_structure）+ 能力分臂 | **front-door pass 从 L1 构造**；人永不逐格式手写（lit 中的 L2 样例仅为测试靶 IR / 曳光弹） | **TTGIR**（调度/布局/能力决策物化成 IR 的那层） |
| **L3 发射层** | emission plan → emitc 方言 → C（RVV/IME intrinsics）→ clang → .o | conversion passes | LLVM IR → PTX |

**三条铁律**：
- 每个生成的 kernel 都存在其 L2 MLIR 工件——可打印、可 lit、可 diff（region-vs-monolith 逐字节 diff 比的就是它的产物）、可被 verifier 拒、可被 pass 变换、带 provenance。
- "body 由机制构造" 的准确含义 = **L2 由 pass 产生而非人手打**。这不是"没有 MLIR"，这是"是编译器"的定义本身（Triton 用户也从不手写 TTGIR）。
- 手写 L2 交付生产 = 违背 [G-0]（等价于把知识搬回 C++ 的镜像错误）。

## 2. 谁写什么（MLIR 母语版）

- **格式接入者**：写 L1 实例（几行高层 op/标记）+ 表行（格式参数、能力谓词）。表行 = LLVM **TableGen 模式**的正统实践——表驱动的指令选择从不使 LLVM 变成"指令库"，同理于我们。
- **家族/形状族作者**（一次性）：**扩展方言本身**——新 ODS op（如超块 loop op 的新参数轴、grid-gather brick）+ 构造/合法性/lowering pattern。这是编译器开发，不是 kernel 开发。
- **没有人**写：逐格式的 L2 body、逐格式的 C。

## 3. 编译器身份四问（vs "声明式 kernel 库"的分界）

对任一段编译知识问：①它住在 IR（op/属性/类型/verifier/pattern）还是 C++ 控制流？②verifier 能拒绝其非法形态吗？③pass 能在不碰 C++ 的前提下变换它吗？④provenance 能证明其来源吗？——**四个都是 = 编译器**；kernel 库四个都不是（库 = 逐 (格式×架构) 的成品代码，知识在 C++，加格式=写代码）。G1 的每次 monolith 退役，就是把某段知识的四问答案从"否"翻成"是"。
辅助量尺：`emitc.call_opaque` 计数——每个 opaque call 都是知识仍滞留 C++ 的路标；其趋零曲线与烧减曲线并列。

## 4. [NG-1] 的精确含义："不搜索" ≠ "不测量"

**允许（且正在做）**：有界旋钮枚举（空间由构造约束成小集合）→ 上板全测 → instance-hash 键控记忆（SEL-3）→ `reason=measured` 归因。效果 = "小空间测完记住"。
**禁止**：开放式搜索、学习成本模型、启发式大空间探索（TVM/Ansor/AutoTVM 类）。
**三个理由**：①赛道——搜索式调优是他人深耕十年的领域，进入即接受致命对比；我们的主张是能力键控**裁剪并结构化**了要搜的空间（互补定位，[L-4]）；②归因——每个选择必须是声明事实的可读函数（D-4 reason 链）；搜索器吐出的 config 是黑箱，会腰斩本项目的核心卖点之一；③必要性——空间有界时搜索退化为枚举，引入搜索器是负资产。
**外部插点**：变体空间接口已文档化，外部 tuner 可接；那是别人的工作，不是我们的主张。

## 5. 与 G1 的关系（本文档为何此刻常驻）

G1 = **把知识从 C++ 搬进 L2** 的燃减战役：monolith 退役、front-door 构造、provenance 判强弱义、六态翻转。当前两个诚实缺口即 G1 的重构本体：
- **option-2 桥**：repack 的 L1→L2 正门尚走 quant_contraction 桥，`typed_repack` 未进 emission-plan allowlist——"源函数→对象导出"链差一段；
- **残余 opaque leaf**：部分 brick 内仍有 `call_opaque` 条目，随格式翻转逐个 typed 化。
性能测量（硅验/换装/网格叙事）是 G1 的**证据伴随**，不是替代主线：主线永远是 L2 覆盖的扩大与 C++ 知识的清空。

## 6. 术语速查（汇报统一用语）

| 说法 | 准确含义 |
|---|---|
| "构造"（constructed 强义） | L2 body 由 front-door 从 L1 + 模式库产生，provenance 无 opaque 条目 |
| "声明" | L1 op 实例 + 表行（TableGen 式） |
| "原语/brick" | 方言中的 L2 组件 op（编译器的一部分，非运行库） |
| "模式库" | L2 op 集 + 其构造/合法/lowering patterns（≈ TTGIR 的 op 与 pattern 集） |
| "换键不改条目" | 同一 L2 pattern，不同能力实例 → 不同分臂/旋钮取值 |
| "没有 MLIR"（禁用语） | 任何语境下禁止；正确表述 = "L2 由 pass 构造而非人写" |

---
*本文档由 AI 辅助整理；与总纲冲突处以总纲为准。*
