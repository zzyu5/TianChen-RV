# PRD · K线 · 补充令二 §三 量税实验（前置·可能推翻 ISSUE-014 第②腿）

> **权威** = 补充令二（2026-07-18）§三·§五.15（选择键值 per-format 板测定·直觉投影不可信）。
> **性质** = 板测实验（半天·高信息·前置于数值松绑落地）。**性能第一。**

## 一、两条命令

1. **objdump 对手数 FMA 形态**：反汇编部署对手 K-quant vec_dot `_vl128`（q4_K/q6_K）+ block-dot decode·
   数 `vfmacc`/`vfmadd`/`vfnmacc`/`vfwmacc` 等 FMA 指令 + 重排痕迹 —— **看对手用不用 FMA、用不用重排**。
2. **同一 kernel 编两份板上跑**：`-ffp-contract=off` + 串行折叠 **vs** `-ffp-contract=on` + K 路累加器。**税 = 两者 cold 差**。
   （挑一个 fold-bound 候选格·如 q4_K vec_dot 或一个浮点 reduction 格）。

## 二、★ISSUE-014 第②腿裁断（本实验核心产出）

ISSUE-014 判 fold@M=1「真但空洞」的第②腿 = **对称性**（「对手 block-dot 同样 per-block fold 无行摊·在比值中相消」）。
**前提 = "对手也串行付账"·而无人 objdump 过对手是否用 FMA。**
- **对手用 FMA/重排** ⟹ **不相消** ⟹ fold 对我方是**真成本** ⟹ **第②腿垮·8 格 K-quant decode 墙记全部重写**·
  且 **[构建保真三重] 旗标不对称**（我方 off/对手 on）⟹ 那批对比有效性重看。
- **对手也串行**（无 FMA）⟹ 相消成立 ⟹ 收益只在 K 路累加器（canon 自估 2–4×）。

## 三、预注册判读

- 税 **< 2%** → 数值线关掉·别再提（回去打真墙·§四）。
- 税 **2–4×** → 全线松绑·headline 全换（最快变体 + ULP 界）。
- **第②腿裁断**：对手 FMA 计数 >0 ∧ 我方 off ⟹ 第②腿垮（登记 ISSUE·8 格墙记重写提案）；对手 FMA=0 ⟹ 相消成立（第②腿保留）。

## 四、验收

1. **对手 objdump FMA 形态**（q4_K/q6_K vec_dot `_vl128` + block-dot·全量数·非截断·grep -c）。
2. **税数字**（两份编译 cold·2-seed·同格）。
3. **第②腿裁断**（相消 or 不相消·带对手 FMA 计数证据）。
4. **旗标对称性核**（我方 vs 对手编译旗标·是否让手·[构建保真三重]）。
5. **0 造数**·板测真数·**禁 commit·禁 add -A**·"某物不存在"禁截断（objdump 全量）。

## 五、触碰集 / 遗留

- 触碰：板端 objdump + 两份编译（scratchpad·仓库侧零落盘）+ 台账/journal。用 rvv 板。
- 独占（与并行线 iq3_xxs emit / D-2a research 不相交·纯 objdump+板测·不改 lib/）。
- **遗留**：若第②腿垮 → 8 格 K-quant decode 墙记重写 = 后续 task（提案入 ISSUES·队列动作立即执行）。§四 诚实边界：此线大概率不解 q4_K 整数 weight-reconstruction 墙·直接兑现只 q4_1/q5_1。
