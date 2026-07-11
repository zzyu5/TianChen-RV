# 三具名 GAP 修复评估(本地·并行)

parent: `07-11-G5-wiring`（G5 接线战役）· status: in_progress · workflow `wez16hgku` 在飞。

评估性任务（本地·可与板/lib 线并行）：对以下三具名 GAP 做**根因 + 可修性 + Amdahl 传导预估 + 排序**，产出立项建议（不实施修复，评估先行）。

## 三具名 GAP
1. **[GAP-CLANG-GATHER-TRAP]** — clang 向量 gather 陷阱。
2. **[GAP-DEQ-KQUANT-UNPACK]** — K-quant dequant unpack 缺口。
3. **[GAP-FWD-M8-VSETVL]** — forward M8 vsetvl 路径缺口。

## 每 GAP 产出
- **根因**：反汇编/代码定位（性能常驻规则①：修性能前先反汇编认瓶颈）。
- **可修性**：本地可修 / 需板 / 需上游；工程量级。
- **Amdahl 传导预估**：目标占相内时间比例 × 预期改善 = e2e 上限；★kernel 因子输入必须与目标部署【同域】（同编译器/同 deployed variant/同输入路径），否则 garbage-in（[CASE-COMPILER-ASYMMETRY]）。上限低于噪声地板者只能以机制/方法学名义立项。
- **排序**：三者按 (可修性 × Amdahl 上限) 排优先级，给立项建议。

## 边界
- 纯评估，不实施；立项属必问①（若建议新战役）或走既定队列执行（自决）视归类而定。
