# PRD — MIG-5 / 行动书 §3.4：家族/宽度选择器闭式化（预批 B-SEL-CF·A 线·判决实验前置）

## 权限 & 缘起
【现行法·预批 B-SEL-CF】（行动书 §五 已裁·§3.4「可与管道并行·输入 VLEN 现成·不等管道」）。这是**唯一"真新写"公式**（非接管道·非补输入）·核心 thesis = capability 驱动选择。**A 线任务·判决实验前置硬规**：本 prd 已含判决实验设计（§判决实验），无它不开工。

## 已知事实（省得你重走·但**必须先自验锚点**）
- 生产 dispatch 路径**已断言零 static_order**（`RVVLowerQuantContraction.cpp:1530` [SEL-1-T5] invariant）。
- reduction 路 `selectIntegerCoreLMUL`（`lib/Plugin/RVV/FrontDoor/RVVReductionSourceFrontDoor.cpp:344`）**已随 VLEN 真翻 m2↔m1**（r5.1 判决实验 JE2 GREEN·静态 argmin·capability 真消费）。
- ⚠**PRD 锚点可能 naive-grep 误判**（迁移#1 就撞过：PRD 说的位置与实际不符·agent 诚实重路由才对）。**你的第一步 = 机核验证下列锚点真实位置**（`grep -n` 定义体）·**与 PRD 不符时按实际重路由**（照迁移#1 范式·在交付里标"PRD 锚点误判·实际在 X"）·**禁按错锚硬写**。

## 做（两条半新公式·§3.4·各带判决实验）
**核心洞察**：选择器**已翻**（JE2/JE3），但翻的逻辑是 **argmin/枚举兜底**散在选择器体里·**不是一条具名闭式 f**。本役 = 把它**提为公式层家（`include/Weft/Plugin/RVV/RVVGearboxSchedule.h`·已住 `getRVVCodebookGatherAnchorLMUL`@2713 + `rvvRegisterPressure*`@2141）里的显式闭式 f**（同 MIG-0/MIG-B 范式），选择器改**调用该 f**·并证它承重（非兜底摆设）。

1. **家族/宽度选择器闭式化**：`θ_width = f(VLEN, dataWidth)` —— 取 LMUL 使**有效寄存器组宽度恒定**（`m2@VLEN128 ≡ m1@VLEN256`·即 VLMAX×SEW 不变）。写成 `RVVGearboxSchedule.h` 里的具名 `constexpr`/`inline` 闭式（如 `getRVVEffectiveWidthInvariantLMUL(VLEN, sew)`）·reduction/宽度选择器改调用它替 argmin 兜底。
2. **宽化收益判据入选择器**：「VLEN 是数据宽两倍时窄发射白扔半通道」现活在战报散文·写成 leaf 宽度选择的**显式判据**（消费 VLEN + dataWidth·两侧齐全·零新字段）。

## ★判决实验（拧1·A 线最高假重构风险——写出 f 但选择器仍走兜底 = 公式版假重构）
> **committed lit**：同一 fixture·**只翻能力文件里的 VLEN（128↔256·可用 `rvv.vlenb_bytes` 或 march·单变量）**·选择器输出的**家族/宽度必须翻转**（如 m2@128 → m1@256·有效宽度恒定）·**且落一条带 pin 的「能力翻转→改判」记录**（reason 非 static_order）。
> **反向保险**：伪造它的唯一办法是真让选择器消费 f 的输出——加了 f 但输出不随 VLEN 翻 = 摆设当场现形。
> **byte-exact 底**：正常值（现役板 VLEN）下·emit 逐字节不变（闭式 f 在现役点 == 旧兜底值·md5 对齐 CORE==PROD）。翻 VLEN 才见改判。

## 触碰集（先自验·禁碰在飞 agent 文件）
- **主**：`include/Weft/Plugin/RVV/RVVGearboxSchedule.h`（加闭式 f·公式层的家）+ 宽度/家族选择器体（**自验**：`RVVReductionSourceFrontDoor.cpp` selectIntegerCoreLMUL / `RVVContractionPathSelection.cpp`·按实际重路由）+ 判决 lit（`test/Conversion/RVV/` 新建）。
- 🔴 **禁碰**：`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（迁移#1）·`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`（census agent 只读中·迁移#1 待集成）·`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`（刚集成 B1·避免再动）。若你的重路由指向这些·**停·在交付里报"需碰禁区·请 supervisor 裁"**·别硬改。

## 门
- **判决实验 committed + PASS**（翻 VLEN→宽度翻·带 pin 改判记录）· byte-exact 现役点（CORE==PROD md5 对齐）· 生产路径 `git grep static_order` 仍零（[SEL-1-T5] 不破）· 无 inline-asm · 未 git commit。
- **三闸 §四.5**（若碰发射侧）：裸格式字面量/板值 value_or/march 解析·计数 Δ≤0·readings 入交付。
- **禁假重构**：f 写出但选择器不消费 = 失败。独立复核会**只跑判决实验不看 diff**。

## 汇报（自然·首节判决实验 x/y）
判决实验 x/y + 闭式 f 落在 RVVGearboxSchedule.h 的签名 + 改判记录 pin + byte-exact md5 + 锚点是否与 PRD 一致（不一致标实际位置）。final message 放关键结论（非仅在文件）。
