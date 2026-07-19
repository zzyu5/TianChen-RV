# PRD — W2 生产者边界二选一 + widen-to-m1 在①保护下补重测

## 触碰集（W2 专属·含进料口写权）
5 个 SourceFrontDoor（`RVVCodebookDotSourceFrontDoor` / `RVVDequantDotSourceFrontDoor` / `RVVPackedI4DotSourceFrontDoor` / `RVVReductionSourceFrontDoor` / `RVVMonolithicBlockDotSourceFrontDoor`）· `RVVCapabilityProfile.{h,cpp}` · `RVVProbedCapabilityAxesMaterialization.cpp`（探测层·进料口写权归本流）· schema/描述符（进料口）· widen-to-m1 board experiment kernels（`experiments/active/` 新建独立目录·别碰 r-dequant）。**禁碰**：RVVLowerQuantContraction/Gearbox（W1）· RVVToEmitCGridCodebook（W4）· r-dequant kernels（W3/W5）。F25(reduction baked-g 补输入) 归本流（在 RVVReductionSourceFrontDoor）。

## ① 5 constructor 生产者边界二选一（禁再"休眠以后再说"）
W3b 发现 5 个 SourceFrontDoor constructor 跑在 source-only IR·resolveRVVMinimumVLEN fallback march。**逐个二选一·机核证据**：
- **A. 证它构造时已 stamp minimum_vlen（及其余 c 事实）到它产出的 provider op** → 正当生产（生产者本就该解析 march 一次产 fact）·从"债"改判"正当生产"。判决实验：constructor 产出后·下游消费者读的 minimum_vlen 来自 constructor stamp 的 provider·不是自己 re-parse。机核：grep constructor 是否 setAttr minimum_vlen 到它 create 的 capability op。
- **B. 未 stamp（产空 provider）** → 真债·给探测层挂生产驱动（census：探测→provider 生产 pipeline 零挂接·唯一调用者是测试）·让生产路径真物化 minimum_vlen。判决实验：挂上后·不带 provider 的输入编译时 provider op 携带 minimum_vlen·下游读它。
- **禁第三态**。5 个各出 {A 正当生产 / B 已挂生产驱动} + 机核。

## ② F25 补输入（reduction baked-g）
F25 签名只吃 c·g 全 baked（selectIntegerCoreLMUL reduction 路·JE2 已证随 VLEN 翻 m2↔m1）。判 baked g 常量对现格式集是否恒真·会破就把 g 提真参数·θ 随 g 动（committed lit）。

## ③ widen-to-m1 在①(寄存器压力不等式)保护下补重测（补充令纠正 [GAP-P1]）
[GAP-P1] 不是永久物理事实·是"旧发射器会 spill"的实测边界。**用寄存器压力不等式(W1 commit 0e9faee0a·rvvRegisterPressureLegal)精确算 spill·找不 spill 的格式·在其上重测 m1 vs mf2 board·给数据**：
- 还输 = [GAP-P1] 真墙钉死；有格式赢 = 边界该松（**结论 supervisor 裁·agent 只给数据**）。
- 🔴 **仍禁自决硬写 VLEN 投影**（违 canon + 抢裁）· board 重测是独立 experiment kernels（m1-chain vs mf2-chain·byte-exact·2-seed cold）。
- 立场文腿一中性措辞（纯论文侧文本·canon 本体不动·挂 ISSUE-117）："宽度可行域由寄存器压力不等式闭式定；可行域内往最宽走是否有收益·待发射器成熟后由实测标定"。agent 可改立场文该句。

## 门 + 交付
- byte-exact（constructor 改动若动 emit 须 byte-exact 验）· lit 绿 · 三 grep 只减不增。
- **交付首节自带「本流判决实验 x/y」** + 5 constructor 二选一表 + widen-to-m1 重测数据（spill-free 格式 × m1 vs mf2 ratio）。
- 账面：不 git commit（报清单）· 0 造数 · rvv 编译 + board。
