# PRD — W1 A线拔管道: deriveMinimumVLEN 收敛到探测层一处 + 判决(能力文件 vs march 打架)

## 权威 = r5.1 goal（本 goal 即令·非文档）
A线「θ 的 c 侧输入从 march 旁路真拔到类型化能力表」。**判决实验 = 能力文件写 vlen256·march 写 zvl128b（故意打架）·承重 θ 跟能力文件走才算通·跟 march 走 = 旁路还活**。**反向保险：`git grep deriveMinimumVLEN` 调用点收敛到探测层一处**（旁路定义 = 多处重解析·多处还在 = 没完成）。

## 病（机核·supervisor 已定位）
生产路径 VLEN 靠各 pass 重解析 march：**4 个前门 + repack** 直调 `deriveMinimumVLEN(march)` 而非读 in-IR 类型化事实：
- `RVVDequantDotSourceFrontDoor.cpp` / `RVVPackedI4DotSourceFrontDoor.cpp` / `RVVCodebookDotSourceFrontDoor.cpp` / `RVVReductionSourceFrontDoor.cpp`（各 1 处 `deriveMinimumVLEN(` 喂 selectGenericSchedule 的 legality gate）。
- 探测层已有正确读者：`resolveRVVMinimumVLEN(module)`（`RVVCapabilityProfile.cpp`·优先读 in-IR `rvv.vlenb_bytes`/`minimum_vlen` 类型化事实·缺则回退 deriveMinimumVLEN·T1 已建）。`materializeRVVProviderCapabilityAxes` 已用 `readRVVProviderVLenBBytes`。

## 做（拔管道·byte-exact 现役点·翻能力文件才见改判）
1. **收敛调用点**：4 前门的 `deriveMinimumVLEN(march)` → **`resolveRVVMinimumVLEN(module)`**（读 in-IR 类型化事实·march 只在探测层出现一次）。`git grep 'deriveMinimumVLEN('` 生产消费点收敛到**仅探测层定义 + resolve 内部回退**（前门 0 直调）。
2. **判决实验 lit**（committed·单源判决法·伪造它唯一办法=真接管道）：能力实例文件写 `rvv.vlenb_bytes bytes=32`（=VLEN256）·pass `-march=rv64gcv_zvl128b`（=128·**故意打架**）→ 某承重 θ（如该前门下游 strip/lmul）**跟能力文件走（256 派生值）·非 march（128）**。跟 march 走 = 旁路还活 = FAIL。**反向保险 lit**：加第二 march（zvl512b）·θ 恒跟 vlenb·invariant to march。
3. **t1d 注记删**：`tools/visibility/t1d_dual_instance.py` 或相关处「march= is a PASS OPTION, not an in-IR fact」注记·收敛后删得掉即证通。
4. **byte-exact**：现役板（能力文件与 march 一致时）emit 逐字节不变（md5 对齐·CORE==PROD）·只有故意打架 fixture 才见 θ 跟能力文件。

## 门（六数·all-or-nothing·禁假绿）
- **判决实验 PASS**（能力 vlen256 vs march zvl128b·θ 跟能力文件·反向 zvl512b 恒跟 vlenb）· `deriveMinimumVLEN(` 生产直调收敛（前门 0）· byte-exact 现役点 md5 对齐 · **翻 march 看 θ 跟着翻 = 假绿·不算管道通**（必须 θ 跟能力文件·不跟 march）· t1d 注记可删 · 无 inline-asm · 未 git commit。
- 🔴 禁碰: repack width 选择器闭式化(W2)·选择器 f 补输入(W3)·B线发射器(W4/W5)。**worktree 隔离·主会话串行集成**。

## 汇报（首节判决实验 x/y）
收敛前后 `deriveMinimumVLEN(` 直调数(前门→0) + 判决实验结果(θ 跟能力文件 vs march) + byte-exact md5 + t1d 注记是否可删 + 锚点是否与 PRD 一致(不一致标实际)。禁新建分析文档·结论进 final message。
