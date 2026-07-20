# PRD — 能力层补口: version 管道收敛 + vreg_count 正名（柱一主线·能力层审计钉出的 2 top ✗）

## 权威 = 能力层审计@HEAD(d23a9e32c) + 用户主线令(g/c→plan·柱一)
审计判决能力层 ~1.5/5(机制真·广度窄·表未建)。本役补 2 个 pinpointed 缺口(= 用户 5 阶段的 ②补管道 + ③reg budget)·byte-exact·判决实验前置。

## 做 A — version 轴管道收敛(② 补管道·VLEN 拔管道漏的姊妹轴·一行级)
- **病**(审计机核): `RVVLowerQuantContraction.cpp:1661` 仍 `deriveRVVVersion(march, isaVectorHints)` = **march 重解析·且载重**(穿进 selectRepackAccumulatorLMUL + 全家 lowerToRepackGem{v,m})。typed reader **`readRVVProviderRVVVersion` 已建且他处已用**(`RVVCapabilityProfile.cpp:499`·`RVVRepackStripWidthMaterialization.cpp:134` 已消费)——此前门没换。
- **做**: :1661 的 `deriveRVVVersion(march,...)` → **`readRVVProviderRVVVersion(module)`**(读 in-IR provider version 事实·march 只探测层一次)。自验是否有其它 load-bearing `deriveRVVVersion(march)` 生产直调·一并收敛。
- **判决 lit**(能力 vs march 真绿): 能力文件写 rvv_version(如 provider 标 RVV1.0)·march 写冲突值(如 rv64gc_xtheadvector→0.7)→ isRVV0p7/repack accumulator θ **跟能力文件·非 march**。翻 march≠假绿。

## 做 B — vreg_count 正名(③ reg budget·审计 top ✗·硬编码→能力事实)
- **病**(审计机核): `getRVVArchitecturalVectorRegisterCount()` body = `return 32;`(`RVVCapabilityProfile.cpp:564`)·+ **8 处 `constexpr ...VectorRegisterBudget/Count = 32`**(`RVVGearboxSchedule.h`·连具名 authority 都不读)。register-budget 有 typed 读路径 + decisive lit(`rvv-register-pressure-inequality-decisive.mlir` budget 32↔9 翻累加器 LMUL)·但**源是硬编 32**="读端真源端假"。
- **做**: ① 探测层产出 `vreg_count`/`vector_register_budget` **in-IR 能力事实**(materializeRVVProviderCapabilityAxes 盖章·默认 32·可被能力文件覆写) + reader(如 `readRVVProviderVregCount`)。② `getRVVArchitecturalVectorRegisterCount()` 与 8 处 constexpr → **读该能力事实**(缺则 fallback 32·byte-exact)。③ reg-pressure 不等式吃能力事实的 budget·非常量。
- **判决 lit**(能力事实驱动预算): 能力文件写 vreg_count=16(假想窄寄存器板)→ reg-pressure 可行域/累加器 LMUL **随之翻**(复用/仿 budget 32↔9 decisive lit·但源改为读 in-IR 能力事实·证 θ 跟能力事实非硬编码 32)。现役板(32==默认)emit byte-exact。

## 门(byte-exact·判决·禁假绿)
- **byte-exact 现役点**: version 现役板(能力==march)+ vreg 现役板(能力==32默认) emit md5 对齐·CORE==PROD·全套 lit 无 NEW 失败(基线 3 pre-existing)。
- **判决 PASS**: version 能力 vs march→θ 跟能力(翻 march≠假绿)· vreg 能力文件写 ≠32→budget-θ 跟能力事实(非硬编 32)。
- 反向保险: `git grep deriveRVVVersion(march` 生产直调收敛· `git grep 'VectorRegister.*= 32'` constexpr 旁路降(→读能力事实)。
- 🔴 禁碰 grid/dequant 发射器·B线格· 无 inline-asm· 未 git commit。VERIFY ANCHORS FIRST(naive-grep 教训)。

## 汇报(首节 判决 x/y + byte-exact)
version: :1661 收敛+判决(θ 跟能力 version vs march) + deriveRVVVersion(march 生产直调数 before/after。vreg: 能力事实落地 file:line + 8 constexpr 收敛数 + budget-θ 跟能力事实判决 + byte-exact md5。全套失败数(应仍 3)。禁新建分析文档·结论进 final message。
