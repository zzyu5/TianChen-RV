# 方法学:kernel 微基准的两大隐形混淆

> 性质:**设计空间知识 + 已落治理的说明**,不是战役,不是新数字。
> 治理执法在别处(实验总纲v1 §1 第9/10条 + `T3_step3/board_ab.sh` preflight),本页只**命名混淆 + 指纹 + 指针**,不重复条文。
> 起因:P2c deferred "1.69× vs factory" 被公平复测证伪 = 100% 此两类混淆之一(见 gap-log GAP-1/P2c-A、P2c-B)。

kernel 微基准(单核、固频、cache-resident、best-of-N)看似干净,但两类混淆能把
**工具链缺陷误报成算法/布局赢**。两者都不改变正确性(bit-exact 照过),只污染 ns 比值,
所以只靠 correctness gate 抓不到,必须靠反汇编指纹 + 构建保真政策拦。

---

## ① fp16 libcall(对手 march 漏能力 → 残废对手)

**机制.** 板上有硬件 zfh,但对手编译 `march` 漏 zfh → 对手每读一次 per-block
`(float)*(const _Float16*)` scale 就降级成软浮点 `__extendhfsf2` libcall。q8_0 每 dot 有
256 次这样的转换 → 对手被拖慢 ~2×。这不是我方赢,是**对手残废**(crippled build)。

**指纹.** 对手热路径 objdump 出现 `__extendhfsf2`(软浮点半精转换 libcall)。
去 libcall 后 factory 单侧 **1793→881ns**(半个运行时蒸发);同源核在含-zfh march 下无此符号。

**已落拦截.** 实验总纲v1 **§1 第10条(对手构建保真)** + `board_ab.sh` preflight 门①(march
完整性:板实测扩展 ⊄ 编译 march 则 FAIL)+ 门②(双侧 objdump libcall 扫描:任一侧含
`__extendhfsf2` 类则 FAIL,fail-closed)。这次混淆的指纹正是门②要抓的东西。

## ② 编译器版本敏感(跨指纹不可比)

**机制.** 同一 kernel `.o` 从 clang-17 换到 clang-20,kernel-ns 差 **3.5–4×** 且 **our/factory
排序翻转**(哪侧更快随 clang 代际变——clang-20 把 factory 的简单循环优化得比手写核还狠)。
编译器代际差 >> 我们要测的算法/布局差,任何跨这两个指纹并列或推断趋势的比较一律作废。

**指纹.** step-3 双板测量用 clang-17/18;后来的 deferred "1.69×" 用 clang-20——**从不在同一
编译器基线**。同 `.o` 跨 clang 的 3.5–4× + 排序翻转即证据(见 gap-log GAP-1/P2c-B)。

**已落拦截.** 实验总纲v1 **§1 第9条(工具链政策)**:perf 双板统一最新稳定 clang(当前 clang-20)、
两侧同版本同旗标、`march` 从板实测 `hwprobe`/`/proc/cpuinfo` 生成并串入环境指纹(第1条:指纹
任一分量变 → 同指纹格自动 stale)+ `board_ab.sh` preflight 门③(同编译器断言)/ 门④(指纹-格
匹配)。厂商编译器现实(K1/Bianbu clang-18)= 记 limitation 脚注,**不做第二套测量**。

---

**指针(不重复条文).** 执法条款见 `docs/TianChen-RV_实验总纲v1.md` §1 第9/10条;preflight 四门
在 `experiments/T3_step3/board_ab.sh`;被证伪的原始误判命名在
`.trellis/tasks/07-04-m-flat-nb-loop-layer/evidence/gap-log.md` GAP-1/P2c-A(fp16 libcall)、
GAP-1/P2c-B(跨 clang 指纹)。本页是这两条治理为何存在的**设计空间说明**,治理本身已落。
