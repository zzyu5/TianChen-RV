# Perf layer-4 特性化档案(指令级 emit 轴收官)

> 性质:**设计空间知识(layer-4 characterization)**,不是战役、不是新 beat 数字。
> 起因:perf 指令级 emit golf 正式收官——P2c null / 2a null / item4 赢 / 2c q5_K board-split / P3 regression。
> 本页把这一轴的**四个可迁移特性化 + 一条忠实性发现**命名入账;每条 = 命名 + 指纹 + 指针,不重复条文。
> 姊妹页 `方法学-kernel微基准两大隐形混淆.md` 管**测量混淆**(fp16-libcall / 跨-clang);本页管**emit 层杠杆的真伪**。
> 台账:两板条件赢在 `T8_winloss_gap_ledger.csv`(q8_0-item4 / q5_K-2c),分板数在 `T3_A`(rvv/VLEN128)/`T3_B`(k1/VLEN256)。

指令级 emit golf(单核 vec_dot 逐指令抠 ns)反复 null 后收敛出一条结构规律:**能被 clang 抹平的 emit-层改动
= 机器中性、零 perf;只有 clang 尊重的【C-结构杠杆】才真移机器码**。四特性化把"哪些杠杆是真的、哪些是幻觉"钉死,
供 repack 布局级战役与后续任何 emit 调整复用(不必重踩)。所有条目 KERNEL-micro-only,e2e 大概率 wash
(见 memory `kernel-wins-dont-transplant`),不进 beat 语境。

---

## ① machine-neutral vsetvli — emit-层删 vsetvli 零机器效果(2a null)

**命名.** skeleton-closure 2a 的前提:objdump 对账我方 vs factory 的 vsetvli(19→≤7)→ 删共享 emitter 骨架里的
显式 `__riscv_vsetvl` → 关 −8.3% q8_0 税。**证伪.**

**指纹.** 删 emit-层显式 setvl(F1 跳 pre-loop scope setvl 全 10 typed 格 + F2 q6_K super-block coalesce)后,
机器 `.text` **前后 byte-IDENTICAL**——clang-20 `-O2 -march=rv64gcv` 的 VSETVLI-insertion pass 自己重推最优放置、
**无视 C 里的显式 setvl**(删的是 clang 已 DCE 的死码)。原诊断的 "19 vsetvli" 是 deferred(死)变体;shipped
per-block 是 6 vs factory 5(delta 1、clang 插的)。

**特性化.** emit-层 vsetvli 数量是**机器中性的表面统计**——clang 拥有 setvl 放置的最终决定权,emit 层数它没意义。
任何"我方比 factory 多 N 条 vsetvli 所以慢"的诊断都要先过"删了机器码变不变"这一关。**F1+F2 已 revert**
(无收益改动不留树上 = 正面纪律)。

**指针.** gap-log GAP-1/2a-vsetvli-MACHINE-NEUTRAL;诊断稿 `07-03-m-flat-s5a1-perblock-load/research/skeleton-vsetvli-diagnosis-2a.md`。

## ② machine-gen-C 结构税 — emitc→C vs 手写 C 的可改杠杆(item4 赢)

**命名.** −8.3% q8_0 emitter-调度税**不在**可删的 emit-vsetvli(①证伪),而在【C-结构差】:我方 emitc→translate→C
vs ggml 手写 C,clang 把两者编成**略不同的机器码**。关键可动杠杆 = fp16-scale `fcvt.s.h` 的**放置位置**。

**指纹.** item4 = 把 `fcvt.s.h` 从整数核**前**挪到 fold 段(像 factory),+80/−80 **纯语句 reschedule**、数值 IDENT
(byte-exact vs §1 sep-left-assoc oracle,0 FMA)。**objdump 双板确认 fcvt.s.h 真移位**:
HEAD/B = `flh→fcvt.s.h→vwredsum→fmul`(早发);item4/A = `vwredsum→flh→fcvt.s.h→fmul`(后发)。
**clang -O2 不抹平**(对比①的 vsetvli machine-neutral)→ 这是**真机器改动**。结果:k1/VLEN256 our-m2 4096→3617ns,
从慢 factory 8.3% 翻成【快 4.37%】;rvv/VLEN128 本无税、item4 中性 parity。

**特性化.** machine-gen-C 相对手写-C 有真实结构税,但税住在 clang **尊重**的 C-结构选择(语句顺序/落点)里,
**不住在** clang 会自己重推的低层旋钮(vsetvli/寄存器分配)。**区分测试 = objdump 前后 diff**:真移(item4)→ 值得改;
被抹平(2a)→ 别追。★附带更正:finale 的"+11% m1-over-m2 fill 优势"**DISSOLVES**——那是 fcvt 税打 m2 比 m1 重、
**非 fill-utilization**;item4 修 m2 后 m1(3621)≈m2(3617)统计相同。**无 fill-lever Win-B;真赢 = C-结构 reschedule。**

**指针.** gap-log GAP-1/q8_0-item4;commit e7449feb;objdump `experiments/ondevice-q8_0-mbf/seal/fold_segment_objdump.txt`。

## ③ deferred-reduce 的微架构条件性 — 符号随 uarch 翻转(2c q5_K board-split)

**命名.** q5_K 我方核用 **deferred-reduce**(strip 累加延后、一次宽 `vredsum`;objdump ours vredsum **0** vs factory **9**);
代价是 **aux8[256] scratch round-trip**(ours `vse8` **8** vs factory register-resident **0**)。这两个力**方向相反**。

**指纹(objdump 双板机制分解).** 符号随微架构翻转:
- **k1(弱核 + 宽 vredsum 板,VLEN256)**:deferred 省的 > aux8 round-trip 花的 → **+2.7~4.8% 赢**(median +2.73%)。
- **rvv(快核,VLEN128)**:aux8 的 8×`vse8` + 256B 回读 > deferred 省的 → **−15~17.5% 输**(ours 1914.6 vs factory 1629.4ns)。

**特性化.** deferred-reduce 不是无条件赢的模式——它把成本从"per-strip reduce 延迟"搬到"aux8 store/readback 带宽",
**净值取决于板的 vredsum 吞吐 vs L1 round-trip 相对代价**。弱核 + 宽 vredsum 吃这套;快核被 aux8 逆风翻。
这是 **capability-keyed 模式库为何必须【换键不改条目】跨板迁移**的一个反面标本:同一条目在两板 diff≠0(board-split),
故**不能**作为 universal Win-B 封,只能作 board-conditional 登记。**⚠ 2c 当时把 aux8 round-trip 当唯一瓶颈 = triage 误判**
(④揭示真瓶颈是 decode-width,非 aux8)。

**指针.** gap-log GAP-1/q5_K-aux8-roundtrip;`experiments/ondevice-q5_K/{k1,rvv}_ab_raw.txt`(objdump seal + AB)。

## ④ static ≠ runtime — 预检全绿但运行时 2.2× 回归(P3 regression)

**命名.** ③的关闭动作 = P3 register-resident decode(消 aux8[256],预期翻正 rvv)。机器层**静态预检全 GREEN**:
`vse8` 8→0、spills 6→0、insns 513→232、byte-exact scalar-mirror。用户裁 Option 1 落地。**证伪.**

**指纹.** 硬件双板重测(preflight 4/4、on-device bit-exact 256/256 双板、confound-clean、paired):register-resident
**不但没翻正、反而双板大回归**——rvv 0.851→0.383(**2.2× 更慢**)、k1 +2.7% 赢翻成 −34% 输。factory byte-flat、
全 delta 在我方。★**根因 objdump 归因 = decode-width 降级(m2→mf2)**:aux8 形态 decode 在 WIDE `e8m2` vl=32/64
+ 廉价 L1 aux8 round-trip;register-resident 把 decode 融进每 MAC strip 在 NARROW `e8mf2` vl=8 → decode 工作量
×4(rvv)~×8(k1 半 datapath 空)。宽度罚 >> 消掉的 aux8 store。**★aux8 store 从来不是瓶颈——③ triage 归因错。**

**特性化.** `232-insn/0-spill/0-vse8` 预检 headline = **静态计数假象、不 transduce**(每指令只 8 lane)。
**静态 objdump 计数 ≠ 运行时延迟**;唯硬件重测(+ width 归因)抓到。预检-先行 + 硬件-验证纪律又一次救命
(①抓 machine-neutral、②确认 fcvt 真移、本条抓 static-count-red-herring)。**P3 register-resident-narrow 死路已 revert。**

**指针.** gap-log GAP-1/q5_K-P3-register-resident FALSIFIED;`experiments/ondevice-q5_K/{k1,rvv}_ab_p3_regres_raw.txt`;
诊断修正已 bank(commit 8e30ea2d + 0d83bdd9)。

---

## 忠实性发现(附带,非 perf 但同轴入账)

**命名.** 2c q5_K correctness gate 揭示:**我方 q5_K 核比 ggml 自己的 riscv 出厂核更忠于 ggml scalar 参考折叠。**

**指纹.** 双板 correctness gate(`{k1,rvv}_ab_raw.txt` CORRECTNESS 段):
- 我方 vs GENERIC q5_K oracle:**256/256,worst ULP=0**(BIT-EXACT)。
- factory(ggml 出厂 riscv 核)vs 同一 GENERIC oracle:**55/256,worst ULP=130**(NOT bit-exact)。
- 我方 vs FACTORY-fold oracle:55/256(即我方**不**匹配 factory 的粗折叠)。

**特性化.** ggml 的 riscv vec_dot 出厂核对参考 scalar 做了**有损的折叠简化**(worst ULP=130);我方 typed 构造核
逐块折叠**忠于 generic 参考**(ULP=0)。故三方关系 = **我方 256/256 忠于 ggml scalar generic > ggml 出厂 riscv 核 55/256**。
这是 correctness 轴的一个正面主张(不是 perf beat):在数值忠实性上我方核**优于对标框架自己的同-ISA 核**——
可辩护、objdump/gate 双证。**不外推成 perf 赢**(perf 是 board-split;忠实性是无条件、双板一致)。

**指针.** `experiments/ondevice-q5_K/{k1,rvv}_ab_raw.txt` CORRECTNESS 段;gap-log GAP-1/q5_K-aux8-roundtrip(★比 factory 更忠 ggml scalar)。

---

## 收官定性(指令级 emit 轴)

- **P2c(deferred-ordered fold)= null**:round-trip-elimination 假设证伪(vsetvli churn > 省的往返;physical)。
- **2a(vsetvli 删)= null**:①machine-neutral,clang 抹平。
- **item4(fcvt reschedule)= 赢**:②C-结构杠杆,clang 尊重,objdump 真移,k1/VLEN256 +4.37%(board-conditional)。
- **2c(q5_K deferred-reduce)= board-split**:③微架构条件性(k1 赢/rvv 输)+ 忠实性正面。
- **P3(register-resident)= regression**:④static≠runtime,decode-width 降级。

**结论(→ 宪法 §7):指令级 emit 微调默认降级为 sanity 层活动;例外需 item4 级双证据**——
(a) 真【C-结构杠杆】(不是 clang 会自己重推的低层旋钮)+ (b) clang 尊重(objdump 确认真移机器码、非被抹平)。
beat 的真住址在编译器够不着的结构层(repack/权重布局、跨板装载期调度、e2e 分相),不在指令级 golf。
