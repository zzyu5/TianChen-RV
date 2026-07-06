# M-FLAT 循环 scaffold 设计 + 可行性判决

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑)· **base:** refactor/full-refactor-m1
**只研究不改代码**(写进 `research/`)。

## 为什么(读我)

2 次 STOP + primary-source 核实证实:**flat 家族(q8_0 等 7 格)翻 constructed-strong 的唯一通路 = 替换循环体 `emitFlatBlockDot` = 需一个 typed nb 循环 body。** 砖①②③ 是搁浅的**标量单块**原语(对 emit 文件 116 插入/0 删除,纯加法),**任何单块 route 翻不动 flat 格、删不掉弱体**。根因锚定:砖① `BlockFp16ScaleProductOp::verify`(`RVVDialectWideningOps.cpp:9750-9765`)硬要 scale base 是导入 `RuntimeABIValueType`+LHS/RHSInputBuffer role,**读不了 per-iteration `base+ib*stride` 块 scale**。

**前 2 次尝试 STOP 的教训 = 没先设计循环 scaffold 就派实现。这一砖就是补这个设计**(scaffold-first:先把最硬的载重循环体设计清楚,再谈实现)。全接线墙已实测在 `../07-03-m-flat-wire-first-strong-route/research/wiring-cost-map.md`(§二墙表 + §四根因)。

## 产出物(全部写进 `research/loop-scaffold-design.md`,code-anchored)

1. **per-block-source 原语契约**:怎么让 typed 原语读 per-iteration 块 scale(`base + ib*stride`)。选一:扩砖① `BlockFp16ScaleProductOp` 的 source 契约(放宽 verifier 收循环归纳变量派生的偏移)vs 新兄弟 op。给出 op 签名 + verifier 契约 + 它接受什么 SSA(归纳变量?stride?导入 base + 计算 offset?)。**这是根因缺口,重点设计。**
2. **nb 循环 body op 结构**:typed 循环 op 长什么样(region-based 循环 op vs `scf.for` 包裹 vs attrs)?loop-carried f32 累加器怎么表达?body = per-block-source(#1)+ 既有 widening_product/standalone_reduce + 砖②(computed-scale dequant)+ 砖③(f32 累加)+ load/store/setvl,**零 opaque helper**。给出 body 的 primitive-ID 链。
3. **loop-aware allowlist validator**:怎么递归进循环 region 做 **allowlist** 校验(注意 cost-map 警告:`rejectMixedPreRealizedContractionBody` 是 **blocklist**,把砖当模板参会**拒**砖体——要反写成 allowlist:每个 body op ∈ {per-block-source+砖②③+widening/reduce/load/store/setvl/withvl+循环 op})。给出它挂哪个真实调用点、既有单块 rejectMixed 路怎么零回归。
4. **wire-in 替换 emitFlatBlockDot**:这个 typed 循环 route 怎么注册(front-door / route-identity / protocol / op-kind,参 cost-map §二墙表)+ 怎么把 q8_0 dispatch 从 `emitFlatBlockDot` 改到新 route → **翻 q8_0 六态格 constructed-weak→constructed** + **删掉 emitFlatBlockDot 里 q8_0 那条(或整体,若 5 格共享弱体一起翻)手写体 → Δ手写LOC<0**。emit-consistency:新 route lower 到与 `emitFlatBlockDot` q8_0 路**字节一致**的 emitc(emitc.for + 每块 typed 原语),lit 锁。
5. **砖①②③ 复用/扩展/弃用判定**(诚实):哪些能原样复用、哪些要改契约、哪些其实白建。别粉饰。
6. **对抗性可行性判决**(必须给,别乐观):这是 **tractable-多会话**(估:几会话/什么序)还是**真设计墙**(MLIR/管线架构上无法表达 typed nb 循环 body 的某处)?**点名最硬的单个子问题。** 前 2 次尝试都过度乐观 → 你要**反向压测**你的判决:主动找"为什么这也会 STOP"的理由,找不到才敢判 tractable。若判真墙,说清架构原因(那是 parent 改道信号)。

## 红线

- **只研究不改代码。** 不建 op、不改 verifier、不动 dispatch。产出是蓝图 + 判决。
- 数值 bit-exact = pending-hardware,不涉。
- 每个设计断言 code-anchored(文件:行)。不臆断方言有某 op——grep 确认(尤其 #1 的归纳变量/stride 表达、#3 的标量-i32-extract 桥,cost-map 阻断 B 未逐 op 确认)。
- **不造 n/N 计数指标。** 进度语言只用 ΔC_construct + Δ手写LOC。

## 权威 spec

core-invariants(I4/I5/I7,强义=零 opaque helper)· 执行总纲 [K-4] 六态 · burn-down 纪律 · cost-map(`../07-03-m-flat-wire-first-strong-route/research/wiring-cost-map.md`)· 砖④ PRD(`../07-03-m-flat-brick4-block-loop/prd.md`)。
