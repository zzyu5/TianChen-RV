# 完整重构 —— 能力驱动 RISC-V 执行层成熟化(引擎轴 × 证据轴)

> **状态:ACTIVE(2026-07-02 经 `/goal` 放行开工)。** REVIEW-GATE 已由用户 `/goal` 指令解除——用户令"按 trellis 流程推进 PRD 直到达成科研目标 + 成熟 compiler + 完成实验",即对本 program 的放行。本 PRD 定义**父级 program** 的目标、双轴 pillar 与子任务分解;子任务按下方修正执行序逐一 spin+落地。范围裁决:父级 program PRD(M1→M4)· **两轴并行(用户定:证据线与引擎线一起做,从一开始就是成熟 compiler,不是先证据后引擎)** · 理想项(X-SCALAR / IME 自有 GEMM)纳入为 gated M3+ pillar。
>
> **权威三总纲(docs/):** `科研目标总纲v2`(证什么·目标态)· `执行总纲v2`(现在到哪·代码锚点·钉快照)· `实验总纲v1`(怎么证·表集 T0–T8)。spec 已吸收三总纲思想(`.trellis/spec/` C1/C2/C3′ + [L-6]/[L-8]/[K-4]/[S-5]/[F-2′]/[F-6]/[SEL-2] + 实验宪法)。experiments/ 已建空表模板。**本 PRD 是把三总纲落成可执行 pillar 的桥。**

---

## Frame(不可动摇)

完成 [G-0] 终态:**一个以"能力事实"为唯一决策源的 RISC-V 执行层编译器** —— 接一个新家族 = 一个插件 + 几行表,核心零改动;kernel 主体由机制构造(强义,[L-8])而非手写;每次变体选择可归因;每条结构性主张机器可检验。**成熟度即 novelty,且成熟度进 CI 不进 slides([G-1] 双轴)。** 我们瞄强论文(C1/C2/C3′),不写靠 reframe 的保守版;rigor(byte-exact / 证据状态 / 不 over-claim / 快照纪律)是审美不是保守,保留。

---

## 三个交付指标(本 PRD 按此组织;pillar = 达成它们的执行计划)

- **指标① 成熟 compiler(覆盖率):** 四指标进 CI —— C_dispatch / **C_construct 强义(燃减主指标)** / C_construct+ 弱义(过渡·不 gate)/ C_attr 分级。目标值见里程碑 [COV-3′]。"我们是一个编译器,不是一条编译器路径。"
- **指标② 量化强论文创新点(逐点:是什么·要怎样·status):**
  - **C1 合取存在性 → 可复制协议**:一份带关系 schema 同驱动编译期生成 + fail-closed 运行期守卫,跨范式(向量→矩阵 MAC)且跨独立家族原封复用;零分支 + schema 复用由 [F-1..F-6] 机检。**现状:结构已证(唯一结构性证明)、板无关、今天最强**;缺口 = schema 工件化([S-5])+ 独立家族广度见证([X-SCALAR])+ 外部接入(M4)。
  - **C2 泛化代价 → 边际成本规律**:逐家族接入代价成边际递减曲线 + 成本住址分解。**现状:首点可复算(目录 LOC)**;缺口 = 自动 ledger + 第三家族成曲线。
  - **C3′ 能力键控模式库 → 带实测与迁移的模板**:P1–P8 以能力谓词表达、由机制选出(归因)、换键不改条目迁移、对 tuned-ggml 分相报增量。**现状:收缩族已数据化、选择器现为常量分**;缺口 = 统一模式注册表 + 归因 JSONL + 能力先验层 + body 强义构造。
- **指标③ 最终性能实验:** 两板 × {micro, e2e},过 [PERF-1] 八门,落 `experiments/` 表集(T3/T6);先测 T-N 噪声地板 + 对手解析探针,再判任何效应。**parity-now / beat 待验**,[NG-4] 未过门前 beat 措辞禁入。

---

## 双轴 Pillar 执行计划(达成①②③的 how-to)

> 每 pillar 挂:总纲条款 · experiments 表 · 执行总纲现状 · 里程碑。bounded exit = **可机检的状态**(非"改了某文件")。

### 证据轴(M1 主体 —— 便宜、解锁一切、关闭 C1/C2/C_attr 机检缺口;实验前置工具即证据线,一次建设两处收益)

| pillar | 内容 | 条款 | bounded exit |
|---|---|---|---|
| **E0** | 清除代码注释里的 beat 措辞越界(零成本,即刻)。**非 round-owner**——cleanup/metadata-only 不套完整 task→PRD→implement→check 仪式,主会话直接编辑 5 处注释 + 一个 quick commit 清 NG-4 越界即可 | [NG-4]/[L-1]/[L-7] | 全仓无未过 [PERF-1] 的 beat 断言 |
| **E1** | 起草 `schema.def` v1(六项反写已备)+ 操作门 | [S-5]/[F-2′] | schema.def 落盘 + 版本日志 + 逐 PR diff∩schema.def=∅ 门在 CI |
| **E2a** | **op-attribute `kind` → `target_kind`/`region_kind` 消歧**(纯代码,有界,动 ODS;**须先于 E1 schema.def v1**——免契约文档写消歧注记 + 减 [F-1] grep 假阳性)。**本轮第一个真 module。** | 命名消歧 | op 侧 kind rename 完成 + byte-exact 门(clean relink + BEFORE==AFTER) |
| **E2b** | 家族代码目录归拢(IME/RVV 从 `lib/{Dialect,Plugin,Conversion,Target}` 挪进 `plugins/<fam>/`;blast radius 大,单列一个 module 慎做,别当热身)。解锁 F-3(现"不可判定"的 C1 falsifier),值得靠前排 | [F-3] | 单一 `plugins/<family>/` 布局 + 收容门可判定 |
| **E3** | Falsifier 组进 CI(curated 正则四型谓词 + 判读规程 + fuzz + 独立性判据) | [F-1]/[F-5]/[F-6] | 每 PR 触发、假阳性排除、CI 绿 |
| **E4** | 编译期归因 JSONL 出口(reason 三值枚举 + candidates/keys/declared_instance_hash/ts)+ 装载期最小解析记录 | [D-4①]/[D-2a] | C_attr^CT 抽样 100% + 每进程一条解析记录 |
| **E5** | provenance 清单 → 六态自动读出 + [L-8] 执法 | [K-4]/[L-8] | T0 六态脚本可判、弱充强被 CI 拦 |
| **E6** | 覆盖率脚本(四指标)+ 分母定稿(钉 ggml 版本,低比特族移入分子)+ ledger 脚本(复算首点) | [COV-1/2]/[LED-1] | T0/T2/T7 出数进 CI 报告 |
| **E7** | 编译期门未知即拒自足(continue→拒绝,统一 symbol/ID 语义) | [D-1] | fuzz 单跑该门不漏 |
| **E8** | T-N 噪声地板 + 对手解析探针(实验前置) | 实验宪法 §1.3/1.4 | 每板噪声地板 + opponent_map 工件在,vs-framework 无探针即 INVALID |

### 引擎轴(M2 主推进 —— C_construct 强义燃减是主战场)

| pillar | 内容 | 条款 | bounded exit |
|---|---|---|---|
| **G1** | body 模式库(把 17 dispatch-wired + 7 constructed-weak 推向强义 constructed;大网格/超块码本参数化;Fork 收敛)+ 统一模式注册表 | [K-2]/[PAT-1] | C_construct 强义 ≥40%;注册表为数据(带 metrics_hook/status) |
| **G2** | 泛型操作数层成为唯一发射权威(掩码族之一经泛型层重发射,字节精确不回退) | [K-3b] | 双路收敛,泛型层接管块点积权威 |
| **G3** | SEL-1 能力先验层(纯成本之上加先验;GEMM∧ime→矩阵范式 / 否则最宽 LMUL / 否则默认)**硬绑先于/同于 P7** | [SEL-1]/[SEL-2] | T4b 消融"矩阵静默落败"复现→消失 |
| **G4** | [K-2b] 宽钳位双区 body(关闭钳位回归缺口环)→ 同板复测 | [K-2b]/[K-6] | GAP-1 首实例:命名→关闭→复测→归因引用 gap-ID |
| **G5** | X-ZVFH 接入(kind=sub_ext, implies rvv.v)含 [S-2] 闭包修复(同里程碑验收) | [X-ZVFH]/[S-2] | @zvfh 注册为能力事实 + f16 路径 + 闭包不误拒 |
| **G6** | 结构化参数 + provenance/trust + kind 闭合枚举 + uarch quirk 表 | [S-1]/[S-4] | params 命名空间化;每核 quirk 表供 P6 键控 |

### 硬件/性能轴(M3)

| pillar | 内容 | 条款 | bounded exit |
|---|---|---|---|
| **P1** | 资源感知 cost(读真 VLEN/ELEN/vreg/mask/tail,非 1-bit) | [S-7]/[SEL-3] | cost 能力感知且隔离,重测校定 |
| **P2** | 真硬件探测 + 运行期链(hwprobe→事实→instance-hash→调度表) | [S-3]/[D-2b]/[D-3] | provenance=hwprobe/trust=measured;C_attr^RT 启用 |
| **P3** | 两板 micro+e2e 实测填 experiments/(T3/T6),分相、固频钉核、同版本 ggml、按 instance-hash 存库 | [COV-5]/[PERF-1] | T-N 过 + 八门逐项落列 |
| **P4** | 首个机制合成 prefill 赢(过 PERF-1 八门) | [PERF-1] | 八门全绿 + 机制合成归因 |

### 理想/存在性轴(M3+ · pre-declared · gated · 回退不改任何已发表主张)

| pillar | 内容 | 条款 | gate / 回退 |
|---|---|---|---|
| **X1** | X-SCALAR 独立家族广度见证(owned 内核 = 三值 2-bit vec_dot 标量路;保底 = q4_0 dequant;Zbb 作能力门非主力;fp16 scale 走 zfh/软件不破独立性)→ C2 第二数据点 | [X-SCALAR]/[F-6]/[C2] | gated on owned-kernel 数学核查(已定);**回退:C1 轨1/C2 首点独立成立** |
| **X2** | IME 自有机制构造 GEMM + 净范式 2×2×M 消融(轨2) | 实验总纲 §3 轨2 | **重活 ≥ block-dot 构造,gated on [K-2] 原语;默认回退 = 轨1(结构+硅上正确)+ T5d 方法学,P7=open** |
| **X3** | 外部接入实录(M4) | [P-4]/[C1-4] | 文档足以让非核心作者独立接一个家族 |

---

## 里程碑(可机检 ↔ 论文档位;详见科研总纲 §6)

- **M1 = 证据线闭合**:E0–E8 绿(falsifier 组 + F-2′ + F-6 进 CI;C_attr^CT=100% + 装载期解析;schema.def v1 + 目录归拢;分母定稿 + 四指标脚本;C_dispatch(A 类)=100%、全局 ≥80%;编译期门自足)。→ **Reframe 档可动笔**。
- **M2 = 引擎线主推进**:G1–G6(C_construct 强义 ≥40%;泛型发射权威;zvfh+闭包;统一注册表;全局 C_dispatch ≥90%)。→ **投 TACO**。
- **M3**:P1–P4 + X1(家族#3 + ledger 第二点)+ ≥1 机制合成 prefill 赢 + 双板迁移常青;C_construct ≥70%。→ Mid 实证 / major revision。
- **M4**:X3 外部接入实录;C_construct ≥90%;(条件)X2 若达成 + AME。→ Full 冲刺 / 第二篇。

---

## Scale 诚实(拒 small-scope,非拒 slow-scope)

**multi-quarter 到 multi-year 工程。** 全 body-shape 强义构造 + 证据机检层全套 + 能力 live-on-silicon + resource cost + 一个实测 beat = 大工程。**两轴并行(用户定):证据线(E0–E8,便宜、先出量、关 C1/C2/C_attr 机检缺口)与引擎线(G1 body 强义构造主战场 + G3 SEL-1 先验层)一起做——目标是从一开始就是一个成熟 compiler,不是"先证据后引擎";硬件轴(P)与理想轴(X)排后 / gated。** 诚实说规模不是 hedging 野心;野心不带 scale 诚实就会"以为快到了"再次卡半途。

## Scope 纪律 / 保住 working

- **非推倒重写**:保 dequant production-e2e / IME N2 结构 / 3 强义 constructed 形状 / Fork B 已合并平面体 / 现有 lit —— 任一回归 = 停。
- **守 byte-exact 门 + falsifier**:每改必过(clean relink + BEFORE==AFTER + dependent md5 + 板级封存);任何触碰派发/边界的改动**必复跑 falsifier 零环**([NG-3] per-dispatch 永禁)。
- **spec 只写稳定契约**(零现值);一切数字(六态/覆盖率/LED/Win)留执行总纲 + CI + experiments/,携快照 ID([GOV-3])。
- 实现走 sub-agent(trellis-implement → trellis-check),主会话不写代码;每 pillar 落地后同步执行总纲现状 + 相应 experiments 表。

## Out of Scope

- **[NG-1]** 搜索式自动调优(仅外部 tuner 插点文档)· **[NG-2]** 图级/端到端框架 · **[NG-3]** per-dispatch 强制门 · **[NG-6]** upstream LLVM/MLIR 合入 · 前端 linalg/tosa · discrete-card offload。
- **X2 IME 自有 GEMM 明确不排进 M1/M2 前置**(重活、gated);**X-AME** 硬件 gated、论文不押注。

---

## 子任务分解草案(future children,待批才 spin;首批建议 = 证据线 E0–E8)

| 轴 | 草拟子任务 | 依赖 |
|---|---|---|
| 证据 | E0 越界清理(quick commit) · E2a kind 改名 · E1 schema.def+F-2′ · E2b 目录归拢 · E3 falsifier CI · E4 归因 JSONL+D-2a · E5 provenance 六态 · E6 覆盖率+ledger 脚本 · E7 D-1 自足 · E8 T-N+对手探针 | **E2a→E1**(改名先于契约固化);E1→E5/E6;E2b 解锁 F-3;E8 独立 |
| 引擎 | G1 body 模式库+注册表 · G2 K-3b 权威 · G3 SEL-1 先验层 · G4 K-2b 缺口环 · G5 zvfh+闭包 · G6 结构化参数+uarch | E5/E6 后;G3 先于 P7 |
| 硬件 | P1 资源 cost · P2 hwprobe+运行期链 · P3 两板实测 · P4 首 beat | E8/G1 后 |
| 理想 | X1 X-SCALAR owned 内核 · X2 IME 自有 GEMM(gated)· X3 外部接入 | G1 原语后;X 回退解耦 |

## Decision(ADR-lite)—— 已由 `/goal` 放行(2026-07-02)

**已定:** 范围 = 父级 program PRD(M1→M4)· **两轴并行(证据线 × 引擎线一起做,从一开始就是成熟 compiler)** · 理想项纳入 M3+ gated。
**放行后的修正执行序(本 session 定,写回 durable):**
- (B′) **修正首批序**——PRD 原稿把 E1 列进首批却漏了 E2(而 E2 改名须先于 E1),已解:**E0(quick commit,非 round)→ E2a kind 改名(第一个真 module,E1 前置)→ E1 schema.def v1 → E3/E4/E8 + E2b 目录归拢(单列,解锁 F-3)**。**G1(引擎线 C_construct 主战场)gated 在 E5/E6 之后**——六态自动读出 + 覆盖率脚本是度量 C_construct 的尺子,没尺子无法给 G1 打分;故证据线 E 系列先行**不违背**"两轴并行"用户指令(先造尺再拧旋钮)。G3(SEL-1)硬绑先于/同于 P7,不急于首批。
- (C′) **快照已重钉**:核查稿钉 7185a62b,现 HEAD=**7d781994**;E0 靶点(beat 措辞)已 grep 复验仍在(5 处,多出 `RVVGearboxSchedule.h:2809`)。动 E2a/E1 的锚点(ODS kind attr / `CapabilityModel.h`)前逐一 spot-check,不必重跑完整 [B-1..B-8]。
- **循环纪律**:此 program multi-quarter~multi-year,单 session 完不成;每轮把一个 coherent module 做到 durable(implement→check→commit→journal + 精确续接点),不浅做凑数。

### ⚖ 进度真理 = 燃减(2026-07-03 结构性刹车,用户+高级 AI 裁定)

**headline 进度指标从此只认两个数:`ΔC_construct(强义)` + `Δ手写 body LOC`(应为负)。** 证据轴产出(E 系列)一律记为**基建(造尺)**,单独列,**不计入"进展"**。真实重构的审美签名是**删除**:某手写 body 被删、由模式库原语构造替代、字节精确、六态翻 constructed——燃减曲线下行那天才是重构开始那天。

- **诚实起跑线(HEAD 20c1d714):`C_construct 强义 = 0/24`,`Δ手写 body LOC = 0`。E 系列 6 done 全是基建、燃减为零。**
- **硬约束(制度化担心,非靠记得):**每轮 module 引擎轴(改编译器本体能力):证据轴 ≥ **1:1**;引擎栏连续两轮为空 → 自动触发本类 review。
- **契约须有消费者:**声明工件(schema.def 等)须在同一里程碑内获得**第一个机器消费者**(被代码生成/校验),否则是海报不是契约。
- **玩具化四征兆**(自检):产物越来越 meta(度量度量的脚本/关于 spec 的 spec)· done 事后被降级 · 门禁零流量 · 口径一调再调而被测物纹丝不动。**成熟化四征兆:**kernel 本体 LOC 降而能力升 · 门禁拦到真实违规 · 同一机制出现第二个复用者 · 外人按文档独立做成一件事。
- **门禁流量制:**每 falsifier bounded-exit 加"故意提交一次违规、证明 CI 拦得住";季度零流量的门是删除候选。

### 落地进度台账(⚠ 证据轴 E 系列 = **基建/造尺,非"进展"**;每条带 commit)

| pillar | 状态 | commit | 备注 |
|---|---|---|---|
| **E0** beat 越界清理 | ✅ done | b128e3d5 | 5 处 "beats ggml ~13%" 未封断言清除;残留 3 处 "beat" 合规 |
| **E2a** kind→target_kind/region_kind | ✅ done | 6e16322b | 有界机械 rename;byte-exact 门 PASS;child task 已 completed;trellis-implement+check 双复核 |
| **E1** schema.def v1 + F-2′ 操作门 | ✅ done | c8f3e945 | 纯声明工件(schema/ canonical JSON + VERSIONLOG + .trellis/scripts/check_schema_gate.py);六项 target 形态;SHA256 653127c6;门 self-test 16/16;不动 C++(收敛=G6/E4/E5/P2) |
| _(spec 侧对齐)_ kind→闭合枚举+subclass | ✅ done | 93a71386 | 并发 spec 编辑巩固:ime/scalar 插件 kind 闭合枚举+subclass;core-invariants [S-5]①;tcrv-exec-contract target 消歧(了结 E2a deferred 语义张力) |
| **E4** 归因 JSONL+D-2a | ✅ done | 5824e30c | 附加式 option-gated JSONL(VariantSelection sink,四 SelectionKind)+ DeclaredInstanceHash helper + [D-2a] 编译期 stamp;**reason 编码=用户裁决选项2 `static_order`**(独立过渡值,不复用 prior;守卫字段删;能力键控做在主键;SEL-1 后 static_order→0=燃减信号);canon 四文档 reason 枚举扩四值;三 agent 6 验收全 PASS。**⚠ C_attr^CT 口径:E4 交付的是"跑起来即 100%"的机制,非已关闭的门——连续强制(across 覆盖分母)的门要 E3(CI 接线)+ E6(分母定稿)才关闭。别把此行读成 M1 硬门已满足。** |
| **E7** D-1 门自足 fail-closed | ✅ done | 63d7a0e4 | **⚠ 防御性硬化,非活 bug 修复**:PRD 原前提错——`VariantOp::verify`(ExecOps.cpp:841,adb57f8a)**已在 parse 期拒** kernel-unknown requires,`:150 continue` 正常不可达。E7 = silent-continue(fail-open)→ hard reject,使门独立于 verifier 兜底而自足([I7]/[D-1]);Unknown 硬拒不可 dispatch-guard;隔离 lit(disable-verifier)证自足。**load-bearing 取决于 F-5(E3)是否在未验证 IR 上跑该门——接 E3 须确认。** 双 agent 6 验收 PASS |
| **E6** 覆盖率+ledger 脚本 | ✅ done | 54465ee7 | G1 度量尺子(leverage)。schema/coverage-roster.v1.json(92 键分母,ggml pin 6eab471)+ coverage-sixstate.v1.json(手工六态,pending-E5)+ coverage_metrics.py + family_ledger.py(--self-test 12/12+7/7)。**honest 分母揭真相:vec_dot C_construct 强义=0/24、全局 C_dispatch 39/92=42%(远低 M1≥80%)**——in-code 24/24 口径是假象。IME ledger code_LOC 1866/2484;check 解开"659"谜=6 方言测试。**✅ 已裁(20c1d714):** test_LOC = 全量 1003;ggml pin 换 tag-anchored 锁文件(b9652@6eab471 可验证) |
| **M-FLAT 里程碑** 平面 block-dot 强构造家族(原"G1 切片1") | 🔨 **引擎轴领跑(多会话) M-FLAT 3/5** | 砖①②③ ✅ | **设计面板证伪"便宜切片"→用户批准转家族里程碑**(run wf_355dffa7-c51)。四面墙(fp16-scale opaque / DequantizeOp verifier 拒计算 scale / i32 累加器 / validator 不递归循环 region)= 净新增,q4_0 同墙、循环不能外提。**度量=[PAT-1] 注册表原语 `n/5 mechanized`**(`schema/pattern-registry.v1.json`,非 C_construct);**headline C_construct 保持 0/24 至砖⑤闭合+删弱 body→全平面家族台阶翻转**(不斜坡不折算)。**砖①②③ done**:①`block_fp16_scale_product`(fp16→f32 scale 重建,拆墙①)②`block_computed_scale_dequant`(拆墙②,零回归 DequantizeOp)③`cross_block_f32_accumulate`(f32 跨块累加,拆墙③,替 emitc.add)—各 build+lit+回归 347/347 净零;emit byte-identical 到 flat 家族 SumiTimesScales fold;C_construct 0/24 未动。裁决 `07-03-g1-q8-0-strong-construct/research/design-verdict-escalate.md` |
| E3 切片 falsifier 门进 CI + 红队 | ✅ done | (本轮) | F-2′ schema.def 门(check_schema_gate.py)→ `.github/workflows/falsifier-gate.yml` + 红队证明(2 门模式各故意违规→FAIL→复原,真实捕获输出)+ 8/8 re-runnable driver。门禁流量制落地。守本轮引擎:证据 1:1。**⚠ report --check 上 push 武装分支级 S-6 门:后续动 schema/ 的 agent(E4/E5/G6/P2)须同 PR 更 VERSIONLOG。** |
| E5 provenance 六态自动读出 + L-8 执法 | ⬜ **跟随(非领跑)** | — | 由 G1 切片 1 的真实构造器形状反向塑形 provenance;与 G1 同 module 收尾。[K-4]/[L-8];发射侧写 pattern-primitive ID 列表 → 六态脚本自动判强/弱义(替 E6 手工表)+ 弱充强 CI 拦。消 E6 sixstate 的 auto_readout:pending-E5 |
| G4 宽钳位双区 body | ⬜ **等硅(不本轮)** | — | 价值=关闭真实 perf 缺口+GAP-1 闭环,二者需 perf 复测=硬件+T-N 地板(E8 未建);本轮做只得 built-but-unclosed=反模式。待真在硅上测时做,不为它先建 E8 |
| E2b 目录归拢 / E3 falsifier CI / E8 T-N+对手探针 | ⬜ pending | — | **E3 已半就位**(coordinator F-1 manifest 判读规程 ce9c1c7c + E1 的 F-2′ 门 + E7 让 F-5 可依赖门自足;剩 .github + F-5 fuzz + F-6 独立性 + 每家族正则);E2b 解锁 F-3 |

**⚠ 并行 coordinator agent(同 branch):** 有第二个 agent 做**只读 spec 审计修复**(ce9c1c7c/6a4a2f4f;协调 note = `SPEC-AUDIT-FIXES-NOTE.md`)。分工:它修 spec/docs 矛盾(kind 闭合枚举/profile-provider 谓词/[F-1] manifest 判读规程/[X-SCALAR] popcount 护栏/[SEL-2] forced-stub 解锁),留 E1-S5/E6-LED/G1-PAT 字段 schema 给我(declare-once 在总纲)。**动 spec 前先 re-read**(文件可能在我编辑间被改);频繁 commit + 验证已在处理并发,无 lost-update(已核 E4 canon 编辑与 audit-fix 共存)。

**下轮续接点(2026-07-03 引擎轴领跑 → M-FLAT 里程碑,设计面板证伪切片后):** 刹车 = 真删/真建引擎本体,不写治理文档(那是玩具化征兆①)。**"便宜切片"经代码级核验不存在**——整个平面 block-dot 家族的强构造是一个多会话净新增-ODS 里程碑(四面墙,q4_0 同墙,循环不能外提)。用户批准转 **M-FLAT 家族里程碑**,引擎轴领跑,守 1:1:

- **M-FLAT 里程碑(引擎轴领跑,多会话):** 5 砖 = ①fp16-scale 重建原语 ②computed-scale dequant ③f32 跨块累加 op ④块循环 body op+loop-aware validator ⑤闭合(接线+删弱 body+全家台阶)。**度量 = [PAT-1] 注册表 `n/5 mechanized`**(canon,非 C_construct);**headline C_construct 保持 0/24 至砖⑤**(台阶不斜坡,绝不折算)。**本轮真工作 = 砖①**(fp16→f32 per-block scale 重建 typed 原语,ODS+verifier+emitc lowering+emit-consistency lit,字节精确 f32⊇fp16),子任务 `07-03-m-flat-brick1-fp16-scale`。DequantDot 单块档 = 砖①②的集成插座(非替代)。
- **证据位(本轮守 1:1):** E3 切片 = 把**一个** falsifier/门(F-2′ schema.def 门 或 falsifier manifest)做成 CI-可执行形 + **红队证明**(故意提交违规、CI 拦得住、复原)= 门禁流量制。触 .github + test/,不碰砖① 的 lib/Dialect/RVV。
- **E5 provenance / G4 宽钳位 / E2b / E8 排后**:E5 由第一个真实构造器(砖④/⑤)形状反塑,不先造格式;G4 等硅(需 perf 闭环+T-N,本轮做只得 built-but-unclosed);E2b blast radius 大慎做;E8 独立。
- **并行写入者纪律**([[full-refactor-program-active]]):动 spec/共享文件前 re-read,不 commit 对方 in-flight 编辑。

**✅ 两裁决已落(20c1d714,非阻塞项清空):** ① `test_LOC` = 全量 1003(方言 659+lowering 344+e2e 0),口径在 family_ledger.py 定死、所有家族一致;② ggml pin 换成 tag-anchored 锁文件(b9652 @ 6eab471,schema/ggml-pin.lock.json + fetch_ggml_pin.py 可验证,epoch 1 树未移)。

## Technical Notes

三总纲 = 权威(docs/);spec 已吸收(.trellis/spec/);experiments/ 空表已建。执行总纲 §10 TOP-10 = 证据线 E0–E8 的落地清单;实验总纲 §6 前置工具 = 同一批。**三条术语定案已入 canon+spec:** ① Win 登记阶梯 A/S(sanity)/B(kernel 级,B1/B2 由对手探针定)/C(相级 e2e 过 [PERF-1]),Win-S 为历史误标 "Win-B" 的 alias(登记簿留注、不改史);② `kind` 闭合枚举 + `subclass` 保原始类目(schema.def ①,minor),profile 只作 provider(不带 kind)、declared-instance-hash 对**展开事实集**取(profile ≡ 显式列表哈希相同);③ op-attribute kind → `target_kind`/`region_kind`,随 E2 目录归拢、先于 E1 schema.def v1。**一次核查快照 = 7185a62b(执行总纲);仓库已前进(现 6e2e4e56 + 后续),spin 子任务前须重钉快照重跑基线确认([A-2])。**
