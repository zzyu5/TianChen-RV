# full-refactor 母 program pillar 对账 — /goal 真欠地图（2026-07-13 · 只读对账）

> **性质**：纯 read-only 对账分析（禁改 code/spec/schema·禁 git commit/add）。把 PRD 母 program 的 pillar 逐条对账**战役实际进度**（非 formal task.json），识别 `/goal` 真正剩余的高价值核心工作。
> **快照锚**：branch `refactor/full-refactor-m1` · HEAD `ef3720af`（recon 机检 repo_snapshot 一致）· perf-covered **7/83** · C_construct 强义 **84/91=92.31%** · M4 certified 84/91·blocked_on_IME=0·reconciliation_ok=True。
> **对账口径**：formal `task.json` = 13 completed / 4 in_progress / 20 planning，**大面积 stale**（母 PRD `[DEBT-TASK]` 已自认）；本报告以 **git log + docs/reports 战役报告 + falsifier-gate.yml + recon 机检脚本** 为准。
> **定调（承 2026-07-11/12 纠偏）**：**结构轴已收口 ∧ 测量轴近 headline ceiling**。2026-07-12 战略 recon 的 #1 剩余项（[P-4] 外部接入实录）**在其后已 DONE-via-campaign**（Tier-2 clean-room `2fec2471`）；本报告是该 recon 之后的刷新。

---

## 0. 一句话结论

**结构脊柱（M1 证据线 + M2 引擎线 + M4 收口 + C1 外部接入协议）实质全部收口**：四覆盖率里程碑 [COV-3] M1/M2/M3/M4 门**全部达标**（C_construct 强义 92.31% ≥ M4 的 90%）；falsifier [F-1..F-6] **六门全进 CI**；C1 头牌外部接入实录 **Tier-2 正面兑现达标**。**真-remaining 已非结构工程，而是三类低-中价值长尾**：① C1/C2 的**便宜 canon/协议收口**（<300 口径必问 + GAP-A 第二注册点·合计最短 distance-to-C）；② perf/emitter **成熟度长尾**（K-quant S6 tiling · G5 未接线 22 格·皆 micro↛e2e 约束、只能机制名义）；③ **P1 资源感知 cost**（唯一从未交付的 PRD pillar，但项目已论证其 gilding、低价值）。**没有大块未做的核心结构工作。**

---

## 1. 逐 pillar 状态表

> 图例：**DONE**=战役已落（引证据）· **REMAINING**=真欠（写 bounded-exit 差距）· **SUPERSEDED**=被战役 subsume/重定义。
> ⚠ **codename 碰撞**：PRD 引擎轴表的 `G4`/`G5` 指**具体 pillar**（宽钳位 / zvfh）；而 git/memory 的 `G4`/`G5` 是**战役代号**（G4=IME 家族战役=映射 X2；G5=接线战役=映射 P3/实验）。下表按 **PRD pillar 语义**判，碰撞处显式标注。

### 1.1 证据轴 E0–E8（M1）

| pillar | 判定 | 证据指针 | REMAINING 的 bounded-exit 差距 |
|---|---|---|---|
| **E0** beat 越界清理 | **DONE** | `b128e3d5`（5 处未封 beat 断言清除） | — |
| **E1** schema.def v1 + F-2′ | **DONE**（= [S-5] 工件化） | `c8f3e945` · schema.def 实现为 `schema/*.v1.json` canonical + `schema/VERSIONLOG.md` · CI job `schema-def-gate`（self-test 16/16 + 红队 + F-2′ 操作门 diff∩schema=∅） | — |
| **E2a** kind→target_kind/region_kind | **DONE** | `6e16322b`（有界机械 rename·byte-exact 门 PASS）+ spec `93a71386`（kind 闭合枚举） | — |
| **E2b** 家族目录归拢 | **SUPERSEDED** | 用户裁「MLIR 分层为主 + family 清单机检」（FALSIFIER-INDEX F-3 行·「gated on 目录归拢」前提 **WITHDRAWN**） | 跨 4 lib 根大搬迁**不做**（违 MLIR 生态惯例）；F-3 收容改由 `schema/family-manifest.v1.json` + `check_family_locality.py` 机检保证。物理归拢无价值。 |
| **E3** falsifier 组进 CI | **DONE** | `460790d7`·`ccb7c226`·`61596dcc` · `falsifier-gate.yml` **[F-1..F-6] 六门 + 6 相邻门全进 CI**（schema-def/opponent-pin/monolith-retire/frontdoor-provenance/cert-requirements/retired-index/f6/f5/f1/f4/f3）+ 红队流量证明 | — |
| **E4** 归因 JSONL + D-2a | **DONE（编译期）· partial（装载期）** | `5824e30c`（VariantSelection JSONL sink，reason=static_order）+ F-4 门 `f4-attribution-jsonl` 进 CI | **装载期 D-2a**（每进程 1 条解析记录）+ 调度/合法性阶段归因 = M2 增量·未做（C_attr load=0）。非 M1 门、非高价值。 |
| **E5** provenance 六态自动读出 + L-8 | **partial** | `coverage-sixstate.v1.json` + `e5_strong_readout.py` · C_construct 强义 84/91 机检对账（missing_sixstate_keys=[]·reconciliation_ok） | 强/弱义标签现为**手工赋值**（`all_strong_pending_e5=False`·`pending_e5_keys=0`）；**从 provenance manifest 自动读出**的执法未接。不阻塞任何门（M4 门已由手工强义标签 + 机检零未定义达成）。低价值。 |
| **E6** 覆盖率 + ledger 脚本 | **DONE** | `54465ee7` · `coverage_metrics.py`（四指标 report）+ `family_ledger.py`（--self-test 12/12+7/7）+ ggml pin tag-anchored 锁文件 | — |
| **E7** 编译期门自足 fail-closed | **DONE** | `63d7a0e4`（silent-continue→hard reject·[I7]/[D-1]·隔离 lit 证自足） | — |
| **E8** T-N 噪声地板 + 对手解析探针 | **DONE-via-campaign** | opponent-facts-pin 门在 CI（`check_opponent_facts_pin.sh`）· `opponent_map` 工件 · T8 台账 205 行 · [CASE-COMPILER-ASYMMETRY]（板出货编译器身份探针） | — |

**M1 里程碑 = ✅ 完成**（按燃减纪律记「基建/造尺」，工件全落；F-1/F-2′/F-5/F-6/F-4/F-3 六门进 CI = M1 证据线闭合）。

### 1.2 引擎轴 G1–G6（M2）

| pillar | 判定 | 证据指针 | REMAINING 差距 |
|---|---|---|---|
| **G1** body 模式库 + 统一注册表（C_construct 强义 ≥40%） | **DONE**（远超门） | M-FLAT 家族 + `schema/pattern-registry.v1.json`（PAT-1..3 作数据）· **C_construct 强义 = 84/91 = 92.31%**（vs M2 门 ≥40%）· certified 84/91 | — |
| **G2** 泛型操作数层唯一发射权威 | **DONE（结构）** | `g2-fuse-rms-norm-mul` 实验·泛型层接管块点积 | ⚠ e2e null（norm 占 decode 0.05%·[G3 四问] micro↛e2e 档案级教材）——结构达成、性能无传导（**非失败**，Amdahl 上限 < 噪声）。 |
| **G3** SEL-1 能力先验层 | **DONE** | SEL-1-T5 先验（`2942f603`）· T4b 选择器四配置消融（`13706e79`）· attribution reason=static_order（SEL-1 后→0 = 燃减信号） | — |
| **G4**（PRD）[K-2b] 宽钳位双区 body | **REMAINING（parked·低价值）** | PRD 台账「⬜ 等硅」 | 需 perf 复测（硬件 + T-N 地板）才闭 GAP-1 环；本轮做只得 built-but-unclosed（反模式）。**低价值 · 等硅**。⚠ 勿与 git 的「G4 战役=IME 家族」混（后者=X2·已 DONE）。 |
| **G5**（PRD）X-ZVFH 接入 + [S-2] 闭包 | **DONE** | `rvv.zvfh`/`rvv.zvfhmin` 已注册（`RVVCapabilityProfile.cpp:57-60` + 传递闭包 zvfh→zvfhmin→zve32f）· zvfh f16 board 实测 DONE（`0219d816`/`abf57268`） | — ⚠ 勿与 git 的「G5 战役=接线」混（后者=P3/实验轴）。 |
| **G6** 结构化参数 + provenance/trust + kind 闭合枚举 + uarch quirk 表 | **partial** | kind 闭合枚举 + subclass DONE（`93a71386`） | **uarch quirk 表**（供 P6 键控）+ params 命名空间化 = 需主会核实是否落地；低价值（无消费者则玩具化征兆①）。 |

**M2 里程碑 = ✅ 实质完成**（C_construct 强义 92.31% ≫ 40%·泛型发射权威·zvfh+闭包·注册表作数据·全局 C_dispatch 93.41% ≥ 90%）。

### 1.3 硬件/性能轴 P1–P4（M3）

| pillar | 判定 | 证据指针 | REMAINING 差距 |
|---|---|---|---|
| **P1** 资源感知 cost（读真 VLEN/ELEN/vreg/mask/tail） | **REMAINING（唯一从未交付 · 但已论证低价值）** | `estimateVariantCost` 现为 **SEL-1 能力先验**（单一 base cost·"no runtime performance claim"）·`computeBlockDotShapeCostCore` 自述 capability-blind | bounded-exit = cost 读真资源谓词且隔离、重测校定。**但** memory `backend-maturity` 裁「capability-blind BY DESIGN 非 gap」（live path=离线 memoization·cold argmin 永不决定生产选择·Win-A upside 是 emitter-gated 非 cost-ranking-gated）→ 追此 = **gilding 风险**。诚实缺口、非高价值杠杆。 |
| **P2** hwprobe 运行期链 + C_attr^RT | **REMAINING** | — | provenance=hwprobe/trust=measured 未建·C_attr^RT=0·装载期解析记录（D-2a）未接。M3/M4 门不强制（M4 仅「若 D-2b 完成」启用 C_attr^RT）。中-低价值。 |
| **P3** 两板 micro+e2e 实测（T3/T6） | **DONE-via-campaign** | T3 板 126 行 · T8 205 行 · 双板异板 · G5-wiring 板批 · perf-covered recon 机检入 CI | — |
| **P4** 首个机制合成 prefill 赢过八门 | **partial-DONE** | **sealed Win-K1-VLEN RATIFIED**（`SEALED-WIN-REGISTRY.md`·k1 vl=16 满宽 1.085× 编译器对称 clean）· q4_0 5.9× prefill（⚠ 大半 routing 白嫖·非 kernel 质量赢） | 多数 kernel 赢 ④门 = micro-only（e2e washes·kernel-wins-dont-transplant）；「过完整八门」的通用 Win-B 仍受 micro↛e2e 约束。已有 1 个 sealed win（够 M3「≥1 机制合成赢」门的最低义）。 |

**M3 里程碑 = ⚠ 部分**：C_construct ≥70% ✅（92.31%）· ≥1 sealed win ✅（Win-K1-VLEN）· 双板迁移常青 ✅（[XFER-1] 7/7）· **P1 资源 cost / P2 hwprobe 运行期链未建**（低-中价值）。

### 1.4 理想/存在性轴 X1–X3（M3+）+ M4 收口

| pillar | 判定 | 证据指针 | REMAINING 差距 |
|---|---|---|---|
| **X1** X-SCALAR 独立家族广度见证 | **DONE（家族本体）· partial（收口）** | owned 内核落地：`tq2_0_q8_k_vec_dot`（`f96f767a`·三值 2-bit·NO popcount）+ `dequantize_row_q4_0`（`2dd654d8`）· **F-6 双断言机检 + 独立门在 CI**（判据④ landed·闭包∩rvv.*=∅）· C2 第二点 1501 raw（LED-2） | **XS-M0 canon 正名**（执行总纲/FALSIFIER 旧「空 STUB」状态栏 stale·docs 债）+ **<300 口径 canon 裁**（实测 1501 vs canon [C2-1] <300 = 4-5×·见 §2-C2）。均 **canon-必问**·小。 |
| **X2** IME 自有机制构造 GEMM | **DONE** | 三格 silicon-sealed：`gemm_tile/{q4_0,q8_0,q4_K}@ime`（stock gcc-13+token·vmadot 0xe210312b·0-diff）· forward-wired 全 3/3（flat/int8-direct/super-block）· M4 blocked_on_IME 3→**0**（certified 81→84） | — （X2「重活 gated」在 M4 收口内兑现；IME perf=N3 长尾另计） |
| **X3** 外部接入实录（M4·[P-4]·[C1-4]） | **DONE-via-campaign** | **Tier-2 clean-room 正面兑现达标**（`2fec2471`）：非协议作者 agent（forbidden-peek=0）真接 `weft_demo` 家族→**build 绿**（weft-opt 258/258·zero core edit 除注册表）+ **falsifier 六门 + 相邻门全 GREEN**（14/14·F-3 ratchet 40→46）+ **触碰集符 [F-3]** · Tier-1 followability + 8-gap fixed（`98c6fafe`） | **GAP-A**（own-EmitC-backend 家族须第二注册点 `BuiltinBackendEmitters.cpp`·未在 [GAP-P4-REGISTER] recipe / `shared_allowances`·真协议缺口·本演练以 ③reuse-existing 规避）+ **GAP-B/C**（文档可发现性次缺）+ **Tier-3 真第三方**（optional·消除同栖仓库 in-house 偏置）。 |

**M4 里程碑 = ✅ 实质达成**：C_construct ≥90% ✅（92.31%·M4 三分类零未定义格）· [P-4] 外部接入实录 ✅（Tier-2）· X2 IME GEMM ✅。剩 GAP-A（协议完整性小缺口）+ Tier-3（optional）。

---

## 2. 三交付指标现状

### 指标① 成熟 compiler 覆盖率（[COV-3] 逐门对账）

`coverage_metrics.py report`（HEAD `ef3720af`·分母 91·A 类 82/B 类 9/C 类 0）：

| 指标 | 现值 | [COV-3] 里程碑 target | 判定 |
|---|---|---|---|
| **C_dispatch 全局** | **85/91 = 93.41%** | M1 ≥80% · M2 ≥90% | ✅ 全达标 |
| C_dispatch A 类 | 76/82 = 92.68% | M1 A 类 = 100% | ⚠ 92.68%（差 6 格·= 声明例外/负控格·非真欠·见下） |
| **C_construct 强义（燃减主指标）** | **84/91 = 92.31%** | M2 ≥40% · M3 ≥70% · **M4 ≥90%** | ✅ **M4 门达标**（92.31% ≥ 90%） |
| C_construct+ 弱义 | 85/91 = 93.41% | 过渡·不 gate | — |
| C_attr | ^CT partial（E4 in-IR + F-4 JSONL）· load 0 · ^RT 0 | M1 C_attr^CT=100% | ⚠ 编译期归因 F-4 已进 CI（脚本 note stale 写「no JSONL yet」是旧值）；装载期/^RT=0（M2+/P2 增量） |

**关键判定（回答 /goal Q1）**：任务预设「T8 提 ~40-41」= **stale**（旧 in-code 或旧分母口径）。**当前 C_construct 强义现值 = 84/91 = 92.31%，已超 [COV-3′] M4 target（≥90%）**。A 类 92.68%（非 100%）的 6 格差 = M4 三分类里的 `声明例外 7`（含 `vec_dot/mxfp4` codified ZERO-MODEL 负控 + 6 个 `gemm_tile/{q1_0,iq1_s,iq1_m,iq3_xxs,iq3_s,nvfp4}@rvv`·全因 ggml 无 repack-GEMM 参考层 0-hit 或 board 证 LOSS）——**这些是「有据拒做/外部依赖」的声明例外，非真欠格**。要它们转 certified 须 **ggml 上游建 repack-GEMM 或 board 翻转 LOSS**（外部依赖，非我方工作）。**覆盖率轴 = 实质 DONE**。

### 指标② novelty C1/C2/C3′ 逐点缺口（回答 /goal Q2）

| 贡献 | 闭合度 | 已闭合 | **真欠缺口** |
|---|---|---|---|
| **C1（头牌·模板协议本体）** | **结构最强·实质收口** | 合取机检(T1)·[F-1..F-6] 全 CI·schema.def 逐 PR 审计([S-5]✅)·**跨范式**(RVV 向量→IME 矩阵 MAC forward-wired)·**跨独立家族**(X-SCALAR 向量缺席·F-6 判据④ landed)·**外部接入 Tier-2 正面兑现**([C1-4]✅) | **[S-5]/[X-SCALAR] 均已做**（任务疑问点厘清：schema 工件化=`schema/*.v1.json`+F-2′门 DONE；X-SCALAR owned 内核+F-6 双断言 DONE）。剩 **GAP-A**（第二注册点未纳 recipe/allowances·own-backend 家族真缺口·小）+ **Tier-3**（optional）。 |
| **C2（模板经济学·边际成本）** | **最薄·曲线出图但口径必问** | **LED-2 三家族曲线出图**（`c46678e1`）：RVV substrate(163629·不可比原点)/IME 2484(anchor)/X-SCALAR 1501(≈60%)·四列住址分解·family_ledger.py 自动可复算 | **[LED-2] 是「3 家族多样性点」但仅 2 个 apples-to-apples 成本点**（IME/X-SCALAR·RVV=substrate 非 onboarding delta）。**X-SCALAR 是 full 独立家族接入**（非投影点·付真 emitter 成本 907 raw）。**zvfh 子扩展点未登记为 C2 ledger row**（事实已注册但未量成本点）。**★<300 口径 canon 必问**：canon [C2-1] <300 vs 实测 1501 = 4-5×；LED-2 已给「两 regime」reframe（独立家族付真 emitter 成本 / 集成子扩展近零）= 比 <300 更可辩护，但须 canon 裁决落地。 |
| **C3′（模板产出质量·能力键控模式库）** | **强-部分** | 注册表作数据(PAT-1..3)·归因 JSONL(E4/F-4 CI)·**SEL-1 能力先验层 DONE**(G3·2942f603)·[XFER-1] 7/7 迁移命中·T3p 模式消融+T4b 选择器消融·LAW-FIRST 4 例·perf 证词(q4_0 5.9×/Win-K1-VLEN 1.085×/IME 2.09×/q5_K 1.641×/FLAT 全绿) | perf-covered 低(7/83)·多数 kernel 赢不传导 e2e(micro↛e2e)·vs-框架同-ISA beat 仅少数格·body 强义构造 = C_construct 92.31% 已覆盖 ≥70% kernel 体([C3-1] 门达标)。**剩 = perf/emitter 长尾**（见指标③）。 |

**SEL-1/attribution 均已做**（任务疑问点厘清）。C3′「统一模式注册表 + 归因 JSONL + 能力先验层 + body 强义」四缺口**全部闭合**；C3′ 剩余纯 perf 传导长尾。

### 指标③ 性能实验 T-tables 欠账清单（回答 /goal Q3）

| 项 | 状态 | 欠账 |
|---|---|---|
| **perf-covered 7/83（头条测量轴）** | recon 机检零未定义格 | 分类：绿 7·黄-未接线 **22**·黄-传导稀释 3·黄-物理墙 12·黄-对手更强 12·声明例外 27·Σ83。**最大欠账 = 黄-未接线 22**（emitted 核未接 ggml forward·唯一拉绿杠杆=G5 接线战役·但**易得绿尽**·剩多产黄格带账非新绿） |
| 铺面① FLAT | ✅ | 5 格 gcc-symmetric 幸存 |
| 铺面② K-quant 对称重填 | **partial** | q5_K/q4_K e2e 在 k1-clang 传导·撤回的 S6 数未按对称编译器重填·iq/tq 8 行全 LOSS |
| 铺面③ 流式抽样 | **partial** | 8 格·三具名 GAP triage 完（`813f6462`） |
| 铺面④ T4b 选择器四配置消融 | ✅ | `13706e79` |
| 铺面⑤ T3p 模式消融 | ✅ | `658e5c0f` |
| **X-SCALAR/zvfh perf** | zvfh f16 board 实测 DONE | X-SCALAR 标量核 board bit-exact = pending-hardware（非阻塞 emit 轴） |
| **T5d（厂商路径对照）** | **0 行·defer** | 未产出（低优先·[NG-4] beat 未过门禁措辞） |
| **T4a（归因 JSONL 表）** | **absent** | JSONL 出口在位（E4）但表未成形（低价值） |
| **T6（整模型 e2e 分相双板）** | landed·决定性 negative | 多数格新绿 0（接线 gap + micro↛e2e 封顶） |

---

## 3. ★真-remaining 高价值工作 shortlist（排序）

> 排序键 = **distance-to-C1/C2/C3′ × (价值 / 成本)**·trunk-discipline。perf 轴近 ceiling → 倾向**便宜的结构/协议/canon 收口** > 更多 perf 打磨。

| # | 项 | 是什么 | 推进 | bounded-exit（可机检状态） | 工作量 | 板？ | 可逆？ | canon-必问？ |
|---|---|---|---|---|---|---|---|---|
| **1** | **C2 <300 口径 canon 裁 + LED-2「两 regime」reframe 落地** | 用 LED-2 的「独立家族付真 emitter 成本 / 集成子扩展近零」reframe 替换 canon [C2-1] 的「<300」直觉；X-SCALAR 1501 作真数据点；zvfh 可作第 3 子扩展成本点 | **C2（最薄）** | canon [C2-1] 措辞更新 + ledger 口径一致 + recon 对账绿 | 小（docs+裁决） | 否 | 是 | **是**（措辞宪法级·必问） |
| **2** | **GAP-A 收口**（协议完整性） | 把 `BuiltinBackendEmitters.cpp` 的 `kBuiltinBackendEmitters[]` 行加入 `family-manifest.v1.json` `shared_allowances.plugin_registration` + 补 [GAP-P4-REGISTER] recipe 第二注册点 | **C1（头牌协议完整性）** | own-EmitC-backend 家族接入可保 [F-3] 触碰集清洁·falsifier 全绿 | 小（schema+docs） | 否 | 是 | **是**（schema/canon 域·必问） |
| **3** | **XS-M0 canon 正名** | 执行总纲/FALSIFIER 旧「X-SCALAR 空 STUB」状态栏 → 「owned 内核已落地」（docs-代码同步债） | C1/C2 诚实性 | 状态栏与代码事实一致·check_docs_canon 绿 | 小（docs） | 否 | 是 | 是（canon 措辞·可与 #1 打包一次裁） |
| **4** | **K-quant S6 tiling 成熟度**（`07-08-G3-L1-maturity`） | 全展开 regfile spill → 能力键控 tiling 原语 | 成熟度 + C3′（**仅机制/方法学名义**） | pattern-registry 新 tiling 原语 mechanized + objdump 证 spill 消 | **高**（多会话·regfile spill 难） | 是 | 是 | 否（自决·测量总攻队列内） |
| **5** | **G5-wiring 续**（未接线 22 格 + IME session） | 真 ggml 数据路径接线 + forward hook + e2e A==B | 实验③ + C1 跨范式广度 | 零未定义格 completion·新绿或黄格带账 | 低-中 | 是 | 是 | 否（G5 既定队列·预注册免回门） |
| **6** | **Tier-3 真外部第三方接入** | 真跨组织作者按协议独立接一家族 | C1（消除 in-house 偏置·最强证据） | 真第三方 build 绿 + 六门 | 高（需外部渠道） | 否 | 是 | 否（optional·挂钩子不立任务） |
| **7** | **P1 资源感知 cost** | cost 读真 VLEN/ELEN/vreg/mask/tail | 成熟度（PRD 唯一未交付 pillar） | cost 资源感知且隔离·重测校定 | 中-高 | 是 | 是 | 否 | **⚠ 已论证 gilding·低价值**（capability-blind BY DESIGN·cold argmin 不决定生产选择） |

**已 SUPERSEDED / OVERTAKEN 不必做**：E2b 物理目录归拢（清单机检取代）· `retire-q1_0-monolith`（q1_0 dequant certified `6dcb5db4`·monolith-retire 门在 CI·需主会核 whitelist 是否仍挂 q1_0 vec_dot）· `07-08-G3-frontdoor` 旁路清零（已完成·frontdoor-provenance 门在 CI）· `line-A/C/D`（框架旧·被测量总攻取代）· `construct-gemm-tile-m1`（被 X2 IME 家族取代）· `wire-opponent-facts-pin-ci`（opponent-facts-pin 门**已在 CI**=DONE-via-campaign）。

---

## 4. 6 个 formal sub-task 逐个裁定（回答 /goal Q5）

| formal sub-task | task.json | 裁定 | 依据 |
|---|---|---|---|
| `07-07-line-A-board` | planning | **SUPERSEDED** | 板测量走测量总攻 + G5 板批；框架旧 |
| `07-07-line-C-coverage` | in_progress | **大半 SUPERSEDED** | certified 84/91；剩流式/前向抽样归入铺面③ |
| `07-07-line-D-evidence` | planning | **SUPERSEDED** | 被「测量总攻 + G5」取代 |
| `07-07-fix-blockdot-zfh-packager` | planning | **降级·parked（真 gap 但 deliberate）** | zvfh packager 硬编码=真 caveat 但低紧急（无非-zvfh 目标·fix 会重引 softfloat·XS-M2 已 DEFER）·**非 superseded，是 parked** |
| `07-07-retire-q1_0-monolith` | planning | **OVERTAKEN·需主会核** | q1_0 dequant certified（`6dcb5db4`）·monolith-retire 门在 CI；**主会核**：`schema/monolith-retire-whitelist.v1.json` 是否仍挂 q1_0 vec_dot monolith（若挂=真尾巴、若已退=OVERTAKEN） |
| `07-07-wire-opponent-facts-pin-ci` | planning | **DONE-via-campaign** | `opponent-facts-pin` job **已在 `falsifier-gate.yml`**（`check_opponent_facts_pin.sh`·ggml-pin bump→STALE-fail）；formal task 未 close = task.json stale |

**结论**：6 个中 **4 SUPERSEDED/OVERTAKEN·1 DONE-via-campaign·1 parked（低价值）**——**无一 genuinely-needed 高价值**。formal task.json 的 planning 状态全 stale。

---

## 5. 明确推荐下一项（trunk-discipline）

**推荐 = shortlist #1 + #2 + #3 打包成「一次 canon-必问 + 协议小收口」批**（C2 <300 口径裁 + GAP-A 第二注册点 + XS-M0 正名）。

**理由（最短 distance-to-C）**：
1. **结构脊柱已收口**——M1/M2/M3/M4 覆盖率门全达标（C_construct 92.31%）、falsifier 六门进 CI、C1 外部接入 Tier-2 正面兑现。**没有大块未做的核心结构工程**；继续磨 perf-covered（撞未接线 gap + micro↛e2e）是**低边际、非 trunk**。
2. **C2 是三贡献最薄的证据链**，且其收口**不是工程而是 canon 裁决**——LED-2 曲线已出图，只差把「<300」直觉换成更可辩护的「两 regime」reframe。这是**单项最便宜的 novelty 加固**。
3. **GAP-A 是 C1 头牌协议的真完整性缺口**（Tier-2 演练暴露·own-backend 家族无法保 [F-3] 清洁），修法明确（镜像既有 blessing）、成本小、闭 C1 最后一个协议毛刺。
4. 三项**同域（docs/schema/canon 措辞）、可逆、文件集不相交**，可一次批处理、一次报用户裁。

**权限归类**：#1（<300 口径）+ #3（XS-M0 正名）+ #2（allowances 改）**均 canon/schema 级 = 必问用户**（措辞宪法 / 门定义变更）。按决策权限卡，这**不是**新战役（既定 pillar 收口），但**触碰 canon 定义 = 必问裁决**——正确动作 = **打包成一次 canon-必问，列清 4 个待裁点（<300 口径 reframe · GAP-A allowances · XS-M0 状态栏 · 可选 zvfh 第 3 成本点），请用户一次裁**，而非自决改 canon。

**若用户偏好自决工程推进**（不走 canon-必问）：唯一 LIVE 的自决项 = **shortlist #4 K-quant S6 tiling**（机制/方法学名义·测量总攻队列内·预注册免回门），但**须 objdump-first Amdahl 预估**——memory 判 q3_K/q6_K@k1 即便修好未必传导 e2e（clang 亦不传导），若 < 噪声地板则**禁 perf 名义、只能机制名义立项**。高成本、低 perf-payoff，价值次于 #1-#3 的便宜 novelty 收口。

**禁停机**：队列非空（shortlist 7 项 + 长尾）；到达此对账里程碑的正确动作 = 报备 + 请用户裁 #1-#3 canon 批的排序，或按权限卡取 #4 自决直行——**不暂停**。

---

## 附：证据源清单（本对账所据）

- PRD：`.trellis/tasks/07-02-full-refactor/prd.md`（三交付指标 + E0-E8/G1-G6/P1-P4/X1-X3 pillar + [COV-3] 里程碑门）
- recon 机检：`perf_covered_metrics.py report`（7/83·reconciliation_ok=True）· `coverage_metrics.py report`（C_construct 强义 84/91=92.31%·m4_classification certified 84·blocked_on_IME 0）
- CI：`.github/workflows/falsifier-gate.yml`（[F-1..F-6] 六门 + 6 相邻门）· `docs/method/FALSIFIER-INDEX.md`（[C1-4] Tier-2 达标登记）
- 战役报告：`2026-07-12-P4-Tier-2-cleanroom-接入演练-T1c.md`（外部接入实录）· `2026-07-12-C2-LED-2-边际曲线-三家族点.md`（C2 曲线）· `2026-07-11-M4-三分类终态全表.md`（M4 真 100%）· `2026-07-11-X-SCALAR-AUDIT后续-排期报告.md`（X-SCALAR 状态 + <300 口径缺口）· `2026-07-12-full-refactor-战略-recon-下一战役.md`（前一日 recon·#1 项已 overtaken）
- spec：`.trellis/spec/index.md`（C1/C2/C3′ 终态 + 证据门）
- git log `ef3720af..`（Tier-2/RENAME/C2-LED-2/IME-triple/perf-covered 战役 commit）
