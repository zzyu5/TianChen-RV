# A5 · 独立复核报告（收口·§四.3 复验制·纯案头·不改任何文件）

> **复核员**：A5 独立复核员（施工方自证无效·独立验工件·不重跑板测·不改任何文件）。
> **对象**：G8 全量 0.8 攻坚交付物（A2 batch1-8 + A2.5 + 线 B + recon + 收口报告）。
> **日期**：2026-07-16 · 度假自治连续推进。
> **判决基线**：诚实第一·抓到错就报错·没错就说没错·禁附和。
> **总评**：**过（带 3 项瑕疵·0 头条级错误·0 头条级虚高）**。四档 Σ=分母机算真过、成色诚实纪律守得极稳、5 个 C3′ 负结果坐实、禁继承/禁互推兑现、度假自治零越权。瑕疵集中在 recon 脚本 2 处 hardcoded 显示串 + T3 note 列 2 格 hand-brick 成色过标——**均不污染收口报告头条**。

---

## ① 逐项复核结论（过/瑕疵/错）

### 1. recon 完整性 —— **过（带瑕疵 F-1）**
- **Σ四档=板分母（机算·非 hardcode）**：`recon_master_rebuild.py` 机算 `tot==denom[board]`（line 382/407·stats 逐行累加）→ **rvv 102 ✓ / k1 105 ✓**。真机算、真过。
- **certified**：`coverage_metrics.py report` → `m4_classification.certified=101 / denominator=108 / reconciliation_ok=true`（undefined_cells=[]）。**101/108 reconciliation_ok 坐实**。收口报告 §五 "84/91→101/108" 正确。
- **perf-covered**：`perf_covered_metrics.py report` → `9/83·reconciliation_ok=true·three_source_consistent=true`（headline=classification=ledger=9·anti_gate_ok=true·49 声明例外全 certified）。**9/83 坐实**。
- **⚠ 瑕疵 F-1（三 recon 有矛盾）**：`recon_master_rebuild.py:422` **hardcode** `certified="84/91"`，注记却写 "coverage_metrics.py"。但 coverage_metrics.py 现输出 **101/108**（regime-split 〇.1 后 roster 93→110 已改）。→ **recon_master_rebuild 数字字典打印的 84/91 是 stale·与 coverage_metrics.py 及收口报告自身矛盾**。属"外"账辅助显示行（非单分母头条·头条机算正确），严重度低，但是真·跨 recon 不一致，须修。

### 2. 成色诚实核 —— **过（带瑕疵 F-2）**
- **标量类-k1 41/51**：batch8 逐格核对——18 dequant 全 `test-only-not-in-denom`标注、opp 全 SCALAR（rvv=0·三源 objdump 确证）、big-multiple（q5_K 5.15×/tq2_0 4.59×/iq4_nl 3.81×）**强制标 opp-immaturity·禁称硬赢**、3 具名-X（iq3_xxs/iq3_s gather-bound·nvfp4 ldexpf）如实负结果、0 verified hand-brick。**诚实**。
- **iq/tq gemm 便宜档 rvv gcc-death**：batch6 rvv-gcc 部署域 **7/7 具名-X**（iq2_xs/iq2_s/mxfp4 输给 gcc 纯标量 ref vec=0）如实、clang-18 换编译器恢复 6.8–11.2× 坐实 [CASE-COMPILER-ASYMMETRY]、单一诚实数 2.5–4.8× best-vs-best、便宜档禁称硬赢。**诚实**。
- **IME 2/2 输**：batch7 q4_0 输 5.1×·q4_K 输 20×·q8_0 结构 void（BLOCKED），明标"强对手负结果·非便宜档·0 我方赢"，且强调"与 e2e perf-covered 绿分开报·禁互推"。**如实负结果·诚实**。
- **收口报告头条**：§三 "本役 0 verified hand-brick"·"真赢=2 hand-brick(q4_K/q2_K@k1)"·§一标量类明标"禁称硬赢"。**头条成色诚实无虚高**。
- **⚠ 瑕疵 F-2（T3 note 列 hand-brick 成色过标）**：`recon_master_rebuild.py:122-123`（→ T3_master_rebuild.csv note 列）为 **q5_K@k1(4.448×) / q6_K@k1(1.769×)** 标 `"真 hand-brick repack STRONG"` + opp 打 `(REAL)` 标签，与 q2_K/q4_K 同 "STRONG" 措辞。**但权威 sealed 主表报告（2026-07-15-G8-主表重铸·line 114/191）grade 更谨慎**：q5_K = "k1 vs block-dot-DEPLOYED（**8x8 前提证伪**）"、q5_K/q6_K@k1 = "若升成色须重建 native-layout hand-brick 对手·**A-B DEFERRED**"。即 q5_K 强对手前提已被证伪、q6_K 成色升级仍延后未坐实。→ recon note 把这 2 格过标为 STRONG hand-brick，**与 sealed 主表自相矛盾**。**注**：此过标**不污染头条**——`真硬赢 hand-brick=2` 与收口报告 §三 均正确保持 2（见 F-3）。仅 T3 内部数据 note 列与 sealed 报告口径不一致。

### 3. 禁继承核 —— **过**
- **decode 真独立测量非继承 prefill**：batch4/5 用 **核 body 形态**（`typed_repack_gemv_loop_body` vs `block_dot`）作判别键、非符号名；首次构造 M=1 GEVM harness（`flat_gevm_m1_driver.cpp`/`kquant_gevm_m1_driver_b5.cpp`）；T3 gemm decode/prefill 分立双行（[K-10]）。
- **IME 非继承 native**：batch7 明记"q4_K@ime 数据串行 bug 前科·真测 IME kernel-sym vmadot·非 e2e/native-gemm 继承"，且从 vendor 符号表+源码 dispatch 双证 q8_0 无 IME 对手。
- **batch1 禁继承 vindicated**：archive gcc-15 对称 8/8 LOSS → clang-18 对称 3 PASS@rvv，差异全在 ours 侧 codegen（tq2_0 534ns[gcc]→104ns[clang]）。
- **抽查 T3 cold 溯源到 batch raw（5 格·全对）**：`gemm q4_K decode` rvv 0.361/k1 1.535（=batch4 §2.4）· `gemm iq2_xs prefill` rvv 0.379[gcc-death]/k1 10.3（=batch6 §2.1/2.2）· `gemm iq2_xs decode` rvv 1.0/k1 3.81（=batch6 §2.4）· `vec_dot iq2_xs` rvv 0.529/k1 0.474（=batch1 §2.1/2.2）· `vec_dot tq2_0` rvv 0.975/k1 0.879（=batch1）。**逐格对应 batch 实录·0 造数**。
- **sweep 报告自查串行 bug 防线**：主表 §六.1 记 q4_K@ime 串行误显 PASS/1.187 已订正为 pending·§三."结构拆不造赢·手调-rvv 仍 5 非虚增 9"。

### 4. 禁互推核 —— **过**
- 四赛道口径分明、无越界外推：**kernel-sym 四档头条（recon·clang-18 对称 micro）** / **perf-covered 9/83（e2e 系统账·`perf_covered_metrics.py`）** / **certified 101/108（构造轴·`coverage_metrics.py`）** / **IME kernel-sym（vs vendor 手调 IME）vs IME e2e 绿**。
- 每 batch 顶注均 "NOT e2e·NOT perf-covered·不入系统账·[NG-4]"·batch4/5 明写 "禁写'加速 N kernel'于 e2e 语境"。
- **IME 双赛道无混淆**：perf-covered 9/83 含 q4_0@ime/q8_0@ime 转绿（e2e stock-parity）；IME kernel-sym（batch7）同格 vs vendor 手调 IME = 2/2 输。batch7 明记 "T3 现有 gemm|{q4_0,q8_0}|ime:绿 不改（那是 e2e 账）·本役数据入 kernel-sym-vs-vendor 附列·禁互推"。**两个对手身份、两个账、正确分立**。

### 5. C3′ 边界核 —— **过（5 负结果全坐实）**
- **[KQUANT-DECODE]**（batch5）：byte-exact gate 4 格×双板×双编译器×2-seed = 全 0/512 mismatch；objdump vsetvl 实数（gcc 922–2952 vs clang 14–81）；预注册预期"未测不作结论·0 造数"（预判不作结论坐实）。11/12 具名-X·q2_K@k1 唯 1 near-parity 明标"非 win·成色低不称赢"。
- **[IME-VENDOR]**（batch7）：ZERO-MODEL int32 byte-exact（vmadot 0xe210312b 真硅 512/512+64/64）；vendor 符号级探针+源码 dispatch；gap 随格式复杂度扩张量化（q4_0 25ms→q4_K 110ms vs vendor 4.9→5.5ms）。
- **[gcc-death iq/tq]**（batch6）：objdump vec=0 纯标量 ref 探针 + clang 消歧 6.8–11.2×。
- **[iq1_s-HOLISTIC]**（A2.5）：byte-exact 0mism/0ULP（M=1&M=8·双 seed）；objdump 结构靶命中表（vmv.x.s 10→2 追平 opp）；三杠杆逐一证伪 ratio 恒 ~0.41；**"真-fixable 假设本役证伪·HOT 反退 10%"如实报**·双板确认。教科书级诚实负结果。
- **[emitter-maturity dequant/preduce]**（batch3/8）：iq3 gather-bound/nvfp4 ldexpf/product_reduce offbin 具名墙。
- **预判不作结论纪律**：batch5 "super-block fold@M=1 rvv 大概率 LOSS·未测不作结论·0 造数"——预注册后仍逐格真测，未拿预判当结论。**坐实**。

### 6. 度假自治合规核 —— **过**
- **PENDING_RULINGS 13 项全触硬冻结·无该必问却自决**：canon 措辞（PR-2/4/5/6/7·全"只加订正注不改条文"）· 分母/头条口径（PR-3/12/13·"非扩分母"·单分母制机算）· 板采购（PR-1/8·"采购不执行·硬冻结"）· 队列/基建（PR-8 runner 供应 DORMANT）。每项载保守默认 + agent 建议 + 状态。**未见任何硬冻结项被擅自执行**。
- **canon 条文零擅改**：G8 攻坚期唯 2 处 spec 触碰——(a) `capability-model/index.md [S-2]`（1249568b7·〇.3 授权的能力措辞两层拆分·且是**向诚实方向收窄**：把"load-time transitive closure"订正为"compile-time today·load-time per [D-2a] 目标契约·未实现"；[I7] 核心不变量条文本体零改→延 PR-4）；(b) `mlir-testing-contract.md`（3df4a278c·T-X 六列 DEFINITION-ONLY·PR-1 默认③明确授权·非 canon 红线）。**两处均属授权范围·硬 canon 条文正确延后至 PENDING·合规**。
- **分母只随 recon 机算·无手改**：单分母 = 主表非 N/A-hw ∧ 非 q1_0（机算）；Σ=分母机检；N/A-hw 机判（ime.present×板实例）。**未见手改分母**。

---

## ② 抓到的错/虚高/不一致清单

> **0 头条级错误·0 头条级虚高**。以下 3 项均为内部工件瑕疵（recon 脚本 / T3 数据），不改变收口报告任何头条主张。留主会执行。

| # | 类别 | 位置 | 问题 | 严重度 |
|---|---|---|---|---|
| **F-1** | 跨-recon 不一致（stale） | `.trellis/scripts/recon_master_rebuild.py:422` | 数字字典 hardcode `certified="84/91"`，注记标 "coverage_metrics.py"；但 coverage_metrics.py 现机算 **101/108**（regime-split 〇.1 后）。recon 打印值与 live 脚本+收口报告自相矛盾。 | 低（外账显示行·非头条·头条机算正确） |
| **F-2** | 成色过标（vs sealed 报告） | `.trellis/scripts/recon_master_rebuild.py:122-123`（→ T3 note 列） | q5_K@k1/q6_K@k1 标 `"真 hand-brick repack STRONG"` + `(REAL)`，与权威 sealed 主表（2026-07-15·line 114/191）"q5_K 8x8 前提证伪"/"q5_K/q6_K 成色升级 A-B DEFERRED"矛盾。这 2 格是 手调-tier PASS 但**成色未坐实为 STRONG hand-brick 赢**。 | 中（T3 内部 note·若外部查 T3 会误得 4 STRONG hand-brick 而非权威 2） |
| **F-3** | Hardcoded 显示串（鲁棒性） | `.trellis/scripts/recon_master_rebuild.py:424` | `真硬赢 hand-brick="2"` 亦为 **hardcode**（非从 CSV 机算）。当前"2"与权威审计一致（**未虚高·实为保守正确**），但若 CSV/COLD 漂移（如 F-2 的 q5_K/q6_K 过标）此串不会更新→无机检守护。**与 F-1 同类**（recon 有 2 处 hardcode 冒充"机算/coverage_metrics.py"）。 | 低（当前值正确·仅缺机检） |

**关键正面确认（非瑕疵·澄清）**：
- 收口报告 §三 line 31 **正确**保持 "2 hand-brick(q4_K/q2_K@k1 GEMM prefill)"——F-2 的 T3 过标**未泄漏进头条**。
- 手调-rvv 头条 8/24 含 clang-18-micro PASS（iq2_s 对手冷态坍塌·iq4_xs/iq4_nl DIVERGE·tq2_0 clang-parity）——**非 rvv 部署（gcc-15）赢**，但 batch1 §6+PR-9 已如实分域披露（rvv 主表 = clang-18 对称 micro·gcc-deploy 强制脚注），且报告 §三"所有 PASS=弱赢分层·计数≠强赢"+真硬赢锁 2，**成色纪律守住·非虚高**（仅提示：§一头条表未内联 clang-micro caveat，读者需配 §三 读）。

---

## ③ 成色诚实评级：**HIGH（诚实第一·守得住）**

- **禁称硬赢 / 计数≠强赢**：全 8 batch + A2.5 + 收口报告逐层强制执行。便宜档、opp-immaturity、near-parity、compiler-artifact、cold-collapse、DIVERGE、gather-bound-loss、gcc-death 全部如实分层披露。
- **0 hand-brick 就说 0**：batch4/5/8/A2.5 明写 "0 verified hand-brick"。
- **负结果当知识**：5 个 C3′ 边界 + IME 2/2 输 + iq1_s 真-fixable 证伪，全部诚实映射为"能力键控适用边界"，无粉饰。
- **头条真硬赢锁 2**（q4_K/q2_K@k1·byte-verified），未把 cross-op/弱对手/near-parity/clang-micro PASS 混入硬赢。
- **唯一成色瑕疵 = F-2**（T3 note 列 q5_K/q6_K@k1 过标 STRONG），**局部于内部工件·未上头条**。若无此瑕疵评级为 EXEMPLARY。

---

## ④ 建议修正（留主会执行·A5 不改任何文件）

1. **F-1（须修）**：`recon_master_rebuild.py:422` 的 `"84/91"` 改为从 `coverage_metrics.py report` 读取（或至少更新为 `"101/108"`），并使注记与真源一致。当前 stale 值会误导任何读 recon 数字字典者。
2. **F-2（须修·成色一致性）**：`recon_master_rebuild.py:122-123` 把 q5_K@k1/q6_K@k1 的 note 从 `"真 hand-brick repack STRONG"` 订正为与 sealed 主表一致的谨慎措辞（如 q5_K:`"REAL-opp·8x8 前提证伪·成色升级 DEFERRED"`·q6_K:`"REAL-opp·成色升级 A-B DEFERRED·非 verified-STRONG"`）。避免 T3 与 2026-07-15 sealed 报告口径打架、避免外部查 T3 误得 4 STRONG hand-brick。（属成色措辞·若视为 canon 级则登记 PENDING 待裁。）
3. **F-3（建议·鲁棒性）**：`recon_master_rebuild.py:424` 的 `真硬赢 hand-brick="2"` 改为从 CSV 按 filter（手调 ∧ REAL ∧ PASS ∧ k1 ∧ byte-verified-STRONG）机算，并在机算口径中排除 F-2 两格（8x8 证伪/DEFERRED）。使"2"有机检守护、不再是 hardcode。同时消除 recon "机算"宣称与 3 处 hardcode 的措辞张力。
4. **（可选·presentation）**：收口报告 §一四档表可为 手调-rvv 8/24 内联一句 "clang-18 对称 micro·rvv 部署 gcc-15 见 §三/PR-9" caveat，防头条被单独引用时误读为 rvv 部署赢。（非错误·仅防误读。）

---

## 复核结论

**过（带 3 项瑕疵·0 头条级错误·0 头条级虚高）。** 三 recon 机算头条（Σ=分母 102/105 ✓·certified 101/108 reconciliation_ok·perf-covered 9/83 reconciliation_ok）全部真过；成色诚实纪律（禁称硬赢·计数≠强赢·0 hand-brick 就说 0·负结果当知识）守得极稳；5 个 C3′ 负结果 byte-exact+objdump+预判不作结论全坐实；禁继承（body-form 判别键·M=1 GEVM 独立 harness·抽查 5 格溯源对）+ 禁互推（四赛道分立·IME 双账不混）兑现；度假自治 13 PENDING 零越权、canon 条文零擅改、分母只随机算。**3 项瑕疵（F-1 certified stale·F-2 q5_K/q6_K@k1 note 过标·F-3 hand-brick hardcode）全部局限于 recon 脚本/T3 内部工件，均未污染收口报告任何头条主张**。建议主会按 ④ 修 recon 3 处，其余交付诚实可信。
