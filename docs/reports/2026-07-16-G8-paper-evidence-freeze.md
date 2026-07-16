# G8 论文素材冻结点 — 本轮净新增证据（一页式 · 2026-07-16 · 本轮定格 · paper-ready）

> **性质**：纯案头冻结·**无板·无代码·无新主张·无新数字**（全部数字引自已 commit 的表行/schema/report·**0 造数**）。
> **用途**：把 G8 战役本轮产出的新证据按三贡献 C1/C2/C3′ 冻结成 **净新增 + commit 指针 + 成色 + 诚实状态** 的一页式清单。
> 权威索引本体 = `docs/reports/2026-07-10-paper-evidence-index.md`（本 doc 与其**交叉引用**·索引各分节含标注「本轮 G8 冻结」的对应证据行）。
> **成色诚实 = 命门**（[NG-4] 措辞锁）：计数≠强赢·负结果如实标负·记忆=authority 非翻盘·传导=机制正结果非 perf-beat·四账禁互推·perf-covered 9/83 硬冻结。

---

## C1 — 模板协议本体（合取存在性 → 可复制协议）

**净新增①：[SEL-3] 测量记忆层 LANDED = 能力键控 authority 层。**
- **是什么**：版本化 schema（`measurement-memory.v1.json` v1.0.0·superschema 一般化 `tiling-measurements.v1.json` 到单 `rows[]` 按 `(declared_instance_hash, kernel, variant)` + `variant_axis` 键控）+ **字节精确门 fail-closed**（byte-exact gate·ZERO-MODEL·三要件）+ 冷启动 `lookupMeasurement` 出 **reason=measured**（authority 序 = 硬件实测 > 本缓存 > 静态先验）+ 离线**写回谐调器** proof-of-mechanism。驱动 fail-closed 运行期调度守卫 → 属 C1 协议本体（能力键控 authority 层）。
- **commit 指针**：`26b875ac1`（T-SEL3-1/2 落地+灌库）· `513b464d6`（T-SEL3-3 视图轴参数化·byte-exact 零回归）· `0b8abb239`（T-SEL3-5 T4b 两列真数验收）· `0eb8a4543`（T-SEL3-4 写回谐调器）。文件：`schema/measurement-memory.v1.json`·`include/Weft/Plugin/RVV/RVVRepackTilingSelection.h`（`lookupMeasurement`/`RVVMeasurementAxis`/`priorTilingVariantForShape`）·`experiments/active/result-tables/T4b_selector_ablation.md` §2.1b。
- **成色 / 诚实状态**：**mechanism proven**。**★诚实边界（T4b §2.1b 命门·[NG-4]）**：`lookupMeasurement` 命中的 6 格 L2 记忆 verdict **与能力 shape/stride 先验一致**（先验 `.h:122-131`/`.h:431-434` 本身已良设计·冷启动即选中正确 variant）→ 记忆兑现价值 = **测量 argmin AUTHORITY + fail-closed-revalidate + byte-exact provenance**·**不是**把错先验翻正。**禁夸成"记忆翻盘"**。真"更快变体"内容 = 4 格（3 cliff-tiling + 1 col_outer 2.47×）·2 格（q6_K/q3_K）为 measured plain-fallback（"测量说别 tile"）。strip-width VLEN 子轴仍 [GAP-P1] 未键控（预留空 seed·如实标 gap 非已实现）。

**净新增②：regime-split 结构判据落地（recon 机算）。**
- **是什么**：roster 93→110 · certified 84/91→**101/108** · 全 `recon_master_rebuild.py` 机算（snapshot `g8-master-rebuild-v2-single-denom`·Σ=分母✓）。
- **commit / 指针**：`docs/reports/2026-07-16-G8-全量攻坚收口报告.md` §五（四裁决落地·PENDING_RULINGS 13 项全触硬冻结·canon 条文零改·分母只随 recon）。
- **成色 / 诚实状态**：machine-anchored（recon 机算·非人填·anti-gate=分母修正非为过门）。

---

## C2 — 模板经济学（泛化代价 → 边际成本规律）

**净新增：[SEL-3] 灌库 = 零新机制复用存量。**
- **是什么**：12 L2 行 verbatim migrate（10 sp4_tiling + 2 loop_order·copy-transform·live `tiling-measurements.v1.json` UNCHANGED）+ 168 T3 coverage seed 只读转录（single-point vs-opponent·selection_valid=false·byte_exact_gate=pending）= **沉淀既有实测入库·非重造新机制**·**ZERO new timing·0 造数**。
- **commit / 指针**：`26b875ac1`。`schema/measurement-memory.v1.json` `$meta.provenance` + `$meta.honesty_boundary_L2_only.count`（12 selection_valid=true + 168 coverage seeds）。
- **成色 / 诚实状态**：machine-anchored（纯转录）。**诚实边界**：这是"复用存量、零新机制"的**泛化代价证据**（边际成本低=复用既有实测）·**不新增 C2 extension-family 曲线点**——C2 独立家族曲线仍 **2/≥3（honest-missing）不变**（IME 2484 + X-SCALAR 1501）。

---

## C3′ — 模板产出质量（能力键控优化模式库 + 实测/迁移）

**净新增①（★机制·正判据）：[DISCRIMINATOR-OPPONENT-BOUND-TYPE]。**
- **是什么**：kernel 算力赢**传导 e2e IFF 对手 compute-bound·washes IFF memory-bound**。判别键 = **对手 bound-type**（由 相(M)×板×出货编译器 对该 kernel 计算路的 autovec 质量共同决定·**非格式名·非 kernel 倍数大小**）。**正交** gcc-death 编译器轴（q4_K 同格式 @k1(clang)=1.101× 传导 vs @rvv(gcc)=0.334× gcc-death·两轴分立归因）。三点隔离 CONFIRMED：① 同板同编译器同核族同相·唯一差=格式（q4_0@k1 decode 0.857× WASH vs q5@k1 decode 1.97× 传导）② 同格式跨板（q5@rvv wash vs q5@k1 传导）③ 相轴（q4_K@k1 prefill 传导 vs decode isolation 0.995×）。
- **commit / 指针**：`d50b09a66`（T6 立册）。`experiments/active/result-tables/T6_e2e_transduction_ledger.md`（四列会计主表）· `experiments/active/g8-stage3-attack/e2e-transduction-batch1.md`。
- **成色 / 诚实状态**：**机制级正结果**·= `[[kernel-wins-dont-transplant-to-e2e]]` 的**正判据形式**（何时传导·非仅何时 wash）。**[NG-4]：非 perf-beat 声明·非 e2e 系统账绿格新增**（perf-covered 9/83 硬冻结）。

**净新增②：2 genuine 强 e2e 传导（deployed==proven）。**
- **q4_K@k1 e2e prefill 1.101× = 首个 e2e beat-hand-brick**：对手 = 真出货手调 GEMM `ggml_gemm_q4_K_16x1_q8_K`（compute-bound·objdump 767 insn/80 vwmacc·M=128 复用）·我方 HEAD-live emitc GEMM byte-verified drop-in（nbad=0·md5 9e057adb·deployed==proven）·greedy token-identical。倍数中等（Amdahl 稀释 1.187→1.101）但**是最强对手类·真传导**（非 wash·非 routing 白嫖·净新发射体 body-swap）= **beat-hand-tuned**。
- **q5@k1 e2e 2×**（C1 selector deployed·弱对手 block-dot compute-bound）：q5_0 decode 1.966×（fresh clean）/ prefill 2.213×·q5_1 decode 2.073× / prefill 2.344×（deployed==proven byte-equivalence·.so df88afa3/43569a46 = sealed byte-id）= **beat-weak-baseline**（clang-18 弱 5th-bit autovec·对手真出货 block-dot 非稻草人·合法赢）。
- **commit / 指针**：`972ab5c2a`（q4_K batch1）· `195b11910`（batch1 收尾·q5 fresh + washout 三点隔离 + attribution 订正）。`T6_e2e_transduction_ledger.md` §1 + `e2e-transduction-batch1.md` §1/§3。
- **成色 / 诚实状态**：**2 genuine 强传导·成色分层诚实**（前者 beat-hand-tuned·后者 beat-weak-baseline）·deployed==proven·**相×板×编译器域锁**·**四账禁互推**。

**净新增③（★负结果=边界=知识）：5 个教科书级 C3′ 负结果。**
1. **[C3′-KQUANT-DECODE-BOUNDARY]**：K-quant super-block repack-GEVM decode@M=1·fold epilogue 不 amortize@M=1（判别键 = sub-block 数 16 vs 8·q4_K@k1 win 不泛化）→ front-door 应 **DECLINE** K-quant repack@decode。
2. **[IME-VENDOR-NEGATIVE]（最锋利）**：首个 kernel-sym 对真硅 SpacemiT vendor 手调 IME 核正面度量·**2/2 输**（q4_0 5×·q4_K 20×）·**gap 随格式复杂度扩张**（我方 fold 随 super-block 25→110ms·vendor 单核 4.9→5.5ms）= [PAT-1] format-keyed 边界最锋利证据·q8_0@ime 结构 void（坐实 e2e "q8_0 beat"=赢 RVV-repack 非 IME）。
3. **iq/tq gemm gcc-death**：prefill rvv-gcc（deploy MAIN）7/7 输纯标量 ref·[CASE-KQUANT-GCC-CODEGEN] 扩展到 iq/tq repack-GEMM（同格 clang 6.8-11.2× 快于 gcc）。
4. **[GAP-IQ1S]-RESOLVED-HOLISTIC**：iq1_s emitter-maturity = holistic micro-scheduling（非单一可隔离轴）·三 byte-exact 杠杆逐一精确追平 opp 单一优势轴·ratio 恒 ~0.40-0.47·**翻正门=结构级重写超单旋钮**（真-fixable≠保证翻正的诚实案例·G2 未翻正双板确认）。
5. **dequant/product_reduce emitter-maturity 墙**：iq3 gather-bound（vrgather）+ nvfp4 ldexpf + product_reduce offbin 被 scalar 追平。
- **commit / 指针**：`docs/reports/2026-07-16-G8-全量攻坚收口报告.md` §四。
- **成色 / 诚实状态**：**如实标负**（负结果 = 能力键控适用边界诚实映射·**不粉饰成正**）·部署级洞察 = front-door 应 DECLINE {K-quant repack@decode · IME@复杂格式 · gcc-death 路}·[PAT-1] format-keyed 适用边界 C3′。

---

## ★成色诚实自查（命门 · [NG-4] · 全 doc 同锁）

1. **[SEL-3] = authority 非翻盘**：6 格 L2 记忆 verdict 与能力先验一致·价值 = 测量 argmin AUTHORITY + fail-closed + byte-exact provenance·非翻正错先验（T4b §2.1b 命门）。**记忆全面翻盘 = 无。**
2. **[DISCRIMINATOR] = 机制正结果非 perf-beat**：这是"何时传导"的判别律·不是新性能主张·不新增绿格。
3. **5 负结果如实标负**：负结果 = 边界 = 知识·不改写成正·映射能力键控的适用边界。
4. **计数 ≠ 强赢**：本役真硬赢强手调 = 2 hand-brick（q4_K/q2_K@k1 GEMM prefill·byte-verified）+ q5@k1 C1 + q4_K@k1 e2e（首个 e2e beat-hand-brick）·**余多 vs 弱对手 / parity-by-roofline / opp-immaturity / compiler-artifact**·大量 PASS 非 vector-superiority。
5. **四账禁互推**：kernel-sym（单分母四档 micro）/ perf-covered 9/83（fair-protocol ≥parity）/ certified 101/108（构造轴 byte-exact）/ e2e 系统账（T6 传导）/ [SEL-3] 记忆——**五赛道各自成立·永不互推**（一格可 kernel 赢而 e2e wash·IME 5.51×→0.86× 铁证）。
6. **perf-covered 9/83 未动**：本轮**永不增头条格数**·q4_K@k1 e2e beat = 成色升级候选（PR-14·登记不擅改）·canon 分母/roster $meta/头条口径/排序不动。

---

## 冻结标记

- **本轮定格（paper-ready）**：C1 二净新增（[SEL-3] authority 层 + regime-split recon）· C2 一净新增（[SEL-3] 灌库=复用存量）· C3′ 三净新增（[DISCRIMINATOR] 正判据 + 2 强 e2e 传导 + 5 负结果）。
- **触碰集**：本 doc（新建）+ `docs/reports/2026-07-10-paper-evidence-index.md`（加行·交叉引用本 doc）。**零** canon / roster $meta / schema / perf-covered 计数 / lib / include 改动。
- **数字来源**：全部引自已 commit 的 T6 ledger（`d50b09a66`）/ batch1 report（`972ab5c2a`/`195b11910`）/ T4b §2.1b（`0b8abb239`）/ measurement-memory schema（`26b875ac1`/`513b464d6`/`0eb8a4543`）/ G8 收口报告。**0 造数。**

---

*freeze doc 初版 2026-07-16 · 纯案头 · 未 git commit（主会话独立复验后提交）· [NG-4] 成色诚实锁 · 四账禁互推 · perf-covered 9/83 硬冻结。*
