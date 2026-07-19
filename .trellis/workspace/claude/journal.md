# Workspace Journal · claude · 《开测篇》战役

## 简报快照 · 2026-07-18 (§一.4 进度条)

**census 机算态**（`experiments/master/T3_master_rebuild.csv`·108 行四元组·纯字段枚举·无预处理归并）：

| 档 | rvv | k1 |
|---|---|---|
| PASS 类（PASS + PASS(decode-M1-GEVM) + PASS-DEPLOYED） | 68 | 72 |
| **具名-X 非PASS 余额** | **27**（20 + 7 decode-M1） | **28**（22 + 4 decode-M1 + 2 IME） |
| **pending 余额** | **7**（fold 5 + 照测 1 + 真 1） | **5**（fold 3 + 真 1 + IME 1） |
| 终态（域外/N-A-hw） | 6 | 3 |

- **各档非PASS 余额**：rvv 27 具名-X / k1 28 具名-X（两档硬收口目标面）。
- **pending 余额**：rvv 7 / k1 5。
- **本轮新判定格数**：**0**（两 agent 施工中·尚无 verdict 落地）。
- **本轮板测批次**：**2 派**（mech-① rvv/k1 ‖ S线 scalar）·**0 回**。
- **constructed 待板端门数**：2（mech-① 满展开 leaf 待板测翻正 · S线 scalar harness 待板测 S1）。

> 新判定数连续两轮不涨则置顶报因（§一.4）——本轮为战役重启首轮·基线。

## 本轮编排决策（自决直行·§1.1 记一行）

1. **建 mech-① task**（`07-18-k-mech1-vlen-leaf`）+ 派 trellis-implement（agent K）。
   - 依据：机制① 命中 8 格 = gemm_tile.sh 覆盖的 4 IQ 格 = **唯一 harness 就绪的攻坚机制**·整环可走通。
   - 判据厘清：攻坚 ≠ 去重·门 = 数值正确(vs oracle)+板测 cold≥0.8+**spill 真降**（非产物不变）。
   - re-roll trap 铁律入 prd（三次证伪教训·每格先反汇编·spill 未降立即停该格具名）。
   - 独占 `lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp` + rvv/k1 板。

2. **建 S线 task**（`07-18-s-scalar-tx-s1`）+ 派 trellis-implement（agent S）。
   - ★突破依据：**超锐(scalar) 板 verified = 真物理 no-V 硅**（isa=`rv64imafdch_...`·无 v/zve·clang-18 装讫）
     = B4 案头（2026-07-15）「手头无 no-V 真硅」前提**已证伪**（该板后上线）。
   - 后果：T-X 列1 (b)窄豁免 → **(a)物理真硅**·「validated on silicon」字面成立；
     [对手法 §板别提醒] ggml `_generic` 在 no-V 板满足部署要件 = **真对手非便宜档**·S1 干净对局。
   - 独占 scalar harness（`tools/bench/cells/`）+ scalar 板·与 agent K 不相交。

## 触碰集回避（并行纪律·主会话此刻禁碰）

两 agent 覆盖：`RVVToEmitCGridCodebook.cpp`、RVV emitter、scalar harness、**issues 登记簿**、
**runs.log**、**master**、**K-attack-fanout-ledger.md**。⟹ 主会话此刻只动本 journal + 只读侦察。

## 待裁登记（agent 携带·主会话不重复登记以免撞 issues 文件）

- T-X 定义层归属（重构后无 testing 层·measurement/evidence?）—— agent S 携带登记·保守默认落 experiments/。
- PR-1 采购 no-V 板 → 已无需（超锐即真硅）·agent S 标 RESOLVED-BY-FACT。
- ISSUE-098（master 一等写者）/ ISSUE-099（FLAT 通道）/ ISSUE-101（判别键 canon）—— 存量·不阻塞。

## 第三线暂缓（判断记录）

E线 §五 明示「K线收口后开·板时让 K」→ 不启。
R线阶段一去重 batch3 触碰共享 emitter 头（`RVVToEmitCSupport.h` 广泛 include）·与 agent K 的
GridCodebook.cpp 编辑有**头文件重编译/potential 交织风险** → **暂缓至一 agent 回**（避免 emitter 文件竞态·
[[parallel-lines-need-disjoint-files]]）。两线已覆盖两条最高优先线（K+S），不反射性 fan-out 第三线。

---

## agent K 回报 · 机制① 反汇编诊断（2026-07-18·NO CODE·re-roll trap 避）

**结论 = 诚实边界（§六）·但非「架构不可达」**：PRD 机制①「VLEN 专化**满展开**」经反汇编证伪 = **满展开 no-op**。

**反汇编诊断（真硬件·`RVVToEmitCGridCodebook.cpp` 现产物 iq3_xxs_gemm.c）**：
- **满展开已满足**：现 leaf 只剩 3 结构循环（row/col/**block**）·super-block×tile 计算体已全直线。再展开=复制同 body=纯 re-roll。
- **storm+spill 板不变·非门**：OURS leaf.o 两板逐一相同 → ins=15238/15240·**vset=5397**（`e16,m1`×2698 + `e8,mf2`×2688 交替抖动）·标量 sp-spill=**56**·向量整寄存器 spill=**77**。
- **同 leaf·同 storm·verdict 相反**（判别铁证）：cold_X **rvv iq3_s 1.34 WIN / iq3_xxs 0.95** vs **k1 iq3_s 0.61 / iq3_xxs 0.65 LOSS**（VLEN128 vs VLEN256·目标 0.8）⟹ storm 非 k1 gap 因。
- **真 k1 lever = VLEN256 半宽欠用**：iq3 group 固定 AVL=8·编译器 `mf2`/`m1`（VLEN128 满/VLEN256 vl=8/16 半宽）。对手 `vec_dot_iq3_xxs_q8_K_vl256` vset=41·宽 LMUL（`e8,m2`=64lane/`e32,m4`/`e16,m2`）+ AVL 对齐 256b。

**三档墙**：①满展开旋钮=无（已展开）②我方内禀 storm/spill=板不变非门 ③**真 lever=VLEN256 width-widening**（对手已达+OURS 同格式 rvv 赢=天花板双证·**非架构不可达**·但是重结构机制·非满展开）。

**产出**：ISSUE-102 登记（判据级·机制① re-scope 待裁）+ 扇出台账机制① 结果列改「诊断证伪·待 re-scope」+ task status=blocked(ISSUE-102)。
**铁律守**：0 造数（无 author·无新 cold·k1/rvv 数引既有 measure.log）·sealed 9/83 未动·master 未动·无 git commit·无 add -A。
**触碰集**：`RVVToEmitCGridCodebook.cpp`=**只读**（未改·避 re-roll）·issues/性能与测量.md（+ISSUE-102）·K-attack-fanout-ledger.md（机制①行）·task.json·本 journal。板端 /tmp 诊断 asm（可复跑）。

**复跑命令**（诊断·非测量·无主表行）：
- rvv：`scp iq3_xxs_gemm.c rvv:/tmp/kmech1_diag/; ssh rvv 'clang-18.1.8 -O3 -march=rv64gcv_... -x c++ ... -c; objdump -d → ours.asm'`
- k1：同·`clang-18 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs`；对手 `objdump --disassemble=ggml_vec_dot_iq3_xxs_q8_K_vl256 /data/k1build-stock/bin/libggml-cpu.so`

---

## agent S 回报 · scalar 真 no-V 硅 T-X 六列 + S1 对局（2026-07-18·真板测·全绿）

**结论 = T-X 六列全绿·S1 真数落账（enablement·NON-Win·[L-6]）**。超锐(scalar) 板真跑复现，零造数。

**板事实（verified）**：isa=`rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`（无 v/zve=**(a)物理 no-V**）·clang-18.1.8·Fedora42/8-core。⟹ B4 案头「手头无 no-V 真硅」证伪·**列1 去窄豁免标**·「validated on silicon」字面成立。

**harness 建成契约合规**：`tools/bench/cells/scalar_vec_dot.sh`（签名 `<board> <mode> <fmt>`·**仓库侧零写盘**·全 stdout·同 gemm_tile.sh 契约·ISSUE-090）。复用 A3 `weft_scalar_tq2_0_kernel.cpp`+`fp16util.cpp`；新驱动 `scalar_s1_driver.cpp`（住 A3 数据格·只读）**链接真部署 ggml 符号**（升级 A3 内联 reimpl）。git status 证：experiments/master·runs/·runs.log **零触碰**。

**T-X 六列**：
- 列1 (a)物理真硅·verified isa 指纹填入·去窄豁免。
- 列2 clang-18 `-march=rv64gc -mabi=lp64d`·owned weft_scalar 内核·双折叠口径（`contract=on` symmetric-with-deployed-ggml + `contract=off` ZERO-MODEL 中性）。
- 列3 零向量 objdump 机检 **PASS**：kernel.o scalar_ins=191 / vector_mnemonic=0 / vset=0 / cpop_clmul=0 / riscv_intrinsic=0 / weft_rvv=0 / fp16_libcall=0；部署 ggml lib v_mnemonic=0。
- 列4 byte-correct **GREEN**：ZERO-MODEL ours==独立 int-oracle==真部署 ggml（symmetric 三方全等·seed 0xD00D `0x46a18a33`）；反空心两臂 inject1/2 BITES-OK（三方交叉正确隔离故障方）；weft_h2f 9/9 IEEE。
- 列5 逐竞品：codegen 竞品 heteroMx/xDSL-RVV/10x-IREE=0（**codegen 竞品产出为零**）；S1 cold 对局 ratio_cold_X **1.0650**(s1)/**1.0583**(s2)·IQR 1.5–2.3%·**标非 Win·enablement·[L-6]**。
- 列6 deploy==proven（同一 s1on 二进制）+ 措辞铁线逐字。

**S1 对局真数**（N=25·2-seed·flush 32MiB·core idle 100%·contract=on symmetric·byte-exact GREEN 先于计时）：
- seed 0x1357：ours 844905ns / ggml 899805ns → **1.0650**
- seed 0xACE2：ours 846105ns / ggml 895406ns → **1.0583**
- 对手 = 真部署 `ggml_vec_dot_tq2_0_q8_K`（no-V==generic·部署事实要件满足·对手法 §3.4）。**NON-Win·不进系统账/perf-covered/分母**。

**关键 finding**：① ggml `GGML_CPU_FP16_TO_FP32`（rv64gc no-zfh 生成路径）用 `ggml_cpu_init()` 填的查表·不 init 则 x.d 折 0=静默全零输出（driver 已 init）。② ours-vs-ggml byte-exact 仅在**对称 `-ffp-contract`** 下恒成立（部署 ggml 带 FMA fold）；`contract=off` 下 ours==int-oracle 恒 byte-exact（ZERO-MODEL 权威门），ours-vs-ggml 差 ≤ 数 ULP=部署库 FMA fold 收缩（IEEE 合法·[K-5] 浮点 ULP 条款·非缺陷）。⟹ 干净对局须对称构建（同 rvv/k1 harness 惯例）。

**待裁登记（判据级·保守默认续推·未停轮）**：
- **ISSUE-103**（spec树与治理.md）：T-X 六列证据表 spec 层归属（重构后无 testing 层·measurement/evidence?）·保守默认 = 暂落 `experiments/` 活证据（TX-six-column-evidence.md 已更新真硅态）·**未擅建 spec 层**。**连带 B4 PR-1「采购 no-V 板」= RESOLVED-BY-FACT**（超锐即真硅·无需采购）。
- **ISSUE-104**（门与工具.md）：scalar S 线「bench 能调」被四处夹住（runner block=ISSUE-061 · op→harness 名解析 · parser gemm_tile 专用 · 主表无 scalar 列/[L-6] 本不该有）·保守默认 = harness 契约合规+直跑真数·S1 数落 T-X 证据表非主表·**不解 runner block·不加 scalar 列·不改 parser·不改 ISSUE-061 态**。

**铁律守**：0 造数（全真板复现）·[L-6] scalar 不作贡献基线（列5 不升 Win·倍数不进系统账/perf-covered·不扩分母）·sealed 9/83 未动·master 未动·无 git commit·无 add -A·未擅建 spec 层。
**触碰集**：`tools/bench/cells/scalar_vec_dot.sh`（新）· `experiments/.../A3-xscalar-rv64gc/{scalar_s1_driver.cpp 新, TX-six-column-evidence.md 更新}`（数据格）· issues/{spec树与治理.md +103, 门与工具.md +104, index.md +2 行}· 本 journal · task.json。scalar 板 /tmp 临时（可复跑·仓库侧零落盘）。与 agent K 完全不相交（scalar 板≠rvv/k1·scalar_vec_dot.sh≠GridCodebook.cpp）。

---

## main 会话续推 · 机制① re-scope + issues 登记簿完整性修复（2026-07-18）

### A. 机制① re-scope 裁断（agent K 诊断后）
「满展开」经反汇编证伪 = no-op（避第 4 次 re-roll trap·agent K 未 author 码）。真 k1 lever = **VLEN256 半宽欠用**（判别铁证：同 leaf·同 storm 5397·同 spill 56/77·rvv WIN vs k1 LOSS）。
**裁断（§延后裁决·判为必问≠停下）**：agent 置 task=blocked 等 ISSUE-102 裁 → **纠正为续推**。过判别式：author 宽化不需先答「哪种算对」（诊断已定唯一 lever·对手证·rvv 同格式赢）·ISSUE-102 待裁仅**机制①称谓标签**·不阻塞工程（k1 iq3 翻 0.8 由 §二 授权）·停 blocked = 违 §1.3。⟹ re-scope 为 VLEN256 宽化（能力键控·[K-10] 参数级·先例 Win-K1-VLEN vl=16）·task 回 in_progress·派 **agent ac**（iq3_xxs@k1 单格先证·width 真宽验证铁律）。prd 顶 RE-SCOPE 节 + ledger 机制①行 + task.json 已落。

### B. issues 登记簿完整性修复（lost-update + 累积缺行）
**发现**：agent K/S 并发写 issues/index.md → **ISSUE-102 lost-update**（K 写 102·S 后写覆盖）；深挖出**§三 全册索引累积缺 9 行**（095–098·100–104·仅 099 前已补）+ 计数行陈旧（101/001..094·实为 104）。这是「开测前查号」硬 hazard（9 条 issue 对上岗不可见）。
**修（文书级·本体唯一存在·§三 该列出）**：① 建 **`.trellis/scripts/issues_census.py`**（§六「登记簿计数一律机算」持久宿主·类比 recon·leading-token 分布 + RAW 尾串透明枚举·禁隐性归并）② §三补 9 行（001–104 全覆盖·零缺号验证过）③ §二小计刷（性能 36→39·门与工具 19→24·spec树 14→16·机算）④ 计数行 101→104 + 分布刷·引脚本为权威。census 复算 104/零缺号/零重号 ✓。

### C. 并行态（★已全部完成·2026-07-18 更新）
- ~~agent ac（K 宽化·k1）~~ **完成·已 check·已提交·已归档**（iq3_xxs@k1 proven WIN 1.38·NOT deployed）。
- ~~agent af（S 线 check）~~ **完成·全绿·S 线已归档**。
- **★GridCodebook.cpp / RVVToEmitC.cpp 现空闲**（无活跃并行线·后续 emit agent 可安全编辑）。

### D. §一.4 进度条（本轮）
非PASS 余额未变（rvv 27/k1 28·施工中未落新 verdict）·pending rvv 7/k1 5·本轮新判定 0（S 线 T-X 是 enablement 非主表 verdict·K 宽化待板测）·板测批次 K+1(宽化)/S+1(S1 已回待 check)·constructed 待板端门 = 宽化 leaf 1。**新判定连续 1 轮为 0**（下轮仍 0 则置顶报因·但本轮有实质产出：机制① re-scope 定性 + S 线 T-X 六列全绿 + 登记簿完整性修复）。

---

## 本轮收口 + 关键路径裁断（2026-07-18·两 task 归档）

### 收口
- **K宽化 check 全绿**：iq3_xxs@k1 proven WIN 1.38（byte-exact·手调档硬赢·objdump 真宽·2 seed）·trellis-check 7 点全实证。★**deployed-vs-proven**：deployed 叶仍 VLEN128=0.65 ⟹ master 维持具名-X（不 overclaim）·proven 归台账·部署 gated **ISSUE-105**（GEN_SEAL 对 k1 发 VLEN128=半宽根因·判据级）。
- **S线 check 全绿**：T-X 六列 + S1·板上独立复现·**停止条件 2 达成**。
- **落账**：master 不动（sha256 dc876ef6 守恒）·5 commit（issues完整性/S线/K-mech1/2archive·精确路径·排除 order文档+workspace）·两 task 归档。

### ★关键路径裁断（非PASS 按档分类后）
停止条件 1 硬门（标量类/通用向量档）**几乎全是 dequant 轴**（rvv 9 标量类多为 dequant·k1 4 标量类 3 dequant）⟹ **收口必经前置 = R线 dequant 真向量发射器（§四.1·立项批准·PR-31 销案）**。此为**当前关键路径**：R线停止条件本身 + 解锁标量门 + 不依赖 ISSUE-105（与 K线 k1 部署解耦）。手调档（rvv18/k1 24）= best-effort·部分 gated ISSUE-105。
- **顺带（★上轮过头·research 订正）**：K宽化 iq3_xxs@k1（同算子·half 8vs16·objdump 双产物·verdict 0.65/1.38）**只【部分】满足 §四.4**——能力键控侧全要件（两配置·objdump 双产物·归因日志）✓·**但 [SEL-3] 首次真实改判（记忆 verdict 超 authority·ISSUE-035）未满足**（K宽化是**能力先验/VLEN 事实驱动**·方向与 authority 一致·非「记忆翻先验」·evidence §2.3 明文）。⟹ 停止条件 3 的 strip-width 改判**尚未达成**·[SEL-3] 那件仍欠（宽度测量键仍空 seed）。

### 下一主攻
R线 dequant 真向量发射器（§四.1）——单格/小批先证 C4a 参数化路线（同 K 宽化单格先证纪律）·逐格前门→板测→cold·终结 codegen 抽签。
- 待裁不阻塞：ISSUE-102（机制①称谓）/103（T-X层）/104（scalar runner）/105（k1 部署·判据级·真高扇出 lever）。

---

## agent (implement) 回报 · R线前置A emit · iq3_xxs@rvv OWNED 真向量发射器（2026-07-18·真板测）

**结论 = C4a 路线【机制证毕】：owned 真向量 dequant emit 落地 + byte-exact GREEN·但 naive 首版 cold = 大 LOSS 0.18×（真数·具名-X·勿外推倍率）。**

**前置 A（emit·工·自决直行）DONE**：
- **新 owned body** = `emitDequantizeRowIQ3XXSVectorBody`（`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp` 尾·+decl 于 `RVVToEmitCInternal.h`），由 `emitDequantizeRowIQGridBodyShared`（`RVVToEmitCForwardElementwise.cpp`）**仅对 iq3_xxs** 路由（其余 iq2*/iq3_s 仍走 scalar forwarder·未动）。
- **复用机体** = GridCodebook vec_dot 的 vluxei16 grid gather + 符号折叠（vmv/vand/vmsne/vneg/vmerge）·**尾部换成** int→float（vsext_vf4 + vfcvt_f_x_v）+ 运行时 db 标量（vfmul_vf）+ 单位存储（vse32）。无 reduction·无 q8。
- **宽度非假旋钮**：8-lane group = iq3_xxs grid-of-4×2 固定几何（非可调键）·core i8=m1（与既有 iq3_xxs vec_dot 一致）·i32m4/f32m4 为派生宽·自洽·**未焊字面量假旋钮**（避 ISSUE-031/033）。表 decl（weft_iq3xxs_grid/ksigns/kmask）从 canonical kIQ3XXS* 派生·函数局部 static·无 op-attr。

**验收（真板 rvv·8-core·core8 idle100%·gov performance·flush224MiB）**：
1. **emit 真向量 ✓**：`OWNED_src_vec_intrinsics=13 distinct/771 calls`（vsetvl 除外·baseline=2 皆死 vsetvl=ISSUE-001 铁证的反面）。objdump leaf.o：vector_mnemonic=578·vset=129·gather=32·vfp=64。13 类 = {vle8 vle16 vluxei16 vreinterpret vmv_v_x vand vmsne vneg vmerge vsext_vf4 vfcvt_f_x_v vfmul_vf vse32}。
2. **G1 byte-exact GREEN ✓**（ZERO-MODEL·decode-only·nb=4096·1048576 elem·corpus grid_idx 256/256 + sign_sel 128/128 COMPLETE）：ours-vs-oracle mism=0·ggml-vs-oracle mism=0。**反空心三臂真触发**：[B] oracle-fault→ours RED(mism2)·[C] DUT-fault→ours RED-ONLY(mism1·ggml PASS)·[D] leaf 单字节 grid-fault→ours RED-ONLY(mism1027·ggml PASS·咬到 EMITTED RISC-V)。
3. **rvv cold（真·2 seed·N=25）**：s1 ratio_cold=**0.1817**（ours 2012988ns iqr0.41% / ggml 365722ns iqr0.88%）·s2=**0.1824**（ours 2016508 / ggml 367881）。⟹ **具名-X LOSS**（§三.4 <0.8）。byte-exact 先于计时（两 seed 均 mism=0 PASS）。
   - **★真数·非预测**：baseline scalar harness 曾 0.51×·codegen 抽签 interim 记 0.95·**owned naive 真向量 = 0.18×（比 scalar 还慢 ~2.8×）**。这是 de-lottery 后的确定性真值。

**★瓶颈具名（反汇编 + 结构诊断·非焊字面量硬凑）**：naive 首版**逐 group vl=2 indexed gather + 栈数组往返**（每 group：store 2×u16 到栈→取址→vle16→vluxei16 vl=2）= GridCodebook vec_dot 明文 HOIST 掉的「fractional-LMUL 2-element scalarization storm」（该文件 §737-748 pair-batched 宽 gather 正为此）。叠加 mixed LMUL/AVL 反复 vsetvl（leaf.o vset=129）+ 8-lane 用 m4（VLEN128 下 32-lane 容量仅填 8）。⟹ **正确但慢**。**已知 lever = sub-block/pair 批量宽 gather + vsetvl 提升**（GridCodebook 现成范式）·**未在本 task 施工**（避 re-roll trap·需板端逐步反汇编·超单格先证 scope·记 §六遗留）。

**工 vs 判据级**：本 task 只做「工」（author emit + 板测）·**未命中判据级**（未改对手政策/scope/判据/门·未改 harness 结构规则）。verdict 具名-X 遵现行判据（非改判）。**无新 ISSUE 登记需要**。

**入账留 main 会话**（遵 handoff·ISSUE-098 master 一等写者 + recon-dict 通道）：**未跑全 bench runner**（它耦合 master 直写）·**未动 master·未动 runs.log·未动 sealed 9/83**。真 cold 原始数在上（供 main 走 recon-dict 入账：iq3_xxs@rvv dequant 从 interim 0.95 → 真测 0.18× 具名-X·确定性·真向量）。

**触碰集**：`lib/Conversion/RVV/{RVVToEmitCGridCodebook.cpp(+body), RVVToEmitCForwardElementwise.cpp(iq3_xxs 路由), RVVToEmitCInternal.h(decl)}`·`test/Conversion/RVV/rvv-to-emitc-ggml-dequantize-row-iq3-xxs.mlir`(CHECK owned 向量)·`experiments/active/r-dequant/{kernels/iq3_xxs_dequant.c 重生, GEN_SEAL.txt}`·rvv 板 /tmp（临时·仓库侧零落盘）。**无 git commit·无 add -A**。
**回归门**：iq 9 格 dequantize-row FileCheck 全 PASS（其余 4 IQ 未受影响）·全 lit 951/954（3 失 = Scripts bundle-abi e2e·clean 树亦失·环境态·与本改无关·已 stash 复验）。

**遗留 / 扇出**：① iq3_xxs perf 优化（批量宽 gather·§六·未做）② grid 族扇出（iq3_s/iq2*·同 C4a·逐格 byte-exact+板测·禁外推·本 task 只首格）③ @k1 部署 gated ISSUE-105（本 @rvv 先证不涉）。

---

## agent (implement) 回报 · K线 product_reduce.sh harness 审核+完成+板测（2026-07-18·真板测·rvv+k1）

**结论 = 前一 agent 半成品【契约合规且正确·唯一缺口 = chmod +x】；3 格 × 2 板 = 6 cell 全 byte-exact GREEN·全 cold PASS(SANITY 档)；offset_binary_n3@k1 翻 0.797 具名-X → 0.860 PASS。**

### A. harness 审核（product_reduce.sh + preduce_driver.c + kernels/）
- **契约合规 ✓**：仓库侧零落盘（全 stdout + 板端 /tmp 临时·无 scp 回）·fmt 白名单 HARNESS-VOID exit2（ssh 前）·rvv|k1 双板均 clang-18 单世界·HARNESS_RC 打印+exit·assets 只读（ASSET_ROOT 可覆写）·单世界对称（我方向量核 + scalar-ref 同 clang-18 同板→清 gcc 车道欠 ISSUE-099）。
- **ZERO-MODEL [K-5] 真 ✓**：scalar-ref 从原始字节 b[]/a[]/c[] 独立重算整数（向量核走 vand/vsrl/vwmul/vwredsum/vrgather·scalar 走纯 C int·不共享 bit→lane 解码）·非捕获 intermediates。INJECT=2 oracle-poison 咬出 261763 mismatch = 独立性实证（非空心镜像）。
- **反空心三臂真触发 ✓**（6 cell 全）：[A] INJECT=0 mism=0·[B] INJECT=1(ours[mid]+1) mism=1·[C] INJECT=2(oracle 常量+1) mism=261763。
- **verify/sanity/measure 三 mode 齐 ✓**（verify+measure 已实跑·sanity 结构同 measure 减 load-gate）。
- **唯一缺口 = chmod +x**：原 -rw（范本 cells 皆 +x·bench 经 `bash harness` 调用故非硬需·但按约定补）→ **已 chmod +x**。driver/kernels 无需改。
- **小观察（非缺陷·不改）**：objdump 探针标签「vec_core>>vsetvl」对 codebook 呈 vec_core=7<vsetvl=8（e32m1 vl→多 vsetvli）·但真向量信号 = non_vset_vector=15>>vsetvl=8·探针不 gate（授权门 = byte-exact ZERO-MODEL）。

### B. verify（byte-exact ZERO-MODEL · 6 cell 全 GREEN）
| fmt | rvv | k1(VLEN256) |
|---|---|---|
| q4_0_nibble | mism=0·A/B/C bite | mism=0·A/B/C bite |
| offset_binary_n3 | mism=0·A/B/C bite | mism=0·A/B/C bite |
| codebook_n3 | mism=0·A/B/C bite | mism=0·A/B/C bite |

sample 值跨板恒等（-5790/-926/-8510）= VLEN-invariant 整数核实证。scalar-ref caliber=标量类（本 roll 0 向量·codegen-lottery·透明报）·ggml_standalone=ABSENT（internal sub-primitive·T-CENSUS §一.F）。

### C. cold measure（ratio_cold_med = scalar-ref/ours·>1=ours 快·≥0.8=PASS SANITY·2 seed·N=25·flush224MiB）
| fmt | rvv s1/s2 | k1 s1/s2 |
|---|---|---|
| q4_0_nibble | 2.841 / 2.843 | 1.873 / 1.874 |
| offset_binary_n3 | 1.595 / 1.590 | **0.8605 / 0.8606** |
| codebook_n3 | 1.385 / 1.376 | 1.406 / 1.405 |

全 6 cell ≥0.8 → **PASS(SANITY 档)**。load-gate 全过（idle 98–100%·gov performance·IQR 0.1–2.3%）。

### D. verdict + 翻 PASS
- **翻 PASS = 1 格**：offset_binary_n3@k1 从 0.797 具名-X → **0.860 PASS**（两 seed 一致 0.8605/0.8606·IQR 0.12%·core5 idle98%·稳健非噪声）。因翻 PASS·**无需走攻坚环**。
  - 诚实注：k1 该格 ratio<1（ours 27.8ms 慢于 scalar 23.9ms·VLEN256 下 16-lane 块半宽欠用 + offbin 5th-bit 提取重）·0.860 = 便宜档薄 PASS·非赢。
- 其余 5 cell 本就 PASS（recon 「其余 PASS」印证·rvv 大倍数）。
- **★成色**：对手 = constructed scalar-ref（product_reduce 无 ggml standalone 框架核·internal sub-primitive）·**便宜档 [NG-4] kernel-axis SANITY·非 e2e·非 perf-covered·大倍数(2.84/1.87×)禁称硬赢**。

### E. 入账留 main 会话（遵 handoff）
- **未动 master·未动 runs.log·未动 sealed 9/83·无 git commit·无 add -A**。上面 6 cell cold 原始数供 main 走 recon-dict 入账（offset_binary_n3@k1 0.797→0.860 是唯一 verdict 变动·rvv 3 格 + k1 其余 2 格维持 PASS）。
- 未跑全 bench runner（step3/step5 解析格式为 dequant/vec_dot 族·与 product_reduce 输出 schema 不匹配·且 step5 耦合 master 直写 = ISSUE-091 gated）；按 task 直调 harness（= bench `_run_harness` 同路 `bash harness board mode fmt`）取 raw。

### F. 触碰集 / 遗留
- **触碰**：`tools/bench/cells/product_reduce.sh`（仅 chmod +x·内容零改）·rvv+k1 板 /tmp/bench_cells_product_reduce_*（临时·仓库侧零落盘）。preduce_driver.c / kernels/*.pr.c 零改（审核判定完整）。
- **遗留**：① bench step3/step5 解析器不认 product_reduce 输出 schema（int_mismatch=/OURS_<fmt>/OPP-IDENTITY≠dequant 族 mism=/OURS_leaf/OPP）→ 若要走全 runner 入账需扩 parser 或加 product_reduce 分支（本 task scope 外·记）。② offset_binary_n3@k1 0.860 薄 PASS·若板载波动或有回退风险（IQR 极紧·当前稳）。

---

## 2026-07-18 · task `07-18-07-18-k-vecdot-harness`（建 vec_dot.sh K-quant harness + baseline q4_K/q6_K@rvv）

**目标**：建 `tools/bench/cells/vec_dot.sh`（K-quant vec_dot 攻坚通道·解锁最大手调攻坚面）+ baseline 测 1-2 格（不求翻·先接线）。ISSUE-104 命名先厘清。

### A. 命名厘清（ISSUE-104 · 结论）
- **facet 2 命名/接线【解】**：`vec_dot.sh` 与 `scalar_vec_dot.sh` **无 filesystem 冲突**（两文件并存）。`cell_harness(op)` 按 `cells/<op>.sh` 解析·主表 K-quant vec_dot 行 `op=vec_dot` ⟹ **`vec_dot.sh` 直接接线**（谓词实证：`cell_harness('vec_dot')`→`cells/vec_dot.sh`·本 task 前抛 `CellRecipeMissing`）。runner self-test **11/11 PASS**（未破）。
- **真「夹」= harness 粒度（op-全板 vs op×板）判据级·未钦定**：runner op-only 解析 ⟹ `bench vec_dot tq2_0 --board scalar` 仍路由到 `vec_dot.sh`（非 scalar_vec_dot.sh）⟹ `vec_dot.sh` **board-guard scalar→HARNESS-VOID 指向 scalar_vec_dot.sh**（不冒充）。**未擅定粒度**（保守默认·分流）。
- **facet 3 parser gemm_tile 专用·证续存**：runner step3/step4 硬认 gemm_tile 标记（`ABI-GATE`/`T2 ours`/`GEMM_PREFILL`），认不出 vec_dot 标记族（`# VECDOT_CORRECT`/`# BYTE_EXACT`/`# VECDOT_COLD`/`# OPP …vl128_runtime_path`）。**未改 parser**（判据级）。
- **登记**：已 append ISSUE-104【★2026-07-18 补录·第二具体实例】。

### B. harness 契约合规（同 dequantize_row.sh / scalar_vec_dot.sh）
- 住 `tools/bench/cells/vec_dot.sh`·签名 `<board> <mode> <fmt>`·**仓库侧零写盘**（全 stdout·板端 /tmp 临时·无 scp 回）·`+x`·assets 只读（VEC_DOT_ASSET_ROOT/VEC_DOT_KERNEL_ROOT 可覆写）·白名单外 HARNESS-VOID exit2（ssh 前）·三 mode（verify/sanity/measure）。
- **clang-18 对称 + fp-contract=on**（有 fp reduction·同 scalar S1）·march=大 march·opp lib=`build-clang18-rv64gcv`·CLANG_WORLD。ours 叶=g7-census/vecdot-rvv/kernels（只读）·driver+oracle=新资产 `experiments/active/k-vecdot-harness/kquant_vecdot_driver.c`（独占·含 q4_K/q6_K 独立 ZERO-MODEL int-oracle）。

### C. verify（[K-5] 3-way byte-exact ZERO-MODEL · 4-arm 反空心 · 双格 GREEN）
| fmt | [A]clean 3-way | [B]inject-ours | [C]inject-oracle | [D]leaf-seed-fault | fp16 golden | OPP 档位 |
|---|---|---|---|---|---|---|
| q4_K | ALL=true worst_ulp=0 (ours=ggml=oracle=-2245) | BITES-OK (og=1 oi=1) | BITES-OK (oi=1 gi=1) | BITES-OK (og=64 oi=64·咬 emit) | PASS 9/9 | `_vl128` ins198/rvv105 **手调** |
| q6_K | ALL=true worst_ulp=0 (=12986) | BITES-OK | BITES-OK | BITES-OK (og=64·咬 emit) | PASS 9/9 | `_vl128` ins210/rvv84 **手调** |
- **ZERO-MODEL 真独立**：oracle 纯 int64 从原始块字节重算（q4_K 6-bit scale utmp 解包 + min 项；q6_K -32 offset + 2-bit qh 拼接），**不共享 bit→lane 解码**·byte-match BOTH ours AND 部署 ggml（三码路一致=反空心 by construction）。INT-mode 填充（d=1.0/dmin=0/nibble/±1）⟹ 精确整数 fold·序无关·byte-exact 恒。
- **OPP 家族身份探针**（§3.4·按家族一次）：`ggml_vec_dot_q4_K_q8_K`→ family[generic,vl128,vl256]·VLEN128 运行路径=`_vl128`（rvv>0=**手调档 STRONG**·非便宜 _generic）。q6_K family+vl512。

### D. cold measure（ratio_cold_X = 部署对手/ours·<1=ours 慢·2 seed·N=25·flush224MiB·core8 idle100% gov=performance）
| fmt | s1(0x1357) | s2(0xACE2) | verdict | ours_ns / opp_ns |
|---|---|---|---|---|
| q4_K | **0.1657** (iqr0.20%) | **0.1634** (iqr0.10%) | **具名-X LOSS** | 1322245 / 219081 |
| q6_K | **0.2041** (iqr0.09%) | **0.2020** (iqr0.14%) | **具名-X LOSS** | 2247189 / 458682 |
- 双格 byte-exact ALL=true（measure 内 verify 恒过）·cpu_md5 前后守恒·STRAY=0。**具名-X = weight-reconstruction floor vs 手调 vl128**（q4_K 与 g7-census gcc-15 0.167 一致印证；clang-18 世界 q6_K 0.20<census 0.283·新基线·成色如实）。
- **成色**：[NG-4] kernel-axis·非 e2e·对手 = 手调档 STRONG（非便宜档）·大幅 LOSS 如实·**攻坚 baseline**（供 K-attack-fanout item3/4 机制②③施工·非翻·非头条）。

### E. runs/ 落点 + 入账
- **未走 runner 全路**：`bench vec_dot …` 经 runner 会在 step3/4 落 VOID(解析不出·parser gemm_tile 专用=facet 3)⟹ **runs/ 自动落点 gated on parser 泛化**（判据级·未改 parser）。**未直写 master·未直写 runs.log·未 fabricate runs/ 条目**（禁绕 runner fail-closed 闸）。baseline 真数经本 journal + 报告交付·驱动/harness 资产耐久。
- **未动 sealed·无 git commit·无 add -A**。

### F. 触碰集 / 遗留
- **触碰（独占）**：新建 `tools/bench/cells/vec_dot.sh`（+x）· 新建 `experiments/active/k-vecdot-harness/{kquant_vecdot_driver.c,MANIFEST.md}`· append ISSUE-104 补录（`.trellis/spec/issues/门与工具.md`）· 本 journal。board rvv /tmp/bench_cells_vec_dot_*（临时·仓库侧零落盘）。g7-census/vecdot-rvv/kernels 只读未改。
- **后续攻坚该 harness 覆盖**：q4_K/q6_K（已接·oracle 已建）+ **待补 oracle 扩格 = q2_K/q3_K/q5_K**（叶已存 census·白名单 HARNESS-VOID 待补 int-oracle）。k1 分支 gated ISSUE-105（VLEN256 半宽 fixture）。
- **遗留**：① runner parser 泛化（vec_dot 标记族）= runs/ 自动入账前置·判据级 ISSUE-104 facet 3。② harness 粒度 op-全板 vs op×板未钦定·判据级 ISSUE-104 facet 2。③ q2_K/q3_K/q5_K oracle 待补（本 task 只 q4_K/q6_K baseline）。

---

## 2026-07-18 · task `07-18-07-18-r-dequant-nongrid`（dequant non-grid 真向量 emit 首格·q8_0@rvv·**PASS 真收口**）

**目标**：选一个 representative non-grid dequant 格 author owned 真向量 emit → 翻 PASS = 首个 owned-construction 标量门真收口（区别 grid 天花板 ISSUE-107）。前一 agent 网断·干净重启。

### A. 选格 = q8_0（非 q4_1）
- 读 `RVVToEmitC.cpp:906` isGgmlDequantizeRowBody 分叉：q8_0/q4_0/q4_1/q5_0/q5_1 全走 CONSTRUCTED typed-loop-body → 但 decode leaf 是 **scalar per-element loop**（`y[j]=qs[j]*d`·靠 host autovec = codegen-lottery·owned 向量 intrinsic=0）。
- **选 q8_0**：家族头（emit 基础最全·"最易先证"）· decode 最简（纯 `qs[j]*d`·**无 add/min ⟹ 无 fp-contraction 歧义**·byte-exact 最稳）·连续 32-lane 存·**非 grid ⟹ 无 vluxei gather 墙**。**弃 q4_1**：`x0*d+m` 在 `-ffp-contract=on` 下有 FMA-contraction byte 失配风险 + nibble deinterleave 更复杂。

### B. emit（owned 真向量·同 iq3_xxs 范式）
- 新增 `emitDequantizeRowQ8_0VectorBody`（`RVVToEmitCForwardElementwise.cpp`）：每块 ONE 32-lane pipeline = `vle8(i8m2) → vsext_vf4(i32m8) → vfcvt_f_x_v(f32m8) → vfmul_vf(d)(f32m8) → vse32`。宽度=固定 QK8_0=32 块几何·LMUL 派生非旋钮（[K-5] 读自结构·非焊死字面量致类型不匹配）。
- 路由：`emitTypedDequantizeRowLoopBody` decode_model=="q8_0" → vector body（仅 CONSTRUCTED 路·dispatch-wired monolith 保 scalar shared body·**iq3_xxs 先例**）。header decl 补 `RVVToEmitCInternal.h`。
- clean rebuild weft-opt（亲见 link OK·仅既存 warning）。生成叶 `experiments/active/r-dequant/kernels/q8_0_dequant.c`（39 行·leaf_md5=bef46c25…）。**owned intrinsic 前=0(scalar)→后=5 distinct/12 calls**（vsetvl 除外·ISSUE-001 反面）。

### C. harness 扩格 + 资产（独占）
- `cells/dequantize_row.sh` 扩认 q8_0：per-fmt DRV/LEAF/TBL(空)/GSED/OPPSYM/NB。q8_0 无 codebook table（skip scp）·FAULT anchor=`+ 2;`→`+ 1;`（quant 偏移·in-bounds·value-changing·`cmp -l|wc -l`==1 验证过）·OPPSYM=`dequantize_row_q8_0`·NB_MEASURE=4096（k=131072 与 iq3_xxs 工作集齐）。
- 新建 driver `experiments/active/r-dequant/q8_0_dequant_row_driver.cpp`（独立 ZERO-MODEL int8 oracle·NOTHING shared·corpus=全 256 int8 quant 值·3-arm inject）。**契约合规**：全 stdout·仓库侧零落盘。

### D. verify（`dequantize_row.sh rvv verify q8_0`·GREEN）
- **[A] CLEAN**：ours-vs-oracle **PASS 0/131072** · ggml-vs-oracle PASS 0/131072 · CORPUS quant_val **256/256 COMPLETE**。
- **OWNED probe**：`OWNED_src_vec_intrinsics=5 distinct/12 calls`（≫2·真向量 emit·gather=0）。
- **对手身份**：`dequantize_row_q8_0 [ins=419 non_vset_vector=194]` = **诚实重 autovec 部署核**（非弱 scalar）。
- **3-arm 反空心全咬**：[B]oracle-fault mism=1→RED · [C]DUT-fault mism=1@k/2·ggml-vs-oracle PASS→RED-only · [D]leaf-byte-fault mism=130426·ggml-vs-oracle PASS→RED-only（咬 EMITTED RISC-V）。cpu_md5 前后守恒·STRAY=0。

### E. cold measure（ratio_cold = 部署对手/ours·≥0.8=PASS·2 seed·N=25·flush224MiB·core8 idle100% gov=performance）
| fmt | s1(0x1357) | s2(0xACE2) | verdict | ours_ns / opp_ns |
|---|---|---|---|---|
| q8_0 | **2.3311** (iqr1.38%) | **2.2721** (iqr0.75%) | **PASS·WIN ~2.3×** | 185561 / 432561 |
- 双 seed byte-exact ALL=true（measure 内 verify 恒过）·cpu_md5 守恒·STRAY=0·load-gate 过。
- **★verdict = PASS 真收口**（cold ~2.30 ≫ 0.8）= **首个 standalone 标量类硬门格 owned-construction 真收口**（de-lottery：interim autovec 值 → determined owned 真向量测定值）。
- **non-grid 真无 gather 墙 = 确认**（leaf gather=0·2.3× WIN·对比 grid iq3_xxs @0.36 天花板 ISSUE-107）。
- **成色**：对手=部署 `dequantize_row_q8_0` **标量类档 autovec**（§〇.1·honest 重 autovec 194 vec ins·非便宜 scalar·非手调 vendor）。[NG-4] kernel-axis·非 e2e。合法标量类硬门 PASS + construction [L-8] 真赢。**勿以此外推 grid 族或 K-quant 位重建格**。

### F. lit / quality
- 改 3 lit 测（q8_0 CONSTRUCTED 路全经 vector body）：`rvv-to-emitc-ggml-dequantize-row-q8-0` / `rvv-to-emitc-typed-dequantize-row-loop-body`（原 float-`mul` CHECK 失配→改 vector intrinsic CHECK）/ `rvv-dequantize-row-stream-front-door-construct`（EMIT 段补 vector CHECK）。叙事订正（constructed=vector·monolith=scalar shared）。
- **lit 全绿**：RVV Conversion **284/284 PASS**（proper lit runner）。全树 954 中 3 失 = `Scripts/rvv-generated-bundle-abi-e2e-*` **product-reduction 纯 Python self-test**（并行 K-product-reduce 线 WIP·**与本改无关**·我 touch 集不含该 Python·纯 Python self-test 不受 C++ emit 改动影响）。iq3_xxs 回归干净。

### G. 入账留 main 会话（遵 handoff）
- **未直写 master·未直写 runs.log·未动 sealed 9/83·无 git commit·无 add -A**。上面 q8_0 cold 双 seed 原始数供 main 走 recon-dict 入账（dequantize_row|q8_0@rvv：interim autovec → **真测 PASS 2.33/2.27·WIN**·verdict 变动）。
- 未走 bench runner 全路（step5 耦合 master 直写 = ISSUE-091 gated + parser dequant 族已认但 runner 落点 gated ISSUE-090/091）；按 task 直调 harness 取 raw（= bench `_run_harness` 同路）。

### H. 触碰集 / 遗留
- **触碰（独占 dequant emit + q8_0 格）**：`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`（+`emitDequantizeRowQ8_0VectorBody`·路由 1 分支）· `RVVToEmitCInternal.h`（decl）· 3 lit 测 · `tools/bench/cells/dequantize_row.sh`（扩 q8_0）· 新建 `experiments/active/r-dequant/q8_0_dequant_row_driver.cpp` + `kernels/q8_0_dequant.c` · append `GEN_SEAL.txt` q8_0 段 · 本 journal。board rvv /tmp/bench_cells_dequantize_row_rvv_q8_0（临时·零落盘）。**未碰 GridCodebook.cpp / cells/product_reduce.sh / cells/vec_dot.sh（并行线）**。
- **遗留**：① non-grid 扇出（q4_0/q5_0/q5_1/q8_0✓ + q4_1 须处理 fp-contract）逐格 byte-exact+板测·禁外推。② @k1 gated ISSUE-105（VLEN256 半宽）。③ grid 族 5 格天花板 gated ISSUE-107（三选一战略裁）。④ q4_1 nibble+min 真向量若做须解 `x0*d+m` FMA-contraction 一致性（vfmadd vs vfmul+vfadd）。

---

## Session · K线 手调档 q6_K vec_dot 完整攻坚环（2026-07-18·task `07-18-07-18-k-q6k-vecdot-attack`）

**verdict = <0.8 → 经环具名 + 三档墙（同 q4_K weight-reconstruction floor·[MECH-WEIGHT-RECONSTRUCTION-BOUND] 2nd 板测数据点·泛化确认·mechanism③ 2nd 实例）。best m1 0.214 << 0.8。零净源码改动（旋钮 wiring 实验后 revert·repo byte-identical）。**

### A. 完整攻坚环（禁跳环·全走·板 rvv·亲见）
1. **①解剖（objdump·亲见）**：ours q6_K vec_dot leaf（mf2 默认·sealed md5 4de70f5162）= scalar 413 / vector 265 / **vset 70** / non_vset_vec 195；**70×vle8 + 8×vse8（aux8[256] scratch 权重重建 store）** + 32 vwmul + 32 vwmacc（16 sub-block × 2 halves·AVL=8 mf2）+ 12 vand/10 vsrl/8 vsub/8 vsll/8 vor（6-bit ql+qh 解包·−32 offset）+ 1 vfmul/1 vfcvt/1 vfadd（deferred fp32 fold·**NO min-term**·single accumulator）。OPP `_vl128`（caliber=手调 STRONG·机判）= ins 210 / non_vset_vec 84·**0×vse8（register-resident·无 scratch store）**·**16×vwredsum.vs**（per-sub-block 归约）+ 12 vmacc/4 vwmul/4 vmul。→ 成本中心 = **aux8[256] 权重重建 store→load roundtrip**（同 q4_K）·**唯一结构差异 = q6_K 无 min-term serial MAC**（q4_K 有 16-deep scalar min-term·q6_K single-accum no-min）。
2. **②构造（mechanism③ (SEW,LMUL,VLEN) 参数化）**：q6_K 旋钮**未接线**（`emitTypedSuperBlockScalesTimesSumiLoopBody` 的 cx6 硬编码 `coreLmul=mf2`·line 3486-3494·折叠逻辑在 `emitQ6_KSuperBlockAux32Core` line 145-327 已存但不可达）。**临时 wiring**（读 `loopBody.getIntegerCoreLmul()` → deriveWideningChain → cx6·同 q4_K 2964 dual-body 范式·54-line diff `scratchpad/q6k_knob_wiring.diff`）使旋钮生效 → 前门 IR 注入 `integer_core_lmul="m1"/"m2"` → regen leaf。**mechanism② clang-18 已是 harness baseline。**
3. **③前门 byte-exact（[K-5] ZERO-MODEL·board·亲见）**：
   - **mf2**（sealed）: `BYTE_EXACT ALL=true`·worst_ulp=0（regen byte-identical to sealed·regression-safe）。
   - **m1**（vset 39）: `BYTE_EXACT ALL=true`·worst_ulp=0·anti-hollow 3-arm BITES-OK。fold-back（16-lane strip → vslidedown 8 + vadd + vget canonical 8）byte-exact。
   - **m2**（vset 36）: `BYTE_EXACT ALL=false`·worst_ulp 2.4e9（bits_ours=0xc6415800 negative garbage）= **q6_K sub_block=16 STRUCTURAL cap**：32-lane strip 跨 sub-block scale 边界（wrong-scale fold）+ aux8[256] over-read。⟹ **m2 byte-INVALID·排除（byte-exact 铁律·非数据点）**·q6_K 旋钮封顶 m1 = 比 q4_K（sub_block=32·支持 m2）**窄一档**。
4. **④板测 cold（rvv·2seed·N=25·flush 224MiB·load-gated idle 100%）**：
   - **mf2**: vset 70·ratio **0.2025/0.2016**（ours_med 2246509/2248209 ns·opp 454822/453262）= baseline 具名-X（对 PRD 0.202 复现）。
   - **m1: vset 39·ratio 0.2142/0.2149（最佳有效·ours_med 2115509/2114129 ns·byte-exact）** — 比 mf2 快 ~6% wall-clock。
   - m2: byte-INVALID·不测。
5. **⑤ <0.8 → 具名墙**（best m1 0.214 << 0.8）。

### B. 关键诊断（perf stat·board·亲见·为何旋钮无效）
- objdump 证 vset 真降 **70→39**（re-roll-trap gate PASS·真宽化 vint32m2→m4·非空转）·但 cold ratio 仅 0.202→0.214（旋钮减指令 vset 44%·wall-clock 仅 6%·**disproportion = stall-bound 签名**）。
- **perf stat（m1·flush 224·blended）= IPC 0.12 · 86.87% backend-idle · 84.52% frontend-idle · 7.30 stalled-cycles/insn · 191.6M cache-miss**。**mf2 同（IPC 0.12·86.88% backend-idle）**。
- **flush=0 隔离（reps=50·ours+opp 无 flush 噪声）**：mf2 **1,556,441,617 insn** / 15.16B cycles vs m1 **1,527,331,992 insn** / 15.10B cycles（IPC 0.10·cache-miss 375M）= 旋钮减指令 ~29M（1.9%）·cycles 仅降 0.4% ⟹ **核 memory/dependency stall-bound·非 instruction-throughput-bound**·旋钮减指令对 stall-bound 核**无效**。
- **q6_K IPC 0.12 略高于 q4_K 0.08** = q6_K 无 min-term serial dependency chain（single accumulator）·与 q6_K ratio 略高（0.21 vs q4_K 0.19）自洽。

### C. 三档墙分类（对手 = 手调 `_vl128` STRONG·机判 caliber=手调·成色如实）
1. **可攻坚旋钮**：compute-width `integer_core_lmul`（mf2/m1）·byte-exact + 减指令（vset 70→39）proven·但 cold 平（0.202→0.214）·**EXHAUSTED·best 0.214 << 0.8**。**q6_K sub_block=16 封顶 m1**（m2 byte-INVALID·比 q4_K 窄一档）。
2. **我方内禀墙**：aux8[256] scratch **store→load roundtrip**（8×vse8 + 70×vle8）。perf stat IPC 0.10-0.12·86.87% backend-idle·375M cache-miss ⟹ stall-bound。(SEW,LMUL,VLEN) 旋钮**不触碰**·= [MECH-WEIGHT-RECONSTRUCTION-BOUND] floor **2nd 板测坐实**。**★关键泛化证据：q6_K 无 min-term（q4_K 有）·仍同 floor ⟹ 墙 = 权重重建 roundtrip 本身·非 min-term serial MAC。**
3. **对手结构优势**：OPP register-resident（0 scratch store）+ **16×vwredsum.vs** per-sub-block 归约。翻此墙须**结构级 register-fusion（unpack↔compute 融合·消 aux8 roundtrip·非旋钮·独立大构造·未施工）**。

### D. 触碰集 / 决策 / 遗留
- **触碰（读+实验+revert·零净仓库源码写）**：`RVVToEmitCKQuant.cpp`（临时 wiring cx6 旋钮·54-line diff·**已 revert·repo byte-identical·weft-opt clean rebuild 复原·mf2 regen md5 4de70f5162 复验**）· 跑 `tools/bench/cells/vec_dot.sh`（rvv verify+measure·m1/m2 leaf 全 scratchpad `VEC_DOT_KERNEL_ROOT`）· weft-opt IR-injection 全在 scratchpad。**未碰 sealed leaf**（q6_K.kernel.c md5 4de70f5162 维持）·**未碰 master/census/git**·**未碰并行线**（`RVVToEmitCForwardElementwise.cpp`/`GridCodebook.cpp`）。
- **★revert 决策**：旋钮 EXHAUSTED（无部署价值）+ 保留 wiring 会使 **m2 对 q6_K reachable-but-byte-INVALID**（footgun·revert 前 m2 被 cx6 静默忽略=安全）·加 m2 guard = 为死旋钮过度工程 ⟹ revert（同 q4_K「零源码改动」范式·clean tree·diff 存档可复现）。**若主会话要 q6_K 旋钮 parity（对齐 q4_K/q5_K）·须叠加 m2 sub_block guard·主会话裁。**
- **入账**：`experiments/active/result-tables/K-attack-fanout-ledger.md` 机制③行 + ★诚实边界（q6_K 2nd 数据点·floor 泛化确认）。
- **遗留 / 交主会话**：① **[MECH-WEIGHT-RECONSTRUCTION-BOUND] floor 现有 2 板测数据点（q4_K 0.186 + q6_K 0.214·同结构·同 stall-bound·q6_K 无 min-term 仍同 floor = 强泛化证据）** ⟹ 主会话可 **mechanism-level 具名上报 q2_K/q3_K/q5_K vec_dot 群**（同 aux8 权重重建结构·likely 同 floor·ISSUE-109·PRD §四遗留授权）而非逐格补 oracle 走环（待主会话裁）。② 翻 0.8 唯一已识别 lever = register-fusion（消 aux8 roundtrip·匹配 OPP register-resident·独立结构级大构造·≠ISSUE-100 架构不可达·= emitter-maturity 缺口·in-principle 可达·未施工）。③ @k1 gated ISSUE-105。

---

## Session · K线 手调档 q4_K vec_dot 完整攻坚环（2026-07-18·task `07-18-07-18-k-kquant-vecdot-attack`）

**verdict = <0.8 → 经环具名 + 逐指令墙（三档·带 objdump+perf 证据·同 ISSUE-100 nvfp4 范式·≠架构不可达）。best m1 0.186 << 0.8。零源码改动。**

### A. 完整攻坚环（禁跳环·全走）
1. **①解剖（objdump·board rvv·亲见）**：ours q4_K vec_dot leaf（默认 mf2）= scalar 445 / vector 225 / **vset 70** / non_vset_vec 155；68×vle8 + **8×vse8（aux8[256] scratch 存）** + 32 vwmul + 32 vwmacc（AVL=8 mf2·4 quarter/subblock）+ 16 lh/16 add（scalar min-term）。OPP `_vl128`（caliber=手调 STRONG）= ins 198 / non_vset_vec 105·**0×vse8（无 scratch store·register-resident）**·8×**vwredsum.vs**（per-subblock 归约）+ vectorized min-term。→ 成本中心 = aux8 权重重建 store→load roundtrip + scalar min-term + vsetvl storm。
2. **②构造（mechanism③ (SEW,LMUL,VLEN) 参数化）**：= 既存 `integer_core_lmul` 旋钮（emitter `RVVToEmitCKQuant.cpp:2217` deriveWideningChain·front-door `RVVMonolithicBlockDotSourceFrontDoor.cpp:1229` 默认 LEFT-OFF=mf2）。**零源码改动**：weft-opt 前门 IR dump → 注入 `integer_core_lmul="m2"/"m1"` 到 `weft_rvv.q4_k_scaled_dot` → 续 lower → mlir-translate。mechanism② clang-18 已是 harness baseline（无额外增益）。
3. **③前门 byte-exact（[K-5] ZERO-MODEL·board）**：m1+m2 两档 `BYTE_EXACT ALL=true`（ours==oracle==ggml·worst_ulp=0）·anti-hollow 3-arm BITES-OK。fold-back（32/16 wide lane → canonical 8）设计使宽化 byte-exact。
4. **④板测 cold（rvv·2seed·N=25·flush 224MiB·load-gated）**：
   - mf2（sealed 默认）: vset 70·ratio **0.162**（1321985/1320945 ns·opp 213881/215961）
   - **m1: vset 39·ratio 0.186（最佳·1173145/1173825 ns）· byte-exact**
   - m2: vset 20·ratio 0.168（1293005/1292265 ns·byte-exact）— **i32m8 累加器 register-cliff → 比 m1 差**
5. **⑤ <0.8 → 具名墙**。

### B. 关键诊断（为何旋钮无效）
- objdump 证 vset 真降 **70→39→20**（re-roll-trap gate PASS·非空转·真宽化）·但 cold ratio **平且非单调**（0.162→0.186→0.168）。
- **perf stat（board·ours-dominated blended）= IPC 0.08 · 96.2% backend-idle · 95.8% frontend-idle · 2.2B cache-miss** ⟹ 核是 **memory/dependency stall-bound**·**非 instruction-throughput-bound** ⟹ 旋钮减指令对 stall-bound 核**无用**（同理 min-term 向量化亦无用·非 memory 瓶颈）。perf record 采样在此板 PMU 不可用（counting-only·同 memory-note k1 无 sampling·rvv 亦 record 无 sample）·用 perf stat 聚合。

### C. 三档墙分类（对手 = 手调 `_vl128` STRONG·成色如实）
1. **可攻坚旋钮**：compute-width `integer_core_lmul`（mf2/m1/m2）·byte-exact+减指令 proven·但 cold 平·**EXHAUSTED·best 0.186**。
2. **我方内禀墙**：aux8[256] scratch **store→load roundtrip**（8×vse8 + 68×vle8）+ scalar min-term 16-deep serial MAC。(SEW,LMUL,VLEN) 旋钮**不触碰**·= [MECH-WEIGHT-RECONSTRUCTION-BOUND] floor 板测坐实（ledger 旧「repack prefill 0.94×parity/M=1 无行摊销更难」预测在 vec_dot M=1 证实）。
3. **对手结构优势**：OPP register-resident（0 scratch）+ vwredsum.vs + vectorized min-term。翻此墙须**结构级 register-fusion（unpack↔compute 融合·非旋钮·独立大构造·未施工）**。

### D. 触碰集 / 遗留
- **触碰（读+实验·零仓库源码写）**：读 `RVVToEmitCKQuant.cpp`（emitter 机制）+ `RVVMonolithicBlockDotSourceFrontDoor.cpp`（front-door 旋钮点）+ `RVVToEmitCSupport.cpp`（deriveWideningChain）· 跑 `tools/bench/cells/vec_dot.sh`（B 建·rvv verify+measure）· weft-opt IR-injection 实验全在 scratchpad。**未碰 sealed leaf**（`experiments/.../kernels/q4_K.kernel.c` md5 892b6cf8 维持·regen byte-identical 已验）·**未碰 master/census/git**·**未碰并行线**（`RVVToEmitCForwardElementwise.cpp`/`GridCodebook.cpp` 的 M 是 R线 dequant·非我）。
- **入账**：`experiments/active/result-tables/K-attack-fanout-ledger.md` 机制③行 + 状态行 + ★诚实边界（board-tested 负结果·§二.4 机制入库）。
- **遗留 / 交主会话**：① 部署问题 = 若 m1（best 0.186·仍 <0.8）不值部署（<PASS）·具名-X 维持。② **翻 0.8 的唯一已识别 lever = register-fusion（消 aux8 roundtrip·匹配 OPP register-resident dataflow）= 独立结构级大构造**·非本 knob-攻坚 scope·主会话可裁是否排队/登 ISSUE（≠ISSUE-100 架构不可达·此为 emitter-maturity 结构缺口·in-principle 可达）。③ 扇出 q5_K vec_dot 同结构（aux8+min-term）likely 同墙·未证·禁外推。

---

## Session · C1线 [D-2a] A+B 施工（装载期解析器 + 每进程一条记录·2026-07-18·task `07-18-07-18-c1-d2a-ab`）

**verdict = [C1-1] 头牌后半句「起步」兑现（A+B 落地·纯工·byte-exact·lit 无回归·0 造数·未触 ISSUE-105/GEN_SEAL/master）。C（运行期归因链）+ D（部署通道）仍遗留。**

### A. 装载期解析器（[D-2a] 步①②③④·复用 hash 算法）
- 新件 `include/Weft/Support/LoadTimeCapabilityResolution.h` + `lib/Support/LoadTimeCapabilityResolution.cpp`：
  - `resolveLoadTimeCapabilities(facts, candidates, ts)` = **纯解析器**：消费展开后事实集（buildFromKernel* 已展开 profile）→ **复用 `computeDeclaredInstanceHash`**（zero 重实现 hash·profile==显式列表同哈希继承）→ 算 fail-closed resolved_variant_set。
  - resolved 集键控走 `TargetCapabilitySet::isCapabilityAvailableByID`（能力事实·**I3 零 family/symbol 名分支**）；absent/unknown guard → 排除（**I7 fail-closed·[S-2] 默认拒**）；空可用 guard → 空集（不合成 route）。sort+dedup = order-invariant I4 镜像。
- **机检（unittest `weft-load-time-resolution-test`·GREEN）**：A profile==显式同哈希（instance_ab==instance_ba·符号名/序无关·不同事实集判别性不同哈希）；I3 同事实不同符号名 → 相同 resolved 集 + 「rvv_wide(available fact) IN / generic_narrow(unavailable) OUT」证 fact-not-name；I7 unavailable+unknown 双排除 + 空集不合成；serialize canonical shape。

### B. 每进程一条记录（[D-4]② 硬门=每进程 1 条）
- `LoadTimeResolutionCache`（owns `std::once_flag`）= per-process 机制：首 resolve() 算+缓存·后续调用返回同一缓存记录**零重算零逐次校验**（**[NG-3] compute-once·热路径查缓存**）。unittest 128 次 dispatch → `resolveCount()==1` 机检。
- 记录形状 `{declared_instance_hash, ts, resolved_variant_set}` + `serializeLoadTimeResolutionRecord`（canonical JSONL·固定键序）。
- **CI 门 `tools/gates/check_d2a_resolution_record_jsonl.py`**（类比 check_f4·stdlib-only·--self-test + real 双模）：
  - self-test 12/12 判别性 GREEN（缺键/非hex/未排序/重复/非string/非list/非string-ts → RED；per-process **恰 1 条**·2 条=per-dispatch 泄漏[NG-3]→RED·0 条→RED）。
  - real 模式跑 `weft-load-time-resolution-test --emit-jsonl` → **恰 1 record·shape 合规 GREEN**；binary 非零退出（I3/I7/NG-3 守法 machine-check 挂）→ **RED**（已用 `/bin/false` 实证 RED·transitive 守法覆盖）。

### C. 红线守法（负控实证·research §5 施工契约）
- [NG-3]：cache once_flag·热路径查缓存·门检 per-process 恰 1（>1→RED）·未引入逐次检查（DispatchRuntimeGuard.cpp 未碰）。
- I3：resolved 键控 `isCapabilityAvailableByID`·符号名不变量机检·零 family 分支。
- I7：absent/unknown→排除·空集不合成·`isCapabilityAvailableByID` unknown→false 天然 fail-closed。
- [部署§六.13] 两层降级：新件头文件明标「现系统事实=编译期 fail-closed / 本 task 实现=装载期解析器 COMPONENT+shape+per-process 门(起步) / [D-2b]/M2 未实现=live per-process drop 入部署二进制·禁描述为现状」。

### D. 验证结果
- Lint/TypeCheck N/A（C++/CMake 项目）。**Build**：weft-load-time-resolution-test + weft-opt clean rebuild GREEN（[[build-incremental-unreliable]] 亲见·libWeftSupport.a 重链·全 consumer link OK）。
- **lit**：Support/ + Transforms/VariantSelection/ 9/9 PASS（含新 `load-time-capability-resolution.test` + 既有 attribution-jsonl-instance-hash 哈希路径无回归）。**check_f4 gate**：self-test + real 5/5 GREEN（共享 DeclaredInstanceHash 编译期路径 byte-exact 未动）。

### E. 触碰集 / 不触确认 / 遗留
- **触碰（新增 5 件 + 加性 3 编辑·独占·与并行线 R-dequant/K-regfusion 不相交）**：新 `include/Weft/Support/LoadTimeCapabilityResolution.h`·`lib/Support/LoadTimeCapabilityResolution.cpp`·`test/Support/LoadTimeCapabilityResolutionTest.cpp`·`test/Support/load-time-capability-resolution.test`·`tools/gates/check_d2a_resolution_record_jsonl.py`；加性编辑 `lib/Support/CMakeLists.txt`(+1)·`test/CMakeLists.txt`(+14)·`test/lit.cfg.py`(+1)。
- **不触确认**：未碰 GEN_SEAL/master/census/harness（git status 中 `GEN_SEAL.txt`/`RVVToEmitC*`/`iq3_xxs*` 的 M 全属并行 R-dequant 线·非本 task）。未触 ISSUE-105 部署判据（未改宽度分发/GEN_SEAL·k1 具名-X 0.6474 / sha256 dc876ef6 守恒）。未 commit·未 add。
- **遗留 / 交主会话**：① **C = [D-4]③ 运行期归因链**（runtime hwprobe 消费者→事实→hash 键控 dispatch·整层新建·最大·红线密集）= 后续 task·**须先固本 task 的 [NG-3] 守法门作施工契约**（防退化 per-dispatch）——本 task 已把守法门备好。② **D = ISSUE-105 部署通道**（一份二进制多宽度+装载期选宽+per-board re-measure）= 判据级·待裁·单独走·不耦合。③ C1 头牌后半句**完整契约含 C**；本 task = A+B「起步」兑现·[D-2b]/M2 live-drop-into-shipped-binary 显式 deferred（未描述为现状）。

## Session · K线 ISSUE-109 register-fusion q4_K vec_dot（2026-07-18·task `07-18-07-18-k-regfusion-q4k`）

**verdict = <0.8 → 具名-X 维持（cold 0.152·byte-exact worst_ulp=0）。register-fusion 已试·re-roll trap PASS（aux8 roundtrip 真消）·但 WALL 未动 = 机制级新发现：墙 ≠ aux8 roundtrip·墙 = latency/dependency-bound。lever 清单缩·新 lever = vwredsum 结构（未试·≠架构不可达）。**

### A. 构造（结构级 register-fusion·gated·default 不变）
- 新 emit `emitQ4_KFusedUnpackScaledDot`（`RVVToEmitCKQuant.cpp`）：融合 BRICK1(unpack→aux8) + BRICK3(reload aux8→MAC)。per 32B packed chunk（4 chunk）unpack 两 nibble 半（各 = 32-elem sub-block）**留寄存器**·直接 vwmul i16m4 × sub-block q8（m2 载）→ vwmacc i32m8（scale fused）→ 宽 32-lane aux32·尾 VLEN-agnostic fold-back（vslidedown+vadd+vget）到 canonical-8。**消 aux8[256]·消 8×vse8 存·消 32×vle8 权重 reload**。
- gated behind `integer_core_lmul="fused"`（BRICK3 属性·op verifier `RVVDialectWideningOps.cpp` 加性接受·default mf2 不变）·`!hasQh`（q5_K 不融合·留 aux8 路）。**default q4_K/q5_K emit BYTE-IDENTICAL to sealed（regen 验证·609/609 RVV lit PASS·零回归）**。
- 复现管线证：sealed q4_K.kernel.c md5 892b6cf8 未动；fused leaf 经 weft-opt(IR 注入 fused)→mlir-translate 生成到 scratch KERNEL_ROOT（sealed 不碰）。

### B. re-roll trap（objdump 板 rvv·亲见·真消非 re-shape）
| | baseline mf2（sealed） | fused |
|---|---|---|
| vse8.v（aux8 存） | **8** | **0** ✓真消 |
| vle8.v（含 32 aux8 reload） | **68** | **12**（4 packed + 8 q8·reload 全消）✓ |
| vwmacc / vwmul | 32 / 32 | 8 / 8 |
| vsetvl | 66 | 18+3 |
| scalar_ins / vector | 445 / 225 | 248 / 71 |
| 残留 | — | vs8r.v/vl8r.v ×1（i32m8 累加器单次 spill·m8 register-cliff·≪ 原 roundtrip） |

### C. byte-exact（[K-5]·ZERO-MODEL·harness INT-fill 序无关免费）
- `bits_ours=0xc50c5000 == bits_ggml == bits_int_oracle`·**worst_ulp=0**·`BYTE_EXACT ALL=true`·3-arm anti-hollow 全 BITES-OK。整数 fold 序无关（fill dmin=0·d=1.0·total<2^24）⟹ chunk-interleave 重组 = m2 anchor per-lane 整数和 bit-identical。ULP 界 = **0**（无浮点尾巴税·此 fill）。

### D. 板测 cold（rvv·2seed·N=25·flush 224MiB）+ perf stat = 机制级新发现
- **cold ratio 0.1521 / 0.1516**（vs baseline mf2 0.162 / m1-best 0.186 / m2 0.168）= **略差·未动墙**。绝对 ours ≈1.425M ns（baseline mf2 ≈1.337M ns·fused 反而慢 ~7%）。
- perf stat（ours-dominated blended）：**IPC 0.08→0.17（2.1×↑）·cache-miss 2.2B→66M（33×↓·L1d-load-miss 45M）**。⟹ 指令效率与内存流量大幅改善·**wall-time 不动**。
- **★机制级裁断**：aux8 roundtrip **不是墙**。消它（objdump+perf 双证真消·33× fewer miss·2× IPC）**未移 wall** ⟹ 墙 = **latency/dependency-bound**（cold DRAM 载延迟暴露 + 串行 i32m8 累加器依赖链 + m8 register-cliff 单次 spill·融合反而**收紧临界路径**·aux8 scratch 曾起 decoupling/overlap 作用）。对手 `_vl128` register-resident + **vwredsum.vs per-sub-block**（独立归约·ILP/MLP 高·无长累加器链）。
- **[MECH-WEIGHT-RECONSTRUCTION-BOUND] 精修**：weight-reconstruction *memory traffic* 非成本中心；成本中心 = *load-latency + accumulator-dependency*。register-fusion lever = **EXHAUSTED（板测负结果）**。

### E. 触碰集 / 遗留
- **触碰（本 task·4 件·独占 K-quant emit TU）**：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（新 `emitQ4_KFusedUnpackScaledDot` + 单调 monolith 分支）·`RVVToEmitCInternal.h`（声明）·`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`（verifier 加性 "fused"）·`test/Dialect/RVV/q4-k-scaled-dot-dataflow.mlir`（verifier 消息更新）。**未碰并行线**（`RVVToEmitCForwardElementwise`/`GridCodebook.cpp`/iq3_xxs/dequantize_row.sh 的 M 全属并行 R-dequant 线）。未碰 sealed/master/census/git。未 commit·未 add。
- **遗留 / 交主会话**：① register-fusion = 板测负结果（真消 roundtrip 未动墙）⟹ **ISSUE-109 characterization 须更新**：wall ≠ aux8 roundtrip·= latency/dependency-bound·register-fusion lever EXHAUSTED（governance 级 issue 写·主会话裁）。② **新 lever = vwredsum.vs per-sub-block 归约结构**（匹配 OPP dataflow·独立归约 ILP/MLP·无长 i32m8 链·未试·≠架构不可达·lever 清单非空）。③ fused emit 现为 dormant gated 能力（default 不变）·主会话裁 keep-dormant / 迭代 vwredsum / revert。④ q6_K 同面（PRD 遗留）；@k1 gated ISSUE-105。
