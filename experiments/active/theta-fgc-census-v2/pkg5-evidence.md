# 包5 · 实证包(板间行为差异现成数据) — θ=f(g,c) 三角色全景盘点 v2

HEAD pin = `d173f4c2e2d8eda79d5fd5e2a79eb0f9b925a80c`

> 只读·零施工·零板时·禁下结论。每条原文照抄 + file:line/run-id 出处。
> 框架换算备注(仅供对齐·非判断)：`half_lanes` / `AVL` / `vwmacc 计数` / `strip 宽` = **θ 输出**；`VLEN`/`march`/`rvv_version`/`supported_lmul`/`fractional-LMUL 有无` = **c 板侧输入**；`sub_block`/`n_sb`/`码本尺寸` = **g 格式侧输入**。以下不做「焊死/公式/缺陷」判断，只列事实。

---

## 项1 · 宽化战役逐格数（BASE/WIDE·改了哪些参数·objdump 计数）

### 1a. 逐格 cold_X（VLEN128 BASE → VLEN256 WIDE·k1）
来源：`.trellis/spec/issues/门与工具.md:393`(ISSUE-105·RESOLVED)、`:406`(扇出) + `experiments/active/result-tables/K-attack-fanout-ledger.md:13`(机制①)。

原文照抄（ISSUE-105 `门与工具.md:393`）：
> **RESOLVED（部署完成·2026-07-18·★地盘+4）** … **4 格 deployed PASS**：iq3_xxs 1.38 / iq3_s 1.21 / iq1_s 1.34 / iq1_m 1.79（master k1 具名-X→PASS … byte-exact 4-arm·objdump 真宽 AVL8→16 非 re-roll…）·**counts 不变**（certified 101/108·perf-covered 9/83·hand-brick 2·k1 标量类 headline 43→47）

原文照抄（扇出 `门与工具.md:406`）：
> **扇出（★2026-07-18 已证 3 格·a6ad CONFIRMED）**：`iq3_s@k1` 0.61→**1.20 WIN** · `iq1_s@k1` 0.59→**1.34 WIN** · `iq1_m@k1` 0.59→**1.77 WIN**（2.94× compound·spill-storm 810→42 清零）——**同 VLEN256 半宽病·逐格 byte-exact + 2-seed 板测·真宽 objdump 证·非外推**。iq2_xxs/iq2_xs/iq2_s/iq4_nl objdump-真宽但 harness 未建

原文照抄（iq3_xxs 逐指令·`门与工具.md:398-401` + ledger `:13`）：
> `iq3_xxs@k1` … → **cold 0.65 LOSS → 1.38 WIN** … objdump 真宽（AVL `vsetivli zero,8,e32,m2`→`zero,16`·vset 5397→2515·gather 1024→512·非 re-roll）·2 seed（1.3838/1.3782）

BASE→WIDE 逐格（汇总·全数字原文·k1 板）：

| 格 | BASE cold(VLEN128 half_lanes=8) | WIDE cold(VLEN256 half_lanes=16) | 出处 |
|---|---|---|---|
| iq3_xxs@k1 | 0.65 (0.6474) | 1.38 (2seed 1.3838/1.3782) | 门与工具.md:399-401·ISSUE-005 性能与测量.md:45 |
| iq3_s@k1 | 0.61 (0.6136) | 1.20 (部署后 1.21) | 门与工具.md:406·:393 |
| iq1_s@k1 | 0.59 (0.5919) | 1.34 | 门与工具.md:406·:393 |
| iq1_m@k1 | 0.59 (0.5945) | 1.77 (部署后 1.79) | 门与工具.md:406·:393 |

改了哪些参数（θ 变化·原文）：`march=rv64gcv`(half_lanes=8·VLEN128) → `march=rv64gcv_zvl256b`(half_lanes=16·VLEN256)；构造/emit-time fixture 选择（门与工具.md:393「`deriveRepackHalfLanes` 已存·仅换 emit march」）。

objdump 计数（iq3_xxs·BASE→WIDE·原文）：
- AVL：`vsetivli zero,8,e32,m2` → `zero,16`（8→16）
- vset：5397 → 2515（门与工具.md:401）／另处 5397→2515（ledger）
- gather：1024 → 512
- spill-storm（iq1 系 compound）：810 → 42（门与工具.md:406）

诊断处 objdump（机制①诊断·`性能与测量.md:347-349`·ISSUE-102）原文：
> OURS iq3_xxs leaf.o **两板逐一相同** = ins=15238/15240·**vset=5397**（配置分布 `e16,m1`×2698 + `e8,mf2`×2688 交替 = e8↔e16 宽度抖动）· 标量 sp-spill=56 · 向量整寄存器 spill(vs*r/vl*re)=77
> 对手 `vec_dot_iq3_xxs_q8_K_vl256`：**vset=41**·宽 LMUL（`e8,m2`=64 lane / `e32,m4` / `e16,m2`）+ AVL 对齐 VLEN256（`zero,16,e8,mf2`=vl16 满）

### 1b. iq2/iq4 宽化战报（成色 upgrade / 具名-X 墙）
来源：`K-attack-fanout-ledger.md:5`。原文照抄：
> **★iq2/iq4 harness 建成+板测(裁决2·af60 复核)**：iq2_xxs/iq2_xs/iq2_s **成色 upgrade**(scalar-ref 便宜档→部署 OPP-X vl256 手调·真硬赢 CROSSOP·同算子 OPP-S 缺席·PASS 计数不增非新 flip)·iq4_nl **具名-X 墙**(同算子 OPP-S hand-brick beats 2.35×·widening 解 narrow-vl 墙①但 codebook-gather 墙② ISSUE-021 立·非硬赢)

### 1c. 宽化脚本 run-id 指针（现成·未新跑）
- `experiments/active/g8-stage3-attack/P2-grid4-raw/run_widen_flip_p2.sh`（iq3/iq1 grid4 宽化 flip 驱动）
- `experiments/active/g8-stage3-attack/P1-remainder-raw/run_widen_flip_iq4nl.sh`（iq4_nl 宽化 flip）
- run 目录 `experiments/runs/20260717T193812Z-iq3_xxs-k1-a4db452a/`（objdump 前后 + verify/measure stdout + leaf_iq3_xxs_vlen256_half16.c + row.csv·门与工具.md:413 指针）

---

## 项2 · T-P 全行 + T1d 双实例（原样）

### 2a. T-P（`experiments/active/result-tables/T-P_construction_param_chain.csv`·全 16 行）
表头：`cell,leg,capability_fact,plan_param,closed_form,closed_form_site,predicted,observed,product_form,consistency,leg_status,repro_cmd`

**q3_K@rvv 链（half_lanes = θ·随 march=c 输入变）·原文照抄关键列**：
- `1_fact_to_plan_closedform` 闭式：`half_lanes = (minVLEN<128) ? 0 : min(minVLEN/16, 16)`，site `RVVLowerQuantContraction.cpp:916-921`
  - march=`<empty>` ⇒ deriveMinimumVLEN=0 ⇒ predicted 0 / observed 0 / `no repack op (block-dot stub, fail-closed)` / MATCH（行2）
  - march=`rv64gc` ⇒ minVLEN=0 ⇒ 0 / 0 / fail-closed / MATCH（行3）
  - march=`rv64gcv_zvl64b` ⇒ minVLEN=128 ⇒ 8 / 8 / `numHalves=2 disjoint strip(s)` / MATCH（行4）
  - march=`rv64gcv_zvfhmin` ⇒ minVLEN=128 ⇒ 8 / 8 / `numHalves=2` / MATCH（行5）
  - march=`rv64gcv_zvl256b_zvfhmin` ⇒ minVLEN=256 ⇒ **16 / 16 / `numHalves=1 disjoint strip(s)`** / MATCH（行6）
  - march=`rv64gcv_zvl512b` ⇒ minVLEN=512 ⇒ 16 / 16 / numHalves=1 / MATCH（行7）
  - march=`rv64gcv_zvl1024b` ⇒ minVLEN=1024 ⇒ 16 / 16 / numHalves=1 / MATCH（行8）
- `3_product_to_objdump`（site `RVVLowerQuantContraction.cpp:1423`·`numHalves = 16 / half_lanes`）：
  - march=`rv64gcv_zvfhmin`(minVLEN=128·half_lanes=8) ⇒ numHalves=2·**vwmacc.vx=2048**·ELF `q3k.rv64gcv_zvfhmin.o (115784 bytes)`（行9）
  - march=`rv64gcv_zvl256b_zvfhmin`(minVLEN=256·half_lanes=16) ⇒ numHalves=1·**vwmacc.vx=1024**·ELF `q3k.rv64gcv_zvl256b_zvfhmin.o (58808 bytes)`（行10）
  - 编译链注：`DONE-local (clang++-20 --target=riscv64 + llvm-objdump-20; NO board)`
- `3_consistency_verdict`（VLEN128 vs VLEN256·same fixture·march= only）：`half_lanes 8 vs 16`·`ratio(vwmacc.vx) == ratio(numHalves) == 2/1`·predicted 2.0000 / observed 2.0000 (2048/1024)·**CONSISTENT**（行11）

**q4_0@ime 链（rpfIn/rpfAcc/vreg_floor(njw)/deploy-decision = θ·随 vlen_bits=c 变）·site `IMEBackendEmissionDriver.cpp:446-497`·闭式**：
`rpfIn=max(1,ceil(4*8*8/vlen)); rpfAcc=max(1,ceil(4*4*32/vlen)); floor(njw)=rpfIn*(1+njw)+njw*rpfAcc<=32; rpfIn!=1 => narrow`
- vlen_bits=128 ⇒ `rpfIn=2 rpfAcc=4 floor=None narrow fallback` predicted / observed `rpfIn=2 rpfAcc=None floor=None narrow fallback [unobserved: rpfAcc]` / `emitted leaf: narrow(no _w2 leaf)` / MATCH(2 field)（行12）
- vlen_bits=256 ⇒ `rpfIn=1 rpfAcc=2 floor=7 deploy IME-VMADOT-TILE-W2` predicted==observed / `emitted leaf: mac_kloop_w2` / MATCH(5 field)（行13）
- vlen_bits=512 ⇒ `rpfIn=1 rpfAcc=1 floor=5 deploy IME-VMADOT-TILE-W2` predicted==observed / `mac_kloop_w2` / MATCH(5 field)（行14）
- `probe_table_vs_live`(vlen_bits=512·vreg_floor(njw=2))：`floor = rpfIn*(1+njw)+njw*rpfAcc = 1*3+2*1 = 5`·site `:355 (table=7) vs :483-485 (live)`·`live != table => computed, not looked up`·**table=7 live=5**·**DIVERGE-as-predicted**（行15）
- `3_product_to_objdump`(vlen_bits=256·njw=2/mac_kloop_w2 leaf)：**PENDING·`BLOCKED-toolchain`**——`upstream clang-20 rejects xsmtvdotii (…unsupported non-standard user-level extension 'xsmtvdotii'); needs SpacemiT GCC-15.2 fork on k1. Board time = P line. NOT ATTEMPTED.`（行16）

### 2b. T1d 双实例（`experiments/active/result-tables/T1d_dual_instance.csv`·原样）
表头注（行1）：`# T1d dual-instance demonstration -- ONE schema, TWO board fact instances, core diff = 0. Structural evidence (C1); NO perf claim.`
schema 行3：`schema_sha256[:16]=52db859bb046b492`·`weft_opt_sha256[:16]=135d7bee06b6090f`

**两 fact 实例（原文关键列）**：
- `inst-rvv`：board=rvv·fact_vlen_bits=**128**·hints=`rv64gcv_zvl128b`·ime_status=**missing**·`legal_variant_set_size=1`·set=`{rvv_typed_body}`·infeasible=`{ime_keyed_body}`·guard_keys=`{"rvv":"available","spacemit_ime":"unavailable"}`·selection_chosen=**rvv_typed_body**·selection_reason=**only_feasible**·**core_diff=0**·declared_instance_hash=`bff6246419427188939ec1ab2f987d81c9325249a7b8601d0c19bdcab5c2fb2f`（行7）
- `inst-k1`：board=k1·fact_vlen_bits=**256**·hints=`rv64gcv_zvl256b`·ime_status=**available**·`legal_variant_set_size=2`·set=`{ime_keyed_body,rvv_typed_body}`·infeasible=`{}`·guard_keys=`{"rvv":"available","spacemit_ime":"available"}`·selection_chosen=**ime_keyed_body**·selection_reason=**static_order**·**core_diff=0**·declared_instance_hash=`55684ba31fc409ee427c605b825f24bdf98491839304a2b45e1025e328aa475f`（行8）

**probes（SECTION 2·原文）**：
- `P1-hash-divergence`(rvv|k1)·PASS·`ONE byte-identical body, two fact instances -> 2 distinct declared_instance_hash (rvv=bff624641942... k1=55684ba31fc4...)`（行12）
- `P4-march-channel`(rvv|k1)·PASS·`march= is a PASS OPTION, not an in-IR fact: minimum_vlen 128->256 flips chosen LMUL m2->m1 (reason=prior) while declared_instance_hash stays IDENTICAL (842c4a01e778...)`（行18）
  - [框架备注·非判断]：此行 minimum_vlen(c 输入)128→256 → chosen LMUL(θ) m2→m1·reason=prior·hash 不变。

**core diff=0 机检（SECTION 3·原文）**：
- `C1-f1-zero-branch`·PASS·`[f1-zero-branch] GREEN: zero family-name-keyed branch across 69 core files (I3 holds)`（行22）
- `C2-fact-confined-diff`·PASS·`non-fact surface byte-identical; sha256(stripped)[:16]=65320bac3b27ebb4; retained body: chars=1256 lines=25 (emitted, not transcribed)`（行23）
- `C3-same-core-binary`·PASS·`sha256(build-weft/bin/weft-opt)[:16]=135d7bee06b6090f ; pipeline=--weft-select-variants`（行24）

---

## 项3 · iq3_s 板间翻转原始记录（VLEN128→256 half_lanes 8→16）

来源：`.trellis/spec/issues/性能与测量.md:162`(ISSUE-019)、`:348`(ISSUE-102)、ISSUE-005 `:45`、门与工具.md:406。

原文照抄（ISSUE-019 `性能与测量.md:162`·同一份发射产物板间翻转）：
> 活证 = iq3_s 板间翻转：同一份发射产物 rvv 1.34 赢 / k1 0.61 输。

原文照抄（ISSUE-102 `性能与测量.md:348`·同 leaf·storm 相同·verdict 相反）：
> **同一 leaf·storm 相同·verdict 相反**：cold_X **rvv iq3_s 1.34 WIN** / iq3_xxs 0.95 near-parity（VLEN128）·**k1 iq3_s 0.61 / iq3_xxs 0.65 LOSS**（VLEN256·目标 0.8）。⟹ **storm 不是 k1 gap 的判别因**

原文照抄（ISSUE-005 `性能与测量.md:45`·8 board-cell 落账数）：
> vs 部署的 vl 专化对手 rvv 0.67 / 0.57 / 0.95 / 1.34 · k1 0.59 / 0.59 / 0.65 / 0.61（8 格输 6）… iq3_s 板间翻转（rvv 赢/k1 输·同 leaf）→ K 线手调攻坚候选。
> （逐格取自 `experiments/active/g8-stage3-attack/P2-grid4-raw/*_measure.log` 的 `ratio_cold_X`·2-seed）

**翻转前后（同 leaf·c 板输入变·θ half_lanes 变）**：
- iq3_s@rvv VLEN128(half_lanes=8·VLMAX=8 满用) = **1.34 WIN**
- iq3_s@k1 VLEN128-fixture(half_lanes=8·跑在 VLEN256 硅=半宽) = **0.61 LOSS**
- iq3_s@k1 宽化 VLEN256(half_lanes=16) = **1.20 WIN**（部署后 1.21·门与工具.md:406/:393）

半宽根因（ISSUE-102 `性能与测量.md:349`·原文）：
> **真 k1 bottleneck = VLEN256 半宽欠用**：iq3 group 处理固定 **AVL=8**（8 元素 grid group），编译器选 `mf2`/`m1`（VLEN128 下 VLMAX=8 满用·VLEN256 下 VLMAX=16 → **每条向量 op vl=8/16 半宽**）

---

## 项4 · K1 家族反证 FINDING 关键行

> 说明：supervisor 具名 `:59/:106/:114-116`。仓内匹配的 K-quant decode 逐板 board-difference 文件 = `experiments/active/g8-stage3-attack/A2-batch5-kquant-decode-M1.md`（探针在 :59·per-board 判读表在 :104-118）+ 配对反证 `A2-batch4-gemm-decode-M1.md`。以下原文照抄，含 opp 身份探针与逐格数。

### 4a. n_sb 判别键反证（家族级·ISSUE-101·`性能与测量.md:334-335`）原文：
> **q4_K@k1 n_sb=8 → PASS 1.535**（batch4）vs **q5_K@k1 n_sb=8 → named-X LOSS**（batch5）——**同板·同编译器·n_sb 均=8·verdict 相反** ⟹ **sub-block 数不是判别键**。

### 4b. q4_K@k1 vs @rvv 板+对手强度分裂（`A2-batch4-gemm-decode-M1.md:21`·:91·:95·:108）原文：
> | **q4_K** | **0.066× named-X**(GCC-DEATH) | **0.361× named-X** | **1.535× PASS** | byte-exact 0/512(vs ggml) | ★**board+opp-strength 分裂**：k1 赢**弱 opp**(0.80ms)·rvv 输**强 opp**(0.22ms native-vec)·super-block fold@M=1 不摊销·rvv-gcc = **[CASE-KQUANT-GCC-CODEGEN] gcc-death**(vsetvl 1387 vs clang 57) |（:21）

> **★board+opp-strength 分裂（honest 关键）**：verdict 由 **opp 强度** + **board** 定：rvv q4_K block-dot(gcc-15 stock·0.22ms) 比 k1(clang-18 stock·0.80ms) **快 3.6×** … → k1 赢弱 opp·rvv 输强 opp。**非我方 kernel 双板异**(k1 leaf 0.53ms ≈ rvv-clang leaf 0.60ms·近同)·**是 opp 强度异**。（:95）

> | k1·clang-18(what-if) | vsetvl=57 clean | 0.522–0.526 | 0.804–0.806 | 1.53/1.53/1.54/1.54 | **1.535** | **PASS** | 赢·但 opp=k1 q4_K block-dot **偏弱**(0.80ms)·非 hand-brick·hl8 leaf VLEN256 half-util(未占满宽·真部署 hl16 更快)·front-door declines(what-if) |（:91）

### 4c. A2-batch5 opp 身份探针（`:59`）+ 逐板 decode 判读表（`:104-118`）原文：
探针（`:59`）：
> **探针（opp 身份·符号级）**：rvv gcc-15 stock `ggml_vec_dot_qX_K_q8_K` @ q2_K=0x92316 / q3_K=0x930a6 / q5_K=0x93146 / q6_K=0x935de；k1 stock @ q2_K=0xa034c / q3_K=0xa15da / q5_K=0xa1ed8 / q6_K=0xa21d2。**均 stock block-dot·native-RVV inline·非 hand-brick**。

逐板判读表（`:107-114`·rvv vs k1 同格 cold）：
> | gemm_tile | q2_K | decode | **named-X** | **0.0685**(gcc GCC-DEATH)/0.3635(clang18) | PASS | **0.9585**(clang18·what-if·near-parity) |
> | gemm_tile | q3_K | decode | **named-X** | **0.0830**(gcc)/0.2189(clang) | **named-X** | **0.4252**(clang18·what-if) |
> | gemm_tile | q5_K | decode | **named-X** | **0.1273**(gcc)/0.5040(clang) | **named-X** | **0.6806**(clang18·what-if)·[GAP-Q5K-QH-REGCLIFF] per-board |
> | gemm_tile | q6_K | decode | **named-X** | **0.0535**(gcc)/0.2595(clang) | **named-X** | **0.3771**(clang18·what-if) |

gcc_death_flag（`:114`）原文：`gcc_death_flag=**TRUE 全 4 格@rvv-gcc**（vsetvl 922–2952·[CASE-KQUANT-GCC-CODEGEN]）`
provenance（`:115`）：`A2-batch5-kquant-decode-M1-raw/logs/{rvv,k1}_run.log` + `{rvv,k1}_build_seal.txt`

---

## 项5 · C920 9/9 关键行 —— 【部分采到 + 缺什么见下】

> 采到：C920 = T-Head XuanTie xtheadvector **RVV0.7 板**（板间行为差异·ISA-generation 轴）的现成证据 = capability-model 层 lit（编译器证据·非板测）。
> 采不到：与「C920」直接绑定的「**9/9**」板测/通过数——仓内 `9/9` 命中全是 fp16 `h2f 9/9 textbook IEEE`（各 driver）与 `bench --self-test 9/9`（门与工具.md:86）+ T4b/T-CENSUS 的 9/9，**均与 C920 无关**（详见 cant_collect）。

### 5a. C920 板间行为差异（同 body → 不同合法性·c 板输入=rvv_version 驱动 θ）
来源：`test/Conversion/EmitC/rvv-capability-profile-divergence-rvv07-vs-rvv10-policy.mlir`。原文照抄：
> * Profile A (--march=rv64gcv, RVV1.0): the agnostic-policy body is ACCEPTED and lowered to __riscv_vadd_vv_i32m1.（:10-11）
> * Profile B (--march=rv64gc_xtheadvector, RVV0.7 / C920): the IDENTICAL body is REJECTED fail-closed -- RVV0.7 LACKS the ratified ta/ma policy, so the version capability gates the agnostic-policy body out (no EmitC emitted).（:12-14）

FileCheck STAMP（`:92-99`·同一 bare @rvv provider·march 派生的 c 事实差）原文：
> STAMP-RVV10: `rvv_version = "1.0"` · `supported_lmul = "mf8,mf4,mf2,m1,m2,m4,m8"` · `supported_sew = "8,16,32,64"`
> STAMP-RVV07: `rvv_version = "0.7"` · `supported_lmul = "m1,m2,m4,m8"` · `supported_sew = "8,16,32,64"`

硬件依据（`:24-25`）原文：
> the C920 runs 0.7.1 `th.v*` under rv64gc_xtheadvector and SIGILLs on a 1.0 rv64gcv binary (proven on the real silicon).

### 5b. C920 = fractional-LMUL 有无（c 输入·源注释）
来源：`include/Weft/Plugin/RVV/RVVCapabilityProfile.h`。原文照抄：
> :32 `HARDWARE FACT (proven on the C920): rv64gc_xtheadvector runs 0.7.1 th.v*;`
> :133 `* RVV0.7.1 (XuanTie xtheadvector on the C920) has NO fractional LMUL at all`
配套源点：`RVVCapabilityProfile.cpp:264`(RVV0.7.1 NO fractional LMUL)·`RVVRepackStripWidthMaterialization.cpp:37,103`·`Passes.td:386`(RVV0.7 ALSO stamps integer_core_lmul)·`RVVToEmitC.cpp:208,2307`。

### 5c. C920 相关 lit 资产（现成·数量）
- C++ 测：`test/CMakeLists.txt:228` `weft-rvv-capability-profile-divergence-test` + lit `test/Plugin/rvv-capability-profile-divergence.test:1`
- EmitC divergence .mlir（4 支）：`rvv-capability-profile-divergence-f64-coverage.mlir` / `-live-probed-march.mlir` / `-one-kernel-two-profiles.mlir` / `-rvv07-vs-rvv10-policy.mlir`

---

## 项6 · 累加器 LMUL 两次证伪原始记录（[GAP-P1]·候选 f 形状/板/败在什么数）

### 6a. 候选 f 形状 + 现行 θ（源码原文·`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`）
`selectRepackAccumulatorLMUL`（`:1287-1304`）原文关键行：
> `if (isRVV0p7) return {/*useM1=*/true, "correctness-rvv0p7"};`
> `constexpr std::int64_t kVectorRegisterBudget = 32;`
> `const std::int64_t m1ChainRegisterFootprint = getRVVLMULRegisterFootprint("m2") + getRVVLMULRegisterFootprint("m4");`  // 候选 f 形状 = i16m2 product + i32m4 accumulator
> `const bool m1Constructible = capabilityHalfLanes != 0 && m1ChainRegisterFootprint <= kVectorRegisterBudget;`
> `// [GAP-P1]: default mf2 -- never blind-widest; only a board measurement flips.`
> `return {/*useM1=*/false, "capability-default-mf2"};`

候选身份 + 空表（`:1271-1279`）原文：
> `// STAGE THREE populates this per-format (x board) board-measured m1-vs-mf2 crossover. EMPTY today: nullopt for every format => the mf2 default holds (byte-exact). q4_0 is the documented m1-faster CANDIDATE and q8_0 the mf2-faster candidate, but NEITHER is asserted here without a board number ([GAP-P1]: no projection; the board key is threaded in at STAGE THREE).`
> `inline std::optional<bool> lookupRepackMeasuredM1Faster(...) { return std::nullopt; }`

reroll-禁令（`:1255-1257`）原文：
> `// We do NOT blind-select the wider m1 (that is exactly what the gate4 widest-legal selector does for a DIFFERENT kernel path; reusing it here verbatim would re-commit [GAP-P1]).`

[框架备注·非判断]：源注释显式称冷启动 mf2 为 [GAP-P1]（`ISSUE-117` `性能与测量.md:424`：「冷启动 mf2 = **[GAP-P1] IRON RULE 刻意钉**」；`:424` 「widest-legal 公式 `core_lmul=f(VLEN,块宽)` 被 [GAP-P1] 明禁（widen-to-m1 falsified 2×·byte-exact hazard）」）。

### 6b. 证伪 #1 + #2（q4_K repack GEMM full-unroll 寄存器压力「opening」·板=rvv/VLEN128）
来源：`experiments/archive/l1-kquant/l1-reroll-q4k-repack-gemm/reroll_findings.md`。
Board（文首）：`rvv / VLEN128, core 8, 2.6 GHz, governor=performance (measured 2026-07-08)`

Bottom-line 原文（两 lever 均证伪）：
> Both cheap levers on the q4_K repack GEMM's full-unroll register-pressure opening are now falsified on silicon: **byte-exact reorder (#1) = NULL (+0.65%, noise floor)**; **naive structural RE-ROLL (#2) = -11% throughput, vsetvli +43%, spill +40%, parity 0.962x -> 0.860x**.

证伪 #2 逐指令表（objdump·原文）：
> | PRE  -O2 (measured binary) | vsetvli **53** | spill 84 | reload 88 | vwmacc 2240 | textB 23486 |
> | POST -O2 (measured binary) | vsetvli **76** | spill 118 | reload 127 | vwmacc 320 | textB 10220 |
A/B（败在什么数·原文）：
> nr=64, N=12: ours_pre 4.2398 -> ours_post 3.7878 GMAC/s => **0.893 (~-10.7% REGRESSION)**
> nr=16, N=10: ours_pre 4.2460 -> ours_post 3.7853 GMAC/s => **0.892 (~-10.8%)** SHAPE-ROBUST
> PRE parity = 0.962x → POST parity = 0.860x（~10.6% FURTHER from parity）
byte-exact 门（`:0. Byte-exact identity gate`）：int/norm regime `IDENTICAL, 0 mismatch`（RE-ROLL 保 op order）。

### 6c. 相邻 m1-widen 板数（k1/VLEN256·q8_0 mbf sweep）
来源：`experiments/archive/perf-historical/ondevice-q8_0-mbf/results.csv`。原文照抄（尾注）：
> `mbf2/mbf1 = 0.975 (mbf2 is 2.5% faster). mbf2 vs autovec still ~0.77-0.91x (< 1.0). DECISION: NOT FIXED.`
> `mechanism: … q8_0 34-byte block stride (2B scale gap) blocks a 64-wide load, so mbf does NOT fill the register. Register-fill fix needs integer_core_lmul=m1 (e8m1@VLEN256=32=one block).`
CSV 数据行（board=k1·SpacemiT-X60·vlen 256）：`core_mbf1 … kernel_ns 6900.4 … kernel_vs_ref 0.75-0.88` / `core_mbf2 … 6727.0 … 0.77-0.91`。

### 6d. RVV1.0 侧 mf2 单一合法（same-board 候选集·相邻证据）
`experiments/archive/g7/g7-l1-kernelsym-fullfill/rvv-batch/evidence.md:42`原文：
> **m1 whole-LMUL core 是 RVV0.7.1(`xtheadvector`) 专属分支**（passes.td:386-394）·**非本板(rv64gcv/RVV1.0) 合法 same-board 候选**（跨-ISA·板不跑 xtheadvector 码）
`SEL3-measurement-memory-design.md:24`原文：
> mf2 vs m1 | **{mf2} 单一合法**（双板·m1=RVV0.7 跨-ISA 非 same-board·`[repack-winA-always-mf2]`）

---

## 项7 · ISSUE-120 现状原文（iq2_xs/iq2_s VLEN256 byte-broken）

来源：`.trellis/spec/issues/发射器与架构.md:216-222`(ISSUE-120)。原文照抄：
> **状态**：待修（correctness 缺口·**pre-existing 非新引入**·iq2_xs 一直 rvv-only）。
> **实质**（task `07-19-deploy-iq2xs-fused` 板测暴露）：deployed iq2_xs vec_dot serial emit（`emitIQ2XSSuperBlockGridBody`）在 **VLEN256（k1）byte-broken**——`ours_vs_int_oracle=false`·mism=512（ggml+oracle 一致·仅我方错）。**pre-existing 原 `iq2_xs.kernel.c` 同样 broken**（非部署 agent 的改动）。根因 = pair-batched `vget_i8m4_i8m1` 结构 **VLEN128-form**（LMUL8 32-lane 假设·VLEN256 崩）。iq2_s 同 pair-batched 结构·**likely 同病**。
> **影响**：iq2_xs（+likely iq2_s）vec_dot **任何 k1/VLEN256 claim 前须先修**（per-VLEN 专化 或 unbatched VLEN-universal 路）。当前 master iq2_xs@k1 若有 PASS/verdict 须复核（可能 byte-broken 未察）。
> **保守默认**：iq2_xs/iq2_s vec_dot 维持 rvv-only（k1 不 claim）·待 per-VLEN 修。
> **出处**：task `07-19-deploy-iq2xs-fused`（VLEN256 byte-exact 门 mism=512·pre-existing 复核）。

配套（ISSUE-112 §簇B `性能与测量.md:406`·同缺口登记）原文：
> **+ 新 correctness 缺口**：iq2_xs/iq2_s pair-batched serial emit **VLEN256 byte-broken**（登记 [[ISSUE-120]]·需 per-VLEN 专化或 unbatched VLEN-universal 路方可 k1 claim）

[框架备注·非判断]：ISSUE-120 = 「同一 serial emit 结构在 VLEN256(c 板输入) 下 byte-broken」的板间行为差异事实·pre-existing。

---

## 采集计数（本包 7 项）

- **采到 6 项**：项1(宽化逐格+objdump) · 项2(T-P 16 行全 + T1d 双实例全) · 项3(iq3_s 板间翻转) · 项4(K1 家族反证:n_sb 反证 + A2-batch4/5 逐板) · 项6(累加器 LMUL 两证伪) · 项7(ISSUE-120)。
- **部分采到 1 项**：项5(C920) —— C920 板间行为差异（RVV0.7-vs-RVV1.0 divergence·supported_lmul m1,m2,m4,m8 vs mf8..m8·fractional-LMUL 有无·fail-closed）**采到**；但与「C920」绑定的具体「**9/9**」数**采不到**（见 cant_collect）。
