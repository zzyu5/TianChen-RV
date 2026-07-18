# Research: ISSUE-112 §簇A objdump-recon 墙型图（公式墙 vs 脾气墙）

- **Query**: 反汇编簇A 6 board-cell 对手部署核 → 按「公式墙 vs 脾气墙」判据逐格裁墙型（非「赢没赢」）
- **Scope**: internal / board-attack（read-only 反汇编·零代码改·不 commit）
- **Date**: 2026-07-19
- **Boards**: `ssh rvv`（VLEN128·isa 有 zvfh/zve64d）· `ssh k1`（VLEN256·isa 含 `ime`）· 两板 objdump 均可用
- **对手二进制两路**：① **clang18-symmetric**（`/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin/libggml-cpu.so`·符号 `_vl128`/`_vl256`·**无 `.isra`** = clang-18 编）② **deploy native default**（`/home/ubuntu/tcrv-llamacpp/build/bin/...`·符号 `_vl128.isra.0` = **gcc** 编·IPA-SRA 后缀）。**kernel-axis 数用 clang18-symmetric 路**（我方 DUT 亦 clang-18·[CASE-COMPILER-ASYMMETRY] 守）。

---

## 一、逐格墙型表（一行一 board-cell）

| # | 格 | board | cold | opp 符号 | objdump 指令直方（clang18-sym） | 对手结构 | 编译器对称 | **墙型裁定** |
|---|---|---|---|---|---|---|---|---|
| 1 | vec_dot iq2_xxs | rvv | 0.697 | `ggml_vec_dot_iq2_xxs_q8_K_vl128` | TOT=132 VEC=42 vset=10 **gather=4**(vluxei16) reduce=2(vwredsum) mac=4 | grid-codebook **GATHER** + tiny widening-reduce | ✅ clang132 ≈ gcc133（C-intrinsic·非 asm） | **公式墙候选·必攻** |
| 2 | vec_dot iq2_xs | rvv | 0.529 | `ggml_vec_dot_iq2_xs_q8_K_vl128` | TOT=251 VEC=122 vset=49 **gather=8**(vluxei16) **reduce=16**(vwredsum) mac=8 | grid GATHER(8) + **16× tiny per-sub-block vwredsum** 主导 | ✅ clang251 ≈ gcc229（C-intrinsic） | **公式墙候选·必攻** |
| 3 | vec_dot tq1_0 | rvv | 0.207 | `ggml_vec_dot_tq1_0_q8_K_vl128` | TOT=131 VEC=67 vset=24 **gather=0** reduce=1 mac=13(vwmulu.vx=5) | ternary **无 gather**·×powers-of-3 解包(vwmulu)+MAC+末 reduce | ✅ clang131 ≈ gcc132（C-intrinsic） | **公式墙候选·必攻（gap 最大）** |
| 4 | vec_dot tq1_0 | k1 | 0.606 | `ggml_vec_dot_tq1_0_q8_K_vl256` | TOT=172 VEC=126 vset=33 gather=0 reduce=1 mac=31 | ternary·**主循环 e8,m1(32 lane 满宽)+e16,m2**·无 gather | ✅（C-intrinsic·`_vl256` 真宽化专化） | **公式墙候选·必攻·VLEN256 半宽 co-factor 证伪** |
| 5 | gemm q4_0@ime | k1 | 0.196 | `spacemit_kernels::ime1::gemm_kernel_i8i4` | TOT≈663 **手写 asm** vmadot=32 **vfmul.vf=40** vrgather=8 vfwcvt=16 vse32=40 | 单体手排 asm tile(4-acc·深展开·unpack/MAC/scale-epilogue/store 交织) | ❌ vendor **纯手写 asm** vs 我方 clang-C-driver+asm-leaf | **脾气墙(编译器行为墙)·登记边界·e2e 已 cover** |
| 6 | gemm q4_K@ime | k1 | 0.049 | 同 `gemm_kernel_i8i4`（q4_K→q4_1x16 offline requant 喂入） | 同上手写 asm·外加 q4_K super-block 复杂度 | vendor 把 super-block 折叠**移出热核**(offline requant)·我方两级 dmin/bsums fold **在核内** | ❌ 同 #5·极端不对称 | **脾气墙(编译器行为墙)+格式复杂度放大·登记边界·e2e 已 cover** |

---

## 二、关键裁定依据（据 objdump·非直觉）

### #1/#2 iq2 grid vec_dot@rvv —— **NOT 同 iq3-dequant gather 墙**（PRD 核心问答）
- **iq3-dequant gather 墙（ISSUE-107·天花板 0.36）的成因**：对手 `dequantize_row_iq3_xxs` **HW_GATHER=0**（37 条标量 load 解码·避开 gather），我方 owned 真向量**被迫 gather** ⟹ gather 是**我方单侧惩罚**（HW-gather 吞吐天花板）。
- **iq2 vec_dot 的结构相反**：对手**自己就用 gather**（`vluxei16.v`=4/8·索引 256-entry iq2 grid 码本）⟹ **gather 是双侧对称量·非差分惩罚**。vec_dot 有 q8_K reduction 结构（`vwredsum.vs`），与 dequant streaming 不同。
- **⟹ 结论：vec_dot iq2 ≠ iq3-dequant gather 墙**。对手优势不在「我方多付的 gather」，而在**可读的 C-intrinsic 结构本身**（grid-decode + tiny-reduction），且 clang≈gcc 证**非 inline-asm**（对手无「读不到的手排量」）。**这落在公式墙定义侧**（对手赢在可读的量）。

### #2 iq2_xs 的 16× vwredsum = 已知可键控 tiny-reduction 族
- iq2_xs 主导量 = **16 条 `vwredsum.vs`**（每 sub-block 一条 serial 归约链），与 **ISSUE-020**（iq1_m sum2 符号和向量化·已就绪）/ **ISSUE-109**（K-quant vwredsum floor）同族。这是**闭式可键控**的归约批处理杠杆·非 opaque。

### #3/#4 tq1_0 ternary —— 无 gather·纯可读算术·非 grid 非 codebook
- 对手用 **`vwmulu.vx`（×{1,3,9,27,81…} powers-of-3）+ vsrl + vadd** 解包三值·再 MAC·末 1 reduce。**gather=0**。整核可读 C-intrinsic。
- **rvv gap 最大（0.207≈慢 4.8×）**：对手 131 insn 67 向量 op 满向量化·此 gap 强烈暗示**我方现无 owned 向量化 ternary leaf**（弱/标量路 vs 对手满向量）⟹ 建 owned leaf 上升空间最大。

### #4 tq1_0@k1 AVL 半宽问题 —— **证伪**（PRD 明问）
- 对手 `_vl256` 主循环 `vsetvli zero,a6,e8,m1`（32 lane 满宽）+ `e16,m2`（双组），仅解包子段用 `e8,mf2`（半寄存器·语义所需·非欠用）。
- **对手已吃满 256b·非 vl128-on-vl256 半宽核** ⟹ **tq1_0@k1 不属 F3 VLEN256-widening 族·无「免费宽化」杠杆**。杠杆 = 与 rvv 同的 owned ternary vec_dot leaf（非宽度杠杆）。

### #5/#6 IME 两格 —— vendor 手写 asm·「scale-fold 融入 vmadot」候选杠杆被 objdump 证伪
- **对手 = 手写汇编**：`gemm_kernel_i8i4` 是单体 ~663-insn 汇编（局部 asm label `RESULT_SAVE5/RESULT_SAVE7/END6`·非编译器生成）·k1 objdump 原生解码 `vmadot`（`0xe2..382b` 家族·32 条·4-acc tile）。
- **我方 = clang-C-driver + inline-asm vmadot leaf + 独立 C scale-fold loop**（`experiments/active/g8-stage3-attack/A2-batch7-ime-kernel-sym-raw/ime_q40_kernelsym.cpp:51` `weft_ime_vmadot_mac_kloop` 是 `__asm__ volatile` vmadot；scale 走另一 C 折叠环）。
- **★候选杠杆「scale-fold epilogue 融入 vmadot MAC」= 红鲱鱼**：vendor **自己也用独立 `vfmul.vf` epilogue（40 条）**做 scale·**并未**把 scale 折进 vmadot——**物理上不可能**（vmadot 是整数指令·per-block f32/f16 scale 无法进整数 MAC）。⟹ 真 gap driver = vendor 单体手排 tile（深展开·寄存器驻留 4-row acc·unpack/MAC/epilogue/store 交织流水）vs 我方 tiny-asm-leaf-called-from-C + 独立 fold。
- **⟹ 脾气墙（编译器行为墙）定义命中**：对手赢在**手写 asm 调度/寄存器分配**（我方 C-intrinsic/emitc 读不到的量）。[CASE-COMPILER-ASYMMETRY] 极端案（vendor 纯 asm vs 我方 clang-C）。
- **#6 q4_K 更锋（20× vs 5.1×）**：vendor 把 q4_K super-block 折叠**移出热核**（offline `repack_q4_k_to_q4_1_16_bl` requant 成 q4_1x16）·我方两级 dmin/bsums fold **在核内** ⟹ [PAT-1] format-keyed 边界最锋利·C3′ format-keyed 负结果素材。

---

## 三、汇总（供 main 排下一波攻坚）

### 公式墙（必攻·按扇出/gap 排序）—— 4 board-cell
| 优先 | 格 | live-lever（可读的量） | 入库后扇几格 |
|---|---|---|---|
| **P1** | tq1_0 vec_dot @rvv(0.207)+@k1(0.606) | **owned 向量化 ternary vec_dot leaf** = `vwmulu.vx` powers-of-3 解包 + MAC + reduce（无 gather·纯算术·gap 最大=upside 最大） | tq1_0 ×2 板 + tq2_0 ×2 板（ternary 族）≈ **2-4 格** |
| **P2** | iq2 grid vec_dot @rvv：iq2_xxs(0.697)+iq2_xs(0.529) | **owned iq2 grid vec_dot leaf** = `vluxei16` grid-decode + tiny `vwredsum` 归约（iq2_xs 16× 归约挂 ISSUE-020/109 归约批处理族·可复用） | iq2_xxs/iq2_xs(/iq2_s) @rvv ≈ **2-3 格** + 参照 iq2@k1（3 格 F3 结构推断） |

- **共性**：两族对手皆 **C-intrinsic（clang≈gcc·非 inline-asm）**·结构可读可键控 ⟹ 落公式墙侧·**三步只走了第一步（objdump）**·步 2-3（建 owned leaf → 板测证伪或达 parity）= 后续攻坚 task（本 recon 不建 leaf·不下承诺）。

### 脾气墙 / 编译器行为墙（登记边界·转打第一块）—— 2 board-cell
| 格 | 三步登记依据（objdump 看到读不到的量） | 裁定 |
|---|---|---|
| gemm q4_0@ime(0.196) | 对手 = 663-insn **手写 asm** 单体 tile·候选 epilogue-fusion 杠杆被证伪（vendor 亦独立 vfmul.vf epilogue·scale 无法折进整数 vmadot） | **kernel-sym 负结果具名冻结·编译器行为墙**·**明示 e2e 已 cover（tie-stock 1.0088×·非 e2e 依据）**·🔴 禁 inline-asm 手排绕过 |
| gemm q4_K@ime(0.049) | 同手写 asm 墙 + vendor offline requant 移复杂度出核·[PAT-1] format-keyed 放大 20× | 同上·**e2e 已 cover（黄 0.909×）**·C3′ format-keyed 边界负结果素材·禁手排绕过 |

- **两赛道禁互推**（[CASE-COMPILER-ASYMMETRY]）：IME 两格 kernel-sym 负结果**不得贬 e2e**（e2e 走部署路已 cover）。
- **候选杠杆订正**：ISSUE-112 §簇A 为 IME 两格拟的「scale-fold epilogue 融入 vmadot MAC」杠杆经 objdump **证伪**（vendor 未做·ISA 不可能）⟹ 该 lever 不成立·honest-null 更准（编译器行为墙·非「未试公式杠杆」）。

---

## 四、Caveats / 待 main 注意
1. **vec_dot M=1 micro-artifact 存疑（regime 归属·近簇B）**：本 4 公式墙格皆 vec_dot（M=1·无行摊销）。census a-4 曾判 iq3 vec_dot@rvv 为「system-covered / M=1 micro artifact」（因 iq3 gemm@rvv PASS）。但 **iq2 gemm@rvv decode = 具名-X 0.60·tq1_0 gemm = scalar-ref cheap-tier**（recon line 203/136）——**非干净 system-cover**·故 iq2/tq1_0 vec_dot 是**真欠攻面**（非 micro artifact 白豁免）。此 regime-归属属判据级·**agent 不自决**（PRD：簇B 不碰）·登记供裁。
2. **无 owned vec_dot leaf**：iq2/tq1_0 vec_dot cold（0.697/0.529/0.207/0.606）来自 T3 master matmul 账·对手 tier=手调·**无 OURS leaf 记录**（对手单侧 objdump）。攻坚第一步 = 建 owned leaf·然后板测才知落 parity 还是撞归约/M=1 floor。
3. **cold 数字禁外推**：iq2_s@rvv、tq2_0 vec_dot 等未逐格 objdump·墙型「同族推断」须板测坐实。
4. **禁越权**：本任务 read-only·未改任何代码·未 commit·未下攻坚承诺（PRD §性质）。攻坚立项 = 后续 task。

---

## 附录 A：原始 objdump 指令直方（board 实测·2026-07-19）

### rvv（clang18-symmetric 路 `build-clang18-rv64gcv`·无 .isra）
```
iq2_xxs _vl128 : TOTAL=132 VEC=42 vset=10 gather=4 reduce=2 mac=4 vload=5 vstore=10 vmv/merge=5 calls=0
  top: vsetvli=8 vsll.vi=4 vluxei16.v=4 vle8.v=4 vwredsum.vs=2 vwmul.vv=2 vwcvtu.x.x.v=2 vsrl.vv=2 vnsrl.wi=2 vmul.vv=2 vand.vx=2
iq2_xs  _vl128 : TOTAL=251 VEC=122 vset=49 gather=8 reduce=16 mac=8 vload=8 vstore=49 vmv/merge=17 calls=0
  top: vsetvli=40 vwredsum.vs=16 vmv.x.s=16 vsll.vi=8 vluxei16.v=8 vwmul.vv=4 vsrl.vi=4 vmul.vv=4 vle8.v=4 vle16.v=4 vand.vx=4
tq1_0   _vl128 : TOTAL=131 VEC=67 vset=24 gather=0 reduce=1 mac=13 vload=8 vstore=24 vmv/merge=3 calls=0
  top: vsetvli=21 vle8.v=8 vwmulu.vx=5 vwcvt.x.x.v=5 vsrl.vi=5 vadd.vi=5 vmul.vv=4 vadd.vv=3 vmul.vx=2 vmacc.vv=2 vwredsum.vs=1
```
### rvv（deploy native default 路 `tcrv-llamacpp/build`·gcc `.isra.0`·对拍对称性）
```
iq2_xxs _vl128.isra.0 : TOTAL=133 VEC=47 gather=4 reduce=2 mac=4   (≈ clang·差 1 insn)
iq2_xs  _vl128.isra.0 : TOTAL=229 VEC=96  gather=8 reduce=16 mac=8 (≈ clang·差 22 insn·同 gather/reduce)
tq1_0   _vl128.isra.0 : TOTAL=132 VEC=63 gather=0 reduce=1 mac=13  (≈ clang·差 1 insn)
```
> **对称性判据**：clang vs gcc 指令数近同（132/133·251/229·131/132）·gather/reduce/mac 计数一致 ⟹ 对手为 **C-intrinsic**（两编译器独立调度得近同码）·**非 inline-asm 手排**（若手排则 asm 逐字·两编译器字节全同）。

### k1（VLEN256·ime build `build-ime`）
```
tq1_0 vec_dot _vl256 : TOTAL=172 VEC=126 vset=33 gather=0 reduce=1 mac=31 float=4
  vset detail: 主循环 vsetvli e8,m1(AVL=a6≈32满宽) + e16,m2 ; 解包子段 e8,mf2(半寄存器·语义所需)
  ⟹ 满宽·非半宽欠用 ⟹ 不属 F3 VLEN256-widening 族
vendor gemm_kernel_i8i4 (q4_0/q4_K@ime opp) : span d29e6..d3300 ≈663 insn 手写 asm
  vmadot=32  vsetvli=52  vse32.v=40  vfmul.vf=40(scale epilogue·独立·非折进 vmadot)
  vle8.v=26  vle16.v=24  vfwcvt.f.f.v=16  vand.vi=16  vfcvt.f.x.v=10  vrgather.vv=8(nibble unpack)  flw=10(scale load)
  asm labels: RESULT_SAVE5 / RESULT_SAVE7 / END6 (手写汇编·非编译器符号)
```

## 附录 B：对手/我方身份与源锚
- **rvv vec_dot 对手**：`ggml_vec_dot_{iq2_xxs,iq2_xs,tq1_0}_q8_K_vl128`（clang18-sym）/ `..._vl128.isra.0`（gcc deploy）·ggml 上游 `_vl128` 手写 intrinsic 专化（tier=手调·非便宜 `_generic` 档）。
- **k1 IME 对手**：`spacemit_kernels::ime1::gemm_kernel_i8i4`（vendor 手写 asm·真硅 vmadot `0xe210312b`）·`build-ime/bin/libggml-cpu.so`。q4_K@ime 喂入路 = `repack_q4_k_to_q4_1_16_bl`→q4_1x16 offline requant。
- **我方 IME kernel-sym OURS**：`ime_q40_kernelsym.cpp` / `ime_q4k_kernelsym.cpp`（`experiments/active/g8-stage3-attack/A2-batch7-ime-kernel-sym-raw/`）·inline-asm vmadot leaf + C fold。int32 core byte-exact（0xe210312b seal 血统）。
- **cold/verdict 权威**：`experiments/master/T3_master_rebuild.csv` + `.trellis/scripts/recon_master_rebuild.py`（TIER 108-115 / IME_KERNELSYM 252-254）。
- **杠杆现状**：`experiments/active/result-tables/named-X-leverage-census.md` §三 #2-5,10-11 · ISSUE-112（`.trellis/spec/issues/性能与测量.md:379`）· canon [K-4] 墙型词汇表（`.trellis/spec/canon/覆盖状态机与选择归因.md:22`）。
