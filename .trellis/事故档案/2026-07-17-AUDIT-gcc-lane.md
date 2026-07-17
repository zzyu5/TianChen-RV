# AUDIT · gcc-lane：「零 gcc 字样」是否等于「零 gcc 数据」

> **性质**：纯只读审计 · **不裁去向**（改数据/重测 = 硬冻结 + 必问）· 只出事实 + 影响面。
> **工具**：`tools/visibility/audit_gcc_lane.py`（机算 · 全部计数 tool-emit · 散文零裸数）
> **复跑**：`python3 tools/visibility/audit_gcc_lane.py`
> **只读自证**：全程 `git status --porcelain experiments/active/result-tables/ .trellis/scripts/` = 空。
> 变异测试全在 scratchpad 副本上跑，真仓库 CSV / recon **零改动**。

---

## 〇 · 用户问：PR-17 裁了双板都用 clang，为什么还有 gcc？

**答**：因为 PR-17 的验收门 **只验了输出文本里的「gcc」字样，没验数据的来源**。
本审计用变异测试证明：**该门是恒真的**——即便输入 100% 自认 gcc，门仍读 0，且账面输出**逐字节不变**。
在门读 0 的同时，**28 个格的 verdict 仍由 gcc-15.2 车道的测量数据决定**。

---

## 一 · 核心：标签 vs 数据（工具 emit · 禁手写）

板 CSV 36 列 schema 里两个关键列（`sed -n` 确认自 T3_A 表头行）：

| 列 | 名字 | 角色 |
|---|---|---|
| f[33] | `compiler_axis(clang18-sym=MAIN-verdict\|gcc=deploy-footnote)` | **标签**：本行 verdict 的编译器轴 |
| f[35] | `hardgate_0p8(in-denom\|test-only-not-in-denom\|sealed)` | **数据**：verdict token + 倍数 + 来源域 |

### 1.1 全表交叉（162 canonical 行 = rvv 82 + k1 80）

```
board  LABEL域(f33)            DATA域   行数
k1     CLANG-MAIN|gcc-脚注      clang       1
k1     CLANG-ONLY             clang      79
rvv    CLANG-MAIN|gcc-脚注      clang       9
rvv    CLANG-ONLY             clang      32
rvv    CLANG-ONLY             gcc        15   ← ★标签说 clang·数据是 gcc
rvv    GCC-MAIN(诚实)           gcc        23
rvv    无编译器标注                 无标注       3
```

**k1 板 gcc 数据 = 0**（与 PR-11 per-lane「k1 = clang-18」一致）。**全部 gcc 数据在 rvv 板**。

> **★自纠一处（前科防范）**：初版检测器把 f[24] `board_fp` 也当来源指纹，**误报 k1 五格**。
> 实因两板共用同一 provenance 串 `T9-668f1d45(rvv-gcc-deploy)`，那是**跨板引用**、非 k1 自身来源。
> k1 行 f[33] = `clang18-sym=MAIN=DEPLOYMENT-VALID(k1 ships clang-18)`、f[29] 是 k1 自己的数。
> 已改为**只认描述本行自身测量的三处**（f[33] MAIN 声明 / f[35] verdict TOKEN 区 / f[29]），
> 并把 `*成色` 之后的注记区排除（跨板脚注住那里）。修正后 k1 = 0 gcc。

---

## 二 · 污染格清单（自己独立扫 · 非只查主会话给的 6 个）

**判据**：DATA 域 = gcc **∧** 未被 recon `CLANG_WORLD` 以 clang 重测数覆盖 **∧** 在分母内。
**结果 = 28 格**（主会话给的 6 个只是其中的 `具名-X` 子集；**另有 22 格是 PASS**）。

### 2.1 按四档归口（= recon 头条口径 · rvv 板）

| 四档 | 状态 | 格数 | 格 |
|---|---|---|---|
| 标量类 | **PASS** | **14** | dequant: iq1_s, iq2_xxs, iq3_xxs, iq4_nl, mxfp4, q2_K, q3_K, q4_1, q6_K, q8_0, tq1_0 (11) + product_reduce: codebook_n3, offset_binary_n3, q4_0_nibble (3) |
| 标量类 | **具名-X** | **6** | dequant: iq1_m, iq3_s, iq4_xs, q4_K, q5_K, tq2_0 |
| 通用向量 | **PASS** | **8** | gemm_tile@prefill: q4_0, q4_1, q5_0, q5_1, q8_0 (5) + quantize_row: q8_0, q8_1, q8_K (3) |

**合计 22 PASS + 6 具名-X = 28。**
（主会话点名的 6 格 `dequantize_row|{q4_K,q5_K,iq1_m,iq3_s,iq4_xs,tq2_0}` 被本扫描独立复现 ✓，
但**只占 28 的 21%**；**漏掉的 22 格全是 PASS**——即计入头条分子的那一侧。）

### 2.2 两种污染形态（须分开报 · 诚实边界）

| 形态 | 格数 | 板 f[33] 标签 | master note 标签 | 证据 |
|---|---|---|---|---|
| **甲：逐格假标签** | **17** | `clang18-sym`（15 格）/ `gcc15.2-deploy-lane`（q4_1, q8_0 2 格） | **★逐格明写「单世界clang(PR-17)」** | 见 2.3 |
| **乙：快照级标签** | **11** | `gcc-15-deploy=MAIN` / `gcc15.2-deploy-lane`（**诚实自认**） | **对编译器沉默** | 见 2.4 |

工具机算小计：`★声称单世界clang=17 | 对编译器沉默=11`（17 + 11 = 28 ✓）

> **★自纠二处（前科 8 防范：数 generator 的 INTENT 而非真消费）**：初版把 `gemm_tile@decode`
> 与 `engine=ime` 的格也算进污染（曾报 33）。**错**——recon `:309`（`sed` 确认）有
> `skip_t3 = (eng=="ime") or (op=="gemm_tile" and regime=="decode")`：这两类格**不读板 CSV 的 t3 行**
> （ime 走 `IME_KERNELSYM`、gemm decode 走 `GEMM_DECODE`），故**不因该板行的 gcc 数受污染**。
> 工具已复刻 recon 自己的这道门 → 只数**真消费**板行的格。修正后 = 28。

### 2.3 形态甲 · 逐格假标签（17 格 · dequant）

15 格的**同一行内自相矛盾**（`sed -n 56p` 已确认 T3_A:56）：

```
f[0]  = dequant|iq1_m|streaming|arity1|f32
f[33] = clang18-sym                                    ← 标签：clang
f[35] = in-denom*FAIL0.8-DEQ-NAMED-X-AUTOPROMOTED-gcc15.2-lane(0.35x-vs-真向量对手
        ·gather-tax/emit-headroom·byte-exact·decree禁躲门·domain=gcc-15.2-deploy-lane)
                                          ↑ 数据：gcc-15.2 车道
```

**★另 2 格（q4_1 / q8_0）是新发现，主会话未点名**：板标签**本来诚实**
（f[33] = `gcc15.2-deploy-lane(rvv shipped·DEQ per-lane deploy-matched…)`，
f[29] = `1.284@N24-2seed` / `0.838@N24-2seed` = 真 gcc-lane 数），
**是 recon 把它们重贴成 clang 的**。这两格直接关系 PR-17 原文：

> PR-17：「DEQ FLAT-5 rvv **5/5 PASS**（旧 3 具名-X q4_0/q5_0/q5_1）… **q4_1/q8_0 本已 PASS**」

即「真-对称 clang-18 域 5/5 PASS」实为 **3 格 clang 重测 + 2 格 gcc-lane 数原地沿用**。
q4_0/q5_0/q5_1 确被 `CLANG_WORLD` 以 clang 数覆盖（q4_0: 板 gcc `0.703` → clang `2.51`）= **翻转执行正确**；
q4_1/q8_0 **从未在 clang 世界重测**，其 PASS 仍是 gcc 数。

### 2.4 形态乙 · 快照级标签（11 格）

板 f[33] **诚实自认 gcc 为 MAIN**，例：

```
gemm|q5_1: f[33] = gcc-15-deploy=MAIN(rvv shipped gcc·clean部署数)|clang18-micro=opp-insensitive~1.002x(T9§5双证)
           f[29] = 1.412@nr16(prefill-GEMM)     → master: gemm_tile,q5_1,rvv,prefill,…,PASS,1.412
quantize|q8_0: f[33] = gcc15.2-deploy-lane(rvv shipped·DEQ per-lane deploy-matched·[CASE-COMPILER-ASYMMETRY]避clang-micro-artifact)
```

master 的 per-cell note 对编译器**沉默**，但整个工件被盖章
`snapshot g8-master-final-clang-world`（`recon:25`，落在 stdout + rowclue 头）。
∴ 这 11 格的 gcc 数**在 clang-world 横幅下入账**——标签不在格里，在快照上。

---

## 三 · 头条影响（★禁预测结果）

recon 本次运行头条（rvv）：**标量类 36/51 · 通用向量 22/27 · 手调 8/24 · Σ=102**。

- **标量类-rvv 36/51**：其 PASS 分子 **36 中有 14 格**的数来自 gcc 车道；另 6 格 gcc 数计在 `具名-X`（分母侧）。
- **通用向量-rvv 22/27**：其 PASS 分子 **22 中有 8 格**的数来自 gcc 车道。

**★若按单世界 clang 重测 → 结果【未测】。**
本审计**不预测**任何格的翻转方向、**不预测**头条增减（PR-37 · 「预判不作结论」）。
可说的只有：这 28 格现以 gcc 数入账，而 A1 evidence.md 记录了**双向**素材
（方向 A gcc 杀我方 kernel / 方向 B clang 杀对手 autovec），故**方向不可由先验推定**。

> **观察到的一处口径差（不解释成因 · 未调查）**：PR-17 原文写「标量类-rvv 24/51→**34/51**」，
> recon 本次运行 emit **36/51**（X 19→9 与原文一致）。差 +2 PASS。本审计**不给因果**，只记事实。

---

## 四 · recon 的贴标机制（file:line · 全部 `sed -n Np` 确认）

**`.trellis/scripts/recon_master_rebuild.py:290`**：

```python
note = (note+" · " if note else "")+"scalar源 autovec→标量类(§〇.1)·per-format clang18 autovec·单世界clang(PR-17)"
```

**这是一句无条件赋值**，落在 `elif op == "dequantize_row":` 分支（:287）内，
**对全部 24 个 dequant 行生效**，**不读 f[33]、不问数据来自哪个编译器**。
`CLANG_WORLD`（:229 声明，:356-360 应用）随后只覆盖 3 个 DEQ 格的 note；
其余 17 格保留 :290 的「单世界clang(PR-17)」戳。

**数据侧的两条路（也已 sed 确认）**：
- `:47` `v=f[35]` → verdict **TOKEN** 分类成 PASS/具名-X（**文本不透传**）
- `:61` `craw=f[29]` → 正则取**数字**（**文本不透传**）

∴ 板 CSV 里 f[33]/f[35] 的 **gcc 字样从结构上不可能到达 master**。

---

## 五 · ★PR-17 验收门是不是空心的（本审计最重要一问）

PR-17 原文（`docs/PENDING_RULINGS.md:34`）：

> 「**recon 输出/TIER/头条零 gcc 字样**（grep csv/clue/stdout = 0）」

**——原文自己写的就是「零 gcc 字～样～」。门的观测量是字样。**

### 5.1 我自己设计并跑的证伪实验（变异测试）

**设计**：
- **对照组 A** = 现状 recon 输出。
- **实验组 B** = 把两板 CSV **每一行**（162 行）的 f[33] 改写成
  `gcc-15.2-EVERY-NUMBER-IN-THIS-ROW-IS-GCC-MEASURED`，并在 f[35] 末尾追加
  `·PROVENANCE=gcc-15.2-lane-ALL`。**数字（f[29]）与 verdict token 一字不改**
  = 构造「100% 数据自认 gcc」的世界。
- 判据：门若验**数据来源** → B 组 grep 必 >0（门应炸）；门若只验**字样** → B 组仍 0。

**实跑输出**（`python3 tools/visibility/audit_gcc_lane.py` · 表5）：

```
[对照组 A · 现状]
  grep -ci gcc T3_master_rebuild.csv = 0
  grep -ci gcc T3_master_rowclue.txt = 0
  grep -ci gcc recon stdout          = 0
  master sha256[:16] = 1a50018c7ce104c3

[实验组 B · 全 162 行溯源字样改成尖叫式 gcc]
  grep -ci gcc master_MUT.csv        = 0     ← 门【没炸】
  grep -ci gcc rowclue_MUT.txt       = 0
  grep -ci gcc recon stdout          = 0
  master sha256[:16] = 1a50018c7ce104c3      ← 与 A 组【逐字节相同】
```

### 5.2 判定：**★门是空心的**

输入侧 100% 自认 gcc，门仍读 **0**，且账面输出 **sha256 完全相同**。
∴ **门的读数与数据的编译器来源之间没有任何函数依赖 = 恒真检查。**

**机制**（非臆断 · 由 §四 的 file:line 直接支持）：recon 的 note 全部来自脚本内**硬编码字面量**，
从板 CSV 只取 f[29] 的**数字**与 f[35] 的 **token 分类**；文本不透传
→ gcc 字样**结构上不可能**出现在输出里 → `grep=0` 是**恒等式，不是证据**。

**「grep=0」为真，与「数据全是 gcc」可以同时为真。**本审计正是这一情形：门读 0，28 格靠 gcc 数。

> **这正是前科第 10 类（空心检查 ×5）的同型复发**：门验的是**标签**，不是**那个东西**。

### 5.3 一个非空心的门长什么样（仅示形 · 不裁采纳）

门若要真验数据，须**读 f[33]/f[35] 的来源域**并断言：
「凡在分母内且未被 `CLANG_WORLD` 覆盖的行，其 f[33] MAIN 声明 ∧ f[35] token 区**不得**含 gcc 车道指纹」。
`audit_gcc_lane.py` 表3 即该断言的实现——它现在返回 **28**，非 0。

---

## 六 · 全仓其余 gcc 依赖（逐条 + 合法性）

### 6.1 合法 · CRT/libgcc/汇编器定位（**非 gcc 编译**）

clang-18 编译，仅借 gcc 的 CRT/libgcc/GNU as。**A1 既定 recipe · 钦定 · 合法**。

| 条 | 证据 |
|---|---|
| `--gcc-install-dir` | `experiments/active/g8-stage3-attack/P1-backfill7-raw/run_rvv_dequant_p1.sh:29`（`sed` 确认）：`LDFLAGS_RVV="--gcc-install-dir=$GT/lib/gcc/riscv64-unknown-linux-gnu/15.2.0"`；同文件 :22-24 自述「照抄 A1 verified recipe」 |
| `--gcc-toolchain=gcc-15.2.0` | `A1-rvv-clang18-unify/evidence.md:14` |
| `-L$GTOOL/lib -Wl,-rpath` 解 `libgcc_s` | 同上 :15（lld 不搜 gcc lib 路径） |
| `-fno-integrated-as` → GNU as | 同上 :15（clang 集成汇编器拒收 `quants.c` 内联 vsetvli） |

**★须诚实并记**（A1 evidence.md:16 自述）：clang 世界产物的 `.comment` **含 `GCC 15.2.0`**
（= `-fno-integrated-as` 的 GNU as + libgcc），原文标注「正常」。
∴ **「单世界 clang」= clang-18 做 codegen，但汇编/CRT/libgcc 仍是 gcc 组件**——已披露、合法，**不是**违规残留。

### 6.2 合法 · PR-17 明令保留的归档素材夹

| 条 | 证据 |
|---|---|
| `T-VALIDITY_compiler_symmetry_ledger.md` **§STAGE-3** | `:159`（`sed` 确认）：「★ STAGE-3 gcc 存量归档（PR-17 单世界 clang 终裁 · [CASE-COMPILER-ASYMMETRY] 双向素材夹）」·全文 61 处 gcc |
| `T-VALIDITY-STAGE1_{rvv,k1}_symmetric_remeasure.md` | 同族归档 |
| `tools/e2e-harness/board/case-compiler-asymmetry/` | [CASE-COMPILER-ASYMMETRY] 素材夹（PR-17 明令保留） |
| `docs/reports/2026-07-10-CASE-COMPILER-ASYMMETRY-casefile.md` | 案卷 |

### 6.3 活的 gcc 编译车道（**非归档** · 仍可跑出 gcc 数）

**这些是 §二 那 28 格数据的同源车道**。审计**不裁**其去留，只列。

| 条 | file:line（`sed` 确认） | 性质 |
|---|---|---|
| `stream_paired_board.sh` | `:26` `build_ledger(){  # $1 = clang\|gcc  -> ONLY the OUR-kernel .o codegen varies; driver+link ALWAYS g++ (neutral,` | **双账本 harness**：可建 gcc 臂；建的核 = deq_*/qnt_*/fwd_*（与 §二 的 dequant/quantize 同族） |
| `fullmarch_rebuild.sh` | `:25` `CC="${CC:-/opt/tcrv-toolchains/gcc-15.2.0/bin/gcc}"` | gcc-15.2 为**默认** CC（可 env 覆盖） |
| `flat_gemm_paired.sh` | `:16` `GCC=$(command -v riscv64-unknown-linux-gnu-gcc \|\| command -v gcc)` | FLAT gemm 配对测量 |
| `g7-l1-kernelsym-fullfill/run_rvv_cold.sh` | `:10` 同上 | kernel-sym cold 测量 |
| `tools/perf-isolation/mixed_build.sh` | 全文 24 处 gcc | 混合构建 |

### 6.4 中性 · driver/link 恒 g++（两臂相同 · 不构成不对称）

`stream_paired_board.sh:26-38` 明写「driver + FINAL LINK always g++-15 (neutral): only kernel codegen differs
between ledgers」。**两臂同 driver ⇒ 不产生编译器不对称**。属既定 flat recipe，**合法**。

### 6.5 中性 · 测试工具

`decode_snapshots.sh:47` 用 gcc 编 `tcrv_memhog.c`（内存压力工具，非被测核）→ 合法。

---

## 七 · 机核（可原样复制粘贴复跑）

```bash
cd /home/kingdom/phdworks/TianchenRV

# 1) 全审计（表1 标签×数据 / 表2 逐格 / 表3 污染格+证据行 / 表3b master层 / 表4+4b 头条 / 表5 空心测试）
python3 tools/visibility/audit_gcc_lane.py

# 2) PR-17 门读数（复现「grep=0」）
for f in experiments/active/result-tables/T3_master_rebuild.csv \
         experiments/active/result-tables/T3_master_rowclue.txt; do
  echo -n "$f : "; grep -ci gcc "$f"; done
python3 .trellis/scripts/recon_master_rebuild.py | grep -ci gcc

# 3) 贴标行 + 数据两条路（file:line 自证）
sed -n '290p' .trellis/scripts/recon_master_rebuild.py   # 无条件贴「单世界clang(PR-17)」
sed -n '47p;61p' .trellis/scripts/recon_master_rebuild.py # 只取 f[35] token / f[35]→f[29] 数字
sed -n '309p' .trellis/scripts/recon_master_rebuild.py    # skip_t3(ime/gemm-decode 不消费板行)
sed -n '25p'  .trellis/scripts/recon_master_rebuild.py    # SNAPSHOT=g8-master-final-clang-world

# 4) 同行自相矛盾（板 CSV 原文）
sed -n '56p' experiments/active/result-tables/T3_A_board_A_rvv1.0_vlen128.csv \
  | awk -F',' '{print "f0 ="$1; print "f33="$34; print "f35="substr($36,1,80)}'

# 5) 只读自证：真仓库零改动
git status --porcelain experiments/active/result-tables/ .trellis/scripts/   # 期望：空
```

recon 可重入且确定性：本审计跑了 3 次 recon，`git status` 恒空（输出逐字节相同）。

---

## 八 · 诚实边界 / 未做

1. **不裁去向**。改数据 / 重测 / 改 recon / 改 PR-17 = **硬冻结 + 必问**。本文只出事实 + 影响面。
2. **零重测预测**。28 格若按单世界 clang 重测 → **未测**。方向不推定（A1 记录了双向素材）。
3. **「污染」是口径词，不是错误指控**。28 格的数**本身可能都是真实测量**；
   问题是 **它们被呈现的编译器域 ≠ 它们的来源域**。形态甲（17 格逐格假标签）与
   形态乙（11 格快照级标签、板标签诚实）**性质不同**，已分开报，不合并成一个数当棒子。
4. **未追**：那 28 格各自的原始测量脚本/session 逐格溯源（只追到车道 §6.3，未逐格钉到 raw）。
5. **未追**：PR-17 原文 34/51 vs recon 现 36/51 的 +2 差因（只记事实，不给因果）。
6. **未查**：`experiments/archive/**` 与旧 28 列 STALE 行（PR-17 明令归档保留，非本审计范围）。
7. **k1 板**：本审计判 k1 = 0 gcc 数据。依据是 f[33] 自认 clang-18-deploy + f[29] 为 k1 自身数 +
   board B 全表无 `gcc15.2-lane`。**未**独立复验 k1 的 .so 实际由何编译器构建（信板 CSV 自述）。
8. **并行写入者**：审计期间 `experiments/active/g8-stage3-attack/P2-grid4-{prereg.md,raw/}`
   由**他人**新增（mtime 09:01/09:07，早于本工具 09:09）。**非本审计所为**；本审计只新建
   `tools/visibility/audit_gcc_lane.py` 与本文件。

---

## 九 · 一句话结论

> **PR-17 的门验的是「gcc 这三个字母有没有印在纸上」，不是「数字是不是 gcc 量出来的」。**
> 变异测试证明该门恒真（输入 100% gcc → 仍读 0 → 输出逐字节不变）。
> 门读 0 的同时，**28 格**（22 PASS + 6 具名-X）的 verdict 由 gcc-15.2 车道数据决定：
> **17 格被 recon:290 逐格贴上「单世界clang(PR-17)」**，**11 格**在
> `snapshot g8-master-final-clang-world` 横幅下沉默入账。
> 其中 **q4_1/q8_0 两格直接坐实 PR-17 原文的「DEQ FLAT-5 rvv 5/5 PASS」实为 3 格 clang 重测 + 2 格 gcc 数沿用。**
