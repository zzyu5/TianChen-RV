# ISSUE-119 · ForwardElementwise dequant-row 宽谓词穷举到 100%（只读 census）

> **只读 census 交付物**。零代码改。仅静态阅读事实 + 机算命令。不下「该删/该迁」越权结论（列候选·裁由上游）。

## 0 · Pin（工作树 == HEAD 实测）

- `git rev-parse HEAD`（收工 pin）= **`20a5d52473f17dc9d54134f65b14adc87e561db4`**
- census 开工 pin = `fbec0b72d3d667b77d4046d127fdd11d5093c010`；会话中并行 agent 推进 HEAD（触碰 `RVVLowerQuantContraction.cpp`/`RVVCapabilityProfile.cpp`/tests·非本目标文件）。
- **目标文件跨两 pin 逐字节相同**：`git diff --stat fbec0b7 20a5d52 -- lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` = **空**（→ 本 census 对当前 HEAD 依然 100% 有效·所有行号不变）。
- `git status --porcelain lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` = **空**（工作树 == HEAD·目标文件无未提交漂移）。
- 目标文件 = `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`（**6134 行**）。
- 未 commit。仅写本交付物一个文件。

---

## 1 · 可复现「宽谓词」判据（分母定义·下个 agent 机算复跑得同数）

**「宽谓词」的忠实含义**（对齐 census pkg3 line 116 + ISSUE-119）= **dequant-row 发射体内、绑定到「每格块布局描述符字段」的裸整数字面量**——即变量名 ∈ `{stride, blockStride, qk, dOff, mOff, qhOff, qsOff, sub}`（ISSUE-119 命名字段集 `block_stride`/`quant_byte_offset`/`scale_byte_offset`/`qh_byte_offset`/`qk` 的代码对应）被赋一个（可经编译期 `isX ? A : B` 三元选择的）字面量。这正是「转描述符」迁移会替换为 typed getter 的常量集。**「宽」= 块宽/块几何**（qk = 块长、stride = 块字节步长、*Off = AoS 块内字节偏移），**非 SIMD 向量宽**（后者见 §5 邻轴 A·census 归 θ 桶非 g）。

**判据命令（分母 = 命中集·code-only·剥 `//` 行注释·dequant 家族范围 2616–6129）**：

```bash
F=lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp
awk 'NR>=2616 && NR<=6129 {
       line=$0; sub("//.*","",line);
       if (line ~ /(^|[^A-Za-z_])(stride|blockStride|qk|dOff|mOff|qhOff|qsOff|sub)[ \t]*=[ \t]*[^;]*[0-9]/)
         print NR": "$0
     }' "$F"
# 计数： ... | wc -l   →   59
```

- 范围 `2616–6129` = dequant-row 发射家族（`emitGgmlDequantizeRow` 2616 → `emitGgmlDequantizeRowExtended` 尾 6129）·由 §2 函数边界表机核界定（`grep -nE '::(emit|construct)[A-Za-z0-9_]+\($'`）。
- `sub("//.*","",line)` 剥行注释·避免把注释里的 `qk=32` 类文字误计（[[no-head-truncation-for-absence-claims]]：用精确谓词·不用有界窗口）。`/*name=*/N` 向量包装器命名实参**不是 `//` 注释**·故保留并命中（见 locus 3/6）。
- RHS `[^;]*[0-9]` 容忍编译期三元链（`isMx ? 32 : … : 256`）与多语句行（停在首个 `;`·每行计一次）。

**★分母 = 59 条命中行**（`grep -c` 可复跑）。（窄版判据 = RHS 紧跟字面量、排除 `/*name=*/` 包装器 = 43 行；本 census 取**广版 59** 为准·含包装器·§4 记该 16 行 = 8 个 call-site 跨 2 行排版。）

---

## 2 · 三分类计数（替换账本「~51」）

| 类别 | 数 | 说明 |
|---|---|---|
| **(c) 焊死-g** N1 | **59** | 全部 59 条 = 机制体内裸格式/几何字面量·发射体**只读 `format` 字符串 + ABI operand**·**无描述符字段/无 IR 宽度可读**（[K-10] MAINTAIN 形态·律2 视角 = 该入描述符）。 |
| **(a) 派生/f-消费** N2 | **0** | dequant-row 布局 g 轴**零派生**：没有一处由 coreLmul/VLEN/描述符字段算出（`deriveWideningChain` 类）。ISSUE-119 的本体即「描述符零字段·全烘焙」——N2=0 是该问题的定义性事实·非略过。（唯一擦边 = 2975 行 `half = qk/2` 是派生子式·但该行命中的是**焊死** `qk=32`。） |
| **(其它)** N3 | **0** | 无第三态。 |
| **Σ = 分母** | **59** | |

**板轴 vs 结构轴**：

- **板轴（VLEN128/256 分叉）命中 = 0**。机核证：`grep -nE '\bif\b[^;]*\b(VLEN|vlen|isRVV0p7|getVLEN|board|Vlen)\b | (VLEN|vlen|isRVV0p7|getVLEN)[^;]*\?'` 全文 = **NONE**（无运行期板分叉）。且 dequant 范围内 `value_or|getStripLmul|coreLmul|wideLmul` = **NONE**（dequant 路**零 strip-LMUL θ**·纯定长发射）。
- ⟹ **59 条全部 = 结构常量（两板同值·board-invariant）**。发射体顶注（2624 行）明写 dequant「per-strip f32 stores need no reduction, so LMUL/strip-count are correctness-free」。文件内所有 `VLEN>=128 / 256` 文字均为**注释**或 **ggml 超块常量 `QK_K=256`**·非运行期比较。

---

## 3 · 100% 穷举表（59 条·file:line + 现值 + 类别·全 (c) 焊死-g·全结构轴）

**Locus 1 — `emitGgmlDequantizeRow` legacy 标量 nibble 事实（分发线 monolith-fallback 的 q4_0..q8_0 头）** — 6 行
| line | 现值 | 备注 |
|---|---|---|
| 2658 | `int64_t stride=0,dOff=0,mOff=0,qhOff=0,qsOff=0,sub=0` | 6 字段零初始声明（每格分支覆盖） |
| 2661 | q4_0: `stride=18; dOff=0; qsOff=2; sub=8` | |
| 2663 | q4_1: `stride=20; dOff=0; mOff=2; qsOff=4` | |
| 2665 | q5_0: `stride=22; dOff=0; qhOff=2; qsOff=6; sub=16` | |
| 2667 | q5_1: `stride=24; dOff=0; mOff=2; qhOff=4; qsOff=8` | |
| 2670 | q8_0: `stride=34; dOff=0; qsOff=2` | |

**Locus 2 — nibble/q8_0 shared+vector body 的 qk/stride/qsOff 声明** — 4 行
| line | 现值 | 函数 |
|---|---|---|
| 2709 | `const int64_t qk = 32` | `emitDequantizeRowNibbleBodyShared` |
| 2975 | `const int64_t qk = 32, half = qk/2` | `emitDequantizeRowNibbleVectorBody`（half=派生子式·命中=qk焊死） |
| 3259 | `const int64_t qk=32, stride=34, qsOff=2` | `emitDequantizeRowQ8_0BodyShared` |
| 3394 | `const int64_t qk=32, stride=34, qsOff=2` | `emitDequantizeRowQ8_0VectorBody`（与 3259 同值·重复烘焙） |

**Locus 3 — nibble 向量包装器命名实参（`emitDequantizeRow{Q4_0,Q5_0,Q4_1,Q5_1}VectorBody`）** — 8 行（4 格 × 2 行）
| line | 现值 |
|---|---|
| 3215–3216 | q4_0: `/*stride=*/18,/*dOff=*/0,/*mOff=*/0,/*qhOff=*/0,/*qsOff=*/2, /*sub=*/8` |
| 3224–3225 | q5_0: `/*stride=*/22,…,/*qhOff=*/2,/*qsOff=*/6,/*sub=*/16` |
| 3233–3234 | q4_1: `/*stride=*/20,…,/*mOff=*/2,/*qsOff=*/4,/*sub=*/0` |
| 3242–3243 | q5_1: `/*stride=*/24,…,/*mOff=*/2,/*qhOff=*/4,/*qsOff=*/8,/*sub=*/0` |

**Locus 4 — nibble `*BodyShared` 包装器命名实参（`emitDequantizeRow{Q4_0,Q4_1,Q5_0,Q5_1}BodyShared`）** — 8 行（4 格 × 2 行·**与 Locus 1/3 第三次重复烘焙同一 4 格事实**）
| line | 现值 |
|---|---|
| 3506–3507 | q4_0: `/*stride=*/18,…,/*qsOff=*/2,/*sub=*/8` |
| 3517–3518 | q4_1: `/*stride=*/20,…,/*mOff=*/2,/*qsOff=*/4` |
| 3529–3530 | q5_0: `/*stride=*/22,…,/*qhOff=*/2,/*qsOff=*/6,/*sub=*/16` |
| 3541–3542 | q5_1: `/*stride=*/24,…,/*qhOff=*/4,/*qsOff=*/8` |

**Locus 5 — K-quant/超块 构造向量叶 几何声明** — 5 行
| line | 现值 | 函数 |
|---|---|---|
| 3574 | `qk=256, stride=isQ5?176:144, qsOff=isQ5?48:16` | `emitDequantizeRowQ45KVectorBody`（三元=两焊死字面量·仍 (c)） |
| 3575 | `qhOff=16` (q5_K only) | 同上 |
| 3752 | `qk=256, stride=84` | `emitDequantizeRowQ2KVectorBody` |
| 3879 | `qk=256, stride=110` | `emitDequantizeRowQ3KVectorBody` |
| 4057 | `qk=256, stride=210` | `emitDequantizeRowQ6KVectorBody` |

**Locus 6 — codebook / ternary 构造向量体 几何声明** — 5 行
| line | 现值 | 函数 |
|---|---|---|
| 4330 | `qk = isMx?32:isNl?32:isNv?64:256` | `emitDequantizeRowCodebookVectorBody`（三元链） |
| 4331 | `stride = isMx?17:isNl?18:isNv?36:136` | 同上 |
| 4599 | `qk = 256` | `emitDequantizeRowTernaryVectorBody` |
| 4600 | `stride = isTq1?54:66` | 同上 |
| 4601 | `dOff = isTq1?52:64` | 同上（tq1/tq2 fp16 d 偏移） |

**Locus 7 — `emitGgmlDequantizeRowExtended` 标量 monolith-fallback per-format stride switch（ISSUE-119 明命名 locus）** — 20 行
| line | 现值 |
|---|---|
| 5366 | `int64_t qk=256, stride=0`（默认声明） |
| 5368–5372 | Q2K 84 / Q3K 110 / Q4K 144 / Q5K 176 / Q6K 210 |
| 5373–5374 | MXFP4 `qk=32; stride=17` / NVFP4 `qk=64; stride=36` |
| 5375–5378 | TQ1 54 / TQ2 66 / Q10 `qk=128; stride=18` / IQ4NL `qk=32; stride=18` |
| 5379–5386 | IQ2XXS 66 / IQ2XS 74 / IQ2S 82 / IQ3XXS 98 / IQ3S 110 / IQ1S 50 / IQ1M 56 / IQ4XS 136 |

**Locus 8 — extended 体内 codebook/K-quant 内联 qsOff/dOff 声明** — 3 行
| line | 现值 | 上下文 |
|---|---|---|
| 5427 | `int64_t dOff=0, qsOff=2` | IQ4NL/MXFP4 nibble codebook 分支 |
| 5430 | `qsOff = 1` | MXFP4 覆盖 |
| 5663 | `qsOff = isQ5 ? 48 : 16` | Q4K/Q5K 分支 |

**Σ = 6+4+8+8+5+5+20+3 = 59** ✓（== `grep -c`）。

---

## 4 · 候选迁移点（律2 违例·类比 MIG-1 entryLanes·**列候选·不裁**）

全 59 条均为「机制体点入源·描述符零字段」的律2 面（发射体读 `format` 字符串再 switch·而非读 typed getter）。ISSUE-119 目标描述符字段 = `block_stride`/`quant_byte_offset`/`scale_byte_offset`/`qh_byte_offset`/`qk`。候选按值/扇出排序：

1. **[最高扇出] extended 标量 stride switch（Locus 7·5366–5386·20 行·19 格）** = ISSUE-119 命名的 monolith-fallback 核·转 `GgmlDequantizeRowOp` per-format stamp `block_stride`+`qk` 后发射体逐格改读。**先例键** = `RVVOps.td:2645/2677` LoadOp `OptionalAttr<block_stride>`+fail-closed（census pkg3 line 58）。
2. **构造向量叶几何（Locus 5/6·3574/3575/3752/3879/4057/4330/4331/4599/4600/4601·10 行）** = per-format `qk/stride/qsOff/qhOff/dOff` 焊在叶·同该入描述符。
3. **legacy 4-格 nibble 事实的三重复制（Locus 1+3+4·2658–2670 / 3215–3242 / 3506–3541·22 行）** = 同一 q4_0/q4_1/q5_0/q5_1 的 `stride/dOff/mOff/qhOff/qsOff/sub` **在三处独立焊死**（标量分发线 + 向量包装器 + BodyShared 包装器）。迁描述符可**一次消三处**（若消费侧统一读 stamp）。
4. **q8_0 双烘焙（3259 == 3394）** / **nibble qk（2709/2975）** / **extended 内联 qsOff（5427/5430/5663）** = 余量。

> ⚠ **不下越权结论**：以上仅列「可进迁移队列的 (c) 焊死-g 候选」·是否迁、迁序、是否回门/独立 scoping = 上游裁（ISSUE-119 保守默认 = 维持 baked·byte-exact）。

---

## 5 · 邻轴（明标·**不计入 §2 分母 59**·仅为答 PRD 分类问 + 诚实边界）

**邻轴 A — SIMD 向量宽/lane/fold-widen 档位（PRD 例举的「strip 宽/lane 数/fold-widen 档位」）**
- 机核：dequant 范围 2616–6129 内 `v(int|uint|float)*m*_t` 类型名字面量 = **37 处·13 个 distinct 种**（`vuint8m1_t`×7 / `vint32m4_t`×6 / `vfloat32m4_t`×6 / `vuint32m4_t`×4 / `vint8m1_t`×3 / `vint8m2_t`×2 / `vint32m8_t`×2 / `vfloat32m8_t`×2 / `vuint8m2_t` `vuint32m8_t` `vuint16m4_t` `vuint16m2_t` `vint16m2_t` 各 1）。
- **类别 = (a) 派生**：发射体自注（3377–3383 行）明写「the pipeline LMULs (i8m2 …, i32m8/f32m8 …) are **DERIVED from** that [fixed 32-lane block] width, **NOT literal knobs**」；4586–4587 行「nLanes<=16 so every widened LMUL fits VLMAX at VLEN128 … on any VLEN>=128」。**结构轴（board-invariant·非板分叉）**。
- **为何不入 g 分母**：census 约定（pkg3 line 120 + LEDGER §一）向量宽 = **θ/派生桶**（包1/2）·非 g 反向表。此处枚举仅为完整回答 PRD 的「(a)派生 + 板轴 vs 结构」分类问·并证「板轴命中=0」。

**邻轴 B — 内联字节偏移（census 自认「未穷举的小常量」·pkg3 §Caveat）**
- extended（4974–6129）+ 向量叶把大量 per-format 偏移**内联**烘焙（非命名标量·故不入 §1 谓词）：如 q2_K `d@80/dmin@82`、q6_K `ql@0/qh@128/scales@192/d@208`、`sizeLit(66+ib32)` 等。
- 可复现下界代理：`grep -nE 'fp16ReadAt\(xb, *[0-9]+\)'` = fp16 scale 偏移读·distinct 偏移值 = `{0,2,16,52,64,80,82,108,208}`（= per-format d/scale 字节偏移·全 (c) 焊死-g·结构轴）。
- **类别 = (c) 焊死-g·结构轴**·与分母同类·但为保持谓词清晰**不折进 59**（它们是消费块几何的解码算术·非命名描述符字段）。这是「命名标量分母」之外的**残余迁移面**。

**邻轴 C — 量化-encode 侧命名布局（OUT of dequant scope·非 ISSUE-119）**
- quantize_row q8_0/q8_1/q8_K 路（1089–2138）另烘 3 命名声明行（1133 `blockStride=34…` / 1372–1373 `blockStride=36…sumOffset=2` / 1614 `blockStride=292…`）+ 3 包装器实参行（1117/1356/1594）。记此为邻侧 encode 清单·**不属 dequant-row·不计入 59**。

---

## 6 · 与 ISSUE-119 现行条文交叉核（`.trellis/spec/issues/发射器与架构.md:208-214`·**只报·不改**）

| ISSUE-119 条文 | 本穷举 | 一致? |
|---|---|---|
| 「dequant-row **~40** g 轴烘焙」（issue 本体 line 208/211/213） | 精确命名标量分母 = **59** | **量级一致·数值皆低估**：~40 是 issue 文·census pkg3 headline 作 **~51**（line 116/123/154）·二者皆 <59。此即 PRD 指的「未穷举到 100%」缺口——现钉死 = **59**（可复跑）。 |
| 命名字段集 `block_stride`/`quant_byte_offset`/`scale_byte_offset`/`qh_byte_offset`/`qk` | 代码变量名 `stride`/`qsOff`/`dOff`+`scaleOff`/`qhOff`/`qk` | **一致**（同一布局字段集）。 |
| locus = 「`emitGgmlDequantizeRowExtended` 的 format-string switch/多变量声明·19 格 @5333-5351」 | 确认 = 现 5366–5386（Locus 7·19 格 stride switch·pin 漂移致行号平移~33 行·e6283a5fc 两遍重构） | **确认·但 locus 不止一处**：ISSUE-119 只提 extended monolith-fallback·**漏记**同一事实在 legacy 标量线（2658-2670）+ 构造向量叶（Locus 5/6）+ 双包装器（Locus 3/4）的**重复烘焙**（这正是 ~40/~51 低估的来源）。→ 一致性注·非改条文。 |
| 「monolith-fallback-keyed-by-format 架构」 | 确认（4197-4306 shared body 全为 keyed-by-`format` 字符串的 thin forwarder / dispatch） | **一致**。 |
| 保守默认 = ~40 维持 baked·byte-exact·独立 scoping 待排 | 本 census 零改·全维持 baked | **一致**。 |
| census「pin-shape 3（2709 qk=32·3575 qhOff=16·4564 qk=256）」 | 2709 qk=32 ✓ / 3575 qhOff=16 ✓ / extended qk=256 现 @5366（旧 4564 漂移） | **一致**（pin-shape ⊂ 59）。 |

**结论**：ISSUE-119 的**性质/字段集/架构判断全部一致**；唯一不一致 = **分母数**（issue 「~40」/ census 「~51」皆低估·真命名标量分母 = **59**），且 issue 文低估了**烘焙 locus 的数目**（只记 extended·漏记向量叶+legacy+双包装器的重复）。均为「未穷举到 100%」的表现·本 census 予以钉死。**不改 ISSUE-119 条文**（措辞变更入 issues 由上游裁）。

---

## 7 · 复跑清单（下个 agent 机算验证）

```bash
F=lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp
# 分母 59（§1 谓词）
awk 'NR>=2616 && NR<=6129 { l=$0; sub("//.*","",l);
  if (l ~ /(^|[^A-Za-z_])(stride|blockStride|qk|dOff|mOff|qhOff|qsOff|sub)[ \t]*=[ \t]*[^;]*[0-9]/) c++ } END{print c}' "$F"   # → 59
# 板轴 = 0（无运行期 VLEN 分叉）
grep -nE '\bif\b[^;]*\b(VLEN|vlen|isRVV0p7|getVLEN|board)\b|(VLEN|vlen|isRVV0p7|getVLEN)[^;]*\?' "$F"                          # → NONE
# dequant 路零 strip-LMUL θ
awk 'NR>=2616 && NR<=6129' "$F" | grep -nE 'value_or|getStripLmul|coreLmul|wideLmul'                                            # → NONE
# 邻轴 A：SIMD 宽字面量 37 处 / 13 种
awk 'NR>=2616 && NR<=6129 {while(match($0,/v(int|uint|float)[0-9]+m?f?[0-9]+_t/)){c++;$0=substr($0,RSTART+RLENGTH)}} END{print c}' "$F"  # → 37
```
