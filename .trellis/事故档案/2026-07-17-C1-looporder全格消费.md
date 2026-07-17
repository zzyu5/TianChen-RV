# C1 · loop-order 一等属性全格消费（收敛工单 A2 / 空格表 B15）

日期：2026-07-17 · 分支 `refactor/full-refactor-m1` · 起始 HEAD `012a7580d`
触碰集：`lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp` · `test/Conversion/RVV/` 六份 fixture · 本报告
未碰：`schema/` · `docs/canon/` · `tools/` · `.github/` · `experiments/` · 板（无 `ssh rvv`/`ssh k1`）

---

## 一、结论速览

| 项 | 结果 |
|---|---|
| loop-order 每格消费 | **18/18**（机测；修前 16/18） |
| col-outer fixture 空心门 | **0**（修前 **5 空心**） |
| 「全部发射器读取」 | 对**有 nest 自由度的路径**已达成；其余**架构不可达**（见 §四） |
| 「不读者构建失败」 | **未实施** — 评估完毕，属跨线 op 契约变更（爆炸半径 53 fixture），**只登记不执行** |
| emitted C 字节等价 | 18/18 **byte-identical**（真基线重建对比） |
| build / lit | 绿（940/943；3 例失败为**既存**、与本线无关） |

**★ B2 前提被证伪**：知识架构定案 B2 称 loop-order「仅 q4_K 一格 realize·余格发射器单固定 nest **不读**」。机测证明**每一条 live 的 typed repack-GEMM 发射路径都读**。B2 的说法来源是前门一段**过期注释**（详见 §三.1），非代码事实 —— [GOV-8] 正例。

---

## 二、动手前的实况核对（[GOV-8]）

复跑：

```bash
cd /home/kingdom/phdworks/TianchenRV
grep -rInE 'loop_order|loopOrder|col_outer|row_outer' lib/ include/ | awk -F: '{print $1}' | sort | uniq -c | sort -rn
```

实测（本会话）：

```
48 lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp
41 include/Weft/Plugin/RVV/RVVRepackTilingSelection.h
28 lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp
 1 lib/Conversion/RVV/RVVToEmitCInternal.h
```

且 `test/` 下已存在 **13 份 `-col-outer` fixture**（q2_K/q3_K/q5_K/q6_K/tq1_0/tq2_0/iq4_nl/iq4_xs/iq2_s/iq2_xxs/mxfp4/full-body/prior-override）—— 与「仅 q4_K 一格」不符，遂自行机数，未采信 B2。

---

## 三、机测清单（逐格）

### 三.1 消费结构（谁读、在哪读）

发射侧只有一个 dispatcher 读属性：`RVVToEmitCBlockQuantLinear.cpp::emitTypedRepackGemmLoopBody`（`:3144`），**两个 resolution 站点**：

| 站点 | 读什么 | 供给谁 |
|---|---|---|
| `:3175` `siblingColGroupOuter` | `weft_rvv.loop_order` + `..._selection_reason` | 11 个 sibling body（K-quant×4 / ternary×2 / codebook×3 / grid×2）+ 内联 flat 全身（`lane_wise_vector_scale`，`:4029`） |
| `:3642` `colGroupOuter` | 同上（q4_K 自有 arm） | `emitRepackKQuantGemmBodyQ4K` |

12 个 body 签名带 `bool colGroupOuter` 并 `if (colGroupOuter)` 分叉；第 13 条路径是 flat 内联全身。

**住址倒挂属实但是刻意的 measured-gate**：`:3180-3196` —— 前门选了 `col_outer` 但 `selection_reason != "measured"` 时，发射器**拒绝翻转**、realize 成 `row_outer`，并发一条 `weft_emitc.loop_order_override` record。这是与 full-LMUL[B] / [ROLL] 同纪律的「未实测不翻船」闸，不是遗漏。

**过期注释（B2 的来源）** `RVVLowerQuantContraction.cpp` 原 `:1157-1162`：

> "the sibling K-quant / flat / codebook GEMM emitters currently carry a single fixed nest and do NOT yet read it"

`siblingColGroupOuter` 是后加的，注释未同步。本轮已改写为机测事实。

### 三.2 逐格消费（修后）

复跑（自包含，直接粘贴）：见 §六 脚本 A 段。实测：

```
iq2-s stamp=1 diff=100 YES     q2-K  stamp=1 diff=108 YES     q5-0 stamp=1 diff=68  YES
iq2-xs stamp=1 diff=100 YES    q3-K  stamp=1 diff=108 YES     q5-1 stamp=1 diff=68  YES
iq2-xxs stamp=1 diff=100 YES   q4-0  stamp=1 diff=68  YES     q5-K stamp=1 diff=108 YES
iq4-nl stamp=1 diff=100 YES    q4-1  stamp=1 diff=68  YES     q6-K stamp=1 diff=108 YES
iq4-xs stamp=1 diff=100 YES    q4-K  stamp=1 diff=108 YES     q8-0 stamp=1 diff=68  YES
mxfp4 stamp=1 diff=100 YES                                    tq1-0 stamp=1 diff=100 YES
                                                              tq2-0 stamp=1 diff=100 YES
=> CONSUMES 18/18, NOT 0/18
```

判据：`stamp≥1`（前门确实盖章）**且** `diff>0`（属性翻转 ⇒ 产物变）。**修前 tq1-0 / tq2-0 = `stamp=0 diff=0`**。

---

## 四、可达性判定（含架构不可达具名上报）

总令原文：「**全部**发射器读取，逐格 lit 证明属性变→产物 nest 变；**不读者构建失败**。」

### 四.1 架构不可达 —— 具名上报（令文「架构不可达上报例外」）

loop-order 自由度的**存在前提**是「两个独立的 group 循环」（activation row-group × weight col-group，每个 `out[y,x]` 是私有 K-累加 ⇒ 换序 byte-exact）。**只有 prefill GEMM 有这个前提**。以下路径**结构上无 nest 可翻**，对其要求「消费 loop-order」无意义：

复跑：

```bash
cd /home/kingdom/phdworks/TianchenRV
for f in lib/Conversion/RVV/*.cpp lib/Plugin/IME/*.cpp lib/Plugin/Scalar/*.cpp; do
  printf "%-42s actGB=%-3s wgtGB=%-3s Gemm_fns=%s\n" "$(basename $f)" \
    "$(grep -c 'act_group_base' $f)" "$(grep -c 'weight_group_base' $f)" \
    "$(grep -cE '::emit[A-Za-z]*Gemm' $f)"
done
```

实测：**唯一**同时具备两个 group base 的文件是 `RVVToEmitCBlockQuantLinear.cpp`（actGB=14 / wgtGB=32）；其余 13 个发射器文件 **actGB=0 / wgtGB=0 / Gemm_fns=0**：

| 发射器 | 本性 | loop-order 有意义？ |
|---|---|---|
| `RVVToEmitCKQuant` / `CodebookFp4` / `GridCodebook` / `TernaryBinary` | block-dot / super-block 核（decode 侧） | 否（无 group nest） |
| `RVVToEmitCDeferredDequant` / `ForwardElementwise` | 逐元素流式 dequant / elementwise | 否（无 loop 序自由度） |
| `Plugin/IME/*` · `Plugin/Scalar/*` | 后端 emission driver | 否 |
| 前门 9 个 **GEVM**(`TypedRepackGemvLoopBodyOp`) builder | decode：**单 row group** | 否（换序对象不存在） |

⇒ **「全部发射器读取 + 不读者构建失败」按字面不可达**。可达且已达成的**正确写法**：

> **凡结构上具备 loop-order 自由度的发射路径（= 全部 9 个 typed repack-GEMM 构造点 / 13 条 GEMM 发射路径 / 18 个量化格式），loop-order 均为一等 IR 属性且被发射器消费（18/18 机测）。GEVM / dequant / elementwise / block-dot 路径无 group-nest 自由度，不在该属性的定义域内。**

这是**架构本性**，非未完成 —— 不降标、不静默拖延，据实上报。

### 四.2 「不读者构建失败」形态评估（**未实施**，登记）

| 形态 | 可行？ | 证据 / 代价 |
|---|---|---|
| (a) ODS 必填 attr / verifier 拒绝缺失 | 技术可行，**本可抓住本轮 ternary 漏洞** | `weft_rvv.loop_order` 是 **discardable** 方言属性（不在 `*.td`），verifier 不认识。改必填的**爆炸半径 = 53 份 fixture**（70 份含 `typed_repack_gemm_loop_body`，其中 53 份无 loop_order、靠发射器缺省 row_outer）。属**共享 ODS/verifier 契约变更 · 跨线串行**，不单方面动。 |
| (b) 前门盖章点覆盖检查 | **已以等价形态落地** | 新增 fixture（§五.3）在「漏调用」与「轴重新耦合」两种回归下均转红（双负控实证）。 |
| (c) `.value_or("mf2")` 清零 | **不碰**（越界） | 机核：`grep -c '\.value_or("mf2")' lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` = **18**（brief 数字属实，已独立复核；`:2311` = `loopBody.getIntegerCoreLmul().value_or("mf2")`）。同类病但属**宽度轴 F-7/C9 域**，只登记。 |

---

## 五、本轮改动

### 五.1 前门：两个正交轴解耦（真 bug 修复）

**根因（两条独立成因，缺一不可）**：

1. `stampTilingSelection` 内 `classifyTilingBottleneckShape(foldModel)` 返回 `nullopt` 即 early-return（`:1084`），而 **loop_order 盖章写在这个 return 之后**。`ternary_single_fp16_scale` **不在** classifier 表内（`RVVRepackTilingSelection.h:105-116`）⇒ 早退 ⇒ loop_order 一并被跳过。**两个正交轴共用了一个闸**。
2. 9 个 `TypedRepackGemmLoopBodyOp` 构造点里，**ternary 那个（原 `:3362`）根本没调用盖章**（唯一两轴都不盖的构造点）。

**修法**：`stampTilingSelection` → `stampScheduleSelections`；hash / minVLEN 上提为两轴共用；抽出 `stampLoopOrderSelection` 并在 tiling classifier 早退**之前**调用；ternary GEMM builder 补上调用。

**发射中立性（非空口）**：ternary 的 `weightStride < activationStride`（tq1_0 **864**、tq2_0 **1056** vs q8_K activation **1168**）⇒ `repackColGroupOuterForLayout` 返回 false ⇒ layout prior = **row_outer** = 本就出货的 nest，且 override record 只在 `col_outer` 时发 ⇒ 产物不变。

### 五.2 五份空心 fixture 修实（**本轮最重要的发现之一**）

修前实测：`q2-K` / `q3-K` / `q5-K` / `q6-K` / `full-body` 五份 col-outer fixture **属性翻成 `row_outer` 仍全绿** = 空心门。

**空心成因（已解剖，非猜测）**：原 CHECK 形如「`CHECK: for` → `CHECK: literal "3360"` → `CHECK: for` → `CHECK: literal "1168"`」。但两种 nest 发出的字面量**多重集相同**（col/row 两侧 `3360`×3、`1168`×4），FileCheck 前向扫描在 row_outer 产物里照样能凑齐该序列。

**判别键**：外层 group 循环的 `callee=` 标记 —— 全文件**各仅一次**（`callee=weight_group_base` / `callee=act_group_base`），col_outer ⇒ 权重 group base 在**外**层。故以 `CHECK-NEXT` 钉在各自 for 头上。

⚠️ 注意：这 5 份 fixture 的**发射器本身没有 bug** —— 机测 col/row 产物 diff=108 行，**发射器确实翻**；纯属测试判据太弱。

### 五.3 新增前门盖章回归 fixture

`test/Conversion/RVV/rvv-ternary-gemm-loop-order-stamp-frontdoor.mlir` —— 钉住 ternary 前门盖章（`row_outer` / `prior` / 完整 [D-4] record）。

---

## 六、复跑与负控

### 脚本（自包含；仓库根执行，可原样复制粘贴）

```bash
bash docs/reports/2026-07-17-C1-looporder-verify.sh
```

（A 段 = 逐格消费；B 段 = 空心门检查。）实测尾行：

```
=> CONSUMES 18/18, NOT 0/18
=> HOLLOW = 0
```

### 负控（证明门会红）—— 三组，全部实证

| 负控 | 做法 | 结果 |
|---|---|---|
| **N1 · 13 份 col-outer fixture** | `loop_order` 翻 `row_outer` 后跑原 CHECK | **全 13 转红**（修前 5 份仍绿 = 空心） |
| **N2 · 新前门 fixture / 漏调用** | 删掉 ternary builder 的 `stampScheduleSelections` 调用 + 重建 | **转红** |
| **N3 · 新前门 fixture / 轴重耦合** | 把 `stampLoopOrderSelection` 移回 classifier 早退之后 + 重建 | **转红** |
| **N4 · byte-exact 等价检查自身** | 往 AFTER 产物注入 1 行 | **diff 2 ⇒ 转红**（检查本身会红） |

两次负控后均已 `cp` 还原并重建，fixture 复绿。

### emitted C 字节等价（真基线重建，非推理）

以 `git show HEAD:<file>` 换回原版 → 重建 → 采 18 格全流水产物 → 换回本轮版 → 重建 → 逐格 diff：

```
CHANGED = 0 / 18  => emitted C byte-exact across the fix
```

（含 27 333 行的 iq2-s、23 209 行的 q4_K 等大格。）

### build / lit

```
Total Discovered Tests: 943   Passed: 940 (99.68%)   Failed: 3 (0.32%)
```

3 例失败 = `Scripts/rvv-generated-bundle-abi-e2e-{self-test,explicit-…,pre-realized-…}`，**既存、与本线无关**，证据：
- `git status --porcelain scripts/rvv_generated_bundle_abi_e2e.py` **空**（脚本与 HEAD 一致）；
- 该 `.test` 内 `weft-opt` 出现次数 = **0**（纯 python，无编译器参与）；
- 直跑 `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` 在未改脚本上即 `AssertionError`。

---

## 七、诚实缺口 / 欠账

1. **性能未主张**：本轮**零**性能主张。loop-order 的翻转在 17/18 格上被 measured-gate 挡住（仅 q4_K 有 `col_outer` measured seed，board hash `3cd23a4e`）。「铺开消费」= **机制/结构**成果，**不是**性能成果；e2e 传导无证据、不外推。**未碰板**。
2. **17/18 格 reason=prior**：意味着 col_outer 在这些格上**从未实测**。要真翻需 Stage-3 板 A/B 播种 —— 属 **P 线板时**，本轮如实标 **board-pending**。
3. **「不读者构建失败」未落地**：§四.2(a) 是唯一能机械杜绝本轮 ternary 漏洞的形态，但触及共享 ODS/verifier 契约、爆炸半径 53 fixture ⇒ **须裁决**，未单方面动。**当前替代 = §五.3 fixture 双负控**（能抓回归，但不能像必填 attr 那样抓「新增构造点忘盖章」的**未来第 10 个 builder**）。
4. **ternary 的 col_outer 路径无 fixture 覆盖真实前门产物**：ternary 前门现出 `row_outer`，其 `-col-outer` fixture 仍是手工盖章（`measured`）。真实前门永不产出该组合，直到板上播种。
5. **`emitRepackGemmQ4_1Q8_1`（`:7498`）= 死代码**：机核 call site **0**（仅定义 `:7498` + 声明 `RVVToEmitCInternal.h:1467`）。它有真 nest 自由度却不读 loop_order —— 但**不可达**，故不计入分母。**登记待清**，本轮未删（删除属不可逆动作）。
6. **`.value_or("mf2")` ×18** 同类病（plan 没送到 emitter、emitter 自己补缺省），属宽度轴 F-7/C9 域，**只登记不执行**。

## 七.5 触碰集实况（并发线共树）

本线实际改/建 **9 个文件**，全部在授权触碰集内：

```
M  lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp
M  test/Conversion/RVV/rvv-to-emitc-repack-gemm-{q2-K,q3-K,q5-K,q6-K}-q8-K-col-outer.mlir
M  test/Conversion/RVV/rvv-to-emitc-typed-repack-gemm-loop-body-vlen128-full-body-col-outer.mlir
??  test/Conversion/RVV/rvv-ternary-gemm-loop-order-stamp-frontdoor.mlir
??  docs/reports/2026-07-17-C1-looporder全格消费.md
??  docs/reports/2026-07-17-C1-looporder-verify.sh
```

⚠️ 工作树内另有**并发线**的脏文件（`docs/canon/…v4-草案.md`、`include/Weft/Support/CapabilityModel.h`、`lib/Support/CapabilityModel.cpp`、`tools/e2e-harness/board/iq4nl_gemm_paired_driver.c`、`tools/lint/…`、`experiments/active/…`）—— **本线一次都没有写过**，与本报告无关，入库时勿归到本线名下。（子代理直写主树，`worktree 隔离` 不可信 —— 既有记忆条目。）

未 `git add` / 未 `git commit`（主会话统一入库）。

## 八、措辞纪律自查

- 未称任何「硬赢 / 实质胜利」；本轮无性能数字。
- loop-order 属性在 18 格上属**强义 constructed**（前门 selector 真实计算并盖章、发射器真实分叉、属性变⇒产物 nest 变 机测），故可称「由机制构造」；未对任何**弱义**路径沾「构造/变换」。
- 「18/18 消费」严格限定于 **kernel/emission 轴**，**未**与 perf-covered 9/83 或任何系统账混谈。
