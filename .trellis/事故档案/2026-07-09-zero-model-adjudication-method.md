# ZERO-MODEL 终审法 — 数值正确性争议的标准裁决工具 (G3-cert-hardening 方法学 codify)

> ############################################################################
> **★[CASE CLOSED — 本法已被 M4 应用并裁死 min-term 案,commit `4f765790`(2026-07-10 线D 收尾指针)]**
> 本 spec 确立的 ZERO-MODEL 对位终审法**已被 M4 决定性终审实际执行**并**推翻了 M2c 的 min-term 误诊**:kernel 对真
> mat-quant dispatch **整数逐位一致**、且比 ggml 自身 generic 更近 int-exact(见 `docs/reports/2026-07-09-minterm-fold-audit.md`
> 顶 CASE-CLOSED 指针 + T8 `[CASE-MINTERM]` 卷宗)。**故文末"待逐项对位"队列的结局(2026-07-10 更新)**:
> `q4_K/q5_K repack-GEMM` 由 `inferred-defective` → **CASE CLOSED = 正确(FULL cert,`schema/cert-lineage.v1.json`
> M4-q4K/M4-q5K,cert-requirements-gate GREEN)**;`q2_K` = validated-via-shared-fold(M4-proven `kquant_dmin_bsums_min`)、
> **direct dmin≠0 cert OWED**;`repack GEVM` = **OWED**(M4 头对头走 GEMM 路径);`block-dot q4_k_min_term` = 低风险未直测,
> 仍在未来 dmin≠0 直测清单。**方法本身不变、入宪有效**(定罪与翻案同等严谨;默认走 ZERO-MODEL 对位终审两步、不再多轮 oracle 互搏);
> 本收尾仅登记该法**首个判例(min-term)已 CASE CLOSED**,不重写正文(append-only)。链 memory `[[zero-model-adjudication-cert-hardening]]`。
> ############################################################################

**性质**: 方法学 spec（防复发入宪的 *rule* 半边；CI *gate* 半边见 `tools/lint/check_cert_requirements.py`
+ `schema/cert-lineage.v1.json` + `.github/workflows/falsifier-gate.yml` 的 `cert-requirements-gate`）。
不改任何 kernel / 任何数值。
**入宪裁决（2026-07-09 G3-cert-hardening）**: 数值正确性争议的**标准终审工具 = ZERO-MODEL 法**；
「**定罪与翻案同等严谨**」；「**此类争议默认走 ZERO-MODEL → 对位终审两步，不再多轮 oracle 互搏**」。
**触发本 spec 的判例**: M2c 误诊（`docs/reports/2026-07-09-minterm-fold-audit.md`、snapshot commit `53666846`）。

---

## 0. 一句话定义

> **ZERO-MODEL 法** = 从**被测物自己的实际输入**出发、用一份**不与被测物共享任何读取/折叠实现**的
> 参照，**独立重算全部算术项**，并**逐项对比被测物捕获的内部 intermediates**；**0-mismatch ⟺ 正确**，
> 任一项 mismatch ⟺ 定位到该项的缺陷。

它不是"再跑一个 oracle 看端到端相对误差是否够小"（那是 M2c 之前 4 轮都没抓到缺陷的方式）；它是
**把被测物的算术拆成项、对每一项做零模型对位**。"零模型"= 参照对被测物的内部实现**零假设、零复用**：
既不借它的 quantizer，也不借它的 bsums/fold 读法，也不拿它的另一档配置来自比。

---

## 1. 背景：为什么"多轮 oracle 互搏"会漏、且必须废止

M2c 之前，q4_K repack-GEMM 的 min-term 缺陷经历了 **≥4 轮 oracle 对照全部 0-mismatch / rel≪1e-4**
（M0 tracer、M1 repacker cert、q2_K/q5_K SILICON numeric、L1 identity 门；见 min-fold 审计 §3 行 2–13），
却**全部漏抓**一个真实的 rel **5%–314%** 缺陷。根因是三类**结构性盲区**，任一都让"再来一轮 oracle"无效：

1. **输入路径不同源（用错 oracle）**。被测 dispatch 走 mat-quant（`ggml_quantize_mat_q8_K_4x1`,
   interleaved-q8_Kx4）；对照 oracle 却喂 row-quant（`quantize_row_q8_K`, plain）。两者激活布局不同，
   对照根本没走到被测的 interleaved 列-parity 算术 → 缺陷在对照里不存在，"0-mismatch"是**关于另一条路径**的。
2. **空心证书（自比 / 共享实现）**。PRE≡POST / untiled≡tiled / S1≡S6 的自比只证**不变量**（保序、保 tiling），
   不证**绝对正确性**：M2c 恰恰证明 S1 与 S6 **同时**带同一缺陷、自比仍 0-mismatch。号称"独立"却复用被测
   bsums 读法的 harness（#6 q2_K 5.77e-7、#11 q5_K 8.0e-7）同理——**看不到共享代码里的缺陷**。
3. **语料未激活该算术项**。INT / byte-exact 行全 `dmin=0` → min-term 从未点火。一个从不激活 min 的语料，
   跑多少轮都不可能抓 min 缺陷。

**结论**：这三类盲区是**同一 oracle 的属性**，不是随机噪声。所以"换个 oracle 再跑一轮"、"两个 oracle 互搏看谁对"
是**无收敛的**——它们可能共享同一盲区（都用 row-quant、都自比、都 dmin=0）。**必须**换成一个对被测物内部
**零复用、逐项对位**的裁决，即 ZERO-MODEL。

---

## 2. ZERO-MODEL 法定义（procedure）

给定一个数值正确性争议（"这个 fold/kernel/repack 到底对不对？"），执行：

1. **取被测物的实际输入**。**不是**另造一份"等价"输入——是被测 dispatch **当次真实喂进去的**权重块与
   激活块（含其真实 quantizer 产物的字节）。这封杀盲区 1。
2. **枚举算术项**。把被测物这一 fold 的数学拆成**独立可对位的项**（对 K-quant min-fold:
   `d·q` 主项、`dmin·Σbsums` min 项、per-sub-block scale、符号；对 flat: `scale·q`；对 codebook: 表查 + 累加）。
   项的清单即 `schema/cert-lineage.v1.json` 的 `fold_model_required_terms[fold_model]` 所锚定者。
3. **独立重算每一项**。用一份**不 import 被测代码**的参照（逐元素标量重算即可）算出每项应有值。
   参照的 quantizer 必须**与被测同源**（同 mat-quant / 同 interleaved），但**折叠/读取实现必须独立**。
   这同时满足三要件 ②（同源输入）与 ③（独立 oracle）。
4. **捕获被测物的内部 intermediates 并逐项对位**。不是只比最终输出——是把被测物**每一项的中间值**
   dump 出来，与步骤 3 的独立值**逐项**比。**0-mismatch（bit-exact 或 ULP-约束内）⟺ 正确**；
   任一项 mismatch ⟺ **定位到那一项**（M2c 即由此把缺陷钉到 min 项的偶/奇列 parity: R/correct 0.947/0.637）。
5. **语料覆盖每一项**（三要件 ①）。步骤 1 的输入必须让每个枚举项**非退化地点火**：min-fold 需 `dmin≠0`、
   `mins≠0`、非退化 scale、符号双向。未点火的项**不得**计入"已验证"，须标 partial。

**"0-mismatch = 正确"的成立条件**：仅当步骤 3–5 全部满足（同源输入 + 独立参照 + 全项语料 + 逐项对位）。
缺任一条，"0-mismatch"退回到**只对被测的一个投影成立**，不是正确性终审——这正是 M2c 前那几轮的状态。

---

## 3. 对位终审「两步」协议（替代多轮 oracle 互搏）

数值正确性争议**默认**按此两步走，**不再**开多轮 oracle 互搏：

- **第一步 — 语料 + 血缘对齐**。按 §2.1/§2.3 备好被测实际输入 + 一份同源-输入、独立-实现的参照；
  按 `fold_model_required_terms` 列出必激活的算术项，并构造语料使其全部非退化点火。
  （此步的**声明**即 cert-lineage 三要件条目；CI 的 `cert-requirements-gate` 守其**不空心、不narrow、不冒充**。）
- **第二步 — 逐项零模型对位**。dump 被测内部 intermediates，与独立参照逐项比：**全 0-mismatch → 判正确
  （byte-exact / claim=full 成立）**；**任一项 mismatch → 判缺陷，并输出该项的定位签名**（哪一项、哪一列/strip、
  相对幅度）。

**终止性**：两步产生**单一裁决**（正确 or 定位到项的缺陷），不进入"再换个 oracle"的循环。若第二步出现
mismatch 而根因存疑，回到第一步**收紧语料/血缘**（例如补 GEVM 的 plain-q8_K 布局对位），**仍是两步**，不是新一轮互搏。

---

## 4. 「定罪与翻案同等严谨」（对称性规则）

裁决对**两个方向施加同一 ZERO-MODEL 门槛**：

- **定罪**（判某 kernel 错）：须由一次满足 §2 全条件的 ZERO-MODEL 对位给出**定位到项**的 mismatch。
  笼统的"端到端 rel 大"不足以定罪到某一 fold（可能是 routing / 量化器 / 别的项）。
- **翻案 / 认 byte-exact**（判某 kernel 对、撤销先前定罪）：门槛**完全相同**——须由一次满足 §2 全条件的
  ZERO-MODEL 对位给出**全项 0-mismatch**。**不得**用"换了个更宽松的 oracle 现在过了"或"自比不变量成立"来翻案。
  （M2-fix 撤销的教训：不满足同源输入 + 独立参照的"过了"是**关于另一条路径**的，不能翻案。）

即：**认错和认对，证据规格一致**。任何一侧想改变现状（定罪或翻案），都要交出同一份 ZERO-MODEL 逐项对位。

---

## 5. 与「证书三要件」CI gate 的关系（rule ↔ gate 分工）

本 spec 是**站着的规则**；`cert-requirements-gate` 是**站着的 CI 守卫**。二者对应同三要件：

| 要件 | ZERO-MODEL 中的位置（rule） | cert-lineage 声明字段（gate 守） |
|---|---|---|
| ① 语料完备 | §2.5 每项非退化点火；未点火标 partial | `corpus_terms.{required,exercised}` + `fold_model_required_terms` |
| ② 输入同源 | §2.1/§2.3 取被测实际输入、参照 quantizer 同源 | `input_path_lineage.{tested_quantizer, oracle_quantizer, dispatch_faithful}` |
| ③ oracle 独立 | §2.3 参照零复用被测读取/折叠实现 | `oracle_independence ∈ {independent, self-compare, shared-impl}` |

**分工**：CI gate 在每次 PR **静态**拦截"声明层"的复发（空心证书冒充独立、narrow 掉 fold 必激活项、
claim=full 却跨-quantizer/未点火）；ZERO-MODEL 是**争议真起时**的**动态终审动作**——实际把项算出来对位。
gate 保证"没人再写出 M2c 那种声明"；ZERO-MODEL 保证"真要裁时，用对的方法一次裁死"。

---

## 6. 触发条件（何时默认走 ZERO-MODEL）

- 任何"某 fold/kernel/repack 到底 byte-exact 与否"的**争议**（定罪或翻案），**默认**走本法。
- 任何要把一个数值 cert 从 partial 升到 **claim=full** 的动作，其证据**必须**是一次 ZERO-MODEL 对位
  （否则 `cert-requirements-gate` 会因血缘不达三要件而 RED）。
- **不需要**走本法的场合：emit-golden lit（FileCheck 文本，结构上不触数值，见 min-fold 审计 §3 行 14）；
  纯**不变量**声明（保序 / 保 tiling 的自比，**只**能声明为不变量、**不得**声明为正确性）；
  性能/吞吐争议（那是相×板×格式探针轴，另一套规则）。

---

## 7. 未决对位清单（本法的第一批 owed 对位，供并行记账/修复线消费）

ZERO-MODEL 法一经采纳，下列**尚未做过合格逐项对位**者进入 owed 队列（判定来自 min-fold 审计 §4）：

- **q4_K repack GEVM**（plain-q8_K 布局）：从未经真-dispatch `dmin≠0` 逐项对位 → `pending-audit`，修复前不得认 byte-exact。
- **q5_K / q2_K repack GEMM+GEVM**：共享 `kquant_dmin_bsums_min` fold，`inferred-defective`；各须独立 ZERO-MODEL 对位。
- **block-dot 超块 `q4_k_min_term`**（标量 order-free，结构不同、风险低但未直测）：建议纳入未来 `dmin≠0` 直测清单。

> 这些条目的**实际逐项对位与 cert 补标**由并行记账/修复线执行；本 spec 只**确立方法与队列**，不代跑、不 commit。
