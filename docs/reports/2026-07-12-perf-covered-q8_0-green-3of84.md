# perf-covered 3/84 — q8_0 full-stack correctness-carrier 绿格登记（用户追认 · G5-M1b）

> 登记 HEAD = `55022402`（G5-M1b R1 HIT casefile）· 生成 2026-07-12 · 触碰集 = docs 域（登记 + SOP + [GAP] 立卷）。
> **口径**（承 `2026-07-11-perf-covered-baseline.md`·不改口径）：`perf-covered = 已构造格中经公平协议（八门 + 双账本 + 对手探针）测得 ≥parity/赢 的格数 / 已构造格数(84 certified)`。
> **本报告 = 登记册增量**：2/84 → **3/84**（+q8_0）。用户裁决 2026-07-12 追认 M1b 战果正式入册。数字住报告 + ROADMAP，不写进 spec。

---

## 0. 登记结论

**perf-covered = 3 / 84 = 3.57%**。分子三格（逐格披露账本 + 八门 + 登记性质）：

| # | 格 | 登记性质 | 账本 | 八门 | 数据 | 依据 |
|---|---|---|---|---|---|---|
| 1 | `gemm_tile/q4_K` | **kernel-account sealed Win** | kernel（编译器对称） | full 八门 | k1/VLEN256 e2e prefill 1.085× vs 真出货 hand-brick·byte-exact | Win-K1-VLEN RATIFIED |
| 2 | `gemm_tile/q4_0` | **routing 绿格**（★带注记：上游本有可用路径） | 系统账 | 5/8 门·disclosed | e2e prefill 5.92× 系统账 routing 赢 | q4_0 sealed·[NG-4] 措辞 LOCKED |
| 3 | **`gemm_tile/q8_0`** | **★full-stack correctness-carrier 绿格（无星号）** | 系统账 | 八门（selector 状态见 §2） | e2e **prefill 4.350× / decode 3.812×**（CI 排除 parity）·correctness GREEN·部署五验+反汇编 vl=8 | G5-M1b `55022402` R1 HIT |

**★q8_0 的登记性质与 q4_0 的本质区别（用户裁决 2026-07-12·〇.1）**：q4_0 = routing 绿格（**上游本有可用 kernel**，我方翻 gate 路由·correctness 白送·带"上游本有路径"注记）；q8_0 = **full-stack correctness-carrier 绿格**——完整穿透链条无外援：**能力事实 → 发射正确 vl=8 变体 → 部署 → 真路由 → 正确性修复（上游 VLEN128 破损）→ e2e 兑现**。论文价值 = **C1 + C3′ 的无星号性能实证**（不背 q4_0 的注记）。此为 **G5 决定性曳光弹**。

---

## 1. q8_0 证据链（M1b `55022402`·casefile `experiments/active/g5-wiring/M1b-q8_0/`）

- **能力事实 → 正确变体**：q8_0 repack GEVM+GEMM 经同一 q4_0 front door 构造（typed_repack_gem{v,m}_loop_body·fullI8 core·d-only fold）·emit .inc md5 `b5177a4d…`·**54 个 AVL 常量全 = 8**（vl=8 VLEN128-correct）。
- **部署 + 真路由**：deploy_patch 挂点③（gate flip + `#include` emitted kernel + VLEN128 intercept 分支拦截破损上游）·nm/objdump seal 证两符号 `vsetivli zero,8,e32,m2`（vl=8·零 vl=16·部署==证过）·banner GEVM×8+GEMM×8 FIRE。
- **正确性修复**：greedy A==B byte-identical 三 prompt coherent（"The capital of France is **Paris.**"）vs M1 破损上游 "olta" garbage。**CORRECTNESS_GATE GREEN**。
- **e2e 兑现**：prefill 4.350×（CI[4.343,4.355]）/ decode 3.812×（CI[3.801,3.818]）双 DIFFERENCE·DVFS 锁 2.6GHz·IQR≤0.39%·同树 .so swap gcc-15.2.0 对称。
- **mirage 锚**：M1 破损 vl=16 跳半列 prefill 78.11× MIRAGE → M1b 正确 vl=8 做全工 39.191 ≈ 半 = 全量正确工确证。

---

## 2. ✅ 张力 A LANDED → deployed=proven（selector-自然路由·2026-07-12·commit 40ac20ca）

M1b 的路由曾是**能力键控构造强制 repack**（front-door `block_dot_compute_heavy=true`）——但那是 provenance 记为 structurally-FALSE 的谎（q8_0 的诚实事实 compute_heavy=false）。**load-bearing 发现**：q8_0 **无** block-dot decline 路（identity lowering 仅 q4_0-nibble）→ 诚实事实下 selector DECLINE → lowering ERROR 即便 VLEN128 → 这正是 M1b 必须强制 compute_heavy 的原因。

**张力 A 修法（LANDED·40ac20ca）**：加 `block_dot_memory_bound`（OptionalAttr<BoolAttr>·`block_dot_compute_heavy` 的 **roofline dual**）——q8_0 结构事实（最宽线性量化 stride34 + lean vec_dot → DRAM 带宽受限 → repack x16 连续流除冗余内存流量）。benefit = `compute_heavy || memory_bound` 双 roofline arm。**能力键控做主键·非 measured-guard**（fact 从 block 字节布局 + ggml vec_dot roofline 结构读出·quants.c:435·M1b 数字只 CONFIRM 不 establish·reason=capability 类）。

**★deployed=proven 达成**：default-compile q8_0（诚实事实·无强制）→ selector **自然**选 repack（reason `repack-kept-q8_0-memory-bound-vlen128-decode`）→ emitted .inc **字节等于** M1b `b5177a4d` → M1b board 4.35×/3.81× **传导·无需重测板** → **部署变体 == 证过变体**（closes deployed≠proven caveat）。over-flip audit：q8_0 唯一 flip（新 attr 默认 absent）·余格 byte-identical·ZERO-MODEL exact·attribution=capability·lit 904/907（3 失败 pre-existing 无关）。**八门⑥ selector-routing 从"强制探针"升"自然路由"**。

> **sealed-pin debt（诚实记录）**：`block_dot_memory_bound` 的 provenance 锚（quants.c:435）本应进 opponent-facts 审计 pin（对称既有 `block_dot_compute_heavy` 锚），但该 pin 住 `experiments/sealed/`·禁 casual 改（sealed immutability canon）→ pin-registry 扩展 **revert·记为 debt**（provenance 暂存 selector 注释·日后经 sanctioned STALE-flip 补锚）。opponent-pin 门只验 pin_sha·未破。

---

## 3. ★[GAP-Q8_0-VLEN128-KERNEL] 立卷（用户裁决 〇.2·T8/T5d 类·外部证据·C1 供弹）

**观测**：上游 ggml `arch/riscv/repack.cpp` 的 q8_0 VLEN256-硬编码 repack kernel（hardcode AVL=16）在 **VLEN128 上破损**（vl 钳 8 → 16-way 交织只算 8 列 → garbage）。上游 `case128 = //TODO` 从未实现。

**意义（对手侧铁证·给 C1 叙事直接供弹）**：**手写库的 VLEN 不可移植性**——上游一份**死宽度** kernel 跨板即错（VLEN256 写死→VLEN128 garbage），正是我方"**能力驱动发射**"论点的**对手侧铁证**：我方按 `vlen` 能力事实发射**正确变体**（vl=8 VLEN128-correct），能力驱动 ≠ 死宽度手写。此条 = **模板协议本体（C1）** 的外部证据（对手做不到的可移植性 = 模板价值证词）。立卷 T8（案例库）+ T5d（厂商/上游路径对照）。

---

## 4. ★MIRAGE 教训固化 → G5 接线 SOP（用户裁决 〇.3）

**教训**（G5-M1 曳光弹 f8b8dabb）：翻 gate 得 8.69× 高数 = MIRAGE（破损上游跳半列做少工的假快）·output garbage。

**SOP（写进 G5 接线流程·所有接线格强制）**：
1. **correctness-first 铁律**：接线的 **perf 数字在正确性门（greedy A==B / byte-exact）绿之前一律不入台账**。
2. **破损上游默认怀疑**：翻 gate 得异常高数（>已知 kernel 优势倍数）→ **默认怀疑上游 VLEN 破损**（死宽度硬编码），查 objdump vl + output coherence 再论。
3. **部署五验**：nm 符号 + objdump vl seal + banner FIRE + A-tree restore byte-exact + 零 stock 永久改动。
4. **每接线格套用**（L-接线②③ 尤其·防第二个 q8_0-MIRAGE）。

---

## 5. 后续（登记册与 ROADMAP 同步）

- ROADMAP 头条 perf-covered 2/84 → **3/84**（本报告为登记依据·q8_0 无星号绿格）。
- 下一格候选：**q4_K（L-接线② 曳光弹·在飞）** → 若绿 = 4/84；q5_0/q5_1 复用 q4_K scaffold → 潜在 6/84。
- 关联：`2026-07-11-perf-covered-baseline.md`（基线口径）· `SEALED-WIN-REGISTRY.md`（q8_0 = sealed-Win 候选·待张力 A selector-自然确认后评估入册）· memory `q4-0-e2e-is-routing-not-kernel`（correctness-carrier=q8_0 非 q4_0）· `measurement-offensive-perf-covered`。
