# 旧口径「83 / 28 逐格排除名单」→ 新单分母字典 映射说明（一页 · 纯案头 · B2-caliber-audit 交付项⑥）

> **交付定位**：欠账项——把 G8 阶段三前的「83→28 逐格排除名单」旧口径，改用**新主表行清单工件**
> `experiments/active/result-tables/T3_master_rowclue.txt`（snapshot `g8-master-rebuild-v2-single-denom`）
> 交付，并给「旧口径→新字典」一页对应。**数据零改**（仅口径正名）。
> **权威源**：新字典 = `T3_master_rowclue.txt` + `T3_master_rebuild.csv`（recon `recon_master_rebuild.py` 机算）。

---

## 一、旧口径都是什么（先把「83 / 28」这些数拆清）

阶段三前散落多个计数宇宙，互不同源，"逐格排除名单"就是在这些宇宙里挑格：

| 旧数 | 旧含义 | 来源 |
|---|---|---|
| **83** | perf-covered **系统账 e2e** 分母 = roster/M4 SINGLE-SOURCE 的 **(op,format,engine)** 结构分母 | `perf_covered_metrics.py`；9/83 头条 |
| **28** | **gemm roster 格数**（= 25 rvv-engine 格[含 q4_0-gemm decode+prefill 2 行] + 3 ime） | 主表 §4.2 |
| **46** | 「真 matmul 对局 kernel」= 24 vec_dot + 22 gemm（22 = 28 − 6 declared-exception gemm） | 二次整顿令 v2 报告 |
| **20/28** | matmul 每板 attacked/total（board-pt·板当计数主键） | 旧 census（作废） |
| **29/43** | rvv 侧 attacked/total（29/43 = matmul 20/28 + dequant 9/15） | 旧 census（作废） |
| **30/28**、**19+17** | 板×格混算的更早派生数 | 旧 census（作废） |
| **kernel-sym 12** | T9 ≥parity 的 **board-点（format×board）**计数（q4_1@rvv 与 q4_1@k1 计 2） | T9（board-pt 口径作废） |

**"逐格排除名单"的旧动作** = 从全 roster 里**逐格剔除**「非 matmul 对局」与「无对手声明例外」，把分母收窄到"真战斗面"（83→46→28→…）。**病根** = 板×格混算 + 多分母打架（`item 7/8/12` 已诊）。

---

## 二、新单分母字典（换掉旧"排除名单"的那把尺）

新口径**不再靠逐格排除收窄分母**，而是**单分母制 + 四档穷尽互斥**：每格永远留在分母里、只带一个档位 + 一个格内状态。

- **主表 = 91 行** = matmul 52（gemm 28 + vec_dot 24）+ forward 9 + dequant 24 + quantize 3 + product_reduce 3。
- **板分母**：rvv **85** / k1 **88**（= 91 − N/A-hw 3 − q1_0 域外 1；k1 无 N/A-hw 故 88）。
- **四档**（Σ = 板分母·机算✓）：手调 / 通用向量 / 标量类 / UNRESOLVED(0)。
- **格内状态**：PASS / 具名-X / 挂起 / pending。头条 = PASS / 该档全量行数。

**关键差异**：旧口径"排除"= 把格**移出分母**；新口径把同一批格**收进分母的某个档 + pending 状态**。所以旧"排除名单"里的绝大多数格，在新字典里**不是消失、是落档挂 pending**。

---

## 三、旧"排除名单"逐类 → 新字典落点（核心对应表）

| 旧口径里被"排除/收窄掉"的格类 | 计数 | 新单分母字典落点（rowclue 档 / 状态） | 说明 |
|---|---|---|---|
| **N/A-hw**（IME@rvv 3 格） | 3 | **不在板分母**（rvv 85 已扣） | 唯一仍"排除"的一类·机判 `ime.present`=False |
| **q1_0**（内部格式 vec_dot+gemm） | 1（域外） | **永久域外**（不入板分母·§〇.2） | Weft-internal·无第三方对手·仍排除 |
| dequant 24（除 q1_0 = 23） | 23 | **标量类**（rowclue rvv L56-78 / k1 L156-178）·多为 pending | 旧"DEQ 子账不进头条"废止→计入标量类全档 |
| quantize 3（q8_0/q8_1/q8_K） | 3 | **通用向量**（rowclue L35-37 / L136-138）·pending-真〔V-纠〕 | 旧误标 scalar→复核证手写 `__riscv_v` intrinsic |
| product_reduce 3 | 3 | **标量类**（rowclue L53-55 / L153-155）·pending | 标量参考兜底 sanity 层 |
| forward 9（softmax/rms_norm/silu/scale/rope/add/mul/cpy/gelu） | 9 | **通用向量 4 + 标量类 5**（rowclue L46-49/L91-95） | 独立桶·非 matmul·gelu 挂起 |
| declared-exception gemm（iq/tq/fp4 same-op repack absent） | 12 | **标量类**（rowclue gemm\|iq*/tq*/mxfp4/nvfp4@rvv·pending-真） | 旧从 28 剔的 6 + mxfp4 vec_dot + 扩展·今落标量仗待补 |
| FLAT gemm kernel-sym（q4_0 decode+prefill/q4_1/q5_0/q5_1/q8_0） | 6/板 | **通用向量**（rowclue L38-43/L139-144）·pending-fold | 别处有 cold（T9/onw2）·折入即可 |
| IME gemm（q4_0/q8_0/q4_K@ime） | 3 | **手调**（k1 列·rowclue L125-127）·pending | rvv 列 = N/A-hw·kernel-sym 未测 |
| iq/ternary vec_dot（iq2*/iq3*/iq4_xs/tq*/mxfp4） | 9-10 | **手调**〔清偿〕（rowclue L13-22/L110-119）·pending | 旧 UNRESOLVED·objdump 清偿落手调·待接线补 cold |

**校验**：旧"排除名单"的每一格都在新字典有唯一落点，无悬空。真正仍被排除的只剩 **N/A-hw 3 + q1_0 1**；其余全部**收进分母挂 pending**（这正是单分母制取代"逐格排除"的核心）。

---

## 四、旧头条数 → 新头条数 对应（换尺后不丢历史）

| 旧头条 | 新字典对应 | 是否作废 |
|---|---|---|
| perf-covered **9/83** | **保留**（系统账 e2e 独立赛道·非 kernel-sym·禁与四档互推） | 不作废（另轨） |
| certified **84/91** | **保留**（构造轴 byte-exact·非 0.8 轴） | 不作废（另轨） |
| **46**（真 matmul kernel） | matmul 52 行（gemm 28 + vec_dot 24）·22→全 28 gemm 收进分母（不再剔 6） | 换尺·并入单分母 |
| **28**（gemm roster） | 主表 gemm 28 行（内含 matmul 52） | 换尺·保留为分组数 |
| **20/28**、**29/43**、**30/28**、**19+17** | 手调/通用向量/标量类三档头条（rvv 5/20·11/21·13/44 ｜ k1 5/24·12/20·5/44） | **作废**（board×格混算·换 (op,format,engine) 主键） |
| **kernel-sym 12**（board-pt） | 三档 PASS 分档计（board 降为属性列·不作计数主键） | **作废**（board-pt 无正当性） |

---

## 五、一句话结论

「83→28 逐格排除名单」= 旧"多分母 + 板×格混算 + 逐格剔除收窄战斗面"的产物；新**单分母制**把同一批格**全部收进 rvv85/k188 分母**、每格只挂一个档 + pending 状态，**唯一仍排除的是 N/A-hw 3 + q1_0 1**。逐格对应见 `T3_master_rowclue.txt`（每档附机打行清单）。perf-covered 9/83 与 certified 84/91 是**独立赛道·不被本换尺替代**。

*证据：`experiments/active/result-tables/T3_master_rowclue.txt`（rowclue 机算）· `T3_master_rebuild.csv` · `recon_master_rebuild.py`（snapshot g8-master-rebuild-v2-single-denom）· 旧口径散见 `docs/reports/2026-07-15-G8-全景报告-v2.md` item 7/8/12 + `docs/reports/2026-07-15-G8-全景对账报告-kernel-unit.md`。*
