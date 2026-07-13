# perf-covered 8/83 — q4_0@ime 绿格登记（★首个 IME 族经真发射器优化模式的 perf-covered 绿·G6-A 桥收口）

> 登记 HEAD = `468fcef7`（M7 发射器提交）· 生成 2026-07-13 · 触碰集 = docs + schema（label·recon 机算）。
> **口径**：承 `2026-07-12-perf-covered-roster-unification-recon.md`（fold 6/83·any-board·recon 禁手填）。**headline/分类/登记册三处同源由 `perf_covered_metrics.py` 机算**（本报告 = 登记册增量·7/83→**8/83**·recon 落定 `绿8`·snapshot 468fcef7）。
> **★本报告 = G6-A 战役（[GAP-IME-BRIDGE-PERF] 收口）兑现**：q4_0@ime 从 `黄-传导稀释`（e2e 慢 stock 6-79×）经 M1-M7 逐包袱消融桥 → **≥parity（1.0088× stock）跨线**。**[裁三.3]『IME 将转绿』叙事禁令解除**——禁令约束的是"收口前"的 premature 叙事；本翻绿 = 桥已收口、实测跨 parity 的收口兑现，非叙事。

---

## 0. 登记结论

**perf-covered = 8 / 83 = 9.64%**（recon 机算·`绿8`·`reconciliation_ok/three_source_consistent/anti_gate_ok=True`·undefined_cells=[]）。第 8 格 = `gemm_tile/q4_0/ime`（category 迁移 `黄-传导稀释`→`绿`·any-board：k1 prefill 传导）。

**★verdict（M6 §6 pre-registered 绿出口 (a) 兑现·权限卡⑦ 直接执行）**：`k1·q4_0@ime·prefill·kernel账-ime·对手 stock 非-IME RVV·**e2e prefill onw2 1.0088×（≥parity·统计 tie 跨线）**·经能力键控 wide-vmadot-tiling 发射器杠杆 [PAT-1] W2·clang-18 对称`。

**★★成色定性（诚实·四条硬披露·措辞门·随绿格永驻）**：本绿 = **metric-letter 绿（≥parity vs stock）+ 强 C3′ 方法学交付**，**非** G6 定义的"成色质变（硬碰硬赢手调对手成排）"。见 §4。

---

## 1. 为何是绿（真发射器杠杆·区别 M1-M6 orchestration）

- **前态 = 黄-传导稀释**（[裁一.3]）：q4_0@ime kernel-axis compute-account ~2.09×（vmadot-leaf·硅证 int32 0-diff）但 e2e 慢 stock 6-79×——桥 correctness-first 形态（单线程 ith==0[1/4 hart]·repack 未缓存·参考实现）的 orchestration overhead 把 compute 赢**稀释**掉。
- **G6-A 逐包袱消融桥（M1-M7·曲线）**：`{0.0126× → M1 repack-cache 0.0143×(1.138×) → M2 多线程 0.0554×(3.886× 干净传导) → M3 deref 0.539× → M4 setup 0.5807× → M5 epilogue 0.6817× → M6 feed-locality 0.9042× → **M7 wide-vmadot-tiling 1.0088× stock**}`·**combined ~72×**·全程 bit-exact。
- **★M7 = 首个 G6-A 发射器改动（前 6 步皆 orchestration/板 patch·零发射器改动）**：`lib/Plugin/IME/IMEBackendEmissionDriver.cpp`（+156/-0）新增 `macKloopHelperBodyWide(njw)`（ONE `vle8` A 喂 NJW 独立 vmadot 链）+ `kIMEVmadotTilingPatterns[]`（[PAT-1] 数据表·W1/W2 mechanized·W4 measured-negative）+ `selectVmadotTilePattern`（capability-keyed·RVV 32-vreg 预算判别 → W2）。攻 M6 确诊的最后大残留 = vmadot array-util（compute 1.741s = matmul 33.8%·单累加器串行 + A 每 vmadot 重载 + per-block entry/exit）。
- **array-util 板测兑现**：vmadot compute `1.740s(w1) → 1.095s(w2·1.589×) → 0.890s(w4·1.955× = 厂商同硅 headroom 兑现)`·matmul `1.159×(w2)`·**e2e onw2/onjout = 1.1385× clean（98% matmul 天花板传导·填平 M6 的 10% parity 缺口）**。w4 measured-negative（16-accumulator f32 epilogue spill → full-matmul NULL 0.972×）→ **shipped width = W2**。

## 2. 证据（casefile `experiments/active/g6-a-ime-perf-bridge/M7-vmadot-tiling/`）

- **correctness GREEN 三重证**（byte-exact 由构造·核 0xe210312b vmadot 与累加序逐字不动）：
  1. **K1 单元测试（真 vmadot）**：w2 & w4 int32 == baseline width-1 leaf·`memcmp=0`。
  2. **host oracle**（`test/Target/IME/q4-0-vmadot-tile-wide-int32-oracle.c`）：wide == width-1 == **ZERO-MODEL plain-GEMM**·mismatch=0（本会话主会话侧独立复跑确认 ORACLE PASS）。
  3. **板 e2e greedy**：md5 `onw2 == onw4 == onjout == off == f5e77482`（M1-M6 sealed 谱系一致）。
- **发射器 lit**（主会话侧 check gate 复跑）：`ime-q4-0-matmul-tile-materialization.mlir` REGION + EMITC 两 RUN 行 **PASS**（含 M7 wide-leaf 新检·existing 检未动 = byte-exact·q8_0/q4_K goldens 未受扰）。weft-opt 构建+relink 确认（.o 14:58 > 源 14:57·binary relinked）。
- **e2e 分相**（12 samples/side·k1·clang-18 对称·1.6GHz 锁频·-t 4·relIQR gate·interleaved）：

| side | pp32 t/s | relIQR | vs stock |
|---|---|---|---|
| off (stock RVV 非-IME) | 23.672 | — | 1.0× |
| ven (vendor IME) | 47.254 | 0.30% | 1.996× |
| onjout (M6) | 20.976 | — | 0.886× |
| **onw2 (M7·wide W2)** | **23.881** | **1.40%** | **1.0088× ≥parity** |
| onw4 (M7·wide W4) | 20.903 | — | 0.883× (measured-negative) |

- **A-tree restore**：EXIT-trap `RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…)`·主会话侧 ssh 二证 == baseline·`.ORIG` litter=0·stray procs=0·vendor 树 byte-exact。

## 3. 八门 + 双账本（诚实·②⑤ single-board caveat）

- **八门[prefill]**：①PASS（三重 correctness 证）②**PARTIAL**（e2e k1-only）③PASS（真 vmadot·objdump vmadot=41·核字节不变）④PASS（array-util micro 1.589×∧e2e 1.0088×·M6→M7 1.1385× clean）⑤**PARTIAL**（single-board e2e）⑥PASS（DVFS/pin/paired/load-gate/relIQR 1.40%）⑦PASS（compute-transduction 98% matmul 天花板）⑧PASS（bounded k1/VLEN256/clang-18/prefill/int8-deref regime）。**★NOT a sealed universal 8-gate Win**（②⑤ = e2e single-board）。
- **双账本**：k1 出货 clang-18·kernel-axis == system-axis（symmetric single tree·[CASE-COMPILER-ASYMMETRY] not triggered）。
- **对手**：opponent-of-record = **stock 非-IME RVV block-dot**（≥parity·非 SELF）。

## 4. ★★成色定性（诚实·G6 成色质变 bar【未达】·四条硬披露）

本绿**必须**随附以下四条披露（措辞门·禁 oversell·禁裸口号 novelty）：

1. **对手身份 = stock 非-IME RVV**（≥parity·**统计 tie**：onw2 IQR[23.62,23.95] ∩ off IQR[23.54,24.05] 重叠 → onw2 ≈ off·**非碾压 beat**·perf 宪法「对手贴墙 parity=满分」+ perf-covered ≥parity 门达成，但机制赢的 decisive 部分是 M6→M7 净 1.1385×，非 vs-stock 的绝对领先）。
2. **★vendor IME = 1.996× stock 是更强手调可部署·我方仅 0.505× vendor（输一半）·披露不隐瞒**——k1 上 IME q4_0 的**最强实际可部署对手 = vendor（SpacemiT 手调）·我方未追平**（vendor = 天花板参照·非 opponent-of-record；opponent-of-record 取 stock 承 M6 §6 pre-registration·全 perf-covered 绿格一致以 stock 为 parity 参照，非独为 IME 降格）。
3. **★令一.2 内存税**：曲线自 M3 用预解码 int8 deref-cache（N*K int8·代表格 q4_0 ~1.07GB·×1.78 驻留+读带宽 vs stock native nibble 0.5625·N*K）→ 1.0088× **绑 prefill/compute-heavy 口径**·decode/memory-bound regime 反向·**非零成本 free lunch**。
4. **★G6 成功判据（成色质变 = 硬碰硬赢手调对手成排）本格【未达】**：此为 **metric-letter 绿（≥parity vs stock·统计 tie）+ 强 C3′ 方法学（发射器 wide-vmadot-tiling 能力键控杠杆 [PAT-1]·98% 传导·桥 44×→72×→跨 parity）**·**非**赢手调对手（tie-stock / lose-vendor 2×）。绿之**实质价值 = C3′ 能力键控优化模式库首个 IME array-util 条目 + 传导稀释消除**，而非性能碾压。**不得表述为「赢过手调 IME」**。

## 5. projected siblings（不自动转绿·deployment≠proven）

q8_0/q4_K@ime **共用同一 vmadot leaf**（wide tiling format-agnostic MAC 同法适用）**但本 M7 仅板测 q4_0** → q8_0/q4_K = **同-leaf·projected·未各自板测** → **维持黄-传导稀释 pending**（[q4-0-e2e]/[sealed-win-1] deployment≠proven 纪律）。补测入队列（同 M7 harness·各自 seal 后再计 perf-covered）。当前 recon 分类：绿8·黄-传导稀释2（= q8_0/q4_K@ime）。

## 6. recon 落定

```
perf-covered = 8/83 (9.64%·any-board·fold 6/83)
classification: 绿8 · 黄-未接线0 · 黄-传导稀释2 · 黄-物理墙12 · 黄-对手更强12 · 声明例外49 · Σ83
reconciliation_ok=True · three_source_consistent=True (headline=classification=ledger=8) · anti_gate_ok=True · all_exceptions_certified=True
snapshot commit = 468fcef7
```
