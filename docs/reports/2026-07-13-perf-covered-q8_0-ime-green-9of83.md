# perf-covered 9/83 — q8_0@ime 绿格登记（★最强 IME 绿·真 beat stock 2.233×·无内存税·G6-A M7 铺满同族）

> 登记 HEAD = `9acb6f3b`（M7 q80/q4k 板测提交）· 生成 2026-07-13 · 触碰集 = docs + schema（label·recon 机算）。
> **口径**：承 `2026-07-13-perf-covered-q4_0-ime-green-8of83.md`（any-board·recon 禁手填）。**三处同源由 `perf_covered_metrics.py` 机算**（本报告 = 登记册增量·8/83→**9/83**·recon 落定 `绿9`·snapshot 9acb6f3b）。
> **★本报告 = M7 wide-vmadot-tiling 杠杆铺到 IME 三格剩二的【分裂结果】**：q8_0@ime 转绿（本报告）· q4_K@ime 维持黄（array-util format-keyed 边界·见 §5）。

---

## 0. 登记结论

**perf-covered = 9 / 83 = 10.84%**（recon 机算·`绿9`·`reconciliation_ok/three_source_consistent/anti_gate_ok=True`·黄-传导稀释=1[仅剩 q4_K@ime]）。第 9 格 = `gemm_tile/q8_0/ime`（category 迁移 `黄-传导稀释`→`绿`·any-board：k1 prefill 传导）。

**★verdict（M7 预注册出口 (a) 兑现·权限卡⑦·同 q4_0@ime 先例但更强）**：`k1·q8_0@ime·prefill·kernel账-ime int8-direct·对手 stock 非-IME block-dot·**e2e prefill onw2 2.233× stock（★真 beat·无 IQR 重叠·非 tie）· 赢 vendor 2.269×**·经 [PAT-1] wide-vmadot-tiling W2·clang-18 对称`。

**★★成色定性（诚实·四条硬披露）**：q8_0@ime **beat stock 真赢（非 tie·区别 q4_0@ime）+ beat 弱 vendor**，成色**强于 q4_0@ime**——但**所赢 vendor 的 q8_0 IME 路弱（0.984× stock·非强手调）**，故是"我方产出最快可得 q8_0 IME 核"非"赢强手调竞品"。仍**非**"成排赢强手调"的完整 G6 成色质变。见 §4。

---

## 1. 为何转绿（真 beat·区别 q4_0@ime tie + q4_K@ime <parity）

- **前态 = 黄-传导稀释**（[裁一.3]·M3 raw agg_q8_0 prefill 0.166×·correctness-first 桥未消包袱）。
- **M7 全桥板测（M1-M7·含 wide-vmadot-tiling W2）**：`onw2 = 23.74 t/s · onw2/off = 2.233×（真碾压·无 IQR 重叠）· onw2/ven = 2.269×（赢 vendor）· onw2/onjout = 1.086×（array-util 增量）`。
- **array-util 传导（format-invariant leaf·format-keyed 传导）**：compute-isolation ~1.56-1.59×（**同 q4_0/q4_K·wide-vmadot-tiling 是 format-agnostic MAC**），但 **full-matmul 传导分裂**：q8_0 **1.106×** vs q4_K 1.045×——q8_0 int8-direct **轻 epilogue** 让 array-util 增量传导到 e2e·beat stock（IME 硬件 2.233× vs 非-IME block-dot）。
- **对比 q4_0@ime**（tie stock·lose vendor 2×）：q8_0@ime **beat stock 真赢 + beat vendor**·且**无内存税**（int8-direct·deref 不膨胀·区别 q4_0 ×1.78）→ **成色更干净更强**。

## 2. 证据（casefile `experiments/active/g6-a-ime-perf-bridge/M7-vmadot-tiling-q80-q4k/`）

- **部署五验**：build-clean 0-err·banner routes real q8_0（M=32 N=2048 K=2048·tilew=2）·vmadot count 51（baseline 32）·反向控制 off/ven banner=0·load-gate extern_jobs=0·1.6GHz 锁频·-t4·pin0-3。
- **correctness GREEN 三重证**（byte-exact 硬门·核 0xe210312b vmadot 逐字不动）：
  1. **wide-neutral 硬门**：板 `onw2==onjout` IDENTICAL（wide sumi[0..15]/[16..31] 各 bit-identical width-1·epilogue 共享→由构造）。
  2. **板 e2e greedy md5**：`onjout==onw2==onw4==off==ven==5b3fa260`（全同·**含 vendor**·byte-identical）。
  3. **wide-tiling leaf int32-exact**：已由 q4_0 M7 host ZERO-MODEL oracle 证（`test/Target/IME/q4-0-vmadot-tile-wide-int32-oracle.c`·wide==width-1==plain-GEMM·同 leaf 复用·format-agnostic MAC）。
- **分相 e2e**（12 samples/side·k1·clang-18 对称·1.6GHz·-t4·relIQR<2%·interleaved）：

| side | pp32 t/s | vs stock |
|---|---|---|
| off (stock 非-IME block-dot) | ~10.63 | 1.0× |
| ven (vendor IME·q8_0 路弱) | ~10.46 | 0.984× |
| onjout (M6) | — | 2.057× |
| **onw2 (M7 wide W2)** | **23.74** | **2.233× ≥parity 真 beat** |

- **A-tree restore**：EXIT-trap + 独立第二证·`ime.cpp==40962c7e·so==71cc4d29==baseline`·markers=0·litter=0·stray=0·vendor 树 byte-exact。

## 3. 八门 + 双账本（诚实·②⑤ single-board caveat）

- **八门[prefill]**：①PASS（三重 correctness·md5 含 vendor 全同）②**PARTIAL**（e2e k1-only）③PASS（真 vmadot·count 51·核字节不变）④PASS（array-util micro 1.56×∧e2e 2.233×·onw2/onjout 1.086×）⑤**PARTIAL**（single-board）⑥PASS（DVFS/pin/paired/load-gate）⑦PASS（compute-transduction·int8-direct 轻 epilogue）⑧PASS（bounded k1/VLEN256/clang-18/prefill·无内存税）。**NOT sealed universal 8-gate**（②⑤ single-board）。
- **双账本**：k1 clang-18·kernel==system（[CASE-COMPILER-ASYMMETRY] not triggered）。
- **对手**：opponent-of-record = **stock 非-IME block-dot**（真 beat 2.233×·非 SELF）+ vendor IME（弱 q8_0 路·2.269× 赢·披露弱）。

## 4. ★★成色定性（诚实·四条硬披露·G6 成色质变判读）

1. **对手身份 = stock 非-IME block-dot·真碾压 2.233×**（**无 IQR 重叠·非 tie**·区别 q4_0@ime tie-stock）。
2. **★★诚实前提 = 所赢 vendor 的 q8_0 IME 路【弱】**（VEN/OFF=0.984× stock·vendor 未优化 q8_0 IME）——故本格 = "**我方产出最快可得 q8_0 IME 核**（beat stock 非-IME 2.233× + beat 弱 vendor 2.269×）"·**非**"赢强手调竞品"（无强手调 q8_0 IME 对手存在）。
3. **★无内存税**（q8_0 int8-direct·deref cache 不膨胀·区别 q4_0 int8-deref ×1.78·成色更干净）。
4. **★G6 成色质变（硬碰硬赢手调对手成排）判读**：本格 beat stock 真赢（非 tie）+ beat 弱 vendor·成色**强于 q4_0@ime**（real beat vs tie/lose）·是**最强 IME 绿 + 朝成色质变一步**·**但仍非"成排赢强手调"**（单格·vendor q8_0 路弱非强手调·非"成排"）。绿之**实质 = IME 硬件优势（2.233× vs 非-IME block-dot）经 [PAT-1] wide-vmadot-tiling 发射器杠杆传导**·**不得表述为"碾压强 IME 竞品"**。

## 5. sibling 分裂：q4_K@ime 维持黄（array-util format-keyed 边界·C3′ 负结果）

同 M7 板测的 **q4_K@ime = 0.909× <parity（decisive·维持黄-传导稀释·现 measured 非 projected）**。根因 = **array-util 杠杆的 format-keyed 适用边界**：q4_K two-level fold（scale+min）**重 non-vmadot epilogue** 把 array-util 增量（compute-isolation ~1.56× 同 q8_0·format-invariant）稀释到 e2e 仅 1.024×；stock q4_K 已快（14.61 t/s）·vendor 强（1.493×）。**这是 C3′ 负结果（同价值）**：wide-vmadot-tiling 对 int8-direct 轻 epilogue 格（q8_0）传导转绿、对 super-block 重 fold 格（q4_K）被 epilogue 稀释不足——**能力键控优化模式的 format-keyed 适用边界实证**。非物理墙（vendor 1.493× 证上限）·是更深 epilogue 发射器工程目标（M7 深度外·机制目标非 warranted perf battle）。

## 6. recon 落定

```
perf-covered = 9/83 (10.84%·any-board·fold 备选)
classification: 绿9 · 黄-未接线0 · 黄-传导稀释1(q4_K@ime) · 黄-物理墙12 · 黄-对手更强12 · 声明例外49 · Σ83
reconciliation_ok=True · three_source_consistent=True (headline=classification=ledger=9) · anti_gate_ok=True · all_exceptions_certified=True
snapshot commit = 9acb6f3b
```
