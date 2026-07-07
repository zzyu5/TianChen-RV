# [GAP-GRID] grid-decode 侦察画像

- **状态**: ★ 侦察态(SCOUT-only)—— **未立项 / 待用户裁 / 未实现 / 零代码改动**。本文只读了 `lib/`、`include/`、
  board 反汇编与 ODS,只写这一份 `.md`。所有"杠杆"仅为枚举 + 粗判,**不是方案、不是承诺、不是数字**。
- **裁决锚**: 裁决四 [GAP-GRID] 侦察(画像,不立项)。
- **日期**: 2026-07-07
- **代码 HEAD**: `0ad9cf6d`(只读;未 commit;由用户提交)
- **上游事实来源(数据 cell,均已存在)**:
  - `experiments/active/format-micro-rvv-vlen128/gapsb_family_validate/raw/board_medians.txt`(板中位数 A/B + vsG + roofline)
  - `experiments/active/format-micro-rvv-vlen128/gapsb_family_validate/raw/objdump_counts.csv`(逐助记词 PRE/POST 计数)
  - `experiments/active/format-micro-rvv-vlen128/gapsb_family_validate/objdump_pre_post.md`([GAP-SB] PRE→POST 机器码表)
  - `experiments/active/format-micro-rvv-vlen128/gapsb_family_validate/raw/disasm_iq3_{s,xxs}_POST.txt`(我方导出 .o 反汇编)
  - `experiments/active/format-micro-rvv-vlen128/gap_triage_batch2c/disasm/iq3_{s,xxs}.factory_vl128.objdump.txt`(厂商反汇编)
  - `lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`(grid decode emitter,只读)
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:7357/8138/8388`(iq3_xxs / iq2_xs / iq3_s 语义 ODS)

---

## 0. 一句话结论(先给,细节在下)

★ 侦察态,未立项:**board 反汇编推翻了"512-entry 查表(gather)主导关键路径"这个前提**。
[GAP-SB] 已把 grid **gather 本身**打到极小(iq3_s/iq3_xxs `vluxei16` = **4 条/kernel**,AVL=2 storm 46→0)。
POST 关键路径真正的主导 = **喂给 gather 的"逐 lane 变量移位 索引/符号组装",被我方 emitter 标量化 + 栈回程(stack round-trip)**——
iq3_s = 128 条 `sh` + 108 条 `lbu` + 142 条 `slli` 标量风暴;iq3_xxs = 120 `lbu` + 118 `slli` + 56 `sh`。
gather(`vluxei16`)在总指令里 **≈0.3%**。因此"表驻留寄存器 / vrgather 分段 / 子表拆分"这三个候选打的是 **不主导的 gather**;
**最高杠杆候选 = 索引/符号的 in-vector 组装(照厂商:`vsrl.vv`/`vsll.vi`/`vand`/`vor` 建索引,免栈回程)**,预计影响 **2 格确证(iq3_s、iq3_xxs)+ 2 格同族未测(iq1_s、iq1_m)**,空间成本 **≈0**(不新增表、不改权重布局)。

---

## 1. 反汇编占比 + 关键路径角色

### 1.1 板测锚(为什么 iq3 是 grid-decode-bound 的候选)

来自 `board_medians.txt`(rvv VLEN128 core8,cache-cold,byte-exact ULP0,PREFLIGHT 5/5):

| fmt | PRE.ours | POST.ours | A/B | factory(generic) | vsG.POST | flip | POST GB/s(占 roofline) |
|-----|---------:|----------:|----:|-----------------:|---------:|------|------------------------:|
| iq3_s   | 1570.2 | 1220.1 | 1.29× | 682.8 | **0.56× LOSS** | LOSS→LOSS | 0.330(5.2%) |
| iq3_xxs | 1949.3 | 1612.3 | 1.21× | 618.1 | **0.38× LOSS** | LOSS→LOSS | 0.243(3.8%) |
| iq2_s   |  376.3 |  344.0 | 1.09× | 749.9 | 2.18× WIN | WIN→WIN | 1.084(17.1%) |
| iq2_xs  |  352.9 |  319.1 | 1.11× | 481.4 | 1.51× WIN | WIN→WIN | 1.151(18.1%) |

- [GAP-SB] 让 4 格都变快(A/B 1.09–1.29×),但 iq3 两格仍 **LOSS**,且只跑到 roofline 的 3.8–5.2%(latency/compute-bound,非带宽 bound)。
- iq2 两格本就 WIN。**同批里 iq2_s 用的是最大的表(1024 entry / 8192 B)却是最快的 WIN(2.18×)——已足以说明"表大小"不是判据。**

### 1.2 POST 反汇编指令直方图(我方导出 .o,`disasm_*_POST.txt`,`grep`+`uniq -c`)

**iq3_s POST(共 1183 条指令)**——标量组装风暴,向量 gather 极小:

| 类别 | 助记词 | 计数 | 占比 |
|------|--------|-----:|-----:|
| 标量组装 | `slli` | 142 | 12.0% |
| 标量组装(**栈回程**) | `sh`(store-halfword 到栈上索引数组) | 128 | 10.8% |
| 标量组装(读打包字节) | `lbu` | 108 | 9.1% |
| 标量组装 | `addi`/`or`/`andi`/`add`/`mul` | 81/64/60/43/35 | 合计 ~24% |
| **GEARBOX(真算术)** | `vwredsum.vs` / `vwmul.vv` | 32 / 32 | 各 2.7% |
| 符号折叠 | `vrsub.vi`/`vmv.v.x`/`vmsne.vi`/`vand.vv` | 32 各 | — |
| **grid GATHER** | **`vluxei16.v`** | **4** | **0.34%** |
| grid 索引回读 | `vle16.v` | 4 | 0.34% |
| 配置 | `vsetvli`/`csrr` | 38 / 36 | — |

**iq3_xxs POST(共 1097 条指令)**——同型标量风暴 + `sw`(32):

| 类别 | 助记词 | 计数 |
|------|--------|-----:|
| 标量组装 | `lbu` 120 / `slli` 118 / `addi` 82 / `add` 83 / `sh` 56 / `sw` 32 / `srli` 33 / `mul` 29 / `or` 24 |
| **GEARBOX** | `vwredsum.vs` 32 / `vwmul.vv` 32 |
| **grid GATHER** | **`vluxei16.v` 4** / `vle16.v` 4 / `vse16.v` 4 / `vzext.vf2` 4 / `vsll.vi` 4 |

**对照 WIN 格(无标量风暴):**
- **iq2_s POST(418 条)**:top 无任何 `sh`;`addi`52 / `add`29 / `vsetivli`27 / `mul`23 / `vwredsum`16 / `vwmul`16 / `vle8`16 / `slli`16 / **`vzext.vf2`12**。
- **iq2_xs POST(354 条)**:top 无任何 `sh`;`addi`38 / `add`25 / `mul`21 / `vwredsum`16 / `vwmul`16 / `slli`15 / **`vsll.vi`12 / `vzext.vf2`8**。

### 1.3 关键路径角色判定

- **gather 不是主导**:iq3_s `vluxei16`=4 条(0.34%),iq3_xxs 同。[GAP-SB] 已把 32→4 gather + AVL=2 storm 46→0
  (`objdump_pre_post.md`),GEARBOX(vwmul/vwredsum 32/32)PRE→POST **未变 = byte-exact 保号**。
- **主导 = 标量"逐 lane 变量移位 索引/符号组装" + 栈回程**。反汇编实证(`disasm_iq3_s_POST.txt:310-413`):emitter 用 GPR 逐元素算索引,
  再 `sh` 存进栈上 `gridoff[]` 数组(含大量 `sh zero` 填 pad slot),然后 `vle16` 把索引数组从栈回读、才 `vluxei16` gather。
  这条 **"标量算 → 栈存 → 向量回读 → gather"** 链是关键路径,不是 gather 时延。
- **两个 iq3 LOSS 的共同根因**(ODS 语义,`RVVOps.td:8388`/`7357`):
  - **iq3_s**:9-bit 索引 = `idx = qs[..] | ((qh[ib] << (8-2l)) & 256)` —— 第 9 位从**独立 qh 位面**、以**逐 lane 变量移位** `(8-2l)` 注入 → 我方标量化(128 `sh` 风暴)。
  - **iq3_xxs**:grid 索引本身是**直接字节** `q3[..]`(不需注入),但**符号** = `ksigns_iq2xs[(aux32 >> 7*l) & 127]` —— 一个**逐 lane 变量移位**抽取 + **二级表查**,我方标量化(实证 `disasm_iq3_xxs_POST.txt:44-104` 一串 `lbu` 直接读 q3/aux/ksigns 字节)。
- **WIN 格为何无风暴**(ODS `RVVOps.td:8138`):**iq2_xs** 索引 = `w & 511`,`w` 是 uint16 的 `qs` 字 —— 9-bit 索引是一个 uint16 的**定位低 9 位字段**,`vle16`+`vand` 就地拿到,**无逐 lane 变量移位、无栈回程**;符号选择 `sel = w>>9` 同样 in-vector。**iq2_s** 同理(POST 用 `vzext.vf2` in-vector,无 `sh`)。
- **∴ 判据不是"表大小",是"索引/符号能否 in-vector 组装"**:凡是**逐 lane 变量移位的跨面位注入 / 二级选择器抽取**,我方当前 emitter 就退化成标量 + 栈回程 → grid-decode-bound。

### 1.4 厂商(factory)怎么做(照抄目标,只读 `*.factory_vl128.objdump.txt`)

- **iq3_s factory**(`.L283` 循环体):**索引 in-vector 组装** —— `vlse8`(strided load qs)+`vle8`+`vsrl.vv`+`vwcvtu.x.x.v`+`vsll.vi`+`vand.vi`+`vwcvtu`+`vsll.vi`+`vor.vv` 直接在向量里拼出 16-bit 索引 → **一条** `vluxei16` gather → `vrgather.vv`(符号 shuffle)→ `vwmulsu.vv`+`vwredsum.vs`。**无栈回程,无 `sh` 风暴**(带一条 odd/even 标量尾)。
- **iq3_xxs factory**(`.L235` 循环体):8 字节小 `vse8`/`vle32` 暂存 aux → `vrgather.vv` 展开 grid 索引 → `vsrl.vv`/`vand.vx`/`vnsrl.wi`/`vsll.vi` in-vector 抽字段 → **两条** `vluxei16`(ksigns + grid)→ `vmul.vv` 符号折叠 → `vwmul`/`vwredsum`。大体 in-vector。
- 关键对照:**厂商也用内存 gather(`vluxei16`)查 grid**;它没有把 grid 塞进寄存器、也没用 vrgather 查 512 表。差距**全在"喂索引"这一段** —— 厂商 in-vector,我方标量 + 栈回程。

---

## 2. 可用杠杆盘点(★枚举 + 粗判,不实现、不评估到写代码级)

### 表事实(空间可行性的地基)
| grid | 声明(`RVVMonolithicBlockDotFamily.h`) | entry | 每 entry | 字节 | 索引 |
|------|----------------------------------------|------:|---------|-----:|------|
| iq3xxs | `int32[256]` | 256 | 4×int8 | **1024 B** | 8-bit 直接字节 |
| iq3s | `int32[512]` | 512 | 4×int8 | **2048 B** | 9-bit(qh 注入) |
| iq2xxs | `int64[256]` | 256 | 8×int8 | 2048 B | 8-bit 打包 aux |
| iq2xs | `int64[512]` | 512 | 8×int8 | 4096 B | 9-bit(`w&511` in-vector) |
| iq2s | `int64[1024]` | 1024 | 8×int8 | **8192 B** | 10-bit in-vector |

VLEN128 参照:1 向量寄存器 = 16 B;m8 组 = 128 B;**整个 32 寄存器文件 = 512 B**。

| # | 杠杆 | 打的是哪段 | 空间/可行性粗判 | 粗评级 |
|---|------|-----------|----------------|--------|
| L1 | **表驻留寄存器组**(512 表常驻向量寄存器免内存 gather) | gather | **INFEASIBLE@VLEN128**:最小的 iq3xxs 表(1024 B)= 整个寄存器文件的 **2×**;iq3s=4×;iq2s=**16×**。放不下。即便到 VLEN512 也要吃光整个寄存器文件才勉强放 iq3xxs。且打的是**不主导**的 gather。 | **否决** |
| L2 | **vrgather 分段**(段内查代替内存 gather) | gather | vrgather 只能在寄存器组内索引;m8@VLEN128 = 128 B = 32 个 int32 entry,而表 256–512 entry → 需 8–16 段子表 + 掩码 merge。厂商自己只用 vrgather 做 8 元素小 shuffle,**不用它查 grid**。复杂高、收益低、且仍打不主导的 gather。 | **低** |
| L3 | **子表拆分(分层查)** | gather | 256/512 平坦 LUT 无天然层次,拆分要引入 selector 分支 + merge。无自然因子化。且打不主导的 gather。 | **低** |
| L4 | **预解码到中间格式**(grid decode 提前、存解码后 int8) | 整个 decode | 把每块索引→解码 int8 码**离线展开**存进 repack 布局,热循环免组装 + 免 gather。空间:iq3_s 块 110 B→256 int8 = **2.3× 膨胀**;iq3_xxs 98 B→256 B = **2.6×**。↑ 流字节 → **伤 memory-bound 的 e2e decode**(micro 是 latency-bound、e2e 是带宽-bound)。这是**权重布局改动 = [repack] 战役地界**,不是 grid-decode kernel 修。 | **中(但另一战役 + 改内存性)** |
| **L5** | **★索引/符号 in-vector 组装**(不在任务原列表,但反汇编指向的真瓶颈)——把"逐 lane 变量移位位注入 / 二级选择器抽取 + 栈回程"换成 in-vector `vsll.vv`(per-lane 移位向量)/`vand.vx`(mask 256)/`vor.vv`,照厂商 `vsrl.vv`/`vsll.vi`/`vand.vi`/`vor.vv` | **真主导段**(标量组装 + 栈回程) | 直接砍掉 iq3_s 的 128 `sh`+108 `lbu`+142 `slli` 风暴 / iq3_xxs 的 ksigns 抽取风暴。**同一 constructed kernel 的 emitter 成熟度改动**,不新增表、不改权重布局 → **空间 ≈0**。保 byte-exact 轴(GEARBOX 不动)。 | **最高(按证据)** |

---

## 3. 预计影响格清单(枚举 + 粗判每格 grid 大小)

| fmt | grid 大小 | 索引/符号方式 | 当前态 | 是否 grid-decode-bound? |
|-----|----------|--------------|--------|-------------------------|
| **iq3_s** | int32[512] / 2048 B | qh 9th-bit 逐 lane 变量注入 | LOSS 0.56× | **是(确证)** — 标量 qh 注入风暴 |
| **iq3_xxs** | int32[256] / 1024 B | ksigns 逐 lane 变量抽取(二级表) | LOSS 0.38× | **是(确证)** — 标量 ksigns 抽取风暴 |
| iq1_s | uint64 grid-of-8(iq3_xxs 的 iq1_s sibling,ODS `RVVOps.td:7362`) | 选择器/scale 解码 | 本批未测 | **疑似同族(未测)** — 需 board 复核 |
| iq1_m | 同 iq1_s 家族 | 同上 | 本批未测 | **疑似同族(未测)** |
| iq2_xxs | int64[256] / 2048 B | 打包 aux(已被 [GAP-SB] 修) | WIN 2.57× | 否(gather 已 16→8 in-vector) |
| iq2_xs | int64[512] / 4096 B | `w&511` in-vector | WIN 1.51× | 否(索引就地) |
| iq2_s | int64[1024] / 8192 B | 10-bit in-vector | WIN 2.18× | 否(最大表却最快,判据反证) |
| iq4_xs / iq4_nl | **16-entry** `kvalues_iq4nl` NL codebook(非 256–512 grid) | 微码本 | iq4_xs 2592→2592、gather 0 | 否 — **微码本(16 entry)另一杠杆**(emitter vluxei / tiny-codebook,非 grid) |
| mxfp4 / nvfp4 / fp4 | 16-entry FP4 码本(微) | 微码本 | — | 否 — 同微码本类 |

**分层小结**:真"256–512 entry grid + 逐 lane 变量移位组装"这一类 = **iq3_s、iq3_xxs 确证 + iq1_s、iq1_m 疑似(未测)**,共 **≤4 格**。
iq2 三格是 in-vector / 已修(反例),iq4/fp4 是 **16-entry 微码本**(不同杠杆,别混进 grid-decode)。

---

## 4. 交回结论(★侦察态,待用户裁立项,未实现)

> **若立项**:最高杠杆候选 = **X = grid 索引/符号的 in-vector 组装**(替换"逐 lane 变量移位位注入 / 二级选择器抽取 + 栈回程"这段标量风暴,照厂商 `vsrl.vv`/`vsll.vi`/`vand`/`vor` in-vector 建索引;**gather 本身别再动,[GAP-SB] 已打到 4 条**);
> 预计影响 **Y = 2 格确证(iq3_s、iq3_xxs)+ 2 格同族未测(iq1_s、iq1_m)**;
> 空间成本 **Z ≈ 0**(不新增表、不改权重布局、保 byte-exact GEARBOX 不动;区别于 L4 预解码/repack 的 2.3–2.6× 流字节膨胀 = 另一战役)。

★★ **红线复述**:以上全部为 **侦察画像**,**未立项、未实现、零代码改动**(只读 `lib/`/`include/`/ODS/board 反汇编,只写本 `.md`)。
是否立项、是否采纳 L5(或改走 L4 repack、或都不做)由**用户裁**。本文不 commit(由用户提交)。

### 关键 caveat / 不确定
- L1/L2/L3 打的是 **不主导的 gather**;把它们当"grid-decode 杠杆"是被"查表主导"这个前提误导 —— 反汇编实证 gather ≈0.3%,已被 [GAP-SB] 收口。
- L5 的**具体收益数字未知**:本文只给"砍掉 X 条标量指令"的**结构判**,**没有**任何预测 speedup;真收益必须 board A/B 实测(且 micro 胜未必传导 e2e,见项目 memory `kernel-wins-dont-transplant-to-e2e`)。
- iq1_s / iq1_m 的"同族"是**基于 ODS sibling 关系的推断**,**未经 board 反汇编复核**;立项前应先取其 POST 反汇编确认是否同型标量风暴。
- L5 是否真能保 byte-exact(逐 lane 变量移位向量化后的 fold order)属**实现期风险**,本侦察不背书,应由 implement + byte-exact gate 验。
