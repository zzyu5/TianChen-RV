# PRD — B线第一块批2: tiny-codebook 16-entry dequant de-lottery

## 贡献归属
C3′ de-lottery（[L-8] 强义 construction）——用 owned 真向量 emit 替代宿主 autovec codegen 抽签（ISSUE-002），逐格关 per-format 敞口。**非 perf 硬赢**（成色=便宜档·见下）。

## 目标（4 格 @rvv·扇出=4）
`iq4_nl` · `iq4_xs`(census-F7) · `mxfp4` · `nvfp4` 的 `dequantize_row` 从 **scalar-forwarder（autovec-lottery）** 升为 **owned 真向量 emit**（vrgather 16-entry 码本原语），byte-exact GREEN，板测 cold verdict。

## 机制（扇出来源 = shared body）
- 现状：`emitDequantizeRowCodebookGridBodyShared`（`lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp:3611`）仍是 scalar forwarder → `emitGgmlDequantizeRowExtended`（scalar）。
- 施工：author **1 个 owned vrgather 16-entry 码本 body**（16 项码本寄存器常驻·`vrgather` 索引解包·**非大 grid vluxei 内存 gather** → 无 gather 墙），参数化 scale 粒度覆盖 4 格：
  - `mxfp4`：E8M0 block-shared scale（**已有 3.47× WIN 正锚**·ledger 机制⑤·vrgather 路已存 → 适配置信最高·从此格起）
  - `nvfp4`：4×UE4M3 sub-scale
  - `iq4_xs`：super-block signed-6 scale
  - `iq4_nl`：flat fp16 scale
- family 内首格（mxfp4）适配后·余 3 格 = 换 scale-decode 段复制。
- **sealed 整数核 / vec_dot 路不得动**（本任务只碰 dequant forward 路·收尾核对 vec_dot emitter md5 未变）。

## harness 建设（收割真前置·全 blocked 须先建）
`tools/bench/cells/dequantize_row.sh` case（L56–79）现只认 5 block + iq3_xxs；4 收割格触 `HARNESS-VOID bad fmt`（L78）。须建：
1. **4 owned kernel**（`experiments/active/r-dequant/kernels/{iq4_nl,iq4_xs,mxfp4,nvfp4}_dequant.c`·含 vrgather 码本 decl）—— 复制批1（a9ec·commit ababfada9）的 kernel/driver 骨架方法学。
2. **4 driver**（LEAFC owned kernel + DRVC driver + GSED fault 探针 + OPPSYM）。
3. **oracle 对拍**：ggml `dequantize_row_{iq4_nl,iq4_xs,mxfp4,nvfp4}`（住 libggml-base）—— ZERO-MODEL 独立重算。
4. **case 扩 4 条**（dequantize_row.sh 契约 L43–55）。

## byte-exact 门（硬前置·先证再测 perf）
- **mism=0**（全语料·参照批1 mism=0/262144）·**3-arm bite**（owned vs oracle vs 参照）·**CORPUS COMPLETE**（非采样）。
- **FMA-contract 一致**：d×(码本值)×scale 的 fp 运算序须与 oracle 数值契约一致。
  - **整数解包段（码本索引 / bit-unpack）= byte-exact 守**（0 容忍）。
  - **浮点缩放段允许重排变体**：须标 `numerics.reassoc_ok` + ULP 界（|Δ|≤ 明示界）·不得静默漂移。
- **门未过 = 不进 perf 测**（fail-closed·跑不过如实记具名-X 不是资产）。

## 板测（byte-exact 过后）
- board = **rvv**（`ssh rvv` 免密·VLEN128·系统 clang18）。
- **2-seed** cold（ours / opp）·opp = 部署 `dequantize_row_*`（标量类·autovec）。
- verdict：`cold ≥ 0.8 → PASS`（0.8 硬门）·`< 0.8 → 具名-X`（如实·不粉饰）。

## 成色（钉死·禁越界）
- **全 4 格 = 便宜档**（opp = host-autovec-of-scalar-C·opp-immaturity·§三.12）。
- 价值 = **关 ISSUE-002 该格 codegen-lottery 敞口 + [L-8] 强义 construction**·**非 perf 硬赢**。
- `nvfp4@rvv` 已 lottery-PASS 2.1736 → de-lottery = 换 owned·关敞口·**非涨数**（成色 upgrade 非新地盘）。
- **禁称硬赢·禁外推 grid（iq1_s/iq1_m 仍 gather 墙排除）/ vec_dot**。

## 墙风险（低-中）
- 16 项 vrgather = 正确原语（非 vluxei 内存 gather）→ **无 gather 墙**。
- `nvfp4` vec_dot@rvv 曾诊「码本 TABLE spill」（census F5）·但那是 vec_dot·**dequant streaming 应更干净**（若 dequant 侧也撞 spill → 三步 objdump 登记·不硬凿）。

## 交付（首节三项·回主会话）
1. **消灭待办 + 退休哨兵计数**：本批 owned 收割几格 byte-exact GREEN·头条 count Δ（几格 具名-X→PASS 真地盘 vs 几格 lottery→owned 成色 upgrade·分开报）。
2. **剩余待办各自未试杠杆 + 责任人**（禁「已尽力/架构不可达/资产」除非走完三步 objdump 登记）。
3. **本批使哪些论文侧读数过期**（标量类-rvv 分母 / de-lottery 计数 / census-F7 iq4_xs 覆盖）。

## 账面纪律
- **构造轴 headline 走独立 check 再 commit**（byte-exact 复验）。
- master 入账 = `recon_master_rebuild.py` CLANG_WORLD dict（参照批1 commit 47ab36df9 的 5 条 K-quant 范式）·regime=""·board=="rvv"·cold≥0.8?PASS:具名-X。
- git add 只纳源 + kernel + harness·**勿纳 build-wt**（worktree 隔离·cherry-pick 到主树只取源路径）。

## 参照
- 批1 收割：commit `ababfada9`（K-quant super-block 5格·方法学范式）+ master 入账 `47ab36df9`。
- 收割 scope：archive `07-19-block1-harvest-scope`（`research/harvest-batch-plan.md` 批2 节 + `de-lottery-applicability-boundary.md`）。
- de-lottery 范式：CLANG_WORLD dict `recon_master_rebuild.py:271-285`（q4_0/q5_0/q8_0 owned 条目）。
