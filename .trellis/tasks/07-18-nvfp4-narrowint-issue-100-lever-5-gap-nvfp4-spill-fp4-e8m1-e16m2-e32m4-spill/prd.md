# PRD · nvfp4 窄位宽整数乘加 lever（ISSUE-100·§五.5·攻 [GAP-NVFP4-SPILL] 墙）

> **权威** = 补充令二 §五.5（ISSUE-100 窄位宽整数乘加·**从未试**）+ ISSUE-100 订正（§六 premature-架构不可达·float re-roll 路已证尽 ≠ 整数杠杆已试）+ §六（杠杆清单非空禁写架构不可达·本 task 打这个非空 lever·板测后方定 flip / 真空-honest-null）。
> **性质** = 发射器攻坚（结构级·整数化点积·换掉 float 三态宽度阶梯）。**对手 = ggml 架构覆盖空洞（0 处 nvfp4 override）= opp-immaturity 便宜档**——翻正 ≠ 硬赢强手调（§三.12·必标注）。

## 一、背景：float re-roll 路已证尽·真墙 = 三态 spill

前序完整攻坚环（ISSUE-100·`experiments/active/g8-stage3-attack/nvfp4-rvv/`）证 **float 路** exhausted：
- 构造 re-roll 4-way + hoist fcvt.s.h → G1 byte-exact GREEN → **G2 板测 NULL（1.0028/1.0015/1.0046·远不足 0.57→0.8）** → 反汇编 spill 前后不变。
- **墙** = `[GAP-NVFP4-VLEN128-SUBBLOCK-GRANULARITY-SPILL]`：371 指令中 44 条（22×csrr vlenb + 11 spill + 11 reload）**0 贡献**。根因 = **`e8m1`/`e16m2`/`e32m4` 三态同时存活**强制 spill（QK_NVFP4=64 super-block·VLEN128 vl=8/op·三个 SEW/LMUL 宽度阶梯并存）。

## 二、本 task 的 lever：FP4 码本整数化·单宽度阶梯整数乘加

**核心** = 消掉 float 三态宽度阶梯（那是 spill 墙的根因），改走**单一整数宽度阶梯**：
- **FP4 (e2m1) 码本值** = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}。**×2 后全整数**：{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}（int8 可容）。
- 点积 = Σ_i (fp4_w_i × q8_a_i)。整数化：Σ_i (2·fp4_w_i)·q8_a_i = **int8 × int8 → int16/int32 vwmacc 累加**（单宽度阶梯·**无 e8→e16→e32 float 三态**）。
- **per-block 标量 fold**：末尾一次 `× 0.5 × fp8_block_scale × q8_scale`（标量·非每元素 float FMA）。
- ⟹ 目标：消掉 22×csrr vlenb + spill/reload（三态并存消失）·拿回那 44 条 0-贡献指令。

**复用/参考**：现 `emitNVFP4BlockDotBodyShared`（`lib/Conversion/RVV/RVVToEmitCCodebookFp4.cpp`·sole live caller）。新增整数路变体（gate 后·默认路径不动·byte-identical）。

## 三、★数值口径（补充令二 §二·必读）

整数累加与 float oracle（5 棵折叠树·规格）的舍入**可能不同**（整数精确和 × 一次 scale ≠ float running-sum）：
- **先试整数 byte-exact**：若整数路凑巧 byte-exact（量化点积 int32 零舍入·免费）→ ULP=0·headline 直用。
- **若非 byte-exact** → 按 §二 **数值松绑变体**：headline 取该最快变体·**强制报 ULP 上界**（三数并列：最快 X× + ULP 界 Y + 精确 Z×）。**唯一硬约束：报数必须带 ULP 界·不说精度的数不是数。**
- **禁**：无 ULP 界的裸速度数。

## 四、★预期 —— 预测不是实测（先测不预告·[§五.15]·ISSUE 自承低置信）

ISSUE-100 自承此杠杆「与 [GAP-P1] 同构·低置信」。**先测·不预告。**
- cold≥0.8 → nvfp4 翻（关 ISSUE-100·消一个具名-X）·**但 opp-immaturity 成色**（对手=覆盖空洞·非强手调）·标 §三.12。
- cold<0.8 → 具名 + 窄位宽整数乘加 lever 板测 EXHAUSTED·objdump 逐指令墙（三态是否真消/spill 是否降）·**此后若杠杆清单真空 → 方可 §六 架构不可达/honest-null**（那才是合法终审）。

## 五、验收

1. **owned 整数路 emit**（objdump 证：三态 spill 消失/降·单整数宽度阶梯·vwmacc·非 float e8m1/e16m2/e32m4 并存·OWNED 确定性）。反汇编对比 baseline 371 指令·数 csrr vlenb / spill。
2. **byte-exact 或 ULP 界**（[K-5] ZERO-MODEL·nvfp4×q8_0 oracle·md5 全链）·**报数带 ULP 界**（§二硬约束）。
3. **cold 2-seed**（`vec_dot.sh rvv verify/measure nvfp4`·flush224·idle 100%·gov performance）·verdict（先测·≥0.8 翻 / <0.8 墙）。
4. **成色**：对手 = `arch/riscv/quants.c` **0 处 nvfp4 override** = ggml 架构覆盖空洞·host-autovec scalar 路 = **opp-immaturity 便宜档**（§三.12）——翻正**不得**称硬赢强手调（对比 q4_K vwredsum 那条才是强手调）。
5. **0 造数**·byte-exact/ULP 硬门·**sealed 不动**·master 不直写（recon-dict 留 main）·**禁 commit·禁 add -A**·objdump 全量。

## 六、触碰集 / 遗留

- 触碰：`lib/Conversion/RVV/RVVToEmitCCodebookFp4.cpp`（`emitNVFP4BlockDotBodyShared` 加整数路变体·gate 后）+ 可能 `RVVToEmitCInternal.h`（decl）+ `test/`（对应 nvfp4 block-dot test）+ `vec_dot.sh` + bench 跑（rvv）。**与并行线 vwredsum（KQuant·`RVVToEmitCKQuant.cpp`）TU 不相交**——但**共享 build dir·须串行**（本 task 等 vwredsum 线完成后独占 weft-opt 重建窗口）。
- **遗留**：mxfp4 同族 FP4 码本（若整数化 work·同结构可扩·但 mxfp4 已 CLANG_WORLD 重测·非本 task）。verdict 定 ISSUE-100 收口（翻 / 真墙·真空则合法架构不可达）。
