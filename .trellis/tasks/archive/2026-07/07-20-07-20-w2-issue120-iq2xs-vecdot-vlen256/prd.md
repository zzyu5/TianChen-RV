# PRD — W2 / ISSUE-120：iq2_xs(+iq2_s) vec_dot VLEN256 byte-broken 修复（correctness·裁6·门先于计时）

## 权限 & 优先级
【现行法·correctness 修复】。行动书 §五 裁6 + §优先序②：**「正确性门先于一切计时」·correctness 债不与工程债同队列**·且挡着宽化向 iq2_xs/iq2_s 扩面。**这是 correctness 任务·不是 perf**——**禁做任何计时/perf claim**·只要 byte-exact 双板绿。

## 病（ISSUE-120·发射器与架构.md:216·板测暴露）
- deployed `iq2_xs` vec_dot serial emit `emitIQ2XSSuperBlockGridBody`（**自验位置**：`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp:1475`）在 **VLEN256（k1）byte-broken**：`ours_vs_int_oracle=false`·mism=512（ggml+oracle 一致·仅我方错）。
- **pre-existing**（原 `iq2_xs.kernel.c` 同样 broken·非新引入）。
- **根因** = pair-batched 结构（`vget_i8m4_i8m1` 类·LMUL8 32-lane 假设）是 **VLEN128-form**·VLEN256（64-lane）崩。
- **iq2_s**（`emitIQ2SSuperBlockGridBody`·`RVVToEmitCGridCodebook.cpp` 附近·**自验**）**pair-batched 同结构·likely 同病**·一并核。

## 做（VLEN-universal 修·byte-exact 双板·ZERO-MODEL）
1. **先自验根因**：`grep -n` 定位 `emitIQ2XSSuperBlockGridBody` / `emitIQ2SSuperBlockGridBody` 真实体·读出 pair-batched 结构（LMUL/lane 假设写死处）。⚠**PRD 锚点可能 naive-grep 误判**（前几个 agent 撞过·诚实重路由才对·别按错锚硬改）。
2. **VLEN-universal 修**：把 VLEN128-form 的 32-lane 假设改成 **VLEN-自适应**（读 minimum_vlen/vlenb 事实定 lane·或 unbatched VLEN-universal 路）。**优先 unbatched-VLEN-universal**（一条路两板都对）·除非 unbatched 破 byte-exact 才 per-VLEN 专化。
3. **iq2_s 同修**（若自验确认同病）。
4. 🔴 **严禁 inline-asm / 钉死调度**·严禁 value_or 自补几何。

## ★验收 = ZERO-MODEL byte-exact 双板（correctness·非计时）
- **k1 VLEN256**：`ours_vs_int_oracle` mism **0/N**（从实际输入零复用重算全算术项·ZERO-MODEL·非 output-vs-output）·CORPUS 完备（grid entries + signs 全覆盖·非单向量）·3-arm anti-hollow（ours==ggml==independent-oracle）。
- **rvv VLEN128**：同样 mism 0/N（**修完不许回归 VLEN128**·原本 VLEN128 是对的·别修坏）。
- **CORE==PROD**（前门构造路 == 部署 fallback 路 md5 一致·修的是部署真出的核）。
- 判决观测点（committed lit 或 board harness）：修前 VLEN256 mism=512·修后 mism=0·VLEN128 全程 mism=0。

## 触碰集（先自验·禁碰在飞 agent 文件）
- **主**：`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（emitIQ2XS/IQ2S SuperBlock body·**自验行号**）+ 可能连带 `RVVToEmitCInternal.h`（签名·若加 VLEN 参数）+ board harness（`experiments/active/` 独立目录）。
- 🔴 **禁碰**：选择器侧（`RVVReductionSourceFrontDoor.cpp`/`RVVContractionPathSelection.cpp`/`RVVGearboxSchedule.h`·MIG-5 agent 在改）·`RVVLowerQuantContraction.cpp`·`RVVToEmitCForwardElementwise.cpp`。若需碰·停·报 supervisor。
- **板**：k1（VLEN256·主验）+ rvv（VLEN128·回归守）。`ssh k1` / `ssh rvv`。

## 门
- byte-exact 双板 mism=0（ZERO-MODEL·CORPUS·3-arm）· CORE==PROD · VLEN128 零回归 · pre-existing 修（标 pre-existing 非新引入）· 无 inline-asm · **禁计时/perf claim（correctness 任务）** · 未 git commit。

## 汇报（自然·首节 byte-exact 双板 x/y）
根因（pair-batched VLEN128-form 具体处）+ 修法（unbatched or per-VLEN）+ k1 VLEN256 mism + rvv VLEN128 mism + iq2_s 是否同病同修 + CORE==PROD md5 + 锚点是否与 PRD 一致。final message 放关键结论。
