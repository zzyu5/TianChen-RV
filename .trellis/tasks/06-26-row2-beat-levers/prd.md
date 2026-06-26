# row-②-beat 杠杆 cm4/cm5/cm6（board，HIGH 风险）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。**这是腿(2)——唯一能拿"成熟 compiler / 反超 ggml"headline 的一类**：综合一个 ggml 没手写的 within-kernel 形状。已授权（D-beat=YES）。**坐两颗雷上**：Win-C structural-NULL [[winc-structural-null]] + kernel-wins-不传导 [[kernel-wins-dont-transplant-to-e2e]]。

## Goal
不止追平——**emit 一个 ggml 没写过的 in-kernel 形状**（wider-LMUL / VLEN-tuned strip / multi-accumulator）并**实测更快**。这是"copy-then-adapt、multi-day、timed-out-2×"emitter 类。

## In-scope（3 原子项）
- **cm4 VLEN256 wide-cover**：VLEN256 上把现仍 VLEN128-形（半寄存器）的 kernel 综合成满-VLMAX 宽形——真 row-② 候选。**先做（cm5/cm6 复用其 wide-LMUL emitter 技术）**。
- **cm5 K-quant 多-LMUL emit**：≥2 个**不同指令**的 LMUL 形（非只改 vsetvl 数）；adopt schedule autotuner（接口 adopters 实现时复核，已扩到 q1_0/tq2_0/tq1_0/iq2_xxs）。**宽形必须实测更快，否则 = NO claim（placeholder 不算成功）**。
- **cm6 multi-accumulator LMUL**：defer-eligible——**先 pre-board 可行性证明，再 build**。

## DoD
- **byte-exact gate 先过**（vs scalar oracle + vs ggml `_vlN`）；非-NULL VLEN-flip lit；objdump 板上确认 LMUL。
- perf micro **且** e2e（被触及格），**两板**；**landing = TARGET 非 banked**：结果可能是 beat / parity / NULL / board-pending——如实标，beat headline 不预支。
- 触及格交 [[06-26-table-retest-fill]] 重测（cm4→VLEN256 wide-cover 格；cm5→选定 K-quant 格）。

## Deps / Risk
soft dep：cm4 先于 cm5/cm6。cm6 须 pre-board 可行性证明。**HIGH timeout/NULL 风险**（infra 之前 timed-out 2×）。**must-NOT**：把 placeholder/未实测当 beat；decode micro win 不声明 e2e（须实测传导）。
