# PRD — W5 载体 rvv A′ 上板（排期已到·转完成时）

## 触碰集（W5 专属·板窗）
`experiments/active/r5.1-w1-carrier/` 的 rvv A′ 双 VLEN 双 body + D4 experiment kernels/harness + rvv 板测。**禁碰**：任何 lib/ 源（W1/W2/W4）· r-dequant kernels（W3 只读）· grid emitter（W4）。W5 是纯 experiment/board·与代码流天然不相交。

## 背景（W1 载体批 a3b0c00f1 遗留·排期 2026-07-20/21 已到）
上波 W1 载体批把 rvv A′ 结构半 pin + 板排期。本波**一次板窗兑现 A′+D4**·A′ 从"条件式"转"完成时"·落 pin。

## 做
- rvv A′ 双 VLEN（VLEN128 + VLEN256 form）双 body 上板：byte-correct（ZERO-MODEL mism=0·两 VLEN 各对拍）+ 一次板窗兑现两主张。
- D4 一并兑现（W1 载体批 FINDING 里的 D4 项）。
- 落 FINDING + pin（run-id · board identity · md5）· git add -f 入库（tracked FINDING）。
- 板不可达则如实报"板窗未开·排期顺延"（唯一允许·不造数）。

## 交付
- **交付首节自带「本流 A′ 上板 x/y 主张兑现」** + rvv A′ pin（run-id · byte-correct 证 · A′ 转完成时）+ D4 状态。
- 账面：git add -f 只纳 FINDING + raw logs（不纳 build/worktree）· 0 造数 · run-id 找不回重跑一次（唯一允许补测）。
