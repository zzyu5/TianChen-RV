# PRD — iq4_nl gemm pending-fold 收尾（唯一 pending 非墙格）

## 缘起
master `gemm_tile iq4_nl @rvv = pending-fold`（唯一 pending·非已攻坚墙）。收尾判：翻/honest-null。

## 做
- 核 iq4_nl gemm@rvv 现 fold 状态（为何 pending）·完成 fold（若缺 fold 逻辑）·board 测。
- 前置核 objdump（有无宽可窄/gather·像 grid dequant 分轴）。byte-exact 先于计时·board 2-seed cold @rvv·vs 部署 ≥0.8 入账。
- 真翻不了→走完攻坚环证墙型（M=1 结构 or 编译器行为）honest-null。

## 触碰集
iq4_nl gemm emitter（`lib/Conversion/RVV/` 或 `lib/Plugin/RVV/` 的 iq4_nl gemm/codebook path·先 grep 定位）+ iq4_nl experiment（experiments/active/ 独立目录）+ board。🔴 禁碰：RVVLowerQuantContraction（B 流）· RVVToEmitCGridCodebook 的 grid dequant/vec_dot body（grid agents 只读）· capability/SourceFrontDoor。碰同文件报冲突。

## 门 + 交付
- byte-exact·lit 绿·CORE==PROD·board·无 inline-asm。
- **交付首节自带「本流翻了 x / honest-null y」** + {前置核·lever·byte-exact·board ratio·墙型} + 文件清单。不 git commit·0 造数。
