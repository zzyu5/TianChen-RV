# PRD — grid dequant 全族翻 k1 半（iq3_xxs/iq3_s/iq2_xs dequant @k1）

## 缘起（全族翻两板·修缺口不撤）
r5.1-c/d 翻了 grid dequant @rvv（iq3_xxs 1.39/iq3_s 2.08/iq2_xs 3.88·真地盘·codegen-STRUCTURE 墙）。但 master 显示 **iq3_xxs/iq3_s dequant @k1 仍具名-X**。我方 emitter flip（emitDequantizeRowIQ3XXS/IQ3S/IQ2XSVectorBody）**板无关**（同 emit·k1 VLEN256 彩票叶更爆）——@k1 board 重测应同样翻。补 k1 半。

## 做（纯 board 重测/部署·发射器已改·勿再动 emitter）
- 用**已翻的 emitter**（主树 emitDequantizeRowIQ3XXS/IQ3S/IQ2XS·r5.1-c/d 已 deployed·勿改）regen iq3_xxs/iq3_s/iq2_xs dequant kernel·@**k1**（`ssh k1`·VLEN256）board 测。
- 前置核：k1 deployed 现 leaf objdump（有宽/有 gather → 翻·像 rvv）。
- byte-exact @k1 VLEN256（ZERO-MODEL mism=0·3-arm·CORPUS·k1 无 PMU 用 footprint 恒等）· 2-seed cold · vs 部署 ≥0.8 入账。
- iq1_m @k1（前置核·若无宽=lever-N/A de-lottery·同 rvv）。

## 触碰集
grid experiment（experiments/active/ 独立目录·k1 kernels/harness）+ k1 board。🔴 禁碰：任何 lib/ 源（emitter 已翻·勿改·只 regen 消费）· r-dequant committed kernels · RVVLowerQuantContraction/Gearbox/capability/SourceFrontDoor。**纯 board 消费已翻 emitter·零源改**。

## 门
- byte-exact 先于计时（@k1 VLEN256·两板不回归）· board 2-seed cold · vs 部署 ≥0.8 入账 · CORE==PROD（k1 regen md5）。
- 🔴 无 inline-asm。真翻不了 @k1 → 走完环证墙型（k1 VLEN256 gather 结构 vs rvv 差异·honest-null 具名）。

## 交付
- **交付首节自带「本流 k1 翻了 x 格 / y honest-null」** + 每格 {前置核·byte-exact@k1·board ratio·墙型} + 文件清单（k1 experiment·零源改）。
- 板不可达则如实报（唯一允许不造数）。0 造数·sealed md5 变须报。
