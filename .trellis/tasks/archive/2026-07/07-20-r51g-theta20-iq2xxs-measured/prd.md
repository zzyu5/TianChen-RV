# PRD — T3 (HIGH): θ20 iq2_xxs measured-table(2nd measured θ·板 A/B seed·续 q8 范式)

## 价值(measured 机制已证 q8·扩 2nd θ·真 WIN 候选)
θ20=`RVVToEmitCKQuant.cpp:5971 getIntegerCoreLmul value_or "m2"`(iq2_xxs·IQ2XXSGridBodyContext:2792 有 coreLmul 字段·**无 exhausted-negative**·区别 θ18/θ19 K-quant widen 脾气墙 EXHAUSTED)=真 gearbox·真 WIN 候选。机制同 q8_0 measured-gate 刚证模板(commit 2ab1f9d4d)。

## 做(续 q8 范式·registration-as-DATA)
- 板 A/B seed(`ssh rvv`·iq2_xxs decode m2 vs m1·2-seed cold·byte-exact·同 q8 [GAP-P1] 做法参照)。
- m1 板测更快且 byte-exact + spill-free(rvvRegisterPressureLegal 核)→建 measured 表(registration-as-DATA·仿 kRepackMeasuredM1FasterMeasurements·**无 format-switch·无 sentinel**)驱动 iq2_xxs。m2 更快=保持 m2 default(诚实非硬塞)。
- 判决 lit: measured 表有 iq2_xxs 行→m1·无→default m2。

## 触碰集
`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`(iq2_xxs body/context+measured 表·KQuant/grid 路·**非 repack front door**)+KQuant lit+iq2_xxs experiment(experiments/active/ 独立目录)+board。🔴 禁碰: RVVLowerQuantContraction(q8 表)·RVVCapabilityProfile(T1)·RVVToEmitCCodebookFp4(T2)。

## 门
- byte-exact 先于计时(m1==m2==oracle·LMUL 翻不改算术)·board 2-seed cold·CORE==PROD·无 inline-asm·spill-free 核·measured 表非空=measured-table θ 1→2。

## 交付
- **交付首节「判决实验 x/y + iq2_xxs board m1 vs m2 数据」**+measured 表 diff(2nd measured θ)+墙型(m1 赢/m2 保持)+文件清单。0 造数·不 commit·sealed md5 变须报。
