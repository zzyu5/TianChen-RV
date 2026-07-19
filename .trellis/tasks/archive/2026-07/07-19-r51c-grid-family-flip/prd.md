# PRD — grid 族全族翻（扩 iq2 面 + 裁2 grid 墙可翻后的族级收割）

## 缘起（Stop-hook 真缺口 + 裁2）
W4 证 grid gather 墙=codegen-STRUCTURE 可翻（iq3_xxs dequant 0.36→1.39·commit 9d74daa96）·并明说「窄化 per-entry lever 对 vec_dot grid 族是有前景后继杠杆·未试」。B线「扩 iq2 面」+「grid 族全族翻」= 把 grid 具名-X 格用同 lever 翻。**遇挫不退·输了修了再打。**

## W4 已证 lever（复用·非新发明）
攻坚环 bisect 铁证（W4·板上三点）：baseline gather=8/0.33 → vB slideup **gather-free 但仍 0.35（单纯消 gather 不解墙）** → vC **窄化 per-entry（opponent 形状·owned intrinsic·scalar sh2add 指针 + vle8 vl=小 + 廉价 m1 widening）gather=0/1.36**。真 lever = **gather-free 装配（必要）+ 窄化 per-entry 廉价 m1 widening（主导·旧 wide-LMUL m8 vsext_vf4/vfcvt 才是主成本中心）**。参照：`experiments/active/r51w4-grid-schedule/research/vC_narrow_perentry_standalone_win.c` + emitDequantizeRowIQ3XXSVectorBody（已 deployed）。

## 目标格（grid 具名-X·按可达性排）
1. **iq2_xs vec_dot@rvv 0.617**（ISSUE-120 已修 byte-correct·grid 块内积·emitIQ2XSSuperBlockGridBody）——「扩 iq2 面」主标的。试窄化 per-entry lever（vec_dot 族·W4 说有前景未试）。
2. **iq3_s dequant**（0.526/0.665·grid）——若无 owned body 则 de-lottery+owned 构造（同 W4 iq3_xxs 范式）。
3. **iq2_s / iq2_xxs vec_dot@k1**（byte-correct 后）——延伸（依赖 board）。

## 触碰集
`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（grid 发射器·iq2_xs/iq3_s body）+ grid experiment（experiments/active/ 新建独立目录）+ 板测。🔴 禁碰：RVVLowerQuantContraction/Gearbox·capability/SourceFrontDoor·r-dequant kernels·W4 已改的 emitDequantizeRowIQ3XXSVectorBody（除只读参照 lever）。

## 门（byte-exact 先于计时·硬）
- byte-exact（ZERO-MODEL mism=0·3-arm anti-hollow·CORPUS·两 VLEN 若涉 k1）· board 2-seed cold。VLEN128 不回归（iq2_xs@rvv 现 byte-exact 别破）。
- vs 部署 ≥0.8 入账（门 0.8·哲学锁①）。CORE==PROD（deployed emitter 真产出·regen md5 验）· golden lit 同步。
- 🔴 严禁内联汇编/钉死调度手排绕 clang（发射层调度锁≠手排·裁2 红线）。真翻不了→走完攻坚环（objdump 证 compiled leaf.o 仍 gather 且 owned 结构被重向量化不可锁）证硬件吞吐墙才 honest-null。

## 交付
- **交付首节自带「本流翻了 x 格 / y 格走完环 honest-null」** + 每格 {lever·byte-exact·board ratio·墙型判定} + 文件清单。
- 账面：不 git commit（报清单供主会话合并）· 0 造数 · sealed md5 变须报 · 禁碰守界。
