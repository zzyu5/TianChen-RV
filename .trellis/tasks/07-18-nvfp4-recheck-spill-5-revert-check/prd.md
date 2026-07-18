# PRD · nvfp4 re-diagnosis 独立复核（裁决5·正负对称）

> **权威** = 用户裁决5（2026-07-18）：「nvfp4 revert + 改判：补一个独立 check 复核『真墙=码本表 spill』。改判/revert 此后走 check·正负结果对称。」
> **性质** = 独立复核（非重攻·验证已 commit 的 re-diagnosis 断言）。**这是补做的 check**（主会话先前 revert+改判 nvfp4 时未走独立 check·裁决5 要求补上·确立正负对称纪律）。

## 一、被复核的断言（已 commit `3f8f9e569`·你独立验真）
task `07-18-nvfp4-narrowint`（a72d 单源·未独立 check）板测结论 + 主会话 revert + 改判：
1. **narrow-int MAC lever 板测 EXHAUSTED**：FP4 码本×2 整数化·i8×i8→i16→i32 vwmacc·vlmul_trunc products 降一档 LMUL → byte-exact ULP=0 → **cold NULL（attack_over_base 1.010–1.012×·ratio 0.70<0.8·不 flip）**。
2. **★真墙 re-diagnosis（推翻 ISSUE-100 原「三态 register-footprint」假说）= 溢出对象是码本 TABLE（`v8`）非 products**：clang-18 把码本表 `vs1r.v v8` spill 在交织的 scalar UE4M3 float-scale 计算之间·per sub-block·**对 product/reduce LMUL 不敏感**（narrow 释放 product 寄存器·表仍 spill 11×·三态仅 LMUL 阶梯下移一档 e8m1/e16m2/e32m4→e8mf2/e16m1/e32m2·spill 11/11/22 逐字节不变）。
3. **env-gated 变体已 revert**（`WEFT_NVFP4_INT_NARROW` 侧信道）·sealed 默认 byte-identical（md5 `afad1b43`·lit 499/499）。

## 二、你要独立验的（objdump 聚焦·测速度有配额·此 check 主证在 objdump 非板窗）
证据在 `experiments/active/g8-stage3-attack/nvfp4-rvv/narrow-int-lever/`（untracked·kernel 源 `nvfp4_narrow_INT.kernel.c` + `raw/` 全 objdump + md5 链）。
1. **码本表 spill 真是 `v8`·不是 products（核心 re-diagnosis）**：读 saved objdump（base + narrow）·确认 `vs1r.v v8`（spill）+ `vl1r.v`（reload）的对象是**码本 LUT 寄存器**（`vrgather` 的 table operand）·非 product/accumulator·且 spill 计数 base↔narrow 逐字节不变（11/11/22）。**独立交叉对齐 kernel 源的 vrgather↔v8 分配**——证 narrow 释放 product 寄存器（LMUL 下移）而表仍 spill = 墙 ≠ 三态 register-footprint = clang reg-alloc 行为。
2. **sealed 默认 byte-identical**：forced clean rebuild weft-opt（当前 clean 源·env-unset）→ regen nvfp4 默认 kernel → md5 `afad1b43`（不变）·lit 相关绿。**证 revert 干净·无残留**。
3. **cold NULL 抽验（配额内·可选）**：若 objdump 已充分坐实 re-diagnosis·**cold 不必重跑**（a72d 已测 1.01×·NULL 非争议点）；仅当 objdump 不足以定论时才补一次 2-seed。**说清哪些读得到（objdump）·哪些没读（未重测 cold·引 a72d）。**
4. **成色**：对手 = `arch/riscv/quants.c` 0 处 nvfp4 override = 覆盖空洞 opp-immaturity 便宜档——确认 revert 后 ISSUE-100/ledger/census 措辞如实（禁称硬赢·even 若翻正）。

## 三、正负对称纪律（裁决5）
- 这是**负结果**（EXHAUSTED·NULL）的独立复核——与正结果（flip）同等走 check。verdict = re-diagnosis 站得住 / 不站得住。
- 若 re-diagnosis **REFUTED**（如 spill 其实是 products·或 base 不复现 sealed）→ flag main 回退改判。
- 若 **CONFIRMED** → ISSUE-100/ledger/census 的「码本表 spill」措辞坐实·杠杆清单（table-residency/scalar-scale hoist 低置信 maturity-gated）成立。

## 四、纪律
- **0 造数**·objdump 全量·测速度配额内（objdump 优先·cold 非必需重跑）·**禁 commit·禁 add**·master/sealed 不动。
- forced clean rebuild weft-opt（验 sealed 默认 byte-identical）。判据级（改判措辞）不自决·验证证据链留 main。

## 五、汇报
返回：(1) 码本表 spill=v8 re-diagnosis CONFIRMED/REFUTED + objdump 逐指令证据（v8 是 vrgather table·spill base↔narrow 不变）；(2) sealed 默认 byte-identical 是否守住（md5）；(3) cold 是否需重测（说清读到/没读）；(4) 成色措辞是否如实；(5) 若 REFUTED·main 回退建议。**触碰集 = 只读 evidence + regen 验证·零 repo 写。**
