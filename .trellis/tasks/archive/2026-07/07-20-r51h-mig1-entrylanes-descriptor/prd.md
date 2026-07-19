# PRD — 迁移 #1: entryLanes 格式几何 焊死→描述符（还律2债 + 模块化公式层第一步）

## 缘起（论文侧审计 B1·律2 违反·确认）
B 线翻 grid dequant（iq3_s/iq2_xs）时·往发射体烘了格式常数：`RVVToEmitCGridCodebook.cpp:2832 entryLanes=4`（iq3_xxs）·`:3091 entryLanes=8`（iq3_s）·`:3301 entryLanes=8`（iq2_xs）+ `groupLanes=8`(:504/1002/2532)。这是 grid 码本格式几何（g 轴值·codebook entry byte-width）·**律2「机制体里不能出现具体格式值」明禁**。审计确认在 census pin d173f4c2e 不存在=本轮新欠债·§四.5 三闸零执行。律2 是柱一唯一不靠赌的核·必须还。

## 做（模块化迁移范式·g 值→描述符）
- entryLanes/groupLanes 是**每格式的 g 几何**（iq3_xxs entry=4·iq3_s/iq2_xs entry=8·grid u32 vs u64）→ 应住**描述符**（grid op 的 typed op-attr）·发射体**读**它·非就地写死。
- 范式：grid op（GgmlBlockDotIQ2XXS.../iq3 grid op）加 typed attr `codebook_entry_lanes`（或复用已有几何 attr 派生）·emitter 从 op 读 entryLanes·**去掉 body 里的 `const int64_t entryLanes = N`**。若 attr 缺席=fail-closed 带诊断（严禁 value_or 自补到 g 轴·套 ISSUE-118 放行路 B 自证制）。
- 🔴 判决实验（伪造不了·committed lit）：改 op 的 codebook_entry_lanes attr 值→regen 产物随之变（byte-diff≠0）；attr 缺席→verify-fail/诊断（反摆设）。正常值→regen-diff=0（byte-exact）。

## ★§四.5 三闸（每次 owned-emit 落地必跑·Δ≤0 才过·本役补执行痕迹）
提交带三 grep 读数：① 裸格式字面量（grid emitter 内 `= 4;`/`= 8;` 等格式常数）② 板值+value_or ③ march 解析。**相对基线只减不增**（entryLanes 迁走后①应 -N）。读数进 commit message（审计说「从没一次读数进过 commit」·本役补）。

## 触碰集
`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`（entryLanes/groupLanes→读 op-attr）+ grid op ODS（若加 attr·`include/Weft/Dialect/RVV/IR/RVVOps.td`）+ grid golden lit。🔴 禁碰：RVVLowerQuantContraction（B1 跑中·q8 measured 表）·RVVCapabilityProfile（T1）·RVVToEmitCCodebookFp4（T2）·RVVToEmitCKQuant（T3）。

## 门
- byte-exact（正常 attr 值→emit 逐字节不变·除判决实验 conflicting attr）· lit 绿 · CORE==PROD（regen md5·iq3_xxs 9d05ecad/iq3_s ca6bd415/iq2_xs 6fc3560e 正常值下不变）· 无 inline-asm · **三闸 Δ≤0 读数进 commit**。

## 交付
- **交付首节自带「判决实验 x/y + 三闸 Δ（①裸格式字面量 -N）」** + entryLanes 迁移 diff（body 焊死→op-attr 读）+ byte-exact 证（正常值 regen-diff=0）+ 文件清单。0 造数·不 git commit。
