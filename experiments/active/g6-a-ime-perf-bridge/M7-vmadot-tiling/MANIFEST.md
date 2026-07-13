# cell MANIFEST — g6-a-ime-perf-bridge / M7-vmadot-tiling

- `evidence.md`

## emitter change (source-of-truth · lib/ · reproduce via weft-opt)
- `lib/Plugin/IME/IMEBackendEmissionDriver.cpp` (WIDE vmadot leaf 生成器 macKloopHelperBodyWide + [PAT-1] 数据表 kIMEVmadotTilingPatterns + capability-keyed selectVmadotTilePattern·核 baseline/q4_0/q8_0/q4_K helper 逐字不动·weft-opt build-clean)
- `test/Conversion/EmitC/ime-q4-0-matmul-tile-materialization.mlir` (尾部 M7 wide-leaf EMITC 检·existing 检未动·lit PASS)
- `test/Target/IME/q4-0-vmadot-tile-wide-int32-oracle.c` (host oracle·wide int32==width-1==ZERO-MODEL·ORACLE PASS)

## board harness (reproduce entry · tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/forward-route-patch-q40-tilew.py` (q4_0 M7 WIDE-tiling patcher: vmadot_mac_kloop_w2/w4 leaves + wide nj-outer vec matmul + wide compute-isolation profiling·测过·build-clean 0-err)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/run-m7-q40-board.sh` (板 runner·EXIT-trap restore·md5 双证)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/agg_m7.py` (array-util compute-decomposition + M7 benefit 聚合器)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/raw/run.log` (完整原始日志·840 行)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/raw/k1_wide_unit_bytexact.c` (K1 单测·real vmadot·w2/w4 int32==baseline memcmp=0)

## board raw archive (raw/): run.log · prof_{compute_l1,compute_l1_w2,compute_l1_w4,onjout_full,onw2_full,onw4_full}.err · c_{off,onjout,onw2,onw4}.out · k1_wide_unit_bytexact.c
## board scratch (ephemeral·restored): k1:/tmp/g6m7q40 ; vendor ime.cpp/.so = reversible patch·md5 双证回基线

## ★ VERDICT
G6-A M7 vmadot-tiling（**首个 G6-A 发射器改动·最后深潜·令一.3**）：M6 后 vmadot COMPUTE(1.741s/33.8%)=唯一剩 actionable 大杠杆（厂商 1.996× 同硅同 vmadot=阵列 ~2× headroom）。**WIDE 输出 tiling**：一次 vle8 载 4×8 A(v0)·喂 NJW 个独立 vmadot 链入 NJW 个 4×4 int32 累加器（隐藏阵列延迟 + A-load/入口摊薄）·**纯 schedule 变·0xe210312b vmadot 与累加序不动 → byte-exact 由构造**。**★array-util 板测确诊（compute-isolation·固定 L1）：vmadot COMPUTE 1.740s(w1)→1.095s(w2·1.589×)→0.890s(w4·1.955×=厂商 headroom 兑现)**；**matmul 5.348→4.614s(w2·1.159×)**；**e2e pp32 onw2=23.881 t/s=1.0088× stock=≥PARITY（跨线·预注册出口 a）·onw2/onjout=1.1385× vs M6（填平 M6 10% parity 缺口·1.159× matmul 98% 传导）**。**w4 measured-negative**（compute 1.955× 真·full-matmul 0.972× NULL·16-累加器 f32 epilogue-spill·边界档案化）→ **shipped=W2**。**byte-exact 硬门 GREEN·三重证**：K1 单测 w2/w4 int32==baseline(memcmp=0·real vmadot)·host oracle int32==ZERO-MODEL·板 e2e md5 onw2==onw4==onjout==off==**f5e77482**（M1-M6 sealed 全谱系）。**发射器 [PAT-1] 注册**（数据表 W1/W2 mechanized·W4 measured-negative + capability-keyed selector[vreg budget] + materialization lit + host oracle）=C3′ 能力键控优化模式库首个 IME array-util 条目（schema JSON 行 coordinator 补·本 agent 不触 schema）。**成色（honest）**：q4_0@ime 板测 ≥parity（对手=stock·statistical tie·vs-vendor 0.5054× 差距不隐瞒）→ 成色质变候选；**q8_0/q4_K@ime 共用同 leaf 但未各自板测→不自动转绿（deployment≠proven·perf-covered 最终计数=coordinator 裁）**。A-tree restore md5 双证 clean（ime/so==baseline·markers=0·litter=0·stray=0·EXIT-trap + 独立第二证）。IME 冲刺止于 M7（令一.3）。
