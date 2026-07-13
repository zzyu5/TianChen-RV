# cell MANIFEST — g6-a-ime-perf-bridge / M5-matmul-internal

- `evidence.md`

## board harness (reproduce entry · tools/e2e-harness/board/g6-m5-ime-matmul-internal/)
- `tools/e2e-harness/board/g6-m5-ime-matmul-internal/forward-route-patch-q40-matmul.py` (q4_0 M5 matmul-internal patcher: vec epilogue + 2 profiling 变体·测过·build-clean 0-err)
- `tools/e2e-harness/board/g6-m5-ime-matmul-internal/run-m5-q40-board.sh` (板 runner·EXIT-trap restore·md5 双证)
- `tools/e2e-harness/board/g6-m5-ime-matmul-internal/run-m5-bg.sh` (launcher·nohup)
- `tools/e2e-harness/board/g6-m5-ime-matmul-internal/agg_m5.py` (matmul-internal 分解 + epilogue-vec benefit 聚合器)
- `tools/e2e-harness/board/g6-m5-ime-matmul-internal/raw/run.log` (完整原始日志·672 行)

## board raw archive (raw/): run.log · prof_{onpar_full,onpar_noepi,onpar_nomad,onepivec}.err · c_{off,onpar,onepivec}.out
## board scratch (ephemeral·restored): k1:/tmp/g6m5q40 ; vendor ime.cpp/.so = reversible patch·md5 双证回基线

## ★ VERDICT
G6-A M5 matmul 内部分解 + 标量 epilogue 向量化：板测 profile 认瓶颈 = **① vmadot 计算+B/A feed 5.763s（68.5%·最大）· ② 标量 f32 epilogue fold 2.646s marginal（31.5%·次大·bit-exact 可攻）**。攻 ②：4×4 tile fold **4-wide 向量化 across 4 独立输出列**（每列跨 b 序不变=bit-exact·逐 lane 复刻标量 emit `vfmul.vf(dA,dW)+vfcvt.f.x(frag)+vfmacc.vv(融合==fmadd.s)`·板 objdump 确诊融合形态）·4 行累加器 register-resident 跨 b（objdump v9-v12 零 spill）。**标量 epi marginal 2.646s → vec 1.410s（1.88× 更紧）· matmul 8.409→7.173s（1.172×）· e2e onepivec/onpar=1.1423×**（14.10→16.11 t/s·**97.5% 近满传导**）·correctness **byte-identical**（onepivec==onpar==off·md5 f5e77482 同 M1-M4 sealed 全谱系·vmadot=38 核字节不变）。曲线 {…M4 0.5968×→**M5 0.6817× stock**}（慢 stock 1.68×→1.47×）。**预注册出口 (b)**：0.6817× stock < parity·0.3412× vendor → 不转绿·IME 三格维持黄-传导稀释。**下一最大残留换位到 vmadot 计算+feed（5.763s=新 matmul 80.3%）·两子路：(a) B-feed loop-interchange（orchestration·无发射器·bit-exact·候选 M6·先 profile feed 份额）· (b) vmadot array-utilization（★须改发射器 vmadot 核 tiling→已停·报主会话协调）**。A-tree restore md5 双证 clean（ime/so==baseline·markers=0·litter=0·stray=0·独立第二证）。禁触域全清（零发射器改动·同 M1-M4）。
