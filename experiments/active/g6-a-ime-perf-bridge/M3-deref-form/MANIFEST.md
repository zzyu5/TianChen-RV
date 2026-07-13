# cell MANIFEST — g6-a-ime-perf-bridge / M3-deref-form

- `evidence.md`

## board harness (reproduce entry · tools/e2e-harness/board/g6-m3-ime-deref/)
- `tools/e2e-harness/board/g6-m3-ime-deref/forward-route-patch-q40-deref.py` (q4_0 M3 de-reference-form patcher·测过)
- `tools/e2e-harness/board/g6-m3-ime-deref/run-m3-q40-board.sh` (板 runner·EXIT-trap restore·md5 双证)
- `tools/e2e-harness/board/g6-m3-ime-deref/run-m3-bg.sh` (launcher·nohup)
- `tools/e2e-harness/board/g6-m3-ime-deref/agg_m3.py` (profile-split + bundle-benefit 聚合器)
- `tools/e2e-harness/board/g6-m3-ime-deref/raw/run.log` (完整原始日志)

## board raw archive (raw/): run.log · prof_{onmt,onderef_dq,onderef}.err · c_{off,onmt,onderef}.out
## board scratch (ephemeral·restored): k1:/tmp/g6m3q40 ; vendor ime.cpp/.so = reversible patch·md5 双证回基线

## ★ VERDICT
G6-A M3 去参考形态 = 三包袱**单步最大杠杆**：matmul 关键路径 **11.49× internal**（缓存已解 B 86.7% + 寄存器 epilogue 4.6%·核 vmadot/标量算术逐字不变）·**e2e onderef/onmt=9.98×**（1.31→13.09 t/s）·correctness byte-neutral GREEN（onderef==onmt==off·vmadot=36）。曲线终点 {0.0126→0.0143→0.0554→**0.5529× stock**}（44× total·慢 stock 79×→1.81×）。**预注册出口 (b)**：0.5529× stock < parity·0.2769× vendor → 不转绿·IME 三格维持黄-传导稀释·剩余 1.81× 逐项工程归因（串行 setup/标量 epilogue/deref-cache 带宽/tiling/barrier·禁物理墙·厂商 1.997× 反证）。A-tree restore md5 双证 clean（ime/so==baseline·markers=0·litter=0·stray=0）。禁触域全清。
