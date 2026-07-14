# cell MANIFEST — g6-a-ime-perf-bridge / M4-parsetup

- `evidence.md`

## board harness (reproduce entry · tools/e2e-harness/board/g6-m4-ime-parsetup/)
- `tools/e2e-harness/board/g6-m4-ime-parsetup/forward-route-patch-q40-parsetup.py` (q4_0 M4 parallel-setup patcher·测过)
- `tools/e2e-harness/board/g6-m4-ime-parsetup/run-m4-q40-board.sh` (板 runner·EXIT-trap restore·md5 双证)
- `tools/e2e-harness/board/g6-m4-ime-parsetup/run-m4-bg.sh` (launcher·nohup)
- `tools/e2e-harness/board/g6-m4-ime-parsetup/agg_m4.py` (serial-setup profile-split + bundle-benefit 聚合器)
- `tools/e2e-harness/board/g6-m4-ime-parsetup/raw/run.log` (完整原始日志·920 行)

## board raw archive (raw/): run.log · prof_{onmt,onderef,onpar}.err · c_{off,onderef,onpar}.out
## board scratch (ephemeral·restored): k1:/tmp/g6m4q40 ; vendor ime.cpp/.so = reversible patch·md5 双证回基线

## ★ VERDICT
G6-A M4 串行 setup 并行化 = 第四包袱：activation quant **0.954s 串行 → 0.252s 4-hart 并行关键路径（3.79×·省 0.702s）** + lightened alloc（resize-免-memset·省 0.014s）·ith==0 crit-path **9.459s → 8.705s（1.087× Amdahl 天花板）**·**e2e onpar/onderef=1.077×**（13.067→14.071 t/s·**99% 近满传导**——matmul 塌缩后 kernel 主导 e2e·比 M3 0.87 更干净）·correctness byte-neutral GREEN（onpar==onderef==off·md5 f5e77482·vmadot=36 未加核·quant 逐行纯函数+不相交写+barrier #1b 数值中性）。曲线更新 {…0.5393×(M3)→**0.5807×(M4) stock**}（combined 47.33×·慢 stock 1.81×→1.72×）。**预注册出口 (b)**：0.5807× stock < parity·0.2974× vendor → 不转绿·IME 三格维持黄-传导稀释·**下一最大残留换位到 matmul 内部**（8.35s=onpar crit 95.9%·串行地板降到 4.1%）·M5 预注册攻 残留② 标量 epilogue 向量化。A-tree restore md5 双证 clean（ime/so==baseline·markers=0·litter=0·stray=0）。禁触域全清（零发射器改动）。
