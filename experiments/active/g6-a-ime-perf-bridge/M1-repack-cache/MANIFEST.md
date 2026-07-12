# cell MANIFEST — g6-a-ime-perf-bridge / M1-repack-cache

- `evidence.md`

## board harness (reproduce entry · tools/e2e-harness/board/g6-m1-ime-repack-cache/)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/forward-route-patch-q40-cache.py` (q4_0 M1 patcher·测过)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/forward-route-patch-q4k-cache.py` (q4_K·就绪未测)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/forward-route-patch-q80-cache.py` (q8_0·就绪未测)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/run-m1-q40-board.sh` (板 runner·EXIT-trap restore)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/run-m1-bg.sh` (launcher)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/agg_m1.py` (12-sample 聚合器)
- `tools/e2e-harness/board/g6-m1-ime-repack-cache/raw/run_q4_0_m1.log` (完整原始日志)

## board scratch (ephemeral·restored): k1:/tmp/g6m1q40 ; vendor ime.cpp/.so = reversible patch·md5 双证回基线

## ★ VERDICT
G6-A M1 repack 缓存化 = 真但小(桥 1.138×·e2e 0.0128×→0.0145×·装载期 12.79s 单列)·correctness byte-neutral·IME 三格维持黄-传导稀释未改标。X-0 profile 纠正过归因(repack 11.8% 非主导·matmul 88% 单线程主导)→M2 多线程=最大杠杆。
