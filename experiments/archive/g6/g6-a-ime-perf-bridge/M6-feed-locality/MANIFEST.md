# cell MANIFEST — g6-a-ime-perf-bridge / M6-feed-locality

- `evidence.md`

## board harness (reproduce entry · tools/e2e-harness/board/g6-m6-ime-feed-locality/)
- `tools/e2e-harness/board/g6-m6-ime-feed-locality/forward-route-patch-q40-feedloc.py` (q4_0 M6 B-feed-locality patcher: nj-outer vec matmul + compute-isolation/nj-outer profiling 变体·测过·build-clean 0-err)
- `tools/e2e-harness/board/g6-m6-ime-feed-locality/run-m6-q40-board.sh` (板 runner·EXIT-trap restore·md5 双证)
- `tools/e2e-harness/board/g6-m6-ime-feed-locality/run-m6-bg.sh` (launcher·nohup)
- `tools/e2e-harness/board/g6-m6-ime-feed-locality/agg_m6.py` (compute-vs-feed 分解 + interchange benefit 聚合器)
- `tools/e2e-harness/board/g6-m6-ime-feed-locality/raw/run.log` (完整原始日志·675 行)

## board raw archive (raw/): run.log · prof_{onepivec_full,onepivec_noepi,compute_l1,onjout_full,onjout_noepi}.err · c_{off,onepivec,onjout}.out
## board scratch (ephemeral·restored): k1:/tmp/g6m6q40 ; vendor ime.cpp/.so = reversible patch·md5 双证回基线

## ★ VERDICT
G6-A M6 B-feed locality：**compute-isolation 探针（固定 L1 scratch）板测确诊 feed 是真瓶颈——vmadot 计算仅 1.741s（23.4%）· B/A feed 5.708s（76.6%）**（M5 假设兑现·非臆断）。攻 feed：**loop interchange `mi{nj{b}}→nj{mi{b}}`（nj-outer·B 每列 tile L1 流一次跨 mi 复用·A L1-resident）·纯 tile-order 重排=bit-exact 由构造**（每 Cf[m,n] 跨 b 序不变·单 writer）。**feed 5.708→2.064s（2.77× 减）· matmul 7.386→5.149s（1.435×·省 30.3%）· e2e onjout/onepivec=1.3637×**（16.05→21.88 t/s·**95% 近满传导**）·correctness **byte-identical**（onjout==onepivec==off·md5 f5e77482 同 M1-M5 sealed 全谱系·vmadot=41 核字节不变）。曲线 {…M5 0.6817×→**M6 0.9042× stock**}（慢 stock 1.51×→**1.11×·逼近 parity**）。**预注册出口 (b·增量显著逼近 parity)**：0.9042× stock < parity（差 10%）·0.4633× vendor → 不转绿·IME 三格维持黄-传导稀释。**下一最大残留换位到 vmadot 计算/array-utilization（1.741s=新 matmul 33.8%）→ 须改发射器 vmadot 核 tiling（M7·请协调）·feed_nj（40.1%）已近 DRAM once-through 地板（orchestration 耗尽）**。A-tree restore md5 双证 clean（ime/so==baseline·markers=0·litter=0·stray=0·独立第二证 + coordinator ssh 复证）。禁触域全清（零发射器改动·同 M1-M5）。
