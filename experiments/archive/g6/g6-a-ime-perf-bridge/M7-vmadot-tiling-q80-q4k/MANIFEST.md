# cell MANIFEST — g6-a-ime-perf-bridge / M7-vmadot-tiling-q80-q4k

- `evidence.md`

## board harness (reproduce entry · tools/e2e-harness/board/)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q80/forward-route-patch-q80-tilew.py` (q8_0 M7 full-bridge WIDE-tiling patcher: int8-direct gather + q4_0-verbatim int8-reading deref/epi/vec/njouter/w2/w4 family + wide compute-isolation profiling·板 build-clean 0-err·routes real q8_0)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q80/run-m7-q80-board.sh` (q8_0 板 runner·off/ven/onjout/onw2/onw4·EXIT-trap restore md5 双证)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q80/raw/run-q80-board.log` (q8_0 完整原始日志)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q4k/forward-route-patch-q4k-tilew.py` (q4_K M7 full-bridge WIDE-tiling patcher: super-block two-level fold deref cache[Bdec+sc/mm/d/dmin] + asum 预算 + 4-wide vectorized two-level epilogue + wide W2·板 build-clean 0-err·routes real q4_K)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q4k/run-m7-q4k-board.sh` (q4_K 板 runner·off/ven/onjout/onw2·EXIT-trap restore md5 双证)
- `tools/e2e-harness/board/g6-m7-ime-vmadot-tiling-q4k/raw/run-q4k-board.log` (q4_K 完整原始日志)

## board raw archive (raw/): run-q80-board.log · run-q4k-board.log (含 PROFILE/CORRECTNESS/PERF 全段·banner·md5·restore)
## board scratch (ephemeral·restored): k1:/tmp/g6m7q80 · k1:/tmp/g6m7q4k ; vendor ime.cpp/.so = reversible patch·md5 双证回基线 (40962c7e/71cc4d29)
## models (board /data·公开可得·within-format apples-to-apples): tinyllama-q8_0.gguf (1.09 GiB) · tinyllama-1.1b-Q4_K_M.gguf (636 MiB)

## ★ VERDICT
G6-A M7 wide-vmadot-tiling 铺到 IME 三格剩二（projection→proof·各格独立板测独立 seal·deployment≠proven 纪律）。**format-agnostic MAC**（同一 vmadot 0xe210312b leaf·wide W2 逐字同 q4_0 M7）·核 leaf 与累加序不动 → byte-exact 由构造（onw2==onjout wide-neutral 硬门 + 对 stock byte-exact 两格 GREEN）。**★分裂结果**：**q8_0@ime onw2=23.74 t/s=2.233× stock（碾压·无 IQR 重叠·非 tie）·2.269× 赢 vendor（vendor 弱 0.984× stock）→ 转绿·perf-covered 8→9/83**（成色强于 q4_0=beat 双方 vs tie/lose·诚实前提=所赢 vendor 的 q8_0 路弱·内存税无：q8_0 int8-direct deref 无膨胀·区别 q4_0 ×1.78）。**q4_K@ime onw2=13.28 t/s=0.909× stock < parity（decisive loss）→ 维持黄-传导稀释**（array-util 杠杆不足：two-level fold[scale+min] 大 non-vmadot 分量·compute 1.561× 传导到 e2e 仅 1.024×·stock q4_K 已快 14.61·vendor 强 1.493×）。**array-util compute-isolation ~1.56-1.59× 两格一致**（== q4_0·同 leaf·format-invariant）·**full-matmul 传导分裂**（q8_0 1.106× vs q4_K 1.045×）= wide-tiling 的 **format-keyed 适用边界实证**（C3′ 优化模式库：q8_0-keyed green-carrier·q4_K-keyed transduction-insufficient 负结果条目·同价值·设计空间资产）。A-tree restore md5 双证 clean 两格（ime/so==baseline·markers=0·litter=0·stray=0·EXIT-trap + 独立第二证）。schema label + perf-covered recon + [PAT-1] JSON 行 = 主会话补（本 agent 不触 schema·recon 机算）。
