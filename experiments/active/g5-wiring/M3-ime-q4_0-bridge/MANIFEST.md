# MANIFEST — G5-M3 IME q4_0 forward bridge 曳光弹（session 1）

> one-file-per-backtick-bullet · md5 pinned · **无 git**（主会话 commit）· HEAD 未变
> session 1 交付 = bridge #1 scale-fold epilogue + #2 runtime shape · correctness GREEN（host + K1 硅）· emitter 实发 + lit · **partial·多-session**（#3/#4/#5 forward-hook = next-session）

## lib/ 桥代码（emitter 扩展 · 我独占 lib/）
- `lib/Plugin/IME/IMEBackendEmissionDriver.cpp` — md5 `d5bd6e48fc4e5f5df17b3e444822a574` · 新 `q40ScaleFoldMatmulHelperBody()` + `kQ40ScaleFoldMatmulHelperName` · `IMEQ40MatMulTileToEmitCFunc` 追加第二 `extern "C"` wrapper `..._slice_f32`（int32 wrapper/seal object 不动）

## ODS（本 session 未改 · 记录基线 md5）
- `include/TianChenRV/Dialect/IME/IR/IMEOps.td` — md5 `5fd2d3938c9cb9ecfd2a9d7f448e8fab` · **UNCHANGED**（scale-fold 走既有 Q40MatMulTileOp 追加 wrapper·无新 op/attr·q4_K emitter 已 emit f32 fold 之体例先例）

## correctness harness（host oracle + K1 硅 seal）
- `test/Target/IME/q4-0-matmul-tile-scalefold-oracle.c` — md5 `84a2e1acd70942c4a486ccd0e5cbef25` · host ZERO-MODEL：int32 core exact + scale-fold vs canonical q4_0×q8_0（bounded-ULP）·4 shape
- `test/Target/IME/q4-0-matmul-tile-scalefold-k1seal.c` — md5 `1c1ea33b4d4ec73de6c883400927f1e6` · = oracle 逐字·唯一差异 EMITTER-VERBATIM `vmadot` asm 叶子·真 K1 硅

## lit golden（emitted 桥 · byte/token-exact vs harness）
- `test/Conversion/EmitC/ime-q4-0-matmul-tile-materialization.mlir` — md5 `2ce6c0522519ca58aa4c991ec1bb6865` · 扩展 EMITC：int32 kernel+wrapper → `scale_fold_epilogue` verbatim（`Cf[...] += dA[...]*dW[...]`）→ 第二 wrapper `..._slice_f32` + `call_opaque "..._matmul_f32"`（REGION + EMITC 全绿·q8_0/q4_K/mma/matmul 回归绿·经 clean-rebuild tcrv-opt 验证）

## board harness
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-scalefold-seal.sh` — md5 `572cdfbd62e9308498b1d910c5f0f9ea` · k1 build+objdump(`vmadot` engage)+`taskset -c 0-3` run · 不触 vendor ggml

## casefile
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/evidence.md` — md5 `62f9d78a91fc5cb92a074c3866821390` · verdict + 设计 + correctness 证据 + 双账本 + next-step
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/host-oracle.txt` — md5 `3d665987d8b9dd5b2bcfe74eaf54d8fc` · host ORACLE PASS 原始
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/board-k1-seal.txt` — md5 `ba8fae59fbd5a2bd95ea65daf296c15f` · K1 SEAL PASS 原始（vmadot_count=2·exit=0）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/emitted-f32-helper.c` — md5 `995b6dfd7ed754b017c329736932e2d0` · tcrv-opt 实发 f32 kernel（解码 verbatim）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/harness-f32-helper.c` — md5 `458a2dec6bd4b2ca5746fdff1b8c4874` · harness f32 kernel（token-identical 于 emitted·仅注释/换行差）

## build（我独占 · 绝不动 build/）
- `build-ime-bridge/` — host tcrv-opt（系统 LLVM-20·ninja·clean）· **gitignored**（未列入 casefile·仅本机验证 lit）

---

# MANIFEST append — session 2（集成层 4 缺口·forward integration）

> session 2 交付 = #4 权重 repack 桥 + #3 激活 quant/pack 桥（板 UT vs 真 ggml native byte-exact·MIRAGE 过）· 整桥单 tensor mul_mat A==B（bit-exact float order·真硅 vmadot·随机+真模型 tensor）· #5 forward hook = reversible reachability probe（q4_0 PREFILL hook 在真 llama forward fires·非 traffic-routing）· **board restored md5 双证零 stock 改动** · **partial·多-session**（#5 full 路由 + 真-llama e2e = next-session）
> **无 git**（主会话 commit）· HEAD 未变 · session-1 evidence.md 主体禁改（未动）

## board harness（session 2·我独占 tools/e2e-harness/board/g5-m3-ime-q4_0/）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/g5m3_bridge_ut.c` — md5 `5b4b1f4ef134b67b4387e0a473d2ccfa` · #4 权重 repack + #3 激活 quant/pack byte-exact vs 真 ggml native（`quantize_row_q4_0_ref`/`quantize_row_q8_0_ref`/`dequantize_row_q4_0` link libggml-base）+ 整桥单 tensor A==B vs stock q4_0×q8_0 块-dot（`max_abs_vs_float_order=0`·normwise ~1e-7）· EMITTER-VERBATIM `vmadot` 叶子（0xe210312b·silicon seal）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/g5m3_realtensor_ab.c` — md5 `1df5d8cc1d9b3c1af48d44a99f75aa42` · 真模型 tensor A==B（gguf API 抽 `tinyllama-q4_0.gguf` 的 `token_embd.weight` 真 native 字节·N=64×K=512 sub-tile·bit-exact float order·真硅 vmadot）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-bridge-ut.sh` — md5 `16d534939a1cd2480acaeabb0e9199c2` · build+objdump(vmadot)+`taskset -c 0-3` 跑 g5m3_bridge_ut + g5m3_realtensor_ab · 不触 vendor ggml
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-forward-probe.sh` — md5 `9ed68dbe71f6787b21d657ff94b84fa8` · **reversible** vendor forward-hook reachability probe driver（backup→patch→rebuild ggml-cpu→llama-bench prefill ON/OFF→restore clean source+.o+ORIG .so binary→md5 双证零改动→删 .ORIG litter）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/forward-probe-patch.py` — md5 `b96f9d6fe14483cbe75fe3c7b45b292e` · env-gated（`TCRV_IME_BRIDGE_PROBE`）q4_0-only prefill-only one-shot banner 插入 `forward_mul_mat` gemm_n 后（默认 OFF=零行为改动·post-write self-verify）

## casefile（session 2）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/session2_forward-integration.md` — md5 `cbf7e5ff586c1394a32dc59617f265cd` · verdict + #4/#3/④ correctness 证据 + #5 reachability probe + board-restored md5 双证 + 诚实边界 + next-step
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/session2-bridge-ut.txt` — md5 `f9966f05bd76d8839523f62beead77ac` · BRIDGE-UT PASS 原始（#4/#3 byte-exact·full A==B·vmadot_count=1·ut_exit=0）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/session2-realtensor-ab.txt` — md5 `e141befe113468437f5a12c89b37f6da` · REALTENSOR PASS 原始（token_embd.weight·rt_exit=0）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/session2-forward-probe.txt` — md5 `d1e05828118cec8252f18c063332c9f9` · probe banner fires（gemm_m=64 n=2048 k=2048·use_ime1:1）+ control OFF banner=0

## board vendor（read-only 对照 + #5 probe 目标·测后 restored·md5 零改动·stock 不留改动）
- `/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/spacemit/ime.cpp` — restored md5 `40962c7e7c732bf472ae88cef89ced8d`（==pre-session baseline·probe 残留=0）
- `/home/bianbu/tcrv-k1-llama/build-ime/bin/libggml-cpu.so.0.15.1` — restored md5 `71cc4d295dac29382a0a7d4d5bd0c425`（==pre-session baseline·probe strings=0·vmadot=32 intact）

---

# MANIFEST append — session 3（forward traffic routing 收口·forward-wired F→T）

> session 3 交付 = **真 q4_0 prefill 流量路由经我方 fragment-major scale-fold vmadot IME 核**（env-gated parallel tcrv `tensor_traits`·passthrough repack + prefill mul_mat 拦截·5/5 banner·shipped .so vmadot 32→33）+ 真-llama e2e greedy A==B（ON vs vendor IME 4/5 逐字同 / vs stock RVV 3/5·分歧=near-tie argmax flip·vendor IME 自身亦偏离 stock·MIRAGE 决定性 de-risk·生成全连贯 in-family）· **board restored md5 双证零 stock 改动（EXIT-trap 强制）** · **forward-wired=T**（跨范式 family#2 forward 最后一里）· correctness 硬锚仍 session-2 单 tensor bit-exact seal
> **无 git**（主会话 commit）· HEAD 未变 · session-1/2 evidence.md/session2_forward-integration.md 主体禁改（未动）

## board harness（session 3·我独占 tools/e2e-harness/board/g5-m3-ime-q4_0/）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py` — md5 `3b183d8ac09193ce81047aa8a79a009c` · **reversible** env-gated（`TCRV_IME_Q40_BRIDGE`）parallel tcrv `tensor_traits` 注册 patcher：插入 `tcrv_q4_0_tensor_traits`（passthrough native repack + prefill mul_mat 拦截跑 EMITTER-VERBATIM #4/#3/scale-fold vmadot 核）+ `get_optimal_repack_type` Q4_0 env-gated 选择（env OFF=vendor 字节不变）·post-write self-verify（markers=2/trait=1/selection=1）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-forward-route.sh` — md5 `469899e960d3f73517a6140ea8fcd862` · 单 prompt 三方 greedy A==B（OFF build-off / VEN build-ime env-off / ON build-ime env-on·`llama-completion`）+ objdump vmadot + banner·**EXIT-trap 强制 restore**（clean source→clean .o→ORIG .so binary→md5 双证零改动·清 .ORIG）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-forward-route-multi.sh` — md5 `77a993c14134f1d96d8922ca7221bb8f` · 多 prompt（5）A==B seal·逐 prompt ON==VEN / ON==OFF / banner·**EXIT-trap restore**

## casefile（session 3）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/session3_forward-wired.md` — md5 `f2b39b0918de3414d3ccdbc878d92ab6` · verdict + 路由方案（env-gated parallel tcrv tensor_traits·passthrough correctness-safe）+ forward-wired=T 证据（banner+vmadot）+ e2e A==B（ON vs vendor IME 4/5 / stock RVV 3/5·near-tie flip 分析·MIRAGE de-risk）+ board-restored md5 双证 + 诚实边界 + next-step
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/session3-forward-route.txt` — md5 `52465c400cc6ea06c2774bfddb24caa8` · 单 prompt 原始（banner M=15 N=2048 K=2048·vmadot 32→33·ON==VEN IDENTICAL·ON==OFF DIFFER near-tie·RESTORE md5 ZERO-CHANGE·litter 0）
- `experiments/active/g5-wiring/M3-ime-q4_0-bridge/raw/session3-multiprompt-ab.txt` — md5 `ab9baf16a022aebe09d4a3265375da60` · 5 prompt A==B 原始（ON==VEN 4/5·ON==OFF 3/5·BANNER_TOTAL=5·逐 prompt OFF/ON 生成·RESTORE md5 双证零改动）

## board vendor（read-only 对照 + session-3 routing patch 目标·测后 restored·md5 零改动·stock 不留改动）
- `/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/spacemit/ime.cpp` — restored md5 `40962c7e7c732bf472ae88cef89ced8d`（==pre-session baseline·route-marker 残留=0）
- `/home/bianbu/tcrv-k1-llama/build-ime/bin/libggml-cpu.so.0.15.1` — restored md5 `71cc4d295dac29382a0a7d4d5bd0c425`（==pre-session baseline·route strings=0·vmadot 回 32 intact）
