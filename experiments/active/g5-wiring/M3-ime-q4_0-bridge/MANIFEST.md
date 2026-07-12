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
