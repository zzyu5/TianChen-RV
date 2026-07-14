# G5-M3 — IME triple **e2e ON/OFF paired perf**（q4_0·q8_0·q4_K @ime · 转正 provisional → 出口 · k1）

> **campaign**: G5 接线战役 · **M3-ime-e2e-pairing** = ratified IME triple forward-wired 之后的 **perf 侧补测**（用户 2026-07-12 裁一.3·转正 provisional·板批捎带）。
> **立项理由（裁一.1/一.2）**：三格（q4_0@ime `9aeaab2d` / q8_0@ime `d73096da` / q4_K@ime `1cb56fc7`）已 forward-wired 但**打回 provisional**（仅正确性证据·无实测 e2e perf 配对·"传导稀释"标签必须携 `measured_e2e_delta_pointer`+`amdahl_pointer`·缺→auto-provisional）。correctness 侧 logit-ULP 已闭（`8e94c120`·within-ULP·near-tie correctness-neutral）。**本任务补 perf 侧实测 e2e Δ → 转正。**
> **board**: `ssh k1`（SpacemiT X60·VLEN256·IME harts 0–3·`taskset -c 0-3`·threads 4·governor=performance·freq locked 1.6 GHz）·出货 **clang-18 两 build 对称**（kernel==system 收敛·双账本同数）·models = tinyllama-q4_0.gguf / q8_0.gguf / 1.1b-Q4_K_M.gguf（board-resident·复用）。
> **禁 git**（主会话 commit·本 casefile = 证据 + verdict）·**board 可逆**（vendor `cp *.ORIG` + EXIT/INT/TERM-trap restore + md5 双证零改动）·**禁 llama-cli**（用 llama-bench）·**MANIFEST/gitignore**。

---

## 0. 一页速览（verdict·三格逐格·出口=黄-传导稀释 finalized）

**A/B = 同会话开/关 IME 路由（llama-bench·n=10/side·pp32 prefill + tg64 decode 分相·interleaved on/ven/off × 2 pass·freq 恒 1.6 GHz）。banner on=2 / ven=0 / off=0（三格皆·证 ON 真路由我方 IME 核·VEN/OFF 不路由）。**

| 格 | phase | ON/OFF (tcrvIME / stockRVV) | ON/VEN (tcrvIME / vendorIME) | VEN/OFF (vendor 头顶空间) | **出口** |
|---|---|---|---|---|---|
| **q4_0**（flat·nibble） | **prefill pp32** | **0.0128×** | 0.0065× | **1.977×**（真 vendor 头顶） | **黄-传导稀释** |
|  | decode tg64 | 0.715× | 0.515× | 1.389× | （native 对照） |
| **q8_0**（flat·int8-direct） | **prefill pp32** | **0.166×** | 0.167× | **0.992×**（物理墙·vendor=stock） | **黄-传导稀释** |
|  | decode tg64 | 0.868× | 0.876× | 0.991× | （native 对照） |
| **q4_K**（super-block·两级 fold） | **prefill pp32** | **0.0617×** | 0.0414× | **1.493×**（真 vendor 头顶） | **黄-传导稀释** |
|  | decode tg64 | 0.618× | 0.654× | 0.945× | （native 对照） |

**compute-account（kernel micro）增益锚 = ~2.09×**（tcrv IME register-resident batched `vmadot` leaf vs RVV-vector-path·k1 硅·sealed `[GAP-IME-LEAF-PIPELINE]`）。
**三格 prefill e2e ON/OFF（0.0128× / 0.166× / 0.0617×）≪ compute-account 2.09×** → **micro↛e2e·传导稀释 finalized**（两数齐：kernel 增益 2.09× × e2e 稀释实测）。

**一句话**：三格 IME forward bridge 在真 K1 llama e2e prefill 上**不是稀释到 parity·而是净退化**（ON/OFF 0.013–0.166×·比 stock 慢 6–79×）——2.09× 的 compute-account `vmadot`-leaf 增益被**桥集成开销 Amdahl-淹没**（单线程 `ith==0` 只 1/4 hart 算 + **每次 prefill 调用重新 repack 权重**（native→fragment-major·未缓存）+ 朴素 scale-fold epilogue·`vmadot` 微核在桥墙钟里可忽略）。**非物理墙**：**vendor IME 在同模型/同板/同工具链上给出真 e2e prefill 增益（q4_0 1.98× · q4_K 1.49×）**证 prefill regime 有头顶空间（q8_0 例外·vendor=stock 0.99× = q8_0 prefill 本身内存墙）。decode 三格 ON<OFF（0.62–0.87×）——decode M=1 从不路由我方核（native fallback·设计如此·对照），但 env-gated native-passthrough 权重布局把 decode 逼上比纯 stock RVV 更慢的路径 → 轻退化。**板 stock 零改动（md5 双证·三格各 EXIT-trap + 终审独立复核）。八门：perf-axis·kernel-vs-opponent 仅编译器对称有效——此处 on/ven/off 全 clang-18（kernel==system 对称✓）·但结果=LOSS/稀释·非绿·非 perf-covered。**

---

## 1. 方法（同会话开/关 IME 路由配对 llama-bench）

### 1.1 三 side（复用三格 forward-route env-gate patch·同 build-ime 二进制 env-toggle + build-off 参照）
- **ON** = `build-ime` `TCRV_IME_{Q40,Q80,Q4K}_BRIDGE=1` —— 我方 tcrv IME 核路由真 prefill mul_mat（banner FIRES·`.so` vmadot 32→33）。
- **VEN** = `build-ime` env unset —— vendor IME 路径（IME-buffer 格的**部署基线**·同二进制 env-toggle·byte-identical A/B）。
- **OFF** = `build-off` —— stock RVV vec_dot（无 IME）= 2.09× compute-account 的 **RVV-vector 基**（稀释比对同基）。
- 三 side **interleaved per pass**（drift cancel）· `-p 32`（prefill·M=32>1 → IME 阵列）· `-n 64`（decode·M=1 → native fallback 对照）· `-r 5 -o json` · **PASSES=2 → n=10/side**。

### 1.2 regime（★裁一.3 注意）
forward-wiring **只路由 prefill**（`compute_forward` 判据 `M(ne11)>1`）·decode M=1 有意 native fallback。故 **e2e prefill = IME 核实测**·**decode 走 native（对照·我方核缺席）**。八门内适用项·双账本 clang 对称收敛（on/ven/off 全 clang-18·kernel==system·同数注记）。

### 1.3 DVFS / board-load / 可逆
- governor=performance·4 pinned hart（0-3）恒 1.6 GHz（freq 每 call 捕获·三格 side-freqs 全 `{1600000}` 单值·无 DVFS 漂移）。
- board-load gate = informational（consumers 住 hart 5-7·pinned 0-3 空闲·interleaved A/B 对消对称背景负载；load 记录见 run log）。
- 每格 EXIT/INT/TERM-trap restore：clean source → `make ggml-cpu`（clean .o）→ `.so` 覆盖回 ORIG binary → md5 双证。

---

## 2. 逐格结果（median t/s·n=10/side·IQR 见 raw/agg_*.txt）

### 2.1 q4_0（flat·nibble·tinyllama-q4_0.gguf）
```
pp32  ON=0.295  VEN=45.64  OFF=23.09   |  ON/OFF=0.0128x  ON/VEN=0.0065x  VEN/OFF=1.977x
tg64  ON=3.89   VEN=7.56   OFF=5.44    |  ON/OFF=0.715x   ON/VEN=0.515x   VEN/OFF=1.389x
```
- prefill 我方桥 **0.295 t/s vs stock 23.09 t/s = 慢 79×**（IQR 极紧·ON side 确定性·CV~0.06%）。
- **VEN/OFF=1.98×** = vendor IME q4_0 真 e2e prefill 增益（头顶空间存在·regime 非墙）。
- **出口 = 黄-传导稀释**（0.0128× ≪ 2.09×·净退化）。

### 2.2 q8_0（flat·int8-direct·tinyllama-q8_0.gguf）
```
pp32  ON=1.726  VEN=10.31  OFF=10.40   |  ON/OFF=0.166x   ON/VEN=0.167x   VEN/OFF=0.992x
tg64  ON=3.667  VEN=4.186  OFF=4.224   |  ON/OFF=0.868x   ON/VEN=0.876x   VEN/OFF=0.991x
```
- prefill 我方桥 **1.726 vs stock 10.40 = 慢 6.0×**（int8-direct 无 nibble decode·比 q4_0 桥快·但仍稀释）。
- **VEN/OFF=0.99× = 物理墙**：vendor IME q8_0 = stock parity（q8_0 prefill 本身内存墙·无 IME 头顶空间·与 correctness 案 vendor q8_0 IME≡stock bit-identical 一致）。→ q8_0 = 稀释-于-walled-regime。
- **出口 = 黄-传导稀释**（0.166× ≪ 2.09×）。

### 2.3 q4_K（super-block·两级 fold·tinyllama-1.1b-Q4_K_M.gguf）
```
pp32  ON=0.882  VEN=21.32  OFF=14.28   |  ON/OFF=0.0617x  ON/VEN=0.0414x  VEN/OFF=1.493x
tg64  ON=4.280  VEN=6.547  OFF=6.924   |  ON/OFF=0.618x   ON/VEN=0.654x   VEN/OFF=0.945x
```
- prefill 我方桥 **0.882 vs stock 14.28 = 慢 16×**（super-block 两级 fold + q8_K requant + per-call repack）。
- **VEN/OFF=1.49×** = vendor IME q4_K 真 e2e prefill 增益（头顶空间存在）。
- **出口 = 黄-传导稀释**（0.0617× ≪ 2.09×·净退化）。

---

## 3. 判读（三出口判·稀释 finalized·amdahl 传导）

### 3.1 出口 = 稀释（非物理墙·非意外增益）
预注册三出口（裁一.3）：
- **稀释**（prefill e2e < compute-account 增益·kernel 增益实测 × e2e 稀释实测两数齐）→ **黄-传导稀释 finalized**。**三格全中**：prefill e2e ON/OFF（0.0128 / 0.166 / 0.0617）≪ compute-account **2.09×**。两数齐（2.09× sealed × 此处实测 e2e ratio）。
- **物理墙**（e2e ≈parity·双方同贴内存墙）——**不适用我方桥**（ON 是退化非 parity）；但 **q8_0 vendor 侧 VEN/OFF=0.99× 确证 q8_0 prefill 物理墙**（vendor optimized IME 亦打不过 stock）→ 记为稀释-于-walled-regime 语境·非我方桥出口。
- **意外增益**（e2e prefill ≥parity 净赢）——**零格**（如 vendor-ceiling 先例预期·全接线 IME GEMM 无干净 e2e 赢）。

### 3.2 amdahl 传导（为何 2.09× 不传导·甚至反向）
compute-account **2.09×** = tcrv IME **register-resident batched `vmadot` leaf**（`[GAP-IME-LEAF-PIPELINE]` 已闭）vs RVV-vector-path·**kernel-隔离**微核。e2e prefill 桥墙钟里，`vmadot` 微核 = 可忽略小项；**Amdahl-主导项 = 桥集成 glue**：
1. **单线程 `ith==0`**：路由的 mul_mat 只 1/4 pinned hart 计算（其余 3 hart 空转·`ggml_barrier` 等）→ ≥4× 惩罚。
2. **每次 prefill 调用重 repack 权重**（native→fragment-major·**未缓存**·每 forward 每层每 routed mul_mat 重来）→ 主导墙钟（权重 repack 成本 ∝ 权重量·与 M 无关·故 pp32 稀释最烈）。
3. **朴素 scale-fold epilogue** + q8_K/q8_0 per-call requant 激活。
→ 2.09× 微核增益被 (1)(2)(3) **Amdahl 淹没**·净 e2e = 退化。**与 `kernel-wins-dont-transplant-to-e2e`（IME 5.51× kernel → 0.86× decode）+ M0 recon（tcrv 2.09× compute-account 即便接线也不 e2e 传导）逐字一致**。

### 3.3 vendor-ceiling = 稀释非墙的判别证据
vendor IME（optimized·cached repack·multi-threaded）在**同模型/同板/同 clang-18** 给真 e2e prefill 增益：q4_0 **1.98×** · q4_K **1.49×**（q8_0 0.99× walled）。→ prefill regime **有头顶空间**（q4_0/q4_K）·稀释是**我方桥实现未优化**（reference correctness-first·非阵列跑不起）·**非 IME-paradigm 的物理不能**。这是 C3′ 的诚实边界素材：**接线正确性闭环 ≠ 部署性能就绪**（桥 = correctness reference·perf 需 cached-repack + 多线程 emit·未做）。

### 3.4 decode 对照（native fallback·我方核缺席 + 布局副作用）
decode tg64 三格 ON/OFF = 0.62–0.87×。decode M=1 从不命中我方桥（`M>1` 判据·banner 仅 prefill 触发）→ 我方 `vmadot` 核**缺席 decode**（regime 对照·符合设计）。ON<OFF 的轻退化源 = env-gated **native-passthrough repack** 改了权重 buffer 布局·把 decode M=1 逼上比纯 stock RVV 更慢的 GEVM 路径（非我方核·是布局副作用）。ON≈VEN decode（0.52–0.88×·同 build-ime 布局）证 decode 差异是 build-ime-vendor-decode vs build-off-stock-decode 的二进制/核家族差·非我方核。

---

## 4. 板 restored（md5 双证零 stock 改动·三格各 EXIT-trap + 终审独立复核）
每格 run 后 trap：clean source → `make ggml-cpu` → `.so` 覆盖回 ORIG → md5 双证。三格各：
```
q4_0 / q8_0 / q4_K  各:  RESTORE md5 ZERO-CHANGE OK
                         ime.cpp = 40962c7e7c732bf472ae88cef89ced8d  == baseline ✓
                         .so     = 71cc4d295dac29382a0a7d4d5bd0c425  == baseline ✓
                         src_route_left=0 · litter_left=0
终审独立复核: procs=0 · ime.cpp/so md5==baseline · route_markers{q40,q80,q4k}=0 · ORIG_litter=0 · /tmp scratch 清 0
```
（`/tmp/g5m3pair/<fmt>/` bench `.err`/`route_build.log` = board-local scratch·harvest run.log 到 casefile 后显式清除·非 vendor 树。）

---

## 5. 诚实边界（[NG-4]·perf-axis·非绿）
- **已证（perf 转正证据）**：三格 IME forward bridge e2e prefill = **传导稀释 finalized**（ON/OFF 0.0128 / 0.166 / 0.0617 ≪ compute-account 2.09×·两数齐·micro↛e2e）·板 md5 零改动。measured_e2e_delta_pointer + amdahl_pointer 齐 → **provisional 可转正为「黄-传导稀释」**。
- **诚实限制**：① 单 PP=32 prefill 点（real 短-prompt M~15 regime·= 稀释最烈点；桥 repack-bound 故 throughput ∝ PP·大 PP 稀释略缓但仍 ≪1×·见 §3.2）·TG=64 decode·n=10/side（ON side 确定性 CV~0.06%·OFF/VEN 多线程仍 IQR 紧）。② 我方桥 = **correctness-first reference**（单线程 + per-call repack）·**未优化**·此数字**不是 IME-paradigm 的性能上限**（vendor 1.49–1.98× 才是同板 IME 头顶）。③ decode 我方核缺席（对照·非我方核数字）。
- **非绿·非 perf-covered**（八门·kernel-vs-opponent 编译器对称但结果 LOSS）·与三格 correctness 案预注册「forward-wired 后 IME 格 = 黄-传导稀释带账非绿」逐字兑现。**禁"实质胜利/perf 赢"表述·如实报稀释出口。**

## durable files（见 MANIFEST）
- `tools/e2e-harness/board/g5-m3-ime-e2e-pairing/run_pair_board.sh`（board-side 三-side interleaved paired llama-bench·env-toggle + build-off·EXIT/INT/TERM-trap md5 restore·nohup 承载）
- `tools/e2e-harness/board/g5-m3-ime-e2e-pairing/run-pair-bg.sh`（launcher·scp + nohup + logfile-poll·per-format）
- `tools/e2e-harness/board/g5-m3-ime-e2e-pairing/agg_pair.py`（run.log ###AB block → per-side median/IQR + ON/OFF·ON/VEN·VEN/OFF ratio）
- `raw/run_q4_0.log` · `raw/run_q8_0.log` · `raw/run_q4_K.log`（三格 llama-bench json 全输出 + banner + restore md5）
- `raw/agg_q4_0.txt` · `raw/agg_q8_0.txt` · `raw/agg_q4_K.txt`（三格聚合表 + ratio）
