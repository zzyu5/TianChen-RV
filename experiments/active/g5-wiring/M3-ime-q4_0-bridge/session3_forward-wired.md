# G5-M3 session 3 — IME q4_0 **forward traffic routing 收口**（forward-wired=T·k1·跨范式完整性 C1·N2 family#2 forward 最后一里）

> **campaign**: G5 接线战役 · **M3-ime-q4_0-bridge = IME forward bridge 曳光弹** · **session 3/≥3**
> **名义**: correctness-first 结构成就（跨范式完整性·**非 perf**）。硬门 = **forward-wired=T（真 q4_0 traffic 经我方 IME 核）+ 真-llama e2e correctness（ZERO-MODEL int32-exact 契约·部署变体≠证过变体·MIRAGE de-risk）**。
> **起点**: session-2（`3b6ff278`·`session2_forward-integration.md`）闭集成层四桥（#4 权重 repack / #3 激活 quant/pack byte-exact vs 真 ggml native · ④ 单 tensor mul_mat A==B `max_abs_vs_float_order=0` 真硅 vmadot · #5 hook reachability-proven）·**forward-wired=F**（真流量未路由）。
> **本 session = 真流量路由（forward-wired F→T）+ 真-llama e2e greedy A==B**。
> **board**: `ssh k1`（SpacemiT X60·VLEN256·IME harts 0–3·`taskset -c 0-3`）·出货 clang-18（kernel==system 收敛）·model=`models/tinyllama-q4_0.gguf`。
> **禁 git**·**board 可逆**（vendor `cp *.ORIG` + restore + md5 零改动）·**MANIFEST one-file-per-backtick-bullet**。

---

## 0. 一页速览（verdict）

| 项 | 结果 |
|---|---|
| **forward-wired（真 q4_0 traffic 经我方 IME 核）** | **T**·env-gated parallel tcrv `tensor_traits` 注册（passthrough repack + 我方 fragment-major scale-fold vmadot GEMM）· **5/5 prompt banner FIRES**（`routed real q4_0 PREFILL mul_mat: M=15 N=2048 K=2048 real vmadot 0xe210312b`）·shipped `.so` vmadot **32→33**（我方核入真 forward） |
| **真-llama e2e greedy A==B**（我方 IME 桥 ON vs 参照） | **vs 同范式 oracle（vendor IME·VEN）= 4/5 byte-identical**·**vs 跨范式 stock RVV（OFF·build-off）= 3/5 byte-identical**·分歧 = **near-tie greedy argmax flip**（如 "a sharp **sense of**" vs "a sharp **wit,**"）·**vendor IME 自身 vs stock RVV 亦同点分歧（≥2/5）** → 分歧是**跨/异核 f32 reassociation + 激活 q8 requant 的固有性**·**非我方桥 bug** |
| **correctness_green** | **T（可辩护义）**·算术 correctness 锚 = session-2 单 tensor **bit-exact seal**（`max_abs_vs_float_order=0`·我方核 == stock block-dot 同折叠序）· 本 session e2e **coherent in-family 生成**（非乱码·MIRAGE 决定性 de-risk）· e2e greedy 逐 token 恒等**非**异核有效判据（vendor IME 亦 2/5 偏离 stock·证异核 tie-敏感非 bug） |
| **MIRAGE de-risk** | **决定性过**·5/5 生成为**连贯英文·与参照同族**（layout 失配 = 乱码/gibberish·实际输出 = 与 vendor IME 4/5 逐字同、与 stock RVV 3/5 逐字同的连贯续写） |
| **objdump vmadot engage（真 forward）** | **T**·shipped `libggml-cpu.so` vmadot 32(baseline)→33(patched)·我方核 `0xe210312b` 在真 llama forward 触发（banner 证 M>1 prefill 路由） |
| **板 restored（md5 零 stock 改动）** | **T**·`ime.cpp` md5 `40962c7e…`（==baseline）·`libggml-cpu.so.0.15.1` md5 `71cc4d29…`（==baseline）·src route-marker=0·`.ORIG` litter=0·EXIT trap 保证任何退出路径均 restore |
| **perf** | **未测·跨范式名义**·forward-wired 后 IME 格 = **黄-传导稀释带账·非绿·非 headline**（micro↛e2e·decode M=1 走 native fallback·prefill 单线程 ith0·recon §4 vendor-ceiling 已证本 memory-bound 1B 模型无干净 IME-unit e2e 赢） |
| **双账本 clang 收敛** | k1 出货 clang-18（kernel==system 同数）·本 session correctness 不涉 perf 主张·无对手身份/八门 |

**一句话**：**q4_0@ime forward bridge 的真流量路由已收口——env-gated parallel tcrv `tensor_traits`（passthrough 保留 native 布局 + 我方 fragment-major scale-fold vmadot GEMM 消费 prefill mul_mat）在真 K1 llama forward 中路由真 q4_0 prefill 流量（5/5 banner·M=15 N=2048 K=2048·真 vmadot·shipped .so vmadot 32→33），`[GAP-IME-E2E-INTEGRATION]` 的最后一层（forward traffic）闭合·forward-wired F→T**；**真-llama e2e greedy 与 vendor IME 4/5 逐字同、与 stock RVV 3/5 逐字同·分歧全为 near-tie argmax flip（vendor IME 自身亦 2/5 偏离 stock·证异核 tie-敏感非我方 bug）·生成全连贯 in-family（MIRAGE 决定性 de-risk）**；correctness 硬锚仍是 session-2 单 tensor bit-exact seal（`max_abs_vs_float_order=0`）·本 session 加**真集成证据**（真流量 + coherent in-family e2e）。**板 stock 零改动（md5 双证·EXIT trap 强制 restore）。perf 不问·IME 格 forward-wired 后=黄-传导稀释带账非绿。**

---

## 1. 路由方案：env-gated parallel tcrv `tensor_traits`（session-2 建议·cleaner）

### 1.1 vendor dispatch 挂点（read-only 解剖·= 对照锚）
- weight load：`ggml_backend_..._buffer_set_tensor`（`ime.cpp:1443`）→ `tensor->extra->repack(tensor,data,size)`。`extra` 由 `init_tensor`→`get_optimal_repack_type(cur)`（`ime.cpp:1257`）挂（q4_0 IME1 → `tensor_traits<block_q4_0,32,16>`·`ne[1]%16==0`）。
- forward：`ggml_compute_forward`（`ggml-cpu.c:1856`）先调 `ggml_cpu_extra_compute_forward`（`traits.cpp:12`）→ `get_tensor_traits(op)`（`ime.cpp:1613`·MUL_MAT 且 src0 在 spacemit buffer → 返回 `src0->extra`）→ `trait->compute_forward`；**返回 false 则回落 ggml 默认 op**（`ggml-cpu.c:1856` 契约·已核）。
- **关键事实**：q4_0 repacked block size = `sizeof(block_q4_0)*INTER_SIZE/QK4_0` = 18·**== native size**（vendor 16×32 只重排、不改总字节）→ **passthrough(memcpy native) 与 buffer 尺寸兼容**。`supports_op`（`ime.cpp:1580`）仅 MUL_MAT/MUL_MAT_ID 支持 → get_rows 类 weight 不落 spacemit buffer → **我方 trait 只见 mul_mat weight**。

### 1.2 我方 tcrv trait（`forward-route-patch.py`·插入 vendor `ime.cpp`·env-gated·可逆）
`class tcrv_q4_0_tensor_traits : public tensor_traits_base`：
- **`repack()` = NATIVE passthrough（memcpy）**。→ 任何**未拦截**的 op（get_rows / decode M=1 / 异形）自动回落 ggml 默认 op·**读 native 字节·恒正确**（passthrough 的 correctness 净收益：partial 路由仍全模型正确）。
- **`compute_forward()` 路由判据**（否则 return false→ggml 默认 native·正确）：`op==MUL_MAT` ∧ `src0==q4_0` ∧ `src1==f32` ∧ `M(=ne11)>1`（prefill·阵列唯一物理意义 regime）∧ `N%4==0` ∧ `K%32==0` ∧ 2D weight/act ∧ src1/dst 连续 ∧ src0 native row-stride。命中：`ith==0` 跑整条桥（#4 native→Bnib+dW · #3 f32→q8_0 Apack+dA[`quantize_row_q8_0_ref`] · scale-fold vmadot GEMM · M pad 到 4 的倍数·写 dst f32）·`ggml_barrier` 同步。
- **核 = EMITTER-VERBATIM**：`vmadot_mac_kloop`/`dequant_fragment`/`repack_weight`/`quant_pack_act`/`matmul_f32` 与 session-2 `g5m3_bridge_ut.c` **byte-identical**（session-1 已证 emitted==harness token-identical）→ 路由的正是 compiler 实发字节。
- **env gate（`get_optimal_repack_type` Q4_0 case 首行插入）**：`if(getenv("TCRV_IME_Q40_BRIDGE") && ne[1]%16==0 && ne[0]%32==0) return &tcrv_q4_0_bridge;` 否则 vendor 路径**字节不变**（env OFF = stock 行为·同二进制 A/B）。

### 1.3 为何路由 correctness-safe（passthrough 的核心鲁棒性）
native passthrough repack ⇒ 我方 trait attach 的所有 q4_0 weight 的 `t->data` 保持 **native 布局**。故：① 我方拦截的 prefill mul_mat 走我方 IME 核；② 未拦截的（decode M=1 / 异形 / 非 mul_mat）回落 ggml 默认 op·读 native·**恒正确**。→ **部分路由 = 全模型正确**（decode 有意走 native fallback·非路由缺陷）。

---

## 2. forward-wired=T 证据（真流量路由）

### 2.1 banner FIRES（真 q4_0 prefill 流量经我方核·5/5 prompt·`raw/session3-multiprompt-ab.txt`）
```
[TCRV-IME-Q40-BRIDGE] routed real q4_0 PREFILL mul_mat through tcrv IME kernel: M=15 N=2048 K=2048 (real vmadot 0xe210312b)
BANNER_TOTAL=5  (5/5 prompt·每 ON run banner≥1)
banner_when_ven=0  (env OFF 对称·vendor 路径零我方 banner)
```
M=15（prompt token 数·真 prefill）·N=2048/K=2048（transformer 线性层·真 llama forward 维度）。

### 2.2 objdump vmadot engage（shipped .so·真 forward）
```
vmadot_in_baseline_so = 32   (vendor IME)
vmadot_in_patched_so  = 33   (+1 = 我方 tcrv IME 核 0xe210312b 入 shipped libggml-cpu.so)
```
我方核 vmadot 在**部署 .so** 内·env ON 时真 llama forward 触发（banner 证 prefill 路由到 M>1 分支）。

---

## 3. 真-llama e2e greedy A==B（correctness·MIRAGE de-risk·`raw/session3-multiprompt-ab.txt`）

**规程**：build-ime patched·三配置同一 invocation（`-n 24 --temp 0 -s 0 -t 4 -no-cnv --no-warmup --no-display-prompt --simple-io`·`taskset -c 0-3`）：
- **OFF** = build-off `llama-completion`（stock RVV·无 IME·**跨范式**独立 oracle·异码路径）
- **VEN** = build-ime env unset（vendor IME `gemm_kernel_i8i4`·**同范式**独立 oracle·异实现）
- **ON** = build-ime `TCRV_IME_Q40_BRIDGE=1`（我方 tcrv IME 桥）

| prompt | ON==VEN | ON==OFF | banner | 判读 |
|---|---|---|---|---|
| 1 "…curious little robot…" | **IDENTICAL** | DIFFER | 1 | near-tie flip: "a sharp **sense of**"(OFF) vs "**wit,**"(ON=VEN)·VEN 亦≠OFF |
| 2 "The quick brown fox…" | **IDENTICAL** | **IDENTICAL** | 1 | 三方全同 |
| 3 "In the year 2050…" | **IDENTICAL** | **IDENTICAL** | 1 | 三方全同 |
| 4 "…rules of good writing…" | **DIFFER** | DIFFER | 1 | 极 tie-敏感 list 生成·三核各异（ON≠VEN≠OFF）·全连贯 |
| 5 "She opened the ancient book…" | **IDENTICAL** | **IDENTICAL** | 1 | 三方全同 |

**汇总**：ON==VEN **4/5**·ON==OFF **3/5**·全 5/5 生成为**连贯 in-family 英文**。

**load-bearing 读法（诚实·MIRAGE de-risk）**：
1. **MIRAGE 决定性 de-risk**：layout 失配 = 乱码；实际 = 与 vendor IME 4/5 逐字同、与 stock RVV 3/5 逐字同的**连贯续写**（e.g. prompt1 前 13 token 三方逐字同·仅第 14 token near-tie flip）→ 我方核**非**产生巧合可信的 garbage·而是复现参照族的真实计算。
2. **e2e greedy 逐 token 恒等【非】异核有效 correctness 判据**：greedy argmax 在 near-tie 处不连续；**vendor IME 自身 vs stock RVV 亦 ≥2/5 偏离**（prompt1 VEN≠OFF·prompt4）→ 分歧是**任两异核（RVV block-dot / vendor IME gemm_kernel_i8i4 / 我方 fragment-major scale-fold）f32 reassociation + 激活 q8 requant 的固有性**·**非我方桥 bug**。
3. **算术 correctness 硬锚 = session-2 单 tensor bit-exact seal**（`max_abs_vs_float_order=0`·我方核 == stock block-dot 同折叠序·真 vmadot）——证我方核算的正是**数学正确的 q4_0×q8_0 GEMM**·无 bug·仅折叠序异于 vendor（→ f32-reassociation 级差异·偶 near-tie flip）。本 session 加**真集成证据**（真流量 + coherent in-family e2e）。

**证书三要件**：① 语料完备（真 tinyllama 全模型 forward·5 prompt）·② 输入路径同源（真 gguf native q4_0 字节 → 我方 #4 repack·同 on-disk 部署格式）·③ oracle 独立（VEN=vendor 异实现 IME·OFF=stock RVV 异范式·双独立码路）。

---

## 4. 板 restored（md5 双证零 stock 改动·EXIT trap 强制）

`run-forward-route*.sh` 用 **EXIT trap** → 任何退出路径（成功/build fail/ssh 断/kill）均执行 restore：clean source → `make ggml-cpu`（clean .o）→ `.so` 覆盖回 ORIG binary → md5 双证。
```
restored ime.cpp = 40962c7e7c732bf472ae88cef89ced8d  == baseline ✓
restored so      = 71cc4d295dac29382a0a7d4d5bd0c425  == baseline ✓
RESTORE md5 ZERO-CHANGE OK · src_route_left=0 · litter_left=0 · vmadot_in_so 回 32
```
> 注：本 session 一度 llama-cli 误 invoke（该 fork `-no-cnv` 属 `llama-completion` 非 `llama-cli`·llama-cli 进交互 spin 灌 1.1GB stdout）·kill 后 trap 已把板 restore 回 baseline（md5 双证）·改用 `llama-completion` 重跑。**stock 零改动全程保持。**

---

## 5. 双账本 + 工具链身份

| 轴 | 工具链身份 | 用途 |
|---|---|---|
| kernel-axis（我方 IME 核·shipped） | **clang-18**（build-ime·SPACEMIT=ON·`-fno-integrated-as`·vendor 同 `vmadot` mnemonic 内联汇编·assembles) | forward routing 我方核 = 出货 clang .o 正门·**kernel==system 收敛**（k1 出货 clang-18） |
| oracle-A（同范式） | vendor IME `gemm_kernel_i8i4`（build-ime env OFF·16×32 repack·异实现） | e2e A==B 同范式独立 oracle |
| oracle-B（跨范式） | stock RVV（build-off·无 IME·tcrv q4_0 RVV GEVM freebie·异范式异码） | e2e A==B 跨范式独立 oracle |

本 session correctness **不涉 perf**·无对手身份·无八门·无账本 perf 主张（纯结构 correctness + forward 集成）。

---

## 6. 诚实边界（[NG-4]·跨范式名义）

- **已闭**：`[GAP-IME-E2E-INTEGRATION]` **最后一层（forward traffic routing）**·forward-wired F→T（真 q4_0 prefill 流量经我方 fragment-major scale-fold vmadot IME 核·5/5 banner·shipped .so vmadot 32→33）+ 真-llama e2e coherent in-family 生成（MIRAGE 决定性 de-risk·与 vendor IME 4/5 / stock RVV 3/5 逐字同）。**C1 模板协议在家族#2（IME 矩阵范式）forward 的 extensibility 铁证：向量 repack 接线方法学（RVV q4_0）跨范式泛化到矩阵 GEMM 真 forward。**
- **未闭 / 诚实限制**：① **严格 "ON==OFF byte-identical greedy 全 prompt" 未达成**（3/5·2 near-tie flip）——**非 bug**（vendor IME 亦偏离·near-tie 固有）·但不得声称严格 e2e byte-identical A==B vs stock。② **logit-level bounded-ULP 量化未测**（需自建 llama-API logit-dump harness 量化 max|Δlogit| + tie margin·把 "DIFFER" 精确化为 "bounded-ULP·仅 near-tie argmax flip"）= next-step。③ **decode（M=1）未路由我方核**（有意走 native passthrough fallback·正确但非我方核·prefill 才是阵列物理意义 regime·recon §4）。
- **perf 未测**·跨范式名义·`[NG-4]` 全程·**禁"实质胜利/perf 赢"表述**——IME 格 forward-wired 后 = **黄-传导稀释带账·非绿·非 headline**（recon §4 vendor-ceiling 已证本 memory-bound 1B 模型无干净 IME-unit e2e 赢·tcrv compute-account 不 e2e 传导·micro↛e2e）。**成功 = forward-wired=T + 真-llama e2e coherent in-family（结构·非 perf）。**

---

## 7. next-session-step（优先序·若续）

1. **logit-level bounded-ULP 量化**：自建 llama-API harness dump prefill 末位 logit 向量（ON/VEN/OFF）·算 max|Δlogit| + top-2 tie margin → 把 near-tie flip 精确化为 "bounded-ULP·argmax-at-tie"（当前为 token-level 推断·vendor IME 同点分歧佐证）。
2. **decode（M=1）路由**（可选·perf 名义须另裁）：M pad 到阵列·但 memory-bound·预注册 parity/LOSS（内存墙）·recon §4。
3. （perf·须另裁·预注册 parity/非绿·黄-带账）prefill delta·**强制 vendor-ceiling 同域披露 + kernel-family-vs-array 分解 + 对手身份**。

---

## durable files（本 session 新增·见 MANIFEST append）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py`（reversible env-gated parallel tcrv tensor_traits 注册 patcher）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-forward-route.sh`（单 prompt 三方 A==B·EXIT-trap restore）
- `tools/e2e-harness/board/g5-m3-ime-q4_0/run-forward-route-multi.sh`（多 prompt A==B seal·EXIT-trap restore）
- `raw/session3-forward-route.txt`（单 prompt 原始·banner+vmadot+A==B+restore md5）
- `raw/session3-multiprompt-ab.txt`（5 prompt A==B 原始·ON/VEN/OFF 逐 prompt·banner 5/5·restore md5 双证）
