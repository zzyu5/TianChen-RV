# A2 batch7 — IME kernel-sym cold microbench（k1·q4_0/q8_0/q4_K @ime·我方 IME GEMM 发射体 vs stock vendor IME 派发）

> **任务**：线 A·A2-batch7 = **IME kernel-sym cold microbench**（k1·3 格 gemm_tile/{q4_0,q8_0,q4_K}@ime）。我方 = weft format-keyed IME GEMM 发射体（`vmadot_mac_leaf` 0xe210312b·front-door）·**对手 = stock SpacemiT vendor IME 派发**（`spacemit_kernels::ime1::gemm_kernel_i8i4`·真硬手调 IME 核）。
> **赛道**：**IME-GEMM 轴 kernel-sym MICRO**（prefill regime = IME 天然域·M>1）。**NOT e2e·NOT perf-covered·不入系统账**。[NG-4]。
> **口径铁线**：cold 唯一·**N=25 median+relIQR+2-seed**·**禁一切继承**（★q4_K@ime 数据串行 bug 前科·真测 IME kernel-sym·非 e2e/native-gemm 继承）·对手身份探针（符号级 vendor IME）·byte-exact（vmadot int32 0-diff·真硅 0xe210312b）·**ZERO-MODEL 正确门**·预注册判读（cold≥0.8=PASS/<0.8=named-X+墙）·**0 样本不造数·成色诚实**（vendor=真手调强对手·**非便宜档**·输给强手调=honest 强对手负结果·别粉饰）。
> **测于**：2026-07-16 · k1(VLEN256·SpacemiT-X60·clang-18.1.8·gov=performance·taskset 0-3 IME harts)·`use_ime1:1 use_ime2:0`（板自证 ime1 = live vendor IME 路）。**rvv 列 = N/A-hw（硬件缺席·机判·IME=SpacemiT 专属）**。stock vendor `.so` 只读·md5 `71cc4d29` before==after UNCHANGED·无 git。

---

## 0. ★净结论

**★关键结构发现（决定 3 格去向·从 vendor 符号表 + 源码坐实）**：vendor（`build-ime` libggml-cpu.so·`GGML_CPU_RISCV64_SPACEMIT=ON`）的 IME GEMM = **唯一一个** `ime1::gemm_kernel_i8i4`（i8 激活 × i4 权重·0xe210312b vmadot·`use_ime1:1`）。所有格经 `riscv64_spacemit::tensor_traits<block_X>` **requant 进这一个核**。routing 实测（`ime.cpp:317`）：
- **q4_0 → IME**（`gemm_kernel_i8i4` 原生 i8i4·**真手调强对手存在**）。
- **q4_K → IME**（`repack_q4_k_to_q4_1_16_bl` 先 requant 成 q4_1·再 `gemm_kernel_i8i4`·**低精度近似**·vendor 核同 q4_0）。
- **q8_0 → ✗ 无 IME**（dispatch 仅 q4_0/q4_1/q4_K 进 ime1·q8_0 = 8-bit 权重·i8i4 单元装不下→落 `ggml::cpu::repack`=**RVV 回退非 IME**）。★**q8_0@ime 无 vendor IME 对手**（坐实 e2e "beat 弱 vendor 0.984×" = 赢 RVV-repack 非赢 IME）。

| 格 | vendor IME 对手 | kernel-sym cold（k1·我方最优 vs vendor·N=25·2-seed） | 判读 | 成色 |
|---|---|---:|:--:|---|
| **q4_0@ime** | ✅ `ime1::gemm_kernel_i8i4`（真手调 IME 强对手） | **0.196×**（ours_predec 25.0ms / vendor 4.89ms） | **named-X** | ★**输给真手调 vendor IME ~5.1×**·**强对手负结果·非便宜档**·墙=scale-fold epilogue 未融进 MAC（vendor fp16-scale 在 vmadot 内融·我方独立 f32 scalar fold 主导）·[PAT-1] 兑现 |
| **q8_0@ime** | ✗ **不存在**（q8_0 不进 vendor IME） | **N/A-vendor-IME** | **BLOCKED-结构** | vendor 无 q8_0 IME 核·唯一"vendor"=RVV-repack 回退（=e2e 弱 vendor）·**IME-vs-IME 无对手** |
| **q4_K@ime** | ✅ `gemm_kernel_i8i4`（经 q4_1 requant·faithful 到 fp16+zp 舍入） | **0.049×**（ours 110ms / vendor 5.48ms） | **named-X** | ★**输给 vendor IME ~20×**（gap 由 q4_0 5× 扩到 q4_K 20×）·墙=**super-block two-level fold epilogue**（per-sub-block sc/m/d/dmin scalar fold）·我方 fold 随格式复杂度暴涨·vendor requant 吸收格式→吞吐几不变·**C3′ [PAT-1] format-keyed 边界最锋利证据** |

**★成色诚实（本役与 batch4/6 关键差异）**：batch4/6 对手 = stock block-dot / scalar-ref（**便宜档·big multiple·我方赢=cheap**）。**本役对手 = SpacemiT 生产级手调 IME 核**（真硅 vmadot·wide-tiled·in-kernel scale fold）=**真强对手**。**2/2 可测格全输**（q4_0 5.1×·q4_K 20×）= honest **强对手负结果**（C3′ 边界·**非失败·非便宜档·是最有价值的诚实负结果**——首个 kernel-sym 对真手调硬件 vendor IME 核的正面度量）。**0 hand-brick 于我方赢面**（我方全输·无需 hand-brick lint）。★**gap 随格式复杂度扩张**（q4_0 5× → q4_K 20×）：我方 fold epilogue 随 super-block 结构暴涨（q4_0 25ms → q4_K 110ms = 4.4×）·vendor requant 把所有格式塞进同一 `gemm_kernel_i8i4`（q4_0 4.9ms → q4_K 5.5ms = 几不变）→ **发射器成熟度 gap 的锋利具象 = scale-fold 融合缺失**。

---

## 1. ★harness 构造实录（durable·可复现）

**驱动**：`A2-batch7-ime-kernel-sym-raw/ime_q40_kernelsym.cpp`（新造·链接 build-ime vendor `.so`）。
- **OURS**（三形态·全 pre-quant 输入·产 float·[NG-4] kernel-sym 单元 = 取 pre-quant 激活 + pre-pack 权重 → float）：
  - **leaf（Form A·front-door 叶）**：q4_0 nibble 解码 **IN-LOOP** + `vmadot_mac_kloop`（4x4x8·0xe210312b）+ per-block fp16-scale fold。= 纯 front-door 发射叶。
  - **predec（Form B·deployed）**：权重 setup 期预解码成 int8（= G6-A M3 `repack_dequant_weight`）·`vmadot_mac_kloop` + fold。**= 我方最优全核**（与 vendor "权重预 repack" 对称）。
  - **w4（G6-A M7 wide-vmadot-tiling·`macKloopHelperBodyWide njw=4`）**：ONE A frag 喂 4 独立 B col-tile·4 输出 4x4 tile·pre-decoded。
- **VENDOR**（stock IME 派发·符号级探针）：`spacemit_kernels::ime1::gemm_kernel_i8i4(blk_len=32, quant_a, quant_b=block_q4_0x16, zp=null, C_f32, count_m=4, count_n=16, k_blks=K/32, ldc=N)`。
  - 权重 pack = 复现 `make_block_q4_0x16`/`repack_q4_0_to_q4_0_16_bl`（block<4,16>=288B·16 fp16 d + 256B interleaved nibble·16 列/组）。
  - 激活 quant = **调 vendor 自家导出** `ime1::quantize_a_4row_i8(32, X, K, QA)`（4-row interleave·row_stride_a=nb*36）。
  - gemm 调用循环 = 复现 `ime.cpp` tcm 路（m-tile of 4 × n-tile of 16·b_col += 16*row_stride_b）。
- **链接**：`clang-18 -O3 -march=rv64gcv_..._zvl256b -fno-integrated-as -L build-ime/bin -lggml-cpu -lstdc++`。vendor 符号（mangled `_ZN16spacemit_kernels4ime116gemm_kernel_i8i4E...`）直链。objdump 证 vmadot 0xe210312b（leaf 1 + w4 4 = 6 处）。

**★正确门（ZERO-MODEL·no fabrication）**：
- **(1) OURS int32 core**：real-vmadot int32 partial == 独立重算 partial = **int32-EXACT byte-exact**（seal `q4-0-matmul-tile-int32-k1seal` 复跑：decode 512/512 + int32 MAC 64/64 bit-exact·0xe210312b 真硅）。leaf==predec==w4 int32 全 **maxdiff=0.000e+00**（三形态 bit-identical）。
- **(2) OURS scale-fold f32** vs canonical q4_0×q8_0 = maxrel **7.1e-4**（K=2048·f32 fold reassociation·benign·int32 core exact）。
- **(3) VENDOR** vs true-f32 = Frobenius-rel **0.0651** == ours_vs_true **0.0650**（**vendor 与 ours 量化误差同**）·vendor_vs_ours = **0.0013**（vendor 与 ours 逐元素 agree 到 0.13%·sample C[0,0..3] 逐值吻合）→ **坐实 vendor 调用正确**（算同一 q4_0 GEMM·差 0.13% = 激活 quant scheme 微异 canonical-q8_0 vs vendor-quantize_a）·**非 layout bug**。

---

## 2. ★q4_0@ime 逐形态 cold（k1·prefill·N=25 median+relIQR·2-seed{0xC0FFEE1,0x1357ACE}·taskset 0-3）

### 主 shape M=64 N=512 K=2048（IME prefill 天然域）

| 形态 | ours ms(s1/s2) | vendor ms(s1/s2) | **ratio vendor/ours(s1/s2)** | predreg | 说明 |
|---|---:|---:|---:|:--:|---|
| **predec（我方最优·deployed）** | 24.96/24.90 | 4.89/4.81 | **0.196/0.193** | **named-X** | base vmadot 单 tile·pre-decoded 权重·**输 vendor 5.1×** |
| w4（wide-tiling G6-A M7） | 30.10/29.52 | 4.89/4.81 | **0.163/0.163** | named-X | ★wide tiling **更慢**（epilogue-bound·array-util 被 scale-fold 稀释·[PAT-1] 兑现） |
| leaf（front-door 叶·in-loop decode） | 208.6/208.3 | 4.89/4.81 | **0.023/0.023** | named-X | 每 call 重解码权重·最差·非部署形 |

### 复核 shape M=32 N=256 K=1024（seed A）
predec **0.200×**·w4 0.143×·leaf 0.023×（GATE PASS·比值稳）→ **shape-robust**。

**★判读（预注册·<0.8=named-X+墙）**：**q4_0@ime = named-X 0.196×**（我方最优 predec form）。
- **墙①（主·具名）= scale-fold epilogue 未融进 MAC**：vendor `gemm_kernel_i8i4`（M4 kernel `SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl`）把 fp16 per-block scale **在 vmadot 内核 in-register 融**；我方发射体 int32 MAC 后走**独立 scalar f32 fold 循环**（O(M·N·nb) strided f32·主导 cold 时间）。= [PAT-1] format-keyed epilogue 边界（G6-A array-util 杠杆 e2e payoff 稀释同源·[CASE 令一.2]）。
- **墙②（次·具名）= wide-vmadot 在全核 cold 无 payoff**：w4（0.163×）比 predec（0.196×）**更慢**——G6-A M7 的 w4 1.955× 是**孤立 compute 段**的 array-util·**全核 cold 下 epilogue/内存主导**·wide MAC 省的 compute 被 heavier fold 抵消（w4 存 64 int32→fold locality 更差）。**印证 G6-A [PAT-1]**：wide tiling payoff = format-keyed on epilogue weight·非 universal。
- **成色**：**强对手负结果·非便宜档**。vendor = SpacemiT 生产手调 IME 核（真硅 vmadot·in-kernel fold·预 repack 权重）。输 5.1× 是**诚实 C3′ 边界**（我方 IME 发射体成熟度未达 vendor·具体 gap = **scale-fold 融合** + **发射器未产 in-kernel fold**）。**0 verified 我方赢·无 hand-brick lint 适用**（我方全输）。

---

## 3. q4_K@ime 逐 cold（k1·prefill·N=25·2-seed）+ q8_0@ime BLOCKED

### 3.1 q4_K@ime（✅ 本役完成·驱动 `ime_q4k_kernelsym.cpp`）

**harness**：OURS = weft q4_K IME 核（`weft_ime_q4_K_vmadot_matmul` 谱系·vmadot two-level dmin/bsums fold → float·pre-decoded·加 per-32-block 激活 scale d_a）。VENDOR = q4_K → **q4_1x16 requant**（复现 `repack_q4_k_to_q4_1_16_bl`/`make_block_q4_1x16`·带 zp）→ **同一** `ime1::gemm_kernel_i8i4`（zp≠null·`b_col_zp=b_col`）。W = **dequant(realistic q4_K bytes)**（受控 sc∈[20,63] m∈[0,15] 避退化块）·两侧消费同一 q4_K。

| 板·shape | ours ms(s1/s2) | vendor ms(s1/s2) | **ratio(s1/s2)** | predreg | gate |
|---|---:|---:|---:|:--:|:--:|
| k1 M=64 N=512 K=2048 | 110.16/110.13 | 5.48/5.35 | **0.0497/0.0485** | **named-X** | PASS |
| k1 M=32 N=256 K=1024（复核） | 13.69 | 0.676 | **0.0494** | named-X | PASS |

- **正确门**：ours maxrel(vs qa·W_dequant)=1.5e-5～1.5e-3（int core exact + f32 fold·benign）。**vendor Frobenius-rel vs ours = 0.0119**（vendor 与 ours agree ~1.2%·= q4_1-requant fp16(d1)+zp 整数舍入·**坐实 vendor q4_K 调用正确**·vendor_vs_true 0.0128 ≈ ours_vs_true 0.0036 同量级）。
- **判读 = named-X 0.049×**（输 vendor IME **~20×**）。**墙（主·具名）= super-block two-level fold epilogue**：q4_K per-sub-block（8/super-block）scalar fold `d_a·(d·sc·sumi − dmin·m·asum)`·**O(M·N·nsb·8) f32 scalar** 主导 cold。**vendor 把 q4_K requant 成 q4_1 塞进同一 i8i4 核**（吞吐 5.5ms ≈ q4_0 4.9ms·+12% for 304B/16-col + zp）→ **格式复杂度全被 requant 吸收**。
- **★C3′ 最锋利证据（[PAT-1] format-keyed 适用边界）**：我方 fold **随格式暴涨**（q4_0 25ms → q4_K 110ms = **4.4×**·super-block 结构税）·vendor **几不变**（4.9→5.5ms = 1.12×）→ gap 从 q4_0 5.1× 扩到 q4_K 20×。**发射器成熟度 gap = scale-fold 未融进 MAC**（vendor in-kernel fp16 fold·我方独立 scalar epilogue·同 e2e G6-A q4_K@ime 0.909× 稀释同源·令一.2）。

### 3.2 q8_0@ime = BLOCKED-结构（无 vendor IME 核）

vendor dispatch（`ime.cpp:317` `if constexpr q4_0/q4_1/q4_K → gemm_kernel_i8i4`）**仅** q4_0/q4_1/q4_K 进 `ime1`；q8_0（8-bit 权重）**不进 IME**（i8i4 单元装不下 8-bit 权重·`std::is_same_v` 不含 q8_0）→ 落 `ggml::cpu::repack::tensor_traits<block_q8_0>` = **RVV 回退非 IME**（符号表证：spacemit namespace 无 `<block_q8_0>` tensor_traits·只 `repack::` 通用）。**∴ q8_0@ime 的 "vendor IME 派发" 不存在**·kernel-sym IME-vs-IME **无对手可测 = BLOCKED-结构**（**非未做·是结构事实**）。**坐实 e2e "q8_0@ime beat 弱 vendor 0.984×" = 赢 RVV-repack·非赢 IME**（本役从符号表 + 源码 dispatch 双证）。

---

## 4. ★T3 回填清单（★留主会/机算入库·本 agent 不动 T3·**IME kernel-sym 轴·禁继承·NOT e2e/perf-covered**）

> **轴区分铁线**：本役 = **IME-GEMM kernel-sym cold micro**（我方 IME 发射体 vs **vendor 手调 IME 核**）。**与 e2e perf-covered 绿（q4_0@ime/q8_0@ime 转绿·对手=stock 非-IME parity）是不同赛道·禁互推**（perf-covered 绿 = stock-parity 系统账；本役 = vendor-IME kernel-sym·两个对手身份·[NG-4]）。故 T3 现有 `gemm_tile|{q4_0,q8_0}|ime:绿(GREEN)` **不改**（那是 e2e 账）·本役数据入 **kernel-sym-vs-vendor 附列/脚注**。

**engine=ime·k1 列·rvv=N/A-hw（IME=SpacemiT 专属·硬件缺席·机判）·禁继承（q4_K@ime 串行 bug 前科·真测 vmadot cold）**：

| kernel(op\|format) | engine | k1 kernel-sym-vs-vendorIME cold | opponent(探针) | 判读 | 成色 tag |
|---|---|---:|---|:--:|---|
| gemm_tile\|q4_0 | ime | **0.196×**（ours_predec 25.0ms / vendor 4.89ms·N=25·2-seed） | `spacemit_kernels::ime1::gemm_kernel_i8i4`（真手调 vendor IME·0xe210312b） | **named-X** | 强对手负结果·非便宜档·墙=scale-fold 未融·[PAT-1] |
| gemm_tile\|q8_0 | ime | **N/A-vendor-IME**（BLOCKED-结构） | ✗ 不存在（q8_0 不进 vendor IME·`ime.cpp:317`） | **BLOCKED** | IME-vs-IME 无对手·vendor q8_0=RVV-repack 回退 |
| gemm_tile\|q4_K | ime | **0.049×**（ours 110ms / vendor 5.48ms·N=25·2-seed） | `ime1::gemm_kernel_i8i4`（q4_1 requant·zp≠null） | **named-X** | 输 20×·墙=super-block two-level fold epilogue·C3′ format-keyed 边界最锋利 |

**36-col 关键列**：cold_ratio=**kernel-sym-vs-vendorIME**（q4_0 0.196 / q4_K 0.049 / q8_0 N/A）；opponent_grade=**vendor-hand-tuned-IME**（**非** stock/scalar-ref 便宜档·真强对手）；opponent_symbol=`spacemit_kernels::ime1::gemm_kernel_i8i4`（0xd29e6·32 vmadot·`use_ime1:1`）；compiler_axis=**k1 clang-18(ours .o) vs vendor stock gcc-13 Bianbu .so**（非对称·ours=clang·vendor=gcc·**披露·非域混杂 artifact**：vendor 是 as-shipped 出货核·kernel-sym 对 as-shipped 手调对手=合法赛道·同 CLAUDE.md perf 规则③ shipped-baseline）；hardgate_0p8=**in-denom**（q4_0/q4_K named-X 双格入分母=真负结果·q8_0 BLOCKED-结构·域外）；ledger_account=`ime-kernel-sym-vs-vendor`·regime=**prefill**（IME 天然域·≠decode·decode IME 无用）；rvv 列=**N/A-hw**（IME 硬件缺席·机判·别填别继承）。
- **provenance**：`A2-batch7-ime-kernel-sym-raw/logs/{q40,q4k}_run.log`（cold raw·2-seed·2-shape·gate·探针·md5）。
- **★成色标注硬线**：本役 2/2 可测格**全 named-X（输强手调 vendor）**·**0 我方赢**·**禁写成 IME 胜绩**。这是 C3′ **诚实负结果**（发射器成熟度未达硬件 vendor·具名 gap=scale-fold 融合缺失）·与 e2e perf-covered 绿（stock-parity 系统账·另一赛道）**分开报·禁混口径**。

## 5. 污染纪律 + restore

- **cold 协议**：32MiB flush/rep·N=25 median+relIQR·2-seed·within-proc·core-pinned（taskset 0-3 IME harts·gov=performance）。M=64 N=512 K=2048（weight pack ~576KB·flush 保 cold）。
- **stock 只读**：vendor `libggml-cpu.so.0.15.1` md5 `71cc4d29` before==after UNCHANGED（仅链接·未改 .so·未 git）。
- **域**：k1 clang-18 部署对称（我方 .o）+ vendor stock `.so`（gcc-13 Bianbu build-ime·GGML_CPU_RISCV64_SPACEMIT=ON）。**rvv 列 N/A-hw（IME=SpacemiT 专属·硬件缺席·机判）**。

## durable files
- `A2-batch7-ime-kernel-sym.md`（本文）
- `A2-batch7-ime-kernel-sym-raw/ime_q40_kernelsym.cpp`（q4_0 kernel-sym paired 驱动·OURS 三形态{leaf/predec/w4} + VENDOR ime1::gemm_kernel_i8i4·ZERO-MODEL gate·链接 build-ime .so）
- `A2-batch7-ime-kernel-sym-raw/ime_q4k_kernelsym.cpp`（q4_K kernel-sym paired 驱动·OURS two-level fold + VENDOR q4_1x16-requant→gemm_kernel_i8i4·W=dequant(realistic q4_K)·gate）
- `A2-batch7-ime-kernel-sym-raw/logs/{q40,q4k}_run.log`（cold raw·2-seed·2-shape·byte-exact seal·md5 before==after·vendor 符号探针）

## 6. ★净结论一句话
本役首次以 kernel-sym cold 正面度量 weft IME GEMM 发射体 vs **SpacemiT 生产手调 vendor IME 核**（`ime1::gemm_kernel_i8i4`·真硅 vmadot）。**q4_0 输 5.1×·q4_K 输 20×·q8_0 无 vendor IME（BLOCKED-结构）**——2/2 可测格全 named-X·**0 我方赢**·**强对手诚实负结果**（**非便宜档·与 batch4/6 cheap-win 反向**）。具名 gap = **scale-fold epilogue 未融进 vmadot MAC**（vendor in-kernel fp16 fold·我方独立 scalar epilogue·随 super-block 复杂度暴涨 q4_0 25→q4_K 110ms）·= C3′ [PAT-1] format-keyed 适用边界最锋利证据 + IME 发射器成熟度目标具象。**与 e2e perf-covered 绿（stock-parity·另一赛道）分开报·禁混。** 全程 byte-exact（vmadot 0xe210312b 真硅 seal 512/512+64/64）·stock md5 unchanged·rvv=N/A-hw。
