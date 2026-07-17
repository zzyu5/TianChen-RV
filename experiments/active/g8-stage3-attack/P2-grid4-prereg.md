# P2 — grid 四格 repack-GEMM · 双板 8 格 · 预注册判读（PREREG · 测前写死 · 事后找补=违例）

> **状态**：**本文 = 预注册。零板上计时命令已执行。禁开测。**
> **范围**：`gemm_tile|{iq1_s,iq1_m,iq3_xxs,iq3_s}` × {rvv, k1} = **8 board-cells**（prefill nr=16 GEMM）。
> **起点真相**：这四格 byte-exact **仅在 x86 emitter-model 上**证过（C4a）· **零上板** · **零性能** · T3 `disp=pending-真(待补标量仗)`。
> **铁线**：cold 唯一 · 0 样本永不造数 · **[L-10] 板上必须自己再过一次 byte-exact** · 双板 clang-18 单世界 · 零 gcc 字样（`--gcc-install-dir` = clang CRT 定位 flag，既定 build recipe，非 gcc 世界）· **禁事前承诺性能（本文不含任何性能预测）** · 禁碰硬冻结（102/105 · 101/108 · 9/83 · roster `$meta` · 队序 = 只登记不执行）· 不改 T3/recon/lib/ · 禁 git add/commit。

---

## 0. 八格总表（逐格 · 档位 · 预注册判读）

| # | 格 | 板 | 注册对手（T3） | 档位 | as-shipped 真对手 | 判读键 |
|---|---|---|---|---|---|---|
| 1 | gemm_tile\|iq1_s | rvv | `ggml_vec_dot_iq1_s_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl128` | §5 |
| 2 | gemm_tile\|iq1_m | rvv | `ggml_vec_dot_iq1_m_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl128` | §5 |
| 3 | gemm_tile\|iq3_xxs | rvv | `ggml_vec_dot_iq3_xxs_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl128` | §5 |
| 4 | gemm_tile\|iq3_s | rvv | `ggml_vec_dot_iq3_s_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl128` | §5 |
| 5 | gemm_tile\|iq1_s | k1 | `ggml_vec_dot_iq1_s_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl256` | §5 |
| 6 | gemm_tile\|iq1_m | k1 | `ggml_vec_dot_iq1_m_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl256` | §5 |
| 7 | gemm_tile\|iq3_xxs | k1 | `ggml_vec_dot_iq3_xxs_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl256` | §5 |
| 8 | gemm_tile\|iq3_s | k1 | `ggml_vec_dot_iq3_s_q8_K_generic` | **标量类·便宜档** | OPP-X 手调 `_vl256` | §5 |

**★档位写死（不可翻）**：注册对手 = ggml `_generic` **标量参考兜底** ⟹ **便宜档** ⟹ **任何倍数（含大倍数）禁称硬赢**，必标 `compiler-artifact` / `opp-immaturity`。**0 verified hand-brick**（本役结构上不可能产出 hand-brick win：同算子对手不存在，§4.3）。

---

## 1. [GOV-8] 既有 harness 实况（**已核 · 别重建**）

### 1.1 可**直接复用**（不重写）

| 工件 | 复用什么 | 动否 |
|---|---|---|
| `P1-remainder-raw/iq4nl_gemm_prefill_p1r.cpp` | **ZERO-MODEL 骨架**：单一真源 W/A → 派生视图 → per-(r,c) oracle 零复用重算 → 三方门 → `worst_tolratio` 可审 → 反空心注入 1/2/3 → cold within-proc paired + flush-before-each + median/relIQR/2-seed | **照抄骨架** |
| `P1-remainder-raw/run_gemm_prefill_p1r.sh` | **双板 build recipe**（rvv `env.sh`+binutils-2.46.1 收 zvfh + CRT 定位 flag / k1 Bianbu clang-18）· **DUAL-AGREE 探针**（`awk -F'\t' '$3~/^[a-z]/'` + 整文件第二法）· load-gate · 单实例 `pgrep -x` · stock md5 before/after · **link flag 内联**（k1 zsh 不做 word-split，`"$L"` 会当单参数——静默断裂陷阱） | **照抄** |
| `P1-backfill7-prereg.md` | §4.4 具名-X 完整环 · §4.5 VOID 枚举 · §5 噪声门 · §6 措辞模板 | **照抄判据** |
| `tools/oracle-repack/oracle_repack_{iq1_s,iq1_m,iq3_xxs,iq3_s}.cpp` | **① 参考半 `ref_block()`**（ggml-canonical 生解码）→ **板上 oracle 直接用** · **② 系统化语料生成器 + coverage 计数器** · **③ `make_block_<fmt>x16()`（repack=mat-quant）** · **④ `check_layout_pins()`** | **复用①②③④** |
| `A2-batch6-iqtq-gemm-scalar-raw/` | 同类范式（iq/tq gemm vs scalar-ref·nr16·2-seed·cold）+ 三域对照纪律 | 参照 |

### 1.2 **必须改**（逐条 · 精确 · 每条都是本役已确证的实况）

1. **★[GOV-8-A] leaf 导出要两个 pass（batch6 只用一个）**。这四格的 fixture 装的是**抽象前门 op**（`weft_rvv.quant_contraction`），不是已降好的 typed body：
   ```
   build-weft/bin/weft-opt test/Conversion/RVV/rvv-emit-quant-contraction-<f>-repack-gemm-prefill-vlen128.mlir \
     --weft-rvv-lower-quant-contraction=march=rv64gcv \
     --weft-rvv-lower-to-emitc \
   | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp > kernels_grid4/<fmt>_gemm.c
   ```
   漏掉 `--weft-rvv-lower-quant-contraction=march=rv64gcv` ⟹ 抽象 op 不降 ⟹ 导出失败/空壳。**fixture 命名也与 batch6 不同**（batch6 = `rvv-to-emitc-repack-gemm-<f>-q8-K.mlir`；本役 = `rvv-emit-quant-contraction-<f>-repack-gemm-prefill-vlen128.mlir`）。
2. **★★[GOV-8-B] ABI 与 P1/batch6 **不同**——照抄 P1 的 extern 声明 = 静默错绑**。本役实测导出签名（四格**完全一致**）：
   ```c
   extern "C" void weft_emitc_ggml_repack_gemm_<fmt>_q8_K_kernel_ggml_repack_gemm_<fmt>_q8_K(
       size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
   //         v1         v2        v3         v4               v5               v6         v7
   ```
   **参数含义由代码实证反解**（非猜）：`v11 = v1/4` = 行组数（q8_Kx4 交织 4）⟹ **v1=nr**；`v12 = v7/16` = 列组数（x16 交织）⟹ **v7=nc**；store 点 `v4 + ((v13*4+r)*v2 + v17*16)` ⟹ **v2=bs**（输出行 stride，单位 float）；`__riscv_vsetvl_e32m1(v3)` ⟹ **v3=n**（=K）。
   对比 P1 iq4_nl = `(n, s, vx, vy, nr, nc, bs)` —— **顺序全变**。RISC-V LP64D 下整数与指针同走 a0–a7 ⟹ **错绑不必然崩，可能静默算出"看着合理"的数**（= 前科"探针静默退化恰好产生看似合理的小数字"的同型陷阱）。**⟹ 驱动必须带 §3.5 ABI 自证门。**
3. **[GOV-8-C] 对手换人**：P1 的 OPP-S（同算子 `ggml_gemm_iq4_nl_16x1_q8_0`）**本役不存在**（§4.3 已探针证）⟹ 删 OPP-S 分支，改为 **OPP-G（注册对手·`_generic`）+ OPP-X（as-shipped 手调 thunk）** 双对手（§4）。
4. **[GOV-8-D] 活化换格式**：P1 = `block_q8_0`/`block_q8_0x4`（34B/136B）；本役 = **`block_q8_K`（292B）/ `block_q8_Kx4`（1168B·quants@16 `pos*4+c`·bsums@1040 `g16*4+c`）**。本役已复算：`4+256+32=292` ✓ `16+1024+128=1168` ✓ `16+1024=1040` ✓（与 fixture 属性一致）。**iq1_s/iq1_m 读 bsums；iq3_xxs/iq3_s 无 bsums 面**（§2.3）。
5. **[GOV-8-E] oracle 换算术**：P1 = 4-bit nibble + codebook；本役 = **grid gather + ls + delta/sign 面**，四格 fold_model 各异（§2.3）⟹ 每格一个 `ref_block()`（**从 `tools/oracle-repack/` 取现成参考半，不新写**）。
6. **[GOV-8-F] 注入点换**：P1 INJECT=3 翻 codebook 字节（`sed 's/-127, -104, -83, -65,/…-64,/'`）；本役 leaf 内嵌的是 **`weft_iq1s_grid[2048]` / `weft_iq3xxs_grid[256]` / `weft_iq2xxs_signs64[1024]` 等表** ⟹ 注入点 = 翻**导出 leaf .c 里 grid 表的一个字节**，且必须 `cmp -l | wc -l` 自证**恰好 1 字节**差（P1 已有该自证，照抄）。
7. **[GOV-8-G] `weft-opt` 必须 forced 干净重建后再导出**（下条）。

### 1.3 leaf 导出（front-door 现生 · **禁 stale-archive** · **禁复用本会话 /tmp 试导产物**）

- **★强制**：`build-weft/bin/weft-opt` 现二进制 mtime = **07:46**，而 `HEAD=94a6898cc`（C4a-5 iq3_s，触 `lib/`）提交于 **07:54** —— 二进制**晚于**其源的提交时刻这一点无法从 mtime 证伪（很可能是先构建后提交），**但我不以推测代证**（前科：未经测试的因果断言 ×4）。⟹ **预注册动作 = forced 干净重建 `weft-opt` 后再导出**，并把 `weft-opt` md5 + `git rev-parse HEAD` + `git status --porcelain lib/ include/` 写入 `GEN_SEAL.txt`。
  - 本会话已用**当前（未重建）**二进制试导四格：四格均 `quant_contraction_left=0`、行数 21536/25168/24745/24489 —— **此产物仅用于反解 ABI，禁进测量**（= "陈旧基准数污染三轮"前科的对治）。
- **GEN_SEAL.txt 必录**：`weft-opt md5` · `HEAD` · `lib/ include/ 是否 dirty` · 每格 `<fmt>_gemm.c md5 + lines + 导出命令全文`。
- **报数纪律（写死）**：**本役报出的每一个数都要标（a）测量树状态 = `HEAD=<sha>` + `lib/ dirty=<yes|no>`，（b）是否 forced rebuild = `YES`。** 缺任一 → 该数不得入表。

---

## 2. 已执行的只读探针（**事实 · 非测量 · 本会话实跑**）

### 2.1 构造件在位（四格齐）

| 格 | registry | 前门 facts | prefill GEMM fixture | leaf 导出 | x86 oracle |
|---|---|---|---|---|---|
| iq1_s | ✓ | `kIq1SDecodeFacts` | `rvv-emit-quant-contraction-iq1-s-repack-gemm-prefill-vlen128.mlir` | ✓ 21536 行 | `oracle_repack_iq1_s.cpp` |
| iq1_m | ✓ | `kIq1MDecodeFacts` | `…-iq1-m-…` | ✓ 25168 行 | `oracle_repack_iq1_m.cpp` |
| iq3_xxs | ✓ | `kIq3XxsDecodeFacts` | `…-iq3-xxs-…` | ✓ 24745 行 | `oracle_repack_iq3_xxs.cpp` |
| iq3_s | ✓ | `kIq3SDecodeFacts` | `…-iq3-s-…` | ✓ 24489 行 | `oracle_repack_iq3_s.cpp` |

decode/GEVM 侧 fixture 也在（`rvv-emit-identity-quant-contraction-<f>-repack-vlen128.mlir`·`m_regime="decode"`·符号 `…_repack_gemv_…`），**但 T3 本轮在册 8 格 = gemm_tile(prefill)** ⟹ **GEVM 不在本役范围**；若日后测 decode，**必须独立 M=1 实测·禁继承本役 nr16 数**（§9）。

### 2.2 对手符号（**双板 nm -D 实探 · 本会话**）

| 板 | `libggml-cpu.so` md5 | `_generic`（注册对手） | 派发 thunk（as-shipped） | `ggml_gemm_<fmt>*`（同算子） |
|---|---|---|---|---|
| rvv | `e85fceda47606a115c5fbb0021817cdc` | 四格全 **PRESENT**（iq1_s@0x78d08 · iq1_m@0x79f56 · iq3_xxs@0x76bb8 · iq3_s@0x770d0） | 四格全 PRESENT（iq1_s@0xbc7c0 · iq1_m@0xbd13c · iq3_xxs@0xc08b8 · iq3_s@0xc0180） | **四格全 ABSENT** |
| k1 | `871169a0123139692177468b3c8578be` | 四格全 **PRESENT**（iq1_s@0x5dc72 · iq1_m@0x5eeec · iq3_xxs@0x5bb82 · iq3_s@0x5c022） | 四格全 PRESENT（iq1_s@0xa2c8c · iq1_m@0xa361e · iq3_xxs@0xa6d94 · iq3_s@0xa665c） | **四格全 count=0** |

k1 静态符号表另见手调克隆：`ggml_vec_dot_iq1_s_q8_K_vl128 @0xa2cd8` / `_vl256 @0xa2fb8` / `_vl512 @0xa31ea` / `_vl1024 @0xa33f6`（`t` = local）。

**⟹ 三条结论**：① T3 注记「无 repack GEMM → 标量参考兜底」**经探针证实**（同算子对手真不存在）；② 但**兜底的不是 `_generic`**——as-shipped 前缀路是 thunk → 手调 `_vlNNN`（§4.2）；③ **OPP-S 结构性缺席 ⟹ 本役不可能产出 hand-brick win**（§4.3）。

### 2.3 布局事实（**从 fixture 属性取 · 已与 ggml-common.h 复算对齐**）

| 格 | plain stride | x16 stride | ls@ | sign/delta@ | bsums | fold_model | 折算常数 |
|---|---:|---:|---:|---:|:--:|---|---:|
| iq1_s | **50** | 1312 | 32 | 160（±1 delta 条） | **@1040 有** | `grid_ternary_delta_eighth` | 0.125f |
| iq1_m | **56** | 1824 | 32 | 288 | **@1040 有** | `grid_ternary_delta_groupsum_eighth` | 0.125f |
| iq3_xxs | **98** | 1696 | 32 | 1184 | 无 | `grid_sign_dual_entry_single_scale_quarter` | 0.25f |
| iq3_s | **110** | 2720 | 32 | 2208 | 无 | `grid_sign_dual_entry_single_scale_unit` | 1.0（unit） |

plain stride 已按 `ggml-common.h` 逐格复算：iq1_s `2+32+16=50` ✓ · iq1_m `32+16+8=56` ✓（无 d 域，scales 编码）· iq3_xxs `2+96=98` ✓ · iq3_s `2+64+8+32+4=110` ✓。

### 2.4 ★发现：shipped fixture 里的假注释（第 4 起 · **本役不修 · 报给主会**）

`test/Conversion/RVV/rvv-emit-quant-contraction-iq1-s-repack-gemm-prefill-vlen128.mlir:33` 注释写 **"PLAIN block_iq1_s (stride 74, qs @2)"**，而同文件 :37 的属性写 `weight_block_stride = 50`，其 decode 姊妹 fixture :40 写 "stride 50"，`ggml-common.h` 复算 = **50**。⟹ **注释假，属性真**（74 无来源）。**仅注释缺陷，不影响降低/正确性**（驱动 lowering 的是属性）。与 C4a 提交日志「修好两条假注释」「同一假声明第二次进 shipped code」同族 ⟹ **第 4 起**。**不在本役修（禁改 lib/·此为 test/ 且非本役触碰集）**，登记待主会处置。

### 2.5 ★leaf 形状事实（**非性能预测 · 纯静态**）

四格导出 leaf 的 `__riscv_vse32_v_f32m2(…, 8)` —— **vl 常数恒 = 8**（四格一致）。⟹ leaf 是 **VLEN128 形状**（m2@SEW32@VLEN128 = 8 lane 满宽）；在 **k1 VLEN256 上 m2 = 16 lane，只用 8 = 半宽**。这与 P1 iq4_nl leaf 的处置相同（P1 记「leaf is VLEN-agnostic (all vl constants are 8)」）。
**⟹ 登记为已知形状事实，进 §7 措辞的 `{{vl_note}}` 空位。禁由此预测性能（[shape↛score] 第三次撞墙 · PR-37）。**

---

## 3. ★★板上 byte-exact 方案（**本轮头号 · [L-10]**）

### 3.1 x86 oracle 证了什么 / 没证什么（**先把话说死**）

`tools/oracle-repack/oracle_repack_iq1_s.cpp` 的结构（本会话实读）：
- `ref_block(const block_iq1_s*, const int8_t* q8, const int16_t* bsums, …)` = **ggml-canonical 参考**：读**原始 plain** 块，生解 `ls=2*((qh>>12)&7)+1`、`delta=qh&0x8000?-1:1`、`idx=qs[4*ib+l]|(((qh>>3l)&7)<<8)`，按 ggml 自己的 `(const int8_t*)(iq1s_grid+idx)` 取格。
- `mine_col(const block_iq1_sx16*, int c, …)` = **EMITTER-MODEL**（文件自己的注释就叫 "EMITTER-MODEL"）：**x86 C++ 手写的"我们建模的 leaf 行为"**——读重排后的 flat 条带、`grid_bytes[idx*8+j]` 平表 gather。

⟹ **C4a 的 0-mismatch = `ref_block` vs `mine_col` = 「ggml 参考」vs「我们对自己发射器的 x86 建模」**。
**它证的是：我们建模的布局/算术自洽。**
**它没证的是：`weft-opt` 真发射出来的那段 RISC-V，在真板上算得对。**
两者之间隔着：emitter 真实降低（vs 我们脑内模型）· RISC-V intrinsic 语义 · 编译器 codegen · 真硅（VLEN/tail/mask 策略）。**⟹ x86 绿 ≠ 板上绿。禁以 x86 oracle 冒充板上正确性。**

### 3.2 板上方案一句话

**把 `mine_col` 换成板上真跑的那段发射码，`ref_block` 原地留任 oracle。**
即：DUT = **交叉编译上板的真 leaf**；ORACLE = **`ref_block`（ggml-canonical 生解码）**。这样板上跑的这一遍**严格强于** x86 那一遍（同一 oracle，真 DUT 取代模型 DUT），且**复用已审代码**、不新造 oracle。

### 3.3 两层正确门（**T1 = 真 byte-exact · T2 = 测量形状门 · 两层都过才准报性能**）

#### T1 —— **板上整数路径 byte-exact 证书**（[K-5] 的正解 · `==` 位相等，不是"容差内"）

问题：leaf 只吐 f32（整数证书 sumi/sumi1 不出口）⟹ 直接比 f32 只能容差比 ⟹ 那不叫 byte-exact。
解法：**构造一段令 fp 折算【可证精确】的语料**，于是 f32 输出**恰好等于**整数表达式，可以 `==` 比。

- 取 **nb=1（K=256）** + **`d_x·d_y = 1.0`**（iq1_s/iq3_xxs/iq3_s 的 fp16 `d` 置 `0x3C00`=1.0 精确；q8_K 的 f32 `d` 置 1.0f；**iq1_m 无 d 域**，其 fp16 scale 由 `scales[]` 四 nibble 拼出 ⟹ 置成 `0x3C00`）。
- 则每个 (r,c) 输出 = `(float)(sumi + C*sumi1)`，C = §2.3 折算常数（0.125/0.125/0.25/1.0 —— **全是 2 的幂或 1** ⟹ 乘法本身精确）。
- **精确性已复算（iq1_s）**：`|sumi| ≤ 8 子块 × ls_max15 × (32×127=4064) = 487,680`；`|sumi1| ≤ 8 × 15 × (2×16×127=4064) = 487,680` ⟹ `|out| ≤ 487,680 + 487,680/8 = 548,640`，需 **20 位整数 + 3 位小数 = 23 位 ≤ 24 位尾数** ⟹ **f32 精确表示** ⟹ `ours[i] == (float)(sumi + 0.125*sumi1)` **可位相等**。（2^24 = 16,777,216 ≫ 548,640。）
- **★其余三格的界【不由我口算断言】**：驱动**运行时自算**每格上界并**自证**——若任一 `|中间量| ≥ 2^24` 或折算常数非 2 的幂 ⟹ 打印并 **`VOID-EXACTNESS-PRECONDITION`**，**不降级成容差门蒙混**。（对治前科「数标注不是数算术」：精确性由机器判，不由我写。）
- **门**：`mism == 0`，判据 = **f32 位相等 `==`**（非容差）。这就是 **[K-5] 整数路径字节精确 · 在板上 · 对真发射码**。
- **语料完备**（[K-5b]①）：nb=1 但 **nc=512 列 × 每块 8 子块 × 4 组 = 16,384 个 idx 槽** ⟹ 直接复用 `oracle_repack_*.cpp` 里**系统化扫描生成器**（idx 走 [0,2047] 全循环；ls 走 stride-3 循环覆盖全 8 档（3 与 8 互质）；符号位在 ls 循环上交替）+ **coverage 计数器**（`cov_idx` 集合 / `cov_delta[2]` / `cov_ls[8]`；iq3_xxs/iq3_s 另加 sign 面与 dual-entry 双入口覆盖）。**任一轴不满 → 驱动返回非零 → `VOID-CORPUS`**。**覆盖是结构性的（系统扫描），不是撞运气**，且**由计数器打印证明、不靠断言**。

#### T2 —— **测量形状 fp 门**（[L-10] 的第二半：**测的形状必须也过门**）

- 形状 = **真测形状**：K=2048 · nr=16 · nc=512 · 随机真实 `d`（`d_x·d_y ≠ 1`）· nb=64 跨块 f32 累加 ⟹ 折算重排序**必然**存在 ⟹ 只能容差门。
- 门 = P1 的**有原理**容差：`tol = TOLK · nb · FLT_EPSILON · amag`（`amag` = oracle 在 double 下算的 Σ|每块贡献|，前向误差界 `|fl(Σ)-Σ| ≤ nb·eps/(1-nb·eps)·Σ|term|`），TOLK=4.0。**`worst_tolratio` 必打印**（空心门会显示 ratio≈1；活门显示 ratio≪1 而故障远超 1）。
- **理由**：这是 "部署的 ≠ 证过的" 的对治——**T1 证的形状（nb=1）不是我们计时的形状（nb=64）**，所以计时形状**自己**也要过一道门。**两层都过，才准报性能数。**

### 3.4 [K-5b] 三要件 · 逐条落地

| 要件 | 落地 | 反空心 |
|---|---|---|
| **③ oracle 独立（禁与被测物共享解码实现）** | oracle = `ref_block`：读**原始 qh/qs**、按 ggml 生解 `ls/delta/idx`、按 ggml 自己的 `(int8_t*)(iq1s_grid+idx)` 取格。DUT = 真 leaf：读**重排 flat 条带**、`vluxei16` 平表 gather。**布局不同 · 索引推导不同 · 表读法不同**。 | 唯一共享 = `iq1s_grid` 本身 = **格式常量，非实现**（x86 oracle 已就此立论）。**★更强的一点**：oracle 的表来自 `tools/oracle-repack/iq1s_grid.h`，leaf 的表是**编译器从 registry 重建**的 `weft_iq1s_grid[2048]` ⟹ **两份独立副本** ⟹ registry 表若错，oracle 抓得到。 |
| **② 输入路径同源** | **单一真源**：一条 RNG 生成 plain `W`（`block_<fmt>`）+ plain `A`（`block_q8_K`）。所有消费者都是它俩的纯函数：`packed = make_block_<fmt>x16(W)`（复用 oracle 的 mat-quant）· `apack = make_q8_Kx4(A)` · OPP-G/OPP-X 直吃 plain W/A · oracle 只读 plain W/A。**全程零第二条 RNG 流**（P1 前轮 VOID 成因就是两条独立 RNG）。**零 intermediate 捕获重放。** | 驱动打印 `W/A` 的 md5-ish 指纹一次，三方共用同一份。 |
| **① 语料完备** | §3.3-T1 系统化扫描 + coverage 计数器 + 不满即非零退出。 | **MEASURE 形状（T2）不主张语料完备**（它是随机真 d）——完备性由 T1 承担，**两者分开报，禁互推**。 |

### 3.5 ★ABI 自证门（**[GOV-8-B] 的对治 · 新增 · P1 无此门**）

因为 ABI 顺序反直觉且错绑可能**静默**产出合理数，驱动必须在任何测量前跑：
1. **形状可分辨探针**：取 `nr=8, nc=32, bs=nc, K=256`，且**令 nr≠nc≠bs≠K 两两不等**（如 `nr=8, nc=32, bs=64, K=256`，输出缓冲按 bs 分配）⟹ 若 `nr`/`nc`/`bs` 任两者错位，输出的**非零区域形状**立刻不同。驱动断言：`s[r*bs+c]` 在 `r<nr, c<nc` 全被写过（哨兵值 `NaN` 预填，测后检查无残留 NaN），且 `r≥nr` 或 `c≥nc` 区域**仍是哨兵**（未被越界写）。任一违反 ⟹ **`VOID-ABI`**。
2. **`bs≠nc` 必测**：P1 调用恒传 `bs=nc`，会把 `bs`/`nc` 错位掩盖掉。本役**故意取 `bs=nc+16`** 跑一次哨兵检查，逼出 `v2` 是不是真 `bs`。
3. 通过后再进 T1。

### 3.6 反空心注入（**四臂 · 每臂预期写死 · 不达预期 = 门是空心的 = VOID**）

| 臂 | 手法 | **预注册预期** |
|---|---|---|
| `INJECT=0` | 干净 | GATE-OURS **PASS**（T1 位相等 mism=0） |
| `INJECT=1` | **oracle 常量故障**：`ref_block` 里 grid 索引旋转 `idx→(idx+457)&2047` | **三方全红**（ours/OPP-G/OPP-X 都 vs 坏 oracle） |
| `INJECT=2` | **DUT 输出故障**：`ours[mid] += 1.0f` | **仅 OURS 红** |
| `INJECT=3` | **★leaf 单字节故障**（[L-10] 专属臂）：翻导出 leaf `.c` 里 `weft_<fmt>_grid` 表的**一个字节**，`cmp -l \| wc -l` 自证**恰好 1** | **仅 OURS 红** ⟹ 证明**门咬的是真发射码**，不是模型、不是驱动 |

`INJECT=3` 是本役相对 x86 那一遍**新增的关键证据**：它证明板上这道门确实绑在**被发射出来的那段 RISC-V** 上。

---

## 4. 对手口径（**测前写死** · 三对手 · 一个不存在）

### 4.1 OPP-G = **注册对手**（T3 在册 · 便宜档）
`ggml_vec_dot_<fmt>_q8_K_generic`，逐 (r,c) over plain W/A，nrc=1 ⟹ scalar-ref GEMM。**T3 tier=标量类 的那个对手**。**便宜档 · 禁称硬赢。**

### 4.2 OPP-X = **as-shipped 真路**（**强制陪跑 · CROSSOP**）
`ggml_vec_dot_<fmt>_q8_K`（派发 thunk → 按 VLEN 选手调 `_vl128`/`_vl256`），逐 (r,c) over plain W/A，nrc=1。

**★为什么强制**：这四格 ggml **无 repack GEMM**（§2.2 探针证）⟹ prefill 时 `mul_mat` 落到**逐 (r,c) 的 `ggml_vec_dot_<fmt>_q8_K`** ⟹ 派发到**手调 `_vlNNN`**，**不是 `_generic`**。`_generic` 只在该 arch 无 `_vlNNN` 时才可达 = **arch 参考兜底**。
⟹ **这两块板上真正会跑的 prefill 对手是 OPP-X，不是注册的 OPP-G。**
⟹ **只报 OPP-G 的比值 = 报了一个"赢过板上根本不会执行的代码"的数** = 本役最大的诚实风险。
⟹ **写死：任何 PASS 措辞若不同时给出 OPP-X 比值，一律不得发出（§7.1 模板已把 OPP-X 设为必填位）。**

### 4.3 OPP-S = **同算子对手** ⟹ **结构性不存在**
`ggml_gemm_<fmt>_*` 双板探针 **ABSENT / count=0**（§2.2）⟹ 无同算子 repack-GEMM 可比。
⟹ **`hand-brick win` 资格本役【结构上不可授予】**（P1 的授予三条件之一「OPP-S 真手调」永假）。**写死：本役产出 `0 verified hand-brick`，不接受任何"赢了就算硬赢"的读法。**

### 4.4 对手身份三证 + **DUAL-AGREE**（每格 · 每板 · 缺一 = `VOID-PROBE`）
1. **符号名**：`nm -D` 取 `_generic` 与 thunk 地址（§2.2 已录，测时重录并比对 md5 一致）；`nm`（静态）取 `_vlNNN` 本地克隆。
2. **objdump 剖面**：`classify()` = P1 的 **TAB 字段感知 awk**（`awk -F'\t' '$3 ~ /^[a-z]/ {ins++; if($3~/^v/) rvv++; if($3~/^(vlux|vloxei|vrgather)/) gat++; if($3~/^vset/) vs++}'`）。
   **★DUAL-AGREE 强制**：法一 = `objdump --disassemble=<sym>`；法二 = 整文件 `objdump -d` + awk 范围截取。**两法 classify 字符串必须逐字相等**，否则 `DUAL-DISAGREE ⟹ VOID-PROBE`。
   **前科钉死**：单一 `grep -E "\tv[a-z]"` **禁用**——GNU grep 不解释 `\t`，会**静默退化**成匹配字面 `t`，且**退化方向恰好产出看似合理的小数字**（vset 计数冒充 rvv 计数）。
3. **源码归属**：`_vlNNN` = SpacemiT 手调 riscv 专化（`ggml-cpu/arch/riscv/`）；`_generic` = arch-fallback 通用 C 参考。给 `file:line`。
4. **thunk→克隆 绑定证**（证"板上真跑的是哪一个"）：反汇编 thunk 显示其**读 `vlenb` 并按之分派**；驱动**打印 `__riscv_vlenb()*8`**（rvv 应=128 → `_vl128`；k1 应=256 → `_vl256`）；再对被选中的克隆做 classify。三者一致 ⟹ 绑定成立。**若 thunk 分派逻辑与 vlenb 读数不能对上 ⟹ `VOID-PROBE`，不猜。**

---

## 5. ★逐格判读规则（**测前写死** · 落地照判 · 事后找补=违例）

### 5.0 通用
- **cold 唯一**：`ratio_cold = opp_med / ours_med`（>1 = 我方快）。**热 micro 一律不入账、不进措辞。**
- **样本**：N≥20（实取 **N=25**）中位数 + bootstrap CI · **2-seed** · 同会话 within-proc paired A/B/C 交替 · **每计时区前独立 flush**（rvv 224MiB > L3；k1 32MiB = 64× L2，k1 无 L3 ⟹ 真 DRAM cold）。**禁 warmup-discard**（= 改计时协议 = 改判据）。
- **形状**：**prefill nr=16 GEMM**（nr%4==0 · nc%16==0 · K=2048 · nc=512 · bs=nc）。**禁**用 decode 数充数 · **禁**由本役 prefill 数回填任何 decode 行。
- **★正确门前置（写死 · 不可绕）**：**T1（板上位相等）∧ T2（测量形状容差门）∧ ABI 自证门 ∧ 语料完备 —— 全过，才准跑计时。任一不过 ⟹ `VOID-CORRECTNESS`（或对应 VOID）⟹ 该格【禁报任何性能数】**，T3 维持 `pending`。
- **顺序写死**：rvv **先跑完**再跑 k1 · 禁并发（`.so` 竞争前科）· 每板同时仅一个 bench 进程。

### 5.1 判据（对 **OPP-G**（注册对手）读数）
- **两 seed 均 `ratio_cold ≥ 0.8`** → **PASS[0.8]**（**便宜档 · 禁称硬赢**）。
- **任一 seed `< 0.8`** → **走完 §5.2 完整环** → **具名-X[0.8]**。
- **跨 0.8 分裂**（一 seed ≥0.8、一 <0.8）→ **具名-X（保守）** + 标 `seed-split`。
- **OPP-X 比值**：**必测必报**，标 `CROSSOP` + `as-shipped`。**不决定本格 verdict**（本格注册对手是 OPP-G），**但强制出现在措辞里**（§4.2）。**OPP-X <0.8 不改本格 PASS/X 判定，但必须在成色注里明写"as-shipped 真路对手比值 = {{…}}×"**。
- **0.8 硬门应用 = 只登记不执行**：分母 102/105 = **硬冻结（禁碰）**。逐格产出 = `verdict[0.8] ∈ {PASS, 具名-X, VOID-*}` + **账归属建议**，**入表由主会话机算**。
- **账归属建议（沿 batch6 §5 既定 · 主会/用户裁）**：`scalar-ref(cheap)` ⟹ **建议 `test-only-not-in-denom`**（对手 = `_generic` arch-fallback，非 as-shipped 手调框架，**不入 0.8 硬门 matmul 分母**）。**★张力必须登记（不自决）**：batch3 有「对手真向量 → 升门」decree；若本役 objdump 显示某板 `_generic` 被 clang-18 autovec 出 `rvv>0`，则**字面触发升门**，与「§〇.1 源归属 = 标量类」冲突 ⟹ **canon 级口径冲突 ⟹ 必问**。本轮**只登记张力 + 双读数，不动 102/105**。

### 5.2 具名-X 的"完整环"（<0.8 时**必须走完**才准记 X · 懒认输 = 违例）
1. **对手身份确证**：§4.4 三证 + DUAL-AGREE → 排除稻草人 / 我方误链。
2. **我方核自探针**：`objdump` 数 vsetvl / gather / spill / size / fp16 libcall（`__truncsfhf2`/`__extendhfsf2` ⟹ SOFTFP 病）。
3. **墙具名**（**必须落到机制名，不许写"慢"**）：候选 = `codebook-gather-bound` / `vsetvli-storm` / `regfile-spill` / `narrow-vl(vl=8 @VLEN256 半宽)` / `memory-bound near-parity` / `opp-autovec-artifact`。
4. **可修性判**：可修（emitter 成熟度缺口 → 入队列，给队列项名）vs 不可修（uarch / 格式结构 → C3′ 负结果 · 档案级教材）。
5. **登记**：具名-X + 墙 + 可修性 + provenance 指针。**禁**只写 X 不给墙。**禁外推**（本格 X 不推同族 / 同板 / 他 regime）。

### 5.3 VOID 条件（**任一触发 → 该格 void · 如实报 · 0 样本永不造数**）
`VOID-EXPORT`（front-door leaf 导不出 / weft-opt 未 forced rebuild / GEN_SEAL 缺项）· `VOID-BUILD`（编译/链接 fail）· **`VOID-ABI`**（§3.5 哨兵门破）· **`VOID-EXACTNESS-PRECONDITION`**（§3.3 T1 精确性前提机检不过）· **`VOID-CORPUS`**（覆盖轴不满）· **`VOID-CORRECTNESS`**（T1 位相等 mism>0 或 T2 容差门 mism>0）· `VOID-PROBE`（三证不齐 / DUAL-DISAGREE / thunk 绑定对不上 / 非 as-shipped）· `VOID-LOAD`（load-gate idle<70%）· `VOID-NOISE`（§6 门破）· `VOID-HYGIENE`（stock `.so` md5 before≠after · STRAY>0）。
**void ≠ 失败 = 诚实产出。宁可无数，不可编数。禁贴伪造的命令输出。**

---

## 6. 噪声自检门（测前 · 测中 · 写死）

**历史地板（同类 · 同板 · 同协议 · 仅取 relIQR 形状，不取其数值结论）**：GEMM 类（K=2048·nc=512）ours relIQR ≈ **0.15–0.5%**；opp ≈ 0.1–7.2%（cold-outlier 易发）。batch6 同类 GEMM 实测 ours 2-seed 差 <2%（k1）/<5%（rvv）。

**测前（3 轮重测-重测 sanity · 同 config 连跑 3 轮）**：

| 门 | 阈 |
|---|---|
| ours relIQR | ≤ **5%**（每轮） |
| opp relIQR（OPP-G 与 OPP-X 各自） | ≤ **8%**（cold-outlier 容差） |
| 3 轮 ratio 中位数两两相对极差 | ≤ **2%** |

门不过 ⟹ **不开测**（先 load-gate 换核 / 查 co-tenant / kill 竞争进程）；**两次尝试仍不过 ⟹ `VOID-NOISE` 如实报**（停机规则：两次超时 = 必问去向）。

**测中**：任一格 relIQR ≥ **3× 该格测前 sanity 实测值** ⟹ **整会话作废重跑**。**禁摘录 · 禁挑轮次 · 禁"取好的那个 seed" · 禁 warmup-discard。**

**单实例（写死）**：测前 `pgrep -x <bench名>` 清零 + `pkill -x`（**`pgrep -f` 禁用**——会匹配到本条 ssh 命令行自身，P1 已踩）；`taskset` 单核钉；**rvv 只用 core 8–15（`0,1` = co-tenant vLLM，绝不碰）**；k1 idle-pick 0–7；测后 `STRAY` 计数入 log。**本役不动任何 LIVE 部署 `.so`**（纯 kernel-axis micro · 只读 stock）；**stock md5 before==after 双证**（rvv `e85fceda…` · k1 `871169a0…`）。

**验树 baseline（写死）**：测前记 `HEAD` + `git status --porcelain lib/ include/` + `weft-opt md5`；测后复记，**不一致 ⟹ 整批 VOID**。

---

## 7. ★措辞预注册（**留数字空位 · 落地填数即用 · 措辞不再构成回门理由**）

> 空位 `{{…}}`。模板空缺处标「待裁」⟹ **只问空缺、不问整体去向**（权限卡 ⑦⑧）。

### 7.1 PASS 模板（**便宜档 · 含强制 OPP-X 位 · 含成色注**）
> **`gemm_tile/{{fmt}}@{{board}}` prefill = PASS[0.8]**（cold **`{{ratio_G}}×`** vs **OPP-G**·seed1 `{{r_s1}}` / seed2 `{{r_s2}}`·N=25 中位·bootstrap CI `[{{ci_lo}}, {{ci_hi}}]`·relIQR ours `{{o_iqr}}%` / opp `{{p_iqr}}%`）。
> **板上正确性（[L-10] 已闭）**：**T1 板上整数路径 byte-exact = `{{mism1}}`/`{{tot1}}` mismatch，判据 = f32 【位相等 `==`】**（nb=1·d_x·d_y=1.0·精确性前提由驱动机检通过）·语料完备实测 `grid_idx {{cov_idx}}/2048 · ls {{cov_ls}}/8 · delta {{cov_d}}/2{{·sign …}}`（系统化扫描·计数器打印）。**T2 测量形状（K=2048·nr=16·nc=512·真随机 d）**：`{{mism2}}`/`{{tot2}}`·`worst_tolratio={{wtr}}`（≪1 = 活门）。**反空心四臂**：INJECT=1 三方全红 ✓ · INJECT=2 仅 OURS 红 ✓ · **INJECT=3（leaf 单字节 grid 故障·恰 1 字节差自证）仅 OURS 红 ✓ ⟹ 门咬的是【真发射的 RISC-V 码】，非 x86 模型**。
> **★成色（诚实 · 铁线 · 不可翻）**：注册对手 = **ggml `_generic` 标量参考兜底** ⟹ **便宜档** ⟹ **禁称硬赢**，标 `{{compiler-artifact | opp-immaturity}}`（objdump：opp ins=`{{ins_G}}` rvv=`{{rvv_G}}` ⟹ `{{纯标量 vec=0 | clang-autovec 编胖}}`）。
> **★★as-shipped 真路对照（必填 · CROSSOP）**：这四格 ggml **无 repack GEMM**（`ggml_gemm_{{fmt}}_*` 双板探针 ABSENT）⟹ 板上 prefill 真正执行的是 **thunk → 手调 `ggml_vec_dot_{{fmt}}_q8_K_vl{{128|256}}`（SpacemiT riscv 专化）**，**不是 `_generic`**。**vs OPP-X = `{{ratio_X}}×`** ⟹ `{{PASS|具名-X}}`（vlenb 实测 `{{vlen}}` · 三证 + DUAL-AGREE ✓）。**⟹ 上面那个 `{{ratio_G}}×` 是【赢过板上不会执行的 arch-fallback 参考】，禁读作"beat ggml"。**
> **同算子门**：OPP-S **结构性不存在** ⟹ **hand-brick 资格不授予 · 本格 0 verified hand-brick**。
> **形状注**：leaf `vse32 vl=8` 恒定 = VLEN128 形状；`{{k1: 于 VLEN256 上 m2=16 lane 仅用 8 = 半宽 | rvv: VLEN128 满宽}}`（**静态事实·非性能归因**）。
> **账**：`gemm_tile` matmul **kernel-sym MICRO** · 建议 **`test-only-not-in-denom`**（scalar-ref 便宜档）· **NOT e2e · NOT perf-covered 9/83 · 不入 matmul headline · [NG-4]**。硬冻结未碰。
> **测量树**：`HEAD={{sha}}` · `lib/ dirty={{no}}` · **weft-opt forced rebuild = YES**（md5 `{{md5}}`）· stock `.so` md5 before==after（`{{md5so}}`）。

### 7.2 具名-X 模板（**须走完 §5.2 完整环**）
> **`gemm_tile/{{fmt}}@{{board}}` prefill = 具名-X[0.8]**（cold `{{ratio_G}}×` vs OPP-G·seed1 `{{r_s1}}` / seed2 `{{r_s2}}`·N=25·CI `[{{ci_lo}}, {{ci_hi}}]`{{·seed-split}}）。
> **板上正确性 = 过**（T1 `{{mism1}}`/`{{tot1}}` 位相等 · T2 `{{mism2}}`/`{{tot2}}`·wtr=`{{wtr}}`·四臂反空心 ✓）⟹ **输的是性能，不是正确性**（我方核在真板上算得对，[L-10] 已闭）。
> **对手身份三证**：符号 `{{sym}}` @`{{addr}}` · objdump ins=`{{ins}}` rvv=`{{rvv}}` gather=`{{gat}}` vset=`{{vset}}`（**DUAL-AGREE ✓**）· 源归属 `{{file:line}}` ⟹ **as-shipped `{{标量类 arch-fallback}}`**（非稻草人）。
> **我方核自探针**：vsetvl=`{{vs}}` gather=`{{g}}` spill=`{{spill}}` size=`{{sz}}`B fp16=`{{cleanfp|SOFTFP}}`。
> ★**墙（具名 · 机制级）**：`{{codebook-gather-bound | vsetvli-storm | regfile-spill | narrow-vl(vl=8@VLEN256 半宽) | memory-bound near-parity | opp-autovec-artifact}}` —— `{{一句话机制}}`。
> **可修性**：`{{可修(emitter 成熟度缺口·入队列 → {{队列项}}) | 不可修(uarch/格式结构·C3′ 负结果·档案级教材)}}`。
> **成色**：**便宜档**（对手 = `_generic` 兜底）⟹ **输给标量参考 = 我方核 `{{codegen}}` 问题，比"输给手调"更该记**。as-shipped 真路 OPP-X = `{{ratio_X}}×`。
> **provenance**：`{{raw 路径}}`。**禁外推**。**测量树**：`HEAD={{sha}}` · forced rebuild=YES。

### 7.3 VOID 模板
> **`gemm_tile/{{fmt}}@{{board}}` prefill = VOID-`{{EXPORT|BUILD|ABI|EXACTNESS-PRECONDITION|CORPUS|CORRECTNESS|PROBE|LOAD|NOISE|HYGIENE}}`** · **未测 · 0 样本 · 不造数**。
> **精确缺口**：`{{一句话·可执行}}`。**已就绪**：`{{已有工件}}`。**剩余构造**：`{{具体步骤}}`。**板**：`{{rvv|k1}}`。
> T3 该格 **维持 `pending-真(待补标量仗)` · 禁填**。**预判不作结论。**

### 7.4 净结论骨架（收口用）
> P2 八格：**PASS `{{p}}` / 具名-X `{{x}}` / VOID `{{v}}`**。**[L-10] 闭环**：`{{n}}` 格在**板上自己过了 byte-exact**（T1 位相等 + T2 测量形状门 + INJECT=3 证门咬真发射码）⟹ C4a 的 x86 emitter-model 绿**已被板上独立复证 / 未被复证**`{{…}}`。**便宜档 `{{n}}` 格 · 全部禁称硬赢**（对手 = `_generic` arch-fallback）。**OPP-S 结构性不存在 ⟹ 0 verified hand-brick**。**as-shipped 真路（OPP-X 手调 `_vlNNN`）比值已逐格并列，禁以 OPP-G 数冒充"beat ggml"**。账：`{{n}}` 格建议 `test-only-not-in-denom` · **NOT e2e · NOT perf-covered 9/83**。**硬冻结未碰**（102/105 · 101/108 · 9/83 · roster `$meta` · 队序 = 只登记不执行）。**零 gcc 输出。** 未决 canon 张力：`{{scalar-ref 升门 decree 冲突 | 无}}`。

---

## 8. ★禁继承的历史锚（**登记 · 不得作本轮结果 · 不得作对照头条**）

| 锚 | 值 | 为何禁 |
|---|---|---|
| `vec_dot\|{iq1_s,iq1_m,iq3_xxs,iq3_s}` 双板 cold（T3 在册） | rvv 0.365 / 0.152 / 0.156 / 0.213 · k1 0.361 / 0.25 / 0.192 / 0.2 | **不同轴**（vec_dot 非 repack-GEMM）· **不同对手**（手调 `_vlNNN` 非 `_generic`）· **不同 regime**。**禁继承 · 禁当预测。** |
| `T3p-X_xfer1_prediction_k1.csv` 四格 `LOSS/NO-OP (pred)` + `[GAP-CLANG-GATHER-TRAP]` | 预测值 | **是 prediction 不是 measurement**（`(pred)` 标记 + `measured_k1=no`）。**禁事前承诺性能 · 禁把预测当结论**（[shape↛score] 第三次撞墙 · PR-37）。**本文不含任何性能预测。** |
| batch6 §2.1–2.3 的 7 格 iq/tq 比值（2.1–10.3× / 0.35–0.75× / 2.5–4.8×） | — | **不同格式**（那 7 格不含本役四格）· batch6 §4 明确把本役四格判为 BLOCKED。**禁外推。** |
| C4a x86 oracle 的 `0-mismatch` | — | **是 emitter-model 上的绿**（§3.1）⟹ **不得冒充板上正确性**。板上必须自己再过一次（§3.3）。 |
| A2-batch6 rvv 双域（gcc/clang）结论 | — | 本役 = **单世界 clang-18 双板对称**（PR-17）⟹ 不复用 gcc 域读数、输出零 gcc 字样。 |

---

## 9. 风险登记（**测前 · 诚实 · 不掩**）

1. **★ABI 静默错绑**（最高）：新 ABI `(nr, bs, n, s, vx, vy, nc)` 与 P1 `(n, s, vx, vy, nr, nc, bs)` 全序不同；LP64D 下整数/指针同寄存器族 ⟹ 错绑可能**不崩而静默出合理数**。**对治 = §3.5 哨兵 ABI 自证门 + 故意 `bs≠nc`。**
2. **★T1 精确性前提在非-iq1_s 三格未经我复算**：我只手算了 iq1_s（23 位 ≤ 24）。**对治 = 驱动运行时机检并在不满足时 `VOID-EXACTNESS-PRECONDITION`，不降级成容差蒙混**（对治「数标注不是数算术」前科）。
3. **★iq1_m 无 `d` 域**：其 fp16 scale 由 `scales[]` nibble 拼出 ⟹ T1 的"令 scale=1.0"需拼 `0x3C00`，**比其余三格易错**。**对治 = `check_layout_pins()` 复用 + T1 前先断言 `实际scale == 1.0f`，不等即 VOID。**
4. **★OPP-G vs OPP-X 的报数诱惑**：OPP-G 是便宜档，比值可能很大；OPP-X 是板上真路。**只报 OPP-G = 报了赢过不会执行的代码。对治 = §7.1 模板把 OPP-X 设为必填位，缺位不得发出。**
5. **`weft-opt` 与 HEAD 的一致性无法由 mtime 证**（binary 07:46 < commit 07:54，很可能先构建后提交）。**对治 = forced 干净重建 + GEN_SEAL 记 md5/HEAD/dirty；本会话试导产物仅用于反解 ABI，禁进测量。**
6. **`_generic` 被 clang-18 autovec ⟹ 升门 decree 字面触发**，与「源归属 = 标量类」冲突 = **canon 级 ⟹ 必问**，本轮只登记（§5.1）。
7. **leaf vl=8 @ k1 VLEN256 半宽**：静态事实已录（§2.5）。**风险 = 被读成性能预测。对治 = 只作 §5.2 墙候选之一，须由 objdump + 实测支撑才可具名，禁事前断言。**
8. **grid 表体积**（iq1_s 2048×u64 = 16KB；iq3_s 表更大）⟹ leaf `.o` 巨大 + cold flush 需覆盖。**对治 = flush 尺寸沿用 P1（rvv 224MiB / k1 32MiB）并在 seal 记 leaf size。**
9. **rvv co-tenant vLLM**（core 0,1）⟹ 噪声源。**对治 = load-gate idle-pick 8–15 + relIQR 门 + 测中 3× 突劣整会话作废。**
10. **假注释族（§2.4 第 4 起）**：说明该 fixture 家族的注释**不可作事实依据** ⟹ **本役所有布局事实一律取【属性 + ggml-common.h 复算】，不取注释。**（已照此执行。）

---

## 10. 预注册的后续动作（权限卡 ⑦⑧ · 判据满足即直行 · 不回门）

- 判据满足（T1∧T2∧ABI∧语料∧探针∧噪声全过）⟹ **直接按 §5.1 判 + §7 模板填数发布**，事后报备一行，**不回门**。
- 触 `VOID-*` ⟹ 按 §7.3 如实报 void + T3 维持 pending，**不造数、不降级门**。
- 触 **canon 级**（§5.1 升门张力 / §2.4 假注释处置 / 分母口径）⟹ **只登记 + 必问**，不自决。
- 触 **停机规则**（噪声两次不过 / 预注册判读失败无预案）⟹ 报去向、请裁。
