# G8 §六.3 — [BUG-K1-Q2K-GEMM-WRONG] ZERO-MODEL adjudication

> **性质**: 定夺 ours q2_K GEMM 是 **真 kernel bug** 还是 **feeding-mismatch**。前序 8x8
> (kquant-gemm-k1-8x8/evidence.md §4) 判 "ours q2_K 核算错(真 bug)·WIN 撤销"。本役 ZERO-MODEL
> 实测 **翻案该结论**: ours q2_K 核在其**自身(部署)布局**上 byte-exact-fp 正确; 前序 all-wrong =
> **喂了 ggml stock repack 的置换 scale 布局**(feeding-mismatch)。
> **[NG-4] correctness-only**·非 e2e·stock .so 只读(md5 871169a0)·core0 pin·NO git·主树/build 未动。

---

## 0. 定夺 (headline)

**ours q2_K GEMM = CORRECT (feeding-mismatch·前序喂错布局)。NOT a real bug。NO deployed correctness 缺陷。**

k1 VLEN256 · ours **实际编译核**(census 导出 `weft_emitc_ggml_repack_gemm_q2_K_q8_K`, unrolled=deploy default)
· 一份**独立 canonical scalar ref**(ggml vec_dot_q2_K_q8_K 整数 isum/summs 从量化定义重算) · 三 shape:

| shape | ours on **STRAIGHT**(ours/部署布局) | ours on **GGML-STOCK**(置换布局) | STOCK gemm on GGML-STOCK |
|---|---|---|---|
| nr8 nc256 K2048 | **nbad=0/2048** max_rel 9.4e-5 ✔ | nbad=2028/2048 max_rel **2.0** ✘ | nbad=0/2048 ✔ |
| nr4 nc32 K2048 | **nbad=0/128** max_rel 1.1e-6 ✔ | nbad=128/128 max_rel **1.98** ✘ | nbad=0/128 ✔ |
| nr16 nc256 K512 | **nbad=0/4096** max_rel 8.6e-5 ✔ | nbad=4066/4096 max_rel **2.0** ✘ | nbad=0/4096 ✔ |

- **ours-STRAIGHT nbad=0** = ours 核在自身布局上 byte-exact-fp(残差 1e-6..1e-4 = IEEE reassoc 噪声, 非结构错) → **核算对**。
- **ours-GGML-STOCK max_rel 2.0 nbad≈99%** = **精确复现前序 8x8 的 max_rel 2.0 all-wrong** → 前序喂的是 ggml stock repack。
- **STOCK-on-GGML-STOCK nbad=0** = ggml stock repack+gemm 自洽(**验证独立 ref 正确** + 置换布局确是 stock 产物)。
- 原始: `raw/run_adjudicate.log` · `raw/build_seal.txt`。

---

## 1. 根因: ours 期望布局 vs ggml-canonical 差异 (仅 scale 子块顺序)

ours q2_K GEMM 读 `block_q2_Kx16`(stride 1344): `d[16]@0 · dmin[16]@32 · scales[256]@64 · qs[1024]@320`
(MLIR fixture `weight_*_byte_offset` 印证; oracle_repack_q2K.cpp 结构定义)。**内部字节排布两处**:

| 区 | ours 期望 (STRAIGHT) | ggml stock `make_block_q2_Kx16` | 一致? |
|---|---|---|---|
| qs[1024]@320 | `qs[i*16+c]=in[c].qs[i]` (byte-granular) | interleave_block=**1** → **同 byte-granular** | ✔ **同** |
| scales[256]@64 | `scales[s*16+c]=in[c].scales[s]` (**顺序** s=0..15) | **"Sequential-Parallel" 置换** {0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15} | ✘ **异** |

- 唯一差异 = **256B scales 区的子块顺序**(qs/d/dmin/stride 全同)。本役 `ggml-real-repack vs hand-permuted diff bytes = 0`
  证实置换序 = `{0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15}`(ggml repack.cpp:3163-3197 even-low/odd-low/even-high/odd-high)。
- ours 把物理 chunk s 当子块 s 读 → 除 chunk 0/8/15 外全错 → nbad≈99% max_rel 2.0。
- **interleave_block=1** 确认: `repack<block_q2_K,1,16>` → `repack_q2_K_to_q2_K_16_bl(t,1,...)`(repack.cpp:3955)。

**为何 q4_K 前序 drop-in PASS 而 q2_K 不**: ggml stock `make_block_q4_Kx16`(repack.cpp:2913, interleave=1 分支)把
6-bit packed scale 拆成特定 unpacked 布局, **ours q4_K emitter 复刻该布局** → 与 stock **顺序一致** → drop-in nbad=0。
q2_K stock 独有 "Sequential-Parallel" scale 置换, ours q2_K emitter 用**顺序**布局(未复刻置换) → 与 stock **不** drop-in。
∴ q4_K 恰好 drop-in stock / q2_K **需部署补丁改 stock repack**(见 §2)。前序 8x8 §4 caveat 假设"q2_K/q4_K 皆 drop-in
same layout"是**错的**——正是它 flag 的 "若 ours-q2_K 独一无二地期望非 canonical 布局则属喂法失配" 情形。

---

## 2. 部署自洽性: 无 deployed 缺陷 (repack 与 GEMM 成对补丁)

`tools/e2e-harness/board/g5-q2_K-k1/deploy_patch_q2_K_emitted.py` **同时补丁两处**(自身注释 line 9-17):
1. **GEN repack.cpp `make_block_q2_Kx16`**: stock "Sequential-Parallel" scale 置换 → **ours STRAIGHT 16-way**
   (`scales[s*16+c]=in[c].scales[s]`)。注: "Our emitted kernel expects the straight-interleave scale layout"。
2. **arch/riscv GEMM/GEVM body** → 调 ours emitted 核。
- 结论: 部署 = **ours make(straight) + ours kernel**, 自洽正确; stock = **stock make(perm) + stock kernel**, 自洽正确。
  唯一 A/B diff = q2_K 计算本体。**部署路径无 correctness 缺陷。**
- ours q2_K GEMM **未**以 "consume stock ggml 未补丁 repack buffer" 方式部署(那才会错)。前序 8x8 census 用的是
  **未补丁 stock repack**(`repack<q2_K,1,16>` 自由函数) → 对 ours 是错误的 feed, 对 stock gemm 是正确的 feed。

---

## 3. 前序 cert 覆盖判断 (924dc31f / 05f96739 / verify) — 未漏本路径

| cert | 测什么 | 布局 | 覆盖本 GEMM 路径? |
|---|---|---|---|
| oracle_repack_q2K.cpp (host, 924dc31f 家族) | emitter-model 算术 vs canonical ref | **STRAIGHT** (make_block_q2_Kx16 顺序) | 是·byte-exact-integer(本役 host 复跑 GREEN: 7 shape + GEMM x4 int-mismatch=0, 负控 2BIT/SCALE-ROT/MIN-ROT 各 100%) |
| kquant_repack_verify_q2K.c (silicon) | **实际编译核** vs canonical ref | **STRAIGHT** (`blk[64+s*16+c]`) | 是·部署布局 |
| 05f96739 (KNEST G1) | 结构/指令账 + q2_K G1 静态 | STRAIGHT | 是 |

- 三 cert 全测 **STRAIGHT = 部署布局** → **非空心证书**: 输入路径与部署 repack(补丁后)**同源**, 覆盖了本 GEMM 路径。
- 前序 8x8 census 的 all-wrong **不 contradict** 这些 cert: 它测的是 **stock 未补丁 repack** 布局(部署不用的 feed),
  是 census 自身的 harness 选择, 非 cert 覆盖的路径。∴ 无证书失效, 无 deployed bug。
- 本役新增第四证据: ours **实际编译核** on STRAIGHT nbad=0 (silicon) + on stock-perm all-wrong (silicon) 双向对照,
  把 "STRAIGHT 正确 / stock-perm 错" 钉在同一 harness 同一 ref。

---

## 4. 时序可否公平重测 (feeding-mismatch 后续)

可以。straight vs perm **仅 scale 子块顺序异**: stride 1344 同 · qs/d/dmin 同 · 激活 q8_Kx4 同 · work-volume 内容无关。
∴ ours q2_K 的运行时与 scale 顺序**无关**。前序 8x8 census 的 q2_K timing (1.364×/1.353× vs 16x1 hand-brick) 本就
在 stride-1344 buffer 上测(WoursT), 对手 stock gemm 也在 stride-1344(其 perm buffer)→ **timing 公平**, 现 correctness
已证 → **该 ratio 可作 timing-valid WIN 重新起复**(caveat: ours 需部署补丁改 stock repack, 非 stock drop-in; 与 q4_K
的 "stock-drop-in byte-verified" 成色不同——q2_K = "部署补丁-repack byte-verified", q4_K = "stock-drop-in byte-verified")。

---

## 5. 跨板 (rvv) 影响

**无 bug → 无跨板修法需求。** ours q2_K 核算术正确(byte-exact-fp), rvv 侧同 emitter 同 straight 布局同样自洽
(rvv 部署若走同 straight-repack 补丁则同样正确)。**先别改发射器**(本役无源码改动)——无 bug 可修; 若主会要统一
"stock-drop-in" 成色, 那是**让 ours q2_K emitter 复刻 stock Sequential-Parallel 置换**的 enhancement(消除 repack 补丁
依赖, 使 q2_K 与 q4_K 同为 stock-drop-in), **非 correctness 修复**·结构级·主会裁。

---

## 6. 板卫生 · 触碰 · 禁 git

- core0 pin(idle 100%)· loadavg 2.5–2.6 co-tenant · stock .so 只读 md5 **871169a0**(前后一致)· scratch
  `/tmp/g8k1_q2kadj`(用完已清)· census 核 `/tmp/g8k1_gemm_census/kernels`(前役产物·只读复用·未改)。
- 触碰(本机·未 commit): 本 casefile `q2k-correctness/`(q2k_adjudicate.cpp · run_adjudicate.sh · evidence.md · raw/)。
  **未 git add/commit · 未改 T3_B/T8/发射器**(主会审整合)。
