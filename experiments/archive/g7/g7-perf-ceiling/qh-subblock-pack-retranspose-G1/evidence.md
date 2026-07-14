# G7 — [GAP-EMIT-KNEST-QH-SUBBLOCK-PACK] 可行性评估：q5_K qh re-transpose 恢复 vlm native-mask（纯本地 G1·零板·零源码改动）

> 前序：q5k-knest-G1（`../../g7-l2-gevm-redesign/q5k-knest-G1/evidence.md`·`7a45a2fe`）具名 **[GAP-EMIT-KNEST-QH-SUBBLOCK-PACK]**（q5_K sub-block-bit-packed qh 阻断 full REDESIGN-B vlm 杠杆）；native-mask 单-bit-plane 元件 **DEPLOYED 到 q5_0/q5_1/q3_K**（`../../g7-l2-gevm-redesign/q3k-native-mask-deploy/evidence.md`·`df7dae7d`·组合式 extensibility 3/4 qh 格）。
> **本线目标（诚实前提·定 scope）**：评估 **re-transpose q5_K qh repack 布局** 能否恢复 vlm native-mask 杠杆·**完成 native-mask 组合式 extensibility 到 q5_K（4/4 qh 格）= C1 完整性评估** + informs **值不值得付 repack-layout 改的风险**。**非 perf mover**（[CASE-MICRO-E2E]·q5_K M=1 GEVM instruction-count 3.76× stock 结构限已证·本线 perf=零·明记）。
> 依据 canon：[VERIFY-LADDER] G1 · [K-10] · [CASE-MICRO-E2E] · [CASE-COMPILER-ASYMMETRY]（部署 gcc-15.2 域·VLEN128）。
> **止于 G1（静态账·hand-construct）· 禁 git · 禁上板 · 禁改 tracked 源码**（emitter/packer 只读·hand-construct 全在 `raw/`）。

---

## ★裁决 TL;DR
1. **结构可行 = ✅ 是**：re-transpose = 对每个 qh position m 的 **16-column × 8-sub-block bit-tile 做位转置**（byte-per-column → 16-bit-mask-per-sub-block），使单 sub-block 的 mask 变 `vlm_v_b16`-loadable（lane c = bit c·与 q5_0/q5_1 REDESIGN-B 同构）。
2. **byte-exact = ✅ PASS（含 packing round-trip）**：`raw/be_retranspose` 硬门全绿——(A) qh round-trip 双射 0 mismatch（popcount 2096=2096·纯 permutation·**权重值不变只重排**）·nibble 区零改动·(B) 5th-bit decode stock≡current≡retrans **0/4096**·(C) full 5-bit weight **0/4096**。
3. **qh-recon 削减 = ✅ 兑现（比 sub-block KNEST 更强）**：rolled per-(ii,h)（gcc-15.2·VLEN128）v-insn **16→12（−25%）**·qh-recon-arith **8→4（−50%）**·`vsll/vor/vsrl-sel` 全 RETIRED·mask **vlm 直装**（非 vand+vmsne 抽取）= **full q5_0 vlm 杠杆**（2 op/臂·vs OLD 4·vs sub-block-native-KNEST 3）。
4. **影响面 = ⚠ HIGHER-RISK（三重）**：① **repack-layout 改**（offline packer `kqr_repack_q5_K` verbatim→bit-transpose）② **两路一致强制**（GEVM leaf ∧ GEMM leaf **共享 block_q5_Kx16 ABI**·两 leaf qh-recon 须同改）③ **sealed md5 re-seal**（q5_K kernel `ba30ba54` 变·须板 re-validate）。**RAW-qh 路（flat dequant injectQh）读 on-disk `x[i].qh`·不读 repack·UNAFFECTED**（收窄 blast radius）。
5. **值不值得（informs）= ⚠ 不值得（现在）·durable 结论**：re-transpose 结构可行 + byte-exact·但 **①C1 组合式 extensibility 已被 3/4 vlm 格 + q5_K 的 sub-block-native-mask KNEST（`../q5k-knest-G1` 已 byte-exact 证·decode-only 低风险）充分证**；**②"native-mask 到 q5_K"的【廉价路】= sub-block native-mask KNEST（decode-only swap·同 q3_K deploy 低风险·无 packer/无两路/无 re-seal）·非 re-transpose**；③re-transpose 唯一净增 = uniform vlm 杠杆（3→2 op/臂）·**perf=零**·不值 repack-layout 三重风险。
6. **诚实定性**：**C1 组合式 extensibility 完整性评估·非 perf**（[CASE-MICRO-E2E]）·**有效 informs = 不值得/higher-risk**：q5_K 用非-vlm qh 解码（现 OLD·或廉价 sub-block-native-KNEST）已 byte-exact 工作·组合式 extensibility 3/4 vlm + q5_K-覆盖已足证泛化·re-transpose 是 perf-零 refinement 不抵 repack 风险。设计与 byte-exact 模型已就绪（本 casefile），若主会话未来立项可直取。

---

## 1. ★阻断根因精确刻画（q5_K sub-block-pack vs q5_0/q5_1/q3_K vlm 布局）

### 1.1 vlm-loadable 的充要条件（从 3 个已部署格反推）
`vlm_v_b{N}` 直装 native mask 要求：**对一个 decode 步·N 个 vector lane 上的 mask 是【连续 N-bit run·byte-aligned·每 lane 一 bit·lane l = bit l】**。
- **q5_0/q5_1（REDESIGN-B·`RVVToEmitCBlockQuantLinear.cpp:1651-1720`）**：transposed qh = per-K-position 的 16-bit mask（`mask[e] bit c = col c 的 qh bit e`·`q51_repack_cert.c:154`）→ `qhStripMaskBytes=half/8`·`vlm_v_b16` 直装·lane c = bit c。**1 bit/element·转置后 16-col 连续**。
- **q3_K（deployed·`df7dae7d`）**：hmask 3rd-bit **单-bit plane**·同 native-mask（`vmseq`+`vadd_mu`）。
- **∴ 3/4 qh 格 = 单-bit-plane·mask 连续可 vlm/native**。

### 1.2 q5_K 为何阻断（sub-block-bit-packed byte）
`block_q5_K.qh[32]`：256 bit = 每 element 1 高位·但 **byte qh[m] 的 8 bit 属 8 个不同 sub-block**（同 intra-position m·ggml `u1/u2<<=2` plane walk）。current repack（`kquant_repacker.h:119`·**verbatim copy**）：
```
blk[256 + m*16 + c] = x.qh[m];   /* 32 qh bytes·"no bit transpose"·8 planes/byte 保留 */
```
decode 单 sub-block s（`RVVToEmitCBlockQuantLinear.cpp:9133-9140` GEVM · `9942-9949` GEMM·**同 idiom**）：
```
loSel = vsrl(qhStrip, sLoBit);  loBit = vsll(vand(loSel,0x01),4);  nLo = vor(loNib, loBit);   // 4 op/臂
```
**根因**：对 16-col strip 解 sub-block s·相关 16 bit = 16 个不同 byte 的 **bit-position s**（byte-strided·**非连续 bit-run**）→ **不可 vlm 直装**·必 `vand(1<<s)+vmsne` 抽（或现行 OLD 的 `vsrl+vand+vsll+vor`）。**"byte 装 8 sub-block bit" 令单 sub-block mask 在 lane 间 strided·破坏 vlm 的 lane=bit 连续性**——这就是 sub-block-pack 阻断。

---

## 2. ★re-transpose 布局设计

### 2.1 核心变换 = 每 position m 的 16×8 bit-tile 位转置
| | current（sub-block-pack）| **re-transpose（sub-block-plane）** |
|---|---|---|
| 组织 | byte per (m, col c) = 8 sub-block bit | 16-bit mask per (m, sub-block s) = 16 col bit |
| 存放 | `blk[256 + m*16 + c]` | `blk[256 + (m*8 + s)*2 + {0,1}]`（2 byte LE） |
| 单 sub-block mask | bit s of 16 byte（strided·**非 vlm**）| 16-bit 连续（**vlm_v_b16·lane c=bit c**）|
| 尺寸 | 512 byte | **512 byte（恒等·纯 permutation）** |

packer 改（`kqr_repack_q5_K`·verbatim → bit-transpose）：
```c
for(int m=0;m<32;++m) for(int s=0;s<8;++s){
    uint16_t mask=0; for(int c=0;c<16;++c) mask |= ((qh_col[c][m]>>s)&1u)<<c;
    blk[256+(m*8+s)*2+0]=mask&0xFF; blk[256+(m*8+s)*2+1]=mask>>8;   // 8-plane -> 8 masks
}
```
**VLEN 宽度无关**：VLEN256（half=16·numHalves=1）读整 2-byte mask（vlm_v_b16）；VLEN128（half=8·numHalves=2）读 low/high byte 为两 8-col half-strip（vlm_v_b8·`qhStripMaskBytes=half/8` 同 q5_0 泛化）→ **同一 2-byte LE 布局天然覆盖两 VLEN**（byte0=col0-7·byte1=col8-15）。

### 2.2 re-transpose 上的 native decode（full q5_0 杠杆）
```
loMask = vlm_v_b16(blk + 256 + (m*8 + sLoBit)*2);          // mask 白嫖·非抽取
nLo    = vadd_vx_u8mf2_mu(loMask, loNib, loNib, 16);       // masked +16 融入(K-quant·无 -16 offset)
```
`vsrl/vand/vsll/vor` 全撤 → **2 op/臂**（vs OLD 4·vs sub-block-native-KNEST 3）。

---

## 3. ★可行性 G1（byte-exact 硬门 + qh-recon 静态账 + 影响面）

### 3.1 byte-exact 硬门（含 packing round-trip·`raw/be_retranspose`·0/4096）
| 门 | 结果 |
|---|---|
| (A) qh round-trip 双射 mismatch | **0**（byte-exact permutation·**权重值不变只重排**）|
| ├ nibble 区 byte-diff（cur vs new）| **0**（nibble 完全不动·re-transpose 只碰 qh 区）|
| └ qh popcount cur / new | **2096 / 2096 EQUAL**（permutation 保 bit·无增无删）|
| (B) 5th-bit decode stock≡current≡retrans | **0/4096** |
| (C) full 5-bit weight `nibble\|(bit<<4)` | **0/4096** |
**代数**：re-transpose 是 512-bit qh 空间的双射（每 (col c, pos m, sub s) bit 唯一映到 mask(m,s) 的 bit c 再逆回）·decode 读同一 bit·`vadd_mu(+16)` 与 stock `qh&u?16:0` 恒等（K-quant 无 offset-binary·bias 在 6-bit MIN）。

### 3.2 qh-recon 削减（rolled per-(ii,h)·gcc-15.2 -O3·VLEN128 e8mf2·`raw/qh_recon_variants.s`）
| 变体 | v-insn | vsetvli | qh-recon-arith（除 shared nibble/load）| op/臂 |
|---|---|---|---|---|
| **OLD_CUR（现部署）** | 16 | 2 | 8（vsrl2+vsll2+vor2+vand2）| 4 |
| SB_KNEST（`../q5k-knest-G1` 设计·未部署）| 14 | 2 | 6（vand2+vmsne2+vadd2）| 3 |
| **RETRANS（本设计）** | **12** | 2 | **4（vlm2+vadd2）** | **2** |
- **RETRANS vs OLD：−25% v-insn / −50% qh-recon-arith**（`vsll/vor` RETIRED·qh mask **vlm 直装** 非 `vand+vmsne` 抽取·1 vle8 替 2）。
- **RETRANS vs SB_KNEST：−14% v-insn / −33%**（re-transpose 唯一净增于 KNEST = 省掉 mask 抽取的 vand+vmsne → vlm）。
- **注**：instruction-count vs **stock**（手写 block-dot）仍 **~3.76× NOT MET**（M=1 element-wise broadcast 结构限·家族一致·[CASE-MICRO-E2E]）——re-transpose **不改 perf-gate 结论**（同 q3_K deploy·−削重建 ≠ ≤1.10×）。

### 3.3 影响面评估（★repack-layout 改·两路一致）
| 消费者 | 读什么 qh | re-transpose 须改？| 风险 |
|---|---|---|---|
| `emitRepackKQuantGemvBodyQ5K`（GEVM decode·8634）| **repacked** block_q5_Kx16 | **是**（qh-recon leaf）| 中 |
| `emitRepackKQuantGemmBodyQ5K`（GEMM decode·9314）| **repacked·SAME ABI**（`:9374` "SAME weight ABI as GEVM"）| **是·两路一致强制**（同 512B qh @256·同 idiom `9942-9949`）| **高**（prefill 路·须与 GEVM 逐 bit 一致）|
| `kqr_repack_q5_K`（offline packer·`kquant_repacker.h:119`）| 写 repacked | **是**（verbatim→bit-transpose）| 中（机械·byte-exact 可证）|
| `RVVToEmitCKQuant.cpp` injectQh（flat dequant·989-1015）| **RAW on-disk** `x[i].qh`（非 repack）| **否·UNAFFECTED** | — |
| sealed cert md5 `ba30ba54`（block_q5_Kx16/2816·`kquant_repack_cert.c:167`）| repacked | **重算·re-seal** | **高·板 re-validate**（irreversibility） |

**⇒ 影响面 = HIGHER-RISK 三重**：repack-layout 改（非 decode-leaf 局部）+ GEMM∧GEVM 两路 qh 逐-bit 一致（共享 ABI）+ sealed md5 re-seal（板依赖）。对比 q3_K native-mask deploy = **decode-only·单 leaf·无 packer·无 re-seal**（`df7dae7d` 3 文件·242/242 绿）——**re-transpose 风险量级远高**。

---

## 4. ★可行性裁决

| 判据 | 结果 |
|---|---|
| (a) 恢复 vlm native-mask？ | **✅ 是**（16-bit mask lane c=bit c·full q5_0 杠杆·2 op/臂）|
| (b) byte-exact（packing round-trip + decode）？ | **✅ PASS**（双射 0·decode 0/4096·popcount 恒等·nibble 不动）|
| (c) 影响面可控？ | **⚠ 部分·HIGHER-RISK**：packer + GEMM∧GEVM 两路一致 + sealed re-seal（板）·唯一 UNAFFECTED = flat dequant（读 raw qh）|
| **结构可行？** | **✅ 结构可行 ∧ byte-exact·但影响面 = repack-layout 三重风险（非局部 decode swap）** |

---

## 5. ★值不值得（informs·judgment）

**收益（C1 组合式 extensibility 完整闭合）**：native-mask 泛化到全 4 qh 格（q5_0/q5_1/q3_K + q5_K）·q5_K 得 **full uniform vlm 杠杆**（2 op/臂·与 3 vlm 格齐）。
**成本/风险**：repack-layout 改（packer）+ GEMM∧GEVM 两路逐-bit 一致（共享 ABI·较单 leaf 高）+ sealed md5 `ba30ba54` re-seal（板 re-validate·irreversibility）。
**perf**：**零**（[CASE-MICRO-E2E]·M=1 GEVM 3.76× stock 结构限不动·qh-recon −50% 不逼近 ≤1.10×·GEMM prefill memory-bound）。

**★judgment = 不值得（现在）·durable informs**，三点：
1. **C1 组合式 extensibility 已充分证**：3/4 qh 格 vlm-native deployed（q5_0/q5_1/q3_K）+ q5_K 由 **sub-block native-mask KNEST**（`../q5k-knest-G1` 已 byte-exact 证）覆盖 → **泛化已足证·非需 4/4 uniform vlm**。
2. **"native-mask 到 q5_K" 有【廉价路】**：sub-block native-mask KNEST = **decode-only swap**（vand(1<<s)+vmsne+vadd_mu·3 op/臂·同 q3_K deploy 低风险 profile·**无 packer·无两路·无 re-seal**）——这才是把 native-mask 元件带到 q5_K 的正确低风险工具·**re-transpose 非之**。
3. **re-transpose 唯一净增 = uniform vlm（3→2 op/臂）**·是 **perf-零 refinement**·不抵 repack-layout 三重风险。

**诚实结论**：re-transpose **结构可行 + byte-exact-provable**·但 **不值得付 repack-layout 改风险 for C1-completeness-only**——q5_K 用非-vlm qh 解码（现 OLD·或廉价 sub-block-native-KNEST）**已 byte-exact 工作**·组合式 extensibility **3/4 vlm + q5_K-covered 已足证泛化**。设计 + byte-exact 模型就绪（本 casefile）·若主会话未来以别的名义（如 uniform-emitter-maturity）立项可直取；本线以 **有效 informs = 不值得/higher-risk** 收口。

---

## 6. ★具名 + 诚实定性
- **[GAP-EMIT-KNEST-QH-SUBBLOCK-PACK] 评估 = CLOSED-as-informs**：结构可行 ∧ byte-exact ∧ **判定不值得**（repack-layout 三重风险 vs perf-零 C1-completeness·且廉价 sub-block-KNEST 路已可达"native-mask 到 q5_K"）。**非 blocker-未解·是 informed-decline**。
- **诚实定性**：**C1 组合式 extensibility 完整性评估·非 perf**（[CASE-MICRO-E2E]）。**不值得/higher-risk 与"可行"同为有效 informs**·本线以诚实至上收口——负判读（不值得）是本线的正结果。
- **structural-gap 图景**：这是最后一个 named 结构 gap 的评估·**完成 structural-gap 图景**（re-transpose 结构可行但 informed-decline·q5_K qh 以 sub-block-pack + 非-vlm decode 定格·byte-exact 工作）。

**禁上板 · 禁 git · 无 schema label 改动 · tracked 源码只读未改（hand-construct 全在 `raw/`）。**

---
### 复现
```
GCC=/home/kingdom/spacemit-ime/spacemit-toolchain-linux-glibc-x86_64-v1.2.4/bin/riscv64-unknown-linux-gnu-gcc
cd raw
gcc -O2 byteexact_retranspose.c -o be_retranspose && ./be_retranspose        # (A)(B)(C) byte-exact 硬门 0-mismatch
$GCC -march=rv64gcv_zvfh -mabi=lp64d -O3 -S qh_recon_variants.c -o qh_recon_variants.s   # OLD/SB_KNEST/RETRANS 三变体
# 计数：grep -cE '^\s+v[a-z]' (v-insn) · '^\s+vset'(vsetvli) · vsll/vor/vand/vmsne/vadd/vlm 分类
```
