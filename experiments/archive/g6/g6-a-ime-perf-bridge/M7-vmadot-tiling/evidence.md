# G6-A M7 — IME 性能桥 vmadot-tiling（array-utilization·发射器改动·令一.3 最后深潜）

> 战役 G6-A（IME correctness 桥 → performance 桥·续攻 parity/成色质变·**最后深潜**）· 板 k1·clang-18(Bianbu 18.1.8) 对称·`-fno-integrated-as`(SpacemiT binutils 认 vmadot)·governor=performance·1.6GHz 锁频·**-t 4**·loadavg_pre=3.03(post=5.89=restore rebuild·非 measure 期)·12 samples/side·relIQR gate· 生成 2026-07-13。
> 代表格 = **q4_0@ime**（同 M1-M6·banner M=32 N=2048 K=2048）。建于 M1-M6 之上（M7 = **首个发射器改动**：WIDE vmadot output-tiling·A-fragment in-register 复用跨 NJW 列 tile）。**这是 G6-A 最后深潜**（令一.3·IME 冲刺止于本里程碑）。
> **byte-exact 是硬门·锚不动摇**：M7 输出 vs M6 byte-identical（md5 **f5e77482** 全谱系·oracle/A-B identity·K1 单测 memcmp=0·real vmadot）。

## 0. 结论（★PARITY CROSSED·出口 a·成色质变候选·vmadot array-util 是最后大杠杆·全程 byte-exact）
M6 后残留三分：compute 1.741s(33.8%·vmadot array-util) + feed_nj 2.064s(40.1%·地板) + vec-epi 1.345s(26.1%)。M6 casefile 判 **vmadot compute/array-util = 唯一剩 actionable 大杠杆**（厂商 VEN/OFF=1.996× 同 K1 硅同 vmadot = 阵列有 ~2× headroom·我方窄 4×4 tile 欠 fed）。
**M7 攻 compute**：WIDE 输出 tiling — 一次 `vle8` 载入 4×8 A fragment(v0)·喂 **NJW 个独立 vmadot 链**(v2/v4/…)入 NJW 个 4×4 int32 累加器。(a) 隐藏 vmadot 阵列延迟（NJW 个 MAC 在飞·非单累加器串行）·(b) A-load + per-block vsetvli/clear/store 入口摊薄 NJW 倍。**纯 loop/复用 schedule 变·0xe210312b vmadot MAC 与 int32 累加序不动 → byte-exact 由构造**。
- **★array-util 板测确诊（compute-isolation·fixed L1 scratch·同调用数）**：vmadot COMPUTE **1.740s(w1) → 1.095s(w2·1.589×) → 0.890s(w4·1.955×)**——w4 compute 达 1.955× ≈ **厂商同硅 headroom 兑现**（阵列本可 ~2×·实测证实）。
- **matmul(full vec)**：M6 5.348s → **M7-w2 4.614s(1.159×·省 0.734s/13.7%)**；**M7-w4 5.503s(0.972×·反慢)** ← w4 compute 最快但 full-matmul NULL（16-累加器 f32 epilogue 溢出寄存器·抵消 compute 赢·**measured-negative**）。
- **★e2e pp32（12 samples/side·interleaved·同 -t 4）**：off 23.672 · ven 47.254 · onjout(M6) 20.976 · **onw2(M7-w2) 23.881** · onw4 20.903。
  - **onw2/off = 1.0088× stock → ≥ PARITY（跨过 parity 线）**·**onw2/onjout = 1.1385× vs M6**（array-util 赢 e2e 传导·1.159× matmul 天花板的 98%）·onw2/ven = 0.5054×（厂商 1.996× 天花板参照）。
  - **w2 是 shipped 宽度**；w4 measured-negative（compute 赢真·full-matmul epilogue-spill NULL·档案化）。
- **byte-exact 硬门 GREEN**：md5 onw2 == onw4 == onjout == off == **f5e77482**（M1-M6 sealed 全谱系）·全 banner routed·K1 单测 w2/w4 int32==baseline(memcmp=0·real vmadot)·host oracle int32==ZERO-MODEL。

**曲线更新**（本 run 一致口径·相对本 run off）：{… M6 0.9042× stock（sealed）→ 本 run onjout 0.8861× → **M7-w2 1.0088× stock**}。**M6→M7-w2 净 delta = 1.1385×（clean·relIQR onw2 1.40%）填平 M6 的 10% parity 缺口**。**预注册出口 (a) 触发**（≥parity）·下详 §6 成色定性（honest：parity=统计 tie·非碾压；机制赢 decisive；vs-vendor 差距仍在）。

## 1. X-0 / 反汇编·array-util 具体处（M6 profile compute 34% = 新最大残留）
M6 后 vmadot COMPUTE = 1.741s(compute_l1 探针·固定 L1 scratch·纯计算+L1 延迟)。反汇编现 leaf（`vmadot_mac_kloop`·per-block 调用·kt=fpb=4）：
```
vsetvli e8 · vmv.v.i v2/v3(clear)         # 入口
1: vle8 v0(A) · vle8 v1(B) · vmadot v2,v0,v1 · addi×2 · bnez   # 4 iter: 4 vmadot·8 load
vsetvli e32 · vse32 v2/v3(store)          # 出口
```
**array under-utilization 三处**：① **单累加器串行**：每 vmadot 写 v2·下条读 v2（accumulate）→ vmadot 延迟 L 未隐藏·阵列空转 1/L；② **A 每 vmadot 重载**（v0 无跨列复用）；③ **入口/出口 per-block**（64 blocks/tile 各付 2×vsetvli e8↔e32 toggle + 2 clear + 2 store·仅 4 vmadot 摊薄）。K1 = **VLEN256**（vlenb=32·4×8 fragment=1 m1 reg·32 vregs·probe 实测）→ 寄存器预算充裕支持加宽。厂商 1.996× 同硅同 vmadot = 阵列 ~2× headroom 直证。

## 2. 实现（发射器 WIDE 输出 tiling·[PAT-1] 正门·byte-exact 由构造）
**发射器改动**（`lib/Plugin/IME/IMEBackendEmissionDriver.cpp`·首个 G6-A 发射器改动）：
- **`macKloopHelperBodyWide(helperName, mnemonic, njw)`**：生成 WIDE leaf——A=v0(载一次)·B0..N=v1/v6/v7/v8·acc 对 v2:v3/v4:v5/v10:v11/v12:v13·K-loop 每 fragment 1×vle A + NJW×vle B + NJW×vmadot(同 v0)·出口 NJW 存储。`frag[w*16..+15]`=tile w。寄存器映射与 K1-sealed 板 mirror 逐字一致。
- **[PAT-1] 注册为数据**（`kIMEVmadotTilingPatterns[]`·schema/pattern-registry.v1.json 行形状 {pattern_id, requires/discriminant, status, metrics_hook}·**在发射器内为数据表·不触 schema/**）：
  - `IME-VMADOT-TILE-W1-baseline` njw=1 floor=3 **mechanized** hook=materialization lit
  - `IME-VMADOT-TILE-W2-Areuse` njw=2 floor=7 **mechanized** hook=wide-int32-oracle + 本 casefile
  - `IME-VMADOT-TILE-W4-Areuse-epilogue-spill` njw=4 floor=13 **measured-negative** hook=本 casefile（w4 compute 1.955× 但 full-matmul NULL·epilogue-spill 边界=设计空间资产·不可选）
- **capability-keyed selector**（`selectVmadotTilePattern`）：**discriminant = vreg budget**（RVV 架构 32 vregs·VLEN-invariant·每 4×8 fragment=1 m1 reg·NJW=w floor=1 A+w B+2w acc）→ 选最宽 `mechanized` 且 floor≤32 者 = **W2**（W4 measured-negative 不可选）。q4_0 lowering 在 module 末尾 emit 选中的 WIDE leaf（**order-neutral 到 int32/f32 seal·golden 稳定**）。
- **★byte-exact 由构造**：tile w int32 = vmadot over (A_kf · B_w,kf^T)·kf 序=baseline → == width-1 leaf 跑 NJW 次·int add 结合律无关序（order-exact）。0xe210312b vmadot 与累加序**逐字不动**。板 mirror（vendor ime.cpp patch·同 M1-M6 orchestration 手法）加 `vmadot_mac_kloop_w2/w4` + `matmul_f32_range_deref_epi_vec_njouter_w2/w4`（nj-outer·nj 步进 NJW·B-locality 保留·epilogue 折 NJW*4 列·每列 b 序不变=bit-exact；奇尾走 M6 单-tile 体）。

## 3. Correctness GREEN（硬门·byte-identical·三重证）
```
[K1 单测·real vmadot]  wide_unit.c: w2_vs_base memcmp=0 · w4_vs_base memcmp=0   [★真硅 int32==baseline]
[host oracle·ZERO-MODEL] q4-0-vmadot-tile-wide-int32-oracle.c: widetile mismatch=0 · zero-model mismatch=0 · ORACLE PASS
[板 e2e greedy 24-tok] onw2_vs_onjout=IDENTICAL · onw4_vs_onjout=IDENTICAL · onw2_vs_off=IDENTICAL · onw4_vs_off=IDENTICAL
   md5 onjout==onw2==onw4==off == f5e77482dbd78bb0543b9a64c1c2f29c   [同 M1-M6 sealed 全谱系]
[banner] onjout/onw2/onw4 各 routed(=1·tilew=2/4 确认)·ven/off=0
[发射器 lit] ime-q4-0-matmul-tile-materialization.mlir EMITC+REGION PASS（含 M7 wide-leaf 尾检）· q8_0/q4_K goldens PASS(未受影响·wide-leaf 仅 q4_0)
```
WIDE tiling 数值中性硬门达成（schedule 重排·每 sub-tile 累加序不变）。

## 4. ON/OFF e2e（包袱-收益·M6→M7·pp32·median·4-hart vs 厂商 4-hart）
| side | pp32 t/s | relIQR | n | vs stock |
|---|---|---|---|---|
| off (stock RVV) | 23.6720 | 2.18% | 12 | 1.0× |
| ven (vendor IME) | 47.2538 | 0.30% | 12 | 1.996× |
| onjout (桥·M6 B-feed locality) | 20.9763 | 4.20% | 12 | 0.8861× |
| **onw2 (桥·M7 WIDE-2 array-util)** | **23.8811** | 1.40% | 12 | **1.0088×** |
| onw4 (桥·M7 WIDE-4·measured-negative) | 20.9027 | 0.63% | 12 | 0.8830× |

- **M7-w2 增量 = onw2/onjout = 1.1385×**（array-util 包袱-收益·e2e·1.159× matmul 天花板 98% 传导）·**onw2/off = 1.0088× stock = ≥parity（跨线）**。
- **honest**：onw2 IQR[23.62,23.95] 与 off IQR[23.54,24.05] 重叠 → onw2 ≈ off = **统计 tie / parity**（非碾压 beat）。但 perf 宪法「对手贴墙 parity=满分」+ perf-covered=≥parity → **≥parity 门达成**；且 **M6→M7 净 1.1385× clean（relIQR 1.40%）= 机制赢 decisive·填平 M6 10% 缺口**。
- w4 = measured-negative（compute 1.955× 真·full-matmul 0.972× NULL·16-累加器 f32 epilogue-spill）。
- 本 run onjout 0.886×（M6 sealed 0.9042×·环境 loadavg 略高·run-to-run）；**内 run 口径**（onw2/onjout·onw2/off 同条件 interleaved）为准。

## 5. A-tree restore（md5 双证 clean）
板 EXIT-trap restore（run.log）：`RESTORE md5 ZERO-CHANGE OK (ime=40962c7e… so=71cc4d29…) · src_route_left=0 · litter_left=0 · ALL_DONE_q40_m7`。独立复核（第二证·主会话侧 ssh）：`ime.cpp md5=40962c7e…==baseline · libggml-cpu.so md5=71cc4d29…==baseline · .ORIG litter=0 · stray procs=0`。vendor 树 byte-exact 回基线·lib/ 未改（板侧）。

## 6. 曲线 + 预注册出口判读（照判 → 出口 a·成色质变·honest）+ 成色定性
**判据落地 = 预注册出口 (a)**：onw2 **1.0088× stock ≥ parity（跨线）** → **IME q4_0@ime 转绿 成色质变候选（硬碰 stock ≥parity·byte-exact·real K1）**。成色注记：**对手=stock（≥parity·statistical tie）·另列 vs vendor 差距=0.5054×（厂商 1.996× 天花板·未追平·不隐瞒）**。
- **perf-covered 计数（honest·per-format 板测律·禁 projection）**：**q4_0@ime = 板测-green（measured ≥parity）**。q8_0/q4_K@ime **共用同一 vmadot leaf**（wide tiling 同法适用·format-agnostic MAC）**但本 M7 仅板测 q4_0** → q8_0/q4_K = **同-leaf·projected·未各自板测** → **不自动转绿**（deployment≠proven·[q4-0-e2e]/[sealed-win-1] 纪律）。建议 coordinator：(i) 同 M7 harness 跑 q8_0/q4_K（同 array-util 杠杆·各自 seal）再计 perf-covered，或 (ii) q8_0/q4_K 暂维持黄 pending。**perf-covered 最终计数 = coordinator 裁**（registry 决定·非本 agent 单方设）。
- **[PAT-1] 行状态**：W1/W2 mechanized·W4 measured-negative（发射器内数据表已注册；schema/pattern-registry.v1.json 的 JSON 行 = coordinator 补·本 agent 不触 schema）。
- **★M7 交付定性（最后深潜·令一.3）**：array-util 板测确诊 compute 1.59×(w2)/1.955×(w4·厂商 headroom 兑现)·matmul 1.159×(w2)·**e2e onw2 1.0088× stock = ≥parity 跨线（成色质变候选·honest：parity tie + M6→M7 1.1385× clean 机制赢）**·**全程 byte-identical（md5 f5e77482·三重证：K1 单测/host oracle/e2e）·核 vmadot 逐字不动**·w4 measured-negative 档案化（epilogue-spill 边界=设计空间资产）。**IME 冲刺止于 M7**（令一.3）。发射器 [PAT-1] 注册（数据表+capability-keyed selector+materialization lit+host oracle）= C3′ 能力键控优化模式库首个 IME array-util 条目。

### 6.1 Coordinator 裁决（主会话·2026-07-13·pre-reg 出口 a 执行·权限卡⑦）
**q4_0@ime → 绿（perf-covered 7→8/83）**——照 M6 §6 预注册（stock 为 parity 参照·M6 时结果未知=真预注册）执行；移 goalpost 到"须赢 vendor"= 回门违例，不做。**强制诚实注记（措辞门·随绿格永驻）**：
1. **对手身份**：opponent-of-record = **stock 非-IME RVV**（≥parity·**统计 tie·IQR 重叠 [23.62,23.95]∩[23.54,24.05]·非碾压 beat**）。
2. **★vendor 更强可部署·披露不隐瞒**：vendor IME（SpacemiT 手调）= **1.996× stock·我方仅 0.505× vendor**（输一半）——k1 上 IME q4_0 的**最强实际可部署对手是 vendor·我方未追平**。
3. **★令一.2 内存税前传**（M3 deref-cache 承载·M7 走同 deref 路径）：本曲线用**预解码 int8 deref-cache**（N*K int8·代表格 q4_0 ~1.07GB·×1.78 驻留+读带宽 vs stock native nibble 0.5625·N*K）→ 逼近-parity 的 1.0088× **绑 prefill/compute-heavy 口径 + int8-deref 内存税**·decode/memory-bound regime 反向·**非零成本 free lunch**（详 M3 §2/§Residuals-3·M6 §6.内存口径披露）。
4. **★G6 成功判据（成色质变=硬碰硬赢手调对手成排）——本格【未达】**：M7 是 **metric-letter 绿（≥parity vs stock·统计 tie）+ 强 C3′ 方法学（发射器 wide-vmadot-tiling 能力键控杠杆 [PAT-1]·98% 传导）**，**非**"硬碰硬赢手调对手"（tie-stock / lose-vendor 2×）。绿之**实质价值 = C3′ 能力键控优化模式库首个 IME array-util 条目 + 传导稀释消除（桥 44×→72×→跨 parity）**，而非性能碾压。perf-covered +1 是测量轴 ≥parity 门的诚实兑现，**不得表述为"赢过手调 IME"**。
5. **q8_0/q4_K@ime 维持黄**：same-leaf projected·M7 未各自板测·deployment≠proven（[q4-0-e2e]/[sealed-win-1] 纪律）·补测入队列（同 M7 harness·各自 seal 后再计）。

## 7. 触碰文件清单（本 M7 任务·发射器改动明列·主树 A-tree 全可逆·板 restore md5 双证）
**发射器改动（源真理·未 commit·coordinator 提交）**：
- `lib/Plugin/IME/IMEBackendEmissionDriver.cpp`：+`#include <cassert>`·+`struct IMEVmadotTilingPattern`+`kIMEVmadotTilingPatterns[]`（[PAT-1] 数据表·W1/W2 mechanized·W4 measured-negative）+`selectVmadotTilePattern`（capability-keyed·vreg budget）+`macKloopHelperBodyWide`（WIDE leaf 生成器）+ q4_0 lowering module 末尾 emit 选中 WIDE leaf（golden-safe·order-neutral）。**核 `macKloopHelperBody`(baseline)/q4_0/q8_0/q4_K helper 逐字不动**。
- `test/Conversion/EmitC/ime-q4-0-matmul-tile-materialization.mlir`：+尾部 M7 wide-leaf EMITC 检（[PAT-1] provenance + W2 leaf 结构·existing 检未动）。
- `test/Target/IME/q4-0-vmadot-tile-wide-int32-oracle.c`（新增·host oracle·int32==ZERO-MODEL==baseline·ORACLE PASS）。
**板 harness（主树新增·未 commit）**：`tools/e2e-harness/board/g6-m7-ime-vmadot-tiling/{forward-route-patch-q40-tilew.py, run-m7-q40-board.sh, agg_m7.py, raw/*}` + 本 casefile + MANIFEST。
**板 A-tree（临时·测后 restore·md5 双证 clean）**：`ggml/src/ggml-cpu/spacemit/ime.cpp`(patch→build→restore) + `build-ime` .o/.so。
**未触**：`schema/`（[PAT-1] JSON 行 = coordinator 补）· `RVVToEmitCBlockQuantLinear.cpp`（L2 rvv q4_K 在用）· T8 · ROADMAP · FALSIFIER-INDEX · ODS/IMEOps.td（WIDE leaf 走 module-末尾 emit·未加 op attr·若要 shipped-default 走 selector 可后续加 OptionalAttr·flag coordinator）。
