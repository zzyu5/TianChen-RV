# G5-M2 q6_K — L-7 反汇编钉死（二.1 · fallback-exclusion + 0.07× 指令构成）

> **APPEND-ONLY casefile**（用户裁二.1·2026-07-12）· 不改 `evidence.md` 主体。
> board `ssh rvv` openEuler VLEN128 gcc-15.2.0 · analysis-only（禁 perf-measurement·0.07× 已在案 evidence.md §七）。
> raw = `L7_raw_wrapper-nm-objdump.txt`（nm + wrapper objdump + emitted objdump 全指令统计 + hot-body window）。
> A-tree 测后 restore 验 clean：sources = baseline（deb61a29/57851439/99131cf7）· live .so = 05a62e6a（OFF-pristine·0 q6_K syms）。

## 判决（一行）
**首发身体不成熟 → provisional 转正 = 黄-对手更强 final**。forward 确走**我方 emitted vl=8 q6_K repack kernel**（GEMM prefill / GEVM decode），**generic scalar fallback 在 VLEN128 板上指令级 provably unreachable**（beq VLEN==128 恒 taken·generic 是 not-taken fall-through）。0.07× = **全展开 vectorized repack + regfile-spill**（NOT scalar·NOT table）。**非部署病**（未白嫖 stock·未走 fallback）。

---

## 一、fallback-exclusion（★二.1 最关键·用户明文要求"显式排除 forward 路由到 generic fallback"）

用户警告：**fallback 也输出正确·correctness GREEN 不能排除它**。故排除不靠 correctness（A==B 对 emitted / generic / stock 三路都成立），靠**指令级路由证 + 慢速方向证**三条独立链：

### 1a. 指令级：generic fallback 在 VLEN128 板上 UNREACHABLE（wrapper objdump）
两 wrapper（`ggml_gemm_q6_K_16x1_q8_K` 51 insns / `ggml_gemv_...` 50 insns）是**纯 dispatcher**（零 compute），结构同构：
```
csrr a7,vlenb ; sll a7,a7,3   ; a7 = VLENB*8 = VLEN(bits)
li   t1,128
beq  a7,t1, <emitted-path>     ; VLEN==128 => TAKEN（本板恒真）
j    <..._generic@plt>         ; NOT-TAKEN fall-through = scalar fallback => 本板永不执行
<emitted-path>: ... fprintf(banner) ... ; j <tcrv_emitc_...kernel@plt>   ; tail-call EMITTED
```
- 本板 `__riscv_vlenb()*8 == 128` → `beq` **恒 taken** → generic 分支（`f22c4: j ..._generic` / `f223e: j ..._generic`）是 **not-taken 死路**。
- banner（`fprintf` "TCRV G5-M2 EMITTED …ENGAGED"）**在 VLEN==128 分支内**·其两个出口（f22e2/f2336·f2258/f22b2）**全 tail-jump 到 emitted kernel**。
- ⇒ **banner-fire ⟺ emitted kernel 是被调者**（generic scalar 结构性排除·非"correctness 相同故不可分辨"）。

### 1b. 运行时：banner 真在 forward 里 fire（live routing probe）
ON 部署 → 真 llama-completion forward → **emitted GEVM banner fire**（probe#1 ×4 · probe#2 ×8·`ENGAGED n=4096 nc=128 nb=16`）·文本 coherent·无 logit NaN。evidence.md §六 已记 batched-prefill **16 GEMM banners**（llama-bench pp128）。banner 只可能来自 1a 的 emitted 分支 → forward 真走 emitted。

### 1c. 慢速方向：0.07× 本身排除 stock block-dot（attribution lock）
若 forward 路由到 **stock ggml q6_K block-dot**（dispatch 返 nullptr·白嫖上游 hand-tuned），ON 应 ≈ OFF（≈1×）。实测 ON = **0.07×（~15× 慢于 OFF-stock）**。**15× 慢只能来自我方 emitted kernel 真跑**——白嫖/旁路给不出减速。⇒ stock 排除。

**三链交叉**：generic（1a 指令级 unreachable）✗ · stock（1c 方向）✗ · **emitted repack ✓**（1a+1b+1c 同指）。**forward-routes-to = repack（我方 emitted vl=8）**。

## 二、反向控制（reverse control · 证我方 repack 是承载·非旁路）
用户要求"移除/gate-off 我方 repack → 行为变"。**correctness 输出不变**（A==B·fallback trap·不可作判据）；**routing + perf 双双翻转**：

| 变体 | nm q6_K emitted syms | banner | forward 路由 | perf |
|---|---|---|---|---|
| **ON**（scaffold present·cff0585c） | **2**（gemm+gevm T） | **fire** | emitted vl=8 repack | 0.07× |
| **OFF**（scaffold 移除·05a62e6a） | **0** | 无 | stock block-dot | ≈1×（4.14 t/s baseline） |

移除我方 repack scaffold（OFF）→ emitted 符号从二进制消失（nm=0）→ forward 回退 stock → banner 消失 → perf 回 1×。**routing/perf 随我方 scaffold 存废翻转 ⇒ 我方 emitted repack 是 0.07× 的承载·非旁观**。

## 三、0.07× 指令级构成（用户要求确认：标量逐元素？逐块表？全展开 vectorized repack？）
emitted GEMM（prefill 载体·83,858 insns·branches 12/backward 5 = **无 loop·全展开**）：
- **VECTORIZED**（非 scalar·非 table）：dot = `vwmacc.vx` 2048 + `vwmacc.vv` 256；6-bit 双平面 decode 全向量形——`vand.vi` 768（0xF/0x3 掩码）· `vsll.vi` 512（qh<<4）· `vor.vv` 512（ql|qh<<4）· `vsub.vx` 512（−32 bias）· `vsrl.vi` 640。无 table gather（无 vluxei）·无 scalar 元素循环。
- **VL=8 窄**：6,065 `e8,mf2`（VLEN128 下 8-bit mf2 = VL8）· imm=16/64 = 0（VLEN128-safe·部署==证过）。
- **★病灶 = regfile spill**：`vs1r.v` 1,980 + `vl1re16.v` 2,027（+ vs2r/vl2re32 254）= 整-向量-寄存器 spill/reload 到栈（全展开体活跃向量值 >32 架构 vreg）· 6,278 `vsetvli` 重配。

⇒ **0.07×-指令构成 = 全展开-vectorized-repack**（super-block 有符号 int8 scale 解包 + 双平面 6-bit ql|qh decode + −32 bias 全向量·但 ~84k 指令全展开·~2000 整寄存器 spill·~6300 vsetvli 重配·无调度）= **first-emission 未调度身体**。**非 scalar·非 table**。

## 四、部署五验（逐项）
1. **同树物理 .so swap**：live 在 q6ON(cff0585c)/q6OFF(05a62e6a) 间 cp 交换·同 gcc-15.2.0 树·唯一 diff = q6_K scaffold ✓
2. **nm ON≠OFF**：ON 2 emitted T syms / OFF 0 ✓
3. **banner engage**：ON forward 真 fire emitted GEVM banner（live probe ×4/×8·evidence.md 16 GEMM）✓
4. **objdump vl=8**：emitted 两符号 vsetivli imm=8 only（gemm 2995·gevm 1175）·imm=16/64 = 0 ✓
5. **对手 stock 非 SELF**：opponent = stock ggml q6_K block-dot（同树·仅 scaffold diff·上游 rvv 无 q6_K repack body·恒 nullptr）✓

## 五、归因双层 + 可修性（用户裁二.1 判决树"首发不成熟"分支）
- **first-emission 未成熟（具名 GAP [GAP-KQUANT-GCC-CODEGEN]）**：emitted vl=8 repack 全展开无调度·~84k 指令·~2000 整-vreg spill·~6300 vsetvli 重配 → gcc-15 codegen 未消 spill/未调度。
- **weight-reconstruction-bound 上限（XFER-1）**：q6_K 每权重需 super-block scale 解包 + 双平面 6-bit（ql 低半字节 | qh 2-bit 高平面）+ −32 bias·arithmetic-intensity 上限低（比 q4_K 0.42× 更狠·6-bit 双平面 decode 重）·weight-recon floor。
- **可修性 = 低**：S6 tiling（token-tile schedule）**在 q6_K 上 NULL on gcc**（首发不成熟主因是 spill·非 tiling 缺失）·且撞 weight-recon floor（物理下限·改 codegen 不移动 arithmetic-intensity 天花板）。结构造得出（C1 net-new scaffold correctness GREEN）≠ 性能立得住（集成层 [NG-4] 不以 perf 名义入台账）。

## 六、A-tree restore（板可逆·验 clean）
- sources byte-exact baseline：repack.cpp=deb61a29 · repack.h=57851439 · arch/riscv/repack.cpp=99131cf7 ✓
- live .so = 05a62e6a（OFF-pristine·0 q6_K syms）✓
- 无 q6_K .inc stray（build_seal step4 rm + restore·ARCHDIR 干净）✓
- **RESTORED = TRUE**。

## 七、结论
**判决 = 首发身体不成熟 → 归因双层 → provisional 转正 = 黄-对手更强 final**。
**非部署病**（forward 确走我方 emitted repack·未白嫖 stock·generic fallback 指令级 unreachable）。T8 ledger 行 #182 provisional 暂标可转正（本查 L-7 满足·配合 L-11 同域标注[已标]）——转正动作由主会话执行（禁 git·append-only 已交付本 casefile）。
