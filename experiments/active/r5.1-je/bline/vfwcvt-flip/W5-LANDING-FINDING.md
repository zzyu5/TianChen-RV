# W5 B线 landing FINDING — q4_K dequant 0.55→1.01 落地经 owned emit（更正归因）

HEAD-base `848d61f31` · emitter `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp::emitDequantizeRowQ45KVectorBody` · 板 `rvv`(VLEN128·64c) · 主会话独立 check（PRD「headline flip 主会话独立 check 再入账·正负对称」）。

## 结论（vs 部署 ≥0.8 = 入账·门 0.8）
| 格 | 前 | 后 | byte-exact | 判 |
|---|---|---|---|---|
| **q4_K** dequant@rvv | 0.5528/0.5505 | **1.0055/1.0156** | mism=0/262144·3-arm 咬·CORPUS 32/32 | ★HEADLINE FLIP·vs-部署 PASS |
| **q5_K** dequant@rvv | 1.1610/1.1718 | **2.1931/2.1941** | mism=0/262144·3-arm 咬·CORPUS 32/32+fifthbit 2/2 | 改善·shared fn·无回归 |

新 kernel md5：q4_K=`a4e23387e648362bbf329bcbecd8842e`(316行·8 distinct)·q5_K=`f01f6ad6d3462560719cde6003fad44f`(384行·10 distinct)。weft-opt=`f9a7b1670b1bfae098df502b672c68f3`。

## ★归因更正（W5 首波 headline 的错误 + 主会话独立 check 抓到）
W5 首波 EMITTER-PATCH.txt 把 flip 归于 **vfwcvt widen-shrink**，用一个**手写 8-pipeline kernel**（md5 1f5698c3·gen_q4k_vfwcvt.py 文本生成·**非 emitter 产出**）证 1.02。**独立 check 否决了这个归因**：

**在 emitter 真产出上板测三点 bisect**（非手写 kernel）：
- OLD emitter（vfcvt·标量交错）：**0.55**
- vfwcvt swap only（标量仍交错）：**0.58**（widen = 次要·+0.03）
- vfwcvt + **two-pass 标量前置**：**1.01**（★调度 = 主导·+0.43）

**手写 kernel(1.02) 恰好 scalars-first**，故「证成」——但 emitter 首 patch 产出是**标量逐组交错**，只 0.58。判别实验（同一 harness）：手写 kernel=1.02·emitter 首 patch 产出=0.58·两者都 byte-exact → **1.02 是手写 kernel 结构的产物·widen swap 单独达不到**（「部署变体≠证过变体」陷阱）。

## 真 lever（asm 铁证）
两 kernel 向量指令谱**完全相同**（24 vsetvli·8 vzext.vf2·8 vfwcvt·8 vfmv·8 vfmsac·8 vse32·4 vle8·4 vand·4 vsrl），仅调度顺序差：
- **交错(0.58)**：`vfmv.v.f/vfmsac.vf` 消费刚算出的 `fa2/fa3`（~15 条前的 fmul.s）→ in-order 板 scalar→vector 依赖停顿。
- **scalars-first(1.01)**：clang 把 16 个 scale 全前置批量 fmul.s → 8 向量流水背靠背无中断。

**WALL-TYPE 更正**：ISSUE-109/§四.2 记 q4_K「perf-cold < parity·墙」——**证伪**：不是微架构墙·是**我方 codegen 结构**（标量 prep 交错进向量区）。发射层两个合法动作（源重排 + intrinsic swap·**无 inline-asm/无手排调度**）翻盘。

## 施工（emitter diff·byte-exact 同值仅顺序变）
`emitDequantizeRowQ45KVectorBody` 单循环 `{scale prep; pipeline}×4` → **拆两趟**：pass1 算全 8 scale 存 `dLo[4]/mlLo[4]/dHi[4]/mlHi[4]`·pass2 跑全 8 pipeline。+ widen 链 `vzext.vf4+vreinterpret+vfcvt` → `vzext.vf2+vfwcvt`（nibble∈0..15<2^31·unsigned widen==signed·byte-exact）。

## 门 + 三闸
- byte-exact 先于计时：ZERO-MODEL mism=0·ggml-vs-oracle PASS·3-arm anti-hollow 全咬（D 咬到发射的 RISC-V）·CORPUS COMPLETE。
- 格式隔离：q2_K/q3_K/q6_K regen md5 == committed（未受 Q45K 改动·仅 q4_K/q5_K 变）。
- lit：Conversion/RVV 287/287 PASS·两 fixture FileCheck 更新+PASS（CORE==PROD）。
- 三闸①三 grep：0 新增裸格式字面量/板值+value_or/march 解析（纯 reorder+array+intrinsic-string swap）。②无新 θ（LMUL 由固定 32-lane 几何 derived·非旋钮）。③无字段需求单。
- 2-seed compiler-symmetric（我方 clang-18·对手同 clang-18 部署路 dequantize_row_q4_K）。

## 成色（对手与档位 §三.12·哲学锁①）
opponent = SOLE deployed `dequantize_row_q4_K` host-autovec scalar-C（codegen-lottery on OPP side·便宜档）。per 哲学锁① **vs-部署 ≥0.8 = 入账·不问对手强弱**；opp-immaturity 只作论文注记·不作「算不算赢」判据。DURABLE = de-lottery [L-8] 强义（owned emit 拥有向量化）+ 现在 owned emit **vs-部署 PASS**（首波是 named-loss）。
