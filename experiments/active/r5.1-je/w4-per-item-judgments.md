# W4 逐条判 · 交付（零板时·逐值判定同格计分）

HEAD = `d55f9ab4e` · census pin `d173f4c2e`(HEAD 祖先·领先 7 commit) · 零源改·零板时。判决实验用 `build/weft/bin/weft-opt`(tree clean⟹binary==HEAD·pinned-valid)。

## 逐条计数（禁打包）
- **单侧 f = 18/18 逐条判**（全 = 合法单侧+理由·0 待裁·0 已补输入）· 摆设名单 3（F4/F21b 零输入 param + F7.shape·**均诚实/文档化·0 隐藏 bug**）
- **软默认 = 7/7 逐处登记**（θ31-34 辩护闭合 · θ18/19/20 byte-exact-preserving + 具名前门-stamp 改造点）
- **零消费键 = 9/9 二选一**（删除-lean 4 · I8-证据线 canon-question→ISSUES 4 · 接上-target 1）+ 已删 3 键考古 = 删对了（grep=0 live·0 dangling）

## ① 18 单侧 f 判定（全合法单侧+理由）
10 c-only：F7(feasibility 只问 c·shape 经 F6/F9 进 θ) · F25(reduction g baked·JE2 pin) · F26(★本职解析 c·唯一权威) · F27/F28/F29/F30(march→版本/LMUL/SEW·纯 c legality) · F31(vlenBits·g weightInterleave verifier frozen==16·JE1 pin) · F33/F35(budget→rung·纯 c)。
5 g-only：F5(foldModel→shape) · F6(shape→prior) · F12(stride→col/row prior) · F19(scaleModel→板测表·JE3 FLIP-2 pin) · F24(kernelKey→registry)。
3 零输入：F4(EMPTY measured 表 placeholder·[GAP-P1] STAGE THREE·scaleModel 注释掉·摆设-param) · F21b(EMPTY rolled 表·两参数注释掉·摆设-param) · F37(常量工厂·N3 Win-C 最小 m1 rung)。

**F7 判决实验（新钉·shape→θ 真动·反摆设）**：
```
OPT=build/weft/bin/weft-opt; FIX=test/Conversion/RVV/rvv-sel1-t3-tiling-rollout-gate7.mlir
$OPT $FIX -split-input-file --weft-rvv-lower-quant-contraction=march=rv64gcv \
  | grep -oE 'weft_rvv.tiling_variant = "[^"]*"|weft_rvv.tiling_selection_reason = "[^"]*"'
```
q6_K→plain(measured) · q2_K→s6_tiled · q5_K→s6_tiled · q4_0→plain(prior) · iq4_nl→plain(prior)。**5 格式→2 变体·θ 随 shape(g) 真动⟹F7 的 (void)shape 是正确 scoping(feasibility 只问 c)·shape 语义未丢(经 F6/F9 进 θ)。**

## ② 7 软默认逐处（θ18 是 ISSUE-002 承重活口）
- **θ18**(KQuant:2918 mf2)：q4_K DECODE 前门把 integer_core_lmul **UNSTAMPED BY DESIGN**·value_or 复现 legacy byte-identical·fail-close 会破 q4_K e2e。**★ISSUE-002 承重活口**（该 mf2 路向量内容是否 host-autovec=[L-8] 存疑）·具名改造点=前门 stamp(gated 回门)·保守 MAINTAIN。
- **θ19**(KQuant:3694 mf2)：+★值域重载 finding（同 attr 槽载 LMUL{mf2,m1,m2} 与 emit-mode sentinel{fused/vwredsum/minterm-vec/mlp}·默认 mf2∉sentinel⟹四门 dormant）。
- **θ20**(KQuant:5971 m2)：iq2_xxs decode UNSTAMPED·★同 op 内 g fail-closed(num_groups 硬失败) / θ 软默认 并存(有意不对称:g=correctness→fail-closed·θ=optional perf→软默认)。
- **θ31/32/33/34**(BQL:14151/14152/14438/14455)：辩护**闭合**（保守默认 mbf=1/robust/per-block/strict + fail-closed 守卫）。
- 结论：0 处需立即 fail-closed·θ18/19/20 具名前门-stamp 改造点(gated)·θ18 优先(ISSUE-002)。

## ③ 9 零消费键二选一 + 3 键考古
- **删除-lean 4**：selected_march/selected_mabi/march:value/mabi:value（纯 redundant 镜像·== pass-option·无消费者·安全删）。
- **I8-证据线 canon-question→ISSUES 4**：clang:version/cmake:version/source_sha256/binary_sha256（I8 证据线不参与 compute·"接上"违 I8/"删除"丢 provenance·且连证据消费者都无=stamp-never-read·**canon 措辞裁决 gated·不自决**）。
- **接上-target 1**：vlenb_bytes（真板事实 VLEN 字节·×8==deriveMinimumVLEN{128,256}·零消费根因=全栈走 march-string bypass·接上=threading-convergence fork·JE1 B2 标的·替代=删除-redundant）。
- **无第三态"留着以后用"**。已删 3 键(cachelineBytes/imePresent/deriveIMEPresent·commit f4cc0b437 JE4)考古=删对了。

## ④ ISSUE-119 三栏 scoping（先出不修）
- **改动面**：RVVToEmitCForwardElementwise.cpp `:5295`+`:5332` 双 switch(~19 格 stride Q2K84/Q3K110/Q4K144/Q5K176/Q6K210) + inline 6 处(:1133/1372/1614/3743/3870/4048)·census ~40/51。注:`:421 getBlockStride()` 已接 op-attr(RVVOps.td:2665 先例)⟹backlog=switch/inline baked 面非已 wired。
- **架构选项**：①描述符转移(前门 stamp 全布局+发射体改读·套阶段2 路B·工作量≫ISSUE-118) ②monolith-fallback 维持(保守) ③混合(仅热格 q4_K/q6_K 先转)。
- **byte-exact 风险+判决实验**：转描述符 stamp 值≠现 baked→byte-broken·缓解=fail-closed。判决实验(先出未跑):q4_K stamp block_stride=144 regen-diff=0 + 故意 stamp 错值(143)须 verify-fail/byte-diff≠0(反摆设)。保守默认=先出不修(独立回门 gated)。

## ⑤ census 新 pin @HEAD d55f9ab4e（律2 腿二从"抓到 N 处"→"代码正在通过检查")
census pin d173f4c2e 是 W3-前基线；HEAD(+7 commit)已落地 5 项收敛：
- fail-closed 哨兵 ×8 矛盾→**18**(868c90940 修) · `git grep -c 'repack integer core requires an explicit integer_core_lmul' lib`=18
- march ISA-token 泄漏探测层外 L1→**0**(2580990ef) · `git grep containsRVVVectorISAHint lib/Conversion`=0
- 3 死键 live→**删净**(f4cc0b437) · grep=0 live·0 dangling
- BQL fail-closed 读=18(一致) · 9 零消费键=9(仍全零·未接未删·③) · 7 软默认=7(BQL4+KQuant3)
**剩余待整改(全登记 gated 非 silent)**：9 零消费键(③) + ISSUE-119 ~40 baked(④) + θ18/19/20 前门 stamp(②)。

## 问题三分法
- **新欠债**：①验证器侧 17 处 value_or(RVVDialectWideningOps·census "7 软默认"只 scope emitter)=判定合法(verifier 须镜像 emitter 默认才能 ZERO-MODEL)·census 口径应并记"7 emitter+17 verifier-mirror" ②ISSUE-119 inline 常量分散 6 处 census 未逐列(④已补)。
- **本来有被照亮**：F7 shape→θ 首次 pinned · F25/F31 合法 c-only 复判确认 · 4 I8 证据线键非 compute 料口(canon-question) · θ18 ISSUE-002 承重活口。
- **真阻塞(gated)**：θ18/19/20 前门 stamp(回门) · ISSUE-119 描述符转移(独立回门) · 4 I8 证据线键去留(canon 裁决·门禁自改) · vlenb_bytes 接上(threading fork) · ISSUE-002/[L-8](硬冻结)。
