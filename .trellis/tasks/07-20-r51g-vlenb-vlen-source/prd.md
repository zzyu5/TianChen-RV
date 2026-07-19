# PRD — T1 (TOP): vlenb_bytes 真板 VLEN 事实 → 喂 minimum_vlen 源（能力驱动 made literal）

## 授权 + 价值（用户裁「接上·全同意」·census 刷新排 Rank-1）
census 证：VLEN 是最承重能力输入·却还猜 march 不吃板。`rvv.vlenb_bytes`(真板 VLEN·producer `RVVCapabilityProfile.cpp:719-723`)**零读者**；8-消费者 VLEN 管道(strip_width/tiling/accumulator-LMUL/gather-VLMAX·经 readRVVProviderMinimumVLEN)吃 `deriveMinimumVLEN(-march)` 字符串解析(:415)。W2/W3b 铺好了管子·**源头还 march 猜**。接上 vlenb=真板 VLEN 成为已铺管道承重输入=「吃能力事实非吃 march 串」字面实现(论文核心)。

## 做
- `RVVCapabilityProfile.cpp materializeRVVProviderCapabilityAxes`(~:403-454·stamp minimum_vlen :415): **优先 `facts.vlenbBytes × 8`**(真板 VLEN·当 probed fact 存在)·march 派生降 fallback(vlenbBytes==0)。keep no-clobber(hand-authored 判决 fixture 赢)。
- 判决实验(committed lit·伪造不了): 能力实例 **vlenb=16(→VLEN128) vs vlenb=32(→VLEN256)·march 不变(vlenb⊥march)**·看 θ(strip_width θ2/family θ5/tiling θ6/loop-order θ7)跟 vlenb 走=真通·跟 march=旁路活。

## 触碰集
`lib/Plugin/RVV/RVVCapabilityProfile.{h,cpp}` + capability lit。🔴 禁碰: RVVLowerQuantContraction·RVVToEmitCGridCodebook/CodebookFp4/KQuant(T2/T3)·SourceFrontDoor 体。

## 门
- byte-exact(正常 vlenb×8==deriveMinimumVLEN(march)同值·除判决 conflicting)·lit 绿·三 grep 只减不增。无 inline-asm·judgment 自证。

## 交付
- **交付首节「判决实验 x/y」**(vlenb⊥march→θ 跟 vlenb)+vlenb 接上证据(:719 零读者→接上)+三 grep Δ+文件清单。0 造数·不 commit。
