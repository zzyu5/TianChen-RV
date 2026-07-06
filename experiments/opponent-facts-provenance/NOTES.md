# opponent-facts-provenance — 结构化 opponent-fact 的 ggml 溯源 + pin 绑定

> **为什么存在.** `tcrv_rvv.quant_contraction` 的 routing 读两条**结构化 opponent-fact**(而非格式名):
> `opponent_vlen_native_floor`(i64,optional)+ `block_dot_compute_heavy`(bool,optional)。这些 fact 的
> **值** 不是凭空写的——它们是 **pinned ggml 树某几段代码的事实**。本 cell 把每个 fact 值 **绑定到 pin
> 的具体行锚**,并挂一道 CI 门:pin 变了、fact 溯源就可能失效 → 置 **STALE**。

## pin(唯一真源 = `schema/ggml-pin.lock.json`)

- tag `b9652` / sha `6eab47181cbd3532c88a105682b81b4729ab809b` / epoch 1 / 2026-06-15。
- 源文件:`ggml/src/ggml-cpu/arch/riscv/{quants.c, repack.cpp}`。

## fact ↔ ggml 行锚(machine 版见 `opponent-facts.pin.json`)

| quant | fact | 值 | ggml 溯源(pinned) |
|---|---|---|---|
| q4_K | `opponent_vlen_native_floor` | **128** | `quants.c:2064` per-VLEN dispatch;`:1770` @128 **inline RVV asm**;`:1975` @256 tuned;`:1634` xtheadvector |
| q4_0 | `opponent_vlen_native_floor` | **absent** | `quants.c:222` 仅一非-VLEN-specialized body(m1) |
| q4_0 | `block_dot_compute_heavy` | **true** | `quants.c:222` nibble decode + per-block vredsum + scattered reads |
| q8_0 | `block_dot_compute_heavy` | **absent(lean)** | `quants.c:435` 已精简(无 nibble decode,m2) |

repack 对手:`repack.cpp` 16x1 `gemv:260` / `gemm:983`(hand_tuned_repack = Win-B 对手)。

> **事实更正锁(见 `RVVContractionPathSelection.cpp` roster):** q4_K@128 **不是** scalar_fallback,而是
> roster 里**最强**对手(literal inline RVV asm `:1770`)。roster 无一格是 scalar/generic——每格都面对
> hand-tuned 对手,故任何 repack-vs-block-dot 赢都是 Win-B 候选、不是 product-gap。

## 消费链

- 声明层:`RVVOps.td` `GgmlQuantContractionOp` 携 `opponent_vlen_native_floor` / `block_dot_compute_heavy`
  为结构 attr。
- 选择层:`RVVLowerQuantContraction.cpp` 从 IR 读这两 attr → 填 `ContractionOpponentFacts` →
  `selectContractionAlgorithm`(`RVVContractionPathSelection.cpp`)。**C++ 从不 switch 格式名**;roster 注释
  是这些 fact 值的经验依据(READ-ONLY 逐格核对 pin)。

## CI 门:`tools/lint/check_opponent_facts_pin.sh`

- 比对 `opponent-facts.pin.json` 的 `pin_sha` vs `schema/ggml-pin.lock.json` 的 `sha`。
- **相等 → GREEN**(溯源仍对准当前 pin)。
- **不等 → STALE(红)**:pin 被 bump,cited 行锚可能已漂移,fact 溯源**必须重验**(full re-verify 用
  `.trellis/scripts/fetch_ggml_pin.py` fetch tag 后重读行锚,再更新本 cell 的 `pin_sha` + 行号)。
- 与 `ggml-pin.lock.json` 的宪法一致:换 upstream 树 = 新 epoch + 全 stale;本门是该宪法在
  opponent-fact 溯源上的落地子门。

self-test:`bash tools/lint/check_opponent_facts_pin.sh --self-test`(注入假 sha → 必须红)。
