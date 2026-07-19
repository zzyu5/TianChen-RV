# PRD — W3 36 dequant 格核真向量运算指令数（控制 vs 抽签）

## 触碰集（W3 专属·零板时·READ-ONLY 分析）
只读 `experiments/active/r-dequant/kernels/*.c`（18 格）+ objdump 板端 leaf.o（若已有 raw·否则本地交叉编译到 .o 反汇编·**不改任何源**）。写：仅本 task research/ 目录出账表。**禁碰**任何 lib/ 源。

## 病（补充令裁3：36 dequant"已收"不认账·标 pending-核）
上波报"非-grid dequant 全 PASS·收割 COMPLETE"——**账未核**。逐格出**真向量运算指令数**判"我方发射控制"还是"codegen 抽签"。

## 做（逐格·36 板格 = 18 格式 × 2 板·或现有单板先核）
- 每格数**真向量运算指令**（vsetvl 配置指令与死值**不计**）：
  - **2 条（全是设置向量长度/vle 设长）= codegen 抽签**（host autovec·我方没控制真运算）。
  - **>2 条真运算（vand/vsrl/vzext/vfwcvt/vfmul/vfmacc/vwmacc/vluxei 等）= 我方发射控制**。
- **连提取命令**（机算断言进交付必须带提取代码·如 `objdump -d leaf.o | grep -cE '^\s+v(and|srl|zext|fwcvt|...)'`·排除 vset*）。
- 抽签碰巧 ≥0.8 的格标 **"抽签-赢"**（按门 0.8 入账·但成色注明**非发射控制·喂不了 C1/C2 锚**）。
- 我方发射控制的格标 **"控制"**（真 owned emit·可喂 C1/C2）。

## 交付
- **交付首节自带「本流 x 格控制 / y 格抽签-赢」** + 逐格真向量指令数表（格式 · 板 · 指令数 · 判定 · 提取命令）。
- 若大批是抽签 → 结论"dequant 真向量发射器(PR-31)本波该真做把抽签格变我方控制"（供 W1/主会话）。
- 账面：0 造数 · 机算断言带提取代码 · "某物不存在"禁 head/窗口截断（用精确 grep -c）。
