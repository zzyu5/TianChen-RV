# [RENAME+schema] 变更表准备(独占会话铺路)

parent: `07-11-G5-wiring`（G5 接线战役）· status: **in_progress** · agent `a921c8f4` 在飞 · 域: **文档**。
（slug 用 ASCII `RENAME-schema-changetable` = 变更表；避免 Chinese-in-slug 路径风险。）

三线并行之线③（用户 2026-07-12 裁五）。本任务 = **准备变更表 / 铺路**（文档域·可逆），为后续机械替换独占会话提供 旧→新 映射与影响面清单。**实际机械替换排在 K-quant 收口后独占执行**（见占位任务 `RENAME-schema-exclusive-session`·planning·gated）。

## 变更表三类
1. **pattern 编号 / registry**：pattern 命名/编号方案 旧→新 + registry 条目影响面。
2. **方言前缀**：MLIR 方言 op 前缀 旧→新（tcrv/rvv 等）+ 全 .td/.cpp.inc/lit 影响面。
3. **项目改名**：项目级 命名 旧→新 + docs/canon/spec 影响面。

## 交付（本任务·准备段）
- 三类各出 旧→新 映射表 + 影响面文件清单（供独占会话机械替换消费）。
- MOVES 映射草案（byte-exact 零漂移预期·CI 全绿门预注册）。

## 排期
- **准备（本任务·in_progress）**：变更表铺路，可逆·并行安全。
- **执行（gated·独占会话）**：排在 **K-quant 收口后**独占执行（`RENAME-schema-exclusive-session`）——机械替换需独占会话防交织（并行线需文件集不相交·此改动天然跨全树）。
