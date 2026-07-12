# [RENAME+schema] 机械替换独占会话

parent: `07-11-G5-wiring`（G5 接线战役）· status: **planning** · **gated on K-quant 收口 + 变更表** · 域: **定义层**。

用户 2026-07-12 批准。本任务 = **执行段**：消费变更表（`07-12-RENAME-schema-changetable`·in_progress）铺好的 旧→新 映射，做全树机械替换。**独占会话执行**——改动天然跨全树（pattern 编号/registry + 方言前缀 + 项目改名），并行线需文件集不相交，此改动与一切并行线相交，故必须独占防交织。

## Gate（双重）
1. **K-quant 收口**：接线主战场（q6_K 及后续 K-quant）先收口。
2. **变更表就绪**：`07-12-RENAME-schema-changetable` 交付三类 旧→新 映射 + 影响面清单 + MOVES 草案。

## 执行门（预注册）
- **byte-exact 零漂移**：机械替换不得改任何可测行为（ODS/emit/数值全等·对照 build-incremental-unreliable 用 forced/clean rebuild + BEFORE/AFTER-EQUALITY 指纹）。
- **CI 全绿**：全 lint / falsifier 六门 / lit 绿。
- **MOVES 映射**：完整 旧→新 MOVES 映射登记（可审计·可回滚）。

## 判据
- 三门齐（byte-exact 零漂移 ∧ CI 全绿 ∧ MOVES 映射完整）方可结项。
- 独占会话：执行期间不起并行线（防交织·此改动跨全树）。
