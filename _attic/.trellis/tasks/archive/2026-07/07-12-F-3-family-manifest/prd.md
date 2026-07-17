# F-3 family-manifest 机检(用户裁:清单⊆检非跨根搬迁)

parent: `07-11-G5-wiring`（G5 接线战役）· status: **in_progress** · agent `a3a1173b` 在飞 · 域: **falsifier**。

三线并行之线②（用户 2026-07-12 裁五）。用户裁定收窄 F-3 = **清单 ⊆ 检**（每家族维护 manifest 白名单 + 机检其 locality），**非**跨根目录物理搬迁（避免破坏既有布局/历史）。

## 内容
1. **每家族 manifest**：为各量化家族（flat / K-quant / IQ / IME 等）建 family-manifest（声明该家族所属文件清单）。
2. **check_family_locality.py**：机检脚本——验每家族实际文件 ⊆ 其 manifest 声明（清单子集检查·非搬迁）。
3. **接 CI**：check_family_locality.py 进 CI。

## 目标：falsifier 六门全进 CI
[F-1 .. F-6] 六门齐进 CI。本任务落地 F-3；其余门（F-1/F-2/F-4/F-5/F-6）现状核对，缺口列账。

## 判据
- 清单子集检查绿：每家族实际文件 ⊆ manifest；越界即 RED。
- 明确边界：manifest = 白名单声明 + 机检，**不**触发跨根搬迁（用户裁定）。
