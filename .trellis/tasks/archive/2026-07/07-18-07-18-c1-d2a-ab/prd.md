# PRD · C1 · 裁决1 [D-2a] A+B 施工（装载期解析器 + 每进程一条记录）

> **权威** = 用户裁决1（2026-07-18）·补充令二 §五.3。**C1 头牌主张的另一半。** 地形 = 归档 `07-18-c1-d2a-recon` research（7 文件）。
> **性质** = 编译器/运行时侧构造（纯工·不触 ISSUE-105 部署判据级·D 单独走裁决）。

## 一、目标（[D-2a] A+B·首攻里程碑）

[C1-1] 头牌后半句「fail-closed 运行期（装载期解析形态起步）调度守卫」= 目标契约·未实现。本 task 落 **A+B**：
- **A · 装载期解析器**：消费 schema 事实实例 → **profile 先展开成规范化事实集** → 对**展开后事实集**算 `declared-instance-hash`
  （profile 写法与显式列表**同哈希**）。**复用 `lib/Support/DeclaredInstanceHash.cpp`**（hash 算法已备齐·over 展开后事实集·按 id 排序·省 symbolName·SHA256·`.h:25`「compile time once, never per dispatch」）。delta = 把消费搬到**部署二进制进程启动时**（live per-process）。
- **B · 每进程一条解析记录**（[D-4]② 第二级归因·硬门=每进程 1 条）：形状 `{declared_instance_hash, ts, resolved_variant_set}`·I4 镜像·**热路径零逐次**。

## 二、★红线守法（不可破·research §5）

- **[NG-3]**：per-dispatch 强制检查**永禁** —— 装载期算一次 hash + 每进程一条记录 + **热路径查缓存零逐次**。（现热路径已守 [NG-3]·`DispatchRuntimeGuard.cpp` 只读 caller ABI int·勿引入逐次检查。）
- **I3 零分支**：分发按 declared-instance-hash / 能力事实键控·**禁 family 名分支**（复用 `RVVRepackTilingSelection.h` shape/hash 键控范式）。
- **I7 fail-closed 自足**：缺/未知能力 → fail-closed·**不从 I4 镜像合成 route**（复用空可行集 fail-safe + static_order 诚实标注）。
- **[部署§六.13] 两层降级标注**：新增运行期物件须标「load-time/运行期=目标契约·未实现→本 task 实现」·禁与编译期 fail-closed 混写未标注。

## 三、验收

1. **A 装载期解析器**：进程启动消费 schema → 展开 → hash（复用 DeclaredInstanceHash·profile==显式列表同哈希机检）。
2. **B 每进程一条记录**：live drop 硬门（每进程 1 条·非 per-dispatch）· CI 门可检（类比 `check_f4_attribution_jsonl.py`）。
3. **[NG-3]/I3/I7 守法测试**：负控证（引入 per-dispatch 检查 → 门红 / family 名分支 → 门红 / 缺能力非 fail-closed → 门红）。
4. **不触 ISSUE-105 部署**（判据级待裁·D 单独）·**不改 GEN_SEAL/master**（保守默认现行·k1 维持具名-X 0.6474·sha256 dc876ef6 守恒）。
5. **byte-exact/lit**（无回归·[[build-incremental-unreliable]] clean rebuild）·**0 造数**·**禁 commit·禁 add -A**·"某物不存在"禁截断。

## 四、触碰集 / 遗留

- 触碰：`lib/Support/DeclaredInstanceHash.*`（复用+运行时消费）+ 新装载期解析器/记录物件（lib/Support 或 lib/Runtime）+ 守法测试 + [部署§六.13] 标注。**独占**（与并行线 register-fusion〔KQuant〕/FMA-probe/iq3_xxs dequant 不相交）。本地 build/lit（无需板）。
- **遗留**：C（运行期归因链·大·先固 [NG-3] 守法测试）= 后续 task。D（ISSUE-105 部署通道·判据级）= 单独裁决·不与本 task 耦合。C1 头牌后半句「起步」= A+B 兑现·完整契约含 C。
