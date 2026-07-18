# PRD · K线 · ISSUE-109 register-fusion q4_K vec_dot（§五.1·top perf lever·别排队）

> **权威** = 补充令二 §五.1（ISSUE-109 register-fusion·**别排队**·q4_K/q6_K vec_dot 0.166/0.202→0.8 需 4–5×）。
> **性质** = 结构级发射器攻坚（消 weight-reconstruction memory roundtrip）。**性能第一。** 承 q4_K/q6_K 环诊断（2 板测点·floor=aux8 roundtrip 本身）。

## 一、目标（消 aux8[256] scratch store→load roundtrip）

q4_K vec_dot@rvv = 具名-X 0.166（best 旋钮 m1 0.186）。**真墙（board perf 证·ISSUE-109）= weight-reconstruction memory roundtrip**：
ours 8×vse8（aux8[256] scratch STORE）+ 68/70×vle8（reload）·stall-bound（IPC 0.08·96.2% backend-idle·2.2B cache-miss）。
对手 `_vl128` = **register-resident**（0 scratch store）+ vwredsum.vs。

**lever = 结构级 register-fusion**：unpack↔compute 融合·**权重解包后直接留寄存器参与 MAC·消 aux8 scratch roundtrip**（匹配对手 register-resident dataflow）。**≠架构不可达**（emitter-maturity 结构缺口·in-principle 可达）。

## 二、方法（结构级·re-roll trap 铁律）

1. **解剖**（已有·ISSUE-109）：aux8[256] scratch roundtrip 位置（Region A 8×vse8 存·Region C 68×vle8 重载）。objdump 现状 leaf。
2. **author register-fusion body**（`RVVToEmitCKQuant.cpp` q4_K vec_dot·独占此 TU）：unpack 权重 → **留寄存器**（非 aux8 scratch）→ 直接 MAC（vwmacc/vwredsum）。消 store→load roundtrip。
3. **★re-roll trap 铁律**（三次证伪）：objdump 证 **aux8 vse8/vle8 roundtrip 真消**（8×vse8→0·68×vle8→大降·非同构 re-shape）+ perf stat 证 **backend-idle 真降·IPC 真升**（stall-bound→throughput）。**roundtrip 未真消 = 无效 = 停·具名**。
4. **前门 byte-exact**（[K-5]·oracle 已建·q4_K vec_dot）。整数路径 byte-exact 免费（Σ decode·decode int32 零舍入·补充令二 §四）。
5. **板测 cold**（`vec_dot.sh rvv verify/measure q4_K`·2-seed）+ perf stat。

## 三、验收

1. **register-fusion 真消 roundtrip**（objdump aux8 vse8/vle8 前后 + perf stat backend-idle/IPC 前后·有数）。
2. **byte-exact GREEN**（[K-5]·harness·整数零舍入）。
3. **cold 2-seed**·verdict（**先测不预告**·[§五.15]）：
   - cold≥0.8 → **q4_K vec_dot 真收口**（**真赢手调 = 硬赢·可称**·消灭 1 个具名-X·§六）。
   - cold<0.8 → 具名 + 真墙（register-fusion 已试·lever 清单缩·带 objdump/perf 新墙证据·非架构不可达则记新 lever）。
4. **报数带 ULP 界**（补充令二 §二·若浮点尾巴有税·q4_K 有 f32 scale fold）。
5. **成色**：对手=手调 `_vl128` STRONG（**非便宜档**·真赢=硬赢可称）。**0 造数**·byte-exact 硬门·sealed 不动·master 不直写·**禁 commit·禁 add -A**·objdump 全量。

## 四、触碰集 / 遗留

- 触碰：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（q4_K vec_dot register-fusion body·独占·与并行线 FMA-probe〔objdump 只读〕/iq3_xxs dequant emit/[D-2a] 不相交）+ 前门 lit + `vec_dot.sh` + bench 跑。用 rvv 板。clean rebuild+亲见。
- **遗留**：q4_K 证后 → q6_K vec_dot 同 register-fusion（同 floor·扩）+ K-quant decode 8 格（同 weight-reconstruction 面）。@k1 gated ISSUE-105。若 register-fusion 后仍 <0.8 → 记新墙 + 新 lever（禁写架构不可达除非 lever 清单空）。
