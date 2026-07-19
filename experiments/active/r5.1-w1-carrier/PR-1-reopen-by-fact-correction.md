# PR-1 · 状态行改写：REOPENED-BY-FACT-CORRECTION（2026-07-19·事实更正·不推翻采购否决）

> **W1 载体批 · 裁1 子项③**。PR-1 原登记（`docs/PENDING_RULINGS.md` §登记表·现档 `_attic/docs/PENDING_RULINGS.md:16`）·tracked 承接（`.trellis/spec/issues/spec树与治理.md` §连带 + `experiments/active/g8-stage3-attack/B4-xscalar-boardsource-and-TX.md:75`）。本文件是**git-可查的状态行改写**。

## PR-1 原态（截录·勿改旧册字节）

- **问题**：B4 X-SCALAR 真板采购（向量缺席板 rv64gc 无 V）：手头无合规板时是否采购？
- **原状态**：**RESOLVED**（future work · 采购否决）。
- **原 case-head premise**：「**手头【无】物理 no-V 真硅**（rvv/k1 均带 V）→ (a) 类完整证词归 future work · 在飞路径唯 (b) 窄豁免」。

## 事实更正（premise 已为假）

**premise「手头无物理 no-V 真硅」= 事实过期（FALSE）。** `scalar`（超锐）板 = **(a) 类物理 no-V 真硅**，本批四路机检 + 硬件 SIGILL 陷阱实证钉死：
- run-id `carrier-w1-scalar-20260719T092244Z`（`raw/RAW-scalar-rerun-carrier-w1-scalar-20260719T092244Z.rawlog.txt`·md5 `6af17b5f22936c8929c892f4f16634af`）。
- isa `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`（无 v/zve）· DT `rv64imafdcbh` · AT_HWCAP `V_bit21=0` · V-opcode `.word 0x0C007057` → `SIGILL`（硬件陷阱）· clang 18.1.8 · Fedora42/8-core。
- 详见同批 `scalar-noV-fourway-machinecheck-FINDING.md` + `L3L4-scalar-silicon-byteexact-FINDING.md`（拥有内核在其上 byte-exact·mism=0）。

## 新状态行

**PR-1 = REOPENED-BY-FACT-CORRECTION（2026-07-19）**——RESOLVED 的裁决部分建立在现已证伪的 premise 上，故以事实更正重开，供下轮再裁。

**重开范围（精确·避免越权）**：
1. **重开的是 premise 与其所决定的「(a) 归 future work」**——现 (a) 类物理 no-V 证词 = **在飞/已兑现**（本批已在真硅 byte-exact），不再是 future work。
2. **★不推翻采购否决**：板支出硬冻结（[决策权限卡] 采购=否决）**仍现行法**。本板是**到货**（已在手·physical no-V rv64gc）**不是采购**——无采购动作发生 ⟹ 采购否决那条硬冻结**不被本更正触及**。
3. **canon 侧只登记不执行**：铁线4（`canon/测量判据.md:108`）此前已据实订正；**词表订正**（scalar「永不作贡献基线」动分档口径）+ 任何 canon 级措辞改动 = **硬冻结·自动解锁进待裁队列·下轮裁**（本 agent 不执行·[决策权限卡]）。
4. **S 线开测仍 gated [ISSUE-061]**：物理板在手 ⇏ 家族身份已验收；重开 PR-1 **不**解锁 S 线开测。

## 该 add
- 本文件（git-可查的 PR-1 状态行改写）。
- 连带 tracked 修正：`experiments/active/g8-stage3-attack/B4-xscalar-boardsource-and-TX.md`（原 PR-1 回填行「手头无物理 no-V 真硅」加事实更正指针·见该文件 diff）。
- （主会话可选）`_attic/docs/PENDING_RULINGS.md:16` 原登记状态行 `RESOLVED` → 追记 `REOPENED-BY-FACT-CORRECTION 2026-07-19`（若要 git-可查须 `git add -f`·因 `_attic/` 属归档区·由主会话裁是否再 track 归档）。
