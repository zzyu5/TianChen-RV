# PRD — T2 (HIGH): θ14 CodebookFp4 coreLmul 焊死 → 真闭式 f(VLEN, codebook_size)

## 价值(21 焊死里最干净 焊死→f·census 排 Rank-2)
θ14=`RVVToEmitCCodebookFp4.cpp:109 coreLmul="m1"`(硬编码)。是 codebook gather 正确性地板(16 项码本需 VLMAX≥16)·**独特: vtypes 已 string 派生**(`"vint8"+coreLmul+"_t"`:112-116)=焊死是等着被表达的闭式: `coreLmul_min=满足 VLMAX≥codebook_size 的最小 LMUL`(VLEN128→m1/VLEN256→mf2 均 VLMAX=16)。正确性-pin→表达出的能力公式=论文精确演示。

## 做
- `:109` 字面 "m1" → 闭式 `f(VLEN,SEW,codebook_size)`=满足 VLMAX(LMUL,SEW)≥codebook_size 的最小 LMUL(复用 getRVVStripVLMAXElements/footprint 权威·别新造)。
- 保守 verifier(硬编 m1 那个·`RVVDialectControlOps.cpp` region predicate): `VLMAX≥codebook_size` 替 `==m1`。
- 判决 lit: 改 VLEN·公式出 m1@128/mf2@256(VLMAX 均 16)·改 codebook_size 随之变。
- ★byte-exact 落地: VLEN128 仍解 m1(零输出变·安全提交); VLEN256 narrowing deploy 需 measured gate·**本役先落公式+verifier(byte-exact)·VLEN256 flip 延后**(infra 先行)。

## 触碰集
`lib/Conversion/RVV/RVVToEmitCCodebookFp4.cpp`+`lib/Dialect/RVV/IR/RVVDialectControlOps.cpp`(verifier)+codebook fp4 lit。🔴 禁碰: RVVCapabilityProfile(T1)·RVVToEmitCKQuant(T3)·RVVLowerQuantContraction·grid dequant body。

## 门
- byte-exact@VLEN128(仍 m1)·lit 绿·CORE==PROD·无 inline-asm·三 grep 只减不增·判决证公式随 VLEN/codebook_size 真动。

## 交付
- **交付首节「判决实验 x/y」**+θ14 焊死→闭式 diff(焊死 21→20)+byte-exact@VLEN128 证+文件清单。0 造数·不 commit。
