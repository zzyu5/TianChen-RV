# MANIFEST — g8-stage1-clang-env (G8 阶段一.三·双板 clang-18 对称环境定稿)

- evidence.md — 双板 clang-18 环境就绪证明主文 (对表/flags/对手全量重编/样例A-B/就绪门表/IME carve-out)
- board-identity-table.md — 双板 clang-18 身份对表 (路径·march·as·gcc-runtime·-fno-integrated-as·VLEN·pin·IME carve-out)
- flags-finalized.md — kernel-axis clang-18 flags 定稿 (rvv rich march + -fno-integrated-as + --gcc-install-dir · k1 canonical · IME 域分离)
- preflight-assertions.md — 阶段三预飞门断言 A/B/C/D/E (clang-18 域·含 D1 对手两blocker解·D2 IME carve-out·E clang-17退役)
- rvv/raw/sample_ab_q4_0_clang18.txt — rvv q4_0 clang-18 对称样例 A/B raw + 双 objdump (现行·GATE=PASS ratio 2.19)
- rvv/raw/opponent_full_recompile_clang18.txt — rvv 对手 vericurve 树全量 clang-18 重编 + 符号探针 (含 vcreate-4 + inline-asm·两 blocker 解)
- rvv/raw/sample_ab_q4_0_clang17.txt — [VOID 作废] clang-17 样例·仅环境备份·不入主表
- k1/raw/sample_ab_q4k_clang18.txt — k1 q4_K clang-18 样例 A/B raw + 双 objdump (Bianbu 18.1.8·nbad=0 ratio 1.20)
