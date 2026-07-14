	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zvbb1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvkb1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"sc_zvbb.c"
	.text
	.globl	zvbb_lift                       # -- Begin function zvbb_lift
	.p2align	1
	.type	zvbb_lift,@function
	.variant_cc	zvbb_lift
zvbb_lift:                              # @zvbb_lift
	.cfi_startproc
# %bb.0:
	vsetvli	zero, a0, e8, mf2, ta, ma
	vand.vi	v8, v8, 15
	vand.vi	v9, v9, 3
	vwsll.vi	v10, v9, 4
	vwsll.vi	v9, v8, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vor.vv	v8, v10, v9
	ret
.Lfunc_end0:
	.size	zvbb_lift, .Lfunc_end0-zvbb_lift
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
