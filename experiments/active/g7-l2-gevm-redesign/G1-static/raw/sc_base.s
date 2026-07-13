	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"sc_base.c"
	.text
	.globl	base_lift                       # -- Begin function base_lift
	.p2align	1
	.type	base_lift,@function
	.variant_cc	base_lift
base_lift:                              # @base_lift
	.cfi_startproc
# %bb.0:
	vsetvli	zero, a0, e8, mf2, ta, ma
	vand.vi	v8, v8, 15
	vand.vi	v9, v9, 3
	vsll.vi	v9, v9, 4
	vor.vv	v9, v9, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v8, v9
	ret
.Lfunc_end0:
	.size	base_lift, .Lfunc_end0-base_lift
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
