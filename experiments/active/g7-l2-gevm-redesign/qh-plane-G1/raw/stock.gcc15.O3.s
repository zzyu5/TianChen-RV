	.file	"stock_q5_0.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.align	1
	.globl	ggml_vec_dot_q5_0_q8_0
	.type	ggml_vec_dot_q5_0_q8_0, @function
ggml_vec_dot_q5_0_q8_0:
.LFB1:
	.cfi_startproc
	li	a4,31
	ble	a0,a4,.L7
	li	a4,32
	vsetvli	zero,a4,e32,m1,ta,ma
	csrr	a6,vlenb
	li	a2,16
	vmv.v.i	v12,0
	fmv.s.x	fa3,zero
	sraiw	a0,a0,5
	addi	a3,a3,6
	addi	a5,a5,2
	beq	a6,a2,.L3
	li	a2,0
	vsetivli	zero,16,e8,m1,ta,ma
.L4:
	vle8.v	v1,0(a3)
	addi	a6,a3,-4
	vsetvli	zero,a4,e8,m2,ta,ma
	flh	fa5,-2(a5)
	vlm.v	v0,0(a6)
	flh	fa4,-6(a3)
	vle8.v	v8,0(a5)
	addiw	a2,a2,1
	vsetivli	zero,16,e8,m1,ta,ma
	fcvt.s.h	fa5,fa5
	fcvt.s.h	fa4,fa4
	vand.vi	v4,v1,15
	addi	a5,a5,34
	vsrl.vi	v1,v1,4
	addi	a3,a3,22
	vsetvli	zero,a4,e8,m2,ta,ma
	fmul.s	fa5,fa5,fa4
	vmnot.m	v0,v0
	vsetvli	zero,a4,e8,m1,ta,ma
	vslideup.vi	v4,v1,16
	vmv1r.v	v2,v4
	vsetvli	zero,a4,e8,m2,ta,mu
	vmv2r.v	v10,v2
	vadd.vi	v10,v2,-16,v0.t
	vwmul.vv	v4,v10,v8
	vsetvli	zero,zero,e16,m4,ta,ma
	vwredsum.vs	v4,v4,v12
	vsetivli	zero,16,e32,m4,ta,ma
	vmv.x.s	a6,v4
	fcvt.s.w	fa4,a6
	fmadd.s	fa3,fa4,fa5,fa3
	bgt	a0,a2,.L4
	fsw	fa3,0(a1)
	ret
.L3:
	li	a2,0
	vsetivli	zero,16,e8,m1,ta,ma
.L6:
	vle8.v	v1,0(a3)
	addi	a6,a3,-4
	vsetvli	zero,a4,e8,m2,ta,ma
	flh	fa5,-2(a5)
	vlm.v	v0,0(a6)
	flh	fa4,-6(a3)
	vle8.v	v2,0(a5)
	addiw	a2,a2,1
	vsetivli	zero,16,e8,m1,ta,ma
	fcvt.s.h	fa5,fa5
	fcvt.s.h	fa4,fa4
	vand.vi	v4,v1,15
	addi	a5,a5,34
	vsetvli	zero,a4,e8,m2,ta,ma
	addi	a3,a3,22
	vmnot.m	v0,v0
	fmul.s	fa5,fa5,fa4
	vsetivli	zero,16,e8,m1,ta,ma
	vsrl.vi	v5,v1,4
	vsetvli	zero,a4,e8,m2,ta,mu
	vmv2r.v	v6,v4
	vadd.vi	v6,v4,-16,v0.t
	vwmul.vv	v8,v6,v2
	vsetvli	zero,zero,e16,m4,ta,ma
	vwredsum.vs	v8,v8,v12
	vsetivli	zero,16,e32,m4,ta,ma
	vmv.x.s	a6,v8
	fcvt.s.w	fa4,a6
	fmadd.s	fa3,fa5,fa4,fa3
	bgt	a0,a2,.L6
	fsw	fa3,0(a1)
	ret
.L7:
	fmv.s.x	fa3,zero
	fsw	fa3,0(a1)
	ret
	.cfi_endproc
.LFE1:
	.size	ggml_vec_dot_q5_0_q8_0, .-ggml_vec_dot_q5_0_q8_0
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
