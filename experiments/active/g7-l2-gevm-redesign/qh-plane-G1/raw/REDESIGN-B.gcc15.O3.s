	.file	"q5_0_gevm_redesignB_full.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.align	1
	.globl	weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0
	.type	weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0, @function
weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0:
.LFB0:
	.cfi_startproc
	srli	a4,a2,4
	beq	a4,zero,.L16
	srli	t6,a0,5
	vsetivli	zero,8,e32,m2,ta,ma
	li	t0,352
	vmv.v.i	v18,0
	mul	t0,t6,t0
	vsetvli	zero,zero,e16,m1,ta,ma
	slli	a4,a4,6
	vmv.v.i	v12,0
	mv	t4,a1
	vsetvli	zero,zero,e8,mf2,ta,mu
	add	t5,a1,a4
	addi	a6,a3,288
.L6:
	vmv2r.v	v16,v18
	add	t3,t0,a6
	vmv2r.v	v14,v18
	beq	t6,zero,.L3
	mv	a7,a5
	add	t3,t0,a6
.L5:
	vmv1r.v	v8,v12
	addi	t1,a6,-288
	vmv1r.v	v9,v12
	addi	a3,a6,-256
	vmv1r.v	v10,v12
	addi	a2,a7,2
	vmv1r.v	v11,v12
	mv	a4,a6
.L4:
	addi	a1,a3,8
	vle8.v	v1,0(a3)
	vle8.v	v2,0(a1)
	addi	a1,a4,1
	vlm.v	v0,0(a1)
	addi	a0,a4,33
	vlm.v	v7,0(a0)
	addi	a1,a4,32
	vlm.v	v6,0(a4)
	lb	a0,0(a2)
	vlm.v	v5,0(a1)
	lb	a1,16(a2)
	vand.vi	v3,v1,15
	addi	a3,a3,16
	vand.vi	v4,v2,15
	addi	a2,a2,1
	vmnot.m	v0,v0
	addi	a4,a4,2
	vsrl.vi	v2,v2,4
	vadd.vi	v4,v4,-16,v0.t
	vmnot.m	v0,v7
	vsrl.vi	v1,v1,4
	vadd.vi	v2,v2,-16,v0.t
	vmnot.m	v0,v6
	vwmacc.vx	v9,a0,v4
	vadd.vi	v3,v3,-16,v0.t
	vmnot.m	v0,v5
	vwmacc.vx	v8,a1,v2
	vadd.vi	v1,v1,-16,v0.t
	vwmacc.vx	v11,a0,v3
	vwmacc.vx	v10,a1,v1
	bne	a6,a3,.L4
	vsetvli	zero,zero,e16,m1,ta,ma
	addi	a4,a6,-272
	vle16.v	v6,0(t1)
	flh	fa5,0(a7)
	vle16.v	v1,0(a4)
	addi	a6,a6,352
	vwadd.vv	v4,v11,v10
	addi	a7,a7,34
	vwadd.vv	v2,v9,v8
	vsetvli	zero,zero,e32,m2,ta,ma
	vfcvt.f.x.v	v4,v4
	vsetvli	zero,zero,e16,m1,ta,ma
	vfwmul.vf	v8,v6,fa5
	vfwmul.vf	v6,v1,fa5
	vsetvli	zero,zero,e32,m2,ta,ma
	vfcvt.f.x.v	v2,v2
	vfmacc.vv	v16,v4,v8
	vfmacc.vv	v14,v2,v6
	beq	t3,a6,.L21
	vsetvli	zero,zero,e8,mf2,ta,mu
	j	.L5
.L21:
	vsetvli	zero,zero,e8,mf2,ta,mu
.L3:
	vse32.v	v16,0(t4)
	addi	a4,t4,32
	addi	t4,t4,64
	mv	a6,t3
	vse32.v	v14,0(a4)
	bne	t5,t4,.L6
.L16:
	ret
	.cfi_endproc
.LFE0:
	.size	weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0, .-weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
