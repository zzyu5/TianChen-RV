	.file	"q5_0_gevm_base.c"
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
	vsetivli	zero,8,e16,m1,ta,ma
	li	t0,352
	vid.v	v10
	mul	t0,t6,t0
	vsetvli	zero,zero,e32,m2,ta,ma
	vmv.v.i	v18,0
	slli	a4,a4,6
	vsetvli	zero,zero,e16,m1,ta,ma
	mv	t4,a1
	vadd.vi	v11,v10,8
	add	t5,a1,a4
	vmv.v.i	v12,0
	addi	a6,a3,288
.L6:
	vmv2r.v	v16,v18
	add	t3,t0,a6
	vmv2r.v	v14,v18
	beq	t6,zero,.L3
	mv	a7,a5
	add	t3,t0,a6
.L5:
	vmv1r.v	v6,v12
	addi	t1,a6,-288
	vmv1r.v	v7,v12
	addi	a4,a6,-256
	vmv1r.v	v8,v12
	addi	a2,a7,2
	vmv1r.v	v9,v12
	mv	a3,a6
	j	.L4
.L20:
	vsetvli	zero,zero,e16,m1,ta,ma
.L4:
	addi	a1,a3,32
	vlse16.v	v1,0(a3),zero
	vlse16.v	v2,0(a1),zero
	addi	a1,a4,8
	vle8.v	v4,0(a4)
	lb	a0,16(a2)
	vle8.v	v3,0(a1)
	lb	a1,0(a2)
	addi	a4,a4,16
	addi	a3,a3,2
	vsrl.vv	v5,v1,v11
	addi	a2,a2,1
	vsrl.vv	v13,v2,v10
	vsrl.vv	v1,v1,v10
	vsrl.vv	v2,v2,v11
	vand.vi	v13,v13,1
	vand.vi	v5,v5,1
	vand.vi	v2,v2,1
	vand.vi	v1,v1,1
	vsll.vi	v13,v13,4
	vsll.vi	v2,v2,4
	vsll.vi	v5,v5,4
	vsll.vi	v1,v1,4
	vsetvli	zero,zero,e8,mf2,ta,ma
	vsrl.vi	v21,v4,4
	vsrl.vi	v20,v3,4
	vnsrl.wi	v5,v5,0
	vnsrl.wi	v1,v1,0
	vnsrl.wi	v13,v13,0
	vnsrl.wi	v2,v2,0
	vand.vi	v4,v4,15
	vand.vi	v3,v3,15
	vor.vv	v13,v21,v13
	vor.vv	v4,v4,v1
	vor.vv	v3,v3,v5
	vor.vv	v2,v20,v2
	vadd.vi	v5,v13,-16
	vadd.vi	v3,v3,-16
	vadd.vi	v1,v2,-16
	vadd.vi	v4,v4,-16
	vwmacc.vx	v8,a0,v5
	vwmacc.vx	v6,a0,v1
	vwmacc.vx	v9,a1,v4
	vwmacc.vx	v7,a1,v3
	bne	a6,a4,.L20
	vsetvli	zero,zero,e16,m1,ta,ma
	addi	a4,a6,-272
	vle16.v	v13,0(t1)
	flh	fa5,0(a7)
	vle16.v	v1,0(a4)
	addi	a6,a6,352
	vwadd.vv	v4,v9,v8
	addi	a7,a7,34
	vwadd.vv	v2,v7,v6
	vsetvli	zero,zero,e32,m2,ta,ma
	vfcvt.f.x.v	v4,v4
	vsetvli	zero,zero,e16,m1,ta,ma
	vfwmul.vf	v8,v13,fa5
	vfwmul.vf	v6,v1,fa5
	vsetvli	zero,zero,e32,m2,ta,ma
	vfcvt.f.x.v	v2,v2
	vfmacc.vv	v16,v4,v8
	vfmacc.vv	v14,v2,v6
	beq	t3,a6,.L21
	vsetvli	zero,zero,e16,m1,ta,ma
	j	.L5
.L21:
	vsetvli	zero,zero,e16,m1,ta,ma
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
