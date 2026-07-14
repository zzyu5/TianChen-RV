	.file	"stock_q6_K_vl128.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q6_K_vl128.c"
	.align	3
.LC1:
	.string	"n % QK_K == 0"
	.align	3
.LC2:
	.string	"nrc == 1"
	.text
	.align	1
	.globl	ggml_vec_dot_q6_K_q8_K_vl128
	.type	ggml_vec_dot_q6_K_q8_K_vl128, @function
ggml_vec_dot_q6_K_q8_K_vl128:
.LFB1:
	.cfi_startproc
	addi	sp,sp,-96
	.cfi_def_cfa_offset 96
	sd	s5,40(sp)
	sd	ra,88(sp)
	.cfi_offset 21, -56
	.cfi_offset 1, -8
	andi	s5,a0,0xff
	bne	s5,zero,.L10
	li	a4,1
	bne	a7,a4,.L11
	sd	s10,0(sp)
	li	a4,255
	.cfi_offset 26, -96
	mv	s10,a1
	ble	a0,a4,.L6
	fmv.s.x	fa5,zero
	sd	s0,80(sp)
	sd	s1,72(sp)
	sd	s2,64(sp)
	sd	s3,56(sp)
	sd	s4,48(sp)
	sd	s6,32(sp)
	sd	s7,24(sp)
	sd	s8,16(sp)
	sd	s9,8(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	.cfi_offset 25, -88
	sraiw	s6,a0,8
	addi	s4,a3,418
	addi	s3,a5,4
	li	s2,32
	li	s1,64
	li	s0,128
	li	a1,48
.L5:
	lhu	a5,-210(s4)
	flw	fa3,-4(s3)
	addi	a2,s4,-418
	fcvt.s.wu	fa4,a5
	addi	a0,s4,-226
	addi	s7,s4,-290
	fmul.s	fa4,fa4,fa3
#APP
# 46 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q6_K_vl128.c" 1
	addi s8, a2, 32
	ld t0, 0(a0)
	addi a0, a0, 8
	slli t6, t0, 1 * 8
	lb zero, 0(a2)
	slli t5, t0, 2 * 8
	slli t4, t0, 3 * 8
	lb zero, 0(s8)
	slli t3, t0, 4 * 8
	slli t2, t0, 5 * 8
	lb zero, 0(s7)
	lb zero, 31(s8)
	slli t1, t0, 6 * 8
	srai a7, t0, 56
	vsetvli zero, s2, e8, m2
	vle8.v v8, (a2)
	srai t6, t6, 56
	srai t5, t5, 56
	srai t4, t4, 56
	srai t3, t3, 56
	vle8.v v10, (s8)
	addi a2, a2, 64
	slli t0, t0, 7 * 8
	srai t2, t2, 56
	srai t1, t1, 56
	srai t0, t0, 56
	vle8.v v4, (s7)
	vsrl.vi v12, v8, 4
	vsrl.vi v14, v10, 4
	lb zero, 0(s3)
	vand.vi v8, v8, 0xF
	vand.vi v10, v10, 0xF
	lb zero, 32(s3)
	vsll.vi v0, v4, 4
	vsll.vi v2, v4, 2
	lb zero, 64(s3)
	vsrl.vi v6, v4, 2
	vand.vx v0, v0, a1
	lb zero, 96(s3)
	vand.vx v2, v2, a1
	vand.vx v4, v4, a1
	vand.vx v6, v6, a1
	vor.vv v8, v8, v0
	lb zero, 127(s3)
	vor.vv v10, v10, v2
	vor.vv v12, v12, v4
	vor.vv v14, v14, v6
	vsetvli zero, s0, e8, m8
	vle8.v v0, (s3)
	vsub.vx v8, v8, s2
	vsetvli zero, s1, e8, m4
	vwmul.vv v16, v0, v8
	vwmul.vv v24, v4, v12
	vsetivli zero, 16, e16, m2
	vmv.v.x v0, zero
	vwredsum.vs v10, v16, v0
	vwredsum.vs v9, v18, v0
	vwredsum.vs v8, v20, v0
	vwredsum.vs v7, v22, v0
	vwredsum.vs v11, v24, v0
	vwredsum.vs v12, v26, v0
	vwredsum.vs v13, v28, v0
	vwredsum.vs v14, v30, v0
	vsetivli zero, 4, e32, m1
	vmul.vx v0, v10, t0
	vmul.vx v1, v9, t1
	vmacc.vx v0, t2, v8
	vmacc.vx v1, t3, v7
	vmacc.vx v0, t4, v11
	vmacc.vx v1, t5, v12
	vmacc.vx v0, t6, v13
	vmacc.vx v1, a7, v14
	vadd.vv v0, v0, v1
	vfcvt.f.x.v v0, v0
	vfmv.f.s fa3, v0
	fmadd.s fa5, fa4, fa3, fa5
# 0 "" 2
#NO_APP
	addi	s7,s4,-258
	addi	s8,s3,128
#APP
# 46 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q6_K_vl128.c" 1
	addi s9, a2, 32
	ld t0, 0(a0)
	addi a0, a0, 8
	slli t6, t0, 1 * 8
	lb zero, 0(a2)
	slli t5, t0, 2 * 8
	slli t4, t0, 3 * 8
	lb zero, 0(s9)
	slli t3, t0, 4 * 8
	slli t2, t0, 5 * 8
	lb zero, 0(s7)
	lb zero, 31(s9)
	slli t1, t0, 6 * 8
	srai a7, t0, 56
	vsetvli zero, s2, e8, m2
	vle8.v v8, (a2)
	srai t6, t6, 56
	srai t5, t5, 56
	srai t4, t4, 56
	srai t3, t3, 56
	vle8.v v10, (s9)
	addi a2, a2, 64
	slli t0, t0, 7 * 8
	srai t2, t2, 56
	srai t1, t1, 56
	srai t0, t0, 56
	vle8.v v4, (s7)
	vsrl.vi v12, v8, 4
	vsrl.vi v14, v10, 4
	lb zero, 0(s8)
	vand.vi v8, v8, 0xF
	vand.vi v10, v10, 0xF
	lb zero, 32(s8)
	vsll.vi v0, v4, 4
	vsll.vi v2, v4, 2
	lb zero, 64(s8)
	vsrl.vi v6, v4, 2
	vand.vx v0, v0, a1
	lb zero, 96(s8)
	vand.vx v2, v2, a1
	vand.vx v4, v4, a1
	vand.vx v6, v6, a1
	vor.vv v8, v8, v0
	lb zero, 127(s8)
	vor.vv v10, v10, v2
	vor.vv v12, v12, v4
	vor.vv v14, v14, v6
	vsetvli zero, s0, e8, m8
	vle8.v v0, (s8)
	vsub.vx v8, v8, s2
	vsetvli zero, s1, e8, m4
	vwmul.vv v16, v0, v8
	vwmul.vv v24, v4, v12
	vsetivli zero, 16, e16, m2
	vmv.v.x v0, zero
	vwredsum.vs v10, v16, v0
	vwredsum.vs v9, v18, v0
	vwredsum.vs v8, v20, v0
	vwredsum.vs v7, v22, v0
	vwredsum.vs v11, v24, v0
	vwredsum.vs v12, v26, v0
	vwredsum.vs v13, v28, v0
	vwredsum.vs v14, v30, v0
	vsetivli zero, 4, e32, m1
	vmul.vx v0, v10, t0
	vmul.vx v1, v9, t1
	vmacc.vx v0, t2, v8
	vmacc.vx v1, t3, v7
	vmacc.vx v0, t4, v11
	vmacc.vx v1, t5, v12
	vmacc.vx v0, t6, v13
	vmacc.vx v1, a7, v14
	vadd.vv v0, v0, v1
	vfcvt.f.x.v v0, v0
	vfmv.f.s fa3, v0
	fmadd.s fa5, fa4, fa3, fa5
# 0 "" 2
#NO_APP
	addiw	s5,s5,1
	addi	s4,s4,210
	addi	s3,s3,292
	bgt	s6,s5,.L5
	ld	s0,80(sp)
	.cfi_restore 8
	ld	ra,88(sp)
	.cfi_restore 1
	ld	s1,72(sp)
	.cfi_restore 9
	ld	s2,64(sp)
	.cfi_restore 18
	ld	s3,56(sp)
	.cfi_restore 19
	ld	s4,48(sp)
	.cfi_restore 20
	ld	s6,32(sp)
	.cfi_restore 22
	ld	s7,24(sp)
	.cfi_restore 23
	ld	s8,16(sp)
	.cfi_restore 24
	ld	s9,8(sp)
	.cfi_restore 25
	fsw	fa5,0(s10)
	ld	s5,40(sp)
	.cfi_restore 21
	ld	s10,0(sp)
	.cfi_restore 26
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
.L6:
	.cfi_def_cfa_offset 96
	.cfi_offset 1, -8
	.cfi_offset 21, -56
	.cfi_offset 26, -96
	fmv.s.x	fa5,zero
	ld	ra,88(sp)
	.cfi_restore 1
	ld	s5,40(sp)
	.cfi_restore 21
	fsw	fa5,0(s10)
	ld	s10,0(sp)
	.cfi_restore 26
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
.L10:
	.cfi_def_cfa_offset 96
	.cfi_offset 1, -8
	.cfi_offset 21, -56
	lla	a3,.LANCHOR0
	li	a2,18
	lla	a1,.LC0
	lla	a0,.LC1
	sd	s0,80(sp)
	sd	s1,72(sp)
	sd	s2,64(sp)
	sd	s3,56(sp)
	sd	s4,48(sp)
	sd	s6,32(sp)
	sd	s7,24(sp)
	sd	s8,16(sp)
	sd	s9,8(sp)
	sd	s10,0(sp)
	.cfi_remember_state
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	.cfi_offset 25, -88
	.cfi_offset 26, -96
	call	__assert_fail@plt
.L11:
	.cfi_restore_state
	lla	a3,.LANCHOR0
	li	a2,19
	lla	a1,.LC0
	lla	a0,.LC2
	sd	s0,80(sp)
	sd	s1,72(sp)
	sd	s2,64(sp)
	sd	s3,56(sp)
	sd	s4,48(sp)
	sd	s6,32(sp)
	sd	s7,24(sp)
	sd	s8,16(sp)
	sd	s9,8(sp)
	sd	s10,0(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	.cfi_offset 25, -88
	.cfi_offset 26, -96
	call	__assert_fail@plt
	.cfi_endproc
.LFE1:
	.size	ggml_vec_dot_q6_K_q8_K_vl128, .-ggml_vec_dot_q6_K_q8_K_vl128
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 29
__PRETTY_FUNCTION__.0:
	.string	"ggml_vec_dot_q6_K_q8_K_vl128"
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
