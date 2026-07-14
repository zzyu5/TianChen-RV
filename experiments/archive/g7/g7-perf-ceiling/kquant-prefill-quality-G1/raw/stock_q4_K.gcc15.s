	.file	"stock_q4_K.clean.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"experiments/active/g7-perf-ceiling/kquant-prefill-quality-G1/raw/stock_q4_K.clean.c"
	.align	3
.LC1:
	.string	"n % QK_K == 0"
	.align	3
.LC2:
	.string	"nrc == 1"
	.text
	.align	1
	.globl	ggml_vec_dot_q4_K_q8_K_vl128
	.type	ggml_vec_dot_q4_K_q8_K_vl128, @function
ggml_vec_dot_q4_K_q8_K_vl128:
.LFB1:
	.cfi_startproc
	addi	sp,sp,-96
	.cfi_def_cfa_offset 96
	sd	ra,88(sp)
	andi	t5,a0,0xff
	.cfi_offset 1, -8
	bne	t5,zero,.L10
	li	a4,1
	bne	a7,a4,.L11
	sd	s8,16(sp)
	li	a4,255
	.cfi_offset 24, -80
	mv	s8,a1
	ble	a0,a4,.L6
	fmv.s.x	fa5,zero
	li	t3,1061109760
	addiw	t3,t3,-193
	li	t1,252645376
	addiw	t1,t1,-241
	li	a7,50528256
	addiw	a7,a7,771
	sd	s0,80(sp)
	sd	s1,72(sp)
	sd	s2,64(sp)
	sd	s3,56(sp)
	sd	s4,48(sp)
	sd	s5,40(sp)
	sd	s6,32(sp)
	sd	s7,24(sp)
	sraiw	a0,a0,8
	addi	a5,a5,4
	mv	a4,sp
	addi	t4,sp,8
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
.L5:
	lhu	a6,0(a3)
	lhu	a2,2(a3)
	flw	fa2,-4(a5)
	fcvt.s.wu	fa4,a6
	fcvt.s.wu	fa3,a2
	addi	a6,a5,256
	fmul.s	fa4,fa4,fa2
	fmul.s	fa3,fa3,fa2
	addi	a2,a3,16
#APP
# 54 "experiments/active/g7-perf-ceiling/kquant-prefill-quality-G1/raw/stock_q4_K.clean.c" 1
	li t0, 8
	vsetivli zero, 4, e32, m1, ta, ma
	vle32.v v1, (a3)
	vslide1down.vx v1, v1, zero
	vmv.v.x v16, zero
	vslidedown.vi v2, v1, 2
	vmv1r.v v3, v2
	vslideup.vi v2, v3, 1
	vsetivli zero, 2, e32, m1, ta, ma
	vmv.v.i v4, 4
	vand.vx v8, v1, t3
	vslide1up.vx v5, v4, zero
	vsrl.vi v6, v1, 6
	vsrl.vv v7, v2, v5
	vsse32.v v8, (a4), t0
	vand.vx v0, v6, a7
	vand.vx v2, v7, t1
	vsll.vi v6, v0, 4
	addi t6, a4, 4
	vor.vv v1, v6, v2
	vsse32.v v1, (t6), t0
	vsetivli zero, 8, e16, m1, ta, ma
	vle32.v v2, (a6)
	vnsrl.wi v0, v2, 0
	vnsrl.wi v1, v2, 16
	vadd.vv v2, v0, v1
	vle8.v v3, (t4)
	vzext.vf2 v4, v3
	vwmul.vv v6, v4, v2
	vsetivli zero, 4, e32, m1, ta, ma
	vredsum.vs v0, v6, v16
	vredsum.vs v0, v7, v0
	vfcvt.f.x.v v0, v0
	vfmv.f.s fa2, v0
	vsetivli zero, 16, e8, m1, ta, ma
	vle8.v v0, (a2)
	fnmsub.s fa5, fa3, fa2, fa5
	addi s0, a2, 64
	addi s1, a2, 16
	addi s2, a2, 32
	addi s3, a2, 48
	addi s4, a5, 64
	vle8.v v1, (s1)
	vle8.v v2, (s2)
	addi s5, a5, 16
	addi s1, s1, 64
	addi s6, a5, 32
	vle8.v v3, (s3)
	vle8.v v8, (a5)
	addi s2, s2, 64
	addi s7, a5, 48
	addi s3, s3, 64
	vsrl.vi v4, v0, 4
	vle8.v v9, (s5)
	vle8.v v10, (s6)
	vand.vi v0, v0, 0xF
	addi s5, s5, 64
	vsrl.vi v5, v1, 4
	addi s6, s6, 64
	vle8.v v11, (s7)
	vle8.v v12, (s4)
	vand.vi v1, v1, 0xF
	addi s7, s7, 64
	vsrl.vi v6, v2, 4
	addi s4, s4, 64
	vle8.v v13, (s5)
	vle8.v v14, (s6)
	vand.vi v2, v2, 0xF
	addi s5, s5, 64
	vsrl.vi v7, v3, 4
	addi s6, s6, 64
	vwmul.vv v16, v0, v8
	vle8.v v15, (s7)
	vle8.v v0, (s0)
	vand.vi v3, v3, 0xF
	addi s7, s7, 64
	vwmul.vv v24, v2, v12
	vwmul.vv v20, v4, v10
	vwmul.vv v28, v6, v14
	vwmacc.vv v16, v1, v9
	vle8.v v1, (s1)
	vle8.v v2, (s2)
	vwmacc.vv v24, v3, v13
	vwmacc.vv v20, v5, v11
	vwmacc.vv v28, v7, v15
	addi s0, s4, 64
	addi s1, s5, 64
	vle8.v v3, (s3)
	vle8.v v8, (s4)
	addi s2, s6, 64
	addi s3, s7, 64
	vsrl.vi v4, v0, 4
	vle8.v v9, (s5)
	vle8.v v10, (s6)
	vand.vi v0, v0, 0xF
	vsrl.vi v5, v1, 4
	vsrl.vi v7, v3, 4
	vand.vi v3, v3, 0xF
	vle8.v v11, (s7)
	vle8.v v12, (s0)
	vand.vi v1, v1, 0xF
	vsrl.vi v6, v2, 4
	vand.vi v2, v2, 0xF
	vwmul.vv v18, v0, v8
	vle8.v v13, (s1)
	vle8.v v14, (s2)
	vwmul.vv v26, v2, v12
	vwmul.vv v22, v4, v10
	vwmul.vv v30, v6, v14
	vwmacc.vv v18, v1, v9
	vle8.v v15, (s3)
	vwmacc.vv v26, v3, v13
	vwmacc.vv v22, v5, v11
	vwmacc.vv v30, v7, v15
	vmv.v.x v0, zero
	vsetivli zero, 16, e16, m2, ta, ma
	vwredsum.vs v4, v16, v0
	lbu t6, 0(a4)
	vwredsum.vs v5, v20, v0
	lbu t0, 1(a4)
	vwredsum.vs v6, v24, v0
	lbu t2, 2(a4)
	vwredsum.vs v7, v28, v0
	lbu a1, 3(a4)
	vwredsum.vs v8, v18, v0
	lbu s0, 4(a4)
	vwredsum.vs v9, v22, v0
	lbu s1, 5(a4)
	vwredsum.vs v10, v26, v0
	lbu s2, 6(a4)
	vwredsum.vs v11, v30, v0
	lbu s3, 7(a4)
	vsetivli zero, 4, e32, m1, ta, ma
	vmul.vx v0, v4, t6
	vmul.vx v1, v8, s0
	vmacc.vx v0, t0, v5
	vmacc.vx v1, s1, v9
	vmacc.vx v0, t2, v6
	vmacc.vx v1, s2, v10
	vmacc.vx v0, a1, v7
	vmacc.vx v1, s3, v11
	vfcvt.f.x.v v0, v0
	vfcvt.f.x.v v1, v1
	vfmv.f.s fa1, v0
	vfmv.f.s fa2, v1
	fadd.s fa1, fa1, fa2
	fmadd.s fa5, fa4, fa1, fa5
# 0 "" 2
#NO_APP
	addiw	t5,t5,1
	addi	a3,a3,144
	addi	a5,a5,292
	bgt	a0,t5,.L5
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
	ld	s5,40(sp)
	.cfi_restore 21
	ld	s6,32(sp)
	.cfi_restore 22
	ld	s7,24(sp)
	.cfi_restore 23
	fsw	fa5,0(s8)
	ld	s8,16(sp)
	.cfi_restore 24
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
.L6:
	.cfi_def_cfa_offset 96
	.cfi_offset 1, -8
	.cfi_offset 24, -80
	fmv.s.x	fa5,zero
	ld	ra,88(sp)
	.cfi_restore 1
	fsw	fa5,0(s8)
	ld	s8,16(sp)
	.cfi_restore 24
	addi	sp,sp,96
	.cfi_def_cfa_offset 0
	jr	ra
.L10:
	.cfi_def_cfa_offset 96
	.cfi_offset 1, -8
	lla	a3,.LANCHOR0
	li	a2,17
	lla	a1,.LC0
	lla	a0,.LC1
	sd	s0,80(sp)
	sd	s1,72(sp)
	sd	s2,64(sp)
	sd	s3,56(sp)
	sd	s4,48(sp)
	sd	s5,40(sp)
	sd	s6,32(sp)
	sd	s7,24(sp)
	sd	s8,16(sp)
	.cfi_remember_state
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	call	__assert_fail@plt
.L11:
	.cfi_restore_state
	lla	a3,.LANCHOR0
	li	a2,18
	lla	a1,.LC0
	lla	a0,.LC2
	sd	s0,80(sp)
	sd	s1,72(sp)
	sd	s2,64(sp)
	sd	s3,56(sp)
	sd	s4,48(sp)
	sd	s5,40(sp)
	sd	s6,32(sp)
	sd	s7,24(sp)
	sd	s8,16(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	call	__assert_fail@plt
	.cfi_endproc
.LFE1:
	.size	ggml_vec_dot_q4_K_q8_K_vl128, .-ggml_vec_dot_q4_K_q8_K_vl128
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 29
__PRETTY_FUNCTION__.0:
	.string	"ggml_vec_dot_q4_K_q8_K_vl128"
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
