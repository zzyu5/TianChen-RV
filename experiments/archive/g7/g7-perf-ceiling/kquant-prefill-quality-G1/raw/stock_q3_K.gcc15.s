	.file	"stock_q3_K_vl128.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q3_K_vl128.c"
	.align	3
.LC1:
	.string	"n % QK_K == 0"
	.align	3
.LC2:
	.string	"nrc == 1"
	.text
	.align	1
	.globl	ggml_vec_dot_q3_K_q8_K_vl128
	.type	ggml_vec_dot_q3_K_q8_K_vl128, @function
ggml_vec_dot_q3_K_q8_K_vl128:
.LFB1:
	.cfi_startproc
	addi	sp,sp,-128
	.cfi_def_cfa_offset 128
	mv	a4,a0
	sd	ra,120(sp)
	andi	a0,a0,0xff
	.cfi_offset 1, -8
	bne	a0,zero,.L10
	li	a2,1
	bne	a7,a2,.L11
	sd	s11,24(sp)
	li	a2,255
	.cfi_offset 27, -104
	mv	s11,a1
	ble	a4,a2,.L6
	li	a2,4
	fmv.s.x	fa4,zero
	sd	s2,96(sp)
	.cfi_offset 18, -32
	slli	s2,a2,32
	sd	s0,112(sp)
	sd	s1,104(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	li	s0,252645376
	addiw	s0,s0,-241
	li	s1,50528256
	addiw	s1,s1,771
	sd	s3,88(sp)
	sd	s4,80(sp)
	sd	s5,72(sp)
	sd	s6,64(sp)
	sd	s7,56(sp)
	sd	s8,48(sp)
	sd	s9,40(sp)
	sd	s10,32(sp)
	add	s2,s2,a2
	sraiw	a1,a4,8
	addi	a3,a3,96
	add	a5,a5,a2
	mv	t5,sp
	addi	t2,sp,8
	li	a6,32
	li	t4,128
	li	t3,64
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	.cfi_offset 25, -88
	.cfi_offset 26, -96
.L5:
	addi	t1,a3,-96
#APP
# 43 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q3_K_vl128.c" 1
	vsetivli zero, 12, e8, m1
	vle8.v v0, (a3)
	vmv1r.v v2, v0
	vsetivli zero, 2, e64, m1
	vmv.v.x v9, s2
	vslidedown.vi v1, v0, 1
	vslide1up.vx v8, v9, zero
	vslideup.vi v0, v2, 1
	vsetivli zero, 4, e32, m1
	vid.v v9
	vmv.x.s a4, v1
	vsll.vi v9, v9, 1
	vmv.v.x v1, a4
	vsrl.vv v4, v1, v9
	vsrl.vv v2, v0, v8
	vand.vx v5, v4, s1
	vand.vx v3, v2, s0
	vsll.vi v6, v5, 4
	vor.vv v7, v6, v3
	vsetivli zero, 16, e8, m1
	vsub.vx v0, v7, a6
	vse8.v v0, (t5)
# 0 "" 2
#NO_APP
	li	a4,0
	addi	t6,a3,-64
	mv	a2,a7
#APP
# 79 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q3_K_vl128.c" 1
	lb zero, 31(t6)
	vsetvli zero, a6, e8, m2, ta, mu
	vle8.v v8, (t6)
	vsrl.vi v10, v8, 2
	vsrl.vi v12, v8, 4
	vsrl.vi v14, v8, 6
	lb zero, 64(a5)
	vand.vi v8, v8, 3
	vand.vi v10, v10, 3
	vand.vi v12, v12, 3
	vle8.v v2, (t1)
	lb zero, 127(a5)
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v8, v8, -4, v0.t
	lb zero, 0(a5)
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v10, v10, -4, v0.t
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v12, v12, -4, v0.t
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v14, v14, -4, v0.t
	vsetvli zero, t4, e8, m8
	vle8.v v0, (a5)
	lb t0, 0(t5)
	lb s3, 1(t5)
	lb s4, 2(t5)
	lb s5, 3(t5)
	vsetvli zero, t3, e8, m4
	vwmul.vv v16, v0, v8
	vwmul.vv v24, v4, v12
	vsetivli zero, 16, e16, m2
	vmv.v.x v0, zero
	vwredsum.vs v8, v16, v0
	lb s6, 4(t5)
	lb s7, 5(t5)
	vwredsum.vs v9, v18, v0
	vwredsum.vs v10, v20, v0
	vwredsum.vs v11, v22, v0
	vwredsum.vs v12, v24, v0
	lb s8, 6(t5)
	lb s9, 7(t5)
	vwredsum.vs v13, v26, v0
	vwredsum.vs v14, v28, v0
	vwredsum.vs v15, v30, v0
	vsetivli zero, 4, e32, m1
	vmul.vx v0, v8, t0
	vmul.vx v1, v9, s3
	vmacc.vx v0, s4, v10
	vmacc.vx v1, s5, v11
	vmacc.vx v0, s6, v12
	vmacc.vx v1, s7, v13
	vmacc.vx v0, s8, v14
	vmacc.vx v1, s9, v15
	vmv.x.s t0, v0
	vmv.x.s s3, v1
	add a4, a4, t0
	add a4, a4, s3
# 0 "" 2
#NO_APP
	addi	t6,a3,-32
	addi	t0,a5,128
#APP
# 79 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q3_K_vl128.c" 1
	lb zero, 31(t6)
	vsetvli zero, a6, e8, m2, ta, mu
	vle8.v v8, (t6)
	vsrl.vi v10, v8, 2
	vsrl.vi v12, v8, 4
	vsrl.vi v14, v8, 6
	lb zero, 64(t0)
	vand.vi v8, v8, 3
	vand.vi v10, v10, 3
	vand.vi v12, v12, 3
	vle8.v v2, (t1)
	lb zero, 127(t0)
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v8, v8, -4, v0.t
	lb zero, 0(t0)
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v10, v10, -4, v0.t
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v12, v12, -4, v0.t
	vand.vx v4, v2, a2
	slli a2, a2, 1
	vmseq.vx v0, v4, zero
	vadd.vi v14, v14, -4, v0.t
	vsetvli zero, t4, e8, m8
	vle8.v v0, (t0)
	lb s3, 0(t2)
	lb s4, 1(t2)
	lb s5, 2(t2)
	lb s6, 3(t2)
	vsetvli zero, t3, e8, m4
	vwmul.vv v16, v0, v8
	vwmul.vv v24, v4, v12
	vsetivli zero, 16, e16, m2
	vmv.v.x v0, zero
	vwredsum.vs v8, v16, v0
	lb s7, 4(t2)
	lb s8, 5(t2)
	vwredsum.vs v9, v18, v0
	vwredsum.vs v10, v20, v0
	vwredsum.vs v11, v22, v0
	vwredsum.vs v12, v24, v0
	lb s9, 6(t2)
	lb s10, 7(t2)
	vwredsum.vs v13, v26, v0
	vwredsum.vs v14, v28, v0
	vwredsum.vs v15, v30, v0
	vsetivli zero, 4, e32, m1
	vmul.vx v0, v8, s3
	vmul.vx v1, v9, s4
	vmacc.vx v0, s5, v10
	vmacc.vx v1, s6, v11
	vmacc.vx v0, s7, v12
	vmacc.vx v1, s8, v13
	vmacc.vx v0, s9, v14
	vmacc.vx v1, s10, v15
	vmv.x.s s3, v0
	vmv.x.s s4, v1
	add a4, a4, s3
	add a4, a4, s4
# 0 "" 2
#NO_APP
	lhu	a2,12(a3)
	flw	fa2,-4(a5)
	fcvt.s.w	fa3,a4
	fcvt.s.wu	fa5,a2
	addiw	a0,a0,1
	addi	a3,a3,110
	fmul.s	fa5,fa5,fa2
	addi	a5,a5,292
	fmadd.s	fa4,fa3,fa5,fa4
	bgt	a1,a0,.L5
	ld	s0,112(sp)
	.cfi_restore 8
	ld	ra,120(sp)
	.cfi_restore 1
	ld	s1,104(sp)
	.cfi_restore 9
	ld	s2,96(sp)
	.cfi_restore 18
	ld	s3,88(sp)
	.cfi_restore 19
	ld	s4,80(sp)
	.cfi_restore 20
	ld	s5,72(sp)
	.cfi_restore 21
	ld	s6,64(sp)
	.cfi_restore 22
	ld	s7,56(sp)
	.cfi_restore 23
	ld	s8,48(sp)
	.cfi_restore 24
	ld	s9,40(sp)
	.cfi_restore 25
	ld	s10,32(sp)
	.cfi_restore 26
	fsw	fa4,0(s11)
	ld	s11,24(sp)
	.cfi_restore 27
	addi	sp,sp,128
	.cfi_def_cfa_offset 0
	jr	ra
.L6:
	.cfi_def_cfa_offset 128
	.cfi_offset 1, -8
	.cfi_offset 27, -104
	fmv.s.x	fa4,zero
	ld	ra,120(sp)
	.cfi_restore 1
	fsw	fa4,0(s11)
	ld	s11,24(sp)
	.cfi_restore 27
	addi	sp,sp,128
	.cfi_def_cfa_offset 0
	jr	ra
.L10:
	.cfi_def_cfa_offset 128
	.cfi_offset 1, -8
	lla	a3,.LANCHOR0
	li	a2,17
	lla	a1,.LC0
	lla	a0,.LC1
	sd	s0,112(sp)
	sd	s1,104(sp)
	sd	s2,96(sp)
	sd	s3,88(sp)
	sd	s4,80(sp)
	sd	s5,72(sp)
	sd	s6,64(sp)
	sd	s7,56(sp)
	sd	s8,48(sp)
	sd	s9,40(sp)
	sd	s10,32(sp)
	sd	s11,24(sp)
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
	.cfi_offset 25, -88
	.cfi_offset 26, -96
	.cfi_offset 27, -104
	call	__assert_fail@plt
.L11:
	.cfi_restore_state
	lla	a3,.LANCHOR0
	li	a2,18
	lla	a1,.LC0
	lla	a0,.LC2
	sd	s0,112(sp)
	sd	s1,104(sp)
	sd	s2,96(sp)
	sd	s3,88(sp)
	sd	s4,80(sp)
	sd	s5,72(sp)
	sd	s6,64(sp)
	sd	s7,56(sp)
	sd	s8,48(sp)
	sd	s9,40(sp)
	sd	s10,32(sp)
	sd	s11,24(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	.cfi_offset 19, -40
	.cfi_offset 20, -48
	.cfi_offset 21, -56
	.cfi_offset 22, -64
	.cfi_offset 23, -72
	.cfi_offset 24, -80
	.cfi_offset 25, -88
	.cfi_offset 26, -96
	.cfi_offset 27, -104
	call	__assert_fail@plt
	.cfi_endproc
.LFE1:
	.size	ggml_vec_dot_q3_K_q8_K_vl128, .-ggml_vec_dot_q3_K_q8_K_vl128
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 29
__PRETTY_FUNCTION__.0:
	.string	"ggml_vec_dot_q3_K_q8_K_vl128"
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
