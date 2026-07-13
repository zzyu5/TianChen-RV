	.file	"stock_q2_K_vl128.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q2_K_vl128.c"
	.align	3
.LC1:
	.string	"nrc == 1"
	.text
	.align	1
	.globl	ggml_vec_dot_q2_K_q8_K_vl128
	.type	ggml_vec_dot_q2_K_q8_K_vl128, @function
ggml_vec_dot_q2_K_q8_K_vl128:
.LFB1:
	.cfi_startproc
	addi	sp,sp,-48
	.cfi_def_cfa_offset 48
	sd	ra,40(sp)
	li	a4,1
	.cfi_offset 1, -8
	bne	a7,a4,.L9
	sd	s2,16(sp)
	li	a4,255
	.cfi_offset 18, -32
	mv	s2,a1
	ble	a0,a4,.L5
	fmv.s.x	fa5,zero
	sd	s0,32(sp)
	sd	s1,24(sp)
	sraiw	a0,a0,8
	addi	a5,a5,260
	li	t3,0
	mv	a2,sp
	addi	t1,sp,8
	.cfi_offset 8, -16
	.cfi_offset 9, -24
.L4:
	flw	fa2,-260(a5)
	lhu	a6,82(a3)
	lhu	a4,80(a3)
	fneg.s	fa1,fa2
	fcvt.s.wu	fa3,a6
	fcvt.s.wu	fa4,a4
	fmul.s	fa3,fa3,fa1
	fmul.s	fa4,fa4,fa2
#APP
# 43 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q2_K_vl128.c" 1
	vsetivli zero, 16, e8, m1
	vmv.v.x v8, zero
	lb zero, 15(a3)
	vle8.v v1, (a3)
	vle8.v v2, (a5)
	addi a1, a5, 16
	vand.vi v0, v1, 0xF
	vsrl.vi v1, v1, 4
	vle8.v v3, (a1)
	vse8.v v0, (a2)
	vsetivli zero, 16, e16, m2
	vzext.vf2 v0, v1
	vwmul.vv v4, v0, v2
	vsetivli zero, 16, e32, m4
	vredsum.vs v8, v4, v8
	vmv.x.s a4, v8
# 0 "" 2
#NO_APP
	fcvt.s.w	fa2,a4
	addi	a7,a5,-256
	li	a4,0
	addi	a6,a3,16
	fmadd.s	fa5,fa2,fa3,fa5
#APP
# 72 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q2_K_vl128.c" 1
	lb zero, 31(a6)
	addi a1, a6, 16
	addi t4, a7, 16
	vsetivli zero, 16, e8, m1
	vle8.v v0, (a6)
	vle8.v v1, (a1)
	vsrl.vi v2, v0, 2
	vsrl.vi v3, v1, 2
	vsrl.vi v4, v0, 4
	addi a1, a7, 32
	vle8.v v8, (a7)
	vle8.v v9, (t4)
	addi t4, t4, 32
	vsrl.vi v5, v1, 4
	vsrl.vi v6, v0, 6
	vsrl.vi v7, v1, 6
	vle8.v v10, (a1)
	vle8.v v11, (t4)
	addi a1, a1, 32
	addi t4, t4, 32
	vand.vi v0, v0, 0x3
	vand.vi v1, v1, 0x3
	vand.vi v2, v2, 0x3
	vle8.v v12, (a1)
	vle8.v v13, (t4)
	addi a1, a1, 32
	addi t4, t4, 32
	vand.vi v3, v3, 0x3
	vand.vi v4, v4, 0x3
	vand.vi v5, v5, 0x3
	vle8.v v14, (a1)
	vle8.v v15, (t4)
	vwmul.vv v16, v0, v8
	vwmul.vv v18, v1, v9
	vwmul.vv v20, v2, v10
	vwmul.vv v22, v3, v11
	vwmul.vv v24, v4, v12
	vwmul.vv v26, v5, v13
	vwmul.vv v28, v6, v14
	vwmul.vv v30, v7, v15
	vsetivli zero, 8, e16, m1
	vmv.v.x v0, zero
	lbu a1, 0(a2)
	vwredsum.vs v8, v16, v0
	vwredsum.vs v9, v18, v0
	lbu t4, 1(a2)
	vwredsum.vs v10, v20, v0
	vwredsum.vs v11, v22, v0
	lbu t5, 2(a2)
	vwredsum.vs v12, v24, v0
	vwredsum.vs v13, v26, v0
	lbu t6, 3(a2)
	vwredsum.vs v14, v28, v0
	vwredsum.vs v15, v30, v0
	lbu t0, 4(a2)
	vwredsum.vs v8, v17, v8
	vwredsum.vs v9, v19, v9
	lbu t2, 5(a2)
	vwredsum.vs v10, v21, v10
	vwredsum.vs v11, v23, v11
	lbu s0, 6(a2)
	vwredsum.vs v12, v25, v12
	vwredsum.vs v13, v27, v13
	lbu s1, 7(a2)
	vwredsum.vs v14, v29, v14
	vwredsum.vs v15, v31, v15
	vsetivli zero, 4, e32, m1
	vmul.vx v0, v8, a1
	vmul.vx v1, v9, t4
	vmacc.vx v0, t5, v10
	vmacc.vx v1, t6, v11
	vmacc.vx v0, t0, v12
	vmacc.vx v1, t2, v13
	vmacc.vx v0, s0, v14
	vmacc.vx v1, s1, v15
	vmv.x.s a1, v0
	vmv.x.s t4, v1
	add a4, a4, a1
	add a4, a4, t4
# 0 "" 2
#NO_APP
	addi	a6,a3,48
	addi	a7,a5,-128
#APP
# 72 "experiments/active/g7-l2-gevm-redesign/kquant-family-knest-G1/raw/stock_q2_K_vl128.c" 1
	lb zero, 31(a6)
	addi t4, a6, 16
	addi t5, a7, 16
	vsetivli zero, 16, e8, m1
	vle8.v v0, (a6)
	vle8.v v1, (t4)
	vsrl.vi v2, v0, 2
	vsrl.vi v3, v1, 2
	vsrl.vi v4, v0, 4
	addi t4, a7, 32
	vle8.v v8, (a7)
	vle8.v v9, (t5)
	addi t5, t5, 32
	vsrl.vi v5, v1, 4
	vsrl.vi v6, v0, 6
	vsrl.vi v7, v1, 6
	vle8.v v10, (t4)
	vle8.v v11, (t5)
	addi t4, t4, 32
	addi t5, t5, 32
	vand.vi v0, v0, 0x3
	vand.vi v1, v1, 0x3
	vand.vi v2, v2, 0x3
	vle8.v v12, (t4)
	vle8.v v13, (t5)
	addi t4, t4, 32
	addi t5, t5, 32
	vand.vi v3, v3, 0x3
	vand.vi v4, v4, 0x3
	vand.vi v5, v5, 0x3
	vle8.v v14, (t4)
	vle8.v v15, (t5)
	vwmul.vv v16, v0, v8
	vwmul.vv v18, v1, v9
	vwmul.vv v20, v2, v10
	vwmul.vv v22, v3, v11
	vwmul.vv v24, v4, v12
	vwmul.vv v26, v5, v13
	vwmul.vv v28, v6, v14
	vwmul.vv v30, v7, v15
	vsetivli zero, 8, e16, m1
	vmv.v.x v0, zero
	lbu t4, 0(t1)
	vwredsum.vs v8, v16, v0
	vwredsum.vs v9, v18, v0
	lbu t5, 1(t1)
	vwredsum.vs v10, v20, v0
	vwredsum.vs v11, v22, v0
	lbu t6, 2(t1)
	vwredsum.vs v12, v24, v0
	vwredsum.vs v13, v26, v0
	lbu t0, 3(t1)
	vwredsum.vs v14, v28, v0
	vwredsum.vs v15, v30, v0
	lbu t2, 4(t1)
	vwredsum.vs v8, v17, v8
	vwredsum.vs v9, v19, v9
	lbu a1, 5(t1)
	vwredsum.vs v10, v21, v10
	vwredsum.vs v11, v23, v11
	lbu s0, 6(t1)
	vwredsum.vs v12, v25, v12
	vwredsum.vs v13, v27, v13
	lbu s1, 7(t1)
	vwredsum.vs v14, v29, v14
	vwredsum.vs v15, v31, v15
	vsetivli zero, 4, e32, m1
	vmul.vx v0, v8, t4
	vmul.vx v1, v9, t5
	vmacc.vx v0, t6, v10
	vmacc.vx v1, t0, v11
	vmacc.vx v0, t2, v12
	vmacc.vx v1, a1, v13
	vmacc.vx v0, s0, v14
	vmacc.vx v1, s1, v15
	vmv.x.s t4, v0
	vmv.x.s t5, v1
	add a4, a4, t4
	add a4, a4, t5
# 0 "" 2
#NO_APP
	fcvt.s.w	fa3,a4
	addiw	t3,t3,1
	addi	a3,a3,84
	fmadd.s	fa5,fa3,fa4,fa5
	addi	a5,a5,292
	bgt	a0,t3,.L4
	ld	s0,32(sp)
	.cfi_restore 8
	ld	ra,40(sp)
	.cfi_restore 1
	ld	s1,24(sp)
	.cfi_restore 9
	fsw	fa5,0(s2)
	ld	s2,16(sp)
	.cfi_restore 18
	addi	sp,sp,48
	.cfi_def_cfa_offset 0
	jr	ra
.L5:
	.cfi_def_cfa_offset 48
	.cfi_offset 1, -8
	.cfi_offset 18, -32
	fmv.s.x	fa5,zero
	ld	ra,40(sp)
	.cfi_restore 1
	fsw	fa5,0(s2)
	ld	s2,16(sp)
	.cfi_restore 18
	addi	sp,sp,48
	.cfi_def_cfa_offset 0
	jr	ra
.L9:
	.cfi_def_cfa_offset 48
	.cfi_offset 1, -8
	lla	a3,.LANCHOR0
	li	a2,17
	lla	a1,.LC0
	lla	a0,.LC1
	sd	s0,32(sp)
	sd	s1,24(sp)
	sd	s2,16(sp)
	.cfi_offset 8, -16
	.cfi_offset 9, -24
	.cfi_offset 18, -32
	call	__assert_fail@plt
	.cfi_endproc
.LFE1:
	.size	ggml_vec_dot_q2_K_q8_K_vl128, .-ggml_vec_dot_q2_K_q8_K_vl128
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 29
__PRETTY_FUNCTION__.0:
	.string	"ggml_vec_dot_q2_K_q8_K_vl128"
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
