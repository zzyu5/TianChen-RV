	.file	"q3k_gevm_variants.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.align	1
	.globl	cur_q3k
	.type	cur_q3k, @function
cur_q3k:
.LFB1:
	.cfi_startproc
	srli	a5,a4,4
	beq	a5,zero,.L16
	vsetivli	zero,16,e32,m2,ta,ma
	addi	sp,sp,-32
	.cfi_def_cfa_offset 32
	vmv.v.i	v14,0
	sd	s2,8(sp)
	sd	s1,16(sp)
	sd	s0,24(sp)
	slli	a5,a5,6
	vmv2r.v	v10,v14
	.cfi_offset 18, -24
	.cfi_offset 9, -16
	.cfi_offset 8, -8
	srli	s1,a0,8
	mv	s2,a3
	add	s0,a1,a5
	mv	a3,a1
	beq	s1,zero,.L3
	li	a5,1824
	vmv.v.i	v12,0
	mul	s1,s1,a5
	vsetvli	zero,zero,e16,m1,ta,ma
	addi	a2,a2,2
	vmv.v.i	v7,0
	sd	s3,0(sp)
	.cfi_offset 19, -32
	vsetvli	zero,zero,e8,mf2,ta,ma
	li	t0,8
	add	t2,a2,s1
.L6:
	mv	t1,s2
	sub	a1,t2,s1
.L5:
	addi	a5,a1,254
	vle8.v	v9,0(a1)
	vle8.v	v16,0(a5)
	addi	a5,a1,510
	vle8.v	v5,0(a5)
	addi	a5,a1,270
	vle8.v	v1,0(a5)
	addi	a5,a1,526
	vle8.v	v4,0(a5)
	addi	a5,a1,286
	vle8.v	v8,0(a5)
	addi	a5,a1,542
	vle8.v	v3,0(a5)
	addi	a5,a1,302
	vand.vi	v16,v16,1
	lbu	t4,4(t1)
	vand.vi	v5,v5,3
	lbu	t3,5(t1)
	vand.vi	v1,v1,1
	lbu	a6,6(t1)
	vsll.vi	v16,v16,2
	lbu	a0,7(t1)
	vle8.v	v6,0(a5)
	addi	a5,a1,558
	vsll.vi	v1,v1,2
	flw	fa5,0(t1)
	vand.vi	v4,v4,3
	addi	s3,a1,-2
	vor.vv	v5,v5,v16
	addi	a7,a1,16
	vand.vi	v8,v8,1
	addi	a2,t1,8
	vle8.v	v2,0(a5)
	li	a4,1
	vor.vv	v4,v4,v1
	addi	a5,a1,574
	vadd.vi	v5,v5,-4
	vmv1r.v	v1,v7
	vsll.vi	v8,v8,2
	vand.vi	v3,v3,3
	vwmacc.vx	v1,t4,v5
	vand.vi	v5,v6,1
	vadd.vi	v4,v4,-4
	vor.vv	v3,v3,v8
	vsll.vi	v5,v5,2
	vand.vi	v2,v2,3
	vwmacc.vx	v1,t3,v4
	vadd.vi	v3,v3,-4
	vor.vv	v2,v2,v5
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v4,v9
	vsetvli	zero,zero,e8,mf2,ta,ma
	vwmacc.vx	v1,a6,v3
	vadd.vi	v2,v2,-4
	vmv2r.v	v8,v12
	vwmacc.vx	v1,a0,v2
	vsetvli	zero,zero,e16,m1,ta,ma
	vwmacc.vv	v8,v4,v1
.L4:
	vsetvli	zero,zero,e8,mf2,ta,ma
	addi	a0,a5,-256
	vle8.v	v2,0(a0)
	addi	a0,a5,-240
	vle8.v	v1,0(a0)
	addi	a0,a5,-224
	vle8.v	v16,0(a5)
	addi	a6,a5,32
	vle8.v	v6,0(a0)
	addi	a0,a5,16
	vle8.v	v5,0(a0)
	addi	a0,a5,-208
	vle8.v	v4,0(a0)
	addi	a0,a5,48
	vsrl.vx	v2,v2,a4
	lbu	t6,0(a2)
	vsrl.vx	v1,v1,a4
	lbu	t5,1(a2)
	vand.vi	v16,v16,3
	lbu	t4,2(a2)
	vand.vi	v2,v2,1
	lbu	t3,3(a2)
	vand.vi	v1,v1,1
	addi	a2,a2,4
	vsrl.vx	v6,v6,a4
	addi	a5,a5,64
	vsll.vi	v2,v2,2
	vle8.v	v3,0(a6)
	vsll.vi	v1,v1,2
	vor.vv	v16,v16,v2
	vand.vi	v5,v5,3
	vand.vi	v6,v6,1
	vsrl.vx	v4,v4,a4
	addi	a4,a4,1
	vor.vv	v5,v5,v1
	vle8.v	v2,0(a0)
	vadd.vi	v16,v16,-4
	vmv1r.v	v1,v7
	vsll.vi	v6,v6,2
	vand.vi	v3,v3,3
	vwmacc.vx	v1,t6,v16
	vand.vi	v4,v4,1
	vadd.vi	v5,v5,-4
	vor.vv	v3,v3,v6
	vsll.vi	v4,v4,2
	vand.vi	v2,v2,3
	vwmacc.vx	v1,t5,v5
	vadd.vi	v3,v3,-4
	vor.vv	v2,v2,v4
	vle8.v	v4,0(a7)
	addi	a7,a7,16
	vwmacc.vx	v1,t4,v3
	vadd.vi	v2,v2,-4
	vwmacc.vx	v1,t3,v2
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v2,v4
	vwmacc.vv	v8,v2,v1
	bne	a4,t0,.L4
	vsetvli	zero,zero,e32,m2,ta,ma
	addi	a1,a1,1824
	vle16.v	v1,0(s3)
	addi	t1,t1,292
	vfcvt.f.x.v	v8,v8
	vsetvli	zero,zero,e16,m1,ta,ma
	vfwcvt.f.f.v	v2,v1
	vsetvli	zero,zero,e32,m2,ta,ma
	vfmul.vf	v2,v2,fa5
	vfmacc.vv	v10,v8,v2
	beq	a1,t2,.L25
	vsetvli	zero,zero,e8,mf2,ta,ma
	j	.L5
.L3:
	.cfi_restore 19
	vse32.v	v14,0(a3)
	addi	a5,a3,64
	addi	a3,a3,128
	beq	a5,s0,.L14
	vse32.v	v14,0(a5)
	bne	a3,s0,.L3
.L14:
	ld	s0,24(sp)
	.cfi_restore 8
	ld	s2,8(sp)
	.cfi_restore 18
	ld	s1,16(sp)
	.cfi_restore 9
	addi	sp,sp,32
	.cfi_def_cfa_offset 0
	jr	ra
.L25:
	.cfi_def_cfa_offset 32
	.cfi_offset 8, -8
	.cfi_offset 9, -16
	.cfi_offset 18, -24
	.cfi_offset 19, -32
	vsetvli	zero,zero,e8,mf2,ta,ma
	vse32.v	v10,0(a3)
	addi	a3,a3,64
	beq	s0,a3,.L19
	vmv2r.v	v10,v14
	add	t2,t2,s1
	j	.L6
.L19:
	ld	s0,24(sp)
	.cfi_restore 8
	ld	s3,0(sp)
	.cfi_restore 19
	ld	s2,8(sp)
	.cfi_restore 18
	ld	s1,16(sp)
	.cfi_restore 9
	addi	sp,sp,32
	.cfi_def_cfa_offset 0
	jr	ra
.L16:
	ret
	.cfi_endproc
.LFE1:
	.size	cur_q3k, .-cur_q3k
	.align	1
	.globl	knest_q3k
	.type	knest_q3k, @function
knest_q3k:
.LFB2:
	.cfi_startproc
	srli	a5,a4,4
	beq	a5,zero,.L41
	vsetivli	zero,16,e32,m2,ta,ma
	addi	sp,sp,-32
	.cfi_def_cfa_offset 32
	vmv.v.i	v14,0
	sd	s2,8(sp)
	sd	s1,16(sp)
	sd	s0,24(sp)
	slli	a5,a5,6
	vmv2r.v	v10,v14
	.cfi_offset 18, -24
	.cfi_offset 9, -16
	.cfi_offset 8, -8
	srli	s2,a0,8
	mv	s0,a1
	add	s1,a1,a5
	beq	s2,zero,.L28
	li	a5,1824
	vmv.v.i	v12,0
	mul	s2,s2,a5
	vsetvli	zero,zero,e16,m1,ta,ma
	vmv.v.i	v7,0
	sd	s3,0(sp)
	.cfi_offset 19, -32
	li	t6,1
	vsetvli	zero,zero,e8,mf2,ta,mu
	mv	s3,a3
	li	t5,8
	add	a3,a2,s2
.L31:
	mv	t2,s3
	sub	t0,a3,s2
.L30:
	flw	fa5,0(t2)
	vmv2r.v	v8,v12
	addi	a6,t0,2
	addi	a5,t0,512
	addi	a4,t2,4
	li	a1,0
	j	.L29
.L47:
	vsetvli	zero,zero,e8,mf2,ta,mu
.L29:
	addi	a2,a5,-256
	vle8.v	v1,0(a5)
	vle8.v	v0,0(a2)
	addi	a2,a5,-240
	vle8.v	v16,0(a2)
	sllw	a2,t6,a1
	addi	a0,a5,16
	vle8.v	v17,0(a6)
	vle8.v	v4,0(a0)
	addi	a0,a5,-224
	vle8.v	v6,0(a0)
	addi	t1,a5,32
	vand.vi	v2,v1,3
	addi	a7,a5,-208
	vand.vx	v0,v0,a2
	lbu	t4,0(a4)
	vand.vx	v16,v16,a2
	addi	a0,a5,48
	vle8.v	v3,0(t1)
	lbu	t3,1(a4)
	vmseq.vi	v0,v0,0
	lbu	t1,2(a4)
	vmv1r.v	v1,v7
	addiw	a1,a1,1
	vle8.v	v5,0(a7)
	lbu	a7,3(a4)
	vadd.vi	v2,v2,-4,v0.t
	addi	a4,a4,4
	vmseq.vi	v0,v16,0
	addi	a6,a6,16
	vand.vi	v4,v4,3
	addi	a5,a5,64
	vand.vx	v6,v6,a2
	vwmacc.vx	v1,t4,v2
	vadd.vi	v4,v4,-4,v0.t
	vle8.v	v2,0(a0)
	vmseq.vi	v0,v6,0
	vand.vi	v3,v3,3
	vwmacc.vx	v1,t3,v4
	vand.vx	v5,v5,a2
	vadd.vi	v3,v3,-4,v0.t
	vand.vi	v2,v2,3
	vmseq.vi	v0,v5,0
	vwmacc.vx	v1,t1,v3
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v3,v17
	vsetvli	zero,zero,e8,mf2,ta,mu
	vadd.vi	v2,v2,-4,v0.t
	vwmacc.vx	v1,a7,v2
	vsetvli	zero,zero,e16,m1,ta,ma
	vwmacc.vv	v8,v3,v1
	bne	a1,t5,.L47
	vsetvli	zero,zero,e32,m2,ta,ma
	addi	t2,t2,292
	vle16.v	v1,0(t0)
	addi	t0,t0,1824
	vfcvt.f.x.v	v8,v8
	vsetvli	zero,zero,e16,m1,ta,ma
	vfwcvt.f.f.v	v2,v1
	vsetvli	zero,zero,e32,m2,ta,ma
	vfmul.vf	v2,v2,fa5
	vfmacc.vv	v10,v8,v2
	beq	a3,t0,.L49
	vsetvli	zero,zero,e8,mf2,ta,mu
	j	.L30
.L28:
	.cfi_restore 19
	vse32.v	v14,0(s0)
	addi	a5,s0,64
	addi	s0,s0,128
	beq	a5,s1,.L39
	vse32.v	v14,0(a5)
	bne	s0,s1,.L28
.L39:
	ld	s0,24(sp)
	.cfi_restore 8
	ld	s2,8(sp)
	.cfi_restore 18
	ld	s1,16(sp)
	.cfi_restore 9
	addi	sp,sp,32
	.cfi_def_cfa_offset 0
	jr	ra
.L49:
	.cfi_def_cfa_offset 32
	.cfi_offset 8, -8
	.cfi_offset 9, -16
	.cfi_offset 18, -24
	.cfi_offset 19, -32
	vsetvli	zero,zero,e8,mf2,ta,mu
	vse32.v	v10,0(s0)
	addi	s0,s0,64
	beq	s0,s1,.L44
	vmv2r.v	v10,v14
	add	a3,a3,s2
	j	.L31
.L44:
	ld	s0,24(sp)
	.cfi_restore 8
	ld	s3,0(sp)
	.cfi_restore 19
	ld	s2,8(sp)
	.cfi_restore 18
	ld	s1,16(sp)
	.cfi_restore 9
	addi	sp,sp,32
	.cfi_def_cfa_offset 0
	jr	ra
.L41:
	ret
	.cfi_endproc
.LFE2:
	.size	knest_q3k, .-knest_q3k
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
