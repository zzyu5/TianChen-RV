	.file	"q5k_gevm_variants.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.align	1
	.globl	cur_q5k
	.type	cur_q5k, @function
cur_q5k:
.LFB3:
	.cfi_startproc
	srli	a5,a4,4
	beq	a5,zero,.L31
	vsetivli	zero,16,e32,m2,ta,ma
	mv	a4,a1
	vmv.v.i	v22,0
	slli	a5,a5,6
	srli	a1,a0,8
	add	a0,a4,a5
	vmv2r.v	v10,v22
	beq	a1,zero,.L3
	li	a5,292
	vmv.v.i	v20,0
	mul	a5,a1,a5
	vsetvli	zero,zero,e16,m1,ta,ma
	vmv.v.i	v8,0
	addi	sp,sp,-144
	.cfi_def_cfa_offset 144
	mv	a6,a3
	li	a7,4096
	addi	a7,a7,-1280
	sd	s6,88(sp)
	sd	s2,120(sp)
	sd	s1,128(sp)
	sd	s11,48(sp)
	sd	s10,56(sp)
	mul	a7,a1,a7
	sd	s9,64(sp)
	sd	s8,72(sp)
	sd	s7,80(sp)
	sd	s5,96(sp)
	sd	s4,104(sp)
	sd	s3,112(sp)
	sd	s0,136(sp)
	.cfi_offset 22, -56
	.cfi_offset 18, -24
	.cfi_offset 9, -16
	.cfi_offset 27, -96
	.cfi_offset 26, -88
	.cfi_offset 25, -80
	.cfi_offset 24, -72
	.cfi_offset 23, -64
	.cfi_offset 21, -48
	.cfi_offset 20, -40
	.cfi_offset 19, -32
	.cfi_offset 8, -8
	mv	a3,a2
	li	s2,48
	li	s1,-64
	li	t2,16
	li	s6,1024
	add	a1,a6,a5
.L17:
	sd	a7,32(sp)
	sd	a4,40(sp)
	addi	s0,a3,512
	mv	a2,a3
	mv	a5,a6
	sd	a0,24(sp)
	sd	a1,16(sp)
	mv	a7,a3
	mv	a4,a6
.L15:
	addi	a3,a2,32
	vmv2r.v	v6,v20
	vle16.v	v1,0(a3)
	addi	a3,a2,192
	vle8.v	v9,0(a3)
	flw	fa5,0(a5)
	vmv2r.v	v4,v20
	li	s3,0
	addi	s8,a2,208
	addi	t5,a5,68
	addi	a3,a5,260
	vfwcvt.f.f.v	v12,v1
	addi	a0,a2,80
	vsetvli	zero,zero,e8,mf2,ta,ma
	li	a6,0
	vand.vx	v16,v9,s1
	li	t1,2
	vand.vi	v15,v9,3
	li	t6,0
	vand.vi	v14,v9,12
	addi	s7,a2,224
	vsetvli	zero,zero,e32,m2,ta,ma
	addi	s5,a2,240
	vfmul.vf	v12,v12,fa5
	mv	a1,a5
	vsetvli	zero,zero,e8,mf2,ta,ma
	vsrl.vi	v16,v16,2
	vsll.vi	v15,v15,4
	vsll.vi	v14,v14,2
	vand.vx	v9,v9,s2
.L14:
	addi	a5,a0,-16
	vle8.v	v1,0(a5)
	vand.vi	v25,v1,15
	vsrl.vi	v3,v1,4
	bne	s3,zero,.L4
	vle8.v	v19,0(s8)
	addi	t3,a0,16
	vle8.v	v17,0(s7)
	addi	a5,a0,32
	vle8.v	v2,0(a0)
	vle8.v	v26,0(t3)
	vle8.v	v1,0(s5)
	vle8.v	v18,0(a5)
	vand.vi	v29,v19,3
	vand.vi	v28,v17,3
	vand.vi	v19,v19,12
	vand.vi	v17,v17,12
	vand.vi	v31,v2,15
	vand.vi	v30,v26,15
	vsrl.vi	v2,v2,4
	vsll.vi	v17,v17,2
	vsll.vi	v29,v29,4
	vsll.vi	v19,v19,2
	vsll.vi	v28,v28,4
	vsrl.vi	v26,v26,4
	vor.vv	v24,v14,v3
	vor.vv	v19,v19,v2
	vor.vv	v26,v17,v26
	vor.vv	v27,v15,v25
	vor.vv	v29,v29,v31
	vor.vv	v28,v28,v30
	vand.vi	v2,v1,3
	vand.vi	v3,v1,12
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v25,v27
	vzext.vf2	v1,v29
	vzext.vf2	v27,v24
	vzext.vf2	v17,v28
	vzext.vf2	v24,v19
	vzext.vf2	v19,v26
	vsetvli	zero,zero,e8,mf2,ta,ma
	vand.vi	v26,v18,15
	vsll.vi	v2,v2,4
	vsll.vi	v3,v3,2
	vsrl.vi	v18,v18,4
.L5:
	lhu	t4,0(a3)
	vor.vv	v3,v3,v18
	lhu	t3,2(a3)
	vor.vv	v2,v2,v26
	vsetvli	zero,zero,e16,m1,ta,ma
	lhu	a5,6(a3)
	vzext.vf2	v26,v3
	li	s10,0
	addi	s4,t6,512
	vzext.vf2	v18,v2
	addw	t3,t3,t4
	vwmacc.vx	v6,t3,v27
	lhu	t4,4(a3)
	lhu	t3,8(a3)
	addi	t0,t5,-64
	addw	a5,a5,t4
	vwmacc.vx	v6,a5,v24
	lhu	t4,10(a3)
	lhu	a5,12(a3)
	mv	s9,s0
	addw	t3,t3,t4
	vwmacc.vx	v6,t3,v19
	lhu	t3,14(a3)
	addw	a5,a5,t3
	vwmacc.vx	v6,a5,v26
	beq	a6,zero,.L8
	addi	s9,a2,512
.L11:
	vsetvli	zero,zero,e8,mf2,ta,ma
	addi	a5,s9,-256
	vmv1r.v	v19,v8
	mv	t3,t0
	vmv1r.v	v24,v8
.L10:
	vle8.v	v26,0(a5)
	add	t4,s4,a5
	vle8.v	v2,0(t4)
	lbu	s11,0(t3)
	lbu	t4,32(t3)
	addi	a5,a5,16
	addi	t3,t3,1
	vsrl.vi	v3,v26,4
	vsrl.vi	v27,v26,5
	vand.vi	v26,v2,15
	vand.vi	v3,v3,1
	vand.vi	v27,v27,1
	vsrl.vi	v2,v2,4
	vsll.vi	v3,v3,4
	vsll.vi	v27,v27,4
	vor.vv	v3,v26,v3
	vor.vv	v2,v2,v27
	vwmacc.vx	v24,s11,v3
	vwmacc.vx	v19,t4,v2
	bne	s9,a5,.L10
	vsetvli	zero,zero,e16,m1,ta,ma
	addi	t0,t0,16
	vwmacc.vv	v4,v25,v24
	addi	s9,s9,256
	vwmacc.vv	v4,v1,v19
	beq	s10,t2,.L9
	li	s10,16
	j	.L11
.L42:
	vsetvli	zero,zero,e16,m1,ta,ma
	ld	t0,8(sp)
	vwmacc.vv	v4,v25,v26
	addi	s9,s9,256
	addi	t0,t0,16
	vwmacc.vv	v4,v1,v24
	beq	s10,t2,.L9
	li	s10,16
.L8:
	vsetvli	zero,zero,e8,mf2,ta,ma
	addi	a5,s9,-256
	vmv1r.v	v24,v8
	mv	t3,t0
	vmv1r.v	v26,v8
	sd	t0,8(sp)
.L7:
	vle8.v	v3,0(a5)
	add	t4,s4,a5
	vle8.v	v2,0(t4)
	lbu	t0,0(t3)
	lbu	t4,32(t3)
	addi	a5,a5,16
	addi	t3,t3,1
	vsrl.vi	v19,v3,1
	vand.vi	v3,v3,1
	vsrl.vi	v27,v2,4
	vand.vi	v19,v19,1
	vsll.vi	v3,v3,4
	vand.vi	v2,v2,15
	vsll.vi	v19,v19,4
	vor.vv	v2,v2,v3
	vor.vv	v3,v27,v19
	vwmacc.vx	v26,t0,v2
	vwmacc.vx	v24,t4,v3
	bne	s9,a5,.L7
	j	.L42
.L3:
	.cfi_def_cfa_offset 0
	.cfi_restore 8
	.cfi_restore 9
	.cfi_restore 18
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	vse32.v	v22,0(a4)
	addi	a5,a4,64
	addi	a4,a4,128
	beq	a0,a5,.L31
	vse32.v	v22,0(a5)
	bne	a0,a4,.L3
	ret
.L9:
	.cfi_def_cfa_offset 144
	.cfi_offset 8, -8
	.cfi_offset 9, -16
	.cfi_offset 18, -24
	.cfi_offset 19, -32
	.cfi_offset 20, -40
	.cfi_offset 21, -48
	.cfi_offset 22, -56
	.cfi_offset 23, -64
	.cfi_offset 24, -72
	.cfi_offset 25, -80
	.cfi_offset 26, -88
	.cfi_offset 27, -96
	li	s10,0
	addi	s9,t6,1024
	addi	s4,a6,3
	mv	t0,s0
	mv	t4,t5
	sd	s0,8(sp)
.L13:
	vsetvli	zero,zero,e8,mf2,ta,ma
	addi	a5,t0,-256
	vmv1r.v	v19,v8
	mv	t3,t4
	vmv1r.v	v24,v8
.L12:
	vle8.v	v1,0(a5)
	add	s0,s9,a5
	vle8.v	v2,0(s0)
	lbu	s11,0(t3)
	lbu	s0,32(t3)
	addi	a5,a5,16
	addi	t3,t3,1
	vsrl.vx	v3,v1,t1
	vsrl.vx	v1,v1,s4
	vand.vi	v25,v2,15
	vand.vi	v3,v3,1
	vand.vi	v1,v1,1
	vsrl.vi	v2,v2,4
	vsll.vi	v3,v3,4
	vsll.vi	v1,v1,4
	vor.vv	v3,v25,v3
	vor.vv	v2,v2,v1
	vwmacc.vx	v24,s11,v3
	vwmacc.vx	v19,s0,v2
	bne	a5,t0,.L12
	vsetvli	zero,zero,e16,m1,ta,ma
	addi	t4,t4,16
	vwmacc.vv	v4,v17,v24
	addi	t0,a5,256
	vwmacc.vv	v4,v18,v19
	beq	s10,t2,.L43
	li	s10,16
	j	.L13
.L43:
	ld	s0,8(sp)
	addiw	s3,s3,1
	addi	t1,t1,4
	addi	a6,a6,4
	addi	t5,t5,128
	addi	a3,a3,16
	addi	a0,a0,64
	bne	t6,s6,.L21
	vsetvli	zero,zero,e32,m2,ta,ma
	ld	a3,16(sp)
	vle16.v	v1,0(a2)
	addi	s0,s0,2047
	vfcvt.f.x.v	v4,v4
	addi	a2,a2,2047
	vfcvt.f.x.v	v6,v6
	addi	a5,a1,292
	vsetvli	zero,zero,e16,m1,ta,ma
	addi	s0,s0,769
	addi	a2,a2,769
	vfwcvt.f.f.v	v2,v1
	vsetvli	zero,zero,e32,m2,ta,ma
	vfmul.vf	v2,v2,fa5
	vfmacc.vv	v10,v4,v2
	vfnmsac.vv	v10,v12,v6
	beq	a3,a5,.L44
	vsetvli	zero,zero,e16,m1,ta,ma
	j	.L15
.L4:
	vle8.v	v24,0(s8)
	addi	t3,a0,16
	vle8.v	v26,0(s7)
	addi	a5,a0,32
	vle8.v	v19,0(a0)
	vle8.v	v17,0(t3)
	vle8.v	v2,0(s5)
	vle8.v	v18,0(a5)
	vand.vx	v29,v24,s1
	vand.vx	v28,v26,s1
	vand.vx	v24,v24,s2
	vsrl.vi	v31,v19,4
	vsrl.vi	v30,v17,4
	vsrl.vi	v29,v29,2
	vand.vi	v17,v17,15
	vand.vi	v19,v19,15
	vsrl.vi	v28,v28,2
	vand.vx	v26,v26,s2
	vor.vv	v1,v16,v3
	vor.vv	v19,v24,v19
	vor.vv	v27,v9,v25
	vor.vv	v26,v26,v17
	vor.vv	v29,v29,v31
	vor.vv	v28,v28,v30
	vand.vx	v3,v2,s1
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v25,v27
	vzext.vf2	v17,v26
	vzext.vf2	v27,v1
	vzext.vf2	v24,v29
	vzext.vf2	v1,v19
	vzext.vf2	v19,v28
	vsetvli	zero,zero,e8,mf2,ta,ma
	vand.vi	v26,v18,15
	vsrl.vi	v3,v3,2
	vand.vx	v2,v2,s2
	vsrl.vi	v18,v18,4
	j	.L5
.L21:
	li	t6,1024
	vsetvli	zero,zero,e8,mf2,ta,ma
	j	.L14
.L44:
	mv	a6,a4
	vsetvli	zero,zero,e16,m1,ta,ma
	ld	a4,40(sp)
	ld	a0,24(sp)
	mv	a1,a3
	mv	a3,a7
	ld	a7,32(sp)
	vse32.v	v10,0(a4)
	addi	a4,a4,64
	beq	a0,a4,.L28
	vmv2r.v	v10,v22
	add	a3,a3,a7
	j	.L17
.L28:
	ld	s0,136(sp)
	.cfi_restore 8
	ld	s11,48(sp)
	.cfi_restore 27
	ld	s10,56(sp)
	.cfi_restore 26
	ld	s9,64(sp)
	.cfi_restore 25
	ld	s8,72(sp)
	.cfi_restore 24
	ld	s7,80(sp)
	.cfi_restore 23
	ld	s6,88(sp)
	.cfi_restore 22
	ld	s5,96(sp)
	.cfi_restore 21
	ld	s4,104(sp)
	.cfi_restore 20
	ld	s3,112(sp)
	.cfi_restore 19
	ld	s2,120(sp)
	.cfi_restore 18
	ld	s1,128(sp)
	.cfi_restore 9
	addi	sp,sp,144
	.cfi_def_cfa_offset 0
	jr	ra
.L31:
	ret
	.cfi_endproc
.LFE3:
	.size	cur_q5k, .-cur_q5k
	.align	1
	.globl	knest_q5k
	.type	knest_q5k, @function
knest_q5k:
.LFB4:
	.cfi_startproc
	srli	a5,a4,4
	beq	a5,zero,.L69
	vsetivli	zero,16,e32,m2,ta,ma
	slli	a5,a5,6
	vmv.v.i	v20,0
	srli	a0,a0,8
	mv	a4,a1
	add	a1,a1,a5
	vmv2r.v	v8,v20
	beq	a0,zero,.L47
	li	a5,292
	vmv.v.i	v18,0
	mul	a5,a0,a5
	li	a7,4096
	addi	a7,a7,-1280
	vsetvli	zero,zero,e16,m1,ta,ma
	vmv.v.i	v17,0
	addi	sp,sp,-160
	.cfi_def_cfa_offset 160
	mv	a6,a3
	sd	s11,64(sp)
	sd	s2,136(sp)
	sd	s0,152(sp)
	mv	t1,a2
	sd	s10,72(sp)
	sd	s9,80(sp)
	mul	a0,a0,a7
	sd	s7,96(sp)
	sd	s6,104(sp)
	sd	s5,112(sp)
	sd	s4,120(sp)
	sd	s3,128(sp)
	sd	s1,144(sp)
	.cfi_offset 27, -96
	.cfi_offset 18, -24
	.cfi_offset 8, -8
	.cfi_offset 26, -88
	.cfi_offset 25, -80
	.cfi_offset 23, -64
	.cfi_offset 22, -56
	.cfi_offset 21, -48
	.cfi_offset 20, -40
	.cfi_offset 19, -32
	.cfi_offset 9, -16
	li	s11,-64
	li	s0,1
	li	a3,16
	li	s2,16
	add	a2,a6,a5
.L57:
	mv	a5,t1
	mv	s10,a6
	sd	t1,40(sp)
	sd	a4,48(sp)
	sd	a2,32(sp)
	sd	a0,56(sp)
.L55:
	addi	a2,a5,32
	vmv2r.v	v6,v18
	vle16.v	v1,0(a2)
	addi	a2,a5,192
	vle8.v	v2,0(a2)
	flw	fa5,0(s10)
	vmv2r.v	v4,v18
	li	a4,48
	li	s1,0
	addi	s5,a5,208
	mv	s7,a1
	vfwcvt.f.f.v	v12,v1
	addi	a0,s10,260
	vsetvli	zero,zero,e8,mf2,ta,ma
	addi	t5,a5,80
	vand.vx	v16,v2,s11
	li	t6,2
	vand.vi	v15,v2,3
	li	t2,68
	vand.vi	v14,v2,12
	li	a2,0
	vsetvli	zero,zero,e32,m2,ta,ma
	addi	s4,a5,224
	vfmul.vf	v12,v12,fa5
	addi	s6,a5,240
	vsetvli	zero,zero,e8,mf2,ta,ma
	addi	s9,s10,-64
	vsrl.vi	v16,v16,2
	addi	s3,a5,512
	vsll.vi	v15,v15,4
	sd	a5,24(sp)
	vsll.vi	v14,v14,2
	mv	a1,a6
	vand.vx	v2,v2,a4
.L54:
	addi	a5,t5,-16
	vle8.v	v22,0(s5)
	vle8.v	v1,0(a5)
	addi	a4,t5,16
	vand.vi	v23,v1,15
	vsrl.vi	v10,v1,4
	bne	s1,zero,.L48
	vle8.v	v1,0(s4)
	addi	a5,t5,32
	vle8.v	v26,0(a4)
	vle8.v	v11,0(t5)
	vand.vi	v24,v22,3
	vand.vi	v27,v22,12
	vle8.v	v3,0(s6)
	vand.vi	v29,v1,3
	vand.vi	v1,v1,12
	vand.vi	v28,v26,15
	vand.vi	v30,v11,15
	vsll.vi	v22,v1,2
	vsrl.vi	v11,v11,4
	vsrl.vi	v1,v26,4
	vsll.vi	v24,v24,4
	vsll.vi	v27,v27,2
	vsll.vi	v29,v29,4
	vle8.v	v25,0(a5)
	vor.vv	v27,v27,v11
	vor.vv	v26,v15,v23
	vor.vv	v10,v14,v10
	vor.vv	v24,v24,v30
	vor.vv	v29,v29,v28
	vor.vv	v28,v22,v1
	vand.vi	v11,v3,12
	vand.vi	v1,v3,3
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v23,v26
	vzext.vf2	v3,v24
	vzext.vf2	v26,v10
	vzext.vf2	v24,v27
	vzext.vf2	v10,v29
	vzext.vf2	v22,v28
	vsetvli	zero,zero,e8,mf2,ta,ma
	vand.vi	v27,v25,15
	vsll.vi	v1,v1,4
	vsll.vi	v11,v11,2
	vsrl.vi	v25,v25,4
.L49:
	lhu	a4,0(a0)
	vor.vv	v11,v11,v25
	lhu	a5,2(a0)
	vor.vv	v1,v1,v27
	vsetvli	zero,zero,e16,m1,ta,ma
	lhu	a7,4(a0)
	lhu	a6,6(a0)
	vzext.vf2	v25,v11
	addw	a5,a5,a4
	vzext.vf2	v11,v1
	vwmacc.vx	v6,a5,v26
	lhu	a4,8(a0)
	addw	a6,a6,a7
	lhu	a7,10(a0)
	lhu	a5,12(a0)
	vwmacc.vx	v6,a6,v24
	lhu	a6,14(a0)
	addw	a4,a4,a7
	slliw	t3,s1,2
	addw	a5,a5,a6
	vwmacc.vx	v6,a4,v22
	addiw	t1,t6,-1
	li	t0,0
	addi	t4,a2,512
	sd	s1,16(sp)
	vwmacc.vx	v6,a5,v25
	sllw	t3,s0,t3
	sllw	t1,s0,t1
	add	a6,s9,t2
	mv	a7,s3
	sd	s3,8(sp)
	mv	s1,a2
.L51:
	vsetvli	zero,zero,e8,mf2,ta,mu
	addi	a5,a7,-256
	vmv1r.v	v22,v17
	mv	a4,a6
	vmv1r.v	v24,v17
	sd	a6,0(sp)
	vmv1r.v	v25,v3
	vmv1r.v	v26,v2
.L50:
	vle8.v	v2,0(a5)
	add	a2,t4,a5
	vle8.v	v1,0(a2)
	lbu	a6,0(a4)
	lbu	a2,32(a4)
	addi	a5,a5,16
	addi	a4,a4,1
	vand.vx	v0,v2,t3
	vand.vi	v3,v1,15
	vand.vx	v2,v2,t1
	vmsne.vi	v0,v0,0
	vsrl.vi	v1,v1,4
	vadd.vx	v3,v3,a3,v0.t
	vmsne.vi	v0,v2,0
	vwmacc.vx	v24,a6,v3
	vadd.vx	v1,v1,a3,v0.t
	vwmacc.vx	v22,a2,v1
	bne	a5,a7,.L50
	vsetvli	zero,zero,e16,m1,ta,ma
	ld	a6,0(sp)
	vwmacc.vv	v4,v23,v24
	addi	a7,a5,256
	vmv1r.v	v3,v25
	vmv1r.v	v2,v26
	addi	a6,a6,16
	vwmacc.vv	v4,v25,v22
	bne	t0,s2,.L58
	addi	a2,s1,1024
	ld	s1,16(sp)
	addiw	t1,t6,1
	ld	a4,8(sp)
	sllw	t1,s0,t1
	li	a7,0
	sllw	t3,s0,t6
	add	t4,t2,s10
	sd	t0,8(sp)
.L53:
	vsetvli	zero,zero,e8,mf2,ta,mu
	addi	a5,a4,-256
	vmv1r.v	v22,v17
	mv	a6,t4
	vmv1r.v	v23,v17
	sd	t4,0(sp)
	vmv1r.v	v24,v2
.L52:
	vle8.v	v2,0(a5)
	add	t4,a5,a2
	vle8.v	v1,0(t4)
	lbu	t0,0(a6)
	lbu	t4,32(a6)
	addi	a5,a5,16
	addi	a6,a6,1
	vand.vx	v0,v2,t3
	vand.vi	v3,v1,15
	vand.vx	v2,v2,t1
	vmsne.vi	v0,v0,0
	vsrl.vi	v1,v1,4
	vadd.vx	v3,v3,a3,v0.t
	vmsne.vi	v0,v2,0
	vwmacc.vx	v23,t0,v3
	vadd.vx	v1,v1,a3,v0.t
	vwmacc.vx	v22,t4,v1
	bne	a5,a4,.L52
	vsetvli	zero,zero,e16,m1,ta,ma
	ld	t4,0(sp)
	vwmacc.vv	v4,v10,v23
	addi	a4,a5,256
	vmv1r.v	v2,v24
	addi	t4,t4,16
	vwmacc.vv	v4,v11,v22
	beq	a7,s2,.L78
	ld	a7,8(sp)
	j	.L53
.L47:
	.cfi_def_cfa_offset 0
	.cfi_restore 8
	.cfi_restore 9
	.cfi_restore 18
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 23
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 27
	vse32.v	v20,0(a4)
	addi	a5,a4,64
	addi	a4,a4,128
	beq	a5,a1,.L69
	vse32.v	v20,0(a5)
	bne	a4,a1,.L47
	ret
.L58:
	.cfi_def_cfa_offset 160
	.cfi_offset 8, -8
	.cfi_offset 9, -16
	.cfi_offset 18, -24
	.cfi_offset 19, -32
	.cfi_offset 20, -40
	.cfi_offset 21, -48
	.cfi_offset 22, -56
	.cfi_offset 23, -64
	.cfi_offset 25, -80
	.cfi_offset 26, -88
	.cfi_offset 27, -96
	li	t0,16
	j	.L51
.L78:
	addi	a5,s1,1
	li	a4,2
	addi	t2,t2,128
	addiw	t6,t6,4
	addi	a0,a0,16
	addi	t5,t5,64
	li	s1,1
	bne	a5,a4,.L74
	ld	a5,24(sp)
	vsetvli	zero,zero,e32,m2,ta,ma
	vfcvt.f.x.v	v4,v4
	ld	a4,32(sp)
	vfcvt.f.x.v	v6,v6
	addi	s10,s10,292
	vle16.v	v1,0(a5)
	addi	a5,a5,2047
	vsetvli	zero,zero,e16,m1,ta,ma
	mv	a6,a1
	addi	a5,a5,769
	mv	a1,s7
	vfwcvt.f.f.v	v2,v1
	vsetvli	zero,zero,e32,m2,ta,ma
	vfmul.vf	v2,v2,fa5
	vfmacc.vv	v8,v4,v2
	vfnmsac.vv	v8,v12,v6
	beq	s10,a4,.L79
	vsetvli	zero,zero,e16,m1,ta,ma
	j	.L55
.L48:
	vle8.v	v27,0(s4)
	addi	a5,t5,32
	vle8.v	v3,0(a4)
	vle8.v	v11,0(t5)
	vand.vx	v29,v22,s11
	vle8.v	v1,0(s6)
	vle8.v	v25,0(a5)
	li	a5,48
	vand.vx	v28,v27,s11
	vsrl.vi	v26,v3,4
	vsrl.vi	v30,v11,4
	vand.vi	v3,v3,15
	vand.vi	v11,v11,15
	vand.vx	v22,v22,a5
	vsrl.vi	v29,v29,2
	vsrl.vi	v28,v28,2
	vand.vx	v27,v27,a5
	vor.vv	v22,v22,v11
	vor.vv	v24,v2,v23
	vor.vv	v27,v27,v3
	vor.vv	v10,v16,v10
	vor.vv	v28,v28,v26
	vor.vv	v29,v29,v30
	vand.vx	v11,v1,s11
	vsetvli	zero,zero,e16,m1,ta,ma
	vzext.vf2	v23,v24
	vzext.vf2	v26,v10
	vzext.vf2	v3,v22
	vzext.vf2	v10,v27
	vzext.vf2	v24,v29
	vzext.vf2	v22,v28
	vsetvli	zero,zero,e8,mf2,ta,ma
	vand.vi	v27,v25,15
	vsrl.vi	v11,v11,2
	vand.vx	v1,v1,a5
	vsrl.vi	v25,v25,4
	j	.L49
.L74:
	vsetvli	zero,zero,e8,mf2,ta,ma
	j	.L54
.L79:
	ld	a4,48(sp)
	vsetvli	zero,zero,e16,m1,ta,ma
	ld	t1,40(sp)
	ld	a0,56(sp)
	vse32.v	v8,0(a4)
	addi	a4,a4,64
	mv	a2,s10
	beq	a4,s7,.L66
	vmv2r.v	v8,v20
	add	t1,t1,a0
	j	.L57
.L66:
	ld	s0,152(sp)
	.cfi_restore 8
	ld	s11,64(sp)
	.cfi_restore 27
	ld	s10,72(sp)
	.cfi_restore 26
	ld	s9,80(sp)
	.cfi_restore 25
	ld	s7,96(sp)
	.cfi_restore 23
	ld	s6,104(sp)
	.cfi_restore 22
	ld	s5,112(sp)
	.cfi_restore 21
	ld	s4,120(sp)
	.cfi_restore 20
	ld	s3,128(sp)
	.cfi_restore 19
	ld	s2,136(sp)
	.cfi_restore 18
	ld	s1,144(sp)
	.cfi_restore 9
	addi	sp,sp,160
	.cfi_def_cfa_offset 0
	jr	ra
.L69:
	ret
	.cfi_endproc
.LFE4:
	.size	knest_q5k, .-knest_q5k
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
