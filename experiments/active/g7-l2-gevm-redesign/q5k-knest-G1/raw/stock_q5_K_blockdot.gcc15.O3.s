	.file	"stock_q5_K_blockdot.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC0:
	.string	"experiments/active/g7-l2-gevm-redesign/q5k-knest-G1/raw/stock_q5_K_blockdot.c"
	.align	3
.LC1:
	.string	"n % QK_K == 0"
	.text
	.align	1
	.globl	stock_q5_K
	.type	stock_q5_K, @function
stock_q5_K:
.LFB1:
	.cfi_startproc
	csrr	t0,vlenb
	addi	sp,sp,-112
	.cfi_def_cfa_offset 112
	slli	t1,t0,4
	sd	s9,24(sp)
	sd	s8,32(sp)
	sd	s7,40(sp)
	sd	s6,48(sp)
	sd	s5,56(sp)
	sd	s4,64(sp)
	sd	s3,72(sp)
	sd	s2,80(sp)
	sd	s1,88(sp)
	sd	s0,96(sp)
	sd	ra,104(sp)
	sub	sp,sp,t1
	.cfi_escape 0xf,0xb,0x72,0,0x92,0xa2,0x38,0,0x40,0x1e,0x23,0x70,0x22
	.cfi_offset 25, -88
	.cfi_offset 24, -80
	.cfi_offset 23, -72
	.cfi_offset 22, -64
	.cfi_offset 21, -56
	.cfi_offset 20, -48
	.cfi_offset 19, -40
	.cfi_offset 18, -32
	.cfi_offset 9, -24
	.cfi_offset 8, -16
	.cfi_offset 1, -8
	andi	t1,a0,0xff
	bne	t1,zero,.L10
	li	a4,255
	ble	a0,a4,.L5
	fmv.s.x	fa1,zero
	vsetivli	zero,8,e32,m2,ta,ma
	csrr	a4,vlenb
	vmv.s.x	v1,zero
	slli	a4,a4,4
	addi	a4,a4,8
	sraiw	t0,a0,8
	li	t5,808464384
	addi	t5,t5,48
	li	t4,252645376
	addi	t4,t4,-241
	li	t3,1061109760
	addi	t3,t3,-193
	add	s1,sp,a4
	fmv.s	fa2,fa1
	addi	a5,a5,260
	addi	a3,a3,4
	li	t6,4
	li	a4,32
	li	a0,16
	li	s0,64
	li	t2,-128
	j	.L4
.L9:
	vsetivli	zero,8,e16,m1,ta,ma
.L4:
	vlse16.v	v2,0(a5),t6
	addi	a2,a3,12
	vsetvli	zero,a4,e8,m2,ta,mu
	csrr	a6,vlenb
	vle8.v	v4,0(a2)
	addi	a2,a3,44
	vle8.v	v10,0(a2)
	addi	a2,a3,76
	vle8.v	v8,0(a2)
	addi	a2,a3,108
	vle8.v	v6,0(a2)
	addi	a2,a3,140
	vle8.v	v16,0(a2)
	li	a2,12
	mul	a2,a2,a6
	lw	a7,8(a3)
	vand.vi	v14,v4,1
	addi	s6,a5,-64
	vand.vx	v18,v4,a0
	addi	s5,a5,-160
	and	s4,a7,t4
	vand.vx	v24,v4,s0
	vmsne.vi	v0,v14,0
	srliw	a7,a7,4
	and	a7,a7,t4
	vmsne.vi	v3,v24,0
	csrr	s9,vlenb
	vand.vi	v24,v4,2
	addiw	t1,t1,1
	addi	a3,a3,176
	add	a2,a2,sp
	vs2r.v	v16,0(a2)
	addi	a2,a5,-256
	vand.vi	v16,v10,15
	vsrl.vi	v10,v10,4
	vadd.vx	v16,v16,a0,v0.t
	vle8.v	v30,0(a2)
	addi	a2,a5,-192
	vle8.v	v28,0(a2)
	addi	a2,a5,-128
	vle8.v	v12,0(a2)
	slli	a2,a6,3
	add	a2,a2,sp
	ld	a6,-176(a3)
	srli	s8,a6,34
	vs2r.v	v12,0(a2)
	addi	a2,a5,-224
	vand.vi	v12,v4,4
	and	s8,s8,t5
	or	a7,s8,a7
	csrr	s8,vlenb
	vmsne.vi	v0,v12,0
	slli	s8,s8,2
	add	s8,s8,sp
	vand.vi	v12,v6,15
	vle8.v	v14,0(a2)
	and	s2,a6,t3
	andi	s7,s2,0xff
	srli	a2,a6,2
	srai	a6,a6,32
	and	a6,a6,t3
	and	a2,a2,t5
	vs2r.v	v14,0(sp)
	or	a2,a2,s4
	vand.vi	v14,v8,15
	srli	s3,s2,16
	andi	s3,s3,0xff
	addi	s4,a5,-96
	vadd.vx	v14,v14,a0,v0.t
	vmsne.vi	v0,v18,0
	vmsne.vi	v18,v24,0
	vwmul.vv	v20,v14,v28
	vadd.vx	v12,v12,a0,v0.t
	vwmul.vv	v24,v16,v30
	vmv1r.v	v0,v18
	vs4r.v	v20,0(s8)
	csrr	s8,vlenb
	slli	s8,s8,3
	add	s8,s8,sp
	vadd.vx	v10,v10,a0,v0.t
	vl2re8.v	v14,0(s8)
	csrr	s8,vlenb
	slli	s8,s8,3
	add	s8,s8,sp
	vwmul.vv	v28,v12,v14
	vsrl.vi	v12,v8,4
	vl2re8.v	v8,0(sp)
	vsetvli	zero,zero,e16,m4,ta,ma
	vs4r.v	v28,0(s8)
	addi	s8,a5,-32
	vwmul.vx	v16,v24,s7
	srli	s7,s2,8
	andi	s7,s7,0xff
	vle8.v	v14,0(s6)
	andi	s6,a2,0xff
	vsetvli	zero,zero,e8,m2,ta,ma
	vwmul.vv	v24,v10,v8
	vsrl.vi	v6,v6,4
	vsetvli	zero,zero,e32,m8,ta,ma
	vle8.v	v10,0(s5)
	srli	s5,a2,16
	vredsum.vs	v8,v16,v1
	andi	s5,s5,0xff
	vsetvli	zero,zero,e16,m4,ta,ma
	vwmul.vx	v16,v24,s7
	addi	s7,a5,2
	vsetivli	zero,8,e16,m1,ta,ma
	addi	a5,a5,292
	vs2r.v	v10,0(sp)
	vlse16.v	v9,0(s7),t6
	csrr	s7,vlenb
	slli	s7,s7,4
	vsetvli	zero,a4,e32,m8,ta,ma
	add	s7,s7,sp
	vredsum.vs	v8,v16,v8
	sw	a7,12(s7)
	vsetvli	zero,zero,e8,m2,ta,mu
	csrr	a7,vlenb
	vmv1r.v	v0,v3
	slli	a7,a7,4
	vand.vi	v20,v4,8
	add	a7,a7,sp
	vand.vx	v10,v4,a4
	sw	a6,8(a7)
	vand.vx	v4,v4,t2
	csrr	a6,vlenb
	slli	a6,a6,4
	add	a6,a6,sp
	sw	a2,4(a6)
	csrr	a7,vlenb
	li	a6,12
	mul	a6,a6,a7
	srli	s7,s2,24
	andi	s7,s7,0xff
	add	a6,a6,sp
	vl2re8.v	v16,0(a6)
	slli	a6,a7,4
	add	a6,a6,sp
	sw	s2,0(a6)
	li	a6,12
	mul	a6,a6,s9
	lhu	s2,-180(a3)
	srli	a7,a2,8
	vand.vi	v18,v16,15
	andi	a7,a7,0xff
	fcvt.s.wu	fa5,s2
	slli	s2,s9,2
	vadd.vx	v18,v18,a0,v0.t
	add	s2,s2,sp
	vmsne.vi	v0,v20,0
	srli	a2,a2,24
	andi	a2,a2,0xff
	vwmul.vv	v24,v18,v14
	vadd.vx	v12,v12,a0,v0.t
	add	a6,a6,sp
	vs4r.v	v24,0(a6)
	vl4re16.v	v20,0(s2)
	lhu	s2,-178(a3)
	vmv2r.v	v18,v12
	flw	fa3,-552(a5)
	vsetivli	zero,0,e32,m1,ta,ma
	vmv.x.s	a6,v8
	fcvt.s.wu	fa4,s2
	vsetvli	zero,a4,e16,m4,ta,ma
	slli	s2,s9,3
	add	s2,s2,sp
	vwmul.vx	v24,v20,s3
	fmul.s	fa4,fa4,fa3
	vsetvli	zero,zero,e8,m2,ta,mu
	fmul.s	fa5,fa5,fa3
	vmsne.vi	v0,v10,0
	vsrl.vi	v10,v16,4
	vle8.v	v14,0(s4)
	vadd.vx	v6,v6,a0,v0.t
	vmsne.vi	v0,v4,0
	vle8.v	v12,0(s8)
	vl2re8.v	v4,0(sp)
	vadd.vx	v10,v10,a0,v0.t
	vsetivli	zero,8,e8,mf2,ta,ma
	vle8.v	v0,0(s1)
	vsetvli	zero,a4,e32,m8,ta,ma
	vredsum.vs	v3,v24,v1
	vl4re16.v	v28,0(s2)
	li	s2,12
	mul	s2,s2,s9
	vsetvli	zero,zero,e8,m2,ta,ma
	vwmul.vv	v24,v6,v14
	vwmul.vv	v20,v18,v4
	vwmul.vv	v4,v10,v12
	vsetivli	zero,8,e16,m1,ta,ma
	vadd.vv	v2,v2,v9
	vsetvli	zero,a4,e16,m4,ta,ma
	vwmul.vx	v8,v28,s6
	vs4r.v	v24,0(sp)
	vmv4r.v	v28,v20
	add	s2,s2,sp
	vl4re16.v	v24,0(s2)
	vwmul.vx	v16,v28,s7
	vsetivli	zero,8,e16,m1,ta,ma
	vzext.vf2	v28,v0
	vsetvli	zero,a4,e32,m8,ta,ma
	vredsum.vs	v29,v8,v1
	vsetvli	zero,zero,e16,m4,ta,ma
	vwmul.vx	v8,v24,s5
	vl4re16.v	v24,0(sp)
	vsetivli	zero,8,e16,m1,ta,ma
	vwmul.vv	v30,v2,v28
	vsetvli	zero,a4,e32,m8,ta,ma
	vredsum.vs	v2,v16,v3
	vsetvli	zero,zero,e16,m4,ta,ma
	vwmul.vx	v16,v24,a7
	vsetvli	zero,zero,e32,m8,ta,ma
	vredsum.vs	v3,v8,v1
	vsetvli	zero,zero,e16,m4,ta,ma
	vwmul.vx	v8,v4,a2
	vsetivli	zero,8,e32,m2,ta,ma
	vredsum.vs	v30,v30,v1
	vsetvli	zero,a4,e32,m8,ta,ma
	vredsum.vs	v16,v16,v29
	vmv.x.s	a2,v2
	vredsum.vs	v8,v8,v3
	addw	a2,a6,a2
	vmv.x.s	a7,v30
	vmv.x.s	a6,v16
	fcvt.s.w	fa3,a7
	addw	a2,a2,a6
	vmv.x.s	a6,v8
	fnmsub.s	fa2,fa3,fa4,fa2
	addw	a2,a2,a6
	fcvt.s.w	fa4,a2
	fmadd.s	fa1,fa4,fa5,fa1
	bgt	t0,t1,.L9
	fadd.s	fa2,fa2,fa1
.L3:
	csrr	t0,vlenb
	fsw	fa2,0(a1)
	slli	t1,t0,4
	add	sp,sp,t1
	.cfi_remember_state
	.cfi_def_cfa_offset 112
	ld	ra,104(sp)
	.cfi_restore 1
	ld	s9,24(sp)
	.cfi_restore 25
	ld	s8,32(sp)
	.cfi_restore 24
	ld	s7,40(sp)
	.cfi_restore 23
	ld	s6,48(sp)
	.cfi_restore 22
	ld	s5,56(sp)
	.cfi_restore 21
	ld	s4,64(sp)
	.cfi_restore 20
	ld	s3,72(sp)
	.cfi_restore 19
	ld	s2,80(sp)
	.cfi_restore 18
	ld	s1,88(sp)
	.cfi_restore 9
	ld	s0,96(sp)
	.cfi_restore 8
	addi	sp,sp,112
	.cfi_def_cfa_offset 0
	jr	ra
.L5:
	.cfi_restore_state
	fmv.s.x	fa2,zero
	j	.L3
.L10:
	lla	a3,.LANCHOR0
	li	a2,25
	lla	a1,.LC0
	lla	a0,.LC1
	call	__assert_fail@plt
	.cfi_endproc
.LFE1:
	.size	stock_q5_K, .-stock_q5_K
	.section	.rodata
	.align	3
	.set	.LANCHOR0,. + 0
	.type	__PRETTY_FUNCTION__.0, @object
	.size	__PRETTY_FUNCTION__.0, 11
__PRETTY_FUNCTION__.0:
	.string	"stock_q5_K"
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
