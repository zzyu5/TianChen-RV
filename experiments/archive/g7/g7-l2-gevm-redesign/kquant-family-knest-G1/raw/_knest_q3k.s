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
