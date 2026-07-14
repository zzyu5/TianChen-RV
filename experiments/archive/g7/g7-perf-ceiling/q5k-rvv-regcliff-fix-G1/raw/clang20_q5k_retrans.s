	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"strip_body_regpressure.c"
	.text
	.globl	strip_body                      # -- Begin function strip_body
	.p2align	1
	.type	strip_body,@function
strip_body:                             # @strip_body
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -624
	.cfi_def_cfa_offset 624
	sd	ra, 616(sp)                     # 8-byte Folded Spill
	sd	s0, 608(sp)                     # 8-byte Folded Spill
	sd	s1, 600(sp)                     # 8-byte Folded Spill
	sd	s2, 592(sp)                     # 8-byte Folded Spill
	sd	s3, 584(sp)                     # 8-byte Folded Spill
	sd	s4, 576(sp)                     # 8-byte Folded Spill
	sd	s5, 568(sp)                     # 8-byte Folded Spill
	sd	s6, 560(sp)                     # 8-byte Folded Spill
	sd	s7, 552(sp)                     # 8-byte Folded Spill
	sd	s8, 544(sp)                     # 8-byte Folded Spill
	sd	s9, 536(sp)                     # 8-byte Folded Spill
	sd	s10, 528(sp)                    # 8-byte Folded Spill
	sd	s11, 520(sp)                    # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	.cfi_offset s6, -64
	.cfi_offset s7, -72
	.cfi_offset s8, -80
	.cfi_offset s9, -88
	.cfi_offset s10, -96
	.cfi_offset s11, -104
	csrr	a7, vlenb
	li	t0, 10
	mul	a7, a7, t0
	sub	sp, sp, a7
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xf0, 0x04, 0x22, 0x11, 0x0a, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 624 + 10 * vlenb
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	sd	a3, 112(sp)                     # 8-byte Folded Spill
	beqz	a3, .LBB0_20
# %bb.1:
	mv	s0, a0
	sd	a2, 32(sp)                      # 8-byte Folded Spill
	sd	a4, 40(sp)                      # 8-byte Folded Spill
	li	a4, 0
	vmv.v.i	v12, 0
	slliw	a7, a5, 3
	addi	s9, sp, 488
	addi	s4, sp, 424
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	addi	s2, sp, 456
	addi	t3, sp, 392
	addi	s5, sp, 408
	slliw	a0, a5, 4
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	s10, sp, 280
	addi	s11, sp, 312
	addi	ra, sp, 344
	addi	a0, a6, 128
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	addi	a0, a1, 80
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	addi	a0, a6, 3
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	addi	a0, a6, 131
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	li	a0, 11
	addi	s7, sp, 440
	addi	a5, sp, 376
	addi	t2, sp, 248
	li	t5, 16
	li	t4, 64
	vmv2r.v	v20, v8
	vmv2r.v	v16, v8
	vmv2r.v	v14, v8
	add	a3, a7, s0
	slli	a2, a0, 8
	addi	a0, a3, 256
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 512
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 768
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 1024
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	addi	a0, a7, 64
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	addiw	a0, a7, 192
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	addiw	a0, a7, 208
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a7, 224
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	sd	a7, 24(sp)                      # 8-byte Folded Spill
	addiw	a0, a7, 240
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	sd	a1, 200(sp)                     # 8-byte Folded Spill
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	sd	s0, 96(sp)                      # 8-byte Folded Spill
	sd	a2, 80(sp)                      # 8-byte Folded Spill
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_10 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_14 Depth 3
                                        #       Child Loop BB0_16 Depth 3
	addi	a0, sp, 512
	vs2r.v	v20, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 512
	vs2r.v	v16, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 512
	vs2r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a3, 6
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 512
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	li	a7, 0
	li	a0, 1168
	mul	t1, a4, a0
	sd	a4, 120(sp)                     # 8-byte Folded Spill
	mul	t0, a4, a2
	li	a0, 1
	add	t1, t1, a1
	add	t0, t0, s0
	addi	a1, t1, 144
	sd	a1, 192(sp)                     # 8-byte Folded Spill
	flw	fa3, 0(t1)
	flw	fa2, 4(t1)
	flw	fa5, 8(t1)
	flw	fa4, 12(t1)
	ld	a2, 72(sp)                      # 8-byte Folded Reload
	add	a2, a2, t0
	addi	s1, t1, 1040
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v26, (a2)
	ld	a1, 64(sp)                      # 8-byte Folded Reload
	add	a1, a1, t0
	sd	a1, 176(sp)                     # 8-byte Folded Spill
	ld	a1, 56(sp)                      # 8-byte Folded Reload
	add	a1, a1, t0
	sd	a1, 168(sp)                     # 8-byte Folded Spill
	ld	a1, 48(sp)                      # 8-byte Folded Reload
	add	a1, a1, t0
	sd	a1, 160(sp)                     # 8-byte Folded Spill
	vand.vi	v8, v26, 3
	vand.vi	v9, v26, 12
	vsll.vi	v8, v8, 4
	csrr	a1, vlenb
	slli	a2, a1, 3
	add	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 512
	vs1r.v	v8, (a1)                        # Unknown-size Folded Spill
	vsll.vi	v8, v9, 2
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	addi	a1, a1, 512
	vs1r.v	v8, (a1)                        # Unknown-size Folded Spill
	vmv2r.v	v28, v12
	vmv2r.v	v24, v12
	vmv2r.v	v22, v12
	vmv2r.v	v20, v12
	sd	s1, 184(sp)                     # 8-byte Folded Spill
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_10 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_14 Depth 3
                                        #       Child Loop BB0_16 Depth 3
	slli	a2, a7, 6
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	addw	a3, a3, a2
	add	a2, t0, a3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a2)
	andi	t6, a0, 1
	addiw	a4, a3, 16
	addiw	a2, a3, 32
	vand.vi	v9, v8, 15
	vsrl.vi	v8, v8, 4
	addiw	a0, a3, 48
	beqz	t6, .LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	csrr	a1, vlenb
	slli	a3, a1, 3
	add	a1, a1, a3
	add	a1, a1, sp
	addi	a1, a1, 512
	vl1r.v	v10, (a1)                       # Unknown-size Folded Reload
	vor.vv	v9, v10, v9
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	addi	a1, a1, 512
	vl1r.v	v10, (a1)                       # Unknown-size Folded Reload
	vor.vv	v8, v10, v8
	add	a4, a4, t0
	ld	a1, 176(sp)                     # 8-byte Folded Reload
	vle8.v	v10, (a1)
	add	a2, a2, t0
	ld	a1, 168(sp)                     # 8-byte Folded Reload
	vle8.v	v11, (a1)
	add	a0, a0, t0
	ld	a1, 160(sp)                     # 8-byte Folded Reload
	vle8.v	v14, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v9
	vzext.vf2	v9, v8
	vle8.v	v8, (a4)
	vle8.v	v16, (a2)
	vle8.v	v17, (a0)
	vse16.v	v15, (s7)
	vse16.v	v9, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v10, 3
	vand.vi	v10, v10, 12
	vand.vi	v15, v8, 15
	vsll.vi	v9, v9, 4
	vor.vv	v15, v9, v15
	vand.vi	v9, v11, 3
	vand.vi	v11, v11, 12
	vsrl.vi	v8, v8, 4
	vsll.vi	v10, v10, 2
	vor.vv	v8, v10, v8
	vand.vi	v10, v16, 15
	vsll.vi	v9, v9, 4
	vor.vv	v10, v9, v10
	vand.vi	v9, v14, 3
	vsrl.vi	v16, v16, 4
	vsll.vi	v11, v11, 2
	vor.vv	v11, v11, v16
	vand.vi	v16, v17, 15
	vsll.vi	v9, v9, 4
	vor.vv	v9, v9, v16
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v16, v15
	vse16.v	v16, (s2)
	vzext.vf2	v15, v8
	vse16.v	v15, (t3)
	vzext.vf2	v8, v10
	addi	a0, sp, 472
	vse16.v	v8, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v14, 12
	vsrl.vi	v10, v17, 4
	vsll.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	vse16.v	v14, (s5)
	j	.LBB0_6
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=2
	li	a1, 48
	vand.vx	v10, v26, a1
	li	a3, -64
	vand.vx	v11, v26, a3
	add	a4, a4, t0
	ld	s0, 176(sp)                     # 8-byte Folded Reload
	vle8.v	v14, (s0)
	add	a2, a2, t0
	ld	s0, 168(sp)                     # 8-byte Folded Reload
	vle8.v	v15, (s0)
	add	a0, a0, t0
	ld	s0, 160(sp)                     # 8-byte Folded Reload
	vle8.v	v16, (s0)
	vsrl.vi	v11, v11, 2
	vor.vv	v9, v10, v9
	vle8.v	v10, (a4)
	vle8.v	v17, (a2)
	vor.vv	v8, v11, v8
	vle8.v	v11, (a0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v9
	vzext.vf2	v9, v8
	vse16.v	v19, (s7)
	vse16.v	v9, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v14, a1
	vand.vx	v9, v14, a3
	vand.vi	v14, v10, 15
	vor.vv	v8, v8, v14
	vand.vx	v14, v15, a1
	vand.vx	v15, v15, a3
	vsrl.vi	v10, v10, 4
	vsrl.vi	v9, v9, 2
	vor.vv	v10, v9, v10
	vand.vi	v9, v17, 15
	vor.vv	v14, v14, v9
	vand.vx	v9, v16, a1
	vsrl.vi	v17, v17, 4
	vsrl.vi	v15, v15, 2
	vor.vv	v15, v15, v17
	vand.vi	v17, v11, 15
	vor.vv	v9, v9, v17
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v8
	vse16.v	v17, (s2)
	vzext.vf2	v8, v10
	vse16.v	v8, (t3)
	vzext.vf2	v8, v14
	addi	a0, sp, 472
	vse16.v	v8, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v16, a3
	vsrl.vi	v10, v11, 4
	vsrl.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v15
	vse16.v	v11, (s5)
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e8, mf2, ta, ma
	vor.vv	v8, v8, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v8
	vse16.v	v10, (s4)
	vle16.v	v8, (a5)
	vzext.vf2	v10, v9
	vse16.v	v10, (s9)
	slli	s3, a7, 6
	beqz	t6, .LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=2
	lh	s6, 1040(t1)
	lh	a1, 1042(t1)
	lh	a2, 1044(t1)
	lh	s8, 1046(t1)
	lh	a4, 1048(t1)
	lh	s0, 1050(t1)
	lh	a0, 1052(t1)
	lh	a3, 1054(t1)
	add	a4, a4, s6
	add	a1, a1, s0
	add	a0, a0, a2
	vmv2r.v	v10, v12
	vwmacc.vx	v10, a4, v8
	vse32.v	v10, (t2)
	vmv2r.v	v10, v12
	vwmacc.vx	v10, a1, v8
	vse32.v	v10, (s10)
	vmv2r.v	v10, v12
	vwmacc.vx	v10, a0, v8
	vse32.v	v10, (s11)
	vmv2r.v	v10, v12
	add	a3, a3, s8
	vwmacc.vx	v10, a3, v8
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=2
	add	a2, s1, s3
	vle32.v	v10, (t2)
	lh	s6, 0(a2)
	lh	s8, 2(a2)
	lh	s0, 4(a2)
	lh	a1, 6(a2)
	lh	a0, 8(a2)
	lh	a3, 10(a2)
	lh	a4, 12(a2)
	lh	a2, 14(a2)
	add	a0, a0, s6
	vwmacc.vx	v10, a0, v8
	vse32.v	v10, (t2)
	vle32.v	v10, (s10)
	add	a3, a3, s8
	vwmacc.vx	v10, a3, v8
	vse32.v	v10, (s10)
	vle32.v	v10, (s11)
	add	a4, a4, s0
	vwmacc.vx	v10, a4, v8
	vse32.v	v10, (s11)
	vle32.v	v10, (ra)
	add	a1, a1, a2
	vwmacc.vx	v10, a1, v8
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	li	s8, 0
	vse32.v	v10, (ra)
	vle16.v	v9, (t3)
	add	a0, s1, s3
	vle32.v	v10, (t2)
	lh	a1, 16(a0)
	lh	a2, 18(a0)
	lh	s6, 20(a0)
	lh	s3, 22(a0)
	lh	s0, 24(a0)
	lh	a4, 26(a0)
	lh	a3, 28(a0)
	lh	t3, 30(a0)
	add	a1, a1, s0
	vwmacc.vx	v10, a1, v9
	vse32.v	v10, (t2)
	vle32.v	v10, (s10)
	add	a2, a2, a4
	vwmacc.vx	v10, a2, v9
	vse32.v	v10, (s10)
	vle32.v	v10, (s11)
	vle16.v	v8, (s5)
	add	a3, a3, s6
	vwmacc.vx	v10, a3, v9
	vse32.v	v10, (s11)
	vle32.v	v10, (ra)
	lh	a1, 32(a0)
	lh	a2, 34(a0)
	lh	a3, 36(a0)
	lh	s6, 38(a0)
	add	t3, t3, s3
	vwmacc.vx	v10, t3, v9
	vse32.v	v10, (ra)
	vle32.v	v10, (t2)
	lh	s0, 40(a0)
	lh	a4, 42(a0)
	lh	a5, 44(a0)
	lh	t3, 46(a0)
	add	a1, a1, s0
	vwmacc.vx	v10, a1, v8
	vse32.v	v10, (t2)
	vle32.v	v10, (s10)
	add	a2, a2, a4
	vwmacc.vx	v10, a2, v8
	vse32.v	v10, (s10)
	vle32.v	v10, (s11)
	mv	s5, s4
	vle16.v	v9, (s4)
	add	a3, a3, a5
	vwmacc.vx	v10, a3, v8
	vse32.v	v10, (s11)
	vle32.v	v10, (ra)
	lh	s3, 48(a0)
	lh	a2, 50(a0)
	lh	a3, 52(a0)
	lh	a4, 54(a0)
	add	t3, t3, s6
	vwmacc.vx	v10, t3, v8
	vse32.v	v10, (ra)
	vle32.v	v10, (t2)
	lh	a5, 56(a0)
	lh	s0, 58(a0)
	lh	a1, 60(a0)
	lh	a0, 62(a0)
	add	a5, a5, s3
	vwmacc.vx	v10, a5, v9
	vse32.v	v10, (t2)
	vle32.v	v10, (s10)
	add	a2, a2, s0
	add	a1, a1, a3
	slli	s6, a7, 10
	vwmacc.vx	v10, a2, v9
	slli	s3, a7, 9
	vse32.v	v10, (s10)
	vle32.v	v10, (s11)
	slli	a7, a7, 2
	vle16.v	v7, (s7)
	mv	s7, s2
	vle16.v	v31, (s2)
	vwmacc.vx	v10, a1, v9
	vse32.v	v10, (s11)
	vle32.v	v10, (ra)
	ld	t3, 200(sp)                     # 8-byte Folded Reload
	add	t3, t3, s3
	add	a0, a0, a4
	mv	s2, a6
	add	a3, a6, a7
	vwmacc.vx	v10, a0, v9
	vse32.v	v10, (ra)
	ld	a2, 232(sp)                     # 8-byte Folded Reload
	add	a2, a2, s6
	vmv1r.v	v9, v18
	vmv1r.v	v2, v18
	vmv1r.v	v10, v18
	vmv1r.v	v1, v18
	vmv1r.v	v5, v18
	vmv1r.v	v4, v18
	vmv1r.v	v6, v18
	vmv1r.v	v3, v18
.LBB0_10:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v11, (a2)
	vlm.v	v0, (a3)
	addi	a0, a3, 1
	add	a1, t3, s8
	addi	s8, s8, 4
	addi	a3, a3, 8
	vlm.v	v8, (a0)
	lbu	a0, 16(a1)
	lbu	a4, 17(a1)
	lbu	s9, 18(a1)
	lbu	s4, 19(a1)
	lbu	s1, 144(a1)
	lbu	s0, 145(a1)
	lbu	a5, 146(a1)
	lbu	a1, 147(a1)
	vand.vi	v14, v11, 15
	vsrl.vi	v11, v11, 4
	vadd.vx	v14, v14, t5, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v11, v11, t5, v0.t
	vwmacc.vx	v9, a0, v14
	vwmacc.vx	v5, s1, v11
	vwmacc.vx	v2, a4, v14
	vwmacc.vx	v4, s0, v11
	vwmacc.vx	v10, s9, v14
	vwmacc.vx	v6, a5, v11
	vwmacc.vx	v1, s4, v14
	vwmacc.vx	v3, a1, v11
	addi	a2, a2, 16
	bne	s8, t4, .LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v28, v7, v9
	vwmacc.vv	v22, v7, v10
	ld	a3, 224(sp)                     # 8-byte Folded Reload
	add	a3, a3, s6
	ld	a2, 152(sp)                     # 8-byte Folded Reload
	add	a2, a2, a7
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	add	a0, a0, s3
	ld	a1, 192(sp)                     # 8-byte Folded Reload
	add	s3, s3, a1
	vmv1r.v	v15, v18
	vmv1r.v	v9, v18
	vmv1r.v	v17, v18
	vmv1r.v	v11, v18
	vmv1r.v	v16, v18
	vmv1r.v	v10, v18
	vmv1r.v	v19, v18
	vmv1r.v	v14, v18
.LBB0_12:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v27, (a3)
	vlm.v	v0, (a2)
	addi	a4, a2, 1
	addi	a3, a3, 16
	vlm.v	v8, (a4)
	lbu	a4, 0(a0)
	lbu	a5, 1(a0)
	lbu	s0, 2(a0)
	lbu	s8, 3(a0)
	lbu	a1, 128(a0)
	lbu	s1, 129(a0)
	lbu	s9, 130(a0)
	lbu	s4, 131(a0)
	addi	a0, a0, 4
	vand.vi	v30, v27, 15
	vsrl.vi	v27, v27, 4
	vadd.vx	v30, v30, t5, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v27, v27, t5, v0.t
	vwmacc.vx	v15, a4, v30
	vwmacc.vx	v16, a1, v27
	vwmacc.vx	v9, a5, v30
	vwmacc.vx	v10, s1, v27
	vwmacc.vx	v17, s0, v30
	vwmacc.vx	v19, s9, v27
	vwmacc.vx	v11, s8, v30
	vwmacc.vx	v14, s4, v27
	addi	a2, a2, 8
	bne	a0, s3, .LBB0_12
# %bb.13:                               #   in Loop: Header=BB0_3 Depth=2
	li	s3, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v7, v2
	vwmacc.vv	v20, v7, v1
	vwmacc.vv	v28, v31, v5
	vwmacc.vv	v22, v31, v6
	addi	a0, sp, 472
	vle16.v	v5, (a0)
	addi	s9, sp, 488
	vle16.v	v6, (s9)
	ld	a3, 144(sp)                     # 8-byte Folded Reload
	add	a3, a3, a7
	vwmacc.vv	v24, v31, v4
	vwmacc.vv	v20, v31, v3
	vwmacc.vv	v28, v7, v15
	vwmacc.vv	v22, v7, v17
	vwmacc.vv	v28, v31, v16
	vwmacc.vv	v24, v7, v9
	vwmacc.vv	v22, v31, v19
	vwmacc.vv	v20, v7, v11
	vwmacc.vv	v24, v31, v10
	vwmacc.vv	v20, v31, v14
	ld	a2, 216(sp)                     # 8-byte Folded Reload
	add	a2, a2, s6
	vmv1r.v	v9, v18
	vmv1r.v	v2, v18
	vmv1r.v	v10, v18
	vmv1r.v	v1, v18
	vmv1r.v	v3, v18
	vmv1r.v	v31, v18
	vmv1r.v	v4, v18
	vmv1r.v	v7, v18
.LBB0_14:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v11, (a2)
	addi	a1, a3, -1
	add	a4, t3, s3
	addi	s3, s3, 4
	vlm.v	v0, (a1)
	vlm.v	v8, (a3)
	lbu	a1, 272(a4)
	lbu	a5, 273(a4)
	lbu	s0, 274(a4)
	lbu	s1, 275(a4)
	lbu	a0, 400(a4)
	lbu	s4, 401(a4)
	lbu	s8, 402(a4)
	lbu	a4, 403(a4)
	addi	a3, a3, 8
	vand.vi	v14, v11, 15
	vsrl.vi	v11, v11, 4
	vadd.vx	v14, v14, t5, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v11, v11, t5, v0.t
	vwmacc.vx	v9, a1, v14
	vwmacc.vx	v2, a5, v14
	vwmacc.vx	v10, s0, v14
	vwmacc.vx	v1, s1, v14
	vwmacc.vx	v3, a0, v11
	vwmacc.vx	v31, s4, v11
	vwmacc.vx	v4, s8, v11
	vwmacc.vx	v7, a4, v11
	addi	a2, a2, 16
	bne	s3, t4, .LBB0_14
# %bb.15:                               #   in Loop: Header=BB0_3 Depth=2
	li	a0, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v28, v5, v9
	vwmacc.vv	v22, v5, v10
	ld	s3, 136(sp)                     # 8-byte Folded Reload
	add	s3, s3, a7
	ld	a1, 208(sp)                     # 8-byte Folded Reload
	add	s6, s6, a1
	vmv1r.v	v19, v18
	vmv1r.v	v14, v18
	vmv1r.v	v15, v18
	vmv1r.v	v9, v18
	vmv1r.v	v17, v18
	vmv1r.v	v10, v18
	vmv1r.v	v16, v18
	vmv1r.v	v11, v18
.LBB0_16:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v27, (s6)
	addi	a1, s3, -1
	add	a2, t3, a0
	addi	a0, a0, 4
	vlm.v	v0, (a1)
	vlm.v	v8, (s3)
	lbu	a1, 336(a2)
	lbu	a3, 337(a2)
	lbu	a4, 338(a2)
	lbu	a5, 339(a2)
	lbu	s1, 464(a2)
	lbu	s0, 465(a2)
	lbu	a7, 466(a2)
	lbu	a2, 467(a2)
	addi	s3, s3, 8
	vand.vi	v30, v27, 15
	vsrl.vi	v27, v27, 4
	vadd.vx	v30, v30, t5, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v27, v27, t5, v0.t
	vwmacc.vx	v19, a1, v30
	vwmacc.vx	v14, a3, v30
	vwmacc.vx	v15, a4, v30
	vwmacc.vx	v9, a5, v30
	vwmacc.vx	v17, s1, v27
	vwmacc.vx	v10, s0, v27
	vwmacc.vx	v16, a7, v27
	vwmacc.vx	v11, a2, v27
	addi	s6, s6, 16
	bne	a0, t4, .LBB0_16
# %bb.17:                               #   in Loop: Header=BB0_3 Depth=2
	li	a0, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v5, v2
	vwmacc.vv	v20, v5, v1
	vwmacc.vv	v28, v6, v3
	vwmacc.vv	v22, v6, v4
	vwmacc.vv	v24, v6, v31
	vwmacc.vv	v20, v6, v7
	vwmacc.vv	v28, v5, v19
	vwmacc.vv	v22, v5, v15
	vwmacc.vv	v28, v6, v17
	vwmacc.vv	v24, v5, v14
	vwmacc.vv	v22, v6, v16
	vwmacc.vv	v20, v5, v9
	vwmacc.vv	v24, v6, v10
	vwmacc.vv	v20, v6, v11
	li	a7, 1
	mv	a6, s2
	mv	s4, s5
	mv	s2, s7
	addi	t3, sp, 392
	addi	s5, sp, 408
	addi	s7, sp, 440
	addi	a5, sp, 376
	ld	s1, 184(sp)                     # 8-byte Folded Reload
	bnez	t6, .LBB0_3
# %bb.18:                               #   in Loop: Header=BB0_2 Depth=1
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	add	t0, t0, a0
	vle16.v	v19, (t0)
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v26, v28
	addi	a0, sp, 512
	vl2r.v	v8, (a0)                        # Unknown-size Folded Reload
	vle32.v	v14, (t2)
	vfcvt.f.x.v	v28, v24
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 512
	vl2r.v	v10, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v16, v19
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v24, v16, fa3
	vfmacc.vv	v8, v24, v26
	vle32.v	v24, (s10)
	vfcvt.f.x.v	v26, v22
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 512
	vl2r.v	v22, (a0)                       # Unknown-size Folded Reload
	vfmul.vf	v30, v16, fa2
	vfmacc.vv	v10, v30, v28
	vle32.v	v28, (s11)
	vfcvt.f.x.v	v20, v20
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 512
	vl2r.v	v30, (a0)                       # Unknown-size Folded Reload
	vfmul.vf	v6, v16, fa5
	vfmacc.vv	v22, v6, v26
	vle32.v	v26, (ra)
	ld	a4, 120(sp)                     # 8-byte Folded Reload
	addi	a4, a4, 1
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	ld	a0, 232(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 232(sp)                     # 8-byte Folded Spill
	ld	a0, 224(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 224(sp)                     # 8-byte Folded Spill
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	ld	a0, 216(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	addi	a0, t0, 32
	vle16.v	v19, (a0)
	vfcvt.f.x.v	v14, v14
	vfcvt.f.x.v	v24, v24
	vfcvt.f.x.v	v28, v28
	vfcvt.f.x.v	v26, v26
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v6, v19
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v16, v16, fa4
	vfmacc.vv	v30, v16, v20
	vfmul.vf	v20, v6, fa3
	vfmul.vf	v16, v6, fa2
	vfnmsub.vv	v20, v14, v8
	vfmul.vf	v14, v6, fa5
	vfmul.vf	v8, v6, fa4
	vfnmsub.vv	v16, v24, v10
	vfnmsub.vv	v14, v28, v22
	vfnmsub.vv	v8, v26, v30
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	ld	a1, 104(sp)                     # 8-byte Folded Reload
	ld	s0, 96(sp)                      # 8-byte Folded Reload
	bne	a4, a0, .LBB0_2
# %bb.19:
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	ld	a2, 32(sp)                      # 8-byte Folded Reload
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_21
.LBB0_20:
	slliw	a0, a5, 3
	vmv2r.v	v14, v8
	vmv2r.v	v16, v8
	vmv2r.v	v20, v8
.LBB0_21:
	slli	a0, a0, 2
	slli	a1, a4, 2
	add	a0, a0, a2
	slli	a2, a4, 3
	slli	a3, a4, 4
	sub	a3, a3, a1
	vse32.v	v20, (a0)
	add	a1, a1, a0
	add	a2, a2, a0
	add	a0, a0, a3
	vse32.v	v16, (a1)
	vse32.v	v14, (a2)
	vse32.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 624
	ld	ra, 616(sp)                     # 8-byte Folded Reload
	ld	s0, 608(sp)                     # 8-byte Folded Reload
	ld	s1, 600(sp)                     # 8-byte Folded Reload
	ld	s2, 592(sp)                     # 8-byte Folded Reload
	ld	s3, 584(sp)                     # 8-byte Folded Reload
	ld	s4, 576(sp)                     # 8-byte Folded Reload
	ld	s5, 568(sp)                     # 8-byte Folded Reload
	ld	s6, 560(sp)                     # 8-byte Folded Reload
	ld	s7, 552(sp)                     # 8-byte Folded Reload
	ld	s8, 544(sp)                     # 8-byte Folded Reload
	ld	s9, 536(sp)                     # 8-byte Folded Reload
	ld	s10, 528(sp)                    # 8-byte Folded Reload
	ld	s11, 520(sp)                    # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	.cfi_restore s6
	.cfi_restore s7
	.cfi_restore s8
	.cfi_restore s9
	.cfi_restore s10
	.cfi_restore s11
	addi	sp, sp, 624
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	strip_body, .Lfunc_end0-strip_body
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
