	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl256b1p0_zvl32b1p0_zvl64b1p0"
	.file	"strip_body_regpressure.c"
	.text
	.globl	strip_body                      # -- Begin function strip_body
	.p2align	1
	.type	strip_body,@function
strip_body:                             # @strip_body
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -560
	.cfi_def_cfa_offset 560
	sd	ra, 552(sp)                     # 8-byte Folded Spill
	sd	s0, 544(sp)                     # 8-byte Folded Spill
	sd	s1, 536(sp)                     # 8-byte Folded Spill
	sd	s2, 528(sp)                     # 8-byte Folded Spill
	sd	s3, 520(sp)                     # 8-byte Folded Spill
	sd	s4, 512(sp)                     # 8-byte Folded Spill
	sd	s5, 504(sp)                     # 8-byte Folded Spill
	sd	s6, 496(sp)                     # 8-byte Folded Spill
	sd	s7, 488(sp)                     # 8-byte Folded Spill
	sd	s8, 480(sp)                     # 8-byte Folded Spill
	sd	s9, 472(sp)                     # 8-byte Folded Spill
	sd	s10, 464(sp)                    # 8-byte Folded Spill
	sd	s11, 456(sp)                    # 8-byte Folded Spill
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
	csrr	a6, vlenb
	slli	a6, a6, 3
	sub	sp, sp, a6
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xb0, 0x04, 0x22, 0x11, 0x08, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 560 + 8 * vlenb
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	sd	a3, 104(sp)                     # 8-byte Folded Spill
	beqz	a3, .LBB0_20
# %bb.1:
	mv	s1, a0
	sd	a2, 24(sp)                      # 8-byte Folded Spill
	sd	a4, 32(sp)                      # 8-byte Folded Spill
	li	a4, 0
	vmv.v.i	v10, 0
	slliw	a6, a5, 3
	addi	t3, sp, 360
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	addi	s0, sp, 328
	addi	s5, sp, 344
	slliw	a0, a5, 4
	sd	a0, 80(sp)                      # 8-byte Folded Spill
	addi	s6, sp, 216
	addi	s7, sp, 248
	addi	s11, sp, 280
	addi	a0, a1, 80
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	li	a0, 11
	addi	a5, sp, 312
	addi	s2, sp, 184
	li	s4, 64
	vmv2r.v	v20, v8
	vmv2r.v	v18, v8
	vmv2r.v	v14, v8
	addi	a3, a6, 64
	sd	a3, 120(sp)                     # 8-byte Folded Spill
	addiw	a3, a6, 192
	sd	a3, 72(sp)                      # 8-byte Folded Spill
	addiw	a3, a6, 208
	sd	a3, 64(sp)                      # 8-byte Folded Spill
	addiw	a3, a6, 224
	sd	a3, 56(sp)                      # 8-byte Folded Spill
	add	a3, a6, s1
	slli	a2, a0, 8
	addi	a0, a3, 256
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 512
	sd	a0, 160(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 768
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 1024
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	sd	a6, 16(sp)                      # 8-byte Folded Spill
	addiw	a0, a6, 240
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	mv	t5, a1
	sd	a1, 96(sp)                      # 8-byte Folded Spill
	sd	s1, 88(sp)                      # 8-byte Folded Spill
	sd	a2, 48(sp)                      # 8-byte Folded Spill
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_10 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_14 Depth 3
                                        #       Child Loop BB0_16 Depth 3
	addi	a0, sp, 448
	vs2r.v	v20, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 448
	vs2r.v	v18, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 448
	vs2r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a3, 6
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 448
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	li	s8, 0
	li	a0, 1168
	mul	ra, a4, a0
	sd	a4, 112(sp)                     # 8-byte Folded Spill
	mul	a0, a4, a2
	li	a2, 1
	add	ra, ra, a1
	add	a7, s1, a0
	addi	a0, ra, 144
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	flw	fa3, 0(ra)
	flw	fa2, 4(ra)
	flw	fa5, 8(ra)
	flw	fa4, 12(ra)
	ld	a3, 72(sp)                      # 8-byte Folded Reload
	add	a3, a3, a7
	addi	s9, ra, 1040
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v13, (a3)
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	ld	s10, 56(sp)                     # 8-byte Folded Reload
	add	s10, s10, a7
	ld	t0, 40(sp)                      # 8-byte Folded Reload
	add	t0, t0, a7
	vand.vi	v8, v13, 3
	vand.vi	v9, v13, 12
	vsll.vi	v28, v8, 4
	vsll.vi	v29, v9, 2
	vmv2r.v	v26, v10
	vmv2r.v	v24, v10
	vmv2r.v	v22, v10
	vmv2r.v	v20, v10
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_10 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_14 Depth 3
                                        #       Child Loop BB0_16 Depth 3
	slli	a3, s8, 6
	ld	a0, 120(sp)                     # 8-byte Folded Reload
	addw	a3, a3, a0
	add	a4, a7, a3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a4)
	andi	t1, a2, 1
	addiw	s1, a3, 16
	addiw	a4, a3, 32
	vand.vi	v9, v8, 15
	vsrl.vi	v8, v8, 4
	addiw	a2, a3, 48
	beqz	t1, .LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	vor.vv	v9, v28, v9
	vor.vv	v8, v29, v8
	add	s1, s1, a7
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	vle8.v	v12, (a0)
	add	a4, a4, a7
	vle8.v	v14, (s10)
	add	a2, a2, a7
	vle8.v	v15, (t0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v9
	vzext.vf2	v9, v8
	vle8.v	v8, (s1)
	vle8.v	v30, (a4)
	vle8.v	v31, (a2)
	addi	a0, sp, 376
	vse16.v	v17, (a0)
	vse16.v	v9, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v12, 3
	vand.vi	v12, v12, 12
	vand.vi	v17, v8, 15
	vsll.vi	v9, v9, 4
	vor.vv	v17, v9, v17
	vand.vi	v9, v14, 3
	vand.vi	v14, v14, 12
	vsrl.vi	v8, v8, 4
	vsll.vi	v12, v12, 2
	vor.vv	v8, v12, v8
	vand.vi	v12, v30, 15
	vsll.vi	v9, v9, 4
	vor.vv	v12, v9, v12
	vand.vi	v9, v15, 3
	vsrl.vi	v30, v30, 4
	vsll.vi	v14, v14, 2
	vor.vv	v14, v14, v30
	vand.vi	v30, v31, 15
	vsll.vi	v9, v9, 4
	vor.vv	v9, v9, v30
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v30, v17
	addi	a0, sp, 392
	vse16.v	v30, (a0)
	vzext.vf2	v17, v8
	vse16.v	v17, (s0)
	vzext.vf2	v8, v12
	addi	a0, sp, 408
	vse16.v	v8, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v15, 12
	vsrl.vi	v12, v31, 4
	vsll.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vse16.v	v15, (s5)
	j	.LBB0_6
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=2
	li	a1, 48
	vand.vx	v12, v13, a1
	li	a3, -64
	vand.vx	v14, v13, a3
	add	s1, s1, a7
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	vle8.v	v15, (a0)
	add	a4, a4, a7
	vle8.v	v17, (s10)
	add	a2, a2, a7
	vle8.v	v30, (t0)
	vsrl.vi	v14, v14, 2
	vor.vv	v9, v12, v9
	vle8.v	v12, (s1)
	vle8.v	v31, (a4)
	vor.vv	v8, v14, v8
	vle8.v	v14, (a2)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v7, v9
	vzext.vf2	v9, v8
	addi	a0, sp, 376
	vse16.v	v7, (a0)
	vse16.v	v9, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v15, a1
	vand.vx	v9, v15, a3
	vand.vi	v15, v12, 15
	vor.vv	v8, v8, v15
	vand.vx	v15, v17, a1
	vand.vx	v17, v17, a3
	vsrl.vi	v12, v12, 4
	vsrl.vi	v9, v9, 2
	vor.vv	v12, v9, v12
	vand.vi	v9, v31, 15
	vor.vv	v15, v15, v9
	vand.vx	v9, v30, a1
	vsrl.vi	v31, v31, 4
	vsrl.vi	v17, v17, 2
	vor.vv	v17, v17, v31
	vand.vi	v31, v14, 15
	vor.vv	v9, v9, v31
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v31, v8
	addi	a0, sp, 392
	vse16.v	v31, (a0)
	vzext.vf2	v8, v12
	vse16.v	v8, (s0)
	vzext.vf2	v8, v15
	addi	a0, sp, 408
	vse16.v	v8, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v30, a3
	vsrl.vi	v12, v14, 4
	vsrl.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v17
	vse16.v	v14, (s5)
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e8, mf2, ta, ma
	vor.vv	v8, v8, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v12, v8
	vse16.v	v12, (t3)
	vle16.v	v8, (a5)
	vzext.vf2	v12, v9
	addi	a0, sp, 424
	vse16.v	v12, (a0)
	slli	a6, s8, 6
	beqz	t1, .LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=2
	lh	t2, 1040(ra)
	lh	a1, 1042(ra)
	lh	a2, 1044(ra)
	lh	a3, 1046(ra)
	lh	a4, 1048(ra)
	lh	a5, 1050(ra)
	lh	s1, 1052(ra)
	lh	a0, 1054(ra)
	add	a4, a4, t2
	add	a1, a1, a5
	add	a2, a2, s1
	vmv2r.v	v14, v10
	vwmacc.vx	v14, a4, v8
	vse32.v	v14, (s2)
	vmv2r.v	v14, v10
	vwmacc.vx	v14, a1, v8
	vse32.v	v14, (s6)
	vmv2r.v	v14, v10
	vwmacc.vx	v14, a2, v8
	vse32.v	v14, (s7)
	vmv2r.v	v14, v10
	add	a0, a0, a3
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=2
	add	a3, s9, a6
	vle32.v	v14, (s2)
	lh	t2, 0(a3)
	lh	s1, 2(a3)
	lh	a5, 4(a3)
	lh	a0, 6(a3)
	lh	a1, 8(a3)
	lh	a2, 10(a3)
	lh	a4, 12(a3)
	lh	a3, 14(a3)
	add	a1, a1, t2
	vwmacc.vx	v14, a1, v8
	vse32.v	v14, (s2)
	vle32.v	v14, (s6)
	add	a2, a2, s1
	vwmacc.vx	v14, a2, v8
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	add	a4, a4, a5
	vwmacc.vx	v14, a4, v8
	vse32.v	v14, (s7)
	vle32.v	v14, (s11)
	add	a0, a0, a3
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	vwmacc.vx	v14, a0, v8
	li	t4, 0
	vse32.v	v14, (s11)
	mv	s3, s0
	vle16.v	v9, (s0)
	add	a2, s9, a6
	vle32.v	v14, (s2)
	lh	a0, 16(a2)
	lh	a1, 18(a2)
	lh	a3, 20(a2)
	lh	a6, 22(a2)
	lh	a5, 24(a2)
	lh	s1, 26(a2)
	lh	a4, 28(a2)
	lh	t2, 30(a2)
	add	a0, a0, a5
	vwmacc.vx	v14, a0, v9
	vse32.v	v14, (s2)
	vle32.v	v14, (s6)
	add	a1, a1, s1
	vwmacc.vx	v14, a1, v9
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	mv	t6, s5
	vle16.v	v8, (s5)
	add	a3, a3, a4
	vwmacc.vx	v14, a3, v9
	vse32.v	v14, (s7)
	vle32.v	v14, (s11)
	lh	a0, 32(a2)
	lh	a1, 34(a2)
	lh	a3, 36(a2)
	lh	a4, 38(a2)
	add	a6, a6, t2
	vwmacc.vx	v14, a6, v9
	vse32.v	v14, (s11)
	vle32.v	v14, (s2)
	lh	a5, 40(a2)
	lh	s1, 42(a2)
	lh	s0, 44(a2)
	lh	a6, 46(a2)
	add	a0, a0, a5
	vwmacc.vx	v14, a0, v8
	vse32.v	v14, (s2)
	vle32.v	v14, (s6)
	add	a1, a1, s1
	vwmacc.vx	v14, a1, v8
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	mv	s5, t3
	vle16.v	v9, (t3)
	add	a3, a3, s0
	vwmacc.vx	v14, a3, v8
	vse32.v	v14, (s7)
	vle32.v	v14, (s11)
	lh	a0, 48(a2)
	lh	a1, 50(a2)
	lh	a3, 52(a2)
	lh	a5, 54(a2)
	add	a4, a4, a6
	vwmacc.vx	v14, a4, v8
	vse32.v	v14, (s11)
	vle32.v	v14, (s2)
	lh	a4, 56(a2)
	lh	s1, 58(a2)
	lh	s0, 60(a2)
	lh	a2, 62(a2)
	add	a0, a0, a4
	vwmacc.vx	v14, a0, v9
	vse32.v	v14, (s2)
	vle32.v	v14, (s6)
	add	a1, a1, s1
	vwmacc.vx	v14, a1, v9
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	add	a3, a3, s0
	vwmacc.vx	v14, a3, v9
	slli	t2, s8, 10
	slli	s8, s8, 9
	addi	a0, sp, 376
	vle16.v	v7, (a0)
	vse32.v	v14, (s7)
	vle32.v	v14, (s11)
	addi	a0, sp, 392
	vle16.v	v6, (a0)
	add	a2, a2, a5
	add	t3, t5, s8
	vwmacc.vx	v14, a2, v9
	vse32.v	v14, (s11)
	ld	a6, 168(sp)                     # 8-byte Folded Reload
	add	a6, a6, t2
	vmv1r.v	v8, v16
	vmv1r.v	v3, v16
	vmv1r.v	v9, v16
	vmv1r.v	v2, v16
	vmv1r.v	v31, v16
	vmv1r.v	v5, v16
	vmv1r.v	v30, v16
	vmv1r.v	v4, v16
.LBB0_10:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v12, (a6)
	add	a0, t3, t4
	addi	t4, t4, 4
	lbu	a1, 16(a0)
	lbu	a2, 17(a0)
	lbu	a3, 18(a0)
	lbu	a4, 19(a0)
	lbu	a5, 144(a0)
	lbu	s0, 145(a0)
	lbu	s1, 146(a0)
	lbu	a0, 147(a0)
	vand.vi	v14, v12, 15
	vsrl.vi	v12, v12, 4
	vwmacc.vx	v8, a1, v14
	vwmacc.vx	v31, a5, v12
	vwmacc.vx	v3, a2, v14
	vwmacc.vx	v5, s0, v12
	vwmacc.vx	v9, a3, v14
	vwmacc.vx	v30, s1, v12
	vwmacc.vx	v2, a4, v14
	vwmacc.vx	v4, a0, v12
	addi	a6, a6, 16
	bne	t4, s4, .LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v7, v8
	vwmacc.vv	v22, v7, v9
	ld	a4, 160(sp)                     # 8-byte Folded Reload
	add	a4, a4, t2
	ld	a2, 176(sp)                     # 8-byte Folded Reload
	add	a2, a2, s8
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	add	s8, s8, a0
	vmv1r.v	v9, v16
	vmv1r.v	v1, v16
	vmv1r.v	v15, v16
	vmv1r.v	v0, v16
	vmv1r.v	v14, v16
	vmv1r.v	v12, v16
	vmv1r.v	v17, v16
	vmv1r.v	v8, v16
.LBB0_12:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v18, (a4)
	lbu	a0, 0(a2)
	lbu	a3, 1(a2)
	lbu	t4, 2(a2)
	lbu	a6, 3(a2)
	lbu	s0, 128(a2)
	lbu	a1, 129(a2)
	lbu	s1, 130(a2)
	lbu	a5, 131(a2)
	addi	a2, a2, 4
	vand.vi	v19, v18, 15
	vsrl.vi	v18, v18, 4
	vwmacc.vx	v9, a0, v19
	vwmacc.vx	v14, s0, v18
	vwmacc.vx	v1, a3, v19
	vwmacc.vx	v12, a1, v18
	vwmacc.vx	v15, t4, v19
	vwmacc.vx	v17, s1, v18
	vwmacc.vx	v0, a6, v19
	vwmacc.vx	v8, a5, v18
	addi	a4, a4, 16
	bne	a2, s8, .LBB0_12
# %bb.13:                               #   in Loop: Header=BB0_3 Depth=2
	li	t4, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v7, v3
	vwmacc.vv	v20, v7, v2
	vwmacc.vv	v26, v6, v31
	vwmacc.vv	v22, v6, v30
	addi	a0, sp, 408
	vle16.v	v31, (a0)
	addi	a0, sp, 424
	vle16.v	v30, (a0)
	vwmacc.vv	v24, v6, v5
	vwmacc.vv	v20, v6, v4
	vwmacc.vv	v26, v7, v9
	vwmacc.vv	v22, v7, v15
	vwmacc.vv	v26, v6, v14
	vwmacc.vv	v24, v7, v1
	vwmacc.vv	v22, v6, v17
	vwmacc.vv	v20, v7, v0
	vwmacc.vv	v24, v6, v12
	vwmacc.vv	v20, v6, v8
	ld	a4, 152(sp)                     # 8-byte Folded Reload
	add	a4, a4, t2
	vmv1r.v	v8, v16
	vmv1r.v	v3, v16
	vmv1r.v	v9, v16
	vmv1r.v	v2, v16
	vmv1r.v	v4, v16
	vmv1r.v	v7, v16
	vmv1r.v	v5, v16
	vmv1r.v	v6, v16
.LBB0_14:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v12, (a4)
	add	a0, t3, t4
	addi	t4, t4, 4
	lbu	a1, 272(a0)
	lbu	a3, 273(a0)
	lbu	a5, 274(a0)
	lbu	a6, 275(a0)
	lbu	s0, 400(a0)
	lbu	s1, 401(a0)
	lbu	a2, 402(a0)
	lbu	a0, 403(a0)
	vand.vi	v14, v12, 15
	vsrl.vi	v12, v12, 4
	vwmacc.vx	v8, a1, v14
	vwmacc.vx	v4, s0, v12
	vwmacc.vx	v3, a3, v14
	vwmacc.vx	v7, s1, v12
	vwmacc.vx	v9, a5, v14
	vwmacc.vx	v5, a2, v12
	vwmacc.vx	v2, a6, v14
	vwmacc.vx	v6, a0, v12
	addi	a4, a4, 16
	bne	t4, s4, .LBB0_14
# %bb.15:                               #   in Loop: Header=BB0_3 Depth=2
	li	a2, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v31, v8
	vwmacc.vv	v22, v31, v9
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	add	t2, t2, a0
	vmv1r.v	v0, v16
	vmv1r.v	v15, v16
	vmv1r.v	v14, v16
	vmv1r.v	v9, v16
	vmv1r.v	v1, v16
	vmv1r.v	v8, v16
	vmv1r.v	v17, v16
	vmv1r.v	v12, v16
.LBB0_16:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v18, (t2)
	add	a0, t3, a2
	addi	a2, a2, 4
	lbu	a1, 336(a0)
	lbu	a3, 337(a0)
	lbu	a4, 338(a0)
	lbu	a6, 339(a0)
	lbu	s1, 464(a0)
	lbu	s0, 465(a0)
	lbu	a5, 466(a0)
	lbu	a0, 467(a0)
	vand.vi	v19, v18, 15
	vsrl.vi	v18, v18, 4
	vwmacc.vx	v0, a1, v19
	vwmacc.vx	v1, s1, v18
	vwmacc.vx	v15, a3, v19
	vwmacc.vx	v8, s0, v18
	vwmacc.vx	v14, a4, v19
	vwmacc.vx	v17, a5, v18
	vwmacc.vx	v9, a6, v19
	vwmacc.vx	v12, a0, v18
	addi	t2, t2, 16
	bne	a2, s4, .LBB0_16
# %bb.17:                               #   in Loop: Header=BB0_3 Depth=2
	li	a2, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v31, v3
	vwmacc.vv	v20, v31, v2
	vwmacc.vv	v26, v30, v4
	vwmacc.vv	v22, v30, v5
	vwmacc.vv	v24, v30, v7
	vwmacc.vv	v20, v30, v6
	vwmacc.vv	v26, v31, v0
	vwmacc.vv	v22, v31, v14
	vwmacc.vv	v26, v30, v1
	vwmacc.vv	v24, v31, v15
	vwmacc.vv	v22, v30, v17
	vwmacc.vv	v20, v31, v9
	vwmacc.vv	v24, v30, v8
	vwmacc.vv	v20, v30, v12
	li	s8, 1
	mv	t3, s5
	mv	s0, s3
	mv	s5, t6
	addi	a5, sp, 312
	bnez	t1, .LBB0_3
# %bb.18:                               #   in Loop: Header=BB0_2 Depth=1
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	vle16.v	v17, (a0)
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v26, v26
	addi	a1, sp, 448
	vl2r.v	v8, (a1)                        # Unknown-size Folded Reload
	vle32.v	v14, (s2)
	vfcvt.f.x.v	v28, v24
	csrr	a1, vlenb
	slli	a1, a1, 1
	add	a1, a1, sp
	addi	a1, a1, 448
	vl2r.v	v12, (a1)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v18, v17
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v24, v18, fa3
	vfmacc.vv	v8, v24, v26
	vle32.v	v24, (s6)
	vfcvt.f.x.v	v26, v22
	csrr	a1, vlenb
	slli	a1, a1, 2
	add	a1, a1, sp
	addi	a1, a1, 448
	vl2r.v	v22, (a1)                       # Unknown-size Folded Reload
	vfmul.vf	v30, v18, fa2
	vfmacc.vv	v12, v30, v28
	vle32.v	v28, (s7)
	vfcvt.f.x.v	v20, v20
	csrr	a1, vlenb
	li	a2, 6
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 448
	vl2r.v	v30, (a1)                       # Unknown-size Folded Reload
	vfmul.vf	v6, v18, fa5
	vfmacc.vv	v22, v6, v26
	vle32.v	v26, (s11)
	ld	a4, 112(sp)                     # 8-byte Folded Reload
	addi	a4, a4, 1
	addi	t5, t5, 1168
	ld	a2, 48(sp)                      # 8-byte Folded Reload
	ld	a1, 168(sp)                     # 8-byte Folded Reload
	add	a1, a1, a2
	sd	a1, 168(sp)                     # 8-byte Folded Spill
	ld	a1, 160(sp)                     # 8-byte Folded Reload
	add	a1, a1, a2
	sd	a1, 160(sp)                     # 8-byte Folded Spill
	ld	a1, 176(sp)                     # 8-byte Folded Reload
	addi	a1, a1, 1168
	sd	a1, 176(sp)                     # 8-byte Folded Spill
	ld	a1, 152(sp)                     # 8-byte Folded Reload
	add	a1, a1, a2
	sd	a1, 152(sp)                     # 8-byte Folded Spill
	addi	a0, a0, 32
	vle16.v	v17, (a0)
	vfcvt.f.x.v	v14, v14
	vfcvt.f.x.v	v24, v24
	vfcvt.f.x.v	v28, v28
	vfcvt.f.x.v	v26, v26
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v6, v17
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v18, v18, fa4
	vfmacc.vv	v30, v18, v20
	vfmul.vf	v20, v6, fa3
	vfmul.vf	v18, v6, fa2
	vfnmsub.vv	v20, v14, v8
	vfmul.vf	v14, v6, fa5
	vfmul.vf	v8, v6, fa4
	vfnmsub.vv	v18, v24, v12
	vfnmsub.vv	v14, v28, v22
	vfnmsub.vv	v8, v26, v30
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	ld	a1, 96(sp)                      # 8-byte Folded Reload
	ld	s1, 88(sp)                      # 8-byte Folded Reload
	bne	a4, a0, .LBB0_2
# %bb.19:
	ld	a4, 32(sp)                      # 8-byte Folded Reload
	ld	a2, 24(sp)                      # 8-byte Folded Reload
	ld	a0, 16(sp)                      # 8-byte Folded Reload
	j	.LBB0_21
.LBB0_20:
	slliw	a0, a5, 3
	vmv2r.v	v14, v8
	vmv2r.v	v18, v8
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
	vse32.v	v18, (a1)
	vse32.v	v14, (a2)
	vse32.v	v8, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	sp, sp, a0
	.cfi_def_cfa sp, 560
	ld	ra, 552(sp)                     # 8-byte Folded Reload
	ld	s0, 544(sp)                     # 8-byte Folded Reload
	ld	s1, 536(sp)                     # 8-byte Folded Reload
	ld	s2, 528(sp)                     # 8-byte Folded Reload
	ld	s3, 520(sp)                     # 8-byte Folded Reload
	ld	s4, 512(sp)                     # 8-byte Folded Reload
	ld	s5, 504(sp)                     # 8-byte Folded Reload
	ld	s6, 496(sp)                     # 8-byte Folded Reload
	ld	s7, 488(sp)                     # 8-byte Folded Reload
	ld	s8, 480(sp)                     # 8-byte Folded Reload
	ld	s9, 472(sp)                     # 8-byte Folded Reload
	ld	s10, 464(sp)                    # 8-byte Folded Reload
	ld	s11, 456(sp)                    # 8-byte Folded Reload
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
	addi	sp, sp, 560
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	strip_body, .Lfunc_end0-strip_body
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
