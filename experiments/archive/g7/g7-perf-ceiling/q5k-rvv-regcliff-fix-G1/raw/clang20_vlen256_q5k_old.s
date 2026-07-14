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
	addi	sp, sp, -592
	.cfi_def_cfa_offset 592
	sd	ra, 584(sp)                     # 8-byte Folded Spill
	sd	s0, 576(sp)                     # 8-byte Folded Spill
	sd	s1, 568(sp)                     # 8-byte Folded Spill
	sd	s2, 560(sp)                     # 8-byte Folded Spill
	sd	s3, 552(sp)                     # 8-byte Folded Spill
	sd	s4, 544(sp)                     # 8-byte Folded Spill
	sd	s5, 536(sp)                     # 8-byte Folded Spill
	sd	s6, 528(sp)                     # 8-byte Folded Spill
	sd	s7, 520(sp)                     # 8-byte Folded Spill
	sd	s8, 512(sp)                     # 8-byte Folded Spill
	sd	s9, 504(sp)                     # 8-byte Folded Spill
	sd	s10, 496(sp)                    # 8-byte Folded Spill
	sd	s11, 488(sp)                    # 8-byte Folded Spill
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
	li	a7, 10
	mul	a6, a6, a7
	sub	sp, sp, a6
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xd0, 0x04, 0x22, 0x11, 0x0a, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 592 + 10 * vlenb
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	sd	a3, 112(sp)                     # 8-byte Folded Spill
	beqz	a3, .LBB0_24
# %bb.1:
	mv	s1, a0
	sd	a2, 32(sp)                      # 8-byte Folded Spill
	sd	a4, 40(sp)                      # 8-byte Folded Spill
	li	a3, 0
	vmv.v.i	v10, 0
	slliw	a2, a5, 3
	addi	t3, sp, 456
	addi	t5, sp, 392
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	addi	t6, sp, 424
	addi	a6, sp, 360
	addi	s10, sp, 440
	addi	s4, sp, 376
	slliw	a0, a5, 4
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	s6, sp, 248
	addi	s7, sp, 280
	addi	s8, sp, 312
	addi	a0, a1, 80
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	li	a0, 11
	addi	s3, sp, 408
	addi	a5, sp, 344
	addi	s9, sp, 216
	li	ra, 64
	vmv2r.v	v20, v8
	vmv2r.v	v18, v8
	vmv2r.v	v14, v8
	addi	a4, a2, 64
	sd	a4, 128(sp)                     # 8-byte Folded Spill
	addiw	a4, a2, 192
	sd	a4, 80(sp)                      # 8-byte Folded Spill
	addiw	a4, a2, 208
	sd	a4, 72(sp)                      # 8-byte Folded Spill
	addiw	a4, a2, 224
	sd	a4, 64(sp)                      # 8-byte Folded Spill
	addiw	a4, a2, 240
	sd	a4, 56(sp)                      # 8-byte Folded Spill
	sd	a2, 24(sp)                      # 8-byte Folded Spill
	add	a2, a2, s1
	slli	a4, a0, 8
	addi	a0, a2, 256
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	addi	a0, a2, 512
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	addi	a0, a2, 768
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	sd	a1, 176(sp)                     # 8-byte Folded Spill
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	sd	s1, 96(sp)                      # 8-byte Folded Spill
	sd	a4, 48(sp)                      # 8-byte Folded Spill
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_11 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_20 Depth 3
	addi	a0, sp, 480
	vs2r.v	v20, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 480
	vs2r.v	v18, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 480
	vs2r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 6
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 480
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	li	t2, 0
	li	a0, 1168
	mul	s11, a3, a0
	sd	a3, 120(sp)                     # 8-byte Folded Spill
	mul	a0, a3, a4
	li	a3, 1
	add	s11, s11, a1
	add	a7, s1, a0
	addi	a0, s11, 144
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	flw	fa3, 0(s11)
	flw	fa2, 4(s11)
	flw	fa5, 8(s11)
	flw	fa4, 12(s11)
	ld	a2, 80(sp)                      # 8-byte Folded Reload
	add	a2, a2, a7
	addi	s0, s11, 1040
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v13, (a2)
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	vand.vi	v8, v13, 3
	vand.vi	v9, v13, 12
	vsll.vi	v8, v8, 4
	csrr	a0, vlenb
	slli	a1, a0, 3
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 480
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vsll.vi	v8, v9, 2
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 480
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vmv2r.v	v26, v10
	vmv2r.v	v24, v10
	vmv2r.v	v22, v10
	vmv2r.v	v20, v10
	sd	s0, 160(sp)                     # 8-byte Folded Spill
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_11 Depth 3
                                        #       Child Loop BB0_15 Depth 3
                                        #       Child Loop BB0_18 Depth 3
                                        #       Child Loop BB0_20 Depth 3
	slli	a2, t2, 6
	ld	a4, 128(sp)                     # 8-byte Folded Reload
	addw	a4, a4, a2
	add	a2, a7, a4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a2)
	andi	t0, a3, 1
	addiw	a3, a4, 16
	addiw	a2, a4, 32
	vand.vi	v9, v8, 15
	vsrl.vi	v8, v8, 4
	addiw	a1, a4, 48
	beqz	t0, .LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	csrr	a0, vlenb
	slli	a4, a0, 3
	add	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 480
	vl1r.v	v12, (a0)                       # Unknown-size Folded Reload
	vor.vv	v9, v12, v9
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 480
	vl1r.v	v12, (a0)                       # Unknown-size Folded Reload
	vor.vv	v8, v12, v8
	add	a3, a3, a7
	ld	a0, 152(sp)                     # 8-byte Folded Reload
	vle8.v	v12, (a0)
	add	a2, a2, a7
	ld	a0, 144(sp)                     # 8-byte Folded Reload
	vle8.v	v14, (a0)
	add	a1, a1, a7
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	vle8.v	v15, (a0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v9
	vzext.vf2	v9, v8
	vle8.v	v8, (a3)
	vle8.v	v18, (a2)
	vle8.v	v19, (a1)
	vse16.v	v17, (s3)
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
	vand.vi	v12, v18, 15
	vsll.vi	v9, v9, 4
	vor.vv	v12, v9, v12
	vand.vi	v9, v15, 3
	vsrl.vi	v18, v18, 4
	vsll.vi	v14, v14, 2
	vor.vv	v14, v14, v18
	vand.vi	v18, v19, 15
	vsll.vi	v9, v9, 4
	vor.vv	v9, v9, v18
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v18, v17
	vse16.v	v18, (t6)
	vzext.vf2	v17, v8
	vse16.v	v17, (a6)
	vzext.vf2	v8, v12
	vse16.v	v8, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v15, 12
	vsrl.vi	v12, v19, 4
	vsll.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vse16.v	v15, (s4)
	j	.LBB0_6
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=2
	li	a0, 48
	vand.vx	v12, v13, a0
	li	a4, -64
	vand.vx	v14, v13, a4
	add	a3, a3, a7
	ld	s1, 152(sp)                     # 8-byte Folded Reload
	vle8.v	v15, (s1)
	add	a2, a2, a7
	ld	s1, 144(sp)                     # 8-byte Folded Reload
	vle8.v	v17, (s1)
	add	a1, a1, a7
	ld	s1, 136(sp)                     # 8-byte Folded Reload
	vle8.v	v18, (s1)
	vsrl.vi	v14, v14, 2
	vor.vv	v9, v12, v9
	vle8.v	v12, (a3)
	vle8.v	v19, (a2)
	vor.vv	v8, v14, v8
	vle8.v	v14, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v28, v9
	vzext.vf2	v9, v8
	vse16.v	v28, (s3)
	vse16.v	v9, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v15, a0
	vand.vx	v9, v15, a4
	vand.vi	v15, v12, 15
	vor.vv	v8, v8, v15
	vand.vx	v15, v17, a0
	vand.vx	v17, v17, a4
	vsrl.vi	v12, v12, 4
	vsrl.vi	v9, v9, 2
	vor.vv	v12, v9, v12
	vand.vi	v9, v19, 15
	vor.vv	v15, v15, v9
	vand.vx	v9, v18, a0
	vsrl.vi	v19, v19, 4
	vsrl.vi	v17, v17, 2
	vor.vv	v17, v17, v19
	vand.vi	v19, v14, 15
	vor.vv	v9, v9, v19
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v8
	vse16.v	v19, (t6)
	vzext.vf2	v8, v12
	vse16.v	v8, (a6)
	vzext.vf2	v8, v15
	vse16.v	v8, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v18, a4
	vsrl.vi	v12, v14, 4
	vsrl.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v17
	vse16.v	v14, (s4)
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e8, mf2, ta, ma
	vor.vv	v8, v8, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v12, v8
	vse16.v	v12, (t5)
	vle16.v	v8, (a5)
	vzext.vf2	v12, v9
	vse16.v	v12, (t3)
	slli	t1, t2, 6
	beqz	t0, .LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=2
	lh	t4, 1040(s11)
	lh	a1, 1042(s11)
	lh	a2, 1044(s11)
	lh	a3, 1046(s11)
	lh	a4, 1048(s11)
	lh	a5, 1050(s11)
	lh	s1, 1052(s11)
	lh	a0, 1054(s11)
	add	a4, a4, t4
	add	a1, a1, a5
	add	a2, a2, s1
	vmv2r.v	v14, v10
	vwmacc.vx	v14, a4, v8
	vse32.v	v14, (s9)
	vmv2r.v	v14, v10
	vwmacc.vx	v14, a1, v8
	vse32.v	v14, (s6)
	vmv2r.v	v14, v10
	vwmacc.vx	v14, a2, v8
	vse32.v	v14, (s7)
	vmv2r.v	v14, v10
	add	a0, a0, a3
	vwmacc.vx	v14, a0, v8
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=2
	add	a2, s0, t1
	vle32.v	v14, (s9)
	lh	t4, 0(a2)
	lh	a4, 2(a2)
	lh	a5, 4(a2)
	lh	s1, 6(a2)
	lh	a0, 8(a2)
	lh	a1, 10(a2)
	lh	a3, 12(a2)
	lh	a2, 14(a2)
	add	a0, a0, t4
	vwmacc.vx	v14, a0, v8
	vse32.v	v14, (s9)
	vle32.v	v14, (s6)
	add	a1, a1, a4
	vwmacc.vx	v14, a1, v8
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	add	a3, a3, a5
	vwmacc.vx	v14, a3, v8
	vse32.v	v14, (s7)
	vle32.v	v14, (s8)
	add	a2, a2, s1
	vwmacc.vx	v14, a2, v8
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	li	t4, 0
	vse32.v	v14, (s8)
	vle16.v	v9, (a6)
	add	a1, s0, t1
	vle32.v	v14, (s9)
	lh	a0, 16(a1)
	lh	a2, 18(a1)
	lh	a3, 20(a1)
	lh	t1, 22(a1)
	lh	a5, 24(a1)
	lh	s1, 26(a1)
	lh	a4, 28(a1)
	lh	s2, 30(a1)
	add	a0, a0, a5
	vwmacc.vx	v14, a0, v9
	vse32.v	v14, (s9)
	vle32.v	v14, (s6)
	add	a2, a2, s1
	vwmacc.vx	v14, a2, v9
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	mv	s5, s4
	vle16.v	v8, (s4)
	add	a3, a3, a4
	vwmacc.vx	v14, a3, v9
	vse32.v	v14, (s7)
	vle32.v	v14, (s8)
	lh	a0, 32(a1)
	lh	a2, 34(a1)
	lh	a3, 36(a1)
	lh	a4, 38(a1)
	add	t1, t1, s2
	vwmacc.vx	v14, t1, v9
	vse32.v	v14, (s8)
	vle32.v	v14, (s9)
	lh	a5, 40(a1)
	lh	s1, 42(a1)
	lh	s0, 44(a1)
	lh	t1, 46(a1)
	add	a0, a0, a5
	vwmacc.vx	v14, a0, v8
	vse32.v	v14, (s9)
	vle32.v	v14, (s6)
	add	a2, a2, s1
	vwmacc.vx	v14, a2, v8
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	mv	a6, t5
	vle16.v	v9, (t5)
	add	a3, a3, s0
	vwmacc.vx	v14, a3, v8
	vse32.v	v14, (s7)
	vle32.v	v14, (s8)
	lh	a0, 48(a1)
	lh	a2, 50(a1)
	lh	a3, 52(a1)
	lh	a5, 54(a1)
	add	a4, a4, t1
	vwmacc.vx	v14, a4, v8
	vse32.v	v14, (s8)
	vle32.v	v14, (s9)
	lh	a4, 56(a1)
	lh	s1, 58(a1)
	lh	s0, 60(a1)
	lh	a1, 62(a1)
	add	a0, a0, a4
	vwmacc.vx	v14, a0, v9
	vse32.v	v14, (s9)
	vle32.v	v14, (s6)
	add	a2, a2, s1
	add	a3, a3, s0
	slli	t1, t2, 10
	vwmacc.vx	v14, a2, v9
	vse32.v	v14, (s6)
	vle32.v	v14, (s7)
	slli	t5, t2, 9
	slli	t3, t2, 2
	vle16.v	v30, (s3)
	vwmacc.vx	v14, a3, v9
	vse32.v	v14, (s7)
	vle32.v	v14, (s8)
	mv	s4, t6
	vle16.v	v31, (t6)
	addi	t2, t3, 1
	add	a1, a1, a5
	vwmacc.vx	v14, a1, v9
	vse32.v	v14, (s8)
	ld	s2, 176(sp)                     # 8-byte Folded Reload
	add	s2, s2, t5
	ld	a4, 208(sp)                     # 8-byte Folded Reload
	vmv1r.v	v8, v16
	vmv1r.v	v3, v16
	vmv1r.v	v9, v16
	vmv1r.v	v2, v16
	vmv1r.v	v6, v16
	vmv1r.v	v5, v16
	vmv1r.v	v7, v16
	vmv1r.v	v4, v16
	j	.LBB0_11
.LBB0_10:                               #   in Loop: Header=BB0_11 Depth=3
	vand.vi	v17, v14, 15
	vsrl.vi	v14, v14, 4
	vand.vi	v15, v15, 1
	vsrl.vx	v12, v12, t2
	add	a0, s2, t4
	addi	t4, t4, 4
	vsll.vi	v15, v15, 4
	vand.vi	v12, v12, 1
	lbu	a1, 16(a0)
	lbu	a2, 17(a0)
	lbu	a5, 18(a0)
	lbu	t6, 19(a0)
	lbu	s1, 144(a0)
	lbu	a3, 145(a0)
	lbu	s0, 146(a0)
	lbu	a0, 147(a0)
	vsll.vi	v12, v12, 4
	vor.vv	v15, v17, v15
	vor.vv	v12, v14, v12
	vwmacc.vx	v8, a1, v15
	vwmacc.vx	v6, s1, v12
	vwmacc.vx	v3, a2, v15
	vwmacc.vx	v5, a3, v12
	vwmacc.vx	v9, a5, v15
	vwmacc.vx	v7, s0, v12
	vwmacc.vx	v2, t6, v15
	vwmacc.vx	v4, a0, v12
	addi	a4, a4, 16
	beq	t4, ra, .LBB0_13
.LBB0_11:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	a0, a4, t1
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v12, (a4)
	vle8.v	v14, (a0)
	vmv1r.v	v15, v12
	bnez	t0, .LBB0_10
# %bb.12:                               #   in Loop: Header=BB0_11 Depth=3
	vsrl.vx	v15, v12, t3
	j	.LBB0_10
.LBB0_13:                               #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v30, v8
	vwmacc.vv	v22, v30, v9
	ld	a5, 200(sp)                     # 8-byte Folded Reload
	add	a4, a5, t1
	ld	a2, 192(sp)                     # 8-byte Folded Reload
	add	a2, a2, t5
	ld	t4, 168(sp)                     # 8-byte Folded Reload
	add	t4, t4, t5
	vmv1r.v	v9, v16
	vmv1r.v	v1, v16
	vmv1r.v	v15, v16
	vmv1r.v	v0, v16
	vmv1r.v	v14, v16
	vmv1r.v	v12, v16
	vmv1r.v	v17, v16
	vmv1r.v	v8, v16
	j	.LBB0_15
.LBB0_14:                               #   in Loop: Header=BB0_15 Depth=3
	vand.vi	v29, v19, 15
	vsrl.vi	v19, v19, 4
	vand.vi	v28, v28, 1
	vsrl.vx	v18, v18, t2
	lbu	a0, 0(a2)
	lbu	s0, 1(a2)
	lbu	a1, 2(a2)
	lbu	s1, 3(a2)
	lbu	t5, 128(a2)
	lbu	t6, 129(a2)
	lbu	s10, 130(a2)
	lbu	s3, 131(a2)
	addi	a4, a4, 16
	addi	a2, a2, 4
	vsll.vi	v28, v28, 4
	vand.vi	v18, v18, 1
	vsll.vi	v18, v18, 4
	vor.vv	v28, v29, v28
	vor.vv	v18, v19, v18
	vwmacc.vx	v9, a0, v28
	vwmacc.vx	v14, t5, v18
	vwmacc.vx	v1, s0, v28
	vwmacc.vx	v12, t6, v18
	vwmacc.vx	v15, a1, v28
	vwmacc.vx	v17, s10, v18
	vwmacc.vx	v0, s1, v28
	vwmacc.vx	v8, s3, v18
	addi	a5, a5, 16
	beq	a2, t4, .LBB0_17
.LBB0_15:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v18, (a5)
	vle8.v	v19, (a4)
	vmv1r.v	v28, v18
	bnez	t0, .LBB0_14
# %bb.16:                               #   in Loop: Header=BB0_15 Depth=3
	vsrl.vx	v28, v18, t3
	j	.LBB0_14
.LBB0_17:                               #   in Loop: Header=BB0_3 Depth=2
	li	a2, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v30, v3
	vwmacc.vv	v20, v30, v2
	vwmacc.vv	v26, v31, v6
	vwmacc.vv	v22, v31, v7
	addi	t4, t3, 2
	addi	t2, t3, 3
	addi	s10, sp, 440
	vle16.v	v6, (s10)
	addi	t3, sp, 456
	vle16.v	v7, (t3)
	vwmacc.vv	v24, v31, v5
	vwmacc.vv	v20, v31, v4
	vwmacc.vv	v26, v30, v9
	vwmacc.vv	v22, v30, v15
	vwmacc.vv	v26, v31, v14
	vwmacc.vv	v24, v30, v1
	vwmacc.vv	v22, v31, v17
	vwmacc.vv	v20, v30, v0
	vwmacc.vv	v24, v31, v12
	vwmacc.vv	v20, v31, v8
	ld	a4, 184(sp)                     # 8-byte Folded Reload
	add	a4, a4, t1
	ld	s1, 208(sp)                     # 8-byte Folded Reload
	vmv1r.v	v8, v16
	vmv1r.v	v3, v16
	vmv1r.v	v9, v16
	vmv1r.v	v2, v16
	vmv1r.v	v4, v16
	vmv1r.v	v30, v16
	vmv1r.v	v5, v16
	vmv1r.v	v31, v16
.LBB0_18:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v12, (a4)
	vle8.v	v14, (s1)
	add	a0, s2, a2
	addi	a2, a2, 4
	addi	s1, s1, 16
	lbu	a1, 272(a0)
	lbu	s3, 273(a0)
	lbu	t6, 274(a0)
	lbu	t5, 275(a0)
	lbu	a3, 400(a0)
	lbu	s0, 401(a0)
	lbu	a5, 402(a0)
	lbu	a0, 403(a0)
	vand.vi	v15, v12, 15
	vsrl.vi	v12, v12, 4
	vsrl.vx	v17, v14, t4
	vsrl.vx	v14, v14, t2
	vand.vi	v17, v17, 1
	vand.vi	v14, v14, 1
	vsll.vi	v17, v17, 4
	vsll.vi	v14, v14, 4
	vor.vv	v15, v15, v17
	vor.vv	v12, v12, v14
	vwmacc.vx	v8, a1, v15
	vwmacc.vx	v4, a3, v12
	vwmacc.vx	v3, s3, v15
	vwmacc.vx	v30, s0, v12
	vwmacc.vx	v9, t6, v15
	vwmacc.vx	v5, a5, v12
	vwmacc.vx	v2, t5, v15
	vwmacc.vx	v31, a0, v12
	addi	a4, a4, 16
	bne	a2, ra, .LBB0_18
# %bb.19:                               #   in Loop: Header=BB0_3 Depth=2
	li	a2, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v6, v8
	vwmacc.vv	v22, v6, v9
	ld	a4, 200(sp)                     # 8-byte Folded Reload
	vmv1r.v	v0, v16
	vmv1r.v	v14, v16
	vmv1r.v	v15, v16
	vmv1r.v	v8, v16
	vmv1r.v	v1, v16
	vmv1r.v	v9, v16
	vmv1r.v	v17, v16
	vmv1r.v	v12, v16
.LBB0_20:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	a0, a4, t1
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v18, (a4)
	add	a1, s2, a2
	addi	a2, a2, 4
	addi	a0, a0, 512
	lbu	a3, 336(a1)
	lbu	a5, 337(a1)
	lbu	t6, 338(a1)
	lbu	t5, 339(a1)
	vle8.v	v19, (a0)
	lbu	a0, 464(a1)
	lbu	s0, 465(a1)
	lbu	s1, 466(a1)
	lbu	a1, 467(a1)
	vsrl.vx	v28, v18, t4
	vsrl.vx	v18, v18, t2
	vand.vi	v28, v28, 1
	vand.vi	v18, v18, 1
	vand.vi	v29, v19, 15
	vsrl.vi	v19, v19, 4
	vsll.vi	v28, v28, 4
	vsll.vi	v18, v18, 4
	vor.vv	v28, v29, v28
	vor.vv	v18, v19, v18
	vwmacc.vx	v0, a3, v28
	vwmacc.vx	v1, a0, v18
	vwmacc.vx	v14, a5, v28
	vwmacc.vx	v9, s0, v18
	vwmacc.vx	v15, t6, v28
	vwmacc.vx	v17, s1, v18
	vwmacc.vx	v8, t5, v28
	vwmacc.vx	v12, a1, v18
	addi	a4, a4, 16
	bne	a2, ra, .LBB0_20
# %bb.21:                               #   in Loop: Header=BB0_3 Depth=2
	li	a3, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v6, v3
	vwmacc.vv	v20, v6, v2
	vwmacc.vv	v26, v7, v4
	vwmacc.vv	v22, v7, v5
	vwmacc.vv	v24, v7, v30
	vwmacc.vv	v20, v7, v31
	vwmacc.vv	v26, v6, v0
	vwmacc.vv	v22, v6, v15
	vwmacc.vv	v26, v7, v1
	vwmacc.vv	v24, v6, v14
	vwmacc.vv	v22, v7, v17
	vwmacc.vv	v20, v6, v8
	vwmacc.vv	v24, v7, v9
	vwmacc.vv	v20, v7, v12
	li	t2, 1
	mv	t5, a6
	mv	t6, s4
	addi	a6, sp, 360
	mv	s4, s5
	addi	s3, sp, 408
	addi	a5, sp, 344
	ld	s0, 160(sp)                     # 8-byte Folded Reload
	bnez	t0, .LBB0_3
# %bb.22:                               #   in Loop: Header=BB0_2 Depth=1
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	vle16.v	v17, (a0)
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v26, v26
	addi	a1, sp, 480
	vl2r.v	v8, (a1)                        # Unknown-size Folded Reload
	vle32.v	v14, (s9)
	vfcvt.f.x.v	v28, v24
	csrr	a1, vlenb
	slli	a1, a1, 1
	add	a1, a1, sp
	addi	a1, a1, 480
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
	addi	a1, a1, 480
	vl2r.v	v22, (a1)                       # Unknown-size Folded Reload
	vfmul.vf	v30, v18, fa2
	vfmacc.vv	v12, v30, v28
	vle32.v	v28, (s7)
	vfcvt.f.x.v	v20, v20
	csrr	a1, vlenb
	li	a2, 6
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 480
	vl2r.v	v30, (a1)                       # Unknown-size Folded Reload
	vfmul.vf	v6, v18, fa5
	vfmacc.vv	v22, v6, v26
	vle32.v	v26, (s8)
	ld	a3, 120(sp)                     # 8-byte Folded Reload
	addi	a3, a3, 1
	ld	a1, 176(sp)                     # 8-byte Folded Reload
	addi	a1, a1, 1168
	sd	a1, 176(sp)                     # 8-byte Folded Spill
	ld	a4, 48(sp)                      # 8-byte Folded Reload
	ld	a1, 208(sp)                     # 8-byte Folded Reload
	add	a1, a1, a4
	sd	a1, 208(sp)                     # 8-byte Folded Spill
	ld	a1, 200(sp)                     # 8-byte Folded Reload
	add	a1, a1, a4
	sd	a1, 200(sp)                     # 8-byte Folded Spill
	ld	a1, 192(sp)                     # 8-byte Folded Reload
	addi	a1, a1, 1168
	sd	a1, 192(sp)                     # 8-byte Folded Spill
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
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	add	a0, a0, a4
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	ld	a1, 104(sp)                     # 8-byte Folded Reload
	ld	s1, 96(sp)                      # 8-byte Folded Reload
	bne	a3, a0, .LBB0_2
# %bb.23:
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	ld	a2, 32(sp)                      # 8-byte Folded Reload
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_25
.LBB0_24:
	slliw	a0, a5, 3
	vmv2r.v	v14, v8
	vmv2r.v	v18, v8
	vmv2r.v	v20, v8
.LBB0_25:
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
	li	a1, 10
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 592
	ld	ra, 584(sp)                     # 8-byte Folded Reload
	ld	s0, 576(sp)                     # 8-byte Folded Reload
	ld	s1, 568(sp)                     # 8-byte Folded Reload
	ld	s2, 560(sp)                     # 8-byte Folded Reload
	ld	s3, 552(sp)                     # 8-byte Folded Reload
	ld	s4, 544(sp)                     # 8-byte Folded Reload
	ld	s5, 536(sp)                     # 8-byte Folded Reload
	ld	s6, 528(sp)                     # 8-byte Folded Reload
	ld	s7, 520(sp)                     # 8-byte Folded Reload
	ld	s8, 512(sp)                     # 8-byte Folded Reload
	ld	s9, 504(sp)                     # 8-byte Folded Reload
	ld	s10, 496(sp)                    # 8-byte Folded Reload
	ld	s11, 488(sp)                    # 8-byte Folded Reload
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
	addi	sp, sp, 592
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	strip_body, .Lfunc_end0-strip_body
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
