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
	beqz	a3, .LBB0_20
# %bb.1:
	mv	s1, a0
	sd	a2, 32(sp)                      # 8-byte Folded Spill
	sd	a4, 40(sp)                      # 8-byte Folded Spill
	li	a4, 0
	slliw	a2, a5, 3
	addi	s5, sp, 456
	addi	t0, sp, 392
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	addi	t4, sp, 424
	addi	s2, sp, 360
	addi	s8, sp, 440
	addi	a6, sp, 376
	slliw	a0, a5, 4
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	addi	s3, sp, 248
	addi	ra, sp, 280
	addi	a0, a1, 80
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	li	a0, 11
	addi	s6, sp, 408
	addi	a5, sp, 344
	addi	t6, sp, 216
	li	s11, 16
	li	s7, 64
	vmv2r.v	v20, v8
	vmv2r.v	v16, v8
	vmv2r.v	v12, v8
	add	a3, a2, s1
	slli	s0, a0, 8
	addi	a0, a3, 256
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 512
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	addi	a0, a3, 768
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	addi	a0, a2, 64
	sd	a0, 128(sp)                     # 8-byte Folded Spill
	addiw	a0, a2, 192
	sd	a0, 72(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 208
	sd	a0, 64(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 224
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	sd	a2, 24(sp)                      # 8-byte Folded Spill
	addiw	a0, a2, 240
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	sd	a1, 176(sp)                     # 8-byte Folded Spill
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	sd	s1, 96(sp)                      # 8-byte Folded Spill
	sd	s0, 80(sp)                      # 8-byte Folded Spill
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
                                        #       Child Loop BB0_10 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_14 Depth 3
                                        #       Child Loop BB0_16 Depth 3
	addi	a0, sp, 480
	vs2r.v	v20, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 480
	vs2r.v	v16, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 480
	vs2r.v	v12, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 6
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 480
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	li	t3, 0
	li	a0, 1168
	mul	s4, a4, a0
	sd	a4, 120(sp)                     # 8-byte Folded Spill
	mul	a7, a4, s0
	add	s4, s4, a1
	add	a7, a7, s1
	addi	a0, s4, 144
	sd	a0, 168(sp)                     # 8-byte Folded Spill
	flw	fa3, 0(s4)
	flw	fa2, 4(s4)
	flw	fa5, 8(s4)
	flw	fa4, 12(s4)
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	addi	s9, s4, 1040
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v26, (a0)
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 152(sp)                     # 8-byte Folded Spill
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 144(sp)                     # 8-byte Folded Spill
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	add	a0, a0, a7
	sd	a0, 136(sp)                     # 8-byte Folded Spill
	vand.vi	v8, v26, 3
	vand.vi	v9, v26, 12
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
	li	a0, 1
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v20, 0
	vmv.v.i	v28, 0
	vmv.v.i	v24, 0
	vmv.v.i	v22, 0
	sd	s9, 160(sp)                     # 8-byte Folded Spill
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_10 Depth 3
                                        #       Child Loop BB0_12 Depth 3
                                        #       Child Loop BB0_14 Depth 3
                                        #       Child Loop BB0_16 Depth 3
	slli	a1, t3, 6
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	addw	a3, a3, a1
	add	a1, a7, a3
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a1)
	andi	t1, a0, 1
	addiw	a2, a3, 16
	addiw	a1, a3, 32
	vand.vi	v9, v8, 15
	vsrl.vi	v8, v8, 4
	addiw	a0, a3, 48
	beqz	t1, .LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=2
	csrr	a3, vlenb
	slli	a4, a3, 3
	add	a3, a3, a4
	add	a3, a3, sp
	addi	a3, a3, 480
	vl1r.v	v10, (a3)                       # Unknown-size Folded Reload
	vor.vv	v9, v10, v9
	csrr	a3, vlenb
	slli	a3, a3, 3
	add	a3, a3, sp
	addi	a3, a3, 480
	vl1r.v	v10, (a3)                       # Unknown-size Folded Reload
	vor.vv	v8, v10, v8
	add	a2, a2, a7
	ld	a3, 152(sp)                     # 8-byte Folded Reload
	vle8.v	v10, (a3)
	add	a1, a1, a7
	ld	a3, 144(sp)                     # 8-byte Folded Reload
	vle8.v	v11, (a3)
	add	a0, a0, a7
	ld	a3, 136(sp)                     # 8-byte Folded Reload
	vle8.v	v14, (a3)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v9
	vzext.vf2	v9, v8
	vle8.v	v8, (a2)
	vle8.v	v16, (a1)
	vle8.v	v17, (a0)
	vse16.v	v15, (s6)
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
	vse16.v	v16, (t4)
	vzext.vf2	v15, v8
	vse16.v	v15, (s2)
	vzext.vf2	v8, v10
	vse16.v	v8, (s8)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v14, 12
	vsrl.vi	v10, v17, 4
	vsll.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	vse16.v	v14, (a6)
	j	.LBB0_6
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=2
	li	a3, 48
	vand.vx	v10, v26, a3
	li	a4, -64
	vand.vx	v11, v26, a4
	add	a2, a2, a7
	ld	s1, 152(sp)                     # 8-byte Folded Reload
	vle8.v	v14, (s1)
	add	a1, a1, a7
	ld	s1, 144(sp)                     # 8-byte Folded Reload
	vle8.v	v15, (s1)
	add	a0, a0, a7
	ld	s1, 136(sp)                     # 8-byte Folded Reload
	vle8.v	v16, (s1)
	vsrl.vi	v11, v11, 2
	vor.vv	v9, v10, v9
	vle8.v	v10, (a2)
	vle8.v	v17, (a1)
	vor.vv	v8, v11, v8
	vle8.v	v11, (a0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v9
	vzext.vf2	v9, v8
	vse16.v	v19, (s6)
	vse16.v	v9, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v14, a3
	vand.vx	v9, v14, a4
	vand.vi	v14, v10, 15
	vor.vv	v8, v8, v14
	vand.vx	v14, v15, a3
	vand.vx	v15, v15, a4
	vsrl.vi	v10, v10, 4
	vsrl.vi	v9, v9, 2
	vor.vv	v10, v9, v10
	vand.vi	v9, v17, 15
	vor.vv	v14, v14, v9
	vand.vx	v9, v16, a3
	vsrl.vi	v17, v17, 4
	vsrl.vi	v15, v15, 2
	vor.vv	v15, v15, v17
	vand.vi	v17, v11, 15
	vor.vv	v9, v9, v17
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v8
	vse16.v	v17, (t4)
	vzext.vf2	v8, v10
	vse16.v	v8, (s2)
	vzext.vf2	v8, v14
	vse16.v	v8, (s8)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v8, v16, a4
	vsrl.vi	v10, v11, 4
	vsrl.vi	v8, v8, 2
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v15
	vse16.v	v11, (a6)
.LBB0_6:                                #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e8, mf2, ta, ma
	vor.vv	v8, v8, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v8
	vse16.v	v10, (t0)
	vle16.v	v8, (a5)
	vzext.vf2	v10, v9
	vse16.v	v10, (s5)
	slli	t2, t3, 6
	addi	s0, sp, 312
	beqz	t1, .LBB0_8
# %bb.7:                                #   in Loop: Header=BB0_3 Depth=2
	lh	a0, 1040(s4)
	lh	a1, 1042(s4)
	lh	a2, 1044(s4)
	lh	t5, 1046(s4)
	lh	a4, 1048(s4)
	lh	a5, 1050(s4)
	lh	s1, 1052(s4)
	lh	a3, 1054(s4)
	add	a0, a0, a4
	add	a1, a1, a5
	add	a2, a2, s1
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v12, 0
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vx	v10, a0, v8
	vse32.v	v10, (t6)
	vmv2r.v	v10, v12
	vwmacc.vx	v10, a1, v8
	vse32.v	v10, (s3)
	vmv2r.v	v10, v12
	vwmacc.vx	v10, a2, v8
	vse32.v	v10, (ra)
	vmv2r.v	v10, v12
	add	a3, a3, t5
	vwmacc.vx	v10, a3, v8
	j	.LBB0_9
.LBB0_8:                                #   in Loop: Header=BB0_3 Depth=2
	add	a1, s9, t2
	vle32.v	v10, (t6)
	lh	t5, 0(a1)
	lh	a3, 2(a1)
	lh	a4, 4(a1)
	lh	a5, 6(a1)
	lh	s1, 8(a1)
	lh	a0, 10(a1)
	lh	a2, 12(a1)
	lh	a1, 14(a1)
	add	t5, t5, s1
	vwmacc.vx	v10, t5, v8
	vse32.v	v10, (t6)
	vle32.v	v10, (s3)
	add	a0, a0, a3
	vwmacc.vx	v10, a0, v8
	vse32.v	v10, (s3)
	vle32.v	v10, (ra)
	add	a2, a2, a4
	vwmacc.vx	v10, a2, v8
	vse32.v	v10, (ra)
	vle32.v	v10, (s0)
	add	a1, a1, a5
	vwmacc.vx	v10, a1, v8
.LBB0_9:                                #   in Loop: Header=BB0_3 Depth=2
	li	t5, 0
	vse32.v	v10, (s0)
	vle16.v	v9, (s2)
	add	a0, s9, t2
	vle32.v	v10, (t6)
	lh	a1, 16(a0)
	lh	a2, 18(a0)
	lh	a3, 20(a0)
	lh	t2, 22(a0)
	lh	a5, 24(a0)
	lh	s1, 26(a0)
	lh	a4, 28(a0)
	lh	s2, 30(a0)
	add	a1, a1, a5
	vwmacc.vx	v10, a1, v9
	vse32.v	v10, (t6)
	vle32.v	v10, (s3)
	add	a2, a2, s1
	vwmacc.vx	v10, a2, v9
	vse32.v	v10, (s3)
	vle32.v	v10, (ra)
	vle16.v	v8, (a6)
	add	a3, a3, a4
	vwmacc.vx	v10, a3, v9
	vse32.v	v10, (ra)
	vle32.v	v10, (s0)
	lh	a1, 32(a0)
	lh	a2, 34(a0)
	lh	a3, 36(a0)
	lh	a4, 38(a0)
	add	t2, t2, s2
	vwmacc.vx	v10, t2, v9
	vse32.v	v10, (s0)
	vle32.v	v10, (t6)
	lh	a5, 40(a0)
	lh	s1, 42(a0)
	lh	t2, 44(a0)
	lh	s2, 46(a0)
	add	a1, a1, a5
	vwmacc.vx	v10, a1, v8
	vse32.v	v10, (t6)
	vle32.v	v10, (s3)
	add	a2, a2, s1
	vwmacc.vx	v10, a2, v8
	vse32.v	v10, (s3)
	vle32.v	v10, (ra)
	mv	a6, t0
	vle16.v	v9, (t0)
	add	a3, a3, t2
	vwmacc.vx	v10, a3, v8
	vse32.v	v10, (ra)
	vle32.v	v10, (s0)
	lh	t2, 48(a0)
	lh	a2, 50(a0)
	lh	a3, 52(a0)
	lh	a5, 54(a0)
	add	a4, a4, s2
	vwmacc.vx	v10, a4, v8
	vse32.v	v10, (s0)
	vle32.v	v10, (t6)
	lh	a4, 56(a0)
	lh	s1, 58(a0)
	lh	a1, 60(a0)
	lh	s2, 62(a0)
	add	a4, a4, t2
	vwmacc.vx	v10, a4, v9
	vse32.v	v10, (t6)
	vle32.v	v10, (s3)
	add	a2, a2, s1
	slli	t2, t3, 10
	vwmacc.vx	v10, a2, v9
	slli	t0, t3, 9
	slli	t3, t3, 2
	mv	s9, s3
	vse32.v	v10, (s3)
	vle32.v	v10, (ra)
	vle16.v	v6, (s6)
	mv	s6, t4
	vle16.v	v31, (t4)
	add	a1, a1, a3
	vwmacc.vx	v10, a1, v9
	mv	s10, ra
	vse32.v	v10, (ra)
	vle32.v	v10, (s0)
	li	a0, 1
	sll	ra, a0, t3
	li	a0, 2
	sll	s5, a0, t3
	add	a5, a5, s2
	vwmacc.vx	v10, a5, v9
	vse32.v	v10, (s0)
	ld	s2, 176(sp)                     # 8-byte Folded Reload
	add	s2, s2, t0
	ld	a3, 208(sp)                     # 8-byte Folded Reload
	vmv1r.v	v9, v18
	vmv1r.v	v2, v18
	vmv1r.v	v10, v18
	vmv1r.v	v1, v18
	vmv1r.v	v5, v18
	vmv1r.v	v4, v18
	vmv1r.v	v7, v18
	vmv1r.v	v3, v18
.LBB0_10:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	a0, a3, t2
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v8, (a3)
	add	a1, s2, t5
	addi	t5, t5, 4
	vle8.v	v11, (a0)
	lbu	a0, 16(a1)
	lbu	a5, 17(a1)
	lbu	s1, 18(a1)
	lbu	s0, 19(a1)
	lbu	a4, 144(a1)
	lbu	a2, 145(a1)
	lbu	t4, 146(a1)
	lbu	a1, 147(a1)
	vand.vx	v12, v8, ra
	vand.vx	v8, v8, s5
	vand.vi	v13, v11, 15
	vsrl.vi	v11, v11, 4
	vmsne.vi	v0, v12, 0
	vmsne.vi	v8, v8, 0
	vadd.vx	v13, v13, s11, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v11, v11, s11, v0.t
	vwmacc.vx	v9, a0, v13
	vwmacc.vx	v2, a5, v13
	vwmacc.vx	v10, s1, v13
	vwmacc.vx	v1, s0, v13
	vwmacc.vx	v5, a4, v11
	vwmacc.vx	v4, a2, v11
	vwmacc.vx	v7, t4, v11
	vwmacc.vx	v3, a1, v11
	addi	a3, a3, 16
	bne	t5, s7, .LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_3 Depth=2
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v28, v6, v9
	vwmacc.vv	v22, v6, v10
	ld	s1, 200(sp)                     # 8-byte Folded Reload
	add	a3, s1, t2
	ld	a5, 192(sp)                     # 8-byte Folded Reload
	add	a5, a5, t0
	ld	t5, 168(sp)                     # 8-byte Folded Reload
	add	t5, t5, t0
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
	vle8.v	v8, (a3)
	vle8.v	v0, (s1)
	lbu	s0, 0(a5)
	lbu	a1, 1(a5)
	lbu	a0, 2(a5)
	lbu	a4, 3(a5)
	lbu	t0, 128(a5)
	lbu	s3, 129(a5)
	lbu	t4, 130(a5)
	lbu	s8, 131(a5)
	addi	a3, a3, 16
	addi	a5, a5, 4
	vand.vi	v27, v8, 15
	vsrl.vi	v30, v8, 4
	vand.vx	v8, v0, ra
	vand.vx	v12, v0, s5
	vmsne.vi	v0, v8, 0
	vmsne.vi	v8, v12, 0
	vadd.vx	v27, v27, s11, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v30, v30, s11, v0.t
	vwmacc.vx	v15, s0, v27
	vwmacc.vx	v9, a1, v27
	vwmacc.vx	v17, a0, v27
	vwmacc.vx	v11, a4, v27
	vwmacc.vx	v16, t0, v30
	vwmacc.vx	v10, s3, v30
	vwmacc.vx	v19, t4, v30
	vwmacc.vx	v14, s8, v30
	addi	s1, s1, 16
	bne	a5, t5, .LBB0_12
# %bb.13:                               #   in Loop: Header=BB0_3 Depth=2
	li	t4, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v6, v2
	vwmacc.vv	v20, v6, v1
	vwmacc.vv	v28, v31, v5
	vwmacc.vv	v22, v31, v7
	li	a0, 4
	sll	t0, a0, t3
	li	a0, 8
	sll	t3, a0, t3
	addi	s8, sp, 440
	vle16.v	v5, (s8)
	addi	s5, sp, 456
	vle16.v	v7, (s5)
	vwmacc.vv	v24, v31, v4
	vwmacc.vv	v20, v31, v3
	vwmacc.vv	v28, v6, v15
	vwmacc.vv	v22, v6, v17
	vwmacc.vv	v28, v31, v16
	vwmacc.vv	v24, v6, v9
	vwmacc.vv	v22, v31, v19
	vwmacc.vv	v20, v6, v11
	vwmacc.vv	v24, v31, v10
	vwmacc.vv	v20, v31, v14
	ld	a5, 184(sp)                     # 8-byte Folded Reload
	add	a5, a5, t2
	ld	a3, 208(sp)                     # 8-byte Folded Reload
	vmv1r.v	v9, v18
	vmv1r.v	v2, v18
	vmv1r.v	v10, v18
	vmv1r.v	v1, v18
	vmv1r.v	v3, v18
	vmv1r.v	v31, v18
	vmv1r.v	v4, v18
	vmv1r.v	v6, v18
.LBB0_14:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v8, (a5)
	vle8.v	v11, (a3)
	add	a1, s2, t4
	addi	t4, t4, 4
	addi	a3, a3, 16
	lbu	s1, 272(a1)
	lbu	s0, 273(a1)
	lbu	a0, 274(a1)
	lbu	a2, 275(a1)
	lbu	a4, 400(a1)
	lbu	t5, 401(a1)
	lbu	s3, 402(a1)
	lbu	a1, 403(a1)
	vand.vi	v12, v8, 15
	vsrl.vi	v13, v8, 4
	vand.vx	v8, v11, t0
	vand.vx	v11, v11, t3
	vmsne.vi	v0, v8, 0
	vmsne.vi	v8, v11, 0
	vadd.vx	v12, v12, s11, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v13, v13, s11, v0.t
	vwmacc.vx	v9, s1, v12
	vwmacc.vx	v2, s0, v12
	vwmacc.vx	v10, a0, v12
	vwmacc.vx	v1, a2, v12
	vwmacc.vx	v3, a4, v13
	vwmacc.vx	v31, t5, v13
	vwmacc.vx	v4, s3, v13
	vwmacc.vx	v6, a1, v13
	addi	a5, a5, 16
	bne	t4, s7, .LBB0_14
# %bb.15:                               #   in Loop: Header=BB0_3 Depth=2
	li	a4, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v28, v5, v9
	vwmacc.vv	v22, v5, v10
	ld	a5, 200(sp)                     # 8-byte Folded Reload
	vmv1r.v	v19, v18
	vmv1r.v	v14, v18
	vmv1r.v	v15, v18
	vmv1r.v	v9, v18
	vmv1r.v	v17, v18
	vmv1r.v	v10, v18
	vmv1r.v	v16, v18
	vmv1r.v	v11, v18
	mv	s3, s9
	mv	ra, s10
.LBB0_16:                               #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_3 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	add	a0, a5, t2
	vsetvli	zero, zero, e8, mf2, ta, mu
	vle8.v	v8, (a5)
	add	a1, s2, a4
	addi	a4, a4, 4
	addi	a0, a0, 512
	lbu	a2, 336(a1)
	lbu	a3, 337(a1)
	lbu	t5, 338(a1)
	lbu	t4, 339(a1)
	vle8.v	v12, (a0)
	lbu	a0, 464(a1)
	lbu	s0, 465(a1)
	lbu	s1, 466(a1)
	lbu	a1, 467(a1)
	vand.vx	v13, v8, t0
	vand.vx	v8, v8, t3
	vmsne.vi	v0, v13, 0
	vmsne.vi	v8, v8, 0
	vand.vi	v13, v12, 15
	vsrl.vi	v12, v12, 4
	vadd.vx	v13, v13, s11, v0.t
	vmv1r.v	v0, v8
	vadd.vx	v12, v12, s11, v0.t
	vwmacc.vx	v19, a2, v13
	vwmacc.vx	v17, a0, v12
	vwmacc.vx	v14, a3, v13
	vwmacc.vx	v10, s0, v12
	vwmacc.vx	v15, t5, v13
	vwmacc.vx	v16, s1, v12
	vwmacc.vx	v9, t4, v13
	vwmacc.vx	v11, a1, v12
	addi	a5, a5, 16
	bne	a4, s7, .LBB0_16
# %bb.17:                               #   in Loop: Header=BB0_3 Depth=2
	li	a0, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v24, v5, v2
	vwmacc.vv	v20, v5, v1
	vwmacc.vv	v28, v7, v3
	vwmacc.vv	v22, v7, v4
	vwmacc.vv	v24, v7, v31
	vwmacc.vv	v20, v7, v6
	vwmacc.vv	v28, v5, v19
	vwmacc.vv	v22, v5, v15
	vwmacc.vv	v28, v7, v17
	vwmacc.vv	v24, v5, v14
	vwmacc.vv	v22, v7, v16
	vwmacc.vv	v20, v5, v9
	vwmacc.vv	v24, v7, v10
	vwmacc.vv	v20, v7, v11
	li	t3, 1
	mv	t0, a6
	mv	t4, s6
	addi	s2, sp, 360
	addi	a6, sp, 376
	addi	s6, sp, 408
	addi	a5, sp, 344
	ld	s9, 160(sp)                     # 8-byte Folded Reload
	bnez	t1, .LBB0_3
# %bb.18:                               #   in Loop: Header=BB0_2 Depth=1
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	add	a7, a7, a0
	vle16.v	v19, (a7)
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v26, v28
	addi	a0, sp, 480
	vl2r.v	v8, (a0)                        # Unknown-size Folded Reload
	vle32.v	v14, (t6)
	vfcvt.f.x.v	v28, v24
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 480
	vl2r.v	v10, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v16, v19
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v24, v16, fa3
	vfmacc.vv	v8, v24, v26
	vle32.v	v24, (s3)
	vfcvt.f.x.v	v26, v22
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 480
	vl2r.v	v22, (a0)                       # Unknown-size Folded Reload
	vfmul.vf	v30, v16, fa2
	vfmacc.vv	v10, v30, v28
	vle32.v	v28, (ra)
	vfcvt.f.x.v	v20, v20
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 480
	vl2r.v	v30, (a0)                       # Unknown-size Folded Reload
	vfmul.vf	v6, v16, fa5
	vfmacc.vv	v22, v6, v26
	addi	a0, sp, 312
	vle32.v	v26, (a0)
	ld	a4, 120(sp)                     # 8-byte Folded Reload
	addi	a4, a4, 1
	ld	a0, 176(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	ld	s0, 80(sp)                      # 8-byte Folded Reload
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	add	a0, a0, s0
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	add	a0, a0, s0
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	addi	a0, a7, 32
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
	vfmul.vf	v12, v6, fa5
	vfmul.vf	v8, v6, fa4
	vfnmsub.vv	v16, v24, v10
	vfnmsub.vv	v12, v28, v22
	vfnmsub.vv	v8, v26, v30
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	add	a0, a0, s0
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	ld	a0, 112(sp)                     # 8-byte Folded Reload
	ld	a1, 104(sp)                     # 8-byte Folded Reload
	ld	s1, 96(sp)                      # 8-byte Folded Reload
	bne	a4, a0, .LBB0_2
# %bb.19:
	ld	a4, 40(sp)                      # 8-byte Folded Reload
	ld	a2, 32(sp)                      # 8-byte Folded Reload
	ld	a0, 24(sp)                      # 8-byte Folded Reload
	j	.LBB0_21
.LBB0_20:
	slliw	a0, a5, 3
	vmv2r.v	v12, v8
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
	vse32.v	v12, (a2)
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
