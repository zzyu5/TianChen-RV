	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"old.c"
	.text
	.globl	weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K # -- Begin function weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K
	.p2align	1
	.type	weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K,@function
weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K: # @weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -496
	.cfi_def_cfa_offset 496
	sd	ra, 488(sp)                     # 8-byte Folded Spill
	sd	s0, 480(sp)                     # 8-byte Folded Spill
	sd	s1, 472(sp)                     # 8-byte Folded Spill
	sd	s2, 464(sp)                     # 8-byte Folded Spill
	sd	s3, 456(sp)                     # 8-byte Folded Spill
	sd	s4, 448(sp)                     # 8-byte Folded Spill
	sd	s5, 440(sp)                     # 8-byte Folded Spill
	sd	s6, 432(sp)                     # 8-byte Folded Spill
	sd	s7, 424(sp)                     # 8-byte Folded Spill
	sd	s8, 416(sp)                     # 8-byte Folded Spill
	sd	s9, 408(sp)                     # 8-byte Folded Spill
	sd	s10, 400(sp)                    # 8-byte Folded Spill
	sd	s11, 392(sp)                    # 8-byte Folded Spill
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
	addi	sp, sp, -1616
	.cfi_def_cfa_offset 2112
	csrr	a5, vlenb
	li	a6, 20
	mul	a5, a5, a6
	sub	sp, sp, a5
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xc0, 0x10, 0x22, 0x11, 0x14, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 2112 + 20 * vlenb
	sd	a2, 24(sp)                      # 8-byte Folded Spill
	sd	a1, 40(sp)                      # 8-byte Folded Spill
	srli	a4, a4, 4
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	sd	a4, 32(sp)                      # 8-byte Folded Spill
	bnez	a4, .LBB0_1
	j	.LBB0_6
.LBB0_1:
	sd	zero, 56(sp)                    # 8-byte Folded Spill
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	srli	a4, a0, 8
	li	a0, 9
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v29, 0
	li	a5, 292
	slli	s1, a0, 8
	mul	a0, a4, s1
	sd	a0, 16(sp)                      # 8-byte Folded Spill
	sd	a3, 80(sp)                      # 8-byte Folded Spill
	sd	a4, 72(sp)                      # 8-byte Folded Spill
	sd	s1, 64(sp)                      # 8-byte Folded Spill
	j	.LBB0_3
.LBB0_2:                                #   in Loop: Header=BB0_3 Depth=1
	ld	a2, 56(sp)                      # 8-byte Folded Reload
	slli	a0, a2, 6
	ld	a1, 40(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	vsetivli	zero, 8, e32, m2, ta, ma
	vse32.v	v14, (a0)
	addi	a0, a0, 32
	addi	a2, a2, 1
	vse32.v	v12, (a0)
	vmv.v.i	v8, 0
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	sd	a2, 56(sp)                      # 8-byte Folded Spill
	bne	a2, a0, .LBB0_3
	j	.LBB0_6
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
	vsetivli	zero, 1, e8, m1, ta, ma
	vmv2r.v	v12, v8
	vmv2r.v	v14, v8
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	li	a1, 256
	bltu	a0, a1, .LBB0_2
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	li	a2, 0
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	ld	a1, 16(sp)                      # 8-byte Folded Reload
	mul	a0, a1, a0
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	vmv.v.i	v10, 0
.LBB0_5:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	sd	a2, 1952(sp)                    # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs2r.v	v10, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	slli	a1, a0, 4
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	mul	s0, a2, s1
	mul	s1, a2, a5
	vmv1r.v	v20, v29
	vmv1r.v	v21, v29
	vmv1r.v	v23, v29
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	add	s0, s0, a0
	add	s1, s1, a3
	addi	a0, s0, 64
	addi	a1, s0, 192
	addi	a2, s0, 80
	addi	a3, s0, 208
	addi	s9, s0, 96
	addi	t0, s0, 256
	addi	a6, s0, 272
	addi	a7, s0, 288
	lbu	t4, 4(s1)
	sd	t4, 1912(sp)                    # 8-byte Folded Spill
	lbu	s2, 5(s1)
	sd	s2, 1872(sp)                    # 8-byte Folded Spill
	lbu	s11, 6(s1)
	sd	s11, 1896(sp)                   # 8-byte Folded Spill
	lbu	t5, 36(s1)
	sd	t5, 1936(sp)                    # 8-byte Folded Spill
	lbu	t6, 37(s1)
	sd	t6, 1888(sp)                    # 8-byte Folded Spill
	lbu	s10, 38(s1)
	sd	s10, 1928(sp)                   # 8-byte Folded Spill
	addi	a5, s0, 304
	addi	t1, s0, 320
	addi	t2, s0, 336
	addi	t3, s0, 352
	lbu	ra, 7(s1)
	sd	ra, 1920(sp)                    # 8-byte Folded Spill
	lbu	s7, 8(s1)
	sd	s7, 1856(sp)                    # 8-byte Folded Spill
	lbu	s4, 9(s1)
	sd	s4, 1864(sp)                    # 8-byte Folded Spill
	lbu	s6, 10(s1)
	sd	s6, 1816(sp)                    # 8-byte Folded Spill
	lbu	s3, 39(s1)
	sd	s3, 1840(sp)                    # 8-byte Folded Spill
	lbu	s8, 40(s1)
	sd	s8, 1848(sp)                    # 8-byte Folded Spill
	lbu	s5, 41(s1)
	sd	s5, 1808(sp)                    # 8-byte Folded Spill
	lbu	a4, 42(s1)
	sd	a4, 1960(sp)                    # 8-byte Folded Spill
	vle8.v	v28, (a0)
	csrr	a0, vlenb
	li	a4, 14
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs1r.v	v28, (a0)                       # Unknown-size Folded Spill
	addi	a0, s0, 368
	vle8.v	v31, (a1)
	csrr	a1, vlenb
	li	a4, 18
	mul	a1, a1, a4
	add	a1, a1, sp
	addi	a1, a1, 2000
	vs1r.v	v31, (a1)                       # Unknown-size Folded Spill
	addi	a1, s0, 384
	vle8.v	v6, (a2)
	csrr	a2, vlenb
	li	a4, 13
	mul	a2, a2, a4
	add	a2, a2, sp
	addi	a2, a2, 2000
	vs1r.v	v6, (a2)                        # Unknown-size Folded Spill
	addi	a2, s0, 400
	vle8.v	v7, (a3)
	csrr	a3, vlenb
	slli	a4, a3, 4
	add	a3, a3, a4
	add	a3, a3, sp
	addi	a3, a3, 2000
	vs1r.v	v7, (a3)                        # Unknown-size Folded Spill
	addi	a3, s0, 416
	vle8.v	v10, (s9)
	csrr	a4, vlenb
	li	s9, 19
	mul	a4, a4, s9
	add	a4, a4, sp
	addi	a4, a4, 2000
	vs1r.v	v10, (a4)                       # Unknown-size Folded Spill
	vle8.v	v14, (t0)
	vle8.v	v10, (a6)
	vle8.v	v17, (a7)
	lbu	s9, 11(s1)
	sd	s9, 1800(sp)                    # 8-byte Folded Spill
	lbu	a4, 12(s1)
	sd	a4, 1968(sp)                    # 8-byte Folded Spill
	lbu	a4, 13(s1)
	sd	a4, 1976(sp)                    # 8-byte Folded Spill
	lbu	a4, 14(s1)
	sd	a4, 1984(sp)                    # 8-byte Folded Spill
	vle8.v	v18, (a5)
	addi	t0, s0, 432
	vle8.v	v12, (t1)
	addi	a4, s0, 448
	vle8.v	v25, (t2)
	addi	a5, s0, 464
	vle8.v	v27, (t3)
	addi	t1, s0, 480
	vle8.v	v11, (a0)
	addi	a0, s0, 496
	vle8.v	v15, (a1)
	addi	t2, s0, 512
	vle8.v	v26, (a2)
	addi	a7, s0, 528
	vle8.v	v24, (a3)
	addi	a6, s0, 544
	vle8.v	v8, (t0)
	vle8.v	v9, (a4)
	vle8.v	v16, (a5)
	lbu	a2, 20(s1)
	sd	a2, 1944(sp)                    # 8-byte Folded Spill
	lbu	a1, 21(s1)
	sd	a1, 1880(sp)                    # 8-byte Folded Spill
	lbu	a5, 22(s1)
	sd	a5, 1904(sp)                    # 8-byte Folded Spill
	vle8.v	v19, (t1)
	addi	a3, s0, 560
	vle8.v	v22, (a0)
	addi	a4, s0, 576
	vle8.v	v13, (t2)
	addi	a0, s0, 592
	vle8.v	v30, (a7)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v28, v28, 15
	vand.vi	v29, v14, 15
	vwmacc.vx	v20, t4, v29
	vand.vi	v29, v31, 3
	vsrl.vi	v14, v14, 4
	vwmacc.vx	v21, t5, v14
	vand.vi	v14, v13, 15
	vwmacc.vx	v23, a2, v14
	vand.vi	v14, v6, 15
	vsll.vi	v29, v29, 4
	vor.vv	v31, v29, v28
	vand.vi	v28, v7, 3
	vsll.vi	v28, v28, 4
	vor.vv	v6, v28, v14
	vand.vi	v14, v10, 15
	vwmacc.vx	v20, s2, v14
	vle8.v	v29, (a6)
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v21, t6, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v23, a1, v10
	vand.vi	v10, v17, 15
	vwmacc.vx	v20, s11, v10
	vle8.v	v10, (a3)
	vsrl.vi	v14, v17, 4
	vwmacc.vx	v21, s10, v14
	vand.vi	v14, v29, 15
	vwmacc.vx	v23, a5, v14
	vand.vi	v14, v18, 15
	vwmacc.vx	v20, ra, v14
	vle8.v	v17, (a4)
	lbu	a3, 23(s1)
	sd	a3, 1824(sp)                    # 8-byte Folded Spill
	lbu	a1, 24(s1)
	sd	a1, 1832(sp)                    # 8-byte Folded Spill
	lbu	a4, 25(s1)
	lbu	a2, 26(s1)
	sd	a2, 1792(sp)                    # 8-byte Folded Spill
	vsrl.vi	v14, v18, 4
	vwmacc.vx	v21, s3, v14
	vand.vi	v14, v10, 15
	vwmacc.vx	v23, a3, v14
	vand.vi	v14, v12, 15
	vwmacc.vx	v20, s7, v14
	vle8.v	v18, (a0)
	addi	a0, s0, 608
	vsrl.vi	v12, v12, 4
	vwmacc.vx	v21, s8, v12
	vand.vi	v12, v17, 15
	vwmacc.vx	v23, a1, v12
	vand.vi	v12, v25, 15
	vwmacc.vx	v20, s4, v12
	vle8.v	v12, (a0)
	addi	a0, s0, 624
	vsrl.vi	v14, v25, 4
	vwmacc.vx	v21, s5, v14
	vand.vi	v14, v18, 15
	vwmacc.vx	v23, a4, v14
	vand.vi	v14, v27, 15
	vwmacc.vx	v20, s6, v14
	vle8.v	v25, (a0)
	addi	a0, s0, 640
	vsrl.vi	v14, v27, 4
	ld	a1, 1960(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v21, a1, v14
	vand.vi	v14, v12, 15
	vwmacc.vx	v23, a2, v14
	vand.vi	v14, v11, 15
	vwmacc.vx	v20, s9, v14
	vle8.v	v28, (a0)
	vsrl.vi	v11, v11, 4
	lbu	a0, 43(s1)
	sd	a0, 1784(sp)                    # 8-byte Folded Spill
	lbu	a3, 44(s1)
	sd	a3, 1768(sp)                    # 8-byte Folded Spill
	lbu	a7, 45(s1)
	sd	a7, 1776(sp)                    # 8-byte Folded Spill
	lbu	a6, 46(s1)
	sd	a6, 1736(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v21, a0, v11
	vand.vi	v11, v25, 15
	lbu	a0, 27(s1)
	sd	a0, 1752(sp)                    # 8-byte Folded Spill
	lbu	a5, 28(s1)
	sd	a5, 1760(sp)                    # 8-byte Folded Spill
	lbu	a1, 29(s1)
	sd	a1, 1744(sp)                    # 8-byte Folded Spill
	lbu	a2, 30(s1)
	sd	a2, 1728(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a0, v11
	vand.vi	v11, v15, 15
	ld	a0, 1968(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a0, v11
	addi	a0, s0, 656
	vle8.v	v0, (a0)
	vsrl.vi	v11, v15, 4
	vwmacc.vx	v21, a3, v11
	vand.vi	v11, v28, 15
	vwmacc.vx	v23, a5, v11
	vand.vi	v11, v26, 15
	ld	a0, 1976(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a0, v11
	addi	a0, s0, 672
	vle8.v	v7, (a0)
	vsrl.vi	v11, v26, 4
	vwmacc.vx	v21, a7, v11
	vand.vi	v11, v0, 15
	vwmacc.vx	v23, a1, v11
	vand.vi	v11, v24, 15
	ld	a0, 1984(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a0, v11
	addi	a0, s0, 688
	vle8.v	v5, (a0)
	vsrl.vi	v11, v24, 4
	vwmacc.vx	v21, a6, v11
	vand.vi	v11, v7, 15
	vwmacc.vx	v23, a2, v11
	vand.vi	v11, v8, 15
	lbu	a0, 15(s1)
	sd	a0, 1720(sp)                    # 8-byte Folded Spill
	lbu	t1, 16(s1)
	sd	t1, 1640(sp)                    # 8-byte Folded Spill
	lbu	a7, 17(s1)
	sd	a7, 1680(sp)                    # 8-byte Folded Spill
	lbu	a6, 18(s1)
	sd	a6, 1696(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v20, a0, v11
	addi	a0, s0, 704
	vle8.v	v24, (a0)
	vsrl.vi	v8, v8, 4
	lbu	a0, 47(s1)
	sd	a0, 1712(sp)                    # 8-byte Folded Spill
	lbu	a2, 48(s1)
	sd	a2, 1664(sp)                    # 8-byte Folded Spill
	lbu	t2, 49(s1)
	sd	t2, 1672(sp)                    # 8-byte Folded Spill
	lbu	t0, 50(s1)
	sd	t0, 1600(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v21, a0, v8
	vand.vi	v8, v5, 15
	lbu	a0, 31(s1)
	sd	a0, 1632(sp)                    # 8-byte Folded Spill
	lbu	a5, 32(s1)
	sd	a5, 1656(sp)                    # 8-byte Folded Spill
	lbu	a3, 33(s1)
	sd	a3, 1688(sp)                    # 8-byte Folded Spill
	lbu	a1, 34(s1)
	sd	a1, 1592(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a0, v8
	vand.vi	v8, v9, 15
	vwmacc.vx	v20, t1, v8
	addi	a0, s0, 720
	vle8.v	v8, (a0)
	vsrl.vi	v9, v9, 4
	vwmacc.vx	v21, a2, v9
	vand.vi	v9, v24, 15
	vwmacc.vx	v23, a5, v9
	vand.vi	v9, v16, 15
	vwmacc.vx	v20, a7, v9
	addi	a0, s0, 736
	vle8.v	v9, (a0)
	vsrl.vi	v11, v16, 4
	vwmacc.vx	v21, t2, v11
	vand.vi	v11, v8, 15
	vwmacc.vx	v23, a3, v11
	vand.vi	v11, v19, 15
	vwmacc.vx	v20, a6, v11
	addi	a0, s0, 752
	vle8.v	v16, (a0)
	vsrl.vi	v11, v19, 4
	vwmacc.vx	v21, t0, v11
	vand.vi	v11, v9, 15
	lbu	a0, 19(s1)
	sd	a0, 1648(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a1, v11
	vand.vi	v11, v22, 15
	vsrl.vi	v14, v22, 4
	vwmacc.vx	v20, a0, v11
	vand.vi	v11, v16, 15
	lbu	a0, 35(s1)
	sd	a0, 1704(sp)                    # 8-byte Folded Spill
	lbu	a5, 51(s1)
	sd	a5, 1616(sp)                    # 8-byte Folded Spill
	lbu	a3, 52(s1)
	sd	a3, 1608(sp)                    # 8-byte Folded Spill
	lbu	a2, 53(s1)
	sd	a2, 1624(sp)                    # 8-byte Folded Spill
	lbu	a1, 54(s1)
	sd	a1, 1584(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v21, a5, v14
	vwmacc.vx	v23, a0, v11
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v14, 0
	vmv.v.i	v26, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v31
	vwmacc.vv	v26, v11, v20
	addi	a0, s0, 224
	vle8.v	v14, (a0)
	vzext.vf2	v31, v6
	vwmacc.vv	v26, v31, v21
	addi	a0, s0, 112
	vle8.v	v15, (a0)
	csrr	a0, vlenb
	li	a5, 10
	mul	a0, a0, a5
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs1r.v	v15, (a0)                       # Unknown-size Folded Spill
	vwmacc.vv	v26, v11, v23
	addi	a0, s0, 240
	vle8.v	v20, (a0)
	vmv.v.i	v19, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v13, 4
	vwmacc.vx	v19, a3, v11
	addi	a0, s0, 768
	vle8.v	v1, (a0)
	vsrl.vi	v11, v30, 4
	vwmacc.vx	v19, a2, v11
	addi	a0, s0, 784
	vle8.v	v2, (a0)
	vsrl.vi	v11, v29, 4
	vwmacc.vx	v19, a1, v11
	addi	a0, s0, 800
	vle8.v	v6, (a0)
	vsrl.vi	v10, v10, 4
	lbu	a0, 55(s1)
	sd	a0, 1576(sp)                    # 8-byte Folded Spill
	lbu	a3, 56(s1)
	sd	a3, 1568(sp)                    # 8-byte Folded Spill
	lbu	a2, 57(s1)
	sd	a2, 1560(sp)                    # 8-byte Folded Spill
	lbu	a1, 58(s1)
	sd	a1, 1552(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, a0, v10
	addi	a0, s0, 816
	vle8.v	v4, (a0)
	vsrl.vi	v10, v17, 4
	vwmacc.vx	v19, a3, v10
	addi	a0, s0, 832
	vle8.v	v29, (a0)
	vsrl.vi	v10, v18, 4
	vwmacc.vx	v19, a2, v10
	addi	a0, s0, 848
	vle8.v	v3, (a0)
	vsrl.vi	v10, v12, 4
	vwmacc.vx	v19, a1, v10
	addi	a0, s0, 864
	vle8.v	v30, (a0)
	vsrl.vi	v10, v25, 4
	lbu	a0, 59(s1)
	sd	a0, 1544(sp)                    # 8-byte Folded Spill
	lbu	a3, 60(s1)
	sd	a3, 1536(sp)                    # 8-byte Folded Spill
	lbu	a2, 61(s1)
	sd	a2, 1528(sp)                    # 8-byte Folded Spill
	lbu	a1, 62(s1)
	sd	a1, 1520(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, a0, v10
	addi	a0, s0, 880
	vle8.v	v17, (a0)
	vsrl.vi	v10, v28, 4
	vwmacc.vx	v19, a3, v10
	addi	a0, s0, 896
	vle8.v	v25, (a0)
	vsrl.vi	v10, v0, 4
	vwmacc.vx	v19, a2, v10
	addi	a0, s0, 912
	vle8.v	v28, (a0)
	vsrl.vi	v10, v7, 4
	vwmacc.vx	v19, a1, v10
	addi	a0, s0, 928
	vle8.v	v12, (a0)
	vsrl.vi	v10, v5, 4
	lbu	a0, 63(s1)
	sd	a0, 1512(sp)                    # 8-byte Folded Spill
	lbu	a3, 64(s1)
	sd	a3, 1504(sp)                    # 8-byte Folded Spill
	lbu	a2, 65(s1)
	sd	a2, 1496(sp)                    # 8-byte Folded Spill
	lbu	a1, 66(s1)
	sd	a1, 1488(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, a0, v10
	addi	a0, s0, 1024
	vle8.v	v23, (a0)
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v19, a3, v10
	addi	a0, s0, 1040
	vle8.v	v24, (a0)
	vsrl.vi	v8, v8, 4
	vwmacc.vx	v19, a2, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v7, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v9, 4
	vwmacc.vx	v19, a1, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v5, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v16, 4
	lbu	a2, 67(s1)
	sd	a2, 1480(sp)                    # 8-byte Folded Spill
	lbu	a1, 68(s1)
	sd	a1, 1416(sp)                    # 8-byte Folded Spill
	lbu	a0, 69(s1)
	sd	a0, 1456(sp)                    # 8-byte Folded Spill
	lbu	a6, 70(s1)
	sd	a6, 1464(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, a2, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v9, 0
	vwmacc.vv	v26, v31, v19
	csrr	a2, vlenb
	li	a3, 19
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2000
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v8, 15
	vand.vi	v10, v1, 15
	vwmacc.vx	v7, a1, v10
	vand.vi	v10, v14, 3
	vmv1r.v	v21, v14
	csrr	a1, vlenb
	slli	a1, a1, 2
	add	a1, a1, sp
	addi	a1, a1, 2000
	vs1r.v	v14, (a1)                       # Unknown-size Folded Spill
	lbu	a1, 100(s1)
	sd	a1, 1408(sp)                    # 8-byte Folded Spill
	vsrl.vi	v11, v1, 4
	lbu	a3, 101(s1)
	sd	a3, 1448(sp)                    # 8-byte Folded Spill
	lbu	a7, 102(s1)
	sd	a7, 1432(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a1, v11
	lbu	a1, 84(s1)
	sd	a1, 1472(sp)                    # 8-byte Folded Spill
	vand.vi	v11, v23, 15
	lbu	a5, 85(s1)
	sd	a5, 1440(sp)                    # 8-byte Folded Spill
	lbu	a2, 86(s1)
	sd	a2, 1424(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v9, a1, v11
	vand.vi	v11, v15, 15
	vsll.vi	v10, v10, 4
	vor.vv	v18, v10, v8
	vand.vi	v8, v20, 3
	csrr	a1, vlenb
	slli	t0, a1, 2
	add	a1, a1, t0
	add	a1, a1, sp
	addi	a1, a1, 2000
	vs1r.v	v20, (a1)                       # Unknown-size Folded Spill
	vsll.vi	v8, v8, 4
	vor.vv	v19, v8, v11
	vand.vi	v8, v2, 15
	vwmacc.vx	v7, a0, v8
	addi	a0, s0, 1056
	vle8.v	v16, (a0)
	vsrl.vi	v8, v2, 4
	vwmacc.vx	v5, a3, v8
	vand.vi	v8, v24, 15
	vwmacc.vx	v9, a5, v8
	vand.vi	v8, v6, 15
	vwmacc.vx	v7, a6, v8
	addi	a0, s0, 1072
	vle8.v	v31, (a0)
	vsrl.vi	v8, v6, 4
	vwmacc.vx	v5, a7, v8
	vand.vi	v8, v16, 15
	vwmacc.vx	v9, a2, v8
	vand.vi	v8, v4, 15
	lbu	a0, 71(s1)
	sd	a0, 1400(sp)                    # 8-byte Folded Spill
	lbu	t1, 72(s1)
	sd	t1, 1352(sp)                    # 8-byte Folded Spill
	lbu	a7, 73(s1)
	sd	a7, 1360(sp)                    # 8-byte Folded Spill
	lbu	a6, 74(s1)
	sd	a6, 1384(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v7, a0, v8
	addi	a0, s0, 1088
	vle8.v	v6, (a0)
	vsrl.vi	v8, v4, 4
	lbu	a0, 103(s1)
	sd	a0, 1392(sp)                    # 8-byte Folded Spill
	lbu	a2, 104(s1)
	sd	a2, 1344(sp)                    # 8-byte Folded Spill
	lbu	t2, 105(s1)
	sd	t2, 1376(sp)                    # 8-byte Folded Spill
	lbu	t0, 106(s1)
	sd	t0, 1320(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a0, v8
	vand.vi	v8, v31, 15
	lbu	a0, 87(s1)
	sd	a0, 1328(sp)                    # 8-byte Folded Spill
	lbu	a5, 88(s1)
	sd	a5, 1336(sp)                    # 8-byte Folded Spill
	lbu	a3, 89(s1)
	sd	a3, 1368(sp)                    # 8-byte Folded Spill
	lbu	a1, 90(s1)
	sd	a1, 1312(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v9, a0, v8
	vand.vi	v8, v29, 15
	vwmacc.vx	v7, t1, v8
	addi	a0, s0, 1104
	vle8.v	v1, (a0)
	vsrl.vi	v8, v29, 4
	vwmacc.vx	v5, a2, v8
	vand.vi	v8, v6, 15
	vwmacc.vx	v9, a5, v8
	vand.vi	v8, v3, 15
	vwmacc.vx	v7, a7, v8
	addi	a0, s0, 1120
	vle8.v	v4, (a0)
	vsrl.vi	v8, v3, 4
	vwmacc.vx	v5, t2, v8
	vand.vi	v8, v1, 15
	vwmacc.vx	v9, a3, v8
	vand.vi	v8, v30, 15
	vwmacc.vx	v7, a6, v8
	addi	a0, s0, 1136
	vle8.v	v3, (a0)
	vsrl.vi	v8, v30, 4
	vwmacc.vx	v5, t0, v8
	vand.vi	v8, v4, 15
	vwmacc.vx	v9, a1, v8
	vand.vi	v8, v17, 15
	lbu	a0, 75(s1)
	sd	a0, 1304(sp)                    # 8-byte Folded Spill
	lbu	t0, 76(s1)
	sd	t0, 1224(sp)                    # 8-byte Folded Spill
	lbu	a7, 77(s1)
	sd	a7, 1248(sp)                    # 8-byte Folded Spill
	lbu	a6, 78(s1)
	sd	a6, 1280(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v7, a0, v8
	addi	a0, s0, 1152
	vle8.v	v2, (a0)
	vsrl.vi	v8, v17, 4
	lbu	a0, 107(s1)
	sd	a0, 1296(sp)                    # 8-byte Folded Spill
	lbu	a2, 108(s1)
	sd	a2, 1240(sp)                    # 8-byte Folded Spill
	lbu	t2, 109(s1)
	sd	t2, 1272(sp)                    # 8-byte Folded Spill
	lbu	t1, 110(s1)
	sd	t1, 1288(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a0, v8
	vand.vi	v8, v3, 15
	lbu	a0, 91(s1)
	sd	a0, 1216(sp)                    # 8-byte Folded Spill
	lbu	a3, 92(s1)
	sd	a3, 1232(sp)                    # 8-byte Folded Spill
	lbu	a5, 93(s1)
	sd	a5, 1264(sp)                    # 8-byte Folded Spill
	lbu	a1, 94(s1)
	sd	a1, 1256(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v9, a0, v8
	vand.vi	v8, v25, 15
	vwmacc.vx	v7, t0, v8
	addi	a0, s0, 1168
	vle8.v	v8, (a0)
	vsrl.vi	v10, v25, 4
	vwmacc.vx	v5, a2, v10
	vand.vi	v10, v2, 15
	vwmacc.vx	v9, a3, v10
	vand.vi	v10, v28, 15
	vwmacc.vx	v7, a7, v10
	addi	a0, s0, 1184
	vle8.v	v0, (a0)
	vsrl.vi	v10, v28, 4
	vwmacc.vx	v5, t2, v10
	vand.vi	v10, v8, 15
	vwmacc.vx	v9, a5, v10
	vand.vi	v10, v12, 15
	vwmacc.vx	v7, a6, v10
	addi	a0, s0, 944
	vle8.v	v11, (a0)
	vsrl.vi	v10, v12, 4
	vwmacc.vx	v5, t1, v10
	vand.vi	v10, v0, 15
	vwmacc.vx	v9, a1, v10
	vand.vi	v10, v11, 15
	lbu	a0, 79(s1)
	sd	a0, 1208(sp)                    # 8-byte Folded Spill
	lbu	t2, 80(s1)
	sd	t2, 1160(sp)                    # 8-byte Folded Spill
	lbu	t0, 81(s1)
	sd	t0, 1168(sp)                    # 8-byte Folded Spill
	lbu	a6, 82(s1)
	sd	a6, 1184(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v7, a0, v10
	addi	a0, s0, 960
	vle8.v	v10, (a0)
	vsrl.vi	v11, v11, 4
	lbu	a0, 111(s1)
	sd	a0, 1200(sp)                    # 8-byte Folded Spill
	lbu	a2, 112(s1)
	sd	a2, 1152(sp)                    # 8-byte Folded Spill
	lbu	t1, 113(s1)
	sd	t1, 1120(sp)                    # 8-byte Folded Spill
	lbu	a7, 114(s1)
	sd	a7, 1176(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a0, v11
	addi	a0, s0, 1200
	vle8.v	v12, (a0)
	lbu	a0, 95(s1)
	sd	a0, 1144(sp)                    # 8-byte Folded Spill
	lbu	a3, 96(s1)
	sd	a3, 1136(sp)                    # 8-byte Folded Spill
	lbu	a5, 97(s1)
	sd	a5, 1128(sp)                    # 8-byte Folded Spill
	lbu	a1, 98(s1)
	sd	a1, 1104(sp)                    # 8-byte Folded Spill
	vand.vi	v11, v12, 15
	vwmacc.vx	v9, a0, v11
	vand.vi	v11, v10, 15
	vwmacc.vx	v7, t2, v11
	vsrl.vi	v10, v10, 4
	addi	a0, s0, 1216
	vle8.v	v29, (a0)
	addi	a0, s0, 976
	vle8.v	v11, (a0)
	vwmacc.vx	v5, a2, v10
	vand.vi	v10, v29, 15
	vwmacc.vx	v9, a3, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v7, t0, v10
	vsrl.vi	v10, v11, 4
	addi	a0, s0, 1232
	vle8.v	v25, (a0)
	addi	a0, s0, 992
	vle8.v	v11, (a0)
	vwmacc.vx	v5, t1, v10
	vand.vi	v10, v25, 15
	vwmacc.vx	v9, a5, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v7, a6, v10
	vsrl.vi	v10, v11, 4
	vwmacc.vx	v5, a7, v10
	addi	a0, s0, 1248
	vle8.v	v28, (a0)
	addi	a0, s0, 1008
	vle8.v	v10, (a0)
	lbu	a0, 83(s1)
	sd	a0, 1192(sp)                    # 8-byte Folded Spill
	vand.vi	v11, v28, 15
	vwmacc.vx	v9, a1, v11
	vand.vi	v11, v10, 15
	vwmacc.vx	v7, a0, v11
	vsrl.vi	v10, v10, 4
	lbu	a5, 99(s1)
	sd	a5, 1112(sp)                    # 8-byte Folded Spill
	addi	a0, s0, 1264
	lbu	a1, 115(s1)
	sd	a1, 1080(sp)                    # 8-byte Folded Spill
	lbu	a3, 116(s1)
	sd	a3, 1088(sp)                    # 8-byte Folded Spill
	vle8.v	v30, (a0)
	lbu	a2, 117(s1)
	sd	a2, 1096(sp)                    # 8-byte Folded Spill
	lbu	a6, 118(s1)
	sd	a6, 1072(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a1, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v9, a5, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v18
	vwmacc.vv	v26, v10, v7
	vzext.vf2	v7, v19
	vwmacc.vv	v26, v7, v5
	vwmacc.vv	v26, v10, v9
	vmv.v.i	v5, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v9, v23, 4
	vwmacc.vx	v5, a3, v9
	addi	a0, s0, 128
	vle8.v	v14, (a0)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs1r.v	v14, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v9, v24, 4
	vwmacc.vx	v5, a2, v9
	addi	a0, s0, 144
	vle8.v	v15, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 3
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs1r.v	v15, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v9, v16, 4
	vwmacc.vx	v5, a6, v9
	addi	a0, s0, 1280
	vle8.v	v13, (a0)
	vsrl.vi	v9, v31, 4
	lbu	a0, 119(s1)
	sd	a0, 1064(sp)                    # 8-byte Folded Spill
	lbu	a3, 120(s1)
	sd	a3, 1056(sp)                    # 8-byte Folded Spill
	lbu	a2, 121(s1)
	sd	a2, 1048(sp)                    # 8-byte Folded Spill
	lbu	a1, 122(s1)
	sd	a1, 1040(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a0, v9
	addi	a0, s0, 1296
	vle8.v	v10, (a0)
	vsrl.vi	v9, v6, 4
	vwmacc.vx	v5, a3, v9
	addi	a0, s0, 1312
	vle8.v	v19, (a0)
	vsrl.vi	v9, v1, 4
	vwmacc.vx	v5, a2, v9
	addi	a0, s0, 1328
	vle8.v	v18, (a0)
	vsrl.vi	v9, v4, 4
	vwmacc.vx	v5, a1, v9
	addi	a0, s0, 1344
	vle8.v	v16, (a0)
	vsrl.vi	v9, v3, 4
	lbu	a0, 123(s1)
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	lbu	a3, 124(s1)
	sd	a3, 1024(sp)                    # 8-byte Folded Spill
	lbu	a2, 125(s1)
	sd	a2, 1016(sp)                    # 8-byte Folded Spill
	lbu	a1, 126(s1)
	sd	a1, 1008(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v5, a0, v9
	addi	a0, s0, 1360
	vle8.v	v4, (a0)
	vsrl.vi	v9, v2, 4
	vwmacc.vx	v5, a3, v9
	addi	a0, s0, 1376
	vle8.v	v6, (a0)
	vsrl.vi	v8, v8, 4
	vwmacc.vx	v5, a2, v8
	addi	a0, s0, 1392
	vle8.v	v17, (a0)
	vsrl.vi	v8, v0, 4
	vwmacc.vx	v5, a1, v8
	addi	a0, s0, 1408
	vle8.v	v8, (a0)
	vsrl.vi	v9, v12, 4
	lbu	a0, 127(s1)
	sd	a0, 1000(sp)                    # 8-byte Folded Spill
	lbu	a3, 128(s1)
	sd	a3, 992(sp)                     # 8-byte Folded Spill
	lbu	a2, 129(s1)
	sd	a2, 984(sp)                     # 8-byte Folded Spill
	lbu	a1, 130(s1)
	sd	a1, 976(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v5, a0, v9
	addi	a0, s0, 1536
	vle8.v	v31, (a0)
	vsrl.vi	v9, v29, 4
	vwmacc.vx	v5, a3, v9
	addi	a0, s0, 1552
	vle8.v	v29, (a0)
	vsrl.vi	v9, v25, 4
	vwmacc.vx	v5, a2, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v9, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v28, 4
	vwmacc.vx	v5, a1, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v12, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v30, 4
	lbu	a2, 131(s1)
	sd	a2, 968(sp)                     # 8-byte Folded Spill
	lbu	a1, 132(s1)
	sd	a1, 896(sp)                     # 8-byte Folded Spill
	lbu	a0, 133(s1)
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	lbu	a7, 134(s1)
	sd	a7, 960(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v5, a2, v11
	vand.vi	v11, v14, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v7, v5
	csrr	a2, vlenb
	li	a3, 18
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2000
	vl1r.v	v14, (a2)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	li	a6, 48
	vand.vx	v14, v14, a6
	vand.vi	v15, v15, 15
	vor.vv	v28, v14, v11
	csrr	a2, vlenb
	slli	a3, a2, 4
	add	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2000
	vl1r.v	v11, (a2)                       # Unknown-size Folded Reload
	vand.vx	v11, v11, a6
	vor.vv	v30, v11, v15
	vand.vi	v11, v13, 15
	vwmacc.vx	v9, a1, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v25, 0
	lbu	a1, 164(s1)
	sd	a1, 888(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v13, 4
	lbu	a5, 165(s1)
	sd	a5, 952(sp)                     # 8-byte Folded Spill
	lbu	t0, 166(s1)
	sd	t0, 920(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v12, a1, v11
	lbu	a1, 148(s1)
	sd	a1, 928(sp)                     # 8-byte Folded Spill
	vand.vi	v11, v31, 15
	lbu	a2, 149(s1)
	sd	a2, 944(sp)                     # 8-byte Folded Spill
	lbu	a3, 150(s1)
	sd	a3, 912(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v25, a1, v11
	vand.vi	v11, v10, 15
	vwmacc.vx	v9, a0, v11
	addi	a0, s0, 1568
	vle8.v	v7, (a0)
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v12, a5, v10
	vand.vi	v10, v29, 15
	vwmacc.vx	v25, a2, v10
	vand.vi	v10, v19, 15
	vwmacc.vx	v9, a7, v10
	addi	a0, s0, 1584
	vle8.v	v5, (a0)
	vsrl.vi	v10, v19, 4
	vwmacc.vx	v12, t0, v10
	vand.vi	v10, v7, 15
	vwmacc.vx	v25, a3, v10
	vand.vi	v10, v18, 15
	lbu	a0, 135(s1)
	sd	a0, 904(sp)                     # 8-byte Folded Spill
	lbu	t2, 136(s1)
	sd	t2, 824(sp)                     # 8-byte Folded Spill
	lbu	t0, 137(s1)
	sd	t0, 848(sp)                     # 8-byte Folded Spill
	lbu	a7, 138(s1)
	sd	a7, 872(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v9, a0, v10
	addi	a0, s0, 1600
	vle8.v	v13, (a0)
	vsrl.vi	v10, v18, 4
	lbu	a0, 167(s1)
	sd	a0, 880(sp)                     # 8-byte Folded Spill
	lbu	a2, 168(s1)
	sd	a2, 840(sp)                     # 8-byte Folded Spill
	lbu	t3, 169(s1)
	sd	t3, 864(sp)                     # 8-byte Folded Spill
	lbu	t1, 170(s1)
	sd	t1, 808(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v12, a0, v10
	vand.vi	v10, v5, 15
	lbu	a0, 151(s1)
	sd	a0, 816(sp)                     # 8-byte Folded Spill
	lbu	a5, 152(s1)
	sd	a5, 832(sp)                     # 8-byte Folded Spill
	lbu	a3, 153(s1)
	sd	a3, 856(sp)                     # 8-byte Folded Spill
	lbu	a1, 154(s1)
	sd	a1, 800(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v25, a0, v10
	vand.vi	v10, v16, 15
	vwmacc.vx	v9, t2, v10
	addi	a0, s0, 1616
	vle8.v	v3, (a0)
	vsrl.vi	v10, v16, 4
	vwmacc.vx	v12, a2, v10
	vand.vi	v10, v13, 15
	vwmacc.vx	v25, a5, v10
	vand.vi	v10, v4, 15
	vwmacc.vx	v9, t0, v10
	addi	a0, s0, 1632
	vle8.v	v18, (a0)
	vsrl.vi	v10, v4, 4
	vwmacc.vx	v12, t3, v10
	vand.vi	v10, v3, 15
	vwmacc.vx	v25, a3, v10
	vand.vi	v10, v6, 15
	vwmacc.vx	v9, a7, v10
	addi	a0, s0, 1648
	vle8.v	v4, (a0)
	vsrl.vi	v10, v6, 4
	vwmacc.vx	v12, t1, v10
	vand.vi	v10, v18, 15
	vwmacc.vx	v25, a1, v10
	vand.vi	v10, v17, 15
	lbu	a0, 139(s1)
	sd	a0, 792(sp)                     # 8-byte Folded Spill
	lbu	t1, 140(s1)
	sd	t1, 712(sp)                     # 8-byte Folded Spill
	lbu	t0, 141(s1)
	sd	t0, 760(sp)                     # 8-byte Folded Spill
	lbu	a7, 142(s1)
	sd	a7, 720(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v9, a0, v10
	addi	a0, s0, 1664
	vle8.v	v1, (a0)
	vsrl.vi	v10, v17, 4
	lbu	a0, 171(s1)
	sd	a0, 784(sp)                     # 8-byte Folded Spill
	lbu	a2, 172(s1)
	sd	a2, 752(sp)                     # 8-byte Folded Spill
	lbu	t3, 173(s1)
	sd	t3, 768(sp)                     # 8-byte Folded Spill
	lbu	t2, 174(s1)
	sd	t2, 776(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v12, a0, v10
	vand.vi	v10, v4, 15
	lbu	a0, 155(s1)
	sd	a0, 704(sp)                     # 8-byte Folded Spill
	lbu	a3, 156(s1)
	sd	a3, 744(sp)                     # 8-byte Folded Spill
	lbu	a5, 157(s1)
	sd	a5, 736(sp)                     # 8-byte Folded Spill
	lbu	a1, 158(s1)
	sd	a1, 728(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v25, a0, v10
	vand.vi	v10, v8, 15
	vwmacc.vx	v9, t1, v10
	addi	a0, s0, 1424
	vle8.v	v10, (a0)
	vsrl.vi	v8, v8, 4
	vwmacc.vx	v12, a2, v8
	vand.vi	v8, v1, 15
	vwmacc.vx	v25, a3, v8
	vand.vi	v8, v10, 15
	vwmacc.vx	v9, t0, v8
	addi	a0, s0, 1680
	vle8.v	v17, (a0)
	vsrl.vi	v8, v10, 4
	vwmacc.vx	v12, t3, v8
	addi	a0, s0, 1440
	vle8.v	v8, (a0)
	vand.vi	v10, v17, 15
	vwmacc.vx	v25, a5, v10
	addi	a0, s0, 1696
	vle8.v	v0, (a0)
	vand.vi	v10, v8, 15
	vwmacc.vx	v9, a7, v10
	vsrl.vi	v8, v8, 4
	vwmacc.vx	v12, t2, v8
	vand.vi	v8, v0, 15
	vwmacc.vx	v25, a1, v8
	addi	a0, s0, 1456
	lbu	a1, 143(s1)
	sd	a1, 696(sp)                     # 8-byte Folded Spill
	vle8.v	v8, (a0)
	lbu	t3, 144(s1)
	sd	t3, 648(sp)                     # 8-byte Folded Spill
	lbu	t1, 145(s1)
	sd	t1, 664(sp)                     # 8-byte Folded Spill
	lbu	a7, 146(s1)
	sd	a7, 672(sp)                     # 8-byte Folded Spill
	vand.vi	v10, v8, 15
	vwmacc.vx	v9, a1, v10
	vsrl.vi	v8, v8, 4
	lbu	a0, 175(s1)
	sd	a0, 688(sp)                     # 8-byte Folded Spill
	lbu	t4, 176(s1)
	sd	t4, 640(sp)                     # 8-byte Folded Spill
	lbu	t2, 177(s1)
	sd	t2, 656(sp)                     # 8-byte Folded Spill
	lbu	t0, 178(s1)
	sd	t0, 680(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v12, a0, v8
	addi	a0, s0, 1472
	addi	a1, s0, 1712
	lbu	a2, 159(s1)
	sd	a2, 632(sp)                     # 8-byte Folded Spill
	lbu	a3, 160(s1)
	sd	a3, 624(sp)                     # 8-byte Folded Spill
	vle8.v	v15, (a1)
	lbu	a1, 161(s1)
	sd	a1, 616(sp)                     # 8-byte Folded Spill
	vle8.v	v8, (a0)
	lbu	a5, 162(s1)
	sd	a5, 608(sp)                     # 8-byte Folded Spill
	vand.vi	v10, v15, 15
	vwmacc.vx	v25, a2, v10
	vand.vi	v10, v8, 15
	vwmacc.vx	v9, t3, v10
	vsrl.vi	v8, v8, 4
	addi	a0, s0, 1728
	vle8.v	v11, (a0)
	addi	a0, s0, 1488
	vle8.v	v10, (a0)
	vwmacc.vx	v12, t4, v8
	vand.vi	v8, v11, 15
	vwmacc.vx	v25, a3, v8
	vand.vi	v8, v10, 15
	vwmacc.vx	v9, t1, v8
	vsrl.vi	v8, v10, 4
	addi	a0, s0, 1744
	vle8.v	v16, (a0)
	addi	a0, s0, 1504
	vle8.v	v10, (a0)
	vwmacc.vx	v12, t2, v8
	vand.vi	v8, v16, 15
	vwmacc.vx	v25, a1, v8
	vand.vi	v8, v10, 15
	vwmacc.vx	v9, a7, v8
	vsrl.vi	v8, v10, 4
	vwmacc.vx	v12, t0, v8
	addi	a0, s0, 1760
	vle8.v	v8, (a0)
	addi	a0, s0, 1520
	vle8.v	v10, (a0)
	lbu	a0, 147(s1)
	sd	a0, 600(sp)                     # 8-byte Folded Spill
	vand.vi	v14, v8, 15
	vwmacc.vx	v25, a5, v14
	vand.vi	v14, v10, 15
	vwmacc.vx	v9, a0, v14
	vsrl.vi	v10, v10, 4
	lbu	a3, 163(s1)
	sd	a3, 592(sp)                     # 8-byte Folded Spill
	addi	a0, s0, 1776
	lbu	a5, 179(s1)
	sd	a5, 584(sp)                     # 8-byte Folded Spill
	lbu	a1, 180(s1)
	sd	a1, 568(sp)                     # 8-byte Folded Spill
	vle8.v	v6, (a0)
	lbu	a2, 181(s1)
	sd	a2, 560(sp)                     # 8-byte Folded Spill
	lbu	a0, 182(s1)
	sd	a0, 576(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v12, a5, v10
	vand.vi	v10, v6, 15
	vwmacc.vx	v25, a3, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v28
	vwmacc.vv	v26, v10, v9
	vzext.vf2	v9, v30
	vwmacc.vv	v26, v9, v12
	vwmacc.vv	v26, v10, v25
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v31, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v10, a1, v12
	vsrl.vi	v12, v29, 4
	vwmacc.vx	v10, a2, v12
	vsrl.vi	v12, v7, 4
	vwmacc.vx	v10, a0, v12
	addi	a0, s0, 160
	vle8.v	v14, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 3
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs1r.v	v14, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v12, v5, 4
	lbu	a0, 183(s1)
	sd	a0, 552(sp)                     # 8-byte Folded Spill
	lbu	a3, 184(s1)
	sd	a3, 544(sp)                     # 8-byte Folded Spill
	lbu	a2, 185(s1)
	sd	a2, 536(sp)                     # 8-byte Folded Spill
	lbu	a1, 186(s1)
	sd	a1, 528(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v10, a0, v12
	addi	a0, s0, 176
	vle8.v	v22, (a0)
	csrr	a0, vlenb
	li	a5, 6
	mul	a0, a0, a5
	add	a0, a0, sp
	addi	a0, a0, 2000
	vs1r.v	v22, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v12, v13, 4
	vwmacc.vx	v10, a3, v12
	addi	a0, s0, 1792
	vle8.v	v19, (a0)
	vsrl.vi	v12, v3, 4
	vwmacc.vx	v10, a2, v12
	addi	a0, s0, 1808
	vle8.v	v2, (a0)
	vsrl.vi	v12, v18, 4
	vwmacc.vx	v10, a1, v12
	addi	a0, s0, 1824
	vle8.v	v3, (a0)
	vsrl.vi	v12, v4, 4
	lbu	a0, 187(s1)
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	lbu	a3, 188(s1)
	sd	a3, 512(sp)                     # 8-byte Folded Spill
	lbu	a2, 189(s1)
	sd	a2, 504(sp)                     # 8-byte Folded Spill
	lbu	a1, 190(s1)
	sd	a1, 496(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v10, a0, v12
	addi	a0, s0, 1840
	vle8.v	v4, (a0)
	vsrl.vi	v12, v1, 4
	vwmacc.vx	v10, a3, v12
	addi	a0, s0, 1856
	vle8.v	v13, (a0)
	vsrl.vi	v12, v17, 4
	vwmacc.vx	v10, a2, v12
	addi	a0, s0, 1872
	vle8.v	v17, (a0)
	vsrl.vi	v12, v0, 4
	vwmacc.vx	v10, a1, v12
	addi	a0, s0, 1888
	vle8.v	v18, (a0)
	vsrl.vi	v12, v15, 4
	lbu	a0, 191(s1)
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	lbu	a3, 192(s1)
	sd	a3, 480(sp)                     # 8-byte Folded Spill
	lbu	a2, 193(s1)
	sd	a2, 472(sp)                     # 8-byte Folded Spill
	lbu	a1, 194(s1)
	sd	a1, 464(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v10, a0, v12
	addi	a0, s0, 1904
	vle8.v	v5, (a0)
	vsrl.vi	v11, v11, 4
	vwmacc.vx	v10, a3, v11
	addi	s4, s0, 2047
	addi	a0, s4, 1
	vle8.v	v29, (a0)
	vsrl.vi	v11, v16, 4
	vwmacc.vx	v10, a2, v11
	addi	a0, s4, 17
	vle8.v	v30, (a0)
	vsrl.vi	v8, v8, 4
	vwmacc.vx	v10, a1, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v6, 4
	lbu	a3, 195(s1)
	sd	a3, 456(sp)                     # 8-byte Folded Spill
	lbu	a2, 196(s1)
	sd	a2, 448(sp)                     # 8-byte Folded Spill
	lbu	a1, 197(s1)
	sd	a1, 440(sp)                     # 8-byte Folded Spill
	lbu	a0, 198(s1)
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v10, a3, v11
	vand.vi	v11, v14, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v9, v10
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v9, v21, a6
	vand.vi	v10, v22, 15
	vor.vv	v12, v9, v11
	vand.vx	v9, v20, a6
	vor.vv	v31, v9, v10
	vand.vi	v9, v19, 15
	vwmacc.vx	v8, a2, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v7, 0
	lbu	a5, 228(s1)
	sd	a5, 424(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v9, v19, 4
	lbu	a3, 229(s1)
	sd	a3, 432(sp)                     # 8-byte Folded Spill
	lbu	a2, 230(s1)
	sd	a2, 416(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v7, a5, v9
	vand.vi	v9, v2, 15
	vwmacc.vx	v8, a1, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v9, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v2, 4
	vwmacc.vx	v7, a3, v10
	lbu	a3, 212(s1)
	sd	a3, 392(sp)                     # 8-byte Folded Spill
	vand.vi	v10, v29, 15
	lbu	a1, 213(s1)
	sd	a1, 408(sp)                     # 8-byte Folded Spill
	lbu	a5, 214(s1)
	sd	a5, 376(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v9, a3, v10
	vand.vi	v10, v3, 15
	vwmacc.vx	v8, a0, v10
	addi	a0, s4, 33
	vle8.v	v16, (a0)
	vsrl.vi	v10, v3, 4
	vwmacc.vx	v7, a2, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v9, a1, v10
	vand.vi	v10, v4, 15
	lbu	a0, 199(s1)
	sd	a0, 384(sp)                     # 8-byte Folded Spill
	lbu	a1, 200(s1)
	sd	a1, 368(sp)                     # 8-byte Folded Spill
	lbu	a7, 201(s1)
	sd	a7, 304(sp)                     # 8-byte Folded Spill
	lbu	a6, 202(s1)
	sd	a6, 328(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v8, a0, v10
	addi	a0, s4, 49
	vle8.v	v19, (a0)
	vsrl.vi	v10, v4, 4
	lbu	a0, 231(s1)
	sd	a0, 360(sp)                     # 8-byte Folded Spill
	lbu	a3, 232(s1)
	sd	a3, 352(sp)                     # 8-byte Folded Spill
	lbu	a2, 233(s1)
	sd	a2, 320(sp)                     # 8-byte Folded Spill
	lbu	t0, 234(s1)
	sd	t0, 344(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v7, a0, v10
	vand.vi	v10, v16, 15
	vwmacc.vx	v9, a5, v10
	vand.vi	v10, v13, 15
	vwmacc.vx	v8, a1, v10
	addi	a0, s4, 65
	vle8.v	v10, (a0)
	vsrl.vi	v11, v13, 4
	vwmacc.vx	v7, a3, v11
	vand.vi	v11, v19, 15
	lbu	a0, 215(s1)
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	lbu	a1, 216(s1)
	sd	a1, 312(sp)                     # 8-byte Folded Spill
	lbu	a3, 217(s1)
	sd	a3, 336(sp)                     # 8-byte Folded Spill
	lbu	a5, 218(s1)
	sd	a5, 264(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v9, a0, v11
	vand.vi	v11, v17, 15
	vwmacc.vx	v8, a7, v11
	addi	a0, s4, 81
	vle8.v	v13, (a0)
	vsrl.vi	v11, v17, 4
	vwmacc.vx	v7, a2, v11
	vand.vi	v11, v10, 15
	vwmacc.vx	v9, a1, v11
	vand.vi	v11, v18, 15
	vwmacc.vx	v8, a6, v11
	addi	a0, s0, 1920
	vle8.v	v14, (a0)
	vsrl.vi	v11, v18, 4
	vwmacc.vx	v7, t0, v11
	vand.vi	v11, v13, 15
	vwmacc.vx	v9, a3, v11
	vand.vi	v11, v5, 15
	lbu	a0, 203(s1)
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	lbu	a2, 204(s1)
	sd	a2, 256(sp)                     # 8-byte Folded Spill
	lbu	t0, 205(s1)
	sd	t0, 224(sp)                     # 8-byte Folded Spill
	lbu	a6, 206(s1)
	sd	a6, 200(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v8, a0, v11
	addi	a0, s4, 97
	vle8.v	v17, (a0)
	vsrl.vi	v11, v5, 4
	lbu	a0, 235(s1)
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	lbu	a3, 236(s1)
	sd	a3, 280(sp)                     # 8-byte Folded Spill
	lbu	a1, 237(s1)
	sd	a1, 240(sp)                     # 8-byte Folded Spill
	lbu	a7, 238(s1)
	sd	a7, 248(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v7, a0, v11
	vand.vi	v11, v17, 15
	vwmacc.vx	v9, a5, v11
	vand.vi	v11, v14, 15
	vwmacc.vx	v8, a2, v11
	addi	a0, s0, 1936
	vle8.v	v11, (a0)
	vsrl.vi	v14, v14, 4
	vwmacc.vx	v7, a3, v14
	addi	a0, s4, 113
	vle8.v	v18, (a0)
	lbu	a0, 219(s1)
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	lbu	a2, 220(s1)
	sd	a2, 208(sp)                     # 8-byte Folded Spill
	lbu	a3, 221(s1)
	sd	a3, 232(sp)                     # 8-byte Folded Spill
	lbu	t2, 222(s1)
	sd	t2, 168(sp)                     # 8-byte Folded Spill
	vand.vi	v14, v18, 15
	vwmacc.vx	v9, a0, v14
	vand.vi	v14, v11, 15
	vwmacc.vx	v8, t0, v14
	vsrl.vi	v14, v11, 4
	addi	a0, s4, 129
	vle8.v	v11, (a0)
	vwmacc.vx	v7, a1, v14
	addi	a0, s0, 1952
	vle8.v	v14, (a0)
	vand.vi	v15, v11, 15
	vwmacc.vx	v9, a2, v15
	addi	a0, s4, 145
	vle8.v	v15, (a0)
	vand.vi	v20, v14, 15
	vwmacc.vx	v8, a6, v20
	vsrl.vi	v14, v14, 4
	vwmacc.vx	v7, a7, v14
	vand.vi	v14, v15, 15
	vwmacc.vx	v9, a3, v14
	addi	a0, s0, 1968
	lbu	a1, 207(s1)
	sd	a1, 192(sp)                     # 8-byte Folded Spill
	vle8.v	v14, (a0)
	lbu	a2, 208(s1)
	sd	a2, 160(sp)                     # 8-byte Folded Spill
	lbu	t0, 209(s1)
	sd	t0, 112(sp)                     # 8-byte Folded Spill
	lbu	a6, 210(s1)
	sd	a6, 120(sp)                     # 8-byte Folded Spill
	vand.vi	v20, v14, 15
	vwmacc.vx	v8, a1, v20
	vsrl.vi	v14, v14, 4
	addi	a0, s0, 1984
	addi	a1, s4, 161
	lbu	a5, 239(s1)
	sd	a5, 176(sp)                     # 8-byte Folded Spill
	lbu	a3, 240(s1)
	sd	a3, 184(sp)                     # 8-byte Folded Spill
	lbu	t1, 241(s1)
	sd	t1, 104(sp)                     # 8-byte Folded Spill
	vle8.v	v6, (a1)
	lbu	a7, 242(s1)
	sd	a7, 144(sp)                     # 8-byte Folded Spill
	vle8.v	v20, (a0)
	vwmacc.vx	v7, a5, v14
	vand.vi	v14, v6, 15
	vwmacc.vx	v9, t2, v14
	vand.vi	v14, v20, 15
	vwmacc.vx	v8, a2, v14
	vsrl.vi	v14, v20, 4
	vwmacc.vx	v7, a3, v14
	addi	a0, s0, 2000
	addi	a1, s4, 177
	lbu	a2, 223(s1)
	sd	a2, 96(sp)                      # 8-byte Folded Spill
	lbu	s11, 224(s1)
	vle8.v	v5, (a1)
	lbu	s10, 225(s1)
	vle8.v	v14, (a0)
	lbu	s9, 226(s1)
	vand.vi	v20, v5, 15
	vwmacc.vx	v9, a2, v20
	vand.vi	v20, v14, 15
	vwmacc.vx	v8, t0, v20
	vsrl.vi	v20, v14, 4
	addi	a0, s4, 193
	vle8.v	v14, (a0)
	addi	a0, s0, 2016
	vle8.v	v21, (a0)
	vwmacc.vx	v7, t1, v20
	vand.vi	v20, v14, 15
	vwmacc.vx	v9, s11, v20
	vand.vi	v20, v21, 15
	vwmacc.vx	v8, a6, v20
	vsrl.vi	v20, v21, 4
	vwmacc.vx	v7, a7, v20
	addi	a0, s4, 209
	vle8.v	v20, (a0)
	addi	a0, s0, 2032
	vle8.v	v21, (a0)
	lbu	s5, 211(s1)
	vand.vi	v22, v20, 15
	vwmacc.vx	v9, s10, v22
	vand.vi	v22, v21, 15
	vwmacc.vx	v8, s5, v22
	vsrl.vi	v21, v21, 4
	lbu	s3, 243(s1)
	lbu	s6, 244(s1)
	lbu	s7, 245(s1)
	lbu	s8, 246(s1)
	vwmacc.vx	v7, s3, v21
	addi	a0, s4, 225
	vle8.v	v22, (a0)
	addi	a0, s4, 241
	vle8.v	v24, (a0)
	lbu	s2, 227(s1)
	vand.vi	v21, v22, 15
	vwmacc.vx	v9, s9, v21
	vand.vi	v21, v24, 15
	vwmacc.vx	v9, s2, v21
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v21, v12
	vwmacc.vv	v26, v21, v8
	vzext.vf2	v8, v31
	vwmacc.vv	v26, v8, v7
	vwmacc.vv	v26, v21, v9
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v29, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v9, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v9, s6, v12
	vsrl.vi	v12, v30, 4
	vwmacc.vx	v9, s7, v12
	vsrl.vi	v12, v16, 4
	vwmacc.vx	v9, s8, v12
	vsrl.vi	v12, v19, 4
	lbu	t3, 247(s1)
	lbu	t4, 248(s1)
	lbu	t5, 249(s1)
	lbu	t6, 250(s1)
	vwmacc.vx	v9, t3, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v9, t4, v10
	addi	a0, s0, 264
	vle8.v	v4, (a0)
	vsrl.vi	v10, v13, 4
	vwmacc.vx	v9, t5, v10
	addi	a0, s0, 280
	vle8.v	v25, (a0)
	vsrl.vi	v10, v17, 4
	vwmacc.vx	v9, t6, v10
	addi	a0, s0, 296
	vle8.v	v16, (a0)
	vsrl.vi	v10, v18, 4
	lbu	a7, 251(s1)
	lbu	t0, 252(s1)
	lbu	t1, 253(s1)
	lbu	t2, 254(s1)
	vwmacc.vx	v9, a7, v10
	addi	a0, s0, 312
	vle8.v	v17, (a0)
	vsrl.vi	v10, v11, 4
	vwmacc.vx	v9, t0, v10
	addi	a0, s0, 328
	vle8.v	v31, (a0)
	vsrl.vi	v10, v15, 4
	vwmacc.vx	v9, t1, v10
	addi	a0, s0, 344
	vle8.v	v7, (a0)
	vsrl.vi	v10, v6, 4
	vwmacc.vx	v9, t2, v10
	addi	a0, s0, 360
	vle8.v	v29, (a0)
	vsrl.vi	v10, v5, 4
	lbu	a1, 255(s1)
	lbu	a2, 256(s1)
	lbu	a3, 257(s1)
	lbu	a6, 258(s1)
	vwmacc.vx	v9, a1, v10
	addi	a0, s0, 376
	vle8.v	v30, (a0)
	vsrl.vi	v10, v14, 4
	vwmacc.vx	v9, a2, v10
	addi	a0, s0, 392
	vle8.v	v18, (a0)
	vsrl.vi	v10, v20, 4
	vwmacc.vx	v9, a3, v10
	addi	a0, s0, 408
	vle8.v	v20, (a0)
	vsrl.vi	v10, v22, 4
	vwmacc.vx	v9, a6, v10
	addi	a0, s0, 424
	vle8.v	v19, (a0)
	flw	fa5, 0(s1)
	vsrl.vi	v10, v24, 4
	lbu	a0, 259(s1)
	lh	a5, 260(s1)
	sd	a5, 136(sp)                     # 8-byte Folded Spill
	lh	a5, 262(s1)
	sd	a5, 128(sp)                     # 8-byte Folded Spill
	lh	a5, 264(s1)
	sd	a5, 152(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v9, a0, v10
	vle16.v	v10, (s0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v8, v9
	addi	ra, s0, 520
	vle8.v	v8, (ra)
	vfwcvt.f.f.v	v12, v10
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v10, v12, fa5
	vfcvt.f.x.v	v14, v26
	csrr	a5, vlenb
	li	ra, 11
	mul	a5, a5, ra
	add	a5, a5, sp
	addi	a5, a5, 2000
	vl2r.v	v12, (a5)                       # Unknown-size Folded Reload
	vfmadd.vv	v14, v10, v12
	csrr	a5, vlenb
	li	ra, 11
	mul	a5, a5, ra
	add	a5, a5, sp
	addi	a5, a5, 2000
	vs2r.v	v14, (a5)                       # Unknown-size Folded Spill
	addi	a5, s0, 536
	vle8.v	v10, (a5)
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v9, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v4, 15
	ld	a5, 1912(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a5, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v12, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v4, 4
	ld	a5, 1936(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a5, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v13, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v8, 15
	ld	a5, 1944(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a5, v11
	vand.vi	v11, v25, 15
	ld	a5, 1872(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a5, v11
	addi	a5, s0, 552
	vle8.v	v11, (a5)
	vsrl.vi	v14, v25, 4
	ld	a5, 1888(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a5, v14
	vand.vi	v14, v10, 15
	ld	a5, 1880(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a5, v14
	vand.vi	v14, v16, 15
	ld	a5, 1896(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a5, v14
	addi	a5, s0, 568
	vle8.v	v15, (a5)
	vsrl.vi	v14, v16, 4
	ld	a5, 1928(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a5, v14
	vand.vi	v14, v11, 15
	ld	a5, 1904(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a5, v14
	vand.vi	v14, v17, 15
	ld	a5, 1920(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a5, v14
	addi	a5, s0, 584
	vle8.v	v16, (a5)
	vsrl.vi	v14, v17, 4
	ld	a5, 1840(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a5, v14
	vand.vi	v14, v15, 15
	ld	a5, 1824(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a5, v14
	vand.vi	v14, v31, 15
	ld	a5, 1856(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a5, v14
	addi	a5, s0, 600
	vle8.v	v17, (a5)
	vsrl.vi	v14, v31, 4
	ld	a5, 1848(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a5, v14
	vand.vi	v14, v16, 15
	ld	a5, 1832(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a5, v14
	vand.vi	v14, v7, 15
	ld	a5, 1864(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a5, v14
	addi	a5, s0, 616
	vle8.v	v31, (a5)
	vsrl.vi	v14, v7, 4
	ld	a5, 1808(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a5, v14
	vand.vi	v14, v17, 15
	vwmacc.vx	v13, a4, v14
	vand.vi	v14, v29, 15
	ld	a4, 1816(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	addi	a5, s0, 632
	vle8.v	v5, (a5)
	vsrl.vi	v14, v29, 4
	ld	a4, 1960(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v14
	vand.vi	v14, v31, 15
	ld	a4, 1792(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v14
	vand.vi	v14, v30, 15
	ld	a4, 1800(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	addi	a5, s0, 648
	vle8.v	v4, (a5)
	vsrl.vi	v14, v30, 4
	ld	a4, 1784(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v14
	vand.vi	v14, v5, 15
	ld	a4, 1752(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v14
	vand.vi	v14, v18, 15
	ld	a4, 1968(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	addi	a5, s0, 664
	vle8.v	v14, (a5)
	vsrl.vi	v18, v18, 4
	ld	a4, 1768(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v18
	vand.vi	v18, v4, 15
	ld	a4, 1760(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v18
	vand.vi	v18, v20, 15
	ld	a4, 1976(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v18
	addi	a5, s0, 680
	vle8.v	v18, (a5)
	vsrl.vi	v20, v20, 4
	ld	a4, 1776(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v20
	vand.vi	v20, v14, 15
	ld	a4, 1744(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v20
	vand.vi	v20, v19, 15
	ld	a4, 1984(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v20
	addi	a5, s0, 440
	vle8.v	v20, (a5)
	vsrl.vi	v19, v19, 4
	ld	a4, 1736(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v19
	vand.vi	v19, v18, 15
	ld	a4, 1728(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v19
	vand.vi	v19, v20, 15
	ld	a4, 1720(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	addi	a5, s0, 696
	vle8.v	v26, (a5)
	vsrl.vi	v19, v20, 4
	addi	a5, s0, 456
	vle8.v	v20, (a5)
	ld	a4, 1712(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v19
	vand.vi	v19, v26, 15
	ld	a4, 1632(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v19
	vand.vi	v19, v20, 15
	ld	a4, 1640(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vsrl.vi	v19, v20, 4
	addi	a5, s0, 712
	vle8.v	v27, (a5)
	addi	a5, s0, 472
	vle8.v	v20, (a5)
	ld	a4, 1664(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v19
	vand.vi	v19, v27, 15
	ld	a4, 1656(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v19
	vand.vi	v19, v20, 15
	ld	a4, 1680(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vsrl.vi	v19, v20, 4
	addi	a5, s0, 728
	vle8.v	v22, (a5)
	ld	a4, 1672(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v19
	addi	a5, s0, 488
	vle8.v	v19, (a5)
	vand.vi	v20, v22, 15
	ld	a4, 1688(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v20
	addi	a5, s0, 744
	vle8.v	v3, (a5)
	vand.vi	v20, v19, 15
	ld	a4, 1696(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v20
	vsrl.vi	v19, v19, 4
	ld	a4, 1600(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v19
	vand.vi	v19, v3, 15
	ld	a4, 1592(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v19
	addi	a5, s0, 72
	addi	a4, s0, 200
	vle8.v	v21, (a4)
	addi	a4, s0, 504
	vle8.v	v23, (a5)
	csrr	a5, vlenb
	slli	ra, a5, 1
	add	a5, a5, ra
	add	a5, a5, sp
	addi	a5, a5, 2000
	vs1r.v	v23, (a5)                       # Unknown-size Folded Spill
	vle8.v	v19, (a4)
	vand.vi	v20, v21, 3
	vmv1r.v	v30, v21
	vsll.vi	v20, v20, 4
	vand.vi	v24, v23, 15
	vor.vv	v24, v20, v24
	vand.vi	v20, v19, 15
	ld	a4, 1648(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v20
	vsrl.vi	v19, v19, 4
	ld	a4, 1616(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v19
	addi	a4, s0, 88
	addi	a5, s0, 216
	vle8.v	v21, (a5)
	addi	a5, s0, 760
	vle8.v	v20, (a4)
	csrr	a4, vlenb
	slli	a4, a4, 1
	add	a4, a4, sp
	addi	a4, a4, 2000
	vs1r.v	v20, (a4)                       # Unknown-size Folded Spill
	vle8.v	v19, (a5)
	vand.vi	v25, v21, 3
	vsll.vi	v25, v25, 4
	vand.vi	v7, v20, 15
	vor.vv	v25, v25, v7
	vand.vi	v7, v19, 15
	ld	a4, 1704(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v7
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v2, v24
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v6, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v6, v2, v9
	vzext.vf2	v24, v25
	vwmacc.vv	v6, v24, v12
	addi	a4, s0, 776
	vle8.v	v9, (a4)
	vwmacc.vv	v6, v2, v13
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v13, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a4, 1608(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v9, 15
	ld	a4, 1416(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v12
	vsrl.vi	v12, v9, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v9, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a4, 1408(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v12
	vsrl.vi	v10, v10, 4
	ld	a4, 1624(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v11, 4
	ld	a4, 1584(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v15, 4
	ld	a4, 1576(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v16, 4
	ld	a4, 1568(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v17, 4
	ld	a4, 1560(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v31, 4
	ld	a4, 1552(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v5, 4
	ld	a4, 1544(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v4, 4
	ld	a4, 1536(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v14, 4
	ld	a4, 1528(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vsrl.vi	v10, v18, 4
	ld	a4, 1520(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	addi	a4, s0, 792
	vle8.v	v14, (a4)
	vsrl.vi	v10, v26, 4
	ld	a4, 1512(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	addi	a4, s0, 808
	vle8.v	v16, (a4)
	vsrl.vi	v10, v27, 4
	ld	a4, 1504(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	addi	a4, s0, 824
	vle8.v	v17, (a4)
	vsrl.vi	v10, v22, 4
	ld	a4, 1496(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	addi	a4, s0, 840
	vle8.v	v18, (a4)
	vsrl.vi	v10, v3, 4
	ld	a4, 1488(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	addi	a4, s0, 1032
	vle8.v	v10, (a4)
	vsrl.vi	v11, v19, 4
	ld	a4, 1480(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v11
	addi	a4, s0, 1048
	vle8.v	v12, (a4)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v6, v24, v13
	vmv.v.i	v11, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v10, 15
	ld	a4, 1472(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v13
	vand.vi	v13, v14, 15
	ld	a4, 1456(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v13
	addi	a4, s0, 1064
	vle8.v	v13, (a4)
	vsrl.vi	v14, v14, 4
	ld	a4, 1448(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	vand.vi	v14, v12, 15
	ld	a4, 1440(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v14
	vand.vi	v14, v16, 15
	ld	a4, 1464(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v14
	addi	a4, s0, 1080
	vle8.v	v15, (a4)
	vsrl.vi	v14, v16, 4
	ld	a4, 1432(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	vand.vi	v14, v13, 15
	ld	a4, 1424(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v14
	vand.vi	v14, v17, 15
	ld	a4, 1400(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v14
	addi	a4, s0, 1096
	vle8.v	v16, (a4)
	vsrl.vi	v14, v17, 4
	ld	a4, 1392(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	vand.vi	v14, v15, 15
	ld	a4, 1328(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v14
	vand.vi	v14, v18, 15
	ld	a4, 1352(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v14
	addi	a4, s0, 856
	vle8.v	v14, (a4)
	vsrl.vi	v17, v18, 4
	ld	a4, 1344(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v17
	vand.vi	v17, v16, 15
	ld	a4, 1336(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v17
	vand.vi	v17, v14, 15
	ld	a4, 1360(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v17
	addi	a4, s0, 1112
	vle8.v	v17, (a4)
	vsrl.vi	v14, v14, 4
	addi	a4, s0, 872
	vle8.v	v18, (a4)
	ld	a4, 1376(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	vand.vi	v14, v17, 15
	ld	a4, 1368(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v14
	vand.vi	v14, v18, 15
	ld	a4, 1384(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v14
	vsrl.vi	v14, v18, 4
	addi	a4, s0, 1128
	vle8.v	v2, (a4)
	addi	a4, s0, 888
	vle8.v	v18, (a4)
	ld	a4, 1320(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	vand.vi	v14, v2, 15
	ld	a4, 1312(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v14
	vand.vi	v14, v18, 15
	ld	a4, 1304(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v14
	vsrl.vi	v14, v18, 4
	addi	a4, s0, 1144
	vle8.v	v1, (a4)
	addi	a4, s0, 904
	vle8.v	v18, (a4)
	ld	a4, 1296(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v14
	vand.vi	v14, v1, 15
	ld	a4, 1216(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v14
	vand.vi	v14, v18, 15
	ld	a4, 1224(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v14
	vsrl.vi	v18, v18, 4
	addi	a4, s0, 1160
	vle8.v	v14, (a4)
	addi	a4, s0, 920
	vle8.v	v19, (a4)
	ld	a4, 1240(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v18
	vand.vi	v18, v14, 15
	ld	a4, 1232(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v18
	vand.vi	v18, v19, 15
	ld	a4, 1248(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v18
	vsrl.vi	v19, v19, 4
	addi	a4, s0, 1176
	vle8.v	v18, (a4)
	addi	a4, s0, 936
	vle8.v	v22, (a4)
	ld	a4, 1272(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vand.vi	v19, v18, 15
	ld	a4, 1264(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v19
	vand.vi	v19, v22, 15
	ld	a4, 1280(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v19
	vsrl.vi	v19, v22, 4
	addi	a4, s0, 1192
	vle8.v	v26, (a4)
	addi	a4, s0, 952
	vle8.v	v22, (a4)
	ld	a4, 1288(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vand.vi	v19, v26, 15
	ld	a4, 1256(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v19
	vand.vi	v19, v22, 15
	ld	a4, 1208(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v19
	vsrl.vi	v19, v22, 4
	addi	a4, s0, 1208
	vle8.v	v27, (a4)
	addi	a4, s0, 968
	vle8.v	v22, (a4)
	ld	a4, 1200(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vand.vi	v19, v27, 15
	ld	a4, 1144(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v19
	vand.vi	v19, v22, 15
	ld	a4, 1160(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v19
	vsrl.vi	v19, v22, 4
	addi	a4, s0, 1224
	vle8.v	v22, (a4)
	addi	a4, s0, 984
	vle8.v	v24, (a4)
	ld	a4, 1152(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vand.vi	v19, v22, 15
	ld	a4, 1136(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v19
	vand.vi	v19, v24, 15
	ld	a4, 1168(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v19
	vsrl.vi	v19, v24, 4
	addi	a4, s0, 1240
	vle8.v	v24, (a4)
	ld	a4, 1120(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	addi	a4, s0, 1000
	vle8.v	v19, (a4)
	vand.vi	v25, v24, 15
	ld	a4, 1128(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v25
	addi	a4, s0, 1256
	vle8.v	v25, (a4)
	vand.vi	v31, v19, 15
	ld	a4, 1184(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v31
	vsrl.vi	v19, v19, 4
	ld	a4, 1176(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	vand.vi	v19, v25, 15
	ld	a4, 1104(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v19
	addi	a4, s0, 104
	addi	a5, s0, 232
	vle8.v	v31, (a5)
	addi	a5, s0, 1016
	vle8.v	v20, (a4)
	csrr	a4, vlenb
	add	a4, a4, sp
	addi	a4, a4, 2000
	vs1r.v	v20, (a4)                       # Unknown-size Folded Spill
	vle8.v	v19, (a5)
	vand.vi	v5, v31, 3
	vsll.vi	v5, v5, 4
	vand.vi	v3, v20, 15
	vor.vv	v0, v5, v3
	vand.vi	v5, v19, 15
	ld	a4, 1192(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v8, a4, v5
	vsrl.vi	v19, v19, 4
	ld	a4, 1080(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v9, a4, v19
	addi	a4, s0, 120
	addi	a5, s0, 248
	vle8.v	v5, (a5)
	addi	a5, s0, 1272
	vle8.v	v20, (a4)
	addi	a4, sp, 2000
	vs1r.v	v20, (a4)                       # Unknown-size Folded Spill
	vle8.v	v19, (a5)
	vand.vi	v23, v5, 3
	vsll.vi	v23, v23, 4
	vand.vi	v28, v20, 15
	vor.vv	v23, v23, v28
	vand.vi	v28, v19, 15
	ld	a4, 1112(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v28
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v28, v0
	vwmacc.vv	v6, v28, v8
	vzext.vf2	v8, v23
	vwmacc.vv	v6, v8, v9
	addi	a4, s0, 1288
	vle8.v	v9, (a4)
	vwmacc.vv	v6, v28, v11
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v10, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v23, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a4, 1088(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v9, 15
	ld	a4, 896(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v11
	vsrl.vi	v9, v9, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v11, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a4, 888(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vsrl.vi	v9, v12, 4
	ld	a4, 1096(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v13, 4
	ld	a4, 1072(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v15, 4
	ld	a4, 1064(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v16, 4
	ld	a4, 1056(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v17, 4
	ld	a4, 1048(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v2, 4
	ld	a4, 1040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v1, 4
	ld	a4, 1032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v14, 4
	ld	a4, 1024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v18, 4
	ld	a4, 1016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v26, 4
	ld	a4, 1008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v27, 4
	ld	a4, 1000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v22, 4
	ld	a4, 992(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v24, 4
	ld	a4, 984(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	vsrl.vi	v9, v25, 4
	ld	a4, 976(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	addi	a4, s0, 1544
	vle8.v	v12, (a4)
	vsrl.vi	v9, v19, 4
	ld	a4, 968(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v23, a4, v9
	addi	a4, s0, 1304
	vle8.v	v9, (a4)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v6, v8, v23
	vmv.v.i	v13, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v12, 15
	ld	a4, 928(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 936(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	addi	a4, s0, 1560
	vle8.v	v15, (a4)
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1320
	vle8.v	v9, (a4)
	ld	a4, 952(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v8
	vand.vi	v8, v15, 15
	ld	a4, 944(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 960(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1576
	vle8.v	v16, (a4)
	addi	a4, s0, 1336
	vle8.v	v9, (a4)
	ld	a4, 920(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v8
	vand.vi	v8, v16, 15
	ld	a4, 912(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 904(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1592
	vle8.v	v17, (a4)
	addi	a4, s0, 1352
	vle8.v	v9, (a4)
	ld	a4, 880(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v8
	vand.vi	v8, v17, 15
	ld	a4, 816(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 824(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1608
	vle8.v	v2, (a4)
	addi	a4, s0, 1368
	vle8.v	v9, (a4)
	ld	a4, 840(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v8
	vand.vi	v8, v2, 15
	ld	a4, 832(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 848(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1624
	vle8.v	v1, (a4)
	addi	a4, s0, 1384
	vle8.v	v9, (a4)
	ld	a4, 864(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v8
	vand.vi	v8, v1, 15
	ld	a4, 856(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 872(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1640
	vle8.v	v14, (a4)
	addi	a4, s0, 1400
	vle8.v	v9, (a4)
	ld	a4, 808(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v8
	vand.vi	v8, v14, 15
	ld	a4, 800(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 792(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v8
	vsrl.vi	v9, v9, 4
	addi	a4, s0, 1656
	vle8.v	v8, (a4)
	addi	a4, s0, 1416
	vle8.v	v18, (a4)
	ld	a4, 784(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vand.vi	v9, v8, 15
	ld	a4, 704(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v9
	vand.vi	v9, v18, 15
	ld	a4, 712(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v9
	vsrl.vi	v9, v18, 4
	addi	a4, s0, 1672
	vle8.v	v26, (a4)
	addi	a4, s0, 1432
	vle8.v	v18, (a4)
	ld	a4, 752(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vand.vi	v9, v26, 15
	ld	a4, 744(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v9
	vand.vi	v9, v18, 15
	ld	a4, 760(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v9
	vsrl.vi	v9, v18, 4
	addi	a4, s0, 1688
	vle8.v	v27, (a4)
	addi	a4, s0, 1448
	vle8.v	v18, (a4)
	ld	a4, 768(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vand.vi	v9, v27, 15
	ld	a4, 736(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v9
	vand.vi	v9, v18, 15
	ld	a4, 720(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v9
	vsrl.vi	v9, v18, 4
	addi	a4, s0, 1704
	vle8.v	v22, (a4)
	addi	a4, s0, 1464
	vle8.v	v18, (a4)
	ld	a4, 776(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vand.vi	v9, v22, 15
	ld	a4, 728(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v9
	vand.vi	v9, v18, 15
	ld	a4, 696(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v9
	vsrl.vi	v9, v18, 4
	addi	a4, s0, 1720
	vle8.v	v24, (a4)
	addi	a4, s0, 1480
	vle8.v	v18, (a4)
	ld	a4, 688(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vand.vi	v9, v24, 15
	ld	a4, 632(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v9
	vand.vi	v9, v18, 15
	ld	a4, 648(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v9
	vsrl.vi	v9, v18, 4
	addi	a4, s0, 1736
	vle8.v	v25, (a4)
	addi	a4, s0, 1496
	vle8.v	v18, (a4)
	ld	a4, 640(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v9
	vand.vi	v9, v25, 15
	ld	a4, 624(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v9
	vand.vi	v9, v18, 15
	ld	a4, 664(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v9
	vsrl.vi	v18, v18, 4
	addi	a4, s0, 1752
	vle8.v	v9, (a4)
	addi	a4, s0, 1512
	vle8.v	v19, (a4)
	ld	a4, 656(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v18
	vand.vi	v18, v9, 15
	ld	a4, 616(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v18
	vand.vi	v18, v19, 15
	ld	a4, 672(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v18
	vsrl.vi	v18, v19, 4
	addi	a4, s0, 1768
	vle8.v	v23, (a4)
	addi	a4, s0, 1528
	vle8.v	v19, (a4)
	ld	a4, 680(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v18
	vand.vi	v18, v23, 15
	ld	a4, 608(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v18
	vand.vi	v18, v19, 15
	ld	a4, 600(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a4, v18
	vsrl.vi	v18, v19, 4
	addi	a4, s0, 1784
	vle8.v	v28, (a4)
	ld	a4, 584(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a4, v18
	addi	a4, s0, 136
	vle8.v	v20, (a4)
	vand.vi	v19, v28, 15
	ld	a4, 592(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v19
	addi	a4, s0, 152
	vle8.v	v19, (a4)
	vand.vi	v0, v20, 15
	li	a4, 48
	vand.vx	v29, v30, a4
	vmv1r.v	v3, v30
	vor.vv	v29, v29, v0
	vand.vi	v0, v19, 15
	li	a4, 48
	vand.vx	v30, v21, a4
	vor.vv	v30, v30, v0
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v0, v29
	vwmacc.vv	v6, v0, v10
	vzext.vf2	v10, v30
	vwmacc.vv	v6, v10, v11
	vwmacc.vv	v6, v0, v13
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v12, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v12, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a4, 568(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v11, v15, 4
	ld	a4, 560(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v11, v16, 4
	ld	a4, 576(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v11, v17, 4
	ld	a4, 552(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v11, v2, 4
	ld	a4, 544(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v11, v1, 4
	ld	a4, 536(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v11, v14, 4
	ld	a4, 528(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v11
	vsrl.vi	v8, v8, 4
	ld	a4, 520(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v26, 4
	ld	a4, 512(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v27, 4
	ld	a4, 504(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v22, 4
	ld	a4, 496(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v24, 4
	ld	a4, 488(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v25, 4
	ld	a4, 480(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	ld	a4, 472(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v23, 4
	ld	a4, 464(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	addi	a4, s0, 1800
	vle8.v	v8, (a4)
	vsrl.vi	v9, v28, 4
	ld	a4, 456(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v9
	addi	a4, s0, 1816
	vle8.v	v9, (a4)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v6, v10, v12
	vmv.v.i	v12, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v8, 15
	ld	a4, 448(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v13, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	ld	a4, 424(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 440(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	addi	a4, s4, 9
	vle8.v	v18, (a4)
	vsrl.vi	v8, v9, 4
	addi	a4, s0, 1832
	vle8.v	v9, (a4)
	ld	a4, 432(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v18, 15
	ld	a4, 392(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 400(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s4, 25
	vle8.v	v17, (a4)
	addi	a4, s0, 1848
	vle8.v	v9, (a4)
	ld	a4, 416(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v17, 15
	ld	a4, 408(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 384(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s4, 41
	vle8.v	v2, (a4)
	addi	a4, s0, 1864
	vle8.v	v9, (a4)
	ld	a4, 360(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v2, 15
	ld	a4, 376(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 368(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s4, 57
	vle8.v	v1, (a4)
	addi	a4, s0, 1880
	vle8.v	v9, (a4)
	ld	a4, 352(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v1, 15
	ld	a4, 296(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 304(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s4, 73
	vle8.v	v0, (a4)
	addi	a4, s0, 1896
	vle8.v	v9, (a4)
	ld	a4, 320(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v0, 15
	ld	a4, 312(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 328(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s4, 89
	vle8.v	v15, (a4)
	addi	a4, s0, 1912
	vle8.v	v9, (a4)
	ld	a4, 344(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v15, 15
	ld	a4, 336(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v9, 15
	ld	a4, 288(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v9, 4
	addi	a4, s4, 105
	vle8.v	v14, (a4)
	addi	a4, s0, 1928
	vle8.v	v10, (a4)
	ld	a4, 272(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v14, 15
	ld	a4, 264(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v10, 15
	ld	a4, 256(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v10, 4
	addi	a4, s4, 121
	vle8.v	v26, (a4)
	addi	a4, s0, 1944
	vle8.v	v10, (a4)
	ld	a4, 280(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v26, 15
	ld	a4, 216(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v10, 15
	ld	a4, 224(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v10, 4
	addi	a4, s4, 137
	vle8.v	v27, (a4)
	addi	a4, s0, 1960
	vle8.v	v10, (a4)
	ld	a4, 240(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v27, 15
	ld	a4, 208(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v10, 15
	ld	a4, 200(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v10, 4
	addi	a4, s4, 153
	vle8.v	v22, (a4)
	addi	a4, s0, 1976
	vle8.v	v10, (a4)
	ld	a4, 248(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v22, 15
	ld	a4, 232(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v10, 15
	ld	a4, 192(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v10, 4
	addi	a4, s4, 169
	vle8.v	v24, (a4)
	addi	a4, s0, 1992
	vle8.v	v10, (a4)
	ld	a4, 176(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v24, 15
	ld	a4, 168(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v10, 15
	ld	a4, 160(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v8, v10, 4
	addi	a4, s4, 185
	vle8.v	v25, (a4)
	addi	a4, s0, 2008
	vle8.v	v10, (a4)
	ld	a4, 184(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v8
	vand.vi	v8, v25, 15
	ld	a4, 96(sp)                      # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v8
	vand.vi	v8, v10, 15
	ld	a4, 112(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v8
	vsrl.vi	v10, v10, 4
	addi	a4, s4, 201
	vle8.v	v9, (a4)
	addi	a4, s0, 2024
	vle8.v	v11, (a4)
	ld	a4, 104(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vand.vi	v10, v9, 15
	vwmacc.vx	v16, s11, v10
	vand.vi	v10, v11, 15
	ld	a4, 120(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a4, v10
	vsrl.vi	v10, v11, 4
	addi	a4, s4, 217
	vle8.v	v23, (a4)
	addi	a4, s0, 2040
	vle8.v	v11, (a4)
	ld	a4, 144(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v13, a4, v10
	vand.vi	v10, v23, 15
	vwmacc.vx	v16, s10, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v12, s5, v10
	addi	a4, s4, 233
	vsrl.vi	v10, v11, 4
	vle8.v	v28, (a4)
	vwmacc.vx	v13, s3, v10
	li	a5, 48
	addi	a4, s0, 168
	vle8.v	v10, (a4)
	vand.vi	v11, v28, 15
	vwmacc.vx	v16, s9, v11
	addi	a4, s0, 184
	vle8.v	v11, (a4)
	vand.vi	v29, v10, 15
	vand.vx	v30, v31, a5
	vor.vv	v29, v30, v29
	addi	a4, s4, 249
	vle8.v	v30, (a4)
	vand.vi	v8, v11, 15
	vand.vx	v4, v5, a5
	vor.vv	v8, v4, v8
	vand.vi	v4, v30, 15
	vwmacc.vx	v16, s2, v4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v4, v29
	vmv.v.i	v29, 0
	vwmacc.vv	v6, v4, v12
	vzext.vf2	v12, v8
	vwmacc.vv	v6, v12, v13
	vwmacc.vv	v6, v4, v16
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v18, 4
	vmv1r.v	v13, v29
	vwmacc.vx	v13, s6, v8
	vsrl.vi	v8, v17, 4
	vwmacc.vx	v13, s7, v8
	vsrl.vi	v8, v2, 4
	vwmacc.vx	v13, s8, v8
	vsrl.vi	v8, v1, 4
	vwmacc.vx	v13, t3, v8
	vsrl.vi	v8, v0, 4
	vwmacc.vx	v13, t4, v8
	vsrl.vi	v8, v15, 4
	vwmacc.vx	v13, t5, v8
	vsrl.vi	v8, v14, 4
	vwmacc.vx	v13, t6, v8
	vsrl.vi	v8, v26, 4
	vwmacc.vx	v13, a7, v8
	vsrl.vi	v8, v27, 4
	vwmacc.vx	v13, t0, v8
	vsrl.vi	v8, v22, 4
	vwmacc.vx	v13, t1, v8
	vsrl.vi	v8, v24, 4
	vwmacc.vx	v13, t2, v8
	li	t2, -64
	vsrl.vi	v8, v25, 4
	vwmacc.vx	v13, a1, v8
	vsrl.vi	v8, v9, 4
	vwmacc.vx	v13, a2, v8
	vsrl.vi	v8, v23, 4
	vwmacc.vx	v13, a3, v8
	vsrl.vi	v8, v28, 4
	vwmacc.vx	v13, a6, v8
	addi	a1, s0, 16
	vle16.v	v8, (a1)
	vsrl.vi	v9, v30, 4
	vwmacc.vx	v13, a0, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v6, v12, v13
	vfwcvt.f.f.v	v12, v8
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v8, v12, fa5
	vfcvt.f.x.v	v12, v6
	csrr	a0, vlenb
	slli	a1, a0, 4
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl2r.v	v14, (a0)                       # Unknown-size Folded Reload
	vfmadd.vv	v12, v8, v14
	csrr	a0, vlenb
	li	a1, 14
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v8, (a0)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	csrr	a0, vlenb
	li	a1, 18
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v18, (a0)                       # Unknown-size Folded Reload
	vand.vi	v9, v18, 12
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v14, (a0)                       # Unknown-size Folded Reload
	vsrl.vi	v14, v14, 4
	vsll.vi	v9, v9, 2
	vor.vv	v15, v9, v8
	csrr	a0, vlenb
	slli	a1, a0, 4
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vand.vi	v8, v22, 12
	csrr	a0, vlenb
	li	a1, 19
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v9, (a0)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	vsll.vi	v8, v8, 2
	vor.vv	v14, v8, v14
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v23, (a0)                       # Unknown-size Folded Reload
	vand.vi	v8, v23, 12
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v16, (a0)                       # Unknown-size Folded Reload
	vsrl.vi	v16, v16, 4
	vsll.vi	v8, v8, 2
	vor.vv	v17, v8, v9
	csrr	a0, vlenb
	slli	a1, a0, 2
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v24, (a0)                       # Unknown-size Folded Reload
	vand.vi	v8, v24, 12
	vsll.vi	v8, v8, 2
	vor.vv	v16, v8, v16
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v8, (a0)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vx	v9, v18, t2
	vsrl.vi	v9, v9, 2
	vor.vv	v18, v9, v8
	csrr	a0, vlenb
	slli	a1, a0, 3
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v8, (a0)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vx	v9, v22, t2
	vsrl.vi	v9, v9, 2
	vor.vv	v22, v9, v8
	csrr	a0, vlenb
	slli	a1, a0, 3
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v8, (a0)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vx	v9, v23, t2
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v23, (a0)                       # Unknown-size Folded Reload
	vsrl.vi	v23, v23, 4
	vand.vx	v24, v24, t2
	vsrl.vi	v9, v9, 2
	vsrl.vi	v24, v24, 2
	vor.vv	v25, v9, v8
	vor.vv	v23, v24, v23
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v26, 0
	vmv.v.i	v8, 0
	ld	a1, 136(sp)                     # 8-byte Folded Reload
	ld	a0, 128(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v24, v15
	vwmacc.vx	v8, a1, v24
	lh	a0, 266(s1)
	lh	a5, 268(s1)
	lh	a2, 270(s1)
	lh	a6, 272(s1)
	ld	t1, 152(sp)                     # 8-byte Folded Reload
	add	t1, t1, a0
	vzext.vf2	v15, v14
	vwmacc.vx	v8, t1, v15
	add	t0, a2, a5
	vzext.vf2	v14, v17
	vwmacc.vx	v8, t0, v14
	lh	a2, 274(s1)
	lh	a7, 276(s1)
	lh	a3, 278(s1)
	lh	a0, 280(s1)
	add	a6, a6, a2
	vzext.vf2	v14, v16
	vwmacc.vx	v8, a6, v14
	add	a7, a7, a3
	vzext.vf2	v14, v18
	vwmacc.vx	v8, a7, v14
	lh	a2, 282(s1)
	lh	a5, 284(s1)
	lh	a3, 286(s1)
	lh	a4, 288(s1)
	add	a2, a2, a0
	vzext.vf2	v14, v22
	vwmacc.vx	v8, a2, v14
	add	a3, a3, a5
	addi	a0, s0, 32
	lh	a5, 290(s1)
	vzext.vf2	v14, v25
	vwmacc.vx	v8, a3, v14
	vle16.v	v14, (a0)
	add	a4, a4, a5
	li	a5, 292
	vzext.vf2	v15, v23
	vwmacc.vx	v8, a4, v15
	vfwcvt.f.f.v	v16, v14
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v14, v16, fa5
	vfcvt.f.x.v	v8, v8
	csrr	a0, vlenb
	li	t3, 11
	mul	a0, a0, t3
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl2r.v	v22, (a0)                       # Unknown-size Folded Reload
	vfnmsac.vv	v22, v8, v14
	csrr	a0, vlenb
	slli	t3, a0, 1
	add	a0, a0, t3
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v8, (a0)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	vand.vi	v9, v3, 12
	vsll.vi	v9, v9, 2
	vor.vv	v14, v9, v8
	vmv2r.v	v8, v26
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vwmacc.vx	v8, a1, v15
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v14, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v14, v14, 4
	vand.vi	v15, v21, 12
	vsll.vi	v15, v15, 2
	vor.vv	v14, v15, v14
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vwmacc.vx	v8, t1, v15
	csrr	a0, vlenb
	add	a0, a0, sp
	addi	a0, a0, 2000
	vl1r.v	v14, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v14, v14, 4
	vand.vi	v15, v31, 12
	vsll.vi	v15, v15, 2
	vor.vv	v14, v15, v14
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	vwmacc.vx	v8, t0, v15
	addi	a0, sp, 2000
	vl1r.v	v14, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v14, v14, 4
	vand.vi	v15, v5, 12
	vsll.vi	v15, v15, 2
	vor.vv	v14, v15, v14
	vsrl.vi	v15, v20, 4
	vand.vx	v16, v3, t2
	vsrl.vi	v16, v16, 2
	vor.vv	v15, v16, v15
	vsrl.vi	v16, v19, 4
	vand.vx	v17, v21, t2
	vsrl.vi	v17, v17, 2
	vor.vv	v16, v17, v16
	vsrl.vi	v10, v10, 4
	vand.vx	v17, v31, t2
	vsrl.vi	v17, v17, 2
	vor.vv	v10, v17, v10
	vsrl.vi	v11, v11, 4
	vand.vx	v17, v5, t2
	vsrl.vi	v17, v17, 2
	vor.vv	v11, v17, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v14
	vwmacc.vx	v8, a6, v17
	ld	s1, 64(sp)                      # 8-byte Folded Reload
	vzext.vf2	v14, v15
	vwmacc.vx	v8, a7, v14
	addi	a0, s0, 48
	vzext.vf2	v14, v16
	vwmacc.vx	v8, a2, v14
	ld	a2, 1952(sp)                    # 8-byte Folded Reload
	vle16.v	v14, (a0)
	vzext.vf2	v15, v10
	vwmacc.vx	v8, a3, v15
	ld	a3, 80(sp)                      # 8-byte Folded Reload
	vzext.vf2	v10, v11
	vwmacc.vx	v8, a4, v10
	ld	a4, 72(sp)                      # 8-byte Folded Reload
	vfwcvt.f.f.v	v10, v14
	vmv2r.v	v14, v22
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v10, v10, fa5
	vfcvt.f.x.v	v8, v8
	vfnmsac.vv	v12, v8, v10
	addi	a2, a2, 1
	vmv.v.v	v10, v22
	vmv.v.v	v8, v12
	beq	a2, a4, .LBB0_7
	j	.LBB0_5
.LBB0_7:                                #   in Loop: Header=BB0_3 Depth=1
	j	.LBB0_2
.LBB0_6:
	csrr	a0, vlenb
	li	a1, 20
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 496
	addi	sp, sp, 1616
	.cfi_def_cfa_offset 496
	ld	ra, 488(sp)                     # 8-byte Folded Reload
	ld	s0, 480(sp)                     # 8-byte Folded Reload
	ld	s1, 472(sp)                     # 8-byte Folded Reload
	ld	s2, 464(sp)                     # 8-byte Folded Reload
	ld	s3, 456(sp)                     # 8-byte Folded Reload
	ld	s4, 448(sp)                     # 8-byte Folded Reload
	ld	s5, 440(sp)                     # 8-byte Folded Reload
	ld	s6, 432(sp)                     # 8-byte Folded Reload
	ld	s7, 424(sp)                     # 8-byte Folded Reload
	ld	s8, 416(sp)                     # 8-byte Folded Reload
	ld	s9, 408(sp)                     # 8-byte Folded Reload
	ld	s10, 400(sp)                    # 8-byte Folded Reload
	ld	s11, 392(sp)                    # 8-byte Folded Reload
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
	addi	sp, sp, 496
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K, .Lfunc_end0-weft_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
