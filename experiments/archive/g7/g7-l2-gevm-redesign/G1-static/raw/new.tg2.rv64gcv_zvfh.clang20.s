	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"new.c"
	.text
	.globl	weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K # -- Begin function weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K
	.p2align	1
	.type	weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K,@function
weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K: # @weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -2032
	.cfi_def_cfa_offset 2032
	sd	ra, 2024(sp)                    # 8-byte Folded Spill
	sd	s0, 2016(sp)                    # 8-byte Folded Spill
	sd	s1, 2008(sp)                    # 8-byte Folded Spill
	sd	s2, 2000(sp)                    # 8-byte Folded Spill
	sd	s3, 1992(sp)                    # 8-byte Folded Spill
	sd	s4, 1984(sp)                    # 8-byte Folded Spill
	sd	s5, 1976(sp)                    # 8-byte Folded Spill
	sd	s6, 1968(sp)                    # 8-byte Folded Spill
	sd	s7, 1960(sp)                    # 8-byte Folded Spill
	sd	s8, 1952(sp)                    # 8-byte Folded Spill
	sd	s9, 1944(sp)                    # 8-byte Folded Spill
	sd	s10, 1936(sp)                   # 8-byte Folded Spill
	sd	s11, 1928(sp)                   # 8-byte Folded Spill
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
	addi	sp, sp, -544
	.cfi_def_cfa_offset 2576
	csrr	a5, vlenb
	li	a6, 48
	mul	a5, a5, a6
	sub	sp, sp, a5
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0x90, 0x14, 0x22, 0x11, 0x30, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 2576 + 48 * vlenb
	sd	a2, 32(sp)                      # 8-byte Folded Spill
	sd	a1, 48(sp)                      # 8-byte Folded Spill
	srli	a4, a4, 5
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	sd	a4, 40(sp)                      # 8-byte Folded Spill
	bnez	a4, .LBB0_1
	j	.LBB0_6
.LBB0_1:
	li	a4, 0
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	srli	a5, a0, 8
	li	a0, 9
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v31, 0
	li	s1, 292
	li	t4, 1
	lui	a1, 1
	li	a2, 17
	slli	s0, a0, 8
	slli	t4, t4, 11
	addiw	a0, a1, -2040
	sd	a0, 520(sp)                     # 8-byte Folded Spill
	addiw	t5, a1, -2032
	addiw	a0, a1, -2024
	sd	a0, 504(sp)                     # 8-byte Folded Spill
	addiw	t6, a1, -2016
	addiw	a0, a1, -2008
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	addiw	s3, a1, -2000
	addiw	a0, a1, -1992
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	addiw	s4, a1, -1984
	addiw	a0, a1, -1976
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	mul	a0, a5, s0
	sd	a0, 24(sp)                      # 8-byte Folded Spill
	slli	s5, a2, 7
	sd	a3, 96(sp)                      # 8-byte Folded Spill
	sd	a5, 88(sp)                      # 8-byte Folded Spill
	sd	s0, 80(sp)                      # 8-byte Folded Spill
	sd	t4, 528(sp)                     # 8-byte Folded Spill
	sd	t5, 512(sp)                     # 8-byte Folded Spill
	sd	t6, 496(sp)                     # 8-byte Folded Spill
	sd	s3, 480(sp)                     # 8-byte Folded Spill
	sd	s4, 464(sp)                     # 8-byte Folded Spill
	sd	s5, 448(sp)                     # 8-byte Folded Spill
	j	.LBB0_3
.LBB0_2:                                #   in Loop: Header=BB0_3 Depth=1
	ld	a4, 64(sp)                      # 8-byte Folded Reload
	slli	a0, a4, 7
	ld	a1, 72(sp)                      # 8-byte Folded Reload
	slli	a1, a1, 6
	addi	a4, a4, 1
	ld	a2, 48(sp)                      # 8-byte Folded Reload
	add	a0, a0, a2
	add	a1, a1, a2
	vsetivli	zero, 8, e32, m2, ta, ma
	vse32.v	v12, (a0)
	addi	a0, a0, 32
	vse32.v	v16, (a0)
	addi	a0, a1, 32
	vse32.v	v20, (a1)
	vse32.v	v0, (a0)
	vmv.v.i	v8, 0
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	bne	a4, a0, .LBB0_3
	j	.LBB0_6
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_5 Depth 2
	sd	a4, 64(sp)                      # 8-byte Folded Spill
	slli	a0, a4, 1
	addi	a1, a0, 1
	sd	a1, 72(sp)                      # 8-byte Folded Spill
	vsetivli	zero, 1, e8, m1, ta, ma
	vmv2r.v	v0, v8
	vmv2r.v	v20, v8
	vmv2r.v	v16, v8
	vmv2r.v	v12, v8
	ld	a1, 56(sp)                      # 8-byte Folded Reload
	li	a2, 256
	bltu	a1, a2, .LBB0_2
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	li	a4, 0
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	mul	a0, a1, a0
	ld	a2, 72(sp)                      # 8-byte Folded Reload
	mul	a1, a1, a2
	ld	a2, 32(sp)                      # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 112(sp)                     # 8-byte Folded Spill
	add	a1, a1, a2
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v8, 0
	vmv.v.i	v18, 0
	vmv.v.i	v14, 0
	vmv.v.i	v10, 0
.LBB0_5:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	sd	a4, 440(sp)                     # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs2r.v	v18, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 20
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs2r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 42
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs2r.v	v10, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 46
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	mul	ra, a4, s1
	mul	a0, a4, s0
	sd	a0, 408(sp)                     # 8-byte Folded Spill
	vmv1r.v	v11, v31
	vmv1r.v	v16, v31
	vmv1r.v	v17, v31
	add	ra, ra, a3
	ld	s11, 112(sp)                    # 8-byte Folded Reload
	add	s11, s11, a0
	addi	a0, s11, 64
	addi	a1, s11, 192
	addi	a2, s11, 80
	addi	a4, s11, 208
	addi	t2, s11, 256
	addi	t0, s11, 272
	addi	a6, s11, 288
	lbu	a3, 4(ra)
	sd	a3, 544(sp)                     # 8-byte Folded Spill
	lbu	s3, 5(ra)
	sd	s3, 552(sp)                     # 8-byte Folded Spill
	lbu	s4, 6(ra)
	sd	s4, 800(sp)                     # 8-byte Folded Spill
	lbu	s2, 36(ra)
	sd	s2, 560(sp)                     # 8-byte Folded Spill
	lbu	s5, 37(ra)
	sd	s5, 584(sp)                     # 8-byte Folded Spill
	lbu	t4, 38(ra)
	sd	t4, 792(sp)                     # 8-byte Folded Spill
	addi	a5, s11, 304
	addi	s0, s11, 320
	addi	t1, s11, 336
	addi	a7, s11, 352
	lbu	t5, 7(ra)
	sd	t5, 784(sp)                     # 8-byte Folded Spill
	lbu	s6, 8(ra)
	sd	s6, 704(sp)                     # 8-byte Folded Spill
	lbu	s8, 9(ra)
	sd	s8, 728(sp)                     # 8-byte Folded Spill
	lbu	s1, 10(ra)
	lui	t3, 1
	add	t3, t3, sp
	sd	s1, -1648(t3)                   # 8-byte Folded Spill
	vle8.v	v13, (a0)
	csrr	a0, vlenb
	li	t3, 45
	mul	a0, a0, t3
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v13, (a0)                       # Unknown-size Folded Spill
	lbu	t6, 39(ra)
	sd	t6, 696(sp)                     # 8-byte Folded Spill
	lbu	s10, 40(ra)
	sd	s10, 720(sp)                    # 8-byte Folded Spill
	lbu	s9, 41(ra)
	sd	s9, 752(sp)                     # 8-byte Folded Spill
	lbu	s7, 42(ra)
	sd	s7, 768(sp)                     # 8-byte Folded Spill
	vle8.v	v15, (a1)
	addi	a1, s11, 368
	vle8.v	v26, (a2)
	csrr	a0, vlenb
	li	a2, 44
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v26, (a0)                       # Unknown-size Folded Spill
	addi	a2, s11, 384
	vle8.v	v27, (a4)
	addi	a4, s11, 400
	vle8.v	v8, (t2)
	addi	a0, s11, 416
	vle8.v	v12, (t0)
	addi	s1, s11, 432
	vle8.v	v29, (a6)
	addi	t3, s11, 448
	vle8.v	v10, (a5)
	addi	a6, s11, 512
	vle8.v	v23, (s0)
	addi	t2, s11, 528
	vle8.v	v31, (t1)
	addi	t0, s11, 544
	vle8.v	v7, (a7)
	vle8.v	v25, (a1)
	vle8.v	v24, (a2)
	lbu	a5, 20(ra)
	sd	a5, 568(sp)                     # 8-byte Folded Spill
	lbu	a2, 21(ra)
	sd	a2, 576(sp)                     # 8-byte Folded Spill
	lbu	a1, 22(ra)
	sd	a1, 776(sp)                     # 8-byte Folded Spill
	vle8.v	v22, (a4)
	addi	a7, s11, 560
	vle8.v	v18, (a0)
	addi	s0, s11, 576
	vle8.v	v19, (s1)
	addi	t1, s11, 592
	vle8.v	v9, (t3)
	addi	a4, s11, 608
	vle8.v	v21, (a6)
	vle8.v	v20, (t2)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v13, 15
	vand.vi	v14, v8, 15
	vwmacc.vx	v11, a3, v14
	vand.vi	v14, v15, 3
	vmv1r.v	v2, v15
	csrr	a0, vlenb
	li	a3, 36
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v15, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v8, v8, 4
	vwmacc.vx	v16, s2, v8
	vand.vi	v8, v21, 15
	vwmacc.vx	v17, a5, v8
	vand.vi	v15, v26, 15
	vsll.vi	v8, v14, 4
	vor.vv	v8, v8, v13
	vand.vi	v13, v27, 3
	vmv1r.v	v14, v27
	csrr	a0, vlenb
	li	a3, 37
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v27, (a0)                       # Unknown-size Folded Spill
	vsll.vi	v13, v13, 4
	vor.vv	v26, v13, v15
	vand.vi	v13, v12, 15
	vwmacc.vx	v11, s3, v13
	vle8.v	v27, (t0)
	vsrl.vi	v12, v12, 4
	vwmacc.vx	v16, s5, v12
	vand.vi	v12, v20, 15
	vwmacc.vx	v17, a2, v12
	vand.vi	v12, v29, 15
	vwmacc.vx	v11, s4, v12
	vle8.v	v28, (a7)
	lbu	a5, 23(ra)
	sd	a5, 688(sp)                     # 8-byte Folded Spill
	lbu	s1, 24(ra)
	sd	s1, 712(sp)                     # 8-byte Folded Spill
	lbu	a3, 25(ra)
	sd	a3, 744(sp)                     # 8-byte Folded Spill
	lbu	a2, 26(ra)
	sd	a2, 760(sp)                     # 8-byte Folded Spill
	vsrl.vi	v12, v29, 4
	vwmacc.vx	v16, t4, v12
	vand.vi	v12, v27, 15
	vwmacc.vx	v17, a1, v12
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, t5, v12
	vle8.v	v30, (s0)
	addi	a0, s11, 624
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v16, t6, v10
	vand.vi	v10, v28, 15
	vwmacc.vx	v17, a5, v10
	vand.vi	v10, v23, 15
	vwmacc.vx	v11, s6, v10
	vle8.v	v5, (t1)
	vsrl.vi	v10, v23, 4
	vwmacc.vx	v16, s10, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v17, s1, v10
	vand.vi	v10, v31, 15
	vwmacc.vx	v11, s8, v10
	vle8.v	v29, (a4)
	vsrl.vi	v10, v31, 4
	vwmacc.vx	v16, s9, v10
	vand.vi	v10, v5, 15
	vwmacc.vx	v17, a3, v10
	vand.vi	v10, v7, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1648(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vle8.v	v23, (a0)
	vsrl.vi	v10, v7, 4
	vwmacc.vx	v16, s7, v10
	vand.vi	v10, v29, 15
	vwmacc.vx	v17, a2, v10
	vand.vi	v10, v25, 15
	lbu	a0, 11(ra)
	sd	a0, 736(sp)                     # 8-byte Folded Spill
	lbu	a2, 12(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -2040(a1)                   # 8-byte Folded Spill
	lbu	a7, 13(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a7, -1960(a1)                   # 8-byte Folded Spill
	lbu	a6, 14(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a6, -1896(a1)                   # 8-byte Folded Spill
	vwmacc.vx	v11, a0, v10
	addi	a0, s11, 640
	vle8.v	v31, (a0)
	vsrl.vi	v10, v25, 4
	lbu	a0, 43(ra)
	sd	a0, 680(sp)                     # 8-byte Folded Spill
	lbu	s1, 44(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	s1, -1984(a1)                   # 8-byte Folded Spill
	lbu	a4, 45(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a4, -1920(a1)                   # 8-byte Folded Spill
	lbu	a5, 46(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a5, -1912(a1)                   # 8-byte Folded Spill
	vwmacc.vx	v16, a0, v10
	vand.vi	v10, v23, 15
	lbu	a0, 27(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -2048(a1)                   # 8-byte Folded Spill
	lbu	s0, 28(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	s0, -1968(a1)                   # 8-byte Folded Spill
	lbu	a1, 29(ra)
	lui	a3, 1
	add	a3, a3, sp
	sd	a1, -1904(a3)                   # 8-byte Folded Spill
	lbu	a3, 30(ra)
	lui	t0, 1
	add	t0, t0, sp
	sd	a3, -1888(t0)                   # 8-byte Folded Spill
	vwmacc.vx	v17, a0, v10
	vand.vi	v10, v24, 15
	vwmacc.vx	v11, a2, v10
	addi	a0, s11, 656
	vle8.v	v4, (a0)
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v16, s1, v10
	vand.vi	v10, v31, 15
	vwmacc.vx	v17, s0, v10
	vand.vi	v10, v22, 15
	vwmacc.vx	v11, a7, v10
	addi	a0, s11, 672
	vle8.v	v7, (a0)
	vsrl.vi	v10, v22, 4
	vwmacc.vx	v16, a4, v10
	vand.vi	v10, v4, 15
	vwmacc.vx	v17, a1, v10
	vand.vi	v10, v18, 15
	vwmacc.vx	v11, a6, v10
	addi	a0, s11, 688
	vle8.v	v6, (a0)
	vsrl.vi	v10, v18, 4
	vwmacc.vx	v16, a5, v10
	vand.vi	v10, v7, 15
	vwmacc.vx	v17, a3, v10
	vand.vi	v10, v19, 15
	lbu	a0, 15(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1936(a1)                   # 8-byte Folded Spill
	lbu	a5, 16(ra)
	sd	a5, 2008(sp)                    # 8-byte Folded Spill
	lbu	a7, 17(ra)
	sd	a7, 2032(sp)                    # 8-byte Folded Spill
	lbu	a6, 18(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a6, -1992(a1)                   # 8-byte Folded Spill
	vwmacc.vx	v11, a0, v10
	addi	a0, s11, 704
	vle8.v	v18, (a0)
	vsrl.vi	v10, v19, 4
	lbu	a0, 47(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1944(a1)                   # 8-byte Folded Spill
	lbu	s0, 48(ra)
	sd	s0, 2016(sp)                    # 8-byte Folded Spill
	lbu	s1, 49(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	s1, -2024(a1)                   # 8-byte Folded Spill
	lbu	a4, 50(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a4, -1952(a1)                   # 8-byte Folded Spill
	vwmacc.vx	v16, a0, v10
	vand.vi	v10, v6, 15
	lbu	a0, 31(ra)
	sd	a0, 2000(sp)                    # 8-byte Folded Spill
	lbu	a1, 32(ra)
	sd	a1, 2024(sp)                    # 8-byte Folded Spill
	lbu	a2, 33(ra)
	lui	a3, 1
	add	a3, a3, sp
	sd	a2, -2000(a3)                   # 8-byte Folded Spill
	lbu	a3, 34(ra)
	lui	t0, 1
	add	t0, t0, sp
	sd	a3, -1976(t0)                   # 8-byte Folded Spill
	vwmacc.vx	v17, a0, v10
	vand.vi	v10, v9, 15
	vwmacc.vx	v11, a5, v10
	addi	a0, s11, 464
	vle8.v	v10, (a0)
	vsrl.vi	v9, v9, 4
	vwmacc.vx	v16, s0, v9
	vand.vi	v9, v18, 15
	vwmacc.vx	v17, a1, v9
	vand.vi	v9, v10, 15
	vwmacc.vx	v11, a7, v9
	addi	a0, s11, 720
	vle8.v	v19, (a0)
	vsrl.vi	v9, v10, 4
	addi	a0, s11, 480
	vle8.v	v10, (a0)
	vwmacc.vx	v16, s1, v9
	vand.vi	v9, v19, 15
	vwmacc.vx	v17, a2, v9
	vand.vi	v9, v10, 15
	vwmacc.vx	v11, a6, v9
	vsrl.vi	v9, v10, 4
	vwmacc.vx	v16, a4, v9
	addi	a0, s11, 736
	vle8.v	v22, (a0)
	addi	a0, s11, 496
	vle8.v	v9, (a0)
	lbu	a0, 19(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1928(a1)                   # 8-byte Folded Spill
	vand.vi	v10, v22, 15
	vwmacc.vx	v17, a3, v10
	vand.vi	v10, v9, 15
	vwmacc.vx	v11, a0, v10
	vsrl.vi	v9, v9, 4
	lbu	a4, 35(ra)
	lui	a0, 1
	add	a0, a0, sp
	sd	a4, -2016(a0)                   # 8-byte Folded Spill
	addi	a0, s11, 752
	lbu	a5, 51(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a5, -2008(a1)                   # 8-byte Folded Spill
	lbu	a2, 52(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -2032(a1)                   # 8-byte Folded Spill
	vle8.v	v24, (a0)
	lbu	a3, 53(ra)
	sd	a3, 2040(sp)                    # 8-byte Folded Spill
	lbu	a1, 54(ra)
	lui	a0, 1
	add	a0, a0, sp
	sd	a1, -1656(a0)                   # 8-byte Folded Spill
	vwmacc.vx	v16, a5, v9
	vand.vi	v9, v24, 15
	vwmacc.vx	v17, a4, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v8
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v10, v11
	vzext.vf2	v25, v26
	vwmacc.vv	v8, v25, v16
	vwmacc.vv	v8, v10, v17
	vmv.v.i	v26, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v21, 4
	vwmacc.vx	v26, a2, v10
	addi	a0, s11, 96
	vle8.v	v12, (a0)
	csrr	a0, vlenb
	li	a2, 41
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v20, 4
	vwmacc.vx	v26, a3, v10
	addi	a0, s11, 224
	vle8.v	v3, (a0)
	vsrl.vi	v10, v27, 4
	vwmacc.vx	v26, a1, v10
	addi	a0, s11, 112
	vle8.v	v15, (a0)
	csrr	a0, vlenb
	li	a1, 40
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v15, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v28, 4
	lbu	a0, 55(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1664(a1)                   # 8-byte Folded Spill
	lbu	a3, 56(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a3, -1672(a1)                   # 8-byte Folded Spill
	lbu	a2, 57(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -1680(a1)                   # 8-byte Folded Spill
	lbu	a1, 58(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1688(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v26, a0, v10
	addi	a0, s11, 240
	vle8.v	v0, (a0)
	vsrl.vi	v10, v30, 4
	vwmacc.vx	v26, a3, v10
	addi	a0, s11, 768
	vle8.v	v27, (a0)
	vsrl.vi	v10, v5, 4
	vwmacc.vx	v26, a2, v10
	addi	a0, s11, 784
	vle8.v	v28, (a0)
	vsrl.vi	v10, v29, 4
	vwmacc.vx	v26, a1, v10
	addi	a0, s11, 800
	vle8.v	v30, (a0)
	vsrl.vi	v10, v23, 4
	lbu	a0, 59(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1696(a1)                   # 8-byte Folded Spill
	lbu	a3, 60(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a3, -1704(a1)                   # 8-byte Folded Spill
	lbu	a2, 61(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -1712(a1)                   # 8-byte Folded Spill
	lbu	a1, 62(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1720(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v26, a0, v10
	addi	a0, s11, 816
	vle8.v	v29, (a0)
	vsrl.vi	v10, v31, 4
	vwmacc.vx	v26, a3, v10
	addi	a0, s11, 832
	vle8.v	v11, (a0)
	vsrl.vi	v10, v4, 4
	vwmacc.vx	v26, a2, v10
	addi	a0, s11, 848
	vle8.v	v16, (a0)
	vsrl.vi	v10, v7, 4
	vwmacc.vx	v26, a1, v10
	addi	a0, s11, 864
	vle8.v	v17, (a0)
	vsrl.vi	v10, v6, 4
	lbu	a0, 63(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1728(a1)                   # 8-byte Folded Spill
	lbu	a3, 64(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a3, -1736(a1)                   # 8-byte Folded Spill
	lbu	a2, 65(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -1744(a1)                   # 8-byte Folded Spill
	lbu	a1, 66(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1752(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v26, a0, v10
	addi	a0, s11, 1024
	vle8.v	v20, (a0)
	vsrl.vi	v10, v18, 4
	vwmacc.vx	v26, a3, v10
	addi	a0, s11, 1040
	vle8.v	v21, (a0)
	vsrl.vi	v10, v19, 4
	vwmacc.vx	v26, a2, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v19, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v22, 4
	vwmacc.vx	v26, a1, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v23, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v24, 4
	lbu	a3, 67(ra)
	sd	a3, 1136(sp)                    # 8-byte Folded Spill
	lbu	a2, 68(ra)
	lui	a0, 1
	add	a0, a0, sp
	sd	a2, -1768(a0)                   # 8-byte Folded Spill
	lbu	a0, 69(ra)
	sd	a0, 1904(sp)                    # 8-byte Folded Spill
	lbu	a1, 70(ra)
	sd	a1, 1984(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v26, a3, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v22, 0
	vwmacc.vv	v8, v25, v26
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v12, 15
	vand.vi	v12, v27, 15
	vwmacc.vx	v19, a2, v12
	vand.vi	v12, v3, 3
	csrr	a2, vlenb
	slli	a2, a2, 5
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vs1r.v	v3, (a2)                        # Unknown-size Folded Spill
	lbu	a4, 100(ra)
	lui	a2, 1
	add	a2, a2, sp
	sd	a4, -1760(a2)                   # 8-byte Folded Spill
	vsrl.vi	v13, v27, 4
	lbu	a3, 101(ra)
	sd	a3, 1960(sp)                    # 8-byte Folded Spill
	lbu	a2, 102(ra)
	sd	a2, 1968(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a4, v13
	lbu	s1, 84(ra)
	sd	s1, 1872(sp)                    # 8-byte Folded Spill
	vand.vi	v13, v20, 15
	lbu	a4, 85(ra)
	sd	a4, 1976(sp)                    # 8-byte Folded Spill
	lbu	a5, 86(ra)
	sd	a5, 1992(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v22, s1, v13
	vand.vi	v13, v15, 15
	vsll.vi	v12, v12, 4
	vor.vv	v26, v12, v10
	vand.vi	v10, v0, 3
	csrr	a6, vlenb
	slli	a7, a6, 5
	add	a6, a6, a7
	add	a6, a6, sp
	addi	a6, a6, 2047
	addi	a6, a6, 417
	vs1r.v	v0, (a6)                        # Unknown-size Folded Spill
	vsll.vi	v10, v10, 4
	vor.vv	v27, v10, v13
	vand.vi	v10, v28, 15
	vwmacc.vx	v19, a0, v10
	addi	a0, s11, 1056
	vle8.v	v24, (a0)
	vsrl.vi	v10, v28, 4
	vwmacc.vx	v23, a3, v10
	vand.vi	v10, v21, 15
	vwmacc.vx	v22, a4, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v19, a1, v10
	addi	a0, s11, 1072
	vle8.v	v28, (a0)
	vsrl.vi	v10, v30, 4
	vwmacc.vx	v23, a2, v10
	vand.vi	v10, v24, 15
	vwmacc.vx	v22, a5, v10
	vand.vi	v10, v29, 15
	lbu	a0, 71(ra)
	sd	a0, 1952(sp)                    # 8-byte Folded Spill
	lbu	a5, 72(ra)
	sd	a5, 1792(sp)                    # 8-byte Folded Spill
	lbu	a7, 73(ra)
	sd	a7, 1856(sp)                    # 8-byte Folded Spill
	lbu	a6, 74(ra)
	sd	a6, 1896(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, a0, v10
	addi	a0, s11, 1088
	vle8.v	v30, (a0)
	vsrl.vi	v10, v29, 4
	lbu	a0, 103(ra)
	sd	a0, 1920(sp)                    # 8-byte Folded Spill
	lbu	s0, 104(ra)
	sd	s0, 1824(sp)                    # 8-byte Folded Spill
	lbu	s1, 105(ra)
	sd	s1, 1880(sp)                    # 8-byte Folded Spill
	lbu	a4, 106(ra)
	sd	a4, 1936(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a0, v10
	vand.vi	v10, v28, 15
	lbu	a0, 87(ra)
	sd	a0, 1784(sp)                    # 8-byte Folded Spill
	lbu	a1, 88(ra)
	sd	a1, 1848(sp)                    # 8-byte Folded Spill
	lbu	a2, 89(ra)
	sd	a2, 1888(sp)                    # 8-byte Folded Spill
	lbu	a3, 90(ra)
	sd	a3, 1928(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v22, a0, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v19, a5, v10
	addi	a0, s11, 1104
	vle8.v	v4, (a0)
	vsrl.vi	v10, v11, 4
	vwmacc.vx	v23, s0, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v16, 15
	vwmacc.vx	v19, a7, v10
	addi	a0, s11, 1120
	vle8.v	v5, (a0)
	vsrl.vi	v10, v16, 4
	vwmacc.vx	v23, s1, v10
	vand.vi	v10, v4, 15
	vwmacc.vx	v22, a2, v10
	vand.vi	v10, v17, 15
	vwmacc.vx	v19, a6, v10
	addi	a0, s11, 880
	vle8.v	v11, (a0)
	vsrl.vi	v10, v17, 4
	vwmacc.vx	v23, a4, v10
	vand.vi	v10, v5, 15
	vwmacc.vx	v22, a3, v10
	vand.vi	v10, v11, 15
	lbu	a0, 75(ra)
	sd	a0, 1912(sp)                    # 8-byte Folded Spill
	lbu	s1, 76(ra)
	sd	s1, 1720(sp)                    # 8-byte Folded Spill
	lbu	a5, 77(ra)
	sd	a5, 1760(sp)                    # 8-byte Folded Spill
	lbu	a6, 78(ra)
	sd	a6, 1776(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, a0, v10
	addi	a0, s11, 896
	vle8.v	v10, (a0)
	vsrl.vi	v11, v11, 4
	lbu	a0, 107(ra)
	sd	a0, 1944(sp)                    # 8-byte Folded Spill
	lbu	s0, 108(ra)
	sd	s0, 1744(sp)                    # 8-byte Folded Spill
	lbu	a4, 109(ra)
	sd	a4, 1808(sp)                    # 8-byte Folded Spill
	lbu	a7, 110(ra)
	sd	a7, 1840(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a0, v11
	addi	a0, s11, 1136
	vle8.v	v25, (a0)
	lbu	a0, 91(ra)
	sd	a0, 1712(sp)                    # 8-byte Folded Spill
	lbu	a1, 92(ra)
	sd	a1, 1752(sp)                    # 8-byte Folded Spill
	lbu	a2, 93(ra)
	sd	a2, 1800(sp)                    # 8-byte Folded Spill
	lbu	a3, 94(ra)
	sd	a3, 1832(sp)                    # 8-byte Folded Spill
	vand.vi	v11, v25, 15
	vwmacc.vx	v22, a0, v11
	vand.vi	v11, v10, 15
	vwmacc.vx	v19, s1, v11
	vsrl.vi	v10, v10, 4
	addi	a0, s11, 1152
	vle8.v	v7, (a0)
	addi	a0, s11, 912
	vle8.v	v11, (a0)
	vwmacc.vx	v23, s0, v10
	vand.vi	v10, v7, 15
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v19, a5, v10
	vsrl.vi	v10, v11, 4
	addi	a0, s11, 1168
	vle8.v	v31, (a0)
	vwmacc.vx	v23, a4, v10
	addi	a0, s11, 928
	vle8.v	v10, (a0)
	vand.vi	v11, v31, 15
	vwmacc.vx	v22, a2, v11
	addi	a0, s11, 1184
	vle8.v	v29, (a0)
	vand.vi	v11, v10, 15
	vwmacc.vx	v19, a6, v11
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v23, a7, v10
	vand.vi	v10, v29, 15
	vwmacc.vx	v22, a3, v10
	addi	a0, s11, 944
	lbu	a1, 79(ra)
	sd	a1, 1816(sp)                    # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s0, 80(ra)
	sd	s0, 1592(sp)                    # 8-byte Folded Spill
	lbu	t0, 81(ra)
	sd	t0, 1624(sp)                    # 8-byte Folded Spill
	lbu	a6, 82(ra)
	sd	a6, 1704(sp)                    # 8-byte Folded Spill
	vand.vi	v11, v10, 15
	vwmacc.vx	v19, a1, v11
	vsrl.vi	v10, v10, 4
	lbu	a0, 111(ra)
	sd	a0, 1864(sp)                    # 8-byte Folded Spill
	lbu	a2, 112(ra)
	sd	a2, 1600(sp)                    # 8-byte Folded Spill
	lbu	a5, 113(ra)
	sd	a5, 1648(sp)                    # 8-byte Folded Spill
	lbu	a7, 114(ra)
	sd	a7, 1736(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a0, v10
	addi	a0, s11, 960
	addi	a1, s11, 1200
	lbu	a3, 95(ra)
	sd	a3, 1584(sp)                    # 8-byte Folded Spill
	lbu	a4, 96(ra)
	sd	a4, 1616(sp)                    # 8-byte Folded Spill
	vle8.v	v6, (a1)
	lbu	a1, 97(ra)
	sd	a1, 1696(sp)                    # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s1, 98(ra)
	sd	s1, 1728(sp)                    # 8-byte Folded Spill
	vand.vi	v11, v6, 15
	vwmacc.vx	v22, a3, v11
	vand.vi	v11, v10, 15
	vwmacc.vx	v19, s0, v11
	vsrl.vi	v10, v10, 4
	addi	a0, s11, 1216
	vle8.v	v17, (a0)
	addi	a0, s11, 976
	vle8.v	v11, (a0)
	vwmacc.vx	v23, a2, v10
	vand.vi	v10, v17, 15
	vwmacc.vx	v22, a4, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v19, t0, v10
	vsrl.vi	v10, v11, 4
	addi	a0, s11, 1232
	vle8.v	v11, (a0)
	addi	a0, s11, 992
	vle8.v	v12, (a0)
	vwmacc.vx	v23, a5, v10
	vand.vi	v10, v11, 15
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	vwmacc.vx	v19, a6, v10
	vsrl.vi	v10, v12, 4
	vwmacc.vx	v23, a7, v10
	addi	a0, s11, 1248
	vle8.v	v16, (a0)
	addi	a0, s11, 1008
	vle8.v	v10, (a0)
	lbu	a0, 83(ra)
	sd	a0, 1768(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v16, 15
	vwmacc.vx	v22, s1, v12
	vand.vi	v12, v10, 15
	vwmacc.vx	v19, a0, v12
	vsrl.vi	v10, v10, 4
	lbu	a3, 99(ra)
	sd	a3, 1656(sp)                    # 8-byte Folded Spill
	addi	a0, s11, 1264
	lbu	a4, 115(ra)
	sd	a4, 1664(sp)                    # 8-byte Folded Spill
	lbu	a1, 116(ra)
	sd	a1, 1680(sp)                    # 8-byte Folded Spill
	vle8.v	v18, (a0)
	lbu	a2, 117(ra)
	sd	a2, 1672(sp)                    # 8-byte Folded Spill
	lbu	a0, 118(ra)
	sd	a0, 1688(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v23, a4, v10
	vand.vi	v10, v18, 15
	vwmacc.vx	v22, a3, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v26
	vwmacc.vv	v8, v10, v19
	vzext.vf2	v19, v27
	vwmacc.vv	v8, v19, v23
	vwmacc.vv	v8, v10, v22
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v20, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v20, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v20, a1, v10
	vsrl.vi	v10, v21, 4
	vwmacc.vx	v20, a2, v10
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v20, a0, v10
	vsrl.vi	v10, v28, 4
	lbu	a3, 119(ra)
	sd	a3, 1632(sp)                    # 8-byte Folded Spill
	lbu	a0, 120(ra)
	sd	a0, 1640(sp)                    # 8-byte Folded Spill
	lbu	a2, 121(ra)
	sd	a2, 1608(sp)                    # 8-byte Folded Spill
	lbu	a1, 122(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1776(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v20, a3, v10
	vsrl.vi	v10, v30, 4
	vwmacc.vx	v20, a0, v10
	addi	a0, s11, 128
	vle8.v	v12, (a0)
	csrr	a0, vlenb
	li	a3, 38
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v4, 4
	vwmacc.vx	v20, a2, v10
	addi	a0, s11, 144
	vle8.v	v13, (a0)
	csrr	a0, vlenb
	li	a2, 39
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v13, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v5, 4
	vwmacc.vx	v20, a1, v10
	addi	a0, s11, 1280
	vle8.v	v27, (a0)
	vsrl.vi	v10, v25, 4
	lbu	a0, 123(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1784(a1)                   # 8-byte Folded Spill
	lbu	a3, 124(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a3, -1792(a1)                   # 8-byte Folded Spill
	lbu	a2, 125(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -1800(a1)                   # 8-byte Folded Spill
	lbu	a1, 126(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1808(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v20, a0, v10
	addi	a0, s11, 1296
	vle8.v	v26, (a0)
	vsrl.vi	v10, v7, 4
	vwmacc.vx	v20, a3, v10
	addi	a0, s11, 1312
	vle8.v	v24, (a0)
	vsrl.vi	v10, v31, 4
	vwmacc.vx	v20, a2, v10
	addi	a0, s11, 1328
	vle8.v	v23, (a0)
	vsrl.vi	v10, v29, 4
	vwmacc.vx	v20, a1, v10
	addi	a0, s11, 1344
	vle8.v	v22, (a0)
	vsrl.vi	v10, v6, 4
	lbu	a0, 127(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a0, -1816(a1)                   # 8-byte Folded Spill
	lbu	a3, 128(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a3, -1824(a1)                   # 8-byte Folded Spill
	lbu	a2, 129(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -1832(a1)                   # 8-byte Folded Spill
	lbu	a1, 130(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1840(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v20, a0, v10
	addi	a0, s11, 1536
	vle8.v	v21, (a0)
	vsrl.vi	v10, v17, 4
	vwmacc.vx	v20, a3, v10
	addi	a0, s11, 1552
	vle8.v	v17, (a0)
	vsrl.vi	v10, v11, 4
	vwmacc.vx	v20, a2, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v11, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v16, 4
	vwmacc.vx	v20, a1, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v18, 4
	lbu	a2, 131(ra)
	lui	a0, 1
	add	a0, a0, sp
	sd	a2, -1848(a0)                   # 8-byte Folded Spill
	lbu	a3, 132(ra)
	sd	a3, 824(sp)                     # 8-byte Folded Spill
	lbu	a0, 133(ra)
	sd	a0, 1488(sp)                    # 8-byte Folded Spill
	lbu	a1, 134(ra)
	sd	a1, 1560(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v20, a2, v10
	vand.vi	v10, v12, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v19, v20
	vsetvli	zero, zero, e8, mf2, ta, ma
	li	a7, 48
	vand.vx	v12, v2, a7
	vand.vi	v13, v13, 15
	vor.vv	v19, v12, v10
	vand.vx	v10, v14, a7
	vor.vv	v20, v10, v13
	vand.vi	v10, v27, 15
	vwmacc.vx	v11, a3, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	lbu	a2, 164(ra)
	sd	a2, 1576(sp)                    # 8-byte Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v27, 4
	lbu	a4, 165(ra)
	sd	a4, 1536(sp)                    # 8-byte Folded Spill
	lbu	a3, 166(ra)
	sd	a3, 1544(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v16, a2, v10
	lbu	s1, 148(ra)
	sd	s1, 1480(sp)                    # 8-byte Folded Spill
	vand.vi	v10, v21, 15
	lbu	a5, 149(ra)
	sd	a5, 1552(sp)                    # 8-byte Folded Spill
	lbu	a2, 150(ra)
	sd	a2, 1568(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v18, s1, v10
	vand.vi	v10, v26, 15
	vwmacc.vx	v11, a0, v10
	addi	a0, s11, 1568
	vle8.v	v25, (a0)
	vsrl.vi	v10, v26, 4
	vwmacc.vx	v16, a4, v10
	vand.vi	v10, v17, 15
	vwmacc.vx	v18, a5, v10
	vand.vi	v10, v24, 15
	vwmacc.vx	v11, a1, v10
	addi	a0, s11, 1584
	vle8.v	v26, (a0)
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v16, a3, v10
	vand.vi	v10, v25, 15
	vwmacc.vx	v18, a2, v10
	vand.vi	v10, v23, 15
	lbu	a0, 135(ra)
	sd	a0, 1528(sp)                    # 8-byte Folded Spill
	lbu	s1, 136(ra)
	sd	s1, 1384(sp)                    # 8-byte Folded Spill
	lbu	t0, 137(ra)
	sd	t0, 1440(sp)                    # 8-byte Folded Spill
	lbu	a6, 138(ra)
	sd	a6, 1448(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v11, a0, v10
	addi	a0, s11, 1600
	vle8.v	v28, (a0)
	vsrl.vi	v10, v23, 4
	lbu	a0, 167(ra)
	sd	a0, 1512(sp)                    # 8-byte Folded Spill
	lbu	s0, 168(ra)
	sd	s0, 1424(sp)                    # 8-byte Folded Spill
	lbu	a5, 169(ra)
	sd	a5, 1464(sp)                    # 8-byte Folded Spill
	lbu	a4, 170(ra)
	sd	a4, 1504(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v16, a0, v10
	vand.vi	v10, v26, 15
	lbu	a0, 151(ra)
	sd	a0, 1376(sp)                    # 8-byte Folded Spill
	lbu	a1, 152(ra)
	sd	a1, 1432(sp)                    # 8-byte Folded Spill
	lbu	a2, 153(ra)
	sd	a2, 1456(sp)                    # 8-byte Folded Spill
	lbu	a3, 154(ra)
	sd	a3, 1496(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v18, a0, v10
	vand.vi	v10, v22, 15
	vwmacc.vx	v11, s1, v10
	addi	a0, s11, 1360
	vle8.v	v10, (a0)
	vsrl.vi	v12, v22, 4
	vwmacc.vx	v16, s0, v12
	vand.vi	v12, v28, 15
	vwmacc.vx	v18, a1, v12
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, t0, v12
	addi	a0, s11, 1616
	vle8.v	v29, (a0)
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v16, a5, v10
	addi	a0, s11, 1376
	vle8.v	v10, (a0)
	vand.vi	v12, v29, 15
	vwmacc.vx	v18, a2, v12
	addi	a0, s11, 1632
	vle8.v	v30, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, a6, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v16, a4, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v18, a3, v10
	addi	a0, s11, 1392
	lbu	a1, 139(ra)
	sd	a1, 1472(sp)                    # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s0, 140(ra)
	sd	s0, 1296(sp)                    # 8-byte Folded Spill
	lbu	t1, 141(ra)
	sd	t1, 1336(sp)                    # 8-byte Folded Spill
	lbu	t0, 142(ra)
	sd	t0, 1352(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, a1, v12
	vsrl.vi	v10, v10, 4
	lbu	a0, 171(ra)
	sd	a0, 1520(sp)                    # 8-byte Folded Spill
	lbu	a2, 172(ra)
	sd	a2, 1320(sp)                    # 8-byte Folded Spill
	lbu	a5, 173(ra)
	sd	a5, 1368(sp)                    # 8-byte Folded Spill
	lbu	a6, 174(ra)
	sd	a6, 1408(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v16, a0, v10
	addi	a0, s11, 1408
	addi	a1, s11, 1648
	lbu	a3, 155(ra)
	sd	a3, 1288(sp)                    # 8-byte Folded Spill
	lbu	a4, 156(ra)
	sd	a4, 1328(sp)                    # 8-byte Folded Spill
	vle8.v	v31, (a1)
	lbu	a1, 157(ra)
	sd	a1, 1360(sp)                    # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s1, 158(ra)
	sd	s1, 1400(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v31, 15
	vwmacc.vx	v18, a3, v12
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, s0, v12
	vsrl.vi	v10, v10, 4
	addi	a0, s11, 1664
	vle8.v	v6, (a0)
	addi	a0, s11, 1424
	vle8.v	v12, (a0)
	vwmacc.vx	v16, a2, v10
	vand.vi	v10, v6, 15
	vwmacc.vx	v18, a4, v10
	vand.vi	v10, v12, 15
	vwmacc.vx	v11, t1, v10
	vsrl.vi	v10, v12, 4
	addi	a0, s11, 1680
	vle8.v	v5, (a0)
	vwmacc.vx	v16, a5, v10
	addi	a0, s11, 1440
	vle8.v	v10, (a0)
	vand.vi	v12, v5, 15
	vwmacc.vx	v18, a1, v12
	addi	a0, s11, 1696
	vle8.v	v7, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, t0, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v16, a6, v10
	vand.vi	v10, v7, 15
	vwmacc.vx	v18, s1, v10
	addi	a0, s11, 1456
	lbu	a1, 143(ra)
	sd	a1, 1392(sp)                    # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s0, 144(ra)
	sd	s0, 1128(sp)                    # 8-byte Folded Spill
	lbu	t1, 145(ra)
	sd	t1, 1176(sp)                    # 8-byte Folded Spill
	lbu	a6, 146(ra)
	sd	a6, 1280(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, a1, v12
	vsrl.vi	v10, v10, 4
	lbu	a0, 175(ra)
	sd	a0, 1416(sp)                    # 8-byte Folded Spill
	lbu	a2, 176(ra)
	sd	a2, 1160(sp)                    # 8-byte Folded Spill
	lbu	a5, 177(ra)
	sd	a5, 1224(sp)                    # 8-byte Folded Spill
	lbu	t0, 178(ra)
	sd	t0, 1312(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v16, a0, v10
	addi	a0, s11, 1472
	addi	a1, s11, 1712
	lbu	a3, 159(ra)
	sd	a3, 1120(sp)                    # 8-byte Folded Spill
	lbu	a4, 160(ra)
	sd	a4, 1168(sp)                    # 8-byte Folded Spill
	vle8.v	v4, (a1)
	lbu	a1, 161(ra)
	sd	a1, 1272(sp)                    # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s1, 162(ra)
	sd	s1, 1304(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v4, 15
	vwmacc.vx	v18, a3, v12
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, s0, v12
	vsrl.vi	v10, v10, 4
	addi	a0, s11, 1728
	vle8.v	v24, (a0)
	addi	a0, s11, 1488
	vle8.v	v12, (a0)
	vwmacc.vx	v16, a2, v10
	vand.vi	v10, v24, 15
	vwmacc.vx	v18, a4, v10
	vand.vi	v10, v12, 15
	vwmacc.vx	v11, t1, v10
	vsrl.vi	v10, v12, 4
	addi	a0, s11, 1744
	vle8.v	v22, (a0)
	addi	a0, s11, 1504
	vle8.v	v12, (a0)
	vwmacc.vx	v16, a5, v10
	vand.vi	v10, v22, 15
	vwmacc.vx	v18, a1, v10
	vand.vi	v10, v12, 15
	vwmacc.vx	v11, a6, v10
	vsrl.vi	v10, v12, 4
	vwmacc.vx	v16, t0, v10
	addi	a0, s11, 1760
	vle8.v	v23, (a0)
	addi	a0, s11, 1520
	vle8.v	v10, (a0)
	lbu	a0, 147(ra)
	sd	a0, 1344(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v23, 15
	vwmacc.vx	v18, s1, v12
	vand.vi	v12, v10, 15
	vwmacc.vx	v11, a0, v12
	vsrl.vi	v10, v10, 4
	lbu	a3, 163(ra)
	sd	a3, 1264(sp)                    # 8-byte Folded Spill
	addi	a0, s11, 1776
	lbu	a4, 179(ra)
	sd	a4, 1256(sp)                    # 8-byte Folded Spill
	lbu	a1, 180(ra)
	sd	a1, 1240(sp)                    # 8-byte Folded Spill
	vle8.v	v27, (a0)
	lbu	a2, 181(ra)
	sd	a2, 1232(sp)                    # 8-byte Folded Spill
	lbu	a0, 182(ra)
	sd	a0, 1248(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v16, a4, v10
	vand.vi	v10, v27, 15
	vwmacc.vx	v18, a3, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v19
	vwmacc.vv	v8, v10, v11
	vzext.vf2	v19, v20
	vwmacc.vv	v8, v19, v16
	vwmacc.vv	v8, v10, v18
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v21, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v20, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v20, a1, v10
	vsrl.vi	v10, v17, 4
	vwmacc.vx	v20, a2, v10
	vsrl.vi	v10, v25, 4
	vwmacc.vx	v20, a0, v10
	vsrl.vi	v10, v26, 4
	lbu	a3, 183(ra)
	sd	a3, 1192(sp)                    # 8-byte Folded Spill
	lbu	a2, 184(ra)
	sd	a2, 1200(sp)                    # 8-byte Folded Spill
	lbu	a1, 185(ra)
	sd	a1, 1208(sp)                    # 8-byte Folded Spill
	lbu	a0, 186(ra)
	sd	a0, 1216(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v20, a3, v10
	vsrl.vi	v10, v28, 4
	vwmacc.vx	v20, a2, v10
	vsrl.vi	v10, v29, 4
	vwmacc.vx	v20, a1, v10
	vsrl.vi	v10, v30, 4
	vwmacc.vx	v20, a0, v10
	addi	a0, s11, 160
	vle8.v	v12, (a0)
	csrr	a0, vlenb
	li	a1, 34
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v31, 4
	lbu	a0, 187(ra)
	sd	a0, 1184(sp)                    # 8-byte Folded Spill
	lbu	a3, 188(ra)
	sd	a3, 1152(sp)                    # 8-byte Folded Spill
	lbu	a2, 189(ra)
	sd	a2, 1144(sp)                    # 8-byte Folded Spill
	lbu	a1, 190(ra)
	sd	a1, 1112(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v20, a0, v10
	addi	a0, s11, 176
	vle8.v	v13, (a0)
	csrr	a0, vlenb
	li	a4, 35
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v13, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v6, 4
	vwmacc.vx	v20, a3, v10
	addi	a0, s11, 1792
	vle8.v	v26, (a0)
	vsrl.vi	v10, v5, 4
	vwmacc.vx	v20, a2, v10
	addi	a0, s11, 1808
	vle8.v	v25, (a0)
	vsrl.vi	v10, v7, 4
	vwmacc.vx	v20, a1, v10
	addi	a0, s11, 1824
	vle8.v	v28, (a0)
	vsrl.vi	v10, v4, 4
	lbu	a0, 191(ra)
	sd	a0, 1096(sp)                    # 8-byte Folded Spill
	lbu	a3, 192(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a3, -1856(a1)                   # 8-byte Folded Spill
	lbu	a2, 193(ra)
	lui	a1, 1
	add	a1, a1, sp
	sd	a2, -1864(a1)                   # 8-byte Folded Spill
	lbu	a1, 194(ra)
	lui	a4, 1
	add	a4, a4, sp
	sd	a1, -1872(a4)                   # 8-byte Folded Spill
	vwmacc.vx	v20, a0, v10
	ld	a0, 528(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v11, (a0)
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v20, a3, v10
	ld	a0, 512(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v16, (a0)
	vsrl.vi	v10, v22, 4
	vwmacc.vx	v20, a2, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v17, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v23, 4
	vwmacc.vx	v20, a1, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v27, 4
	lbu	a3, 195(ra)
	lui	a0, 1
	add	a0, a0, sp
	sd	a3, -1880(a0)                   # 8-byte Folded Spill
	lbu	a2, 196(ra)
	sd	a2, 808(sp)                     # 8-byte Folded Spill
	lbu	a0, 197(ra)
	sd	a0, 1032(sp)                    # 8-byte Folded Spill
	lbu	a1, 198(ra)
	sd	a1, 1056(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v20, a3, v10
	vand.vi	v10, v12, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v19, v20
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vx	v12, v3, a7
	vand.vi	v13, v13, 15
	vor.vv	v20, v12, v10
	vand.vx	v10, v0, a7
	vor.vv	v21, v10, v13
	vand.vi	v10, v26, 15
	vwmacc.vx	v17, a2, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v19, 0
	lbu	a4, 228(ra)
	sd	a4, 1064(sp)                    # 8-byte Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v26, 4
	lbu	a3, 229(ra)
	sd	a3, 1040(sp)                    # 8-byte Folded Spill
	lbu	a2, 230(ra)
	sd	a2, 1088(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v18, a4, v10
	lbu	s1, 212(ra)
	sd	s1, 1024(sp)                    # 8-byte Folded Spill
	vand.vi	v10, v11, 15
	lbu	a4, 213(ra)
	sd	a4, 1048(sp)                    # 8-byte Folded Spill
	lbu	a5, 214(ra)
	sd	a5, 1080(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v19, s1, v10
	vand.vi	v10, v25, 15
	vwmacc.vx	v17, a0, v10
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v22, (a0)
	vsrl.vi	v10, v25, 4
	vwmacc.vx	v18, a3, v10
	vand.vi	v10, v16, 15
	vwmacc.vx	v19, a4, v10
	vand.vi	v10, v28, 15
	vwmacc.vx	v17, a1, v10
	addi	a0, s11, 1840
	vle8.v	v12, (a0)
	vsrl.vi	v10, v28, 4
	vwmacc.vx	v18, a2, v10
	vand.vi	v10, v22, 15
	vwmacc.vx	v19, a5, v10
	vand.vi	v10, v12, 15
	lbu	a0, 199(ra)
	sd	a0, 1072(sp)                    # 8-byte Folded Spill
	lbu	s1, 200(ra)
	sd	s1, 920(sp)                     # 8-byte Folded Spill
	lbu	a5, 201(ra)
	sd	a5, 944(sp)                     # 8-byte Folded Spill
	lbu	a6, 202(ra)
	sd	a6, 968(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v17, a0, v10
	addi	a0, s11, 1856
	vle8.v	v10, (a0)
	vsrl.vi	v12, v12, 4
	lbu	a0, 231(ra)
	sd	a0, 1104(sp)                    # 8-byte Folded Spill
	lbu	s0, 232(ra)
	sd	s0, 952(sp)                     # 8-byte Folded Spill
	lbu	a4, 233(ra)
	sd	a4, 976(sp)                     # 8-byte Folded Spill
	lbu	a7, 234(ra)
	sd	a7, 1008(sp)                    # 8-byte Folded Spill
	vwmacc.vx	v18, a0, v12
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v23, (a0)
	lbu	a0, 215(ra)
	sd	a0, 936(sp)                     # 8-byte Folded Spill
	lbu	a1, 216(ra)
	sd	a1, 960(sp)                     # 8-byte Folded Spill
	lbu	a2, 217(ra)
	sd	a2, 984(sp)                     # 8-byte Folded Spill
	lbu	a3, 218(ra)
	sd	a3, 1000(sp)                    # 8-byte Folded Spill
	vand.vi	v12, v23, 15
	vwmacc.vx	v19, a0, v12
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v24, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, s1, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v18, s0, v10
	vand.vi	v10, v24, 15
	vwmacc.vx	v19, a1, v10
	addi	a0, s11, 1872
	vle8.v	v10, (a0)
	lui	a1, 1
	addiw	a0, a1, -1968
	sd	a0, 392(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v25, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, a5, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v18, a4, v10
	vand.vi	v10, v25, 15
	vwmacc.vx	v19, a2, v10
	addi	a0, s11, 1888
	vle8.v	v10, (a0)
	addiw	a0, a1, -1952
	lui	t1, 1
	sd	a0, 400(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v26, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, a6, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v18, a7, v10
	vand.vi	v10, v26, 15
	vwmacc.vx	v19, a3, v10
	addi	a0, s11, 1904
	lbu	a1, 203(ra)
	sd	a1, 992(sp)                     # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s0, 204(ra)
	sd	s0, 832(sp)                     # 8-byte Folded Spill
	lbu	a5, 205(ra)
	sd	a5, 848(sp)                     # 8-byte Folded Spill
	lbu	a6, 206(ra)
	sd	a6, 872(sp)                     # 8-byte Folded Spill
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, a1, v12
	vsrl.vi	v10, v10, 4
	lbu	a0, 235(ra)
	sd	a0, 1016(sp)                    # 8-byte Folded Spill
	lbu	a2, 236(ra)
	sd	a2, 856(sp)                     # 8-byte Folded Spill
	lbu	s1, 237(ra)
	sd	s1, 880(sp)                     # 8-byte Folded Spill
	lbu	a7, 238(ra)
	sd	a7, 912(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v18, a0, v10
	addiw	a1, t1, -1936
	sd	a1, 384(sp)                     # 8-byte Folded Spill
	addi	t0, s11, 1920
	add	a1, a1, s11
	lbu	a0, 219(ra)
	sd	a0, 840(sp)                     # 8-byte Folded Spill
	lbu	a3, 220(ra)
	sd	a3, 864(sp)                     # 8-byte Folded Spill
	vle8.v	v27, (a1)
	lbu	a1, 221(ra)
	sd	a1, 888(sp)                     # 8-byte Folded Spill
	lbu	a4, 222(ra)
	sd	a4, 904(sp)                     # 8-byte Folded Spill
	vle8.v	v10, (t0)
	vand.vi	v12, v27, 15
	vwmacc.vx	v19, a0, v12
	ld	a0, 448(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v28, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, s0, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v18, a2, v10
	vand.vi	v10, v28, 15
	vwmacc.vx	v19, a3, v10
	addi	a0, s11, 1936
	vle8.v	v10, (a0)
	addiw	a0, t1, -1904
	sd	a0, 368(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v29, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, a5, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v18, s1, v10
	vand.vi	v10, v29, 15
	vwmacc.vx	v19, a1, v10
	addi	a0, s11, 1952
	vle8.v	v10, (a0)
	addiw	a0, t1, -1888
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v30, (a0)
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, a6, v12
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v18, a7, v10
	vand.vi	v10, v30, 15
	vwmacc.vx	v19, a4, v10
	addi	a0, s11, 1968
	lbu	a1, 207(ra)
	sd	a1, 896(sp)                     # 8-byte Folded Spill
	vle8.v	v10, (a0)
	lbu	s0, 208(ra)
	sd	s0, 816(sp)                     # 8-byte Folded Spill
	lbu	t0, 209(ra)
	sd	t0, 624(sp)                     # 8-byte Folded Spill
	lbu	a6, 210(ra)
	sd	a6, 648(sp)                     # 8-byte Folded Spill
	vand.vi	v12, v10, 15
	vwmacc.vx	v17, a1, v12
	vsrl.vi	v10, v10, 4
	lbu	a0, 239(ra)
	sd	a0, 928(sp)                     # 8-byte Folded Spill
	lbu	a2, 240(ra)
	sd	a2, 608(sp)                     # 8-byte Folded Spill
	lbu	a5, 241(ra)
	sd	a5, 632(sp)                     # 8-byte Folded Spill
	lbu	a7, 242(ra)
	sd	a7, 664(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v18, a0, v10
	addiw	a1, t1, -1872
	sd	a1, 360(sp)                     # 8-byte Folded Spill
	addi	a0, s11, 1984
	add	a1, a1, s11
	lbu	a3, 223(ra)
	sd	a3, 600(sp)                     # 8-byte Folded Spill
	lbu	a4, 224(ra)
	sd	a4, 616(sp)                     # 8-byte Folded Spill
	vle8.v	v10, (a1)
	lbu	a1, 225(ra)
	sd	a1, 640(sp)                     # 8-byte Folded Spill
	vle8.v	v12, (a0)
	lbu	s1, 226(ra)
	sd	s1, 656(sp)                     # 8-byte Folded Spill
	vand.vi	v13, v10, 15
	vwmacc.vx	v19, a3, v13
	vand.vi	v13, v12, 15
	vwmacc.vx	v17, s0, v13
	vsrl.vi	v13, v12, 4
	addiw	a0, t1, -1856
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v12, (a0)
	addi	a0, s11, 2000
	vle8.v	v14, (a0)
	vwmacc.vx	v18, a2, v13
	vand.vi	v13, v12, 15
	vwmacc.vx	v19, a4, v13
	vand.vi	v13, v14, 15
	vwmacc.vx	v17, t0, v13
	vsrl.vi	v14, v14, 4
	addiw	a0, t1, -1840
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v13, (a0)
	addi	a0, s11, 2016
	vle8.v	v15, (a0)
	vwmacc.vx	v18, a5, v14
	vand.vi	v14, v13, 15
	vwmacc.vx	v19, a1, v14
	vand.vi	v14, v15, 15
	vwmacc.vx	v17, a6, v14
	vsrl.vi	v14, v15, 4
	vwmacc.vx	v18, a7, v14
	addiw	a0, t1, -1824
	sd	a0, 352(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v14, (a0)
	addi	a0, s11, 2032
	vle8.v	v15, (a0)
	lbu	a0, 211(ra)
	sd	a0, 672(sp)                     # 8-byte Folded Spill
	vand.vi	v31, v14, 15
	vwmacc.vx	v19, s1, v31
	vand.vi	v31, v15, 15
	vwmacc.vx	v17, a0, v31
	vsrl.vi	v31, v15, 4
	lbu	s5, 227(ra)
	addiw	a0, t1, -1808
	sd	a0, 344(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	lbu	s6, 243(ra)
	lbu	s4, 244(ra)
	vle8.v	v15, (a0)
	lbu	s2, 245(ra)
	lbu	s3, 246(ra)
	vwmacc.vx	v18, s6, v31
	sd	s6, 144(sp)                     # 8-byte Folded Spill
	vand.vi	v31, v15, 15
	vwmacc.vx	v19, s5, v31
	sd	s5, 152(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v31, v20
	vwmacc.vv	v8, v31, v17
	vzext.vf2	v17, v21
	vwmacc.vv	v8, v17, v18
	vwmacc.vv	v8, v31, v19
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v11, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v18, s4, v11
	sd	s4, 232(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v16, 4
	vwmacc.vx	v18, s2, v11
	sd	s2, 224(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v22, 4
	vwmacc.vx	v18, s3, v11
	sd	s3, 200(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v23, 4
	lbu	a3, 247(ra)
	sd	a3, 536(sp)                     # 8-byte Folded Spill
	lbu	t5, 248(ra)
	lbu	s1, 249(ra)
	lbu	t6, 250(ra)
	vwmacc.vx	v18, a3, v11
	vsrl.vi	v11, v24, 4
	vwmacc.vx	v18, t5, v11
	sd	t5, 216(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v25, 4
	vwmacc.vx	v18, s1, v11
	sd	s1, 184(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v26, 4
	vwmacc.vx	v18, t6, v11
	sd	t6, 192(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v27, 4
	lbu	s0, 251(ra)
	lbu	t4, 252(ra)
	lbu	t3, 253(ra)
	lbu	t2, 254(ra)
	vwmacc.vx	v18, s0, v11
	sd	s0, 176(sp)                     # 8-byte Folded Spill
	vsrl.vi	v11, v28, 4
	vwmacc.vx	v18, t4, v11
	sd	t4, 592(sp)                     # 8-byte Folded Spill
	addi	a0, s11, 72
	vle8.v	v19, (a0)
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v19, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v11, v29, 4
	vwmacc.vx	v18, t3, v11
	addi	a0, s11, 200
	vle8.v	v20, (a0)
	vsrl.vi	v11, v30, 4
	vwmacc.vx	v18, t2, v11
	addi	a0, s11, 88
	vle8.v	v21, (a0)
	csrr	a0, vlenb
	li	a1, 28
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v21, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v10, v10, 4
	lbu	t0, 255(ra)
	lbu	a7, 256(ra)
	lbu	a6, 257(ra)
	lbu	a5, 258(ra)
	vwmacc.vx	v18, t0, v10
	sd	t0, 168(sp)                     # 8-byte Folded Spill
	addi	a0, s11, 216
	vle8.v	v22, (a0)
	vsrl.vi	v10, v12, 4
	vwmacc.vx	v18, a7, v10
	addi	a0, s11, 264
	vle8.v	v12, (a0)
	vsrl.vi	v10, v13, 4
	vwmacc.vx	v18, a6, v10
	sd	a6, 160(sp)                     # 8-byte Folded Spill
	addi	a0, s11, 280
	vle8.v	v13, (a0)
	flw	fa5, 0(ra)
	vsrl.vi	v10, v14, 4
	vwmacc.vx	v18, a5, v10
	vle16.v	v14, (s11)
	vsrl.vi	v10, v15, 4
	lbu	a4, 259(ra)
	lh	a0, 260(ra)
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	lh	a0, 262(ra)
	sd	a0, 416(sp)                     # 8-byte Folded Spill
	lh	a0, 264(ra)
	sd	a0, 432(sp)                     # 8-byte Folded Spill
	vwmacc.vx	v18, a4, v10
	sd	a4, 208(sp)                     # 8-byte Folded Spill
	addi	a0, s11, 520
	vle8.v	v11, (a0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v17, v18
	vmv.v.i	v10, 0
	vfwcvt.f.f.v	v16, v14
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v14, v16, fa5
	vfcvt.f.x.v	v8, v8
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vl2r.v	v16, (a0)                       # Unknown-size Folded Reload
	vfmadd.vv	v8, v14, v16
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vmv.v.i	v15, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v19, 15
	vand.vi	v9, v12, 15
	ld	a2, 544(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a2, v9
	vand.vi	v9, v20, 3
	vmv1r.v	v6, v20
	csrr	a0, vlenb
	li	a1, 23
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v20, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v12, v12, 4
	ld	a3, 560(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a3, v12
	vand.vi	v12, v11, 15
	ld	s10, 568(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, s10, v12
	vand.vi	v12, v21, 15
	vsll.vi	v9, v9, 4
	vor.vv	v8, v9, v8
	vand.vi	v9, v22, 3
	vmv1r.v	v3, v22
	csrr	a0, vlenb
	li	a1, 22
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v22, (a0)                       # Unknown-size Folded Spill
	vsll.vi	v9, v9, 4
	vor.vv	v17, v9, v12
	vand.vi	v9, v13, 15
	ld	s7, 552(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, s7, v9
	addi	a0, s11, 536
	vle8.v	v16, (a0)
	vsrl.vi	v9, v13, 4
	addi	a0, s11, 296
	vle8.v	v12, (a0)
	ld	s8, 584(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, s8, v9
	vand.vi	v9, v16, 15
	ld	s9, 576(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, s9, v9
	vand.vi	v9, v12, 15
	ld	a0, 800(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v12, 4
	addi	a0, s11, 552
	vle8.v	v12, (a0)
	addi	a0, s11, 312
	vle8.v	v13, (a0)
	ld	a0, 792(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v12, 15
	ld	a0, 776(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v13, 15
	ld	a0, 784(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v13, 4
	addi	a0, s11, 568
	vle8.v	v13, (a0)
	addi	a0, s11, 328
	vle8.v	v18, (a0)
	ld	a0, 696(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v13, 15
	ld	a0, 688(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v18, 15
	ld	a0, 704(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v18, 4
	addi	a0, s11, 584
	vle8.v	v18, (a0)
	addi	a0, s11, 344
	vle8.v	v19, (a0)
	ld	a0, 720(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v18, 15
	ld	a0, 712(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v19, 15
	ld	a0, 728(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v19, 4
	addi	a0, s11, 600
	vle8.v	v19, (a0)
	addi	a0, s11, 360
	vle8.v	v20, (a0)
	ld	a0, 752(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v19, 15
	ld	a0, 744(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v20, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1648(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v20, 4
	addi	a0, s11, 616
	vle8.v	v20, (a0)
	addi	a0, s11, 376
	vle8.v	v21, (a0)
	ld	a0, 768(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v20, 15
	ld	a0, 760(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v21, 15
	ld	a0, 736(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v21, 4
	addi	a0, s11, 632
	vle8.v	v21, (a0)
	addi	a0, s11, 392
	vle8.v	v22, (a0)
	ld	a0, 680(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v21, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2048(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v22, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2040(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v22, 4
	addi	a0, s11, 648
	vle8.v	v22, (a0)
	addi	a0, s11, 408
	vle8.v	v23, (a0)
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1984(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v22, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1968(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v23, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1960(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v23, 4
	addi	a0, s11, 664
	vle8.v	v23, (a0)
	addi	a0, s11, 424
	vle8.v	v24, (a0)
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1920(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v23, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1904(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v24, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1896(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v24, 4
	addi	a0, s11, 680
	vle8.v	v24, (a0)
	addi	a0, s11, 440
	vle8.v	v25, (a0)
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1912(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v24, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1888(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v25, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1936(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v25, 4
	addi	a0, s11, 696
	vle8.v	v25, (a0)
	addi	a0, s11, 456
	vle8.v	v26, (a0)
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1944(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v25, 15
	ld	a0, 2000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v26, 15
	ld	a0, 2008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v26, 4
	addi	a0, s11, 712
	vle8.v	v26, (a0)
	addi	a0, s11, 472
	vle8.v	v27, (a0)
	ld	a0, 2016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v26, 15
	ld	a0, 2024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v27, 15
	ld	a0, 2032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v27, 4
	addi	a0, s11, 728
	vle8.v	v27, (a0)
	addi	a0, s11, 488
	vle8.v	v28, (a0)
	sd	a2, 8(sp)                       # 8-byte Folded Spill
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2024(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v27, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2000(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vand.vi	v9, v28, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1992(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v9
	vsrl.vi	v9, v28, 4
	addi	a0, s11, 744
	vle8.v	v28, (a0)
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1952(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	addi	a0, s11, 504
	vle8.v	v9, (a0)
	vand.vi	v29, v28, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1976(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v29
	addi	a0, s11, 760
	vle8.v	v29, (a0)
	vand.vi	v30, v9, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1928(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v30
	vsrl.vi	v9, v9, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2008(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v9
	vand.vi	v9, v29, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2016(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v30, v8
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v4, 0
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v30, v10
	vzext.vf2	v10, v17
	vwmacc.vv	v8, v10, v14
	vwmacc.vv	v8, v30, v15
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v11, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -2032(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v16, 4
	ld	a0, 2040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v12, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1656(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v13, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1664(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v18, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1672(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v19, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1680(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v20, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1688(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v21, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1696(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v22, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1704(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v23, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1712(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v24, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1720(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v25, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1728(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v26, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1736(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	addi	a0, s11, 104
	vle8.v	v16, (a0)
	csrr	a0, vlenb
	li	a1, 27
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v16, (a0)                       # Unknown-size Folded Spill
	vsrl.vi	v11, v27, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1744(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	addi	a0, s11, 232
	vle8.v	v17, (a0)
	vsrl.vi	v11, v28, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1752(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	addi	a0, s11, 776
	vle8.v	v12, (a0)
	vsrl.vi	v11, v29, 4
	ld	a0, 1136(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	addi	a0, s11, 1032
	vle8.v	v11, (a0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v10, v14
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v12, 15
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1768(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v13
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v12, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1760(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v15, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v11, 15
	ld	a0, 1872(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v12
	vand.vi	v12, v17, 3
	vmv1r.v	v2, v17
	csrr	a0, vlenb
	slli	a1, a0, 4
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v17, (a0)                       # Unknown-size Folded Spill
	vsll.vi	v12, v12, 4
	vand.vi	v13, v16, 15
	vor.vv	v16, v12, v13
	addi	a0, s11, 120
	addi	a1, s11, 248
	vle8.v	v17, (a1)
	addi	a1, s11, 792
	vle8.v	v18, (a0)
	csrr	a0, vlenb
	li	a2, 26
	mul	a0, a0, a2
	ld	a2, 8(sp)                       # 8-byte Folded Reload
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v18, (a0)                       # Unknown-size Folded Spill
	vle8.v	v12, (a1)
	vand.vi	v13, v17, 3
	vmv1r.v	v1, v17
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v17, (a0)                       # Unknown-size Folded Spill
	vsll.vi	v13, v13, 4
	vand.vi	v17, v18, 15
	vor.vv	v18, v13, v17
	vand.vi	v13, v12, 15
	ld	a0, 1904(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v13
	vsrl.vi	v12, v12, 4
	addi	a0, s11, 1048
	vle8.v	v17, (a0)
	addi	a0, s11, 808
	vle8.v	v13, (a0)
	ld	a0, 1960(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v12
	vand.vi	v12, v17, 15
	ld	a0, 1976(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v12
	vand.vi	v12, v13, 15
	ld	a0, 1984(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v12
	vsrl.vi	v13, v13, 4
	addi	a0, s11, 1064
	vle8.v	v12, (a0)
	addi	a0, s11, 824
	vle8.v	v19, (a0)
	ld	a0, 1968(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v13
	vand.vi	v13, v12, 15
	ld	a0, 1992(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v13
	vand.vi	v13, v19, 15
	ld	a0, 1952(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v13
	vsrl.vi	v19, v19, 4
	addi	a0, s11, 1080
	vle8.v	v13, (a0)
	addi	a0, s11, 840
	vle8.v	v20, (a0)
	ld	a0, 1920(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v19
	vand.vi	v19, v13, 15
	ld	a0, 1784(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v19
	vand.vi	v19, v20, 15
	ld	a0, 1792(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v19
	vsrl.vi	v20, v20, 4
	addi	a0, s11, 1096
	vle8.v	v19, (a0)
	addi	a0, s11, 856
	vle8.v	v21, (a0)
	ld	a0, 1824(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v20
	vand.vi	v20, v19, 15
	ld	a0, 1848(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v20
	vand.vi	v20, v21, 15
	ld	a0, 1856(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v20
	vsrl.vi	v21, v21, 4
	addi	a0, s11, 1112
	vle8.v	v20, (a0)
	addi	a0, s11, 872
	vle8.v	v22, (a0)
	ld	a0, 1880(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v21
	vand.vi	v21, v20, 15
	ld	a0, 1888(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v21
	vand.vi	v21, v22, 15
	ld	a0, 1896(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v21
	vsrl.vi	v22, v22, 4
	addi	a0, s11, 1128
	vle8.v	v21, (a0)
	addi	a0, s11, 888
	vle8.v	v23, (a0)
	ld	a0, 1936(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v22
	vand.vi	v22, v21, 15
	ld	a0, 1928(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v22
	vand.vi	v22, v23, 15
	ld	a0, 1912(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v22
	vsrl.vi	v23, v23, 4
	addi	a0, s11, 1144
	vle8.v	v22, (a0)
	addi	a0, s11, 904
	vle8.v	v24, (a0)
	ld	a0, 1944(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v23
	vand.vi	v23, v22, 15
	ld	a0, 1712(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v23
	vand.vi	v23, v24, 15
	ld	a0, 1720(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v23
	vsrl.vi	v24, v24, 4
	addi	a0, s11, 1160
	vle8.v	v23, (a0)
	addi	a0, s11, 920
	vle8.v	v25, (a0)
	ld	a0, 1744(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v24
	vand.vi	v24, v23, 15
	ld	a0, 1752(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v24
	vand.vi	v24, v25, 15
	ld	a0, 1760(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v24
	vsrl.vi	v25, v25, 4
	addi	a0, s11, 1176
	vle8.v	v24, (a0)
	addi	a0, s11, 936
	vle8.v	v26, (a0)
	ld	a0, 1808(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v25
	vand.vi	v25, v24, 15
	ld	a0, 1800(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v25
	vand.vi	v25, v26, 15
	ld	a0, 1776(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v25
	vsrl.vi	v26, v26, 4
	addi	a0, s11, 1192
	vle8.v	v25, (a0)
	addi	a0, s11, 952
	vle8.v	v27, (a0)
	ld	a0, 1840(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v26
	vand.vi	v26, v25, 15
	ld	a0, 1832(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v26
	vand.vi	v26, v27, 15
	ld	a0, 1816(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v26
	vsrl.vi	v27, v27, 4
	addi	a0, s11, 1208
	vle8.v	v26, (a0)
	addi	a0, s11, 968
	vle8.v	v28, (a0)
	ld	a0, 1864(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vand.vi	v27, v26, 15
	ld	a0, 1584(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v27
	vand.vi	v27, v28, 15
	ld	a0, 1592(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v27
	vsrl.vi	v27, v28, 4
	addi	a0, s11, 1224
	vle8.v	v28, (a0)
	addi	a0, s11, 984
	vle8.v	v29, (a0)
	ld	a0, 1600(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vand.vi	v27, v28, 15
	ld	a0, 1616(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v27
	vand.vi	v27, v29, 15
	ld	a0, 1624(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v27
	vsrl.vi	v27, v29, 4
	addi	a0, s11, 1240
	vle8.v	v29, (a0)
	addi	a0, s11, 1000
	vle8.v	v30, (a0)
	ld	a0, 1648(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vand.vi	v27, v29, 15
	ld	a0, 1696(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v27
	vand.vi	v27, v30, 15
	ld	a0, 1704(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v27
	vsrl.vi	v27, v30, 4
	addi	a0, s11, 1256
	vle8.v	v30, (a0)
	ld	a0, 1736(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	addi	a0, s11, 1016
	vle8.v	v27, (a0)
	vand.vi	v31, v30, 15
	ld	a0, 1728(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v31
	addi	a0, s11, 1272
	vle8.v	v31, (a0)
	vand.vi	v7, v27, 15
	ld	a0, 1768(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a0, v7
	vsrl.vi	v27, v27, 4
	ld	a0, 1664(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vand.vi	v27, v31, 15
	ld	a0, 1656(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v27
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v27, v16
	vwmacc.vv	v8, v27, v10
	vzext.vf2	v10, v18
	vwmacc.vv	v8, v10, v14
	vwmacc.vv	v8, v27, v15
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v11, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a0, 1680(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v17, 4
	ld	a0, 1672(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v12, 4
	ld	a0, 1688(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v13, 4
	ld	a0, 1632(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v19, 4
	ld	a0, 1640(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v20, 4
	ld	a0, 1608(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v21, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1776(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v22, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1784(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v23, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1792(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v24, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1800(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v25, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1808(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v26, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1816(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v28, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1824(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v29, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1832(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v30, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1840(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v31, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1848(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	addi	a0, s11, 136
	vle8.v	v11, (a0)
	csrr	a0, vlenb
	li	a1, 25
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v11, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v10, v14
	addi	a0, s11, 152
	vle8.v	v13, (a0)
	csrr	a0, vlenb
	li	a1, 24
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v13, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v11, 15
	li	a0, 48
	vand.vx	v11, v6, a0
	vor.vv	v10, v11, v10
	addi	a0, s11, 1288
	vle8.v	v12, (a0)
	vand.vi	v11, v13, 15
	li	a0, 48
	vand.vx	v13, v3, a0
	vor.vv	v11, v13, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v12, 15
	ld	a0, 824(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v13
	vsrl.vi	v12, v12, 4
	addi	a0, s11, 1544
	vle8.v	v15, (a0)
	addi	a0, s11, 1304
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vle8.v	v13, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a0, 1576(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v17, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v15, 15
	ld	a0, 1480(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v12
	vand.vi	v12, v13, 15
	ld	a0, 1488(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v12
	vsrl.vi	v12, v13, 4
	addi	a0, s11, 1560
	vle8.v	v18, (a0)
	addi	a0, s11, 1320
	vle8.v	v13, (a0)
	ld	a0, 1536(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v12
	vand.vi	v12, v18, 15
	ld	a0, 1552(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v12
	vand.vi	v12, v13, 15
	ld	a0, 1560(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v12
	vsrl.vi	v13, v13, 4
	addi	a0, s11, 1576
	vle8.v	v12, (a0)
	addi	a0, s11, 1336
	vle8.v	v19, (a0)
	ld	a0, 1544(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v13
	vand.vi	v13, v12, 15
	ld	a0, 1568(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v13
	vand.vi	v13, v19, 15
	ld	a0, 1528(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v13
	vsrl.vi	v19, v19, 4
	addi	a0, s11, 1592
	vle8.v	v13, (a0)
	addi	a0, s11, 1352
	vle8.v	v20, (a0)
	ld	a0, 1512(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v19
	vand.vi	v19, v13, 15
	ld	a0, 1376(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v19
	vand.vi	v19, v20, 15
	ld	a0, 1384(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v19
	vsrl.vi	v20, v20, 4
	addi	a0, s11, 1608
	vle8.v	v19, (a0)
	addi	a0, s11, 1368
	vle8.v	v21, (a0)
	ld	a0, 1424(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v20
	vand.vi	v20, v19, 15
	ld	a0, 1432(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v20
	vand.vi	v20, v21, 15
	ld	a0, 1440(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v20
	vsrl.vi	v21, v21, 4
	addi	a0, s11, 1624
	vle8.v	v20, (a0)
	addi	a0, s11, 1384
	vle8.v	v22, (a0)
	ld	a0, 1464(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v21
	vand.vi	v21, v20, 15
	ld	a0, 1456(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v21
	vand.vi	v21, v22, 15
	ld	a0, 1448(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v21
	vsrl.vi	v22, v22, 4
	addi	a0, s11, 1640
	vle8.v	v21, (a0)
	addi	a0, s11, 1400
	vle8.v	v23, (a0)
	ld	a0, 1504(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v22
	vand.vi	v22, v21, 15
	ld	a0, 1496(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v22
	vand.vi	v22, v23, 15
	ld	a0, 1472(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v22
	vsrl.vi	v23, v23, 4
	addi	a0, s11, 1656
	vle8.v	v22, (a0)
	addi	a0, s11, 1416
	vle8.v	v24, (a0)
	ld	a0, 1520(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v23
	vand.vi	v23, v22, 15
	ld	a0, 1288(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v23
	vand.vi	v23, v24, 15
	ld	a0, 1296(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v23
	vsrl.vi	v24, v24, 4
	addi	a0, s11, 1672
	vle8.v	v23, (a0)
	addi	a0, s11, 1432
	vle8.v	v25, (a0)
	ld	a0, 1320(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v24
	vand.vi	v24, v23, 15
	ld	a0, 1328(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v24
	vand.vi	v24, v25, 15
	ld	a0, 1336(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v24
	vsrl.vi	v25, v25, 4
	addi	a0, s11, 1688
	vle8.v	v24, (a0)
	addi	a0, s11, 1448
	vle8.v	v26, (a0)
	ld	a0, 1368(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v25
	vand.vi	v25, v24, 15
	ld	a0, 1360(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v25
	vand.vi	v25, v26, 15
	ld	a0, 1352(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v25
	vsrl.vi	v26, v26, 4
	addi	a0, s11, 1704
	vle8.v	v25, (a0)
	addi	a0, s11, 1464
	vle8.v	v27, (a0)
	ld	a0, 1408(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v26
	vand.vi	v26, v25, 15
	ld	a0, 1400(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v26
	vand.vi	v26, v27, 15
	ld	a0, 1392(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v26
	vsrl.vi	v27, v27, 4
	addi	a0, s11, 1720
	vle8.v	v26, (a0)
	addi	a0, s11, 1480
	vle8.v	v28, (a0)
	ld	a0, 1416(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v27
	vand.vi	v27, v26, 15
	ld	a0, 1120(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v27
	vand.vi	v27, v28, 15
	ld	a0, 1128(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vsrl.vi	v27, v28, 4
	addi	a0, s11, 1736
	vle8.v	v28, (a0)
	addi	a0, s11, 1496
	vle8.v	v29, (a0)
	ld	a0, 1160(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v27
	vand.vi	v27, v28, 15
	ld	a0, 1168(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v27
	vand.vi	v27, v29, 15
	ld	a0, 1176(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vsrl.vi	v27, v29, 4
	addi	a0, s11, 1752
	vle8.v	v29, (a0)
	addi	a0, s11, 1512
	vle8.v	v30, (a0)
	ld	a0, 1224(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v27
	vand.vi	v27, v29, 15
	ld	a0, 1272(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v27
	vand.vi	v27, v30, 15
	ld	a0, 1280(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v27
	vsrl.vi	v27, v30, 4
	addi	a0, s11, 1768
	vle8.v	v30, (a0)
	ld	a0, 1312(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v27
	addi	a0, s11, 1528
	vle8.v	v27, (a0)
	vand.vi	v31, v30, 15
	ld	a0, 1304(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v31
	addi	a0, s11, 1784
	vle8.v	v31, (a0)
	vand.vi	v7, v27, 15
	ld	a0, 1344(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v7
	vsrl.vi	v27, v27, 4
	ld	a0, 1256(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v27
	vand.vi	v27, v31, 15
	ld	a0, 1264(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v27
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v27, v10
	vwmacc.vv	v8, v27, v14
	vzext.vf2	v10, v11
	vwmacc.vv	v8, v10, v16
	vwmacc.vv	v8, v27, v17
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v11, v15, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a0, 1240(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v18, 4
	ld	a0, 1232(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v12, 4
	ld	a0, 1248(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v13, 4
	ld	a0, 1192(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v19, 4
	ld	a0, 1200(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v20, 4
	ld	a0, 1208(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v21, 4
	ld	a0, 1216(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v22, 4
	ld	a0, 1184(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v23, 4
	ld	a0, 1152(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v24, 4
	ld	a0, 1144(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v25, 4
	ld	a0, 1112(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v26, 4
	ld	a0, 1096(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v28, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1856(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v29, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1864(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v30, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1872(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	vsrl.vi	v11, v31, 4
	lui	a0, 1
	add	a0, a0, sp
	ld	a0, -1880(a0)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a0, v11
	addi	a0, s11, 168
	vle8.v	v11, (a0)
	csrr	a0, vlenb
	li	a1, 19
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v11, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v10, v14
	addi	a0, s11, 184
	vle8.v	v12, (a0)
	csrr	a0, vlenb
	li	a1, 18
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 2047
	addi	a0, a0, 417
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v11, 15
	li	a0, 48
	vand.vx	v11, v2, a0
	vor.vv	v11, v11, v10
	addi	a0, s11, 1800
	vle8.v	v10, (a0)
	vand.vi	v12, v12, 15
	li	a0, 48
	vand.vx	v13, v1, a0
	vor.vv	v14, v13, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v15, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v10, 15
	ld	a0, 808(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v12
	vsrl.vi	v12, v10, 4
	ld	a0, 520(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v10, (a0)
	addi	a0, s11, 1816
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vle8.v	v13, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a0, 1064(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v17, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v10, 15
	ld	a0, 1024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v12
	vand.vi	v12, v13, 15
	ld	a0, 1032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v12
	vsrl.vi	v12, v13, 4
	ld	a0, 504(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v18, (a0)
	addi	a0, s11, 1832
	vle8.v	v13, (a0)
	ld	a0, 1040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v12
	vand.vi	v12, v18, 15
	ld	a0, 1048(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v12
	vand.vi	v12, v13, 15
	ld	a0, 1056(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v12
	vsrl.vi	v12, v13, 4
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v19, (a0)
	addi	a0, s11, 1848
	vle8.v	v13, (a0)
	ld	a0, 1088(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v12
	vand.vi	v12, v19, 15
	ld	a0, 1080(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v12
	vand.vi	v12, v13, 15
	ld	a0, 1072(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v12
	vsrl.vi	v12, v13, 4
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v20, (a0)
	ld	a0, 1104(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v12
	addi	a0, s11, 1864
	vle8.v	v13, (a0)
	vand.vi	v12, v20, 15
	ld	a0, 936(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v12
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	add	a0, a0, s11
	vle8.v	v12, (a0)
	vand.vi	v21, v13, 15
	ld	a0, 920(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v21
	vsrl.vi	v13, v13, 4
	ld	a0, 952(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v13
	vand.vi	v13, v12, 15
	ld	a0, 960(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v13
	addi	a0, s11, 1880
	vle8.v	v21, (a0)
	addiw	a0, t1, -1960
	sd	a0, 320(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v13, (a0)
	vand.vi	v22, v21, 15
	ld	a0, 944(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v22
	vsrl.vi	v21, v21, 4
	ld	a0, 976(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v21
	vand.vi	v21, v13, 15
	ld	a0, 984(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v21
	addi	a0, s11, 1896
	vle8.v	v22, (a0)
	addiw	a0, t1, -1944
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v21, (a0)
	vand.vi	v23, v22, 15
	ld	a0, 968(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v23
	vsrl.vi	v22, v22, 4
	ld	a0, 1008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v22
	vand.vi	v22, v21, 15
	ld	a0, 1000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v22
	addi	a0, s11, 1912
	vle8.v	v23, (a0)
	addiw	a0, t1, -1928
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v22, (a0)
	vand.vi	v24, v23, 15
	ld	a0, 992(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v24
	vsrl.vi	v23, v23, 4
	ld	a0, 1016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v23
	vand.vi	v23, v22, 15
	ld	a0, 840(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v23
	addi	a0, s11, 1928
	vle8.v	v24, (a0)
	addiw	a0, t1, -1912
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v23, (a0)
	vand.vi	v25, v24, 15
	ld	a0, 832(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v25
	vsrl.vi	v24, v24, 4
	ld	a0, 856(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v24
	vand.vi	v24, v23, 15
	ld	a0, 864(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v24
	addi	a0, s11, 1944
	vle8.v	v25, (a0)
	addiw	a0, t1, -1896
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v24, (a0)
	vand.vi	v26, v25, 15
	ld	a0, 848(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v26
	vsrl.vi	v25, v25, 4
	ld	a0, 880(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v25
	vand.vi	v25, v24, 15
	ld	a0, 888(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v25
	addi	a0, s11, 1960
	vle8.v	v26, (a0)
	addiw	a0, t1, -1880
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v25, (a0)
	vand.vi	v27, v26, 15
	ld	a0, 872(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v27
	vsrl.vi	v26, v26, 4
	ld	a0, 912(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v26
	vand.vi	v26, v25, 15
	ld	a0, 904(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v26
	addi	a0, s11, 1976
	vle8.v	v27, (a0)
	addiw	a0, t1, -1864
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v26, (a0)
	vand.vi	v28, v27, 15
	ld	a0, 896(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v28
	vsrl.vi	v27, v27, 4
	ld	a0, 928(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v27
	vand.vi	v27, v26, 15
	ld	a0, 600(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v27
	addi	a0, s11, 1992
	vle8.v	v28, (a0)
	addiw	a0, t1, -1848
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v27, (a0)
	vand.vi	v29, v28, 15
	ld	a0, 816(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v29
	vsrl.vi	v28, v28, 4
	ld	a0, 608(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v28
	vand.vi	v28, v27, 15
	ld	a0, 616(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v28
	addi	a0, s11, 2008
	vle8.v	v28, (a0)
	addiw	a0, t1, -1832
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v29, (a0)
	vand.vi	v30, v28, 15
	ld	a0, 624(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v30
	vsrl.vi	v28, v28, 4
	ld	a0, 632(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v28
	vand.vi	v28, v29, 15
	ld	a0, 640(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v28
	addi	a0, s11, 2024
	vle8.v	v28, (a0)
	addiw	a0, t1, -1816
	sd	a0, 240(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v30, (a0)
	vand.vi	v31, v28, 15
	ld	a0, 648(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v31
	vsrl.vi	v28, v28, 4
	ld	a0, 664(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a0, v28
	vand.vi	v28, v30, 15
	ld	a0, 656(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v17, a0, v28
	addi	a0, s11, 2040
	vle8.v	v28, (a0)
	addiw	a0, t1, -1800
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	add	a0, a0, s11
	vle8.v	v31, (a0)
	vand.vi	v7, v28, 15
	ld	a0, 672(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v15, a0, v7
	vsrl.vi	v28, v28, 4
	vwmacc.vx	v16, s6, v28
	vand.vi	v28, v31, 15
	vwmacc.vx	v17, s5, v28
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v28, v11
	vwmacc.vv	v8, v28, v15
	vzext.vf2	v11, v14
	vwmacc.vv	v8, v11, v16
	vwmacc.vv	v8, v28, v17
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v10, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v14, s4, v10
	vsrl.vi	v10, v18, 4
	vwmacc.vx	v14, s2, v10
	vsrl.vi	v10, v19, 4
	vwmacc.vx	v14, s3, v10
	vsrl.vi	v10, v20, 4
	ld	t1, 536(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, t1, v10
	vsrl.vi	v10, v12, 4
	vwmacc.vx	v14, t5, v10
	vsrl.vi	v10, v13, 4
	vwmacc.vx	v14, s1, v10
	vsrl.vi	v10, v21, 4
	vwmacc.vx	v14, t6, v10
	vsrl.vi	v10, v22, 4
	vwmacc.vx	v14, s0, v10
	vsrl.vi	v10, v23, 4
	vwmacc.vx	v14, t4, v10
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v14, t3, v10
	mv	s0, t3
	sd	t3, 136(sp)                     # 8-byte Folded Spill
	vsrl.vi	v10, v25, 4
	vwmacc.vx	v14, t2, v10
	mv	t3, t2
	vsrl.vi	v10, v26, 4
	vwmacc.vx	v14, t0, v10
	vsrl.vi	v10, v27, 4
	vwmacc.vx	v14, a7, v10
	mv	s1, a7
	sd	a7, 128(sp)                     # 8-byte Folded Spill
	vsrl.vi	v10, v29, 4
	vwmacc.vx	v14, a6, v10
	vsrl.vi	v10, v30, 4
	vwmacc.vx	v14, a5, v10
	mv	s2, a5
	sd	a5, 120(sp)                     # 8-byte Folded Spill
	vsrl.vi	v10, v31, 4
	vwmacc.vx	v14, a4, v10
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v11, v14
	addi	a0, s11, 16
	vle16.v	v10, (a0)
	ld	a0, 104(sp)                     # 8-byte Folded Reload
	ld	a1, 408(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	addi	a1, a0, 256
	vle8.v	v12, (a1)
	vfwcvt.f.f.v	v14, v10
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v10, v14, fa5
	vfcvt.f.x.v	v14, v8
	csrr	a1, vlenb
	li	a4, 20
	mul	a1, a1, a4
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl2r.v	v8, (a1)                        # Unknown-size Folded Reload
	vfmadd.vv	v14, v10, v8
	csrr	a1, vlenb
	li	a4, 20
	mul	a1, a1, a4
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs2r.v	v14, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v11, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v12, 15
	vwmacc.vx	v11, a2, v8
	vsrl.vi	v8, v12, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v12, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vwmacc.vx	v12, a3, v8
	addi	a1, a0, 64
	addi	a5, a0, 512
	vle8.v	v13, (a5)
	addi	a5, a0, 192
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vle8.v	v9, (a5)
	vle8.v	v10, (a1)
	csrr	a1, vlenb
	slli	a2, a1, 4
	sub	a1, a2, a1
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v10, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v13, 15
	vwmacc.vx	v14, s10, v8
	vand.vi	v8, v9, 3
	vmv1r.v	v6, v9
	csrr	a1, vlenb
	li	a2, 10
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v9, (a1)                        # Unknown-size Folded Spill
	vsll.vi	v8, v8, 4
	vand.vi	v9, v10, 15
	vor.vv	v8, v8, v9
	addi	a1, a0, 80
	addi	a5, a0, 208
	vle8.v	v15, (a5)
	addi	a5, a0, 272
	vle8.v	v16, (a1)
	csrr	a1, vlenb
	li	a2, 14
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v16, (a1)                       # Unknown-size Folded Spill
	vle8.v	v9, (a5)
	vand.vi	v10, v15, 3
	vmv1r.v	v3, v15
	csrr	a1, vlenb
	slli	a1, a1, 3
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v15, (a1)                       # Unknown-size Folded Spill
	vsll.vi	v10, v10, 4
	vand.vi	v15, v16, 15
	vor.vv	v16, v10, v15
	vand.vi	v10, v9, 15
	vwmacc.vx	v11, s7, v10
	vsrl.vi	v9, v9, 4
	addi	a1, a0, 528
	vle8.v	v15, (a1)
	addi	a1, a0, 288
	vle8.v	v10, (a1)
	vwmacc.vx	v12, s8, v9
	vand.vi	v9, v15, 15
	vwmacc.vx	v14, s9, v9
	vand.vi	v9, v10, 15
	ld	a1, 800(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 544
	vle8.v	v20, (a1)
	addi	a1, a0, 304
	vle8.v	v10, (a1)
	ld	a1, 792(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v20, 15
	ld	a1, 776(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	ld	a1, 784(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 560
	vle8.v	v21, (a1)
	addi	a1, a0, 320
	vle8.v	v10, (a1)
	ld	a1, 696(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v21, 15
	ld	a1, 688(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	ld	a1, 704(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 576
	vle8.v	v18, (a1)
	addi	a1, a0, 336
	vle8.v	v10, (a1)
	ld	a1, 720(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v18, 15
	ld	a1, 712(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	ld	a1, 728(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 592
	vle8.v	v19, (a1)
	addi	a1, a0, 352
	vle8.v	v10, (a1)
	ld	a1, 752(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v19, 15
	ld	a1, 744(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1648(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 608
	vle8.v	v22, (a1)
	addi	a1, a0, 368
	vle8.v	v10, (a1)
	ld	a1, 768(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v22, 15
	ld	a1, 760(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	ld	a1, 736(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 624
	vle8.v	v24, (a1)
	addi	a1, a0, 384
	vle8.v	v10, (a1)
	ld	a1, 680(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v24, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2048(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2040(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 640
	vle8.v	v25, (a1)
	addi	a1, a0, 400
	vle8.v	v10, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1984(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v25, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1968(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1960(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 656
	vle8.v	v26, (a1)
	addi	a1, a0, 416
	vle8.v	v10, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1920(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v26, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1904(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v10, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1896(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v10, 4
	addi	a1, a0, 672
	vle8.v	v10, (a1)
	addi	a1, a0, 432
	vle8.v	v17, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1912(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v10, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1888(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v17, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1936(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v17, 4
	addi	a1, a0, 688
	vle8.v	v17, (a1)
	addi	a1, a0, 448
	vle8.v	v23, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1944(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v17, 15
	ld	a1, 2000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v23, 15
	ld	a1, 2008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v23, 4
	addi	a1, a0, 704
	vle8.v	v23, (a1)
	addi	a1, a0, 464
	vle8.v	v27, (a1)
	ld	a1, 2016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v23, 15
	ld	a1, 2024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v27, 15
	ld	a1, 2032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v27, 4
	addi	a1, a0, 720
	vle8.v	v27, (a1)
	addi	a1, a0, 480
	vle8.v	v28, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2024(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v27, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2000(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vand.vi	v9, v28, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1992(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v9
	vsrl.vi	v9, v28, 4
	addi	a1, a0, 736
	vle8.v	v28, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1952(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	addi	a1, a0, 496
	vle8.v	v9, (a1)
	vand.vi	v29, v28, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1976(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v29
	addi	a1, a0, 752
	vle8.v	v29, (a1)
	vand.vi	v30, v9, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1928(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v30
	vsrl.vi	v9, v9, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2008(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v9
	vand.vi	v9, v29, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2016(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v30, v8
	vmv2r.v	v8, v4
	vwmacc.vv	v8, v30, v11
	vzext.vf2	v11, v16
	vwmacc.vv	v8, v11, v12
	vwmacc.vv	v8, v30, v14
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v13, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v13, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2032(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v15, 4
	ld	a1, 2040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v20, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1656(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v21, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1664(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v18, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1672(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v19, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1680(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v22, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1688(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v24, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1696(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v25, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1704(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v12, v26, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1712(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsrl.vi	v10, v10, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1720(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v10
	vsrl.vi	v10, v17, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1728(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v10
	vsrl.vi	v10, v23, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1736(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v10
	vsrl.vi	v10, v27, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1744(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v10
	vsrl.vi	v10, v28, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1752(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v10
	addi	a1, a0, 768
	vle8.v	v10, (a1)
	vsrl.vi	v12, v29, 4
	ld	a1, 1136(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v13, a1, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v11, v13
	vmv.v.i	v11, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v10, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1768(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v12
	vsrl.vi	v10, v10, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v12, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1760(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	addi	a1, a0, 96
	addi	a5, a0, 1024
	vle8.v	v14, (a5)
	addi	a5, a0, 224
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v15, 0
	vle8.v	v13, (a5)
	vle8.v	v16, (a1)
	csrr	a1, vlenb
	li	a2, 13
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v16, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v14, 15
	ld	a1, 1872(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 3
	vmv1r.v	v5, v13
	csrr	a1, vlenb
	slli	a2, a1, 2
	add	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v13, (a1)                       # Unknown-size Folded Spill
	vsll.vi	v10, v10, 4
	vand.vi	v13, v16, 15
	vor.vv	v16, v10, v13
	addi	a1, a0, 112
	addi	a5, a0, 240
	vle8.v	v17, (a5)
	addi	a5, a0, 784
	vle8.v	v18, (a1)
	csrr	a1, vlenb
	li	a2, 12
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v18, (a1)                       # Unknown-size Folded Spill
	vle8.v	v10, (a5)
	vand.vi	v13, v17, 3
	vmv1r.v	v4, v17
	csrr	a1, vlenb
	slli	a1, a1, 2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v17, (a1)                       # Unknown-size Folded Spill
	vsll.vi	v13, v13, 4
	vand.vi	v17, v18, 15
	vor.vv	v21, v13, v17
	vand.vi	v13, v10, 15
	ld	a1, 1904(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v13
	vsrl.vi	v10, v10, 4
	addi	a1, a0, 1040
	vle8.v	v20, (a1)
	addi	a1, a0, 800
	vle8.v	v13, (a1)
	ld	a1, 1960(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v20, 15
	ld	a1, 1976(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1984(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1056
	vle8.v	v22, (a1)
	addi	a1, a0, 816
	vle8.v	v13, (a1)
	ld	a1, 1968(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v22, 15
	ld	a1, 1992(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1952(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1072
	vle8.v	v24, (a1)
	addi	a1, a0, 832
	vle8.v	v13, (a1)
	ld	a1, 1920(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v24, 15
	ld	a1, 1784(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1792(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1088
	vle8.v	v18, (a1)
	addi	a1, a0, 848
	vle8.v	v13, (a1)
	ld	a1, 1824(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v18, 15
	ld	a1, 1848(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1856(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1104
	vle8.v	v19, (a1)
	addi	a1, a0, 864
	vle8.v	v13, (a1)
	ld	a1, 1880(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v19, 15
	ld	a1, 1888(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1896(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1120
	vle8.v	v26, (a1)
	addi	a1, a0, 880
	vle8.v	v13, (a1)
	ld	a1, 1936(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v26, 15
	ld	a1, 1928(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1912(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1136
	vle8.v	v28, (a1)
	addi	a1, a0, 896
	vle8.v	v13, (a1)
	ld	a1, 1944(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v28, 15
	ld	a1, 1712(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1720(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1152
	vle8.v	v30, (a1)
	addi	a1, a0, 912
	vle8.v	v13, (a1)
	ld	a1, 1744(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v30, 15
	ld	a1, 1752(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1760(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1168
	vle8.v	v7, (a1)
	addi	a1, a0, 928
	vle8.v	v13, (a1)
	ld	a1, 1808(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v10
	vand.vi	v10, v7, 15
	ld	a1, 1800(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1776(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v10
	vsrl.vi	v13, v13, 4
	addi	a1, a0, 1184
	vle8.v	v10, (a1)
	addi	a1, a0, 944
	vle8.v	v17, (a1)
	ld	a1, 1840(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v13
	vand.vi	v13, v10, 15
	ld	a1, 1832(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v13
	vand.vi	v13, v17, 15
	ld	a1, 1816(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v13
	vsrl.vi	v17, v17, 4
	addi	a1, a0, 1200
	vle8.v	v13, (a1)
	addi	a1, a0, 960
	vle8.v	v23, (a1)
	ld	a1, 1864(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v17
	vand.vi	v17, v13, 15
	ld	a1, 1584(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v17
	vand.vi	v17, v23, 15
	ld	a1, 1592(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v17
	vsrl.vi	v23, v23, 4
	addi	a1, a0, 1216
	vle8.v	v17, (a1)
	addi	a1, a0, 976
	vle8.v	v25, (a1)
	ld	a1, 1600(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v23
	vand.vi	v23, v17, 15
	ld	a1, 1616(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v23
	vand.vi	v23, v25, 15
	ld	a1, 1624(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v23
	vsrl.vi	v25, v25, 4
	addi	a1, a0, 1232
	vle8.v	v23, (a1)
	addi	a1, a0, 992
	vle8.v	v27, (a1)
	ld	a1, 1648(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v25
	vand.vi	v25, v23, 15
	ld	a1, 1696(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v25
	vand.vi	v25, v27, 15
	ld	a1, 1704(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v25
	vsrl.vi	v27, v27, 4
	addi	a1, a0, 1248
	vle8.v	v25, (a1)
	ld	a1, 1736(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v27
	addi	a1, a0, 1008
	vle8.v	v27, (a1)
	vand.vi	v29, v25, 15
	ld	a1, 1728(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v29
	addi	a1, a0, 1264
	vle8.v	v29, (a1)
	vand.vi	v31, v27, 15
	ld	a1, 1768(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v11, a1, v31
	vsrl.vi	v27, v27, 4
	ld	a1, 1664(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v12, a1, v27
	vand.vi	v27, v29, 15
	ld	a1, 1656(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v15, a1, v27
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v27, v16
	vwmacc.vv	v8, v27, v11
	vzext.vf2	v11, v21
	vwmacc.vv	v8, v11, v12
	vwmacc.vv	v8, v27, v15
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v14, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1680(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v20, 4
	ld	a1, 1672(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v22, 4
	ld	a1, 1688(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v24, 4
	ld	a1, 1632(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v18, 4
	ld	a1, 1640(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v19, 4
	ld	a1, 1608(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v26, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1776(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v28, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1784(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v30, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1792(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v7, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1800(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v10, v10, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1808(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1816(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v17, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1824(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v23, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1832(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v25, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1840(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v29, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1848(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	addi	a1, a0, 128
	vle8.v	v10, (a1)
	csrr	a1, vlenb
	li	a2, 11
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v10, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v11, v14
	addi	a1, a0, 144
	vle8.v	v12, (a1)
	csrr	a1, vlenb
	slli	a2, a1, 3
	add	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v12, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v10, 15
	li	a1, 48
	vand.vx	v11, v6, a1
	vor.vv	v11, v11, v10
	addi	a1, a0, 1280
	vle8.v	v10, (a1)
	vand.vi	v12, v12, 15
	li	a1, 48
	vand.vx	v13, v3, a1
	vor.vv	v12, v13, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v10, 15
	ld	a1, 824(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v13
	vsrl.vi	v16, v10, 4
	addi	a1, a0, 1536
	vle8.v	v15, (a1)
	addi	a1, a0, 1296
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v20, 0
	vle8.v	v10, (a1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1576(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v16
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v15, 15
	ld	a1, 1480(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vand.vi	v13, v10, 15
	ld	a1, 1488(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v13
	vsrl.vi	v10, v10, 4
	addi	a1, a0, 1552
	vle8.v	v22, (a1)
	addi	a1, a0, 1312
	vle8.v	v13, (a1)
	ld	a1, 1536(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v22, 15
	ld	a1, 1552(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1560(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1568
	vle8.v	v26, (a1)
	addi	a1, a0, 1328
	vle8.v	v13, (a1)
	ld	a1, 1544(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v26, 15
	ld	a1, 1568(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1528(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1584
	vle8.v	v28, (a1)
	addi	a1, a0, 1344
	vle8.v	v13, (a1)
	ld	a1, 1512(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v28, 15
	ld	a1, 1376(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1384(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1600
	vle8.v	v18, (a1)
	addi	a1, a0, 1360
	vle8.v	v13, (a1)
	ld	a1, 1424(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v18, 15
	ld	a1, 1432(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1440(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1616
	vle8.v	v19, (a1)
	addi	a1, a0, 1376
	vle8.v	v13, (a1)
	ld	a1, 1464(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v19, 15
	ld	a1, 1456(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1448(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1632
	vle8.v	v30, (a1)
	addi	a1, a0, 1392
	vle8.v	v13, (a1)
	ld	a1, 1504(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v30, 15
	ld	a1, 1496(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1472(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1648
	vle8.v	v7, (a1)
	addi	a1, a0, 1408
	vle8.v	v13, (a1)
	ld	a1, 1520(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v7, 15
	ld	a1, 1288(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1296(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1664
	vle8.v	v3, (a1)
	addi	a1, a0, 1424
	vle8.v	v13, (a1)
	ld	a1, 1320(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v3, 15
	ld	a1, 1328(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1336(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	addi	a1, a0, 1680
	vle8.v	v1, (a1)
	addi	a1, a0, 1440
	vle8.v	v13, (a1)
	ld	a1, 1368(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v10
	vand.vi	v10, v1, 15
	ld	a1, 1360(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vand.vi	v10, v13, 15
	ld	a1, 1352(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v13, v13, 4
	addi	a1, a0, 1696
	vle8.v	v10, (a1)
	addi	a1, a0, 1456
	vle8.v	v17, (a1)
	ld	a1, 1408(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v13
	vand.vi	v13, v10, 15
	ld	a1, 1400(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vand.vi	v13, v17, 15
	ld	a1, 1392(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v13
	vsrl.vi	v17, v17, 4
	addi	a1, a0, 1712
	vle8.v	v13, (a1)
	addi	a1, a0, 1472
	vle8.v	v21, (a1)
	ld	a1, 1416(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v17
	vand.vi	v17, v13, 15
	ld	a1, 1120(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v17
	vand.vi	v17, v21, 15
	ld	a1, 1128(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v17
	vsrl.vi	v21, v21, 4
	addi	a1, a0, 1728
	vle8.v	v17, (a1)
	addi	a1, a0, 1488
	vle8.v	v23, (a1)
	ld	a1, 1160(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v21
	vand.vi	v21, v17, 15
	ld	a1, 1168(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v21
	vand.vi	v21, v23, 15
	ld	a1, 1176(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v21
	vsrl.vi	v23, v23, 4
	addi	a1, a0, 1744
	vle8.v	v21, (a1)
	addi	a1, a0, 1504
	vle8.v	v24, (a1)
	ld	a1, 1224(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v23
	vand.vi	v23, v21, 15
	ld	a1, 1272(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v23
	vand.vi	v23, v24, 15
	ld	a1, 1280(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v23
	vsrl.vi	v24, v24, 4
	addi	a1, a0, 1760
	vle8.v	v23, (a1)
	ld	a1, 1312(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v24
	addi	a1, a0, 1520
	vle8.v	v24, (a1)
	vand.vi	v25, v23, 15
	ld	a1, 1304(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v25
	addi	a1, a0, 1776
	vle8.v	v25, (a1)
	vand.vi	v27, v24, 15
	ld	a1, 1344(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v27
	vsrl.vi	v24, v24, 4
	ld	a1, 1256(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v24
	vand.vi	v24, v25, 15
	ld	a1, 1264(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v24
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v24, v11
	vwmacc.vv	v8, v24, v14
	vzext.vf2	v11, v12
	vwmacc.vv	v8, v11, v20
	vwmacc.vv	v8, v24, v16
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v15, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v14, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1240(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v22, 4
	ld	a1, 1232(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v26, 4
	ld	a1, 1248(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v28, 4
	ld	a1, 1192(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v18, 4
	ld	a1, 1200(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v19, 4
	ld	a1, 1208(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v30, 4
	ld	a1, 1216(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v7, 4
	ld	a1, 1184(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v3, 4
	ld	a1, 1152(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v12, v1, 4
	ld	a1, 1144(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v12
	vsrl.vi	v10, v10, 4
	ld	a1, 1112(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v13, 4
	ld	a1, 1096(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v17, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1856(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v21, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1864(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v23, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1872(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	vsrl.vi	v10, v25, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1880(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v14, a1, v10
	addi	a1, a0, 160
	vle8.v	v10, (a1)
	csrr	a1, vlenb
	li	a2, 6
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v10, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v11, v14
	addi	a1, a0, 176
	vle8.v	v12, (a1)
	csrr	a1, vlenb
	slli	a2, a1, 3
	sub	a1, a2, a1
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v12, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v10, 15
	li	a1, 48
	vand.vx	v11, v5, a1
	vor.vv	v11, v11, v10
	addi	a1, a0, 1792
	vle8.v	v10, (a1)
	vand.vi	v12, v12, 15
	li	a1, 48
	vand.vx	v13, v4, a1
	vor.vv	v14, v13, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v10, 15
	ld	a1, 808(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v12
	vsrl.vi	v18, v10, 4
	ld	a1, 528(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v20, (a1)
	addi	a1, a0, 1808
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v26, 0
	vle8.v	v10, (a1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1064(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v18
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v22, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v20, 15
	ld	a1, 1024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v12
	vand.vi	v12, v10, 15
	ld	a1, 1032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v12
	vsrl.vi	v10, v10, 4
	ld	a1, 512(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v28, (a1)
	addi	a1, a0, 1824
	vle8.v	v12, (a1)
	ld	a1, 1040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v28, 15
	ld	a1, 1048(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 1056(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vsrl.vi	v10, v12, 4
	ld	a1, 496(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v30, (a1)
	addi	a1, a0, 1840
	vle8.v	v12, (a1)
	ld	a1, 1088(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v30, 15
	ld	a1, 1080(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 1072(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vsrl.vi	v10, v12, 4
	ld	a1, 480(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v7, (a1)
	addi	a1, a0, 1856
	vle8.v	v12, (a1)
	ld	a1, 1104(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v7, 15
	ld	a1, 936(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 920(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vsrl.vi	v10, v12, 4
	ld	a1, 464(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v18, (a1)
	addi	a1, a0, 1872
	vle8.v	v12, (a1)
	ld	a1, 952(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v18, 15
	ld	a1, 960(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 944(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	ld	a1, 392(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v10, v12, 4
	vle8.v	v19, (a1)
	addi	a1, a0, 1888
	vle8.v	v12, (a1)
	ld	a1, 976(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v19, 15
	ld	a1, 984(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 968(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	ld	a1, 400(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v10, v12, 4
	vle8.v	v3, (a1)
	addi	a1, a0, 1904
	vle8.v	v12, (a1)
	ld	a1, 1008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v3, 15
	ld	a1, 1000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 992(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	ld	a1, 384(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v10, v12, 4
	vle8.v	v1, (a1)
	addi	a1, a0, 1920
	vle8.v	v12, (a1)
	ld	a1, 1016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v1, 15
	ld	a1, 840(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 832(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	vsrl.vi	v10, v12, 4
	ld	a1, 448(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v0, (a1)
	addi	a1, a0, 1936
	vle8.v	v12, (a1)
	ld	a1, 856(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v10
	vand.vi	v10, v0, 15
	ld	a1, 864(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v10
	vand.vi	v10, v12, 15
	ld	a1, 848(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v10
	ld	a1, 368(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v12, v12, 4
	vle8.v	v10, (a1)
	addi	a1, a0, 1952
	vle8.v	v13, (a1)
	ld	a1, 880(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v12
	vand.vi	v12, v10, 15
	ld	a1, 888(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v12
	vand.vi	v12, v13, 15
	ld	a1, 872(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v12
	ld	a1, 376(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v12, v13, 4
	vle8.v	v24, (a1)
	addi	a1, a0, 1968
	vle8.v	v13, (a1)
	ld	a1, 912(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v12
	vand.vi	v12, v24, 15
	ld	a1, 904(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, a1, v12
	vand.vi	v12, v13, 15
	ld	a1, 896(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v12
	ld	a1, 360(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v13, v13, 4
	vle8.v	v12, (a1)
	addi	a1, a0, 1984
	vle8.v	v15, (a1)
	ld	a1, 928(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, a1, v13
	vand.vi	v13, v12, 15
	ld	s10, 600(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v22, s10, v13
	vand.vi	v13, v15, 15
	ld	a1, 816(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	ld	a1, 328(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v15, v15, 4
	vle8.v	v13, (a1)
	addi	a1, a0, 2000
	vle8.v	v17, (a1)
	ld	s9, 608(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, s9, v15
	vand.vi	v15, v13, 15
	ld	s8, 616(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, s8, v15
	vand.vi	v15, v17, 15
	ld	s7, 624(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, s7, v15
	ld	a1, 336(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v17, v17, 4
	vle8.v	v15, (a1)
	addi	a1, a0, 2016
	vle8.v	v21, (a1)
	ld	s6, 632(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, s6, v17
	vand.vi	v17, v15, 15
	ld	s4, 640(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, s4, v17
	vand.vi	v17, v21, 15
	ld	s5, 648(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, s5, v17
	ld	a1, 352(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v21, v21, 4
	vle8.v	v17, (a1)
	ld	s3, 664(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, s3, v21
	addi	a1, a0, 2032
	vle8.v	v21, (a1)
	vand.vi	v23, v17, 15
	ld	t6, 656(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, t6, v23
	ld	a1, 344(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v23, (a1)
	vand.vi	v25, v21, 15
	ld	a2, 672(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a2, v25
	vsrl.vi	v21, v21, 4
	ld	t4, 144(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v26, t4, v21
	vand.vi	v21, v23, 15
	ld	t5, 152(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v22, t5, v21
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v21, v11
	vwmacc.vv	v8, v21, v16
	vzext.vf2	v11, v14
	vwmacc.vv	v8, v11, v26
	vwmacc.vv	v8, v21, v22
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v14, v20, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a3, 232(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a3, v14
	vsrl.vi	v14, v28, 4
	ld	a4, 224(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a4, v14
	vsrl.vi	v14, v30, 4
	ld	a7, 200(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a7, v14
	vsrl.vi	v14, v7, 4
	vwmacc.vx	v16, t1, v14
	vsrl.vi	v14, v18, 4
	ld	a6, 216(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a6, v14
	vsrl.vi	v14, v19, 4
	ld	t1, 184(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, t1, v14
	vsrl.vi	v14, v3, 4
	ld	t0, 192(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, t0, v14
	vsrl.vi	v14, v1, 4
	ld	t2, 176(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, t2, v14
	vsrl.vi	v14, v0, 4
	ld	a1, 592(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v14
	vsrl.vi	v10, v10, 4
	vwmacc.vx	v16, s0, v10
	vsrl.vi	v10, v24, 4
	vwmacc.vx	v16, t3, v10
	vsrl.vi	v10, v12, 4
	ld	s0, 168(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, s0, v10
	vsrl.vi	v10, v13, 4
	vwmacc.vx	v16, s1, v10
	vsrl.vi	v10, v15, 4
	ld	s1, 160(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, s1, v10
	vsrl.vi	v10, v17, 4
	vwmacc.vx	v16, s2, v10
	vsrl.vi	v10, v23, 4
	ld	s2, 208(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, s2, v10
	vle16.v	v10, (a0)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v11, v16
	addi	a1, a0, 264
	vle8.v	v11, (a1)
	vfwcvt.f.f.v	v12, v10
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v12, v12, fa5
	vfcvt.f.x.v	v14, v8
	csrr	a1, vlenb
	li	a5, 42
	mul	a1, a1, a5
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl2r.v	v8, (a1)                        # Unknown-size Folded Reload
	vfmadd.vv	v14, v12, v8
	csrr	a1, vlenb
	li	a5, 42
	mul	a1, a1, a5
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs2r.v	v14, (a1)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v16, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v11, 15
	ld	a1, 544(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v8
	vsrl.vi	v8, v11, 4
	addi	a1, a0, 520
	vle8.v	v18, (a1)
	addi	a1, a0, 280
	vle8.v	v9, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v19, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 560(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v18, 15
	ld	a1, 568(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 552(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v8
	vsrl.vi	v8, v9, 4
	addi	a1, a0, 536
	vle8.v	v20, (a1)
	addi	a1, a0, 296
	vle8.v	v9, (a1)
	ld	a1, 584(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v8
	vand.vi	v8, v20, 15
	ld	a1, 576(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 800(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v8
	vsrl.vi	v9, v9, 4
	addi	a1, a0, 552
	vle8.v	v8, (a1)
	addi	a1, a0, 312
	vle8.v	v11, (a1)
	ld	a1, 792(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v9
	vand.vi	v9, v8, 15
	ld	a1, 776(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v9
	vand.vi	v9, v11, 15
	ld	a1, 784(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v9
	vsrl.vi	v11, v11, 4
	addi	a1, a0, 568
	vle8.v	v9, (a1)
	addi	a1, a0, 328
	vle8.v	v12, (a1)
	ld	a1, 696(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v9, 15
	ld	a1, 688(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 704(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v11
	vsrl.vi	v11, v12, 4
	addi	a1, a0, 584
	vle8.v	v24, (a1)
	addi	a1, a0, 344
	vle8.v	v12, (a1)
	ld	a1, 720(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v24, 15
	ld	a1, 712(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 728(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v11
	vsrl.vi	v11, v12, 4
	addi	a1, a0, 600
	vle8.v	v12, (a1)
	addi	a1, a0, 360
	vle8.v	v13, (a1)
	ld	a1, 752(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 744(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v13, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1648(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 616
	vle8.v	v14, (a1)
	addi	a1, a0, 376
	vle8.v	v13, (a1)
	ld	a1, 768(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v14, 15
	ld	a1, 760(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 736(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v11
	vsrl.vi	v13, v13, 4
	addi	a1, a0, 632
	vle8.v	v11, (a1)
	addi	a1, a0, 392
	vle8.v	v15, (a1)
	ld	a1, 680(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v11, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2048(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v15, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2040(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vsrl.vi	v13, v15, 4
	addi	a1, a0, 648
	vle8.v	v15, (a1)
	addi	a1, a0, 408
	vle8.v	v17, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1984(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v15, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1968(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v17, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1960(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vsrl.vi	v13, v17, 4
	addi	a1, a0, 664
	vle8.v	v26, (a1)
	addi	a1, a0, 424
	vle8.v	v17, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1920(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v26, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1904(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v17, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1896(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vsrl.vi	v13, v17, 4
	addi	a1, a0, 680
	vle8.v	v22, (a1)
	addi	a1, a0, 440
	vle8.v	v17, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1912(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v22, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1888(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v17, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1936(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vsrl.vi	v13, v17, 4
	addi	a1, a0, 696
	vle8.v	v23, (a1)
	addi	a1, a0, 456
	vle8.v	v17, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1944(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v23, 15
	ld	a1, 2000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v17, 15
	ld	a1, 2008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vsrl.vi	v13, v17, 4
	addi	a1, a0, 712
	vle8.v	v21, (a1)
	addi	a1, a0, 472
	vle8.v	v17, (a1)
	ld	a1, 2016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v21, 15
	ld	a1, 2024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v17, 15
	ld	a1, 2032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v13
	vsrl.vi	v17, v17, 4
	addi	a1, a0, 728
	vle8.v	v13, (a1)
	addi	a1, a0, 488
	vle8.v	v25, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2024(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v17
	vand.vi	v17, v13, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2000(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v17
	vand.vi	v17, v25, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1992(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v17
	vsrl.vi	v17, v25, 4
	addi	a1, a0, 744
	vle8.v	v27, (a1)
	addi	a1, a0, 504
	vle8.v	v25, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1952(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v17
	vand.vi	v17, v27, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1976(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v17
	vand.vi	v17, v25, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1928(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v16, a1, v17
	vsrl.vi	v17, v25, 4
	addi	a1, a0, 760
	vle8.v	v28, (a1)
	addi	a1, a0, 200
	vle8.v	v25, (a1)
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2008(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v17
	vand.vi	v17, v28, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2016(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v17
	vand.vi	v17, v25, 3
	vmv1r.v	v29, v25
	vsll.vi	v30, v17, 4
	addi	a1, a0, 72
	addi	a5, a0, 88
	vle8.v	v31, (a1)
	csrr	a1, vlenb
	slli	a1, a1, 1
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v31, (a1)                       # Unknown-size Folded Spill
	addi	a1, a0, 216
	vle8.v	v25, (a1)
	vle8.v	v17, (a5)
	csrr	a1, vlenb
	slli	a5, a1, 1
	add	a1, a1, a5
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v17, (a1)                       # Unknown-size Folded Spill
	vand.vi	v7, v31, 15
	vor.vv	v30, v30, v7
	vand.vi	v7, v25, 3
	vsll.vi	v7, v7, 4
	vand.vi	v5, v17, 15
	vor.vv	v7, v7, v5
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v5, v30
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v0, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v0, v5, v16
	vzext.vf2	v16, v7
	vwmacc.vv	v0, v16, v19
	vwmacc.vv	v0, v5, v10
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v18, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -2032(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v10
	vsrl.vi	v10, v20, 4
	ld	a1, 2040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v10
	vsrl.vi	v8, v8, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1656(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v9, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1664(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v24, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1672(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v12, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1680(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v14, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1688(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v11, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1696(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v15, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1704(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v26, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1712(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v22, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1720(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v23, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1728(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v21, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1736(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v13, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1744(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v27, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1752(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	addi	a1, a0, 776
	vle8.v	v8, (a1)
	vsrl.vi	v9, v28, 4
	ld	a1, 1136(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v0, v16, v18
	vmv.v.i	v18, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v8, 15
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1768(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v9
	vsrl.vi	v8, v8, 4
	addi	a1, a0, 1032
	vle8.v	v19, (a1)
	addi	a1, a0, 792
	vle8.v	v9, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1760(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v20, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v19, 15
	ld	a1, 1872(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1904(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v9, 4
	addi	a1, a0, 1048
	vle8.v	v3, (a1)
	addi	a1, a0, 808
	vle8.v	v9, (a1)
	ld	a1, 1960(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vand.vi	v8, v3, 15
	ld	a1, 1976(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1984(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v9, v9, 4
	addi	a1, a0, 1064
	vle8.v	v8, (a1)
	addi	a1, a0, 824
	vle8.v	v11, (a1)
	ld	a1, 1968(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v9
	vand.vi	v9, v8, 15
	ld	a1, 1992(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v9
	vand.vi	v9, v11, 15
	ld	a1, 1952(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v9
	vsrl.vi	v11, v11, 4
	addi	a1, a0, 1080
	vle8.v	v9, (a1)
	addi	a1, a0, 840
	vle8.v	v12, (a1)
	ld	a1, 1920(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v9, 15
	ld	a1, 1784(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 1792(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v12, 4
	addi	a1, a0, 1096
	vle8.v	v24, (a1)
	addi	a1, a0, 856
	vle8.v	v12, (a1)
	ld	a1, 1824(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v24, 15
	ld	a1, 1848(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 1856(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v12, 4
	addi	a1, a0, 1112
	vle8.v	v12, (a1)
	addi	a1, a0, 872
	vle8.v	v13, (a1)
	ld	a1, 1880(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 1888(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1896(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1128
	vle8.v	v14, (a1)
	addi	a1, a0, 888
	vle8.v	v13, (a1)
	ld	a1, 1936(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v14, 15
	ld	a1, 1928(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1912(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1144
	vle8.v	v30, (a1)
	addi	a1, a0, 904
	vle8.v	v13, (a1)
	ld	a1, 1944(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v30, 15
	ld	a1, 1712(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1720(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1160
	vle8.v	v7, (a1)
	addi	a1, a0, 920
	vle8.v	v13, (a1)
	ld	a1, 1744(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v7, 15
	ld	a1, 1752(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1760(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1176
	vle8.v	v28, (a1)
	addi	a1, a0, 936
	vle8.v	v13, (a1)
	ld	a1, 1808(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v28, 15
	ld	a1, 1800(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1776(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1192
	vle8.v	v22, (a1)
	addi	a1, a0, 952
	vle8.v	v13, (a1)
	ld	a1, 1840(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v22, 15
	ld	a1, 1832(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1816(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1208
	vle8.v	v23, (a1)
	addi	a1, a0, 968
	vle8.v	v13, (a1)
	ld	a1, 1864(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vand.vi	v11, v23, 15
	ld	a1, 1584(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1592(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v11
	vsrl.vi	v13, v13, 4
	addi	a1, a0, 1224
	vle8.v	v11, (a1)
	addi	a1, a0, 984
	vle8.v	v15, (a1)
	ld	a1, 1600(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vand.vi	v13, v11, 15
	ld	a1, 1616(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v13
	vand.vi	v13, v15, 15
	ld	a1, 1624(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v13
	vsrl.vi	v15, v15, 4
	addi	a1, a0, 1240
	vle8.v	v13, (a1)
	addi	a1, a0, 1000
	vle8.v	v16, (a1)
	ld	a1, 1648(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v15
	vand.vi	v15, v13, 15
	ld	a1, 1696(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v15
	vand.vi	v15, v16, 15
	ld	a1, 1704(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v15
	vsrl.vi	v15, v16, 4
	addi	a1, a0, 1256
	vle8.v	v5, (a1)
	addi	a1, a0, 1016
	vle8.v	v16, (a1)
	ld	a1, 1736(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v15
	vand.vi	v15, v5, 15
	ld	a1, 1728(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v15
	vand.vi	v15, v16, 15
	ld	a1, 1768(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v15
	vsrl.vi	v15, v16, 4
	addi	a1, a0, 1272
	vle8.v	v4, (a1)
	addi	a1, a0, 232
	vle8.v	v26, (a1)
	ld	a1, 1664(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v15
	vand.vi	v15, v4, 15
	ld	a1, 1656(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v20, a1, v15
	vand.vi	v15, v26, 3
	vsll.vi	v15, v15, 4
	addi	a1, a0, 104
	addi	a5, a0, 120
	vle8.v	v21, (a1)
	addi	a1, sp, 2047
	addi	a1, a1, 417
	vs1r.v	v21, (a1)                       # Unknown-size Folded Spill
	addi	a1, a0, 248
	vle8.v	v17, (a1)
	vle8.v	v16, (a5)
	csrr	a1, vlenb
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vs1r.v	v16, (a1)                       # Unknown-size Folded Spill
	vand.vi	v2, v21, 15
	vor.vv	v15, v15, v2
	vand.vi	v2, v17, 3
	vsll.vi	v2, v2, 4
	vand.vi	v6, v16, 15
	vor.vv	v6, v2, v6
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v2, v15
	vwmacc.vv	v0, v2, v18
	vzext.vf2	v15, v6
	vwmacc.vv	v0, v15, v10
	vwmacc.vv	v0, v2, v20
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v19, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1680(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v10
	vsrl.vi	v10, v3, 4
	ld	a1, 1672(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v10
	vsrl.vi	v8, v8, 4
	ld	a1, 1688(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v9, 4
	ld	a1, 1632(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v24, 4
	ld	a1, 1640(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v12, 4
	ld	a1, 1608(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v14, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1776(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v30, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1784(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v7, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1792(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v28, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1800(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v22, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1808(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v23, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1816(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v11, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1824(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v13, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1832(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v5, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1840(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	addi	a1, a0, 1288
	vle8.v	v8, (a1)
	vsrl.vi	v9, v4, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1848(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v0, v15, v18
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v8, 15
	ld	a1, 824(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v9
	vsrl.vi	v8, v8, 4
	addi	a1, a0, 1544
	vle8.v	v18, (a1)
	addi	a1, a0, 1304
	vle8.v	v9, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v19, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1576(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v5, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v18, 15
	ld	a1, 1480(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1488(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsrl.vi	v8, v9, 4
	addi	a1, a0, 1560
	vle8.v	v4, (a1)
	addi	a1, a0, 1320
	vle8.v	v9, (a1)
	ld	a1, 1536(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v8
	vand.vi	v8, v4, 15
	ld	a1, 1552(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1560(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsrl.vi	v9, v9, 4
	addi	a1, a0, 1576
	vle8.v	v8, (a1)
	addi	a1, a0, 1336
	vle8.v	v11, (a1)
	ld	a1, 1544(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v9
	vand.vi	v9, v8, 15
	ld	a1, 1568(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v9
	vand.vi	v9, v11, 15
	ld	a1, 1528(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v9
	vsrl.vi	v11, v11, 4
	addi	a1, a0, 1592
	vle8.v	v9, (a1)
	addi	a1, a0, 1352
	vle8.v	v12, (a1)
	ld	a1, 1512(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v9, 15
	ld	a1, 1376(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 1384(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v12, 4
	addi	a1, a0, 1608
	vle8.v	v24, (a1)
	addi	a1, a0, 1368
	vle8.v	v12, (a1)
	ld	a1, 1424(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v24, 15
	ld	a1, 1432(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 1440(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v12, 4
	addi	a1, a0, 1624
	vle8.v	v12, (a1)
	addi	a1, a0, 1384
	vle8.v	v13, (a1)
	ld	a1, 1464(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v12, 15
	ld	a1, 1456(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1448(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1640
	vle8.v	v14, (a1)
	addi	a1, a0, 1400
	vle8.v	v13, (a1)
	ld	a1, 1504(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v14, 15
	ld	a1, 1496(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1472(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1656
	vle8.v	v30, (a1)
	addi	a1, a0, 1416
	vle8.v	v13, (a1)
	ld	a1, 1520(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v30, 15
	ld	a1, 1288(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1296(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1672
	vle8.v	v7, (a1)
	addi	a1, a0, 1432
	vle8.v	v13, (a1)
	ld	a1, 1320(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v7, 15
	ld	a1, 1328(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1336(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1688
	vle8.v	v28, (a1)
	addi	a1, a0, 1448
	vle8.v	v13, (a1)
	ld	a1, 1368(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v28, 15
	ld	a1, 1360(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1352(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1704
	vle8.v	v22, (a1)
	addi	a1, a0, 1464
	vle8.v	v13, (a1)
	ld	a1, 1408(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v22, 15
	ld	a1, 1400(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1392(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v11, v13, 4
	addi	a1, a0, 1720
	vle8.v	v23, (a1)
	addi	a1, a0, 1480
	vle8.v	v13, (a1)
	ld	a1, 1416(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v11
	vand.vi	v11, v23, 15
	ld	a1, 1120(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v11
	vand.vi	v11, v13, 15
	ld	a1, 1128(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v11
	vsrl.vi	v13, v13, 4
	addi	a1, a0, 1736
	vle8.v	v11, (a1)
	addi	a1, a0, 1496
	vle8.v	v15, (a1)
	ld	a1, 1160(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v13
	vand.vi	v13, v11, 15
	ld	a1, 1168(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v13
	vand.vi	v13, v15, 15
	ld	a1, 1176(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v13
	vsrl.vi	v15, v15, 4
	addi	a1, a0, 1752
	vle8.v	v13, (a1)
	addi	a1, a0, 1512
	vle8.v	v20, (a1)
	ld	a1, 1224(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v15
	vand.vi	v15, v13, 15
	ld	a1, 1272(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v15
	vand.vi	v15, v20, 15
	ld	a1, 1280(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v15
	vsrl.vi	v15, v20, 4
	addi	a1, a0, 1768
	vle8.v	v6, (a1)
	addi	a1, a0, 1528
	vle8.v	v20, (a1)
	ld	a1, 1312(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v15
	vand.vi	v15, v6, 15
	ld	a1, 1304(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v15
	vand.vi	v15, v20, 15
	ld	a1, 1344(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v15
	vsrl.vi	v15, v20, 4
	addi	a1, a0, 1784
	vle8.v	v2, (a1)
	ld	a1, 1256(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v19, a1, v15
	addi	a1, a0, 136
	vle8.v	v3, (a1)
	vand.vi	v15, v2, 15
	ld	a1, 1264(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v15
	addi	a1, a0, 152
	vle8.v	v20, (a1)
	vand.vi	v15, v3, 15
	li	a1, 48
	vand.vx	v31, v29, a1
	vmv1r.v	v21, v29
	vor.vv	v15, v31, v15
	vand.vi	v31, v20, 15
	li	a1, 48
	vand.vx	v29, v25, a1
	vor.vv	v29, v29, v31
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v31, v15
	vwmacc.vv	v0, v31, v10
	vzext.vf2	v10, v29
	vwmacc.vv	v0, v10, v19
	vwmacc.vv	v0, v31, v5
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v15, v18, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v18, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1240(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v15
	vsrl.vi	v15, v4, 4
	ld	a1, 1232(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v15
	vsrl.vi	v8, v8, 4
	ld	a1, 1248(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v9, 4
	ld	a1, 1192(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v24, 4
	ld	a1, 1200(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v12, 4
	ld	a1, 1208(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v14, 4
	ld	a1, 1216(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v30, 4
	ld	a1, 1184(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v7, 4
	ld	a1, 1152(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v28, 4
	ld	a1, 1144(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v22, 4
	ld	a1, 1112(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v23, 4
	ld	a1, 1096(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v11, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1856(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v13, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1864(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	vsrl.vi	v8, v6, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1872(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v8
	addi	a1, a0, 1800
	vle8.v	v8, (a1)
	vsrl.vi	v9, v2, 4
	lui	a1, 1
	add	a1, a1, sp
	ld	a1, -1880(a1)                   # 8-byte Folded Reload
	vwmacc.vx	v18, a1, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v0, v10, v18
	vmv.v.i	v10, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v8, 15
	ld	a1, 808(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v9
	vsrl.vi	v8, v8, 4
	ld	a1, 520(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v2, (a1)
	addi	a1, a0, 1816
	vle8.v	v9, (a1)
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v5, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	ld	a1, 1064(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v4, 0
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v2, 15
	ld	a1, 1024(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1032(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsrl.vi	v8, v9, 4
	ld	a1, 504(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v16, (a1)
	addi	a1, a0, 1832
	vle8.v	v9, (a1)
	ld	a1, 1040(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v16, 15
	ld	a1, 1048(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1056(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsrl.vi	v8, v9, 4
	ld	a1, 488(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v15, (a1)
	addi	a1, a0, 1848
	vle8.v	v9, (a1)
	ld	a1, 1088(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v15, 15
	ld	a1, 1080(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 1072(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsrl.vi	v8, v9, 4
	ld	a1, 472(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v24, (a1)
	addi	a1, a0, 1864
	vle8.v	v9, (a1)
	ld	a1, 1104(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v24, 15
	ld	a1, 936(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 920(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	vsrl.vi	v8, v9, 4
	ld	a1, 456(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v13, (a1)
	addi	a1, a0, 1880
	vle8.v	v9, (a1)
	ld	a1, 952(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v13, 15
	ld	a1, 960(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 944(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 320(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v14, (a1)
	addi	a1, a0, 1896
	vle8.v	v9, (a1)
	ld	a1, 976(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v14, 15
	ld	a1, 984(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 968(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 312(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v30, (a1)
	addi	a1, a0, 1912
	vle8.v	v9, (a1)
	ld	a1, 1008(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v30, 15
	ld	a1, 1000(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 992(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 296(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v7, (a1)
	addi	a1, a0, 1928
	vle8.v	v9, (a1)
	ld	a1, 1016(sp)                    # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v7, 15
	ld	a1, 840(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 832(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 288(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v28, (a1)
	addi	a1, a0, 1944
	vle8.v	v9, (a1)
	ld	a1, 856(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v28, 15
	ld	a1, 864(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 848(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 280(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v22, (a1)
	addi	a1, a0, 1960
	vle8.v	v9, (a1)
	ld	a1, 880(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v22, 15
	ld	a1, 888(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 872(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 272(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v23, (a1)
	addi	a1, a0, 1976
	vle8.v	v9, (a1)
	ld	a1, 912(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v23, 15
	ld	a1, 904(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v4, a1, v8
	vand.vi	v8, v9, 15
	ld	a1, 896(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 264(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v12, (a1)
	addi	a1, a0, 1992
	vle8.v	v9, (a1)
	ld	a1, 928(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v5, a1, v8
	vand.vi	v8, v12, 15
	vwmacc.vx	v4, s10, v8
	vand.vi	v8, v9, 15
	ld	a1, 816(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v10, a1, v8
	ld	a1, 256(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v8, v9, 4
	vle8.v	v11, (a1)
	addi	a1, a0, 2008
	vle8.v	v18, (a1)
	vwmacc.vx	v5, s9, v8
	vand.vi	v8, v11, 15
	vwmacc.vx	v4, s8, v8
	vand.vi	v8, v18, 15
	vwmacc.vx	v10, s7, v8
	ld	a1, 248(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v18, v18, 4
	vle8.v	v9, (a1)
	addi	a1, a0, 2024
	vle8.v	v19, (a1)
	vwmacc.vx	v5, s6, v18
	vand.vi	v18, v9, 15
	vwmacc.vx	v4, s4, v18
	vand.vi	v18, v19, 15
	vwmacc.vx	v10, s5, v18
	ld	a1, 240(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vsrl.vi	v18, v19, 4
	vle8.v	v29, (a1)
	addi	a1, a0, 2040
	vle8.v	v19, (a1)
	vwmacc.vx	v5, s3, v18
	vand.vi	v18, v29, 15
	vwmacc.vx	v4, t6, v18
	vand.vi	v18, v19, 15
	vwmacc.vx	v10, a2, v18
	addi	a1, a0, 168
	vsrl.vi	v18, v19, 4
	vle8.v	v19, (a1)
	vwmacc.vx	v5, t4, v18
	li	a2, 48
	addi	a1, a0, 184
	vle8.v	v18, (a1)
	vand.vi	v31, v19, 15
	vand.vx	v6, v26, a2
	vor.vv	v31, v6, v31
	ld	a1, 304(sp)                     # 8-byte Folded Reload
	add	a1, a1, a0
	vle8.v	v6, (a1)
	vand.vi	v8, v18, 15
	vand.vx	v27, v17, a2
	vor.vv	v8, v27, v8
	vand.vi	v27, v6, 15
	vwmacc.vx	v4, t5, v27
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v27, v31
	vwmacc.vv	v0, v27, v10
	vzext.vf2	v10, v8
	vwmacc.vv	v0, v10, v5
	vmv.v.i	v31, 0
	vwmacc.vv	v0, v27, v4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v2, 4
	vmv1r.v	v27, v31
	vwmacc.vx	v27, a3, v8
	vsrl.vi	v8, v16, 4
	vwmacc.vx	v27, a4, v8
	vsrl.vi	v8, v15, 4
	vwmacc.vx	v27, a7, v8
	vsrl.vi	v8, v24, 4
	ld	a1, 536(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v27, a1, v8
	vsrl.vi	v8, v13, 4
	vwmacc.vx	v27, a6, v8
	vsrl.vi	v8, v14, 4
	vwmacc.vx	v27, t1, v8
	vsrl.vi	v8, v30, 4
	vwmacc.vx	v27, t0, v8
	vsrl.vi	v8, v7, 4
	vwmacc.vx	v27, t2, v8
	vsrl.vi	v8, v28, 4
	ld	a1, 592(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v27, a1, v8
	li	t2, -64
	vsrl.vi	v8, v22, 4
	ld	a1, 136(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v27, a1, v8
	vsrl.vi	v8, v23, 4
	vwmacc.vx	v27, t3, v8
	vsrl.vi	v8, v12, 4
	vwmacc.vx	v27, s0, v8
	vsrl.vi	v8, v11, 4
	ld	a1, 128(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v27, a1, v8
	vsrl.vi	v8, v9, 4
	vwmacc.vx	v27, s1, v8
	vsrl.vi	v8, v29, 4
	ld	a1, 120(sp)                     # 8-byte Folded Reload
	vwmacc.vx	v27, a1, v8
	addi	a1, a0, 16
	vle16.v	v8, (a1)
	vsrl.vi	v9, v6, 4
	vwmacc.vx	v27, s2, v9
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v0, v10, v27
	vfwcvt.f.f.v	v10, v8
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v8, v10, fa5
	vfcvt.f.x.v	v0, v0
	csrr	a1, vlenb
	li	a2, 46
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl2r.v	v10, (a1)                       # Unknown-size Folded Reload
	vfmadd.vv	v0, v8, v10
	csrr	a1, vlenb
	li	a2, 45
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v8, (a1)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	csrr	a1, vlenb
	li	a2, 36
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v13, (a1)                       # Unknown-size Folded Reload
	vand.vi	v9, v13, 12
	vsll.vi	v9, v9, 2
	vor.vv	v8, v9, v8
	csrr	a1, vlenb
	li	a2, 44
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	csrr	a1, vlenb
	li	a2, 37
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v14, (a1)                       # Unknown-size Folded Reload
	vand.vi	v10, v14, 12
	vsll.vi	v10, v10, 2
	vor.vv	v10, v10, v9
	csrr	a1, vlenb
	li	a2, 41
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	csrr	a1, vlenb
	slli	a1, a1, 5
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v15, (a1)                       # Unknown-size Folded Reload
	vand.vi	v11, v15, 12
	vsll.vi	v11, v11, 2
	vor.vv	v11, v11, v9
	csrr	a1, vlenb
	li	a2, 40
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	csrr	a1, vlenb
	slli	a2, a1, 5
	add	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v16, (a1)                       # Unknown-size Folded Reload
	vand.vi	v12, v16, 12
	vsll.vi	v12, v12, 2
	vor.vv	v12, v12, v9
	csrr	a1, vlenb
	li	a2, 38
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	vand.vx	v13, v13, t2
	vsrl.vi	v13, v13, 2
	vor.vv	v13, v13, v9
	csrr	a1, vlenb
	li	a2, 39
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	vand.vx	v14, v14, t2
	vsrl.vi	v14, v14, 2
	vor.vv	v14, v14, v9
	csrr	a1, vlenb
	li	a2, 34
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	vand.vx	v15, v15, t2
	vsrl.vi	v15, v15, 2
	vor.vv	v15, v15, v9
	csrr	a1, vlenb
	li	a2, 35
	mul	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 2047
	addi	a1, a1, 417
	vl1r.v	v9, (a1)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	vand.vx	v16, v16, t2
	vsrl.vi	v16, v16, 2
	vor.vv	v16, v16, v9
	ld	a7, 424(sp)                     # 8-byte Folded Reload
	ld	a1, 416(sp)                     # 8-byte Folded Reload
	add	a7, a7, a1
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v22, v8
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v28, 0
	vmv.v.i	v8, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vx	v8, a7, v22
	lh	s1, 266(ra)
	lh	a4, 268(ra)
	lh	a1, 270(ra)
	lh	a5, 272(ra)
	ld	t1, 432(sp)                     # 8-byte Folded Reload
	add	t1, t1, s1
	vzext.vf2	v22, v10
	vwmacc.vx	v8, t1, v22
	add	t0, a1, a4
	vzext.vf2	v10, v11
	vwmacc.vx	v8, t0, v10
	lh	a1, 274(ra)
	lh	a2, 276(ra)
	lh	s0, 278(ra)
	lh	a4, 280(ra)
	add	a5, a5, a1
	vzext.vf2	v10, v12
	vwmacc.vx	v8, a5, v10
	add	s0, s0, a2
	vzext.vf2	v10, v13
	vwmacc.vx	v8, s0, v10
	lh	a1, 282(ra)
	lh	a2, 284(ra)
	lh	s1, 286(ra)
	lh	a6, 288(ra)
	add	a1, a1, a4
	vzext.vf2	v10, v14
	vwmacc.vx	v8, a1, v10
	lh	a4, 290(ra)
	add	s1, s1, a2
	vzext.vf2	v10, v15
	vwmacc.vx	v8, s1, v10
	addi	a2, s11, 32
	vle16.v	v10, (a2)
	add	a6, a6, a4
	vzext.vf2	v11, v16
	vwmacc.vx	v8, a6, v11
	vfwcvt.f.f.v	v12, v10
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v10, v12, fa5
	vfcvt.f.x.v	v8, v8
	csrr	a2, vlenb
	li	a3, 30
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl2r.v	v6, (a2)                        # Unknown-size Folded Reload
	vfnmsac.vv	v6, v8, v10
	csrr	a2, vlenb
	li	a3, 29
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	csrr	a2, vlenb
	li	a3, 23
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v13, (a2)                       # Unknown-size Folded Reload
	vand.vi	v9, v13, 12
	vsll.vi	v9, v9, 2
	vor.vv	v8, v9, v8
	csrr	a2, vlenb
	li	a3, 28
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v9, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v9, v9, 4
	csrr	a2, vlenb
	li	a3, 22
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v14, (a2)                       # Unknown-size Folded Reload
	vand.vi	v10, v14, 12
	vsll.vi	v10, v10, 2
	vor.vv	v9, v10, v9
	csrr	a2, vlenb
	li	a3, 27
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v10, (a2)                       # Unknown-size Folded Reload
	vsrl.vi	v10, v10, 4
	csrr	a2, vlenb
	slli	a3, a2, 4
	add	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v15, (a2)                       # Unknown-size Folded Reload
	vand.vi	v11, v15, 12
	vsll.vi	v11, v11, 2
	vor.vv	v10, v11, v10
	csrr	a2, vlenb
	li	a3, 26
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v11, (a2)                       # Unknown-size Folded Reload
	vsrl.vi	v11, v11, 4
	csrr	a2, vlenb
	slli	a2, a2, 4
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v16, (a2)                       # Unknown-size Folded Reload
	vand.vi	v12, v16, 12
	vsll.vi	v12, v12, 2
	vor.vv	v11, v12, v11
	csrr	a2, vlenb
	li	a3, 25
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v12, (a2)                       # Unknown-size Folded Reload
	vsrl.vi	v12, v12, 4
	vand.vx	v13, v13, t2
	vsrl.vi	v13, v13, 2
	vor.vv	v12, v13, v12
	csrr	a2, vlenb
	li	a3, 24
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v13, (a2)                       # Unknown-size Folded Reload
	vsrl.vi	v13, v13, 4
	vand.vx	v14, v14, t2
	vsrl.vi	v14, v14, 2
	vor.vv	v13, v14, v13
	csrr	a2, vlenb
	li	a3, 19
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v14, (a2)                       # Unknown-size Folded Reload
	vsrl.vi	v14, v14, 4
	vand.vx	v15, v15, t2
	vsrl.vi	v15, v15, 2
	vor.vv	v14, v15, v14
	csrr	a2, vlenb
	li	a3, 18
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v15, (a2)                       # Unknown-size Folded Reload
	vsrl.vi	v15, v15, 4
	vand.vx	v16, v16, t2
	vsrl.vi	v16, v16, 2
	vor.vv	v15, v16, v15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v16, v8
	vmv2r.v	v22, v28
	vwmacc.vx	v22, a7, v16
	vzext.vf2	v8, v9
	vwmacc.vx	v22, t1, v8
	vzext.vf2	v8, v10
	vwmacc.vx	v22, t0, v8
	vzext.vf2	v8, v11
	vwmacc.vx	v22, a5, v8
	vzext.vf2	v8, v12
	vwmacc.vx	v22, s0, v8
	vzext.vf2	v8, v13
	vwmacc.vx	v22, a1, v8
	addi	a2, s11, 48
	vzext.vf2	v8, v14
	vle16.v	v9, (a2)
	vwmacc.vx	v22, s1, v8
	vzext.vf2	v8, v15
	vwmacc.vx	v22, a6, v8
	vfwcvt.f.f.v	v10, v9
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v8, v10, fa5
	vfcvt.f.x.v	v10, v22
	csrr	a2, vlenb
	li	a3, 20
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl2r.v	v4, (a2)                        # Unknown-size Folded Reload
	vfnmsac.vv	v4, v10, v8
	csrr	a2, vlenb
	slli	a3, a2, 4
	sub	a2, a3, a2
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v8, v8, 4
	csrr	a2, vlenb
	li	a3, 10
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v11, (a2)                       # Unknown-size Folded Reload
	vand.vi	v9, v11, 12
	vsll.vi	v9, v9, 2
	vor.vv	v10, v9, v8
	csrr	a2, vlenb
	li	a3, 14
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v24, (a2)                       # Unknown-size Folded Reload
	vand.vi	v9, v24, 12
	vsll.vi	v9, v9, 2
	vor.vv	v12, v9, v8
	csrr	a2, vlenb
	li	a3, 13
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	csrr	a2, vlenb
	slli	a3, a2, 2
	add	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v27, (a2)                       # Unknown-size Folded Reload
	vand.vi	v9, v27, 12
	vsll.vi	v9, v9, 2
	vor.vv	v13, v9, v8
	csrr	a2, vlenb
	li	a3, 12
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	csrr	a2, vlenb
	slli	a2, a2, 2
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v30, (a2)                       # Unknown-size Folded Reload
	vand.vi	v9, v30, 12
	vsll.vi	v9, v9, 2
	vor.vv	v14, v9, v8
	csrr	a2, vlenb
	slli	a2, a2, 1
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vmv1r.v	v2, v21
	vand.vi	v9, v21, 12
	vsll.vi	v9, v9, 2
	vor.vv	v15, v9, v8
	csrr	a2, vlenb
	slli	a3, a2, 1
	add	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vi	v9, v25, 12
	vsll.vi	v9, v9, 2
	vor.vv	v16, v9, v8
	addi	a2, sp, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vi	v9, v26, 12
	vsll.vi	v9, v9, 2
	vor.vv	v21, v9, v8
	csrr	a2, vlenb
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vi	v9, v17, 12
	vsll.vi	v9, v9, 2
	vor.vv	v22, v9, v8
	csrr	a2, vlenb
	li	a3, 11
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vx	v9, v11, t2
	vsrl.vi	v9, v9, 2
	vor.vv	v23, v9, v8
	csrr	a2, vlenb
	slli	a3, a2, 3
	add	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vx	v9, v24, t2
	vsrl.vi	v9, v9, 2
	vor.vv	v24, v9, v8
	csrr	a2, vlenb
	li	a3, 6
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v8, (a2)                        # Unknown-size Folded Reload
	vsrl.vi	v8, v8, 4
	vand.vx	v9, v27, t2
	vsrl.vi	v9, v9, 2
	vor.vv	v27, v9, v8
	vsrl.vi	v8, v3, 4
	vand.vx	v9, v2, t2
	vsrl.vi	v11, v20, 4
	vand.vx	v20, v25, t2
	vsrl.vi	v9, v9, 2
	vsrl.vi	v20, v20, 2
	vor.vv	v25, v9, v8
	vor.vv	v20, v20, v11
	vmv2r.v	v8, v28
	vsrl.vi	v11, v19, 4
	vand.vx	v19, v26, t2
	vsrl.vi	v19, v19, 2
	vor.vv	v19, v19, v11
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v10
	vwmacc.vx	v8, a7, v11
	vmv2r.v	v10, v28
	vzext.vf2	v26, v15
	vwmacc.vx	v10, a7, v26
	vzext.vf2	v15, v12
	vwmacc.vx	v8, t1, v15
	vzext.vf2	v12, v16
	vwmacc.vx	v10, t1, v12
	vzext.vf2	v12, v13
	vwmacc.vx	v8, t0, v12
	vzext.vf2	v12, v21
	vwmacc.vx	v10, t0, v12
	vzext.vf2	v12, v14
	vwmacc.vx	v8, a5, v12
	vzext.vf2	v12, v22
	vwmacc.vx	v10, a5, v12
	ld	a5, 88(sp)                      # 8-byte Folded Reload
	ld	a3, 96(sp)                      # 8-byte Folded Reload
	vzext.vf2	v12, v23
	vwmacc.vx	v8, s0, v12
	vzext.vf2	v12, v25
	vwmacc.vx	v10, s0, v12
	vzext.vf2	v12, v24
	vwmacc.vx	v8, a1, v12
	vzext.vf2	v12, v20
	csrr	a2, vlenb
	li	a4, 42
	mul	a2, a2, a4
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl2r.v	v20, (a2)                       # Unknown-size Folded Reload
	vwmacc.vx	v10, a1, v12
	vzext.vf2	v12, v27
	vwmacc.vx	v8, s1, v12
	vzext.vf2	v12, v19
	vwmacc.vx	v10, s1, v12
	addi	a1, a0, 32
	csrr	a2, vlenb
	slli	a4, a2, 3
	sub	a2, a4, a2
	add	a2, a2, sp
	addi	a2, a2, 2047
	addi	a2, a2, 417
	vl1r.v	v12, (a2)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v12, 4
	vand.vx	v13, v30, t2
	vsrl.vi	v13, v13, 2
	vor.vv	v12, v13, v12
	vle16.v	v13, (a1)
	addi	a0, a0, 48
	vsrl.vi	v14, v18, 4
	vand.vx	v15, v17, t2
	vsrl.vi	v15, v15, 2
	vor.vv	v14, v15, v14
	vle16.v	v15, (a0)
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v16, v12
	vwmacc.vx	v8, a6, v16
	vfwcvt.f.f.v	v16, v13
	vzext.vf2	v12, v14
	vwmacc.vx	v10, a6, v12
	ld	s0, 80(sp)                      # 8-byte Folded Reload
	li	s1, 292
	vfwcvt.f.f.v	v12, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v14, v16, fa5
	vmv.v.v	v16, v4
	vfmul.vf	v12, v12, fa5
	vfcvt.f.x.v	v8, v8
	vfcvt.f.x.v	v10, v10
	vfnmsac.vv	v20, v8, v14
	vfnmsac.vv	v0, v10, v12
	vmv.v.v	v12, v6
	vmv.v.v	v18, v6
	vmv.v.v	v14, v4
	vmv.v.v	v10, v20
	mv	a4, a0
	vmv.v.v	v8, v0
	beq	a0, a5, .LBB0_7
	j	.LBB0_5
.LBB0_7:                                #   in Loop: Header=BB0_3 Depth=1
	j	.LBB0_2
.LBB0_6:
	csrr	a0, vlenb
	li	a1, 48
	mul	a0, a0, a1
	add	sp, sp, a0
	.cfi_def_cfa sp, 2032
	addi	sp, sp, 544
	.cfi_def_cfa_offset 2032
	ld	ra, 2024(sp)                    # 8-byte Folded Reload
	ld	s0, 2016(sp)                    # 8-byte Folded Reload
	ld	s1, 2008(sp)                    # 8-byte Folded Reload
	ld	s2, 2000(sp)                    # 8-byte Folded Reload
	ld	s3, 1992(sp)                    # 8-byte Folded Reload
	ld	s4, 1984(sp)                    # 8-byte Folded Reload
	ld	s5, 1976(sp)                    # 8-byte Folded Reload
	ld	s6, 1968(sp)                    # 8-byte Folded Reload
	ld	s7, 1960(sp)                    # 8-byte Folded Reload
	ld	s8, 1952(sp)                    # 8-byte Folded Reload
	ld	s9, 1944(sp)                    # 8-byte Folded Reload
	ld	s10, 1936(sp)                   # 8-byte Folded Reload
	ld	s11, 1928(sp)                   # 8-byte Folded Reload
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
	addi	sp, sp, 2032
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K, .Lfunc_end0-weft_emitc_ggml_repack_gevm_ct_q4_K_q8_K_kernel_ggml_repack_gevm_ct_q4_K_q8_K
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
