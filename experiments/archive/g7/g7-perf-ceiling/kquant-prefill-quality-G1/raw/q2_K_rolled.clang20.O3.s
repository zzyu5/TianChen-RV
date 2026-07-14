	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"tcrv_emitted_gemm_q2_K.ROLLED.inc"
	.text
	.globl	tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K # -- Begin function tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K
	.p2align	1
	.type	tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K,@function
tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K: # @tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K
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
	csrr	a7, vlenb
	slli	t0, a7, 5
	sub	a7, t0, a7
	sub	sp, sp, a7
	.cfi_escape 0x0f, 0x0e, 0x72, 0x00, 0x11, 0xf0, 0x0f, 0x22, 0x11, 0x1f, 0x92, 0xa2, 0x38, 0x00, 0x1e, 0x22 # sp + 2032 + 31 * vlenb
	.cfi_remember_state
	sd	a6, 112(sp)                     # 8-byte Folded Spill
	sd	a3, 416(sp)                     # 8-byte Folded Spill
	sd	a2, 144(sp)                     # 8-byte Folded Spill
	sd	a1, 104(sp)                     # 8-byte Folded Spill
	srli	a4, a4, 2
	sd	a0, 248(sp)                     # 8-byte Folded Spill
	sd	a4, 96(sp)                      # 8-byte Folded Spill
	beqz	a4, .LBB0_2
# %bb.1:
	li	a0, 16
	bgeu	a5, a0, .LBB0_3
.LBB0_2:
	csrr	a0, vlenb
	slli	a1, a0, 5
	sub	a0, a1, a0
	add	sp, sp, a0
	.cfi_def_cfa sp, 2032
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
.LBB0_3:
	.cfi_restore_state
	li	a2, 0
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	srli	s1, a0, 8
	srli	a5, a5, 4
	sd	a5, 136(sp)                     # 8-byte Folded Spill
	li	a4, 1168
	li	a5, 1344
	addi	t2, sp, 1800
	addi	t5, sp, 1832
	addi	a6, sp, 1864
	addi	t6, sp, 1896
	addi	ra, sp, 1560
	addi	s10, sp, 1592
	addi	t3, sp, 1624
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v0, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v7, 0
	addi	s3, sp, 1480
	addi	s5, sp, 1496
	addi	t1, sp, 1512
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	addi	a0, a1, 320
	sd	a0, 88(sp)                      # 8-byte Folded Spill
	ld	a0, 416(sp)                     # 8-byte Folded Reload
	addi	a3, a0, 272
	sd	a3, 208(sp)                     # 8-byte Folded Spill
	addi	a3, a1, 576
	sd	a3, 80(sp)                      # 8-byte Folded Spill
	addi	a3, a0, 336
	sd	a3, 200(sp)                     # 8-byte Folded Spill
	addi	a3, a1, 832
	sd	a3, 72(sp)                      # 8-byte Folded Spill
	addi	a3, a0, 915
	sd	a3, 192(sp)                     # 8-byte Folded Spill
	addi	a3, a1, 1088
	sd	a3, 64(sp)                      # 8-byte Folded Spill
	addi	a0, a0, 979
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	addi	a0, a1, 328
	sd	a0, 56(sp)                      # 8-byte Folded Spill
	addi	a0, a1, 584
	sd	a0, 48(sp)                      # 8-byte Folded Spill
	addi	a0, a1, 840
	sd	a0, 40(sp)                      # 8-byte Folded Spill
	addi	a0, a1, 1096
	sd	a0, 32(sp)                      # 8-byte Folded Spill
	addi	a7, sp, 1528
	mul	a1, s1, a4
	sd	s1, 368(sp)                     # 8-byte Folded Spill
	mul	a3, s1, a5
	sd	a1, 24(sp)                      # 8-byte Folded Spill
	sd	a3, 128(sp)                     # 8-byte Folded Spill
	j	.LBB0_5
.LBB0_4:                                #   in Loop: Header=BB0_5 Depth=1
	ld	a2, 120(sp)                     # 8-byte Folded Reload
	addi	a2, a2, 1
	ld	a0, 208(sp)                     # 8-byte Folded Reload
	ld	a1, 24(sp)                      # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 208(sp)                     # 8-byte Folded Spill
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 200(sp)                     # 8-byte Folded Spill
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 192(sp)                     # 8-byte Folded Spill
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 184(sp)                     # 8-byte Folded Spill
	ld	a0, 96(sp)                      # 8-byte Folded Reload
	beq	a2, a0, .LBB0_2
.LBB0_5:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
                                        #       Child Loop BB0_9 Depth 3
                                        #         Child Loop BB0_10 Depth 4
                                        #         Child Loop BB0_12 Depth 4
                                        #         Child Loop BB0_14 Depth 4
                                        #         Child Loop BB0_16 Depth 4
                                        #       Child Loop BB0_20 Depth 3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
                                        #         Child Loop BB0_25 Depth 4
                                        #         Child Loop BB0_27 Depth 4
	li	s1, 0
	mul	a0, a1, a2
	sd	a2, 120(sp)                     # 8-byte Folded Spill
	slli	a1, a2, 2
	addi	a2, a0, 1043
	sd	a2, 408(sp)                     # 8-byte Folded Spill
	addi	a2, a0, 979
	sd	a2, 400(sp)                     # 8-byte Folded Spill
	addi	a2, a0, 400
	sd	a2, 392(sp)                     # 8-byte Folded Spill
	addi	a2, a0, 336
	sd	a2, 384(sp)                     # 8-byte Folded Spill
	ld	a2, 416(sp)                     # 8-byte Folded Reload
	add	a0, a0, a2
	sd	a0, 376(sp)                     # 8-byte Folded Spill
	ld	a4, 112(sp)                     # 8-byte Folded Reload
	mul	a0, a1, a4
	addi	a2, a1, 1
	addi	a5, a1, 2
	addi	a1, a1, 3
	slli	a0, a0, 2
	mul	a2, a2, a4
	mul	a5, a5, a4
	mul	a1, a1, a4
	ld	a4, 104(sp)                     # 8-byte Folded Reload
	add	a0, a0, a4
	sd	a0, 176(sp)                     # 8-byte Folded Spill
	slli	a2, a2, 2
	slli	a5, a5, 2
	slli	a1, a1, 2
	add	a2, a2, a4
	sd	a2, 168(sp)                     # 8-byte Folded Spill
	add	a5, a5, a4
	sd	a5, 160(sp)                     # 8-byte Folded Spill
	add	a1, a1, a4
	sd	a1, 152(sp)                     # 8-byte Folded Spill
	ld	a0, 32(sp)                      # 8-byte Folded Reload
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	ld	a0, 40(sp)                      # 8-byte Folded Reload
	sd	a0, 304(sp)                     # 8-byte Folded Spill
	ld	a0, 48(sp)                      # 8-byte Folded Reload
	sd	a0, 296(sp)                     # 8-byte Folded Spill
	ld	a0, 56(sp)                      # 8-byte Folded Reload
	sd	a0, 288(sp)                     # 8-byte Folded Spill
	ld	a0, 64(sp)                      # 8-byte Folded Reload
	sd	a0, 280(sp)                     # 8-byte Folded Spill
	ld	a0, 72(sp)                      # 8-byte Folded Reload
	sd	a0, 272(sp)                     # 8-byte Folded Spill
	ld	a0, 80(sp)                      # 8-byte Folded Reload
	sd	a0, 264(sp)                     # 8-byte Folded Spill
	ld	a0, 88(sp)                      # 8-byte Folded Reload
	sd	a0, 256(sp)                     # 8-byte Folded Spill
	j	.LBB0_7
.LBB0_6:                                #   in Loop: Header=BB0_7 Depth=2
	ld	a0, 240(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 32
	ld	a1, 232(sp)                     # 8-byte Folded Reload
	addi	a1, a1, 32
	ld	a2, 224(sp)                     # 8-byte Folded Reload
	addi	a2, a2, 32
	ld	a3, 216(sp)                     # 8-byte Folded Reload
	addi	a4, a3, 32
	ld	s1, 320(sp)                     # 8-byte Folded Reload
	addi	s1, s1, 1
	ld	a3, 128(sp)                     # 8-byte Folded Reload
	ld	a5, 256(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 256(sp)                     # 8-byte Folded Spill
	ld	a5, 264(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 264(sp)                     # 8-byte Folded Spill
	ld	a5, 272(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 272(sp)                     # 8-byte Folded Spill
	ld	a5, 280(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 280(sp)                     # 8-byte Folded Spill
	ld	a5, 288(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 288(sp)                     # 8-byte Folded Spill
	ld	a5, 296(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 296(sp)                     # 8-byte Folded Spill
	ld	a5, 304(sp)                     # 8-byte Folded Reload
	add	a5, a5, a3
	sd	a5, 304(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vse32.v	v16, (a0)
	vse32.v	v22, (a1)
	vse32.v	v18, (a2)
	vse32.v	v20, (a4)
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	add	a0, a0, a3
	sd	a0, 312(sp)                     # 8-byte Folded Spill
	ld	a0, 136(sp)                     # 8-byte Folded Reload
	mv	t6, s11
	mv	t5, t0
	beq	s1, a0, .LBB0_4
.LBB0_7:                                #   Parent Loop BB0_5 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_9 Depth 3
                                        #         Child Loop BB0_10 Depth 4
                                        #         Child Loop BB0_12 Depth 4
                                        #         Child Loop BB0_14 Depth 4
                                        #         Child Loop BB0_16 Depth 4
                                        #       Child Loop BB0_20 Depth 3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
                                        #         Child Loop BB0_25 Depth 4
                                        #         Child Loop BB0_27 Depth 4
	sd	s1, 320(sp)                     # 8-byte Folded Spill
	mul	a0, a3, s1
	ld	a1, 144(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	sd	a0, 424(sp)                     # 8-byte Folded Spill
	vsetivli	zero, 1, e8, m1, ta, ma
	vmv2r.v	v18, v0
	vmv2r.v	v8, v0
	vmv2r.v	v24, v0
	vmv2r.v	v12, v0
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	li	a1, 255
	bltu	a1, a0, .LBB0_8
	j	.LBB0_18
.LBB0_8:                                #   in Loop: Header=BB0_7 Depth=2
	li	a1, 0
	vsetivli	zero, 8, e32, m2, ta, ma
	vmv.v.i	v20, 0
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	ld	a0, 280(sp)                     # 8-byte Folded Reload
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	ld	a0, 272(sp)                     # 8-byte Folded Reload
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	ld	a0, 264(sp)                     # 8-byte Folded Reload
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	ld	a2, 208(sp)                     # 8-byte Folded Reload
	ld	a3, 256(sp)                     # 8-byte Folded Reload
	vmv.v.i	v8, 0
	vmv.v.i	v16, 0
	vmv.v.i	v14, 0
	vmv.v.i	v22, 0
	vmv.v.i	v10, 0
.LBB0_9:                                #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_10 Depth 4
                                        #         Child Loop BB0_12 Depth 4
                                        #         Child Loop BB0_14 Depth 4
                                        #         Child Loop BB0_16 Depth 4
	sd	a2, 448(sp)                     # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a2, 21
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v16, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 23
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 25
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v10, (a0)                       # Unknown-size Folded Spill
	li	a0, 1168
	mul	s9, a1, a0
	sd	s9, 360(sp)                     # 8-byte Folded Spill
	li	a0, 1344
	sd	a1, 432(sp)                     # 8-byte Folded Spill
	mul	a1, a1, a0
	vmv2r.v	v18, v20
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	add	a5, a0, a1
	ld	a0, 376(sp)                     # 8-byte Folded Reload
	add	s9, s9, a0
	addi	a0, a5, 64
	addi	a1, a5, 80
	addi	a2, a5, 96
	addi	a4, a5, 112
	addi	s1, a5, 128
	vle8.v	v16, (a0)
	addi	a0, a5, 144
	vle8.v	v15, (a1)
	addi	a1, a5, 160
	vle8.v	v14, (a2)
	sd	a5, 440(sp)                     # 8-byte Folded Spill
	addi	a2, a5, 176
	vle8.v	v13, (a4)
	vle8.v	v12, (s1)
	vle8.v	v11, (a0)
	vle8.v	v9, (a1)
	vle8.v	v8, (a2)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v16, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	a1, sp, 1656
	vse16.v	v17, (a1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v15, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	s11, sp, 1672
	vse16.v	v17, (s11)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v14, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	s1, sp, 1688
	vse16.v	v17, (s1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v13, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	s0, sp, 1704
	vse16.v	v17, (s0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v12, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	mv	a5, t3
	addi	t3, sp, 1720
	vse16.v	v17, (t3)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v11, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	t4, sp, 1736
	vse16.v	v17, (t4)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v9, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	s7, sp, 1752
	vse16.v	v17, (s7)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v10, v8, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v10
	addi	s8, sp, 1768
	vse16.v	v17, (s8)
	vle16.v	v10, (a1)
	lh	a0, 1040(s9)
	lh	a1, 1042(s9)
	lh	a2, 1044(s9)
	lh	a4, 1046(s9)
	vwmacc.vx	v18, a0, v10
	vse32.v	v18, (a7)
	vmv2r.v	v18, v20
	vwmacc.vx	v18, a1, v10
	vse32.v	v18, (ra)
	vmv2r.v	v18, v20
	vwmacc.vx	v18, a2, v10
	vse32.v	v18, (s10)
	vmv2r.v	v18, v20
	vwmacc.vx	v18, a4, v10
	vle16.v	v10, (s11)
	vse32.v	v18, (a5)
	vle32.v	v18, (a7)
	lh	a0, 1048(s9)
	lh	a1, 1050(s9)
	lh	a2, 1052(s9)
	lh	a4, 1054(s9)
	vwmacc.vx	v18, a0, v10
	vse32.v	v18, (a7)
	vle32.v	v18, (ra)
	vwmacc.vx	v18, a1, v10
	vse32.v	v18, (ra)
	vle32.v	v18, (s10)
	vwmacc.vx	v18, a2, v10
	vse32.v	v18, (s10)
	vle32.v	v18, (a5)
	vwmacc.vx	v18, a4, v10
	vle16.v	v10, (s1)
	vse32.v	v18, (a5)
	vle32.v	v18, (a7)
	lh	a0, 1056(s9)
	lh	a1, 1058(s9)
	lh	a2, 1060(s9)
	lh	a4, 1062(s9)
	vwmacc.vx	v18, a0, v10
	vse32.v	v18, (a7)
	vle32.v	v18, (ra)
	vwmacc.vx	v18, a1, v10
	vse32.v	v18, (ra)
	vle32.v	v18, (s10)
	vwmacc.vx	v18, a2, v10
	vse32.v	v18, (s10)
	vle32.v	v18, (a5)
	vwmacc.vx	v18, a4, v10
	vle16.v	v10, (s0)
	vse32.v	v18, (a5)
	vle32.v	v18, (a7)
	lh	a0, 1064(s9)
	lh	a1, 1066(s9)
	lh	a2, 1068(s9)
	lh	a4, 1070(s9)
	vwmacc.vx	v18, a0, v10
	vse32.v	v18, (a7)
	vle32.v	v18, (ra)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v16, v16, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v17, v16
	addi	a0, sp, 1784
	vse16.v	v17, (a0)
	vwmacc.vx	v18, a1, v10
	vse32.v	v18, (ra)
	vle32.v	v16, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v15, v15, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v18, v15
	vse16.v	v18, (t2)
	vwmacc.vx	v16, a2, v10
	vse32.v	v16, (s10)
	vle32.v	v16, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v14, v14, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	addi	a0, sp, 1816
	vse16.v	v15, (a0)
	vwmacc.vx	v16, a4, v10
	vle16.v	v10, (t3)
	vse32.v	v16, (a5)
	vle32.v	v14, (a7)
	lh	a0, 1072(s9)
	lh	a1, 1074(s9)
	lh	a2, 1076(s9)
	lh	a4, 1078(s9)
	vwmacc.vx	v14, a0, v10
	vse32.v	v14, (a7)
	vle32.v	v14, (ra)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v13, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v16, v13
	vse16.v	v16, (t5)
	vwmacc.vx	v14, a1, v10
	vse32.v	v14, (ra)
	vle32.v	v14, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v12, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v13, v12
	addi	a0, sp, 1848
	vse16.v	v13, (a0)
	vwmacc.vx	v14, a2, v10
	vse32.v	v14, (s10)
	vle32.v	v12, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v11, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	vse16.v	v14, (a6)
	vwmacc.vx	v12, a4, v10
	vle16.v	v11, (t4)
	vse32.v	v12, (a5)
	vle32.v	v12, (a7)
	lh	a2, 1080(s9)
	lh	a4, 1082(s9)
	lh	a1, 1084(s9)
	lh	a0, 1086(s9)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v9, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v9
	vwmacc.vx	v12, a2, v11
	vse32.v	v12, (a7)
	vle32.v	v12, (ra)
	addi	a2, sp, 1880
	vse16.v	v10, (a2)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v8, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v10, v8
	vwmacc.vx	v12, a4, v11
	vse32.v	v12, (ra)
	vle32.v	v8, (s10)
	vse16.v	v10, (t6)
	addi	a2, sp, 1272
	vse16.v	v7, (a2)
	mv	s11, a3
	addi	a3, sp, 1528
	addi	a7, sp, 1304
	mv	t6, s3
	addi	s3, sp, 1288
	vse16.v	v7, (s3)
	vwmacc.vx	v8, a1, v11
	vse32.v	v8, (s10)
	vle32.v	v8, (a5)
	vse16.v	v7, (a7)
	addi	t0, sp, 1320
	vse16.v	v7, (t0)
	addi	s4, sp, 1352
	addi	a6, sp, 1336
	vse16.v	v7, (a6)
	vwmacc.vx	v8, a0, v11
	vle16.v	v10, (s7)
	vse32.v	v8, (a5)
	vle32.v	v8, (a3)
	lh	a0, 1088(s9)
	lh	a1, 1090(s9)
	lh	a2, 1092(s9)
	lh	a4, 1094(s9)
	vwmacc.vx	v8, a0, v10
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vse16.v	v7, (s4)
	addi	t2, sp, 1384
	mv	s2, t1
	addi	t1, sp, 1368
	vse16.v	v7, (t1)
	vse16.v	v7, (t2)
	vwmacc.vx	v8, a1, v10
	vse32.v	v8, (ra)
	vle32.v	v8, (s10)
	addi	s6, sp, 1416
	mv	s3, s5
	addi	s5, sp, 1400
	vse16.v	v7, (s5)
	vse16.v	v7, (s6)
	addi	t4, sp, 1432
	vse16.v	v7, (t4)
	vwmacc.vx	v8, a2, v10
	vse32.v	v8, (s10)
	vle32.v	v8, (a5)
	addi	t5, sp, 1448
	vse16.v	v7, (t5)
	addi	t3, sp, 1464
	vse16.v	v7, (t3)
	vse16.v	v7, (t6)
	vwmacc.vx	v8, a4, v10
	vle16.v	v10, (s8)
	vse32.v	v8, (a5)
	vle32.v	v8, (a3)
	lh	a0, 1096(s9)
	lh	a1, 1098(s9)
	lh	a2, 1100(s9)
	lh	s7, 1102(s9)
	vwmacc.vx	v8, a0, v10
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vse16.v	v7, (s3)
	ld	a3, 408(sp)                     # 8-byte Folded Reload
	ld	s1, 360(sp)                     # 8-byte Folded Reload
	add	a3, a3, s1
	ld	a0, 400(sp)                     # 8-byte Folded Reload
	add	a0, a0, s1
	vwmacc.vx	v8, a1, v10
	ld	a4, 392(sp)                     # 8-byte Folded Reload
	add	a4, a4, s1
	ld	s0, 384(sp)                     # 8-byte Folded Reload
	add	s0, s0, s1
	ld	t5, 416(sp)                     # 8-byte Folded Reload
	add	a3, a3, t5
	sd	a3, 352(sp)                     # 8-byte Folded Spill
	vse32.v	v8, (ra)
	vle32.v	v8, (s10)
	add	a0, a0, t5
	sd	a0, 336(sp)                     # 8-byte Folded Spill
	add	a4, a4, t5
	sd	a4, 328(sp)                     # 8-byte Folded Spill
	add	t5, t5, s0
	vwmacc.vx	v8, a2, v10
	vse32.v	v8, (s10)
	vle32.v	v8, (a5)
	flw	fa2, 0(s9)
	flw	fa5, 4(s9)
	flw	fa3, 8(s9)
	sd	s9, 344(sp)                     # 8-byte Folded Spill
	flw	fa4, 12(s9)
	vwmacc.vx	v8, s7, v10
	vse32.v	v8, (a5)
	vse16.v	v7, (s2)
	ld	a1, 448(sp)                     # 8-byte Folded Reload
	sd	s11, 360(sp)                    # 8-byte Folded Spill
	mv	a2, s11
.LBB0_10:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a2)
	lbu	a0, -256(a1)
	addi	a3, sp, 1272
	vle16.v	v10, (a3)
	vand.vi	v11, v8, 3
	vsrl.vi	v9, v8, 2
	vsrl.vi	v13, v8, 4
	vsrl.vi	v8, v8, 6
	vand.vi	v12, v9, 3
	vand.vi	v9, v13, 3
	vand.vi	v8, v8, 3
	vwmacc.vx	v10, a0, v11
	addi	t3, sp, 1272
	vse16.v	v10, (a3)
	addi	a0, sp, 1288
	vle16.v	v10, (a0)
	lbu	s7, -128(a1)
	lbu	s8, -127(a1)
	lbu	s9, -126(a1)
	lbu	s10, -125(a1)
	vwmacc.vx	v10, s7, v12
	vse16.v	v10, (a0)
	vle16.v	v10, (a7)
	lbu	a4, 0(a1)
	lbu	a5, 1(a1)
	lbu	ra, 2(a1)
	lbu	s7, 3(a1)
	vwmacc.vx	v10, a4, v9
	vse16.v	v10, (a7)
	vle16.v	v10, (t0)
	lbu	a4, 128(a1)
	lbu	s1, 129(a1)
	lbu	a3, 130(a1)
	lbu	s11, 131(a1)
	vwmacc.vx	v10, a4, v8
	vse16.v	v10, (t0)
	vle16.v	v10, (a6)
	lbu	a4, -255(a1)
	lbu	a0, -254(a1)
	lbu	s0, -253(a1)
	vwmacc.vx	v10, a4, v11
	vse16.v	v10, (a6)
	vle16.v	v10, (s4)
	vwmacc.vx	v10, s8, v12
	vse16.v	v10, (s4)
	vle16.v	v10, (t1)
	vwmacc.vx	v10, a5, v9
	vse16.v	v10, (t1)
	vle16.v	v10, (t2)
	vwmacc.vx	v10, s1, v8
	vse16.v	v10, (t2)
	vle16.v	v10, (s5)
	vwmacc.vx	v10, a0, v11
	vse16.v	v10, (s5)
	vle16.v	v10, (s6)
	vwmacc.vx	v10, s9, v12
	vse16.v	v10, (s6)
	vle16.v	v10, (t4)
	vwmacc.vx	v10, ra, v9
	vse16.v	v10, (t4)
	addi	a0, sp, 1448
	vle16.v	v10, (a0)
	vwmacc.vx	v10, a3, v8
	vse16.v	v10, (a0)
	addi	a0, sp, 1464
	vle16.v	v10, (a0)
	vwmacc.vx	v10, s0, v11
	vse16.v	v10, (a0)
	vle16.v	v10, (t6)
	vwmacc.vx	v10, s10, v12
	vse16.v	v10, (t6)
	vle16.v	v10, (s3)
	vwmacc.vx	v10, s7, v9
	vse16.v	v10, (s3)
	vle16.v	v9, (s2)
	addi	a1, a1, 4
	vwmacc.vx	v9, s11, v8
	vse16.v	v9, (s2)
	addi	a2, a2, 16
	bne	a1, t5, .LBB0_10
# %bb.11:                               #   in Loop: Header=BB0_9 Depth=3
	csrr	a0, vlenb
	li	a1, 19
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v22, (a0)                       # Unknown-size Folded Spill
	addi	a0, sp, 1784
	vle16.v	v19, (a0)
	vle16.v	v28, (t3)
	vle16.v	v8, (a6)
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v4, (s5)
	addi	a0, sp, 1464
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	slli	a1, a0, 4
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1816
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1288
	vle16.v	v27, (a0)
	vle16.v	v8, (s4)
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v2, (s6)
	vle16.v	v8, (t6)
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1848
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v1, (a7)
	vle16.v	v8, (t1)
	csrr	a0, vlenb
	li	a1, 12
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v29, (t4)
	vle16.v	v8, (s3)
	csrr	a0, vlenb
	slli	a1, a0, 4
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1880
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 28
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v24, (t0)
	vle16.v	v8, (t2)
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1448
	vle16.v	v26, (a0)
	vle16.v	v8, (s2)
	csrr	a0, vlenb
	li	a1, 14
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	s11, sp, 1016
	vse16.v	v7, (s11)
	addi	a6, sp, 1032
	vse16.v	v7, (a6)
	addi	t0, sp, 1048
	vse16.v	v7, (t0)
	addi	a7, sp, 1064
	vse16.v	v7, (a7)
	addi	t3, sp, 1080
	vse16.v	v7, (t3)
	addi	t4, sp, 1096
	vse16.v	v7, (t4)
	addi	s5, sp, 1112
	vse16.v	v7, (s5)
	addi	t2, sp, 1128
	vse16.v	v7, (t2)
	addi	t1, sp, 1144
	vse16.v	v7, (t1)
	addi	t5, sp, 1160
	vse16.v	v7, (t5)
	addi	a1, sp, 1176
	vse16.v	v7, (a1)
	addi	a1, sp, 1192
	vse16.v	v7, (a1)
	addi	ra, sp, 1208
	vse16.v	v7, (ra)
	addi	s6, sp, 1224
	vse16.v	v7, (s6)
	addi	t6, sp, 1240
	vse16.v	v7, (t6)
	addi	s4, sp, 1256
	vse16.v	v7, (s4)
	ld	a1, 464(sp)                     # 8-byte Folded Reload
	ld	a2, 456(sp)                     # 8-byte Folded Reload
	ld	s3, 328(sp)                     # 8-byte Folded Reload
.LBB0_12:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vle8.v	v8, (a2)
	lbu	a0, -256(a1)
	vle16.v	v10, (s11)
	vand.vi	v11, v8, 3
	vsrl.vi	v9, v8, 2
	vsrl.vi	v13, v8, 4
	vsrl.vi	v8, v8, 6
	vand.vi	v12, v9, 3
	vand.vi	v9, v13, 3
	vand.vi	v8, v8, 3
	vwmacc.vx	v10, a0, v11
	vse16.v	v10, (s11)
	vle16.v	v10, (a6)
	lbu	s1, -128(a1)
	lbu	s2, -127(a1)
	lbu	s7, -126(a1)
	lbu	s8, -125(a1)
	vwmacc.vx	v10, s1, v12
	vse16.v	v10, (a6)
	vle16.v	v10, (t0)
	lbu	s1, 0(a1)
	lbu	a5, 1(a1)
	lbu	s11, 2(a1)
	lbu	s9, 3(a1)
	vwmacc.vx	v10, s1, v9
	vse16.v	v10, (t0)
	vle16.v	v10, (a7)
	lbu	s1, 128(a1)
	lbu	s0, 129(a1)
	lbu	a3, 130(a1)
	lbu	s10, 131(a1)
	vwmacc.vx	v10, s1, v8
	vse16.v	v10, (a7)
	vle16.v	v10, (t3)
	lbu	s1, -255(a1)
	lbu	a0, -254(a1)
	lbu	a4, -253(a1)
	vwmacc.vx	v10, s1, v11
	vse16.v	v10, (t3)
	vle16.v	v10, (t4)
	vwmacc.vx	v10, s2, v12
	vse16.v	v10, (t4)
	vle16.v	v10, (s5)
	vwmacc.vx	v10, a5, v9
	vse16.v	v10, (s5)
	vle16.v	v10, (t2)
	vwmacc.vx	v10, s0, v8
	vse16.v	v10, (t2)
	vle16.v	v10, (t1)
	vwmacc.vx	v10, a0, v11
	vse16.v	v10, (t1)
	vle16.v	v10, (t5)
	vwmacc.vx	v10, s7, v12
	vse16.v	v10, (t5)
	addi	a0, sp, 1176
	vle16.v	v10, (a0)
	vwmacc.vx	v10, s11, v9
	addi	s11, sp, 1016
	vse16.v	v10, (a0)
	addi	a0, sp, 1192
	vle16.v	v10, (a0)
	vwmacc.vx	v10, a3, v8
	vse16.v	v10, (a0)
	vle16.v	v10, (ra)
	vwmacc.vx	v10, a4, v11
	vse16.v	v10, (ra)
	vle16.v	v10, (s6)
	vwmacc.vx	v10, s8, v12
	vse16.v	v10, (s6)
	vle16.v	v10, (t6)
	vwmacc.vx	v10, s9, v9
	vse16.v	v10, (t6)
	vle16.v	v9, (s4)
	addi	a1, a1, 4
	vwmacc.vx	v9, s10, v8
	vse16.v	v9, (s4)
	addi	a2, a2, 16
	bne	a1, s3, .LBB0_12
# %bb.13:                               #   in Loop: Header=BB0_9 Depth=3
	ld	a3, 440(sp)                     # 8-byte Folded Reload
	addi	a0, a3, 192
	addi	a1, a3, 208
	addi	a2, a3, 224
	addi	a4, a3, 240
	vle8.v	v11, (a0)
	addi	a0, a3, 256
	vle8.v	v23, (a1)
	addi	a1, a3, 272
	vle8.v	v25, (a2)
	addi	a2, a3, 288
	vle8.v	v30, (a4)
	addi	a4, a3, 304
	addi	a3, sp, 1528
	vle32.v	v8, (a3)
	vle8.v	v22, (a0)
	vle8.v	v10, (a1)
	vle8.v	v5, (a2)
	vle8.v	v31, (a4)
	vsrl.vi	v12, v11, 4
	vsrl.vi	v13, v23, 4
	vsrl.vi	v14, v25, 4
	vsrl.vi	v15, v30, 4
	vsrl.vi	v16, v22, 4
	vsrl.vi	v17, v10, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v18, v12
	addi	a1, sp, 1656
	vse16.v	v18, (a1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v12, v5, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v18, v13
	addi	s10, sp, 1672
	vse16.v	v18, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v13, v31, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v18, v14
	vzext.vf2	v14, v15
	vzext.vf2	v15, v16
	vzext.vf2	v16, v17
	vzext.vf2	v17, v12
	vzext.vf2	v12, v13
	addi	a5, sp, 1688
	vse16.v	v18, (a5)
	addi	s9, sp, 1704
	vse16.v	v14, (s9)
	addi	s8, sp, 1720
	vse16.v	v15, (s8)
	mv	s3, t6
	addi	t6, sp, 1736
	vse16.v	v16, (t6)
	addi	s2, sp, 1752
	vse16.v	v17, (s2)
	addi	s7, sp, 1768
	vse16.v	v12, (s7)
	vle16.v	v12, (a1)
	ld	a0, 344(sp)                     # 8-byte Folded Reload
	lh	s0, 1104(a0)
	lh	a1, 1106(a0)
	lh	a2, 1108(a0)
	lh	a4, 1110(a0)
	vwmacc.vx	v8, s0, v12
	vse32.v	v8, (a3)
	mv	s4, ra
	addi	ra, sp, 1560
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v12
	vse32.v	v8, (ra)
	addi	s0, sp, 1592
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v12
	vse32.v	v8, (s0)
	addi	s1, sp, 1624
	vle32.v	v8, (s1)
	vle16.v	v13, (s10)
	vwmacc.vx	v8, a4, v12
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	s10, 1112(a0)
	lh	a1, 1114(a0)
	lh	a2, 1116(a0)
	lh	a4, 1118(a0)
	vwmacc.vx	v8, s10, v13
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v13
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v13
	vse32.v	v8, (s0)
	vle32.v	v8, (s1)
	vle16.v	v12, (a5)
	vwmacc.vx	v8, a4, v13
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	a5, 1120(a0)
	lh	a1, 1122(a0)
	lh	a2, 1124(a0)
	lh	a4, 1126(a0)
	vwmacc.vx	v8, a5, v12
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v12
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v12
	vse32.v	v8, (s0)
	vle32.v	v8, (s1)
	vle16.v	v13, (s9)
	vwmacc.vx	v8, a4, v12
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	a5, 1128(a0)
	lh	a1, 1130(a0)
	lh	a2, 1132(a0)
	lh	a4, 1134(a0)
	vwmacc.vx	v8, a5, v13
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v13
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v13
	vse32.v	v8, (s0)
	vle32.v	v8, (s1)
	vle16.v	v12, (s8)
	vwmacc.vx	v8, a4, v13
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	a5, 1136(a0)
	lh	a1, 1138(a0)
	lh	a2, 1140(a0)
	lh	a4, 1142(a0)
	vwmacc.vx	v8, a5, v12
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v12
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v12
	vse32.v	v8, (s0)
	vle32.v	v8, (s1)
	vle16.v	v13, (t6)
	vwmacc.vx	v8, a4, v12
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	a5, 1144(a0)
	lh	a1, 1146(a0)
	lh	a2, 1148(a0)
	lh	a4, 1150(a0)
	vwmacc.vx	v8, a5, v13
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v13
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v13
	vse32.v	v8, (s0)
	vle32.v	v8, (s1)
	vle16.v	v12, (s2)
	vwmacc.vx	v8, a4, v13
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	a5, 1152(a0)
	lh	a1, 1154(a0)
	lh	a2, 1156(a0)
	lh	a4, 1158(a0)
	vwmacc.vx	v8, a5, v12
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v12
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v12
	vse32.v	v8, (s0)
	vle32.v	v8, (s1)
	vle16.v	v14, (s7)
	vwmacc.vx	v8, a4, v12
	vse32.v	v8, (s1)
	vle32.v	v8, (a3)
	lh	a5, 1160(a0)
	lh	a1, 1162(a0)
	lh	a2, 1164(a0)
	lh	a4, 1166(a0)
	vwmacc.vx	v8, a5, v14
	vse32.v	v8, (a3)
	vle32.v	v8, (ra)
	vwmacc.vx	v8, a1, v14
	vse32.v	v8, (ra)
	vle32.v	v8, (s0)
	vwmacc.vx	v8, a2, v14
	vse32.v	v8, (s0)
	vle32.v	v16, (s1)
	vmv2r.v	v8, v20
	vmv2r.v	v12, v20
	addi	a0, sp, 1800
	vle16.v	v18, (a0)
	vwmacc.vx	v16, a4, v14
	vle16.v	v3, (s11)
	vse32.v	v16, (s1)
	vle16.v	v14, (t3)
	csrr	a1, vlenb
	slli	a2, a1, 2
	add	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 1920
	vs1r.v	v14, (a1)                       # Unknown-size Folded Spill
	vle16.v	v0, (t1)
	vwmacc.vv	v8, v19, v28
	vle16.v	v14, (s4)
	csrr	a1, vlenb
	slli	a2, a1, 3
	add	a1, a1, a2
	add	a1, a1, sp
	addi	a1, a1, 1920
	vs1r.v	v14, (a1)                       # Unknown-size Folded Spill
	vwmacc.vv	v12, v19, v4
	addi	a1, sp, 1832
	vle16.v	v6, (a1)
	csrr	a2, vlenb
	li	a3, 30
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 1920
	vl1r.v	v14, (a2)                       # Unknown-size Folded Reload
	vwmacc.vv	v8, v14, v27
	vle16.v	v27, (a6)
	vwmacc.vv	v12, v14, v2
	vle16.v	v14, (t4)
	csrr	a2, vlenb
	slli	a2, a2, 2
	add	a2, a2, sp
	addi	a2, a2, 1920
	vs1r.v	v14, (a2)                       # Unknown-size Folded Spill
	csrr	a2, vlenb
	li	a3, 29
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 1920
	vl1r.v	v14, (a2)                       # Unknown-size Folded Reload
	vwmacc.vv	v8, v14, v1
	vle16.v	v28, (t5)
	vwmacc.vv	v12, v14, v29
	vle16.v	v14, (s6)
	csrr	a2, vlenb
	slli	a2, a2, 3
	add	a2, a2, sp
	addi	a2, a2, 1920
	vs1r.v	v14, (a2)                       # Unknown-size Folded Spill
	csrr	a2, vlenb
	li	a3, 28
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 1920
	vl1r.v	v14, (a2)                       # Unknown-size Folded Reload
	vwmacc.vv	v8, v14, v24
	addi	s10, sp, 1864
	vle16.v	v1, (s10)
	vwmacc.vv	v12, v14, v26
	addi	s11, sp, 1896
	vle16.v	v14, (s11)
	csrr	a2, vlenb
	li	a3, 27
	mul	a2, a2, a3
	add	a2, a2, sp
	addi	a2, a2, 1920
	vs1r.v	v14, (a2)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v11, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	addi	a2, sp, 1784
	vse16.v	v14, (a2)
	vle16.v	v29, (t0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v23, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	vse16.v	v14, (a0)
	vle16.v	v11, (s5)
	csrr	a0, vlenb
	slli	a2, a0, 1
	add	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v11, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v25, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	addi	a0, sp, 1816
	vse16.v	v14, (a0)
	addi	a0, sp, 1176
	vle16.v	v24, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v30, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	vse16.v	v14, (a1)
	vle16.v	v11, (s3)
	csrr	a0, vlenb
	slli	a1, a0, 3
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v11, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v22, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v11
	addi	a0, sp, 1848
	vse16.v	v14, (a0)
	vle16.v	v26, (a7)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v10, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v10
	mv	a6, s10
	vse16.v	v11, (s10)
	vle16.v	v10, (t2)
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v5, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v10
	addi	a0, sp, 1880
	vse16.v	v11, (a0)
	addi	a0, sp, 1192
	vle16.v	v5, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v10, v31, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v10
	vse16.v	v11, (s11)
	addi	a0, sp, 1256
	vle16.v	v10, (a0)
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	addi	s1, sp, 760
	vse16.v	v7, (s1)
	addi	a7, sp, 776
	vse16.v	v7, (a7)
	addi	t0, sp, 792
	vse16.v	v7, (t0)
	addi	t1, sp, 808
	vse16.v	v7, (t1)
	addi	t2, sp, 824
	vse16.v	v7, (t2)
	addi	t3, sp, 840
	vse16.v	v7, (t3)
	addi	t4, sp, 856
	vse16.v	v7, (t4)
	addi	s3, sp, 872
	vse16.v	v7, (s3)
	addi	s4, sp, 888
	vse16.v	v7, (s4)
	addi	s5, sp, 904
	vse16.v	v7, (s5)
	addi	s6, sp, 920
	vse16.v	v7, (s6)
	addi	s10, sp, 936
	vse16.v	v7, (s10)
	addi	s11, sp, 952
	vse16.v	v7, (s11)
	addi	a0, sp, 968
	vse16.v	v7, (a0)
	addi	a0, sp, 984
	vse16.v	v7, (a0)
	addi	a0, sp, 1000
	vse16.v	v7, (a0)
	ld	a4, 480(sp)                     # 8-byte Folded Reload
	ld	a1, 472(sp)                     # 8-byte Folded Reload
	ld	ra, 336(sp)                     # 8-byte Folded Reload
.LBB0_14:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v10, (a1)
	lbu	a0, -387(a4)
	vle16.v	v17, (s1)
	vand.vi	v15, v10, 3
	vsrl.vi	v11, v10, 2
	vsrl.vi	v14, v10, 4
	vsrl.vi	v10, v10, 6
	vand.vi	v16, v11, 3
	vand.vi	v14, v14, 3
	vand.vi	v11, v10, 3
	vwmacc.vx	v17, a0, v15
	vse16.v	v17, (s1)
	vle16.v	v10, (a7)
	lbu	a5, -259(a4)
	lbu	t5, -258(a4)
	lbu	t6, -257(a4)
	lbu	s2, -256(a4)
	vwmacc.vx	v10, a5, v16
	vse16.v	v10, (a7)
	vle16.v	v10, (t0)
	lbu	a5, -131(a4)
	lbu	s0, -130(a4)
	lbu	s9, -129(a4)
	lbu	s7, -128(a4)
	vwmacc.vx	v10, a5, v14
	vse16.v	v10, (t0)
	vle16.v	v10, (t1)
	lbu	a5, -3(a4)
	lbu	a2, -2(a4)
	lbu	s1, -1(a4)
	lbu	s8, 0(a4)
	vwmacc.vx	v10, a5, v11
	vse16.v	v10, (t1)
	vle16.v	v10, (t2)
	lbu	a5, -386(a4)
	lbu	a0, -385(a4)
	lbu	a3, -384(a4)
	vwmacc.vx	v10, a5, v15
	addi	a5, sp, 984
	vse16.v	v10, (t2)
	vle16.v	v10, (t3)
	vwmacc.vx	v10, t5, v16
	vse16.v	v10, (t3)
	vle16.v	v10, (t4)
	vwmacc.vx	v10, s0, v14
	addi	s0, sp, 1000
	vse16.v	v10, (t4)
	vle16.v	v10, (s3)
	vwmacc.vx	v10, a2, v11
	addi	a2, sp, 968
	vse16.v	v10, (s3)
	vle16.v	v10, (s4)
	vwmacc.vx	v10, a0, v15
	vse16.v	v10, (s4)
	vle16.v	v10, (s5)
	vwmacc.vx	v10, t6, v16
	vse16.v	v10, (s5)
	vle16.v	v10, (s6)
	vwmacc.vx	v10, s9, v14
	vse16.v	v10, (s6)
	vle16.v	v10, (s10)
	vwmacc.vx	v10, s1, v11
	addi	s1, sp, 760
	vse16.v	v10, (s10)
	vle16.v	v10, (s11)
	vwmacc.vx	v10, a3, v15
	vse16.v	v10, (s11)
	vle16.v	v10, (a2)
	vwmacc.vx	v10, s2, v16
	vse16.v	v10, (a2)
	vle16.v	v10, (a5)
	vwmacc.vx	v10, s7, v14
	vse16.v	v10, (a5)
	vle16.v	v10, (s0)
	addi	a4, a4, 4
	vwmacc.vx	v10, s8, v11
	vse16.v	v10, (s0)
	addi	a1, a1, 16
	bne	a4, ra, .LBB0_14
# %bb.15:                               #   in Loop: Header=BB0_9 Depth=3
	vmv2r.v	v30, v20
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v8, v18, v3
	vmv1r.v	v4, v18
	vwmacc.vv	v12, v18, v0
	addi	a0, sp, 1784
	vle16.v	v0, (a0)
	vle16.v	v22, (s1)
	vle16.v	v23, (t2)
	vle16.v	v25, (s4)
	vle16.v	v10, (s11)
	addi	a0, sp, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	addi	a0, sp, 1816
	vle16.v	v3, (a0)
	vle16.v	v14, (a7)
	vle16.v	v15, (t3)
	vle16.v	v16, (s5)
	vle16.v	v10, (a2)
	csrr	a0, vlenb
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 18
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v19, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v19, v10
	addi	a0, sp, 1848
	vle16.v	v11, (a0)
	vwmacc.vv	v8, v6, v27
	vle16.v	v17, (t0)
	vmv1r.v	v2, v6
	vwmacc.vv	v12, v6, v28
	vle16.v	v28, (t4)
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v18, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v10, v18
	vle16.v	v27, (s6)
	vwmacc.vv	v8, v1, v29
	vle16.v	v10, (a5)
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	vwmacc.vv	v12, v1, v24
	addi	a0, sp, 1880
	vle16.v	v6, (a0)
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 12
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v18, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v10, v18
	vle16.v	v24, (t1)
	csrr	a0, vlenb
	li	a1, 27
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v8, v10, v26
	vle16.v	v26, (s3)
	vwmacc.vv	v12, v10, v5
	vle16.v	v5, (s10)
	csrr	a0, vlenb
	li	a1, 28
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v18, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v10, v18
	vle16.v	v10, (s0)
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	addi	t4, sp, 504
	vse16.v	v7, (t4)
	addi	a7, sp, 520
	vse16.v	v7, (a7)
	addi	t0, sp, 536
	vse16.v	v7, (t0)
	addi	t1, sp, 552
	vse16.v	v7, (t1)
	addi	t2, sp, 568
	vse16.v	v7, (t2)
	addi	t3, sp, 584
	vse16.v	v7, (t3)
	addi	s3, sp, 600
	vse16.v	v7, (s3)
	addi	s4, sp, 616
	vse16.v	v7, (s4)
	addi	s5, sp, 632
	vse16.v	v7, (s5)
	addi	s6, sp, 648
	vse16.v	v7, (s6)
	addi	s9, sp, 664
	vse16.v	v7, (s9)
	addi	s10, sp, 680
	vse16.v	v7, (s10)
	addi	s11, sp, 696
	vse16.v	v7, (s11)
	addi	a0, sp, 712
	vse16.v	v7, (a0)
	addi	a0, sp, 728
	vse16.v	v7, (a0)
	addi	a0, sp, 744
	vse16.v	v7, (a0)
	ld	a3, 496(sp)                     # 8-byte Folded Reload
	ld	a1, 488(sp)                     # 8-byte Folded Reload
	ld	ra, 352(sp)                     # 8-byte Folded Reload
.LBB0_16:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_9 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v10, (a1)
	lbu	a0, -387(a3)
	vle16.v	v21, (t4)
	vand.vi	v19, v10, 3
	vsrl.vi	v18, v10, 2
	vsrl.vi	v29, v10, 4
	vsrl.vi	v10, v10, 6
	vand.vi	v20, v18, 3
	vand.vi	v18, v29, 3
	vand.vi	v29, v10, 3
	vwmacc.vx	v21, a0, v19
	vse16.v	v21, (t4)
	vle16.v	v10, (a7)
	lbu	a4, -259(a3)
	lbu	t4, -258(a3)
	lbu	t5, -257(a3)
	lbu	t6, -256(a3)
	vwmacc.vx	v10, a4, v20
	vse16.v	v10, (a7)
	vle16.v	v10, (t0)
	lbu	a4, -131(a3)
	lbu	s1, -130(a3)
	lbu	s8, -129(a3)
	lbu	s2, -128(a3)
	vwmacc.vx	v10, a4, v18
	vse16.v	v10, (t0)
	vle16.v	v10, (t1)
	lbu	a4, -3(a3)
	lbu	a2, -2(a3)
	lbu	a5, -1(a3)
	lbu	s7, 0(a3)
	vwmacc.vx	v10, a4, v29
	vse16.v	v10, (t1)
	vle16.v	v10, (t2)
	lbu	a4, -386(a3)
	lbu	a0, -385(a3)
	lbu	s0, -384(a3)
	vwmacc.vx	v10, a4, v19
	vse16.v	v10, (t2)
	vle16.v	v10, (t3)
	vwmacc.vx	v10, t4, v20
	vse16.v	v10, (t3)
	vle16.v	v10, (s3)
	vwmacc.vx	v10, s1, v18
	vse16.v	v10, (s3)
	vle16.v	v10, (s4)
	vwmacc.vx	v10, a2, v29
	vse16.v	v10, (s4)
	vle16.v	v10, (s5)
	vwmacc.vx	v10, a0, v19
	vse16.v	v10, (s5)
	vle16.v	v10, (s6)
	vwmacc.vx	v10, t5, v20
	vse16.v	v10, (s6)
	vle16.v	v10, (s9)
	vwmacc.vx	v10, s8, v18
	addi	t4, sp, 504
	addi	s1, sp, 744
	vse16.v	v10, (s9)
	vle16.v	v10, (s10)
	vwmacc.vx	v10, a5, v29
	vse16.v	v10, (s10)
	vle16.v	v10, (s11)
	vwmacc.vx	v10, s0, v19
	addi	a4, sp, 728
	addi	a2, sp, 712
	vse16.v	v10, (s11)
	vle16.v	v10, (a2)
	vwmacc.vx	v10, t6, v20
	vse16.v	v10, (a2)
	vle16.v	v10, (a4)
	vwmacc.vx	v10, s2, v18
	vse16.v	v10, (a4)
	vle16.v	v10, (s1)
	addi	a3, a3, 4
	vwmacc.vx	v10, s7, v29
	vse16.v	v10, (s1)
	addi	a1, a1, 16
	bne	a3, ra, .LBB0_16
# %bb.17:                               #   in Loop: Header=BB0_9 Depth=3
	csrr	a0, vlenb
	slli	a1, a0, 2
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v30, v4, v10
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v2, v10
	csrr	a0, vlenb
	slli	a1, a0, 1
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v1, v10
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 27
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v29, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v30, v29, v10
	vwmacc.vv	v8, v0, v22
	vwmacc.vv	v30, v0, v23
	vwmacc.vv	v12, v0, v25
	vwmacc.vv	v8, v3, v14
	vwmacc.vv	v30, v3, v15
	vwmacc.vv	v12, v3, v16
	vwmacc.vv	v8, v11, v17
	vwmacc.vv	v30, v11, v28
	vwmacc.vv	v12, v11, v27
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v6, (a0)                        # Unknown-size Folded Spill
	vwmacc.vv	v8, v6, v24
	vwmacc.vv	v30, v6, v26
	vwmacc.vv	v12, v6, v5
	addi	a1, sp, 1800
	vle16.v	v10, (a1)
	vle16.v	v15, (t4)
	vwmacc.vv	v8, v10, v15
	addi	t5, sp, 1832
	vle16.v	v6, (t5)
	vle16.v	v18, (a7)
	vle16.v	v14, (a6)
	vle16.v	v19, (t0)
	addi	t6, sp, 1896
	vle16.v	v15, (t6)
	vle16.v	v20, (t1)
	vwmacc.vv	v8, v6, v18
	ld	a0, 440(sp)                     # 8-byte Folded Reload
	vle16.v	v21, (a0)
	vwmacc.vv	v8, v14, v19
	vwmacc.vv	v8, v15, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v8, v8
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v18, v21
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v20, v18, fa2
	csrr	a3, vlenb
	li	a5, 21
	mul	a3, a3, a5
	add	a3, a3, sp
	addi	a3, a3, 1920
	vl2r.v	v16, (a3)                       # Unknown-size Folded Reload
	vfmacc.vv	v16, v20, v8
	vle16.v	v8, (s5)
	vle16.v	v9, (s6)
	vle16.v	v20, (s9)
	vle16.v	v21, (s10)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v12, v10, v8
	vmv.v.v	v5, v10
	vwmacc.vv	v12, v6, v9
	vwmacc.vv	v12, v14, v20
	vmv1r.v	v28, v1
	vmv.v.v	v1, v14
	vwmacc.vv	v12, v15, v21
	addi	a0, a0, 32
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v8, v12
	addi	a7, sp, 1528
	vle32.v	v12, (a7)
	vle16.v	v20, (a0)
	vfmul.vf	v22, v18, fa3
	csrr	a0, vlenb
	li	a3, 19
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v26, (a0)                       # Unknown-size Folded Reload
	vfmacc.vv	v26, v22, v8
	vfcvt.f.x.v	v8, v12
	addi	s10, sp, 1592
	vle32.v	v24, (s10)
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v22, v20
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v12, v22, fa2
	vfnmsub.vv	v12, v8, v16
	vfcvt.f.x.v	v20, v24
	vfmul.vf	v8, v22, fa3
	vfnmsub.vv	v8, v20, v26
	vmv.v.i	v16, 0
	vmv.v.i	v20, 0
	csrr	a0, vlenb
	li	a3, 18
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a3, a0, 4
	add	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v24, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v20, v10, v24
	csrr	a0, vlenb
	li	a3, 30
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v24, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v10, v24
	csrr	a0, vlenb
	li	a3, 29
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a3, a0, 4
	sub	a0, a3, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v24, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v10, v24
	csrr	a0, vlenb
	li	a3, 28
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a3, 14
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v24, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v10, v24
	csrr	a0, vlenb
	slli	a3, a0, 3
	add	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v4, v10
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v2, v10
	csrr	a0, vlenb
	slli	a3, a0, 3
	sub	a0, a3, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v28, v10
	vle16.v	v24, (t2)
	mv	t2, a1
	vle16.v	v10, (s11)
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v10, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v29, v10
	vle16.v	v25, (t3)
	vle16.v	v27, (a2)
	addi	a0, sp, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v0, v10
	vle16.v	v28, (s3)
	vle16.v	v0, (a4)
	vwmacc.vv	v30, v5, v24
	vmv.v.v	v14, v5
	vle16.v	v24, (s4)
	vwmacc.vv	v30, v6, v25
	vle16.v	v26, (s1)
	csrr	a0, vlenb
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v3, v10
	addi	ra, sp, 1560
	vle32.v	v4, (ra)
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v20, v11, v10
	addi	t3, sp, 1624
	vle32.v	v2, (t3)
	ld	a1, 432(sp)                     # 8-byte Folded Reload
	addi	a1, a1, 1
	ld	a3, 360(sp)                     # 8-byte Folded Reload
	addi	a3, a3, 1344
	ld	a2, 448(sp)                     # 8-byte Folded Reload
	addi	a2, a2, 1168
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1344
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1344
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v4, v4
	vfcvt.f.x.v	v2, v2
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v30, v1, v28
	vwmacc.vv	v30, v15, v24
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v28, v18, fa5
	vfmul.vf	v10, v18, fa4
	csrr	a0, vlenb
	li	a4, 13
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v18, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a4, 10
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v19, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v20, v19, v18
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v24, v22, fa5
	vfmul.vf	v18, v22, fa4
	vfcvt.f.x.v	v22, v30
	csrr	a0, vlenb
	li	a4, 30
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v30, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v20, v14, v30
	csrr	a0, vlenb
	li	a4, 23
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v30, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmacc.vv	v30, v28, v22
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v20, v6, v27
	vsetvli	zero, zero, e32, m2, ta, ma
	vfnmsub.vv	v24, v4, v30
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v20, v1, v0
	vwmacc.vv	v20, v15, v26
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v14, v20
	vmv.v.i	v20, 0
	csrr	a0, vlenb
	li	a4, 25
	mul	a0, a0, a4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v16, (a0)                       # Unknown-size Folded Reload
	vfmacc.vv	v16, v10, v14
	vfnmsub.vv	v18, v2, v16
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1344
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	vmv.v.v	v16, v12
	vmv.v.v	v14, v24
	vmv.v.v	v22, v8
	vmv.v.v	v10, v18
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	addi	s5, sp, 1496
	addi	t1, sp, 1512
	addi	s3, sp, 1480
	beq	a1, a0, .LBB0_18
	j	.LBB0_9
.LBB0_18:                               #   in Loop: Header=BB0_7 Depth=2
	ld	a0, 320(sp)                     # 8-byte Folded Reload
	slli	a0, a0, 6
	ld	a1, 176(sp)                     # 8-byte Folded Reload
	add	a2, a1, a0
	ld	a1, 168(sp)                     # 8-byte Folded Reload
	add	a3, a1, a0
	ld	a1, 160(sp)                     # 8-byte Folded Reload
	add	a4, a1, a0
	ld	a1, 152(sp)                     # 8-byte Folded Reload
	add	a0, a0, a1
	vsetivli	zero, 8, e32, m2, ta, ma
	sd	a2, 240(sp)                     # 8-byte Folded Spill
	vse32.v	v12, (a2)
	sd	a3, 232(sp)                     # 8-byte Folded Spill
	vse32.v	v24, (a3)
	sd	a4, 224(sp)                     # 8-byte Folded Spill
	vse32.v	v8, (a4)
	sd	a0, 216(sp)                     # 8-byte Folded Spill
	vse32.v	v18, (a0)
	vmv.v.i	v0, 0
	vmv.v.i	v16, 0
	vmv.v.i	v22, 0
	vmv.v.i	v18, 0
	vmv.v.i	v20, 0
	ld	a0, 248(sp)                     # 8-byte Folded Reload
	addi	a7, sp, 1528
	mv	s11, t6
	addi	a3, sp, 1336
	addi	t6, sp, 1432
	li	a1, 255
	mv	t0, t5
	bltu	a1, a0, .LBB0_19
	j	.LBB0_6
.LBB0_19:                               #   in Loop: Header=BB0_7 Depth=2
	li	a1, 0
	ld	a0, 184(sp)                     # 8-byte Folded Reload
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	ld	a0, 312(sp)                     # 8-byte Folded Reload
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	ld	a0, 192(sp)                     # 8-byte Folded Reload
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	ld	a0, 304(sp)                     # 8-byte Folded Reload
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	ld	a0, 200(sp)                     # 8-byte Folded Reload
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	ld	a0, 296(sp)                     # 8-byte Folded Reload
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	ld	a4, 208(sp)                     # 8-byte Folded Reload
	ld	a2, 288(sp)                     # 8-byte Folded Reload
	vmv2r.v	v8, v0
	vmv2r.v	v12, v0
	vmv2r.v	v10, v0
	vmv2r.v	v14, v0
.LBB0_20:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_21 Depth 4
                                        #         Child Loop BB0_23 Depth 4
                                        #         Child Loop BB0_25 Depth 4
                                        #         Child Loop BB0_27 Depth 4
	sd	a4, 448(sp)                     # 8-byte Folded Spill
	sd	a2, 440(sp)                     # 8-byte Folded Spill
	csrr	a0, vlenb
	li	a2, 19
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 21
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v12, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 23
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v10, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 25
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs2r.v	v8, (a0)                        # Unknown-size Folded Spill
	li	a0, 1168
	mul	s2, a1, a0
	li	a0, 1344
	sd	a1, 360(sp)                     # 8-byte Folded Spill
	mul	a1, a1, a0
	vsetivli	zero, 8, e8, mf2, ta, ma
	vmv2r.v	v16, v0
	ld	a0, 424(sp)                     # 8-byte Folded Reload
	add	a5, a0, a1
	ld	t4, 376(sp)                     # 8-byte Folded Reload
	add	s9, t4, s2
	addi	a0, a5, 72
	addi	a1, a5, 88
	addi	a2, a5, 104
	addi	a4, a5, 120
	addi	s1, a5, 136
	vle8.v	v20, (a0)
	addi	a0, a5, 152
	vle8.v	v15, (a1)
	addi	a1, a5, 168
	vle8.v	v14, (a2)
	sd	a5, 432(sp)                     # 8-byte Folded Spill
	addi	a2, a5, 184
	vle8.v	v13, (a4)
	vle8.v	v12, (s1)
	vle8.v	v11, (a0)
	vle8.v	v9, (a1)
	vle8.v	v8, (a2)
	vsrl.vi	v18, v20, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	a1, sp, 1656
	vse16.v	v19, (a1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v15, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	a5, sp, 1672
	vse16.v	v19, (a5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v14, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	s1, sp, 1688
	vse16.v	v19, (s1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v13, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	s0, sp, 1704
	vse16.v	v19, (s0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v12, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	mv	t4, s11
	mv	s11, t3
	addi	t3, sp, 1720
	vse16.v	v19, (t3)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v11, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	t5, sp, 1736
	vse16.v	v19, (t5)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v9, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	s7, sp, 1752
	vse16.v	v19, (s7)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v18, v8, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v18
	addi	s8, sp, 1768
	vse16.v	v19, (s8)
	vle16.v	v18, (a1)
	lh	a0, 1040(s9)
	lh	a1, 1042(s9)
	lh	a2, 1044(s9)
	lh	a4, 1046(s9)
	vwmacc.vx	v16, a0, v18
	vse32.v	v16, (a7)
	vmv2r.v	v16, v0
	vwmacc.vx	v16, a1, v18
	vse32.v	v16, (ra)
	vmv2r.v	v16, v0
	vwmacc.vx	v16, a2, v18
	vse32.v	v16, (s10)
	vmv2r.v	v16, v0
	vwmacc.vx	v16, a4, v18
	vle16.v	v18, (a5)
	vse32.v	v16, (s11)
	vle32.v	v16, (a7)
	lh	a0, 1048(s9)
	lh	a1, 1050(s9)
	lh	a2, 1052(s9)
	lh	a4, 1054(s9)
	vwmacc.vx	v16, a0, v18
	vse32.v	v16, (a7)
	vle32.v	v16, (ra)
	vwmacc.vx	v16, a1, v18
	vse32.v	v16, (ra)
	vle32.v	v16, (s10)
	vwmacc.vx	v16, a2, v18
	vse32.v	v16, (s10)
	vle32.v	v16, (s11)
	vwmacc.vx	v16, a4, v18
	vle16.v	v18, (s1)
	vse32.v	v16, (s11)
	vle32.v	v16, (a7)
	lh	a0, 1056(s9)
	lh	a1, 1058(s9)
	lh	a2, 1060(s9)
	lh	a4, 1062(s9)
	vwmacc.vx	v16, a0, v18
	vse32.v	v16, (a7)
	vle32.v	v16, (ra)
	vwmacc.vx	v16, a1, v18
	vse32.v	v16, (ra)
	vle32.v	v16, (s10)
	vwmacc.vx	v16, a2, v18
	vse32.v	v16, (s10)
	vle32.v	v16, (s11)
	vwmacc.vx	v16, a4, v18
	vle16.v	v18, (s0)
	vse32.v	v16, (s11)
	vle32.v	v16, (a7)
	lh	a0, 1064(s9)
	lh	a1, 1066(s9)
	lh	a2, 1068(s9)
	lh	a4, 1070(s9)
	vwmacc.vx	v16, a0, v18
	vse32.v	v16, (a7)
	vle32.v	v16, (ra)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v19, v20, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v20, v19
	addi	a0, sp, 1784
	vse16.v	v20, (a0)
	vwmacc.vx	v16, a1, v18
	vse32.v	v16, (ra)
	vle32.v	v16, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v15, v15, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v19, v15
	vse16.v	v19, (t2)
	vwmacc.vx	v16, a2, v18
	vse32.v	v16, (s10)
	vle32.v	v16, (s11)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v14, v14, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v14
	addi	a0, sp, 1816
	vse16.v	v15, (a0)
	vwmacc.vx	v16, a4, v18
	vle16.v	v14, (t3)
	vse32.v	v16, (s11)
	vle32.v	v16, (a7)
	lh	a0, 1072(s9)
	lh	a1, 1074(s9)
	lh	a2, 1076(s9)
	lh	a4, 1078(s9)
	vwmacc.vx	v16, a0, v14
	vse32.v	v16, (a7)
	vle32.v	v16, (ra)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v13, v13, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v13
	vse16.v	v15, (t0)
	vwmacc.vx	v16, a1, v14
	vse32.v	v16, (ra)
	vle32.v	v16, (s10)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v12, v12, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v13, v12
	addi	a0, sp, 1848
	vse16.v	v13, (a0)
	vwmacc.vx	v16, a2, v14
	vse32.v	v16, (s10)
	vle32.v	v12, (s11)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v11, v11, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v15, v11
	vse16.v	v15, (a6)
	vwmacc.vx	v12, a4, v14
	vle16.v	v11, (t5)
	vse32.v	v12, (s11)
	vle32.v	v12, (a7)
	lh	a2, 1080(s9)
	lh	a4, 1082(s9)
	lh	a1, 1084(s9)
	lh	a0, 1086(s9)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v9, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v9
	vwmacc.vx	v12, a2, v11
	vse32.v	v12, (a7)
	vle32.v	v12, (ra)
	addi	a2, sp, 1880
	vse16.v	v14, (a2)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v8, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v14, v8
	vwmacc.vx	v12, a4, v11
	vse32.v	v12, (ra)
	vle32.v	v8, (s10)
	vse16.v	v14, (t4)
	addi	t0, sp, 1272
	vse16.v	v7, (t0)
	addi	a6, sp, 1288
	vse16.v	v7, (a6)
	vwmacc.vx	v8, a1, v11
	vse32.v	v8, (s10)
	vle32.v	v8, (s11)
	addi	s4, sp, 1304
	vse16.v	v7, (s4)
	addi	a5, sp, 1320
	vse16.v	v7, (a5)
	vse16.v	v7, (a3)
	vwmacc.vx	v8, a0, v11
	vle16.v	v11, (s7)
	vse32.v	v8, (s11)
	vle32.v	v8, (a7)
	lh	a0, 1088(s9)
	lh	a1, 1090(s9)
	lh	a2, 1092(s9)
	lh	a4, 1094(s9)
	vwmacc.vx	v8, a0, v11
	vse32.v	v8, (a7)
	vle32.v	v8, (ra)
	addi	t2, sp, 1352
	vse16.v	v7, (t2)
	addi	a0, sp, 1368
	vse16.v	v7, (a0)
	addi	a0, sp, 1384
	mv	t4, a3
	mv	a3, a0
	vse16.v	v7, (a0)
	vwmacc.vx	v8, a1, v11
	vse32.v	v8, (ra)
	vle32.v	v8, (s10)
	addi	a0, sp, 1400
	vse16.v	v7, (a0)
	addi	s6, sp, 1416
	vse16.v	v7, (s6)
	vse16.v	v7, (t6)
	vwmacc.vx	v8, a2, v11
	vse32.v	v8, (s10)
	vle32.v	v8, (s11)
	addi	a0, sp, 1448
	vse16.v	v7, (a0)
	addi	a0, sp, 1464
	vse16.v	v7, (a0)
	vse16.v	v7, (s3)
	vwmacc.vx	v8, a4, v11
	vle16.v	v11, (s8)
	vse32.v	v8, (s11)
	vle32.v	v8, (a7)
	lh	a0, 1096(s9)
	lh	a1, 1098(s9)
	lh	a2, 1100(s9)
	lh	s0, 1102(s9)
	vwmacc.vx	v8, a0, v11
	vse32.v	v8, (a7)
	mv	a7, a3
	vle32.v	v8, (ra)
	vse16.v	v7, (s5)
	ld	t3, 408(sp)                     # 8-byte Folded Reload
	add	t3, t3, s2
	ld	s7, 400(sp)                     # 8-byte Folded Reload
	add	s7, s7, s2
	vwmacc.vx	v8, a1, v11
	ld	a0, 392(sp)                     # 8-byte Folded Reload
	add	a0, a0, s2
	ld	a4, 384(sp)                     # 8-byte Folded Reload
	add	a4, a4, s2
	ld	t5, 416(sp)                     # 8-byte Folded Reload
	add	t3, t3, t5
	sd	t3, 352(sp)                     # 8-byte Folded Spill
	vse32.v	v8, (ra)
	vle32.v	v8, (s10)
	add	s7, s7, t5
	sd	s7, 336(sp)                     # 8-byte Folded Spill
	mv	s7, a5
	add	a0, a0, t5
	sd	a0, 328(sp)                     # 8-byte Folded Spill
	add	t5, t5, a4
	vwmacc.vx	v8, a2, v11
	vse32.v	v8, (s10)
	vle32.v	v8, (s11)
	flw	fa2, 0(s9)
	flw	fa5, 4(s9)
	flw	fa3, 8(s9)
	sd	s9, 344(sp)                     # 8-byte Folded Spill
	flw	fa4, 12(s9)
	vwmacc.vx	v8, s0, v11
	vse32.v	v8, (s11)
	vse16.v	v7, (t1)
	ld	s0, 448(sp)                     # 8-byte Folded Reload
	ld	a1, 440(sp)                     # 8-byte Folded Reload
	mv	s8, t6
.LBB0_21:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_20 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a1)
	lbu	a0, -256(s0)
	vle16.v	v10, (t0)
	vand.vi	v11, v8, 3
	vsrl.vi	v9, v8, 2
	vsrl.vi	v13, v8, 4
	vsrl.vi	v8, v8, 6
	vand.vi	v12, v9, 3
	vand.vi	v9, v13, 3
	vand.vi	v8, v8, 3
	vwmacc.vx	v10, a0, v11
	vse16.v	v10, (t0)
	vle16.v	v10, (a6)
	lbu	s1, -128(s0)
	lbu	t6, -127(s0)
	lbu	s9, -126(s0)
	lbu	s10, -125(s0)
	vwmacc.vx	v10, s1, v12
	vse16.v	v10, (a6)
	vle16.v	v10, (s4)
	lbu	a5, 0(s0)
	lbu	s1, 1(s0)
	mv	t3, t1
	mv	t1, s5
	mv	s5, s3
	lbu	s3, 2(s0)
	lbu	s11, 3(s0)
	vwmacc.vx	v10, a5, v9
	vse16.v	v10, (s4)
	vle16.v	v10, (s7)
	lbu	a5, 128(s0)
	lbu	a2, 129(s0)
	lbu	a3, 130(s0)
	lbu	ra, 131(s0)
	vwmacc.vx	v10, a5, v8
	vse16.v	v10, (s7)
	vle16.v	v10, (t4)
	lbu	a5, -255(s0)
	lbu	a0, -254(s0)
	lbu	s2, -253(s0)
	vwmacc.vx	v10, a5, v11
	addi	a6, sp, 1288
	vse16.v	v10, (t4)
	vle16.v	v10, (t2)
	vwmacc.vx	v10, t6, v12
	addi	t6, sp, 1368
	addi	a4, sp, 1464
	addi	a5, sp, 1448
	vse16.v	v10, (t2)
	vle16.v	v10, (t6)
	vwmacc.vx	v10, s1, v9
	addi	s1, sp, 1400
	vse16.v	v10, (t6)
	vle16.v	v10, (a7)
	vwmacc.vx	v10, a2, v8
	vse16.v	v10, (a7)
	vle16.v	v10, (s1)
	vwmacc.vx	v10, a0, v11
	vse16.v	v10, (s1)
	vle16.v	v10, (s6)
	vwmacc.vx	v10, s9, v12
	vse16.v	v10, (s6)
	vle16.v	v10, (s8)
	vwmacc.vx	v10, s3, v9
	mv	s3, s5
	mv	s5, t1
	mv	t1, t3
	vse16.v	v10, (s8)
	vle16.v	v10, (a5)
	vwmacc.vx	v10, a3, v8
	vse16.v	v10, (a5)
	vle16.v	v10, (a4)
	vwmacc.vx	v10, s2, v11
	vse16.v	v10, (a4)
	vle16.v	v10, (s3)
	vwmacc.vx	v10, s10, v12
	vse16.v	v10, (s3)
	vle16.v	v10, (s5)
	vwmacc.vx	v10, s11, v9
	vse16.v	v10, (s5)
	vle16.v	v9, (t3)
	addi	s0, s0, 4
	vwmacc.vx	v9, ra, v8
	vse16.v	v9, (t3)
	addi	a1, a1, 16
	bne	s0, t5, .LBB0_21
# %bb.22:                               #   in Loop: Header=BB0_20 Depth=3
	addi	a0, sp, 1784
	vle16.v	v31, (a0)
	vle16.v	v25, (t0)
	vle16.v	v8, (t4)
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v6, (s1)
	vle16.v	v8, (a4)
	csrr	a0, vlenb
	slli	a1, a0, 4
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1816
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v5, (a6)
	vle16.v	v8, (t2)
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v24, (s6)
	vle16.v	v8, (s3)
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1848
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v3, (s4)
	vle16.v	v8, (t6)
	csrr	a0, vlenb
	li	a1, 12
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v2, (s8)
	vle16.v	v8, (s5)
	csrr	a0, vlenb
	slli	a1, a0, 4
	sub	a0, a1, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1880
	vle16.v	v8, (a0)
	csrr	a0, vlenb
	li	a1, 28
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v4, (s7)
	vle16.v	v8, (a7)
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vle16.v	v9, (a5)
	vle16.v	v8, (t1)
	csrr	a0, vlenb
	li	a1, 14
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1016
	vse16.v	v7, (a0)
	addi	a6, sp, 1032
	vse16.v	v7, (a6)
	addi	t3, sp, 1048
	vse16.v	v7, (t3)
	addi	s3, sp, 1064
	vse16.v	v7, (s3)
	addi	a7, sp, 1080
	vse16.v	v7, (a7)
	addi	t0, sp, 1096
	vse16.v	v7, (t0)
	addi	t2, sp, 1112
	vse16.v	v7, (t2)
	addi	s8, sp, 1128
	vse16.v	v7, (s8)
	addi	t1, sp, 1144
	vse16.v	v7, (t1)
	addi	s2, sp, 1160
	vse16.v	v7, (s2)
	addi	s5, sp, 1176
	vse16.v	v7, (s5)
	addi	t4, sp, 1192
	vse16.v	v7, (t4)
	addi	s7, sp, 1208
	vse16.v	v7, (s7)
	addi	s6, sp, 1224
	vse16.v	v7, (s6)
	addi	s4, sp, 1240
	vse16.v	v7, (s4)
	addi	s3, sp, 1256
	vse16.v	v7, (s3)
	ld	a1, 464(sp)                     # 8-byte Folded Reload
	ld	s4, 456(sp)                     # 8-byte Folded Reload
	ld	t3, 328(sp)                     # 8-byte Folded Reload
.LBB0_23:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_20 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vle8.v	v8, (s4)
	lbu	a0, -256(a1)
	addi	a2, sp, 1016
	vle16.v	v10, (a2)
	vand.vi	v12, v8, 3
	vsrl.vi	v11, v8, 2
	vsrl.vi	v14, v8, 4
	vsrl.vi	v8, v8, 6
	vand.vi	v13, v11, 3
	vand.vi	v11, v14, 3
	vand.vi	v8, v8, 3
	vwmacc.vx	v10, a0, v12
	addi	t2, sp, 1016
	vse16.v	v10, (a2)
	vle16.v	v10, (a6)
	lbu	s0, -128(a1)
	lbu	s9, -127(a1)
	lbu	s10, -126(a1)
	lbu	s11, -125(a1)
	vwmacc.vx	v10, s0, v13
	vse16.v	v10, (a6)
	addi	a0, sp, 1048
	vle16.v	v10, (a0)
	lbu	s0, 0(a1)
	lbu	a5, 1(a1)
	lbu	a3, 2(a1)
	lbu	ra, 3(a1)
	vwmacc.vx	v10, s0, v11
	vse16.v	v10, (a0)
	addi	a2, sp, 1064
	vle16.v	v10, (a2)
	lbu	s0, 128(a1)
	lbu	a4, 129(a1)
	lbu	s1, 130(a1)
	lbu	a0, 131(a1)
	vwmacc.vx	v10, s0, v8
	vse16.v	v10, (a2)
	vle16.v	v10, (a7)
	lbu	s0, -255(a1)
	lbu	t5, -254(a1)
	lbu	t6, -253(a1)
	vwmacc.vx	v10, s0, v12
	vse16.v	v10, (a7)
	vle16.v	v10, (t0)
	vwmacc.vx	v10, s9, v13
	vse16.v	v10, (t0)
	addi	a2, sp, 1112
	vle16.v	v10, (a2)
	vwmacc.vx	v10, a5, v11
	vse16.v	v10, (a2)
	vle16.v	v10, (s8)
	vwmacc.vx	v10, a4, v8
	vse16.v	v10, (s8)
	vle16.v	v10, (t1)
	vwmacc.vx	v10, t5, v12
	vse16.v	v10, (t1)
	vle16.v	v10, (s2)
	vwmacc.vx	v10, s10, v13
	vse16.v	v10, (s2)
	vle16.v	v10, (s5)
	vwmacc.vx	v10, a3, v11
	vse16.v	v10, (s5)
	vle16.v	v10, (t4)
	vwmacc.vx	v10, s1, v8
	vse16.v	v10, (t4)
	vle16.v	v10, (s7)
	vwmacc.vx	v10, t6, v12
	vse16.v	v10, (s7)
	vle16.v	v10, (s6)
	vwmacc.vx	v10, s11, v13
	vse16.v	v10, (s6)
	addi	a2, sp, 1240
	vle16.v	v10, (a2)
	vwmacc.vx	v10, ra, v11
	vse16.v	v10, (a2)
	vle16.v	v10, (s3)
	addi	a1, a1, 4
	vwmacc.vx	v10, a0, v8
	vse16.v	v10, (s3)
	addi	s4, s4, 16
	bne	a1, t3, .LBB0_23
# %bb.24:                               #   in Loop: Header=BB0_20 Depth=3
	ld	a4, 432(sp)                     # 8-byte Folded Reload
	addi	a0, a4, 200
	addi	a1, a4, 216
	addi	a2, a4, 232
	addi	a3, a4, 248
	vle8.v	v11, (a0)
	addi	a0, a4, 264
	vle8.v	v21, (a1)
	addi	a1, a4, 280
	vle8.v	v23, (a2)
	addi	a2, a4, 296
	vle8.v	v28, (a3)
	addi	a3, a4, 312
	mv	t3, t4
	addi	t4, sp, 1528
	vle32.v	v12, (t4)
	vle8.v	v20, (a0)
	vle8.v	v8, (a1)
	vle8.v	v10, (a2)
	vle8.v	v29, (a3)
	vsrl.vi	v14, v11, 4
	vsrl.vi	v15, v21, 4
	vsrl.vi	v16, v23, 4
	vsrl.vi	v17, v28, 4
	vsrl.vi	v18, v20, 4
	vsrl.vi	v19, v8, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v22, v14
	addi	a1, sp, 1656
	vse16.v	v22, (a1)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v14, v10, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v22, v15
	addi	s11, sp, 1672
	vse16.v	v22, (s11)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vsrl.vi	v15, v29, 4
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v22, v16
	vzext.vf2	v16, v17
	vzext.vf2	v17, v18
	vzext.vf2	v18, v19
	vzext.vf2	v19, v14
	vzext.vf2	v14, v15
	addi	s10, sp, 1688
	vse16.v	v22, (s10)
	addi	s9, sp, 1704
	vse16.v	v16, (s9)
	addi	s0, sp, 1720
	mv	ra, s8
	mv	s8, s0
	vse16.v	v17, (s0)
	addi	t5, sp, 1736
	vse16.v	v18, (t5)
	addi	t6, sp, 1752
	vse16.v	v19, (t6)
	mv	s5, s2
	addi	s2, sp, 1768
	vse16.v	v14, (s2)
	vle16.v	v14, (a1)
	ld	a4, 344(sp)                     # 8-byte Folded Reload
	lh	a0, 1104(a4)
	lh	a1, 1106(a4)
	lh	a2, 1108(a4)
	lh	a3, 1110(a4)
	vwmacc.vx	v12, a0, v14
	vse32.v	v12, (t4)
	addi	s0, sp, 1560
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v14
	vse32.v	v12, (s0)
	addi	s1, sp, 1592
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v14
	vse32.v	v12, (s1)
	addi	a5, sp, 1624
	vle32.v	v12, (a5)
	vle16.v	v15, (s11)
	vwmacc.vx	v12, a3, v14
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1112(a4)
	lh	a1, 1114(a4)
	lh	a2, 1116(a4)
	lh	a3, 1118(a4)
	vwmacc.vx	v12, a0, v15
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v15
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v15
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vle16.v	v14, (s10)
	vwmacc.vx	v12, a3, v15
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1120(a4)
	lh	a1, 1122(a4)
	lh	a2, 1124(a4)
	lh	a3, 1126(a4)
	vwmacc.vx	v12, a0, v14
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v14
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v14
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vle16.v	v15, (s9)
	vwmacc.vx	v12, a3, v14
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1128(a4)
	lh	a1, 1130(a4)
	lh	a2, 1132(a4)
	lh	a3, 1134(a4)
	vwmacc.vx	v12, a0, v15
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v15
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v15
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vle16.v	v14, (s8)
	vwmacc.vx	v12, a3, v15
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1136(a4)
	lh	a1, 1138(a4)
	lh	a2, 1140(a4)
	lh	a3, 1142(a4)
	vwmacc.vx	v12, a0, v14
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v14
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v14
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vle16.v	v15, (t5)
	vwmacc.vx	v12, a3, v14
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1144(a4)
	lh	a1, 1146(a4)
	lh	a2, 1148(a4)
	lh	a3, 1150(a4)
	vwmacc.vx	v12, a0, v15
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v15
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v15
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vle16.v	v14, (t6)
	vwmacc.vx	v12, a3, v15
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1152(a4)
	lh	a1, 1154(a4)
	lh	a2, 1156(a4)
	lh	a3, 1158(a4)
	vwmacc.vx	v12, a0, v14
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v14
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v14
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vle16.v	v15, (s2)
	vwmacc.vx	v12, a3, v14
	vse32.v	v12, (a5)
	vle32.v	v12, (t4)
	lh	a0, 1160(a4)
	lh	a1, 1162(a4)
	lh	a2, 1164(a4)
	lh	a3, 1166(a4)
	vwmacc.vx	v12, a0, v15
	vse32.v	v12, (t4)
	vle32.v	v12, (s0)
	vwmacc.vx	v12, a1, v15
	vse32.v	v12, (s0)
	vle32.v	v12, (s1)
	vwmacc.vx	v12, a2, v15
	vse32.v	v12, (s1)
	vle32.v	v12, (a5)
	vmv2r.v	v26, v0
	addi	a4, sp, 1800
	vle16.v	v14, (a4)
	vwmacc.vx	v12, a3, v15
	vle16.v	v30, (t2)
	vse32.v	v12, (a5)
	vle16.v	v12, (a7)
	csrr	a0, vlenb
	slli	a1, a0, 2
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	vle16.v	v22, (t1)
	vwmacc.vv	v26, v31, v25
	vle16.v	v12, (s7)
	csrr	a0, vlenb
	slli	a1, a0, 3
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	vwmacc.vv	v0, v31, v6
	addi	s2, sp, 1832
	vle16.v	v15, (s2)
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v12, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v12, v5
	vle16.v	v5, (a6)
	vwmacc.vv	v0, v12, v24
	vle16.v	v12, (t0)
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v12, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v12, v3
	vle16.v	v24, (s5)
	vwmacc.vv	v0, v12, v2
	vle16.v	v12, (s6)
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v12, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 28
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v12, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v12, v4
	addi	a1, sp, 1864
	vle16.v	v3, (a1)
	vwmacc.vv	v0, v12, v9
	addi	a2, sp, 1896
	vle16.v	v9, (a2)
	csrr	a0, vlenb
	li	a3, 27
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v9, (a0)                        # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v11, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v9
	addi	a0, sp, 1784
	vse16.v	v11, (a0)
	addi	a0, sp, 1048
	vle16.v	v25, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v21, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v9
	vse16.v	v11, (a4)
	addi	a0, sp, 1112
	vle16.v	v9, (a0)
	csrr	a0, vlenb
	slli	a3, a0, 1
	add	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v9, (a0)                        # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v23, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v9
	addi	a0, sp, 1816
	vse16.v	v11, (a0)
	addi	a0, sp, 1176
	vle16.v	v2, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v28, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v9
	vse16.v	v11, (s2)
	addi	a0, sp, 1240
	vle16.v	v9, (a0)
	csrr	a0, vlenb
	slli	a3, a0, 3
	sub	a0, a3, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v9, (a0)                        # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v9, v20, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v9
	addi	a0, sp, 1848
	vse16.v	v11, (a0)
	addi	a0, sp, 1064
	vle16.v	v16, (a0)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v8, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v8
	vse16.v	v11, (a1)
	vle16.v	v8, (ra)
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v10, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v8
	addi	a0, sp, 1880
	vse16.v	v11, (a0)
	vle16.v	v20, (t3)
	vsetvli	zero, zero, e8, mf2, ta, ma
	vand.vi	v8, v29, 15
	vsetvli	zero, zero, e16, m1, ta, ma
	vzext.vf2	v11, v8
	vse16.v	v11, (a2)
	vle16.v	v8, (s3)
	csrr	a0, vlenb
	li	a1, 6
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v8, (a0)                        # Unknown-size Folded Spill
	addi	s1, sp, 760
	vse16.v	v7, (s1)
	addi	a6, sp, 776
	vse16.v	v7, (a6)
	addi	a7, sp, 792
	vse16.v	v7, (a7)
	addi	t0, sp, 808
	vse16.v	v7, (t0)
	addi	t1, sp, 824
	vse16.v	v7, (t1)
	addi	t2, sp, 840
	vse16.v	v7, (t2)
	addi	t3, sp, 856
	vse16.v	v7, (t3)
	addi	t5, sp, 872
	vse16.v	v7, (t5)
	addi	s3, sp, 888
	vse16.v	v7, (s3)
	addi	s4, sp, 904
	vse16.v	v7, (s4)
	addi	s5, sp, 920
	vse16.v	v7, (s5)
	addi	s6, sp, 936
	vse16.v	v7, (s6)
	addi	s7, sp, 952
	vse16.v	v7, (s7)
	addi	a0, sp, 968
	vse16.v	v7, (a0)
	addi	a0, sp, 984
	vse16.v	v7, (a0)
	addi	a0, sp, 1000
	vse16.v	v7, (a0)
	ld	a3, 480(sp)                     # 8-byte Folded Reload
	ld	a1, 472(sp)                     # 8-byte Folded Reload
	ld	t6, 336(sp)                     # 8-byte Folded Reload
.LBB0_25:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_20 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v8, (a1)
	lbu	a0, -387(a3)
	vle16.v	v9, (s1)
	vand.vi	v12, v8, 3
	vsrl.vi	v10, v8, 2
	vsrl.vi	v11, v8, 4
	vsrl.vi	v8, v8, 6
	vand.vi	v13, v10, 3
	vand.vi	v11, v11, 3
	vand.vi	v8, v8, 3
	vwmacc.vx	v9, a0, v12
	vse16.v	v9, (s1)
	vle16.v	v9, (a6)
	lbu	a4, -259(a3)
	lbu	s8, -258(a3)
	lbu	s9, -257(a3)
	lbu	s10, -256(a3)
	vwmacc.vx	v9, a4, v13
	vse16.v	v9, (a6)
	vle16.v	v9, (a7)
	lbu	a4, -131(a3)
	lbu	s0, -130(a3)
	lbu	s1, -129(a3)
	lbu	s11, -128(a3)
	vwmacc.vx	v9, a4, v11
	vse16.v	v9, (a7)
	vle16.v	v9, (t0)
	lbu	a4, -3(a3)
	lbu	a2, -2(a3)
	lbu	a5, -1(a3)
	lbu	ra, 0(a3)
	vwmacc.vx	v9, a4, v8
	vse16.v	v9, (t0)
	vle16.v	v9, (t1)
	lbu	a4, -386(a3)
	lbu	a0, -385(a3)
	lbu	t4, -384(a3)
	vwmacc.vx	v9, a4, v12
	vse16.v	v9, (t1)
	vle16.v	v9, (t2)
	vwmacc.vx	v9, s8, v13
	vse16.v	v9, (t2)
	vle16.v	v9, (t3)
	vwmacc.vx	v9, s0, v11
	addi	s0, sp, 1000
	vse16.v	v9, (t3)
	vle16.v	v9, (t5)
	vwmacc.vx	v9, a2, v8
	addi	a2, sp, 968
	vse16.v	v9, (t5)
	vle16.v	v9, (s3)
	vwmacc.vx	v9, a0, v12
	vse16.v	v9, (s3)
	vle16.v	v9, (s4)
	vwmacc.vx	v9, s9, v13
	vse16.v	v9, (s4)
	vle16.v	v9, (s5)
	vwmacc.vx	v9, s1, v11
	addi	s1, sp, 760
	vse16.v	v9, (s5)
	vle16.v	v9, (s6)
	vwmacc.vx	v9, a5, v8
	addi	a4, sp, 984
	vse16.v	v9, (s6)
	vle16.v	v9, (s7)
	vwmacc.vx	v9, t4, v12
	vse16.v	v9, (s7)
	vle16.v	v9, (a2)
	vwmacc.vx	v9, s10, v13
	vse16.v	v9, (a2)
	vle16.v	v9, (a4)
	vwmacc.vx	v9, s11, v11
	vse16.v	v9, (a4)
	vle16.v	v9, (s0)
	addi	a3, a3, 4
	vwmacc.vx	v9, ra, v8
	vse16.v	v9, (s0)
	addi	a1, a1, 16
	bne	a3, t6, .LBB0_25
# %bb.26:                               #   in Loop: Header=BB0_20 Depth=3
	vsetvli	zero, zero, e32, m2, ta, ma
	vmv.v.i	v28, 0
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v14, v30
	vmv1r.v	v6, v14
	vwmacc.vv	v0, v14, v22
	addi	a0, sp, 1784
	vle16.v	v8, (a0)
	vle16.v	v30, (s1)
	vle16.v	v22, (t1)
	vle16.v	v21, (s3)
	vle16.v	v9, (s7)
	addi	a0, sp, 1920
	vs1r.v	v9, (a0)                        # Unknown-size Folded Spill
	addi	a0, sp, 1816
	vle16.v	v10, (a0)
	vle16.v	v23, (a6)
	vle16.v	v12, (t2)
	vle16.v	v13, (s4)
	vle16.v	v9, (a2)
	csrr	a0, vlenb
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v9, (a0)                        # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 18
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v31, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a1, 10
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v9, (a0)                        # Unknown-size Folded Reload
	vwmacc.vv	v28, v31, v9
	addi	a0, sp, 1848
	vle16.v	v11, (a0)
	vwmacc.vv	v26, v15, v5
	vle16.v	v5, (a7)
	vmv1r.v	v4, v15
	vwmacc.vv	v0, v15, v24
	vle16.v	v14, (t3)
	csrr	a0, vlenb
	li	a1, 30
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v9, (a0)                        # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v28, v9, v15
	vle16.v	v15, (s5)
	vwmacc.vv	v26, v3, v25
	vle16.v	v9, (a4)
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v9, (a0)                        # Unknown-size Folded Spill
	vwmacc.vv	v0, v3, v2
	addi	a0, sp, 1880
	vle16.v	v9, (a0)
	csrr	a0, vlenb
	li	a1, 29
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v17, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 12
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v18, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v28, v17, v18
	vle16.v	v2, (t0)
	csrr	a0, vlenb
	li	a1, 27
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v17, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v17, v16
	vle16.v	v19, (t5)
	vwmacc.vv	v0, v17, v20
	vle16.v	v20, (s6)
	csrr	a0, vlenb
	li	a1, 28
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v16, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v17, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v28, v16, v17
	vle16.v	v16, (s0)
	csrr	a0, vlenb
	li	a1, 13
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v16, (a0)                       # Unknown-size Folded Spill
	addi	s0, sp, 504
	vse16.v	v7, (s0)
	addi	a6, sp, 520
	vse16.v	v7, (a6)
	addi	a7, sp, 536
	vse16.v	v7, (a7)
	addi	t0, sp, 552
	vse16.v	v7, (t0)
	addi	t1, sp, 568
	vse16.v	v7, (t1)
	addi	t2, sp, 584
	vse16.v	v7, (t2)
	addi	t3, sp, 600
	vse16.v	v7, (t3)
	addi	t4, sp, 616
	vse16.v	v7, (t4)
	addi	t5, sp, 632
	vse16.v	v7, (t5)
	addi	s3, sp, 648
	vse16.v	v7, (s3)
	addi	s4, sp, 664
	vse16.v	v7, (s4)
	addi	s5, sp, 680
	vse16.v	v7, (s5)
	addi	s6, sp, 696
	vse16.v	v7, (s6)
	addi	a0, sp, 712
	vse16.v	v7, (a0)
	addi	a0, sp, 728
	vse16.v	v7, (a0)
	addi	a0, sp, 744
	vse16.v	v7, (a0)
	ld	a3, 496(sp)                     # 8-byte Folded Reload
	ld	a1, 488(sp)                     # 8-byte Folded Reload
	ld	t6, 352(sp)                     # 8-byte Folded Reload
.LBB0_27:                               #   Parent Loop BB0_5 Depth=1
                                        #     Parent Loop BB0_7 Depth=2
                                        #       Parent Loop BB0_20 Depth=3
                                        # =>      This Inner Loop Header: Depth=4
	vsetvli	zero, zero, e8, mf2, ta, ma
	vle8.v	v16, (a1)
	lbu	a0, -387(a3)
	vle16.v	v24, (s0)
	vand.vi	v17, v16, 3
	vsrl.vi	v18, v16, 2
	vsrl.vi	v25, v16, 4
	vsrl.vi	v31, v16, 6
	vand.vi	v18, v18, 3
	vand.vi	v16, v25, 3
	vand.vi	v25, v31, 3
	vwmacc.vx	v24, a0, v17
	vse16.v	v24, (s0)
	vle16.v	v24, (a6)
	lbu	a4, -259(a3)
	lbu	s7, -258(a3)
	lbu	a2, -257(a3)
	lbu	a5, -256(a3)
	vwmacc.vx	v24, a4, v18
	vse16.v	v24, (a6)
	vle16.v	v24, (a7)
	lbu	a4, -131(a3)
	lbu	s0, -130(a3)
	lbu	s1, -129(a3)
	lbu	a0, -128(a3)
	vwmacc.vx	v24, a4, v16
	vse16.v	v24, (a7)
	vle16.v	v24, (t0)
	lbu	a4, -3(a3)
	lbu	s8, -2(a3)
	lbu	s9, -1(a3)
	lbu	s10, 0(a3)
	vwmacc.vx	v24, a4, v25
	vse16.v	v24, (t0)
	vle16.v	v24, (t1)
	lbu	a4, -386(a3)
	lbu	s11, -385(a3)
	lbu	ra, -384(a3)
	vwmacc.vx	v24, a4, v17
	vse16.v	v24, (t1)
	vle16.v	v24, (t2)
	vwmacc.vx	v24, s7, v18
	vse16.v	v24, (t2)
	vle16.v	v24, (t3)
	vwmacc.vx	v24, s0, v16
	addi	s2, sp, 728
	addi	a4, sp, 712
	vse16.v	v24, (t3)
	vle16.v	v24, (t4)
	vwmacc.vx	v24, s8, v25
	addi	s0, sp, 504
	addi	s7, sp, 744
	vse16.v	v24, (t4)
	vle16.v	v24, (t5)
	vwmacc.vx	v24, s11, v17
	vse16.v	v24, (t5)
	vle16.v	v24, (s3)
	vwmacc.vx	v24, a2, v18
	vse16.v	v24, (s3)
	vle16.v	v24, (s4)
	vwmacc.vx	v24, s1, v16
	vse16.v	v24, (s4)
	vle16.v	v24, (s5)
	vwmacc.vx	v24, s9, v25
	vse16.v	v24, (s5)
	vle16.v	v24, (s6)
	vwmacc.vx	v24, ra, v17
	vse16.v	v24, (s6)
	vle16.v	v17, (a4)
	vwmacc.vx	v17, a5, v18
	vse16.v	v17, (a4)
	vle16.v	v17, (s2)
	vwmacc.vx	v17, a0, v16
	vse16.v	v17, (s2)
	vle16.v	v16, (s7)
	addi	a3, a3, 4
	vwmacc.vx	v16, s10, v25
	vse16.v	v16, (s7)
	addi	a1, a1, 16
	bne	a3, t6, .LBB0_27
# %bb.28:                               #   in Loop: Header=BB0_20 Depth=3
	csrr	a0, vlenb
	slli	a1, a0, 2
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v16, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v28, v6, v16
	csrr	a0, vlenb
	slli	a0, a0, 2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v16, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v28, v4, v16
	csrr	a0, vlenb
	slli	a1, a0, 1
	add	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v16, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v28, v3, v16
	csrr	a0, vlenb
	li	a1, 27
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v31, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v16, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v28, v31, v16
	vwmacc.vv	v26, v8, v30
	vwmacc.vv	v28, v8, v22
	vwmacc.vv	v0, v8, v21
	vwmacc.vv	v26, v10, v23
	vwmacc.vv	v28, v10, v12
	vwmacc.vv	v0, v10, v13
	vwmacc.vv	v26, v11, v5
	vwmacc.vv	v28, v11, v14
	vwmacc.vv	v0, v11, v15
	vwmacc.vv	v26, v9, v2
	vwmacc.vv	v28, v9, v19
	vwmacc.vv	v0, v9, v20
	addi	a1, sp, 1800
	vle16.v	v12, (a1)
	vle16.v	v13, (s0)
	addi	a5, sp, 1832
	vle16.v	v24, (a5)
	vle16.v	v14, (a6)
	vwmacc.vv	v26, v12, v13
	vwmacc.vv	v26, v24, v14
	addi	a6, sp, 1864
	vle16.v	v30, (a6)
	vle16.v	v15, (a7)
	addi	s11, sp, 1896
	vle16.v	v25, (s11)
	vle16.v	v16, (t0)
	ld	a2, 432(sp)                     # 8-byte Folded Reload
	addi	a0, a2, 16
	vle16.v	v17, (a0)
	vwmacc.vv	v26, v30, v15
	vwmacc.vv	v26, v25, v16
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v18, v26
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v20, v17
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v16, v20, fa2
	csrr	a0, vlenb
	li	a3, 19
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v26, (a0)                       # Unknown-size Folded Reload
	vfmacc.vv	v26, v16, v18
	vle16.v	v15, (t5)
	vle16.v	v16, (s3)
	vle16.v	v17, (s4)
	vle16.v	v18, (s5)
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v0, v12, v15
	vmv.v.v	v5, v12
	vwmacc.vv	v0, v24, v16
	vwmacc.vv	v0, v30, v17
	vwmacc.vv	v0, v25, v18
	addi	a0, a2, 48
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v16, v0
	addi	a7, sp, 1528
	vle32.v	v18, (a7)
	vle16.v	v15, (a0)
	vfmul.vf	v22, v20, fa3
	csrr	a0, vlenb
	li	a2, 21
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v12, (a0)                       # Unknown-size Folded Reload
	vfmacc.vv	v12, v22, v16
	vfcvt.f.x.v	v18, v18
	addi	s10, sp, 1592
	vle32.v	v22, (s10)
	vsetvli	zero, zero, e16, m1, ta, ma
	vfwcvt.f.f.v	v0, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v16, v0, fa2
	vfnmsub.vv	v16, v18, v26
	vfcvt.f.x.v	v22, v22
	vfmul.vf	v18, v0, fa3
	vfnmsub.vv	v18, v22, v12
	vmv.v.i	v12, 0
	vmv.v.i	v26, 0
	csrr	a0, vlenb
	li	a2, 18
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a2, a0, 4
	add	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v15, v22
	csrr	a0, vlenb
	li	a2, 30
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a0, a0, 4
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v15, v22
	csrr	a0, vlenb
	li	a2, 29
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	slli	a2, a0, 4
	sub	a0, a2, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v15, v22
	csrr	a0, vlenb
	li	a2, 28
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	csrr	a0, vlenb
	li	a2, 14
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v15, v22
	csrr	a0, vlenb
	slli	a2, a0, 3
	add	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v6, v15
	csrr	a0, vlenb
	slli	a0, a0, 3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v4, v15
	csrr	a0, vlenb
	slli	a2, a0, 3
	sub	a0, a2, a0
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v15, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v3, v15
	vle16.v	v15, (t1)
	vle16.v	v14, (s6)
	csrr	a0, vlenb
	li	a2, 30
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vs1r.v	v14, (a0)                       # Unknown-size Folded Spill
	csrr	a0, vlenb
	li	a2, 6
	mul	a0, a0, a2
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v31, v22
	vle16.v	v22, (t2)
	mv	t2, a1
	vle16.v	v7, (a4)
	addi	a0, sp, 1920
	vl1r.v	v23, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v8, v23
	vle16.v	v8, (t3)
	vle16.v	v31, (s2)
	vwmacc.vv	v28, v5, v15
	vmv.v.v	v14, v5
	vle16.v	v15, (t4)
	vwmacc.vv	v28, v24, v22
	vle16.v	v6, (s7)
	csrr	a0, vlenb
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v22, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v10, v22
	addi	ra, sp, 1560
	vle32.v	v22, (ra)
	csrr	a0, vlenb
	li	a1, 11
	mul	a0, a0, a1
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v10, (a0)                       # Unknown-size Folded Reload
	vwmacc.vv	v26, v11, v10
	addi	t3, sp, 1624
	vle32.v	v4, (t3)
	ld	a1, 360(sp)                     # 8-byte Folded Reload
	addi	a1, a1, 1
	ld	a2, 440(sp)                     # 8-byte Folded Reload
	addi	a2, a2, 1344
	mv	t0, a5
	ld	a4, 448(sp)                     # 8-byte Folded Reload
	addi	a4, a4, 1168
	ld	a0, 456(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1344
	sd	a0, 456(sp)                     # 8-byte Folded Spill
	ld	a0, 464(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 464(sp)                     # 8-byte Folded Spill
	ld	a0, 472(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1344
	sd	a0, 472(sp)                     # 8-byte Folded Spill
	ld	a0, 480(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 480(sp)                     # 8-byte Folded Spill
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v2, v22
	vfcvt.f.x.v	v4, v4
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v28, v30, v8
	vwmacc.vv	v28, v25, v15
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v10, v20, fa5
	csrr	a0, vlenb
	li	a3, 13
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v8, (a0)                        # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v9, v8
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmul.vf	v22, v0, fa5
	vfmul.vf	v8, v20, fa4
	vfmul.vf	v20, v0, fa4
	vmv.v.i	v0, 0
	vfcvt.f.x.v	v28, v28
	csrr	a0, vlenb
	li	a3, 30
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl1r.v	v12, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v14, v12
	csrr	a0, vlenb
	li	a3, 23
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v12, (a0)                       # Unknown-size Folded Reload
	vsetvli	zero, zero, e32, m2, ta, ma
	vfmacc.vv	v12, v10, v28
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v24, v7
	vsetvli	zero, zero, e32, m2, ta, ma
	vfnmsub.vv	v22, v2, v12
	vsetvli	zero, zero, e16, m1, ta, ma
	vwmacc.vv	v26, v30, v31
	vwmacc.vv	v26, v25, v6
	vsetvli	zero, zero, e32, m2, ta, ma
	vfcvt.f.x.v	v10, v26
	csrr	a0, vlenb
	li	a3, 25
	mul	a0, a0, a3
	add	a0, a0, sp
	addi	a0, a0, 1920
	vl2r.v	v12, (a0)                       # Unknown-size Folded Reload
	vfmacc.vv	v12, v8, v10
	vfnmsub.vv	v20, v4, v12
	ld	a0, 488(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1344
	sd	a0, 488(sp)                     # 8-byte Folded Spill
	ld	a0, 496(sp)                     # 8-byte Folded Reload
	addi	a0, a0, 1168
	sd	a0, 496(sp)                     # 8-byte Folded Spill
	vmv.v.v	v8, v20
	vmv.v.v	v12, v18
	vmv.v.v	v10, v22
	vmv.v.v	v14, v16
	vsetvli	zero, zero, e16, m1, ta, ma
	vmv.v.i	v7, 0
	ld	a0, 368(sp)                     # 8-byte Folded Reload
	addi	s5, sp, 1496
	addi	t1, sp, 1512
	addi	s3, sp, 1480
	addi	a3, sp, 1336
	addi	t6, sp, 1432
	beq	a1, a0, .LBB0_29
	j	.LBB0_20
.LBB0_29:                               #   in Loop: Header=BB0_7 Depth=2
	j	.LBB0_6
.Lfunc_end0:
	.size	tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K, .Lfunc_end0-tcrv_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K
	.cfi_endproc
                                        # -- End function
	.ident	"Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
