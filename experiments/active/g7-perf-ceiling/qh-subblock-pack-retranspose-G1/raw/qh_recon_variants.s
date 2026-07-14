	.file	"qh_recon_variants.c"
	.option pic
	.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.attribute unaligned_access, 1
	.attribute stack_align, 16
	.text
	.align	1
	.globl	recon_old_cur
	.type	recon_old_cur, @function
recon_old_cur:
.LFB0:
	.cfi_startproc
	vsetivli	zero,8,e8,mf2,ta,ma
	vle8.v	v1,0(a1)
	vle8.v	v2,0(a0)
	vsrl.vi	v3,v1,2
	vsrl.vi	v1,v1,3
	vand.vi	v4,v2,15
	vand.vi	v3,v3,1
	vand.vi	v1,v1,1
	vsrl.vi	v2,v2,4
	vsll.vi	v3,v3,4
	vsll.vi	v1,v1,4
	vor.vv	v3,v4,v3
	vor.vv	v2,v2,v1
	vsetvli	a5,zero,e8,mf2,ta,ma
	vse8.v	v3,0(a2)
	vse8.v	v2,0(a3)
	ret
	.cfi_endproc
.LFE0:
	.size	recon_old_cur, .-recon_old_cur
	.align	1
	.globl	recon_sb_knest
	.type	recon_sb_knest, @function
recon_sb_knest:
.LFB1:
	.cfi_startproc
	vsetivli	zero,8,e8,mf2,ta,mu
	li	a4,16
	vle8.v	v2,0(a1)
	vle8.v	v1,0(a0)
	vand.vi	v0,v2,4
	vand.vi	v3,v1,15
	vand.vi	v2,v2,8
	vmsne.vi	v0,v0,0
	vsrl.vi	v1,v1,4
	vadd.vx	v3,v3,a4,v0.t
	vmsne.vi	v0,v2,0
	vadd.vx	v1,v1,a4,v0.t
	vsetvli	a5,zero,e8,mf2,ta,ma
	vse8.v	v3,0(a2)
	vse8.v	v1,0(a3)
	ret
	.cfi_endproc
.LFE1:
	.size	recon_sb_knest, .-recon_sb_knest
	.align	1
	.globl	recon_retrans
	.type	recon_retrans, @function
recon_retrans:
.LFB2:
	.cfi_startproc
	vsetivli	zero,8,e8,mf2,ta,mu
	vle8.v	v1,0(a0)
	vlm.v	v0,0(a1)
	vlm.v	v3,0(a2)
	li	a2,16
	vand.vi	v2,v1,15
	vsrl.vi	v1,v1,4
	vadd.vx	v2,v2,a2,v0.t
	vmv1r.v	v0,v3
	vadd.vx	v1,v1,a2,v0.t
	vsetvli	a5,zero,e8,mf2,ta,ma
	vse8.v	v2,0(a3)
	vse8.v	v1,0(a4)
	ret
	.cfi_endproc
.LFE2:
	.size	recon_retrans, .-recon_retrans
	.ident	"GCC: (gf3b8c022145) 15.2.0"
	.section	.note.GNU-stack,"",@progbits
