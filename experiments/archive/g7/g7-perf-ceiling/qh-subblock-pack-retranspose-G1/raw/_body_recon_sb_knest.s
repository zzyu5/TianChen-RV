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
