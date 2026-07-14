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
