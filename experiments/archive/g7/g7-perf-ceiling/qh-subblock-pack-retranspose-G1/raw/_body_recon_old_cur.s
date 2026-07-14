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
