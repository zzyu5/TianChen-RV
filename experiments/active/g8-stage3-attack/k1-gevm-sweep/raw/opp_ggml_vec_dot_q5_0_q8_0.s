000000000009f81c <ggml_vec_dot_q5_0_q8_0>:
   9f81c:	02000613          	li	a2,32
   9f820:	08c54b63          	blt	a0,a2,9f8b6 <ggml_vec_dot_q5_0_q8_0+0x9a>
   9f824:	c2202773          	csrr	a4,vlenb
   9f828:	0055551b          	srliw	a0,a0,0x5
   9f82c:	0d067057          	vsetvli	zero,a2,e32,m1,ta,ma
   9f830:	4641                	li	a2,16
   9f832:	5e003457          	vmv.v.i	v8,0
   9f836:	08c71563          	bne	a4,a2,9f8c0 <ggml_vec_dot_q5_0_q8_0+0xa4>
   9f83a:	0699                	addi	a3,a3,6
   9f83c:	0789                	addi	a5,a5,2
   9f83e:	4659                	li	a2,22
   9f840:	02c50533          	mul	a0,a0,a2
   9f844:	9536                	add	a0,a0,a3
   9f846:	f00007d3          	fmv.w.x	fa5,zero
   9f84a:	02000613          	li	a2,32
   9f84e:	cc087057          	vsetivli	zero,16,e8,m1,ta,ma
   9f852:	02068487          	vle8.v	v9,(a3)
   9f856:	2697b557          	vand.vi	v10,v9,15
   9f85a:	a29235d7          	vsrl.vi	v11,v9,4
   9f85e:	ffc68713          	addi	a4,a3,-4
   9f862:	04167057          	vsetvli	zero,a2,e8,m2,ta,mu
   9f866:	02b70487          	vlm.v	v9,(a4)
   9f86a:	02078607          	vle8.v	v12,(a5)
   9f86e:	7694a057          	vmnot.m	v0,v9
   9f872:	00a83557          	vadd.vi	v10,v10,-16,v0.t
   9f876:	eea62857          	vwmul.vv	v16,v10,v12
   9f87a:	0ca07057          	vsetvli	zero,zero,e16,m4,ta,ma
   9f87e:	c70404d7          	vwredsum.vs	v9,v16,v8
   9f882:	0d307057          	vsetvli	zero,zero,e32,m8,ta,ma
   9f886:	ffa69707          	flh	fa4,-6(a3)
   9f88a:	ffe79687          	flh	fa3,-2(a5)
   9f88e:	42902757          	vmv.x.s	a4,v9
   9f892:	40270753          	fcvt.s.h	fa4,fa4
   9f896:	402686d3          	fcvt.s.h	fa3,fa3
   9f89a:	10d77753          	fmul.s	fa4,fa4,fa3
   9f89e:	d00776d3          	fcvt.s.w	fa3,a4
   9f8a2:	78d777c3          	fmadd.s	fa5,fa4,fa3,fa5
   9f8a6:	06d9                	addi	a3,a3,22
   9f8a8:	02278793          	addi	a5,a5,34
   9f8ac:	faa691e3          	bne	a3,a0,9f84e <ggml_vec_dot_q5_0_q8_0+0x32>
   9f8b0:	00f5a027          	fsw	fa5,0(a1)
   9f8b4:	8082                	ret
   9f8b6:	f00007d3          	fmv.w.x	fa5,zero
   9f8ba:	00f5a027          	fsw	fa5,0(a1)
   9f8be:	8082                	ret
   9f8c0:	0789                	addi	a5,a5,2
   9f8c2:	0699                	addi	a3,a3,6
   9f8c4:	02200613          	li	a2,34
   9f8c8:	02c50533          	mul	a0,a0,a2
   9f8cc:	953e                	add	a0,a0,a5
   9f8ce:	f00007d3          	fmv.w.x	fa5,zero
   9f8d2:	02000613          	li	a2,32
   9f8d6:	cc087057          	vsetivli	zero,16,e8,m1,ta,ma
   9f8da:	02068487          	vle8.v	v9,(a3)
   9f8de:	2697b557          	vand.vi	v10,v9,15
   9f8e2:	a29234d7          	vsrl.vi	v9,v9,4
   9f8e6:	0c067057          	vsetvli	zero,a2,e8,m1,ta,ma
   9f8ea:	3a983557          	vslideup.vi	v10,v9,16
   9f8ee:	ffc68713          	addi	a4,a3,-4
   9f8f2:	04167057          	vsetvli	zero,a2,e8,m2,ta,mu
   9f8f6:	02b70487          	vlm.v	v9,(a4)
   9f8fa:	02078607          	vle8.v	v12,(a5)
   9f8fe:	7694a057          	vmnot.m	v0,v9
   9f902:	00a83557          	vadd.vi	v10,v10,-16,v0.t
   9f906:	eea62857          	vwmul.vv	v16,v10,v12
   9f90a:	0ca07057          	vsetvli	zero,zero,e16,m4,ta,ma
   9f90e:	c70404d7          	vwredsum.vs	v9,v16,v8
   9f912:	0d307057          	vsetvli	zero,zero,e32,m8,ta,ma
   9f916:	ffa69707          	flh	fa4,-6(a3)
   9f91a:	ffe79687          	flh	fa3,-2(a5)
   9f91e:	42902757          	vmv.x.s	a4,v9
   9f922:	40270753          	fcvt.s.h	fa4,fa4
   9f926:	402686d3          	fcvt.s.h	fa3,fa3
   9f92a:	10d77753          	fmul.s	fa4,fa4,fa3
   9f92e:	d00776d3          	fcvt.s.w	fa3,a4
   9f932:	78d777c3          	fmadd.s	fa5,fa4,fa3,fa5
   9f936:	02278793          	addi	a5,a5,34
   9f93a:	06d9                	addi	a3,a3,22
   9f93c:	f8a79de3          	bne	a5,a0,9f8d6 <ggml_vec_dot_q5_0_q8_0+0xba>
   9f940:	bf85                	j	9f8b0 <ggml_vec_dot_q5_0_q8_0+0x94>

