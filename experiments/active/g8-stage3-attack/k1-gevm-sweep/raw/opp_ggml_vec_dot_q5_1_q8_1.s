000000000009f942 <ggml_vec_dot_q5_1_q8_1>:
   9f942:	02000713          	li	a4,32
   9f946:	0ae54563          	blt	a0,a4,9f9f0 <ggml_vec_dot_q5_1_q8_1+0xae>
   9f94a:	c2202673          	csrr	a2,vlenb
   9f94e:	0055581b          	srliw	a6,a0,0x5
   9f952:	0d077057          	vsetvli	zero,a4,e32,m1,ta,ma
   9f956:	48c1                	li	a7,16
   9f958:	5e003457          	vmv.v.i	v8,0
   9f95c:	09161f63          	bne	a2,a7,9f9fa <ggml_vec_dot_q5_1_q8_1+0xb8>
   9f960:	0691                	addi	a3,a3,4
   9f962:	0791                	addi	a5,a5,4
   9f964:	4661                	li	a2,24
   9f966:	02c80633          	mul	a2,a6,a2
   9f96a:	9636                	add	a2,a2,a3
   9f96c:	f00007d3          	fmv.w.x	fa5,zero
   9f970:	02000813          	li	a6,32
   9f974:	00468713          	addi	a4,a3,4
   9f978:	cc087057          	vsetivli	zero,16,e8,m1,ta,ma
   9f97c:	02070487          	vle8.v	v9,(a4)
   9f980:	2697b557          	vand.vi	v10,v9,15
   9f984:	a29235d7          	vsrl.vi	v11,v9,4
   9f988:	04187057          	vsetvli	zero,a6,e8,m2,ta,mu
   9f98c:	02b68007          	vlm.v	v0,(a3)
   9f990:	02078607          	vle8.v	v12,(a5)
   9f994:	28a8c557          	vor.vx	v10,v10,a7,v0.t
   9f998:	eea62857          	vwmul.vv	v16,v10,v12
   9f99c:	0ca07057          	vsetvli	zero,zero,e16,m4,ta,ma
   9f9a0:	c70404d7          	vwredsum.vs	v9,v16,v8
   9f9a4:	0d307057          	vsetvli	zero,zero,e32,m8,ta,ma
   9f9a8:	ffc69707          	flh	fa4,-4(a3)
   9f9ac:	ffc79687          	flh	fa3,-4(a5)
   9f9b0:	42902757          	vmv.x.s	a4,v9
   9f9b4:	40270753          	fcvt.s.h	fa4,fa4
   9f9b8:	402686d3          	fcvt.s.h	fa3,fa3
   9f9bc:	ffe69607          	flh	fa2,-2(a3)
   9f9c0:	ffe79587          	flh	fa1,-2(a5)
   9f9c4:	10d77753          	fmul.s	fa4,fa4,fa3
   9f9c8:	d00776d3          	fcvt.s.w	fa3,a4
   9f9cc:	40260653          	fcvt.s.h	fa2,fa2
   9f9d0:	402585d3          	fcvt.s.h	fa1,fa1
   9f9d4:	10b67653          	fmul.s	fa2,fa2,fa1
   9f9d8:	60d77743          	fmadd.s	fa4,fa4,fa3,fa2
   9f9dc:	00e7f7d3          	fadd.s	fa5,fa5,fa4
   9f9e0:	06e1                	addi	a3,a3,24
   9f9e2:	02478793          	addi	a5,a5,36
   9f9e6:	f8c697e3          	bne	a3,a2,9f974 <ggml_vec_dot_q5_1_q8_1+0x32>
   9f9ea:	00f5a027          	fsw	fa5,0(a1)
   9f9ee:	8082                	ret
   9f9f0:	f00007d3          	fmv.w.x	fa5,zero
   9f9f4:	00f5a027          	fsw	fa5,0(a1)
   9f9f8:	8082                	ret
   9f9fa:	0791                	addi	a5,a5,4
   9f9fc:	0691                	addi	a3,a3,4
   9f9fe:	02400613          	li	a2,36
   9fa02:	02c80633          	mul	a2,a6,a2
   9fa06:	963e                	add	a2,a2,a5
   9fa08:	f00007d3          	fmv.w.x	fa5,zero
   9fa0c:	02000713          	li	a4,32
   9fa10:	00468513          	addi	a0,a3,4
   9fa14:	cc087057          	vsetivli	zero,16,e8,m1,ta,ma
   9fa18:	02050487          	vle8.v	v9,(a0)
   9fa1c:	2697b557          	vand.vi	v10,v9,15
   9fa20:	a29234d7          	vsrl.vi	v9,v9,4
   9fa24:	0c077057          	vsetvli	zero,a4,e8,m1,ta,ma
   9fa28:	3a983557          	vslideup.vi	v10,v9,16
   9fa2c:	04177057          	vsetvli	zero,a4,e8,m2,ta,mu
   9fa30:	02b68007          	vlm.v	v0,(a3)
   9fa34:	02078607          	vle8.v	v12,(a5)
   9fa38:	28a8c557          	vor.vx	v10,v10,a7,v0.t
   9fa3c:	eea62857          	vwmul.vv	v16,v10,v12
   9fa40:	0ca07057          	vsetvli	zero,zero,e16,m4,ta,ma
   9fa44:	c70404d7          	vwredsum.vs	v9,v16,v8
   9fa48:	0d307057          	vsetvli	zero,zero,e32,m8,ta,ma
   9fa4c:	ffc69707          	flh	fa4,-4(a3)
   9fa50:	ffc79687          	flh	fa3,-4(a5)
   9fa54:	42902557          	vmv.x.s	a0,v9
   9fa58:	40270753          	fcvt.s.h	fa4,fa4
   9fa5c:	402686d3          	fcvt.s.h	fa3,fa3
   9fa60:	ffe69607          	flh	fa2,-2(a3)
   9fa64:	ffe79587          	flh	fa1,-2(a5)
   9fa68:	10d77753          	fmul.s	fa4,fa4,fa3
   9fa6c:	d00576d3          	fcvt.s.w	fa3,a0
   9fa70:	40260653          	fcvt.s.h	fa2,fa2
   9fa74:	402585d3          	fcvt.s.h	fa1,fa1
   9fa78:	10b67653          	fmul.s	fa2,fa2,fa1
   9fa7c:	60d77743          	fmadd.s	fa4,fa4,fa3,fa2
   9fa80:	00e7f7d3          	fadd.s	fa5,fa5,fa4
   9fa84:	02478793          	addi	a5,a5,36
   9fa88:	06e1                	addi	a3,a3,24
   9fa8a:	f8c793e3          	bne	a5,a2,9fa10 <ggml_vec_dot_q5_1_q8_1+0xce>
   9fa8e:	bfb1                	j	9f9ea <ggml_vec_dot_q5_1_q8_1+0xa8>

