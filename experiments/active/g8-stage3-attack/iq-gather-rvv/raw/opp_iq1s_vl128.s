
/tmp/g8s3/rvv/quants_opp.o:     file format elf64-littleriscv


Disassembly of section .text:

0000000000003f56 <ggml_vec_dot_iq1_s_q8_K_vl128>:
    3f56:	7119                	addi	sp,sp,-128
    3f58:	fc86                	sd	ra,120(sp)
    3f5a:	f8a2                	sd	s0,112(sp)
    3f5c:	f4a6                	sd	s1,104(sp)
    3f5e:	f0ca                	sd	s2,96(sp)
    3f60:	ecce                	sd	s3,88(sp)
    3f62:	e8d2                	sd	s4,80(sp)
    3f64:	e4d6                	sd	s5,72(sp)
    3f66:	c22027f3          	csrr	a5,vlenb
    3f6a:	078e                	slli	a5,a5,0x3
    3f6c:	40f10133          	sub	sp,sp,a5
    3f70:	0ff57793          	zext.b	a5,a0
    3f74:	2c079363          	bnez	a5,423a <.LBB29_7>
    3f78:	4785                	li	a5,1
    3f7a:	2ef71363          	bne	a4,a5,4260 <.LBB29_8>
    3f7e:	8521                	srai	a0,a0,0x8
    3f80:	28a05c63          	blez	a0,4218 <.LBB29_5>
    3f84:	4481                	li	s1,0
    3f86:	cd147057          	vsetivli	zero,8,e32,m2,ta,ma
    3f8a:	5e0fb457          	vmv.v.i	v8,-1
    3f8e:	c2202773          	csrr	a4,vlenb
    3f92:	0706                	slli	a4,a4,0x1
    3f94:	20e72733          	sh1add	a4,a4,a4
    3f98:	970a                	add	a4,a4,sp
    3f9a:	04070713          	addi	a4,a4,64
    3f9e:	22870427          	vs2r.v	v8,(a4)
    3fa2:	5e00b457          	vmv.v.i	v8,1
    3fa6:	c2202773          	csrr	a4,vlenb
    3faa:	070a                	slli	a4,a4,0x2
    3fac:	970a                	add	a4,a4,sp
    3fae:	04070713          	addi	a4,a4,64
    3fb2:	22870427          	vs2r.v	v8,(a4)
    3fb6:	48003737          	lui	a4,0x48003
    3fba:	0716                	slli	a4,a4,0x5
    3fbc:	070d                	addi	a4,a4,3 # 48003003 <.Lpcrel_hi414+0x47ff7f13>
    3fbe:	0742                	slli	a4,a4,0x10
    3fc0:	0da07057          	vsetvli	zero,zero,e64,m4,ta,ma
    3fc4:	5e074457          	vmv.v.x	v8,a4
    3fc8:	0098                	addi	a4,sp,64
    3fca:	62870427          	vs4r.v	v8,(a4)
    3fce:	02000713          	li	a4,32
    3fd2:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    3fd6:	5208a857          	vid.v	v16
    3fda:	4791                	li	a5,4
    3fdc:	8707e857          	vdiv.vx	v16,v16,a5
    3fe0:	0d307057          	vsetvli	zero,zero,e32,m8,ta,ma
    3fe4:	42006a57          	vmv.s.x	v20,zero
    3fe8:	02260613          	addi	a2,a2,34
    3fec:	12400793          	li	a5,292
    3ff0:	02f502b3          	mul	t0,a0,a5
    3ff4:	f00007d3          	fmv.w.x	fa5,zero
    3ff8:	4809                	li	a6,2
    3ffa:	78e1                	lui	a7,0xffff8

0000000000003ffc <.Lpcrel_hi148>:
    3ffc:	00000517          	auipc	a0,0x0
    4000:	00050793          	mv	a5,a0
    4004:	04000313          	li	t1,64
    4008:	08000393          	li	t2,128
    400c:	01810e13          	addi	t3,sp,24
    4010:	01c10e93          	addi	t4,sp,28
    4014:	02010f13          	addi	t5,sp,32
    4018:	02410f93          	addi	t6,sp,36
    401c:	02810913          	addi	s2,sp,40
    4020:	02c10993          	addi	s3,sp,44
    4024:	03010a13          	addi	s4,sp,48
    4028:	03410a93          	addi	s5,sp,52
    402c:	f0138753          	fli.s	fa4,0x1p-3

0000000000004030 <.LBB29_4>:
    4030:	cc847057          	vsetivli	zero,8,e16,m1,ta,ma
    4034:	02065e07          	vle16.v	v28,(a2)
    4038:	a3c63ad7          	vsrl.vi	v21,v28,12
    403c:	2753bad7          	vand.vi	v21,v21,7
    4040:	e3586b57          	vwmulu.vx	v22,v21,a6
    4044:	0d107057          	vsetvli	zero,zero,e32,m2,ta,ma
    4048:	0360bb57          	vadd.vi	v22,v22,1
    404c:	0c807057          	vsetvli	zero,zero,e16,m1,ta,ma
    4050:	27c8cad7          	vand.vx	v21,v28,a7
    4054:	63503057          	vmseq.vi	v0,v21,0
    4058:	0d107057          	vsetvli	zero,zero,e32,m2,ta,ma
    405c:	c2202573          	csrr	a0,vlenb
    4060:	0506                	slli	a0,a0,0x1
    4062:	20a52533          	sh1add	a0,a0,a0
    4066:	950a                	add	a0,a0,sp
    4068:	04050513          	addi	a0,a0,64 # 403c <.LBB29_4+0xc>
    406c:	22850407          	vl2r.v	v8,(a0)
    4070:	c2202573          	csrr	a0,vlenb
    4074:	050a                	slli	a0,a0,0x2
    4076:	950a                	add	a0,a0,sp
    4078:	04050513          	addi	a0,a0,64
    407c:	22850507          	vl2r.v	v10,(a0)
    4080:	5c850c57          	vmerge.vvm	v24,v8,v10,v0
    4084:	fe060513          	addi	a0,a2,-32
    4088:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    408c:	02050d07          	vle8.v	v26,(a0)
    4090:	33c80057          	vrgather.vv	v0,v28,v16
    4094:	0088                	addi	a0,sp,64
    4096:	62850407          	vl4r.v	v8,(a0)
    409a:	a2040e57          	vsrl.vv	v28,v0,v8
    409e:	27c3be57          	vand.vi	v28,v28,7
    40a2:	97c43e57          	vsll.vi	v28,v28,8
    40a6:	4ba32057          	vzext.vf2	v0,v26
    40aa:	2bc00e57          	vor.vv	v28,v28,v0
    40ae:	97c1be57          	vsll.vi	v28,v28,3
    40b2:	cda47057          	vsetivli	zero,8,e64,m4,ta,ma
    40b6:	07c7d007          	vluxei16.v	v0,(a5),v28
    40ba:	00968433          	add	s0,a3,s1
    40be:	00440513          	addi	a0,s0,4
    40c2:	0c237057          	vsetvli	zero,t1,e8,m4,ta,ma
    40c6:	02050207          	vle8.v	v4,(a0)
    40ca:	0c23f057          	vsetvli	zero,t2,e8,m4,ta,ma
    40ce:	ee022457          	vwmul.vv	v8,v0,v4
    40d2:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    40d6:	c68a0457          	vwredsum.vs	v8,v8,v20
    40da:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    40de:	020e6427          	vse32.v	v8,(t3)
    40e2:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    40e6:	c6ca0457          	vwredsum.vs	v8,v12,v20
    40ea:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    40ee:	020ee427          	vse32.v	v8,(t4)
    40f2:	cda47057          	vsetivli	zero,8,e64,m4,ta,ma
    40f6:	07d7d407          	vluxei16.v	v8,(a5),v29
    40fa:	04440513          	addi	a0,s0,68
    40fe:	0c237057          	vsetvli	zero,t1,e8,m4,ta,ma
    4102:	02050607          	vle8.v	v12,(a0)
    4106:	0c23f057          	vsetvli	zero,t2,e8,m4,ta,ma
    410a:	ee862057          	vwmul.vv	v0,v8,v12
    410e:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    4112:	c60a0457          	vwredsum.vs	v8,v0,v20
    4116:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    411a:	020f6427          	vse32.v	v8,(t5)
    411e:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    4122:	c64a0457          	vwredsum.vs	v8,v4,v20
    4126:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    412a:	020fe427          	vse32.v	v8,(t6)
    412e:	cda47057          	vsetivli	zero,8,e64,m4,ta,ma
    4132:	07e7d407          	vluxei16.v	v8,(a5),v30
    4136:	08440513          	addi	a0,s0,132
    413a:	0c237057          	vsetvli	zero,t1,e8,m4,ta,ma
    413e:	02050607          	vle8.v	v12,(a0)
    4142:	0c23f057          	vsetvli	zero,t2,e8,m4,ta,ma
    4146:	ee862057          	vwmul.vv	v0,v8,v12
    414a:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    414e:	c60a0457          	vwredsum.vs	v8,v0,v20
    4152:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    4156:	02096427          	vse32.v	v8,(s2)
    415a:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    415e:	c64a0457          	vwredsum.vs	v8,v4,v20
    4162:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    4166:	0209e427          	vse32.v	v8,(s3)
    416a:	cda47057          	vsetivli	zero,8,e64,m4,ta,ma
    416e:	07f7d407          	vluxei16.v	v8,(a5),v31
    4172:	0c440513          	addi	a0,s0,196
    4176:	0c237057          	vsetvli	zero,t1,e8,m4,ta,ma
    417a:	02050607          	vle8.v	v12,(a0)
    417e:	0c23f057          	vsetvli	zero,t2,e8,m4,ta,ma
    4182:	ee862057          	vwmul.vv	v0,v8,v12
    4186:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    418a:	c60a0457          	vwredsum.vs	v8,v0,v20
    418e:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    4192:	020a6427          	vse32.v	v8,(s4)
    4196:	0ca77057          	vsetvli	zero,a4,e16,m4,ta,ma
    419a:	c64a0457          	vwredsum.vs	v8,v4,v20
    419e:	cd00f057          	vsetivli	zero,1,e32,m1,ta,ma
    41a2:	020ae427          	vse32.v	v8,(s5)
    41a6:	cd147057          	vsetivli	zero,8,e32,m2,ta,ma
    41aa:	020e6407          	vle32.v	v8,(t3)
    41ae:	10440513          	addi	a0,s0,260
    41b2:	cc987057          	vsetivli	zero,16,e16,m2,ta,ma
    41b6:	02055507          	vle16.v	v10,(a0)
    41ba:	cc847057          	vsetivli	zero,8,e16,m1,ta,ma
    41be:	b2a03657          	vnsrl.wi	v12,v10,0
    41c2:	b2a836d7          	vnsrl.wi	v13,v10,16
    41c6:	c6c6a557          	vwadd.vv	v10,v12,v13
    41ca:	0d107057          	vsetvli	zero,zero,e32,m2,ta,ma
    41ce:	97642457          	vmul.vv	v8,v22,v8
    41d2:	976c2657          	vmul.vv	v12,v22,v24
    41d6:	96c52557          	vmul.vv	v10,v12,v10
    41da:	028a2457          	vredsum.vs	v8,v8,v20
    41de:	42802557          	vmv.x.s	a0,v8
    41e2:	fde61687          	flh	fa3,-34(a2)
    41e6:	00042607          	flw	fa2,0(s0)
    41ea:	02aa2457          	vredsum.vs	v8,v10,v20
    41ee:	42802457          	vmv.x.s	s0,v8
    41f2:	402686d3          	fcvt.s.h	fa3,fa3
    41f6:	10d676d3          	fmul.s	fa3,fa2,fa3
    41fa:	d0057653          	fcvt.s.w	fa2,a0
    41fe:	d00475d3          	fcvt.s.w	fa1,s0
    4202:	60e5f643          	fmadd.s	fa2,fa1,fa4,fa2
    4206:	78c6f7c3          	fmadd.s	fa5,fa3,fa2,fa5
    420a:	12448493          	addi	s1,s1,292
    420e:	03260613          	addi	a2,a2,50
    4212:	e0929fe3          	bne	t0,s1,4030 <.LBB29_4>
    4216:	a019                	j	421c <.LBB29_6>

0000000000004218 <.LBB29_5>:
    4218:	f00007d3          	fmv.w.x	fa5,zero

000000000000421c <.LBB29_6>:
    421c:	00f5a027          	fsw	fa5,0(a1)
    4220:	c2202573          	csrr	a0,vlenb
    4224:	050e                	slli	a0,a0,0x3
    4226:	912a                	add	sp,sp,a0
    4228:	70e6                	ld	ra,120(sp)
    422a:	7446                	ld	s0,112(sp)
    422c:	74a6                	ld	s1,104(sp)
    422e:	7906                	ld	s2,96(sp)
    4230:	69e6                	ld	s3,88(sp)
    4232:	6a46                	ld	s4,80(sp)
    4234:	6aa6                	ld	s5,72(sp)
    4236:	6109                	addi	sp,sp,128
    4238:	8082                	ret

000000000000423a <.LBB29_7>:
    423a:	00000517          	auipc	a0,0x0
    423e:	00050513          	mv	a0,a0

0000000000004242 <.Lpcrel_hi143>:
    4242:	00000597          	auipc	a1,0x0
    4246:	00058593          	mv	a1,a1

000000000000424a <.Lpcrel_hi144>:
    424a:	00000617          	auipc	a2,0x0
    424e:	00060693          	mv	a3,a2
    4252:	6605                	lui	a2,0x1
    4254:	ae66061b          	addiw	a2,a2,-1306 # ae6 <.LBB8_3+0xa>
    4258:	00000097          	auipc	ra,0x0
    425c:	000080e7          	jalr	ra # 4258 <.Lpcrel_hi144+0xe>

0000000000004260 <.LBB29_8>:
    4260:	00000517          	auipc	a0,0x0
    4264:	00050513          	mv	a0,a0

0000000000004268 <.Lpcrel_hi146>:
    4268:	00000597          	auipc	a1,0x0
    426c:	00058593          	mv	a1,a1

0000000000004270 <.Lpcrel_hi147>:
    4270:	00000617          	auipc	a2,0x0
    4274:	00060693          	mv	a3,a2
    4278:	6605                	lui	a2,0x1
    427a:	ae76061b          	addiw	a2,a2,-1305 # ae7 <.LBB8_3+0xb>
    427e:	00000097          	auipc	ra,0x0
    4282:	000080e7          	jalr	ra # 427e <.Lpcrel_hi147+0xe>
