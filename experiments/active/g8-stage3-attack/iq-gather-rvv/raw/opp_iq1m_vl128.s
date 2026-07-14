
/tmp/g8s3/rvv/quants_opp.o:     file format elf64-littleriscv


Disassembly of section .text:

0000000000004a4e <ggml_vec_dot_iq1_m_q8_K_vl128>:
    4a4e:	711d                	addi	sp,sp,-96
    4a50:	ec86                	sd	ra,88(sp)
    4a52:	e8a2                	sd	s0,80(sp)
    4a54:	e4a6                	sd	s1,72(sp)
    4a56:	e0ca                	sd	s2,64(sp)
    4a58:	fc4e                	sd	s3,56(sp)
    4a5a:	f852                	sd	s4,48(sp)
    4a5c:	f456                	sd	s5,40(sp)
    4a5e:	f05a                	sd	s6,32(sp)
    4a60:	ec5e                	sd	s7,24(sp)
    4a62:	e862                	sd	s8,16(sp)
    4a64:	c22027f3          	csrr	a5,vlenb
    4a68:	0792                	slli	a5,a5,0x4
    4a6a:	40f10133          	sub	sp,sp,a5
    4a6e:	0ff57793          	zext.b	a5,a0
    4a72:	2e079063          	bnez	a5,4d52 <.LBB34_9>
    4a76:	4785                	li	a5,1
    4a78:	30f71063          	bne	a4,a5,4d78 <.LBB34_10>
    4a7c:	40855293          	srai	t0,a0,0x8
    4a80:	2a505563          	blez	t0,4d2a <.LBB34_7>
    4a84:	4381                	li	t2,0
    4a86:	cd287057          	vsetivli	zero,16,e32,m4,ta,ma
    4a8a:	5e003457          	vmv.v.i	v8,0
    4a8e:	0808                	addi	a0,sp,16
    4a90:	62850427          	vs4r.v	v8,(a0)
    4a94:	00040537          	lui	a0,0x40
    4a98:	0521                	addi	a0,a0,8 # 40008 <.Lpcrel_hi414+0x34f18>
    4a9a:	cd147057          	vsetivli	zero,8,e32,m2,ta,ma
    4a9e:	5e054757          	vmv.v.x	v14,a0
    4aa2:	00800537          	lui	a0,0x800
    4aa6:	0521                	addi	a0,a0,8 # 800008 <.Lpcrel_hi414+0x7f4f18>
    4aa8:	5e054857          	vmv.v.x	v16,a0
    4aac:	01010537          	lui	a0,0x1010
    4ab0:	1015051b          	addiw	a0,a0,257 # 1010101 <.Lpcrel_hi414+0x1005011>
    4ab4:	02051713          	slli	a4,a0,0x20
    4ab8:	953a                	add	a0,a0,a4
    4aba:	cdb87057          	vsetivli	zero,16,e64,m8,ta,ma
    4abe:	5e054c57          	vmv.v.x	v24,a0
    4ac2:	c2202573          	csrr	a0,vlenb
    4ac6:	050a                	slli	a0,a0,0x2
    4ac8:	950a                	add	a0,a0,sp
    4aca:	0541                	addi	a0,a0,16
    4acc:	e2850c27          	vs8r.v	v24,(a0)
    4ad0:	f00007d3          	fmv.w.x	fa5,zero
    4ad4:	12400813          	li	a6,292
    4ad8:	03800893          	li	a7,56
    4adc:	433d                	li	t1,15
    4ade:	0322                	slli	t1,t1,0x8
    4ae0:	70000e93          	li	t4,1792

0000000000004ae4 <.Lpcrel_hi176>:
    4ae4:	00000517          	auipc	a0,0x0
    4ae8:	00050f13          	mv	t5,a0
    4aec:	02000f93          	li	t6,32
    4af0:	f0138753          	fli.s	fa4,0x1p-3
    4af4:	c2202573          	csrr	a0,vlenb
    4af8:	4739                	li	a4,14
    4afa:	02e50533          	mul	a0,a0,a4
    4afe:	950a                	add	a0,a0,sp
    4b00:	0541                	addi	a0,a0,16 # 4af4 <.Lpcrel_hi176+0x10>
    4b02:	22850727          	vs2r.v	v14,(a0)
    4b06:	c2202573          	csrr	a0,vlenb
    4b0a:	050a                	slli	a0,a0,0x2
    4b0c:	20a52533          	sh1add	a0,a0,a0
    4b10:	950a                	add	a0,a0,sp
    4b12:	0541                	addi	a0,a0,16
    4b14:	22850827          	vs2r.v	v16,(a0)

0000000000004b18 <.LBB34_4>:
    4b18:	03038e33          	mul	t3,t2,a6
    4b1c:	9e36                	add	t3,t3,a3
    4b1e:	004e0513          	addi	a0,t3,4
    4b22:	031387b3          	mul	a5,t2,a7
    4b26:	00f60c33          	add	s8,a2,a5
    4b2a:	020c0b93          	addi	s7,s8,32
    4b2e:	030c0413          	addi	s0,s8,48
    4b32:	033c4983          	lbu	s3,51(s8)
    4b36:	034c1a03          	lh	s4,52(s8)
    4b3a:	030c5903          	lhu	s2,48(s8)
    4b3e:	036c1a83          	lh	s5,54(s8)
    4b42:	0f09f993          	andi	s3,s3,240
    4b46:	004a5493          	srli	s1,s4,0x4
    4b4a:	0064fa33          	and	s4,s1,t1
    4b4e:	800af493          	andi	s1,s5,-2048
    4b52:	48b49a93          	bclri	s5,s1,0xb
    4b56:	4b05                	li	s6,1
    4b58:	0818                	addi	a4,sp,16
    4b5a:	62870c07          	vl4r.v	v24,(a4)
    4b5e:	9f81be57          	vmv4r.v	v28,v24

0000000000004b62 <.LBB34_5>:
    4b62:	cc847057          	vsetivli	zero,8,e16,m1,ta,ma
    4b66:	020b8407          	vle8.v	v8,(s7)
    4b6a:	4a8324d7          	vzext.vf2	v9,v8
    4b6e:	96943457          	vsll.vi	v8,v9,8
    4b72:	2a940457          	vor.vv	v8,v9,v8
    4b76:	cc987057          	vsetivli	zero,16,e16,m2,ta,ma
    4b7a:	020c0487          	vle8.v	v9,(s8)
    4b7e:	4a832557          	vzext.vf2	v10,v8
    4b82:	4a932657          	vzext.vf2	v12,v9
    4b86:	96a70457          	vsll.vv	v8,v10,v14
    4b8a:	268ec457          	vand.vx	v8,v8,t4
    4b8e:	2ac40457          	vor.vv	v8,v12,v8
    4b92:	9681b257          	vsll.vi	v4,v8,3
    4b96:	26a80457          	vand.vv	v8,v10,v16
    4b9a:	7a803057          	vmsgtu.vi	v0,v8,0
    4b9e:	0db07057          	vsetvli	zero,zero,e64,m8,ta,ma
    4ba2:	c2202773          	csrr	a4,vlenb
    4ba6:	070a                	slli	a4,a4,0x2
    4ba8:	970a                	add	a4,a4,sp
    4baa:	0741                	addi	a4,a4,16
    4bac:	e2870407          	vl8r.v	v8,(a4)
    4bb0:	5c8fb457          	vmerge.vim	v8,v8,-1,v0
    4bb4:	cda47057          	vsetivli	zero,8,e64,m4,ta,ma
    4bb8:	064f5007          	vluxei16.v	v0,(t5),v4
    4bbc:	0c1ff057          	vsetvli	zero,t6,e8,m2,ta,ma
    4bc0:	02050307          	vle8.v	v6,(a0)
    4bc4:	00041783          	lh	a5,0(s0)
    4bc8:	ee032857          	vwmul.vv	v16,v0,v6
    4bcc:	ee832a57          	vwmul.vv	v20,v8,v6
    4bd0:	00179493          	slli	s1,a5,0x1
    4bd4:	88b9                	andi	s1,s1,14
    4bd6:	0485                	addi	s1,s1,1
    4bd8:	0027d713          	srli	a4,a5,0x2
    4bdc:	8b39                	andi	a4,a4,14
    4bde:	0705                	addi	a4,a4,1
    4be0:	cc987057          	vsetivli	zero,16,e16,m2,ta,ma
    4be4:	f704ee57          	vwmacc.vx	v28,s1,v16
    4be8:	f7276e57          	vwmacc.vx	v28,a4,v18
    4bec:	f744ec57          	vwmacc.vx	v24,s1,v20
    4bf0:	f7676c57          	vwmacc.vx	v24,a4,v22
    4bf4:	02050713          	addi	a4,a0,32
    4bf8:	0c1ff057          	vsetvli	zero,t6,e8,m2,ta,ma
    4bfc:	02070407          	vle8.v	v8,(a4)
    4c00:	ee242857          	vwmul.vv	v16,v2,v8
    4c04:	eea42057          	vwmul.vv	v0,v10,v8
    4c08:	0057d713          	srli	a4,a5,0x5
    4c0c:	8b39                	andi	a4,a4,14
    4c0e:	0705                	addi	a4,a4,1
    4c10:	83a1                	srli	a5,a5,0x8
    4c12:	8bb9                	andi	a5,a5,14
    4c14:	0785                	addi	a5,a5,1
    4c16:	cc987057          	vsetivli	zero,16,e16,m2,ta,ma
    4c1a:	f7076e57          	vwmacc.vx	v28,a4,v16
    4c1e:	f727ee57          	vwmacc.vx	v28,a5,v18
    4c22:	f6076c57          	vwmacc.vx	v24,a4,v0
    4c26:	f627ec57          	vwmacc.vx	v24,a5,v2
    4c2a:	cda47057          	vsetivli	zero,8,e64,m4,ta,ma
    4c2e:	065f5407          	vluxei16.v	v8,(t5),v5
    4c32:	04050713          	addi	a4,a0,64
    4c36:	0c1ff057          	vsetvli	zero,t6,e8,m2,ta,ma
    4c3a:	02070807          	vle8.v	v16,(a4)
    4c3e:	00241703          	lh	a4,2(s0)
    4c42:	ee882a57          	vwmul.vv	v20,v8,v16
    4c46:	eec82057          	vwmul.vv	v0,v12,v16
    4c4a:	c22027f3          	csrr	a5,vlenb
    4c4e:	078a                	slli	a5,a5,0x2
    4c50:	20f7a7b3          	sh1add	a5,a5,a5
    4c54:	978a                	add	a5,a5,sp
    4c56:	07c1                	addi	a5,a5,16
    4c58:	22878807          	vl2r.v	v16,(a5)
    4c5c:	00171793          	slli	a5,a4,0x1
    4c60:	8bb9                	andi	a5,a5,14
    4c62:	0785                	addi	a5,a5,1
    4c64:	00275493          	srli	s1,a4,0x2
    4c68:	88b9                	andi	s1,s1,14
    4c6a:	0485                	addi	s1,s1,1
    4c6c:	cc987057          	vsetivli	zero,16,e16,m2,ta,ma
    4c70:	f747ee57          	vwmacc.vx	v28,a5,v20
    4c74:	f764ee57          	vwmacc.vx	v28,s1,v22
    4c78:	f607ec57          	vwmacc.vx	v24,a5,v0
    4c7c:	f624ec57          	vwmacc.vx	v24,s1,v2
    4c80:	06050793          	addi	a5,a0,96
    4c84:	0c1ff057          	vsetvli	zero,t6,e8,m2,ta,ma
    4c88:	02078407          	vle8.v	v8,(a5)
    4c8c:	eee42057          	vwmul.vv	v0,v14,v8
    4c90:	eea42657          	vwmul.vv	v12,v10,v8
    4c94:	00575793          	srli	a5,a4,0x5
    4c98:	8bb9                	andi	a5,a5,14
    4c9a:	0785                	addi	a5,a5,1
    4c9c:	8321                	srli	a4,a4,0x8
    4c9e:	8b39                	andi	a4,a4,14
    4ca0:	0705                	addi	a4,a4,1
    4ca2:	cc987057          	vsetivli	zero,16,e16,m2,ta,ma
    4ca6:	f6c7ee57          	vwmacc.vx	v28,a5,v12
    4caa:	f6e76e57          	vwmacc.vx	v28,a4,v14
    4cae:	c22024f3          	csrr	s1,vlenb
    4cb2:	40b9                	li	ra,14
    4cb4:	021484b3          	mul	s1,s1,ra
    4cb8:	948a                	add	s1,s1,sp
    4cba:	04c1                	addi	s1,s1,16
    4cbc:	22848707          	vl2r.v	v14,(s1)
    4cc0:	f607ec57          	vwmacc.vx	v24,a5,v0
    4cc4:	f6276c57          	vwmacc.vx	v24,a4,v2
    4cc8:	0ba1                	addi	s7,s7,8
    4cca:	0c41                	addi	s8,s8,16
    4ccc:	08050513          	addi	a0,a0,128
    4cd0:	001b7713          	andi	a4,s6,1
    4cd4:	0411                	addi	s0,s0,4
    4cd6:	4b01                	li	s6,0
    4cd8:	e80715e3          	bnez	a4,4b62 <.LBB34_5>
    4cdc:	00c95513          	srli	a0,s2,0xc
    4ce0:	00a9e533          	or	a0,s3,a0
    4ce4:	015a6733          	or	a4,s4,s5
    4ce8:	8d59                	or	a0,a0,a4
    4cea:	0d207057          	vsetvli	zero,zero,e32,m4,ta,ma
    4cee:	42006457          	vmv.s.x	v8,zero
    4cf2:	03c424d7          	vredsum.vs	v9,v28,v8
    4cf6:	42902757          	vmv.x.s	a4,v9
    4cfa:	03842457          	vredsum.vs	v8,v24,v8
    4cfe:	000e2687          	flw	fa3,0(t3)
    4d02:	428027d7          	vmv.x.s	a5,v8
    4d06:	f4050653          	fmv.h.x	fa2,a0
    4d0a:	40260653          	fcvt.s.h	fa2,fa2
    4d0e:	10c6f6d3          	fmul.s	fa3,fa3,fa2
    4d12:	d0077653          	fcvt.s.w	fa2,a4
    4d16:	d007f5d3          	fcvt.s.w	fa1,a5
    4d1a:	60e5f643          	fmadd.s	fa2,fa1,fa4,fa2
    4d1e:	0385                	addi	t2,t2,1
    4d20:	78c6f7c3          	fmadd.s	fa5,fa3,fa2,fa5
    4d24:	de539ae3          	bne	t2,t0,4b18 <.LBB34_4>
    4d28:	a019                	j	4d2e <.LBB34_8>

0000000000004d2a <.LBB34_7>:
    4d2a:	f00007d3          	fmv.w.x	fa5,zero

0000000000004d2e <.LBB34_8>:
    4d2e:	00f5a027          	fsw	fa5,0(a1)
    4d32:	c2202573          	csrr	a0,vlenb
    4d36:	0512                	slli	a0,a0,0x4
    4d38:	912a                	add	sp,sp,a0
    4d3a:	60e6                	ld	ra,88(sp)
    4d3c:	6446                	ld	s0,80(sp)
    4d3e:	64a6                	ld	s1,72(sp)
    4d40:	6906                	ld	s2,64(sp)
    4d42:	79e2                	ld	s3,56(sp)
    4d44:	7a42                	ld	s4,48(sp)
    4d46:	7aa2                	ld	s5,40(sp)
    4d48:	7b02                	ld	s6,32(sp)
    4d4a:	6be2                	ld	s7,24(sp)
    4d4c:	6c42                	ld	s8,16(sp)
    4d4e:	6125                	addi	sp,sp,96
    4d50:	8082                	ret

0000000000004d52 <.LBB34_9>:
    4d52:	00000517          	auipc	a0,0x0
    4d56:	00050513          	mv	a0,a0

0000000000004d5a <.Lpcrel_hi171>:
    4d5a:	00000597          	auipc	a1,0x0
    4d5e:	00058593          	mv	a1,a1

0000000000004d62 <.Lpcrel_hi172>:
    4d62:	00000617          	auipc	a2,0x0
    4d66:	00060693          	mv	a3,a2
    4d6a:	6605                	lui	a2,0x1
    4d6c:	c846061b          	addiw	a2,a2,-892 # c84 <.Lpcrel_hi51>
    4d70:	00000097          	auipc	ra,0x0
    4d74:	000080e7          	jalr	ra # 4d70 <.Lpcrel_hi172+0xe>

0000000000004d78 <.LBB34_10>:
    4d78:	00000517          	auipc	a0,0x0
    4d7c:	00050513          	mv	a0,a0

0000000000004d80 <.Lpcrel_hi174>:
    4d80:	00000597          	auipc	a1,0x0
    4d84:	00058593          	mv	a1,a1

0000000000004d88 <.Lpcrel_hi175>:
    4d88:	00000617          	auipc	a2,0x0
    4d8c:	00060693          	mv	a3,a2
    4d90:	6605                	lui	a2,0x1
    4d92:	c856061b          	addiw	a2,a2,-891 # c85 <.Lpcrel_hi51+0x1>
    4d96:	00000097          	auipc	ra,0x0
    4d9a:	000080e7          	jalr	ra # 4d96 <.Lpcrel_hi175+0xe>
