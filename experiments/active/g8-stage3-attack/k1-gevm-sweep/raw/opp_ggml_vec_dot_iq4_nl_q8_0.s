00000000000a7b44 <ggml_vec_dot_iq4_nl_q8_0>:
   a7b44:	c2202673          	csrr	a2,vlenb
   a7b48:	060e                	slli	a2,a2,0x3
   a7b4a:	00365713          	srli	a4,a2,0x3
   a7b4e:	4841                	li	a6,16
   a7b50:	8636                	mv	a2,a3
   a7b52:	86be                	mv	a3,a5
   a7b54:	01071363          	bne	a4,a6,a7b5a <ggml_vec_dot_iq4_nl_q8_0+0x16>
   a7b58:	a011                	j	a7b5c <ggml_vec_dot_iq4_nl_q8_0_vl128>
   a7b5a:	aa39                	j	a7c78 <ggml_vec_dot_iq4_nl_q8_0_vl256>

