00000000000a2c8c <ggml_vec_dot_iq1_s_q8_K>:
   a2c8c:	c22022f3          	csrr	t0,vlenb
   a2c90:	028e                	slli	t0,t0,0x3
   a2c92:	f8028293          	addi	t0,t0,-128
   a2c96:	0072d313          	srli	t1,t0,0x7
   a2c9a:	12e6                	slli	t0,t0,0x39
   a2c9c:	0062e2b3          	or	t0,t0,t1
   a2ca0:	4309                	li	t1,2
   a2ca2:	00534a63          	blt	t1,t0,a2cb6 <ggml_vec_dot_iq1_s_q8_K+0x2a>
   a2ca6:	02028363          	beqz	t0,a2ccc <ggml_vec_dot_iq1_s_q8_K+0x40>
   a2caa:	4305                	li	t1,1
   a2cac:	00629e63          	bne	t0,t1,a2cc8 <ggml_vec_dot_iq1_s_q8_K+0x3c>
   a2cb0:	8636                	mv	a2,a3
   a2cb2:	86be                	mv	a3,a5
   a2cb4:	a611                	j	a2fb8 <ggml_vec_dot_iq1_s_q8_K_vl256>
   a2cb6:	430d                	li	t1,3
   a2cb8:	00628d63          	beq	t0,t1,a2cd2 <ggml_vec_dot_iq1_s_q8_K+0x46>
   a2cbc:	431d                	li	t1,7
   a2cbe:	00629563          	bne	t0,t1,a2cc8 <ggml_vec_dot_iq1_s_q8_K+0x3c>
   a2cc2:	8636                	mv	a2,a3
   a2cc4:	86be                	mv	a3,a5
   a2cc6:	af05                	j	a33f6 <ggml_vec_dot_iq1_s_q8_K_vl1024>
   a2cc8:	a097306f          	j	166d0 <ggml_vec_dot_iq1_s_q8_K_generic@plt>
   a2ccc:	8636                	mv	a2,a3
   a2cce:	86be                	mv	a3,a5
   a2cd0:	a021                	j	a2cd8 <ggml_vec_dot_iq1_s_q8_K_vl128>
   a2cd2:	8636                	mv	a2,a3
   a2cd4:	86be                	mv	a3,a5
   a2cd6:	ab11                	j	a31ea <ggml_vec_dot_iq1_s_q8_K_vl512>

