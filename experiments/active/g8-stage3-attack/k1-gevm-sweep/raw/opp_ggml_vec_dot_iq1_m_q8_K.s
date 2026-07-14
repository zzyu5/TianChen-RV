00000000000a361e <ggml_vec_dot_iq1_m_q8_K>:
   a361e:	c22022f3          	csrr	t0,vlenb
   a3622:	028e                	slli	t0,t0,0x3
   a3624:	f8028293          	addi	t0,t0,-128
   a3628:	0072d313          	srli	t1,t0,0x7
   a362c:	12e6                	slli	t0,t0,0x39
   a362e:	0062e2b3          	or	t0,t0,t1
   a3632:	4309                	li	t1,2
   a3634:	00534a63          	blt	t1,t0,a3648 <ggml_vec_dot_iq1_m_q8_K+0x2a>
   a3638:	02028463          	beqz	t0,a3660 <ggml_vec_dot_iq1_m_q8_K+0x42>
   a363c:	4305                	li	t1,1
   a363e:	00629f63          	bne	t0,t1,a365c <ggml_vec_dot_iq1_m_q8_K+0x3e>
   a3642:	8636                	mv	a2,a3
   a3644:	86be                	mv	a3,a5
   a3646:	a62d                	j	a3970 <ggml_vec_dot_iq1_m_q8_K_vl256>
   a3648:	430d                	li	t1,3
   a364a:	00628e63          	beq	t0,t1,a3666 <ggml_vec_dot_iq1_m_q8_K+0x48>
   a364e:	431d                	li	t1,7
   a3650:	00629663          	bne	t0,t1,a365c <ggml_vec_dot_iq1_m_q8_K+0x3e>
   a3654:	8636                	mv	a2,a3
   a3656:	86be                	mv	a3,a5
   a3658:	1510006f          	j	a3fa8 <ggml_vec_dot_iq1_m_q8_K_vl1024>
   a365c:	ec57206f          	j	16520 <ggml_vec_dot_iq1_m_q8_K_generic@plt>
   a3660:	8636                	mv	a2,a3
   a3662:	86be                	mv	a3,a5
   a3664:	a021                	j	a366c <ggml_vec_dot_iq1_m_q8_K_vl128>
   a3666:	8636                	mv	a2,a3
   a3668:	86be                	mv	a3,a5
   a366a:	abf1                	j	a3c46 <ggml_vec_dot_iq1_m_q8_K_vl512>

