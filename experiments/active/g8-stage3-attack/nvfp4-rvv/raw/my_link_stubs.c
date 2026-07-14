/* Fail-closed stubs for symbols referenced elsewhere in quants_generic_opp.o's
 * monolithic TU but PROVEN OFF the timed nvfp4 vec_dot path (we only call
 * ggml_vec_dot_nvfp4_q8_0). abort() makes any accidental reachability loud. */
#include <stdlib.h>
#define STUB(name) void name(void) { abort(); }
STUB(quantize_row_q1_0_ref)
STUB(quantize_row_q4_0_ref)
STUB(quantize_row_q4_1_ref)
STUB(quantize_row_q5_0_ref)
STUB(quantize_row_q5_1_ref)
STUB(quantize_row_q8_0_ref)
STUB(quantize_row_q8_1_ref)
STUB(quantize_row_q8_K_ref)
STUB(quantize_row_q2_K_ref)
STUB(quantize_row_q3_K_ref)
STUB(quantize_row_q4_K_ref)
STUB(quantize_row_q5_K_ref)
STUB(quantize_row_q6_K_ref)
STUB(quantize_row_tq1_0_ref)
STUB(quantize_row_tq2_0_ref)
STUB(quantize_row_mxfp4_ref)
STUB(quantize_row_nvfp4_ref)
STUB(quantize_row_iq4_nl_ref)
STUB(quantize_iq4_xs)
