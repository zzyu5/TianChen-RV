// Fail-closed stubs for opponent-object symbols that are PROVEN OFF the timed
// path (quants_opp.o / quants_generic_opp.o are monolithic TUs; the driver only
// calls ggml_vec_dot_q5_0_q8_0 / ggml_vec_dot_q5_1_q8_1, but the linker needs
// every symbol referenced anywhere in those TUs resolved). Same methodology as
// the G8 §六 build_seal.txt "opp_link_stubs.o ... PROVEN off timed path,
// fail-closed abort". None of these are reachable from the q5_0/q5_1 GEVM-vs-
// block-dot comparison in driver.cpp -- if one fires, abort() makes that loud
// rather than silently corrupting a measurement.
#include <cstdlib>
extern "C" {
void quantize_row_q1_0_ref(...) { abort(); }
void quantize_row_q4_0_ref(...) { abort(); }
void quantize_row_q4_1_ref(...) { abort(); }
void quantize_row_q5_0_ref(...) { abort(); }
void quantize_row_q5_1_ref(...) { abort(); }
void quantize_row_q8_0_ref(...) { abort(); }
void quantize_row_q8_1_ref(...) { abort(); }
void quantize_row_q8_K_ref(...) { abort(); }
void quantize_row_q2_K_ref(...) { abort(); }
void quantize_row_q3_K_ref(...) { abort(); }
void quantize_row_q4_K_ref(...) { abort(); }
void quantize_row_q5_K_ref(...) { abort(); }
void quantize_row_q6_K_ref(...) { abort(); }
void quantize_row_tq1_0_ref(...) { abort(); }
void quantize_row_tq2_0_ref(...) { abort(); }
void quantize_row_mxfp4_ref(...) { abort(); }
void quantize_row_nvfp4_ref(...) { abort(); }
void quantize_row_iq4_nl_ref(...) { abort(); }
void quantize_iq4_xs(...) { abort(); }
// NOTE: ggml_vec_dot_{q1_0,q3_K,q6_K,iq1_s,iq1_m,iq4_xs}_*_generic are already
// DEFINED in quants_generic_opp.o (confirmed by "multiple definition" link
// error on first attempt) -- do NOT stub those here.
}
