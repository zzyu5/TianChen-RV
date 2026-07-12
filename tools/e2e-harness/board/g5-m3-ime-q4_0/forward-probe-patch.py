f = "ggml/src/ggml-cpu/spacemit/ime.cpp"
s = open(f).read()
inc = '#include <cstdio>  // for GGML_ASSERT\n'
assert inc in s, "cstdio anchor not found"
if "#include <cstdlib>" not in s:
    s = s.replace(inc, inc + "#include <cstdlib>  // [TCRV-PROBE] std::getenv\n", 1)
anchor = "        const int64_t gemm_n = ne01;\n"
assert s.count(anchor) == 1, "expected exactly one gemm_n anchor"
block = '''
        // [TCRV-IME-BRIDGE-PROBE] session-2 reversible forward-hook reachability probe.
        // Env-gated (default OFF -> zero behavior change; ON/OFF symmetric). Proves the
        // q4_0 PREFILL hook point (where a tcrv IME q4_0 forward-bridge would route) is
        // reached in the real llama forward with real q4_0 traffic. NOT routing here:
        // w_data is vendor-repacked; routing the fragment-major tcrv kernel needs a
        // vendor-layout adaptation (documented next-session).
        if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
            static bool tcrv_probe_done = false;
            if (!tcrv_probe_done && ith == 0 && gemm_m > 1 && std::getenv("TCRV_IME_BRIDGE_PROBE")) {
                tcrv_probe_done = true;
                fprintf(stderr,
                        "[TCRV-IME-BRIDGE-PROBE] forward_mul_mat q4_0 PREFILL hook reached: "
                        "gemm_m=%lld gemm_n=%lld gemm_k=%lld\\n",
                        (long long) gemm_m, (long long) gemm_n, (long long) gemm_k);
            }
        }
'''
s = s.replace(anchor, anchor + block, 1)
open(f, "w").write(s)
# post-write verify
t = open(f).read()
assert "PREFILL hook reached" in t, "probe fprintf missing after write"
assert "#include <cstdlib>" in t, "cstdlib include missing after write"
print("patched OK; probe_markers=%d" % t.count("TCRV-IME-BRIDGE-PROBE"))
