// G5-M3 IME logit-level bounded-ULP harness -- self-built llama-API logit dumper.
//
// Purpose: quantify the near-tie argmax flips seen in the ratified IME triple
// {q4_0, q8_0, q4_K}@ime real-llama e2e A==B seal. Instead of only observing that
// greedy generation diverges, we dump the FULL per-position logit vectors under
// three backends over an IDENTICAL teacher-forced token sequence, so ON/VEN/OFF
// logits are directly comparable position-by-position:
//   OFF = build-off  (stock RVV, no IME)   -- cross-paradigm oracle
//   VEN = build-ime  (env unset, vendor IME) -- same-paradigm oracle
//   ON  = build-ime  (TCRV_IME_*_BRIDGE=1)   -- our tcrv IME bridge
//
// Modes:
//   gendump <model> <prompts_file> <out_dir> <tag> <n_gen>
//       greedy-generate (this backend) n_gen tokens per prompt -> canonical token
//       sequence (BOS + prompt + gen) written to <out_dir>/seq_<i>.txt and prompt
//       length to <out_dir>/meta_<i>.txt; then teacher-force that sequence in ONE
//       batch (M = seqlen > 1 => routes through IME under the bridge) and dump
//       per-position logits to <out_dir>/logits_<tag>_<i>.bin. Also dumps
//       <out_dir>/vocab.tsv once (id \t escaped-piece).
//   dump <model> <prompts_file> <out_dir> <tag>
//       read the canonical seq_<i>.txt (produced by the OFF gendump run) and
//       teacher-force + dump logits (same bin format). Use for VEN and ON so all
//       three backends evaluate the SAME fixed input tokens.
//
// .bin format: int32 n_pos, int32 n_vocab, then n_pos*n_vocab float32 (row-major).
//
// Board-reversible: touches nothing in the vendor tree; scratch under out_dir only.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

static std::vector<std::string> read_lines(const char * path) {
    std::vector<std::string> v;
    std::ifstream f(path);
    std::string l;
    while (std::getline(f, l)) {
        if (!l.empty() && l[l.size() - 1] == '\r') l.pop_back();
        if (!l.empty()) v.push_back(l);
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model> <gendump|dump> <prompts_file> <out_dir> <tag> [n_gen]\n", argv[0]);
        return 2;
    }
    const char * model_path  = argv[1];
    std::string  mode        = argv[2];
    const char * prompts_file = argv[3];
    std::string  out_dir     = argv[4];
    std::string  tag         = argv[5];
    int          n_gen       = (argc > 6) ? atoi(argv[6]) : 24;

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed: %s\n", model_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = 1024;
    cparams.n_batch        = 1024;
    cparams.n_ubatch       = 1024;   // full sequence in one ubatch => deterministic single prefill
    cparams.n_threads      = 4;
    cparams.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    // one-time vocab dump (gendump only)
    if (mode == "gendump") {
        std::string vp = out_dir + "/vocab.tsv";
        FILE * vf = fopen(vp.c_str(), "w");
        if (vf) {
            char buf[256];
            for (int t = 0; t < n_vocab; ++t) {
                int nc = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
                std::string piece = (nc > 0) ? std::string(buf, nc) : std::string("");
                for (char & c : piece) if (c == '\n' || c == '\t' || c == '\r') c = ' ';
                fprintf(vf, "%d\t%s\n", t, piece.c_str());
            }
            fclose(vf);
        }
    }

    auto prompts = read_lines(prompts_file);

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        int idx = (int) pi + 1;
        std::vector<llama_token> seq;
        char seqpath[512];  snprintf(seqpath,  sizeof(seqpath),  "%s/seq_%d.txt",  out_dir.c_str(), idx);
        char metapath[512]; snprintf(metapath, sizeof(metapath), "%s/meta_%d.txt", out_dir.c_str(), idx);

        if (mode == "gendump") {
            const std::string & text = prompts[pi];
            std::vector<llama_token> toks(text.size() + 8);
            int nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false);
            if (nt < 0) { toks.resize(-nt); nt = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false); }
            toks.resize(nt);
            int n_prompt = (int) toks.size();
            seq = toks;

            // greedy generation on this backend
            llama_memory_clear(mem, true);
            llama_batch b = llama_batch_init(1024, 0, 1);
            b.n_tokens = n_prompt;
            for (int i = 0; i < n_prompt; ++i) {
                b.token[i] = toks[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
                b.logits[i] = (i == n_prompt - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prompt decode failed p%d\n", idx); return 1; }
            int cur = n_prompt;
            int last_logit_idx = n_prompt - 1;
            for (int g = 0; g < n_gen; ++g) {
                float * lg = llama_get_logits_ith(ctx, last_logit_idx);
                int best = 0; float bv = lg[0];
                for (int v = 1; v < n_vocab; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
                if (llama_vocab_is_eog(vocab, best)) break;
                seq.push_back(best);
                b.n_tokens = 1;
                b.token[0] = best; b.pos[0] = cur; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "gen decode failed p%d g%d\n", idx, g); return 1; }
                last_logit_idx = 0;
                cur++;
            }
            llama_batch_free(b);

            std::ofstream sf(seqpath);
            for (size_t i = 0; i < seq.size(); ++i) sf << seq[i] << (i + 1 < seq.size() ? " " : "\n");
            sf.close();
            std::ofstream mf(metapath);
            mf << n_prompt << "\n";
            mf.close();
        } else {
            std::ifstream sf(seqpath);
            llama_token t;
            while (sf >> t) seq.push_back(t);
            if (seq.empty()) { fprintf(stderr, "empty/missing seq %s\n", seqpath); continue; }
        }

        // teacher-forced single-batch dump (all positions request logits)
        int n = (int) seq.size();
        llama_memory_clear(mem, true);
        llama_batch b = llama_batch_init(n, 0, 1);
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) { b.token[i] = seq[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "teacher-force decode failed p%d (n=%d)\n", idx, n); return 1; }

        char binpath[512]; snprintf(binpath, sizeof(binpath), "%s/logits_%s_%d.bin", out_dir.c_str(), tag.c_str(), idx);
        FILE * of = fopen(binpath, "wb");
        int32_t hdr[2] = { n, n_vocab };
        fwrite(hdr, sizeof(int32_t), 2, of);
        for (int i = 0; i < n; ++i) {
            float * lg = llama_get_logits_ith(ctx, i);
            fwrite(lg, sizeof(float), n_vocab, of);
        }
        fclose(of);
        llama_batch_free(b);
        fprintf(stderr, "[%s] prompt %d: n_pos=%d n_vocab=%d -> %s\n", tag.c_str(), idx, n, n_vocab, binpath);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
