// iq2_xxs x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle (GRID codebook + SIGN plane).
//
// Gate: does the compiler-emitted iq2_xxs repack kernel compute the CORRECT
// super-block GRID-of-8 + SIGN-plane dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-super-block block_iq2_xxs exactly as ggml's canonical
// ggml_vec_dot_iq2_xxs_q8_K -- each 32-element sub-block reads 4 uint16 into
// aux32[2]; aux8[0..3] are 4 GRID INDICES into the fixed 256-entry iq2xxs_grid
// (each entry = 8 packed int8 grid bytes); the top nibble aux32[1]>>28 gives the
// per-sub-block scale ls = 2*(aux32[1]>>28)+1; and (aux32[1]>>7*l)&127 selects a
// 7-bit SIGN SELECTOR into ksigns_iq2xs, whose bit j (kmask 1<<j) flips grid[j].
// The block dot is bsum = sum_ib32 ls * (sum_{l,j} q8 * grid[j] * (+-1)), scaled
// by fp16(x.d)*y.d and finally by 0.125.
//
// The EMITTER-MODEL reads the REPACKED block_iq2_xxsx16 and reproduces the
// emitter's EXACT organization: per (ib32, group l, column) it stores a raw grid
// INDEX byte + a raw 7-bit sign SELECTOR byte + a per-(sub-block, column) ls
// scale; the model then GATHERS grid[idx*8+j] from a flat int8 grid table and
// GATHERS a +-1 sign from a flat sign plane signs64[sel*8+j] (exactly the two
// vluxei16 gathers + the vmul-onto-grid sign fold the emitter emits), accumulates
// the i32 sub-block dot, weights it by ls, and sums. The two paths use DIFFERENT
// data layouts, so agreement is strong evidence.
//
// ONE integer certificate is compared BYTE-EXACT (int64 equality):
//   bsum = sum over 8 sub-blocks of ls * (sum over 32 of q8 * grid[j] * sign)
// It is a pure integer (no fp reassociation). The single grid*sign*q8 product
// magnitude (|43*1*127| < 5504) fits i16 but the 32-element sub-block dot and the
// ls-weighted accumulation overflow it, so the emitter (and this model) accumulate
// the sub-block dot in i32 and the ls-weighted block sum in i32; the model tracks
// the running |bsum|.
//
// NEGATIVE CONTROLS perturb the REFERENCE; a mismatch SPIKE proves the
// corresponding iq2_xxs structure was genuinely exercised by the emitter-model:
//   GRID  : the REFERENCE fetches a ROTATED grid entry (idx -> (idx+37)&255) so a
//           DIFFERENT 8-byte grid vector is used. A real grid GATHER diverges from
//           the wrong entry -> the int certificate spikes. Load-bearing "real grid
//           table lookup, NOT fake / NOT a fixed pattern" control.
//   SIGN  : the REFERENCE IGNORES the sign plane (every sign forced +1). A real
//           sign-plane application diverges -> the int certificate spikes.
//   SCALE : the REFERENCE perturbs the per-sub-block ls scale (ls -> ls+2). A real
//           per-sub-block scale weight diverges -> the int certificate spikes.
//
// Build: g++ -O2 -std=c++17 oracle_repack_iq2_xxs.cpp -o /tmp/oracle_iq2xxs && \
//        /tmp/oracle_iq2xxs

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>

#define QK_K 256

// ---- canonical ggml iq2xxs_grid[256] (uint64; each entry = 8 packed int8 grid
// bytes, every byte <= 0x2b < 128 so int8 == uint8 numeric read). ----
static const int64_t iq2xxs_grid[256] = {
    578721382704613384LL, 578721382704613419LL, 578721382704617753LL, 578721382704622344LL,
    578721382704622379LL, 578721382705727513LL, 578721382705731848LL, 578721382706907144LL,
    578721382706907179LL, 578721382706916104LL, 578721382706916139LL, 578721382989826073LL,
    578721382989830408LL, 578721382990940168LL, 578721382990949128LL, 578721382992119833LL,
    578721382992124168LL, 578721383291815944LL, 578721383291815979LL, 578721383291824939LL,
    578721383294109739LL, 578721455719057433LL, 578721455719061768LL, 578721455720171528LL,
    578721455720175897LL, 578721456004270088LL, 578721456306264328LL, 578721456307383048LL,
    578721533028468744LL, 578721533028468779LL, 578721533030762539LL, 578721533615671339LL,
    578740074402285593LL, 578740074402289928LL, 578740074403399688LL, 578740074404579353LL,
    578740074404583688LL, 578740074687498248LL, 578740074687498283LL, 578740074687507208LL,
    578740074689792008LL, 578740074989488153LL, 578740074989492488LL, 578740074990602248LL,
    578740074991786248LL, 578740147416729608LL, 578740147416729643LL, 578740147416738568LL,
    578740147419023368LL, 578740147701946667LL, 578740147704245017LL, 578740148003932168LL,
    578740148005046297LL, 578740224726149913LL, 578740224727255048LL, 578740225011353608LL,
    578740225313347848LL, 578740225315641608LL, 578759865611585544LL, 578759865611589913LL,
    578759865611594504LL, 578759865612704008LL, 578759865613888264LL, 578759865896798233LL,
    578759865896802568LL, 578759865897912328LL, 578759865897912363LL, 578759866198797064LL,
    578759938626033928LL, 578759938911242248LL, 578760015935440939LL, 578760015936559368LL,
    583506457308694553LL, 583506457308698888LL, 583506457309808648LL, 583506457310988313LL,
    583506457593907208LL, 583506457596200968LL, 583506457895901448LL, 583506457897011208LL,
    583506457897015577LL, 583506530323138568LL, 583506530323147528LL, 583506530325432328LL,
    583506530609465352LL, 583506530609474347LL, 583506530910341128LL, 583506607634848008LL,
    583506607917766937LL, 583525149006366728LL, 583525149006375688LL, 583525149008660488LL,
    583525149008664857LL, 583525149291588377LL, 583525149593569288LL, 583525222021933832LL,
    583525222308317227LL, 583525299330222088LL, 583525299331340587LL, 583544940215666713LL,
    583544940215671048LL, 583544940216780808LL, 583544940500879368LL, 583544940802869273LL,
    583545013230110728LL, 583545013230115097LL, 583545013819607048LL, 583545090825848857LL,
    588573006889486344LL, 588573006889486379LL, 588573006889495339LL, 588573007174703368LL,
    588573007176992793LL, 588573007476688904LL, 588573007476688939LL, 588573079906233113LL,
    588573080189152008LL, 588573157213341704LL, 588573157213341739LL, 588591698587158553LL,
    588591698587162888LL, 588591698588272648LL, 588591698872371208LL, 588591698873489707LL,
    588591771601602568LL, 588591771886815257LL, 588591771889113352LL, 588591849499330568LL,
    588611489796467464LL, 588611489798752264LL, 588611490384779528LL, 588611640405530888LL,
    1803700481349388313LL, 1803700481349392648LL, 1803700481350502408LL, 1803700481350511368LL,
    1803700481351682073LL, 1803700481351686408LL, 1803700481634600968LL, 1803700481634609928LL,
    1803700481635719467LL, 1803700481636894728LL, 1803700481936590873LL, 1803700481936595208LL,
    1803700481937704968LL, 1803700554363832328LL, 1803700554366126088LL, 1803700554651338777LL,
    1803700554951034888LL, 1803700554951039257LL, 1803700631673243673LL, 1803700631674357768LL,
    1803700631958465288LL, 1803700631959574827LL, 1803700631960759048LL, 1803719173047060488LL,
    1803719173047069448LL, 1803719173049354248LL, 1803719173634263048LL, 1803719173635386137LL,
    1803719246062618667LL, 1803719246063802632LL, 1803719323370915848LL, 1803738964256360473LL,
    1803738964256364808LL, 1803738964257474568LL, 1803738964541573128LL, 1803738964541577497LL,
    1803739037270804488LL, 1803739037557140232LL, 1803739037558310937LL, 1803739037858007083LL,
    1803739114865432857LL, 1803739115168532488LL, 1808485555953469448LL, 1808485555953478408LL,
    1808485555954583577LL, 1808485555954592537LL, 1808485555955763208LL, 1808485556540672008LL,
    1808485556540680968LL, 1808485628967917832LL, 1808485629253126187LL, 1808485629557414152LL,
    1808485706865641497LL, 1808504248239458312LL, 1808504248239458347LL, 1808504320665594667LL,
    1808504397974997017LL, 1808504398261328136LL, 1808524038860441608LL, 1808524038861555737LL,
    1808524038861564697LL, 1808524039147952392LL, 1808524112160098312LL, 1808524189184305928LL,
    1813552105534265608LL, 1813552105535375368LL, 1813552105819473928LL, 1813552105821776648LL,
    1813552178548705288LL, 1813552178835036441LL, 1813552255859239688LL, 1813552256145623048LL,
    1813570797231933448LL, 1813570797231937817LL, 1813570870247491592LL, 1813570870247491627LL,
    1813570870833584392LL, 1813590588726446123LL, 3100737174032091144LL, 3100737174032091179LL,
    3100737174032100139LL, 3100737174317303833LL, 3100737174619293739LL, 3100737247046539528LL,
    3100737247047658248LL, 3100737247331747848LL, 3100737324357060633LL, 3100755865729763353LL,
    3100755865729767688LL, 3100755865730877448LL, 3100755865730881817LL, 3100755866014976008LL,
    3100755866017269768LL, 3100755938744207368LL, 3100755939029424427LL, 3100755939332528392LL,
    3100756016053627673LL, 3100756016338831368LL, 3100756016341125128LL, 3100775656939063339LL,
    3100775729953511688LL, 3100775807264032793LL, 3105522248636176648LL, 3105522248637286408LL,
    3105522248638470408LL, 3105522248921384968LL, 3105522249225668633LL, 3105522321651734827LL,
    3105522322237818888LL, 3105522399245244697LL, 3105540940333844488LL, 3105540940336138283LL,
    3105540940619061512LL, 3105541013634615321LL, 3105560732130347033LL, 3105560804559882248LL,
    3110588798216964139LL, 3110588798503290888LL, 3110588798804171033LL, 3110588871231417113LL,
    3110588948540819464LL, 3110607489915759368LL, 3110627281410263048LL, 3110627354138384648LL};

// ---- canonical ggml ksigns_iq2xs[128] (each entry is an 8-bit sign mask; bit j
// set => grid[j] negated). ----
static const uint8_t ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139,
    12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23,
    24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163,
    36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187,
    60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71,
    72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83,
    212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235,
    108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119,
    120, 249, 250, 123, 252, 125, 126, 255};

// ---- flat int8 grid byte table (256*8) and +-1 sign plane (128*8), EXACTLY the
// two tables the emitter emits (tcrv_iq2xxs_grid viewed as int8 + the derived
// tcrv_iq2xxs_signs64). ----
static int8_t grid_bytes[256 * 8];
static int8_t signs64[128 * 8];
static void build_tables() {
    for (int e = 0; e < 256; ++e) {
        uint64_t u = (uint64_t)iq2xxs_grid[e];
        for (int j = 0; j < 8; ++j)
            grid_bytes[e * 8 + j] = (int8_t)((u >> (8 * j)) & 0xff);
    }
    for (int s = 0; s < 128; ++s)
        for (int j = 0; j < 8; ++j)
            signs64[s * 8 + j] = (ksigns_iq2xs[s] & (1 << j)) ? -1 : 1;
}

// ---- original per-super-block block_iq2_xxs (ggml-common.h), 66 bytes ----
struct block_iq2_xxs {
    float    d;         // super-block scale (fp16 in ggml; float here does not
                        // affect the byte-exact INTEGER certificate)
    uint16_t qs[QK_K / 8];  // 32 uint16
};

// ---- block_q8_K activation, d(fp32) + 256 int8 quants + 16 int16 bsums ----
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- repacked block_iq2_xxsx16 (block-as-lane convention), 1184 B ----
//   d[16]           @ +0    (32 B, fp16 in the real kernel; float here)
//   ls[8][16]       @ +32   (128 B, per (sub-block, column) scale 2*(hi4)+1)
//   gidx[8][4][16]  @ +160  (512 B, raw grid index aux8[l] per (ib32,l,column))
//   ssel[8][4][16]  @ +672  (512 B, raw 7-bit sign selector per (ib32,l,column))
struct block_iq2_xxsx16 {
    float   d[16];
    int8_t  ls[8][16];
    uint8_t gidx[8][4][16];
    uint8_t ssel[8][4][16];
};

static block_iq2_xxsx16 make_block_iq2_xxsx16(const block_iq2_xxs *in) {
    block_iq2_xxsx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    for (int c = 0; c < 16; ++c)
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t a0 = (uint32_t)in[c].qs[4 * ib32] |
                          ((uint32_t)in[c].qs[4 * ib32 + 1] << 16);
            uint32_t a1 = (uint32_t)in[c].qs[4 * ib32 + 2] |
                          ((uint32_t)in[c].qs[4 * ib32 + 3] << 16);
            out.ls[ib32][c] = (int8_t)(2 * (a1 >> 28) + 1);
            for (int l = 0; l < 4; ++l) {
                out.gidx[ib32][l][c] = (uint8_t)((a0 >> (8 * l)) & 0xff);
                out.ssel[ib32][l][c] = (uint8_t)((a1 >> (7 * l)) & 127);
            }
        }
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_GRID = 1, DQ_SIGN = 2, DQ_SCALE = 3 };

// ---- REFERENCE (ggml canonical ggml_vec_dot_iq2_xxs_q8_K). Returns the integer
// certificate bsum = sum_ib32 ls*(sum_{l,j} q8 * grid[j] * sign). q8 is the
// 256-int8 activation quant array for this super-block. ----
static int64_t ref_block(const block_iq2_xxs *x, const int8_t *q8, DqMode mode) {
    int64_t bsum = 0;
    const uint16_t *q2 = x->qs;
    int q8pos = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t a0 = (uint32_t)q2[4 * ib32] | ((uint32_t)q2[4 * ib32 + 1] << 16);
        uint32_t a1 =
            (uint32_t)q2[4 * ib32 + 2] | ((uint32_t)q2[4 * ib32 + 3] << 16);
        uint8_t aux8[4] = {(uint8_t)(a0 & 0xff), (uint8_t)((a0 >> 8) & 0xff),
                           (uint8_t)((a0 >> 16) & 0xff),
                           (uint8_t)((a0 >> 24) & 0xff)};
        int64_t ls = 2 * (a1 >> 28) + 1;
        if (mode == DQ_SCALE) ls += 2;  // SCALE control: perturb the ls scale
        int64_t sumi = 0;
        for (int l = 0; l < 4; ++l) {
            int idx = aux8[l];
            if (mode == DQ_GRID) idx = (idx + 37) & 255;  // GRID: wrong entry
            const int8_t *grid = &grid_bytes[idx * 8];
            uint8_t signs = ksigns_iq2xs[(a1 >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j) {
                int sgn = (mode == DQ_SIGN)
                              ? 1  // SIGN control: ignore the sign plane
                              : ((signs & (1 << j)) ? -1 : 1);
                sumi += (int64_t)q8[q8pos] * grid[j] * sgn;
                q8pos++;
            }
        }
        bsum += sumi * ls;
    }
    return bsum;
}

// ---- EMITTER-MODEL for column c of a repacked group, dotted with a per-column
// activation quant accessor act(pos). Reproduces the emitter's EXACT lane-wise
// grid-GATHER + sign-plane-GATHER organization (i32 sub-block dot, ls-weighted
// i32 block accumulator). ----
template <typename ActFn>
static int64_t mine_col(const block_iq2_xxsx16 *b, int c, ActFn act,
                        int64_t &maxAbsBsum) {
    int64_t bsum = 0;
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        int64_t sumi = 0;
        for (int grp = 0; grp < 4; ++grp) {
            int idx = b->gidx[ib32][grp][c];
            int sel = b->ssel[ib32][grp][c];
            for (int jj = 0; jj < 8; ++jj) {
                int k = ib32 * 32 + grp * 8 + jj;
                int gb = grid_bytes[idx * 8 + jj];     // REAL grid GATHER
                int sgn = signs64[sel * 8 + jj];       // REAL sign-plane GATHER
                int w = gb * sgn;
                sumi += (int64_t)act(k) * w;
            }
        }
        bsum += sumi * (int64_t)b->ls[ib32][c];
        if (std::llabs(bsum) > maxAbsBsum) maxAbsBsum = std::llabs(bsum);
    }
    return bsum;
}

static void build_iq2_block(block_iq2_xxs *x, std::mt19937 &rng, int col,
                            int blk) {
    std::uniform_int_distribution<int> gi(0, 255);   // grid index
    std::uniform_int_distribution<int> si(0, 127);   // sign selector
    std::uniform_int_distribution<int> sc(0, 15);    // 4-bit scale field
    x->d = 0.010f + 0.0006f * ((col + 3 * blk) % 11);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t a0 = (uint32_t)gi(rng) | ((uint32_t)gi(rng) << 8) |
                      ((uint32_t)gi(rng) << 16) | ((uint32_t)gi(rng) << 24);
        uint32_t a1 = (uint32_t)si(rng) | ((uint32_t)si(rng) << 7) |
                      ((uint32_t)si(rng) << 14) | ((uint32_t)si(rng) << 21) |
                      ((uint32_t)sc(rng) << 28);
        x->qs[4 * ib32 + 0] = (uint16_t)(a0 & 0xffff);
        x->qs[4 * ib32 + 1] = (uint16_t)(a0 >> 16);
        x->qs[4 * ib32 + 2] = (uint16_t)(a1 & 0xffff);
        x->qs[4 * ib32 + 3] = (uint16_t)(a1 >> 16);
    }
}

static void build_q8_K_block(block_q8_K *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int i = 0; i < QK_K; ++i) {
        int dc = ((i * 7 + blk * 3) % 21) - 10;
        int v = q8d(rng) + dc;
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        a->qs[i] = (int8_t)v;
    }
    for (int i = 0; i < QK_K / 16; ++i) a->bsums[i] = 0;  // UNREAD by iq2_xxs
}

int main() {
    build_tables();
    printf("# iq2_xxs x q8_K 16x1-REPACKED oracle (GRID codebook + SIGN plane).\n");
    printf("# BYTE-EXACT integer certificate bsum (grid*sign*q8, ls-weighted):\n");
    printf("#   reference (original iq2_xxs) vs emitter-model (block_iq2_xxsx16).\n");
    printf("#   final *s = 0.125f * fp16(x.d)*y.d * bsum (fp, reported as norm).\n\n");

    std::mt19937 rng(20260708u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 256, 1},   {16, 4096, 1}, {32, 256, 4},  {32, 2560, 4},
        {256, 256, 1},  {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int64_t globalMaxBsum = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_K, ng = nc / 16;

        std::vector<block_iq2_xxs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq2_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        std::vector<block_iq2_xxsx16> vx((size_t)ng * nb);
        std::vector<block_iq2_xxs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq2_xxsx16(tmp16.data());
            }

        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int64_t maxBsum = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r)
            for (int g = 0; g < ng; ++g)
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_iq2_xxs *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_block(xo, a->qs, DQ_NORMAL);
                        const block_iq2_xxsx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_col(b, col, actFn, maxBsum);
                        if (ri != mi) mismatch++;
                        refF  += 0.125 * (double)xo->d * (double)a->d * (double)ri;
                        mineF += 0.125 * (double)b->d[col] * (double)a->d *
                                 (double)mi;
                    }
                    double e = std::fabs(refF - mineF);
                    if (e > maxAbsErr) maxAbsErr = e;
                    sumsqRef += refF * refF;
                }
        int ncnt = nr * nc;
        double rms = std::sqrt(sumsqRef / ncnt);
        double norm = rms > 0 ? maxAbsErr / rms : maxAbsErr;
        if (norm > worstNorm) worstNorm = norm;
        if (maxBsum > globalMaxBsum) globalMaxBsum = maxBsum;
        bool bug = (mismatch != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : int-mismatch=%-6lld  "
               "maxAbsBsum=%-9lld  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, (long long)maxBsum, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved block_q8_Kx4 addressing: pack 4 plain q8_K rows into
    // one block_q8_Kx4 (d[4] fp32 @+0, qs @+16, byte for flat pos p / column c at
    // 16 + p*4 + c) and confirm the emitter-model's interleaved read is
    // BYTE-EXACT. ----
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq2_xxs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq2_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_iq2_xxsx16> vx((size_t)ng * nb);
        std::vector<block_iq2_xxs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq2_xxsx16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        // interleaved qs plane: [nb][256*4], byte at pos*4 + c.
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] =
                        row[(size_t)c * nb + l].qs[p];
        long long mism = 0, total = 0;
        int64_t dummy = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ri = ref_block(&orig[(size_t)gcol * nb + l],
                                               row[(size_t)c * nb + l].qs,
                                               DQ_NORMAL);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int64_t mi = mine_col(&vx[(size_t)g * nb + l], col, actFn,
                                              dummy);
                        total++;
                        if (ri != mi) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16 pos*4+c): int-mismatch = "
               "%lld / %lld  %s\n",
               mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS on one shape (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large mismatch fraction\n"
           "# proves the iq2_xxs feature is genuinely exercised by the emitter-model):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_iq2_xxs> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_iq2_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_iq2_xxsx16> vx((size_t)ng * nb);
        std::vector<block_iq2_xxs> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_iq2_xxsx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8_K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        struct Ctl { const char *name; DqMode mode; };
        Ctl ctls[3] = {
            {"GRID  (rotate grid index +37, wrong entry)  [bsum]", DQ_GRID},
            {"SIGN  (ignore sign plane, all +1)           [bsum]", DQ_SIGN},
            {"SCALE (perturb per-sub-block ls +2)         [bsum]", DQ_SCALE},
        };
        for (auto &ct : ctls) {
            long long total = 0, mism = 0;
            int64_t dummy = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int64_t rI = ref_block(&orig[(size_t)gcol * nb + l],
                                                   a->qs, ct.mode);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mI = mine_col(&vx[(size_t)g * nb + l], col,
                                                  actFn, dummy);
                            total++;
                            if (rI != mI) mism++;
                        }
                    }
            printf("  %-50s : perturbed-ref vs emitter mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   ct.name, mism, total, 100.0 * mism / total,
                   mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i32_BSUM %lld   VERDICT %s\n",
           worstNorm, (long long)globalMaxBsum,
           anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
