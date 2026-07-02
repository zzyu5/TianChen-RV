//===- RVVIQ2XSBlockDotSourceFrontDoor.cpp ------------------------------===//
//
// Track B auto-lowering, the SUPER-BLOCK GRID-CODEBOOK rung -- the LAST iq* member of
// the deep IQ tail's super-block-codebook bucket, the SIBLING of iq2_xxs. The
// COMPILER auto-constructs the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE attr-less tcrv_rvv.iq2_xs_q8_k_block_dot op, from a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq2_xs_q8_K` OPERATOR IDENTITY,
// instead of a per-kernel hand-authored super-block grid-codebook block-dot emitter
// input.
//
// WHAT THIS RUNG COMBINES: iq2_xs REUSES iq2_xxs's packed-GRID + separate-SIGN-PLANE
// mechanism -- each 9-bit weight index reads a packed uint64 GRID entry (8 int8 grid
// values) via an INDEXED pointer-arith lookup (grid_table + idx*8, NOT a vrgather --
// the 4096-byte grid cannot broadcast into a vreg), the per-element SIGN is read from
// a separate SIGN PLANE ksigns_iq2xs[128] gated by the kmask {1<<j} selector, and the
// per-group dot folds in the INTEGER domain -- wrapped in the q6_K/q4_K-style
// super-block structure (QK_K==256, 8 sub-blocks of 32, the q8_K activation, the
// integer bsum accumulation, the per-super-block fp32 fold, the trailing 0.125f
// factor). The genuinely-new super-block pieces vs iq2_xxs -- (a) a 512-entry GRID
// with the 9-bit index read DIRECTLY from each uint16 qs word (w&511 / w>>9, NO aux1
// packing) and (b) an EXPLICIT per-sub-block scales[8] byte stream (each byte two
// 4-bit scales: ls1=2*(sc&0xf)+1 for groups 0,1 / ls2=2*(sc>>4)+1 for groups 2,3, two
// sumi resets/two bsum contributions per sub-block) -- are first-class STRUCTURE
// inside the op + its existing iq2_xs emitter. So the front door does NOT hand-roll
// any of it; it supplies (a) the iq2_xs super-block-format CONSTANTS as the typed
// integer attrs the verifier pins AND (b) the two GRID-codebook tables as the
// structural array attrs the verifier pins (grid to size 512 int64, ksigns to size
// 128 int32).
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY (the bucket verdict). The super-block
// grid-codebook op takes the EXISTING super-block monolithic route family (shared
// with q4_K / iq4_xs). The grid + ksigns are OP attrs consumed by the emitter, NOT a
// route-family concern: the emission plan (buildMonolithicBlockDotEmissionPlan) and
// the target export candidate validator both key ONLY off the op name -> route family
// + the kind/scale_model attrs + the ordered ABI roles; neither reads the grid. So
// the shared SuperBlock family + the grid/ksigns-attr stamping COMPOSE cleanly: one
// table row (SuperBlock, the 4-role ggml vec_dot ABI) + this front door, zero new
// mechanism.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP. iq2_xs is NOT in any schedule-descriptor
// autotuner. UNLIKE iq4_nl/iq4_xs (whose 16-entry codebook vrgather makes m1 a
// broadcast-table VLMAX legality fact), the iq2_xs grid lookup is an INDEXED gather
// over a pointer (grid_table + idx*8) at the 8-lane group width, so the integer core
// does NOT pin an m1 table anchor -- m1 is kept on setvl/with_vl and the widening
// reduce result, but no shape knob is stamped and there is no VLEN128-vs-VLEN256
// byte-flip. The constructed attr-less op lowers DIRECTLY at the iq2_xs emitter's
// shape, VLEN-independent. The NEW content is the auto-CONSTRUCTION feeding the
// existing iq2_xs emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVIQ2XSBlockDotSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Transforms/VariantMaterialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

namespace tcrvexec = ::tianchenrv::tcrv::exec;
namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

// The DISTINCT marker the source module carries to route to THIS iq2_xs super-block
// grid-codebook front door (NOT the MVP/dequant/q4_0/q8_0/iq4_nl/q4_K/iq4_xs
// markers). Each front-door pass checks its own marker and early-returns on a
// mismatch, so the passes are mutually exclusive and the sibling lits are
// byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_iq2_xs_q8_K_block_dot_source");
constexpr llvm::StringLiteral kSeedAttrName("tcrv_rvv.lowering_seed");

constexpr llvm::StringLiteral kRVVCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kFallbackCapabilitySymbol("scalar_fallback");
constexpr llvm::StringLiteral kConservativeFallbackCapabilityKind("fallback");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");
constexpr llvm::StringLiteral kFallbackRoleAttrName("fallback_role");
constexpr llvm::StringLiteral kSourceKernelBoundaryAttrName("source_kernel");
constexpr llvm::StringLiteral kSelectedVariantAttrName("selected_variant");
constexpr llvm::StringLiteral kSelectedPathRoleAttrName("selected_path_role");
constexpr llvm::StringLiteral kStatusAttrName("status");
constexpr llvm::StringLiteral kRequiredCapabilitiesAttrName(
    "required_capabilities");
constexpr llvm::StringLiteral kRVVConstructionProtocolAttrName(
    "rvv_construction_protocol");
constexpr llvm::StringLiteral kRVVConstructionProtocol(
    "extension-family-construction-protocol.v1");
constexpr llvm::StringLiteral kDispatchPolicy(
    "rvv-iq2-xs-q8-k-block-dot-source-front-door-case");

// The ggml iq2_xs super-block-format facts (ggml-common.h): QK_K == 256, 8 sub-blocks
// of 32 elements, block_iq2_xs stride 74 (fp16 d @0 | 32 uint16 qs words @2 | 8 uint8
// scales @66), block_q8_K stride 292 (fp32 d @0 | 256 int8 qs @4 | 16 int16 bsums
// @260 UNUSED -- iq2_xs is symmetric, no min term). These are iq2_xs CONSTANTS the
// front door supplies (not derivable from a generic source) and are the exact values
// the tcrv_rvv.iq2_xs_q8_k_block_dot verifier pins. NOTE casing: the op kind is the
// LOWER-k form "ggml_iq2_xs_q8_k_block_dot"; the kernel/marker/variant symbols carry
// the canonical ggml UPPER-K "q8_K".
constexpr std::int64_t kQK = 256;
constexpr std::int64_t kSubBlock = 32;
constexpr std::int64_t kWeightBlockStride = 74;
constexpr std::int64_t kActivationBlockStride = 292;
constexpr std::int64_t kWeightDByteOffset = 0;
constexpr std::int64_t kWeightQsByteOffset = 2;
constexpr std::int64_t kWeightScalesByteOffset = 66;
constexpr std::int64_t kActivationDByteOffset = 0;
constexpr std::int64_t kActivationQuantByteOffset = 4;

// ggml's iq2xs_grid[512]: the 512-entry packed-uint64 GRID codebook the 9-bit weight
// index (w & 511) looks up (each uint64 entry encodes 8 int8 grid values, read via a
// (const int8_t *) cast) -- the load-bearing structural fact of the iq2_xs GRID
// class. The verifier pins its size to EXACTLY 512. Supplied here as the iq2_xs
// constant a generic source could not derive, consumed structurally into the realized
// body (a `static const int64_t[512]` decl + the indexed vluxei16 gather), NOT a
// string plan read. Carried as int64 rendering ggml's exact uint64 literals.
constexpr std::array<std::int64_t, 512> kIQ2XSGrid = {
    578721382704613384, 578721382704613419, 578721382704617753, 578721382704622344,
    578721382704622379, 578721382705727513, 578721382705731848, 578721382705731883,
    578721382705736473, 578721382706907144, 578721382706907179, 578721382706911513,
    578721382706916104, 578721382989826073, 578721382989830408, 578721382989830443,
    578721382989835033, 578721382990940168, 578721382990940203, 578721382990944537,
    578721382990949128, 578721382992119833, 578721382992124168, 578721383291815944,
    578721383291815979, 578721383291820313, 578721383291824904, 578721383292930073,
    578721383292934408, 578721383292939033, 578721383294109704, 578721455719057433,
    578721455719061768, 578721455719061803, 578721455719066393, 578721455720171528,
    578721455720171563, 578721455720175897, 578721455720180488, 578721455720180523,
    578721455721351193, 578721455721355528, 578721456004270088, 578721456004270123,
    578721456004274457, 578721456004279048, 578721456005384217, 578721456005388552,
    578721456006563848, 578721456006572808, 578721456306259993, 578721456306264328,
    578721456307374088, 578721533028468744, 578721533028468779, 578721533028473113,
    578721533028477704, 578721533029582873, 578721533029587208, 578721533030762504,
    578721533313681433, 578721533313685768, 578721533314795528, 578721533314799897,
    578721533615671304, 578721533615680299, 578740074402285593, 578740074402289928,
    578740074402289963, 578740074402294553, 578740074403399688, 578740074403399723,
    578740074403404057, 578740074403408648, 578740074404579353, 578740074404583688,
    578740074687498248, 578740074687498283, 578740074687502617, 578740074687507208,
    578740074688612377, 578740074688616712, 578740074688616747, 578740074689792008,
    578740074989488153, 578740074989492488, 578740074990602248, 578740147416729608,
    578740147416729643, 578740147416733977, 578740147416738568, 578740147417843737,
    578740147417848072, 578740147419023368, 578740147701942297, 578740147701946632,
    578740147703056392, 578740147704236057, 578740148003932168, 578740224726140953,
    578740224726145288, 578740224727255048, 578740224728439083, 578740225011353608,
    578740225011353643, 578740225313347848, 578759865611585544, 578759865611585579,
    578759865611589913, 578759865611594504, 578759865611594539, 578759865612699673,
    578759865612704008, 578759865613879304, 578759865613883673, 578759865896798233,
    578759865896802568, 578759865897912328, 578759865897921288, 578759866198788104,
    578759866201081864, 578759866201090859, 578759938626029593, 578759938626033928,
    578759938627143688, 578759938911242248, 578759939213232153, 578759939213241113,
    578760015935440904, 578760015937734664, 578760015937743624, 578760016523761963,
    578760016524937224, 583506457308694553, 583506457308698888, 583506457308698923,
    583506457308703513, 583506457309808648, 583506457309808683, 583506457309813017,
    583506457309817608, 583506457310988313, 583506457310992648, 583506457593907208,
    583506457593907243, 583506457593911577, 583506457593916168, 583506457595021337,
    583506457595025672, 583506457596200968, 583506457596209963, 583506457895897113,
    583506457895901448, 583506457897011208, 583506530323138568, 583506530323138603,
    583506530323142937, 583506530323147528, 583506530324252697, 583506530324257032,
    583506530325432328, 583506530608351257, 583506530608355592, 583506530609465352,
    583506530910341128, 583506530911459592, 583506530911459627, 583506607632549913,
    583506607632554248, 583506607632554283, 583506607633664008, 583506607917762568,
    583506607920056328, 583525149006366728, 583525149006366763, 583525149006371097,
    583525149006375688, 583525149007480857, 583525149007485192, 583525149008660488,
    583525149291579417, 583525149291583752, 583525149291588377, 583525149292693512,
    583525149293877512, 583525149593569288, 583525222020810777, 583525222020815112,
    583525222021924872, 583525222306023432, 583525299330222088, 583525299331340552,
    583525299615443737, 583544940215666713, 583544940215671048, 583544940216780808,
    583544940216780843, 583544940500879368, 583544940501997832, 583544940802873643,
    583545013230110728, 583545013230115097, 583545013517621547, 583545090825848857,
    583545091129027353, 588573006889486344, 588573006889486379, 588573006889490713,
    588573006889495304, 588573006889495339, 588573006890600473, 588573006890604808,
    588573006891780104, 588573007174699033, 588573007174703368, 588573007175813128,
    588573007476688904, 588573007478982664, 588573079903930393, 588573079903934728,
    588573079905044488, 588573080189143048, 588573080189152008, 588573080191441177,
    588573157213341704, 588573157215635499, 588573157800544264, 588573157802846984,
    588591698587158553, 588591698587162888, 588591698588272648, 588591698589461273,
    588591698872371208, 588591771601602568, 588591771886815257, 588591771887929387,
    588591772189928217, 588591848911013913, 588591848912137003, 588591849500514603,
    588611489796458504, 588611489796467464, 588611489796467499, 588611489798752264,
    588611490082789657, 588611490383670024, 588611490385954859, 588611563098417928,
    588611563399219208, 588611640120322824, 588611640122607624, 588611640707516459,
    588611640707525384, 588611640707525419, 1803700481349388313, 1803700481349392648,
    1803700481349392683, 1803700481349397273, 1803700481350502408, 1803700481350502443,
    1803700481350506777, 1803700481350511368, 1803700481351682073, 1803700481351686408,
    1803700481634600968, 1803700481634601003, 1803700481634605337, 1803700481634609928,
    1803700481634609963, 1803700481635715097, 1803700481635719432, 1803700481636894728,
    1803700481636899097, 1803700481936590873, 1803700481936595208, 1803700481937704968,
    1803700554363832328, 1803700554363832363, 1803700554363836697, 1803700554363841288,
    1803700554364946457, 1803700554364950792, 1803700554366126088, 1803700554649045017,
    1803700554649049352, 1803700554650159112, 1803700554951034888, 1803700554951039257,
    1803700554953328683, 1803700631673243673, 1803700631673248008, 1803700631674357768,
    1803700631674357803, 1803700631675546393, 1803700631958456328, 1803719173047060488,
    1803719173047060523, 1803719173047064857, 1803719173047069448, 1803719173048174617,
    1803719173048178952, 1803719173048183577, 1803719173049354248, 1803719173332273177,
    1803719173332277512, 1803719173333387272, 1803719173634263048, 1803719173635381512,
    1803719246061504537, 1803719246061508872, 1803719246062618632, 1803719246063802632,
    1803719246346717192, 1803719246649830187, 1803719323370915848, 1803719323370924843,
    1803719323656132872, 1803719323657242632, 1803738964256360473, 1803738964256364808,
    1803738964257474568, 1803738964541573128, 1803738964541577497, 1803738964542691592,
    1803738964543866923, 1803739037270804488, 1803739037271918617, 1803739037556021512,
    1803739037557131272, 1803739037558319897, 1803739114580220168, 1808485555953469448,
    1808485555953469483, 1808485555953473817, 1808485555953478408, 1808485555954583577,
    1808485555954587912, 1808485555955763208, 1808485555955772168, 1808485556238682137,
    1808485556238686472, 1808485556239796232, 1808485556540672008, 1808485628967913497,
    1808485628967917832, 1808485628969027592, 1808485628969031961, 1808485629253126152,
    1808485629253126187, 1808485706277324808, 1808485706562541832, 1808485706866830123,
    1808504247651141657, 1808504247651145992, 1808504247652255752, 1808504247653435417,
    1808504247936354312, 1808504247938648072, 1808504248238344217, 1808504248240637977,
    1808504320665585672, 1808504320665594632, 1808504321252788232, 1808504321252797192,
    1808504397977290777, 1808504398262512392, 1808504398564493337, 1808524038860441608,
    1808524038861560072, 1808524039145654297, 1808524039146768392, 1808524039448767257,
    1808524111876008747, 1808524112160098312, 1808524112160098347, 1808524189771503897,
    1813552105534261273, 1813552105534265608, 1813552105535375368, 1813552105819473928,
    1813552105820592392, 1813552105821767723, 1813552106121468203, 1813552106123766553,
    1813552178548705288, 1813552255860414728, 1813552256143338283, 1813552256446433323,
    1813570797231933448, 1813570797233051947, 1813570870247491592, 1813570870531590152,
    1813570870531594521, 1813570870835878152, 1813590588441233433, 1813590588728748843,
    1813590661457975577, 1813590738765093163, 1813590739051419912, 1813590739052595243,
    3100737174032091144, 3100737174032091179, 3100737174032095513, 3100737174032100104,
    3100737174033205273, 3100737174033209608, 3100737174034384904, 3100737174034393899,
    3100737174317303833, 3100737174317308168, 3100737174318417928, 3100737174619293704,
    3100737174619293739, 3100737174621596424, 3100737174621596459, 3100737247046535193,
    3100737247046539528, 3100737247046539563, 3100737247047649288, 3100737247331747848,
    3100737247332861977, 3100737247332870937, 3100737324355946504, 3100737324358240264,
    3100737324943149064, 3100737324943149099, 3100737324945442824, 3100737324945451784,
    3100755865729763353, 3100755865729767688, 3100755865730877448, 3100755865730877483,
    3100755865730881817, 3100755866014976008, 3100755866017269768, 3100755866316974873,
    3100755938744207368, 3100755939029424392, 3100755939333708057, 3100756016054741768,
    3100756016341134123, 3100775656939063304, 3100775656939072264, 3100775656941361433,
    3100775657225399083, 3100775657526265864, 3100775657526265899, 3100775657528568584,
    3100775729953511723, 3100775807265212459, 3100775807850121224, 3100775807850130184,
    3100775807851239723, 3100775807852423944, 3105522248636172313, 3105522248636176648,
    3105522248637286408, 3105522248921384968, 3105522248922503467, 3105522249223379208,
    3105522321650616328, 3105522321652910123, 3105522321938127112, 3105522399246358827,
    3105522399547239193, 3105540940333844488, 3105540940333848857, 3105540940619061512,
    3105540940620171272, 3105540940620180232, 3105541013350591257, 3105541013936605192,
    3105541013936605227, 3105541090942912537, 3105560731829471257, 3105560732132645163,
    3105560804842810137, 3105560881868118297, 3105560882154506248, 3110588798216964104,
    3110588798216964139, 3110588798216973064, 3110588798216973099, 3110588798219257864,
    3110588798219266859, 3110588798806460424, 3110588871517734937, 3110588871517743897,
    3110588871820908843, 3110588948540819464, 3110588948540819499, 3110588948540828424,
    3110588948543122219, 3110588949128022024, 3110588949130315784, 3110607490199848968,
    3110607490502957337, 3110607640526002457, 3110607640826817288, 3110627281123945259,
    3110627281126230024, 3110627281126230059, 3110627281126238984, 3110627281713432584,
    3110627281713441544, 3110627354138384648, 3110627354725587208, 3110627354725587243,
    3110627431450094344, 3110627431450094379, 3110627432036108313, 3110627432037296939};

// ggml's ksigns_iq2xs[128]: the 128-entry SIGN PLANE the 7-bit sign selector (w >> 9)
// indexes -- byte b of selector j flips the grid sign when (ksigns[j] & (1<<b)). The
// verifier pins its size to EXACTLY 128; the entries reach 255 (beyond int8), so they
// are carried in an i32 attr to represent them losslessly. The emitter DERIVES its
// expanded signs64[1024] table (8 int8 signs per selector) from this ksigns plane;
// there is NO separate signs64 op-attr the front door stamps.
constexpr std::array<std::int32_t, 128> kIQ2XSKsigns = {
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

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml IQ2_XS x Q8_K super-block grid-codebook block-dot source "
                     "front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the super-block grid-codebook dot has no
//     compact generic vector form, so recognition is by the vec_dot ABI roles,
//     not by a straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct IQ2XSBlockDotSourceMatch {
  mlir::func::FuncOp func;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
}

bool isRank1F32MemRef(mlir::Type type) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.getElementType().isF32();
}

// Match the ggml `ggml_vec_dot_iq2_xs_q8_K` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the super-block grid-codebook block-dot intent marker -- the WHAT is
// the operator identity (iq2_xs weight x q8_K activation super-block dot-product ->
// fp32 out), NOT a generic dataflow body. The super-block loop + the explicit int4
// scale extraction + the indexed grid/signs gather + the integer bsum fold are
// iq2_xs STRUCTURE the constructed op carries; the source func body is the bounded
// intent shell (it may be a bare `return`).
mlir::FailureOr<IQ2XSBlockDotSourceMatch>
matchIQ2XSBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, iq2_xs weight memref<?xi8>, q8_K "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "iq2_xs weight rank-1 i8 memref, q8_K activation rank-1 i8 memref");

  return IQ2XSBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.iq2_xs_q8_k_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). NO shape knob: iq2_xs is
//     NOT in any schedule autotuner, and the indexed grid gather does NOT pin an m1
//     table anchor (the 4096-byte grid cannot broadcast), so the constructed op
//     lowers at the iq2_xs emitter's shape. This is byte-identical to the
//     hand-authored iq2_xs block-dot emitter input.
//===----------------------------------------------------------------------===//

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

void createCapability(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef symbol, llvm::StringRef id,
                      llvm::StringRef kind) {
  mlir::OperationState state(loc, tcrvexec::CapabilityOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("id", builder.getStringAttr(id));
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("status", builder.getStringAttr("available"));
  (void)builder.create(state);
}

mlir::ArrayAttr createRequires(mlir::OpBuilder &builder, llvm::StringRef symbol) {
  return builder.getArrayAttr({symbolRef(builder, symbol)});
}

tcrvrvv::PolicyAttr createAgnosticPolicy(mlir::OpBuilder &builder) {
  return tcrvrvv::PolicyAttr::get(builder.getContext(),
                                  tcrvrvv::TailPolicy::Agnostic,
                                  tcrvrvv::MaskPolicy::Agnostic);
}

mlir::Value createRuntimeABIValue(mlir::OpBuilder &builder, mlir::Location loc,
                                  llvm::StringRef role, llvm::StringRef cName,
                                  llvm::StringRef cType, llvm::StringRef purpose,
                                  mlir::Type resultType) {
  mlir::OperationState state(loc,
                             tcrvrvv::RuntimeABIValueOp::getOperationName());
  state.addAttribute("role", builder.getStringAttr(role));
  state.addAttribute("c_name", builder.getStringAttr(cName));
  state.addAttribute("c_type", builder.getStringAttr(cType));
  state.addAttribute("ownership",
                     builder.getStringAttr("target-export-abi-owned"));
  state.addAttribute("purpose", builder.getStringAttr(purpose));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

tcrvrvv::SetVLOp createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value n, std::int64_t sew,
                             llvm::StringRef lmul, tcrvrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, tcrvrvv::SetVLOp::getOperationName());
  state.addOperands(n);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addTypes(tcrvrvv::VLType::get(builder.getContext()));
  return llvm::cast<tcrvrvv::SetVLOp>(builder.create(state));
}

tcrvrvv::WithVLOp createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value vl, std::int64_t sew,
                               llvm::StringRef lmul, tcrvrvv::PolicyAttr policy,
                               llvm::StringRef kernelName,
                               llvm::StringRef selectedVariantSymbol,
                               mlir::ArrayAttr requires) {
  mlir::OperationState state(loc, tcrvrvv::WithVLOp::getOperationName());
  state.addOperands(vl);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addAttribute(kSourceKernelBoundaryAttrName,
                     builder.getStringAttr(kernelName));
  state.addAttribute(kSelectedVariantAttrName,
                     symbolRef(builder, selectedVariantSymbol));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(kSelectedPathRoleAttrName,
                     builder.getStringAttr(stringifyVariantEmissionRole(
                         VariantEmissionRole::DispatchCase)));
  state.addAttribute(kStatusAttrName,
                     builder.getStringAttr("selected-lowering-boundary"));
  state.addAttribute(kRequiredCapabilitiesAttrName, requires);
  state.addAttribute(kRVVConstructionProtocolAttrName,
                     builder.getStringAttr(kRVVConstructionProtocol));
  state.addRegion();
  auto withVL = llvm::cast<tcrvrvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

// The ATTR-LESS ggml IQ2_XS x Q8_K super-block grid-codebook dot-product op: the
// bounded WHAT (kind, scale model, super-block-format facts, the 512-entry packed
// grid, the 128-entry ksigns sign plane) is stamped, but NO shape knob -- iq2_xs is
// NOT in any schedule autotuner and the indexed grid gather does not pin an m1 table
// anchor, so the op lowers at the iq2_xs emitter's shape. The super-block loop + the
// explicit int4 scale extraction + the indexed grid/signs gather + the widening
// product/reduce + the integer bsum fold are first-class STRUCTURE inside this op.
// The grid + ksigns are supplied here as the structural array attrs the verifier pins
// (grid to size 512 int64, ksigns to size 128 int32).
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc,
                             tcrvrvv::GgmlBlockDotIQ2XSQ8KOp::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_iq2_xs_q8_k_block_dot"));
  state.addAttribute(
      "scale_model",
      builder.getStringAttr(
          "per-half-int4-explicit-scales-grid-codebook-int-domain"));
  state.addAttribute("qk", builder.getI64IntegerAttr(kQK));
  state.addAttribute("sub_block", builder.getI64IntegerAttr(kSubBlock));
  state.addAttribute("weight_block_stride",
                     builder.getI64IntegerAttr(kWeightBlockStride));
  state.addAttribute("activation_block_stride",
                     builder.getI64IntegerAttr(kActivationBlockStride));
  state.addAttribute("weight_d_byte_offset",
                     builder.getI64IntegerAttr(kWeightDByteOffset));
  state.addAttribute("weight_qs_byte_offset",
                     builder.getI64IntegerAttr(kWeightQsByteOffset));
  state.addAttribute("weight_scales_byte_offset",
                     builder.getI64IntegerAttr(kWeightScalesByteOffset));
  state.addAttribute("activation_d_byte_offset",
                     builder.getI64IntegerAttr(kActivationDByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("grid",
                     builder.getDenseI64ArrayAttr(llvm::ArrayRef<std::int64_t>(
                         kIQ2XSGrid.data(), kIQ2XSGrid.size())));
  state.addAttribute("ksigns",
                     builder.getDenseI32ArrayAttr(llvm::ArrayRef<std::int32_t>(
                         kIQ2XSKsigns.data(), kIQ2XSKsigns.size())));
  state.addTypes(tcrvrvv::VectorType::get(builder.getContext(),
                                          builder.getI32Type(), "m1"));
  return builder.create(state)->getResult(0);
}

tcrvexec::VariantOp createVariant(mlir::OpBuilder &builder, mlir::Location loc,
                                  llvm::StringRef selectedVariantSymbol,
                                  mlir::ArrayAttr requires,
                                  tcrvrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, tcrvexec::VariantOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(selectedVariantSymbol));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(kRequiresAttrName, requires);
  state.addAttribute("tcrv_rvv.policy", policy);
  state.addRegion();
  auto variant = llvm::cast<tcrvexec::VariantOp>(builder.create(state));
  variant.getBody().emplaceBlock();
  return variant;
}

mlir::LogicalResult createConservativeFallbackCapability(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Operation *failOp,
    const ExtensionPluginRegistry &registry, llvm::StringRef capabilitySymbol) {
  llvm::SmallVector<PluginCapability, 1> fallbackCapabilities;
  registry.collectCapabilitiesByKind(kConservativeFallbackCapabilityKind,
                                     fallbackCapabilities);
  if (fallbackCapabilities.size() != 1)
    return fail(failOp,
                llvm::Twine("source front door requires exactly one "
                            "plugin-declared conservative-fallback capability "
                            "(kind '") +
                    kConservativeFallbackCapabilityKind + "'); found " +
                    llvm::Twine(fallbackCapabilities.size()));
  const PluginCapability &fallbackCapability = fallbackCapabilities.front();
  createCapability(builder, loc, capabilitySymbol, fallbackCapability.getID(),
                   fallbackCapability.getKind());
  return mlir::success();
}

mlir::FailureOr<std::string> materializeConservativeFallbackVariantViaPlugin(
    mlir::OpBuilder &builder, tcrvexec::KernelOp kernel,
    mlir::Operation *highLevelOp, const ExtensionPluginRegistry &registry,
    llvm::StringRef fallbackVariantSymbol) {
  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities) {
    (void)fail(kernel, llvm::Twine("could not build a capability scope for "
                                   "kernel @") +
                           kernel.getSymName() + ": " +
                           llvm::toString(capabilities.takeError()));
    return mlir::failure();
  }

  VariantProposalRequest request(highLevelOp, kernel, *capabilities);
  llvm::SmallVector<VariantProposal, 4> proposals;
  if (llvm::Error error = registry.collectVariantProposals(request, proposals)) {
    (void)fail(kernel, llvm::Twine("failed to collect variant proposals for "
                                   "kernel @") +
                           kernel.getSymName() + ": " +
                           llvm::toString(std::move(error)));
    return mlir::failure();
  }

  const VariantProposal *fallbackProposal = nullptr;
  for (const VariantProposal &proposal : proposals) {
    if (proposal.getFallbackRole() != VariantFallbackRole::ConservativeFallback)
      continue;
    if (fallbackProposal) {
      (void)fail(kernel, "requires exactly one conservative-fallback variant "
                         "proposal; the registry produced more than one");
      return mlir::failure();
    }
    fallbackProposal = &proposal;
  }
  if (!fallbackProposal) {
    (void)fail(kernel, "requires a conservative-fallback variant proposal from "
                       "a fallback-owning plugin; none was produced");
    return mlir::failure();
  }

  VariantProposal scopedProposal = *fallbackProposal;
  scopedProposal.setVariantName(fallbackVariantSymbol);
  if (llvm::Error error = transforms::materializeVariantProposals(
          builder, request, scopedProposal)) {
    (void)fail(kernel, llvm::Twine("failed to materialize the conservative "
                                   "fallback variant for kernel @") +
                           kernel.getSymName() + ": " +
                           llvm::toString(std::move(error)));
    return mlir::failure();
  }
  return fallbackProposal->getOriginPlugin().str();
}

void createDispatch(mlir::OpBuilder &builder, mlir::Location loc,
                    llvm::StringRef selectedVariantSymbol,
                    llvm::StringRef fallbackVariantSymbol,
                    llvm::StringRef fallbackOrigin) {
  mlir::OperationState dispatchState(loc,
                                     tcrvexec::DispatchOp::getOperationName());
  dispatchState.addRegion();
  auto dispatch =
      llvm::cast<tcrvexec::DispatchOp>(builder.create(dispatchState));
  dispatch.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&dispatch.getBody().front());

  mlir::OperationState caseState(loc,
                                 tcrvexec::DispatchCaseOp::getOperationName());
  caseState.addAttribute("target", symbolRef(builder, selectedVariantSymbol));
  caseState.addAttribute(kOriginAttrName,
                         builder.getStringAttr(getRVVExtensionPluginName()));
  caseState.addAttribute("policy", builder.getStringAttr(kDispatchPolicy));
  (void)builder.create(caseState);

  mlir::OperationState fallbackState(loc,
                                     tcrvexec::FallbackOp::getOperationName());
  fallbackState.addAttribute("target",
                             symbolRef(builder, fallbackVariantSymbol));
  fallbackState.addAttribute(kOriginAttrName,
                             builder.getStringAttr(fallbackOrigin));
  fallbackState.addAttribute(kFallbackRoleAttrName,
                             builder.getStringAttr(
                                 kConservativeFallbackRoleValue));
  (void)builder.create(fallbackState);
}

mlir::LogicalResult
materializeKernel(mlir::OpBuilder &builder, llvm::StringRef kernelName,
                  const ExtensionPluginRegistry &registry,
                  IQ2XSBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_iq2_xs_q8_K_block_dot";
  std::string fallbackVariantSymbol = "rvv_iq2_xs_q8_K_block_dot_scalar_fallback";

  mlir::OperationState kernelState(loc, tcrvexec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel = llvm::cast<tcrvexec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  if (mlir::failed(createConservativeFallbackCapability(
          builder, loc, source.func, registry, kFallbackCapabilitySymbol)))
    return mlir::failure();
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  tcrvexec::VariantOp rvvVariant =
      createVariant(builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrvrvv::RuntimeABIValueType::get(builder.getContext());
  mlir::Type indexType = builder.getIndexType();

  // The ggml ggml_vec_dot_iq2_xs_q8_K ABI value set the super-block grid-codebook
  // block-dot op consumes -- n, s, vx, vy (the same FOUR the board-validated emitter
  // input declares, in declaration order n/s/vx/vy). The super-block loop, the
  // explicit int4 scale extraction, the indexed grid/signs gather, and the integer
  // bsum fold are op structure; the op consumes exactly these four (vx iq2_xs weight
  // base, vy q8_K activation base, s fp32 output, n element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "iq2xs-weight", runtimeABIType);
  mlir::Value vy =
      createRuntimeABIValue(builder, loc, "rhs-input-buffer", "vy",
                            "const uint8_t *", "q8k-act", runtimeABIType);

  tcrvrvv::SetVLOp setvl =
      createSetVL(builder, loc, n, /*sew=*/32, "m1", policy);
  tcrvrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), /*sew=*/32, "m1", policy,
                   kernelName, selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  // The auto-constructed attr-less super-block grid-codebook block dot-product op (the
  // explicit int4 scale extraction + the indexed 512-grid/ksigns gather + the integer
  // bsum fold are first-class STRUCTURE inside this op; the indexed grid gather does
  // NOT pin an m1 table anchor -- no shape knob is stamped, iq2_xs is NOT in any
  // autotuner).
  (void)createBlockDot(builder, loc, vx, vy, s, n, setvl.getVl());

  mlir::FailureOr<std::string> fallbackOrigin =
      materializeConservativeFallbackVariantViaPlugin(
          builder, kernel, source.func, registry, fallbackVariantSymbol);
  if (mlir::failed(fallbackOrigin))
    return mlir::failure();

  builder.setInsertionPointToEnd(&kernel.getBody().front());
  createDispatch(builder, loc, selectedVariantSymbol, fallbackVariantSymbol,
                 *fallbackOrigin);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// (3) The pass: marker-gated, source-only.
//===----------------------------------------------------------------------===//

bool hasStaleRVVLoweringSeedMetadata(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    found = op->hasAttr(kSeedAttrName);
  });
  return found;
}

mlir::LogicalResult requireRVVSourceOnlyModule(mlir::ModuleOp module) {
  mlir::Operation *staleOp = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (staleOp || op == module.getOperation())
      return;
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "tcrv" || dialect == "tcrv_rvv" || dialect == "tcrv_toy" ||
        dialect == "tcrv_tensorext_lite")
      staleOp = op;
  });
  if (!staleOp)
    return mlir::success();
  return fail(staleOp,
              "source materializer requires RVV source-only MLIR input; "
              "pre-existing selected-boundary or variant residue is not "
              "accepted");
}

std::string getKernelName(mlir::ModuleOp module) {
  auto kernelNameAttr =
      module->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (kernelNameAttr && !kernelNameAttr.getValue().trim().empty())
    return kernelNameAttr.getValue().trim().str();
  return "rvv_iq2_xs_q8_K_block_dot_from_vector_source";
}

class MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass(
      const MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.iq2_xs_q8_k_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + explicit int4 scale "
           "extraction + the 512-grid/ksigns indexed gather + the integer bsum fold "
           "are first-class op structure); iq2_xs is NOT in any schedule autotuner "
           "and the indexed grid gather does not pin an m1 table anchor, so the op "
           "lowers at the iq2_xs emitter's shape (COVERAGE, not a flip)";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect, tcrvexec::TCRVExecDialect,
                    tcrvrvv::TCRVRVVDialect>();
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (!registry) {
      module.emitError()
          << "RVV iq2_xs x q8_K super-block grid-codebook block-dot source front "
             "door requires an injected extension-plugin registry to dispatch the "
             "conservative fallback";
      signalPassFailure();
      return;
    }

    auto marker =
        module->getAttrOfType<mlir::StringAttr>(kSourceFrontDoorAttrName);
    if (!marker || marker.getValue().trim() != kAcceptedMarkerValue)
      return; // not our marker: leave the module untouched.

    if (hasStaleRVVLoweringSeedMetadata(module)) {
      (void)fail(module, "rejected stale tcrv_rvv.lowering_seed metadata as RVV "
                         "source-route authority");
      signalPassFailure();
      return;
    }
    if (mlir::failed(requireRVVSourceOnlyModule(module))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
    module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    if (funcs.size() != 1) {
      (void)fail(module, "source module must contain exactly one RVV iq2_xs x "
                         "q8_K super-block grid-codebook block-dot source function "
                         "candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<IQ2XSBlockDotSourceMatch> source =
        matchIQ2XSBlockDotSourceFunc(funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    std::string kernelName = getKernelName(module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(
            materializeKernel(builder, kernelName, *registry, *source))) {
      signalPassFailure();
      return;
    }

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  const ExtensionPluginRegistry *registry = nullptr;
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVIQ2XSBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVIQ2XSBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVIQ2XSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.iq2_xs_q8_k_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source (the SUPER-BLOCK GRID-"
      "CODEBOOK rung -- q4_K's super-block scaffold + iq2_xxs's packed 512-grid + "
      "ksigns sign plane); iq2_xs is NOT in any schedule autotuner and the indexed "
      "grid gather does not pin an m1 table anchor, so the op lowers at the iq2_xs "
      "emitter's shape (COVERAGE, not a flip)",
      [registryPtr] {
        return createMaterializeRVVIQ2XSBlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
