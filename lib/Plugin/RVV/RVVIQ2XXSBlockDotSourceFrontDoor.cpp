//===- RVVIQ2XXSBlockDotSourceFrontDoor.cpp ----------------------------===//
//
// Track B auto-lowering, the SUPER-BLOCK-GRID-CODEBOOK rung -- the FINAL iq*
// super-block-codebook bucket, closing the literal block-dot zoo. The COMPILER
// auto-constructs the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE attr-less tcrv_rvv.iq2_xxs_q8_k_block_dot op, from a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq2_xxs_q8_K` OPERATOR IDENTITY,
// instead of a per-kernel hand-authored super-block grid-codebook block-dot emitter
// input.
//
// WHAT THIS RUNG COMBINES: iq2_xxs is the FIRST member of the GRID-codebook class --
// genuinely NEW vs iq4_nl/iq4_xs's 16-entry per-nibble table. Each weight byte
// INDEXES a large packed GRID codebook (`iq2xxs_grid[256]`, a uint64 table where each
// entry encodes 8 int8 values), the per-element SIGN is read from a separate SIGN
// PLANE (`ksigns_iq2xs[128]`), and a per-group 4-bit scale `ls = 2*(aux1>>28)+1`
// folds in the INTEGER domain. That grid+sign integer core is wrapped in the
// q6_K/q4_K-style super-block structure (QK_K==256, 8 sub-blocks of 32, the q8_K
// activation with its leading fp32 scale, the per-super-block integer bsum, the fp32
// fold with the trailing iq2_xxs 1/8 factor). ALL of it -- the super-block loop, the
// aux-word reassembly from little-endian byte loads, the 4-bit scale, the vluxei16
// GRID + DERIVED signs64 gathers, the sign fold onto the grid, the widening
// product/reduce, and the integer-domain fold -- is first-class STRUCTURE inside the
// op + its existing iq2_xxs emitter. So the front door does NOT hand-roll any of it;
// it supplies (a) the iq2_xxs super-block-format CONSTANTS as the typed integer attrs
// the verifier pins AND (b) the two GRID-codebook structural tables -- the 256-entry
// uint64 grid as the DenseI64ArrayAttr the verifier pins to size 256, and the
// 128-entry ksigns sign plane as the DenseI32ArrayAttr the verifier pins to size 128
// (ksigns values reach 255, beyond int8, so an i32 attr carries them losslessly).
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY (the bucket verdict). The super-block-
// grid-codebook op takes the EXISTING super-block monolithic route family (shared
// with q4_K). The grid + ksigns are OP attrs consumed by the emitter, NOT a
// route-family concern: the emission plan (buildMonolithicBlockDotEmissionPlan) and
// the target export candidate validator both key ONLY off the op name -> route family
// + the kind/scale_model attrs + the ordered ABI roles; neither reads the grid or
// ksigns. So the shared SuperBlock family + iq2_xxs's grid/ksigns-attr stamping
// COMPOSE cleanly: one table row (SuperBlock, the 4-role ggml vec_dot ABI) + this
// front door, zero new mechanism.
//
// HONEST FRAMING -- COVERAGE AT THE DEFAULT ANCHOR (the tq2_0 framing, distinct from
// iq4_xs). UNLIKE q4_K/iq4_xs, iq2_xxs IS in a schedule-descriptor autotuner (it
// adopts TunableScheduleOpInterface; kernel key "iq2_xxs"): the 32-lane grid+sign
// gather+dot sub-block body straddles the i8 strip VLMAX boundary between
// VLEN128/256, so the gearbox stamps "m2" at VLEN128 (e8m1 VLMAX 16 < 32) and the
// lighter "m1" at VLEN256 from the SAME getRVVStripVLMAXElements truth source. There
// IS a VLEN128-vs-VLEN256 flip for iq2_xxs. BUT this front door leaves the constructed
// op ATTR-LESS (no integer_core_lmul): it lowers at the iq2_xxs emitter's DEFAULT
// "m2" integer-core anchor (the VLEN-universal-safe floor, the byte-exact CORE target
// the hand-authored emitter lit pins). The gearbox is free to REFINE m2->m1 at
// VLEN256 through a separate schedule pass -- NOT stamped here and NOT exercised by
// this coverage rung. The NEW content is the auto-CONSTRUCTION feeding the existing
// iq2_xxs emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVIQ2XXSBlockDotSourceFrontDoor.h"

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

// The DISTINCT marker the source module carries to route to THIS iq2_xxs super-block
// grid-codebook front door (NOT the MVP/dequant/q4_0/q8_0/iq4_nl/q4_K/iq4_xs/tq2_0
// markers). Each front-door pass checks its own marker and early-returns on a
// mismatch, so the passes are mutually exclusive and the sibling lits are
// byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_iq2_xxs_q8_K_block_dot_source");
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
    "rvv-iq2-xxs-q8-k-block-dot-source-front-door-case");

// The ggml iq2_xxs super-block-format facts (ggml-common.h): QK_K == 256, 8
// sub-blocks of 32 elements, block_iq2_xxs stride 66 (fp16 d @0 | uint16 qs[32] @2),
// block_q8_K stride 292 (fp32 d @0 | 256 int8 qs @4 | 16 int16 bsums @260 UNUSED --
// iq2_xxs accumulates its own integer bsum, no min term). These are iq2_xxs CONSTANTS
// the front door supplies (not derivable from a generic source) and are the exact
// values the tcrv_rvv.iq2_xxs_q8_k_block_dot verifier pins. NOTE casing: the op kind
// is the LOWER-k form "ggml_iq2_xxs_q8_k_block_dot"; the kernel/marker/variant symbols
// carry the canonical ggml UPPER-K "q8_K".
constexpr std::int64_t kQK = 256;
constexpr std::int64_t kSubBlock = 32;
constexpr std::int64_t kWeightBlockStride = 66;
constexpr std::int64_t kActivationBlockStride = 292;
constexpr std::int64_t kWeightDByteOffset = 0;
constexpr std::int64_t kWeightQsByteOffset = 2;
constexpr std::int64_t kActivationDByteOffset = 0;
constexpr std::int64_t kActivationQuantByteOffset = 4;

// ggml's iq2xxs_grid[256]: the packed uint64 GRID codebook the weight byte indexes
// (each entry encodes 8 int8 grid values; the per-index lookup is `grid_table_i8 +
// idx*8`, NOT a vrgather). This is the load-bearing structural fact of the GRID class;
// the verifier pins its size to EXACTLY 256. Carried as int64[256] rendering ggml's
// exact uint64 literals (all 256 entries fit the signed-int64 range). Copied VERBATIM
// from the CORE authority (rvv-to-emitc-iq2-xxs-q8-k-block-dot.mlir); consumed
// structurally into the realized body (a `static const int64_t[256]` decl + the
// vluxei16 grid gather), NOT a string plan read.
constexpr std::array<std::int64_t, 256> kIQ2XXSGrid = {
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

// ggml's ksigns_iq2xs[128]: the SIGN PLANE indexed by the 7-bit sign selector
// [0,127]. The verifier pins its size to EXACTLY 128. Carried as int32[128] because
// ksigns values reach 255 (beyond int8) and must be carried losslessly. Copied
// VERBATIM from the CORE authority; consumed structurally (the emitter DERIVES the
// signs64 +-1 table from these selectors in-emitter), NOT a string plan read.
constexpr std::array<std::int32_t, 128> kIQ2XXSKsigns = {
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
  op->emitError() << "ggml IQ2_XXS x Q8_K super-block grid-codebook block-dot "
                     "source front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the super-block grid-codebook dot has no
//     compact generic vector form, so recognition is by the vec_dot ABI roles,
//     not by a straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct IQ2XXSBlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_iq2_xxs_q8_K` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the super-block grid-codebook block-dot intent marker -- the WHAT is
// the operator identity (iq2_xxs weight x q8_K activation super-block dot-product ->
// fp32 out), NOT a generic dataflow body. The super-block loop + the aux-word
// reassembly + the 4-bit scale + the grid+sign gather + the integer/float fold are
// iq2_xxs STRUCTURE the constructed op carries; the source func body is the bounded
// intent shell (it may be a bare `return`).
mlir::FailureOr<IQ2XXSBlockDotSourceMatch>
matchIQ2XXSBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, iq2_xxs weight memref<?xi8>, q8_K "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "iq2_xxs weight rank-1 i8 memref, q8_K activation rank-1 i8 "
                "memref");

  return IQ2XXSBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.iq2_xxs_q8_k_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). The SHAPE knob
//     (integer_core_lmul, iq2_xxs's Win-A) is left OFF: the constructed op lowers at
//     the iq2_xxs emitter's default m2 anchor. iq2_xxs IS in a schedule autotuner (m2
//     at VLEN128, m1 at VLEN256), but the front door does NOT stamp the VLEN256 m1
//     refinement -- the default m2 form is the byte-exact target the hand-authored
//     emitter lit pins. This is byte-identical to the hand-authored iq2_xxs block-dot
//     emitter input.
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

// The ATTR-LESS ggml IQ2_XXS x Q8_K super-block-grid-codebook dot-product op: the
// bounded WHAT (kind, scale model, super-block-format facts, the 256-entry grid, the
// 128-entry ksigns sign plane) is stamped, but NO shape knob (integer_core_lmul) --
// so the iq2_xxs emitter lowers at its default m2 integer-core anchor (the
// VLEN-universal floor). iq2_xxs IS in a schedule autotuner (it can be refined m2->m1
// at VLEN256 by a separate schedule pass), but the front door does NOT stamp that
// refinement here; the default m2 form IS the byte-exact target the hand-authored
// emitter lit pins. The super-block loop + the aux-word reassembly + the 4-bit scale
// + the grid+sign vluxei16 gathers + the sign fold + the widening product/reduce +
// the integer-domain fold are first-class STRUCTURE inside this op. The grid is the
// packed uint64 iq2xxs_grid[256] (verifier size 256); the ksigns sign plane is
// ksigns_iq2xs[128] (verifier size 128, carried as i32 for lossless 0..255 values).
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(
      loc, tcrvrvv::GgmlBlockDotIQ2XXSQ8KOp::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_iq2_xxs_q8_k_block_dot"));
  state.addAttribute(
      "scale_model",
      builder.getStringAttr("per-group-int4-grid-codebook-scale-int-domain"));
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
  state.addAttribute("activation_d_byte_offset",
                     builder.getI64IntegerAttr(kActivationDByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("grid",
                     builder.getDenseI64ArrayAttr(llvm::ArrayRef<std::int64_t>(
                         kIQ2XXSGrid.data(), kIQ2XXSGrid.size())));
  state.addAttribute("ksigns",
                     builder.getDenseI32ArrayAttr(llvm::ArrayRef<std::int32_t>(
                         kIQ2XXSKsigns.data(), kIQ2XXSKsigns.size())));
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
                  IQ2XXSBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_iq2_xxs_q8_K_block_dot";
  std::string fallbackVariantSymbol =
      "rvv_iq2_xxs_q8_K_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_iq2_xxs_q8_K ABI value set the super-block-grid-codebook
  // block-dot op consumes -- n, s, vx, vy (the same FOUR the board-validated emitter
  // input declares, in declaration order n/s/vx/vy). The super-block loop, the
  // aux-word reassembly, the 4-bit scale, the grid+sign gather, and the
  // integer/float fold are op structure; the op consumes exactly these four (vx
  // iq2_xxs weight base, vy q8_K activation base, s fp32 output, n element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "iq2xxs-weight", runtimeABIType);
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

  // The auto-constructed attr-less super-block-grid-codebook block dot-product op
  // (the aux-word reassembly + the 4-bit scale + the 256-entry grid + 128-entry
  // ksigns vluxei16 gathers + the sign fold + the integer-domain fold are first-class
  // STRUCTURE inside this op; iq2_xxs's Win-A integer_core_lmul is left unstamped --
  // the default m2 anchor IS the byte-exact target, the gearbox's VLEN256 m1
  // refinement is a separate schedule-pass concern).
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
  return "rvv_iq2_xxs_q8_K_block_dot_from_vector_source";
}

class MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass(
      const MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.iq2_xxs_q8_k_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + aux-word reassembly + "
           "4-bit scale + the 256-entry grid / 128-entry ksigns gather + sign fold "
           "+ the integer-domain fold are first-class op structure); iq2_xxs IS in a "
           "schedule autotuner (m2 at VLEN128, m1 at VLEN256), but the front door "
           "leaves the op attr-less so it lowers at the emitter's default m2 anchor "
           "(COVERAGE at the default anchor -- the VLEN256 m1 refinement is a "
           "separate pass)";
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
          << "RVV iq2_xxs x q8_K super-block grid-codebook block-dot source front "
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
      (void)fail(module, "source module must contain exactly one RVV iq2_xxs x "
                         "q8_K super-block grid-codebook block-dot source function "
                         "candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<IQ2XXSBlockDotSourceMatch> source =
        matchIQ2XXSBlockDotSourceFunc(funcs.front());
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
createMaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVIQ2XXSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.iq2_xxs_q8_k_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source (the SUPER-BLOCK-GRID-"
      "CODEBOOK rung -- q4_K's super-block scaffold + iq2_xxs's 256-entry packed "
      "uint64 grid + 128-entry ksigns sign plane); iq2_xxs IS in a schedule "
      "autotuner (m2 at VLEN128, m1 at VLEN256), but the front door leaves the op "
      "attr-less so it lowers at the emitter's default m2 anchor (COVERAGE at the "
      "default anchor, not a flip -- the VLEN256 m1 refinement is a separate pass)",
      [registryPtr] {
        return createMaterializeRVVIQ2XXSBlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
