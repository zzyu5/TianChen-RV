//===- RVVIQ3SBlockDotSourceFrontDoor.cpp ------------------------------===//
//
// Track B auto-lowering, the SUPER-BLOCK GRID-CODEBOOK rung -- the LAST iq*
// super-block-codebook bucket, closing the literal block-dot zoo. The COMPILER
// auto-constructs the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE attr-less tcrv_rvv.iq3_s_q8_k_block_dot op, from a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq3_s_q8_K` OPERATOR IDENTITY,
// instead of a per-kernel hand-authored super-block GRID-codebook block-dot emitter
// input.
//
// WHAT THIS RUNG COMBINES: iq3_s is a RE-COMPOSITION of three already-built GRID-
// codebook mechanisms wrapped in the q6_K-style super-block structure (QK_K==256, 8
// sub-blocks of 32, the q8_K activation with its leading fp32 scale): (a) the iq3
// GRID-of-4 codebook (each iq3s_grid[512] entry packs FOUR int8 grid values, a
// uint32 table -- the 9-bit index, 512 entries, an indexed vle8(4) over
// grid_table + idx*4, NOT a vrgather), (b) the qh 9th-bit plane (a SINGLE bit mask
// 256, the two grid indices of a group taking shifts 8-2l and 7-2l), and (c) the
// EXPLICIT per-group signs read from a dedicated signs[QK_K/8] memory region (NOT a
// ksigns table). The EXPLICIT per-sub-block 4-bit scales are the two-nibble split
// applied in the INTEGER domain. All of it is first-class STRUCTURE inside the op +
// its existing iq3_s emitter. So the front door does NOT hand-roll any of it; it
// supplies (a) the iq3_s super-block-format CONSTANTS as the typed integer attrs the
// verifier pins AND (b) the 512-entry uint32 grid as the structural DenseI32ArrayAttr
// the verifier pins to size 512.
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY (the bucket verdict). The super-block
// GRID-codebook op takes the EXISTING super-block monolithic route family (shared
// with q4_K / iq4_xs). The grid is an OP attr consumed by the emitter, NOT a route-
// family concern: the emission plan (buildMonolithicBlockDotEmissionPlan) and the
// target export candidate validator both key ONLY off the op name -> route family +
// the kind/scale_model attrs + the ordered ABI roles; neither reads the grid. So the
// shared SuperBlock family + the iq* grid-attr stamping COMPOSE cleanly: one table
// row (SuperBlock, the 4-role ggml vec_dot ABI) + this front door, zero new
// mechanism.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP (the q4_K / iq4_xs framing). iq3_s is
// NOT in any schedule-descriptor autotuner. The constructed attr-less op lowers
// DIRECTLY at the emitter's m1 integer-core anchor -- m1 is carried by the op, NOT a
// gearbox-selected or VLEN-flipped shape. So this front door does NOT ride a gearbox:
// there is no iq3_s schedule autotuner. The signs are an EXPLICIT memory region
// (byte offset 74), NOT a table attr -- so no bespoke sign plane is stamped, only the
// weight_signs_byte_offset the verifier pins. The NEW content is the auto-
// CONSTRUCTION feeding the existing iq3_s emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVIQ3SBlockDotSourceFrontDoor.h"

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

// The DISTINCT marker the source module carries to route to THIS iq3_s super-block
// GRID-codebook front door (NOT the MVP/dequant/q4_0/q8_0/iq4_nl/q4_K/iq4_xs
// markers). Each front-door pass checks its own marker and early-returns on a
// mismatch, so the passes are mutually exclusive and the sibling lits are
// byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_iq3_s_q8_K_block_dot_source");
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
    "rvv-iq3-s-q8-k-block-dot-source-front-door-case");

// The ggml iq3_s super-block-format facts (ggml-common.h): QK_K == 256, 8 sub-blocks
// of 32 elements, block_iq3_s stride 110 (fp16 d @0 | uint8 qs[64] grid-index bytes
// @2 | uint8 qh[8] 9th-bit plane @66 | uint8 signs[32] EXPLICIT sign region @74 |
// uint8 scales[4] two-nibble scales @106), block_q8_K stride 292 (fp32 d @0 | 256
// int8 qs @4 | 16 int16 bsums @260 UNUSED -- iq3_s reads no bsums). These are iq3_s
// CONSTANTS the front door supplies (not derivable from a generic source) and are the
// exact values the tcrv_rvv.iq3_s_q8_k_block_dot verifier pins. NOTE casing: the op
// kind is the LOWER-k form "ggml_iq3_s_q8_k_block_dot"; the kernel/marker/variant
// symbols carry the canonical ggml UPPER-K "q8_K".
constexpr std::int64_t kQK = 256;
constexpr std::int64_t kSubBlock = 32;
constexpr std::int64_t kWeightBlockStride = 110;
constexpr std::int64_t kActivationBlockStride = 292;
constexpr std::int64_t kWeightDByteOffset = 0;
constexpr std::int64_t kWeightQsByteOffset = 2;
constexpr std::int64_t kWeightQhByteOffset = 66;
constexpr std::int64_t kWeightSignsByteOffset = 74;
constexpr std::int64_t kWeightScalesByteOffset = 106;
constexpr std::int64_t kActivationDByteOffset = 0;
constexpr std::int64_t kActivationQuantByteOffset = 4;

// ggml's iq3s_grid[512]: the 512-entry uint32 GRID-of-4 codebook (each entry packs
// FOUR int8 grid values; the 9-bit grid index [0,511] loads 4 bytes per index via an
// indexed pointer-arith lookup grid_i8 + idx*4, NOT a vrgather -- the 2048-byte table
// cannot broadcast into a vreg). This is the load-bearing structural fact of the
// GRID class; the verifier pins its size to EXACTLY 512. It is carried as int32[512]
// (each ggml uint32 literal < 0x80000000, so a positive int32 carries the value
// losslessly), copied VERBATIM from the op's byte-exact CORE test / ggml ggml-common.h
// iq3s_grid[512], and consumed structurally into the realized body (a
// `static const uint32_t[512]` decl + the indexed grid load), NOT a string plan read.
constexpr std::array<std::int32_t, 512> kIQ3SGrid = {
    16843009, 16843011, 16843013, 16843019, 16843023, 16843521, 16843523, 16843525,
    16843529, 16843533, 16844033, 16844035, 16844043, 16844551, 16845057, 16845061,
    16845067, 16845071, 16845571, 16845575, 16846081, 16846085, 16846595, 16846601,
    16846607, 16974081, 16974083, 16974085, 16974089, 16974593, 16974595, 16974603,
    16975105, 16975111, 16975119, 16975619, 16975627, 16976137, 16977155, 16977163,
    16977669, 17105153, 17105155, 17105163, 17105167, 17105665, 17105671, 17105677,
    17106179, 17106187, 17106689, 17106697, 17107205, 17107211, 17107215, 17107715,
    17107719, 17108737, 17108743, 17236231, 17236739, 17236747, 17237249, 17237253,
    17237763, 17237767, 17237773, 17238281, 17238785, 17238789, 17239311, 17239811,
    17239819, 17367297, 17367815, 17367823, 17368323, 17368329, 17368837, 17369345,
    17369351, 17369859, 17370881, 17498373, 17498377, 17499393, 17499397, 17499405,
    17499911, 17500419, 17500427, 17500431, 17501453, 17501959, 17629453, 17629955,
    17629959, 17630979, 17632005, 17633027, 17760513, 17760517, 17760521, 17761537,
    17761541, 17761549, 17762055, 17763073, 17763081, 50397441, 50397443, 50397445,
    50397449, 50397953, 50397955, 50397959, 50397963, 50397967, 50398465, 50398469,
    50398979, 50398985, 50398989, 50400009, 50400013, 50400515, 50401029, 50528513,
    50528515, 50528519, 50528525, 50529025, 50529033, 50529539, 50530049, 50530055,
    50530563, 50531073, 50531077, 50532097, 50532109, 50659585, 50660101, 50660107,
    50660111, 50660609, 50660617, 50661125, 50661633, 50661639, 50662155, 50662657,
    50663173, 50790659, 50790665, 50790671, 50791169, 50791175, 50791683, 50791695,
    50792193, 50792201, 50792707, 50793733, 50794241, 50921735, 50921739, 50922245,
    50922249, 50923267, 50923271, 50923781, 50923789, 50924289, 50924297, 51052803,
    51053313, 51053319, 51053827, 51054337, 51054341, 51055363, 51184897, 51184905,
    51184911, 51185929, 51185933, 51314947, 51314951, 51315457, 51315461, 51315971,
    51316491, 51316995, 51318021, 51318529, 83951873, 83951875, 83951879, 83951883,
    83951887, 83952385, 83952389, 83952393, 83952397, 83952899, 83952903, 83952911,
    83953409, 83953413, 83953923, 83953927, 83953931, 83954433, 83954437, 83954959,
    83955457, 83955463, 83955467, 84082945, 84082949, 84083457, 84083463, 84083471,
    84083973, 84083979, 84084483, 84084489, 84084997, 84085507, 84214019, 84214025,
    84214031, 84215043, 84215047, 84215553, 84215567, 84216067, 84216583, 84216591,
    84217603, 84217609, 84345089, 84345093, 84345099, 84345603, 84346117, 84346121,
    84346627, 84346631, 84347141, 84347649, 84348173, 84476163, 84476175, 84477185,
    84477191, 84477701, 84477707, 84478211, 84479749, 84479755, 84607241, 84607747,
    84608261, 84608783, 84609281, 84609799, 84610817, 84738305, 84738309, 84738319,
    84739331, 84740875, 84741379, 84869387, 84869891, 84870413, 84870913, 84871431,
    84871937, 117506309, 117506819, 117506823, 117506827, 117506831, 117507333, 117507843,
    117507847, 117507851, 117508357, 117508361, 117508367, 117508867, 117509383, 117509891,
    117637379, 117637383, 117637387, 117637897, 117638403, 117638407, 117639425, 117640449,
    117640965, 117640973, 117768449, 117768965, 117769473, 117769989, 117769993, 117771009,
    117899523, 117900033, 117900041, 117900547, 117900551, 117900559, 117901057, 117901571,
    117901575, 117901583, 117902091, 117903111, 118030599, 118031107, 118031117, 118031621,
    118032131, 118033157, 118033665, 118033673, 118161667, 118162177, 118162181, 118162699,
    118163205, 118163721, 118164237, 118165255, 118293261, 118294787, 118423811, 118423815,
    118424833, 118424837, 118425355, 151060737, 151060745, 151061253, 151061761, 151061769,
    151061775, 151062277, 151062787, 151063297, 151064321, 151191813, 151191823, 151192323,
    151192327, 151192837, 151193345, 151193355, 151193863, 151194371, 151194379, 151322883,
    151322887, 151323393, 151323403, 151323907, 151324423, 151324929, 151325455, 151325957,
    151326465, 151453961, 151454467, 151454471, 151454977, 151454981, 151455491, 151455499,
    151585025, 151585029, 151586057, 151586575, 151587073, 151588611, 151716107, 151716111,
    151717123, 151719173, 151847687, 151848713, 151850241, 151978753, 151978763, 151979777,
    151980295, 151980803, 184615173, 184615681, 184615689, 184616197, 184617217, 184617225,
    184617231, 184617733, 184618253, 184618761, 184746243, 184746247, 184746251, 184746757,
    184747267, 184747781, 184749829, 184877313, 184877827, 184878343, 184878849, 184878861,
    184879879, 185008389, 185008399, 185008897, 185009423, 185010441, 185010947, 185011467,
    185011975, 185139459, 185139465, 185140481, 185140997, 185141517, 185271045, 185271565,
    185273091, 185273095, 185403653, 185532677, 185532681, 185533701, 218170115, 218170119,
    218170123, 218171139, 218171143, 218172673, 218300673, 218301697, 218301711, 218303753,
    218432261, 218433289, 218433797, 218434315, 218434821, 218435329, 218562817, 218563337,
    218563843, 218564865, 218694923, 218695943, 218696965, 218824961, 218824967, 218826505,
    218828033, 218956043, 218958081, 219087619, 219087623, 251724033, 251724041, 251724047,
    251725057, 251725061, 251725581, 251726081, 251726601, 251727109, 251855109, 251855619,
    251856137, 251857159, 251857163, 251986179, 251986185, 251986689, 251986701, 251987203,
    251987713, 251988739, 252117253, 252118789, 252118795, 252119815, 252248323, 252248331,
    252248839, 252249345, 252250881, 252380421, 252381445, 252510469, 252512003, 252641537};

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml IQ3_S x Q8_K super-block GRID-codebook block-dot "
                     "source front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the super-block GRID-codebook dot has no
//     compact generic vector form, so recognition is by the vec_dot ABI roles,
//     not by a straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct IQ3SBlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_iq3_s_q8_K` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the super-block GRID-codebook block-dot intent marker -- the WHAT is
// the operator identity (iq3_s weight x q8_K activation super-block dot-product ->
// fp32 out), NOT a generic dataflow body. The super-block loop + the two-nibble scale
// split + the qh 9th-bit injection + the grid-of-4 lookup + the sign-plane apply +
// the float-domain fold are iq3_s STRUCTURE the constructed op carries; the source
// func body is the bounded intent shell (it may be a bare `return`).
mlir::FailureOr<IQ3SBlockDotSourceMatch>
matchIQ3SBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, iq3_s weight memref<?xi8>, q8_K "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "iq3_s weight rank-1 i8 memref, q8_K activation rank-1 i8 memref");

  return IQ3SBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.iq3_s_q8_k_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). NO shape knob: iq3_s is
//     NOT in any schedule autotuner, so the constructed op lowers at the emitter's
//     m1 integer-core anchor. This is byte-identical to the hand-authored iq3_s
//     block-dot emitter input.
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

// The ATTR-LESS ggml IQ3_S x Q8_K super-block GRID-codebook dot-product op: the
// bounded WHAT (kind, scale model, super-block-format facts, the 512-entry grid) is
// stamped, but NO shape knob -- iq3_s is NOT in any schedule autotuner, so the op
// lowers at the emitter's m1 integer-core anchor. The super-block loop + the two-
// nibble scale split + the qh 9th-bit injection + the grid-of-4 lookup + the sign-
// plane apply + the widening product/reduce + the float-domain fold are first-class
// STRUCTURE inside this op. The grid is the SAME iq3s_grid[512] uint32 table ggml
// carries -- supplied here as the structural DenseI32ArrayAttr the verifier pins to
// size 512. The signs are an EXPLICIT memory region (byte offset 74), NOT a table
// attr.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc,
                             tcrvrvv::GgmlBlockDotIQ3SQ8KOp::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_iq3_s_q8_k_block_dot"));
  state.addAttribute(
      "scale_model",
      builder.getStringAttr(
          "per-sub-block-int4-explicit-scales-grid-of-4-codebook-qh-plane-explicit-signs-int-domain"));
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
  state.addAttribute("weight_qh_byte_offset",
                     builder.getI64IntegerAttr(kWeightQhByteOffset));
  state.addAttribute("weight_signs_byte_offset",
                     builder.getI64IntegerAttr(kWeightSignsByteOffset));
  state.addAttribute("weight_scales_byte_offset",
                     builder.getI64IntegerAttr(kWeightScalesByteOffset));
  state.addAttribute("activation_d_byte_offset",
                     builder.getI64IntegerAttr(kActivationDByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("grid",
                     builder.getDenseI32ArrayAttr(llvm::ArrayRef<std::int32_t>(
                         kIQ3SGrid.data(), kIQ3SGrid.size())));
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
                  IQ3SBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_iq3_s_q8_K_block_dot";
  std::string fallbackVariantSymbol = "rvv_iq3_s_q8_K_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_iq3_s_q8_K ABI value set the super-block GRID-codebook
  // block-dot op consumes -- n, s, vx, vy (the same FOUR the board-validated emitter
  // input declares, in declaration order n/s/vx/vy). The super-block loop, the two-
  // nibble scale split, the qh 9th-bit injection, the grid-of-4 lookup, the sign-plane
  // apply, and the float-domain fold are op structure; the op consumes exactly these
  // four (vx iq3_s weight base, vy q8_K activation base, s fp32 output, n element
  // count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "iq3s-weight", runtimeABIType);
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

  // The auto-constructed attr-less super-block GRID-codebook block dot-product op
  // (the two-nibble scale split + the qh 9th-bit injection + the 512-entry grid-of-4
  // lookup + the sign-plane apply + the float-domain fold are first-class STRUCTURE
  // inside this op; no shape knob is stamped, iq3_s is NOT in any autotuner and lowers
  // at the emitter's m1 integer-core anchor).
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
  return "rvv_iq3_s_q8_K_block_dot_from_vector_source";
}

class MaterializeRVVIQ3SBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVIQ3SBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVIQ3SBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVIQ3SBlockDotSourceFrontDoorPass(
      const MaterializeRVVIQ3SBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVIQ3SBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVIQ3SBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.iq3_s_q8_k_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + two-nibble scale split "
           "+ qh 9th-bit injection + the 512-entry grid-of-4 lookup + the explicit "
           "sign-plane apply + the float-domain fold are first-class op structure); "
           "iq3_s is NOT in any schedule autotuner, so the op lowers at the emitter's "
           "m1 integer-core anchor (COVERAGE, not a flip)";
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
          << "RVV iq3_s x q8_K super-block GRID-codebook block-dot source front "
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
      (void)fail(module, "source module must contain exactly one RVV iq3_s x "
                         "q8_K super-block GRID-codebook block-dot source "
                         "function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<IQ3SBlockDotSourceMatch> source =
        matchIQ3SBlockDotSourceFunc(funcs.front());
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
createMaterializeRVVIQ3SBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVIQ3SBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVIQ3SBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.iq3_s_q8_k_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source (the SUPER-BLOCK "
      "GRID-CODEBOOK rung -- q4_K's super-block scaffold + the iq3 512-entry "
      "grid-of-4 codebook + the qh 9th-bit plane + the explicit signs region); "
      "iq3_s is NOT in any schedule autotuner, so the op lowers at the emitter's "
      "m1 grid-lookup integer-core anchor (COVERAGE, not a flip)",
      [registryPtr] {
        return createMaterializeRVVIQ3SBlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
