//===- RVVIQ3XXSBlockDotSourceFrontDoor.cpp ----------------------------===//
//
// Track B auto-lowering, the SUPER-BLOCK-GRID-CODEBOOK rung -- the LAST literal
// block-dot bucket to close (the deep iq* super-block-codebook tail). Where iq4_xs
// wraps a 16-entry vrgather codebook in the q4_K super-block, iq3_xxs wraps a
// GRID-of-4 codebook (a 256-entry uint32 table, each entry 4 int8 grid values) plus
// a 128-entry ksigns sign plane in the SAME super-block machinery. The COMPILER
// auto-constructs the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE attr-less tcrv_rvv.iq3_xxs_q8_k_block_dot op, from a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq3_xxs_q8_K` OPERATOR IDENTITY,
// instead of a per-kernel hand-authored super-block-grid-codebook block-dot emitter
// input.
//
// WHAT THIS RUNG COMBINES: iq3_xxs is a GRID-codebook super-block quant. The weight
// byte indexes a 256-entry uint32 GRID (iq3xxs_grid[256], each entry packs 4 int8
// grid values -- an indexed grid-of-4 lookup, NOT an arithmetic decode); the per-
// element SIGN is read from the 128-entry ksigns_iq2xs[128] sign plane (with the
// inline {1<<j} kmask selector); and the per-group 4-bit scale ls = 2*(aux32>>28)+1
// folds in the INTEGER domain (bsum += sumi*ls), with the iq3_xxs trailing 0.25f
// factor applied at the store. All of that -- the super-block loop, the aux32
// little-endian reassembly from the SEPARATE gas region (weight byte 66 = qs +
// QK_K/4), the grid-of-4 gather, the two-4-lane sign-plane passes, the widening
// product/reduce, the per-super-block fp32 fold -- is first-class STRUCTURE inside
// the op + its existing iq3_xxs emitter. So the front door does NOT hand-roll any of
// it; it supplies (a) the iq3_xxs super-block-format CONSTANTS as the typed integer
// attrs the verifier pins AND (b) the two GRID-codebook tables as the structural
// DenseI32ArrayAttrs the verifier pins to size 256 (grid) and 128 (ksigns).
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY (the bucket verdict). The super-block-
// grid-codebook op takes the EXISTING super-block monolithic route family (shared
// with q4_K). The grid + ksigns tables are OP attrs consumed by the emitter, NOT a
// route-family concern: the emission plan (buildMonolithicBlockDotEmissionPlan) and
// the target export candidate validator both key ONLY off the op name -> route
// family + the kind/scale_model attrs + the ordered ABI roles; neither reads the
// grid. So the shared SuperBlock family + the grid/ksigns-attr stamping COMPOSE
// cleanly: one table row (SuperBlock, the 4-role ggml vec_dot ABI) + this front
// door, zero new mechanism. This CLOSES the literal block-dot zoo.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP (the q4_K framing). iq3_xxs is NOT in
// any schedule-descriptor autotuner. The op PINS the m1 integer-core anchor -- m1 is
// the VLEN>=128 (rv64gcv => Zvl128b) capability class the emitter targets, NOT a
// gearbox-selected or VLEN-flipped shape. The 1024-byte GRID does NOT broadcast into
// a vreg; the grid-of-4 lookup is a HARDWARE vluxei16 indexed gather (revectorized
// over the i32 grid base), so there is no codebook-VLMAX legality fact and NO
// VLEN128-vs-VLEN256 byte-flip for iq3_xxs. So this front door does NOT ride a
// gearbox: the constructed attr-less op lowers DIRECTLY at the emitter's m1 anchor,
// VLEN-independent. The NEW content is the auto-CONSTRUCTION feeding the existing
// iq3_xxs emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVIQ3XXSBlockDotSourceFrontDoor.h"

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

// The DISTINCT marker the source module carries to route to THIS iq3_xxs super-block
// grid-codebook front door (NOT the MVP/dequant/q4_0/q8_0/iq4_nl/q4_K/iq4_xs
// markers). Each front-door pass checks its own marker and early-returns on a
// mismatch, so the passes are mutually exclusive and the sibling lits are
// byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_iq3_xxs_q8_K_block_dot_source");
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
    "rvv-iq3-xxs-q8-k-block-dot-source-front-door-case");

// The ggml iq3_xxs super-block-format facts (ggml-common.h): QK_K == 256, 8
// sub-blocks of 32 elements, block_iq3_xxs stride 98 (fp16 d @0 | uint8 qs[96] @2 --
// the first 64 qs bytes are grid index bytes, the last 32 are the aux/gas stream @66
// = qs + QK_K/4), block_q8_K stride 292 (fp32 d @0 | 256 int8 qs @4 | 16 int16 bsums
// @260 UNUSED). These are iq3_xxs CONSTANTS the front door supplies (not derivable
// from a generic source) and are the exact values the tcrv_rvv.iq3_xxs_q8_k_block_dot
// verifier pins. NOTE casing: the op kind is the LOWER-k form
// "ggml_iq3_xxs_q8_k_block_dot"; the kernel/marker/variant symbols carry the
// canonical ggml UPPER-K "q8_K".
constexpr std::int64_t kQK = 256;
constexpr std::int64_t kSubBlock = 32;
constexpr std::int64_t kWeightBlockStride = 98;
constexpr std::int64_t kActivationBlockStride = 292;
constexpr std::int64_t kWeightDByteOffset = 0;
constexpr std::int64_t kWeightQsByteOffset = 2;
constexpr std::int64_t kWeightGasByteOffset = 66;
constexpr std::int64_t kActivationDByteOffset = 0;
constexpr std::int64_t kActivationQuantByteOffset = 4;

// ggml's iq3xxs_grid[256]: the GRID-of-4 codebook the weight byte indexes (each of
// the 256 uint32 entries packs 4 int8 grid values; the per-index decode loads 4
// bytes, an 8-lane sign group is assembled from TWO indices). Carried here as
// int32[256] rendering ggml's exact uint32 literals -- every entry < 0x80000000 so a
// positive int32 carries the value losslessly. The verifier pins its size to EXACTLY
// 256. It is supplied here as the iq3_xxs constant a generic source could not derive,
// and consumed structurally into the realized body (a `static const uint32_t[256]`
// decl read via a `(const int8_t *)` cast + the vluxei16 grid gather), NOT a string
// plan read.
constexpr std::array<std::int32_t, 256> kIQ3XXSGrid = {
    67372036, 67372052, 67372068, 67374092, 67374108, 67374142, 67376132, 67376148,
    67378188, 67380244, 67386908, 67386924, 67896332, 67896348, 67898372, 67898388,
    67900428, 67900460, 67902468, 67902484, 67904524, 67906596, 67911172, 68420612,
    68420628, 68420644, 68422668, 68424708, 68424724, 68426764, 68426780, 68426814,
    68430860, 68430910, 68435500, 68944908, 68944958, 68946948, 68946964, 68949036,
    68959748, 69471260, 69475390, 69477412, 69479486, 69484060, 69484076, 69993484,
    69993534, 69999636, 70003732, 70523948, 70530084, 71175172, 71175204, 71175220,
    71181340, 71185420, 201589772, 201589788, 201591812, 201591828, 201593868, 201593884,
    201595908, 201595924, 201595940, 201598014, 201600004, 202114052, 202114068, 202116108,
    202118148, 202118164, 202638348, 202638364, 202640388, 202640404, 202642444, 202644484,
    202653204, 203162628, 203162644, 203166724, 203168780, 203170868, 203174964, 203686924,
    203686956, 203697156, 204215300, 204215332, 204219444, 204226060, 204735532, 205394964,
    205399044, 335807492, 335807508, 335809548, 335809564, 335811588, 335811604, 335811636,
    335813644, 335815700, 336331788, 336331804, 336331820, 336333828, 336333844, 336335884,
    336337924, 336344092, 336344126, 336346628, 336856068, 336856084, 336858124, 336858174,
    336860164, 336860180, 336862270, 336864260, 336866348, 337380364, 337382404, 337382436,
    337395204, 337395236, 337910828, 337914908, 338428956, 338433086, 338437132, 338443812,
    339608588, 339608604, 339610676, 339616812, 470025228, 470027268, 470027284, 470029324,
    470029340, 470035460, 470037548, 470040084, 470549508, 470549524, 470553604, 470555660,
    470557732, 470557748, 471073804, 471073820, 471075844, 471077932, 471084052, 471088660,
    471600140, 471604252, 472128516, 472130622, 472137236, 472646660, 472646708, 472650772,
    472656940, 473173028, 473177140, 473183260, 473832476, 473838596, 604242980, 604245054,
    604249132, 604249150, 604253212, 604253246, 604782116, 605295620, 605297726, 605299716,
    605303812, 605303860, 605815870, 605824044, 606340132, 606350348, 606352420, 606868524,
    606872604, 606879236, 608044076, 608046084, 608046100, 608050180, 738462740, 738468876,
    738475524, 738984964, 738985012, 738989108, 738995244, 739511332, 739515412, 739524116,
    740033556, 740043804, 740559876, 740561948, 740561982, 740572692, 741082132, 741088268,
    741616644, 742265892, 742269972, 872682532, 872686628, 872686644, 872690724, 873206796,
    873214988, 873729086, 873739300, 874257412, 874257460, 874783780, 875299884, 875310100,
    875830300, 876479516, 876483596, 1040450588, 1040450604, 1040450622, 1040452612, 1040456724,
    1040460820, 1040978996, 1040983044, 1041501204, 1041507372, 1041509396, 1042023428, 1042025516,
    1042029596, 1042035716, 1042551820, 1042555916, 1043072004, 1043072020, 1043076132, 1043602436
};

// ggml's ksigns_iq2xs[128]: the sign plane the 7-bit sign selector ((aux32>>7l)&127)
// indexes; the per-lane sign bit is masked out with the inline {1<<j} kmask. Carried
// as int32[128] -- ksigns values reach 255 (beyond int8), so an i32 attr is required
// to carry them losslessly. The verifier pins its size to EXACTLY 128. Consumed
// structurally into the realized body (a `static const uint8_t[128]` decl + the
// broadcast/vand/vmsne/vmerge sign application), NOT a string plan read.
constexpr std::array<std::int32_t, 128> kIQ3XXSKsigns = {
    0, 129, 130, 3, 132, 5, 6, 135,
    136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23,
    24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39,
    40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183,
    184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71,
    72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215,
    216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231,
    232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119,
    120, 249, 250, 123, 252, 125, 126, 255
};

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml IQ3_XXS x Q8_K super-block grid-codebook block-dot "
                     "source front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the super-block-grid-codebook dot has no
//     compact generic vector form, so recognition is by the vec_dot ABI roles,
//     not by a straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct IQ3XXSBlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_iq3_xxs_q8_K` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the super-block-grid-codebook block-dot intent marker -- the WHAT is
// the operator identity (iq3_xxs weight x q8_K activation super-block dot-product ->
// fp32 out), NOT a generic dataflow body. The super-block loop + the grid-of-4 gather
// + the sign-plane passes + the int-domain scale fold are iq3_xxs STRUCTURE the
// constructed op carries; the source func body is the bounded intent shell (it may be
// a bare `return`).
mlir::FailureOr<IQ3XXSBlockDotSourceMatch>
matchIQ3XXSBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, iq3_xxs weight memref<?xi8>, q8_K "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "iq3_xxs weight rank-1 i8 memref, q8_K activation rank-1 i8 memref");

  return IQ3XXSBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.iq3_xxs_q8_k_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). NO shape knob: iq3_xxs is
//     NOT in any schedule autotuner, and the grid-of-4 gather lowers at the VLEN>=128
//     m1 anchor in the op emitter, so the constructed op lowers at that m1 anchor.
//     This is byte-identical to the hand-authored iq3_xxs block-dot emitter input.
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

// The ATTR-LESS ggml IQ3_XXS x Q8_K super-block-grid-codebook dot-product op: the
// bounded WHAT (kind, scale model, super-block-format facts, the 256-entry grid, the
// 128-entry ksigns sign plane) is stamped, but NO shape knob -- iq3_xxs is NOT in any
// schedule autotuner and the grid-of-4 gather lowers at the VLEN>=128 m1 anchor in
// the emitter. The super-block loop + the aux32 reassembly + the grid-of-4 gather +
// the sign-plane passes + the widening product/reduce + the int-domain scale fold are
// first-class STRUCTURE inside this op. The grid is the SAME iq3xxs_grid[256] and the
// sign plane the SAME ksigns_iq2xs[128] the hand-authored emitter input carries --
// supplied here as the structural DenseI32ArrayAttrs the verifier pins to 256 and
// 128.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(
      loc, tcrvrvv::GgmlBlockDotIQ3XXSQ8KOp::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_iq3_xxs_q8_k_block_dot"));
  state.addAttribute(
      "scale_model",
      builder.getStringAttr(
          "per-group-int4-grid-of-4-codebook-scale-int-domain"));
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
  state.addAttribute("weight_gas_byte_offset",
                     builder.getI64IntegerAttr(kWeightGasByteOffset));
  state.addAttribute("activation_d_byte_offset",
                     builder.getI64IntegerAttr(kActivationDByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("grid",
                     builder.getDenseI32ArrayAttr(llvm::ArrayRef<std::int32_t>(
                         kIQ3XXSGrid.data(), kIQ3XXSGrid.size())));
  state.addAttribute("ksigns",
                     builder.getDenseI32ArrayAttr(llvm::ArrayRef<std::int32_t>(
                         kIQ3XXSKsigns.data(), kIQ3XXSKsigns.size())));
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
                  IQ3XXSBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_iq3_xxs_q8_K_block_dot";
  std::string fallbackVariantSymbol =
      "rvv_iq3_xxs_q8_K_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_iq3_xxs_q8_K ABI value set the super-block-grid-codebook
  // block-dot op consumes -- n, s, vx, vy (the same FOUR the board-validated
  // emitter input declares, in declaration order n/s/vx/vy). The super-block loop,
  // the grid-of-4 gather, the sign-plane passes, and the int-domain scale fold are op
  // structure; the op consumes exactly these four (vx iq3_xxs weight base, vy q8_K
  // activation base, s fp32 output, n element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "iq3xxs-weight", runtimeABIType);
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
  // (the grid-of-4 gather + the sign-plane passes + the int-domain scale fold are
  // first-class STRUCTURE inside this op; the grid-of-4 gather lowers at the VLEN>=128
  // m1 anchor in the emitter -- no shape knob is stamped, iq3_xxs is NOT in any
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
  return "rvv_iq3_xxs_q8_K_block_dot_from_vector_source";
}

class MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass(
      const MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.iq3_xxs_q8_k_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + aux32 reassembly + the "
           "grid-of-4 gather + the sign-plane passes + the int-domain scale fold are "
           "first-class op structure); iq3_xxs is NOT in any schedule autotuner and "
           "the grid-of-4 gather lowers at the VLEN>=128 m1 anchor in the emitter, so "
           "the op lowers at that m1 integer-core anchor (COVERAGE, not a flip)";
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
          << "RVV iq3_xxs x q8_K super-block grid-codebook block-dot source front "
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
      (void)fail(module, "source module must contain exactly one RVV iq3_xxs x "
                         "q8_K super-block grid-codebook block-dot source function "
                         "candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<IQ3XXSBlockDotSourceMatch> source =
        matchIQ3XXSBlockDotSourceFunc(funcs.front());
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
createMaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVIQ3XXSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.iq3_xxs_q8_k_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source (the SUPER-BLOCK-GRID-"
      "CODEBOOK rung -- q4_K's super-block scaffold + the iq3 256-entry grid-of-4 "
      "codebook + the ksigns_iq2xs[128] sign plane); iq3_xxs is NOT in any schedule "
      "autotuner, so the op lowers at the emitter's VLEN>=128 m1 grid-gather anchor "
      "(COVERAGE, not a flip)",
      [registryPtr] {
        return createMaterializeRVVIQ3XXSBlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
