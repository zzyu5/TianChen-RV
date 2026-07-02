//===- RVVQ2KBlockDotSourceFrontDoor.cpp --------------------------------===//
//
// Track B auto-lowering, the SUPER-BLOCK rung (a q2_K sibling): the COMPILER
// auto-constructs the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE attr-less tcrv_rvv.q2_k_q8_k_block_dot op, from a marked
// GENERIC source carrying the ggml `ggml_vec_dot_q2_K_q8_K` OPERATOR IDENTITY,
// instead of a per-kernel hand-authored super-block block-dot emitter input.
//
// q2_K is the 2-bit modern K-quant -- a 256-element SUPER-BLOCK = 16 sub-blocks of
// 16 elements. The payload is 16 packed 4-bit-scale/4-bit-min bytes scales (@0) | 64
// packed 2-bit-weight bytes qs (@16) | fp16 super-block d (@80) | fp16 super-block
// dmin (@82); AoS stride 84. It is SIMPLER than q4_K's K4b in three ways: (1) the
// weights are 2-bit, unpacked by `(qs >> shift) & 3` over shifts {0,2,4,6}; (2) the
// per-sub-block scales/mins are SIMPLE 4-bit nibbles of the direct scales[16] bytes
// (NO 6-bit utmp/kmask bit-dance); (3) the positive fold is SCALAR -- one
// per-super-block int isum folded `sumf += dall*isum - dmin*summs` (NO 8-lane sums
// vector, NO post-loop horizontal sum). ALL of that -- the super-block loop, the
// 2-bit unpack, the 4-bit nibble scale/min, the scalar fp32 fold+min -- is
// FIRST-CLASS STRUCTURE inside that op + its existing q2_K emitter
// (RVVToEmitCKQuant.cpp). So the front door does NOT hand-roll any of it; it supplies
// the q2_K super-block-format CONSTANTS as the typed integer attrs the verifier pins.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP (the q2_K sibling's framing). q2_K has
// NO VLEN128-vs-VLEN256 byte-flip, AND is NOT in any schedule-descriptor autotuner.
// So this front door does NOT "ride the existing gearbox": there is no q2_K gearbox
// to ride. The constructed attr-less op lowers at the q2_K emitter's DEFAULT
// integer-core anchor, VLEN-independent. q2_K carries NO integer_core_lmul knob at
// all (unlike q4_K/q6_K). The NEW content is the auto-CONSTRUCTION feeding the
// existing q2_K emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVQ2KBlockDotSourceFrontDoor.h"

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

namespace tcrvexec = ::tianchenrv::tcrv::exec;
namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

// The DISTINCT marker the source module carries to route to THIS q2_K super-block
// front door (NOT the MVP/dequant/q4_0/iq4_nl markers). Each front-door pass checks
// its own marker and early-returns on a mismatch, so the passes are mutually
// exclusive and the sibling lits are byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_q2_K_q8_K_block_dot_source");
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
    "rvv-q2-k-q8-k-block-dot-source-front-door-case");

// The ggml q2_K super-block-format facts (ggml-common.h): QK_K == 256, 16 sub-blocks
// of 16 elements, block_q2_K stride 84 (16 packed 4-bit-scale/4-bit-min bytes scales
// @0 | 64 packed 2-bit-weight bytes qs @16 | fp16 d @80 | fp16 dmin @82), block_q8_K
// stride 292 (fp32 d @0 | 256 int8 qs @4 | 16 int16 bsums @260). These are q2_K
// CONSTANTS the front door supplies (they are not derivable from a generic source)
// and are the exact values the tcrv_rvv.q2_k_q8_k_block_dot verifier pins -- so the
// auto-constructed op passes its bounded-attr verifier. NOTE casing: the op kind is
// the LOWER-k form "ggml_q2_k_q8_k_block_dot"; the kernel/marker symbols carry the
// canonical ggml UPPER-K "q2_K".
constexpr std::int64_t kQK = 256;
constexpr std::int64_t kSubBlock = 16;
constexpr std::int64_t kWeightBlockStride = 84;
constexpr std::int64_t kActivationBlockStride = 292;
constexpr std::int64_t kWeightScalesByteOffset = 0;
constexpr std::int64_t kWeightQsByteOffset = 16;
constexpr std::int64_t kWeightDByteOffset = 80;
constexpr std::int64_t kWeightDminByteOffset = 82;
constexpr std::int64_t kActivationDByteOffset = 0;
constexpr std::int64_t kActivationQuantByteOffset = 4;
constexpr std::int64_t kActivationBsumsByteOffset = 260;

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml Q2_K x Q8_K super-block block-dot source front door "
                     "failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the super-block dot has no compact
//     generic vector form, so recognition is by the vec_dot ABI roles, not by a
//     straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct Q2KBlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_q2_K_q8_K` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the super-block block-dot intent marker -- the WHAT is the operator
// identity (q2_K weight x q8_K activation super-block dot-product -> fp32 out), NOT
// a generic dataflow body. The super-block loop + the 2-bit unpack + the 4-bit
// nibble scale/min + the scalar fp32 fold+min are q2_K STRUCTURE the constructed op
// carries; the source func body is the bounded intent shell (it may be a bare
// `return`).
mlir::FailureOr<Q2KBlockDotSourceMatch>
matchQ2KBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, q2_K weight memref<?xi8>, q8_K "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "q2_K weight rank-1 i8 memref, q8_K activation rank-1 i8 memref");

  return Q2KBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.q2_k_q8_k_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). The SHAPE knob
//     q2_K carries NO shape knob: the constructed op lowers at
//     the q2_K emitter's default anchor (there is no q2_K schedule autotuner to
//     stamp a capability shape). This is byte-identical to the hand-authored q2_K
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

// The ATTR-LESS ggml Q2_K x Q8_K super-block dot-product op: the bounded WHAT (kind,
// scale model, super-block-format facts) is stamped. q2_K has NO shape knob at all
// -- so the q2_K emitter lowers at its default integer-core anchor. There is no q2_K
// schedule autotuner, so the front door does NOT defer shape selection to a gearbox
// (unlike q4_0/iq4_nl); the default form IS the byte-exact target the hand-authored
// emitter lit pins. The 2-bit unpack + the 4-bit nibble scale/min + the scalar fp32
// fold+min are first-class STRUCTURE inside this op.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc,
                             tcrvrvv::GgmlBlockDotQ2KQ8KOp::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind", builder.getStringAttr("ggml_q2_k_q8_k_block_dot"));
  state.addAttribute(
      "scale_model",
      builder.getStringAttr(
          "per-sub-block-uint4-scale-i32-domain-scalar-fp32-fold-min"));
  state.addAttribute("qk", builder.getI64IntegerAttr(kQK));
  state.addAttribute("sub_block", builder.getI64IntegerAttr(kSubBlock));
  state.addAttribute("weight_block_stride",
                     builder.getI64IntegerAttr(kWeightBlockStride));
  state.addAttribute("activation_block_stride",
                     builder.getI64IntegerAttr(kActivationBlockStride));
  state.addAttribute("weight_scales_byte_offset",
                     builder.getI64IntegerAttr(kWeightScalesByteOffset));
  state.addAttribute("weight_qs_byte_offset",
                     builder.getI64IntegerAttr(kWeightQsByteOffset));
  state.addAttribute("weight_d_byte_offset",
                     builder.getI64IntegerAttr(kWeightDByteOffset));
  state.addAttribute("weight_dmin_byte_offset",
                     builder.getI64IntegerAttr(kWeightDminByteOffset));
  state.addAttribute("activation_d_byte_offset",
                     builder.getI64IntegerAttr(kActivationDByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("activation_bsums_byte_offset",
                     builder.getI64IntegerAttr(kActivationBsumsByteOffset));
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
                  Q2KBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_q2_K_q8_K_block_dot";
  std::string fallbackVariantSymbol = "rvv_q2_K_q8_K_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_q2_K_q8_K ABI value set the super-block block-dot op
  // consumes -- n, s, vx, vy (the same FOUR the board-validated emitter input
  // declares, in declaration order n/s/vx/vy). The super-block loop, the 2-bit
  // unpack, the 4-bit nibble scale/min, and the scalar fp32 fold+min are
  // op structure; the op consumes exactly these four (vx q2_K weight base, vy q8_K
  // activation base, s fp32 output, n element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "q2-weight", runtimeABIType);
  mlir::Value vy =
      createRuntimeABIValue(builder, loc, "rhs-input-buffer", "vy",
                            "const uint8_t *", "q8-act", runtimeABIType);

  tcrvrvv::SetVLOp setvl =
      createSetVL(builder, loc, n, /*sew=*/32, "m1", policy);
  tcrvrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), /*sew=*/32, "m1", policy,
                   kernelName, selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  // The auto-constructed attr-less super-block block dot-product op (the 2-bit
  // unpack + 4-bit nibble scale/min + scalar fp32 fold+min are
  // first-class STRUCTURE inside this op; q2_K has NO shape knob -- the default
  // anchor IS the byte-exact target).
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
  return "rvv_q2_K_q8_K_block_dot_from_vector_source";
}

class MaterializeRVVQ2KBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVQ2KBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVQ2KBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVQ2KBlockDotSourceFrontDoorPass(
      const MaterializeRVVQ2KBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVQ2KBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVQ2KBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.q2_k_q8_k_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + 2-bit unpack + "
           "4-bit nibble scale/min + the scalar fp32 fold+min are first-class op "
           "structure); q2_K is NOT in any schedule autotuner, so the op lowers "
           "at the emitter's default integer-core anchor (q2_K carries no "
           "shape knob at all -- COVERAGE, not a "
           "flip)";
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
          << "RVV q2_K x q8_K super-block block-dot source front door requires "
             "an injected extension-plugin registry to dispatch the "
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
      (void)fail(module, "source module must contain exactly one RVV q2_K x "
                         "q8_K super-block block-dot source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<Q2KBlockDotSourceMatch> source =
        matchQ2KBlockDotSourceFunc(funcs.front());
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
createMaterializeRVVQ2KBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVQ2KBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVQ2KBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.q2_k_q8_k_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source; q2_K is NOT in any "
      "schedule autotuner, so the op lowers at the emitter's default anchor "
      "(COVERAGE, not a flip -- q2_K carries no shape knob at all)",
      [registryPtr] {
        return createMaterializeRVVQ2KBlockDotSourceFrontDoorPass(*registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
