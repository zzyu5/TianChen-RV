//===- RVVIQ4XSBlockDotSourceFrontDoor.cpp ------------------------------===//
//
// Track B auto-lowering, the SUPER-BLOCK-CODEBOOK rung -- the INTERSECTION of the
// two rungs already in hand: the SUPER-BLOCK rung (RVVQ4KBlockDotSourceFrontDoor)
// and the CODEBOOK rung (RVVIQ4NLBlockDotSourceFrontDoor). The COMPILER
// auto-constructs the complete tcrv.exec.kernel + variant + dispatch/fallback
// scaffold around ONE attr-less tcrv_rvv.iq4_xs_q8_k_block_dot op, from a marked
// GENERIC source carrying the ggml `ggml_vec_dot_iq4_xs_q8_K` OPERATOR IDENTITY,
// instead of a per-kernel hand-authored super-block-codebook block-dot emitter
// input.
//
// WHAT THIS RUNG COMBINES: iq4_xs is the SUPER-BLOCK variant of iq4_nl. It REUSES
// iq4_nl's 16-entry non-linear int8 CODEBOOK (the SAME kvalues_iq4nl[16] the 4-bit
// nibble INDEXES via vrgather -- NOT an arithmetic decode) but wraps that
// codebook-gather integer core in the q4_K/q6_K super-block structure (QK_K==256,
// 8 sub-blocks of 32, the q8_K activation with its leading fp32 scale). The one
// genuinely-new super-block piece -- the iq4_xs per-sub-block SIGNED 6-bit scale
// extracted from scales_l[4]+scales_h (the q4_K-style cross-byte bit dance) biased
// -32 and applied in the FLOAT domain (NO min term, symmetric like q6_K) -- is
// first-class STRUCTURE inside the op + its existing iq4_xs emitter. So the front
// door does NOT hand-roll any of it; it supplies (a) the iq4_xs super-block-format
// CONSTANTS as the typed integer attrs the verifier pins AND (b) the 16-entry
// codebook as the structural DenseI8ArrayAttr the verifier pins to size 16.
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY (the bucket verdict). The super-block-
// codebook op takes the EXISTING super-block monolithic route family (shared with
// q4_K). The codebook is an OP attr consumed by the emitter, NOT a route-family
// concern: the emission plan (buildMonolithicBlockDotEmissionPlan) and the target
// export candidate validator both key ONLY off the op name -> route family + the
// kind/scale_model attrs + the ordered ABI roles; neither reads the codebook. So
// the shared SuperBlock family + iq4_nl's codebook-attr stamping COMPOSE cleanly:
// one table row (SuperBlock, the 4-role ggml vec_dot ABI) + this front door, zero
// new mechanism.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP (the q4_K framing, the INVERSE of
// iq4_nl's). iq4_xs is NOT in any schedule-descriptor autotuner. The codebook
// gather is inherently a Zvl128b legality fact (the broadcast codebook register's
// VLMAX must be >= 16 to index all 16 table entries), so the op PINS the m1
// integer-core anchor -- m1 is a capability-legality fact carried by the op, NOT a
// gearbox-selected or VLEN-flipped shape. So this front door does NOT ride a
// gearbox (unlike iq4_nl): there is no iq4_xs schedule autotuner. The constructed
// attr-less op lowers DIRECTLY at the emitter's m1 anchor, VLEN-independent
// (VLEN128 and VLEN256 emit the SAME bytes). The NEW content is the
// auto-CONSTRUCTION feeding the existing iq4_xs emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVIQ4XSBlockDotSourceFrontDoor.h"

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

// The DISTINCT marker the source module carries to route to THIS iq4_xs super-block
// codebook front door (NOT the MVP/dequant/q4_0/q8_0/iq4_nl/q4_K markers). Each
// front-door pass checks its own marker and early-returns on a mismatch, so the
// passes are mutually exclusive and the sibling lits are byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_iq4_xs_q8_K_block_dot_source");
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
    "rvv-iq4-xs-q8-k-block-dot-source-front-door-case");

// The ggml iq4_xs super-block-format facts (ggml-common.h): QK_K == 256, 8
// sub-blocks of 32 elements, block_iq4_xs stride 136 (fp16 d @0 | uint16 scales_h
// @2 | 4 scales_l bytes @4 | 128 nibble bytes @8), block_q8_K stride 292 (fp32 d
// @0 | 256 int8 qs @4 | 16 int16 bsums @260 UNUSED -- iq4_xs is symmetric, no min
// term). These are iq4_xs CONSTANTS the front door supplies (not derivable from a
// generic source) and are the exact values the tcrv_rvv.iq4_xs_q8_k_block_dot
// verifier pins. NOTE casing: the op kind is the LOWER-k form
// "ggml_iq4_xs_q8_k_block_dot"; the kernel/marker/variant symbols carry the
// canonical ggml UPPER-K "q8_K".
constexpr std::int64_t kQK = 256;
constexpr std::int64_t kSubBlock = 32;
constexpr std::int64_t kWeightBlockStride = 136;
constexpr std::int64_t kActivationBlockStride = 292;
constexpr std::int64_t kWeightDByteOffset = 0;
constexpr std::int64_t kWeightScalesHByteOffset = 2;
constexpr std::int64_t kWeightScalesLByteOffset = 4;
constexpr std::int64_t kWeightQsByteOffset = 8;
constexpr std::int64_t kActivationDByteOffset = 0;
constexpr std::int64_t kActivationQuantByteOffset = 4;

// ggml's kvalues_iq4nl[16]: the 16-entry NON-LINEAR int8 lookup table the codebook
// gather indexes (the per-element weight value is kvalues_iq4nl[nibble], NOT an
// arithmetic decode) -- iq4_xs REUSES iq4_nl's table VERBATIM. This is the
// load-bearing structural fact of the codebook class; the verifier pins its size
// to EXACTLY 16. It is supplied here as the iq4_xs constant a generic source could
// not derive, and consumed structurally into the realized body (a
// `static const int8_t[16]` decl + the vle8 table load + the vrgather), NOT a
// string plan read.
constexpr std::array<std::int8_t, 16> kIQ4XSCodebook = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml IQ4_XS x Q8_K super-block codebook block-dot source "
                     "front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the super-block-codebook dot has no
//     compact generic vector form, so recognition is by the vec_dot ABI roles,
//     not by a straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct IQ4XSBlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_iq4_xs_q8_K` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the super-block-codebook block-dot intent marker -- the WHAT is the
// operator identity (iq4_xs weight x q8_K activation super-block dot-product ->
// fp32 out), NOT a generic dataflow body. The super-block loop + the signed 6-bit
// scale extraction + the codebook gather + the float-domain fold are iq4_xs
// STRUCTURE the constructed op carries; the source func body is the bounded intent
// shell (it may be a bare `return`).
mlir::FailureOr<IQ4XSBlockDotSourceMatch>
matchIQ4XSBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, iq4_xs weight memref<?xi8>, q8_K "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "iq4_xs weight rank-1 i8 memref, q8_K activation rank-1 i8 memref");

  return IQ4XSBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.iq4_xs_q8_k_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). NO shape knob: iq4_xs is
//     NOT in any schedule autotuner, and the codebook gather pins m1 in the op
//     emitter, so the constructed op lowers at the emitter's m1 anchor. This is
//     byte-identical to the hand-authored iq4_xs block-dot emitter input.
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

// The ATTR-LESS ggml IQ4_XS x Q8_K super-block-codebook dot-product op: the bounded
// WHAT (kind, scale model, super-block-format facts, the 16-entry codebook) is
// stamped, but NO shape knob -- iq4_xs is NOT in any schedule autotuner and the
// codebook gather pins m1 in the emitter, so the op lowers at the emitter's m1
// integer-core anchor. The super-block loop + the signed 6-bit scale bit-dance +
// the codebook gather + the widening product/reduce + the float-domain fold are
// first-class STRUCTURE inside this op. The codebook is the SAME kvalues_iq4nl[16]
// table iq4_nl carries -- supplied here as the structural DenseI8ArrayAttr the
// verifier pins to size 16.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc,
                             tcrvrvv::GgmlBlockDotIQ4XSQ8KOp::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_iq4_xs_q8_k_block_dot"));
  state.addAttribute(
      "scale_model",
      builder.getStringAttr(
          "per-sub-block-int6-signed-codebook-scale-float-domain"));
  state.addAttribute("qk", builder.getI64IntegerAttr(kQK));
  state.addAttribute("sub_block", builder.getI64IntegerAttr(kSubBlock));
  state.addAttribute("weight_block_stride",
                     builder.getI64IntegerAttr(kWeightBlockStride));
  state.addAttribute("activation_block_stride",
                     builder.getI64IntegerAttr(kActivationBlockStride));
  state.addAttribute("weight_d_byte_offset",
                     builder.getI64IntegerAttr(kWeightDByteOffset));
  state.addAttribute("weight_scales_h_byte_offset",
                     builder.getI64IntegerAttr(kWeightScalesHByteOffset));
  state.addAttribute("weight_scales_l_byte_offset",
                     builder.getI64IntegerAttr(kWeightScalesLByteOffset));
  state.addAttribute("weight_qs_byte_offset",
                     builder.getI64IntegerAttr(kWeightQsByteOffset));
  state.addAttribute("activation_d_byte_offset",
                     builder.getI64IntegerAttr(kActivationDByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("codebook",
                     builder.getDenseI8ArrayAttr(llvm::ArrayRef<std::int8_t>(
                         kIQ4XSCodebook.data(), kIQ4XSCodebook.size())));
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
                  IQ4XSBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_iq4_xs_q8_K_block_dot";
  std::string fallbackVariantSymbol = "rvv_iq4_xs_q8_K_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_iq4_xs_q8_K ABI value set the super-block-codebook
  // block-dot op consumes -- n, s, vx, vy (the same FOUR the board-validated
  // emitter input declares, in declaration order n/s/vx/vy). The super-block loop,
  // the signed 6-bit scale bit-dance, the codebook gather, and the float-domain
  // fold are op structure; the op consumes exactly these four (vx iq4_xs weight
  // base, vy q8_K activation base, s fp32 output, n element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "iq4xs-weight", runtimeABIType);
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

  // The auto-constructed attr-less super-block-codebook block dot-product op (the
  // signed 6-bit scale bit-dance + the 16-entry codebook gather + the float-domain
  // fold are first-class STRUCTURE inside this op; the codebook gather pins m1 in
  // the emitter -- no shape knob is stamped, iq4_xs is NOT in any autotuner).
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
  return "rvv_iq4_xs_q8_K_block_dot_from_vector_source";
}

class MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass(
      const MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.iq4_xs_q8_k_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + signed 6-bit scale "
           "bit-dance + the 16-entry codebook gather + the float-domain fold are "
           "first-class op structure); iq4_xs is NOT in any schedule autotuner and "
           "the codebook gather pins m1 in the emitter, so the op lowers at the "
           "emitter's m1 integer-core anchor (COVERAGE, not a flip)";
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
          << "RVV iq4_xs x q8_K super-block codebook block-dot source front door "
             "requires an injected extension-plugin registry to dispatch the "
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
      (void)fail(module, "source module must contain exactly one RVV iq4_xs x "
                         "q8_K super-block codebook block-dot source function "
                         "candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<IQ4XSBlockDotSourceMatch> source =
        matchIQ4XSBlockDotSourceFunc(funcs.front());
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
createMaterializeRVVIQ4XSBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVIQ4XSBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVIQ4XSBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.iq4_xs_q8_k_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source (the SUPER-BLOCK-"
      "CODEBOOK rung -- q4_K's super-block scaffold + iq4_nl's 16-entry codebook); "
      "iq4_xs is NOT in any schedule autotuner, so the op lowers at the emitter's "
      "m1 codebook-gather anchor (COVERAGE, not a flip)",
      [registryPtr] {
        return createMaterializeRVVIQ4XSBlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
