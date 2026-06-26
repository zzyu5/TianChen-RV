//===- RVVIQ4NLBlockDotSourceFrontDoor.cpp ------------------------------===//
//
// Track B auto-lowering, the CODEBOOK rung (one step ABOVE the nibble-unpack rung
// RVVQ40BlockDotSourceFrontDoor): the COMPILER auto-constructs the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE attr-less
// tcrv_rvv.iq4_nl_q8_0_block_dot op, from a marked GENERIC source carrying the ggml
// `ggml_vec_dot_iq4_nl_q8_0` OPERATOR IDENTITY, instead of a per-kernel
// hand-authored codebook block-dot emitter input.
//
// What this rung opens vs the q4_0 sibling: the q4_0 rung's nibble decodes
// ARITHMETICALLY (`nibble - 8`, offset-binary). The CODEBOOK class does NOT: the
// 4-bit nibble INDEXES a 16-entry NON-LINEAR int8 table (ggml's kvalues_iq4nl[16])
// via a vrgather. That gather -- the table broadcast above the block loop + the
// per-lane lookup -- is FIRST-CLASS STRUCTURE inside the op and its existing FP4
// codebook emitter (RVVToEmitCCodebookFp4.cpp). So the front door does NOT hand-roll
// the gather as fragile vector ops; it supplies the 16-entry codebook as the
// structural DenseI8ArrayAttr the op verifier pins, and the gather/product/reduce
// chain is carried by the op.
//
// THE CAPABILITY FLIP IS REAL HERE -- the honest inversion vs the q4_0 front door.
// q4_0 anchors at m1 at EVERY Zvl128b tier (no VLEN-byte flip). The codebook gather
// must index ALL 16 table entries, so its legal integer-core anchor is the
// VLEN-capability fact "the strip VLMAX spans the 16-entry table-index range": at
// VLEN128 only m1 reaches VLMAX 16 (mf2 -> 8 < 16, PRUNED); at VLEN256 mf2 ALSO
// reaches VLMAX 16 (the ggml `_vl256` shape) and wins the capability-blind cost tie
// on the lighter peak-live footprint. So the SAME attr-less op FLIPS m1@VLEN128 ->
// mf2@VLEN256 -- a byte-different emitted codebook core (vrgather_vv_i8m1 vs
// vrgather_vv_i8mf2). This rung CLEARS the non-NULL bar the q4_0 rung could not.
//
// The flip is the EXISTING unified schedule autotuner's, already board-sealed in
// rvv-iq4-nl-q8-0-block-dot-autotuner-divergence.mlir. This front door constructs
// the ATTR-LESS op (no integer_core_lmul / multi_block_factor / strip_elision /
// minimum_vlen) and DEFERS shape selection to the unmodified
// --tcrv-rvv-materialize-schedule pass: the constructed op is byte-identical to
// that pass's hand-authored input op (modulo the variant symbol name), so the
// capability flip rides the existing gearbox byte-for-byte with NO re-implemented
// selection to drift. The NEW content is the auto-CONSTRUCTION feeding that
// gearbox; the front door REPRODUCES the sealed flip, it does not invent one.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVIQ4NLBlockDotSourceFrontDoor.h"

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

// The DISTINCT marker the source module carries to route to THIS iq4_nl codebook
// front door (NOT the MVP/dequant/q4_0 markers). Each front-door pass checks its own
// marker and early-returns on a mismatch, so the passes are mutually exclusive and
// the sibling lits are byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_iq4_nl_q8_0_block_dot_source");
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
    "rvv-iq4-nl-q8-0-block-dot-source-front-door-case");

// The ggml block-format facts (ggml-common.h): QK4_NL == QK8_0 == 32, block_iq4_nl
// stride 18 (byte-identical SHAPE to block_q4_0: fp16 d + 16 nibble bytes),
// block_q8_0 stride 34, quants at byte offset +2, q8 high half at +16. These are
// iq4_nl CONSTANTS the front door supplies (not derivable from a generic source) and
// are the exact values the tcrv_rvv.iq4_nl_q8_0_block_dot verifier pins.
constexpr std::int64_t kQK = 32;
constexpr std::int64_t kWeightBlockStride = 18;
constexpr std::int64_t kActivationBlockStride = 34;
constexpr std::int64_t kQuantByteOffset = 2;
constexpr std::int64_t kActivationHighByteOffset = 16;

// ggml's kvalues_iq4nl[16]: the 16-entry NON-LINEAR int8 lookup table the codebook
// gather indexes (the per-element weight value is kvalues_iq4nl[nibble], NOT an
// arithmetic decode). This is the load-bearing structural fact of the codebook class;
// the verifier pins its size to EXACTLY 16. It is supplied here as the iq4_nl
// constant a generic source could not derive, and consumed structurally into the
// realized body (a `static const int8_t[16]` decl + the vle8 table load + the
// vrgather), NOT a string plan read.
constexpr std::array<std::int8_t, 16> kIQ4NLCodebook = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml IQ4_NL x Q8_0 codebook block-dot source front door "
                     "failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the block-dot has no compact generic
//     vector form, so recognition is by the vec_dot ABI roles, not by a
//     straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct IQ4NLBlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_iq4_nl_q8_0` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the codebook block-dot intent marker -- the WHAT is the operator
// identity (iq4_nl weight x q8_0 activation block dot-product -> fp32 out), NOT a
// generic dataflow body. The block loop + the per-block dual fp16 scale + the
// codebook gather are iq4_nl STRUCTURE the constructed op carries; the source func
// body is the bounded intent shell (it may be a bare `return`).
mlir::FailureOr<IQ4NLBlockDotSourceMatch>
matchIQ4NLBlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, iq4_nl weight memref<?xi8>, q8 "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "iq4_nl weight rank-1 i8 memref, q8 activation rank-1 i8 memref");

  return IQ4NLBlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the attr-less tcrv_rvv.iq4_nl_q8_0_block_dot op
//     scaffold (kernel + variant + dispatch/fallback). The SHAPE knobs are left
//     OFF: shape selection (and the m1<->mf2 VLEN flip) is the existing unified
//     autotuner's job, so the capability flip rides that pass unchanged.
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

// The ATTR-LESS ggml IQ4_NL x Q8_0 codebook block dot-product op: the bounded WHAT
// (kind, scale model, block-format facts, the 16-entry codebook) is stamped, but NO
// shape knobs (integer_core_lmul / multi_block_factor / strip_elision /
// minimum_vlen) -- so isSchedulePinned() is false, minimum_vlen defaults to the
// conservative 128 floor, and the existing --tcrv-rvv-materialize-schedule autotuner
// is free to select the capability-driven shape (and the m1<->mf2 VLEN flip). This is
// byte-identical to the autotuner-divergence lit's hand-authored input op.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc,
                             tcrvrvv::GgmlBlockDotIQ4NLQ80Op::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_iq4_nl_q8_0_block_dot"));
  state.addAttribute("scale_model",
                     builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
  state.addAttribute("qk", builder.getI64IntegerAttr(kQK));
  state.addAttribute("weight_block_stride",
                     builder.getI64IntegerAttr(kWeightBlockStride));
  state.addAttribute("activation_block_stride",
                     builder.getI64IntegerAttr(kActivationBlockStride));
  state.addAttribute("quant_byte_offset",
                     builder.getI64IntegerAttr(kQuantByteOffset));
  state.addAttribute("activation_high_byte_offset",
                     builder.getI64IntegerAttr(kActivationHighByteOffset));
  state.addAttribute("codebook",
                     builder.getDenseI8ArrayAttr(llvm::ArrayRef<std::int8_t>(
                         kIQ4NLCodebook.data(), kIQ4NLCodebook.size())));
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
                  IQ4NLBlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_iq4_nl_q8_0_block_dot";
  std::string fallbackVariantSymbol =
      "rvv_iq4_nl_q8_0_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_iq4_nl_q8_0 ABI value set the codebook block-dot op
  // consumes -- n, s, vx, vy (the same FOUR the board-validated emitter input
  // declares, in declaration order n/s/vx/vy). The codebook gather, the block loop,
  // and the dual fp16 scale are op structure; the op consumes exactly these four
  // (vx weight base, vy q8 activation base, s fp32 output, n element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "iq4-weight", runtimeABIType);
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

  // The auto-constructed attr-less codebook block dot-product op (the 16-entry
  // codebook gather + per-block fp16 scale + reduce are first-class STRUCTURE inside
  // this op; the m1<->mf2 VLEN anchor flip is the autotuner's, deferred here).
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
  return "rvv_iq4_nl_q8_0_block_dot_from_vector_source";
}

class MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass() = default;
  MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass(
      const MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the attr-less tcrv_rvv.iq4_nl_q8_0_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the 16-entry codebook gather + per-block "
           "fp16 scale are first-class op structure); shape selection and the "
           "m1<->mf2 VLEN anchor flip are left to the existing capability-driven "
           "--tcrv-rvv-materialize-schedule autotuner";
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
          << "RVV iq4_nl x q8_0 codebook block-dot source front door requires "
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
      (void)fail(module, "source module must contain exactly one RVV iq4_nl x "
                         "q8_0 codebook block-dot source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<IQ4NLBlockDotSourceMatch> source =
        matchIQ4NLBlockDotSourceFunc(funcs.front());
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
createMaterializeRVVIQ4NLBlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVIQ4NLBlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVIQ4NLBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door",
      "Auto-construct the attr-less tcrv_rvv.iq4_nl_q8_0_block_dot op + scaffold "
      "from a marked ggml vec_dot operator-identity source; the REAL m1<->mf2 "
      "VLEN capability flip rides the existing unified schedule autotuner",
      [registryPtr] {
        return createMaterializeRVVIQ4NLBlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
