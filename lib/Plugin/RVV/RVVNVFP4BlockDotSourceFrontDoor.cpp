//===- RVVNVFP4BlockDotSourceFrontDoor.cpp ------------------------------===//
//
// Track B auto-lowering, the SECOND FP4-codebook rung (the sibling of the flat
// iq4_nl codebook rung RVVIQ4NLBlockDotSourceFrontDoor -- and the one that CLOSES the
// literal block-dot zoo): the COMPILER auto-constructs the complete
// tcrv.exec.kernel + variant + dispatch/fallback scaffold around ONE
// tcrv_rvv.nvfp4_q8_0_block_dot op, from a marked GENERIC source carrying the ggml
// `ggml_vec_dot_nvfp4_q8_0` OPERATOR IDENTITY, instead of a per-kernel hand-authored
// codebook block-dot emitter input.
//
// WHAT THIS RUNG IS: NVFP4 is NVIDIA's FP4. It REUSES mxfp4's 16-entry FP4 (e2m1)
// DOUBLED int8 CODEBOOK verbatim -- the 4-bit nibble INDEXES that table via a
// vrgather, NOT an arithmetic decode -- and the SAME asymmetric signed widening
// integer core the iq4_nl/mxfp4 siblings use. The genuinely-new nvfp4 facts, all
// first-class STRUCTURE inside the op + its existing FP4 codebook emitter
// (RVVToEmitCCodebookFp4.cpp), are: (1) a QK=64 SUPER-block of four 16-element
// sub-blocks (block_nvfp4 = {uint8_t d[4]; uint8_t qs[32]}, stride 36: four UE4M3
// sub-block scales then the 32 packed FP4 nibble bytes at +4); (2) a per-sub-block
// UE4M3 (unsigned 4-exp/3-man fp8) weight scale reconstructed structurally to fp32
// (the two ldexpf branches + the *0.5f half-form matching the doubled codebook + the
// e==0/e==0x7F specials -> 0.0f), NOT mxfp4's E8M0 bit dance; (3) a FLAT block_q8_0
// activation stream (stride 34) where one super-block spans TWO q8_0 blocks. So the
// front door does NOT hand-roll any of it; it supplies the nvfp4 super-block-format
// CONSTANTS as the typed integer attrs the verifier pins AND the 16-entry codebook as
// the structural DenseI8ArrayAttr the verifier pins to size 16.
//
// STRUCTURAL FINDING -- NO NEW ROUTE FAMILY (the bucket verdict). nvfp4's FLAT
// block_q8_0 activation puts it on the EXISTING flat monolithic route family (shared
// with q4_0/iq4_nl). The codebook + the UE4M3 scale are OP attrs consumed by the
// emitter, NOT route-family concerns: the emission plan
// (buildMonolithicBlockDotEmissionPlan) and the target-export candidate validator key
// ONLY off op name -> route family + the kind/scale_model attrs + the ordered ABI
// roles; neither reads the codebook or the scale decode. So the shared Flat family +
// nvfp4's codebook/UE4M3 stamping COMPOSE cleanly: one table row (Flat + the 4-role
// ggml vec_dot ABI n/s/vx/vy) + this front door, zero new mechanism.
//
// HONEST FRAMING -- COVERAGE, NOT A NEW FLIP, and the front door STAMPS the m1
// anchor (the inversion vs the iq4_nl sibling). Unlike iq4_nl -- a
// TunableScheduleOpInterface op that leaves the constructed op attr-less and DEFERS
// the m1<->mf2 VLEN flip to the unified --tcrv-rvv-materialize-schedule gearbox --
// the nvfp4 op is NOT tunable: the verifier admits ONLY the m1 codebook-gather anchor
// (mf2's VLMAX is < 16 at VLEN=128, so there is no legal shape to flip to), and there
// is NO nvfp4 schedule autotuner. VLEN128 and VLEN256 emit the SAME bytes. But the FP4
// codebook emitter fail-closes on an attr-less op (I7: an un-anchored codebook op
// below VLEN=128 would silently gather index 0), so this front door STAMPS
// integer_core_lmul="m1" DIRECTLY on the constructed op -- the single legal anchor
// carried as an op fact, NOT a gearbox choice. The constructed op is byte-identical to
// the hand-authored CORE emitter fixture (rvv-to-emitc-nvfp4-q8-0-block-dot), modulo
// the kernel/variant symbol names. The NEW content is the auto-CONSTRUCTION feeding
// the existing nvfp4 emitter unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVNVFP4BlockDotSourceFrontDoor.h"

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

// The DISTINCT marker the source module carries to route to THIS nvfp4 FP4-codebook
// front door (NOT the MVP/dequant/q4_0/q8_0/iq4_nl/q4_K/iq4_xs markers). Each
// front-door pass checks its own marker and early-returns on a mismatch, so the
// passes are mutually exclusive and the sibling lits are byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "ggml_nvfp4_q8_0_block_dot_source");
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
    "rvv-nvfp4-q8-0-block-dot-source-front-door-case");

// The ggml nvfp4 super-block-format facts (ggml-common.h): QK_NVFP4 == 64, four
// 16-element sub-blocks (QK_NVFP4_SUB == 16), block_nvfp4 = {uint8_t d[4]; uint8_t
// qs[32]} stride 36 (four UE4M3 sub-block scale bytes then the 32 packed FP4 nibble
// bytes at +4), block_q8_0 stride 34 (fp16 d + 32 int8 quants at +2), the
// per-sub-block q8 high half QK_NVFP4_SUB/2 == 8 lanes on (+8). These are nvfp4
// CONSTANTS the front door supplies (not derivable from a generic source) and are
// the exact values the tcrv_rvv.nvfp4_q8_0_block_dot verifier pins.
constexpr std::int64_t kQK = 64;
constexpr std::int64_t kQKSub = 16;
constexpr std::int64_t kWeightBlockStride = 36;
constexpr std::int64_t kActivationBlockStride = 34;
constexpr std::int64_t kWeightQuantByteOffset = 4;
constexpr std::int64_t kActivationQuantByteOffset = 2;
constexpr std::int64_t kActivationHighByteOffset = 8;

// The 16-entry FP4 (e2m1) DOUBLED int8 codebook kvalues_mxfp4[16]: the non-linear
// int8 lookup table the codebook gather indexes (the per-element weight value is
// kvalues_mxfp4[nibble], NOT an arithmetic decode). It is the SAME table mxfp4
// carries (2 x E2M1); the *0.5f UE4M3 half-form scale compensates the doubling so the
// dot stays an exact integer dot. This is the load-bearing structural fact of the
// codebook class; the verifier pins its size to EXACTLY 16. It is supplied here as the
// nvfp4 constant a generic source could not derive, and consumed structurally into the
// realized body (a `static const int8_t[16]` decl + the vle8 table load + the
// vrgather), NOT a string plan read.
constexpr std::array<std::int8_t, 16> kNVFP4Codebook = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "ggml NVFP4 x Q8_0 codebook block-dot source front door "
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

struct NVFP4BlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_nvfp4_q8_0` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the codebook block-dot intent marker -- the WHAT is the operator
// identity (nvfp4 weight x q8_0 activation super-block dot-product -> fp32 out), NOT a
// generic dataflow body. The super-block loop + the four sub-blocks + the per-sub-block
// UE4M3 scale + the codebook gather are nvfp4 STRUCTURE the constructed op carries; the
// source func body is the bounded intent shell (it may be a bare `return`).
mlir::FailureOr<NVFP4BlockDotSourceMatch>
matchNVFP4BlockDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, nvfp4 weight memref<?xi8>, q8 "
                "activation memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "nvfp4 weight rank-1 i8 memref, q8 activation rank-1 i8 memref");

  return NVFP4BlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the tcrv_rvv.nvfp4_q8_0_block_dot op scaffold
//     (kernel + variant + dispatch/fallback). The front door STAMPS
//     integer_core_lmul="m1" -- nvfp4 is NOT in any schedule autotuner (it is not a
//     TunableScheduleOpInterface op), the codebook gather admits ONLY the m1 anchor,
//     and the FP4 emitter fail-closes on an attr-less op, so the single legal anchor
//     is carried as an op fact. This is byte-identical to the hand-authored nvfp4
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

// The ggml NVFP4 x Q8_0 super-block FP4-codebook dot-product op: the bounded WHAT
// (kind, UE4M3 half-form scale model, super-block-format facts, the 16-entry
// codebook) is stamped, AND the integer_core_lmul="m1" anchor is stamped directly --
// nvfp4 is NOT a TunableScheduleOpInterface op, the codebook gather admits ONLY the m1
// anchor (mf2's VLMAX < 16 at VLEN=128 cannot host the 16-entry gather), and the FP4
// emitter fail-closes on an attr-less op, so the single legal anchor is carried as an
// op fact rather than deferred to a gearbox. The super-block loop + the four
// sub-blocks + the UE4M3 -> fp32 scale decode + the codebook gather + the widening
// product/reduce + the per-sub-block fp32 fold are first-class STRUCTURE inside this
// op. This is byte-identical to the hand-authored nvfp4 block-dot emitter input.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc,
                             tcrvrvv::GgmlBlockDotNVFP4Q80Op::getOperationName());
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("ggml_nvfp4_q8_0_block_dot"));
  state.addAttribute("scale_model",
                     builder.getStringAttr("ue4m3-half-per-sub-block"));
  state.addAttribute("qk", builder.getI64IntegerAttr(kQK));
  state.addAttribute("qk_sub", builder.getI64IntegerAttr(kQKSub));
  state.addAttribute("weight_block_stride",
                     builder.getI64IntegerAttr(kWeightBlockStride));
  state.addAttribute("activation_block_stride",
                     builder.getI64IntegerAttr(kActivationBlockStride));
  state.addAttribute("weight_quant_byte_offset",
                     builder.getI64IntegerAttr(kWeightQuantByteOffset));
  state.addAttribute("activation_quant_byte_offset",
                     builder.getI64IntegerAttr(kActivationQuantByteOffset));
  state.addAttribute("activation_high_byte_offset",
                     builder.getI64IntegerAttr(kActivationHighByteOffset));
  state.addAttribute("codebook",
                     builder.getDenseI8ArrayAttr(llvm::ArrayRef<std::int8_t>(
                         kNVFP4Codebook.data(), kNVFP4Codebook.size())));
  state.addAttribute("integer_core_lmul", builder.getStringAttr("m1"));
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
                  NVFP4BlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_nvfp4_q8_0_block_dot";
  std::string fallbackVariantSymbol =
      "rvv_nvfp4_q8_0_block_dot_scalar_fallback";

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

  // The ggml ggml_vec_dot_nvfp4_q8_0 ABI value set the codebook block-dot op consumes
  // -- n, s, vx, vy (the same FOUR the board-validated emitter input declares, in
  // declaration order n/s/vx/vy). The super-block loop, the four sub-blocks, the
  // UE4M3 scale decode, and the codebook gather are op structure; the op consumes
  // exactly these four (vx nvfp4 weight base, vy q8 activation base, s fp32 output, n
  // element count).
  mlir::Value n = createRuntimeABIValue(builder, loc, "runtime-element-count",
                                        "n", "size_t", "n", indexType);
  mlir::Value s = createRuntimeABIValue(builder, loc, "output-buffer", "s",
                                        "float *", "out", runtimeABIType);
  mlir::Value vx =
      createRuntimeABIValue(builder, loc, "lhs-input-buffer", "vx",
                            "const uint8_t *", "nvfp4-weight", runtimeABIType);
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

  // The auto-constructed nvfp4 super-block FP4-codebook dot-product op (the UE4M3
  // scale decode + the 16-entry codebook gather + the per-sub-block fp32 fold are
  // first-class STRUCTURE inside this op; the m1 codebook-gather anchor is stamped
  // directly -- nvfp4 is NOT in any autotuner and the FP4 emitter requires it).
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
  return "rvv_nvfp4_q8_0_block_dot_from_vector_source";
}

class MaterializeRVVNVFP4BlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVNVFP4BlockDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVNVFP4BlockDotSourceFrontDoorPass() = default;
  MaterializeRVVNVFP4BlockDotSourceFrontDoorPass(
      const MaterializeRVVNVFP4BlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVNVFP4BlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVNVFP4BlockDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the tcrv_rvv.nvfp4_q8_0_block_dot op + "
           "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
           "operator-identity source (the super-block loop + the four sub-blocks + "
           "the UE4M3 -> fp32 scale decode + the 16-entry FP4 codebook gather are "
           "first-class op structure); nvfp4 is NOT in any schedule autotuner and "
           "the codebook gather admits only the m1 anchor, so the front door stamps "
           "integer_core_lmul=\"m1\" directly (COVERAGE, not a flip)";
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
          << "RVV nvfp4 x q8_0 FP4-codebook block-dot source front door requires "
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
      (void)fail(module, "source module must contain exactly one RVV nvfp4 x "
                         "q8_0 FP4-codebook block-dot source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<NVFP4BlockDotSourceMatch> source =
        matchNVFP4BlockDotSourceFunc(funcs.front());
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
createMaterializeRVVNVFP4BlockDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVNVFP4BlockDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVNVFP4BlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door",
      "Auto-construct the tcrv_rvv.nvfp4_q8_0_block_dot op + scaffold from a marked "
      "ggml vec_dot operator-identity source (the SECOND FP4-codebook rung -- "
      "mxfp4's DOUBLED e2m1 codebook + a QK=64 super-block of UE4M3-scaled "
      "sub-blocks over a flat q8_0 activation); nvfp4 is NOT tunable and the "
      "codebook gather admits only the m1 anchor, so the front door stamps "
      "integer_core_lmul=\"m1\" directly (COVERAGE, not a flip)",
      [registryPtr] {
        return createMaterializeRVVNVFP4BlockDotSourceFrontDoorPass(
            *registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
