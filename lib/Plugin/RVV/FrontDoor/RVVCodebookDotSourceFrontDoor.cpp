//===- RVVCodebookDotSourceFrontDoor.cpp --------------------------------===//
//
// Track B G2, the BOUNDED first step: the COMPILER auto-CONSTRUCTS the codebook
// (vrgather) INTEGER-CORE body from a marked GENERIC source carrying the codebook
// -core operator identity, instead of routing a monolithic op to a per-kernel
// hand emitter. The auto-constructed body is the single-strip generic-op
// composition
//   tcrv_rvv.codebook_table_broadcast (the 16-entry kvalues table -> values vreg)
//   load x3 (UNSIGNED packed-i4 weight + the two plain-i8 q8 activation halves)
//     -> tcrv_rvv.codebook_gather_x_i8_product (the nibble split + vrgather
//        codebook decode + asymmetric widening product -- the codebook variant of
//        the nibble core, REUSING the SAME emitOffsetBinaryProductFromDecodedValue
//        product tail the hand-written block-dot strip calls)
//     -> tcrv_rvv.standalone_reduce (signed widening reduce, i16 -> i32)
//     -> tcrv_rvv.store
// the unchanged --tcrv-rvv-lower-to-emitc emitter consumes verbatim. This proves
// the generic auto-construction mechanism (proven for q4_0 nibble in G1) reaches a
// STRUCTURALLY DIFFERENT family: the codebook is NOT the q4_0 xor/sll/sra
// arithmetic decode; each 4-bit nibble is an INDEX into a non-linear int8 table
// gathered by vrgather.
//
// HONEST SCOPE -- the codebook integer CORE only (NOT the full codebook KERNEL).
// There is NO nb = n / QK outer block loop, NO per-block fp16 scale read, NO fp32
// fold, and NO once-above-loop table hoisting (the table_broadcast is emitted
// per-strip here, vs hoisted above the block loop in the monolithic kernel); those
// axes need NEW generic ODS vocabulary and are full G2, DEFERRED. The monolithic
// tcrv_rvv.iq4_nl_q8_0_block_dot / mxfp4 / nvfp4 ops, their KERNEL front doors, and
// the hand emitters (RVVToEmitCCodebookFp4.cpp) all STAY -- this adds a SEPARATE
// rung-3 front door, like dequant-vs-reduction.
//
// CAPABILITY framing -- the codebook DOES flip (the q4_0 sibling did NOT). The
// capability consultation is the SAME shared schedule authority the rung-1/2 front
// doors use (selectCodebookCoreLMUL: enumerateRVVCodebookShapeCandidates +
// selectGenericSchedule fed deriveMinimumVLEN(march)), but here the SELECTED i8
// anchor is THREADED into the body types (NOT pinned). At VLEN128 only m1 reaches
// VLMAX 16 (the 16-entry table needs every lane indexable; mf2 -> 8 < 16 is
// PRUNED), so the emit is vrgather_vv_i8m1 + i16m2 product; at VLEN256 mf2 is
// admitted and its lighter footprint wins, so the emit is vrgather_vv_i8mf2 + i16m1
// product. The VLEN128 vs VLEN256 emit DIFFER -- the genuine capability flip,
// demonstrated at the bounded-core granularity (the property the board-sealed FP4
// brick #2 records). At VLEN0 every candidate is pruned -> fail-closed (I7). This
// is MECHANISM/parity, NOT a speed beat (G5).
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVCodebookDotSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVGearboxSchedule.h"
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

// The DISTINCT marker the source module carries to route to THIS codebook-gather
// integer-core front door (mutually exclusive with the MVP / dequant / packed-i4
// nibble / q4_0-KERNEL / iq4_nl-KERNEL markers). Each front-door pass checks its
// own marker and early-returns on a mismatch, so every existing front-door lit is
// byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "bounded_codebook_gather_dot_source");
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
    "rvv-codebook-gather-dot-source-front-door-case");

// The codebook-gather product op's pinned bounded-surface attrs (the verifier
// requires these exact strings; the LMUL is NOT pinned -- it flips with VLEN).
constexpr llvm::StringLiteral kCodebookProductKind(
    "signed_codebook_gather_x_i8_product");
constexpr llvm::StringLiteral kCodebookProductRelation(
    "codebook-gather-i8-x-i8x2-to-i16");

// The structured-const C symbol the codebook table decl declares + ggml's
// kvalues_iq4nl[16] -- the 16-entry NON-LINEAR int8 lookup table the codebook
// gather indexes (the load-bearing structural fact of the codebook class). Pinned
// to the iq4_nl table so the auto-constructed core is byte-comparable to the
// existing iq4_nl block-dot codebook-decode chain.
constexpr llvm::StringLiteral kCodebookTableSymbol("tcrv_iq4_nl_kvalues");
constexpr std::array<std::int8_t, 16> kIQ4NLCodebook = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113};

// The standalone-reduction strip framing the with_vl/setvl carry (the i32m1
// reduction accumulator); SAME at every VLEN tier (the flip is in the i8 gather /
// i16 product anchors, NOT the reduction).
constexpr std::int64_t kReductionStripSEW = 32;
constexpr llvm::StringLiteral kReductionStripLMUL("m1");
constexpr llvm::StringLiteral kReductionLMUL("m1");

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "bounded RVV codebook-gather dot source front door failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the codebook INTEGER-CORE
//     operator identity (six ABI roles, like the packed-i4 nibble core).
//===----------------------------------------------------------------------===//

struct CodebookDotSourceMatch {
  mlir::func::FuncOp func;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
}

// Match the codebook integer-core OPERATOR-IDENTITY source signature:
//   func(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>,
//        %acc: memref<?xi32>, %out: memref<?xi32>, %n: index)
// holding ONLY the codebook-core intent marker. The six roles are the packed-i4
// weight (each byte two table-index nibbles), the two plain-i8 q8 activation
// halves paired with the low/high nibbles, the i32 accumulator seed, the i32
// output, and the runtime element count. The body is the bounded intent shell (it
// may be a bare `return`): the codebook split + gather are STRUCTURE the
// constructed codebook_gather op carries, so this front door does NOT pretend to
// recognize the gather from generic vector ops.
mlir::FailureOr<CodebookDotSourceMatch>
matchCodebookDotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func, "source function must have a body for the codebook "
                      "integer-core intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 6 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly six inputs and no results: "
                "weight/qlo/qhi memref<?xi8>, acc memref<?xi32>, out "
                "memref<?xi32>, and n index (the codebook integer-core operator "
                "identity)");
  if (!isRank1MemRef(type.getInput(0), 8) ||
      !isRank1MemRef(type.getInput(1), 8) ||
      !isRank1MemRef(type.getInput(2), 8) ||
      !isRank1MemRef(type.getInput(3), 32) ||
      !isRank1MemRef(type.getInput(4), 32) || !type.getInput(5).isIndex())
    return fail(func,
                "source function inputs must be a packed-i4 weight rank-1 i8 "
                "memref, two plain-i8 q8 activation-half rank-1 i8 memrefs, an "
                "i32 acc seed rank-1 memref, an i32 out rank-1 memref, and an n "
                "index");

  return CodebookDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) The capability-fact-driven codebook integer-core ANCHOR selection (FLIP).
//===----------------------------------------------------------------------===//

// Run the SHARED codebook schedule authority to select the codebook i8 gather
// anchor at this VLEN tier. enumerateRVVCodebookShapeCandidates enumerates the
// {m1, mf2} anchor set and PRUNES any whose strip VLMAX < 16 (the gather must
// index the whole 16-entry table). The RETURN VALUE is THREADED into the body
// types (the genuine flip): m1 at VLEN128 (mf2 pruned, VLMAX 8 < 16), mf2 at
// VLEN256 (both legal, the lighter footprint wins). Fail-closed (nullopt -> I7)
// at VLEN0 where every candidate is pruned (the codebook class is Zvl128b-gated).
std::optional<std::string>
selectCodebookCoreLMUL(llvm::StringRef march, llvm::StringRef isaVectorHints) {
  std::int64_t minimumVLEN = deriveMinimumVLEN(march, isaVectorHints);

  llvm::SmallVector<RVVBlockDotShapeCandidate, 12> typed =
      enumerateRVVCodebookShapeCandidates(
          minimumVLEN, kRVVCodebookShapeVectorRegisterBudget);
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (const RVVBlockDotShapeCandidate &candidate : typed)
    candidates.push_back(toGenericBlockDotCandidate(candidate));

  static constexpr llvm::StringRef kRequiredKnobKeys[] = {"lmul"};
  std::optional<GenericScheduleSelection> selected = selectGenericSchedule(
      candidates, /*recordText=*/std::nullopt,
      /*kernelKey=*/"codebook_gather_x_i8_product", march, kRequiredKnobKeys);
  if (!selected)
    return std::nullopt; // every candidate pruned -> fail-closed (I7).

  for (const NamedKnob &knob : selected->candidate.knobs)
    if (knob.recordKey == "lmul")
      return knob.value;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// (3) Body builder: auto-construct the tcrv_rvv codebook integer-core body.
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

tcrvrvv::RuntimeABIValueOp
createRuntimeABIValue(mlir::OpBuilder &builder, mlir::Location loc,
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
  return llvm::cast<tcrvrvv::RuntimeABIValueOp>(builder.create(state));
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

mlir::Value createRVVLoad(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value buffer, mlir::Value vl,
                          mlir::Type vectorType) {
  mlir::OperationState state(loc, tcrvrvv::LoadOp::getOperationName());
  state.addOperands({buffer, vl});
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

// The codebook table broadcast: the 16-entry non-linear int8 table materialized
// as a structured const + broadcast-loaded into the `values` vreg the gather
// indexes (FIRST-CLASS codebook STRUCTURE, distinct from the q4_0 arithmetic
// decode which needs no table).
mlir::Value createCodebookTableBroadcast(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::ArrayRef<std::int8_t> codebook,
                                         llvm::StringRef tableSymbol,
                                         mlir::Type tableType) {
  mlir::OperationState state(
      loc, tcrvrvv::CodebookTableBroadcastOp::getOperationName());
  state.addAttribute("codebook", builder.getDenseI8ArrayAttr(codebook));
  state.addAttribute("table_symbol", builder.getStringAttr(tableSymbol));
  state.addTypes(tableType);
  return builder.create(state)->getResult(0);
}

// The codebook integer core's auto-constructed step: the nibble split + vrgather
// codebook decode + asymmetric widening product (the FIRST-CLASS generic op that
// carries the codebook gather as typed STRUCTURE, lowered by the SAME asymmetric
// product tail the hand-written block-dot strip calls). The compiler reaches the
// codebook gather by CONSTRUCTING this op -- no new emitter vocabulary, no
// hand-rolled vand/vsrl/vrgather.
mlir::Value createCodebookGatherProduct(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value weight,
                                        mlir::Value activationLow,
                                        mlir::Value activationHigh,
                                        mlir::Value table, mlir::Value vl,
                                        mlir::Type productType) {
  mlir::OperationState state(
      loc, tcrvrvv::CodebookGatherXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, table, vl});
  state.addAttribute("kind", builder.getStringAttr(kCodebookProductKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(kCodebookProductRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

mlir::Value createStandaloneReduce(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value input, mlir::Value accumulatorSeed,
                                   mlir::Value vl, mlir::Type resultType) {
  mlir::OperationState state(loc,
                             tcrvrvv::StandaloneReduceOp::getOperationName());
  state.addOperands({input, accumulatorSeed, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("signed_widening_reduce_add"));
  state.addAttribute(
      "accumulator_layout",
      builder.getStringAttr("scalar-i32-seed-lane0-from-accumulator-input"));
  state.addAttribute(
      "result_layout",
      builder.getStringAttr("store-standalone-reduction-lane0-to-output-scalar"));
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

void createRVVStore(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value buffer, mlir::Value value, mlir::Value vl) {
  mlir::OperationState state(loc, tcrvrvv::StoreOp::getOperationName());
  state.addOperands({buffer, value, vl});
  (void)builder.create(state);
}

tcrvexec::VariantOp
createVariant(mlir::OpBuilder &builder, mlir::Location loc,
              llvm::StringRef selectedVariantSymbol, mlir::ArrayAttr requires,
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
                  CodebookDotSourceMatch source, llvm::StringRef coreLMUL) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_codebook_gather_dot_i8";
  std::string fallbackVariantSymbol =
      "rvv_codebook_gather_dot_i8_scalar_fallback";

  // The THREADED codebook integer-core widening ladder: the UNSIGNED packed-i4
  // weight + the codebook table + the plain-i8 activation halves load at the
  // gearbox-selected i8 anchor (coreLMUL: m1 at VLEN128, mf2 at VLEN256), the
  // asymmetric widening product is the next-wider i16 rung (m2 / m1 -- the flip),
  // and the signed widening reduce is the fixed i32 m1 accumulator. The strip
  // vsetvl the reduction issues is e32m1 (the i32m1 standalone-reduction framing,
  // unchanged across tiers; the flip is in the i8 gather / i16 product).
  llvm::StringRef loadLMUL = coreLMUL;
  llvm::StringRef productLMUL = getRVVNextWiderLMUL(loadLMUL);
  if (productLMUL.empty())
    return fail(source.func,
                llvm::Twine("no wider i16 LMUL rung for codebook integer-core i8 "
                            "anchor '") +
                    loadLMUL + "'");

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
  // AUDIT-ONLY provenance (NOT the route/dtype authority): the THREADED codebook
  // integer-core anchor the body realizes -- the genuine flip form (i8m1-i16m2 at
  // VLEN128, i8mf2-i16m1 at VLEN256).
  rvvVariant->setAttr(
      "tcrv_rvv.codebook_integer_core_anchor",
      builder.getStringAttr(
          ("i8" + loadLMUL + "-i16" + productLMUL + "-i32m1-vlen-flip").str()));
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrvrvv::RuntimeABIValueType::get(builder.getContext());
  // The six codebook integer-core ABI roles, in the order the emitted C signature
  // pins (so the output buffer is arg4): the UNSIGNED packed-i4 weight, the two
  // plain-i8 q8 activation halves, the i32 acc seed, the i32 out, and the runtime
  // element count. The weight C type is `const uint8_t *` (the gather index lanes
  // run on the u8 lane).
  auto weight = createRuntimeABIValue(builder, loc, "lhs-input-buffer", "w",
                                      "const uint8_t *", "q4-weight",
                                      runtimeABIType);
  auto qlo =
      createRuntimeABIValue(builder, loc, "rhs-input-buffer", "qlo",
                            "const int8_t *", "q8-low", runtimeABIType);
  auto qhi =
      createRuntimeABIValue(builder, loc, "rhs-input-buffer", "qhi",
                            "const int8_t *", "q8-high", runtimeABIType);
  auto acc =
      createRuntimeABIValue(builder, loc, "accumulator-input-buffer", "acc",
                            "const int32_t *", "acc", runtimeABIType);
  auto out = createRuntimeABIValue(builder, loc, "output-buffer", "out",
                                   "int32_t *", "out", runtimeABIType);
  auto n = createRuntimeABIValue(builder, loc, "runtime-element-count", "n",
                                 "size_t", "n", builder.getIndexType());

  tcrvrvv::SetVLOp setvl =
      createSetVL(builder, loc, n.getResult(), kReductionStripSEW,
                  kReductionStripLMUL, policy);
  tcrvrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), kReductionStripSEW,
                   kReductionStripLMUL, policy, kernelName,
                   selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  mlir::Type ui8VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getIntegerType(8, /*isSigned=*/false),
      loadLMUL);
  mlir::Type i8VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getI8Type(), loadLMUL);
  mlir::Type i16VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getI16Type(), productLMUL);
  mlir::Type i32VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), kReductionLMUL);

  // The codebook table broadcast (the 16-entry kvalues -> i8 values vreg), then
  // the three source loads: the UNSIGNED packed-i4 weight + the two plain-i8 q8
  // activation halves (NOT codebook-decoded -- only the weight is).
  mlir::Value table = createCodebookTableBroadcast(
      builder, loc, kIQ4NLCodebook, kCodebookTableSymbol, i8VecType);
  mlir::Value loadedWeight =
      createRVVLoad(builder, loc, weight.getResult(), setvl.getVl(), ui8VecType);
  mlir::Value loadedQLow =
      createRVVLoad(builder, loc, qlo.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedQHigh =
      createRVVLoad(builder, loc, qhi.getResult(), setvl.getVl(), i8VecType);
  // The auto-constructed codebook nibble split + vrgather decode + asymmetric
  // widening product -> i16/wider.
  mlir::Value product = createCodebookGatherProduct(
      builder, loc, loadedWeight, loadedQLow, loadedQHigh, table, setvl.getVl(),
      i16VecType);
  // The signed widening reduce -> i32/m1, carrying the i32 acc seed.
  mlir::Value reduced = createStandaloneReduce(
      builder, loc, product, acc.getResult(), setvl.getVl(), i32VecType);
  createRVVStore(builder, loc, out.getResult(), reduced, setvl.getVl());

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
// (4) The pass: marker-gated, march-option-driven.
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
  return "rvv_codebook_gather_dot_i8_from_source";
}

class MaterializeRVVCodebookDotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVCodebookDotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVCodebookDotSourceFrontDoorPass() = default;
  MaterializeRVVCodebookDotSourceFrontDoorPass(
      const MaterializeRVVCodebookDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVCodebookDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVCodebookDotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-codebook-gather-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the tcrv_rvv codebook (vrgather) integer-CORE body "
           "(codebook_table_broadcast + load x3 + codebook_gather_x_i8_product "
           "+ standalone_reduce + store) from a marked generic codebook-core "
           "source, with the codebook i8 gather anchor SELECTED (m1/mf2 VLEN "
           "flip) by the shared codebook schedule authority from the "
           "deriveMinimumVLEN capability fact. BOUNDED Track B G2 step: the "
           "codebook integer CORE only (no block loop / fp16 scale / fp32 fold).";
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
          << "RVV codebook-gather dot source front door requires an injected "
             "extension-plugin registry to dispatch the conservative fallback";
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
      (void)fail(module, "source module must contain exactly one RVV codebook "
                         "integer-core source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<CodebookDotSourceMatch> source =
        matchCodebookDotSourceFunc(funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    // The capability-fact-driven codebook anchor SELECTION (the flip): m1 at
    // VLEN128, mf2 at VLEN256, fail-closed (I7) at VLEN0 where every codebook
    // candidate is pruned (the codebook gather needs VLMAX >= 16).
    std::optional<std::string> coreLMUL =
        selectCodebookCoreLMUL(march, isaVectorHints);
    if (!coreLMUL) {
      (void)fail(module, llvm::Twine("the capability profile (march='") + march +
                             "') prunes every legal codebook i8 gather anchor "
                             "(VLMAX < 16 for the 16-entry table); no schedule "
                             "is selectable (fail-closed)");
      signalPassFailure();
      return;
    }

    std::string kernelName = getKernelName(module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeKernel(builder, kernelName, *registry, *source,
                                       *coreLMUL))) {
      signalPassFailure();
      return;
    }

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  const ExtensionPluginRegistry *registry = nullptr;

  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("The capability-derivation -march the codebook i8 gather "
                     "anchor is selected from (e.g. rv64gcv, rv64gcv_zvl256b). "
                     "Empty => no guaranteed VLEN tier => fail-closed."),
      llvm::cl::init("")};
  Pass::Option<std::string> isaVectorHints{
      *this, "isa-vector-hints",
      llvm::cl::desc("Optional probed ISA/vector hint string folded into the "
                     "capability VLEN derivation alongside -march."),
      llvm::cl::init("")};
};

} // namespace

std::unique_ptr<::mlir::Pass>
createMaterializeRVVCodebookDotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVCodebookDotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVCodebookDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-codebook-gather-dot-source-front-door",
      "Auto-construct the tcrv_rvv codebook (vrgather) integer-CORE body "
      "(codebook_table_broadcast + load x3 + codebook_gather_x_i8_product + "
      "standalone_reduce + store) from a marked generic codebook-core source "
      "(BOUNDED Track B G2: the codebook integer core only, the i8 gather anchor "
      "SELECTED with the m1/mf2 VLEN flip)",
      [registryPtr] {
        return createMaterializeRVVCodebookDotSourceFrontDoorPass(*registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
