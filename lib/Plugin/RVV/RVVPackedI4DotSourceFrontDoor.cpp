//===- RVVPackedI4DotSourceFrontDoor.cpp ---------------------------------===//
//
// Track B G1, the BOUNDED first step: the COMPILER auto-CONSTRUCTS the q4_0
// nibble INTEGER-CORE body from a marked GENERIC source carrying the nibble-core
// operator identity, instead of routing a monolithic op to a per-kernel hand
// emitter. The auto-constructed body is the single-strip generic-op composition
//   load x3 (packed-i4 weight + the two plain-i8 q8 activation halves)
//     -> tcrv_rvv.packed_i4_offset_binary_x_i8_product (the offset-binary nibble
//        decode + asymmetric widening product -- ALREADY a first-class generic op,
//        lowered by the SAME emitOffsetBinaryDecodeProductValue the hand-written
//        block-dot strip calls)
//     -> tcrv_rvv.standalone_reduce (signed widening reduce, i16 -> i32)
//     -> tcrv_rvv.store
// the unchanged --tcrv-rvv-lower-to-emitc emitter consumes verbatim. This proves
// auto-construction REACHES nibble-decode through the generic-op mechanism: the
// nibble unpack itself needs NO new emitter vocabulary; it is already a typed op.
//
// HONEST SCOPE -- the nibble integer CORE only (NOT the full q4_0 KERNEL). There is
// NO nb = n / QK outer block loop, NO per-block dual fp16 scale read, and NO left-
// associative fp32 fold; those three axes need NEW generic ODS vocabulary and are
// full G1, DEFERRED. The monolithic tcrv_rvv.q4_0_q8_0_block_dot op, its KERNEL
// front door (RVVQ40BlockDotSourceFrontDoor), and the hand emitter
// (emitQ4_0Q8_0BlockDot) all STAY -- this adds a SEPARATE rung-3 front door, like
// dequant-vs-reduction.
//
// CAPABILITY framing -- NO FLIP claimed. The capability consultation is the SAME
// shared block-dot schedule authority the rung-1/2 front doors use
// (selectIntegerCoreLMUL: enumerateBlockDotShapeCandidates + selectGenericSchedule
// fed deriveMinimumVLEN(march)), run here as the LEGALITY GATE (fail-closed via I7
// if the integer-core path is pruned), NOT as the nibble anchor source. q4_0's
// nibble HALF-block integer core is pinned at i8mf4-i16mf2-i32m1 at every Zvl128b
// tier -- there is NO VLEN128-vs-VLEN256 byte-flip here. This is the documented
// q4_0 no-flip property (RVVQ40BlockDotSourceFrontDoor.cpp:22-33) and the gearbox's
// own packed-i4 candidate (`signed-i4n2-in-i8mf4-i16mf2-i32m1-f32m1`, the single
// anchor rung listed for packed-i4). The mf4 isolated-core strip is the NARROWER
// (4-element strip) sibling of the full-block plain-int8 anchor the gate navigates;
// the bounded step pins it byte-identical to the existing nibble-core lit and
// DEFERS both the gearbox-selected full-kernel SHAPE (m1/factor=4/elided) and the
// full kernel to G1.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVPackedI4DotSourceFrontDoor.h"

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::plugin::rvv {
namespace {

namespace tcrvexec = ::tianchenrv::tcrv::exec;
namespace tcrvrvv = ::tianchenrv::tcrv::rvv;

// The DISTINCT marker the source module carries to route to THIS packed-i4 nibble
// integer-core front door (NOT the MVP's "bounded_widening_dot_reduce_source", the
// dequant rung's "..._dequantize_source", nor the q4_0 KERNEL front door's
// "ggml_q4_0_q8_0_block_dot_source"). Each front-door pass checks its own marker
// and early-returns on a mismatch, so the four passes are mutually exclusive and
// every existing front-door lit is byte-unchanged.
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
constexpr llvm::StringLiteral kAcceptedMarkerValue(
    "bounded_packed_i4_offset_binary_dot_source");
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
    "rvv-packed-i4-offset-binary-dot-source-front-door-case");

// The plain-int8 K=32 block the LEGALITY GATE reasons about (the same fixed block
// the dequant/reduction rungs feed the shared schedule authority). The gate only
// asks "is a legal integer-core widening schedule selectable at this VLEN tier?";
// the nibble anchor itself is pinned (see below), not read from the gate.
constexpr std::int64_t kContractionBlockLen = 32;

// The PINNED nibble integer-core anchor: the i8 packed-i4 weight + plain-i8
// activations load at LMUL mf4, the asymmetric widening product is i16 mf2, the
// signed widening reduce is i32 m1, and the strip vsetvl the emitter issues for
// this anchor is e32m1 (SEW=32, LMUL=m1 -- the mf4 strip's documented spelling,
// getRVVBlockDotStripSEW("mf4")==32 / getRVVBlockDotStripLMUL("mf4")=="m1"). This
// is the q4_0 no-flip isolated-core form, byte-identical to the existing
// rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir lit.
constexpr llvm::StringLiteral kNibbleCoreLoadLMUL("mf4");
constexpr std::int64_t kNibbleCoreStripSEW = 32;
constexpr llvm::StringLiteral kNibbleCoreStripLMUL("m1");
constexpr llvm::StringLiteral kPackedI4ProductKind(
    "signed_packed_i4_offset_binary_x_i8_product");
constexpr llvm::StringLiteral kPackedI4ProductRelation(
    "offset-binary-i4mf4-x-i8mf4x2-to-i16mf2");

mlir::LogicalResult fail(mlir::Operation *op, llvm::Twine message) {
  op->emitError() << "bounded RVV packed-i4 offset-binary dot source front door "
                     "failed: "
                  << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the nibble INTEGER-CORE operator
//     identity. Like the q4_0 KERNEL matcher, recognition is by the integer-core
//     ABI roles, NOT a straight-line generic vector dataflow -- the offset-binary
//     nibble decode has no compact generic vector form (it is exactly the
//     first-class packed_i4 op the body builder constructs).
//===----------------------------------------------------------------------===//

struct PackedI4DotSourceMatch {
  mlir::func::FuncOp func;
};

bool isRank1MemRef(mlir::Type type, unsigned bitwidth) {
  auto memref = llvm::dyn_cast<mlir::MemRefType>(type);
  return memref && memref.getRank() == 1 &&
         memref.getElementType().isInteger(bitwidth);
}

// Match the nibble integer-core OPERATOR-IDENTITY source signature:
//   func(%weight: memref<?xi8>, %qlo: memref<?xi8>, %qhi: memref<?xi8>,
//        %acc: memref<?xi32>, %out: memref<?xi32>, %n: index)
// holding ONLY the packed-i4 nibble-core intent marker. The six roles are the
// packed-i4 weight (each byte two offset-binary nibbles), the two plain-i8 q8
// activation halves paired with the low/high nibbles, the i32 accumulator seed,
// the i32 output, and the runtime element count. The body is the bounded intent
// shell (it may be a bare `return`): the offset-binary decode + asymmetric product
// are q4_0 STRUCTURE the constructed packed_i4 op carries, so this front door does
// NOT pretend to recognize the nibble unpack from generic vector ops.
mlir::FailureOr<PackedI4DotSourceMatch>
matchPackedI4DotSourceFunc(mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(func, "source function must have a body for the packed-i4 "
                      "nibble integer-core intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 6 || type.getNumResults() != 0)
    return fail(func,
                "source function must have exactly six inputs and no results: "
                "weight/qlo/qhi memref<?xi8>, acc memref<?xi32>, out "
                "memref<?xi32>, and n index (the packed-i4 nibble integer-core "
                "operator identity)");
  if (!isRank1MemRef(type.getInput(0), 8) ||
      !isRank1MemRef(type.getInput(1), 8) ||
      !isRank1MemRef(type.getInput(2), 8) ||
      !isRank1MemRef(type.getInput(3), 32) ||
      !isRank1MemRef(type.getInput(4), 32) || !type.getInput(5).isIndex())
    return fail(func,
                "source function inputs must be packed-i4 weight rank-1 i8 "
                "memref, two plain-i8 q8 activation-half rank-1 i8 memrefs, an "
                "i32 acc seed rank-1 memref, an i32 out rank-1 memref, and an n "
                "index");

  return PackedI4DotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) The capability-fact-driven integer-core LEGALITY GATE.
//===----------------------------------------------------------------------===//

// Run the SHARED block-dot schedule authority to confirm a legal integer-core
// widening schedule is selectable at this VLEN tier -- byte-identical to the
// dequant/reduction rungs' selectIntegerCoreLMUL (the {m1,m2} plain-int8 K=32
// descriptor, factorCap=1). The RETURN VALUE is the full-block plain-int8 anchor;
// the nibble HALF-block uses the NARROWER mf4 rung of the SAME i8->i16->i32
// widening ladder. This call is the capability LEGALITY GATE only (fail-closed via
// nullopt -> I7 when every candidate is pruned), NOT the nibble anchor source --
// the q4_0 nibble core is the no-flip mf4 form, pinned below.
std::optional<std::string>
selectIntegerCoreLMUL(llvm::StringRef march, llvm::StringRef isaVectorHints) {
  std::int64_t minimumVLEN = deriveMinimumVLEN(march, isaVectorHints);

  static constexpr llvm::StringLiteral kCoreLMULs[] = {"m1", "m2"};
  RVVBlockDotKernelDescriptor descriptor{
      /*coreLMULs=*/kCoreLMULs,
      /*quantFormat=*/"plain-int8",
      /*blockLen=*/kContractionBlockLen,
      /*stripSEW=*/getRVVBlockDotStripSEW,
      /*vectorRegisterCost=*/getRVVQ80ShapeVectorRegisterCost};
  // factorCap=1: the gate reasons about the single-block robust integer core; the
  // multi_block unroll axis is a separate Win-A brick, not folded in here.
  descriptor.factorCap = 1;

  llvm::SmallVector<RVVBlockDotShapeCandidate, 18> typed =
      enumerateBlockDotShapeCandidates(descriptor, minimumVLEN,
                                       kRVVQ80ShapeVectorRegisterBudget);
  llvm::SmallVector<GenericScheduleCandidate> candidates;
  for (const RVVBlockDotShapeCandidate &candidate : typed)
    candidates.push_back(toGenericBlockDotCandidate(candidate));

  static constexpr llvm::StringRef kRequiredKnobKeys[] = {"lmul"};
  std::optional<GenericScheduleSelection> selected = selectGenericSchedule(
      candidates, /*recordText=*/std::nullopt,
      /*kernelKey=*/"packed_i4_offset_binary_dot_i8", march, kRequiredKnobKeys);
  if (!selected)
    return std::nullopt; // every candidate pruned -> fail-closed (I7).

  for (const NamedKnob &knob : selected->candidate.knobs)
    if (knob.recordKey == "lmul")
      return knob.value;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// (3) Body builder: auto-construct the tcrv_rvv RVV-dialect nibble integer-core
//     body (the single-strip packed_i4 product-reduce composition).
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

// The nibble integer core's auto-constructed step: the offset-binary packed-i4
// weight x plain-i8 activation-halves asymmetric widening product, the FIRST-CLASS
// generic op that already carries the nibble decode + one-sided unpack + asymmetric
// product as typed STRUCTURE (lowered by the SAME emitOffsetBinaryDecodeProductValue
// the hand-written block-dot strip calls). The compiler reaches nibble-decode by
// CONSTRUCTING this op -- no new emitter vocabulary, no hand-rolled vxor/vsll/vsra.
mlir::Value createPackedI4OffsetBinaryProduct(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value vl,
    mlir::Type productType) {
  mlir::OperationState state(
      loc, tcrvrvv::PackedI4OffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute("kind", builder.getStringAttr(kPackedI4ProductKind));
  state.addAttribute("product_relation",
                     builder.getStringAttr(kPackedI4ProductRelation));
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
                  PackedI4DotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = "rvv_packed_i4_offset_binary_dot_i8";
  std::string fallbackVariantSymbol =
      "rvv_packed_i4_offset_binary_dot_i8_scalar_fallback";

  // The PINNED nibble integer-core widening ladder: the i8 packed-i4 weight +
  // plain-i8 activation halves load at mf4, the asymmetric widening product is the
  // next-wider mf2, and the signed widening reduce is i32 m1. The strip vsetvl the
  // emitter issues for the mf4 anchor is e32m1 (the documented mf4 spelling). This
  // is the q4_0 no-flip isolated-core form -- NOT a VLEN byte-flip.
  llvm::StringRef loadLMUL = kNibbleCoreLoadLMUL;
  llvm::StringRef productLMUL = getRVVNextWiderLMUL(loadLMUL);
  if (productLMUL.empty())
    return fail(source.func,
                llvm::Twine("no wider LMUL rung for nibble integer-core load "
                            "anchor '") +
                    loadLMUL + "'");
  llvm::StringRef reduceLMUL = getRVVNextWiderLMUL(productLMUL);
  if (reduceLMUL.empty())
    return fail(source.func,
                llvm::Twine("no wider LMUL rung for nibble integer-core product "
                            "anchor '") +
                    productLMUL + "'");

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
  // AUDIT-ONLY provenance (NOT the route/dtype authority): the no-flip nibble
  // integer-core anchor the body below realizes structurally. Recorded as the
  // PINNED form (never the gate's plain-int8 m1/m2 return, which would mislead).
  rvvVariant->setAttr(
      "tcrv_rvv.packed_i4_integer_core_anchor",
      builder.getStringAttr("i8mf4-i16mf2-i32m1-no-vlen-flip"));
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrvrvv::RuntimeABIValueType::get(builder.getContext());
  // The six nibble integer-core ABI roles, in the order the emitted C signature
  // pins (so the output buffer is arg4, matching the nibble-core lit): the
  // packed-i4 weight, the two plain-i8 q8 activation halves, the i32 acc seed, the
  // i32 out, and the runtime element count.
  auto weight = createRuntimeABIValue(builder, loc, "lhs-input-buffer", "w",
                                      "const int8_t *", "q4-weight",
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

  tcrvrvv::SetVLOp setvl = createSetVL(builder, loc, n.getResult(),
                                       kNibbleCoreStripSEW, kNibbleCoreStripLMUL,
                                       policy);
  tcrvrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), kNibbleCoreStripSEW,
                   kNibbleCoreStripLMUL, policy, kernelName,
                   selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  mlir::Type i8VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getI8Type(), loadLMUL);
  mlir::Type i16VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getI16Type(), productLMUL);
  mlir::Type i32VecType = tcrvrvv::VectorType::get(
      builder.getContext(), builder.getI32Type(), reduceLMUL);

  // The three i8/mf4 source loads: the packed-i4 weight + the two plain-i8 q8
  // activation halves (NOT nibble-decoded -- only the weight is).
  mlir::Value loadedWeight =
      createRVVLoad(builder, loc, weight.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedQLow =
      createRVVLoad(builder, loc, qlo.getResult(), setvl.getVl(), i8VecType);
  mlir::Value loadedQHigh =
      createRVVLoad(builder, loc, qhi.getResult(), setvl.getVl(), i8VecType);
  // The auto-constructed nibble decode + asymmetric widening product (the
  // first-class generic op): offset-binary unpack of the weight, asymmetric
  // i8(decoded) x i8(plain) widening product -> i16/mf2.
  mlir::Value product = createPackedI4OffsetBinaryProduct(
      builder, loc, loadedWeight, loadedQLow, loadedQHigh, setvl.getVl(),
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
  return "rvv_packed_i4_offset_binary_dot_i8_from_source";
}

class MaterializeRVVPackedI4DotSourceFrontDoorPass final
    : public mlir::PassWrapper<MaterializeRVVPackedI4DotSourceFrontDoorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVPackedI4DotSourceFrontDoorPass() = default;
  MaterializeRVVPackedI4DotSourceFrontDoorPass(
      const MaterializeRVVPackedI4DotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVPackedI4DotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        registry(other.registry) {}
  explicit MaterializeRVVPackedI4DotSourceFrontDoorPass(
      const ExtensionPluginRegistry *registry)
      : registry(registry) {}

  llvm::StringRef getArgument() const final {
    return "tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door";
  }
  llvm::StringRef getDescription() const final {
    return "Auto-construct the tcrv_rvv q4_0 nibble integer-CORE body (load x3 + "
           "packed_i4_offset_binary_x_i8_product + standalone_reduce + store) "
           "from a marked generic nibble-core source, with the integer-core path "
           "legality gated by the shared block-dot schedule authority from the "
           "deriveMinimumVLEN capability fact. BOUNDED Track B G1 step: the "
           "nibble integer CORE only (no block loop / fp16 scale / fp32 fold).";
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
          << "RVV packed-i4 offset-binary dot source front door requires an "
             "injected extension-plugin registry to dispatch the conservative "
             "fallback";
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
      (void)fail(module, "source module must contain exactly one RVV packed-i4 "
                         "nibble integer-core source function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<PackedI4DotSourceMatch> source =
        matchPackedI4DotSourceFunc(funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    // The capability LEGALITY GATE: confirm a legal integer-core widening
    // schedule is selectable at this VLEN tier (fail-closed via I7 otherwise).
    // The selected plain-int8 anchor is NOT the nibble anchor (pinned mf4); this
    // call is the legality gate only.
    std::optional<std::string> integerCoreLMUL =
        selectIntegerCoreLMUL(march, isaVectorHints);
    if (!integerCoreLMUL) {
      (void)fail(module, llvm::Twine("the capability profile (march='") + march +
                             "') prunes every legal integer-core anchor; no "
                             "schedule is selectable (fail-closed)");
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

  Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc("The capability-derivation -march the integer-core path "
                     "legality is gated from (e.g. rv64gcv, rv64gcv_zvl256b). "
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
createMaterializeRVVPackedI4DotSourceFrontDoorPass(
    const ExtensionPluginRegistry &registry) {
  return std::make_unique<MaterializeRVVPackedI4DotSourceFrontDoorPass>(
      &registry);
}

llvm::Error registerRVVPackedI4DotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  out.push_back(SourceFrontDoorPassRegistration(
      ownerPlugin,
      "tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door",
      "Auto-construct the tcrv_rvv q4_0 nibble integer-CORE body (load x3 + "
      "packed_i4_offset_binary_x_i8_product + standalone_reduce + store) from a "
      "marked generic nibble-core source (BOUNDED Track B G1: the nibble integer "
      "core only, capability-gated)",
      [registryPtr] {
        return createMaterializeRVVPackedI4DotSourceFrontDoorPass(*registryPtr);
      },
      SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
          ExplicitOnly));
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
