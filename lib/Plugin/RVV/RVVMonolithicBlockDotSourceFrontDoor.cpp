//===- RVVMonolithicBlockDotSourceFrontDoor.cpp -------------------------===//
//
// The ONE table-driven monolithic ggml block-dot source front door -- the collapse
// of the 24 former per-op scaffold-constructor passes into a single generic pass
// driven by `monolithicBlockDotOpTable()`. Each table row carries the per-op
// construction DATA; this file carries the ONE shared construction MECHANISM.
//
// For a marked GENERIC source carrying a ggml `ggml_vec_dot_<op>` OPERATOR IDENTITY
// (the vec_dot ABI roles), the pass auto-constructs the complete tcrv.exec.kernel +
// variant + dispatch/fallback scaffold around ONE attr-less (modulo the row's
// integer_core_lmul) tcrv_rvv.<op>_block_dot op, instead of a per-kernel
// hand-authored block-dot emitter input. The per-block scale model, the integer
// decode/product/reduce core, the super-block bit-dance, the codebook gather, and
// the deferred fold are FIRST-CLASS STRUCTURE inside that op and its existing
// emitter; the front door supplies only the block-format CONSTANTS + codebook/grid
// DATA a generic source cannot derive (the values the op verifier pins). Shape
// selection (where a gearbox exists) is DEFERRED to the unmodified schedule
// autotuner -- the constructed op is byte-identical to the hand-authored emitter
// input, so any capability flip rides the existing gearbox byte-for-byte.
//
// One pass class parameterized by a table-row pointer registers ONE CLI front-door
// argument per row; each instance early-returns unless the module marker matches its
// row, so the family of front doors stays mutually exclusive and every sibling lit
// is byte-unchanged.
//
//===----------------------------------------------------------------------===//

#include "TianChenRV/Plugin/RVV/RVVMonolithicBlockDotSourceFrontDoor.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVCapabilityProfile.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVMonolithicBlockDotFamily.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Support/RuntimeABI.h"
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

// The marker/kernel module attributes shared by the whole family (identical across
// every former per-op front door; only the marker VALUE is per-op DATA in the row).
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "tcrv_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("tcrv_rvv.source_kernel");
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

// The single generic pass description (shown in --help for every family front door;
// not a byte-exact-gated string, so it is shared rather than per-op prose).
constexpr llvm::StringLiteral kPassDescription(
    "Auto-construct the attr-less tcrv_rvv.<op>_block_dot op + "
    "kernel/variant/dispatch/fallback scaffold from a marked ggml vec_dot "
    "operator-identity source (the block loop, scale model, integer core, "
    "super-block bit-dance, codebook gather, and deferred fold are first-class op "
    "structure); shape selection is left to the existing capability-driven schedule "
    "autotuner (one table-driven front door across the whole ggml block-dot family)");

mlir::LogicalResult fail(const MonolithicBlockDotOpEntry &entry,
                         mlir::Operation *op, llvm::Twine message) {
  op->emitError() << entry.failPrefix << message;
  return mlir::failure();
}

//===----------------------------------------------------------------------===//
// (1) Matcher: the marked generic source carrying the ggml vec_dot OPERATOR
//     IDENTITY (signature recognition -- the block-dot has no compact generic
//     vector form, so recognition is by the vec_dot ABI roles, not by a
//     straight-line dataflow pattern).
//===----------------------------------------------------------------------===//

struct BlockDotSourceMatch {
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

// Match the ggml `ggml_vec_dot_<op>` OPERATOR-IDENTITY source signature:
//   func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)
// holding ONLY the block-dot intent marker -- the WHAT is the operator identity
// (weight x activation block dot-product -> fp32 out), NOT a generic dataflow body.
// The block/super-block loop, scale model, integer core, codebook gather, and fold
// are op STRUCTURE the constructed op carries; the source func body is the bounded
// intent shell (it may be a bare `return`).
mlir::FailureOr<BlockDotSourceMatch>
matchBlockDotSourceFunc(const MonolithicBlockDotOpEntry &entry,
                        mlir::func::FuncOp func) {
  if (func.isDeclaration())
    return fail(entry, func,
                "source function must have a body for the block-dot intent shell");

  mlir::FunctionType type = func.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0)
    return fail(entry, func,
                "source function must have exactly four inputs and no results: "
                "out memref<?xf32>, n index, weight memref<?xi8>, activation "
                "memref<?xi8> (the ggml vec_dot operator identity)");
  if (!isRank1F32MemRef(type.getInput(0)) || !type.getInput(1).isIndex() ||
      !isRank1MemRef(type.getInput(2), 8) || !isRank1MemRef(type.getInput(3), 8))
    return fail(entry, func,
                "source function inputs must be out rank-1 f32 memref, n index, "
                "weight rank-1 i8 memref, activation rank-1 i8 memref");

  return BlockDotSourceMatch{func};
}

//===----------------------------------------------------------------------===//
// (2) Body builder: auto-construct the tcrv_rvv.<op>_block_dot op scaffold
//     (kernel + variant + dispatch/fallback) generically from the table row.
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

// The C type + result type + purpose an ABI role projects to, matching the former
// per-op front doors exactly (the vec_dot prototype is family-invariant per role).
llvm::StringRef abiRoleCType(support::RuntimeABIParameterRole role) {
  switch (role) {
  case support::RuntimeABIParameterRole::OutputBuffer:
    return "float *";
  case support::RuntimeABIParameterRole::LHSInputBuffer:
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return "const uint8_t *";
  case support::RuntimeABIParameterRole::RHSScalarValue:
    return "int32_t";
  default:
    return "size_t";
  }
}

mlir::Type abiRoleResultType(support::RuntimeABIParameterRole role,
                             mlir::Type runtimeABIType, mlir::Type indexType,
                             mlir::Type i32Type) {
  switch (role) {
  case support::RuntimeABIParameterRole::OutputBuffer:
  case support::RuntimeABIParameterRole::LHSInputBuffer:
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return runtimeABIType;
  case support::RuntimeABIParameterRole::RHSScalarValue:
    return i32Type;
  default:
    return indexType;
  }
}

llvm::StringRef abiRolePurpose(const MonolithicBlockDotOpEntry &entry,
                               support::RuntimeABIParameterRole role,
                               llvm::StringRef cName) {
  switch (role) {
  case support::RuntimeABIParameterRole::OutputBuffer:
    return "out";
  case support::RuntimeABIParameterRole::LHSInputBuffer:
    return entry.weightPurpose;
  case support::RuntimeABIParameterRole::RHSInputBuffer:
    return entry.activationPurpose;
  default:
    return cName;
  }
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

// ---------------------------------------------------------------------------
// Typed flat block-dot loop-body construction (M-FLAT step 5b, q8_0 only).
//
// The gated typed path builds the COMPLETE per-block typed chain
// (tcrv_rvv.typed_flat_block_dot_loop_body region: brick 1 dual-fp16 scale
// product -> two per-block i8 loads -> signed widening product -> standalone
// reduce -> lane0 scalar extract -> brick 2 computed-scale dequant -> brick 3
// cross-block f32 accumulate -> yield) instead of the ONE monolith block-dot op.
// The integer-core helpers are copied near-verbatim from the sibling
// RVVReductionSourceFrontDoor.cpp; the six scalar-domain typed ops have no C++
// builder and are built directly from OperationState + the ODS shape.
// ---------------------------------------------------------------------------

// Loop-capable per-block i8 load: base + block_index*block_stride
// (+ quant_byte_offset). Adapts the sibling single-block createRVVLoad with the
// trailing block_index operand + block_stride/quant_byte_offset facts.
mlir::Value createRVVBlockLoad(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value buffer, mlir::Value vl,
                               mlir::Value blockIndex, std::int64_t blockStride,
                               std::int64_t quantByteOffset,
                               mlir::Type vectorType) {
  mlir::OperationState state(loc, tcrvrvv::LoadOp::getOperationName());
  state.addOperands({buffer, vl, blockIndex});
  state.addAttribute("block_stride", builder.getI64IntegerAttr(blockStride));
  state.addAttribute("quant_byte_offset",
                     builder.getI64IntegerAttr(quantByteOffset));
  state.addTypes(vectorType);
  return builder.create(state)->getResult(0);
}

mlir::Value createWideningProduct(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value lhs, mlir::Value rhs,
                                  mlir::Value vl, mlir::Type productType,
                                  llvm::StringRef productRelation) {
  mlir::OperationState state(loc,
                             tcrvrvv::WideningProductOp::getOperationName());
  state.addOperands({lhs, rhs, vl});
  state.addAttribute("kind", builder.getStringAttr("signed_widening_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// The q4_0 asymmetric offset-binary packed-i4 x plain-i8 integer core: ONE
// packed-i4 weight operand (each i8 packs two offset-binary nibbles) + TWO plain
// int8 activation operands (the q8 low half paired with the low nibbles, the q8
// high half with the high nibbles) -> ONE widened i16 product. The m1 flat-cohort
// rung (i4m1 weight x i8m1 low/high activation -> i16m2) mirrors the mf4 anchor
// rung; the verifier admits both.
mlir::Value createPackedI4OffsetBinaryProduct(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value vl,
    mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, tcrvrvv::PackedI4OffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("signed_packed_i4_offset_binary_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
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

// brick 1: per-block d_x * d_y fp16 scale product (block_index-sourced form).
mlir::Value createBlockFp16ScaleProduct(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value lhsScaleBase,
                                        mlir::Value rhsScaleBase,
                                        mlir::Value blockIndex,
                                        std::int64_t lhsBlockStride,
                                        std::int64_t rhsBlockStride) {
  mlir::OperationState state(
      loc, tcrvrvv::BlockFp16ScaleProductOp::getOperationName());
  state.addOperands({lhsScaleBase, rhsScaleBase, blockIndex});
  state.addAttribute("kind",
                     builder.getStringAttr("dual_fp16_per_block_scale_product"));
  state.addAttribute("scale_model",
                     builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
  state.addAttribute("lhs_block_stride",
                     builder.getI64IntegerAttr(lhsBlockStride));
  state.addAttribute("rhs_block_stride",
                     builder.getI64IntegerAttr(rhsBlockStride));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

// integer-core -> scalar bridge: i32 m1 vector lane0 -> scalar i32 sumi.
mlir::Value createTypedVectorLane0ToScalarExtract(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value input,
                                                  mlir::Value vl) {
  mlir::OperationState state(
      loc, tcrvrvv::TypedVectorLane0ToScalarExtractOp::getOperationName());
  state.addOperands({input, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("vector_lane0_to_scalar_i32_extract"));
  state.addAttribute("extract_relation",
                     builder.getStringAttr("i32m1-lane0-to-scalar-i32"));
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

// brick 2: (float)sumi * computed scale. computed_scale is brick 1's f32 (NOT an
// imported ABI value -- the verifier structurally requires f32 here).
mlir::Value createBlockComputedScaleDequant(mlir::OpBuilder &builder,
                                            mlir::Location loc, mlir::Value sumi,
                                            mlir::Value computedScale) {
  mlir::OperationState state(
      loc, tcrvrvv::BlockComputedScaleDequantOp::getOperationName());
  state.addOperands({sumi, computedScale});
  state.addAttribute("kind", builder.getStringAttr("computed_scale_sumi_dequant"));
  state.addAttribute(
      "dequant_relation",
      builder.getStringAttr("scalar-i32-sumi-to-f32-computed-scale-f32"));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

// brick 3: sumf + term (cross-block fp32 fold, strict ascending block order).
mlir::Value createCrossBlockF32Accumulate(mlir::OpBuilder &builder,
                                          mlir::Location loc, mlir::Value acc,
                                          mlir::Value term) {
  mlir::OperationState state(
      loc, tcrvrvv::CrossBlockF32AccumulateOp::getOperationName());
  state.addOperands({acc, term});
  state.addAttribute("kind",
                     builder.getStringAttr("cross_block_f32_scalar_accumulate"));
  state.addAttribute("accumulate_order",
                     builder.getStringAttr("strict-ascending-block-carried"));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

void createTypedFlatBlockDotLoopYield(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value accNext) {
  mlir::OperationState state(
      loc, tcrvrvv::TypedFlatBlockDotLoopYieldOp::getOperationName());
  state.addOperands(accNext);
  (void)builder.create(state);
}

// The gated typed q8_0 path: build the whole per-block typed chain inside a new
// tcrv_rvv.typed_flat_block_dot_loop_body region (result-less, region-carrying),
// replacing the ONE monolith block-dot op. The loop-body op's block strides / qk /
// quant offset come from entry.facts (by-name lookup); multi_block_factor is OMITTED
// (absent = mbf 1). weight/activation/out/n are the shared variant-scope ABI values;
// zeroSeed is the variant-scope reduce seed (dominates the in-region reduce); vl is
// the setvl VL referenced freely by the in-region loads/product/reduce/extract.
void createTypedFlatBlockDotLoopChain(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      const MonolithicBlockDotOpEntry &entry,
                                      mlir::Value weight, mlir::Value activation,
                                      mlir::Value out, mlir::Value n,
                                      mlir::Value vl, mlir::Value zeroSeed) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed flat block-dot chain: missing block-format fact");
  };
  std::int64_t qk = factByName("qk");
  std::int64_t weightStride = factByName("weight_block_stride");
  std::int64_t activationStride = factByName("activation_block_stride");
  std::int64_t quantByteOffset = factByName("quant_byte_offset");

  // Format branch: q8_0 (plain signed i8xi8 whole-block, m2 anchor, sumi-first
  // fold) vs q4_0 (asymmetric offset-binary packed-i4 x i8 HALF-block, m1 anchor,
  // left-assoc fold + the q8 high-half activation strip). Only these two flat ops
  // take the typed loop path; everything else stays the monolith op. q8_0's chain
  // is byte-unchanged from before the branch.
  const bool isQ40 =
      entry.opName == tcrvrvv::GgmlBlockDotQ40Q80Op::getOperationName();
  std::int64_t activationHighOffset =
      isQ40 ? factByName("activation_high_byte_offset") : 0;

  mlir::OperationState loopState(
      loc, tcrvrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute("kind",
                         builder.getStringAttr("typed_flat_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  loopState.addAttribute(
      "fold_model",
      builder.getStringAttr(isQ40 ? "left_assoc" : "sumi_times_scales"));
  loopState.addAttribute("integer_core_lmul",
                         builder.getStringAttr(isQ40 ? "m1" : "m2"));
  loopState.addAttribute("strip_elision", builder.getStringAttr("elided"));
  // mbf==1 pin: do NOT stamp multi_block_factor (absent = factor 1).
  loopState.addRegion();
  auto loop = llvm::cast<tcrvrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  mlir::MLIRContext *ctx = builder.getContext();
  // The integer-core widths: q8_0 anchors i8m2 -> i16m4; q4_0's half-block
  // packed-i4 core anchors i8m1 -> i16m2. The i32m1 reduce lane is shared.
  llvm::StringRef coreLmul = isQ40 ? "m1" : "m2";
  llvm::StringRef wideLmul = isQ40 ? "m2" : "m4";
  mlir::Type i8VecType =
      tcrvrvv::VectorType::get(ctx, builder.getI8Type(), coreLmul);
  mlir::Type i16VecType =
      tcrvrvv::VectorType::get(ctx, builder.getI16Type(), wideLmul);
  mlir::Type i32VecType =
      tcrvrvv::VectorType::get(ctx, builder.getI32Type(), "m1");

  // brick 1: the per-block d_x * d_y fp16 scale product over block_index (shared;
  // the AoS strides differ per format but the scale model is identical).
  mlir::Value dd = createBlockFp16ScaleProduct(
      builder, loc, weight, activation, blockIndex, weightStride,
      activationStride);

  // The vector integer core diverges by format. q8_0: two per-block i8 loads ->
  // signed widening product. q4_0: ONE packed-i4 weight load + TWO plain-i8
  // activation loads (the q8 low half at quant_off, the q8 high half at
  // quant_off + activation_high_byte_offset) -> the asymmetric offset-binary
  // packed-i4 x i8 product.
  mlir::Value prod;
  if (isQ40) {
    // packed-i4 weight strip (base + ib*18 + 2).
    mlir::Value wv =
        createRVVBlockLoad(builder, loc, weight, vl, blockIndex, weightStride,
                           quantByteOffset, i8VecType);
    // q8 low half (base + ib*34 + 2) and high half (base + ib*34 + 2 + 16).
    mlir::Value avLow =
        createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                           activationStride, quantByteOffset, i8VecType);
    mlir::Value avHigh = createRVVBlockLoad(
        builder, loc, activation, vl, blockIndex, activationStride,
        quantByteOffset + activationHighOffset, i8VecType);
    // asymmetric offset-binary packed-i4 x i8 product (i4m1 x i8m1x2 -> i16m2).
    prod = createPackedI4OffsetBinaryProduct(
        builder, loc, wv, avLow, avHigh, vl, i16VecType,
        "offset-binary-i4m1-x-i8m1x2-to-i16m2");
  } else {
    // q8_0: two per-block i8 loads (base + ib*stride + quant_off).
    mlir::Value wv =
        createRVVBlockLoad(builder, loc, weight, vl, blockIndex, weightStride,
                           quantByteOffset, i8VecType);
    mlir::Value av =
        createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                           activationStride, quantByteOffset, i8VecType);
    // signed widening product (i8m2 x i8m2 -> i16m4).
    prod = createWideningProduct(builder, loc, wv, av, vl, i16VecType,
                                 "signed-i8m2xi8m2-to-i16m4");
  }
  // reduce i16<wide> -> i32m1 lane0, then extract lane0 -> scalar i32 sumi.
  mlir::Value red =
      createStandaloneReduce(builder, loc, prod, zeroSeed, vl, i32VecType);
  mlir::Value sumi =
      createTypedVectorLane0ToScalarExtract(builder, loc, red, vl);
  // brick 2: (float)sumi * (d_x * d_y).
  mlir::Value bterm = createBlockComputedScaleDequant(builder, loc, sumi, dd);
  // brick 3: sumf + term (cross-block fp32 fold).
  mlir::Value accNext = createCrossBlockF32Accumulate(builder, loc, acc, bterm);
  createTypedFlatBlockDotLoopYield(builder, loc, accNext);
}

// The ggml block dot-product op for this row: the bounded WHAT (kind, scale model,
// block-format i64 facts, and any codebook/grid/ksigns DATA) is stamped from the
// table row. Shape knobs are NOT stamped (the op lowers at the emitter default,
// leaving the schedule autotuner free) EXCEPT where the row pins integer_core_lmul
// (the one op -- nvfp4 -- whose sealed reference is its m1 anchor). The scale model,
// integer core, super-block bit-dance, codebook gather, and deferred fold are
// first-class STRUCTURE inside this op.
mlir::Value createBlockDot(mlir::OpBuilder &builder, mlir::Location loc,
                           const MonolithicBlockDotOpEntry &entry,
                           mlir::Value weight, mlir::Value activation,
                           mlir::Value out, mlir::Value n, mlir::Value vl) {
  mlir::OperationState state(loc, entry.opName);
  state.addOperands({weight, activation, out, n, vl});
  state.addAttribute("kind", builder.getStringAttr(entry.kind));
  state.addAttribute("scale_model", builder.getStringAttr(entry.scaleModel));
  for (const MonolithicBlockDotI64Attr &fact : entry.facts)
    state.addAttribute(fact.name, builder.getI64IntegerAttr(fact.value));
  if (!entry.codebook.empty())
    state.addAttribute("codebook", builder.getDenseI8ArrayAttr(entry.codebook));
  if (!entry.gridI64.empty())
    state.addAttribute("grid", builder.getDenseI64ArrayAttr(entry.gridI64));
  if (!entry.gridI32.empty())
    state.addAttribute("grid", builder.getDenseI32ArrayAttr(entry.gridI32));
  if (!entry.ksigns.empty())
    state.addAttribute("ksigns", builder.getDenseI32ArrayAttr(entry.ksigns));
  if (!entry.integerCoreLmul.empty())
    state.addAttribute("integer_core_lmul",
                       builder.getStringAttr(entry.integerCoreLmul));
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
    const MonolithicBlockDotOpEntry &entry, mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Operation *failOp,
    const ExtensionPluginRegistry &registry, llvm::StringRef capabilitySymbol) {
  llvm::SmallVector<PluginCapability, 1> fallbackCapabilities;
  registry.collectCapabilitiesByKind(kConservativeFallbackCapabilityKind,
                                     fallbackCapabilities);
  if (fallbackCapabilities.size() != 1)
    return fail(entry, failOp,
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
    const MonolithicBlockDotOpEntry &entry, mlir::OpBuilder &builder,
    tcrvexec::KernelOp kernel, mlir::Operation *highLevelOp,
    const ExtensionPluginRegistry &registry,
    llvm::StringRef fallbackVariantSymbol) {
  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(kernel);
  if (!capabilities) {
    (void)fail(entry, kernel,
               llvm::Twine("could not build a capability scope for kernel @") +
                   kernel.getSymName() + ": " +
                   llvm::toString(capabilities.takeError()));
    return mlir::failure();
  }

  VariantProposalRequest request(highLevelOp, kernel, *capabilities);
  llvm::SmallVector<VariantProposal, 4> proposals;
  if (llvm::Error error = registry.collectVariantProposals(request, proposals)) {
    (void)fail(entry, kernel,
               llvm::Twine("failed to collect variant proposals for kernel @") +
                   kernel.getSymName() + ": " +
                   llvm::toString(std::move(error)));
    return mlir::failure();
  }

  const VariantProposal *fallbackProposal = nullptr;
  for (const VariantProposal &proposal : proposals) {
    if (proposal.getFallbackRole() != VariantFallbackRole::ConservativeFallback)
      continue;
    if (fallbackProposal) {
      (void)fail(entry, kernel,
                 "requires exactly one conservative-fallback variant proposal; "
                 "the registry produced more than one");
      return mlir::failure();
    }
    fallbackProposal = &proposal;
  }
  if (!fallbackProposal) {
    (void)fail(entry, kernel,
               "requires a conservative-fallback variant proposal from a "
               "fallback-owning plugin; none was produced");
    return mlir::failure();
  }

  VariantProposal scopedProposal = *fallbackProposal;
  scopedProposal.setVariantName(fallbackVariantSymbol);
  if (llvm::Error error = transforms::materializeVariantProposals(
          builder, request, scopedProposal)) {
    (void)fail(entry, kernel,
               llvm::Twine("failed to materialize the conservative fallback "
                           "variant for kernel @") +
                   kernel.getSymName() + ": " + llvm::toString(std::move(error)));
    return mlir::failure();
  }
  return fallbackProposal->getOriginPlugin().str();
}

void createDispatch(mlir::OpBuilder &builder, mlir::Location loc,
                    llvm::StringRef dispatchPolicy,
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
  caseState.addAttribute("policy", builder.getStringAttr(dispatchPolicy));
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
                  const MonolithicBlockDotOpEntry &entry,
                  BlockDotSourceMatch source) {
  mlir::Location loc = source.func.getLoc();
  tcrvrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = entry.variantSymbol.str();
  std::string fallbackVariantSymbol =
      (entry.variantSymbol + "_scalar_fallback").str();

  mlir::OperationState kernelState(loc, tcrvexec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addRegion();
  auto kernel = llvm::cast<tcrvexec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  if (mlir::failed(createConservativeFallbackCapability(
          entry, builder, loc, source.func, registry, kFallbackCapabilitySymbol)))
    return mlir::failure();
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  tcrvexec::VariantOp rvvVariant =
      createVariant(builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      tcrvrvv::RuntimeABIValueType::get(builder.getContext());
  mlir::Type indexType = builder.getIndexType();
  mlir::Type i32Type = builder.getI32Type();

  // The ggml vec_dot ABI value set, in the row's declared role order (the same order
  // the board-validated emitter input declares). The block-dot op consumes only
  // weight/activation/out/n; any stride/scalar params (q4_0/q8_0's 8-role strided
  // prototype) are present so the exported C signature matches ggml's vec_dot
  // prototype -- they are dropped by the block-dot lowering.
  mlir::Value weight, activation, out, n;
  for (const MonolithicBlockDotABIRole &role : entry.abiRoles()) {
    mlir::Value value = createRuntimeABIValue(
        builder, loc, support::stringifyRuntimeABIParameterRole(role.role),
        role.cName, abiRoleCType(role.role),
        abiRolePurpose(entry, role.role, role.cName),
        abiRoleResultType(role.role, runtimeABIType, indexType, i32Type));
    switch (role.role) {
    case support::RuntimeABIParameterRole::RuntimeElementCount:
      n = value;
      break;
    case support::RuntimeABIParameterRole::OutputBuffer:
      out = value;
      break;
    case support::RuntimeABIParameterRole::LHSInputBuffer:
      weight = value;
      break;
    case support::RuntimeABIParameterRole::RHSInputBuffer:
      activation = value;
      break;
    default:
      break;
    }
  }

  // GATED typed flat-loop path (M-FLAT step 5b/6): q8_0 and q4_0's front doors
  // construct the COMPLETE per-block TYPED LOOP chain
  // (typed_flat_block_dot_loop_body region) instead of the ONE monolith block-dot
  // op. The typed core runs at the SEW8 byte anchor, so the setvl/with_vl config
  // is sew=8 (NOT the shared monolith sew=32/m1) with the per-format integer-core
  // LMUL: q8_0's plain whole-block core anchors m2, q4_0's half-block packed-i4
  // core anchors m1. Every other (monolith) row stays byte-unchanged.
  // Dispatch/coherence follow the constructed op format-agnostically.
  // multi_block_factor is pinned to 1 (absent on the loop-body op).
  const bool isQ80TypedFlat =
      entry.opName == tcrvrvv::GgmlBlockDotQ80Q80Op::getOperationName();
  const bool isQ40TypedFlat =
      entry.opName == tcrvrvv::GgmlBlockDotQ40Q80Op::getOperationName();
  const bool typedFlatLoopPath = isQ80TypedFlat || isQ40TypedFlat;
  const std::int64_t configSEW = typedFlatLoopPath ? 8 : 32;
  const llvm::StringRef configLMUL =
      isQ40TypedFlat ? "m1" : (isQ80TypedFlat ? "m2" : "m1");

  // The per-block reduce seed (0), a variant-scope value that dominates the
  // in-region standalone_reduce. Only the typed path needs it; adding it to the
  // monolith path would perturb its byte-exact ABI value set.
  mlir::Value zeroSeed;
  if (typedFlatLoopPath)
    zeroSeed = createRuntimeABIValue(builder, loc, "accumulator-input-buffer",
                                     "zero_seed", "const int32_t *",
                                     "loop-body:reduce-seed", runtimeABIType);

  tcrvrvv::SetVLOp setvl =
      createSetVL(builder, loc, n, configSEW, configLMUL, policy);
  tcrvrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), configSEW, configLMUL, policy,
                   kernelName, selectedVariantSymbol, rvvRequires);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  if (typedFlatLoopPath) {
    // The auto-constructed typed flat block-dot loop chain (brick 1 -> integer
    // core -> brick 2 -> brick 3 -> yield are op structure inside the region).
    createTypedFlatBlockDotLoopChain(builder, loc, entry, weight, activation,
                                     out, n, setvl.getVl(), zeroSeed);
  } else {
    // The auto-constructed block dot-product op (the scale model, integer core,
    // super-block bit-dance, codebook gather, and deferred fold are op structure).
    (void)createBlockDot(builder, loc, entry, weight, activation, out, n,
                         setvl.getVl());
  }

  mlir::FailureOr<std::string> fallbackOrigin =
      materializeConservativeFallbackVariantViaPlugin(
          entry, builder, kernel, source.func, registry, fallbackVariantSymbol);
  if (mlir::failed(fallbackOrigin))
    return mlir::failure();

  builder.setInsertionPointToEnd(&kernel.getBody().front());
  createDispatch(builder, loc, entry.dispatchPolicy, selectedVariantSymbol,
                 fallbackVariantSymbol, *fallbackOrigin);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// (3) The pass: marker-gated, source-only, one instance per table row.
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

mlir::LogicalResult
requireRVVSourceOnlyModule(const MonolithicBlockDotOpEntry &entry,
                           mlir::ModuleOp module) {
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
  return fail(entry, staleOp,
              "source materializer requires RVV source-only MLIR input; "
              "pre-existing selected-boundary or variant residue is not "
              "accepted");
}

std::string getKernelName(const MonolithicBlockDotOpEntry &entry,
                          mlir::ModuleOp module) {
  auto kernelNameAttr =
      module->getAttrOfType<mlir::StringAttr>(kSourceKernelAttrName);
  if (kernelNameAttr && !kernelNameAttr.getValue().trim().empty())
    return kernelNameAttr.getValue().trim().str();
  return entry.kernelDefault.str();
}

class MaterializeRVVMonolithicBlockDotSourceFrontDoorPass final
    : public mlir::PassWrapper<
          MaterializeRVVMonolithicBlockDotSourceFrontDoorPass,
          mlir::OperationPass<mlir::ModuleOp>> {
public:
  MaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
      const MonolithicBlockDotOpEntry *entry,
      const ExtensionPluginRegistry *registry)
      : entry(entry), registry(registry) {}

  llvm::StringRef getArgument() const final { return entry->passArgument; }
  llvm::StringRef getDescription() const final { return kPassDescription; }

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
          << "RVV monolithic ggml block-dot source front door requires an "
             "injected extension-plugin registry to dispatch the conservative "
             "fallback";
      signalPassFailure();
      return;
    }

    auto marker =
        module->getAttrOfType<mlir::StringAttr>(kSourceFrontDoorAttrName);
    if (!marker || marker.getValue().trim() != entry->markerValue)
      return; // not our marker: leave the module untouched.

    if (hasStaleRVVLoweringSeedMetadata(module)) {
      (void)fail(*entry, module,
                 "rejected stale tcrv_rvv.lowering_seed metadata as RVV "
                 "source-route authority");
      signalPassFailure();
      return;
    }
    if (mlir::failed(requireRVVSourceOnlyModule(*entry, module))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<mlir::func::FuncOp, 2> funcs;
    module.walk([&](mlir::func::FuncOp func) { funcs.push_back(func); });
    if (funcs.size() != 1) {
      (void)fail(*entry, module,
                 "source module must contain exactly one RVV block-dot source "
                 "function candidate");
      signalPassFailure();
      return;
    }

    mlir::FailureOr<BlockDotSourceMatch> source =
        matchBlockDotSourceFunc(*entry, funcs.front());
    if (mlir::failed(source)) {
      signalPassFailure();
      return;
    }

    std::string kernelName = getKernelName(*entry, module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(
            materializeKernel(builder, kernelName, *registry, *entry, *source))) {
      signalPassFailure();
      return;
    }

    module->removeAttr(kSourceFrontDoorAttrName);
    module->removeAttr(kSourceKernelAttrName);
  }

private:
  const MonolithicBlockDotOpEntry *entry = nullptr;
  const ExtensionPluginRegistry *registry = nullptr;
};

std::unique_ptr<::mlir::Pass>
createMaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
    const MonolithicBlockDotOpEntry *entry,
    const ExtensionPluginRegistry *registry) {
  return std::make_unique<MaterializeRVVMonolithicBlockDotSourceFrontDoorPass>(
      entry, registry);
}

} // namespace

llvm::Error registerRVVMonolithicBlockDotSourceFrontDoorPasses(
    llvm::StringRef ownerPlugin, const ExtensionPluginRegistry &registry,
    llvm::SmallVectorImpl<SourceFrontDoorPassRegistration> &out) {
  const ExtensionPluginRegistry *registryPtr = &registry;
  for (const MonolithicBlockDotOpEntry &entry : monolithicBlockDotOpTable()) {
    const MonolithicBlockDotOpEntry *entryPtr = &entry;
    out.push_back(SourceFrontDoorPassRegistration(
        ownerPlugin, entry.passArgument, kPassDescription,
        [entryPtr, registryPtr] {
          return createMaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
              entryPtr, registryPtr);
        },
        SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
            ExplicitOnly));
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::plugin::rvv
