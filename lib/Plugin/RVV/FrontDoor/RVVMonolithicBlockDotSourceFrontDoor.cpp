//===- RVVMonolithicBlockDotSourceFrontDoor.cpp -------------------------===//
//
// The ONE table-driven monolithic ggml block-dot source front door -- the collapse
// of the 24 former per-op scaffold-constructor passes into a single generic pass
// driven by `monolithicBlockDotOpTable()`. Each table row carries the per-op
// construction DATA; this file carries the ONE shared construction MECHANISM.
//
// For a marked GENERIC source carrying a ggml `ggml_vec_dot_<op>` OPERATOR IDENTITY
// (the vec_dot ABI roles), the pass auto-constructs the complete weft.exec.kernel +
// variant + dispatch/fallback scaffold around ONE attr-less (modulo the row's
// integer_core_lmul) weft_rvv.<op>_block_dot op, instead of a per-kernel
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

#include "Weft/Plugin/RVV/RVVMonolithicBlockDotSourceFrontDoor.h"
#include "Weft/Plugin/RVV/RVVFormulaCatalog.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVCapabilityProfile.h"
#include "Weft/Plugin/RVV/RVVExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVGearboxSchedule.h"
#include "Weft/Plugin/RVV/RVVSourceScheduleFormula.h"
#include "Weft/Plugin/RVV/RVVMonolithicBlockDotFamily.h"
#include "Weft/Support/CapabilityModel.h"
#include "Weft/Support/DeclaredInstanceHash.h"
#include "Weft/Support/RuntimeABI.h"

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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>

namespace weft::plugin::rvv {
namespace {

namespace weftexec = ::weft::exec;
namespace weftrvv = ::weft::rvv;

// The marker/kernel module attributes shared by the whole family (identical across
// every former per-op front door; only the marker VALUE is per-op DATA in the row).
constexpr llvm::StringLiteral kSourceFrontDoorAttrName(
    "weft_rvv.source_front_door");
constexpr llvm::StringLiteral kSourceKernelAttrName("weft_rvv.source_kernel");
constexpr llvm::StringLiteral kSeedAttrName("weft_rvv.lowering_seed");

constexpr llvm::StringLiteral kRVVCapabilitySymbol("rvv");
constexpr llvm::StringLiteral kOriginAttrName("origin");
constexpr llvm::StringLiteral kRequiresAttrName("requires");

// The single generic pass description (shown in --help for every family front door;
// not a byte-exact-gated string, so it is shared rather than per-op prose).
constexpr llvm::StringLiteral kPassDescription(
    "Auto-construct the attr-less weft_rvv.<op>_block_dot op + "
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
// (2) Body builder: auto-construct the weft_rvv.<op>_block_dot op scaffold
//     (kernel + variant + dispatch/fallback) generically from the table row.
//===----------------------------------------------------------------------===//

mlir::FlatSymbolRefAttr symbolRef(mlir::OpBuilder &builder,
                                  llvm::StringRef symbol) {
  return mlir::FlatSymbolRefAttr::get(builder.getContext(), symbol);
}

void createCapability(mlir::OpBuilder &builder, mlir::Location loc,
                      llvm::StringRef symbol, llvm::StringRef id,
                      llvm::StringRef kind) {
  mlir::OperationState state(loc, weftexec::CapabilityOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(symbol));
  state.addAttribute("id", builder.getStringAttr(id));
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("status", builder.getStringAttr("available"));
  (void)builder.create(state);
}

mlir::ArrayAttr createRequires(mlir::OpBuilder &builder, llvm::StringRef symbol) {
  return builder.getArrayAttr({symbolRef(builder, symbol)});
}

weftrvv::PolicyAttr createAgnosticPolicy(mlir::OpBuilder &builder) {
  return weftrvv::PolicyAttr::get(builder.getContext(),
                                  weftrvv::TailPolicy::Agnostic,
                                  weftrvv::MaskPolicy::Agnostic);
}

mlir::Value createRuntimeABIValue(mlir::OpBuilder &builder, mlir::Location loc,
                                  llvm::StringRef role, llvm::StringRef cName,
                                  llvm::StringRef cType, llvm::StringRef purpose,
                                  mlir::Type resultType) {
  mlir::OperationState state(loc,
                             weftrvv::RuntimeABIValueOp::getOperationName());
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

weftrvv::SetVLOp createSetVL(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value n, std::int64_t sew,
                             llvm::StringRef lmul, weftrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weftrvv::SetVLOp::getOperationName());
  state.addOperands(n);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addTypes(weftrvv::VLType::get(builder.getContext()));
  return llvm::cast<weftrvv::SetVLOp>(builder.create(state));
}

weftrvv::WithVLOp createWithVL(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value vl, std::int64_t sew,
                               llvm::StringRef lmul,
                               weftrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weftrvv::WithVLOp::getOperationName());
  state.addOperands(vl);
  state.addAttribute("sew", builder.getI64IntegerAttr(sew));
  state.addAttribute("lmul", builder.getStringAttr(lmul));
  state.addAttribute("policy", policy);
  state.addRegion();
  auto withVL = llvm::cast<weftrvv::WithVLOp>(builder.create(state));
  withVL.getBody().emplaceBlock();
  return withVL;
}

// ---------------------------------------------------------------------------
// Typed flat block-dot loop-body construction (M-FLAT step 5b, q8_0 only).
//
// The gated typed path builds the COMPLETE per-block typed chain
// (weft_rvv.typed_flat_block_dot_loop_body region: brick 1 dual-fp16 scale
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
  mlir::OperationState state(loc, weftrvv::LoadOp::getOperationName());
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
                             weftrvv::WideningProductOp::getOperationName());
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
      loc, weftrvv::PackedI4OffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("signed_packed_i4_offset_binary_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// The q4_1 asymmetric UNSIGNED-nibble packed-i4 x plain-i8 integer core: ONE
// packed-i4 weight operand (each u8 packs two UNSIGNED nibbles in [0,15]) + TWO
// plain int8 activation operands (the q8 low half paired with the low nibbles, the
// q8 high half with the high nibbles) -> ONE widened i16 product. The q4_1 `-8`
// shift is folded into the block minimum (a separate min brick), so the decode is
// a plain unsigned split (vand 0x0F / vsrl 0x04) with NO bias -- DISTINCT from the
// q4_0 offset-binary sibling. The m1 rung (i4m1 x i8m1 low/high -> i16m2) is the
// only declared rung.
mlir::Value createUnsignedNibbleXI8Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value vl,
    mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::UnsignedNibbleXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, vl});
  state.addAttribute("kind",
                     builder.getStringAttr("unsigned_nibble_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// The q5_0 asymmetric FIVE-BIT offset-binary packed weight x plain-i8 integer
// core: ONE UNSIGNED packed-i4 weight operand (each u8 packs two nibbles) MERGED
// with a per-element 5th bit read from the scalar-domain qh field (the qh_source
// gate-only token from a preceding block_five_bit_qh_source brick, NOT a byte
// offset baked on THIS op) + TWO plain int8 activation operands (the q8 low half
// paired with the low nibbles, the q8 high half with the high nibbles) -> ONE
// widened i16 product. The `-16` offset-binary bias is applied in the decode. The
// m1 flat-cohort rung (i4m1 + qh x i8m1 low/high -> i16m2) is the only rung.
mlir::Value createFiveBitOffsetBinaryXI8Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value qhSource, mlir::Value activationLow, mlir::Value activationHigh,
    mlir::Value vl, mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::FiveBitOffsetBinaryXI8ProductOp::getOperationName());
  state.addOperands({weight, qhSource, activationLow, activationHigh, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("five_bit_offset_binary_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

// iq4_nl codebook table broadcast (the CODEBOOK class's prerequisite structure,
// 2nd primitive class): the 16-entry non-linear int8 kvalues table materialized as
// a structured const + broadcast-loaded ONCE into the i8 vreg the gather indexes.
// Consumes NO SSA operands (the table is a compile-time constant); the codebook
// array + the C symbol are the two structural attrs.
mlir::Value createCodebookTableBroadcast(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         llvm::ArrayRef<std::int8_t> codebook,
                                         llvm::StringRef tableSymbol,
                                         mlir::Type tableType) {
  mlir::OperationState state(
      loc, weftrvv::CodebookTableBroadcastOp::getOperationName());
  state.addAttribute("codebook", builder.getDenseI8ArrayAttr(codebook));
  state.addAttribute("table_symbol", builder.getStringAttr(tableSymbol));
  state.addTypes(tableType);
  return builder.create(state)->getResult(0);
}

// The iq4_nl asymmetric CODEBOOK-GATHER packed-i4 x plain-i8 integer core: ONE
// UNSIGNED packed-i4 weight operand (each u8 packs two 4-bit table INDICES) + TWO
// plain int8 activation operands (the q8 low half paired with the low nibbles, the
// q8 high half with the high nibbles) + the broadcast codebook `table` operand ->
// ONE widened i16 product. Unlike the q4_0 offset-binary / q4_1 unsigned-nibble
// siblings, the 4-bit nibble is an INDEX vrgather-decoded through the 16-entry
// kvalues table (NOT an arithmetic decode); it then feeds the SAME asymmetric
// widening product tail (vwmul low + vwmacc high) -> i16m2 the offset-binary
// sibling uses. The m1 rung (i4m1 idx -> gather -> i8m1 x i8m1 low/high -> i16m2)
// is the only declared rung.
mlir::Value createCodebookGatherXI8Product(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value weight,
    mlir::Value activationLow, mlir::Value activationHigh, mlir::Value table,
    mlir::Value vl, mlir::Type productType, llvm::StringRef productRelation) {
  mlir::OperationState state(
      loc, weftrvv::CodebookGatherXI8ProductOp::getOperationName());
  state.addOperands({weight, activationLow, activationHigh, table, vl});
  state.addAttribute(
      "kind", builder.getStringAttr("signed_codebook_gather_x_i8_product"));
  state.addAttribute("product_relation", builder.getStringAttr(productRelation));
  state.addTypes(productType);
  return builder.create(state)->getResult(0);
}

mlir::Value createStandaloneReduce(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value input, mlir::Value accumulatorSeed,
                                   mlir::Value vl, mlir::Type resultType) {
  mlir::OperationState state(loc,
                             weftrvv::StandaloneReduceOp::getOperationName());
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
      loc, weftrvv::BlockFp16ScaleProductOp::getOperationName());
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

// Family-B min brick: per-block dual-fp16 MIN/SUM correction product `m_x * s_y`
// (block_index-sourced form). Distinct op TYPE from brick 1; names the SAME
// weight/activation ABI bases + block_index so the emitter's per-block base memo
// hits. The min/sum headers sit at lhs_min_byte_offset / rhs_sum_byte_offset
// within each block base (2/2 for q4_1).
mlir::Value createBlockFp16MinProduct(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhsMinBase,
    mlir::Value rhsSumBase, mlir::Value blockIndex, std::int64_t lhsBlockStride,
    std::int64_t rhsBlockStride, std::int64_t lhsMinByteOffset,
    std::int64_t rhsSumByteOffset) {
  mlir::OperationState state(
      loc, weftrvv::BlockFp16MinProductOp::getOperationName());
  state.addOperands({lhsMinBase, rhsSumBase, blockIndex});
  state.addAttribute("kind",
                     builder.getStringAttr("dual_fp16_per_block_min_product"));
  state.addAttribute("scale_model",
                     builder.getStringAttr("dual-fp16-per-block-m_x.s_y"));
  state.addAttribute("lhs_min_byte_offset",
                     builder.getI64IntegerAttr(lhsMinByteOffset));
  state.addAttribute("rhs_sum_byte_offset",
                     builder.getI64IntegerAttr(rhsSumByteOffset));
  state.addAttribute("lhs_block_stride",
                     builder.getI64IntegerAttr(lhsBlockStride));
  state.addAttribute("rhs_block_stride",
                     builder.getI64IntegerAttr(rhsBlockStride));
  state.addTypes(builder.getF32Type());
  return builder.create(state)->getResult(0);
}

// q5_0 qh-source brick: the per-block 5th-bit SOURCE (block_index-sourced form).
// Names the SAME weight ABI base + block_index as the nibble weight load so the
// emitter per-block base memo hits; the qh header sits at qh_byte_offset within
// each weight block base. Its i32 result is a GATE-ONLY token the five-bit product
// op names (never materialized standalone) -- the bytes are re-read from this
// brick's own qh_base + qh_byte_offset in the emitter (operand/source-driven, NOT
// a descriptor-baked offset on the product op).
mlir::Value createBlockFiveBitQhSource(mlir::OpBuilder &builder,
                                       mlir::Location loc, mlir::Value qhBase,
                                       mlir::Value blockIndex,
                                       std::int64_t blockStride,
                                       std::int64_t qhByteOffset) {
  mlir::OperationState state(
      loc, weftrvv::BlockFiveBitQhSourceOp::getOperationName());
  state.addOperands({qhBase, blockIndex});
  state.addAttribute("kind",
                     builder.getStringAttr("block_five_bit_qh_source"));
  state.addAttribute("qh_byte_offset", builder.getI64IntegerAttr(qhByteOffset));
  state.addAttribute("block_stride", builder.getI64IntegerAttr(blockStride));
  state.addTypes(builder.getI32Type());
  return builder.create(state)->getResult(0);
}

// integer-core -> scalar bridge: i32 m1 vector lane0 -> scalar i32 sumi.
mlir::Value createTypedVectorLane0ToScalarExtract(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Value input,
                                                  mlir::Value vl) {
  mlir::OperationState state(
      loc, weftrvv::TypedVectorLane0ToScalarExtractOp::getOperationName());
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
mlir::Value createBlockComputedScaleDequant(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value sumi,
    mlir::Value computedScale, mlir::Value minTerm = mlir::Value()) {
  mlir::OperationState state(
      loc, weftrvv::BlockComputedScaleDequantOp::getOperationName());
  if (minTerm)
    state.addOperands({sumi, computedScale, minTerm});
  else
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
      loc, weftrvv::CrossBlockF32AccumulateOp::getOperationName());
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
      loc, weftrvv::TypedFlatBlockDotLoopYieldOp::getOperationName());
  state.addOperands(accNext);
  (void)builder.create(state);
}

// The gated typed q8_0 path: build the whole per-block typed chain inside a new
// weft_rvv.typed_flat_block_dot_loop_body region (result-less, region-carrying),
// replacing the ONE monolith block-dot op. The loop-body op's block strides / qk /
// quant offset come from entry.facts (by-name lookup); multi_block_factor is OMITTED
// (absent = mbf 1). weight/activation/out/n are the shared variant-scope ABI values;
// zeroSeed is the variant-scope reduce seed (dominates the in-region reduce); vl is
// the setvl VL referenced freely by the in-region loads/product/reduce/extract.
// `lmul` is the integer-core LMUL schedule ("m1"|"m2"): it drives the
// integer_core_lmul stamp, the coreLmul/wideLmul region vector types, and (for the
// q8_0 whole-block path) the widening product_relation as ONE consistent source, so
// the four verifier-cross-pinned knobs cannot diverge. It MUST agree with the
// enclosing setvl/with_vl config LMUL (the caller passes the same value to both).
// q8_0 defaults to "m2" (the anchor); the half-block packed-i4 formats always pass
// "m1" (their region product_relation pins i8m1).
void createTypedFlatBlockDotLoopChain(mlir::OpBuilder &builder,
                                      mlir::Location loc,
                                      const MonolithicBlockDotOpEntry &entry,
                                      mlir::Value weight, mlir::Value activation,
                                      mlir::Value out, mlir::Value n,
                                      mlir::Value vl, mlir::Value zeroSeed,
                                      llvm::StringRef lmul) {
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
      entry.opName == weftrvv::GgmlBlockDotQ40Q80Op::getOperationName();
  // q4_1 (Family-B): shares q4_0's HALF-block m1 packed-i4 shape (3 loads, m1
  // core), diverging only in {u8 weight load, unsigned-nibble product op, the
  // added MIN brick, the scale_plus_min fold}. Every q4_0-guarded knob below is
  // shared (isQ40 || isQ41) EXCEPT the three-way fold_model.
  const bool isQ41 =
      entry.opName == "weft_rvv.q4_1_q8_1_block_dot";
  // q5_0 (five-bit): shares the HALF-block m1 packed-i4 shape (3 loads, m1 core,
  // u8 weight load like q4_1), diverging in {the qh 5th-bit source brick, the
  // five-bit offset-binary product with the `-16` bias, the ScalesTimesSumi fold,
  // and a DISTINCT activation quant offset (weight qs@6, activation qs@2)}.
  const bool isQ50 =
      entry.opName == "weft_rvv.q5_0_q8_0_block_dot";
  // q5_1 (Family-B five-bit, M-FLAT cohort LAST cell): the UNION of q5_0's
  // five-bit integer core (qh 5th-bit brick + five-bit product) and q4_1's MIN
  // term (min brick + scale_plus_min fold). Every knob is shared with EITHER q5_0
  // (qh brick, u8 weight, five-bit product, divergent quant offsets) OR q4_1 (min
  // brick, scale_plus_min fold) -- no q5_1-only knob. The ONE arithmetic delta vs
  // q5_0 (applyOffsetBias=false) lives entirely in the emit driver.
  const bool isQ51 =
      entry.opName == "weft_rvv.q5_1_q8_1_block_dot";
  // iq4_nl (CODEBOOK class, 2nd primitive class): shares the q4_1 HALF-block m1
  // 3-load shape (a u8 packed-i4 weight + the two plain-i8 q8 halves), but the
  // weight nibble is a codebook INDEX (vrgather through the 16-entry kvalues
  // table) NOT an arithmetic decode, so its integer core is the two codebook
  // bricks (codebook_table_broadcast + codebook_gather_x_i8_product) rather than a
  // packed-i4 product. Its fold is SumiTimesScales (the else-default, node-identical
  // to q8_0). It is NOT in isHalfBlock (that flag gates the offset-binary/unsigned
  // packed-i4 product branch); its high-half activation offset is sourced the SAME
  // way (activation_high_byte_offset).
  const bool isIq4Nl = entry.opName == "weft_rvv.iq4_nl_q8_0_block_dot";
  const bool isHalfBlock = isQ40 || isQ41 || isQ50 || isQ51;
  std::int64_t activationHighOffset =
      (isHalfBlock || isIq4Nl) ? factByName("activation_high_byte_offset") : 0;

  mlir::OperationState loopState(
      loc, weftrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
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
      builder.getStringAttr(isQ50 ? "scales_times_sumi"
                                   : ((isQ41 || isQ51) ? "scale_plus_min"
                                            : (isQ40 ? "left_assoc"
                                                     : "sumi_times_scales"))));
  loopState.addAttribute("integer_core_lmul", builder.getStringAttr(lmul));
  // strip_elision is a VLEN-legality knob, not free. The elided single-cover emits
  // ONE vsetvl(blockLen); that only covers the whole strip when blockLen <= VLMAX.
  // q8_0's whole-block core has blockLen = qk = 32, so at m2 (VLMAX 32 @VLEN128) the
  // elided cover is whole-block-legal, but at m1 (VLMAX 16 @VLEN128) it would
  // SILENTLY cover half the block -- illegal below VLEN256. The half-block packed-i4
  // formats have blockLen = qk/2 = 16, so their m1 elided cover is whole-strip-legal
  // at VLEN128. So the ONLY form that must fall back to the VLEN-robust re-strip is
  // q8_0 at m1; every currently-live path (q8_0-m2, half-block-m1) keeps "elided"
  // byte-for-byte.
  // iq4_nl's codebook core is a half-block-length strip (qk/2 = 16 elements at
  // e8m1, VLMAX 16 @VLEN128 = whole-strip-legal), so like the packed-i4 half-block
  // formats it keeps "elided"; only q8_0's whole-block (blockLen = qk = 32) m1 core
  // needs the VLEN-robust re-strip.
  loopState.addAttribute(
      "strip_elision",
      builder.getStringAttr((!isHalfBlock && !isIq4Nl && lmul == "m1") ? "robust"
                                                                       : "elided"));
  // Deterministic final schedule fields are constructed explicitly.  The
  // emitter never interprets absence as a code-shape choice.
  loopState.addAttribute("multi_block_factor", builder.getI64IntegerAttr(1));
  loopState.addAttribute("fold_structure", builder.getStringAttr("per-block"));
  loopState.addAttribute("numerics_tier", builder.getStringAttr("strict"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  mlir::MLIRContext *ctx = builder.getContext();
  // The integer-core widths derive from the ONE `lmul` schedule (widening steps
  // LMUL up one rung): q8_0 anchors i8m2 -> i16m4 (lmul "m2") but is also
  // constructible at i8m1 -> i16m2 (lmul "m1"); q4_0's half-block packed-i4 core
  // anchors i8m1 -> i16m2 (the caller always passes "m1"). The i32m1 reduce lane
  // is shared.
  llvm::StringRef coreLmul = lmul;
  llvm::StringRef wideLmul = (lmul == "m1") ? "m2" : "m4";
  mlir::Type i8VecType =
      weftrvv::VectorType::get(ctx, builder.getI8Type(), coreLmul);
  // q4_1's packed weight strip is UNSIGNED (the nibble value IS the weight, the
  // `-8` folded into the block minimum), so the region weight LoadOp carries a
  // u8-m1 vector type (cf. the codebook front door). q4_0's weight strip stays
  // signed i8.
  mlir::Type ui8VecType = weftrvv::VectorType::get(
      ctx, builder.getIntegerType(8, /*isSigned=*/false), coreLmul);
  mlir::Type i16VecType =
      weftrvv::VectorType::get(ctx, builder.getI16Type(), wideLmul);
  mlir::Type i32VecType =
      weftrvv::VectorType::get(ctx, builder.getI32Type(), "m1");

  // brick 1: the per-block d_x * d_y fp16 scale product over block_index (shared;
  // the AoS strides differ per format but the scale model is identical).
  mlir::Value dd = createBlockFp16ScaleProduct(
      builder, loc, weight, activation, blockIndex, weightStride,
      activationStride);

  // q4_1 min brick: the per-block m_x * s_y correction product (Family-B). Names
  // the SAME weight/activation ABI bases + block_index as brick 1 so the emitter
  // per-block base memo hits; the min/sum headers sit at weight_min_byte_offset /
  // activation_sum_byte_offset. Null (no brick) for q8_0/q4_0.
  mlir::Value minTerm;
  if (isQ41 || isQ51)
    minTerm = createBlockFp16MinProduct(
        builder, loc, weight, activation, blockIndex, weightStride,
        activationStride, factByName("weight_min_byte_offset"),
        factByName("activation_sum_byte_offset"));

  // q5_0 qh-source brick: the per-block 5th-bit SOURCE token the five-bit product
  // names. Like the q4_1 min brick, names the SAME weight ABI base + block_index
  // (the qh header lives WITHIN the weight block), so the emitter per-block base
  // memo hits. Null (no brick) for q8_0/q4_0/q4_1.
  mlir::Value qhSource;
  if (isQ50 || isQ51)
    qhSource = createBlockFiveBitQhSource(builder, loc, weight, blockIndex,
                                          weightStride,
                                          factByName("weight_qh_byte_offset"));

  // The vector integer core diverges by format. q8_0: two per-block i8 loads ->
  // signed widening product. q4_0/q4_1: ONE packed-i4 weight load + TWO plain-i8
  // activation loads (the q8 low half at quant_off, the q8 high half at
  // quant_off + activation_high_byte_offset) -> the asymmetric packed-i4 x i8
  // product. q4_0 loads the weight SIGNED + offset-binary decode; q4_1 loads it
  // UNSIGNED (u8) + unsigned-nibble decode.
  // Activation quant offset. For q4_0/q4_1 the weight and activation qs offsets
  // COINCIDE, so both loads use quant_byte_offset. q5_0 is the FIRST flat op where
  // they DIVERGE (weight qs@6, activation qs@2): its avLow/avHigh MUST read the
  // separate activation_quant_byte_offset fact (the weight strip still uses
  // quant_byte_offset). Keeping this = quantByteOffset off the q5_0 path leaves the
  // q4_0/q4_1 loads byte-identical.
  std::int64_t activationQuantByteOffset =
      (isQ50 || isQ51) ? factByName("activation_quant_byte_offset")
                       : quantByteOffset;

  mlir::Value prod;
  if (isIq4Nl) {
    // iq4_nl CODEBOOK integer core (2nd primitive class): the 16-entry non-linear
    // int8 kvalues table broadcast ONCE + the u8 packed-i4 weight strip + the two
    // plain-i8 q8 halves -> the asymmetric codebook-gather product. The weight and
    // activation qs offsets COINCIDE (weight qs@2, activation qs@2, like q4_0/q4_1),
    // so both use quantByteOffset; the high half sits at quant_off +
    // activation_high_byte_offset. The nibble is a vrgather INDEX (NOT an arithmetic
    // decode); the gathered i8 weight lanes feed the SAME widening product tail
    // (i8m1 x i8m1x2 -> i16m2). fold_model is SumiTimesScales (the else-default).
    mlir::Value table = createCodebookTableBroadcast(
        builder, loc, entry.codebook, "weft_iq4_nl_kvalues", i8VecType);
    mlir::Value wv =
        createRVVBlockLoad(builder, loc, weight, vl, blockIndex, weightStride,
                           quantByteOffset, ui8VecType);
    mlir::Value avLow =
        createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                           activationStride, quantByteOffset, i8VecType);
    mlir::Value avHigh = createRVVBlockLoad(
        builder, loc, activation, vl, blockIndex, activationStride,
        quantByteOffset + activationHighOffset, i8VecType);
    prod = createCodebookGatherXI8Product(
        builder, loc, wv, avLow, avHigh, table, vl, i16VecType,
        "codebook-gather-i8-x-i8x2-to-i16");
  } else if (isHalfBlock) {
    // packed-i4 weight strip (base + ib*stride + quant_off). q4_1/q5_0 = u8 (the
    // nibble decode is UNSIGNED), q4_0 = signed i8.
    mlir::Value wv = createRVVBlockLoad(
        builder, loc, weight, vl, blockIndex, weightStride, quantByteOffset,
        (isQ41 || isQ50 || isQ51) ? ui8VecType : i8VecType);
    // q8 low half (activation quant_off) and high half (+ activation_high_offset).
    mlir::Value avLow = createRVVBlockLoad(builder, loc, activation, vl, blockIndex,
                                           activationStride,
                                           activationQuantByteOffset, i8VecType);
    mlir::Value avHigh = createRVVBlockLoad(
        builder, loc, activation, vl, blockIndex, activationStride,
        activationQuantByteOffset + activationHighOffset, i8VecType);
    if (isQ50 || isQ51)
      // asymmetric FIVE-BIT offset-binary packed-i4 (+ qh 5th bit) x i8 product
      // (i4m1 + qh x i8m1x2 -> i16m2). The qh 5th-bit source is the qh brick's
      // gate-only token, NOT a byte offset on this product op. The `-16` bias is a
      // pure EMIT-driver choice (q5_0 on, q5_1 off -- q5_1's bias lives in its
      // per-block MIN scale); the op is structurally identical for both formats.
      prod = createFiveBitOffsetBinaryXI8Product(
          builder, loc, wv, qhSource, avLow, avHigh, vl, i16VecType,
          "five-bit-offset-binary-i4m1-x-i8m1x2-to-i16m2");
    else if (isQ41)
      // asymmetric UNSIGNED-nibble packed-i4 x i8 product (i4m1 x i8m1x2 -> i16m2).
      prod = createUnsignedNibbleXI8Product(
          builder, loc, wv, avLow, avHigh, vl, i16VecType,
          "unsigned-nibble-i4m1-x-i8m1x2-to-i16m2");
    else
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
    // signed widening product: i8m2 x i8m2 -> i16m4 (lmul "m2", the anchor) or
    // i8m1 x i8m1 -> i16m2 (lmul "m1", the byte-anchor dot-reduce rung). Both
    // relations are admitted by the widening_product verifier; the arithmetic is
    // bit-identical (LMUL is a schedule knob, not an arithmetic one).
    prod = createWideningProduct(builder, loc, wv, av, vl, i16VecType,
                                 (lmul == "m1") ? "signed-i8m1xi8m1-to-i16m2"
                                                : "signed-i8m2xi8m2-to-i16m4");
  }
  // reduce i16<wide> -> i32m1 lane0, then extract lane0 -> scalar i32 sumi.
  mlir::Value red =
      createStandaloneReduce(builder, loc, prod, zeroSeed, vl, i32VecType);
  mlir::Value sumi =
      createTypedVectorLane0ToScalarExtract(builder, loc, red, vl);
  // brick 2: (float)sumi * (d_x * d_y) (+ m_x*s_y min_term for q4_1).
  mlir::Value bterm =
      createBlockComputedScaleDequant(builder, loc, sumi, dd, minTerm);
  // brick 3: sumf + term (cross-block fp32 fold).
  mlir::Value accNext = createCrossBlockF32Accumulate(builder, loc, acc, bterm);
  createTypedFlatBlockDotLoopYield(builder, loc, accNext);
}

// ---------------------------------------------------------------------------
// The q1_0 (BINARY {-1,+1}-sign class) sibling of createTypedFlatBlockDotLoopChain
// -- the LAST flat block-dot family member, the genuine structural-special case
// (C_construct 26->27). Unlike q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl (a single per-block
// integer core folded by the shared brick 1 (scale) -> brick 2 (dequant) -> brick 3
// (cross-block accumulate) chain), q1_0's per-super-block contribution is a
// FOUR-sub-block binary sign decode with a DISTINCT TWO-LEVEL fp32 fold
// (`d0 * Σ_k(d1_k * sumi_block_k)`), so no existing flat brick chain expresses it.
// The net-new marginal cost is ONE brick -- the q1_0 BINARY-sign integer core
// (GgmlBlockDotQ10Q80BinarySignCoreOp) -- carrying the whole per-super-block body;
// the two-level fold is emitter-inlined (the super-block scalar-core precedent
// tq1_0/iq1_s applied to the FLAT loop op). q1_0's activation is a FLAT block_q8_0
// stream (four 34-byte q8_0 blocks per q1_0 super-block), so it uses the FLAT loop
// op (typed_flat_block_dot_loop_body, fold_model "flat_binary_two_level") -- NOT the
// super-block one (q8_K). The OUTER with_vl frame stays SEW32/m1 (like the monolith
// / iq4_nl): the e8m2 binary sign decode runs its OWN vsetvl INSIDE the brick, so
// this chain is dispatched OUTSIDE typedFlatLoopPath (which would force SEW8). The
// loop op + brick are left attr-less (default m2 anchor = the byte-exact CORE
// target); q1_0's Win-A gearbox lives on the binary-sign brick (kernel key "q1_0").
void createTypedFlatBlockDotLoopChainQ10(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed flat binary-sign q1_0 chain: missing block-format fact");
  };
  std::int64_t qk = factByName("qk");                            // 128
  std::int64_t weightStride = factByName("weight_block_stride");         //  18
  std::int64_t activationStride = factByName("activation_block_stride"); //  34
  std::int64_t blocksPerWeight = factByName("activation_blocks_per_weight"); // 4
  std::int64_t weightQuantOffset = factByName("weight_quant_byte_offset");   // 2
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");                 //   2

  mlir::Type i32ScalarType = builder.getI32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute("kind",
                         builder.getStringAttr("typed_flat_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "flat_binary_two_level" KEYS the emitter dispatch (the q1_0 branch)
  // + the two-level scalar fold; the emitter disambiguates q1_0 by the in-region
  // binary-sign integer-core brick op TYPE. integer_core_lmul / multi_block_factor /
  // strip_elision are LEFT OFF (attr-less = the default m2 anchor, byte-exact target).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("flat_binary_two_level"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the q1_0 BINARY-sign integer core (the FOUR q8_0 sub-blocks, each a
  // vlm_v_b{ratio} packed-bit sign mask + vle8 q8 + i8-domain vneg/vmerge -> ONE
  // vwredsum i8->i16m1, plus the emitter-inlined two-level fp32 fold). The LIVE
  // operands are the weight base (%vx) + activation base (%vy) + n + vl +
  // block_index; it produces ONE scalar i32 SSA result (a placeholder -- the
  // emitter re-emits the whole body including the fold). Per-super-block address
  // vx + ib*18, vy + (ib*4 + k)*34. Left attr-less so the gearbox is free to stamp
  // integer_core_lmul m2/m1 from the VLEN capability fact (kernel key "q1_0").
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotQ10Q80BinarySignCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, blockIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_q1_0_q8_0_binary_sign_core"));
    s.addAttribute("scale_model",
                   builder.getStringAttr("binary-sign-per-bit"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("activation_blocks_per_weight",
                   builder.getI64IntegerAttr(blocksPerWeight));
    s.addAttribute("weight_quant_byte_offset",
                   builder.getI64IntegerAttr(weightQuantOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The loop yield carries the loop-carried acc UNCHANGED (the two-level fold is
  // emitter-inlined by the binary-sign brick lowering, byte-exact = the monolith
  // sumf; mirrors the tq1_0 scalar-core yield contract). The flat verifier requires
  // an f32 acc_next -- acc IS the f32 region arg 1.
  createTypedFlatBlockDotLoopYield(builder, loc, acc);
}

// ---------------------------------------------------------------------------
// The nvfp4 (SECOND FP4-CODEBOOK class, NVIDIA's FP4) sibling of
// createTypedFlatBlockDotLoopChainQ10 -- the LAST dispatch-wired vec_dot to flip
// (C_construct 27->28, closing the ① G1 literal-block-dot zoo). nvfp4 is a
// SUPER-BLOCK codebook quant (block_nvfp4 = {uint8_t d[4]; uint8_t qs[32]}, QK=64,
// four 16-element sub-blocks) whose 64 elements span TWO block_q8_0 activation
// blocks -- a FLAT block_q8_0 stream (like q1_0's four-block stream), so it uses the
// FLAT loop op (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") --
// NOT the super-block one (q8_K). Like q1_0 it REUSES mxfp4's 16-entry DOUBLED e2m1
// codebook GATHER (vrgather_vv_i8m1) verbatim; the genuinely-new fact is the
// per-SUB-block UE4M3 fp8 weight scale (ldexpf-based, HALF form). The whole
// per-super-block body (the four UE4M3-scaled codebook sub-blocks + the per-sub-block
// fp32 fold `sumf += (dy*d)*sumi`) is carried by ONE net-new codebook integer-core
// brick (GgmlBlockDotNVFP4Q80CodebookCoreOp) + emitter-inlined through the SAME
// emitNVFP4BlockDotBodyShared the retired monolith emitter called, so the emit is
// byte-identical to the monolith. The OUTER with_vl frame stays SEW32/m1 (like q1_0
// / the monolith): the e8m1 codebook strip runs its OWN vsetvl INSIDE the brick, so
// this chain is dispatched OUTSIDE typedFlatLoopPath (which would force SEW8). The
// loop op + brick are left attr-less on the shape knob (the codebook gather pins m1;
// the emitter's fixed vrgather/i8m1/i16m2 codebook dot matches the untuned monolith
// byte-identically). The brick's per-super-block addressing keys off the loop
// induction variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedFlatBlockDotLoopChainNvfp4(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed flat nvfp4 codebook chain: missing block-format fact");
  };
  std::int64_t qk = factByName("qk");                          // 64 (QK_NVFP4)
  std::int64_t qkSub = factByName("qk_sub");                   // 16 (QK_NVFP4_SUB)
  std::int64_t weightStride = factByName("weight_block_stride");        //  36
  std::int64_t activationStride = factByName("activation_block_stride"); //  34
  std::int64_t weightQuantOffset = factByName("weight_quant_byte_offset");   // 4
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   2
  std::int64_t activationHighOffset =
      factByName("activation_high_byte_offset");                //   8

  mlir::Type i32ScalarType = builder.getI32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedFlatBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute("kind",
                         builder.getStringAttr("typed_flat_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "flat_nvfp4_codebook" KEYS the emitter dispatch (the nvfp4 branch) +
  // the per-sub-block UE4M3-codebook fold; the emitter disambiguates nvfp4 by the
  // in-region codebook integer-core brick op TYPE.  The loop-level scheduling
  // axes are inapplicable to this closed body; the codebook brick still records
  // its fixed m1 compute anchor explicitly.
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("flat_nvfp4_codebook"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedFlatBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value blockIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value acc = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the nvfp4 FP4-CODEBOOK integer core (the 16-entry DOUBLED e2m1 codebook
  // broadcast + the four UE4M3-scaled 16-element sub-blocks' nibble split + vrgather
  // decode + asymmetric vwmul/vwmacc widening product + seed-0 vwredsum, wrapped in
  // the per-sub-block UE4M3 fp8 weight scale + the two-q8_0-block/half addressing +
  // the per-sub-block float fold into sumf). The LIVE operands are the weight base
  // (%vx) + activation base (%vy) + n + vl + block_index; it produces ONE scalar i32
  // SSA result (an UNUSED per-super-block placeholder -- nvfp4's fold is per-sub-block
  // float, no single scalar state). Per-super-block address vx + ib*36, vy +
  // (2*ib + s/2)*34. The 16-entry codebook (kvalues_mxfp4[16]) is carried as a
  // DenseI8ArrayAttr like the monolith.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotNVFP4Q80CodebookCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, blockIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_nvfp4_q8_0_codebook_core"));
    s.addAttribute("scale_model",
                   builder.getStringAttr("ue4m3-half-per-sub-block"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("qk_sub", builder.getI64IntegerAttr(qkSub));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_quant_byte_offset",
                   builder.getI64IntegerAttr(weightQuantOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("activation_high_byte_offset",
                   builder.getI64IntegerAttr(activationHighOffset));
    s.addAttribute("codebook", builder.getDenseI8ArrayAttr(entry.codebook));
    s.addAttribute("integer_core_lmul", builder.getStringAttr("m1"));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The loop yield carries the loop-carried acc UNCHANGED (the per-sub-block fold is
  // emitter-inlined by the codebook brick lowering, byte-exact = the monolith sumf;
  // mirrors the q1_0 core's yield contract). The flat verifier requires an f32
  // acc_next -- acc IS the f32 region arg 1.
  createTypedFlatBlockDotLoopYield(builder, loc, acc);
}

// ---------------------------------------------------------------------------
// Typed SUPER-BLOCK block-dot loop-body construction (M-FLAT q4_K milestone-3).
//
// The super-block sibling of createTypedFlatBlockDotLoopChain: it assembles the
// q4_K/q5_K OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a DUAL accumulator
// (the `sums` 8-lane fp32 vector + the `sumf` scalar fp32), with the 5 in-loop
// q4_K bricks (nibble_unpack -> scale_min_bit_dance -> scaled_dot -> min_term ->
// sums_fold_scale_d) + the dual yield inside the region, replacing the ONE
// monolith weft_rvv.q4_k_q8_k_block_dot op. Every brick's per-super-block
// addressing keys off the loop induction variable (region arg 0), so the emit is
// operand-driven (anti-bypass): a changed brick base operand emits a different
// base, and a dropped block_index fails to legalize.
//
// The bricks' aux8/scales/aux32 SCRATCH operands are function-scoped scratch the
// super-block loop emitter DECLARES itself (int8_t aux8[256] / uint32_t utmp[4] /
// int32_t aux32[8]); it never reads these operand slots in the loop form (it walks
// only the LIVE weight/activation/q8 bases + block_index). So the front door wires
// the vestigial scratch slots to the existing weight ABI base (%vx) rather than
// minting placeholder runtime_abi_values -- keeping the exported ggml vec_dot C
// signature the exact 4-role n/s/vx/vy list (byte-identical to the monolith),
// eliminating milestone-2's 3 dead scratch parameters. The brick verifiers relax
// their aux8/aux32 C-type check in the loop form (block_index present) since the
// scratch is emitter-owned there. The integer-core LMUL is left at the emitter
// default (mf2, no integer_core_lmul stamp on the loop op or the scaled-dot brick),
// so the untuned construction lowers byte-identically to the untuned monolith.
void createTypedSuperBlockBlockDotLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block chain: missing block-format fact");
  };
  // The OPTIONAL q5_K qh 5th-bit plane offset: PRESENT in kQ5KFacts (16), ABSENT
  // in kQ4KFacts. Its presence is the ONLY q5_K-vs-q4_K difference the front door
  // stamps -- it flows onto BRICK 1 (the nibble unpack) as weight_qh_byte_offset;
  // the emitter reads it back to set cx.hasQh. Everything else (the 5 bricks, the
  // dual accumulator, the loop op) is format-shared.
  auto factByNameOpt = [&](llvm::StringRef name) -> std::optional<std::int64_t> {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    return std::nullopt;
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32
  std::int64_t weightStride = factByName("weight_block_stride");   // 144 q4/176 q5
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");  // 16 q4/48 q5
  std::optional<std::int64_t> weightQhOffset =
      factByNameOpt("weight_qh_byte_offset");                  //  16 (q5_K only)
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 //   4
  std::int64_t weightDminOffset = factByName("weight_dmin_byte_offset"); //  2
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  std::int64_t bsumsOffset = factByName("activation_bsums_byte_offset"); // 260
  std::int64_t numSubBlocks = qk / subBlock;                   //   8

  mlir::MLIRContext *ctx = builder.getContext();
  mlir::Type i32VecType =
      weftrvv::VectorType::get(ctx, builder.getI32Type(), "m1");
  mlir::Type f32M2VecType =
      weftrvv::VectorType::get(ctx, builder.getF32Type(), "m2");

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  loopState.addAttribute(
      "fold_model", builder.getStringAttr("super_block_two_level_scale_min"));
  // integer_core_lmul is LEFT OFF (emitter default mf2) -- byte-identical to the
  // untuned monolith export.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sums = body.addArgument(f32M2VecType, loc);
  mlir::Value sumf = body.addArgument(builder.getF32Type(), loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK 1: plain 4-bit nibble unpack -> aux8[256] scratch (Region A). Weight base
  // + block_index (per-super-block address vx + ib*144).
  {
    mlir::OperationState s(loc, weftrvv::Q4KNibbleUnpackOp::getOperationName());
    s.addOperands({weight, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_nibble_unpack"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    // q5_K ONLY: stamp the qh 5th-bit plane offset so the emitter injects the 5th
    // bit (cx.hasQh). Absent for q4_K -> byte-identical q4_K unpack.
    if (weightQhOffset)
      s.addAttribute("weight_qh_byte_offset",
                     builder.getI64IntegerAttr(*weightQhOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 2: 6-bit scale/min bit-dance -> utmp[4] scratch (Region B).
  {
    mlir::OperationState s(loc,
                           weftrvv::Q4KScaleMinBitDanceOp::getOperationName());
    s.addOperands({weight, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_scale_min_bit_dance"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 3: per-sub-block uint6-scaled i32 dot + fold-back (Region C). The LIVE
  // operand is the q8 activation base (%vy); the aux8/scales scratch slots are
  // vestigial (emitter-owned), so they are wired to the weight base (%vx). q8 lives
  // at yb + activation_quant_byte_offset (4).
  {
    mlir::OperationState s(loc, weftrvv::Q4KScaledDotOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_scaled_dot"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // This closed constructor pins the byte-identical deterministic anchor.
    // A later formula may construct a different legal anchor, but emission never
    // interprets an absent field as mf2.
    s.addAttribute("integer_core_lmul", builder.getStringAttr("mf2"));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 4: MIN term (sumf -= dmin * sum(mins * bsums)) -- the SCALAR sumf chain.
  // The scales scratch slot is vestigial -> wired to the weight base (%vx).
  {
    mlir::OperationState s(loc, weftrvv::Q4KMinTermOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_min_term"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("num_sub_blocks", builder.getI64IntegerAttr(numSubBlocks));
    s.addAttribute("bsums_byte_offset", builder.getI64IntegerAttr(bsumsOffset));
    s.addAttribute("weight_dmin_byte_offset",
                   builder.getI64IntegerAttr(weightDminOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK 6: deferred positive fold (sums += d * (float)aux32) -- the 8-lane fp32
  // sums VECTOR chain. The aux32 scratch slot is vestigial -> wired to %vx.
  {
    mlir::OperationState s(loc,
                           weftrvv::Q4KSumsFoldScaleDOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_sums_fold_scale_d"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("num_sub_blocks", builder.getI64IntegerAttr(numSubBlocks));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    // A-line g-axis debake (路 B): stamp the canonical fp32 sums lane count as a
    // FORMAT-DEFINED descriptor fact (8), NOT a subBlock/2 derivation (byte-
    // INEXACT for q6_K). The EmitC fold reads coreOp.getNumLanes() fail-closed.
    s.addAttribute("num_lanes", builder.getI64IntegerAttr(8));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // The DUAL carried-out accumulators (sums vector + sumf scalar).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sums, sumf});
    (void)builder.create(s);
  }
}

// ---------------------------------------------------------------------------
// Typed SUPER-BLOCK SINGLE-accumulator block-dot loop-body construction
// (M-FLAT q6_K milestone-2).
//
// The q6_K sibling of createTypedSuperBlockBlockDotLoopChain. q6_K has NO
// per-block min, so it assembles the OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a SINGLE accumulator
// (the `sums` 8-lane fp32 vector -- no `sumf` scalar), with just TWO in-loop
// bricks inside the region: the q6_K INTEGER CORE
// (weft_rvv.q6_k_q8_k_aux32_partial: the 2-bit qh + 8-bit signed scale unpack into
// the per-super-block aux32[8]) followed by the REUSED no-min positive fold
// (weft_rvv.q4_k_sums_fold_scale_d: sums += fp16(x.d) * y.d * (float)aux32, the
// SAME positive fold q4_K/q5_K use, differing only in the d byte offset 208), then
// a SINGLE yield naming the `sums` vector. It replaces the ONE monolith
// weft_rvv.q6_k_q8_k_block_dot op. fold_model "scales_times_sumi" KEYS the
// single-accumulator arity (the loop op verifier rejects a sumf-carrying yield
// here). Each brick's per-super-block addressing keys off the loop induction
// variable (region arg 0), so the emit is operand-driven (anti-bypass).
//
// The bricks' aux32/output SCRATCH operands are function-scoped scratch the loop
// emitter DECLARES itself (int8_t aux8[256] / the aux32 accumulator vector); it
// never reads these operand slots in the loop form (it walks only the LIVE
// weight/activation bases + block_index). So the front door wires the vestigial
// scratch slots to the existing weight ABI base (%vx) rather than minting
// placeholder runtime_abi_values -- keeping the exported ggml vec_dot C signature
// the exact 4-role n/s/vx/vy list (byte-identical to the monolith). The aux32 op
// and fold brick verifiers relax their output/aux32 C-type check in the loop form
// (block_index present) since the scratch is emitter-owned there. The
// integer-core LMUL is left at the emitter default (mf2, no integer_core_lmul
// stamp), so the untuned construction lowers byte-identically to the untuned
// monolith.
void createTypedSuperBlockScalesTimesSumiLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block single-accum chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  16 (q6_K/q3_K)
  std::int64_t weightStride = factByName("weight_block_stride");        // 210/110
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 // 192/96
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      // 208/108
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  // q6_K and q3_K share this SINGLE-accumulator no-min chain (fold_model
  // "scales_times_sumi"); the ONLY difference is the integer-core BRICK and its
  // weight sub-plane offsets. q6_K reads the qh 5th/6th-bit plane @128; q3_K reads
  // the hmask high-bit plane @0 + the 2-bit qs plane @32. The reused positive fold
  // + single `sums` yield are IDENTICAL.
  const bool isQ3K = entry.opName == "weft_rvv.q3_k_q8_k_block_dot";
  std::int64_t weightQhOffset = isQ3K ? 0 : factByName("weight_qh_byte_offset");
  std::int64_t weightHmaskOffset =
      isQ3K ? factByName("weight_hmask_byte_offset") : 0;      //   0
  std::int64_t weightQsOffset =
      isQ3K ? factByName("weight_qs_byte_offset") : 0;         //  32

  mlir::MLIRContext *ctx = builder.getContext();
  mlir::Type i32VecType =
      weftrvv::VectorType::get(ctx, builder.getI32Type(), "m1");
  mlir::Type f32M2VecType =
      weftrvv::VectorType::get(ctx, builder.getF32Type(), "m2");

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scales_times_sumi" KEYS the SINGLE-accumulator arity (no min).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scales_times_sumi"));
  // integer_core_lmul is LEFT OFF (emitter default mf2) -- byte-identical to the
  // untuned monolith export.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sums = body.addArgument(f32M2VecType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the aux32 INTEGER CORE (the per-super-block aux32[8] integer state).
  // The LIVE operands are the weight base (%vx) + activation base (%vy) +
  // block_index; the output int32_t* scratch slot is emitter-owned (vestigial) ->
  // wired to %vx. Per-super-block address vx + ib*stride, vy + ib*292.
  // q6_K: the 2-bit qh + 8-bit signed scale unpack. q3_K: the 2-bit +
  // SUBTRACTIVE-hmask unpack + the SIGNED 6-bit scale dance (its OWN brick op).
  {
    mlir::OperationState s(
        loc, isQ3K ? weftrvv::GgmlBlockDotQ3KQ8KAux32Op::getOperationName()
                   : weftrvv::GgmlBlockDotQ6KQ8KAux32Op::getOperationName());
    s.addOperands({weight, activation, weight, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr(isQ3K ? "ggml_q3_k_q8_k_aux32_partial"
                                               : "ggml_q6_k_q8_k_aux32_partial"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(isQ3K
                                  ? "per-sub-block-int6-signed-scale-i32-domain"
                                  : "per-sub-block-int8-scale-i32-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    if (isQ3K) {
      s.addAttribute("weight_hmask_byte_offset",
                     builder.getI64IntegerAttr(weightHmaskOffset));
      s.addAttribute("weight_qs_byte_offset",
                     builder.getI64IntegerAttr(weightQsOffset));
    } else {
      s.addAttribute("weight_qh_byte_offset",
                     builder.getI64IntegerAttr(weightQhOffset));
    }
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // BRICK: the REUSED no-min positive fold (sums += fp16(x.d) * y.d *
  // (float)aux32) -- the 8-lane fp32 `sums` VECTOR chain, NO min term. The LIVE
  // operands are the weight base (%vx, the fp16 d) + activation base (%vy, the
  // fp32 y.d @0) + block_index; the aux32 scratch slot is emitter-owned
  // (vestigial) -> wired to %vx. weight_d_byte_offset flows from entry.facts
  // (block_q6_K d @208 / block_q3_K d @108); sub_block/num_sub_blocks carry the
  // fold's fixed 8-lane facts (32/8), which the 8-lane fp fold does not read.
  {
    mlir::OperationState s(loc,
                           weftrvv::Q4KSumsFoldScaleDOp::getOperationName());
    s.addOperands({weight, weight, activation, vl, sbIndex});
    s.addAttribute("kind", builder.getStringAttr("q4_k_sums_fold_scale_d"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(32));
    s.addAttribute("num_sub_blocks", builder.getI64IntegerAttr(8));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    // A-line g-axis debake (路 B): stamp the canonical fp32 sums lane count as a
    // FORMAT-DEFINED descriptor fact (8). The EmitC fold reads
    // coreOp.getNumLanes() fail-closed (no baked default).
    s.addAttribute("num_lanes", builder.getI64IntegerAttr(8));
    s.addTypes(i32VecType);
    (void)builder.create(s);
  }
  // The SINGLE carried-out accumulator (the `sums` vector ONLY -- no sumf).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sums});
    (void)builder.create(s);
  }
}

// ---------------------------------------------------------------------------
// Typed SUPER-BLOCK SCALAR-accumulator block-dot loop-body construction
// (M-FLAT q2_K milestone-2).
//
// The q2_K sibling of createTypedSuperBlockScalesTimesSumiLoopChain. q2_K HAS a
// per-block min (like q4_K/q5_K) but its whole fold is a SINGLE per-super-block
// SCALAR `sumf += dall*isum - dmin*summs` (isum/summs being the two SCALAR
// integer states of the q2_K integer core), NOT an 8-lane deferred vector. So it
// assembles the OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a SINGLE `sumf`
// SCALAR accumulator (region args (index, sumf:f32)), with just ONE in-loop brick
// inside the region: the q2_K INTEGER CORE
// (weft_rvv.q2_k_q8_k_integer_core: the 2-bit unpack + PLAIN uint4-nibble scale/
// min + per-sub-block scalar i32 dot producing the two scalar states isum +
// summs), then a SINGLE yield naming the `sumf` scalar. The scalar fold itself
// (dall*isum - dmin*summs, fp16 d@80/dmin@82) has NO separate fold brick -- its
// offsets are FIXED block_q2_K constants, so the SCALAR-accumulator lowering
// emitter inlines it. It replaces the ONE monolith weft_rvv.q2_k_q8_k_block_dot
// op. fold_model "scalar_scale_min" KEYS the scalar-accumulator arity (the loop
// op verifier rejects an 8-lane `sums` vector region here). The brick's
// per-super-block addressing keys off the loop induction variable (region arg 0),
// so the emit is operand-driven (anti-bypass).
//
// The integer-core LMUL is NOT stamped (q2_K carries no shape knob -- its dot is
// FIXED at e8m1/i16m2/i32m1 in the emitter, matching the untuned monolith), so
// the untuned construction lowers byte-identically to the untuned monolith.
void createTypedSuperBlockScalarScaleMinLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block scalar-accum chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  16 (q2_K)
  std::int64_t weightStride = factByName("weight_block_stride");        //  84
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //  16
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  std::int64_t activationBsumsOffset =
      factByName("activation_bsums_byte_offset");              // 260

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_scale_min" KEYS the SCALAR-accumulator arity (q2_K).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_scale_min"));
  // integer_core_lmul is LEFT OFF (q2_K carries no shape knob; the emitter's
  // fixed e8m1/i16m2/i32m1 dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the q2_K INTEGER CORE (the 2-bit unpack + PLAIN uint4-nibble scale/min
  // + per-sub-block scalar i32 dot). The LIVE operands are the weight base (%vx) +
  // activation base (%vy) + n + vl + block_index; it produces the two SCALAR i32
  // states isum + summs (NO output pointer -- q2_K's states are scalar registers,
  // unlike q6_K's aux32[8] memory state). Per-super-block address vx + ib*84,
  // vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotQ2KQ8KIntegerCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_q2_k_q8_k_integer_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-uint4-scale-i32-domain-min"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("activation_bsums_byte_offset",
                   builder.getI64IntegerAttr(activationBsumsOffset));
    s.addTypes({i32ScalarType, i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector, no second min-term operand).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq1_s sibling of createTypedSuperBlockScalarScaleMinLoopChain. iq1_s is a
// super-block quant whose per-sub-block decode is a codebook GRID GATHER
// (decode_model=lookup -- the 2048-entry TERNARY iq1s_grid + a vluxei16 gather),
// NOT an arithmetic bit-unpack, but its whole fold is a SINGLE per-super-block
// SCALAR `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)` -- the SAME scalar-
// accumulator arity as q2_K, with sumi (the qh-scaled POSITIVE ternary-grid dot) and
// sumi1 (the iq1_s DELTA-bsum integer sum) being the two SCALAR i32 states of the
// iq1_s grid core. So it assembles the OUTER nb = n/QK_K loop as ONE region-carrying
// weft_rvv.typed_super_block_block_dot_loop_body op carrying a SINGLE `sumf` SCALAR
// accumulator (region args (index, sumf:f32)), with just ONE in-loop brick inside
// the region: the iq1_s TERNARY-grid INTEGER CORE
// (weft_rvv.iq1_s_q8_k_grid_core: the 11-bit grid index build from qs+qh + the
// vluxei16 ternary-grid gather + the signed widening grid dot + the qh-encoded
// per-sub-block scale + the delta-bsum sum, producing the two scalar states sumi +
// sumi1), then a SINGLE yield naming the `sumf` scalar. The scalar delta fold itself
// (d*((float)sumi + 0.125f*(float)sumi1), fp16 x.d @0 / fp32 y.d @0) has NO separate
// fold brick -- its offsets are FIXED block_iq1_s / block_q8_K constants, so the
// SCALAR-accumulator GRID lowering emitter inlines it. It replaces the ONE monolith
// weft_rvv.iq1_s_q8_k_block_dot op (RETIRED the SAME action as the flip). fold_model
// "scalar_delta_grid" KEYS the scalar-accumulator arity AND the iq1_s grid emitter
// (the loop op verifier rejects an 8-lane `sums` vector region here; the emitter
// dispatch keys the grid gather off this fold_model). The brick's per-super-block
// addressing keys off the loop induction variable (region arg 0), so the emit is
// operand-driven (anti-bypass).
//
// The integer-core LMUL is NOT stamped (iq1_s carries no shape knob -- its grid dot
// is FIXED at the emitter's vluxei16/i8m2/i16m4/i32m1 anchor, matching the untuned
// monolith), so the untuned construction lowers byte-identically to the untuned
// monolith.
void createTypedSuperBlockScalarDeltaGridLoopChain(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable("typed super-block scalar-delta-grid chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq1_s)
  std::int64_t weightStride = factByName("weight_block_stride");        //  50
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  34
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4
  std::int64_t activationBsumsOffset =
      factByName("activation_bsums_byte_offset");              // 260

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the iq1_s
  // GRID emitter (vs q2_K's arithmetic "scalar_scale_min").
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq1_s carries no shape knob; the emitter's fixed
  // vluxei16/i8m2 grid dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq1_s TERNARY-grid INTEGER CORE (the 11-bit grid index build from
  // qs+qh + the vluxei16 ternary-grid gather + the signed widening grid dot + the
  // qh-encoded per-sub-block scale + the delta-bsum sum). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces the
  // two SCALAR i32 states sumi + sumi1 (NO output pointer -- iq1_s's states are scalar
  // registers, like q2_K's). Per-super-block address vx + ib*50, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ1SQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq1_s_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "per-sub-block-qh-scale-ternary-grid-codebook-delta-bsum-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("activation_bsums_byte_offset",
                   builder.getI64IntegerAttr(activationBsumsOffset));
    // A-line g-axis debake (路 B): stamp the iq1_s grid group count as a
    // FORMAT-DEFINED descriptor fact (4 grid groups per sub-block), so the EmitC
    // grid loop reads coreOp.getGroupsPerSub() instead of a baked literal.
    s.addAttribute("groups_per_sub", builder.getI64IntegerAttr(4));
    s.addTypes({i32ScalarType, i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector, no second delta-term operand; the scalar delta fold is
  // emitter-inlined).
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq1_m sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the C2
// marginal-cost payoff. iq1_m REUSES the WHOLE iq1_s super-block SCALAR-accumulator
// GRID scaffold (the SAME loop op weft_rvv.typed_super_block_block_dot_loop_body with
// fold_model "scalar_delta_grid", the SAME single `sumf` scalar accumulator + region
// (index, sumf:f32) contract, the SAME selector SuperBlockScalarDeltaGrid, the SAME
// emitter dispatch, the SAME single-yield): the ONLY marginal cost is a DISTINCT
// in-loop brick -- the iq1_m TERNARY-grid integer core
// (weft_rvv.iq1_m_q8_k_grid_core, producing the two scalar states sumi1 (the qh-index
// half-scaled grid dot) + sumi2 (the per-group four-sign delta sum)) -- because
// iq1_m's integer decode is structurally different (a packed-scale fp16 reconstruct,
// TWO per-sub-block half scales ls1/ls2, a half-split per-half grid dot, a per-group
// FRESH Σq8 delta with FOUR independent signs, and NO bsums). The scalar delta fold
// itself (d*((float)sumi1 + IQ1M_DELTA*(float)sumi2), IQ1M_DELTA=0.125f, the packed
// iq1m_scale fp16 reconstruct + fp32 y.d @0) has NO separate fold brick -- the
// SCALAR-accumulator GRID lowering emitter inlines it, keyed off the iq1_m brick
// identity. It resolves to iq1_m's OWN export entry by fold_model + weight_block_stride
// 56 (vs iq1_s 50). The brick's per-super-block addressing keys off the loop induction
// variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq1M(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq1_m chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq1_m)
  std::int64_t weightStride = factByName("weight_block_stride");        //  56
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   0
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  32
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 //  48
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid
  // emitter; weight_block_stride 56 disambiguates iq1_m from iq1_s (50).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq1_m carries no shape knob; the emitter's fixed
  // vluxei16/i8m1/i16m2 grid dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq1_m TERNARY-grid INTEGER CORE (the packed iq1m_scale fp16 reconstruct
  // + the per-half vluxei16 grid dot with two half scales ls1/ls2 + the per-group
  // four-sign delta via a fresh Σq8). The LIVE operands are the weight base (%vx) +
  // activation base (%vy) + n + vl + block_index; it produces the two SCALAR i32
  // states sumi1 + sumi2 (NO output pointer -- iq1_m's states are scalar registers,
  // like iq1_s's). Per-super-block address vx + ib*56, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ1MQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq1_m_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "packed-iq1m-scale-per-half-scale-ternary-grid-codebook-per-group-"
            "delta-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq1_m grid group count as a
    // FORMAT-DEFINED descriptor fact (4 grid groups per sub-block).
    s.addAttribute("groups_per_sub", builder.getI64IntegerAttr(4));
    s.addTypes({i32ScalarType, i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector, no second delta-term operand; the scalar delta fold is
  // emitter-inlined). IDENTICAL to iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq3_xxs sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3
// coverage payoff. iq3_xxs is a super-block GRID/codebook quant whose whole fold is
// the SAME SINGLE per-super-block SCALAR accumulator arity as iq1_s (fold_model
// "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the whole
// iq1_s super-block SCALAR-accumulator GRID scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the
// SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY
// marginal cost is a DISTINCT in-loop brick -- the iq3_xxs GRID-of-4 integer core
// (weft_rvv.iq3_xxs_q8_k_grid_core, producing the ONE scalar state bsum) -- because
// iq3_xxs's decode is structurally different from iq1: the 256-entry uint32
// iq3xxs_grid GRID-of-4 gathered via vluxei16_v_i32m1 (vs iq1_s's uint64 grid-of-8),
// the per-sign-group ksigns_iq2xs SIGN plane (per-group 7-bit selectors), the
// per-sub-block aux32 4-bit scale ls, and the trailing 0.25f factor. The scalar fold
// itself (`sumf += d*(float)bsum` then `*s = 0.25f*sumf`, fp16 x.d @0 / fp32 y.d @0)
// has NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter inlines
// it, keyed off the iq3_xxs brick identity. It resolves to iq3_xxs's OWN export entry
// by fold_model + weight_block_stride 98 (vs iq1_s 50, iq1_m 56). The brick's
// per-super-block addressing keys off the loop induction variable (region arg 0), so
// the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq3_xxs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq3_xxs)
  std::int64_t weightStride = factByName("weight_block_stride");        //  98
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightGasOffset = factByName("weight_gas_byte_offset");  //  66
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid
  // emitter; weight_block_stride 98 disambiguates iq3_xxs from iq1_s (50)/iq1_m (56).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq3_xxs carries no shape knob; the emitter's fixed
  // vluxei16/i8m1/i16m2 grid-of-4 dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq3_xxs GRID-of-4 INTEGER CORE (the aux32 4-bit-scale + 4-sign-group
  // ksigns decode + the two-index-per-group vluxei16_v_i32m1 grid gather + the signed
  // widening grid dot + the per-sub-block scale fold into bsum). The LIVE operands are
  // the weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces
  // the ONE SCALAR i32 state bsum (NO output pointer -- iq3_xxs's bsum is a scalar
  // register, like iq1_s's states). Per-super-block address vx + ib*98, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ3XXSQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq3_xxs_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-aux32-scale-grid-of-4-codebook-"
                              "ksigns-sign-plane-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_gas_byte_offset",
                   builder.getI64IntegerAttr(weightGasOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq3_xxs grid sub-structure counts as
    // FORMAT-DEFINED descriptor facts (4 sign groups per sub-block, 8 grid index
    // bytes per sub-block, 8 grid lanes per sign group).
    s.addAttribute("num_groups", builder.getI64IntegerAttr(4));
    s.addAttribute("indices_per_sub_block", builder.getI64IntegerAttr(8));
    s.addAttribute("group_lanes", builder.getI64IntegerAttr(8));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector; the scalar fold + trailing 0.25f are emitter-inlined). IDENTICAL to
  // iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq3_s sibling of createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs -- the
// C_construct 22->23 payoff (EXPLICIT-SIGNS variant). iq3_s is the iq3_xxs GRID-of-4
// sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
// per-super-block SCALAR accumulator arity as iq1_s/iq3_xxs (fold_model
// "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the whole
// iq1_s super-block SCALAR-accumulator GRID scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the SAME
// selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost
// is a DISTINCT in-loop brick -- the iq3_s GRID-of-4 explicit-signs integer core
// (weft_rvv.iq3_s_q8_k_grid_core, producing the ONE scalar state bsum) -- because iq3_s's
// decode swaps THREE mechanisms from iq3_xxs to iq2_s: the 512-entry uint32 iq3s_grid
// GRID-of-4 gathered via vluxei16_v_i32m1 (a LARGER 9-bit-index table than iq3_xxs's
// 256-entry 8-bit one), the qh 9th-bit inject (mask 256, the two passes taking shifts
// 8-2l and 7-2l), the per-lane sign read from an EXPLICIT per-sub-block signs region (at
// weight offset 74, NO ksigns plane), and the per-sub-block scale from the EXPLICIT
// two-nibble scales[] (at weight offset 106). The scalar fold itself (`sumf +=
// d*(float)bsum` then `*s = sumf` -- NO trailing factor -- fp16 x.d @0 / fp32 y.d @0) has
// NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter inlines it, keyed
// off the iq3_s brick identity. It resolves to iq3_s's OWN export entry by fold_model +
// weight_block_stride 110 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s
// 82). The brick's per-super-block addressing keys off the loop induction variable (region
// arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq3s(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq3_s chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq3_s)
  std::int64_t weightStride = factByName("weight_block_stride");        // 110
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  66
  std::int64_t weightSignsOffset =
      factByName("weight_signs_byte_offset");                  //  74
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                 // 106
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 110 disambiguates iq3_s from iq1_s (50)/iq1_m (56)/iq3_xxs (98)/
  // iq2_xxs (66)/iq2_xs (74)/iq2_s (82).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq3_s carries no shape knob; the emitter's fixed
  // vluxei16/i8m1/i16m2 grid-of-4 dot matches the untuned monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq3_s GRID-of-4 EXPLICIT-SIGNS INTEGER CORE (the explicit two-nibble scale
  // + qh 9th-bit inject + explicit per-sub-block signs decode + the two-index-per-group
  // vluxei16_v_i32m1 grid gather + the signed widening grid dot + the per-sub-block scale
  // fold into bsum). The LIVE operands are the weight base (%vx) + activation base (%vy) +
  // n + vl + block_index; it produces the ONE SCALAR i32 state bsum (NO output pointer --
  // iq3_s's bsum is a scalar register, like iq3_xxs's states). Per-super-block address vx +
  // ib*110, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ3SQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq3_s_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-explicit-scale-grid-of-4-codebook-"
                              "qh-plane-explicit-signs-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_signs_byte_offset",
                   builder.getI64IntegerAttr(weightSignsOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq3_s grid sub-structure counts as
    // FORMAT-DEFINED descriptor facts (4 sign groups per sub-block, 8 grid index
    // bytes per sub-block, 4 explicit sign bytes per sub-block, 8 grid lanes per
    // sign group).
    s.addAttribute("num_groups", builder.getI64IntegerAttr(4));
    s.addAttribute("indices_per_sub_block", builder.getI64IntegerAttr(8));
    s.addAttribute("signs_per_sub_block", builder.getI64IntegerAttr(4));
    s.addAttribute("group_lanes", builder.getI64IntegerAttr(8));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane
  // `sums` vector; the scalar fold is emitter-inlined, and iq3_s applies NO trailing
  // factor). IDENTICAL to iq1_s/iq3_xxs's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq4_xs sibling of the scalar-delta-grid chain builders -- the C_construct 23->24
// payoff (the FIRST super-block CODEBOOK member vs the grid siblings). iq4_xs is the
// SUPER-BLOCK rung of the flat iq4_nl codebook -- a super-block CODEBOOK quant whose whole
// fold is the SAME SINGLE per-super-block SCALAR accumulator arity as iq1_s/iq3_s
// (fold_model "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the
// whole iq1_s super-block SCALAR-accumulator scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the SAME
// selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost is
// a DISTINCT in-loop brick -- the iq4_xs CODEBOOK integer core
// (weft_rvv.iq4_xs_q8_k_codebook_core) -- because iq4_xs's decode swaps the grid gather for
// iq4_nl's 16-entry non-linear int8 CODEBOOK gather (the SAME kvalues_iq4nl[16] vrgather
// lookup iq4_nl uses, carried as a DenseI8ArrayAttr:$codebook) wrapped in the q4_K-style
// super-block SIGNED 6-bit scale bit-dance (per-sub-block ls = ((scales_l>>...)&0xf) |
// (((scales_h>>...)&0x3)<<4) biased -32). UNLIKE the grid siblings iq4_xs's fold runs
// PER-SUB-BLOCK in float (`sumf += (d4d8*(ls-32))*sumi`, 8 fp folds per super-block, NO
// trailing factor), but the single-scalar accumulator arity is identical, so the whole
// per-super-block body is emitter-inlined keyed off the codebook-core brick identity. It
// resolves to iq4_xs's OWN export entry by fold_model + weight_block_stride 136 (vs iq1_s
// 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s 82, iq3_s 110). The brick carries
// NO gearbox (the codebook gather pins m1). The brick's per-super-block addressing keys off
// the loop induction variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq4xs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq4_xs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32
  std::int64_t weightStride = factByName("weight_block_stride");        // 136
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightScalesHOffset =
      factByName("weight_scales_h_byte_offset");                //   2
  std::int64_t weightScalesLOffset =
      factByName("weight_scales_l_byte_offset");                //   4
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   8
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                   //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   4

  mlir::MLIRContext *ctx = builder.getContext();
  (void)ctx;
  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid/codebook
  // emitter dispatch; weight_block_stride 136 disambiguates iq4_xs from iq1_s (50)/iq1_m
  // (56)/iq3_xxs (98)/iq2_xxs (66)/iq2_xs (74)/iq2_s (82)/iq3_s (110).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF (iq4_xs carries no shape knob; the codebook gather pins
  // m1, and the emitter's fixed vrgather/i8m1/i16m2 codebook dot matches the untuned
  // monolith byte-identically).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq4_xs CODEBOOK INTEGER CORE (the 16-entry non-linear int8 codebook
  // broadcast + the per-sub-block nibble split + vrgather decode + asymmetric vwmul/vwmacc
  // widening product + seed-0 vwredsum, wrapped in the q4_K-style super-block signed 6-bit
  // scale bit-dance + the per-sub-block float fold into sumf). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces ONE scalar
  // i32 SSA result (an UNUSED per-super-block placeholder -- iq4_xs's fold is per-sub-block
  // float, no single scalar state). Per-super-block address vx + ib*136, vy + ib*292. The
  // 16-entry codebook (kvalues_iq4nl[16]) is carried as a DenseI8ArrayAttr like the monolith.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ4XSQ8KCodebookCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq4_xs_q8_k_codebook_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "per-sub-block-signed-6bit-scale-codebook-gather-float-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_scales_h_byte_offset",
                   builder.getI64IntegerAttr(weightScalesHOffset));
    s.addAttribute("weight_scales_l_byte_offset",
                   builder.getI64IntegerAttr(weightScalesLOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addAttribute("codebook", builder.getDenseI8ArrayAttr(entry.codebook));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the per-sub-block float fold is emitter-inlined, and iq4_xs applies NO trailing
  // factor). IDENTICAL to iq1_s/iq3_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The tq2_0 sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the FIRST TQ-family
// member (C_construct 24->25). tq2_0 is the 2-bit TERNARY ({-1,0,+1}) TriLM K-quant whose
// whole fold is the SAME SINGLE per-super-block SCALAR accumulator arity as iq1_s/iq3_s
// (fold_model "scalar_delta_grid", single `sumf` scalar, emitter-inlined fold), REUSING the
// whole iq1_s super-block SCALAR-accumulator scaffold (the SAME loop op
// weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield contract, the SAME
// selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost is a
// DISTINCT in-loop brick -- the tq2_0 FUSED 2-bit TERNARY integer core
// (weft_rvv.tq2_0_q8_k_ternary_core) -- because tq2_0's decode is ARITHMETIC (q2_K's 2-bit
// `(qs>>shift)&3` unpack + the per-element `-1` ternary bias, NO grid/codebook gather). Its
// whole fold is a single per-super-block scalar `sumf += (float)sumi * d` with `d = fp16(x.d
// @64) * y.d @0` and NO trailing factor, so the whole per-super-block body is emitter-inlined
// keyed off the ternary-core brick identity. It shares weight_block_stride 66 with iq2_xxs but
// dispatches by its OWN DISTINCT brick op type. UNLIKE the iq4_xs codebook sibling the brick
// PRESERVES tq2_0's Win-A integer_core_lmul m2/m1 gearbox (kernel key "tq2_0", left attr-less
// at construction = the default m2 anchor). The brick's per-super-block addressing keys off
// the loop induction variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainTq20(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid tq2_0 chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t weightStride = factByName("weight_block_stride");        //  66
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   0
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //  64
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                   //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   4

  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the emitter dispatch;
  // the emitter disambiguates tq2_0 from the iq* siblings (and from iq2_xxs which shares
  // stride 66) by the in-region ternary-core brick op TYPE.
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF here (attr-less construction = the default m2 anchor, the
  // byte-exact CORE target); tq2_0's Win-A gearbox lives on the ternary-core brick below and
  // is refined m2->m1 at VLEN>=256 by the separate schedule pass.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the tq2_0 FUSED 2-bit TERNARY INTEGER CORE (the 32-byte qs chunk load + the 4
  // 2-bit planes each unpacked to 32 ternary lanes via vand/vsrl + the `-1` bias vsub and
  // vwmacc'd DIRECTLY against the matching 32 q8 lanes into a wide i16 accumulator + ONE
  // vwredsum per chunk into the per-super-block scalar sumi). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces ONE scalar
  // i32 SSA result (the per-super-block sumi placeholder -- the emitter re-emits the whole
  // body including the fold). Per-super-block address vx + ib*66, vy + ib*292. Left attr-less
  // so the gearbox is free to stamp integer_core_lmul m2/m1 from the VLEN capability fact.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotTQ20Q8KTernaryCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_tq2_0_q8_k_ternary_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr(
            "ternary-2bit-fused-plane-single-fp16-scale-i32-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the scalar fold `sumf += (float)sumi * d` is emitter-inlined, and tq2_0 applies
  // NO trailing factor). IDENTICAL to iq1_s/iq3_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The tq1_0 sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the SECOND TQ-family
// member (C_construct 25->26), the base-3-packed sibling of tq2_0. tq1_0 is the BASE-3
// TERNARY ({-1,0,+1}) TriLM K-quant whose whole fold is the SAME SINGLE per-super-block
// SCALAR accumulator arity as tq2_0/iq1_s (fold_model "scalar_delta_grid", single `sumf`
// scalar, emitter-inlined fold), REUSING the WHOLE tq2_0 ternary scaffold at C2 marginal cost
// (the SAME loop op weft_rvv.typed_super_block_block_dot_loop_body, the SAME single-yield
// contract, the SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY
// marginal cost is a DISTINCT in-loop brick -- the tq1_0 BASE-3 TERNARY integer core
// (weft_rvv.tq1_0_q8_k_ternary_core) -- because tq1_0's decode is base-3 (the `q =
// (uint8_t)(byte*pow3[l]); xi = ((uint16_t)q*3)>>8; xi-1` power-of-three trit unpack over the
// qs+qh weight arrays, NOT tq2_0's `(qs>>shift)&3` 2-bit field). Its whole fold is a single
// per-super-block scalar `sumf += (float)sumi * d` with `d = fp16(x.d @52) * y.d @0` and NO
// trailing factor, so the whole per-super-block body is emitter-inlined keyed off the
// ternary-core brick identity. It resolves to its OWN export entry by fold_model +
// weight_block_stride 54 (UNIQUE among the scalar_delta_grid bricks -- tq2_0/iq2_xxs are 66),
// so NO stride tie-breaker is needed. Unlike tq2_0, the realized tq1_0 brick is
// VLEN-universal and has no inert LMUL schedule field. The brick's per-super-block addressing keys off the loop induction
// variable (region arg 0), so the emit is operand-driven (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainTq10(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid tq1_0 chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t weightStride = factByName("weight_block_stride");        //  54
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   0
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  48
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //  52
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                   //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");               //   4

  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the emitter dispatch;
  // the emitter disambiguates tq1_0 from the iq* / tq2_0 siblings by the in-region base-3
  // ternary-core brick op TYPE (and tq1_0's stride 54 is UNIQUE among these bricks).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // The tq1_0 ternary-core brick below has one fixed realized vector body.
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the tq1_0 BASE-3 TERNARY INTEGER CORE (the three base-3 unpack regions -- qs main
  // + qs tail + qh -- each `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` decoded
  // into an element-ordered aux8[256], then the flat-256 widened i8*i8 dot into the
  // per-super-block scalar sumi). The LIVE operands are the weight base (%vx) + activation
  // base (%vy) + n + vl + block_index; it produces ONE scalar i32 SSA result (the
  // per-super-block sumi placeholder -- the emitter re-emits the whole body including the
  // fold). Per-super-block address vx + ib*54, vy + ib*292. Left attr-less so the gearbox is
  // free to stamp integer_core_lmul m2/m1 from the VLEN capability fact.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotTQ10Q8KTernaryCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_tq1_0_q8_k_ternary_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("ternary-base3-single-fp16-scale-i32-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the scalar fold `sumf += (float)sumi * d` is emitter-inlined, and tq1_0 applies
  // NO trailing factor). IDENTICAL to tq2_0/iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq2_xxs sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3 coverage
// payoff (SIGN-PLANE signs64 variant). iq2_xxs is another iq1_s grid sibling -- a
// super-block GRID/codebook quant whose whole fold is the SAME SINGLE per-super-block
// SCALAR accumulator arity as iq1_s (fold_model "scalar_delta_grid", single `sumf`
// scalar, emitter-inlined fold), REUSING the whole iq1_s super-block SCALAR-accumulator
// GRID scaffold (the SAME loop op, the SAME single-yield contract, the SAME selector
// SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal cost is a
// DISTINCT in-loop brick -- the iq2_xxs GRID-of-8 integer core
// (weft_rvv.iq2_xxs_q8_k_grid_core, producing the ONE scalar state bsum) -- because
// iq2_xxs's decode is structurally different: the 256-entry uint64 iq2xxs_grid GRID-of-8
// gathered via vluxei16_v_i64<core> (vs iq3_xxs's uint32 grid-of-4), the SIGN read via a
// SECOND vluxei16 gather over the DERIVED keven_signs_q2xs signs64 sign plane (vs
// iq3_xxs's per-group scalar ksigns fold), the per-sub-block aux1 4-bit scale ls, and the
// trailing 0.125f factor. The scalar fold itself (`sumf += d*(float)bsum` then `*s =
// 0.125f*sumf`, fp16 x.d @0 / fp32 y.d @0) has NO separate fold brick -- the
// SCALAR-accumulator GRID lowering emitter inlines it, keyed off the iq2_xxs brick
// identity. It resolves to iq2_xxs's OWN export entry by fold_model + weight_block_stride
// 66 (vs iq1_s 50, iq1_m 56, iq3_xxs 98). The brick carries the SAME Win-A
// integer_core_lmul gearbox (kernel key "iq2_xxs") so the m2->m1 VLEN capability
// selection is preserved on the constructed op. The brick's per-super-block addressing
// keys off the loop induction variable (region arg 0), so the emit is operand-driven
// (anti-bypass).
void createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq2_xxs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq2_xxs)
  std::int64_t weightStride = factByName("weight_block_stride");        //  66
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 66 disambiguates iq2_xxs from iq1_s (50)/iq1_m (56)/iq3_xxs (98).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  // integer_core_lmul is LEFT OFF here (the constructed brick lowers at the emitter's m2
  // default = the retired monolith's byte-exact default; the unified autotuner REFINES
  // m2->m1 at VLEN256 by stamping the brick -- the Win-A gearbox, preserved).
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq2_xxs GRID-of-8 INTEGER CORE (the aux1 4-bit-scale + 4-sign-group decode
  // + the 4-index vluxei16_v_i64<core> grid gather + the SECOND signs64 vluxei16 gather +
  // the vmul-onto-grid sign fold + the signed widening grid dot + the per-sub-block scale
  // fold into bsum). The LIVE operands are the weight base (%vx) + activation base (%vy) +
  // n + vl + block_index; it produces the ONE SCALAR i32 state bsum (NO output pointer --
  // iq2_xxs's bsum is a scalar register). Per-super-block address vx + ib*66, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ2XXSQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq2_xxs_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-sub-block-aux1-scale-grid-of-8-codebook-"
                              "signs64-sign-plane-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq2_xxs grid sign-group count as a
    // FORMAT-DEFINED descriptor fact (4 sign groups per sub-block).
    s.addAttribute("num_groups", builder.getI64IntegerAttr(4));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY -- no 8-lane `sums`
  // vector; the scalar fold + trailing 0.125f are emitter-inlined). IDENTICAL to iq1_s's
  // yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq2_xs sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3 coverage
// payoff (SIGN-PLANE signs64 variant, PER-HALF explicit scale). iq2_xs is the iq2_xxs grid
// sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
// per-super-block SCALAR accumulator arity as iq1_s (fold_model "scalar_delta_grid", single
// `sumf` scalar, emitter-inlined fold), REUSING the whole iq1_s super-block
// SCALAR-accumulator GRID scaffold (the SAME loop op, the SAME single-yield contract, the
// SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal
// cost is a DISTINCT in-loop brick -- the iq2_xs per-half-explicit-scale GRID integer core
// (weft_rvv.iq2_xs_q8_k_grid_core, producing the ONE scalar state bsum) -- because iq2_xs's
// decode is structurally different from iq2_xxs: the 512-entry uint64 iq2xs_grid indexed by
// the 9-bit `w & 511` of each uint16 qs word (vs iq2_xxs's 256-entry aux1-interleaved
// grid), the SIGN read via a SECOND vluxei16 gather over the DERIVED keven_signs_q2xs
// signs64 sign plane keyed by the 7-bit `w >> 9` selector, and -- the load-bearing delta --
// the EXPLICIT per-sub-block 4-bit scales[8] byte stream splitting each sub-block into TWO
// 16-lane HALVES with distinct scales ls1/ls2 (vs iq2_xxs's single aux1 scale). The scalar
// fold itself (`sumf += d*(float)bsum` then `*s = 0.125f*sumf`, fp16 x.d @0 / fp32 y.d @0)
// has NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter inlines it,
// keyed off the iq2_xs brick identity. It resolves to iq2_xs's OWN export entry by
// fold_model + weight_block_stride 74 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66). The
// brick's per-super-block addressing keys off the loop induction variable (region arg 0),
// so the emit is operand-driven (anti-bypass). UNLIKE iq2_xxs the brick carries NO
// integer_core_lmul gearbox (fixed 16-lane per-half shape, not in any autotuner).
void createTypedSuperBlockScalarDeltaGridLoopChainIq2xs(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq2_xs chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq2_xs)
  std::int64_t weightStride = factByName("weight_block_stride");        //  74
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                  //  66
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 74 disambiguates iq2_xs from iq1_s (50)/iq1_m (56)/iq3_xxs (98)/
  // iq2_xxs (66).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq2_xs per-half-explicit-scale GRID INTEGER CORE (the per-sub-block explicit
  // 4-bit scale byte -> ls1/ls2 + the two-half 16-lane decode + the 2-index vluxei16_v_i64m1
  // grid gather + the SECOND signs64 vluxei16 gather + the vmul-onto-grid sign fold + the
  // signed widening grid dot + the per-half scale fold into bsum). The LIVE operands are the
  // weight base (%vx) + activation base (%vy) + n + vl + block_index; it produces the ONE
  // SCALAR i32 state bsum (NO output pointer -- iq2_xs's bsum is a scalar register).
  // Per-super-block address vx + ib*74, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ2XSQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq2_xs_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-half-int4-explicit-scales-grid-codebook-"
                              "signs64-sign-plane-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq2_xs per-16-lane-half group count
    // as a FORMAT-DEFINED descriptor fact (2 grid/sign groups per 16-lane half).
    s.addAttribute("num_groups_per_half", builder.getI64IntegerAttr(2));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY). IDENTICAL to
  // iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
}

// The iq2_s sibling of createTypedSuperBlockScalarDeltaGridLoopChain -- the L3 coverage
// payoff (SIGN-PLANE explicit-signs variant, PER-HALF explicit scale). iq2_s is the iq2_xs
// grid sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
// per-super-block SCALAR accumulator arity as iq1_s (fold_model "scalar_delta_grid", single
// `sumf` scalar, emitter-inlined fold), REUSING the whole iq1_s super-block
// SCALAR-accumulator GRID scaffold (the SAME loop op, the SAME single-yield contract, the
// SAME selector SuperBlockScalarDeltaGrid, the SAME emitter dispatch). The ONLY marginal
// cost is a DISTINCT in-loop brick -- the iq2_s per-half-explicit-scale GRID integer core
// (weft_rvv.iq2_s_q8_k_grid_core, producing the ONE scalar state bsum) -- because iq2_s's
// decode is structurally different from iq2_xs: the 1024-entry uint64 iq2s_grid indexed by
// the 10-bit `qs[l] | ((qh<<(8-2l))&0x300)` (a single index byte + 2 qh-plane bits, vs
// iq2_xs's 9-bit `w & 511` of a uint16 qs word), the SIGN read via a SECOND vluxei16 gather
// over the UNIVERSAL signs256 explicit-sign-byte plane keyed by the RAW sign byte read
// DIRECTLY from the sign region at qs+32 (vs iq2_xs's DERIVED keven_signs_q2xs plane keyed
// by `w >> 9`), and -- shared with iq2_xs -- the EXPLICIT per-sub-block 4-bit scales[8]
// byte stream splitting each sub-block into TWO 16-lane HALVES with distinct scales ls1/ls2.
// The scalar fold itself (`sumf += d*(float)bsum` then `*s = 0.125f*sumf`, fp16 x.d @0 /
// fp32 y.d @0) has NO separate fold brick -- the SCALAR-accumulator GRID lowering emitter
// inlines it, keyed off the iq2_s brick identity. It resolves to iq2_s's OWN export entry by
// fold_model + weight_block_stride 82 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs
// 74). The brick's per-super-block addressing keys off the loop induction variable (region
// arg 0), so the emit is operand-driven (anti-bypass). Like iq2_xs the brick carries NO
// integer_core_lmul gearbox (fixed 16-lane per-half shape, not in any autotuner).
void createTypedSuperBlockScalarDeltaGridLoopChainIq2s(
    mlir::OpBuilder &builder, mlir::Location loc,
    const MonolithicBlockDotOpEntry &entry, mlir::Value weight,
    mlir::Value activation, mlir::Value out, mlir::Value n, mlir::Value vl) {
  auto factByName = [&](llvm::StringRef name) -> std::int64_t {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    llvm_unreachable(
        "typed super-block scalar-delta-grid iq2_s chain: missing fact");
  };
  std::int64_t qk = factByName("qk");                          // 256 (QK_K)
  std::int64_t subBlock = factByName("sub_block");             //  32 (iq2_s)
  std::int64_t weightStride = factByName("weight_block_stride");        //  82
  std::int64_t activationStride = factByName("activation_block_stride"); // 292
  std::int64_t weightDOffset = factByName("weight_d_byte_offset");      //   0
  std::int64_t weightQsOffset = factByName("weight_qs_byte_offset");    //   2
  std::int64_t weightSignsOffset =
      factByName("weight_signs_byte_offset");                  //  34
  std::int64_t weightQhOffset = factByName("weight_qh_byte_offset");    //  66
  std::int64_t weightScalesOffset =
      factByName("weight_scales_byte_offset");                  //  74
  std::int64_t activationDOffset =
      factByName("activation_d_byte_offset");                  //   0
  std::int64_t activationQuantOffset =
      factByName("activation_quant_byte_offset");              //   4

  mlir::Type i32ScalarType = builder.getI32Type();
  mlir::Type f32ScalarType = builder.getF32Type();

  mlir::OperationState loopState(
      loc, weftrvv::TypedSuperBlockBlockDotLoopBodyOp::getOperationName());
  loopState.addOperands({weight, activation, out, n});
  loopState.addAttribute(
      "kind", builder.getStringAttr("typed_super_block_block_dot_loop_body"));
  loopState.addAttribute("qk", builder.getI64IntegerAttr(qk));
  loopState.addAttribute("weight_block_stride",
                         builder.getI64IntegerAttr(weightStride));
  loopState.addAttribute("activation_block_stride",
                         builder.getI64IntegerAttr(activationStride));
  // fold_model "scalar_delta_grid" KEYS the SCALAR-accumulator arity + the grid emitter;
  // weight_block_stride 82 disambiguates iq2_s from iq1_s (50)/iq1_m (56)/iq3_xxs (98)/
  // iq2_xxs (66)/iq2_xs (74).
  loopState.addAttribute("fold_model",
                         builder.getStringAttr("scalar_delta_grid"));
  loopState.addRegion();
  auto loop = llvm::cast<weftrvv::TypedSuperBlockBlockDotLoopBodyOp>(
      builder.create(loopState));

  mlir::Block &body = loop.getBody().emplaceBlock();
  mlir::Value sbIndex = body.addArgument(builder.getIndexType(), loc);
  mlir::Value sumf = body.addArgument(f32ScalarType, loc);

  mlir::OpBuilder::InsertionGuard bodyGuard(builder);
  builder.setInsertionPointToStart(&body);

  // BRICK: the iq2_s per-half-explicit-scale GRID INTEGER CORE (the per-sub-block explicit
  // 4-bit scale byte -> ls1/ls2 + the qh-plane byte + the two-half 16-lane decode + the
  // 2-index vluxei16_v_i64m1 grid gather + the SECOND signs256 vluxei16 gather + the
  // vmul-onto-grid sign fold + the signed widening grid dot + the per-half scale fold into
  // bsum). The LIVE operands are the weight base (%vx) + activation base (%vy) + n + vl +
  // block_index; it produces the ONE SCALAR i32 state bsum (NO output pointer -- iq2_s's
  // bsum is a scalar register). Per-super-block address vx + ib*82, vy + ib*292.
  {
    mlir::OperationState s(
        loc, weftrvv::GgmlBlockDotIQ2SQ8KGridCoreOp::getOperationName());
    s.addOperands({weight, activation, n, vl, sbIndex});
    s.addAttribute("kind",
                   builder.getStringAttr("ggml_iq2_s_q8_k_grid_core"));
    s.addAttribute(
        "scale_model",
        builder.getStringAttr("per-half-int4-explicit-scales-grid-codebook-qh-"
                              "plane-explicit-signs-int-domain"));
    s.addAttribute("qk", builder.getI64IntegerAttr(qk));
    s.addAttribute("sub_block", builder.getI64IntegerAttr(subBlock));
    s.addAttribute("weight_block_stride",
                   builder.getI64IntegerAttr(weightStride));
    s.addAttribute("activation_block_stride",
                   builder.getI64IntegerAttr(activationStride));
    s.addAttribute("weight_d_byte_offset",
                   builder.getI64IntegerAttr(weightDOffset));
    s.addAttribute("weight_qs_byte_offset",
                   builder.getI64IntegerAttr(weightQsOffset));
    s.addAttribute("weight_signs_byte_offset",
                   builder.getI64IntegerAttr(weightSignsOffset));
    s.addAttribute("weight_qh_byte_offset",
                   builder.getI64IntegerAttr(weightQhOffset));
    s.addAttribute("weight_scales_byte_offset",
                   builder.getI64IntegerAttr(weightScalesOffset));
    s.addAttribute("activation_d_byte_offset",
                   builder.getI64IntegerAttr(activationDOffset));
    s.addAttribute("activation_quant_byte_offset",
                   builder.getI64IntegerAttr(activationQuantOffset));
    // A-line g-axis debake (路 B): stamp the iq2_s grid group counts as
    // FORMAT-DEFINED descriptor facts (4 grid groups per sub-block, 2 grid/sign
    // groups per 16-lane half).
    s.addAttribute("groups_per_sub", builder.getI64IntegerAttr(4));
    s.addAttribute("num_groups_per_half", builder.getI64IntegerAttr(2));
    s.addTypes({i32ScalarType});
    (void)builder.create(s);
  }
  // The SINGLE carried-out SCALAR accumulator (the `sumf` scalar ONLY). IDENTICAL to
  // iq1_s's yield -- the shared scaffold.
  {
    mlir::OperationState s(
        loc, weftrvv::TypedSuperBlockBlockDotLoopYieldOp::getOperationName());
    s.addOperands({sumf});
    (void)builder.create(s);
  }
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
  state.addTypes(weftrvv::VectorType::get(builder.getContext(),
                                          builder.getI32Type(), "m1"));
  return builder.create(state)->getResult(0);
}

weftexec::VariantOp createVariant(mlir::OpBuilder &builder, mlir::Location loc,
                                  llvm::StringRef selectedVariantSymbol,
                                  mlir::ArrayAttr requires,
                                  weftrvv::PolicyAttr policy) {
  mlir::OperationState state(loc, weftexec::VariantOp::getOperationName());
  state.addAttribute("sym_name", builder.getStringAttr(selectedVariantSymbol));
  state.addAttribute(kOriginAttrName,
                     builder.getStringAttr(getRVVExtensionPluginName()));
  state.addAttribute(kRequiresAttrName, requires);
  state.addAttribute("weft_rvv.policy", policy);
  state.addRegion();
  auto variant = llvm::cast<weftexec::VariantOp>(builder.create(state));
  variant.getBody().emplaceBlock();
  return variant;
}

// ---------------------------------------------------------------------------
// [D-4] SCHEDULE-STAGE fill-LMUL attribution sink. A canonical-JSON side channel
// mirroring the exec-stage buildSelectionAttributionRecord FORM (sorted keys,
// deterministic escaping via llvm::json::Value) but NOT routed through it -- this
// is a DIFFERENT stage with DIFFERENT keys (candidates/chosen/minimum_vlen, no
// keys_evaluated). The sink is a pure side effect: option-gated OFF by default and
// writes to a stream only, so it NEVER touches the constructed IR or the exported
// object -- the byte-identity of the untuned construction is preserved. reason
// mirrors only the already-constructed RVVSourceScheduleFormula result.
// ---------------------------------------------------------------------------

void appendScheduleAttributionTimestamp(llvm::raw_ostream &os, bool noTimestamp) {
  if (noTimestamp) {
    // Fixed sentinel keeps lit/FileCheck output byte-deterministic.
    os << llvm::json::Value("0");
    return;
  }
  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  os << llvm::json::Value(buffer);
}

std::string buildScheduleFillAttributionRecord(
    llvm::StringRef kernelName, llvm::ArrayRef<std::string> candidates,
    llvm::StringRef chosen, llvm::StringRef reason, std::int64_t minimumVLEN,
    const RVVNumericsTierChoice &numericsChoice,
    llvm::StringRef declaredInstanceHash, bool noTimestamp) {
  std::string line;
  llvm::raw_string_ostream os(line);
  // Canonical top-level key order (sorted): candidates, chosen,
  // declared_instance_hash, kernel, minimum_vlen, numerics_reason,
  // numerics_tier, reason, ts. numerics_reason/numerics_tier ride ONLY on the
  // chooseNumericsTier output (they cannot be forged here), exactly as the
  // fill-LMUL attribution mirrors the formula result. The [GAP-NUM] tier is a
  // schedule-stage policy pick attributed at the SAME sink.
  os << '{';
  os << "\"candidates\":[";
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    if (index)
      os << ',';
    os << llvm::json::Value(candidates[index]);
  }
  os << ']';
  os << ",\"chosen\":" << llvm::json::Value(chosen.str());
  os << ",\"declared_instance_hash\":"
     << llvm::json::Value(declaredInstanceHash.str());
  os << ",\"kernel\":" << llvm::json::Value(kernelName.str());
  os << ",\"minimum_vlen\":" << minimumVLEN;
  os << ",\"numerics_reason\":"
     << llvm::json::Value(
            stringifyRVVNumericsTierReason(numericsChoice.reason).str());
  os << ",\"numerics_tier\":"
     << llvm::json::Value(stringifyRVVNumericsTier(numericsChoice.tier).str());
  os << ",\"reason\":" << llvm::json::Value(reason.str());
  os << ",\"ts\":";
  appendScheduleAttributionTimestamp(os, noTimestamp);
  os << '}';
  os.flush();
  return line;
}

mlir::LogicalResult
materializeKernel(mlir::OpBuilder &builder, llvm::StringRef kernelName,
                  const ExtensionPluginRegistry &registry,
                  const MonolithicBlockDotOpEntry &entry,
                  BlockDotSourceMatch source, llvm::StringRef march,
                  llvm::StringRef isaVectorHints, bool numericsReassocOk,
                  llvm::raw_ostream *attributionStream,
                  bool attributionNoTimestamp) {
  (void)registry;
  mlir::Location loc = source.func.getLoc();
  weftrvv::PolicyAttr policy = createAgnosticPolicy(builder);
  std::string selectedVariantSymbol = entry.variantSymbol.str();

  mlir::OperationState kernelState(loc, weftexec::KernelOp::getOperationName());
  kernelState.addAttribute("sym_name", builder.getStringAttr(kernelName));
  kernelState.addAttribute("construction_domain",
                           builder.getStringAttr("riscv-execution"));
  kernelState.addAttribute("problem", symbolRef(builder, "canonical_problem"));
  kernelState.addRegion();
  auto kernel = llvm::cast<weftexec::KernelOp>(builder.create(kernelState));
  kernel.getBody().emplaceBlock();

  mlir::OpBuilder::InsertionGuard kernelGuard(builder);
  builder.setInsertionPointToStart(&kernel.getBody().front());

  auto requiredProblemFact = [&](llvm::StringRef name)
      -> std::optional<std::int64_t> {
    for (const MonolithicBlockDotI64Attr &fact : entry.facts)
      if (fact.name == name)
        return fact.value;
    return std::nullopt;
  };
  std::optional<std::int64_t> qk = requiredProblemFact("qk");
  std::optional<std::int64_t> weightStride =
      requiredProblemFact("weight_block_stride");
  std::optional<std::int64_t> activationStride =
      requiredProblemFact("activation_block_stride");
  if (!qk || !weightStride || !activationStride)
    return fail(entry, source.func,
                "formula row lacks required canonical problem block geometry");
  mlir::OperationState problemState(
      loc, weftexec::QuantizedBlockDotProblemOp::getOperationName());
  problemState.addAttribute("sym_name",
                            builder.getStringAttr("canonical_problem"));
  problemState.addAttribute("weight_encoding",
                            builder.getStringAttr(entry.kind));
  problemState.addAttribute("activation_encoding",
                            builder.getStringAttr(entry.activationPurpose));
  problemState.addAttribute("topology",
                            builder.getStringAttr(entry.scaleModel));
  problemState.addAttribute("qk", builder.getI64IntegerAttr(*qk));
  problemState.addAttribute("weight_block_stride",
                            builder.getI64IntegerAttr(*weightStride));
  problemState.addAttribute("activation_block_stride",
                            builder.getI64IntegerAttr(*activationStride));
  (void)builder.create(problemState);

  createCapability(builder, loc, kRVVCapabilitySymbol, "rvv", "isa-vector");
  mlir::ArrayAttr rvvRequires = createRequires(builder, kRVVCapabilitySymbol);

  weftexec::VariantOp rvvVariant =
      createVariant(builder, loc, selectedVariantSymbol, rvvRequires, policy);
  mlir::OpBuilder::InsertionGuard variantGuard(builder);
  builder.setInsertionPointToStart(&rvvVariant.getBody().front());

  mlir::Type runtimeABIType =
      weftrvv::RuntimeABIValueType::get(builder.getContext());
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
      entry.opName == "weft_rvv.q8_0_q8_0_block_dot";
  const bool isQ40TypedFlat =
      entry.opName == weftrvv::GgmlBlockDotQ40Q80Op::getOperationName();
  const bool isQ41TypedFlat =
      entry.opName == "weft_rvv.q4_1_q8_1_block_dot";
  const bool isQ50TypedFlat =
      entry.opName == "weft_rvv.q5_0_q8_0_block_dot";
  const bool isQ51TypedFlat =
      entry.opName == "weft_rvv.q5_1_q8_1_block_dot";
  // iq4_nl (CODEBOOK class, 2nd primitive class): the 3rd flat DECODE axis flips to
  // the typed flat loop path (its codebook branch constructs the codebook_table_
  // broadcast + codebook_gather_x_i8_product bricks). Unlike the plain flat cores,
  // its OUTER with_vl frame stays SEW32/m1 (the standalone_reduce codebook framing;
  // the e8m1 codebook gather runs its own vsetvl INSIDE the region), so configSEW
  // stays 32 for it (below).
  const bool isIq4NlTypedFlat =
      entry.opName == "weft_rvv.iq4_nl_q8_0_block_dot";
  const bool typedFlatLoopPath = isQ80TypedFlat || isQ40TypedFlat ||
                                 isQ41TypedFlat || isQ50TypedFlat ||
                                 isQ51TypedFlat || isIq4NlTypedFlat;
  // GATED typed SUPER-BLOCK path (M-FLAT q4_K milestone-3): q4_K's front door
  // constructs the COMPLETE per-super-block TYPED DUAL-accumulator loop chain
  // (typed_super_block_block_dot_loop_body region) instead of the ONE monolith
  // weft_rvv.q4_k_q8_k_block_dot op. Unlike the flat typed path, the super-block
  // integer core runs at the shared SEW32/m1 config (the SAME setvl/with_vl the
  // monolith used -- the per-sub-block e8/i16/i32 widening lives INSIDE the bricks),
  // so configSEW/configLMUL are UNCHANGED from the monolith path (32/"m1"), and
  // there is NO zero_seed (the dual accumulators are seeded internally by the
  // lowering). Every other (monolith) row stays byte-unchanged.
  const bool isQ4KTypedSuperBlock =
      entry.opName == "weft_rvv.q4_k_q8_k_block_dot";
  // q5_K first flip: q5_K == q4_K + the qh 5th-bit plane. It takes the SAME typed
  // super-block dual-accumulator loop chain (the 5 shared bricks + dual yield),
  // the ONLY addition being BRICK 1's weight_qh_byte_offset attr (stamped from
  // kQ5KFacts inside the chain builder). The stride-176 facts + qh offset flow
  // through entry.facts, so no q5_K-specific construction code is needed here.
  const bool isQ5KTypedSuperBlock =
      entry.opName == "weft_rvv.q5_k_q8_k_block_dot";
  const bool isTypedSuperBlock = isQ4KTypedSuperBlock || isQ5KTypedSuperBlock;
  // q6_K first flip: q6_K has NO per-block min, so it flips to the typed super-block
  // SINGLE-accumulator loop chain (fold_model "scales_times_sumi" -- the aux32
  // integer core + the no-min positive fold + a single `sums` yield), NOT the
  // q4_K/q5_K dual chain. Its stride-210 facts flow through entry.facts.
  const bool isQ6KTypedSuperBlock =
      entry.opName == "weft_rvv.q6_k_q8_k_block_dot";
  // q3_K first flip: q3_K is SYMMETRIC (NO per-block min), so it flips to the SAME
  // typed super-block SINGLE-accumulator loop chain as q6_K (fold_model
  // "scales_times_sumi" -- the q3_K aux32 integer core + the reused no-min positive
  // fold + a single `sums` yield). The chain builder disambiguates q3_K (stride
  // 110, hmask/qs planes) from q6_K (stride 210, qh plane) by entry.opName; its
  // stride-110 facts flow through entry.facts.
  const bool isQ3KTypedSuperBlock =
      entry.opName == "weft_rvv.q3_k_q8_k_block_dot";
  // q2_K first flip: q2_K HAS a per-block min (like q4_K/q5_K) but its whole fold
  // is a SINGLE per-super-block SCALAR `sumf += dall*isum - dmin*summs`, so it
  // flips to the typed super-block SCALAR-accumulator loop chain (fold_model
  // "scalar_scale_min" -- the q2_K integer core + the emitter-inlined scalar fold
  // + a single `sumf` scalar yield), NOT the q4_K/q5_K dual nor the q6_K
  // single-vector chain. Its stride-84 facts flow through entry.facts.
  const bool isQ2KTypedSuperBlock =
      entry.opName == "weft_rvv.q2_k_q8_k_block_dot";
  // iq1_s flip (L3 M3): iq1_s is a super-block GRID/codebook quant whose whole fold
  // is a SINGLE per-super-block SCALAR `sumf += d*((float)sumi + IQ1S_DELTA*
  // (float)sumi1)` (the SAME scalar-accumulator arity as q2_K), so it flips to the
  // typed super-block SCALAR-accumulator loop chain (fold_model "scalar_delta_grid"
  // -- the iq1_s ternary-grid integer core + the emitter-inlined scalar delta fold +
  // a single `sumf` scalar yield), NOT the q4_K/q5_K dual, the q6_K single-vector,
  // nor q2_K's arithmetic scalar chain. Its stride-50 facts + the iq1s_grid flow
  // through entry.facts. The monolith op weft_rvv.iq1_s_q8_k_block_dot is retired, so
  // this gate keys off the entry.opName STRING (no op type reference).
  const bool isIq1sTypedSuperBlock =
      entry.opName == "weft_rvv.iq1_s_q8_k_block_dot";
  // iq1_m flip (L3): iq1_m is the iq1_s sibling -- a super-block GRID/codebook quant
  // whose whole fold is the SAME SINGLE per-super-block SCALAR `sumf += d*((float)sumi1
  // + IQ1M_DELTA*(float)sumi2)` (fold_model "scalar_delta_grid"), REUSING the whole
  // iq1_s scaffold; the only marginal cost is the DISTINCT iq1_m ternary-grid integer
  // core brick (packed-scale reconstruct + half-split grid dot + per-group four-sign
  // delta). It flips to the typed super-block SCALAR-accumulator loop chain, resolving
  // to its OWN export entry by weight_block_stride 56 (vs iq1_s 50). The monolith op
  // weft_rvv.iq1_m_q8_k_block_dot is retired, so this gate keys off the entry.opName
  // STRING (no op type reference).
  const bool isIq1mTypedSuperBlock =
      entry.opName == "weft_rvv.iq1_m_q8_k_block_dot";
  // iq3_xxs flip (L3 coverage): iq3_xxs is another iq1_s grid sibling -- a super-block
  // GRID/codebook quant whose whole fold is the SAME SINGLE per-super-block SCALAR
  // accumulator arity (fold_model "scalar_delta_grid"), REUSING the whole iq1_s
  // scaffold; the only marginal cost is the DISTINCT iq3_xxs GRID-of-4 integer-core
  // brick (i32 iq3xxs_grid vluxei16 gather + aux32 4-bit-scale + 4-sign-group ksigns +
  // 0.25f trailing factor). It flips to the typed super-block SCALAR-accumulator loop
  // chain, resolving to its OWN export entry by weight_block_stride 98 (vs iq1_s 50,
  // iq1_m 56). The monolith op weft_rvv.iq3_xxs_q8_k_block_dot is retired, so this gate
  // keys off the entry.opName STRING (no op type reference).
  const bool isIq3xxsTypedSuperBlock =
      entry.opName == "weft_rvv.iq3_xxs_q8_k_block_dot";
  // iq2_xxs flip (L3 coverage, SIGN-PLANE signs64 variant): iq2_xxs is another iq1_s grid
  // sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
  // per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"), REUSING the
  // whole iq1_s scaffold; the only marginal cost is the DISTINCT iq2_xxs GRID-of-8
  // integer-core brick (i64 iq2xxs_grid vluxei16 gather + the SECOND signs64 vluxei16
  // gather over the DERIVED keven_signs_q2xs sign plane + aux1 4-bit-scale + 4-sign-group
  // decode + 0.125f trailing factor). It flips to the typed super-block SCALAR-accumulator
  // loop chain, resolving to its OWN export entry by weight_block_stride 66 (vs iq1_s 50,
  // iq1_m 56, iq3_xxs 98). The brick carries the SAME Win-A integer_core_lmul gearbox. The
  // monolith op weft_rvv.iq2_xxs_q8_k_block_dot is retired, so this gate keys off the
  // entry.opName STRING (no op type reference).
  const bool isIq2xxsTypedSuperBlock =
      entry.opName == "weft_rvv.iq2_xxs_q8_k_block_dot";
  // iq2_xs flip (L3 coverage, SIGN-PLANE signs64 variant, PER-HALF explicit scale): iq2_xs
  // is the iq2_xxs grid sibling -- a super-block GRID/codebook quant whose whole fold is the
  // SAME SINGLE per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"),
  // REUSING the whole iq1_s scaffold; the only marginal cost is the DISTINCT iq2_xs
  // per-half-explicit-scale GRID integer-core brick (the 512-entry iq2xs_grid vluxei16_v_i64
  // gather indexed by `w & 511` + the SECOND signs64 vluxei16 gather over the DERIVED
  // keven_signs_q2xs sign plane keyed by `w >> 9` + the EXPLICIT per-sub-block 4-bit
  // scales[8] two-half split ls1/ls2 + 0.125f trailing factor). It flips to the typed
  // super-block SCALAR-accumulator loop chain, resolving to its OWN export entry by
  // weight_block_stride 74 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66). UNLIKE iq2_xxs
  // the brick carries NO gearbox (fixed 16-lane per-half shape). The monolith op
  // weft_rvv.iq2_xs_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isIq2xsTypedSuperBlock =
      entry.opName == "weft_rvv.iq2_xs_q8_k_block_dot";
  // iq2_s flip (L3 coverage, SIGN-PLANE explicit-signs variant, PER-HALF explicit scale):
  // iq2_s is the iq2_xs grid sibling -- a super-block GRID/codebook quant whose whole fold is
  // the SAME SINGLE per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"),
  // REUSING the whole iq1_s scaffold; the only marginal cost is the DISTINCT iq2_s
  // per-half-explicit-scale GRID integer-core brick (the 1024-entry iq2s_grid vluxei16_v_i64
  // gather indexed by `qs[l] | ((qh<<(8-2l))&0x300)` + the SECOND signs256 vluxei16 gather
  // over the UNIVERSAL explicit-sign-byte plane keyed by the RAW sign byte + the EXPLICIT
  // per-sub-block 4-bit scales[8] two-half split ls1/ls2 + 0.125f trailing factor). It flips
  // to the typed super-block SCALAR-accumulator loop chain, resolving to its OWN export entry
  // by weight_block_stride 82 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74). Like
  // iq2_xs the brick carries NO gearbox (fixed 16-lane per-half shape). The monolith op
  // weft_rvv.iq2_s_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isIq2sTypedSuperBlock =
      entry.opName == "weft_rvv.iq2_s_q8_k_block_dot";
  // iq3_s flip (C_construct 22->23, EXPLICIT-SIGNS variant): iq3_s is the iq3_xxs GRID-of-4
  // sibling -- a super-block GRID/codebook quant whose whole fold is the SAME SINGLE
  // per-super-block SCALAR accumulator arity (fold_model "scalar_delta_grid"), REUSING the
  // whole iq1_s scaffold; the only marginal cost is the DISTINCT iq3_s GRID-of-4
  // explicit-signs integer-core brick (the 512-entry iq3s_grid vluxei16_v_i32m1 gather
  // indexed by `qs[l] | ((qh<<(8-2l))&256)` + the EXPLICIT per-sub-block sign bytes at
  // offset 74 folded via the inline kmask {1<<j} + the explicit two-nibble scales at offset
  // 106 + NO trailing factor). It flips to the typed super-block SCALAR-accumulator loop
  // chain, resolving to its OWN export entry by weight_block_stride 110 (vs iq1_s 50, iq1_m
  // 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s 82). UNLIKE iq3_xxs there is NO ksigns
  // plane (the signs are an explicit memory region); NO gearbox (fixed grid-of-4 shape).
  // The monolith op weft_rvv.iq3_s_q8_k_block_dot is retired, so this gate keys off the
  // entry.opName STRING (no op type reference).
  const bool isIq3sTypedSuperBlock =
      entry.opName == "weft_rvv.iq3_s_q8_k_block_dot";
  // iq4_xs flip (C_construct 23->24, the FIRST super-block CODEBOOK member): iq4_xs is the
  // SUPER-BLOCK rung of the flat iq4_nl codebook -- a super-block CODEBOOK quant whose whole
  // fold is the SAME SINGLE per-super-block SCALAR accumulator arity (fold_model
  // "scalar_delta_grid"), REUSING the whole iq1_s scaffold; the only marginal cost is the
  // DISTINCT iq4_xs CODEBOOK integer-core brick (iq4_nl's 16-entry vrgather codebook gather
  // + the q4_K-style signed 6-bit scale bit-dance + the per-sub-block float fold `sumf +=
  // (d4d8*(ls-32))*sumi`, NO trailing factor). It flips to the typed super-block
  // SCALAR-accumulator loop chain, resolving to its OWN export entry by weight_block_stride
  // 136 (vs iq1_s 50, iq1_m 56, iq3_xxs 98, iq2_xxs 66, iq2_xs 74, iq2_s 82, iq3_s 110).
  // UNLIKE the grid siblings the fold runs per-sub-block in float, but the single-scalar
  // accumulator arity is identical; NO gearbox (the codebook gather pins m1). The monolith
  // op weft_rvv.iq4_xs_q8_k_block_dot is retired, so this gate keys off the entry.opName
  // STRING (no op type reference).
  const bool isIq4xsTypedSuperBlock =
      entry.opName == "weft_rvv.iq4_xs_q8_k_block_dot";
  // tq2_0 flip (C_construct 24->25, the FIRST TQ-family member): tq2_0 is the 2-bit TERNARY
  // ({-1,0,+1}) TriLM K-quant whose whole fold is the SAME SINGLE per-super-block SCALAR
  // accumulator arity (fold_model "scalar_delta_grid"), REUSING the whole iq1_s scaffold; the
  // only marginal cost is the DISTINCT tq2_0 FUSED 2-bit TERNARY integer-core brick (q2_K's
  // 2-bit `(qs>>shift)&3` unpack + the per-element `-1` bias + the fused-plane vwmacc dot,
  // producing the ONE scalar state sumi -- NO grid/codebook gather). It flips to the typed
  // super-block SCALAR-accumulator loop chain, resolving to its OWN export entry by the marker
  // pass name (it SHARES weight_block_stride 66 with iq2_xxs but the DISTINCT brick op type
  // disambiguates the emitter). UNLIKE the iq4_xs codebook sibling the brick PRESERVES tq2_0's
  // Win-A integer_core_lmul m2/m1 gearbox (kernel key "tq2_0"). The monolith op
  // weft_rvv.tq2_0_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isTq20TypedSuperBlock =
      entry.opName == "weft_rvv.tq2_0_q8_k_block_dot";
  // tq1_0 flip (C_construct 25->26, the SECOND TQ-family member): tq1_0 is the BASE-3
  // TERNARY ({-1,0,+1}) TriLM K-quant whose whole fold is the SAME SINGLE per-super-block
  // SCALAR accumulator arity (fold_model "scalar_delta_grid"), REUSING the whole tq2_0
  // ternary scaffold at C2 marginal cost; the only marginal cost is the DISTINCT tq1_0 BASE-3
  // TERNARY integer-core brick (the qs+qh base-3 trit unpack into aux8[256] + the flat-256
  // widened dot producing the ONE scalar state sumi -- NO grid/codebook gather). It flips to
  // the typed super-block SCALAR-accumulator loop chain, resolving to its OWN export entry by
  // the marker pass name (its weight_block_stride 54 is UNIQUE among the scalar_delta_grid
  // bricks, so no stride tie-breaker is needed). Unlike tq2_0, tq1_0 has one
  // VLEN-universal realized body and no inert LMUL schedule field. The monolith op
  // weft_rvv.tq1_0_q8_k_block_dot is retired, so this gate keys off the entry.opName STRING
  // (no op type reference).
  const bool isTq10TypedSuperBlock =
      entry.opName == "weft_rvv.tq1_0_q8_k_block_dot";
  // q1_0 flip (C_construct 26->27, the LAST flat block-dot family member): q1_0 is
  // the BINARY {-1,+1}-sign class whose per-super-block contribution is a
  // FOUR-sub-block binary sign decode with a DISTINCT TWO-LEVEL fp32 fold
  // (`d0 * Σ_k(d1_k * sumi_block_k)`). It flips to the FLAT loop chain
  // (typed_flat_block_dot_loop_body, fold_model "flat_binary_two_level") carrying
  // ONE net-new binary-sign integer-core brick (the whole per-super-block body +
  // the emitter-inlined two-level fold), NOT any existing single-core flat brick
  // chain. It is DELIBERATELY kept OUT of typedFlatLoopPath: q1_0's OUTER with_vl
  // frame stays SEW32/m1 (like the monolith / iq4_nl -- the e8m2 binary sign
  // decode runs its OWN vsetvl INSIDE the brick), whereas typedFlatLoopPath forces
  // the SEW8 outer config. The monolith op weft_rvv.q1_0_q8_0_block_dot is retired,
  // so this gate keys off the entry.opName STRING (no op type reference).
  const bool isQ10TypedFlat = entry.opName == "weft_rvv.q1_0_q8_0_block_dot";
  // nvfp4 flip (C_construct 27->28, the LAST dispatch-wired vec_dot, closing the ①
  // G1 literal-block-dot zoo): nvfp4 (NVIDIA's FP4, the SECOND FP4-CODEBOOK class) is
  // a SUPER-BLOCK codebook quant whose 64 elements span TWO block_q8_0 activation
  // blocks -- a FLAT block_q8_0 stream (like q1_0). It flips to the FLAT loop chain
  // (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") carrying ONE
  // net-new codebook integer-core brick (the per-super-block body + the
  // emitter-inlined per-sub-block UE4M3-codebook fold), NOT any existing flat brick
  // chain. Like q1_0 it is DELIBERATELY kept OUT of typedFlatLoopPath: nvfp4's OUTER
  // with_vl frame stays SEW32/m1 (the e8m1 codebook strip runs its OWN vsetvl INSIDE
  // the brick), whereas typedFlatLoopPath forces the SEW8 outer config. The monolith
  // op weft_rvv.nvfp4_q8_0_block_dot is retired, so this gate keys off the
  // entry.opName STRING (no op type reference).
  const bool isNvfp4TypedFlat =
      entry.opName == "weft_rvv.nvfp4_q8_0_block_dot";
  // iq4_nl frames its OUTER with_vl at SEW32/m1 (the codebook standalone_reduce
  // framing; the e8m1 gather core runs its own vsetvl inside the region), unlike the
  // plain flat cores which frame the OUTER config at SEW8.
  const std::int64_t configSEW =
      (typedFlatLoopPath && !isIq4NlTypedFlat) ? 8 : 32;
  // The typed-flat integer-core LMUL schedule, the ONE source fed to BOTH the
  // setvl/with_vl config (configLMUL) AND the loop-body chain (integer_core_lmul
  // stamp / coreLmul-wideLmul / q8_0 product_relation), so the verifier-cross-pinned
  // knobs cannot diverge. [SEL-1] step 2: the schedule is now SELECTED by the
  // capability-keyed fill-optimal LMUL prior instead of hardcoded. The rule lives
  // ONLY in the shared cost-agnostic helper (NG-1); the front door merely supplies
  // the per-format constructible candidate set + the block element span + the VLEN
  // fact derived from the selected -march, then requests a fill-optimal LMUL:
  //   - q8_0 (whole-block plain-i8 core, constructible at BOTH m1 and m2 per step
  //     1b): VLEN>=256 fills the qk=32 block at m1 (util 1.0) => m1; VLEN==128 ties
  //     m1/m2 at util 1.0 => tiebreak widest => m2; no -march / sub-128 fails safe
  //     to the widest default m2 = today's hardcode (byte-identical construction).
  //   - the half-block packed-i4 formats pin m1 (their region product_relation is
  //     i8m1), so their candidate set is the single-member {m1} => reason
  //     only_feasible, m1 at every VLEN.
  // The monolith (non-typed) path keeps its SEW32 "m1" config untouched.
  std::int64_t typedFlatQk = 0;
  for (const MonolithicBlockDotI64Attr &fact : entry.facts)
    if (fact.name == "qk")
      typedFlatQk = fact.value;
  // The half-block packed-i4 core covers qk/2 elements per strip (the low OR high
  // nibble half); q8_0's contiguous plain-i8 core covers the whole qk block. The
  // iq4_nl codebook core is likewise a qk/2 half-block strip (its {m1} candidate is
  // pinned by the 16-entry gather VLMAX fact, so this fill-LMUL query returns m1).
  const bool isTypedFlatHalfBlock = isQ40TypedFlat || isQ41TypedFlat ||
                                    isQ50TypedFlat || isQ51TypedFlat ||
                                    isIq4NlTypedFlat;
  const std::int64_t typedFlatBlockLen =
      isTypedFlatHalfBlock ? (typedFlatQk / 2) : typedFlatQk;
  llvm::SmallVector<std::string, 2> typedFlatLMULCandidates;
  if (isQ80TypedFlat)
    typedFlatLMULCandidates = {"m1", "m2"};
  else
    typedFlatLMULCandidates = {"m1"};
  const std::int64_t minimumVLEN = resolveRVVMinimumVLEN(
      source.func->getParentOfType<mlir::ModuleOp>(), march, isaVectorHints);
  llvm::Expected<RVVSourceSchedulePlan> sourceSchedule =
      constructRVVSourceScheduleFormula(
          {RVVSourceScheduleMechanism::FillOptimal,
           /*sew=*/8, typedFlatBlockLen, typedFlatLMULCandidates},
          {minimumVLEN,
           resolveRVVVectorRegisterBudget(
               source.func->getParentOfType<mlir::ModuleOp>())},
          RVVSourceScheduleNoStaticContext{});
  if (!sourceSchedule) {
    source.func.emitError() << llvm::toString(sourceSchedule.takeError());
    return mlir::failure();
  }
  const llvm::StringRef typedFlatLmul = sourceSchedule->integerCoreLMUL;
  const llvm::StringRef configLMUL = typedFlatLoopPath ? typedFlatLmul : "m1";

  // The per-block reduce seed (0), a variant-scope value that dominates the
  // in-region standalone_reduce. Only the typed path needs it; adding it to the
  // monolith path would perturb its byte-exact ABI value set.
  mlir::Value zeroSeed;
  if (typedFlatLoopPath)
    zeroSeed = createRuntimeABIValue(builder, loc, "accumulator-input-buffer",
                                     "zero_seed", "const int32_t *",
                                     "loop-body:reduce-seed", runtimeABIType);

  weftrvv::SetVLOp setvl =
      createSetVL(builder, loc, n, configSEW, configLMUL, policy);
  weftrvv::WithVLOp withVL =
      createWithVL(builder, loc, setvl.getVl(), configSEW, configLMUL, policy);

  mlir::OpBuilder::InsertionGuard withVLGuard(builder);
  builder.setInsertionPointToStart(&withVL.getBody().front());

  if (typedFlatLoopPath) {
    // The auto-constructed typed flat block-dot loop chain (brick 1 -> integer
    // core -> brick 2 -> brick 3 -> yield are op structure inside the region).
    createTypedFlatBlockDotLoopChain(builder, loc, entry, weight, activation,
                                     out, n, setvl.getVl(), zeroSeed,
                                     typedFlatLmul);
  } else if (isTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK dual-accumulator loop chain (the 5
    // shared q4_K/q5_K bricks + the dual yield are op structure inside the region;
    // q5_K additionally stamps BRICK 1's qh offset from entry.facts).
    createTypedSuperBlockBlockDotLoopChain(builder, loc, entry, weight,
                                           activation, out, n, setvl.getVl());
  } else if (isQ6KTypedSuperBlock || isQ3KTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SINGLE-accumulator loop chain (the
    // q6_K/q3_K aux32 integer core + the reused no-min positive fold + the single
    // `sums` yield are op structure inside the region -- no MIN term, no sumf
    // scalar). The chain builder keys the q3_K vs q6_K integer-core brick off
    // entry.opName; both are SYMMETRIC no-min super-blocks sharing the fold.
    createTypedSuperBlockScalesTimesSumiLoopChain(builder, loc, entry, weight,
                                                  activation, out, n,
                                                  setvl.getVl());
  } else if (isQ2KTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain (the
    // q2_K integer core producing the two scalar states isum + summs + the single
    // `sumf` scalar yield are op structure inside the region; the scalar fold
    // `sumf += dall*isum - dmin*summs` is emitter-inlined -- no separate fold
    // brick, no 8-lane sums vector, no post-loop horizontal add).
    createTypedSuperBlockScalarScaleMinLoopChain(builder, loc, entry, weight,
                                                 activation, out, n,
                                                 setvl.getVl());
  } else if (isIq1sTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain (the
    // iq1_s ternary-grid integer core producing the two scalar states sumi + sumi1 +
    // the single `sumf` scalar yield are op structure inside the region; the scalar
    // delta fold `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)` is
    // emitter-inlined -- no separate fold brick, no 8-lane sums vector, no post-loop
    // horizontal add). Same scalar-accumulator arity as q2_K, distinct GRID core +
    // fold arithmetic (fold_model "scalar_delta_grid").
    createTypedSuperBlockScalarDeltaGridLoopChain(builder, loc, entry, weight,
                                                  activation, out, n,
                                                  setvl.getVl());
  } else if (isIq1mTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq1_m
    // variant: the SAME scaffold as iq1_s (fold_model "scalar_delta_grid", single
    // `sumf` scalar yield, emitter-inlined scalar delta fold) with the DISTINCT iq1_m
    // ternary-grid integer core brick (producing sumi1 + sumi2). The C2 marginal-cost
    // payoff -- the second GRID/codebook family member reuses the whole iq1_s scaffold
    // and only adds a variant brick. Resolves to iq1_m's OWN export entry (stride 56).
    createTypedSuperBlockScalarDeltaGridLoopChainIq1M(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isIq3xxsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain,
    // iq3_xxs variant: the SAME scaffold as iq1_s (fold_model "scalar_delta_grid",
    // single `sumf` scalar yield, emitter-inlined scalar fold) with the DISTINCT
    // iq3_xxs GRID-of-4 integer-core brick (producing the ONE scalar state bsum via the
    // i32 iq3xxs_grid vluxei16 gather + aux32 4-bit-scale + 4-sign-group ksigns decode).
    // Another L3 coverage payoff -- the third GRID/codebook family member reuses the
    // whole iq1_s scaffold and only adds a variant brick. Resolves to iq3_xxs's OWN
    // export entry (stride 98).
    createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs(builder, loc, entry, weight,
                                                        activation, out, n,
                                                        setvl.getVl());
  } else if (isIq2xxsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq2_xxs
    // variant (SIGN-PLANE signs64): the SAME scaffold as iq1_s (fold_model
    // "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined scalar fold) with
    // the DISTINCT iq2_xxs GRID-of-8 integer-core brick (producing the ONE scalar state
    // bsum via the i64 iq2xxs_grid vluxei16 gather + the SECOND signs64 vluxei16 gather
    // over the DERIVED keven_signs_q2xs sign plane + aux1 4-bit-scale + 4-sign-group
    // decode). Another L3 coverage payoff -- the FOURTH GRID/codebook family member reuses
    // the whole iq1_s scaffold and only adds a variant brick (+ preserves its own Win-A
    // m2/m1 gearbox on the brick). Resolves to iq2_xxs's OWN export entry (stride 66).
    createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs(builder, loc, entry, weight,
                                                        activation, out, n,
                                                        setvl.getVl());
  } else if (isIq2xsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq2_xs
    // variant (SIGN-PLANE signs64, PER-HALF explicit scale): the SAME scaffold as iq1_s
    // (fold_model "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined scalar
    // fold) with the DISTINCT iq2_xs per-half-explicit-scale GRID integer-core brick
    // (producing the ONE scalar state bsum via the 512-entry iq2xs_grid vluxei16_v_i64
    // gather indexed by `w & 511` + the SECOND signs64 vluxei16 gather over the DERIVED
    // keven_signs_q2xs sign plane keyed by `w >> 9` + the EXPLICIT per-sub-block 4-bit
    // scales[8] two-half split ls1/ls2). Another L3 coverage payoff -- the FIFTH
    // GRID/codebook family member reuses the whole iq1_s scaffold and only adds a variant
    // brick (NO gearbox -- fixed 16-lane per-half shape). Resolves to iq2_xs's OWN export
    // entry (stride 74).
    createTypedSuperBlockScalarDeltaGridLoopChainIq2xs(builder, loc, entry, weight,
                                                       activation, out, n,
                                                       setvl.getVl());
  } else if (isIq2sTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq2_s
    // variant (SIGN-PLANE explicit-signs, PER-HALF explicit scale): the SAME scaffold as
    // iq1_s (fold_model "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined
    // scalar fold) with the DISTINCT iq2_s per-half-explicit-scale GRID integer-core brick
    // (producing the ONE scalar state bsum via the 1024-entry iq2s_grid vluxei16_v_i64
    // gather indexed by `qs[l] | ((qh<<(8-2l))&0x300)` + the SECOND signs256 vluxei16 gather
    // over the UNIVERSAL explicit-sign-byte plane keyed by the RAW sign byte + the EXPLICIT
    // per-sub-block 4-bit scales[8] two-half split ls1/ls2). Another L3 coverage payoff --
    // the SIXTH GRID/codebook family member reuses the whole iq1_s scaffold and only adds a
    // variant brick (NO gearbox -- fixed 16-lane per-half shape). Resolves to iq2_s's OWN
    // export entry (stride 82).
    createTypedSuperBlockScalarDeltaGridLoopChainIq2s(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isIq3sTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator GRID loop chain, iq3_s
    // variant (EXPLICIT-SIGNS, qh 9th-bit inject, explicit two-nibble scales): the SAME
    // scaffold as iq1_s (fold_model "scalar_delta_grid", single `sumf` scalar yield,
    // emitter-inlined scalar fold) with the DISTINCT iq3_s GRID-of-4 explicit-signs
    // integer-core brick (producing the ONE scalar state bsum via the 512-entry iq3s_grid
    // vluxei16_v_i32m1 gather indexed by `qs[l] | ((qh<<(8-2l))&256)` + the EXPLICIT
    // per-sub-block sign bytes at offset 74 folded via the inline kmask {1<<j} + the
    // explicit two-nibble scales at offset 106). Another C_construct payoff -- the SEVENTH
    // GRID/codebook family member reuses the whole iq1_s scaffold and only adds a variant
    // brick (NO gearbox, NO ksigns plane, NO trailing factor). Resolves to iq3_s's OWN
    // export entry (stride 110).
    createTypedSuperBlockScalarDeltaGridLoopChainIq3s(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isIq4xsTypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain, iq4_xs
    // variant (the FIRST super-block CODEBOOK member vs the grid siblings): the SAME
    // scaffold as iq1_s (fold_model "scalar_delta_grid", single `sumf` scalar yield,
    // emitter-inlined fold) with the DISTINCT iq4_xs CODEBOOK integer-core brick (iq4_nl's
    // 16-entry vrgather codebook gather + the q4_K-style signed 6-bit scale bit-dance + the
    // per-sub-block float fold `sumf += (d4d8*(ls-32))*sumi`, NO trailing factor). Another
    // C_construct payoff -- reuses the whole iq1_s scaffold and only adds a variant codebook
    // integer-core brick (NO gearbox -- the codebook gather pins m1). Resolves to iq4_xs's
    // OWN export entry (stride 136).
    createTypedSuperBlockScalarDeltaGridLoopChainIq4xs(builder, loc, entry, weight,
                                                       activation, out, n,
                                                       setvl.getVl());
  } else if (isTq20TypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain, tq2_0 variant
    // (the FIRST TQ-family member): the SAME scaffold as iq1_s (fold_model "scalar_delta_grid",
    // single `sumf` scalar yield, emitter-inlined fold) with the DISTINCT tq2_0 FUSED 2-bit
    // TERNARY integer-core brick (q2_K's 2-bit unpack + the `-1` ternary bias + the fused-plane
    // vwmacc dot producing the ONE scalar state sumi, then the emitter-inlined scalar fold
    // `sumf += (float)sumi * d`, NO trailing factor). C_construct payoff -- reuses the whole
    // iq1_s scaffold and only adds a variant ternary integer-core brick that PRESERVES tq2_0's
    // Win-A m2/m1 gearbox (kernel key "tq2_0"). Resolves to tq2_0's OWN export entry by the
    // marker pass (SHARES stride 66 with iq2_xxs, disambiguated by the brick op type).
    createTypedSuperBlockScalarDeltaGridLoopChainTq20(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isTq10TypedSuperBlock) {
    // The auto-constructed typed SUPER-BLOCK SCALAR-accumulator loop chain, tq1_0 variant
    // (the SECOND TQ-family member): the SAME scaffold as tq2_0/iq1_s (fold_model
    // "scalar_delta_grid", single `sumf` scalar yield, emitter-inlined fold) with the DISTINCT
    // tq1_0 BASE-3 TERNARY integer-core brick (the qs+qh base-3 trit unpack -- `q=(uint8_t)
    // (byte*pow3[l]); xi=((uint16_t)q*3)>>8; xi-1` -- into aux8[256] + the flat-256 widened dot
    // producing the ONE scalar state sumi, then the emitter-inlined scalar fold
    // `sumf += (float)sumi * d`, NO trailing factor). C_construct payoff -- REUSES the whole
    // tq2_0 ternary scaffold at C2 marginal cost and only adds a base-3 variant integer-core
    // brick with one fixed VLEN-universal vector body. Resolves to tq1_0's
    // OWN export entry by the marker pass (stride 54 is UNIQUE, no tie-breaker needed).
    createTypedSuperBlockScalarDeltaGridLoopChainTq10(builder, loc, entry, weight,
                                                      activation, out, n,
                                                      setvl.getVl());
  } else if (isQ10TypedFlat) {
    // The auto-constructed typed FLAT loop chain, q1_0 variant (the LAST flat
    // block-dot family member, C_construct 26->27): the FLAT loop op
    // (typed_flat_block_dot_loop_body, fold_model "flat_binary_two_level")
    // carrying ONE net-new BINARY-sign integer-core brick (the four q8_0
    // sub-blocks' vlm_v_b{ratio} packed-bit sign mask + i8-domain vneg/vmerge ->
    // vwredsum, plus the emitter-inlined TWO-LEVEL fp32 fold `d0 * Σ_k(d1_k *
    // sumi_block_k)`). Unlike q8_0/q4_0/q5_0 (a single per-block core folded by the
    // shared scale->dequant->accumulate brick chain), q1_0's four-sub-block
    // two-level structure needs its OWN brick + fold; the net-new marginal cost is
    // the DISTINCT binary-sign brick (preserving q1_0's Win-A m2/m1 gearbox, kernel
    // key "q1_0"). The OUTER config stays SEW32/m1 (isQ10TypedFlat is OUT of
    // typedFlatLoopPath), byte-exact to the monolith frame.
    createTypedFlatBlockDotLoopChainQ10(builder, loc, entry, weight, activation,
                                        out, n, setvl.getVl());
  } else if (isNvfp4TypedFlat) {
    // The auto-constructed typed FLAT loop chain, nvfp4 variant (the LAST
    // dispatch-wired vec_dot, C_construct 27->28): the FLAT loop op
    // (typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") carrying
    // ONE net-new FP4-CODEBOOK integer-core brick (the four UE4M3-scaled 16-element
    // sub-blocks' mxfp4 16-entry vrgather codebook gather + the two-q8_0-block/half
    // addressing -> vwredsum, plus the emitter-inlined per-sub-block fp32 fold
    // `sumf += (dy*d)*(float)sumi`). Like q1_0 nvfp4's activation is a block_q8_0
    // stream, so it uses the FLAT loop op, NOT the q8_K super-block one; the net-new
    // marginal cost is the DISTINCT codebook brick (the codebook gather pins m1, NO
    // gearbox). The OUTER config stays SEW32/m1 (isNvfp4TypedFlat is OUT of
    // typedFlatLoopPath), byte-exact to the monolith frame.
    createTypedFlatBlockDotLoopChainNvfp4(builder, loc, entry, weight, activation,
                                          out, n, setvl.getVl());
  } else {
    // The auto-constructed block dot-product op (the scale model, integer core,
    // super-block bit-dance, codebook gather, and deferred fold are op structure).
    (void)createBlockDot(builder, loc, entry, weight, activation, out, n,
                         setvl.getVl());
  }

  // [D-4] schedule-stage fill-LMUL attribution (side channel, option-gated OFF by
  // default). Emit ONE record for the typed-flat LMUL decision after the kernel is
  // fully built. Pure side effect: no IR / object change, so byte-identity holds.
  // declared_instance_hash mirrors the exec sink: the SHA-256 of the constructed
  // kernel's declared capability instance.
  if (attributionStream && typedFlatLoopPath) {
    std::string declaredInstanceHash;
    if (llvm::Expected<support::TargetCapabilitySet> capabilities =
            support::TargetCapabilitySet::buildFromKernelChecked(kernel))
      declaredInstanceHash =
          support::computeDeclaredInstanceHash(*capabilities);
    else
      llvm::consumeError(capabilities.takeError());
    // [GAP-NUM] the schedule-stage numerics-tier pick, attributed alongside the
    // fill-LMUL pick. The typed-flat block-dots all carry an fp cross-block fold,
    // so they ARE fp-order-sensitive; the tier is then keyed by the
    // `numerics.reassoc_ok` policy fact (fail-closed: absent => strict).
    const RVVNumericsTierChoice numericsChoice = chooseNumericsTier(
        numericsReassocOk, /*kernelIsFpOrderSensitive=*/true);
    *attributionStream << buildScheduleFillAttributionRecord(
                              kernelName, typedFlatLMULCandidates,
                              sourceSchedule->integerCoreLMUL,
                              sourceSchedule->analyticReason, minimumVLEN,
                              numericsChoice, declaredInstanceHash,
                              attributionNoTimestamp)
                       << "\n";
  }
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
    if (dialect == "weft" || dialect == "weft_rvv" || dialect == "weft_toy" ||
        dialect == "weft_tensorext_lite")
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

  // PassWrapper::clonePass copy-constructs the pass; the cl::opt-backed options are
  // NOT copyable, so the option members are re-registered against the new *this
  // (their values are then copied by MLIR's copyOptionValuesFrom). Only the
  // non-option state is forwarded here.
  MaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
      const MaterializeRVVMonolithicBlockDotSourceFrontDoorPass &other)
      : mlir::PassWrapper<MaterializeRVVMonolithicBlockDotSourceFrontDoorPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        entry(other.entry), registry(other.registry) {}

  llvm::StringRef getArgument() const final { return entry->passArgument; }
  llvm::StringRef getDescription() const final { return kPassDescription; }

  // [SEL-1] the selected -march whose guaranteed minimum VLEN keys the
  // fill-optimal integer-core LMUL of the constructed typed-flat block-dot (e.g.
  // rv64gcv => VLEN 128 => q8_0 m2; rv64gcv_zvl256b => VLEN 256 => q8_0 m1). Empty
  // / sub-128 => the widest sufficient default (byte-identical to the untuned
  // construction). This is the ONLY input that flips the constructed LMUL.
  ::mlir::Pass::Option<std::string> march{
      *this, "march",
      llvm::cl::desc(
          "Selected RISC-V -march whose guaranteed minimum VLEN keys the "
          "[SEL-1] fill-optimal integer-core LMUL of the constructed typed-flat "
          "block-dot. Empty / sub-128 => the widest sufficient default "
          "(byte-identical to the untuned construction)."),
      llvm::cl::init("")};
  ::mlir::Pass::Option<std::string> isaVectorHints{
      *this, "isa-vector-hints",
      llvm::cl::desc("Optional probed isa/vector-hint string augmenting the "
                     "-march evidence for the minimum-VLEN derivation."),
      llvm::cl::init("")};
  ::mlir::Pass::Option<std::string> attributionJsonl{
      *this, "attribution-jsonl",
      llvm::cl::desc(
          "Optional path to a [D-4] schedule-stage attribution JSONL sink. When "
          "set, one canonical-JSON record of the typed-flat fill-LMUL decision "
          "(candidates, chosen, reason, minimum_vlen, declared_instance_hash) is "
          "written. Pure side channel: OFF by default, never touches the "
          "constructed IR or the exported object."),
      llvm::cl::init("")};
  ::mlir::Pass::Option<bool> attributionJsonlNoTimestamp{
      *this, "attribution-jsonl-no-timestamp",
      llvm::cl::desc("Emit a fixed sentinel ts instead of the wall-clock time so "
                     "the attribution record is byte-deterministic for lit."),
      llvm::cl::init(false)};
  ::mlir::Pass::Option<bool> numericsReassocOk{
      *this, "numerics-reassoc-ok",
      llvm::cl::desc(
          "[GAP-NUM] the `numerics.reassoc_ok` (kind=policy) capability fact: a "
          "build/permission gate authorizing the [flat-block-dot-fp-fold-oracle "
          "§5] reassociation (relaxed) numerics tier. FAIL-CLOSED: OFF by default "
          "=> the strict §1 byte-exact tier (the paper headline). When ON the "
          "chooseNumericsTier selector unlocks the relaxed tier for fp-order "
          "sensitive kernels; the relaxed body is verified against a declared ULP "
          "upper bound, never §1, and never a headline."),
      llvm::cl::init(false)};

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect, weftexec::WEFTExecDialect,
                    weftrvv::WEFTRVVDialect>();
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
                 "rejected stale weft_rvv.lowering_seed metadata as RVV "
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

    // [D-4] schedule-stage attribution sink: open ONCE, option-gated OFF by
    // default (empty --attribution-jsonl leaves the stream null => no record =>
    // byte-identical construction). Mirrors the exec-stage sink's file handling.
    std::optional<llvm::raw_fd_ostream> attributionFile;
    llvm::raw_ostream *attributionStream = nullptr;
    if (!attributionJsonl.empty()) {
      std::error_code ec;
      attributionFile.emplace(attributionJsonl, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        module.emitError()
            << "RVV block-dot source front door could not open the [D-4] "
               "attribution JSONL sink '"
            << attributionJsonl << "': " << ec.message();
        signalPassFailure();
        return;
      }
      attributionStream = &*attributionFile;
    }

    std::string kernelName = getKernelName(*entry, module);
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    if (mlir::failed(materializeKernel(builder, kernelName, *registry, *entry,
                                       *source, march, isaVectorHints,
                                       numericsReassocOk, attributionStream,
                                       attributionJsonlNoTimestamp))) {
      signalPassFailure();
      return;
    }

    // W2 §(1) B: the constructed body now carries an RVV provider op; materialize
    // the c facts (typed minimum_vlen + support axes) onto it through the ONE
    // shared producer, so a downstream resolveRVVMinimumVLEN reads the provider
    // fact instead of re-parsing -march (I1/I4). This closes the "constructed
    // empty provider" debt: the front door is the legitimate producer.
    (void)materializeRVVProviderCapabilityAxes(module, march, isaVectorHints);

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
        formula_catalog::kMonolithicBlockDotConstruction,
        [entryPtr, registryPtr] {
          return createMaterializeRVVMonolithicBlockDotSourceFrontDoorPass(
              entryPtr, registryPtr);
        },
        SourceFrontDoorPassRegistration::DefaultArtifactFrontDoorPolicy::
            ExplicitOnly));
  }
  return llvm::Error::success();
}

} // namespace weft::plugin::rvv
