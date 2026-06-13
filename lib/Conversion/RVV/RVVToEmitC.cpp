#include "TianChenRV/Conversion/RVV/RVVToEmitC.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Transforms/Passes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv {
namespace transforms {

#define GEN_PASS_DEF_RVVLOWERTOEMITC
#include "TianChenRV/Transforms/Passes.h.inc"

} // namespace transforms
} // namespace tianchenrv

namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace {

namespace tcrvrvv = ::tianchenrv::tcrv::rvv;
namespace emitc = ::mlir::emitc;

//===----------------------------------------------------------------------===//
// Pure C-type derivation (the typed-fact replacement for the legacy string
// re-parser's getEmitCTypeForCType): turn the runtime ABI value's `c_type`
// spelling into a concrete emitc type. Pointers nest; everything else is opaque.
//===----------------------------------------------------------------------===//

mlir::Type emitCTypeForCTypeSpelling(mlir::MLIRContext *context,
                                     llvm::StringRef cType) {
  cType = cType.trim();
  if (cType.ends_with("*")) {
    llvm::StringRef pointee = cType.drop_back().rtrim();
    return emitc::PointerType::get(context,
                                   emitCTypeForCTypeSpelling(context, pointee));
  }
  return emitc::OpaqueType::get(context, cType);
}

//===----------------------------------------------------------------------===//
// Pure SEW/LMUL/dtype intrinsic name mangler. This is the ONE legitimate
// survivor of the legacy `Twine("__riscv_...")+sew+lmul` assembly: a pure
// function over operand TYPE facts (sew/lmul/dtype) and the op mnemonic, with
// no string plan and no operand-expression concatenation. `mnemonic` is the
// RVV intrinsic verb (vsetvl/vle/vse/vadd/vsub/vmul); the caller derives it
// from the typed source op (binary `kind` -> vadd/vsub/vmul, load -> vle,
// store -> vse, setvl -> vsetvl).
//===----------------------------------------------------------------------===//

std::string riscvIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                               llvm::StringRef lmul, llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic;
  if (mnemonic == "vsetvl") {
    // __riscv_vsetvl_e<sew><lmul>
    os << "_e" << sew << lmul;
  } else if (mnemonic == "vle" || mnemonic == "vse") {
    // __riscv_vle<sew>_v_<dtype><lmul> / __riscv_vse<sew>_v_<dtype><lmul>
    os << sew << "_v_" << dtype << lmul;
  } else if (mnemonic == "vlse" || mnemonic == "vsse") {
    // strided load/store: __riscv_vlse<sew>_v_<dtype><lmul> /
    // __riscv_vsse<sew>_v_<dtype><lmul>
    os << sew << "_v_" << dtype << lmul;
  } else if (mnemonic == "vmv_v_x" || mnemonic == "vfmv_v_f") {
    // scalar splat: __riscv_vmv_v_x_<dtype><lmul> (int) /
    // __riscv_vfmv_v_f_<dtype><lmul> (float)
    os << "_" << dtype << lmul;
  } else if (mnemonic == "vfcvt_f_x_v") {
    // int->float convert: __riscv_vfcvt_f_x_v_<dtype><lmul>
    os << "_" << dtype << lmul;
  } else if (mnemonic == "vfmul_vf") {
    // scalar-vector float multiply: __riscv_vfmul_vf_<dtype><lmul>
    os << "_" << dtype << lmul;
  } else if (mnemonic == "vmerge") {
    // masked merge: __riscv_vmerge_vvm_<dtype><lmul>
    os << "_vvm_" << dtype << lmul;
  } else {
    // arithmetic vv form: __riscv_v<op>_vv_<dtype><lmul>
    os << "_vv_" << dtype << lmul;
  }
  os.flush();
  return name;
}

/// The compare-producing mask intrinsic name:
///   __riscv_v<cmp>_vv_<dtype><lmul>_b<maskbits>
/// where maskbits = sew/lmul-derived predicate width (i32/m1 -> b32). This is
/// the same `<dtype><lmul>` + `_b<maskbits>` shape the legacy
/// getElementwiseMaskIntrinsicSuffix produces.
std::string riscvCompareIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                                      llvm::StringRef lmul, llvm::StringRef dtype,
                                      unsigned maskBits) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic << "_vv_" << dtype << lmul << "_b" << maskBits;
  os.flush();
  return name;
}

/// The mask-composition intrinsic name:
///   __riscv_v<op>_mm_b<maskbits>
/// for mask-and (vmand) over two predicate masks of the same (sew, lmul). This
/// mirrors the legacy mask-and callee shape (`__riscv_vmand_mm_b32`).
std::string riscvMaskComposeIntrinsicName(llvm::StringRef mnemonic,
                                          unsigned maskBits) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic << "_mm_b" << maskBits;
  os.flush();
  return name;
}

/// The reduction intrinsic name:
///   __riscv_v<red>_vs_<dtype><lmul>_<dtype>m1
/// (e.g. vredsum/vredmin/vredmax). The reduction always lands its scalar result
/// in lane 0 of an m1 destination vector, so the result suffix is ALWAYS
/// `<dtype>m1` regardless of the source lmul -- byte-identical to the legacy
/// getRVVSelectedBodyReductionIntrinsicForMnemonic
/// (RVVEmitCRoutePlanning.cpp:5087-5090,
/// `__riscv_<mnemonic>_vs_i<sew><lmul>_i<sew>m1`).
std::string riscvReductionIntrinsicName(llvm::StringRef mnemonic, unsigned sew,
                                        llvm::StringRef lmul,
                                        llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic << "_vs_" << dtype << lmul << "_" << dtype
     << "m1";
  os.flush();
  return name;
}

/// The multiply-accumulate intrinsic name:
///   __riscv_vmacc_vv_<dtype><lmul>
/// The fused 3-read vmacc writes into the accumulator vector: the C call order
/// is (accumulator, lhs, rhs, vl). Byte-identical to the legacy
/// deriveMAccIntrinsic (RVVEmitCMAccRouteFamilyPlanOwners.cpp:960-969,
/// `__riscv_vmacc_vv_i<sew><lmul>`), which is i32-only (the legacy derivation
/// returns nullopt for non-SEW32) -- so the caller restricts macc to i32.
std::string riscvMAccIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                   llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vmacc_vv_" << dtype << lmul;
  os.flush();
  return name;
}

/// The element->byte offset scale intrinsic name for an index vector:
///   __riscv_vmul_vx_<idtype><lmul>
/// (e.g. u32/m1 -> vmul_vx_u32m1). The indexed gather/scatter scales the
/// element index vector by the element byte width via a vector-scalar multiply
/// before the ordered indexed memory access -- byte-identical to the legacy
/// indexed-load/store oracle (`__riscv_vmul_vx_u32m1(indices, 4, vl)`).
std::string riscvIndexScaleIntrinsicName(llvm::StringRef idtype,
                                         llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vmul_vx_" << idtype << lmul;
  os.flush();
  return name;
}

/// The ordered indexed memory intrinsic name:
///   __riscv_vloxei<eew>_v_<dtype><lmul> (gather) /
///   __riscv_vsoxei<eew>_v_<dtype><lmul> (scatter)
/// where eew is the index element width (32). The "ox" form is the ordered
/// indexed access -- byte-identical to the legacy indexed-load/store oracle
/// (`__riscv_vloxei32_v_i32m1` / `__riscv_vsoxei32_v_i32m1`).
std::string riscvIndexedMemoryIntrinsicName(llvm::StringRef mnemonic,
                                            unsigned indexEEW,
                                            llvm::StringRef dtype,
                                            llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_" << mnemonic << indexEEW << "_v_" << dtype << lmul;
  os.flush();
  return name;
}

/// The mask-from-buffer compare intrinsic name:
///   __riscv_vmsne_vx_<dtype><lmul>_b<maskbits>
/// The base-memory masked families compute their predicate by loading the mask
/// buffer as a data vector and testing each lane != 0 -- byte-identical to the
/// legacy mask_load oracle (`__riscv_vmsne_vx_i32m1_b32(maskvec, 0, vl)`).
std::string riscvMaskNonzeroIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                          llvm::StringRef dtype,
                                          unsigned maskBits) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vmsne_vx_" << dtype << lmul << "_b" << maskBits;
  os.flush();
  return name;
}

/// The masked unit-stride load intrinsic name:
///   __riscv_vle<sew>_v_<dtype><lmul>_tumu
/// The masked unit-stride load uses the tail-undisturbed mask-undisturbed
/// policy form so inactive/tail lanes keep the passthrough vector --
/// byte-identical to the legacy masked_load oracle
/// (`__riscv_vle32_v_i32m1_tumu`). Call order is (mask, passthrough, ptr, vl).
std::string riscvMaskedLoadIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                         llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vle" << sew << "_v_" << dtype << lmul << "_tumu";
  os.flush();
  return name;
}

/// The masked unit-stride store intrinsic name:
///   __riscv_vse<sew>_v_<dtype><lmul>_m
/// The masked unit-stride store writes only active (mask-true) lanes; inactive
/// and tail lanes keep their memory contents -- byte-identical to the legacy
/// masked_store oracle (`__riscv_vse32_v_i32m1_m`). Call order is
/// (mask, ptr, value, vl).
std::string riscvMaskedStoreIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                          llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vse" << sew << "_v_" << dtype << lmul << "_m";
  os.flush();
  return name;
}

/// The masked byte-strided load intrinsic name:
///   __riscv_vlse<sew>_v_<dtype><lmul>_tumu
/// The masked strided load reads the source at a runtime byte stride but only
/// writes active (mask-true) lanes; inactive/tail lanes keep the passthrough
/// vector via the _tumu policy form -- byte-identical to the legacy computed-
/// mask masked-strided-load oracle (`__riscv_vlse32_v_i32m1_tumu`). Call order
/// is (mask, passthrough, ptr, byteStride, vl).
std::string riscvMaskedStridedLoadIntrinsicName(unsigned sew,
                                                llvm::StringRef lmul,
                                                llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vlse" << sew << "_v_" << dtype << lmul << "_tumu";
  os.flush();
  return name;
}

/// The masked byte-strided store intrinsic name:
///   __riscv_vsse<sew>_v_<dtype><lmul>_m
/// The masked strided store writes only active (mask-true) lanes at a runtime
/// byte stride; inactive/tail lanes keep their memory contents -- byte-identical
/// to the legacy computed-mask masked-strided-store oracle
/// (`__riscv_vsse32_v_i32m1_m`). Call order is (mask, ptr, byteStride, value,
/// vl).
std::string riscvMaskedStridedStoreIntrinsicName(unsigned sew,
                                                 llvm::StringRef lmul,
                                                 llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vsse" << sew << "_v_" << dtype << lmul << "_m";
  os.flush();
  return name;
}

/// The masked indexed (gather) load intrinsic name:
///   __riscv_vluxei<eew>_v_<dtype><lmul>_tumu
/// The masked indexed gather reads scattered elements at byte offsets but only
/// writes active lanes (passthrough preserved on inactive/tail lanes via _tumu)
/// -- byte-identical to the legacy computed-mask indexed-gather oracle
/// (`__riscv_vluxei32_v_i32m1_tumu`). Note the UNORDERED "ux" form (the legacy
/// masked gather is unordered). Call order is
/// (mask, passthrough, data_base, byteIndices, vl).
std::string riscvMaskedIndexedLoadIntrinsicName(unsigned indexEEW,
                                                llvm::StringRef dtype,
                                                llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vluxei" << indexEEW << "_v_" << dtype << lmul << "_tumu";
  os.flush();
  return name;
}

/// The masked indexed (scatter) store intrinsic name:
///   __riscv_vsoxei<eew>_v_<dtype><lmul>_m
/// The masked indexed scatter writes active lanes to scattered byte offsets;
/// inactive/tail lanes are skipped (no write) -- byte-identical to the legacy
/// computed-mask indexed-scatter oracle (`__riscv_vsoxei32_v_i32m1_m`). Note the
/// ORDERED "ox" form. Call order is (mask, dst_base, byteIndices, value, vl).
std::string riscvMaskedIndexedStoreIntrinsicName(unsigned indexEEW,
                                                 llvm::StringRef dtype,
                                                 llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vsoxei" << indexEEW << "_v_" << dtype << lmul << "_m";
  os.flush();
  return name;
}

//===----------------------------------------------------------------------===//
// Segment2 (interleaved 2-field) intrinsic + tuple-type name manglers. The
// segment2 memory family carries a TUPLE C type (`vint<sew>m<lmul>x2_t`) that
// the field vectors are packed into (interleave/store) and extracted from
// (deinterleave/load). The tuple type and segment intrinsics are byte-identical
// to the legacy segment2 string-plan oracle:
//   tuple type:        vint<sew>m<lmul>x2_t        (e.g. vint32m1x2_t)
//   tuple create:      __riscv_vcreate_v_<dtype><lmul>x2(f0, f1)
//   field extract:     __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, idx)
//   segment load:      __riscv_vlseg2e<sew>_v_<dtype><lmul>x2(ptr, vl)
//   segment store:     __riscv_vsseg2e<sew>_v_<dtype><lmul>x2(ptr, tuple, vl)
//   masked seg load:   __riscv_vlseg2e<sew>_v_<dtype><lmul>x2_tumu(mask, pass, ptr, vl)
//   masked seg store:  __riscv_vsseg2e<sew>_v_<dtype><lmul>x2_m(mask, ptr, tuple, vl)
//===----------------------------------------------------------------------===//

/// The segment2 tuple C type spelling: `vint<sew>m<lmul>x2_t` (the bounded i32
/// slice -> vint32m1x2_t). Only the signed-integer grid the slice uses is
/// named; an element the converter cannot name returns "" so the caller fails
/// the match and the body falls back.
std::string riscvSegment2TupleCType(unsigned sew, llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "vint" << sew << lmul << "x2_t";
  os.flush();
  return name;
}

/// The segment2 tuple-create intrinsic name:
///   __riscv_vcreate_v_<dtype><lmul>x2
/// Packs the two field vectors into one segment2 tuple value (the
/// interleave/store and masked-store/load passthrough path). Byte-identical to
/// the legacy segment2 oracle (`__riscv_vcreate_v_i32m1x2`).
std::string riscvSegment2TupleCreateIntrinsicName(llvm::StringRef dtype,
                                                  llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vcreate_v_" << dtype << lmul << "x2";
  os.flush();
  return name;
}

/// The segment2 field-extract intrinsic name:
///   __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>
/// Extracts one field vector (index 0 or 1) from a segment2 tuple value (the
/// deinterleave/load path). Byte-identical to the legacy segment2 oracle
/// (`__riscv_vget_v_i32m1x2_i32m1`).
std::string riscvSegment2FieldExtractIntrinsicName(llvm::StringRef dtype,
                                                   llvm::StringRef lmul) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vget_v_" << dtype << lmul << "x2_" << dtype << lmul;
  os.flush();
  return name;
}

/// The segment2 unit-stride interleaved load intrinsic name:
///   __riscv_vlseg2e<sew>_v_<dtype><lmul>x2
/// Loads the two interleaved fields from the interleaved source into one
/// segment2 tuple value. Byte-identical to the legacy segment2 oracle
/// (`__riscv_vlseg2e32_v_i32m1x2`). Call order is (ptr, vl).
std::string riscvSegment2LoadIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                           llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vlseg2e" << sew << "_v_" << dtype << lmul << "x2";
  os.flush();
  return name;
}

/// The segment2 unit-stride interleaved store intrinsic name:
///   __riscv_vsseg2e<sew>_v_<dtype><lmul>x2
/// Stores one segment2 tuple value into the interleaved destination.
/// Byte-identical to the legacy segment2 oracle
/// (`__riscv_vsseg2e32_v_i32m1x2`). Call order is (ptr, tuple, vl).
std::string riscvSegment2StoreIntrinsicName(unsigned sew, llvm::StringRef lmul,
                                            llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vsseg2e" << sew << "_v_" << dtype << lmul << "x2";
  os.flush();
  return name;
}

/// The masked segment2 interleaved load intrinsic name:
///   __riscv_vlseg2e<sew>_v_<dtype><lmul>x2_tumu
/// The masked segment2 load reads the two interleaved fields but only writes
/// active (mask-true) lanes; inactive/tail lanes keep the passthrough tuple via
/// the _tumu policy form -- byte-identical to the legacy computed-mask segment2
/// load oracle (`__riscv_vlseg2e32_v_i32m1x2_tumu`). Call order is
/// (mask, passthrough_tuple, ptr, vl).
std::string riscvMaskedSegment2LoadIntrinsicName(unsigned sew,
                                                 llvm::StringRef lmul,
                                                 llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vlseg2e" << sew << "_v_" << dtype << lmul << "x2_tumu";
  os.flush();
  return name;
}

/// The masked segment2 interleaved store intrinsic name:
///   __riscv_vsseg2e<sew>_v_<dtype><lmul>x2_m
/// The masked segment2 store writes only active (mask-true) lanes; inactive/tail
/// lanes keep their interleaved memory contents -- byte-identical to the legacy
/// computed-mask segment2 store oracle (`__riscv_vsseg2e32_v_i32m1x2_m`). Call
/// order is (mask, ptr, tuple, vl).
std::string riscvMaskedSegment2StoreIntrinsicName(unsigned sew,
                                                  llvm::StringRef lmul,
                                                  llvm::StringRef dtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vsseg2e" << sew << "_v_" << dtype << lmul << "x2_m";
  os.flush();
  return name;
}

/// Predicate mask width for an (sew, lmul) config, mirroring the legacy
/// getElementwiseMaskIntrinsicSuffix maskBits table. Returns 0 for an
/// unsupported pair (so the caller fails the match and falls back).
unsigned maskWidthForConfig(unsigned sew, llvm::StringRef lmul) {
  if (sew == 32 && lmul == "m1")
    return 32;
  if (sew == 32 && lmul == "m2")
    return 16;
  if (sew == 64 && lmul == "m1")
    return 64;
  if (sew == 64 && lmul == "m2")
    return 32;
  return 0;
}

/// The C dtype token ("i32"/"i64"/"f32") for the vector element, used by the
/// load/store/arithmetic intrinsic suffix and the `vint<sew>m<lmul>_t` /
/// `vfloat<sew>m<lmul>_t` opaque type.
llvm::StringRef vectorDType(tcrvrvv::VectorType type) {
  if (type.getElementType().isSignlessInteger(8))
    return "i8";
  if (type.getElementType().isSignlessInteger(16))
    return "i16";
  if (type.getElementType().isSignlessInteger(32))
    return "i32";
  if (type.getElementType().isSignlessInteger(64))
    return "i64";
  if (type.getElementType().isF32())
    return "f32";
  return "";
}

/// True for a floating-point vector element (the f32 compare/select/dequant
/// family uses the f-prefixed RVV intrinsics).
bool isFloatVector(tcrvrvv::VectorType type) {
  return llvm::isa<mlir::FloatType>(type.getElementType());
}

/// The C scalar element spelling a memory buffer of `type` MUST point at, for
/// the load/store intrinsics to be type-correct: i32 -> "int32_t", i64 ->
/// "int64_t", f32 -> "float". Returns "" for an element the converter cannot
/// name (so the caller fails the match).
llvm::StringRef vectorScalarCType(tcrvrvv::VectorType type) {
  if (type.getElementType().isSignlessInteger(8))
    return "int8_t";
  if (type.getElementType().isSignlessInteger(16))
    return "int16_t";
  if (type.getElementType().isSignlessInteger(32))
    return "int32_t";
  if (type.getElementType().isSignlessInteger(64))
    return "int64_t";
  if (type.getElementType().isF32())
    return "float";
  return "";
}

/// True iff `bufferValue` is an emitc pointer whose pointee opaque C type names
/// the scalar element of `vectorType`. The runtime ABI value's `c_type`
/// (e.g. "const int64_t *") becomes the function parameter pointer type; the
/// load/store intrinsic width is driven by the typed VECTOR element. If the two
/// disagree (e.g. a `const int32_t *` buffer feeding a `vle64`/i64 load), the
/// generated C dereferences the pointer at the wrong element width — a malformed
/// body the legacy path rejected. Reject it here too (return false -> the caller
/// fails the match and the body falls back unchanged) rather than emit broken C.
bool bufferPointeeMatchesVectorElement(mlir::Value bufferValue,
                                       tcrvrvv::VectorType vectorType) {
  auto pointerType =
      llvm::dyn_cast<emitc::PointerType>(bufferValue.getType());
  if (!pointerType)
    return false;
  auto pointeeOpaque =
      llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
  if (!pointeeOpaque)
    return false;
  llvm::StringRef scalar = vectorScalarCType(vectorType);
  if (scalar.empty())
    return false;
  // The pointee spelling may carry qualifiers (e.g. "const int32_t"); require
  // the exact scalar token to appear so "int32_t" does not match "int64_t".
  return pointeeOpaque.getValue().contains(scalar);
}

//===----------------------------------------------------------------------===//
// Provenance comments, byte-identical to the legacy materializer
// (TCRVEmitCLowerableMaterializer.cpp makeRouteSourceProvenanceComment /
// makeStepProvenanceComment). Reproduced so the rendered C carries the same
// `// tcrv_emitc.*` lines and stays byte-equivalent to the hardware-validated
// golden.
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kOpInterface = "TCRVEmitCLowerableOpInterface";

std::string routeSourceComment(llvm::StringRef opName, llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.route_source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

std::string stepComment(llvm::StringRef opName, llvm::StringRef role,
                        llvm::StringRef callee) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << opName << " role=" << role
     << " op_interface=" << kOpInterface << " callee=" << callee;
  os.flush();
  return text;
}

//===----------------------------------------------------------------------===//
// VariantOp -> emitc.func driver.
//
// The runtime ABI values, setvl, the with_vl scope, and the body ops all live
// as siblings/children inside one tcrv.exec.variant region. The runtime ABI
// values become the C function parameters (not in-body values), so the whole
// variant body is restructured into a single top-level emitc.func with the
// with_vl scope rendered as an emitc.for loop. The per-beachhead-op emit logic
// (setvl/load/binary/store, the for-loop, the AVL `n - i` and pointer `base + i`
// arithmetic) is owned here and is sourced ENTIRELY from the typed op SSA
// Values + typed `!tcrv_rvv.*` types — never from a string plan.
//===----------------------------------------------------------------------===//

struct AbiParam {
  tcrvrvv::RuntimeABIValueOp op;
  std::string cType;
  mlir::Type emitcType;
};

class VariantToEmitCFunc final
    : public mlir::OpConversionPattern<tcrv::exec::VariantOp> {
public:
  VariantToEmitCFunc(const mlir::TypeConverter &typeConverter,
                     mlir::MLIRContext *context)
      : mlir::OpConversionPattern<tcrv::exec::VariantOp>(typeConverter,
                                                         context) {}

  mlir::LogicalResult
  matchAndRewrite(tcrv::exec::VariantOp variant, OpAdaptor /*adaptor*/,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = variant.getContext();

    // Only the selected lowering boundary (a variant carrying a with_vl scope)
    // is a beachhead body. Variants without a with_vl scope (e.g. the scalar
    // fallback) are left for the legacy path / other families.
    tcrvrvv::WithVLOp scope;
    for (mlir::Operation &op : variant.getBody().front()) {
      if (auto withVL = llvm::dyn_cast<tcrvrvv::WithVLOp>(op)) {
        scope = withVL;
        break;
      }
    }
    if (!scope)
      return rewriter.notifyMatchFailure(variant, "no with_vl scope to lower");

    tcrvrvv::SetVLOp preLoopSetVL =
        scope.getVl().getDefiningOp<tcrvrvv::SetVLOp>();
    if (!preLoopSetVL)
      return rewriter.notifyMatchFailure(
          variant, "with_vl vl token must be defined by tcrv_rvv.setvl");
    mlir::Value avlSource = preLoopSetVL.getAvl();
    auto avlAbi = avlSource.getDefiningOp<tcrvrvv::RuntimeABIValueOp>();
    if (!avlAbi)
      return rewriter.notifyMatchFailure(
          variant, "setvl avl must be a runtime ABI value");

    // Tail/mask policy scope guard. The converter emits tail/mask-AGNOSTIC RVV
    // intrinsic forms (e.g. __riscv_vadd_vv_i32m1) and does NOT model an
    // undisturbed destination for compute/load ops (which would need a
    // passthrough / `_tu`/`_tum` form). A body that requests an undisturbed tail
    // or mask policy must NOT be silently lowered as agnostic — that would emit
    // semantically wrong C. Fail the match so the unsupported-policy body falls
    // through unchanged (the legacy path rejected it explicitly; the conversion
    // simply does not take it over).
    //
    // EXCEPTION — the masked unit-store family. A pure masked-store body
    // (mask_load + payload load + masked_store, the ONLY store being a
    // masked_store) carries an undisturbed SCOPE policy because the masked store
    // skips inactive/tail lanes and leaves their destination memory contents
    // untouched. The masked-store `_m` intrinsic form HONORS that undisturbed
    // semantics by construction, so this specific body shape is correctly
    // lowered even under an undisturbed scope policy. Allow undisturbed ONLY for
    // that shape; every other op (which would need an agnostic or `_tu`/`_tum`
    // form) still forces the agnostic-only refusal so no compute/load is
    // silently mislowered.
    //
    // EXCEPTION 2 — the computed-mask masked-store family. A body whose only
    // store is a single tcrv_rvv.masked_store but whose mask is computed by a
    // tcrv_rvv.compare in the same scope (the runtime-scalar / vector
    // computed-mask store shape: load[+splat] -> compare -> masked_store) also
    // carries an undisturbed scope policy that the masked-store `_m` form
    // honors. The compare/splat/load steps emit agnostic intrinsics whose
    // results are fully defined over the active VL, so the undisturbed semantics
    // live entirely in the `_m` store -- correctly lowered. Allow undisturbed
    // for that shape too; anything else still forces the agnostic-only refusal.
    tcrvrvv::PolicyAttr policy = preLoopSetVL.getPolicy();
    if (policy.getTail() != tcrvrvv::TailPolicy::Agnostic ||
        policy.getMask() != tcrvrvv::MaskPolicy::Agnostic) {
      if (!isPureMaskedStoreBody(scope) &&
          !isComputedMaskMaskedStoreBody(scope))
        return rewriter.notifyMatchFailure(
            variant, "only tail/mask-agnostic policy is convertible (except a "
                     "pure masked-store body under undisturbed policy)");
    }

    // Collect the runtime ABI values as ordered function parameters.
    llvm::SmallVector<AbiParam, 4> params;
    for (mlir::Operation &op : variant.getBody().front()) {
      if (auto abi = llvm::dyn_cast<tcrvrvv::RuntimeABIValueOp>(op)) {
        AbiParam param;
        param.op = abi;
        param.cType = abi.getCType().str();
        param.emitcType = emitCTypeForCTypeSpelling(context, param.cType);
        params.push_back(param);
      }
    }
    if (params.empty())
      return rewriter.notifyMatchFailure(variant, "no runtime ABI parameters");

    // Duplicate runtime ABI c_name guard. Every runtime ABI value becomes a
    // distinct C function parameter, so two ABI values sharing a c_name is a
    // malformed callable contract the legacy route path rejects (e.g. an
    // indexed gather whose `data` and `index` buffers are both named "data").
    // The conversion renders parameters positionally (vN), so it would silently
    // accept the collision and bypass that rejection; refuse a duplicate c_name
    // so the malformed body falls back to the legacy validator.
    {
      llvm::StringSet<> seenCNames;
      for (const AbiParam &param : params) {
        tcrvrvv::RuntimeABIValueOp abiOp = param.op;
        if (!seenCNames.insert(abiOp.getCName()).second)
          return rewriter.notifyMatchFailure(
              variant, "duplicate runtime ABI c_name (malformed callable "
                       "contract; legacy validator owns the diagnostic)");
      }
    }

    // Derive the function name exactly as the export path does:
    // tcrv_emitc_<kernel>_<variant>.
    auto kernel = variant->getParentOfType<tcrv::exec::KernelOp>();
    if (!kernel)
      return rewriter.notifyMatchFailure(variant, "variant has no kernel");
    std::string functionName =
        ("tcrv_emitc_" + kernel.getSymName() + "_" + variant.getSymName())
            .str();

    // exec-binding contract gate (family-generic). When the selected variant
    // requests exec ABI bindings (`tcrv_rvv.require_exec_abi_bindings = true`),
    // every runtime ABI value MUST carry an `exec_binding` symbol to its
    // tcrv.exec ABI declaration -- the legacy route-family path rejects a
    // missing binding. If a body that opts into the contract has an unbound ABI
    // value, this conversion must NOT take it over (it would materialize C
    // without honoring the contract the legacy validator enforces). Fall back so
    // the legacy validator still rejects it.
    if (auto requireBindings = variant->getAttrOfType<mlir::BoolAttr>(
            "tcrv_rvv.require_exec_abi_bindings");
        requireBindings && requireBindings.getValue()) {
      for (const AbiParam &param : params)
        if (!param.op->hasAttr("exec_binding"))
          return rewriter.notifyMatchFailure(
              variant, "runtime ABI value missing required exec_binding "
                       "(contract enforced by the legacy validator)");
    }

    // Capability config gate (family-generic, I1-honoring). The selected
    // variant's `requires` names the RVV capability provider; that provider is a
    // queryable tcrv.exec.capability / tcrv.exec.target MLIR object that may
    // declare `supported_sew` / `supported_lmul`. If present and they EXCLUDE
    // the typed body's (sew, lmul), the capability gates this body out -- the
    // legacy route-family path rejects it ("supported_sew fact ... does not
    // include typed body SEW"). The conversion must respect that legality gate
    // and fall back, not materialize C the capability forbids. Reading the attrs
    // straight off the provider op keeps capability the authority (no string
    // model). When the provider declares no restriction the gate is silent.
    {
      unsigned bodySEW = static_cast<unsigned>(preLoopSetVL.getSew());
      llvm::StringRef bodyLMUL = preLoopSetVL.getLmul();
      if (mlir::failed(checkCapabilityConfigGate(rewriter, variant, kernel,
                                                 bodySEW, bodyLMUL)))
        return mlir::failure();
    }

    // Build a standalone top-level emitc module: the standard headers the RVV
    // intrinsic body needs, then the function. This mirrors the legacy
    // materializer's module shape (includes + func) so the rendered C is
    // byte-equivalent to the hardware-validated golden.
    mlir::Location loc = variant.getLoc();
    auto module = variant->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder::InsertionGuard moduleGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    for (llvm::StringRef header :
         {"stddef.h", "stdint.h", "riscv_vector.h"})
      rewriter.create<emitc::IncludeOp>(loc, header,
                                        /*is_standard_include=*/true);

    rewriter.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 4> paramTypes;
    for (const AbiParam &param : params)
      paramTypes.push_back(param.emitcType);
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, /*results=*/{});

    llvm::SmallVector<mlir::NamedAttribute, 1> funcAttrs;
    funcAttrs.push_back(rewriter.getNamedAttr(
        "specifiers", rewriter.getStrArrayAttr({"extern", "\"C\""})));
    auto func = rewriter.create<emitc::FuncOp>(loc, functionName, functionType,
                                               funcAttrs);
    mlir::Block *entry = func.addEntryBlock();

    // Map each runtime ABI value SSA result to its function block argument.
    llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
    for (auto [index, param] : llvm::enumerate(params))
      valueMap[param.op.getResult()] = entry->getArgument(index);

    // segment2 deinterleave: maps a segment2_load field result SSA value to the
    // (loaded tuple value, field index). The downstream tcrv_rvv.move that
    // sources the field emits the __riscv_vget extract from this tuple.
    llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
        segmentFieldMap;

    rewriter.setInsertionPointToStart(entry);

    // Vector/VL/AVL config facts from the typed scope and types.
    unsigned sew = static_cast<unsigned>(preLoopSetVL.getSew());
    llvm::StringRef lmul = preLoopSetVL.getLmul();
    mlir::Type sizeType = emitc::OpaqueType::get(context, "size_t");

    mlir::Value avlArg = valueMap.lookup(avlAbi.getResult());

    // Scope provenance: // tcrv_emitc.route_source_op=tcrv_rvv.with_vl role=scope
    rewriter.create<emitc::VerbatimOp>(
        loc, routeSourceComment(scope.getTCRVEmitCLowerableSourceOpName(),
                                scope.getTCRVEmitCLowerableSourceRole()));

    // Pre-loop full-chunk setvl: __riscv_vsetvl_e<sew><lmul>(n).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", sew, lmul, "");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                         preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                         setvlCallee));
    mlir::Value vlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         setvlCallee, mlir::ValueRange{avlArg})
            .getResult(0);

    // Standalone reduction pre-loop seed: out[0] = acc[0]. Runs BEFORE the loop
    // (between the pre-loop full-chunk setvl and the for-loop), seeding the
    // scalar accumulator carried through the output cell across runtime chunks.
    bool standaloneReduction = isStandaloneReductionBody(scope);
    if (standaloneReduction) {
      if (mlir::failed(
              emitStandaloneReductionPreLoopSeed(rewriter, loc, scope, valueMap)))
        return mlir::failure();
    }

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();

    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // Remaining-AVL setvl: size_t v = n - i; __riscv_vsetvl_e...(v).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                           preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                           setvlCallee));
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);

      // Build the emit order. Bodies emit in IR order, EXCEPT the pure
      // masked-store family: the legacy route path emits the payload `load`
      // BEFORE the `mask_load` two-step (its realized IR carries them in the
      // opposite order). Reorder ONLY that shape so the rendered C stays
      // byte-identical to the legacy oracle; every other body keeps IR order.
      llvm::SmallVector<mlir::Operation *, 8> orderedOps;
      for (mlir::Operation &op : scope.getBody().front())
        orderedOps.push_back(&op);
      if (isPureMaskedStoreBody(scope)) {
        // Move the single plain payload load ahead of the mask_load.
        auto maskLoadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<tcrvrvv::MaskLoadOp>(op);
        });
        auto loadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<tcrvrvv::LoadOp>(op);
        });
        if (maskLoadIt != orderedOps.end() && loadIt != orderedOps.end() &&
            loadIt > maskLoadIt) {
          mlir::Operation *loadOp = *loadIt;
          orderedOps.erase(loadIt);
          orderedOps.insert(
              llvm::find_if(orderedOps,
                            [](mlir::Operation *op) {
                              return llvm::isa<tcrvrvv::MaskLoadOp>(op);
                            }),
              loadOp);
        }
      }

      // Computed-mask indexed memory ordering. The string-plan owner (the
      // byte-order the harness ordered-token validator depends on) emits the
      // index_load + its element->byte scale EARLY -- right after the first
      // compare-LHS load, before the splat / remaining loads / compare. The
      // realized IR instead carries index_load after those loads. Reorder ONLY a
      // computed-mask indexed body (one carrying a masked_indexed load/store) so
      // the rendered C keeps the legacy index-early order; every other body
      // keeps IR order. The byte-scale is emitted at index_load time (see
      // emitIndexLoad) so it immediately follows the index_load in that order.
      bool hasMaskedIndexed = llvm::any_of(orderedOps, [](mlir::Operation *op) {
        return llvm::isa<tcrvrvv::MaskedIndexedLoadOp,
                         tcrvrvv::MaskedIndexedStoreOp>(op);
      });
      if (hasMaskedIndexed) {
        auto indexLoadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<tcrvrvv::IndexLoadOp>(op);
        });
        auto firstLoadIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<tcrvrvv::LoadOp>(op);
        });
        if (indexLoadIt != orderedOps.end() &&
            firstLoadIt != orderedOps.end() && indexLoadIt > firstLoadIt) {
          mlir::Operation *indexLoadOp = *indexLoadIt;
          orderedOps.erase(indexLoadIt);
          // Re-find the first load (the erase may have shifted iterators) and
          // insert the index_load immediately after it.
          auto insertAfter = llvm::find_if(orderedOps, [](mlir::Operation *op) {
            return llvm::isa<tcrvrvv::LoadOp>(op);
          });
          orderedOps.insert(std::next(insertAfter), indexLoadOp);
        }
      }

      // Computed-mask widening dot-reduce ordering. The string-plan owner emits
      // the compare-produced mask IMMEDIATELY after its two compare-input loads
      // (the byte order the harness ordered-token validator depends on), BEFORE
      // the two dot-product input loads. The realized IR instead carries the
      // compare AFTER all four loads (its operands and the dot inputs are
      // siblings). Reorder ONLY a computed-mask dot-reduce body (one carrying a
      // masked_widening_dot_reduce) so the rendered C keeps the legacy
      // mask-early order; every other body keeps IR order. The compare is moved
      // to immediately follow the last of its own (load/strided_load) operands.
      bool hasMaskedDotReduce = llvm::any_of(orderedOps, [](mlir::Operation *op) {
        return llvm::isa<tcrvrvv::MaskedWideningDotReduceOp>(op);
      });
      if (hasMaskedDotReduce) {
        auto compareIt = llvm::find_if(orderedOps, [](mlir::Operation *op) {
          return llvm::isa<tcrvrvv::CompareOp>(op);
        });
        if (compareIt != orderedOps.end()) {
          mlir::Operation *compareOp = *compareIt;
          // Find the last position among the compare's operand-defining ops in
          // the current order; the compare reorders to immediately after it.
          long lastOperandPos = -1;
          for (mlir::Value operand : compareOp->getOperands()) {
            mlir::Operation *def = operand.getDefiningOp();
            if (!def)
              continue;
            auto pos = llvm::find(orderedOps, def);
            if (pos != orderedOps.end())
              lastOperandPos = std::max<long>(
                  lastOperandPos, std::distance(orderedOps.begin(), pos));
          }
          if (lastOperandPos >= 0) {
            orderedOps.erase(compareIt);
            // The erase shifts positions after compareIt; recompute the insert
            // point by re-finding the last operand def.
            mlir::Operation *anchor = nullptr;
            long anchorPos = -1;
            for (mlir::Value operand : compareOp->getOperands()) {
              mlir::Operation *def = operand.getDefiningOp();
              if (!def)
                continue;
              auto pos = llvm::find(orderedOps, def);
              if (pos != orderedOps.end()) {
                long p = std::distance(orderedOps.begin(), pos);
                if (p > anchorPos) {
                  anchorPos = p;
                  anchor = def;
                }
              }
            }
            if (anchor) {
              auto anchorIt = llvm::find(orderedOps, anchor);
              orderedOps.insert(std::next(anchorIt), compareOp);
            } else {
              orderedOps.insert(orderedOps.begin(), compareOp);
            }
          }
        }
      }

      // The standalone reduction in-loop running seed reads back out[0]; the
      // store cell (the output buffer base) is the body's store target. Resolve
      // it once so the standalone reduce ops can read/seed it.
      mlir::Value standaloneOutBuffer;
      if (standaloneReduction) {
        for (mlir::Operation *opPtr : orderedOps)
          if (auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(opPtr))
            standaloneOutBuffer = valueMap.lookup(store.getBuffer());
        if (!standaloneOutBuffer)
          return rewriter.notifyMatchFailure(
              variant, "standalone reduction output buffer unmapped");
      }

      // Convert each body op in emit order. Body holds the typed dataflow ops
      // for the elementwise/memory families.
      for (mlir::Operation *opPtr : orderedOps) {
        mlir::Operation &op = *opPtr;
        if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
          if (mlir::failed(emitLoad(rewriter, loc, load, valueMap, inductionVar,
                                    bodyVL)))
            return mlir::failure();
        } else if (auto broadcast =
                       llvm::dyn_cast<tcrvrvv::BroadcastLoadOp>(op)) {
          if (mlir::failed(emitBroadcastLoad(rewriter, loc, broadcast, valueMap,
                                             bodyVL)))
            return mlir::failure();
        } else if (auto splat = llvm::dyn_cast<tcrvrvv::SplatOp>(op)) {
          if (mlir::failed(emitSplat(rewriter, loc, splat, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto stridedLoad =
                       llvm::dyn_cast<tcrvrvv::StridedLoadOp>(op)) {
          if (mlir::failed(emitStridedLoad(rewriter, loc, stridedLoad, valueMap,
                                           inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto indexLoad = llvm::dyn_cast<tcrvrvv::IndexLoadOp>(op)) {
          if (mlir::failed(emitIndexLoad(rewriter, loc, indexLoad, valueMap,
                                         inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto indexedLoad =
                       llvm::dyn_cast<tcrvrvv::IndexedLoadOp>(op)) {
          if (mlir::failed(emitIndexedLoad(rewriter, loc, indexedLoad, valueMap,
                                           bodyVL)))
            return mlir::failure();
        } else if (auto maskLoad = llvm::dyn_cast<tcrvrvv::MaskLoadOp>(op)) {
          if (mlir::failed(emitMaskLoad(rewriter, loc, maskLoad, valueMap,
                                        inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedLoad =
                       llvm::dyn_cast<tcrvrvv::MaskedLoadOp>(op)) {
          if (mlir::failed(emitMaskedLoad(rewriter, loc, maskedLoad, valueMap,
                                          inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedStridedLoad =
                       llvm::dyn_cast<tcrvrvv::MaskedStridedLoadOp>(op)) {
          if (mlir::failed(emitMaskedStridedLoad(rewriter, loc,
                                                 maskedStridedLoad, valueMap,
                                                 inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedIndexedLoad =
                       llvm::dyn_cast<tcrvrvv::MaskedIndexedLoadOp>(op)) {
          if (mlir::failed(emitMaskedIndexedLoad(rewriter, loc,
                                                 maskedIndexedLoad, valueMap,
                                                 bodyVL)))
            return mlir::failure();
        } else if (auto move = llvm::dyn_cast<tcrvrvv::MoveOp>(op)) {
          if (mlir::failed(
                  emitMove(rewriter, loc, move, valueMap, segmentFieldMap)))
            return mlir::failure();
        } else if (auto segLoad =
                       llvm::dyn_cast<tcrvrvv::Segment2LoadOp>(op)) {
          if (mlir::failed(emitSegment2Load(rewriter, loc, segLoad, valueMap,
                                            segmentFieldMap, inductionVar,
                                            bodyVL)))
            return mlir::failure();
        } else if (auto segStore =
                       llvm::dyn_cast<tcrvrvv::Segment2StoreOp>(op)) {
          if (mlir::failed(emitSegment2Store(rewriter, loc, segStore, valueMap,
                                             inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedSegLoad =
                       llvm::dyn_cast<tcrvrvv::MaskedSegment2LoadOp>(op)) {
          if (mlir::failed(emitMaskedSegment2Load(rewriter, loc, maskedSegLoad,
                                                  valueMap, inductionVar,
                                                  bodyVL)))
            return mlir::failure();
        } else if (auto maskedSegStore =
                       llvm::dyn_cast<tcrvrvv::MaskedSegment2StoreOp>(op)) {
          if (mlir::failed(emitMaskedSegment2Store(rewriter, loc,
                                                   maskedSegStore, valueMap,
                                                   inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto compare = llvm::dyn_cast<tcrvrvv::CompareOp>(op)) {
          if (mlir::failed(emitCompare(rewriter, loc, compare, valueMap,
                                       bodyVL)))
            return mlir::failure();
        } else if (auto maskedBinary =
                       llvm::dyn_cast<tcrvrvv::MaskedBinaryOp>(op)) {
          if (mlir::failed(emitMaskedBinary(rewriter, loc, maskedBinary,
                                            valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto maskedMacc =
                       llvm::dyn_cast<tcrvrvv::MaskedMAccOp>(op)) {
          if (mlir::failed(emitMaskedMAcc(rewriter, loc, maskedMacc, valueMap,
                                          bodyVL)))
            return mlir::failure();
        } else if (auto macc = llvm::dyn_cast<tcrvrvv::MAccOp>(op)) {
          if (mlir::failed(emitMAcc(rewriter, loc, macc, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto maskAnd = llvm::dyn_cast<tcrvrvv::MaskAndOp>(op)) {
          if (mlir::failed(emitMaskAnd(rewriter, loc, maskAnd, valueMap,
                                       bodyVL)))
            return mlir::failure();
        } else if (auto dequantize =
                       llvm::dyn_cast<tcrvrvv::DequantizeOp>(op)) {
          if (mlir::failed(emitDequantize(rewriter, loc, dequantize, valueMap,
                                          bodyVL)))
            return mlir::failure();
        } else if (auto select = llvm::dyn_cast<tcrvrvv::SelectOp>(op)) {
          if (mlir::failed(emitSelect(rewriter, loc, select, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto reduce = llvm::dyn_cast<tcrvrvv::ReduceOp>(op)) {
          if (mlir::failed(emitReduce(rewriter, loc, reduce, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto standaloneReduce =
                       llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
          if (mlir::failed(emitStandaloneReduce(rewriter, loc, standaloneReduce,
                                                valueMap, standaloneOutBuffer,
                                                bodyVL)))
            return mlir::failure();
        } else if (auto maskedStandaloneReduce =
                       llvm::dyn_cast<tcrvrvv::MaskedStandaloneReduceOp>(op)) {
          if (mlir::failed(emitMaskedStandaloneReduce(
                  rewriter, loc, maskedStandaloneReduce, valueMap,
                  standaloneOutBuffer, bodyVL)))
            return mlir::failure();
        } else if (auto wproduct =
                       llvm::dyn_cast<tcrvrvv::WideningProductOp>(op)) {
          if (mlir::failed(emitWideningProduct(rewriter, loc, wproduct,
                                               valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto wmacc =
                       llvm::dyn_cast<tcrvrvv::WideningMAccOp>(op)) {
          if (mlir::failed(
                  emitWideningMAcc(rewriter, loc, wmacc, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto dotReduce =
                       llvm::dyn_cast<tcrvrvv::WideningDotReduceOp>(op)) {
          if (mlir::failed(emitWideningDotReduce(rewriter, loc, dotReduce,
                                                 valueMap, standaloneOutBuffer,
                                                 bodyVL)))
            return mlir::failure();
        } else if (auto maskedDotReduce =
                       llvm::dyn_cast<tcrvrvv::MaskedWideningDotReduceOp>(op)) {
          if (mlir::failed(emitMaskedWideningDotReduce(
                  rewriter, loc, maskedDotReduce, valueMap, standaloneOutBuffer,
                  bodyVL)))
            return mlir::failure();
        } else if (auto binary = llvm::dyn_cast<tcrvrvv::BinaryOp>(op)) {
          if (mlir::failed(emitBinary(rewriter, loc, binary, valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto stridedStore =
                       llvm::dyn_cast<tcrvrvv::StridedStoreOp>(op)) {
          if (mlir::failed(emitStridedStore(rewriter, loc, stridedStore,
                                            valueMap, inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto indexedStore =
                       llvm::dyn_cast<tcrvrvv::IndexedStoreOp>(op)) {
          if (mlir::failed(emitIndexedStore(rewriter, loc, indexedStore,
                                            valueMap, bodyVL)))
            return mlir::failure();
        } else if (auto maskedStore =
                       llvm::dyn_cast<tcrvrvv::MaskedStoreOp>(op)) {
          if (mlir::failed(emitMaskedStore(rewriter, loc, maskedStore, valueMap,
                                           inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedStridedStore =
                       llvm::dyn_cast<tcrvrvv::MaskedStridedStoreOp>(op)) {
          if (mlir::failed(emitMaskedStridedStore(rewriter, loc,
                                                  maskedStridedStore, valueMap,
                                                  inductionVar, bodyVL)))
            return mlir::failure();
        } else if (auto maskedIndexedStore =
                       llvm::dyn_cast<tcrvrvv::MaskedIndexedStoreOp>(op)) {
          if (mlir::failed(emitMaskedIndexedStore(rewriter, loc,
                                                  maskedIndexedStore, valueMap,
                                                  bodyVL)))
            return mlir::failure();
        } else if (auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
          // The standalone reduction stores its lane-0 scalar result back to the
          // output buffer BASE (no `+ i` offset, VL=1): the running scalar lives
          // in out[0] across chunks. Mirror the legacy scalar-output store.
          if (isStandaloneReductionOp(store.getValue().getDefiningOp())) {
            auto resultVecType = llvm::dyn_cast<tcrvrvv::VectorType>(
                store.getValue().getType());
            mlir::Value value = valueMap.lookup(store.getValue());
            if (!resultVecType || !value || !standaloneOutBuffer)
              return rewriter.notifyMatchFailure(
                  store, "standalone reduction store operand unmapped");
            if (mlir::failed(emitStandaloneReductionScalarStore(
                    rewriter, loc, store, standaloneOutBuffer, value,
                    resultVecType)))
              return mlir::failure();
          } else {
            // The reduce family stores only lane 0 of the reduction result back
            // to the output chunk base, so its store VL is the literal 1 (not
            // the running chunk VL). Detect a reduce-sourced store and emit
            // VL=1; every other (elementwise) store keeps the chunk VL. This
            // mirrors the legacy `tcrv_rvv.reduction_store_vl = "1"` fact.
            mlir::Value storeVL = bodyVL;
            if (auto reduceDef =
                    store.getValue().getDefiningOp<tcrvrvv::ReduceOp>()) {
              mlir::StringAttr layout = reduceDef.getResultLayoutAttr();
              if (layout && layout.getValue() ==
                                "store-reduction-lane0-to-output-chunk-base")
                storeVL =
                    rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
            }
            if (mlir::failed(emitStore(rewriter, loc, store, valueMap,
                                       inductionVar, storeVL)))
              return mlir::failure();
          }
        } else {
          return rewriter.notifyMatchFailure(
              variant, "unsupported op in with_vl beachhead body");
        }
      }
    }

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(variant);
    return mlir::success();
  }

private:
  /// True iff the with_vl body is the pure masked unit-store shape: its only
  /// store-like op is exactly one tcrv_rvv.masked_store, and it contains NO
  /// compute op (binary/macc/compare/select/reduce/dequantize/...) that would
  /// require an agnostic or `_tu`/`_tum` intrinsic form the converter does not
  /// model under undisturbed policy. Such a body is the masked-store family
  /// (mask_load + payload load + masked_store) whose undisturbed scope policy is
  /// honored by the masked-store `_m` intrinsic. Any other shape (a plain store,
  /// a compute op, an extra store) is NOT this exception and stays refused.
  static bool isPureMaskedStoreBody(tcrvrvv::WithVLOp scope) {
    bool sawMaskedStore = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      // Compute / plain-store ops are not part of the pure masked-store shape.
      if (llvm::isa<tcrvrvv::BinaryOp, tcrvrvv::MaskedBinaryOp, tcrvrvv::MAccOp,
                    tcrvrvv::MaskedMAccOp, tcrvrvv::CompareOp, tcrvrvv::SelectOp,
                    tcrvrvv::ReduceOp, tcrvrvv::DequantizeOp, tcrvrvv::MaskAndOp,
                    tcrvrvv::StoreOp, tcrvrvv::StridedStoreOp,
                    tcrvrvv::IndexedStoreOp>(op))
        return false;
      if (llvm::isa<tcrvrvv::MaskedStoreOp>(op)) {
        if (sawMaskedStore)
          return false; // more than one store is not the bounded shape
        sawMaskedStore = true;
      }
    }
    return sawMaskedStore;
  }

  /// True iff the with_vl body is the runtime-scalar computed-mask masked-store
  /// shape: its only store-like op is exactly one tcrv_rvv.masked_store whose
  /// predicate is produced by a tcrv_rvv.compare in the same scope, and whose
  /// compare RHS is a tcrv_rvv.splat of a runtime scalar (load -> splat ->
  /// compare -> masked_store, the RuntimeScalarComputedMaskStore family). That
  /// body carries an undisturbed scope policy honored by the masked-store `_m`
  /// form: the compare/splat/load steps emit agnostic intrinsics whose results
  /// are fully defined over the active VL, so the undisturbed semantics live
  /// entirely in the `_m` store -- correctly lowered.
  ///
  /// The runtime-scalar (splat) compare RHS is REQUIRED: it is the only
  /// legitimate unit store-only computed-mask family. A unit masked_store fed by
  /// a VECTOR-vector compare with no load-merge and no splat is NOT a real
  /// family (its closest sibling, ComputedMaskUnitLoadStore, is a load-merge);
  /// such a body must fall back so the legacy validator rejects it (the
  /// stage2-masked-store-negative contract). Any other compute op, plain/
  /// strided/indexed store, masked load, or a second store likewise drops the
  /// body out of this bounded exception so nothing is mislowered under
  /// undisturbed policy.
  static bool isComputedMaskMaskedStoreBody(tcrvrvv::WithVLOp scope) {
    bool sawComputedMaskedStore = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      // Any compute op other than the mask chain (compare/splat) is out of this
      // shape, as is any other store-like op or a masked LOAD (a load-merge body
      // needs a `_tumu` masked load whose passthrough this exception does not
      // cover).
      if (llvm::isa<tcrvrvv::BinaryOp, tcrvrvv::MaskedBinaryOp, tcrvrvv::MAccOp,
                    tcrvrvv::MaskedMAccOp, tcrvrvv::SelectOp, tcrvrvv::ReduceOp,
                    tcrvrvv::DequantizeOp, tcrvrvv::MaskAndOp,
                    tcrvrvv::MaskedLoadOp, tcrvrvv::MaskedStridedLoadOp,
                    tcrvrvv::MaskedIndexedLoadOp, tcrvrvv::StoreOp,
                    tcrvrvv::StridedStoreOp, tcrvrvv::IndexedStoreOp,
                    tcrvrvv::MaskedStridedStoreOp,
                    tcrvrvv::MaskedIndexedStoreOp>(op))
        return false;
      if (auto maskedStore = llvm::dyn_cast<tcrvrvv::MaskedStoreOp>(op)) {
        if (sawComputedMaskedStore)
          return false; // more than one store is not the bounded shape
        auto compare =
            maskedStore.getMask().getDefiningOp<tcrvrvv::CompareOp>();
        if (!compare)
          return false; // a buffer-mask store is the isPureMaskedStoreBody shape
        // The legitimate unit store-only computed-mask family is runtime-scalar:
        // its compare RHS is a splat of a runtime scalar. Refuse a vector-vector
        // compare unit store-only body (not a real family).
        if (!compare.getRhs().getDefiningOp<tcrvrvv::SplatOp>())
          return false;
        sawComputedMaskedStore = true;
      }
    }
    return sawComputedMaskedStore;
  }

  /// Capability config gate (I1-honoring). The selected variant's `requires`
  /// symbols resolve to tcrv.exec.capability / tcrv.exec.target provider ops in
  /// the kernel; those are queryable MLIR objects that may declare
  /// `supported_sew` / `supported_lmul` as a comma-separated allow-list. If a
  /// resolved provider declares one of these and it does NOT include the typed
  /// body's (sew, lmul), the capability gates this body out: fail the match so
  /// the body falls back to the legacy validator (which rejects it with the
  /// "supported_sew fact ... does not include typed body SEW" diagnostic).
  /// Reading the attrs straight off the provider op keeps the capability the
  /// legality authority -- no string capability model is imported. The gate is
  /// silent when a provider declares no restriction (the common case).
  mlir::LogicalResult
  checkCapabilityConfigGate(mlir::ConversionPatternRewriter &rewriter,
                            tcrv::exec::VariantOp variant,
                            tcrv::exec::KernelOp kernel, unsigned bodySEW,
                            llvm::StringRef bodyLMUL) const {
    auto requiresAttr = variant->getAttrOfType<mlir::ArrayAttr>("requires");
    if (!requiresAttr)
      return mlir::success();
    std::string bodySEWToken = llvm::Twine(bodySEW).str();
    // The provider property is a comma-separated allow-list (e.g. "32,64").
    auto listIncludes = [](llvm::StringRef list, llvm::StringRef token) {
      llvm::SmallVector<llvm::StringRef, 4> entries;
      list.split(entries, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (llvm::StringRef entry : entries)
        if (entry.trim() == token)
          return true;
      return false;
    };
    for (mlir::Attribute entry : requiresAttr) {
      auto symbolRef = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(entry);
      if (!symbolRef)
        continue;
      // Resolve the requires symbol to a provider op in the kernel body. The
      // provider is a capability or target op carrying the optional supported_*
      // allow-lists.
      mlir::Operation *provider = nullptr;
      for (mlir::Operation &op : kernel.getBody().front()) {
        auto sym = op.getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName());
        if (sym && sym.getValue() == symbolRef.getValue()) {
          provider = &op;
          break;
        }
      }
      if (!provider)
        continue;
      if (auto supportedSEW =
              provider->getAttrOfType<mlir::StringAttr>("supported_sew")) {
        llvm::StringRef value = supportedSEW.getValue().trim();
        if (!value.empty() && !listIncludes(value, bodySEWToken))
          return rewriter.notifyMatchFailure(
              variant, "capability provider supported_sew excludes typed body "
                       "SEW (capability gates this body out)");
      }
      if (auto supportedLMUL =
              provider->getAttrOfType<mlir::StringAttr>("supported_lmul")) {
        llvm::StringRef value = supportedLMUL.getValue().trim();
        if (!value.empty() && !listIncludes(value, bodyLMUL))
          return rewriter.notifyMatchFailure(
              variant, "capability provider supported_lmul excludes typed body "
                       "LMUL (capability gates this body out)");
      }
    }
    return mlir::success();
  }

  /// load(%abi, %vl) -> ptr = base + i; __riscv_vle<sew>_v_<dtype><lmul>(ptr, vl)
  mlir::LogicalResult
  emitLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::LoadOp load,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(load.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(load, "load result not typed vector");
    mlir::Value base = valueMap.lookup(load.getBuffer());
    if (!base)
      return rewriter.notifyMatchFailure(load, "load buffer not an ABI param");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          load, "load buffer C type disagrees with loaded vector element");

    // Resolve the EmitC vector type FIRST, before creating any emitc op, so a
    // family the beachhead converter does not cover (e.g. lmul m2) fails the
    // match cleanly and rolls back, instead of leaving a half-converted
    // call_opaque whose result is still a `!tcrv_rvv.vector<...>` type.
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(load, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vle", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(load.getTCRVEmitCLowerableSourceOpName(),
                         load.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{ptr, bodyVL})
            .getResult(0);
    valueMap[load.getLoaded()] = loaded;
    return mlir::success();
  }

  /// binary{kind}(%lhs,%rhs,%vl) -> __riscv_v<op>_vv_<dtype><lmul>(lhs,rhs,vl)
  mlir::LogicalResult
  emitBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::BinaryOp binary,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(binary.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(binary,
                                         "binary result not typed vector");
    mlir::Value lhs = valueMap.lookup(binary.getLhs());
    mlir::Value rhs = valueMap.lookup(binary.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(binary, "binary operand unmapped");

    std::optional<llvm::StringRef> mnemonic = binaryMnemonic(binary.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(binary, "unsupported binary kind");
    // Resolve the EmitC vector type FIRST (see emitLoad): a non-beachhead lmul
    // must fail the match and roll back, not emit an un-lowered result type.
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(binary,
                                         "vector type not convertible");
    std::string callee =
        riscvIntrinsicName(*mnemonic, vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(binary.getTCRVEmitCLowerableSourceOpName(),
                         binary.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[binary.getResult()] = result;
    return mlir::success();
  }

  /// reduce{kind}(%input,%acc,%vl) ->
  ///   __riscv_v<red>_vs_<dtype><lmul>_<dtype>m1(input, acc, vl)
  ///
  /// The generic `reduce` family (operation kind ReduceAdd, memory form
  /// vector-rhs-load) seeds the reduction with the rhs-loaded accumulator VECTOR
  /// (lane 0 holds the running seed for this VL chunk) and writes the lane-0
  /// reduction result straight back to the output chunk base with a VL=1 store.
  /// That per-chunk store is the same `emitStore` path; the VL=1 detail is
  /// handled by `reduceResultStoreVL` below. Here we only emit the reduction
  /// call itself.
  ///
  /// Malformed-body guard: this converter only takes the per-chunk
  /// vector-seeded shape (accumulator typed VECTOR + the chunk-base result
  /// layout). A body whose reduce carries the scalar-carry standalone layout, an
  /// unsupported kind, or an unconvertible (dtype, lmul) is NOT lowered here --
  /// notifyMatchFailure rolls the conversion back so the legacy owner/validators
  /// still see (and reject/own) it.
  mlir::LogicalResult
  emitReduce(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::ReduceOp reduce,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(reduce, "reduce result not vector");
    // The accumulator seed must be a typed vector (the rhs-load chunk seed).
    if (!llvm::isa<tcrvrvv::VectorType>(reduce.getAccumulator().getType()))
      return rewriter.notifyMatchFailure(
          reduce, "reduce accumulator is not a typed vector seed");
    // Malformed-body guard (mirrors the legacy `isReduction && hasRHSBroadcastLike`
    // rejection, RVVEmitCRoutePlanning.cpp:20663-20667): the reduction route
    // requires an EXPLICIT vector input AND accumulator LOAD. A
    // broadcast/splat-seeded accumulator (or input) is NOT in this bounded slice
    // -- the running per-chunk lane-0 seed must be a real loaded vector, not a
    // scalar splat. Reject any input/accumulator not produced by a plain
    // tcrv_rvv.load so a broadcast/splat-seeded reduce body falls back to the
    // legacy validator (which errors) instead of being silently mislowered.
    if (!reduce.getInput().getDefiningOp<tcrvrvv::LoadOp>() ||
        !reduce.getAccumulator().getDefiningOp<tcrvrvv::LoadOp>())
      return rewriter.notifyMatchFailure(
          reduce, "reduce input/accumulator must be explicit vector loads "
                  "(broadcast/splat seed is outside the convertible slice)");
    // Only the per-chunk-base result layout is the `reduce` family. The
    // scalar-carry standalone layout is a DIFFERENT (still-owned) family.
    mlir::StringAttr resultLayout = reduce.getResultLayoutAttr();
    if (!resultLayout ||
        resultLayout.getValue() !=
            "store-reduction-lane0-to-output-chunk-base")
      return rewriter.notifyMatchFailure(
          reduce, "reduce result layout outside the convertible reduce family");

    mlir::Value input = valueMap.lookup(reduce.getInput());
    mlir::Value accumulator = valueMap.lookup(reduce.getAccumulator());
    if (!input || !accumulator)
      return rewriter.notifyMatchFailure(reduce, "reduce operand unmapped");

    std::optional<llvm::StringRef> mnemonic =
        reductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(reduce, "unsupported reduce kind");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(reduce, "vector type not convertible");
    std::string callee = riscvReductionIntrinsicName(
        *mnemonic, vectorElementWidth(vectorType), vectorType.getLmul(),
        vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduce.getTCRVEmitCLowerableSourceOpName(),
                         reduce.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{input, accumulator,
                                                          bodyVL})
            .getResult(0);
    valueMap[reduce.getResult()] = result;
    return mlir::success();
  }

  //===--------------------------------------------------------------------===//
  // Standalone (scalar-carry-through-memory) reduction family.
  //
  // Unlike the per-chunk `reduce` family (which seeds with a freshly loaded
  // accumulator vector and writes the lane-0 result to the chunk base), the
  // standalone reduction carries ONE scalar accumulator through the output
  // memory cell `out[0]` across runtime VL chunks:
  //   PRE-LOOP: out[0] = acc[0]    (splat the seed read from the accumulator
  //                                 buffer into a lane-0 m1 vector, store VL=1).
  //   IN-LOOP : out[0] = reduce(input_chunk, splat(out[0]))   (read the running
  //                                 scalar back, splat it, horizontal-reduce the
  //                                 input chunk over it, store back VL=1).
  // The seed/result vector is ALWAYS lmul m1 (the reduction lands its scalar in
  // lane 0 of an m1 destination); the SOURCE may be a wider/fractional lmul
  // (m2, or the widening mf2 rung). Byte-identical to the legacy
  // RVVEmitCReductionAccumulationStatementPlanOwners.cpp standalone oracle.
  //===--------------------------------------------------------------------===//

  /// True iff `op` is a (masked) standalone reduction carrying the scalar-carry
  /// output layout. This is the discriminator separating the standalone family
  /// from the per-chunk `reduce` family.
  static bool isStandaloneReductionOp(mlir::Operation *op) {
    if (!op)
      return false;
    if (auto standalone = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
      mlir::StringAttr layout = standalone.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-standalone-reduction-lane0-to-output-scalar";
    }
    if (auto masked = llvm::dyn_cast<tcrvrvv::MaskedStandaloneReduceOp>(op)) {
      mlir::StringAttr layout = masked.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-standalone-reduction-lane0-to-output-scalar";
    }
    // The widening dot-product reduction (plain / strided-input / computed-mask)
    // carries the SAME scalar-carry-through-output structure: an i32 seed read
    // from the accumulator-input buffer pre-loop, a running seed read back from
    // out[0] each chunk, and a lane-0 result stored to the output base (VL=1).
    // Its result layout is `store-dot-reduction-lane0-to-output-scalar`.
    if (auto dot = llvm::dyn_cast<tcrvrvv::WideningDotReduceOp>(op)) {
      mlir::StringAttr layout = dot.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-dot-reduction-lane0-to-output-scalar";
    }
    if (auto maskedDot =
            llvm::dyn_cast<tcrvrvv::MaskedWideningDotReduceOp>(op)) {
      mlir::StringAttr layout = maskedDot.getResultLayoutAttr();
      return layout && layout.getValue() ==
                           "store-dot-reduction-lane0-to-output-scalar";
    }
    return false;
  }

  /// True iff the with_vl body is a standalone reduction body (it carries a
  /// scalar-carry standalone_reduce / masked_standalone_reduce).
  static bool isStandaloneReductionBody(tcrvrvv::WithVLOp scope) {
    for (mlir::Operation &op : scope.getBody().front())
      if (isStandaloneReductionOp(&op))
        return true;
    return false;
  }

  /// The reduction mnemonic for the standalone family, mirroring the legacy
  /// getRVVSelectedBodyStandaloneReductionIntrinsic kind table. The widening
  /// kinds reduce a (fractional-lmul) i16 source into an i32 accumulator via
  /// vwredsum; the plain kinds use vred{sum,min,max}. Unknown kinds fail the
  /// match so the body falls back to the legacy validators unchanged.
  static std::optional<llvm::StringRef>
  standaloneReductionMnemonic(llvm::StringRef kind) {
    if (kind == "add")
      return llvm::StringRef("vredsum");
    if (kind == "min")
      return llvm::StringRef("vredmin");
    if (kind == "max")
      return llvm::StringRef("vredmax");
    if (kind == "signed_widening_reduce_add")
      return llvm::StringRef("vwredsum");
    if (kind == "unsigned_widening_reduce_add")
      return llvm::StringRef("vwredsumu");
    return std::nullopt;
  }

  /// The inactive-lane neutral element a computed-mask reduction merges into
  /// the masked-out source lanes before reducing, mirroring the legacy
  /// getRVVStandaloneReductionStatementPlanInactiveNeutral table:
  ///   add -> "0"; min -> INT_MAX (sew-sized); max -> INT_MIN (sew-sized).
  static std::optional<llvm::StringRef>
  maskedStandaloneReductionNeutral(llvm::StringRef kind, unsigned sew) {
    if (kind == "add")
      return llvm::StringRef("0");
    if (kind == "min")
      return sew == 64 ? llvm::StringRef("9223372036854775807")
                       : llvm::StringRef("2147483647");
    if (kind == "max")
      return sew == 64 ? llvm::StringRef("(-9223372036854775807-1)")
                       : llvm::StringRef("(-2147483647-1)");
    return std::nullopt;
  }

  /// The standalone reduction intrinsic name:
  ///   __riscv_v<red>_vs_<srcDtype><srcLmul>_<resultDtype>m1
  /// For the non-widening forms srcDtype==resultDtype and srcLmul is the input
  /// lmul (e.g. vredsum_vs_i32m2_i32m1). For the widening forms the source is a
  /// narrower fractional-lmul vector (i16/mf2) and the result is i32/m1
  /// (vwredsum_vs_i16mf2_i32m1). Byte-identical to the legacy
  /// getRVVSelectedBodyStandaloneReductionIntrinsic shape.
  static std::string standaloneReductionIntrinsicName(
      llvm::StringRef mnemonic, llvm::StringRef srcDtype,
      llvm::StringRef srcLmul, llvm::StringRef resultDtype) {
    std::string name;
    llvm::raw_string_ostream os(name);
    os << "__riscv_" << mnemonic << "_vs_" << srcDtype << srcLmul << "_"
       << resultDtype << "m1";
    os.flush();
    return name;
  }

  /// Read `buffer[0]` as a scalar then splat it into a lane-0 m1 seed vector via
  /// vmv_v_x with VL literal 1 -- the scalar-carry primitive shared by the
  /// pre-loop seed and the in-loop running-seed read. `resultVecType` is the
  /// reduction RESULT vector type (always m1); its element drives the splat
  /// intrinsic dtype/sew and the `int32_t v = base[0];` temp element. The
  /// provenance comment carries the reduction op's source-op name/role so the
  /// rendered C matches the legacy `vmv_v_x` step. Returns the splat result
  /// Value, or nullptr on an unconvertible/typeless shape (caller fails match).
  mlir::Value emitScalarSeedSplat(mlir::ConversionPatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value buffer,
                                  tcrvrvv::VectorType resultVecType,
                                  llvm::StringRef sourceOpName,
                                  llvm::StringRef sourceRole) const {
    auto pointer = llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(buffer);
    if (!pointer)
      return nullptr;
    if (!bufferPointeeMatchesVectorElement(buffer, resultVecType))
      return nullptr;
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType)
      return nullptr;
    if (isFloatVector(resultVecType))
      return nullptr; // standalone reduction is the integer-accumulator slice.
    std::string callee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, callee));
    // base[0]: subscript -> lvalue -> load reads the first scalar element. The
    // pointee const-ness (const int32_t* acc vs int32_t* out) flows through the
    // lvalue value type, so the seed temp prints `const int32_t` / `int32_t` to
    // match the legacy oracle automatically.
    mlir::Value index =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp subscriptOp =
        rewriter.create<emitc::SubscriptOp>(loc, pointer, index);
    auto lvalueType =
        llvm::cast<emitc::LValueType>(subscriptOp.getResult().getType());
    mlir::Value scalar =
        rewriter
            .create<emitc::LoadOp>(loc, lvalueType.getValueType(),
                                   subscriptOp.getResult())
            .getResult();
    mlir::Value one =
        rewriter.create<emitc::LiteralOp>(loc, getSizeType(rewriter), "1");
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                     mlir::ValueRange{scalar, one})
        .getResult(0);
  }

  static mlir::Type getSizeType(mlir::ConversionPatternRewriter &rewriter) {
    return emitc::OpaqueType::get(rewriter.getContext(), "size_t");
  }

  /// Emit the pre-loop seed of a standalone reduction: out[0] = acc[0]. Reads
  /// the accumulator-seed buffer's first element, splats it into a lane-0 m1
  /// vector, and stores it to the output buffer BASE with VL=1. Runs between the
  /// pre-loop full-chunk setvl and the for-loop. Returns failure (caller falls
  /// back) on a malformed standalone body (missing/mismatched acc/out buffers,
  /// unconvertible result type).
  mlir::LogicalResult emitStandaloneReductionPreLoopSeed(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::Operation *reduceOp = nullptr;
    tcrvrvv::StoreOp storeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (isStandaloneReductionOp(&op))
        reduceOp = &op;
      if (auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(op))
        storeOp = store;
    }
    if (!reduceOp || !storeOp)
      return rewriter.notifyMatchFailure(scope,
                                         "standalone reduction body missing "
                                         "reduce/store");

    mlir::Value accSeed;
    mlir::Value resultValue;
    llvm::StringRef sourceOpName;
    llvm::StringRef sourceRole;
    if (auto standalone = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(reduceOp)) {
      accSeed = standalone.getAccumulatorSeed();
      resultValue = standalone.getResult();
      sourceOpName = standalone.getTCRVEmitCLowerableSourceOpName();
      sourceRole = standalone.getTCRVEmitCLowerableSourceRole();
    } else if (auto masked =
                   llvm::dyn_cast<tcrvrvv::MaskedStandaloneReduceOp>(reduceOp)) {
      accSeed = masked.getAccumulatorSeed();
      resultValue = masked.getResult();
      sourceOpName = masked.getTCRVEmitCLowerableSourceOpName();
      sourceRole = masked.getTCRVEmitCLowerableSourceRole();
    } else if (auto dot =
                   llvm::dyn_cast<tcrvrvv::WideningDotReduceOp>(reduceOp)) {
      accSeed = dot.getAccumulatorSeed();
      resultValue = dot.getResult();
      sourceOpName = dot.getTCRVEmitCLowerableSourceOpName();
      sourceRole = dot.getTCRVEmitCLowerableSourceRole();
    } else {
      auto maskedDot =
          llvm::cast<tcrvrvv::MaskedWideningDotReduceOp>(reduceOp);
      accSeed = maskedDot.getAccumulatorSeed();
      resultValue = maskedDot.getResult();
      sourceOpName = maskedDot.getTCRVEmitCLowerableSourceOpName();
      sourceRole = maskedDot.getTCRVEmitCLowerableSourceRole();
    }
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(resultValue.getType());
    if (!resultVecType || resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          scope, "standalone reduction result must be an m1 vector");

    mlir::Value accBuffer = valueMap.lookup(accSeed);
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!accBuffer || !outBuffer)
      return rewriter.notifyMatchFailure(
          scope, "standalone reduction acc/out buffers unmapped");

    // out[0] = acc[0]: splat the accumulator seed read, store to out base VL=1.
    mlir::Value seedSplat = emitScalarSeedSplat(rewriter, loc, accBuffer,
                                                resultVecType, sourceOpName,
                                                sourceRole);
    if (!seedSplat)
      return rewriter.notifyMatchFailure(
          scope, "standalone reduction pre-loop seed not convertible");
    if (mlir::failed(emitStandaloneReductionScalarStore(
            rewriter, loc, storeOp, outBuffer, seedSplat, resultVecType)))
      return mlir::failure();
    return mlir::success();
  }

  /// Store a lane-0 reduction vector to the output buffer BASE (no `+ i`) with
  /// VL literal 1 -- the scalar-output store shared by the pre-loop seed and the
  /// in-loop result. Mirrors the legacy `vse<sew>_v_<dtype>m1(out, v, 1)`.
  mlir::LogicalResult emitStandaloneReductionScalarStore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::StoreOp store, mlir::Value outBuffer, mlir::Value value,
      tcrvrvv::VectorType resultVecType) const {
    if (!bufferPointeeMatchesVectorElement(outBuffer, resultVecType))
      return rewriter.notifyMatchFailure(
          store, "standalone reduction output buffer element mismatch");
    if (!convertVectorTypeToEmitC(resultVecType))
      return rewriter.notifyMatchFailure(store,
                                         "standalone result type not convertible");
    std::string callee =
        riscvIntrinsicName("vse", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(store.getTCRVEmitCLowerableSourceOpName(),
                         store.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value one =
        rewriter.create<emitc::LiteralOp>(loc, getSizeType(rewriter), "1");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{outBuffer, value, one});
    return mlir::success();
  }

  /// In-loop plain standalone reduce:
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1);
  ///   v<rd>m1 red = __riscv_v<red>_vs_<src><srcLmul>_<rd>m1(input, seed, vl);
  /// The output buffer is the body's store target (the scalar-carry cell). The
  /// store itself is emitted by the store dispatch (to base, VL=1).
  mlir::LogicalResult
  emitStandaloneReduce(mlir::ConversionPatternRewriter &rewriter,
                       mlir::Location loc, tcrvrvv::StandaloneReduceOp reduce,
                       llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                       mlir::Value outBuffer, mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getInput().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(reduce,
                                         "standalone reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          reduce, "standalone reduce result must be m1");
    // The input must be an explicit vector load (no broadcast/splat seed) --
    // mirrors the legacy reduction input-load requirement. The
    // widening-product-reduce family chains the reduction directly onto the
    // tcrv_rvv.widening_product result (load -> load -> vwmul -> vwredsum), so a
    // widening_product input is the OTHER convertible producer: it is a real
    // typed vector dataflow value, not a broadcast/splat scalar seed.
    {
      mlir::Operation *inputDef = reduce.getInput().getDefiningOp();
      if (!inputDef ||
          !llvm::isa<tcrvrvv::LoadOp, tcrvrvv::WideningProductOp>(inputDef))
        return rewriter.notifyMatchFailure(
            reduce, "standalone reduce input must be an explicit vector load or "
                    "widening_product result");
    }
    std::optional<llvm::StringRef> mnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(reduce,
                                         "unsupported standalone reduce kind");
    mlir::Value input = valueMap.lookup(reduce.getInput());
    if (!input)
      return rewriter.notifyMatchFailure(reduce, "standalone reduce input "
                                                 "unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType || !convertVectorTypeToEmitC(srcVecType))
      return rewriter.notifyMatchFailure(reduce,
                                         "standalone reduce type not convertible");
    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        reduce.getTCRVEmitCLowerableSourceOpName(),
        reduce.getTCRVEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(
          reduce, "standalone reduce running seed not convertible");
    std::string callee = standaloneReductionIntrinsicName(
        *mnemonic, vectorDType(srcVecType), srcVecType.getLmul(),
        vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduce.getTCRVEmitCLowerableSourceOpName(),
                         reduce.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{input, seed, bodyVL})
            .getResult(0);
    valueMap[reduce.getResult()] = result;
    return mlir::success();
  }

  /// In-loop computed-mask standalone reduce:
  ///   v<src><sl> neutral = vmv_v_x_<src><sl>(<NEUTRAL>, vl);  // input lmul
  ///   v<src><sl> masked  = vmerge_vvm_<src><sl>(neutral, source, mask, vl);
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1);
  ///   v<rd>m1 red = __riscv_v<red>_vs_<src><sl>_<rd>m1(masked, seed, vl);
  /// The neutral fills the masked-out lanes so they don't affect the reduction.
  mlir::LogicalResult emitMaskedStandaloneReduce(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedStandaloneReduceOp reduce,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, mlir::Value outBuffer,
      mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getInput().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce result must be m1");
    std::optional<llvm::StringRef> mnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(
          reduce, "unsupported masked standalone reduce kind");
    std::optional<llvm::StringRef> neutral = maskedStandaloneReductionNeutral(
        reduce.getKind(), vectorElementWidth(srcVecType));
    if (!neutral)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce neutral not derivable");
    mlir::Value mask = valueMap.lookup(reduce.getMask());
    mlir::Value source = valueMap.lookup(reduce.getInput());
    if (!mask || !source)
      return rewriter.notifyMatchFailure(reduce,
                                         "masked standalone reduce operand "
                                         "unmapped");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcEmitC = convertVectorTypeToEmitC(srcVecType);
    if (!resultEmitC || !srcEmitC)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce type not convertible");

    // Neutral splat over the masked-out lanes (input lmul, running bodyVL).
    std::string neutralCallee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(srcVecType),
                           srcVecType.getLmul(), vectorDType(srcVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduce.getTCRVEmitCLowerableSourceOpName(),
                         reduce.getTCRVEmitCLowerableSourceRole(),
                         neutralCallee));
    mlir::Value neutralLiteral =
        rewriter.create<emitc::LiteralOp>(loc, resultIntScalarType(rewriter),
                                          neutral->str());
    mlir::Value neutralVec =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{srcEmitC}, neutralCallee,
                mlir::ValueRange{neutralLiteral, bodyVL})
            .getResult(0);

    // Merge active source lanes over the neutral background.
    std::string mergeCallee =
        riscvIntrinsicName("vmerge", vectorElementWidth(srcVecType),
                           srcVecType.getLmul(), vectorDType(srcVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduce.getTCRVEmitCLowerableSourceOpName(),
                         reduce.getTCRVEmitCLowerableSourceRole(), mergeCallee));
    mlir::Value masked =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{srcEmitC}, mergeCallee,
                mlir::ValueRange{neutralVec, source, mask, bodyVL})
            .getResult(0);

    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        reduce.getTCRVEmitCLowerableSourceOpName(),
        reduce.getTCRVEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(
          reduce, "masked standalone reduce running seed not convertible");

    std::string callee = standaloneReductionIntrinsicName(
        *mnemonic, vectorDType(srcVecType), srcVecType.getLmul(),
        vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduce.getTCRVEmitCLowerableSourceOpName(),
                         reduce.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         callee,
                                         mlir::ValueRange{masked, seed, bodyVL})
            .getResult(0);
    valueMap[reduce.getResult()] = result;
    return mlir::success();
  }

  //===--------------------------------------------------------------------===//
  // Widening contraction family (signed low-precision products / dot-reduce /
  // widening macc). Byte-identical to the legacy
  // RVVEmitCContractionRouteFamilyPlanOwners.cpp DirectContraction oracle.
  //===--------------------------------------------------------------------===//

  /// widening_product(%lhs,%rhs,%vl){kind=signed_widening_product} ->
  ///   v<rd><rl> p = __riscv_vwmul_vv_<rd><rl>(lhs, rhs, vl);
  /// The vwmul widens the narrower (fractional-lmul) source multiplicands into
  /// the RESULT (one-step-wider) vector type, so the intrinsic dtype/lmul come
  /// from the RESULT type (i8/mf4 x i8/mf4 -> i16/mf2 == __riscv_vwmul_vv_i16mf2).
  /// Only the signed i-source slice is in scope; the unsigned (ui) source uses a
  /// vwmulu intrinsic the legacy provider has NOT productionized for this body
  /// converter, so an unsigned source falls back to the legacy path unchanged.
  mlir::LogicalResult
  emitWideningProduct(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, tcrvrvv::WideningProductOp product,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(product.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(product.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(product,
                                         "widening product types not vectors");
    if (product.getKind() != "signed_widening_product")
      return rewriter.notifyMatchFailure(
          product, "only the signed widening product is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(
          product, "widening product type not convertible");
    mlir::Value lhs = valueMap.lookup(product.getLhs());
    mlir::Value rhs = valueMap.lookup(product.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(product,
                                         "widening product operand unmapped");
    // The widened product intrinsic dtype/lmul derive from the RESULT vector.
    std::string callee =
        riscvIntrinsicName("vwmul", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(product.getTCRVEmitCLowerableSourceOpName(),
                         product.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         callee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[product.getResult()] = result;
    return mlir::success();
  }

  /// widening_macc(%lhs,%rhs,%acc,%vl){kind=signed_widening_macc_add} ->
  ///   v<rd><rl> r = __riscv_vwmacc_vv_<rd><rl>(acc, lhs, rhs, vl);
  /// The fused widening multiply-accumulate widens the narrower i16 source
  /// multiplicands and accumulates into the i32 accumulator vector. The C call
  /// order is (accumulator, lhs, rhs, vl); the intrinsic dtype/lmul derive from
  /// the RESULT (i32/m1) vector. The accumulator is an explicit loaded i32
  /// vector chunk (the per-chunk macc seed), not a scalar-carry cell.
  mlir::LogicalResult
  emitWideningMAcc(mlir::ConversionPatternRewriter &rewriter,
                   mlir::Location loc, tcrvrvv::WideningMAccOp macc,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(macc.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(macc.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(macc,
                                         "widening macc types not vectors");
    if (macc.getKind() != "signed_widening_macc_add")
      return rewriter.notifyMatchFailure(
          macc, "only the signed widening macc-add is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(macc,
                                         "widening macc type not convertible");
    mlir::Value lhs = valueMap.lookup(macc.getLhs());
    mlir::Value rhs = valueMap.lookup(macc.getRhs());
    mlir::Value accumulator = valueMap.lookup(macc.getAccumulator());
    if (!lhs || !rhs || !accumulator)
      return rewriter.notifyMatchFailure(macc, "widening macc operand unmapped");
    std::string callee =
        riscvIntrinsicName("vwmacc", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(macc.getTCRVEmitCLowerableSourceOpName(),
                         macc.getTCRVEmitCLowerableSourceRole(), callee));
    // vwmacc destination read-modify-writes the accumulator: (acc, lhs, rhs, vl).
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, callee,
                mlir::ValueRange{accumulator, lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[macc.getResult()] = result;
    return mlir::success();
  }

  /// widening_dot_reduce(%lhs,%rhs,%acc,%vl){kind=signed_widening_dot_reduce_add}
  /// is the scalar-carry-through-output dot product. The pre-loop seed
  /// (out[0]=acc[0]) and the lane-0 result store (out base, VL=1) are handled by
  /// the shared standalone-reduction machinery; the in-loop dataflow is:
  ///   v<rd>m1 p = __riscv_vwmul_vv_<rd>m1(lhs, rhs, vl);      // widened product
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1); // running seed
  ///   v<rd>m1 red = __riscv_vredsum_vs_<rd>m1_<rd>m1(p, seed, vl);
  /// The product widens i16/mf2 multiplicands into the i32/m1 result, so the
  /// vwmul/vredsum dtype/lmul derive from the RESULT (i32/m1) vector.
  mlir::LogicalResult
  emitWideningDotReduce(mlir::ConversionPatternRewriter &rewriter,
                        mlir::Location loc, tcrvrvv::WideningDotReduceOp dot,
                        llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                        mlir::Value outBuffer, mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dot.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dot.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(dot, "dot reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(dot, "dot reduce result must be m1");
    if (dot.getKind() != "signed_widening_dot_reduce_add")
      return rewriter.notifyMatchFailure(
          dot, "only the signed widening dot-reduce-add is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(dot, "dot reduce type not convertible");
    mlir::Value lhs = valueMap.lookup(dot.getLhs());
    mlir::Value rhs = valueMap.lookup(dot.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(dot, "dot reduce operand unmapped");

    // Widened product (result dtype/lmul): vwmul_vv_i32m1(lhs, rhs, vl).
    std::string productCallee =
        riscvIntrinsicName("vwmul", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dot.getTCRVEmitCLowerableSourceOpName(),
                         dot.getTCRVEmitCLowerableSourceRole(), productCallee));
    mlir::Value product =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         productCallee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);

    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        dot.getTCRVEmitCLowerableSourceOpName(),
        dot.getTCRVEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(dot,
                                         "dot reduce running seed not "
                                         "convertible");

    // Plain horizontal reduction of the i32 product over the running seed.
    std::string reduceCallee = riscvReductionIntrinsicName(
        "vredsum", vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dot.getTCRVEmitCLowerableSourceOpName(),
                         dot.getTCRVEmitCLowerableSourceRole(), reduceCallee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, reduceCallee,
                mlir::ValueRange{product, seed, bodyVL})
            .getResult(0);
    valueMap[dot.getResult()] = result;
    return mlir::success();
  }

  /// masked_widening_dot_reduce(%mask,%lhs,%rhs,%acc,%vl) is the predicated
  /// scalar-carry dot product. The pre-loop seed / lane-0 store are shared. The
  /// in-loop dataflow zeroes inactive product lanes before reducing:
  ///   v<rd>m1 zero = vmv_v_x_<rd>m1(0, vl);                       // running vl
  ///   v<rd>m1 mp   = __riscv_vwmul_vv_<rd>m1_m(mask, lhs, rhs, vl); // masked
  ///   v<rd>m1 mrg  = __riscv_vmerge_vvm_<rd>m1(zero, mp, mask, vl); // 0 inactive
  ///   <celt> r = out[0]; v<rd>m1 seed = vmv_v_x_<rd>m1(r, 1);
  ///   v<rd>m1 red = __riscv_vredsum_vs_<rd>m1_<rd>m1(mrg, seed, vl);
  mlir::LogicalResult emitMaskedWideningDotReduce(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedWideningDotReduceOp dot,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap, mlir::Value outBuffer,
      mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dot.getResult().getType());
    auto lhsVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dot.getLhs().getType());
    if (!resultVecType || !lhsVecType)
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce types not vectors");
    if (resultVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce result must be m1");
    if (dot.getKind() != "signed_masked_widening_dot_reduce_add")
      return rewriter.notifyMatchFailure(
          dot, "only the signed masked widening dot-reduce-add is convertible");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    if (!resultEmitC || !convertVectorTypeToEmitC(lhsVecType))
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce type not "
                                         "convertible");
    mlir::Value mask = valueMap.lookup(dot.getMask());
    mlir::Value lhs = valueMap.lookup(dot.getLhs());
    mlir::Value rhs = valueMap.lookup(dot.getRhs());
    if (!mask || !lhs || !rhs)
      return rewriter.notifyMatchFailure(dot,
                                         "masked dot reduce operand unmapped");

    // Zero background over the running VL (the inactive-lane neutral for add).
    std::string zeroCallee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dot.getTCRVEmitCLowerableSourceOpName(),
                         dot.getTCRVEmitCLowerableSourceRole(), zeroCallee));
    mlir::Value zeroLiteral = rewriter.create<emitc::LiteralOp>(
        loc, resultIntScalarType(rewriter), "0");
    mlir::Value zeroVec =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         zeroCallee,
                                         mlir::ValueRange{zeroLiteral, bodyVL})
            .getResult(0);

    // Masked widened product: vwmul_vv_<rd>m1_m(mask, lhs, rhs, vl).
    std::string productCallee =
        riscvIntrinsicName("vwmul", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType)) +
        "_m";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dot.getTCRVEmitCLowerableSourceOpName(),
                         dot.getTCRVEmitCLowerableSourceRole(), productCallee));
    mlir::Value maskedProduct =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, productCallee,
                mlir::ValueRange{mask, lhs, rhs, bodyVL})
            .getResult(0);

    // Merge the masked product over the zero background (inactive lanes -> 0).
    std::string mergeCallee =
        riscvIntrinsicName("vmerge", vectorElementWidth(resultVecType),
                           resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dot.getTCRVEmitCLowerableSourceOpName(),
                         dot.getTCRVEmitCLowerableSourceRole(), mergeCallee));
    mlir::Value merged =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, mergeCallee,
                mlir::ValueRange{zeroVec, maskedProduct, mask, bodyVL})
            .getResult(0);

    // In-loop running-seed read: out[0] -> splat m1.
    mlir::Value seed = emitScalarSeedSplat(
        rewriter, loc, outBuffer, resultVecType,
        dot.getTCRVEmitCLowerableSourceOpName(),
        dot.getTCRVEmitCLowerableSourceRole());
    if (!seed)
      return rewriter.notifyMatchFailure(
          dot, "masked dot reduce running seed not convertible");

    std::string reduceCallee = riscvReductionIntrinsicName(
        "vredsum", vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dot.getTCRVEmitCLowerableSourceOpName(),
                         dot.getTCRVEmitCLowerableSourceRole(), reduceCallee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, reduceCallee,
                mlir::ValueRange{merged, seed, bodyVL})
            .getResult(0);
    valueMap[dot.getResult()] = result;
    return mlir::success();
  }

  /// The C scalar literal type the neutral splat constant prints as. The neutral
  /// is an integer literal (0 / INT_MAX / INT_MIN-shaped); emitc renders the
  /// literal verbatim, so an opaque int placeholder suffices for the SSA type.
  static mlir::Type resultIntScalarType(mlir::ConversionPatternRewriter &r) {
    return emitc::OpaqueType::get(r.getContext(), "int");
  }

  /// store(%abi,%val,%vl) -> ptr = base + i; __riscv_vse<sew>_v_<dtype><lmul>(...)
  mlir::LogicalResult
  emitStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::StoreOp store,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value inductionVar, mlir::Value storeVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(store.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(store, "store value not typed vector");
    mlir::Value base = valueMap.lookup(store.getBuffer());
    mlir::Value value = valueMap.lookup(store.getValue());
    if (!base || !value)
      return rewriter.notifyMatchFailure(store, "store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          store, "store buffer C type disagrees with stored vector element");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(store, "vector type not convertible");

    std::string callee =
        riscvIntrinsicName("vse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(store.getTCRVEmitCLowerableSourceOpName(),
                         store.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, callee,
                                         mlir::ValueRange{ptr, value, storeVL});
    return mlir::success();
  }

  /// broadcast_load(%abi,%vl) -> scalar = base[0]; __riscv_vmv_v_x_<dtype><lmul>
  /// The legacy materializer renders the RHS broadcast operand `rhs[0]` via an
  /// emitc.subscript + emitc.load reading the first element, then splats that
  /// scalar with vmv_v_x. Reproduced exactly so the rendered C carries the
  /// `const int32_t vN = base[0];` temp + the vmv_v_x splat.
  mlir::LogicalResult
  emitBroadcastLoad(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, tcrvrvv::BroadcastLoadOp broadcast,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(broadcast.getBroadcast().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(broadcast,
                                         "broadcast result not typed vector");
    mlir::Value base = valueMap.lookup(broadcast.getBuffer());
    if (!base)
      return rewriter.notifyMatchFailure(broadcast,
                                         "broadcast buffer not an ABI param");
    auto pointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(base);
    if (!pointer)
      return rewriter.notifyMatchFailure(
          broadcast, "broadcast buffer must be a pointer-typed ABI param");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          broadcast,
          "broadcast buffer C type disagrees with broadcast vector element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(broadcast,
                                         "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vmv_v_x", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(broadcast.getTCRVEmitCLowerableSourceOpName(),
                         broadcast.getTCRVEmitCLowerableSourceRole(), callee));
    // base[0]: subscript -> lvalue -> load reads the first scalar element.
    mlir::Value index = rewriter.create<emitc::LiteralOp>(
        loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp subscriptOp =
        rewriter.create<emitc::SubscriptOp>(loc, pointer, index);
    auto lvalueType =
        llvm::cast<emitc::LValueType>(subscriptOp.getResult().getType());
    mlir::Value scalar =
        rewriter
            .create<emitc::LoadOp>(loc, lvalueType.getValueType(),
                                   subscriptOp.getResult())
            .getResult();
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{scalar, bodyVL})
            .getResult(0);
    valueMap[broadcast.getBroadcast()] = result;
    return mlir::success();
  }

  /// splat(%scalar,%vl) -> __riscv_vmv_v_x_<dtype><lmul>(scalar, vl). The scalar
  /// is a runtime ABI value mapped to a function parameter directly.
  mlir::LogicalResult
  emitSplat(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::SplatOp splat,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(splat.getBroadcast().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(splat, "splat result not typed vector");
    mlir::Value scalar = valueMap.lookup(splat.getScalar());
    if (!scalar)
      return rewriter.notifyMatchFailure(splat, "splat scalar unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(splat, "vector type not convertible");
    // Float splat uses vfmv_v_f (float scalar); integer splat uses vmv_v_x.
    llvm::StringRef splatMnemonic =
        isFloatVector(vectorType) ? "vfmv_v_f" : "vmv_v_x";
    std::string callee =
        riscvIntrinsicName(splatMnemonic, vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(splat.getTCRVEmitCLowerableSourceOpName(),
                         splat.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{scalar, bodyVL})
            .getResult(0);
    valueMap[splat.getBroadcast()] = result;
    return mlir::success();
  }

  /// compare(%lhs,%rhs,%vl){kind} ->
  ///   __riscv_v<cmp>_vv_<dtype><lmul>_b<maskbits>(lhs, rhs, vl) -> vbool<n>_t
  mlir::LogicalResult
  emitCompare(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              tcrvrvv::CompareOp compare,
              llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
              mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<tcrvrvv::MaskType>(compare.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(compare, "compare result not typed mask");
    auto operandVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(compare.getLhs().getType());
    if (!operandVecType)
      return rewriter.notifyMatchFailure(compare,
                                         "compare operand not typed vector");
    mlir::Value lhs = valueMap.lookup(compare.getLhs());
    mlir::Value rhs = valueMap.lookup(compare.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(compare, "compare operand unmapped");
    std::optional<llvm::StringRef> mnemonic =
        compareMnemonic(compare.getKind(), isFloatVector(operandVecType));
    if (!mnemonic)
      return rewriter.notifyMatchFailure(compare, "unsupported compare kind");
    mlir::Type maskEmitCType = getTypeConverter()->convertType(maskType);
    if (!maskEmitCType ||
        maskEmitCType.getDialect().getNamespace() !=
            emitc::EmitCDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(compare, "mask type not convertible");
    unsigned sew = vectorElementWidth(operandVecType);
    unsigned maskBits = maskWidthForConfig(sew, operandVecType.getLmul());
    if (maskBits == 0)
      return rewriter.notifyMatchFailure(compare, "unsupported mask config");
    std::string callee = riscvCompareIntrinsicName(
        *mnemonic, sew, operandVecType.getLmul(),
        vectorDType(operandVecType), maskBits);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(compare.getTCRVEmitCLowerableSourceOpName(),
                         compare.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{maskEmitCType},
                                         callee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[compare.getMask()] = result;
    return mlir::success();
  }

  /// select(%mask,%true,%false,%vl) ->
  ///   __riscv_vmerge_vvm_<dtype><lmul>(false, true, mask, vl)
  /// vmerge keeps the FALSE vector on inactive lanes and the TRUE vector on
  /// active lanes, so the operand order is (false_vec, true_vec, mask, vl) --
  /// byte-identical to the legacy compare-select select step.
  mlir::LogicalResult
  emitSelect(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
             tcrvrvv::SelectOp select,
             llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
             mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(select.getSelected().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(select,
                                         "select result not typed vector");
    mlir::Value mask = valueMap.lookup(select.getMask());
    mlir::Value trueValue = valueMap.lookup(select.getTrueValue());
    mlir::Value falseValue = valueMap.lookup(select.getFalseValue());
    if (!mask || !trueValue || !falseValue)
      return rewriter.notifyMatchFailure(select, "select operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(select, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vmerge", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(select.getTCRVEmitCLowerableSourceOpName(),
                         select.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{falseValue, trueValue, mask, bodyVL})
            .getResult(0);
    valueMap[select.getSelected()] = result;
    return mlir::success();
  }

  /// mask_and(%a,%b,%vl){kind} ->
  ///   __riscv_vmand_mm_b<maskbits>(a, b, vl)
  /// composes two predicate masks of the same (sew, lmul) into one mask.
  mlir::LogicalResult
  emitMaskAnd(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
              tcrvrvv::MaskAndOp maskAnd,
              llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
              mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<tcrvrvv::MaskType>(maskAnd.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(maskAnd, "mask_and result not mask");
    mlir::Value lhs = valueMap.lookup(maskAnd.getLhs());
    mlir::Value rhs = valueMap.lookup(maskAnd.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(maskAnd, "mask_and operand unmapped");
    std::optional<llvm::StringRef> mnemonic = maskAndMnemonic(maskAnd.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(maskAnd, "unsupported mask_and kind");
    mlir::Type maskEmitCType = getTypeConverter()->convertType(maskType);
    if (!maskEmitCType ||
        maskEmitCType.getDialect().getNamespace() !=
            emitc::EmitCDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(maskAnd, "mask type not convertible");
    unsigned sew = 0;
    if (maskType.getElementType().isSignlessInteger(32))
      sew = 32;
    else if (maskType.getElementType().isSignlessInteger(64))
      sew = 64;
    unsigned maskBits = maskWidthForConfig(sew, maskType.getLmul());
    if (maskBits == 0)
      return rewriter.notifyMatchFailure(maskAnd, "unsupported mask config");
    std::string callee = riscvMaskComposeIntrinsicName(*mnemonic, maskBits);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskAnd.getTCRVEmitCLowerableSourceOpName(),
                         maskAnd.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{maskEmitCType},
                                         callee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[maskAnd.getMask()] = result;
    return mlir::success();
  }

  /// dequantize(%source_i32,%scale,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy i32->f32 runtime-scale dequant sequence:
  ///   converted = __riscv_vfcvt_f_x_v_<dtype><lmul>(source, vl);
  ///   result    = __riscv_vfmul_vf_<dtype><lmul>(converted, scale, vl);
  /// The result vector type (f32) drives the intrinsic suffix; the scale is a
  /// runtime ABI float value mapped to a function parameter directly.
  mlir::LogicalResult
  emitDequantize(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 tcrvrvv::DequantizeOp dequantize,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const {
    // Scope guard: only the dequant-CLAMP epilogue (a compare-select body that
    // begins with one dequantize and then clamps via compare/select) is in this
    // family. The STANDALONE dequantize (load -> dequantize -> store, no select)
    // is the Dequantization owner's consumer and is Gearbox-unrolled by a
    // separate schedule pass the simple for-loop here does NOT reproduce. Refuse
    // to convert a dequantize whose enclosing body carries no tcrv_rvv.select so
    // the standalone body fails to fully legalize and falls back unchanged.
    bool bodyHasSelect = false;
    if (mlir::Block *block = dequantize->getBlock()) {
      for (mlir::Operation &sibling : *block)
        if (llvm::isa<tcrvrvv::SelectOp>(sibling)) {
          bodyHasSelect = true;
          break;
        }
    }
    if (!bodyHasSelect)
      return rewriter.notifyMatchFailure(
          dequantize, "standalone dequantize (no select) is out of scope");
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequantize.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequantize,
                                         "dequantize result not f32 vector");
    mlir::Value source = valueMap.lookup(dequantize.getSource());
    mlir::Value scale = valueMap.lookup(dequantize.getScale());
    if (!source || !scale)
      return rewriter.notifyMatchFailure(dequantize,
                                         "dequantize operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType)
      return rewriter.notifyMatchFailure(dequantize,
                                         "vector type not convertible");
    unsigned sew = vectorElementWidth(resultVecType);
    llvm::StringRef lmul = resultVecType.getLmul();
    llvm::StringRef dtype = vectorDType(resultVecType);

    // converted = vfcvt_f_x_v(source_i32, vl) -- int->float reinterpret-convert.
    std::string convertCallee = riscvIntrinsicName("vfcvt_f_x_v", sew, lmul,
                                                   dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dequantize.getTCRVEmitCLowerableSourceOpName(),
                         dequantize.getTCRVEmitCLowerableSourceRole(),
                         convertCallee));
    mlir::Value converted =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType},
                                         convertCallee,
                                         mlir::ValueRange{source, bodyVL})
            .getResult(0);

    // result = vfmul_vf(converted, scale, vl) -- runtime f32 scale multiply.
    std::string scaleCallee = riscvIntrinsicName("vfmul_vf", sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(dequantize.getTCRVEmitCLowerableSourceOpName(),
                         dequantize.getTCRVEmitCLowerableSourceRole(),
                         scaleCallee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, scaleCallee,
                mlir::ValueRange{converted, scale, bodyVL})
            .getResult(0);
    valueMap[dequantize.getResult()] = result;
    return mlir::success();
  }

  /// masked_binary(%mask,%passthrough,%lhs,%rhs,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy masked merge sequence:
  ///   active = __riscv_v<op>_vv_<dtype><lmul>(lhs, rhs, vl);
  ///   result = __riscv_vmerge_vvm_<dtype><lmul>(passthrough, active, mask, vl);
  mlir::LogicalResult
  emitMaskedBinary(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::MaskedBinaryOp masked,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(masked.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked result not typed vector");
    mlir::Value passthrough = valueMap.lookup(masked.getPassthrough());
    mlir::Value lhs = valueMap.lookup(masked.getLhs());
    mlir::Value rhs = valueMap.lookup(masked.getRhs());
    mlir::Value mask = valueMap.lookup(masked.getMask());
    if (!passthrough || !lhs || !rhs || !mask)
      return rewriter.notifyMatchFailure(masked, "masked operand unmapped");
    std::optional<llvm::StringRef> mnemonic = binaryMnemonic(masked.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(masked, "unsupported masked kind");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(masked, "vector type not convertible");
    unsigned sew = vectorElementWidth(vectorType);
    llvm::StringRef lmul = vectorType.getLmul();
    llvm::StringRef dtype = vectorDType(vectorType);

    // active = vadd/vsub/vmul over the two operand vectors (unmasked).
    std::string arithCallee = riscvIntrinsicName(*mnemonic, sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(masked.getTCRVEmitCLowerableSourceOpName(),
                         masked.getTCRVEmitCLowerableSourceRole(), arithCallee));
    mlir::Value active =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType},
                                         arithCallee,
                                         mlir::ValueRange{lhs, rhs, bodyVL})
            .getResult(0);

    // result = vmerge(passthrough, active, mask, vl) -- inactive lanes keep the
    // passthrough vector.
    std::string mergeCallee = riscvIntrinsicName("vmerge", sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(masked.getTCRVEmitCLowerableSourceOpName(),
                         masked.getTCRVEmitCLowerableSourceRole(), mergeCallee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, mergeCallee,
                mlir::ValueRange{passthrough, active, mask, bodyVL})
            .getResult(0);
    valueMap[masked.getResult()] = result;
    return mlir::success();
  }

  /// macc(%lhs,%rhs,%accumulator,%vl){kind} ->
  ///   __riscv_vmacc_vv_<dtype><lmul>(accumulator, lhs, rhs, vl)
  /// The fused multiply-accumulate (acc += lhs * rhs) writes into the
  /// accumulator vector, so the C call order is (accumulator, lhs, rhs, vl) --
  /// byte-identical to the legacy plain/scalar-broadcast MAcc compute step
  /// (RVVEmitCRoutePlanning oracle: `vmacc_vv_i32m1(acc_vec, lhs_vec, rhs_vec,
  /// vl)`). The scalar-broadcast rung is the SAME op whose rhs is fed by a
  /// tcrv_rvv.splat (lowered by emitSplat); only the operand source differs, the
  /// macc lowering is identical.
  ///
  /// Malformed-body guard: the legacy macc derivation (deriveMAccIntrinsic) is
  /// SEW32-only and requires the explicit separate-accumulator + output-store
  /// layout contracts. A macc whose kind/layout, (dtype, lmul) config, or
  /// operand mapping is outside this bounded slice is NOT lowered here --
  /// notifyMatchFailure rolls the conversion back so the legacy validator still
  /// sees (and rejects/owns) it. Type-correctness is preserved: every operand is
  /// the same typed vector, and the result type is resolved before any emitc op
  /// is created so a non-beachhead config rolls back cleanly.
  mlir::LogicalResult
  emitMAcc(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::MAccOp macc,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(macc.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(macc, "macc result not typed vector");
    // Layout/kind contract guard (mirrors MAccOp::verify + the plain-macc
    // route-family plan): only the bounded add / separate-vector-accumulator /
    // output-store slice is convertible. A body with another kind/layout falls
    // back unchanged rather than being mislowered as a plain fused macc.
    if (macc.getKind() != "add")
      return rewriter.notifyMatchFailure(macc, "unsupported macc kind");
    std::optional<llvm::StringRef> accumulatorLayout =
        macc.getAccumulatorLayout();
    std::optional<llvm::StringRef> resultLayout = macc.getResultLayout();
    if (!accumulatorLayout ||
        *accumulatorLayout != "separate-i32-vector-accumulator-input" ||
        !resultLayout ||
        *resultLayout != "store-multiply-accumulate-result-to-output-buffer")
      return rewriter.notifyMatchFailure(
          macc, "macc accumulator/result layout outside the convertible slice");
    // The fused vmacc intrinsic is SEW32-only in the legacy derivation; an i64
    // (or any non-i32) macc has no __riscv_vmacc_vv_i64* form and must fall back
    // unchanged instead of emitting a non-existent intrinsic.
    if (!vectorType.getElementType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          macc, "macc only lowers the SEW32 fused vmacc slice");

    // Operand-source structural guards (mirror the legacy plain/scalar-broadcast
    // MAcc route-family slice). The bounded slice requires:
    //   - lhs and accumulator are explicit tcrv_rvv.load results,
    //   - rhs is EITHER an explicit tcrv_rvv.load (plain macc) OR a
    //     tcrv_rvv.splat (scalar-broadcast macc),
    //   - NO tcrv_rvv.broadcast_load feeds the macc (broadcast/splat-load macc is
    //     out of the bounded slice -- legacy: "broadcast/splat macc is not in
    //     this bounded slice"),
    //   - if a tcrv_rvv.splat exists in the body the macc MUST consume it as rhs
    //     (a body that splats but bypasses it is the malformed scalar-broadcast
    //     composition the legacy validator rejects).
    // A body outside this shape is NOT lowered here; notifyMatchFailure rolls the
    // conversion back so the legacy validator still sees (and rejects) it.
    mlir::Operation *lhsDef = macc.getLhs().getDefiningOp();
    mlir::Operation *rhsDef = macc.getRhs().getDefiningOp();
    mlir::Operation *accDef = macc.getAccumulator().getDefiningOp();
    if (!llvm::isa_and_present<tcrvrvv::LoadOp>(lhsDef) ||
        !llvm::isa_and_present<tcrvrvv::LoadOp>(accDef))
      return rewriter.notifyMatchFailure(
          macc, "macc lhs/accumulator must be explicit vector loads");
    bool rhsIsLoad = llvm::isa_and_present<tcrvrvv::LoadOp>(rhsDef);
    bool rhsIsSplat = llvm::isa_and_present<tcrvrvv::SplatOp>(rhsDef);
    if (!rhsIsLoad && !rhsIsSplat)
      return rewriter.notifyMatchFailure(
          macc, "macc rhs must be an explicit vector load or scalar splat "
                "(broadcast/splat-load macc is out of the bounded slice)");
    // If the enclosing body carries a tcrv_rvv.splat, the macc must consume it
    // as rhs (scalar-broadcast composition contract). A splatting body whose macc
    // bypasses the splat is the malformed scalar-broadcast macc.
    if (mlir::Block *block = macc->getBlock())
      for (mlir::Operation &sibling : *block)
        if (llvm::isa<tcrvrvv::SplatOp>(sibling) && !rhsIsSplat)
          return rewriter.notifyMatchFailure(
              macc, "scalar-broadcast macc body must consume the splat result "
                    "as rhs");
    // Accumulator ABI-role binding guard: the accumulator load must read the
    // accumulator-input-buffer ABI value, NOT the output buffer (legacy:
    // "accumulator load to bind accumulator-input-buffer, not output buffer").
    auto accLoad = llvm::cast<tcrvrvv::LoadOp>(accDef);
    auto accBufferAbi =
        accLoad.getBuffer().getDefiningOp<tcrvrvv::RuntimeABIValueOp>();
    if (!accBufferAbi ||
        accBufferAbi.getRole() != "accumulator-input-buffer")
      return rewriter.notifyMatchFailure(
          macc, "macc accumulator load must bind the accumulator-input-buffer "
                "ABI role");

    mlir::Value lhs = valueMap.lookup(macc.getLhs());
    mlir::Value rhs = valueMap.lookup(macc.getRhs());
    mlir::Value accumulator = valueMap.lookup(macc.getAccumulator());
    if (!lhs || !rhs || !accumulator)
      return rewriter.notifyMatchFailure(macc, "macc operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(macc, "vector type not convertible");
    std::string callee =
        riscvMAccIntrinsicName(vectorElementWidth(vectorType),
                               vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(macc.getTCRVEmitCLowerableSourceOpName(),
                         macc.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{accumulator, lhs, rhs, bodyVL})
            .getResult(0);
    valueMap[macc.getResult()] = result;
    return mlir::success();
  }

  /// masked_macc(%mask,%lhs,%rhs,%accumulator,%vl){kind} lowers to TWO calls,
  /// byte-identical to the legacy computed-mask macc sequence:
  ///   active = __riscv_vmacc_vv_<dtype><lmul>(accumulator, lhs, rhs, vl);
  ///   result = __riscv_vmerge_vvm_<dtype><lmul>(accumulator, active, mask, vl);
  /// The fused macc multiplies/accumulates on every lane; the merge then keeps
  /// the ACCUMULATOR vector on inactive lanes (the passthrough) and the macc
  /// result on active (mask-true) lanes -- the same passthrough = accumulator
  /// contract the legacy oracle emits.
  mlir::LogicalResult
  emitMaskedMAcc(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 tcrvrvv::MaskedMAccOp masked,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(masked.getResult().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked_macc result not typed vector");
    if (masked.getKind() != "add")
      return rewriter.notifyMatchFailure(masked, "unsupported masked_macc kind");
    if (masked.getAccumulatorLayout() !=
            "separate-i32-vector-accumulator-input" ||
        masked.getResultLayout() !=
            "store-multiply-accumulate-result-to-output-buffer")
      return rewriter.notifyMatchFailure(
          masked,
          "masked_macc accumulator/result layout outside the convertible slice");
    // SEW32-only fused vmacc (see emitMAcc); the masked rung shares the same
    // vmacc derivation.
    if (!vectorType.getElementType().isSignlessInteger(32))
      return rewriter.notifyMatchFailure(
          masked, "masked_macc only lowers the SEW32 fused vmacc slice");
    auto maskType =
        llvm::dyn_cast<tcrvrvv::MaskType>(masked.getMask().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(masked,
                                         "masked_macc mask not typed mask");
    mlir::Value mask = valueMap.lookup(masked.getMask());
    mlir::Value lhs = valueMap.lookup(masked.getLhs());
    mlir::Value rhs = valueMap.lookup(masked.getRhs());
    mlir::Value accumulator = valueMap.lookup(masked.getAccumulator());
    if (!mask || !lhs || !rhs || !accumulator)
      return rewriter.notifyMatchFailure(masked, "masked_macc operand unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(masked, "vector type not convertible");
    unsigned sew = vectorElementWidth(vectorType);
    llvm::StringRef lmul = vectorType.getLmul();
    llvm::StringRef dtype = vectorDType(vectorType);

    // active = vmacc_vv(accumulator, lhs, rhs, vl) -- fused multiply-accumulate
    // on every lane.
    std::string maccCallee = riscvMAccIntrinsicName(sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(masked.getTCRVEmitCLowerableSourceOpName(),
                         masked.getTCRVEmitCLowerableSourceRole(), maccCallee));
    mlir::Value active =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, maccCallee,
                mlir::ValueRange{accumulator, lhs, rhs, bodyVL})
            .getResult(0);

    // result = vmerge(accumulator, active, mask, vl) -- inactive lanes keep the
    // accumulator passthrough vector.
    std::string mergeCallee = riscvIntrinsicName("vmerge", sew, lmul, dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(masked.getTCRVEmitCLowerableSourceOpName(),
                         masked.getTCRVEmitCLowerableSourceRole(), mergeCallee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, mergeCallee,
                mlir::ValueRange{accumulator, active, mask, bodyVL})
            .getResult(0);
    valueMap[masked.getResult()] = result;
    return mlir::success();
  }

  /// strided_load(%abi,%stride,%vl) ->
  ///   off = i * stride; ptr = base + off;
  ///   bytestride = (ptrdiff_t)stride * (ptrdiff_t)4;
  ///   __riscv_vlse<sew>_v_<dtype><lmul>(ptr, bytestride, vl)
  mlir::LogicalResult
  emitStridedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::StridedLoadOp load,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(load.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(load, "strided load result not vector");
    mlir::Value base = valueMap.lookup(load.getBuffer());
    mlir::Value stride = valueMap.lookup(load.getStride());
    if (!base || !stride)
      return rewriter.notifyMatchFailure(load, "strided load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          load, "strided load buffer C type disagrees with loaded element");
    // Byte-stride contract guard. The base-memory strided family (its loaded
    // vector flows through a tcrv_rvv.move into a plain store) is driven by a
    // runtime BYTE stride; a body in that shape whose stride ABI value is NOT a
    // byte-stride role is malformed (the legacy validator rejects "source
    // byte-strided load requires source-byte-stride runtime ABI value"). Refuse
    // it so the malformed body falls back rather than being mislowered as an
    // element-strided load.
    if (loadedFeedsMove(load) && !isByteStride(load.getStride()))
      return rewriter.notifyMatchFailure(
          load, "base-memory strided load requires a byte-stride ABI role");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(load, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vlse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(load.getTCRVEmitCLowerableSourceOpName(),
                         load.getTCRVEmitCLowerableSourceRole(), callee));
    // Two strided rungs share tcrv_rvv.strided_load: the base-memory family
    // passes a runtime BYTE stride (uint8_t-cast addressing, stride AS-IS), the
    // elementwise family passes an ELEMENT stride (scaled-element pointer +
    // ptrdiff_t byte stride). Pick the addressing from the typed ABI role.
    mlir::Value ptr;
    mlir::Value byteStride;
    if (isByteStride(load.getStride())) {
      ptr = emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
      if (!ptr)
        return rewriter.notifyMatchFailure(
            load, "strided load base must be a pointer-typed ABI param");
      byteStride = stride;
    } else {
      ptr = emitScaledPointer(rewriter, loc, base, inductionVar, stride);
      byteStride = emitByteStride(rewriter, loc, stride, vectorType);
    }
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{ptr, byteStride,
                                                          bodyVL})
            .getResult(0);
    valueMap[load.getLoaded()] = loaded;
    return mlir::success();
  }

  /// strided_store(%abi,%val,%stride,%vl) ->
  ///   off = i * stride; ptr = base + off;
  ///   bytestride = (ptrdiff_t)stride * (ptrdiff_t)4;
  ///   __riscv_vsse<sew>_v_<dtype><lmul>(ptr, bytestride, val, vl)
  mlir::LogicalResult
  emitStridedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::StridedStoreOp store,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(store.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(store, "strided store value not vector");
    mlir::Value base = valueMap.lookup(store.getBuffer());
    mlir::Value value = valueMap.lookup(store.getValue());
    mlir::Value stride = valueMap.lookup(store.getStride());
    if (!base || !value || !stride)
      return rewriter.notifyMatchFailure(store, "strided store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          store, "strided store buffer C type disagrees with stored element");
    // Byte-stride contract guard (see emitStridedLoad): the base-memory strided
    // store family (its stored value comes from a tcrv_rvv.move of a unit-stride
    // load) requires a byte-stride ABI role. Refuse a non-byte-stride role in
    // that shape so the malformed body falls back instead of being mislowered.
    if (storedValueFromMove(store) && !isByteStride(store.getStride()))
      return rewriter.notifyMatchFailure(
          store, "base-memory strided store requires a byte-stride ABI role");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(store, "vector type not convertible");
    std::string callee =
        riscvIntrinsicName("vsse", vectorElementWidth(vectorType),
                           vectorType.getLmul(), vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(store.getTCRVEmitCLowerableSourceOpName(),
                         store.getTCRVEmitCLowerableSourceRole(), callee));
    // See emitStridedLoad: select the byte-stride vs element-stride addressing
    // from the stride ABI role.
    mlir::Value ptr;
    mlir::Value byteStride;
    if (isByteStride(store.getStride())) {
      ptr = emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
      if (!ptr)
        return rewriter.notifyMatchFailure(
            store, "strided store base must be a pointer-typed ABI param");
      byteStride = stride;
    } else {
      ptr = emitScaledPointer(rewriter, loc, base, inductionVar, stride);
      byteStride = emitByteStride(rewriter, loc, stride, vectorType);
    }
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{ptr, byteStride, value, bodyVL});
    return mlir::success();
  }

  /// Resolve the segment2 field-vector (sew, lmul, dtype) facts plus the tuple C
  /// type and per-field emitc vector type. Returns false (the caller fails the
  /// match) for any element/lmul outside the named grid so a non-i32 / non-{m1}
  /// segment2 body falls back unconverted rather than being mislowered.
  struct Segment2Facts {
    unsigned sew = 0;
    llvm::StringRef lmul;
    llvm::StringRef dtype;
    mlir::Type fieldVecType;
    mlir::Type tupleType;
  };
  bool resolveSegment2Facts(mlir::ConversionPatternRewriter &rewriter,
                            tcrvrvv::VectorType fieldType,
                            Segment2Facts &out) const {
    // Only the signed-integer field grid is in scope for the bounded segment2
    // slice (the tuple type spelling is vint<sew>m<lmul>x2_t). A float field
    // would need a different tuple/intrinsic family.
    if (isFloatVector(fieldType))
      return false;
    out.sew = vectorElementWidth(fieldType);
    out.lmul = fieldType.getLmul();
    out.dtype = vectorDType(fieldType);
    if (out.dtype.empty() || out.sew == 0)
      return false;
    // The per-field vector type must lower through the bounded converter grid;
    // reject otherwise so a non-beachhead (sew, lmul) falls back.
    out.fieldVecType = convertVectorTypeToEmitC(fieldType);
    if (!out.fieldVecType)
      return false;
    out.tupleType = emitc::OpaqueType::get(
        rewriter.getContext(), riscvSegment2TupleCType(out.sew, out.lmul));
    return true;
  }

  /// The interleaved segment2 base pointer: `base + (i * 2)`. The interleaved
  /// memory holds the two fields adjacent per element, so the segment base
  /// advances by 2*chunk -- byte-identical to the legacy segment2 oracle
  /// (`size_t off = i * 2; ptr = base + off`).
  mlir::Value emitSegment2InterleavedPointer(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value base, mlir::Value inductionVar) const {
    mlir::Value two = rewriter.create<emitc::LiteralOp>(
        loc, inductionVar.getType(), "2");
    mlir::Value off = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, two);
    return rewriter.create<emitc::AddOp>(loc, base.getType(), base, off);
  }

  /// segment2_load(%src,%vl) -> field0, field1. The interleaved deinterleave
  /// load reads one segment2 tuple from the interleaved source, then the two
  /// tcrv_rvv.move ops extract the fields (emitSegment2FieldExtract via
  /// emitMove). Here we emit ONLY the tuple load and record (tuple, index) for
  /// each field result so the move-sourced vget can resolve it:
  ///   ptr = src + (i * 2);
  ///   vint<sew>m<lmul>x2_t tuple = __riscv_vlseg2e<sew>_v_<dtype><lmul>x2(ptr, vl)
  mlir::LogicalResult emitSegment2Load(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::Segment2LoadOp segLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
          &segmentFieldMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segLoad.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segLoad,
                                         "only segment_count = 2 is in scope");
    if (segLoad.getSourceMemoryForm() != "segment2-interleaved-unit-stride-load")
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load source memory form outside the slice");
    auto field0Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segLoad.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segLoad.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load field vector type not convertible");
    mlir::Value base = valueMap.lookup(segLoad.getSource());
    if (!base)
      return rewriter.notifyMatchFailure(segLoad,
                                         "segment2_load source not an ABI param");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segLoad, "segment2_load source C type disagrees with field element");

    std::string callee =
        riscvSegment2LoadIntrinsicName(facts.sew, facts.lmul, facts.dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segLoad.getTCRVEmitCLowerableSourceOpName(),
                         segLoad.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitSegment2InterleavedPointer(rewriter, loc, base, inductionVar);
    mlir::Value tuple =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{facts.tupleType},
                                         callee, mlir::ValueRange{ptr, bodyVL})
            .getResult(0);
    // The two move ops sourced from these field results emit the vget extracts.
    segmentFieldMap[segLoad.getField0()] = {tuple, 0};
    segmentFieldMap[segLoad.getField1()] = {tuple, 1};
    return mlir::success();
  }

  /// segment2_store(%dst,%field0,%field1,%vl). The interleave store packs the
  /// two field vectors into one tuple, then stores it to the interleaved
  /// destination (TWO emitted steps, byte-identical to the legacy oracle):
  ///   vint<sew>m<lmul>x2_t tuple = __riscv_vcreate_v_<dtype><lmul>x2(f0, f1);
  ///   ptr = dst + (i * 2);
  ///   __riscv_vsseg2e<sew>_v_<dtype><lmul>x2(ptr, tuple, vl)
  mlir::LogicalResult
  emitSegment2Store(mlir::ConversionPatternRewriter &rewriter,
                    mlir::Location loc, tcrvrvv::Segment2StoreOp segStore,
                    llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                    mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segStore.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segStore,
                                         "only segment_count = 2 is in scope");
    if (segStore.getDestinationMemoryForm() !=
        "segment2-interleaved-unit-stride-store")
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store destination memory form outside the slice");
    auto field0Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segStore.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segStore.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store field vector type not convertible");
    mlir::Value base = valueMap.lookup(segStore.getDestination());
    mlir::Value field0 = valueMap.lookup(segStore.getField0());
    mlir::Value field1 = valueMap.lookup(segStore.getField1());
    if (!base || !field0 || !field1)
      return rewriter.notifyMatchFailure(segStore,
                                         "segment2_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store dst C type disagrees with field element");
    // Field-binding guard: the interleave field0/field1 operands must bind the
    // field0/field1 input loads (structural authority via the load buffer role).
    // A body that swaps them (segment2_store %dst, %field1, %field0) is the
    // operand-binding negative the legacy provider rejects -- fall back so it is
    // not silently mislowered.
    if (!fieldVectorBindsLoadRole(segStore.getField0(),
                                  segStore.getField0Role()) ||
        !fieldVectorBindsLoadRole(segStore.getField1(),
                                  segStore.getField1Role()))
      return rewriter.notifyMatchFailure(
          segStore, "segment2_store must consume matching field0/field1 load "
                    "results bound to the field0/field1 input roles");

    // Step 1: pack the two fields into one tuple.
    std::string createCallee =
        riscvSegment2TupleCreateIntrinsicName(facts.dtype, facts.lmul);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segStore.getTCRVEmitCLowerableSourceOpName(),
                         segStore.getTCRVEmitCLowerableSourceRole(),
                         createCallee));
    mlir::Value tuple =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{facts.tupleType},
                                         createCallee,
                                         mlir::ValueRange{field0, field1})
            .getResult(0);

    // Step 2: store the tuple to the interleaved destination.
    std::string storeCallee =
        riscvSegment2StoreIntrinsicName(facts.sew, facts.lmul, facts.dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segStore.getTCRVEmitCLowerableSourceOpName(),
                         segStore.getTCRVEmitCLowerableSourceRole(),
                         storeCallee));
    mlir::Value ptr =
        emitSegment2InterleavedPointer(rewriter, loc, base, inductionVar);
    rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{}, storeCallee,
                                         mlir::ValueRange{ptr, tuple, bodyVL});
    return mlir::success();
  }

  /// masked_segment2_load(%src,%mask,%pass0,%pass1,%vl) -> field0, field1. The
  /// computed-mask segment2 load packs the two old-destination passthroughs into
  /// a tuple, masked-loads the interleaved source into a result tuple, then
  /// extracts the two fields (FOUR emitted steps, byte-identical to the legacy
  /// computed-mask segment2 load oracle):
  ///   pass = __riscv_vcreate_v_<dtype><lmul>x2(old0, old1);
  ///   ptr = src + (i * 2);
  ///   tuple = __riscv_vlseg2e<sew>_v_<dtype><lmul>x2_tumu(mask, pass, ptr, vl);
  ///   field0 = __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, 0);
  ///   field1 = __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, 1)
  /// The mask MUST come from a compare in the same scope (the computed-mask
  /// family authority); refuse any other producer so a malformed body falls
  /// back to the legacy validator.
  mlir::LogicalResult emitMaskedSegment2Load(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedSegment2LoadOp segLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segLoad.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segLoad,
                                         "only segment_count = 2 is in scope");
    if (segLoad.getSourceMemoryForm() !=
            "segment2-interleaved-unit-stride-load" ||
        segLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load form/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(segLoad.getMask()))
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load mask must come from a compare in the "
                   "same scope");
    auto field0Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segLoad.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segLoad.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load field vector type not convertible");
    mlir::Value base = valueMap.lookup(segLoad.getSource());
    mlir::Value mask = valueMap.lookup(segLoad.getMask());
    mlir::Value pass0 = valueMap.lookup(segLoad.getPassthrough0());
    mlir::Value pass1 = valueMap.lookup(segLoad.getPassthrough1());
    if (!base || !mask || !pass0 || !pass1)
      return rewriter.notifyMatchFailure(segLoad,
                                         "masked_segment2_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segLoad, "masked_segment2_load src C type disagrees with field "
                   "element");

    // Step 1: pack the two passthroughs into the passthrough tuple.
    std::string createCallee =
        riscvSegment2TupleCreateIntrinsicName(facts.dtype, facts.lmul);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segLoad.getTCRVEmitCLowerableSourceOpName(),
                         segLoad.getTCRVEmitCLowerableSourceRole(),
                         createCallee));
    mlir::Value passTuple =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{facts.tupleType},
                                         createCallee,
                                         mlir::ValueRange{pass0, pass1})
            .getResult(0);

    // Step 2: masked tuple load from the interleaved source.
    std::string loadCallee =
        riscvMaskedSegment2LoadIntrinsicName(facts.sew, facts.lmul, facts.dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segLoad.getTCRVEmitCLowerableSourceOpName(),
                         segLoad.getTCRVEmitCLowerableSourceRole(), loadCallee));
    mlir::Value ptr =
        emitSegment2InterleavedPointer(rewriter, loc, base, inductionVar);
    mlir::Value tuple =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{facts.tupleType}, loadCallee,
                mlir::ValueRange{mask, passTuple, ptr, bodyVL})
            .getResult(0);

    // Steps 3 & 4: extract the two field vectors.
    std::string getCallee =
        riscvSegment2FieldExtractIntrinsicName(facts.dtype, facts.lmul);
    mlir::Value field0 = emitSegment2FieldExtract(
        rewriter, loc, segLoad.getTCRVEmitCLowerableSourceOpName(),
        segLoad.getTCRVEmitCLowerableSourceRole(), getCallee, tuple,
        facts.fieldVecType, 0);
    mlir::Value field1 = emitSegment2FieldExtract(
        rewriter, loc, segLoad.getTCRVEmitCLowerableSourceOpName(),
        segLoad.getTCRVEmitCLowerableSourceRole(), getCallee, tuple,
        facts.fieldVecType, 1);
    valueMap[segLoad.getField0()] = field0;
    valueMap[segLoad.getField1()] = field1;
    return mlir::success();
  }

  /// masked_segment2_store(%dst,%mask,%field0,%field1,%vl). The computed-mask
  /// segment2 store packs the two payload fields into a tuple, then
  /// masked-stores it to the interleaved destination (TWO emitted steps,
  /// byte-identical to the legacy computed-mask segment2 store oracle):
  ///   tuple = __riscv_vcreate_v_<dtype><lmul>x2(f0, f1);
  ///   ptr = dst + (i * 2);
  ///   __riscv_vsseg2e<sew>_v_<dtype><lmul>x2_m(mask, ptr, tuple, vl)
  /// The mask MUST come from a compare in the same scope; refuse otherwise.
  mlir::LogicalResult emitMaskedSegment2Store(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedSegment2StoreOp segStore,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const {
    if (segStore.getSegmentCount() != 2)
      return rewriter.notifyMatchFailure(segStore,
                                         "only segment_count = 2 is in scope");
    if (segStore.getDestinationMemoryForm() !=
            "segment2-interleaved-unit-stride-store" ||
        segStore.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store form/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(segStore.getMask()))
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store mask must come from a compare in the "
                    "same scope");
    auto field0Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segStore.getField0().getType());
    auto field1Type =
        llvm::dyn_cast<tcrvrvv::VectorType>(segStore.getField1().getType());
    if (!field0Type || !field1Type || field0Type != field1Type)
      return rewriter.notifyMatchFailure(
          segStore,
          "masked_segment2_store fields must be matching typed vectors");
    Segment2Facts facts;
    if (!resolveSegment2Facts(rewriter, field0Type, facts))
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store field vector type not convertible");
    mlir::Value base = valueMap.lookup(segStore.getDestination());
    mlir::Value mask = valueMap.lookup(segStore.getMask());
    mlir::Value field0 = valueMap.lookup(segStore.getField0());
    mlir::Value field1 = valueMap.lookup(segStore.getField1());
    if (!base || !mask || !field0 || !field1)
      return rewriter.notifyMatchFailure(segStore,
                                         "masked_segment2_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, field0Type))
      return rewriter.notifyMatchFailure(
          segStore, "masked_segment2_store dst C type disagrees with field "
                    "element");

    // Step 1: pack the two payload fields into one tuple.
    std::string createCallee =
        riscvSegment2TupleCreateIntrinsicName(facts.dtype, facts.lmul);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segStore.getTCRVEmitCLowerableSourceOpName(),
                         segStore.getTCRVEmitCLowerableSourceRole(),
                         createCallee));
    mlir::Value tuple =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{facts.tupleType},
                                         createCallee,
                                         mlir::ValueRange{field0, field1})
            .getResult(0);

    // Step 2: masked tuple store to the interleaved destination.
    std::string storeCallee =
        riscvMaskedSegment2StoreIntrinsicName(facts.sew, facts.lmul,
                                              facts.dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(segStore.getTCRVEmitCLowerableSourceOpName(),
                         segStore.getTCRVEmitCLowerableSourceRole(),
                         storeCallee));
    mlir::Value ptr =
        emitSegment2InterleavedPointer(rewriter, loc, base, inductionVar);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, storeCallee,
        mlir::ValueRange{mask, ptr, tuple, bodyVL});
    return mlir::success();
  }

  /// Emit one __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, idx) field
  /// extract, with the step provenance comment carrying the (op name, role) of
  /// the segment op that owns the extract.
  mlir::Value emitSegment2FieldExtract(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      llvm::StringRef opName, llvm::StringRef role, llvm::StringRef callee,
      mlir::Value tuple, mlir::Type fieldVecType, unsigned index) const {
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
    mlir::Value indexLiteral = rewriter.create<emitc::LiteralOp>(
        loc, rewriter.getIndexType(), llvm::Twine(index).str());
    return rewriter
        .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{fieldVecType}, callee,
                                     mlir::ValueRange{tuple, indexLiteral})
        .getResult(0);
  }

  /// move{copy}(%src,%vl) -> passthrough OR segment2 field extract. Two shapes:
  ///
  ///  (a) base-memory movement family: the loaded vector is marked as the store
  ///      value with a no-op copy move (structural body authority). The copy
  ///      carries no compute, so it maps the result SSA value to the same emitc
  ///      Value -- the legacy oracle emits NO call for it.
  ///
  ///  (b) segment2 deinterleave family: the move's source is a tcrv_rvv.move-
  ///      observed segment2_load field result. The legacy oracle emits a
  ///      __riscv_vget_v_<dtype><lmul>x2_<dtype><lmul>(tuple, idx) extract at the
  ///      move's position (role=compute). Detect the segmentFieldMap entry and
  ///      emit the vget; otherwise the plain identity copy.
  ///
  /// Only kind = "copy" is in this bounded slice; any other movement kind falls
  /// back so a semantically meaningful move is never silently dropped.
  mlir::LogicalResult
  emitMove(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::MoveOp move,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
               &segmentFieldMap) const {
    if (move.getKind() != "copy")
      return rewriter.notifyMatchFailure(move, "unsupported move kind");
    // segment2 deinterleave: the move sources a segment2_load field result.
    auto segIt = segmentFieldMap.find(move.getSource());
    if (segIt != segmentFieldMap.end()) {
      auto vectorType =
          llvm::dyn_cast<tcrvrvv::VectorType>(move.getResult().getType());
      if (!vectorType)
        return rewriter.notifyMatchFailure(
            move, "segment2 move result not a typed vector");
      Segment2Facts facts;
      if (!resolveSegment2Facts(rewriter, vectorType, facts))
        return rewriter.notifyMatchFailure(
            move, "segment2 move field vector type not convertible");
      mlir::Value tuple = segIt->second.first;
      unsigned index = segIt->second.second;
      std::string callee =
          riscvSegment2FieldExtractIntrinsicName(facts.dtype, facts.lmul);
      mlir::Value field = emitSegment2FieldExtract(
          rewriter, loc, move.getTCRVEmitCLowerableSourceOpName(),
          move.getTCRVEmitCLowerableSourceRole(), callee, tuple,
          facts.fieldVecType, index);
      valueMap[move.getResult()] = field;
      return mlir::success();
    }
    mlir::Value source = valueMap.lookup(move.getSource());
    if (!source)
      return rewriter.notifyMatchFailure(move, "move source unmapped");
    valueMap[move.getResult()] = source;
    return mlir::success();
  }

  /// index_load(%abi,%vl) ->
  ///   ptr = index_buf + i; __riscv_vle<eew>_v_u<eew>m<lmul>(ptr, vl)
  /// Loads the UNSIGNED element-index/offset vector for an indexed gather/
  /// scatter. The index buffer is read unit-stride (`index_buf + i`), exactly
  /// like a plain load but into the unsigned index vector type -- byte-identical
  /// to the legacy index_load oracle (`__riscv_vle32_v_u32m1(index + i, vl)`).
  mlir::LogicalResult
  emitIndexLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                tcrvrvv::IndexLoadOp indexLoad,
                llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto indexVecType =
        llvm::dyn_cast<tcrvrvv::IndexVectorType>(indexLoad.getLoaded().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "index_load result not index vector");
    mlir::Value base = valueMap.lookup(indexLoad.getIndex());
    if (!base)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "index_load buffer not an ABI param");
    // The index buffer MUST be an unsigned-32 pointer (the index vectors are
    // u32). A mismatched index pointee width would dereference at the wrong
    // element width -- reject so the malformed body falls back.
    if (!indexBufferIsU32(base))
      return rewriter.notifyMatchFailure(
          indexLoad, "index_load buffer C type is not a uint32_t pointer");
    mlir::Type vecType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!vecType)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "index vector type not convertible");
    unsigned eew = static_cast<unsigned>(indexLoad.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(indexLoad,
                                         "only EEW=32 index loads are in scope");
    std::string callee = riscvIntrinsicName("vle", eew, indexVecType.getLmul(),
                                            "u32");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(indexLoad.getTCRVEmitCLowerableSourceOpName(),
                         indexLoad.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{ptr, bodyVL})
            .getResult(0);

    // Computed-mask indexed path: the index_load feeds a masked indexed
    // gather/scatter. The string-plan byte-order (which the harness ordered-
    // token validator depends on) emits the element->byte scale EARLY, right
    // after the index_load and before the splat/loads/compare. Emit the scale
    // here and map the index_load result to the SCALED byte-offset vector so the
    // masked-indexed emitter consumes it directly (it skips its own scale). The
    // base-memory plain indexed path leaves the raw index here and scales inside
    // its own emitter (no early ordering constraint).
    if (mlir::Operation *maskedConsumer = maskedIndexedConsumer(indexLoad)) {
      tcrvrvv::VectorType dataVectorType = maskedIndexedDataVectorType(
          maskedConsumer);
      if (!dataVectorType)
        return rewriter.notifyMatchFailure(
            indexLoad, "masked indexed consumer data vector not typed");
      llvm::StringRef consumerOpName;
      llvm::StringRef consumerRole;
      if (auto load =
              llvm::dyn_cast<tcrvrvv::MaskedIndexedLoadOp>(maskedConsumer)) {
        consumerOpName = load.getTCRVEmitCLowerableSourceOpName();
        consumerRole = load.getTCRVEmitCLowerableSourceRole();
      } else {
        auto store = llvm::cast<tcrvrvv::MaskedIndexedStoreOp>(maskedConsumer);
        consumerOpName = store.getTCRVEmitCLowerableSourceOpName();
        consumerRole = store.getTCRVEmitCLowerableSourceRole();
      }
      mlir::Value byteIndices = emitIndexByteScale(
          rewriter, loc, consumerOpName, consumerRole, loaded, vecType,
          indexVecType, dataVectorType, bodyVL);
      valueMap[indexLoad.getLoaded()] = byteIndices;
      return mlir::success();
    }

    valueMap[indexLoad.getLoaded()] = loaded;
    return mlir::success();
  }

  /// The single masked indexed gather/scatter op that consumes `indexLoad`'s
  /// result, or null if the index feeds a plain (base-memory) indexed op. The
  /// computed-mask indexed path scales the index early (see emitIndexLoad); the
  /// plain path scales inside its own emitter.
  static mlir::Operation *maskedIndexedConsumer(tcrvrvv::IndexLoadOp indexLoad) {
    for (mlir::Operation *user : indexLoad.getLoaded().getUsers())
      if (llvm::isa<tcrvrvv::MaskedIndexedLoadOp,
                    tcrvrvv::MaskedIndexedStoreOp>(user))
        return user;
    return nullptr;
  }

  /// The data (payload) vector type of a masked indexed gather (its loaded
  /// result) or scatter (its stored value), used to size the element->byte
  /// index scale.
  static tcrvrvv::VectorType
  maskedIndexedDataVectorType(mlir::Operation *maskedConsumer) {
    if (auto load = llvm::dyn_cast<tcrvrvv::MaskedIndexedLoadOp>(maskedConsumer))
      return llvm::dyn_cast<tcrvrvv::VectorType>(load.getLoaded().getType());
    auto store = llvm::cast<tcrvrvv::MaskedIndexedStoreOp>(maskedConsumer);
    return llvm::dyn_cast<tcrvrvv::VectorType>(store.getValue().getType());
  }

  /// indexed_load(%data,%indices,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   loaded = __riscv_vloxei<eew>_v_<dtype><lmul>(data_base, bytes, vl)
  /// The element index vector is scaled to a BYTE offset vector, then the
  /// ordered indexed (gather) access reads `data_base[byte_offset]` per lane.
  /// The data base is NOT offset by the induction var (a gather reads scattered
  /// elements relative to the buffer head) -- byte-identical to the legacy
  /// indexed_load oracle.
  mlir::LogicalResult
  emitIndexedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::IndexedLoadOp indexedLoad,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(indexedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load result not typed vector");
    if (indexedLoad.getOffsetUnit() != "element")
      return rewriter.notifyMatchFailure(
          indexedLoad, "only element-offset indexed loads are in scope");
    auto indexVecType = llvm::dyn_cast<tcrvrvv::IndexVectorType>(
        indexedLoad.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load indices not index vector");
    mlir::Value dataBase = valueMap.lookup(indexedLoad.getData());
    mlir::Value indices = valueMap.lookup(indexedLoad.getIndices());
    if (!dataBase || !indices)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dataBase, vectorType))
      return rewriter.notifyMatchFailure(
          indexedLoad, "indexed_load data C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!vecType || !indexEmitCType)
      return rewriter.notifyMatchFailure(indexedLoad,
                                         "indexed_load type not convertible");
    unsigned eew = static_cast<unsigned>(indexedLoad.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          indexedLoad, "only EEW=32 indexed loads are in scope");

    mlir::Value byteIndices = emitIndexByteScale(
        rewriter, loc, indexedLoad.getTCRVEmitCLowerableSourceOpName(),
        indexedLoad.getTCRVEmitCLowerableSourceRole(), indices, indexEmitCType,
        indexVecType, vectorType, bodyVL);
    std::string callee = riscvIndexedMemoryIntrinsicName(
        "vloxei", eew, vectorDType(vectorType), vectorType.getLmul());
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(indexedLoad.getTCRVEmitCLowerableSourceOpName(),
                         indexedLoad.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{dataBase, byteIndices, bodyVL})
            .getResult(0);
    valueMap[indexedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

  /// indexed_store(%dst,%indices,%value,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   __riscv_vsoxei<eew>_v_<dtype><lmul>(dst_base, bytes, value, vl)
  /// The element index vector is byte-scaled, then the ordered indexed
  /// (scatter) access writes `dst_base[byte_offset] = value[lane]`. The dst
  /// base is NOT offset by the induction var -- byte-identical to the legacy
  /// indexed_store oracle. Only the unique-index slice is accepted (duplicate
  /// resolution is not modeled).
  mlir::LogicalResult
  emitIndexedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                   tcrvrvv::IndexedStoreOp indexedStore,
                   llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                   mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(indexedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(indexedStore,
                                         "indexed_store value not typed vector");
    if (indexedStore.getOffsetUnit() != "element")
      return rewriter.notifyMatchFailure(
          indexedStore, "only element-offset indexed stores are in scope");
    if (indexedStore.getIndexUniqueness() != "unique")
      return rewriter.notifyMatchFailure(
          indexedStore, "only unique-index indexed stores are in scope");
    auto indexVecType = llvm::dyn_cast<tcrvrvv::IndexVectorType>(
        indexedStore.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(
          indexedStore, "indexed_store indices not index vector");
    mlir::Value dstBase = valueMap.lookup(indexedStore.getDestination());
    mlir::Value indices = valueMap.lookup(indexedStore.getIndices());
    mlir::Value value = valueMap.lookup(indexedStore.getValue());
    if (!dstBase || !indices || !value)
      return rewriter.notifyMatchFailure(indexedStore,
                                         "indexed_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dstBase, vectorType))
      return rewriter.notifyMatchFailure(
          indexedStore,
          "indexed_store destination C type disagrees with stored element");
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!convertVectorTypeToEmitC(vectorType) || !indexEmitCType)
      return rewriter.notifyMatchFailure(indexedStore,
                                         "indexed_store type not convertible");
    unsigned eew = static_cast<unsigned>(indexedStore.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          indexedStore, "only EEW=32 indexed stores are in scope");

    mlir::Value byteIndices = emitIndexByteScale(
        rewriter, loc, indexedStore.getTCRVEmitCLowerableSourceOpName(),
        indexedStore.getTCRVEmitCLowerableSourceRole(), indices, indexEmitCType,
        indexVecType, vectorType, bodyVL);
    std::string callee = riscvIndexedMemoryIntrinsicName(
        "vsoxei", eew, vectorDType(vectorType), vectorType.getLmul());
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(indexedStore.getTCRVEmitCLowerableSourceOpName(),
                         indexedStore.getTCRVEmitCLowerableSourceRole(),
                         callee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{dstBase, byteIndices, value, bodyVL});
    return mlir::success();
  }

  /// The element->byte index scale shared by indexed load and store:
  ///   __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl)
  /// where elemBytes is the data element byte width (4 for i32). The verbatim
  /// step comment is carried from the indexed source op so the rendered C keeps
  /// the `callee=__riscv_vmul_vx_u32m1` provenance line.
  mlir::Value
  emitIndexByteScale(mlir::ConversionPatternRewriter &rewriter,
                     mlir::Location loc, llvm::StringRef sourceOpName,
                     llvm::StringRef sourceRole, mlir::Value indices,
                     mlir::Type indexEmitCType,
                     tcrvrvv::IndexVectorType indexVecType,
                     tcrvrvv::VectorType dataVectorType,
                     mlir::Value bodyVL) const {
    std::string scaleCallee =
        riscvIndexScaleIntrinsicName("u32", indexVecType.getLmul());
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(sourceOpName, sourceRole, scaleCallee));
    unsigned elemBytes = vectorElementWidth(dataVectorType) / 8;
    mlir::Value bytesLiteral = rewriter.create<emitc::LiteralOp>(
        loc, emitc::OpaqueType::get(rewriter.getContext(), "size_t"),
        llvm::Twine(elemBytes).str());
    return rewriter
        .create<emitc::CallOpaqueOp>(
            loc, mlir::TypeRange{indexEmitCType}, scaleCallee,
            mlir::ValueRange{indices, bytesLiteral, bodyVL})
        .getResult(0);
  }

  /// mask_load(%abi,%vl) -> TWO calls:
  ///   maskvec = __riscv_vle<sew>_v_<dtype><lmul>(mask_buf + i, vl);
  ///   mask    = __riscv_vmsne_vx_<dtype><lmul>_b<maskbits>(maskvec, 0, vl)
  /// The base-memory masked families compute their predicate from a runtime mask
  /// BUFFER: load it unit-stride as a data vector, then test each lane != 0 to
  /// produce the vbool predicate -- byte-identical to the legacy mask_load
  /// oracle. The mask is genuine mask_load authority (NOT a compare on data),
  /// which is exactly the legality the negative fixtures require.
  mlir::LogicalResult
  emitMaskLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
               tcrvrvv::MaskLoadOp maskLoad,
               llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
               mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto maskType =
        llvm::dyn_cast<tcrvrvv::MaskType>(maskLoad.getLoaded().getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(maskLoad,
                                         "mask_load result not typed mask");
    if (maskLoad.getMaskMemoryForm() != "unit-stride-mask-load")
      return rewriter.notifyMatchFailure(
          maskLoad, "only unit-stride mask loads are in scope");
    mlir::Value base = valueMap.lookup(maskLoad.getMask());
    if (!base)
      return rewriter.notifyMatchFailure(maskLoad,
                                         "mask_load buffer not an ABI param");
    // The mask buffer is read at the predicate element width (the data vector
    // element width); a mismatched pointee would test the wrong width.
    unsigned sew = 0;
    if (maskType.getElementType().isSignlessInteger(32))
      sew = 32;
    else if (maskType.getElementType().isSignlessInteger(64))
      sew = 64;
    else
      return rewriter.notifyMatchFailure(maskLoad, "unsupported mask element");
    llvm::StringRef dtype;
    if (sew == 32)
      dtype = "i32";
    else
      dtype = "i64";
    if (!maskBufferPointeeMatches(base, dtype))
      return rewriter.notifyMatchFailure(
          maskLoad, "mask_load buffer C type disagrees with mask element width");
    unsigned maskBits = maskWidthForConfig(sew, maskType.getLmul());
    if (maskBits == 0)
      return rewriter.notifyMatchFailure(maskLoad, "unsupported mask config");
    mlir::Type maskEmitCType = getTypeConverter()->convertType(maskType);
    if (!maskEmitCType ||
        maskEmitCType.getDialect().getNamespace() !=
            emitc::EmitCDialect::getDialectNamespace())
      return rewriter.notifyMatchFailure(maskLoad, "mask type not convertible");
    mlir::Type dataVecEmitCType =
        emitc::OpaqueType::get(rewriter.getContext(),
                               ("vint" + llvm::Twine(sew) +
                                maskType.getLmul() + "_t")
                                   .str());

    // Step 1: unit-stride load the mask buffer as a data vector.
    std::string loadCallee = riscvIntrinsicName("vle", sew, maskType.getLmul(),
                                                dtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskLoad.getTCRVEmitCLowerableSourceOpName(),
                         maskLoad.getTCRVEmitCLowerableSourceRole(),
                         loadCallee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    mlir::Value maskVec =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{dataVecEmitCType},
                                         loadCallee,
                                         mlir::ValueRange{ptr, bodyVL})
            .getResult(0);

    // Step 2: lane != 0 -> predicate mask.
    std::string maskCallee =
        riscvMaskNonzeroIntrinsicName(sew, maskType.getLmul(), dtype, maskBits);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskLoad.getTCRVEmitCLowerableSourceOpName(),
                         maskLoad.getTCRVEmitCLowerableSourceRole(),
                         maskCallee));
    mlir::Value zeroLiteral = rewriter.create<emitc::LiteralOp>(
        loc, emitc::OpaqueType::get(rewriter.getContext(), "int"), "0");
    mlir::Value mask =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{maskEmitCType},
                                         maskCallee,
                                         mlir::ValueRange{maskVec, zeroLiteral,
                                                          bodyVL})
            .getResult(0);
    valueMap[maskLoad.getLoaded()] = mask;
    return mlir::success();
  }

  /// masked_load(%abi,%mask,%passthrough,%vl) ->
  ///   ptr = src + i;
  ///   __riscv_vle<sew>_v_<dtype><lmul>_tumu(mask, passthrough, ptr, vl)
  /// The masked unit-stride load reads the source unit-stride but only writes
  /// active (mask-true) lanes; inactive/tail lanes keep the passthrough vector
  /// (the old destination) via the _tumu policy form -- byte-identical to the
  /// legacy masked_load oracle. The mask MUST come from mask_load authority (not
  /// a data compare): reject a compare-sourced mask so a masked body that lacks
  /// explicit mask_load authority falls back (the negative-fixture contract).
  mlir::LogicalResult
  emitMaskedLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                 tcrvrvv::MaskedLoadOp maskedLoad,
                 llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                 mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(maskedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_load result not typed vector");
    if (maskedLoad.getMemoryForm() != "masked-unit-load" ||
        maskedLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_load memory form/policy outside the slice");
    // The predicate authority is structural: the BASE-memory masked family reads
    // its mask from an explicit tcrv_rvv.mask_load buffer; the computed-mask
    // memory family produces it from a tcrv_rvv.compare in the same VL scope.
    // Both lower to the byte-identical `_tumu` masked-load form. Accept either
    // authority, but refuse a mask from any other op so a malformed body (an
    // unmaterialized or out-of-family mask producer) falls back to the legacy
    // validator rather than being mislowered.
    if (!isMaskFromMaskLoadOrCompare(maskedLoad.getMask()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_load mask must come from explicit mask_load "
                      "buffer authority or a compare in the same scope");
    mlir::Value base = valueMap.lookup(maskedLoad.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedLoad.getMask());
    mlir::Value passthrough = valueMap.lookup(maskedLoad.getPassthrough());
    if (!base || !mask || !passthrough)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_load buffer C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "vector type not convertible");
    std::string callee =
        riscvMaskedLoadIntrinsicName(vectorElementWidth(vectorType),
                                     vectorType.getLmul(),
                                     vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedLoad.getTCRVEmitCLowerableSourceOpName(),
                         maskedLoad.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{mask, passthrough, ptr, bodyVL})
            .getResult(0);
    valueMap[maskedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

  /// masked_store(%abi,%mask,%value,%vl) ->
  ///   ptr = dst + i;
  ///   __riscv_vse<sew>_v_<dtype><lmul>_m(mask, ptr, value, vl)
  /// The masked unit-stride store writes only active (mask-true) lanes; inactive
  /// and tail lanes keep their memory contents (no passthrough needed -- the
  /// store simply skips them) -- byte-identical to the legacy masked_store
  /// oracle. The mask MUST come from explicit mask_load authority (NOT a data
  /// compare): the negative fixture rejects a compare-sourced masked store.
  mlir::LogicalResult
  emitMaskedStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
                  tcrvrvv::MaskedStoreOp maskedStore,
                  llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                  mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(maskedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(maskedStore,
                                         "masked_store value not typed vector");
    if (maskedStore.getMemoryForm() != "masked-unit-store" ||
        maskedStore.getInactiveLanePolicy() !=
            "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_store memory form/policy outside the slice");
    // See emitMaskedLoad: accept a mask from explicit mask_load buffer authority
    // (base-memory family) OR a compare in the same scope (computed-mask family);
    // refuse any other producer so a malformed body falls back.
    if (!isMaskFromMaskLoadOrCompare(maskedStore.getMask()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_store mask must come from explicit mask_load "
                       "buffer authority or a compare in the same scope");
    mlir::Value base = valueMap.lookup(maskedStore.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedStore.getMask());
    mlir::Value value = valueMap.lookup(maskedStore.getValue());
    if (!base || !mask || !value)
      return rewriter.notifyMatchFailure(maskedStore,
                                         "masked_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_store buffer C type disagrees with stored element");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(maskedStore,
                                         "vector type not convertible");
    std::string callee =
        riscvMaskedStoreIntrinsicName(vectorElementWidth(vectorType),
                                      vectorType.getLmul(),
                                      vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedStore.getTCRVEmitCLowerableSourceOpName(),
                         maskedStore.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                                    inductionVar);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{mask, ptr, value, bodyVL});
    return mlir::success();
  }

  /// masked_strided_load(%abi,%mask,%passthrough,%stride,%vl) ->
  ///   ptr = (elem_t*)((uint8_t*)src + i * stride);
  ///   __riscv_vlse<sew>_v_<dtype><lmul>_tumu(mask, passthrough, ptr, stride, vl)
  /// The computed-mask masked byte-strided load reads the source at a runtime
  /// BYTE stride but only writes active (mask-true) lanes; inactive/tail lanes
  /// keep the passthrough (old-destination) vector via the _tumu policy form --
  /// byte-identical to the legacy computed-mask masked-strided-load oracle. The
  /// stride MUST be a byte-stride ABI role and the mask MUST be compare/mask_load
  /// authority; otherwise the malformed body falls back.
  mlir::LogicalResult emitMaskedStridedLoad(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedStridedLoadOp maskedLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(maskedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load result not typed vector");
    if (maskedLoad.getMemoryForm() != "masked-strided-load" ||
        maskedLoad.getStrideUnit() != "byte" ||
        maskedLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load form/unit/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedLoad.getMask()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load mask must come from explicit "
                      "mask_load buffer authority or a compare in the same "
                      "scope");
    if (!isByteStride(maskedLoad.getStride()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load requires a byte-stride ABI role");
    mlir::Value base = valueMap.lookup(maskedLoad.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedLoad.getMask());
    mlir::Value passthrough = valueMap.lookup(maskedLoad.getPassthrough());
    mlir::Value stride = valueMap.lookup(maskedLoad.getStride());
    if (!base || !mask || !passthrough || !stride)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_strided_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedLoad,
          "masked_strided_load buffer C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    if (!vecType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "vector type not convertible");
    std::string callee = riscvMaskedStridedLoadIntrinsicName(
        vectorElementWidth(vectorType), vectorType.getLmul(),
        vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedLoad.getTCRVEmitCLowerableSourceOpName(),
                         maskedLoad.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
    if (!ptr)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_strided_load base must be a pointer-typed ABI "
                      "param");
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{mask, passthrough, ptr, stride, bodyVL})
            .getResult(0);
    valueMap[maskedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

  /// masked_strided_store(%abi,%mask,%value,%stride,%vl) ->
  ///   ptr = (elem_t*)((uint8_t*)dst + i * stride);
  ///   __riscv_vsse<sew>_v_<dtype><lmul>_m(mask, ptr, stride, value, vl)
  /// The computed-mask masked byte-strided store writes only active (mask-true)
  /// lanes at a runtime BYTE stride; inactive/tail lanes keep their memory
  /// contents -- byte-identical to the legacy computed-mask masked-strided-store
  /// oracle. Byte-stride role + compare/mask_load mask authority required.
  mlir::LogicalResult emitMaskedStridedStore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedStridedStoreOp maskedStore,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value inductionVar, mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(maskedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store value not typed vector");
    if (maskedStore.getMemoryForm() != "masked-strided-store" ||
        maskedStore.getStrideUnit() != "byte" ||
        maskedStore.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_strided_store form/unit/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedStore.getMask()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store mask must come from explicit "
                       "mask_load buffer authority or a compare in the same "
                       "scope");
    if (!isByteStride(maskedStore.getStride()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store requires a byte-stride ABI role");
    mlir::Value base = valueMap.lookup(maskedStore.getBuffer());
    mlir::Value mask = valueMap.lookup(maskedStore.getMask());
    mlir::Value value = valueMap.lookup(maskedStore.getValue());
    mlir::Value stride = valueMap.lookup(maskedStore.getStride());
    if (!base || !mask || !value || !stride)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(base, vectorType))
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_strided_store buffer C type disagrees with stored element");
    if (!convertVectorTypeToEmitC(vectorType))
      return rewriter.notifyMatchFailure(maskedStore,
                                         "vector type not convertible");
    std::string callee = riscvMaskedStridedStoreIntrinsicName(
        vectorElementWidth(vectorType), vectorType.getLmul(),
        vectorDType(vectorType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedStore.getTCRVEmitCLowerableSourceOpName(),
                         maskedStore.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value ptr =
        emitByteStridedPointer(rewriter, loc, base, inductionVar, stride);
    if (!ptr)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_strided_store base must be a pointer-typed ABI "
                       "param");
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{mask, ptr, stride, value, bodyVL});
    return mlir::success();
  }

  /// masked_indexed_load(%data,%indices,%mask,%passthrough,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   loaded = __riscv_vluxei<eew>_v_<dtype><lmul>_tumu(mask, passthrough,
  ///                                                     data_base, bytes, vl)
  /// The computed-mask masked indexed gather byte-scales the element indices,
  /// then reads scattered elements but only writes active (mask-true) lanes
  /// (passthrough preserved on inactive/tail lanes via _tumu) -- byte-identical
  /// to the legacy computed-mask indexed-gather oracle. Only the element-offset,
  /// EEW=32 slice with compare/mask_load mask authority is accepted.
  mlir::LogicalResult emitMaskedIndexedLoad(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedIndexedLoadOp maskedLoad,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(maskedLoad.getLoaded().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load result not typed vector");
    if (maskedLoad.getMemoryForm() != "masked-indexed-load" ||
        maskedLoad.getOffsetUnit() != "element" ||
        maskedLoad.getInactiveLanePolicy() !=
            "preserve-passthrough-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load form/unit/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedLoad.getMask()))
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load mask must come from explicit "
                      "mask_load buffer authority or a compare in the same "
                      "scope");
    auto indexVecType = llvm::dyn_cast<tcrvrvv::IndexVectorType>(
        maskedLoad.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(
          maskedLoad, "masked_indexed_load indices not index vector");
    mlir::Value dataBase = valueMap.lookup(maskedLoad.getData());
    mlir::Value indices = valueMap.lookup(maskedLoad.getIndices());
    mlir::Value mask = valueMap.lookup(maskedLoad.getMask());
    mlir::Value passthrough = valueMap.lookup(maskedLoad.getPassthrough());
    if (!dataBase || !indices || !mask || !passthrough)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_indexed_load operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dataBase, vectorType))
      return rewriter.notifyMatchFailure(
          maskedLoad,
          "masked_indexed_load data C type disagrees with loaded element");
    mlir::Type vecType = convertVectorTypeToEmitC(vectorType);
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!vecType || !indexEmitCType)
      return rewriter.notifyMatchFailure(maskedLoad,
                                         "masked_indexed_load type not "
                                         "convertible");
    unsigned eew = static_cast<unsigned>(maskedLoad.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          maskedLoad, "only EEW=32 masked indexed loads are in scope");

    // The element->byte index scale was emitted early at index_load time (the
    // computed-mask index-early order), so `indices` is already the byte-offset
    // vector -- consume it directly.
    mlir::Value byteIndices = indices;
    std::string callee = riscvMaskedIndexedLoadIntrinsicName(
        eew, vectorDType(vectorType), vectorType.getLmul());
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedLoad.getTCRVEmitCLowerableSourceOpName(),
                         maskedLoad.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value loaded =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{vecType}, callee,
                mlir::ValueRange{mask, passthrough, dataBase, byteIndices,
                                 bodyVL})
            .getResult(0);
    valueMap[maskedLoad.getLoaded()] = loaded;
    return mlir::success();
  }

  /// masked_indexed_store(%dst,%indices,%mask,%value,%vl) -> TWO calls:
  ///   bytes = __riscv_vmul_vx_u<eew>m<lmul>(indices, elemBytes, vl);
  ///   __riscv_vsoxei<eew>_v_<dtype><lmul>_m(mask, dst_base, bytes, value, vl)
  /// The computed-mask masked indexed scatter byte-scales the element indices,
  /// then writes active (mask-true) lanes to scattered byte offsets; inactive/
  /// tail lanes are skipped -- byte-identical to the legacy computed-mask
  /// indexed-scatter oracle. Only the element-offset, unique-index, EEW=32 slice
  /// with compare/mask_load mask authority is accepted.
  mlir::LogicalResult emitMaskedIndexedStore(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::MaskedIndexedStoreOp maskedStore,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const {
    auto vectorType =
        llvm::dyn_cast<tcrvrvv::VectorType>(maskedStore.getValue().getType());
    if (!vectorType)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store value not typed vector");
    if (maskedStore.getMemoryForm() != "masked-indexed-store" ||
        maskedStore.getOffsetUnit() != "element" ||
        maskedStore.getIndexUniqueness() != "unique" ||
        maskedStore.getInactiveLanePolicy() != "preserve-output-on-false-lanes")
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_indexed_store form/unit/uniqueness/policy outside the slice");
    if (!isMaskFromMaskLoadOrCompare(maskedStore.getMask()))
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store mask must come from explicit "
                       "mask_load buffer authority or a compare in the same "
                       "scope");
    auto indexVecType = llvm::dyn_cast<tcrvrvv::IndexVectorType>(
        maskedStore.getIndices().getType());
    if (!indexVecType)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store indices not index vector");
    mlir::Value dstBase = valueMap.lookup(maskedStore.getDestination());
    mlir::Value indices = valueMap.lookup(maskedStore.getIndices());
    mlir::Value mask = valueMap.lookup(maskedStore.getMask());
    mlir::Value value = valueMap.lookup(maskedStore.getValue());
    if (!dstBase || !indices || !mask || !value)
      return rewriter.notifyMatchFailure(
          maskedStore, "masked_indexed_store operand unmapped");
    if (!bufferPointeeMatchesVectorElement(dstBase, vectorType))
      return rewriter.notifyMatchFailure(
          maskedStore,
          "masked_indexed_store destination C type disagrees with stored "
          "element");
    mlir::Type indexEmitCType = convertIndexVectorTypeToEmitC(indexVecType);
    if (!convertVectorTypeToEmitC(vectorType) || !indexEmitCType)
      return rewriter.notifyMatchFailure(maskedStore,
                                         "masked_indexed_store type not "
                                         "convertible");
    unsigned eew = static_cast<unsigned>(maskedStore.getIndexEew());
    if (eew != 32)
      return rewriter.notifyMatchFailure(
          maskedStore, "only EEW=32 masked indexed stores are in scope");

    // The element->byte index scale was emitted early at index_load time (the
    // computed-mask index-early order), so `indices` is already the byte-offset
    // vector -- consume it directly.
    mlir::Value byteIndices = indices;
    std::string callee = riscvMaskedIndexedStoreIntrinsicName(
        eew, vectorDType(vectorType), vectorType.getLmul());
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(maskedStore.getTCRVEmitCLowerableSourceOpName(),
                         maskedStore.getTCRVEmitCLowerableSourceRole(), callee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, callee,
        mlir::ValueRange{mask, dstBase, byteIndices, value, bodyVL});
    return mlir::success();
  }

  /// True iff `bufferValue` is an emitc pointer whose pointee names the unsigned
  /// 32-bit index element ("uint32_t"). The index buffer C type
  /// (e.g. "const uint32_t *") becomes the index pointer; the index_load reads
  /// it at u32 width.
  static bool indexBufferIsU32(mlir::Value bufferValue) {
    auto pointerType =
        llvm::dyn_cast<emitc::PointerType>(bufferValue.getType());
    if (!pointerType)
      return false;
    auto pointeeOpaque =
        llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
    if (!pointeeOpaque)
      return false;
    return pointeeOpaque.getValue().contains("uint32_t");
  }

  /// True iff `bufferValue` is an emitc pointer whose pointee names the mask
  /// element scalar ("int32_t" / "int64_t"). The mask buffer is loaded at the
  /// data element width before the nonzero test.
  static bool maskBufferPointeeMatches(mlir::Value bufferValue,
                                       llvm::StringRef dtype) {
    auto pointerType =
        llvm::dyn_cast<emitc::PointerType>(bufferValue.getType());
    if (!pointerType)
      return false;
    auto pointeeOpaque =
        llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
    if (!pointeeOpaque)
      return false;
    llvm::StringRef scalar = (dtype == "i32") ? "int32_t" : "int64_t";
    return pointeeOpaque.getValue().contains(scalar);
  }

  /// Convert a `!tcrv_rvv.index_vector<...>` to its EmitC type, accepting only a
  /// genuinely-lowered emitc type (see convertVectorTypeToEmitC for why the
  /// identity fallback must be rejected).
  mlir::Type
  convertIndexVectorTypeToEmitC(tcrvrvv::IndexVectorType type) const {
    mlir::Type converted = getTypeConverter()->convertType(type);
    if (!converted)
      return nullptr;
    if (converted.getDialect().getNamespace() !=
        mlir::emitc::EmitCDialect::getDialectNamespace())
      return nullptr;
    return converted;
  }

  /// Scaled element pointer: off = induction * stride; ptr = base + off.
  /// Mirrors the legacy materializer parseScaledPointerExpression path
  /// (`base + (induction * stride)`): an emitc.mul of the size_t induction and
  /// stride, then an emitc.add onto the pointer base.
  mlir::Value emitScaledPointer(mlir::ConversionPatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value base,
                                mlir::Value inductionVar,
                                mlir::Value stride) const {
    mlir::Value scaledOffset = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, stride);
    return rewriter.create<emitc::AddOp>(loc, base.getType(), base,
                                         scaledOffset);
  }

  /// Byte stride: (ptrdiff_t)stride * (ptrdiff_t)4. Mirrors the legacy
  /// materializer parseSimpleProductExpression path for `stride * 4` with
  /// cType "ptrdiff_t": cast the runtime stride and the element-size literal to
  /// ptrdiff_t, then multiply.
  mlir::Value emitByteStride(mlir::ConversionPatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value stride,
                             tcrvrvv::VectorType vectorType) const {
    mlir::Type ptrdiffType =
        emitc::OpaqueType::get(rewriter.getContext(), "ptrdiff_t");
    unsigned byteWidth = vectorElementWidth(vectorType) / 8;
    mlir::Value strideCast =
        rewriter.create<emitc::CastOp>(loc, ptrdiffType, stride);
    mlir::Value sizeLiteral = rewriter.create<emitc::LiteralOp>(
        loc, stride.getType(), llvm::Twine(byteWidth).str());
    mlir::Value sizeCast =
        rewriter.create<emitc::CastOp>(loc, ptrdiffType, sizeLiteral);
    return rewriter.create<emitc::MulOp>(loc, ptrdiffType, strideCast, sizeCast);
  }

  /// True when the strided op carries a runtime BYTE stride (the base-memory
  /// movement family) rather than an element stride (the elementwise family).
  /// The two are distinguished by the stride's defining runtime ABI value role:
  /// `source-byte-stride` / `destination-byte-stride` are byte strides passed
  /// AS-IS to vlse/vsse; `*-input-stride` / `output-stride` are element strides
  /// the elementwise path scales by the element width. Returning the wrong one
  /// would emit numerically wrong addressing, so the distinction is taken
  /// straight from the typed ABI role fact, not a heuristic.
  static bool isByteStride(mlir::Value strideToken) {
    auto abi = strideToken.getDefiningOp<tcrvrvv::RuntimeABIValueOp>();
    if (!abi)
      return false;
    return abi.getRole().ends_with("byte-stride");
  }

  /// True when the strided-load result is consumed by a tcrv_rvv.move (the
  /// base-memory strided movement shape: strided_load -> move{copy} -> store).
  /// The elementwise strided family feeds its strided_load into a compute op
  /// (binary), never a move, so this cleanly separates the two rungs.
  static bool loadedFeedsMove(tcrvrvv::StridedLoadOp load) {
    return llvm::any_of(load.getLoaded().getUsers(), [](mlir::Operation *user) {
      return llvm::isa<tcrvrvv::MoveOp>(user);
    });
  }

  /// True when the strided-store value is produced by a tcrv_rvv.move (the
  /// base-memory unit-load -> move{copy} -> strided_store shape). The
  /// elementwise strided store's value comes from a compute op, never a move.
  static bool storedValueFromMove(tcrvrvv::StridedStoreOp store) {
    return llvm::isa_and_present<tcrvrvv::MoveOp>(
        store.getValue().getDefiningOp());
  }

  /// True iff a masked-memory predicate `mask` is produced by an in-family mask
  /// authority: either an explicit tcrv_rvv.mask_load buffer (the base-memory
  /// masked family) or a tcrv_rvv.compare in the same VL scope (the
  /// computed-mask memory family). Both lower to the byte-identical masked-load
  /// `_tumu` / masked-store `_m` forms, so the converter accepts either; any
  /// other producer is malformed and must fall back.
  static bool isMaskFromMaskLoadOrCompare(mlir::Value mask) {
    mlir::Operation *def = mask.getDefiningOp();
    return llvm::isa_and_present<tcrvrvv::MaskLoadOp, tcrvrvv::CompareOp>(def);
  }

  /// True iff a segment2 store's field operand binds the EXPECTED field role,
  /// resolved structurally from the field vector's defining tcrv_rvv.load buffer
  /// ABI role. The interleave family carries the two field input loads as the
  /// segment2_store field0/field1 operands; a body that swaps them
  /// (segment2_store %dst, %field1, %field0) binds the wrong field role and is a
  /// malformed body the legacy provider rejects ("segment2_store to consume
  /// matching field0 and field1 load results"). Resolving the binding from the
  /// typed load buffer role (not the ABI c_name) honors I5. When the field
  /// operand is NOT a plain load (e.g. a computed-mask update's add result), the
  /// caller does not apply this guard.
  static bool fieldVectorBindsLoadRole(mlir::Value fieldVector,
                                       llvm::StringRef expectedRole) {
    auto load = fieldVector.getDefiningOp<tcrvrvv::LoadOp>();
    if (!load)
      return false;
    auto abi = load.getBuffer().getDefiningOp<tcrvrvv::RuntimeABIValueOp>();
    if (!abi)
      return false;
    return abi.getRole() == expectedRole;
  }

  /// Byte-stride scaled pointer: ptr = (elem_t*)((uint8_t*)base + i * stride).
  /// The base-memory strided family receives a runtime BYTE stride, so the
  /// element pointer is computed in BYTE space: cast the element base to
  /// `uint8_t*` (preserving const), add `i * stride` bytes, then cast back to
  /// the element pointer type -- byte-identical to the legacy base-memory
  /// strided oracle (`(const uint8_t*)base + i*stride; (const int32_t*)...`).
  mlir::Value emitByteStridedPointer(mlir::ConversionPatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value base,
                                     mlir::Value inductionVar,
                                     mlir::Value stride) const {
    auto pointerType = llvm::dyn_cast<emitc::PointerType>(base.getType());
    if (!pointerType)
      return nullptr;
    auto pointeeOpaque =
        llvm::dyn_cast<emitc::OpaqueType>(pointerType.getPointee());
    if (!pointeeOpaque)
      return nullptr;
    // Preserve a leading `const` qualifier on the byte pointer so a
    // `const int32_t *` base becomes `const uint8_t *` (not `uint8_t *`).
    llvm::StringRef pointee = pointeeOpaque.getValue();
    std::string bytePointee =
        (pointee.contains("const") ? "const uint8_t" : "uint8_t");
    mlir::Type bytePtrType = emitc::PointerType::get(
        emitc::OpaqueType::get(rewriter.getContext(), bytePointee));
    mlir::Value byteBase =
        rewriter.create<emitc::CastOp>(loc, bytePtrType, base);
    mlir::Value byteOffset = rewriter.create<emitc::MulOp>(
        loc, inductionVar.getType(), inductionVar, stride);
    mlir::Value bytePtr =
        rewriter.create<emitc::AddOp>(loc, bytePtrType, byteBase, byteOffset);
    return rewriter.create<emitc::CastOp>(loc, base.getType(), bytePtr);
  }

  static unsigned vectorElementWidth(tcrvrvv::VectorType type) {
    if (auto intType =
            llvm::dyn_cast<mlir::IntegerType>(type.getElementType()))
      return intType.getWidth();
    if (auto floatType =
            llvm::dyn_cast<mlir::FloatType>(type.getElementType()))
      return floatType.getWidth();
    return 0;
  }

  /// Convert a `!tcrv_rvv.vector<...>` to its EmitC type, but ONLY accept a
  /// result the beachhead converter genuinely lowered (an `emitc` type). The
  /// driver registers an identity fallback conversion so unrelated IR is never
  /// illegalized; that identity would otherwise pass an unhandled vector type
  /// (e.g. lmul m2) straight through, letting a half-converted call_opaque keep
  /// a `!tcrv_rvv.vector<...>` result and silently corrupt the module. Rejecting
  /// any non-emitc result here makes a non-beachhead family fail the match and
  /// roll back cleanly, so the export seam falls back to the legacy path.
  mlir::Type convertVectorTypeToEmitC(tcrvrvv::VectorType type) const {
    mlir::Type converted = getTypeConverter()->convertType(type);
    if (!converted)
      return nullptr;
    if (converted.getDialect().getNamespace() !=
        mlir::emitc::EmitCDialect::getDialectNamespace())
      return nullptr;
    return converted;
  }

  static std::optional<llvm::StringRef>
  binaryMnemonic(llvm::StringRef kind) {
    if (kind == "add")
      return llvm::StringRef("vadd");
    if (kind == "sub")
      return llvm::StringRef("vsub");
    if (kind == "mul")
      return llvm::StringRef("vmul");
    return std::nullopt;
  }

  /// The reduction mnemonic for tcrv_rvv.reduce / tcrv_rvv.standalone_reduce,
  /// mirroring the legacy getRVVSelectedBodyReductionIntrinsic /
  /// getRVVSelectedBodyStandaloneReductionIntrinsic kind tables (add -> vredsum,
  /// min -> vredmin, max -> vredmax). Unknown kinds fail the match so the body
  /// falls back to the legacy validators unchanged.
  static std::optional<llvm::StringRef> reductionMnemonic(llvm::StringRef kind) {
    if (kind == "add")
      return llvm::StringRef("vredsum");
    if (kind == "min")
      return llvm::StringRef("vredmin");
    if (kind == "max")
      return llvm::StringRef("vredmax");
    return std::nullopt;
  }

  /// The vector-vector compare predicate mnemonic, mirroring the legacy
  /// getRVVSelectedBody{,Float}CompareIntrinsicForPredicate tables for the
  /// vv-form predicates the compare-select bodies use. Integer:
  /// eq/slt/sle -> vmseq/vmslt/vmsle. Float (f-prefixed):
  /// eq/slt/sle -> vmfeq/vmflt/vmfle.
  static std::optional<llvm::StringRef> compareMnemonic(llvm::StringRef kind,
                                                        bool isFloat) {
    if (isFloat) {
      if (kind == "eq")
        return llvm::StringRef("vmfeq");
      if (kind == "slt")
        return llvm::StringRef("vmflt");
      if (kind == "sle")
        return llvm::StringRef("vmfle");
      return std::nullopt;
    }
    if (kind == "eq")
      return llvm::StringRef("vmseq");
    if (kind == "slt")
      return llvm::StringRef("vmslt");
    if (kind == "sle")
      return llvm::StringRef("vmsle");
    return std::nullopt;
  }

  /// The mask-composition mnemonic for tcrv_rvv.mask_and. The Stage-2 slice
  /// supports only kind = "and" (-> vmand), matching the op's verifier.
  static std::optional<llvm::StringRef> maskAndMnemonic(llvm::StringRef kind) {
    if (kind == "and")
      return llvm::StringRef("vmand");
    return std::nullopt;
  }
};

} // namespace

void populateRVVToEmitCTypeConversions(mlir::TypeConverter &typeConverter) {
  // !tcrv_rvv.vl -> emitc.opaque<"size_t"> (the RVV vector-length token is the
  // C size_t produced by __riscv_vsetvl_*).
  typeConverter.addConversion(
      [](tcrvrvv::VLType type) -> std::optional<mlir::Type> {
        return emitc::OpaqueType::get(type.getContext(), "size_t");
      });

  // !tcrv_rvv.vector<i<sew>, "m<lmul>"> -> emitc.opaque<"vint<sew>m<lmul>_t">,
  // !tcrv_rvv.vector<f<sew>, "m<lmul>"> -> emitc.opaque<"vfloat<sew>m<lmul>_t">.
  // The elementwise family covers the bounded {i32,i64} x {m1,m2} grid the
  // selected-body rungs use (e.g. i32/m1 -> vint32m1_t, i64/m1 -> vint64m1_t,
  // i32/m2 -> vint32m2_t, i64/m2 -> vint64m2_t). The compare-select family adds
  // the f32/m1 float grid (f32/m1 -> vfloat32m1_t) for the f32-clamp /
  // dequant-clamp rungs. Other (dtype, lmul) pairs are left unconverted on
  // purpose so later families extend the grid explicitly.
  typeConverter.addConversion(
      [](tcrvrvv::VectorType type) -> std::optional<mlir::Type> {
        llvm::StringRef lmul = type.getLmul();
        if (type.getElementType().isF32()) {
          // Float grid: only f32/m1 is in scope for the compare-select family.
          if (lmul != "m1")
            return std::nullopt;
          return emitc::OpaqueType::get(type.getContext(), "vfloat32m1_t");
        }
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(8))
          sew = 8;
        else if (type.getElementType().isSignlessInteger(16))
          sew = 16;
        else if (type.getElementType().isSignlessInteger(32))
          sew = 32;
        else if (type.getElementType().isSignlessInteger(64))
          sew = 64;
        else
          return std::nullopt;
        // The widening standalone-reduce source is a FRACTIONAL-LMUL i16
        // vector (i16/mf2 -> vint16mf2_t). The full-LMUL grid stays {m1,m2};
        // i16 also admits its fractional mf2 rung. The signed widening-product
        // / widening-dot contraction family adds the FRACTIONAL-LMUL i8 source
        // rung (i8/mf4 -> vint8mf4_t): the low-precision multiplicand loaded
        // before the vwmul widening to i16/mf2. Other (sew, lmul) pairs are
        // left unconverted on purpose so later families extend explicitly.
        bool inScope = false;
        if (sew == 8)
          inScope = lmul == "mf4";
        else if (sew == 16)
          inScope = lmul == "mf2" || lmul == "m1" || lmul == "m2";
        else
          inScope = lmul == "m1" || lmul == "m2";
        if (!inScope)
          return std::nullopt;
        std::string name = ("vint" + llvm::Twine(sew) + lmul + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !tcrv_rvv.index_vector<i<sew>, "m<lmul>"> ->
  // emitc.opaque<"vuint<sew>m<lmul>_t">. The bounded indexed gather/scatter
  // slice loads an UNSIGNED index/offset vector (the element offsets the
  // ordered indexed access scales to bytes), so the C type is the unsigned
  // vector form (i32/m1 -> vuint32m1_t) -- byte-identical to the legacy indexed
  // oracle's vuint32m1_t index vector. Only the m1 grid the slice uses is in
  // scope; other (sew, lmul) pairs stay unconverted so the converter is scoped.
  typeConverter.addConversion(
      [](tcrvrvv::IndexVectorType type) -> std::optional<mlir::Type> {
        llvm::StringRef lmul = type.getLmul();
        if (lmul != "m1")
          return std::nullopt;
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(32))
          sew = 32;
        else
          return std::nullopt;
        std::string name = ("vuint" + llvm::Twine(sew) + lmul + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !tcrv_rvv.mask<i<sew>, "m<lmul>"> -> emitc.opaque<"vbool<maskbits>_t">. The
  // predicate mask C type is vbool<maskbits>_t where maskbits = SEW/LMUL_ratio
  // (the sew/lmul-derived width matching maskWidthForConfig): i32/m1 -> 32,
  // i64/m1 -> 64, i32/m2 -> 16, i64/m2 -> 32. Pairs maskWidthForConfig does not
  // know are left unconverted so the masked converter stays scoped to the grid.
  typeConverter.addConversion(
      [](tcrvrvv::MaskType type) -> std::optional<mlir::Type> {
        unsigned sew = 0;
        if (type.getElementType().isSignlessInteger(32) ||
            type.getElementType().isF32())
          sew = 32;
        else if (type.getElementType().isSignlessInteger(64))
          sew = 64;
        else
          return std::nullopt;
        unsigned maskBits = maskWidthForConfig(sew, type.getLmul());
        if (maskBits == 0)
          return std::nullopt;
        std::string name = ("vbool" + llvm::Twine(maskBits) + "_t").str();
        return emitc::OpaqueType::get(type.getContext(), name);
      });

  // !tcrv_rvv.runtime_abi_value carries its concrete C type in the defining
  // op's c_type attribute (e.g. "const int32_t *", "int32_t *", "size_t"),
  // which a pure type-keyed conversion cannot recover. The VariantToEmitCFunc
  // pattern derives the function parameter type from that attr directly; the
  // token type itself maps to an opaque placeholder so it never blocks
  // legalization.
  typeConverter.addConversion(
      [](tcrvrvv::RuntimeABIValueType type) -> std::optional<mlir::Type> {
        return emitc::OpaqueType::get(type.getContext(), "void");
      });
}

void populateRVVElementwiseToEmitCPatterns(mlir::TypeConverter &typeConverter,
                                           mlir::RewritePatternSet &patterns) {
  patterns.add<VariantToEmitCFunc>(typeConverter, patterns.getContext());
}

bool convertRVVModuleToEmitC(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();

  // The conversion patterns construct emitc ops/types. When this driver runs
  // outside the pass framework (the live artifact-export materialization seam),
  // the EmitC dialect is only registered, not loaded, in the translate context
  // -- so eagerly load it here. This mirrors the `--tcrv-rvv-lower-to-emitc`
  // pass's `dependentDialects` and makes the driver self-sufficient for both
  // callers. Loading is idempotent.
  context->loadDialect<mlir::emitc::EmitCDialect>();

  mlir::TypeConverter typeConverter;
  // Identity for any type the beachhead conversions do not rewrite, so the
  // pass never illegalizes unrelated IR.
  typeConverter.addConversion([](mlir::Type type) { return type; });
  populateRVVToEmitCTypeConversions(typeConverter);

  mlir::ConversionTarget target(*context);
  // emitc is the lowering destination dialect.
  target.addLegalDialect<mlir::emitc::EmitCDialect>();
  // A tcrv.exec.variant that carries a tcrv_rvv.with_vl selected-lowering
  // boundary is illegal and must be converted into an emitc.func. Variants
  // without a with_vl scope (e.g. scalar fallbacks) stay legal so unconverted
  // families fall through unchanged.
  target.addDynamicallyLegalOp<tcrv::exec::VariantOp>(
      [](tcrv::exec::VariantOp variant) {
        for (mlir::Operation &op : variant.getBody().front())
          if (llvm::isa<tcrv::rvv::WithVLOp>(op))
            return false;
        return true;
      });
  // Everything else (kernels, dispatch, capabilities, other dialects) stays
  // legal; the beachhead conversion rewrites the variant subtree atomically.
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  mlir::RewritePatternSet patterns(context);
  populateRVVElementwiseToEmitCPatterns(typeConverter, patterns);

  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns))))
    return false;

  // The beachhead conversion lowers the selected variant body into a
  // standalone emitc.func + headers. Once a function was produced, drop the
  // now-emptied tcrv.exec scaffolding (kernel/capability/dispatch) for that
  // kernel so the module is a clean, translatable EmitC module matching the
  // legacy materializer's output shape. Kernels that still carry a tcrv_rvv
  // body (unconverted families) are left untouched.
  bool producedFunc = false;
  module.walk([&](mlir::emitc::FuncOp) { producedFunc = true; });
  if (producedFunc) {
    llvm::SmallVector<tcrv::exec::KernelOp, 1> drainedKernels;
    module.walk([&](tcrv::exec::KernelOp kernel) {
      bool hasRVVBody = false;
      kernel.walk([&](tcrv::rvv::WithVLOp) { hasRVVBody = true; });
      if (!hasRVVBody)
        drainedKernels.push_back(kernel);
    });
    for (tcrv::exec::KernelOp kernel : drainedKernels)
      kernel.erase();

    // Module-level tcrv.exec capability/target scaffolding (e.g. a
    // `tcrv.exec.target @rvv_profile` provider declared at module scope and
    // referenced by the kernel's `target = @...`) is description-source IR the
    // legacy materializer discarded when it built its fresh emitc-only module.
    // Once the converted kernel(s) are drained it dangles, and a leftover
    // non-emitc top-level op makes the export handoff reject the module as
    // not-clean and fall back to the (now-retired) string route. Drop any
    // top-level tcrv.exec op that carries NO RVV body (the same drain criterion
    // used for kernels), so the materialized module is the clean emitc-only
    // shape the handoff expects. A top-level tcrv.exec op that still carries a
    // with_vl boundary (an unconverted family) is preserved so the
    // fullyConverted walk still reports a partial conversion.
    llvm::SmallVector<mlir::Operation *, 1> drainedExecOps;
    for (mlir::Operation &op : module.getBody()->getOperations()) {
      if (op.getName().getDialectNamespace() !=
          tcrv::exec::TCRVExecDialect::getDialectNamespace())
        continue;
      if (llvm::isa<tcrv::exec::KernelOp>(op))
        continue; // kernels handled above
      bool hasRVVBody = false;
      op.walk([&](tcrv::rvv::WithVLOp) { hasRVVBody = true; });
      if (!hasRVVBody)
        drainedExecOps.push_back(&op);
    }
    for (mlir::Operation *op : drainedExecOps)
      op->erase();

    // The materialized artifact is a STANDALONE emitc module (the legacy string
    // route built a fresh emitc-only module from scratch and discarded the
    // source IR). The source-front-door families (e.g. the bounded vector
    // source) leave a top-level non-RVV `func.func` source alongside the
    // converted kernel; the conversion correctly never touches it (it is not
    // RVV), but it must NOT ride along in the materialized emitc module — a
    // leftover non-emitc op makes the export handoff / `translateToCpp` reject
    // the module as not-clean and fall back to the (now-retired) legacy string
    // route. Drop the source body ops (anything that is neither an emitc op nor
    // a tcrv op) so the materialized module matches the legacy materializer's
    // clean emitc-only output. tcrv leftovers are deliberately preserved so the
    // fullyConverted walk below can still detect a genuinely partial conversion
    // and report a fall-back.
    llvm::SmallVector<mlir::Operation *, 2> drainedSourceOps;
    for (mlir::Operation &op : module.getBody()->getOperations()) {
      llvm::StringRef dialect = op.getName().getDialectNamespace();
      if (dialect != mlir::emitc::EmitCDialect::getDialectNamespace() &&
          dialect != tcrv::exec::TCRVExecDialect::getDialectNamespace() &&
          dialect != tcrvrvv::TCRVRVVDialect::getDialectNamespace())
        drainedSourceOps.push_back(&op);
    }
    for (mlir::Operation *op : drainedSourceOps)
      op->erase();
  }

  // Strangler-fig gate: report a FULL legalization only when this driver
  // ACTUALLY materialized an emitc.func from an RVV body. A body with no RVV
  // content (a non-RVV family — toy/template/tensorext-lite, or a scalar
  // fallback variant) has no with_vl boundary, so the variant is already legal
  // and `applyPartialConversion` trivially succeeds WITHOUT producing any
  // function. Returning success in that case would tell every caller (the PATH
  // R materialization gate, the export seam, the emission-plan/validation/
  // boundary gates, the --tcrv-rvv-lower-to-emitc pass) that the UNCHANGED body
  // is the "materialized" module — broken/unmaterialized output for every
  // non-RVV family. So a no-emitc.func conversion is NEVER a full conversion:
  // gate on `producedFunc` so those callers all fall through to the legacy path
  // UNCHANGED.
  if (!producedFunc)
    return false;

  // Additionally, report a full legalization only when no RVV dataflow op, no
  // `tcrv_rvv` leftover TYPE (a half-converted op whose operand/result still
  // carries `!tcrv_rvv.*`), and no unrealized_conversion_cast survives. A
  // partial conversion (a func produced but RVV leftovers remain — e.g. a
  // not-yet-covered op in the same body) returns false so the export seam falls
  // back to the legacy string path unchanged.
  auto carriesRVVType = [](mlir::Operation *op) {
    auto isRVV = [](mlir::Type type) {
      return type.getDialect().getNamespace() ==
             tcrvrvv::TCRVRVVDialect::getDialectNamespace();
    };
    return llvm::any_of(op->getOperandTypes(), isRVV) ||
           llvm::any_of(op->getResultTypes(), isRVV);
  };
  bool fullyConverted = true;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() ==
            tcrvrvv::TCRVRVVDialect::getDialectNamespace() ||
        llvm::isa<mlir::UnrealizedConversionCastOp>(op) || carriesRVVType(op)) {
      fullyConverted = false;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return fullyConverted;
}

} // namespace rvv
} // namespace conversion
} // namespace tianchenrv

namespace tianchenrv {
namespace transforms {

namespace {

class RVVLowerToEmitCPass final
    : public impl::RVVLowerToEmitCBase<RVVLowerToEmitCPass> {
public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Run the single shared conversion driver (the same one the live
    // artifact-export materialization seam calls). It runs the
    // TypeConverter/ConversionTarget/patterns + applyPartialConversion and
    // drains the emptied tcrv.exec scaffolding for converted kernels.
    if (conversion::rvv::convertRVVModuleToEmitC(module))
      return;

    // The driver returns false either for a clean structural no-op (an
    // unconverted family whose ops the target keeps legal and which the patterns
    // leave untouched) or for a real conversion failure that left an illegal
    // with_vl-carrying variant behind. Only the latter is a pass failure.
    bool unlegalizedScopeRemains = false;
    module.walk([&](tcrv::rvv::WithVLOp) {
      unlegalizedScopeRemains = true;
      return mlir::WalkResult::interrupt();
    });
    if (unlegalizedScopeRemains)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createRVVLowerToEmitCPass() {
  return std::make_unique<RVVLowerToEmitCPass>();
}

} // namespace transforms
} // namespace tianchenrv
