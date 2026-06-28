#include "TianChenRV/Conversion/RVV/RVVToEmitCSupport.h"

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace tianchenrv {
namespace conversion {
namespace rvv {
namespace detail {

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
  } else if (mnemonic == "vwcvt_x_x_v") {
    // signed integer widening convert: __riscv_vwcvt_x_x_v_<dtype><lmul>, where
    // <dtype><lmul> is the WIDENED RESULT type (i32m1 for i16mf2->i32m1, i64m2
    // for i32m1->i64m2) -- byte-identical to the legacy widening-conversion
    // oracle (__riscv_vwcvt_x_x_v_i32m1 / __riscv_vwcvt_x_x_v_i64m2).
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

/// The deferred wide-LMUL widening-accumulate intrinsic name:
///   __riscv_vwadd_wv_<dtype><lmul>
/// where <dtype><lmul> is the WIDE accumulator vector (i32/m8): the `wv` form
/// adds a NARROWER source vector (the i16/m4 widening product) into the wider
/// i32/m8 accumulator, widening as it accumulates. Byte-identical to the winning
/// algorithm var_v_m2_a1.c (`__riscv_vwadd_wv_i32m8(vacc, p, vl)`).
std::string riscvWideningAccumulateIntrinsicName(unsigned accSEW,
                                                 llvm::StringRef accLmul,
                                                 llvm::StringRef accDtype) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "__riscv_vwadd_wv_" << accDtype << accLmul;
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

/// The indexed memory intrinsic name (mnemonic-agnostic):
///   __riscv_<mnemonic><eew>_v_<dtype><lmul>
/// where eew is the INDEX element width. The mnemonic selects the family:
///   - ordered: `vloxei` (gather) / `vsoxei` (scatter) -- byte-identical to the
///     legacy indexed-load/store oracle (`__riscv_vloxei32_v_i32m1` /
///     `__riscv_vsoxei32_v_i32m1`).
///   - unordered: `vluxei` (gather) / `vsuxei` (scatter) -- same lane->index
///     mapping, ordering unconstrained (used by the iq1_s grid gather,
///     `__riscv_vluxei16_v_i64m2`; order-free because the result feeds an
///     integer-associative reduction, so byte-exact vs the scalar gather).
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

/// True iff the vector element is an UNSIGNED integer (`ui8`/`ui16`/`ui32`).
/// The unsigned low-precision widening-product/reduce family (vwmulu/vwredsumu)
/// keys its intrinsic dtype token and vector C type on this; signed/signless
/// integers and floats are not unsigned.
bool isUnsignedVector(tcrvrvv::VectorType type) {
  auto intType = llvm::dyn_cast<mlir::IntegerType>(type.getElementType());
  return intType && intType.getSignedness() ==
                        mlir::IntegerType::SignednessSemantics::Unsigned;
}

/// The C dtype token ("i32"/"i64"/"f32", or the "u8"/"u16"/"u32" unsigned
/// rungs) for the vector element, used by the load/store/arithmetic intrinsic
/// suffix and the `vint<sew>m<lmul>_t` / `vuint<sew>m<lmul>_t` /
/// `vfloat<sew>m<lmul>_t` opaque type. The unsigned rung mirrors the legacy
/// unsigned widening-product oracle (u8mf4 -> __riscv_vle8_v_u8mf4 /
/// __riscv_vwmulu_vv_u16mf2).
llvm::StringRef vectorDType(tcrvrvv::VectorType type) {
  if (isUnsignedVector(type)) {
    auto intType = llvm::cast<mlir::IntegerType>(type.getElementType());
    switch (intType.getWidth()) {
    case 8:
      return "u8";
    case 16:
      return "u16";
    case 32:
      return "u32";
    case 64:
      return "u64";
    default:
      return "";
    }
  }
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
  if (type.getElementType().isF64())
    return "f64";
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
  if (isUnsignedVector(type)) {
    auto intType = llvm::cast<mlir::IntegerType>(type.getElementType());
    switch (intType.getWidth()) {
    case 8:
      return "uint8_t";
    case 16:
      return "uint16_t";
    case 32:
      return "uint32_t";
    case 64:
      return "uint64_t";
    default:
      return "";
    }
  }
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
  if (type.getElementType().isF64())
    return "double";
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
// Provenance comments, byte-identical to the legacy string-route materializer
// that once rendered them (makeRouteSourceProvenanceComment /
// makeStepProvenanceComment, since retired). Reproduced so the rendered C
// carries the same `// tcrv_emitc.*` lines and stays byte-equivalent to the
// hardware-validated golden.
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

// Function-scoped local variable declaration provenance (the i32 dequant
// accumulator carried across runtime VL chunks), byte-equivalent to the legacy
// `// tcrv_emitc.local_variable=...` line so the e2e harness provenance parser
// reads the same `loop_accumulator_source` fact off the converted C.
std::string localVariableComment(llvm::StringRef varName, llvm::StringRef opName,
                                 llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.local_variable=" << varName << " source_op=" << opName
     << " role=" << role << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

// Function-scoped local variable assignment provenance (seed and per-slice
// accumulator reassignment), byte-equivalent to the legacy
// `// tcrv_emitc.assign target=...` line.
std::string assignComment(llvm::StringRef target, llvm::StringRef opName,
                          llvm::StringRef role) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.assign target=" << target << " source_op=" << opName
     << " role=" << role << " op_interface=" << kOpInterface;
  os.flush();
  return text;
}

//===----------------------------------------------------------------------===//
// Op emission helper: the shared `verbatim step-comment + call_opaque` idiom
// the per-kernel emitters open-code. Reproduces the hand-spliced pair exactly
// (same callee mangle, same step comment, same single-result call), so the
// emitted C is byte-identical to the inline form.
//===----------------------------------------------------------------------===//

mlir::Value emitVCall(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::Type resultType, llvm::StringRef mnemonic,
                      llvm::StringRef suffix, mlir::ValueRange operands,
                      llvm::StringRef opName, llvm::StringRef role) {
  std::string callee = ("__riscv_" + mnemonic + "_" + suffix).str();
  rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
  return rewriter
      .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultType}, callee,
                                   operands)
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Single-source i8 -> i16 -> i32 widening-chain LMUL derivation. The one place
// the q4_K/q6_K integer cores + the FP4 codebook emitters resolve their widened
// MAC LMULs, replacing the divergent inline derivations (one of which dropped
// the "m2" base into the "m1" branch -- the latent widening-chain bug seam).
//===----------------------------------------------------------------------===//

namespace {

// One x2 step up the LMUL ladder: mf4 -> mf2 -> m1 -> m2 -> m4 -> m8. The
// integer-core anchor is verifier-bounded well below m8, so the saturating
// default is never reached for an in-tree base.
llvm::StringRef widenOneStep(llvm::StringRef lmul) {
  if (lmul == "mf4")
    return "mf2";
  if (lmul == "mf2")
    return "m1";
  if (lmul == "m1")
    return "m2";
  if (lmul == "m2")
    return "m4";
  if (lmul == "m4")
    return "m8";
  return "m8";
}

// The i8 lane count the base LMUL spans at the canonical VLEN=128 integer-core
// anchor (the byte-exact strip width the q4_K/q6_K cores assume): mf2=8, m1=16,
// m2=32. Unknown bases fall back to the mf2-shaped 8 (the legacy else-branch).
int64_t i8StripWidthAtVlen128(llvm::StringRef lmul) {
  if (lmul == "mf4")
    return 4;
  if (lmul == "mf2")
    return 8;
  if (lmul == "m1")
    return 16;
  if (lmul == "m2")
    return 32;
  if (lmul == "m4")
    return 64;
  if (lmul == "m8")
    return 128;
  return 8;
}

} // namespace

WideningChain deriveWideningChain(llvm::StringRef base) {
  WideningChain chain;
  llvm::StringRef l16 = widenOneStep(base);
  chain.l8 = base.str();
  chain.l16 = l16.str();
  chain.l32 = widenOneStep(l16).str();
  chain.stripWidth = i8StripWidthAtVlen128(base);
  chain.foldGroups = chain.stripWidth / 8;
  return chain;
}

} // namespace detail
} // namespace rvv
} // namespace conversion
} // namespace tianchenrv
