#include "TianChenRV/Conversion/RVV/RVVToEmitC.h"

#include "TianChenRV/Conversion/EmitC/BackendEmissionRegistry.h"
#include "TianChenRV/Conversion/EmitC/TypedBackendEmissionDriver.h"
#include "TianChenRV/Conversion/RVV/RVVBackendEmissionDriver.h"
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
    llvm::SmallVector<llvm::StringRef, 4> headers = {"stddef.h", "stdint.h",
                                                     "riscv_vector.h"};
    // The forward-pass bodies that call scalar libm need <math.h> so the
    // emitted TU is self-contained: F3 rms_norm (1/sqrtf(mean+eps)) and F6 rope
    // (cosf/sinf per dim-pair angle). ONLY these bodies add the header -- every
    // other (quant-dot / elementwise) kernel keeps the original three-header
    // list byte-identical (additivity).
    if (isGgmlRmsNormF32Body(scope) || isGgmlRopeNormF32Body(scope))
      headers.push_back("math.h");
    for (llvm::StringRef header : headers)
      rewriter.create<emitc::IncludeOp>(loc, header,
                                        /*is_standard_include=*/true);

    rewriter.setInsertionPointToEnd(module.getBody());

    llvm::SmallVector<mlir::Type, 4> paramTypes;
    for (const AbiParam &param : params)
      paramTypes.push_back(param.emitcType);
    // Every forward-pass / quant-dot kernel returns void (results = {}) EXCEPT
    // the F5b soft_max, which faithfully matches ggml's bare
    // `ggml_float ggml_vec_soft_max_f32(...)` signature: it RETURNS the f64 sum.
    // Guard the result type tightly so every other body builds the function type
    // byte-identically (additivity); only the soft_max body carries a `double`
    // result, paired with the single `return (double)...;` emitted in its branch.
    llvm::SmallVector<mlir::Type, 1> resultTypes;
    if (isGgmlVecSoftMaxF32Body(scope))
      resultTypes.push_back(emitc::OpaqueType::get(context, "double"));
    mlir::FunctionType functionType =
        rewriter.getFunctionType(paramTypes, resultTypes);

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

    // The low-precision Gearbox product-reduce-dequantize body owns a dedicated
    // routine: a function-scoped i32 accumulator variable carried across the
    // main (unroll-N) + scalar tail loops, then the dequant epilogue. It is NOT
    // the single-scope `out[0]`-memory-carry standalone reduction path.
    // The DEFERRED-WIDE low-precision contraction (N3 resource-aware
    // max-legal-LMUL schedule, the measured ssh-rvv winner) owns a dedicated
    // routine: a function-scoped i32m8 vector accumulator zero-seeded at its own
    // VLMAX, a strip loop that defers the i16m4 widening product into it via
    // vwadd.wv, ONE trailing vredsum + scalar acc[0] add, then the dequant
    // epilogue. The structural marker is tcrv_rvv.widening_accumulate.
    // The 2nd-family (i16 dot-reduce) DEFERRED-WIDE schedule (N3 resource-aware
    // max-legal-LMUL, the measured ssh-rvv winner dot_wide_deferred) owns a
    // dedicated routine: a function-scoped i32m8 vector accumulator zero-seeded
    // at its own VLMAX, a strip loop that defers the i32m8 widening product into
    // it via NON-widening vadd.vv, ONE trailing vredsum + scalar acc[0] add,
    // then an i32 lane-0 store (NO dequant). The structural marker is
    // tcrv_rvv.deferred_accumulate.
    // The ggml Q4_0 x Q8_0 block dot-product (tcrv_rvv.q4_0_q8_0_block_dot) owns
    // a dedicated routine: an outer emitc.for block loop over nb = n / QK, the
    // per-block address arithmetic + dual fp16 scalar scale reads, an inner
    // strip loop reusing INC-1's offset-binary asymmetric i4xi8 integer core
    // into a per-block i32 scalar, the left-assoc fp32 accumulate, and the *s
    // store. It does NOT use the single-scope runtime-VL strip loop. The
    // structural marker is the tcrv_rvv.q4_0_q8_0_block_dot op identity.
    if (isQ4_0Q8_0BlockDotBody(scope)) {
      if (mlir::failed(emitQ4_0Q8_0BlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q4_0 x Q8_0 GEMM tile (tcrv_rvv.q4_0_q8_0_gemm_tile) is the
    // WEIGHT-DECODE-REUSE sibling of the per-row block dot: the SAME per-(block,
    // column) integer+scale arithmetic, but the offset-binary weight decode is
    // HOISTED above an inner M-column loop so all M activation columns reuse the
    // SAME decoded v0/v1 nibble lanes. It produces M byte-exact outputs (one per
    // column, each == ggml_vec_dot_q4_0_q8_0(weight_row, column_j)). The
    // structural marker is the tcrv_rvv.q4_0_q8_0_gemm_tile op identity.
    if (isQ4_0Q8_0GemmTileBody(scope)) {
      if (mlir::failed(emitQ4_0Q8_0GemmTile(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q4_0 x Q8_0 FULL GEMM (tcrv_rvv.q4_0_q8_0_gemm) wraps the
    // weight-decode-reuse tile in an outer weight-ROW loop (NR rows) x an inner
    // column-strip loop (the nc activation columns in M-wide strips). Every
    // output element s[row][col] is byte-exact vs an independent
    // ggml_vec_dot_q4_0_q8_0(weight_row, column_col). The structural marker is
    // the tcrv_rvv.q4_0_q8_0_gemm op identity.
    if (isQ4_0Q8_0GemmBody(scope)) {
      if (mlir::failed(emitQ4_0Q8_0Gemm(rewriter, loc, scope, avlArg, sizeType,
                                        valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q8_0 x Q8_0 block dot-product (tcrv_rvv.q8_0_q8_0_block_dot) is
    // the Family-A sibling of the Q4_0 route: the SAME block-loop / unroll /
    // tail / scale / store structure, with a plain i8 x i8 widening-product
    // integer core (no nibble decode) and ggml's q8_0 fold order. The structural
    // marker is the tcrv_rvv.q8_0_q8_0_block_dot op identity.
    if (isQ8_0Q8_0BlockDotBody(scope)) {
      if (mlir::failed(emitQ8_0Q8_0BlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q4_1 x Q8_1 block dot-product (tcrv_rvv.q4_1_q8_1_block_dot) is
    // the Family-B sibling (scale+MIN, asymmetric): the SAME block-loop / unroll
    // / tail / scale / store structure, with an UNSIGNED-nibble i4 x i8 widening-
    // product integer core (no offset-binary `-8` decode) and ggml's q4_1 fold
    // (the (d_x*d_y)*sumi scale term plus the m_x*s_y MIN term). The structural
    // marker is the tcrv_rvv.q4_1_q8_1_block_dot op identity.
    if (isQ4_1Q8_1BlockDotBody(scope)) {
      if (mlir::failed(emitQ4_1Q8_1BlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q5_0 x Q8_0 block dot-product (tcrv_rvv.q5_0_q8_0_block_dot) is
    // the Family-A 5-bit sibling: the SAME block-loop / unroll / tail / scale /
    // store structure as the Q4_0 route, with a 5-BIT-weight integer core (the
    // unsigned nibble unpack + per-element 5th-bit injection from the qh field +
    // offset-binary `-16` bias) and ggml's q5_0/q8_0 scales-first fold order. The
    // structural marker is the tcrv_rvv.q5_0_q8_0_block_dot op identity.
    if (isQ5_0Q8_0BlockDotBody(scope)) {
      if (mlir::failed(emitQ5_0Q8_0BlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q5_1 x Q8_1 block dot-product (tcrv_rvv.q5_1_q8_1_block_dot) is
    // the Family-B 5-bit sibling that COMBINES q5_0's 5-bit weight
    // reconstruction (the unsigned nibble unpack + per-element 5th-bit injection
    // from the qh field, but with NO offset-binary `-16` bias) with q4_1's
    // scale+MIN fold (the (d_x*d_y)*sumi scale term plus the m_x*s_y MIN term).
    // The structural marker is the tcrv_rvv.q5_1_q8_1_block_dot op identity.
    if (isQ5_1Q8_1BlockDotBody(scope)) {
      if (mlir::failed(emitQ5_1Q8_1BlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q6_K x Q8_K super-block INTEGER partial
    // (tcrv_rvv.q6_k_q8_k_aux32_partial) is the K-quant K1 increment: the
    // super-block 6-bit ql+qh unpack into an element-ordered int8_t aux8[256]
    // scratch (the exact `_generic` permutation), a nested sub-block loop
    // applying the per-sub-block int8 scale in the i32 domain into an 8-lane
    // aux32 accumulator vector, and a vse32 store of that aux32 state through the
    // output pointer (NO fp32 fold -- that is K2). The structural marker is the
    // tcrv_rvv.q6_k_q8_k_aux32_partial op identity.
    if (isQ6_KQ8_KAux32PartialBody(scope)) {
      if (mlir::failed(emitQ6_KQ8_KAux32Partial(rewriter, loc, scope, avlArg,
                                                sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q6_K x Q8_K FULL super-block dot-product
    // (tcrv_rvv.q6_k_q8_k_block_dot) is the K-quant K2 increment: the SAME K1
    // super-block integer core (6-bit ql+qh unpack into aux8[256], nested
    // sub-block int8-scaled i32 accumulation into the 8-lane aux32) PLUS the
    // deferred two-level fp32 fold (the carried 8-lane fp32 `sums` accumulator,
    // the per-super-block fp16-d x fp32-d multiply, the lane-wise sums[l] +=
    // d*aux32[l] via SEPARATE vfmul/vfadd, and the SEQUENTIAL horizontal sum
    // into the fp32 *s). The structural marker is the
    // tcrv_rvv.q6_k_q8_k_block_dot op identity.
    if (isQ6_KQ8_KBlockDotBody(scope)) {
      if (mlir::failed(emitQ6_KQ8_KBlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q4_K x Q8_K super-block INTEGER partial
    // (tcrv_rvv.q4_k_q8_k_aux_partial) is the q4_K K4a increment: the super-block
    // 4-bit nibble unpack into an element-ordered int8_t aux8[256] scratch (NO
    // bias), the STRUCTURED 6-bit scale/min bit-dance (the get_scale_min_k4 /
    // utmp/kmask cross-byte decode as scalar emitc bitwise ops) producing the 8
    // scales + 8 mins, a nested sub-block loop applying the per-sub-block UINT6
    // scale in the i32 domain into an 8-lane aux32 accumulator vector, and a
    // vse32 store of aux32 + a vse8 store of the 16 decoded scale/min bytes
    // through the two output pointers (NO fp32 d/dmin fold + NO min term -- that
    // is K4b). The structural marker is the tcrv_rvv.q4_k_q8_k_aux_partial op
    // identity.
    if (isQ4_KQ8_KAux32PartialBody(scope)) {
      if (mlir::failed(emitQ4_KQ8_KAux32Partial(rewriter, loc, scope, avlArg,
                                                sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The ggml Q4_K x Q8_K FULL super-block dot-product
    // (tcrv_rvv.q4_k_q8_k_block_dot) is the q4_K K4b increment: the SAME K4a
    // super-block integer core (4-bit unpack into aux8[256], the 6-bit scale/min
    // bit-dance, nested sub-block uint6-scaled i32 accumulation into the 8-lane
    // aux32) PLUS the deferred two-level fp32 fold (the carried 8-lane fp32
    // `sums` accumulator, the per-super-block fp16-d x fp32-d multiply, the
    // lane-wise sums[l] += d*aux32[l] via SEPARATE vfmul/vfadd) AND the q4_K min
    // term (the integer sumi = Σ bsums[j]*mins[j/2] carried in a scalar sumf
    // accumulator via sumf -= dmin*sumi IN-LOOP), then the SEQUENTIAL horizontal
    // sum into the fp32 *s. The structural marker is the
    // tcrv_rvv.q4_k_q8_k_block_dot op identity.
    if (isQ4_KQ8_KBlockDotBody(scope)) {
      if (mlir::failed(emitQ4_KQ8_KBlockDot(rewriter, loc, scope, avlArg,
                                            sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F1 op (tcrv_rvv.ggml_vec_scale_f32) is the FIRST non-dot
    // f32 elementwise family member: the in-place per-lane multiply y[i] *= v
    // over a flat unit-stride f32 buffer. It owns a dedicated routine -- ONE f32
    // strip loop (vsetvl_e32m<L>(n-i) / vle32 / vfmul_vf / vse32) -- a DIFFERENT
    // shape from the block-quantized integer dot ops (no AoS block loop, no
    // packed-int decode, no integer widening, no per-block fp16 scale). The
    // structural marker is the tcrv_rvv.ggml_vec_scale_f32 op identity.
    if (isGgmlVecScaleF32Body(scope)) {
      if (mlir::failed(emitGgmlVecScaleF32(rewriter, loc, scope, avlArg,
                                           sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F3 op (tcrv_rvv.ggml_rms_norm_f32) is the FIRST non-dot
    // f32 REDUCTION op: the row Sx^2 scalar-double fold -> scalar 1/sqrtf -> the
    // vectorized normalize y[i] = x[i] * scale. It owns a dedicated routine (a
    // structured scalar-double accumulator loop + a scalar rsqrt + a single f32
    // strip loop) -- a NEW shape vs the elementwise scale (F1) because the
    // reduction must replicate ggml's scalar-double ascending fold for
    // byte-exactness (a vectorized vfredusum would fold in f32 and a tree order).
    // The structural marker is the tcrv_rvv.ggml_rms_norm_f32 op identity.
    if (isGgmlRmsNormF32Body(scope)) {
      if (mlir::failed(emitGgmlRmsNormF32(rewriter, loc, scope, avlArg,
                                          sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F5 op (tcrv_rvv.ggml_vec_silu_f32) is the FIRST non-dot
    // VECTORIZED TRANSCENDENTAL: the per-strip silu y[i] = x[i]*sigmoid(x[i])
    // where sigmoid runs ggml's EXACT vectorized exp polynomial
    // (ggml_v_expf_m2). It owns a dedicated routine (an m2 f32 strip loop whose
    // body is the node-for-node ggml_v_expf_m2 intrinsic chain + the silu
    // neg/exp/+1/div) -- a NEW shape vs F1/F3 because the exp polynomial must be
    // replicated bit-for-bit (the slow-path vmerge value graph emitted
    // unconditionally; ggml's vcpop short-circuit is a pure perf branch whose
    // taken/not-taken paths are bitwise-equal). Marker: the op identity.
    if (isGgmlVecSiluF32Body(scope)) {
      if (mlir::failed(emitGgmlVecSiluF32(rewriter, loc, scope, avlArg,
                                          sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F5b op (tcrv_rvv.ggml_vec_soft_max_f32) COMBINES F5's
    // vectorized transcendental (the SHARED exact ggml_v_expf_m2 polynomial) with
    // a NEW reduction shape: y[i] = e^{x[i]-max} written per strip, and the f64
    // sum accumulated via the WIDENING reduce vfwredusum_vs_f32m2_f64m1 into a
    // loop-carried f64m1 accumulator (NOT F3's scalar-ascending fold) -- ggml's
    // EXACT method (vec.cpp:584-592). It owns a dedicated routine and is the only
    // forward-pass op whose function RETURNS a scalar (the f64 sum), so its
    // branch emits its own `return (double)...` from the emitter's result value.
    // Marker: the op identity.
    if (isGgmlVecSoftMaxF32Body(scope)) {
      mlir::FailureOr<mlir::Value> sum =
          emitGgmlVecSoftMaxF32(rewriter, loc, scope, avlArg, sizeType, valueMap);
      if (mlir::failed(sum))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, *sum);
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F4 op (tcrv_rvv.quantize_row_q8_0) is the f32 -> QUANT
    // BRIDGE: the per-32-block max-abs REDUCTION (vfredmax of vfabs) + the scalar
    // scale (d = amax/127, id = d ? 1/d : 0) + the f32->i16->i8 NARROWING CONVERT
    // (vfncvt round-to-nearest-even + vncvt truncate), written into the AoS
    // block_q8_0 buffer (the fp16 d at byte 0, the 32 int8 qs at byte 2). It owns
    // a dedicated routine -- an outer block loop, a single e32m8 strip per block
    // (vl=32, relying on Zvl128b => VLEN>=128), the structured d?1/d:0
    // emitc.cmp/emitc.if conditional, the native (_Float16)d AoS store -- a NEW
    // shape vs F1/F3/F5/F5b (a reduction + a narrowing convert + a structured
    // scalar branch) and vs the integer block-dot ops. Marker: the op identity.
    if (isGgmlQuantizeRowQ80Body(scope)) {
      if (mlir::failed(emitGgmlQuantizeRowQ80(rewriter, loc, scope, avlArg,
                                              sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The forward-pass F6 op (tcrv_rvv.ggml_rope_norm_f32) is the COMPOSITION
    // rung: ggml's NORMAL rope (rotary position embedding) for one head row. It
    // owns a dedicated routine -- a SINGLE scalar per-pair loop that carries the
    // iterative f32 angle recurrence (theta *= theta_scale), computes cosf/sinf
    // per pair via scalar libm call_opaque (the sanctioned opaque seam, a
    // DIFFERENT byte-exactness axis -- libm-linked, not a vectorized polynomial),
    // and applies the f32 rotation y[2p]=x0*cos-x1*sin / y[2p+1]=x0*sin+x1*cos
    // with each output's a*b-c*d GROUPED into ONE emitc.expression (token-
    // identical to ggml's single C expression, so it contracts identically under
    // every -ffp-contract mode -> byte-exact regardless of the build flag).
    // Marker: the op identity.
    if (isGgmlRopeNormF32Body(scope)) {
      if (mlir::failed(emitGgmlRopeNormF32(rewriter, loc, scope, avlArg,
                                           sizeType, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    if (isDeferredWideDotReduceBody(scope)) {
      if (mlir::failed(emitDeferredWideDotReduceBody(
              rewriter, loc, variant, scope, preLoopSetVL, avlArg, vlmax,
              sizeType, setvlCallee, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    if (isDeferredWideDequantBody(scope)) {
      if (mlir::failed(emitDeferredWideDequantBody(
              rewriter, loc, variant, scope, preLoopSetVL, avlArg, vlmax,
              sizeType, setvlCallee, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    if (isLowPrecisionDequantBody(scope)) {
      if (mlir::failed(emitLowPrecisionDequantBody(
              rewriter, loc, variant, scope, preLoopSetVL, avlArg, vlmax,
              sizeType, setvlCallee, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // The STANDALONE i32->f32 runtime-scale dequant body (load -> dequantize ->
    // store, no product/reduce/accumulator) owns a dedicated Gearbox-unrolled
    // two-slice setvl loop emitter -- the same VL-loop machinery, expanded
    // `tcrv_rvv.gearbox.unroll` times. It is NOT the product-reduce dequant
    // path (no accumulator) nor the single-slice emitScopeForLoop (which the
    // emitDequantize guard would refuse). Detect and emit it here.
    if (isStandaloneDequantBody(scope)) {
      if (mlir::failed(emitStandaloneDequantBody(
              rewriter, loc, scope, preLoopSetVL, avlArg, vlmax, sizeType,
              setvlCallee, valueMap)))
        return mlir::failure();
      rewriter.create<emitc::ReturnOp>(loc, mlir::Value());
      rewriter.eraseOp(variant);
      return mlir::success();
    }

    // Standalone reduction pre-loop seed: out[0] = acc[0]. Runs BEFORE the loop
    // (between the pre-loop full-chunk setvl and the for-loop), seeding the
    // scalar accumulator carried through the output cell across runtime chunks.
    bool standaloneReduction = isStandaloneReductionBody(scope);
    if (standaloneReduction) {
      if (mlir::failed(
              emitStandaloneReductionPreLoopSeed(rewriter, loc, scope, valueMap)))
        return mlir::failure();
    }

    // Emit the scope's runtime VL for-loop (setvl-tail chunk loop) + body walk.
    // Extracted into a reusable per-scope helper so a multi-scope body (the
    // Gearbox dequant producer/tail/consumer scopes) can drive it once per
    // scope; the single-scope families call it exactly once, unchanged.
    if (mlir::failed(emitScopeForLoop(rewriter, loc, variant, scope,
                                      preLoopSetVL, avlArg, vlmax, sizeType,
                                      setvlCallee, valueMap, segmentFieldMap,
                                      standaloneReduction)))
      return mlir::failure();

    rewriter.create<emitc::ReturnOp>(loc, mlir::Value());

    rewriter.eraseOp(variant);
    return mlir::success();
  }

private:
  /// Emit one selected-body VL scope as a runtime setvl-tail chunk loop:
  ///   for (size_t i = 0; i < n; i += vlmax) { vl = setvl(n - i); <body> }
  /// walking the scope's typed dataflow ops in (reorder-adjusted) IR order.
  /// This is the per-scope emission shared by single-scope families (called
  /// once) and the multi-scope Gearbox dequant body (called per scope). It does
  /// NOT emit the function epilogue/return; the caller owns that.
  mlir::LogicalResult emitScopeForLoop(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      llvm::DenseMap<mlir::Value, std::pair<mlir::Value, unsigned>>
          &segmentFieldMap,
      bool standaloneReduction) const {
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
        } else if (auto widenConvert =
                       llvm::dyn_cast<tcrvrvv::WideningConvertOp>(op)) {
          if (mlir::failed(emitWideningConvert(rewriter, loc, widenConvert,
                                               valueMap, bodyVL)))
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
        } else if (auto offsetBinaryProduct = llvm::dyn_cast<
                       tcrvrvv::PackedI4OffsetBinaryXI8ProductOp>(op)) {
          if (mlir::failed(emitPackedI4OffsetBinaryXI8Product(
                  rewriter, loc, offsetBinaryProduct, valueMap, bodyVL)))
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

    return mlir::success();
  }

  /// True iff `scope` is a low-precision Gearbox product-reduce-dequantize body:
  /// it dequantizes an i32 accumulator built from a signed widening or packed-i4
  /// nibble-unpack product reduced over i8mf4 sources, then stores the f32
  /// result. Structurally: the scope carries at least one tcrv_rvv.dequantize
  /// whose source is a tcrv_rvv.standalone_reduce of a tcrv_rvv.widening_product
  /// or tcrv_rvv.packed_i4_nibble_unpack_product, plus a store. This is the
  /// dequant family the dedicated multi-loop + function-scoped accumulator
  /// routine owns; every other body falls to the single-scope emitScopeForLoop.
  static bool isLowPrecisionDequantBody(tcrvrvv::WithVLOp scope) {
    bool sawDequantChain = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      auto dequant = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op);
      if (!dequant)
        continue;
      mlir::Operation *reduceDef = dequant.getSource().getDefiningOp();
      auto reduce = llvm::dyn_cast_or_null<tcrvrvv::StandaloneReduceOp>(
          reduceDef);
      if (!reduce)
        return false;
      mlir::Operation *productDef = reduce.getInput().getDefiningOp();
      if (!llvm::isa_and_nonnull<tcrvrvv::WideningProductOp,
                                 tcrvrvv::PackedI4NibbleUnpackProductOp>(
              productDef))
        return false;
      sawDequantChain = true;
    }
    return sawDequantChain;
  }

  /// True iff `scope` is the DEFERRED-WIDE low-precision contraction body (the
  /// N3 resource-aware max-legal-LMUL schedule, the measured ssh-rvv winner
  /// var_v_m2_a1.c): the dequant sources a trailing tcrv_rvv.standalone_reduce
  /// whose input is a tcrv_rvv.widening_accumulate (the i32m8 deferred vector
  /// accumulate). The widening_accumulate op is the STRUCTURAL marker (I5): the
  /// conversion emits the deferred-wide algorithm because the op is in the body,
  /// not because of any metadata. A body with the narrow per-iteration
  /// vwredsum-into-scalar reduce (isLowPrecisionDequantBody) has NO
  /// widening_accumulate, so the two recognizers are disjoint.
  static bool isDeferredWideDequantBody(tcrvrvv::WithVLOp scope) {
    bool sawDeferredChain = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      auto dequant = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op);
      if (!dequant)
        continue;
      auto reduce = dequant.getSource().getDefiningOp<tcrvrvv::StandaloneReduceOp>();
      if (!reduce)
        return false;
      if (!reduce.getInput().getDefiningOp<tcrvrvv::WideningAccumulateOp>())
        return false;
      sawDeferredChain = true;
    }
    return sawDeferredChain;
  }

  /// True iff `scope` is the 2nd-family (i16 dot-reduce) DEFERRED-WIDE body: a
  /// tcrv_rvv.store whose stored value is a tcrv_rvv.standalone_reduce (i32m8 ->
  /// i32m1) whose input is a tcrv_rvv.deferred_accumulate (the i32m8 NON-widening
  /// vadd.vv deferred accumulate). The deferred_accumulate op is the STRUCTURAL
  /// marker (I5): emission follows op identity. Disjoint from the byte path
  /// (which has a WideningAccumulateOp + a DequantizeOp) and from the narrow
  /// per-iter dot-reduce (which has a WideningDotReduceOp, no deferred_accumulate).
  /// True iff `scope` is the ggml Q4_0 x Q8_0 block dot-product body: a with_vl
  /// scope whose ONLY compute op is a single tcrv_rvv.q4_0_q8_0_block_dot (the
  /// whole AoS block loop + per-block dual fp16 scale + fp32 accumulation + *s
  /// store is the op's emission, not a body of sub-ops). The op identity is the
  /// dispatch key; the emitter owns the structured expansion.
  static bool isQ4_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ40Q80Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The GEMM-tile (weight-decode reuse) recognizer: a with_vl scope whose ONLY
  /// compute op is a single tcrv_rvv.q4_0_q8_0_gemm_tile. The op identity is the
  /// dispatch key; the emitter owns the structured weight-reuse expansion.
  static bool isQ4_0Q8_0GemmTileBody(tcrvrvv::WithVLOp scope) {
    bool sawTile = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlGemmTileQ40Q80Op>(op)) {
        if (sawTile)
          return false;
        sawTile = true;
      } else {
        return false;
      }
    }
    return sawTile;
  }

  /// The full-GEMM (NR rows x nc cols) recognizer: a with_vl scope whose ONLY
  /// compute op is a single tcrv_rvv.q4_0_q8_0_gemm. The op identity is the
  /// dispatch key; the emitter owns the structured row x column-strip expansion.
  static bool isQ4_0Q8_0GemmBody(tcrvrvv::WithVLOp scope) {
    bool sawGemm = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlGemmQ40Q80Op>(op)) {
        if (sawGemm)
          return false;
        sawGemm = true;
      } else {
        return false;
      }
    }
    return sawGemm;
  }

  /// The Family-A sibling recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q8_0_q8_0_block_dot.
  static bool isQ8_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ80Q80Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The Family-B sibling recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q4_1_q8_1_block_dot.
  static bool isQ4_1Q8_1BlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ41Q81Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The Family-A 5-bit sibling recognizer: a with_vl scope whose ONLY compute
  /// op is a single tcrv_rvv.q5_0_q8_0_block_dot.
  static bool isQ5_0Q8_0BlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ50Q80Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The Family-B 5-bit sibling recognizer: a with_vl scope whose ONLY compute
  /// op is a single tcrv_rvv.q5_1_q8_1_block_dot.
  static bool isQ5_1Q8_1BlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ51Q81Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The K-quant K1 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q6_k_q8_k_aux32_partial (the Q6_K x Q8_K super-block
  /// integer aux32 partial).
  static bool isQ6_KQ8_KAux32PartialBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ6KQ8KAux32Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The K-quant K2 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.q6_k_q8_k_block_dot (the Q6_K x Q8_K super-block FULL
  /// block dot-product producing the fp32 *s).
  static bool isQ6_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ6KQ8KOp>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The q4_K K4a recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q4_k_q8_k_aux_partial (the Q4_K x Q8_K super-block integer aux32 +
  /// decoded scale/min partial -- the INTEGER CORE before the fp32 d/dmin fold).
  static bool isQ4_KQ8_KAux32PartialBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ4KQ8KAux32Op>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The q4_K K4b recognizer: a with_vl scope whose ONLY compute op is a single
  /// tcrv_rvv.q4_k_q8_k_block_dot (the Q4_K x Q8_K super-block FULL block
  /// dot-product producing the fp32 *s -- K4a's integer core + the deferred fp32
  /// fold + the q4_K min term).
  static bool isQ4_KQ8_KBlockDotBody(tcrvrvv::WithVLOp scope) {
    bool sawBlockDot = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlBlockDotQ4KQ8KOp>(op)) {
        if (sawBlockDot)
          return false;
        sawBlockDot = true;
      } else {
        return false;
      }
    }
    return sawBlockDot;
  }

  /// The forward-pass F1 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.ggml_vec_scale_f32 (the f32 in-place elementwise scale
  /// y[i] *= v). The op identity is the dispatch key; the emitter owns the
  /// structured f32 strip-loop expansion (vle32 / vfmul_vf / vse32).
  static bool isGgmlVecScaleF32Body(tcrvrvv::WithVLOp scope) {
    bool sawScale = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlVecScaleF32Op>(op)) {
        if (sawScale)
          return false;
        sawScale = true;
      } else {
        return false;
      }
    }
    return sawScale;
  }

  /// The forward-pass F3 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.ggml_rms_norm_f32 (the f32 row rms_norm: Sx^2 scalar-double
  /// reduce -> scalar 1/sqrtf(mean+eps) -> vectorized y[i] = x[i]*scale). The op
  /// identity is the dispatch key; the emitter owns the structured scalar-double
  /// reduction + scalar rsqrt + f32 normalize strip-loop expansion.
  static bool isGgmlRmsNormF32Body(tcrvrvv::WithVLOp scope) {
    bool sawRmsNorm = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlRmsNormF32Op>(op)) {
        if (sawRmsNorm)
          return false;
        sawRmsNorm = true;
      } else {
        return false;
      }
    }
    return sawRmsNorm;
  }

  /// True iff the with_vl body is EXACTLY a single tcrv_rvv.ggml_vec_silu_f32
  /// (the f32 silu: y[i] = x[i]*sigmoid(x[i]), via ggml's EXACT vectorized exp
  /// polynomial). The op identity is the dispatch key; the emitter owns the
  /// structured strip-loop + the node-for-node ggml_v_expf_m2 intrinsic chain.
  static bool isGgmlVecSiluF32Body(tcrvrvv::WithVLOp scope) {
    bool sawSilu = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlVecSiluF32Op>(op)) {
        if (sawSilu)
          return false;
        sawSilu = true;
      } else {
        return false;
      }
    }
    return sawSilu;
  }

  /// True iff the with_vl body is EXACTLY a single tcrv_rvv.ggml_vec_soft_max_f32
  /// (the F5b f32 soft_max: y[i] = e^{x[i]-max}, returning the f64 sum via the
  /// widening reduce). The op identity is the dispatch key; the emitter owns the
  /// structured m2 strip loop with the shared exp polynomial + the loop-carried
  /// f64m1 vfwredusum accumulator + the f64 scalar return.
  static bool isGgmlVecSoftMaxF32Body(tcrvrvv::WithVLOp scope) {
    bool sawSoftMax = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlVecSoftMaxF32Op>(op)) {
        if (sawSoftMax)
          return false;
        sawSoftMax = true;
      } else {
        return false;
      }
    }
    return sawSoftMax;
  }

  /// True iff the with_vl body is EXACTLY a single tcrv_rvv.quantize_row_q8_0
  /// (the F4 f32->block_q8_0 activation quantizer: per-32-block amax reduction +
  /// scale + f32->i16->i8 narrowing convert + the fp16 d / int8 qs AoS store).
  /// The op identity is the dispatch key; the emitter owns the structured block
  /// loop with the vfredmax reduction, the d?1/d:0 conditional, and the
  /// vfncvt/vncvt narrowing chain.
  static bool isGgmlQuantizeRowQ80Body(tcrvrvv::WithVLOp scope) {
    bool sawQuantize = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlQuantizeRowQ80Op>(op)) {
        if (sawQuantize)
          return false;
        sawQuantize = true;
      } else {
        return false;
      }
    }
    return sawQuantize;
  }

  /// The forward-pass F6 recognizer: a with_vl scope whose ONLY compute op is a
  /// single tcrv_rvv.ggml_rope_norm_f32 (the f32 NORMAL rope: the iterative f32
  /// angle recurrence + scalar libm cosf/sinf cache + the per-pair f32 rotation).
  /// The op identity is the dispatch key; the emitter owns the structured scalar
  /// per-pair loop expansion.
  static bool isGgmlRopeNormF32Body(tcrvrvv::WithVLOp scope) {
    bool sawRope = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (llvm::isa<tcrvrvv::GgmlRopeNormF32Op>(op)) {
        if (sawRope)
          return false;
        sawRope = true;
      } else {
        return false;
      }
    }
    return sawRope;
  }

  static bool isDeferredWideDotReduceBody(tcrvrvv::WithVLOp scope) {
    bool sawDeferredChain = false;
    for (mlir::Operation &op : scope.getBody().front()) {
      auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(op);
      if (!store)
        continue;
      auto reduce =
          store.getValue().getDefiningOp<tcrvrvv::StandaloneReduceOp>();
      if (!reduce)
        return false;
      if (!reduce.getInput().getDefiningOp<tcrvrvv::DeferredAccumulateOp>())
        return false;
      sawDeferredChain = true;
    }
    return sawDeferredChain;
  }

  /// Emit one product/reduce slice into the function-scoped accumulator variable
  /// `accVar`: load lhs/rhs at the given pointer offset, widening (or packed-i4
  /// nibble-unpack) product, then reduce into the accumulator (reading the
  /// variable as the running seed and reassigning it). `lhsBuffer`/`rhsBuffer`
  /// are the ABI pointer params; `loadVL` is the slice's runtime VL; `offset`
  /// (may be null) is added to the induction var for the second unroll slice.
  mlir::LogicalResult emitDequantProductReduceSlice(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::LoadOp lhsLoad, tcrvrvv::LoadOp rhsLoad,
      mlir::Operation *productOp, tcrvrvv::StandaloneReduceOp reduce,
      mlir::Value lhsBuffer, mlir::Value rhsBuffer, mlir::Value sliceOffset,
      mlir::Value accVar, tcrvrvv::VectorType accVecType, mlir::Value loadVL,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Load both i8mf4 sources at (base + offset).
    if (mlir::failed(emitLoad(rewriter, loc, lhsLoad, valueMap, sliceOffset,
                              loadVL)))
      return mlir::failure();
    if (mlir::failed(emitLoad(rewriter, loc, rhsLoad, valueMap, sliceOffset,
                              loadVL)))
      return mlir::failure();
    // The product (plain widening, or the packed-i4 nibble-unpack chain).
    if (auto product = llvm::dyn_cast<tcrvrvv::WideningProductOp>(productOp)) {
      if (mlir::failed(
              emitWideningProduct(rewriter, loc, product, valueMap, loadVL)))
        return mlir::failure();
    } else if (auto packed =
                   llvm::dyn_cast<tcrvrvv::PackedI4NibbleUnpackProductOp>(
                       productOp)) {
      if (mlir::failed(emitPackedI4NibbleUnpackProduct(rewriter, loc, packed,
                                                       valueMap, loadVL)))
        return mlir::failure();
    } else {
      return rewriter.notifyMatchFailure(productOp,
                                         "unsupported dequant product op");
    }
    // Reduce into the running accumulator variable: seed = dot_acc_vec;
    // dot_acc_vec = vwredsum(product, seed, vl).
    mlir::Value product = valueMap.lookup(reduce.getInput());
    if (!product)
      return rewriter.notifyMatchFailure(reduce,
                                         "dequant reduce input unmapped");
    auto srcVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getInput().getType());
    if (!srcVecType)
      return rewriter.notifyMatchFailure(reduce,
                                         "dequant reduce input not a vector");
    std::optional<llvm::StringRef> mnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!mnemonic)
      return rewriter.notifyMatchFailure(reduce,
                                         "unsupported dequant reduce kind");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC || !convertVectorTypeToEmitC(srcVecType))
      return rewriter.notifyMatchFailure(reduce,
                                         "dequant reduce type not convertible");
    mlir::Value seed =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    std::string callee = standaloneReductionIntrinsicName(
        *mnemonic, vectorDType(srcVecType), srcVecType.getLmul(),
        vectorDType(accVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduce.getTCRVEmitCLowerableSourceOpName(),
                         reduce.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value reduced =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{accEmitC}, callee,
                                         mlir::ValueRange{product, seed, loadVL})
            .getResult(0);
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec",
                           reduce.getTCRVEmitCLowerableSourceOpName(),
                           reduce.getTCRVEmitCLowerableSourceRole()));
    rewriter.create<emitc::AssignOp>(loc, accVar, reduced);
    valueMap[reduce.getResult()] = reduced;
    return mlir::success();
  }

  /// One product/reduce slice in the typed dequant body, grouped for emission:
  /// the i8mf4 lhs/rhs loads, the (widening or packed-i4) product op, and the
  /// standalone reduce that folds the product into the accumulator.
  struct DequantSlice {
    tcrvrvv::LoadOp lhsLoad;
    tcrvrvv::LoadOp rhsLoad;
    mlir::Operation *productOp = nullptr;
    tcrvrvv::StandaloneReduceOp reduce;
  };

  /// Emit the low-precision Gearbox product-reduce-dequantize body as a real
  /// emitc function body: a function-scoped i32 accumulator variable seeded from
  /// acc[0], a main chunk loop carrying `unroll_factor` product/reduce slices
  /// (step = vlmax * unroll_factor), a scalar tail loop over the remainder when
  /// unrolled, then the dequant epilogue (scalar extract -> f32 -> *scale ->
  /// store; with an optional clamp via splat/compare/select). The caller has
  /// already emitted the function entry, the runtime-ABI valueMap, the scope
  /// provenance comment, and the pre-loop full-chunk setvl (`vlmax`).
  mlir::LogicalResult emitLowPrecisionDequantBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Walk the scope into ordered product/reduce slices + the dequant epilogue.
    llvm::SmallVector<DequantSlice, 2> slices;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp storeOp;
    llvm::SmallVector<mlir::Operation *, 4> epilogueOps;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        // Group lhs/rhs loads with the product/reduce they feed; the first load
        // of each slice opens a new DequantSlice.
        if (slices.empty() || slices.back().reduce)
          slices.push_back(DequantSlice{});
        if (!slices.back().lhsLoad)
          slices.back().lhsLoad = load;
        else
          slices.back().rhsLoad = load;
      } else if (llvm::isa<tcrvrvv::WideningProductOp,
                           tcrvrvv::PackedI4NibbleUnpackProductOp>(op)) {
        if (slices.empty())
          return rewriter.notifyMatchFailure(scope,
                                             "dequant product before its loads");
        slices.back().productOp = &op;
      } else if (auto reduce = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
        if (slices.empty() || !slices.back().productOp)
          return rewriter.notifyMatchFailure(scope,
                                             "dequant reduce before its product");
        slices.back().reduce = reduce;
      } else if (auto deq = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op)) {
        dequant = deq;
        epilogueOps.push_back(&op);
      } else if (auto store = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        storeOp = store;
        epilogueOps.push_back(&op);
      } else if (llvm::isa<tcrvrvv::SplatOp, tcrvrvv::CompareOp,
                           tcrvrvv::SelectOp>(op)) {
        epilogueOps.push_back(&op);
      } else {
        // Any unexpected op (e.g. a leftover vsetvl_region_marker, the gearbox
        // handoff) means this is not the fully-typed dequant body the routine
        // owns yet -- fall back to legacy so nothing is mislowered.
        return rewriter.notifyMatchFailure(
            &op, "unsupported op in low-precision dequant body");
      }
    }
    if (slices.empty() || !dequant || !storeOp)
      return rewriter.notifyMatchFailure(
          scope, "dequant body missing product/reduce slices or epilogue");
    for (const DequantSlice &slice : slices)
      if (!slice.lhsLoad || !slice.rhsLoad || !slice.productOp || !slice.reduce)
        return rewriter.notifyMatchFailure(scope,
                                           "incomplete dequant product slice");

    // The accumulator vector type (i32 m1) and the acc[0] seed buffer.
    auto accVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(slices.front().reduce.getType());
    if (!accVecType || accVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(scope,
                                         "dequant accumulator not an m1 vector");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC)
      return rewriter.notifyMatchFailure(
          scope, "dequant accumulator type not convertible");
    mlir::Value lhsBuffer = valueMap.lookup(slices.front().lhsLoad.getBuffer());
    mlir::Value rhsBuffer = valueMap.lookup(slices.front().rhsLoad.getBuffer());
    mlir::Value accBuffer =
        valueMap.lookup(slices.front().reduce.getAccumulatorSeed());
    if (!lhsBuffer || !rhsBuffer || !accBuffer)
      return rewriter.notifyMatchFailure(scope,
                                         "dequant body buffers unmapped");

    // unroll_factor (structural op attr): the main loop carries this many copies
    // of the single typed product/reduce slice and steps by vlmax * unroll_factor;
    // absent or 1 == a single plain loop. The body carries exactly ONE typed slice
    // (the template); the conversion expands it `unroll` times. Each expanded copy
    // is byte-identical except for its sliceIndex-derived pointer offset and VL,
    // computed below -- so a single-slice unroll=2 body emits exactly the legacy
    // two-slice grouped C.
    int64_t unroll = 1;
    if (auto u = scope->getAttrOfType<mlir::IntegerAttr>("unroll_factor"))
      unroll = u.getInt();
    if (unroll < 1)
      return rewriter.notifyMatchFailure(
          scope, "dequant unroll_factor must be a positive integer");
    if (slices.size() != 1)
      return rewriter.notifyMatchFailure(
          scope, "dequant body must carry exactly one typed product/reduce slice "
                 "(unrolled via the with_vl unroll_factor attr)");

    llvm::StringRef reduceOpName =
        slices.front().reduce.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef reduceRole =
        slices.front().reduce.getTCRVEmitCLowerableSourceRole();

    // Function-scoped i32 accumulator variable: vint32m1_t dot_acc_vec;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("dot_acc_vec", reduceOpName, reduceRole));
    auto accLValueType = emitc::LValueType::get(accEmitC);
    auto accVar = rewriter.create<emitc::VariableOp>(
        loc, accLValueType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // Optional grouped tail bound: tail_start = (n / (vlmax*2)) * (vlmax*2).
    mlir::Value tailStart;
    if (unroll > 1) {
      mlir::Value unrollLit = rewriter.create<emitc::LiteralOp>(
          loc, sizeType, llvm::Twine(unroll).str());
      mlir::Value step =
          rewriter.create<emitc::MulOp>(loc, sizeType, vlmax, unrollLit);
      mlir::Value chunks =
          rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, step);
      tailStart = rewriter.create<emitc::MulOp>(loc, sizeType, chunks, step);
    }

    // Seed the accumulator from acc[0]: dot_acc_vec = vmv_v_x(acc[0], 1).
    mlir::Value seedSplat = emitScalarSeedSplat(rewriter, loc, accBuffer,
                                                accVecType, reduceOpName,
                                                reduceRole);
    if (!seedSplat)
      return rewriter.notifyMatchFailure(scope,
                                         "dequant accumulator seed not convertible");
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec", reduceOpName, reduceRole));
    rewriter.create<emitc::AssignOp>(loc, accVar, seedSplat);

    // The main chunk loop: for (i = 0; i < <bound>; i += vlmax*unroll).
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value mainBound = tailStart ? tailStart : avlArg;
    mlir::Value mainStep = vlmax;
    if (unroll > 1) {
      mlir::Value unrollLit = rewriter.create<emitc::LiteralOp>(
          loc, sizeType, llvm::Twine(unroll).str());
      mainStep = rewriter.create<emitc::MulOp>(loc, sizeType, vlmax, unrollLit);
    }
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, zero, mainBound, mainStep,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      mlir::Value inductionVar = mainLoop.getInductionVar();
      const DequantSlice &slice = slices.front();
      // Expand the single typed slice `unroll` times. The per-copy pointer offset
      // and VL derive from sliceIndex (not from any slice's own ops), so emitting
      // slices.front() with sliceIndex 0..unroll-1 reproduces the legacy unrolled
      // grouped C byte-for-byte.
      for (int64_t sliceIndex = 0; sliceIndex < unroll; ++sliceIndex) {
        // The slice's runtime VL is setvl(n - i - sliceIndex*vlmax); the second
        // unroll slice loads at (base + i + vlmax) and its VL excludes the first
        // slice's lanes.
        mlir::Value sliceOffset = inductionVar;
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
        if (sliceIndex > 0) {
          // base + i + sliceIndex*vlmax ; remaining -= sliceIndex*vlmax (via vl)
          for (int64_t k = 0; k < sliceIndex; ++k)
            sliceOffset =
                rewriter.create<emitc::AddOp>(loc, sizeType, sliceOffset, vlmax);
          for (int64_t k = 0; k < sliceIndex; ++k)
            remaining =
                rewriter.create<emitc::SubOp>(loc, sizeType, remaining, vlmax);
        }
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                             preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                             setvlCallee));
        mlir::Value sliceVL =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             setvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);
        if (mlir::failed(emitDequantProductReduceSlice(
                rewriter, loc, slice.lhsLoad, slice.rhsLoad, slice.productOp,
                slice.reduce, lhsBuffer, rhsBuffer, sliceOffset, accVar,
                accVecType, sliceVL, valueMap)))
          return mlir::failure();
      }
    }

    // The scalar tail loop (only when unrolled): one slice over the remainder,
    // for (i = tail_start; i < n; i += vlmax).
    if (unroll > 1) {
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, tailStart, avlArg,
                                                    vlmax, /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(tailLoop.getBody());
      mlir::Value inductionVar = tailLoop.getInductionVar();
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                           preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                           setvlCallee));
      mlir::Value sliceVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);
      const DequantSlice &slice = slices.front();
      if (mlir::failed(emitDequantProductReduceSlice(
              rewriter, loc, slice.lhsLoad, slice.rhsLoad, slice.productOp,
              slice.reduce, lhsBuffer, rhsBuffer, inductionVar, accVar,
              accVecType, sliceVL, valueMap)))
        return mlir::failure();
    }

    // Dequant epilogue (run once after the loops): scalar extract -> f32 ->
    // *scale -> splat -> store; optionally clamped via splat/compare/select.
    return emitDequantEpilogue(rewriter, loc, variant, dequant, storeOp,
                               epilogueOps, accVar, accVecType, valueMap);
  }

  /// Emit the DEFERRED-WIDE low-precision contraction body (the N3 resource-aware
  /// max-legal-LMUL schedule, the measured ssh-rvv winner var_v_m2_a1.c) as a
  /// real emitc function body:
  ///   vint32m8_t dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vsetvlmax_e32m8());
  ///   for (i = 0; i < n; i += vlmax) {
  ///     vl = setvl(n - i);
  ///     a = vle8_v_i8m2(lhs + i, vl);  b = vle8_v_i8m2(rhs + i, vl);
  ///     p = vwmul_vv_i16m4(a, b, vl);
  ///     dot_acc_vec = vwadd_wv_i32m8(dot_acc_vec, p, vl);   // DEFERRED accumulate
  ///   }
  ///   vint32m1_t vred = vredsum_vs_i32m8_i32m1(dot_acc_vec, zero, vlmax_acc);
  ///   int32_t sum = acc[0] + vmv_x_s_i32m1_i32(vred);       // SCALAR acc[0] add
  ///   out[0] = (float) sum * scale;                          // dequant epilogue
  /// The i32m8 accumulator is zero-seeded (NOT splatted from acc[0]); acc[0] is
  /// added as a SCALAR after the single trailing vredsum. There is NO
  /// per-iteration vwredsum. The body is body-determined: the structural marker
  /// is tcrv_rvv.widening_accumulate (I5).
  mlir::LogicalResult emitDeferredWideDequantBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Walk the single deferred-wide slice + dequant epilogue.
    tcrvrvv::LoadOp lhsLoad;
    tcrvrvv::LoadOp rhsLoad;
    tcrvrvv::WideningProductOp product;
    tcrvrvv::WideningAccumulateOp accumulate;
    tcrvrvv::StandaloneReduceOp reduce;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp storeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        if (!lhsLoad)
          lhsLoad = load;
        else if (!rhsLoad)
          rhsLoad = load;
        else
          return rewriter.notifyMatchFailure(
              scope, "deferred-wide body carries more than two loads");
      } else if (auto p = llvm::dyn_cast<tcrvrvv::WideningProductOp>(op)) {
        product = p;
      } else if (auto a = llvm::dyn_cast<tcrvrvv::WideningAccumulateOp>(op)) {
        accumulate = a;
      } else if (auto r = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
        reduce = r;
      } else if (auto d = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op)) {
        dequant = d;
      } else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        storeOp = s;
      } else {
        return rewriter.notifyMatchFailure(
            &op, "unsupported op in deferred-wide low-precision dequant body");
      }
    }
    if (!lhsLoad || !rhsLoad || !product || !accumulate || !reduce || !dequant ||
        !storeOp)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide body missing a load/product/accumulate/reduce/"
                 "dequant/store step");

    // The i32m8 deferred accumulator type and the acc[0] seed buffer.
    auto accVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(accumulate.getResult().getType());
    if (!accVecType || accVecType.getLmul() != "m8")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide accumulator not an i32m8 vector");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide accumulator type not convertible");
    auto reduceVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    if (!reduceVecType || reduceVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide trailing reduction result not an m1 vector");
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!reduceEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide trailing reduction type not convertible");

    mlir::Value lhsBuffer = valueMap.lookup(lhsLoad.getBuffer());
    mlir::Value rhsBuffer = valueMap.lookup(rhsLoad.getBuffer());
    mlir::Value accBuffer = valueMap.lookup(reduce.getAccumulatorSeed());
    if (!lhsBuffer || !rhsBuffer || !accBuffer)
      return rewriter.notifyMatchFailure(scope,
                                         "deferred-wide body buffers unmapped");

    llvm::StringRef accOpName =
        accumulate.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef accRole = accumulate.getTCRVEmitCLowerableSourceRole();
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);

    // Function-scoped i32m8 accumulator variable: vint32m8_t dot_acc_vec;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("dot_acc_vec", accOpName, accRole));
    auto accLValueType = emitc::LValueType::get(accEmitC);
    auto accVar = rewriter.create<emitc::VariableOp>(
        loc, accLValueType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // The accumulator's own VLMAX: vlmax_acc = __riscv_vsetvlmax_e32m8();
    std::string accVsetvlmaxCallee =
        ("__riscv_vsetvlmax_e" + llvm::Twine(accSEW) + accLmul).str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(accOpName, accRole, accVsetvlmaxCallee));
    mlir::Value accVlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         accVsetvlmaxCallee,
                                         mlir::ValueRange{})
            .getResult(0);

    // Zero-seed: dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vlmax_acc);
    std::string accZeroSplatCallee =
        riscvIntrinsicName("vmv_v_x", accSEW, accLmul, accDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(accOpName, accRole, accZeroSplatCallee));
    mlir::Value zeroLit =
        rewriter.create<emitc::LiteralOp>(loc, getSizeType(rewriter), "0");
    mlir::Value accSeed =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{accEmitC},
                                         accZeroSplatCallee,
                                         mlir::ValueRange{zeroLit, accVlmax})
            .getResult(0);
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec", accOpName, accRole));
    rewriter.create<emitc::AssignOp>(loc, accVar, accSeed);

    // The strip loop: for (i = 0; i < n; i += vlmax).
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      mlir::Value inductionVar = mainLoop.getInductionVar();
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                           preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                           setvlCallee));
      mlir::Value sliceVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);
      // Wide i8m2 loads at base + i, the i16m4 widening product (both reuse the
      // narrow emitters -- they derive <dtype><lmul> from the wide typed
      // vectors, so they emit vle8_v_i8m2 / vwmul_vv_i16m4 unchanged).
      if (mlir::failed(emitLoad(rewriter, loc, lhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(emitLoad(rewriter, loc, rhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(
              emitWideningProduct(rewriter, loc, product, valueMap, sliceVL)))
        return mlir::failure();
      mlir::Value productValue = valueMap.lookup(product.getResult());
      if (!productValue)
        return rewriter.notifyMatchFailure(accumulate,
                                           "deferred-wide product unmapped");
      // DEFERRED accumulate: dot_acc_vec = vwadd_wv_i32m8(dot_acc_vec, p, vl).
      std::string accumulateCallee =
          riscvWideningAccumulateIntrinsicName(accSEW, accLmul, accDtype);
      mlir::Value runningAcc =
          rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(accOpName, accRole, accumulateCallee));
      mlir::Value accumulated =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{accEmitC}, accumulateCallee,
                  mlir::ValueRange{runningAcc, productValue, sliceVL})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("dot_acc_vec", accOpName, accRole));
      rewriter.create<emitc::AssignOp>(loc, accVar, accumulated);
      valueMap[accumulate.getResult()] = accumulated;
    }

    // Trailing reduction + scalar epilogue: ONE vredsum folds the i32m8
    // accumulator, then acc[0] is added as a SCALAR, then dequant.
    return emitDeferredWideEpilogue(rewriter, loc, variant, reduce, dequant,
                                    storeOp, accVar, accVecType, reduceVecType,
                                    accVlmax, accBuffer, valueMap);
  }

  /// The deferred-wide trailing reduction + scalar acc[0] add + dequant store:
  ///   vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vsetvlmax_e32m1());
  ///   vint32m1_t vred  = __riscv_vredsum_vs_i32m8_i32m1(dot_acc_vec, vzero, vlmax_acc);
  ///   int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  ///   float scaled = (float) sum * scale;
  ///   out[0] = vse32(vfmv_v_f(scaled, 1));
  /// The acc[0] seed is added as a SCALAR (the i32m8 accumulator was zero-seeded).
  mlir::LogicalResult emitDeferredWideEpilogue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::StandaloneReduceOp reduce,
      tcrvrvv::DequantizeOp dequant, tcrvrvv::StoreOp storeOp, mlir::Value accVar,
      tcrvrvv::VectorType accVecType, tcrvrvv::VectorType reduceVecType,
      mlir::Value accVlmax, mlir::Value accBuffer,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::Value scale = valueMap.lookup(dequant.getScale());
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!scale || !outBuffer)
      return rewriter.notifyMatchFailure(
          dequant, "deferred-wide dequant epilogue operands unmapped");
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequant.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant result not an f32 vector");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!resultEmitC || !accEmitC || !reduceEmitC)
      return rewriter.notifyMatchFailure(
          dequant, "deferred-wide epilogue type not convertible");

    llvm::StringRef reduceOpName =
        reduce.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef reduceRole = reduce.getTCRVEmitCLowerableSourceRole();
    llvm::StringRef opName = dequant.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = dequant.getTCRVEmitCLowerableSourceRole();
    mlir::Type sizeType = getSizeType(rewriter);
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);
    llvm::StringRef reduceDtype = vectorDType(reduceVecType);

    // The reduction destination m1 VLMAX + zero seed:
    //   vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vsetvlmax_e32m1());
    std::string redVsetvlmaxCallee =
        ("__riscv_vsetvlmax_e" + llvm::Twine(accSEW) + reduceVecType.getLmul())
            .str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, redVsetvlmaxCallee));
    mlir::Value reduceVlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         redVsetvlmaxCallee,
                                         mlir::ValueRange{})
            .getResult(0);
    std::string zeroSeedCallee = riscvIntrinsicName(
        "vmv_v_x", accSEW, reduceVecType.getLmul(), reduceDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, zeroSeedCallee));
    mlir::Value zeroLit =
        rewriter.create<emitc::LiteralOp>(loc, getSizeType(rewriter), "0");
    mlir::Value reduceZero =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{reduceEmitC},
                                         zeroSeedCallee,
                                         mlir::ValueRange{zeroLit, reduceVlmax})
            .getResult(0);

    // ONE trailing vredsum over the i32m8 accumulator:
    //   vint32m1_t vred = __riscv_vredsum_vs_i32m8_i32m1(dot_acc_vec, vzero, vlmax_acc);
    std::optional<llvm::StringRef> reduceMnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!reduceMnemonic)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide trailing reduce kind unsupported");
    std::string reduceCallee = standaloneReductionIntrinsicName(
        *reduceMnemonic, accDtype, accLmul, reduceDtype);
    mlir::Value accValue =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, reduceCallee));
    mlir::Value reduced =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{reduceEmitC}, reduceCallee,
                mlir::ValueRange{accValue, reduceZero, accVlmax})
            .getResult(0);
    valueMap[reduce.getResult()] = reduced;

    // Extract lane 0: int32_t reduced_scalar = __riscv_vmv_x_s_i32m1_i32(vred);
    std::string extractCallee =
        ("__riscv_vmv_x_s_" + reduceDtype + reduceVecType.getLmul() + "_" +
         reduceDtype)
            .str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, extractCallee));
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    mlir::Value reducedScalar =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                         extractCallee,
                                         mlir::ValueRange{reduced})
            .getResult(0);

    // acc[0] as a SCALAR: int32_t sum = acc[0] + reduced_scalar.
    auto accPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(accBuffer);
    if (!accPointer)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide acc seed buffer not a pointer");
    mlir::Value seedIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp seedSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, accPointer, seedIndex);
    auto seedLValueType =
        llvm::cast<emitc::LValueType>(seedSubscript.getResult().getType());
    mlir::Value seedScalar =
        rewriter
            .create<emitc::LoadOp>(loc, seedLValueType.getValueType(),
                                   seedSubscript.getResult())
            .getResult();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, "scalar_acc0_add"));
    mlir::Value sum =
        rewriter.create<emitc::AddOp>(loc, i32Type, seedScalar, reducedScalar);

    // Dequant: float scaled = (float) sum * scale; out[0] = vse32(vfmv_v_f(scaled, 1)).
    unsigned resSEW = vectorElementWidth(resultVecType);
    llvm::StringRef resLmul = resultVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(resultVecType);
    mlir::Type floatType =
        emitc::OpaqueType::get(rewriter.getContext(), "float");
    mlir::Value asFloat =
        rewriter.create<emitc::CastOp>(loc, floatType, sum).getResult();
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, asFloat, scale);
    std::string splatCallee =
        riscvIntrinsicName("vfmv_v_f", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, splatCallee));
    mlir::Value one = rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    mlir::Value dequantSplat =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         splatCallee,
                                         mlir::ValueRange{scaled, one})
            .getResult(0);
    valueMap[dequant.getResult()] = dequantSplat;
    std::string storeCallee =
        riscvIntrinsicName("vse", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(storeOp.getTCRVEmitCLowerableSourceOpName(),
                         storeOp.getTCRVEmitCLowerableSourceRole(), storeCallee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, storeCallee,
        mlir::ValueRange{outBuffer, dequantSplat, one});
    return mlir::success();
  }

  /// Emit the 2nd-family (i16 dot-reduce) deferred-wide body as a real RVV
  /// strip loop (the measured ssh-rvv winner dot_wide_deferred):
  ///   size_t vlmax_acc = __riscv_vsetvlmax_e32m8();
  ///   vint32m8_t dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vlmax_acc);
  ///   for (i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e16m4(n - i);
  ///     vint16m4_t a = __riscv_vle16_v_i16m4(lhs + i, vl);
  ///     vint16m4_t b = __riscv_vle16_v_i16m4(rhs + i, vl);
  ///     vint32m8_t p = __riscv_vwmul_vv_i32m8(a, b, vl);   // SINGLE widening
  ///     dot_acc_vec = __riscv_vadd_vv_i32m8(dot_acc_vec, p, vl); // NON-widening
  ///   }
  ///   // ONE trailing vredsum + scalar acc[0] add + i32 lane-0 store.
  /// The structural marker is tcrv_rvv.deferred_accumulate (I5).
  mlir::LogicalResult emitDeferredWideDotReduceBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::WithVLOp scope,
      tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg, mlir::Value vlmax,
      mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::LoadOp lhsLoad;
    tcrvrvv::LoadOp rhsLoad;
    tcrvrvv::WideningProductOp product;
    tcrvrvv::DeferredAccumulateOp accumulate;
    tcrvrvv::StandaloneReduceOp reduce;
    tcrvrvv::StoreOp storeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto load = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        if (!lhsLoad)
          lhsLoad = load;
        else if (!rhsLoad)
          rhsLoad = load;
        else
          return rewriter.notifyMatchFailure(
              scope, "deferred-wide dot-reduce body carries more than two loads");
      } else if (auto p = llvm::dyn_cast<tcrvrvv::WideningProductOp>(op)) {
        product = p;
      } else if (auto a = llvm::dyn_cast<tcrvrvv::DeferredAccumulateOp>(op)) {
        accumulate = a;
      } else if (auto r = llvm::dyn_cast<tcrvrvv::StandaloneReduceOp>(op)) {
        reduce = r;
      } else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        storeOp = s;
      } else {
        return rewriter.notifyMatchFailure(
            &op, "unsupported op in deferred-wide dot-reduce body");
      }
    }
    if (!lhsLoad || !rhsLoad || !product || !accumulate || !reduce || !storeOp)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce body missing a load/product/"
                 "accumulate/reduce/store step");

    auto accVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(accumulate.getResult().getType());
    if (!accVecType || accVecType.getLmul() != "m8")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce accumulator not an i32m8 vector");
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!accEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce accumulator type not convertible");
    auto reduceVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(reduce.getResult().getType());
    if (!reduceVecType || reduceVecType.getLmul() != "m1")
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce trailing reduction result not m1");
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!reduceEmitC)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce trailing reduction not convertible");

    mlir::Value lhsBuffer = valueMap.lookup(lhsLoad.getBuffer());
    mlir::Value rhsBuffer = valueMap.lookup(rhsLoad.getBuffer());
    mlir::Value accBuffer = valueMap.lookup(reduce.getAccumulatorSeed());
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!lhsBuffer || !rhsBuffer || !accBuffer || !outBuffer)
      return rewriter.notifyMatchFailure(
          scope, "deferred-wide dot-reduce body buffers unmapped");

    llvm::StringRef accOpName = accumulate.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef accRole = accumulate.getTCRVEmitCLowerableSourceRole();
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);

    // Function-scoped i32m8 accumulator: vint32m8_t dot_acc_vec;
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("dot_acc_vec", accOpName, accRole));
    auto accLValueType = emitc::LValueType::get(accEmitC);
    auto accVar = rewriter.create<emitc::VariableOp>(
        loc, accLValueType, emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // The accumulator's own VLMAX: vlmax_acc = __riscv_vsetvlmax_e32m8();
    std::string accVsetvlmaxCallee =
        ("__riscv_vsetvlmax_e" + llvm::Twine(accSEW) + accLmul).str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(accOpName, accRole, accVsetvlmaxCallee));
    mlir::Value accVlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         accVsetvlmaxCallee, mlir::ValueRange{})
            .getResult(0);

    // Zero-seed: dot_acc_vec = __riscv_vmv_v_x_i32m8(0, vlmax_acc);
    std::string accZeroSplatCallee =
        riscvIntrinsicName("vmv_v_x", accSEW, accLmul, accDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(accOpName, accRole, accZeroSplatCallee));
    mlir::Value zeroLit =
        rewriter.create<emitc::LiteralOp>(loc, getSizeType(rewriter), "0");
    mlir::Value accSeed =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{accEmitC},
                                         accZeroSplatCallee,
                                         mlir::ValueRange{zeroLit, accVlmax})
            .getResult(0);
    rewriter.create<emitc::VerbatimOp>(
        loc, assignComment("dot_acc_vec", accOpName, accRole));
    rewriter.create<emitc::AssignOp>(loc, accVar, accSeed);

    // The strip loop: for (i = 0; i < n; i += vlmax).
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto mainLoop = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                                  /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(mainLoop.getBody());
      mlir::Value inductionVar = mainLoop.getInductionVar();
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                           preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                           setvlCallee));
      mlir::Value sliceVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);
      // Wide i16m4 loads at base + i, the i32m8 SINGLE-step widening product
      // (both reuse the narrow emitters -- they derive <dtype><lmul> from the
      // wide typed vectors, so they emit vle16_v_i16m4 / vwmul_vv_i32m8).
      if (mlir::failed(emitLoad(rewriter, loc, lhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(emitLoad(rewriter, loc, rhsLoad, valueMap, inductionVar,
                                sliceVL)))
        return mlir::failure();
      if (mlir::failed(
              emitWideningProduct(rewriter, loc, product, valueMap, sliceVL)))
        return mlir::failure();
      mlir::Value productValue = valueMap.lookup(product.getResult());
      if (!productValue)
        return rewriter.notifyMatchFailure(accumulate,
                                           "deferred-wide dot-reduce product "
                                           "unmapped");
      // DEFERRED NON-widening accumulate: dot_acc_vec =
      //   __riscv_vadd_vv_i32m8(dot_acc_vec, p, vl). riscvIntrinsicName's
      //   default arm appends the "_vv_<dtype><lmul>" arithmetic form.
      std::string accumulateCallee =
          riscvIntrinsicName("vadd", accSEW, accLmul, accDtype);
      mlir::Value runningAcc =
          rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(accOpName, accRole, accumulateCallee));
      mlir::Value accumulated =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{accEmitC}, accumulateCallee,
                  mlir::ValueRange{runningAcc, productValue, sliceVL})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("dot_acc_vec", accOpName, accRole));
      rewriter.create<emitc::AssignOp>(loc, accVar, accumulated);
      valueMap[accumulate.getResult()] = accumulated;
    }

    // Trailing reduction + scalar epilogue + i32 lane-0 store.
    return emitDeferredWideDotReduceEpilogue(
        rewriter, loc, reduce, storeOp, accVar, accVecType, reduceVecType,
        accVlmax, accBuffer, outBuffer, valueMap);
  }

  /// The deferred-wide dot-reduce trailing reduction + scalar acc[0] add + i32
  /// lane-0 store (NO dequant):
  ///   vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vsetvlmax_e32m1());
  ///   vint32m1_t vred  = __riscv_vredsum_vs_i32m8_i32m1(dot_acc_vec, vzero, vlmax_acc);
  ///   int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  ///   out[0] = vse32(__riscv_vmv_v_x_i32m1(sum, 1));
  mlir::LogicalResult emitDeferredWideDotReduceEpilogue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::StandaloneReduceOp reduce, tcrvrvv::StoreOp storeOp,
      mlir::Value accVar, tcrvrvv::VectorType accVecType,
      tcrvrvv::VectorType reduceVecType, mlir::Value accVlmax,
      mlir::Value accBuffer, mlir::Value outBuffer,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    mlir::Type reduceEmitC = convertVectorTypeToEmitC(reduceVecType);
    if (!accEmitC || !reduceEmitC)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide dot-reduce epilogue type not convertible");

    llvm::StringRef reduceOpName = reduce.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef reduceRole = reduce.getTCRVEmitCLowerableSourceRole();
    mlir::Type sizeType = getSizeType(rewriter);
    unsigned accSEW = vectorElementWidth(accVecType);
    llvm::StringRef accLmul = accVecType.getLmul();
    llvm::StringRef accDtype = vectorDType(accVecType);
    llvm::StringRef reduceDtype = vectorDType(reduceVecType);

    // The reduction destination m1 VLMAX + zero seed.
    std::string redVsetvlmaxCallee =
        ("__riscv_vsetvlmax_e" + llvm::Twine(accSEW) + reduceVecType.getLmul())
            .str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, redVsetvlmaxCallee));
    mlir::Value reduceVlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         redVsetvlmaxCallee, mlir::ValueRange{})
            .getResult(0);
    std::string zeroSeedCallee = riscvIntrinsicName(
        "vmv_v_x", accSEW, reduceVecType.getLmul(), reduceDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, zeroSeedCallee));
    mlir::Value zeroLit =
        rewriter.create<emitc::LiteralOp>(loc, getSizeType(rewriter), "0");
    mlir::Value reduceZero =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{reduceEmitC},
                                         zeroSeedCallee,
                                         mlir::ValueRange{zeroLit, reduceVlmax})
            .getResult(0);

    // ONE trailing vredsum over the i32m8 accumulator.
    std::optional<llvm::StringRef> reduceMnemonic =
        standaloneReductionMnemonic(reduce.getKind());
    if (!reduceMnemonic)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide dot-reduce trailing reduce kind unsupported");
    std::string reduceCallee = standaloneReductionIntrinsicName(
        *reduceMnemonic, accDtype, accLmul, reduceDtype);
    mlir::Value accValue =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, reduceCallee));
    mlir::Value reduced =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{reduceEmitC},
                                         reduceCallee,
                                         mlir::ValueRange{accValue, reduceZero,
                                                          accVlmax})
            .getResult(0);
    valueMap[reduce.getResult()] = reduced;

    // Extract lane 0: int32_t reduced_scalar = __riscv_vmv_x_s_i32m1_i32(vred);
    std::string extractCallee =
        ("__riscv_vmv_x_s_" + reduceDtype + reduceVecType.getLmul() + "_" +
         reduceDtype)
            .str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, extractCallee));
    mlir::Type i32Type =
        emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    mlir::Value reducedScalar =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                         extractCallee,
                                         mlir::ValueRange{reduced})
            .getResult(0);

    // acc[0] as a SCALAR: int32_t sum = acc[0] + reduced_scalar.
    auto accPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(accBuffer);
    if (!accPointer)
      return rewriter.notifyMatchFailure(
          reduce, "deferred-wide dot-reduce acc seed buffer not a pointer");
    mlir::Value seedIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp seedSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, accPointer, seedIndex);
    auto seedLValueType =
        llvm::cast<emitc::LValueType>(seedSubscript.getResult().getType());
    mlir::Value seedScalar =
        rewriter
            .create<emitc::LoadOp>(loc, seedLValueType.getValueType(),
                                   seedSubscript.getResult())
            .getResult();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(reduceOpName, reduceRole, "scalar_acc0_add"));
    mlir::Value sum =
        rewriter.create<emitc::AddOp>(loc, i32Type, seedScalar, reducedScalar);

    // out[0] = vse32(__riscv_vmv_v_x_i32m1(sum, 1)).
    unsigned resSEW = vectorElementWidth(reduceVecType);
    llvm::StringRef resLmul = reduceVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(reduceVecType);
    std::string splatCallee =
        riscvIntrinsicName("vmv_v_x", resSEW, resLmul, resDtype);
    llvm::StringRef storeOpName = storeOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef storeRole = storeOp.getTCRVEmitCLowerableSourceRole();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(storeOpName, storeRole, splatCallee));
    mlir::Value one = rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    mlir::Value storeSplat =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{reduceEmitC},
                                         splatCallee,
                                         mlir::ValueRange{sum, one})
            .getResult(0);
    std::string storeCallee =
        riscvIntrinsicName("vse", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(storeOpName, storeRole, storeCallee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, storeCallee,
        mlir::ValueRange{outBuffer, storeSplat, one});
    return mlir::success();
  }

  /// True iff `scope` is the STANDALONE i32->f32 runtime-scale dequant body: a
  /// with_vl scope whose ONLY compute is a single load -> dequantize -> store
  /// (no product / reduce / accumulator / clamp). Structurally: exactly one
  /// tcrv_rvv.load producing an i32 vector, exactly one tcrv_rvv.dequantize
  /// (kind `i32_to_f32_scaled`, i32->f32) sourcing that load, exactly one
  /// tcrv_rvv.store of the f32 result, and nothing else. This is the dequant
  /// shape the legacy string materializer Gearbox-unrolls (u2) into a two-slice
  /// runtime-avl setvl loop; `emitStandaloneDequantBody` reproduces that loop.
  /// Any extra op, a different dtype/kind, or a missing piece returns false so
  /// the body falls to the (guarded) emitScopeForLoop path and then the legacy
  /// materializer -- no mislower.
  static bool isStandaloneDequantBody(tcrvrvv::WithVLOp scope) {
    tcrvrvv::LoadOp load;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp store;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto l = llvm::dyn_cast<tcrvrvv::LoadOp>(op)) {
        if (load)
          return false; // more than one load is not the standalone shape
        load = l;
      } else if (auto d = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op)) {
        if (dequant)
          return false;
        dequant = d;
      } else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op)) {
        if (store)
          return false;
        store = s;
      } else {
        return false; // any other op (product/reduce/select/...) excludes it
      }
    }
    if (!load || !dequant || !store)
      return false;
    // The dequantize must source the load and feed the store, kind/dtype pinned.
    if (dequant.getKind() != "i32_to_f32_scaled")
      return false;
    if (dequant.getSource() != load.getLoaded())
      return false;
    if (store.getValue() != dequant.getResult())
      return false;
    auto srcVec =
        llvm::dyn_cast<tcrvrvv::VectorType>(load.getLoaded().getType());
    auto resVec =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequant.getResult().getType());
    if (!srcVec || !resVec)
      return false;
    if (!srcVec.getElementType().isSignlessInteger(32) ||
        !resVec.getElementType().isF32())
      return false;
    return true;
  }

  /// Emit the standalone i32->f32 runtime-scale dequant body as the legacy
  /// Gearbox-unrolled (u<unroll>) two-slice runtime-avl setvl loop, byte-
  /// identical to the legacy string materializer:
  ///   v5 = setvl(n);  v6 = v5 * unroll;
  ///   for (i = 0; i < n; i += v6) {
  ///     <slice 0>  setvl(n - i)               load(base+i) -> dequant -> store
  ///     <slice 1>  setvl((n - i) - vl0)        load(base+i+vl0) -> ... -> store
  ///     ...
  ///   }
  /// Each slice k recomputes the remaining AVL FRESH from (n - i) and subtracts
  /// the runtime VLs of the prior slices in this iteration; the pointer offset
  /// for slice k>0 is the running sum of prior slice VLs (one extra pointer add
  /// per accumulated VL, matching the legacy `v18=base+i; v19=v18+vl0` form).
  /// The two-slice remaining-VL setvl covers the tail naturally -- there is NO
  /// separate scalar tail loop (unlike the product-reduce dequant routine).
  /// The unroll factor is read from the realized scope's `tcrv_rvv.gearbox.unroll`
  /// attribute (the Gearbox schedule fact); absent or non-positive fails the
  /// match so the body falls back to the legacy materializer unchanged.
  mlir::LogicalResult emitStandaloneDequantBody(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, tcrvrvv::SetVLOp preLoopSetVL, mlir::Value avlArg,
      mlir::Value vlmax, mlir::Type sizeType, llvm::StringRef setvlCallee,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    // Resolve the single load/dequantize/store template ops.
    tcrvrvv::LoadOp load;
    tcrvrvv::DequantizeOp dequant;
    tcrvrvv::StoreOp store;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto l = llvm::dyn_cast<tcrvrvv::LoadOp>(op))
        load = l;
      else if (auto d = llvm::dyn_cast<tcrvrvv::DequantizeOp>(op))
        dequant = d;
      else if (auto s = llvm::dyn_cast<tcrvrvv::StoreOp>(op))
        store = s;
    }
    if (!load || !dequant || !store)
      return rewriter.notifyMatchFailure(
          scope, "standalone dequant body missing load/dequantize/store");

    mlir::Value scale = valueMap.lookup(dequant.getScale());
    if (!scale)
      return rewriter.notifyMatchFailure(scope,
                                         "standalone dequant scale unmapped");

    // The Gearbox unroll factor (schedule fact carried on the realized scope).
    // The simple two-slice expansion below reproduces the legacy u<unroll> loop;
    // require it present and >= 1 so an un-scheduled body falls back.
    int64_t unroll = 0;
    if (auto u =
            scope->getAttrOfType<mlir::IntegerAttr>("tcrv_rvv.gearbox.unroll"))
      unroll = u.getInt();
    if (unroll < 1)
      return rewriter.notifyMatchFailure(
          scope, "standalone dequant body missing positive "
                 "tcrv_rvv.gearbox.unroll schedule fact");

    // v6 = v5 * unroll (the unrolled loop step). For unroll == 1 the step is
    // plain vlmax (no literal multiply, matching the un-unrolled single loop).
    mlir::Value step = vlmax;
    if (unroll > 1) {
      mlir::Value unrollLit = rewriter.create<emitc::LiteralOp>(
          loc, sizeType, llvm::Twine(unroll).str());
      step = rewriter.create<emitc::MulOp>(loc, sizeType, vlmax, unrollLit);
    }

    // for (size_t i = 0; i < n; i += step) { <unroll slices> }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, step,
                                               /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());
      mlir::Value inductionVar = forOp.getInductionVar();

      // priorVLSum accumulates the runtime VLs of the prior slices in this
      // iteration: slice 0 sees null (offset 0), slice k>0 sees the sum of
      // vl0..vl(k-1). For u2 this is exactly vl0, matching the legacy golden.
      mlir::Value priorVLSum;
      for (int64_t sliceIndex = 0; sliceIndex < unroll; ++sliceIndex) {
        // Remaining AVL for this slice: recompute (n - i) FRESH, then subtract
        // the prior slices' VLs one at a time (the legacy golden subtracts vl0
        // for the second slice via a dedicated `v16 = v15 - v9`).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(preLoopSetVL.getTCRVEmitCLowerableSourceOpName(),
                             preLoopSetVL.getTCRVEmitCLowerableSourceRole(),
                             setvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
        if (priorVLSum)
          remaining =
              rewriter.create<emitc::SubOp>(loc, sizeType, remaining,
                                            priorVLSum);
        mlir::Value sliceVL =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             setvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        // load(base + i [+ priorVLSum]) -> dequant chain -> store(out + i [+ ...])
        if (mlir::failed(emitLoad(rewriter, loc, load, valueMap, inductionVar,
                                  sliceVL, priorVLSum)))
          return mlir::failure();
        mlir::Value source = valueMap.lookup(load.getLoaded());
        if (!source)
          return rewriter.notifyMatchFailure(
              scope, "standalone dequant load result unmapped");
        if (mlir::failed(emitDequantizeChain(rewriter, loc, dequant, source,
                                             scale, valueMap, sliceVL)))
          return mlir::failure();
        if (mlir::failed(emitStore(rewriter, loc, store, valueMap, inductionVar,
                                   sliceVL, priorVLSum)))
          return mlir::failure();

        // Accumulate this slice's VL for the NEXT slice's offset/remaining. The
        // final slice has no successor, so skip the (otherwise dead) add to stay
        // byte-identical to the legacy golden (no trailing `vl0 + vl1`).
        if (sliceIndex + 1 < unroll)
          priorVLSum = priorVLSum ? rewriter.create<emitc::AddOp>(
                                        loc, sizeType, priorVLSum, sliceVL)
                                  : sliceVL;
      }
    }
    return mlir::success();
  }

  /// Emit the dequant epilogue once: extract the accumulator's lane-0 scalar
  /// (vmv_x_s), convert to f32, multiply by the runtime scale, then either store
  /// the scalar f32 (the plain dequant), or run the f32 clamp (splat/compare/
  /// select against the lower/upper bounds) before storing. The accumulator is
  /// read from the function-scoped variable, not any in-loop SSA value.
  mlir::LogicalResult emitDequantEpilogue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrv::exec::VariantOp variant, tcrvrvv::DequantizeOp dequant,
      tcrvrvv::StoreOp storeOp, llvm::ArrayRef<mlir::Operation *> epilogueOps,
      mlir::Value accVar, tcrvrvv::VectorType accVecType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    bool hasClamp = llvm::any_of(epilogueOps, [](mlir::Operation *op) {
      return llvm::isa<tcrvrvv::SelectOp>(op);
    });
    mlir::Value scale = valueMap.lookup(dequant.getScale());
    mlir::Value outBuffer = valueMap.lookup(storeOp.getBuffer());
    if (!scale || !outBuffer)
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant epilogue operands unmapped");
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequant.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant result not an f32 vector");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type accEmitC = convertVectorTypeToEmitC(accVecType);
    if (!resultEmitC || !accEmitC)
      return rewriter.notifyMatchFailure(dequant,
                                         "dequant epilogue type not convertible");
    llvm::StringRef opName = dequant.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = dequant.getTCRVEmitCLowerableSourceRole();
    unsigned resSEW = vectorElementWidth(resultVecType);
    llvm::StringRef resLmul = resultVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(resultVecType);
    mlir::Type sizeType = getSizeType(rewriter);

    // int32_t dot_acc_scalar = __riscv_vmv_x_s_i32m1_i32(dot_acc_vec);
    mlir::Value accValue =
        rewriter.create<emitc::LoadOp>(loc, accEmitC, accVar).getResult();
    std::string extractCallee =
        ("__riscv_vmv_x_s_" + vectorDType(accVecType) + accVecType.getLmul() +
         "_" + vectorDType(accVecType))
            .str();
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, extractCallee));
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int32_t");
    mlir::Value scalar =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                         extractCallee,
                                         mlir::ValueRange{accValue})
            .getResult(0);
    // float f = (float) dot_acc_scalar; float scaled = f * scale;
    mlir::Type floatType =
        emitc::OpaqueType::get(rewriter.getContext(), "float");
    mlir::Value asFloat =
        rewriter.create<emitc::CastOp>(loc, floatType, scalar).getResult();
    mlir::Value scaled =
        rewriter.create<emitc::MulOp>(loc, floatType, asFloat, scale);

    // The scaled f32 lane-0 result splat: vfmv_v_f(scaled, 1). Both the plain
    // dequant store and the clamp's compare inputs consume this lane-0 vector.
    std::string splatCallee =
        riscvIntrinsicName("vfmv_v_f", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, splatCallee));
    mlir::Value one = rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    mlir::Value dequantSplat =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         splatCallee,
                                         mlir::ValueRange{scaled, one})
            .getResult(0);
    valueMap[dequant.getResult()] = dequantSplat;

    mlir::Value valueToStore = dequantSplat;
    if (hasClamp) {
      // The f32 clamp runs on the lane-0 vector at VL=1: splat lower/upper
      // bounds, compare (vmflt), and select (vmerge) against the dequant result,
      // reusing the already-converted Splat/Compare/Select handlers. The store
      // target is the final selected value.
      for (mlir::Operation *epOp : epilogueOps) {
        if (auto splat = llvm::dyn_cast<tcrvrvv::SplatOp>(epOp)) {
          if (mlir::failed(emitSplat(rewriter, loc, splat, valueMap, one)))
            return mlir::failure();
        } else if (auto compare = llvm::dyn_cast<tcrvrvv::CompareOp>(epOp)) {
          if (mlir::failed(emitCompare(rewriter, loc, compare, valueMap, one)))
            return mlir::failure();
        } else if (auto select = llvm::dyn_cast<tcrvrvv::SelectOp>(epOp)) {
          if (mlir::failed(emitSelect(rewriter, loc, select, valueMap, one)))
            return mlir::failure();
          valueToStore = valueMap.lookup(select.getSelected());
        }
      }
      if (!valueToStore)
        return rewriter.notifyMatchFailure(
            variant, "dequant-clamp select result unmapped");
    }

    // vse32 store the lane-0 result to out base (VL=1).
    std::string storeCallee =
        riscvIntrinsicName("vse", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(storeOp.getTCRVEmitCLowerableSourceOpName(),
                         storeOp.getTCRVEmitCLowerableSourceRole(), storeCallee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, storeCallee,
        mlir::ValueRange{outBuffer, valueToStore, one});
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
  /// When `extraOffset` is set, a SECOND pointer add is emitted after the
  /// `base + i` add: ptr2 = ptr + extraOffset (mirroring the legacy unrolled
  /// `v18 = base + i; v19 = v18 + priorVL` two-add form for the second u2 slice).
  /// Existing single-slice callers pass the default null and emit one add.
  mlir::LogicalResult
  emitLoad(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
           tcrvrvv::LoadOp load,
           llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
           mlir::Value inductionVar, mlir::Value bodyVL,
           mlir::Value extraOffset = {}) const {
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
    if (extraOffset)
      ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), ptr, extraOffset);
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

    std::optional<llvm::StringRef> mnemonic =
        binaryMnemonic(binary.getKind(), isFloatVector(vectorType));
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
    // typed vector dataflow value, not a broadcast/splat scalar seed. The
    // asymmetric offset-binary packed-i4 x plain-i8 product (ggml Q4_0 x Q8_0
    // integer core) chains the SAME way (load x3 -> one-sided decode + vwmul/
    // vwmacc -> vwredsum), so its i16mf2 result is likewise a real typed vector
    // producer for the reduction.
    {
      mlir::Operation *inputDef = reduce.getInput().getDefiningOp();
      if (!inputDef ||
          !llvm::isa<tcrvrvv::LoadOp, tcrvrvv::WideningProductOp,
                     tcrvrvv::PackedI4OffsetBinaryXI8ProductOp>(inputDef))
        return rewriter.notifyMatchFailure(
            reduce, "standalone reduce input must be an explicit vector load, "
                    "widening_product, or packed-i4 offset-binary x i8 product "
                    "result");
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
    // The signed rung emits vwmul; the unsigned low-precision rung (ui8 source ->
    // ui16 result, kind=unsigned_widening_product) emits vwmulu, byte-identical
    // to the legacy unsigned widening-product oracle. Both must agree: a signed
    // kind on unsigned vectors (or vice versa) is a malformed body -> fall back.
    const bool unsignedProduct =
        product.getKind() == "unsigned_widening_product";
    if (product.getKind() != "signed_widening_product" && !unsignedProduct)
      return rewriter.notifyMatchFailure(
          product, "only the signed/unsigned widening product is convertible");
    if (unsignedProduct != isUnsignedVector(resultVecType) ||
        unsignedProduct != isUnsignedVector(lhsVecType))
      return rewriter.notifyMatchFailure(
          product, "widening product kind/signedness mismatch with vector types");
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
    std::string callee = riscvIntrinsicName(
        unsignedProduct ? "vwmulu" : "vwmul", vectorElementWidth(resultVecType),
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

  /// packed_i4_nibble_unpack_product(%lhs,%rhs,%vl) lowers to the FIXED signed
  /// i4-nibble sign-extend + widening-product intrinsic chain (each i8 packs two
  /// signed 4-bit nibbles), byte-equivalent to the legacy packed-i4 oracle:
  ///   // low nibble: shift the i4 into the high nibble (vsll 4), widen-multiply,
  ///   // arithmetic-shift the i16 product back down (vsra 8) -- this both
  ///   // sign-extends the low nibble and rescales the product.
  ///   v8  lhs_low = vsll_vx_i8mf4(lhs, 4, vl);
  ///   v8  rhs_low = vsll_vx_i8mf4(rhs, 4, vl);
  ///   v16 low_p   = vwmul_vv_i16mf2(lhs_low, rhs_low, vl);
  ///   v16 product = vsra_vx_i16mf2(low_p, 8, vl);
  ///   // high nibble: arithmetic-shift sign-extends the high nibble in place,
  ///   // then a widening multiply-accumulate adds its product.
  ///   v8  lhs_high = vsra_vx_i8mf4(lhs, 4, vl);
  ///   v8  rhs_high = vsra_vx_i8mf4(rhs, 4, vl);
  ///   v16 product  = vwmacc_vv_i16mf2(product, lhs_high, rhs_high, vl);
  /// The op carries the nibble-unpack STRUCTURE typed; the conversion never
  /// reads operand_form / unpack_intent candidate-mirror strings to choose it.
  mlir::LogicalResult emitPackedI4NibbleUnpackProduct(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::PackedI4NibbleUnpackProductOp packed,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(packed.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(packed.getLhs().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(packed,
                                         "packed-i4 product types not vectors");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcEmitC = convertVectorTypeToEmitC(srcVecType);
    if (!resultEmitC || !srcEmitC)
      return rewriter.notifyMatchFailure(
          packed, "packed-i4 product type not convertible");
    mlir::Value lhs = valueMap.lookup(packed.getLhs());
    mlir::Value rhs = valueMap.lookup(packed.getRhs());
    if (!lhs || !rhs)
      return rewriter.notifyMatchFailure(packed,
                                         "packed-i4 product operand unmapped");
    llvm::StringRef opName = packed.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = packed.getTCRVEmitCLowerableSourceRole();
    unsigned srcSEW = vectorElementWidth(srcVecType);
    llvm::StringRef srcLmul = srcVecType.getLmul();
    llvm::StringRef srcDtype = vectorDType(srcVecType);
    unsigned resSEW = vectorElementWidth(resultVecType);
    llvm::StringRef resLmul = resultVecType.getLmul();
    llvm::StringRef resDtype = vectorDType(resultVecType);
    mlir::Type u8Type = emitc::OpaqueType::get(rewriter.getContext(), "uint8_t");

    // Shift-by-immediate intrinsics spell as __riscv_<mnemonic>_<dtype><lmul>
    // (the `vx` form is part of the mnemonic), distinct from the `_vv_` form
    // riscvIntrinsicName builds for the binary product intrinsics.
    auto shiftCallee = [](llvm::StringRef mnemonic, llvm::StringRef dtype,
                          llvm::StringRef lmul) -> std::string {
      return ("__riscv_" + mnemonic + "_" + dtype + lmul).str();
    };
    auto shift = [&](llvm::StringRef mnemonic, llvm::StringRef dtype,
                     llvm::StringRef lmul, mlir::Type vecType, mlir::Value src,
                     llvm::StringRef amount) -> mlir::Value {
      std::string callee = shiftCallee(mnemonic, dtype, lmul);
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, u8Type, amount.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                       mlir::ValueRange{src, amt, bodyVL})
          .getResult(0);
    };
    (void)srcSEW;
    (void)resSEW;

    // Low nibble: vsll(4) both sources into the high nibble.
    mlir::Value lhsLow =
        shift("vsll_vx", srcDtype, srcLmul, srcEmitC, lhs, "4");
    mlir::Value rhsLow =
        shift("vsll_vx", srcDtype, srcLmul, srcEmitC, rhs, "4");
    // vwmul widening product of the shifted low nibbles.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, mulCallee));
    mlir::Value lowProduct =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         mulCallee,
                                         mlir::ValueRange{lhsLow, rhsLow, bodyVL})
            .getResult(0);
    // vsra(8) rescales the i16 product (sign-extends + undoes the 2x4 shift).
    mlir::Value product =
        shift("vsra_vx", resDtype, resLmul, resultEmitC, lowProduct, "8");
    // High nibble: vsra(4) sign-extends the high nibble of each source in place.
    mlir::Value lhsHigh =
        shift("vsra_vx", srcDtype, srcLmul, srcEmitC, lhs, "4");
    mlir::Value rhsHigh =
        shift("vsra_vx", srcDtype, srcLmul, srcEmitC, rhs, "4");
    // vwmacc adds the high-nibble widening product into the accumulator.
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, maccCallee));
    mlir::Value pairSum =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, maccCallee,
                mlir::ValueRange{product, lhsHigh, rhsHigh, bodyVL})
            .getResult(0);
    valueMap[packed.getResult()] = pairSum;
    return mlir::success();
  }

  /// packed_i4_offset_binary_x_i8_product(%weight,%act_low,%act_high,%vl) lowers
  /// to the ASYMMETRIC offset-binary one-sided decode + plain-i8 widening
  /// product/accumulate chain (the integer core of ggml Q4_0 x Q8_0). ONLY the
  /// packed-i4 weight is nibble-decoded; the two int8 activations stay plain:
  ///   // offset-binary -> two's-complement: ggml's (nibble - 8) equals the
  ///   // signed 4-bit value of (nibble XOR 0x8). XOR-ing both nibbles of every
  ///   // byte with 0x88 converts each offset-binary nibble into a
  ///   // two's-complement 4-bit lane, decoded by the existing sign-extend.
  ///   v8  w_xor = vxor_vx_i8mf4(weight, 0x88, vl);    // (cast to int literal)
  ///   // low nibble: shift the i4 into the high nibble (vsll 4), arithmetic
  ///   // shift back (vsra 4) to sign-extend it into a plain [-8,7] i8 lane.
  ///   v8  w_low_sh = vsll_vx_i8mf4(w_xor, 4, vl);
  ///   v8  v0       = vsra_vx_i8mf4(w_low_sh, 4, vl);   // decoded low nibble i8
  ///   // high nibble: arithmetic-shift sign-extends it in place.
  ///   v8  v1 = vsra_vx_i8mf4(w_xor, 4, vl);            // decoded high nibble i8
  ///   // asymmetric widening product: i8(decoded) x i8(plain activation) -> i16.
  ///   v16 product = vwmul_vv_i16mf2(v0, act_low, vl);  // low half  <-> q8[0..15]
  ///   v16 product = vwmacc_vv_i16mf2(product, v1, act_high, vl); // high half
  /// The plain int8 activations are NOT shifted and there is NO vsra(product,8)
  /// rescale -- v0/v1 are real signed i8 lanes, distinct from the symmetric
  /// packed-i4 path which shifts BOTH operands and rescales the product. The op
  /// carries the offset-binary one-sided unpack STRUCTURE typed; the conversion
  /// never reads operand_form / mirror strings to choose it.
  /// Emit the COMPLETE ggml ggml_vec_dot_q4_0_q8_0 block kernel for one
  /// tcrv_rvv.q4_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph):
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);   // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 16; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e<W>m1(16 - c);  // mf4: e32m1 / m1: e8m1
  ///       vint8<L>_t  w  = __riscv_vle8_v_i8<L>(xb + 2 + c, vl);
  ///       vint8<L>_t  y0 = __riscv_vle8_v_i8<L>(yb + 2 + c, vl);
  ///       vint8<L>_t  y1 = __riscv_vle8_v_i8<L>(yb + 2 + 16 + c, vl);
  ///       vint16<W>_t p  = <offset-binary decode/product>(w, y0, y1, vl);
  ///       vint32m1_t  seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t  red  = __riscv_vwredsum_vs_i16<W>_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + ((float)sumi * d_x) * d_y;  // ggml left-assoc order
  ///   }
  ///   *s = sumf;
  /// The strip loop anchors at the integer_core_lmul resource fact: "mf4" (the
  /// INC-2a default: i8mf4 -> i16mf2, vsetvl_e32m1, 4 chunks at VLEN=128) or "m1"
  /// (i8m1 -> i16m2, vsetvl_e8m1, ONE strip + ONE vwredsum per block at VLEN=128,
  /// matching ggml's hand-written reduction anchor). Both are byte-exact (same
  /// integer set; integer add is order-independent) and stay VLEN-robust (a VLEN
  /// < 128 board re-strips correctly via the sumi-carrying seed).
  /// The block format facts (QK/strides/offsets/scale model) are the op's typed
  /// attrs (I4 mirror); the emission is the op's fixed structure. The integer
  /// core reuses emitOffsetBinaryDecodeProductValue (the SAME nodes INC-1
  /// emits). The ONE sanctioned opaque piece is the scalar fp16->fp32 read,
  /// which goes through emitc.call_opaque exactly as INC-1 emits its intrinsics.
  mlir::LogicalResult emitQ4_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ40Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ40Q80Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-block reduction anchor). It is the
    // *how* (vector grouping / strip width), never the *what*: the dot product is
    // byte-exact either way (vwredsum sums the same integer set; integer add is
    // order-independent). The chosen i8 source LMUL ("mf4"/"m1") drives the
    // widened i16 product LMUL ("mf2"/"m2") and the i8 load/vsetvl spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    // The two outer-loop SHAPE knobs (also bounded resource/scheduling facts, the
    // *how* not the *what*; both are byte-exact -- the fp32 folds stay in strict
    // ascending block order, only the independent integer cores overlap). The
    // verifier bounds the factor to 1|2|4 and the elision to robust|elided, and
    // forbids "elided" unless the integer core anchors at m1.
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (the verifier bounds the
    // source to mf4|m1, so the product is mf2|m2 respectively).
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    // The i8<core> / i16<wide> / i32m1 vector types of the integer core (same
    // spelling convertVectorTypeToEmitC produces; the inner work is INC-1's
    // decode/product chain, re-anchored at the chosen LMUL).
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // Per-block address arithmetic for a block at outer-loop value ib plus a
    // compile-time block offset within the unroll group: const uint8_t *xb =
    // vx + (ib + blockOffset)*18; const uint8_t *yb = vy + (ib + blockOffset)*34.
    // For the no-unroll (factor 1) form blockOffset is 0 and the resulting nodes
    // (mul ib*18, add base, mul ib*34, add base) are byte-identical to the
    // committed single-block emission.
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The two scalar fp16->fp32 reads (the ONE sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // The offset-binary asymmetric i4xi8 decode/product + reduce for ONE strip,
    // seeded with the carried sumi, returning the next scalar sumi (INC-1 nodes
    // re-anchored at the chosen integer-core LMUL). vl is the active strip VL;
    // chunkOffset is the within-half-block byte offset (a value: the strip
    // induction var for the robust loop, or a 0 literal for the elided core).
    // When carrySumi is set the seed lane0 is loaded from sumiVar (the robust
    // strip carries the running sumi across strips); otherwise the seed lane0 is
    // a "0" literal (the elided core runs the strip exactly once). The seed load
    // is emitted at the seed position (right before vmv_v_x), preserving the
    // committed single-block node order byte-for-byte.
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      };
      std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadChunk = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         loadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      mlir::Value w = loadChunk(chunkPtr(xb, weightPtrType, quantOffset));
      mlir::Value y0 = loadChunk(chunkPtr(yb, activationPtrType, quantOffset));
      mlir::Value y1 =
          loadChunk(chunkPtr(yb, activationPtrType, quantOffset + highOffset));

      mlir::FailureOr<mlir::Value> product = emitOffsetBinaryDecodeProductValue(
          rewriter, loc, w, y0, y1, vl, i8CoreType, i16WideType, "i8", coreLmul,
          16, wideLmul, "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      // vwredsum is LMUL-independent in result: the i16<wide> source reduces
      // into i32m1 lane 0 (the same integer set; the callee tracks the source
      // LMUL).
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce, and returns the sumi lvalue (the caller folds it in order). When
    // forceRobust is set the inner strip loop is always kept (the nb % factor
    // tail and the factor-1/robust shapes); otherwise stripElided selects the
    // single-vsetvl(16) elided core (m1-only, VLEN >= 128).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      // int32_t sumi = 0;  (RESET each block -- the integer dot is per-block)
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) (caps the
        // active vl at 16 when VLMAX >= 16, covering the whole half-block) + ONE
        // strip reduce. NO inner strip loop, NO sumi carry (seed lane0 = 0; the
        // strip can only run once).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, sizeLit(0), vl, sumiVar, /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 16 weight bytes, the loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). The mf4 core uses vsetvl_e32m1 (4 strips at
      // VLEN=128, INC-2a byte-identical); the m1 core uses vsetvl_e8m1 (ONE
      // strip at VLEN=128). Both stay VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        // size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
    };

    // The left-associative fp32 accumulate sumf = sumf + ((float)sumi*d_x)*d_y
    // (ggml EXACT order), grouped into ONE emitc.expression so mlir-translate
    // renders it as a SINGLE C statement and the compiler fuses the SAME FMA
    // ggml does under -ffp-contract=on/default (byte-exact across all four
    // modes). The two emitc.load temps stay OUTSIDE the expression (load lacks
    // the CExpression trait). For multi-block unrolls the caller invokes this in
    // STRICT ascending block order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value timesDx =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block (addresses + scales + integer core + fold) at outer-loop
    // value ib plus a compile-time block offset within the unroll group. The
    // integer cores of several blocks in a group may interleave, but emitFold is
    // invoked in strict ascending blockOffset order by the caller, so the fp32
    // folds stay byte-exact.
    // The integer core of ONE block (addresses + scales + sumi), WITHOUT the
    // fold. The factor independent cores of an unroll group are emitted
    // back-to-back so the two/four reductions sit adjacent and overlap (the
    // latency-overlap lever); the folds are emitted afterwards in strict
    // ascending block order. dX/dY/sumi are returned so the caller can fold in
    // order. For factor 1 the caller emits the core then its fold immediately,
    // so the node order is unchanged from the committed single-block emission.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      mlir::Value dX = fp16Read(xb);
      mlir::Value dY = fp16Read(yb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY};
    };

    if (multiBlockFactor == 1) {
      // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the no-unroll form;
      // byte-identical to the committed single-block emission (factor absent or
      // 1, robust or elided per strip_elision). The core then its fold emit
      // back-to-back, preserving the committed node order.
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY);
    } else {
      // Multi-block unroll: a main loop stepping by factor over the
      // nb - nb%factor full groups -- emit ALL factor independent integer cores
      // FIRST (so the factor reductions are adjacent and the compiler overlaps
      // them: the latency-overlap lever), THEN the factor folds in strict
      // ascending block order (the fp non-associativity boundary) -- then a
      // robust single-block scalar tail over the nb % factor remainder.
      //
      // size_t nb_main = nb - nb % factor;  (the largest multiple of factor)
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                    factorLit,
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY);
      }
      // for (size_t ibt = nb_main; ibt < nb; ibt += 1) { robust single block }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_dot_q5_0_q8_0 block dot-product for one
  /// tcrv_rvv.q5_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-A 5-bit sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop /
  /// unroll / tail / scale-read / store / strip scaffolding STRUCTURE; the two
  /// kernel-specific differences are (a) the integer core decodes a 5-BIT weight
  /// (the q4_1 unsigned nibble unpack PLUS the per-element 5th high bit from the
  /// 32-bit qh field PLUS the offset-binary `-16` bias, via
  /// emitFiveBitOffsetBinaryDecodeProductValue) and (b) the fp32 fold is ggml's
  /// q5_0 order `sumf = sumf + (d_x*d_y)*sumi` (the dual scales multiplied FIRST,
  /// q8_0's order -- NOT the Q4_0 sibling's `((sumi*d_x)*d_y)`).
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*22;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);   // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     uint16_t qhLo = *(const uint16_t *)(xb + 2);  // 5th-bit field, ALIGNED
  ///     uint16_t qhHi = *(const uint16_t *)(xb + 4);  // (two halves, not u32)
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 16; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
  ///       vuint8<L>_t w  = __riscv_vle8_v_u8<L>(xb + 6 + c, vl);
  ///       vint8<L>_t  y0 = __riscv_vle8_v_i8<L>(yb + 2 + c, vl);
  ///       vint8<L>_t  y1 = __riscv_vle8_v_i8<L>(yb + 2 + 16 + c, vl);
  ///       vint16<W>_t p  = <5-bit offset-binary decode/product>(w, y0, y1,
  ///                          qh&0xFFFF, qh>>16, c, vl);
  ///       vint32m1_t  seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t  red  = __riscv_vwredsum_vs_i16<W>_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + (d_x * d_y) * (float)sumi;   // ggml q5_0 order
  ///   }
  ///   *s = sumf;
  /// The strip loop anchors at integer_core_lmul ("mf4" default / "m1") exactly as
  /// the Q4_0 sibling; the nibble half-block shape is byte-identical. The 5th-bit
  /// injection uses a `vid + c` per-lane shift so it is correct for any chunk
  /// offset c (the mf4 multi-strip path) and any VLEN. The block-format facts are
  /// the op's typed attrs (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ5_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ50Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ50Q80Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-block reduction anchor). The chosen
    // i8 source LMUL drives the widened i16 product LMUL and the i8/u16 spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (mf4->mf2, m1->m2). The
    // u16 bit-extraction runs at the SAME wide LMUL.
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    std::string u16WideTypeName = ("vuint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type u16WideType = emitc::OpaqueType::get(ctx, u16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();          // weight qs (6)
    int64_t actQuantOffset = blockDot.getActivationQuantByteOffset(); // q8 qs (2)
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t qhOffset = blockDot.getWeightQhByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";
    // The qh 5th-bit field is read as TWO aligned 16-bit halves (NOT one 32-bit
    // load): the q5_0 stride is 22 and qhOffset is 2, so xb+2 and xb+4 are
    // 2-aligned for any block, whereas a uint32_t load at xb+2 would be MISALIGNED
    // (UB / trap-emulate). Little-endian, the low/high halves are exactly the two
    // uint16_t at qhOffset and qhOffset+2, value-identical to (qh & 0xFFFF) /
    // (qh >> 16) -- so the decode is unchanged, just alignment-safe (matching
    // _generic's memcpy safety, the byte-exact value is the same).

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*22;
    // const uint8_t *yb = vy + (ib+blockOffset)*34.
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The scalar fp16->fp32 read (the sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };
    // The block_q5_0 per-element 5th-bit field, read as two ALIGNED 16-bit halves
    // (the qh bits for elements 0..15 at xb+qhOffset; for elements 16..31 at
    // xb+qhOffset+2). Each is a `(uint16_t)*(const uint16_t *)` scalar read (the
    // sanctioned opaque piece), 2-aligned and value-identical to the LE
    // (qh & 0xFFFF) / (qh >> 16) halves the decode broadcasts per strip.
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
    struct QhHalves {
      mlir::Value low16;
      mlir::Value high16;
    };
    auto qhRead = [&](mlir::Value xb) -> QhHalves {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qh_field"));
      auto readHalf = [&](int64_t byteOffset) -> mlir::Value {
        mlir::Value ptr = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, xb, sizeLit(byteOffset));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                         u16ReadCallee, mlir::ValueRange{ptr})
            .getResult(0);
      };
      mlir::Value low16 = readHalf(qhOffset);
      mlir::Value high16 = readHalf(qhOffset + 2);
      return QhHalves{low16, high16};
    };

    // The 5-bit offset-binary asymmetric decode/product + reduce for ONE strip,
    // seeded with the carried sumi, returning the next scalar sumi.
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value qhLow16, mlir::Value qhHigh16,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed, mlir::Type castType) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
      };
      // The weight nibble bytes load as UNSIGNED (the decode masks/shifts run on
      // the u8 lane); the q8 halves load as plain SIGNED i8.
      std::string wLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      std::string yLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadW = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, wLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                         wLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, yLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         yLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      // The q5_0 WEIGHT nibbles are at quantOffset (6); the q8_0 ACTIVATION int8
      // quants are at actQuantOffset (2, distinct -- the q8_0 block has no qh).
      mlir::Value w = loadW(chunkPtr(xb, weightPtrType, quantOffset, u8PtrType));
      mlir::Value y0 =
          loadY(chunkPtr(yb, activationPtrType, actQuantOffset, i8PtrType));
      mlir::Value y1 = loadY(chunkPtr(yb, activationPtrType,
                                      actQuantOffset + highOffset, i8PtrType));

      mlir::FailureOr<mlir::Value> product =
          emitFiveBitOffsetBinaryDecodeProductValue(
              rewriter, loc, w, y0, y1, qhLow16, qhHigh16, chunkOffset, vl,
              i8CoreType, u8CoreType, u16WideType, i16WideType, coreLmul,
              wideLmul, 16, wideLmul, "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce(s), and returns the sumi lvalue (the caller folds it in order).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb, mlir::Value qhLow16,
            mlir::Value qhHigh16,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) + ONE strip
        // reduce. NO inner strip loop, NO sumi carry (c = 0, seed lane0 = 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, qhLow16, qhHigh16, sizeLit(0), vl, sumiVar,
                            /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 16 weight bytes. The loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, qhLow16, qhHigh16, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
    };

    // The fp32 accumulate sumf = sumf + (d_x*d_y)*(float)sumi (ggml q5_0/q8_0
    // scales-first order), grouped into ONE emitc.expression so mlir-translate
    // renders it as a SINGLE C statement and the compiler fuses the SAME FMA
    // ggml does under -ffp-contract=on/default (byte-exact across all four
    // modes). The two emitc.load temps stay OUTSIDE the expression. For
    // multi-block unrolls the caller invokes this in STRICT ascending block
    // order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value scales =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scales, sumiFloat);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      mlir::Value dX = fp16Read(xb);
      mlir::Value dY = fp16Read(yb);
      QhHalves qh = qhRead(xb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, qh.low16, qh.high16, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY};
    };

    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY);
    } else {
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                    factorLit,
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_dot_q5_1_q8_1 block dot-product for one
  /// tcrv_rvv.q5_1_q8_1_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-B 5-bit sibling that COMBINES the two prior breadth increments,
  /// sharing the block-loop / unroll / tail / strip scaffolding STRUCTURE; the
  /// two kernel-specific facts are (a) the integer core decodes an UNSIGNED 5-bit
  /// weight in [0,31] -- the SAME nibble unpack + 5th-bit injection q5_0 uses
  /// (emitFiveBitOffsetBinaryDecodeProductValue) but with applyOffsetBias=false
  /// so the `-16` bias is dropped (the bias lives in q5_1's separate per-block
  /// MIN scale, exactly like q4_1), and (b) the fp32 fold reads FOUR per-block
  /// fp16 scales (d_x,m_x at weight+0/+2; d_y,s_y at activation+0/+2) and folds
  /// ggml's EXACT statement sumf += (d_x*d_y)*sumi + m_x*s_y (IDENTICAL to q4_1).
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*24;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*36;
  ///     float d_x = (float)*(const _Float16 *)(xb);    // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     float m_x = (float)*(const _Float16 *)(xb+2);
  ///     float s_y = (float)*(const _Float16 *)(yb+2);
  ///     uint16_t qhLo = *(const uint16_t *)(xb + 4);   // 5th-bit field, ALIGNED
  ///     uint16_t qhHi = *(const uint16_t *)(xb + 6);   // (two halves, not u32)
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 16; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e<W>m1(16 - c);
  ///       vuint8<L>_t w  = __riscv_vle8_v_u8<L>(xb + 8 + c, vl);
  ///       vint8<L>_t  y0 = __riscv_vle8_v_i8<L>(yb + 4 + c, vl);
  ///       vint8<L>_t  y1 = __riscv_vle8_v_i8<L>(yb + 4 + 16 + c, vl);
  ///       vint16<W>_t p  = <5-bit UNSIGNED decode/product>(w, y0, y1,
  ///                          qh&0xFFFF, qh>>16, c, vl);   // NO -16 bias
  ///       vint32m1_t  seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t  red  = __riscv_vwredsum_vs_i16<W>_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + ((d_x * d_y) * (float)sumi + m_x * s_y);  // ggml q5_1
  ///   }
  ///   *s = sumf;
  /// The block-format facts (QK/strides/offsets/scale model) are the op's typed
  /// attrs (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ5_1Q8_1BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ51Q81Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ51Q81Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-half-block anchor). The chosen i8
    // source LMUL drives the widened i16 product LMUL and the i8/u16 spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (mf4->mf2, m1->m2). The
    // u16 bit-extraction runs at the SAME wide LMUL.
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    std::string u16WideTypeName = ("vuint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type u16WideType = emitc::OpaqueType::get(ctx, u16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();          // weight qs (8)
    int64_t actQuantOffset = blockDot.getActivationQuantByteOffset(); // q8 qs (4)
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t qhOffset = blockDot.getWeightQhByteOffset();
    int64_t weightMinOffset = blockDot.getWeightMinByteOffset();
    int64_t activationSumOffset = blockDot.getActivationSumByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";
    // The qh 5th-bit field is read as TWO aligned 16-bit halves (NOT one 32-bit
    // load): the q5_1 stride is 24 and qhOffset is 4, so xb+4 and xb+6 are
    // 2-aligned for any block, whereas a uint32_t load at xb+4 is fine but the
    // two-u16 form matches the q5_0 emitter and stays alignment-safe. Little-
    // endian, the low/high halves are exactly (qh & 0xFFFF) / (qh >> 16).

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*24;
    // const uint8_t *yb = vy + (ib+blockOffset)*36.
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // A scalar fp16->fp32 read at a per-block byte offset (the sanctioned opaque
    // piece, a typed emitc.call_opaque node). q5_1 reads FOUR per block: d_x at
    // weight+0, m_x at weight+`weight_min_byte_offset`; d_y at activation+0, s_y
    // at activation+`activation_sum_byte_offset`.
    auto fp16ReadAt = [&](mlir::Value blockBase, mlir::Type ptrType,
                          int64_t byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, ptrType, blockBase,
                                             sizeLit(byteOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee, mlir::ValueRange{addr})
          .getResult(0);
    };
    // The block_q5_1 per-element 5th-bit field, read as two ALIGNED 16-bit halves
    // (the qh bits for elements 0..15 at xb+qhOffset; for elements 16..31 at
    // xb+qhOffset+2). Each is a `(uint16_t)*(const uint16_t *)` scalar read, the
    // sanctioned opaque piece, value-identical to the LE (qh & 0xFFFF) /
    // (qh >> 16) halves the decode broadcasts per strip.
    llvm::StringRef u16ReadCallee = "(uint16_t)*(const uint16_t *)";
    struct QhHalves {
      mlir::Value low16;
      mlir::Value high16;
    };
    auto qhRead = [&](mlir::Value xb) -> QhHalves {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qh_field"));
      auto readHalf = [&](int64_t byteOffset) -> mlir::Value {
        mlir::Value ptr = rewriter.create<emitc::AddOp>(
            loc, weightPtrType, xb, sizeLit(byteOffset));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u32Type},
                                         u16ReadCallee, mlir::ValueRange{ptr})
            .getResult(0);
      };
      mlir::Value low16 = readHalf(qhOffset);
      mlir::Value high16 = readHalf(qhOffset + 2);
      return QhHalves{low16, high16};
    };

    // The 5-bit UNSIGNED decode/product + reduce for ONE strip, seeded with the
    // carried sumi, returning the next scalar sumi. The decode SHARES q5_0's
    // emitter with applyOffsetBias=false so the `-16` is dropped (the q5_1 weight
    // is an unsigned q5 in [0,31]; the bias lives in the per-block MIN scale).
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value qhLow16, mlir::Value qhHigh16,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed, mlir::Type castType) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
      };
      // The weight nibble bytes load as UNSIGNED (the decode masks/shifts run on
      // the u8 lane); the q8 halves load as plain SIGNED i8.
      std::string wLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      std::string yLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadW = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, wLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                         wLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      auto loadY = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, yLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         yLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      // The q5_1 WEIGHT nibbles are at quantOffset (8); the q8_1 ACTIVATION int8
      // quants are at actQuantOffset (4, distinct -- the q8_1 block has no qh but
      // carries the precomputed sum s before its quants).
      mlir::Value w = loadW(chunkPtr(xb, weightPtrType, quantOffset, u8PtrType));
      mlir::Value y0 =
          loadY(chunkPtr(yb, activationPtrType, actQuantOffset, i8PtrType));
      mlir::Value y1 = loadY(chunkPtr(yb, activationPtrType,
                                      actQuantOffset + highOffset, i8PtrType));

      mlir::FailureOr<mlir::Value> product =
          emitFiveBitOffsetBinaryDecodeProductValue(
              rewriter, loc, w, y0, y1, qhLow16, qhHigh16, chunkOffset, vl,
              i8CoreType, u8CoreType, u16WideType, i16WideType, coreLmul,
              wideLmul, 16, wideLmul, "i16", opName, role,
              /*applyOffsetBias=*/false);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce(s), and returns the sumi lvalue (the caller folds it in order).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb, mlir::Value qhLow16,
            mlir::Value qhHigh16,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) + ONE strip
        // reduce. NO inner strip loop, NO sumi carry (c = 0, seed lane0 = 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, qhLow16, qhHigh16, sizeLit(0), vl, sumiVar,
                            /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 16 weight bytes. The loop
      // STEP is the loop-invariant VLMAX; the per-chunk active vl is recomputed
      // inside as vsetvl(16 - c). VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, qhLow16, qhHigh16, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
    };

    // The fp32 accumulate sumf = sumf + (d_x*d_y)*sumi + m_x*s_y (ggml q5_1 EXACT
    // statement, quants.c:394 -- IDENTICAL to q4_1's), grouped into ONE
    // emitc.expression so mlir-translate renders it as a SINGLE C statement
    // matching ggml's fp associativity / contraction byte-for-byte. The two
    // summed products (the scale term and the MIN term) are the Family-B
    // distinction from q5_0's single block term. The emitc.load temps stay
    // OUTSIDE the expression. For multi-block unrolls the caller invokes this in
    // STRICT ascending block order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY,
                        mlir::Value mX, mlir::Value sY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // (d_x * d_y) FIRST, then * sumi -- ggml q5_1's scale term order.
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value scaleTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scaleProduct,
                                          sumiFloat);
        // m_x * s_y -- the per-block MIN/SUM correction term.
        mlir::Value minTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, mX, sY);
        // ggml's EXACT tree (quants.c:394): `sumf += (d_x*d_y)*sumi + m_x*s_y;`.
        // The `+` between the two products binds BEFORE the `+=`, so the two
        // products are summed FIRST, then added to sumf: sumf + (scaleTerm +
        // minTerm). This grouping is fp-significant (`(sumf+A)+B != sumf+(A+B)`)
        // -- byte-exactness requires this exact tree.
        mlir::Value blockTerm =
            rewriter.create<emitc::AddOp>(loc, floatType, scaleTerm, minTerm);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block's integer core (addresses + four scales + qh + sumi),
    // WITHOUT the fold; the caller folds in strict ascending block order.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
      mlir::Value mX;
      mlir::Value sY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      // ggml reads d_x then d_y for the scale product, then m_x and s_y for the
      // MIN term; read them in that order for a stable node sequence.
      mlir::Value dX = fp16ReadAt(xb, weightPtrType, 0);
      mlir::Value dY = fp16ReadAt(yb, activationPtrType, 0);
      mlir::Value mX = fp16ReadAt(xb, weightPtrType, weightMinOffset);
      mlir::Value sY = fp16ReadAt(yb, activationPtrType, activationSumOffset);
      QhHalves qh = qhRead(xb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, qh.low16, qh.high16, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY, mX, sY};
    };

    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
    } else {
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                    factorLit,
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY, core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

  /// Emit the ggml Q4_0 x Q8_0 GEMM tile (weight-decode reuse) for one
  /// tcrv_rvv.q4_0_q8_0_gemm_tile op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// WEIGHT-DECODE-REUSE sibling of emitQ4_0Q8_0BlockDot: ONE weight row times M
  /// activation columns, decoding each q4_0 weight block ONCE and reusing the
  /// decoded v0/v1 nibble lanes across all M columns:
  ///   float sumf[M];  for (j) sumf[j] = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*18;            // weight block (shared)
  ///     float d_x = (float)*(const _Float16 *)(xb);
  ///     // HOISTED weight decode (m1 whole-half-block, VLEN >= 128):
  ///     size_t vl = __riscv_vsetvl_e8m1(16);
  ///     vint8m1_t w  = __riscv_vle8_v_i8m1(xb + 2, vl);
  ///     vint8m1_t v0, v1 = <offset-binary decode>(w, vl);   // reused M-fold
  ///     for (size_t j = 0; j < M; j += 1) {        // inner column loop
  ///       const uint8_t *yb = vy + j*by + ib*34;
  ///       float d_y = (float)*(const _Float16 *)(yb);
  ///       vint8m1_t y0 = __riscv_vle8_v_i8m1(yb + 2, vl);
  ///       vint8m1_t y1 = __riscv_vle8_v_i8m1(yb + 2 + 16, vl);
  ///       vint16m2_t p = <product from decoded>(v0, v1, y0, y1, vl);
  ///       vint32m1_t seed = __riscv_vmv_v_x_i32m1(0, 1);
  ///       vint32m1_t red  = __riscv_vwredsum_vs_i16m2_i32m1(p, seed, vl);
  ///       int32_t sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///       sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  // ggml order
  ///     }
  ///   }
  ///   for (j) s[j] = sumf[j];
  /// Each column j computes EXACTLY ggml_vec_dot_q4_0_q8_0(weight_row, column_j)
  /// (the weight decode is shared, but the per-column products / reduction / fold
  /// are identical, with M INDEPENDENT fp32 accumulators), so the M outputs are
  /// byte-exact vs M independent vec_dot calls. The weight decode reuses
  /// emitOffsetBinaryDecodeValue (the decode half INC-1 emits); the per-column
  /// product reuses emitOffsetBinaryProductFromDecodedValue (the product half).
  /// The m1 whole-half-block decode is correct at VLEN >= 128 (the board's
  /// mandated full-V floor; vsetvl_e8m1(16) caps the active vl at 16 when
  /// VLMAX >= 16) -- the same justification as the block-dot strip_elision form.
  mlir::LogicalResult emitQ4_0Q8_0GemmTile(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlGemmTileQ40Q80Op tile;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto t = llvm::dyn_cast<tcrvrvv::GgmlGemmTileQ40Q80Op>(op))
        tile = t;
    }
    if (!tile)
      return rewriter.notifyMatchFailure(scope, "gemm-tile body missing the op");

    mlir::Value weightBase = valueMap.lookup(tile.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(tile.getActivationBase());
    mlir::Value columnStride = valueMap.lookup(tile.getActivationColumnStride());
    mlir::Value output = valueMap.lookup(tile.getOutput());
    if (!weightBase || !activationBase || !columnStride || !output)
      return rewriter.notifyMatchFailure(tile, "gemm-tile ABI operand unmapped");

    llvm::StringRef opName = tile.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = tile.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = tile.getQk();
    int64_t weightStride = tile.getWeightBlockStride();
    int64_t activationStride = tile.getActivationBlockStride();
    int64_t quantOffset = tile.getQuantByteOffset();
    int64_t highOffset = tile.getActivationHighByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block
    int64_t cols = tile.getActivationCols();

    // The weight decode anchors at the m1 whole-half-block form (one
    // vsetvl_e8m1(16) covers the 16 nibble bytes at VLEN >= 128); the product
    // widens i8m1 -> i16m2. These are the *how* (vector grouping), never the
    // *what*: the dot product is byte-exact (vwredsum sums the same integer set).
    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf[M];  (the M INDEPENDENT fp32 column accumulators, an emitc
    // array). Zero each lane: for (size_t j = 0; j < M; ++j) sumf[j] = 0.0f;
    // -- emitted as M explicit assigns so the init is a simple structured node
    // sequence (M is a small bounded compile-time tile).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    mlir::Type sumfArrayType = emitc::ArrayType::get({cols}, floatType);
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, sumfArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sumfArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumfVar.getResult());
    auto sumfElem = [&](mlir::Value j) -> mlir::Value {
      return rewriter
          .create<emitc::SubscriptOp>(loc, sumfArray, mlir::ValueRange{j})
          .getResult();
    };
    for (int64_t j = 0; j < cols; ++j) {
      mlir::Value jIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(j));
      rewriter.create<emitc::AssignOp>(
          loc, sumfElem(jIdx),
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
    }

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // Per-block weight address arithmetic: const uint8_t *xb = vx + ib*18.
    auto blockBaseValue = [&](mlir::Value idx, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The scalar fp16->fp32 read (the ONE sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // A typed i8m1 chunk load: __riscv_vle8_v_i8m1(ptr + fixed, vl).
    std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    auto loadChunk = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                         mlir::Value vl) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                       loadCallee, mlir::ValueRange{cast, vl})
          .getResult(0);
    };

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the AoS weight-block
    // loop. The weight decode is hoisted to the TOP of this body (once per
    // block); the inner M-column loop reuses it.
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    mlir::LogicalResult blockStatus = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      // Shared weight block: address + fp16 scale + HOISTED decode.
      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "block_base_x");
      mlir::Value dX = fp16Read(xb);

      // size_t vl = __riscv_vsetvl_e8m1(16);  (m1 whole-half-block, VLEN>=128).
      std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, setvlCallee));
      mlir::Value vl =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);

      // vint8m1_t w = vle8(xb + 2, vl);  then the offset-binary decode into
      // v0/v1 -- the SHARED decoded nibble lanes reused across all M columns.
      mlir::Value w = loadChunk(xb, weightPtrType, quantOffset, vl);
      std::pair<mlir::Value, mlir::Value> decoded = emitOffsetBinaryDecodeValue(
          rewriter, loc, w, vl, i8CoreType, "i8", coreLmul, opName, role);

      // The inner M-column loop. Each column j: address (vy + j*by + ib*34),
      // its fp16 scale, its two q8 halves, the product against the HOISTED
      // v0/v1, the per-column reduce, and the ascending-block-order fp32 fold
      // into sumf[j]. M independent accumulators -> M byte-exact vec_dot results.
      auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                   sizeLit(cols), sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
      mlir::LogicalResult colStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard colGuard(rewriter);
        rewriter.setInsertionPointToStart(colLoop.getBody());
        mlir::Value j = colLoop.getInductionVar();

        // const uint8_t *yb = vy + j*by + ib*34;
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "column_base_y"));
        mlir::Value colOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, j, columnStride);
        mlir::Value ybCol = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, activationBase, colOff);
        mlir::Value yb = blockBaseValue(ib, ybCol, activationPtrType,
                                        activationStride, "block_base_y");
        mlir::Value dY = fp16Read(yb);

        // vint8m1_t y0 = vle8(yb + 2, vl);  vint8m1_t y1 = vle8(yb + 2 + 16, vl);
        mlir::Value y0 = loadChunk(yb, activationPtrType, quantOffset, vl);
        mlir::Value y1 =
            loadChunk(yb, activationPtrType, quantOffset + highOffset, vl);

        // The product half against the HOISTED decoded weight lanes (byte-
        // identical nodes to the per-row block dot's vwmul/vwmacc).
        mlir::FailureOr<mlir::Value> product =
            emitOffsetBinaryProductFromDecodedValue(
                rewriter, loc, decoded.first, decoded.second, y0, y1, vl,
                i16WideType, 16, wideLmul, "i16", opName, role);
        if (mlir::failed(product)) {
          colStatus = mlir::failure();
        } else {
          // Per-column reduce: seed lane0 = 0 (the m1 strip runs once at
          // VLEN >= 128), vwredsum, extract scalar sumi.
          std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, seedCallee));
          mlir::Value sumiSeed =
              rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
          mlir::Value seed =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i32m1Type}, seedCallee,
                      mlir::ValueRange{sumiSeed, sizeLit(1)})
                  .getResult(0);
          std::string reduceCallee =
              ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, reduceCallee));
          mlir::Value red =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                      mlir::ValueRange{*product, seed, vl})
                  .getResult(0);
          std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, extractCallee));
          mlir::Value sumi =
              rewriter
                  .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                               extractCallee,
                                               mlir::ValueRange{red})
                  .getResult(0);

          // sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  -- ggml's exact
          // left-associative order, grouped into ONE emitc.expression so
          // mlir-translate renders it as a SINGLE C statement the compiler
          // fuses into the SAME FMA ggml does under -ffp-contract=on/default
          // (byte-exact across all four modes). The emitc.load of sumf[j] stays
          // OUTSIDE the expression (load lacks the CExpression trait). Each
          // column accumulates ib ascending into its OWN sumf[j], so the fp32
          // non-associativity boundary is per-column-identical to vec_dot.
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "fp32_accumulate"));
          mlir::Value sumfElemLval = sumfElem(j);
          mlir::Value sumfCur =
              rewriter.create<emitc::LoadOp>(loc, floatType, sumfElemLval)
                  .getResult();
          auto accumExpr = rewriter.create<emitc::ExpressionOp>(
              loc, floatType, /*do_not_inline=*/false);
          {
            mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
            mlir::Block *exprBlock =
                rewriter.createBlock(&accumExpr.getRegion());
            rewriter.setInsertionPointToStart(exprBlock);
            mlir::Value sumiFloat =
                rewriter.create<emitc::CastOp>(loc, floatType, sumi)
                    .getResult();
            mlir::Value timesDx =
                rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
            mlir::Value blockTerm =
                rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
            mlir::Value sumfNext =
                rewriter.create<emitc::AddOp>(loc, floatType, sumfCur,
                                              blockTerm);
            rewriter.create<emitc::YieldOp>(loc, sumfNext);
          }
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumf", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumfElemLval,
                                           accumExpr.getResult());
        }
      }
      if (mlir::failed(colStatus))
        blockStatus = mlir::failure();
    }
    if (mlir::failed(blockStatus))
      return mlir::failure();

    // for (j) s[j] = sumf[j];  -- the M-output store through the float * pointer
    // (M explicit structured assigns; s[0..M-1] contiguous, the bs ABI stride
    // is G2).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(tile, "gemm-tile output not a pointer");
    mlir::Value lastStored;
    for (int64_t j = 0; j < cols; ++j) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_s"));
      mlir::Value jIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(j));
      emitc::SubscriptOp outSubscript =
          rewriter.create<emitc::SubscriptOp>(loc, outPointer, jIdx);
      mlir::Value sumfVal =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfElem(jIdx))
              .getResult();
      rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfVal);
      lastStored = sumfVal;
    }

    valueMap[tile.getResult()] = lastStored;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml Q4_0 x Q8_0 FULL GEMM (NR weight rows x nc
  /// activation columns) for one tcrv_rvv.q4_0_q8_0_gemm op as fully STRUCTURED
  /// emitc nodes (I5; ZERO raw() strings -- every value is a node in the IR
  /// graph). It is the FULL-MATMUL wrapper of emitQ4_0Q8_0GemmTile: an outer
  /// emitc.for weight-ROW loop over nr, and an inner emitc.for column-strip loop
  /// over the nc columns in M-wide strips, each strip running G1's
  /// weight-decode-reuse tile body (decode each (row, block) weight ONCE, reuse
  /// the decoded v0/v1 nibble lanes across the M columns of the strip). The
  /// shape is:
  ///   for (size_t ir = 0; ir < nr; ++ir) {
  ///     const uint8_t *xr = vx + ir*bx;      // weight row ir
  ///     float *sr = s + ir*bs;               // output row ir
  ///     for (size_t cb = 0; cb < nc; cb += M) {
  ///       size_t active = (nc - cb < M) ? (nc - cb) : M;  // tail clamp
  ///       float sumf[M]; for (j<active) sumf[j] = 0;
  ///       const uint8_t *yb0 = vy + cb*by;    // first column of the strip
  ///       for (size_t ib = 0; ib < n/QK; ++ib) {
  ///         const uint8_t *xb = xr + ib*18;
  ///         float d_x = (float)*(_Float16*)xb;
  ///         size_t vl = vsetvl_e8m1(16);
  ///         vint8m1_t w = vle8(xb+2, vl);  decode -> v0,v1;   // HOISTED once
  ///         for (size_t j = 0; j < active; ++j) {
  ///           const uint8_t *yb = yb0 + j*by + ib*34;
  ///           float d_y = (float)*(_Float16*)yb;
  ///           y0=vle8(yb+2); y1=vle8(yb+2+16);
  ///           prod = vwmul/vwmacc(v0,y0,v1,y1);
  ///           sumi = vmv_x_s(vwredsum(prod));
  ///           sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  // ggml order
  ///         }
  ///       }
  ///       for (size_t j = 0; j < active; ++j) sr[cb + j] = sumf[j];
  ///     }
  ///   }
  /// Each (ir, cb+j) output is EXACTLY ggml_vec_dot_q4_0_q8_0(weight_row_ir,
  /// column_(cb+j)) -- the weight decode is shared across the strip, but the
  /// per-(block, column) products / reduction / ascending-block-order fp32 fold
  /// are identical to vec_dot, with one INDEPENDENT fp32 accumulator per column,
  /// so every output is byte-exact vs the per-(row,col) vec_dot ggml runs at
  /// VLEN=128 (the repack GEMM disabled at `case 128`). The weight decode reuses
  /// emitOffsetBinaryDecodeValue + emitOffsetBinaryProductFromDecodedValue
  /// (INC-1's decode/product halves) -- the SAME nodes the per-row block dot and
  /// the GEMM tile emit. The m1 whole-half-block decode is correct at VLEN >= 128
  /// (the board's mandated full-V floor).
  mlir::LogicalResult emitQ4_0Q8_0Gemm(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlGemmQ40Q80Op gemm;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto g = llvm::dyn_cast<tcrvrvv::GgmlGemmQ40Q80Op>(op))
        gemm = g;
    }
    if (!gemm)
      return rewriter.notifyMatchFailure(scope, "gemm body missing the op");

    mlir::Value weightBase = valueMap.lookup(gemm.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(gemm.getActivationBase());
    mlir::Value columnStride = valueMap.lookup(gemm.getActivationColumnStride());
    mlir::Value output = valueMap.lookup(gemm.getOutput());
    mlir::Value rowCount = valueMap.lookup(gemm.getRowCount());
    mlir::Value columnCount = valueMap.lookup(gemm.getColumnCount());
    mlir::Value weightRowStride = valueMap.lookup(gemm.getWeightRowStride());
    mlir::Value outputRowStride = valueMap.lookup(gemm.getOutputRowStride());
    if (!weightBase || !activationBase || !columnStride || !output ||
        !rowCount || !columnCount || !weightRowStride || !outputRowStride)
      return rewriter.notifyMatchFailure(gemm, "gemm ABI operand unmapped");

    llvm::StringRef opName = gemm.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = gemm.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();
    mlir::Type floatPtrType = output.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = gemm.getQk();
    int64_t weightStride = gemm.getWeightBlockStride();
    int64_t activationStride = gemm.getActivationBlockStride();
    int64_t quantOffset = gemm.getQuantByteOffset();
    int64_t highOffset = gemm.getActivationHighByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block
    int64_t cols = gemm.getActivationCols();

    // The weight decode anchors at the m1 whole-half-block form (one
    // vsetvl_e8m1(16) covers the 16 nibble bytes at VLEN >= 128); the product
    // widens i8m1 -> i16m2. The *how* (vector grouping), never the *what*: the
    // dot product is byte-exact (vwredsum sums the same integer set).
    llvm::StringRef coreLmul = "m1";
    llvm::StringRef wideLmul = "m2";
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, "vint8m1_t");
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, "vint16m2_t");
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // The scalar fp16->fp32 read (the ONE sanctioned opaque piece, a typed
    // emitc.call_opaque node, exactly how INC-1 emits its intrinsics).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // A typed i8m1 chunk load: __riscv_vle8_v_i8m1(ptr + fixed, vl).
    std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
    auto loadChunk = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                         mlir::Value vl) -> mlir::Value {
      mlir::Value full =
          rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      mlir::Value cast =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                       loadCallee, mlir::ValueRange{cast, vl})
          .getResult(0);
    };

    // size_t nb = n / QK;  (the contraction block count, shared across rows/cols)
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    if (!llvm::isa<mlir::TypedValue<emitc::PointerType>>(output))
      return rewriter.notifyMatchFailure(gemm, "gemm output not a pointer");

    // ===== Outer weight-ROW loop: for (size_t ir = 0; ir < nr; ++ir) =====
    auto rowLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), rowCount,
                                                 sizeLit(1),
                                                 /*bodyBuilder=*/nullptr);
    mlir::LogicalResult rowStatus = mlir::success();
    {
      mlir::OpBuilder::InsertionGuard rowGuard(rewriter);
      rewriter.setInsertionPointToStart(rowLoop.getBody());
      mlir::Value ir = rowLoop.getInductionVar();

      // const uint8_t *xr = vx + ir*bx;   float *sr = s + ir*bs;
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "weight_row_base"));
      mlir::Value rowWeightOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ir, weightRowStride);
      mlir::Value xr = rewriter.create<emitc::AddOp>(loc, weightPtrType,
                                                     weightBase, rowWeightOff);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "output_row_base"));
      mlir::Value rowOutOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ir, outputRowStride);
      mlir::Value sr = rewriter.create<emitc::AddOp>(loc, floatPtrType, output,
                                                     rowOutOff);

      auto srPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(sr);

      // emitStrip(cb, colBound): one M-wide column strip starting at column cb,
      // processing `colBound` columns (colBound is the inner-loop trip count).
      // The FULL strips pass the COMPILE-TIME CONSTANT M (sizeLit(cols)) so the
      // inner column loop is a constant-trip loop the C compiler fully unrolls
      // -- recovering G1's tile shape (the per-column vwredsum/vmv_x_s/fp32-fold
      // chains overlap across the unrolled columns; a runtime bound serializes
      // them and ships a regression). The ONE tail strip passes the runtime
      // remainder (nc % M). The math is identical for either bound, so every
      // output stays byte-exact vs per-(row,col) vec_dot; only the inner
      // trip-count SHAPE differs (the same "full groups + tail" pattern the
      // block dot's multi_block_factor uses).
      auto emitStrip = [&](mlir::Value cb,
                           mlir::Value colBound) -> mlir::LogicalResult {
        // const uint8_t *yb0 = vy + cb*by;  -- the first activation column of
        // this strip; column j of the strip is yb0 + j*by.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "strip_base_y"));
        mlir::Value stripColOff =
            rewriter.create<emitc::MulOp>(loc, sizeType, cb, columnStride);
        mlir::Value yb0 = rewriter.create<emitc::AddOp>(
            loc, activationPtrType, activationBase, stripColOff);

        // float sumf[M]; for (j < colBound) sumf[j] = 0.0f;  -- the M-wide fp32
        // accumulator array (M bounded). The init is over colBound (a partial
        // tail strip leaves the unused lanes untouched).
        rewriter.create<emitc::VerbatimOp>(
            loc, localVariableComment("sumf", opName, role));
        mlir::Type sumfArrayType = emitc::ArrayType::get({cols}, floatType);
        auto sumfVar = rewriter.create<emitc::VariableOp>(
            loc, sumfArrayType, emitc::OpaqueAttr::get(ctx, ""));
        auto sumfArray =
            llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sumfVar.getResult());
        auto sumfElem = [&](mlir::Value j) -> mlir::Value {
          return rewriter
              .create<emitc::SubscriptOp>(loc, sumfArray, mlir::ValueRange{j})
              .getResult();
        };
        {
          auto zeroLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0),
                                                        colBound, sizeLit(1),
                                                        /*bodyBuilder=*/nullptr);
          mlir::OpBuilder::InsertionGuard zg(rewriter);
          rewriter.setInsertionPointToStart(zeroLoop.getBody());
          rewriter.create<emitc::AssignOp>(
              loc, sumfElem(zeroLoop.getInductionVar()),
              rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
        }

        // ===== The weight-decode-reuse block loop (G1's tile body) =====
        // for (size_t ib = 0; ib < nb; ++ib) { decode weight ONCE; for j<colBound }
        auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        mlir::LogicalResult blockStatus = mlir::success();
        {
          mlir::OpBuilder::InsertionGuard blockGuard(rewriter);
          rewriter.setInsertionPointToStart(blockLoop.getBody());
          mlir::Value ib = blockLoop.getInductionVar();

          // Shared weight block: address (xr + ib*18) + fp16 scale + HOISTED
          // decode into v0/v1, computed ONCE and reused across the strip cols.
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, "block_base_x"));
          mlir::Value xbOff =
              rewriter.create<emitc::MulOp>(loc, sizeType, ib,
                                            sizeLit(weightStride));
          mlir::Value xb = rewriter.create<emitc::AddOp>(loc, weightPtrType, xr,
                                                         xbOff);
          mlir::Value dX = fp16Read(xb);

          // size_t vl = __riscv_vsetvl_e8m1(16);  (m1 whole-half-block, VLEN>=128)
          std::string setvlCallee = riscvIntrinsicName("vsetvl", 8, "m1", "");
          rewriter.create<emitc::VerbatimOp>(
              loc, stepComment(opName, role, setvlCallee));
          mlir::Value vl =
              rewriter
                  .create<emitc::CallOpaqueOp>(
                      loc, mlir::TypeRange{sizeType}, setvlCallee,
                      mlir::ValueRange{sizeLit(halfBlock)})
                  .getResult(0);

          // vint8m1_t w = vle8(xb + 2, vl);  then offset-binary decode -> v0/v1
          // (the SHARED decoded nibble lanes reused across all strip columns).
          mlir::Value w = loadChunk(xb, weightPtrType, quantOffset, vl);
          std::pair<mlir::Value, mlir::Value> decoded =
              emitOffsetBinaryDecodeValue(rewriter, loc, w, vl, i8CoreType,
                                          "i8", coreLmul, opName, role);

          // ===== The inner COLUMN loop (reuses the hoisted v0/v1) =====
          // for (size_t j = 0; j < colBound; ++j) { ... }  -- colBound is the
          // constant M for full strips (unrollable) or the runtime tail.
          auto colLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), colBound,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
          mlir::LogicalResult colStatus = mlir::success();
          {
            mlir::OpBuilder::InsertionGuard colGuard(rewriter);
            rewriter.setInsertionPointToStart(colLoop.getBody());
            mlir::Value j = colLoop.getInductionVar();

            // const uint8_t *yb = yb0 + j*by + ib*34;
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "column_base_y"));
            mlir::Value colOff =
                rewriter.create<emitc::MulOp>(loc, sizeType, j, columnStride);
            mlir::Value ybCol = rewriter.create<emitc::AddOp>(
                loc, activationPtrType, yb0, colOff);
            rewriter.create<emitc::VerbatimOp>(
                loc, stepComment(opName, role, "block_base_y"));
            mlir::Value ybBlockOff =
                rewriter.create<emitc::MulOp>(loc, sizeType, ib,
                                              sizeLit(activationStride));
            mlir::Value yb = rewriter.create<emitc::AddOp>(
                loc, activationPtrType, ybCol, ybBlockOff);
            mlir::Value dY = fp16Read(yb);

            // vint8m1_t y0 = vle8(yb+2, vl);  y1 = vle8(yb+2+16, vl);
            mlir::Value y0 = loadChunk(yb, activationPtrType, quantOffset, vl);
            mlir::Value y1 = loadChunk(yb, activationPtrType,
                                       quantOffset + highOffset, vl);

            // The product half against the HOISTED decoded weight lanes (byte-
            // identical nodes to the per-row block dot's vwmul/vwmacc).
            mlir::FailureOr<mlir::Value> product =
                emitOffsetBinaryProductFromDecodedValue(
                    rewriter, loc, decoded.first, decoded.second, y0, y1, vl,
                    i16WideType, 16, wideLmul, "i16", opName, role);
            if (mlir::failed(product)) {
              colStatus = mlir::failure();
            } else {
              // Per-column reduce: seed lane0 = 0 (the m1 strip runs once at
              // VLEN >= 128), vwredsum, extract scalar sumi.
              std::string seedCallee =
                  riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, seedCallee));
              mlir::Value sumiSeed =
                  rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                      .getResult();
              mlir::Value seed =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{i32m1Type}, seedCallee,
                          mlir::ValueRange{sumiSeed, sizeLit(1)})
                      .getResult(0);
              std::string reduceCallee =
                  ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, reduceCallee));
              mlir::Value red =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{i32m1Type}, reduceCallee,
                          mlir::ValueRange{*product, seed, vl})
                      .getResult(0);
              std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, extractCallee));
              mlir::Value sumi =
                  rewriter
                      .create<emitc::CallOpaqueOp>(
                          loc, mlir::TypeRange{i32Type}, extractCallee,
                          mlir::ValueRange{red})
                      .getResult(0);

              // sumf[j] = sumf[j] + ((float)sumi * d_x) * d_y;  -- ggml's exact
              // left-associative order, grouped into ONE emitc.expression so
              // mlir-translate renders it as a SINGLE C statement the compiler
              // fuses into the SAME FMA ggml does under -ffp-contract=on/default
              // (byte-exact across all four modes). Each column accumulates ib
              // ascending into its OWN sumf[j], so the fp32 non-associativity
              // boundary is per-column-identical to vec_dot.
              rewriter.create<emitc::VerbatimOp>(
                  loc, stepComment(opName, role, "fp32_accumulate"));
              mlir::Value sumfElemLval = sumfElem(j);
              mlir::Value sumfCur =
                  rewriter.create<emitc::LoadOp>(loc, floatType, sumfElemLval)
                      .getResult();
              auto accumExpr = rewriter.create<emitc::ExpressionOp>(
                  loc, floatType, /*do_not_inline=*/false);
              {
                mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
                mlir::Block *exprBlock =
                    rewriter.createBlock(&accumExpr.getRegion());
                rewriter.setInsertionPointToStart(exprBlock);
                mlir::Value sumiFloat =
                    rewriter.create<emitc::CastOp>(loc, floatType, sumi)
                        .getResult();
                mlir::Value timesDx =
                    rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat, dX);
                mlir::Value blockTerm =
                    rewriter.create<emitc::MulOp>(loc, floatType, timesDx, dY);
                mlir::Value sumfNext = rewriter.create<emitc::AddOp>(
                    loc, floatType, sumfCur, blockTerm);
                rewriter.create<emitc::YieldOp>(loc, sumfNext);
              }
              rewriter.create<emitc::VerbatimOp>(
                  loc, assignComment("sumf", opName, role));
              rewriter.create<emitc::AssignOp>(loc, sumfElemLval,
                                               accumExpr.getResult());
            }
          }
          if (mlir::failed(colStatus))
            blockStatus = mlir::failure();
        }
        if (mlir::failed(blockStatus))
          return mlir::failure();

        // for (size_t j = 0; j < colBound; ++j) sr[cb + j] = sumf[j];  -- the
        // strip output store through the row pointer.
        auto storeLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), colBound,
                                                       sizeLit(1),
                                                       /*bodyBuilder=*/nullptr);
        mlir::OpBuilder::InsertionGuard sg(rewriter);
        rewriter.setInsertionPointToStart(storeLoop.getBody());
        mlir::Value j = storeLoop.getInductionVar();
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "store_s"));
        mlir::Value outIdx =
            rewriter.create<emitc::AddOp>(loc, sizeType, cb, j);
        emitc::SubscriptOp outSubscript =
            rewriter.create<emitc::SubscriptOp>(loc, srPtr, outIdx);
        mlir::Value sumfVal =
            rewriter.create<emitc::LoadOp>(loc, floatType, sumfElem(j))
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(),
                                         sumfVal);
        return mlir::success();
      };

      // size_t ncFull = (nc / M) * M;  -- the full-strip span (a multiple of M).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "full_strip_span"));
      mlir::Value ncGroups =
          rewriter.create<emitc::DivOp>(loc, sizeType, columnCount,
                                        sizeLit(cols));
      mlir::Value ncFull =
          rewriter.create<emitc::MulOp>(loc, sizeType, ncGroups, sizeLit(cols));

      // ===== FULL column-strip loop: for (cb = 0; cb < ncFull; cb += M) =====
      // The inner column loop trip count is the COMPILE-TIME CONSTANT M, so the
      // C compiler fully unrolls it (recovering G1's overlapping reductions).
      auto stripLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), ncFull,
                                                     sizeLit(cols),
                                                     /*bodyBuilder=*/nullptr);
      mlir::LogicalResult stripStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard stripGuard(rewriter);
        rewriter.setInsertionPointToStart(stripLoop.getBody());
        stripStatus = emitStrip(stripLoop.getInductionVar(), sizeLit(cols));
      }
      if (mlir::failed(stripStatus))
        rowStatus = mlir::failure();

      // ===== ONE tail strip: if (ncFull < nc) emitStrip(ncFull, nc-ncFull) =====
      // The final nc % M columns (0 when nc is a multiple of M). The tail's
      // inner column loop is the runtime remainder; it runs at most once per
      // row, so the (un-unrolled) tail does not dominate the cost.
      if (mlir::succeeded(rowStatus)) {
        mlir::Value tailCount =
            rewriter.create<emitc::SubOp>(loc, sizeType, columnCount, ncFull);
        mlir::Value hasTail =
            rewriter
                .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, ncFull,
                                      columnCount)
                .getResult();
        auto tailIf = rewriter.create<emitc::IfOp>(loc, hasTail,
                                                   /*addThenBlock=*/true,
                                                   /*addElseBlock=*/false);
        {
          mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
          rewriter.setInsertionPointToStart(&tailIf.getThenRegion().front());
          mlir::LogicalResult tailStatus = emitStrip(ncFull, tailCount);
          rewriter.create<emitc::YieldOp>(loc);
          if (mlir::failed(tailStatus))
            rowStatus = mlir::failure();
        }
      }
    }
    if (mlir::failed(rowStatus))
      return mlir::failure();

    // The op result is the typed i32m1 token; the GEMM writes through *s so the
    // token has no consumer. Seed it with a zero i32m1 lane-0 so the valueMap
    // entry is well-formed (mirrors the per-row block dot's unused result).
    std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
    mlir::Value zeroLane =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0").getResult();
    mlir::Value resultTok =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                         seedCallee,
                                         mlir::ValueRange{zeroLane, sizeLit(1)})
            .getResult(0);
    valueMap[gemm.getResult()] = resultTok;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_dot_q4_1_q8_1 block kernel for one
  /// tcrv_rvv.q4_1_q8_1_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-B sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop / unroll /
  /// tail / strip-elision scaffolding, differing in exactly two kernel-specific
  /// parts (the q4_1 quantization is scale+MIN, asymmetric):
  ///   (a) the integer core decodes UNSIGNED nibbles [0,15]
  ///       (emitUnsignedNibbleDecodeProductValue: vand 0x0F / vsrl 0x04 on the u8
  ///       weight lane -> reinterpret -> the SAME signed vwmul/vwmacc against the
  ///       plain q8 halves), NOT the offset-binary `-8` chain. The weight chunk is
  ///       therefore loaded as u8 (the activations stay i8);
  ///   (b) the fold reads FOUR per-block fp16 scales (d_x,m_x at weight+0/+2;
  ///       d_y,s_y at activation+0/+2) and folds ggml's EXACT statement
  ///       sumf += (d_x*d_y)*sumi + m_x*s_y -- the (d_x*d_y) scale product times
  ///       sumi and the m_x*s_y MIN/SUM correction are SUMMED FIRST (the `+`
  ///       binds before the `+=`), then added to sumf: sumf + (A + B), an
  ///       fp-significant grouping required for byte-exactness.
  /// The block-format facts (QK/strides/offsets/scale model) are the op's typed
  /// attrs (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ4_1Q8_1BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ41Q81Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ41Q81Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (mf4 default,
    // or m1 to match ggml's one-vwredsum-per-half-block anchor). q4_1's nibble
    // half-block matches q4_0's, so the anchor set is q4_0's {mf4, m1}. It is the
    // *how* (vector grouping / strip width), never the *what*: the dot product is
    // byte-exact either way. The chosen i8 source LMUL drives the widened i16
    // product LMUL and the load/vsetvl spelling.
    llvm::StringRef coreLmul = "mf4";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (mf4->mf2, m1->m2).
    llvm::StringRef wideLmul = (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string u8CoreTypeName = ("vuint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    // The UNSIGNED weight lane the unsigned-nibble decode runs on (q4_1-only;
    // the activations stay i8).
    mlir::Type u8CoreType = emitc::OpaqueType::get(ctx, u8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();
    int64_t highOffset = blockDot.getActivationHighByteOffset();
    int64_t weightMinOffset = blockDot.getWeightMinByteOffset();
    int64_t activationSumOffset = blockDot.getActivationSumByteOffset();
    int64_t halfBlock = qk / 2; // 16 nibble bytes / q8 half lanes per block

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*20;
    // const uint8_t *yb = vy + (ib+blockOffset)*36.
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // A scalar fp16->fp32 read at a per-block byte offset (the ONE sanctioned
    // opaque piece, a typed emitc.call_opaque node). q4_1 reads FOUR per block:
    // d_x at weight+0, m_x at weight+`weight_min_byte_offset`; d_y at
    // activation+0, s_y at activation+`activation_sum_byte_offset`.
    auto fp16ReadAt = [&](mlir::Value blockBase, mlir::Type ptrType,
                          int64_t byteOffset) -> mlir::Value {
      mlir::Value addr = blockBase;
      if (byteOffset != 0)
        addr = rewriter.create<emitc::AddOp>(loc, ptrType, blockBase,
                                             sizeLit(byteOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{addr})
          .getResult(0);
    };

    // The UNSIGNED-nibble asymmetric i4xi8 decode/product + reduce for ONE strip,
    // seeded with the carried sumi, returning the next scalar sumi. The weight
    // chunk is loaded as UNSIGNED (vle8_v_u8<L>) so the vand/vsrl logical decode
    // is correct; the activations stay i8. vl is the active strip VL; chunkOffset
    // is the within-half-block byte offset (the strip induction var for the
    // robust loop, or a 0 literal for the elided core).
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          mlir::Type castPtrType, int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, castPtrType, full)
            .getResult();
      };
      // The weight load is UNSIGNED (vle8_v_u8<L>); the q8 activations are i8.
      std::string weightLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "u8");
      std::string actLoadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadU8 = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, weightLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8CoreType},
                                         weightLoadCallee,
                                         mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      auto loadI8 = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, actLoadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         actLoadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      mlir::Value w =
          loadU8(chunkPtr(xb, weightPtrType, u8PtrType, quantOffset));
      mlir::Value y0 =
          loadI8(chunkPtr(yb, activationPtrType, i8PtrType, quantOffset));
      mlir::Value y1 = loadI8(
          chunkPtr(yb, activationPtrType, i8PtrType, quantOffset + highOffset));

      mlir::FailureOr<mlir::Value> product =
          emitUnsignedNibbleDecodeProductValue(rewriter, loc, w, y0, y1, vl,
                                               i8CoreType, u8CoreType,
                                               i16WideType, coreLmul, 16,
                                               wideLmul, "i16", opName, role);
      if (mlir::failed(product))
        return mlir::failure();

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{*product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce, returns the sumi lvalue. When forceRobust is set the inner strip
    // loop is always kept; otherwise stripElided selects the single-vsetvl(16)
    // elided core (m1-only, VLEN >= 128).
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      unsigned setvlSEW = (coreLmul == "m1") ? 8 : 32;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, "m1", "");

      if (!forceRobust && stripElided) {
        // Elided core (m1-only, VLEN >= 128): ONE vsetvl_e8m1(16) + ONE strip
        // reduce. NO inner strip loop, NO sumi carry.
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(halfBlock)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, sizeLit(0), vl, sumiVar, /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 16 weight bytes; the loop STEP
      // is the loop-invariant VLMAX, the per-chunk active vl is vsetvl(16 - c).
      // Stays VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(halfBlock)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(halfBlock), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(halfBlock), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
    };

    // The fp32 accumulate sumf = sumf + (d_x*d_y)*sumi + m_x*s_y (ggml q4_1 EXACT
    // statement, quants.c:319), grouped into ONE emitc.expression so
    // mlir-translate renders it as a SINGLE C statement matching ggml's fp
    // associativity / contraction byte-for-byte. The two summed products
    // (the scale term and the MIN term) are the Family-B distinction from q4_0's
    // single block term. The emitc.load temps stay OUTSIDE the expression (load
    // lacks the CExpression trait). For multi-block unrolls the caller invokes
    // this in STRICT ascending block order, preserving fp non-associativity.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY,
                        mlir::Value mX, mlir::Value sY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // (d_x * d_y) FIRST, then * sumi -- ggml q4_1's scale term order.
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value scaleTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, scaleProduct,
                                          sumiFloat);
        // m_x * s_y -- the per-block MIN/SUM correction term.
        mlir::Value minTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, mX, sY);
        // ggml's EXACT tree (quants.c:319): `sumf += (d_x*d_y)*sumi + m_x*s_y;`.
        // The `+` between the two products binds BEFORE the `+=`, so the two
        // products are summed FIRST, then added to sumf: sumf + (scaleTerm +
        // minTerm). This grouping is fp-significant (`(sumf+A)+B != sumf+(A+B)`)
        // -- byte-exactness requires this exact tree.
        mlir::Value blockTerm =
            rewriter.create<emitc::AddOp>(loc, floatType, scaleTerm, minTerm);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block's integer core (addresses + four scales + sumi), WITHOUT
    // the fold; the caller folds in strict ascending block order.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
      mlir::Value mX;
      mlir::Value sY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      // ggml reads d_x then d_y for the scale product, then m_x and s_y for the
      // MIN term; read them in that order for a stable node sequence.
      mlir::Value dX = fp16ReadAt(xb, weightPtrType, 0);
      mlir::Value dY = fp16ReadAt(yb, activationPtrType, 0);
      mlir::Value mX = fp16ReadAt(xb, weightPtrType, weightMinOffset);
      mlir::Value sY = fp16ReadAt(yb, activationPtrType, activationSumOffset);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY, mX, sY};
    };

    if (multiBlockFactor == 1) {
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
    } else {
      // Multi-block unroll: emit ALL factor independent integer cores FIRST (so
      // they overlap), THEN the factor folds in strict ascending block order,
      // then a robust single-block scalar tail over the nb % factor remainder.
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                    factorLit,
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY, core.mX, core.sY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY, core->mX, core->sY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_dot_q8_0_q8_0 block kernel for one
  /// tcrv_rvv.q8_0_q8_0_block_dot op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph). It is the
  /// FAMILY-A sibling of emitQ4_0Q8_0BlockDot, sharing its block-loop / unroll /
  /// tail / scale-read / store scaffolding STRUCTURE; the ONLY kernel-specific
  /// differences are (a) the integer core is a plain 32-element signed widening
  /// product (vle8 x2 -> vwmul_vv i8->i16 -> vwredsum, NO nibble decode / offset-
  /// binary bias / high-low split) and (b) the fp32 fold is ggml's q8_0 order
  /// `sumf = sumf + (float)sumi * (d_x * d_y)` (scales multiplied FIRST, distinct
  /// from the Q4_0 sibling's `((sumi*d_x)*d_y)`).
  ///   float sumf = 0.0f;
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*34;           // emitc.mul + emitc.add
  ///     const uint8_t *yb = vy + ib*34;
  ///     float d_x = (float)*(const _Float16 *)(xb);   // emitc.call_opaque
  ///     float d_y = (float)*(const _Float16 *)(yb);
  ///     int32_t sumi = 0;
  ///     for (size_t c = 0; c < 32; c += vl) {     // strip loop, VLEN-robust
  ///       size_t vl = __riscv_vsetvl_e8m2(32 - c);     // m2: whole block @128
  ///       vint8m2_t  vx0 = __riscv_vle8_v_i8m2(xb + 2 + c, vl);
  ///       vint8m2_t  vy0 = __riscv_vle8_v_i8m2(yb + 2 + c, vl);
  ///       vint16m4_t p   = __riscv_vwmul_vv_i16m4(vx0, vy0, vl);
  ///       vint32m1_t seed = __riscv_vmv_v_x_i32m1(sumi, 1);
  ///       vint32m1_t red  = __riscv_vwredsum_vs_i16m4_i32m1(p, seed, vl);
  ///       sumi = __riscv_vmv_x_s_i32m1_i32(red);
  ///     }
  ///     sumf = sumf + (float)sumi * (d_x * d_y);   // ggml q8_0 order
  ///   }
  ///   *s = sumf;
  /// The strip loop anchors at integer_core_lmul: "m2" (the whole 32-element
  /// block in one strip + one vwredsum at VLEN=128, matching ggml's hand-written
  /// reduction anchor), "m1" (two 16-element strips), or "mf4" (eight 4-element
  /// strips). All are byte-exact and VLEN-robust via the sumi-carrying seed. The
  /// elided shape (single vsetvl_e8m2(32), no inner loop) is correct only at
  /// VLEN >= 128 and requires the m2 anchor (verifier-enforced).
  mlir::LogicalResult emitQ8_0Q8_0BlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ80Q80Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ80Q80Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The integer core's LMUL is a bounded resource/scheduling fact (m2 default
    // to match ggml's one-vwredsum-per-block anchor, or m1/mf4 to re-strip the
    // 32-element block). It is the *how* (vector grouping / strip width), never
    // the *what*: the dot product is byte-exact for any anchor (vwredsum sums the
    // same integer set; integer add is order-independent). The chosen i8 source
    // LMUL drives the widened i16 product LMUL and the i8 load/vsetvl spelling.
    llvm::StringRef coreLmul = "m2";
    if (std::optional<llvm::StringRef> attrLmul = blockDot.getIntegerCoreLmul())
      coreLmul = *attrLmul;
    int64_t multiBlockFactor = blockDot.getMultiBlockFactor().value_or(1);
    llvm::StringRef stripElision = blockDot.getStripElision().value_or("robust");
    bool stripElided = stripElision == "elided";
    // i8 source LMUL -> the next-wider i16 product LMUL (m2->m4, m1->m2,
    // mf4->mf2). The verifier bounds the source to mf4|m1|m2.
    llvm::StringRef wideLmul =
        (coreLmul == "m2") ? "m4" : (coreLmul == "m1") ? "m2" : "mf2";
    std::string i8CoreTypeName = ("vint8" + coreLmul + "_t").str();
    std::string i16WideTypeName = ("vint16" + wideLmul + "_t").str();
    mlir::Type i8CoreType = emitc::OpaqueType::get(ctx, i8CoreTypeName);
    mlir::Type i16WideType = emitc::OpaqueType::get(ctx, i16WideTypeName);
    mlir::Type i32m1Type = emitc::OpaqueType::get(ctx, "vint32m1_t");

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();
    int64_t weightStride = blockDot.getWeightBlockStride();
    int64_t activationStride = blockDot.getActivationBlockStride();
    int64_t quantOffset = blockDot.getQuantByteOffset();
    // q8_0 has NO high-half split: the integer core spans the whole QK=32-element
    // block, not the Q4_0 sibling's 16-lane half-block.
    int64_t blockLen = qk;

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float sumf = 0.0f;  (function-scoped accumulator across the block loop)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumfVar,
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));

    // size_t nb = n / QK;
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    // Per-block address arithmetic: const uint8_t *xb = vx + (ib+blockOffset)*34;
    // const uint8_t *yb = vy + (ib+blockOffset)*34 (both operands block_q8_0).
    auto blockBaseValue = [&](mlir::Value ib, int64_t blockOffset,
                              mlir::Value base, mlir::Type ptrType,
                              int64_t stride, const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value idx = ib;
      if (blockOffset != 0)
        idx = rewriter.create<emitc::AddOp>(loc, sizeType, ib,
                                            sizeLit(blockOffset));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, idx, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The two scalar fp16->fp32 reads (the ONE sanctioned opaque piece).
    auto fp16Read = [&](mlir::Value blockBase) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                       fp16ReadCallee,
                                       mlir::ValueRange{blockBase})
          .getResult(0);
    };

    // The plain i8 x i8 widening product + reduce for ONE strip, seeded with the
    // carried sumi, returning the next scalar sumi. vl is the active strip VL;
    // chunkOffset is the within-block byte offset (the strip induction var for
    // the robust loop, or a 0 literal for the elided core). When carrySumi is set
    // the seed lane0 is loaded from sumiVar; otherwise it is a "0" literal.
    auto emitStripReduce = [&](mlir::Value xb, mlir::Value yb,
                               mlir::Value chunkOffset, mlir::Value vl,
                               mlir::Value sumiVar,
                               bool carrySumi) -> mlir::FailureOr<mlir::Value> {
      auto chunkPtr = [&](mlir::Value base, mlir::Type ptrType,
                          int64_t fixed) -> mlir::Value {
        mlir::Value withFixed =
            rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
        mlir::Value full =
            rewriter.create<emitc::AddOp>(loc, ptrType, withFixed, chunkOffset);
        return rewriter.create<emitc::CastOp>(loc, i8PtrType, full).getResult();
      };
      std::string loadCallee = riscvIntrinsicName("vle", 8, coreLmul, "i8");
      auto loadChunk = [&](mlir::Value ptr) -> mlir::Value {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        return rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i8CoreType},
                                         loadCallee, mlir::ValueRange{ptr, vl})
            .getResult(0);
      };
      mlir::Value vx0 = loadChunk(chunkPtr(xb, weightPtrType, quantOffset));
      mlir::Value vy0 = loadChunk(chunkPtr(yb, activationPtrType, quantOffset));

      // Plain signed widening product: i8 x i8 -> i16 (NO nibble decode).
      std::string mulCallee = riscvIntrinsicName("vwmul", 16, wideLmul, "i16");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value product =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i16WideType},
                                           mulCallee,
                                           mlir::ValueRange{vx0, vy0, vl})
              .getResult(0);

      // Reduce into the per-block scalar: seed lane0 = sumi, vwredsum, extract.
      std::string seedCallee = riscvIntrinsicName("vmv_v_x", 32, "m1", "i32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, seedCallee));
      mlir::Value sumiSeed =
          carrySumi ? rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar)
                          .getResult()
                    : rewriter.create<emitc::LiteralOp>(loc, i32Type, "0")
                          .getResult();
      mlir::Value one = sizeLit(1);
      mlir::Value seed =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           seedCallee,
                                           mlir::ValueRange{sumiSeed, one})
              .getResult(0);
      std::string reduceCallee =
          ("__riscv_vwredsum_vs_i16" + wideLmul + "_i32m1").str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reduceCallee));
      mlir::Value red =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32m1Type},
                                           reduceCallee,
                                           mlir::ValueRange{product, seed, vl})
              .getResult(0);
      std::string extractCallee = "__riscv_vmv_x_s_i32m1_i32";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, extractCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{i32Type},
                                       extractCallee, mlir::ValueRange{red})
          .getResult(0);
    };

    // The per-block integer core: declares int32_t sumi = 0, runs the strip
    // reduce over the whole 32-element block, returns the sumi lvalue. When
    // forceRobust is set the inner strip loop is always kept (the tail / robust
    // shapes); otherwise stripElided selects the single-vsetvl(32) elided core.
    auto emitIntegerCore =
        [&](mlir::Value xb, mlir::Value yb,
            bool forceRobust) -> mlir::FailureOr<mlir::Value> {
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type), emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, sumiVar, rewriter.create<emitc::LiteralOp>(loc, i32Type, "0"));

      // The vsetvl SEW/LMUL spelling of the anchor: m2/m1 use vsetvl_e8<lmul>
      // (byte strips), mf4 uses vsetvl_e32m1 (4-element strips at VLEN=128).
      unsigned setvlSEW = (coreLmul == "mf4") ? 32 : 8;
      llvm::StringRef setvlLmul = (coreLmul == "mf4") ? "m1" : coreLmul;
      std::string innerSetvlCallee =
          riscvIntrinsicName("vsetvl", setvlSEW, setvlLmul, "");

      if (!forceRobust && stripElided) {
        // Elided core (m2-only, VLEN >= 128): ONE vsetvl_e8m2(32) (caps the
        // active vl at 32 when VLMAX >= 32, covering the whole block) + ONE
        // strip reduce. NO inner strip loop, NO sumi carry (seed lane0 = 0).
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{sizeLit(blockLen)})
                .getResult(0);
        mlir::FailureOr<mlir::Value> sumi = emitStripReduce(
            xb, yb, sizeLit(0), vl, sumiVar, /*carrySumi=*/false);
        if (mlir::failed(sumi))
          return mlir::failure();
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("sumi", opName, role));
        rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        return sumiVar.getResult();
      }

      // Robust core: the inner strip loop over the 32 block bytes; the loop STEP
      // is the loop-invariant VLMAX, the per-chunk active vl is vsetvl(32 - c).
      // Stays VLEN-robust via the sumi-carrying seed.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, innerSetvlCallee));
      mlir::Value innerVlmax =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           innerSetvlCallee,
                                           mlir::ValueRange{sizeLit(blockLen)})
              .getResult(0);
      auto innerLoop = rewriter.create<emitc::ForOp>(
          loc, sizeLit(0), sizeLit(blockLen), innerVlmax,
          /*bodyBuilder=*/nullptr);
      mlir::LogicalResult innerStatus = mlir::success();
      {
        mlir::OpBuilder::InsertionGuard innerGuard(rewriter);
        rewriter.setInsertionPointToStart(innerLoop.getBody());
        mlir::Value c = innerLoop.getInductionVar();

        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, innerSetvlCallee));
        mlir::Value remaining =
            rewriter.create<emitc::SubOp>(loc, sizeType, sizeLit(blockLen), c);
        mlir::Value vl =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                             innerSetvlCallee,
                                             mlir::ValueRange{remaining})
                .getResult(0);

        mlir::FailureOr<mlir::Value> sumi =
            emitStripReduce(xb, yb, c, vl, sumiVar, /*carrySumi=*/true);
        if (mlir::failed(sumi)) {
          innerStatus = mlir::failure();
        } else {
          rewriter.create<emitc::VerbatimOp>(
              loc, assignComment("sumi", opName, role));
          rewriter.create<emitc::AssignOp>(loc, sumiVar, *sumi);
        }
      }
      if (mlir::failed(innerStatus))
        return mlir::failure();
      return sumiVar.getResult();
    };

    // The fp32 accumulate sumf = sumf + (float)sumi * (d_x * d_y) (ggml's EXACT
    // q8_0 order: the two scales are multiplied FIRST, then by the integer sum;
    // distinct from the Q4_0 sibling's left-assoc ((sumi*d_x)*d_y)). Grouped into
    // ONE emitc.expression so mlir-translate renders it as a SINGLE C statement
    // and the compiler fuses the SAME FMA ggml does under -ffp-contract=on/
    // default. For multi-block unrolls the caller invokes this in STRICT
    // ascending block order, preserving fp non-associativity byte-exactly.
    auto emitFold = [&](mlir::Value sumiVar, mlir::Value dX, mlir::Value dY) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp32_accumulate"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // d_x * d_y FIRST (ggml q8_0 order).
        mlir::Value scaleProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dX, dY);
        mlir::Value blockTerm =
            rewriter.create<emitc::MulOp>(loc, floatType, sumiFloat,
                                          scaleProduct);
        mlir::Value sumfNext =
            rewriter.create<emitc::AddOp>(loc, floatType, sumfCur, blockTerm);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, accumExpr.getResult());
    };

    // One full block's integer core (addresses + scales + sumi), WITHOUT the
    // fold; the caller folds in strict ascending block order.
    struct BlockCore {
      mlir::Value sumiVar;
      mlir::Value dX;
      mlir::Value dY;
    };
    auto emitBlockCore =
        [&](mlir::Value ib, int64_t blockOffset,
            bool forceRobust) -> mlir::FailureOr<BlockCore> {
      mlir::Value xb = blockBaseValue(ib, blockOffset, weightBase,
                                      weightPtrType, weightStride, "block_base_x");
      mlir::Value yb =
          blockBaseValue(ib, blockOffset, activationBase, activationPtrType,
                         activationStride, "block_base_y");
      mlir::Value dX = fp16Read(xb);
      mlir::Value dY = fp16Read(yb);
      mlir::FailureOr<mlir::Value> sumiVar =
          emitIntegerCore(xb, yb, forceRobust);
      if (mlir::failed(sumiVar))
        return mlir::failure();
      return BlockCore{*sumiVar, dX, dY};
    };

    if (multiBlockFactor == 1) {
      // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the no-unroll form.
      auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                     sizeLit(1),
                                                     /*bodyBuilder=*/nullptr);
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::FailureOr<BlockCore> core =
          emitBlockCore(blockLoop.getInductionVar(), 0, /*forceRobust=*/false);
      if (mlir::failed(core))
        return mlir::failure();
      emitFold(core->sumiVar, core->dX, core->dY);
    } else {
      // Multi-block unroll: a main loop stepping by factor over nb - nb%factor
      // full groups -- emit ALL factor independent cores FIRST (the latency-
      // overlap lever), THEN the factor folds in strict ascending block order --
      // then a robust single-block scalar tail over the nb % factor remainder.
      mlir::Value factorLit = sizeLit(multiBlockFactor);
      mlir::Value nbRem =
          rewriter.create<emitc::RemOp>(loc, sizeType, nb, factorLit);
      mlir::Value nbMain =
          rewriter.create<emitc::SubOp>(loc, sizeType, nb, nbRem);
      auto mainLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nbMain,
                                                    factorLit,
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mainLoop.getBody());
        llvm::SmallVector<BlockCore> cores;
        for (int64_t k = 0; k < multiBlockFactor; ++k) {
          mlir::FailureOr<BlockCore> core =
              emitBlockCore(mainLoop.getInductionVar(), k, /*forceRobust=*/false);
          if (mlir::failed(core))
            return mlir::failure();
          cores.push_back(*core);
        }
        for (const BlockCore &core : cores)
          emitFold(core.sumiVar, core.dX, core.dY);
      }
      auto tailLoop = rewriter.create<emitc::ForOp>(loc, nbMain, nb, sizeLit(1),
                                                    /*bodyBuilder=*/nullptr);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(tailLoop.getBody());
        mlir::FailureOr<BlockCore> core =
            emitBlockCore(tailLoop.getInductionVar(), 0, /*forceRobust=*/true);
        if (mlir::failed(core))
          return mlir::failure();
        emitFold(core->sumiVar, core->dX, core->dY);
      }
    }

    // *s = sumf;  (structured scalar store through the output pointer)
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    mlir::Value sumfFinal =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumfFinal);

    valueMap[blockDot.getResult()] = sumfFinal;
    return mlir::success();
  }

  /// The bounded set of structural facts + EmitC types the q6_K super-block
  /// integer core needs. Shared by the K1 (aux32 partial) and K2 (full block
  /// dot) emitters so both drive the SAME unpack + sub-block-loop nodes; the
  /// integer core is byte-pinned by the K1 ssh-rvv artifact and must stay
  /// node-identical for either caller.
  struct Q6_KIntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type i32ImmType;
    mlir::Type i8ElemType;
    mlir::Type u8m2Type;
    mlir::Type i8m2Type;
    mlir::Type i8mf2Type;
    mlir::Type i16m1Type;
    mlir::Type i32m2Type;
    mlir::Type i8PtrType;
    mlir::Type u8PtrType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t subBlock;
    int64_t qhOffset;
    int64_t scalesOffset;
    int64_t q8Offset;
    int64_t numSubBlocks;
    int64_t half;
  };

  /// Emit ONE super-block's integer core (the 6-bit ql+qh unpack into the
  /// element-ordered aux8[256] scratch, biased -32, then the nested sub-block
  /// loop applying the per-sub-block int8 scale in the i32 domain) at the
  /// current insertion point (INSIDE an already-open super-block loop body),
  /// and RETURN the per-super-block aux32[8] integer-state vector (vint32m2).
  /// This is the genuinely-new super-block integer machinery, shared verbatim
  /// by K1 (which then vse32-stores the returned vector) and K2 (which folds it
  /// into the deferred fp32 accumulator). The emission mirrors _generic
  /// (quants.c:826-847) node-for-node: the aux32 accumulator is RESET to zero
  /// each super-block, the unpack carries the entire ql/qh->element permutation
  /// (the dot then reads aux8 contiguously), and the 8-lane accumulation is
  /// integer (order-free). RETURNS the aux32 lvalue VARIABLE (not a load), so
  /// each caller can materialize the load at its own statement position (K1's
  /// emitted C is byte-pinned by its ssh-rvv artifact).
  mlir::TypedValue<emitc::LValueType> emitQ6_KSuperBlockAux32Core(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q6_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    // base + fixed byte offset, cast to a typed (const uint8_t* / const int8_t*).
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };
    // u8 immediate vector op: __riscv_<mnemonic>_<dtype><lmul>(src, imm, vl).
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef imm, mlir::Value vl) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8m2").str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, imm.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.u8m2Type}, callee,
                                       mlir::ValueRange{src, amt, vl})
          .getResult(0);
    };
    auto u8VVOp = [&](llvm::StringRef mnemonic, mlir::Value a, mlir::Value b,
                      mlir::Value vl) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8m2").str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.u8m2Type}, callee,
                                       mlir::ValueRange{a, b, vl})
          .getResult(0);
    };
    auto u8Load = [&](mlir::Value ptr, mlir::Value vl) -> mlir::Value {
      std::string callee = "__riscv_vle8_v_u8m2";
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.u8m2Type}, callee,
                                       mlir::ValueRange{ptr, vl})
          .getResult(0);
    };

    // ---- (A) 6-bit ql+qh unpack into aux8 (element-ordered, biased -32) ----
    // Two 128-element chunks; per chunk a 32-wide e8m2 strip computes the four
    // element groups a[c+0/+32/+64/+96] and stores each into aux8.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "unpack_6bit"));
    std::string unpackSetvl = "__riscv_vsetvl_e8m2";
    for (int64_t chunk = 0; chunk < 2; ++chunk) {
      int64_t qlChunk = chunk * 64; // ql advances 64 bytes per 128-elem chunk
      int64_t qhChunk = chunk * 32; // qh advances 32 bytes per 128-elem chunk
      int64_t aChunk = chunk * 128; // aux8 element base for this chunk
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, unpackSetvl));
      mlir::Value vlu =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.sizeType},
                                           unpackSetvl,
                                           mlir::ValueRange{sizeLit(32)})
              .getResult(0);
      mlir::Value ql0Ptr =
          byteOffsetPtr(xb, cx.weightPtrType, qlChunk + 0, cx.u8PtrType);
      mlir::Value ql1Ptr =
          byteOffsetPtr(xb, cx.weightPtrType, qlChunk + 32, cx.u8PtrType);
      mlir::Value qhPtr =
          byteOffsetPtr(xb, cx.weightPtrType, cx.qhOffset + qhChunk,
                        cx.u8PtrType);
      mlir::Value ql0 = u8Load(ql0Ptr, vlu);
      mlir::Value ql1 = u8Load(ql1Ptr, vlu);
      mlir::Value qhv = u8Load(qhPtr, vlu);

      auto emitGroup = [&](mlir::Value qlByte, bool lowNibble,
                           llvm::StringRef shiftImm, int64_t aBase) {
        mlir::Value nib =
            lowNibble ? u8ImmOp("vand_vx", qlByte, "0x0F", vlu)
                      : u8ImmOp("vsrl_vx", qlByte, "0x04", vlu);
        mlir::Value qhShift =
            shiftImm == "0" ? qhv : u8ImmOp("vsrl_vx", qhv, shiftImm, vlu);
        mlir::Value qhMasked = u8ImmOp("vand_vx", qhShift, "0x03", vlu);
        mlir::Value hb = u8ImmOp("vsll_vx", qhMasked, "0x04", vlu);
        mlir::Value q6u = u8VVOp("vor_vv", nib, hb, vlu);
        std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reCallee));
        mlir::Value q6i =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8m2Type},
                                             reCallee, mlir::ValueRange{q6u})
                .getResult(0);
        std::string subCallee = "__riscv_vsub_vx_i8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, subCallee));
        mlir::Value bias =
            rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "32");
        mlir::Value aVal =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8m2Type},
                                             subCallee,
                                             mlir::ValueRange{q6i, bias, vlu})
                .getResult(0);
        mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(aChunk + aBase));
        mlir::Value dstElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, aux8Array,
                                            mlir::ValueRange{dstIdx})
                .getResult();
        mlir::Value dstPtr =
            rewriter
                .create<emitc::ApplyOp>(
                    loc, emitc::PointerType::get(cx.i8ElemType), "&", dstElem)
                .getResult();
        std::string storeCallee = "__riscv_vse8_v_i8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, storeCallee));
        rewriter.create<emitc::CallOpaqueOp>(
            loc, mlir::TypeRange{}, storeCallee,
            mlir::ValueRange{dstPtr, aVal, vlu});
      };
      // The exact _generic permutation (quants.c:828-831).
      emitGroup(ql0, /*lowNibble=*/true, "0", 0);
      emitGroup(ql1, /*lowNibble=*/true, "0x02", 32);
      emitGroup(ql0, /*lowNibble=*/false, "0x04", 64);
      emitGroup(ql1, /*lowNibble=*/false, "0x06", 96);
    }

    // ---- (B) per-sub-block int8-scaled i32 dot into the 8-lane aux32 ----
    // vint32m2_t aux32 = __riscv_vmv_v_x_i32m2(0, 8);  (RESET per super-block)
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux32", opName, role));
    auto aux32Var = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(cx.i32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string aux32SeedCallee = "__riscv_vmv_v_x_i32m2";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, aux32SeedCallee));
    mlir::Value zeroImm =
        rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "0");
    mlir::Value aux32Zero =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i32m2Type},
                                         aux32SeedCallee,
                                         mlir::ValueRange{zeroImm, sizeLit(8)})
            .getResult(0);
    rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Zero);

    // The q8 / scales bases for this super-block.
    mlir::Value q8Base =
        byteOffsetPtr(yb, cx.activationPtrType, cx.q8Offset, cx.i8PtrType);
    mlir::Value scBase =
        byteOffsetPtr(xb, cx.weightPtrType, cx.scalesOffset, cx.i8PtrType);

    // for (size_t js = 0; js < 16; js += 1) { ... }
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "sub_block_loop"));
    auto subLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), sizeLit(cx.numSubBlocks), sizeLit(1),
        /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard subGuard(rewriter);
      rewriter.setInsertionPointToStart(subLoop.getBody());
      mlir::Value js = subLoop.getInductionVar();

      // int scale = (int)scales[js];  (scalar int8 sign-extended load)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scale_load"));
      mlir::Value scElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc, llvm::cast<mlir::TypedValue<emitc::PointerType>>(scBase),
                  js)
              .getResult();
      mlir::Type constI8Type = emitc::OpaqueType::get(ctx, "const int8_t");
      mlir::Value scI8 =
          rewriter.create<emitc::LoadOp>(loc, constI8Type, scElem).getResult();
      mlir::Value scale =
          rewriter.create<emitc::CastOp>(loc, cx.i32ImmType, scI8).getResult();

      // js*16 -- the sub-block's first element offset into aux8 / q8.
      mlir::Value subBase =
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, js,
                                        sizeLit(cx.subBlock));

      // One half of 8: vwmul i8xi8 -> i16, then vwmacc.vx aux32 += scale*i16.
      auto emitHalf = [&](int64_t halfOffset) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_half"));
        std::string halfSetvl = "__riscv_vsetvl_e8mf2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, halfSetvl));
        mlir::Value vl8 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.sizeType},
                                             halfSetvl,
                                             mlir::ValueRange{sizeLit(cx.half)})
                .getResult(0);
        mlir::Value off = subBase;
        if (halfOffset != 0)
          off = rewriter.create<emitc::AddOp>(loc, cx.sizeType, subBase,
                                              sizeLit(halfOffset));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, q8Base, off)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, aux8Base, off)
                .getResult();
        std::string loadCallee = "__riscv_vle8_v_i8mf2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value q8v =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8mf2Type},
                                             loadCallee,
                                             mlir::ValueRange{q8Ptr, vl8})
                .getResult(0);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value av =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8mf2Type},
                                             loadCallee,
                                             mlir::ValueRange{aPtr, vl8})
                .getResult(0);
        std::string mulCallee = "__riscv_vwmul_vv_i16m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, mulCallee));
        mlir::Value p =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i16m1Type},
                                             mulCallee,
                                             mlir::ValueRange{q8v, av, vl8})
                .getResult(0);
        std::string maccCallee = "__riscv_vwmacc_vx_i32m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, maccCallee));
        mlir::Value aux32Cur =
            rewriter.create<emitc::LoadOp>(loc, cx.i32m2Type, aux32Var)
                .getResult();
        mlir::Value aux32Next =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{cx.i32m2Type}, maccCallee,
                    mlir::ValueRange{aux32Cur, scale, p, vl8})
                .getResult(0);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("aux32", opName, role));
        rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Next);
      };
      emitHalf(0);
      emitHalf(cx.half);
    }

    // The per-super-block aux32[8] state as the lvalue VARIABLE; the caller
    // loads it at its own statement position (K1 stores, K2 folds).
    return llvm::cast<mlir::TypedValue<emitc::LValueType>>(
        aux32Var.getResult());
  }

  /// Emit the ggml ggml_vec_dot_q6_K_q8_K INTEGER CORE (the K-quant K1
  /// increment) for one tcrv_rvv.q6_k_q8_k_aux32_partial op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob). It reproduces the
  /// per-super-block aux32[8] integer state EXACTLY as _generic computes it right
  /// before the fp32 d-multiply, byte-exact, and stores it through the output
  /// pointer (NO fp32 fold -- that is K2):
  ///   size_t nb = n / 256;
  ///   int8_t aux8[256];
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*210;  const uint8_t *yb = vy + ib*292;
  ///     // (A) unpack 256 weights into aux8 (element-ordered, biased -32):
  ///     //     two 128-element chunks, each a 4-strip 32-wide e8m2 vand/vsrl/
  ///     //     vsll/vor/vsub reconstruct + vse8 store at the matching offset.
  ///     // (B) per-sub-block int8-scaled i32 dot into the 8-lane aux32 vector:
  ///     vint32m2_t aux32 = vmv_v_x_i32m2(0, 8);   // RESET per super-block
  ///     for (size_t js = 0; js < 16; js += 1) {
  ///       int32_t scale = scales[js];             // int8 -> i32 scalar
  ///       // two halves of 8: vwmul i8xi8->i16 then vwmacc.vx i32 += scale*i16
  ///     }
  ///     vse32_v_i32m2(out + ib*8, aux32, 8);      // store aux32[8] state
  ///   }
  /// The unpack carries the ENTIRE ql/qh->element permutation (the dot then reads
  /// aux8 contiguously exactly as _generic's `a += 8`); the per-sub-block scale is
  /// applied in the i32 domain (vwmacc.vx); the 8-lane aux32 accumulation is
  /// order-free (integer add). The block-format facts are the op's typed attrs
  /// (I4 mirror); the emission is the op's fixed structure.
  mlir::LogicalResult emitQ6_KQ8_KAux32Partial(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ6KQ8KAux32Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ6KQ8KAux32Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q6_K partial body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q6_K partial ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u8Type = emitc::OpaqueType::get(ctx, "uint8_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 16
    int64_t weightStride = blockDot.getWeightBlockStride();         // 210
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qhOffset = blockDot.getWeightQhByteOffset();           // 128
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   // 192
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             // 16
    int64_t half = subBlock / 2;                      // 8

    // The integer-core vector types. The 6-bit unpack runs 32-wide chunks at
    // e8m2 (VLMAX = 32 at VLEN >= 128); the per-sub-block half-strip runs 8-wide
    // at e8m1 -> i16m1 product; the aux32 lane-collapsed accumulator is e32m2
    // (VLMAX = 8 at VLEN >= 128, exactly the 8 aux32 lanes).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination, mirroring _generic's aux8 -- the permutation lives here so the
    // per-sub-block dot reads aux8 contiguously).
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    // const int8_t *aux8 base pointer = &aux8[0] (for the vle8 contiguous reads).
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // Per-super-block base address arithmetic (vx + ib*210, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The bounded integer-core context shared with the K2 full-block-dot
    // emitter (the unpack + sub-block-loop nodes are byte-pinned by the K1
    // ssh-rvv artifact; both callers drive them identically).
    Q6_KIntegerCoreContext cx{
        opName,        role,          sizeType,         i32ImmType,
        i8ElemType,    u8m2Type,      i8m2Type,         i8mf2Type,
        i16m1Type,     i32m2Type,     i8PtrType,        u8PtrType,
        weightPtrType, activationPtrType, subBlock,     qhOffset,
        scalesOffset,  q8Offset,      numSubBlocks,     half};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K1/K2 integer core: unpack -> aux8 -> per-sub-block i32 dot,
      // returning the per-super-block aux32[8] lvalue.
      mlir::TypedValue<emitc::LValueType> aux32Var =
          emitQ6_KSuperBlockAux32Core(rewriter, loc, cx, xb, yb, aux8Array,
                                      aux8Base);

      // vse32_v_i32m2(out + ib*8, aux32, 8);  -- store the aux32[8] state.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_aux32"));
      mlir::Type i32PtrType = output.getType();
      mlir::Value outOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(8));
      mlir::Value outPtr =
          rewriter.create<emitc::AddOp>(loc, i32PtrType, output, outOff);
      mlir::Value aux32Final =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      std::string storeCallee = "__riscv_vse32_v_i32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, storeCallee));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, storeCallee,
          mlir::ValueRange{outPtr, aux32Final, sizeLit(8)});
    }

    // The op's i32 m1 result token: the lowering writes the aux32 state through
    // the output pointer (no scalar fold), so the token has no live use; bind it
    // to the zero seed literal to keep the value map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[blockDot.getResult()] = resultToken;
    (void)u8Type;
    return mlir::success();
  }

  /// Shared context for the q4_K super-block integer core (K4a + K4b), carrying
  /// the EmitC types + the block-format facts the unpack / bit-dance / sub-block
  /// dot need. The factoring mirrors q6_K's Q6_KIntegerCoreContext so the K4a
  /// emitted C stays byte-identical when K4b reuses the same core.
  struct Q4_KIntegerCoreContext {
    llvm::StringRef opName;
    llvm::StringRef role;
    mlir::Type sizeType;
    mlir::Type i32ImmType;
    mlir::Type u32Type;
    mlir::Type i8ElemType;
    mlir::Type u8m2Type;
    mlir::Type i8m2Type;
    mlir::Type i8mf2Type;
    mlir::Type i16m1Type;
    mlir::Type i32m2Type;
    mlir::Type i8PtrType;
    mlir::Type u8PtrType;
    mlir::Type constU32Type;
    mlir::Type u32PtrType;
    mlir::Type weightPtrType;
    mlir::Type activationPtrType;
    int64_t subBlock;
    int64_t scalesOffset;
    int64_t qsOffset;
    int64_t q8Offset;
    int64_t numSubBlocks;
    int64_t quarter;
  };

  /// The q4_K super-block integer core's result handles: the per-super-block
  /// aux32[8] integer-state lvalue (vint32m2, RESET each super-block) AND the
  /// const uint8_t* into the decoded utmp -- the 8 6-bit scales (bytes 0..7)
  /// followed by the 8 6-bit mins (bytes 8..15), the same type-pun _generic
  /// does with scales=&utmp[0]/mins=&utmp[2]. K4a vse8-stores both scales+mins
  /// and vse32-stores aux32; K4b folds aux32 into the deferred fp32 sums and
  /// reads the mins for the min term -- so the helper exposes both.
  struct Q4_KCoreResult {
    mlir::TypedValue<emitc::LValueType> aux32Var;
    mlir::Value scalesU8;
  };

  /// Emit ONE super-block's integer core (the plain 4-bit nibble unpack into the
  /// element-ordered aux8[256] scratch with NO bias; the STRUCTURED scalar 6-bit
  /// scale/min bit-dance via utmp/kmask; the nested sub-block loop applying the
  /// per-sub-block UINT6 scale in the i32 domain) at the current insertion point
  /// (INSIDE an already-open super-block loop body), and RETURN the
  /// per-super-block aux32[8] lvalue + the decoded scales/mins pointer. This is
  /// the genuinely-new q4_K super-block integer machinery, shared verbatim by
  /// K4a (which then vse8-stores the 16 scale/min bytes and vse32-stores the
  /// returned aux32) and K4b (which folds aux32 into the deferred fp32 sums and
  /// reads the mins for the min term). The emission mirrors _generic
  /// (quants.c:675-710) node-for-node: aux32 is RESET to zero each super-block,
  /// the unpack carries the entire q4 nibble->element layout (the dot then reads
  /// aux8 contiguously), and the 8-lane accumulation is integer (order-free).
  /// RETURNS the aux32 lvalue VARIABLE (not a load), so each caller can
  /// materialize the load at its own statement position (K4a's emitted C is
  /// byte-pinned by its ssh-rvv artifact).
  /// `scaleMinOutput`/`ib` are non-null ONLY for K4a (which exposes the 16
  /// decoded scale/min bytes through a uint8_t* output): when set, the helper
  /// vse8-stores the bytes IN-PLACE right after the bit-dance (before the aux32
  /// init), exactly where K4a has it today, so K4a's emitted node sequence stays
  /// byte-identical. K4b passes nullptr (no scale/min output) so the store is
  /// elided and the helper's order is (A unpack)(B bit-dance)(C sub-block dot).
  Q4_KCoreResult emitQ4_KSuperBlockAux32Core(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      const Q4_KIntegerCoreContext &cx, mlir::Value xb, mlir::Value yb,
      mlir::TypedValue<emitc::ArrayType> aux8Array, mlir::Value aux8Base,
      mlir::TypedValue<emitc::ArrayType> utmpArray, mlir::Value scaleMinOutput,
      mlir::Value ib) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    llvm::StringRef opName = cx.opName;
    llvm::StringRef role = cx.role;
    int64_t qk = cx.subBlock * cx.numSubBlocks; // 256
    int64_t quarter = cx.quarter;               // 8

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.sizeType,
                                               std::to_string(v));
    };
    auto u32Lit = [&](llvm::StringRef v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, cx.u32Type, v.str());
    };
    auto byteOffsetPtr = [&](mlir::Value base, mlir::Type ptrType, int64_t fixed,
                             mlir::Type castType) -> mlir::Value {
      mlir::Value full = base;
      if (fixed != 0)
        full = rewriter.create<emitc::AddOp>(loc, ptrType, base, sizeLit(fixed));
      return rewriter.create<emitc::CastOp>(loc, castType, full).getResult();
    };

    // ---- (A) 4-bit nibble unpack into aux8 (element-ordered, NO bias) ----
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "unpack_4bit"));
    std::string unpackSetvl = "__riscv_vsetvl_e8m2";
    auto u8Load = [&](mlir::Value ptr, mlir::Value vl) -> mlir::Value {
      std::string callee = "__riscv_vle8_v_u8m2";
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.u8m2Type}, callee,
                                       mlir::ValueRange{ptr, vl})
          .getResult(0);
    };
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef imm, mlir::Value vl) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8m2").str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, imm.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.u8m2Type}, callee,
                                       mlir::ValueRange{src, amt, vl})
          .getResult(0);
    };
    for (int64_t chunk = 0; chunk < qk / 64; ++chunk) {
      int64_t qsChunk = chunk * 32; // q4 advances 32 bytes per 64-elem chunk
      int64_t aChunk = chunk * 64;  // aux8 element base for this chunk
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, unpackSetvl));
      mlir::Value vlu =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.sizeType},
                                           unpackSetvl,
                                           mlir::ValueRange{sizeLit(32)})
              .getResult(0);
      mlir::Value qsPtr =
          byteOffsetPtr(xb, cx.weightPtrType, cx.qsOffset + qsChunk,
                        cx.u8PtrType);
      mlir::Value q4 = u8Load(qsPtr, vlu);

      auto emitNibble = [&](bool lowNibble, int64_t aBase) {
        mlir::Value nib =
            lowNibble ? u8ImmOp("vand_vx", q4, "0x0F", vlu)
                      : u8ImmOp("vsrl_vx", q4, "0x04", vlu);
        std::string reCallee = "__riscv_vreinterpret_v_u8m2_i8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, reCallee));
        mlir::Value q4i =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8m2Type},
                                             reCallee, mlir::ValueRange{nib})
                .getResult(0);
        mlir::Value dstIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(aChunk + aBase));
        mlir::Value dstElem =
            rewriter
                .create<emitc::SubscriptOp>(loc, aux8Array,
                                            mlir::ValueRange{dstIdx})
                .getResult();
        mlir::Value dstPtr =
            rewriter
                .create<emitc::ApplyOp>(
                    loc, emitc::PointerType::get(cx.i8ElemType), "&", dstElem)
                .getResult();
        std::string storeCallee = "__riscv_vse8_v_i8m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, storeCallee));
        rewriter.create<emitc::CallOpaqueOp>(
            loc, mlir::TypeRange{}, storeCallee,
            mlir::ValueRange{dstPtr, q4i, vlu});
      };
      emitNibble(/*lowNibble=*/true, 0);   // a[0..31]  = q4[l] & 0xF
      emitNibble(/*lowNibble=*/false, 32); // a[32..63] = q4[l] >> 4
    }

    // ---- (B) the 6-bit scale/min bit-dance (utmp/kmask), STRUCTURED scalar
    // emitc.bitwise ops (NO raw string), mirroring _generic (quants.c:685-690).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "scale_min_bit_dance"));
    mlir::Value kmask1 = u32Lit("0x3f3f3f3f");
    mlir::Value kmask2 = u32Lit("0x0f0f0f0f");
    mlir::Value kmask3 = u32Lit("0x03030303");
    mlir::Value shift4 = u32Lit("4");
    mlir::Value shift6 = u32Lit("6");
    mlir::Value scWordPtr =
        byteOffsetPtr(xb, cx.weightPtrType, cx.scalesOffset, cx.u32PtrType);
    auto loadWord = [&](int64_t idx) -> mlir::Value {
      mlir::Value wIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(idx));
      mlir::Value wElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(scWordPtr),
                  wIdx)
              .getResult();
      mlir::Value cw =
          rewriter.create<emitc::LoadOp>(loc, cx.constU32Type, wElem)
              .getResult();
      return rewriter.create<emitc::CastOp>(loc, cx.u32Type, cw).getResult();
    };
    mlir::Value w0 = loadWord(0);
    mlir::Value w1 = loadWord(1);
    mlir::Value w2 = loadWord(2);
    auto bAnd = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseAndOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    auto bOr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseOrOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    auto bShr = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseRightShiftOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    auto bShl = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      return rewriter.create<emitc::BitwiseLeftShiftOp>(loc, cx.u32Type, a, b)
          .getResult();
    };
    mlir::Value u3 =
        bOr(bAnd(bShr(w2, shift4), kmask2),
            bShl(bAnd(bShr(w1, shift6), kmask3), shift4));
    mlir::Value u2 = bAnd(w1, kmask1);
    mlir::Value u1 =
        bOr(bAnd(w2, kmask2),
            bShl(bAnd(bShr(w0, shift6), kmask3), shift4));
    mlir::Value u0 = bAnd(w0, kmask1);
    auto storeUtmp = [&](int64_t idx, mlir::Value v) {
      mlir::Value uIdx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(idx));
      mlir::Value uElem =
          rewriter
              .create<emitc::SubscriptOp>(loc, utmpArray,
                                          mlir::ValueRange{uIdx})
              .getResult();
      rewriter.create<emitc::AssignOp>(
          loc, llvm::cast<mlir::TypedValue<emitc::LValueType>>(uElem), v);
    };
    storeUtmp(0, u0);
    storeUtmp(1, u1);
    storeUtmp(2, u2);
    storeUtmp(3, u3);

    // scales = (const uint8_t *)&utmp[0]  (the 8 6-bit scales, contiguous with
    // the 8 mins -- 16 bytes total).
    mlir::Value utmpIndex0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value utmpElem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, utmpArray,
                                        mlir::ValueRange{utmpIndex0})
            .getResult();
    mlir::Value utmpWordPtr =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(cx.u32Type),
                                    "&", utmpElem0)
            .getResult();
    mlir::Value scalesU8 =
        rewriter.create<emitc::CastOp>(loc, cx.u8PtrType, utmpWordPtr)
            .getResult();

    // (K4a only) vse8 the 16 decoded [scales[0..7], mins[0..7]] bytes to
    // scaleMinOutput + ib*16 -- emitted IN-PLACE here (before the aux32 init) so
    // K4a's node sequence is unchanged. K4b passes a null scaleMinOutput.
    if (scaleMinOutput) {
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_scale_min"));
      std::string smSetvl = "__riscv_vsetvl_e8m1";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, smSetvl));
      mlir::Value vlsm =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.sizeType},
                                           smSetvl,
                                           mlir::ValueRange{sizeLit(16)})
              .getResult(0);
      std::string smLoad = "__riscv_vle8_v_u8m1";
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, smLoad));
      mlir::Type u8m1Type = emitc::OpaqueType::get(ctx, "vuint8m1_t");
      mlir::Value smVec =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{u8m1Type},
                                           smLoad,
                                           mlir::ValueRange{scalesU8, vlsm})
              .getResult(0);
      mlir::Type smOutPtrType = scaleMinOutput.getType();
      mlir::Value smOff =
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, ib, sizeLit(16));
      mlir::Value smOutPtr = rewriter.create<emitc::AddOp>(
          loc, smOutPtrType, scaleMinOutput, smOff);
      std::string smStore = "__riscv_vse8_v_u8m1";
      rewriter.create<emitc::VerbatimOp>(loc,
                                         stepComment(opName, role, smStore));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, smStore,
          mlir::ValueRange{smOutPtr, smVec, vlsm});
    }

    // ---- (C) per-sub-block uint6-scaled i32 dot into the 8-lane aux32 ----
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux32", opName, role));
    auto aux32Var = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(cx.i32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string aux32SeedCallee = "__riscv_vmv_v_x_i32m2";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, aux32SeedCallee));
    mlir::Value zeroImm =
        rewriter.create<emitc::LiteralOp>(loc, cx.i32ImmType, "0");
    mlir::Value aux32Zero =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i32m2Type},
                                         aux32SeedCallee,
                                         mlir::ValueRange{zeroImm, sizeLit(8)})
            .getResult(0);
    rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Zero);

    mlir::Value q8Base =
        byteOffsetPtr(yb, cx.activationPtrType, cx.q8Offset, cx.i8PtrType);

    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "sub_block_loop"));
    auto subLoop = rewriter.create<emitc::ForOp>(
        loc, sizeLit(0), sizeLit(cx.numSubBlocks), sizeLit(1),
        /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard subGuard(rewriter);
      rewriter.setInsertionPointToStart(subLoop.getBody());
      mlir::Value js = subLoop.getInductionVar();

      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "scale_load"));
      mlir::Value scElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(scalesU8),
                  js)
              .getResult();
      mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
      mlir::Value scU8 =
          rewriter.create<emitc::LoadOp>(loc, constU8Type, scElem).getResult();
      mlir::Value scale =
          rewriter.create<emitc::CastOp>(loc, cx.i32ImmType, scU8).getResult();

      mlir::Value subBase =
          rewriter.create<emitc::MulOp>(loc, cx.sizeType, js,
                                        sizeLit(cx.subBlock));

      auto emitQuarter = [&](int64_t quarterOffset) {
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, "sub_block_quarter"));
        std::string quarterSetvl = "__riscv_vsetvl_e8mf2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, quarterSetvl));
        mlir::Value vl8 =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.sizeType},
                                             quarterSetvl,
                                             mlir::ValueRange{sizeLit(quarter)})
                .getResult(0);
        mlir::Value off = subBase;
        if (quarterOffset != 0)
          off = rewriter.create<emitc::AddOp>(loc, cx.sizeType, subBase,
                                              sizeLit(quarterOffset));
        mlir::Value q8Ptr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, q8Base, off)
                .getResult();
        mlir::Value aPtr =
            rewriter.create<emitc::AddOp>(loc, cx.i8PtrType, aux8Base, off)
                .getResult();
        std::string loadCallee = "__riscv_vle8_v_i8mf2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value q8v =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8mf2Type},
                                             loadCallee,
                                             mlir::ValueRange{q8Ptr, vl8})
                .getResult(0);
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, loadCallee));
        mlir::Value av =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i8mf2Type},
                                             loadCallee,
                                             mlir::ValueRange{aPtr, vl8})
                .getResult(0);
        std::string mulCallee = "__riscv_vwmul_vv_i16m1";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, mulCallee));
        mlir::Value p =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{cx.i16m1Type},
                                             mulCallee,
                                             mlir::ValueRange{q8v, av, vl8})
                .getResult(0);
        std::string maccCallee = "__riscv_vwmacc_vx_i32m2";
        rewriter.create<emitc::VerbatimOp>(
            loc, stepComment(opName, role, maccCallee));
        mlir::Value aux32Cur =
            rewriter.create<emitc::LoadOp>(loc, cx.i32m2Type, aux32Var)
                .getResult();
        mlir::Value aux32Next =
            rewriter
                .create<emitc::CallOpaqueOp>(
                    loc, mlir::TypeRange{cx.i32m2Type}, maccCallee,
                    mlir::ValueRange{aux32Cur, scale, p, vl8})
                .getResult(0);
        rewriter.create<emitc::VerbatimOp>(
            loc, assignComment("aux32", opName, role));
        rewriter.create<emitc::AssignOp>(loc, aux32Var, aux32Next);
      };
      emitQuarter(0);
      emitQuarter(quarter);
      emitQuarter(2 * quarter);
      emitQuarter(3 * quarter);
    }

    return Q4_KCoreResult{
        llvm::cast<mlir::TypedValue<emitc::LValueType>>(aux32Var.getResult()),
        scalesU8};
  }

  /// Emit the ggml ggml_vec_dot_q4_K_q8_K INTEGER CORE (the q4_K K4a increment)
  /// for one tcrv_rvv.q4_k_q8_k_aux_partial op as fully STRUCTURED emitc nodes
  /// (I5; no verbatim C-control-flow blob, no raw() -- including the 6-bit
  /// scale/min bit-dance, which is scalar emitc.bitwise_and/_or/_left_shift/
  /// _right_shift). It reproduces ggml _generic's per-super-block INTEGER state
  /// EXACTLY right before the fp32 d/dmin fold -- (1) aux32[8] (the per-sub-block
  /// uint6-scaled i32 accumulator) and (2) the 8 decoded 6-bit scales + 8 decoded
  /// 6-bit mins -- byte-exact, and writes both through the two output pointers
  /// (NO fp32 fold, NO min term -- that is K4b):
  ///   size_t nb = n / 256;  int8_t aux8[256];  uint32_t utmp[4];
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*144;  const uint8_t *yb = vy + ib*292;
  ///     // (A) unpack 256 weights into aux8 (element-ordered, plain 4-bit, NO
  ///     //     bias): FOUR 64-element chunks, each vand 0x0F (low nibble ->
  ///     //     a[0..31]) + vsrl 4 (high nibble -> a[32..63]).
  ///     // (B) the 6-bit scale/min bit-dance (utmp/kmask), scalar emitc bitwise:
  ///     //     read 12 packed bytes as 3 uint32, shuffle into utmp[4], store the
  ///     //     16 bytes [scales[0..7], mins[0..7]] through scalemin_out+ib*16.
  ///     // (C) per-sub-block uint6-scaled i32 dot into the 8-lane aux32 vector:
  ///     vint32m2_t aux32 = vmv_v_x_i32m2(0, 8);   // RESET per super-block
  ///     for (size_t js = 0; js < 8; js += 1) {
  ///       int32_t scale = scales[js];             // uint8 ZERO-EXTENDED scalar
  ///       // four quarters of 8: vwmul i8xi8->i16 then vwmacc.vx i32 += scale*i16
  ///     }
  ///     vse32_v_i32m2(aux32_out + ib*8, aux32, 8); // store aux32[8] state
  ///   }
  /// The unpack carries the entire q4 nibble->element layout (the dot then reads
  /// aux8 contiguously exactly as _generic's `a += 8`); the per-sub-block scale is
  /// UINT8 zero-extended (not q6_K's sign-extended int8) and applied in the i32
  /// domain (vwmacc.vx); the 8-lane aux32 accumulation is order-free (integer
  /// add). The block-format facts are the op's typed attrs (I4 mirror); the
  /// emission is the op's fixed structure. A SEPARATE core from q6_K's
  /// emitQ6_KSuperBlockAux32Core (different unpack, no qh, no bias, uint scale) so
  /// q6_K's emitted bytes stay byte-identical (additive).
  mlir::LogicalResult emitQ4_KQ8_KAux32Partial(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ4KQ8KAux32Op blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ4KQ8KAux32Op>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K partial body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value aux32Output = valueMap.lookup(blockDot.getAux32Output());
    mlir::Value scaleMinOutput = valueMap.lookup(blockDot.getScaleminOutput());
    if (!weightBase || !activationBase || !aux32Output || !scaleMinOutput)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q4_K partial ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int32_t");
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 32
    int64_t weightStride = blockDot.getWeightBlockStride();         // 144
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   //   4
    int64_t qsOffset = blockDot.getWeightQsByteOffset();           //  16
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             //   8
    int64_t quarter = 8;                              // 8-elem quarters

    // The integer-core vector types. The 4-bit unpack runs 32-wide chunks at
    // e8m2 (VLMAX = 32 at VLEN >= 128); the per-sub-block quarter-strip runs
    // 8-wide at e8mf2 -> i16m1 product; the aux32 lane-collapsed accumulator is
    // e32m2 (VLMAX = 8 at VLEN >= 128, exactly the 8 aux32 lanes).
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    mlir::Type u32PtrType =
        emitc::PointerType::get(constU32Type);

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    // The integer-core context shared with K4b (identical unpack + bit-dance +
    // per-sub-block-loop nodes).
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        activationPtrType, subBlock,  scalesOffset,  qsOffset,
        q8Offset,      numSubBlocks,  quarter};

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination, mirroring _generic's aux8 -- the layout lives here so the
    // per-sub-block dot reads aux8 contiguously).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // uint32_t utmp[4];  (function-scoped scratch for the bit-dance; the 16 bytes
    // utmp[0..3] laid out are exactly [scales[0..7], mins[0..7]] -- the same
    // type-pun _generic does with scales=&utmp[0]/mins=&utmp[2]).
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // Per-super-block base address arithmetic (vx + ib*144, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K4a/K4b integer core (the 4-bit unpack + the bit-dance + the
      // in-place 16-byte scale/min store gated on the non-null scaleMinOutput +
      // the per-sub-block i32 dot), returning the per-super-block aux32 lvalue.
      Q4_KCoreResult core = emitQ4_KSuperBlockAux32Core(
          rewriter, loc, cx, xb, yb, aux8Array, aux8Base, utmpArray,
          scaleMinOutput, ib);
      mlir::TypedValue<emitc::LValueType> aux32Var = core.aux32Var;

      // vse32_v_i32m2(aux32_out + ib*8, aux32, 8);  -- store the aux32[8] state.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "store_aux32"));
      mlir::Type i32PtrType = aux32Output.getType();
      mlir::Value outOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(8));
      mlir::Value outPtr =
          rewriter.create<emitc::AddOp>(loc, i32PtrType, aux32Output, outOff);
      mlir::Value aux32Final =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      std::string storeCallee = "__riscv_vse32_v_i32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, storeCallee));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, storeCallee,
          mlir::ValueRange{outPtr, aux32Final, sizeLit(8)});
    }

    // The op's i32 m1 result token: the lowering writes the aux32/scalemin state
    // through the output pointers (no scalar fold), so the token has no live use;
    // bind it to the zero seed literal to keep the value map total.
    mlir::Value resultToken =
        rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
    valueMap[blockDot.getResult()] = resultToken;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_dot_q4_K_q8_K block kernel (the q4_K K4b
  /// increment) for one tcrv_rvv.q4_k_q8_k_block_dot op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob). It reuses the SAME K4a
  /// super-block integer core (emitQ4_KSuperBlockAux32Core: the 4-bit unpack
  /// into aux8[256] + the 6-bit scale/min bit-dance + the nested sub-block
  /// uint6-scaled i32 accumulation, returning the per-super-block aux32[8] AND
  /// the decoded scales/mins pointer) and adds the DEFERRED two-level fp32 fold
  /// (the q6_K K2 mechanism), the q4_K MIN term, and the fp32 *s store, mirroring
  /// _generic (quants.c:711-717) so byte-exactness is by construction:
  ///   float sums[8]; vfloat32m2_t sumsv = vfmv_v_f_f32m2(0.0f, 8); // ONCE
  ///   float sumf = 0.0f;                                            // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*144;  const uint8_t *yb = vy + ib*292;
  ///     <K4a integer core -> aux32, scalesU8 (mins = scalesU8 + 8)>;
  ///     float dy = *(const float *)(yb + 0);          // fp32 activation d, ONCE
  ///     // ---- the MIN term (q4_K-specific) ----
  ///     int sumi = 0;
  ///     for (size_t j = 0; j < 16; ++j)
  ///       sumi += (int)bsums[j] * (int)mins[j/2];     // bsums int16 SIGN-EXT
  ///     // ---- the deferred fp32 positive fold (q6_K K2) ----
  ///     float d = (float)*(const _Float16 *)(xb+0) * dy;
  ///     vfloat32m2_t af = vfcvt_f_x_v_f32m2(aux32, 8);
  ///     vfloat32m2_t pr = vfmul_vf_f32m2(af, d, 8);    // SEPARATE mul (NOT fma)
  ///     sumsv = vfadd_vv_f32m2(sumsv, pr, 8);          // SEPARATE add (NOT fma)
  ///     // ---- the MIN subtraction (single emitc.expression) ----
  ///     float dmin = (float)*(const _Float16 *)(xb+2) * dy;
  ///     sumf = sumf - dmin * (float)sumi;              // ONE C statement
  ///   }
  ///   float sums8[8];  vse32_v_f32m2(sums8, sumsv, 8); // lane l -> sums8[l]
  ///   sumf += sums8[0]; ...; sumf += sums8[7];         // SEQUENTIAL l=0..7
  ///   *s = sumf;
  /// The fp32 byte-exactness pivots match q6_K K2 EXACTLY for the positive fold:
  /// the 8 lanes are independent, the d-multiply is a SEPARATE vfmul then a
  /// SEPARATE vfadd (NEVER fused), and the final horizontal sum is SEQUENTIAL
  /// ascending l=0..7. The NEW q4_K piece is the MIN term: the integer
  /// sumi = Σ_{j=0..15} bsums[j] * mins[j/2] (bsums int16 SIGN-extended, each
  /// decoded uint6 min spanning TWO consecutive bsums via j/2, integer/order-free)
  /// then `sumf -= dmin * sumi` carried in the SCALAR sumf accumulator IN-LOOP
  /// (in super-block order) -- distinct from q6_K's post-loop-only sumf -- and
  /// summed with the 8 lanes only AFTER the loop. The MIN subtraction is ONE
  /// emitc.expression so it renders as ggml's single C statement. The
  /// block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitQ4_KQ8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ4KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ4KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q4_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q4_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type u32Type = emitc::OpaqueType::get(ctx, "uint32_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 32
    int64_t weightStride = blockDot.getWeightBlockStride();         // 144
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t weightDOffset = blockDot.getWeightDByteOffset();       //   0
    int64_t weightDminOffset = blockDot.getWeightDminByteOffset(); //   2
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   //   4
    int64_t qsOffset = blockDot.getWeightQsByteOffset();           //  16
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t bsumsOffset = blockDot.getActivationBsumsByteOffset(); // 260
    int64_t numSubBlocks = qk / subBlock;             //   8
    int64_t quarter = 8;                              // 8-elem quarters
    int64_t numLanes = 8;                             // aux32/sums lanes
    int64_t numBsums = qk / 16;                       //  16

    // The integer-core vector types (shared with K4a via the context) + the fp32
    // fold types.
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    mlir::Type constU32Type = emitc::OpaqueType::get(ctx, "const uint32_t");
    mlir::Type u32PtrType = emitc::PointerType::get(constU32Type);
    mlir::Type i32Type = emitc::OpaqueType::get(ctx, "int");
    mlir::Type constU8Type = emitc::OpaqueType::get(ctx, "const uint8_t");
    mlir::Type constI16Type = emitc::OpaqueType::get(ctx, "const int16_t");
    mlir::Type constI16PtrType = emitc::PointerType::get(constI16Type);
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination shared with the integer core).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // uint32_t utmp[4];  (function-scoped scratch for the bit-dance; the 16 bytes
    // utmp[0..3] are exactly [scales[0..7], mins[0..7]]).
    mlir::Type utmpArrayType = emitc::ArrayType::get({4}, u32Type);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("utmp", opName, role));
    auto utmpVar = rewriter.create<emitc::VariableOp>(
        loc, utmpArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto utmpArray =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(utmpVar.getResult());

    // float sums8[8];  (function-scoped scratch the 8-lane fp32 accumulator
    // vector is stored into before the SEQUENTIAL horizontal sum).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 positive accumulator, declared + zeroed ONCE OUTSIDE the super-block
    // loop (unlike aux32, which resets each super-block; mirrors _generic's
    // `memset(sums, 0, ...)`).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string sumsSeedCallee = "__riscv_vfmv_v_f_f32m2";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, sumsSeedCallee));
    mlir::Value zeroF =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    mlir::Value sumsZero =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                         sumsSeedCallee,
                                         mlir::ValueRange{zeroF, sizeLit(8)})
            .getResult(0);
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsZero);

    // float sumf = 0.0f;  -- the carried SCALAR fp32 accumulator the MIN
    // subtractions land in IN-LOOP (in super-block order); declared + zeroed ONCE
    // OUTSIDE the loop (distinct from q6_K's post-loop-only sumf -- mirrors
    // _generic's `float sumf = 0;` before the super-block loop).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto sumfVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value sumfZero =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    rewriter.create<emitc::AssignOp>(loc, sumfVar, sumfZero);

    // Per-super-block base address arithmetic (vx + ib*144, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The integer-core context shared with K4a (identical unpack + bit-dance +
    // per-sub-block-loop nodes). K4b passes a null scaleMinOutput / ib so the
    // helper does NOT emit the K4a-only 16-byte scale/min store.
    Q4_KIntegerCoreContext cx{
        opName,        role,          sizeType,      i32ImmType,
        u32Type,       i8ElemType,    u8m2Type,      i8m2Type,
        i8mf2Type,     i16m1Type,     i32m2Type,     i8PtrType,
        u8PtrType,     constU32Type,  u32PtrType,    weightPtrType,
        activationPtrType, subBlock,  scalesOffset,  qsOffset,
        q8Offset,      numSubBlocks,  quarter};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K4a/K4b integer core: unpack + bit-dance + per-sub-block i32
      // dot, returning the per-super-block aux32[8] lvalue AND the decoded
      // scales/mins pointer (mins = scalesU8 + 8). K4b passes a null
      // scaleMinOutput so the helper elides the K4a 16-byte scale/min store.
      Q4_KCoreResult core = emitQ4_KSuperBlockAux32Core(
          rewriter, loc, cx, xb, yb, aux8Array, aux8Base, utmpArray,
          /*scaleMinOutput=*/nullptr, /*ib=*/nullptr);
      mlir::TypedValue<emitc::LValueType> aux32Var = core.aux32Var;
      mlir::Value scalesU8 = core.scalesU8;

      // The fp32 activation scale dy = *(const float *)(yb + 0), loaded ONCE and
      // reused by both the positive d and the min-term dmin (mirrors _generic
      // reading `y[i].d` once).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_activation_d"));
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();

      // ---- the q4_K MIN term (the new piece vs q6_K) ----
      // int sumi = 0; for (j=0..15) sumi += (int)bsums[j] * (int)mins[j/2];
      // bsums are int16 (SIGN-extended on load); mins are the decoded uint6
      // scalesU8[8..15] (mins = scalesU8 + 8); each min spans TWO consecutive
      // bsums (mins[j/2]). Integer multiply/add -> order-free, a scalar loop.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "min_term_bsums"));
      mlir::Value bsumsAddr = yb;
      if (bsumsOffset != 0)
        bsumsAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                                  sizeLit(bsumsOffset));
      mlir::Value bsumsPtr =
          rewriter.create<emitc::CastOp>(loc, constI16PtrType, bsumsAddr)
              .getResult();
      // int sumi = 0;  (the per-super-block scalar min accumulator).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("sumi", opName, role));
      auto sumiVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(i32Type),
          emitc::OpaqueAttr::get(ctx, ""));
      mlir::Value sumiZero =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "0");
      rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiZero);
      for (int64_t j = 0; j < numBsums; ++j) {
        // (int)bsums[j]  -- int16 sign-extended to int.
        mlir::Value bIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(j));
        mlir::Value bElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(bsumsPtr),
                    bIdx)
                .getResult();
        mlir::Value bs16 =
            rewriter.create<emitc::LoadOp>(loc, constI16Type, bElem)
                .getResult();
        mlir::Value bs =
            rewriter.create<emitc::CastOp>(loc, i32Type, bs16).getResult();
        // (int)mins[j/2]  -- the decoded uint6 min, scalesU8[8 + j/2].
        mlir::Value mIdx = rewriter.create<emitc::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(8 + j / 2));
        mlir::Value mElem =
            rewriter
                .create<emitc::SubscriptOp>(
                    loc,
                    llvm::cast<mlir::TypedValue<emitc::PointerType>>(scalesU8),
                    mIdx)
                .getResult();
        mlir::Value mU8 =
            rewriter.create<emitc::LoadOp>(loc, constU8Type, mElem).getResult();
        mlir::Value m =
            rewriter.create<emitc::CastOp>(loc, i32Type, mU8).getResult();
        // sumi += bsums[j] * mins[j/2];  (integer, order-free).
        mlir::Value prod =
            rewriter.create<emitc::MulOp>(loc, i32Type, bs, m).getResult();
        mlir::Value sumiCur =
            rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
        mlir::Value sumiNext =
            rewriter.create<emitc::AddOp>(loc, i32Type, sumiCur, prod)
                .getResult();
        rewriter.create<emitc::AssignOp>(loc, sumiVar, sumiNext);
      }

      // ---- the DEFERRED two-level fp32 positive fold (q6_K K2 mechanism) ----
      // float d = (float)*(const _Float16 *)(xb + 0) * dy;  -- the fp16 weight
      // super-block scale (byte 0) times the fp32 activation scale (loaded once).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dxAddr})
              .getResult(0);
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
      mlir::Value aux32Val =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, cvtCallee));
      mlir::Value af =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                           cvtCallee,
                                           mlir::ValueRange{aux32Val, sizeLit(8)})
              .getResult(0);
      // vfloat32m2_t pr = __riscv_vfmul_vf_f32m2(af, d, 8);  -- SEPARATE multiply.
      std::string mulCallee = "__riscv_vfmul_vf_f32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value pr =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                           mulCallee,
                                           mlir::ValueRange{af, d, sizeLit(8)})
              .getResult(0);
      // sums = __riscv_vfadd_vv_f32m2(sums, pr, 8);  -- SEPARATE add (NEVER fma).
      std::string addCallee = "__riscv_vfadd_vv_f32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, addCallee));
      mlir::Value sumsCur =
          rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
      mlir::Value sumsNext =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{f32m2Type}, addCallee,
                  mlir::ValueRange{sumsCur, pr, sizeLit(8)})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sums", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);

      // ---- the MIN subtraction: sumf -= dmin * (float)sumi ----
      // float dmin = (float)*(const _Float16 *)(xb + 2) * dy;  -- the fp16 weight
      // min scale (byte 2) times the SAME fp32 activation scale.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_dmin"));
      mlir::Value dmAddr = xb;
      if (weightDminOffset != 0)
        dmAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDminOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      mlir::Value dmx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dmAddr})
              .getResult(0);
      mlir::Value dmin =
          rewriter.create<emitc::MulOp>(loc, floatType, dmx, dy).getResult();
      // sumf = sumf - dmin * (float)sumi;  -- ONE emitc.expression so it renders
      // as ggml's single C statement (quants.c:714 `sumf -= dmin * sumi`) and
      // tracks its contraction. The emitc.load temps stay OUTSIDE the expression.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "min_subtract"));
      mlir::Value sumiFinal =
          rewriter.create<emitc::LoadOp>(loc, i32Type, sumiVar).getResult();
      mlir::Value sumfCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
      auto minExpr = rewriter.create<emitc::ExpressionOp>(
          loc, floatType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&minExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value sumiFloat =
            rewriter.create<emitc::CastOp>(loc, floatType, sumiFinal)
                .getResult();
        // dmin * (float)sumi  -- the min product (ggml binds `dmin * sumi`
        // before the `-=`).
        mlir::Value minProduct =
            rewriter.create<emitc::MulOp>(loc, floatType, dmin, sumiFloat);
        // sumf - (dmin * sumi)  -- the `-=` subtraction tree.
        mlir::Value sumfNext =
            rewriter.create<emitc::SubOp>(loc, floatType, sumfCur, minProduct);
        rewriter.create<emitc::YieldOp>(loc, sumfNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sumf", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumfVar, minExpr.getResult());
    }

    // ---- the SEQUENTIAL horizontal sum, l = 0 .. 7 (AFTER the super-block loop;
    // sumf already holds the accumulated MIN subtractions) ----
    // vse32_v_f32m2(&sums8[0], sums, 8);  -- materialize lane l at sums8[l].
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums_lanes"));
    mlir::Value sums8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sums8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sums8Array,
                                        mlir::ValueRange{sums8Index0})
            .getResult();
    mlir::Value sums8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(floatType), "&",
                                    sums8Elem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    std::string laneStoreCallee = "__riscv_vse32_v_f32m2";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, laneStoreCallee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, laneStoreCallee,
        mlir::ValueRange{sums8Base, sumsFinal, sizeLit(8)});

    // sumf += sums8[0]; sumf += sums8[1]; ...; sumf += sums8[7];  -- the
    // SEQUENTIAL ascending fp32 horizontal sum, STARTING from the carried sumf
    // (which holds the MIN subtractions), NEVER a vfredusum. This mirrors
    // _generic's `for (l=0..7) sumf += sums[l]` after the super-block loop, with
    // sumf already carrying the in-loop min subtractions.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "horizontal_sum"));
    auto loadLane = [&](int64_t l) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(l));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(loc, sums8Array, mlir::ValueRange{idx})
              .getResult();
      return rewriter.create<emitc::LoadOp>(loc, floatType, elem).getResult();
    };
    mlir::Value sumf =
        rewriter.create<emitc::LoadOp>(loc, floatType, sumfVar).getResult();
    for (int64_t l = 0; l < numLanes; ++l)
      sumf =
          rewriter.create<emitc::AddOp>(loc, floatType, sumf, loadLane(l))
              .getResult();

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q4_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_dot_q6_K_q8_K block kernel (the K-quant K2
  /// increment) for one tcrv_rvv.q6_k_q8_k_block_dot op as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-control-flow blob). It reuses the SAME K1
  /// super-block integer core (emitQ6_KSuperBlockAux32Core: the 6-bit unpack into
  /// aux8[256] + the nested sub-block int8-scaled i32 accumulation, returning the
  /// per-super-block aux32[8]) and adds the DEFERRED two-level fp32 fold + the
  /// fp32 *s store, mirroring _generic (quants.c:813-852) so byte-exactness is by
  /// construction:
  ///   float sums[8];  vfloat32m2_t sumsv = vfmv_v_f_f32m2(0.0f, 8);  // ONCE
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     const uint8_t *xb = vx + ib*210;  const uint8_t *yb = vy + ib*292;
  ///     vint32m2_t aux32 = <K1 integer core>;          // RESET per super-block
  ///     float d = (float)*(const _Float16 *)(xb+208) * *(const float *)(yb+0);
  ///     vfloat32m2_t af = vfcvt_f_x_v_f32m2(aux32, 8); // (float)aux32[l], RNE
  ///     vfloat32m2_t pr = vfmul_vf_f32m2(af, d, 8);    // SEPARATE mul (NOT fma)
  ///     sumsv = vfadd_vv_f32m2(sumsv, pr, 8);          // SEPARATE add (NOT fma)
  ///   }
  ///   float sums8[8];  vse32_v_f32m2(sums8, sumsv, 8); // lane l -> sums8[l]
  ///   float sumf = sums8[0]; sumf += sums8[1]; ...; sumf += sums8[7]; // l=0..7
  ///   *s = sumf;
  /// The fp32 byte-exactness pivots: the 8 lanes are INDEPENDENT (so vectorizing
  /// the fold body is byte-safe), the d-multiply is a SEPARATE vfmul then a
  /// SEPARATE vfadd (NEVER a fused vfmacc/vfmadd -- that diverges from _generic's
  /// scalar mul/add under -ffp-contract=off, the INC-2a FMA class), and the final
  /// cross-lane horizontal sum is SEQUENTIAL ascending l=0..7 via 8 scalar
  /// emitc.add (NEVER a vector vfredusum -- fp add is non-associative). Lane l
  /// <-> index l is preserved by the vse32 store and pinned by the K1 aux32[l]
  /// <-> l ordering. The block-format facts are the op's typed attrs (I4 mirror).
  mlir::LogicalResult emitQ6_KQ8_KBlockDot(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlBlockDotQ6KQ8KOp blockDot;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto bd = llvm::dyn_cast<tcrvrvv::GgmlBlockDotQ6KQ8KOp>(op))
        blockDot = bd;
    }
    if (!blockDot)
      return rewriter.notifyMatchFailure(scope,
                                         "q6_K block-dot body missing the op");

    mlir::Value weightBase = valueMap.lookup(blockDot.getWeightBase());
    mlir::Value activationBase = valueMap.lookup(blockDot.getActivationBase());
    mlir::Value output = valueMap.lookup(blockDot.getOutput());
    if (!weightBase || !activationBase || !output)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q6_K block-dot ABI operand unmapped");

    llvm::StringRef opName = blockDot.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = blockDot.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type i32ImmType = emitc::OpaqueType::get(ctx, "int");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type weightPtrType = weightBase.getType();
    mlir::Type activationPtrType = activationBase.getType();

    // The block-format structural facts come straight off the typed attrs (I4).
    int64_t qk = blockDot.getQk();                    // 256
    int64_t subBlock = blockDot.getSubBlock();        // 16
    int64_t weightStride = blockDot.getWeightBlockStride();         // 210
    int64_t activationStride = blockDot.getActivationBlockStride(); // 292
    int64_t qhOffset = blockDot.getWeightQhByteOffset();           // 128
    int64_t scalesOffset = blockDot.getWeightScalesByteOffset();   // 192
    int64_t weightDOffset = blockDot.getWeightDByteOffset();       // 208
    int64_t activationDOffset = blockDot.getActivationDByteOffset();//   0
    int64_t q8Offset = blockDot.getActivationQuantByteOffset();    //   4
    int64_t numSubBlocks = qk / subBlock;             // 16
    int64_t half = subBlock / 2;                      // 8
    int64_t numLanes = numSubBlocks / 2;              //   8 (aux32/sums lanes)

    // The integer-core vector types (shared with K1 via the context) + the fp32
    // fold types.
    mlir::Type i8ElemType = emitc::OpaqueType::get(ctx, "int8_t");
    mlir::Type u8m2Type = emitc::OpaqueType::get(ctx, "vuint8m2_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");
    mlir::Type i8mf2Type = emitc::OpaqueType::get(ctx, "vint8mf2_t");
    mlir::Type i16m1Type = emitc::OpaqueType::get(ctx, "vint16m1_t");
    mlir::Type i32m2Type = emitc::OpaqueType::get(ctx, "vint32m2_t");
    mlir::Type f32m2Type = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const int8_t"));
    mlir::Type u8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const uint8_t"));
    llvm::StringRef fp16ReadCallee = "(float)*(const _Float16 *)";

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / QK_K;
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // int8_t aux8[256];  (function-scoped scratch, the element-ordered unpack
    // destination shared with the integer core).
    mlir::Type aux8ArrayType = emitc::ArrayType::get({qk}, i8ElemType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("aux8", opName, role));
    auto aux8Var = rewriter.create<emitc::VariableOp>(
        loc, aux8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto aux8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(aux8Var.getResult());
    mlir::Value aux8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value aux8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, aux8Array,
                                        mlir::ValueRange{aux8Index0})
            .getResult();
    mlir::Value aux8Base =
        rewriter.create<emitc::ApplyOp>(loc, i8PtrType, "&", aux8Elem0)
            .getResult();

    // float sums8[8];  (function-scoped scratch the 8-lane fp32 accumulator
    // vector is stored into before the SEQUENTIAL horizontal sum).
    mlir::Type sums8ArrayType = emitc::ArrayType::get({numLanes}, floatType);
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums8", opName, role));
    auto sums8Var = rewriter.create<emitc::VariableOp>(
        loc, sums8ArrayType, emitc::OpaqueAttr::get(ctx, ""));
    auto sums8Array =
        llvm::cast<mlir::TypedValue<emitc::ArrayType>>(sums8Var.getResult());

    // vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);  -- the carried 8-lane
    // fp32 accumulator, declared + zeroed ONCE OUTSIDE the super-block loop
    // (unlike aux32, which resets each super-block; this mirrors _generic's
    // `memset(sums, 0, ...)` before the loop).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sums", opName, role));
    auto sumsVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f32m2Type),
        emitc::OpaqueAttr::get(ctx, ""));
    std::string sumsSeedCallee = "__riscv_vfmv_v_f_f32m2";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, sumsSeedCallee));
    mlir::Value zeroF =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
    mlir::Value sumsZero =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                         sumsSeedCallee,
                                         mlir::ValueRange{zeroF, sizeLit(8)})
            .getResult(0);
    rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsZero);

    // Per-super-block base address arithmetic (vx + ib*210, vy + ib*292).
    auto blockBaseValue = [&](mlir::Value ib, mlir::Value base,
                              mlir::Type ptrType, int64_t stride,
                              const char *step) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, step));
      mlir::Value off =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(stride));
      return rewriter.create<emitc::AddOp>(loc, ptrType, base, off);
    };

    // The integer-core context shared with K1 (identical unpack + sub-block-loop
    // nodes).
    Q6_KIntegerCoreContext cx{
        opName,        role,          sizeType,         i32ImmType,
        i8ElemType,    u8m2Type,      i8m2Type,         i8mf2Type,
        i16m1Type,     i32m2Type,     i8PtrType,        u8PtrType,
        weightPtrType, activationPtrType, subBlock,     qhOffset,
        scalesOffset,  q8Offset,      numSubBlocks,     half};

    // The outer super-block loop: for (size_t ib = 0; ib < nb; ib += 1).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "super_block_loop"));
    auto blockLoop = rewriter.create<emitc::ForOp>(loc, sizeLit(0), nb,
                                                   sizeLit(1),
                                                   /*bodyBuilder=*/nullptr);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(blockLoop.getBody());
      mlir::Value ib = blockLoop.getInductionVar();

      mlir::Value xb = blockBaseValue(ib, weightBase, weightPtrType,
                                      weightStride, "super_block_base_x");
      mlir::Value yb = blockBaseValue(ib, activationBase, activationPtrType,
                                      activationStride, "super_block_base_y");

      // The shared K1/K2 integer core: unpack -> aux8 -> per-sub-block i32 dot,
      // returning the per-super-block aux32[8] lvalue.
      mlir::TypedValue<emitc::LValueType> aux32Var =
          emitQ6_KSuperBlockAux32Core(rewriter, loc, cx, xb, yb, aux8Array,
                                      aux8Base);

      // ---- (C) the DEFERRED two-level fp32 fold ----
      // float d = (float)*(const _Float16 *)(xb + 208) * *(const float *)(yb+0);
      // The weight super-block scale is fp16 (read via the one sanctioned opaque
      // seam); the activation scale is fp32 (a plain float load).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fold_scale_d"));
      mlir::Value dxAddr = xb;
      if (weightDOffset != 0)
        dxAddr = rewriter.create<emitc::AddOp>(loc, weightPtrType, xb,
                                               sizeLit(weightDOffset));
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fcvt.s.h"));
      mlir::Value dx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           fp16ReadCallee,
                                           mlir::ValueRange{dxAddr})
              .getResult(0);
      // *(const float *)(yb + 0): cast the activation base to const float * and
      // load element 0 (the fp32 q8_K scale d).
      mlir::Value dyAddr = yb;
      if (activationDOffset != 0)
        dyAddr = rewriter.create<emitc::AddOp>(loc, activationPtrType, yb,
                                               sizeLit(activationDOffset));
      mlir::Type constFloatPtrType =
          emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
      mlir::Value dyPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, dyAddr)
              .getResult();
      mlir::Value dyIndex0 =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      mlir::Value dyElem =
          rewriter
              .create<emitc::SubscriptOp>(
                  loc,
                  llvm::cast<mlir::TypedValue<emitc::PointerType>>(dyPtr),
                  dyIndex0)
              .getResult();
      mlir::Type constFloatType = emitc::OpaqueType::get(ctx, "const float");
      mlir::Value dy =
          rewriter.create<emitc::LoadOp>(loc, constFloatType, dyElem)
              .getResult();
      // d = dx * dy  (the fp16->fp32 weight scale times the fp32 activation
      // scale, a single fp32 multiply matching _generic's `const float d = ...`).
      mlir::Value d =
          rewriter.create<emitc::MulOp>(loc, floatType, dx, dy).getResult();

      // vfloat32m2_t af = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
      //   ((float)aux32[l], RNE -- matches C's `(float)aux32[l]` lane-wise)
      mlir::Value aux32Val =
          rewriter.create<emitc::LoadOp>(loc, i32m2Type, aux32Var).getResult();
      std::string cvtCallee = "__riscv_vfcvt_f_x_v_f32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, cvtCallee));
      mlir::Value af =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                           cvtCallee,
                                           mlir::ValueRange{aux32Val, sizeLit(8)})
              .getResult(0);
      // vfloat32m2_t pr = __riscv_vfmul_vf_f32m2(af, d, 8);  -- SEPARATE multiply
      // (NEVER a fused vfmacc: byte-exact only if the mul and add stay distinct,
      // matching _generic's scalar `d * aux32[l]` then `sums[l] += ...`).
      std::string mulCallee = "__riscv_vfmul_vf_f32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value pr =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32m2Type},
                                           mulCallee,
                                           mlir::ValueRange{af, d, sizeLit(8)})
              .getResult(0);
      // sums = __riscv_vfadd_vv_f32m2(sums, pr, 8);  -- SEPARATE add (NEVER fma).
      std::string addCallee = "__riscv_vfadd_vv_f32m2";
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, addCallee));
      mlir::Value sumsCur =
          rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
      mlir::Value sumsNext =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{f32m2Type}, addCallee,
                  mlir::ValueRange{sumsCur, pr, sizeLit(8)})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sums", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumsVar, sumsNext);
    }

    // ---- (D) the SEQUENTIAL horizontal sum, l = 0 .. 7 ----
    // vse32_v_f32m2(&sums8[0], sums, 8);  -- materialize lane l at sums8[l]
    // (the lane <-> index correspondence the sequential sum relies on).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_sums_lanes"));
    mlir::Value sums8Index0 =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    mlir::Value sums8Elem0 =
        rewriter
            .create<emitc::SubscriptOp>(loc, sums8Array,
                                        mlir::ValueRange{sums8Index0})
            .getResult();
    mlir::Value sums8Base =
        rewriter
            .create<emitc::ApplyOp>(loc, emitc::PointerType::get(floatType), "&",
                                    sums8Elem0)
            .getResult();
    mlir::Value sumsFinal =
        rewriter.create<emitc::LoadOp>(loc, f32m2Type, sumsVar).getResult();
    std::string laneStoreCallee = "__riscv_vse32_v_f32m2";
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, laneStoreCallee));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, mlir::TypeRange{}, laneStoreCallee,
        mlir::ValueRange{sums8Base, sumsFinal, sizeLit(8)});

    // float sumf = 0.0f; sumf += sums8[0]; sumf += sums8[1]; ...; sumf +=
    // sums8[7];  -- the SEQUENTIAL ascending fp32 horizontal sum (NEVER a
    // vfredusum: fp add is non-associative; this exactly mirrors _generic's
    // `float sumf = 0; for (l=0..7) sumf += sums[l]` INCLUDING the leading
    // `0.0f +` -- 8 adds from l=0, so byte-exactness holds even at the -0.0
    // edge).
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sumf", opName, role));
    auto loadLane = [&](int64_t l) -> mlir::Value {
      mlir::Value idx = rewriter.create<emitc::LiteralOp>(
          loc, rewriter.getIndexType(), std::to_string(l));
      mlir::Value elem =
          rewriter
              .create<emitc::SubscriptOp>(loc, sums8Array, mlir::ValueRange{idx})
              .getResult();
      return rewriter.create<emitc::LoadOp>(loc, floatType, elem).getResult();
    };
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "horizontal_sum"));
    mlir::Value sumf =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f").getResult();
    for (int64_t l = 0; l < numLanes; ++l)
      sumf =
          rewriter.create<emitc::AddOp>(loc, floatType, sumf, loadLane(l))
              .getResult();

    // *s = sumf;  (structured scalar store through the float * output pointer).
    auto outPointer =
        llvm::dyn_cast<mlir::TypedValue<emitc::PointerType>>(output);
    if (!outPointer)
      return rewriter.notifyMatchFailure(blockDot,
                                         "q6_K block-dot output not a pointer");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "store_s"));
    mlir::Value outIndex =
        rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
    emitc::SubscriptOp outSubscript =
        rewriter.create<emitc::SubscriptOp>(loc, outPointer, outIndex);
    rewriter.create<emitc::AssignOp>(loc, outSubscript.getResult(), sumf);

    valueMap[blockDot.getResult()] = sumf;
    return mlir::success();
  }

  /// Value-level emission of the ggml offset-binary asymmetric i4xi8
  /// decode/product chain, factored so BOTH the standalone
  /// tcrv_rvv.packed_i4_offset_binary_x_i8_product op (INC-1) AND the block-dot
  /// inner loop (INC-2a) drive the SAME structured nodes. Returns the i16
  /// widening product value (low + high halves) for the given already-mapped
  /// i8mf4 weight/activation values, their EmitC types, and the active VL:
  ///   w_xor   = vxor_vx_i8mf4(weight, 0x88, vl);  // offset-binary->two's-compl
  ///   w_low   = vsll_vx_i8mf4(w_xor, 4, vl);
  ///   v0      = vsra_vx_i8mf4(w_low, 4, vl);       // low nibble i8
  ///   v1      = vsra_vx_i8mf4(w_xor, 4, vl);       // high nibble i8
  ///   product = vwmul_vv_i16mf2(v0, actLow, vl);   // low half  <-> q8[0..15]
  ///   product = vwmacc_vv_i16mf2(product, v1, actHigh, vl); // + high half
  /// Pure node construction; no string plan read.
  mlir::FailureOr<mlir::Value> emitOffsetBinaryDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weight, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value bodyVL, mlir::Type srcEmitC, mlir::Type resultEmitC,
      llvm::StringRef srcDtype, llvm::StringRef srcLmul, unsigned resSEW,
      llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
      llvm::StringRef role) const {
    // The combined decode + product is the back-to-back composition of the two
    // factored halves (decode -> v0/v1, then the asymmetric widening product),
    // emitting the SAME six nodes in the SAME order INC-1's symmetric packed-i4
    // emitter does. The block-dot G1 GEMM tile (INC-14) hoists the decode half
    // ABOVE the M-column loop and replays the product half per column, reusing
    // the decoded v0/v1 lanes -- that weight-decode-reuse split is exactly why
    // these are factored. Callers that do NOT reuse the decode (the per-row
    // strip core, the standalone packed-i4 op) get byte-identical node output.
    std::pair<mlir::Value, mlir::Value> decoded = emitOffsetBinaryDecodeValue(
        rewriter, loc, weight, bodyVL, srcEmitC, srcDtype, srcLmul, opName, role);
    return emitOffsetBinaryProductFromDecodedValue(
        rewriter, loc, decoded.first, decoded.second, actLow, actHigh, bodyVL,
        resultEmitC, resSEW, resLmul, resDtype, opName, role);
  }

  /// The DECODE half of the offset-binary asymmetric i4xi8 chain, factored so
  /// the G1 GEMM tile (INC-14) can HOIST it above the M-column loop and reuse
  /// the decoded nibble lanes across all M activation columns (weight-decode
  /// reuse). Returns the (v0, v1) signed-i8 low/high nibble lanes:
  ///   w_xor = vxor_vx(weight, 0x88, vl);   // offset-binary -> two's-complement
  ///   w_low = vsll_vx(w_xor, 4, vl);
  ///   v0    = vsra_vx(w_low, 4, vl);        // low nibble i8 (sign-extended)
  ///   v1    = vsra_vx(w_xor, 4, vl);        // high nibble i8 (sign-extended)
  /// The four nodes are byte-identical to the leading four nodes of the combined
  /// emitter, so a back-to-back decode+product caller is unchanged.
  std::pair<mlir::Value, mlir::Value> emitOffsetBinaryDecodeValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weight, mlir::Value bodyVL, mlir::Type srcEmitC,
      llvm::StringRef srcDtype, llvm::StringRef srcLmul, llvm::StringRef opName,
      llvm::StringRef role) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");
    mlir::Type u8Type = emitc::OpaqueType::get(rewriter.getContext(), "uint8_t");

    // Scalar-immediate intrinsics spell __riscv_<mnemonic>_<dtype><lmul> (the
    // `vx` form is part of the mnemonic), as in the symmetric packed-i4 emitter.
    auto immCallee = [](llvm::StringRef mnemonic, llvm::StringRef dtype,
                        llvm::StringRef lmul) -> std::string {
      return ("__riscv_" + mnemonic + "_" + dtype + lmul).str();
    };
    auto immOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                     llvm::StringRef amount,
                     mlir::Type amtType) -> mlir::Value {
      std::string callee = immCallee(mnemonic, srcDtype, srcLmul);
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, amtType, amount.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcEmitC}, callee,
                                       mlir::ValueRange{src, amt, bodyVL})
          .getResult(0);
    };

    // Offset-binary -> two's-complement: xor both nibbles of every packed byte
    // with 0x88 (the vxor.vx scalar is a signed int).
    mlir::Value weightXor = immOp("vxor_vx", weight, "0x88", i32Type);
    // Low nibble: shift into the high nibble then arithmetic-shift back to
    // sign-extend it into a plain signed [-8,7] i8 lane.
    mlir::Value weightLowShifted = immOp("vsll_vx", weightXor, "4", u8Type);
    mlir::Value v0 = immOp("vsra_vx", weightLowShifted, "4", u8Type);
    // High nibble: arithmetic-shift sign-extends it in place.
    mlir::Value v1 = immOp("vsra_vx", weightXor, "4", u8Type);
    return {v0, v1};
  }

  /// The PRODUCT half of the offset-binary asymmetric i4xi8 chain, factored from
  /// the decode so the G1 GEMM tile (INC-14) can replay it per activation column
  /// against the SAME hoisted decoded v0/v1 lanes. Returns the i16 widening
  /// product (low + high halves):
  ///   product = vwmul_vv(v0, actLow, vl);                // low <-> q8[0..15]
  ///   product = vwmacc_vv(product, v1, actHigh, vl);     // + high half
  /// The two nodes are byte-identical to the trailing two nodes of the combined
  /// emitter, so a back-to-back decode+product caller is unchanged.
  mlir::FailureOr<mlir::Value> emitOffsetBinaryProductFromDecodedValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value v0, mlir::Value v1, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value bodyVL, mlir::Type resultEmitC, unsigned resSEW,
      llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
      llvm::StringRef role) const {
    // Asymmetric widening product: decoded i8 weight x plain i8 activation.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, mulCallee));
    mlir::Value lowProduct =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         mulCallee,
                                         mlir::ValueRange{v0, actLow, bodyVL})
            .getResult(0);
    // vwmacc accumulates the high-nibble x high-activation widening product.
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, maccCallee));
    mlir::Value pairSum =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, maccCallee,
                mlir::ValueRange{lowProduct, v1, actHigh, bodyVL})
            .getResult(0);
    return pairSum;
  }

  /// The UNSIGNED-nibble asymmetric i4xi8 decode/product for ONE strip -- the
  /// Family-B (q4_1) variant of emitOffsetBinaryDecodeProductValue, added cleanly
  /// alongside it (the offset-binary one is byte-untouched). ggml q4_1 decodes
  /// the nibbles as UNSIGNED [0,15] (NO offset-binary `-8` bias, NO XOR-0x88):
  ///   x_a = vand_vx_u8<L>(weightU8, 0x0F, vl);   // low nibble  [0,15]
  ///   x_l = vsrl_vx_u8<L>(weightU8, 0x04, vl);   // high nibble [0,15] (LOGICAL)
  ///   v0  = vreinterpret_v_u8<L>_i8<L>(x_a);     // value-identity for 0..15
  ///   v1  = vreinterpret_v_u8<L>_i8<L>(x_l);
  ///   product = vwmul_vv_i16<W>(v0, actLow, vl);            // low <-> q8[0..15]
  ///   product = vwmacc_vv_i16<W>(product, v1, actHigh, vl); // + high half
  /// The weight lane MUST be u8 (vsrl is a LOGICAL shift; there is no
  /// vsrl_vx_i8); the reinterprets are free (value-identity for 0..15). Pure node
  /// construction; no string plan read.
  mlir::FailureOr<mlir::Value> emitUnsignedNibbleDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weightU8, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value bodyVL, mlir::Type srcI8EmitC, mlir::Type srcU8EmitC,
      mlir::Type resultEmitC, llvm::StringRef srcLmul, unsigned resSEW,
      llvm::StringRef resLmul, llvm::StringRef resDtype, llvm::StringRef opName,
      llvm::StringRef role) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");

    // Scalar-immediate intrinsics spell __riscv_<mnemonic>_u8<lmul> (the `vx`
    // form is part of the mnemonic). The decode runs on the UNSIGNED weight lane.
    auto immOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                     llvm::StringRef amount) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, amount.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcU8EmitC}, callee,
                                       mlir::ValueRange{src, amt, bodyVL})
          .getResult(0);
    };

    // Low nibble: mask the low 4 bits ([0,15]). High nibble: logical-shift the
    // top 4 bits down ([0,15]). Both on the UNSIGNED lane.
    mlir::Value xLow = immOp("vand_vx", weightU8, "0x0F");
    mlir::Value xHigh = immOp("vsrl_vx", weightU8, "0x04");

    // Reinterpret the unsigned nibble lanes to signed i8 (value-identity for
    // 0..15) so they feed the SAME signed widening product the offset-binary core
    // uses. The reinterpret intrinsic is __riscv_vreinterpret_v_u8<L>_i8<L>.
    auto reinterpret = [&](mlir::Value src) -> mlir::Value {
      std::string callee =
          ("__riscv_vreinterpret_v_u8" + srcLmul + "_i8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcI8EmitC}, callee,
                                       mlir::ValueRange{src})
          .getResult(0);
    };
    mlir::Value v0 = reinterpret(xLow);
    mlir::Value v1 = reinterpret(xHigh);

    // Asymmetric widening product: decoded i8 weight x plain i8 activation.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, mulCallee));
    mlir::Value lowProduct =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         mulCallee,
                                         mlir::ValueRange{v0, actLow, bodyVL})
            .getResult(0);
    // vwmacc accumulates the high-nibble x high-activation widening product.
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, maccCallee));
    mlir::Value pairSum =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, maccCallee,
                mlir::ValueRange{lowProduct, v1, actHigh, bodyVL})
            .getResult(0);
    return pairSum;
  }

  /// The 5-BIT offset-binary asymmetric i5xi8 decode/product for ONE strip -- the
  /// Family-A (q5_0) variant of the nibble decoders, added cleanly alongside the
  /// offset-binary (q4_0) and unsigned (q4_1) ones (both byte-untouched). ggml
  /// q5_0 reconstructs each weight from a 4-bit nibble PLUS a per-element 5th high
  /// bit packed in a separate 32-bit qh field, then applies the offset-binary
  /// `-16` bias: for element i (low half) value = `(int8_t)(((qs[i]&0x0F) |
  /// (((qh>>i)&1)<<4)) - 16)`, for element i (high half) value =
  /// `(int8_t)(((qs[i]>>4) | (((qh>>(i+16))&1)<<4)) - 16)`. This emitter realizes
  /// that, for a strip covering element indices [c, c+vl) within the half-block:
  ///   xLow  = vand_vx_u8(weightU8, 0x0F, vl);  // low nibble  [0,15]
  ///   xHigh = vsrl_vx_u8(weightU8, 0x04, vl);  // high nibble [0,15] (LOGICAL)
  ///   // per-lane 5th bit: a vid+c shift vector selects each lane's qh bit.
  ///   idx   = vid_v_u16<W>(vl);                 // 0..vl-1
  ///   sh    = vadd_vx_u16<W>(idx, c, vl);       // absolute element index
  ///   lowB  = vmv_v_x_u16<W>(qhLow16, vl);      // qh & 0xFFFF broadcast
  ///   lowB  = vsrl_vv_u16<W>(lowB, sh, vl);     // shift each lane's bit to b0
  ///   lowB  = vand_vx_u16<W>(lowB, 1, vl);      // isolate {0,1}
  ///   lowB  = vsll_vx_u16<W>(lowB, 4, vl);      // place at bit 4 -> {0,16}
  ///   lowHB = vncvt_x_x_w_u8<core>(lowB, vl);   // narrow u16->u8
  ///   (same for the high half off qhHigh16 = qh>>16)
  ///   fiveLow  = vor_vv_u8<core>(xLow,  lowHB, vl);  // [0,31]
  ///   fiveHigh = vor_vv_u8<core>(xHigh, hiHB, vl);
  ///   v0 = vsub_vx_i8<core>(vreinterpret(fiveLow),  16, vl);  // [-16,15]
  ///   v1 = vsub_vx_i8<core>(vreinterpret(fiveHigh), 16, vl);
  ///   product = vwmul_vv_i16<W>(v0, actLow, vl);            // low <-> q8[0..15]
  ///   product = vwmacc_vv_i16<W>(product, v1, actHigh, vl); // + high half
  /// The shift uses `vid + c` (the absolute element index), so the per-strip 5th
  /// bit aligns for ANY chunk offset c (the robust mf4 multi-strip path) and ANY
  /// VLEN. The high half reads `qh >> 16` (bits 16..31), the low half `qh & 0xFFFF`
  /// (bits 0..15). The bit extraction runs in u16 at the wide LMUL (the same one
  /// the widening product uses); the nibble/OR/bias chain runs in u8/i8 at the
  /// core LMUL. Pure node construction; no string plan read.
  ///
  /// `applyOffsetBias` selects the bias model so the q5_1 (Family-B, scale+MIN,
  /// UNSIGNED 5-bit weight) kernel SHARES this exact unpack + 5th-bit injection:
  /// q5_1's ggml reconstruction is `(qs & 0xF) | ((qh>>j)<<4)&0x10` for the low
  /// half and `(qs>>4) | (qh>>(j+12))&0x10` for the high half -- the SAME low
  /// bit j / high bit (j+16) selection q5_0 uses, just with NO `-16` (the bias
  /// lives in q5_1's separate per-block MIN scale, exactly like q4_1). With
  /// applyOffsetBias=false the final `vsub 16` is skipped (the unsigned q5 in
  /// [0,31] feeds the signed widening product directly, value-identity), so the
  /// q5_0 emission (applyOffsetBias=true) stays byte-identical and the q5_1
  /// emission inherits the whole 5-bit reconstruction without duplicating it.
  mlir::FailureOr<mlir::Value> emitFiveBitOffsetBinaryDecodeProductValue(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::Value weightU8, mlir::Value actLow, mlir::Value actHigh,
      mlir::Value qhLow16, mlir::Value qhHigh16, mlir::Value chunkOffset,
      mlir::Value bodyVL, mlir::Type srcI8EmitC, mlir::Type srcU8EmitC,
      mlir::Type wideU16EmitC, mlir::Type resultEmitC, llvm::StringRef srcLmul,
      llvm::StringRef wideLmul, unsigned resSEW, llvm::StringRef resLmul,
      llvm::StringRef resDtype, llvm::StringRef opName, llvm::StringRef role,
      bool applyOffsetBias = true) const {
    mlir::Type i32Type = emitc::OpaqueType::get(rewriter.getContext(), "int");

    // u8 scalar-immediate op on the nibble lane (vand 0x0F / vsrl 0x04).
    auto u8ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                       llvm::StringRef amount) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, amount.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcU8EmitC}, callee,
                                       mlir::ValueRange{src, amt, bodyVL})
          .getResult(0);
    };

    // u16 (wide-LMUL) ops for the per-lane 5th-bit extraction.
    auto u16ImmOp = [&](llvm::StringRef mnemonic, mlir::Value src,
                        llvm::StringRef amount) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      mlir::Value amt =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, amount.str());
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{wideU16EmitC}, callee,
                                       mlir::ValueRange{src, amt, bodyVL})
          .getResult(0);
    };

    // The per-element 5th-bit lane from a broadcast 16-bit qh half: a vid+c shift
    // vector selects each lane's bit, isolate {0,1}, place at bit 4 -> {0,16},
    // narrow u16->u8. Identical for both halves off their respective qh scalar.
    auto fifthBitLane = [&](mlir::Value qhHalf16) -> mlir::Value {
      // idx = vid_v_u16<W>(vl);  sh = vadd_vx_u16<W>(idx, c, vl);
      std::string vidCallee = ("__riscv_vid_v_u16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, vidCallee));
      mlir::Value idx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{wideU16EmitC},
                                           vidCallee,
                                           mlir::ValueRange{bodyVL})
              .getResult(0);
      std::string addCallee = ("__riscv_vadd_vx_u16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, addCallee));
      mlir::Value sh =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{wideU16EmitC}, addCallee,
                  mlir::ValueRange{idx, chunkOffset, bodyVL})
              .getResult(0);
      // bits = vmv_v_x_u16<W>(qhHalf16, vl);  (broadcast the 16-bit qh half)
      std::string bcastCallee = ("__riscv_vmv_v_x_u16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, bcastCallee));
      mlir::Value bits =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{wideU16EmitC}, bcastCallee,
                  mlir::ValueRange{qhHalf16, bodyVL})
              .getResult(0);
      // bits = vsrl_vv_u16<W>(bits, sh, vl);  (per-lane bit -> b0)
      std::string srlCallee = ("__riscv_vsrl_vv_u16" + wideLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, srlCallee));
      mlir::Value srl =
          rewriter
              .create<emitc::CallOpaqueOp>(
                  loc, mlir::TypeRange{wideU16EmitC}, srlCallee,
                  mlir::ValueRange{bits, sh, bodyVL})
              .getResult(0);
      // bits = vand 1 -> {0,1}; bits = vsll 4 -> {0,16}.
      mlir::Value masked = u16ImmOp("vand_vx", srl, "0x1");
      mlir::Value shifted = u16ImmOp("vsll_vx", masked, "0x4");
      // narrow u16<W> -> u8<core>.
      std::string ncvtCallee = ("__riscv_vncvt_x_x_w_u8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, ncvtCallee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcU8EmitC},
                                       ncvtCallee,
                                       mlir::ValueRange{shifted, bodyVL})
          .getResult(0);
    };

    // Low/high 4-bit nibbles (unsigned), then OR in each half's 5th bit.
    mlir::Value xLow = u8ImmOp("vand_vx", weightU8, "0x0F");
    mlir::Value xHigh = u8ImmOp("vsrl_vx", weightU8, "0x04");
    mlir::Value lowHB = fifthBitLane(qhLow16);
    mlir::Value hiHB = fifthBitLane(qhHigh16);

    auto u8VVOp = [&](llvm::StringRef mnemonic, mlir::Value a,
                      mlir::Value b) -> mlir::Value {
      std::string callee = ("__riscv_" + mnemonic + "_u8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcU8EmitC}, callee,
                                       mlir::ValueRange{a, b, bodyVL})
          .getResult(0);
    };
    mlir::Value fiveLow = u8VVOp("vor_vv", xLow, lowHB);   // [0,31]
    mlir::Value fiveHigh = u8VVOp("vor_vv", xHigh, hiHB);  // [0,31]

    // Reinterpret to signed (value-identity for 0..31), then -- for the
    // offset-binary q5_0 path only -- apply the `-16` bias -> i8 [-16,15],
    // feeding the SAME signed widening product. When applyOffsetBias is false
    // (the q5_1 UNSIGNED 5-bit path: ggml's q5_1 reconstructs an unsigned q5 in
    // [0,31] with NO `-16`, mirroring q4_1's unsigned nibble) the bias `vsub` is
    // skipped, so the 5th-bit injection is SHARED byte-for-byte and only the
    // final bias op differs. The two halves still differ ONLY in which qh bits
    // feed them (already handled upstream).
    auto reinterpretBias = [&](mlir::Value u) -> mlir::Value {
      std::string reCallee =
          ("__riscv_vreinterpret_v_u8" + srcLmul + "_i8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, reCallee));
      mlir::Value s =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcI8EmitC},
                                           reCallee, mlir::ValueRange{u})
              .getResult(0);
      if (!applyOffsetBias)
        return s;
      std::string subCallee = ("__riscv_vsub_vx_i8" + srcLmul).str();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, subCallee));
      mlir::Value bias =
          rewriter.create<emitc::LiteralOp>(loc, i32Type, "16");
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{srcI8EmitC},
                                       subCallee,
                                       mlir::ValueRange{s, bias, bodyVL})
          .getResult(0);
    };
    mlir::Value v0 = reinterpretBias(fiveLow);
    mlir::Value v1 = reinterpretBias(fiveHigh);

    // Asymmetric widening product: decoded i8 weight x plain i8 activation.
    std::string mulCallee = riscvIntrinsicName("vwmul", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, mulCallee));
    mlir::Value lowProduct =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultEmitC},
                                         mulCallee,
                                         mlir::ValueRange{v0, actLow, bodyVL})
            .getResult(0);
    std::string maccCallee =
        riscvIntrinsicName("vwmacc", resSEW, resLmul, resDtype);
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, maccCallee));
    mlir::Value pairSum =
        rewriter
            .create<emitc::CallOpaqueOp>(
                loc, mlir::TypeRange{resultEmitC}, maccCallee,
                mlir::ValueRange{lowProduct, v1, actHigh, bodyVL})
            .getResult(0);
    return pairSum;
  }

  /// Emit the COMPLETE ggml ggml_vec_scale_f32 forward-pass op (the FIRST
  /// non-dot f32 elementwise family member) for one tcrv_rvv.ggml_vec_scale_f32
  /// op as fully STRUCTURED emitc nodes (I5; no verbatim C-string blob -- every
  /// value is a node in the IR graph):
  ///   for (size_t i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e32m<L>(n - i);    // emitc.sub + call_opaque
  ///     vfloat32m<L>_t ay = __riscv_vle32_v_f32m<L>(y + i, vl);
  ///     vfloat32m<L>_t ny = __riscv_vfmul_vf_f32m<L>(ay, v, vl);  // scalar bcast
  ///     __riscv_vse32_v_f32m<L>(y + i, ny, vl);       // in-place store back
  ///   }
  /// `y` is read AND written in place (the first forward-pass op whose single
  /// buffer is both input and output); `v` is the runtime f32 scalar broadcast
  /// into every lane. Byte-exactness to ggml's real op (vec.h:733-739) is
  /// UNCONDITIONAL: a bare per-lane fp32 multiply (no FMA -> -ffp-contract
  /// cannot bite; no cross-lane reduction -> LMUL/tail/strip-count are
  /// correctness-free). The LMUL is the bounded resource/scheduling knob
  /// (default m8, matching ggml). The intrinsics are emitc.call_opaque nodes
  /// (the one sanctioned opaque piece, exactly how the dot kernels emit theirs).
  mlir::LogicalResult emitGgmlVecScaleF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlVecScaleF32Op scaleOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto s = llvm::dyn_cast<tcrvrvv::GgmlVecScaleF32Op>(op))
        scaleOp = s;
    }
    if (!scaleOp)
      return rewriter.notifyMatchFailure(scope, "scale body missing the op");

    mlir::Value buffer = valueMap.lookup(scaleOp.getBuffer());
    mlir::Value scalar = valueMap.lookup(scaleOp.getScalar());
    if (!buffer || !scalar)
      return rewriter.notifyMatchFailure(scaleOp,
                                         "scale ABI operand unmapped");

    llvm::StringRef opName = scaleOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = scaleOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type bufferPtrType = buffer.getType();

    // The f32 strip-loop LMUL is a bounded resource/scheduling fact (default m8,
    // matching ggml's hand-written path). It is the *how* (vector grouping /
    // strip width), never the *what*: the result is byte-exact at any anchor
    // (every lane is multiplied by the same scalar v). The verifier bounds it to
    // m1|m2|m4|m8 (carried as "strip_lmul" so the op holds no forbidden
    // dataflow-parameter "lmul" at the I5 boundary).
    llvm::StringRef lmul = scaleOp.getStripLmul().value_or("m8");
    std::string f32VecTypeName = ("vfloat32" + lmul + "_t").str();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, f32VecTypeName);
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m<L>(n).
    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, setvlCallee));
    mlir::Value vlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         setvlCallee, mlir::ValueRange{avlArg})
            .getResult(0);

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // Remaining-AVL setvl: size_t vl = __riscv_vsetvl_e32m<L>(n - i).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, setvlCallee));
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);

      // In-place element pointer: float *p = y + i.
      mlir::Value elemPtr = rewriter.create<emitc::AddOp>(
          loc, bufferPtrType, buffer, inductionVar);
      mlir::Value loadPtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, elemPtr).getResult();

      // vfloat32m<L>_t ay = __riscv_vle32_v_f32m<L>(p, vl);
      std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      mlir::Value ay =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           loadCallee,
                                           mlir::ValueRange{loadPtr, bodyVL})
              .getResult(0);

      // vfloat32m<L>_t ny = __riscv_vfmul_vf_f32m<L>(ay, v, vl);  scalar bcast.
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value ny =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           mulCallee,
                                           mlir::ValueRange{ay, scalar, bodyVL})
              .getResult(0);

      // __riscv_vse32_v_f32m<L>(p, ny, vl);  store back in place.
      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, storeCallee));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, storeCallee,
          mlir::ValueRange{loadPtr, ny, bodyVL});
    }

    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_compute_forward_rms_norm_f32 forward-pass op
  /// (non-fused, no weight; ops.cpp:3758-3817) for ONE row as fully STRUCTURED
  /// emitc nodes (I5; no verbatim C-string blob -- every value is a node):
  ///   double sum = 0.0;                          // ggml_float accumulator
  ///   for (size_t i = 0; i < ne00; ++i) {        // SCALAR ascending fold
  ///     float p = x[i] * x[i];                    // f32 product (one f32 round)
  ///     sum = sum + (double)p;                    // widen-to-double then add
  ///   }
  ///   float mean  = (float)(sum / (double)ne00);  // divide in double, cast after
  ///   float scale = 1.0f / sqrtf(mean + eps);     // f32 add/sqrtf/reciprocal
  ///   for (size_t i = 0; i < ne00; i += vlmax) {  // VECTORIZED normalize strip
  ///     size_t vl = __riscv_vsetvl_e32m<L>(ne00 - i);
  ///     vfloat32m<L>_t vx = __riscv_vle32_v_f32m<L>(x + i, vl);
  ///     vfloat32m<L>_t vy = __riscv_vfmul_vf_f32m<L>(vx, scale, vl);
  ///     __riscv_vse32_v_f32m<L>(y + i, vy, vl);
  ///   }
  /// BYTE-EXACTNESS to ggml depends on the reduction METHOD, not a tolerance:
  /// ggml folds Sx^2 in `ggml_float` (= double), SCALAR, in strict ASCENDING
  /// index order (ops.cpp:3791-3795 -- explicitly NOT vectorized). The cast chain
  /// is load-bearing: each product rounds in f32 first, is WIDENED to double,
  /// then accumulated in double (NOT (double)x*(double)x). The f32->double widen
  /// between the f32 product and the double add is also an FMA barrier (different
  /// types -> -ffp-contract cannot fuse). The mean divides in double and casts to
  /// f32 AFTER; the scale is sqrtf-then-reciprocal in f32 (NOT a fast-rsqrt).
  /// Only the final normalize is vectorized -- a bare per-lane vfmul_vf (no FMA,
  /// no reduction), byte-exact at any LMUL because every lane is multiplied by
  /// the same scalar `scale`. The reduction is emitted as STRUCTURED scalar emitc
  /// nodes (variable/load/mul/cast/add/assign), NOT a raw string.
  mlir::LogicalResult emitGgmlRmsNormF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRmsNormF32Op rmsOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto r = llvm::dyn_cast<tcrvrvv::GgmlRmsNormF32Op>(op))
        rmsOp = r;
    }
    if (!rmsOp)
      return rewriter.notifyMatchFailure(scope, "rms_norm body missing the op");

    mlir::Value input = valueMap.lookup(rmsOp.getInput());
    mlir::Value outputBuf = valueMap.lookup(rmsOp.getOutput());
    mlir::Value eps = valueMap.lookup(rmsOp.getEps());
    if (!input || !outputBuf || !eps)
      return rewriter.notifyMatchFailure(rmsOp,
                                         "rms_norm ABI operand unmapped");

    llvm::StringRef opName = rmsOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = rmsOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = outputBuf.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type doubleType = emitc::OpaqueType::get(ctx, "double");
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // ggml_float sum = 0.0;  (the SCALAR double accumulator -- ggml_float is
    // double; ops.cpp:3791). emitc.for has no iter_args, so the loop-carried
    // accumulator is an emitc.variable lvalue + emitc.assign, exactly as the
    // block-dot kernels carry the fp32 *s accumulator.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("sum", opName, role));
    auto sumVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(doubleType),
        emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(
        loc, sumVar, rewriter.create<emitc::LiteralOp>(loc, doubleType, "0.0"));

    // for (size_t i = 0; i < ne00; ++i) { ... }  -- the SCALAR ascending fold
    // (step 1). This loop is NOT vectorized: a vectorized vfredusum would fold in
    // f32 with a tree order and break byte-exactness vs ggml.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "scalar_double_reduce"));
    mlir::Value zeroIdx =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value oneStep =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    auto reduceFor = rewriter.create<emitc::ForOp>(loc, zeroIdx, avlArg, oneStep,
                                                   /*bodyBuilder=*/nullptr);
    mlir::Value redIdx = reduceFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(reduceFor.getBody());

      // const float *xp = (const float *)(x + i);  float xi = xp[0];
      mlir::Value xElemPtr = rewriter.create<emitc::AddOp>(
          loc, inputPtrType, input, redIdx);
      auto xElemPtrCast =
          llvm::cast<mlir::TypedValue<emitc::PointerType>>(
              rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xElemPtr)
                  .getResult());
      mlir::Value xElemIndex =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      emitc::SubscriptOp xSubscript =
          rewriter.create<emitc::SubscriptOp>(loc, xElemPtrCast, xElemIndex);
      auto xLValueType =
          llvm::cast<emitc::LValueType>(xSubscript.getResult().getType());
      mlir::Value xiLoaded =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     xSubscript.getResult())
              .getResult();

      // The load-bearing cast chain, grouped into ONE emitc.expression so
      // mlir-translate renders ONE C statement: sum = sum + (double)(xi * xi).
      // The f32 product rounds FIRST, is WIDENED to double, then added in double.
      // The widen sits BETWEEN the f32 multiply and the double add -- an FMA
      // barrier (different types), so -ffp-contract cannot fuse the two. The
      // emitc.load temps stay OUTSIDE the expression (load lacks the CExpression
      // trait).
      mlir::Value sumCur =
          rewriter.create<emitc::LoadOp>(loc, doubleType, sumVar).getResult();
      auto accumExpr = rewriter.create<emitc::ExpressionOp>(
          loc, doubleType, /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&accumExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        // float p = xi * xi;  (f32 product -- one f32 rounding)
        mlir::Value prod =
            rewriter.create<emitc::MulOp>(loc, floatType, xiLoaded, xiLoaded);
        // (double)p  -- widen the f32 product to double (the FMA barrier).
        mlir::Value prodWide =
            rewriter.create<emitc::CastOp>(loc, doubleType, prod).getResult();
        // sum + (double)p  -- accumulate in double.
        mlir::Value sumNext =
            rewriter.create<emitc::AddOp>(loc, doubleType, sumCur, prodWide);
        rewriter.create<emitc::YieldOp>(loc, sumNext);
      }
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("sum", opName, role));
      rewriter.create<emitc::AssignOp>(loc, sumVar, accumExpr.getResult());
    }

    // float mean = (float)(sum / (double)ne00);  -- divide in DOUBLE, cast to
    // f32 AFTER the division (ggml's `const float mean = sum/ne00;` with
    // ne00 promoted to double for the ggml_float divide; ops.cpp:3797).
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "mean"));
    mlir::Value sumFinal =
        rewriter.create<emitc::LoadOp>(loc, doubleType, sumVar).getResult();
    mlir::Value ne00Wide =
        rewriter.create<emitc::CastOp>(loc, doubleType, avlArg).getResult();
    mlir::Value meanDouble =
        rewriter.create<emitc::DivOp>(loc, doubleType, sumFinal, ne00Wide);
    mlir::Value mean =
        rewriter.create<emitc::CastOp>(loc, floatType, meanDouble).getResult();

    // float scale = 1.0f / sqrtf(mean + eps);  -- the add, sqrtf, and reciprocal
    // are ALL f32 (ggml's `1.0f/sqrtf(mean + eps)`; ops.cpp:3798). sqrtf is the
    // scalar libm call (call_opaque, the one sanctioned opaque seam) -- a true
    // IEEE correctly-rounded sqrt, NOT a hardware fast-rsqrt7. The reciprocal is
    // a separate f32 divide.
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "scale"));
    mlir::Value meanPlusEps =
        rewriter.create<emitc::AddOp>(loc, floatType, mean, eps);
    mlir::Value sqrtVal =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                         "sqrtf",
                                         mlir::ValueRange{meanPlusEps})
            .getResult(0);
    mlir::Value oneF =
        rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
    mlir::Value scale =
        rewriter.create<emitc::DivOp>(loc, floatType, oneF, sqrtVal);

    // The VECTORIZED normalize strip (step 4): y[i] = x[i] * scale. The NORMALIZE
    // strip LMUL is a bounded resource/scheduling fact (default m8, matching
    // ggml's ggml_vec_scale_f32 apply path). It is byte-exact at any anchor (a
    // bare per-lane vfmul_vf -- no FMA, no reduction). This is the SAME strip
    // machinery F1 (scale) emits, except two-buffer (x in, y out) instead of
    // in-place: byte-identical (both one f32 multiply per lane), avoiding ggml's
    // memcpy+in-place-scale.
    llvm::StringRef lmul = rmsOp.getStripLmul().value_or("m8");
    std::string f32VecTypeName = ("vfloat32" + lmul + "_t").str();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, f32VecTypeName);

    std::string setvlCallee = riscvIntrinsicName("vsetvl", 32, lmul, "");
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, setvlCallee));
    mlir::Value vlmax =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                         setvlCallee, mlir::ValueRange{avlArg})
            .getResult(0);

    mlir::Value normZero =
        rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto normFor = rewriter.create<emitc::ForOp>(loc, normZero, avlArg, vlmax,
                                                 /*bodyBuilder=*/nullptr);
    mlir::Value normIdx = normFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(normFor.getBody());

      // size_t vl = __riscv_vsetvl_e32m<L>(ne00 - i);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, setvlCallee));
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, normIdx);
      mlir::Value bodyVL =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{sizeType},
                                           setvlCallee,
                                           mlir::ValueRange{remaining})
              .getResult(0);

      // const float *xp = x + i;  float *yp = y + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, normIdx);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();
      mlir::Value yPtr =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, outputBuf, normIdx);
      mlir::Value yStorePtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();

      // vfloat32m<L>_t vx = __riscv_vle32_v_f32m<L>(xp, vl);
      std::string loadCallee = riscvIntrinsicName("vle", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, loadCallee));
      mlir::Value vx =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           loadCallee,
                                           mlir::ValueRange{xLoadPtr, bodyVL})
              .getResult(0);

      // vfloat32m<L>_t vy = __riscv_vfmul_vf_f32m<L>(vx, scale, vl);
      std::string mulCallee = riscvIntrinsicName("vfmul_vf", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, mulCallee));
      mlir::Value vy =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{f32VecType},
                                           mulCallee,
                                           mlir::ValueRange{vx, scale, bodyVL})
              .getResult(0);

      // __riscv_vse32_v_f32m<L>(yp, vy, vl);
      std::string storeCallee = riscvIntrinsicName("vse", 32, lmul, "f32");
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, storeCallee));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, storeCallee,
          mlir::ValueRange{yStorePtr, vy, bodyVL});
    }

    return mlir::success();
  }

  /// Emit ggml's EXACT vectorized minimax exp polynomial `ggml_v_expf_m2`
  /// (vec.h:1324-1360) as a node-for-node STRUCTURED emitc chain, returning the
  /// `e^X` result as a `vfloat32m2_t` value. `X` is the already-computed m2 f32
  /// argument vector (silu feeds `vfneg(x)`; soft_max feeds `vfsub_vf(x, max)`),
  /// `bodyVL` the active strip VL.
  ///
  /// This is the ONE shared replication of ggml's exp polynomial, factored out
  /// of the F5 silu emitter so the F5b soft_max emitter consumes the IDENTICAL
  /// chain (the byte-exactness invariant: both emit the SAME instructions ggml's
  /// source compiles to). The slow-path overflow/underflow vmerge value graph is
  /// emitted UNCONDITIONALLY -- ggml's `if (!vcpop_m(c))` is a pure performance
  /// short-circuit whose fast path `k + j*k` equals the slow path's c-false lane
  /// `k + k*j` bit-for-bit (fp multiply commutes; RISC-V yields the canonical NaN
  /// regardless of operand order), so there is no data-dependent branch. Pinned
  /// at m2 (the m2-tied vbool16_t mask / vuint32m2_t reinterpret types).
  mlir::Value emitGgmlVExpfM2(mlir::ConversionPatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value X,
                              mlir::Value bodyVL, mlir::Type sizeType,
                              llvm::StringRef opName, llvm::StringRef role) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type u32VecType = emitc::OpaqueType::get(ctx, "vuint32m2_t");
    mlir::Type boolType = emitc::OpaqueType::get(ctx, "vbool16_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type u32ScalarType = emitc::OpaqueType::get(ctx, "uint32_t");

    auto vcall = [&](mlir::Type resTy, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resTy}, callee, args)
          .getResult(0);
    };
    auto fimm = [&](llvm::StringRef tok) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, floatType, tok);
    };
    auto uimm = [&](llvm::StringRef tok) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, u32ScalarType, tok);
    };
    auto shiftImm = [&]() -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, "23");
    };

    // const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
    mlir::Value r = vcall(f32VecType, "__riscv_vfmv_v_f_f32m2",
                          mlir::ValueRange{fimm("0x1.8p23f"), bodyVL});
    // const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
    mlir::Value z = vcall(
        f32VecType, "__riscv_vfmacc_vf_f32m2",
        mlir::ValueRange{r, fimm("0x1.715476p+0f"), X, bodyVL});
    // const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    mlir::Value n = vcall(f32VecType, "__riscv_vfsub_vv_f32m2",
                          mlir::ValueRange{z, r, bodyVL});
    // const vfloat32m2_t b = vfnmsac(vfnmsac(x, 0x1.62e4p-1f, n), 0x1.7f7d1cp-20f, n);
    mlir::Value bInner = vcall(
        f32VecType, "__riscv_vfnmsac_vf_f32m2",
        mlir::ValueRange{X, fimm("0x1.62e4p-1f"), n, bodyVL});
    mlir::Value b = vcall(
        f32VecType, "__riscv_vfnmsac_vf_f32m2",
        mlir::ValueRange{bInner, fimm("0x1.7f7d1cp-20f"), n, bodyVL});
    // const vuint32m2_t e = __riscv_vsll_vx_u32m2(vreinterpret_v_f32m2_u32m2(z), 23, vl);
    mlir::Value zBits = vcall(u32VecType, "__riscv_vreinterpret_v_f32m2_u32m2",
                              mlir::ValueRange{z});
    mlir::Value e = vcall(u32VecType, "__riscv_vsll_vx_u32m2",
                          mlir::ValueRange{zBits, shiftImm(), bodyVL});
    // const vfloat32m2_t k = vreinterpret(vadd_vx_u32m2(e, 0x3f800000, vl));
    mlir::Value kBits = vcall(u32VecType, "__riscv_vadd_vx_u32m2",
                              mlir::ValueRange{e, uimm("0x3f800000"), bodyVL});
    mlir::Value k = vcall(f32VecType, "__riscv_vreinterpret_v_u32m2_f32m2",
                          mlir::ValueRange{kBits});
    // const vbool16_t c = vmfgt_vf(vfabs_v(n), 126.0f, vl);
    mlir::Value absN1 = vcall(f32VecType, "__riscv_vfabs_v_f32m2",
                              mlir::ValueRange{n, bodyVL});
    mlir::Value c = vcall(boolType, "__riscv_vmfgt_vf_f32m2_b16",
                          mlir::ValueRange{absN1, fimm("126.0f"), bodyVL});
    // const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    mlir::Value u = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                          mlir::ValueRange{b, b, bodyVL});
    // const vfloat32m2_t j = vfmacc_vv(
    //     vfmul_vf(b, 0x1.ffffecp-1f),
    //     vfmacc_vv(
    //         vfmacc_vf(vfmv_v_f(0x1.fffdb6p-2f), 0x1.555e66p-3f, b),
    //         vfmacc_vf(vfmv_v_f(0x1.573e2ep-5f), 0x1.0e4020p-7f, b),
    //         u),
    //     u);
    mlir::Value jOuterA = vcall(
        f32VecType, "__riscv_vfmul_vf_f32m2",
        mlir::ValueRange{b, fimm("0x1.ffffecp-1f"), bodyVL});
    mlir::Value jc0 = vcall(f32VecType, "__riscv_vfmv_v_f_f32m2",
                            mlir::ValueRange{fimm("0x1.fffdb6p-2f"), bodyVL});
    mlir::Value jInnerA = vcall(
        f32VecType, "__riscv_vfmacc_vf_f32m2",
        mlir::ValueRange{jc0, fimm("0x1.555e66p-3f"), b, bodyVL});
    mlir::Value jc1 = vcall(f32VecType, "__riscv_vfmv_v_f_f32m2",
                            mlir::ValueRange{fimm("0x1.573e2ep-5f"), bodyVL});
    mlir::Value jInnerB = vcall(
        f32VecType, "__riscv_vfmacc_vf_f32m2",
        mlir::ValueRange{jc1, fimm("0x1.0e4020p-7f"), b, bodyVL});
    mlir::Value jMid = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                             mlir::ValueRange{jInnerA, jInnerB, u, bodyVL});
    mlir::Value j = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                          mlir::ValueRange{jOuterA, jMid, u, bodyVL});

    // Slow path emitted UNCONDITIONALLY (the vcpop short-circuit is a pure perf
    // branch; the slow path's c-false/|n|<=192 lanes are bitwise-equal to the
    // fast path k + j*k). vec.h:1348-1359.
    // const vbool16_t dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    mlir::Value dm = vcall(boolType, "__riscv_vmfle_vf_f32m2_b16",
                           mlir::ValueRange{n, fimm("0.0f"), bodyVL});
    // const vuint32m2_t d = vmerge_vxm(vmv_v_x(0, vl), 0x82000000, dm, vl);
    mlir::Value dZero =
        vcall(u32VecType, "__riscv_vmv_v_x_u32m2",
              mlir::ValueRange{uimm("0"), bodyVL});
    mlir::Value d =
        vcall(u32VecType, "__riscv_vmerge_vxm_u32m2",
              mlir::ValueRange{dZero, uimm("0x82000000"), dm, bodyVL});
    // const vfloat32m2_t s1 = vreinterpret(vadd_vx_u32m2(d, 0x7f000000, vl));
    mlir::Value s1Bits = vcall(u32VecType, "__riscv_vadd_vx_u32m2",
                               mlir::ValueRange{d, uimm("0x7f000000"), bodyVL});
    mlir::Value s1 = vcall(f32VecType, "__riscv_vreinterpret_v_u32m2_f32m2",
                           mlir::ValueRange{s1Bits});
    // const vfloat32m2_t s2 = vreinterpret(vsub_vv_u32m2(e, d, vl));
    mlir::Value s2Bits = vcall(u32VecType, "__riscv_vsub_vv_u32m2",
                               mlir::ValueRange{e, d, bodyVL});
    mlir::Value s2 = vcall(f32VecType, "__riscv_vreinterpret_v_u32m2_f32m2",
                           mlir::ValueRange{s2Bits});
    // const vfloat32m2_t r1 = vmerge_vvm(
    //     vfmacc_vv(k, k, j, vl),
    //     vfmul_vv(vfmacc_vv(s2, s2, j, vl), s1, vl),
    //     c, vl);
    mlir::Value r1False = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                                mlir::ValueRange{k, k, j, bodyVL});
    mlir::Value r1TrueInner = vcall(f32VecType, "__riscv_vfmacc_vv_f32m2",
                                    mlir::ValueRange{s2, s2, j, bodyVL});
    mlir::Value r1True = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                               mlir::ValueRange{r1TrueInner, s1, bodyVL});
    mlir::Value r1 = vcall(f32VecType, "__riscv_vmerge_vvm_f32m2",
                           mlir::ValueRange{r1False, r1True, c, bodyVL});
    // return vmerge_vvm(r1, vfmul_vv(s1, s1, vl),
    //                   vmfgt_vf(vfabs_v(n), 192.0f, vl), vl);
    mlir::Value absN2 = vcall(f32VecType, "__riscv_vfabs_v_f32m2",
                              mlir::ValueRange{n, bodyVL});
    mlir::Value overMask = vcall(boolType, "__riscv_vmfgt_vf_f32m2_b16",
                                 mlir::ValueRange{absN2, fimm("192.0f"),
                                                  bodyVL});
    mlir::Value s1Sq = vcall(f32VecType, "__riscv_vfmul_vv_f32m2",
                             mlir::ValueRange{s1, s1, bodyVL});
    return vcall(f32VecType, "__riscv_vmerge_vvm_f32m2",
                 mlir::ValueRange{r1, s1Sq, overMask, bodyVL});
  }

  /// Emit the COMPLETE ggml ggml_vec_silu_f32 forward-pass op (the FFN
  /// activation y[i] = x[i]*sigmoid(x[i]), sigmoid(x) = 1/(1+e^{-x});
  /// vec.cpp:380 + ggml_v_silu_m2 vec.h:1363) as fully STRUCTURED emitc nodes
  /// (I5; no verbatim C-string blob -- every value is a node):
  ///   for (size_t i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e32m2(n - i);
  ///     vfloat32m2_t vx = __riscv_vle32_v_f32m2(x + i, vl);
  ///     vfloat32m2_t vy = ggml_v_silu_m2(vx, vl);     // expanded node-for-node
  ///     __riscv_vse32_v_f32m2(y + i, vy, vl);
  ///   }
  /// silu = vfneg(x) -> ggml_v_expf_m2 -> vfadd 1.0f -> vfdiv(x, 1+exp).
  ///
  /// BYTE-EXACTNESS to ggml's REAL vectorized silu hinges on replicating
  /// ggml_v_expf_m2 (vec.h:1324-1360) node-for-node -- a fully vectorized minimax
  /// exp polynomial built ENTIRELY from __riscv_v intrinsics (NO libm expf). Each
  /// intrinsic is one emitc.call_opaque node (the one sanctioned opaque seam, as
  /// the dot kernels emit theirs) with the IDENTICAL magic-constant bit patterns
  /// (0x1.8p23f, 0x1.715476p+0f, 0x1.62e4p-1f, 0x1.7f7d1cp-20f, the degree-5
  /// polynomial coefficients, 0x3f800000, 0x82000000/0x7f000000/126.0f/192.0f).
  /// ggml's `if (!vcpop_m(c))` is a pure performance short-circuit -- the fast
  /// path k + j*k equals the slow path's c-false lane k + k*j bit-for-bit (fp
  /// multiply commutes; RISC-V yields the canonical NaN regardless of operand
  /// order) -- so the slow-path vmerge value graph is emitted UNCONDITIONALLY as
  /// a straight-line chain: same output bits for every input (normal, saturating
  /// tails, NaN/inf/denormal), no data-dependent branch. Pinned at m2 (matching
  /// ggml's vsetvl_e32m2 path and the m2-tied vbool16_t/vuint32m2_t types).
  mlir::LogicalResult emitGgmlVecSiluF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlVecSiluF32Op siluOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto s = llvm::dyn_cast<tcrvrvv::GgmlVecSiluF32Op>(op))
        siluOp = s;
    }
    if (!siluOp)
      return rewriter.notifyMatchFailure(scope, "silu body missing the op");

    mlir::Value input = valueMap.lookup(siluOp.getInput());
    mlir::Value outputBuf = valueMap.lookup(siluOp.getOutput());
    if (!input || !outputBuf)
      return rewriter.notifyMatchFailure(siluOp, "silu ABI operand unmapped");

    llvm::StringRef opName = siluOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = siluOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = outputBuf.getType();

    // The strip loop is fixed at m2 (ggml's vsetvl_e32m2 path); the exp
    // polynomial's m2-tied mask/reinterpret types live in emitGgmlVExpfM2.
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // Small helpers: a structured emitc.call_opaque node, a literal float
    // immediate (rendered as the exact C hex-float token).
    auto vcall = [&](mlir::Type resTy, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resTy}, callee, args)
          .getResult(0);
    };
    auto fimm = [&](llvm::StringRef tok) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, floatType, tok);
    };

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m2(n).
    mlir::Value vlmax =
        vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{avlArg});

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // size_t vl = __riscv_vsetvl_e32m2(n - i);
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{remaining});

      // const float *xp = x + i;  float *yp = y + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, inductionVar);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();
      mlir::Value yPtr = rewriter.create<emitc::AddOp>(loc, outputPtrType,
                                                       outputBuf, inductionVar);
      mlir::Value yStorePtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();

      // vfloat32m2_t vx = __riscv_vle32_v_f32m2(xp, vl);
      mlir::Value vx = vcall(f32VecType, "__riscv_vle32_v_f32m2",
                             mlir::ValueRange{xLoadPtr, bodyVL});

      // ===== ggml_v_silu_m2(vx, vl): neg -> ggml_v_expf_m2 -> +1 -> div =====
      // const vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
      mlir::Value negX = vcall(f32VecType, "__riscv_vfneg_v_f32m2",
                               mlir::ValueRange{vx, bodyVL});

      // exp(neg_x) via the SHARED node-for-node ggml_v_expf_m2 replication
      // (vec.h:1324) -- the IDENTICAL polynomial chain soft_max (F5b) consumes.
      mlir::Value expNegX =
          emitGgmlVExpfM2(rewriter, loc, negX, bodyVL, sizeType, opName, role);

      // const vfloat32m2_t one_plus = __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
      mlir::Value onePlus =
          vcall(f32VecType, "__riscv_vfadd_vf_f32m2",
                mlir::ValueRange{expNegX, fimm("1.0f"), bodyVL});
      // vfloat32m2_t vy = __riscv_vfdiv_vv_f32m2(x, one_plus, vl);
      mlir::Value vy = vcall(f32VecType, "__riscv_vfdiv_vv_f32m2",
                             mlir::ValueRange{vx, onePlus, bodyVL});

      // __riscv_vse32_v_f32m2(yp, vy, vl);
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "__riscv_vse32_v_f32m2"));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, "__riscv_vse32_v_f32m2",
          mlir::ValueRange{yStorePtr, vy, bodyVL});
    }

    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_vec_soft_max_f32 forward-pass op (F5b: the
  /// attention softmax core y[i] = e^{x[i]-max}, RETURNING the f64 sum
  /// Sum_i e^{x[i]-max}; vec.cpp:531 + the __riscv_v path vec.cpp:584-592) as
  /// fully STRUCTURED emitc nodes (I5; no verbatim C blob -- every value is a
  /// node). Faithful to ggml's BARE function: `max` is an INPUT, no normalize.
  ///   vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);
  ///   for (size_t i = 0; i < n; i += vlmax) {
  ///     size_t vl = __riscv_vsetvl_e32m2(n - i);
  ///     vfloat32m2_t val = ggml_v_expf_m2(
  ///         __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(x + i, vl), max, vl),
  ///         vl);                                   // shared exp polynomial
  ///     __riscv_vse32_v_f32m2(y + i, val, vl);
  ///     vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);  // f32->f64
  ///   }
  ///   return (double)__riscv_vfmv_f_s_f64m1_f64(vsum);
  ///
  /// The sum is the byte-exactness crux: ggml accumulates in `ggml_float` =
  /// DOUBLE via the WIDENING reduce vfwredusum_vs_f32m2_f64m1 into a SINGLE f64m1
  /// accumulator carried across strips (NOT F3's scalar-ascending fold). The
  /// f64m1 accumulator is a loop-carried opaque-vector lvalue (emitc.for has no
  /// iter_args), exactly as F3 carries its scalar double `sum`, but the value
  /// type is the opaque vector vfloat64m1_t. exp(x-max) reuses the SHARED
  /// node-for-node ggml_v_expf_m2 chain (emitGgmlVExpfM2), so y[] and each val
  /// are bit-identical to ggml's silu/soft_max. Returns the f64 sum value (the
  /// dispatch wraps it in the function's `return`).
  mlir::FailureOr<mlir::Value> emitGgmlVecSoftMaxF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlVecSoftMaxF32Op softMaxOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto s = llvm::dyn_cast<tcrvrvv::GgmlVecSoftMaxF32Op>(op))
        softMaxOp = s;
    }
    if (!softMaxOp)
      return rewriter.notifyMatchFailure(scope, "soft_max body missing the op");

    mlir::Value outputBuf = valueMap.lookup(softMaxOp.getOutput());
    mlir::Value input = valueMap.lookup(softMaxOp.getInput());
    mlir::Value maxArg = valueMap.lookup(softMaxOp.getMax());
    if (!outputBuf || !input || !maxArg)
      return rewriter.notifyMatchFailure(softMaxOp,
                                         "soft_max ABI operand unmapped");

    llvm::StringRef opName = softMaxOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = softMaxOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = outputBuf.getType();

    // The strip loop is fixed at m2 (ggml's vsetvl_e32m2 path); the f64 reduce
    // accumulator is f64m1 (ggml's vfwredusum_vs_f32m2_f64m1 destination).
    mlir::Type f32VecType = emitc::OpaqueType::get(ctx, "vfloat32m2_t");
    mlir::Type f64VecType = emitc::OpaqueType::get(ctx, "vfloat64m1_t");
    mlir::Type doubleType = emitc::OpaqueType::get(ctx, "double");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    auto vcall = [&](mlir::Type resTy, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resTy}, callee, args)
          .getResult(0);
    };

    // vfloat64m1_t vsum = __riscv_vfmv_v_f_f64m1(0, 1);  -- the SCALAR f64
    // accumulator (ggml_float = double) carried across strips as a loop-carried
    // opaque-vector lvalue (emitc.for has no iter_args, exactly as F3 carries its
    // scalar double sum). ggml seeds it with vl=1.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("vsum", opName, role));
    auto vsumVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(f64VecType),
        emitc::OpaqueAttr::get(ctx, ""));
    mlir::Value zeroDouble =
        rewriter.create<emitc::LiteralOp>(loc, doubleType, "0.0");
    mlir::Value oneVL = rewriter.create<emitc::LiteralOp>(loc, sizeType, "1");
    mlir::Value vsumInit = vcall(f64VecType, "__riscv_vfmv_v_f_f64m1",
                                 mlir::ValueRange{zeroDouble, oneVL});
    rewriter.create<emitc::AssignOp>(loc, vsumVar, vsumInit);

    // Pre-loop full-chunk VLMAX: size_t vlmax = __riscv_vsetvl_e32m2(n).
    mlir::Value vlmax =
        vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{avlArg});

    // for (size_t i = 0; i < n; i += vlmax) { ... }
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    auto forOp = rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax,
                                               /*bodyBuilder=*/nullptr);
    mlir::Value inductionVar = forOp.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(forOp.getBody());

      // size_t vl = __riscv_vsetvl_e32m2(n - i);
      mlir::Value remaining =
          rewriter.create<emitc::SubOp>(loc, sizeType, avlArg, inductionVar);
      mlir::Value bodyVL =
          vcall(sizeType, "__riscv_vsetvl_e32m2", mlir::ValueRange{remaining});

      // const float *xp = x + i;  float *yp = y + i;
      mlir::Value xPtr =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, inductionVar);
      mlir::Value xLoadPtr =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xPtr)
              .getResult();
      mlir::Value yPtr = rewriter.create<emitc::AddOp>(loc, outputPtrType,
                                                       outputBuf, inductionVar);
      mlir::Value yStorePtr =
          rewriter.create<emitc::CastOp>(loc, floatPtrType, yPtr).getResult();

      // vfloat32m2_t vx = __riscv_vle32_v_f32m2(xp, vl);
      mlir::Value vx = vcall(f32VecType, "__riscv_vle32_v_f32m2",
                             mlir::ValueRange{xLoadPtr, bodyVL});

      // vfloat32m2_t sub = __riscv_vfsub_vf_f32m2(vx, max, vl);  -- x[i]-max
      // (with x=-inf giving -inf, which the exp polynomial flushes to 0).
      mlir::Value sub = vcall(f32VecType, "__riscv_vfsub_vf_f32m2",
                              mlir::ValueRange{vx, maxArg, bodyVL});

      // vfloat32m2_t val = ggml_v_expf_m2(sub, vl) -- the SHARED node-for-node
      // exp polynomial (vec.h:1324), bit-identical to ggml's silu/soft_max.
      mlir::Value val =
          emitGgmlVExpfM2(rewriter, loc, sub, bodyVL, sizeType, opName, role);

      // __riscv_vse32_v_f32m2(yp, val, vl);   -- write y[i] = e^{x[i]-max}.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "__riscv_vse32_v_f32m2"));
      rewriter.create<emitc::CallOpaqueOp>(
          loc, mlir::TypeRange{}, "__riscv_vse32_v_f32m2",
          mlir::ValueRange{yStorePtr, val, bodyVL});

      // vsum = __riscv_vfwredusum_vs_f32m2_f64m1(val, vsum, vl);  -- the WIDENING
      // f32->f64 reduce into the loop-carried f64m1 accumulator (ggml's EXACT
      // fold; the byte-exactness crux for the returned sum). Load the current
      // vsum, fold, assign back.
      mlir::Value vsumCur =
          rewriter.create<emitc::LoadOp>(loc, f64VecType, vsumVar).getResult();
      mlir::Value vsumNext =
          vcall(f64VecType, "__riscv_vfwredusum_vs_f32m2_f64m1",
                mlir::ValueRange{val, vsumCur, bodyVL});
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("vsum", opName, role));
      rewriter.create<emitc::AssignOp>(loc, vsumVar, vsumNext);
    }

    // return (double)__riscv_vfmv_f_s_f64m1_f64(vsum);  -- extract the f64 lane-0
    // scalar (ggml's `return (ggml_float)__riscv_vfmv_f_s_f64m1_f64(vsum)`,
    // vec.cpp:592). The intrinsic already yields a `double`; the explicit cast
    // mirrors ggml's `(ggml_float)` and keeps the return-type contract.
    mlir::Value vsumFinal =
        rewriter.create<emitc::LoadOp>(loc, f64VecType, vsumVar).getResult();
    mlir::Value sumScalar = vcall(doubleType, "__riscv_vfmv_f_s_f64m1_f64",
                                  mlir::ValueRange{vsumFinal});
    rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "return"));
    mlir::Value sumReturn =
        rewriter.create<emitc::CastOp>(loc, doubleType, sumScalar).getResult();
    return sumReturn;
  }

  /// Emit the COMPLETE ggml `quantize_row_q8_0` RVV-path forward-pass op (the F4
  /// f32 -> block_q8_0 activation quantizer; riscv/quants.c:32-71) for one
  /// tcrv_rvv.quantize_row_q8_0 op as fully STRUCTURED emitc nodes (I5; no
  /// verbatim C-string blob -- every value is a node in the IR graph):
  ///   size_t nb = n / 32;
  ///   for (size_t ib = 0; ib < nb; ib += 1) {
  ///     size_t vl = 32;                                    // = QK8_0 (one strip)
  ///     const float *xb = x + ib*32;
  ///     uint8_t *yb = vy + ib*34;                          // AoS block cursor
  ///     vfloat32m8_t v_x  = __riscv_vle32_v_f32m8(xb, vl);
  ///     vfloat32m8_t vabs = __riscv_vfabs_v_f32m8(v_x, vl);
  ///     vfloat32m1_t tmp  = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  ///     vfloat32m1_t vmax = __riscv_vfredmax_vs_f32m8_f32m1(vabs, tmp, vl);
  ///     float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);
  ///     float d  = amax / 127.0f;                          // scalar f32
  ///     float id = 0.0f; if (d != 0.0f) { id = 1.0f / d; } // d ? 1/d : 0
  ///     *(_Float16 *)(yb + 0) = (_Float16)d;               // fcvt.h.s (rne)
  ///     vfloat32m8_t x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);
  ///     vint16m4_t vi = __riscv_vfncvt_x_f_w_i16m4(x0, vl);  // f32->i16 (rne)
  ///     vint8m2_t  vs = __riscv_vncvt_x_x_w_i8m2(vi, vl);    // i16->i8 truncate
  ///     __riscv_vse8_v_i8m2(yb + 2, vs, vl);                // the 32 int8 qs
  ///   }
  /// BYTE-EXACTNESS to the DEPLOYED kernel matches ggml's EXACT RVV method, NOT
  /// the scalar `_ref` (which rounds with `roundf`, round-half-AWAY). The two
  /// cruxes: (1) vfncvt_x_f_w_i16m4 rounds with the dynamic frm = round-to-
  /// nearest-EVEN -- replicating that exact intrinsic inherits rne + the i8
  /// saturating clamp + every edge case (do NOT reason about rounding); (2) the
  /// board is __riscv_zfhmin, so GGML_CPU_FP32_TO_FP16(d) is the native
  /// (_Float16)d cast (fcvt.h.s, rne) -- a STRUCTURAL _Float16 store, not a
  /// software fp16 pack. The `id = d ? 1/d : 0` conditional is load-bearing (the
  /// all-zero block: amax=0 => d=0 must give id=0 so every q=0; a bare 1/d gives
  /// inf/NaN) -- emitted as a STRUCTURED emitc.cmp + emitc.if, NOT a raw string.
  /// The block-format facts (qk/stride/offsets) are the op's typed attrs (I4);
  /// the emission is the op's fixed structure. vl is hard-pinned to QK8_0=32 (one
  /// e32m8 strip per block, relying on Zvl128b => VLEN>=128, the same capability
  /// the q4_0 mb4-elided shape uses) -- replicating ggml's `size_t vl = QK8_0`.
  /// The intrinsic spellings are HARD-CODED callees (vfabs/vfredmax/vfncvt/vncvt),
  /// matching ggml's exact path -- never synthesized (a _rm/_tu suffix would
  /// change the rounding mode or fail to compile).
  mlir::LogicalResult emitGgmlQuantizeRowQ80(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlQuantizeRowQ80Op quantOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto q = llvm::dyn_cast<tcrvrvv::GgmlQuantizeRowQ80Op>(op))
        quantOp = q;
    }
    if (!quantOp)
      return rewriter.notifyMatchFailure(scope, "quantize body missing the op");

    mlir::Value input = valueMap.lookup(quantOp.getInput());
    mlir::Value output = valueMap.lookup(quantOp.getOutput());
    if (!input || !output)
      return rewriter.notifyMatchFailure(quantOp,
                                         "quantize ABI operand unmapped");

    llvm::StringRef opName = quantOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = quantOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type boolType = rewriter.getI1Type();
    mlir::Type half16Type = emitc::OpaqueType::get(ctx, "_Float16");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type half16PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "_Float16"));
    mlir::Type i8PtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "int8_t"));
    // The block's 32 lanes ride in ONE e32m8 strip (ggml's `size_t vl = QK8_0`);
    // the wide f32 / m1 reduce / i16m4 / i8m2 types are ggml's exact path types.
    mlir::Type f32m8Type = emitc::OpaqueType::get(ctx, "vfloat32m8_t");
    mlir::Type f32m1Type = emitc::OpaqueType::get(ctx, "vfloat32m1_t");
    mlir::Type i16m4Type = emitc::OpaqueType::get(ctx, "vint16m4_t");
    mlir::Type i8m2Type = emitc::OpaqueType::get(ctx, "vint8m2_t");

    // The AoS block-format structural facts come straight off the typed attrs
    // (I4): qk=32 (block length / lanes), block_stride=34 (the AoS stride), the
    // fp16 d at byte 0, the 32 int8 qs at byte 2.
    int64_t qk = quantOp.getQk();
    int64_t blockStride = quantOp.getBlockStride();
    int64_t scaleOffset = quantOp.getScaleByteOffset();
    int64_t quantOffset = quantOp.getQuantByteOffset();

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };
    auto vcall = [&](mlir::Type resultType, llvm::StringRef callee,
                     mlir::ValueRange args) -> mlir::Value {
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, callee));
      return rewriter
          .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{resultType}, callee,
                                       args)
          .getResult(0);
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // size_t nb = n / 32;  (ggml's `const int nb = k / QK8_0`). n % 32 == 0 is a
    // ggml contract (no tail), exactly as the q4_0 block-dot requires.
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "block_count"));
    mlir::Value nb =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(qk));

    // for (size_t ib = 0; ib < nb; ib += 1) { ... }  -- the AoS block loop.
    mlir::Value zero = rewriter.create<emitc::LiteralOp>(loc, sizeType, "0");
    mlir::Value one = sizeLit(1);
    auto blockFor = rewriter.create<emitc::ForOp>(loc, zero, nb, one,
                                                  /*bodyBuilder=*/nullptr);
    mlir::Value ib = blockFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(blockFor.getBody());

      // size_t vl = 32;  (ggml hard-pins `size_t vl = QK8_0` and relies on
      // Zvl128b => VLEN>=128 so the e32m8 strip holds all 32 lanes at once: ONE
      // vfabs, ONE vfredmax, ONE vfncvt per block).
      mlir::Value vl = sizeLit(qk);

      // const float *xb = x + ib*32;  (the f32 source block).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "x_block"));
      mlir::Value xOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(qk));
      mlir::Value xbRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, xOff);
      mlir::Value xb =
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xbRaw)
              .getResult();

      // uint8_t *yb = vy + ib*34;  (the AoS block_q8_0 byte cursor).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "y_block"));
      mlir::Value yOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, ib, sizeLit(blockStride));
      mlir::Value yb =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, yOff);

      // v_x = __riscv_vle32_v_f32m8(xb, vl);
      mlir::Value vx =
          vcall(f32m8Type, "__riscv_vle32_v_f32m8", mlir::ValueRange{xb, vl});

      // The amax reduction (ggml riscv/quants.c:47-50): vfabs -> vfredmax seeded
      // with a 0.0f f32m1 -> extract lane 0. The intrinsic callees are HARD-CODED
      // (ggml's exact spellings) -- never synthesized.
      mlir::Value vabs =
          vcall(f32m8Type, "__riscv_vfabs_v_f32m8", mlir::ValueRange{vx, vl});
      mlir::Value zeroF =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f");
      mlir::Value redSeed = vcall(f32m1Type, "__riscv_vfmv_v_f_f32m1",
                                  mlir::ValueRange{zeroF, vl});
      mlir::Value vmax =
          vcall(f32m1Type, "__riscv_vfredmax_vs_f32m8_f32m1",
                mlir::ValueRange{vabs, redSeed, vl});
      mlir::Value amax = vcall(floatType, "__riscv_vfmv_f_s_f32m1_f32",
                               mlir::ValueRange{vmax});

      // float d = amax / 127.0f;  (ggml's `amax / ((1 << 7) - 1)`; the divisor
      // is the f32 literal 127.0f so the divide is a single f32 round).
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "d"));
      mlir::Value c127 =
          rewriter.create<emitc::LiteralOp>(loc, floatType, "127.0f");
      mlir::Value d = rewriter.create<emitc::DivOp>(loc, floatType, amax, c127);

      // float id = 0.0f; if (d != 0.0f) { id = 1.0f / d; }  -- the load-bearing
      // `id = d ? 1.0f/d : 0.0f` conditional, STRUCTURED (emitc.cmp + emitc.if,
      // NOT a raw string). The all-zero block (amax=0 => d=0) takes the else and
      // keeps id=0, so every q=0 (a bare 1/d would give inf, then 0*inf=NaN).
      rewriter.create<emitc::VerbatimOp>(
          loc, localVariableComment("id", opName, role));
      auto idVar = rewriter.create<emitc::VariableOp>(
          loc, emitc::LValueType::get(floatType),
          emitc::OpaqueAttr::get(ctx, ""));
      rewriter.create<emitc::AssignOp>(
          loc, idVar,
          rewriter.create<emitc::LiteralOp>(loc, floatType, "0.0f"));
      mlir::Value dNonZero = rewriter.create<emitc::CmpOp>(
          loc, boolType, emitc::CmpPredicate::ne, d, zeroF);
      auto idIf = rewriter.create<emitc::IfOp>(loc, dNonZero,
                                               /*addThenBlock=*/true,
                                               /*addElseBlock=*/false);
      {
        mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
        rewriter.setInsertionPointToStart(&idIf.getThenRegion().front());
        mlir::Value oneF =
            rewriter.create<emitc::LiteralOp>(loc, floatType, "1.0f");
        mlir::Value recip =
            rewriter.create<emitc::DivOp>(loc, floatType, oneF, d);
        rewriter.create<emitc::AssignOp>(loc, idVar, recip);
        rewriter.create<emitc::YieldOp>(loc);
      }
      mlir::Value id =
          rewriter.create<emitc::LoadOp>(loc, floatType, idVar).getResult();

      // *(_Float16 *)(yb + 0) = (_Float16)d;  -- the fp16 d store. The board is
      // __riscv_zfhmin, so GGML_CPU_FP32_TO_FP16(d) is the native (_Float16)d
      // cast (fcvt.h.s, rne). STRUCTURED: cast the byte cursor to _Float16 *,
      // subscript [0] (an lvalue), cast d to _Float16, assign.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "fp16_d_store"));
      mlir::Value dPtrRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                        sizeLit(scaleOffset));
      auto dPtr = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, half16PtrType, dPtrRaw)
              .getResult());
      mlir::Value dIndex =
          rewriter.create<emitc::LiteralOp>(loc, rewriter.getIndexType(), "0");
      emitc::SubscriptOp dSubscript =
          rewriter.create<emitc::SubscriptOp>(loc, dPtr, dIndex);
      mlir::Value dHalf =
          rewriter.create<emitc::CastOp>(loc, half16Type, d).getResult();
      rewriter.create<emitc::AssignOp>(loc, dSubscript.getResult(), dHalf);

      // x0 = __riscv_vfmul_vf_f32m8(v_x, id, vl);  -- scale every lane by id.
      mlir::Value x0 = vcall(f32m8Type, "__riscv_vfmul_vf_f32m8",
                             mlir::ValueRange{vx, id, vl});

      // The NARROWING CONVERT (ggml riscv/quants.c:60-61): f32 -> i16 (the
      // rounding crux: vfncvt_x_f_w_i16m4 = dynamic frm = round-to-nearest-EVEN),
      // then i16 -> i8 truncate (vncvt). Both callees are ggml's exact spellings.
      mlir::Value vi = vcall(i16m4Type, "__riscv_vfncvt_x_f_w_i16m4",
                             mlir::ValueRange{x0, vl});
      mlir::Value vs = vcall(i8m2Type, "__riscv_vncvt_x_x_w_i8m2",
                             mlir::ValueRange{vi, vl});

      // __riscv_vse8_v_i8m2(yb + 2, vs, vl);  -- store the 32 int8 qs.
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "qs_store"));
      mlir::Value qsPtrRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, yb,
                                        sizeLit(quantOffset));
      mlir::Value qsPtr =
          rewriter.create<emitc::CastOp>(loc, i8PtrType, qsPtrRaw).getResult();
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "__riscv_vse8_v_i8m2"));
      rewriter.create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{},
                                           "__riscv_vse8_v_i8m2",
                                           mlir::ValueRange{qsPtr, vs, vl});
    }

    return mlir::success();
  }

  /// Emit the COMPLETE ggml ggml_compute_forward_rope_f32 op for ONE head row,
  /// the GGML_ROPE_TYPE_NORMAL variant (the rope llama-2 / LLM_ARCH_LLAMA uses),
  /// as fully STRUCTURED emitc nodes (I5; no verbatim C-string blob):
  ///   float theta = theta_base;                        // the recurrence seed
  ///   size_t n_pairs = n_dims / 2;
  ///   for (size_t p = 0; p < n_pairs; ++p) {           // SCALAR per-pair loop
  ///     float cos_t = cosf(theta);  float sin_t = sinf(theta);  // scalar libm
  ///     const float *xp = (const float *)(x + 2*p);
  ///     float *yp = (float *)(y + 2*p);
  ///     float x0 = xp[0];  float x1 = xp[1];           // CONSECUTIVE pair
  ///     yp[0] = x0*cos_t - x1*sin_t;                   // two SEPARATE muls then
  ///     yp[1] = x0*sin_t + x1*cos_t;                   //   a sub/add (no FMA)
  ///     theta = theta * theta_scale;                   // iterative f32 recurrence
  ///   }
  /// BYTE-EXACTNESS has TWO axes (both stated honestly in the result):
  /// (1) cosf/sinf are SCALAR libm (one emitc.call_opaque each -- the sanctioned
  ///     opaque seam, NOT a raw string), so the angles are bit-exact vs ggml's
  ///     cache[] only when the SAME libm is linked (a libm-tolerance otherwise).
  ///     This differs from F5's exp (a vectorized polynomial we replicate); rope
  ///     is NOT a vectorized-transcendental problem.
  /// (2) The rotation x0*cos - x1*sin is a*b - c*d, an FP-contraction hazard.
  ///     ggml's rotation is a SINGLE C expression (ops.cpp:5808-5809), so each
  ///     output's a*b - c*d is GROUPED into ONE emitc.expression here (the F3
  ///     rms_norm emitc.expression FMA-fix discipline) -> ONE C statement
  ///     token-identical to ggml's -> clang makes the IDENTICAL contraction
  ///     decision under EVERY -ffp-contract mode, so the kernel is byte-exact vs
  ///     ggml under default/on/off/fast (NOT just off), independent of the build
  ///     flag. Separate-statement products would block intra-statement fusion and
  ///     diverge from ggml under the default `on`.
  /// theta_base (= pos as f32) and theta_scale (= powf(freq_base, -2/n_dims)) are
  /// PRECOMPUTED runtime f32 inputs, so the kernel makes no powf call -- the only
  /// libm calls are the per-pair cosf/sinf. The loop is SCALAR (not vectorized):
  /// cos/sin are scalar libm so the faithful structure IS ggml's scalar per-pair
  /// rotation (vectorizing would gather scalars into vectors with no exactness
  /// gain). The recurrence theta is a loop-carried emitc.variable lvalue +
  /// emitc.assign (emitc.for has no iter_args), exactly as F3's scalar-double sum.
  mlir::LogicalResult emitGgmlRopeNormF32(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::WithVLOp scope, mlir::Value avlArg, mlir::Type sizeType,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap) const {
    tcrvrvv::GgmlRopeNormF32Op ropeOp;
    for (mlir::Operation &op : scope.getBody().front()) {
      if (auto r = llvm::dyn_cast<tcrvrvv::GgmlRopeNormF32Op>(op))
        ropeOp = r;
    }
    if (!ropeOp)
      return rewriter.notifyMatchFailure(scope, "rope body missing the op");

    mlir::Value input = valueMap.lookup(ropeOp.getInput());
    mlir::Value output = valueMap.lookup(ropeOp.getOutput());
    mlir::Value thetaBase = valueMap.lookup(ropeOp.getThetaBase());
    mlir::Value thetaScale = valueMap.lookup(ropeOp.getThetaScale());
    if (!input || !output || !thetaBase || !thetaScale)
      return rewriter.notifyMatchFailure(ropeOp, "rope ABI operand unmapped");

    llvm::StringRef opName = ropeOp.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = ropeOp.getTCRVEmitCLowerableSourceRole();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type inputPtrType = input.getType();
    mlir::Type outputPtrType = output.getType();
    mlir::Type floatType = emitc::OpaqueType::get(ctx, "float");
    mlir::Type constFloatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "const float"));
    mlir::Type floatPtrType =
        emitc::PointerType::get(emitc::OpaqueType::get(ctx, "float"));
    mlir::Type indexType = rewriter.getIndexType();

    auto sizeLit = [&](int64_t v) -> mlir::Value {
      return rewriter.create<emitc::LiteralOp>(loc, sizeType, std::to_string(v));
    };

    rewriter.create<emitc::VerbatimOp>(loc, routeSourceComment(opName, role));

    // float theta = theta_base;  -- the iterative f32 angle recurrence seed
    // (ggml's `float theta = theta_base;` ops.cpp:5711). emitc.for has no
    // iter_args, so the loop-carried theta is an emitc.variable lvalue +
    // emitc.assign, exactly as F3 (rms_norm) carries its scalar-double sum.
    rewriter.create<emitc::VerbatimOp>(
        loc, localVariableComment("theta", opName, role));
    auto thetaVar = rewriter.create<emitc::VariableOp>(
        loc, emitc::LValueType::get(floatType), emitc::OpaqueAttr::get(ctx, ""));
    rewriter.create<emitc::AssignOp>(loc, thetaVar, thetaBase);

    // size_t n_pairs = n_dims / 2;  (ggml steps i0 by 2 over [0, ne0)).
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(opName, role, "pair_count"));
    mlir::Value nPairs =
        rewriter.create<emitc::DivOp>(loc, sizeType, avlArg, sizeLit(2));

    // for (size_t p = 0; p < n_pairs; p += 1) { ... }  -- the SCALAR per-pair
    // rotation loop. NOT vectorized: cos/sin are scalar libm (one call per pair),
    // so the faithful structure IS ggml's scalar loop.
    mlir::Value zero = sizeLit(0);
    mlir::Value one = sizeLit(1);
    auto pairFor = rewriter.create<emitc::ForOp>(loc, zero, nPairs, one,
                                                 /*bodyBuilder=*/nullptr);
    mlir::Value p = pairFor.getInductionVar();
    {
      mlir::OpBuilder::InsertionGuard bodyGuard(rewriter);
      rewriter.setInsertionPointToStart(pairFor.getBody());

      // float theta_p = theta;  (read the loop-carried recurrence value).
      mlir::Value thetaCur =
          rewriter.create<emitc::LoadOp>(loc, floatType, thetaVar).getResult();

      // float cos_t = cosf(theta_p);  float sin_t = sinf(theta_p);  -- the SCALAR
      // libm angle cache (ggml's rope_yarn cosf/sinf, ops.cpp:5703-5704). Each is
      // ONE emitc.call_opaque (the sanctioned opaque seam) -- the byte-exactness
      // axis that depends on linking the SAME libm ggml links (NOT a raw string,
      // NOT a vectorized polynomial).
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "cosf"));
      mlir::Value cosT =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           "cosf", mlir::ValueRange{thetaCur})
              .getResult(0);
      rewriter.create<emitc::VerbatimOp>(loc, stepComment(opName, role, "sinf"));
      mlir::Value sinT =
          rewriter
              .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{floatType},
                                           "sinf", mlir::ValueRange{thetaCur})
              .getResult(0);

      // const float *xp = (const float *)(x + 2*p);  float *yp = (float *)(y+2*p)
      // -- the CONSECUTIVE pair (NORMAL: ic = i0, x0=x[2p], x1=x[2p+1]).
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "pair_ptr"));
      mlir::Value pairOff =
          rewriter.create<emitc::MulOp>(loc, sizeType, p, sizeLit(2));
      mlir::Value xpRaw =
          rewriter.create<emitc::AddOp>(loc, inputPtrType, input, pairOff);
      auto xp = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, constFloatPtrType, xpRaw)
              .getResult());
      mlir::Value ypRaw =
          rewriter.create<emitc::AddOp>(loc, outputPtrType, output, pairOff);
      auto yp = llvm::cast<mlir::TypedValue<emitc::PointerType>>(
          rewriter.create<emitc::CastOp>(loc, floatPtrType, ypRaw).getResult());

      // float x0 = xp[0];  float x1 = xp[1];  -- the consecutive pair loads.
      mlir::Value idx0 = rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
      mlir::Value idx1 = rewriter.create<emitc::LiteralOp>(loc, indexType, "1");
      // The subscript on a `const float *` yields an lvalue of `const float`; the
      // load must take that exact value type (a fixed `float` mismatches the
      // emitc.load verifier). The loaded scalar then participates as a plain f32.
      emitc::SubscriptOp x0Sub =
          rewriter.create<emitc::SubscriptOp>(loc, xp, idx0);
      auto xLValueType =
          llvm::cast<emitc::LValueType>(x0Sub.getResult().getType());
      mlir::Value x0 =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     x0Sub.getResult())
              .getResult();
      emitc::SubscriptOp x1Sub =
          rewriter.create<emitc::SubscriptOp>(loc, xp, idx1);
      mlir::Value x1 =
          rewriter
              .create<emitc::LoadOp>(loc, xLValueType.getValueType(),
                                     x1Sub.getResult())
              .getResult();

      // The ROTATION. Each output's `a*b - c*d` / `a*b + c*d` is grouped into ONE
      // emitc.expression, so mlir-translate renders it as ONE C statement
      // TOKEN-IDENTICAL to ggml's source (ops.cpp:5808-5809 -- the rotation is a
      // single C expression there). Then clang makes the IDENTICAL contraction
      // decision under EVERY -ffp-contract mode (fuses under on/fast, two-rounding
      // under off) -- so the kernel is byte-exact vs ggml regardless of the build
      // flag, NOT only at off. Emitting the two products as separate statements
      // would block intra-statement fusion and diverge from ggml's fused form
      // under the default `on`. The subscript-loads and the cosf/sinf
      // call_opaque results stay OUTSIDE the expression (load/call_opaque lack the
      // CExpression trait). This mirrors the F3 rms_norm emitc.expression FMA fix.
      mlir::Value idx0Lit =
          rewriter.create<emitc::LiteralOp>(loc, indexType, "0");
      mlir::Value idx1Lit =
          rewriter.create<emitc::LiteralOp>(loc, indexType, "1");
      // yp[0] = x0*cos_t - x1*sin_t;  (one expression -> one C statement)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "rotate_lo"));
      auto loExpr = rewriter.create<emitc::ExpressionOp>(loc, floatType,
                                                         /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&loExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value x0cos =
            rewriter.create<emitc::MulOp>(loc, floatType, x0, cosT);
        mlir::Value x1sin =
            rewriter.create<emitc::MulOp>(loc, floatType, x1, sinT);
        mlir::Value y0 =
            rewriter.create<emitc::SubOp>(loc, floatType, x0cos, x1sin);
        rewriter.create<emitc::YieldOp>(loc, y0);
      }
      // yp is a DISTINCT subscript chain from xp (the loads addressed the const
      // xp; the stores address the mutable yp -- ggml writes dst, reads src,
      // which may alias but the consecutive pair is fully READ before written).
      emitc::SubscriptOp y0Sub =
          rewriter.create<emitc::SubscriptOp>(loc, yp, idx0Lit);
      rewriter.create<emitc::AssignOp>(loc, y0Sub.getResult(),
                                       loExpr.getResult());

      // yp[1] = x0*sin_t + x1*cos_t;  (one expression -> one C statement)
      rewriter.create<emitc::VerbatimOp>(
          loc, stepComment(opName, role, "rotate_hi"));
      auto hiExpr = rewriter.create<emitc::ExpressionOp>(loc, floatType,
                                                         /*do_not_inline=*/false);
      {
        mlir::OpBuilder::InsertionGuard exprGuard(rewriter);
        mlir::Block *exprBlock = rewriter.createBlock(&hiExpr.getRegion());
        rewriter.setInsertionPointToStart(exprBlock);
        mlir::Value x0sin =
            rewriter.create<emitc::MulOp>(loc, floatType, x0, sinT);
        mlir::Value x1cos =
            rewriter.create<emitc::MulOp>(loc, floatType, x1, cosT);
        mlir::Value y1 =
            rewriter.create<emitc::AddOp>(loc, floatType, x0sin, x1cos);
        rewriter.create<emitc::YieldOp>(loc, y1);
      }
      emitc::SubscriptOp y1Sub =
          rewriter.create<emitc::SubscriptOp>(loc, yp, idx1Lit);
      rewriter.create<emitc::AssignOp>(loc, y1Sub.getResult(),
                                       hiExpr.getResult());

      // theta = theta * theta_scale;  -- the iterative f32 recurrence step
      // (ggml's `theta *= theta_scale;` ops.cpp:5719).
      rewriter.create<emitc::VerbatimOp>(
          loc, assignComment("theta", opName, role));
      mlir::Value thetaNext =
          rewriter.create<emitc::MulOp>(loc, floatType, thetaCur, thetaScale);
      rewriter.create<emitc::AssignOp>(loc, thetaVar, thetaNext);
    }

    return mlir::success();
  }

  mlir::LogicalResult emitPackedI4OffsetBinaryXI8Product(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      tcrvrvv::PackedI4OffsetBinaryXI8ProductOp packed,
      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
      mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(packed.getResult().getType());
    auto srcVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(packed.getWeight().getType());
    if (!resultVecType || !srcVecType)
      return rewriter.notifyMatchFailure(
          packed, "offset-binary packed-i4 x i8 product types not vectors");
    mlir::Type resultEmitC = convertVectorTypeToEmitC(resultVecType);
    mlir::Type srcEmitC = convertVectorTypeToEmitC(srcVecType);
    if (!resultEmitC || !srcEmitC)
      return rewriter.notifyMatchFailure(
          packed, "offset-binary packed-i4 x i8 product type not convertible");
    mlir::Value weight = valueMap.lookup(packed.getWeight());
    mlir::Value actLow = valueMap.lookup(packed.getActivationLow());
    mlir::Value actHigh = valueMap.lookup(packed.getActivationHigh());
    if (!weight || !actLow || !actHigh)
      return rewriter.notifyMatchFailure(
          packed, "offset-binary packed-i4 x i8 product operand unmapped");
    llvm::StringRef opName = packed.getTCRVEmitCLowerableSourceOpName();
    llvm::StringRef role = packed.getTCRVEmitCLowerableSourceRole();
    mlir::FailureOr<mlir::Value> pairSum = emitOffsetBinaryDecodeProductValue(
        rewriter, loc, weight, actLow, actHigh, bodyVL, srcEmitC, resultEmitC,
        vectorDType(srcVecType), srcVecType.getLmul(),
        vectorElementWidth(resultVecType), resultVecType.getLmul(),
        vectorDType(resultVecType), opName, role);
    if (mlir::failed(pairSum))
      return mlir::failure();
    valueMap[packed.getResult()] = *pairSum;
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
  /// When `extraOffset` is set, a SECOND pointer add is emitted after the
  /// `base + i` add: ptr2 = ptr + extraOffset (mirroring the legacy unrolled
  /// `v23 = base + i; v24 = v23 + priorVL` two-add form for the second u2 slice).
  /// Existing single-slice callers pass the default null and emit one add.
  mlir::LogicalResult
  emitStore(mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
            tcrvrvv::StoreOp store,
            llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
            mlir::Value inductionVar, mlir::Value storeVL,
            mlir::Value extraOffset = {}) const {
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
    if (extraOffset)
      ptr = rewriter.create<emitc::AddOp>(loc, base.getType(), ptr, extraOffset);
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
    mlir::Value source = valueMap.lookup(dequantize.getSource());
    mlir::Value scale = valueMap.lookup(dequantize.getScale());
    if (!source || !scale)
      return rewriter.notifyMatchFailure(dequantize,
                                         "dequantize operand unmapped");
    return emitDequantizeChain(rewriter, loc, dequantize, source, scale,
                               valueMap, bodyVL);
  }

  /// The bare vfcvt+vfmul dequant chain shared by the clamp epilogue
  /// (`emitDequantize`) and the standalone load->dequantize->store body
  /// (`emitStandaloneDequantBody`):
  ///   converted = __riscv_vfcvt_f_x_v_<dtype><lmul>(source, vl);
  ///   result    = __riscv_vfmul_vf_<dtype><lmul>(converted, scale, vl);
  /// `source`/`scale` are the already-mapped EmitC values; the result vector
  /// type (f32) drives the intrinsic suffix. Sets valueMap[result] so the
  /// downstream store lookup resolves. Fails closed (notifyMatchFailure) on a
  /// non-f32 result or an unconvertible vector type so a malformed body rolls
  /// back to the legacy materializer unchanged (no mislower).
  mlir::LogicalResult
  emitDequantizeChain(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, tcrvrvv::DequantizeOp dequantize,
                      mlir::Value source, mlir::Value scale,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value bodyVL) const {
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(dequantize.getResult().getType());
    if (!resultVecType || !isFloatVector(resultVecType))
      return rewriter.notifyMatchFailure(dequantize,
                                         "dequantize result not f32 vector");
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

  /// widening_convert(%source,%vl){kind} lowers to ONE call, byte-identical to
  /// the legacy signed widening-conversion oracle:
  ///   result = __riscv_vwcvt_x_x_v_<resultDtype><resultLmul>(source, vl);
  /// The widened RESULT vector type (i32/m1 for i16mf2->i32m1, i64/m2 for
  /// i32m1->i64m2) drives the intrinsic suffix; the source is the loaded narrow
  /// vector. Only the two bounded SIGNED widening kinds the op verifier accepts
  /// (`sign_extend_widen_vf2` i16mf2->i32m1 and `widen_i32_to_i64` i32m1->i64m2)
  /// are convertible; both map to the same signed `vwcvt_x_x_v` callee. Any
  /// other kind or an unexpected source/result type pairing fails the match so
  /// the body falls back to the legacy validator unchanged (no mislower -- the
  /// unsigned widening convert would need `vwcvtu`, which this does NOT emit).
  mlir::LogicalResult
  emitWideningConvert(mlir::ConversionPatternRewriter &rewriter,
                      mlir::Location loc, tcrvrvv::WideningConvertOp convert,
                      llvm::DenseMap<mlir::Value, mlir::Value> &valueMap,
                      mlir::Value bodyVL) const {
    llvm::StringRef kind = convert.getKind();
    if (kind != "sign_extend_widen_vf2" && kind != "widen_i32_to_i64")
      return rewriter.notifyMatchFailure(
          convert, "unsupported widening_convert kind");

    auto sourceVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(convert.getSource().getType());
    auto resultVecType =
        llvm::dyn_cast<tcrvrvv::VectorType>(convert.getResult().getType());
    if (!sourceVecType || !resultVecType)
      return rewriter.notifyMatchFailure(
          convert, "widening_convert source/result not typed vector");

    // Bounded (source,result) grid the signed vwcvt_x_x_v oracle covers. Refuse
    // any other pairing (including unsigned, which would need vwcvtu) so a
    // malformed/out-of-grid body falls back instead of mislowering.
    auto isSignedIntVec = [](tcrvrvv::VectorType vec, unsigned sew,
                             llvm::StringRef lmul) {
      return vec.getElementType().isSignlessInteger(sew) &&
             vec.getLmul() == lmul;
    };
    bool gridOk = false;
    if (kind == "sign_extend_widen_vf2")
      gridOk = isSignedIntVec(sourceVecType, 16, "mf2") &&
               isSignedIntVec(resultVecType, 32, "m1");
    else // widen_i32_to_i64
      gridOk = isSignedIntVec(sourceVecType, 32, "m1") &&
               isSignedIntVec(resultVecType, 64, "m2");
    if (!gridOk)
      return rewriter.notifyMatchFailure(
          convert, "widening_convert source/result outside the bounded signed "
                   "widening grid");

    mlir::Value source = valueMap.lookup(convert.getSource());
    if (!source)
      return rewriter.notifyMatchFailure(convert,
                                         "widening_convert source unmapped");
    mlir::Type vecType = convertVectorTypeToEmitC(resultVecType);
    if (!vecType)
      return rewriter.notifyMatchFailure(
          convert, "widening_convert result type not convertible");

    // The callee suffix is the WIDENED RESULT type (i32m1 / i64m2).
    std::string callee = riscvIntrinsicName(
        "vwcvt_x_x_v", vectorElementWidth(resultVecType),
        resultVecType.getLmul(), vectorDType(resultVecType));
    rewriter.create<emitc::VerbatimOp>(
        loc, stepComment(convert.getTCRVEmitCLowerableSourceOpName(),
                         convert.getTCRVEmitCLowerableSourceRole(), callee));
    mlir::Value result =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, mlir::TypeRange{vecType}, callee,
                                         mlir::ValueRange{source, bodyVL})
            .getResult(0);
    valueMap[convert.getResult()] = result;
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
    std::optional<llvm::StringRef> mnemonic =
        binaryMnemonic(masked.getKind(), isFloatVector(vectorType));
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

  /// The elementwise binary mnemonic. Integer kinds map to the v-prefixed forms
  /// (vadd/vsub/vmul); float vectors (f32/f64) map to the f-prefixed forms
  /// (vfadd/vfsub/vfmul) -- the floating-point arithmetic family the SEW=64
  /// double-precision coverage rung needs. `isFloat` keys the f-prefix the same
  /// way compareMnemonic does, so a single emitBinary path covers both.
  static std::optional<llvm::StringRef>
  binaryMnemonic(llvm::StringRef kind, bool isFloat) {
    if (isFloat) {
      if (kind == "add")
        return llvm::StringRef("vfadd");
      if (kind == "sub")
        return llvm::StringRef("vfsub");
      if (kind == "mul")
        return llvm::StringRef("vfmul");
      return std::nullopt;
    }
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
        if (type.getElementType().isF64()) {
          // f64/m1 -> vfloat64m1_t: the SEW=64 double-precision elementwise
          // rung. Gated to m1 (the bounded coverage increment); other LMUL
          // widths stay unconverted so later families extend explicitly. This
          // rung is reachable only on a capability profile whose supported_sew
          // allow-list includes 64 (full-V); a SEW=32-capped (zve32*) profile
          // gates the SEW=64 body out before it reaches here.
          if (lmul != "m1")
            return std::nullopt;
          return emitc::OpaqueType::get(type.getContext(), "vfloat64m1_t");
        }
        // Unsigned low-precision rung: ui8/mf4, ui16/mf2, ui32/m1 ->
        // vuint<sew>m<lmul>_t (the legacy unsigned widening-product/reduce oracle
        // loads u8 sources into vuint8mf4_t, widens to vuint16mf2_t, and reduces
        // into vuint32m1_t). Only the grid those families use is in scope.
        if (isUnsignedVector(type)) {
          auto intType = llvm::cast<mlir::IntegerType>(type.getElementType());
          unsigned uSew = intType.getWidth();
          bool inScope = false;
          if (uSew == 8)
            inScope = lmul == "mf4";
          else if (uSew == 16)
            inScope = lmul == "mf2" || lmul == "m1" || lmul == "m2";
          else if (uSew == 32)
            inScope = lmul == "m1" || lmul == "m2";
          else if (uSew == 64)
            inScope = lmul == "m1" || lmul == "m2";
          if (!inScope)
            return std::nullopt;
          std::string name = ("vuint" + llvm::Twine(uSew) + lmul + "_t").str();
          return emitc::OpaqueType::get(type.getContext(), name);
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
        // The deferred-wide max-legal-LMUL contraction (N3, the measured ssh-rvv
        // winner var_v_m2_a1.c) adds the wide rungs i8/m2 (strip load), i16/m4
        // (wide widening product), and i32/m8 (the loop-carried deferred vector
        // accumulator). These extend the in-scope grid for the deferred-wide
        // path; the narrow i8mf4/i16mf2/i32m1 rungs are unchanged.
        bool inScope = false;
        if (sew == 8)
          inScope = lmul == "mf4" || lmul == "m2";
        else if (sew == 16)
          inScope = lmul == "mf2" || lmul == "m1" || lmul == "m2" ||
                    lmul == "m4";
        else if (sew == 32)
          inScope = lmul == "m1" || lmul == "m2" || lmul == "m8";
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

//===----------------------------------------------------------------------===//
// RVV typed-emission backend driver. This is the RVV implementation of the
// shared `TypedBackendEmissionDriver` seam: it supplies ONLY the RVV-specific
// pieces (type conversions, target legality, patterns, the post-conversion
// kernel drain, the RVV-body pre/post-check), while the generic harness
// `convertModuleWithBackendEmitter` owns the boilerplate. A future RVM family
// adds a sibling driver and registers it — no core edit. `convertRVVModuleToEmitC`
// stays the same entry point (pass + plugin probe still call it directly), now
// implemented by delegating to the shared harness with this driver.
//===----------------------------------------------------------------------===//

namespace {

class RVVBackendEmissionDriver final
    : public ::tianchenrv::conversion::emitc::TypedBackendEmissionDriver {
public:
  llvm::StringRef getBackendName() const override { return "rvv"; }

  void
  populateTypeConversions(mlir::TypeConverter &typeConverter) const override {
    populateRVVToEmitCTypeConversions(typeConverter);
  }

  void configureConversionTarget(mlir::ConversionTarget &target) const override {
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
  }

  void
  populateLoweringPatterns(mlir::TypeConverter &typeConverter,
                           mlir::RewritePatternSet &patterns) const override {
    populateRVVElementwiseToEmitCPatterns(typeConverter, patterns);
  }

  llvm::LogicalResult postConversionCleanup(mlir::ModuleOp module) const override;

  bool moduleHasBackendBody(mlir::ModuleOp module) const override {
    // The module carries an RVV body if any op is a tcrv_rvv op OR still carries
    // a tcrv_rvv-typed operand/result (a half-converted op). This is both the
    // registry pre-check and the harness's per-backend "no RVV leftover" gate.
    auto isRVVType = [](mlir::Type type) {
      return type.getDialect().getNamespace() ==
             tcrvrvv::TCRVRVVDialect::getDialectNamespace();
    };
    bool hasRVV = false;
    module.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() ==
              tcrvrvv::TCRVRVVDialect::getDialectNamespace() ||
          llvm::any_of(op->getOperandTypes(), isRVVType) ||
          llvm::any_of(op->getResultTypes(), isRVVType)) {
        hasRVV = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return hasRVV;
  }
};

llvm::LogicalResult
RVVBackendEmissionDriver::postConversionCleanup(mlir::ModuleOp module) const {
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

  // The strangler-fig success gate (producedFunc + no RVV leftover op/type + no
  // unrealized_conversion_cast) is owned by the shared harness
  // (convertModuleWithBackendEmitter): `producedFunc` and the unrealized-cast
  // check are dialect-agnostic, and the RVV-leftover check is supplied by this
  // driver's `moduleHasBackendBody`. Cleanup itself never fails.
  return llvm::success();
}

} // namespace

void registerRVVBackendEmitter(
    ::tianchenrv::conversion::emitc::BackendEmissionRegistry &registry) {
  // Function-local static: owned by this translation unit, outlives the
  // registry, no global-init-order hazard.
  static const RVVBackendEmissionDriver driver;
  registry.registerBackend(driver);
}

bool convertRVVModuleToEmitC(mlir::ModuleOp module) {
  // The RVV->emitc conversion is now the shared `TypedBackendEmissionDriver`
  // harness parameterized by the RVV driver. The `--tcrv-rvv-lower-to-emitc`
  // pass and the plugin route probe still call this entry point directly (they
  // need the in-place gate, not the registry's clone-and-try). The behavior is
  // IDENTICAL to the pre-extraction monolithic driver.
  static const RVVBackendEmissionDriver driver;
  return ::tianchenrv::conversion::emitc::convertModuleWithBackendEmitter(module,
                                                                          driver);
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
