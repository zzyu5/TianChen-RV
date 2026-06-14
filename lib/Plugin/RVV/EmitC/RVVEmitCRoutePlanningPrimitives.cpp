//===- RVVEmitCRoutePlanningPrimitives.cpp - RVV intrinsic/type builders --===//
//
// Behavior-preserving split out of RVVEmitCRoutePlanning.cpp: the pure
// (operation / sew / lmul / count) -> StringRef RVV primitive builders -- the
// intrinsic-name / vector-type / CType / mask / segment / config-predicate
// helpers. These were file-local `static` helpers in the monolith; they are
// promoted to external linkage (named namespace) and declared in the co-located
// implementation-private RVVEmitCRoutePlanningInternal.h so the route-analysis /
// route-derivation half of the monolith can call them across the TU boundary.
//
// Pure code relocation -- the helper bodies are verbatim; the derived-text
// content they synthesize is byte-identical.
//
//===----------------------------------------------------------------------===//

#include "RVVEmitCRoutePlanningInternal.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace tianchenrv::plugin::rvv {

llvm::StringRef internRVVSelectedBodyDerivedText(std::string text) {
  static llvm::StringSet<> textPool;
  return textPool.insert(std::move(text)).first->getKey();
}

llvm::StringRef getRVVSelectedBodyIntegerElementTypeName(std::int64_t sew) {
  if (sew == tcrv::rvv::getRVVSEW16Bits())
    return "i16";
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return "i32";
  if (sew == tcrv::rvv::getRVVSEW64Bits())
    return "i64";
  return {};
}

llvm::StringRef
getRVVSelectedBodyUnsignedIntegerElementTypeName(std::int64_t sew) {
  if (sew <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("u") + llvm::Twine(sew)).str());
}

llvm::StringRef
getRVVSelectedBodyUnsignedMLIRIntegerElementTypeName(std::int64_t sew) {
  if (sew <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("ui") + llvm::Twine(sew)).str());
}

llvm::StringRef getRVVSelectedBodySignedScalarCType(std::int64_t sew) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("int") + llvm::Twine(sew) + "_t").str());
}

llvm::StringRef getRVVSelectedBodyUnsignedScalarCType(std::int64_t sew) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("uint") + llvm::Twine(sew) + "_t").str());
}

llvm::StringRef getRVVSelectedBodyConstInputPointerCType(std::int64_t sew) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("const int") + llvm::Twine(sew) + "_t *").str());
}

llvm::StringRef
getRVVSelectedBodyUnsignedConstInputPointerCType(std::int64_t sew) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("const uint") + llvm::Twine(sew) + "_t *").str());
}

llvm::StringRef getRVVSelectedBodyOutputPointerCType(std::int64_t sew) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("int") + llvm::Twine(sew) + "_t *").str());
}

llvm::StringRef
getRVVSelectedBodyUnsignedOutputPointerCType(std::int64_t sew) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("uint") + llvm::Twine(sew) + "_t *").str());
}

llvm::StringRef getRVVSelectedBodyFloatElementTypeName(std::int64_t sew) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return "f32";
  return {};
}

llvm::StringRef getRVVSelectedBodyFloatOutputPointerCType(std::int64_t sew) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty())
    return {};
  return "float *";
}

llvm::StringRef getRVVSelectedBodyFloatConstInputPointerCType(
    std::int64_t sew) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty())
    return {};
  return "const float *";
}

llvm::StringRef getRVVSelectedBodyElementByteSize(std::int64_t sew) {
  if (sew <= 0 || sew % 8 != 0)
    return {};
  return internRVVSelectedBodyDerivedText(llvm::Twine(sew / 8).str());
}

llvm::StringRef getRVVSelectedBodyVectorTypeName(std::int64_t sew,
                                                 llvm::StringRef lmul) {
  llvm::StringRef elementType = getRVVSelectedBodyIntegerElementTypeName(sew);
  if (elementType.empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("!tcrv_rvv.vector<") + elementType + ", \"" + lmul +
       "\">")
          .str());
}

llvm::StringRef getRVVSelectedBodyUnsignedVectorTypeName(std::int64_t sew,
                                                         llvm::StringRef lmul) {
  llvm::StringRef elementType =
      getRVVSelectedBodyUnsignedMLIRIntegerElementTypeName(sew);
  if (elementType.empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("!tcrv_rvv.vector<") + elementType + ", \"" + lmul +
       "\">")
          .str());
}

llvm::StringRef getRVVSelectedBodyFloatVectorTypeName(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  llvm::StringRef elementType = getRVVSelectedBodyFloatElementTypeName(sew);
  if (elementType.empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("!tcrv_rvv.vector<") + elementType + ", \"" + lmul +
       "\">")
          .str());
}

llvm::StringRef getRVVSelectedBodyMaskTypeName(std::int64_t sew,
                                               llvm::StringRef lmul) {
  llvm::StringRef elementType = getRVVSelectedBodyIntegerElementTypeName(sew);
  if (elementType.empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("!tcrv_rvv.mask<") + elementType + ", \"" + lmul + "\">")
          .str());
}

llvm::StringRef getRVVSelectedBodyFloatMaskTypeName(std::int64_t sew,
                                                    llvm::StringRef lmul) {
  llvm::StringRef elementType = getRVVSelectedBodyFloatElementTypeName(sew);
  if (elementType.empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("!tcrv_rvv.mask<") + elementType + ", \"" + lmul + "\">")
          .str());
}

llvm::StringRef getRVVSelectedBodySignedVectorCType(std::int64_t sew,
                                                    llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("vint") + llvm::Twine(sew) + lmul + "_t").str());
}

llvm::StringRef getRVVSelectedBodyUnsignedVectorCType(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty() ||
      lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("vuint") + llvm::Twine(sew) + lmul + "_t").str());
}

llvm::StringRef getRVVSelectedBodyFloatVectorCType(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("vfloat") + llvm::Twine(sew) + lmul + "_t").str());
}

std::optional<std::int64_t>
getRVVSelectedBodyMaskBitWidth(std::int64_t sew, llvm::StringRef lmul) {
  if (sew <= 0)
    return std::nullopt;
  if (lmul == tcrv::rvv::getRVVLMULM1())
    return sew;
  if (lmul == tcrv::rvv::getRVVLMULM2())
    return sew / 2;
  if (lmul == tcrv::rvv::getRVVLMULMF2())
    return sew * 2;
  return std::nullopt;
}

llvm::StringRef getRVVSelectedBodyMaskCType(std::int64_t sew,
                                            llvm::StringRef lmul) {
  std::optional<std::int64_t> maskBits =
      getRVVSelectedBodyMaskBitWidth(sew, lmul);
  if (!maskBits)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("vbool") + llvm::Twine(*maskBits) + "_t").str());
}

llvm::StringRef getRVVSelectedBodySetVLIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vsetvl_e") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getRVVSelectedBodyVectorLoadIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef
getRVVSelectedBodyUnsignedVectorLoadIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty() ||
      lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) + "_v_u" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyFloatVectorLoadIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) + "_v_f" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyStridedLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vlse") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyScalarSplatIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vmv_v_x_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef
getRVVSelectedBodyUnsignedScalarSplatIntrinsic(std::int64_t sew,
                                               llvm::StringRef lmul) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty() ||
      lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vmv_v_x_u") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getRVVSelectedBodyFloatScalarSplatIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vfmv_v_f_f") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getRVVSelectedBodyStoreIntrinsic(std::int64_t sew,
                                                 llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyUnsignedStoreIntrinsic(std::int64_t sew,
                                                         llvm::StringRef lmul) {
  if (getRVVSelectedBodyUnsignedIntegerElementTypeName(sew).empty() ||
      lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_u" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyFloatStoreIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vse") + llvm::Twine(sew) + "_v_f" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyI32ToF32DequantConvertIntrinsic() {
  return "__riscv_vfcvt_f_x_v_f32m1";
}

llvm::StringRef getRVVSelectedBodyF32ScalarScaleIntrinsic() {
  return "__riscv_vfmul_vf_f32m1";
}

llvm::StringRef getRVVSelectedBodyStridedStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vsse") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

bool hasRVVSelectedBodyI32M1IndexFacts(std::int64_t sew,
                                       llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
}

llvm::StringRef getRVVSelectedBodyIndexVectorTypeName(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("!tcrv_rvv.index_vector<i") + llvm::Twine(sew) + ", \"" +
       lmul + "\">")
          .str());
}

llvm::StringRef getRVVSelectedBodyIndexVectorCType(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("vuint") + llvm::Twine(sew) + lmul + "_t").str());
}

llvm::StringRef getRVVSelectedBodyIndexLoadIntrinsic(std::int64_t sew,
                                                     llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vle") + llvm::Twine(sew) + "_v_u" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyIndexScaleIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vmul_vx_u") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getRVVSelectedBodyIndexedLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vloxei") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyIndexedStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vsoxei") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyMaskedLoadIntrinsic(std::int64_t sew,
                                                      llvm::StringRef lmul) {
  llvm::StringRef load = getRVVSelectedBodyVectorLoadIntrinsic(sew, lmul);
  if (load.empty())
    return {};
  return internRVVSelectedBodyDerivedText((llvm::Twine(load) + "_tumu").str());
}

llvm::StringRef
getRVVSelectedBodyMaskedStridedLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul) {
  llvm::StringRef load = getRVVSelectedBodyStridedLoadIntrinsic(sew, lmul);
  if (load.empty())
    return {};
  return internRVVSelectedBodyDerivedText((llvm::Twine(load) + "_tumu").str());
}

llvm::StringRef
getRVVSelectedBodyMaskedIndexedLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vluxei") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul + "_tumu")
          .str());
}

llvm::StringRef
getRVVSelectedBodyMaskedIndexedStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul) {
  if (!hasRVVSelectedBodyI32M1IndexFacts(sew, lmul))
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vsoxei") + llvm::Twine(sew) + "_v_i" +
       llvm::Twine(sew) + lmul + "_m")
          .str());
}

llvm::StringRef getRVVSelectedBodyMaskedStoreIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  llvm::StringRef store = getRVVSelectedBodyStoreIntrinsic(sew, lmul);
  if (store.empty())
    return {};
  return internRVVSelectedBodyDerivedText((llvm::Twine(store) + "_m").str());
}

llvm::StringRef
getRVVSelectedBodyMaskedStridedStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul) {
  llvm::StringRef store = getRVVSelectedBodyStridedStoreIntrinsic(sew, lmul);
  if (store.empty())
    return {};
  return internRVVSelectedBodyDerivedText((llvm::Twine(store) + "_m").str());
}

llvm::StringRef getRVVSelectedBodySegmentTupleCType(std::int64_t sew,
                                                    llvm::StringRef lmul,
                                                    std::int64_t count) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty() ||
      count <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("vint") + llvm::Twine(sew) + lmul + "x" +
       llvm::Twine(count) + "_t")
          .str());
}

llvm::StringRef getRVVSelectedBodySegmentLoadIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul,
                                                       std::int64_t count) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty() ||
      count <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vlseg") + llvm::Twine(count) + "e" +
       llvm::Twine(sew) + "_v_i" + llvm::Twine(sew) + lmul + "x" +
       llvm::Twine(count))
          .str());
}

llvm::StringRef
getRVVSelectedBodyMaskedSegmentLoadIntrinsic(std::int64_t sew,
                                             llvm::StringRef lmul,
                                             std::int64_t count) {
  llvm::StringRef load =
      getRVVSelectedBodySegmentLoadIntrinsic(sew, lmul, count);
  if (load.empty())
    return {};
  return internRVVSelectedBodyDerivedText((llvm::Twine(load) + "_tumu").str());
}

llvm::StringRef getRVVSelectedBodySegmentStoreIntrinsic(std::int64_t sew,
                                                        llvm::StringRef lmul,
                                                        std::int64_t count) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty() ||
      count <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vsseg") + llvm::Twine(count) + "e" +
       llvm::Twine(sew) + "_v_i" + llvm::Twine(sew) + lmul + "x" +
       llvm::Twine(count))
          .str());
}

llvm::StringRef
getRVVSelectedBodyMaskedSegmentStoreIntrinsic(std::int64_t sew,
                                              llvm::StringRef lmul,
                                              std::int64_t count) {
  llvm::StringRef store =
      getRVVSelectedBodySegmentStoreIntrinsic(sew, lmul, count);
  if (store.empty())
    return {};
  return internRVVSelectedBodyDerivedText((llvm::Twine(store) + "_m").str());
}

llvm::StringRef getRVVSelectedBodySegmentFieldExtractIntrinsic(
    std::int64_t sew, llvm::StringRef lmul, std::int64_t count) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty() ||
      count <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vget_v_i") + llvm::Twine(sew) + lmul + "x" +
       llvm::Twine(count) + "_i" + llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodySegmentTupleCreateIntrinsic(
    std::int64_t sew, llvm::StringRef lmul, std::int64_t count) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty() ||
      count <= 0)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vcreate_v_i") + llvm::Twine(sew) + lmul + "x" +
       llvm::Twine(count))
          .str());
}

llvm::StringRef getRVVSelectedBodyArithmeticIntrinsic(
    RVVSelectedBodyOperationKind operation, std::int64_t sew,
    llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  llvm::StringRef mnemonic;
  switch (operation) {
  case RVVSelectedBodyOperationKind::ComputedMaskSegment2UpdateUnitLoad:
  case RVVSelectedBodyOperationKind::Add:
  case RVVSelectedBodyOperationKind::StridedAdd:
  case RVVSelectedBodyOperationKind::ScalarBroadcastAdd:
  case RVVSelectedBodyOperationKind::MaskedAdd:
    mnemonic = "vadd";
    break;
  case RVVSelectedBodyOperationKind::Sub:
  case RVVSelectedBodyOperationKind::ScalarBroadcastSub:
  case RVVSelectedBodyOperationKind::MaskedSub:
    mnemonic = "vsub";
    break;
  case RVVSelectedBodyOperationKind::Mul:
  case RVVSelectedBodyOperationKind::ScalarBroadcastMul:
  case RVVSelectedBodyOperationKind::MaskedMul:
    mnemonic = "vmul";
    break;
  default:
    return {};
  }
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_") + mnemonic + "_vv_i" + llvm::Twine(sew) + lmul)
          .str());
}

llvm::StringRef getRVVSelectedBodyCompareIntrinsicForPredicate(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul) {
  std::optional<std::int64_t> maskBits =
      getRVVSelectedBodyMaskBitWidth(sew, lmul);
  if (!maskBits)
    return {};
  llvm::StringRef mnemonic;
  if (predicateKind == "eq")
    mnemonic = "vmseq";
  else if (predicateKind == "slt")
    mnemonic = "vmslt";
  else if (predicateKind == "sle")
    mnemonic = "vmsle";
  else if (predicateKind == "ne_vx")
    mnemonic = "vmsne";
  else
    return {};
  const char *operandForm = predicateKind == "ne_vx" ? "_vx_i" : "_vv_i";
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_") + mnemonic + operandForm +
       llvm::Twine(sew) + lmul + "_b" + llvm::Twine(*maskBits))
          .str());
}

llvm::StringRef getRVVSelectedBodyFloatCompareIntrinsicForPredicate(
    llvm::StringRef predicateKind, std::int64_t sew, llvm::StringRef lmul) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  std::optional<std::int64_t> maskBits =
      getRVVSelectedBodyMaskBitWidth(sew, lmul);
  if (!maskBits)
    return {};
  llvm::StringRef mnemonic;
  if (predicateKind == "slt")
    mnemonic = "vmflt";
  else if (predicateKind == "sle")
    mnemonic = "vmfle";
  else
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_") + mnemonic + "_vv_f" + llvm::Twine(sew) +
       lmul + "_b" + llvm::Twine(*maskBits))
          .str());
}

llvm::StringRef getRVVSelectedBodySelectIntrinsic(std::int64_t sew,
                                                  llvm::StringRef lmul) {
  if (getRVVSelectedBodyIntegerElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vmerge_vvm_i") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getRVVSelectedBodyFloatSelectIntrinsic(std::int64_t sew,
                                                       llvm::StringRef lmul) {
  if (getRVVSelectedBodyFloatElementTypeName(sew).empty() || lmul.empty())
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vmerge_vvm_f") + llvm::Twine(sew) + lmul).str());
}

llvm::StringRef getRVVSelectedBodyMaskAndIntrinsic(std::int64_t sew,
                                                   llvm::StringRef lmul) {
  std::optional<std::int64_t> maskBits =
      getRVVSelectedBodyMaskBitWidth(sew, lmul);
  if (!maskBits)
    return {};
  return internRVVSelectedBodyDerivedText(
      (llvm::Twine("__riscv_vmand_mm_b") + llvm::Twine(*maskBits)).str());
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return true;
  return sew == tcrv::rvv::getRVVSEW64Bits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
}

bool isRVVSelectedBodyRuntimeScalarComputedMaskStandaloneReductionConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits())
    return lmul == tcrv::rvv::getRVVLMULM1() ||
           lmul == tcrv::rvv::getRVVLMULM2();
  return sew == tcrv::rvv::getRVVSEW64Bits() &&
         lmul == tcrv::rvv::getRVVLMULM1();
}

bool isRVVSelectedBodyStandaloneReductionScalarChannelConfig(
    std::int64_t sew, llvm::StringRef lmul) {
  return sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
         (lmul == tcrv::rvv::getRVVLMULM1() ||
          lmul == tcrv::rvv::getRVVLMULM2());
}

llvm::StringRef getRVVStandaloneReductionScalarResultVectorTypeName(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return getRVVSelectedBodyVectorTypeName(sew, tcrv::rvv::getRVVLMULM1());
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return getRVVSelectedBodyVectorTypeName(sew, tcrv::rvv::getRVVLMULM1());
  return {};
}

llvm::StringRef getRVVStandaloneReductionScalarResultVectorCType(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return getRVVSelectedBodySignedVectorCType(
        sew, tcrv::rvv::getRVVLMULM1());
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return getRVVSelectedBodySignedVectorCType(
        sew, tcrv::rvv::getRVVLMULM1());
  return {};
}

llvm::StringRef getRVVStandaloneReductionScalarSeedSplatIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return getRVVSelectedBodyScalarSplatIntrinsic(
        sew, tcrv::rvv::getRVVLMULM1());
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return getRVVSelectedBodyScalarSplatIntrinsic(
        sew, tcrv::rvv::getRVVLMULM1());
  return {};
}

llvm::StringRef getRVVStandaloneReductionScalarResultStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (sew == tcrv::rvv::getRVVFirstSliceSEWBits() &&
      (lmul == tcrv::rvv::getRVVLMULM1() ||
       lmul == tcrv::rvv::getRVVLMULM2()))
    return getRVVSelectedBodyStoreIntrinsic(sew,
                                            tcrv::rvv::getRVVLMULM1());
  if (sew == tcrv::rvv::getRVVSEW64Bits() &&
      lmul == tcrv::rvv::getRVVLMULM1())
    return getRVVSelectedBodyStoreIntrinsic(sew,
                                            tcrv::rvv::getRVVLMULM1());
  return {};
}

llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(sew, lmul))
    return {};
  return getRVVSelectedBodyMaskedLoadIntrinsic(sew, lmul);
}

llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(
    const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyRuntimeScalarMaskedLoadIntrinsic(config.sew,
                                                            config.lmul);
}

llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(sew, lmul))
    return {};
  return getRVVSelectedBodyMaskedStoreIntrinsic(sew, lmul);
}

llvm::StringRef getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(
    const RVVSelectedBodyConfigProfile &config) {
  return getRVVSelectedBodyRuntimeScalarMaskedStoreIntrinsic(config.sew,
                                                             config.lmul);
}

llvm::StringRef getRVVRuntimeScalarComputedMaskMemoryElementCType(
    std::int64_t sew, llvm::StringRef lmul) {
  if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(sew, lmul))
    return {};
  return getRVVSelectedBodySignedScalarCType(sew);
}

llvm::StringRef getRVVRuntimeScalarComputedMaskMemoryUnitStoreIntrinsic(
    std::int64_t sew, llvm::StringRef lmul) {
  if (!isRVVSelectedBodyRuntimeScalarComputedMaskMemoryConfig(sew, lmul))
    return {};
  return getRVVSelectedBodyStoreIntrinsic(sew, lmul);
}

} // namespace tianchenrv::plugin::rvv
