#ifndef TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H
#define TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace tianchenrv::target::rvv {

struct RVVI32VectorShapeConfig {
  llvm::StringRef shapeID;
  std::int64_t sewBits = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef sewCapabilityID;
  llvm::StringRef lmulCapabilityID;
  llvm::StringRef tailPolicyCapabilityID;
  llvm::StringRef maskPolicyCapabilityID;
  llvm::StringRef vectorType;
  llvm::StringRef vectorSuffix;
  llvm::StringRef setvlSuffix;
  llvm::StringRef diagnosticSpelling;
};

struct RVVI32VectorShapeSelectedPlanMetadataDescriptor {
  llvm::StringRef name;
  llvm::StringRef value;
  llvm::StringRef role;
  llvm::StringRef note;
  llvm::StringRef diagnosticSpelling;
};

inline const RVVI32VectorShapeConfig &getI32M1VectorShapeConfig() {
  static const RVVI32VectorShapeConfig config{
      "i32m1",
      32,
      "m1",
      "agnostic",
      "agnostic",
      "rvv.i32_m1.sew32",
      "rvv.i32_m1.lmul_m1",
      "rvv.i32_m1.tail_policy.agnostic",
      "rvv.i32_m1.mask_policy.agnostic",
      "vint32m1_t",
      "i32m1",
      "e32m1",
      "i32m1"};
  return config;
}

inline const RVVI32VectorShapeConfig &getI32M2VectorShapeConfig() {
  static const RVVI32VectorShapeConfig config{
      "i32m2",
      32,
      "m2",
      "agnostic",
      "agnostic",
      "rvv.i32_m2.sew32",
      "rvv.i32_m2.lmul_m2",
      "rvv.i32_m2.tail_policy.agnostic",
      "rvv.i32_m2.mask_policy.agnostic",
      "vint32m2_t",
      "i32m2",
      "e32m2",
      "i32m2"};
  return config;
}

inline llvm::ArrayRef<const RVVI32VectorShapeConfig *>
getFiniteI32VectorShapeConfigs() {
  static const RVVI32VectorShapeConfig *configs[] = {
      &getI32M1VectorShapeConfig(), &getI32M2VectorShapeConfig()};
  return llvm::ArrayRef(configs);
}

inline const RVVI32VectorShapeConfig *
lookupFiniteI32VectorShapeConfigByShapeID(llvm::StringRef shapeID) {
  shapeID = shapeID.trim();
  for (const RVVI32VectorShapeConfig *config :
       getFiniteI32VectorShapeConfigs()) {
    if (config->shapeID == shapeID)
      return config;
  }
  return nullptr;
}

inline llvm::StringRef getRVVSelectedVectorShapeAttrName() {
  return "tcrv_rvv.selected_vector_shape";
}

inline llvm::StringRef getRVVSelectedVectorSEWAttrName() {
  return "tcrv_rvv.selected_vector_sew";
}

inline llvm::StringRef getRVVSelectedVectorLMULAttrName() {
  return "tcrv_rvv.selected_vector_lmul";
}

inline llvm::StringRef getRVVSelectedTailPolicyAttrName() {
  return "tcrv_rvv.selected_tail_policy";
}

inline llvm::StringRef getRVVSelectedMaskPolicyAttrName() {
  return "tcrv_rvv.selected_mask_policy";
}

inline llvm::StringRef getRVVSelectedVectorTypeAttrName() {
  return "tcrv_rvv.selected_vector_type";
}

inline llvm::StringRef getRVVSelectedVectorSuffixAttrName() {
  return "tcrv_rvv.selected_vector_suffix";
}

inline llvm::StringRef getRVVSelectedSetVLSuffixAttrName() {
  return "tcrv_rvv.selected_setvl_suffix";
}

inline llvm::StringRef getSelectedRVVI32VectorShapeMetadataRole() {
  return "selected-rvv-vector-shape-config";
}

inline llvm::StringRef getSelectedRVVI32VectorShapeMetadataNote() {
  return "compile-time RVV i32 vector-shape config selected by the RVV "
         "plugin and validated against capabilities; not base lane capacity, "
         "runtime AVL/VL, or descriptor element_count";
}

inline llvm::StringRef
getRVVI32VectorShapeSEWMetadataValue(const RVVI32VectorShapeConfig &config) {
  if (config.sewBits == 32)
    return "32";
  return {};
}

inline void appendRVVI32VectorShapeSelectedPlanMetadata(
    const RVVI32VectorShapeConfig &config,
    llvm::SmallVectorImpl<RVVI32VectorShapeSelectedPlanMetadataDescriptor>
        &out) {
  llvm::StringRef role = getSelectedRVVI32VectorShapeMetadataRole();
  llvm::StringRef note = getSelectedRVVI32VectorShapeMetadataNote();
  out.push_back({getRVVSelectedVectorShapeAttrName(), config.shapeID, role,
                 note, "shape"});
  out.push_back({getRVVSelectedVectorSEWAttrName(),
                 getRVVI32VectorShapeSEWMetadataValue(config), role, note,
                 "sew"});
  out.push_back({getRVVSelectedVectorLMULAttrName(), config.lmul, role, note,
                 "lmul"});
  out.push_back({getRVVSelectedTailPolicyAttrName(), config.tailPolicy, role,
                 note, "tail policy"});
  out.push_back({getRVVSelectedMaskPolicyAttrName(), config.maskPolicy, role,
                 note, "mask policy"});
  out.push_back({getRVVSelectedVectorTypeAttrName(), config.vectorType, role,
                 note, "vector type"});
  out.push_back({getRVVSelectedVectorSuffixAttrName(), config.vectorSuffix,
                 role, note, "vector suffix"});
  out.push_back({getRVVSelectedSetVLSuffixAttrName(), config.setvlSuffix, role,
                 note, "setvl suffix"});
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H
