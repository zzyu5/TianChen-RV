#ifndef TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H
#define TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace tianchenrv::target::rvv {

struct RVVVectorShapeConfig {
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
  llvm::StringRef dtypeID;
  llvm::StringRef dtypeDiagnosticSpelling;
};

using RVVSelectedVectorShapeConfig = RVVVectorShapeConfig;
using RVVI32VectorShapeConfig = RVVSelectedVectorShapeConfig;

struct RVVVectorShapeSelectedPlanMetadataDescriptor {
  llvm::StringRef name;
  llvm::StringRef value;
  llvm::StringRef role;
  llvm::StringRef note;
  llvm::StringRef diagnosticSpelling;
};

using RVVI32VectorShapeSelectedPlanMetadataDescriptor =
    RVVVectorShapeSelectedPlanMetadataDescriptor;

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
      "i32m1",
      "i32",
      "i32"};
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
      "i32m2",
      "i32",
      "i32"};
  return config;
}

inline const RVVVectorShapeConfig &getI64M1VectorShapeConfig() {
  static const RVVVectorShapeConfig config{
      "i64m1",
      64,
      "m1",
      "agnostic",
      "agnostic",
      "rvv.i64_m1.sew64",
      "rvv.i64_m1.lmul_m1",
      "rvv.i64_m1.tail_policy.agnostic",
      "rvv.i64_m1.mask_policy.agnostic",
      "vint64m1_t",
      "i64m1",
      "e64m1",
      "i64m1",
      "i64",
      "i64"};
  return config;
}

inline llvm::ArrayRef<const RVVI32VectorShapeConfig *>
getFiniteI32VectorShapeConfigs() {
  static const RVVI32VectorShapeConfig *configs[] = {
      &getI32M1VectorShapeConfig(), &getI32M2VectorShapeConfig()};
  return llvm::ArrayRef(configs);
}

inline llvm::ArrayRef<const RVVVectorShapeConfig *>
getFiniteI64VectorShapeConfigs() {
  static const RVVVectorShapeConfig *configs[] = {
      &getI64M1VectorShapeConfig()};
  return llvm::ArrayRef(configs);
}

inline llvm::StringRef getRVVI32BinarySelectedVectorShapeCapabilityID() {
  return "rvv.i32_binary.selected_vector_shape";
}

inline llvm::StringRef getRVVI32BinarySelectedVectorShapePropertyName() {
  return "shape";
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

inline const RVVVectorShapeConfig *
lookupFiniteI64VectorShapeConfigByShapeID(llvm::StringRef shapeID) {
  shapeID = shapeID.trim();
  for (const RVVVectorShapeConfig *config : getFiniteI64VectorShapeConfigs()) {
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

inline llvm::StringRef getSelectedRVVVectorShapeMetadataRole() {
  return "selected-rvv-vector-shape-config";
}

inline llvm::StringRef getSelectedRVVI32VectorShapeMetadataRole() {
  return getSelectedRVVVectorShapeMetadataRole();
}

inline llvm::StringRef getSelectedRVVVectorShapeMetadataNote() {
  return "compile-time RVV vector-shape config selected by the RVV "
         "plugin and validated against capabilities; not base lane capacity, "
         "runtime AVL/VL, or descriptor element_count";
}

inline llvm::StringRef getSelectedRVVI32VectorShapeMetadataNote() {
  return getSelectedRVVVectorShapeMetadataNote();
}

inline llvm::StringRef
getRVVVectorShapeSEWMetadataValue(const RVVVectorShapeConfig &config) {
  if (config.sewBits == 32)
    return "32";
  if (config.sewBits == 64)
    return "64";
  return {};
}

inline llvm::StringRef
getRVVI32VectorShapeSEWMetadataValue(const RVVI32VectorShapeConfig &config) {
  return getRVVVectorShapeSEWMetadataValue(config);
}

inline void appendRVVVectorShapeSelectedPlanMetadata(
    const RVVVectorShapeConfig &config,
    llvm::SmallVectorImpl<RVVVectorShapeSelectedPlanMetadataDescriptor> &out) {
  llvm::StringRef role = getSelectedRVVVectorShapeMetadataRole();
  llvm::StringRef note = getSelectedRVVVectorShapeMetadataNote();
  out.push_back({getRVVSelectedVectorShapeAttrName(), config.shapeID, role,
                 note, "shape"});
  out.push_back({getRVVSelectedVectorSEWAttrName(),
                 getRVVVectorShapeSEWMetadataValue(config), role, note,
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

inline void appendRVVI32VectorShapeSelectedPlanMetadata(
    const RVVI32VectorShapeConfig &config,
    llvm::SmallVectorImpl<RVVI32VectorShapeSelectedPlanMetadataDescriptor>
        &out) {
  llvm::SmallVector<RVVVectorShapeSelectedPlanMetadataDescriptor, 8>
      genericOut;
  appendRVVVectorShapeSelectedPlanMetadata(config, genericOut);
  out.append(genericOut.begin(), genericOut.end());
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H
