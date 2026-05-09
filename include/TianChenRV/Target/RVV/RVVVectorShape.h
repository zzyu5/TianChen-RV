#ifndef TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H
#define TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H

#include "llvm/ADT/ArrayRef.h"
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
      "e32m1"};
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
      "e32m2"};
  return config;
}

inline llvm::ArrayRef<const RVVI32VectorShapeConfig *>
getFiniteI32VectorShapeConfigs() {
  static const RVVI32VectorShapeConfig *configs[] = {
      &getI32M1VectorShapeConfig(), &getI32M2VectorShapeConfig()};
  return llvm::ArrayRef(configs);
}

} // namespace tianchenrv::target::rvv

#endif // TIANCHENRV_TARGET_RVV_RVVVECTORSHAPE_H
