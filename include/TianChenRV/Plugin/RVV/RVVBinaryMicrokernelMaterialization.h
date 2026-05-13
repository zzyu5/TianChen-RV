#ifndef TIANCHENRV_PLUGIN_RVV_RVVBINARYMICROKERNELMATERIALIZATION_H
#define TIANCHENRV_PLUGIN_RVV_RVVBINARYMICROKERNELMATERIALIZATION_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"
#include "TianChenRV/Target/RVV/RVVVectorShape.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace mlir {
class MLIRContext;
class OpBuilder;
} // namespace mlir

namespace tianchenrv::plugin::rvv {

struct RVVBinaryMicrokernelMaterializationPlan {
  RVVBinarySelectedPlan selectedPlan;
  std::string sourceKind;

  const target::rvv::RVVBinaryIntrinsicDescriptor &getDescriptor() const {
    return selectedPlan.descriptor;
  }

  const target::rvv::RVVBinaryFamilyDescriptor &getFamily() const {
    return *selectedPlan.family;
  }

  const target::rvv::RVVVectorShapeConfig &getShape() const {
    return selectedPlan.getShape();
  }
};

struct RVVBinaryVLDataflowMaterialization {
  const target::rvv::RVVBinarySelectedConfigContract *selectedConfig =
      nullptr;
  mlir::Type vectorType;
  llvm::StringRef microkernelOpName;
  llvm::StringRef loadOpName;
  llvm::StringRef arithmeticOpName;
  llvm::StringRef storeOpName;
  std::int64_t sewBits = 0;
  llvm::StringRef lmul;
  llvm::StringRef tailPolicy;
  llvm::StringRef maskPolicy;
  llvm::StringRef vectorSuffix;
  llvm::StringRef setvlSuffix;
  std::int64_t descriptorElementCount = 0;
};

llvm::Expected<RVVBinaryVLDataflowMaterialization>
buildRVVBinaryVLDataflowMaterialization(
    mlir::MLIRContext *context, const RVVBinarySelectedPlan &selectedPlan);

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 4>>
buildRVVBinaryCallableRuntimeABIParameters(
    tcrv::exec::KernelOp kernel,
    const target::rvv::RVVBinaryIntrinsicDescriptor &descriptor);

const target::rvv::RVVBinaryFamilyDescriptor *
getRVVBinaryMicrokernelFamilyForOp(mlir::Operation *op);

llvm::Error rejectExistingRVVBinaryMicrokernelForSelectedPath(
    tcrv::exec::KernelOp kernel, tcrv::exec::VariantOp variant,
    VariantEmissionRole role);

llvm::Expected<mlir::Operation *> materializeRVVBinaryMicrokernelOp(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::VariantOp variant, VariantEmissionRole role,
    const RVVBinaryMicrokernelMaterializationPlan &plan);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVBINARYMICROKERNELMATERIALIZATION_H
