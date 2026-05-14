#include "TianChenRV/Target/RVV/RVVMicrokernel.h"

#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "llvm/Support/Errc.h"

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kDeletedRVVMicrokernelRouteReason(
    "RVV runtime-callable direct C semantic exporter was deleted; rebuild "
    "requires a materialized MLIR EmitC module source route");

llvm::Error makeDeletedRVVMicrokernelError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV microkernel direct artifact export "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeDeletedRVVMicrokernelError() {
  return makeDeletedRVVMicrokernelError(kDeletedRVVMicrokernelRouteReason);
}

llvm::Expected<RVVBinarySelectedConfigContract>
makeDeletedSelectedConfigAuthorityError(llvm::Twine routeID) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV selected config route authority was "
                  "deleted for route '") +
          routeID +
          "': direct RVV source/header/object routes are no longer "
          "manifest-backed target authority",
      llvm::errc::invalid_argument);
}

} // namespace

llvm::Error exportRVVMicrokernelCForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::raw_ostream &os) {
  (void)module;
  (void)family;
  (void)os;
  return makeDeletedRVVMicrokernelError();
}

llvm::Error validateRVVMicrokernelSourceAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID) {
  (void)module;
  (void)family;
  (void)selectedVariant;
  (void)role;
  return makeDeletedRVVMicrokernelError(
      llvm::Twine("selected source authority validation for route '") +
      routeID + "' was deleted");
}

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID) {
  (void)module;
  (void)family;
  (void)selectedVariant;
  (void)role;
  return makeDeletedSelectedConfigAuthorityError(routeID);
}

llvm::Expected<RVVBinarySelectedConfigContract>
resolveRVVMicrokernelSelectedConfigContractAuthority(
    tcrv::exec::KernelOp kernel, const RVVBinaryFamilyRecord &family,
    llvm::StringRef selectedVariant, llvm::StringRef role,
    llvm::StringRef routeID) {
  (void)kernel;
  (void)family;
  (void)selectedVariant;
  (void)role;
  return makeDeletedSelectedConfigAuthorityError(routeID);
}

llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os) {
  (void)module;
  (void)os;
  return makeDeletedRVVMicrokernelError();
}

llvm::Error exportRVVMicrokernelHeaderForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::raw_ostream &os) {
  (void)module;
  (void)family;
  (void)os;
  return makeDeletedRVVMicrokernelError();
}

llvm::Error exportRVVMicrokernelObjectForBinaryFamily(
    mlir::ModuleOp module, const RVVBinaryFamilyRecord &family,
    llvm::raw_ostream &os) {
  (void)module;
  (void)family;
  (void)os;
  return makeDeletedRVVMicrokernelError();
}

llvm::Error registerRVVMicrokernelTargetExporters(
    TargetArtifactExporterRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

llvm::Error registerRVVMicrokernelPluginTargetExporterBundle(
    PluginTargetArtifactExporterRegistry &registry) {
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVMicrokernelTargetExporters));
}

llvm::Error registerRVVMicrokernelTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  (void)registry;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv
