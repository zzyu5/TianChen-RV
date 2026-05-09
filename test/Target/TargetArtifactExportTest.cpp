#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"
#include "TianChenRV/Support/RuntimeABIContract.h"
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <string>

using namespace tianchenrv::target;

namespace {

using tianchenrv::support::I32VAddRuntimeABIContract;
using tianchenrv::support::I32VAddCallableRuntimeABIParameterBindings;
using tianchenrv::support::RuntimeABIParameter;
using tianchenrv::support::RuntimeABIParameterOwnership;
using tianchenrv::support::RuntimeABIParameterRole;

llvm::Error noopExporter(mlir::ModuleOp, llvm::raw_ostream &) {
  return llvm::Error::success();
}

llvm::Error sourceMarkerExporter(mlir::ModuleOp, llvm::raw_ostream &os) {
  os << "source-artifact\n";
  return llvm::Error::success();
}

llvm::Error objectMarkerExporter(mlir::ModuleOp, llvm::raw_ostream &os) {
  os << "object-artifact\n";
  return llvm::Error::success();
}

llvm::Error headerMarkerExporter(mlir::ModuleOp, llvm::raw_ostream &os) {
  os << "header-artifact\n";
  return llvm::Error::success();
}

llvm::Expected<bool>
neverMatchComposite(llvm::ArrayRef<TargetArtifactCandidate>) {
  return false;
}

llvm::Expected<bool>
alwaysMatchComposite(llvm::ArrayRef<TargetArtifactCandidate>) {
  return true;
}

bool expectSuccess(llvm::Error error, llvm::StringRef context) {
  if (!error)
    return true;
  llvm::errs() << context << ": " << llvm::toString(std::move(error)) << "\n";
  return false;
}

bool expectFailure(llvm::Error error, llvm::StringRef context) {
  if (error) {
    llvm::consumeError(std::move(error));
    return true;
  }
  llvm::errs() << context << ": expected failure\n";
  return false;
}

bool expectErrorContains(llvm::Error error, llvm::StringRef context,
                         std::initializer_list<llvm::StringRef> fragments) {
  if (!error) {
    llvm::errs() << context << ": expected failure\n";
    return false;
  }

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment)) {
      llvm::errs() << context << ": error text missing '" << fragment
                   << "': " << message << "\n";
      return false;
    }
  }
  return true;
}

bool expectRoute(const TargetArtifactExporterRegistry &registry,
                 llvm::StringRef routeID, llvm::StringRef artifactKind,
                 llvm::StringRef originPlugin,
                 llvm::StringRef emissionKind,
                 std::size_t expectedABIParameterCount = 0,
                 bool expectedDirectHelperRoute = false,
                 llvm::StringRef expectedHandoffKind = {},
                 llvm::StringRef expectedComponentGroup = {},
                 llvm::StringRef expectedExternalABIName = {}) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing built-in exporter route '" << routeID << "'\n";
    return false;
  }
  if (exporter->getArtifactKind() != artifactKind ||
      exporter->getOriginPlugin() != originPlugin ||
      exporter->getEmissionKind() != emissionKind || !exporter->getExportFn() ||
      exporter->getRequiredRuntimeABIParameters().size() !=
          expectedABIParameterCount ||
      exporter->hasDirectHelperRoute() != expectedDirectHelperRoute ||
      exporter->getHandoffKind() != expectedHandoffKind ||
      exporter->getComponentGroup() != expectedComponentGroup ||
      exporter->getExternalABIName() != expectedExternalABIName) {
    llvm::errs() << "malformed built-in exporter metadata for route '"
                 << routeID << "'\n";
    return false;
  }
  return true;
}

bool expectRuntimeABIParametersEqual(
    llvm::ArrayRef<RuntimeABIParameter> actual,
    llvm::ArrayRef<RuntimeABIParameter> expected, llvm::StringRef context) {
  if (tianchenrv::support::runtimeABIParametersEqual(actual, expected))
    return true;
  llvm::errs() << context << ": runtime ABI parameters did not match\n";
  return false;
}

bool expectRouteRuntimeABIParameters(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::ArrayRef<RuntimeABIParameter> expected) {
  const TargetArtifactExporter *exporter = registry.lookup(routeID);
  if (!exporter) {
    llvm::errs() << "missing exporter route '" << routeID
                 << "' for ABI parameter check\n";
    return false;
  }
  return expectRuntimeABIParametersEqual(
      exporter->getRequiredRuntimeABIParameters(), expected,
      "route '" + routeID.str() + "' contract parameters");
}

bool expectParameter(const RuntimeABIParameter &parameter,
                     llvm::StringRef cName, llvm::StringRef cType,
                     RuntimeABIParameterRole role,
                     RuntimeABIParameterOwnership ownership,
                     llvm::StringRef context) {
  if (parameter.cName == cName && parameter.cType == cType &&
      parameter.role == role && parameter.ownership == ownership)
    return true;
  llvm::errs() << context << ": malformed parameter\n";
  return false;
}

bool expectI32VAddRuntimeABIContractShape() {
  const I32VAddRuntimeABIContract &contract =
      tianchenrv::support::getI32VAddRuntimeABIContract();
  llvm::ArrayRef<RuntimeABIParameter> callable =
      contract.getCallableParameters();
  if (callable.size() != 4) {
    llvm::errs() << "i32-vadd ABI contract expected 4 callable parameters\n";
    return false;
  }

  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  if (!expectParameter(callable[0], "lhs", "const int32_t *",
                       RuntimeABIParameterRole::LHSInputBuffer, owned,
                       "callable parameter[0]") ||
      !expectParameter(callable[1], "rhs", "const int32_t *",
                       RuntimeABIParameterRole::RHSInputBuffer, owned,
                       "callable parameter[1]") ||
      !expectParameter(callable[2], "out", "int32_t *",
                       RuntimeABIParameterRole::OutputBuffer, owned,
                       "callable parameter[2]") ||
      !expectParameter(callable[3], "n", "size_t",
                       RuntimeABIParameterRole::RuntimeElementCount, owned,
                       "callable parameter[3]"))
    return false;

  llvm::ArrayRef<RuntimeABIParameter> requirements =
      contract.getCallableRoleRequirements();
  if (requirements.size() != callable.size())
    return false;
  for (auto [index, requirement] : llvm::enumerate(requirements)) {
    if (!requirement.cName.empty() || requirement.cType != callable[index].cType ||
        requirement.role != callable[index].role ||
        requirement.ownership != callable[index].ownership) {
      llvm::errs() << "callable role requirement[" << index
                   << "] does not mirror the contract parameter role/type\n";
      return false;
    }
  }

  llvm::ArrayRef<tianchenrv::support::RuntimeABIMemWindowSpec> windows =
      contract.getBufferMemWindowSpecs();
  if (windows.size() != 3 ||
      windows[0].role != RuntimeABIParameterRole::LHSInputBuffer ||
      windows[1].role != RuntimeABIParameterRole::RHSInputBuffer ||
      windows[2].role != RuntimeABIParameterRole::OutputBuffer) {
    llvm::errs() << "i32-vadd ABI contract buffer mem-window order changed\n";
    return false;
  }

  tianchenrv::support::RuntimeABIParamSpec count =
      contract.getRuntimeElementCountParamSpec();
  if (count.role != RuntimeABIParameterRole::RuntimeElementCount ||
      count.cName != "n" || count.cType != "size_t") {
    llvm::errs() << "i32-vadd ABI contract runtime count spec malformed\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 5> dispatch =
      contract.getDispatchRuntimeABIParameters();
  llvm::ArrayRef<RuntimeABIParameter> dispatchRef(dispatch);
  if (dispatch.size() != 5 ||
      !expectRuntimeABIParametersEqual(dispatchRef.take_front(4), callable,
                                       "dispatch callable prefix") ||
      !expectParameter(dispatch[4], "rvv_available", "int",
                       RuntimeABIParameterRole::DispatchAvailabilityGuard,
                       owned, "dispatch availability guard"))
    return false;

  return true;
}

bool expectRuntimeABIParameterRoleLookup() {
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  llvm::SmallVector<RuntimeABIParameter, 5> parameters;
  parameters.push_back(RuntimeABIParameter(
      "rvv_ready", "int", RuntimeABIParameterRole::DispatchAvailabilityGuard,
      owned));
  parameters.push_back(RuntimeABIParameter(
      "runtime_n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
      owned));
  parameters.push_back(RuntimeABIParameter(
      "lhs_ptr", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
      owned));
  parameters.push_back(RuntimeABIParameter(
      "out_ptr", "int32_t *", RuntimeABIParameterRole::OutputBuffer, owned));
  parameters.push_back(RuntimeABIParameter(
      "rhs_ptr", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer,
      owned));

  llvm::Expected<const RuntimeABIParameter *> runtimeN =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          parameters, RuntimeABIParameterRole::RuntimeElementCount,
          "out-of-order dispatch ABI parameter test");
  if (!runtimeN) {
    llvm::errs() << llvm::toString(runtimeN.takeError()) << "\n";
    return false;
  }
  if ((*runtimeN)->cName != "runtime_n") {
    llvm::errs() << "role lookup returned the wrong runtime element-count "
                    "parameter\n";
    return false;
  }

  llvm::Expected<const RuntimeABIParameter *> guard =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          parameters, RuntimeABIParameterRole::DispatchAvailabilityGuard,
          "out-of-order dispatch ABI parameter test");
  if (!guard) {
    llvm::errs() << llvm::toString(guard.takeError()) << "\n";
    return false;
  }
  if ((*guard)->cName != "rvv_ready") {
    llvm::errs() << "role lookup returned the wrong dispatch guard parameter\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 5> missingRuntimeN;
  missingRuntimeN.append(parameters.begin(), parameters.end());
  missingRuntimeN.erase(missingRuntimeN.begin() + 1);
  llvm::Expected<const RuntimeABIParameter *> missing =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          missingRuntimeN, RuntimeABIParameterRole::RuntimeElementCount,
          "missing runtime count role test");
  if (!expectErrorContains(
          missing.takeError(), "missing runtime element-count role rejected",
          {"runtime ABI parameter role lookup failed",
           "missing runtime count role test", "runtime-element-count",
           "found none"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 6> duplicateGuard;
  duplicateGuard.append(parameters.begin(), parameters.end());
  duplicateGuard.push_back(RuntimeABIParameter(
      "rvv_available", "int",
      RuntimeABIParameterRole::DispatchAvailabilityGuard, owned));
  llvm::Expected<const RuntimeABIParameter *> duplicate =
      tianchenrv::support::findUniqueRuntimeABIParameterByRole(
          duplicateGuard, RuntimeABIParameterRole::DispatchAvailabilityGuard,
          "duplicate guard role test");
  if (!expectErrorContains(
          duplicate.takeError(), "duplicate dispatch guard role rejected",
          {"runtime ABI parameter role lookup failed",
           "duplicate guard role test", "dispatch-availability-guard",
           "found duplicate parameters"}))
    return false;

  return true;
}

bool expectDirectCallableRuntimeABIBindingFailure(
    llvm::Expected<I32VAddCallableRuntimeABIParameterBindings> bindings,
    llvm::StringRef context,
    std::initializer_list<llvm::StringRef> fragments) {
  if (bindings) {
    llvm::errs() << context << ": expected failure\n";
    return false;
  }
  return expectErrorContains(bindings.takeError(), context, fragments);
}

bool expectDirectCallableRuntimeABIBinding() {
  constexpr RuntimeABIParameterOwnership owned =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
  llvm::SmallVector<RuntimeABIParameter, 4> reordered;
  reordered.push_back(RuntimeABIParameter(
      "runtime_n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
      owned));
  reordered.push_back(RuntimeABIParameter(
      "dst", "int32_t *", RuntimeABIParameterRole::OutputBuffer, owned));
  reordered.push_back(RuntimeABIParameter(
      "left", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
      owned));
  reordered.push_back(RuntimeABIParameter(
      "right", "const int32_t *", RuntimeABIParameterRole::RHSInputBuffer,
      owned));

  llvm::Expected<I32VAddCallableRuntimeABIParameterBindings> bindings =
      tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
          reordered, "reordered direct callable ABI parameter test");
  if (!bindings) {
    llvm::errs() << llvm::toString(bindings.takeError()) << "\n";
    return false;
  }
  if (bindings->lhs->cName != "left" ||
      bindings->rhs->cName != "right" ||
      bindings->out->cName != "dst" ||
      bindings->runtimeElementCount->cName != "runtime_n") {
    llvm::errs() << "direct callable role binding returned positional names\n";
    return false;
  }

  llvm::SmallVector<RuntimeABIParameter, 4> emptyName;
  emptyName.append(reordered.begin(), reordered.end());
  emptyName[2].cName.clear();
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
              emptyName, "empty direct callable C name test"),
          "empty direct callable C name rejected",
          {"runtime ABI callable parameter role binding failed",
           "empty direct callable C name test", "lhs-input-buffer",
           "requires non-empty C name"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> wrongType;
  wrongType.append(reordered.begin(), reordered.end());
  wrongType[0].cType = "uint64_t";
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
              wrongType, "wrong direct callable runtime count type test"),
          "wrong direct callable runtime count type rejected",
          {"runtime ABI callable parameter role binding failed",
           "wrong direct callable runtime count type test",
           "runtime-element-count", "must use C type 'size_t'"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 4> wrongOwnership;
  wrongOwnership.append(reordered.begin(), reordered.end());
  wrongOwnership[1].ownership = RuntimeABIParameterOwnership::IRModeled;
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
              wrongOwnership, "wrong direct callable output ownership test"),
          "wrong direct callable output ownership rejected",
          {"runtime ABI callable parameter role binding failed",
           "wrong direct callable output ownership test", "output-buffer",
           "must use ownership 'target-export-abi-owned'"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 3> missingRHS;
  missingRHS.append(reordered.begin(), reordered.end() - 1);
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
              missingRHS, "missing direct callable rhs role test"),
          "missing direct callable rhs role rejected",
          {"runtime ABI callable parameter role binding failed",
           "missing direct callable rhs role test", "rhs-input-buffer",
           "found none"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateLHS;
  duplicateLHS.append(reordered.begin(), reordered.end());
  duplicateLHS.push_back(RuntimeABIParameter(
      "also_left", "const int32_t *", RuntimeABIParameterRole::LHSInputBuffer,
      owned));
  if (!expectDirectCallableRuntimeABIBindingFailure(
          tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
              duplicateLHS, "duplicate direct callable lhs role test"),
          "duplicate direct callable lhs role rejected",
          {"runtime ABI callable parameter role binding failed",
           "duplicate direct callable lhs role test", "lhs-input-buffer",
           "found duplicate parameters"}))
    return false;

  llvm::SmallVector<RuntimeABIParameter, 5> directWithDispatchGuard;
  directWithDispatchGuard.append(reordered.begin(), reordered.end());
  directWithDispatchGuard.push_back(RuntimeABIParameter(
      "rvv_available", "int",
      RuntimeABIParameterRole::DispatchAvailabilityGuard, owned));
  return expectDirectCallableRuntimeABIBindingFailure(
      tianchenrv::support::bindI32VAddCallableRuntimeABIParametersByRole(
          directWithDispatchGuard,
          "direct callable rejects dispatch guard role test"),
      "direct callable dispatch guard role rejected",
      {"runtime ABI callable parameter role binding failed",
       "direct callable rejects dispatch guard role test",
       "unsupported direct callable runtime ABI parameter role",
       "dispatch-availability-guard"});
}

bool expectCompositeRoute(const TargetArtifactExporterRegistry &registry,
                          llvm::StringRef routeID, llvm::StringRef artifactKind,
                          llvm::StringRef expectedOwner = {},
                          llvm::StringRef expectedRuntimeABIKind = {},
                          llvm::StringRef expectedRuntimeABIName = {},
                          bool expectedDirectHelperRoute = false,
                          llvm::StringRef expectedComponentGroup = {},
                          llvm::StringRef expectedExternalABIName = {},
                          bool expectedCandidateValidation = false) {
  const TargetArtifactCompositeExporter *matched = nullptr;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    if (exporter.getRouteID() != routeID)
      continue;
    if (matched) {
      llvm::errs() << "duplicate built-in composite route '" << routeID
                   << "'\n";
      return false;
    }
    matched = &exporter;
  }
  if (!matched) {
    llvm::errs() << "missing built-in composite route '" << routeID << "'\n";
    return false;
  }
  if (matched->getArtifactKind() != artifactKind || !matched->getMatchFn() ||
      !matched->getExportFn() || matched->getOwner() != expectedOwner ||
      matched->getRuntimeABIKind() != expectedRuntimeABIKind ||
      matched->getRuntimeABIName() != expectedRuntimeABIName ||
      matched->hasDirectHelperRoute() != expectedDirectHelperRoute ||
      matched->getComponentGroup() != expectedComponentGroup ||
      matched->getExternalABIName() != expectedExternalABIName ||
      static_cast<bool>(matched->getCandidateValidationFn()) !=
          expectedCandidateValidation) {
    llvm::errs() << "malformed built-in composite route metadata for '"
                 << routeID << "'\n";
    return false;
  }
  return true;
}

bool expectSelectedCompositeRoute(
    llvm::Expected<const TargetArtifactCompositeExporter *> selected,
    llvm::StringRef routeID, llvm::StringRef context) {
  if (!selected) {
    llvm::errs() << context << ": " << llvm::toString(selected.takeError())
                 << "\n";
    return false;
  }
  if (!*selected) {
    llvm::errs() << context << ": expected selected composite route\n";
    return false;
  }
  if ((*selected)->getRouteID() != routeID) {
    llvm::errs() << context << ": expected route '" << routeID << "', got '"
                 << (*selected)->getRouteID() << "'\n";
    return false;
  }
  return true;
}

tianchenrv::tcrv::exec::KernelOp findKernel(mlir::ModuleOp module,
                                            llvm::StringRef name) {
  tianchenrv::tcrv::exec::KernelOp kernel;
  module->walk([&](tianchenrv::tcrv::exec::KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

TargetArtifactCandidate makeRVVDispatchCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  const tianchenrv::support::RuntimeABICallableIdentity &abi =
      tianchenrv::support::getI32VAddRuntimeABIContract()
          .getRVVCallableIdentity();
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "dispatch case";
  candidate.origin = "rvv-plugin";
  candidate.routeID = "tcrv-export-rvv-microkernel-c";
  candidate.emissionKind = "rvv-explicit-i32-vadd-microkernel-c-source";
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_rvv.lowering_boundary";
  candidate.runtimeABI = abi.runtimeABI.str();
  candidate.runtimeABIKind = abi.runtimeABIKind.str();
  candidate.runtimeABIName = abi.runtimeABIName.str();
  candidate.runtimeGlueRole = abi.runtimeGlueRole.str();
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32VAddRuntimeABIParameters();
  return candidate;
}

TargetArtifactCandidate makeScalarDispatchFallbackCandidate(
    tianchenrv::tcrv::exec::KernelOp kernel, llvm::StringRef selectedVariant) {
  const tianchenrv::support::RuntimeABICallableIdentity &abi =
      tianchenrv::support::getI32VAddRuntimeABIContract()
          .getScalarCallableIdentity();
  TargetArtifactCandidate candidate;
  candidate.kernel = kernel;
  candidate.selectedVariant = selectedVariant.str();
  candidate.role = "dispatch fallback";
  candidate.origin = "scalar-plugin";
  candidate.routeID = "tcrv-export-scalar-microkernel-c";
  candidate.emissionKind = "scalar-explicit-i32-vadd-microkernel-c-source";
  candidate.artifactKind = "runtime-callable-c-source";
  candidate.loweringBoundary = "tcrv_scalar.lowering_boundary";
  candidate.runtimeABI = abi.runtimeABI.str();
  candidate.runtimeABIKind = abi.runtimeABIKind.str();
  candidate.runtimeABIName = abi.runtimeABIName.str();
  candidate.runtimeGlueRole = abi.runtimeGlueRole.str();
  candidate.runtimeABIParameters =
      tianchenrv::support::getI32VAddRuntimeABIParameters();
  return candidate;
}

bool expectDispatchCompositeRejectsFallbackMismatchForRoute(
    mlir::MLIRContext &context, const TargetArtifactExporterRegistry &registry,
    llvm::StringRef routeID) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @dispatch_link_mismatch {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_rhs_input_buffer {abi_role = "rhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    tcrv.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.runtime_param @abi_dispatch_availability_guard {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    tcrv.exec.variant @rvv_first_slice attributes {origin = "rvv-plugin", requires = [@rvv]} {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.variant @scalar_ir_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {condition = "rvv_available", runtime_guard = @abi_dispatch_availability_guard, runtime_guard_required = true}
      tcrv.exec.fallback @scalar_ir_fallback
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "dispatch fallback mismatch fixture failed to parse\n";
    return false;
  }

  tianchenrv::tcrv::exec::KernelOp kernel =
      findKernel(*module, "dispatch_link_mismatch");
  if (!kernel) {
    llvm::errs() << "dispatch fallback mismatch fixture missing kernel\n";
    return false;
  }

  const TargetArtifactCompositeExporter *dispatchComposite = nullptr;
  for (const TargetArtifactCompositeExporter &exporter :
       registry.getCompositeExporters()) {
    if (exporter.getRouteID() == routeID) {
      dispatchComposite = &exporter;
      break;
    }
  }
  if (!dispatchComposite) {
    llvm::errs() << "missing RVV+scalar dispatch composite route '" << routeID
                 << "'\n";
    return false;
  }

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(makeRVVDispatchCandidate(kernel, "rvv_first_slice"));
  candidates.push_back(makeScalarDispatchFallbackCandidate(
      kernel, "scalar_fallback_first_slice"));

  llvm::Expected<bool> matched = dispatchComposite->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << "dispatch composite match unexpectedly failed before "
                    "route-local validation: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << "dispatch composite unexpectedly declined shaped RVV+scalar "
                    "candidates before route-local validation\n";
    return false;
  }
  if (!dispatchComposite->getCandidateValidationFn()) {
    llvm::errs() << "dispatch composite route '" << routeID
                 << "' lacks route-local candidate preflight validation\n";
    return false;
  }

  return expectErrorContains(
      dispatchComposite->getCandidateValidationFn()(candidates),
      "dispatch fallback IR-link mismatch rejected for " + routeID.str(),
      {"selected scalar dispatch fallback callable route",
       "@scalar_fallback_first_slice", "tcrv.exec.fallback target",
       "@scalar_ir_fallback"});
}

bool expectDispatchCompositeRejectsFallbackMismatch(
    mlir::MLIRContext &context,
    const TargetArtifactExporterRegistry &registry) {
  return expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vadd-dispatch-c") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vadd-dispatch-header") &&
         expectDispatchCompositeRejectsFallbackMismatchForRoute(
             context, registry,
             "tcrv-export-rvv-scalar-i32-vadd-dispatch-object");
}

bool expectGenericHeaderArtifactRouteSelection(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @generic_header_route {
    tcrv.exec.capability @test_cap {id = "test.capability", kind = "test", status = "available"}
    tcrv.exec.variant @selected attributes {origin = "test-plugin", requires = [@test_cap]} {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static test route",
      severity = "note",
      status = "accepted",
      target = @selected,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "supported test route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-emission",
      lowering_pipeline = "test-route",
      runtime_abi = "test-runtime-abi.v1",
      runtime_abi_kind = "test-runtime-abi-kind",
      runtime_abi_name = "test-runtime-abi-name",
      runtime_glue_role = "test-runtime-glue",
      required_capabilities = [@test_cap],
      artifact_kind = "standalone-c-source",
      lowering_boundary = "test.lowering_boundary"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "generic header artifact selection fixture failed to "
                    "parse\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "test-source-composite",
                             "runtime-callable-c-source",
                             alwaysMatchComposite, sourceMarkerExporter)),
                     "register test source composite"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "test-object-composite",
                             "riscv-elf-relocatable-object",
                             alwaysMatchComposite, objectMarkerExporter)),
                     "register test object composite"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "test-header-composite",
                             "runtime-callable-c-header",
                             alwaysMatchComposite, headerMarkerExporter)),
                     "register test header composite"))
    return false;

  std::string sourceOutput;
  llvm::raw_string_ostream sourceStream(sourceOutput);
  if (!expectSuccess(exportTargetSourceArtifact(*module, registry,
                                                sourceStream),
                     "generic source artifact route selected"))
    return false;
  sourceStream.flush();
  if (sourceOutput != "source-artifact\n") {
    llvm::errs() << "generic source artifact selected unexpected output: "
                 << sourceOutput << "\n";
    return false;
  }

  std::string objectOutput;
  llvm::raw_string_ostream objectStream(objectOutput);
  if (!expectSuccess(exportTargetArtifact(*module, registry, objectStream),
                     "generic target artifact route selected"))
    return false;
  objectStream.flush();
  if (objectOutput != "object-artifact\n") {
    llvm::errs() << "generic target artifact selected unexpected output: "
                 << objectOutput << "\n";
    return false;
  }

  std::string headerOutput;
  llvm::raw_string_ostream headerStream(headerOutput);
  if (!expectSuccess(exportTargetHeaderArtifact(*module, registry,
                                                headerStream),
                     "generic header artifact route selected"))
    return false;
  headerStream.flush();
  if (headerOutput != "header-artifact\n") {
    llvm::errs() << "generic header artifact selected unexpected output: "
                 << headerOutput << "\n";
    return false;
  }

  return true;
}

bool expectExporterRejectsRuntimeABIContractMismatch(
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter =
      registry.lookup("tcrv-export-rvv-microkernel-c");
  if (!exporter) {
    llvm::errs() << "missing RVV microkernel route for mismatch test\n";
    return false;
  }

  TargetArtifactCandidate candidate =
      makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                               "rvv_first_slice");
  if (!expectSuccess(validateTargetArtifactCandidateAgainstExporter(
                         candidate, *exporter),
                     "contract-shaped RVV runtime ABI candidate accepted"))
    return false;

  candidate.runtimeABIParameters[3].cType = "long";
  return expectErrorContains(
      validateTargetArtifactCandidateAgainstExporter(candidate, *exporter),
      "contract runtime ABI mismatch rejected by target exporter",
      {"runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'",
       "ownership 'target-export-abi-owned'"});
}

bool expectCompositeCandidateValidationRejects(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID,
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    llvm::StringRef context,
    std::initializer_list<llvm::StringRef> fragments) {
  const TargetArtifactCompositeExporter *composite =
      registry.lookupComposite(routeID);
  if (!composite) {
    llvm::errs() << "missing composite route '" << routeID
                 << "' for preflight test\n";
    return false;
  }
  if (!composite->getCandidateValidationFn()) {
    llvm::errs() << "composite route '" << routeID
                 << "' lacks candidate preflight callback\n";
    return false;
  }

  llvm::Expected<bool> matched = composite->getMatchFn()(candidates);
  if (!matched) {
    llvm::errs() << context << ": match failed before preflight: "
                 << llvm::toString(matched.takeError()) << "\n";
    return false;
  }
  if (!*matched) {
    llvm::errs() << context << ": candidate shape did not match route\n";
    return false;
  }

  return expectErrorContains(composite->getCandidateValidationFn()(candidates),
                             context, fragments);
}

bool expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate candidate =
      makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                               "rvv_first_slice");
  candidate.role = "direct variant";
  candidate.runtimeABIParameters[3].cType = "long";
  llvm::SmallVector<TargetArtifactCandidate, 1> candidates;
  candidates.push_back(candidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "RVV composite helper rejects stale callable ABI for " + routeID.str(),
      {"route id 'tcrv-export-rvv-microkernel-c'",
       "runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'"});
}

bool expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate candidate = makeScalarDispatchFallbackCandidate(
      tianchenrv::tcrv::exec::KernelOp(), "scalar_fallback_first_slice");
  candidate.role = "direct variant";
  candidate.runtimeABIParameters[3].cType = "long";
  llvm::SmallVector<TargetArtifactCandidate, 1> candidates;
  candidates.push_back(candidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "scalar composite helper rejects stale callable ABI for " +
          routeID.str(),
      {"route id 'tcrv-export-scalar-microkernel-c'",
       "runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'"});
}

bool expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
    const TargetArtifactExporterRegistry &registry, llvm::StringRef routeID) {
  TargetArtifactCandidate rvvCandidate =
      makeRVVDispatchCandidate(tianchenrv::tcrv::exec::KernelOp(),
                               "rvv_first_slice");
  TargetArtifactCandidate scalarCandidate =
      makeScalarDispatchFallbackCandidate(tianchenrv::tcrv::exec::KernelOp(),
                                          "scalar_fallback_first_slice");
  scalarCandidate.runtimeABIParameters[3].cType = "long";

  llvm::SmallVector<TargetArtifactCandidate, 2> candidates;
  candidates.push_back(rvvCandidate);
  candidates.push_back(scalarCandidate);

  return expectCompositeCandidateValidationRejects(
      registry, routeID, candidates,
      "dispatch composite rejects stale scalar callable ABI for " +
          routeID.str(),
      {"route id 'tcrv-export-scalar-microkernel-c'",
       "runtime ABI parameter role 'runtime-element-count'",
       "must use c type 'size_t'"});
}

bool expectTargetArtifactBundleDiscovery(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @bundle_manifest_route {
    tcrv.exec.capability @test_cap {id = "test.capability", kind = "test", status = "available"}
    tcrv.exec.variant @selected attributes {origin = "test-plugin", requires = [@test_cap]} {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "selected static test route",
      severity = "note",
      status = "accepted",
      target = @selected,
      selection_kind = "static-variant"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "supported test source route",
      severity = "info",
      status = "supported",
      target = @selected,
      origin = "test-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      emission_kind = "test-source-emission",
      lowering_pipeline = "bundle-source-route",
      lowering_boundary = "test.lowering_boundary",
      runtime_abi = "bundle-runtime-abi.v1",
      runtime_abi_kind = "bundle-runtime-kind",
      runtime_abi_name = "bundle-runtime-name",
      runtime_glue_role = "bundle-runtime-glue",
      required_capabilities = [@test_cap],
      artifact_kind = "runtime-callable-c-source"
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module) {
    llvm::errs() << "target artifact bundle fixture failed to parse\n";
    return false;
  }

  TargetArtifactExporterRegistry registry;
  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "bundle-source-route", "runtime-callable-c-source",
                         "test-plugin", "test-source-emission", noopExporter,
                         {}, /*directHelperRoute=*/true)),
                     "register source bundle route"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "bundle-header-route",
                             "runtime-callable-c-header",
                             alwaysMatchComposite, noopExporter,
                             "test-target-owner", "bundle-runtime-kind",
                             "bundle-runtime-name",
                             /*directHelperRoute=*/true)),
                     "register header bundle route"))
    return false;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "bundle-object-route",
                             "riscv-elf-relocatable-object",
                             alwaysMatchComposite, noopExporter,
                             "test-target-owner", "bundle-runtime-kind",
                             "bundle-runtime-name",
                             /*directHelperRoute=*/true)),
                     "register object bundle route"))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 4> records;
  if (!expectSuccess(collectTargetArtifactBundleRecords(*module, registry,
                                                        records),
                     "collect target artifact bundle records"))
    return false;
  if (records.size() != 3) {
    llvm::errs() << "expected 3 target artifact bundle records, got "
                 << records.size() << "\n";
    return false;
  }

  const TargetArtifactBundleRecord &sourceRecord = records[0];
  if (sourceRecord.artifactKind != "runtime-callable-c-source" ||
      sourceRecord.routeID != "bundle-source-route" ||
      sourceRecord.componentRole != "source" ||
      !sourceRecord.componentGroup.empty() ||
      !sourceRecord.externalABIName.empty() ||
      sourceRecord.owner != "test-plugin" ||
      sourceRecord.selectableVia != "tcrv-export-target-source-artifact" ||
      !sourceRecord.genericFrontDoorSelectable ||
      !sourceRecord.directHelperRoute ||
      sourceRecord.runtimeABIKind != "bundle-runtime-kind" ||
      sourceRecord.runtimeABIName != "bundle-runtime-name" ||
      sourceRecord.evidenceRole != "compiler-artifact") {
    llvm::errs() << "malformed source artifact bundle record\n";
    return false;
  }

  const TargetArtifactBundleRecord &headerRecord = records[1];
  if (headerRecord.artifactKind != "runtime-callable-c-header" ||
      headerRecord.routeID != "bundle-header-route" ||
      headerRecord.componentRole != "header" ||
      !headerRecord.componentGroup.empty() ||
      !headerRecord.externalABIName.empty() ||
      headerRecord.owner != "test-target-owner" ||
      headerRecord.selectableVia != "tcrv-export-target-header-artifact" ||
      headerRecord.evidenceRole != "header-declaration") {
    llvm::errs() << "malformed header artifact bundle record\n";
    return false;
  }

  const TargetArtifactBundleRecord &objectRecord = records[2];
  if (objectRecord.artifactKind != "riscv-elf-relocatable-object" ||
      objectRecord.routeID != "bundle-object-route" ||
      objectRecord.componentRole != "object" ||
      !objectRecord.componentGroup.empty() ||
      !objectRecord.externalABIName.empty() ||
      objectRecord.owner != "test-target-owner" ||
      objectRecord.selectableVia != "tcrv-export-target-artifact" ||
      objectRecord.evidenceRole != "relocatable-object") {
    llvm::errs() << "malformed object artifact bundle record\n";
    return false;
  }

  for (const TargetArtifactBundleRecord &record : records) {
    if (record.selectedVariant != "selected" ||
        record.role != "direct variant" ||
        record.componentVariants.size() != 1 ||
        record.componentVariants.front() != "selected" ||
        record.componentRoles.front() != "direct variant") {
      llvm::errs() << "artifact bundle record lost selected path metadata\n";
      return false;
    }
  }

  return true;
}

bool expectTargetArtifactBundleFileNames() {
  TargetArtifactBundleRecord sourceRecord;
  sourceRecord.artifactKind = "runtime-callable-c-source";
  sourceRecord.routeID = "tcrv-export-rvv/microkernel:c";
  std::string sourceName =
      deriveTargetArtifactBundleFileName(sourceRecord, /*index=*/7);
  if (sourceName !=
      "artifact-7-runtime-callable-c-source-tcrv-export-rvv_microkernel_c.c") {
    llvm::errs() << "unexpected sanitized source bundle file name: "
                 << sourceName << "\n";
    return false;
  }

  TargetArtifactBundleRecord headerRecord;
  headerRecord.artifactKind = "runtime-callable-c-header";
  headerRecord.routeID = "tcrv-export-rvv-microkernel-header";
  if (deriveTargetArtifactBundleFileName(headerRecord, /*index=*/1) !=
      "artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h") {
    llvm::errs() << "unexpected header bundle file name\n";
    return false;
  }

  TargetArtifactBundleRecord objectRecord;
  objectRecord.artifactKind = "riscv-elf-relocatable-object";
  objectRecord.routeID = "tcrv-export-rvv-microkernel-object";
  if (deriveTargetArtifactBundleFileName(objectRecord, /*index=*/2) !=
      "artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o") {
    llvm::errs() << "unexpected object bundle file name\n";
    return false;
  }

  TargetArtifactBundleRecord descriptorRecord;
  descriptorRecord.artifactKind = "runtime-offload-handoff-descriptor";
  descriptorRecord.routeID = "tcrv-export-offload-runtime-descriptor";
  if (deriveTargetArtifactBundleFileName(descriptorRecord, /*index=*/3) !=
      "artifact-3-runtime-offload-handoff-descriptor-tcrv-export-offload-runtime-descriptor.txt") {
    llvm::errs() << "unexpected descriptor bundle file name\n";
    return false;
  }

  return true;
}

TargetArtifactBundleRecord makeDispatchBundleComponentRecord(
    llvm::StringRef artifactKind, llvm::StringRef routeID,
    llvm::StringRef componentRole) {
  TargetArtifactBundleRecord record;
  record.componentVariants.push_back("rvv_first_slice");
  record.componentVariants.push_back("scalar_fallback_first_slice");
  record.componentRoles.push_back("dispatch case");
  record.componentRoles.push_back("dispatch fallback");
  record.componentGroup =
      "rvv-scalar-i32-vadd-dispatch-external-abi.v1";
  record.componentRole = componentRole.str();
  record.externalABIName =
      "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1";
  record.artifactKind = artifactKind.str();
  record.routeID = routeID.str();
  record.owner = "rvv-scalar-dispatch-target";
  record.runtimeABIKind = "rvv-scalar-dispatch-runtime-callable-c-abi";
  record.runtimeABIName =
      "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1";
  llvm::SmallVector<RuntimeABIParameter, 5> parameters =
      tianchenrv::support::getI32VAddRuntimeABIContract()
          .getDispatchRuntimeABIParameters();
  record.runtimeABIParameters.append(parameters.begin(), parameters.end());
  return record;
}

bool expectTargetArtifactBundleComponentContractValidation() {
  llvm::SmallVector<TargetArtifactBundleRecord, 3> records;
  records.push_back(makeDispatchBundleComponentRecord(
      "runtime-callable-c-source",
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-c", "source"));
  records.push_back(makeDispatchBundleComponentRecord(
      "runtime-callable-c-header",
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-header", "header"));
  records.push_back(makeDispatchBundleComponentRecord(
      "riscv-elf-relocatable-object",
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-object", "object"));

  if (!expectSuccess(validateTargetArtifactBundleComponentContract(records),
                     "dispatch bundle component contract accepted"))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> duplicateRole(records);
  duplicateRole[2] = records[1];
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(duplicateRole),
          "duplicate dispatch bundle component role rejected",
          {"duplicate component_role", "header"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 2> missingHeader;
  missingHeader.push_back(records[0]);
  missingHeader.push_back(records[2]);
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingHeader),
          "missing dispatch bundle header component rejected",
          {"requires exactly one source, header, and object component_role"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> missingABI(records);
  missingABI[2].runtimeABIName.clear();
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingABI),
          "missing dispatch bundle object ABI identity rejected",
          {"requires non-empty runtime_abi_name"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedABI(records);
  mismatchedABI[2].runtimeABIKind = "other-runtime-abi-kind";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedABI),
          "mismatched dispatch bundle runtime ABI kind rejected",
          {"mismatched runtime_abi_kind"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedComponents(records);
  mismatchedComponents[2].componentRoles[1] = "direct variant";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedComponents),
          "mismatched dispatch bundle selected component roles rejected",
          {"mismatched selected component roles"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> missingSignature(records);
  missingSignature[2].runtimeABIParameters.clear();
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(missingSignature),
          "missing dispatch bundle runtime ABI signature rejected",
          {"requires non-empty runtime ABI parameter signature"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> duplicateParameterRole(
      records);
  duplicateParameterRole[2].runtimeABIParameters[4] =
      duplicateParameterRole[2].runtimeABIParameters[3];
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(duplicateParameterRole),
          "duplicate dispatch bundle runtime ABI parameter role rejected",
          {"duplicate runtime ABI parameter role", "runtime-element-count"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterType(
      records);
  mismatchedParameterType[2].runtimeABIParameters[3].cType = "long";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterType),
          "mismatched dispatch bundle runtime ABI parameter type rejected",
          {"mismatched runtime ABI parameter signature",
           "runtime-element-count"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterName(
      records);
  mismatchedParameterName[2].runtimeABIParameters[4].cName = "rvv_ready";
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterName),
          "mismatched dispatch bundle runtime ABI parameter name rejected",
          {"mismatched runtime ABI parameter signature",
           "dispatch-availability-guard"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterOwnership(
      records);
  mismatchedParameterOwnership[2].runtimeABIParameters[0].ownership =
      RuntimeABIParameterOwnership::IRModeled;
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(
              mismatchedParameterOwnership),
          "mismatched dispatch bundle runtime ABI parameter ownership rejected",
          {"mismatched runtime ABI parameter signature", "lhs-input-buffer"}))
    return false;

  llvm::SmallVector<TargetArtifactBundleRecord, 3> mismatchedParameterOrder(
      records);
  std::swap(mismatchedParameterOrder[2].runtimeABIParameters[0],
            mismatchedParameterOrder[2].runtimeABIParameters[1]);
  if (!expectErrorContains(
          validateTargetArtifactBundleComponentContract(mismatchedParameterOrder),
          "mismatched dispatch bundle runtime ABI parameter order rejected",
          {"mismatched runtime ABI parameter order"}))
    return false;

  return true;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  TargetArtifactExporterRegistry registry;

  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
                     "register valid exporter"))
    return 1;
  if (!expectSuccess(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-descriptor-route",
                         "runtime-offload-handoff-descriptor",
                         "offload-plugin",
                         "runtime-offload-handoff-descriptor", noopExporter)),
                     "register descriptor exporter"))
    return 1;
  if (!expectSuccess(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "tcrv-test-composite-route",
                             "runtime-callable-c-source", neverMatchComposite,
                             noopExporter)),
                     "register valid composite exporter"))
    return 1;

  const TargetArtifactExporter *exporter = registry.lookup("tcrv-test-route");
  if (!exporter) {
    llvm::errs() << "lookup valid exporter failed\n";
    return 1;
  }
  if (exporter->getArtifactKind() != "standalone-c-source" ||
      exporter->getOriginPlugin() != "test-plugin" ||
      exporter->getEmissionKind() != "test-source" ||
      !exporter->getExportFn()) {
    llvm::errs() << "lookup returned malformed exporter metadata\n";
    return 1;
  }
  if (registry.lookup("missing-route")) {
    llvm::errs() << "lookup unexpectedly found missing route\n";
    return 1;
  }

  const TargetArtifactExporter *descriptorExporter =
      registry.lookup("tcrv-test-descriptor-route");
  if (!descriptorExporter ||
      descriptorExporter->getArtifactKind() !=
          "runtime-offload-handoff-descriptor" ||
      descriptorExporter->getOriginPlugin() != "offload-plugin" ||
      descriptorExporter->getEmissionKind() !=
          "runtime-offload-handoff-descriptor" ||
      !descriptorExporter->getExportFn()) {
    llvm::errs() << "lookup returned malformed descriptor exporter metadata\n";
    return 1;
  }

  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
                     "duplicate route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "", "standalone-c-source", "test-plugin",
                         "test-source", noopExporter)),
                     "empty route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "empty-artifact-kind", "", "test-plugin",
                         "test-source", noopExporter)),
                     "empty artifact kind rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "missing-callback", "standalone-c-source",
                         "test-plugin", "test-source", nullptr)),
                     "null callback rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "tcrv-test-composite-route",
                             "runtime-callable-c-source", neverMatchComposite,
                             noopExporter)),
                     "duplicate composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerExporter(TargetArtifactExporter(
                         "tcrv-test-composite-route", "standalone-c-source",
                         "test-plugin", "test-source", noopExporter)),
                     "single route duplicate of composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "", "runtime-callable-c-source",
                             neverMatchComposite, noopExporter)),
                     "empty composite route rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "empty-composite-artifact-kind", "",
                             neverMatchComposite, noopExporter)),
                     "empty composite artifact kind rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "missing-composite-match",
                             "runtime-callable-c-source", nullptr,
                             noopExporter)),
                     "null composite match rejected"))
    return 1;
  if (!expectFailure(registry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "missing-composite-callback",
                             "runtime-callable-c-source",
                             neverMatchComposite, nullptr)),
                     "null composite callback rejected"))
    return 1;

  TargetArtifactExporterRegistry compositeSelectionRegistry;
  if (!expectSuccess(compositeSelectionRegistry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "source-composite", "runtime-callable-c-source",
                             alwaysMatchComposite, noopExporter)),
                     "register source composite for selection"))
    return 1;
  if (!expectSuccess(compositeSelectionRegistry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "object-composite", "riscv-elf-relocatable-object",
                             alwaysMatchComposite, noopExporter)),
                     "register object composite for selection"))
    return 1;
  if (!expectSuccess(compositeSelectionRegistry.registerCompositeExporter(
                         TargetArtifactCompositeExporter(
                             "header-composite", "runtime-callable-c-header",
                             alwaysMatchComposite, noopExporter)),
                     "register header composite for selection"))
    return 1;
  if (!expectSelectedCompositeRoute(
          selectTargetArtifactCompositeExporter(
              {}, compositeSelectionRegistry, /*sourceOnly=*/true),
          "source-composite", "source-only composite selection"))
    return 1;
  if (!expectSelectedCompositeRoute(
          selectTargetArtifactCompositeExporter(
              {}, compositeSelectionRegistry, /*sourceOnly=*/false),
          "object-composite", "artifact-kind composite selection"))
    return 1;
  if (!expectGenericHeaderArtifactRouteSelection(context))
    return 1;
  if (!expectTargetArtifactBundleDiscovery(context))
    return 1;
  if (!expectTargetArtifactBundleFileNames())
    return 1;
  if (!expectTargetArtifactBundleComponentContractValidation())
    return 1;

  TargetArtifactExporterRegistry builtinRegistry;
  if (!expectSuccess(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "register built-in target artifact exporters"))
    return 1;
  if (!expectI32VAddRuntimeABIContractShape())
    return 1;
  if (!expectRuntimeABIParameterRoleLookup())
    return 1;
  if (!expectDirectCallableRuntimeABIBinding())
    return 1;
  if (builtinRegistry.size() != 5) {
    llvm::errs() << "expected exactly 5 built-in target artifact routes, got "
                 << builtinRegistry.size() << "\n";
    return 1;
  }
  if (builtinRegistry.compositeSize() != 7) {
    llvm::errs() << "expected exactly 7 built-in composite target artifact "
                    "routes, got "
                 << builtinRegistry.compositeSize() << "\n";
    return 1;
  }
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-smoke-probe-c",
                   "standalone-c-source", "rvv-plugin",
                   "rvv-smoke-probe-standalone-c-source", 0,
                   /*expectedDirectHelperRoute=*/true))
    return 1;
  const tianchenrv::support::RuntimeABICallableIdentity &rvvABI =
      tianchenrv::support::getI32VAddRuntimeABIContract()
          .getRVVCallableIdentity();
  constexpr llvm::StringLiteral rvvExternalABIComponentGroup(
      "rvv-i32-vadd-microkernel-external-abi.v1");
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vadd-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/true,
                   /*expectedHandoffKind=*/{}, rvvExternalABIComponentGroup,
                   rvvABI.runtimeABIName))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-rvv-microkernel-c",
          tianchenrv::support::getI32VAddRuntimeABIContract()
              .getCallableRoleRequirements()))
    return 1;
  if (!expectRoute(builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-c",
                   "runtime-callable-c-source", "rvv-plugin",
                   "rvv-explicit-i32-vsub-microkernel-c-source", 4,
                   /*expectedDirectHelperRoute=*/false,
                   /*expectedHandoffKind=*/{},
                   "rvv-i32-vsub-microkernel-external-abi.v1",
                   "rvv-i32-vsub-runtime-callable-c-function.v1"))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-rvv-i32-vsub-microkernel-c",
          tianchenrv::support::getI32VAddRuntimeABIContract()
              .getCallableRoleRequirements()))
    return 1;
  if (!expectRoute(builtinRegistry, "tcrv-export-scalar-microkernel-c",
                   "runtime-callable-c-source", "scalar-plugin",
                   "scalar-explicit-i32-vadd-microkernel-c-source", 4))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-scalar-microkernel-c",
          tianchenrv::support::getI32VAddRuntimeABIContract()
              .getCallableRoleRequirements()))
    return 1;
  if (!expectRoute(builtinRegistry,
                   "tcrv-export-offload-runtime-descriptor",
                   "runtime-offload-handoff-descriptor", "offload-plugin",
                   "runtime-offload-handoff-descriptor", 4,
                   /*expectedDirectHelperRoute=*/false, "runtime-offload"))
    return 1;
  if (!expectRouteRuntimeABIParameters(
          builtinRegistry, "tcrv-export-offload-runtime-descriptor",
          tianchenrv::support::getI32VAddRuntimeABIContract()
              .getCallableRoleRequirements()))
    return 1;
  const TargetArtifactExporter *offloadDescriptorExporter =
      builtinRegistry.lookup("tcrv-export-offload-runtime-descriptor");
  if (!offloadDescriptorExporter ||
      !offloadDescriptorExporter->getCandidateValidationFn()) {
    llvm::errs()
        << "offload descriptor route lacks runtime ABI preflight validator\n";
    return 1;
  }
  const tianchenrv::support::RuntimeABIDispatchIdentity &dispatchABI =
      tianchenrv::support::getI32VAddRuntimeABIContract()
          .getDispatchIdentity();
  constexpr llvm::StringLiteral dispatchExternalABIComponentGroup(
      "rvv-scalar-i32-vadd-dispatch-external-abi.v1");
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-microkernel-header",
          "runtime-callable-c-header", "rvv-plugin", rvvABI.runtimeABIKind,
          rvvABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, rvvExternalABIComponentGroup,
          rvvABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-microkernel-object",
          "riscv-elf-relocatable-object", "rvv-plugin",
          rvvABI.runtimeABIKind, rvvABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, rvvExternalABIComponentGroup,
          rvvABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  const TargetArtifactCompositeExporter *rvvHeaderComposite =
      builtinRegistry.lookupComposite("tcrv-export-rvv-microkernel-header");
  const TargetArtifactCompositeExporter *rvvObjectComposite =
      builtinRegistry.lookupComposite("tcrv-export-rvv-microkernel-object");
  if (!rvvHeaderComposite || !rvvHeaderComposite->getRuntimeABIParametersFn() ||
      !rvvObjectComposite || !rvvObjectComposite->getRuntimeABIParametersFn()) {
    llvm::errs() << "RVV microkernel header/object composites must publish "
                    "runtime ABI parameters through route-local C++ callbacks\n";
    return 1;
  }
  const tianchenrv::support::RuntimeABICallableIdentity &scalarABI =
      tianchenrv::support::getI32VAddRuntimeABIContract()
          .getScalarCallableIdentity();
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-scalar-microkernel-header",
          "runtime-callable-c-header", "scalar-plugin",
          scalarABI.runtimeABIKind, scalarABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/false, /*expectedComponentGroup=*/{},
          /*expectedExternalABIName=*/{},
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-scalar-microkernel-object",
          "riscv-elf-relocatable-object", "scalar-plugin",
          scalarABI.runtimeABIKind, scalarABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/false, /*expectedComponentGroup=*/{},
          /*expectedExternalABIName=*/{},
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
          "runtime-callable-c-source", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, dispatchExternalABIComponentGroup,
          dispatchABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  const TargetArtifactCompositeExporter *dispatchSourceComposite =
      builtinRegistry.lookupComposite(
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-c");
  if (!dispatchSourceComposite ||
      !dispatchSourceComposite->getRuntimeABIParametersFn() ||
      !dispatchSourceComposite->getCandidateValidationFn()) {
    llvm::errs() << "dispatch source composite route must publish runtime ABI "
                    "parameters and route-local candidate preflight through "
                    "C++ callbacks\n";
    return 1;
  }
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-header",
          "runtime-callable-c-header", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, dispatchExternalABIComponentGroup,
          dispatchABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectCompositeRoute(
          builtinRegistry,
          "tcrv-export-rvv-scalar-i32-vadd-dispatch-object",
          "riscv-elf-relocatable-object", "rvv-scalar-dispatch-target",
          dispatchABI.runtimeABIKind, dispatchABI.runtimeABIName,
          /*expectedDirectHelperRoute=*/true, dispatchExternalABIComponentGroup,
          dispatchABI.runtimeABIName,
          /*expectedCandidateValidation=*/true))
    return 1;
  if (!expectFailure(registerBuiltinTargetArtifactExporters(builtinRegistry),
                     "duplicate built-in exporter registration rejected"))
    return 1;
  if (!expectExporterRejectsRuntimeABIContractMismatch(builtinRegistry))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-microkernel-header"))
    return 1;
  if (!expectRVVMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-microkernel-object"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-microkernel-header"))
    return 1;
  if (!expectScalarMicrokernelCompositePreflightRejectsRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-scalar-microkernel-object"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-c"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-header"))
    return 1;
  if (!expectDispatchCompositePreflightRejectsScalarRuntimeABIMismatch(
          builtinRegistry, "tcrv-export-rvv-scalar-i32-vadd-dispatch-object"))
    return 1;
  if (!expectDispatchCompositeRejectsFallbackMismatch(context,
                                                      builtinRegistry))
    return 1;

  return 0;
}
