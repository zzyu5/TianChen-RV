#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kRVVEmitCToCppRouteID(
    "tcrv-rvv-emitc-to-cpp");
constexpr llvm::StringLiteral kRVVI32M1AddObjectArtifactRouteID(
    "tcrv-rvv-i32m1-add-riscv-elf-object");
constexpr llvm::StringLiteral kRVVI32M1AddHeaderArtifactRouteID(
    "tcrv-rvv-i32m1-add-callable-c-header");
constexpr llvm::StringLiteral kRVVI32M1AddObjectTranslateRouteID(
    "tcrv-rvv-i32m1-add-object");
constexpr llvm::StringLiteral kRVVI32M1AddHeaderTranslateRouteID(
    "tcrv-rvv-i32m1-add-header");
constexpr llvm::StringLiteral kRVVI32M1AddEmitCRouteID(
    "rvv-i32m1-add-emitc-route");
constexpr llvm::StringLiteral kRVVI32M1AddEmissionKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVI32M1AddRuntimeABIName(
    "rvv-i32m1-add-callable-c-abi.v1");
constexpr llvm::StringLiteral kRVVI32M1AddRuntimeGlueRole(
    "emitc-cpp-rvv-intrinsic-runtime-glue");
constexpr llvm::StringLiteral kRVVI32M1AddObjectArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRVVI32M1AddHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRVVI32M1AddObjectHandoffKind(
    "materialized-emitc-cpp-to-riscv-elf-object");
constexpr llvm::StringLiteral kRVVI32M1AddCallableComponentGroup(
    "rvv-i32m1-add-callable-artifact-bundle.v1");
constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

llvm::Error makeRVVEmitCToCppRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV EmitC C/C++ translate route failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error makeRVVObjectRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV i32m1 object artifact route failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool variantContainsExplicitTypedRVVBody(tcrv::exec::VariantOp variant) {
  if (!variant || variant.getBody().empty())
    return false;

  bool found = false;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (found || op == variant.getOperation())
      return;
    if (op->getName().getDialectNamespace() == "tcrv_rvv")
      found = true;
  });
  return found;
}

llvm::Error requireRVVVariantLegality(tcrv::exec::VariantOp variant) {
  if (!variant)
    return makeRVVObjectRouteError(
        "requires a materialized tcrv.exec.variant");

  auto originAttr = variant->getAttrOfType<mlir::StringAttr>("origin");
  if (!originAttr || originAttr.getValue() != kRVVPluginName)
    return makeRVVObjectRouteError(
        "materialized RVV variant must be owned by origin 'rvv-plugin'");

  if (variantContainsExplicitTypedRVVBody(variant))
    return llvm::Error::success();

  return makeRVVObjectRouteError(
      "materialized RVV variant requires explicit typed RVV "
      "extension-family body");
}

struct RVVI32M1AddSlice {
  tcrv::rvv::SetVLOp setvl;
  tcrv::rvv::WithVLOp withVL;
  tcrv::rvv::I32LoadOp lhsLoad;
  tcrv::rvv::I32LoadOp rhsLoad;
  tcrv::rvv::I32AddOp add;
  tcrv::rvv::I32StoreOp store;
};

llvm::Error requireAgnosticPolicy(tcrv::rvv::PolicyAttr policy,
                                  llvm::StringRef context) {
  if (!policy)
    return makeRVVObjectRouteError(llvm::Twine(context) +
                                   " requires finite RVV policy metadata");
  if (policy.getTail() != tcrv::rvv::TailPolicy::Agnostic ||
      policy.getMask() != tcrv::rvv::MaskPolicy::Agnostic)
    return makeRVVObjectRouteError(
        llvm::Twine(context) +
        " supports only tail agnostic, mask agnostic policy for the bounded "
        "i32m1 EmitC route");
  return llvm::Error::success();
}

llvm::Expected<RVVI32M1AddSlice>
collectRVVI32M1AddSlice(tcrv::exec::VariantOp variant) {
  llvm::SmallVector<tcrv::rvv::SetVLOp, 2> setvls;
  llvm::SmallVector<tcrv::rvv::WithVLOp, 2> withVLs;
  unsigned rvvOpCount = 0;
  variant.getBody().walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "tcrv_rvv")
      return;
    ++rvvOpCount;
    if (auto setvl = llvm::dyn_cast<tcrv::rvv::SetVLOp>(op))
      setvls.push_back(setvl);
    if (auto withVL = llvm::dyn_cast<tcrv::rvv::WithVLOp>(op))
      withVLs.push_back(withVL);
  });

  if (setvls.size() != 1)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.setvl op");
  if (withVLs.size() != 1)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.with_vl op");

  RVVI32M1AddSlice slice;
  slice.setvl = setvls.front();
  slice.withVL = withVLs.front();

  if (slice.setvl.getSew() != 32 || slice.setvl.getLmul() != "m1")
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route supports only SEW32 LMUL m1 i32 add");
  if (llvm::Error error =
          requireAgnosticPolicy(slice.setvl.getPolicy(), "tcrv_rvv.setvl"))
    return std::move(error);

  auto withVLSEW =
      slice.withVL->getAttrOfType<mlir::IntegerAttr>("sew");
  auto withVLLMUL =
      slice.withVL->getAttrOfType<mlir::StringAttr>("lmul");
  auto withVLPolicy =
      slice.withVL->getAttrOfType<tcrv::rvv::PolicyAttr>("policy");
  if (!withVLSEW || withVLSEW.getInt() != 32 || !withVLLMUL ||
      withVLLMUL.getValue() != "m1")
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires tcrv_rvv.with_vl SEW32 LMUL m1 "
        "metadata");
  if (llvm::Error error =
          requireAgnosticPolicy(withVLPolicy, "tcrv_rvv.with_vl"))
    return std::move(error);
  if (slice.withVL.getVl() != slice.setvl.getVl())
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires tcrv_rvv.with_vl to consume the "
        "visible tcrv_rvv.setvl result");

  llvm::SmallVector<tcrv::rvv::I32LoadOp, 2> loads;
  unsigned addCount = 0;
  unsigned storeCount = 0;
  for (mlir::Operation &op : slice.withVL.getBody().front()) {
    if (auto load = llvm::dyn_cast<tcrv::rvv::I32LoadOp>(op)) {
      loads.push_back(load);
      continue;
    }
    if (auto add = llvm::dyn_cast<tcrv::rvv::I32AddOp>(op)) {
      slice.add = add;
      ++addCount;
      continue;
    }
    if (auto store = llvm::dyn_cast<tcrv::rvv::I32StoreOp>(op)) {
      slice.store = store;
      ++storeCount;
      continue;
    }
    return makeRVVObjectRouteError(
        llvm::Twine("bounded RVV EmitC route does not support op '") +
        op.getName().getStringRef() +
        "' inside tcrv_rvv.with_vl; expected load-add-store only");
  }

  if (loads.size() != 2)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires exactly two tcrv_rvv.i32_load ops");
  if (addCount != 1)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_add op");
  if (storeCount != 1)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires exactly one tcrv_rvv.i32_store op");
  if (rvvOpCount != 6)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route supports only setvl/with_vl/i32_load/"
        "i32_load/i32_add/i32_store");

  for (tcrv::rvv::I32LoadOp load : loads) {
    llvm::StringRef role = load.getBufferRole();
    if (role == support::stringifyRuntimeABIParameterRole(
                    support::RuntimeABIParameterRole::LHSInputBuffer)) {
      if (slice.lhsLoad)
        return makeRVVObjectRouteError(
            "bounded RVV EmitC route requires a unique lhs-input-buffer "
            "load");
      slice.lhsLoad = load;
      continue;
    }
    if (role == support::stringifyRuntimeABIParameterRole(
                    support::RuntimeABIParameterRole::RHSInputBuffer)) {
      if (slice.rhsLoad)
        return makeRVVObjectRouteError(
            "bounded RVV EmitC route requires a unique rhs-input-buffer "
            "load");
      slice.rhsLoad = load;
      continue;
    }
    return makeRVVObjectRouteError(
        llvm::Twine("unsupported RVV i32 load role '") + role +
        "' for bounded EmitC route");
  }

  if (!slice.lhsLoad || !slice.rhsLoad)
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires lhs-input-buffer and "
        "rhs-input-buffer loads");
  if (slice.store.getBufferRole() !=
      support::stringifyRuntimeABIParameterRole(
          support::RuntimeABIParameterRole::OutputBuffer))
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires output-buffer store role");
  if (slice.add.getLhs() != slice.lhsLoad.getLoaded() ||
      slice.add.getRhs() != slice.rhsLoad.getLoaded())
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_add to consume lhs "
        "and rhs load results");
  if (slice.store.getValue() != slice.add.getSum())
    return makeRVVObjectRouteError(
        "bounded RVV EmitC route requires tcrv_rvv.i32_store to consume the "
        "add result");

  return slice;
}

llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
getEmitCSourceProvenance(mlir::Operation *op, llvm::StringRef expectedRole) {
  auto lowerable =
      llvm::dyn_cast<conversion::emitc::TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return makeRVVObjectRouteError(llvm::Twine("operation '") +
                                   op->getName().getStringRef() +
                                   "' must implement " +
                                   kEmitCLowerableOpInterfaceName +
                                   " before RVV EmitC route construction");

  llvm::StringRef sourceRole =
      lowerable.getTCRVEmitCLowerableSourceRole();
  if (sourceRole != expectedRole)
    return makeRVVObjectRouteError(llvm::Twine("operation '") +
                                   op->getName().getStringRef() +
                                   "' reports EmitC source role '" +
                                   sourceRole + "' but RVV route expected '" +
                                   expectedRole + "'");

  conversion::emitc::TCRVEmitCSourceOpProvenance source;
  source.opName = lowerable.getTCRVEmitCLowerableSourceOpName().str();
  source.role = sourceRole.str();
  source.opInterface = kEmitCLowerableOpInterfaceName.str();
  return source;
}

llvm::Error addCallStepFromSource(
    conversion::emitc::TCRVEmitCLowerableRoute &route, mlir::Operation *op,
    llvm::StringRef expectedRole, llvm::StringRef callee,
    llvm::ArrayRef<conversion::emitc::TCRVEmitCCallOpaqueOperand> operands,
    std::optional<conversion::emitc::TCRVEmitCCallOpaqueResult> result =
        std::nullopt) {
  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance> source =
      getEmitCSourceProvenance(op, expectedRole);
  if (!source)
    return source.takeError();

  conversion::emitc::TCRVEmitCCallOpaqueStep step;
  step.sourceOp = std::move(*source);
  step.callee = callee.str();
  step.operands.append(operands.begin(), operands.end());
  step.result = std::move(result);
  route.addCallOpaqueStep(std::move(step));
  return llvm::Error::success();
}

std::string sanitizeIdentifierPart(llvm::StringRef value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  if (sanitized.empty())
    return "unnamed";
  if (!std::isalpha(static_cast<unsigned char>(sanitized.front())) &&
      sanitized.front() != '_')
    sanitized.insert(sanitized.begin(), '_');
  return sanitized;
}

std::string makeEmitCFunctionName(tcrv::exec::KernelOp kernel,
                                  tcrv::exec::VariantOp variant) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "tcrv_emitc_" << sanitizeIdentifierPart(kernel.getSymName()) << "_"
     << sanitizeIdentifierPart(variant.getSymName());
  os.flush();
  return name;
}

std::string makeHeaderGuard(llvm::StringRef functionName) {
  std::string guard("TIANCHENRV_RVV_I32M1_ADD_CALLABLE_");
  for (char c : functionName) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte))
      guard.push_back(static_cast<char>(std::toupper(byte)));
    else
      guard.push_back('_');
  }
  guard.append("_H");
  return guard;
}

llvm::Error validateRVVI32M1AddObjectCandidate(
    const TargetArtifactCandidate &candidate);

struct SelectedRVVI32M1AddTarget {
  tcrv::exec::KernelOp kernel;
  tcrv::exec::VariantOp variant;
  plugin::VariantEmissionRole role = plugin::VariantEmissionRole::DirectVariant;
  TargetArtifactCandidate candidate;
};

llvm::Expected<plugin::VariantEmissionRole>
parseVariantEmissionRole(llvm::StringRef role) {
  if (role == plugin::stringifyVariantEmissionRole(
                  plugin::VariantEmissionRole::DirectVariant))
    return plugin::VariantEmissionRole::DirectVariant;
  if (role == plugin::stringifyVariantEmissionRole(
                  plugin::VariantEmissionRole::DispatchCase))
    return plugin::VariantEmissionRole::DispatchCase;
  if (role == plugin::stringifyVariantEmissionRole(
                  plugin::VariantEmissionRole::DispatchFallback))
    return plugin::VariantEmissionRole::DispatchFallback;

  return makeRVVObjectRouteError(llvm::Twine("selected emission-plan role '") +
                                 role + "' is not supported by the bounded "
                                        "RVV i32m1 add artifact route");
}

tcrv::exec::VariantOp findDirectVariantBySymbol(tcrv::exec::KernelOp kernel,
                                                llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return {};

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<tcrv::exec::VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return {};
}

llvm::Expected<SelectedRVVI32M1AddTarget>
findSelectedRVVI32M1AddTarget(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> allCandidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, allCandidates))
    return std::move(error);

  llvm::SmallVector<TargetArtifactCandidate, 2> matches;
  for (const TargetArtifactCandidate &candidate : allCandidates) {
    if (candidate.routeID != kRVVI32M1AddObjectArtifactRouteID)
      continue;
    if (candidate.artifactKind != kRVVI32M1AddObjectArtifactKind)
      continue;
    if (candidate.origin != kRVVPluginName)
      continue;
    matches.push_back(candidate);
  }

  if (matches.empty())
    return makeRVVObjectRouteError(
        "requires exactly one selected RVV i32m1 add emission-plan candidate "
        "before object/header export; found none");
  if (matches.size() != 1)
    return makeRVVObjectRouteError(
        "requires exactly one selected RVV i32m1 add emission-plan candidate "
        "before object/header export; found multiple ambiguous candidates");

  SelectedRVVI32M1AddTarget target;
  target.candidate = std::move(matches.front());
  if (llvm::Error error = validateRVVI32M1AddObjectCandidate(target.candidate))
    return std::move(error);

  llvm::Expected<plugin::VariantEmissionRole> role =
      parseVariantEmissionRole(target.candidate.role);
  if (!role)
    return role.takeError();
  target.role = *role;

  target.kernel = target.candidate.kernel;
  target.variant =
      findDirectVariantBySymbol(target.kernel, target.candidate.selectedVariant);
  if (!target.variant)
    return makeRVVObjectRouteError(
        llvm::Twine("selected emission-plan candidate target @") +
        target.candidate.selectedVariant +
        " does not resolve to a direct sibling tcrv.exec.variant before "
        "object/header export");

  return target;
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeRVVI32M1AddEmitCModule(mlir::ModuleOp module) {
  llvm::Expected<SelectedRVVI32M1AddTarget> target =
      findSelectedRVVI32M1AddTarget(module);
  if (!target)
    return target.takeError();

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(target->kernel);
  if (!capabilities)
    return capabilities.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route;
  plugin::VariantEmitCLowerableRequest request(
      target->variant, target->kernel, *capabilities,
      target->role);
  if (llvm::Error error = buildRVVI32M1AddEmitCLowerableRoute(request, route))
    return std::move(error);
  if (llvm::Error error = route.verify())
    return std::move(error);

  conversion::emitc::TCRVEmitCMaterializationOptions options;
  options.functionName = makeEmitCFunctionName(target->kernel, target->variant);
  return conversion::emitc::materializeTCRVEmitCLowerableRoute(
      *module.getContext(), route, options);
}

llvm::Expected<std::string>
getValidatedRVVI32M1AddCallableFunctionName(mlir::ModuleOp module) {
  llvm::Expected<SelectedRVVI32M1AddTarget> target =
      findSelectedRVVI32M1AddTarget(module);
  if (!target)
    return target.takeError();

  llvm::Expected<support::TargetCapabilitySet> capabilities =
      support::TargetCapabilitySet::buildFromKernelChecked(target->kernel);
  if (!capabilities)
    return capabilities.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route;
  plugin::VariantEmitCLowerableRequest request(
      target->variant, target->kernel, *capabilities,
      target->role);
  if (llvm::Error error = buildRVVI32M1AddEmitCLowerableRoute(request, route))
    return std::move(error);
  if (llvm::Error error = route.verify())
    return std::move(error);

  return makeEmitCFunctionName(target->kernel, target->variant);
}

llvm::Error requireMaterializedEmitCModule(mlir::ModuleOp module) {
  bool foundEmitCOp = false;
  mlir::Operation *unsupported = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (op == module.getOperation())
      return mlir::WalkResult::advance();

    llvm::StringRef dialectNamespace = op->getName().getDialectNamespace();
    if (dialectNamespace == "emitc") {
      foundEmitCOp = true;
      return mlir::WalkResult::advance();
    }

    unsupported = op;
    return mlir::WalkResult::interrupt();
  });

  if (unsupported)
    return makeRVVEmitCToCppRouteError(
        llvm::Twine("requires an already materialized EmitC module; found "
                    "non-EmitC op '") +
        unsupported->getName().getStringRef() + "'");
  if (!foundEmitCOp)
    return makeRVVEmitCToCppRouteError(
        "requires an already materialized EmitC module with at least one "
        "EmitC op");
  return llvm::Error::success();
}

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  if (llvm::Error error = requireMaterializedEmitCModule(module))
    return error;
  if (mlir::failed(mlir::emitc::translateToCpp(module.getOperation(), os)))
    return makeRVVEmitCToCppRouteError(
        "upstream MLIR EmitC C/C++ emitter rejected the materialized EmitC "
        "module");
  return llvm::Error::success();
}

llvm::Expected<std::string> emitRVVI32M1AddCppSource(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::ModuleOp> clonedModule(module.clone());
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      materializeRVVI32M1AddEmitCModule(*clonedModule);
  if (!emitcModule)
    return emitcModule.takeError();

  std::string generatedSource;
  llvm::raw_string_ostream sourceOS(generatedSource);
  if (llvm::Error error = exportMaterializedRVVEmitCToCpp(**emitcModule,
                                                          sourceOS))
    return std::move(error);
  sourceOS.flush();
  if (generatedSource.empty())
    return makeRVVObjectRouteError(
        "generated C/C++ source is empty before object compile preflight");
  return generatedSource;
}

llvm::Expected<std::string> findRVVClangTool() {
  if (std::optional<std::string> configured =
          llvm::sys::Process::GetEnv("TCRV_RISCV_CLANG")) {
    if (!configured->empty())
      return *configured;
  }

  for (llvm::StringRef candidate :
       {"clang-20", "clang-19", "clang-18", "clang-17", "clang-16",
        "clang-15", "clang-14", "clang"}) {
    llvm::ErrorOr<std::string> path = llvm::sys::findProgramByName(candidate);
    if (path)
      return *path;
  }

  return makeRVVObjectRouteError(
      "reached generated C/C++ source but no clang RISC-V compile tool was "
      "found; searched TCRV_RISCV_CLANG, versioned clang tools, and clang");
}

std::string summarizeToolOutput(llvm::StringRef output) {
  std::string sanitized;
  constexpr std::size_t kMaxBytes = 512;
  sanitized.reserve(std::min<std::size_t>(output.size(), kMaxBytes));
  for (char character : output.take_front(kMaxBytes)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (character == '\n' || character == '\r' || byte == 0)
      sanitized.push_back(' ');
    else if (byte < 0x20 && character != '\t')
      sanitized.push_back(' ');
    else
      sanitized.push_back(character);
  }
  if (output.size() > kMaxBytes)
    sanitized.append("...");
  return sanitized;
}

std::string readFileIfPresent(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return std::string();
  return (*buffer)->getBuffer().str();
}

llvm::Error writeTextFile(llvm::StringRef path, llvm::StringRef text) {
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return makeRVVObjectRouteError(
        llvm::Twine("failed to open generated C/C++ temporary source: ") +
        ec.message());
  file.write(text.data(), text.size());
  file.close();
  if (file.has_error())
    return makeRVVObjectRouteError(
        "failed to write generated C/C++ temporary source");
  return llvm::Error::success();
}

llvm::Error closeTemporaryFD(int fd, llvm::StringRef label) {
  llvm::raw_fd_ostream file(fd, /*shouldClose=*/true);
  file.close();
  if (file.has_error())
    return makeRVVObjectRouteError(llvm::Twine("failed to close temporary ") +
                                   label + " file");
  return llvm::Error::success();
}

llvm::Expected<std::string>
compileGeneratedSourceToRiscvObject(llvm::StringRef generatedSource) {
  llvm::Expected<std::string> clangPath = findRVVClangTool();
  if (!clangPath)
    return clangPath.takeError();

  int sourceFD = -1;
  llvm::SmallString<128> sourcePath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-i32m1-add", "c", sourceFD, sourcePath))
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create generated C/C++ temporary source: ") +
        ec.message());

  int objectFD = -1;
  llvm::SmallString<128> objectPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-i32m1-add", "o", objectFD, objectPath)) {
    llvm::sys::fs::remove(sourcePath);
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create RISC-V object temporary output: ") +
        ec.message());
  }

  int stdoutFD = -1;
  llvm::SmallString<128> stdoutPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-i32m1-add", "stdout", stdoutFD, stdoutPath)) {
    llvm::sys::fs::remove(sourcePath);
    llvm::sys::fs::remove(objectPath);
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create clang stdout capture: ") +
        ec.message());
  }

  int stderrFD = -1;
  llvm::SmallString<128> stderrPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-i32m1-add", "stderr", stderrFD, stderrPath)) {
    llvm::sys::fs::remove(sourcePath);
    llvm::sys::fs::remove(objectPath);
    llvm::sys::fs::remove(stdoutPath);
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create clang stderr capture: ") +
        ec.message());
  }

  struct TemporaryFileCleanup {
    llvm::SmallString<128> sourcePath;
    llvm::SmallString<128> objectPath;
    llvm::SmallString<128> stdoutPath;
    llvm::SmallString<128> stderrPath;
    ~TemporaryFileCleanup() {
      llvm::sys::fs::remove(sourcePath);
      llvm::sys::fs::remove(objectPath);
      llvm::sys::fs::remove(stdoutPath);
      llvm::sys::fs::remove(stderrPath);
    }
  } cleanup{sourcePath, objectPath, stdoutPath, stderrPath};

  if (llvm::Error error = closeTemporaryFD(sourceFD, "source"))
    return std::move(error);
  if (llvm::Error error = closeTemporaryFD(objectFD, "object"))
    return std::move(error);
  if (llvm::Error error = closeTemporaryFD(stdoutFD, "stdout"))
    return std::move(error);
  if (llvm::Error error = closeTemporaryFD(stderrFD, "stderr"))
    return std::move(error);

  if (llvm::Error error = writeTextFile(sourcePath, generatedSource))
    return std::move(error);

  llvm::SmallVector<llvm::StringRef, 10> args;
  args.push_back(*clangPath);
  args.push_back("--target=riscv64-unknown-elf");
  args.push_back("-march=rv64gcv");
  args.push_back("-mabi=lp64d");
  args.push_back("-c");
  args.push_back(sourcePath);
  args.push_back("-o");
  args.push_back(objectPath);

  std::string executionError;
  bool executionFailed = false;
  std::optional<llvm::StringRef> redirects[] = {
      std::nullopt, llvm::StringRef(stdoutPath), llvm::StringRef(stderrPath)};
  int exitCode = llvm::sys::ExecuteAndWait(
      *clangPath, args, std::nullopt, redirects, /*SecondsToWait=*/0,
      /*MemoryLimit=*/0, &executionError, &executionFailed);
  if (executionFailed || exitCode != 0) {
    std::string stderrText = readFileIfPresent(stderrPath);
    std::string stdoutText = readFileIfPresent(stdoutPath);
    std::string toolOutput =
        !stderrText.empty() ? stderrText : stdoutText;
    return makeRVVObjectRouteError(
        llvm::Twine("reached generated C/C++ source but RISC-V clang compile "
                    "preflight failed using '") +
        *clangPath + "' --target=riscv64-unknown-elf -march=rv64gcv "
        "-mabi=lp64d with exit code " +
        llvm::Twine(exitCode) +
        (executionError.empty() ? "" : (": " + executionError)) +
        (toolOutput.empty()
             ? ""
             : (llvm::Twine("; compiler output: ") +
                summarizeToolOutput(toolOutput))));
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath);
  if (!objectBuffer)
    return makeRVVObjectRouteError(
        llvm::Twine("RISC-V clang compile preflight completed but object "
                    "output could not be read: ") +
        objectBuffer.getError().message());

  std::string objectBytes = (*objectBuffer)->getBuffer().str();
  if (objectBytes.empty())
    return makeRVVObjectRouteError(
        "RISC-V clang compile preflight produced an empty object artifact");
  return objectBytes;
}

llvm::Error exportRVVI32M1AddObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  llvm::Expected<std::string> generatedSource =
      emitRVVI32M1AddCppSource(module);
  if (!generatedSource)
    return generatedSource.takeError();

  llvm::Expected<std::string> objectBytes =
      compileGeneratedSourceToRiscvObject(*generatedSource);
  if (!objectBytes)
    return objectBytes.takeError();

  os.write(objectBytes->data(), objectBytes->size());
  return llvm::Error::success();
}

llvm::Error exportRVVI32M1AddCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  llvm::Expected<std::string> functionName =
      getValidatedRVVI32M1AddCallableFunctionName(module);
  if (!functionName)
    return functionName.takeError();

  std::string guard = makeHeaderGuard(*functionName);
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n\n";
  os << "#ifdef __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif\n\n";
  os << "/* tianchenrv.runtime_abi_name: " << kRVVI32M1AddRuntimeABIName
     << " */\n";
  os << "/* tianchenrv.runtime_glue_role: "
     << kRVVI32M1AddRuntimeGlueRole << " */\n";
  os << "/* tianchenrv.object_route: "
     << kRVVI32M1AddObjectArtifactRouteID << " */\n";
  os << "/* tianchenrv.header_route: "
     << kRVVI32M1AddHeaderArtifactRouteID << " */\n";
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters =
      getRVVI32M1AddRuntimeABIParameters();
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    os << "/* tianchenrv.runtime_abi_parameter[" << index << "]: "
       << parameter.cName << " : " << parameter.cType << " : "
       << support::stringifyRuntimeABIParameterRole(parameter.role) << " */\n";
  }

  os << "void " << *functionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    os << parameter.cType;
    llvm::StringRef cType(parameter.cType);
    if (!cType.ends_with("*") && !cType.ends_with("&"))
      os << " ";
    os << parameter.cName;
  }
  os << ");\n\n";
  os << "#ifdef __cplusplus\n";
  os << "}\n";
  os << "#endif\n\n";
  os << "#endif /* " << guard << " */\n";
  return llvm::Error::success();
}

llvm::Error validateRVVI32M1AddObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  if (candidate.runtimeABIKind != "plugin-owned-runtime-abi")
    return makeRVVObjectRouteError(
        llvm::Twine("runtime ABI kind must be 'plugin-owned-runtime-abi' for "
                    "route '") +
        kRVVI32M1AddObjectArtifactRouteID + "'");
  if (candidate.runtimeABIName != kRVVI32M1AddRuntimeABIName)
    return makeRVVObjectRouteError(
        llvm::Twine("runtime ABI name must be '") +
        kRVVI32M1AddRuntimeABIName + "'");
  if (candidate.runtimeGlueRole != kRVVI32M1AddRuntimeGlueRole)
    return makeRVVObjectRouteError(
        llvm::Twine("runtime glue role must be '") +
        kRVVI32M1AddRuntimeGlueRole + "'");
  if (candidate.loweringBoundary != "tcrv_rvv.with_vl")
    return makeRVVObjectRouteError(
        "lowering boundary metadata must name 'tcrv_rvv.with_vl' for the "
        "bounded EmitC route");
  return llvm::Error::success();
}

llvm::Expected<bool> matchRVVI32M1AddCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return false;
  const TargetArtifactCandidate &candidate = candidates.front();
  return candidate.routeID == kRVVI32M1AddObjectArtifactRouteID &&
         candidate.artifactKind == kRVVI32M1AddObjectArtifactKind &&
         candidate.origin == kRVVPluginName;
}

llvm::Error validateRVVI32M1AddCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  if (candidates.size() != 1)
    return makeRVVObjectRouteError(
        "callable header bundle route requires exactly one RVV i32m1 add "
        "object candidate");
  const TargetArtifactCandidate &candidate = candidates.front();
  if (candidate.routeID != kRVVI32M1AddObjectArtifactRouteID)
    return makeRVVObjectRouteError(
        llvm::Twine("callable header bundle route requires object route '") +
        kRVVI32M1AddObjectArtifactRouteID + "'");
  if (candidate.artifactKind != kRVVI32M1AddObjectArtifactKind)
    return makeRVVObjectRouteError(
        llvm::Twine("callable header bundle route requires artifact_kind '") +
        kRVVI32M1AddObjectArtifactKind + "'");
  return validateRVVI32M1AddObjectCandidate(candidate);
}

llvm::Error registerRVVI32M1AddObjectExporter(
    TargetArtifactExporterRegistry &registry) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters =
      getRVVI32M1AddRuntimeABIParameters();
  if (!registry.lookup(kRVVI32M1AddObjectArtifactRouteID)) {
    if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
            kRVVI32M1AddObjectArtifactRouteID, kRVVI32M1AddObjectArtifactKind,
            kRVVPluginName, kRVVI32M1AddEmissionKind,
            exportRVVI32M1AddObjectArtifact, parameters,
            kRVVI32M1AddObjectHandoffKind, validateRVVI32M1AddObjectCandidate,
            kRVVI32M1AddCallableComponentGroup,
            kRVVI32M1AddRuntimeABIName)))
      return error;
  }

  if (registry.lookupComposite(kRVVI32M1AddHeaderArtifactRouteID))
    return llvm::Error::success();
  return registry.registerCompositeExporter(TargetArtifactCompositeExporter(
      kRVVI32M1AddHeaderArtifactRouteID, kRVVI32M1AddHeaderArtifactKind,
      matchRVVI32M1AddCallableHeaderBundle,
      exportRVVI32M1AddCallableHeaderArtifact, kRVVPluginName,
      "plugin-owned-runtime-abi", kRVVI32M1AddRuntimeABIName, parameters,
      kRVVI32M1AddCallableComponentGroup, kRVVI32M1AddRuntimeABIName,
      validateRVVI32M1AddCallableHeaderBundle));
}

} // namespace

llvm::StringRef getRVVI32M1AddObjectArtifactRouteID() {
  return kRVVI32M1AddObjectArtifactRouteID;
}

llvm::StringRef getRVVI32M1AddHeaderArtifactRouteID() {
  return kRVVI32M1AddHeaderArtifactRouteID;
}

llvm::StringRef getRVVI32M1AddEmissionKind() {
  return kRVVI32M1AddEmissionKind;
}

llvm::StringRef getRVVI32M1AddRuntimeABIName() {
  return kRVVI32M1AddRuntimeABIName;
}

llvm::StringRef getRVVI32M1AddRuntimeGlueRole() {
  return kRVVI32M1AddRuntimeGlueRole;
}

llvm::StringRef getRVVI32M1AddCallableComponentGroup() {
  return kRVVI32M1AddCallableComponentGroup;
}

llvm::SmallVector<support::RuntimeABIParameter, 4>
getRVVI32M1AddRuntimeABIParameters() {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters;
  parameters.push_back(support::makeTargetExportABIParameter(
      "lhs", "const int32_t *",
      support::RuntimeABIParameterRole::LHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "rhs", "const int32_t *",
      support::RuntimeABIParameterRole::RHSInputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "out", "int32_t *", support::RuntimeABIParameterRole::OutputBuffer));
  parameters.push_back(support::makeTargetExportABIParameter(
      "n", "size_t", support::RuntimeABIParameterRole::RuntimeElementCount));
  return parameters;
}

llvm::Error buildRVVI32M1AddEmitCLowerableRoute(
    const plugin::VariantEmitCLowerableRequest &request,
    conversion::emitc::TCRVEmitCLowerableRoute &out) {
  if (!request.getVariant())
    return makeRVVObjectRouteError(
        "EmitC route construction requires a materialized tcrv.exec.variant");
  if (!request.getKernel())
    return makeRVVObjectRouteError(
        "EmitC route construction requires an enclosing tcrv.exec.kernel");

  if (llvm::Error error = requireRVVVariantLegality(request.getVariant()))
    return error;

  llvm::Expected<RVVI32M1AddSlice> slice =
      collectRVVI32M1AddSlice(request.getVariant());
  if (!slice)
    return slice.takeError();

  conversion::emitc::TCRVEmitCLowerableRoute route(
      kRVVI32M1AddEmitCRouteID, "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", "size_t");
  route.addTypeMapping("!tcrv_rvv.i32m1", "vint32m1_t");
  for (const support::RuntimeABIParameter &parameter :
       getRVVI32M1AddRuntimeABIParameters())
    route.addABIValueMapping(parameter, parameter.cName);

  llvm::Expected<conversion::emitc::TCRVEmitCSourceOpProvenance>
      withVLSource =
          getEmitCSourceProvenance(slice->withVL.getOperation(), "scope");
  if (!withVLSource)
    return withVLSource.takeError();
  route.addSourceOpProvenance(std::move(*withVLSource));

  using conversion::emitc::TCRVEmitCCallOpaqueOperand;
  using conversion::emitc::TCRVEmitCCallOpaqueResult;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->setvl.getOperation(), "configure",
          "__riscv_vsetvl_e32m1",
          {TCRVEmitCCallOpaqueOperand{"n", "size_t"}},
          TCRVEmitCCallOpaqueResult{"vl", "size_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->lhsLoad.getOperation(), "load",
          "__riscv_vle32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{"lhs", "const int32_t *"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}},
          TCRVEmitCCallOpaqueResult{"lhs_vec", "vint32m1_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->rhsLoad.getOperation(), "load",
          "__riscv_vle32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{"rhs", "const int32_t *"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}},
          TCRVEmitCCallOpaqueResult{"rhs_vec", "vint32m1_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->add.getOperation(), "compute",
          "__riscv_vadd_vv_i32m1",
          {TCRVEmitCCallOpaqueOperand{"lhs_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"rhs_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}},
          TCRVEmitCCallOpaqueResult{"sum_vec", "vint32m1_t"}))
    return error;
  if (llvm::Error error = addCallStepFromSource(
          route, slice->store.getOperation(), "store",
          "__riscv_vse32_v_i32m1",
          {TCRVEmitCCallOpaqueOperand{"out", "int32_t *"},
           TCRVEmitCCallOpaqueOperand{"sum_vec", "vint32m1_t"},
           TCRVEmitCCallOpaqueOperand{"vl", "size_t"}}))
    return error;

  out = std::move(route);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(kRVVPluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(kRVVPluginName))
      if (bundle.getRegistrationFn() == registerRVVI32M1AddObjectExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVI32M1AddObjectExporter));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  auto registerHeaderRoute = [&registry]() -> llvm::Error {
    if (registry.lookup(kRVVI32M1AddHeaderTranslateRouteID))
      return llvm::Error::success();
    return registry.registerRoute(TargetTranslateRoute(
        kRVVI32M1AddHeaderTranslateRouteID,
        "export a selected RVV i32m1 add callable C header for the generated "
        "RISC-V relocatable object ABI",
        exportRVVI32M1AddCallableHeaderArtifact,
        /*requiresBinaryStdout=*/false, kRVVI32M1AddHeaderArtifactRouteID));
  };

  if (!registry.lookup(kRVVEmitCToCppRouteID)) {
    if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
            kRVVEmitCToCppRouteID,
            "export a materialized RVV EmitC module through the MLIR EmitC "
            "C/C++ emitter",
            exportMaterializedRVVEmitCToCpp)))
      return error;
  }
  if (registry.lookup(kRVVI32M1AddObjectTranslateRouteID))
    return registerHeaderRoute();
  if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
      kRVVI32M1AddObjectTranslateRouteID,
      "export a selected RVV i32m1 add path through EmitC, the MLIR EmitC "
      "C/C++ emitter, and clang RISC-V relocatable object compilation",
      exportRVVI32M1AddObjectArtifact,
      /*requiresBinaryStdout=*/true, kRVVI32M1AddObjectArtifactRouteID)))
    return error;
  return registerHeaderRoute();
}

} // namespace tianchenrv::target::rvv
