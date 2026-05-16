#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Dialect/RVV/IR/RVVConfigContract.h"
#include "TianChenRV/Plugin/RVV/RVVConstructionProtocol.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

namespace tianchenrv::target::rvv {
namespace {

constexpr llvm::StringLiteral kRVVEmitCToCppRouteID(
    "tcrv-rvv-emitc-to-cpp");
constexpr llvm::StringLiteral kRVVMaterializedEmitCHeaderRouteID(
    "rvv-i32m1-arithmetic-emitc-route-family.header");
constexpr llvm::StringLiteral kRuntimeCallableCHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticObjectHandoffKind(
    "materialized-emitc-cpp-rvv-intrinsic-object");
constexpr llvm::StringLiteral kRVVMaterializedEmitCBundleComponentGroup(
    "rvv-i32m1-arithmetic-materialized-emitc-bundle.v1");

struct ScopedTempPath {
  llvm::SmallString<128> path;

  ~ScopedTempPath() {
    if (!path.empty())
      (void)llvm::sys::fs::remove(path);
  }
};

llvm::Error makeRVVTargetRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV materialized EmitC target artifact bridge "
                  "failed: ") +
          message,
      llvm::errc::invalid_argument);
}

const plugin::rvv::RVVConstructionManifest &getRVVManifest() {
  return plugin::rvv::getRVVConstructionManifest();
}

llvm::Error requireCandidateField(llvm::StringRef fieldName,
                                  llvm::StringRef actual,
                                  llvm::StringRef expected) {
  if (actual == expected)
    return llvm::Error::success();
  return makeRVVTargetRouteError(llvm::Twine("candidate ") + fieldName +
                                 " must be '" + expected + "' but was '" +
                                 actual + "'");
}

llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp>
getCandidateArithmeticOp(const TargetArtifactCandidate &candidate) {
  llvm::StringRef routeID;
  llvm::StringRef arithmeticOp;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (entry.key == "rvv_emitc_lowerable_route")
      routeID = entry.value;
    if (entry.key == "rvv_arithmetic_op")
      arithmeticOp = entry.value;
  }
  if (routeID.empty())
    return makeRVVTargetRouteError(
        "candidate metadata must carry rvv_emitc_lowerable_route provenance");
  llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp> op =
      plugin::rvv::symbolizeRVVI32M1ArithmeticOpFromEmitCRouteID(routeID);
  if (!op)
    return op.takeError();
  if (arithmeticOp.empty())
    return makeRVVTargetRouteError(
        "candidate metadata must carry rvv_arithmetic_op provenance");
  if (arithmeticOp != plugin::rvv::stringifyRVVI32M1ArithmeticOp(*op))
    return makeRVVTargetRouteError(
        llvm::Twine("candidate rvv_arithmetic_op metadata must match route '") +
        routeID + "'");
  return *op;
}

llvm::Error rejectForbiddenRVVArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    llvm::StringRef key(entry.key);
    std::string lowerKeyStorage = key.lower();
    llvm::StringRef lowerKey(lowerKeyStorage);
    if (lowerKey.contains("element_count") ||
        lowerKey.contains("element-count") || lowerKey.contains("descriptor"))
      return makeRVVTargetRouteError(
          llvm::Twine("candidate artifact metadata key '") + key +
          "' is descriptor-local or hardcoded element-count residue");
  }
  return llvm::Error::success();
}

llvm::Error validateRVVRuntimeAVLVLArtifactMetadata(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = rejectForbiddenRVVArtifactMetadata(candidate))
    return error;

  llvm::SmallVector<support::ArtifactMetadataEntry, 16> rvvMetadata;
  for (const support::ArtifactMetadataEntry &entry :
       candidate.artifactMetadata) {
    if (llvm::StringRef(entry.key).starts_with("tcrv_rvv."))
      rvvMetadata.push_back(entry);
  }
  return tcrv::rvv::verifyRVVI32M1ArithmeticArtifactMetadata(
      rvvMetadata, "selected RVV materialized EmitC candidate");
}

llvm::StringRef lookupArtifactMetadataValue(
    llvm::ArrayRef<support::ArtifactMetadataEntry> metadata,
    llvm::StringRef key) {
  for (const support::ArtifactMetadataEntry &entry : metadata)
    if (entry.key == key)
      return entry.value;
  return {};
}

llvm::Error validateRVVI32M1ArithmeticTargetArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  if (llvm::Error error =
          requireCandidateField("route id", candidate.routeID,
                                manifest.emitcRoute.routeID))
    return error;
  if (llvm::Error error = requireCandidateField(
          "origin", candidate.origin, manifest.family.pluginName))
    return error;
  if (llvm::Error error =
          requireCandidateField("emission kind", candidate.emissionKind,
                                plugin::rvv::getRVVI32M1ArithmeticEmissionKind()))
    return error;
  if (llvm::Error error =
          requireCandidateField("artifact kind", candidate.artifactKind,
                                manifest.emitcRoute.artifactKind))
    return error;
  if (llvm::Error error = requireCandidateField(
          "lowering boundary", candidate.loweringBoundary,
          plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI kind", candidate.runtimeABIKind,
          plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind()))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime glue role", candidate.runtimeGlueRole,
          plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole()))
    return error;

  llvm::Expected<plugin::rvv::RVVI32M1ArithmeticOp> arithmetic =
      getCandidateArithmeticOp(candidate);
  if (!arithmetic)
    return arithmetic.takeError();
  llvm::StringRef runtimeABIName =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(*arithmetic);
  if (llvm::Error error = requireCandidateField(
          "runtime ABI", candidate.runtimeABI, runtimeABIName))
    return error;
  if (llvm::Error error = requireCandidateField(
          "runtime ABI name", candidate.runtimeABIName, runtimeABIName))
    return error;
  if (!support::runtimeABIParametersEqual(
          candidate.runtimeABIParameters,
          plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters()))
    return makeRVVTargetRouteError(
        "candidate runtime ABI parameters must be lhs, rhs, out, n with "
        "stable C types, roles, and target-export ownership");
  if (llvm::Error error = validateRVVRuntimeAVLVLArtifactMetadata(candidate))
    return error;
  return llvm::Error::success();
}

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportMaterializedEmitCModuleToCpp(
      module, os, "RVV EmitC C/C++ translate route");
}

llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os) {
  llvm::ErrorOr<std::string> clang = llvm::sys::findProgramByName("clang");
  if (!clang)
    clang = llvm::sys::findProgramByName(
        "clang", {"/usr/lib/llvm-20/bin", "/usr/local/bin", "/usr/bin"});
  if (!clang)
    return makeRVVTargetRouteError(
        llvm::Twine("requires clang on PATH or a standard LLVM tools path for "
                    "RISC-V object packaging: ") +
        clang.getError().message());

  int sourceFD = -1;
  ScopedTempPath sourcePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-materialized-emitc", "cpp", sourceFD, sourcePath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary C++ source: ") +
        error.message());
  {
    llvm::raw_fd_ostream sourceOS(sourceFD, /*shouldClose=*/true);
    sourceOS << source;
    sourceOS.close();
    if (sourceOS.has_error())
      return makeRVVTargetRouteError(
          "failed to write generated MLIR EmitC C/C++ source before object "
          "packaging");
  }

  ScopedTempPath objectPath;
  objectPath.path = sourcePath.path;
  llvm::sys::path::replace_extension(objectPath.path, "o");

  int stderrFD = -1;
  ScopedTempPath stderrPath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "tcrv-rvv-materialized-emitc-clang", "stderr", stderrFD,
          stderrPath.path))
    return makeRVVTargetRouteError(
        llvm::Twine("failed to create temporary clang stderr file: ") +
        error.message());
  {
    llvm::raw_fd_ostream stderrOS(stderrFD, /*shouldClose=*/true);
    stderrOS.close();
  }

  llvm::SmallVector<llvm::StringRef, 12> args = {
      *clang,
      "-target",
      "riscv64",
      "-O2",
      "-march=rv64gcv",
      "-mabi=lp64d",
      "-c",
      sourcePath.path,
      "-o",
      objectPath.path};
  llvm::SmallVector<std::optional<llvm::StringRef>, 3> redirects = {
      llvm::StringRef(), llvm::StringRef(), llvm::StringRef(stderrPath.path)};
  std::string executeError;
  bool executionFailed = false;
  int result = llvm::sys::ExecuteAndWait(
      *clang, args, std::nullopt, redirects, /*SecondsToWait=*/30,
      /*MemoryLimit=*/0, &executeError, &executionFailed);
  if (executionFailed || result != 0) {
    std::string stderrText;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> stderrBuffer =
        llvm::MemoryBuffer::getFile(stderrPath.path);
    if (stderrBuffer)
      stderrText = (*stderrBuffer)->getBuffer().take_front(512).str();
    return makeRVVTargetRouteError(
        llvm::Twine("clang failed to package materialized EmitC C/C++ source "
                    "as a RISC-V RVV relocatable object; exit=") +
        llvm::Twine(result) + " execution_failed=" +
        (executionFailed ? "true" : "false") + " error='" + executeError +
        "' stderr='" + stderrText + "'");
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objectBuffer =
      llvm::MemoryBuffer::getFile(objectPath.path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!objectBuffer)
    return makeRVVTargetRouteError(
        llvm::Twine("failed to read generated RISC-V object: ") +
        objectBuffer.getError().message());
  if ((*objectBuffer)->getBufferSize() == 0)
    return makeRVVTargetRouteError("generated RISC-V object is empty");
  os << (*objectBuffer)->getBuffer();
  return llvm::Error::success();
}

SelectedEmitCArtifactRouteConfig getRVVI32M1ArithmeticArtifactConfig() {
  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();

  SelectedEmitCArtifactRouteConfig config;
  config.routeID = manifest.emitcRoute.routeID;
  config.artifactKind = manifest.emitcRoute.artifactKind;
  config.originPlugin = manifest.family.pluginName;
  config.routeDescription = "RVV i32m1 materialized EmitC target artifact "
                            "bridge";
  config.candidateValidationFn =
      validateRVVI32M1ArithmeticTargetArtifactCandidate;
  config.routeBuilderFn =
      plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRoute;
  return config;
}

llvm::Expected<mlir::emitc::FuncOp>
getSingleMaterializedEmitCFunction(mlir::ModuleOp module,
                                   llvm::StringRef expectedFunctionName) {
  mlir::emitc::FuncOp selectedFunc;
  unsigned functionCount = 0;
  module->walk([&](mlir::emitc::FuncOp func) {
    ++functionCount;
    if (func.getSymName() == expectedFunctionName)
      selectedFunc = func;
  });

  if (!selectedFunc)
    return makeRVVTargetRouteError(
        llvm::Twine("materialized EmitC header route requires EmitC function "
                    "boundary '") +
        expectedFunctionName + "'");
  if (functionCount != 1)
    return makeRVVTargetRouteError(
        "materialized EmitC header route requires exactly one EmitC function "
        "boundary");
  return selectedFunc;
}

std::string formatCParameter(const support::RuntimeABIParameter &parameter) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << parameter.cType;
  llvm::StringRef cType(parameter.cType);
  if (!cType.ends_with("*") && !cType.ends_with("&"))
    os << " ";
  os << parameter.cName;
  os.flush();
  return text;
}

void printRVVMaterializedEmitCHeaderDeclaration(
    llvm::raw_ostream &os, llvm::StringRef functionName,
    llvm::StringRef objectRouteID, llvm::StringRef runtimeABIName,
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    llvm::ArrayRef<support::ArtifactMetadataEntry> artifactMetadata) {
  os << "#ifndef TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H\n";
  os << "#define TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H\n";
  os << "\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n";
  os << "\n";
  os << "/* tianchenrv.rvv.materialized_emitc_header.version: 1 */\n";
  os << "/* tianchenrv.rvv.selected_object_route: " << objectRouteID
     << " */\n";
  os << "/* tianchenrv.rvv.runtime_abi_name: " << runtimeABIName << " */\n";
  os << "/* tianchenrv.rvv.runtime_avl_source: "
     << lookupArtifactMetadataValue(artifactMetadata,
                                    "tcrv_rvv.runtime_avl_source")
     << " */\n";
  os << "/* tianchenrv.rvv.runtime_avl_abi_parameter: "
     << lookupArtifactMetadataValue(artifactMetadata,
                                    "tcrv_rvv.runtime_avl_abi_parameter")
     << " */\n";
  os << "/* tianchenrv.rvv.vl_def: "
     << lookupArtifactMetadataValue(artifactMetadata, "tcrv_rvv.vl_def")
     << " */\n";
  os << "/* tianchenrv.rvv.vl_scope: "
     << lookupArtifactMetadataValue(artifactMetadata, "tcrv_rvv.vl_scope")
     << " */\n";
  os << "/* tianchenrv.rvv.emitc_loop: "
     << lookupArtifactMetadataValue(artifactMetadata, "tcrv_rvv.emitc_loop")
     << " */\n";
  os << "/* tianchenrv.rvv.loop_induction: "
     << lookupArtifactMetadataValue(artifactMetadata,
                                    "tcrv_rvv.loop_induction")
     << " */\n";
  os << "/* tianchenrv.rvv.loop_step: "
     << lookupArtifactMetadataValue(artifactMetadata, "tcrv_rvv.loop_step")
     << " */\n";
  os << "/* tianchenrv.rvv.remaining_avl: "
     << lookupArtifactMetadataValue(artifactMetadata,
                                    "tcrv_rvv.remaining_avl")
     << " */\n";
  os << "/* tianchenrv.rvv.pointer_advance: "
     << lookupArtifactMetadataValue(artifactMetadata,
                                    "tcrv_rvv.pointer_advance")
     << " */\n";
  os << "/* tianchenrv.rvv.bounded_slice: "
     << lookupArtifactMetadataValue(artifactMetadata,
                                    "tcrv_rvv.bounded_slice")
     << " */\n";
  os << "/* tianchenrv.rvv.multi_vl: "
     << lookupArtifactMetadataValue(artifactMetadata, "tcrv_rvv.multi_vl")
     << " */\n";
  os << "\n";
  os << "void " << functionName << "(";
  for (auto [index, parameter] : llvm::enumerate(runtimeABIParameters)) {
    if (index != 0)
      os << ", ";
    os << formatCParameter(parameter);
  }
  os << ");\n";
  os << "\n";
  os << "#endif /* TIANCHENRV_RVV_MATERIALIZED_EMITC_HEADER_H */\n";
}

llvm::Error exportRVVI32M1ArithmeticTargetArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  llvm::Expected<std::string> source = emitSelectedEmitCArtifactCppSource(
      module, getRVVI32M1ArithmeticArtifactConfig());
  if (!source)
    return source.takeError();
  return compileRVVGeneratedSourceToObject(*source, os);
}

llvm::Error exportRVVI32M1ArithmeticHeaderArtifact(mlir::ModuleOp module,
                                                   llvm::raw_ostream &os) {
  SelectedEmitCArtifactRouteConfig config =
      getRVVI32M1ArithmeticArtifactConfig();

  llvm::Expected<SelectedEmitCArtifactTarget> target =
      selectSelectedEmitCArtifactTarget(module, config);
  if (!target)
    return target.takeError();

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> emitcModule =
      materializeSelectedEmitCArtifactModule(module, config);
  if (!emitcModule)
    return emitcModule.takeError();

  llvm::Expected<std::string> functionName =
      getSelectedEmitCArtifactFunctionName(module, config);
  if (!functionName)
    return functionName.takeError();

  llvm::Expected<mlir::emitc::FuncOp> func =
      getSingleMaterializedEmitCFunction(**emitcModule, *functionName);
  if (!func)
    return func.takeError();

  llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters =
      target->candidate.runtimeABIParameters;
  if ((*func).getFunctionType().getNumInputs() !=
      runtimeABIParameters.size())
    return makeRVVTargetRouteError(
        "materialized EmitC header route function boundary arity must match "
        "the selected ordered runtime ABI parameter signature");

  printRVVMaterializedEmitCHeaderDeclaration(
      os, *functionName, target->candidate.routeID,
      target->candidate.runtimeABIName, runtimeABIParameters,
      target->candidate.artifactMetadata);
  return llvm::Error::success();
}

bool isRVVMaterializedEmitCArtifactCandidate(
    const TargetArtifactCandidate &candidate) {
  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();
  return candidate.origin == manifest.family.pluginName ||
         candidate.routeID == manifest.emitcRoute.routeID;
}

llvm::Expected<const TargetArtifactCandidate *>
selectSingleRVVMaterializedEmitCCandidate(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::SmallVector<const TargetArtifactCandidate *, 2> rvvCandidates;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (isRVVMaterializedEmitCArtifactCandidate(candidate))
      rvvCandidates.push_back(&candidate);
  }

  if (rvvCandidates.empty())
    return static_cast<const TargetArtifactCandidate *>(nullptr);
  if (rvvCandidates.size() != 1 || candidates.size() != 1)
    return makeRVVTargetRouteError(
        "RVV materialized EmitC header/bundle route requires exactly one "
        "selected supported RVV materialized EmitC candidate");

  if (llvm::Error error =
          validateRVVI32M1ArithmeticTargetArtifactCandidate(
              *rvvCandidates.front()))
    return std::move(error);
  return rvvCandidates.front();
}

llvm::Expected<bool> matchRVVI32M1ArithmeticHeaderArtifact(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectSingleRVVMaterializedEmitCCandidate(candidates);
  if (!selected)
    return selected.takeError();
  return *selected != nullptr;
}

llvm::Error validateRVVI32M1ArithmeticHeaderArtifactCandidates(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  llvm::Expected<const TargetArtifactCandidate *> selected =
      selectSingleRVVMaterializedEmitCCandidate(candidates);
  if (!selected)
    return selected.takeError();
  if (!*selected)
    return makeRVVTargetRouteError(
        "RVV materialized EmitC header route requires a selected supported "
        "RVV materialized EmitC candidate");
  return llvm::Error::success();
}

llvm::Expected<llvm::SmallVector<support::RuntimeABIParameter, 5>>
getRVVI32M1ArithmeticHeaderRuntimeABIParameters(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  if (llvm::Error error =
          validateRVVI32M1ArithmeticHeaderArtifactCandidates(candidates))
    return std::move(error);

  llvm::SmallVector<support::RuntimeABIParameter, 5> parameters =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  return parameters;
}

llvm::Expected<TargetArtifactCompositeBundleMetadata>
getRVVI32M1ArithmeticHeaderBundleMetadata(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  if (llvm::Error error =
          validateRVVI32M1ArithmeticHeaderArtifactCandidates(candidates))
    return std::move(error);

  TargetArtifactCompositeBundleMetadata metadata;
  metadata.componentGroup = kRVVMaterializedEmitCBundleComponentGroup.str();
  metadata.handoffKind = kRVVI32M1ArithmeticObjectHandoffKind.str();
  return metadata;
}

llvm::Error registerRVVI32M1ArithmeticTargetArtifactExporter(
    TargetArtifactExporterRegistry &registry) {
  if (llvm::Error error = plugin::rvv::verifyRVVConstructionProtocolReady())
    return error;

  const plugin::rvv::RVVConstructionManifest &manifest = getRVVManifest();

  llvm::SmallVector<support::RuntimeABIParameter, 4> runtimeABIParameters =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
  if (!registry.lookup(manifest.emitcRoute.routeID)) {
    if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
            manifest.emitcRoute.routeID, manifest.emitcRoute.artifactKind,
            manifest.family.pluginName, manifest.emitcRoute.emissionKind,
            exportRVVI32M1ArithmeticTargetArtifact, runtimeABIParameters,
            kRVVI32M1ArithmeticObjectHandoffKind,
            validateRVVI32M1ArithmeticTargetArtifactCandidate,
            kRVVMaterializedEmitCBundleComponentGroup)))
      return error;
  }

  if (!registry.lookupComposite(kRVVMaterializedEmitCHeaderRouteID)) {
    if (llvm::Error error = registry.registerCompositeExporter(
            TargetArtifactCompositeExporter(
                kRVVMaterializedEmitCHeaderRouteID,
                kRuntimeCallableCHeaderArtifactKind,
                matchRVVI32M1ArithmeticHeaderArtifact,
                exportRVVI32M1ArithmeticHeaderArtifact,
                manifest.family.pluginName,
                /*runtimeABIKind=*/{}, /*runtimeABIName=*/{},
                getRVVI32M1ArithmeticHeaderRuntimeABIParameters,
                kRVVMaterializedEmitCBundleComponentGroup,
                /*externalABIName=*/{},
                validateRVVI32M1ArithmeticHeaderArtifactCandidates,
                getRVVI32M1ArithmeticHeaderBundleMetadata)))
      return error;
  }

  return llvm::Error::success();
}

} // namespace

llvm::StringRef getRVVMaterializedEmitCTargetArtifactRouteID() {
  return getRVVManifest().emitcRoute.routeID;
}

llvm::StringRef getRVVMaterializedEmitCHeaderArtifactRouteID() {
  return kRVVMaterializedEmitCHeaderRouteID;
}

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  llvm::StringRef pluginName = getRVVManifest().family.pluginName;
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(pluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(pluginName))
      if (bundle.getRegistrationFn() ==
          registerRVVI32M1ArithmeticTargetArtifactExporter)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      pluginName, registerRVVI32M1ArithmeticTargetArtifactExporter));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.addLoweringBoundaryOp(
      plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName());
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (registry.lookup(kRVVEmitCToCppRouteID))
    return llvm::Error::success();

  return registry.registerRoute(TargetTranslateRoute(
      kRVVEmitCToCppRouteID,
      "export a materialized RVV EmitC module through the MLIR EmitC "
      "C/C++ emitter",
      exportMaterializedRVVEmitCToCpp));
}

} // namespace tianchenrv::target::rvv
