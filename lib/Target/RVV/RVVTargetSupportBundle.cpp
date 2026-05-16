#include "TianChenRV/Target/RVV/RVVTargetSupportBundle.h"

#include "TianChenRV/Plugin/ExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h"
#include "TianChenRV/Target/TargetArtifactExport.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"

#include "mlir/IR/BuiltinOps.h"
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
constexpr llvm::StringLiteral kRVVI32M1AddObjectArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRVVI32M1AddHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRVVI32M1AddObjectHandoffKind(
    "materialized-emitc-cpp-to-riscv-elf-object");
constexpr llvm::StringLiteral kRVVI32M1AddCallableComponentGroup(
    "rvv-i32m1-add-callable-artifact-bundle.v1");

llvm::Error makeRVVObjectRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV i32m1 object artifact route failed: ") +
          message,
      llvm::errc::invalid_argument);
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

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportMaterializedEmitCModuleToCpp(
      module, os, "RVV EmitC C/C++ translate route");
}

SelectedEmitCArtifactRouteConfig getRVVI32M1AddEmitCArtifactConfig() {
  SelectedEmitCArtifactRouteConfig config;
  config.routeID = kRVVI32M1AddObjectArtifactRouteID;
  config.artifactKind = kRVVI32M1AddObjectArtifactKind;
  config.originPlugin = kRVVPluginName;
  config.routeDescription = "RVV i32m1 object artifact route";
  config.candidateValidationFn = validateRVVI32M1AddObjectCandidate;
  config.routeBuilderFn =
      tianchenrv::plugin::rvv::buildRVVI32M1AddEmitCLowerableRoute;
  return config;
}

llvm::Expected<std::string> emitRVVI32M1AddCppSource(mlir::ModuleOp module) {
  return emitSelectedEmitCArtifactCppSource(
      module, getRVVI32M1AddEmitCArtifactConfig());
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
      getSelectedEmitCArtifactFunctionName(
          module, getRVVI32M1AddEmitCArtifactConfig());
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
  os << "/* tianchenrv.runtime_abi_name: "
     << tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIName() << " */\n";
  os << "/* tianchenrv.runtime_glue_role: "
     << tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeGlueRole() << " */\n";
  os << "/* tianchenrv.object_route: "
     << kRVVI32M1AddObjectArtifactRouteID << " */\n";
  os << "/* tianchenrv.header_route: "
     << kRVVI32M1AddHeaderArtifactRouteID << " */\n";
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters =
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIParameters();
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
  if (candidate.runtimeABIKind !=
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIKind())
    return makeRVVObjectRouteError(
        llvm::Twine("runtime ABI kind must be '") +
        tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIKind() +
        "' for route '" +
        kRVVI32M1AddObjectArtifactRouteID + "'");
  if (candidate.runtimeABIName !=
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIName())
    return makeRVVObjectRouteError(
        llvm::Twine("runtime ABI name must be '") +
        tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIName() + "'");
  if (candidate.runtimeGlueRole !=
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeGlueRole())
    return makeRVVObjectRouteError(
        llvm::Twine("runtime glue role must be '") +
        tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeGlueRole() + "'");
  if (candidate.loweringBoundary !=
      tianchenrv::plugin::rvv::getRVVI32M1AddLoweringBoundaryOpName())
    return makeRVVObjectRouteError(
        llvm::Twine("lowering boundary metadata must name '") +
        tianchenrv::plugin::rvv::getRVVI32M1AddLoweringBoundaryOpName() +
        "' for the bounded EmitC route");
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
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIParameters();
  if (!registry.lookup(kRVVI32M1AddObjectArtifactRouteID)) {
    if (llvm::Error error = registry.registerExporter(TargetArtifactExporter(
            kRVVI32M1AddObjectArtifactRouteID, kRVVI32M1AddObjectArtifactKind,
            kRVVPluginName,
            tianchenrv::plugin::rvv::getRVVI32M1AddEmissionKind(),
            exportRVVI32M1AddObjectArtifact, parameters,
            kRVVI32M1AddObjectHandoffKind, validateRVVI32M1AddObjectCandidate,
            kRVVI32M1AddCallableComponentGroup,
            tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIName())))
      return error;
  }

  if (registry.lookupComposite(kRVVI32M1AddHeaderArtifactRouteID))
    return llvm::Error::success();
  return registry.registerCompositeExporter(TargetArtifactCompositeExporter(
      kRVVI32M1AddHeaderArtifactRouteID, kRVVI32M1AddHeaderArtifactKind,
      matchRVVI32M1AddCallableHeaderBundle,
      exportRVVI32M1AddCallableHeaderArtifact, kRVVPluginName,
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIKind(),
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIName(), parameters,
      kRVVI32M1AddCallableComponentGroup,
      tianchenrv::plugin::rvv::getRVVI32M1AddRuntimeABIName(),
      validateRVVI32M1AddCallableHeaderBundle));
}

} // namespace

llvm::StringRef getRVVI32M1AddObjectArtifactRouteID() {
  return kRVVI32M1AddObjectArtifactRouteID;
}

llvm::StringRef getRVVI32M1AddHeaderArtifactRouteID() {
  return kRVVI32M1AddHeaderArtifactRouteID;
}

llvm::StringRef getRVVI32M1AddCallableComponentGroup() {
  return kRVVI32M1AddCallableComponentGroup;
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
