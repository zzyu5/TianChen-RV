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
constexpr llvm::StringLiteral kRVVI32M1ArithmeticObjectArtifactKind(
    "riscv-elf-relocatable-object");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticHeaderArtifactKind(
    "runtime-callable-c-header");
constexpr llvm::StringLiteral kRVVI32M1ArithmeticObjectHandoffKind(
    "materialized-emitc-cpp-to-riscv-elf-object");

struct RVVI32M1ArithmeticTargetRouteDescriptor {
  plugin::rvv::RVVI32M1ArithmeticOp op;
  llvm::StringLiteral mnemonic;
  llvm::StringLiteral objectArtifactRouteID;
  llvm::StringLiteral headerArtifactRouteID;
  llvm::StringLiteral objectTranslateRouteID;
  llvm::StringLiteral headerTranslateRouteID;
  llvm::StringLiteral callableComponentGroup;
  TargetArtifactExportFn objectExportFn;
  TargetArtifactExportFn headerExportFn;
  TargetArtifactCompositeMatchFn headerMatchFn;
  TargetArtifactCompositeCandidateValidationFn headerValidationFn;
  SelectedEmitCArtifactRouteBuilderFn routeBuilderFn;
};

llvm::Error makeRVVObjectRouteError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine(
          "TianChen-RV RVV i32m1 arithmetic object artifact route failed: ") +
          message,
      llvm::errc::invalid_argument);
}

llvm::Error exportRVVI32M1AddObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);
llvm::Error exportRVVI32M1SubObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);
llvm::Error exportRVVI32M1MulObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os);
llvm::Error exportRVVI32M1AddCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os);
llvm::Error exportRVVI32M1SubCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os);
llvm::Error exportRVVI32M1MulCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os);
llvm::Expected<bool>
matchRVVI32M1AddCallableHeaderBundle(llvm::ArrayRef<TargetArtifactCandidate>);
llvm::Expected<bool>
matchRVVI32M1SubCallableHeaderBundle(llvm::ArrayRef<TargetArtifactCandidate>);
llvm::Expected<bool>
matchRVVI32M1MulCallableHeaderBundle(llvm::ArrayRef<TargetArtifactCandidate>);
llvm::Error validateRVVI32M1AddCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate>);
llvm::Error validateRVVI32M1SubCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate>);
llvm::Error validateRVVI32M1MulCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate>);

constexpr RVVI32M1ArithmeticTargetRouteDescriptor
    kRVVI32M1ArithmeticTargetRoutes[] = {
        {plugin::rvv::RVVI32M1ArithmeticOp::Add,
         "add",
         "tcrv-rvv-i32m1-add-riscv-elf-object",
         "tcrv-rvv-i32m1-add-callable-c-header",
         "tcrv-rvv-i32m1-add-object",
         "tcrv-rvv-i32m1-add-header",
         "rvv-i32m1-add-callable-artifact-bundle.v1",
         exportRVVI32M1AddObjectArtifact,
         exportRVVI32M1AddCallableHeaderArtifact,
         matchRVVI32M1AddCallableHeaderBundle,
         validateRVVI32M1AddCallableHeaderBundle,
         plugin::rvv::buildRVVI32M1AddEmitCLowerableRoute},
        {plugin::rvv::RVVI32M1ArithmeticOp::Sub,
         "sub",
         "tcrv-rvv-i32m1-sub-riscv-elf-object",
         "tcrv-rvv-i32m1-sub-callable-c-header",
         "tcrv-rvv-i32m1-sub-object",
         "tcrv-rvv-i32m1-sub-header",
         "rvv-i32m1-sub-callable-artifact-bundle.v1",
         exportRVVI32M1SubObjectArtifact,
         exportRVVI32M1SubCallableHeaderArtifact,
         matchRVVI32M1SubCallableHeaderBundle,
         validateRVVI32M1SubCallableHeaderBundle,
         plugin::rvv::buildRVVI32M1SubEmitCLowerableRoute},
        {plugin::rvv::RVVI32M1ArithmeticOp::Mul,
         "mul",
         "tcrv-rvv-i32m1-mul-riscv-elf-object",
         "tcrv-rvv-i32m1-mul-callable-c-header",
         "tcrv-rvv-i32m1-mul-object",
         "tcrv-rvv-i32m1-mul-header",
         "rvv-i32m1-mul-callable-artifact-bundle.v1",
         exportRVVI32M1MulObjectArtifact,
         exportRVVI32M1MulCallableHeaderArtifact,
         matchRVVI32M1MulCallableHeaderBundle,
         validateRVVI32M1MulCallableHeaderBundle,
         plugin::rvv::buildRVVI32M1MulEmitCLowerableRoute},
};

const RVVI32M1ArithmeticTargetRouteDescriptor &
getRVVI32M1ArithmeticTargetRouteDescriptor(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  for (const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor :
       kRVVI32M1ArithmeticTargetRoutes)
    if (descriptor.op == op)
      return descriptor;
  llvm_unreachable("unknown RVV i32m1 arithmetic target route");
}

llvm::Expected<const RVVI32M1ArithmeticTargetRouteDescriptor *>
getRVVI32M1ArithmeticTargetRouteDescriptorByObjectRoute(
    llvm::StringRef routeID) {
  for (const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor :
       kRVVI32M1ArithmeticTargetRoutes)
    if (routeID == descriptor.objectArtifactRouteID)
      return &descriptor;
  return makeRVVObjectRouteError(
      llvm::Twine("unknown RVV i32m1 arithmetic object route '") + routeID +
      "'");
}

std::string makeHeaderGuard(llvm::StringRef functionName,
                            llvm::StringRef mnemonic) {
  std::string guard("TIANCHENRV_RVV_I32M1_");
  for (char c : mnemonic) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte))
      guard.push_back(static_cast<char>(std::toupper(byte)));
    else
      guard.push_back('_');
  }
  guard.append("_CALLABLE_");
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

llvm::Error validateRVVI32M1ArithmeticObjectCandidate(
    const TargetArtifactCandidate &candidate);

llvm::Error exportMaterializedRVVEmitCToCpp(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportMaterializedEmitCModuleToCpp(
      module, os, "RVV EmitC C/C++ translate route");
}

SelectedEmitCArtifactRouteConfig getRVVI32M1ArithmeticEmitCArtifactConfig(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor =
      getRVVI32M1ArithmeticTargetRouteDescriptor(op);
  SelectedEmitCArtifactRouteConfig config;
  config.routeID = descriptor.objectArtifactRouteID;
  config.artifactKind = kRVVI32M1ArithmeticObjectArtifactKind;
  config.originPlugin = kRVVPluginName;
  config.routeDescription = "RVV i32m1 arithmetic object artifact route";
  config.candidateValidationFn = validateRVVI32M1ArithmeticObjectCandidate;
  config.routeBuilderFn = descriptor.routeBuilderFn;
  return config;
}

llvm::Expected<std::string> emitRVVI32M1ArithmeticCppSource(
    mlir::ModuleOp module, plugin::rvv::RVVI32M1ArithmeticOp op) {
  return emitSelectedEmitCArtifactCppSource(
      module, getRVVI32M1ArithmeticEmitCArtifactConfig(op));
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
compileGeneratedSourceToRiscvObject(llvm::StringRef generatedSource,
                                    llvm::StringRef mnemonic) {
  llvm::Expected<std::string> clangPath = findRVVClangTool();
  if (!clangPath)
    return clangPath.takeError();

  int sourceFD = -1;
  llvm::SmallString<128> sourcePath;
  std::string temporaryStem =
      (llvm::Twine("tcrv-rvv-i32m1-") + mnemonic).str();
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          temporaryStem, "c", sourceFD, sourcePath))
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create generated C/C++ temporary source: ") +
        ec.message());

  int objectFD = -1;
  llvm::SmallString<128> objectPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          temporaryStem, "o", objectFD, objectPath)) {
    llvm::sys::fs::remove(sourcePath);
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create RISC-V object temporary output: ") +
        ec.message());
  }

  int stdoutFD = -1;
  llvm::SmallString<128> stdoutPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          temporaryStem, "stdout", stdoutFD, stdoutPath)) {
    llvm::sys::fs::remove(sourcePath);
    llvm::sys::fs::remove(objectPath);
    return makeRVVObjectRouteError(
        llvm::Twine("failed to create clang stdout capture: ") +
        ec.message());
  }

  int stderrFD = -1;
  llvm::SmallString<128> stderrPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
          temporaryStem, "stderr", stderrFD, stderrPath)) {
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

llvm::Error exportRVVI32M1ArithmeticObjectArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor =
      getRVVI32M1ArithmeticTargetRouteDescriptor(op);
  llvm::Expected<std::string> generatedSource =
      emitRVVI32M1ArithmeticCppSource(module, op);
  if (!generatedSource)
    return generatedSource.takeError();

  llvm::Expected<std::string> objectBytes =
      compileGeneratedSourceToRiscvObject(*generatedSource,
                                          descriptor.mnemonic);
  if (!objectBytes)
    return objectBytes.takeError();

  os.write(objectBytes->data(), objectBytes->size());
  return llvm::Error::success();
}

llvm::Error exportRVVI32M1AddObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportRVVI32M1ArithmeticObjectArtifact(
      module, os, plugin::rvv::RVVI32M1ArithmeticOp::Add);
}

llvm::Error exportRVVI32M1SubObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportRVVI32M1ArithmeticObjectArtifact(
      module, os, plugin::rvv::RVVI32M1ArithmeticOp::Sub);
}

llvm::Error exportRVVI32M1MulObjectArtifact(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  return exportRVVI32M1ArithmeticObjectArtifact(
      module, os, plugin::rvv::RVVI32M1ArithmeticOp::Mul);
}

llvm::Error exportRVVI32M1ArithmeticCallableHeaderArtifact(
    mlir::ModuleOp module, llvm::raw_ostream &os,
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor =
      getRVVI32M1ArithmeticTargetRouteDescriptor(op);
  llvm::Expected<std::string> functionName =
      getSelectedEmitCArtifactFunctionName(
          module, getRVVI32M1ArithmeticEmitCArtifactConfig(op));
  if (!functionName)
    return functionName.takeError();

  std::string guard = makeHeaderGuard(*functionName, descriptor.mnemonic);
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n\n";
  os << "#include <stddef.h>\n";
  os << "#include <stdint.h>\n\n";
  os << "#ifdef __cplusplus\n";
  os << "extern \"C\" {\n";
  os << "#endif\n\n";
  os << "/* tianchenrv.runtime_abi_name: "
     << plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(op) << " */\n";
  os << "/* tianchenrv.runtime_glue_role: "
     << plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole() << " */\n";
  os << "/* tianchenrv.object_route: "
     << descriptor.objectArtifactRouteID << " */\n";
  os << "/* tianchenrv.header_route: "
     << descriptor.headerArtifactRouteID << " */\n";
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();
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

llvm::Error exportRVVI32M1AddCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  return exportRVVI32M1ArithmeticCallableHeaderArtifact(
      module, os, plugin::rvv::RVVI32M1ArithmeticOp::Add);
}

llvm::Error exportRVVI32M1SubCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  return exportRVVI32M1ArithmeticCallableHeaderArtifact(
      module, os, plugin::rvv::RVVI32M1ArithmeticOp::Sub);
}

llvm::Error exportRVVI32M1MulCallableHeaderArtifact(mlir::ModuleOp module,
                                                    llvm::raw_ostream &os) {
  return exportRVVI32M1ArithmeticCallableHeaderArtifact(
      module, os, plugin::rvv::RVVI32M1ArithmeticOp::Mul);
}

llvm::Error validateRVVI32M1ArithmeticObjectCandidate(
    const TargetArtifactCandidate &candidate) {
  llvm::Expected<const RVVI32M1ArithmeticTargetRouteDescriptor *> descriptor =
      getRVVI32M1ArithmeticTargetRouteDescriptorByObjectRoute(
          candidate.routeID);
  if (!descriptor)
    return descriptor.takeError();

  if (candidate.runtimeABIKind !=
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind())
    return makeRVVObjectRouteError(
        llvm::Twine("runtime ABI kind must be '") +
        plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind() +
        "' for route '" + (*descriptor)->objectArtifactRouteID + "'");
  if (candidate.runtimeABIName !=
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName((*descriptor)->op))
    return makeRVVObjectRouteError(
        llvm::Twine("runtime ABI name must be '") +
        plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName((*descriptor)->op) +
        "'");
  if (candidate.runtimeGlueRole !=
      plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole())
    return makeRVVObjectRouteError(
        llvm::Twine("runtime glue role must be '") +
        plugin::rvv::getRVVI32M1ArithmeticRuntimeGlueRole() + "'");
  if (candidate.loweringBoundary !=
      plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName())
    return makeRVVObjectRouteError(
        llvm::Twine("lowering boundary metadata must name '") +
        plugin::rvv::getRVVI32M1ArithmeticLoweringBoundaryOpName() +
        "' for the bounded EmitC route");
  return llvm::Error::success();
}

llvm::Expected<bool> matchRVVI32M1ArithmeticCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  if (candidates.size() != 1)
    return false;
  const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor =
      getRVVI32M1ArithmeticTargetRouteDescriptor(op);
  const TargetArtifactCandidate &candidate = candidates.front();
  return candidate.routeID == descriptor.objectArtifactRouteID &&
         candidate.artifactKind == kRVVI32M1ArithmeticObjectArtifactKind &&
         candidate.origin == kRVVPluginName;
}

llvm::Expected<bool>
matchRVVI32M1AddCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVI32M1ArithmeticCallableHeaderBundle(
      candidates, plugin::rvv::RVVI32M1ArithmeticOp::Add);
}

llvm::Expected<bool>
matchRVVI32M1SubCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVI32M1ArithmeticCallableHeaderBundle(
      candidates, plugin::rvv::RVVI32M1ArithmeticOp::Sub);
}

llvm::Expected<bool>
matchRVVI32M1MulCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return matchRVVI32M1ArithmeticCallableHeaderBundle(
      candidates, plugin::rvv::RVVI32M1ArithmeticOp::Mul);
}

llvm::Error validateRVVI32M1ArithmeticCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates,
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor =
      getRVVI32M1ArithmeticTargetRouteDescriptor(op);
  if (candidates.size() != 1)
    return makeRVVObjectRouteError(
        "callable header bundle route requires exactly one RVV i32m1 "
        "arithmetic object candidate");
  const TargetArtifactCandidate &candidate = candidates.front();
  if (candidate.routeID != descriptor.objectArtifactRouteID)
    return makeRVVObjectRouteError(
        llvm::Twine("callable header bundle route requires object route '") +
        descriptor.objectArtifactRouteID + "'");
  if (candidate.artifactKind != kRVVI32M1ArithmeticObjectArtifactKind)
    return makeRVVObjectRouteError(
        llvm::Twine("callable header bundle route requires artifact_kind '") +
        kRVVI32M1ArithmeticObjectArtifactKind + "'");
  return validateRVVI32M1ArithmeticObjectCandidate(candidate);
}

llvm::Error validateRVVI32M1AddCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return validateRVVI32M1ArithmeticCallableHeaderBundle(
      candidates, plugin::rvv::RVVI32M1ArithmeticOp::Add);
}

llvm::Error validateRVVI32M1SubCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return validateRVVI32M1ArithmeticCallableHeaderBundle(
      candidates, plugin::rvv::RVVI32M1ArithmeticOp::Sub);
}

llvm::Error validateRVVI32M1MulCallableHeaderBundle(
    llvm::ArrayRef<TargetArtifactCandidate> candidates) {
  return validateRVVI32M1ArithmeticCallableHeaderBundle(
      candidates, plugin::rvv::RVVI32M1ArithmeticOp::Mul);
}

llvm::Error registerRVVI32M1ArithmeticObjectExporters(
    TargetArtifactExporterRegistry &registry) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> parameters =
      plugin::rvv::getRVVI32M1ArithmeticRuntimeABIParameters();

  for (const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor :
       kRVVI32M1ArithmeticTargetRoutes) {
    if (!registry.lookup(descriptor.objectArtifactRouteID)) {
      if (llvm::Error error =
              registry.registerExporter(TargetArtifactExporter(
                  descriptor.objectArtifactRouteID,
                  kRVVI32M1ArithmeticObjectArtifactKind, kRVVPluginName,
                  plugin::rvv::getRVVI32M1ArithmeticEmissionKind(),
                  descriptor.objectExportFn, parameters,
                  kRVVI32M1ArithmeticObjectHandoffKind,
                  validateRVVI32M1ArithmeticObjectCandidate,
                  descriptor.callableComponentGroup,
                  plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(
                      descriptor.op))))
        return error;
    }

    if (!registry.lookupComposite(descriptor.headerArtifactRouteID)) {
      if (llvm::Error error = registry.registerCompositeExporter(
              TargetArtifactCompositeExporter(
                  descriptor.headerArtifactRouteID,
                  kRVVI32M1ArithmeticHeaderArtifactKind,
                  descriptor.headerMatchFn, descriptor.headerExportFn,
                  kRVVPluginName,
                  plugin::rvv::getRVVI32M1ArithmeticRuntimeABIKind(),
                  plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(
                      descriptor.op),
                  parameters, descriptor.callableComponentGroup,
                  plugin::rvv::getRVVI32M1ArithmeticRuntimeABIName(
                      descriptor.op),
                  descriptor.headerValidationFn)))
        return error;
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::StringRef getRVVI32M1ArithmeticObjectArtifactRouteID(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticTargetRouteDescriptor(op).objectArtifactRouteID;
}

llvm::StringRef getRVVI32M1ArithmeticHeaderArtifactRouteID(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticTargetRouteDescriptor(op).headerArtifactRouteID;
}

llvm::StringRef getRVVI32M1ArithmeticCallableComponentGroup(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticTargetRouteDescriptor(op).callableComponentGroup;
}

llvm::StringRef getRVVI32M1ArithmeticObjectTranslateRouteID(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticTargetRouteDescriptor(op).objectTranslateRouteID;
}

llvm::StringRef getRVVI32M1ArithmeticHeaderTranslateRouteID(
    plugin::rvv::RVVI32M1ArithmeticOp op) {
  return getRVVI32M1ArithmeticTargetRouteDescriptor(op).headerTranslateRouteID;
}

llvm::Error registerRVVTargetSupportPluginTargetExporterBundles(
    PluginTargetArtifactExporterRegistry &registry) {
  if (const PluginTargetArtifactExporterBundle *existing =
          registry.lookup(kRVVPluginName)) {
    for (const PluginTargetArtifactExporterBundle &bundle :
         registry.lookupAll(kRVVPluginName))
      if (bundle.getRegistrationFn() ==
          registerRVVI32M1ArithmeticObjectExporters)
        return llvm::Error::success();
    (void)existing;
  }
  return registry.registerBundle(PluginTargetArtifactExporterBundle(
      kRVVPluginName, registerRVVI32M1ArithmeticObjectExporters));
}

llvm::Error
configureRVVTargetSupportExtensionBundle(ExtensionBundle &bundle) {
  bundle.setTargetArtifactExporterBundleRegistrationFn(
      registerRVVTargetSupportPluginTargetExporterBundles);
  return llvm::Error::success();
}

llvm::Error registerRVVTargetSupportTargetTranslateRoutes(
    TargetTranslateRouteRegistry &registry) {
  if (!registry.lookup(kRVVEmitCToCppRouteID)) {
    if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
            kRVVEmitCToCppRouteID,
            "export a materialized RVV EmitC module through the MLIR EmitC "
            "C/C++ emitter",
            exportMaterializedRVVEmitCToCpp)))
      return error;
  }

  for (const RVVI32M1ArithmeticTargetRouteDescriptor &descriptor :
       kRVVI32M1ArithmeticTargetRoutes) {
    if (!registry.lookup(descriptor.objectTranslateRouteID)) {
      if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
              descriptor.objectTranslateRouteID,
              "export a selected RVV i32m1 arithmetic path through EmitC, the "
              "MLIR EmitC C/C++ emitter, and clang RISC-V relocatable object "
              "compilation",
              descriptor.objectExportFn,
              /*requiresBinaryStdout=*/true,
              descriptor.objectArtifactRouteID)))
        return error;
    }
    if (!registry.lookup(descriptor.headerTranslateRouteID)) {
      if (llvm::Error error = registry.registerRoute(TargetTranslateRoute(
              descriptor.headerTranslateRouteID,
              "export a selected RVV i32m1 arithmetic callable C header for "
              "the generated RISC-V relocatable object ABI",
              descriptor.headerExportFn,
              /*requiresBinaryStdout=*/false,
              descriptor.headerArtifactRouteID)))
        return error;
    }
  }
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv
