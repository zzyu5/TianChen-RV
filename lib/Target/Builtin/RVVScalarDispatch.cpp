#include "TianChenRV/Target/RVVScalarDispatch.h"

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Target/RVV/RVVMicrokernel.h"
#include "TianChenRV/Target/Scalar/ScalarMicrokernel.h"
#include "TianChenRV/Target/TargetArtifactExport.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace tianchenrv::target::rvv_scalar {
namespace {

using tianchenrv::target::TargetArtifactCandidate;
using tianchenrv::target::TargetArtifactExporter;
using tianchenrv::target::TargetArtifactExporterRegistry;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

constexpr llvm::StringLiteral kRVVPluginName("rvv-plugin");
constexpr llvm::StringLiteral kScalarPluginName("scalar-plugin");
constexpr llvm::StringLiteral kDispatchCaseRole("dispatch case");
constexpr llvm::StringLiteral kDispatchFallbackRole("dispatch fallback");
constexpr llvm::StringLiteral kRuntimeCallableCSourceArtifactKind(
    "runtime-callable-c-source");

constexpr llvm::StringLiteral kRVVRouteID("tcrv-export-rvv-microkernel-c");
constexpr llvm::StringLiteral kRVVEmissionKind(
    "rvv-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kRVVRuntimeABI(
    "rvv-i32-vadd-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kRVVRuntimeABIKind(
    "rvv-runtime-callable-c-abi");
constexpr llvm::StringLiteral kRVVRuntimeABIName(
    "rvv-i32-vadd-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kRVVRuntimeGlueRole(
    "runtime-callable-i32-vadd-function");

constexpr llvm::StringLiteral kScalarRouteID(
    "tcrv-export-scalar-microkernel-c");
constexpr llvm::StringLiteral kScalarEmissionKind(
    "scalar-explicit-i32-vadd-microkernel-c-source");
constexpr llvm::StringLiteral kScalarRuntimeABI(
    "scalar-i32-vadd-runtime-callable-c-abi.v1");
constexpr llvm::StringLiteral kScalarRuntimeABIKind(
    "scalar-runtime-callable-c-abi");
constexpr llvm::StringLiteral kScalarRuntimeABIName(
    "scalar-i32-vadd-runtime-callable-c-function.v1");
constexpr llvm::StringLiteral kScalarRuntimeGlueRole(
    "runtime-callable-i32-vadd-fallback-function");

struct DispatchPair {
  TargetArtifactCandidate rvv;
  TargetArtifactCandidate scalar;
};

llvm::Error makeDispatchError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "TianChen-RV RVV+scalar i32-vadd dispatch C export failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::Error makeModuleDispatchError(llvm::Twine message) {
  return llvm::make_error<llvm::StringError>(
      llvm::Twine("TianChen-RV RVV+scalar i32-vadd dispatch C export failed: ") +
          message,
      llvm::errc::invalid_argument);
}

bool hasCandidateShape(const TargetArtifactCandidate &candidate,
                       llvm::StringRef origin, llvm::StringRef role,
                       llvm::StringRef routeID,
                       llvm::StringRef emissionKind,
                       llvm::StringRef runtimeABI,
                       llvm::StringRef runtimeABIKind,
                       llvm::StringRef runtimeABIName,
                       llvm::StringRef runtimeGlueRole) {
  return candidate.origin == origin && candidate.role == role &&
         candidate.routeID == routeID &&
         candidate.emissionKind == emissionKind &&
         candidate.artifactKind == kRuntimeCallableCSourceArtifactKind &&
         candidate.runtimeABI == runtimeABI &&
         candidate.runtimeABIKind == runtimeABIKind &&
         candidate.runtimeABIName == runtimeABIName &&
         candidate.runtimeGlueRole == runtimeGlueRole;
}

llvm::Error validateAgainstRegisteredRoute(
    const TargetArtifactCandidate &candidate,
    const TargetArtifactExporterRegistry &registry) {
  const TargetArtifactExporter *exporter = registry.lookup(candidate.routeID);
  if (!exporter)
    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("unknown selected callable artifact route id '") +
            candidate.routeID + "'");
  return validateTargetArtifactCandidateAgainstExporter(candidate, *exporter);
}

llvm::Error validateCallableI32VAddABIParameters(
    const TargetArtifactCandidate &candidate) {
  llvm::SmallVector<support::RuntimeABIParameter, 4> expected =
      support::getI32VAddRuntimeABIParameters();
  if (support::runtimeABIParametersEqual(candidate.runtimeABIParameters,
                                         expected))
    return llvm::Error::success();

  return makeDispatchError(
      candidate.kernel,
      llvm::Twine("selected callable artifact route '") + candidate.routeID +
          "' must carry structured lhs/rhs/out/n target-export-owned runtime "
          "ABI parameter metadata");
}

llvm::Expected<DispatchPair> collectDispatchPair(mlir::ModuleOp module) {
  llvm::SmallVector<TargetArtifactCandidate, 4> candidates;
  if (llvm::Error error = collectTargetArtifactCandidates(module, candidates))
    return std::move(error);

  TargetArtifactExporterRegistry registry;
  if (llvm::Error error =
          rvv::registerRVVMicrokernelTargetExporters(registry))
    return std::move(error);
  if (llvm::Error error =
          scalar::registerScalarMicrokernelTargetExporters(registry))
    return std::move(error);

  const TargetArtifactCandidate *rvvCandidate = nullptr;
  const TargetArtifactCandidate *scalarCandidate = nullptr;
  for (const TargetArtifactCandidate &candidate : candidates) {
    if (hasCandidateShape(candidate, kRVVPluginName, kDispatchCaseRole,
                          kRVVRouteID, kRVVEmissionKind, kRVVRuntimeABI,
                          kRVVRuntimeABIKind, kRVVRuntimeABIName,
                          kRVVRuntimeGlueRole)) {
      if (rvvCandidate)
        return makeDispatchError(candidate.kernel,
                                 "requires exactly one supported RVV dispatch "
                                 "case callable route; found duplicate");
      if (llvm::Error error =
              validateAgainstRegisteredRoute(candidate, registry))
        return std::move(error);
      if (llvm::Error error =
              validateCallableI32VAddABIParameters(candidate))
        return std::move(error);
      rvvCandidate = &candidate;
      continue;
    }

    if (hasCandidateShape(candidate, kScalarPluginName, kDispatchFallbackRole,
                          kScalarRouteID, kScalarEmissionKind,
                          kScalarRuntimeABI, kScalarRuntimeABIKind,
                          kScalarRuntimeABIName, kScalarRuntimeGlueRole)) {
      if (scalarCandidate)
        return makeDispatchError(
            candidate.kernel,
            "requires exactly one supported scalar dispatch fallback callable "
            "route; found duplicate");
      if (llvm::Error error =
              validateAgainstRegisteredRoute(candidate, registry))
        return std::move(error);
      if (llvm::Error error =
              validateCallableI32VAddABIParameters(candidate))
        return std::move(error);
      scalarCandidate = &candidate;
      continue;
    }

    return makeDispatchError(
        candidate.kernel,
        llvm::Twine("unsupported supported artifact candidate route '") +
            candidate.routeID + "' for RVV+scalar i32-vadd dispatch export");
  }

  if (!rvvCandidate)
    return makeModuleDispatchError(
        "requires exactly one supported RVV dispatch case callable route; "
        "found none");
  if (!scalarCandidate)
    return makeModuleDispatchError(
        "requires exactly one supported scalar dispatch fallback callable "
        "route; found none");
  if (rvvCandidate->kernel != scalarCandidate->kernel)
    return makeModuleDispatchError(
        "requires RVV dispatch case and scalar fallback callable routes in "
        "the same tcrv.exec.kernel");

  DispatchPair pair;
  pair.rvv = *rvvCandidate;
  pair.scalar = *scalarCandidate;
  return pair;
}

std::string sanitizeCIdentifierComponent(llvm::StringRef value) {
  std::string result;
  result.reserve(std::min<std::size_t>(value.size(), 64));
  for (char character : value.take_front(64)) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (std::isalnum(byte))
      result.push_back(character);
    else
      result.push_back('_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

std::string makeRVVFunctionName(const TargetArtifactCandidate &candidate) {
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_rvv_i32_vadd_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeScalarFunctionName(const TargetArtifactCandidate &candidate) {
  KernelOp kernel = candidate.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_scalar_i32_vadd_microkernel_"
         << sanitizeCIdentifierComponent(kernel.getSymName()) << "_"
         << sanitizeCIdentifierComponent(candidate.selectedVariant);
  stream.flush();
  return name;
}

std::string makeDispatcherFunctionName(const DispatchPair &pair) {
  KernelOp kernel = pair.rvv.kernel;
  std::string name;
  llvm::raw_string_ostream stream(name);
  stream << "tcrv_dispatch_i32_vadd_"
         << sanitizeCIdentifierComponent(kernel.getSymName());
  stream.flush();
  return name;
}

VariantOp findDirectVariant(KernelOp kernel, llvm::StringRef symbol) {
  if (!kernel || kernel.getBody().empty())
    return VariantOp();
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto variant = llvm::dyn_cast<VariantOp>(op);
    if (variant && variant.getSymName() == symbol)
      return variant;
  }
  return VariantOp();
}

void printRequiredCapabilitiesComment(llvm::raw_ostream &os,
                                      llvm::StringRef label,
                                      KernelOp kernel,
                                      llvm::StringRef variantSymbol) {
  os << "/* " << label << "_required_capabilities:";
  VariantOp variant = findDirectVariant(kernel, variantSymbol);
  if (variant) {
    if (auto requires = variant->getAttrOfType<mlir::ArrayAttr>("requires")) {
      for (mlir::Attribute attr : requires) {
        auto symbol = llvm::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
        if (symbol)
          os << " @" << symbol.getValue();
      }
    }
  }
  os << " */\n";
}

void printCandidateMetadata(llvm::raw_ostream &os, llvm::StringRef label,
                            const TargetArtifactCandidate &candidate) {
  os << "/* " << label << "_selected_variant: @"
     << candidate.selectedVariant << " */\n";
  os << "/* " << label << "_selected_role: " << candidate.role << " */\n";
  os << "/* " << label << "_origin: " << candidate.origin << " */\n";
  os << "/* " << label << "_lowering_boundary: "
     << candidate.loweringBoundary << " */\n";
  os << "/* " << label << "_emission_kind: " << candidate.emissionKind
     << " */\n";
  os << "/* " << label << "_artifact_kind: " << candidate.artifactKind
     << " */\n";
  os << "/* " << label << "_artifact_route_id: " << candidate.routeID
     << " */\n";
  os << "/* " << label << "_runtime_abi: " << candidate.runtimeABI
     << " */\n";
  os << "/* " << label << "_runtime_abi_kind: "
     << candidate.runtimeABIKind << " */\n";
  os << "/* " << label << "_runtime_abi_name: "
     << candidate.runtimeABIName << " */\n";
  os << "/* " << label << "_runtime_glue_role: "
     << candidate.runtimeGlueRole << " */\n";
  for (auto [index, parameter] :
       llvm::enumerate(candidate.runtimeABIParameters)) {
    os << "/* " << label << "_runtime_abi_parameter[" << index
       << "]: c_name=" << parameter.cName << ", c_type=" << parameter.cType
       << ", role="
       << support::stringifyRuntimeABIParameterRole(parameter.role)
       << ", ownership="
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
       << " */\n";
  }
  printRequiredCapabilitiesComment(os, label, candidate.kernel,
                                   candidate.selectedVariant);
}

llvm::Error buildEmbeddedCallableSources(mlir::ModuleOp module,
                                         std::string &rvvSource,
                                         std::string &scalarSource) {
  llvm::raw_string_ostream rvvStream(rvvSource);
  if (llvm::Error error = rvv::exportRVVMicrokernelC(module, rvvStream))
    return error;
  rvvStream.flush();

  llvm::raw_string_ostream scalarStream(scalarSource);
  if (llvm::Error error = scalar::exportScalarMicrokernelC(module, scalarStream))
    return error;
  scalarStream.flush();
  return llvm::Error::success();
}

void printDispatcherFunction(llvm::raw_ostream &os,
                             llvm::StringRef dispatcherFunctionName,
                             llvm::StringRef rvvFunctionName,
                             llvm::StringRef scalarFunctionName,
                             llvm::ArrayRef<support::RuntimeABIParameter>
                                 parameters) {
  os << "void " << dispatcherFunctionName << "(";
  for (auto [index, parameter] : llvm::enumerate(parameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") {\n";
  os << "  if (rvv_available) {\n";
  os << "    " << rvvFunctionName << "(lhs, rhs, out, n);\n";
  os << "    return;\n";
  os << "  }\n";
  os << "  " << scalarFunctionName << "(lhs, rhs, out, n);\n";
  os << "}\n";
}

void printDispatchSelfCheckHarness(llvm::raw_ostream &os,
                                   llvm::StringRef dispatcherFunctionName) {
  os << "\n/* Explicit bounded self-check harness for RVV+scalar dispatch "
        "runtime invocation evidence. */\n";
  os << "/* Harness scope: calls the generated dispatcher once with "
        "rvv_available = 0 and once with rvv_available = 1. */\n";
  os << "/* Runtime n is a target/export-owned ABI parameter in this harness; "
        "descriptor-local element_count remains metadata only. */\n";
  os << "#include <stdio.h>\n\n";
  os << "static int " << dispatcherFunctionName
     << "_self_check_one(int rvv_available) {\n";
  os << "  const int32_t lhs[16] = {0, 1, 2, 3, 4, 5, 6, 7, "
        "8, 9, 10, 11, 12, 13, 14, 15};\n";
  os << "  const int32_t rhs[16] = {31, 29, 23, 19, 17, 13, 11, 7, "
        "5, 3, 2, 1, -1, -3, -5, -7};\n";
  os << "  int32_t out[16] = {0};\n";
  os << "  " << dispatcherFunctionName
     << "(lhs, rhs, out, 16, rvv_available);\n";
  os << "  for (size_t index = 0; index < 16; ++index) {\n";
  os << "    if (out[index] != lhs[index] + rhs[index])\n";
  os << "      return 1;\n";
  os << "  }\n";
  os << "  return 0;\n";
  os << "}\n\n";
  os << "int main(void) {\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(0))\n";
  os << "    return 1;\n";
  os << "  if (" << dispatcherFunctionName << "_self_check_one(1))\n";
  os << "    return 2;\n";
  os << "  puts(\"tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok\");\n";
  os << "  return 0;\n";
  os << "}\n";
}

void printDispatchSource(const DispatchPair &pair, llvm::StringRef rvvSource,
                         llvm::StringRef scalarSource, bool includeSelfCheck,
                         llvm::raw_ostream &os) {
  KernelOp kernel = pair.rvv.kernel;
  std::string rvvFunctionName = makeRVVFunctionName(pair.rvv);
  std::string scalarFunctionName = makeScalarFunctionName(pair.scalar);
  std::string dispatcherFunctionName = makeDispatcherFunctionName(pair);
  llvm::SmallVector<support::RuntimeABIParameter, 5> dispatchParameters =
      support::getI32VAddDispatchRuntimeABIParameters();

  os << "/* TianChen-RV RVV+scalar host runtime dispatch C export. */\n";
  os << "/* Scope: one selected RVV i32-vadd dispatch case plus one scalar "
        "i32-vadd dispatch fallback. */\n";
  os << "/* Runtime guard: explicit host-provided rvv_available parameter; "
        "no automatic hardware probe is generated. */\n";
  os << "/* selected_kernel: @" << kernel.getSymName() << " */\n";
  printCandidateMetadata(os, "rvv", pair.rvv);
  printCandidateMetadata(os, "scalar", pair.scalar);
  os << "/* rvv_callable_symbol: " << rvvFunctionName << " */\n";
  os << "/* scalar_callable_symbol: " << scalarFunctionName << " */\n";
  for (auto [index, parameter] : llvm::enumerate(dispatchParameters)) {
    os << "/* dispatch_runtime_abi_parameter[" << index
       << "]: c_name=" << parameter.cName << ", c_type=" << parameter.cType
       << ", role="
       << support::stringifyRuntimeABIParameterRole(parameter.role)
       << ", ownership="
       << support::stringifyRuntimeABIParameterOwnership(parameter.ownership)
       << " */\n";
  }
  os << "/* dispatch_runtime_callable_abi: void " << dispatcherFunctionName
     << "(";
  for (auto [index, parameter] : llvm::enumerate(dispatchParameters)) {
    if (index != 0)
      os << ", ";
    support::printRuntimeABIParameterCDeclaration(os, parameter);
  }
  os << ") */\n\n";

  os << "/* Embedded selected RVV runtime-callable source artifact. */\n";
  os << rvvSource;
  if (!rvvSource.ends_with("\n"))
    os << "\n";
  os << "\n/* Embedded selected scalar runtime-callable fallback source "
        "artifact. */\n";
  os << scalarSource;
  if (!scalarSource.ends_with("\n"))
    os << "\n";
  os << "\n/* Host dispatcher over the two validated callable artifacts. */\n";
  printDispatcherFunction(os, dispatcherFunctionName, rvvFunctionName,
                          scalarFunctionName, dispatchParameters);
  if (includeSelfCheck)
    printDispatchSelfCheckHarness(os, dispatcherFunctionName);
}

} // namespace

llvm::Error exportRVVScalarI32VAddDispatchC(mlir::ModuleOp module,
                                            llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair)
    return pair.takeError();

  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(module, rvvSource, scalarSource))
    return error;

  std::string source;
  llvm::raw_string_ostream stream(source);
  printDispatchSource(*pair, rvvSource, scalarSource,
                      /*includeSelfCheck=*/false, stream);
  stream.flush();
  os << source;
  return llvm::Error::success();
}

llvm::Error exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os) {
  llvm::Expected<DispatchPair> pair = collectDispatchPair(module);
  if (!pair)
    return pair.takeError();

  std::string rvvSource;
  std::string scalarSource;
  if (llvm::Error error =
          buildEmbeddedCallableSources(module, rvvSource, scalarSource))
    return error;

  std::string source;
  llvm::raw_string_ostream stream(source);
  printDispatchSource(*pair, rvvSource, scalarSource,
                      /*includeSelfCheck=*/true, stream);
  stream.flush();
  os << source;
  return llvm::Error::success();
}

} // namespace tianchenrv::target::rvv_scalar
