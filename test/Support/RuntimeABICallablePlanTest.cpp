#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/I32BinaryFamilyRegistry.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <initializer_list>
#include <string>

using tianchenrv::support::RuntimeABIParameter;
using tianchenrv::support::RuntimeABIParameterRole;
using tianchenrv::support::buildI32BinaryCallableABIPlan;
using tianchenrv::support::validateI32BinaryCallableABIParameterMirror;
using tianchenrv::target::i32_binary::getI32VAddFamilyDescriptor;
using tianchenrv::target::i32_binary::getI32VSubFamilyDescriptor;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected ABI callable plan error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("ABI callable plan error text missing '") +
                  fragment + "': " + message);
  }
  return 0;
}

std::string replaceFirst(llvm::StringRef source, llvm::StringRef from,
                         llvm::StringRef to) {
  std::string text = source.str();
  std::size_t pos = text.find(from.str());
  if (pos == std::string::npos)
    return text;
  text.replace(pos, from.size(), to.str());
  return text;
}

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef symbolName) {
  KernelOp result;
  module.walk([&](KernelOp kernel) {
    if (kernel.getSymName() == symbolName)
      result = kernel;
  });
  return result;
}

void addDuplicateLHSWindow(KernelOp kernel) {
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  mlir::OperationState state(kernel.getLoc(), MemWindowOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr("abi_lhs_input_buffer_duplicate"));
  state.addAttribute(tianchenrv::support::kMemWindowPurposeAttrName,
                     builder.getStringAttr(
                         tianchenrv::support::kRuntimeABIBufferPurpose));
  state.addAttribute(tianchenrv::support::kMemWindowBindingAttrName,
                     builder.getStringAttr(
                         tianchenrv::support::kRuntimeABIKernelArgumentBinding));
  state.addAttribute(tianchenrv::support::kMemWindowMemorySpaceAttrName,
                     builder.getStringAttr(
                         tianchenrv::support::kRuntimeABIHostMemorySpace));
  state.addAttribute(tianchenrv::support::kMemWindowABIRoleAttrName,
                     builder.getStringAttr("lhs-input-buffer"));
  state.addAttribute(tianchenrv::support::kMemWindowAccessAttrName,
                     builder.getStringAttr(
                         tianchenrv::support::kRuntimeABIReadAccess));
  state.addAttribute(tianchenrv::support::kMemWindowOwnershipAttrName,
                     builder.getStringAttr("target-export-abi-owned"));
  state.addAttribute(tianchenrv::support::kMemWindowCTypeAttrName,
                     builder.getStringAttr("const int32_t *"));
  builder.create(state);
}

void addDuplicateRuntimeParam(KernelOp kernel) {
  mlir::OpBuilder builder(kernel.getContext());
  builder.setInsertionPointToEnd(&kernel.getBody().front());

  mlir::OperationState state(kernel.getLoc(),
                             RuntimeParamOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr("abi_runtime_element_count_alias"));
  state.addAttribute(tianchenrv::support::kRuntimeParamPurposeAttrName,
                     builder.getStringAttr(
                         tianchenrv::support::kRuntimeABIScalarParamPurpose));
  state.addAttribute(tianchenrv::support::kRuntimeParamABIRoleAttrName,
                     builder.getStringAttr("runtime-element-count"));
  state.addAttribute(tianchenrv::support::kRuntimeParamCNameAttrName,
                     builder.getStringAttr("len_alias"));
  state.addAttribute(tianchenrv::support::kRuntimeParamCTypeAttrName,
                     builder.getStringAttr("size_t"));
  state.addAttribute(tianchenrv::support::kRuntimeParamOwnershipAttrName,
                     builder.getStringAttr("target-export-abi-owned"));
  builder.create(state);
}

constexpr llvm::StringLiteral kValidI32CallableABI = R"mlir(
module {
  tcrv.exec.kernel @abi_kernel {
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
  }
}
)mlir";

constexpr llvm::StringLiteral kMissingMemWindow = R"mlir(
module {
  tcrv.exec.kernel @abi_kernel {
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
  }
}
)mlir";

constexpr llvm::StringLiteral kStaleRuntimeType = R"mlir(
module {
  tcrv.exec.kernel @abi_kernel {
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len",
      c_type = "uint64_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
  }
}
)mlir";

llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan>
buildPlan(mlir::MLIRContext &context, llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return llvm::make_error<llvm::StringError>("failed to parse test module",
                                               llvm::inconvertibleErrorCode());
  KernelOp kernel = findKernel(*module, "abi_kernel");
  return buildI32BinaryCallableABIPlan(kernel, getI32VSubFamilyDescriptor());
}

int runValidPlanTest(mlir::MLIRContext &context) {
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan> plan =
      buildPlan(context, kValidI32CallableABI);
  if (!plan)
    return fail("valid i32 callable ABI plan failed: " +
                llvm::toString(plan.takeError()));

  if (int result = expect(plan->parameters.size() == 4,
                          "valid plan has four callable ABI parameters"))
    return result;
  if (int result = expect(plan->bufferWindows.size() == 3,
                          "valid plan has three buffer windows"))
    return result;
  if (int result = expect(plan->parameters[0].cName == "lhs",
                          "lhs C name comes from shared i32 ABI contract"))
    return result;
  if (int result = expect(plan->parameters[1].cName == "rhs",
                          "rhs C name comes from shared i32 ABI contract"))
    return result;
  if (int result = expect(plan->parameters[2].cName == "out",
                          "out C name comes from shared i32 ABI contract"))
    return result;
  if (int result = expect(plan->parameters[3].cName == "len",
                          "runtime n C name comes from runtime_param IR"))
    return result;
  if (int result = expect(plan->parameters[3].role ==
                              RuntimeABIParameterRole::RuntimeElementCount,
                          "runtime parameter role is runtime-element-count"))
    return result;

  return expectSuccess(
      validateI32BinaryCallableABIParameterMirror(
          findKernel(*parseModule(context, kValidI32CallableABI), "abi_kernel"),
          plan->parameters, plan->parameters, "valid mirror",
          getI32VSubFamilyDescriptor()),
      "valid runtime ABI mirror");
}

int runMalformedIRTests(mlir::MLIRContext &context) {
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan> missingWindow =
      buildPlan(context, kMissingMemWindow);
  if (int result = expectErrorContains(
          missingWindow.takeError(),
          {"runtime ABI mem_window validation failed",
           "requires exactly one tcrv.exec.mem_window with ABI role "
           "'lhs-input-buffer'"}))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> duplicateWindowModule =
      parseModule(context, kValidI32CallableABI);
  if (!duplicateWindowModule)
    return fail("failed to parse duplicate mem_window module");
  KernelOp duplicateWindowKernel =
      findKernel(*duplicateWindowModule, "abi_kernel");
  addDuplicateLHSWindow(duplicateWindowKernel);
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan>
      duplicateWindow = buildI32BinaryCallableABIPlan(
          duplicateWindowKernel, getI32VSubFamilyDescriptor());
  if (int result = expectErrorContains(
          duplicateWindow.takeError(),
          {"runtime ABI mem_window validation failed",
           "requires exactly one tcrv.exec.mem_window with ABI role "
           "'lhs-input-buffer'",
           "found duplicate windows"}))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> duplicateRuntimeModule =
      parseModule(context, kValidI32CallableABI);
  if (!duplicateRuntimeModule)
    return fail("failed to parse duplicate runtime_param module");
  KernelOp duplicateRuntimeKernel =
      findKernel(*duplicateRuntimeModule, "abi_kernel");
  addDuplicateRuntimeParam(duplicateRuntimeKernel);
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan>
      duplicateRuntime = buildI32BinaryCallableABIPlan(
          duplicateRuntimeKernel, getI32VSubFamilyDescriptor());
  if (int result = expectErrorContains(
          duplicateRuntime.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires exactly one tcrv.exec.runtime_param with ABI role "
           "'runtime-element-count'",
           "found duplicate runtime params"}))
    return result;

  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan> staleType =
      buildPlan(context, kStaleRuntimeType);
  if (int result = expectErrorContains(
          staleType.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires attribute 'c_type' = \"size_t\""}))
    return result;

  std::string staleRuntimeOwnership = replaceFirst(
      kValidI32CallableABI,
      "      c_type = \"size_t\",\n"
      "      ownership = \"target-export-abi-owned\",",
      "      c_type = \"size_t\",\n"
      "      ownership = \"ir-modeled\",");
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan>
      staleOwnership = buildPlan(context, staleRuntimeOwnership);
  if (int result = expectErrorContains(
          staleOwnership.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires attribute 'ownership' = \"target-export-abi-owned\""}))
    return result;

  return 0;
}

int runMirrorValidationTests(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kValidI32CallableABI);
  if (!module)
    return fail("failed to parse mirror validation module");
  KernelOp kernel = findKernel(*module, "abi_kernel");
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan> plan =
      buildI32BinaryCallableABIPlan(kernel, getI32VAddFamilyDescriptor());
  if (!plan)
    return fail("failed to build mirror validation plan: " +
                llvm::toString(plan.takeError()));

  llvm::SmallVector<RuntimeABIParameter, 4> missingRuntime(
      plan->parameters.begin(), plan->parameters.end() - 1);
  if (int result = expectErrorContains(
          validateI32BinaryCallableABIParameterMirror(
              kernel, missingRuntime, plan->parameters, "missing mirror",
              getI32VAddFamilyDescriptor()),
          {"missing mirror requires runtime ABI parameter role "
           "'runtime-element-count'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateRole(
      plan->parameters.begin(), plan->parameters.end());
  duplicateRole.push_back(plan->parameters.front());
  if (int result = expectErrorContains(
          validateI32BinaryCallableABIParameterMirror(
              kernel, duplicateRole, plan->parameters, "duplicate mirror",
              getI32VAddFamilyDescriptor()),
          {"duplicate mirror contains duplicate runtime ABI parameter role "
           "'lhs-input-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> staleRuntimeName(
      plan->parameters.begin(), plan->parameters.end());
  staleRuntimeName[3].cName = "n";
  if (int result = expectErrorContains(
          validateI32BinaryCallableABIParameterMirror(
              kernel, staleRuntimeName, plan->parameters, "stale mirror",
              getI32VAddFamilyDescriptor()),
          {"stale mirror runtime ABI parameter role "
           "'runtime-element-count' must mirror IR-backed callable ABI "
           "parameter c_name='len'"}))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  tianchenrv::registerAllDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (int result = runValidPlanTest(context))
    return result;
  if (int result = runMalformedIRTests(context))
    return result;
  if (int result = runMirrorValidationTests(context))
    return result;

  llvm::outs() << "runtime ABI callable plan smoke test passed\n";
  return 0;
}
