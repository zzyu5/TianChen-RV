#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"

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
using tianchenrv::support::FiniteBinaryCallableRuntimeABIParameterBindings;
using tianchenrv::support::FiniteBinaryRuntimeABIContract;
using tianchenrv::support::FiniteBinaryRuntimeABIContractSpec;
using tianchenrv::support::buildFiniteBinaryCallableABIPlan;
using tianchenrv::support::buildRuntimeABIInvocationContract;
using tianchenrv::support::bindFiniteBinaryCallableRuntimeABIParametersByRole;
using tianchenrv::support::formatRuntimeABIInvocationContractCommentBody;
using tianchenrv::support::runtimeABIParametersEqual;
using tianchenrv::support::validateFiniteBinaryCallableABIParameterMirror;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;

namespace {

const FiniteBinaryRuntimeABIContract &getI64AddRuntimeABIContract() {
  static const FiniteBinaryRuntimeABIContract contract(
      FiniteBinaryRuntimeABIContractSpec{"i64-vadd", "const int64_t *",
                                         "int64_t *"});
  return contract;
}

const FiniteBinaryRuntimeABIContract &getI64SubRuntimeABIContract() {
  static const FiniteBinaryRuntimeABIContract contract(
      FiniteBinaryRuntimeABIContractSpec{"i64-vsub", "const int64_t *",
                                         "int64_t *"});
  return contract;
}

const FiniteBinaryRuntimeABIContract &getI64MulRuntimeABIContract() {
  static const FiniteBinaryRuntimeABIContract contract(
      FiniteBinaryRuntimeABIContractSpec{"i64-vmul", "const int64_t *",
                                         "int64_t *"});
  return contract;
}

const FiniteBinaryRuntimeABIContract &getPluginI32RuntimeABIContract() {
  static const FiniteBinaryRuntimeABIContract contract(
      FiniteBinaryRuntimeABIContractSpec{"plugin-owned-i32-binary",
                                         "const int32_t *", "int32_t *"});
  return contract;
}

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

void setRuntimeElementCountCName(KernelOp kernel, llvm::StringRef cName) {
  mlir::OpBuilder builder(kernel.getContext());
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto param = llvm::dyn_cast<RuntimeParamOp>(op);
    if (!param)
      continue;
    auto role = param->getAttrOfType<mlir::StringAttr>(
        tianchenrv::support::kRuntimeParamABIRoleAttrName);
    if (!role || role.getValue() != "runtime-element-count")
      continue;
    param->setAttr(tianchenrv::support::kRuntimeParamCNameAttrName,
                   builder.getStringAttr(cName));
  }
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

constexpr llvm::StringLiteral kValidI64CallableABI = R"mlir(
module {
  tcrv.exec.kernel @abi_kernel {
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len64",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
  }
}
)mlir";

constexpr llvm::StringLiteral kMissingI64MemWindow = R"mlir(
module {
  tcrv.exec.kernel @abi_kernel {
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len64",
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

llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
buildPlan(mlir::MLIRContext &context, llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return llvm::make_error<llvm::StringError>("failed to parse test module",
                                               llvm::inconvertibleErrorCode());
  KernelOp kernel = findKernel(*module, "abi_kernel");
  return buildFiniteBinaryCallableABIPlan(kernel,
                                          getPluginI32RuntimeABIContract());
}

llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
buildFinitePlan(mlir::MLIRContext &context, llvm::StringRef source,
                const FiniteBinaryRuntimeABIContract &contract) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return llvm::make_error<llvm::StringError>("failed to parse test module",
                                               llvm::inconvertibleErrorCode());
  KernelOp kernel = findKernel(*module, "abi_kernel");
  return buildFiniteBinaryCallableABIPlan(kernel, contract);
}

int runValidPlanTest(mlir::MLIRContext &context) {
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> plan =
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
      validateFiniteBinaryCallableABIParameterMirror(
          findKernel(*parseModule(context, kValidI32CallableABI), "abi_kernel"),
          plan->parameters, plan->parameters, "valid mirror",
          getPluginI32RuntimeABIContract()),
      "valid runtime ABI mirror");
}

int runFiniteBinaryContractPrimitiveCoverageTest() {
  const FiniteBinaryRuntimeABIContract &contract =
      getPluginI32RuntimeABIContract();
  if (int result = expect(contract.getFamilyID() == "plugin-owned-i32-binary",
                          "finite binary contract carries caller-owned family "
                          "identifier"))
    return result;
  if (int result = expect(contract.getCallableParameters().size() == 4,
                          "finite binary contract has four callable params"))
    return result;
  if (int result =
          expect(contract.getCallableRoleRequirements().size() == 4,
                 "finite binary contract has four role requirements"))
    return result;
  if (int result = expect(contract.getBufferMemWindowSpecs().size() == 3,
                          "finite binary contract has three mem windows"))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> renamedCallable =
      contract.getCallableParameters("len");
  if (int result = expect(
          renamedCallable.size() == 4 && renamedCallable[3].cName == "len" &&
              renamedCallable[3].role ==
                  RuntimeABIParameterRole::RuntimeElementCount,
          "finite binary callable parameters preserve explicit runtime length "
          "C name"))
    return result;
  if (int result = expect(
          runtimeABIParametersEqual(contract.getCallableParameters(),
                                    getPluginI32RuntimeABIContract()
                                        .getCallableParameters()),
          "finite binary contract exposes stable callable ABI parameter shape"))
    return result;

  tianchenrv::support::RuntimeABIParamSpec countSpec =
      contract.getRuntimeElementCountParamSpec("len");
  if (int result = expect(
          countSpec.cName == "len" &&
              countSpec.role == RuntimeABIParameterRole::RuntimeElementCount,
          "finite binary runtime length spec preserves explicit C name"))
    return result;
  llvm::SmallVector<tianchenrv::support::RuntimeABIParamSpec, 2>
      dispatchSpecs =
          contract.getDispatchRuntimeParamSpecs("len", "dispatch_ready");
  if (int result = expect(
          dispatchSpecs.size() == 2 && dispatchSpecs[0].cName == "len" &&
              dispatchSpecs[0].role ==
                  RuntimeABIParameterRole::RuntimeElementCount &&
              dispatchSpecs[1].cName == "dispatch_ready" &&
              dispatchSpecs[1].role ==
                  RuntimeABIParameterRole::DispatchAvailabilityGuard,
          "finite binary dispatch runtime specs preserve explicit runtime "
          "length and guard C names"))
    return result;
  llvm::SmallVector<RuntimeABIParameter, 5> dispatchParameters =
      contract.getDispatchRuntimeABIParameters("len", "dispatch_ready");
  if (int result = expect(
          dispatchParameters.size() == 5 &&
              dispatchParameters[3].cName == "len" &&
              dispatchParameters[4].cName == "dispatch_ready",
          "finite binary dispatch ABI parameters preserve explicit runtime "
          "length and guard C names"))
    return result;

  llvm::Expected<FiniteBinaryCallableRuntimeABIParameterBindings> bindings =
      bindFiniteBinaryCallableRuntimeABIParametersByRole(
          contract.getCallableParameters(), "finite binary primitive coverage",
          contract);
  if (!bindings)
    return fail(llvm::Twine("finite binary callable ABI binding failed: ") +
                llvm::toString(bindings.takeError()));
  if (int result = expect(bindings->runtimeElementCount->cName == "n",
                          "finite binary callable ABI binding returns the "
                          "runtime element-count role"))
    return result;

  return 0;
}

int runValidI64FamilyPlanTests(mlir::MLIRContext &context) {
  const FiniteBinaryRuntimeABIContract *families[] = {
      &getI64AddRuntimeABIContract(), &getI64SubRuntimeABIContract(),
      &getI64MulRuntimeABIContract()};
  for (const FiniteBinaryRuntimeABIContract *family : families) {
    llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> plan =
        buildFinitePlan(context, kValidI64CallableABI, *family);
    if (!plan)
      return fail(llvm::Twine("valid i64 callable ABI plan failed for ") +
                  family->getFamilyID() + ": " +
                  llvm::toString(plan.takeError()));

    if (int result = expect(plan->parameters.size() == 4,
                            "valid i64 plan has four callable parameters"))
      return result;
    if (int result = expect(plan->parameters[0].cName == "lhs",
                            "i64 lhs C name comes from family ABI contract"))
      return result;
    if (int result = expect(plan->parameters[0].cType == "const int64_t *",
                            "i64 lhs C type comes from exec mem_window"))
      return result;
    if (int result = expect(plan->parameters[2].cType == "int64_t *",
                            "i64 output C type comes from exec mem_window"))
      return result;
    if (int result = expect(plan->parameters[3].cName == "len64",
                            "i64 runtime count C name comes from exec IR"))
      return result;
  }
  return 0;
}

int runMalformedIRTests(mlir::MLIRContext &context) {
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      missingWindow = buildPlan(context, kMissingMemWindow);
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
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      duplicateWindow = buildFiniteBinaryCallableABIPlan(
          duplicateWindowKernel, getPluginI32RuntimeABIContract());
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
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      duplicateRuntime = buildFiniteBinaryCallableABIPlan(
          duplicateRuntimeKernel, getPluginI32RuntimeABIContract());
  if (int result = expectErrorContains(
          duplicateRuntime.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires exactly one tcrv.exec.runtime_param with ABI role "
           "'runtime-element-count'",
           "found duplicate runtime params"}))
    return result;

  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> staleType =
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
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      staleOwnership = buildPlan(context, staleRuntimeOwnership);
  if (int result = expectErrorContains(
          staleOwnership.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires attribute 'ownership' = \"target-export-abi-owned\""}))
    return result;

  return 0;
}

int runMalformedI64IRTests(mlir::MLIRContext &context) {
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      missingWindow = buildFinitePlan(context, kMissingI64MemWindow,
                                      getI64SubRuntimeABIContract());
  if (int result = expectErrorContains(
          missingWindow.takeError(),
          {"runtime ABI mem_window validation failed",
           "requires exactly one tcrv.exec.mem_window with ABI role "
           "'lhs-input-buffer'"}))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> duplicateWindowModule =
      parseModule(context, kValidI64CallableABI);
  if (!duplicateWindowModule)
    return fail("failed to parse duplicate i64 mem_window module");
  KernelOp duplicateWindowKernel =
      findKernel(*duplicateWindowModule, "abi_kernel");
  addDuplicateLHSWindow(duplicateWindowKernel);
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      duplicateWindow = buildFiniteBinaryCallableABIPlan(
          duplicateWindowKernel, getI64SubRuntimeABIContract());
  if (int result = expectErrorContains(
          duplicateWindow.takeError(),
          {"runtime ABI mem_window validation failed",
           "requires exactly one tcrv.exec.mem_window with ABI role "
           "'lhs-input-buffer'",
           "found duplicate windows"}))
    return result;

  std::string staleMemWindowType = replaceFirst(
      kValidI64CallableABI, "c_type = \"const int64_t *\"",
      "c_type = \"const int32_t *\"");
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      staleWindow = buildFinitePlan(context, staleMemWindowType,
                                    getI64SubRuntimeABIContract());
  if (int result = expectErrorContains(
          staleWindow.takeError(),
          {"runtime ABI mem_window validation failed",
           "requires attribute 'c_type' = \"const int64_t *\""}))
    return result;

  std::string staleRuntimeOwnership = replaceFirst(
      kValidI64CallableABI,
      "      c_type = \"size_t\",\n"
      "      ownership = \"target-export-abi-owned\",",
      "      c_type = \"size_t\",\n"
      "      ownership = \"ir-modeled\",");
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      staleOwnership = buildFinitePlan(context, staleRuntimeOwnership,
                                       getI64MulRuntimeABIContract());
  if (int result = expectErrorContains(
          staleOwnership.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires attribute 'ownership' = \"target-export-abi-owned\""}))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> duplicateRuntimeModule =
      parseModule(context, kValidI64CallableABI);
  if (!duplicateRuntimeModule)
    return fail("failed to parse duplicate i64 runtime_param module");
  KernelOp duplicateRuntimeKernel =
      findKernel(*duplicateRuntimeModule, "abi_kernel");
  addDuplicateRuntimeParam(duplicateRuntimeKernel);
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      duplicateRuntime = buildFiniteBinaryCallableABIPlan(
          duplicateRuntimeKernel, getI64AddRuntimeABIContract());
  if (int result = expectErrorContains(
          duplicateRuntime.takeError(),
          {"runtime ABI runtime_param validation failed",
           "requires exactly one tcrv.exec.runtime_param with ABI role "
           "'runtime-element-count'",
           "found duplicate runtime params"}))
    return result;

  mlir::OwningOpRef<mlir::ModuleOp> emptyRuntimeNameModule =
      parseModule(context, kValidI64CallableABI);
  if (!emptyRuntimeNameModule)
    return fail("failed to parse empty-runtime-name i64 module");
  KernelOp emptyRuntimeNameKernel =
      findKernel(*emptyRuntimeNameModule, "abi_kernel");
  setRuntimeElementCountCName(emptyRuntimeNameKernel, "");
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      emptyRuntimeName = buildFiniteBinaryCallableABIPlan(
          emptyRuntimeNameKernel, getI64AddRuntimeABIContract());
  if (int result = expectErrorContains(
          emptyRuntimeName.takeError(),
          {"finite binary runtime ABI callable plan validation failed",
           "requires non-empty C parameter name for ABI role "
           "'runtime-element-count'"}))
    return result;

  return 0;
}

int runMirrorValidationTests(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kValidI32CallableABI);
  if (!module)
    return fail("failed to parse mirror validation module");
  KernelOp kernel = findKernel(*module, "abi_kernel");
  const FiniteBinaryRuntimeABIContract &contract =
      getPluginI32RuntimeABIContract();
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> plan =
      buildFiniteBinaryCallableABIPlan(kernel, contract);
  if (!plan)
    return fail("failed to build mirror validation plan: " +
                llvm::toString(plan.takeError()));

  llvm::SmallVector<RuntimeABIParameter, 4> missingRuntime(
      plan->parameters.begin(), plan->parameters.end() - 1);
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, missingRuntime, plan->parameters, "missing mirror",
              contract),
          {"missing mirror requires runtime ABI parameter role "
           "'runtime-element-count'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateRole(
      plan->parameters.begin(), plan->parameters.end());
  duplicateRole.push_back(plan->parameters.front());
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, duplicateRole, plan->parameters, "duplicate mirror",
              contract),
          {"duplicate mirror contains duplicate runtime ABI parameter role "
           "'lhs-input-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> staleRuntimeName(
      plan->parameters.begin(), plan->parameters.end());
  staleRuntimeName[3].cName = "n";
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, staleRuntimeName, plan->parameters, "stale mirror",
              contract),
          {"stale mirror runtime ABI parameter role "
           "'runtime-element-count' must mirror IR-backed callable ABI "
           "parameter c_name='len'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> reordered(
      plan->parameters.begin(), plan->parameters.end());
  RuntimeABIParameter first = reordered[0];
  reordered[0] = reordered[1];
  reordered[1] = first;
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, reordered, plan->parameters, "reordered mirror",
              contract),
          {"reordered mirror must preserve IR-backed callable ABI parameter "
           "order at index 0",
           "expected role 'lhs-input-buffer'",
           "found role 'rhs-input-buffer'"}))
    return result;

  return 0;
}

int runInvocationContractTests(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kValidI32CallableABI);
  if (!module)
    return fail("failed to parse invocation contract module");
  KernelOp kernel = findKernel(*module, "abi_kernel");
  const FiniteBinaryRuntimeABIContract &contract =
      getPluginI32RuntimeABIContract();
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> plan =
      buildFiniteBinaryCallableABIPlan(kernel, contract);
  if (!plan)
    return fail(llvm::Twine("failed to build invocation contract plan: ") +
                llvm::toString(plan.takeError()));

  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract> direct =
      buildRuntimeABIInvocationContract(
          kernel, contract.getFamilyID(), plan->parameters,
          "plugin-direct-invocation-contract", "plugin_direct_callable",
          "plugin-direct-runtime-abi-kind", "plugin.direct.runtime.fn",
          "plugin-direct-runtime-glue", "len", "plugin-target-export");
  if (!direct)
    return fail(llvm::Twine("valid direct invocation contract failed: ") +
                llvm::toString(direct.takeError()));
  std::string directBody = formatRuntimeABIInvocationContractCommentBody(
      "runtime_abi_invocation_contract", *direct);
  if (int result = expect(
          llvm::StringRef(directBody)
              .contains("runtime_abi_invocation_contract: "
                        "source=plugin-direct-invocation-contract"),
          "direct invocation contract records caller-owned source label"))
    return result;
  if (int result = expect(llvm::StringRef(directBody)
                              .contains("family=plugin-owned-i32-binary"),
                          "direct invocation contract carries caller-owned "
                          "family identifier"))
    return result;
  if (int result = expect(
          llvm::StringRef(directBody)
              .contains("ordered_roles=lhs-input-buffer->rhs-input-buffer->"
                        "output-buffer->runtime-element-count"),
          "direct invocation contract preserves ordered callable roles"))
    return result;
  if (int result = expect(
          llvm::StringRef(directBody)
              .contains("runtime_element_count_c_name=len"),
          "direct invocation contract carries IR-backed runtime length C name"))
    return result;

  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract>
      missingFamily = buildRuntimeABIInvocationContract(
          kernel, "", plan->parameters, "plugin-direct-invocation-contract",
          "plugin_direct_callable", "plugin-direct-runtime-abi-kind",
          "plugin.direct.runtime.fn", "plugin-direct-runtime-glue", "len",
          "plugin-target-export");
  if (int result = expectErrorContains(
          missingFamily.takeError(),
          {"runtime ABI invocation contract field 'family'",
           "requires non-empty bounded text"}))
    return result;

  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract>
      staleRuntimeName = buildRuntimeABIInvocationContract(
          kernel, contract.getFamilyID(), plan->parameters,
          "plugin-direct-invocation-contract", "plugin_direct_callable",
          "plugin-direct-runtime-abi-kind", "plugin.direct.runtime.fn",
          "plugin-direct-runtime-glue", "n", "plugin-target-export");
  if (int result = expectErrorContains(
          staleRuntimeName.takeError(),
          {"runtime ABI invocation contract runtime_element_count_c_name 'n'",
           "must match ordered ABI parameter role 'runtime-element-count' "
           "C name 'len'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> directWithGuard(
      plan->parameters.begin(), plan->parameters.end());
  directWithGuard.push_back(
      contract.getDispatchAvailabilityGuardParameter("dispatch_ready"));
  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract>
      directGuard = buildRuntimeABIInvocationContract(
          kernel, contract.getFamilyID(), directWithGuard,
          "plugin-direct-invocation-contract", "plugin_direct_callable",
          "plugin-direct-runtime-abi-kind", "plugin.direct.runtime.fn",
          "plugin-direct-runtime-glue", "len", "plugin-target-export");
  if (int result = expectErrorContains(
          directGuard.takeError(),
          {"direct callable must not carry a dispatch-availability-guard"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> dispatchParameters =
      contract.getDispatchRuntimeABIParameters("len", "dispatch_ready");
  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract> dispatch =
      buildRuntimeABIInvocationContract(
          kernel, contract.getFamilyID(), dispatchParameters,
          "plugin-dispatch-runtime-abi-contract", "plugin_dispatch_callable",
          "plugin-dispatch-runtime-abi-kind", "plugin.dispatch.runtime.fn",
          /*runtimeGlueRole=*/"", "len", "plugin-dispatch-target",
          "dispatch_ready");
  if (!dispatch)
    return fail(llvm::Twine("valid dispatch invocation contract failed: ") +
                llvm::toString(dispatch.takeError()));
  std::string dispatchBody = formatRuntimeABIInvocationContractCommentBody(
      "dispatch_runtime_abi_invocation_contract", *dispatch);
  if (int result = expect(
          llvm::StringRef(dispatchBody)
              .contains("family=plugin-owned-i32-binary"),
          "dispatch invocation contract carries caller-owned family identifier"))
    return result;
  if (int result = expect(
          llvm::StringRef(dispatchBody)
              .contains("ordered_roles=lhs-input-buffer->rhs-input-buffer->"
                        "output-buffer->runtime-element-count->"
                        "dispatch-availability-guard"),
          "dispatch invocation contract preserves ordered dispatch roles"))
    return result;
  if (int result = expect(
          llvm::StringRef(dispatchBody)
              .contains("dispatch_guard_c_name=dispatch_ready"),
          "dispatch invocation contract carries dispatch guard C name"))
    return result;
  if (int result = expect(
          !llvm::StringRef(dispatchBody).contains("runtime_glue_role="),
          "dispatch invocation contract omits absent runtime glue role"))
    return result;

  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract>
      missingGuard = buildRuntimeABIInvocationContract(
          kernel, contract.getFamilyID(), plan->parameters,
          "plugin-dispatch-runtime-abi-contract", "plugin_dispatch_callable",
          "plugin-dispatch-runtime-abi-kind", "plugin.dispatch.runtime.fn",
          /*runtimeGlueRole=*/"", "len", "plugin-dispatch-target",
          "dispatch_ready");
  if (int result = expectErrorContains(
          missingGuard.takeError(),
          {"requires exactly one dispatch-availability-guard ABI parameter"}))
    return result;

  llvm::Expected<tianchenrv::support::RuntimeABIInvocationContract>
      staleGuard = buildRuntimeABIInvocationContract(
          kernel, contract.getFamilyID(), dispatchParameters,
          "plugin-dispatch-runtime-abi-contract", "plugin_dispatch_callable",
          "plugin-dispatch-runtime-abi-kind", "plugin.dispatch.runtime.fn",
          /*runtimeGlueRole=*/"", "len", "plugin-dispatch-target",
          "other_dispatch_ready");
  if (int result = expectErrorContains(
          staleGuard.takeError(),
          {"dispatch_guard_c_name 'other_dispatch_ready'",
           "must match ordered ABI parameter role "
           "'dispatch-availability-guard' C name 'dispatch_ready'"}))
    return result;

  return 0;
}

int runI64MirrorValidationTests(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kValidI64CallableABI);
  if (!module)
    return fail("failed to parse i64 mirror validation module");
  KernelOp kernel = findKernel(*module, "abi_kernel");
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> plan =
      buildFiniteBinaryCallableABIPlan(kernel, getI64MulRuntimeABIContract());
  if (!plan)
    return fail("failed to build i64 mirror validation plan: " +
                llvm::toString(plan.takeError()));

  llvm::SmallVector<RuntimeABIParameter, 4> missingOutput(
      plan->parameters.begin(), plan->parameters.end());
  missingOutput.erase(missingOutput.begin() + 2);
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, missingOutput, plan->parameters, "i64 missing mirror",
              getI64MulRuntimeABIContract()),
          {"i64 missing mirror requires runtime ABI parameter role "
           "'output-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateRole(
      plan->parameters.begin(), plan->parameters.end());
  duplicateRole.push_back(plan->parameters.front());
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, duplicateRole, plan->parameters, "i64 duplicate mirror",
              getI64MulRuntimeABIContract()),
          {"i64 duplicate mirror contains duplicate runtime ABI parameter role "
           "'lhs-input-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> staleOutputType(
      plan->parameters.begin(), plan->parameters.end());
  staleOutputType[2].cType = "int32_t *";
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, staleOutputType, plan->parameters, "i64 stale mirror",
              getI64MulRuntimeABIContract()),
          {"i64 stale mirror runtime ABI parameter role 'output-buffer' "
           "must mirror IR-backed callable ABI parameter",
           "c_type='int64_t *'"}))
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
  if (int result = runFiniteBinaryContractPrimitiveCoverageTest())
    return result;
  if (int result = runValidI64FamilyPlanTests(context))
    return result;
  if (int result = runMalformedIRTests(context))
    return result;
  if (int result = runMalformedI64IRTests(context))
    return result;
  if (int result = runMirrorValidationTests(context))
    return result;
  if (int result = runInvocationContractTests(context))
    return result;
  if (int result = runI64MirrorValidationTests(context))
    return result;

  llvm::outs() << "runtime ABI callable plan smoke test passed\n";
  return 0;
}
