#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Support/RuntimeABICallablePlan.h"
#include "TianChenRV/Support/RuntimeABIMemWindow.h"
#include "TianChenRV/Support/RuntimeABIParam.h"
#include "TianChenRV/Target/RVV/RVVBinaryDescriptor.h"

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
using tianchenrv::support::buildFiniteBinaryCallableABIPlan;
using tianchenrv::support::buildI32BinaryCallableABIPlan;
using tianchenrv::support::getI32BinaryRuntimeABIContract;
using tianchenrv::support::runtimeABIParametersEqual;
using tianchenrv::support::validateFiniteBinaryCallableABIParameterMirror;
using tianchenrv::support::validateI32BinaryCallableABIParameterMirror;
using tianchenrv::target::rvv::getI64VAddFamilyRegistrationRecord;
using tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord;
using tianchenrv::target::rvv::getI64VSubFamilyRegistrationRecord;
using tianchenrv::target::rvv::getRVVBinaryRuntimeABIContract;
using tianchenrv::target::rvv::RVVBinaryFamilyDescriptor;
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

llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan>
buildPlan(mlir::MLIRContext &context, llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return llvm::make_error<llvm::StringError>("failed to parse test module",
                                               llvm::inconvertibleErrorCode());
  KernelOp kernel = findKernel(*module, "abi_kernel");
  return buildI32BinaryCallableABIPlan(
      kernel, getI32BinaryRuntimeABIContract("i32-vsub"));
}

llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
buildFinitePlan(mlir::MLIRContext &context, llvm::StringRef source,
                const RVVBinaryFamilyDescriptor &family) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return llvm::make_error<llvm::StringError>("failed to parse test module",
                                               llvm::inconvertibleErrorCode());
  KernelOp kernel = findKernel(*module, "abi_kernel");
  return buildFiniteBinaryCallableABIPlan(
      kernel, getRVVBinaryRuntimeABIContract(family));
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
          plan->parameters, plan->parameters, "valid mirror", "i32-vsub"),
      "valid runtime ABI mirror");
}

int runI32BinaryFamilyContractCoverageTest() {
  struct ExpectedI32Contract {
    llvm::StringRef familyID;
    llvm::StringRef rvvRuntimeABIName;
    llvm::StringRef scalarRuntimeABIName;
    llvm::StringRef dispatchRuntimeABIName;
  };
  const ExpectedI32Contract families[] = {
      {"i32-vadd", "rvv-i32-vadd-runtime-callable-c-function.v1",
       "scalar-i32-vadd-runtime-callable-c-function.v1",
       "rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1"},
      {"i32-vsub", "rvv-i32-vsub-runtime-callable-c-function.v1",
       "scalar-i32-vsub-runtime-callable-c-function.v1",
       "rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1"},
      {"i32-vmul", "rvv-i32-vmul-runtime-callable-c-function.v1",
       "scalar-i32-vmul-runtime-callable-c-function.v1",
       "rvv-scalar-i32-vmul-dispatch-runtime-callable-c-function.v1"},
  };
  llvm::ArrayRef<RuntimeABIParameter> baselineParameters =
      getI32BinaryRuntimeABIContract(families[0].familyID)
          .getCallableParameters();
  for (const ExpectedI32Contract &family : families) {
    const tianchenrv::support::I32BinaryRuntimeABIContract &contract =
        getI32BinaryRuntimeABIContract(family.familyID);
    if (int result = expect(contract.getFamilyID() == family.familyID,
                            "i32 binary runtime ABI contract is keyed by "
                            "selected family id, not a descriptor record"))
      return result;
    if (int result = expect(contract.getCallableParameters().size() == 4,
                            "i32 binary contract has four callable params"))
      return result;
    if (int result = expect(contract.getCallableRoleRequirements().size() == 4,
                            "i32 binary contract has four role requirements"))
      return result;
    if (int result = expect(contract.getBufferMemWindowSpecs().size() == 3,
                            "i32 binary contract has three mem windows"))
      return result;
    if (int result = expect(
            contract.getRVVCallableIdentity().runtimeABIName ==
                family.rvvRuntimeABIName,
            "RVV callable ABI identity comes from selected family id"))
      return result;
    if (int result = expect(
            contract.getScalarCallableIdentity().runtimeABIName ==
                family.scalarRuntimeABIName,
            "scalar callable ABI identity comes from selected family id"))
      return result;
    if (int result =
            expect(contract.getDispatchIdentity().runtimeABIName ==
                       family.dispatchRuntimeABIName,
                   "dispatch ABI identity comes from selected family id"))
      return result;
    if (int result = expect(
            runtimeABIParametersEqual(contract.getCallableParameters(),
                                      baselineParameters),
            "i32 add/sub/mul share the callable ABI parameter shape"))
      return result;
  }
  return 0;
}

int runValidI64FamilyPlanTests(mlir::MLIRContext &context) {
  const RVVBinaryFamilyDescriptor *families[] = {
      &getI64VAddFamilyRegistrationRecord(), &getI64VSubFamilyRegistrationRecord(),
      &getI64VMulFamilyRegistrationRecord()};
  for (const RVVBinaryFamilyDescriptor *family : families) {
    llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan> plan =
        buildFinitePlan(context, kValidI64CallableABI, *family);
    if (!plan)
      return fail(llvm::Twine("valid i64 callable ABI plan failed for ") +
                  family->familyID + ": " + llvm::toString(plan.takeError()));

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
          duplicateWindowKernel, getI32BinaryRuntimeABIContract("i32-vsub"));
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
          duplicateRuntimeKernel, getI32BinaryRuntimeABIContract("i32-vsub"));
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

int runMalformedI64IRTests(mlir::MLIRContext &context) {
  llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
      missingWindow = buildFinitePlan(context, kMissingI64MemWindow,
                                      getI64VSubFamilyRegistrationRecord());
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
          duplicateWindowKernel,
          getRVVBinaryRuntimeABIContract(getI64VSubFamilyRegistrationRecord()));
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
                                    getI64VSubFamilyRegistrationRecord());
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
                                       getI64VMulFamilyRegistrationRecord());
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
          duplicateRuntimeKernel,
          getRVVBinaryRuntimeABIContract(getI64VAddFamilyRegistrationRecord()));
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
          emptyRuntimeNameKernel,
          getRVVBinaryRuntimeABIContract(getI64VAddFamilyRegistrationRecord()));
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
  llvm::Expected<tianchenrv::support::I32BinaryCallableABIPlan> plan =
      buildI32BinaryCallableABIPlan(
          kernel, getI32BinaryRuntimeABIContract("i32-vadd"));
  if (!plan)
    return fail("failed to build mirror validation plan: " +
                llvm::toString(plan.takeError()));

  llvm::SmallVector<RuntimeABIParameter, 4> missingRuntime(
      plan->parameters.begin(), plan->parameters.end() - 1);
  if (int result = expectErrorContains(
          validateI32BinaryCallableABIParameterMirror(
              kernel, missingRuntime, plan->parameters, "missing mirror",
              "i32-vadd"),
          {"missing mirror requires runtime ABI parameter role "
           "'runtime-element-count'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateRole(
      plan->parameters.begin(), plan->parameters.end());
  duplicateRole.push_back(plan->parameters.front());
  if (int result = expectErrorContains(
          validateI32BinaryCallableABIParameterMirror(
              kernel, duplicateRole, plan->parameters, "duplicate mirror",
              "i32-vadd"),
          {"duplicate mirror contains duplicate runtime ABI parameter role "
           "'lhs-input-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> staleRuntimeName(
      plan->parameters.begin(), plan->parameters.end());
  staleRuntimeName[3].cName = "n";
  if (int result = expectErrorContains(
          validateI32BinaryCallableABIParameterMirror(
              kernel, staleRuntimeName, plan->parameters, "stale mirror",
              "i32-vadd"),
          {"stale mirror runtime ABI parameter role "
           "'runtime-element-count' must mirror IR-backed callable ABI "
           "parameter c_name='len'"}))
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
      buildFiniteBinaryCallableABIPlan(
          kernel,
          getRVVBinaryRuntimeABIContract(getI64VMulFamilyRegistrationRecord()));
  if (!plan)
    return fail("failed to build i64 mirror validation plan: " +
                llvm::toString(plan.takeError()));

  llvm::SmallVector<RuntimeABIParameter, 4> missingOutput(
      plan->parameters.begin(), plan->parameters.end());
  missingOutput.erase(missingOutput.begin() + 2);
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, missingOutput, plan->parameters, "i64 missing mirror",
              getRVVBinaryRuntimeABIContract(getI64VMulFamilyRegistrationRecord())),
          {"i64 missing mirror requires runtime ABI parameter role "
           "'output-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 5> duplicateRole(
      plan->parameters.begin(), plan->parameters.end());
  duplicateRole.push_back(plan->parameters.front());
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, duplicateRole, plan->parameters, "i64 duplicate mirror",
              getRVVBinaryRuntimeABIContract(getI64VMulFamilyRegistrationRecord())),
          {"i64 duplicate mirror contains duplicate runtime ABI parameter role "
           "'lhs-input-buffer'"}))
    return result;

  llvm::SmallVector<RuntimeABIParameter, 4> staleOutputType(
      plan->parameters.begin(), plan->parameters.end());
  staleOutputType[2].cType = "int32_t *";
  if (int result = expectErrorContains(
          validateFiniteBinaryCallableABIParameterMirror(
              kernel, staleOutputType, plan->parameters, "i64 stale mirror",
              getRVVBinaryRuntimeABIContract(getI64VMulFamilyRegistrationRecord())),
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
  if (int result = runI32BinaryFamilyContractCoverageTest())
    return result;
  if (int result = runValidI64FamilyPlanTests(context))
    return result;
  if (int result = runMalformedIRTests(context))
    return result;
  if (int result = runMalformedI64IRTests(context))
    return result;
  if (int result = runMirrorValidationTests(context))
    return result;
  if (int result = runI64MirrorValidationTests(context))
    return result;

  llvm::outs() << "runtime ABI callable plan smoke test passed\n";
  return 0;
}
