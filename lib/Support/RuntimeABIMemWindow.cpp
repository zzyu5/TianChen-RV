#include "TianChenRV/Support/RuntimeABIMemWindow.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::support {
namespace {

using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::MemWindowOp;

llvm::Error makeMemWindowError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "runtime ABI mem_window validation failed";
  if (kernel)
    stream << " for kernel @" << kernel.getSymName();
  else
    stream << " for kernel <missing>";
  stream << ": " << message;
  stream.flush();
  return llvm::make_error<llvm::StringError>(text,
                                             llvm::errc::invalid_argument);
}

llvm::StringRef getStringAttr(mlir::Operation *op, llvm::StringRef attrName) {
  auto attr = op ? op->getAttrOfType<mlir::StringAttr>(attrName)
                 : mlir::StringAttr();
  if (!attr)
    return {};
  return attr.getValue();
}

llvm::StringRef getRoleName(const RuntimeABIMemWindowSpec &spec) {
  return stringifyRuntimeABIParameterRole(spec.role);
}

llvm::StringRef getExpectedOwnership(const RuntimeABIMemWindowSpec &spec) {
  return spec.ownership;
}

llvm::StringRef getExpectedCType(const RuntimeABIMemWindowSpec &spec) {
  return spec.cType;
}

llvm::StringRef getExpectedAccess(const RuntimeABIMemWindowSpec &spec) {
  return spec.access;
}

llvm::Error requireAttrEquals(KernelOp kernel, MemWindowOp window,
                              llvm::StringRef attrName,
                              llvm::StringRef expected) {
  llvm::StringRef actual = getStringAttr(window.getOperation(), attrName);
  if (actual != expected)
    return makeMemWindowError(
        kernel, llvm::Twine("tcrv.exec.mem_window @") + window.getSymName() +
                    " requires attribute '" + attrName + "' = \"" + expected +
                    "\" for ABI role '" +
                    getStringAttr(window.getOperation(),
                                  kMemWindowABIRoleAttrName) +
                    "\"");
  return llvm::Error::success();
}

llvm::Error validateWindowAgainstSpec(KernelOp kernel, MemWindowOp window,
                                      const RuntimeABIMemWindowSpec &spec) {
  if (!window)
    return makeMemWindowError(
        kernel, llvm::Twine("requires exactly one tcrv.exec.mem_window with "
                            "ABI role '") +
                    getRoleName(spec) + "'");

  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowPurposeAttrName,
                            kRuntimeABIBufferPurpose))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowBindingAttrName,
                            kRuntimeABIKernelArgumentBinding))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowMemorySpaceAttrName,
                            kRuntimeABIHostMemorySpace))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowABIRoleAttrName,
                            getRoleName(spec)))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowAccessAttrName,
                            getExpectedAccess(spec)))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowOwnershipAttrName,
                            getExpectedOwnership(spec)))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, window, kMemWindowCTypeAttrName,
                            getExpectedCType(spec)))
    return error;
  return llvm::Error::success();
}

void collectDirectKernelSymbols(KernelOp kernel,
                                llvm::StringMap<mlir::Operation *> &out) {
  if (!kernel || kernel.getBody().empty())
    return;

  for (mlir::Operation &op : kernel.getBody().front()) {
    auto symbol = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbol)
      out.try_emplace(symbol.getValue(), &op);
  }
}

llvm::Error findWindowForSpec(KernelOp kernel,
                              const RuntimeABIMemWindowSpec &spec,
                              MemWindowOp &out) {
  if (!kernel || kernel.getBody().empty())
    return makeMemWindowError(kernel,
                              "requires a materialized tcrv.exec.kernel body");

  unsigned count = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto window = llvm::dyn_cast<MemWindowOp>(op);
    if (!window)
      continue;

    auto role = window->getAttrOfType<mlir::StringAttr>(
        kMemWindowABIRoleAttrName);
    if (!role || role.getValue() != getRoleName(spec))
      continue;

    out = window;
    ++count;
  }

  if (count > 1)
    return makeMemWindowError(
        kernel, llvm::Twine("requires exactly one tcrv.exec.mem_window with "
                            "ABI role '") +
                    getRoleName(spec) + "'; found duplicate windows");

  return llvm::Error::success();
}

MemWindowOp createWindow(mlir::OpBuilder &builder, KernelOp kernel,
                         const RuntimeABIMemWindowSpec &spec) {
  mlir::OperationState state(kernel.getLoc(), MemWindowOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(spec.symbolName));
  state.addAttribute(kMemWindowPurposeAttrName,
                     builder.getStringAttr(kRuntimeABIBufferPurpose));
  state.addAttribute(kMemWindowBindingAttrName,
                     builder.getStringAttr(kRuntimeABIKernelArgumentBinding));
  state.addAttribute(kMemWindowMemorySpaceAttrName,
                     builder.getStringAttr(kRuntimeABIHostMemorySpace));
  state.addAttribute(kMemWindowABIRoleAttrName,
                     builder.getStringAttr(getRoleName(spec)));
  state.addAttribute(kMemWindowAccessAttrName,
                     builder.getStringAttr(spec.access));
  state.addAttribute(kMemWindowOwnershipAttrName,
                     builder.getStringAttr(spec.ownership));
  state.addAttribute(kMemWindowCTypeAttrName,
                     builder.getStringAttr(spec.cType));
  return llvm::cast<MemWindowOp>(builder.create(state));
}

} // namespace

llvm::SmallVector<RuntimeABIMemWindowSpec, 3>
getI32VAddBufferMemWindowSpecs() {
  llvm::SmallVector<RuntimeABIMemWindowSpec, 3> specs;
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_lhs_input_buffer", RuntimeABIParameterRole::LHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "const int32_t *"));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_rhs_input_buffer", RuntimeABIParameterRole::RHSInputBuffer,
      kRuntimeABIReadAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "const int32_t *"));
  specs.push_back(RuntimeABIMemWindowSpec(
      "abi_output_buffer", RuntimeABIParameterRole::OutputBuffer,
      kRuntimeABIWriteAccess,
      stringifyRuntimeABIParameterOwnership(
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "int32_t *"));
  return specs;
}

llvm::Error ensureRuntimeABIBufferMemWindows(
    KernelOp kernel, mlir::OpBuilder &builder,
    llvm::ArrayRef<RuntimeABIMemWindowSpec> specs) {
  if (!kernel || kernel.getBody().empty())
    return makeMemWindowError(kernel,
                              "requires a materialized tcrv.exec.kernel body");

  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directSymbols);

  for (const RuntimeABIMemWindowSpec &spec : specs) {
    MemWindowOp existing;
    if (llvm::Error error = findWindowForSpec(kernel, spec, existing))
      return error;
    if (existing) {
      if (llvm::Error error = validateWindowAgainstSpec(kernel, existing, spec))
        return error;
      continue;
    }

    auto symbolIt = directSymbols.find(spec.symbolName);
    if (symbolIt != directSymbols.end())
      return makeMemWindowError(
          kernel, llvm::Twine("direct symbol @") + spec.symbolName +
                      " already exists and cannot be reused for "
                      "tcrv.exec.mem_window ABI role '" +
                      getRoleName(spec) + "'");

    MemWindowOp created = createWindow(builder, kernel, spec);
    directSymbols.try_emplace(created.getSymName(), created.getOperation());
  }

  return llvm::Error::success();
}

llvm::Error collectRuntimeABIBufferMemWindows(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIMemWindowSpec> specs,
    llvm::SmallVectorImpl<MemWindowOp> &out) {
  for (const RuntimeABIMemWindowSpec &spec : specs) {
    MemWindowOp window;
    if (llvm::Error error = findWindowForSpec(kernel, spec, window))
      return error;
    if (!window)
      return makeMemWindowError(
          kernel, llvm::Twine("requires exactly one tcrv.exec.mem_window with "
                              "ABI role '") +
                      getRoleName(spec) + "'");
    if (llvm::Error error = validateWindowAgainstSpec(kernel, window, spec))
      return error;
    out.push_back(window);
  }
  return llvm::Error::success();
}

llvm::Error validateRuntimeABIBufferMemWindows(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIMemWindowSpec> specs) {
  llvm::SmallVector<MemWindowOp, 4> ignored;
  return collectRuntimeABIBufferMemWindows(kernel, specs, ignored);
}

} // namespace tianchenrv::support
