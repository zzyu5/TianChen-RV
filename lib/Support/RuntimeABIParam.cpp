#include "TianChenRV/Support/RuntimeABIParam.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

namespace tianchenrv::support {
namespace {

using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::RuntimeParamOp;

llvm::Error makeRuntimeParamError(KernelOp kernel, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "runtime ABI runtime_param validation failed";
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

llvm::StringRef getRoleName(const RuntimeABIParamSpec &spec) {
  return stringifyRuntimeABIParameterRole(spec.role);
}

llvm::Error requireAttrEquals(KernelOp kernel, RuntimeParamOp param,
                              llvm::StringRef attrName,
                              llvm::StringRef expected) {
  if (expected.empty())
    return llvm::Error::success();

  llvm::StringRef actual = getStringAttr(param.getOperation(), attrName);
  if (actual != expected)
    return makeRuntimeParamError(
        kernel, llvm::Twine("tcrv.exec.runtime_param @") +
                    param.getSymName() + " requires attribute '" + attrName +
                    "' = \"" + expected + "\" for ABI role '" +
                    getStringAttr(param.getOperation(),
                                  kRuntimeParamABIRoleAttrName) +
                    "\"");
  return llvm::Error::success();
}

llvm::Error validateParamAgainstSpec(KernelOp kernel, RuntimeParamOp param,
                                     const RuntimeABIParamSpec &spec) {
  if (!param)
    return makeRuntimeParamError(
        kernel, llvm::Twine("requires exactly one tcrv.exec.runtime_param "
                            "with ABI role '") +
                    getRoleName(spec) + "'");

  if (llvm::Error error =
          requireAttrEquals(kernel, param, kRuntimeParamPurposeAttrName,
                            kRuntimeABIScalarParamPurpose))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, param, kRuntimeParamABIRoleAttrName,
                            getRoleName(spec)))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, param, kRuntimeParamCNameAttrName,
                            spec.cName))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, param, kRuntimeParamCTypeAttrName,
                            spec.cType))
    return error;
  if (llvm::Error error =
          requireAttrEquals(kernel, param, kRuntimeParamOwnershipAttrName,
                            spec.ownership))
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

llvm::Error findParamForSpec(KernelOp kernel, const RuntimeABIParamSpec &spec,
                             RuntimeParamOp &out) {
  if (!kernel || kernel.getBody().empty())
    return makeRuntimeParamError(
        kernel, "requires a materialized tcrv.exec.kernel body");

  unsigned count = 0;
  for (mlir::Operation &op : kernel.getBody().front()) {
    auto param = llvm::dyn_cast<RuntimeParamOp>(op);
    if (!param)
      continue;

    auto role = param->getAttrOfType<mlir::StringAttr>(
        kRuntimeParamABIRoleAttrName);
    if (!role || role.getValue() != getRoleName(spec))
      continue;

    out = param;
    ++count;
  }

  if (count > 1)
    return makeRuntimeParamError(
        kernel, llvm::Twine("requires exactly one tcrv.exec.runtime_param "
                            "with ABI role '") +
                    getRoleName(spec) + "'; found duplicate runtime params");

  return llvm::Error::success();
}

RuntimeParamOp createParam(mlir::OpBuilder &builder, KernelOp kernel,
                           const RuntimeABIParamSpec &spec) {
  mlir::OperationState state(kernel.getLoc(),
                             RuntimeParamOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(spec.symbolName));
  state.addAttribute(kRuntimeParamPurposeAttrName,
                     builder.getStringAttr(kRuntimeABIScalarParamPurpose));
  state.addAttribute(kRuntimeParamABIRoleAttrName,
                     builder.getStringAttr(getRoleName(spec)));
  state.addAttribute(kRuntimeParamCNameAttrName,
                     builder.getStringAttr(spec.cName));
  state.addAttribute(kRuntimeParamCTypeAttrName,
                     builder.getStringAttr(spec.cType));
  state.addAttribute(kRuntimeParamOwnershipAttrName,
                     builder.getStringAttr(spec.ownership));
  return llvm::cast<RuntimeParamOp>(builder.create(state));
}

} // namespace

llvm::Error ensureRuntimeABIParams(
    KernelOp kernel, mlir::OpBuilder &builder,
    llvm::ArrayRef<RuntimeABIParamSpec> specs) {
  if (!kernel || kernel.getBody().empty())
    return makeRuntimeParamError(
        kernel, "requires a materialized tcrv.exec.kernel body");

  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directSymbols);

  for (const RuntimeABIParamSpec &spec : specs) {
    RuntimeParamOp existing;
    if (llvm::Error error = findParamForSpec(kernel, spec, existing))
      return error;
    if (existing) {
      if (llvm::Error error =
              validateParamAgainstSpec(kernel, existing, spec))
        return error;
      continue;
    }

    auto symbolIt = directSymbols.find(spec.symbolName);
    if (symbolIt != directSymbols.end())
      return makeRuntimeParamError(
          kernel, llvm::Twine("direct symbol @") + spec.symbolName +
                      " already exists and cannot be reused for "
                      "tcrv.exec.runtime_param ABI role '" +
                      getRoleName(spec) + "'");

    RuntimeParamOp created = createParam(builder, kernel, spec);
    directSymbols.try_emplace(created.getSymName(), created.getOperation());
  }

  return llvm::Error::success();
}

llvm::Error ensureRuntimeABIParamsAllowingExistingCNames(
    KernelOp kernel, mlir::OpBuilder &builder,
    llvm::ArrayRef<RuntimeABIParamSpec> specs) {
  if (!kernel || kernel.getBody().empty())
    return makeRuntimeParamError(
        kernel, "requires a materialized tcrv.exec.kernel body");

  llvm::StringMap<mlir::Operation *> directSymbols;
  collectDirectKernelSymbols(kernel, directSymbols);

  for (const RuntimeABIParamSpec &spec : specs) {
    RuntimeParamOp existing;
    if (llvm::Error error = findParamForSpec(kernel, spec, existing))
      return error;
    if (existing) {
      RuntimeABIParamSpec validationSpec = spec;
      validationSpec.cName.clear();
      if (llvm::Error error =
              validateParamAgainstSpec(kernel, existing, validationSpec))
        return error;
      continue;
    }

    auto symbolIt = directSymbols.find(spec.symbolName);
    if (symbolIt != directSymbols.end())
      return makeRuntimeParamError(
          kernel, llvm::Twine("direct symbol @") + spec.symbolName +
                      " already exists and cannot be reused for "
                      "tcrv.exec.runtime_param ABI role '" +
                      getRoleName(spec) + "'");

    RuntimeParamOp created = createParam(builder, kernel, spec);
    directSymbols.try_emplace(created.getSymName(), created.getOperation());
  }

  return llvm::Error::success();
}

llvm::Error collectRuntimeABIParams(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParamSpec> specs,
    llvm::SmallVectorImpl<RuntimeParamOp> &out) {
  for (const RuntimeABIParamSpec &spec : specs) {
    RuntimeParamOp param;
    if (llvm::Error error = findParamForSpec(kernel, spec, param))
      return error;
    if (!param)
      return makeRuntimeParamError(
          kernel, llvm::Twine("requires exactly one tcrv.exec.runtime_param "
                              "with ABI role '") +
                      getRoleName(spec) + "'");
    if (llvm::Error error = validateParamAgainstSpec(kernel, param, spec))
      return error;
    out.push_back(param);
  }
  return llvm::Error::success();
}

llvm::Error validateRuntimeABIParams(
    KernelOp kernel, llvm::ArrayRef<RuntimeABIParamSpec> specs) {
  llvm::SmallVector<RuntimeParamOp, 4> ignored;
  return collectRuntimeABIParams(kernel, specs, ignored);
}

} // namespace tianchenrv::support
