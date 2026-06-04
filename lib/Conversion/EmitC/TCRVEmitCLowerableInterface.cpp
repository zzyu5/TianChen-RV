#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <utility>

namespace tianchenrv::conversion::emitc {
namespace {

llvm::Error makeRouteError(llvm::StringRef routeID, llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "TianChen-RV EmitC lowerable route";
  if (!routeID.empty())
    os << " '" << routeID << "'";
  os << " is invalid: " << message;
  os.flush();
  return llvm::createStringError(llvm::errc::invalid_argument, text);
}

bool isBoundedSingleLineText(llvm::StringRef value) {
  if (value.empty() || value.size() > 256)
    return false;
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (c == '\n' || c == '\r' || byte == 0)
      return false;
    if (byte < 0x20 && c != '\t')
      return false;
  }
  return true;
}

llvm::Error validateText(llvm::StringRef routeID, llvm::Twine fieldName,
                         llvm::StringRef value) {
  if (isBoundedSingleLineText(value))
    return llvm::Error::success();
  return makeRouteError(routeID, llvm::Twine(fieldName) +
                                     " must be bounded non-empty single-line "
                                     "text");
}

llvm::Error validateHeader(llvm::StringRef routeID,
                           const TCRVEmitCHeaderRequirement &header) {
  if (llvm::Error error = validateText(routeID, "header", header.header))
    return error;
  if (llvm::StringRef(header.header).contains("<") ||
      llvm::StringRef(header.header).contains(">") ||
      llvm::StringRef(header.header).contains("\""))
    return makeRouteError(
        routeID,
        llvm::Twine("header '") + header.header +
            "' must be the raw header spelling without delimiters");
  return llvm::Error::success();
}

llvm::Error validateABIParameter(
    llvm::StringRef routeID, const support::RuntimeABIParameter &parameter,
    llvm::StringRef context) {
  if (llvm::Error error =
          validateText(routeID, llvm::Twine(context) + " c_name",
                       parameter.cName))
    return error;
  if (llvm::Error error =
          validateText(routeID, llvm::Twine(context) + " c_type",
                       parameter.cType))
    return error;
  return llvm::Error::success();
}

} // namespace

TCRVEmitCLowerableRoute::TCRVEmitCLowerableRoute(llvm::StringRef routeID,
                                                 llvm::StringRef routeKind)
    : routeID(routeID.str()), routeKind(routeKind.str()) {}

void TCRVEmitCLowerableRoute::addHeader(llvm::StringRef header) {
  headers.push_back({header.str()});
}

void TCRVEmitCLowerableRoute::addTypeMapping(llvm::StringRef sourceType,
                                             llvm::StringRef cType) {
  typeMappings.push_back({sourceType.str(), cType.str()});
}

void TCRVEmitCLowerableRoute::addABIValueMapping(
    const support::RuntimeABIParameter &parameter, llvm::StringRef valueName) {
  abiMappings.push_back({parameter, valueName.str()});
}

void TCRVEmitCLowerableRoute::addFunctionDeclaration(
    llvm::StringRef name, llvm::StringRef resultCType,
    llvm::ArrayRef<llvm::StringRef> parameterCTypes) {
  TCRVEmitCFunctionDeclaration declaration;
  declaration.name = name.str();
  declaration.resultCType = resultCType.str();
  for (llvm::StringRef parameterCType : parameterCTypes)
    declaration.parameterCTypes.push_back(parameterCType.str());
  functionDeclarations.push_back(std::move(declaration));
}

void TCRVEmitCLowerableRoute::addSourceOpProvenance(
    TCRVEmitCSourceOpProvenance sourceOp) {
  sourceOpProvenance.push_back(std::move(sourceOp));
}

void TCRVEmitCLowerableRoute::addLocalVariable(
    TCRVEmitCLocalVariable variable) {
  localVariables.push_back(std::move(variable));
}

void TCRVEmitCLowerableRoute::addCallOpaqueStep(
    TCRVEmitCCallOpaqueStep step) {
  callOpaqueSteps.push_back(std::move(step));
}

void TCRVEmitCLowerableRoute::addForLoop(TCRVEmitCForLoop loop) {
  forLoops.push_back(std::move(loop));
}

void TCRVEmitCLowerableRoute::addPostLoopStep(
    TCRVEmitCCallOpaqueStep step) {
  postLoopSteps.push_back(std::move(step));
}

static llvm::Error validateCallOpaqueStep(llvm::StringRef routeID,
                                          const TCRVEmitCCallOpaqueStep &step) {
  if (llvm::Error error =
          validateText(routeID, "call_opaque callee", step.callee))
    return error;
  if (llvm::Error error =
          validateText(routeID, "source op name", step.sourceOp.opName))
    return error;
  if (llvm::Error error =
          validateText(routeID, "source op role", step.sourceOp.role))
    return error;
  if (!step.sourceOp.opInterface.empty())
    if (llvm::Error error =
            validateText(routeID, "source op interface",
                         step.sourceOp.opInterface))
      return error;
  for (const TCRVEmitCCallOpaqueOperand &operand : step.operands) {
    if (llvm::Error error =
            validateText(routeID, "call_opaque operand expression",
                         operand.expression))
      return error;
    if (llvm::Error error =
            validateText(routeID, "call_opaque operand C type",
                         operand.cType))
      return error;
  }
  if (step.result) {
    if (llvm::Error error =
            validateText(routeID, "call_opaque result name",
                         step.result->name))
      return error;
    if (llvm::Error error =
            validateText(routeID, "call_opaque result C type",
                         step.result->cType))
      return error;
  }
  return llvm::Error::success();
}

static llvm::Error validateSourceOpProvenance(
    llvm::StringRef routeID, const TCRVEmitCSourceOpProvenance &sourceOp) {
  if (llvm::Error error =
          validateText(routeID, "source op name", sourceOp.opName))
    return error;
  if (llvm::Error error =
          validateText(routeID, "source op role", sourceOp.role))
    return error;
  if (!sourceOp.opInterface.empty())
    if (llvm::Error error =
            validateText(routeID, "source op interface",
                         sourceOp.opInterface))
      return error;
  return llvm::Error::success();
}

static llvm::Error validateLocalVariable(llvm::StringRef routeID,
                                         const TCRVEmitCLocalVariable &var) {
  if (llvm::Error error = validateSourceOpProvenance(routeID, var.sourceOp))
    return error;
  if (llvm::Error error =
          validateText(routeID, "local variable name", var.name))
    return error;
  if (llvm::Error error =
          validateText(routeID, "local variable C type", var.cType))
    return error;
  if (llvm::Error error =
          validateText(routeID, "local variable initial expression",
                       var.initialValue.expression))
    return error;
  if (llvm::Error error =
          validateText(routeID, "local variable initial C type",
                       var.initialValue.cType))
    return error;
  return llvm::Error::success();
}

static llvm::Error validateAssignStep(llvm::StringRef routeID,
                                      const TCRVEmitCAssignStep &step) {
  if (llvm::Error error = validateSourceOpProvenance(routeID, step.sourceOp))
    return error;
  if (llvm::Error error =
          validateText(routeID, "assign target name", step.targetName))
    return error;
  if (llvm::Error error =
          validateText(routeID, "assign value expression",
                       step.value.expression))
    return error;
  if (llvm::Error error =
          validateText(routeID, "assign value C type", step.value.cType))
    return error;
  return llvm::Error::success();
}

llvm::Error TCRVEmitCLowerableRoute::verify() const {
  if (llvm::Error error = validateText(routeID, "route id", routeID))
    return error;
  if (llvm::Error error = validateText(routeID, "route kind", routeKind))
    return error;
  if (headers.empty())
    return makeRouteError(routeID,
                          "requires at least one header requirement");
  if (callOpaqueSteps.empty() && forLoops.empty() && postLoopSteps.empty())
    return makeRouteError(
        routeID,
        "requires at least one emitc.call_opaque construction step or "
        "structured EmitC loop");

  for (const TCRVEmitCHeaderRequirement &header : headers)
    if (llvm::Error error = validateHeader(routeID, header))
      return error;

  for (const TCRVEmitCTypeMapping &mapping : typeMappings) {
    if (llvm::Error error =
            validateText(routeID, "source type", mapping.sourceType))
      return error;
    if (llvm::Error error = validateText(routeID, "C type", mapping.cType))
      return error;
  }

  for (const TCRVEmitCABIValueMapping &mapping : abiMappings) {
    if (llvm::Error error =
            validateABIParameter(routeID, mapping.parameter, "ABI mapping"))
      return error;
    if (llvm::Error error =
              validateText(routeID, "ABI mapping value name", mapping.valueName))
        return error;
  }

  llvm::StringSet<> declaredFunctions;
  for (const TCRVEmitCFunctionDeclaration &declaration :
       functionDeclarations) {
    if (llvm::Error error =
            validateText(routeID, "function declaration name",
                         declaration.name))
      return error;
    if (!declaredFunctions.insert(declaration.name).second)
      return makeRouteError(routeID,
                            llvm::Twine("duplicate function declaration '") +
                                declaration.name + "'");
    if (!declaration.resultCType.empty() &&
        declaration.resultCType != "void")
      if (llvm::Error error =
              validateText(routeID, "function declaration result C type",
                           declaration.resultCType))
        return error;
    for (llvm::StringRef parameterCType : declaration.parameterCTypes)
      if (llvm::Error error =
              validateText(routeID, "function declaration parameter C type",
                           parameterCType))
        return error;
  }

  for (const TCRVEmitCSourceOpProvenance &sourceOp : sourceOpProvenance)
    if (llvm::Error error = validateSourceOpProvenance(routeID, sourceOp))
      return error;

  llvm::StringSet<> localVariableNames;
  for (const TCRVEmitCLocalVariable &variable : localVariables) {
    if (llvm::Error error = validateLocalVariable(routeID, variable))
      return error;
    if (!localVariableNames.insert(variable.name).second)
      return makeRouteError(routeID,
                            llvm::Twine("duplicate local variable '") +
                                variable.name + "'");
  }

  for (const TCRVEmitCCallOpaqueStep &step : callOpaqueSteps)
    if (llvm::Error error = validateCallOpaqueStep(routeID, step))
      return error;

  for (const TCRVEmitCForLoop &loop : forLoops) {
    if (llvm::Error error = validateText(routeID, "loop induction variable",
                                         loop.inductionVarName))
      return error;
    if (llvm::Error error =
            validateText(routeID, "loop lower-bound expression",
                         loop.lowerBound.expression))
      return error;
    if (llvm::Error error = validateText(routeID, "loop lower-bound C type",
                                         loop.lowerBound.cType))
      return error;
    if (llvm::Error error =
            validateText(routeID, "loop upper-bound expression",
                         loop.upperBound.expression))
      return error;
    if (llvm::Error error = validateText(routeID, "loop upper-bound C type",
                                         loop.upperBound.cType))
      return error;
    if (llvm::Error error = validateText(routeID, "loop step expression",
                                         loop.step.expression))
      return error;
    if (llvm::Error error = validateText(routeID, "loop step C type",
                                         loop.step.cType))
      return error;
    if (loop.bodySteps.empty())
      return makeRouteError(routeID,
                            "structured EmitC loop requires at least one "
                            "call_opaque body step");
    for (const TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
      if (llvm::Error error = validateCallOpaqueStep(routeID, step))
        return error;
    for (const TCRVEmitCAssignStep &step : loop.bodyAssignments)
      if (llvm::Error error = validateAssignStep(routeID, step))
        return error;
  }

  for (const TCRVEmitCCallOpaqueStep &step : postLoopSteps)
    if (llvm::Error error = validateCallOpaqueStep(routeID, step))
      return error;

  return llvm::Error::success();
}

llvm::Expected<TCRVEmitCLowerableRoute>
buildTCRVEmitCLowerableRoute(const TCRVEmitCLowerableInterface &lowerable) {
  llvm::Expected<TCRVEmitCLowerableRoute> route =
      lowerable.buildEmitCLowerableRoute();
  if (!route)
    return route.takeError();
  if (llvm::Error error = route->verify())
    return std::move(error);
  return std::move(*route);
}

} // namespace tianchenrv::conversion::emitc
