#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"

#include "llvm/ADT/STLExtras.h"
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

void TCRVEmitCLowerableRoute::addCallOpaqueStep(
    TCRVEmitCCallOpaqueStep step) {
  callOpaqueSteps.push_back(std::move(step));
}

llvm::Error TCRVEmitCLowerableRoute::verify() const {
  if (llvm::Error error = validateText(routeID, "route id", routeID))
    return error;
  if (llvm::Error error = validateText(routeID, "route kind", routeKind))
    return error;
  if (headers.empty())
    return makeRouteError(routeID,
                          "requires at least one header requirement");
  if (callOpaqueSteps.empty())
    return makeRouteError(
        routeID, "requires at least one emitc.call_opaque construction step");

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

  for (const TCRVEmitCCallOpaqueStep &step : callOpaqueSteps) {
    if (llvm::Error error =
            validateText(routeID, "call_opaque callee", step.callee))
      return error;
    if (llvm::Error error =
            validateText(routeID, "source op name", step.sourceOp.opName))
      return error;
    if (llvm::Error error =
            validateText(routeID, "source op role", step.sourceOp.role))
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
  }

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
