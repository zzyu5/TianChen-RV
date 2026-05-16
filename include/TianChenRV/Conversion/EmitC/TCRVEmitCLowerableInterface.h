#ifndef TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEINTERFACE_H
#define TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEINTERFACE_H

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace tianchenrv::conversion::emitc {

struct TCRVEmitCHeaderRequirement {
  std::string header;
};

struct TCRVEmitCTypeMapping {
  std::string sourceType;
  std::string cType;
};

struct TCRVEmitCABIValueMapping {
  support::RuntimeABIParameter parameter;
  std::string valueName;
};

struct TCRVEmitCSourceOpProvenance {
  std::string opName;
  std::string role;
  std::string opInterface;
};

struct TCRVEmitCCallOpaqueOperand {
  std::string expression;
  std::string cType;
};

struct TCRVEmitCCallOpaqueResult {
  std::string name;
  std::string cType;
};

struct TCRVEmitCCallOpaqueStep {
  TCRVEmitCSourceOpProvenance sourceOp;
  std::string callee;
  llvm::SmallVector<TCRVEmitCCallOpaqueOperand, 4> operands;
  std::optional<TCRVEmitCCallOpaqueResult> result;
};

class TCRVEmitCLowerableRoute {
public:
  TCRVEmitCLowerableRoute() = default;
  TCRVEmitCLowerableRoute(llvm::StringRef routeID,
                          llvm::StringRef routeKind);

  llvm::StringRef getRouteID() const { return routeID; }
  llvm::StringRef getRouteKind() const { return routeKind; }
  llvm::ArrayRef<TCRVEmitCHeaderRequirement> getHeaders() const {
    return headers;
  }
  llvm::ArrayRef<TCRVEmitCTypeMapping> getTypeMappings() const {
    return typeMappings;
  }
  llvm::ArrayRef<TCRVEmitCABIValueMapping> getABIMappings() const {
    return abiMappings;
  }
  llvm::ArrayRef<TCRVEmitCSourceOpProvenance> getSourceOpProvenance() const {
    return sourceOpProvenance;
  }
  llvm::ArrayRef<TCRVEmitCCallOpaqueStep> getCallOpaqueSteps() const {
    return callOpaqueSteps;
  }

  void addHeader(llvm::StringRef header);
  void addTypeMapping(llvm::StringRef sourceType, llvm::StringRef cType);
  void addABIValueMapping(const support::RuntimeABIParameter &parameter,
                          llvm::StringRef valueName);
  void addSourceOpProvenance(TCRVEmitCSourceOpProvenance sourceOp);
  void addCallOpaqueStep(TCRVEmitCCallOpaqueStep step);

  llvm::Error verify() const;

private:
  std::string routeID;
  std::string routeKind;
  llvm::SmallVector<TCRVEmitCHeaderRequirement, 4> headers;
  llvm::SmallVector<TCRVEmitCTypeMapping, 4> typeMappings;
  llvm::SmallVector<TCRVEmitCABIValueMapping, 6> abiMappings;
  llvm::SmallVector<TCRVEmitCSourceOpProvenance, 8> sourceOpProvenance;
  llvm::SmallVector<TCRVEmitCCallOpaqueStep, 8> callOpaqueSteps;
};

class TCRVEmitCLowerableInterface {
public:
  virtual ~TCRVEmitCLowerableInterface() = default;

  virtual llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const = 0;
};

llvm::Expected<TCRVEmitCLowerableRoute>
buildTCRVEmitCLowerableRoute(const TCRVEmitCLowerableInterface &lowerable);

} // namespace tianchenrv::conversion::emitc

#endif // TIANCHENRV_CONVERSION_EMITC_TCRVEMITCLOWERABLEINTERFACE_H
