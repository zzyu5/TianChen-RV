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

struct TCRVEmitCFunctionDeclaration {
  std::string name;
  std::string resultCType;
  llvm::SmallVector<std::string, 4> parameterCTypes;
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

struct TCRVEmitCLocalVariable {
  TCRVEmitCSourceOpProvenance sourceOp;
  std::string name;
  std::string cType;
  TCRVEmitCCallOpaqueOperand initialValue;
};

struct TCRVEmitCAssignStep {
  TCRVEmitCSourceOpProvenance sourceOp;
  std::string targetName;
  TCRVEmitCCallOpaqueOperand value;
};

struct TCRVEmitCForLoop {
  std::string inductionVarName;
  TCRVEmitCCallOpaqueOperand lowerBound;
  TCRVEmitCCallOpaqueOperand upperBound;
  TCRVEmitCCallOpaqueOperand step;
  llvm::SmallVector<TCRVEmitCCallOpaqueStep, 8> bodySteps;
  llvm::SmallVector<TCRVEmitCAssignStep, 2> bodyAssignments;
};

class TCRVEmitCLowerableRoute {
public:
  TCRVEmitCLowerableRoute();
  TCRVEmitCLowerableRoute(llvm::StringRef routeID,
                          llvm::StringRef routeKind);
  TCRVEmitCLowerableRoute(const TCRVEmitCLowerableRoute &other);
  TCRVEmitCLowerableRoute &operator=(const TCRVEmitCLowerableRoute &other);
  TCRVEmitCLowerableRoute(TCRVEmitCLowerableRoute &&other);
  TCRVEmitCLowerableRoute &operator=(TCRVEmitCLowerableRoute &&other);

  void reset(llvm::StringRef routeID, llvm::StringRef routeKind);

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
  llvm::ArrayRef<TCRVEmitCFunctionDeclaration> getFunctionDeclarations() const {
    return functionDeclarations;
  }
  llvm::ArrayRef<TCRVEmitCSourceOpProvenance> getSourceOpProvenance() const {
    return sourceOpProvenance;
  }
  llvm::ArrayRef<TCRVEmitCLocalVariable> getLocalVariables() const {
    return localVariables;
  }
  llvm::ArrayRef<TCRVEmitCCallOpaqueStep> getCallOpaqueSteps() const {
    return callOpaqueSteps;
  }
  llvm::ArrayRef<TCRVEmitCForLoop> getForLoops() const { return forLoops; }
  llvm::ArrayRef<TCRVEmitCCallOpaqueStep> getPostLoopSteps() const {
    return postLoopSteps;
  }

  void addHeader(llvm::StringRef header);
  void addTypeMapping(llvm::StringRef sourceType, llvm::StringRef cType);
  void addABIValueMapping(const support::RuntimeABIParameter &parameter,
                          llvm::StringRef valueName);
  void addFunctionDeclaration(
      llvm::StringRef name, llvm::StringRef resultCType = {},
      llvm::ArrayRef<llvm::StringRef> parameterCTypes = {});
  void addSourceOpProvenance(TCRVEmitCSourceOpProvenance sourceOp);
  void addLocalVariable(TCRVEmitCLocalVariable variable);
  void addCallOpaqueStep(TCRVEmitCCallOpaqueStep step);
  void addForLoop(TCRVEmitCForLoop loop);
  void addPostLoopStep(TCRVEmitCCallOpaqueStep step);

  llvm::Error verify() const;

private:
  std::string routeID;
  std::string routeKind;
  llvm::SmallVector<TCRVEmitCHeaderRequirement, 4> headers;
  llvm::SmallVector<TCRVEmitCTypeMapping, 4> typeMappings;
  llvm::SmallVector<TCRVEmitCABIValueMapping, 6> abiMappings;
  llvm::SmallVector<TCRVEmitCFunctionDeclaration, 8> functionDeclarations;
  llvm::SmallVector<TCRVEmitCSourceOpProvenance, 8> sourceOpProvenance;
  llvm::SmallVector<TCRVEmitCLocalVariable, 2> localVariables;
  llvm::SmallVector<TCRVEmitCCallOpaqueStep, 8> callOpaqueSteps;
  llvm::SmallVector<TCRVEmitCForLoop, 2> forLoops;
  llvm::SmallVector<TCRVEmitCCallOpaqueStep, 4> postLoopSteps;
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
