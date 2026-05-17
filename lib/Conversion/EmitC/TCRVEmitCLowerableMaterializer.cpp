#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

namespace tianchenrv::conversion::emitc {
namespace {

llvm::Error makeMaterializerError(llvm::StringRef routeID,
                                  llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "TianChen-RV EmitC materializer";
  if (!routeID.empty())
    os << " for route '" << routeID << "'";
  os << " failed: ";
  message.print(os);
  os.flush();
  return llvm::createStringError(llvm::errc::invalid_argument, text);
}

bool isSafeIdentifier(llvm::StringRef value) {
  if (value.empty() || value.size() > 128)
    return false;
  auto isHead = [](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return std::isalpha(byte) || c == '_';
  };
  auto isTail = [&](char c) {
    unsigned char byte = static_cast<unsigned char>(c);
    return isHead(c) || std::isdigit(byte);
  };
  if (!isHead(value.front()))
    return false;
  return llvm::all_of(value.drop_front(), isTail);
}

bool isSafeProvenanceText(llvm::StringRef value) {
  if (value.empty() || value.size() > 128)
    return false;
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_' || c == '.' || c == '-')
      continue;
    return false;
  }
  return true;
}

bool isSafeExpressionText(llvm::StringRef value) {
  if (value.empty() || value.size() > 256)
    return false;
  for (char c : value) {
    unsigned char byte = static_cast<unsigned char>(c);
    if (std::isalnum(byte) || c == '_' || c == ' ' || c == '&' || c == '[' ||
        c == ']' || c == '(' || c == ')' || c == '+' || c == '-' ||
        c == '*' || c == '/' || c == '%' || c == '.')
      continue;
    return false;
  }
  return true;
}

llvm::Error validateSafeIdentifier(llvm::StringRef routeID,
                                   llvm::StringRef field,
                                   llvm::StringRef value) {
  if (isSafeIdentifier(value))
    return llvm::Error::success();
  return makeMaterializerError(
      routeID, llvm::Twine(field) +
                   " must be a bounded C/EmitC identifier");
}

llvm::Error validateSafeProvenance(const TCRVEmitCLowerableRoute &route,
                                   const TCRVEmitCSourceOpProvenance &source) {
  if (!isSafeProvenanceText(source.opName))
    return makeMaterializerError(
        route.getRouteID(),
        "source op provenance contains unsafe text before EmitC "
        "materialization");
  if (!isSafeProvenanceText(source.role))
    return makeMaterializerError(
        route.getRouteID(),
        "source role provenance contains unsafe text before EmitC "
        "materialization");
  if (!source.opInterface.empty() &&
      !isSafeProvenanceText(source.opInterface))
    return makeMaterializerError(
        route.getRouteID(),
        "source op-interface provenance contains unsafe text before EmitC "
        "materialization");
  return llvm::Error::success();
}

mlir::Type getEmitCTypeForCType(mlir::MLIRContext &context,
                                llvm::StringRef cType) {
  cType = cType.trim();
  if (cType.ends_with("*")) {
    llvm::StringRef pointee = cType.drop_back().rtrim();
    return mlir::emitc::PointerType::get(&context,
                                         getEmitCTypeForCType(context, pointee));
  }
  return mlir::emitc::OpaqueType::get(&context, cType);
}

std::string makeStepProvenanceComment(const TCRVEmitCCallOpaqueStep &step) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.source_op=" << step.sourceOp.opName
     << " role=" << step.sourceOp.role;
  if (!step.sourceOp.opInterface.empty())
    os << " op_interface=" << step.sourceOp.opInterface;
  os << " callee=" << step.callee;
  os.flush();
  return text;
}

std::string
makeRouteSourceProvenanceComment(const TCRVEmitCSourceOpProvenance &source) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc.route_source_op=" << source.opName
     << " role=" << source.role;
  if (!source.opInterface.empty())
    os << " op_interface=" << source.opInterface;
  os.flush();
  return text;
}

mlir::Location makeStepLocation(mlir::OpBuilder &builder,
                                const TCRVEmitCCallOpaqueStep &step) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "tcrv_emitc." << step.sourceOp.opName << "." << step.sourceOp.role;
  if (!step.sourceOp.opInterface.empty())
    os << "." << step.sourceOp.opInterface;
  os.flush();
  return mlir::NameLoc::get(builder.getStringAttr(name),
                            builder.getUnknownLoc());
}

mlir::Location makeRouteSourceLocation(
    mlir::OpBuilder &builder, const TCRVEmitCSourceOpProvenance &source) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "tcrv_emitc.route_source." << source.opName << "." << source.role;
  if (!source.opInterface.empty())
    os << "." << source.opInterface;
  os.flush();
  return mlir::NameLoc::get(builder.getStringAttr(name),
                            builder.getUnknownLoc());
}

void collectExpressionIdentifiers(llvm::StringRef expression,
                                  llvm::SmallVectorImpl<llvm::StringRef> &out) {
  for (std::size_t index = 0; index < expression.size();) {
    char c = expression[index];
    unsigned char byte = static_cast<unsigned char>(c);
    if (!(std::isalpha(byte) || c == '_')) {
      ++index;
      continue;
    }
    std::size_t start = index++;
    while (index < expression.size()) {
      char tail = expression[index];
      unsigned char tailByte = static_cast<unsigned char>(tail);
      if (!(std::isalnum(tailByte) || tail == '_'))
        break;
      ++index;
    }
    out.push_back(expression.slice(start, index));
  }
}

std::optional<std::tuple<llvm::StringRef, char, llvm::StringRef>>
parseSimpleBinaryExpression(llvm::StringRef expression) {
  for (llvm::StringRef separator : {" + ", " - "}) {
    std::pair<llvm::StringRef, llvm::StringRef> parts =
        expression.split(separator);
    if (parts.second.empty())
      continue;
    if (parts.first.empty() || parts.second.empty())
      return std::nullopt;
    char op = separator[1];
    return std::tuple<llvm::StringRef, char, llvm::StringRef>(
        parts.first.trim(), op, parts.second.trim());
  }
  return std::nullopt;
}

class RouteMaterializer {
public:
  RouteMaterializer(mlir::MLIRContext &context,
                    const TCRVEmitCLowerableRoute &route,
                    const TCRVEmitCMaterializationOptions &options)
      : context(context), route(route), options(options), builder(&context) {}

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> run() {
    if (llvm::Error error = route.verify())
      return std::move(error);
    if (llvm::Error error = validateOptions())
      return std::move(error);

    context.getOrLoadDialect<mlir::emitc::EmitCDialect>();

    mlir::Location loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    builder.setInsertionPointToStart(module->getBody());

    for (const TCRVEmitCHeaderRequirement &header : route.getHeaders())
      builder.create<mlir::emitc::IncludeOp>(loc, header.header,
                                             /*is_standard_include=*/true);
    if (llvm::Error error = materializeFunctionDeclarations())
      return std::move(error);

    llvm::SmallVector<mlir::Type, 6> inputTypes;
    if (llvm::Error error = buildFunctionInputTypes(inputTypes))
      return std::move(error);

    mlir::FunctionType functionType =
        builder.getFunctionType(inputTypes, llvm::ArrayRef<mlir::Type>{});
    llvm::SmallVector<mlir::NamedAttribute, 1> functionAttrs;
    if (options.emitExternC) {
      functionAttrs.push_back(builder.getNamedAttr(
          "specifiers", builder.getStrArrayAttr({"extern", "\"C\""})));
    }
    mlir::emitc::FuncOp function = builder.create<mlir::emitc::FuncOp>(
        loc, options.functionName, functionType, functionAttrs);
    mlir::Block *entry = new mlir::Block();
    function.getBody().push_back(entry);
    for (mlir::Type type : inputTypes)
      entry->addArgument(type, loc);

    builder.setInsertionPointToStart(entry);
    initializeValueMap(entry);
    for (const TCRVEmitCSourceOpProvenance &sourceOp :
         route.getSourceOpProvenance())
      if (llvm::Error error = materializeSourceProvenance(sourceOp))
        return std::move(error);
    for (const TCRVEmitCCallOpaqueStep &step : route.getCallOpaqueSteps())
      if (llvm::Error error = materializeStep(step))
        return std::move(error);
    for (const TCRVEmitCForLoop &loop : route.getForLoops())
      if (llvm::Error error = materializeForLoop(loop))
        return std::move(error);
    builder.create<mlir::emitc::ReturnOp>(loc, mlir::Value());

    if (options.verifyModule && mlir::failed(mlir::verify(*module)))
      return makeMaterializerError(route.getRouteID(),
                                   "materialized EmitC module failed MLIR "
                                   "verification");
    return module;
  }

private:
  llvm::Error validateOptions() const {
    if (llvm::Error error = validateSafeIdentifier(
            route.getRouteID(), "EmitC materialized function name",
            options.functionName))
      return error;
    for (llvm::StringRef implicit : options.implicitValueNames)
      if (llvm::Error error = validateSafeIdentifier(
              route.getRouteID(), "implicit EmitC value name", implicit))
        return error;
    return llvm::Error::success();
  }

  llvm::Error
  buildFunctionInputTypes(llvm::SmallVectorImpl<mlir::Type> &inputTypes) {
    llvm::StringSet<> seenValueNames;
    for (const TCRVEmitCABIValueMapping &mapping : route.getABIMappings()) {
      llvm::StringRef cName = mapping.parameter.cName;
      llvm::StringRef valueName = mapping.valueName;
      if (llvm::Error error =
              validateSafeIdentifier(route.getRouteID(), "ABI C name", cName))
        return error;
      if (llvm::Error error = validateSafeIdentifier(
              route.getRouteID(), "ABI value mapping name", valueName))
        return error;
      if (cName != valueName)
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("ABI value mapping for C parameter '") + cName +
                "' must use the same function-boundary value name, got '" +
                valueName + "'");
      if (!seenValueNames.insert(valueName).second)
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("duplicate ABI value mapping name '") + valueName +
                "'");
      inputTypes.push_back(
          getEmitCTypeForCType(context, mapping.parameter.cType));
    }
    return llvm::Error::success();
  }

  void initializeValueMap(mlir::Block *entry) {
    for (auto [index, mapping] : llvm::enumerate(route.getABIMappings()))
      valueMap[mapping.valueName] = entry->getArgument(index);
    for (llvm::StringRef implicit : options.implicitValueNames)
      implicitValues.insert(implicit);
  }

  llvm::Error materializeFunctionDeclarations() {
    mlir::Location loc = builder.getUnknownLoc();
    for (const TCRVEmitCFunctionDeclaration &declaration :
         route.getFunctionDeclarations()) {
      if (llvm::Error error = validateSafeIdentifier(
              route.getRouteID(), "declared EmitC function name",
              declaration.name))
        return error;

      llvm::SmallVector<mlir::Type, 4> inputTypes;
      for (llvm::StringRef parameterCType : declaration.parameterCTypes)
        inputTypes.push_back(getEmitCTypeForCType(context, parameterCType));

      llvm::SmallVector<mlir::Type, 1> resultTypes;
      if (!declaration.resultCType.empty() &&
          declaration.resultCType != "void")
        resultTypes.push_back(getEmitCTypeForCType(context,
                                                   declaration.resultCType));

      mlir::FunctionType functionType =
          builder.getFunctionType(inputTypes, resultTypes);
      llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
      attrs.push_back(builder.getNamedAttr(
          mlir::SymbolTable::getVisibilityAttrName(),
          builder.getStringAttr("private")));
      builder.create<mlir::emitc::FuncOp>(loc, declaration.name, functionType,
                                          attrs);
    }
    return llvm::Error::success();
  }

  llvm::Expected<mlir::Value>
  materializeOperandExpression(const TCRVEmitCCallOpaqueOperand &operand) {
    llvm::StringRef expression = operand.expression;
    if (!isSafeExpressionText(expression))
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("operand expression '") + expression +
              "' contains unsafe text before EmitC materialization");

    if (isSafeIdentifier(expression)) {
      if (mlir::Value value = valueMap.lookup(expression))
        return value;
      if (implicitValues.contains(expression))
        return builder
            .create<mlir::emitc::LiteralOp>(
                builder.getUnknownLoc(),
                getEmitCTypeForCType(context, operand.cType), expression)
            .getResult();
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("operand expression references unknown value name '") +
              expression + "'");
    }

    if (std::optional<std::tuple<llvm::StringRef, char, llvm::StringRef>>
            binary = parseSimpleBinaryExpression(expression)) {
      llvm::StringRef lhsName = std::get<0>(*binary);
      char op = std::get<1>(*binary);
      llvm::StringRef rhsName = std::get<2>(*binary);
      mlir::Value lhs = valueMap.lookup(lhsName);
      mlir::Value rhs = valueMap.lookup(rhsName);
      if (!lhs || !rhs)
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("operand expression '") + expression +
                "' references values that are not materialized in the "
                "current EmitC scope");
      mlir::Type resultType = getEmitCTypeForCType(context, operand.cType);
      if (op == '+')
        return builder
            .create<mlir::emitc::AddOp>(builder.getUnknownLoc(), resultType,
                                        lhs, rhs)
            .getResult();
      if (op == '-')
        return builder
            .create<mlir::emitc::SubOp>(builder.getUnknownLoc(), resultType,
                                        lhs, rhs)
            .getResult();
    }

    llvm::SmallVector<llvm::StringRef, 4> identifiers;
    collectExpressionIdentifiers(expression, identifiers);
    for (llvm::StringRef identifier : identifiers) {
      if (valueMap.contains(identifier) || implicitValues.contains(identifier))
        continue;
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("operand expression '") + expression +
              "' references unknown value name '" + identifier + "'");
    }

    return builder
        .create<mlir::emitc::LiteralOp>(
            builder.getUnknownLoc(),
            getEmitCTypeForCType(context, operand.cType), expression)
        .getResult();
  }

  llvm::Error materializeForLoop(const TCRVEmitCForLoop &loop) {
    if (llvm::Error error = validateSafeIdentifier(
            route.getRouteID(), "loop induction variable",
            loop.inductionVarName))
      return error;

    llvm::Expected<mlir::Value> lower =
        materializeOperandExpression(loop.lowerBound);
    if (!lower)
      return lower.takeError();
    llvm::Expected<mlir::Value> upper =
        materializeOperandExpression(loop.upperBound);
    if (!upper)
      return upper.takeError();
    llvm::Expected<mlir::Value> step =
        materializeOperandExpression(loop.step);
    if (!step)
      return step.takeError();

    mlir::emitc::ForOp forOp = builder.create<mlir::emitc::ForOp>(
        builder.getUnknownLoc(), *lower, *upper, *step,
        /*bodyBuilder=*/nullptr);

    llvm::StringMap<mlir::Value> savedValueMap = valueMap;
    valueMap[loop.inductionVarName] = forOp.getInductionVar();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());
    for (const TCRVEmitCCallOpaqueStep &step : loop.bodySteps)
      if (llvm::Error error = materializeStep(step)) {
        valueMap = std::move(savedValueMap);
        return error;
      }
    valueMap = std::move(savedValueMap);
    return llvm::Error::success();
  }

  llvm::Error materializeStep(const TCRVEmitCCallOpaqueStep &step) {
    if (llvm::Error error = validateSafeProvenance(route, step.sourceOp))
      return error;
    if (step.sourceOp.role == "compute" && !step.result)
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("compute step from source op '") + step.sourceOp.opName +
              "' requires a result value before EmitC materialization");

    builder.create<mlir::emitc::VerbatimOp>(makeStepLocation(builder, step),
                                            makeStepProvenanceComment(step));

    llvm::SmallVector<mlir::Value, 4> operands;
    for (const TCRVEmitCCallOpaqueOperand &operand : step.operands) {
      llvm::Expected<mlir::Value> value = materializeOperandExpression(operand);
      if (!value)
        return value.takeError();
      operands.push_back(*value);
    }

    llvm::SmallVector<mlir::Type, 1> resultTypes;
    if (step.result) {
      if (llvm::Error error = validateSafeIdentifier(
              route.getRouteID(), "call_opaque result name",
              step.result->name))
        return error;
      if (valueMap.contains(step.result->name) ||
          implicitValues.contains(step.result->name))
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("duplicate call_opaque result name '") +
                step.result->name + "'");
      resultTypes.push_back(getEmitCTypeForCType(context, step.result->cType));
    }

    mlir::emitc::CallOpaqueOp call =
        builder.create<mlir::emitc::CallOpaqueOp>(
            makeStepLocation(builder, step), resultTypes, step.callee,
            operands);
    if (step.result)
      valueMap[step.result->name] = call->getResult(0);
    return llvm::Error::success();
  }

  llvm::Error
  materializeSourceProvenance(const TCRVEmitCSourceOpProvenance &sourceOp) {
    if (llvm::Error error = validateSafeProvenance(route, sourceOp))
      return error;
    builder.create<mlir::emitc::VerbatimOp>(
        makeRouteSourceLocation(builder, sourceOp),
        makeRouteSourceProvenanceComment(sourceOp));
    return llvm::Error::success();
  }

  mlir::MLIRContext &context;
  const TCRVEmitCLowerableRoute &route;
  const TCRVEmitCMaterializationOptions &options;
  mlir::OpBuilder builder;
  llvm::StringMap<mlir::Value> valueMap;
  llvm::StringSet<> implicitValues;
};

} // namespace

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeTCRVEmitCLowerableRoute(
    mlir::MLIRContext &context, const TCRVEmitCLowerableRoute &route,
    const TCRVEmitCMaterializationOptions &options) {
  return RouteMaterializer(context, route, options).run();
}

llvm::Error verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
    const TCRVEmitCLowerableRoute &route, llvm::StringRef functionName,
    llvm::ArrayRef<llvm::StringRef> implicitValueNames) {
  mlir::MLIRContext context;
  TCRVEmitCMaterializationOptions options;
  options.functionName = functionName.str();
  for (llvm::StringRef valueName : implicitValueNames)
    options.implicitValueNames.push_back(valueName.str());

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRoute(context, route, options);
  if (!module)
    return module.takeError();
  return llvm::Error::success();
}

} // namespace tianchenrv::conversion::emitc
