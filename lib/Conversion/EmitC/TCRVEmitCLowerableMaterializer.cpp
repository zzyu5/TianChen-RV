#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"

#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>

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
  if (cType == "size_t")
    return mlir::emitc::SizeTType::get(&context);
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

    llvm::SmallVector<mlir::Type, 6> inputTypes;
    if (llvm::Error error = buildFunctionInputTypes(inputTypes))
      return std::move(error);

    mlir::FunctionType functionType =
        builder.getFunctionType(inputTypes, llvm::ArrayRef<mlir::Type>{});
    mlir::emitc::FuncOp function = builder.create<mlir::emitc::FuncOp>(
        loc, options.functionName, functionType);
    mlir::Block *entry = new mlir::Block();
    function.getBody().push_back(entry);
    for (mlir::Type type : inputTypes)
      entry->addArgument(type, loc);

    builder.setInsertionPointToStart(entry);
    initializeValueMap(entry);
    for (const TCRVEmitCCallOpaqueStep &step : route.getCallOpaqueSteps())
      if (llvm::Error error = materializeStep(step))
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
      inputTypes.push_back(getEmitCTypeForCType(context, mapping.parameter.cType));
    }
    return llvm::Error::success();
  }

  void initializeValueMap(mlir::Block *entry) {
    for (auto [index, mapping] : llvm::enumerate(route.getABIMappings()))
      valueMap[mapping.valueName] = entry->getArgument(index);
    for (llvm::StringRef implicit : options.implicitValueNames)
      implicitValues.insert(implicit);
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
                builder.getUnknownLoc(), getEmitCTypeForCType(context, operand.cType),
                expression)
            .getResult();
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("operand expression references unknown value name '") +
              expression + "'");
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
            builder.getUnknownLoc(), getEmitCTypeForCType(context, operand.cType),
            expression)
        .getResult();
  }

  llvm::Error materializeStep(const TCRVEmitCCallOpaqueStep &step) {
    if (llvm::Error error = validateSafeProvenance(route, step.sourceOp))
      return error;
    if (step.sourceOp.role == "compute" && !step.result)
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("compute step from source op '") +
              step.sourceOp.opName + "' requires a result value before "
              "EmitC materialization");

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

  mlir::MLIRContext &context;
  const TCRVEmitCLowerableRoute &route;
  const TCRVEmitCMaterializationOptions &options;
  mlir::OpBuilder builder;
  llvm::StringMap<mlir::Value> valueMap;
  llvm::StringSet<> implicitValues;
};

llvm::Error makeSourceRendererError(llvm::StringRef routeID,
                                    llvm::Twine message) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "TianChen-RV EmitC C source renderer";
  if (!routeID.empty())
    os << " for route '" << routeID << "'";
  os << " failed: ";
  message.print(os);
  os.flush();
  return llvm::createStringError(llvm::errc::invalid_argument, text);
}

class RouteCSourceRenderer {
public:
  RouteCSourceRenderer(const TCRVEmitCLowerableRoute &route,
                       llvm::raw_ostream &os,
                       const TCRVEmitCSourceRenderOptions &options)
      : route(route), os(os), options(options) {}

  llvm::Error run() {
    if (llvm::Error error = route.verify())
      return error;
    if (llvm::Error error = validateOptions())
      return error;
    if (llvm::Error error = initializeABIValues())
      return error;

    llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps =
        route.getCallOpaqueSteps();
    bool usesRuntimeVLLoop = isRuntimeAVLToVLStep(steps.front());
    if (usesRuntimeVLLoop) {
      if (steps.size() < 2)
        return makeSourceRendererError(
            route.getRouteID(),
            "requires a runtime-avl-to-vl step followed by at least one "
            "body call_opaque step");
      if (llvm::Error error = validateRuntimeVLLoopHeader(steps.front()))
        return error;
    } else if (llvm::Error error = validateRuntimeElementCountLoopShape(steps)) {
      return error;
    }

    std::string rendered;
    llvm::raw_string_ostream renderedOS(rendered);
    printFunctionHeader(renderedOS);
    if (usesRuntimeVLLoop) {
      renderedOS << "  size_t " << options.loopIndexName << " = 0;\n";
      renderedOS << "  while (" << options.loopIndexName << " < "
                 << runtimeElementCountValueName << ") {\n";
    } else {
      renderedOS << "  for (size_t " << options.loopIndexName << " = 0; "
                 << options.loopIndexName << " < "
                 << runtimeElementCountValueName << "; ++"
                 << options.loopIndexName << ") {\n";
    }

    bool sawInterfaceBackedCompute = false;
    for (const TCRVEmitCCallOpaqueStep &step : steps) {
      if (llvm::Error error =
              renderStep(step, sawInterfaceBackedCompute, renderedOS))
        return error;
    }

    if (options.requireInterfaceBackedCompute && !sawInterfaceBackedCompute)
      return makeSourceRendererError(
          route.getRouteID(),
          "requires at least one interface-backed compute call_opaque step "
          "before C source rendering");

    if (usesRuntimeVLLoop)
      renderedOS << "    " << options.loopIndexName << " += "
                 << steps.front().result->name << ";\n";
    renderedOS << "  }\n";
    renderedOS << "}\n\n";
    renderedOS.flush();
    os << rendered;
    return llvm::Error::success();
  }

private:
  llvm::Error validateOptions() {
    if (llvm::Error error =
            validateSafeIdentifier(route.getRouteID(), "C function name",
                                   options.functionName))
      return error;
    if (llvm::Error error =
            validateSafeIdentifier(route.getRouteID(), "C loop index name",
                                   options.loopIndexName))
      return error;
    knownValues.insert(options.loopIndexName);
    return llvm::Error::success();
  }

  llvm::Error initializeABIValues() {
    unsigned runtimeElementCountMatches = 0;
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
        return makeSourceRendererError(
            route.getRouteID(),
            llvm::Twine("ABI value mapping for C parameter '") + cName +
                "' must use the same function-boundary value name, got '" +
                valueName + "'");
      if (!knownValues.insert(valueName).second)
        return makeSourceRendererError(
            route.getRouteID(),
            llvm::Twine("duplicate C source value name '") + valueName + "'");
      if (mapping.parameter.role ==
          support::RuntimeABIParameterRole::RuntimeElementCount) {
        runtimeElementCountValueName = valueName.str();
        ++runtimeElementCountMatches;
      }
    }

    if (runtimeElementCountMatches != 1)
      return makeSourceRendererError(
          route.getRouteID(),
          "requires exactly one runtime-element-count ABI mapping before "
          "bounded-loop C source rendering");
    return llvm::Error::success();
  }

  static bool isRuntimeAVLToVLStep(const TCRVEmitCCallOpaqueStep &step) {
    return step.sourceOp.role == "runtime-avl-to-vl";
  }

  llvm::Error
  validateRuntimeVLLoopHeader(const TCRVEmitCCallOpaqueStep &firstStep) const {
    if (!firstStep.result)
      return makeSourceRendererError(
          route.getRouteID(),
          "first call_opaque step must map runtime AVL to a VL result before "
          "bounded-loop C source rendering");
    if (firstStep.result->cType != "size_t")
      return makeSourceRendererError(
          route.getRouteID(),
          "runtime-avl-to-vl result must use C type 'size_t' before "
          "bounded-loop C source rendering");
    return llvm::Error::success();
  }

  llvm::Error validateRuntimeElementCountLoopShape(
      llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps) const {
    for (const TCRVEmitCCallOpaqueStep &step : steps) {
      if (isRuntimeAVLToVLStep(step))
        return makeSourceRendererError(
            route.getRouteID(),
            "runtime-element-count C source rendering cannot mix a "
            "runtime-avl-to-vl call_opaque step into the scalar loop shape");
      if (step.sourceOp.role != "compute" && step.result)
        return makeSourceRendererError(
            route.getRouteID(),
            llvm::Twine("runtime-element-count C source rendering cannot "
                        "materialize non-compute call_opaque result '") +
                step.result->name + "'");
    }
    return llvm::Error::success();
  }

  llvm::Error validateOperandExpression(
      const TCRVEmitCCallOpaqueOperand &operand) const {
    llvm::StringRef expression = operand.expression;
    if (!isSafeExpressionText(expression))
      return makeSourceRendererError(
          route.getRouteID(),
          llvm::Twine("operand expression '") + expression +
              "' contains unsafe text before C source rendering");

    llvm::SmallVector<llvm::StringRef, 4> identifiers;
    collectExpressionIdentifiers(expression, identifiers);
    for (llvm::StringRef identifier : identifiers) {
      if (knownValues.contains(identifier))
        continue;
      return makeSourceRendererError(
          route.getRouteID(),
          llvm::Twine("operand expression '") + expression +
              "' references unknown value name '" + identifier + "'");
    }
    return llvm::Error::success();
  }

  llvm::Error renderStep(const TCRVEmitCCallOpaqueStep &step,
                         bool &sawInterfaceBackedCompute,
                         llvm::raw_ostream &targetOS) {
    if (llvm::Error error = validateSafeProvenance(route, step.sourceOp))
      return error;
    if (llvm::Error error =
            validateSafeIdentifier(route.getRouteID(), "call_opaque callee",
                                   step.callee))
      return error;
    if (step.sourceOp.role == "compute") {
      if (!step.result)
        return makeSourceRendererError(
            route.getRouteID(),
            llvm::Twine("compute step from source op '") +
                step.sourceOp.opName + "' requires a result value before C "
                "source rendering");
      if (options.requireInterfaceBackedCompute &&
          step.sourceOp.opInterface.empty())
        return makeSourceRendererError(
            route.getRouteID(),
            llvm::Twine("compute step from source op '") +
                step.sourceOp.opName +
                "' requires generated op-interface provenance before C "
                "source rendering");
      sawInterfaceBackedCompute = true;
    }

    for (const TCRVEmitCCallOpaqueOperand &operand : step.operands)
      if (llvm::Error error = validateOperandExpression(operand))
        return error;

    if (step.result) {
      if (llvm::Error error = validateSafeIdentifier(
              route.getRouteID(), "call_opaque result name",
              step.result->name))
        return error;
      if (knownValues.contains(step.result->name))
        return makeSourceRendererError(
            route.getRouteID(),
            llvm::Twine("duplicate C source value name '") +
                step.result->name + "'");

      targetOS << "    " << step.result->cType << " " << step.result->name
               << " = " << step.callee << "(";
      printOperandList(step, targetOS);
      targetOS << ");\n";
      knownValues.insert(step.result->name);
      return llvm::Error::success();
    }

    targetOS << "    " << step.callee << "(";
    printOperandList(step, targetOS);
    targetOS << ");\n";
    return llvm::Error::success();
  }

  void printFunctionHeader(llvm::raw_ostream &targetOS) {
    targetOS << "void " << options.functionName << "(";
    for (auto [index, mapping] : llvm::enumerate(route.getABIMappings())) {
      if (index != 0)
        targetOS << ", ";
      support::printRuntimeABIParameterCDeclaration(targetOS,
                                                    mapping.parameter);
    }
    targetOS << ") {\n";
  }

  void printOperandList(const TCRVEmitCCallOpaqueStep &step,
                        llvm::raw_ostream &targetOS) {
    for (auto [index, operand] : llvm::enumerate(step.operands)) {
      if (index != 0)
        targetOS << ", ";
      targetOS << operand.expression;
    }
  }

  const TCRVEmitCLowerableRoute &route;
  llvm::raw_ostream &os;
  const TCRVEmitCSourceRenderOptions &options;
  llvm::StringSet<> knownValues;
  std::string runtimeElementCountValueName;
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

llvm::Error renderTCRVEmitCLowerableRouteAsCFunction(
    const TCRVEmitCLowerableRoute &route, llvm::raw_ostream &os,
    const TCRVEmitCSourceRenderOptions &options) {
  return RouteCSourceRenderer(route, os, options).run();
}

} // namespace tianchenrv::conversion::emitc
