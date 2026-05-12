#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"

#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/Cpp/CppEmitter.h"
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

std::string makeSourceAuthorityComment(llvm::StringRef key,
                                       llvm::StringRef value) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// tcrv_emitc." << key << "=" << value;
  os.flush();
  return text;
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

class RouteCppSourceAuthorityMaterializer {
public:
  RouteCppSourceAuthorityMaterializer(
      mlir::MLIRContext &context, const TCRVEmitCLowerableRoute &route,
      const TCRVEmitCSourceAuthorityOptions &options)
      : context(context), route(route), options(options), builder(&context) {}

  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> run() {
    if (llvm::Error error = route.verify())
      return std::move(error);
    if (llvm::Error error = validateOptions())
      return std::move(error);
    if (llvm::Error error = initializeRuntimeElementCount())
      return std::move(error);
    if (llvm::Error error = validateSourceAuthorityRouteShape())
      return std::move(error);

    context.getOrLoadDialect<mlir::emitc::EmitCDialect>();

    mlir::Location loc = builder.getUnknownLoc();
    mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
    builder.setInsertionPointToStart(module->getBody());

    emitSourceAuthorityHeaders(loc);
    builder.create<mlir::emitc::VerbatimOp>(
        loc, makeSourceAuthorityComment("source_authority",
                                        "mlir_emitc_cpp_emitter"));
    builder.create<mlir::emitc::VerbatimOp>(
        loc, makeSourceAuthorityComment("source_route_id",
                                        route.getRouteID()));
    builder.create<mlir::emitc::VerbatimOp>(
        loc, makeSourceAuthorityComment("source_route_kind",
                                        route.getRouteKind()));
    if (sourceAuthorityShape == SourceAuthorityShape::DispatchControl)
      builder.create<mlir::emitc::VerbatimOp>(
          loc, makeSourceAuthorityComment("dispatch_control_source",
                                          "tcrv.exec.dispatch"));
    if (sourceAuthorityShape == SourceAuthorityShape::DispatchControl)
      builder.create<mlir::emitc::VerbatimOp>(
          loc, makeSourceAuthorityComment("dispatch_guard_value",
                                          options.dispatchGuardValueName));
    builder.create<mlir::emitc::VerbatimOp>(
        loc, makeSourceAuthorityComment("public_function",
                                        options.functionName));

    if (sourceAuthorityShape == SourceAuthorityShape::DispatchControl) {
      mlir::emitc::FuncOp publicFunction = buildPublicFunction(loc);
      if (llvm::Error error = materializeDispatchControlBody(publicFunction))
        return std::move(error);
    } else {
      mlir::emitc::FuncOp helper = buildHelperFunction(loc);
      if (llvm::Error error = materializeHelperBody(helper))
        return std::move(error);

      builder.setInsertionPointAfter(helper);
      mlir::emitc::FuncOp publicFunction = buildPublicFunction(loc);
      if (llvm::Error error = materializePublicWrapper(publicFunction, helper))
        return std::move(error);
    }

    if (options.verifyModule && mlir::failed(mlir::verify(*module)))
      return makeMaterializerError(route.getRouteID(),
                                   "source-authority EmitC module failed "
                                   "MLIR verification");
    return module;
  }

private:
  enum class SourceAuthorityShape {
    RuntimeAVLToVL,
    RuntimeElementCount,
    DispatchControl
  };

  llvm::Error validateOptions() {
    if (llvm::Error error = validateSafeIdentifier(
            route.getRouteID(), "EmitC source-authority function name",
            options.functionName))
      return error;
    if (llvm::Error error =
            validateSafeIdentifier(route.getRouteID(),
                                   "EmitC source-authority loop index name",
                                   options.loopIndexName))
      return error;
    if (!options.dispatchGuardValueName.empty())
      if (llvm::Error error =
              validateSafeIdentifier(route.getRouteID(),
                                     "EmitC dispatch guard value name",
                                     options.dispatchGuardValueName))
        return error;
    helperFunctionName = options.helperFunctionName;
    if (helperFunctionName.empty())
      helperFunctionName =
          (llvm::Twine(options.functionName) + "__tcrv_emitc_body").str();
    if (llvm::Error error =
            validateSafeIdentifier(route.getRouteID(),
                                   "EmitC source-authority helper function "
                                   "name",
                                   helperFunctionName))
      return error;
    if (helperFunctionName == options.functionName)
      return makeMaterializerError(
          route.getRouteID(),
          "source-authority helper function name must differ from the public "
          "function name");
    return llvm::Error::success();
  }

  llvm::Error initializeRuntimeElementCount() {
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
      if (!seenABIValueNames.insert(valueName).second)
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("duplicate ABI value mapping name '") + valueName +
                "'");
      if (mapping.parameter.role ==
          support::RuntimeABIParameterRole::RuntimeElementCount) {
        runtimeElementCountValueName = valueName.str();
        ++runtimeElementCountMatches;
        if (mapping.parameter.cType != "size_t")
          return makeMaterializerError(
              route.getRouteID(),
              "MLIR Cpp source authority currently requires the "
              "runtime-element-count ABI mapping to use C type 'size_t'");
      }
      if (mapping.parameter.role ==
          support::RuntimeABIParameterRole::DispatchAvailabilityGuard) {
        dispatchGuardValueNames.push_back(valueName.str());
        if (mapping.parameter.cType != "int")
          return makeMaterializerError(
              route.getRouteID(),
              "MLIR Cpp source authority currently requires the "
              "dispatch-availability-guard ABI mapping to use C type 'int'");
      }
    }

    if (runtimeElementCountMatches != 1)
      return makeMaterializerError(
          route.getRouteID(),
          "MLIR Cpp source authority requires exactly one "
          "runtime-element-count ABI mapping");
    return llvm::Error::success();
  }

  llvm::Error validateSourceAuthorityRouteShape() {
    llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps =
        route.getCallOpaqueSteps();
    if (steps.empty())
      return makeMaterializerError(
          route.getRouteID(),
          "MLIR Cpp source authority requires at least one call_opaque step");
    if (!options.dispatchGuardValueName.empty()) {
      sourceAuthorityShape = SourceAuthorityShape::DispatchControl;
      return validateDispatchControlRouteShape();
    }
    if (steps.front().sourceOp.role == "runtime-avl-to-vl") {
      sourceAuthorityShape = SourceAuthorityShape::RuntimeAVLToVL;
      return validateRuntimeVLRouteShape();
    }

    sourceAuthorityShape = SourceAuthorityShape::RuntimeElementCount;
    return validateRuntimeElementCountRouteShape();
  }

  llvm::Error validateRuntimeVLRouteShape() const {
    llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps =
        route.getCallOpaqueSteps();
    if (steps.size() < 2)
      return makeMaterializerError(
          route.getRouteID(),
          "MLIR Cpp source authority requires a runtime-avl-to-vl step "
          "followed by bounded body call_opaque steps");
    const TCRVEmitCCallOpaqueStep &first = steps.front();
    if (first.sourceOp.role != "runtime-avl-to-vl")
      return makeMaterializerError(
          route.getRouteID(),
          "MLIR Cpp source authority currently supports only bounded "
          "runtime-avl-to-vl routes");
    if (!first.result || first.result->cType != "size_t")
      return makeMaterializerError(
          route.getRouteID(),
          "runtime-avl-to-vl source-authority step must produce a size_t VL "
          "result");
    if (first.operands.size() != 1)
      return makeMaterializerError(
          route.getRouteID(),
          "runtime-avl-to-vl source-authority step requires exactly one "
          "remaining-element operand");
    std::string expectedRemaining =
        (llvm::Twine(runtimeElementCountValueName) + " - " +
         options.loopIndexName)
            .str();
    if (first.operands.front().expression != expectedRemaining)
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("runtime-avl-to-vl source-authority operand must be '") +
              expectedRemaining + "'");
    return llvm::Error::success();
  }

  llvm::Error validateRuntimeElementCountRouteShape() const {
    for (const TCRVEmitCCallOpaqueStep &step : route.getCallOpaqueSteps()) {
      if (step.sourceOp.role == "runtime-avl-to-vl")
        return makeMaterializerError(
            route.getRouteID(),
            "runtime-element-count MLIR Cpp source authority cannot mix a "
            "runtime-avl-to-vl call_opaque step into the scalar loop shape");
      if (step.sourceOp.role != "compute" && step.result)
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("runtime-element-count MLIR Cpp source authority "
                        "cannot materialize non-compute call_opaque result '") +
                step.result->name + "'");
    }
    return llvm::Error::success();
  }

  llvm::Error validateDispatchControlRouteShape() const {
    if (!seenABIValueNames.contains(options.dispatchGuardValueName))
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("dispatch-control MLIR Cpp source authority requires "
                      "dispatch guard ABI value '") +
              options.dispatchGuardValueName + "'");
    if (dispatchGuardValueNames.size() != 1 ||
        dispatchGuardValueNames.front() != options.dispatchGuardValueName)
      return makeMaterializerError(
          route.getRouteID(),
          "dispatch-control MLIR Cpp source authority requires exactly one "
          "dispatch-availability-guard ABI mapping matching the configured "
          "guard value name");

    llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps =
        route.getCallOpaqueSteps();
    if (steps.size() != 2)
      return makeMaterializerError(
          route.getRouteID(),
          "dispatch-control MLIR Cpp source authority requires exactly two "
          "call_opaque steps: dispatch case and dispatch fallback");
    if (steps[0].sourceOp.role != "dispatch-case-call")
      return makeMaterializerError(
          route.getRouteID(),
          "dispatch-control MLIR Cpp source authority requires the first "
          "call_opaque step to have source role 'dispatch-case-call'");
    if (steps[1].sourceOp.role != "dispatch-fallback-call")
      return makeMaterializerError(
          route.getRouteID(),
          "dispatch-control MLIR Cpp source authority requires the second "
          "call_opaque step to have source role 'dispatch-fallback-call'");
    for (const TCRVEmitCCallOpaqueStep &step : steps) {
      if (step.result)
        return makeMaterializerError(
            route.getRouteID(),
            "dispatch-control MLIR Cpp source authority requires component "
            "call_opaque steps to be void calls");
      if (step.operands.empty())
        return makeMaterializerError(
            route.getRouteID(),
            "dispatch-control MLIR Cpp source authority requires component "
            "calls to carry ABI-ordered callable operands");
    }
    return llvm::Error::success();
  }

  void emitSourceAuthorityHeaders(mlir::Location loc) {
    llvm::StringSet<> emittedHeaders;
    for (const TCRVEmitCHeaderRequirement &header : route.getHeaders()) {
      if (!emittedHeaders.insert(header.header).second)
        continue;
      builder.create<mlir::emitc::IncludeOp>(loc, header.header,
                                             /*is_standard_include=*/true);
    }
    if (emittedHeaders.insert("stdbool.h").second)
      builder.create<mlir::emitc::IncludeOp>(loc, "stdbool.h",
                                             /*is_standard_include=*/true);
  }

  mlir::FunctionType buildHelperFunctionType() {
    llvm::SmallVector<mlir::Type, 8> inputTypes;
    for (const TCRVEmitCABIValueMapping &mapping : route.getABIMappings())
      inputTypes.push_back(
          getEmitCTypeForCType(context, mapping.parameter.cType));
    inputTypes.push_back(mlir::emitc::SizeTType::get(&context));
    return builder.getFunctionType(inputTypes, llvm::ArrayRef<mlir::Type>{});
  }

  mlir::FunctionType buildPublicFunctionType() {
    llvm::SmallVector<mlir::Type, 6> inputTypes;
    for (const TCRVEmitCABIValueMapping &mapping : route.getABIMappings())
      inputTypes.push_back(
          getEmitCTypeForCType(context, mapping.parameter.cType));
    return builder.getFunctionType(inputTypes, llvm::ArrayRef<mlir::Type>{});
  }

  mlir::emitc::FuncOp buildHelperFunction(mlir::Location loc) {
    mlir::emitc::FuncOp function = builder.create<mlir::emitc::FuncOp>(
        loc, helperFunctionName, buildHelperFunctionType());
    function->setAttr("specifiers", builder.getStrArrayAttr({"static"}));
    mlir::Block *entry = new mlir::Block();
    function.getBody().push_back(entry);
    for (mlir::Type type : function.getFunctionType().getInputs())
      entry->addArgument(type, loc);
    return function;
  }

  mlir::emitc::FuncOp buildPublicFunction(mlir::Location loc) {
    mlir::emitc::FuncOp function = builder.create<mlir::emitc::FuncOp>(
        loc, options.functionName, buildPublicFunctionType());
    mlir::Block *entry = new mlir::Block();
    function.getBody().push_back(entry);
    for (mlir::Type type : function.getFunctionType().getInputs())
      entry->addArgument(type, loc);
    return function;
  }

  void initializeFunctionValueMap(mlir::Block *entry,
                                  bool includeLoopIndexArgument) {
    valueMap.clear();
    for (auto [index, mapping] : llvm::enumerate(route.getABIMappings()))
      valueMap[mapping.valueName] = entry->getArgument(index);
    if (includeLoopIndexArgument)
      valueMap[options.loopIndexName] =
          entry->getArgument(route.getABIMappings().size());
  }

  mlir::Value lookupRequiredValue(llvm::StringRef name) const {
    return valueMap.lookup(name);
  }

  llvm::Expected<mlir::Value>
  materializeSubscriptLValueExpression(llvm::StringRef expression,
                                       mlir::Location loc) {
    if (!expression.ends_with((llvm::Twine("[") + options.loopIndexName + "]")
                                  .str()))
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("source-authority operand expression '") + expression +
              "' is not a supported ABI buffer subscript expression");

    llvm::StringRef base =
        expression.drop_back(options.loopIndexName.size() + 2);
    if (llvm::Error error = validateSafeIdentifier(
            route.getRouteID(), "ABI buffer expression base", base))
      return std::move(error);
    mlir::Value pointer = lookupRequiredValue(base);
    if (!pointer)
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("source-authority operand expression references "
                      "unknown ABI buffer '") +
              base + "'");
    mlir::Value offset = lookupRequiredValue(options.loopIndexName);
    if (!offset)
      return makeMaterializerError(
          route.getRouteID(),
          "source-authority body is missing the runtime loop index value");

    auto pointerType =
        llvm::dyn_cast<mlir::emitc::PointerType>(pointer.getType());
    if (!pointerType)
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("source-authority ABI buffer '") + base +
              "' must have EmitC pointer type before subscript emission");
    mlir::Type lvalueType =
        mlir::emitc::LValueType::get(pointerType.getPointee());
    mlir::emitc::SubscriptOp subscript =
        builder.create<mlir::emitc::SubscriptOp>(
            loc, lvalueType, pointer, mlir::ValueRange{offset});
    return subscript.getResult();
  }

  llvm::Expected<mlir::Value>
  materializeAddressOfSubscriptExpression(llvm::StringRef expression,
                                          mlir::Location loc) {
    if (!expression.starts_with("&"))
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("source-authority operand expression '") + expression +
              "' is not a supported ABI buffer address expression");

    llvm::Expected<mlir::Value> lvalue =
        materializeSubscriptLValueExpression(expression.drop_front(), loc);
    if (!lvalue)
      return lvalue.takeError();
    auto lvalueType = llvm::cast<mlir::emitc::LValueType>((*lvalue).getType());
    mlir::Type pointerType =
        mlir::emitc::PointerType::get(&context, lvalueType.getValueType());
    return builder
        .create<mlir::emitc::ApplyOp>(loc, pointerType, "&", *lvalue)
        .getResult();
  }

  llvm::Expected<mlir::Value>
  materializeSourceOperand(const TCRVEmitCCallOpaqueOperand &operand,
                           mlir::Location loc) {
    llvm::StringRef expression = operand.expression;
    if (!isSafeExpressionText(expression))
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("source-authority operand expression '") + expression +
              "' contains unsafe text before EmitC materialization");

    if (isSafeIdentifier(expression)) {
      if (mlir::Value value = lookupRequiredValue(expression))
        return value;
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("source-authority operand expression references "
                      "unknown value name '") +
              expression + "'");
    }

    if (expression.starts_with("&"))
      return materializeAddressOfSubscriptExpression(expression, loc);

    llvm::Expected<mlir::Value> lvalue =
        materializeSubscriptLValueExpression(expression, loc);
    if (!lvalue)
      return lvalue.takeError();
    mlir::Type valueType =
        llvm::cast<mlir::emitc::LValueType>((*lvalue).getType())
            .getValueType();
    return builder
        .create<mlir::emitc::LoadOp>(loc, valueType, *lvalue)
        .getResult();
  }

  llvm::Error materializeRouteStep(const TCRVEmitCCallOpaqueStep &step) {
    if (llvm::Error error = validateSafeProvenance(route, step.sourceOp))
      return error;
    if (llvm::Error error =
            validateSafeIdentifier(route.getRouteID(), "call_opaque callee",
                                   step.callee))
      return error;
    if (step.sourceOp.role == "compute") {
      if (!step.result)
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("compute step from source op '") +
                step.sourceOp.opName +
                "' requires a result value before MLIR Cpp source emission");
      if (options.requireInterfaceBackedCompute &&
          step.sourceOp.opInterface.empty())
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("compute step from source op '") +
                step.sourceOp.opName +
                "' requires generated op-interface provenance before MLIR "
                "Cpp source emission");
      sawInterfaceBackedCompute = true;
    }

    builder.create<mlir::emitc::VerbatimOp>(makeStepLocation(builder, step),
                                            makeStepProvenanceComment(step));

    llvm::SmallVector<mlir::Value, 4> operands;
    for (const TCRVEmitCCallOpaqueOperand &operand : step.operands) {
      llvm::Expected<mlir::Value> value =
          materializeSourceOperand(operand, makeStepLocation(builder, step));
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
      if (valueMap.contains(step.result->name))
        return makeMaterializerError(
            route.getRouteID(),
            llvm::Twine("duplicate source-authority value name '") +
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

  llvm::Error materializeRuntimeVLStep(const TCRVEmitCCallOpaqueStep &step) {
    mlir::Location loc = makeStepLocation(builder, step);
    builder.create<mlir::emitc::VerbatimOp>(loc,
                                            makeStepProvenanceComment(step));
    mlir::Value runtimeN = lookupRequiredValue(runtimeElementCountValueName);
    mlir::Value offset = lookupRequiredValue(options.loopIndexName);
    if (!runtimeN || !offset)
      return makeMaterializerError(
          route.getRouteID(),
          "source-authority runtime VL step is missing runtime n or offset "
          "values");
    mlir::Type sizeTType = mlir::emitc::SizeTType::get(&context);
    mlir::Value remaining =
        builder.create<mlir::emitc::SubOp>(loc, sizeTType, runtimeN, offset)
            .getResult();
    mlir::emitc::CallOpaqueOp call =
        builder.create<mlir::emitc::CallOpaqueOp>(
            loc, mlir::TypeRange{sizeTType}, step.callee,
            mlir::ValueRange{remaining});
    valueMap[step.result->name] = call->getResult(0);
    return llvm::Error::success();
  }

  llvm::Error materializeHelperBody(mlir::emitc::FuncOp helper) {
    mlir::Block &entry = helper.getBody().front();
    builder.setInsertionPointToStart(&entry);
    initializeFunctionValueMap(&entry, /*includeLoopIndexArgument=*/true);

    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value runtimeN = lookupRequiredValue(runtimeElementCountValueName);
    mlir::Value offset = lookupRequiredValue(options.loopIndexName);
    mlir::Value condition =
        builder
            .create<mlir::emitc::CmpOp>(
                loc, builder.getI1Type(), mlir::emitc::CmpPredicate::lt,
                offset, runtimeN)
            .getResult();
    mlir::emitc::IfOp ifOp =
        builder.create<mlir::emitc::IfOp>(loc, condition);
    mlir::Block &thenBlock = ifOp.getThenRegion().front();
    builder.setInsertionPoint(thenBlock.getTerminator());

    llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps = route.getCallOpaqueSteps();
    if (sourceAuthorityShape == SourceAuthorityShape::RuntimeAVLToVL) {
      if (llvm::Error error = materializeRuntimeVLStep(steps.front()))
        return error;
      for (const TCRVEmitCCallOpaqueStep &step : steps.drop_front())
        if (llvm::Error error = materializeRouteStep(step))
          return error;
    } else {
      for (const TCRVEmitCCallOpaqueStep &step : steps)
        if (llvm::Error error = materializeRouteStep(step))
          return error;
    }

    if (options.requireInterfaceBackedCompute && !sawInterfaceBackedCompute)
      return makeMaterializerError(
          route.getRouteID(),
          "MLIR Cpp source authority requires at least one "
          "interface-backed compute call_opaque step");

    mlir::Value increment;
    if (sourceAuthorityShape == SourceAuthorityShape::RuntimeAVLToVL)
      increment = lookupRequiredValue(steps.front().result->name);
    else
      increment =
          builder
              .create<mlir::emitc::LiteralOp>(
                  loc, mlir::emitc::SizeTType::get(&context), "1")
              .getResult();
    mlir::Value nextOffset =
        builder
            .create<mlir::emitc::AddOp>(
                loc, mlir::emitc::SizeTType::get(&context), offset, increment)
            .getResult();
    llvm::SmallVector<mlir::Value, 8> recursiveArgs;
    for (const TCRVEmitCABIValueMapping &mapping : route.getABIMappings())
      recursiveArgs.push_back(lookupRequiredValue(mapping.valueName));
    recursiveArgs.push_back(nextOffset);
    builder.create<mlir::emitc::CallOp>(
        loc, helperFunctionName, mlir::TypeRange{}, recursiveArgs);

    builder.setInsertionPointAfter(ifOp);
    builder.create<mlir::emitc::ReturnOp>(loc, mlir::Value());
    return llvm::Error::success();
  }

  llvm::Error
  materializeDispatchControlBody(mlir::emitc::FuncOp publicFunction) {
    mlir::Block &entry = publicFunction.getBody().front();
    builder.setInsertionPointToStart(&entry);
    initializeFunctionValueMap(&entry, /*includeLoopIndexArgument=*/false);

    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value guard = lookupRequiredValue(options.dispatchGuardValueName);
    if (!guard)
      return makeMaterializerError(
          route.getRouteID(),
          llvm::Twine("dispatch-control body is missing guard value '") +
              options.dispatchGuardValueName + "'");
    mlir::Value zero =
        builder
            .create<mlir::emitc::LiteralOp>(loc, guard.getType(), "0")
            .getResult();
    mlir::Value condition =
        builder
            .create<mlir::emitc::CmpOp>(
                loc, builder.getI1Type(), mlir::emitc::CmpPredicate::ne,
                guard, zero)
            .getResult();
    mlir::emitc::IfOp ifOp =
        builder.create<mlir::emitc::IfOp>(loc, condition);

    llvm::ArrayRef<TCRVEmitCCallOpaqueStep> steps =
        route.getCallOpaqueSteps();
    mlir::Block &thenBlock = ifOp.getThenRegion().front();
    builder.setInsertionPoint(thenBlock.getTerminator());
    if (llvm::Error error = materializeRouteStep(steps.front()))
      return error;
    builder.create<mlir::emitc::VerbatimOp>(loc, "return;");

    builder.setInsertionPointAfter(ifOp);
    if (llvm::Error error = materializeRouteStep(steps[1]))
      return error;
    builder.create<mlir::emitc::ReturnOp>(loc, mlir::Value());
    return llvm::Error::success();
  }

  llvm::Error materializePublicWrapper(mlir::emitc::FuncOp publicFunction,
                                       mlir::emitc::FuncOp helper) {
    mlir::Block &entry = publicFunction.getBody().front();
    builder.setInsertionPointToStart(&entry);
    initializeFunctionValueMap(&entry, /*includeLoopIndexArgument=*/false);

    mlir::Location loc = builder.getUnknownLoc();
    mlir::Value zero =
        builder
            .create<mlir::emitc::LiteralOp>(
                loc, mlir::emitc::SizeTType::get(&context), "0")
            .getResult();
    llvm::SmallVector<mlir::Value, 8> callArgs;
    for (const TCRVEmitCABIValueMapping &mapping : route.getABIMappings())
      callArgs.push_back(lookupRequiredValue(mapping.valueName));
    callArgs.push_back(zero);
    builder.create<mlir::emitc::CallOp>(loc, helper, callArgs);
    builder.create<mlir::emitc::ReturnOp>(loc, mlir::Value());
    return llvm::Error::success();
  }

  mlir::MLIRContext &context;
  const TCRVEmitCLowerableRoute &route;
  const TCRVEmitCSourceAuthorityOptions &options;
  mlir::OpBuilder builder;
  llvm::StringSet<> seenABIValueNames;
  llvm::StringMap<mlir::Value> valueMap;
  std::string helperFunctionName;
  std::string runtimeElementCountValueName;
  llvm::SmallVector<std::string, 1> dispatchGuardValueNames;
  unsigned runtimeElementCountMatches = 0;
  bool sawInterfaceBackedCompute = false;
  SourceAuthorityShape sourceAuthorityShape =
      SourceAuthorityShape::RuntimeAVLToVL;
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

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeTCRVEmitCLowerableRouteSourceAuthority(
    mlir::MLIRContext &context, const TCRVEmitCLowerableRoute &route,
    const TCRVEmitCSourceAuthorityOptions &options) {
  return RouteCppSourceAuthorityMaterializer(context, route, options).run();
}

llvm::Error emitTCRVEmitCLowerableRouteAsCppSource(
    const TCRVEmitCLowerableRoute &route, llvm::raw_ostream &os,
    const TCRVEmitCSourceAuthorityOptions &options) {
  mlir::MLIRContext context;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRouteSourceAuthority(context, route,
                                                        options);
  if (!module)
    return module.takeError();

  std::string source;
  llvm::raw_string_ostream sourceOS(source);
  if (mlir::failed(mlir::emitc::translateToCpp(
          module->get().getOperation(), sourceOS, options.declareVariablesAtTop)))
    return makeMaterializerError(route.getRouteID(),
                                 "MLIR Cpp emitter failed while translating "
                                 "the source-authority EmitC module");
  sourceOS.flush();
  os << source;
  return llvm::Error::success();
}

} // namespace tianchenrv::conversion::emitc
