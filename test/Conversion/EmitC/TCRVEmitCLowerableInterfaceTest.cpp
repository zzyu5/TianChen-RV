#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace tianchenrv::conversion::emitc;

namespace {

constexpr llvm::StringLiteral kEmitCLowerableOpInterfaceName(
    "TCRVEmitCLowerableOpInterface");

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

mlir::Operation *createRVVArithmeticOp(mlir::OpBuilder &builder,
                                       mlir::Location loc,
                                       llvm::StringRef opName,
                                       mlir::Value lhs, mlir::Value rhs,
                                       mlir::Value vl,
                                       mlir::Type resultType) {
  mlir::OperationState state(loc, opName);
  state.addOperands({lhs, rhs, vl});
  state.addTypes(resultType);
  return builder.create(state);
}

int expectRVVArithmeticOpInterface(mlir::Operation *op,
                                   llvm::StringRef expectedOpName) {
  auto lowerable = llvm::dyn_cast<TCRVEmitCLowerableOpInterface>(op);
  if (!lowerable)
    return fail(llvm::Twine(expectedOpName) +
                " must implement generated TCRVEmitCLowerableOpInterface");
  if (int result = expect(lowerable.getTCRVEmitCLowerableSourceOpName() ==
                              expectedOpName,
                          llvm::Twine(expectedOpName) +
                              " reports generated-interface source op name"))
    return result;
  if (int result = expect(lowerable.getTCRVEmitCLowerableSourceRole() ==
                              "compute",
                          llvm::Twine(expectedOpName) +
                              " reports generated-interface source role"))
    return result;
  return 0;
}

class GeneratedRVVArithmeticLowerable final
    : public TCRVEmitCLowerableInterface {
public:
  GeneratedRVVArithmeticLowerable(TCRVEmitCLowerableOpInterface sourceOp,
                                  llvm::StringRef callee,
                                  llvm::StringRef resultName)
      : sourceOpName(sourceOp.getTCRVEmitCLowerableSourceOpName().str()),
        sourceOpRole(sourceOp.getTCRVEmitCLowerableSourceRole().str()),
        callee(callee.str()), resultName(resultName.str()) {}

  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    using tianchenrv::support::RuntimeABIParameter;
    using tianchenrv::support::RuntimeABIParameterOwnership;
    using tianchenrv::support::RuntimeABIParameterRole;

    TCRVEmitCLowerableRoute route(
        "tcrv-export-rvv-microkernel-c",
        "extension-family-ops-to-emitc-call-opaque");
    route.addHeader("stddef.h");
    route.addHeader("stdint.h");
    route.addHeader("riscv_vector.h");
    route.addTypeMapping("!tcrv_rvv.vl", "size_t");
    route.addTypeMapping("!tcrv_rvv.i32m1", "vint32m1_t");
    route.addABIValueMapping(
        RuntimeABIParameter("lhs", "const int32_t *",
                            RuntimeABIParameterRole::LHSInputBuffer,
                            RuntimeABIParameterOwnership::TargetExportABIOwned),
        "lhs");
    route.addABIValueMapping(
        RuntimeABIParameter("rhs", "const int32_t *",
                            RuntimeABIParameterRole::RHSInputBuffer,
                            RuntimeABIParameterOwnership::TargetExportABIOwned),
        "rhs");
    route.addABIValueMapping(
        RuntimeABIParameter("out", "int32_t *",
                            RuntimeABIParameterRole::OutputBuffer,
                            RuntimeABIParameterOwnership::TargetExportABIOwned),
        "out");
    route.addABIValueMapping(
        RuntimeABIParameter(
            "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
            RuntimeABIParameterOwnership::TargetExportABIOwned),
        "n");

    TCRVEmitCCallOpaqueStep setvl;
    setvl.sourceOp = {"tcrv_rvv.setvl", "runtime-avl-to-vl"};
    setvl.callee = "__riscv_vsetvl_e32m1";
    setvl.operands.push_back({"n - offset", "size_t"});
    setvl.result = TCRVEmitCCallOpaqueResult{"vl", "size_t"};
    route.addCallOpaqueStep(std::move(setvl));

    TCRVEmitCCallOpaqueStep lhsLoad;
    lhsLoad.sourceOp = {"tcrv_rvv.i32_load", "buffer-load"};
    lhsLoad.callee = "__riscv_vle32_v_i32m1";
    lhsLoad.operands.push_back({"&lhs[offset]", "const int32_t *"});
    lhsLoad.operands.push_back({"vl", "size_t"});
    lhsLoad.result = TCRVEmitCCallOpaqueResult{"lhs_vec", "vint32m1_t"};
    route.addCallOpaqueStep(std::move(lhsLoad));

    TCRVEmitCCallOpaqueStep rhsLoad;
    rhsLoad.sourceOp = {"tcrv_rvv.i32_load", "buffer-load"};
    rhsLoad.callee = "__riscv_vle32_v_i32m1";
    rhsLoad.operands.push_back({"&rhs[offset]", "const int32_t *"});
    rhsLoad.operands.push_back({"vl", "size_t"});
    rhsLoad.result = TCRVEmitCCallOpaqueResult{"rhs_vec", "vint32m1_t"};
    route.addCallOpaqueStep(std::move(rhsLoad));

    TCRVEmitCCallOpaqueStep arithmetic;
    arithmetic.sourceOp = {sourceOpName, sourceOpRole,
                           kEmitCLowerableOpInterfaceName.str()};
    arithmetic.callee = callee;
    arithmetic.operands.push_back({"lhs_vec", "vint32m1_t"});
    arithmetic.operands.push_back({"rhs_vec", "vint32m1_t"});
    arithmetic.operands.push_back({"vl", "size_t"});
    arithmetic.result = TCRVEmitCCallOpaqueResult{resultName, "vint32m1_t"};
    route.addCallOpaqueStep(std::move(arithmetic));

    TCRVEmitCCallOpaqueStep store;
    store.sourceOp = {"tcrv_rvv.i32_store", "buffer-store"};
    store.callee = "__riscv_vse32_v_i32m1";
    store.operands.push_back({"&out[offset]", "int32_t *"});
    store.operands.push_back({resultName, "vint32m1_t"});
    store.operands.push_back({"vl", "size_t"});
    route.addCallOpaqueStep(std::move(store));

    return route;
  }

private:
  std::string sourceOpName;
  std::string sourceOpRole;
  std::string callee;
  std::string resultName;
};

class InvalidMissingCalleeLowerable final : public TCRVEmitCLowerableInterface {
public:
  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    TCRVEmitCLowerableRoute route("bad-route",
                                  "extension-family-ops-to-emitc-call-opaque");
    route.addHeader("riscv_vector.h");
    TCRVEmitCCallOpaqueStep add;
    add.sourceOp = {"tcrv_rvv.i32_add", "compute"};
    add.result = TCRVEmitCCallOpaqueResult{"sum_vec", "vint32m1_t"};
    route.addCallOpaqueStep(std::move(add));
    return route;
  }
};

void addStandardABI(TCRVEmitCLowerableRoute &route,
                    llvm::StringRef lhsValueName = "lhs") {
  using tianchenrv::support::RuntimeABIParameter;
  using tianchenrv::support::RuntimeABIParameterOwnership;
  using tianchenrv::support::RuntimeABIParameterRole;

  route.addABIValueMapping(
      RuntimeABIParameter("lhs", "const int32_t *",
                          RuntimeABIParameterRole::LHSInputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      lhsValueName);
  route.addABIValueMapping(
      RuntimeABIParameter("rhs", "const int32_t *",
                          RuntimeABIParameterRole::RHSInputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "rhs");
  route.addABIValueMapping(
      RuntimeABIParameter("out", "int32_t *",
                          RuntimeABIParameterRole::OutputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "out");
  route.addABIValueMapping(
      RuntimeABIParameter(
          "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "n");
}

TCRVEmitCLowerableRoute
makeMinimalMaterializerRoute(llvm::StringRef arithmeticSourceOp,
                             llvm::StringRef arithmeticRole,
                             llvm::StringRef callee,
                             std::optional<llvm::StringRef> resultName,
                             llvm::StringRef lhsOperand = "lhs",
                             llvm::StringRef setVLResultName = "vl",
                             llvm::StringRef lhsValueName = "lhs",
                             llvm::StringRef sourceOpInterface =
                                 kEmitCLowerableOpInterfaceName) {
  TCRVEmitCLowerableRoute route(
      "tcrv-export-rvv-microkernel-c",
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", "size_t");
  route.addTypeMapping("!tcrv_rvv.i32m1", "vint32m1_t");
  addStandardABI(route, lhsValueName);

  TCRVEmitCCallOpaqueStep setvl;
  setvl.sourceOp = {"tcrv_rvv.setvl", "runtime-avl-to-vl"};
  setvl.callee = "__riscv_vsetvl_e32m1";
  setvl.operands.push_back({"n - offset", "size_t"});
  setvl.result = TCRVEmitCCallOpaqueResult{setVLResultName.str(), "size_t"};
  route.addCallOpaqueStep(std::move(setvl));

  TCRVEmitCCallOpaqueStep arithmetic;
  arithmetic.sourceOp = {arithmeticSourceOp.str(), arithmeticRole.str(),
                         sourceOpInterface.str()};
  arithmetic.callee = callee;
  arithmetic.operands.push_back({lhsOperand.str(), "const int32_t *"});
  arithmetic.operands.push_back({"rhs", "const int32_t *"});
  arithmetic.operands.push_back({setVLResultName.str(), "size_t"});
  if (resultName)
    arithmetic.result =
        TCRVEmitCCallOpaqueResult{resultName->str(), "vint32m1_t"};
  route.addCallOpaqueStep(std::move(arithmetic));
  return route;
}

TCRVEmitCLowerableRoute makeRouteWithNonAVLFirstStep() {
  TCRVEmitCLowerableRoute route(
      "tcrv-export-rvv-microkernel-c",
      "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("riscv_vector.h");
  route.addTypeMapping("!tcrv_rvv.vl", "size_t");
  addStandardABI(route);

  TCRVEmitCCallOpaqueStep load;
  load.sourceOp = {"tcrv_rvv.i32_load", "buffer-load"};
  load.callee = "__riscv_vle32_v_i32m1";
  load.operands.push_back({"&lhs[offset]", "const int32_t *"});
  load.operands.push_back({"n", "size_t"});
  load.result = TCRVEmitCCallOpaqueResult{"lhs_vec", "vint32m1_t"};
  route.addCallOpaqueStep(std::move(load));

  TCRVEmitCCallOpaqueStep arithmetic;
  arithmetic.sourceOp = {"tcrv_rvv.i32_add", "compute",
                         kEmitCLowerableOpInterfaceName.str()};
  arithmetic.callee = "__riscv_vadd_vv_i32m1";
  arithmetic.operands.push_back({"lhs_vec", "vint32m1_t"});
  arithmetic.operands.push_back({"lhs_vec", "vint32m1_t"});
  arithmetic.operands.push_back({"n", "size_t"});
  arithmetic.result = TCRVEmitCCallOpaqueResult{"sum_vec", "vint32m1_t"};
  route.addCallOpaqueStep(std::move(arithmetic));
  return route;
}

void addStandardScalarABI(TCRVEmitCLowerableRoute &route,
                          bool includeRuntimeElementCount = true) {
  using tianchenrv::support::RuntimeABIParameter;
  using tianchenrv::support::RuntimeABIParameterOwnership;
  using tianchenrv::support::RuntimeABIParameterRole;

  route.addABIValueMapping(
      RuntimeABIParameter("lhs", "const int32_t *",
                          RuntimeABIParameterRole::LHSInputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "lhs");
  route.addABIValueMapping(
      RuntimeABIParameter("rhs", "const int32_t *",
                          RuntimeABIParameterRole::RHSInputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "rhs");
  route.addABIValueMapping(
      RuntimeABIParameter("out", "int32_t *",
                          RuntimeABIParameterRole::OutputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "out");
  if (includeRuntimeElementCount)
    route.addABIValueMapping(
        RuntimeABIParameter(
            "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
            RuntimeABIParameterOwnership::TargetExportABIOwned),
        "n");
}

TCRVEmitCLowerableRoute
makeScalarElementLoopRoute(bool includeRuntimeElementCount = true,
                           bool includeInterfaceProvenance = true) {
  TCRVEmitCLowerableRoute route(
      "tcrv-export-scalar-microkernel-c",
      "typed-scalar-family-op-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addTypeMapping("i32", "int32_t");
  addStandardScalarABI(route, includeRuntimeElementCount);

  TCRVEmitCCallOpaqueStep arithmetic;
  arithmetic.sourceOp = {"tcrv_scalar.i32_vadd_microkernel", "compute"};
  if (includeInterfaceProvenance)
    arithmetic.sourceOp.opInterface = kEmitCLowerableOpInterfaceName.str();
  arithmetic.callee = "tcrv_scalar_i32_add";
  arithmetic.operands.push_back({"lhs[index]", "int32_t"});
  arithmetic.operands.push_back({"rhs[index]", "int32_t"});
  arithmetic.result = TCRVEmitCCallOpaqueResult{"sum", "int32_t"};
  route.addCallOpaqueStep(std::move(arithmetic));

  TCRVEmitCCallOpaqueStep store;
  store.sourceOp = {"tcrv_scalar.i32_vadd_microkernel", "buffer-store"};
  if (includeInterfaceProvenance)
    store.sourceOp.opInterface = kEmitCLowerableOpInterfaceName.str();
  store.callee = "tcrv_scalar_i32_store";
  store.operands.push_back({"&out[index]", "int32_t *"});
  store.operands.push_back({"sum", "int32_t"});
  route.addCallOpaqueStep(std::move(store));
  return route;
}

TCRVEmitCLowerableRoute
makeDispatchControlRoute(bool includeDispatchGuard = true,
                         llvm::StringRef firstRole = "dispatch-case-call") {
  using tianchenrv::support::RuntimeABIParameter;
  using tianchenrv::support::RuntimeABIParameterOwnership;
  using tianchenrv::support::RuntimeABIParameterRole;

  TCRVEmitCLowerableRoute route(
      "tcrv-export-rvv-scalar-i32-vadd-dispatch-c",
      "tcrv-exec-dispatch-control-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addABIValueMapping(
      RuntimeABIParameter("lhs", "const int32_t *",
                          RuntimeABIParameterRole::LHSInputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "lhs");
  route.addABIValueMapping(
      RuntimeABIParameter("rhs", "const int32_t *",
                          RuntimeABIParameterRole::RHSInputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "rhs");
  route.addABIValueMapping(
      RuntimeABIParameter("out", "int32_t *",
                          RuntimeABIParameterRole::OutputBuffer,
                          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "out");
  route.addABIValueMapping(
      RuntimeABIParameter(
          "n", "size_t", RuntimeABIParameterRole::RuntimeElementCount,
          RuntimeABIParameterOwnership::TargetExportABIOwned),
      "n");
  if (includeDispatchGuard)
    route.addABIValueMapping(
        RuntimeABIParameter(
            "rvv_available", "int",
            RuntimeABIParameterRole::DispatchAvailabilityGuard,
            RuntimeABIParameterOwnership::TargetExportABIOwned),
        "rvv_available");

  TCRVEmitCCallOpaqueStep rvvCall;
  rvvCall.sourceOp = {"tcrv.exec.case", firstRole.str()};
  rvvCall.callee = "tcrv_rvv_i32_vadd_microkernel_kernel_rvv";
  rvvCall.operands.push_back({"lhs", "const int32_t *"});
  rvvCall.operands.push_back({"rhs", "const int32_t *"});
  rvvCall.operands.push_back({"out", "int32_t *"});
  rvvCall.operands.push_back({"n", "size_t"});
  route.addCallOpaqueStep(std::move(rvvCall));

  TCRVEmitCCallOpaqueStep scalarCall;
  scalarCall.sourceOp = {"tcrv.exec.fallback", "dispatch-fallback-call"};
  scalarCall.callee = "tcrv_scalar_i32_vadd_microkernel_kernel_scalar";
  scalarCall.operands.push_back({"lhs", "const int32_t *"});
  scalarCall.operands.push_back({"rhs", "const int32_t *"});
  scalarCall.operands.push_back({"out", "int32_t *"});
  scalarCall.operands.push_back({"n", "size_t"});
  route.addCallOpaqueStep(std::move(scalarCall));
  return route;
}

TCRVEmitCMaterializationOptions
makeMaterializerOptions(llvm::StringRef functionName) {
  TCRVEmitCMaterializationOptions options;
  options.functionName = functionName.str();
  options.implicitValueNames.push_back("offset");
  return options;
}

int expectRouteMaterializes(mlir::MLIRContext &context,
                            const TCRVEmitCLowerableRoute &route,
                            llvm::StringRef functionName,
                            llvm::StringRef arithmeticCallee,
                            llvm::StringRef arithmeticSourceOp) {
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRoute(context, route,
                                         makeMaterializerOptions(functionName));
  if (!module)
    return fail(llvm::Twine("expected route to materialize: ") +
                llvm::toString(module.takeError()));

  unsigned includeCount = 0;
  unsigned funcCount = 0;
  unsigned callOpaqueCount = 0;
  bool sawArithmeticCall = false;
  bool sawExpectedFunctionABI = false;
  module->get().walk([&](mlir::emitc::IncludeOp) { ++includeCount; });
  module->get().walk([&](mlir::emitc::FuncOp func) {
    ++funcCount;
    if (func.getSymName() != functionName)
      return;
    sawExpectedFunctionABI = func.getFunctionType().getNumInputs() == 4;
  });
  module->get().walk([&](mlir::emitc::CallOpaqueOp call) {
    ++callOpaqueCount;
    if (call.getCallee() == arithmeticCallee)
      sawArithmeticCall = true;
  });

  if (int result = expect(includeCount >= 1,
                          "materialized route contains emitc.include ops"))
    return result;
  if (int result = expect(funcCount == 1,
                          "materialized route contains one emitc.func"))
    return result;
  if (int result =
          expect(sawExpectedFunctionABI,
                 "materialized EmitC function has four runtime ABI inputs"))
    return result;
  if (int result =
          expect(callOpaqueCount == route.getCallOpaqueSteps().size(),
                 "materialized route contains one emitc.call_opaque per step"))
    return result;
  if (int result =
          expect(sawArithmeticCall,
                 "materialized route contains arithmetic emitc.call_opaque"))
    return result;

  std::string ir;
  llvm::raw_string_ostream os(ir);
  module->get().print(os);
  os.flush();
  std::string expectedSource =
      (llvm::Twine("tcrv_emitc.source_op=") + arithmeticSourceOp).str();
  if (int result =
          expect(llvm::StringRef(ir).contains(expectedSource),
                 "materialized route records source-op provenance"))
    return result;
  if (int result = expect(llvm::StringRef(ir).contains(
                              "op_interface=TCRVEmitCLowerableOpInterface"),
                          "materialized route records op-interface provenance"))
    return result;

  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> reparsed =
      mlir::parseSourceString<mlir::ModuleOp>(ir, parserConfig);
  if (int result = expect(static_cast<bool>(reparsed),
                          "materialized EmitC IR is parseable"))
    return result;
  return 0;
}

int expectMaterializationFails(const TCRVEmitCLowerableRoute &route,
                               llvm::StringRef expectedDiagnostic) {
  mlir::MLIRContext context;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRoute(
          context, route, makeMaterializerOptions("tcrv_emitc_bad_route"));
  if (module)
    return fail("expected route materialization to fail closed");
  std::string message = llvm::toString(module.takeError());
  return expect(llvm::StringRef(message).contains(expectedDiagnostic),
                llvm::Twine("diagnostic contains '") + expectedDiagnostic +
                    "'");
}

int expectRouteRendersCSource(const TCRVEmitCLowerableRoute &route,
                              llvm::StringRef functionName,
                              llvm::StringRef arithmeticCallee,
                              llvm::StringRef resultName) {
  TCRVEmitCLegacyDiagnosticSourceRenderOptions options;
  options.functionName = functionName.str();
  options.loopIndexName = "offset";
  std::string source;
  llvm::raw_string_ostream os(source);
  if (llvm::Error error =
          renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction(route, os,
                                                                   options))
    return fail(llvm::Twine("expected route to render C source: ") +
                llvm::toString(std::move(error)));
  os.flush();

  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("void ") + functionName + "(").str()),
                 "route-rendered source contains function signature"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains("while (offset < n)"),
                 "route-rendered source uses runtime element-count ABI bound"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("size_t vl = __riscv_vsetvl_e32m1("
                                  "n - offset)")
                          .str())),
                 "route-rendered source emits setvl from call_opaque step"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(arithmeticCallee),
                 "route-rendered source emits arithmetic callee from route"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("vint32m1_t ") + resultName + " =").str()),
                 "route-rendered source emits route result binding"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains("offset += vl;"),
                 "route-rendered source advances by route VL result"))
    return result;
  return 0;
}

int expectRouteEmitsCppSourceAuthority(const TCRVEmitCLowerableRoute &route,
                                       llvm::StringRef functionName,
                                       llvm::StringRef arithmeticCallee,
                                       llvm::StringRef arithmeticSourceOp,
                                       llvm::StringRef loopIndexName =
                                           "offset") {
  TCRVEmitCSourceAuthorityOptions options;
  options.functionName = functionName.str();
  options.loopIndexName = loopIndexName.str();

  mlir::MLIRContext sourceContext;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> sourceModule =
      materializeTCRVEmitCLowerableRouteSourceAuthority(sourceContext, route,
                                                        options);
  if (!sourceModule)
    return fail(llvm::Twine("expected source-authority route to materialize: ") +
                llvm::toString(sourceModule.takeError()));

  unsigned funcCount = 0;
  unsigned ifCount = 0;
  unsigned callOpaqueCount = 0;
  sourceModule->get().walk([&](mlir::emitc::FuncOp) { ++funcCount; });
  sourceModule->get().walk([&](mlir::emitc::IfOp) { ++ifCount; });
  sourceModule->get().walk(
      [&](mlir::emitc::CallOpaqueOp) { ++callOpaqueCount; });
  if (int result = expect(funcCount == 2,
                          "source-authority module contains helper and public "
                          "emitc.func ops"))
    return result;
  if (int result = expect(ifCount == 1,
                          "source-authority module models runtime VL loop "
                          "control with EmitC control flow"))
    return result;
  if (int result =
          expect(callOpaqueCount == route.getCallOpaqueSteps().size(),
                 "source-authority module keeps one emitc.call_opaque per "
                 "route step"))
    return result;

  std::string source;
  llvm::raw_string_ostream os(source);
  if (llvm::Error error =
          emitTCRVEmitCLowerableRouteAsCppSource(route, os, options))
    return fail(llvm::Twine("expected route to emit MLIR Cpp source: ") +
                llvm::toString(std::move(error)));
  os.flush();

  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_emitc.source_authority=mlir_emitc_cpp_emitter"),
                 "source-authority output records MLIR Cpp emitter authority"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("static void ") + functionName +
                      "__tcrv_emitc_body")
                         .str()),
                 "source-authority output contains static EmitC helper"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("void ") + functionName + "(").str()),
                 "source-authority output contains public wrapper function"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains("if ("),
                 "source-authority output emits MLIR-modeled control flow"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(arithmeticCallee),
                 "source-authority output emits arithmetic callee from route"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("tcrv_emitc.source_op=") +
                      arithmeticSourceOp)
                         .str()),
                 "source-authority output preserves source-op provenance"))
    return result;
  return 0;
}

int expectRouteSourceAuthorityFails(const TCRVEmitCLowerableRoute &route,
                                    llvm::StringRef expectedDiagnostic,
                                    llvm::StringRef loopIndexName = "offset") {
  TCRVEmitCSourceAuthorityOptions options;
  options.functionName = "tcrv_emitc_bad_source_authority";
  options.loopIndexName = loopIndexName.str();
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::Error error = emitTCRVEmitCLowerableRouteAsCppSource(route, os,
                                                             options);
  if (!error)
    return fail("expected MLIR Cpp source authority to fail closed");
  os.flush();
  if (int result =
          expect(source.empty(),
                 "failed source-authority emission does not emit partial C "
                 "source"))
    return result;
  std::string message = llvm::toString(std::move(error));
  return expect(llvm::StringRef(message).contains(expectedDiagnostic),
                llvm::Twine("source-authority diagnostic contains '") +
                    expectedDiagnostic + "'");
}

int expectDispatchRouteEmitsCppSourceAuthority(
    const TCRVEmitCLowerableRoute &route, llvm::StringRef functionName) {
  TCRVEmitCSourceAuthorityOptions options;
  options.functionName = functionName.str();
  options.dispatchGuardValueName = "rvv_available";
  options.requireInterfaceBackedCompute = false;

  mlir::MLIRContext sourceContext;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> sourceModule =
      materializeTCRVEmitCLowerableRouteSourceAuthority(sourceContext, route,
                                                        options);
  if (!sourceModule)
    return fail(llvm::Twine("expected dispatch route to materialize: ") +
                llvm::toString(sourceModule.takeError()));

  unsigned funcCount = 0;
  unsigned ifCount = 0;
  unsigned callOpaqueCount = 0;
  sourceModule->get().walk([&](mlir::emitc::FuncOp) { ++funcCount; });
  sourceModule->get().walk([&](mlir::emitc::IfOp) { ++ifCount; });
  sourceModule->get().walk(
      [&](mlir::emitc::CallOpaqueOp) { ++callOpaqueCount; });
  if (int result = expect(funcCount == 1,
                          "dispatch source-authority module contains one "
                          "public emitc.func"))
    return result;
  if (int result = expect(ifCount == 1,
                          "dispatch source-authority module models runtime "
                          "guard control with emitc.if"))
    return result;
  if (int result =
          expect(callOpaqueCount == 2,
                 "dispatch source-authority module contains selected case and "
                 "fallback call_opaque ops"))
    return result;

  std::string source;
  llvm::raw_string_ostream os(source);
  if (llvm::Error error =
          emitTCRVEmitCLowerableRouteAsCppSource(route, os, options))
    return fail(llvm::Twine("expected dispatch route to emit Cpp source: ") +
                llvm::toString(std::move(error)));
  os.flush();

  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_emitc.source_authority=mlir_emitc_cpp_emitter"),
                 "dispatch source-authority output records MLIR Cpp emitter "
                 "authority"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch"),
                 "dispatch source-authority output records dispatch control "
                 "source"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("void ") + functionName + "(").str()),
                 "dispatch source-authority output contains public wrapper"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains("rvv_available"),
                 "dispatch source-authority output records guard name"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_rvv_i32_vadd_microkernel_kernel_rvv("),
                 "dispatch source-authority output calls selected RVV "
                 "component"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains("return;"),
                 "dispatch source-authority output returns after selected "
                 "case call"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_scalar_i32_vadd_microkernel_kernel_scalar("),
                 "dispatch source-authority output calls selected scalar "
                 "fallback"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_emitc.source_op=tcrv.exec.case "
                     "role=dispatch-case-call"),
                 "dispatch source-authority output preserves case "
                 "provenance"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_emitc.source_op=tcrv.exec.fallback "
                     "role=dispatch-fallback-call"),
                 "dispatch source-authority output preserves fallback "
                 "provenance"))
    return result;
  return 0;
}

int expectDispatchRouteSourceAuthorityFails(
    const TCRVEmitCLowerableRoute &route,
    llvm::StringRef expectedDiagnostic) {
  TCRVEmitCSourceAuthorityOptions options;
  options.functionName = "tcrv_dispatch_bad_source_authority";
  options.dispatchGuardValueName = "rvv_available";
  options.requireInterfaceBackedCompute = false;
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::Error error =
      emitTCRVEmitCLowerableRouteAsCppSource(route, os, options);
  if (!error)
    return fail("expected dispatch MLIR Cpp source authority to fail closed");
  os.flush();
  if (int result =
          expect(source.empty(),
                 "failed dispatch source-authority emission does not emit "
                 "partial C source"))
    return result;
  std::string message = llvm::toString(std::move(error));
  return expect(llvm::StringRef(message).contains(expectedDiagnostic),
                llvm::Twine("dispatch source-authority diagnostic contains '") +
                    expectedDiagnostic + "'");
}

int expectScalarRouteRendersCSource(const TCRVEmitCLowerableRoute &route,
                                    llvm::StringRef functionName) {
  TCRVEmitCLegacyDiagnosticSourceRenderOptions options;
  options.functionName = functionName.str();
  options.loopIndexName = "index";
  std::string source;
  llvm::raw_string_ostream os(source);
  if (llvm::Error error =
          renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction(route, os,
                                                                   options))
    return fail(llvm::Twine("expected scalar route to render C source: ") +
                llvm::toString(std::move(error)));
  os.flush();

  if (int result =
          expect(llvm::StringRef(source).contains(
                     (llvm::Twine("void ") + functionName + "(").str()),
                 "scalar route-rendered source contains function signature"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "for (size_t index = 0; index < n; ++index)"),
                 "scalar route-rendered source uses runtime element-count "
                 "loop"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "int32_t sum = tcrv_scalar_i32_add(lhs[index], "
                     "rhs[index]);"),
                 "scalar route-rendered source emits compute call from route"))
    return result;
  if (int result =
          expect(llvm::StringRef(source).contains(
                     "tcrv_scalar_i32_store(&out[index], sum);"),
                 "scalar route-rendered source emits store call from route"))
    return result;
  if (int result = expect(!llvm::StringRef(source).contains("offset += "),
                          "scalar route-rendered source does not use a VL "
                          "increment"))
    return result;
  return 0;
}

int expectRouteRenderingFails(const TCRVEmitCLowerableRoute &route,
                              llvm::StringRef expectedDiagnostic) {
  TCRVEmitCLegacyDiagnosticSourceRenderOptions options;
  options.functionName = "tcrv_emitc_bad_render";
  std::string source;
  llvm::raw_string_ostream os(source);
  llvm::Error error =
      renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction(route, os,
                                                               options);
  if (!error)
    return fail("expected route C source rendering to fail closed");
  os.flush();
  if (int result =
          expect(source.empty(),
                 "failed route rendering does not emit partial C source"))
    return result;
  std::string message = llvm::toString(std::move(error));
  return expect(llvm::StringRef(message).contains(expectedDiagnostic),
                llvm::Twine("route renderer diagnostic contains '") +
                    expectedDiagnostic + "'");
}

} // namespace

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<tianchenrv::tcrv::rvv::TCRVRVVDialect>();

  mlir::Location loc = mlir::UnknownLoc::get(&context);
  mlir::Block block;
  mlir::Type vectorType =
      tianchenrv::tcrv::rvv::I32M1VectorType::get(&context);
  mlir::Type vlType = tianchenrv::tcrv::rvv::VLType::get(&context);
  mlir::Value lhs = block.addArgument(vectorType, loc);
  mlir::Value rhs = block.addArgument(vectorType, loc);
  mlir::Value vl = block.addArgument(vlType, loc);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToEnd(&block);

  mlir::Operation *addOp =
      createRVVArithmeticOp(builder, loc, "tcrv_rvv.i32_add", lhs, rhs, vl,
                            vectorType);
  mlir::Operation *subOp =
      createRVVArithmeticOp(builder, loc, "tcrv_rvv.i32_sub", lhs, rhs, vl,
                            vectorType);
  mlir::Operation *mulOp =
      createRVVArithmeticOp(builder, loc, "tcrv_rvv.i32_mul", lhs, rhs, vl,
                            vectorType);

  if (int result =
          expectRVVArithmeticOpInterface(addOp, "tcrv_rvv.i32_add"))
    return result;
  if (int result =
          expectRVVArithmeticOpInterface(subOp, "tcrv_rvv.i32_sub"))
    return result;
  if (int result =
          expectRVVArithmeticOpInterface(mulOp, "tcrv_rvv.i32_mul"))
    return result;

  auto addLowerable = llvm::cast<TCRVEmitCLowerableOpInterface>(addOp);
  GeneratedRVVArithmeticLowerable valid(addLowerable,
                                        "__riscv_vadd_vv_i32m1", "sum_vec");
  llvm::Expected<TCRVEmitCLowerableRoute> route =
      buildTCRVEmitCLowerableRoute(valid);
  if (!route)
    return fail(llvm::Twine("expected valid lowerable route: ") +
                llvm::toString(route.takeError()));

  if (int result =
          expect(route->getHeaders().size() == 3,
                 "common route preserves header requirements"))
    return result;
  if (int result =
          expect(route->getTypeMappings().size() == 2,
                 "common route preserves C type mappings"))
    return result;
  if (int result =
          expect(route->getABIMappings().size() == 4,
                 "common route preserves ABI operand mappings"))
    return result;
  if (int result =
          expect(route->getCallOpaqueSteps().size() == 5,
                 "common route preserves emitc.call_opaque steps"))
    return result;
  if (int result = expect(route->getCallOpaqueSteps()[3].sourceOp.opName ==
                              "tcrv_rvv.i32_add",
                          "common route records typed source op provenance"))
    return result;
  if (int result =
          expect(route->getCallOpaqueSteps()[3].sourceOp.opInterface ==
                     kEmitCLowerableOpInterfaceName.str(),
                 "common route records generated op-interface provenance"))
    return result;
  if (int result =
          expect(route->getCallOpaqueSteps()[3].callee ==
                     "__riscv_vadd_vv_i32m1",
                 "common route records intrinsic call mapping"))
    return result;
  if (int result = expectRouteMaterializes(
          context, *route, "tcrv_emitc_test_add", "__riscv_vadd_vv_i32m1",
          "tcrv_rvv.i32_add"))
    return result;
  if (int result = expectRouteRendersCSource(
          *route, "tcrv_emitc_test_add", "__riscv_vadd_vv_i32m1", "sum_vec"))
    return result;
  if (int result = expectRouteEmitsCppSourceAuthority(
          *route, "tcrv_emitc_test_add", "__riscv_vadd_vv_i32m1",
          "tcrv_rvv.i32_add"))
    return result;

  auto subLowerable = llvm::cast<TCRVEmitCLowerableOpInterface>(subOp);
  GeneratedRVVArithmeticLowerable validSub(
      subLowerable, "__riscv_vsub_vv_i32m1", "difference_vec");
  llvm::Expected<TCRVEmitCLowerableRoute> subRoute =
      buildTCRVEmitCLowerableRoute(validSub);
  if (!subRoute)
    return fail(llvm::Twine("expected valid sub lowerable route: ") +
                llvm::toString(subRoute.takeError()));
  if (int result = expectRouteMaterializes(
          context, *subRoute, "tcrv_emitc_test_sub", "__riscv_vsub_vv_i32m1",
          "tcrv_rvv.i32_sub"))
    return result;
  if (int result =
          expectRouteRendersCSource(*subRoute, "tcrv_emitc_test_sub",
                                    "__riscv_vsub_vv_i32m1",
                                    "difference_vec"))
    return result;
  if (int result = expectRouteEmitsCppSourceAuthority(
          *subRoute, "tcrv_emitc_test_sub", "__riscv_vsub_vv_i32m1",
          "tcrv_rvv.i32_sub"))
    return result;

  auto mulLowerable = llvm::cast<TCRVEmitCLowerableOpInterface>(mulOp);
  GeneratedRVVArithmeticLowerable validMul(
      mulLowerable, "__riscv_vmul_vv_i32m1", "product_vec");
  llvm::Expected<TCRVEmitCLowerableRoute> mulRoute =
      buildTCRVEmitCLowerableRoute(validMul);
  if (!mulRoute)
    return fail(llvm::Twine("expected valid mul lowerable route: ") +
                llvm::toString(mulRoute.takeError()));
  if (int result = expectRouteMaterializes(
          context, *mulRoute, "tcrv_emitc_test_mul", "__riscv_vmul_vv_i32m1",
          "tcrv_rvv.i32_mul"))
    return result;
  if (int result =
          expectRouteRendersCSource(*mulRoute, "tcrv_emitc_test_mul",
                                    "__riscv_vmul_vv_i32m1", "product_vec"))
    return result;
  if (int result = expectRouteEmitsCppSourceAuthority(
          *mulRoute, "tcrv_emitc_test_mul", "__riscv_vmul_vv_i32m1",
          "tcrv_rvv.i32_mul"))
    return result;

  TCRVEmitCLowerableRoute scalarRoute = makeScalarElementLoopRoute();
  if (int result =
          expect(scalarRoute.getCallOpaqueSteps().size() == 2,
                 "scalar route preserves compute and store call_opaque steps"))
    return result;
  if (llvm::Error error = verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
          scalarRoute, "tcrv_emitc_scalar_test_add", {"index"}))
    return fail(llvm::Twine("expected scalar route to materialize: ") +
                llvm::toString(std::move(error)));
  if (int result =
          expectScalarRouteRendersCSource(scalarRoute,
                                          "tcrv_emitc_scalar_test_add"))
    return result;
  if (int result = expectRouteEmitsCppSourceAuthority(
          scalarRoute, "tcrv_emitc_scalar_test_add", "tcrv_scalar_i32_add",
          "tcrv_scalar.i32_vadd_microkernel", "index"))
    return result;

  TCRVEmitCLowerableRoute dispatchRoute = makeDispatchControlRoute();
  if (int result =
          expect(dispatchRoute.getCallOpaqueSteps().size() == 2,
                 "dispatch route preserves case and fallback call steps"))
    return result;
  if (int result = expectDispatchRouteEmitsCppSourceAuthority(
          dispatchRoute, "tcrv_dispatch_i32_vadd_kernel"))
    return result;

  if (int result = expectMaterializationFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add", "compute",
                                       "__riscv_vadd_vv_i32m1", std::nullopt),
          "compute step"))
    return result;
  if (int result = expectMaterializationFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add", "compute",
                                       "__riscv_vadd_vv_i32m1", "sum_vec",
                                       "missing_vec"),
          "unknown value name"))
    return result;
  if (int result = expectMaterializationFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add", "compute",
                                       "__riscv_vadd_vv_i32m1", "n"),
          "duplicate call_opaque result name"))
    return result;
  if (int result = expectMaterializationFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add;bad", "compute",
                                       "__riscv_vadd_vv_i32m1", "sum_vec"),
          "unsafe text"))
    return result;
  if (int result = expectMaterializationFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add", "compute",
                                       "__riscv_vadd_vv_i32m1", "sum_vec",
                                       "lhs", "vl", "lhs_value"),
          "ABI value mapping"))
    return result;
  if (int result = expectRouteRenderingFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add", "compute",
                                       "__riscv_vadd_vv_i32m1", "sum_vec",
                                       "lhs", "vl", "lhs", ""),
          "requires generated op-interface provenance"))
    return result;
  if (int result =
          expectRouteRenderingFails(makeRouteWithNonAVLFirstStep(),
                                    "non-compute call_opaque result"))
    return result;
  if (int result = expectRouteRenderingFails(
          makeScalarElementLoopRoute(/*includeRuntimeElementCount=*/false),
          "requires exactly one runtime-element-count ABI mapping"))
    return result;
  if (int result = expectRouteSourceAuthorityFails(
          makeScalarElementLoopRoute(/*includeRuntimeElementCount=*/false),
          "requires exactly one runtime-element-count ABI mapping", "index"))
    return result;
  if (int result = expectRouteSourceAuthorityFails(
          makeScalarElementLoopRoute(/*includeRuntimeElementCount=*/true,
                                     /*includeInterfaceProvenance=*/false),
          "requires generated op-interface provenance", "index"))
    return result;
  if (int result = expectDispatchRouteSourceAuthorityFails(
          makeDispatchControlRoute(/*includeDispatchGuard=*/false),
          "requires dispatch guard ABI value"))
    return result;
  if (int result = expectDispatchRouteSourceAuthorityFails(
          makeDispatchControlRoute(/*includeDispatchGuard=*/true,
                                   /*firstRole=*/"compute"),
          "source role 'dispatch-case-call'"))
    return result;
  if (int result = expectRouteRenderingFails(
          makeMinimalMaterializerRoute("tcrv_rvv.i32_add", "compute",
                                       "__riscv_vadd_vv_i32m1", "sum_vec",
                                       "missing_vec"),
          "unknown value name"))
    return result;

  InvalidMissingCalleeLowerable invalid;
  llvm::Expected<TCRVEmitCLowerableRoute> badRoute =
      buildTCRVEmitCLowerableRoute(invalid);
  if (badRoute)
    return fail("expected missing call_opaque callee to fail closed");
  std::string message = llvm::toString(badRoute.takeError());
  if (int result = expect(llvm::StringRef(message).contains(
                              "call_opaque callee"),
                          "missing callee diagnostic names call_opaque"))
    return result;

  return 0;
}
