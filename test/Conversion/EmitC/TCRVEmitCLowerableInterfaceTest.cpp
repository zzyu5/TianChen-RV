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
                             llvm::StringRef lhsValueName = "lhs") {
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
                         kEmitCLowerableOpInterfaceName.str()};
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
