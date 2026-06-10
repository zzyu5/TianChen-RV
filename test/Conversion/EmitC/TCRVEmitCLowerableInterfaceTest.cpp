#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"
#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

using namespace tianchenrv::conversion::emitc;
namespace support = tianchenrv::support;

namespace {

constexpr llvm::StringLiteral kCommonInterfaceName(
    "TCRVEmitCLowerableInterfaceTest");

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

void addStandardABI(TCRVEmitCLowerableRoute &route,
                    llvm::StringRef lhsValueName = "lhs") {
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "lhs", "const int32_t *",
          support::RuntimeABIParameterRole::LHSInputBuffer),
      lhsValueName);
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "rhs", "const int32_t *",
          support::RuntimeABIParameterRole::RHSInputBuffer),
      "rhs");
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "out", "int32_t *",
          support::RuntimeABIParameterRole::OutputBuffer),
      "out");
  route.addABIValueMapping(
      support::makeTargetExportABIParameter(
          "n", "size_t",
          support::RuntimeABIParameterRole::RuntimeElementCount),
      "n");
}

TCRVEmitCLowerableRoute makeGenericRoute(llvm::StringRef routeID,
                                         llvm::StringRef lhsOperand = "lhs",
                                         llvm::StringRef resultName = "sum") {
  TCRVEmitCLowerableRoute route(
      routeID, "extension-family-ops-to-emitc-call-opaque");
  route.addHeader("stddef.h");
  route.addHeader("stdint.h");
  route.addTypeMapping("!tcrv_common.ptr_i32", "int32_t *");
  route.addTypeMapping("!tcrv_common.i32", "int32_t");
  addStandardABI(route);

  TCRVEmitCCallOpaqueStep compute;
  compute.sourceOp = {"test.common.compute", "compute",
                      kCommonInterfaceName.str()};
  compute.callee = "tcrv_common_add_i32";
  compute.operands.push_back({lhsOperand.str(), "const int32_t *"});
  compute.operands.push_back({"rhs", "const int32_t *"});
  compute.operands.push_back({"n", "size_t"});
  compute.result = TCRVEmitCCallOpaqueResult{resultName.str(), "int32_t"};
  route.addCallOpaqueStep(std::move(compute));
  return route;
}

class GenericLowerable final : public TCRVEmitCLowerableInterface {
public:
  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    TCRVEmitCLowerableRoute route =
        makeGenericRoute("test-common-emitc-route");

    TCRVEmitCCallOpaqueStep store;
    store.sourceOp = {"test.common.store", "writeback",
                      kCommonInterfaceName.str()};
    store.callee = "tcrv_common_store_i32";
    store.operands.push_back({"out", "int32_t *"});
    store.operands.push_back({"sum", "int32_t"});
    store.operands.push_back({"n", "size_t"});
    route.addCallOpaqueStep(std::move(store));
    return route;
  }
};

class InvalidMissingCalleeLowerable final : public TCRVEmitCLowerableInterface {
public:
  llvm::Expected<TCRVEmitCLowerableRoute>
  buildEmitCLowerableRoute() const override {
    TCRVEmitCLowerableRoute route(
        "bad-route", "extension-family-ops-to-emitc-call-opaque");
    route.addHeader("stdint.h");

    TCRVEmitCCallOpaqueStep compute;
    compute.sourceOp = {"test.common.compute", "compute",
                        kCommonInterfaceName.str()};
    compute.result = TCRVEmitCCallOpaqueResult{"sum", "int32_t"};
    route.addCallOpaqueStep(std::move(compute));
    return route;
  }
};

TCRVEmitCMaterializationOptions
makeMaterializerOptions(llvm::StringRef functionName) {
  TCRVEmitCMaterializationOptions options;
  options.functionName = functionName.str();
  return options;
}

int expectRouteMaterializes(mlir::MLIRContext &context,
                            const TCRVEmitCLowerableRoute &route,
                            llvm::StringRef functionName,
                            llvm::StringRef expectedSourceOp) {
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRoute(context, route,
                                         makeMaterializerOptions(functionName));
  if (!module)
    return fail(llvm::Twine("expected route to materialize: ") +
                llvm::toString(module.takeError()));

  unsigned includeCount = 0;
  unsigned funcCount = 0;
  unsigned callOpaqueCount = 0;
  bool sawExpectedFunctionABI = false;
  module->get().walk([&](mlir::emitc::IncludeOp) { ++includeCount; });
  module->get().walk([&](mlir::emitc::FuncOp func) {
    ++funcCount;
    if (func.getSymName() != functionName)
      return;
    sawExpectedFunctionABI =
        func.getFunctionType().getNumInputs() == route.getABIMappings().size();
  });
  module->get().walk(
      [&](mlir::emitc::CallOpaqueOp) { ++callOpaqueCount; });

  if (int result =
          expect(includeCount == route.getHeaders().size(),
                 "materialized route contains route-declared headers"))
    return result;
  if (int result = expect(funcCount == 1,
                          "materialized route contains one emitc.func"))
    return result;
  if (int result =
          expect(sawExpectedFunctionABI,
                 "materialized EmitC function mirrors route ABI inputs"))
    return result;
  if (int result =
          expect(callOpaqueCount == route.getCallOpaqueSteps().size(),
                 "materialized route contains one emitc.call_opaque per step"))
    return result;

  std::string ir;
  llvm::raw_string_ostream os(ir);
  module->get().print(os);
  os.flush();
  std::string expectedSource =
      (llvm::Twine("tcrv_emitc.source_op=") + expectedSourceOp).str();
  if (int result =
          expect(llvm::StringRef(ir).contains(expectedSource),
                 "materialized route records generic source-op provenance"))
    return result;
  if (int result = expect(llvm::StringRef(ir).contains(
                              "op_interface=TCRVEmitCLowerableInterfaceTest"),
                          "materialized route records generic interface "
                          "provenance"))
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

int expectLowerableRouteBuildFails(
    const TCRVEmitCLowerableInterface &lowerable,
    llvm::StringRef expectedDiagnostic) {
  llvm::Expected<TCRVEmitCLowerableRoute> route =
      buildTCRVEmitCLowerableRoute(lowerable);
  if (route)
    return fail("expected lowerable route verification to fail closed");
  std::string message = llvm::toString(route.takeError());
  return expect(llvm::StringRef(message).contains(expectedDiagnostic),
                llvm::Twine("route diagnostic contains '") +
                    expectedDiagnostic + "'");
}

TCRVEmitCLowerableRoute makeStructuredLoopRoute() {
  TCRVEmitCLowerableRoute route(
      "test-structured-loop-route", "extension-family-ops-to-emitc-loop");
  route.addHeader("stddef.h");
  addStandardABI(route);
  route.addSourceOpProvenance(
      {"test.common.scope", "scope", kCommonInterfaceName.str()});

  TCRVEmitCCallOpaqueStep fullChunkVL;
  fullChunkVL.sourceOp = {"test.common.configure", "configure",
                          kCommonInterfaceName.str()};
  fullChunkVL.callee = "test_setvl";
  fullChunkVL.operands.push_back({"n", "size_t"});
  fullChunkVL.result = TCRVEmitCCallOpaqueResult{"full_chunk_vl", "size_t"};
  route.addCallOpaqueStep(std::move(fullChunkVL));

  TCRVEmitCForLoop loop;
  loop.inductionVarName = "offset";
  loop.lowerBound = {"0", "size_t"};
  loop.upperBound = {"n", "size_t"};
  loop.step = {"full_chunk_vl", "size_t"};

  TCRVEmitCCallOpaqueStep chunkVL;
  chunkVL.sourceOp = {"test.common.configure", "configure",
                      kCommonInterfaceName.str()};
  chunkVL.callee = "test_setvl";
  chunkVL.operands.push_back({"n - offset", "size_t"});
  chunkVL.result = TCRVEmitCCallOpaqueResult{"vl", "size_t"};
  loop.bodySteps.push_back(std::move(chunkVL));

  TCRVEmitCCallOpaqueStep compute;
  compute.sourceOp = {"test.common.compute", "compute",
                      kCommonInterfaceName.str()};
  compute.callee = "test_compute_chunk";
  compute.operands.push_back({"lhs + offset", "const int32_t *"});
  compute.operands.push_back({"vl", "size_t"});
  compute.result = TCRVEmitCCallOpaqueResult{"chunk_result", "int32_t"};
  loop.bodySteps.push_back(std::move(compute));

  route.addForLoop(std::move(loop));
  return route;
}

int expectStructuredLoopRouteMaterializes(mlir::MLIRContext &context) {
  TCRVEmitCLowerableRoute route = makeStructuredLoopRoute();
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRoute(
          context, route, makeMaterializerOptions("tcrv_emitc_loop_route"));
  if (!module)
    return fail(llvm::Twine("expected structured loop route to materialize: ") +
                llvm::toString(module.takeError()));

  unsigned forCount = 0;
  unsigned callOpaqueCount = 0;
  module->get().walk([&](mlir::emitc::ForOp) { ++forCount; });
  module->get().walk(
      [&](mlir::emitc::CallOpaqueOp) { ++callOpaqueCount; });
  if (int result =
          expect(forCount == 1,
                 "structured route materializes one emitc.for loop"))
    return result;
  if (int result =
          expect(callOpaqueCount == 3,
                 "structured route materializes top-level and loop body calls"))
    return result;

  std::string ir;
  llvm::raw_string_ostream os(ir);
  module->get().print(os);
  os.flush();
  if (int result =
          expect(llvm::StringRef(ir).contains(" sub "),
                 "structured route materializes remaining AVL subtraction"))
    return result;
  if (int result =
          expect(llvm::StringRef(ir).contains(" add "),
                 "structured route materializes pointer/index advancement"))
    return result;
  return 0;
}

TCRVEmitCLowerableRoute makePostLoopSubscriptStoreRoute(
    llvm::StringRef targetName = "out[0]") {
  TCRVEmitCLowerableRoute route =
      makeGenericRoute("test-post-loop-subscript-store-route");

  TCRVEmitCAssignStep store;
  store.sourceOp = {"test.common.store", "store", kCommonInterfaceName.str()};
  store.targetName = targetName.str();
  store.value = {"sum", "int32_t"};
  route.addPostLoopAssignment(std::move(store));
  return route;
}

int expectPostLoopSubscriptStoreRouteMaterializes(mlir::MLIRContext &context) {
  TCRVEmitCLowerableRoute route = makePostLoopSubscriptStoreRoute();
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      materializeTCRVEmitCLowerableRoute(
          context, route,
          makeMaterializerOptions("tcrv_emitc_post_loop_store_route"));
  if (!module)
    return fail(llvm::Twine("expected post-loop assignment route to "
                            "materialize: ") +
                llvm::toString(module.takeError()));

  unsigned assignCount = 0;
  unsigned subscriptCount = 0;
  module->get().walk([&](mlir::emitc::AssignOp) { ++assignCount; });
  module->get().walk([&](mlir::emitc::SubscriptOp) { ++subscriptCount; });
  if (int result =
          expect(assignCount == 1,
                 "post-loop subscript store materializes one assign op"))
    return result;
  if (int result =
          expect(subscriptCount == 1,
                 "post-loop subscript store materializes one subscript op"))
    return result;

  std::string ir;
  llvm::raw_string_ostream os(ir);
  module->get().print(os);
  os.flush();
  if (int result = expect(llvm::StringRef(ir).contains(
                              "tcrv_emitc.assign target=out[0] "
                              "source_op=test.common.store"),
                          "post-loop assignment preserves source provenance"))
    return result;
  return 0;
}

} // namespace

int main() {
  GenericLowerable valid;
  llvm::Expected<TCRVEmitCLowerableRoute> route =
      buildTCRVEmitCLowerableRoute(valid);
  if (!route)
    return fail(llvm::Twine("expected valid lowerable route: ") +
                llvm::toString(route.takeError()));

  if (int result =
          expect(route->getHeaders().size() == 2,
                 "common route preserves generic header requirements"))
    return result;
  if (int result =
          expect(route->getTypeMappings().size() == 2,
                 "common route preserves generic C type mappings"))
    return result;
  if (int result =
          expect(route->getABIMappings().size() == 4,
                 "common route preserves ABI operand mappings"))
    return result;
  if (int result =
          expect(route->getCallOpaqueSteps().size() == 2,
                 "common route preserves emitc.call_opaque steps"))
    return result;
  if (int result = expect(route->getCallOpaqueSteps()[0].sourceOp.opName ==
                              "test.common.compute",
                          "common route records generic source op provenance"))
    return result;
  if (int result =
          expect(route->getCallOpaqueSteps()[0].callee ==
                     "tcrv_common_add_i32",
                 "common route records generic call mapping"))
    return result;

  mlir::MLIRContext context;
  if (int result = expectRouteMaterializes(
          context, *route, "tcrv_emitc_test_common_route",
          "test.common.compute"))
    return result;

  if (llvm::Error error = verifyTCRVEmitCLowerableRouteMaterializesToEmitC(
          *route, "tcrv_emitc_verified_common_route"))
    return fail(llvm::Twine("expected materialization verifier to pass: ") +
                llvm::toString(std::move(error)));

  if (int result = expectStructuredLoopRouteMaterializes(context))
    return result;
  if (int result = expectPostLoopSubscriptStoreRouteMaterializes(context))
    return result;

  if (int result = expectMaterializationFails(
          makeGenericRoute("test-bad-unknown-value", "missing_lhs", "sum"),
          "unknown value name"))
    return result;
  if (int result = expectMaterializationFails(
          makeGenericRoute("test-bad-duplicate-result", "lhs", "lhs"),
          "duplicate call_opaque result name"))
    return result;
  if (int result = expectMaterializationFails(
          makePostLoopSubscriptStoreRoute("missing[0]"),
          "unknown subscript base"))
    return result;

  InvalidMissingCalleeLowerable invalid;
  if (int result = expectLowerableRouteBuildFails(invalid,
                                                  "call_opaque callee"))
    return result;

  return 0;
}
