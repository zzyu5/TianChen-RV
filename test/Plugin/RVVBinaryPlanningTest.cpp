#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using tianchenrv::plugin::rvv::RVVBinarySelectedPlan;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expect(bool condition, llvm::Twine message) {
  if (condition)
    return 0;
  return fail(message);
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

template <typename T>
int expectExpectedSuccess(llvm::Expected<T> value, T &out,
                          llvm::Twine context) {
  if (!value)
    return fail(context + ": " + llvm::toString(value.takeError()));
  out = std::move(*value);
  return 0;
}

int expectErrorContains(llvm::Error error, llvm::StringRef fragment) {
  if (!error)
    return fail("expected error");

  std::string message = llvm::toString(std::move(error));
  return expect(llvm::StringRef(message).contains(fragment),
                llvm::Twine("error text contains '") + fragment + "'");
}

std::unique_ptr<mlir::Operation, void (*)(mlir::Operation *)>
makeOperation(mlir::OperationState &state) {
  return {mlir::Operation::create(state),
          [](mlir::Operation *op) {
            if (op)
              op->destroy();
          }};
}

int runI32SelectedPlanTest() {
  RVVBinarySelectedPlan plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinarySelectedPlan(
              tianchenrv::target::rvv::getI32VSubFamilyDescriptor(),
              tianchenrv::target::rvv::getI32M2VectorShapeConfig(), 32,
              "rv64gcv", std::string("lp64d")),
          plan, "build i32-vsub i32m2 selected plan"))
    return result;

  if (int result =
          expect(plan.getFamilyID() == "i32-vsub",
                 "i32 selected plan preserves binary family id"))
    return result;
  if (int result =
          expect(plan.getRouteID() ==
                     "tcrv-export-rvv-i32-vsub-microkernel-c",
                 "i32 selected plan exposes RVV route id"))
    return result;
  if (int result =
          expect(plan.getRuntimeABI() ==
                     "rvv-i32-vsub-runtime-callable-c-abi.v1",
                 "i32 selected plan exposes runtime ABI"))
    return result;
  if (int result =
          expect(plan.getEmissionPath() ==
                     "rvv-explicit-i32-vsub-microkernel-c-source-export",
                 "i32 selected plan exposes readiness emission path"))
    return result;
  if (int result =
          expect(plan.getSetVLIntrinsicName() == "__riscv_vsetvl_e32m2",
                 "i32 selected plan owns setvl intrinsic spelling"))
    return result;
  if (int result =
          expect(plan.getArithmeticIntrinsicName() ==
                     "__riscv_vsub_vv_i32m2",
                 "i32 selected plan appends selected vector suffix to RVV "
                 "arithmetic intrinsic prefix"))
    return result;
  if (int result =
          expect(plan.getLoadIntrinsicName() == "__riscv_vle32_v_i32m2",
                 "i32 selected plan owns load intrinsic spelling"))
    return result;
  return expect(plan.getStoreIntrinsicName() == "__riscv_vse32_v_i32m2",
                "i32 selected plan owns store intrinsic spelling");
}

int runI64SelectedPlanTest() {
  auto expectI64Family =
      [](const tianchenrv::target::rvv::RVVBinaryFamilyDescriptor &family)
      -> int {
    RVVBinarySelectedPlan plan;
    if (int result = expectExpectedSuccess(
            tianchenrv::plugin::rvv::buildRVVBinarySelectedPlan(
                family, tianchenrv::target::rvv::getI64M1VectorShapeConfig(),
                16, "rv64gcv"),
            plan, llvm::Twine("build selected plan for ") + family.familyID))
      return result;

    if (int result =
            expect(plan.getFamilyID() == family.familyID,
                   "i64 selected plan preserves binary family id"))
      return result;
    if (int result =
            expect(plan.getRouteID() == family.routeID,
                   "i64 selected plan exposes family-specific RVV route id"))
      return result;
    if (int result = expect(
            plan.getRuntimeABI() == family.runtimeABI &&
                plan.getRuntimeABIKind() == family.runtimeABIKind &&
                plan.getRuntimeABIName() == family.runtimeABIName &&
                plan.getRuntimeGlueRole() == family.runtimeGlueRole,
            "i64 selected plan exposes descriptor-owned runtime ABI identity"))
      return result;

    llvm::SmallVector<tianchenrv::support::RuntimeABIParameter, 4> parameters =
        plan.descriptor.getCallableRuntimeABIParameters();
    using Role = tianchenrv::support::RuntimeABIParameterRole;
    using Ownership = tianchenrv::support::RuntimeABIParameterOwnership;
    if (int result = expect(
            parameters.size() == 4 && parameters[0].cName == "lhs" &&
                parameters[0].cType == "const int64_t *" &&
                parameters[0].role == Role::LHSInputBuffer &&
                parameters[0].ownership == Ownership::TargetExportABIOwned &&
                parameters[1].cName == "rhs" &&
                parameters[1].cType == "const int64_t *" &&
                parameters[1].role == Role::RHSInputBuffer &&
                parameters[1].ownership == Ownership::TargetExportABIOwned &&
                parameters[2].cName == "out" &&
                parameters[2].cType == "int64_t *" &&
                parameters[2].role == Role::OutputBuffer &&
                parameters[2].ownership == Ownership::TargetExportABIOwned &&
                parameters[3].cName == "n" &&
                parameters[3].cType == "size_t" &&
                parameters[3].role == Role::RuntimeElementCount &&
                parameters[3].ownership == Ownership::TargetExportABIOwned,
            "i64 selected plan exposes descriptor-owned callable ABI "
            "parameters"))
      return result;

    if (int result =
            expect(plan.getArithmeticIntrinsicName() ==
                       (llvm::Twine(family.arithmeticIntrinsicPrefix) +
                        "i64m1")
                           .str(),
                   "i64 selected plan owns RVV C intrinsic spelling"))
      return result;
    return expect(plan.getStoreIntrinsicName() == "__riscv_vse64_v_i64m1",
                  "i64 selected plan owns store intrinsic spelling");
  };

  if (int result =
          expectI64Family(tianchenrv::target::rvv::getI64VAddFamilyDescriptor()))
    return result;
  if (int result =
          expectI64Family(tianchenrv::target::rvv::getI64VSubFamilyDescriptor()))
    return result;
  return expectI64Family(
      tianchenrv::target::rvv::getI64VMulFamilyDescriptor());
}

int runNegativeSelectedPlanTest() {
  llvm::Expected<RVVBinarySelectedPlan> plan =
      tianchenrv::plugin::rvv::buildRVVBinarySelectedPlan(
          tianchenrv::target::rvv::getI64VAddFamilyDescriptor(),
          tianchenrv::target::rvv::getI32M1VectorShapeConfig(), 16,
          "rv64gcv");
  if (plan)
    return fail("expected dtype/shape mismatch error");
  return expectErrorContains(plan.takeError(), "selected vector-shape dtype");
}

int runSelectedShapeMetadataTest() {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  mlir::OpBuilder builder(&context);
  mlir::Location loc = builder.getUnknownLoc();

  mlir::OperationState goodState(loc, "test.rvv_selected_shape");
  tianchenrv::plugin::rvv::addRVVSelectedVectorShapeMetadataToOperationState(
      goodState, &context, tianchenrv::target::rvv::getI64M1VectorShapeConfig());
  auto goodOp = makeOperation(goodState);
  if (int result = expectSuccess(
          tianchenrv::plugin::rvv::validateRVVSelectedVectorShapeMetadata(
              goodOp.get(), "unit test i64 boundary",
              tianchenrv::target::rvv::getI64M1VectorShapeConfig(),
              tianchenrv::plugin::rvv::
                  getRVVBoundarySelectedVectorShapeMetadataNames()),
          "validate complete i64 selected vector-shape metadata"))
    return result;

  mlir::OperationState incompleteState(loc, "test.rvv_incomplete_shape");
  incompleteState.addAttribute("selected_vector_shape",
                               builder.getStringAttr("i32m1"));
  auto incompleteOp = makeOperation(incompleteState);
  if (int result = expectErrorContains(
          tianchenrv::plugin::rvv::validateRVVSelectedVectorShapeMetadata(
              incompleteOp.get(), "unit test incomplete boundary",
              tianchenrv::target::rvv::getI32M1VectorShapeConfig(),
              tianchenrv::plugin::rvv::
                  getRVVBoundarySelectedVectorShapeMetadataNames()),
          "metadata must be complete"))
    return result;

  mlir::OperationState staleState(loc, "test.rvv_stale_shape");
  tianchenrv::plugin::rvv::addRVVSelectedVectorShapeMetadataToOperationState(
      staleState, &context,
      tianchenrv::target::rvv::getI32M1VectorShapeConfig());
  auto staleOp = makeOperation(staleState);
  return expectErrorContains(
      tianchenrv::plugin::rvv::validateRVVSelectedVectorShapeMetadata(
          staleOp.get(), "unit test stale boundary",
          tianchenrv::target::rvv::getI32M2VectorShapeConfig(),
          tianchenrv::plugin::rvv::
              getRVVBoundarySelectedVectorShapeMetadataNames()),
      "selected vector-shape id must be 'i32m2'");
}

} // namespace

int main() {
  if (int result = runI32SelectedPlanTest())
    return result;
  if (int result = runI64SelectedPlanTest())
    return result;
  if (int result = runNegativeSelectedPlanTest())
    return result;
  if (int result = runSelectedShapeMetadataTest())
    return result;

  llvm::outs() << "RVV binary planning smoke test passed\n";
  return 0;
}
