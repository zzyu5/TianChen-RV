#include "TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

using namespace tianchenrv::conversion::emitc;

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

class ValidRVVAddLowerable final : public TCRVEmitCLowerableInterface {
public:
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

    TCRVEmitCCallOpaqueStep add;
    add.sourceOp = {"tcrv_rvv.i32_add", "compute"};
    add.callee = "__riscv_vadd_vv_i32m1";
    add.operands.push_back({"lhs_vec", "vint32m1_t"});
    add.operands.push_back({"rhs_vec", "vint32m1_t"});
    add.operands.push_back({"vl", "size_t"});
    add.result = TCRVEmitCCallOpaqueResult{"sum_vec", "vint32m1_t"};
    route.addCallOpaqueStep(std::move(add));

    return route;
  }
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

} // namespace

int main() {
  ValidRVVAddLowerable valid;
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
          expect(route->getCallOpaqueSteps().size() == 2,
                 "common route preserves emitc.call_opaque steps"))
    return result;
  if (int result = expect(route->getCallOpaqueSteps()[1].sourceOp.opName ==
                              "tcrv_rvv.i32_add",
                          "common route records typed source op provenance"))
    return result;
  if (int result =
          expect(route->getCallOpaqueSteps()[1].callee ==
                     "__riscv_vadd_vv_i32m1",
                 "common route records intrinsic call mapping"))
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
