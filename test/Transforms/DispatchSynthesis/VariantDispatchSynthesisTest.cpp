#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Transforms/Passes.h"
#include "TianChenRV/Transforms/VariantDispatchSynthesis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using tianchenrv::tcrv::exec::DispatchCaseOp;
using tianchenrv::tcrv::exec::DispatchOp;
using tianchenrv::tcrv::exec::FallbackOp;
using tianchenrv::tcrv::exec::KernelOp;

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

mlir::OwningOpRef<mlir::ModuleOp> parseTestModule(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @synthesis_anchor attributes {} {
    tcrv.exec.capability @fast_probe {
      id = "generic.fast.probe",
      kind = "runtime",
      status = "missing"
    }
    tcrv.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain"
    }
    tcrv.exec.capability @conflicting_runtime {
      id = "generic.conflicting.runtime",
      kind = "runtime",
      conflicts = ["build.policy.disable_conflicting_runtime"],
      status = "available"
    }
    tcrv.exec.capability @conflict_policy {
      id = "generic.conflict.policy",
      kind = "build-policy",
      provides = ["build.policy.disable_conflicting_runtime"],
      status = "available"
    }
    tcrv.exec.capability @extra_capability {
      id = "generic.extra",
      kind = "toolchain"
    }
    tcrv.exec.variant @fast_path attributes {
      condition = "runtime_probe_available",
      guard = "generic_fast_guard",
      origin = "fast-plugin",
      policy = "prefer_fast_when_guarded",
      requires = [@fast_probe]
    } {
    }
    tcrv.exec.variant @conflicting_path attributes {
      origin = "conflicting-plugin",
      requires = [@conflicting_runtime]
    } {
    }
	    tcrv.exec.variant @baseline_path attributes {
	      condition = "baseline_fallback_condition",
	      fallback_role = "conservative",
	      origin = "baseline-plugin",
	      policy = "baseline_fallback_policy",
	      requires = [@baseline_capability]
    } {
    }
    tcrv.exec.variant @extra_path attributes {
      condition = "extra_condition",
      guard = "extra_guard",
      origin = "extra-plugin",
      policy = "extra_policy",
      requires = [@extra_capability]
    } {
    }
  }

  tcrv.exec.kernel @already_dispatched attributes {} {
    tcrv.exec.capability @baseline_capability {
      id = "generic.baseline",
      kind = "toolchain"
    }
    tcrv.exec.variant @baseline_path attributes {
      origin = "baseline-plugin",
      requires = [@baseline_capability]
    } {
    }
    tcrv.exec.variant @alternate_path attributes {
      origin = "alternate-plugin",
      requires = [@baseline_capability]
    } {
    }
    tcrv.exec.dispatch attributes {} {
      tcrv.exec.case @alternate_path {condition = "existing_generic_guard"}
      tcrv.exec.fallback @baseline_path
    }
  }
}
)mlir";

  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module->walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

llvm::SmallVector<DispatchOp, 2> collectDirectDispatches(KernelOp kernel) {
  llvm::SmallVector<DispatchOp, 2> dispatches;
  if (!kernel || kernel.getBody().empty())
    return dispatches;

  for (mlir::Operation &op : kernel.getBody().front()) {
    if (auto dispatch = llvm::dyn_cast<DispatchOp>(op))
      dispatches.push_back(dispatch);
  }
  return dispatches;
}

llvm::StringRef getTarget(mlir::Operation *op) {
  auto target = op->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
  if (!target)
    return {};
  return target.getValue();
}

int runSynthesisApiTest(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseTestModule(context);
  if (!module)
    return fail("failed to parse dispatch synthesis test module");

  KernelOp kernel = findKernel(*module, "synthesis_anchor");
  if (int result = expect(static_cast<bool>(kernel), "kernel is present"))
    return result;

  mlir::OpBuilder builder(&context);
  DispatchOp createdDispatch;
  if (int result = expect(mlir::succeeded(
                              tianchenrv::transforms::synthesizeVariantDispatch(
                                  builder, kernel, &createdDispatch)),
                          "dispatch synthesis succeeds"))
    return result;
  if (int result = expect(static_cast<bool>(createdDispatch),
                          "API returns created dispatch op"))
    return result;

  llvm::SmallVector<DispatchOp, 2> dispatches = collectDirectDispatches(kernel);
  if (int result = expect(dispatches.size() == 1,
                          "exactly one direct dispatch is synthesized"))
    return result;
  if (int result = expect(dispatches[0] == createdDispatch,
                          "returned dispatch is the direct kernel child"))
    return result;

  llvm::SmallVector<DispatchCaseOp, 3> cases;
  FallbackOp fallback;
  for (mlir::Operation &op : createdDispatch.getBody().front()) {
    if (auto dispatchCase = llvm::dyn_cast<DispatchCaseOp>(op)) {
      cases.push_back(dispatchCase);
      continue;
    }
    if (auto fallbackCandidate = llvm::dyn_cast<FallbackOp>(op))
      fallback = fallbackCandidate;
  }

  if (int result = expect(cases.size() == 3,
                          "dispatch has typed case ops for non-fallbacks"))
    return result;
  if (int result = expect(static_cast<bool>(fallback),
                          "dispatch has typed fallback op"))
    return result;
  if (int result =
          expect(getTarget(cases[0].getOperation()) == "fast_path",
                 "first case follows original variant order"))
    return result;
  if (int result =
          expect(getTarget(cases[1].getOperation()) == "conflicting_path",
                 "conflicting available variant remains a guarded case"))
    return result;
  if (int result =
          expect(getTarget(cases[2].getOperation()) == "extra_path",
                 "third case skips fallback and preserves order"))
    return result;
  if (int result =
          expect(getTarget(fallback.getOperation()) == "baseline_path",
	                 "fallback is the first available conservative fallback variant"))
    return result;
  if (int result =
          expect(cases[0]->getAttrOfType<mlir::StringAttr>("condition")
                     .getValue() == "runtime_probe_available",
                 "unavailable case inherits variant condition metadata"))
    return result;
  if (int result =
          expect(cases[0]->getAttrOfType<mlir::StringAttr>("guard")
                     .getValue() == "generic_fast_guard",
                 "unavailable case inherits variant guard metadata"))
    return result;
  if (int result =
          expect(cases[0]->getAttrOfType<mlir::StringAttr>("policy")
                     .getValue() == "prefer_fast_when_guarded",
                 "unavailable case preserves plugin policy metadata"))
    return result;
  if (int result =
          expect(cases[1]->getAttrOfType<mlir::StringAttr>("policy")
                     .getValue() == "capability_dispatch_guard",
                 "conflicting available case receives synthesized generic policy guard"))
    return result;
  if (int result =
          expect(cases[0]->getAttrOfType<mlir::BoolAttr>(
                     "runtime_guard_required") &&
                     cases[0]
                         ->getAttrOfType<mlir::BoolAttr>(
                             "runtime_guard_required")
                         .getValue(),
                 "unavailable case receives typed runtime guard requirement"))
    return result;
  if (int result =
          expect(cases[1]->getAttrOfType<mlir::BoolAttr>(
                     "runtime_guard_required") &&
                     cases[1]
                         ->getAttrOfType<mlir::BoolAttr>(
                             "runtime_guard_required")
                         .getValue(),
                 "conflicting case receives typed runtime guard requirement"))
    return result;
  if (int result =
          expect(cases[2]->getAttrOfType<mlir::StringAttr>("condition")
                     .getValue() == "extra_condition",
                 "available case inherits variant condition metadata"))
    return result;
  if (int result =
          expect(cases[2]->getAttrOfType<mlir::StringAttr>("guard")
                     .getValue() == "extra_guard",
                 "available case inherits variant guard metadata"))
    return result;
  if (int result =
          expect(cases[2]->getAttrOfType<mlir::StringAttr>("policy")
                     .getValue() == "extra_policy",
                 "available case inherits variant policy metadata"))
    return result;
  if (int result = expect(cases[0]->getAttrOfType<mlir::StringAttr>("policy")
                                  .getValue() !=
                              "capability_dispatch_guard",
                          "inherited metadata is not overwritten by synthesized guard"))
    return result;
  if (int result = expect(!fallback->getAttr("condition") &&
                              !fallback->getAttr("guard") &&
                              !fallback->getAttr("policy") &&
                              !fallback->getAttr("runtime_guard_required") &&
                              !fallback->getAttr("runtime_guard"),
                          "fallback does not receive dispatch-case metadata"))
    return result;
  if (int result =
          expect(!cases[2]->getAttr("runtime_guard_required"),
                 "available annotated case does not receive typed guard requirement"))
    return result;

  if (int result = expect(mlir::succeeded(mlir::verify(*module)),
                          "synthesized module verifies"))
    return result;

  mlir::PassManager passManager(&context);
  passManager.addPass(
      tianchenrv::transforms::createCheckCapabilityRequiresPass());
  if (int result =
          expect(mlir::succeeded(passManager.run(*module)),
                 "check-capability-requires accepts synthesized dispatch"))
    return result;

  return 0;
}

int runExistingDispatchTest(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseTestModule(context);
  if (!module)
    return fail("failed to parse existing-dispatch test module");

  KernelOp kernel = findKernel(*module, "already_dispatched");
  if (int result =
          expect(collectDirectDispatches(kernel).size() == 1,
                 "existing dispatch precondition holds"))
    return result;

  mlir::OpBuilder builder(&context);
  DispatchOp createdDispatch;
  if (int result = expect(mlir::succeeded(
                              tianchenrv::transforms::synthesizeVariantDispatch(
                                  builder, kernel, &createdDispatch)),
                          "existing dispatch is left unchanged successfully"))
    return result;
  if (int result = expect(!createdDispatch,
                          "API does not report a new dispatch for existing IR"))
    return result;
  if (int result =
          expect(collectDirectDispatches(kernel).size() == 1,
                 "existing dispatch is not duplicated"))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runSynthesisApiTest(context))
    return result;
  if (int result = runExistingDispatchTest(context))
    return result;

  llvm::outs() << "variant dispatch synthesis smoke test passed\n";
  return 0;
}
