#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"

#include "TianChenRV/Support/CapabilityModel.h"
#include "TianChenRV/Target/RVVScalarBinaryFamily.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <map>

using tianchenrv::plugin::rvv::RVVBinarySelectedPlan;
using tianchenrv::plugin::rvv::RVVBinaryEmissionIdentity;
using tianchenrv::plugin::rvv::RVVBinaryFamilyPlanningResolution;
using tianchenrv::plugin::rvv::RVVBinaryProposalPlan;
using tianchenrv::support::CapabilityAvailability;
using tianchenrv::support::CapabilityDescriptor;
using tianchenrv::support::TargetCapabilitySet;
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

mlir::OwningOpRef<mlir::ModuleOp>
parseModule(mlir::MLIRContext &context, llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

KernelOp findKernel(mlir::ModuleOp module, llvm::StringRef name) {
  KernelOp kernel;
  module.walk([&](KernelOp candidate) {
    if (candidate.getSymName() == name)
      kernel = candidate;
  });
  return kernel;
}

CapabilityDescriptor makeAvailableCapability(
    llvm::StringRef symbolName, llvm::StringRef id, llvm::StringRef kind,
    std::map<std::string, std::string> properties = {}) {
  return CapabilityDescriptor(symbolName, id, kind, "available",
                              CapabilityAvailability::Available,
                              std::move(properties));
}

void addBaseRVVFacts(TargetCapabilitySet &capabilities) {
  capabilities.addCapability(makeAvailableCapability(
      "rvv", "rvv", "isa-vector",
      {{"architecture", "riscv64"},
       {"isa_vector_hints", "rv64gcv_zvl128b"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_hart_count", "rvv.hart_count", "uarch", {{"count", "64"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_probe_compile_run", "rvv.probe.compile_run", "toolchain",
      {{"selected_march", "rv64gcv"}, {"selected_mabi", "lp64d"}}));
}

void addI32M2Facts(TargetCapabilitySet &capabilities) {
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_binary_selected_shape",
      "rvv.i32_binary.selected_vector_shape", "isa-vector-config",
      {{"shape", "i32m2"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m2_sew32", "rvv.i32_m2.sew32", "isa-vector-config",
      {{"sew_bits", "32"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m2_lmul_m2", "rvv.i32_m2.lmul_m2", "isa-vector-config",
      {{"lmul", "m2"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m2_tail_agnostic", "rvv.i32_m2.tail_policy.agnostic",
      "isa-vector-config", {{"tail_policy", "agnostic"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m2_mask_agnostic", "rvv.i32_m2.mask_policy.agnostic",
      "isa-vector-config", {{"mask_policy", "agnostic"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_vlenb_bytes", "rvv.vlenb_bytes", "uarch", {{"bytes", "16"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m1_lane_count", "rvv.i32_m1_lane_count", "uarch",
      {{"lanes", "4"}}));
}

void addI64M1Facts(TargetCapabilitySet &capabilities) {
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i64_m1_sew64", "rvv.i64_m1.sew64", "isa-vector-config",
      {{"sew_bits", "64"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i64_m1_lmul_m1", "rvv.i64_m1.lmul_m1", "isa-vector-config",
      {{"lmul", "m1"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i64_m1_tail_agnostic", "rvv.i64_m1.tail_policy.agnostic",
      "isa-vector-config", {{"tail_policy", "agnostic"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i64_m1_mask_agnostic", "rvv.i64_m1.mask_policy.agnostic",
      "isa-vector-config", {{"mask_policy", "agnostic"}}));
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
          expect(plan.getArtifactKind() == "runtime-callable-c-source",
                 "i32 selected plan exposes source artifact kind"))
    return result;
  if (int result = expect(
          plan.getSelectedConfig().getContract().getFamilyID() == "i32-vsub" &&
              plan.getSelectedConfig().getContract().getDTypeID() == "i32" &&
              plan.getSelectedConfig().getContract().getShapeID() == "i32m2" &&
              plan.getSelectedConfig().getContract().getVectorType() ==
                  "vint32m2_t" &&
              plan.getSelectedConfig()
                      .getContract()
                      .getRuntimeElementCountCName() == "n" &&
              plan.getSelectedConfig().getContract().getDescriptorElementCount() ==
                  32,
          "i32 selected plan owns the selected-config contract boundary"))
    return result;
  if (int result = expect(
          plan.getSupportedMessage() ==
              "explicit RVV i32 vector-subtract microkernel C source export "
              "provides a library-style runtime-callable C ABI function for "
              "this selected path; any self-check main is an explicit harness "
              "export and is not the default artifact contract; this is not "
              "generic RVV lowering, runtime integration, arbitrary kernel "
              "emission, correctness, or performance evidence",
          "i32 selected plan exposes planner-owned bounded support message"))
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
    if (int result = expect(
            plan.getSelectedConfig().getContract().getFamilyID() ==
                    family.familyID &&
                plan.getSelectedConfig().getContract().getDTypeID() == "i64" &&
                plan.getSelectedConfig().getContract().getShapeID() ==
                    "i64m1" &&
                plan.getSelectedConfig().getContract().getSEWBits() == 64 &&
                plan.getSelectedConfig().getContract().getVectorSuffix() ==
                    "i64m1" &&
                plan.getSelectedConfig()
                        .getContract()
                        .getDescriptorElementCount() == 16,
            "i64 selected plan owns the i64m1 selected-config contract"))
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

int runEmissionIdentityTest() {
  RVVBinaryEmissionIdentity i32Identity;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryEmissionIdentity(
              tianchenrv::target::rvv::getI32VAddFamilyDescriptor()),
          i32Identity, "build i32-vadd emission identity"))
    return result;
  if (int result =
          expect(i32Identity.getRouteID() ==
                     "tcrv-export-rvv-microkernel-c",
                 "i32 emission identity references target-owned route id"))
    return result;
  if (int result =
          expect(i32Identity.getEmissionPath() ==
                     "rvv-explicit-i32-vadd-microkernel-c-source-export",
                 "i32 emission identity derives readiness path"))
    return result;
  if (int result =
          expect(i32Identity.getArtifactKind() == "runtime-callable-c-source",
                 "i32 emission identity preserves source artifact kind"))
    return result;

  RVVBinaryEmissionIdentity i64VMulIdentity;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryEmissionIdentity(
              tianchenrv::target::rvv::getI64VMulFamilyDescriptor()),
          i64VMulIdentity, "build i64-vmul emission identity"))
    return result;

  const auto &dispatchFamily =
      tianchenrv::target::rvv_scalar::getI64VMulFamilyDescriptor().dispatch;
  if (int result = expect(
          llvm::StringRef(dispatchFamily.rvvRouteID) ==
              i64VMulIdentity.getRouteID(),
          "i64-vmul dispatch family reuses the selected RVV route id"))
    return result;
  if (int result = expect(
          llvm::StringRef(dispatchFamily.rvvEmissionKind) ==
              i64VMulIdentity.getEmissionKind(),
          "i64-vmul dispatch family reuses the selected RVV emission kind"))
    return result;
  if (int result = expect(
          llvm::StringRef(dispatchFamily.rvvRuntimeABIName) ==
              i64VMulIdentity.getRuntimeABIName(),
          "i64-vmul dispatch family reuses the selected RVV ABI name"))
    return result;
  if (int result =
          expect(dispatchFamily.dispatchObjectRouteID ==
                     "tcrv-export-rvv-scalar-i64-vmul-dispatch-object",
                 "i64-vmul dispatch object route remains target-owned"))
    return result;
  return expect(dispatchFamily.selfCheckSuccessMarker ==
                    "tcrv_rvv_scalar_i64_vmul_dispatch_self_check_ok",
                "i64-vmul dispatch success marker remains target-owned");
}

int runProposalPlanRequirementMetadataTest() {
  TargetCapabilitySet i32Capabilities;
  addBaseRVVFacts(i32Capabilities);
  addI32M2Facts(i32Capabilities);

  RVVBinaryProposalPlan i32Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i32Capabilities, "i32-vmul", "unit i32-vmul proposal"),
          i32Plan, "build i32-vmul proposal plan"))
    return result;

  if (int result =
          expect(i32Plan.getFamilyID() == "i32-vmul",
                 "i32 proposal plan preserves finite family id"))
    return result;
  if (int result =
          expect(i32Plan.getSelectedShape().shapeID == "i32m2",
                 "i32 proposal plan selects explicit i32m2 shape"))
    return result;
  if (int result =
          expect(i32Plan.getRequiredCapabilityIDs().size() == 5 &&
                     i32Plan.getRequiredCapabilityIDs()[0] == "rvv" &&
                     i32Plan.getRequiredCapabilityIDs()[1] ==
                         "rvv.i32_m2.sew32" &&
                     i32Plan.getRequiredCapabilityIDs()[2] ==
                         "rvv.i32_m2.lmul_m2" &&
                     i32Plan.getRequiredCapabilityIDs()[3] ==
                         "rvv.i32_m2.tail_policy.agnostic" &&
                     i32Plan.getRequiredCapabilityIDs()[4] ==
                         "rvv.i32_m2.mask_policy.agnostic",
                 "i32 proposal plan owns deterministic requirement ids"))
    return result;
  if (int result =
          expect(i32Plan.selectedPlan.elementCount == 16 &&
                     i32Plan.hasCapacityMetadata() &&
                     *i32Plan.capabilityView.vlenbBytes == 16 &&
                     *i32Plan.capabilityView.i32M1LaneCount == 4,
                 "i32 proposal plan keeps capacity facts separate from "
                 "descriptor-local element count"))
    return result;
  if (int result =
          expect(i32Plan.getCondition() ==
                         "rvv_capability_properties_available" &&
                     i32Plan.getGuard() ==
                         "plugin_local_rvv_property_evidence" &&
                     i32Plan.getPolicy() == "metadata_only_first_slice",
                 "i32 proposal plan carries generic proposal metadata"))
    return result;

  TargetCapabilitySet i64Capabilities;
  addBaseRVVFacts(i64Capabilities);
  addI64M1Facts(i64Capabilities);

  RVVBinaryProposalPlan i64Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i64Capabilities, "i64-vmul", "unit i64-vmul proposal"),
          i64Plan, "build i64-vmul proposal plan"))
    return result;
  if (int result =
          expect(i64Plan.getFamilyID() == "i64-vmul" &&
                     i64Plan.getLoweringDescriptor() ==
                         "i64-vmul-microkernel.v1" &&
                     i64Plan.getSelectedShape().shapeID == "i64m1",
                 "i64 proposal plan preserves finite family and i64m1 shape"))
    return result;
  if (int result =
          expect(i64Plan.getRequiredCapabilityIDs().size() == 5 &&
                     i64Plan.getRequiredCapabilityIDs()[1] ==
                         "rvv.i64_m1.sew64" &&
                     i64Plan.getRequiredCapabilityIDs()[2] ==
                         "rvv.i64_m1.lmul_m1" &&
                     i64Plan.getRequiredCapabilityIDs()[3] ==
                         "rvv.i64_m1.tail_policy.agnostic" &&
                     i64Plan.getRequiredCapabilityIDs()[4] ==
                         "rvv.i64_m1.mask_policy.agnostic",
                 "i64 proposal plan owns i64m1 requirement ids"))
    return result;

  const auto &dispatchFamily =
      tianchenrv::target::rvv_scalar::getI64VMulFamilyDescriptor().dispatch;
  if (int result =
          expect(llvm::StringRef(dispatchFamily.rvvRouteID) ==
                     i64Plan.selectedPlan.getRouteID(),
                 "i64-vmul dispatch representative reuses planner RVV route"))
    return result;
  if (int result = expect(
          llvm::StringRef(dispatchFamily.rvvRuntimeABIName) ==
              i64Plan.selectedPlan.getRuntimeABIName(),
          "i64-vmul dispatch representative reuses planner RVV ABI name"))
    return result;

  llvm::Expected<RVVBinaryProposalPlan> unsupported =
      tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
          i64Capabilities, "i16-vadd", "unit unsupported proposal");
  if (unsupported)
    return fail("expected unsupported finite family proposal error");
  return expectErrorContains(unsupported.takeError(),
                             "frontend lowering family must be");
}

int runDirectDescriptorPlanningContractTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @direct_i64_contract {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i64_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.lowering_descriptor = "i64-vadd-microkernel.v1",
      tcrv_rvv.selected_vector_shape = "i64m1",
      tcrv_rvv.selected_vector_sew = 64 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint64m1_t",
      tcrv_rvv.selected_vector_suffix = "i64m1",
      tcrv_rvv.selected_setvl_suffix = "e64m1"
    } {
    }
  }

  tcrv.exec.kernel @direct_i32m2_contract {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1",
      tcrv_rvv.selected_vector_shape = "i32m2",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m2",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m2_t",
      tcrv_rvv.selected_vector_suffix = "i32m2",
      tcrv_rvv.selected_setvl_suffix = "e32m2"
    } {
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse direct descriptor planning contract module");

  KernelOp i64Kernel = findKernel(*module, "direct_i64_contract");
  RVVBinaryFamilyPlanningResolution i64Resolution;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::resolveRVVBinaryFamilyForProposal(
              i64Kernel, "unit direct i64 descriptor contract"),
          i64Resolution, "resolve direct i64 descriptor contract"))
    return result;
  if (int result =
          expect(i64Resolution.getFamilyID() == "i64-vadd" &&
                     i64Resolution.getLoweringDescriptor() ==
                         "i64-vadd-microkernel.v1" &&
                     i64Resolution.getSourceKind() ==
                         "direct-lowering-descriptor" &&
                     i64Resolution.getDirectSelectedShapeID() == "i64m1",
                 "direct i64 descriptor contract resolves family and shape"))
    return result;

  llvm::SmallVector<llvm::StringRef, 4> i64CapabilityIDs =
      i64Resolution.getDirectSelectedCapabilityIDs();
  if (int result = expect(
          i64CapabilityIDs.size() == 4 &&
              i64CapabilityIDs[0] == "rvv.i64_m1.sew64" &&
              i64CapabilityIDs[1] == "rvv.i64_m1.lmul_m1" &&
              i64CapabilityIDs[2] == "rvv.i64_m1.tail_policy.agnostic" &&
              i64CapabilityIDs[3] == "rvv.i64_m1.mask_policy.agnostic",
          "direct i64 descriptor contract derives i64m1 capability ids"))
    return result;

  TargetCapabilitySet i64Capabilities;
  addBaseRVVFacts(i64Capabilities);
  addI64M1Facts(i64Capabilities);
  RVVBinaryProposalPlan i64Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i64Capabilities, i64Kernel, "unit direct i64 proposal"),
          i64Plan, "build proposal from direct i64 descriptor contract"))
    return result;
  if (int result =
          expect(i64Plan.getFamilyID() == "i64-vadd" &&
                     i64Plan.getSelectedShape().shapeID == "i64m1" &&
                     i64Plan.getRequiredCapabilityIDs().size() == 5 &&
                     i64Plan.getRequiredCapabilityIDs()[1] ==
                         "rvv.i64_m1.sew64" &&
                     i64Plan.getRequiredCapabilityIDs()[4] ==
                         "rvv.i64_m1.mask_policy.agnostic",
                 "direct i64 proposal uses centralized contract capability "
                 "ids"))
    return result;

  KernelOp i32Kernel = findKernel(*module, "direct_i32m2_contract");
  TargetCapabilitySet i32Capabilities;
  addBaseRVVFacts(i32Capabilities);
  addI32M2Facts(i32Capabilities);
  RVVBinaryProposalPlan i32Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i32Capabilities, i32Kernel, "unit direct i32m2 proposal"),
          i32Plan, "build proposal from direct i32m2 descriptor contract"))
    return result;
  return expect(i32Plan.getFamilyID() == "i32-vsub" &&
                    i32Plan.getSelectedShape().shapeID == "i32m2" &&
                    i32Plan.getRequiredCapabilityIDs()[1] ==
                        "rvv.i32_m2.sew32" &&
                    i32Plan.getRequiredCapabilityIDs()[2] ==
                        "rvv.i32_m2.lmul_m2",
                "direct i32 descriptor contract preserves i32m2 selection");
}

int runDirectDescriptorPlanningContractNegativeTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  auto expectResolutionError =
      [&](llvm::StringRef source, llvm::StringRef kernelName,
          llvm::StringRef fragment) -> int {
    mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
    if (!module)
      return fail(llvm::Twine("failed to parse negative module '") +
                  kernelName + "'");
    llvm::Expected<RVVBinaryFamilyPlanningResolution> resolution =
        tianchenrv::plugin::rvv::resolveRVVBinaryFamilyForProposal(
            findKernel(*module, kernelName),
            (llvm::Twine("unit negative ") + kernelName).str());
    if (resolution)
      return fail(llvm::Twine("expected direct descriptor error for '") +
                  kernelName + "'");
    return expectErrorContains(resolution.takeError(), fragment);
  };

  constexpr llvm::StringLiteral unknownDescriptor = R"mlir(
module {
  tcrv.exec.kernel @unknown_direct_descriptor {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_bad attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.lowering_descriptor = "i128-vadd-microkernel.v1"
    } {
    }
  }
}
)mlir";
  if (int result =
          expectResolutionError(unknownDescriptor, "unknown_direct_descriptor",
                                "registered finite RVV binary lowering"))
    return result;

  constexpr llvm::StringLiteral ambiguousDescriptors = R"mlir(
module {
  tcrv.exec.kernel @ambiguous_direct_descriptors {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_add attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.lowering_descriptor = "i64-vadd-microkernel.v1"
    } {
    }
    tcrv.exec.variant @rvv_sub attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.lowering_descriptor = "i64-vsub-microkernel.v1"
    } {
    }
  }
}
)mlir";
  if (int result = expectResolutionError(
          ambiguousDescriptors, "ambiguous_direct_descriptors",
          "ambiguous direct RVV binary descriptors"))
    return result;

  constexpr llvm::StringLiteral shapeMismatch = R"mlir(
module {
  tcrv.exec.kernel @descriptor_shape_mismatch {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i64_bad_shape attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.lowering_descriptor = "i64-vadd-microkernel.v1",
      tcrv_rvv.selected_vector_shape = "i32m1",
      tcrv_rvv.selected_vector_sew = 32 : i64,
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_vector_type = "vint32m1_t",
      tcrv_rvv.selected_vector_suffix = "i32m1",
      tcrv_rvv.selected_setvl_suffix = "e32m1"
    } {
    }
  }
}
)mlir";
  return expectResolutionError(
      shapeMismatch, "descriptor_shape_mismatch",
      "does not belong to finite RVV binary descriptor family 'i64-vadd'");
}

int runNegativeSelectedPlanTest() {
  llvm::Expected<RVVBinarySelectedPlan> plan =
      tianchenrv::plugin::rvv::buildRVVBinarySelectedPlan(
          tianchenrv::target::rvv::getI64VAddFamilyDescriptor(),
          tianchenrv::target::rvv::getI32M1VectorShapeConfig(), 16,
          "rv64gcv");
  if (plan)
    return fail("expected dtype/shape mismatch error");
  return expectErrorContains(plan.takeError(), "selected config mismatch");
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
  if (int result = runEmissionIdentityTest())
    return result;
  if (int result = runProposalPlanRequirementMetadataTest())
    return result;
  if (int result = runDirectDescriptorPlanningContractTest())
    return result;
  if (int result = runDirectDescriptorPlanningContractNegativeTest())
    return result;
  if (int result = runNegativeSelectedPlanTest())
    return result;
  if (int result = runSelectedShapeMetadataTest())
    return result;

  llvm::outs() << "RVV binary planning smoke test passed\n";
  return 0;
}
