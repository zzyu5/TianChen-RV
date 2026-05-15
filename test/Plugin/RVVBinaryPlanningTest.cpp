#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryPlanning.h"

#include "TianChenRV/Support/CapabilityModel.h"

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

void addI32M1Facts(TargetCapabilitySet &capabilities) {
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m1_sew32", "rvv.i32_m1.sew32", "isa-vector-config",
      {{"sew_bits", "32"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m1_lmul_m1", "rvv.i32_m1.lmul_m1", "isa-vector-config",
      {{"lmul", "m1"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m1_tail_agnostic", "rvv.i32_m1.tail_policy.agnostic",
      "isa-vector-config", {{"tail_policy", "agnostic"}}));
  capabilities.addCapability(makeAvailableCapability(
      "rvv_i32_m1_mask_agnostic", "rvv.i32_m1.mask_policy.agnostic",
      "isa-vector-config", {{"mask_policy", "agnostic"}}));
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
              tianchenrv::target::rvv::getI32VSubFamilyRegistrationRecord(),
              tianchenrv::target::rvv::getI32M2VectorShapeConfig(), 32,
              "rv64gcv", std::string("lp64d")),
          plan, "build i32-vsub i32m2 selected plan"))
    return result;

  if (int result =
          expect(plan.getFamilyID() == "i32-vsub",
                 "i32 selected plan preserves binary family id"))
    return result;
  if (int result = expect(plan.getRouteID().empty() &&
                              plan.getRuntimeABI().empty() &&
                              plan.getRuntimeABIKind().empty() &&
                              plan.getRuntimeABIName().empty() &&
                              plan.getRuntimeGlueRole().empty(),
                          "i32 selected plan no longer exposes descriptor "
                          "route or runtime ABI authority"))
    return result;
  if (int result =
          expect(plan.getEmissionPath().empty(),
                 "i32 selected plan no longer exposes direct readiness emission path"))
    return result;
  if (int result =
          expect(plan.getArtifactKind().empty(),
                 "i32 selected plan no longer exposes source artifact kind"))
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
              plan.getSelectedConfig().getContract().getComponentCapacityElementCount() ==
                  32,
          "i32 selected plan owns the selected-config contract boundary"))
    return result;
  if (int result = expect(
          plan.getSupportedMessage() ==
              "RVV i32 vector-subtract selected planning is metadata-only "
              "after direct C route deletion; artifact emission requires a "
              "materialized MLIR EmitC module route",
          "i32 selected plan exposes planner-owned deletion support message"))
    return result;
  return expect(plan.getSetVLIntrinsicName().empty() &&
                    plan.getArithmeticIntrinsicName().empty() &&
                    plan.getLoadIntrinsicName().empty() &&
                    plan.getStoreIntrinsicName().empty(),
                "i32 selected plan no longer exposes RVV intrinsic spellings");
}

int runI64SelectedPlanTest() {
  auto expectI64Family =
      [](const tianchenrv::target::rvv::RVVBinaryFamilyRecord &family)
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
    if (int result = expect(
            plan.getRouteID().empty() && plan.getRuntimeABI().empty() &&
                plan.getRuntimeABIKind().empty() &&
                plan.getRuntimeABIName().empty() &&
                plan.getRuntimeGlueRole().empty(),
            "i64 selected plan no longer exposes descriptor route or "
            "callable ABI mirrors"))
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
                        .getComponentCapacityElementCount() == 16,
            "i64 selected plan owns the i64m1 selected-config contract"))
      return result;
    return expect(plan.getSetVLIntrinsicName().empty() &&
                      plan.getArithmeticIntrinsicName().empty() &&
                      plan.getLoadIntrinsicName().empty() &&
                      plan.getStoreIntrinsicName().empty(),
                  "i64 selected plan no longer exposes RVV intrinsic spellings");
  };

  if (int result =
          expectI64Family(tianchenrv::target::rvv::getI64VAddFamilyRegistrationRecord()))
    return result;
  if (int result =
          expectI64Family(tianchenrv::target::rvv::getI64VSubFamilyRegistrationRecord()))
    return result;
  return expectI64Family(
      tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord());
}

int runEmissionIdentityTest() {
  RVVBinaryEmissionIdentity i32Identity;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryEmissionIdentity(
              tianchenrv::target::rvv::getI32VAddFamilyRegistrationRecord()),
          i32Identity, "build i32-vadd emission identity"))
    return result;
  if (int result =
          expect(i32Identity.getRouteID() ==
                     llvm::StringRef(),
                 "i32 emission identity no longer exposes target-owned route id"))
    return result;
  if (int result =
          expect(i32Identity.getEmissionPath() ==
                     llvm::StringRef(),
                 "i32 emission identity no longer derives direct readiness path"))
    return result;
  if (int result =
          expect(i32Identity.getArtifactKind().empty(),
                 "i32 emission identity no longer preserves source artifact kind"))
    return result;

  RVVBinaryEmissionIdentity i64VMulIdentity;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryEmissionIdentity(
              tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord()),
          i64VMulIdentity, "build i64-vmul emission identity"))
    return result;

  if (int result = expect(i64VMulIdentity.getRouteID().empty(),
                          "i64-vmul emission identity has no selected RVV route id"))
    return result;
  if (int result = expect(i64VMulIdentity.getEmissionKind().empty(),
                          "i64-vmul emission identity has no RVV emission kind"))
    return result;
  return expect(i64VMulIdentity.getRuntimeABIName().empty(),
                "i64-vmul emission identity has no selected RVV ABI name");
}

int runProposalPlanRequirementMetadataTest() {
  TargetCapabilitySet i32Capabilities;
  addBaseRVVFacts(i32Capabilities);
  addI32M2Facts(i32Capabilities);

  RVVBinaryProposalPlan i32Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i32Capabilities,
              tianchenrv::target::rvv::getI32VMulFamilyRegistrationRecord(),
              "unit i32-vmul proposal"),
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
                 "artifact-local component capacity"))
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
              i64Capabilities,
              tianchenrv::target::rvv::getI64VMulFamilyRegistrationRecord(),
              "unit i64-vmul proposal"),
          i64Plan, "build i64-vmul proposal plan"))
    return result;
  if (int result =
          expect(i64Plan.getFamilyID() == "i64-vmul" &&
                     !i64Plan.hasDirectTypedBodyAuthority() &&
                     i64Plan.getSelectedShape().shapeID == "i64m1",
                 "i64 proposal plan preserves typed family and i64m1 shape "
                 "without frontend source authority"))
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

  if (int result = expect(i64Plan.selectedPlan.getRouteID().empty() &&
                              i64Plan.selectedPlan.getRuntimeABIName().empty(),
                          "i64-vmul selected plan no longer mirrors dispatch "
                          "route or ABI descriptor fields"))
    return result;

  return 0;
}

int runMissingTypedBodyPlanningTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<tianchenrv::tcrv::rvv::TCRVRVVDialect>();
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @missing_typed_rvv_body {
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse missing typed-body planning module");

  KernelOp kernel = findKernel(*module, "missing_typed_rvv_body");
  llvm::Expected<RVVBinaryFamilyPlanningResolution> resolution =
      tianchenrv::plugin::rvv::resolveRVVBinaryFamilyForProposal(
          kernel, "unit missing typed RVV body");
  if (resolution)
    return fail("missing typed RVV body unexpectedly selected finite family");
  if (int result = expectErrorContains(
          resolution.takeError(),
          "requires an explicit typed RVV extension-family body"))
    return result;

  TargetCapabilitySet capabilities;
  addBaseRVVFacts(capabilities);
  addI32M1Facts(capabilities);

  llvm::Expected<RVVBinaryProposalPlan> plan =
      tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
          capabilities, kernel, "unit missing typed RVV proposal body");
  if (plan)
    return fail("missing typed RVV body unexpectedly built proposal plan");
  return expectErrorContains(
      plan.takeError(),
      "requires an explicit typed RVV extension-family body");
}

int runDeletedFrontendLoweringPlanningTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<tianchenrv::tcrv::rvv::TCRVRVVDialect>();
  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  constexpr llvm::StringLiteral source = R"mlir(
module {
  tcrv.exec.kernel @frontend_selected attributes {
    tcrv_frontend_lowering = "i32-vsub"
  } {
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse deleted frontend-lowering planning module");

  KernelOp kernel = findKernel(*module, "frontend_selected");
  llvm::Expected<RVVBinaryFamilyPlanningResolution> resolution =
      tianchenrv::plugin::rvv::resolveRVVBinaryFamilyForProposal(
          kernel, "unit deleted frontend lowering");
  if (resolution)
    return fail("frontend-lowering metadata unexpectedly selected RVV family");
  return expectErrorContains(resolution.takeError(),
                             "cannot select an RVV finite binary family from "
                             "deleted kernel metadata 'tcrv_frontend_lowering'");
}

int runDirectTypedBodyPlanningContractTest() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::registerAllDialects(dialectRegistry);
  dialectRegistry.insert<tianchenrv::tcrv::rvv::TCRVRVVDialect>();
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
		      tcrv_rvv.element_count = 16 : i64,
	      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
	      tcrv_rvv.required_march = "rv64gcv",
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
	    tcrv_rvv.i64_vadd_microkernel attributes {
	      element_count = 16 : i64,
	      origin = "rvv-plugin",
	      required_capabilities = [@rvv],
	      required_march = "rv64gcv",
	      role = "direct variant",
	      selected_variant = @rvv_i64_slice,
	      selected_vector_shape = "i64m1",
	      selected_vector_sew = 64 : i64,
	      selected_vector_lmul = "m1",
	      selected_tail_policy = "agnostic",
	      selected_mask_policy = "agnostic",
	      selected_vector_type = "vint64m1_t",
	      selected_vector_suffix = "i64m1",
	      selected_setvl_suffix = "e64m1",
	      source_kernel = "direct_i64_contract"
	    } {
	    ^bb0(%runtime_n: index):
	      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
	      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
	        %lhs = tcrv_rvv.i64_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i64m1
	        %rhs = tcrv_rvv.i64_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i64m1
	        %sum = tcrv_rvv.i64_add %lhs, %rhs, %vl : !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
	        tcrv_rvv.i64_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i64m1, !tcrv_rvv.vl
	      } : !tcrv_rvv.vl
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
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_i32_slice,
      selected_vector_shape = "i32m2",
      selected_vector_sew = 32 : i64,
      selected_vector_lmul = "m2",
      selected_tail_policy = "agnostic",
      selected_mask_policy = "agnostic",
      selected_vector_type = "vint32m2_t",
      selected_vector_suffix = "i32m2",
      selected_setvl_suffix = "e32m2",
      source_kernel = "direct_i32m2_contract"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m2, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
)mlir";

  mlir::OwningOpRef<mlir::ModuleOp> module = parseModule(context, source);
  if (!module)
    return fail("failed to parse direct typed-body planning contract module");

  KernelOp i64Kernel = findKernel(*module, "direct_i64_contract");
  RVVBinaryFamilyPlanningResolution i64Resolution;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::resolveRVVBinaryFamilyForProposal(
              i64Kernel, "unit direct i64 typed-body contract"),
          i64Resolution, "resolve direct i64 typed-body contract"))
    return result;
  if (int result =
          expect(i64Resolution.getFamilyID() == "i64-vadd" &&
                     i64Resolution.hasDirectTypedBodyAuthority() &&
                     i64Resolution.getDirectSelectedShapeID() == "i64m1",
                 "direct i64 typed body with optional legacy mirror resolves "
                 "typed family and shape"))
    return result;

  llvm::SmallVector<llvm::StringRef, 4> i64CapabilityIDs =
      i64Resolution.getDirectSelectedCapabilityIDs();
  if (int result = expect(
          i64CapabilityIDs.size() == 4 &&
              i64CapabilityIDs[0] == "rvv.i64_m1.sew64" &&
              i64CapabilityIDs[1] == "rvv.i64_m1.lmul_m1" &&
              i64CapabilityIDs[2] == "rvv.i64_m1.tail_policy.agnostic" &&
              i64CapabilityIDs[3] == "rvv.i64_m1.mask_policy.agnostic",
          "direct i64 typed-body contract derives i64m1 capability ids"))
    return result;

  TargetCapabilitySet i64Capabilities;
  addBaseRVVFacts(i64Capabilities);
  addI64M1Facts(i64Capabilities);
  RVVBinaryProposalPlan i64Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i64Capabilities, i64Kernel, "unit direct i64 proposal"),
          i64Plan,
          "build proposal from direct i64 typed-body contract"))
    return result;
  if (int result =
          expect(i64Plan.getFamilyID() == "i64-vadd" &&
                     i64Plan.getSelectedShape().shapeID == "i64m1" &&
                     i64Plan.hasDirectTypedBodyAuthority() &&
                     i64Plan.getRequiredCapabilityIDs().size() == 5 &&
                     i64Plan.getRequiredCapabilityIDs()[1] ==
                         "rvv.i64_m1.sew64" &&
                     i64Plan.getRequiredCapabilityIDs()[4] ==
                         "rvv.i64_m1.mask_policy.agnostic",
                 "direct i64 proposal uses typed-body authority and centralized "
                 "contract capability ids without descriptor compute authority"))
    return result;

  KernelOp i32Kernel = findKernel(*module, "direct_i32m2_contract");
  RVVBinaryFamilyPlanningResolution i32Resolution;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::resolveRVVBinaryFamilyForProposal(
              i32Kernel, "unit direct i32 typed-body contract"),
          i32Resolution, "resolve direct i32 typed-body contract"))
    return result;
  if (int result =
          expect(i32Resolution.getFamilyID() == "i32-vadd" &&
                     i32Resolution.hasDirectTypedBodyAuthority() &&
                     i32Resolution.getDirectSelectedShapeID() == "i32m2",
                 "direct i32 typed body resolves family and shape without "
                 "descriptor compute authority"))
    return result;

  TargetCapabilitySet i32Capabilities;
  addBaseRVVFacts(i32Capabilities);
  addI32M2Facts(i32Capabilities);
  RVVBinaryProposalPlan i32Plan;
  if (int result = expectExpectedSuccess(
          tianchenrv::plugin::rvv::buildRVVBinaryProposalPlan(
              i32Capabilities, i32Kernel, "unit direct i32m2 proposal"),
          i32Plan, "build proposal from direct i32m2 typed-body contract"))
    return result;
  return expect(i32Plan.getFamilyID() == "i32-vadd" &&
                    i32Plan.getSelectedShape().shapeID == "i32m2" &&
                    i32Plan.hasDirectTypedBodyAuthority() &&
                    i32Plan.getRequiredCapabilityIDs()[1] ==
                        "rvv.i32_m2.sew32" &&
                    i32Plan.getRequiredCapabilityIDs()[2] ==
                        "rvv.i32_m2.lmul_m2",
                "direct i32 typed body preserves i32m2 selection without "
                "reattaching descriptor compute authority");
}

int runNegativeSelectedPlanTest() {
  llvm::Expected<RVVBinarySelectedPlan> plan =
      tianchenrv::plugin::rvv::buildRVVBinarySelectedPlan(
          tianchenrv::target::rvv::getI64VAddFamilyRegistrationRecord(),
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
  if (int result = runMissingTypedBodyPlanningTest())
    return result;
  if (int result = runDeletedFrontendLoweringPlanningTest())
    return result;
  if (int result = runDirectTypedBodyPlanningContractTest())
    return result;
  if (int result = runNegativeSelectedPlanTest())
    return result;
  if (int result = runSelectedShapeMetadataTest())
    return result;

  llvm::outs() << "RVV binary planning smoke test passed\n";
  return 0;
}
