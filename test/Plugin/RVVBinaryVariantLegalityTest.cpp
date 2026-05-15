#include "TianChenRV/InitTianChenRVDialects.h"
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/RVV/RVVBinaryVariantLegality.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>

using tianchenrv::plugin::VariantLegalityRequest;
using tianchenrv::support::TargetCapabilitySet;
using tianchenrv::tcrv::exec::KernelOp;
using tianchenrv::tcrv::exec::VariantOp;

namespace {

int fail(llvm::Twine message) {
  llvm::errs() << "FAIL: " << message << "\n";
  return 1;
}

int expectSuccess(llvm::Error error, llvm::Twine context) {
  if (!error)
    return 0;

  std::string message = llvm::toString(std::move(error));
  return fail(context + ": " + message);
}

int expectErrorContains(llvm::Error error,
                        std::initializer_list<llvm::StringRef> fragments) {
  if (!error)
    return fail("expected RVV binary variant-legality error");

  std::string message = llvm::toString(std::move(error));
  for (llvm::StringRef fragment : fragments) {
    if (!llvm::StringRef(message).contains(fragment))
      return fail(llvm::Twine("error text missing '") + fragment +
                  "': " + message);
  }
  return 0;
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

VariantOp findVariant(KernelOp kernel, llvm::StringRef name) {
  VariantOp variant;
  kernel->walk([&](VariantOp candidate) {
    if (candidate.getSymName() == name)
      variant = candidate;
  });
  return variant;
}

constexpr llvm::StringLiteral kVariantLegalitySource = R"mlir(
module {
  tcrv.exec.kernel @rvv_variant_legality attributes {} {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      selected_mabi = "lp64d",
      status = "available"
    }
    tcrv.exec.capability @rvv_vlenb_bytes {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 16 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 4 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_sew32 {
      id = "rvv.i32_m1.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lmul_m1 {
      id = "rvv.i32_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_tail_agnostic {
      id = "rvv.i32_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_mask_agnostic {
      id = "rvv.i32_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_sew64 {
      id = "rvv.i64_m1.sew64",
      kind = "isa-vector-config",
      sew_bits = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_lmul_m1 {
      id = "rvv.i64_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_tail_agnostic {
      id = "rvv.i64_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i64_m1_mask_agnostic {
      id = "rvv.i64_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }

    tcrv.exec.variant @i32_vadd attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.vlenb_bytes = 16 : i64,
      tcrv_rvv.base_i32_m1_lanes = 4 : i64,
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

    tcrv.exec.variant @i32_default_typed attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.selected_binary_dtype = "i32",
      tcrv_rvv.selected_binary_family = "i32-vadd",
      tcrv_rvv.selected_binary_operator = "add",
      tcrv_rvv.selected_binary_source_kind = "default-i32-vadd-typed-body-materialization",
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

    tcrv.exec.variant @i32_default_missing_selected_source attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
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

    tcrv.exec.variant @i32_vadd_mirror attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.selected_binary_dtype = "i32",
      tcrv_rvv.selected_binary_family = "i32-vadd",
      tcrv_rvv.selected_binary_operator = "add",
      tcrv_rvv.selected_binary_source_kind = "default-i32-vadd-typed-body-materialization",
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

    tcrv.exec.variant @i64_vmul attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i64_m1_sew64, @rvv_i64_m1_lmul_m1, @rvv_i64_m1_tail_agnostic, @rvv_i64_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.selected_binary_dtype = "i64",
      tcrv_rvv.selected_binary_family = "i64-vmul",
      tcrv_rvv.selected_binary_operator = "multiply",
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

    tcrv_rvv.i64_vmul_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv, @rvv_i64_m1_sew64, @rvv_i64_m1_lmul_m1, @rvv_i64_m1_tail_agnostic, @rvv_i64_m1_mask_agnostic],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @i64_vmul,
      selected_vector_shape = "i64m1",
      selected_vector_sew = 64 : i64,
      selected_vector_lmul = "m1",
      selected_tail_policy = "agnostic",
      selected_mask_policy = "agnostic",
      selected_vector_type = "vint64m1_t",
      selected_vector_suffix = "i64m1",
      selected_setvl_suffix = "e64m1",
      source_kernel = "rvv_variant_legality"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
        %lhs = tcrv_rvv.i64_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i64m1
        %rhs = tcrv_rvv.i64_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i64m1
        %product = tcrv_rvv.i64_mul %lhs, %rhs, %vl : !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
        tcrv_rvv.i64_store %product, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i64m1, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }

    tcrv.exec.variant @i64_vsub attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i64_m1_sew64, @rvv_i64_m1_lmul_m1, @rvv_i64_m1_tail_agnostic, @rvv_i64_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
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

    tcrv.exec.variant @i64_direct_source_without_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i64_m1_sew64, @rvv_i64_m1_lmul_m1, @rvv_i64_m1_tail_agnostic, @rvv_i64_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
      tcrv_rvv.selected_binary_dtype = "i64",
      tcrv_rvv.selected_binary_family = "i64-vmul",
      tcrv_rvv.selected_binary_operator = "multiply",
      tcrv_rvv.selected_binary_source_kind = "direct-typed-microkernel-body",
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

    tcrv.exec.variant @bad_origin attributes {
      origin = "not-rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }

    tcrv.exec.variant @bad_shape attributes {
      origin = "rvv-plugin",
      requires = [@rvv, @rvv_i32_m1_sew32, @rvv_i32_m1_lmul_m1, @rvv_i32_m1_tail_agnostic, @rvv_i32_m1_mask_agnostic],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.element_count = 16 : i64,
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

int runVariantLegalityModuleTest(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parseModule(context, kVariantLegalitySource);
  if (!module)
    return fail("failed to parse RVV binary variant-legality module");

  KernelOp kernel = findKernel(*module, "rvv_variant_legality");
  TargetCapabilitySet capabilities = TargetCapabilitySet::buildFromKernel(kernel);

  auto verifyVariant = [&](llvm::StringRef symbol) {
    return tianchenrv::plugin::rvv::verifyRVVBinaryVariantLegality(
        VariantLegalityRequest(findVariant(kernel, symbol), kernel,
                               capabilities),
        tianchenrv::plugin::rvv::getRVVExtensionPluginName());
  };

  if (int result = expectErrorContains(
          verifyVariant("i32_vadd"),
          {"requires an actual typed RVV extension-family body",
           "metadata cannot make a direct RVV binary variant legal"}))
    return result;
  if (int result = expectErrorContains(
          verifyVariant("i32_default_typed"),
          {"RVV selected-source metadata", "deleted as RVV finite-family "
                                          "legality authority"}))
    return result;
  if (int result = expectErrorContains(
          verifyVariant("i32_default_missing_selected_source"),
          {"requires an actual typed RVV extension-family body",
           "metadata cannot make a direct RVV binary variant legal"}))
    return result;
  if (int result = expectErrorContains(
          verifyVariant("i32_vadd_mirror"),
          {"RVV selected-source metadata", "deleted as RVV finite-family "
                                          "legality authority"}))
    return result;
  if (int result = expectSuccess(verifyVariant("i64_vmul"),
                                 "direct module accepts i64-vmul RVV variant "
                                 "with typed body authority"))
    return result;
  if (int result = expectErrorContains(
          verifyVariant("i64_vsub"),
          {"requires an actual typed RVV extension-family body",
           "metadata cannot make a direct RVV binary variant legal"}))
    return result;
  if (int result = expectErrorContains(
          verifyVariant("i64_direct_source_without_body"),
          {"RVV selected-source metadata", "deleted as RVV finite-family "
                                          "legality authority"}))
    return result;

  if (int result = expectErrorContains(
          verifyVariant("bad_origin"), {"origin", "rvv-plugin"}))
    return result;
  if (int result = expectErrorContains(
          verifyVariant("bad_shape"),
          {"selected vector-shape id", "i32m1"}))
    return result;

  return 0;
}

} // namespace

int main() {
  mlir::DialectRegistry dialectRegistry;
  tianchenrv::plugin::ExtensionPluginRegistry plugins;
  if (int result =
          expectSuccess(tianchenrv::plugin::registerRVVExtensionPlugin(plugins),
                        "register RVV plugin for variant-legality test"))
    return result;
  tianchenrv::registerAllDialects(dialectRegistry);
  tianchenrv::registerPluginDialects(plugins, dialectRegistry);

  mlir::MLIRContext context(dialectRegistry);
  context.loadAllAvailableDialects();

  if (int result = runVariantLegalityModuleTest(context))
    return result;

  llvm::outs() << "RVV binary variant-legality tests passed\n";
  return 0;
}
