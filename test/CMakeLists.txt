configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  WEFT_SOURCE_DIR
  WEFT_BINARY_DIR
  LLVM_RUNTIME_OUTPUT_INTDIR
  LLVM_TOOLS_BINARY_DIR
)

add_executable(weft-plugin-registry-test
  Plugin/PluginRegistryTest.cpp
)

target_link_libraries(weft-plugin-registry-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  MLIRIR
  MLIRSupport
)

add_executable(weft-formula-catalog-test
  Plugin/FormulaCatalogTest.cpp
)

target_link_libraries(weft-formula-catalog-test
  PRIVATE
  WeftBuiltinPlugins
  WeftConversionEmitCBuiltin
  WeftPlugin
  MLIRSupport
)

add_executable(weft-plugin-variant-proposal-test
  Plugin/PluginVariantProposalTest.cpp
)

target_link_libraries(weft-plugin-variant-proposal-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  WeftTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(weft-plugin-variant-legality-test
  Plugin/PluginVariantLegalityTest.cpp
)

target_link_libraries(weft-plugin-variant-legality-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-plugin-variant-cost-test
  Plugin/PluginVariantCostTest.cpp
)

target_link_libraries(weft-plugin-variant-cost-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-plugin-emission-plan-test
  Plugin/PluginEmissionPlanTest.cpp
)

target_link_libraries(weft-plugin-emission-plan-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-scalar-extension-plugin-test
  Plugin/ScalarExtensionPluginTest.cpp
)

target_link_libraries(weft-scalar-extension-plugin-test
  PRIVATE
  WeftInitAll
  WeftBuiltinPlugins
  WeftPlugin
  WeftScalarPlugin
  WeftSupport
  WeftTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-offload-extension-plugin-test
  Plugin/OffloadExtensionPluginTest.cpp
)

target_link_libraries(weft-offload-extension-plugin-test
  PRIVATE
  WeftInitAll
  WeftBuiltinPlugins
  WeftPlugin
  WeftOffloadPlugin
  WeftScalarPlugin
  WeftSupport
  WeftTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-toy-extension-plugin-test
  Plugin/ToyExtensionPluginTest.cpp
)

target_link_libraries(weft-toy-extension-plugin-test
  PRIVATE
  WeftInitAll
  WeftBuiltinPlugins
  WeftPlugin
  WeftSupport
  WeftToyDialect
  WeftToyPlugin
  WeftTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(weft-template-extension-plugin-test
  Plugin/TemplateExtensionPluginTest.cpp
)

target_link_libraries(weft-template-extension-plugin-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  WeftTemplateDialect
  WeftTemplatePlugin
  WeftTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-tensorext-lite-extension-plugin-test
  Plugin/TensorExtLiteExtensionPluginTest.cpp
)

target_link_libraries(weft-tensorext-lite-extension-plugin-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  WeftTensorExtLiteDialect
  WeftTensorExtLitePlugin
  WeftTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-rvv-dialect-test
  Dialect/RVV/RVVDialectTest.cpp
)

target_link_libraries(weft-rvv-dialect-test
  PRIVATE
  WeftInitAll
  WeftRVVDialect
  WeftRVVPlugin
  WeftPlugin
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-rvv-low-precision-lmul-selection-test
  Plugin/RVVLowPrecisionLMULSelectionTest.cpp
)

target_link_libraries(weft-rvv-low-precision-lmul-selection-test
  PRIVATE
  WeftRVVPlugin
  MLIRSupport
)

add_executable(weft-rvv-formula-decision-test
  Plugin/RVVFormulaDecisionTest.cpp
)

target_link_libraries(weft-rvv-formula-decision-test
  PRIVATE
  WeftRVVPlugin
  MLIRSupport
)

add_executable(weft-rvv-q4-0-q8-0-shape-selection-test
  Plugin/RVVQ40Q80ShapeSelectionTest.cpp
)

target_link_libraries(weft-rvv-q4-0-q8-0-shape-selection-test
  PRIVATE
  WeftRVVPlugin
  MLIRSupport
)

add_executable(weft-rvv-capability-profile-divergence-test
  Plugin/RVVCapabilityProfileDivergenceTest.cpp
)

target_link_libraries(weft-rvv-capability-profile-divergence-test
  PRIVATE
  WeftInitAll
  WeftRVVPlugin
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_executable(weft-variant-materialization-test
  Transforms/VariantMaterialization/VariantMaterializationTest.cpp
)

target_link_libraries(weft-variant-materialization-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftSupport
  WeftTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(weft-variant-dispatch-synthesis-test
  Transforms/DispatchSynthesis/VariantDispatchSynthesisTest.cpp
)

target_link_libraries(weft-variant-dispatch-synthesis-test
  PRIVATE
  WeftInitAll
  WeftSupport
  WeftTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(weft-emission-readiness-test
  Transforms/EmissionReadiness/EmissionReadinessTest.cpp
)

target_link_libraries(weft-emission-readiness-test
  PRIVATE
  WeftInitAll
  WeftPlugin
  WeftRVVPlugin
  WeftSupport
  WeftTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(weft-variant-selection-test
  Transforms/VariantSelection/VariantSelectionTest.cpp
)

target_link_libraries(weft-variant-selection-test
  PRIVATE
  WeftInitAll
  WeftBuiltinPlugins
  WeftPlugin
  WeftRVVPlugin
  WeftScalarPlugin
  WeftSupport
  WeftTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
)

add_executable(weft-capability-model-test
  Support/CapabilityModelTest.cpp
)

target_link_libraries(weft-capability-model-test
  PRIVATE
  WeftInitAll
  WeftSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_executable(weft-load-time-resolution-test
  Support/LoadTimeCapabilityResolutionTest.cpp
)

target_link_libraries(weft-load-time-resolution-test
  PRIVATE
  WeftInitAll
  WeftSupport
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_lit_testsuite(check-weft "Running Weft-RV lit tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
  weft-opt
  weft-translate
  weft-capability-model-test
  weft-load-time-resolution-test
  weft-plugin-emission-plan-test
  weft-formula-catalog-test
  weft-plugin-variant-cost-test
  weft-plugin-registry-test
  weft-plugin-variant-legality-test
  weft-plugin-variant-proposal-test
  weft-emission-readiness-test
  weft-offload-extension-plugin-test
  weft-rvv-dialect-test
  weft-rvv-formula-decision-test
  weft-rvv-low-precision-lmul-selection-test
  weft-rvv-q4-0-q8-0-shape-selection-test
  weft-rvv-capability-profile-divergence-test
  weft-scalar-extension-plugin-test
  weft-template-extension-plugin-test
  weft-toy-extension-plugin-test
  weft-variant-dispatch-synthesis-test
  weft-variant-materialization-test
  weft-variant-selection-test
  ARGS
  -sv
)

set_target_properties(check-weft PROPERTIES FOLDER "Tests")
