add_mlir_library(WeftScalarTarget
  ScalarTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftConversionEmitCBuiltin
  WeftScalarEmitCRouteProvider
  WeftScalarFormulaConstruction
  WeftTarget
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRIR
  MLIRSupport
)
