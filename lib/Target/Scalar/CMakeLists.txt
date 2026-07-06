add_mlir_library(TianChenRVScalarTarget
  ScalarTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVConversionEmitCBuiltin
  TianChenRVScalarEmitCRouteProvider
  TianChenRVTarget
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRIR
  MLIRSupport
)
