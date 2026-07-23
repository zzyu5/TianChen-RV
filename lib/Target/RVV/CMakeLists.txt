add_mlir_library(WeftRVVTarget
  RVVTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftRVVConstruction
  WeftRVVDialect
  WeftSupport
  WeftTransforms
  WeftTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
