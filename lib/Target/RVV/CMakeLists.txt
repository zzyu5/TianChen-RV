add_mlir_library(WeftRVVTarget
  RVVTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftRVVConstructionProtocol
  WeftRVVDialect
  WeftSupport
  WeftTransforms
  WeftTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
