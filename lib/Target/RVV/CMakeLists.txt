add_mlir_library(WeftRVVTarget
  RVVTargetProfileBinding.cpp
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
