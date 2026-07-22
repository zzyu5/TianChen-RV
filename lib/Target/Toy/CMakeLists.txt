add_mlir_library(WeftToyTarget
  ToyTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  WeftToyFamilyContract
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
