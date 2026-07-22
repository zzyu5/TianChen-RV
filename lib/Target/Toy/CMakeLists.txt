add_mlir_library(WeftToyTarget
  ToyTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  WeftToyConstructionProtocol
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
