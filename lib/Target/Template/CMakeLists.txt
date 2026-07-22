add_mlir_library(WeftTemplateTarget
  TemplateTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftPlugin
  WeftSupport
  WeftTarget
  WeftTemplateConstructionProtocol
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
