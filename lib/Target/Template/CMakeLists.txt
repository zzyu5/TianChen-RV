add_mlir_library(TianChenRVTemplateTarget
  TemplateTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVTemplateConstructionProtocol
  TianChenRVTemplateEmitCRouteProvider
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
