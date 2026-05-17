add_mlir_library(TianChenRVToyTarget
  ToyTargetSupportBundle.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVExecDialect
  TianChenRVPlugin
  TianChenRVSupport
  TianChenRVTarget
  TianChenRVToyConstructionProtocol
  TianChenRVToyEmitCRouteProvider
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
)
