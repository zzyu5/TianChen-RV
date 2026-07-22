add_mlir_library(WeftRVVConstructionProtocol
  RVVConstructionProtocol.cpp
  RVVFormulaConstruction.cpp
  RVVContractionRouteIdentity.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPluginConstructionProtocol
  WeftRVVDialect
  WeftSupport
  MLIRIR
  MLIRSupport
)
