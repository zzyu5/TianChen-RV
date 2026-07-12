add_mlir_library(WeftPluginConstructionProtocol
  ConstructionProtocol.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  MLIRIR
  MLIRSupport
)
