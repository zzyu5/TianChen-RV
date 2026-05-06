add_mlir_library(TianChenRVSupport
  CapabilityModel.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
