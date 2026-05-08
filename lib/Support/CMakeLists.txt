add_mlir_library(TianChenRVSupport
  CapabilityModel.cpp
  RuntimeABIMemWindow.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
