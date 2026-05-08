add_mlir_library(TianChenRVSupport
  CapabilityModel.cpp
  RuntimeABIParam.cpp
  RuntimeABIMemWindow.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
