add_mlir_library(TianChenRVSupport
  CapabilityModel.cpp
  DeclaredInstanceHash.cpp
  RuntimeABIContract.cpp
  RuntimeABIParam.cpp
  RuntimeABIMemWindow.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
