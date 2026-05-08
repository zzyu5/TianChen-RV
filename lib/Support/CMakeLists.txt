add_mlir_library(TianChenRVSupport
  CapabilityModel.cpp
  RuntimeABICallablePlan.cpp
  RuntimeABIContract.cpp
  RuntimeABIParam.cpp
  RuntimeABIMemWindow.cpp

  LINK_LIBS PUBLIC
  TianChenRVExecDialect
  MLIRIR
  MLIRSupport
)
