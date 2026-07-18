add_mlir_library(WeftSupport
  CapabilityModel.cpp
  DeclaredInstanceHash.cpp
  LoadTimeCapabilityResolution.cpp
  GridDecodePlan.cpp
  RuntimeABIContract.cpp
  RuntimeABIParam.cpp
  RuntimeABIMemWindow.cpp

  LINK_LIBS PUBLIC
  WeftExecDialect
  MLIRIR
  MLIRSupport
)
