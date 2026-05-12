add_mlir_library(TianChenRVConversionEmitC
  TCRVEmitCLowerableMaterializer.cpp
  TCRVEmitCLowerableOpInterface.cpp
  TCRVEmitCLowerableInterface.cpp
  TCRVLowerToEmitC.cpp

  DEPENDS
  MLIRTCRVEmitCLowerableOpInterfaceIncGen

  LINK_LIBS PUBLIC
  TianChenRVSupport
  MLIREmitCDialect
  MLIRIR
  MLIRSupport
  MLIRTargetCpp
)
