add_subdirectory(Construction)
add_subdirectory(EmitC)

add_mlir_library(WeftTensorExtLitePlugin
  TensorExtLiteExtensionPlugin.cpp
  TensorExtLiteSourceFrontDoor.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftPlugin
  WeftSupport
  WeftTensorExtLiteFamilyContract
  WeftTensorExtLiteDialect
  WeftTensorExtLiteTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
