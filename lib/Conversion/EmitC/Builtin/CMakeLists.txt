add_mlir_library(TianChenRVConversionEmitCBuiltin
  BuiltinBackendEmitters.cpp

  LINK_LIBS PUBLIC
  TianChenRVConversionEmitC
  TianChenRVConversionRVV
  TianChenRVIMEBackendEmitter
  TianChenRVToyBackendEmitter
  TianChenRVTemplateBackendEmitter
  TianChenRVTensorExtLiteBackendEmitter
  MLIRIR
  MLIRSupport
)
