add_mlir_dialect_library(MLIRDPUTransforms
  DPUToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/DPU

  DEPENDS
  MLIRDPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRDPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
