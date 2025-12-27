add_llvm_component_library(LLVMDPUTransforms
  DPULegalize.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  )
