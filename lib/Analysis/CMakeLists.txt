add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  Membar.cpp
  PresburgerIndexAnalysis.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  MLIRPresburger
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
)
