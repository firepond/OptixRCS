set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/reduceKernels.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

OPTIX_add_sample_executable( CudaTest target_name

  main.cpp
  
  reduceKernels.cu
  reduceKernels.h

  # OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  )
