set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/reduce_h.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

OPTIX_add_sample_executable( RcsSpeedBranch target_name
  tiny_obj_loader.h

  rcs_params.h
  rcs_predicitor.h
  main.cpp
  
  triangles_rcs.cu
  complex_vector.cu

  # reduce.cu
  # reduce_h.cu

  OPTIONS -rdc true
  )

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/reduce_h.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
  CUDA::cudart
  CUDA::cuda_driver
  )
